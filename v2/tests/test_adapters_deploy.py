# coding=utf-8
"""`adapters/deploy.py` 的装配纪律测试。

核心断言不是"代码能跑"，而是**装配事实可判**：
* LLM 配置齐/缺/半套，各自落到明确的产物与错误；
* TTS 默认/显式关闭，各落到明确的产物与说明；
* 入口把环境变量一路传到装配产物上。
"""

from __future__ import annotations

import asyncio

import pytest

from ruipin.adapters.deploy import (
    ENV_LLM_API_KEY,
    ENV_LLM_BASE_URL,
    ENV_LLM_MODEL,
    ENV_TTS_BASE_URL,
    ENV_TTS_MODEL,
    ENV_TTS_VOICE,
    DeployPlan,
    build_evaluator_from_env,
    build_real_plan,
    build_tts_from_env,
)
from ruipin.adapters.llm_router import EvaluatorRouter
from ruipin.ports import EvalRequest
from ruipin.scoring.rubric import RuleEvaluator

LLM_FULL = {
    ENV_LLM_BASE_URL: "https://api.example.com",
    ENV_LLM_API_KEY: "sk-test",
    ENV_LLM_MODEL: "test-model",
}


# ---------- 评估链 ----------


def test_llm_unconfigured_falls_to_rule_with_note() -> None:
    asm = build_evaluator_from_env({})
    assert isinstance(asm.evaluator, RuleEvaluator)
    assert asm.uses_cloud_llm is False
    assert "rubric" in asm.note


def test_llm_partial_config_raises_with_missing_names() -> None:
    """半套配置必须显式报错：半套 = '到底用没用真 LLM'不可判断。"""
    with pytest.raises(ValueError) as ei:
        build_evaluator_from_env({ENV_LLM_BASE_URL: "https://api.example.com"})
    assert "RUIPIN_LLM_API_KEY" in str(ei.value)
    assert "RUIPIN_LLM_MODEL" in str(ei.value)

    with pytest.raises(ValueError):
        build_evaluator_from_env({ENV_LLM_API_KEY: "sk-x"})


def test_llm_full_config_builds_router_with_fallback() -> None:
    asm = build_evaluator_from_env(LLM_FULL)
    router = asm.evaluator
    assert isinstance(router, EvaluatorRouter)
    # 顺序即优先级：云 LLM 在前，规则评分兜底在后
    assert router.providers == ("openai-compat", "rubric")
    assert asm.uses_cloud_llm is True


def test_llm_blank_values_count_as_unconfigured() -> None:
    """空白串（CI 变量没赋值）与未设置同义，不得当成真配置。"""
    asm = build_evaluator_from_env(
        {ENV_LLM_BASE_URL: "  ", ENV_LLM_API_KEY: "", ENV_LLM_MODEL: "\t"}
    )
    assert isinstance(asm.evaluator, RuleEvaluator)
    assert asm.uses_cloud_llm is False


def test_rule_evaluator_under_router_gets_degraded_annotation() -> None:
    """路由器把规则评分标注为 degraded——LLM 失败时的降级语义（不经装配函数：
    空 key 会在装配期被显式拒绝，这正是 test_llm_partial_config 的纪律）。"""
    from ruipin.domain.errors import Unavailable

    class _AlwaysDown:
        name = "down-cloud"

        async def evaluate(self, req: EvalRequest) -> Any:
            raise Unavailable(self.name, "模拟云供应商不可用")

    from ruipin.adapters.llm_router import EvaluatorRouter as Router

    router = Router([_AlwaysDown(), RuleEvaluator()])
    req = EvalRequest(
        question="介绍你自己",
        answer="我做过三个后端项目，负责数据库设计与接口联调，解决了慢查询问题。",
    )
    res = asyncio.run(router.evaluate(req))
    assert res.provider == "rubric"
    assert res.degraded is True
    assert res.degrade_reason is not None and "down-cloud" in res.degrade_reason


# ---------- TTS 链 ----------


def test_tts_default_url_builds_http_tts() -> None:
    asm = build_tts_from_env({})
    assert asm.tts is not None
    assert asm.tts.name == "qwen3-tts"
    assert asm.voice == "serena"
    # 客户端身份可判：base_url/model 来自装配事实，不是隐式默认漂移
    client = asm.tts._client
    assert client.base_url == "http://127.0.0.1:8091"
    assert client.model == "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    assert client.response_format == "wav"


def test_tts_explicit_empty_url_disables() -> None:
    """显式空串 = 关闭语音；产物 tts=None 且说明写明，不是静默失败。"""
    asm = build_tts_from_env({ENV_TTS_BASE_URL: ""})
    assert asm.tts is None
    assert "纯文本" in asm.note


def test_tts_env_overrides_reach_client() -> None:
    asm = build_tts_from_env(
        {
            ENV_TTS_BASE_URL: "http://127.0.0.1:9999",
            ENV_TTS_MODEL: "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
            ENV_TTS_VOICE: "vivian",
        }
    )
    client = asm.tts._client
    assert client.base_url == "http://127.0.0.1:9999"
    assert client.model == "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
    assert asm.voice == "vivian"


# ---------- 入口装配 ----------


def test_real_plan_wires_env_into_plan() -> None:
    plan = build_real_plan(["--port", "8799", "--token", "tok"], env=LLM_FULL)
    assert plan is not None
    assert isinstance(plan, DeployPlan)
    assert plan.port == 8799
    assert plan.token == "tok"
    assert plan.notes and any("LLM 已启用" in n for n in plan.notes)
    assert any("TTS 已启用" in n for n in plan.notes)
    # server 已按真实组件装配（bridge 工厂注入了 evaluator/tts）
    assert plan.server is not None


def test_real_plan_default_rules_when_no_llm() -> None:
    plan = build_real_plan(["--port", "8799", "--token", "tok"], env={})
    assert plan is not None
    assert any("规则" in n for n in plan.notes)


def test_real_plan_empty_questions_returns_none() -> None:
    assert build_real_plan(["--questions", " "], env={}) is None


def test_build_evaluator_from_llm_config_full_trio_uses_cloud_router():
    from ruipin.adapters.deploy import build_evaluator_from_llm_config

    asm = build_evaluator_from_llm_config("https://api.example.com", "sk-test-key-000000", "deepseek-chat")
    assert asm.uses_cloud_llm is True
    assert "deepseek-chat" in asm.note


def test_build_evaluator_from_llm_config_empty_means_rule_only():
    from ruipin.scoring.rubric import RuleEvaluator
    from ruipin.adapters.deploy import build_evaluator_from_llm_config

    asm = build_evaluator_from_llm_config("", "", "")
    assert asm.uses_cloud_llm is False
    assert isinstance(asm.evaluator, RuleEvaluator)


def test_build_evaluator_from_llm_config_strips_whitespace():
    from ruipin.adapters.deploy import build_evaluator_from_llm_config

    asm = build_evaluator_from_llm_config("  https://api.example.com  ", "  sk-x  ", " m ")
    assert asm.uses_cloud_llm is True
    assert "https://api.example.com" in asm.note
