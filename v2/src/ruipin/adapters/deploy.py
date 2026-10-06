# coding=utf-8
"""部署装配层：从环境变量构造真实 adapter，注入 `build_dev_server`。

定位
----
`ws_server.build_dev_server` 的 docstring 说过：它**不该被改造成生产装配**，
真实实现应由上层部署编排注入。本模块就是那层"部署编排"的第一块落地——
它只做一件事：**把环境变量翻译成 adapter 实例**，然后复用 `build_dev_server`
的既有装配（协议、降级、报告纪律全部原样生效）。

环境变量（全部可选；按"缺什么降什么"的原则逐项生效）
----------------------------------------------------
LLM 评分（三项**齐全**才启用云评分；任一缺失则显式降级为规则评分）::

    RUIPIN_LLM_BASE_URL   # 如 https://api.deepseek.com（不含 /chat/completions）
    RUIPIN_LLM_API_KEY
    RUIPIN_LLM_MODEL      # 如 deepseek-chat

TTS 语音合成（`RUIPIN_TTS_BASE_URL` 设为空串 = 显式关闭语音，纯文本面试）::

    RUIPIN_TTS_BASE_URL   # 默认 http://127.0.0.1:8091（本机 vLLM-Omni 容器）
    RUIPIN_TTS_MODEL      # 默认 Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
    RUIPIN_TTS_VOICE      # 默认 serena（bridge 默认传 "default"，服务端同样接受）

降级语义（复用既有机制，本模块不发明新的）：
* LLM 请求失败 → `EvaluatorRouter` 自动切 `RuleEvaluator`，结果标
  `degraded=True`，`degradation.changed` 下行显式告知；
* TTS 合成失败 → bridge 下发 `degradation.changed`（L4 `tts_unavailable`），
  题目文本照常下发；
* 本模块**绝不**为了"跑起来"而注入任何会编造分数/静音的替身。

启动::

    python -m ruipin.adapters.deploy --port 8787 --token <token>
"""

from __future__ import annotations

import asyncio
import logging
import os
import secrets
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence

from .ws_server import DEFAULT_HOST, serve_forever

# 环境变量名（集中常量：多处引用同一名，避免"同名写两遍"的漂移）
ENV_LLM_BASE_URL = "RUIPIN_LLM_BASE_URL"
ENV_LLM_API_KEY = "RUIPIN_LLM_API_KEY"
ENV_LLM_MODEL = "RUIPIN_LLM_MODEL"
ENV_TTS_BASE_URL = "RUIPIN_TTS_BASE_URL"
ENV_TTS_MODEL = "RUIPIN_TTS_MODEL"
ENV_TTS_VOICE = "RUIPIN_TTS_VOICE"

DEFAULT_TTS_BASE_URL = "http://127.0.0.1:8091"
DEFAULT_TTS_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
DEFAULT_TTS_VOICE = "serena"

LLM_PROVIDER_NAME = "openai-compat"

log = logging.getLogger("ruipin.deploy")


@dataclass(frozen=True)
class DeployPlan:
    """`main()` 的装配产物：已装好的服务端 + 监听参数 + 装配说明。

    与 `ws_server.DevPlan` 分开定义，是因为装配说明（`notes`）是本模块的
    装配事实，塞进对方的 frozen dataclass 等于改它的契约。
    """

    server: Any
    host: str
    port: int
    path: str
    token: str
    questions: tuple[str, ...]
    log_level: int = logging.INFO
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvalAssembly:
    """评估链装配产物。`note` 是给人看的装配说明（进启动横幅/日志）。"""

    evaluator: Any
    uses_cloud_llm: bool
    note: str


@dataclass(frozen=True)
class TtsAssembly:
    """TTS 链装配产物。`tts=None` 表示本部署**显式**不接语音（不是故障）。"""

    tts: Any
    voice: str
    note: str


def _env_get(env: Mapping[str, str], key: str) -> str:
    """从 env 取值并 strip；缺失返回空串。空白串与未设置同义（CI 常见陷阱）。"""
    return (env.get(key) or "").strip()


def build_evaluator_from_env(env: Mapping[str, str] | None = None) -> EvalAssembly:
    """按环境变量装配评估链：云 LLM（可降级规则）或纯规则评分。

    LLM 三项任缺其一 → 不建云供应商（而不是建一个注定失败的）：
    每轮先试一次再降级，除了多一次注定 401/拒绝的请求，没有任何收益。
    降级必须是**显式且廉价**的，而不是"每轮表演一次失败"。
    """
    from ..scoring.rubric import RuleEvaluator

    env = os.environ if env is None else env
    base_url = _env_get(env, ENV_LLM_BASE_URL)
    api_key = _env_get(env, ENV_LLM_API_KEY)
    model = _env_get(env, ENV_LLM_MODEL)

    configured = [bool(base_url), bool(api_key), bool(model)]
    if any(configured) and not all(configured):
        missing = [n for n, ok in zip(("BASE_URL", "API_KEY", "MODEL"), configured) if not ok]
        raise ValueError(
            "LLM 配置不完整：已设置部分 "
            f"RUIPIN_LLM_* 变量但缺少 {', '.join('RUIPIN_LLM_' + m for m in missing)}。"
            "要么三项配齐，要么全部不设（走规则评分）。"
            "半套配置会让'到底用没用真 LLM'变得不可判断。"
        )

    if not all(configured):
        return EvalAssembly(
            evaluator=RuleEvaluator(),
            uses_cloud_llm=False,
            note="LLM 未配置（无 RUIPIN_LLM_*）：评分为规则计算（provider=rubric，非 AI 评分）",
        )

    from .http import HttpxTransport
    from .llm_openai import OpenAICompatEvaluator
    from .llm_router import EvaluatorRouter

    cloud = OpenAICompatEvaluator(
        LLM_PROVIDER_NAME,
        model,
        base_url,
        api_key,
        HttpxTransport(),
    )
    router = EvaluatorRouter([cloud, RuleEvaluator()], name="eval-router")
    return EvalAssembly(
        evaluator=router,
        uses_cloud_llm=True,
        note=f"LLM 已启用（{LLM_PROVIDER_NAME}: {model} @ {base_url}），失败自动降级规则评分",
    )


def build_evaluator_from_llm_config(base_url: str, api_key: str, model: str) -> "EvalAssembly":
    """从设置页下发的显式三项配置装配评估链（运行时热更新用）。

    与 `build_evaluator_from_env` 同一门纪律：三项齐全才建云供应商，
    半套由调用方（HTTP 端点）显式拒绝，这里只收全空或全满。
    全空 → 纯规则评分（用户在设置页清空即视为关闭 AI 评分）。
    """
    from ..scoring.rubric import RuleEvaluator

    base_url = (base_url or "").strip()
    api_key = (api_key or "").strip()
    model = (model or "").strip()

    if not (base_url or api_key or model):
        return EvalAssembly(
            evaluator=RuleEvaluator(),
            uses_cloud_llm=False,
            note="AI 评分已关闭（设置页未配置）：评分为规则计算（provider=rubric，非 AI 评分）",
        )

    from .http import HttpxTransport
    from .llm_openai import OpenAICompatEvaluator
    from .llm_router import EvaluatorRouter

    cloud = OpenAICompatEvaluator(
        LLM_PROVIDER_NAME,
        model,
        base_url,
        api_key,
        HttpxTransport(),
    )
    router = EvaluatorRouter([cloud, RuleEvaluator()], name="eval-router")
    return EvalAssembly(
        evaluator=router,
        uses_cloud_llm=True,
        note=f"AI 评分已启用（{LLM_PROVIDER_NAME}: {model} @ {base_url}），失败自动降级规则评分",
    )


def build_tts_from_env(env: Mapping[str, str] | None = None) -> TtsAssembly:
    """按环境变量装配 TTS 链：真实服务 / 显式关闭。

    本模块**不做**装配期连通性探测：服务死了让运行时降级去显式报告
    （`degradation.changed` L4），装配期探测通过不代表请求时不死，
    反而多一种"探测成功但用了才发现挂了"的歧义状态。
    """
    env = os.environ if env is None else env
    base_url = _env_get(env, ENV_TTS_BASE_URL)

    # 只有"显式给了空串"才是关闭；变量不存在 = 用默认地址（本机容器）。
    # 用 `in` 判存在性：`env.get(key, SENTINEL)` 区分"没设"与"设成空"。
    if ENV_TTS_BASE_URL in env and not base_url:
        return TtsAssembly(tts=None, voice=DEFAULT_TTS_VOICE, note="TTS 已显式关闭：纯文本面试")

    from .http import HttpxStreamTransport
    from .tts import HttpTTS
    from .tts_qwen3 import Qwen3TTSClient

    model = _env_get(env, ENV_TTS_MODEL) or DEFAULT_TTS_MODEL
    voice = _env_get(env, ENV_TTS_VOICE) or DEFAULT_TTS_VOICE
    client = Qwen3TTSClient(
        HttpxStreamTransport(),
        base_url=base_url or DEFAULT_TTS_BASE_URL,
        model=model,
        voice=voice,
        response_format="wav",
        stream_mode=False,  # 整段 WAV：契约最简（见 tts_qwen3.stream_mode 注释）
    )
    tts = HttpTTS("qwen3-tts", client)
    return TtsAssembly(
        tts=tts,
        voice=voice,
        note=f"TTS 已启用（{model} @ {base_url or DEFAULT_TTS_BASE_URL}，voice={voice}）",
    )


def build_real_plan(
    argv: Optional[Sequence[str]] = None,
    *,
    env: Mapping[str, str] | None = None,
    logger: Optional[logging.Logger] = None,
) -> Optional[DeployPlan]:
    """解析命令行并装配"真实组件"服务。参数不合法返回 None（`main` 据此返回 2）。

    与 `ws_server.build_dev_plan` 的唯一区别：evaluator/tts 来自环境变量装配，
    其余（协议、鉴权、报告纪律）走同一个 `build_dev_server`。
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="ruipin-deploy",
        description="锐聘 AI 面试服务端（真实组件装配：评估/TTS 来自环境变量）",
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"监听地址（默认 {DEFAULT_HOST}）")
    parser.add_argument("--port", type=int, default=8787, help="监听端口（默认 8787）")
    parser.add_argument("--path", default="/ws", help="WS 路径（默认 /ws）")
    parser.add_argument("--token", default=None, help="一次性令牌；不传则随机生成并打印")
    parser.add_argument(
        "--questions",
        default=None,
        help="题目列表，逗号分隔；不传用内置示例题",
    )
    parser.add_argument("--log-level", default="INFO", help="日志级别（默认 INFO）")
    args = parser.parse_args(argv)

    lg = logger or log
    token = args.token or secrets.token_urlsafe(18)
    if args.questions is None:
        from .ws_server import DEFAULT_DEV_QUESTIONS

        questions = DEFAULT_DEV_QUESTIONS
    else:
        questions = tuple(q.strip() for q in str(args.questions).split(",") if q.strip())
    if not questions:
        lg.error("题目列表为空，至少要给一道题")
        return None

    eval_asm = build_evaluator_from_env(env)
    tts_asm = build_tts_from_env(env)

    from .ws_server import build_dev_server

    server = build_dev_server(
        questions=questions,
        token=token,
        evaluator=eval_asm.evaluator,
        tts=tts_asm.tts,
    )
    return DeployPlan(
        server=server,
        host=args.host,
        port=args.port,
        path=args.path,
        token=token,
        questions=questions,
        log_level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        notes=(eval_asm.note, tts_asm.note),
    )


def main(
    argv: Optional[Sequence[str]] = None,
    *,
    runner: Optional[Callable[[DeployPlan], int]] = None,
) -> int:
    """命令行入口：起一个"真实组件"服务。

    用法::

        RUIPIN_LLM_BASE_URL=... RUIPIN_LLM_API_KEY=... RUIPIN_LLM_MODEL=... \\
        python -m ruipin.adapters.deploy --port 8787 --token my-token
    """
    plan = build_real_plan(argv)
    if plan is None:
        return 2

    logging.basicConfig(
        level=plan.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log.warning("=" * 72)
    for note in plan.notes:
        log.warning("%s", note)
    log.warning("存储仍是【内存】的、令牌仍是【静态】的（这两项不影响链路验证）。")
    log.warning("=" * 72)
    log.info("令牌（放 Authorization: Bearer 或 ?token=）: %s", plan.token)
    log.info("题目 %d 道", len(plan.questions))

    async def _serve() -> None:
        from .ws_server import (
            make_llm_config_endpoint,
            serve_forever as _serve_forever,
        )

        # 设置页热更新 AI 评分：HTTP 端点 + 评估器可变槽（dev 装配自带）。
        slot = getattr(plan.server, "evaluator_slot", None)
        process_request = None
        if slot is not None:

            def _apply(cfg: Mapping[str, str]) -> tuple[bool, str]:
                base = (cfg.get("base_url") or "").strip()
                key = (cfg.get("api_key") or "").strip()
                model = (cfg.get("model") or "").strip()
                filled = [bool(base), bool(key), bool(model)]
                if any(filled) and not all(filled):
                    missing = [n for n, ok in zip(("BASE_URL", "API_KEY", "MODEL"), filled) if not ok]
                    return False, f"配置不完整：还差 {', '.join(m.lower() for m in missing)}。三项要么都填，要么都留空"
                asm = build_evaluator_from_llm_config(base, key, model)
                slot[0] = asm.evaluator
                log.info("%s", asm.note)
                return True, asm.note

            process_request = make_llm_config_endpoint(slot, verify=plan.server._verify, apply_config=_apply, logger=log)

        await _serve_forever(
            plan.server, host=plan.host, port=plan.port, path=plan.path,
            process_request=process_request,
        )

    run = runner or (lambda p: (asyncio.run(_serve()), 0)[1])
    try:
        return run(plan)
    except KeyboardInterrupt:  # pragma: no cover - 交互式退出
        log.info("已停止")
        return 0
    except OSError as exc:
        log.error(
            "无法监听 %s:%d（%s）。换一个端口重试：--port <空闲端口>",
            plan.host,
            plan.port,
            exc,
        )
        return 3


if __name__ == "__main__":  # pragma: no cover - 入口
    import sys

    sys.exit(main())
