"""OpenAI 兼容评分适配器（`adapters/llm_openai.py`）的测试。

守的回归（本项目第一纪律）
--------------------------
**绝不产出合成分数。** 解析失败 / 缺维度 / 越界 / confidence 越界，一律是"这一次
调用失败"：可重试的重试一次，重试仍失败就抛 `Unavailable`。任何路径都不得吐出
"看起来正常的默认分"——遗留系统的头号缺陷就是崩溃即返回全 60 分。

环境没有 httpx/pytest-asyncio：协程一律 `asyncio.run`，HTTP 一律走 `FakeTransport`，
时钟/退避全部注入，测试零 sleep、零随机、零真实时间。
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Optional

import pytest

from ruipin.adapters.fakes import DeterministicClock
from ruipin.adapters.http import DEFAULT_TIMEOUT_S, FakeTransport, HttpResponse
from ruipin.adapters.llm_openai import (
    DIM_DESCRIPTIONS,
    EVAL_JSON_SCHEMA,
    OPS_LATENCY_KEY,
    SYSTEM_PROMPT,
    OpenAICompatEvaluator,
    api_key_from_env,
    build_eval_schema,
    build_user_prompt,
)
from ruipin.domain.errors import Unavailable
from ruipin.ports import DIMENSIONS, EvalRequest, EvalResult, Evaluator

run = asyncio.run

#: 定价：prompt 2 美元/百万 token、completion 8 美元/百万 token。
PRICING = {
    "deepseek": {"prompt_per_1k": 0.002, "completion_per_1k": 0.008},
    "default": {"prompt_per_1k": 0.001, "completion_per_1k": 0.002},
}

_OK_DIMS: dict[str, float] = {
    "technical": 80.0,
    "communication": 70.0,
    "completeness": 65.0,
    "problem_solving": 75.0,
    "teamwork": 68.0,
    "leadership": 60.0,
}
_FEEDBACK = "技术细节扎实，建议补充重构前后的量化对比。"


class RecordingMeter:
    """内存版 `Meter`：把三条埋点都记下来供断言。"""

    def __init__(self) -> None:
        self.usages: list[Any] = []
        self.observations: list[tuple[str, float]] = []
        self.counters: dict[str, int] = {}

    def record_usage(self, usage) -> None:
        self.usages.append(usage)

    def observe_ms(self, name: str, ms: float) -> None:
        self.observations.append((name, ms))

    def incr(self, name: str, value: int = 1) -> None:
        self.counters[name] = self.counters.get(name, 0) + value


class RecordingSleeper:
    """退避等待的记录器：只记不睡，退避次数与时长因此可断言。"""

    def __init__(self) -> None:
        self.calls: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)


@dataclass
class _Ctx:
    """一次测试的装配：被测适配器 + 全部可断言部件。"""

    ev: OpenAICompatEvaluator
    transport: FakeTransport
    sleeper: RecordingSleeper
    meter: RecordingMeter
    clock: DeterministicClock


def _scores(**overrides: Any) -> dict[str, Any]:
    obj: dict[str, Any] = dict(_OK_DIMS)
    obj.update({"score": 72.0, "feedback": _FEEDBACK, "confidence": 0.88})
    obj.update(overrides)
    return obj


def _openai_resp(
    scores: Optional[dict[str, Any]] = None,
    *,
    status: int = 200,
    finish: str = "stop",
    raw: Optional[str] = None,
    usage: Optional[tuple[int, int]] = (1200, 300),
) -> HttpResponse:
    """拼一个 OpenAI chat completions 风格的响应。"""
    content = raw if raw is not None else json.dumps(_scores() if scores is None else scores, ensure_ascii=False)
    body: dict[str, Any] = {
        "id": "chatcmpl-1",
        "choices": [
            {
                "index": 0,
                "finish_reason": finish,
                "message": {"role": "assistant", "content": content},
            }
        ],
    }
    if usage is not None:
        body["usage"] = {
            "prompt_tokens": usage[0],
            "completion_tokens": usage[1],
        }
    return HttpResponse(
        status=status,
        body=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"content-type": "application/json"},
    )


def _ctx(
    script: list[Any],
    *,
    api_key: Optional[str] = "sk-test-key",
    pricing: Optional[dict] = PRICING,
    max_retries: int = 1,
    advance_s: float = 0.4,
    **kw: Any,
) -> _Ctx:
    clock = DeterministicClock()
    transport = FakeTransport(script, on_call=lambda i: clock.advance(advance_s))
    sleeper = RecordingSleeper()
    meter = RecordingMeter()
    ev = OpenAICompatEvaluator(
        "deepseek",
        "deepseek-chat",
        kw.pop("base_url", "https://api.deepseek.com/v1"),
        api_key,
        transport,
        meter=meter,
        pricing=pricing,
        max_retries=max_retries,
        clock=clock.monotonic,
        sleeper=sleeper,
        **kw,
    )
    return _Ctx(ev, transport, sleeper, meter, clock)


_REQ = EvalRequest(
    question="请介绍一个你主导过的项目，以及你在其中解决的关键问题。",
    answer="我主导了推荐系统的召回重构，用向量检索替换倒排，QPS 提升 3 倍。",
    question_type="technical",
    keywords=("推荐系统", "向量检索", "召回"),
)


# ---------- 场景 1：正常路径 ----------


def test_正常响应产出六维结果且计量如实():
    """防回归：成功路径必须六维齐全、成本与 token 与响应一致、延迟被观测。"""
    c = _ctx([_openai_resp()])

    res = run(c.ev.evaluate(_REQ))

    assert set(res.dims) == set(DIMENSIONS)
    assert res.dims == _OK_DIMS
    assert res.score == 72.0
    assert res.feedback == _FEEDBACK
    assert res.provider == "deepseek"
    assert res.confidence == 0.88
    assert res.latency_ms == 400
    assert res.cost_usd == pytest.approx(0.0048)
    assert res.degraded is False

    assert len(c.meter.usages) == 1
    usage = c.meter.usages[0]
    assert usage.provider == "deepseek"
    assert usage.operation == OPS_LATENCY_KEY
    assert (usage.prompt_tokens, usage.completion_tokens) == (1200, 300)
    assert usage.total_tokens == 1500
    assert usage.cost_usd == pytest.approx(0.0048)
    assert usage.cached is False
    assert c.meter.observations == [(OPS_LATENCY_KEY, 400.0)]
    assert c.transport.n_calls == 1


def test_实现Evaluator协议():
    assert isinstance(_ctx([]).ev, Evaluator)


# ---------- 场景 2：缺维度 ----------


def test_缺维度视为失败重试后仍缺则抛Unavailable():
    """防回归：缺维度绝不能补默认值/补均值——那等于凭空造分。"""
    broken = _scores()
    del broken["leadership"]
    c = _ctx([_openai_resp(broken), _openai_resp(broken)])

    with pytest.raises(Unavailable) as exc:
        run(c.ev.evaluate(_REQ))

    assert "缺少维度 leadership" in exc.value.reason
    assert "2 次尝试均失败" in exc.value.reason
    assert exc.value.provider == "deepseek"
    # 从未产出过 EvalResult：成功才会上报 usage，这里一条都没有
    assert c.meter.usages == []
    assert c.meter.observations == []
    assert c.transport.n_calls == 2


# ---------- 场景 3：越界 ----------


def test_维度120越界抛Unavailable不得截断成100后返回():
    """防回归：把 120 夹成 100 会让"模型跑飞"变成一份看起来正常的满分。"""
    c = _ctx([_openai_resp(_scores(technical=120.0)), _openai_resp(_scores(technical=120.0))])

    with pytest.raises(Unavailable) as exc:
        run(c.ev.evaluate(_REQ))

    assert "维度 technical 越界 [0.0,100.0]: 120.0" in exc.value.reason
    assert c.meter.usages == []


def test_confidence越界同样视为失败不得夹到1():
    c = _ctx([_openai_resp(_scores(confidence=1.5))], max_retries=0)

    with pytest.raises(Unavailable) as exc:
        run(c.ev.evaluate(_REQ))

    assert "confidence 越界 [0.0,1.0]: 1.5" in exc.value.reason


def test_score缺失不得用维度均值兜底():
    """防回归：适配器算均值会让"模型漏给分"与"真实综合分"不可区分。"""
    broken = _scores()
    del broken["score"]
    c = _ctx([_openai_resp(broken)], max_retries=0)

    with pytest.raises(Unavailable) as exc:
        run(c.ev.evaluate(_REQ))

    assert "缺少字段 score" in exc.value.reason


def test_布尔值不得被当成数值分数():
    """防回归：`true` 是 int 子类，不显式排除就会被静默当成 1 分。"""
    c = _ctx([_openai_resp(_scores(score=True))], max_retries=0)

    with pytest.raises(Unavailable) as exc:
        run(c.ev.evaluate(_REQ))

    assert "score 非数值: True" in exc.value.reason


# ---------- 场景 4：429 重试 ----------


def test_429退避重试一次后成功():
    """防回归：限流必须退避重试，且退避次数/时长可断言（不能靠真 sleep）。"""
    c = _ctx([HttpResponse(429, b'{"error":"rate limited"}'), _openai_resp()])

    res = run(c.ev.evaluate(_REQ))

    assert res.dims == _OK_DIMS
    assert c.transport.n_calls == 2
    assert c.sleeper.calls == [0.5]  # backoff_base_s * factor**0
    assert c.meter.usages[-1].prompt_tokens == 1200


def test_429重试耗尽后抛Unavailable并带上状态码():
    busy = HttpResponse(429, b'{"error":"rate limited"}')
    c = _ctx([busy, busy])

    with pytest.raises(Unavailable) as exc:
        run(c.ev.evaluate(_REQ))

    assert "HTTP 429" in exc.value.reason
    assert c.transport.n_calls == 2
    assert c.sleeper.calls == [0.5]


def test_503可重试且退避按指数增长():
    """防回归：5xx 与 429 同属可重试；第二次退避时长要是第一次的 factor 倍。"""
    err = HttpResponse(503, b"unavailable")
    c = _ctx([err, err, err], max_retries=2, backoff_base_s=0.25, backoff_factor=3.0)

    with pytest.raises(Unavailable) as exc:
        run(c.ev.evaluate(_REQ))

    assert "HTTP 503" in exc.value.reason
    assert c.transport.n_calls == 3
    assert c.sleeper.calls == [0.25, 0.75]


def test_max_retries为0时不重试也不退避():
    c = _ctx([HttpResponse(429, b"busy")], max_retries=0)

    with pytest.raises(Unavailable):
        run(c.ev.evaluate(_REQ))

    assert c.transport.n_calls == 1
    assert c.sleeper.calls == []


# ---------- 场景 5：401 不重试 ----------


def test_401不可重试一次即失败():
    """防回归：认证错误重试只会浪费时间并放大限流风险。"""
    c = _ctx([HttpResponse(401, b'{"error":"invalid api key"}')])

    with pytest.raises(Unavailable) as exc:
        run(c.ev.evaluate(_REQ))

    assert "HTTP 401" in exc.value.reason
    assert "不可重试" in exc.value.reason
    assert "invalid api key" in exc.value.reason
    assert c.transport.n_calls == 1
    assert c.sleeper.calls == []


@pytest.mark.parametrize("status", [400, 403, 404])
def test_其它4xx同样不重试(status):
    c = _ctx([HttpResponse(status, b"bad request")])

    with pytest.raises(Unavailable) as exc:
        run(c.ev.evaluate(_REQ))

    assert f"HTTP {status}" in exc.value.reason
    assert c.transport.n_calls == 1


# ---------- 场景 6：网络异常 ----------


def test_网络异常可重试且重试成功后正常返回():
    """防回归：连接重置不是"供应商拒绝"，重试一次通常就能恢复。"""
    c = _ctx([ConnectionResetError("connection reset by peer"), _openai_resp()])

    res = run(c.ev.evaluate(_REQ))

    assert res.score == 72.0
    assert c.transport.n_calls == 2
    assert c.sleeper.calls == [0.5]


def test_httpx未安装这类Unavailable不被当作可重试():
    """防回归：结构性故障（依赖缺失）重试无意义，必须立刻上报。"""
    c = _ctx([Unavailable("httpx", "未安装 httpx")])

    with pytest.raises(Unavailable) as exc:
        run(c.ev.evaluate(_REQ))

    assert exc.value.provider == "httpx"
    assert c.transport.n_calls == 1
    assert c.sleeper.calls == []


# ---------- 场景 7：未配置密钥 ----------


def test_未配置api_key直接抛Unavailable且零网络调用():
    """防回归：密钥缺失是运维问题，绝不能"先发个请求试试"。"""
    c = _ctx([_openai_resp()], api_key=None)

    with pytest.raises(Unavailable) as exc:
        run(c.ev.evaluate(_REQ))

    assert "未配置 API key" in exc.value.reason
    assert exc.value.provider == "deepseek"
    assert c.transport.n_calls == 0
    assert c.meter.usages == []


def test_空白api_key等同未配置():
    c = _ctx([_openai_resp()], api_key="   ")

    with pytest.raises(Unavailable) as exc:
        run(c.ev.evaluate(_REQ))

    assert "未配置 API key" in exc.value.reason
    assert c.transport.n_calls == 0


def test_密钥从环境变量读取且没有默认值():
    """防回归：遗留系统把密钥明文写进 os.environ.get 的默认参数是 P0 事故。"""
    assert api_key_from_env("DEEPSEEK_API_KEY", {"DEEPSEEK_API_KEY": " sk-abc "}) == "sk-abc"
    assert api_key_from_env("DEEPSEEK_API_KEY", {}) is None
    assert api_key_from_env("DEEPSEEK_API_KEY", {"DEEPSEEK_API_KEY": "  "}) is None


def test_密钥不出现在请求体与提示词里():
    c = _ctx([_openai_resp()], api_key="sk-super-secret")

    run(c.ev.evaluate(_REQ))

    payload = json.dumps(c.transport.calls[0].payload, ensure_ascii=False)
    assert "sk-super-secret" not in payload
    assert c.transport.calls[0].headers["Authorization"] == "Bearer sk-super-secret"


# ---------- 场景 8：降级语义 ----------


def test_输出被截断时结果可用但标注degraded():
    """防回归：截断是"拿到了但折损"，必须标 degraded 并给原因；它不是失败。"""
    c = _ctx([_openai_resp(finish="length")])

    res = run(c.ev.evaluate(_REQ))

    assert isinstance(res, EvalResult)
    assert res.dims == _OK_DIMS
    assert res.degraded is True
    assert res.degrade_reason is not None
    assert "截断" in res.degrade_reason
    assert "finish_reason=length" in res.degrade_reason
    # 仍然是成功：计量照常上报
    assert len(c.meter.usages) == 1


def test_没给feedback时标注degraded而不是补一句套话():
    """防回归：编造评语等于编造证据。"""
    c = _ctx([_openai_resp(_scores(feedback=""))])

    res = run(c.ev.evaluate(_REQ))

    assert res.degraded is True
    assert "未给出 feedback" in (res.degrade_reason or "")
    assert res.feedback == ""


def test_content_filter是失败不是降级():
    """防回归：被过滤的输出没有内容，不能当成"降级但可用"。"""
    c = _ctx([_openai_resp(finish="content_filter")], max_retries=0)

    with pytest.raises(Unavailable) as exc:
        run(c.ev.evaluate(_REQ))

    assert "finish_reason=content_filter" in exc.value.reason
    assert c.meter.usages == []


# ---------- 场景 11：成本 ----------


def test_成本按定价与token手算一致():
    """防回归：成本模型算错会让"预算硬顶"形同虚设。"""
    c = _ctx([_openai_resp(usage=(1200, 300))])

    res = run(c.ev.evaluate(_REQ))

    expected = 1200 / 1000 * 0.002 + 300 / 1000 * 0.008  # 0.0024 + 0.0024
    assert expected == pytest.approx(0.0048)
    assert res.cost_usd == pytest.approx(expected)
    assert c.meter.usages[0].cost_usd == pytest.approx(expected)


def test_未配置定价时成本为0且不报错():
    c = _ctx([_openai_resp(usage=(1200, 300))], pricing=None)

    res = run(c.ev.evaluate(_REQ))

    assert res.cost_usd == 0.0
    assert c.meter.usages[0].cost_usd == 0.0
    assert c.meter.usages[0].prompt_tokens == 1200


def test_未命中的provider回退到default定价():
    c = _ctx([_openai_resp(usage=(1000, 1000))], pricing={"default": PRICING["default"]})

    res = run(c.ev.evaluate(_REQ))

    assert res.cost_usd == pytest.approx(1000 / 1000 * 0.001 + 1000 / 1000 * 0.002)


def test_响应没有usage时token记0而不是估算():
    """防回归：估不出来的就如实记 0，不允许为了"好看"编数字。"""
    c = _ctx([_openai_resp(usage=None)])

    res = run(c.ev.evaluate(_REQ))

    assert (c.meter.usages[0].prompt_tokens, c.meter.usages[0].completion_tokens) == (0, 0)
    assert res.cost_usd == 0.0


def test_usage字段类型异常时记0不抛异常():
    c = _ctx([_openai_resp(usage=(-5, "300"))])

    res = run(c.ev.evaluate(_REQ))

    usage = c.meter.usages[0]
    assert usage.prompt_tokens == 0  # 负数无意义，记 0
    assert usage.completion_tokens == 0  # 非数值，记 0
    assert res.latency_ms == 400


def test_没有meter时静默跳过计量():
    """防回归：可观测性不得成为评分的前置条件。"""
    clock = DeterministicClock()
    transport = FakeTransport([_openai_resp()], on_call=lambda i: clock.advance(0.1))
    ev = OpenAICompatEvaluator(
        "deepseek",
        "deepseek-chat",
        "https://api.deepseek.com/v1",
        "sk-test-key",
        transport,
        meter=None,
        pricing=PRICING,
        clock=clock.monotonic,
        sleeper=RecordingSleeper(),
    )

    res = run(ev.evaluate(_REQ))

    assert res.score == 72.0
    assert res.cost_usd == pytest.approx(0.0048)


# ---------- 场景 12：超时 ----------


def test_超时走可重试分支并在耗尽时如实上报():
    """防回归：TimeoutError 必须被当成"请求没发出去"，而不是解析失败。"""
    c = _ctx([TimeoutError("timed out"), _openai_resp()])

    res = run(c.ev.evaluate(_REQ))

    assert res.score == 72.0
    assert c.transport.n_calls == 2
    assert c.sleeper.calls == [0.5]


def test_超时耗尽后异常里带异常类型():
    c = _ctx([TimeoutError("timed out"), TimeoutError("timed out")])

    with pytest.raises(Unavailable) as exc:
        run(c.ev.evaluate(_REQ))

    assert "TimeoutError" in exc.value.reason
    assert "timed out" in exc.value.reason


# ---------- 解析失败的其它形态 ----------


def test_响应体不是JSON视为失败():
    c = _ctx([HttpResponse(200, b"<html>502 Bad Gateway</html>")], max_retries=0)

    with pytest.raises(Unavailable) as exc:
        run(c.ev.evaluate(_REQ))

    assert "响应体不是合法 JSON" in exc.value.reason


def test_响应体顶层不是对象视为失败():
    c = _ctx([HttpResponse(200, b"[1,2,3]")], max_retries=0)

    with pytest.raises(Unavailable) as exc:
        run(c.ev.evaluate(_REQ))

    assert "响应 JSON 顶层不是对象" in exc.value.reason


def test_缺少choices视为失败():
    c = _ctx([HttpResponse(200, b'{"id":"x"}')], max_retries=0)

    with pytest.raises(Unavailable) as exc:
        run(c.ev.evaluate(_REQ))

    assert "响应缺少 choices" in exc.value.reason


def test_choices元素不是对象视为失败():
    c = _ctx([HttpResponse(200, b'{"choices":["oops"]}')], max_retries=0)

    with pytest.raises(Unavailable) as exc:
        run(c.ev.evaluate(_REQ))

    assert "choices[0] 不是对象" in exc.value.reason


def test_message不是对象时视为缺少content():
    c = _ctx(
        [HttpResponse(200, b'{"choices":[{"finish_reason":"stop","message":null}]}')],
        max_retries=0,
    )

    with pytest.raises(Unavailable) as exc:
        run(c.ev.evaluate(_REQ))

    assert "响应缺少 message.content" in exc.value.reason


def test_模型输出不是JSON视为失败():
    c = _ctx([_openai_resp(raw="我不是 JSON，我是一段解释。")], max_retries=0)

    with pytest.raises(Unavailable) as exc:
        run(c.ev.evaluate(_REQ))

    assert "模型输出不是合法 JSON" in exc.value.reason


def test_模型输出顶层是数组视为失败():
    c = _ctx([_openai_resp(raw="[80,70,65]")], max_retries=0)

    with pytest.raises(Unavailable) as exc:
        run(c.ev.evaluate(_REQ))

    assert "模型输出顶层不是对象" in exc.value.reason


def test_feedback缺失或非字符串视为失败():
    c = _ctx([_openai_resp(_scores(feedback=None))], max_retries=0)
    with pytest.raises(Unavailable, match="缺少字段 feedback"):
        run(c.ev.evaluate(_REQ))

    c2 = _ctx([_openai_resp(_scores(feedback=123))], max_retries=0)
    with pytest.raises(Unavailable, match="feedback 不是字符串"):
        run(c2.ev.evaluate(_REQ))


# ---------- 请求构造 ----------


def test_请求走chat_completions且带json_schema强约束():
    """防回归：没有 schema 约束，模型随时会格式漂移（方案 §10）。"""
    c = _ctx([_openai_resp()])

    run(c.ev.evaluate(_REQ))

    call = c.transport.calls[0]
    assert call.url == "https://api.deepseek.com/v1/chat/completions"
    assert call.timeout_s == DEFAULT_TIMEOUT_S
    assert call.headers["Content-Type"] == "application/json"
    payload = call.payload
    assert payload["model"] == "deepseek-chat"
    assert payload["temperature"] == 0.0
    fmt = payload["response_format"]
    assert fmt["type"] == "json_schema"
    assert fmt["json_schema"]["name"] == "interview_evaluation"
    assert fmt["json_schema"]["strict"] is True
    assert fmt["json_schema"]["schema"] is EVAL_JSON_SCHEMA


def test_base_url带斜杠时不会拼出双斜杠():
    c = _ctx([_openai_resp()], base_url="https://api.deepseek.com/v1/")

    run(c.ev.evaluate(_REQ))

    assert c.transport.calls[0].url == "https://api.deepseek.com/v1/chat/completions"


def test_strict_schema可关闭以兼容国产端点():
    c = _ctx([_openai_resp()], strict_schema=False)

    run(c.ev.evaluate(_REQ))

    assert "strict" not in c.transport.calls[0].payload["response_format"]["json_schema"]


def test_提示词带上题目回答关键词与题型():
    c = _ctx([_openai_resp()])

    run(c.ev.evaluate(_REQ))

    messages = c.transport.calls[0].payload["messages"]
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == SYSTEM_PROMPT
    assert "只输出一个 JSON 对象" in messages[0]["content"]
    user = messages[1]["content"]
    assert _REQ.question in user
    assert _REQ.answer in user
    assert "推荐系统、向量检索、召回" in user
    assert "题目类型：technical" in user


def test_没有关键词时提示词显式写未提供():
    prompt = build_user_prompt(EvalRequest(question="q", answer="a", keywords=()))

    assert "（未提供）" in prompt


def test_schema由DIMENSIONS推导缺一不可():
    schema = build_eval_schema()

    assert set(schema["required"]) == set(DIMENSIONS) | {"score", "feedback", "confidence"}
    assert schema["additionalProperties"] is False
    for dim in DIMENSIONS:
        prop = schema["properties"][dim]
        assert prop["minimum"] == 0 and prop["maximum"] == 100
        assert DIM_DESCRIPTIONS[dim]


def test_请求构造不修改传入的EvalRequest():
    """防回归：EvalRequest 是 frozen，适配器不得靠可变共享状态传参。"""
    c = _ctx([_openai_resp()])
    req = EvalRequest(question="q", answer="a", keywords=("k",))

    run(c.ev.evaluate(req))

    assert req.keywords == ("k",)
