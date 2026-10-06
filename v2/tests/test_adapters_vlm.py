"""`adapters/vlm.py` 的测试。

防的回归
--------
1. **成本护栏**：预算不足时抛 `BudgetExceeded`，且**不发起任何网络调用**
   （不静默跳过、不假装成功）。
2. **帧数硬闸**：超过每轮上限抛错（方案 §4.2：每轮 ≤12 帧）。
3. **低置信原样返回**：绝不擅自把 confidence 抬高（否则幻觉证据会混进评分）。
4. **解析失败即失败**：响应不符合 schema → `Unavailable`，绝不合成一个观察结果。

预算用真实实现 `perception.budget.SessionBudget`（只读引用，不修改）；客户端脚本化；
时钟注入，不真睡。
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional, Sequence

import pytest

from ruipin.adapters.fakes import DeterministicClock
from ruipin.adapters.vlm import (
    DEFAULT_MAX_FRAMES,
    DEFAULT_VLM_SCHEMA,
    OPS_LATENCY_KEY,
    CloudVLM,
    VLMResponse,
    estimate_vlm_cost_usd,
)
from ruipin.domain.errors import BudgetExceeded, Unavailable
from ruipin.perception.budget import SessionBudget
from ruipin.ports import VLMPort

run = asyncio.run


class RecordingMeter:
    """内存版 `Meter`。"""

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


class _Call:
    __slots__ = ("frames", "prompt", "schema", "timeout_s")

    def __init__(self, frames, prompt, schema, timeout_s) -> None:
        self.frames = frames
        self.prompt = prompt
        self.schema = schema
        self.timeout_s = timeout_s


class FakeVLMClient:
    """脚本化视觉客户端：按次序返回 `VLMResponse`（或抛异常）。"""

    def __init__(
        self,
        script: Sequence[Any] = (),
        *,
        default: Optional[Any] = None,
        on_call: Optional[Any] = None,
    ) -> None:
        self.script = list(script)
        self.default = default
        self.on_call = on_call
        self.calls: list[_Call] = []

    async def observe(self, frames, prompt, schema, timeout_s) -> VLMResponse:
        index = len(self.calls)
        self.calls.append(_Call(list(frames), prompt, schema, timeout_s))
        if self.on_call is not None:
            self.on_call(index)
        item = self._item_at(index)
        if isinstance(item, BaseException):
            raise item
        return item

    def _item_at(self, index: int) -> Any:
        if index < len(self.script):
            return self.script[index]
        if self.default is not None:
            return self.default
        raise AssertionError(f"FakeVLMClient 脚本已用尽：第 {index + 1} 次调用无预设")


def _payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "summary": "候选人神情平静，目光看向摄像头",
        "confidence": 0.82,
        "labels": ["neutral", "camera"],
        "evidence_frames": [0, 3],
    }
    payload.update(overrides)
    return payload


def _make(client, *, budget=None, meter=None, clock=None, **kwargs):
    clock = clock or DeterministicClock()
    meter = meter if meter is not None else RecordingMeter()
    budget = budget if budget is not None else SessionBudget(cap_usd=1.0)
    vlm = CloudVLM("v-1", client, budget, meter=meter, clock=clock.monotonic, **kwargs)
    return vlm, client, budget, meter


# ---------- 正常路径 ----------


def test_observe_returns_fields_and_records_usage():
    """防回归：正常调用字段正确，且 cost/latency 被 meter 记录。"""
    clock = DeterministicClock()
    client = FakeVLMClient(
        [VLMResponse(_payload(), cost_usd=0.002, latency_ms=120)],
        on_call=lambda _i: clock.advance(0.15),
    )
    vlm, _client, budget, meter = _make(client, clock=clock)
    frames = [b"f0", b"f1", b"f2"]

    obs = run(vlm.observe(frames, "描述候选人状态"))
    assert obs.summary == "候选人神情平静，目光看向摄像头"
    assert obs.confidence == 0.82
    assert obs.labels == ("neutral", "camera")
    assert obs.evidence_frames == (0, 3)
    assert obs.provider == "v-1"
    assert obs.cost_usd == pytest.approx(0.002)
    assert obs.latency_ms == 150

    assert len(meter.usages) == 1
    assert meter.usages[0].operation == OPS_LATENCY_KEY
    assert meter.usages[0].cost_usd == pytest.approx(0.002)
    assert (OPS_LATENCY_KEY, 150.0) in meter.observations
    assert budget.spent == pytest.approx(0.002)


def test_observe_sends_frames_prompt_and_default_schema():
    """防回归：多帧打包成一次调用（§4.2），下发默认 schema。"""
    client = FakeVLMClient([VLMResponse(_payload())])
    vlm, _c, _b, _m = _make(client)
    run(vlm.observe([b"a", b"b"], "prompt-x"))
    call = client.calls[0]
    assert call.frames == [b"a", b"b"]
    assert call.prompt == "prompt-x"
    assert call.schema is DEFAULT_VLM_SCHEMA


def test_observe_uses_custom_schema_when_given():
    """防回归：调用级 schema 覆盖默认 schema（屏幕共享等独立 task schema）。"""
    custom = {"type": "object", "properties": {"code_visible": {"type": "boolean"}}}
    client = FakeVLMClient([VLMResponse(_payload())])
    vlm, _c, _b, _m = _make(client)
    run(vlm.observe([b"a"], "p", schema=custom))
    assert client.calls[0].schema is custom


def test_low_confidence_is_returned_unchanged():
    """防回归（核心）：低 confidence 原样返回，绝不被抬到"看起来还行"的区间。"""
    client = FakeVLMClient([VLMResponse(_payload(confidence=0.15))])
    vlm, _c, _b, _m = _make(client)
    obs = run(vlm.observe([b"a"], "p"))
    assert obs.confidence == 0.15


def test_missing_optional_lists_default_to_empty():
    """防回归：labels / evidence 缺失时给空元组，而不是编造证据帧。"""
    client = FakeVLMClient(
        [VLMResponse({"summary": "s", "confidence": 0.5})]
    )
    vlm, _c, _b, _m = _make(client)
    obs = run(vlm.observe([b"a"], "p"))
    assert obs.labels == () and obs.evidence_frames == ()


# ---------- 成本护栏 ----------


def test_budget_insufficient_raises_and_makes_no_call():
    """防回归（核心）：预算不足 → BudgetExceeded，且 client 调用数为 0。"""
    client = FakeVLMClient([VLMResponse(_payload())])
    budget = SessionBudget(cap_usd=0.001)  # 低于默认 estimate
    vlm, _c, _b, _m = _make(client, budget=budget)
    with pytest.raises(BudgetExceeded) as ei:
        run(vlm.observe([b"a", b"b", b"c", b"d"], "p"))
    assert ei.value.kind == "vlm_usd"
    assert len(client.calls) == 0
    assert budget.spent == 0.0


def test_budget_exceeded_reports_limit_and_used():
    """防回归：护栏抛错要带上限与已用额，告警无需回代码里推。"""
    client = FakeVLMClient([VLMResponse(_payload())])
    budget = SessionBudget(cap_usd=0.0)
    vlm, _c, _b, _m = _make(client, budget=budget)
    with pytest.raises(BudgetExceeded) as ei:
        run(vlm.observe([b"a"], "p"))
    assert ei.value.limit == 0.0
    assert ei.value.used is not None and ei.value.used > 0


def test_actual_cost_exceeding_cap_fails_hard_on_spend():
    """防回归：真实成本超过硬顶时 spend 抛 BudgetExceeded（不静默）。"""
    client = FakeVLMClient([VLMResponse(_payload(), cost_usd=0.01)])
    budget = SessionBudget(cap_usd=0.005)
    vlm, _c, _b, _m = _make(client, budget=budget)
    with pytest.raises(BudgetExceeded):
        run(vlm.observe([b"a"], "p"))


def test_custom_cost_estimator_used_for_precheck():
    """防回归：预算预检走注入的 estimator（可按实际单价标定）。"""
    client = FakeVLMClient([VLMResponse(_payload())])
    budget = SessionBudget(cap_usd=1.0)
    vlm, _c, _b, _m = _make(client, budget=budget, cost_estimator=lambda n: 0.5)
    run(vlm.observe([b"a"], "p"))
    assert budget.spent == pytest.approx(0.0)  # 只花真实成本(0)，预检不扣费


# ---------- 帧数硬闸 ----------


def test_too_many_frames_raises_budget_exceeded_without_call():
    """防回归：帧数超每轮上限 → 抛错且不发请求（§4.2 硬闸）。"""
    client = FakeVLMClient([VLMResponse(_payload())])
    vlm, _c, _b, _m = _make(client, max_frames=2)
    with pytest.raises(BudgetExceeded) as ei:
        run(vlm.observe([b"a", b"b", b"c"], "p"))
    assert ei.value.kind == "vlm_frames"
    assert ei.value.limit == 2 and ei.value.used == 3
    assert len(client.calls) == 0


def test_default_max_frames_is_twelve():
    """防回归：默认帧上限与方案 §4.2 一致（12）。"""
    assert DEFAULT_MAX_FRAMES == 12


def test_empty_frames_is_value_error():
    """防回归：没有帧时明确报错，而不是发一个空请求烧钱。"""
    client = FakeVLMClient([VLMResponse(_payload())])
    vlm, _c, _b, _m = _make(client)
    with pytest.raises(ValueError):
        run(vlm.observe([], "p"))
    assert len(client.calls) == 0


# ---------- 解析失败 → 显式失败 ----------


def test_client_exception_becomes_unavailable():
    """防回归：客户端异常 → Unavailable（不是空观察）。"""
    client = FakeVLMClient([RuntimeError("vlm down")])
    vlm, _c, _b, meter = _make(client)
    with pytest.raises(Unavailable) as ei:
        run(vlm.observe([b"a"], "p"))
    assert "RuntimeError" in ei.value.reason
    assert meter.counters.get("vlm.error") == 1


def test_unavailable_passthrough():
    """防回归：结构性缺失（已是 Unavailable）原样上抛。"""
    client = FakeVLMClient([Unavailable("cloud", "no key")])
    vlm, _c, _b, meter = _make(client)
    with pytest.raises(Unavailable) as ei:
        run(vlm.observe([b"a"], "p"))
    assert ei.value.provider == "cloud"
    assert "vlm.error" not in meter.counters


@pytest.mark.parametrize(
    "payload",
    [
        {"confidence": 0.5},  # 缺 summary
        {"summary": "s"},  # 缺 confidence
        {"summary": "s", "confidence": True},  # bool 冒充数值
        {"summary": "s", "confidence": "0.5"},  # 非数值
        {"summary": "s", "confidence": 1.5},  # 越界
        {"summary": "s", "confidence": -0.1},  # 越界
        {"summary": 5, "confidence": 0.5},  # summary 非字符串
        {"summary": "s", "confidence": 0.5, "labels": "x"},  # labels 非数组
        {"summary": "s", "confidence": 0.5, "labels": [1]},  # labels 元素非字符串
        {"summary": "s", "confidence": 0.5, "evidence_frames": [1.5]},  # 非整数
        {"summary": "s", "confidence": 0.5, "evidence_frames": [-1]},  # 负索引
        {"summary": "s", "confidence": 0.5, "evidence_frames": "0"},  # 非数组
    ],
)
def test_malformed_payload_raises_unavailable(payload):
    """防回归：任何不符合 schema 的响应都是 Unavailable，绝不合成观察。"""
    client = FakeVLMClient([VLMResponse(payload)])
    vlm, _c, _b, _m = _make(client)
    with pytest.raises(Unavailable):
        run(vlm.observe([b"a"], "p"))


def test_non_mapping_payload_raises_unavailable():
    """防回归：payload 顶层非对象 → Unavailable。"""
    client = FakeVLMClient([VLMResponse([1, 2, 3])])  # type: ignore[arg-type]
    vlm, _c, _b, _m = _make(client)
    with pytest.raises(Unavailable) as ei:
        run(vlm.observe([b"a"], "p"))
    assert "不是对象" in ei.value.reason


# ---------- 工具与契约 ----------


def test_estimate_cost_scales_with_frames():
    """防回归：成本预检随帧数单调增长（负帧数按 0 处理）。"""
    assert estimate_vlm_cost_usd(0) < estimate_vlm_cost_usd(10)
    assert estimate_vlm_cost_usd(-5) == estimate_vlm_cost_usd(0)


def test_implements_vlm_port_protocol():
    """防回归：CloudVLM 必须满足 VLMPort。"""
    vlm, _c, _b, _m = _make(FakeVLMClient([VLMResponse(_payload())]))
    assert isinstance(vlm, VLMPort)


def test_no_meter_is_silent_on_success_and_failure():
    """防回归：meter=None 时可观测性完全跳过，不影响观察结果。"""
    ok = CloudVLM("v-1", FakeVLMClient([VLMResponse(_payload())]), SessionBudget(cap_usd=1.0), meter=None)
    assert run(ok.observe([b"a"], "p")).summary == "候选人神情平静，目光看向摄像头"

    bad = CloudVLM("v-1", FakeVLMClient([RuntimeError("x")]), SessionBudget(cap_usd=1.0), meter=None)
    with pytest.raises(Unavailable):
        run(bad.observe([b"a"], "p"))
