"""会话级可观测性（`observability/session_trace.py`）的测试。

钉住的纪律：

1. **失败调用必须留痕** —— `ok=False` 必须带非空 `error`；`ok=True` 不得挂 `error`。
2. **负耗时是错误，不是 0** —— `ended_ms < started_ms` 必须在记录时/取耗时时报错。
3. **排除必须留痕** —— 被排除的贡献要出现在 `explain()` 里，不许静默丢弃。
4. **call_id 唯一递增** —— 重复或倒退都要报错，否则时间线不可信。
5. **区分"没有会话"与"会话无数据"** —— 未知 session 返回 `None`，绝不造空视图。
"""

from __future__ import annotations

import pickle

import pytest

from ruipin.observability.session_trace import (
    Degradation,
    DimensionAttribution,
    DimensionContribution,
    MemoryTraceStore,
    ModelCall,
    SessionDebugView,
    StepTrace,
    TraceCollector,
    UnknownDimension,
)


def _call(
    call_id: int = 1,
    *,
    ok: bool = True,
    error: str | None = None,
    confidence: float | None = 0.9,
    cost_usd: float = 0.01,
    latency_ms: float = 120.0,
    turn_index: int | None = 0,
    provider: str = "deepseek",
    model: str = "v4",
) -> ModelCall:
    return ModelCall(
        call_id=call_id,
        turn_index=turn_index,
        provider=provider,
        model=model,
        prompt="问题",
        response="回答",
        cost_usd=cost_usd,
        latency_ms=latency_ms,
        confidence=confidence,
        ok=ok,
        error=error,
    )


# ========== ModelCall ==========


def test_failed_call_requires_error():
    """纪律：失败调用必须写明原因，否则事后无从归因。"""
    with pytest.raises(ValueError, match="失败调用必须写明 error"):
        _call(ok=False, error=None)
    with pytest.raises(ValueError, match="失败调用必须写明 error"):
        _call(ok=False, error="")


def test_successful_call_must_not_carry_error():
    """纪律：成功却挂 error 会让"到底成没成"含糊。"""
    with pytest.raises(ValueError, match="成功调用不应带 error"):
        _call(ok=True, error="boom")


def test_failed_call_with_error_is_valid():
    call = _call(ok=False, error="timeout", confidence=None)
    assert call.ok is False
    assert call.error == "timeout"


def test_call_rejects_negative_numbers():
    with pytest.raises(ValueError, match="call_id 不能为负"):
        _call(call_id=-1)
    with pytest.raises(ValueError, match="cost_usd 不能为负"):
        _call(cost_usd=-0.01)
    with pytest.raises(ValueError, match="latency_ms 不能为负"):
        _call(latency_ms=-1.0)


def test_call_rejects_out_of_range_confidence():
    with pytest.raises(ValueError, match="置信度"):
        _call(confidence=1.5)
    # confidence=None 表示"这次没给出置信度"，合法。
    assert _call(confidence=None).confidence is None


# ========== StepTrace ==========


def test_step_duration_computation():
    step = StepTrace("asr", 0, started_ms=100, ended_ms=250, timed_out=False, note="")
    assert step.duration_ms == 150


def test_step_duration_raises_on_negative_span():
    """纪律：负耗时说明时钟算错，不能静默归零。"""
    step = StepTrace("asr", 0, started_ms=250, ended_ms=100, timed_out=False, note="")
    with pytest.raises(ValueError, match="早于"):
        _ = step.duration_ms


# ========== DimensionContribution / Attribution ==========


def test_contribution_excluded_reason_cannot_be_empty_string():
    """纪律：要么不排除（None），要么写明原因；空串会让"排除"看起来像没排除。"""
    with pytest.raises(ValueError, match="排除原因"):
        DimensionContribution(0, 0.5, 1.0, 0.9, "p", excluded_reason="")
    assert DimensionContribution(0, 0.5, 1.0, 0.9, "p").excluded is False
    assert (
        DimensionContribution(0, 0.5, 1.0, 0.9, "p", excluded_reason="测不准").excluded is True
    )


def test_attribution_requires_dimension():
    with pytest.raises(ValueError, match="dimension"):
        DimensionAttribution("", (), None)


def test_explain_includes_excluded_contributions_and_reasons():
    """纪律（核心）：被排除的贡献不能被静默丢掉，必须在解释里出现。"""
    attribution = DimensionAttribution(
        dimension="表达清晰度",
        contributions=(
            DimensionContribution(0, 0.8, 1.0, 0.9, "deepseek", call_id=1),
            DimensionContribution(1, 0.2, 1.0, 0.1, "deepseek", call_id=2, excluded_reason="置信度不足"),
            DimensionContribution(2, 0.0, 0.0, 0.0, "asr"),  # call_id 为空，覆盖无调用分支
        ),
        final_value=0.8,
    )
    text = attribution.explain()
    assert "维度 '表达清晰度'" in text
    assert "0.8" in text
    assert "置信度不足" in text, "被排除的贡献及其原因必须出现在解释里"
    assert "第1轮" in text and "call#2" in text
    assert "无调用" in text


def test_explain_handles_none_final_value():
    attribution = DimensionAttribution(
        dimension="生理稳定度",
        contributions=(),
        final_value=None,
    )
    text = attribution.explain()
    assert "不可判定" in text
    assert attribution.excluded == ()


# ========== SessionDebugView ==========


def test_view_requires_session_id():
    with pytest.raises(ValueError, match="session_id"):
        SessionDebugView("")


def test_view_records_events_steps_calls_degradations():
    view = SessionDebugView("s1")
    event = view.append_event("turn_start", "开始第 1 轮", ts=10.0)
    assert event.ts == 10.0

    step = view.append_step("llm", started_ms=0, ended_ms=500, turn_index=0, note="ok")
    assert step.duration_ms == 500

    view.append_call(_call(call_id=1))
    view.append_call(_call(call_id=2, ok=False, error="rate_limited"))
    view.append_degradation(2, "视觉降级为逐帧", ts=11.0)

    snap = view.snapshot()
    assert snap.session_id == "s1"
    assert len(snap.events) == 1
    assert len(snap.steps) == 1
    assert len(snap.calls) == 2
    assert snap.total_cost_usd == pytest.approx(0.02)
    assert len(snap.failed_calls) == 1
    assert snap.degradations == (Degradation(2, "视觉降级为逐帧", 11.0),)


def test_append_step_rejects_reversed_clock():
    view = SessionDebugView("s1")
    with pytest.raises(ValueError, match="早于"):
        view.append_step("llm", started_ms=200, ended_ms=100)


def test_append_call_enforces_unique_increasing_ids():
    view = SessionDebugView("s1")
    view.append_call(_call(call_id=1))
    with pytest.raises(ValueError, match="唯一递增"):
        view.append_call(_call(call_id=1))  # 重复
    with pytest.raises(ValueError, match="唯一递增"):
        view.append_call(_call(call_id=0))  # 倒退
    view.append_call(_call(call_id=2))  # 递增正常
    assert [c.call_id for c in view.snapshot().calls] == [1, 2]


def test_append_degradation_validation():
    view = SessionDebugView("s1")
    with pytest.raises(ValueError, match="等级"):
        view.append_degradation(0, "原因", ts=1.0)
    with pytest.raises(ValueError, match="原因"):
        view.append_degradation(1, "", ts=1.0)


def test_dimension_lookup_and_unknown_error():
    view = SessionDebugView("s1")
    attribution = DimensionAttribution("流利度", (), 0.7)
    view.attributions.append(attribution)
    assert view.dimension("流利度") is attribution

    with pytest.raises(UnknownDimension) as excinfo:
        view.dimension("不存在")
    assert "不存在" in str(excinfo.value)
    assert isinstance(excinfo.value, KeyError)


def test_unknown_dimension_on_empty_view_lists_nothing():
    """没有任何归因时，异常消息不该硬拼空清单（known 为空这条分支要走到）。"""
    view = SessionDebugView("s1")
    with pytest.raises(UnknownDimension) as excinfo:
        view.dimension("流利度")
    assert "已知维度" not in str(excinfo.value)
    assert excinfo.value.known == ()


def test_unknown_dimension_pickles_roundtrip():
    err = UnknownDimension("流利度", ("表达",))
    restored = pickle.loads(pickle.dumps(err))
    assert restored.dimension == "流利度"
    assert restored.known == ("表达",)
    assert str(restored) == str(err)


def test_session_not_found_is_none_not_empty_view():
    """纪律（核心）："没有这个会话"与"会话没数据"必须能区分。"""
    store = MemoryTraceStore()
    assert store.for_session("ghost") is None

    empty_view = SessionDebugView("empty")
    store.add(empty_view)
    found = store.for_session("empty")
    assert found is not None
    snap = found.snapshot()
    assert snap.events == () and snap.steps == () and snap.calls == ()


def test_store_add_and_sessions_order():
    store = MemoryTraceStore()
    assert isinstance(store, TraceCollector), "内存存储应满足收集端口"

    a = SessionDebugView("a")
    b = SessionDebugView("b")
    store.add(a)
    store.add(b)
    assert store.for_session("a") is a
    assert store.sessions() == ("a", "b")


def test_store_add_same_session_keeps_latest():
    store = MemoryTraceStore()
    store.add(SessionDebugView("a"))
    newer = SessionDebugView("a")
    newer.append_event("x", "y", ts=1.0)
    store.add(newer)
    assert store.for_session("a") is newer
    assert len(store.sessions()) == 1
