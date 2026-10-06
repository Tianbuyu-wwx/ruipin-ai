"""测试替身（fakes）与 EvalResult 校验的测试。

核心断言：`fail=True` 时**抛 Unavailable 且不产生任何分数**——这是 v2 相对
遗留系统（崩溃即返回全 60 分）的第一原则，必须有自动化断言守住。
"""

from __future__ import annotations

import asyncio
import time

import pytest

from ruipin.adapters.fakes import (
    DEFAULT_DIMS,
    DeterministicClock,
    FakeEvaluator,
    FakeTTS,
    FakeVLMEvaluator,
)
from ruipin.domain.errors import Unavailable
from ruipin.ports import DIMENSIONS, EvalRequest, EvalResult, Evaluator

# 环境未装 pytest-asyncio，统一用同步测试 + asyncio.run
run = asyncio.run


def _ok_dims() -> dict[str, float]:
    return {d: 60.0 for d in DIMENSIONS}


def _ok_result(**kw) -> EvalResult:
    base = dict(
        dims=_ok_dims(),
        score=60.0,
        feedback="f",
        provider="fake",
        confidence=0.9,
    )
    base.update(kw)
    return EvalResult(**base)


# ---------- FakeEvaluator：正常路径 ----------


def test_implements_evaluator_protocol():
    assert isinstance(FakeEvaluator(), Evaluator)


def test_returns_all_six_dimensions_in_range(fake_evaluator, sample_eval_request):
    res = run(fake_evaluator.evaluate(sample_eval_request))

    assert set(res.dims) == set(DIMENSIONS)
    for d, v in res.dims.items():
        assert 0.0 <= v <= 100.0, d
    assert res.provider == "fake"
    assert 0.0 <= res.confidence <= 1.0
    assert res.degraded is False


def test_score_is_mean_of_dims(fake_evaluator, sample_eval_request):
    res = run(fake_evaluator.evaluate(sample_eval_request))
    assert res.score == 65.0  # DEFAULT_DIMS 均值
    assert res.score == round(sum(DEFAULT_DIMS.values()) / len(DEFAULT_DIMS), 2)


def test_records_calls(fake_evaluator, sample_eval_request):
    other = EvalRequest(question="q2", answer="a2")
    run(fake_evaluator.evaluate(sample_eval_request))
    run(fake_evaluator.evaluate(other))

    assert fake_evaluator.calls == [sample_eval_request, other]


def test_custom_dims_and_provider_are_respected(sample_eval_request):
    dims = {d: 10.0 + i for i, d in enumerate(DIMENSIONS)}
    ev = FakeEvaluator(dims=dims, provider="fake-custom", confidence=0.5)

    res = run(ev.evaluate(sample_eval_request))
    assert res.dims == dims
    assert res.provider == "fake-custom"
    assert ev.name == "fake-custom"
    assert res.confidence == 0.5


def test_result_is_a_fresh_copy_each_call(fake_evaluator, sample_eval_request):
    r1 = run(fake_evaluator.evaluate(sample_eval_request))
    r2 = run(fake_evaluator.evaluate(sample_eval_request))
    r1.dims["technical"] = 0.0
    assert r2.dims["technical"] == DEFAULT_DIMS["technical"]


def test_reset_clears_calls(fake_evaluator, sample_eval_request):
    run(fake_evaluator.evaluate(sample_eval_request))
    fake_evaluator.reset()
    assert fake_evaluator.calls == []


def test_slow_sleeps_before_returning(sample_eval_request):
    ev = FakeEvaluator(slow_s=0.05)
    t0 = time.perf_counter()
    res = run(ev.evaluate(sample_eval_request))
    assert time.perf_counter() - t0 >= 0.05
    assert res.score == 65.0


# ---------- FakeEvaluator：失败路径（禁止合成分数） ----------


def test_fail_raises_unavailable_and_produces_no_score(
    sample_eval_request,
):
    ev = FakeEvaluator(fail=True)

    with pytest.raises(Unavailable) as exc:
        run(ev.evaluate(sample_eval_request))

    assert exc.value.provider == "fake"
    assert "injected failure" in str(exc.value)
    # 调用被记录了（可断言"确实尝试过"），但**没有任何返回值**
    assert ev.calls == [sample_eval_request]


def test_fail_raises_even_with_slow_s(sample_eval_request):
    ev = FakeEvaluator(fail=True, slow_s=0.01)
    with pytest.raises(Unavailable):
        run(ev.evaluate(sample_eval_request))


def test_fail_raises_unavailable_not_base_exception(sample_eval_request):
    """必须是领域异常 Unavailable，而非裸 Exception / 0 分兜底。"""
    ev = FakeEvaluator(fail=True)
    with pytest.raises(Unavailable):
        run(ev.evaluate(sample_eval_request))
    assert not isinstance(Unavailable("p", "r"), ValueError)


# ---------- Golden Master ----------


def test_golden_master_default_result(sample_eval_request):
    res = run(FakeEvaluator().evaluate(sample_eval_request))
    assert res.dims == DEFAULT_DIMS
    assert (res.score, res.provider, res.latency_ms, res.cost_usd) == (
        65.0,
        "fake",
        0,
        0.0,
    )


# ---------- FakeVLMEvaluator ----------


def test_vlm_records_frames_and_provider(sample_eval_request):
    vlm = FakeVLMEvaluator()
    frames = [b"f1", b"f2"]

    res = run(vlm.evaluate(sample_eval_request, frames=frames))

    assert res.provider == "fake-vlm"
    assert vlm.name == "fake-vlm"
    assert vlm.frames_seen == [frames]
    assert vlm.calls == [sample_eval_request]


def test_vlm_records_empty_frames_when_absent(sample_eval_request):
    vlm = FakeVLMEvaluator()
    run(vlm.evaluate(sample_eval_request))
    assert vlm.frames_seen == [[]]


def test_vlm_reset_clears_everything(sample_eval_request):
    vlm = FakeVLMEvaluator(fail=True)
    with pytest.raises(Unavailable):
        run(vlm.evaluate(sample_eval_request, frames=[b"f"]))
    vlm.reset()
    assert vlm.frames_seen == []
    assert vlm.calls == []


def test_vlm_failure_carries_vlm_provider(sample_eval_request):
    vlm = FakeVLMEvaluator(fail=True)
    with pytest.raises(Unavailable) as exc:
        run(vlm.evaluate(sample_eval_request, frames=[b"f"]))
    assert exc.value.provider == "fake-vlm"


# ---------- FakeTTS ----------


def test_tts_streams_chunks_and_records_calls():
    tts = FakeTTS(chunk_size=2)

    async def collect() -> list[bytes]:
        stream = await tts.synth_stream("你好世界")
        return [chunk async for chunk in stream]

    chunks = run(collect())
    assert len(chunks) == 2  # ceil(4 字符 / 2)
    assert all(isinstance(c, bytes) and c for c in chunks)
    assert b"".join(chunks).decode() == "0000|你好世界0001|你好世界"
    assert tts.synth_calls == ["你好世界"]


def test_tts_n_chunks_override():
    tts = FakeTTS(n_chunks=3)

    async def collect() -> list[bytes]:
        stream = await tts.synth_stream("hi")
        return [c async for c in stream]

    assert len(run(collect())) == 3


def test_tts_fail_raises_unavailable_without_yielding_bytes():
    tts = FakeTTS(fail=True)

    async def go():
        with pytest.raises(Unavailable):
            await tts.synth_stream("hi")

    run(go())
    assert tts.synth_calls == ["hi"]  # 尝试过，但未产出任何音频


def test_tts_reset():
    tts = FakeTTS()
    run(tts.synth_stream("a"))
    tts.reset()
    assert tts.synth_calls == []


def test_tts_is_deterministic():
    async def once(text: str) -> bytes:
        stream = await FakeTTS(n_chunks=2).synth_stream(text)
        return b"".join([c async for c in stream])

    assert run(once("same")) == run(once("same"))


# ---------- DeterministicClock ----------


def test_clock_starts_at_zero_and_advances():
    c = DeterministicClock()
    assert c.monotonic() == 0.0
    assert c.now() == 0.0
    assert c.advance(1.5) == 1.5
    assert c.monotonic() == 1.5
    assert c.t == 1.5


def test_clock_advance_accumulates_and_accepts_start():
    c = DeterministicClock(100.0)
    c.advance(0.5)
    c.advance(2.5)
    assert c.monotonic() == 103.0


def test_clock_rejects_negative_advance():
    c = DeterministicClock(10.0)
    with pytest.raises(ValueError):
        c.advance(-1)
    assert c.monotonic() == 10.0


def test_clock_set_and_reset():
    c = DeterministicClock(5.0)
    c.set(42.0)
    assert c.monotonic() == 42.0
    c.reset()
    assert c.monotonic() == 5.0


def test_clock_does_not_move_on_its_own():
    c = DeterministicClock(7.0)
    assert c.monotonic() == c.monotonic() == 7.0


# ---------- EvalResult 校验（端口层 __post_init__） ----------


def test_result_rejects_missing_dimension():
    dims = _ok_dims()
    del dims["leadership"]
    with pytest.raises(ValueError, match="缺少维度"):
        _ok_result(dims=dims)


def test_result_rejects_out_of_range_dimension():
    dims = _ok_dims()
    dims["technical"] = 101.0
    with pytest.raises(ValueError, match="越界"):
        _ok_result(dims=dims)

    dims["technical"] = -0.1
    with pytest.raises(ValueError, match="越界"):
        _ok_result(dims=dims)


def test_result_rejects_non_numeric_dimension():
    dims = _ok_dims()
    dims["technical"] = "90"
    with pytest.raises(TypeError, match="非数值"):
        _ok_result(dims=dims)


def test_result_rejects_bad_confidence():
    with pytest.raises(ValueError, match="confidence"):
        _ok_result(confidence=1.5)
    with pytest.raises(ValueError, match="confidence"):
        _ok_result(confidence=-0.01)


def test_result_rejects_degraded_without_reason():
    with pytest.raises(ValueError, match="degrade_reason"):
        _ok_result(degraded=True, degrade_reason=None)


def test_result_accepts_degraded_with_reason():
    res = _ok_result(degraded=True, degrade_reason="VLM 超时")
    assert res.degraded is True
    assert res.degrade_reason == "VLM 超时"


def test_result_accepts_extra_physio_dimension():
    dims = _ok_dims()
    dims["stress_regulation"] = 55.0
    res = _ok_result(dims=dims)
    assert res.dims["stress_regulation"] == 55.0


def test_fake_can_emit_degraded_result(sample_eval_request):
    ev = FakeEvaluator(degraded=True, degrade_reason="stub")
    res = run(ev.evaluate(sample_eval_request))
    assert res.degraded is True and res.degrade_reason == "stub"
