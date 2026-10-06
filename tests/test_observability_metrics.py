"""`InMemoryMeter` 的单元测试。

防的回归：分位数被写成"最近秩"（小样本上过分乐观）、空样本返回 0（会被读成
"极快"从而把 SLO 判成达标）、样本窗口无界增长、快照里混进 NaN 导致无法进日志。
"""

from __future__ import annotations

import json
import math

import pytest

from ruipin.observability.metrics import (
    DEFAULT_MAX_SAMPLES,
    InMemoryMeter,
    linear_percentile,
    weighted_percentile,
)
from ruipin.ports import Meter, Usage

pytestmark = pytest.mark.unit

# 1..100：分位数可手算 —— h=(n-1)*p，线性插值
ONE_TO_HUNDRED = list(range(1, 101))


def test_linear_percentile_matches_hand_computed_p50_p95():
    """防回归：分位数必须用线性插值（h=(n-1)*p），不能取最近秩。"""
    assert linear_percentile(ONE_TO_HUNDRED, 0.5) == pytest.approx(50.5)   # 99*0.5=49.5
    assert linear_percentile(ONE_TO_HUNDRED, 0.95) == pytest.approx(95.05)  # 99*0.95=94.05
    assert linear_percentile(ONE_TO_HUNDRED, 1.0) == pytest.approx(100.0)   # h=99 整数
    assert linear_percentile(ONE_TO_HUNDRED, 0.01) == pytest.approx(1.99)   # h=0.99


def test_linear_percentile_empty_returns_none_not_zero():
    """防回归：空样本返回 None，0 会被下游读成"快得不可思议"从而误判达标。"""
    assert linear_percentile([], 0.95) is None
    assert linear_percentile([], 0.5) is None


def test_linear_percentile_single_sample_is_that_sample():
    """防回归：单样本不应走 (n-1) 插值把值压成 0。"""
    assert linear_percentile([1234.0], 0.95) == pytest.approx(1234.0)


def test_linear_percentile_rejects_out_of_range_p():
    """防回归：p 越界必须立刻炸，而不是返回一个看似合理的值。"""
    with pytest.raises(ValueError):
        linear_percentile(ONE_TO_HUNDRED, 0.0)
    with pytest.raises(ValueError):
        linear_percentile(ONE_TO_HUNDRED, 1.5)


def test_meter_satisfies_meter_protocol():
    """防回归：InMemoryMeter 必须能直接在核心层顶替 Meter（runtime_checkable）。"""
    assert isinstance(InMemoryMeter(), Meter)


def test_fresh_meter_queries_return_none_or_zero():
    """防回归：无数据时 min/max/mean/分位数返回 None（不是 0），count 返回 0。"""
    m = InMemoryMeter()
    assert m.count("turn.latency_ms") == 0
    assert m.percentile("turn.latency_ms", 0.95) is None
    assert m.mean("turn.latency_ms") is None
    assert m.minimum("turn.latency_ms") is None
    assert m.maximum("turn.latency_ms") is None
    assert m.summary("turn.latency_ms")["count"] == 0
    assert m.summary("turn.latency_ms")["p95"] is None


def test_observe_ms_percentile_and_summary():
    """防回归：具名延迟序列的分位数 / 均值 / 极值要能算对。"""
    m = InMemoryMeter()
    for ms in ONE_TO_HUNDRED:
        m.observe_ms("turn.latency_ms", ms)
    assert m.count("turn.latency_ms") == 100
    assert m.percentile("turn.latency_ms", 0.5) == pytest.approx(50.5)
    assert m.percentile("turn.latency_ms", 0.95) == pytest.approx(95.05)
    assert m.mean("turn.latency_ms") == pytest.approx(50.5)
    assert m.minimum("turn.latency_ms") == pytest.approx(1.0)
    assert m.maximum("turn.latency_ms") == pytest.approx(100.0)


def test_sample_cap_keeps_only_last_n_and_percentile_follows():
    """防回归：内存无界增长。灌 2N 条后只剩最近 N 条，分位数基于最近 N 条。"""
    m = InMemoryMeter(max_samples=5)
    for ms in range(1, 11):          # 1..10，只留 6..10
        m.observe_ms("x", ms)
    assert m.samples("x") == [6.0, 7.0, 8.0, 9.0, 10.0]
    assert m.count("x") == 5
    # 最近 5 条 [6,7,8,9,10]：h=(5-1)*0.5=2 → 8；h=4*0.95=3.8 → 9.8
    assert m.percentile("x", 0.5) == pytest.approx(8.0)
    assert m.percentile("x", 0.95) == pytest.approx(9.8)


def test_default_max_samples_constant():
    """防回归：默认窗口 1000 条，改动要被测试看见。"""
    assert DEFAULT_MAX_SAMPLES == 1000
    assert InMemoryMeter().snapshot()["max_samples"] == 1000


def test_invalid_max_samples_rejected():
    """防回归：窗口上限非法（0 / 非 int）必须拒绝，否则会静默丢样本。"""
    with pytest.raises(ValueError):
        InMemoryMeter(max_samples=0)
    with pytest.raises(TypeError):
        InMemoryMeter(max_samples=5.0)


def test_observe_ms_rejects_nan_infinite_and_negative():
    """防回归：NaN 会污染排序、负数说明时钟算错，脏数据不许入样本。"""
    m = InMemoryMeter()
    with pytest.raises(TypeError):
        m.observe_ms("x", "1200")
    with pytest.raises(ValueError):
        m.observe_ms("x", float("nan"))
    with pytest.raises(ValueError):
        m.observe_ms("x", float("inf"))
    with pytest.raises(ValueError):
        m.observe_ms("x", -1.0)
    assert m.count("x") == 0


def test_counter_plain_and_tagged():
    """防回归：计数器要支持 tags，且 tags 的插入顺序不影响同一个键。"""
    m = InMemoryMeter()
    m.incr("vlm.frames")
    m.incr("vlm.frames", 2)
    m.incr("degrade", 1, tags={"reason": "timeout", "provider": "vlm"})
    m.incr("degrade", 1, tags={"provider": "vlm", "reason": "timeout"})  # 顺序不同
    assert m.counter("vlm.frames") == 3
    assert m.counter("degrade", {"reason": "timeout", "provider": "vlm"}) == 2
    assert m.counter("degrade") == 0
    assert m.counters()["degrade{provider=vlm,reason=timeout}"] == 2
    assert m.counters()["vlm.frames"] == 3


def test_counter_rejects_non_int_value():
    """防回归：计数增量非整数会让计数变成浮点，静默丢精度。"""
    m = InMemoryMeter()
    with pytest.raises(TypeError):
        m.incr("x", 1.5)
    assert m.counter("x") == 0


def test_record_usage_buckets_by_provider_and_operation():
    """防回归：用量必须按 provider/operation 分桶，否则成本无法归因。"""
    m = InMemoryMeter()
    m.record_usage(
        Usage(provider="openai", operation="score", latency_ms=1200,
              prompt_tokens=1000, completion_tokens=200, cost_usd=0.02)
    )
    m.record_usage(
        Usage(provider="openai", operation="score", latency_ms=2400,
              prompt_tokens=1000, completion_tokens=200, cost_usd=0.02, cached=True)
    )
    m.record_usage(
        Usage(provider="qwen", operation="tts", latency_ms=800,
              prompt_tokens=0, completion_tokens=0, cost_usd=0.005)
    )
    buckets = m.usage_buckets()
    score = buckets["openai|score"]
    assert score["calls"] == 2
    assert score["cached_calls"] == 1
    assert score["prompt_tokens"] == 2000
    assert score["completion_tokens"] == 400
    assert score["total_tokens"] == 2400
    assert score["cost_usd"] == pytest.approx(0.04)
    assert score["latency"]["p50"] == pytest.approx(1800.0)   # (1200+2400)/2
    assert buckets["qwen|tts"]["calls"] == 1
    assert m.usage_totals() == {
        "calls": 3,
        "cached_calls": 1,
        "prompt_tokens": 2000,
        "completion_tokens": 400,
        "cost_usd": pytest.approx(0.045),
    }


def test_record_usage_rejects_bad_cost_and_latency():
    """防回归：负成本 / NaN 成本 / 负延迟会让账目与分位数失真，必须拒绝。"""
    m = InMemoryMeter()
    with pytest.raises(ValueError):
        m.record_usage(Usage(provider="p", operation="o", cost_usd=-0.1))
    with pytest.raises(ValueError):
        m.record_usage(Usage(provider="p", operation="o", cost_usd=float("nan")))
    with pytest.raises(ValueError):
        m.record_usage(Usage(provider="p", operation="o", latency_ms=-5))
    assert m.usage_buckets() == {}


def test_weighted_percentile_uniform_weights_equals_linear():
    """防回归：加权分位数与线性插值分位数必须同一口径（权重全 1 时应对拍一致）。"""
    pairs = [(float(v), 1.0) for v in ONE_TO_HUNDRED]
    assert weighted_percentile(pairs, 0.5) == pytest.approx(50.5)
    assert weighted_percentile(pairs, 0.95) == pytest.approx(95.05)


def test_weighted_percentile_hand_computed_two_points():
    """防回归：加权结果要能手算 —— 1 份 10ms + 3 份 20ms，位置轴 [0,2]。"""
    pairs = [(10.0, 1.0), (20.0, 3.0)]
    assert weighted_percentile(pairs, 0.5) == pytest.approx(15.0)   # target=1.0 → 半程
    assert weighted_percentile(pairs, 0.9) == pytest.approx(19.0)   # target=1.8 → 0.9
    assert weighted_percentile(pairs, 1.0) == pytest.approx(20.0)


def test_weighted_percentile_empty_and_invalid_weight():
    """防回归：空样本返回 None；权重非正会让加权失去意义，必须拒绝。"""
    assert weighted_percentile([], 0.95) is None
    assert weighted_percentile([(10.0, 1.0)], 0.95) == pytest.approx(10.0)
    with pytest.raises(ValueError):
        weighted_percentile([(10.0, 0.0), (20.0, 1.0)], 0.95)
    with pytest.raises(ValueError):
        weighted_percentile([(10.0, 1.0)], 1.5)


def test_observe_weighted_ms_and_queries():
    """防回归：加权序列的条数 / 权重合计 / 分位数，且与窗口上限一致。"""
    m = InMemoryMeter(max_samples=3)
    m.observe_weighted_ms("turn.weighted.latency_ms", 9000.0, 1.0)
    m.observe_weighted_ms("turn.weighted.latency_ms", 1000.0, 9.0)
    assert m.count_weighted("turn.weighted.latency_ms") == 2
    assert m.total_weight("turn.weighted.latency_ms") == pytest.approx(10.0)
    # [(1000,9),(9000,1)]：位置轴 r0=(9-1)/2=4, r1=9+(1-1)/2=9；p95 → 4+0.95*5=8.75
    # frac=(8.75-4)/5=0.95 → 1000+0.95*8000=8600
    assert m.percentile_weighted("turn.weighted.latency_ms", 0.95) == pytest.approx(
        8600.0
    )
    assert m.percentile_weighted("nope", 0.95) is None
    assert m.count_weighted("nope") == 0
    assert m.total_weight("nope") == 0.0


def test_observe_weighted_ms_rejects_bad_weight():
    """防回归：0 / 负 / NaN / inf 权重会静默扭曲加权结果。"""
    m = InMemoryMeter()
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            m.observe_weighted_ms("x", 100.0, bad)
    assert m.count_weighted("x") == 0


def test_snapshot_is_strictly_json_serializable():
    """防回归：快照要能直接进日志 / 告警；allow_nan=False 下不许有 NaN/Inf。"""
    m = InMemoryMeter(max_samples=4)
    m.observe_ms("turn.latency_ms", 1500)
    m.observe_ms("turn.latency_ms", 2500)
    m.observe_weighted_ms("turn.weighted.latency_ms", 1500, 3.0)
    m.incr("degrade", 2, tags={"reason": "timeout"})
    m.record_usage(
        Usage(provider="openai", operation="score", latency_ms=1500,
              prompt_tokens=500, completion_tokens=100, cost_usd=0.01)
    )
    snap = m.snapshot()
    text = json.dumps(snap, allow_nan=False, ensure_ascii=False)
    assert json.loads(text)["max_samples"] == 4
    assert snap["latency"]["turn.latency_ms"]["samples"] == [1500.0, 2500.0]
    assert snap["latency"]["turn.latency_ms"]["p50"] == pytest.approx(2000.0)
    assert snap["weighted_latency"]["turn.weighted.latency_ms"]["total_weight"] == 3.0
    assert snap["counters"]["degrade{reason=timeout}"] == 2
    assert snap["usage"]["openai|score"]["cost_usd"] == pytest.approx(0.01)
    assert snap["totals"]["calls"] == 1


def test_empty_snapshot_is_json_serializable_with_none_values():
    """防回归：空台账的快照也要能序列化，且数值字段是 None 而不是 0。"""
    snap = InMemoryMeter().snapshot()
    json.dumps(snap, allow_nan=False)
    assert snap["latency"] == {}
    assert snap["usage"] == {}
    assert snap["totals"]["cost_usd"] == 0.0


def test_summary_and_samples_helpers_share_one_source():
    """防回归：summary() 与 samples() 必须读同一份样本，不能各算一套。"""
    m = InMemoryMeter()
    m.observe_ms("x", 100.0)
    m.observe_ms("x", 300.0)
    summary = m.summary("x")
    assert summary == {
        "count": 2,
        "min": 100.0,
        "max": 300.0,
        "mean": pytest.approx(200.0),
        "p50": pytest.approx(200.0),
        "p95": pytest.approx(290.0),
    }
    assert m.samples("x") == [100.0, 300.0]


def test_no_nan_ever_enters_percentile_path():
    """防回归：任何路径都不许产出 NaN 分位数（会让 SLO 比较恒为 False 而静音）。"""
    m = InMemoryMeter()
    m.observe_ms("x", 0)
    m.observe_ms("x", 0)
    assert not math.isnan(m.percentile("x", 0.95))
    assert m.percentile("x", 0.95) == pytest.approx(0.0)
