"""聚合器测试（方案 N5：可靠性 → 权重 → 重分配）。

红线：关闭生理模块（或任何低可靠性维度）不得导致总分被系统性压低。
所有浮点比较用 pytest.approx。
"""

from __future__ import annotations

import math
import random

import pytest

from ruipin.domain.errors import Unavailable
from ruipin.ports import DimensionScore, PHYSIO_DIM
from ruipin.scoring.aggregate import (
    DEGRADED_GATE_FACTOR,
    GATE_HIGH,
    GATE_LOW,
    DEFAULT_WEIGHTS,
    default_gate,
    aggregate,
)


def ds(value: float, confidence: float = 1.0, degraded: bool = False, provider: str = "test"):
    return DimensionScore(value=value, confidence=confidence, degraded=degraded, provider=provider)


# ---------- 门控 ----------


@pytest.mark.parametrize(
    "conf,expected",
    [
        (0.0, 0.0),
        (GATE_LOW - 0.01, 0.0),
        (GATE_LOW, 0.0),                      # 下界：归零
        ((GATE_LOW + GATE_HIGH) / 2, 0.5),    # 中点：线性段中值
        (GATE_HIGH - 0.01, pytest.approx(0.34 / 0.35)),
        (GATE_HIGH, 1.0),                     # 上界：满权重
        (0.9, 1.0),
        (1.0, 1.0),
    ],
)
def test_default_gate_linear_piece(conf, expected):
    assert default_gate(ds(50.0, conf)) == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize(
    "conf,expected",
    [
        (1.0, 0.5),                     # 满置信但降级 → 折半
        ((GATE_LOW + GATE_HIGH) / 2, 0.25),
        (0.9, 0.5),
        (0.1, 0.0),                     # 归零后再折半仍是 0
    ],
)
def test_degraded_dim_gets_half_weight(conf, expected):
    assert default_gate(ds(50.0, conf, degraded=True)) == pytest.approx(expected, abs=1e-9)


def test_gate_clamps_out_of_range_confidence():
    assert default_gate(ds(50.0, -5.0)) == 0.0
    assert default_gate(ds(50.0, 5.0)) == 1.0
    assert default_gate(ds(50.0, float("nan"))) == 0.0


# ---------- 基本聚合 ----------


def test_two_dims_no_redistribution():
    res = aggregate({"a": ds(80.0), "b": ds(40.0)}, {"a": 0.5, "b": 0.5})
    assert res.score == pytest.approx(60.0)
    assert res.effective_weights == {"a": pytest.approx(0.5), "b": pytest.approx(0.5)}
    assert res.redistributed == {"a": pytest.approx(0.0), "b": pytest.approx(0.0)}
    assert any("未发生权重重分配" in n for n in res.notes)


def test_unnormalized_weights_are_normalized():
    res = aggregate({"a": ds(80.0), "b": ds(40.0)}, {"a": 2.0, "b": 2.0})
    assert res.score == pytest.approx(60.0)
    assert res.effective_weights["a"] == pytest.approx(0.5)


def test_dim_without_weight_is_ignored():
    res = aggregate(
        {"a": ds(80.0), "b": ds(40.0), "c": ds(100.0)},
        {"a": 0.5, "b": 0.5},
    )
    assert res.effective_weights["c"] == pytest.approx(0.0)
    assert res.score == pytest.approx(60.0)


# ---------- 重分配：手算可复核 ----------


def test_redistribution_hand_computed():
    """w a=.6 b=.3 c=.1；conf a=1.0 b=0.5 c=0.2（a 满门、b 线性段、c 归零）。

    gate: a=1.0, b=(0.5-0.4)/0.35=2/7≈0.285714, c=0
    raw : a=.6, b=.3*2/7=0.0857143, c=0 → 合计 0.6857143，缺口 0.3142857
    缺口按 a:b 原权重 0.6:0.3 = 2:1 分摊 → a +0.2095238, b +0.1047619
    eff : a=0.8095238, b=0.1904762, c=0
    score=0.8095238*80 + 0.1904762*60 = 76.190476
    """
    dims = {"a": ds(80.0, 1.0), "b": ds(60.0, 0.5), "c": ds(30.0, 0.2)}
    res = aggregate(dims, {"a": 0.6, "b": 0.3, "c": 0.1})

    assert res.effective_weights["a"] == pytest.approx(0.8095238095, abs=1e-9)
    assert res.effective_weights["b"] == pytest.approx(0.1904761905, abs=1e-9)
    assert res.effective_weights["c"] == pytest.approx(0.0, abs=1e-12)
    assert res.score == pytest.approx(76.1904761905, abs=1e-9)
    assert res.redistributed["a"] == pytest.approx(0.2095238095, abs=1e-9)
    assert res.redistributed["b"] == pytest.approx(0.1047619048, abs=1e-9)
    assert res.redistributed["c"] == pytest.approx(0.0, abs=1e-12)


def test_redistribution_follows_original_weight_ratio():
    dims = {"a": ds(80.0, 1.0), "b": ds(60.0, 0.5), "c": ds(30.0, 0.0)}
    res = aggregate(dims, {"a": 0.6, "b": 0.3, "c": 0.1})
    ratio_got = res.redistributed["a"] / res.redistributed["b"]
    assert ratio_got == pytest.approx(0.6 / 0.3, rel=1e-9)


def test_gated_out_dim_never_revives():
    dims = {"a": ds(80.0, 1.0), "b": ds(0.0, 0.0)}
    res = aggregate(dims, {"a": 0.5, "b": 0.5})
    assert res.effective_weights["b"] == pytest.approx(0.0, abs=1e-12)
    assert res.score == pytest.approx(80.0)


@pytest.mark.parametrize(
    "weights,confs",
    [
        ({"a": 1.0}, [1.0]),
        ({"a": 0.5, "b": 0.5}, [1.0, 1.0]),
        ({"a": 0.5, "b": 0.5}, [1.0, 0.0]),
        ({"a": 0.5, "b": 0.5}, [0.5, 0.5]),
        ({"a": 0.6, "b": 0.3, "c": 0.1}, [1.0, 0.8, 0.75]),
        ({"a": 0.6, "b": 0.3, "c": 0.1}, [1.0, 0.5, 0.45]),
        ({"a": 0.4, "b": 0.3, "c": 0.2, "d": 0.1}, [0.9, 0.1, 0.0, 0.0]),
        ({"a": 0.4, "b": 0.3, "c": 0.2, "d": 0.1}, [0.41, 0.74, 1.0, 0.4]),
        (dict(DEFAULT_WEIGHTS), [0.95, 0.8, 0.7, 0.6, 0.2, 0.1]),
    ],
)
def test_effective_weights_always_sum_to_one(weights, confs):
    dims = {k: ds(50.0 + 5.0 * i, c) for i, (k, c) in enumerate(zip(weights, confs))}
    res = aggregate(dims, weights)
    assert sum(res.effective_weights.values()) == pytest.approx(1.0, abs=1e-12)
    assert all(v >= 0.0 for v in res.effective_weights.values())
    assert min(d.value for d in dims.values()) <= res.score <= max(d.value for d in dims.values())


def test_weights_sum_to_one_random_property():
    rng = random.Random(20261006)
    for _ in range(300):
        n = rng.randint(2, 6)
        keys = [f"d{i}" for i in range(n)]
        raw_w = [rng.random() for _ in keys]
        if sum(raw_w) <= 0:
            continue
        confs = [rng.random() for _ in keys]
        if all(c < GATE_LOW for c in confs):
            continue  # 全不可用，走 Unavailable 分支（另有测试）
        dims = {k: ds(rng.uniform(0, 100), c) for k, c in zip(keys, confs)}
        res = aggregate(dims, dict(zip(keys, raw_w)))
        assert sum(res.effective_weights.values()) == pytest.approx(1.0, abs=1e-12)
        assert all(v >= -1e-12 for v in res.effective_weights.values())
        assert 0.0 <= res.score <= 100.0


# ---------- 不可用必须抛错，绝不给中间分 ----------


def test_all_dims_unavailable_raises():
    dims = {"a": ds(80.0, 0.1), "b": ds(40.0, 0.0)}
    with pytest.raises(Unavailable) as ei:
        aggregate(dims, {"a": 0.5, "b": 0.5})
    assert ei.value.provider == "aggregate"
    assert ei.value.reason == "所有维度不可用"


def test_zero_weights_raises_instead_of_dividing_by_zero():
    dims = {"a": ds(80.0, 1.0), "b": ds(40.0, 1.0)}
    with pytest.raises(Unavailable):
        aggregate(dims, {"a": 0.0, "b": 0.0})


def test_empty_dims_raises():
    with pytest.raises(Unavailable):
        aggregate({}, {})


def test_custom_gate_all_zero_raises():
    dims = {"a": ds(80.0, 1.0), "b": ds(40.0, 1.0)}
    with pytest.raises(Unavailable):
        aggregate(dims, {"a": 0.5, "b": 0.5}, gate=lambda d: 0.0)


def test_custom_gate_is_respected():
    dims = {"a": ds(80.0, 0.0), "b": ds(40.0, 0.0)}
    res = aggregate(dims, {"a": 0.5, "b": 0.5}, gate=lambda d: 1.0)
    assert res.score == pytest.approx(60.0)
    assert res.redistributed == {"a": pytest.approx(0.0), "b": pytest.approx(0.0)}


# ---------- 反事实 ----------


def test_counterfactuals_hand_computed():
    dims = {"a": ds(80.0, 1.0), "b": ds(60.0, 0.5), "c": ds(30.0, 0.2)}
    res = aggregate(dims, {"a": 0.6, "b": 0.3, "c": 0.1})
    # 剔除 a → 只剩 b（权重全给 b）→ 60
    assert res.counterfactuals["a"] == pytest.approx(60.0, abs=1e-9)
    # 剔除 b → 只剩 a → 80
    assert res.counterfactuals["b"] == pytest.approx(80.0, abs=1e-9)
    # 剔除 c（本来就被门控掉）→ 与总分一致
    assert res.counterfactuals["c"] == pytest.approx(res.score, abs=1e-9)
    assert set(res.counterfactuals) == set(dims)


def test_counterfactual_of_only_dim_is_nan():
    res = aggregate({"a": ds(70.0, 1.0)}, {"a": 1.0})
    assert res.score == pytest.approx(70.0)
    assert math.isnan(res.counterfactuals["a"])


# ---------- 红线：生理维度关闭不得压低总分 ----------


def _six_dims() -> dict[str, DimensionScore]:
    return {
        "technical": ds(80.0, 0.95),
        "communication": ds(70.0, 0.90),
        "completeness": ds(75.0, 0.90),
        "problem_solving": ds(78.0, 0.95),
        "teamwork": ds(65.0, 0.80),
        "leadership": ds(60.0, 0.80),
    }


@pytest.mark.parametrize("physio_value", [0.0, 25.0, 100.0])
@pytest.mark.parametrize("physio_conf", [0.0, 0.1, 0.39])
def test_disabling_low_reliability_physio_does_not_change_score(physio_value, physio_conf):
    """红线：生理维度被门控（gate=0）时，开与关的总分必须完全一致。"""
    closed = aggregate(_six_dims(), dict(DEFAULT_WEIGHTS))

    dims = _six_dims()
    dims[PHYSIO_DIM] = ds(physio_value, physio_conf)
    weights = {k: v * 0.92 for k, v in DEFAULT_WEIGHTS.items()}
    weights[PHYSIO_DIM] = 0.08
    opened = aggregate(dims, weights)

    assert opened.score == pytest.approx(closed.score, abs=1e-12)
    assert opened.effective_weights[PHYSIO_DIM] == pytest.approx(0.0, abs=1e-12)
    # 反事实报告：关掉它，分数不变
    assert opened.counterfactuals[PHYSIO_DIM] == pytest.approx(closed.score, abs=1e-12)


def test_high_reliability_physio_does_participate():
    """对照：可靠性达标时生理维度确实计入（否则上一条测试就是恒真式）。"""
    closed = aggregate(_six_dims(), dict(DEFAULT_WEIGHTS))
    dims = _six_dims()
    dims[PHYSIO_DIM] = ds(20.0, 0.9)
    weights = {k: v * 0.92 for k, v in DEFAULT_WEIGHTS.items()}
    weights[PHYSIO_DIM] = 0.08
    opened = aggregate(dims, weights)
    assert opened.score < closed.score          # 低分维度计入 → 拉低
    assert opened.effective_weights[PHYSIO_DIM] == pytest.approx(0.08, abs=1e-12)


def test_no_systematic_penalty_across_reliability_sweep():
    """可靠性从 0 扫到 1：总分单调地"从不含该维度"过渡到"完整含该维度"，
    且**任何一点都不会低于**完全关闭时的分数（该维度取低值时才会低于，这是真实的降分）。
    """
    closed = aggregate(_six_dims(), dict(DEFAULT_WEIGHTS))
    weights7 = {k: v * 0.92 for k, v in DEFAULT_WEIGHTS.items()}
    weights7[PHYSIO_DIM] = 0.08

    physio_high, prev = 100.0, None
    for conf in [0.0, 0.2, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0]:
        dims = _six_dims()
        dims[PHYSIO_DIM] = ds(physio_high, conf)
        s = aggregate(dims, weights7).score
        assert s >= closed.score - 1e-12, f"conf={conf} 时总分被压低了"
        if prev is not None:
            assert s >= prev - 1e-12  # 高价值维度越可信，总分越高（单调）
        prev = s
    assert prev == pytest.approx(0.92 * closed.score + 0.08 * 100.0, abs=1e-9)


# ---------- 降级方向 ----------


def test_degraded_high_value_dim_lowers_score():
    base = aggregate({"a": ds(90.0, 1.0), "b": ds(50.0, 1.0)}, {"a": 0.5, "b": 0.5})
    deg = aggregate(
        {"a": ds(90.0, 1.0, degraded=True), "b": ds(50.0, 1.0)},
        {"a": 0.5, "b": 0.5},
    )
    assert deg.score < base.score
    assert deg.score == pytest.approx(0.375 * 90.0 + 0.625 * 50.0, abs=1e-9)
    assert deg.effective_weights["a"] == pytest.approx(0.375, abs=1e-9)
    assert deg.effective_weights["b"] == pytest.approx(0.625, abs=1e-9)
    assert any("降级" in n for n in deg.notes)


def test_degraded_low_value_dim_raises_score():
    """降级是降权不是扣分：被降级的维度分数低时，总分反而上升。"""
    base = aggregate({"a": ds(20.0, 1.0), "b": ds(80.0, 1.0)}, {"a": 0.5, "b": 0.5})
    deg = aggregate(
        {"a": ds(20.0, 1.0, degraded=True), "b": ds(80.0, 1.0)},
        {"a": 0.5, "b": 0.5},
    )
    assert deg.score > base.score
    assert deg.score == pytest.approx(0.375 * 20.0 + 0.625 * 80.0, abs=1e-9)


# ---------- 报告归因 ----------


def test_notes_explain_redistribution():
    dims = {"a": ds(80.0, 0.95), "b": ds(60.0, 0.30), "c": ds(30.0, 0.10)}
    res = aggregate(dims, {"a": 0.5, "b": 0.3, "c": 0.2})
    assert any("重分配" in n for n in res.notes)
    assert any("b" in n and "可靠性不足" in n for n in res.notes)
    assert any("c" in n and "未计入总分" in n for n in res.notes)
    assert sum(res.effective_weights.values()) == pytest.approx(1.0, abs=1e-12)


def test_degraded_flag_is_noted_even_when_gate_is_full():
    """自定义门控可能忽略 degraded，此时仍要在 notes 里如实标注降级。"""
    dims = {"a": ds(80.0, 1.0, degraded=True, provider="qwen"), "b": ds(40.0, 1.0)}
    res = aggregate(dims, {"a": 0.5, "b": 0.5}, gate=lambda d: 1.0)
    assert res.score == pytest.approx(60.0)
    assert any("降级" in n and "qwen" in n for n in res.notes)


def test_default_weights_sum_to_one_and_cover_six_dims():
    assert sum(DEFAULT_WEIGHTS.values()) == pytest.approx(1.0, abs=1e-12)
    assert set(DEFAULT_WEIGHTS) == {
        "technical", "communication", "completeness",
        "problem_solving", "teamwork", "leadership",
    }
