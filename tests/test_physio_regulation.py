"""regulation.py 测试：可靠性公式、gate 三段边界、标定映射、权重重分配、各关闭分支。"""

from __future__ import annotations

import pytest

from ruipin.ports import DIMENSIONS, PHYSIO_DIM, InterviewEvent
from ruipin.physio.metrics import MIN_VALID_EVENTS
from ruipin.physio.regulation import (
    COMPONENT_WEIGHTS,
    DEFAULT_WEIGHTS,
    GATE_CEIL,
    GATE_FLOOR,
    PLACEHOLDER_CALIBRATION,
    W_MAX,
    PhysioCalibration,
    calibrate,
    evaluate_physio,
    gate,
    redistribute,
    reliability,
)


def mk_events(n, pre=70.0, ratio=0.20, t50=12.0, slope=0.0):
    """n 个等 T50 事件；slope≠0 时 ratio 随轮次线性变化（模拟适应/敏感化）。"""
    return [
        InterviewEvent(
            turn_index=i,
            pre_bpm=pre,
            peak_bpm=pre + pre * (ratio + slope * i),
            t50_s=t50,
        )
        for i in range(n)
    ]


N = 10  # ≥8 的有效事件数


# ---------------------------------------------------------------- reliability


def test_reliability_formula_exact_value():
    """R = 0.40·(10/14) + 0.35·0.6 + 0.15·(1−0.2) + 0.10·0.8"""
    expected = 0.40 * (10 / 14) + 0.35 * 0.6 + 0.15 * 0.8 + 0.10 * 0.8
    r = reliability(
        n_valid=10,
        n_expected=14.0,
        median_snr=0.6,
        reject_ratio=0.2,
        algo_agreement=0.8,
    )
    assert r == pytest.approx(expected)
    assert r == pytest.approx(0.6957142857, abs=1e-9)
    assert 0.0 <= r <= 1.0


@pytest.mark.parametrize(
    "n_valid,n_expected,snr,reject,agree,expected",
    [
        (0, 10.0, 0.0, 1.0, 0.0, 0.0),  # 全无信号
        (10, 10.0, 1.0, 0.0, 1.0, 1.0),  # 全理想
        (10, 0.0, 1.0, 0.0, 1.0, 0.35 + 0.15 + 0.10),  # n_expected=0 → 覆盖率按 0 计（不猜）
        (20, 10.0, 0.5, 0.0, 0.0, 0.4 + 0.175 + 0.15),  # 覆盖率夹到 1
        (10, 10.0, 0.5, 0.5, 0.0, 0.4 + 0.175 + 0.075),
    ],
)
def test_reliability_bounds_and_clamps(n_valid, n_expected, snr, reject, agree, expected):
    r = reliability(
        n_valid=n_valid,
        n_expected=n_expected,
        median_snr=snr,
        reject_ratio=reject,
        algo_agreement=agree,
    )
    assert r == pytest.approx(expected)
    assert 0.0 <= r <= 1.0


def test_reliability_clamps_dirty_inputs():
    r = reliability(
        n_valid=10,
        n_expected=10.0,
        median_snr=5.0,  # 越界输入
        reject_ratio=-2.0,
        algo_agreement=3.0,
    )
    assert r == pytest.approx(1.0)  # 不允许被顶出 [0,1]


def test_reliability_monotone_in_each_input():
    base = dict(n_valid=10, n_expected=20.0, median_snr=0.5, reject_ratio=0.3, algo_agreement=0.5)
    r0 = reliability(**base)
    assert reliability(**{**base, "n_valid": 16}) > r0
    assert reliability(**{**base, "median_snr": 0.8}) > r0
    assert reliability(**{**base, "reject_ratio": 0.1}) > r0
    assert reliability(**{**base, "algo_agreement": 0.9}) > r0


# ---------------------------------------------------------------- gate


@pytest.mark.parametrize(
    "r,expected",
    [
        (-1.0, 0.0),
        (0.0, 0.0),
        (0.39, 0.0),  # 下界外侧
        (GATE_FLOOR, 0.0),  # 0.40：恰好为 0
        (0.41, pytest.approx((0.41 - 0.40) / 0.35)),
        (0.575, 0.5),  # 线性区中点
        (0.74, pytest.approx(0.34 / 0.35)),  # 上界内侧
        (GATE_CEIL, 1.0),  # 0.75：恰好为 1
        (0.80, 1.0),
        (1.0, 1.0),
        (2.0, 1.0),
    ],
)
def test_gate_three_regions(r, expected):
    assert gate(r) == pytest.approx(expected)
    assert 0.0 <= gate(r) <= 1.0


def test_gate_constants():
    assert (GATE_FLOOR, GATE_CEIL) == (0.40, 0.75)
    assert W_MAX == 0.08


def test_gate_continuous_at_boundaries():
    assert gate(0.40) == pytest.approx(0.0)
    assert gate(0.75) == pytest.approx(1.0)
    assert gate(0.74999) < 1.0
    assert gate(0.75001) == pytest.approx(1.0)


# ---------------------------------------------------------------- calibrate


LIN = {0.0: 0.0, 0.5: 10.0, 1.0: 20.0}


@pytest.mark.parametrize(
    "value,expected",
    [
        (-5.0, 0.0),  # 表外下侧 → 夹到最低分位
        (0.0, 0.0),
        (5.0, 25.0),  # 线性插值：p = 0.25
        (10.0, 50.0),
        (15.0, 75.0),
        (20.0, 100.0),
        (30.0, 100.0),  # 表外上侧 → 夹到最高分位
    ],
)
def test_calibrate_linear_interpolation(value, expected):
    assert calibrate(value, LIN) == pytest.approx(expected)


@pytest.mark.parametrize("value,expected", [(0.0, 100.0), (5.0, 75.0), (10.0, 50.0), (20.0, 0.0)])
def test_calibrate_lower_is_better(value, expected):
    """T50 越小越高分 → higher_is_better=False 时映射反向。"""
    assert calibrate(value, LIN, higher_is_better=False) == pytest.approx(expected)


def test_calibrate_placeholder_t50_table():
    """占位标定表：中位数 16 s → 50 分；6 s → 95 分；34 s → 5 分。"""
    tbl = PLACEHOLDER_CALIBRATION.t50
    assert calibrate(16.0, tbl, higher_is_better=False) == pytest.approx(50.0)
    assert calibrate(6.0, tbl, higher_is_better=False) == pytest.approx(95.0)
    assert calibrate(34.0, tbl, higher_is_better=False) == pytest.approx(5.0)
    # 11 s(p=.25) → 16 s(p=.50) 之间插值：13.5 s → p=0.375 → 62.5
    assert calibrate(13.5, tbl, higher_is_better=False) == pytest.approx(62.5)


def test_calibrate_output_always_in_range():
    for v in (-100.0, 0.0, 16.0, 1e6):
        s = calibrate(v, PLACEHOLDER_CALIBRATION.t50, higher_is_better=False)
        assert 0.0 <= s <= 100.0


def test_calibrate_rejects_bad_tables():
    with pytest.raises(ValueError):
        calibrate(1.0, {0.5: 10.0})  # 少于 2 个分位点
    with pytest.raises(ValueError):
        calibrate(1.0, {0.9: 1.0, 0.1: 2.0})  # 分位概率不随观测值递增


# ---------------------------------------------------------------- redistribute


def test_redistribute_full_freed_sum_is_one():
    got = redistribute(DEFAULT_WEIGHTS, PHYSIO_DIM, W_MAX)
    assert PHYSIO_DIM not in got
    assert set(got) == set(DIMENSIONS)
    assert sum(got.values()) == pytest.approx(1.0)
    for v in got.values():
        assert v == pytest.approx(1.0 / len(DIMENSIONS))


@pytest.mark.parametrize("freed", [0.0, 0.01, 0.03, 0.08])
def test_redistribute_conserves_total(freed):
    got = redistribute(DEFAULT_WEIGHTS, PHYSIO_DIM, freed)
    assert sum(got.values()) == pytest.approx((1.0 - W_MAX) + freed)
    assert sum(got.values()) + (W_MAX - freed) == pytest.approx(1.0)


def test_redistribute_preserves_relative_proportions():
    weights = {"a": 0.10, "b": 0.30, "c": 0.52, PHYSIO_DIM: 0.08}
    got = redistribute(weights, PHYSIO_DIM, 0.08)
    assert got["b"] / got["a"] == pytest.approx(3.0)
    assert got["c"] / got["a"] == pytest.approx(5.2)
    assert sum(got.values()) == pytest.approx(1.0)


def test_redistribute_accepts_iterable_and_negative_freed():
    weights = {"a": 0.4, "b": 0.4, PHYSIO_DIM: 0.08}
    got = redistribute(weights, [PHYSIO_DIM], 0.08)
    assert sum(got.values()) == pytest.approx(0.88)
    assert redistribute(weights, PHYSIO_DIM, -1.0) == pytest.approx({"a": 0.4, "b": 0.4})


def test_redistribute_edge_cases():
    assert redistribute({}, PHYSIO_DIM, 0.08) == {}
    assert redistribute({PHYSIO_DIM: 0.08}, PHYSIO_DIM, 0.08) == {}


# ---------------------------------------------------------------- evaluate_physio: 关闭分支


def _unavailable_cases():
    return {
        "未授权": dict(authorized=False),
        "基线缺失": dict(baseline_bpm=None),
        "有效事件不足": dict(events=mk_events(MIN_VALID_EVENTS - 1)),
        "可靠性不足": dict(n_expected=20.0, median_snr=0.05, reject_ratio=0.9, algo_agreement=0.0),
        "心律不规则": dict(ibis=[0.4, 1.2] * 20),
    }


@pytest.mark.parametrize("case", sorted(_unavailable_cases()))
def test_unavailable_branches(case):
    kwargs = dict(
        events=mk_events(N),
        baseline_bpm=68.0,
        n_expected=12.0,
        median_snr=0.7,
        reject_ratio=0.1,
        algo_agreement=0.9,
    )
    kwargs.update(_unavailable_cases()[case])
    res = evaluate_physio(**kwargs)

    assert res.available is False
    assert res.reason, "关闭必须给出明确原因"
    assert res.score is None  # 绝不产出兜底分数
    assert res.weight_applied == pytest.approx(0.0)
    assert sum(res.redistributed_to.values()) == pytest.approx(1.0)  # 权重全额归还
    assert PHYSIO_DIM not in res.redistributed_to
    assert res.t50_median_s is None


def test_reason_text_is_specific():
    """不同关闭原因给出不同的、可展示给候选人的明确文案。"""
    base = dict(n_expected=12.0, median_snr=0.7, reject_ratio=0.1, algo_agreement=0.9)
    r_no_auth = evaluate_physio(events=mk_events(N), baseline_bpm=68.0, authorized=False, **base)
    r_no_base = evaluate_physio(events=mk_events(N), baseline_bpm=None, **base)
    r_few = evaluate_physio(events=mk_events(7), baseline_bpm=68.0, **base)
    r_low_r = evaluate_physio(
        events=mk_events(N),
        baseline_bpm=68.0,
        **{**base, "n_expected": 20.0, "median_snr": 0.05, "reject_ratio": 0.9, "algo_agreement": 0.0},
    )
    r_arr = evaluate_physio(events=mk_events(N), baseline_bpm=68.0, ibis=[0.4, 1.2] * 20, **base)

    assert "授权" in r_no_auth.reason
    assert "基线" in r_no_base.reason
    assert "8" in r_few.reason
    assert "可靠性" in r_low_r.reason
    assert "心律" in r_arr.reason
    reasons = {r_no_auth.reason, r_no_base.reason, r_few.reason, r_low_r.reason, r_arr.reason}
    assert len(reasons) == 5


def test_unauthorized_ignores_signal_quality():
    """未授权时无论信号多好都不计入（授权是前置条件）。"""
    res = evaluate_physio(
        events=mk_events(N),
        baseline_bpm=68.0,
        n_expected=12.0,
        median_snr=1.0,
        reject_ratio=0.0,
        algo_agreement=1.0,
        authorized=False,
    )
    assert res.available is False
    assert res.reliability == pytest.approx(0.0)


# ---------------------------------------------------------------- evaluate_physio: 正常链路


def test_available_high_reliability_full_weight():
    res = evaluate_physio(
        events=mk_events(N),
        baseline_bpm=68.0,
        n_expected=12.0,
        median_snr=0.7,
        reject_ratio=0.1,
        algo_agreement=0.9,
    )
    assert res.available is True
    assert res.reason is None
    r_expected = 0.40 * (10 / 12) + 0.35 * 0.7 + 0.15 * 0.9 + 0.10 * 0.9
    assert res.reliability == pytest.approx(r_expected)  # ≈0.8033 ≥ 0.75
    assert res.weight_applied == pytest.approx(W_MAX)  # gate = 1
    assert sum(res.redistributed_to.values()) + res.weight_applied == pytest.approx(1.0)
    assert res.n_events == N and res.n_valid == N
    assert res.baseline_bpm == pytest.approx(68.0)

    # 分量与合成（占位标定表手算）：T50=12→70；β=0→37.5；σ≈0→95
    assert res.components["S_recovery"] == pytest.approx(70.0)
    assert res.components["S_habituation"] == pytest.approx(37.5)
    assert res.components["S_consistency"] == pytest.approx(95.0)
    assert res.score == pytest.approx(0.6 * 70.0 + 0.25 * 37.5 + 0.15 * 95.0)
    assert res.score == pytest.approx(65.625)
    assert 0.0 <= res.score <= 100.0
    assert res.t50_median_s == pytest.approx(12.0)
    assert res.habituation_slope == pytest.approx(0.0, abs=1e-12)
    assert res.reactivity_ratio_mean == pytest.approx(0.20)


@pytest.mark.parametrize(
    "n_valid,n_expected,snr,reject,agree,exp_gate",
    [
        (10, 20.0, 0.4, 0.6, 0.5, (0.45 - 0.40) / 0.35),  # R = 0.45 → 部分权重
        (10, 20.0, 0.2, 0.6, 0.5, None),  # R = 0.38 < 0.40 → 关闭
        (10, 10.0, 0.8, 0.0, 1.0, 1.0),  # R ≈ 0.93 → 全额
    ],
)
def test_weight_scales_with_reliability(n_valid, n_expected, snr, reject, agree, exp_gate):
    res = evaluate_physio(
        events=mk_events(n_valid),
        baseline_bpm=68.0,
        n_expected=n_expected,
        median_snr=snr,
        reject_ratio=reject,
        algo_agreement=agree,
    )
    if exp_gate is None:
        assert res.available is False
        assert res.weight_applied == pytest.approx(0.0)
        assert res.score is None
    else:
        assert res.available is True
        assert res.weight_applied == pytest.approx(W_MAX * exp_gate, abs=1e-9)
        assert res.weight_applied == pytest.approx(W_MAX * gate(res.reliability))
    assert sum(res.redistributed_to.values()) + res.weight_applied == pytest.approx(1.0)


def test_partial_weight_returns_remainder_proportionally():
    res = evaluate_physio(
        events=mk_events(N),
        baseline_bpm=68.0,
        n_expected=20.0,
        median_snr=0.4,
        reject_ratio=0.6,
        algo_agreement=0.5,
    )
    assert res.available is True
    assert 0.0 < res.weight_applied < W_MAX  # 线性过渡区
    freed = W_MAX - res.weight_applied
    got = res.redistributed_to
    assert sum(got.values()) == pytest.approx(1.0 - res.weight_applied)
    assert got["technical"] / got["leadership"] == pytest.approx(1.0)  # 比例不变


def test_custom_weights_are_honored():
    weights = {"technical": 0.30, "communication": 0.30, "completeness": 0.32, PHYSIO_DIM: W_MAX}
    # 可靠性足够 → 权重用满，无额度可还：其他维度保持原值，合计 0.92 + 0.08 = 1
    full = evaluate_physio(
        events=mk_events(N),
        baseline_bpm=68.0,
        n_expected=12.0,
        median_snr=0.7,
        reject_ratio=0.1,
        algo_agreement=0.9,
        weights=weights,
    )
    assert set(full.redistributed_to) == {"technical", "communication", "completeness"}
    assert full.redistributed_to["technical"] == pytest.approx(0.30)
    assert sum(full.redistributed_to.values()) + full.weight_applied == pytest.approx(1.0)

    # 可靠性不足 → 部分额度归还，按原比例放大到 0.92 + freed
    partial = evaluate_physio(
        events=mk_events(N),
        baseline_bpm=68.0,
        n_expected=20.0,
        median_snr=0.4,
        reject_ratio=0.6,
        algo_agreement=0.5,
        weights=weights,
    )
    freed = W_MAX - partial.weight_applied
    assert partial.redistributed_to["technical"] == pytest.approx(0.30 * (0.92 + freed) / 0.92)
    assert sum(partial.redistributed_to.values()) == pytest.approx(0.92 + freed)


def test_statistics_missing_even_with_enough_events():
    """有效事件 ≥8 但恢复统计量缺失 → 仍不计入，不猜中间分。"""
    events = [InterviewEvent(i, 70.0, 84.0, None) for i in range(N)]
    res = evaluate_physio(
        events=events,
        baseline_bpm=68.0,
        n_expected=12.0,
        median_snr=0.7,
        reject_ratio=0.1,
        algo_agreement=0.9,
    )
    assert res.available is False
    assert res.n_valid == N
    assert res.reason and "统计量" in res.reason
    assert res.score is None
    assert res.weight_applied == pytest.approx(0.0)
    assert sum(res.redistributed_to.values()) == pytest.approx(1.0)


def test_habituation_improves_score():
    """适应良好（β<0）比敏感化（β>0）得分更高，且只有习惯化分量变化。"""
    common = dict(baseline_bpm=68.0, n_expected=12.0, median_snr=0.7, reject_ratio=0.1, algo_agreement=0.9)
    adapting = mk_events(N, ratio=0.25, slope=-0.01)
    sensitizing = mk_events(N, ratio=0.25, slope=0.01)
    a = evaluate_physio(events=adapting, **common)
    s = evaluate_physio(events=sensitizing, **common)

    assert a.habituation_slope < 0 < s.habituation_slope
    assert a.components["S_habituation"] > s.components["S_habituation"]
    assert a.components["S_recovery"] == pytest.approx(s.components["S_recovery"])
    assert a.score > s.score


def test_component_weights_sum_to_one_and_exclude_vagal():
    """v1 分量不含 S_vagal（vmHRV 属 v2，§1.5）。"""
    assert COMPONENT_WEIGHTS == pytest.approx(
        {"S_recovery": 0.60, "S_habituation": 0.25, "S_consistency": 0.15}
    )
    assert sum(COMPONENT_WEIGHTS.values()) == pytest.approx(1.0)
    assert "S_vagal" not in COMPONENT_WEIGHTS


def test_custom_calibration_is_used():
    cal = PhysioCalibration(
        t50={0.0: 0.0, 1.0: 60.0},
        habituation={0.0: -1.0, 1.0: 1.0},
        consistency={0.0: -1.0, 1.0: 1.0},
    )
    res = evaluate_physio(
        events=mk_events(N),
        baseline_bpm=68.0,
        n_expected=12.0,
        median_snr=0.7,
        reject_ratio=0.1,
        algo_agreement=0.9,
        calibration=cal,
    )
    # T50=12 → p=0.2 → 80；β=0 → p=0.5 → 50；σ≈0 → p=0.5 → 50
    assert res.components["S_recovery"] == pytest.approx(80.0)
    assert res.components["S_habituation"] == pytest.approx(50.0)
    assert res.components["S_consistency"] == pytest.approx(50.0)
    assert res.score == pytest.approx(0.6 * 80 + 0.25 * 50 + 0.15 * 50)


def test_default_weights_include_physio_at_cap():
    assert DEFAULT_WEIGHTS[PHYSIO_DIM] == pytest.approx(W_MAX)
    assert sum(DEFAULT_WEIGHTS.values()) == pytest.approx(1.0)
