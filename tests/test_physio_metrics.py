"""metrics.py 测试：合成 HR 序列验证 T50 恢复、n<8 硬约束、ratio 而非 ΔHR、心律检测。"""

from __future__ import annotations

import math

import pytest

from ruipin.ports import InterviewEvent
from ruipin.physio.metrics import (
    ARRHYTHMIA_CV_THRESHOLD,
    CENSORED_T50_S,
    MIN_DELTA_BPM,
    MIN_IBI_SAMPLES,
    MIN_VALID_EVENTS,
    censored_share,
    consistency_sigma,
    detect_arrhythmia,
    estimate_lambda,
    estimate_t50,
    fit_recovery,
    habituation_slope,
    reactivity_ratio_mean,
    t50_median,
    valid_count,
)


# ---------------------------------------------------------------- 工具


def mk_event(i, pre, ratio, t50=12.0, censored=False, valid=True):
    """按 (静息心率, 归一化反应幅度) 构造事件，ΔHR 由二者推出（A4）。"""
    return InterviewEvent(
        turn_index=i,
        pre_bpm=pre,
        peak_bpm=pre + pre * ratio,
        t50_s=t50,
        censored=censored,
        valid=valid,
    )


def synth_hr(pre=70.0, delta=20.0, lam=0.05, window_s=45.0, dt=0.5):
    """合成指数衰减序列 HR(t) = pre + Δ·exp(−λ t)，t 自峰值起算。"""
    n = int(round(window_s / dt)) + 1
    times = [i * dt for i in range(n)]
    hr = [pre + delta * math.exp(-lam * t) for t in times]
    return times, hr


def mean(xs):
    return sum(xs) / len(xs)


def ols_slope(xs, ys):
    mx, my = mean(xs), mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx


# ---------------------------------------------------------------- 合成序列恢复 T50


@pytest.mark.parametrize("lam", [0.03, math.log(2) / 12.0, 0.12])
def test_estimate_t50_recovers_known_lambda(lam):
    """已知 λ 的指数衰减序列 → 恢复出的 T50 ≈ ln2/λ。"""
    times, hr = synth_hr(pre=70.0, delta=20.0, lam=lam)
    expected = math.log(2) / lam
    assert estimate_t50(times, hr, pre=70.0) == pytest.approx(expected, rel=1e-2)
    assert estimate_lambda(times, hr, pre=70.0) == pytest.approx(lam, rel=1e-2)


def test_estimate_t50_independent_of_amplitude_and_baseline():
    """T50 是时间量：ΔHR 与静息水平不同，T50 不变（这正是它与幅值类指标的区别）。"""
    lam = math.log(2) / 10.0
    expected = math.log(2) / lam
    for pre, delta in [(55.0, 8.0), (70.0, 20.0), (90.0, 35.0)]:
        times, hr = synth_hr(pre=pre, delta=delta, lam=lam)
        assert estimate_t50(times, hr, pre=pre) == pytest.approx(expected, rel=1e-2)


def test_fit_recovery_censored_at_window():
    """45 s 内未回落至 50% → 右删失，T50 取 45 s 下限参与。"""
    times, hr = synth_hr(pre=70.0, delta=20.0, lam=0.01)  # 真实 T50 ≈ 69 s
    fit = fit_recovery(times, hr, pre=70.0)
    assert fit.lam == pytest.approx(0.01, rel=1e-2)
    assert fit.censored is True
    assert fit.t50_s == pytest.approx(CENSORED_T50_S)
    assert fit.t50_s < math.log(2) / fit.lam  # 是下限，不是估计值


def test_fit_recovery_abstains_when_not_converging():
    """没有回落（λ≤0）→ 弃权；点数不足 → 弃权；永远返回 None 而非猜一个数。"""
    times = [i * 0.5 for i in range(90)]
    rising = [70.0 + 20.0 * math.exp(0.02 * t) for t in times]  # 反而上升
    assert fit_recovery(times, rising, pre=70.0).lam is None
    assert estimate_t50(times, rising, pre=70.0) is None

    assert fit_recovery([0.0, 0.5], [90.0, 88.0], pre=70.0).lam is None  # 点太少

    below = [65.0] * 90  # 已低于基线，对指数模型无信息
    assert fit_recovery(times, below, pre=70.0).lam is None


def test_fit_recovery_skips_none_samples_and_degenerate_timestamps():
    times = [0.0, 0.5, 1.0, 1.5, 2.0, None]
    hr = [90.0, 88.0, 86.0, 85.0, 84.0, 83.0]
    assert fit_recovery(times, hr, pre=70.0).lam is not None  # None 采样点被跳过

    same_t = [1.0] * 10  # 时间戳无变异 → 无法回归
    assert fit_recovery(same_t, [90.0 - i for i in range(10)], pre=70.0).lam is None


def test_fit_recovery_ignores_points_outside_window():
    """观测窗外（被下一题截断）的采样点不参与拟合。"""
    lam = math.log(2) / 12.0
    times, hr = synth_hr(pre=70.0, delta=20.0, lam=lam, window_s=90.0, dt=0.5)
    fit = fit_recovery(times, hr, pre=70.0, window_s=45.0)
    assert fit.t50_s == pytest.approx(12.0, rel=1e-2)


# ---------------------------------------------------------------- n_valid < 8 硬约束


@pytest.mark.parametrize("n", [0, 1, 5, 7])
def test_below_min_events_returns_none(n):
    events = [mk_event(i, pre=70.0, ratio=0.20) for i in range(n)]
    assert t50_median(events) is None
    assert habituation_slope(events) is None
    assert consistency_sigma(events) is None
    assert reactivity_ratio_mean(events) is None
    assert valid_count(events) == n


@pytest.mark.parametrize("n", [8, 9, 16])
def test_at_or_above_min_events_returns_values(n):
    events = [mk_event(i, pre=70.0, ratio=0.20) for i in range(n)]
    assert t50_median(events) == pytest.approx(12.0)
    assert habituation_slope(events) == pytest.approx(0.0, abs=1e-12)
    assert consistency_sigma(events) == pytest.approx(0.0, abs=1e-9)
    assert reactivity_ratio_mean(events) == pytest.approx(0.20)
    assert valid_count(events) == n


def test_threshold_is_exactly_min_valid_events():
    assert MIN_VALID_EVENTS == 8
    seven = [mk_event(i, 70.0, 0.20) for i in range(7)]
    eight = [mk_event(i, 70.0, 0.20) for i in range(8)]
    assert t50_median(seven) is None and t50_median(eight) is not None


# ---------------------------------------------------------------- T50 中位数


def test_t50_median_robust_to_outlier():
    """中位数对异常值稳健：8 个 10 s + 1 个 200 s → 中位数仍为 10（均值会被带偏）。"""
    events = [mk_event(i, 70.0, 0.20, t50=10.0) for i in range(8)]
    events.append(mk_event(8, 70.0, 0.20, t50=200.0))
    med = t50_median(events)
    assert med == pytest.approx(10.0)
    assert med != pytest.approx(mean([e.t50_s for e in events]))


def test_t50_median_even_count_averages_middle_two():
    events = [mk_event(i, 70.0, 0.20, t50=8.0) for i in range(5)]
    events += [mk_event(i, 70.0, 0.20, t50=20.0) for i in range(5, 10)]
    assert t50_median(events) == pytest.approx(14.0)


def test_censored_events_enter_median_at_lower_bound():
    events = [mk_event(i, 70.0, 0.20, t50=12.0) for i in range(8)]
    events += [mk_event(i, 70.0, 0.20, t50=None, censored=True) for i in range(8, 10)]
    # 10 个值：8 个 12 + 2 个 45 → 排序后中间两个为 12, 12
    assert t50_median(events) == pytest.approx(12.0)
    assert censored_share(events) == pytest.approx(0.2)

    mostly_censored = [mk_event(i, 70.0, 0.20, t50=12.0) for i in range(4)]
    mostly_censored += [
        mk_event(i, 70.0, 0.20, t50=None, censored=True) for i in range(4, 12)
    ]
    assert censored_share(mostly_censored) == pytest.approx(8 / 12)
    assert t50_median(mostly_censored) == pytest.approx(45.0)  # 删失占多数 → 45


# ---------------------------------------------------------------- 习惯化斜率


@pytest.mark.parametrize("slope", [-0.010, -0.005, 0.0, 0.008])
def test_habituation_slope_sign_and_magnitude(slope):
    """β_hab = r ~ turn_index 的 OLS 斜率：负值 = 反应幅度逐轮递减（适应）。"""
    events = [mk_event(i, 70.0, 0.20 + slope * i) for i in range(10)]
    assert habituation_slope(events) == pytest.approx(slope, abs=1e-9)


def test_habituation_is_decreasing_for_adapting_profile():
    decreasing = [mk_event(i, 70.0, 0.25 - 0.01 * i) for i in range(12)]
    increasing = [mk_event(i, 70.0, 0.10 + 0.01 * i) for i in range(12)]
    assert habituation_slope(decreasing) < 0 < habituation_slope(increasing)


def test_habituation_regresses_ratio_not_delta():
    """A4：同一个"比例递减"过程，静息 60 与 90 的人 β_hab 相同，但 ΔHR 斜率不同。"""
    lo = [mk_event(i, 60.0, 0.20 - 0.01 * i) for i in range(10)]
    hi = [mk_event(i, 90.0, 0.20 - 0.01 * i) for i in range(10)]

    assert habituation_slope(lo) == pytest.approx(habituation_slope(hi), abs=1e-12)
    assert habituation_slope(lo) == pytest.approx(-0.01, abs=1e-9)

    def delta_slope(events):
        return ols_slope([float(e.turn_index) for e in events], [e.delta for e in events])

    # 绝对幅值斜率随静息心率缩放 → 不可比（这正是改用 ratio 的理由）
    assert delta_slope(lo) == pytest.approx(-0.6, rel=1e-9)
    assert delta_slope(hi) == pytest.approx(-0.9, rel=1e-9)
    assert delta_slope(lo) != pytest.approx(delta_slope(hi))


# ---------------------------------------------------------------- ratio 而非 ΔHR


def test_ratio_equal_but_delta_differs_across_resting_levels():
    """两个静息心率不同的人、相同比例反应：ratio 相同、ΔHR 不同。"""
    pre_lo, pre_hi, ratio = 60.0, 90.0, 0.20
    lo = [mk_event(i, pre_lo, ratio) for i in range(10)]
    hi = [mk_event(i, pre_hi, ratio) for i in range(10)]

    assert reactivity_ratio_mean(lo) == pytest.approx(ratio)
    assert reactivity_ratio_mean(hi) == pytest.approx(ratio)
    assert mean([e.delta for e in lo]) == pytest.approx(12.0)
    assert mean([e.delta for e in hi]) == pytest.approx(18.0)
    assert mean([e.delta for e in hi]) != pytest.approx(mean([e.delta for e in lo]))


def test_ratio_event_property_matches_definition():
    e = mk_event(0, 80.0, 0.15)
    assert e.delta == pytest.approx(12.0)
    assert e.ratio == pytest.approx(0.15)
    assert mk_event(0, 80.0, 0.0).ratio == pytest.approx(0.0)
    assert InterviewEvent(0, None, None, None).ratio is None


# ---------------------------------------------------------------- valid_count


def test_valid_count_exclusion_rules():
    events = [
        mk_event(0, 70.0, 0.20),  # 有效：ΔHR = 14
        mk_event(1, 70.0, 0.20, valid=False),  # 无效事件
        mk_event(2, 70.0, 0.01),  # ΔHR = 0.7 < 2 → 未激发
        mk_event(3, 70.0, 0.03),  # ΔHR = 2.1 → 计入
        mk_event(4, 70.0, 0.019),  # ΔHR = 1.33 < 2 → 未激发
        InterviewEvent(5, None, None, 12.0),  # 无观测
        mk_event(6, 70.0, -0.10),  # 峰低于基线 → 非应激反应
    ]
    assert valid_count(events) == 2
    assert MIN_DELTA_BPM == 2.0


@pytest.mark.parametrize(
    "peak,expected_count",
    [(72.0, 1), (71.999, 0), (100.0, 1)],  # ΔHR = 2.0 恰好计入；1.999 不计入
)
def test_valid_count_delta_threshold_boundary(peak, expected_count):
    ev = InterviewEvent(turn_index=0, pre_bpm=70.0, peak_bpm=peak, t50_s=12.0)
    assert valid_count([ev]) == expected_count


def test_degenerate_turn_index_abstains():
    """轮次索引无变异（数据异常）→ 不猜斜率，返回 None。"""
    events = [mk_event(0, 70.0, 0.20 + 0.01 * i) for i in range(10)]
    assert valid_count(events) == 10
    assert habituation_slope(events) is None
    assert consistency_sigma(events) is None


def test_enough_events_but_missing_t50_still_abstains():
    """有效事件够但恢复量缺失 → 不产出 T50 中位数（纪律一优先于纪律二的计数）。"""
    events = [InterviewEvent(i, 70.0, 84.0, None) for i in range(10)]
    assert valid_count(events) == 10
    assert t50_median(events) is None
    assert reactivity_ratio_mean(events) == pytest.approx(0.2)


def test_dirty_baseline_abstains_ratio():
    """pre=0 的脏数据：ratio 不可定义 → None，而不是除出一个数。"""
    events = [InterviewEvent(i, 0.0, 84.0, 12.0) for i in range(10)]
    assert reactivity_ratio_mean(events) is None


def test_censored_share_of_empty_is_zero():
    assert censored_share([]) == pytest.approx(0.0)


# ---------------------------------------------------------------- 一致性 σ


def test_consistency_sigma_zero_for_perfectly_consistent():
    flat = [mk_event(i, 70.0, 0.20) for i in range(10)]
    assert consistency_sigma(flat) == pytest.approx(0.0, abs=1e-9)


def test_consistency_sigma_grows_with_noise():
    flat = [mk_event(i, 70.0, 0.20) for i in range(10)]
    noisy = [mk_event(i, 70.0, 0.20 if i % 2 == 0 else 0.24) for i in range(10)]
    s_flat, s_noisy = consistency_sigma(flat), consistency_sigma(noisy)
    assert s_noisy > s_flat
    assert s_noisy == pytest.approx(0.0220193, rel=1e-5)  # 手算残差标准差（自由度 n−2）
    # 反应幅度整体波动更大 → 残差更大
    wilder = [mk_event(i, 70.0, 0.10 if i % 2 == 0 else 0.30) for i in range(10)]
    assert consistency_sigma(wilder) > s_noisy


def test_consistency_sigma_matches_residual_definition():
    events = [mk_event(i, 70.0, 0.20 - 0.01 * i + (0.005 if i % 3 == 0 else -0.002)) for i in range(12)]
    xs = [float(e.turn_index) for e in events]
    ys = [e.ratio for e in events]
    slope = ols_slope(xs, ys)
    rss = sum((y - (mean(ys) + slope * (x - mean(xs)))) ** 2 for x, y in zip(xs, ys))
    expected = math.sqrt(rss / (len(xs) - 2))  # 自由度 n−2
    assert consistency_sigma(events) == pytest.approx(expected, rel=1e-9)


# ---------------------------------------------------------------- 心律不规则（§6.0）


def test_detect_arrhythmia_regular_rhythm_is_false():
    ibis = [0.8 + 0.004 * (i % 3 - 1) for i in range(40)]
    assert detect_arrhythmia(ibis) is False
    assert detect_arrhythmia([0.8] * 40) is False  # 完全规整 → CV = 0


def test_detect_arrhythmia_irregular_rhythm_is_true():
    assert detect_arrhythmia([0.4, 1.2] * 20) is True
    assert detect_arrhythmia([400.0, 1200.0] * 20) is True  # 尺度无关（毫秒）


@pytest.mark.parametrize("dev,expected", [(0.10, False), (0.20, False), (0.30, True), (0.60, True)])
def test_detect_arrhythmia_cv_threshold(dev, expected):
    """CV = |偏离|（对称两值时），阈值 ARRHYTHMIA_CV_THRESHOLD 两侧行为明确。"""
    ibis = [0.8 * (1 - dev), 0.8 * (1 + dev)] * 15
    cv = dev
    assert (cv > ARRHYTHMIA_CV_THRESHOLD) is expected
    assert detect_arrhythmia(ibis) is expected


def test_detect_arrhythmia_abstains_without_enough_beats():
    """样本不足 → 不判（弃权，不猜），与纪律一一致。"""
    assert detect_arrhythmia([]) is False
    assert detect_arrhythmia([0.4, 1.2] * (MIN_IBI_SAMPLES // 2 - 1)) is False
    assert detect_arrhythmia([0.0] * 40) is False  # 非生理值被剔除 → 样本不足


def test_detect_arrhythmia_ignores_non_positive_intervals():
    assert detect_arrhythmia([-1.0] * 40) is False


def test_arrhythmia_constants_are_documented_heuristics():
    assert 0.0 < ARRHYTHMIA_CV_THRESHOLD < 1.0
    assert MIN_IBI_SAMPLES >= 20
