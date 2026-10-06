"""window.py 测试：三算法投票分支、缺算法、SNR 阈值边界、弃权不产出数值。"""

from __future__ import annotations

import pytest

from ruipin.ports import WindowQuality
from ruipin.physio.window import (
    AGREE_TOL_BPM,
    DISAGREE_TOL_BPM,
    SNR_MIN,
    agreement_score,
    compute_snr,
    is_rejected,
    vote,
)

GOOD_SNR = 0.80


# ---------------------------------------------------------------- 投票分支


@pytest.mark.parametrize(
    "chrom,pos,ssr,exp_bpm,exp_spread,exp_rejected",
    [
        # 一致：极差 2 BPM ≤ 3
        (70.0, 71.0, 72.0, 71.0, 2.0, False),
        # 一致：三者完全相同
        (64.0, 64.0, 64.0, 64.0, 0.0, False),
        # 一致边界：极差恰为 3.0（≤3 → 采信）
        (70.0, 73.0, 71.0, 71.0, 3.0, False),
        # 分歧：极差 4 BPM（3–8 → 取中位数但不弃权）
        (70.0, 74.0, 71.0, 71.0, 4.0, False),
        # 分歧边界：极差恰为 8.0（未超 8 → 仍采信中位数）
        (70.0, 78.0, 72.0, 72.0, 8.0, False),
        # 弃权：极差 8.5 > 8
        (68.0, 76.5, 72.0, None, 8.5, True),
        # 弃权：极差 10 > 8
        (70.0, 80.0, 71.0, None, 10.0, True),
    ],
)
def test_vote_branches(chrom, pos, ssr, exp_bpm, exp_spread, exp_rejected):
    w = vote(chrom, pos, ssr, snr=GOOD_SNR)
    assert isinstance(w, WindowQuality)
    if exp_bpm is None:
        assert w.bpm is None
    else:
        assert w.bpm == pytest.approx(exp_bpm)
    assert w.algo_spread == pytest.approx(exp_spread)
    assert w.rejected is exp_rejected
    assert w.snr == pytest.approx(GOOD_SNR)


def test_vote_takes_median_not_mean():
    """中位数而非均值：单个算法跑偏不影响结果。"""
    w = vote(70.0, 71.0, 95.0, snr=GOOD_SNR)
    # 极差 25 > 8 → 弃权（不是取中位数 71 了事）
    assert w.bpm is None
    assert w.rejected is True

    w2 = vote(70.0, 71.0, 71.5, snr=GOOD_SNR)  # 极差 1.5 → 一致
    assert w2.bpm == pytest.approx(71.0)
    assert w2.bpm != pytest.approx((70.0 + 71.0 + 71.5) / 3)
    assert w2.rejected is False


def test_vote_median_is_order_independent():
    a = vote(72.0, 70.0, 71.0, snr=GOOD_SNR)
    b = vote(71.0, 72.0, 70.0, snr=GOOD_SNR)
    assert a.bpm == pytest.approx(71.0)
    assert b.bpm == pytest.approx(a.bpm)


@pytest.mark.parametrize(
    "chrom,pos,ssr,exp_bpm,exp_spread,exp_rejected",
    [
        # 缺一个算法：按剩余两个处理，中位数 = 均值
        (70.0, None, 71.0, 70.5, 1.0, False),
        (None, 70.0, 71.0, 70.5, 1.0, False),
        (70.0, 71.0, None, 70.5, 1.0, False),
        # 缺一个但剩余两个分歧过大 → 依然弃权
        (70.0, 80.0, None, None, 10.0, True),
        (None, 70.0, 80.0, None, 10.0, True),
        # 只剩一个算法：直接采信（极差记为 0，一致率由 agreement_score 降权）
        (None, None, 72.0, 72.0, 0.0, False),
        (72.0, None, None, 72.0, 0.0, False),
        # 全部弃权 → 本窗口弃权
        (None, None, None, None, 0.0, True),
    ],
)
def test_vote_missing_algorithms(chrom, pos, ssr, exp_bpm, exp_spread, exp_rejected):
    w = vote(chrom, pos, ssr, snr=GOOD_SNR)
    if exp_bpm is None:
        assert w.bpm is None
    else:
        assert w.bpm == pytest.approx(exp_bpm)
    assert w.algo_spread == pytest.approx(exp_spread)
    assert w.rejected is exp_rejected


def test_single_algorithm_lowers_agreement():
    """只剩一个算法 → 采信但一致率显著低于三算法一致（§3.3）。"""
    one = vote(None, None, 72.0, snr=GOOD_SNR)
    three = vote(72.0, 72.5, 71.5, snr=GOOD_SNR)
    assert one.bpm == pytest.approx(72.0)
    assert agreement_score(one.algo_spread, n_algos=1) < agreement_score(
        three.algo_spread, n_algos=3
    )


# ---------------------------------------------------------------- SNR


def test_snr_formula_known_components():
    """spec 比 = 2（≈3 dB）、harm 比 = 1、continuity = 1 → 0.6·0.5+0.2·0.5+0.2·1。"""
    med = 10.0
    assert compute_snr(2 * med, med, 1 * med, 1.0) == pytest.approx(0.60)
    # spec 比 = 4 → norm = 4/6
    assert compute_snr(4 * med, med, 1 * med, 1.0) == pytest.approx(
        0.6 * (4 / 6) + 0.2 * 0.5 + 0.2 * 1.0
    )
    # 全零分量（continuity=0）→ 只剩 spec 项
    assert compute_snr(2 * med, med, 0.0, 0.0) == pytest.approx(0.30)


def test_snr_monotonic_in_each_component():
    med = 10.0
    base = compute_snr(2 * med, med, med, 0.8)
    assert compute_snr(4 * med, med, med, 0.8) > base  # 主峰更显著
    assert compute_snr(2 * med, med, 2 * med, 0.8) > base  # 谐波更强
    assert compute_snr(2 * med, med, med, 1.0) > base  # 连续性更好
    assert compute_snr(2 * med, med, med, 0.4) < base


def test_snr_bounds_and_guards():
    med = 10.0
    assert 0.0 <= compute_snr(2 * med, med, med, 0.8) <= 1.0
    assert compute_snr(1e9 * med, med, 1e9 * med, 1.0) == pytest.approx(1.0)  # 饱和
    assert compute_snr(0.0, med, 0.0, 0.0) == pytest.approx(0.0)
    assert compute_snr(2 * med, 0.0, med, 1.0) == pytest.approx(0.2)  # 带内中位为 0
    assert compute_snr(2 * med, med, med, 3.0) == pytest.approx(
        compute_snr(2 * med, med, med, 1.0)
    )  # continuity 夹到 1


@pytest.mark.parametrize(
    "peak,med,harm,cont,expect_reject",
    [
        (20.0, 10.0, 10.0, 1.0, False),  # 0.60 → 通过
        (5.0, 10.0, 2.0, 0.5, True),  # ≈0.253 → 弃权
        (10.0, 10.0, 10.0, 0.0, True),  # 0.6*1/3+0.2*0.5 = 0.30 → 弃权
    ],
)
def test_snr_threshold_decides_window(peak, med, harm, cont, expect_reject):
    snr = compute_snr(peak, med, harm, cont)
    assert (snr < SNR_MIN) is expect_reject
    w = vote(70.0, 70.5, 71.0, snr=snr)
    assert w.rejected is expect_reject
    if expect_reject:
        assert w.bpm is None  # 弃权不产出数值


# ---------------------------------------------------------------- is_rejected


@pytest.mark.parametrize(
    "bpm,snr,spread,expected",
    [
        (70.0, 0.90, 0.0, False),
        (None, 0.90, 0.0, True),  # 算法自身弃权
        (70.0, 0.349, 0.0, True),  # SNR 阈值下侧
        (70.0, 0.350, 0.0, False),  # SNR 阈值边界（<0.35 才弃权）
        (70.0, 0.351, 0.0, False),
        (70.0, 0.90, 8.0, False),  # 极差阈值边界（>8 才弃权）
        (70.0, 0.90, 8.001, True),
        (70.0, 0.90, 30.0, True),
        (None, 0.10, 30.0, True),
    ],
)
def test_is_rejected_boundaries(bpm, snr, spread, expected):
    assert is_rejected(bpm, snr, spread) is expected


def test_is_rejected_consistent_with_vote():
    w = vote(70.0, 71.0, 72.0, snr=0.9)
    assert is_rejected(w.bpm, w.snr, w.algo_spread) is w.rejected


# ---------------------------------------------------------------- 保守默认


def test_vote_without_snr_abstains():
    """未给出 SNR 时按最保守处理：弃权（纪律一：弃权优于猜测）。"""
    w = vote(70.0, 71.0, 72.0)
    assert w.snr == pytest.approx(0.0)
    assert w.rejected is True
    assert w.bpm is None


def test_low_snr_rejects_even_when_algos_agree():
    """算法完全一致但频谱质量不足 → 仍然弃权，绝不输出"大概的数"。"""
    w = vote(70.0, 70.5, 71.0, snr=0.20)
    assert w.algo_spread == pytest.approx(1.0)
    assert w.rejected is True
    assert w.bpm is None


# ---------------------------------------------------------------- 一致率


@pytest.mark.parametrize(
    "spread,n_algos,expected",
    [
        (0.0, 3, 1.0),
        (AGREE_TOL_BPM, 3, 1.0),  # 3.0 仍属"一致"
        (AGREE_TOL_BPM + 0.01, 3, 0.55),  # 进入分歧区间
        (DISAGREE_TOL_BPM, 3, 0.55),
        (DISAGREE_TOL_BPM + 0.01, 3, 0.0),  # 已弃权
        (99.0, 3, 0.0),
        (0.0, 2, 1.0),
        (5.0, 2, 0.55),
        (9.0, 2, 0.0),
        (0.0, 1, 0.30),  # 只剩一个算法：采信但一致率低
        (99.0, 1, 0.30),
        (0.0, 0, 0.0),
    ],
)
def test_agreement_score(spread, n_algos, expected):
    assert agreement_score(spread, n_algos) == pytest.approx(expected)
    assert 0.0 <= agreement_score(spread, n_algos) <= 1.0
