"""置信度测试：边界值、单调性、组成权重的可加性。

关键性质：置信度只反映**证据充分度**，与分数高低正交（高分可以低置信，反之亦然）。
"""

from __future__ import annotations

import pytest

from ruipin.scoring.confidence import (
    CONF_DOWNGRADE,
    DEGRADED_FACTOR,
    FULL_ANSWER_LEN,
    MIN_ANSWER_LEN,
    N_SIGNALS_FULL,
    W_AUDIO,
    W_LENGTH,
    W_SIGNALS,
    W_VIDEO,
    compute_confidence,
    length_evidence,
    should_downgrade,
)


# ---------- 长度证据 ----------


@pytest.mark.parametrize(
    "n,expected",
    [
        (0, 0.0),
        (MIN_ANSWER_LEN, 0.0),            # 下界闭区间：刚好等于下限仍为 0
        (MIN_ANSWER_LEN + 1, 1 / 140),
        (90, 0.5),                         # (90-20)/140
        (FULL_ANSWER_LEN, 1.0),            # 上界闭区间：达到即饱和
        (FULL_ANSWER_LEN + 5000, 1.0),     # 超长不衰减
    ],
)
def test_length_evidence_values(n, expected):
    assert length_evidence(n) == pytest.approx(expected, abs=1e-9)


def test_length_evidence_monotonic_non_decreasing():
    vals = [length_evidence(n) for n in range(0, 400)]
    assert all(a <= b for a, b in zip(vals, vals[1:]))
    assert all(0.0 <= v <= 1.0 for v in vals)


def test_length_evidence_negative_is_clamped():
    assert length_evidence(-100) == 0.0


# ---------- 组合取值（手算可复核） ----------


@pytest.mark.parametrize(
    "answer_len,has_audio,has_video,n_signals,degraded,expected",
    [
        # 0 + 0 + 0 + 1/4 → 0.0625
        (0, False, False, 1, False, 0.0625),
        # 满长度 0.45 + 0 + 0 + 1/4 信号 0.0625
        (FULL_ANSWER_LEN, False, False, 1, False, 0.5125),
        # 半长度 0.225 + 音频 0.20 + 视频 0.10 + 信号满 0.25
        (90, True, True, N_SIGNALS_FULL, False, 0.775),
        # 同上但 provider 降级 → 折半
        (90, True, True, N_SIGNALS_FULL, True, 0.3875),
        # 只有视频，无音频
        (FULL_ANSWER_LEN, False, True, 4, False, 0.80),
    ],
)
def test_compute_confidence_exact_values(
    answer_len, has_audio, has_video, n_signals, degraded, expected
):
    got = compute_confidence(answer_len, has_audio, has_video, n_signals, degraded)
    assert got == pytest.approx(expected, abs=1e-9)


def test_component_weights_are_additive_and_sum_to_one():
    assert W_LENGTH + W_AUDIO + W_VIDEO + W_SIGNALS == pytest.approx(1.0)
    base = compute_confidence(FULL_ANSWER_LEN, False, False, 1)
    assert compute_confidence(FULL_ANSWER_LEN, True, False, 1) - base == pytest.approx(W_AUDIO)
    assert compute_confidence(FULL_ANSWER_LEN, False, True, 1) - base == pytest.approx(W_VIDEO)


# ---------- 单调性 ----------


def test_monotonic_in_length():
    vals = [compute_confidence(n, True, False, 2) for n in range(0, 300)]
    assert all(a <= b for a, b in zip(vals, vals[1:]))


def test_monotonic_in_signals_and_saturates():
    vals = [compute_confidence(100, False, False, k) for k in range(0, 12)]
    assert all(a <= b for a, b in zip(vals, vals[1:]))
    assert vals[N_SIGNALS_FULL] == vals[N_SIGNALS_FULL + 1] == vals[-1]
    assert compute_confidence(100, False, False, 1000) == vals[-1]


def test_modalities_only_increase():
    base = compute_confidence(100, False, False, 2)
    assert compute_confidence(100, True, False, 2) > base
    assert compute_confidence(100, False, True, 2) > base
    assert compute_confidence(100, True, True, 2) > compute_confidence(100, True, False, 2)


def test_degraded_never_increases():
    for n in (0, 50, 90, 160, 1000):
        full = compute_confidence(n, True, True, 4, False)
        deg = compute_confidence(n, True, True, 4, True)
        assert deg == pytest.approx(full * DEGRADED_FACTOR, abs=1e-12)
        assert deg <= full


def test_always_within_unit_interval():
    for n in range(0, 500, 7):
        for k in range(0, 8):
            for deg in (False, True):
                c = compute_confidence(n, True, True, k, deg)
                assert 0.0 <= c <= 1.0


# ---------- 降级判定 ----------


@pytest.mark.parametrize(
    "conf,expected",
    [
        (0.0, True),
        (0.10, True),
        (CONF_DOWNGRADE - 1e-6, True),   # 开区间：略低于阈值即降级
        (CONF_DOWNGRADE, False),         # 闭区间：正好等于阈值不降级
        (0.75, False),
        (1.0, False),
    ],
)
def test_should_downgrade_boundary(conf, expected):
    assert should_downgrade(conf) is expected


def test_downgrade_threshold_is_documented_value():
    assert CONF_DOWNGRADE == 0.50
