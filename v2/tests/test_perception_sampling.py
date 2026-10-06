"""帧采样与变化门控（方案 §4.2）。

覆盖：变化门控生效、采样率限制、每轮/每场上限、统计计数、边界（汉明距离 5/6）、
ROI 外扩与分辨率压制。
"""

from __future__ import annotations

import pytest

from ruipin.perception.sampling import (
    DEFAULT_CHANGE_THRESHOLD,
    DEFAULT_PER_TURN_CAP,
    DEFAULT_RATE_S,
    DEFAULT_SESSION_CAP,
    MAX_LONG_EDGE,
    ROI_EXPAND,
    FrameSampler,
    expand_roi,
    hamming,
    resize_long_edge,
)

ZERO = "0000000000000000"          # 64 bit pHash
DIST_5 = "f100000000000000"        # 与 ZERO 距离 5（0xf=4 bit + 0x1=1 bit）
DIST_6 = "f300000000000000"        # 与 ZERO 距离 6（0xf=4 bit + 0x3=2 bit）


# --------------------------------------------------------------------------
# hamming
# --------------------------------------------------------------------------


def test_hamming_zero_for_identical():
    assert hamming(ZERO, ZERO) == 0
    assert hamming("abcdef0123456789", "abcdef0123456789") == 0


def test_hamming_known_values():
    assert hamming("00", "ff") == 8
    assert hamming("0f0f", "f0f0") == 16
    assert hamming(ZERO, DIST_5) == 5
    assert hamming(ZERO, DIST_6) == 6


def test_hamming_case_insensitive():
    assert hamming("AABB", "aabb") == 0
    assert hamming("FF", "00") == 8


def test_hamming_length_mismatch_raises():
    with pytest.raises(ValueError, match="长度不一致"):
        hamming("abcd", "abc")


def test_hamming_rejects_non_hex():
    with pytest.raises(ValueError):
        hamming("zzzz", "0000")


# --------------------------------------------------------------------------
# 默认值与规格一致性
# --------------------------------------------------------------------------


def test_defaults_match_spec():
    assert DEFAULT_RATE_S == 4.0            # 1 帧 / 4s
    assert DEFAULT_CHANGE_THRESHOLD == 6    # 汉明距离 <6 丢弃
    assert DEFAULT_PER_TURN_CAP == 12       # 每轮上限
    assert DEFAULT_SESSION_CAP == 150       # 每场上限
    assert MAX_LONG_EDGE == 512             # 最长边 ≤512
    assert ROI_EXPAND == 1.4                # 人脸框外扩 1.4×


def test_defaults_wired_into_sampler():
    s = FrameSampler()
    assert (s.rate_s, s.change_threshold, s.per_turn_cap, s.session_cap) == (
        4.0, 6, 12, 150,
    )


# --------------------------------------------------------------------------
# 门控：变化
# --------------------------------------------------------------------------


def test_first_frame_always_sampled():
    s = FrameSampler()
    assert s.should_sample(t_s=0.0, last_sent_t=None, phash=ZERO, last_sent_phash=None)
    assert s.sampled == 0  # 决策不计入，只有 record_sent 才计数


def test_identical_phash_dropped_by_change_gate():
    s = FrameSampler()
    ok = s.should_sample(t_s=10.0, last_sent_t=0.0, phash=ZERO, last_sent_phash=ZERO)
    assert ok is False
    assert s.dropped_by_change == 1
    assert s.dropped_by_rate == 0
    assert s.dropped_by_cap == 0


def test_change_gate_boundary_5_dropped_6_kept():
    s = FrameSampler(change_threshold=6)
    # 距离 5 < 6 → 丢弃
    assert s.should_sample(8.0, 0.0, DIST_5, ZERO) is False
    assert s.dropped_by_change == 1
    # 距离 6 ≥ 6 → 保留
    assert s.should_sample(16.0, 0.0, DIST_6, ZERO) is True
    assert s.dropped_by_change == 1


def test_change_threshold_is_configurable():
    s = FrameSampler(change_threshold=12)
    # 距离 6 在新阈值下被丢弃
    assert s.should_sample(8.0, 0.0, DIST_6, ZERO) is False


# --------------------------------------------------------------------------
# 门控：采样率
# --------------------------------------------------------------------------


def test_rate_gate_blocks_before_rate_s():
    s = FrameSampler(rate_s=4.0)
    assert s.should_sample(3.99, 0.0, DIST_6, ZERO) is False
    assert s.dropped_by_rate == 1


def test_rate_gate_passes_at_rate_s():
    s = FrameSampler(rate_s=4.0)
    assert s.should_sample(4.0, 0.0, DIST_6, ZERO) is True
    assert s.dropped_by_rate == 0


def test_rate_gate_precedence_over_change_gate():
    """同一帧同时违反速率与变化时，只计一次（速率优先）。"""
    s = FrameSampler(rate_s=4.0)
    assert s.should_sample(1.0, 0.0, ZERO, ZERO) is False
    assert (s.dropped_by_rate, s.dropped_by_change, s.dropped_by_cap) == (1, 0, 0)


def test_static_candidate_uploads_almost_nothing():
    """30 s 答题期、每 0.5 s 到达一帧（共 60 帧）、候选人静止：

    首帧必采（保证每轮至少一帧证据），此后速率只放行 52 帧，而这 52 帧
    全部被变化门控丢弃 —— 上传 1 帧 / 60 帧。这正是"静止可省 40–60%"的极端版。
    """
    s = FrameSampler()
    last_t: float | None = None
    last_hash: str | None = None
    for i in range(1, 61):
        t = i * 0.5
        if s.should_sample(t, last_t, ZERO, last_hash):
            s.record_sent()
            last_t, last_hash = t, ZERO
    assert s.sampled == 1
    assert (s.dropped_by_rate, s.dropped_by_change, s.dropped_by_cap) == (7, 52, 0)
    assert s.dropped_by_rate + s.dropped_by_change + s.sampled == 60


# --------------------------------------------------------------------------
# 门控：上限
# --------------------------------------------------------------------------


def test_per_turn_cap_blocks_13th_frame():
    s = FrameSampler(per_turn_cap=12)
    for _ in range(12):
        assert s.should_sample(0.0, None, DIST_6, ZERO) is True
        s.record_sent()
    assert s.turn_sampled == 12
    assert s.turn_exhausted is True
    assert s.should_sample(100.0, 0.0, DIST_6, ZERO) is False
    assert s.dropped_by_cap == 1
    assert s.sampled == 12  # 未多采


def test_begin_turn_resets_only_turn_quota():
    s = FrameSampler(per_turn_cap=2, session_cap=10)
    for _ in range(2):
        s.should_sample(0.0, None, DIST_6, ZERO)
        s.record_sent()
    assert s.turn_exhausted is True
    s.begin_turn()
    assert s.turn_sampled == 0 and s.sampled == 2
    assert s.should_sample(0.0, None, DIST_6, ZERO) is True


def test_session_cap_is_hard_and_survives_begin_turn():
    s = FrameSampler(per_turn_cap=5, session_cap=7)
    while not s.exhausted:
        if s.turn_exhausted:  # 跨轮：轮内配额重置，场次配额继续吃紧
            s.begin_turn()
        assert s.should_sample(0.0, None, DIST_6, ZERO) is True
        s.record_sent()
    assert s.sampled == 7
    s.begin_turn()  # 新一轮也救不回来
    assert s.exhausted is True
    assert s.should_sample(0.0, None, DIST_6, ZERO) is False
    assert s.dropped_by_cap >= 1


def test_cap_precedence_over_rate_and_change():
    s = FrameSampler(session_cap=1)
    s.record_sent()
    assert s.should_sample(0.0, 0.0, ZERO, ZERO) is False  # 同时违反三项
    assert (s.dropped_by_cap, s.dropped_by_rate, s.dropped_by_change) == (1, 0, 0)


def test_remaining_counters():
    s = FrameSampler(per_turn_cap=3, session_cap=5)
    assert s.turn_remaining == 3 and s.remaining == 5
    s.record_sent()
    assert s.turn_remaining == 2 and s.remaining == 4


# --------------------------------------------------------------------------
# 计数与生命周期
# --------------------------------------------------------------------------


def test_stats_counts_are_exact():
    s = FrameSampler(rate_s=4.0, change_threshold=6, per_turn_cap=100, session_cap=100)
    s.should_sample(0.0, None, ZERO, None)          # 首帧：放行
    s.record_sent()
    s.should_sample(1.0, 0.0, DIST_6, ZERO)         # 速率
    s.should_sample(5.0, 0.0, ZERO, ZERO)           # 变化
    s.should_sample(9.0, 5.0, DIST_6, ZERO)         # 放行
    s.record_sent()
    assert s.stats() == {
        "sampled": 2,
        "turn_sampled": 2,
        "dropped_by_rate": 1,
        "dropped_by_change": 1,
        "dropped_by_cap": 0,
    }


def test_record_sent_rejects_negative():
    s = FrameSampler()
    with pytest.raises(ValueError):
        s.record_sent(-1)


def test_reset_clears_everything():
    s = FrameSampler()
    s.should_sample(0.0, 0.0, ZERO, ZERO)
    s.record_sent(3)
    s.reset()
    assert s.stats() == {
        "sampled": 0,
        "turn_sampled": 0,
        "dropped_by_rate": 0,
        "dropped_by_change": 0,
        "dropped_by_cap": 0,
    }
    assert s.exhausted is False


# --------------------------------------------------------------------------
# ROI 裁剪与分辨率
# --------------------------------------------------------------------------


def test_expand_roi_center_preserved():
    assert expand_roi((100, 100, 200, 200)) == (60, 60, 280, 280)


def test_expand_roi_clamped_to_frame():
    # 左上角的框外扩后越界，必须裁回画面内
    assert expand_roi((0, 0, 100, 100), bounds=(1000, 800)) == (0, 0, 140, 140)


def test_expand_roi_scale_one_is_identity():
    assert expand_roi((10, 20, 30, 40), scale=1.0) == (10, 20, 30, 40)


def test_expand_roi_rejects_bad_input():
    with pytest.raises(ValueError):
        expand_roi((0, 0, 100, 100), scale=0.5)
    with pytest.raises(ValueError):
        expand_roi((0, 0, 0, 10))


def test_resize_long_edge_downscales_and_keeps_ratio():
    assert resize_long_edge(1920, 1080) == (512, 288)
    assert resize_long_edge(200, 800) == (128, 512)


def test_resize_long_edge_never_upscales():
    assert resize_long_edge(320, 240) == (320, 240)
    assert resize_long_edge(512, 512) == (512, 512)


def test_resize_long_edge_rejects_bad_input():
    with pytest.raises(ValueError):
        resize_long_edge(0, 10)
    with pytest.raises(ValueError):
        resize_long_edge(10, 10, max_edge=0)
