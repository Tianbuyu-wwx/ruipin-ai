"""等级划分测试。

阈值口径（必须写清楚闭区间在哪一侧）：
- A: score >= 85  （85 属于 A）
- B: 70 <= score < 85（70 属于 B）
- C: 60 <= score < 60+（60 属于 C）
- D: score < 60
即**阈值归入较高的一档**，与"60 分及格"的直觉一致。
"""

from __future__ import annotations

import pytest

from ruipin.scoring.level import A_MIN, B_MIN, C_MIN, LEVELS, Level, classify


@pytest.mark.parametrize(
    "score,expected",
    [
        (100.0, "A"),
        (A_MIN + 0.01, "A"),
        (A_MIN, "A"),            # 85 → A（闭区间在高档）
        (A_MIN - 0.01, "B"),
        (B_MIN + 0.01, "B"),
        (B_MIN, "B"),            # 70 → B
        (B_MIN - 0.01, "C"),
        (C_MIN + 0.01, "C"),
        (C_MIN, "C"),            # 60 → C（及格线含 60）
        (C_MIN - 0.01, "D"),
        (1.0, "D"),
        (0.0, "D"),
    ],
)
def test_threshold_boundaries(score, expected):
    assert classify(score).level == expected


def test_labels_and_colors_are_present():
    for lv in LEVELS:
        assert lv.label
        assert lv.color.startswith("#")
        assert len(lv.color) == 7
    assert [lv.level for lv in LEVELS] == ["A", "B", "C", "D"]
    assert [lv.label for lv in LEVELS] == ["优秀", "良好", "合格", "待提升"]


def test_returns_frozen_dataclass():
    lv = classify(90.0)
    assert isinstance(lv, Level)
    with pytest.raises(Exception):
        lv.level = "B"  # type: ignore[misc]


@pytest.mark.parametrize("bad", [100.01, 101.0, -0.01, -1.0, float("nan")])
def test_out_of_range_raises(bad):
    with pytest.raises(ValueError):
        classify(bad)


def test_monotonic_non_increasing_level():
    order = {"A": 3, "B": 2, "C": 1, "D": 0}
    prev = 0
    for s in range(0, 101):
        cur = order[classify(float(s)).level]
        assert cur >= prev  # 分值上升，等级只能变好
        prev = cur


def test_thresholds_ordered():
    assert C_MIN < B_MIN < A_MIN
