"""等级划分：把 0-100 的分值映射为 A/B/C/D 四档（只做呈现，不参与计算）。"""

from __future__ import annotations

from dataclasses import dataclass

# 阈值：**闭区间归到高档**。
# 85 → A，70 → B，60 → C；85 以下（不含）→ B，70 以下（不含）→ C，60 以下（不含）→ D。
# 之所以取"高档闭区间"：阈值是"达到即算"，与"及格线 60 分含 60"的直觉一致。
A_MIN = 85.0
B_MIN = 70.0
C_MIN = 60.0

SCORE_MIN = 0.0
SCORE_MAX = 100.0


@dataclass(frozen=True)
class Level:
    """等级结果。

    color 为报告用的十六进制色值（A 绿 → D 红，中间蓝/橙），不承载语义。
    """

    level: str
    label: str
    color: str


LEVELS: tuple[Level, ...] = (
    Level("A", "优秀", "#16A34A"),
    Level("B", "良好", "#2563EB"),
    Level("C", "合格", "#D97706"),
    Level("D", "待提升", "#DC2626"),
)

_BY_CODE: dict[str, Level] = {lv.level: lv for lv in LEVELS}


def classify(score: float) -> Level:
    """分值 → 等级。越界抛 ValueError（不做截断，避免把异常输入静默变成 D）。"""
    s = float(score)
    if s != s:  # NaN
        raise ValueError(f"score 非数值: {score!r}")
    if not (SCORE_MIN <= s <= SCORE_MAX):
        raise ValueError(f"score 越界 [{SCORE_MIN},{SCORE_MAX}]: {s}")
    if s >= A_MIN:
        return _BY_CODE["A"]
    if s >= B_MIN:
        return _BY_CODE["B"]
    if s >= C_MIN:
        return _BY_CODE["C"]
    return _BY_CODE["D"]


__all__ = ["Level", "LEVELS", "A_MIN", "B_MIN", "C_MIN", "classify"]
