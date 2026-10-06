"""跨事件聚合统计量 —— 唯一允许进入评分的一层（详设 §4.3 / §4.4）。

纪律二：**禁止对单次事件下判断**。单次 ΔHR 的 SNR 仅 2–5；跨事件回归斜率的
噪声按 ~1/√N 收缩（§1.4）。因此本模块所有函数在 `n_valid < 8` 时返回 `None`——
不是返回 0，不是返回默认值，而是**不产出该统计量**。

纪律 A4：**幅值类结论一律用 `ratio = ΔHR/pre`**，不用绝对 ΔHR。
静息 50 与 90 的人、不同摄像头的增益，都会让绝对幅值不可比；
`T50` 是时间量，天然对这些因素稳健。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from ..ports import InterviewEvent

# --- 参与统计的最小有效事件数（§1.4 / §4.4 硬约束）------------------------
MIN_VALID_EVENTS = 8

# --- 事件剔除规则（§4.3）---------------------------------------------------
MIN_DELTA_BPM = 2.0  # ΔHR < 2 BPM 视为"未激发"，不计入恢复统计
RECOVERY_WINDOW_S = 45.0  # 恢复观测窗长度（t_e + 45s）
CENSORED_T50_S = 45.0  # 右删失时以该下限参与，并标记 censored

# --- 指数恢复拟合 ----------------------------------------------------------
MIN_FIT_POINTS = 4  # 少于该点数不拟合（弃权优于猜测）

# --- 心律不齐启发式检测（§6.0）---------------------------------------------
# 阈值取自经验：正常窦性心律的逐拍间期变异系数通常 <0.15，房颤等不规则节律显著更高。
# 【启发式、非诊断】仅用于"是否自动关闭本模块"的工程判断，不产出任何健康结论。
ARRHYTHMIA_CV_THRESHOLD = 0.25
MIN_IBI_SAMPLES = 20  # 样本不足则不判（同样遵守"弃权"）


@dataclass(frozen=True)
class RecoveryFit:
    """单事件的指数恢复拟合结果：HR(t) = pre + Δ·exp(−λ·t)，t 自峰值起算。"""

    lam: Optional[float]  # 恢复速率 (1/s)；None = 拟合不可信，弃权
    t50_s: Optional[float]  # 半衰期 ln2/λ；右删失时取 CENSORED_T50_S
    censored: bool  # True = 45s 内未回落至 50%，t50 为下限


def _ols(xs: list[float], ys: list[float]) -> Optional[tuple[float, float, float]]:
    """普通最小二乘 y = a + b·x。返回 (斜率, 截距, RSS)；x 无变异时返回 None。"""
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0.0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = my - slope * mx
    rss = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    return slope, intercept, rss


# ---------- 事件筛选 ----------


def valid_count(events: list[InterviewEvent]) -> int:
    """有效且 ΔHR ≥ 2 BPM 的事件数（"未激发"不计入恢复统计，§4.3）。"""
    return sum(
        1
        for e in events
        if e.valid and e.delta is not None and e.delta >= MIN_DELTA_BPM
    )


def _eligible(events: list[InterviewEvent]) -> list[InterviewEvent]:
    return [
        e
        for e in events
        if e.valid and e.delta is not None and e.delta >= MIN_DELTA_BPM
    ]


def _gated(events: list[InterviewEvent]) -> Optional[list[InterviewEvent]]:
    """≥8 个有效事件才放行，否则 None（纪律二）。"""
    pool = _eligible(events)
    return pool if len(pool) >= MIN_VALID_EVENTS else None


# ---------- 单事件恢复拟合 ----------


def fit_recovery(
    times: list[float],
    hr: list[float],
    pre: float,
    *,
    window_s: float = RECOVERY_WINDOW_S,
) -> RecoveryFit:
    """对 HR(t) = pre + Δ·exp(−λ·t) 做对数线性最小二乘（§4.3 的 λ_i / T50_i）。

    取 ln(HR(t) − pre) = ln(Δ) − λ·t，斜率的相反数即 λ；`T50 = ln2/λ`。
    `times` 自峰值时刻起算（t=0 为峰值）。以下情形**弃权**（λ=None）：
    有效点不足、无回落（λ≤0，即 HR 未向基线收敛）。
    45 s 观测窗内未回落至 50% → `censored=True`，`t50_s` 取 45 s 下限参与。
    """
    xs: list[float] = []
    ys: list[float] = []
    for t, h in zip(times, hr):
        if h is None or t is None:
            continue
        if t > window_s:  # 观测窗外截断（可被下一题截断，§4.1）
            continue
        excess = h - pre
        if excess <= 0.0:  # 已回到基线或以下：对指数模型无信息
            continue
        xs.append(float(t))
        ys.append(math.log(excess))

    if len(xs) < MIN_FIT_POINTS:
        return RecoveryFit(lam=None, t50_s=None, censored=False)

    fit = _ols(xs, ys)
    if fit is None:
        return RecoveryFit(lam=None, t50_s=None, censored=False)

    lam = -fit[0]
    if lam <= 0.0:
        return RecoveryFit(lam=None, t50_s=None, censored=False)

    t50 = math.log(2.0) / lam
    if t50 > window_s:
        return RecoveryFit(lam=lam, t50_s=CENSORED_T50_S, censored=True)
    return RecoveryFit(lam=lam, t50_s=t50, censored=False)


def estimate_lambda(
    times: list[float], hr: list[float], pre: float
) -> Optional[float]:
    """单事件恢复速率 λ (1/s)；不可信时 None。"""
    return fit_recovery(times, hr, pre).lam


def estimate_t50(
    times: list[float], hr: list[float], pre: float
) -> Optional[float]:
    """单事件 T50 (s)；不可信时 None，右删失时返回 45 s 下限。"""
    return fit_recovery(times, hr, pre).t50_s


# ---------- 跨事件聚合（唯一可进入评分的统计量）----------


def t50_median(events: list[InterviewEvent]) -> Optional[float]:
    """T50 的中位数（对异常值稳健，§4.4）。

    `censored=True` 的事件按 45 s 下限参与（并可用 `censored_share` 标记占比）。
    有效事件 < 8 时返回 None。
    """
    pool = _gated(events)
    if pool is None:
        return None
    # 右删失事件以 45 s 下限参与（真实 T50 只会更长）
    vals = [CENSORED_T50_S if e.censored else e.t50_s for e in pool]
    vals = [v for v in vals if v is not None]
    if len(vals) < MIN_VALID_EVENTS:
        return None
    s = sorted(vals)
    n = len(s)
    mid = n // 2
    if n % 2:
        return float(s[mid])
    return (float(s[mid - 1]) + float(s[mid])) / 2.0


def censored_share(events: list[InterviewEvent]) -> float:
    """右删失事件占比 ∈ [0,1]：越高说明"恢复慢到测不出"的事件越多，聚合值越偏乐观。"""
    pool = _eligible(events)
    if not pool:
        return 0.0
    return sum(1 for e in pool if e.censored) / len(pool)


def reactivity_ratio_mean(events: list[InterviewEvent]) -> Optional[float]:
    """归一化反应幅度 r = ΔHR/pre 的均值（A4：不用绝对 ΔHR）。

    仅作反应性描述与协变量，不单独计分（§4.4）。有效事件 < 8 时返回 None。
    """
    pool = _gated(events)
    if pool is None:
        return None
    vals = [e.ratio for e in pool if e.ratio is not None]
    if len(vals) < MIN_VALID_EVENTS:
        return None
    return sum(vals) / len(vals)


def _ratio_regression(
    events: list[InterviewEvent],
) -> Optional[tuple[float, float, int]]:
    """对 r_i ~ turn_index 做 OLS。返回 (斜率, RSS, n)；不满足条件时 None。"""
    pool = _gated(events)
    if pool is None:
        return None
    pts = [(float(e.turn_index), e.ratio) for e in pool if e.ratio is not None]
    if len(pts) < MIN_VALID_EVENTS:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    fit = _ols(xs, ys)
    if fit is None:  # 轮次索引无变异（数据异常）→ 不猜
        return None
    return fit[0], fit[2], len(pts)


def habituation_slope(events: list[InterviewEvent]) -> Optional[float]:
    """习惯化斜率 β_hab：对 **r_i ~ turn_index** 做 OLS（§4.4 + A4）。

    负值 = 反应幅度随面试推进递减（适应）；单位：每轮次的 r 变化量。
    **不要**对 ΔHR 回归——绝对幅值跨个体不可比。有效事件 < 8 时返回 None。
    """
    reg = _ratio_regression(events)
    return None if reg is None else reg[0]


def consistency_sigma(events: list[InterviewEvent]) -> Optional[float]:
    """上述回归的残差标准差（σ_resid，自由度 n−2）：越大越不一致、越不可信。"""
    reg = _ratio_regression(events)
    if reg is None:
        return None
    _, rss, n = reg
    if n <= 2:
        return None
    return math.sqrt(max(rss, 0.0) / (n - 2))


# ---------- 心律不规则自动检测（§6.0）----------


def detect_arrhythmia(ibis: list[float]) -> bool:
    """逐拍间期（IBI，秒或毫秒均可，尺度无关）的变异系数超阈值 → True。

    **启发式、非诊断**：仅用于"整场关闭本模块"的工程判断，不构成任何医学结论，
    也不写入任何分数。样本数不足 `MIN_IBI_SAMPLES` 时返回 False（弃权，不猜）。
    """
    if not ibis or len(ibis) < MIN_IBI_SAMPLES:
        return False
    xs = [float(v) for v in ibis if v is not None and v > 0.0]
    if len(xs) < MIN_IBI_SAMPLES:
        return False
    n = len(xs)
    mean = sum(xs) / n
    if mean <= 0.0:
        return False
    var = sum((x - mean) ** 2 for x in xs) / n
    cv = math.sqrt(var) / mean
    return cv > ARRHYTHMIA_CV_THRESHOLD
