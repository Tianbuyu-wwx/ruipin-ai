"""可靠性门控、分位数标定与权重重分配（详设 §4.5 / §4.6 / §4.7）。

纪律三：**测不准 = 不计入，不是给低分**。可靠性不足 → `gate(R)=0` → 权重归零，
并把这份权重按比例还给其他维度（保证总分仍是 100% 加权和，不会因"未评估"而
系统性压低总分——这是验收必须断言的一条红线）。

分量合成（v1，§1.5：**不含** S_vagal，vmHRV 属 v2）：
    S = 0.60·S_recovery + 0.25·S_habituation + 0.15·S_consistency
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Union

from ..ports import DIMENSIONS, PHYSIO_DIM, InterviewEvent, RegulationResult
from .metrics import (
    MIN_VALID_EVENTS,
    consistency_sigma,
    detect_arrhythmia,
    habituation_slope,
    reactivity_ratio_mean,
    t50_median,
    valid_count,
)

# --- 权重上限与门控（§4.7）-------------------------------------------------
W_MAX = 0.08  # 生理维度权重上限 8%
GATE_FLOOR = 0.40  # R < 0.40 → 完全不计入
GATE_CEIL = 0.75  # R ≥ 0.75 → 权重全额

# --- 分量合成权重（§4.6，v1）-----------------------------------------------
COMPONENT_WEIGHTS: dict[str, float] = {
    "S_recovery": 0.60,  # T50 越小越高分：恢复速率文献支撑最强
    "S_habituation": 0.25,  # 负斜率越大越高分：经典适应指标
    "S_consistency": 0.15,  # 残差越小越高分：可靠性项
}

DEFAULT_WEIGHTS: dict[str, float] = {
    **{d: (1.0 - W_MAX) / len(DIMENSIONS) for d in DIMENSIONS},
    PHYSIO_DIM: W_MAX,
}


@dataclass(frozen=True)
class PhysioCalibration:
    """分位数标定表：`{分位概率 → 该分位上的指标观测值}`（§10.2）。

    三张表分别对应 T50_median、−β_hab、−σ_resid。
    """

    t50: Mapping[float, float]
    habituation: Mapping[float, float]
    consistency: Mapping[float, float]


# ⚠️ 占位值：未经标定，仅供链路联调。上线前必须由 ≥200 场真实面试的分位数
# 分布替换（§10.2），否则评分无意义。分位概率 → 观测值，按键值单调递增。
PLACEHOLDER_CALIBRATION = PhysioCalibration(
    t50={0.05: 6.0, 0.25: 11.0, 0.50: 16.0, 0.75: 23.0, 0.95: 34.0},
    habituation={0.05: -0.010, 0.25: -0.002, 0.50: 0.002, 0.75: 0.008, 0.95: 0.018},
    consistency={0.05: -0.090, 0.25: -0.050, 0.50: -0.035, 0.75: -0.022, 0.95: -0.010},
)


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if x < lo else (hi if x > hi else x)


def reliability(
    *,
    n_valid: int,
    n_expected: float,
    median_snr: float,
    reject_ratio: float,
    algo_agreement: float,
) -> float:
    """可靠性 R ∈ [0,1]（§4.7）。

        R = 0.40·(N_valid / N_expected)   # 有效事件占比
          + 0.35·(median SNR)             # 信号质量
          + 0.15·(1 − 弃权窗口占比)        # 时间连续性
          + 0.10·(算法一致率)              # 三算法投票一致比例

    `n_expected` ≤ 0（没有可期待的事件）时，覆盖率项按 0 计——不猜。
    各输入先夹到合法区间，防止脏数据把 R 顶出 [0,1]。
    """
    coverage = 0.0 if n_expected <= 0 else _clamp(n_valid / n_expected)
    snr = _clamp(median_snr)
    keep = _clamp(1.0 - _clamp(reject_ratio))
    agree = _clamp(algo_agreement)
    return _clamp(0.40 * coverage + 0.35 * snr + 0.15 * keep + 0.10 * agree)


def gate(r: float) -> float:
    """权重门控 ∈ [0,1]（§4.7 的公平性核心）。

        gate(R) = 0                    R < 0.40   → 完全不计入
                = (R − 0.40)/0.35      0.40 ≤ R < 0.75
                = 1                    R ≥ 0.75
    """
    if r < GATE_FLOOR:
        return 0.0
    if r >= GATE_CEIL:
        return 1.0
    return (r - GATE_FLOOR) / (GATE_CEIL - GATE_FLOOR)


def calibrate(
    value: float,
    quantiles: Mapping[float, float],
    *,
    higher_is_better: bool = True,
) -> float:
    """分位数映射 → 0–100（§4.5 / §10.2）。

    `quantiles`：`{分位概率 → 该分位上的观测值}`（如 `{0.5: 16.0}` 表示中位数 16 s）。
    先在标定表上做线性插值求出 `value` 所处的分位概率 p，再映射为分数：
    `higher_is_better=True` → p·100；`False`（如 T50，越小越好）→ (1−p)·100。
    落在表外则夹到端点分位，不外推。
    """
    if len(quantiles) < 2:
        raise ValueError("标定表至少需要 2 个分位点")
    pts = sorted((float(v), float(p)) for p, v in quantiles.items())
    for (v0, p0), (v1, p1) in zip(pts, pts[1:]):
        if p1 < p0:
            raise ValueError("标定表非单调：分位概率必须随观测值递增")

    if value <= pts[0][0]:
        p = pts[0][1]
    elif value >= pts[-1][0]:
        p = pts[-1][1]
    else:
        p = pts[-1][1]
        for (v0, p0), (v1, p1) in zip(pts, pts[1:]):
            if v0 <= value <= v1:
                frac = 0.0 if v1 == v0 else (value - v0) / (v1 - v0)
                p = p0 + frac * (p1 - p0)
                break

    score = (p if higher_is_better else 1.0 - p) * 100.0
    return _clamp(score, 0.0, 100.0)


def redistribute(
    weights: Mapping[str, float],
    excluded: Union[str, Iterable[str]],
    freed: float,
) -> dict[str, float]:
    """把 `freed` 的权重按原比例还给其余维度。

    - `weights`：含被排除维度在内的完整权重表（一般合计为 1）
    - `excluded`：不计入的维度名（本模块为 `PHYSIO_DIM`）
    - `freed`：需要归还的权重额度（= W_MAX − W_applied）

    返回**其余维度的新权重**：各自保持原有相对比例，总额为"原合计 + freed"。
    因此 `sum(返回值) + W_applied == 1`，关闭本模块不会系统性压低总分。
    """
    drop = {excluded} if isinstance(excluded, str) else set(excluded)
    base = {k: float(v) for k, v in weights.items() if k not in drop and v > 0.0}
    total = sum(base.values())
    if total <= 0.0 or not base:
        return {}
    scale = (total + max(float(freed), 0.0)) / total
    return {k: v * scale for k, v in base.items()}


def _unavailable(
    reason: str,
    *,
    n_events: int,
    n_valid: int,
    baseline_bpm: Optional[float],
    weights: Mapping[str, float],
) -> RegulationResult:
    """不计入：分数为 None、权重为 0、并把 W_MAX 全额归还其他维度。"""
    return RegulationResult(
        available=False,
        reason=reason,
        baseline_bpm=baseline_bpm,
        n_events=n_events,
        n_valid=n_valid,
        components={},
        score=None,
        reliability=0.0,
        weight_applied=0.0,
        redistributed_to=redistribute(weights, PHYSIO_DIM, W_MAX),
    )


def evaluate_physio(
    *,
    events: list[InterviewEvent],
    baseline_bpm: Optional[float],
    n_expected: float,
    median_snr: float,
    reject_ratio: float,
    algo_agreement: float,
    authorized: bool = True,
    ibis: Optional[list[float]] = None,
    weights: Optional[Mapping[str, float]] = None,
    calibration: Optional[PhysioCalibration] = None,
) -> RegulationResult:
    """产出应激后生理恢复维度的评分与权重（§4.5–4.7）。

    返回 `available=False` 且带明确 `reason` 的情形（按此顺序判定）：
    未授权 → 基线缺失 → 有效事件 < 8 → R < 0.40 → 检出心律不规则。
    这些情形下 `score=None`、`weight_applied=0`，权重全额归还其他维度。

    `weights` 缺省为 `DEFAULT_WEIGHTS`（六维等分 0.92 + 生理 0.08）。
    """
    w = dict(weights) if weights is not None else DEFAULT_WEIGHTS
    cal = calibration if calibration is not None else PLACEHOLDER_CALIBRATION
    n_events = len(events)
    n_valid = valid_count(events)

    if not authorized:
        return _unavailable(
            "未获得生理信号分析的单独授权，本项不计入（其他维度不受影响）",
            n_events=n_events,
            n_valid=n_valid,
            baseline_bpm=None,
            weights=w,
        )
    if baseline_bpm is None:
        return _unavailable(
            "静息基线心率缺失，本项不计入",
            n_events=n_events,
            n_valid=n_valid,
            baseline_bpm=None,
            weights=w,
        )
    if n_valid < MIN_VALID_EVENTS:
        return _unavailable(
            f"有效事件 {n_valid} < {MIN_VALID_EVENTS}："
            "统计量不足以支撑跨事件结论，本项不计入",
            n_events=n_events,
            n_valid=n_valid,
            baseline_bpm=baseline_bpm,
            weights=w,
        )

    r = reliability(
        n_valid=n_valid,
        n_expected=n_expected,
        median_snr=median_snr,
        reject_ratio=reject_ratio,
        algo_agreement=algo_agreement,
    )
    if r < GATE_FLOOR:
        return _unavailable(
            f"可靠性 R={r:.2f} < {GATE_FLOOR:.2f}：信号质量不足，本项不计入"
            "（测不准 = 不计入，而非给低分）",
            n_events=n_events,
            n_valid=n_valid,
            baseline_bpm=baseline_bpm,
            weights=w,
        )
    if ibis is not None and detect_arrhythmia(ibis):
        return _unavailable(
            "检测到心律不规则，本项自动关闭"
            "（启发式检测，非医学诊断；不构成任何健康判断）",
            n_events=n_events,
            n_valid=n_valid,
            baseline_bpm=baseline_bpm,
            weights=w,
        )

    t50 = t50_median(events)
    slope = habituation_slope(events)
    sigma = consistency_sigma(events)
    if t50 is None or slope is None or sigma is None:
        return _unavailable(
            "跨事件统计量不可用（有效观测不足），本项不计入",
            n_events=n_events,
            n_valid=n_valid,
            baseline_bpm=baseline_bpm,
            weights=w,
        )

    components = {
        "S_recovery": calibrate(t50, cal.t50, higher_is_better=False),
        "S_habituation": calibrate(-slope, cal.habituation),
        "S_consistency": calibrate(-sigma, cal.consistency),
    }
    score = sum(COMPONENT_WEIGHTS[k] * components[k] for k in COMPONENT_WEIGHTS)
    score = _clamp(score, 0.0, 100.0)

    applied = W_MAX * gate(r)
    return RegulationResult(
        available=True,
        reason=None,
        baseline_bpm=baseline_bpm,
        n_events=n_events,
        n_valid=n_valid,
        t50_median_s=t50,
        habituation_slope=slope,
        reactivity_ratio_mean=reactivity_ratio_mean(events),
        consistency_sigma=sigma,
        components=components,
        score=score,
        reliability=r,
        weight_applied=applied,
        redistributed_to=redistribute(w, PHYSIO_DIM, W_MAX - applied),
    )
