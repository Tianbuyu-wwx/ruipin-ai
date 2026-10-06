"""生理信号子包：应激后生理恢复（Post-stress Physiological Recovery）。

对外构念仅允许表述为「应激后生理恢复」——描述可观测的生理现象，
不做任何心理特质或健康状态的推断（详见 docs/模块详设-心率与压力调节评估.md §1.5）。

三条硬纪律：
1. 弃权优于猜测：信号质量不足 → 不产出数值。
2. 禁止对单次事件下判断：只使用跨 ≥8 个有效事件的聚合统计量。
3. 测不准 = 不计入（权重归零并归还其他维度），不是给低分。
"""

from .collector import (
    BASELINE_MIN_SNR,
    BASELINE_REST_MS,
    BASELINE_TAIL_MS,
    PEAK_WINDOW_MS,
    PRE_WINDOW_MS,
    RECOVERY_WINDOW_MS,
    StressRegulationCollector,
)
from .metrics import (
    ARRHYTHMIA_CV_THRESHOLD,
    CENSORED_T50_S,
    MIN_DELTA_BPM,
    MIN_IBI_SAMPLES,
    MIN_VALID_EVENTS,
    RecoveryFit,
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
from .regulation import (
    COMPONENT_WEIGHTS,
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
from .window import (
    AGREE_TOL_BPM,
    DISAGREE_TOL_BPM,
    SNR_MIN,
    agreement_score,
    compute_snr,
    is_rejected,
    vote,
)

__all__ = [
    # window
    "AGREE_TOL_BPM",
    "DISAGREE_TOL_BPM",
    "SNR_MIN",
    "agreement_score",
    "compute_snr",
    "is_rejected",
    "vote",
    # metrics
    "ARRHYTHMIA_CV_THRESHOLD",
    "CENSORED_T50_S",
    "MIN_DELTA_BPM",
    "MIN_IBI_SAMPLES",
    "MIN_VALID_EVENTS",
    "RecoveryFit",
    "censored_share",
    "consistency_sigma",
    "detect_arrhythmia",
    "estimate_lambda",
    "estimate_t50",
    "fit_recovery",
    "habituation_slope",
    "reactivity_ratio_mean",
    "t50_median",
    "valid_count",
    # regulation
    "COMPONENT_WEIGHTS",
    "GATE_CEIL",
    "GATE_FLOOR",
    "PLACEHOLDER_CALIBRATION",
    "W_MAX",
    "PhysioCalibration",
    "calibrate",
    "evaluate_physio",
    "gate",
    "redistribute",
    "reliability",
    # collector
    "BASELINE_MIN_SNR",
    "BASELINE_REST_MS",
    "BASELINE_TAIL_MS",
    "PEAK_WINDOW_MS",
    "PRE_WINDOW_MS",
    "RECOVERY_WINDOW_MS",
    "StressRegulationCollector",
]
