"""SLO 定义与违约检测（方案 §12.3；阈值口径见方案自检 A2 的修订）。

设计意图
--------
方案自检 A2 把"越快越好"改成了**体验基准**：真人面试官的回应间隔是 1–3 秒，
所以本系统的单轮延迟目标是"落在真人节奏附近"，而不是越低越好——低于 1s 反而
显得机械，超过 8s 用户会以为卡死。据此定三条 SLO：

===================  =============  ======  ==================================
SLO                  指标           阈值     口径来源
===================  =============  ======  ==================================
单轮延迟 p50         turn.latency   ≤4000ms  方案 §12.3：单轮 p50 ≤ 4000ms
单轮延迟 p95         turn.latency   ≤8000ms  方案 §12.3：单轮 p95 ≤ 8000ms
单轮加权延迟 p95     turn.weighted  ≤7000ms  方案自检 A2：加权 p95 ≤ 7000ms
===================  =============  ======  ==================================

"加权 p95"比"总体 p95"更严（7000 < 8000）：它按流量构成加权（题型 / 供应商各占
多少），防止"大多数请求很快，但某一类题型常年很慢"被总体分位数平均掉。

关键纪律：样本不足 = unknown
----------------------------
样本不够时返回 `unknown`，**绝不返回 ok**。这与 `physio` 的"测不准即弃权"、
`metrics` 的"空样本返回 None 而不是 0"是同一条纪律：宁可承认不知道，也不
给一个乐观默认值让告警静音。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Optional, Protocol, runtime_checkable

__all__ = [
    "DEFAULT_MIN_SAMPLES",
    "HUMAN_INTERVIEWER_GAP_MS",
    "METRIC_TURN_LATENCY_MS",
    "METRIC_TURN_WEIGHTED_LATENCY_MS",
    "SLO",
    "SLOReport",
    "SLOBatch",
    "SLOStatus",
    "LatencySource",
    "DEFAULT_SLOS",
    "SLO_TURN_LATENCY_P50",
    "SLO_TURN_LATENCY_P95",
    "SLO_TURN_WEIGHTED_LATENCY_P95",
    "evaluate_slo",
    "check_all",
]

#: 判定所需的最小样本数。少于它一律 `unknown` —— 20 条以下的分位数没有意义。
DEFAULT_MIN_SAMPLES = 20

#: 真人面试官的回应间隔（ms）：体验基准，不是优化目标。低于 1s 显得机械，
#: 高于 3s 会被感知为迟疑；系统延迟目标以此为锚（方案自检 A2）。
HUMAN_INTERVIEWER_GAP_MS: tuple[int, int] = (1000, 3000)

#: 单轮延迟指标名（编排层用 `observe_ms` 埋点）。
METRIC_TURN_LATENCY_MS = "turn.latency_ms"
#: 按流量构成加权的单轮延迟指标名（用 `observe_weighted_ms` 埋点）。
METRIC_TURN_WEIGHTED_LATENCY_MS = "turn.weighted.latency_ms"


class SLOStatus(str, Enum):
    """SLO 状态。`unknown` 是**一等公民**：样本不足不许报 ok。"""

    OK = "ok"
    BREACHED = "breached"
    UNKNOWN = "unknown"


@runtime_checkable
class LatencySource(Protocol):
    """`evaluate_slo` 需要的分位数查询能力（比 `Meter` 多一层）。

    `InMemoryMeter` 天然满足；换成 Prometheus 适配器时，实现这四个方法即可
    （从 histogram / sketch 里读数）。
    """

    def count(self, name: str) -> int: ...
    def percentile(self, name: str, p: float) -> Optional[float]: ...
    def count_weighted(self, name: str) -> int: ...
    def percentile_weighted(self, name: str, p: float) -> Optional[float]: ...


@dataclass(frozen=True)
class SLO:
    """一条服务等级目标。阈值单位是 ms，作用于某个分位数。"""

    name: str
    metric: str
    threshold_ms: float
    percentile: float
    #: True → 走加权序列（`observe_weighted_ms` / `percentile_weighted`）。
    weighted: bool = False
    #: 判定所需最小样本数；`evaluate_slo` 的 `min_samples` 参数可覆盖。
    min_samples: int = DEFAULT_MIN_SAMPLES
    #: 方案出处（进告警文案，便于回溯"这条阈值哪来的"）。
    source: str = ""

    def __post_init__(self) -> None:
        if not 0.0 < self.percentile <= 1.0:
            raise ValueError(f"percentile 必须落在 (0, 1]：{self.percentile!r}")
        if self.threshold_ms <= 0:
            raise ValueError(f"threshold_ms 必须为正：{self.threshold_ms!r}")
        if self.min_samples < 1:
            raise ValueError(f"min_samples 必须 ≥ 1：{self.min_samples!r}")

    @property
    def label(self) -> str:
        """`p50` / `p95` 这类人类可读的分位数标签。"""
        return f"p{round(self.percentile * 100)}"


@dataclass(frozen=True)
class SLOReport:
    """单条 SLO 的判定结果。`message` 是**中文**的，可直接进告警。"""

    slo: SLO
    status: SLOStatus
    #: 实测分位数值（ms）；`unknown` 时为 `None`。
    observed_ms: Optional[float]
    samples: int
    min_samples: int
    message: str

    def as_dict(self) -> dict[str, Any]:
        """JSON 安全视图（`json.dumps(..., allow_nan=False)` 通过）。"""
        return {
            "name": self.slo.name,
            "metric": self.slo.metric,
            "label": self.slo.label,
            "weighted": self.slo.weighted,
            "threshold_ms": self.slo.threshold_ms,
            "status": self.status.value,
            "observed_ms": self.observed_ms,
            "samples": self.samples,
            "min_samples": self.min_samples,
            "source": self.slo.source,
            "message": self.message,
        }


@dataclass(frozen=True)
class SLOBatch:
    """一批 SLO 的判定结果。"""

    reports: tuple[SLOReport, ...]

    @property
    def any_breached(self) -> bool:
        return any(r.status is SLOStatus.BREACHED for r in self.reports)

    @property
    def any_unknown(self) -> bool:
        return any(r.status is SLOStatus.UNKNOWN for r in self.reports)

    def breached(self) -> tuple[SLOReport, ...]:
        return tuple(r for r in self.reports if r.status is SLOStatus.BREACHED)

    def summary(self) -> str:
        """一行汇总，便于直接贴进日志：`SLO: 1/3 违约, 1/3 样本不足`。"""
        n_breach = len(self.breached())
        n_unknown = sum(1 for r in self.reports if r.status is SLOStatus.UNKNOWN)
        return (
            f"SLO: {n_breach}/{len(self.reports)} 违约, "
            f"{n_unknown}/{len(self.reports)} 样本不足"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "any_breached": self.any_breached,
            "any_unknown": self.any_unknown,
            "summary": self.summary(),
            "reports": [r.as_dict() for r in self.reports],
        }


# ---------- 预置 SLO ----------

SLO_TURN_LATENCY_P50 = SLO(
    name="单轮延迟 p50",
    metric=METRIC_TURN_LATENCY_MS,
    threshold_ms=4000.0,
    percentile=0.50,
    source="方案 §12.3：单轮 p50 ≤ 4000ms（体验基准：真人间隔 1–3s）",
)

SLO_TURN_LATENCY_P95 = SLO(
    name="单轮延迟 p95",
    metric=METRIC_TURN_LATENCY_MS,
    threshold_ms=8000.0,
    percentile=0.95,
    source="方案 §12.3：单轮 p95 ≤ 8000ms",
)

SLO_TURN_WEIGHTED_LATENCY_P95 = SLO(
    name="单轮加权延迟 p95",
    metric=METRIC_TURN_WEIGHTED_LATENCY_MS,
    threshold_ms=7000.0,
    percentile=0.95,
    weighted=True,
    source="方案自检 A2 修订：加权 p95 ≤ 7000ms（按题型/流量构成加权，防长尾被平均掉）",
)

DEFAULT_SLOS: tuple[SLO, ...] = (
    SLO_TURN_LATENCY_P50,
    SLO_TURN_LATENCY_P95,
    SLO_TURN_WEIGHTED_LATENCY_P95,
)


# ---------- 判定 ----------


def evaluate_slo(
    meter: LatencySource, slo: SLO, min_samples: Optional[int] = None
) -> SLOReport:
    """判定一条 SLO。

    * 样本数 < `min_samples`（默认取 `slo.min_samples`）→ `unknown`。
    * 实测值 ≤ 阈值 → `ok`；> 阈值 → `breached`。
    * `meter` 不具备分位数查询能力 → 抛 `TypeError`（不许静默判 ok）。
    """
    need = slo.min_samples if min_samples is None else min_samples
    if need < 1:
        raise ValueError(f"min_samples 必须 ≥ 1：{need!r}")

    if slo.weighted:
        required = ("percentile_weighted", "count_weighted")
    else:
        required = ("percentile", "count")
    missing = [m for m in required if not callable(getattr(meter, m, None))]
    if missing:
        raise TypeError(
            f"meter {type(meter).__name__} 不支持 {missing}，无法评估 SLO "
            f"{slo.name!r}（不许静默判达标）"
        )

    if slo.weighted:
        observed = meter.percentile_weighted(slo.metric, slo.percentile)
        samples = meter.count_weighted(slo.metric)
    else:
        observed = meter.percentile(slo.metric, slo.percentile)
        samples = meter.count(slo.metric)

    label = slo.label
    if samples < need or observed is None:
        return SLOReport(
            slo=slo,
            status=SLOStatus.UNKNOWN,
            observed_ms=None,
            samples=samples,
            min_samples=need,
            message=(
                f"[UNKNOWN] {slo.name}（{slo.metric} {label}）：样本不足 "
                f"{samples}/{need} 条，不计入达标判定"
            ),
        )

    value = f"{observed:.1f}"
    threshold = f"{slo.threshold_ms:.1f}"
    if observed <= slo.threshold_ms:
        status = SLOStatus.OK
        relation = "≤"
    else:
        status = SLOStatus.BREACHED
        relation = ">"
    return SLOReport(
        slo=slo,
        status=status,
        observed_ms=observed,
        samples=samples,
        min_samples=need,
        message=(
            f"[{status.value.upper()}] {slo.name}（{slo.metric} {label}）："
            f"{label}={value}ms {relation} 阈值 {threshold}ms（样本 {samples} 条）"
        ),
    )


def check_all(
    meter: LatencySource,
    slos: Iterable[SLO] = DEFAULT_SLOS,
    min_samples: Optional[int] = None,
) -> SLOBatch:
    """逐条判定，返回汇总；空清单时 `any_breached` 为 False。"""
    reports = tuple(evaluate_slo(meter, slo, min_samples) for slo in slos)
    return SLOBatch(reports=reports)
