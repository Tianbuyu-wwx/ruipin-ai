"""可观测性与成本治理（方案 §12.3）。

三个子模块各管一件事，互不依赖：

* `metrics` —— `InMemoryMeter`：`ruipin.ports.Meter` 的内存实现 / 测试替身，
  出延迟分位数、token / 成本分桶、计数器。纪律：空样本的分位数返回 `None`
  而不是 0。
* `cost` —— `PriceTable` + `CostLedger`：单价表（未知即抛错，绝不按 0 折算）
  与成本台账（硬顶、分供应商硬顶、单场成本推算）。纪律：超顶抛 `BudgetExceeded`
  且**不入账**。
* `slo` —— SLO 定义与违约检测。纪律：样本不足 = `unknown`，不许报"达标"。

共同原则：**测不准 = 不计入**，宁可返回 `None` / `unknown`，也不给乐观默认值。
"""

from ruipin.observability.cost import CostLedger, Price, PriceTable, UnknownPriceError
from ruipin.observability.metrics import (
    DEFAULT_MAX_SAMPLES,
    InMemoryMeter,
    linear_percentile,
    weighted_percentile,
)
from ruipin.observability.slo import (
    DEFAULT_MIN_SAMPLES,
    DEFAULT_SLOS,
    HUMAN_INTERVIEWER_GAP_MS,
    METRIC_TURN_LATENCY_MS,
    METRIC_TURN_WEIGHTED_LATENCY_MS,
    SLO,
    SLOBatch,
    SLOReport,
    SLOStatus,
    SLO_TURN_LATENCY_P50,
    SLO_TURN_LATENCY_P95,
    SLO_TURN_WEIGHTED_LATENCY_P95,
    check_all,
    evaluate_slo,
)
from ruipin.observability.session_trace import (
    Degradation,
    DimensionAttribution,
    DimensionContribution,
    MemoryTraceStore,
    ModelCall,
    SessionDebugSnapshot,
    SessionDebugView,
    StepTrace,
    TraceCollector,
    TraceEvent,
    UnknownDimension,
)

__all__ = [
    # metrics
    "DEFAULT_MAX_SAMPLES",
    "InMemoryMeter",
    "linear_percentile",
    "weighted_percentile",
    # cost
    "CostLedger",
    "Price",
    "PriceTable",
    "UnknownPriceError",
    # slo
    "DEFAULT_MIN_SAMPLES",
    "DEFAULT_SLOS",
    "HUMAN_INTERVIEWER_GAP_MS",
    "METRIC_TURN_LATENCY_MS",
    "METRIC_TURN_WEIGHTED_LATENCY_MS",
    "SLO",
    "SLOBatch",
    "SLOReport",
    "SLOStatus",
    "SLO_TURN_LATENCY_P50",
    "SLO_TURN_LATENCY_P95",
    "SLO_TURN_WEIGHTED_LATENCY_P95",
    "check_all",
    "evaluate_slo",
    # session_trace
    "Degradation",
    "DimensionAttribution",
    "DimensionContribution",
    "MemoryTraceStore",
    "ModelCall",
    "SessionDebugSnapshot",
    "SessionDebugView",
    "StepTrace",
    "TraceCollector",
    "TraceEvent",
    "UnknownDimension",
]

# ---------- 追加：告警规则（alerts）与每日对账（reconciliation）----------
# 说明：以上为既有内容，未做改动；以下为 §12.3「告警 / 每日对账」两块的补齐。

from ruipin.observability.alerts import (  # noqa: E402  (追加在文件末尾，见上)
    DEFAULT_MAX_SD,
    DEFAULT_MIN_LEVEL,
    DEFAULT_QUEUE_WAIT_MS,
    DEFAULT_SUSTAINED_S,
    DEFAULT_WARN_RATIO,
    RULE_DEGRADATION,
    RULE_GROUP_DRIFT,
    RULE_QUEUE_WAIT,
    RULE_SESSION_COST,
    Alert,
    AlertBatch,
    AlertEvaluator,
    AlertSink,
    DegradationWatch,
    GroupScoreDriftWatch,
    MemoryAlertSink,
    QueueWaitWatch,
    SessionCostWatch,
    Severity,
)
from ruipin.observability.reconciliation import (  # noqa: E402
    DEFAULT_TOLERANCE,
    RULE_DAILY_RECONCILIATION,
    ReconcileReport,
    ReconcileVerdict,
    Reconciler,
    reconcile,
)

__all__ += [
    # alerts
    "DEFAULT_MAX_SD",
    "DEFAULT_MIN_LEVEL",
    "DEFAULT_QUEUE_WAIT_MS",
    "DEFAULT_SUSTAINED_S",
    "DEFAULT_WARN_RATIO",
    "RULE_DEGRADATION",
    "RULE_GROUP_DRIFT",
    "RULE_QUEUE_WAIT",
    "RULE_SESSION_COST",
    "Alert",
    "AlertBatch",
    "AlertEvaluator",
    "AlertSink",
    "DegradationWatch",
    "GroupScoreDriftWatch",
    "MemoryAlertSink",
    "QueueWaitWatch",
    "SessionCostWatch",
    "Severity",
    # reconciliation
    "DEFAULT_TOLERANCE",
    "RULE_DAILY_RECONCILIATION",
    "ReconcileReport",
    "ReconcileVerdict",
    "Reconciler",
    "reconcile",
]
