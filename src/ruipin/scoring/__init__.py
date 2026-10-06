"""评分模块（方案 M8 `evaluation` 的核心部分）。

- `rubric`：规则评分器（provider="rubric"，降级阶梯 L2 的真实评分，非兜底）
- `confidence`：置信度与降级判定
- `level`：分值 → A/B/C/D 等级
- `aggregate`：**可靠性 → 权重 → 重分配** 的通用聚合器（方案 N5）

纪律：任何 provider 失败必须抛 `Unavailable`，本模块不提供任何"默认分"入口。
"""

from .aggregate import (
    DEFAULT_WEIGHTS,
    DEGRADED_GATE_FACTOR,
    GATE_HIGH,
    GATE_LOW,
    AggregateResult,
    aggregate,
    default_gate,
)
from .confidence import (
    CONF_DOWNGRADE,
    CONF_GATE_HIGH,
    CONF_GATE_LOW,
    compute_confidence,
    length_evidence,
    should_downgrade,
)
from .level import LEVELS, Level, classify
from .rubric import RuleEvaluator

__all__ = [
    # rubric
    "RuleEvaluator",
    # confidence
    "compute_confidence", "should_downgrade", "length_evidence",
    "CONF_DOWNGRADE", "CONF_GATE_HIGH", "CONF_GATE_LOW",
    # level
    "classify", "Level", "LEVELS",
    # aggregate
    "aggregate", "default_gate", "AggregateResult", "DEFAULT_WEIGHTS",
    "GATE_LOW", "GATE_HIGH", "DEGRADED_GATE_FACTOR",
]
