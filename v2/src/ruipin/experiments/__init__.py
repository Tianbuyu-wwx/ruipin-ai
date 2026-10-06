"""A/B 实验与灰度发布（方案 §12.5）。

三个子模块各管一件事，`assign` → `exposure` → `registry` 串成一条完整链路：

* `assign` —— 会话级确定性分桶（sha256），并区分"未纳入实验"与"对照组"。
* `exposure` —— append-only 曝光日志，把"重复曝光"变成可查询的指标。
* `registry` —— 预注册（主指标 / 护栏 / 最小样本 / 停止规则）与四步判定，
  **护栏优先于主指标**，人工急停优先于一切。

共同纪律：样本不足 / 数据缺失一律显式返回"不可判定"，绝不用默认值或乐观结论糊过去。
"""

from ruipin.experiments.assign import CONTROL, ExperimentSpec, assign
from ruipin.experiments.exposure import ExposureRecord, ExposureSink, MemoryExposureLog
from ruipin.experiments.registry import (
    GUARDRAIL_APPEAL_RATE,
    GUARDRAIL_CLOSED_RATE,
    Decision,
    ExperimentRegistration,
    ExperimentRegistry,
    GuardrailBreach,
    GuardrailDirection,
    GuardrailMetric,
    MissingBaselineVariant,
    StopRule,
    UnknownExperiment,
    VariantObservation,
    Verdict,
)

__all__ = [
    # assign
    "CONTROL",
    "ExperimentSpec",
    "assign",
    # exposure
    "ExposureRecord",
    "ExposureSink",
    "MemoryExposureLog",
    # registry
    "GUARDRAIL_APPEAL_RATE",
    "GUARDRAIL_CLOSED_RATE",
    "Decision",
    "ExperimentRegistration",
    "ExperimentRegistry",
    "GuardrailBreach",
    "GuardrailDirection",
    "GuardrailMetric",
    "MissingBaselineVariant",
    "StopRule",
    "UnknownExperiment",
    "VariantObservation",
    "Verdict",
]
