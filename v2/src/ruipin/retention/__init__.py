"""数据保留（方案 §12.2）。

* `policy` —— 保留策略表（`RetentionPolicy`）、删除凭证（`DeletionReceipt`）、
  删除执行器（`RetentionEnforcer`）与删除端口（`PurgePort`）。

核心纪律：删除必须是**真删**并提供凭证；`PurgePort.delete` 返回数与待删数不一致时
产出 `verified=False`，绝不当作成功。
"""

from __future__ import annotations

from .policy import (
    AUDIT_RETAIN_DAYS,
    DAY_S,
    DEFAULT_REPORT_MONTHS,
    MAX_REPORT_MONTHS,
    MONTH_DAYS,
    DataClass,
    DeletionReceipt,
    PurgePort,
    RetentionAction,
    RetentionEnforcer,
    RetentionPolicy,
    RetentionRule,
)

__all__ = [
    "AUDIT_RETAIN_DAYS",
    "DAY_S",
    "DEFAULT_REPORT_MONTHS",
    "MAX_REPORT_MONTHS",
    "MONTH_DAYS",
    "DataClass",
    "DeletionReceipt",
    "PurgePort",
    "RetentionAction",
    "RetentionEnforcer",
    "RetentionPolicy",
    "RetentionRule",
]
