"""内容安全与题库治理模块（方案 §12.4）。

统一出口。三个子模块：
- `classifier`：生成内容安全过滤（词表 + 属性评论句式 + 组合 fail-closed）；
- `bank`：题库治理与偏差审计；
- `selector`：受限追问模板 + 安全网关。

纪律：拿不到安全判定 == 不可放行；题库治理中"改过内容必须重审"；
样本不足不得用 0.0 冒充统计结论。任何能力失败抛 `Unavailable`，不合成兜底值。
"""

from .bank import (
    MIN_GROUP_SAMPLE,
    BiasAudit,
    GroupStat,
    InvalidReviewTransition,
    QuestionBank,
    QuestionBankItem,
    ReviewStatus,
)
from .classifier import (
    SAFETY_PRIORITY,
    AttributeCommentClassifier,
    CompositeSafetyClassifier,
    LexiconSafetyClassifier,
    SafetyCategory,
    SafetyClassifierPort,
    SafetyVerdict,
    severity_of,
)
from .selector import FollowupTemplate, TemplateSelector

__all__ = [
    # classifier
    "SafetyCategory",
    "SafetyVerdict",
    "SafetyClassifierPort",
    "LexiconSafetyClassifier",
    "AttributeCommentClassifier",
    "CompositeSafetyClassifier",
    "SAFETY_PRIORITY",
    "severity_of",
    # bank
    "ReviewStatus",
    "QuestionBankItem",
    "QuestionBank",
    "InvalidReviewTransition",
    "BiasAudit",
    "GroupStat",
    "MIN_GROUP_SAMPLE",
    # selector
    "FollowupTemplate",
    "TemplateSelector",
]
