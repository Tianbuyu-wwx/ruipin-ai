"""后端编排层（方案 §3）。

`degradation` —— 降级阶梯：显式、可见、可审计，每级都有中文 UI 徽标。
`scheduler`   —— 每轮流水线：ASR → 评分（+1 次重试）→ 追问决策，
                 全程 deadline 判定，投机预生成下一题与评分并发。

红线：编排器**永不合成/default 分数**；评估不可用时 `TurnOutcome.eval is None`。
"""

from .degradation import (
    LEVELS,
    REASON_LEVELS,
    DegradationEvent,
    DegradationLevel,
    DegradationTracker,
    badge_for,
    level_by_name,
    level_of,
)
from .scheduler import (
    MAX_SCORING_RETRIES,
    StepDeadlines,
    StepTimeout,
    TurnContext,
    TurnOutcome,
    TurnScheduler,
)
from .service import (
    DEFAULT_REACTIVITY_THRESHOLD,
    RUBRIC_VERSION,
    Consent,
    InterviewConfig,
    InterviewService,
    PhysioQuality,
    Report,
    TurnRecord,
)

__all__ = [
    "LEVELS",
    "REASON_LEVELS",
    "DegradationEvent",
    "DegradationLevel",
    "DegradationTracker",
    "badge_for",
    "level_by_name",
    "level_of",
    "MAX_SCORING_RETRIES",
    "StepDeadlines",
    "StepTimeout",
    "TurnContext",
    "TurnOutcome",
    "TurnScheduler",
    "DEFAULT_REACTIVITY_THRESHOLD",
    "RUBRIC_VERSION",
    "Consent",
    "InterviewConfig",
    "InterviewService",
    "PhysioQuality",
    "Report",
    "TurnRecord",
]
