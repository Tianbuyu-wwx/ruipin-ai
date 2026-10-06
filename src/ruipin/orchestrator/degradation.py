"""降级阶梯（方案 §3.4）—— 显式、可见、可审计。

红线：任何 level 下都**不生成未经真实计算的分数**。
降级只影响"哪些能力还在"，绝不产生兜底数值；报告中每个维度都携带
provider + confidence + degraded 三元组。

另一条红线：**禁止静默降级**。每一级都必须有中文 UI 徽标文案，
Level 0 为"无徽标"（正常态，不需要打扰用户）。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

MAX_LEVEL = 6


@dataclass(frozen=True)
class DegradationLevel:
    level: int
    name: str
    description: str
    ui_badge: str  # 中文徽标文案；Level 0 为空串（正常无徽标）


LEVELS: tuple[DegradationLevel, ...] = (
    DegradationLevel(0, "normal", "全功能运行", ""),
    DegradationLevel(1, "no_vision", "关闭视觉维度，仅文本评估", "本轮未做视觉分析"),
    DegradationLevel(2, "rubric_rule", "评分降级为规则评分（标注 provider=rubric）", "本轮为规则评分，仅供参考"),
    DegradationLevel(3, "asr_failed", "ASR 失败，仅文本评估并标注无语音", "未获取语音"),
    DegradationLevel(4, "tts_unavailable", "TTS 不可用，纯文本呈现不放音", "无声，文本照常"),
    DegradationLevel(5, "avatar_failed", "形象渲染失败，回退静态形象", "静态形象"),
    DegradationLevel(6, "fatal", "严重故障：只保存答案，事后补评估", "评估稍后生成"),
)

# 原因码 -> 降级等级。**新原因必须显式登记**，未登记的原因会在 escalate 时报错，
# 这是"禁止静默降级"的机械保证：无法把一次故障塞进一个没有徽标的角落。
REASON_LEVELS: dict[str, int] = {
    # L1 视觉
    "vision_timeout": 1,
    "vision_unavailable": 1,
    "vision_budget_exhausted": 1,
    # L2 评分
    "scoring_failed": 2,
    "scoring_timeout": 2,
    "scoring_unretryable": 2,
    "rubric_rule": 2,
    # L3 ASR
    "asr_failed": 3,
    "asr_timeout": 3,
    # L4 TTS
    "tts_unavailable": 4,
    "tts_timeout": 4,
    # L5 形象
    "avatar_failed": 5,
    # L6 严重故障
    "fatal": 6,
    "storage_failed": 6,
}


def level_of(level: int) -> DegradationLevel:
    if not 0 <= level <= MAX_LEVEL:
        raise ValueError(f"降级等级越界 [0,{MAX_LEVEL}]: {level}")
    return LEVELS[level]


def level_by_name(name: str) -> DegradationLevel:
    for lv in LEVELS:
        if lv.name == name:
            return lv
    raise KeyError(f"未知降级名称: {name}")


def badge_for(level: int) -> str:
    return level_of(level).ui_badge


@dataclass(frozen=True)
class DegradationEvent:
    """审计事件：一次降级升级的完整上下文。"""

    seq: int
    reason: str
    level: int
    level_name: str
    ui_badge: str
    detail: str
    ts: float


@dataclass
class DegradationTracker:
    """一会话一实例。记录当前等级与全部触发原因，供审计与 UI 徽标。

    `level` 取已发生事件的**最高等级**（保守口径：降级不因后续某轮恢复而被抹掉，
    会话级审计必须能看到"这场面试最差到过哪一级"）。
    """

    clock: Callable[[], float] = time.time
    _events: list[DegradationEvent] = field(default_factory=list)
    _seq: int = 0

    @property
    def level(self) -> int:
        return max((e.level for e in self._events), default=0)

    @property
    def current(self) -> DegradationLevel:
        return level_of(self.level)

    @property
    def degraded(self) -> bool:
        return self.level > 0

    @property
    def ui_badge(self) -> str:
        """当前等级对应的徽标文案（Level 0 为空串）。"""
        return self.current.ui_badge

    def escalate(
        self, reason: str, level: Optional[int] = None, detail: str = ""
    ) -> DegradationLevel:
        """升级一次降级。未登记的原因码必须显式给 level，否则抛 KeyError。"""
        lv = level if level is not None else REASON_LEVELS.get(reason)
        if lv is None:
            raise KeyError(
                f"未登记的降级原因: {reason!r}；请在 REASON_LEVELS 显式登记或传入 level"
            )
        target = level_of(lv)
        self._seq += 1
        self._events.append(
            DegradationEvent(
                seq=self._seq,
                reason=reason,
                level=target.level,
                level_name=target.name,
                ui_badge=target.ui_badge,
                detail=detail,
                ts=self.clock(),
            )
        )
        return target

    def events(self) -> list[DegradationEvent]:
        """审计视图（返回副本，外部不可改）。"""
        return list(self._events)

    def reasons(self) -> tuple[str, ...]:
        return tuple(e.reason for e in self._events)

    def badges(self) -> tuple[str, ...]:
        """已触发等级的徽标文案（去重、按等级升序、过滤空串）。"""
        seen: dict[int, str] = {}
        for e in self._events:
            if e.ui_badge:
                seen.setdefault(e.level, e.ui_badge)
        return tuple(seen[k] for k in sorted(seen))

    def latest(self) -> Optional[DegradationEvent]:
        return self._events[-1] if self._events else None
