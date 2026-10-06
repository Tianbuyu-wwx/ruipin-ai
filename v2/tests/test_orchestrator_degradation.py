"""降级阶梯（方案 §3.4）—— 显式、可见、可审计。"""

from __future__ import annotations

import pytest

from ruipin.orchestrator.degradation import (
    LEVELS,
    MAX_LEVEL,
    REASON_LEVELS,
    DegradationEvent,
    DegradationLevel,
    DegradationTracker,
    badge_for,
    level_by_name,
    level_of,
)

CJK = lambda s: any("一" <= ch <= "鿿" for ch in s)  # noqa: E731


# --------------------------------------------------------------------------
# 阶梯定义
# --------------------------------------------------------------------------


def test_seven_levels_0_to_6_indexed_by_level():
    assert len(LEVELS) == 7
    for i, lv in enumerate(LEVELS):
        assert lv.level == i
        assert isinstance(lv, DegradationLevel)


def test_level_names_match_spec():
    assert [lv.name for lv in LEVELS] == [
        "normal",
        "no_vision",
        "rubric_rule",
        "asr_failed",
        "tts_unavailable",
        "avatar_failed",
        "fatal",
    ]


def test_descriptions_are_non_empty_chinese():
    for lv in LEVELS:
        assert lv.description
        assert CJK(lv.description)


def test_every_degraded_level_has_visible_chinese_badge():
    """禁止静默降级：L1–L6 必须有中文徽标；L0 正常态无徽标。"""
    assert LEVELS[0].ui_badge == ""
    for lv in LEVELS[1:]:
        assert lv.ui_badge, f"L{lv.level} 缺少徽标文案"
        assert CJK(lv.ui_badge)


def test_badges_are_distinct():
    badges = [lv.ui_badge for lv in LEVELS[1:]]
    assert len(set(badges)) == len(badges)


def test_level_of_and_badge_for():
    assert level_of(2).name == "rubric_rule"
    assert badge_for(3) == "未获取语音"
    assert badge_for(6) == "评估稍后生成"
    assert badge_for(0) == ""


def test_level_of_rejects_out_of_range():
    for bad in (-1, 7, 99):
        with pytest.raises(ValueError):
            level_of(bad)


def test_level_by_name():
    assert level_by_name("fatal").level == 6
    with pytest.raises(KeyError):
        level_by_name("nope")


def test_max_level_constant_consistent():
    assert MAX_LEVEL == 6 == LEVELS[-1].level


# --------------------------------------------------------------------------
# 原因码 -> 等级
# --------------------------------------------------------------------------


def test_every_reason_code_maps_to_valid_level():
    for reason, lv in REASON_LEVELS.items():
        assert 0 <= lv <= MAX_LEVEL, reason
        assert level_of(lv).name


def test_every_degraded_level_is_reachable_by_some_reason():
    reachable = set(REASON_LEVELS.values())
    for lv in range(1, MAX_LEVEL + 1):
        assert lv in reachable, f"L{lv} 没有任何原因码，无法触发"


def test_scoring_and_vision_reasons_land_on_expected_levels():
    assert REASON_LEVELS["vision_timeout"] == 1
    assert REASON_LEVELS["vision_budget_exhausted"] == 1
    assert REASON_LEVELS["scoring_failed"] == 2
    assert REASON_LEVELS["rubric_rule"] == 2
    assert REASON_LEVELS["asr_failed"] == 3
    assert REASON_LEVELS["tts_unavailable"] == 4
    assert REASON_LEVELS["avatar_failed"] == 5
    assert REASON_LEVELS["fatal"] == 6


# --------------------------------------------------------------------------
# Tracker
# --------------------------------------------------------------------------


def test_fresh_tracker_is_level_zero():
    t = DegradationTracker()
    assert t.level == 0
    assert t.degraded is False
    assert t.ui_badge == ""
    assert t.events() == []
    assert t.reasons() == ()
    assert t.latest() is None


def test_escalate_raises_level_and_returns_level_object():
    t = DegradationTracker()
    lv = t.escalate("vision_timeout")
    assert lv.level == 1 and lv.name == "no_vision"
    assert t.level == 1 and t.degraded is True
    assert t.ui_badge == "本轮未做视觉分析"


def test_escalation_path_and_max_semantics():
    t = DegradationTracker()
    t.escalate("vision_timeout")
    t.escalate("scoring_failed")
    t.escalate("asr_failed")
    assert t.level == 3
    # 更低等级的后续事件不会把"最差等级"抹掉（会话级审计口径）
    t.escalate("vision_timeout")
    assert t.level == 3
    assert t.reasons() == ("vision_timeout", "scoring_failed", "asr_failed", "vision_timeout")


def test_events_are_auditable():
    clock = [1000.0]
    t = DegradationTracker(clock=lambda: clock[0])
    clock[0] += 1
    t.escalate("scoring_failed", detail="llm timeout after 3.5s")
    clock[0] += 1
    t.escalate("fatal", detail="storage down")

    events = t.events()
    assert len(events) == 2
    for i, e in enumerate(events, start=1):
        assert isinstance(e, DegradationEvent)
        assert e.seq == i
        assert e.ui_badge, "审计事件必须带上当时的徽标文案"
    assert events[0].reason == "scoring_failed"
    assert events[0].level == 2
    assert events[0].level_name == "rubric_rule"
    assert events[0].detail == "llm timeout after 3.5s"
    assert events[0].ts == 1001.0
    assert events[1].level == 6 and events[1].ts == 1002.0


def test_events_returns_a_copy():
    t = DegradationTracker()
    t.escalate("fatal")
    snapshot = t.events()
    snapshot.clear()
    assert len(t.events()) == 1


def test_unknown_reason_must_be_registered_or_explicit():
    """未登记的原因 = 想静默降级，直接拒绝。"""
    t = DegradationTracker()
    with pytest.raises(KeyError):
        t.escalate("something_new")
    # 显式给 level 则放行（新原因必须先想清楚落在哪一级）
    assert t.escalate("something_new", level=2).level == 2


def test_explicit_level_overrides_mapping():
    t = DegradationTracker()
    assert t.escalate("scoring_failed", level=6).level == 6


def test_invalid_level_rejected():
    t = DegradationTracker()
    with pytest.raises(ValueError):
        t.escalate("custom", level=9)


def test_badges_deduped_and_ordered():
    t = DegradationTracker()
    t.escalate("scoring_failed")
    t.escalate("scoring_failed")
    t.escalate("vision_timeout")
    t.escalate("fatal")
    assert t.badges() == (
        "本轮未做视觉分析",
        "本轮为规则评分，仅供参考",
        "评估稍后生成",
    )


def test_latest_event():
    t = DegradationTracker()
    t.escalate("vision_timeout")
    t.escalate("tts_unavailable")
    assert t.latest().reason == "tts_unavailable"
    assert t.latest().ui_badge == "无声，文本照常"
