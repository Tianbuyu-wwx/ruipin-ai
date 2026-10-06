"""`adapters/avatar.py`（2D rig 口型逻辑）的测试。

防的回归
--------
1. **口型映射漂移**：`viseme_from_word` / `timeline_from_words` 是下游 rig 的唯一
   驱动来源；映射表被误改会让所有形象系统性错位。
2. **时间轴边界错误**：早于首帧 / 两帧之间 / 超出末帧若处理不当，会表现为"说话时
   嘴巴乱动或不动"。
3. **同步校验失效**：`SyncChecker` 若不报警，音画漂移就没人拦。

环境无 pytest-asyncio，且本模块为纯同步逻辑，直接调用即可。
"""

from __future__ import annotations

from ruipin.adapters import avatar as av
from ruipin.ports import VisemeEvent, WordTiming


def test_ascii_word_maps_by_dominant_vowel():
    """防回归：ASCII 词按"开口度最大的元音"取 dominant viseme。"""
    assert av.viseme_from_word("map") == av.VISEME_A
    assert av.viseme_from_word("book") == av.VISEME_O
    assert av.viseme_from_word("see") == av.VISEME_E
    assert av.viseme_from_word("pick") == av.VISEME_I


def test_chinese_word_maps_by_curated_table():
    """防回归：收录汉字按主元音归类；"你好" 主导开口在 好(A)、"世界" 在 界(E)，
    "技术" 的主导开口在 术(u)。"""
    assert av.viseme_from_word("你好") == av.VISEME_A
    assert av.viseme_from_word("世界") == av.VISEME_E
    assert av.viseme_from_word("技术") == av.VISEME_U


def test_empty_and_non_letter_words_are_silence():
    """防回归：空串 / 纯数字 / 纯标点不驱动口型（返回静音，而不是乱动）。"""
    assert av.viseme_from_word("") == av.VISEME_SILENCE
    assert av.viseme_from_word("123") == av.VISEME_SILENCE
    assert av.viseme_from_word("，。！") == av.VISEME_SILENCE


def test_unmapped_cjk_falls_back_to_generic_consonant():
    """防回归：未收录汉字兜底为 CONS（嘴微动），而不是沉默不动。"""
    assert av.is_cjk("鑫") is True
    assert av.is_cjk("A") is False
    assert av.viseme_from_word("鑫") == av.VISEME_CONS


def test_all_visemes_have_openness():
    """防回归：新增 viseme 却忘了给开口度，会让 dominant 选择退化为静音。"""
    for v in av.ALL_VISEMES:
        assert v in av.VISEME_OPENNESS


def test_timeline_from_words_produces_expected_sequence():
    """防回归：词时间轴 → viseme 序列（t_ms 与词同源，末尾补静音闭合）。"""
    words = [WordTiming("你好", 0, 300), WordTiming("世界", 300, 600)]
    events = av.timeline_from_words(words)
    assert events == (
        VisemeEvent(t_ms=0, viseme=av.VISEME_A, weight=1.0),
        VisemeEvent(t_ms=300, viseme=av.VISEME_E, weight=0.7),
        VisemeEvent(t_ms=600, viseme=av.VISEME_SILENCE, weight=0.0),
    )


def test_timeline_from_words_empty_returns_empty():
    """防回归：没有词时间轴时返回空元组——**不编造**任何口型。"""
    assert av.timeline_from_words([]) == ()


def test_build_timeline_sorts_and_sets_brow():
    """防回归：关键帧按 t_ms 排序；情绪标签映射到眉毛位移；未知情绪按 neutral。"""
    unsorted = [VisemeEvent(200, av.VISEME_O), VisemeEvent(0, av.VISEME_A)]
    tl = av.build_timeline(unsorted, 300, emotion="encouraging")
    assert [f.t_ms for f in tl.frames] == [0, 200]
    assert tl.brow == av.BROW_BY_EMOTION["encouraging"]
    assert tl.duration_ms == 300

    tl_unknown = av.build_timeline([], 100, emotion="angry")
    assert tl_unknown.brow == 0.0


def _sample_timeline() -> av.RigTimeline:
    return av.build_timeline(
        [
            VisemeEvent(0, av.VISEME_A, 1.0),
            VisemeEvent(100, av.VISEME_O, 0.5),
            VisemeEvent(200, av.VISEME_SILENCE, 0.0),
        ],
        200,
    )


def test_frame_at_before_first_frame_clamps():
    """防回归：t 早于首帧时 clamp 到首帧（否则会出现"无口型"空窗）。"""
    frame = _sample_timeline().frame_at(-50)
    assert frame.viseme == av.VISEME_A
    assert frame.weight == 1.0


def test_frame_at_between_frames_interpolates_weight():
    """防回归：两帧之间 weight 线性插值（60–120 ms 过渡），口型取左帧。"""
    frame = _sample_timeline().frame_at(50)
    assert frame.viseme == av.VISEME_A
    assert frame.weight == 0.75  # 1.0 -> 0.5 的中点
    assert frame.t_ms == 50


def test_frame_at_exact_keyframe_returns_that_keyframe():
    """防回归：正好落在关键帧上时返回该帧本身（不被插值改写）。"""
    frame = _sample_timeline().frame_at(100)
    assert frame.viseme == av.VISEME_O
    assert frame.weight == 0.5


def test_frame_at_between_later_frames_interpolates():
    """防回归：落在靠后的两帧之间时同样正确插值（不是只处理首段）。"""
    frame = _sample_timeline().frame_at(150)
    assert frame.viseme == av.VISEME_O
    assert frame.weight == 0.25  # 0.5 -> 0.0 的中点


def test_frame_at_after_last_frame_clamps():
    """防回归：t 超出末帧时 clamp（说话结束后维持在末帧静音）。"""
    frame = _sample_timeline().frame_at(999)
    assert frame.viseme == av.VISEME_SILENCE
    assert frame.weight == 0.0


def test_frame_at_empty_timeline_returns_silence():
    """防回归：空时间轴返回静音帧，而不是抛异常。"""
    tl = av.RigTimeline(frames=(), duration_ms=0)
    frame = tl.frame_at(10)
    assert frame.viseme == av.VISEME_SILENCE
    assert frame.weight == 0.0


def test_blink_window_and_disabled():
    """防回归：眨眼窗口确定性；周期/时长非正时关闭眨眼。"""
    tl = av.RigTimeline(frames=(), duration_ms=0, blink_period_ms=4000, blink_duration_ms=120)
    assert tl.blink_at(0) == 0.0
    assert tl.blink_at(2000) == 0.0
    assert tl.blink_at(3900) == 1.0

    no_blink = av.RigTimeline(frames=(), duration_ms=0, blink_period_ms=0)
    assert no_blink.blink_at(3900) == 0.0


def test_sync_checker_passes_within_tolerance_and_alarms_beyond():
    """防回归：音画偏差超阈值必须报警（ok=False），阈值内通过。"""
    checker = av.SyncChecker(tolerance_ms=av.SYNC_TOLERANCE_MS)
    ok = checker.check(1000, 1030)
    assert ok.ok is True and ok.offset_ms == 30

    bad = checker.check(1000, 1060)
    assert bad.ok is False and bad.offset_ms == 60 and bad.tolerance_ms == 40
