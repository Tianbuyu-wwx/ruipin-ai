"""虚拟面试官形象（2D rig）的口型 / 表情驱动逻辑 —— **纯逻辑，零 GPU**。

方案出处（`docs/锐聘AI-v2-技术方案.md`）
--------------------------------------
* **§6.4**：神经说话头实时性不达标——公开基准显示 MuseTalk 512² @ RTX 3090 约
  **0.5 s/帧（≈2 FPS）**，1024² 需 12.5 GB 显存且更慢，**无法用于"边说边生成"**。
  故 Phase 1–3 只做 **2D 骨骼 rig**（Live2D / 分层 PNG + 形变网格，60 FPS、CPU 可跑）。
  本模块只产出 rig 的**驱动数据**（viseme 关键帧 + 眨眼/眉毛），**不做任何神经推理**。
* **§6.5**：口型由 word/phoneme 时间戳 → viseme 时间轴；每个 viseme 带 `weight`
  做 60–120 ms 过渡插值；音画必须共用**同一时钟**（`AudioContext.currentTime`，
  不是 `Date.now()`），否则缓冲抖动会累积成漂移。`SyncChecker` 即该校验。

viseme 映射是**粗略近似**（务必知悉）
------------------------------------
纯 Python 里不引入拼音/音素库，故只按字符的**主元音开口度**归到一个低维口型集合。
它足以驱动卡通 2D rig 的"张合"观感，但**不是**音素级口型：一个词只得到一个
dominant viseme，不还原词内的音素序列。任何"精确口型"承诺前必须先替换 `CHAR_VISEME`
与 `_ASCII_LETTER_VISEME` 这两张表（届时用真实音素→viseme 映射）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from ..ports import VisemeEvent, WordTiming

# ---------- viseme 集合（低维：卡通 2D rig 够用） ----------

#: 闭口 / 静音。
VISEME_SILENCE = "sil"
#: 张口（主元音 a / ang / an …）。
VISEME_A = "A"
#: 半开口（e / en / ei / üe …）。
VISEME_E = "E"
#: 扁齿（i / in / ing …）。
VISEME_I = "I"
#: 圆唇（o / ong / ou …）。
VISEME_O = "O"
#: 撮口（u / ü …）。
VISEME_U = "U"
#: 双唇闭（b / p / m）。
VISEME_MBP = "MBP"
#: 唇齿（f / v）。
VISEME_FV = "FV"
#: 通用辅音（未细分），用于未收录汉字与多数辅音字母。
VISEME_CONS = "CONS"

ALL_VISEMES: tuple[str, ...] = (
    VISEME_SILENCE,
    VISEME_MBP,
    VISEME_FV,
    VISEME_CONS,
    VISEME_I,
    VISEME_U,
    VISEME_E,
    VISEME_O,
    VISEME_A,
)

#: 每个 viseme 的**开口度**（0=闭、1=最大张）。用作 `VisemeEvent.weight`，
#: 也是"取一个词的主导口型"时的排序键。数值为粗略经验值，非实测标定。
VISEME_OPENNESS: dict[str, float] = {
    VISEME_SILENCE: 0.0,
    VISEME_MBP: 0.05,
    VISEME_FV: 0.30,
    VISEME_CONS: 0.40,
    VISEME_I: 0.55,
    VISEME_U: 0.60,
    VISEME_E: 0.70,
    VISEME_O: 0.80,
    VISEME_A: 1.00,
}

# ---------- 字符 → viseme 的两张表（粗略近似，见模块 docstring） ----------

#: ASCII 字母 → viseme。元音按字母直接映射；b/p/m 闭唇、f/v 唇齿，其余为辅音兜底。
_ASCII_LETTER_VISEME: dict[str, str] = {
    **{c: VISEME_A for c in "a"},
    **{c: VISEME_E for c in "e"},
    **{c: VISEME_I for c in "i"},
    **{c: VISEME_O for c in "o"},
    **{c: VISEME_U for c in "u"},
    **{c: VISEME_MBP for c in "bpm"},
    **{c: VISEME_FV for c in "fv"},
}

#: 常用汉字 → viseme（按主元音粗略归类）。**不完整**，未收录汉字一律兜底为
#: `VISEME_CONS`。新增字只需在此追加；这张表就是"口型表"的唯一事实来源。
CHAR_VISEME: dict[str, str] = {
    # i 类（扁齿）
    "你": VISEME_I, "是": VISEME_I, "请": VISEME_I, "一": VISEME_I,
    "自": VISEME_I, "己": VISEME_I, "司": VISEME_I, "经": VISEME_I,
    "历": VISEME_I, "技": VISEME_I, "系": VISEME_I, "计": VISEME_I,
    "日": VISEME_I, "题": VISEME_I, "起": VISEME_I,
    # e 类（半开）
    "的": VISEME_E, "介": VISEME_E, "为": VISEME_E, "什": VISEME_E,
    "么": VISEME_E, "何": VISEME_E, "解": VISEME_E, "决": VISEME_E,
    "队": VISEME_E, "协": VISEME_E, "最": VISEME_E, "结": VISEME_E,
    "设": VISEME_E, "月": VISEME_E, "学": VISEME_E, "生": VISEME_E,
    "问": VISEME_E, "界": VISEME_E,
    # o 类（圆唇）
    "我": VISEME_O, "公": VISEME_O, "中": VISEME_O, "作": VISEME_O,
    "构": VISEME_O, "统": VISEME_O, "说": VISEME_O, "过": VISEME_O,
    # u 类（撮口）
    "目": VISEME_U, "遇": VISEME_U, "如": VISEME_U, "术": VISEME_U,
    "数": VISEME_U, "据": VISEME_U, "组": VISEME_U, "读": VISEME_U,
    # a 类（张口）
    "好": VISEME_A, "绍": VISEME_A, "下": VISEME_A, "想": VISEME_A,
    "来": VISEME_A, "项": VISEME_A, "到": VISEME_A, "大": VISEME_A,
    "挑": VISEME_A, "战": VISEME_A, "团": VISEME_A, "难": VISEME_A,
    "点": VISEME_A, "算": VISEME_A, "法": VISEME_A, "年": VISEME_A,
    "答": VISEME_A, "案": VISEME_A, "吗": VISEME_A,
}

#: 情绪标签 → 眉毛位移（负=压眉/专注，正=挑眉/鼓励）。来源：方案 §6.5 的
#: 三路叠加中"LLM 输出的情绪标签（neutral/encouraging/probing）"。
BROW_BY_EMOTION: dict[str, float] = {
    "neutral": 0.0,
    "encouraging": 0.2,
    "probing": -0.2,
}


def is_cjk(ch: str) -> bool:
    """是否 CJK 统一表意文字（用于判断"未收录汉字"的兜底路径）。"""
    return "\u4e00" <= ch <= "\u9fff"


def _char_viseme(ch: str) -> Optional[str]:
    """单个字符 → viseme；返回 None 表示该字符不驱动口型（标点/空白/数字）。"""
    mapped = CHAR_VISEME.get(ch)
    if mapped is not None:
        return mapped
    lowered = ch.lower()
    mapped = _ASCII_LETTER_VISEME.get(lowered)
    if mapped is not None:
        return mapped
    if is_cjk(ch):
        # 未收录汉字：粗略兜底为通用辅音，避免整词口型不动（比沉默更接近真实）。
        return VISEME_CONS
    return None


def viseme_from_word(word: str) -> str:
    """一个词 → 单个 dominant viseme（取词内**开口度最大**的字符）。

    词内无任何可映射字符（空串 / 纯数字 / 纯标点）时返回 `VISEME_SILENCE`。
    """
    best = VISEME_SILENCE
    best_openness = -1.0
    for ch in word:
        viseme = _char_viseme(ch)
        if viseme is None:
            continue
        openness = VISEME_OPENNESS[viseme]
        if openness > best_openness:
            best_openness = openness
            best = viseme
    return best


def timeline_from_words(words: Iterable[WordTiming]) -> tuple[VisemeEvent, ...]:
    """词时间轴 → viseme 时间轴（每个词一个事件，末尾补一个静音闭合事件）。

    产出的事件 `t_ms` 与 `WordTiming` **同源**（都相对音频起点），供 `SpeechPlan`
    直接下发。没有词时间轴时返回空元组——**不编造**。
    """
    word_list = list(words)
    events: list[VisemeEvent] = []
    for w in word_list:
        viseme = viseme_from_word(w.word)
        events.append(
            VisemeEvent(t_ms=int(w.start_ms), viseme=viseme, weight=VISEME_OPENNESS[viseme])
        )
    if word_list:
        # 末词结束后闭口，避免音频结束而嘴型停在张开状态。
        events.append(
            VisemeEvent(
                t_ms=int(word_list[-1].end_ms),
                viseme=VISEME_SILENCE,
                weight=VISEME_OPENNESS[VISEME_SILENCE],
            )
        )
    return tuple(events)


# ---------- rig 关键帧序列 ----------

#: 眨眼周期与时长（ms）。周期为粗略经验值，仅用于消除"死人感"（方案 §6.5）。
DEFAULT_BLINK_PERIOD_MS = 4000
DEFAULT_BLINK_DURATION_MS = 120


@dataclass(frozen=True)
class RigFrame:
    """某一时刻的形象状态。

    `viseme`/`weight` 是口型；`blink` ∈ [0,1] 是眨眼（1=完全闭合）；`brow` 是
    眉毛位移（负=压眉、正=挑眉）。
    """

    t_ms: int
    viseme: str
    weight: float = 1.0
    blink: float = 0.0
    brow: float = 0.0


@dataclass(frozen=True)
class RigTimeline:
    """rig 关键帧序列 + 时长；`frame_at` 给出任意时刻的形象状态。

    眨眼按周期**确定性**推导（不依赖随机数），保证测试可复现；真实产品若需要
    自然随机微动，应在客户端渲染层加、不影响这里的时间轴契约。
    """

    frames: tuple[RigFrame, ...]
    duration_ms: int
    brow: float = 0.0
    blink_period_ms: int = DEFAULT_BLINK_PERIOD_MS
    blink_duration_ms: int = DEFAULT_BLINK_DURATION_MS

    def blink_at(self, t_ms: int) -> float:
        """周期内靠近周期末尾的 `blink_duration_ms` 窗口内眨眼。"""
        if self.blink_period_ms <= 0 or self.blink_duration_ms <= 0:
            return 0.0
        phase = t_ms % self.blink_period_ms
        return 1.0 if phase >= self.blink_period_ms - self.blink_duration_ms else 0.0

    def frame_at(self, t_ms: int) -> RigFrame:
        """取 `t_ms` 时刻的口型（含三种边界，见下）。

        * 早于首帧：clamp 到首帧；
        * 落在两帧之间：口型取左帧（step），`weight` 线性插值（60–120 ms 过渡）；
        * 超出末帧：clamp 到末帧；
        * 时间轴为空：返回静音帧。
        """
        blink = self.blink_at(t_ms)
        if not self.frames:
            return RigFrame(t_ms, VISEME_SILENCE, VISEME_OPENNESS[VISEME_SILENCE], blink, self.brow)

        for frame in self.frames:
            if frame.t_ms == t_ms:
                return RigFrame(t_ms, frame.viseme, frame.weight, blink, self.brow)

        first = self.frames[0]
        if t_ms < first.t_ms:
            return RigFrame(t_ms, first.viseme, first.weight, blink, self.brow)

        last = self.frames[-1]
        if t_ms > last.t_ms:
            return RigFrame(t_ms, last.viseme, last.weight, blink, self.brow)

        left = first
        right = last
        for i in range(len(self.frames) - 1):
            a, b = self.frames[i], self.frames[i + 1]
            if a.t_ms < t_ms < b.t_ms:
                left, right = a, b
                break
        # 走到这里必有 left.t_ms < t_ms < right.t_ms，故 span 恒 > 0。
        span = right.t_ms - left.t_ms
        ratio = (t_ms - left.t_ms) / span
        weight = left.weight + (right.weight - left.weight) * ratio
        return RigFrame(t_ms, left.viseme, weight, blink, self.brow)


def build_timeline(
    visemes: Iterable[VisemeEvent],
    duration_ms: int,
    *,
    emotion: str = "neutral",
    blink_period_ms: int = DEFAULT_BLINK_PERIOD_MS,
    blink_duration_ms: int = DEFAULT_BLINK_DURATION_MS,
) -> RigTimeline:
    """`SpeechPlan.visemes` + 音频时长 → rig 关键帧序列。

    `emotion` 决定眉毛位移（未知标签按 neutral 处理，不臆造情绪）。
    """
    frames = tuple(
        RigFrame(t_ms=int(ev.t_ms), viseme=ev.viseme, weight=float(ev.weight))
        for ev in visemes
    )
    ordered = tuple(sorted(frames, key=lambda f: f.t_ms))
    return RigTimeline(
        frames=ordered,
        duration_ms=int(duration_ms),
        brow=BROW_BY_EMOTION.get(emotion, 0.0),
        blink_period_ms=int(blink_period_ms),
        blink_duration_ms=int(blink_duration_ms),
    )


# ---------- 音画同步校验（§6.5） ----------

#: 允许的音画时钟偏差（ms）。超过即报警（约 2–3 帧，人眼可辨）。
SYNC_TOLERANCE_MS = 40


@dataclass(frozen=True)
class SyncReport:
    """一次同步校验的结果。`ok=False` 即"报警"。"""

    ok: bool
    offset_ms: int
    tolerance_ms: int


class SyncChecker:
    """校验音频时钟与形象时钟是否同源。

    两路时钟必须来自**同一个音频时钟基准**（方案 §6.5）；此处只做"偏差是否
    超阈值"的判定，把"报警"变成可断言的数据，而不是靠肉眼看画面。
    """

    def __init__(self, tolerance_ms: int = SYNC_TOLERANCE_MS) -> None:
        self.tolerance_ms = int(tolerance_ms)

    def check(self, audio_clock_ms: float, avatar_clock_ms: float) -> SyncReport:
        offset = int(round(float(avatar_clock_ms) - float(audio_clock_ms)))
        return SyncReport(
            ok=abs(offset) <= self.tolerance_ms,
            offset_ms=offset,
            tolerance_ms=self.tolerance_ms,
        )


__all__ = [
    "ALL_VISEMES",
    "BROW_BY_EMOTION",
    "CHAR_VISEME",
    "DEFAULT_BLINK_DURATION_MS",
    "DEFAULT_BLINK_PERIOD_MS",
    "RigFrame",
    "RigTimeline",
    "SYNC_TOLERANCE_MS",
    "SyncChecker",
    "SyncReport",
    "VISEME_A",
    "VISEME_CONS",
    "VISEME_E",
    "VISEME_FV",
    "VISEME_I",
    "VISEME_MBP",
    "VISEME_O",
    "VISEME_OPENNESS",
    "VISEME_SILENCE",
    "VISEME_U",
    "build_timeline",
    "is_cjk",
    "timeline_from_words",
    "viseme_from_word",
]
