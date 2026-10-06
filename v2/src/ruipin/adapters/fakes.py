"""测试替身（fakes）：零外部依赖、完全确定性的端口实现。

纪律与生产适配器一致：**失败抛 `Unavailable`，绝不返回合成/default 分数**
（自检报告 §7：任一 provider 失败 → 不得出现合成分数）。
所以 `fail=True` 的 fake 只会抛异常，不会吐一个"看起来像真的"的 60 分。

所有 fake 都提供 `reset()`，便于在 fixture 之间复用同一个实例。
"""

from __future__ import annotations

import asyncio
import math
from typing import AsyncIterator, Optional

from ..domain.errors import Unavailable
from ..ports import EvalRequest, EvalResult, SpeechPlan, VisemeEvent, WordTiming

# 默认六维：刻意互不相同，便于断言"维度没有被串位/取平均抹平"
DEFAULT_DIMS: dict[str, float] = {
    "technical": 70.0,
    "communication": 65.0,
    "completeness": 60.0,
    "problem_solving": 68.0,
    "teamwork": 72.0,
    "leadership": 55.0,
}  # 均值恰为 65.0，Golden Master 断言用得上


def _score_of(dims: dict[str, float]) -> float:
    return round(sum(dims.values()) / len(dims), 2)


def _split_words(text: str) -> list[str]:
    """替身的时间轴切分规则：优先按空白切，没有空白（中文常见）时按**字符**切。

    这纯粹是为了让替身产出"词数随文本长度增长"的确定性时间轴，
    **不涉及任何真实分词或强制对齐能力**。
    """
    parts = [p for p in text.split() if p]
    if len(parts) <= 1 and len(text) > 1:
        return [ch for ch in text if not ch.isspace()]
    return parts


class FakeEvaluator:
    """`Evaluator` 的确定性替身。

    Args:
        dims: 六维分数；None 则使用 `DEFAULT_DIMS`（缺维度会在 `EvalResult`
            构造时抛 ValueError —— 校验在端口层，不在 fake 里放宽）。
        fail: True 时 `evaluate()` 抛 `Unavailable`。
        slow_s: >0 时先 await 相应秒数，用于超时路径测试。
        provider: 写入 `EvalResult.provider` 与 `self.name`。
        confidence / feedback / degraded / degrade_reason: 结果字段，可定制。
    """

    def __init__(
        self,
        dims: Optional[dict[str, float]] = None,
        fail: bool = False,
        slow_s: float = 0.0,
        provider: str = "fake",
        *,
        confidence: float = 0.9,
        feedback: str = "fake feedback",
        degraded: bool = False,
        degrade_reason: Optional[str] = None,
    ) -> None:
        self.dims: dict[str, float] = dict(dims) if dims else dict(DEFAULT_DIMS)
        self.fail = fail
        self.slow_s = slow_s
        self.provider = provider
        self.name = provider
        self.confidence = confidence
        self.feedback = feedback
        self.degraded = degraded
        self.degrade_reason = degrade_reason
        self.calls: list[EvalRequest] = []

    async def evaluate(self, req: EvalRequest) -> EvalResult:
        self.calls.append(req)
        if self.slow_s > 0:
            await asyncio.sleep(self.slow_s)
        if self.fail:
            raise Unavailable(self.provider, "injected failure by FakeEvaluator")
        return EvalResult(
            dims=dict(self.dims),
            score=_score_of(self.dims),
            feedback=self.feedback,
            provider=self.provider,
            confidence=self.confidence,
            latency_ms=0,
            cost_usd=0.0,
            degraded=self.degraded,
            degrade_reason=self.degrade_reason,
        )

    def reset(self) -> None:
        self.calls = []


class FakeVLMEvaluator(FakeEvaluator):
    """视觉（VLM）替身：在 `FakeEvaluator` 之上额外记录见过的帧。"""

    def __init__(
        self,
        dims: Optional[dict[str, float]] = None,
        fail: bool = False,
        slow_s: float = 0.0,
        provider: str = "fake-vlm",
        **kwargs: object,
    ) -> None:
        super().__init__(dims, fail, slow_s, provider, **kwargs)  # type: ignore[arg-type]
        self.frames_seen: list[list[object]] = []

    async def evaluate(
        self, req: EvalRequest, frames: Optional[list[object]] = None
    ) -> EvalResult:
        self.frames_seen.append(list(frames) if frames else [])
        return await super().evaluate(req)

    def reset(self) -> None:
        super().reset()
        self.frames_seen = []


class FakeTTS:
    """TTS 替身：`synth_stream(text)` 产出可分块的假音频字节，`synth()` 产出完整 `SpeechPlan`。

    **两个方法都必须有**：`TTSPort` 协议要求成对出现。只实现 `synth_stream` 时，
    上行链路（桥接层出题）会因为 `AttributeError: 'FakeTTS' object has no attribute 'synth'`
    而走进"非预期异常"分支——那会把"替身不完整"伪装成"TTS 故障"，
    实测极难分辨。所以这里补齐 `synth()`，让替身**完整满足协议**。

    Args:
        chunk_size: 名义块大小（仅用于决定分块数量，内容不是真实 PCM）。
        n_chunks: 显式指定块数，None 时由文本长度推导。
        fail: True 时 `synth_stream()` / `synth()` 抛 `Unavailable`（不产出任何字节）。
        provider: 写入 `SpeechPlan.provider` 与 `self.name`。
        word_ms: `synth()` 派生的词时间轴的每词时长（ms）。
        sample_rate: 写入 `SpeechPlan.sample_rate`。
        empty_audio: True 时 `synth()` 返回**空音频**的 plan。
            用于验证调用方"空音频也必须显式降级"——静音冒充成功是本项目红线。
    """

    def __init__(
        self,
        chunk_size: int = 32,
        n_chunks: Optional[int] = None,
        fail: bool = False,
        provider: str = "fake-tts",
        *,
        word_ms: int = 200,
        sample_rate: int = 24000,
        empty_audio: bool = False,
    ) -> None:
        self.chunk_size = chunk_size
        self.n_chunks = n_chunks
        self.fail = fail
        self.provider = provider
        self.name = provider
        self.word_ms = int(word_ms)
        self.sample_rate = int(sample_rate)
        self.empty_audio = bool(empty_audio)
        self.synth_calls: list[str] = []
        self.voices: list[str] = []

    async def synth_stream(self, text: str) -> AsyncIterator[bytes]:
        """返回异步字节流；失败时**在 await 处**就抛，不等到迭代。"""
        self.synth_calls.append(text)
        if self.fail:
            raise Unavailable(self.provider, "injected failure by FakeTTS")
        return self._gen(text)

    async def synth(self, text: str, *, voice: str = "default") -> SpeechPlan:
        """完整合成：音频 + 词时间轴 + 口型时间轴。

        `words` / `visemes` 由文本**确定性**派生（每个词一片）。这是替身数据，
        只用来验证"时间轴确实走到了下行帧"，**不代表任何真实对齐质量**。
        """
        self.synth_calls.append(text)
        self.voices.append(voice)
        if self.fail:
            raise Unavailable(self.provider, "injected failure by FakeTTS")
        chunks = [c async for c in self._gen(text)]
        audio = b"" if self.empty_audio else b"".join(chunks)
        words = tuple(
            WordTiming(word=w, start_ms=i * self.word_ms, end_ms=(i + 1) * self.word_ms)
            for i, w in enumerate(_split_words(text))
        )
        visemes = tuple(
            VisemeEvent(t_ms=w.start_ms, viseme=f"v{i % 8}", weight=1.0)
            for i, w in enumerate(words)
        )
        return SpeechPlan(
            audio=audio,
            words=words,
            visemes=visemes,
            sample_rate=self.sample_rate,
            provider=self.provider,
            timing_source="native",
        )

    async def _gen(self, text: str) -> AsyncIterator[bytes]:
        n = self.n_chunks
        if n is None:
            n = max(1, math.ceil(len(text) / max(1, self.chunk_size)))
        for i in range(n):
            yield f"{i:04d}|{text}".encode("utf-8")

    def reset(self) -> None:
        self.synth_calls = []
        self.voices = []


class DeterministicClock:
    """可控时钟：时间只在 `advance()` 时前进，供超时/预算类断言使用。"""

    def __init__(self, start: float = 0.0) -> None:
        self._start = float(start)
        self._t = float(start)

    @property
    def t(self) -> float:
        return self._t

    def monotonic(self) -> float:
        return self._t

    def now(self) -> float:
        """`time.time()` 的替身，与 monotonic 同源（测试里不需要区分）。"""
        return self._t

    def advance(self, dt: float) -> float:
        if dt < 0:
            raise ValueError(f"时间不能倒流: {dt}")
        self._t += float(dt)
        return self._t

    def set(self, t: float) -> None:
        self._t = float(t)

    def reset(self) -> None:
        self._t = self._start


__all__ = [
    "DEFAULT_DIMS",
    "DeterministicClock",
    "FakeEvaluator",
    "FakeTTS",
    "FakeVLMEvaluator",
]
