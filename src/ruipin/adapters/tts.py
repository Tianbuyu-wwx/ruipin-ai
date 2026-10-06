"""语音合成（TTS）适配器：**流式优先** + 强制对齐补齐词时间轴。

方案出处（`docs/锐聘AI-v2-技术方案.md`）
--------------------------------------
* **§6.1**：CosyVoice 2 为实时交互首选（~150 ms 首包 / 24 kHz / 非自回归尾部稳定）。
* **§6.3**：低延迟流式播放链路——LLM 出题 → 首句即送 TTS（不等整段）→
  流式产出 PCM 块 + word_ts（~150 ms 首块）→ Opus 编码 → WS → 客户端抖动缓冲。
* **§6.5**：**强制对齐是主方案**。CosyVoice2 的公开材料只确认了流式与 150 ms
  首包，**未确认**提供词级时间戳 API；因此不把"引擎是否输出时间戳"设为硬门槛，
  默认对**已合成音频**跑一遍强制对齐（ASR 词级对齐 / MFA / Whisper 词时间戳）得到
  `word_ts`。`SpeechPlan.timing_source` 区分 `native` / `forced_alignment` / `none`。

红线（本模块**不做**什么）
--------------------------
1. **绝不返回静音音频冒充成功**。网络失败 / 请求失败 / 空音频一律抛 `Unavailable`。
   遗留系统的头号缺陷就是"崩溃即返回一个看起来正常的默认值"，在语音里等价于
   "返回一段静音当成功"——调用方无法与真实静音区分。
2. **绝不凭文本长度线性编造时间轴**。没有 native 时间戳、也没有 `aligner` 时，
   `words` 为空、`timing_source="none"`。线性编造的时间轴会让口型**系统性错位**，
   且错得均匀、难以察觉——宁可不下发口型，也不下发布满谎言的口型。

依赖注入
--------
HTTP 细节关在注入的 `TTSClient` 后面（生产实现负责真实端点，测试用 `FakeTTSClient`）；
本机 / CI 无需任何第三方库即可跑完整测试。

对齐失败的语义（2026-10 补齐）
-----------------------------
强制对齐是**主方案**（见 `align_qwen3.py`），而一次对齐要 100–300 ms、走网络、
可能失败。失败时怎么办，必须写死：

* **音频已经合成成功了，不因对齐失败而丢弃它。** 与方案 §5 第 5 行
  "形象渲染失败 → 静态形象"同一原则——**没有口型不等于没有声音**。
* 此时 `words` 为空、`timing_source="none"`，并计数 `tts.align_failed`。
  调用方据此下发"静态形象"而不是"没声音"。
* 对齐器**返回空列表**同样落 `none`，**绝不标 `forced_alignment`**：那等于宣称
  对齐产出了时间轴，而实际一个字都没有——这种"标注比事实更乐观"正是最容易骗过
  自检的东西。
"""

from __future__ import annotations

import inspect
import time
from dataclasses import dataclass
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Mapping,
    Optional,
    Protocol,
    Union,
    runtime_checkable,
)

from ..domain.errors import Unavailable
from ..ports import SpeechPlan, Usage, WordTiming
from .avatar import timeline_from_words
from .http import DEFAULT_TIMEOUT_S

#: 计量埋点名。看板按这个 key 聚合，改名要同步改看板。
OPS_LATENCY_KEY = "tts.synth"

#: 采样率。来源：方案 §6.1——CosyVoice 2 输出 24 kHz。
DEFAULT_SAMPLE_RATE = 24000

#: 强制对齐签名：`(audio_bytes, text) -> list[WordTiming]`。
#:
#: 允许返回 `Awaitable`：真实对齐要 100–300 ms 且走网络，同步实现会把整个事件
#: 循环按在地上（同进程其他会话的心跳/下行一起卡）。旧式同步对齐器继续可用，
#: `_resolve_timings` 会自动判断要不要 await。
Aligner = Callable[
    [bytes, str], Union["list[WordTiming]", Awaitable["list[WordTiming]"]]
]


class TTSRequestError(Exception):
    """低层客户端报告的**请求级**失败（如非 2xx / 服务端拒绝）。

    适配器把它统一翻译成 `Unavailable`——语音链路不允许"请求失败但返回静音"。
    """

    def __init__(self, status: int, detail: str = "") -> None:
        self.status = int(status)
        self.detail = detail or ""
        # `status <= 0` 表示"请求根本没拿到 HTTP 状态"（连接重置 / 传输层异常）。
        # 那种情况下打印 "HTTP 0" 是误导——它看起来像个服务端状态码。
        label = f"HTTP {self.status}" if self.status > 0 else "无 HTTP 状态"
        super().__init__(f"{label}: {self.detail}")


@dataclass(frozen=True)
class TTSChunk:
    """流式 TTS 的一块产物。

    引擎若提供**原生**词级时间戳，可挂在对应块上（`words`）；否则留空，
    由 `aligner` 事后补齐。
    """

    audio: bytes
    words: tuple[WordTiming, ...] = ()
    sample_rate: int = DEFAULT_SAMPLE_RATE
    cost_usd: float = 0.0


@runtime_checkable
class TTSClient(Protocol):
    """低层 TTS 客户端（生产实现负责 HTTP / WebSocket 细节）。

    返回异步块流；请求失败（网络 / 非 2xx）应抛 `TTSRequestError`，`HttpTTS`
    会统一转成 `Unavailable`。
    """

    async def stream(
        self, text: str, *, voice: str, timeout_s: float
    ) -> AsyncIterator[TTSChunk]: ...


class TermDictionary:
    """术语发音替换表：把中文术语 / 英文缩写替换成 TTS 更易读的读法。

    例：`{"AIGC": "A I G C", "rPPG": "远程光电容积描记"}`。以**最长键优先**替换，
    避免短键（如 `"AI"`）先命中而把长术语（如 `"AIGC"`）截断成残留。
    """

    def __init__(self, mapping: Optional[Mapping[str, str]] = None) -> None:
        self._map: dict[str, str] = dict(mapping or {})

    def add(self, term: str, reading: str) -> None:
        self._map[term] = reading

    def apply(self, text: str) -> str:
        """按最长键优先顺序做一次性字符串替换。"""
        for term in sorted(self._map, key=len, reverse=True):
            text = text.replace(term, self._map[term])
        return text

    def __len__(self) -> int:
        return len(self._map)

    def __contains__(self, term: object) -> bool:
        return term in self._map


class HttpTTS:
    """`TTSPort` 的实现：编排注入的 `TTSClient`，补齐词/口型时间轴并计量。

    Args:
        name: provider 名（进 `SpeechPlan.provider` 与计量）。
        client: 低层 TTS 客户端（见 `TTSClient`）。
        meter: 可观测端口；None 则跳过计量（可观测性不是合成正确性的前置条件）。
        aligner: 强制对齐函数（同步或异步皆可）；None 时若引擎无 native 时间戳
            则 `words` 为空、`timing_source="none"`。对齐失败**不会**丢掉音频，
            只丢口型（见模块 docstring"对齐失败的语义"）。
        term_dictionary: 术语发音替换表；None 表示不替换。
        sample_rate: 默认采样率（客户端块未携带时使用）。
        timeout_s: 单次请求超时。
        clock: 注入式时钟，保证测试确定性（不真睡、不看真实时间）。
    """

    def __init__(
        self,
        name: str,
        client: TTSClient,
        *,
        meter: Optional[Any] = None,
        aligner: Optional[Aligner] = None,
        term_dictionary: Optional[TermDictionary] = None,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.name = name
        self._client = client
        self.meter = meter
        self.aligner = aligner
        self.term_dictionary = term_dictionary
        self.sample_rate = int(sample_rate)
        self.timeout_s = float(timeout_s)
        self._clock = clock
        #: 最近一次对齐失败的原因。**只用于诊断**——计数走 meter，不在主链路上
        #: 抛异常（对齐失败不该让已经合成好的音频作废）。
        self._last_align_error: Optional[str] = None

    @property
    def last_align_error(self) -> Optional[str]:
        """最近一次对齐失败的原因（None = 未发生过失败）。供诊断/测试读取。"""
        return self._last_align_error

    # ---------- TTSPort ----------

    async def synth_stream(self, text: str, *, voice: str = "default") -> AsyncIterator[bytes]:
        """流式合成：`await` 后得到一个音频块异步迭代器（与 `FakeTTS` 约定一致）。"""
        prepared = self._prepare(text)
        return self._stream(prepared, voice)

    async def synth(self, text: str, *, voice: str = "default") -> SpeechPlan:
        """合成整段并补齐 `words` / `visemes`。**成功才返回，失败抛 `Unavailable`。**"""
        prepared = self._prepare(text)
        start = self._clock()
        parts: list[bytes] = []
        native_words: list[WordTiming] = []
        sample_rate = self.sample_rate
        cost = 0.0

        try:
            async for chunk in self._client.stream(
                prepared, voice=voice, timeout_s=self.timeout_s
            ):
                if chunk.audio:
                    parts.append(chunk.audio)
                if chunk.words:
                    native_words.extend(chunk.words)
                if chunk.sample_rate:
                    sample_rate = int(chunk.sample_rate)
                cost += float(chunk.cost_usd)
        except Unavailable:
            raise
        except TTSRequestError as exc:
            self._incr("tts.error")
            raise Unavailable(self.name, f"TTS 服务请求失败：{exc}") from exc
        except Exception as exc:  # 连接重置 / DNS / 超时
            self._incr("tts.error")
            raise Unavailable(self.name, f"网络异常 {type(exc).__name__}: {exc}") from exc

        latency_ms = int(round((self._clock() - start) * 1000.0))
        audio = b"".join(parts)
        if not audio:
            # 空音频不是"成功但内容为空"——它和真实静音不可区分，必须显式失败。
            self._incr("tts.empty_audio")
            raise Unavailable(self.name, "TTS 未返回任何音频（拒绝用静音冒充成功）")

        words, timing_source = await self._resolve_timings(audio, prepared, native_words)
        plan = SpeechPlan(
            audio=audio,
            words=words,
            visemes=timeline_from_words(words),
            sample_rate=sample_rate,
            provider=self.name,
            latency_ms=latency_ms,
            cost_usd=round(cost, 6),
            timing_source=timing_source,
        )
        self._report(latency_ms, plan.cost_usd)
        return plan

    # ---------- 内部 ----------

    async def _stream(self, text: str, voice: str) -> AsyncIterator[bytes]:
        yielded = False
        try:
            async for chunk in self._client.stream(text, voice=voice, timeout_s=self.timeout_s):
                if chunk.audio:
                    yielded = True
                    yield chunk.audio
        except Unavailable:
            raise
        except TTSRequestError as exc:
            self._incr("tts.error")
            raise Unavailable(self.name, f"TTS 服务请求失败：{exc}") from exc
        except Exception as exc:
            self._incr("tts.error")
            raise Unavailable(self.name, f"网络异常 {type(exc).__name__}: {exc}") from exc
        if not yielded:
            self._incr("tts.empty_audio")
            raise Unavailable(self.name, "TTS 流未产出任何音频（拒绝用静音冒充成功）")

    def _prepare(self, text: str) -> str:
        if self.term_dictionary is None:
            return text
        return self.term_dictionary.apply(text)

    async def _resolve_timings(
        self, audio: bytes, text: str, native_words: list[WordTiming]
    ) -> tuple[tuple[WordTiming, ...], str]:
        """决定词时间轴来源。**绝不线性编造**（见模块 docstring 红线 2）。

        三种来源按优先级：
        1. `native`——引擎自己给了；
        2. `forced_alignment`——对齐器**确实产出了**词；
        3. `none`——其余一切情况（没配对齐器 / 对齐失败 / 对齐跑了但没产出）。

        第 3 条的三个边界必须分开记账，否则"跑了但没跑出东西"会被标成"跑成功了"：
        失败计数 `tts.align_failed`、空产出计数 `tts.align_empty`，两者都不吞掉音频。
        """
        if native_words:
            return tuple(native_words), "native"
        if self.aligner is None:
            return (), "none"

        try:
            produced = self.aligner(audio, text)
            # 对齐要 100–300 ms，真实实现是异步的；旧式同步实现继续可用。
            if inspect.isawaitable(produced):
                produced = await produced
        except Unavailable as exc:
            # 结构性故障（对端返回了明确的能力缺失）：记下来，音频照常交付。
            self._last_align_error = f"{exc.provider}: {exc.reason}"
            self._incr("tts.align_failed")
            return (), "none"
        except Exception as exc:  # noqa: BLE001
            self._last_align_error = f"{type(exc).__name__}: {exc}"
            self._incr("tts.align_failed")
            return (), "none"

        words = tuple(produced) if produced else ()
        if not words:
            self._incr("tts.align_empty")
            return (), "none"
        return words, "forced_alignment"

    def _incr(self, name: str) -> None:
        if self.meter is None:
            return
        self.meter.incr(name)

    def _report(self, latency_ms: int, cost_usd: float) -> None:
        if self.meter is None:
            return
        self.meter.record_usage(
            Usage(
                provider=self.name,
                operation=OPS_LATENCY_KEY,
                latency_ms=latency_ms,
                cost_usd=cost_usd,
                cached=False,
            )
        )
        self.meter.observe_ms(OPS_LATENCY_KEY, float(latency_ms))


__all__ = [
    "DEFAULT_SAMPLE_RATE",
    "OPS_LATENCY_KEY",
    "Aligner",
    "HttpTTS",
    "TTSChunk",
    "TTSClient",
    "TTSRequestError",
    "TermDictionary",
]
