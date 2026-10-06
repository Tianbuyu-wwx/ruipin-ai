"""Qwen3-TTS 低层客户端：接 vLLM-Omni 的 OpenAI 兼容 `/v1/audio/speech`，流式取 PCM。

它实现的是**低层** `tts.TTSClient` 协议，不是 `TTSPort`——流式编排、术语替换、
强制对齐补时间轴、红线把关都在 `tts.HttpTTS` 里，这里只负责"把字节从 HTTP 搬进来"。

选型出处
--------
`docs/开源TTS选型研究-2026-10.md`：
* **主选 Qwen3-TTS 12Hz**（Apache 2.0 / 首包 97 ms@0.6B、101 ms@1.7B / 10 语 + 多方言）。
* ⚠️ 只有 **12Hz** 分支是为超低延迟设计的；25Hz 单码本分支要用 block-wise DiT
  重建波形，官方自述"不适合超低延迟"。`DEFAULT_MODEL` 因此指向 12Hz。

本模块**不解析词级时间戳**
----------------------------
选型研究的结论：自托管开源权重**没有一款原生输出词级时间戳**。词时间轴由
`adapters/align_qwen3.py` 的强制对齐器负责（方案 §6.5 的主方案），`HttpTTS`
会自动把 `timing_source` 标成 `forced_alignment`。

服务端**确实**在非流式路径上支持 `word_timestamps: true` + `X-Word-Timestamps`
响应头；流式路径的时间戳投递方式未经证实。所以这里把它做成**默认关闭**的开关
（`request_word_timestamps`）：开了以后会尝试从响应头读，读到就用（`native`），
读不到就静默退回对齐路径——**绝不会**因为读不到而编一份出来。

采样率不做硬编码假设
--------------------
把 24 kHz 写死而服务端实为 48 kHz，会导致**整段语音以两倍速播放**——不报错、
不崩溃，只是听起来像另一个人。所以：
* `response_format="wav"` 时，从 RIFF 头里**读出真实采样率**（WAV 头可跨块到达，
  这里会先攒够再解析，绝不猜）；
* `response_format="pcm"` 时，采样率由 `sample_rate` 显式给出，调用方负责与服务端一致。

成本口径
--------
`TTSChunk.cost_usd` 恒为 0：自托管没有按量计费，成本体现在 GPU 占用上，由
observability 层另行核算。**不填一个编造的单价数字**——那不是"成本为 0"，
而是"成本口径不在这里"。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, AsyncIterator, Mapping, Optional

from ..domain.errors import Unavailable
from ..ports import WordTiming
from .http import DEFAULT_TIMEOUT_S, HttpStreamTransport
from .tts import TTSChunk, TTSRequestError

#: vLLM-Omni 的 OpenAI 兼容语音端点。
DEFAULT_SPEECH_PATH = "/v1/audio/speech"

#: 默认模型。见模块 docstring：只有 12Hz 分支是为超低延迟设计的。
DEFAULT_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"

#: 默认音色。CustomVoice 分支带一组预置音色，实际可用值以服务端为准。
DEFAULT_VOICE = "default"

#: 词级时间戳的响应头（服务端非流式路径使用）。
WORD_TIMESTAMP_HEADER = "x-word-timestamps"

#: 支持的响应格式。
_PCM = "pcm"
_WAV = "wav"


@dataclass(frozen=True)
class WavHeader:
    """canonical WAV 头里我们真正需要的三项。"""

    sample_rate: int
    channels: int
    bits_per_sample: int
    #: PCM 数据的起始偏移（`data` 块正文的第一字节）。
    data_offset: int


def parse_wav_header(buf: bytes) -> Optional[WavHeader]:
    """解析 WAV 头。**返回 None 表示"还没攒够"，不是"解析失败"**。

    逐块扫描而不是死认 44 字节偏移：`fmt ` 与 `data` 之间可能夹着 `LIST`/`fact`
    等块，那些头在真实服务端上并不罕见。死认偏移会在那些响应上读出一个错的采样率，
    而且错得毫无征兆。

    Raises:
        ValueError: 缓冲已足够判定"这根本不是 RIFF/WAVE"，或块结构损坏。
    """
    if len(buf) < 12:
        return None
    if buf[0:4] != b"RIFF" or buf[8:12] != b"WAVE":
        raise ValueError("响应不是 RIFF/WAVE 容器")

    pos = 12
    fmt: Optional[tuple[int, int, int]] = None  # (channels, sample_rate, bits)
    while True:
        if len(buf) < pos + 8:
            return None
        chunk_id = buf[pos : pos + 4]
        size = int.from_bytes(buf[pos + 4 : pos + 8], "little")
        body = pos + 8
        if chunk_id == b"fmt ":
            if len(buf) < body + 16:
                return None
            channels = int.from_bytes(buf[body + 2 : body + 4], "little")
            sample_rate = int.from_bytes(buf[body + 4 : body + 8], "little")
            bits = int.from_bytes(buf[body + 14 : body + 16], "little")
            if sample_rate <= 0 or channels <= 0:
                raise ValueError(f"WAV 头里的参数不合理：{sample_rate} Hz / {channels} 声道")
            fmt = (channels, sample_rate, bits)
        elif chunk_id == b"data":
            if fmt is None:
                raise ValueError("WAV 的 data 块出现在 fmt 块之前")
            return WavHeader(fmt[1], fmt[0], fmt[2], body)
        # 块大小是奇数时有一个填充字节（RIFF 规范），漏掉它后面全错位。
        pos = body + size + (size & 1)


class WavStreamSplitter:
    """把 WAV 字节流拆成"头 + PCM"：头部可跨块到达，攒够才解析。

    `feed()` 返回该块里**属于 PCM 的部分**；头部之前的字节一律吞掉（不是丢失，
    是本来就不该进 PCM 流）。
    """

    def __init__(self) -> None:
        self._buf = bytearray()
        self._header: Optional[WavHeader] = None

    @property
    def header(self) -> Optional[WavHeader]:
        """已解析出的 WAV 头；None 表示还没解析出来。"""
        return self._header

    @property
    def ready(self) -> bool:
        return self._header is not None

    @property
    def pending_bytes(self) -> int:
        """仍在缓冲、尚未判定归属的字节数（诊断用）。"""
        return len(self._buf)

    def feed(self, chunk: bytes) -> bytes:
        """喂一块，返回其中的 PCM 部分（可能是空）。

        Raises:
            ValueError: 缓冲已足够判定这不是合法 WAV。
        """
        if self._header is not None:
            return chunk
        self._buf.extend(chunk)
        header = parse_wav_header(bytes(self._buf))
        if header is None:
            return b""
        self._header = header
        pcm = bytes(self._buf[header.data_offset :])
        self._buf = bytearray()
        return pcm


def _parse_header_timestamps(raw: str) -> list[WordTiming]:
    """把 `X-Word-Timestamps` 头解析成词时间轴。**形状不对就抛，不猜。**

    服务端给的是 JSON 数组，每项 `{"word": str, "start_ms": int, "end_ms": int}`。
    这里**只认毫秒**：如果服务端实际给的是秒而键名仍是 `start_ms`，猜错会让口型
    以 1000 倍速错位。宁可响亮地失败，也不要产出一份看起来正常的错时间轴。
    """
    import json

    data = json.loads(raw)
    if isinstance(data, Mapping):
        data = data.get("words") or data.get("timestamps") or []
    if not isinstance(data, list):
        raise ValueError(f"时间戳不是数组：{type(data).__name__}")

    out: list[WordTiming] = []
    for i, item in enumerate(data):
        if not isinstance(item, Mapping):
            raise ValueError(f"第 {i} 项不是对象：{type(item).__name__}")
        word = item.get("word", item.get("text"))
        start = item.get("start_ms")
        end = item.get("end_ms")
        if not isinstance(word, str) or start is None or end is None:
            raise ValueError(f"第 {i} 项缺 word/start_ms/end_ms：{dict(item)!r}")
        out.append(WordTiming(word=word, start_ms=int(start), end_ms=int(end)))
    return out


class Qwen3TTSClient:
    """vLLM-Omni `/v1/audio/speech` 的流式客户端，实现 `tts.TTSClient`。

    Args:
        transport: 流式 HTTP 传输（见 `http.HttpStreamTransport`）。
        base_url: 服务根地址，如 `http://127.0.0.1:8020`；端点拼成
            `{base_url}{DEFAULT_SPEECH_PATH}`。
        model: 模型名。默认 12Hz 分支（见模块 docstring）。
        voice: 默认音色；`stream()` 的同名参数可逐次覆盖。
        response_format: `"pcm"` 或 `"wav"`；`"wav"` 时会从头部读出真实采样率。
        sample_rate: `response_format="pcm"` 时的采样率声明。
        api_key: 非空时加 `Authorization: Bearer`。自托管通常不需要。
        language: 可选的目标语言提示（服务端支持时生效）。
        speed: 可选语速。
        extra: 额外透传给服务端的字段。**核心字段（model/input/voice 等）优先级更高**，
            防止 `extra` 静默改掉身份字段。
        request_word_timestamps: 是否请求原生词级时间戳。**默认关闭**，
            因为流式路径的时间戳投递方式未经证实；关闭时词时间轴一律走强制对齐。
    """

    def __init__(
        self,
        transport: HttpStreamTransport,
        *,
        base_url: str = "http://127.0.0.1:8020",
        model: str = DEFAULT_MODEL,
        voice: str = DEFAULT_VOICE,
        response_format: str = _PCM,
        sample_rate: int = 24000,
        api_key: str = "",
        language: str = "",
        speed: Optional[float] = None,
        extra: Optional[Mapping[str, Any]] = None,
        request_word_timestamps: bool = False,
        stream_mode: bool = True,
    ) -> None:
        fmt = str(response_format).lower()
        if fmt not in (_PCM, _WAV):
            raise ValueError(f"response_format 只支持 {_PCM!r} / {_WAV!r}，实际: {response_format!r}")
        if not isinstance(sample_rate, int) or isinstance(sample_rate, bool) or sample_rate <= 0:
            raise ValueError(f"sample_rate 必须是正整数，实际: {sample_rate!r}")

        self.transport = transport
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.voice = voice
        self.response_format = fmt
        self.sample_rate = int(sample_rate)
        self.api_key = api_key
        self.language = language
        self.speed = speed
        self.extra = dict(extra or {})
        self.request_word_timestamps = bool(request_word_timestamps)
        #: 请求体里的 `stream` 字段。True = SSE 流式（vLLM-Omni 返回
        # `event: speech.audio.delta` 分段 WAV）；False = 整段 WAV
        # （`application/audio`，无 SSE 封装）。**为什么可以关**：实测
        # （docs/TTS真实端点实测报告-2026-10.md §7.3）vLLM-Omni 的"流式"
        # 节奏近似整段，协议流式换不来真实首包收益；而 SSE 分段 WAV 需要
        # 专门的解包层。整段模式契约最简单可靠。
        self.stream_mode = bool(stream_mode)

    @property
    def url(self) -> str:
        return f"{self.base_url}{DEFAULT_SPEECH_PATH}"

    def build_request(
        self, text: str, voice: str
    ) -> tuple[dict[str, str], dict[str, Any]]:
        """构造 (headers, payload)。**独立成方法是为了让"发出去的到底是什么"可断言**。

        核心字段写在 `extra` 之后，因此永远不会被 `extra` 覆盖。
        """
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload: dict[str, Any] = dict(self.extra)
        payload.update(
            {
                "model": self.model,
                "input": text,
                "voice": voice,
                "response_format": self.response_format,
                "stream": self.stream_mode,
            }
        )
        if self.language:
            payload["language"] = self.language
        if self.speed is not None:
            payload["speed"] = float(self.speed)
        if self.request_word_timestamps:
            payload["word_timestamps"] = True
        return headers, payload

    # ---------- tts.TTSClient ----------

    async def stream(
        self, text: str, *, voice: str, timeout_s: float = DEFAULT_TIMEOUT_S
    ) -> AsyncIterator[TTSChunk]:
        """流式合成。首个音频块一到就交出，不攒整段。

        Raises:
            不会直接抛 `Unavailable` 之外的结构性故障；非 2xx / 网络异常 / 响应体
            不是合法音频，一律翻译成 `TTSRequestError`（`HttpTTS` 会转成
            `Unavailable`，绝不让"静音"冒充成功）。
        """
        headers, payload = self.build_request(text, voice or self.voice)
        splitter = WavStreamSplitter() if self.response_format == _WAV else None
        sample_rate = self.sample_rate
        native_words: list[WordTiming] = []
        head_seen = False
        head_ok = False
        head_status = 0
        error_detail = ""
        saw_audio = False

        try:
            agen = self.transport.post_stream(self.url, headers, payload, timeout_s)
            async for event in agen:
                if event.is_head:
                    head_seen = True
                    head_ok = event.ok
                    head_status = event.status
                    if not event.ok:
                        continue  # 错误正文在下一步；先接着读出来再抛
                    raw_ts = _find_header(event.headers, WORD_TIMESTAMP_HEADER)
                    if raw_ts and self.request_word_timestamps:
                        native_words = _parse_header_timestamps(raw_ts)
                    continue

                if not head_seen:
                    # 还没见到状态码就先来了 body：传输层违约。
                    # **绝不能把它当成"错误说明"咽下去**——那样音频块会被静默丢弃，
                    # 而 `saw_audio` 仍是 False，最后报出来的是"服务端未给出说明"，
                    # 排查方向会被彻底带偏。
                    raise TTSRequestError(0, "流式响应未给出状态码：传输层协议被破坏")

                if not head_ok:
                    error_detail = event.body.decode("utf-8", errors="replace")
                    continue

                audio = event.body
                if splitter is not None:
                    try:
                        audio = splitter.feed(event.body)
                    except ValueError as exc:
                        raise TTSRequestError(
                            head_status, f"response_format=wav 但响应体不是合法 WAV：{exc}"
                        ) from exc
                    if splitter.header is not None:
                        sample_rate = splitter.header.sample_rate
                if audio:
                    saw_audio = True
                    yield TTSChunk(audio=audio, sample_rate=sample_rate)
        except Exception as exc:  # noqa: BLE001
            # 分支说明：
            # * `Unavailable` —— 结构性故障（例如 httpx 没装），原样上抛，
            #   由 `HttpTTS` 直接转给调用方，不在这里降格成"请求失败"。
            # * `TTSRequestError` —— 已经是本层要抛的类型，原样上抛，别包第二层。
            # * 其余（连接重置 / 解析失败 / 传输层超时）—— 统一翻译成 `TTSRequestError`。
            if isinstance(exc, (Unavailable, TTSRequestError)):
                raise
            raise TTSRequestError(0, f"{type(exc).__name__}: {exc}") from exc

        if not head_seen:
            raise TTSRequestError(0, "流式响应未给出状态码：传输层协议被破坏")

        if not head_ok:
            raise TTSRequestError(
                head_status, error_detail or "服务端返回非 2xx 且未给出说明"
            )

        if splitter is not None and not splitter.ready:
            # 流结束了却始终没攒出一个合法 WAV 头：响应体不是 WAV。
            raise TTSRequestError(
                head_status,
                f"response_format=wav 但响应体不是合法 WAV（残留 {splitter.pending_bytes} 字节）",
            )

        if native_words:
            # 时间戳单独作为**纯文本块**（audio 为空）抽出：`HttpTTS` 会把它收进
            # `native_words`，于是 timing_source 标 native、不再跑对齐。
            yield TTSChunk(audio=b"", words=tuple(native_words), sample_rate=sample_rate)

        if not saw_audio:
            # 空音频不在这里抛：那是 `HttpTTS` 的红线（它统一对 synth/synth_stream
            # 两条路径把"零字节"翻成 Unavailable）。这里只负责把事实如实交出。
            return


def _find_header(headers: Mapping[str, str], name: str) -> str:
    """响应头大小写不敏感地取一个值（HTTP 头名本就不区分大小写）。"""
    target = name.lower()
    for key, value in headers.items():
        if key.lower() == target:
            return value
    return ""


__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_SPEECH_PATH",
    "DEFAULT_VOICE",
    "WORD_TIMESTAMP_HEADER",
    "Qwen3TTSClient",
    "WavHeader",
    "WavStreamSplitter",
    "parse_wav_header",
]
