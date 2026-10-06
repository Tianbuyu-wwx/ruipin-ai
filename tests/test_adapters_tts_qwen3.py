"""`adapters/tts_qwen3.py` 的测试。

防的回归
--------
1. **采样率不许硬编码**：把 24 kHz 写死而服务端实为 48 kHz，会让整段语音以
   两倍速播放——不报错、不崩溃，只是听起来像另一个人。WAV 头必须真的被解析，
   且头部跨块到达时也要能解析出来。
2. **非 2xx / 网络异常必须翻成 `TTSRequestError`**，不能往上漏一个裸异常
   （`HttpTTS` 只认 `TTSRequestError` / `Unavailable`，裸异常会绕过降级路径）。
3. **`extra` 不许静默改掉身份字段**（model / input / voice）。
4. **时间戳头只认毫秒**：解析不出来就抛，绝不猜单位——猜错是 1000 倍错位。

环境无 httpx：全部走脚本化 `FakeStreamTransport`，协程用 `asyncio.run`。
"""

from __future__ import annotations

import asyncio
import json
import struct
from dataclasses import dataclass
from typing import Any, AsyncIterator, Optional, Sequence

import pytest

from ruipin.adapters.http import FakeStreamTransport, HttpxStreamTransport, StreamEvent
from ruipin.adapters.tts import HttpTTS, TTSChunk, TTSRequestError
from ruipin.adapters.tts_qwen3 import (
    DEFAULT_MODEL,
    Qwen3TTSClient,
    WavStreamSplitter,
    parse_wav_header,
)
from ruipin.domain.errors import Unavailable
from ruipin.ports import TTSPort, WordTiming

run = asyncio.run


# ---------- 工具 ----------


def make_wav(pcm: bytes, *, sample_rate: int = 48000, channels: int = 1, bits: int = 16) -> bytes:
    """造一个 canonical WAV（纯测试用，不依赖任何音频库）。"""
    byte_rate = sample_rate * channels * bits // 8
    block_align = channels * bits // 8
    header = b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVE"
    header += b"fmt " + struct.pack(
        "<IHHIIHH", 16, 1, channels, sample_rate, byte_rate, block_align, bits
    )
    header += b"data" + struct.pack("<I", len(pcm))
    return header + pcm


def make_wav_with_extra_chunk(pcm: bytes, *, sample_rate: int = 24000) -> bytes:
    """在 fmt 与 data 之间插一个 LIST 块——真实服务端常有，死认 44 字节偏移会读错。"""
    inner = b"INFOhello!"
    extra = b"LIST" + struct.pack("<I", len(inner)) + inner
    header = b"RIFF" + b"\x00\x00\x00\x00" + b"WAVE"
    header += b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16)
    header += extra
    header += b"data" + struct.pack("<I", len(pcm))
    return header + pcm


@dataclass
class _Call:
    text: str
    voice: str
    timeout_s: float


class FakeTTSClient:
    """脚本化 TTS 客户端（与 `test_adapters_tts.py` 同构）。"""

    def __init__(self, script: Sequence[Any] = (), *, default: Optional[Any] = None) -> None:
        self.script = list(script)
        self.default = default
        self.calls: list[_Call] = []

    async def stream(self, text: str, *, voice: str, timeout_s: float) -> AsyncIterator[TTSChunk]:
        index = len(self.calls)
        self.calls.append(_Call(text, voice, timeout_s))
        item = self.script[index] if index < len(self.script) else self.default
        if item is None:
            raise AssertionError(f"FakeTTSClient 脚本已用尽：第 {index + 1} 次调用无预设")
        if isinstance(item, BaseException):
            raise item
        for chunk in item:
            yield chunk


class RecordingMeter:
    def __init__(self) -> None:
        self.usages: list[Any] = []
        self.observations: list[tuple[str, float]] = []
        self.counters: dict[str, int] = {}

    def record_usage(self, usage) -> None:
        self.usages.append(usage)

    def observe_ms(self, name: str, ms: float) -> None:
        self.observations.append((name, ms))

    def incr(self, name: str, value: int = 1) -> None:
        self.counters[name] = self.counters.get(name, 0) + value


def _collect(client: Qwen3TTSClient, text: str = "你好", voice: str = "v1") -> list[TTSChunk]:
    async def go() -> list[TTSChunk]:
        return [c async for c in client.stream(text, voice=voice)]

    return run(go())


# ---------- WAV 头解析 ----------


def test_parse_wav_header_reads_real_sample_rate():
    """防回归（头号）：采样率必须从 WAV 头里读出来，不是写死的常量。"""
    hdr = parse_wav_header(make_wav(b"\x00" * 8, sample_rate=48000, channels=2, bits=16))
    assert hdr is not None
    assert hdr.sample_rate == 48000
    assert hdr.channels == 2
    assert hdr.bits_per_sample == 16
    assert hdr.data_offset == 44


def test_parse_wav_header_returns_none_until_enough_bytes():
    """防回归：数据不足时返回 None（'还没攒够'），不是抛错——头部本来就会跨块到达。"""
    full = make_wav(b"\x00" * 8)
    for n in (0, 4, 11, 12, 20, 40, 43):
        assert parse_wav_header(full[:n]) is None, n
    assert parse_wav_header(full) is not None


def test_parse_wav_header_skips_intermediate_chunks():
    """防回归：fmt 与 data 之间夹着 LIST 块时，不能死认偏移 44。"""
    hdr = parse_wav_header(make_wav_with_extra_chunk(b"\x01" * 4, sample_rate=24000))
    assert hdr is not None
    assert hdr.sample_rate == 24000
    assert hdr.data_offset == 44 + 8 + len(b"INFOhello!")


def test_parse_wav_header_rejects_non_riff():
    """防回归：根本不是 WAV 时要响亮失败，不能返回一个瞎猜的头。"""
    with pytest.raises(ValueError, match="RIFF/WAVE"):
        parse_wav_header(b"<html>gateway error</html>")


def test_parse_wav_header_rejects_absurd_params():
    """防回归：头里参数不合理（0 Hz）时必须抛，否则会静默产出一份错时间轴。"""
    bad = bytearray(make_wav(b"\x00" * 4, sample_rate=24000))
    bad[24:28] = struct.pack("<I", 0)  # sample_rate 置 0
    with pytest.raises(ValueError, match="不合理"):
        parse_wav_header(bytes(bad))


def test_wav_splitter_handles_header_across_chunks():
    """防回归：头部跨 3 个块到达时，PCM 一个字节都不能丢、也不能把头部混进去。"""
    pcm = bytes(range(256))
    raw = make_wav(pcm, sample_rate=48000)
    splitter = WavStreamSplitter()
    out = b""
    for i in range(0, len(raw), 7):  # 7 字节一块，刻意切在头部中间
        out += splitter.feed(raw[i : i + 7])
    assert out == pcm
    assert splitter.ready and splitter.header is not None
    assert splitter.header.sample_rate == 48000
    assert splitter.pending_bytes == 0


def test_wav_splitter_passes_through_after_header():
    splitter = WavStreamSplitter()
    splitter.feed(make_wav(b"\x01\x02"))
    assert splitter.feed(b"\xaa\xbb") == b"\xaa\xbb"


# ---------- 构造参数校验 ----------


def test_rejects_unknown_response_format():
    """防回归：response_format 只认 pcm/wav，别的写法要在构造期就炸而不是运行时静默降级。"""
    with pytest.raises(ValueError, match="response_format"):
        Qwen3TTSClient(FakeStreamTransport(), response_format="mp3")


def test_rejects_nonpositive_sample_rate():
    with pytest.raises(ValueError, match="sample_rate"):
        Qwen3TTSClient(FakeStreamTransport(), sample_rate=0)
    with pytest.raises(ValueError, match="sample_rate"):
        Qwen3TTSClient(FakeStreamTransport(), sample_rate=True)  # bool 不是采样率


# ---------- 请求构造 ----------


def test_request_uses_openai_compatible_shape():
    client = Qwen3TTSClient(FakeStreamTransport(), model=DEFAULT_MODEL, voice="v9")
    headers, payload = client.build_request("讲一下", "v9")
    assert client.url.endswith("/v1/audio/speech")
    assert payload["model"] == DEFAULT_MODEL
    assert payload["input"] == "讲一下"
    assert payload["voice"] == "v9"
    assert payload["stream"] is True
    assert payload["response_format"] == "pcm"
    assert headers["Content-Type"] == "application/json"
    assert "Authorization" not in headers  # 自托管默认不带鉴权头


def test_extra_cannot_override_identity_fields():
    """防回归：extra 是透传口，不能把 model/input/voice 这些身份字段改掉。"""
    client = Qwen3TTSClient(
        FakeStreamTransport(),
        model="REAL",
        extra={"model": "HACKED", "input": "HACKED", "voice": "HACKED", "temperature": 0.7},
    )
    _headers, payload = client.build_request("真话", "真音色")
    assert payload["model"] == "REAL"
    assert payload["input"] == "真话"
    assert payload["voice"] == "真音色"
    assert payload["temperature"] == 0.7  # 非核心字段照常透传


def test_api_key_adds_authorization_header():
    client = Qwen3TTSClient(FakeStreamTransport(), api_key="sk-x")
    headers, _payload = client.build_request("a", "v")
    assert headers["Authorization"] == "Bearer sk-x"


def test_word_timestamps_only_requested_when_asked():
    """默认**不请求**原生时间戳：流式路径的时间戳投递方式未经证实。"""
    off = Qwen3TTSClient(FakeStreamTransport())
    assert "word_timestamps" not in off.build_request("a", "v")[1]
    on = Qwen3TTSClient(FakeStreamTransport(), request_word_timestamps=True)
    assert on.build_request("a", "v")[1]["word_timestamps"] is True


# ---------- 流式成功路径 ----------


def test_stream_yields_pcm_chunks():
    transport = FakeStreamTransport(script=[[StreamEvent(status=200), b"AA", b"BB", b"CC"]])
    client = Qwen3TTSClient(transport, sample_rate=24000)
    chunks = _collect(client)
    assert [c.audio for c in chunks] == [b"AA", b"BB", b"CC"]
    assert {c.sample_rate for c in chunks} == {24000}
    assert all(c.cost_usd == 0.0 for c in chunks)  # 自托管不编造成本数字
    assert transport.calls[0].url.endswith("/v1/audio/speech")


def test_stream_uses_real_sample_rate_from_wav():
    """防回归（头号）：wav 模式下采样率必须来自响应，而不是构造参数。"""
    wav = make_wav(b"\x01\x02" * 50, sample_rate=48000)
    transport = FakeStreamTransport(
        script=[[StreamEvent(status=200), wav[:9], wav[9:31], wav[31:]]]
    )
    client = Qwen3TTSClient(transport, response_format="wav", sample_rate=24000)
    chunks = _collect(client)
    assert sum(len(c.audio) for c in chunks) == 100
    assert {c.sample_rate for c in chunks} == {48000}
    assert all(c.sample_rate != 24000 for c in chunks)


def test_stream_skips_empty_body_events():
    transport = FakeStreamTransport(script=[[StreamEvent(status=200), b"", b"A", b""]])
    assert [c.audio for c in _collect(Qwen3TTSClient(transport))] == [b"A"]


def test_stream_empty_audio_does_not_raise_here():
    """空音频由 `HttpTTS` 统一裁决（红线 1 只能有一个执行点）。本层如实交出"零块"。"""
    transport = FakeStreamTransport(script=[[StreamEvent(status=200), b"", b""]])
    assert _collect(Qwen3TTSClient(transport)) == []


# ---------- 失败路径 ----------


def test_non_2xx_raises_request_error_with_status_and_detail():
    """防回归：非 2xx 必须带状态码与服务端说明，否则日志里只剩"HTTP 400"。"""
    transport = FakeStreamTransport(
        script=[[StreamEvent(status=400), b'{"error":"unknown voice"}']]
    )
    client = Qwen3TTSClient(transport)
    with pytest.raises(TTSRequestError) as ei:
        _collect(client)
    assert ei.value.status == 400
    assert "unknown voice" in ei.value.detail
    assert ei.value.__str__().startswith("HTTP 400")


def test_non_2xx_without_detail_still_reports_status():
    transport = FakeStreamTransport(script=[[StreamEvent(status=503)]])
    with pytest.raises(TTSRequestError) as ei:
        _collect(Qwen3TTSClient(transport))
    assert ei.value.status == 503
    assert "非 2xx" in ei.value.detail


def test_network_exception_becomes_request_error_without_fake_status():
    """防回归：网络异常没有 HTTP 状态码，不能把它印成 "HTTP 0"（看起来像服务端状态）。"""
    transport = FakeStreamTransport(script=[ConnectionResetError("reset")])
    with pytest.raises(TTSRequestError) as ei:
        _collect(Qwen3TTSClient(transport))
    assert ei.value.status == 0
    assert "无 HTTP 状态" in ei.value.__str__()
    assert "ConnectionResetError" in ei.value.detail


def test_unavailable_passes_through_unwrapped():
    """防回归：结构性故障（如 httpx 未安装）原样上抛，不被降格成"请求失败"。"""
    transport = FakeStreamTransport(script=[Unavailable("httpx", "未安装")])
    with pytest.raises(Unavailable) as ei:
        _collect(Qwen3TTSClient(transport))
    assert ei.value.provider == "httpx"


def test_wav_mode_with_non_wav_body_raises():
    """防回归：声明 wav 却收到 HTML 错误页时，必须抛，不能把错误页当 PCM 播出去。

    并且状态码要如实报 200（服务端认为它成功了），不能报成"无 HTTP 状态"——
    那是网络故障的措辞，会把排查方向从"响应体不对"带偏到"连接断了"。
    """
    transport = FakeStreamTransport(
        script=[[StreamEvent(status=200), b"<html>not a wav at all</html>"]]
    )
    client = Qwen3TTSClient(transport, response_format="wav")
    with pytest.raises(TTSRequestError) as ei:
        _collect(client)
    assert ei.value.status == 200
    assert "不是合法 WAV" in ei.value.detail
    assert "RIFF/WAVE" in ei.value.detail


def test_wav_mode_with_truncated_header_raises():
    """防回归：流结束了头都没攒齐（不是 WAV），不能静默返回空音频。"""
    transport = FakeStreamTransport(script=[[StreamEvent(status=200), b"RIFF\x00\x00"]])
    client = Qwen3TTSClient(transport, response_format="wav")
    with pytest.raises(TTSRequestError, match="合法 WAV"):
        _collect(client)


def test_missing_response_head_raises():
    """防回归：传输层没给状态码（协议被破坏）时必须显式失败。

    这里防的是一个很隐蔽的错法：把 body 当成"错误说明"咽下去。那样音频块被静默
    丢弃、`saw_audio` 仍为 False，最后报出来的是"服务端未给出说明"——排查方向
    会被彻底带偏（会去查服务端，而实际是传输层违约）。
    """
    transport = FakeStreamTransport(script=[[b"AA", b"BB"]])
    with pytest.raises(TTSRequestError, match="未给出状态码") as ei:
        _collect(Qwen3TTSClient(transport))
    assert "BB" not in ei.value.detail  # 音频块绝不能被当成错误说明


def test_empty_stream_without_head_raises():
    """一个事件都没有（传输层直接结束）同样算协议违约，不能算"零块成功"。"""
    transport = FakeStreamTransport(script=[[]])
    with pytest.raises(TTSRequestError, match="未给出状态码"):
        _collect(Qwen3TTSClient(transport))


# ---------- 原生时间戳（默认关闭） ----------


def test_native_word_timestamps_parsed_from_header_when_requested():
    """开启后能从响应头读出原生时间戳，作为**音频之外的附加块**交出。"""
    ts = json.dumps([{"word": "你好", "start_ms": 0, "end_ms": 120}], ensure_ascii=False)
    transport = FakeStreamTransport(
        script=[[StreamEvent(status=200, headers={"X-Word-Timestamps": ts}), b"AA"]]
    )
    client = Qwen3TTSClient(transport, request_word_timestamps=True)
    chunks = _collect(client)
    assert [c.audio for c in chunks] == [b"AA", b""]  # 末块是纯时间戳块
    assert chunks[-1].words == (WordTiming("你好", 0, 120),)


def test_word_timestamp_header_ignored_when_not_requested():
    """防回归：没请求就不要采信——避免把未经验证的字段悄悄标成 native。"""
    ts = json.dumps([{"word": "你好", "start_ms": 0, "end_ms": 120}])
    transport = FakeStreamTransport(
        script=[[StreamEvent(status=200, headers={"X-Word-Timestamps": ts}), b"AA"]]
    )
    chunks = _collect(Qwen3TTSClient(transport))
    assert [c.audio for c in chunks] == [b"AA"]
    assert all(c.words == () for c in chunks)


def test_seconds_in_header_is_rejected_not_guessed():
    """防回归（最危险的一种）：只有 start/end（秒）时**必须抛**。

    若把它当毫秒用，口型会慢 1000 倍；若猜成秒，遇到真毫秒又会快 1000 倍。
    两种都是"图能画出来、看起来正常、但完全错"的输出——所以猜是被禁止的。
    """
    ts = json.dumps([{"word": "你好", "start": 0.0, "end": 0.12}])
    transport = FakeStreamTransport(
        script=[[StreamEvent(status=200, headers={"X-Word-Timestamps": ts}), b"AA"]]
    )
    client = Qwen3TTSClient(transport, request_word_timestamps=True)
    with pytest.raises(TTSRequestError, match="start_ms"):
        _collect(client)


def test_malformed_timestamp_json_raises():
    transport = FakeStreamTransport(
        script=[[StreamEvent(status=200, headers={"X-Word-Timestamps": "{not json"}), b"AA"]]
    )
    client = Qwen3TTSClient(transport, request_word_timestamps=True)
    with pytest.raises(TTSRequestError):
        _collect(client)


# ---------- 与编排层合体 ----------


def test_httpts_end_to_end_forced_alignment_via_aligner():
    """合体：Qwen3TTSClient + 同步对齐器 → SpeechPlan 落 forced_alignment。"""
    transport = FakeStreamTransport(script=[[StreamEvent(status=200), b"AA", b"BB"]])

    def aligner(audio: bytes, text: str) -> list[WordTiming]:
        assert audio == b"AABB"
        return [WordTiming("你好", 0, 100)]

    tts = HttpTTS(
        "qwen3", Qwen3TTSClient(transport), aligner=aligner, meter=RecordingMeter()
    )
    plan = run(tts.synth("你好"))
    assert plan.audio == b"AABB"
    assert plan.timing_source == "forced_alignment"
    assert plan.provider == "qwen3"


def test_httpts_end_to_end_native_from_header():
    """合体：开启原生时间戳后，编排层应标 native 且**不再跑对齐**。"""
    ts = json.dumps([{"word": "原生", "start_ms": 5, "end_ms": 90}])
    transport = FakeStreamTransport(
        script=[[StreamEvent(status=200, headers={"X-Word-Timestamps": ts}), b"AA"]]
    )
    aligner_calls: list[int] = []

    def aligner(audio: bytes, text: str) -> list[WordTiming]:
        aligner_calls.append(1)
        return [WordTiming("不该出现", 0, 1)]

    tts = HttpTTS(
        "qwen3",
        Qwen3TTSClient(transport, request_word_timestamps=True),
        aligner=aligner,
    )
    plan = run(tts.synth("原生"))
    assert plan.timing_source == "native"
    assert plan.words == (WordTiming("原生", 5, 90),)
    assert aligner_calls == []


def test_httpts_translates_request_error_to_unavailable():
    """合体：非 2xx → `Unavailable`，绝不返回静音当成功（红线 1）。"""
    transport = FakeStreamTransport(script=[[StreamEvent(status=503), b"overloaded"]])
    tts = HttpTTS("qwen3", Qwen3TTSClient(transport), meter=RecordingMeter())
    with pytest.raises(Unavailable) as ei:
        run(tts.synth("你好"))
    assert "503" in ei.value.reason


def test_implements_tts_port():
    """Qwen3TTSClient 是低层客户端，真正的 TTSPort 是 HttpTTS 包出来的那一层。"""
    transport = FakeStreamTransport(script=[[StreamEvent(status=200), b"A"]])
    tts = HttpTTS("qwen3", Qwen3TTSClient(transport))
    assert isinstance(tts, TTSPort)


def test_httpx_transport_reports_missing_dependency_as_unavailable(monkeypatch):
    """防回归：httpx 未安装时抛 `Unavailable`，不是一个裸 ImportError。"""
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "httpx":
            raise ImportError("no httpx")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    async def go() -> None:
        async for _ in HttpxStreamTransport().post_stream("http://x", {}, {}, 1.0):
            pass

    with pytest.raises(Unavailable) as ei:
        run(go())
    assert ei.value.provider == "httpx"
