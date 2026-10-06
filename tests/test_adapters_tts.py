"""`adapters/tts.py` 的测试。

防的回归（本项目对语音链路的三条红线）
--------------------------------------
1. **绝不返回静音冒充成功**：网络失败 / 请求失败 / 空音频一律 `Unavailable`，且
   绝不能返回一个"看起来正常"的 `SpeechPlan`（其 audio 为静音）。
2. **绝不凭文本长度线性编造时间轴**：无 native 时间戳且无 aligner 时 `words` 必须为
   空、`timing_source="none"`。线性编造会让口型系统性错位且难以察觉。
3. 术语词典必须只改该改的文本。

环境无 httpx / pytest-asyncio：协程一律 `asyncio.run`，客户端用脚本化 `FakeTTSClient`，
时钟注入 `DeterministicClock`（不真睡、不看真实时间）。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, AsyncIterator, Optional, Sequence

import pytest

from ruipin.adapters.fakes import DeterministicClock
from ruipin.adapters.http import DEFAULT_TIMEOUT_S
from ruipin.adapters.tts import (
    DEFAULT_SAMPLE_RATE,
    OPS_LATENCY_KEY,
    HttpTTS,
    TTSChunk,
    TTSRequestError,
    TermDictionary,
)
from ruipin.domain.errors import Unavailable
from ruipin.ports import TTSPort, WordTiming

run = asyncio.run


# ---------- 测试替身 ----------


class RecordingMeter:
    """内存版 `Meter`：三条埋点都记下来供断言。"""

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


@dataclass
class _Call:
    text: str
    voice: str
    timeout_s: float


class FakeTTSClient:
    """脚本化 TTS 客户端：按调用次序产出预设块流（或抛预设异常）。"""

    def __init__(
        self,
        script: Sequence[Any] = (),
        *,
        default: Optional[Any] = None,
        on_call: Optional[Any] = None,
    ) -> None:
        self.script = list(script)
        self.default = default
        self.on_call = on_call
        self.calls: list[_Call] = []

    async def stream(
        self, text: str, *, voice: str, timeout_s: float
    ) -> AsyncIterator[TTSChunk]:
        index = len(self.calls)
        self.calls.append(_Call(text, voice, timeout_s))
        if self.on_call is not None:
            self.on_call(index)
        item = self._item_at(index)
        if isinstance(item, BaseException):
            raise item
        for chunk in item:
            yield chunk

    def _item_at(self, index: int) -> Any:
        if index < len(self.script):
            return self.script[index]
        if self.default is not None:
            return self.default
        raise AssertionError(f"FakeTTSClient 脚本已用尽：第 {index + 1} 次调用无预设")


def _aligner(audio: bytes, text: str) -> list[WordTiming]:
    """假的强制对齐：返回固定时间轴，与音频/文本无关（测试只关心"有没有跑"）。"""
    assert audio  # 必须真的拿到音频才对齐——对齐的是合成结果
    return [WordTiming("你好", 0, 120), WordTiming("世界", 120, 260)]


def _make(
    script: Sequence[Any] = (),
    *,
    aligner=None,
    term_dictionary=None,
    meter=None,
    clock=None,
) -> tuple[HttpTTS, FakeTTSClient, DeterministicClock, RecordingMeter]:
    clock = clock or DeterministicClock()
    meter = meter if meter is not None else RecordingMeter()
    client = FakeTTSClient(script=script, on_call=lambda _i: clock.advance(0.05))
    tts = HttpTTS(
        "t-tts",
        client,
        meter=meter,
        aligner=aligner,
        term_dictionary=term_dictionary,
        clock=clock.monotonic,
    )
    return tts, client, clock, meter


# ---------- 成功路径 ----------


def test_synth_with_aligner_sets_forced_alignment_and_words():
    """防回归：有 aligner 时必须补齐 words 且 timing_source=forced_alignment。"""
    tts, client, _clock, meter = _make(
        [[TTSChunk(b"AA"), TTSChunk(b"BB", cost_usd=0.0003)]], aligner=_aligner
    )
    plan = run(tts.synth("你好世界", voice="v1"))

    assert plan.audio == b"AABB"
    assert plan.provider == "t-tts"
    assert plan.sample_rate == DEFAULT_SAMPLE_RATE
    assert plan.timing_source == "forced_alignment"
    assert plan.words == (WordTiming("你好", 0, 120), WordTiming("世界", 120, 260))
    # visemes 由 words 推导，末尾补一个静音闭合事件
    assert [v.viseme for v in plan.visemes] == ["A", "E", "sil"]
    assert plan.latency_ms == 50
    assert plan.cost_usd == pytest.approx(0.0003)
    assert client.calls[0].text == "你好世界" and client.calls[0].voice == "v1"
    assert client.calls[0].timeout_s == DEFAULT_TIMEOUT_S

    assert len(meter.usages) == 1
    assert meter.usages[0].operation == OPS_LATENCY_KEY
    assert (OPS_LATENCY_KEY, 50.0) in meter.observations


def test_synth_native_timings_take_precedence_over_aligner():
    """防回归：引擎给了原生时间戳就用 native，不再跑对齐、也不误标 forced_alignment。"""
    native = (WordTiming("原生", 0, 100),)
    tts, _client, _clock, _meter = _make(
        [[TTSChunk(b"X", words=native)]], aligner=_aligner
    )
    plan = run(tts.synth("原生"))
    assert plan.timing_source == "native"
    assert plan.words == native


def test_synth_without_aligner_has_empty_words_not_fabricated():
    """防回归（红线 2）：无 aligner 且无 native 时 words 为空、timing_source='none'，
    **绝不**按文本长度编造线性时间轴。"""
    tts, _client, _clock, _meter = _make([[TTSChunk(b"Z")]], aligner=None)
    plan = run(tts.synth("这是一段相当长的文本用来诱导线性的时间轴编造"))
    assert plan.words == ()
    assert plan.visemes == ()
    assert plan.timing_source == "none"


def test_synth_stream_yields_audio_chunks():
    """防回归：`synth_stream` 必须产出音频块（流式首选路径）。"""
    tts, _client, _clock, _meter = _make([[TTSChunk(b"A"), TTSChunk(b"B"), TTSChunk(b"C")]])

    async def collect() -> list[bytes]:
        stream = await tts.synth_stream("hi")
        return [chunk async for chunk in stream]

    assert run(collect()) == [b"A", b"B", b"C"]


# ---------- 失败路径（红线 1） ----------


def test_synth_request_error_raises_unavailable():
    """防回归：非 2xx / 服务端拒绝 → Unavailable，绝不返回静音当成功。"""
    tts, client, _clock, meter = _make([TTSRequestError(503, "down")])
    with pytest.raises(Unavailable) as ei:
        run(tts.synth("你好"))
    assert "503" in ei.value.reason
    assert len(client.calls) == 1
    assert meter.counters.get("tts.error") == 1


def test_synth_network_exception_raises_unavailable():
    """防回归：连接重置 / DNS / 超时 → Unavailable。"""
    tts, _client, _clock, meter = _make([ConnectionResetError("reset")])
    with pytest.raises(Unavailable) as ei:
        run(tts.synth("你好"))
    assert "ConnectionResetError" in ei.value.reason
    assert meter.counters.get("tts.error") == 1


def test_synth_empty_audio_raises_and_never_returns_silence():
    """防回归（红线 1）：客户端产出空音频块时抛 Unavailable，不返回 SpeechPlan。"""
    tts, _client, _clock, meter = _make([[TTSChunk(b""), TTSChunk(b"")]], aligner=_aligner)
    with pytest.raises(Unavailable) as ei:
        run(tts.synth("你好"))
    assert "空" in ei.value.reason or "音频" in ei.value.reason
    assert meter.counters.get("tts.empty_audio") == 1


def test_synth_zero_chunks_raises_unavailable():
    """防回归：客户端一块都不给（空流）同样按失败处理。"""
    tts, _client, _clock, _meter = _make([[]])
    with pytest.raises(Unavailable):
        run(tts.synth("你好"))


def test_synth_reraises_unavailable_without_double_counting():
    """防回归：结构性故障（已是 Unavailable）原样上抛，不被当成网络错误计数。"""
    tts, _client, _clock, meter = _make([Unavailable("engine", "missing")])
    with pytest.raises(Unavailable) as ei:
        run(tts.synth("你好"))
    assert ei.value.provider == "engine"
    assert "tts.error" not in meter.counters


def test_synth_stream_empty_raises_unavailable():
    """防回归：流式路径空流也要显式失败（消费者不能把"没有音频"当成功）。"""
    tts, _client, _clock, meter = _make([[]])

    async def collect() -> list[bytes]:
        stream = await tts.synth_stream("x")
        return [chunk async for chunk in stream]

    with pytest.raises(Unavailable):
        run(collect())
    assert meter.counters.get("tts.empty_audio") == 1


def test_synth_stream_translates_request_error():
    """防回归：流式路径的请求错误同样翻译为 Unavailable。"""
    tts, _client, _clock, _meter = _make([TTSRequestError(429, "slow down")])

    async def collect() -> list[bytes]:
        stream = await tts.synth_stream("x")
        return [chunk async for chunk in stream]

    with pytest.raises(Unavailable) as ei:
        run(collect())
    assert "429" in ei.value.reason


# ---------- 术语词典 ----------


def test_term_dictionary_replaces_only_terms():
    """防回归：术语替换生效，且不碰非术语文本。"""
    td = TermDictionary({"AIGC": "A I G C"})
    text = "请介绍一下 AIGC，以及 rPPG 的原理"
    assert td.apply(text) == "请介绍一下 A I G C，以及 rPPG 的原理"
    assert "rPPG" in td.apply(text)


def test_term_dictionary_longest_key_first():
    """防回归：存在前缀重叠时按最长键替换，避免 "AI" 截断 "AIGC"。"""
    td = TermDictionary({"AI": "A I", "AIGC": "A I G C"})
    assert td.apply("AIGC") == "A I G C"
    assert len(td) == 2 and "AI" in td


def test_term_dictionary_add():
    """防回归：add() 能增量登记术语（音色库/术语库热更新路径）。"""
    td = TermDictionary()
    td.add("AIGC", "A I G C")
    assert td.apply("AIGC") == "A I G C"
    assert len(td) == 1


def test_synth_chunk_zero_sample_rate_falls_back_to_default():
    """防回归：块未携带采样率（0）时回退实例默认，不写 0 进 SpeechPlan。"""
    tts, _client, _clock, _meter = _make([[TTSChunk(b"A", sample_rate=0)]])
    plan = run(tts.synth("x"))
    assert plan.sample_rate == DEFAULT_SAMPLE_RATE


def test_synth_sends_prepared_text_to_client():
    """防回归：术语替换必须真的作用到送进 TTS 的文本上。"""
    td = TermDictionary({"AIGC": "A I G C"})
    tts, client, _clock, _meter = _make([[TTSChunk(b"A")]], term_dictionary=td)
    run(tts.synth("讲 AIGC"))
    assert client.calls[0].text == "讲 A I G C"


# ---------- 端口契约 ----------


def test_implements_tts_port_protocol():
    """防回归：HttpTTS 必须满足 TTSPort（否则核心层无法换实现）。"""
    tts, _client, _clock, _meter = _make([[TTSChunk(b"A")]])
    assert isinstance(tts, TTSPort)


def test_synth_stream_skips_empty_chunks_but_still_succeeds():
    """防回归：流里的空块被跳过，只要有真实音频就不算失败。"""
    tts, _client, _clock, _meter = _make([[TTSChunk(b""), TTSChunk(b"A"), TTSChunk(b"")]])

    async def collect() -> list[bytes]:
        stream = await tts.synth_stream("x")
        return [chunk async for chunk in stream]

    assert run(collect()) == [b"A"]


def test_synth_stream_reraises_unavailable():
    """防回归：流式路径的结构性故障（已是 Unavailable）原样上抛。"""
    tts, _client, _clock, meter = _make([Unavailable("engine", "missing")])

    async def collect() -> list[bytes]:
        stream = await tts.synth_stream("x")
        return [chunk async for chunk in stream]

    with pytest.raises(Unavailable) as ei:
        run(collect())
    assert ei.value.provider == "engine"
    assert "tts.error" not in meter.counters


def test_synth_stream_translates_generic_network_error():
    """防回归：流式路径的网络异常翻译为 Unavailable 并计数。"""
    tts, _client, _clock, meter = _make([ConnectionResetError("reset")])

    async def collect() -> list[bytes]:
        stream = await tts.synth_stream("x")
        return [chunk async for chunk in stream]

    with pytest.raises(Unavailable) as ei:
        run(collect())
    assert "ConnectionResetError" in ei.value.reason
    assert meter.counters.get("tts.error") == 1


def test_no_meter_is_silent_on_success_and_failure():
    """防回归：meter=None 时可观测性完全跳过，不影响合成正确性。"""
    ok_client = FakeTTSClient(script=[[TTSChunk(b"A")]])
    ok_tts = HttpTTS("t-1", ok_client, meter=None, clock=DeterministicClock().monotonic)
    assert run(ok_tts.synth("x")).audio == b"A"

    bad_client = FakeTTSClient(script=[[TTSChunk(b"")]])
    bad_tts = HttpTTS("t-1", bad_client, meter=None, clock=DeterministicClock().monotonic)
    with pytest.raises(Unavailable):
        run(bad_tts.synth("x"))
