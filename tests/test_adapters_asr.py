"""`adapters/asr.py` 的测试。

防的回归
--------
1. **绝不编造转写文本**：识别不出 → 空串（不是 "嗯" / "未知" / 上一块残留）；
   空缓冲 finalize 直接返回 ""。
2. **累积与复位**：多块 feed 要把音频累积后交给客户端；finalize 后实例可复用于下一轮。
3. **失败显式**：客户端异常 → `Unavailable`，而不是静默返回空串（两者语义不同：
   空串="没识别出"，Unavailable="服务不可用"）。

协程用 `asyncio.run`；时钟注入 `DeterministicClock`；无第三方依赖。
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional, Sequence

import pytest

from ruipin.adapters.asr import OPS_LATENCY_KEY, HttpASR
from ruipin.adapters.fakes import DeterministicClock
from ruipin.domain.errors import Unavailable
from ruipin.ports import ASRPort

run = asyncio.run


class RecordingMeter:
    """内存版 `Meter`。"""

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


class FakeASRClient:
    """脚本化 ASR 客户端。

    `results` 按次序消费；用尽后：若给了 `default` 则用之，否则 `echo=True` 时
    返回解码后的音频（便于验证"累积"），`echo=False` 返回空串。
    `fail=True` 时抛异常（模拟网络/服务故障）。
    """

    def __init__(
        self,
        results: Optional[Sequence[Any]] = None,
        *,
        default: Optional[str] = None,
        fail: bool = False,
        echo: bool = True,
        on_call: Optional[Any] = None,
    ) -> None:
        self.results = list(results) if results else []
        self.default = default
        self.fail = fail
        self.echo = echo
        self.on_call = on_call
        self.calls: list[tuple[bytes, bool]] = []

    async def transcribe(self, audio: bytes, *, is_final: bool) -> str:
        if self.on_call is not None:
            self.on_call(len(self.calls))
        self.calls.append((bytes(audio), is_final))
        if self.fail:
            raise RuntimeError("asr service down")
        if self.results:
            result = self.results.pop(0)
            if isinstance(result, BaseException):
                raise result
            return result
        if self.default is not None:
            return self.default
        return audio.decode("utf-8", "replace") if self.echo else ""


def _make(client: FakeASRClient, *, meter=None, clock=None):
    meter = meter if meter is not None else RecordingMeter()
    clock = clock or DeterministicClock()
    return HttpASR("a-1", client, meter=meter, clock=clock.monotonic), client, meter


def test_feed_accumulates_and_finalize_returns_full_text():
    """防回归：多块 feed 累积音频，finalize 得到完整转写。"""
    asr, client, _meter = _make(FakeASRClient())
    assert run(asr.feed(b"hello")) == "hello"
    assert run(asr.feed(b" world")) == "hello world"
    assert asr.n_bytes == len(b"hello world") and asr.n_chunks == 2

    final = run(asr.finalize())
    assert final == "hello world"
    assert client.calls[-1] == (b"hello world", True)
    # finalize 后复位
    assert asr.n_bytes == 0 and asr.n_chunks == 0


def test_finalize_without_content_returns_empty_string_not_filler():
    """防回归：无内容 finalize 返回 ""，**不是** '嗯' / '未知'，且不发无用请求。"""
    asr, client, _meter = _make(FakeASRClient())
    assert run(asr.finalize()) == ""
    assert client.calls == []


def test_finalize_returns_engine_text_exactly_without_fabrication():
    """防回归：引擎给什么就返回什么（此处引擎给空）——绝不补造文本。"""
    asr, _client, _meter = _make(FakeASRClient(default=""))
    run(asr.feed(b"noise"))
    assert run(asr.finalize()) == ""


def test_finalize_then_reuse_for_next_round():
    """防回归：finalize 复位后，同一实例可用于下一轮而不残留上一轮音频。"""
    asr, _client, _meter = _make(FakeASRClient())
    run(asr.feed(b"a"))
    assert run(asr.finalize()) == "a"
    run(asr.feed(b"b"))
    assert run(asr.finalize()) == "b"  # 不是 "ab"


def test_empty_chunk_feed_returns_empty_without_call():
    """防回归：还没有任何音频时 feed 返回空串，不发起无意义调用、更不编造。"""
    asr, client, _meter = _make(FakeASRClient())
    assert run(asr.feed(b"")) == ""
    assert client.calls == []
    assert asr.n_chunks == 0


def test_whitespace_is_stripped_and_blank_becomes_empty():
    """防回归：纯空白转写归一为空串（避免"看着有内容"的假象）。"""
    # 第 1 次调用（feed）返回空，第 2 次（finalize）返回带空白的文本。
    asr, _client, _meter = _make(FakeASRClient(results=["", "  你好  "]))
    run(asr.feed(b"x"))
    assert run(asr.finalize()) == "你好"

    asr2, _c2, _m2 = _make(FakeASRClient(results=["", "   "]))
    run(asr2.feed(b"x"))
    assert run(asr2.finalize()) == ""


def test_client_exception_becomes_unavailable():
    """防回归：ASR 服务故障 → Unavailable（与"没识别出=空串"必须可区分）。"""
    asr, _client, meter = _make(FakeASRClient(fail=True))
    with pytest.raises(Unavailable) as ei:
        run(asr.feed(b"x"))
    assert "RuntimeError" in ei.value.reason
    assert meter.counters.get("asr.error") == 1


def test_non_string_result_becomes_unavailable():
    """防回归：客户端违约返回非字符串时必须显式失败，不得 str() 硬转。"""
    asr, _client, _meter = _make(FakeASRClient(results=[123]))
    with pytest.raises(Unavailable) as ei:
        run(asr.feed(b"x"))
    assert "非字符串" in ei.value.reason


def test_meter_records_usage_and_latency():
    """防回归：每次调用都如实上报 Usage 与延迟（延迟由注入时钟决定，不真睡）。"""
    clock = DeterministicClock()
    client = FakeASRClient(on_call=lambda _i: clock.advance(0.02))
    meter = RecordingMeter()
    asr = HttpASR("a-1", client, meter=meter, clock=clock.monotonic)
    run(asr.feed(b"x"))
    assert len(meter.usages) == 1
    assert meter.usages[0].operation == OPS_LATENCY_KEY
    assert (OPS_LATENCY_KEY, 20.0) in meter.observations


def test_public_reset_discards_buffer():
    """防回归：reset() 显式丢缓冲（一轮被中断时的清理路径）。"""
    asr, _client, _meter = _make(FakeASRClient())
    run(asr.feed(b"abc"))
    asr.reset()
    assert asr.n_bytes == 0 and asr.n_chunks == 0


def test_implements_asr_port_protocol():
    """防回归：HttpASR 必须满足 ASRPort。"""
    asr, _client, _meter = _make(FakeASRClient())
    assert isinstance(asr, ASRPort)


def test_unavailable_passthrough():
    """防回归：结构性故障（已是 Unavailable）原样上抛，不被当成普通异常计数。"""
    asr, _client, meter = _make(FakeASRClient(results=[Unavailable("engine", "missing")]))
    with pytest.raises(Unavailable) as ei:
        run(asr.feed(b"x"))
    assert ei.value.provider == "engine"
    assert "asr.error" not in meter.counters


def test_no_meter_is_silent_on_success_and_failure():
    """防回归：meter=None 时可观测性完全跳过，不影响识别结果。"""
    ok = HttpASR("a-1", FakeASRClient())
    assert run(ok.feed(b"x")) == "x"

    bad = HttpASR("a-1", FakeASRClient(fail=True))
    with pytest.raises(Unavailable):
        run(bad.feed(b"x"))
