"""`adapters/align_qwen3.py` 的测试。

防的回归
--------
1. **单位不许猜**。这是本模块存在的全部理由：若服务端给的是秒而键名被当成毫秒，
   口型慢 1000 倍；反过来快 1000 倍。两种都"图能画出来、看起来正常、但完全错"。
   所以只认两种"单位写在字段名里"的方言（`start_ms`/`end_ms` = 毫秒；
   `start_time`/`end_time` = 秒，官方 `qwen-asr` 包的原生形状，2026-10 读源码
   实证），其余键名一律抛；两种方言**混用**也抛。
2. **失败必须显式**。请求失败 / 非 2xx / JSON 不合法 / 字段自相矛盾 → `Unavailable`，
   **绝不返回一个"看起来合理"的时间轴兜底**。
3. **对齐是异步的**：同步实现会把事件循环按在地上（同进程其他会话一起卡）。
   编排层 `HttpTTS` 必须能 await 它，而旧式同步对齐器继续可用。

环境无 httpx：全部走脚本化 `FakeTransport`，协程用 `asyncio.run`。
"""

from __future__ import annotations

import asyncio
import base64
import json
import struct
from typing import Any, Optional, Sequence

import pytest

from ruipin.adapters.align_qwen3 import (
    DEFAULT_ALIGN_PATH,
    PROVIDER,
    Qwen3ForcedAligner,
    words_span_ms,
)
from ruipin.adapters.http import FakeTransport, HttpResponse
from ruipin.adapters.tts import HttpTTS, TTSChunk
from ruipin.domain.errors import Unavailable
from ruipin.ports import WordTiming

run = asyncio.run


# ---------- 工具 ----------


def ok(body: Any, status: int = 200) -> HttpResponse:
    raw = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
    return HttpResponse(status=status, body=raw)


class FakeTTSClient:
    """一块音频的 TTS 替身（只为把音频喂给对齐器）。"""

    def __init__(self, audio: bytes = b"PCM-AUDIO") -> None:
        self.audio = audio

    async def stream(self, text: str, *, voice: str, timeout_s: float):
        yield TTSChunk(audio=self.audio)


class RecordingMeter:
    def __init__(self) -> None:
        self.counters: dict[str, int] = {}
        self.usages: list[Any] = []
        self.observations: list[tuple[str, float]] = []

    def incr(self, name: str, value: int = 1) -> None:
        self.counters[name] = self.counters.get(name, 0) + value

    def record_usage(self, usage) -> None:
        self.usages.append(usage)

    def observe_ms(self, name: str, ms: float) -> None:
        self.observations.append((name, ms))


def _align(aligner: Qwen3ForcedAligner, audio: bytes = b"PCM", text: str = "你好"):
    return run(aligner(audio, text))


# ---------- 成功路径：三种容器形状 ----------


def test_parses_top_level_array():
    """顶层数组是最直接的形状。"""
    t = FakeTransport(script=[ok([{"word": "你好", "start_ms": 0, "end_ms": 120}])])
    words = _align(Qwen3ForcedAligner(t))
    assert words == [WordTiming("你好", 0, 120)]


def test_parses_words_object():
    t = FakeTransport(script=[ok({"words": [{"word": "你好", "start_ms": 0, "end_ms": 120}]})])
    assert _align(Qwen3ForcedAligner(t)) == [WordTiming("你好", 0, 120)]


def test_parses_and_flattens_segments():
    """`{"segments": [{"words": [...]}]}` 要展平成一条时间轴，且顺序保持。"""
    t = FakeTransport(
        script=[
            ok(
                {
                    "segments": [
                        {"words": [{"word": "第一", "start_ms": 0, "end_ms": 100}]},
                        {"words": [{"word": "第二", "start_ms": 100, "end_ms": 220}]},
                    ]
                }
            )
        ]
    )
    assert _align(Qwen3ForcedAligner(t)) == [
        WordTiming("第一", 0, 100),
        WordTiming("第二", 100, 220),
    ]


def test_empty_words_is_not_a_failure():
    """合法但为空（音频是静音）时返回 []，不是抛错。

    `HttpTTS` 会把这种情况标成 `timing_source="none"` 并计数 `tts.align_empty`——
    "跑了但没产出"必须与"跑成功了"区分开。
    """
    t = FakeTransport(script=[ok({"words": []})])
    assert _align(Qwen3ForcedAligner(t)) == []


# ---------- 请求构造 ----------


def test_request_shape_and_base64_audio():
    t = FakeTransport(script=[ok({"words": []})])
    aligner = Qwen3ForcedAligner(t, language="zh", granularity="word")
    _align(aligner, audio=b"\x00\x01\x02", text="你好")

    call = t.calls[0]
    assert call.url.endswith(DEFAULT_ALIGN_PATH)
    assert call.payload["text"] == "你好"
    assert call.payload["language"] == "zh"
    assert call.payload["granularity"] == "word"
    assert base64.b64decode(call.payload["audio"]) == b"\x00\x01\x02"
    assert call.timeout_s > 0


def test_extra_cannot_override_core_fields():
    """防回归：extra 是透传口，不能把 audio/text 这些核心字段改掉。"""
    t = FakeTransport(script=[ok({"words": []})])
    aligner = Qwen3ForcedAligner(
        t, extra={"audio": "HACKED", "text": "HACKED", "temperature": 0}
    )
    _align(aligner, audio=b"REAL", text="真文本")
    assert t.calls[0].payload["text"] == "真文本"
    assert base64.b64decode(t.calls[0].payload["audio"]) == b"REAL"
    assert t.calls[0].payload["temperature"] == 0


def test_api_key_adds_authorization_header():
    t = FakeTransport(script=[ok({"words": []})])
    _align(Qwen3ForcedAligner(t, api_key="sk-x"))
    assert t.calls[0].headers["Authorization"] == "Bearer sk-x"


def test_rejects_unknown_granularity():
    with pytest.raises(ValueError, match="granularity"):
        Qwen3ForcedAligner(FakeTransport(), granularity="phoneme")


def test_rejects_nonpositive_timeout():
    with pytest.raises(ValueError, match="timeout_s"):
        Qwen3ForcedAligner(FakeTransport(), timeout_s=0)


# ---------- 单位与字段的严格性（本模块的核心防线） ----------


def test_seconds_aliases_are_rejected():
    """防回归（最危险的一种）：`start`/`end`（秒）必须抛，不能猜成毫秒。"""
    t = FakeTransport(script=[ok([{"word": "你好", "start": 0.0, "end": 0.12}])])
    with pytest.raises(Unavailable) as ei:
        _align(Qwen3ForcedAligner(t))
    assert ei.value.provider == PROVIDER
    assert "start_ms" in ei.value.reason


def test_second_named_alias_start_s_rejected():
    """`start_s`/`end_s` 同样不认——别名越多，猜错的窗口越大。"""
    t = FakeTransport(script=[ok([{"word": "你好", "start_s": 0, "end_s": 1}])])
    with pytest.raises(Unavailable):
        _align(Qwen3ForcedAligner(t))


def test_missing_word_field_rejected():
    t = FakeTransport(script=[ok([{"start_ms": 0, "end_ms": 10}])])
    with pytest.raises(Unavailable, match="word"):
        _align(Qwen3ForcedAligner(t))


def test_end_before_start_rejected_not_silently_fixed():
    """自相矛盾的条目要抛，不能"顺手修正"成一份看着还行的时间轴。"""
    t = FakeTransport(script=[ok([{"word": "你好", "start_ms": 200, "end_ms": 100}])])
    with pytest.raises(Unavailable, match="end_ms"):
        _align(Qwen3ForcedAligner(t))


def test_negative_start_rejected():
    t = FakeTransport(script=[ok([{"word": "你好", "start_ms": -5, "end_ms": 10}])])
    with pytest.raises(Unavailable, match="负"):
        _align(Qwen3ForcedAligner(t))


def test_boolean_is_not_a_number():
    """Python 里 `True` 是 `int` 的实例——不显式排除会把 true 当成 1 ms。"""
    t = FakeTransport(script=[ok([{"word": "你好", "start_ms": True, "end_ms": 10}])])
    with pytest.raises(Unavailable):
        _align(Qwen3ForcedAligner(t))


def test_string_numbers_rejected():
    t = FakeTransport(script=[ok([{"word": "你好", "start_ms": "0", "end_ms": "10"}])])
    with pytest.raises(Unavailable):
        _align(Qwen3ForcedAligner(t))


def test_unknown_container_shape_rejected():
    t = FakeTransport(script=[ok({"result": "whatever"})])
    with pytest.raises(Unavailable, match="形状"):
        _align(Qwen3ForcedAligner(t))


def test_segments_without_words_rejected():
    t = FakeTransport(script=[ok({"segments": [{"start_ms": 0}]})])
    with pytest.raises(Unavailable, match="words"):
        _align(Qwen3ForcedAligner(t))


def test_non_object_item_rejected():
    t = FakeTransport(script=[ok([[1, 2, 3]])])
    with pytest.raises(Unavailable, match="不是对象"):
        _align(Qwen3ForcedAligner(t))


# ---------- 秒方言：官方 qwen-asr 包的原生形状（2026-10 读源码实证） ----------


def test_parses_official_seconds_dialect():
    """官方包 `Qwen3ForcedAligner.align()` 的原生条目：`text`/`start_time`/`end_time`，秒。

    这是 2026-10 读 `qwen_asr` 0.0.6 源码实证的形状（`parse_timestamp()` 产出
    `{"text", "start_time", "end_time"}`，`align()` 里再 `round(x/1000.0, 3)`），
    不是文档推测。
    """
    t = FakeTransport(
        script=[ok([{"text": "你", "start_time": 0.0, "end_time": 0.12},
                     {"text": "好", "start_time": 0.12, "end_time": 0.26}])]
    )
    assert _align(Qwen3ForcedAligner(t)) == [
        WordTiming("你", 0, 120),
        WordTiming("好", 120, 260),
    ]


def test_seconds_dialect_rounds_to_exact_ms():
    """秒 → 毫秒换算必须落在整毫秒上。

    官方实现 `round(x/1000.0, 3)` 本身就是毫秒精度，但浮点乘法会引入
    `0.123 * 1000 == 122.99999999999999` 这类噪声，必须用 round 收敛，
    否则时间轴上全是 122、259 这种"差一"值，对 A/B 工具的误差统计是毒药。
    """
    t = FakeTransport(script=[ok([{"text": "你", "start_time": 0.123, "end_time": 0.456}])])
    assert _align(Qwen3ForcedAligner(t)) == [WordTiming("你", 123, 456)]


def test_seconds_dialect_in_words_container():
    t = FakeTransport(
        script=[ok({"words": [{"text": "hi", "start_time": 0.5, "end_time": 0.75}]})]
    )
    assert _align(Qwen3ForcedAligner(t)) == [WordTiming("hi", 500, 750)]


def test_seconds_dialect_empty_is_not_a_failure():
    t = FakeTransport(script=[ok({"words": []})])
    assert _align(Qwen3ForcedAligner(t)) == []


def test_mixed_dialect_within_one_item_rejected():
    """`word` 配 `start_time` —— 两种方言各取一半。必须抛，不能挑能用的键。"""
    t = FakeTransport(
        script=[ok([{"word": "你好", "start_time": 0.0, "end_time": 0.1}])]
    )
    with pytest.raises(Unavailable, match="不一致"):
        _align(Qwen3ForcedAligner(t))


def test_mixed_dialect_across_items_rejected():
    """第 0 项是毫秒、第 1 项是秒：逐项自适应会把它们拼成一条看似完整的时间轴。

    按第一项定调、其余不一致就抛——脏数据必须当场暴露，而不是被"宽容"掉。
    """
    t = FakeTransport(
        script=[
            ok(
                [
                    {"word": "a", "start_ms": 0, "end_ms": 100},
                    {"word": "b", "start_time": 0.1, "end_time": 0.2},
                ]
            )
        ]
    )
    with pytest.raises(Unavailable, match="不一致"):
        _align(Qwen3ForcedAligner(t))


def test_seconds_dialect_non_string_text_rejected():
    t = FakeTransport(script=[ok([{"text": 7, "start_time": 0.0, "end_time": 0.1}])])
    with pytest.raises(Unavailable, match="text"):
        _align(Qwen3ForcedAligner(t))


def test_seconds_dialect_boolean_rejected():
    """秒方言同样要挡 bool（`True` 是 `int` 实例）。"""
    t = FakeTransport(
        script=[ok([{"text": "你", "start_time": True, "end_time": 0.1}])]
    )
    with pytest.raises(Unavailable):
        _align(Qwen3ForcedAligner(t))


def test_seconds_dialect_end_before_start_rejected():
    t = FakeTransport(
        script=[ok([{"text": "你", "start_time": 0.5, "end_time": 0.2}])]
    )
    with pytest.raises(Unavailable, match="end_time"):
        _align(Qwen3ForcedAligner(t))


def test_seconds_dialect_negative_rejected():
    t = FakeTransport(
        script=[ok([{"text": "你", "start_time": -0.01, "end_time": 0.1}])]
    )
    with pytest.raises(Unavailable, match="负"):
        _align(Qwen3ForcedAligner(t))


# ---------- 失败路径 ----------


def test_non_2xx_raises_unavailable_with_status():
    t = FakeTransport(script=[HttpResponse(status=503, body=b"model loading")])
    with pytest.raises(Unavailable) as ei:
        _align(Qwen3ForcedAligner(t))
    assert "503" in ei.value.reason
    assert "model loading" in ei.value.reason


def test_invalid_json_raises_unavailable():
    t = FakeTransport(script=[HttpResponse(status=200, body=b"<html>oops</html>")])
    with pytest.raises(Unavailable, match="JSON"):
        _align(Qwen3ForcedAligner(t))


def test_network_exception_raises_unavailable():
    t = FakeTransport(script=[ConnectionResetError("reset")])
    with pytest.raises(Unavailable) as ei:
        _align(Qwen3ForcedAligner(t))
    assert "ConnectionResetError" in ei.value.reason


def test_unavailable_from_transport_passes_through():
    """结构性故障（httpx 未装）不该被包成"对齐请求失败"。"""
    t = FakeTransport(script=[Unavailable("httpx", "未安装")])
    with pytest.raises(Unavailable) as ei:
        _align(Qwen3ForcedAligner(t))
    assert ei.value.provider == "httpx"


def test_empty_audio_rejected_without_calling_service():
    """空音频直接拒，不浪费一次请求——空音频与任何文本都对齐不出东西。"""
    t = FakeTransport()
    with pytest.raises(Unavailable, match="音频为空"):
        _align(Qwen3ForcedAligner(t), audio=b"")
    assert t.n_calls == 0


def test_empty_text_rejected_without_calling_service():
    t = FakeTransport()
    with pytest.raises(Unavailable, match="文本为空"):
        _align(Qwen3ForcedAligner(t), text="")
    assert t.n_calls == 0


# ---------- 工具函数 ----------


def test_words_span_ms():
    assert words_span_ms([]) == 0
    assert words_span_ms([WordTiming("a", 0, 100), WordTiming("b", 100, 260)]) == 260


# ---------- 与编排层合体 ----------


def test_httpts_awaits_async_aligner_and_marks_forced_alignment():
    """合体（关键）：`HttpTTS` 必须能 await 异步对齐器，并标 forced_alignment。"""
    t = FakeTransport(script=[ok([{"word": "你好", "start_ms": 0, "end_ms": 120}])])
    tts = HttpTTS("qwen3", FakeTTSClient(b"PCM"), aligner=Qwen3ForcedAligner(t))
    plan = run(tts.synth("你好"))
    assert plan.audio == b"PCM"
    assert plan.timing_source == "forced_alignment"
    assert plan.words == (WordTiming("你好", 0, 120),)
    assert [v.viseme for v in plan.visemes] == ["A", "sil"]


def test_sync_aligner_still_supported():
    """防回归：旧式**同步**对齐器不能被这次改动弄坏（向后兼容）。"""
    calls: list[int] = []

    def sync_aligner(audio: bytes, text: str) -> list[WordTiming]:
        calls.append(1)
        return [WordTiming("同步", 0, 50)]

    tts = HttpTTS("t", FakeTTSClient(b"PCM"), aligner=sync_aligner)
    plan = run(tts.synth("同步"))
    assert plan.timing_source == "forced_alignment"
    assert calls == [1]


def test_aligner_failure_keeps_audio_and_marks_none():
    """防回归（本轮新增的语义）：对齐失败**不丢音频**，只丢口型。

    与方案 §5 第 5 行"形象渲染失败 → 静态形象"同一原则：
    **没有口型不等于没有声音。**
    """
    t = FakeTransport(script=[HttpResponse(status=500, body=b"boom")])
    meter = RecordingMeter()
    tts = HttpTTS("qwen3", FakeTTSClient(b"PCM"), aligner=Qwen3ForcedAligner(t), meter=meter)
    plan = run(tts.synth("你好"))
    assert plan.audio == b"PCM"  # 音频照常交付
    assert plan.words == ()
    assert plan.visemes == ()
    assert plan.timing_source == "none"
    assert meter.counters.get("tts.align_failed") == 1
    assert "500" in (tts.last_align_error or "")


def test_aligner_returning_empty_marks_none_not_forced_alignment():
    """防回归：对齐跑了但没产出时，**不许谎报** forced_alignment。

    谎报的代价：下游据此以为"时间轴已经就绪"，不会走静态形象降级，
    用户看到的是"嘴一直不动但没有任何降级提示"。
    """
    t = FakeTransport(script=[ok({"words": []})])
    meter = RecordingMeter()
    tts = HttpTTS("qwen3", FakeTTSClient(b"PCM"), aligner=Qwen3ForcedAligner(t), meter=meter)
    plan = run(tts.synth("你好"))
    assert plan.audio == b"PCM"
    assert plan.timing_source == "none"
    assert meter.counters.get("tts.align_empty") == 1
    assert "tts.align_failed" not in meter.counters


def test_aligner_raising_bare_exception_is_contained():
    """对齐器抛的是裸异常（不是 Unavailable）时也要被兜住，不能裸奔出 synth()。"""

    def broken(audio: bytes, text: str) -> list[WordTiming]:
        raise RuntimeError("boom")

    meter = RecordingMeter()
    tts = HttpTTS("t", FakeTTSClient(b"PCM"), aligner=broken, meter=meter)
    plan = run(tts.synth("你好"))
    assert plan.audio == b"PCM"
    assert plan.timing_source == "none"
    assert meter.counters.get("tts.align_failed") == 1
    assert "RuntimeError" in (tts.last_align_error or "")


def test_last_align_error_is_none_when_alignment_succeeds():
    t = FakeTransport(script=[ok({"words": [{"word": "好", "start_ms": 0, "end_ms": 10}]})])
    tts = HttpTTS("q", FakeTTSClient(b"PCM"), aligner=Qwen3ForcedAligner(t))
    run(tts.synth("好"))
    assert tts.last_align_error is None
