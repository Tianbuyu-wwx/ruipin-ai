"""语音下行（桥接层的 TTS 接线）测试。

为什么单独一个文件
------------------
TTS 是**本项目唯一"会主动往外说话"的能力**，它的失败模式与评分完全不同：
评分挂了，报告上少一维、明明白白；语音挂了，客户端**什么都听不到**，
而服务端状态机一路正常走到 `report.ready`。那种"一路绿灯但用户没听见一个字"
是最难排查的一类故障，必须用测试把每条路径钉死。

本文件钉住六件事：
1. **不阻塞出题**：题干文本必须立刻下行，音频稍后到（方案 §6.3 首字延迟）。
2. **帧格式符合契约**：音频裸字节 + 口型 UTF-8 JSON，与客户端解码逻辑一致。
3. **口型整条一帧发**：客户端是"整体替换"语义，分帧等于只有最后一帧生效。
4. **空口型也要发**：否则上一题的口型会残留在这一题上继续动。
5. **任何语音失败都是显式 L4 降级**，且**一个音频字节都不许发**。
6. **未配置 TTS ≠ 故障**：不接语音的部署不该天天报降级。

另有一条与"完成度"有关的纪律：这些用例全部**不触碰真实网络与真实音频**，
但走的是真实的 `FakeTTS -> InterviewBridge -> SyncHandler -> 线序字节` 全链路。
"""

from __future__ import annotations

import asyncio
import json

import pytest

from ruipin.adapters.fakes import FakeEvaluator, FakeTTS
from ruipin.adapters.repo_sqlite import SqliteRepo
from ruipin.orchestrator import TurnScheduler
from ruipin.ports import SpeechPlan, TTSPort, VisemeEvent
from ruipin.transport import (
    BridgeConfig,
    ClientType,
    InterviewBridge,
    Opcode,
    ServerType,
    SyncHandler,
    decode_binary,
    make_envelope,
    split_outgoing,
)
from ruipin.transport.bridge import (
    DEFAULT_TTS_CHUNK_BYTES,
    OPCODE_KEY,
    encode_viseme_timeline,
)

QUESTIONS = ("请介绍你的项目经历", "讲一个你解决过的技术难题", "你如何与团队协作")


class _Clock:
    def __init__(self, start: float = 1_000.0) -> None:
        self.t = start

    def __call__(self) -> float:
        return self.t


def _make(
    *,
    tts=None,
    tts_voice: str = "default",
    tts_chunk_bytes: int = DEFAULT_TTS_CHUNK_BYTES,
    questions=QUESTIONS,
):
    repo = SqliteRepo(":memory:")
    clock = _Clock()
    cfg = BridgeConfig(questions=tuple(questions))
    bridge = InterviewBridge(
        repo=repo,
        scheduler=TurnScheduler(FakeEvaluator(), clock=clock),
        config=cfg,
        session_id="s-tts",
        clock=clock,
        tts=tts,
        tts_voice=tts_voice,
        tts_chunk_bytes=tts_chunk_bytes,
    )
    return bridge, clock, repo


def _session_create() -> object:
    return make_envelope(ClientType.SESSION_CREATE, {"candidate_id": "c-1"})


def _consent() -> object:
    return make_envelope(ClientType.CONSENT_GRANT, {"base": True})


def _commit(text: str = "这是我的回答") -> object:
    return make_envelope(ClientType.ANSWER_COMMIT, {"text": text})


def _types(frames) -> list[str]:
    return [f.type for f in frames]


def _of(frames, opcode: Opcode) -> list[object]:
    return [f for f in frames if f.payload.get(OPCODE_KEY) == int(opcode)]


def _audio_bytes(frames) -> bytes:
    return b"".join(bytes(f.payload["data"]) for f in _of(frames, Opcode.TTS_AUDIO))


def _visemes_of(frame) -> list[dict]:
    return json.loads(bytes(frame.payload["data"]).decode("utf-8"))["visemes"]


class GatedTTS:
    """可以在合成中途卡住的 TTS：用来证明"出题不等语音"。

    只放行由测试显式控制，因此"音频还没到"是一个**确定的状态**，
    而不是靠 `sleep` 赌时序（赌时序的测试在本机快、在 CI 慢时随机翻绿）。
    """

    def __init__(self, sample_rate: int = 24000) -> None:
        self.name = "gated"
        self.provider = "gated"
        self.sample_rate = sample_rate
        self.calls: list[str] = []
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def synth(self, text: str, *, voice: str = "default") -> SpeechPlan:
        self.calls.append(text)
        self.started.set()
        await self.release.wait()
        return SpeechPlan(
            audio=b"\x01\x02\x03\x04",
            visemes=(VisemeEvent(t_ms=0, viseme="aa", weight=1.0),),
            sample_rate=self.sample_rate,
            provider=self.provider,
        )

    async def synth_stream(self, text: str):  # pragma: no cover - 协议完整性所需
        raise NotImplementedError


class NoSampleRateTTS(FakeTTS):
    """故意**不带** `sample_rate` 属性的 TTS，用于覆盖"部署未声明采样率"的回退分支。"""

    def __init__(self) -> None:
        super().__init__()
        del self.sample_rate


# ---------------------------------------------------------------- 未配置 TTS


def test_without_tts_there_is_no_audio_no_viseme_and_no_degradation():
    """没接语音的部署：纯文本照常，**不报降级**（配置事实 ≠ 故障）。"""

    async def main():
        bridge, _, _ = _make(tts=None)
        await bridge.handle(_session_create())
        out = await bridge.handle(_consent())
        await bridge.joined()
        return out, bridge.take_pending()

    out, pending = asyncio.run(main())
    assert pending == [], "未配置 TTS 不该有任何推送帧"
    assert _of(out, Opcode.TTS_AUDIO) == []
    assert _of(out, Opcode.VISEME) == []
    assert ServerType.DEGRADATION_CHANGED not in _types(out)
    q = next(f for f in out if f.type == ServerType.QUESTION_START)
    assert q.payload["tts"] is False
    assert q.payload["audio_sample_rate"] == 0, "未配置时应为 0（不存在），不是默认采样率"


def test_question_start_still_carries_the_text_when_tts_is_configured():
    async def main():
        bridge, _, _ = _make(tts=FakeTTS())
        await bridge.handle(_session_create())
        out = await bridge.handle(_consent())
        await bridge.joined()
        return out

    out = asyncio.run(main())
    q = next(f for f in out if f.type == ServerType.QUESTION_START)
    assert q.payload["text"] == QUESTIONS[0]
    assert q.payload["tts"] is True
    assert q.payload["audio_sample_rate"] == 24000


# ---------------------------------------------------------------- 正常路径


def test_audio_and_viseme_frames_are_produced_for_each_question():
    tts = FakeTTS()

    async def main():
        bridge, _, _ = _make(tts=tts)
        await bridge.handle(_session_create())
        await bridge.handle(_consent())
        await bridge.joined()
        return bridge.take_pending()

    pending = asyncio.run(main())
    assert _of(pending, Opcode.TTS_AUDIO), "必须产出音频块"
    vis = _of(pending, Opcode.VISEME)
    assert len(vis) == 1, "口型时间轴必须**整条一帧**（客户端是整体替换语义）"
    assert tts.synth_calls == [QUESTIONS[0]], "每道题合成一次，且合的是题干原文"
    assert set(f.session_id for f in pending) == {"s-tts"}


def test_audio_bytes_round_trip_through_the_wire_format():
    """把下行信封真的编成二进制帧再解回来——证明"帧能上网"。"""

    async def main():
        bridge, _, _ = _make(tts=FakeTTS(n_chunks=2, chunk_size=8), tts_chunk_bytes=8)
        await bridge.handle(_session_create())
        immediate = await bridge.handle(_consent())
        await bridge.joined()
        # 真实链路里两类帧混在同一个 outbox 里，这里照原样合起来验证。
        return immediate + bridge.take_pending()

    frames = asyncio.run(main())
    texts, binaries = split_outgoing(frames)
    assert texts, "文本帧（question.start 等）仍走文本面"
    decoded = [decode_binary(b) for b in binaries]
    ops = [op for op, _ in decoded]
    assert Opcode.TTS_AUDIO in ops and Opcode.VISEME in ops
    audio = b"".join(payload for op, payload in decoded if op is Opcode.TTS_AUDIO)
    # 直接比对字节内容而不是长度：FakeTTS 的块是 `%04d|文本`，长度随文本编码而变，
    # 手算长度只会得到"数字对不上"这种没有信息量的失败。
    expected = b"".join(
        f"{i:04d}|{QUESTIONS[0]}".encode("utf-8") for i in range(2)
    )
    assert audio == expected, "解回来的字节必须与合成产物一字不差"


def test_audio_is_split_into_chunks_of_at_most_the_configured_size():
    tts = FakeTTS()

    async def main():
        bridge, _, _ = _make(tts=tts, tts_chunk_bytes=8)
        await bridge.handle(_session_create())
        await bridge.handle(_consent())
        await bridge.joined()
        return bridge.take_pending()

    pending = asyncio.run(main())
    chunks = [bytes(f.payload["data"]) for f in _of(pending, Opcode.TTS_AUDIO)]
    assert len(chunks) > 1, "块长设成 8 字节就该被切碎（否则分片逻辑根本没生效）"
    assert all(0 < len(c) <= 8 for c in chunks)
    assert b"".join(chunks), "切片后拼回来不能是空的"


def test_viseme_frame_carries_t_ms_viseme_and_weight_as_json():
    async def main():
        bridge, _, _ = _make(tts=FakeTTS())
        await bridge.handle(_session_create())
        await bridge.handle(_consent())
        await bridge.joined()
        return bridge.take_pending()

    pending = asyncio.run(main())
    frame = _of(pending, Opcode.VISEME)[0]
    visemes = _visemes_of(frame)
    assert len(visemes) == len(QUESTIONS[0]), "FakeTTS 按字切词，每条一个口型"
    assert visemes[0]["t_ms"] == 0
    assert isinstance(visemes[0]["viseme"], str)
    assert visemes[0]["weight"] == 1.0
    assert [v["t_ms"] for v in visemes] == sorted(v["t_ms"] for v in visemes), (
        "时间轴必须单调，否则口型会来回跳"
    )


def test_empty_viseme_timeline_is_still_sent_to_clear_the_previous_one():
    """空时间轴也必须发一帧：不然上一题的口型会继续演这一题。"""

    class _NoVisemeTTS(FakeTTS):
        async def synth(self, text: str, *, voice: str = "default") -> SpeechPlan:
            plan = await super().synth(text, voice=voice)
            return SpeechPlan(
                audio=plan.audio,
                words=plan.words,
                visemes=(),  # 引擎没给出时间轴
                sample_rate=plan.sample_rate,
                provider=plan.provider,
                timing_source="none",
            )

    async def main():
        bridge, _, _ = _make(tts=_NoVisemeTTS())
        await bridge.handle(_session_create())
        await bridge.handle(_consent())
        await bridge.joined()
        return bridge.take_pending()

    pending = asyncio.run(main())
    vis = _of(pending, Opcode.VISEME)
    assert len(vis) == 1, "无口型数据时仍要发一帧（空时间轴）"
    assert _visemes_of(vis[0]) == []
    assert _of(pending, Opcode.TTS_AUDIO), "口型为空不影响音频照常下发"


def test_voice_and_sample_rate_are_passed_through():
    tts = FakeTTS(sample_rate=16000)

    async def main():
        bridge, _, _ = _make(tts=tts, tts_voice="female-warm")
        await bridge.handle(_session_create())
        out = await bridge.handle(_consent())
        await bridge.joined()
        return out

    out = asyncio.run(main())
    assert tts.voices == ["female-warm"]
    q = next(f for f in out if f.type == ServerType.QUESTION_START)
    assert q.payload["audio_sample_rate"] == 16000


def test_sample_rate_falls_back_when_the_adapter_does_not_declare_one():
    tts = NoSampleRateTTS()

    async def main():
        bridge, _, _ = _make(tts=tts)
        await bridge.handle(_session_create())
        out = await bridge.handle(_consent())
        await bridge.joined()
        return out

    out = asyncio.run(main())
    q = next(f for f in out if f.type == ServerType.QUESTION_START)
    assert q.payload["audio_sample_rate"] == 24000


def test_each_question_is_synthesized_once():
    tts = FakeTTS()

    async def main():
        bridge, _, _ = _make(tts=tts)
        await bridge.handle(_session_create())
        await bridge.handle(_consent())
        for _ in QUESTIONS:
            await bridge.handle(_commit())
        await bridge.joined()
        return bridge.take_pending()

    asyncio.run(main())
    assert tts.synth_calls == list(QUESTIONS), "题面必须一一对应，不能漏合或重合成"


# ---------------------------------------------------------------- 不阻塞出题


def test_question_start_is_not_delayed_by_a_slow_tts():
    """核心时序约束：题干先到，语音后到（方案 §6.3 的首字延迟）。"""

    async def main():
        tts = GatedTTS()
        bridge, _, _ = _make(tts=tts)
        await bridge.handle(_session_create())
        out = await bridge.handle(_consent())
        # TTS 还卡在 release.wait()，此刻题干必须已经在返回帧里
        assert ServerType.QUESTION_START in _types(out)
        assert bridge.tts_tasks_pending == 1
        assert bridge.take_pending() == [], "语音还没合成完，不该有音频帧"
        tts.release.set()
        await bridge.joined()
        assert bridge.tts_tasks_pending == 0
        return bridge.take_pending()

    pending = asyncio.run(main())
    assert _of(pending, Opcode.TTS_AUDIO)
    assert len(_of(pending, Opcode.VISEME)) == 1


def test_sync_handler_pending_includes_in_flight_synthesis():
    """读循环靠 `pending == 0` 判断"可以收尾了"，漏算 TTS 会**说半句话就断线**。"""

    async def main():
        tts = GatedTTS()
        bridge, _, _ = _make(tts=tts)
        handler = SyncHandler(bridge, asyncio.get_running_loop())
        handler(_session_create())
        handler(_consent())
        await asyncio.sleep(0)  # 让工作协程把队列抽干（但 TTS 仍卡着）
        busy = handler.pending
        before = handler.drain()  # 此刻只该有文本帧，一个音频字节都不该有
        released = bool(_of(before, Opcode.TTS_AUDIO))
        tts.release.set()
        await handler.joined()
        return busy, released, _types(before), handler.drain()

    busy, audio_before, types_before, after = asyncio.run(main())
    assert busy >= 1, "语音还没合成完，pending 不能是 0"
    assert not audio_before, "此刻不该已经有音频帧"
    assert ServerType.QUESTION_START in types_before, "但题干文本帧必须已经到了"
    assert _of(after, Opcode.TTS_AUDIO), "joined() 之后音频必须已进入 outbox"


# ---------------------------------------------------------------- 失败路径


def test_tts_unavailable_yields_explicit_l4_degradation_and_zero_audio():
    async def main():
        bridge, _, _ = _make(tts=FakeTTS(fail=True))
        await bridge.handle(_session_create())
        out = await bridge.handle(_consent())
        await bridge.joined()
        return out, bridge.take_pending()

    out, pending = asyncio.run(main())
    assert _of(pending, Opcode.TTS_AUDIO) == [], "失败时一个音频字节都不许发"
    assert _of(pending, Opcode.VISEME) == [], "失败时也不许发口型（否则口型在动却没有声音）"
    deg = [f for f in pending if f.type == ServerType.DEGRADATION_CHANGED]
    assert len(deg) == 1
    assert deg[0].payload["level"] == 4
    assert "tts_unavailable" in deg[0].payload["reason"]
    assert deg[0].payload["badge"], "降级必须带可展示的中文徽标"
    assert ServerType.QUESTION_START in _types(out), "语音挂了，文本照常出题"


def test_empty_audio_is_treated_as_failure_not_as_silence():
    """红线：**静音不许冒充成功**。空音频必须显式降级。"""

    async def main():
        bridge, _, _ = _make(tts=FakeTTS(empty_audio=True))
        await bridge.handle(_session_create())
        await bridge.handle(_consent())
        await bridge.joined()
        return bridge.take_pending()

    pending = asyncio.run(main())
    assert _of(pending, Opcode.TTS_AUDIO) == []
    deg = [f for f in pending if f.type == ServerType.DEGRADATION_CHANGED]
    assert len(deg) == 1 and deg[0].payload["level"] == 4
    assert "空音频" in deg[0].payload["reason"]


def test_unexpected_tts_exception_becomes_a_visible_degradation():
    """后台任务的异常最容易变成一行 "Task exception was never retrieved"。"""

    class _BoomTTS(FakeTTS):
        async def synth(self, text: str, *, voice: str = "default") -> SpeechPlan:
            raise RuntimeError("解码器内存对齐失败")

    async def main():
        bridge, _, _ = _make(tts=_BoomTTS())
        await bridge.handle(_session_create())
        await bridge.handle(_consent())
        await bridge.joined()
        return bridge.take_pending()

    pending = asyncio.run(main())
    deg = [f for f in pending if f.type == ServerType.DEGRADATION_CHANGED]
    assert len(deg) == 1
    reason = deg[0].payload["reason"]
    assert "RuntimeError" in reason, "必须带异常类型（只写文案无法排查）"
    assert "内存对齐" in reason, "原始信息不能被丢掉"
    assert deg[0].payload["level"] == 4


def test_cancelled_synthesis_is_not_reported_as_a_failure():
    """取消 ≠ 故障：不能往客户端推一条"TTS 挂了"的降级帧。"""

    async def main():
        tts = GatedTTS()
        bridge, _, _ = _make(tts=tts)
        await bridge.handle(_session_create())
        await bridge.handle(_consent())
        assert bridge.tts_tasks_pending == 1
        # 关键：先让任务**真的开始跑**（卡进 release.wait）再取消。
        # 不然取消发生在"任务还没被调度"时，协程体一行都不执行，
        # 于是 `except CancelledError: raise` 那条分支永远没被验证过。
        await asyncio.sleep(0)
        assert tts.started.is_set(), "合成必须已经真正开始，否则测的不是取消在途任务"
        # 刻意检查私有集合：这里要证明的正是"取消路径不留任何降级帧"
        (task,) = list(bridge._tts_tasks)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return bridge.take_pending(), bridge.tts_tasks_pending

    pending, pending_tasks = asyncio.run(main())
    assert pending_tasks == 0, "被取消的任务不能继续占着 pending（否则收尾会永远等下去）"
    assert pending == [], "取消不是故障，不该产生降级帧"


def test_pushing_nothing_is_a_noop():
    """空推送必须无副作用：它不该往 outbox 塞一个"空批次"。"""

    async def main():
        received: list = []
        bridge, _, _ = _make(tts=None)
        bridge.set_pusher(received.extend)
        bridge._push([])  # noqa: SLF001 - 直接验证出口的空输入路径
        return received

    assert asyncio.run(main()) == []


def test_the_interview_still_completes_when_tts_is_broken_throughout():
    """整场语音全挂，文本面试必须照常走到报告——语音是增强项，不是依赖项。"""

    async def main():
        bridge, _, _ = _make(tts=FakeTTS(fail=True))
        await bridge.handle(_session_create())
        frames = await bridge.handle(_consent())
        for _ in QUESTIONS:
            frames += await bridge.handle(_commit())
        frames += await bridge.handle(make_envelope(ClientType.CONTROL_END, {}))
        await bridge.joined()
        frames += bridge.take_pending()
        return frames

    frames = asyncio.run(main())
    assert ServerType.REPORT_READY in _types(frames), "语音挂了也必须出报告"
    assert _of(frames, Opcode.TTS_AUDIO) == []
    assert len([f for f in frames if f.type == ServerType.DEGRADATION_CHANGED]) >= 1


def test_tts_chunk_bytes_must_be_positive():
    with pytest.raises(ValueError, match="tts_chunk_bytes"):
        _make(tts=None, tts_chunk_bytes=0)


# ---------------------------------------------------------------- 出口接线


def test_pusher_takes_over_the_local_pending_queue():
    received: list = []

    async def main():
        bridge, _, _ = _make(tts=FakeTTS(), )
        bridge.set_pusher(received.extend)
        await bridge.handle(_session_create())
        await bridge.handle(_consent())
        await bridge.joined()
        return bridge.take_pending()

    local = asyncio.run(main())
    assert local == [], "绑定了 pusher 之后本地队列不该再收帧（否则帧会被发两次）"
    assert _of(received, Opcode.TTS_AUDIO)


def test_sync_handler_binds_the_pusher_at_construction_time():
    """晚绑定会让"绑定前就合成完的首题语音"永久卡在本地队列里。"""

    async def main():
        bridge, _, _ = _make(tts=FakeTTS())
        handler = SyncHandler(bridge, asyncio.get_running_loop())
        handler(_session_create())
        handler(_consent())
        await handler.joined()
        return handler.drain(), bridge.take_pending()

    drained, local = asyncio.run(main())
    assert _of(drained, Opcode.TTS_AUDIO), "音频必须落在 SyncHandler 的 outbox 里"
    assert local == []


def test_joined_returns_immediately_when_there_is_nothing_to_wait_for():
    async def main():
        bridge, _, _ = _make(tts=FakeTTS())
        await bridge.joined()
        await bridge.joined()
        return bridge.tts_tasks_pending

    assert asyncio.run(main()) == 0


# ---------------------------------------------------------------- 编码器单测


def test_encode_viseme_timeline_shape():
    raw = encode_viseme_timeline(
        (
            VisemeEvent(t_ms=0, viseme="aa", weight=1.0),
            VisemeEvent(t_ms=120, viseme="闭嘴", weight=0.25),
        )
    )
    payload = json.loads(raw.decode("utf-8"))
    assert payload == {
        "visemes": [
            {"t_ms": 0, "viseme": "aa", "weight": 1.0},
            {"t_ms": 120, "viseme": "闭嘴", "weight": 0.25},
        ]
    }
    assert "闭嘴" in raw.decode("utf-8"), "中文 viseme 名必须原样（不能转成 \\uXXXX 让前端难看）"


def test_encode_viseme_timeline_for_empty_input():
    assert json.loads(encode_viseme_timeline(()).decode("utf-8")) == {"visemes": []}


# ---------------------------------------------------------------- 替身自身


def test_fake_tts_splits_whitespace_delimited_words_for_latin_text():
    """替身的时间轴切分：有空白就按词切，没空白（中文）才退化成按字切。"""
    plan = asyncio.run(FakeTTS().synth("Hello world"))
    assert [w.word for w in plan.words] == ["Hello", "world"]
    assert [v.t_ms for v in plan.visemes] == [0, 200], "两条口型按 word_ms 递增"


def test_fake_tts_satisfies_the_whole_tts_port():
    """替身必须**成对**实现 `synth` / `synth_stream`。

    只实现一个的话，桥接层会在出题时撞 `AttributeError`，
    而那个异常会被翻成"TTS 故障"的降级帧——"替身不完整"就此伪装成"能力挂了"。
    """
    from ruipin.ports import TTSPort

    tts = FakeTTS()
    assert isinstance(tts, TTSPort)
    assert isinstance(tts.sample_rate, int) and tts.sample_rate > 0
