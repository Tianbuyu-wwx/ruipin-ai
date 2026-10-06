"""信令 ↔ 业务接线（`transport/bridge.py`）的测试。

这是整个系统里**唯一**同时认识协议与业务的地方，也是最容易出"左耳进右耳出"问题
的地方：网关觉得自己转发成功，编排器觉得自己收到了指令，中间那根线其实是断的。
因此这里的用例以**端到端**为主——用真实的 `Gateway` + `InterviewBridge` +
`InterviewService` + `SqliteRepo` 走完一场面试，而不是逐个函数 mock 过去。

重点钉住四件事：
1. **状态回放完整性**：`state.changed` 的序列必须与领域事件日志**逐条一致**。
   断线重连靠它重建 UI；只要漏发一条，客户端就会卡在一个不存在的状态上。
2. **协议 → 领域对象**：分项授权、生理批次、事件锚点的字段映射不能错位。
3. **失败显式**：未实现的功能、用错帧类型、能力不可用，都必须回结构化 `error`
   或 `degradation.changed`，绝不静默。
4. **红线穿透到协议层**：整场没有真实评估时，`report.ready` 里必须是
   `available=False` / `score=None`，不能因为"加了层网络"就冒出个数字。
"""

from __future__ import annotations

import asyncio

import pytest

from ruipin.adapters.fakes import FakeEvaluator
from ruipin.adapters.repo_sqlite import SqliteRepo
from ruipin.orchestrator import TurnScheduler
from ruipin.transport import (
    CLIENT_TYPES,
    BridgeConfig,
    ClientType,
    ErrorCode,
    Gateway,
    InterviewBridge,
    Opcode,
    ServerType,
    SyncHandler,
    encode_binary,
    encode_text,
    make_envelope,
    media_envelope,
    split_outgoing,
)

QUESTIONS = ("请介绍你的项目经历", "讲一个你解决过的技术难题", "你如何与团队协作")


class _Clock:
    """可控时钟（秒）。桥接层要毫秒时间戳，故内部乘 1000。"""

    def __init__(self, start: float = 1_000.0) -> None:
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


def _run(coro):
    return asyncio.run(coro)


def _make(
    *,
    questions=QUESTIONS,
    physio: bool = False,
    evaluator: FakeEvaluator | None = None,
    adaptive: bool = False,
) -> tuple[InterviewBridge, _Clock, SqliteRepo]:
    repo = SqliteRepo(":memory:")
    clock = _Clock()
    cfg = BridgeConfig(
        questions=tuple(questions),
        buffer_questions=("先深呼吸，说说你的感受",) if adaptive else (),
        physio_enabled=physio,
        adaptive_enabled=adaptive,
    )
    bridge = InterviewBridge(
        repo=repo,
        scheduler=TurnScheduler(evaluator or FakeEvaluator(), clock=clock),
        config=cfg,
        session_id="s-bridge",
        clock=clock,
    )
    return bridge, clock, repo


def _types(frames) -> list[str]:
    return [f.type for f in frames]


def _find(frames, mtype):
    for f in frames:
        if f.type == mtype:
            return f
    return None


async def _drive_full_interview(bridge: InterviewBridge, clock: _Clock) -> list:
    """完整驱动一场面试，返回 `control.end` 的响应帧。"""
    await bridge.handle(make_envelope(ClientType.SESSION_CREATE, {"candidate_id": "c1"}))
    await bridge.handle(make_envelope(ClientType.CONSENT_GRANT, {"base": True}))
    out = []
    for idx, _q in enumerate(QUESTIONS):
        clock.advance(2.0)
        await bridge.handle(make_envelope(ClientType.ANSWER_TEXT, {"text": f"回答{idx}"}))
        clock.advance(20.0)
        out = await bridge.handle(make_envelope(ClientType.ANSWER_COMMIT, {}))
    clock.advance(1.0)
    return await bridge.handle(make_envelope(ClientType.CONTROL_END, {}))


# ---------- 路由表与协议的对拍 ----------


def test_every_client_type_has_a_route():
    """★ 防回归：协议里定义的每一种客户端类型都必须有处理函数。

    漏掉一条的后果是"前端发了、后端装不认识"，而且只在那条路径被走到时才暴露。
    """
    from ruipin.transport.bridge import ROUTES

    assert set(ROUTES) == set(CLIENT_TYPES)


def test_unknown_type_returns_structured_error():
    """防回归：未知类型回 `unknown_type`（宽容路径），而不是崩溃或静默。"""
    bridge, _c, _r = _make()
    out = _run(bridge.handle(make_envelope("future.feature", {})))

    assert _types(out) == [str(ServerType.ERROR)]
    assert out[0].payload["code"] == str(ErrorCode.UNKNOWN_TYPE)
    assert "future.feature" in out[0].payload["message"]


def test_barge_in_is_refused_explicitly():
    """防回归：`control.barge_in` 是方案标注的 Phase 3 功能，必须明确回绝。

    "假装成功"比"明说不支持"危险得多：前端会据此隐藏按钮、产品会以为已交付。
    """
    bridge, _c, _r = _make()
    out = _run(bridge.handle(make_envelope(ClientType.CONTROL_BARGE_IN, {})))

    assert out[0].payload["code"] == str(ErrorCode.UNKNOWN_TYPE)
    assert "尚未实现" in out[0].payload["message"]
    assert "Phase 3" in out[0].payload["message"]


# ---------- 主链路 ----------


def test_session_create_and_consent_reach_the_first_question():
    """防回归：建会话 → 授权 → 立刻出第一题（P0"文本面试闭环"的起点）。"""
    bridge, _c, _r = _make()

    first = _run(bridge.handle(make_envelope(ClientType.SESSION_CREATE, {})))
    assert _types(first) == [str(ServerType.STATE_CHANGED)]
    assert first[0].payload["to"] == "setup"

    second = _run(bridge.handle(make_envelope(ClientType.CONSENT_GRANT, {"base": True})))
    assert second[0].payload["to"] == "greeting"
    assert second[1].payload["to"] == "asking"
    q = _find(second, ServerType.QUESTION_START)
    assert q is not None
    assert q.payload["text"] == QUESTIONS[0]
    assert q.payload["turn_id"] == "s-bridge-t0"
    # 顺序：先迁移到 LISTENING，再下发题目（前端据此渲染"正在听答"）
    last_state = [f for f in second if f.type == str(ServerType.STATE_CHANGED)][-1]
    assert last_state.payload["to"] == "listening"
    assert _types(second)[-1] == str(ServerType.QUESTION_START)


def test_refusing_consent_aborts_without_any_question():
    """防回归：拒绝参加 → ABORTED，且**不出题**（不能先问了再说）。"""
    bridge, _c, _r = _make()
    _run(bridge.handle(make_envelope(ClientType.SESSION_CREATE, {})))
    out = _run(bridge.handle(make_envelope(ClientType.CONSENT_GRANT, {"base": False})))

    assert out[-1].payload["to"] == "aborted"
    assert _find(out, ServerType.QUESTION_START) is None


def test_answer_text_accumulates_into_partial_transcript():
    """防回归：流式作答要累积，不能每来一段就覆盖（前端只显示最后一段是常见事故）。"""
    bridge, _c, _r = _make()
    _run(bridge.handle(make_envelope(ClientType.SESSION_CREATE, {})))
    _run(bridge.handle(make_envelope(ClientType.CONSENT_GRANT, {"base": True})))

    a = _run(bridge.handle(make_envelope(ClientType.ANSWER_TEXT, {"text": "第一段"})))
    b = _run(bridge.handle(make_envelope(ClientType.ANSWER_TEXT, {"text": "，第二段"})))

    assert a[0].payload["text"] == "第一段"
    assert b[0].payload["text"] == "第一段，第二段"


def test_answer_text_with_non_string_payload_is_rejected():
    """防回归：payload.text 不是字符串时回 `bad_frame`，不把 None 拼进转写。"""
    bridge, _c, _r = _make()
    _run(bridge.handle(make_envelope(ClientType.SESSION_CREATE, {})))
    _run(bridge.handle(make_envelope(ClientType.CONSENT_GRANT, {"base": True})))

    out = _run(bridge.handle(make_envelope(ClientType.ANSWER_TEXT, {"text": 123})))
    assert out[0].payload["code"] == str(ErrorCode.BAD_FRAME)


def test_commit_emits_transcript_eval_and_next_question():
    """防回归：一轮作答要按 转写终稿 → 评估 → 下一题 的顺序下行。"""
    bridge, clock, _r = _make()
    _run(bridge.handle(make_envelope(ClientType.SESSION_CREATE, {})))
    _run(bridge.handle(make_envelope(ClientType.CONSENT_GRANT, {"base": True})))
    clock.advance(2.0)
    _run(bridge.handle(make_envelope(ClientType.ANSWER_TEXT, {"text": "我的回答"})))
    clock.advance(20.0)

    out = _run(bridge.handle(make_envelope(ClientType.ANSWER_COMMIT, {})))
    types = _types(out)

    assert str(ServerType.TRANSCRIPT_FINAL) in types
    assert str(ServerType.EVAL_DONE) in types
    assert str(ServerType.QUESTION_START) in types
    assert types.index(str(ServerType.TRANSCRIPT_FINAL)) < types.index(str(ServerType.EVAL_DONE))
    assert types.index(str(ServerType.EVAL_DONE)) < types.index(str(ServerType.QUESTION_START))

    ev = _find(out, ServerType.EVAL_DONE)
    assert ev.payload["turn_index"] == 0
    assert 0.0 <= ev.payload["score"] <= 100.0
    assert set(ev.payload["dims"]) == {
        "technical", "communication", "completeness",
        "problem_solving", "teamwork", "leadership",
    }
    assert _find(out, ServerType.QUESTION_START).payload["text"] == QUESTIONS[1]


def test_final_commit_moves_to_candidate_qa():
    """防回归：最后一题答完 → 进入候选人提问环节，且不再下发新题。"""
    bridge, clock, _r = _make()

    async def go():
        frames = []
        frames += await bridge.handle(make_envelope(ClientType.SESSION_CREATE, {}))
        frames += await bridge.handle(make_envelope(ClientType.CONSENT_GRANT, {"base": True}))
        for idx in range(len(QUESTIONS)):
            clock.advance(2.0)
            frames += await bridge.handle(
                make_envelope(ClientType.ANSWER_TEXT, {"text": f"回答{idx}"})
            )
            clock.advance(20.0)
            frames += await bridge.handle(make_envelope(ClientType.ANSWER_COMMIT, {}))
        clock.advance(1.0)
        frames += await bridge.handle(make_envelope(ClientType.CONTROL_END, {}))
        return frames

    frames = _run(go())

    history = [
        f.payload["to"] for f in frames if f.type == str(ServerType.STATE_CHANGED)
    ]
    assert "candidate_qa" in history
    assert "reporting" in history
    # ★ 题尽之后不得再下发新题：提问环节的题数必须恰好等于题目数
    starts = [f for f in frames if f.type == str(ServerType.QUESTION_START)]
    assert len(starts) == len(QUESTIONS)
    assert history.index("candidate_qa") > history.index("asking")


# ---------- 状态回放完整性（最关键的一条） ----------


def test_state_changed_sequence_matches_domain_event_log_exactly():
    """★ 红线：`state.changed` 的 (from,to,reason) 序列必须与领域日志**逐条一致**。

    这是"断线重连能重建 UI"的唯一依据。少发一条，客户端就会停在一个
    服务端早就离开的状态上；多发一条，客户端会执行一个不存在的迁移。
    所以比对的是**全序列**，不是"包含"或"数量相等"。
    """
    bridge, clock, _r = _make()
    collected: list = []

    async def go():
        collected.extend(await bridge.handle(make_envelope(ClientType.SESSION_CREATE, {})))
        collected.extend(await bridge.handle(make_envelope(ClientType.CONSENT_GRANT, {"base": True})))
        for idx in range(len(QUESTIONS)):
            clock.advance(2.0)
            collected.extend(
                await bridge.handle(make_envelope(ClientType.ANSWER_TEXT, {"text": f"a{idx}"}))
            )
            clock.advance(20.0)
            collected.extend(await bridge.handle(make_envelope(ClientType.ANSWER_COMMIT, {})))
        clock.advance(1.0)
        collected.extend(await bridge.handle(make_envelope(ClientType.CONTROL_END, {})))

    _run(go())

    signaled = [
        (f.payload["from"], f.payload["to"], f.payload["reason"])
        for f in collected
        if f.type == str(ServerType.STATE_CHANGED)
    ]
    logged = [
        (e.from_state.value if e.from_state else None, e.to_state.value, e.event.value)
        for e in bridge.service.session.events
    ]

    assert signaled == logged
    assert len(logged) >= 12  # 一场完整面试至少十几步，防空跑


def test_state_cursor_does_not_resend_old_transitions():
    """防回归：游标要真的前进；重复下发同一批 `state.changed` 会让前端重复播放动画。"""
    bridge, _c, _r = _make()
    first = _run(bridge.handle(make_envelope(ClientType.SESSION_CREATE, {})))
    second = _run(bridge.handle(make_envelope(ClientType.SESSION_CREATE, {})))

    assert _types(first) == [str(ServerType.STATE_CHANGED)]
    # 第二次 create 非法（已在 SETUP）→ 回 error，且**没有**重复的 state.changed
    assert _types(second) == [str(ServerType.ERROR)]
    assert second[0].payload["code"] == str(ErrorCode.BAD_FRAME)


# ---------- 失败必须显式 ----------


def test_invalid_transition_becomes_error_frame_not_exception():
    """防回归：业务异常不能穿透到传输层（会把整条 WS 连接打断）。"""
    bridge, _c, _r = _make()
    _run(bridge.handle(make_envelope(ClientType.SESSION_CREATE, {})))
    _run(bridge.handle(make_envelope(ClientType.CONSENT_GRANT, {"base": True})))

    # 正在听答时再发 consent.grant → GREETING 之外的状态不接受 CONSENT_OK
    out = _run(bridge.handle(make_envelope(ClientType.CONSENT_GRANT, {"base": True})))
    assert _types(out) == [str(ServerType.ERROR)]
    assert "当前状态不允许该操作" in out[0].payload["message"]


def test_unavailable_evaluator_degrades_visibly_but_report_has_no_number():
    """★ 双红线：评估不可用时(a)显式下行降级，(b)报告仍不给任何数字。

    (a) 是"禁止静默降级"（方案 §3.4）；(b) 是"绝不合成兜底分数"（本项目第一纪律）。
    两条一起测，是因为它们很容易被"反正报告里没有分"这种理由吞掉其中一条。
    """
    bridge, clock, _r = _make(evaluator=FakeEvaluator(fail=True))
    out = _run(_drive_full_interview(bridge, clock))

    commit_frames = []  # 只看最后一轮的下行
    assert _find(out, ServerType.REPORT_READY) is not None

    report = _find(out, ServerType.REPORT_READY).payload
    assert report["available"] is False
    assert report["score"] is None
    assert report["level"] is None
    assert "不生成兜底分数" in report["reason"]
    assert commit_frames == []


def test_degradation_frame_is_emitted_on_eval_failure():
    """防回归：每轮失败都要有一条 `degradation.changed`，带中文徽标，不许静默。"""
    bridge, clock, _r = _make(evaluator=FakeEvaluator(fail=True))
    _run(bridge.handle(make_envelope(ClientType.SESSION_CREATE, {})))
    _run(bridge.handle(make_envelope(ClientType.CONSENT_GRANT, {"base": True})))
    clock.advance(2.0)
    _run(bridge.handle(make_envelope(ClientType.ANSWER_TEXT, {"text": "答"})))
    clock.advance(20.0)

    out = _run(bridge.handle(make_envelope(ClientType.ANSWER_COMMIT, {})))
    deg = _find(out, ServerType.DEGRADATION_CHANGED)

    assert deg is not None
    assert deg.payload["level"] >= 1
    assert deg.payload["badge"]  # 非空文案，前端要直接显示
    assert _find(out, ServerType.EVAL_DONE) is None  # 没有真实评估就不该有 eval.done


def test_text_media_frame_is_refused_with_pointer_to_binary():
    """防回归：媒体必须走二进制帧；文本形态静默接受会让前端以为上传成功了。"""
    bridge, _c, _r = _make()
    out = _run(bridge.handle(make_envelope(ClientType.MEDIA_VIDEO, {})))

    assert out[0].payload["code"] == str(ErrorCode.BAD_FRAME)
    assert "二进制" in out[0].payload["message"]


# ---------- 生理链路 ----------


def test_physio_batch_rejected_when_module_disabled():
    """防回归：本场没开生理采集时，批次要被明确拒收（而不是悄悄存下来）。"""
    bridge, _c, _r = _make(physio=False)
    out = _run(bridge.handle(make_envelope(ClientType.PHYSIO_BATCH, {"samples": []})))

    assert out[0].payload["code"] == str(ErrorCode.UNAUTHORIZED)
    assert "未启用" in out[0].payload["message"]


def test_physio_batch_rejected_without_separate_consent():
    """★ 红线（PIPL 第 28 条）：开了采集但**没有单独授权**时，数据必须被拒收。

    注意断言的是"数据没进内存"，而不只是"回了个错误"——"收了再弃用"同样不合规。
    """
    bridge, _c, _r = _make(physio=True)
    _run(bridge.handle(make_envelope(ClientType.SESSION_CREATE, {})))
    _run(bridge.handle(make_envelope(ClientType.CONSENT_GRANT, {"base": True})))  # 未勾生理

    out = _run(
        bridge.handle(
            make_envelope(
                ClientType.PHYSIO_BATCH,
                {"samples": [{"t_ms": 0, "bpm": 71.0, "snr": 0.9}]},
            )
        )
    )

    assert out[0].payload["code"] == str(ErrorCode.UNAUTHORIZED)
    assert "单独授权" in out[0].payload["message"]
    assert bridge.collector is not None
    assert bridge.collector.n_samples == 0


def test_physio_batch_accepted_after_separate_consent_and_reports_baseline():
    """防回归：单独授权后批次被接收；`rest_end` 触发基线计算并回 `metric.hr`。"""
    bridge, _c, _r = _make(physio=True)
    _run(bridge.handle(make_envelope(ClientType.SESSION_CREATE, {})))
    _run(
        bridge.handle(
            make_envelope(ClientType.CONSENT_GRANT, {"base": True, "physiology": True})
        )
    )

    samples = [{"t_ms": t, "bpm": 70.0, "snr": 0.9} for t in range(0, 60_001, 2_000)]
    accepted = _run(
        bridge.handle(make_envelope(ClientType.PHYSIO_BATCH, {"samples": samples}))
    )
    assert accepted == []  # 静默接收（每 2s 一批，不该每批都回帧）

    out = _run(
        bridge.handle(
            make_envelope(
                ClientType.PHYSIO_BATCH,
                {"samples": [], "rest_end": True, "algo_agreement": 1.0},
            )
        )
    )
    metric = _find(out, ServerType.METRIC_HR)
    assert metric is not None
    assert metric.payload["phase"] == "baseline"
    assert metric.payload["available"] is True
    assert metric.payload["baseline_bpm"] == pytest.approx(70.0)
    assert bridge.collector is not None and bridge.collector.n_samples == len(samples)


def test_bad_physio_sample_shape_is_rejected():
    """防回归：样本缺 `t_ms` 时回 `bad_frame`，不许按 0 猜时间。"""
    bridge, _c, _r = _make(physio=True)
    _run(bridge.handle(make_envelope(ClientType.SESSION_CREATE, {})))
    _run(bridge.handle(make_envelope(ClientType.CONSENT_GRANT, {"base": True, "physio": True})))

    out = _run(
        bridge.handle(make_envelope(ClientType.PHYSIO_BATCH, {"samples": [{"bpm": 70.0}]}))
    )
    assert out[0].payload["code"] == str(ErrorCode.BAD_FRAME)
    assert "t_ms" in out[0].payload["message"]


def test_low_snr_rest_period_reason_reaches_the_report():
    """★ 防回归：环境不支持（静息 SNR 过低）的**根因**必须一路传到报告里。

    如果中途被降级成"基线缺失"，候选人会以为是自己没测好，
    而真实原因是灯太暗/脸太小——这是"报错要报根因"的具体体现。
    """
    bridge, _c, _r = _make(physio=True)
    _run(bridge.handle(make_envelope(ClientType.SESSION_CREATE, {})))
    _run(bridge.handle(make_envelope(ClientType.CONSENT_GRANT, {"base": True, "physio": True})))

    dark = [{"t_ms": t, "bpm": 70.0, "snr": 0.15} for t in range(0, 60_001, 2_000)]
    _run(bridge.handle(make_envelope(ClientType.PHYSIO_BATCH, {"samples": dark})))
    out = _run(
        bridge.handle(make_envelope(ClientType.PHYSIO_BATCH, {"samples": [], "rest_end": True}))
    )
    assert _find(out, ServerType.METRIC_HR).payload["available"] is False

    # 答完题并结束
    for idx in range(len(QUESTIONS)):
        _run(bridge.handle(make_envelope(ClientType.ANSWER_TEXT, {"text": f"a{idx}"})))
        _run(bridge.handle(make_envelope(ClientType.ANSWER_COMMIT, {})))
    end = _run(bridge.handle(make_envelope(ClientType.CONTROL_END, {})))

    report = _find(end, ServerType.REPORT_READY).payload
    assert report["physio"]["available"] is False
    assert "环境不支持" in report["physio"]["reason"]
    assert report["physio"]["weight_applied"] == 0.0
    # 生理不计入，但其它维度照常出分（不因它而被压低）
    assert report["available"] is True
    assert "stress_regulation" not in report["effective_weights"]


# ---------- 下行帧的文本/二进制分流 ----------


def test_split_outgoing_separates_media_from_control_frames():
    """防回归：带 opcode 的信封必须走二进制帧，其余走 JSON 文本帧。

    分流写错的后果很隐蔽：前端收到一条 JSON 却以为里面是音频，播放器静默无输出。
    """
    text_env = make_envelope(ServerType.QUESTION_START, {"text": "题目"})
    audio_env = media_envelope(ServerType.QUESTION_TTS_CHUNK, Opcode.TTS_AUDIO, b"\x01\x02\x03")

    texts, binaries = split_outgoing([text_env, audio_env])

    assert len(texts) == 1
    assert len(binaries) == 1
    assert texts[0].startswith("{")
    assert binaries[0][0] == int(Opcode.TTS_AUDIO)  # 首字节即 opcode


def test_media_envelope_carries_opcode_and_data():
    """防回归：媒体信封的 `opcode` 与 `data` 字段是分流依据，不能被改名。"""
    env = media_envelope(ServerType.AVATAR_VISEME, Opcode.VISEME, b"\xaa", t_ms=100, viseme="AA")

    assert env.payload["opcode"] == int(Opcode.VISEME)
    assert env.payload["data"] == b"\xaa"
    assert env.payload["t_ms"] == 100
    assert env.payload["viseme"] == "AA"


def test_roundtrip_through_binary_codec():
    """防回归：分流出来的二进制帧必须能被协议层解回原样。"""
    env = media_envelope(ServerType.QUESTION_TTS_CHUNK, Opcode.TTS_AUDIO, b"\x10\x20")
    _texts, binaries = split_outgoing([env])

    from ruipin.transport import decode_binary

    opcode, payload = decode_binary(binaries[0])
    assert int(opcode) == int(Opcode.TTS_AUDIO)
    assert payload == b"\x10\x20"


# ---------- 与 Gateway 的真实接线 ----------


def test_gateway_plus_bridge_drives_a_full_interview_over_wire_frames():
    """★ 端到端：真实 `Gateway` + `SyncHandler` + `InterviewBridge`，走**线序字节**。

    前面所有用例都是直接调 `bridge.handle(envelope)`；只有这一条会经过
    `encode_text → Gateway.handle_text → decode_text → handler → 下行帧`，
    因此能抓到"信封字段在序列化时丢了/改了名"这类只在真实链路上出现的问题。
    """
    bridge, clock, repo = _make()
    loop = asyncio.new_event_loop()
    try:
        handler = SyncHandler(bridge, loop)
        gateway = Gateway(
            handler=handler,
            clock=clock,
            token_verifier=lambda tok: "s-bridge" if tok == "token-ok" else None,
        )
        assert gateway.authenticate("token-ok") is True

        seq = 0
        pushed: list = []

        async def send(mtype, payload, advance=None):
            nonlocal seq
            seq += 1
            if advance:
                clock.advance(advance)
            raw = encode_text(
                make_envelope(mtype, payload, seq=seq, session_id="s-bridge")
            )
            immediate = gateway.handle_text(raw)
            # 等 SyncHandler 的在途任务**全部**跑完再取推送帧。
            # 早期版本用固定次数的 `await asyncio.sleep(0)`，结果 drain 会抢跑：
            # 上一帧的帧被算到下一帧的响应里，甚至因为时钟被提前推进而伪造出降级。
            await handler.joined()
            pushed.extend(handler.drain())
            return immediate

        async def go():
            await send(ClientType.SESSION_CREATE, {})
            await send(ClientType.CONSENT_GRANT, {"base": True})
            for idx in range(len(QUESTIONS)):
                await send(ClientType.ANSWER_TEXT, {"text": f"回答{idx}"}, advance=2.0)
                await send(ClientType.ANSWER_COMMIT, {}, advance=20.0)
            await send(ClientType.CONTROL_END, {}, advance=1.0)

        loop.run_until_complete(go())

        types = _types(pushed)
        assert str(ServerType.QUESTION_START) in types
        assert str(ServerType.EVAL_DONE) in types
        report = _find(pushed, ServerType.REPORT_READY)
        assert report is not None
        assert report.payload["available"] is True
        assert 0.0 <= report.payload["score"] <= 100.0
        assert bridge.service.session.state.value == "completed"

        # 落库可回放：事件日志逐条与内存一致
        events = repo.load_events("s-bridge")
        assert len(events) == len(bridge.service.session.events)
        assert [e["to"] for e in events] == [
            s.to_state.value for s in bridge.service.session.events
        ]
    finally:
        loop.close()


def test_bridge_config_without_weights_keeps_service_defaults():
    """防回归：协议层没给权重时，要沿用服务层默认权重（含生理 8%），不得覆盖成空。"""
    default_w = BridgeConfig(questions=("q",)).to_interview_config().weights
    assert default_w and abs(sum(default_w.values()) - 1.0) < 1e-6
    assert "stress_regulation" in default_w  # 生理维度的默认席位

    cfg2 = BridgeConfig(questions=("q",), weights={"technical": 0.4})
    assert dict(cfg2.to_interview_config().weights) == {"technical": 0.4}


# ---------- 业务异常 → 结构化下行（绝不炸连接、绝不静默） ----------


def _boom(exc: BaseException):
    def _raise(*_a, **_kw):
        raise exc

    return _raise


def test_unavailable_from_business_becomes_degradation_plus_error(monkeypatch):
    """★ 双帧：能力不可用时要**同时**下行 `degradation.changed` 与 `error`。

    只回 error 而不降级 → 前端不知道界面要降级；只降级不回 error → 调用方以为成功。
    """
    from ruipin.domain.errors import Unavailable

    bridge, _c, _r = _make()
    _run(bridge.handle(make_envelope(ClientType.SESSION_CREATE, {})))
    monkeypatch.setattr(
        bridge.service, "greeting_done", _boom(Unavailable("llm-openai", "网关 502"))
    )

    out = _run(bridge.handle(make_envelope(ClientType.CONSENT_GRANT, {"base": True})))

    assert _types(out) == [str(ServerType.DEGRADATION_CHANGED), str(ServerType.ERROR)]
    assert out[0].payload["level"] >= 1
    assert out[0].payload["badge"]
    assert "llm-openai" in out[0].payload["reason"]
    assert "网关 502" in out[1].payload["message"]


def test_budget_exceeded_becomes_degradation_plus_error(monkeypatch):
    """防回归：预算硬顶同样是"显式下行"，且原因里要带得出具体数额。"""
    from ruipin.domain.errors import BudgetExceeded

    bridge, _c, _r = _make()
    _run(bridge.handle(make_envelope(ClientType.SESSION_CREATE, {})))
    monkeypatch.setattr(
        bridge.service,
        "greeting_done",
        _boom(BudgetExceeded("session_usd", limit=3.0, used=3.5)),
    )

    out = _run(bridge.handle(make_envelope(ClientType.CONSENT_GRANT, {"base": True})))

    assert _types(out) == [str(ServerType.DEGRADATION_CHANGED), str(ServerType.ERROR)]
    assert "3.5" in out[1].payload["message"]
    assert "3.0" in out[1].payload["message"]


def test_other_ruipin_error_becomes_internal_error_only(monkeypatch):
    """防回归：非"能力不可用"的业务异常只回 `internal`，不冒充降级。

    把普通错误也说成降级，会让用户看到"服务已降级"却其实只是参数写错。
    """
    from ruipin.domain.errors import Degraded

    bridge, _c, _r = _make()
    _run(bridge.handle(make_envelope(ClientType.SESSION_CREATE, {})))
    monkeypatch.setattr(
        bridge.service, "greeting_done", _boom(Degraded("rubric 规则命中", level=1))
    )

    out = _run(bridge.handle(make_envelope(ClientType.CONSENT_GRANT, {"base": True})))

    assert _types(out) == [str(ServerType.ERROR)]
    assert out[0].payload["code"] == str(ErrorCode.INTERNAL)
    assert "rubric 规则命中" in out[0].payload["message"]


# ---------- control.skip / control.end 的分支 ----------


def test_control_skip_advances_and_clears_partial_transcript():
    """防回归：跳过时未提交的作答文本必须清空，不能带到下一题去。

    不清空的后果：候选人跳过第 2 题，第 3 题的转写里却带着第 2 题说了一半的话。
    """
    bridge, _c, _r = _make()
    _run(bridge.handle(make_envelope(ClientType.SESSION_CREATE, {})))
    _run(bridge.handle(make_envelope(ClientType.CONSENT_GRANT, {"base": True})))
    _run(bridge.handle(make_envelope(ClientType.ANSWER_TEXT, {"text": "说到一半"})))

    out = _run(bridge.handle(make_envelope(ClientType.CONTROL_SKIP, {})))

    q = _find(out, ServerType.QUESTION_START)
    assert q is not None
    assert q.payload["text"] == QUESTIONS[1]  # 跳到下一题，不是重发当前题
    # 清空证据：下一轮 commit 时转写终稿里不含"说到一半"
    _run(bridge.handle(make_envelope(ClientType.ANSWER_TEXT, {"text": "重新作答"})))
    commit = _run(bridge.handle(make_envelope(ClientType.ANSWER_COMMIT, {})))
    final = _find(commit, ServerType.TRANSCRIPT_FINAL)
    assert final.payload["text"] == "重新作答"


def test_end_while_listening_commits_the_pending_answer():
    """防回归：正在作答时结束，必须**先把这轮跑完**再收尾。

    否则候选人已经说出口、还没点提交的那段话会被直接丢掉。
    """
    bridge, _c, _r = _make()
    _run(bridge.handle(make_envelope(ClientType.SESSION_CREATE, {})))
    _run(bridge.handle(make_envelope(ClientType.CONSENT_GRANT, {"base": True})))
    _run(bridge.handle(make_envelope(ClientType.ANSWER_TEXT, {"text": "最后一段话"})))

    out = _run(bridge.handle(make_envelope(ClientType.CONTROL_END, {})))

    final = _find(out, ServerType.TRANSCRIPT_FINAL)
    assert final is not None and final.payload["text"] == "最后一段话"
    assert _find(out, ServerType.REPORT_READY) is not None
    assert bridge.service.session.state.value == "completed"


def test_end_while_asking_skips_remaining_questions(monkeypatch):
    """防回归：出题途中结束 → 直接转入提问环节，不把剩下的题硬出完。

    这里把 `ask_next` 换成"停在 ASKING"的桩，是为了稳定地命中这条分支
    （正常流程里 ASKING 是瞬态，帧与帧之间停不住）。
    """
    bridge, _c, _r = _make()

    async def go():
        await bridge.handle(make_envelope(ClientType.SESSION_CREATE, {}))
        # 让授权链路走到 ASKING 就停住：桩掉 ask_next，令其不迁移状态
        monkeypatch.setattr(bridge.service, "ask_next", lambda **kw: None)
        await bridge.handle(make_envelope(ClientType.CONSENT_GRANT, {"base": True}))
        assert bridge.service.session.state.value == "asking"
        return await bridge.handle(make_envelope(ClientType.CONTROL_END, {}))

    out = _run(go())

    tos = [f.payload["to"] for f in out if f.type == str(ServerType.STATE_CHANGED)]
    assert "candidate_qa" in tos or "closing" in tos
    assert _find(out, ServerType.QUESTION_START) is None
    assert _find(out, ServerType.REPORT_READY) is not None


def test_end_before_consent_is_a_noop_not_a_fabricated_report():
    """还没授权就结束：**什么都不该发生**。

    这条路径最容易出错——四个状态判断全不命中，如果实现里写成"兜底也给出报告"，
    候选人会在没答一道题的情况下拿到一份报告。空手而归远好过伪造一份结果。
    """
    bridge, _c, _r = _make()
    _run(bridge.handle(make_envelope(ClientType.SESSION_CREATE, {})))
    assert bridge.service.session.state.value == "setup"

    out = _run(bridge.handle(make_envelope(ClientType.CONTROL_END, {})))

    assert out == [], "没有可收尾的东西，就不该产生任何下行帧"
    assert _find(out, ServerType.REPORT_READY) is None
    assert bridge.service.session.state.value == "setup", "状态也不该被推动"


def test_end_while_already_closing_goes_straight_to_the_report():
    """已经在收尾中（CLOSING）再收到 `control.end`：跳过前面的分支，只做最后一步。"""
    bridge, _c, _r = _make()

    async def go():
        await bridge.handle(make_envelope(ClientType.SESSION_CREATE, {}))
        await bridge.handle(make_envelope(ClientType.CONSENT_GRANT, {"base": True}))
        for _ in QUESTIONS:
            await bridge.handle(
                make_envelope(ClientType.ANSWER_COMMIT, {"text": "回答"})
            )
        assert bridge.service.session.state.value == "candidate_qa"
        bridge.service.finish_qa()  # 手动推进到 CLOSING（正常流程里它是瞬态）
        assert bridge.service.session.state.value == "closing"
        return await bridge.handle(make_envelope(ClientType.CONTROL_END, {}))

    out = _run(go())

    assert _find(out, ServerType.REPORT_READY) is not None
    assert bridge.service.session.state.value == "completed"


def test_zero_question_interview_goes_straight_to_qa_then_report():
    """防回归：题目为空时不得卡在 ASKING；应直接进提问环节直到出报告。"""
    bridge, _c, _r = _make(questions=())
    out = _run(_drive_full_interview(bridge, _c))

    assert _find(out, ServerType.REPORT_READY) is not None
    assert bridge.service.session.state.value == "completed"


def test_joined_with_no_pending_tasks_is_a_noop():
    """防回归：没有在途任务时 `joined()` 立刻返回（否则每个空帧都要等一轮）。"""
    bridge, _c, _r = _make()
    loop = asyncio.new_event_loop()
    try:
        handler = SyncHandler(bridge, loop)
        loop.run_until_complete(handler.joined())
        assert handler.drain() == []
    finally:
        loop.close()


def test_skip_without_any_partial_text_still_advances():
    """防回归：没说过话也能跳过（清空逻辑不能因为"没东西可清"而崩）。"""
    bridge, _c, _r = _make()
    _run(bridge.handle(make_envelope(ClientType.SESSION_CREATE, {})))
    _run(bridge.handle(make_envelope(ClientType.CONSENT_GRANT, {"base": True})))

    out = _run(bridge.handle(make_envelope(ClientType.CONTROL_SKIP, {})))

    assert _find(out, ServerType.QUESTION_START).payload["text"] == QUESTIONS[1]


def test_skip_in_candidate_qa_closes_instead_of_asking():
    """防回归：提问环节里按"跳过"应去收尾，不能又冒出一道题出来。"""
    bridge, _c, _r = _make(questions=())
    _run(bridge.handle(make_envelope(ClientType.SESSION_CREATE, {})))
    _run(bridge.handle(make_envelope(ClientType.CONSENT_GRANT, {"base": True})))
    assert bridge.service.session.state.value == "candidate_qa"

    out = _run(bridge.handle(make_envelope(ClientType.CONTROL_SKIP, {})))

    assert _find(out, ServerType.QUESTION_START) is None
    tos = [f.payload["to"] for f in out if f.type == str(ServerType.STATE_CHANGED)]
    assert "closing" in tos


def test_end_when_already_closed_emits_no_duplicate_report():
    """防回归：已结束后再发 `control.end` → 回 `bad_frame`，不再补一份报告。

    重复出报告的后果是客户端把同一场面试渲染两遍。
    """
    bridge, clock, _r = _make()
    _run(_drive_full_interview(bridge, clock))

    out = _run(bridge.handle(make_envelope(ClientType.CONTROL_END, {})))

    assert _types(out) == [str(ServerType.ERROR)]
    assert _find(out, ServerType.REPORT_READY) is None


def test_physio_finalize_without_gate_reason_leaves_reason_empty():
    """防回归：静息信号达标、基线算得出时，不得凭空写一条"不可用原因"。

    这是"没有根因就别编一个"的具体体现：`baseline_gate_reason=None` 时要一路保持空。
    """
    bridge, clock, _r = _make(physio=True)
    _run(bridge.handle(make_envelope(ClientType.SESSION_CREATE, {})))
    _run(bridge.handle(make_envelope(ClientType.CONSENT_GRANT, {"base": True, "physiology": True})))

    good = [{"t_ms": t, "bpm": 70.0, "snr": 0.9} for t in range(0, 60_001, 2_000)]
    _run(bridge.handle(make_envelope(ClientType.PHYSIO_BATCH, {"samples": good})))
    _run(
        bridge.handle(
            make_envelope(
                ClientType.PHYSIO_BATCH,
                {"samples": [], "rest_end": True, "algo_agreement": 1.0},
            )
        )
    )
    assert bridge.collector.baseline_gate_reason is None

    for idx in range(len(QUESTIONS)):
        clock.advance(2.0)
        _run(bridge.handle(make_envelope(ClientType.ANSWER_TEXT, {"text": f"a{idx}"})))
        clock.advance(20.0)
        _run(
            bridge.handle(
                make_envelope(
                    ClientType.PHYSIO_BATCH,
                    {"samples": [{"t_ms": int(clock() * 1000), "bpm": 88.0, "snr": 0.9}]},
                )
            )
        )
        _run(bridge.handle(make_envelope(ClientType.ANSWER_COMMIT, {})))
    end = _run(bridge.handle(make_envelope(ClientType.CONTROL_END, {})))

    report = _find(end, ServerType.REPORT_READY).payload
    assert _find(end, ServerType.REPORT_READY) is not None
    # 关键断言：基线达标时**不能**凭空写一条环境不可用原因；
    # 报告里若仍有 physio.reason，只能是"事件数不足"这类真实统计结论。
    reason = (report["physio"] or {}).get("reason") or ""
    assert "环境不支持" not in reason


def test_frames_from_one_connection_are_processed_strictly_in_serial():
    """★ 防回归（本轮实测到的真 bug）：同一连接的帧必须**严格按序串行**处理。

    第一版 `SyncHandler` 给每帧各开一个 task，于是相邻帧会并发交错：
    `answer.commit` 内部一 `await`（评分），下一帧就开跑了，
    结果 `_pending_text` 被并发清空 → 凭空冒出"无可用作答文本"的假降级，
    让一份本来该出分的报告变成 `available=False`；
    更狠的是 `control.end` 可能在 commit 生效前执行，会话直接卡在 `candidate_qa`。

    这个 bug 在"逐帧 await 调用"的写法下**完全不可见**——所以这里的关键是
    **一口气把 7 帧塞进去，中间一个 await 都不给**，再把结果与串行语义对齐。
    """
    bridge, _c, _r = _make()
    loop = asyncio.new_event_loop()
    try:
        handler = SyncHandler(bridge, loop)

        def fire(idx: int) -> None:
            """连续下发，中间不 await —— 逼出一切并发交错。"""
            handler(make_envelope(ClientType.ANSWER_TEXT, {"text": f"答{idx}"}, seq=idx * 2 + 3))
            handler(make_envelope(ClientType.ANSWER_COMMIT, {}, seq=idx * 2 + 4))

        async def go():
            handler(make_envelope(ClientType.SESSION_CREATE, {}, seq=1))
            handler(make_envelope(ClientType.CONSENT_GRANT, {"base": True}, seq=2))
            for i in range(len(QUESTIONS)):
                fire(i)
            handler(make_envelope(ClientType.CONTROL_END, {}, seq=99))
            await handler.joined()
            return handler.drain()

        frames = loop.run_until_complete(go())
    finally:
        loop.close()

    types = [f.type for f in frames]

    # (1) 三帧答案各自被完整处理 -> 三次评估、三份转写终稿
    assert len([f for f in frames if f.type == str(ServerType.EVAL_DONE)]) == len(QUESTIONS)
    assert len([f for f in frames if f.type == str(ServerType.TRANSCRIPT_FINAL)]) == len(QUESTIONS)

    # (2) 每条答案都带着自己的文本（串行被破坏时这里会出现空文本或串行错位）
    finals = [f.payload["text"] for f in frames if f.type == str(ServerType.TRANSCRIPT_FINAL)]
    assert finals == [f"答{i}" for i in range(len(QUESTIONS))]

    # (3) ★ 不得出现"无可用作答文本"的假降级
    assert not [f for f in frames if f.type == str(ServerType.DEGRADATION_CHANGED)]

    # (4) 报告正常出分（并发交错下这里会是 available=False / score=None）
    report = [f for f in frames if f.type == str(ServerType.REPORT_READY)][0].payload
    assert report["available"] is True
    assert report["n_scored_turns"] == len(QUESTIONS)

    # (5) 会话走到终态，且没有残留的在途工作
    assert bridge.service.session.state.value == "completed"
    assert types[-1] == str(ServerType.REPORT_READY)
    assert handler.pending == 0


def test_pending_reflects_queued_and_running_work():
    """防回归：`pending` 必须同时反映"排队中"和"正在跑"，否则收尾会漏帧。"""
    bridge, _c, _r = _make()
    loop = asyncio.new_event_loop()
    try:
        handler = SyncHandler(bridge, loop)
        assert handler.pending == 0

        handler(make_envelope(ClientType.SESSION_CREATE, {}, seq=1))
        handler(make_envelope(ClientType.CONSENT_GRANT, {"base": True}, seq=2))
        assert handler.pending > 0, "入队后 pending 没动"

        loop.run_until_complete(handler.joined())
        assert handler.pending == 0
        assert handler.drain()  # 收尾后能取到帧
    finally:
        loop.close()


def test_a_raising_bridge_becomes_a_structured_error_frame(monkeypatch):
    """★ 防回归：桥接层真抛异常时，必须转成结构化 `error` 帧，且**不卡死队列**。

    桥接层的 `handle` 自己已经把所有业务异常翻成 error 帧了，所以走到这里
    意味着出现了它没料到的 bug。这种情况下：
    ① 异常不能穿透到服务端读循环（那会让整条连接崩掉）；
    ② 更不能吞掉（用户会以为指令生效了）；
    ③ 而且**后续帧还得继续处理**——一次异常不该把整个会话钉死。
    """
    bridge, _c, _r = _make()
    loop = asyncio.new_event_loop()
    try:
        handler = SyncHandler(bridge, loop)
        inner = bridge.handle
        calls = {"n": 0}

        async def flaky(env):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("注入的桥接层崩溃")
            return await inner(env)

        monkeypatch.setattr(bridge, "handle", flaky)

        handler(make_envelope(ClientType.SESSION_CREATE, {}, seq=1))
        handler(make_envelope(ClientType.SESSION_CREATE, {}, seq=1))
        loop.run_until_complete(handler.joined())
        frames = handler.drain()
    finally:
        loop.close()

    errors = [f for f in frames if f.type == str(ServerType.ERROR)]
    assert len(errors) == 1
    assert errors[0].payload["code"] == str(ErrorCode.INTERNAL)
    assert "注入的桥接层崩溃" in errors[0].payload["message"]
    # ★ 第一帧崩了，第二帧照样被处理
    assert [f for f in frames if f.type == str(ServerType.STATE_CHANGED)]


def test_joined_drains_a_leftover_queue():
    """防回归：万一工作协程没把队列抽干（异常/中断路径），`joined()` 必须兜住。

    漏掉一帧就等于最后一条指令永远不被执行——会话静默停在一个中间状态，
    而且前面一切"正常"。
    """
    bridge, _c, _r = _make()
    loop = asyncio.new_event_loop()
    try:
        handler = SyncHandler(bridge, loop)
        # 直接塞进队列、不唤醒工作协程，模拟"worker 没跑起来"的异常路径
        handler._queue.append(make_envelope(ClientType.SESSION_CREATE, {}, seq=1))
        loop.run_until_complete(handler.joined())
        frames = handler.drain()
    finally:
        loop.close()

    assert [f.type for f in frames] == [str(ServerType.STATE_CHANGED)]
    assert handler.pending == 0


def test_sync_handler_returns_nothing_immediately_and_pushes_later():
    """防回归：`SyncHandler` 的语义是"立即帧为空、结果后续推送"。

    这是刻意的设计（不做假同步等待，否则在事件循环里必然死锁），
    所以必须把这个语义钉在测试里，避免有人"顺手"改成当场返回。
    """
    bridge, _c, _r = _make()
    loop = asyncio.new_event_loop()
    try:
        handler = SyncHandler(bridge, loop)

        immediate = handler(make_envelope(ClientType.SESSION_CREATE, {}))
        assert immediate == []
        assert handler.drain() == []  # 任务只在队列里，还没跑

        loop.run_until_complete(handler.joined())
        pushed = handler.drain()
        assert [f.type for f in pushed] == [str(ServerType.STATE_CHANGED)]
        assert handler.drain() == []  # drain 是取走语义

        assert handler.pending == 0
    finally:
        loop.close()
