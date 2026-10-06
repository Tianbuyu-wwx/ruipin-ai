"""会话级网关：鉴权 / 限流 / seq 去重 / 背压 / 超时 / 降级下行。

时钟一律用 `DeterministicClock`，不 sleep、不依赖真实时间。
每个用例一句中文说明它防的是什么回归。
"""

from __future__ import annotations

import json

import pytest

from ruipin.adapters.fakes import DeterministicClock
from ruipin.domain.errors import RuipinError, Unavailable
from ruipin.transport.gateway import (
    ANON_BUCKET,
    AUTH_REQUIRED_PREFIXES,
    DEFAULT_ACK_EVERY,
    DEFAULT_CREDIT_CAPACITY,
    DEFAULT_IDLE_TIMEOUT_S,
    DEFAULT_RATE_LIMIT_PER_SEC,
    DownlinkSequencer,
    Gateway,
)
from ruipin.transport.protocol import (
    OPCODE_KEY,
    SEQ_RETAIN,
    SEQ_TOLERANCE,
    ClientType,
    Envelope,
    ErrorCode,
    Opcode,
    ServerType,
    encode_binary,
    make_envelope,
)

# ---------------------------------------------------------------- 测试脚手架


class RecordingHandler:
    """记录所有上行信封，并按 `reply` 回帧（默认不回帧）。"""

    def __init__(self, reply=None, exc=None):
        self.seen: list[Envelope] = []
        self.reply = list(reply) if reply else []
        self.exc = exc

    def __call__(self, env: Envelope) -> list[Envelope]:
        self.seen.append(env)
        if self.exc is not None:
            raise self.exc
        return list(self.reply)


def text_frame(
    mtype: str = "session.create",
    *,
    seq: int = 1,
    session_id: str = "s-1",
    payload=None,
    v: int = 1,
    ts: int = 0,
) -> str:
    """构造一条上行文本帧（JSON）。"""
    return json.dumps(
        {
            "v": v,
            "type": mtype,
            "seq": seq,
            "ts": ts,
            "session_id": session_id,
            "payload": {} if payload is None else payload,
        }
    )


def new_gateway(clock: DeterministicClock, handler=None, **kw):
    """构造网关：默认校验器只认令牌 "tok"，绑定到会话 "s-1"。"""
    h = handler if handler is not None else RecordingHandler()
    verifier = kw.pop("token_verifier", lambda t: "s-1" if t == "tok" else None)
    gw = Gateway(h, clock=clock.now, token_verifier=verifier, **kw)
    return gw, h


def code_of(frames) -> str:
    """取单帧 error 的 code，拒绝"多帧/非 error"的含糊断言。"""
    assert len(frames) == 1, f"期望单帧，实际 {frames}"
    assert frames[0].type == "error", f"期望 error 帧，实际 {frames[0].type}"
    return frames[0].payload["code"]


# ---------------------------------------------------------------- 鉴权


def test_authenticate_binds_session_id():
    """合法令牌必须绑定 session_id 并置为已认证，后续帧才放行。"""
    gw, _ = new_gateway(DeterministicClock())
    assert gw.authenticate("tok") is True
    assert gw.authenticated is True
    assert gw.session_id == "s-1"


def test_authenticate_rejects_unknown_token():
    """错误令牌不得放行，且不得留下"半认证"状态。"""
    gw, _ = new_gateway(DeterministicClock())
    assert gw.authenticate("wrong") is False
    assert gw.authenticated is False
    assert gw.session_id is None


def test_authenticate_rejects_empty_session_id_from_verifier():
    """校验器返回空串是"看起来通过其实没通过"，必须判失败。"""
    gw, _ = new_gateway(
        DeterministicClock(), token_verifier=lambda t: "" if t == "tok" else None
    )
    assert gw.authenticate("tok") is False
    assert gw.authenticated is False


def test_default_verifier_denies_all():
    """没配校验器 = 谁也不放行（fail-closed），防止漏配导致裸奔。"""
    gw = Gateway(RecordingHandler())
    assert gw.authenticate("any-token") is False


@pytest.mark.parametrize(
    "mtype",
    ["session.create", "consent.grant", "answer.text", "answer.commit", "media.audio"],
)
def test_unauthenticated_protected_types_return_unauthorized(mtype):
    """session./answer./media./consent. 未认证一律 unauthorized，不得流入业务。"""
    gw, h = new_gateway(DeterministicClock())
    assert code_of(gw.handle_text(text_frame(mtype))) == str(ErrorCode.UNAUTHORIZED)
    assert h.seen == []


def test_auth_required_prefixes_cover_all_protected_groups():
    """鉴权前缀覆盖四个业务组，新增组必须显式登记。"""
    assert AUTH_REQUIRED_PREFIXES == ("session.", "answer.", "media.", "consent.")


def test_control_frames_without_auth_use_anon_bucket():
    """control.* 不受网关鉴权拦截（会话语义归业务层），但计入 anon 限流桶。"""
    gw, h = new_gateway(DeterministicClock(), rate_limit_per_sec=1)
    frames = gw.handle_text(text_frame("control.barge_in", session_id=""))
    assert frames == []
    assert [e.type for e in h.seen] == ["control.barge_in"]
    assert code_of(gw.handle_text(text_frame("control.barge_in", session_id=""))) == str(
        ErrorCode.RATE_LIMITED
    )


def test_authenticated_message_uses_its_own_rate_bucket():
    """anon 桶被打满不得连坐会话桶——限流必须按 session 隔离。"""
    gw, h = new_gateway(DeterministicClock(), rate_limit_per_sec=2)
    for seq in (1, 2):
        gw.handle_text(text_frame("control.barge_in", session_id="", seq=seq))
    assert code_of(
        gw.handle_text(text_frame("control.barge_in", session_id="", seq=3))
    ) == str(ErrorCode.RATE_LIMITED)
    gw.authenticate("tok")
    assert gw.handle_text(text_frame("session.create", seq=4)) == []
    assert [e.type for e in h.seen][-1] == "session.create"


# ---------------------------------------------------------------- 解码与宽容路径


def test_bad_json_returns_bad_frame_error():
    """坏帧必须产出 bad_frame，而不是让 WS 连接抛异常断开。"""
    gw, h = new_gateway(DeterministicClock())
    assert code_of(gw.handle_text("{not json")) == str(ErrorCode.BAD_FRAME)
    assert h.seen == []


def test_unsupported_version_returns_dedicated_code():
    """版本不支持与结构损坏是两回事，code 必须区分（前端要提示升级）。"""
    gw, _ = new_gateway(DeterministicClock())
    assert code_of(gw.handle_text(text_frame(v=2))) == str(
        ErrorCode.UNSUPPORTED_VERSION
    )


def test_unknown_type_returns_unknown_type_error_with_detail():
    """前端比后端新时宽容回 error（unknown_type），并在 detail 里回显类型名。"""
    gw, h = new_gateway(DeterministicClock())
    gw.authenticate("tok")
    frames = gw.handle_text(text_frame("question.hologram"))
    assert code_of(frames) == str(ErrorCode.UNKNOWN_TYPE)
    assert frames[0].payload["detail"] == "question.hologram"
    assert h.seen == []


def test_error_frame_without_detail_has_no_empty_field():
    """未提供的 detail 不应留下空字段，避免前端判空逻辑分叉。"""
    gw, _ = new_gateway(DeterministicClock())
    frames = gw.handle_text("{oops")
    assert "detail" not in frames[0].payload


# ---------------------------------------------------------------- 限流


def test_rate_limit_blocks_the_overflow_message():
    """超出每会话每秒上限的消息必须被拒，且**不得**派发给业务。"""
    gw, h = new_gateway(DeterministicClock(), rate_limit_per_sec=2)
    gw.authenticate("tok")
    for seq in (1, 2):
        assert gw.handle_text(text_frame("answer.text", seq=seq)) == []
    assert code_of(gw.handle_text(text_frame("answer.text", seq=3))) == str(
        ErrorCode.RATE_LIMITED
    )
    assert len(h.seen) == 2


def test_rate_limit_window_rolls_over_after_one_second():
    """滑动窗口过 1 秒必须恢复配额，否则长会话会永久被限死。"""
    clock = DeterministicClock()
    gw, _ = new_gateway(clock, rate_limit_per_sec=1)
    gw.authenticate("tok")
    assert gw.handle_text(text_frame("answer.text", seq=1)) == []
    assert code_of(gw.handle_text(text_frame("answer.text", seq=2))) == str(
        ErrorCode.RATE_LIMITED
    )
    clock.advance(1.0)
    assert gw.handle_text(text_frame("answer.text", seq=3)) == []


def test_rate_limit_boundary_is_exactly_the_limit():
    """恰好等于上限的 N 条必须全部放行（边界包含），只拦第 N+1 条。"""
    gw, h = new_gateway(DeterministicClock(), rate_limit_per_sec=3)
    gw.authenticate("tok")
    for seq in (1, 2, 3):
        assert gw.handle_text(text_frame("answer.text", seq=seq)) == []
    assert len(h.seen) == 3


def test_defaults_are_sane():
    """默认参数必须可用（限流/超时/窗口都是正数），防止配置缺省即失控。"""
    assert DEFAULT_RATE_LIMIT_PER_SEC >= 1
    assert DEFAULT_IDLE_TIMEOUT_S > 0
    assert DEFAULT_CREDIT_CAPACITY >= 1
    assert DEFAULT_ACK_EVERY >= 1


@pytest.mark.parametrize(
    "bad", [{"rate_limit_per_sec": 0}, {"idle_timeout_s": 0}, {"ack_every": 0}]
)
def test_gateway_rejects_bad_config(bad):
    """非法配置必须在构造期炸掉，不能等到线上才表现出怪异行为。"""
    with pytest.raises(ValueError):
        new_gateway(DeterministicClock(), **bad)


# ---------------------------------------------------------------- seq 语义


def test_messages_are_processed_in_arrival_order():
    """同一会话连续消息必须按到达顺序交付业务层（不重排、不并发穿插）。"""
    gw, h = new_gateway(DeterministicClock())
    gw.authenticate("tok")
    for seq in (1, 2, 3):
        gw.handle_text(text_frame("answer.text", seq=seq))
    assert [e.seq for e in h.seen] == [1, 2, 3]


def test_first_frame_any_seq_becomes_baseline():
    """首帧 seq 可以是任意值（客户端可能断线续传），不得误判跳号。"""
    gw, h = new_gateway(DeterministicClock())
    gw.authenticate("tok")
    assert gw.handle_text(text_frame("answer.text", seq=42)) == []
    assert [e.seq for e in h.seen] == [42]


def test_seq_within_tolerance_is_accepted():
    """窗口内的小幅跳号属于正常乱序，必须放行并推进基线。"""
    gw, h = new_gateway(DeterministicClock())
    gw.authenticate("tok")
    gw.handle_text(text_frame("answer.text", seq=1))
    assert gw.handle_text(
        text_frame("answer.text", seq=1 + SEQ_TOLERANCE)
    ) == []
    assert [e.seq for e in h.seen] == [1, 1 + SEQ_TOLERANCE]


def test_duplicate_seq_replays_previous_response_without_reprocessing():
    """重传必须幂等：回放上次的响应，**不**重复调用业务（否则会重复计分）。

    回放的帧必须与首次下发的帧**逐字段相同（含 seq）**：前端正是按 seq 去重，
    换了号它就认不出这是同一帧，同一道题会在页面上出现两次。
    """
    reply = [make_envelope(ServerType.EVAL_DONE, {"score": 88.0})]
    gw, h = new_gateway(DeterministicClock(), RecordingHandler(reply=reply))
    gw.authenticate("tok")
    first = gw.handle_text(text_frame("answer.commit", seq=1))
    second = gw.handle_text(text_frame("answer.commit", seq=1))
    assert [e.type for e in first] == [str(ServerType.EVAL_DONE)]
    assert first[0].seq > 0, "业务帧必须带下行 seq（否则前端按 seq>0 去重形同虚设）"
    assert second == first, "回放必须连 seq 一起回放，不能重新编号"
    assert len(h.seen) == 1


def test_duplicate_seq_after_cache_eviction_returns_duplicate_error():
    """响应缓存有界（SEQ_RETAIN）；淘汰后的重传无法回放，回 duplicate 而不是重跑。"""
    gw, h = new_gateway(DeterministicClock())
    gw.authenticate("tok")
    total = SEQ_RETAIN + 2
    for seq in range(1, total + 1):
        gw.handle_text(text_frame("answer.text", seq=seq))
    assert code_of(gw.handle_text(text_frame("answer.text", seq=2))) == str(
        ErrorCode.DUPLICATE
    )
    assert len(h.seen) == total


def test_seq_far_behind_returns_out_of_order():
    """落后超过容忍窗口说明中间帧全丢了，必须让客户端重来而不是继续处理。"""
    gw, _ = new_gateway(DeterministicClock())
    gw.authenticate("tok")
    for seq in range(1, SEQ_TOLERANCE + 3):
        gw.handle_text(text_frame("answer.text", seq=seq))
    assert code_of(gw.handle_text(text_frame("answer.text", seq=1))) == str(
        ErrorCode.OUT_OF_ORDER
    )


def test_seq_gap_too_large_returns_out_of_order():
    """跳号超过容忍窗口（疑似丢包）必须报错，不能把基线推到天上。"""
    gw, h = new_gateway(DeterministicClock())
    gw.authenticate("tok")
    gw.handle_text(text_frame("answer.text", seq=1))
    assert code_of(
        gw.handle_text(text_frame("answer.text", seq=1 + SEQ_TOLERANCE + 1))
    ) == str(ErrorCode.OUT_OF_ORDER)
    assert len(h.seen) == 1


# ---------------------------------------------------------------- 派发


def test_payload_and_session_id_reach_handler_intact():
    """网关不得改写业务负载与会话标识（传输层只做搬运）。"""
    gw, h = new_gateway(DeterministicClock())
    gw.authenticate("tok")
    gw.handle_text(
        text_frame("answer.text", seq=1, payload={"text": "我主导了召回重构"})
    )
    assert h.seen[0].payload == {"text": "我主导了召回重构"}
    assert h.seen[0].session_id == "s-1"
    assert h.seen[0].type == "answer.text"


def test_handler_ruipin_error_becomes_internal_error_frame():
    """业务抛 RuipinError 不得炸连接，转成 internal 错误帧且带上异常信息。"""
    gw, _ = new_gateway(
        DeterministicClock(), RecordingHandler(exc=Unavailable("llm", "timeout"))
    )
    gw.authenticate("tok")
    frames = gw.handle_text(text_frame("answer.commit", seq=1))
    assert code_of(frames) == str(ErrorCode.INTERNAL)
    assert "Unavailable" in frames[0].payload["message"]
    assert isinstance(Unavailable("a", "b"), RuipinError)


def test_handler_unexpected_exception_becomes_internal_error_frame():
    """非预期异常同样转成 internal 帧（不静默吞，消息里带异常类型）。"""
    gw, _ = new_gateway(
        DeterministicClock(), RecordingHandler(exc=ValueError("boom"))
    )
    gw.authenticate("tok")
    frames = gw.handle_text(text_frame("answer.commit", seq=1))
    assert code_of(frames) == str(ErrorCode.INTERNAL)
    assert "ValueError" in frames[0].payload["message"]


def test_outbound_seq_increments_monotonically():
    """网关产出的下行帧 seq 必须严格递增，前端靠它判断丢帧/回放。"""
    gw, _ = new_gateway(DeterministicClock())
    seqs = []
    for _ in range(3):
        seqs.extend(f.seq for f in gw.handle_text("{oops"))
    assert seqs == [1, 2, 3]


def test_outbound_ts_comes_from_injected_clock():
    """时间戳必须来自注入时钟（毫秒），不得偷偷读真实时间。"""
    clock = DeterministicClock(1000.0)
    gw, _ = new_gateway(clock)
    frames = gw.handle_text("{oops")
    assert frames[0].ts == 1000000
    clock.advance(2.5)
    assert gw.handle_text("{oops")[0].ts == 1002500


# ---------------------------------------------------------------- 媒体 / 背压


def test_media_frame_is_consumed_and_acked_every_n_frames():
    """每消费 ack_every 帧必须回一个 media.ack，客户端靠它继续发。"""
    gw, h = new_gateway(
        DeterministicClock(), credit_capacity=4, ack_every=2
    )
    gw.authenticate("tok")
    first = gw.handle_binary(encode_binary(Opcode.MEDIA_AUDIO, b"a" * 8))
    assert first == []
    second = gw.handle_binary(encode_binary(Opcode.MEDIA_AUDIO, b"b" * 8))
    assert len(second) == 1
    assert second[0].type == "media.ack"
    assert second[0].payload == {"frames": 2, "consumed": 2, "credits": 4}
    assert gw.credit.used == 0
    assert len(h.seen) == 2


def test_media_frame_carries_opcode_size_and_bytes():
    """媒体帧翻译成的信封必须带 opcode/size/data，业务层才能分辨媒体类型。"""
    gw, h = new_gateway(DeterministicClock())
    gw.authenticate("tok")
    gw.handle_binary(encode_binary(Opcode.MEDIA_VIDEO, b"\x01\x02\x03"))
    assert h.seen[0].type == "media.video"
    assert h.seen[0].payload["opcode"] == 0x02
    assert h.seen[0].payload["size"] == 3
    assert h.seen[0].payload["data"] == b"\x01\x02\x03"


def test_media_upload_requires_auth():
    """未认证的媒体帧一律拒绝，防止匿名上传把服务端带宽打满。"""
    gw, h = new_gateway(DeterministicClock())
    assert code_of(gw.handle_binary(encode_binary(Opcode.MEDIA_AUDIO, b"x"))) == str(
        ErrorCode.UNAUTHORIZED
    )
    assert h.seen == []


@pytest.mark.parametrize(
    "raw", [b"\x7f\x00\x00\x00\x00", b"\x01\x00\x00", b"not-bytes"]
)
def test_media_bad_frame_returns_bad_frame_error(raw):
    """未知 opcode 与截断帧都回 bad_frame（并指明原因），不得当成空媒体处理。"""
    gw, h = new_gateway(DeterministicClock())
    gw.authenticate("tok")
    assert code_of(gw.handle_binary(raw)) == str(ErrorCode.BAD_FRAME)
    assert h.seen == []


def test_media_credit_window_exhausted_then_recovered():
    """窗口满时必须拒绝上行；consume 释放后必须恢复（背压的核心回归点）。"""
    gw, h = new_gateway(
        DeterministicClock(),
        credit_capacity=2,
        ack_every=2,
        auto_consume=False,
    )
    gw.authenticate("tok")
    assert gw.handle_binary(encode_binary(Opcode.MEDIA_AUDIO, b"1")) == []
    assert gw.handle_binary(encode_binary(Opcode.MEDIA_AUDIO, b"2")) == []
    assert gw.credit.used == 2
    assert code_of(gw.handle_binary(encode_binary(Opcode.MEDIA_AUDIO, b"3"))) == str(
        ErrorCode.CREDIT_EXHAUSTED
    )
    assert len(h.seen) == 2  # 被拒的帧不能进业务

    acks = gw.consume(2)
    assert len(acks) == 1 and acks[0].payload["consumed"] == 2
    assert gw.credit.available == 2
    assert gw.handle_binary(encode_binary(Opcode.MEDIA_AUDIO, b"4")) == []
    assert len(h.seen) == 3


def test_consume_emits_one_ack_per_ack_every_frames():
    """每次满 ack_every 帧回一个 ack；余数必须留到下次，不能丢。"""
    gw, _ = new_gateway(DeterministicClock(), credit_capacity=16, ack_every=2,
                        auto_consume=False)
    gw.authenticate("tok")
    for _ in range(6):
        gw.handle_binary(encode_binary(Opcode.MEDIA_AUDIO, b"x"))
    assert len(gw.consume(5)) == 2  # 5 帧 → 2 个 ack，余 1 帧留到下次
    assert len(gw.consume(1)) == 1  # 余数补齐 → 再 1 个 ack
    assert gw.credit.used == 0


def test_media_rate_limited():
    """媒体帧同样受限流保护（否则二进制帧可以绕过文本限流）。"""
    gw, h = new_gateway(DeterministicClock(), rate_limit_per_sec=2)
    gw.authenticate("tok")
    for _ in range(2):
        gw.handle_binary(encode_binary(Opcode.MEDIA_AUDIO, b"a"))
    assert code_of(gw.handle_binary(encode_binary(Opcode.MEDIA_AUDIO, b"a"))) == str(
        ErrorCode.RATE_LIMITED
    )
    assert len(h.seen) == 2


# ---------------------------------------------------------------- 超时


def test_idle_timeout_tick_reports_session_timeout():
    """空闲超过阈值必须产出 session.timeout（显式告知，不静默断连）。"""
    clock = DeterministicClock()
    gw, _ = new_gateway(clock, idle_timeout_s=10)
    gw.authenticate("tok")
    clock.advance(10.0)
    frames = gw.tick()
    assert len(frames) == 1
    assert frames[0].type == "session.timeout"
    assert frames[0].payload["code"] == str(ErrorCode.SESSION_TIMEOUT)
    assert frames[0].payload["session_id"] == "s-1"
    assert frames[0].payload["idle_s"] == pytest.approx(10.0)


def test_tick_before_threshold_reports_nothing():
    """未到阈值不得误杀活跃会话。"""
    clock = DeterministicClock()
    gw, _ = new_gateway(clock, idle_timeout_s=10)
    gw.authenticate("tok")
    clock.advance(9.999)
    assert gw.tick() == []


def test_activity_resets_idle_timer():
    """任何上行活动都要刷新空闲计时，否则长会话会被莫名判死。"""
    clock = DeterministicClock()
    gw, _ = new_gateway(clock, idle_timeout_s=10)
    gw.authenticate("tok")
    clock.advance(8.0)
    gw.handle_text(text_frame("answer.text", seq=1))
    clock.advance(8.0)
    assert gw.tick() == []


def test_expired_session_is_reported_only_once():
    """同一会话只报一次超时，避免 tick 循环里刷屏。"""
    clock = DeterministicClock()
    gw, _ = new_gateway(clock, idle_timeout_s=10)
    gw.authenticate("tok")
    clock.advance(10.0)
    assert len(gw.tick()) == 1
    assert gw.tick() == []
    assert gw.expired_sessions == frozenset({"s-1"})


def test_expired_session_rejects_later_text_frames():
    """已超时会话后续文本帧必须回 session_timeout，不得继续派发业务。"""
    clock = DeterministicClock()
    gw, h = new_gateway(clock, idle_timeout_s=10)
    gw.authenticate("tok")
    clock.advance(10.0)
    gw.tick()
    assert code_of(gw.handle_text(text_frame("answer.text", seq=2))) == str(
        ErrorCode.SESSION_TIMEOUT
    )
    assert h.seen == []


def test_expired_session_rejects_later_media_frames():
    """已超时会话后续媒体帧同样拒绝，避免往死会话里灌数据。"""
    clock = DeterministicClock()
    gw, h = new_gateway(clock, idle_timeout_s=10)
    gw.authenticate("tok")
    clock.advance(10.0)
    gw.tick()
    assert code_of(gw.handle_binary(encode_binary(Opcode.MEDIA_AUDIO, b"x"))) == str(
        ErrorCode.SESSION_TIMEOUT
    )
    assert h.seen == []


def test_tick_accepts_explicit_now():
    """tick 支持外部传入 now，便于与真实事件循环时钟对齐。"""
    gw, _ = new_gateway(DeterministicClock(), idle_timeout_s=10)
    gw.authenticate("tok")
    assert gw.tick(now=5.0) == []
    assert len(gw.tick(now=10.0)) == 1


def test_anon_activity_is_also_timed_out():
    """未认证的 anon 桶同样受空闲巡检，防止匿名连接长期占位。"""
    clock = DeterministicClock()
    gw, _ = new_gateway(clock, idle_timeout_s=10)
    gw.handle_text(text_frame("control.barge_in", session_id=""))
    clock.advance(10.0)
    frames = gw.tick()
    assert [f.payload["session_id"] for f in frames] == [ANON_BUCKET]


# ---------------------------------------------------------------- 降级下行


def test_send_degradation_uses_badge_and_reason():
    """降级必须显式下行，payload 带 level/reason/中文徽标（禁止静默降级）。"""
    gw, _ = new_gateway(DeterministicClock())
    gw.authenticate("tok")
    frames = gw.send_degradation(4, "tts_unavailable")
    assert len(frames) == 1
    assert frames[0].type == "degradation.changed"
    assert frames[0].payload == {
        "level": 4,
        "reason": "tts_unavailable",
        "badge": "无声，文本照常",
        "message": "服务已降级至 L4：无声，文本照常（原因：tts_unavailable）",
    }
    assert frames[0].session_id == "s-1"


def test_send_degradation_level_zero_means_recovered():
    """Level 0 是正常态：徽标为空串，文案说明已恢复。"""
    gw, _ = new_gateway(DeterministicClock())
    frames = gw.send_degradation(0, "recovered")
    assert frames[0].payload["badge"] == ""
    assert frames[0].payload["message"] == "服务已恢复正常，全功能运行"


@pytest.mark.parametrize("level", [-1, 7])
def test_send_degradation_rejects_out_of_range_level(level):
    """越界等级必须抛错（不静默降级为 L0），让配置错误立刻暴露。"""
    gw, _ = new_gateway(DeterministicClock())
    with pytest.raises(ValueError):
        gw.send_degradation(level, "bad")


def test_send_degradation_shares_outbound_seq_with_errors():
    """降级帧与错误帧共用一条下行 seq 序列，前端可据此判断顺序。"""
    gw, _ = new_gateway(DeterministicClock())
    gw.handle_text("{oops")
    frames = gw.send_degradation(1, "vision_timeout")
    assert frames[0].seq == 2


# ---------------------------------------------------------------- 下行编号器


def test_downlink_sequencer_skips_media_frames_without_consuming_numbers():
    """媒体帧编成裸字节帧、没有 seq 字段，**不该**占用编号（否则文本 seq 出空洞）。"""
    tracker = DownlinkSequencer()
    media = make_envelope(
        ServerType.QUESTION_TTS_CHUNK,
        {OPCODE_KEY: int(Opcode.TTS_AUDIO), "data": b"abc"},
    )

    assert tracker.stamp(make_envelope(ServerType.STATE_CHANGED, {})).seq == 1
    assert tracker.stamp(media) is media, "媒体帧必须原样透传"
    assert tracker.stamp(make_envelope(ServerType.EVAL_DONE, {})).seq == 2
    assert tracker.last == 2


def test_downlink_sequencer_keeps_existing_seq_and_raises_the_water_level():
    """已编号的帧保留原号，但水位抬到它之上——否则后面的帧会撞号。"""
    tracker = DownlinkSequencer()
    numbered = make_envelope(ServerType.ERROR, {}, seq=7)

    assert tracker.stamp(numbered).seq == 7
    assert tracker.last == 7
    assert tracker.stamp(make_envelope(ServerType.ERROR, {})).seq == 8


def test_downlink_sequencer_rejects_invalid_start():
    """构造期就拒绝非法起点（`True` 也是 int，必须显式排除）。"""
    for bad in (-1, True, 1.5, "3"):
        with pytest.raises(ValueError):
            DownlinkSequencer(bad)


# ---------------------------------------------------------------- 端到端小链路


def test_full_happy_path_single_connection():
    """一问一答的最小闭环：建会话 → 回答 → 提交 → 降级下行，全部走同一网关。"""
    clock = DeterministicClock(1700000000.0)
    handler = RecordingHandler(
        reply=[make_envelope(ServerType.STATE_CHANGED, {"to": "PROCESSING"})]
    )
    gw, h = new_gateway(clock, handler, rate_limit_per_sec=10)
    assert gw.authenticate("tok") is True

    assert gw.handle_text(text_frame("session.create", seq=1)) != []
    assert gw.handle_text(
        text_frame("answer.text", seq=2, payload={"text": "我主导了召回重构"})
    ) != []
    assert gw.handle_text(text_frame("answer.commit", seq=3)) != []
    assert [e.type for e in h.seen] == [
        "session.create",
        "answer.text",
        "answer.commit",
    ]
    assert gw.send_degradation(2, "scoring_failed")[0].payload["badge"] == (
        "本轮为规则评分，仅供参考"
    )
    assert gw.tick() == []  # 活跃会话不该被判超时


def test_client_type_enum_is_what_gateway_expects():
    """网关按字符串前缀鉴权，枚举字面值必须与前缀规则一致。"""
    assert str(ClientType.SESSION_CREATE).startswith("session.")
    assert str(ClientType.MEDIA_SCREEN).startswith("media.")
