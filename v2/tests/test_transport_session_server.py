"""会话服务端循环（`transport/session_server.py`）的测试。

为什么单列一个文件
------------------
前面所有传输层测试都止步于 `gateway.handle_text(raw) -> list[Envelope]`。
但真实系统里**没有人负责"读连接、喂网关、把出来的帧写回连接"**——
缺了这一段，前端能启动却永远连不上后端。这个文件补的就是这一环。

用**假连接**而不是真 WebSocket：本层的价值恰恰在于它可以被纯逻辑测试
（零网络、零端口、零握手），真实 WS 库只是 ~20 行的适配器。

重点钉住四件事：
1. **未认证的连接根本不触达业务**（结构保证，不是"业务层会检查"的纪律）；
2. **推送帧不依赖新输入**——评分是异步的，它的结果可能在客户端沉默时就绪，
   只有"收到帧才去取 outbox"的实现会让它一直压在队列里发不出去；
3. **写失败不能丢掉业务状态**——对端断了是常态，状态推进该完成还得完成；
4. **下行分流正确**：带 opcode 的走二进制帧，其余走 JSON 文本帧。
"""

from __future__ import annotations

import asyncio
import json

import pytest

from ruipin.adapters.fakes import FakeEvaluator
from ruipin.adapters.repo_sqlite import SqliteRepo
from ruipin.orchestrator import TurnScheduler
from ruipin.transport import (
    BridgeConfig,
    ClientType,
    ConnectionClosed,
    ErrorCode,
    InterviewBridge,
    ServerType,
    SessionServer,
    decode_binary,
    encode_binary,
    encode_text,
    make_envelope,
)

QUESTIONS = ("q1", "q2", "q3")


class _Clock:
    def __init__(self, start: float = 3_000.0) -> None:
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


class FakeWS:
    """脚本化的假连接。

    Args:
        script: 依次返回的帧（`str` 文本 / `bytes` 二进制 / 任意对象用于测坏帧）。
        idle_s: 脚本耗尽后，先真实等这么久再抛 `ConnectionClosed`。
            这个"等"是刻意的：用来验证**客户端沉默期间服务端仍会推送**。
        fail_send_after: 成功发送这么多次之后开始抛异常（模拟对端已断）。
    """

    def __init__(self, script=(), *, idle_s: float = 0.0, fail_send_after: int | None = None) -> None:
        self._script = list(script)
        self._idle_s = idle_s
        self._fail_send_after = fail_send_after
        self.sent: list = []
        self.send_attempts = 0

    async def recv(self):
        if self._script:
            return self._script.pop(0)
        if self._idle_s > 0:
            await asyncio.sleep(self._idle_s)
        raise ConnectionClosed("脚本耗尽")

    async def send(self, data):
        self.send_attempts += 1
        if self._fail_send_after is not None and self.send_attempts > self._fail_send_after:
            raise ConnectionResetError("对端已断")
        self.sent.append(data)

    # ---------- 断言助手 ----------

    @property
    def texts(self) -> list[str]:
        return [d for d in self.sent if isinstance(d, str)]

    @property
    def binaries(self) -> list[bytes]:
        return [d for d in self.sent if isinstance(d, bytes)]

    def types(self) -> list[str]:
        out = []
        for d in self.texts:
            out.append(json.loads(d)["type"])
        return out

    def find(self, mtype: str):
        for d in self.texts:
            env = json.loads(d)
            if env["type"] == str(mtype):
                return env
        return None

    def all_of(self, mtype: str) -> list[dict]:
        return [json.loads(d) for d in self.texts if json.loads(d)["type"] == str(mtype)]


def _text(mtype, payload=None, *, seq: int = 0, session_id: str = "s-ws") -> str:
    return encode_text(make_envelope(mtype, payload or {}, seq=seq, session_id=session_id))


def _make_server(
    *,
    evaluator=None,
    built: list | None = None,
    bridges: list | None = None,
    **kw,
) -> SessionServer:
    repo = SqliteRepo(":memory:")
    clock = _Clock()

    def build(sid: str) -> InterviewBridge:
        if built is not None:
            built.append(sid)
        bridge = InterviewBridge(
            repo=repo,
            scheduler=TurnScheduler(evaluator or FakeEvaluator(), clock=clock),
            config=BridgeConfig(questions=QUESTIONS),
            session_id=sid,
            clock=clock,
        )
        if bridges is not None:
            bridges.append(bridge)
        return bridge

    return SessionServer(
        build_bridge=build,
        token_verifier=lambda tok: "s-ws" if tok == "good" else None,
        clock=clock,
        **kw,
    )


def _run(coro):
    return asyncio.run(coro)


FULL_SCRIPT = [
    _text(ClientType.SESSION_CREATE, {}, seq=1),
    _text(ClientType.CONSENT_GRANT, {"base": True}, seq=2),
    _text(ClientType.ANSWER_TEXT, {"text": "答一"}, seq=3),
    _text(ClientType.ANSWER_COMMIT, {}, seq=4),
    _text(ClientType.ANSWER_TEXT, {"text": "答二"}, seq=5),
    _text(ClientType.ANSWER_COMMIT, {}, seq=6),
    _text(ClientType.ANSWER_TEXT, {"text": "答三"}, seq=7),
    _text(ClientType.ANSWER_COMMIT, {}, seq=8),
    _text(ClientType.CONTROL_END, {}, seq=9),
]


# ---------- 鉴权：结构上不触达业务 ----------


def test_unauthenticated_connection_is_refused_without_touching_business():
    """★ 令牌无效 → 回一条 `unauthorized`，并且**根本不构造业务对象**。

    断言的是"业务层没被构造/调用"，而不只是"回了个错误"：
    未认证的连接连一场面试都不该被创建出来。
    """
    built: list[str] = []
    server = _make_server(built=built)
    conn = FakeWS()

    stats = _run(server.serve(conn, token="bad"))

    assert stats.authenticated is False
    assert stats.frames_in == 0
    assert stats.closed_reason == "认证失败"
    assert built == [], "认证失败却构造了业务对象"

    codes = [json.loads(d)["payload"]["code"] for d in conn.texts]
    assert codes == [str(ErrorCode.UNAUTHORIZED)]
    assert conn.types() == [str(ServerType.ERROR)]


def test_token_verifier_returning_empty_string_is_treated_as_failure():
    """防回归：校验器返回空串必须算失败（"看起来通过其实没通过"的经典坑）。"""
    server = SessionServer(
        build_bridge=lambda sid: pytest.fail(f"不该为 {sid!r} 构造业务"),
        token_verifier=lambda tok: "",
        clock=_Clock(),
    )
    stats = _run(server.serve(FakeWS(), token="whatever"))

    assert stats.authenticated is False
    assert stats.session_id == ""


# ---------- 主链路：完整一场面试走线序字节 ----------


def test_full_interview_over_a_fake_connection():
    """★ 端到端：脚本化的 9 条上行帧走完一场面试，下行按帧类型正确分流。"""
    built: list[str] = []
    server = _make_server(built=built)
    conn = FakeWS(FULL_SCRIPT)

    stats = _run(server.serve(conn, token="good"))

    assert stats.authenticated is True
    assert stats.session_id == "s-ws"
    assert built == ["s-ws"]
    assert stats.frames_in == len(FULL_SCRIPT)
    assert stats.closed_reason == "脚本耗尽"

    types = conn.types()
    assert len(conn.all_of(ServerType.QUESTION_START)) == len(QUESTIONS)
    assert len(conn.all_of(ServerType.EVAL_DONE)) == len(QUESTIONS)
    assert len(conn.all_of(ServerType.REPORT_READY)) == 1
    assert types[-1] == str(ServerType.REPORT_READY)

    report = conn.find(ServerType.REPORT_READY)["payload"]
    assert report["available"] is True
    assert report["score"] is not None

    # 状态序列可回放，且末态是 completed
    tos = [e["payload"]["to"] for e in conn.all_of(ServerType.STATE_CHANGED)]
    assert tos[-1] == "completed"
    assert "candidate_qa" in tos

    # 本场没有媒体，不该有任何二进制帧
    assert conn.binaries == []
    assert stats.binary_out == 0
    assert stats.frames_out == stats.text_out + stats.binary_out
    assert stats.bytes_out == sum(len(d) for d in conn.sent)


def test_downlink_text_frames_all_carry_monotonic_seq():
    """★ 下行的**每一条**文本帧都必须带 seq，且严格递增、无空洞。

    这是前端能否按 seq 去重/判丢帧的前提。修复前只有网关自产的帧（`error` /
    `media.ack` / `session.timeout`）有 seq，业务帧（`question.start` /
    `eval.done` / `report.ready`）恒为 0 —— 于是业务帧的去重在真实链路里形同虚设，
    一次重传回放就会让同一道题在页面上出现两次。
    """
    server = _make_server()
    conn = FakeWS(FULL_SCRIPT)

    stats = _run(server.serve(conn, token="good"))

    seqs = [json.loads(d)["seq"] for d in conn.texts]
    assert seqs, "整场面试一条下行文本帧都没有，那这个用例没测到真东西"
    assert all(s > 0 for s in seqs), f"存在未编号的业务帧: {seqs}"
    assert all(b > a for a, b in zip(seqs, seqs[1:])), f"下行 seq 不严格递增: {seqs}"
    assert seqs == list(range(1, len(seqs) + 1)), (
        f"下行 seq 出现空洞（有人编号了却没发出去，或编号点不唯一）: {seqs}"
    )
    assert len(seqs) == stats.text_out, "连接层统计与真实发出的文本帧数对不上"


def test_cross_cutting_limits_are_forwarded_to_the_gateway():
    """★ 会话服务端必须把限流/超时/信用等横切参数转给网关。

    回归：`SessionServer` 建 `Gateway` 时原先一个都不传，于是 `Gateway` 上那四个
    可调项（`rate_limit_per_sec` / `idle_timeout_s` / `credit_capacity` / `ack_every`）
    在真实服务里**恒等于模块默认值**——参数看着存在，部署侧却没有任何入口能改。

    用"第 3 帧被限流"做**行为**验证，而不是去读私有属性：后者只能证明值被存下了，
    证明不了它真的生效。
    """
    server = _make_server(rate_limit_per_sec=2)
    conn = FakeWS(
        [
            _text(ClientType.SESSION_CREATE, {}, seq=1),
            _text(ClientType.CONSENT_GRANT, {"base": True}, seq=2),
            _text(ClientType.ANSWER_TEXT, {"text": "答"}, seq=3),
        ]
    )

    _run(server.serve(conn, token="good"))

    errors = [
        json.loads(d)["payload"]
        for d in conn.texts
        if json.loads(d)["type"] == str(ServerType.ERROR)
    ]
    assert [e["code"] for e in errors] == [str(ErrorCode.RATE_LIMITED)], (
        f"限流参数没有传到网关（每会话 2 条/秒，第 3 帧必须被限流）：{errors}"
    )


def test_bad_frame_does_not_kill_the_connection():
    """防回归：坏帧只回一条 `error`，连接继续活着（WS 断了比报错贵得多）。"""
    server = _make_server()
    script = [
        "这不是 JSON",
        _text(ClientType.SESSION_CREATE, {}, seq=1),
    ]
    conn = FakeWS(script)

    stats = _run(server.serve(conn, token="good"))

    errors = [
        json.loads(d)["payload"]
        for d in conn.texts
        if json.loads(d)["type"] == str(ServerType.ERROR)
    ]
    assert [e["code"] for e in errors] == [str(ErrorCode.BAD_FRAME)]
    # 坏帧之后的那条正常帧仍然被处理了
    assert conn.find(ServerType.STATE_CHANGED) is not None
    assert stats.closed_reason == "脚本耗尽"


def test_unknown_frame_object_is_reported_not_swallowed():
    """防回归：连接层拿到既不是 str 也不是 bytes 的东西，要显式报错而不是静默丢。"""
    server = _make_server()
    conn = FakeWS([12345, None])

    stats = _run(server.serve(conn, token="good"))

    codes = [json.loads(d)["payload"]["code"] for d in conn.texts]
    assert codes == [str(ErrorCode.BAD_FRAME), str(ErrorCode.BAD_FRAME)]
    assert len(stats.errors) == 2
    assert "int" in stats.errors[0] and "NoneType" in stats.errors[1]


# ---------- 关键设计点：推送帧不依赖新输入 ----------


def test_late_async_push_is_delivered_while_the_client_stays_silent():
    """★ 客户端沉默时，异步就绪的推送帧仍必须被发出去。

    做法：评估器故意慢 30ms，客户端在 `answer.commit` 之后**一句话也不说**。
    提交那一轮的响应在 `SyncHandler` 里是空的（它立刻返回、稍后推送），
    所以只有"轮询 outbox"的实现才能把 30ms 后才就绪的 `transcript.final`
    / `eval.done` / `question.start` 送出去。
    如果实现是"收到帧才 drain"，这个用例会**一条推送都收不到**。
    """
    server = _make_server(
        evaluator=FakeEvaluator(slow_s=0.03),
        outbox_poll_s=0.005,
    )
    conn = FakeWS(
        [
            _text(ClientType.SESSION_CREATE, {}, seq=1),
            _text(ClientType.CONSENT_GRANT, {"base": True}, seq=2),
            _text(ClientType.ANSWER_TEXT, {"text": "答"}, seq=3),
            _text(ClientType.ANSWER_COMMIT, {}, seq=4),
        ],
        idle_s=0.2,  # 沉默 200ms（评估只要 30ms，余量 6 倍）
    )

    _run(server.serve(conn, token="good"))

    types = conn.types()
    assert str(ServerType.TRANSCRIPT_FINAL) in types, "异步就绪的转写终稿没被推送"
    assert str(ServerType.EVAL_DONE) in types, "异步就绪的评估结果没被推送"
    assert len(conn.all_of(ServerType.QUESTION_START)) == 2, "下一题没被推送"


def test_outbox_poll_interval_must_be_positive():
    """防回归：轮询间隔为 0 会变成空转死循环，必须构造期就拒。"""
    with pytest.raises(ValueError):
        _make_server(outbox_poll_s=0)


def test_max_frames_limit_stops_the_loop_with_an_explicit_reason():
    """防回归：帧数上限要能停止循环并写清原因（防跑飞的测试/攻击）。"""
    server = _make_server(max_frames=2)
    conn = FakeWS(FULL_SCRIPT)

    stats = _run(server.serve(conn, token="good"))

    assert stats.frames_in == 2
    assert "上限" in stats.closed_reason
    assert stats.closed is True

    with pytest.raises(ValueError):
        _make_server(max_frames=0)


# ---------- 二进制帧走媒体面 ----------


def test_binary_frames_take_the_media_path_and_get_acked():
    """★ 二进制上行走媒体面（含信用窗口），攒够 `ack_every` 后回 `media.ack`。

    分流写错的后果很隐蔽：把音频当文本解会把整条连接搞崩，
    把文本当音频解则静默无输出。
    """
    server = _make_server()
    media = encode_binary(1, b"\x00" * 8)
    conn = FakeWS([media] * 4)

    stats = _run(server.serve(conn, token="good"))

    assert stats.frames_in == 4
    acks = conn.all_of(ServerType.MEDIA_ACK)
    assert len(acks) == 1, "每 4 帧应回一个 media.ack"
    assert acks[0]["payload"]["frames"] == 4
    assert acks[0]["payload"]["consumed"] == 4
    # ack 是文本帧；上行媒体帧本身不该被回显成二进制
    assert conn.binaries == []
    assert stats.binary_out == 0


def test_unauthenticated_connection_never_reads_any_frame():
    """★ 防回归：认证用的是连接建立时的一次性令牌，**先鉴权再读帧**。

    所以令牌无效时连一条上行帧都不会被读走——"未认证不触达业务"由此成为
    结构保证（帧根本没进过网关），而不是"读了再判、判完丢弃"的纪律。
    """
    built: list[str] = []
    server = _make_server(built=built)
    conn = FakeWS([encode_binary(1, b"\x00" * 8)] * 4)

    stats = _run(server.serve(conn, token="bad"))

    assert stats.authenticated is False
    assert stats.frames_in == 0, "未认证却读了上行帧"
    assert built == []
    # 没有 ack、没有任何媒体相关下行
    assert conn.all_of(ServerType.MEDIA_ACK) == []
    assert [json.loads(d)["payload"]["code"] for d in conn.texts] == [
        str(ErrorCode.UNAUTHORIZED)
    ]


# ---------- 写失败不能丢业务状态 ----------


def test_send_failure_is_recorded_and_business_state_still_advances():
    """★ 对端断开（写失败）时：记下原因、不抛异常、**业务状态该推进还得推进**。

    如果写失败直接让循环崩掉，候选人这一轮的作答就白说了——
    而"对端断了"其实是最常见的失败之一，不该是致命的。
    状态是权威，字节只是它的投影：投影丢了可以重连重放，状态丢了就真丢了。
    """
    bridges: list[InterviewBridge] = []
    server = _make_server(bridges=bridges)
    conn = FakeWS(FULL_SCRIPT, fail_send_after=4)

    stats = _run(server.serve(conn, token="good"))

    assert stats.errors, "写失败必须留痕"
    assert any("发送失败" in e for e in stats.errors)
    assert "发送失败" in stats.closed_reason
    # 关键：连接坏了，但面试仍然被完整推进到终态
    assert len(bridges) == 1
    bridge = bridges[0]
    assert bridge.service.session.state.value == "completed"
    # 而且落库的事件一条不少（可回放 -> 重连能重建 UI）
    events = bridge.repo.load_events("s-ws")
    assert len(events) == len(bridge.service.session.events)


# ---------- 接收侧异常 ----------


def test_deferred_handler_dispatches_nothing_until_bound():
    """★ 防回归：未绑定 handler 时**什么都不派发**。

    "未认证不触达业务"之所以是结构保证而不是纪律，就靠这个小对象：
    没绑定 = 没有可派发的目标，不需要业务层配合检查权限。
    """
    from ruipin.transport.session_server import _DeferredHandler

    deferred = _DeferredHandler()
    assert deferred.bound is False
    assert deferred(make_envelope(ClientType.SESSION_CREATE, {})) == []

    seen: list[str] = []
    deferred.bind(lambda env: seen.append(env.type) or [])
    assert deferred.bound is True
    assert deferred(make_envelope(ClientType.SESSION_CREATE, {})) == []
    assert seen == [str(ClientType.SESSION_CREATE)]


def test_outbound_media_frames_are_sent_as_binary_frames():
    """★ 防回归：下行带 `opcode` 的信封必须走**二进制帧**，其余走 JSON 文本帧。

    TTS 音频块与 viseme 是二进制下行。分流写错的后果很隐蔽：
    前端收到一条 JSON 却以为里面是音频，播放器静默无输出——不报错、没声音。
    """
    from ruipin.transport import Opcode, decode_binary, media_envelope

    server = _make_server()
    real_build = server._build

    def build(sid):
        bridge = real_build(sid)
        inner = bridge.handle

        async def handle(env):
            out = await inner(env)
            if env.type == str(ClientType.SESSION_CREATE):
                # 真实系统里这里是 TTS 首块音频
                out = list(out) + [
                    media_envelope(
                        ServerType.QUESTION_TTS_CHUNK,
                        Opcode.TTS_AUDIO,
                        b"\x11\x22",
                        ts=1,
                    )
                ]
            return out

        bridge.handle = handle
        return bridge

    server._build = build
    conn = FakeWS([_text(ClientType.SESSION_CREATE, {}, seq=1)])

    stats = _run(server.serve(conn, token="good"))

    assert len(conn.binaries) == 1, "带 opcode 的信封没有走二进制帧"
    opcode, payload = decode_binary(conn.binaries[0])
    assert int(opcode) == int(Opcode.TTS_AUDIO)
    assert payload == b"\x11\x22"
    assert stats.binary_out == 1
    assert stats.text_out >= 1
    assert stats.frames_out == stats.text_out + stats.binary_out


def test_tick_idle_can_be_switched_off():
    """防回归：`tick_idle=False` 时不做空闲巡检（多连接共用一个巡检器时的常见配置）。

    巡检被关掉时连接照样要能正常收尾——不能因为"少了一次 tick"就漏掉收尾 flush。
    """
    server = _make_server(tick_idle=False)
    conn = FakeWS([_text(ClientType.SESSION_CREATE, {}, seq=1)])

    stats = _run(server.serve(conn, token="good"))

    assert conn.find(ServerType.STATE_CHANGED) is not None
    assert stats.closed_reason == "脚本耗尽"


def test_outbox_poll_interval_is_honoured():
    """防回归：轮询间隔参数真的生效（否则推送会卡到下一秒才出去）。"""
    server = _make_server(outbox_poll_s=0.001)
    assert server._poll_s == 0.001


def test_cancelled_read_is_treated_as_a_normal_close():
    """★ 防回归：**recv 任务被取消**（WS 库关连接时的常态）按正常关闭收尾。

    这跟"本协程被取消"是两回事，必须分开：
    * recv 被库取消 → 连接正在关闭，正常收尾、不记 error、不向调用方抛；
    * 本协程被取消 → CancelledError 从 `await asyncio.wait` 抛出，继续往外抛
      （吞掉它等于拒绝被取消，asyncio 会认为任务没停）。
    早先的实现把两者混在一起并统一 `raise`，于是"对端关连接"会变成一个异常。
    """

    class CancellingWS(FakeWS):
        async def recv(self):
            raise asyncio.CancelledError()

    server = _make_server()
    stats = _run(server.serve(CancellingWS(), token="good"))

    assert stats.closed_reason == "对端关闭: 读取被取消"
    assert stats.errors == [], "正常关闭不该记成故障"
    assert stats.closed is True


def test_serve_can_itself_be_cancelled():
    """防回归：`serve` 所在的任务被取消时，取消必须**继续传播**出去。

    如果这里被吞掉，asyncio 会认为任务没有停止，取消会静默失效——
    真实服务停机时最难排查的一类故障。
    """

    class HangingWS(FakeWS):
        async def recv(self):
            await asyncio.sleep(3600)
            raise ConnectionClosed("永不返回")

    async def main():
        server = _make_server()
        task = asyncio.ensure_future(server.serve(HangingWS(), token="good"))
        await asyncio.sleep(0.02)  # 让循环真正跑起来并停在 recv 上
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    _run(main())


def test_recv_error_closes_the_loop_with_a_readable_reason():
    """防回归：`recv` 抛非关闭异常时也要收尾，并把异常类型写进关闭原因。"""

    class ExplodingWS(FakeWS):
        async def recv(self):
            raise RuntimeError("底层 socket 炸了")

    server = _make_server()
    stats = _run(server.serve(ExplodingWS(), token="good"))

    assert stats.closed_reason == "接收失败: RuntimeError"
    assert "底层 socket 炸了" in stats.errors[0]


def test_connection_closed_is_not_treated_as_a_failure():
    """防回归：正常关闭不是错误——不能记进 errors，否则告警会被噪声淹没。"""
    server = _make_server()
    stats = _run(server.serve(FakeWS(), token="good"))

    assert stats.closed_reason == "脚本耗尽"
    assert stats.errors == []
    assert stats.frames_in == 0
