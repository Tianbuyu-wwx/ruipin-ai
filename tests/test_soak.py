"""P5 硬化：压测 / 长时间运行（Soak）。

为什么单列一个文件
------------------
前面所有测试都是"一场面试"的视角。并发下才会暴露的问题有另一套：
会话之间串数据、共享限流桶互相饿死、信用窗口漏计、埋点数在并发里丢失。
这些问题的共同特点是**单场跑一万次都看不出来**。

本文件用**真实链路**（`Gateway` + `SyncHandler` + `InterviewBridge`
+ `InterviewService` + `SqliteRepo`）跑并发，钉住四件事：

1. **会话隔离**：任一连接只能看到自己的帧、自己的事件日志、自己的报告。
2. **不会互相拖死**：一个会话评估挂了，其他会话的成绩**一分不变**。
3. **背压不漏计**：媒体洪泛下信用窗口既不越界也不静默丢帧。
4. **埋点不丢不重**：并发下 Meter 的总调用数恰好等于各会话之和。

口径声明（别误读）：
* 这里压的是"百级并发以内"的方案假设（方案 §11.1-3）；**不是**容量测试，
  没有测吞吐上限、没有测真实数据库的锁竞争、没有多进程/多 worker。
* 因此本文件的结论只说明"并发语义正确"，不说明"能扛多少并发"。
"""

from __future__ import annotations

import asyncio

import pytest

from ruipin.adapters.fakes import FakeEvaluator
from ruipin.adapters.repo_sqlite import SqliteRepo
from ruipin.domain.errors import Unavailable
from ruipin.observability.metrics import InMemoryMeter
from ruipin.orchestrator import TurnScheduler
from ruipin.ports import EvalResult, Usage
from ruipin.transport import (
    BridgeConfig,
    ClientType,
    ErrorCode,
    Gateway,
    InterviewBridge,
    ServerType,
    SyncHandler,
    encode_binary,
    encode_text,
    make_envelope,
)

QUESTIONS = ("q1", "q2", "q3")


class _Clock:
    """会话内共享的确定性时钟（压测里所有会话共用同一个时间轴）。"""

    def __init__(self, start: float = 2_000.0) -> None:
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


class _MeteredEvaluator:
    """像真实适配器那样**自己上报 Usage**（成本/延迟必须由适配器如实上报）。"""

    def __init__(self, meter, *, cost_usd: float = 0.002, fail: bool = False) -> None:
        self.name = "metered"
        self._inner = FakeEvaluator(fail=fail)
        self._meter = meter
        self._cost = cost_usd

    async def evaluate(self, req) -> EvalResult:
        result = await self._inner.evaluate(req)
        self._meter.record_usage(
            Usage(
                provider="metered",
                operation="score",
                latency_ms=140,
                prompt_tokens=900,
                completion_tokens=130,
                cost_usd=self._cost,
            )
        )
        return result


class _Session:
    """一条"连接"：网关 + 同步适配 + 桥接 + 服务，全部真实。"""

    def __init__(self, repo: SqliteRepo, clock: _Clock, sid: str, evaluator) -> None:
        self.sid = sid
        self.loop = asyncio.get_event_loop()
        self.bridge = InterviewBridge(
            repo=repo,
            scheduler=TurnScheduler(evaluator, clock=clock),
            config=BridgeConfig(questions=QUESTIONS),
            session_id=sid,
            clock=clock,
        )
        self.handler = SyncHandler(self.bridge, self.loop)
        self.gateway = Gateway(
            handler=self.handler,
            clock=clock,
            token_verifier=lambda tok: sid if tok == f"tok-{sid}" else None,
        )
        assert self.gateway.authenticate(f"tok-{sid}") is True
        self.seq = 0
        self.received: list = []

    async def send(self, mtype, payload, *, advance: float | None = None, clock=None):
        self.seq += 1
        if advance and clock is not None:
            clock.advance(advance)
        raw = encode_text(make_envelope(mtype, payload, seq=self.seq, session_id=self.sid))
        immediate = self.gateway.handle_text(raw)
        await self.handler.joined()
        self.received.extend(self.handler.drain())
        return immediate

    async def interview(self, clock: _Clock) -> None:
        await self.send(ClientType.SESSION_CREATE, {"candidate_id": self.sid})
        await self.send(ClientType.CONSENT_GRANT, {"base": True})
        for idx in range(len(QUESTIONS)):
            await self.send(ClientType.ANSWER_TEXT, {"text": f"{self.sid}-答{idx}"}, advance=2.0, clock=clock)
            await self.send(ClientType.ANSWER_COMMIT, {}, advance=20.0, clock=clock)
        await self.send(ClientType.CONTROL_END, {}, advance=1.0, clock=clock)


def _report_of(session: _Session) -> dict:
    for f in session.received:
        if f.type == str(ServerType.REPORT_READY):
            return f.payload
    raise AssertionError(f"会话 {session.sid} 没有收到 report.ready")


# ---------- 1. 并发会话的隔离与完整交付 ----------

N_SESSIONS = 30


def test_soak_concurrent_sessions_stay_isolated_and_all_reach_a_report():
    """★ 30 场并发：每场恰好一份报告、成绩一致、事件日志互不串场。"""
    repo = SqliteRepo(":memory:")
    clock = _Clock()

    async def go():
        sessions = [
            _Session(repo, clock, f"s{i}", FakeEvaluator()) for i in range(N_SESSIONS)
        ]
        await asyncio.gather(*(s.interview(clock) for s in sessions))
        return sessions

    loop = asyncio.new_event_loop()
    try:
        sessions = loop.run_until_complete(go())
    finally:
        loop.close()

    scores = []
    for s in sessions:
        reports = [f for f in s.received if f.type == str(ServerType.REPORT_READY)]
        assert len(reports) == 1, f"{s.sid} 收到了 {len(reports)} 份报告"
        report = reports[0].payload
        assert report["available"] is True
        assert report["score"] is not None
        scores.append(report["score"])

        # 会话内只出现自己的 session_id（串场最隐蔽的形态）
        assert {f.session_id for f in s.received} == {s.sid}
        assert s.bridge.service.session.state.value == "completed"
        # 题号恰好推进 3 次（没有因为并发重放而多出题目）
        assert s.bridge._turn_index == len(QUESTIONS)

        # 落库可回放：事件日志与内存逐条一致
        events = repo.load_events(s.sid)
        assert [e["to"] for e in events] == [
            st.to_state.value for st in s.bridge.service.session.events
        ]
        assert len(repo.load_turns(s.sid)) == len(QUESTIONS)

    # 同样的输入、同样的评估器 ⇒ 成绩必须完全一致（无共享状态污染）
    assert len(set(scores)) == 1, f"并发下成绩出现了漂移: {sorted(set(scores))}"
    assert scores[0] == pytest.approx(65.0, abs=0.01)

    # 事件计数守恒：各会话之和 == 库中总数（没有一条事件被写丢或写重）
    total = sum(len(repo.load_events(s.sid)) for s in sessions)
    assert total == sum(len(s.bridge.service.session.events) for s in sessions)


def test_soak_one_broken_session_does_not_poison_the_others():
    """★ 红线：一场评估挂了，**不得**影响其他场的成绩（多租户隔离的底线）。

    这条在单会话测试里永远测不出来；只有把健康与故障会话放在同一批并发里才暴露。
    """
    repo = SqliteRepo(":memory:")
    clock = _Clock()
    healthy_n = 12

    async def go():
        healthy = [
            _Session(repo, clock, f"ok{i}", FakeEvaluator()) for i in range(healthy_n)
        ]
        broken = _Session(repo, clock, "broken", FakeEvaluator(fail=True))
        await asyncio.gather(*(s.interview(clock) for s in healthy), broken.interview(clock))
        return healthy, broken

    loop = asyncio.new_event_loop()
    try:
        healthy, broken = loop.run_until_complete(go())
    finally:
        loop.close()

    ok_scores = [_report_of(s)["score"] for s in healthy]
    assert len(set(ok_scores)) == 1
    assert ok_scores[0] == pytest.approx(65.0, abs=0.01)

    bad = _report_of(broken)
    assert bad["available"] is False
    assert bad["score"] is None
    assert broken.bridge.service.session.state.value == "completed"

    # 故障会话的降级不能出现在健康会话的帧里
    for s in healthy:
        assert [f for f in s.received if f.type == str(ServerType.DEGRADATION_CHANGED)] == []
        assert [f for f in s.received if f.type == str(ServerType.EVAL_DONE)]


# ---------- 2. 信用窗口背压（媒体洪泛） ----------


def test_soak_credit_window_never_overflows_and_never_drops_silently():
    """★ 10 帧洪泛、窗口容量 4：越界的帧必须**明确报错**，不许静默丢弃。

    静默丢弃是最坏的失败：客户端以为发出去了，服务端从未见过。
    """
    repo = SqliteRepo(":memory:")
    clock = _Clock()
    bridge = InterviewBridge(
        repo=repo,
        scheduler=TurnScheduler(FakeEvaluator(), clock=clock),
        config=BridgeConfig(questions=QUESTIONS),
        session_id="s-media",
        clock=clock,
    )
    loop = asyncio.new_event_loop()
    try:
        handler = SyncHandler(bridge, loop)
        gw = Gateway(
            handler=handler,
            clock=clock,
            token_verifier=lambda t: "s-media",
            credit_capacity=4,
            ack_every=2,
            auto_consume=False,  # 由我们显式消费，才能把窗口压满
        )
        assert gw.authenticate("tok") is True

        media = encode_binary(1, b"\x00" * 16)
        accepted = 0
        rejected = 0
        for _ in range(10):
            out = gw.handle_binary(media)
            if out and out[0].payload.get("code") == str(ErrorCode.CREDIT_EXHAUSTED):
                rejected += 1
            else:
                assert out == [], "未满时不该产生额外帧（也不该回成功码骗客户端）"
                accepted += 1
            assert 0 <= gw.credit.available <= 4, "信用窗口越界"

        assert accepted == 4, "容量 4 应恰好接受 4 帧"
        assert rejected == 10 - 4, "其余帧必须被判为窗口已满"
        assert gw.credit.used == 4

        # 消费 2 帧 → 归还 2 个信用 → 又能接受 2 帧
        acks = gw.consume(2)
        assert len(acks) == 1  # ack_every=2 → 每 2 帧一个 ack
        assert acks[0].payload["frames"] == 2
        assert gw.credit.available == 2

        for _ in range(2):
            out = gw.handle_binary(media)
            assert out == []
        assert gw.credit.used == 4

        # 再消费 4 帧：总计消费 6 → 3 个 ack
        assert len(gw.consume(4)) == 2
        assert gw.credit.used == 0
        assert gw.credit.available == 4

        loop.run_until_complete(handler.joined())
    finally:
        loop.close()


def test_soak_credit_window_rejects_release_beyond_in_flight():
    """防回归：多归还信用必须报错，不许静默把窗口撑大（会绕过背压）。"""
    from ruipin.transport import CreditWindow

    win = CreditWindow(2)
    assert win.try_acquire() is True
    with pytest.raises(ValueError):
        win.release(5)


# ---------- 3. 埋点：并发下不丢不重 ----------


def test_soak_meter_totals_equal_the_sum_over_all_sessions():
    """★ 并发埋点守恒：总调用数 = 会话数 × 每题一次，成本逐次累加不重不漏。

    埋点丢数会让成本模型与告警全部失真——而且**丢了看不出来**，
    因为总量只是"少了一点"。所以这里比对的是精确值。
    """
    repo = SqliteRepo(":memory:")
    clock = _Clock()
    meter = InMemoryMeter()
    n = 8

    async def go():
        sessions = [
            _Session(repo, clock, f"m{i}", _MeteredEvaluator(meter)) for i in range(n)
        ]
        await asyncio.gather(*(s.interview(clock) for s in sessions))
        return sessions

    loop = asyncio.new_event_loop()
    try:
        sessions = loop.run_until_complete(go())
    finally:
        loop.close()

    totals = meter.usage_totals()
    expected_calls = n * len(QUESTIONS)
    assert totals["calls"] == expected_calls
    assert totals["cost_usd"] == pytest.approx(expected_calls * 0.002)
    assert totals["prompt_tokens"] == expected_calls * 900
    assert totals["completion_tokens"] == expected_calls * 130

    # 每个会话都真的拿到分（不是"埋点有数但没出报告"）
    assert all(_report_of(s)["available"] is True for s in sessions)

    # 按 provider 聚合的桶与总量一致（分桶写错会让归因失真）
    buckets = meter.usage_buckets()
    assert sum(b["calls"] for b in buckets.values()) == expected_calls


def test_soak_shared_gateway_rate_limit_is_per_session_not_global():
    """★ 防回归：限流必须是**每会话**一个桶，不能全局共享。

    全局桶会让一个刷屏的候选把整批候选人都限死——多租户系统的经典事故。
    """
    repo = SqliteRepo(":memory:")
    clock = _Clock()
    bridge_a = InterviewBridge(
        repo=repo,
        scheduler=TurnScheduler(FakeEvaluator(), clock=clock),
        config=BridgeConfig(questions=QUESTIONS),
        session_id="A",
        clock=clock,
    )
    bridge_b = InterviewBridge(
        repo=repo,
        scheduler=TurnScheduler(FakeEvaluator(), clock=clock),
        config=BridgeConfig(questions=QUESTIONS),
        session_id="B",
        clock=clock,
    )
    loop = asyncio.new_event_loop()
    try:
        gw_a = Gateway(
            handler=SyncHandler(bridge_a, loop),
            clock=clock,
            token_verifier=lambda t: "A",
            rate_limit_per_sec=1,
        )
        gw_b = Gateway(
            handler=SyncHandler(bridge_b, loop),
            clock=clock,
            token_verifier=lambda t: "B",
            rate_limit_per_sec=1,
        )
        assert gw_a.authenticate("a") and gw_b.authenticate("b")

        first_a = gw_a.handle_text(encode_text(make_envelope(ClientType.SESSION_CREATE, {}, session_id="A")))
        first_b = gw_b.handle_text(encode_text(make_envelope(ClientType.SESSION_CREATE, {}, session_id="B")))
        assert [f.payload.get("code") for f in first_a] != [str(ErrorCode.RATE_LIMITED)]
        assert [f.payload.get("code") for f in first_b] != [str(ErrorCode.RATE_LIMITED)]

        # A 的第二个请求被自己的窗口拦住
        second_a = gw_a.handle_text(encode_text(make_envelope(ClientType.SESSION_CREATE, {}, session_id="A")))
        assert [f.payload["code"] for f in second_a] == [str(ErrorCode.RATE_LIMITED)]

        loop.run_until_complete(gw_a._handler.joined())
        loop.run_until_complete(gw_b._handler.joined())
    finally:
        loop.close()
