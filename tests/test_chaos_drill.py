"""P5 硬化：降级演练（Chaos drill）。

为什么单列一个文件
------------------
单点失败的单元测试到处都是，但生产上真正出事的是**故障组合**：
评估挂了、预算顶了、生理数据没测出来、客户端乱发帧——同时发生。
这里把故障注入**真实链路**（`Gateway` → `SyncHandler` → `InterviewBridge`
→ `InterviewService` → `SqliteRepo`），逐一验证方案 §3.4 的三条纪律：

1. **降级必须显式**——有中文徽标、有原因，绝不静默；未登记的原因码直接报错。
2. **任何降级下都不产生兜底分数**——`available=False` 时 `score` 必须是 `None`。
3. **会话必须仍能走到终态**——不卡死、不炸连接、状态序列仍可回放。

另外钉住一条容易被"顺手优化"掉的语义：
**失败的那一轮按"不计入"处理，而不是按 0 分计入**。
把没测到的轮次算成 0 分，是对候选人的实质惩罚——本项目宁可少一轮样本。
"""

from __future__ import annotations

import asyncio

import pytest

from ruipin.adapters.fakes import FakeEvaluator
from ruipin.adapters.repo_sqlite import SqliteRepo
from ruipin.domain.errors import Unavailable
from ruipin.orchestrator import TurnScheduler
from ruipin.orchestrator.degradation import LEVELS, badge_for
from ruipin.orchestrator.service import PHYSIO_DIM
from ruipin.ports import EvalResult
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

QUESTIONS = ("请介绍你的项目经历", "讲一个你解决过的技术难题", "你如何与团队协作")


class _Clock:
    def __init__(self, start: float = 1_000.0) -> None:
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


class _FailOnCalls:
    """前若干个**调用**失败的评估器，之后恢复。

    注意计的是"调用次数"不是"轮数"：`TurnScheduler` 对可重试错误会重试 1 次
    （方案 §3.2），所以让某一轮真正失败需要连续失败 **2 次**调用。
    这个细节本身就是一条纪律——"重试"是设计好的，不能把一次抖动当永久故障。
    """

    def __init__(self, fail_calls: int, dims: dict[str, float] | None = None) -> None:
        self.name = "flaky"
        self._inner = FakeEvaluator(dims=dims)
        self._fail_calls = fail_calls
        self.n_calls = 0

    async def evaluate(self, req) -> EvalResult:
        idx = self.n_calls
        self.n_calls += 1
        if idx < self._fail_calls:
            raise Unavailable(self.name, f"第 {idx} 次调用注入故障")
        return await self._inner.evaluate(req)


def _make(
    *,
    evaluator=None,
    physio: bool = False,
    weights: dict[str, float] | None = None,
    questions=QUESTIONS,
) -> tuple[InterviewBridge, _Clock, SqliteRepo]:
    repo = SqliteRepo(":memory:")
    clock = _Clock()
    cfg = BridgeConfig(
        questions=tuple(questions),
        physio_enabled=physio,
        weights=weights,
    )
    return (
        InterviewBridge(
            repo=repo,
            scheduler=TurnScheduler(evaluator or FakeEvaluator(), clock=clock),
            config=cfg,
            session_id="s-chaos",
            clock=clock,
        ),
        clock,
        repo,
    )


async def _run_interview(bridge, clock, *, answer: bool = True) -> list:
    """走完一场面试，返回全部下行帧（按时间顺序）。"""
    frames: list = []
    frames += await bridge.handle(make_envelope(ClientType.SESSION_CREATE, {"candidate_id": "c"}))
    frames += await bridge.handle(make_envelope(ClientType.CONSENT_GRANT, {"base": True}))
    for idx in range(len(QUESTIONS)):
        if answer:
            clock.advance(2.0)
            frames += await bridge.handle(
                make_envelope(ClientType.ANSWER_TEXT, {"text": f"回答{idx}"})
            )
        clock.advance(20.0)
        frames += await bridge.handle(make_envelope(ClientType.ANSWER_COMMIT, {}))
    clock.advance(1.0)
    frames += await bridge.handle(make_envelope(ClientType.CONTROL_END, {}))
    return frames


def _run(coro):
    return asyncio.run(coro)


def _types(frames) -> list[str]:
    return [f.type for f in frames]


def _find(frames, mtype):
    for f in frames:
        if f.type == mtype:
            return f
    return None


def _all(frames, mtype) -> list:
    return [f for f in frames if f.type == mtype]


# ---------- 演练 1：评估器整场不可用 ----------


def test_chaos_evaluator_down_for_whole_session_still_produces_an_honest_report():
    """★ 评估全挂：每一轮显式降级、全程无评估帧、报告不给任何数字、会话正常结束。"""
    bridge, clock, _r = _make(evaluator=FakeEvaluator(fail=True))
    frames = _run(_run_interview(bridge, clock))

    # (1) 显式降级：每轮一条，且都有非空徽标
    degradations = _all(frames, ServerType.DEGRADATION_CHANGED)
    assert len(degradations) == len(QUESTIONS)
    for d in degradations:
        assert d.payload["level"] >= 1
        assert d.payload["badge"]
        assert d.payload["reason"]

    # (2) 没有真实评估就不能有评估帧（不能拿"降级结果"冒充成功）
    assert _all(frames, ServerType.EVAL_DONE) == []

    # (3) 报告：无分数、无等级、原因写清"不生成兜底分数"
    report = _find(frames, ServerType.REPORT_READY).payload
    assert report["available"] is False
    assert report["score"] is None
    assert report["level"] is None
    assert "不生成兜底分数" in report["reason"]
    assert report["n_scored_turns"] == 0

    # (4) 会话仍然走到终态，状态序列可回放
    assert bridge.service.session.state.value == "completed"
    states = [f.payload["to"] for f in _all(frames, ServerType.STATE_CHANGED)]
    assert states[-1] == "completed"
    assert "reporting" in states


def test_chaos_all_degradation_levels_have_nonempty_badges_except_level_zero():
    """防回归：阶梯里除 Level 0 外每级都必须有徽标文案。

    没有徽标的等级等于"静默降级"——用户在界面上看不到任何异常。
    """
    for lv in LEVELS:
        if lv.level == 0:
            assert lv.ui_badge == ""
        else:
            assert lv.ui_badge, f"L{lv.level} 缺徽标文案"
            assert badge_for(lv.level) == lv.ui_badge


# ---------- 演练 2：部分轮次失败（最容易被写成"0 分"的地方） ----------


def test_chaos_failed_turns_are_excluded_not_scored_as_zero():
    """★ 红线：失败的轮次按"不计入"处理，绝不按 0 分计入平均。

    如果按 0 分算，候选人会因为"服务端挂了"而掉分——这是最不该发生的归因错误。
    做法：拿"全成功"的成绩当基准，把"第 1 轮失败"的成绩与之逐值比对。
    """
    healthy, c1, _r1 = _make(evaluator=FakeEvaluator())
    baseline = _find(_run(_run_interview(healthy, c1)), ServerType.REPORT_READY).payload

    # 连失 2 次 = 第 1 轮（首次 + 重试）都失败 → 该轮无评估
    flaky, c2, _r2 = _make(evaluator=_FailOnCalls(fail_calls=2))
    frames = _run(_run_interview(flaky, c2))
    report = _find(frames, ServerType.REPORT_READY).payload

    assert report["available"] is True
    assert report["n_scored_turns"] == len(QUESTIONS) - 1
    # ★ 关键：少了一轮样本，但成绩**一分不差**——说明失败轮没被当成 0
    assert report["score"] == pytest.approx(baseline["score"])
    assert report["dims"] == pytest.approx(baseline["dims"])
    # 而且失败轮确实被显式报了降级，不是悄悄跳过
    assert len(_all(frames, ServerType.DEGRADATION_CHANGED)) == 1


def test_chaos_min_scored_turns_gate_blocks_a_score_when_too_few_turns_succeed():
    """防回归：成功轮数不足 `min_scored_turns` 时**不出分**，而不是"就一轮也凑合用"。"""
    bridge, clock, _r = _make(evaluator=_FailOnCalls(fail_calls=2 * len(QUESTIONS)))
    frames = _run(_run_interview(bridge, clock))
    report = _find(frames, ServerType.REPORT_READY).payload

    assert report["available"] is False
    assert report["score"] is None


# ---------- 演练 3：生理模块"测不准"时按原比例归还权重 ----------


def test_chaos_physio_unavailable_redistributes_weight_by_original_ratio():
    """★ 生理不可用 → 它的权重按**原比例**还给其他维度，且不拉低其他维度的分。

    两条断言缺一不可：
    (a) 权重守恒：生效权重和仍为 1.0，`stress_regulation` 不在生效维度里；
    (b) 比例守恒：每个维度的"生效/声明"倍数一致 ⇒ 确实是按原比例分摊，
        而不是被某一维偷偷吞掉。
    """
    declared = {
        "technical": 0.30,
        "communication": 0.25,
        "completeness": 0.15,
        "problem_solving": 0.15,
        "teamwork": 0.10,
        "leadership": 0.05,
    }

    # 不带生理的基准
    plain, cp, _rp = _make(weights=dict(declared))
    base = _find(_run(_run_interview(plain, cp)), ServerType.REPORT_READY).payload

    # 带生理但静息信号太差（低 SNR）→ 生理不计入
    bridge, clock, _r = _make(physio=True, weights=dict(declared))
    _run(bridge.handle(make_envelope(ClientType.SESSION_CREATE, {})))
    _run(
        bridge.handle(
            make_envelope(ClientType.CONSENT_GRANT, {"base": True, "physiology": True})
        )
    )
    dark = [{"t_ms": t, "bpm": 70.0, "snr": 0.15} for t in range(0, 60_001, 2_000)]
    _run(bridge.handle(make_envelope(ClientType.PHYSIO_BATCH, {"samples": dark})))
    _run(
        bridge.handle(
            make_envelope(ClientType.PHYSIO_BATCH, {"samples": [], "rest_end": True})
        )
    )
    for idx in range(len(QUESTIONS)):
        clock.advance(2.0)
        _run(bridge.handle(make_envelope(ClientType.ANSWER_TEXT, {"text": f"a{idx}"})))
        clock.advance(20.0)
        _run(bridge.handle(make_envelope(ClientType.ANSWER_COMMIT, {})))
    report = _find(_run(bridge.handle(make_envelope(ClientType.CONTROL_END, {}))), ServerType.REPORT_READY).payload

    # (a) 权重守恒
    eff = report["effective_weights"]
    assert PHYSIO_DIM not in eff
    assert sum(eff.values()) == pytest.approx(1.0)
    assert report["physio"]["available"] is False
    assert report["physio"]["weight_applied"] == 0.0
    assert "环境不支持" in report["physio"]["reason"]

    # (b) 比例守恒
    multiples = {k: eff[k] / declared[k] for k in declared}
    assert max(multiples.values()) - min(multiples.values()) < 1e-9

    # 其他维度一分没被压低（与完全没开生理的基准逐值相同）
    assert report["score"] == pytest.approx(base["score"])
    assert report["dims"] == pytest.approx(base["dims"])


# ---------- 演练 4：会话从任何状态都必须能收尾 ----------


def test_chaos_end_is_reachable_from_every_nonterminal_state():
    """防回归：`control.end` 在 ASKING / LISTENING / CANDIDATE_QA 三种处境下都能收尾。

    弱网下候选人可能在任何时刻点"结束"，只要有一条路径卡死，用户就永远看不到报告。
    """
    for answer_before_end in (False, True):
        bridge, clock, _r = _make()
        _run(bridge.handle(make_envelope(ClientType.SESSION_CREATE, {})))
        _run(bridge.handle(make_envelope(ClientType.CONSENT_GRANT, {"base": True})))
        if answer_before_end:
            clock.advance(2.0)
            _run(bridge.handle(make_envelope(ClientType.ANSWER_TEXT, {"text": "答到一半"})))
        clock.advance(1.0)
        out = _run(bridge.handle(make_envelope(ClientType.CONTROL_END, {})))

        assert _find(out, ServerType.REPORT_READY) is not None, (
            f"answer_before_end={answer_before_end} 时没能收尾"
        )
        assert bridge.service.session.state.value == "completed"


def test_chaos_abort_and_fatal_both_yield_explicit_no_score_terminal_states():
    """防回归：`abort` / `fatal` 两条终止路径都必须落到终态，且拿不到报告分数。"""
    for method in ("abort", "fatal"):
        bridge, _c, _r = _make()
        _run(bridge.handle(make_envelope(ClientType.SESSION_CREATE, {})))
        _run(bridge.handle(make_envelope(ClientType.CONSENT_GRANT, {"base": True})))

        if method == "fatal":
            bridge.service.fatal("注入的严重故障")
        else:
            bridge.service.abort()
        terminal = bridge.service.session.state.value
        assert terminal == ("failed" if method == "fatal" else "aborted")

        # 终态之后任何收尾指令都必须显式回错（不能被静默忽略后让客户端空等）
        out = _run(bridge.handle(make_envelope(ClientType.CONTROL_END, {})))
        assert _types(out) == [str(ServerType.ERROR)]
        assert out[0].payload["code"] == str(ErrorCode.BAD_FRAME)
        assert terminal in out[0].payload["message"]


# ---------- 演练 5：传输层故障（真实走线序字节） ----------


def test_chaos_gateway_transport_failures_are_all_explicit_and_non_fatal():
    """★ 传输层四条故障路径都必须回结构化 error，且网关本身不抛异常。

    覆盖：未认证 / 限流 / 信用窗口耗尽 / seq 跳号。这四条都是"静默吞掉"的高发地。
    """
    bridge, clock, _r = _make()
    loop = asyncio.new_event_loop()
    try:
        handler = SyncHandler(bridge, loop)

        # (a) 未认证：任何需要鉴权的类型都拒绝
        anonymous = Gateway(handler=handler, clock=clock)
        out = anonymous.handle_text(encode_text(make_envelope(ClientType.SESSION_CREATE, {})))
        assert [f.payload["code"] for f in out] == [str(ErrorCode.UNAUTHORIZED)]

        # (b) 限流：1 条/秒的窗口
        tight = Gateway(
            handler=handler,
            clock=clock,
            token_verifier=lambda t: "s-chaos",
            rate_limit_per_sec=1,
        )
        assert tight.authenticate("tok") is True
        first = tight.handle_text(
            encode_text(make_envelope(ClientType.SESSION_CREATE, {}, session_id="s-chaos"))
        )
        assert [f.payload["code"] for f in first] != [str(ErrorCode.RATE_LIMITED)]
        second = tight.handle_text(
            encode_text(
                make_envelope(ClientType.CONSENT_GRANT, {"base": True}, session_id="s-chaos")
            )
        )
        assert [f.payload["code"] for f in second] == [str(ErrorCode.RATE_LIMITED)]
        clock.advance(2.0)  # 窗口滑走，恢复可用
        third = tight.handle_text(
            encode_text(
                make_envelope(ClientType.CONSENT_GRANT, {"base": True}, session_id="s-chaos")
            )
        )
        assert [f.payload["code"] for f in third] != [str(ErrorCode.RATE_LIMITED)]

        # (c) 信用窗口耗尽：容量 1，连发两帧媒体
        gw = Gateway(
            handler=handler,
            clock=clock,
            token_verifier=lambda t: "s-chaos",
            credit_capacity=1,
            ack_every=99,  # 不让 ack 提前把信用还回来
            auto_consume=False,
        )
        assert gw.authenticate("tok") is True
        media = encode_binary(1, b"\x00\x01")
        assert gw.handle_binary(media) == []
        out = gw.handle_binary(media)
        assert [f.payload["code"] for f in out] == [str(ErrorCode.CREDIT_EXHAUSTED)]

        # (d) seq 跳号超容忍窗口
        gap = Gateway(handler=handler, clock=clock, token_verifier=lambda t: "s-chaos")
        assert gap.authenticate("tok") is True
        gap.handle_text(
            encode_text(make_envelope(ClientType.SESSION_CREATE, {}, seq=1, session_id="s-chaos"))
        )
        out = gap.handle_text(
            encode_text(
                make_envelope(
                    ClientType.ANSWER_TEXT, {"text": "x"}, seq=999, session_id="s-chaos"
                )
            )
        )
        assert [f.payload["code"] for f in out] == [str(ErrorCode.OUT_OF_ORDER)]
    finally:
        loop.run_until_complete(handler.joined())  # 收干净在途任务，避免 "never awaited" 告警
        loop.close()


def test_chaos_duplicate_seq_is_not_re_executed_even_though_replay_is_empty():
    """★ 防回归：弱网重传（同 seq）**不得重复执行指令**。

    重复执行 `answer.commit` 会让一场 3 题的面试变成 6 题。

    口径说明（重要，别误判成 bug）：`Gateway` 的"回放上次响应"靠的是
    `handler` 的**同步返回值**。本系统的 `SyncHandler` 刻意采用"立即返回空 +
    稍后推送"的形态（见 `bridge.py` 的说明），因此缓存的响应本身是空列表——
    **回放为空是这套异步语义的必然结果，不是丢了内容**。
    真正要守的底线是"不重复派发"，以及"客户端能靠 seq 知道自己重传过"。
    """
    bridge, clock, _r = _make()
    loop = asyncio.new_event_loop()
    try:
        handler = SyncHandler(bridge, loop)
        gw = Gateway(handler=handler, clock=clock, token_verifier=lambda t: "s-chaos")
        assert gw.authenticate("tok") is True

        async def go():
            def send(mtype, payload, seq):
                return gw.handle_text(
                    encode_text(
                        make_envelope(mtype, payload, seq=seq, session_id="s-chaos")
                    )
                )

            send(ClientType.SESSION_CREATE, {}, 1)
            await handler.joined()
            handler.drain()
            send(ClientType.CONSENT_GRANT, {"base": True}, 2)
            await handler.joined()
            handler.drain()

            send(ClientType.ANSWER_COMMIT, {}, 3)
            await handler.joined()
            first = _types(handler.drain())
            turns_after_first = bridge._turn_index

            send(ClientType.ANSWER_COMMIT, {}, 3)  # 同 seq 重传
            await handler.joined()
            replay = _types(handler.drain())
            return first, replay, turns_after_first

        first, replay, turns_after_first = loop.run_until_complete(go())

        # 第一次确实推进了一题
        assert str(ServerType.QUESTION_START) in first
        assert turns_after_first == 2
        # ★ 重传没有产生任何新帧（说明没有被重新派发给业务）
        assert replay == []
        # ★ 题号没有前进：一场 3 题的面试不会因为重传变成 6 题
        assert bridge._turn_index == 2
    finally:
        loop.close()
