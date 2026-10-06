"""端到端失败 / 降级 / 熔断路径。

本文件防的回归
--------------
最高优先级红线：**绝不合成兜底分数**。任何一轮没拿到真实评估，这轮就不进聚合；
整场没有真实评估时 `Report.available=False` 且 `score=None`——绝不允许出现
"崩溃即返回全 60 分"那种与真实 60 分不可区分的数字（`test_no_synthetic_fallback_*`）。

其余红线：降级必须可见（有中文徽标、进报告说明），但**降级不是不可用**，分数照出；
未取得生理单独授权时数据**根本不进服务**；达不到最少评分数就不出分；
报告必须能被严格 JSON 解析（NaN 转 None）。
"""

from __future__ import annotations

import asyncio
import json

import pytest

from ruipin.adapters.fakes import DEFAULT_DIMS, DeterministicClock, FakeEvaluator
from ruipin.adapters.repo_sqlite import SqliteRepo
from ruipin.domain.errors import InvalidTransition, Unavailable
from ruipin.domain.states import State
from ruipin.orchestrator import (
    RUBRIC_VERSION,
    Consent,
    InterviewConfig,
    InterviewService,
    TurnScheduler,
)
from ruipin.orchestrator.service import Report
from ruipin.ports import DIMENSIONS, EvalRequest, EvalResult, InterviewEvent
from ruipin.scoring.aggregate import GATE_LOW

pytestmark = pytest.mark.integration

#: 用例期间创建的内存库，由下面的 autouse fixture 统一关闭（避免 sqlite3 连接泄漏）
_OPEN_REPOS: list[SqliteRepo] = []


@pytest.fixture(autouse=True)
def _close_repos():
    _OPEN_REPOS.clear()
    yield
    while _OPEN_REPOS:
        _OPEN_REPOS.pop().close()


QUESTIONS: tuple[str, ...] = (
    "Q1：请介绍一个你主导过的项目。",
    "Q2：线上 P99 延迟翻倍，你怎么排查？",
    "Q3：描述一次技术路线分歧与收敛过程。",
    "Q4：你如何保证模块在迭代中不退化？",
)
SESSION_ID = "s-fail"


def _has_cjk(text: str) -> bool:
    return any("一" <= ch <= "鿿" for ch in text)


class _Env:
    def __init__(self, evaluator, **cfg):
        self.clock = DeterministicClock()
        self.repo = SqliteRepo(":memory:", clock=self.clock.now)
        _OPEN_REPOS.append(self.repo)
        self.evaluator = evaluator
        scheduler = TurnScheduler(evaluator, clock=self.clock.monotonic, sleeper=lambda _s: None)
        self.svc = InterviewService(
            self.repo,
            scheduler,
            InterviewConfig(questions=QUESTIONS, rubric_version=RUBRIC_VERSION, **cfg),
            session_id=SESSION_ID,
            clock=self.clock.now,
        )

    def start(self, *, consent=None):
        self.svc.create()
        self.svc.record_consent(consent or Consent(base=True))
        self.svc.greeting_done()
        return self

    def answer_all(self, text: str = "作答"):
        async def go():
            while True:
                if self.svc.ask_next() is None:
                    return
                await self.svc.submit_answer(text)

        asyncio.run(go())
        return self

    def finish(self):
        self.svc.finish_qa()
        self.svc.close()
        return self.svc.finalize()

    def event_names(self):
        return [e["event"] for e in self.repo.load_events(SESSION_ID)]


class _FailsOnQuestions:
    """对指定题目**恒定**失败的评估器（重试也失败，模拟 provider 硬故障）。

    必须"恒定"失败：调度器对 `Unavailable` 会重试 1 次，若第二次成功则该轮
    会被记为已评分，就构不成"部分轮次失败"的场景。
    """

    name = "scripted"

    def __init__(self, ok: FakeEvaluator, failing: set[str]):
        self._ok = ok
        self.failing = failing
        self.calls: list[str] = []

    async def evaluate(self, req: EvalRequest) -> EvalResult:
        self.calls.append(req.question)
        if req.question in self.failing:
            raise Unavailable("scripted", f"injected failure: {req.question}")
        return await self._ok.evaluate(req)


# --------------------------------------------------------------------------
# 10. 绝不合成兜底分数（最高优先级红线）
# --------------------------------------------------------------------------


def test_no_synthetic_fallback_score_when_every_eval_fails():
    """防回归：整场评估全失败时，报告必须"没有分数"，而不是给一个 60 分。"""
    env = _Env(FakeEvaluator(fail=True))
    env.start()
    env.answer_all()
    report = env.finish()

    assert report.available is False
    assert report.score is None, "没有真实评估时绝不能给出任何数字"
    assert report.level is None
    assert "真实评估轮数" in report.reason
    assert "0 < 1" in report.reason
    assert report.n_scored_turns == 0

    # 判据（为什么这能证明"没有兜底"）：
    # ① dims 必须**完全为空**——不是"填了默认值"，而是这个维度根本没被评估过；
    #    评分器 DEFAULT_DIMS 里的任何数值（60/65/68/70/72）都不允许出现在报告里。
    assert report.dims == {}
    assert not set(DEFAULT_DIMS.values()) & set(report.dims.values())
    assert 60.0 not in report.dims.values()

    # ② 全链路落库物里也搜不到任何兜底数字
    payload = json.loads(json.dumps(report.to_dict(), allow_nan=False))
    assert payload["score"] is None
    assert payload["dims"] == {}
    assert payload["level"] is None
    assert "60" not in payload["reason"]

    # ③ 评估器真的被调用过（4 题 × 2 次尝试：1 次主调 + 1 次重试），
    #    证明"没分"是失败导致的，不是压根没去评
    assert len(env.evaluator.calls) == len(QUESTIONS) * 2

    # ④ 每轮记录都标记未评分、且没有 eval 快照
    rows = env.repo.load_turns(SESSION_ID)
    assert len(rows) == len(QUESTIONS)
    for row in rows:
        assert row["scored"] is False
        assert row["eval"] is None
        assert row["degrade_reason"] is not None and "scoring_failed" in row["degrade_reason"]
    # 失败轮也走 EVAL_DEGRADED（不是假装成功）
    assert env.event_names().count("eval_degraded") == len(QUESTIONS)
    assert env.event_names().count("eval_done") == 0


# --------------------------------------------------------------------------
# 11. 部分轮次失败
# --------------------------------------------------------------------------


def test_only_successfully_scored_turns_enter_aggregation():
    """防回归：一半轮次失败时，只有成功轮进聚合，总分只由成功轮算出。"""
    failing = {QUESTIONS[1], QUESTIONS[3]}
    scripted = _FailsOnQuestions(FakeEvaluator(), failing)
    env = _Env(scripted)
    env.start()
    env.answer_all()
    report = env.finish()

    assert report.available is True
    assert report.n_scored_turns == 2, "只有 2 轮拿到真实评估"
    assert len(env.svc.scored_turns) == 2
    assert {t.question for t in env.svc.scored_turns} == {QUESTIONS[0], QUESTIONS[2]}
    # 失败的两题各被尝试 2 次（主调 + 重试），成功的两题各 1 次
    assert len(scripted.calls) == 6

    rows = env.repo.load_turns(SESSION_ID)
    assert [row["scored"] for row in rows] == [True, False, True, False]

    # 总分只能由成功轮算出：用报告自己的 dims 与生效权重复算一遍
    recomputed = sum(report.dims[k] * report.effective_weights[k] for k in report.dims)
    assert report.score == pytest.approx(round(recomputed, 2), abs=1e-2)
    # 两个成功轮的 dims 完全相同 → 合成分仍等于那组维度值，未被失败轮稀释
    assert report.score == pytest.approx(65.0, abs=1e-6)


# --------------------------------------------------------------------------
# 12. 降级可见，但降级 ≠ 不可用
# --------------------------------------------------------------------------


def test_degraded_result_is_visible_but_still_scored():
    """防回归：降级必须有中文徽标、进报告说明，但分数**照常出**（降级不是不可用）。"""
    env = _Env(
        FakeEvaluator(degraded=True, degrade_reason="模型不可用，改用规则评分（仅供参考）")
    )
    env.start()
    env.svc.ask_next()
    outcome = asyncio.run(env.svc.submit_answer("作答一"))

    assert outcome.eval is not None, "降级结果仍是可用的评估结果"
    assert outcome.degraded is True
    assert outcome.degrade_reason is not None
    assert "模型不可用" in outcome.degrade_reason
    assert outcome.ui_badge, "禁止静默降级：必须有徽标文案"
    assert _has_cjk(outcome.ui_badge), f"徽标必须是中文: {outcome.ui_badge!r}"

    env.svc.ask_next()
    asyncio.run(env.svc.submit_answer("作答二"))
    env.svc.ask_next()
    asyncio.run(env.svc.submit_answer("作答三"))
    env.svc.ask_next()
    asyncio.run(env.svc.submit_answer("作答四"))
    assert env.svc.ask_next() is None
    report = env.finish()

    assert report.available is True
    assert report.score is not None
    assert report.level in {"A", "B", "C", "D"}
    assert any("降级" in note for note in report.notes), report.notes
    # 降级维度权重被折半，但权重总和仍为 1.0（折半的那部分按比例还给了其他维度）
    assert sum(report.effective_weights.values()) == pytest.approx(1.0)
    assert env.event_names().count("eval_degraded") == 4
    # 降级原因随回合落库
    rows = env.repo.load_turns(SESSION_ID)
    assert all("模型不可用" in row["degrade_reason"] for row in rows)


# --------------------------------------------------------------------------
# 13. 中途放弃 / 严重故障
# --------------------------------------------------------------------------


def test_abort_freezes_session_and_blocks_every_further_call():
    """防回归：放弃后会话进入终态，出题/作答/出报告全部拒绝，不留下半成品。"""
    env = _Env(FakeEvaluator())
    env.start()
    env.svc.ask_next()

    assert env.svc.abort() is State.ABORTED
    assert env.svc.session.is_terminal is True

    with pytest.raises(InvalidTransition):
        env.svc.ask_next()
    with pytest.raises(InvalidTransition):
        env.svc.skip()
    with pytest.raises(InvalidTransition):
        env.svc.finalize()

    async def submit() -> None:
        with pytest.raises(InvalidTransition):
            await env.svc.submit_answer("迟到的作答")

    asyncio.run(submit())

    # 放弃的会话没有落任何回合，也不该产出报告
    assert env.repo.load_turns(SESSION_ID) == []
    assert env.repo.load_session(SESSION_ID)["state"] == "aborted"


def test_fatal_ends_session_without_report():
    """防回归：严重故障时进入 FAILED（只保存答案、事后补评估），绝不现场编分数。"""
    env = _Env(FakeEvaluator())
    env.start()
    env.svc.ask_next()
    assert env.svc.fatal(detail="storage down") is State.FAILED
    assert env.svc.session.is_terminal is True
    with pytest.raises(InvalidTransition):
        env.svc.finalize()


def test_silence_timeout_lands_in_processing_without_scored_turn():
    """防回归：静默超时后进入 PROCESSING；此时既不能出题也不能再提交作答。"""
    env = _Env(FakeEvaluator())
    env.start()
    env.svc.ask_next()
    assert env.svc.silence_timeout() is State.PROCESSING
    with pytest.raises(InvalidTransition):
        env.svc.ask_next()

    async def submit() -> None:
        with pytest.raises(InvalidTransition):
            await env.svc.submit_answer("迟到的作答")

    asyncio.run(submit())


# --------------------------------------------------------------------------
# 14/15. 授权
# --------------------------------------------------------------------------


def test_physio_data_rejected_without_separate_consent():
    """防回归（PIPL 第 28 条）：未单独授权时生理数据**根本不进服务**，不是收了再弃用。"""
    env = _Env(FakeEvaluator(), physio_enabled=True)
    env.start(consent=Consent(base=True, physio=False))

    event = InterviewEvent(turn_index=0, pre_bpm=70.0, peak_bpm=91.0, t50_s=12.0)
    assert env.svc.add_physio_event(event) is False
    assert env.svc.physio_events == ()

    env.answer_all()
    report = env.finish()

    assert report.physio is not None
    assert report.physio.available is False
    assert "授权" in report.physio.reason
    assert any("授权" in note for note in report.notes), report.notes
    # 未授权不得影响其余维度的可用性
    assert report.available is True
    assert set(report.dims) == set(DIMENSIONS)
    assert "stress_regulation" not in report.effective_weights


def test_consent_denied_aborts_immediately():
    """防回归：拒绝参加即 ABORTED，之后任何推进都必须被拒绝。"""
    env = _Env(FakeEvaluator())
    env.svc.create()
    assert env.svc.record_consent(Consent(base=False)) is State.ABORTED
    with pytest.raises(InvalidTransition):
        env.svc.greeting_done()
    with pytest.raises(InvalidTransition):
        env.svc.ask_next()
    assert env.repo.load_session(SESSION_ID)["consent"] == {
        "base": False,
        "physio": False,
        "media_recording": False,
    }


# --------------------------------------------------------------------------
# 16. min_scored_turns 闸门
# --------------------------------------------------------------------------


def test_min_scored_turns_gate_blocks_the_report():
    """防回归：真实评估轮数不够时不出分（宁缺毋滥），而不是降低标准凑一个。"""
    env = _Env(FakeEvaluator(), min_scored_turns=3)
    env.start()

    # 只答 2 题，剩下 2 题跳过（跳过不计分、不算一轮）
    env.svc.ask_next()
    asyncio.run(env.svc.submit_answer("作答一"))
    env.svc.ask_next()
    asyncio.run(env.svc.submit_answer("作答二"))
    for _ in range(2):
        env.svc.ask_next()
        assert env.svc.skip() is State.ASKING
    assert env.svc.ask_next() is None

    report = env.finish()

    assert report.available is False
    assert report.score is None
    assert report.level is None
    assert "真实评估轮数 2 < 3" in report.reason
    assert "不生成兜底分数" in report.reason
    assert report.n_scored_turns == 2


def test_aggregate_unavailable_when_no_dimension_passes_the_gate():
    """防回归：轮数够但**没有一个维度可信**时，聚合器拒绝出分（绝不给中间分）。"""
    # confidence=0 → 门控系数 0（< GATE_LOW），所有维度权重归零
    env = _Env(FakeEvaluator(confidence=0.0), min_scored_turns=1)
    env.start()
    env.answer_all()
    report = env.finish()

    assert report.available is False
    assert report.score is None
    assert report.level is None
    assert report.reason is not None and report.reason.startswith("聚合不可用")
    assert report.n_scored_turns == len(QUESTIONS)
    # 维度值仍然如实记录（未评估的维度不会出现，被评估但不可信的会如实写出）
    assert set(report.dims) == set(DIMENSIONS)
    assert GATE_LOW == 0.40


# --------------------------------------------------------------------------
# 17. 报告可序列化
# --------------------------------------------------------------------------


def test_report_to_dict_is_strict_json_serializable():
    """防回归：报告必须能被严格 JSON 解析（NaN 转 None），否则跨服务传输会被拒。"""
    env = _Env(FakeEvaluator(), physio_enabled=True)
    env.start(consent=Consent(base=True, physio=True))
    env.svc.set_baseline(68.0)
    env.svc.set_physio_quality(median_snr=0.9, reject_ratio=0.05, algo_agreement=1.0)
    for i in range(10):
        env.svc.add_physio_event(
            InterviewEvent(turn_index=i, pre_bpm=70.0, peak_bpm=84.0 - 0.4 * i, t50_s=12.0 + 0.3 * i)
        )
    env.answer_all()
    report = env.finish()

    raw = json.dumps(report.to_dict(), ensure_ascii=False, allow_nan=False)
    back = json.loads(raw)

    assert back["session_id"] == SESSION_ID
    assert back["available"] is True
    assert back["score"] == report.score
    assert back["level"] == report.level
    assert back["dims"] == report.dims
    assert back["effective_weights"] == report.effective_weights
    assert back["notes"] == report.notes
    assert back["rubric_version"] == RUBRIC_VERSION
    assert back["n_scored_turns"] == len(QUESTIONS)
    assert back["n_buffer_turns"] == 0
    assert back["physio"]["available"] is True
    assert back["physio"]["n_valid"] == 10
    for key in ("session_id", "available", "reason", "score", "level", "dims",
                "effective_weights", "redistributed", "counterfactuals", "notes",
                "physio", "rubric_version", "n_scored_turns", "n_buffer_turns"):
        assert key in back, f"报告缺少字段 {key}"


def test_nan_counterfactual_is_converted_to_null():
    """防回归：反事实里的 NaN 必须转 None——`json.dumps(allow_nan=False)` 会拒绝 NaN。"""
    with pytest.raises(ValueError):
        json.dumps({"technical": float("nan")}, allow_nan=False)

    report = Report(
        session_id="s-nan",
        available=True,
        reason=None,
        score=65.0,
        level="C",
        dims={"technical": 70.0},
        effective_weights={"technical": 1.0},
        counterfactuals={"technical": float("nan")},
    )
    payload = report.to_dict()
    assert payload["counterfactuals"]["technical"] is None
    assert json.loads(json.dumps(payload, allow_nan=False))["counterfactuals"] == {"technical": None}
