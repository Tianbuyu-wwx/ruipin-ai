"""端到端主链路：正常闭环、可回放、适应性缓冲、生理维度授权与计权。

本文件防的回归（每条都有对应用例）
----------------------------------
1. 整场面试能否从 IDLE 走到 COMPLETED 并产出**可用**报告（不是"看起来可用"）。
2. 事件日志能否**逐条**回放——这是"缓冲题没被评分"唯一的硬证据来源。
3. 缓冲题不计分（自检 A3）：不调用评估器、不进 PROCESSING、日志里没有 EVAL_*。
4. 适应性调节不得改变核心题序列（自检 A3"可比性不被破坏"的可测形式）。
5. 生理维度单独授权；授权且可靠时按 W_MAX 计入，未授权时权重全额归还，
   **总分不被系统性压低**（验收红线）。

纪律：无 sleep、无随机数、无真实时钟——时钟一律注入 `DeterministicClock`。
"""

from __future__ import annotations

import asyncio
from typing import Any, NamedTuple, Optional

import pytest

from ruipin.adapters.fakes import DeterministicClock, FakeEvaluator
from ruipin.adapters.repo_sqlite import SqliteRepo
from ruipin.domain.states import State
from ruipin.orchestrator import (
    RUBRIC_VERSION,
    Consent,
    InterviewConfig,
    InterviewService,
    TurnScheduler,
)
from ruipin.physio.regulation import DEFAULT_WEIGHTS as PHYSIO_DEFAULT_WEIGHTS
from ruipin.physio.regulation import W_MAX
from ruipin.ports import DIMENSIONS, PHYSIO_DIM, InterviewEvent

pytestmark = pytest.mark.integration

#: 用例期间创建的内存库，由下面的 autouse fixture 统一关闭（避免 sqlite3 连接泄漏）
_OPEN_REPOS: list[SqliteRepo] = []


@pytest.fixture(autouse=True)
def _close_repos():
    _OPEN_REPOS.clear()
    yield
    while _OPEN_REPOS:
        _OPEN_REPOS.pop().close()


CORE_QUESTIONS: tuple[str, ...] = (
    "Q1：请介绍一个你主导过的项目。",
    "Q2：线上 P99 延迟翻倍，你怎么排查？",
    "Q3：描述一次技术路线分歧与收敛过程。",
    "Q4：你如何保证模块在迭代中不退化？",
)
BUFFER_QUESTION = "B1：换个轻松的话题，你最近在读什么书？"

#: 弱应激事件：r = 7/70 = 0.10 < 阈值 0.25，不会触发缓冲题
WEAK_EVENT = InterviewEvent(turn_index=0, pre_bpm=70.0, peak_bpm=77.0, t50_s=10.0)
#: 强应激事件：r = 35/70 = 0.50 ≥ 阈值 0.25，会触发缓冲题
STRONG_EVENT = InterviewEvent(turn_index=1, pre_bpm=70.0, peak_bpm=105.0, t50_s=20.0)

SESSION_ID = "s-e2e"


class _Env(NamedTuple):
    clock: DeterministicClock
    repo: SqliteRepo
    svc: InterviewService
    evaluator: Any


def _env(
    *,
    questions: tuple[str, ...] = CORE_QUESTIONS,
    buffer_questions: tuple[str, ...] = (),
    weights: Optional[dict[str, float]] = None,
    adaptive_enabled: bool = False,
    physio_enabled: bool = False,
    evaluator: Optional[Any] = None,
    min_scored_turns: int = 1,
    rubric_version: str = RUBRIC_VERSION,
    reactivity_threshold: float = 0.25,
    session_id: str = SESSION_ID,
) -> _Env:
    """搭一个完全确定性的InterviewService（内存库 + 虚拟时钟）。"""
    clock = DeterministicClock()
    repo = SqliteRepo(":memory:", clock=clock.now)
    ev = evaluator if evaluator is not None else FakeEvaluator()
    scheduler = TurnScheduler(ev, clock=clock.monotonic, sleeper=lambda _s: None)
    config = InterviewConfig(
        questions=questions,
        buffer_questions=buffer_questions,
        weights=dict(weights if weights is not None else PHYSIO_DEFAULT_WEIGHTS),
        adaptive_enabled=adaptive_enabled,
        physio_enabled=physio_enabled,
        reactivity_threshold=reactivity_threshold,
        rubric_version=rubric_version,
        min_scored_turns=min_scored_turns,
    )
    svc = InterviewService(repo, scheduler, config, session_id=session_id, clock=clock.now)
    _OPEN_REPOS.append(repo)
    return _Env(clock, repo, svc, ev)


def _physio_events(n: int = 10, *, pre: float = 70.0, delta: float = 14.0, t50: float = 12.0):
    """n 个固定生理事件：r_i 从 0.20 递减，全部低于阈值（不触发缓冲题）。"""
    return [
        InterviewEvent(
            turn_index=i,
            pre_bpm=pre,
            peak_bpm=pre + delta - 0.4 * i,
            t50_s=t50 + 0.3 * i,
        )
        for i in range(n)
    ]


def _drive(svc: InterviewService, answers: tuple[str, ...]) -> list[str]:
    """按固定脚本跑完整场（含缓冲回合），返回依次呈现的题目文本。"""
    asked: list[str] = []

    async def go() -> None:
        core_i = 0
        while True:
            is_buffer = svc.session.state == State.BUFFER
            question = svc.ask_next()
            if question is None:
                return
            asked.append(question)
            if is_buffer:
                await svc.submit_answer("（缓冲题作答，不评分）")
            else:
                await svc.submit_answer(answers[core_i % len(answers)])
                core_i += 1

    asyncio.run(go())
    return asked


def _finish(svc: InterviewService):
    """CANDIDATE_QA → CLOSING → REPORTING → COMPLETED。"""
    assert svc.finish_qa() is State.CLOSING
    assert svc.close() is State.REPORTING
    return svc.finalize()


def _core_questions_from_log(repo: SqliteRepo, session_id: str) -> list[str]:
    """从**落库的事件日志**（而非测试自己的记账）反推核心题序列。

    判据：`tts_done/tts_failed` 事件里 `payload["scored"] is True` 的才是核心题；
    缓冲题的 payload 带 `scored=False`。这样"核心题序列"是日志可审计的事实。
    """
    return [
        e["payload"]["question"]
        for e in repo.load_events(session_id)
        if e["event"] in ("tts_done", "tts_failed") and e["payload"].get("scored") is True
    ]


def _event_names(repo: SqliteRepo, session_id: str) -> list[str]:
    return [e["event"] for e in repo.load_events(session_id)]


# --------------------------------------------------------------------------
# 1. 完整闭环
# --------------------------------------------------------------------------


def test_full_interview_closed_loop():
    """防回归：整场面试能闭环到 COMPLETED，且报告里的分数字段都是真实值。"""
    env = _env()
    assert env.svc.create(meta={"candidate": "张三"}) is State.SETUP
    assert env.svc.record_consent(Consent(base=True, physio=False)) is State.GREETING
    assert env.svc.greeting_done() is State.ASKING

    asked = _drive(env.svc, ("作答一", "作答二", "作答三", "作答四"))
    assert asked == list(CORE_QUESTIONS)

    # 题出完：_drive 里最后一次 ask_next 返回 None 并把状态推进到 CANDIDATE_QA（不是异常）
    assert env.svc.session.state is State.CANDIDATE_QA

    report = _finish(env.svc)

    assert env.svc.session.state is State.COMPLETED
    assert report.available is True
    assert report.reason is None
    assert report.score is not None and 0.0 <= report.score <= 100.0
    assert report.level in {"A", "B", "C", "D"}
    assert report.n_scored_turns == len(CORE_QUESTIONS)
    assert report.n_buffer_turns == 0
    assert report.rubric_version == RUBRIC_VERSION
    assert set(report.dims) == set(DIMENSIONS)

    # create(meta=...) 真的落库了（不是只改内存）
    stored = env.repo.load_session(SESSION_ID)
    assert stored["meta"]["candidate"] == "张三"
    assert stored["state"] == "completed"


# --------------------------------------------------------------------------
# 2. 持久化可回放
# --------------------------------------------------------------------------


def test_event_log_replays_every_transition_in_order():
    """防回归：事件日志可逐条回放——seq、from、event、to 与内存会话完全一致。"""
    env = _env()
    env.svc.create()
    env.svc.record_consent(Consent(base=True))
    env.svc.greeting_done()
    _drive(env.svc, ("a", "b", "c", "d"))
    _finish(env.svc)

    logged = env.repo.load_events(SESSION_ID)
    memory = env.svc.session.events
    assert len(logged) == len(memory)

    for i, (row, ev) in enumerate(zip(logged, memory)):
        assert row["seq"] == ev.seq == i + 1, f"第 {i} 条事件 seq 对不上"
        assert row["event"] == ev.event.value, f"第 {i} 条事件名对不上"
        assert row["from"] == (ev.from_state.value if ev.from_state else None)
        assert row["to"] == ev.to_state.value

    # 事件序列本身就是主链的指纹：create → consent → greeting → (出题/作答/评分)×4 → 收尾
    assert _event_names(env.repo, SESSION_ID) == [
        "create", "consent_ok", "greeting_done",
        "tts_done", "answer_commit", "eval_done",
        "tts_done", "answer_commit", "eval_done",
        "tts_done", "answer_commit", "eval_done",
        "tts_done", "answer_commit", "eval_done",
        "no_more_questions", "qa_done", "closing_done", "report_done",
    ]


def test_persisted_turn_count_matches_scored_turns():
    """防回归：落库的回合条数与真实评分数一致，且每条都带完整评估快照。"""
    env = _env()
    env.svc.create()
    env.svc.record_consent(Consent(base=True))
    env.svc.greeting_done()
    _drive(env.svc, ("a", "b", "c", "d"))
    report = _finish(env.svc)

    turns = env.repo.load_turns(SESSION_ID)
    assert len(turns) == len(env.svc.scored_turns) == report.n_scored_turns == 4
    for i, row in enumerate(turns):
        assert row["turn_index"] == i
        assert row["scored"] is True
        assert row["eval"] is not None
        assert set(row["eval"]["dims"]) == set(DIMENSIONS)
        assert row["degrade_reason"] is None


# --------------------------------------------------------------------------
# 3. 缓冲题不计分
# --------------------------------------------------------------------------


def test_buffer_turn_is_not_scored_and_leaves_no_eval_event():
    """防回归（自检 A3）：缓冲题不调用评估器、不进 PROCESSING、日志里没有 EVAL_*。"""
    env = _env(
        questions=CORE_QUESTIONS[:3],
        buffer_questions=(BUFFER_QUESTION,),
        adaptive_enabled=True,
        physio_enabled=True,
    )
    env.svc.create()
    env.svc.record_consent(Consent(base=True, physio=True))
    env.svc.greeting_done()

    # 第 1 轮：弱应激 → 不插入缓冲题
    assert env.svc.ask_next() == CORE_QUESTIONS[0]
    assert env.svc.add_physio_event(WEAK_EVENT) is True
    asyncio.run(env.svc.submit_answer("作答一"))
    assert env.svc.session.state is State.ASKING

    # 第 2 轮：强应激（r=0.50）→ 评完分后插入缓冲题
    assert env.svc.ask_next() == CORE_QUESTIONS[1]
    assert env.svc.add_physio_event(STRONG_EVENT) is True
    asyncio.run(env.svc.submit_answer("作答二"))
    assert env.svc.session.state is State.BUFFER

    # 缓冲回合：呈现缓冲题 → 作答 → 直接回 ASKING（不进 PROCESSING）
    assert env.svc.ask_next() == BUFFER_QUESTION
    assert env.svc.session.state is State.BUFFER_LISTENING
    outcome = asyncio.run(env.svc.submit_answer("缓冲作答"))
    assert outcome.eval is None, "缓冲回合绝不能产出评估"
    assert env.svc.session.state is State.ASKING

    # 第 3 轮：补一个弱事件，避免同一个强事件再次触发缓冲
    assert env.svc.ask_next() == CORE_QUESTIONS[2]
    assert env.svc.add_physio_event(WEAK_EVENT) is True
    asyncio.run(env.svc.submit_answer("作答三"))
    assert env.svc.ask_next() is None

    report = _finish(env.svc)

    assert report.n_buffer_turns == 1
    assert report.n_scored_turns == 3
    # 评估器只被调用 3 次（4 个回合里有 1 个是缓冲题）
    assert len(env.evaluator.calls) == 3
    assert len(env.svc.buffer_turns) == 1
    assert env.svc.buffer_turns[0].question == BUFFER_QUESTION
    assert env.svc.buffer_turns[0].eval is None
    assert env.svc.buffer_turns[0].scored is False

    # ★ 缓冲回合区段内没有 eval_done / eval_degraded。
    # 区段界定：事件日志中从 insert_buffer 起、到紧随其后的 buffer_done 止的闭区间
    # （缓冲回合 = BUFFER --tts_done--> BUFFER_LISTENING --buffer_done--> ASKING）。
    names = _event_names(env.repo, SESSION_ID)
    i_insert = names.index("insert_buffer")
    i_done = names.index("buffer_done", i_insert)
    segment = names[i_insert : i_done + 1]
    assert segment == ["insert_buffer", "tts_done", "buffer_done"]
    assert "eval_done" not in segment
    assert "eval_degraded" not in segment
    assert "answer_commit" not in segment, "缓冲回合不该走 answer_commit（不进 PROCESSING）"


def test_repeat_buffer_question_never_reaches_scoring_lane():
    """防回归：缓冲题"重听"后仍在缓冲链路内，不会被送进评分链路。"""
    env = _env(
        questions=CORE_QUESTIONS[:2],
        buffer_questions=(BUFFER_QUESTION,),
        adaptive_enabled=True,
        physio_enabled=True,
    )
    env.svc.create()
    env.svc.record_consent(Consent(base=True, physio=True))
    env.svc.greeting_done()

    env.svc.ask_next()
    env.svc.add_physio_event(STRONG_EVENT)
    asyncio.run(env.svc.submit_answer("作答一"))
    assert env.svc.session.state is State.BUFFER

    env.svc.ask_next()
    assert env.svc.session.state is State.BUFFER_LISTENING
    assert env.svc.repeat_question() is State.BUFFER  # 重听缓冲题：回到呈现态
    env.svc.ask_next()  # 重新呈现缓冲题
    assert env.svc.session.state is State.BUFFER_LISTENING
    asyncio.run(env.svc.submit_answer("缓冲作答"))
    assert env.svc.session.state is State.ASKING

    names = _event_names(env.repo, SESSION_ID)
    i_insert = names.index("insert_buffer")
    i_done = names.index("buffer_done", i_insert)
    segment = names[i_insert : i_done + 1]
    assert "eval_done" not in segment and "eval_degraded" not in segment
    assert "answer_commit" not in segment


def test_buffer_insertion_gates():
    """防回归：缓冲题的四道闸门——开关、库存、有观测、r 超阈值，缺一不可。"""
    cases = {
        "无缓冲题库": dict(buffer_questions=(), add_event=STRONG_EVENT),
        "没有任何生理观测": dict(buffer_questions=(BUFFER_QUESTION,), add_event=None),
        "r 为 None（缺 pre_bpm）": dict(
            buffer_questions=(BUFFER_QUESTION,),
            add_event=InterviewEvent(turn_index=0, pre_bpm=None, peak_bpm=90.0, t50_s=10.0),
        ),
        "r 未达阈值": dict(buffer_questions=(BUFFER_QUESTION,), add_event=WEAK_EVENT),
    }
    for name, kwargs in cases.items():
        env = _env(
            questions=CORE_QUESTIONS[:2],
            buffer_questions=kwargs["buffer_questions"],
            adaptive_enabled=True,
            physio_enabled=True,
        )
        env.svc.create()
        env.svc.record_consent(Consent(base=True, physio=True))
        env.svc.greeting_done()
        env.svc.ask_next()
        if kwargs["add_event"] is not None:
            env.svc.add_physio_event(kwargs["add_event"])
        asyncio.run(env.svc.submit_answer("作答"))
        assert env.svc.session.state is State.ASKING, f"{name}：不该插入缓冲题"
        assert "insert_buffer" not in _event_names(env.repo, SESSION_ID), name

    # 反例：四道闸门全开时必须插入
    env = _env(
        questions=CORE_QUESTIONS[:2],
        buffer_questions=(BUFFER_QUESTION,),
        adaptive_enabled=True,
        physio_enabled=True,
    )
    env.svc.create()
    env.svc.record_consent(Consent(base=True, physio=True))
    env.svc.greeting_done()
    env.svc.ask_next()
    env.svc.add_physio_event(STRONG_EVENT)
    asyncio.run(env.svc.submit_answer("作答"))
    assert env.svc.session.state is State.BUFFER
    assert "insert_buffer" in _event_names(env.repo, SESSION_ID)


# --------------------------------------------------------------------------
# 4. 缓冲题不破坏核心题序列
# --------------------------------------------------------------------------


def test_core_question_sequence_is_identical_with_and_without_adaptive():
    """防回归（自检 A3）：开不开适应性，核心题序列必须逐题相同（可比性不被破坏）。"""
    # A：不开适应性
    env_a = _env(questions=CORE_QUESTIONS[:3])
    env_a.svc.create()
    env_a.svc.record_consent(Consent(base=True))
    env_a.svc.greeting_done()
    _drive(env_a.svc, ("a", "b", "c"))
    _finish(env_a.svc)

    # B：开适应性，并且每题都给一个强应激事件（每题后都插缓冲题）
    env_b = _env(
        questions=CORE_QUESTIONS[:3],
        buffer_questions=(BUFFER_QUESTION,),
        adaptive_enabled=True,
        physio_enabled=True,
    )
    env_b.svc.create()
    env_b.svc.record_consent(Consent(base=True, physio=True))
    env_b.svc.greeting_done()

    async def go_b() -> None:
        for i in range(3):
            env_b.svc.ask_next()
            env_b.svc.add_physio_event(STRONG_EVENT)
            await env_b.svc.submit_answer(f"作答{i}")
            # 缓冲回合
            if env_b.svc.session.state is State.BUFFER:
                env_b.svc.ask_next()
                await env_b.svc.submit_answer("缓冲作答")

    asyncio.run(go_b())
    assert env_b.svc.ask_next() is None  # 题尽 → CANDIDATE_QA
    report_b = _finish(env_b.svc)

    core_a = _core_questions_from_log(env_a.repo, SESSION_ID)
    core_b = _core_questions_from_log(env_b.repo, SESSION_ID)
    assert core_a == core_b == list(CORE_QUESTIONS[:3])
    assert report_b.n_buffer_turns == 3
    assert report_b.n_scored_turns == 3


# --------------------------------------------------------------------------
# 5/6/7. 生理维度
# --------------------------------------------------------------------------


def _run_physio_pair() -> tuple[Any, Any]:
    """同一场面试跑两遍：一次授权生理、一次不授权，返回 (report_auth, report_denied)。"""
    authorized = _env(physio_enabled=True)
    authorized.svc.create()
    authorized.svc.record_consent(Consent(base=True, physio=True))
    authorized.svc.greeting_done()
    authorized.svc.set_baseline(68.0)
    authorized.svc.set_physio_quality(median_snr=0.9, reject_ratio=0.05, algo_agreement=1.0)
    for ev in _physio_events(10):
        assert authorized.svc.add_physio_event(ev) is True
    _drive(authorized.svc, ("a", "b", "c", "d"))
    report_auth = _finish(authorized.svc)

    denied = _env(physio_enabled=True)
    denied.svc.create()
    denied.svc.record_consent(Consent(base=True, physio=False))
    denied.svc.greeting_done()
    denied.svc.set_baseline(68.0)
    denied.svc.set_physio_quality(median_snr=0.9, reject_ratio=0.05, algo_agreement=1.0)
    for ev in _physio_events(10):
        assert denied.svc.add_physio_event(ev) is False  # 未授权：拒收
    _drive(denied.svc, ("a", "b", "c", "d"))
    report_denied = _finish(denied.svc)

    return report_auth, report_denied


def test_physio_dimension_counted_when_authorized_and_reliable():
    """防回归：授权 + 可靠（R≥0.40）时生理维度按权重上限 W_MAX 计入。"""
    report_auth, _ = _run_physio_pair()

    assert report_auth.physio is not None
    assert report_auth.physio.available is True
    assert report_auth.physio.reason is None
    assert report_auth.physio.n_valid >= 8
    assert PHYSIO_DIM in report_auth.dims
    assert PHYSIO_DIM in report_auth.effective_weights
    assert report_auth.effective_weights[PHYSIO_DIM] == pytest.approx(W_MAX)  # 8%
    assert report_auth.dims[PHYSIO_DIM] == pytest.approx(round(report_auth.physio.score, 2))
    assert report_auth.physio.weight_applied == pytest.approx(W_MAX)


def test_physio_off_does_not_lower_total_score():
    """防回归（验收红线）：关掉生理维度**不压低总分**——六维值不变、权重和仍为 1。"""
    report_auth, report_denied = _run_physio_pair()

    dims_auth = {k: report_auth.dims[k] for k in DIMENSIONS}
    dims_denied = {k: report_denied.dims[k] for k in DIMENSIONS}
    assert dims_auth == dims_denied, "六维分值必须与生理模块开关无关"

    assert report_denied.physio is not None
    assert report_denied.physio.available is False
    assert PHYSIO_DIM not in report_denied.dims
    assert PHYSIO_DIM not in report_denied.effective_weights
    assert set(report_denied.effective_weights) == set(DIMENSIONS)
    assert sum(report_denied.effective_weights.values()) == pytest.approx(1.0)

    # 未授权那次的总分 == 六维按生效权重的加权和（用报告自己的数据复算一遍）
    recomputed = sum(report_denied.dims[k] * report_denied.effective_weights[k] for k in DIMENSIONS)
    assert report_denied.score == pytest.approx(round(recomputed, 2), abs=1e-2)


def test_effective_weights_always_sum_to_one():
    """防回归：无论生理维度计入与否，生效权重之和恒为 1.0（重分配守恒）。"""
    report_auth, report_denied = _run_physio_pair()
    assert sum(report_auth.effective_weights.values()) == pytest.approx(1.0)
    assert sum(report_denied.effective_weights.values()) == pytest.approx(1.0)

    # 生理不可用时，那 8% 必须真的还给了其余维度（不是凭空消失）
    six_sum_auth = sum(report_auth.effective_weights[k] for k in DIMENSIONS)
    six_sum_denied = sum(report_denied.effective_weights[k] for k in DIMENSIONS)
    assert six_sum_auth == pytest.approx(1.0 - W_MAX)
    assert six_sum_denied == pytest.approx(1.0)


# --------------------------------------------------------------------------
# 8/9. 跳过 / 重听 / 延长 / TTS 失败
# --------------------------------------------------------------------------


def test_skip_repeat_and_extend_time():
    """防回归：跳过的题不再出现、重听出同一题、延长时限不改变状态语义。"""
    env = _env(questions=CORE_QUESTIONS[:3])
    env.svc.create()
    env.svc.record_consent(Consent(base=True))
    env.svc.greeting_done()

    assert env.svc.ask_next() == CORE_QUESTIONS[0]
    assert env.svc.skip() is State.ASKING  # 跳过的题不再出现

    assert env.svc.ask_next() == CORE_QUESTIONS[1]
    assert env.svc.repeat_question() is State.ASKING
    assert env.svc.ask_next() == CORE_QUESTIONS[1], "重听后必须呈现同一题"

    extend_events_before = _event_names(env.repo, SESSION_ID).count("extend_time")
    assert env.svc.extend_time() is State.LISTENING, "延长时限是自环，不改状态"
    assert _event_names(env.repo, SESSION_ID).count("extend_time") == extend_events_before + 1

    asyncio.run(env.svc.submit_answer("作答"))
    assert env.svc.ask_next() == CORE_QUESTIONS[2], "跳过的 Q1 不应再出现"
    asyncio.run(env.svc.submit_answer("作答"))
    assert env.svc.ask_next() is None  # 题尽 → CANDIDATE_QA

    report = _finish(env.svc)
    assert _core_questions_from_log(env.repo, SESSION_ID) == [
        CORE_QUESTIONS[0], CORE_QUESTIONS[1], CORE_QUESTIONS[1], CORE_QUESTIONS[2],
    ]
    assert report.n_scored_turns == 2  # Q1 被跳过（不计分、不算一轮），只有 Q2/Q3 两轮真实评估


def test_tts_failure_degrades_to_text_presentation():
    """防回归：TTS 失败时题目照常呈现、流程继续，绝不卡死在出题环节。"""
    env = _env(questions=CORE_QUESTIONS[:2])
    env.svc.create()
    env.svc.record_consent(Consent(base=True))
    env.svc.greeting_done()

    question = env.svc.ask_next(tts_ok=False)
    assert question == CORE_QUESTIONS[0], "无声也要把题目文本给到候选人"
    assert env.svc.session.state is State.LISTENING
    assert env.svc.last_question == CORE_QUESTIONS[0]
    asyncio.run(env.svc.submit_answer("作答一"))

    assert env.svc.ask_next(tts_ok=False) == CORE_QUESTIONS[1]
    asyncio.run(env.svc.submit_answer("作答二"))
    assert env.svc.ask_next(tts_ok=False) is None

    report = _finish(env.svc)
    assert report.available is True
    assert report.n_scored_turns == 2
    assert _event_names(env.repo, SESSION_ID).count("tts_failed") == 2
    assert _event_names(env.repo, SESSION_ID).count("tts_done") == 0


def test_last_question_is_empty_before_first_question():
    """防回归：还没出题时 `last_question` 是空串，不能用它当"当前题目"去评分。"""
    env = _env()
    assert env.svc.last_question == ""
    env.svc.create()
    env.svc.record_consent(Consent(base=True))
    env.svc.greeting_done()
    assert env.svc.last_question == ""
    env.svc.ask_next()
    assert env.svc.last_question == CORE_QUESTIONS[0]
