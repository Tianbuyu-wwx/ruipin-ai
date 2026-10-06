"""领域层会话聚合的穷举/全覆盖测试（session.py）。

核心红线：
1. `state` 是**只读 property**，修改状态的唯一途径是 `apply(event)`；
   直接赋值 `session.state = ...` 必须在语言层面就失败（AttributeError）。
2. 构造时 `state` 必须是 `State` 枚举，传字符串抛 TypeError（不能靠 str-Enum 的隐式兼容蒙混过关）。
3. 非法事件**不得留下半个副作用**（状态不变、日志不增长、seq 不前进）。

以上都用 ALL_STATES × ALL_EVENTS 的穷举来守护，而不是挑几条手写。

穷举域与 machine 一致：15 × 23 = 345 组合，合法 60 个、非法 285 个。
"""

from __future__ import annotations

import dataclasses
from itertools import product

import pytest

from ruipin.domain.errors import InvalidTransition
from ruipin.domain.machine import TERMINAL, TRANSITIONS, WILDCARD, can
from ruipin.domain.session import Session, StateEvent
from ruipin.domain.states import ALL_EVENTS, ALL_STATES, Event, State

pytestmark = pytest.mark.unit

ALL_PAIRS = list(product(ALL_STATES, ALL_EVENTS))
LEGAL_PAIRS = [(s, e) for (s, e) in ALL_PAIRS if (s, e) in TRANSITIONS or e in WILDCARD]
ILLEGAL_PAIRS = [(s, e) for (s, e) in ALL_PAIRS if (s, e) not in LEGAL_PAIRS]
NON_TERMINAL = tuple(s for s in ALL_STATES if s not in TERMINAL)

SCORING_EVENTS = (Event.EVAL_DONE, Event.EVAL_DEGRADED)

# 真实面试主路径（18 个事件）
MAIN_PATH = (
    Event.CREATE,
    Event.CONSENT_OK,
    Event.GREETING_DONE,
    # 第 1 题
    Event.TTS_DONE,
    Event.ANSWER_COMMIT,
    Event.EVAL_DONE,
    # 第 2 题（带追问）
    Event.FOLLOWUP_DECISION,
    Event.TTS_DONE,
    Event.ANSWER_COMMIT,
    Event.EVAL_DEGRADED,
    # 第 3 题（带延长时限）
    Event.TTS_DONE,
    Event.EXTEND_TIME,
    Event.ANSWER_COMMIT,
    Event.EVAL_DONE,
    # 收尾
    Event.NO_MORE_QUESTIONS,
    Event.QA_DONE,
    Event.CLOSING_DONE,
    Event.REPORT_DONE,
)

# 缓冲回合：ASKING → BUFFER → BUFFER_LISTENING → ASKING（全程无评分事件）
BUFFER_ROUND = (
    Event.INSERT_BUFFER,
    Event.TTS_DONE,
    Event.BUFFER_DONE,
)


@pytest.fixture
def sess() -> Session:
    """一个全新的 IDLE 会话（名字避开 conftest 里的 session fixture）。"""
    return Session("s-test")


@pytest.fixture
def make_sess():
    """会话工厂：make_sess() / make_sess(State.ASKING)。"""

    def _make(state: State = State.IDLE, session_id: str = "s-test") -> Session:
        return Session(session_id, state)

    return _make


def drive(target: Session, *events: Event) -> Session:
    """按顺序施加事件，便于路径级断言。"""
    for e in events:
        target.apply(e)
    return target


def seqs(target: Session) -> list[int]:
    return [e.seq for e in target.events]


# ==========================================================================
# 1. 构造：state 必须是 State 枚举；state 只读
# ==========================================================================


def test_initial_session_shape(sess):
    """防回归：构造后状态/日志/序号初值被改动（如 seq 从 1 起变成从 0 起）。"""
    assert sess.session_id == "s-test"
    assert sess.state is State.IDLE
    assert sess.events == []
    assert sess.turns == []
    assert sess.meta == {}
    assert sess.is_terminal is False
    assert sess.history() == []
    assert seqs(sess) == []


def test_rejects_string_state():
    """防回归：State 是 str-Enum，传 "idle" 这种裸字符串很容易被误当成合法值；
    一旦放行，后续 transition() 查表就会 miss（"idle" 不是 State.IDLE 的同一对象），
    报错点离现场很远。构造期就拦住。"""
    with pytest.raises(TypeError) as exc:
        Session("x", "idle")
    assert "State" in str(exc.value)
    with pytest.raises(TypeError):
        Session("x", "buffer_listening")
    with pytest.raises(TypeError):
        Session("x", 1)
    with pytest.raises(TypeError):
        Session("x", None)


def test_accepts_state_enum():
    """防回归：正常传枚举被误拒（TypeError 写得太宽，把合法用法也拦了）。"""
    target = Session("x", State.IDLE)
    assert target.state is State.IDLE
    for state in ALL_STATES:
        assert Session("x", state).state is state
    # 显式关键字参数同样可用
    assert Session(session_id="x", state=State.ASKING).state is State.ASKING


def test_state_is_read_only_property():
    """防回归：state 变回可写属性 —— 现系统 app.py 到处直接改 session.status，
    正是这种写法让状态机形同虚设。把禁止写死在语言层面，比写在注释里可靠。"""
    target = Session("s-ro", State.IDLE)
    with pytest.raises(AttributeError):
        target.state = State.ASKING
    with pytest.raises(AttributeError):
        target.state = State.COMPLETED
    assert target.state is State.IDLE, "赋值失败但状态却被改了"
    # 只有 apply 能改
    assert target.apply(Event.CREATE) is State.SETUP
    assert target.state is State.SETUP


def test_state_property_is_defined_without_setter():
    """防回归：state 被改回普通实例属性（`self.state = ...` 直接可写），
    或给 property 加了 setter —— 两种写法都会让"唯一写入口是 apply"失效。"""
    prop = Session.state
    assert isinstance(prop, property)
    assert prop.fset is None, "state property 不该有 setter"
    assert prop.fdel is None
    # 实例字典里不存 state（存在 _state 里），因此 `s.__dict__["state"] = x` 也无效
    target = Session("s-internal", State.IDLE)
    assert "state" not in target.__dict__
    assert "_state" in target.__dict__
    with pytest.raises(AttributeError):
        target.state = State.ASKING


# ==========================================================================
# 2. apply 是唯一写入口
# ==========================================================================


def test_apply_changes_state_and_logs_event(sess):
    """防回归：apply 改了状态却不写日志（审计/重放失效）。"""
    new_state = sess.apply(Event.CREATE)
    assert new_state is State.SETUP
    assert sess.state is State.SETUP
    assert len(sess.events) == 1
    ev = sess.events[0]
    assert ev.seq == 1
    assert ev.from_state is State.IDLE
    assert ev.to_state is State.SETUP
    assert ev.event is Event.CREATE


@pytest.mark.property
def test_seq_increments_from_one_strictly(make_sess):
    """防回归：seq 出现跳号/重复（重放时无法按序对齐）。"""
    for state, event in LEGAL_PAIRS:
        if state in TERMINAL:
            continue
        target = make_sess(state)
        target.apply(event)
        assert seqs(target) == [1]
        assert target.events[0].seq == 1


@pytest.mark.parametrize(("state", "event"), LEGAL_PAIRS, ids=lambda v: v.name)
def test_apply_legal_updates_state_and_log(state, event):
    """防回归：穷举确认每个合法组合都能经 apply 落地（状态+日志+seq 三者一致）。"""
    target = Session("s-legal", state)
    new_state = target.apply(event)
    assert target.state is new_state
    assert len(target.events) == 1
    assert target.events[0].from_state is state
    assert target.events[0].to_state is new_state
    assert target.events[0].event is event


# ==========================================================================
# 3. apply 非法事件：抛错且不留半个副作用（红线）
# ==========================================================================


@pytest.mark.property
@pytest.mark.parametrize(("state", "event"), ILLEGAL_PAIRS, ids=lambda v: v.name)
def test_apply_illegal_raises_without_partial_effect(state, event):
    """防回归：非法事件先改了状态/写了半个日志再抛错，留下不可回放的脏状态。"""
    target = Session("s-illegal", state)
    payload_before = {"seed": "keep"}
    target.meta["k"] = "v"
    target.turns.append(payload_before)

    with pytest.raises(InvalidTransition) as excinfo:
        target.apply(event)

    assert excinfo.value.state is state
    assert excinfo.value.event is event
    assert target.state is state, "状态被非法事件改写了"
    assert target.events == [], "非法事件写入了事件日志"
    assert target.meta == {"k": "v"}
    assert target.turns == [payload_before]
    assert target._seq == 0, "非法事件推进了 seq"


@pytest.mark.property
@pytest.mark.parametrize(("state", "event"), ILLEGAL_PAIRS, ids=lambda v: v.name)
def test_try_apply_illegal_returns_none_without_effect(state, event):
    """防回归：try_apply 非法时静默产生副作用（上层以为无事发生实则状态已污染）。"""
    target = Session("s-try", state)
    assert target.try_apply(event) is None
    assert target.state is state
    assert target.events == []
    assert target._seq == 0
    assert target.history() == []


@pytest.mark.parametrize(("state", "event"), LEGAL_PAIRS, ids=lambda v: v.name)
def test_try_apply_legal_is_equivalent_to_apply(state, event):
    """防回归：try_apply 与 apply 行为不一致（两条调用路径出现分叉）。"""
    via_apply = Session("s-a", state)
    via_try = Session("s-b", state)
    a = via_apply.apply(event, {"x": 1})
    b = via_try.try_apply(event, {"x": 1})
    assert b == a
    assert via_try.state == via_apply.state
    assert via_try.history() == via_apply.history()
    assert seqs(via_try) == seqs(via_apply) == [1]
    assert via_try.events[0].payload == {"x": 1}


def test_apply_does_not_swallow_other_errors(sess):
    """防回归：try_apply 把非 InvalidTransition 的异常也吞成 None（掩盖真实故障）。"""
    drive(sess, Event.CREATE, Event.CONSENT_OK, Event.GREETING_DONE)
    assert sess.try_apply(Event.ANSWER_COMMIT) is None
    with pytest.raises(InvalidTransition):
        sess.apply(Event.ANSWER_COMMIT)
    # 缓冲作答阶段同样不能接受 ANSWER_COMMIT
    drive(sess, Event.INSERT_BUFFER, Event.TTS_DONE)
    assert sess.state is State.BUFFER_LISTENING
    assert sess.try_apply(Event.ANSWER_COMMIT) is None
    assert sess.try_apply(Event.EVAL_DONE) is None


# ==========================================================================
# 4. 查询接口：history / count_event / visited / is_terminal
# ==========================================================================


def test_history_is_ordered_triples(sess):
    """防回归：history 顺序错乱或三元组字段错位（from/event/to）。"""
    drive(sess, Event.CREATE, Event.CONSENT_OK, Event.GREETING_DONE)
    assert sess.history() == [
        (State.IDLE, Event.CREATE, State.SETUP),
        (State.SETUP, Event.CONSENT_OK, State.GREETING),
        (State.GREETING, Event.GREETING_DONE, State.ASKING),
    ]


def test_count_event_counts_repeats_and_absents(sess):
    """防回归：同一事件多次出现只统计一次（漏计循环答题次数）。"""
    drive(
        sess,
        Event.CREATE,
        Event.CONSENT_OK,
        Event.GREETING_DONE,
        Event.TTS_DONE,
        Event.ANSWER_COMMIT,
        Event.EVAL_DONE,
        Event.TTS_DONE,
        Event.ANSWER_COMMIT,
        Event.EVAL_DONE,
    )
    assert sess.count_event(Event.TTS_DONE) == 2
    assert sess.count_event(Event.ANSWER_COMMIT) == 2
    assert sess.count_event(Event.EVAL_DONE) == 2
    assert sess.count_event(Event.CREATE) == 1
    assert sess.count_event(Event.ABORT) == 0
    assert sess.count_event(Event.FATAL) == 0


def test_visited_distinguishes_seen_and_unseen(sess):
    """防回归：visited 对"经过但未停留"的状态误报 True（只看 to_state）。"""
    drive(sess, Event.CREATE, Event.CONSENT_OK)
    assert sess.visited(State.SETUP) is True
    assert sess.visited(State.GREETING) is True
    assert sess.visited(State.IDLE) is False, "IDLE 是 from_state 而非 to_state"
    assert sess.visited(State.ASKING) is False
    assert sess.visited(State.COMPLETED) is False
    sess.apply(Event.GREETING_DONE)
    assert sess.visited(State.ASKING) is True


@pytest.mark.parametrize("state", list(NON_TERMINAL), ids=lambda v: v.name)
def test_is_terminal_false_for_non_terminal(state):
    """防回归：is_terminal 误判中间状态为已结束（提前生成报告）。"""
    assert Session("s-t", state).is_terminal is False


@pytest.mark.parametrize("state", list(NON_TERMINAL), ids=lambda v: v.name)
def test_is_terminal_true_after_abort_from_any_state(state):
    """防回归：某些状态无法被判定为终态（会话永不结束）。"""
    target = Session("s-t", state)
    assert target.apply(Event.ABORT) is State.ABORTED
    assert target.is_terminal is True


@pytest.mark.parametrize("state", list(NON_TERMINAL), ids=lambda v: v.name)
def test_fatal_reaches_failed_from_any_state(state):
    """防回归：FATAL 未能把会话推入 FAILED（故障被静默忽略）。"""
    target = Session("s-f", state)
    assert target.apply(Event.FATAL) is State.FAILED
    assert target.is_terminal is True
    assert target.visited(State.FAILED) is True


def test_is_terminal_true_for_completed(make_sess):
    """防回归：COMPLETED 未被识别为终态（结束后仍可继续答题）。"""
    target = make_sess(State.REPORTING)
    assert target.is_terminal is False
    assert target.apply(Event.REPORT_DONE) is State.COMPLETED
    assert target.is_terminal is True


# ==========================================================================
# 5. 事件日志 append-only / frozen
# ==========================================================================


def test_event_log_is_append_only(sess):
    """防回归：历史事件被就地改写（重放结果与当时不一致）。"""
    drive(sess, Event.CREATE, Event.CONSENT_OK)
    snapshot = list(sess.events)
    first_event = sess.events[0]
    drive(sess, Event.GREETING_DONE, Event.TTS_DONE)
    assert sess.events[:2] == snapshot
    assert sess.events[0] is first_event
    assert first_event.seq == 1
    assert first_event.to_state is State.SETUP
    assert len(sess.events) == 4


def test_seq_is_strictly_monotonic(sess):
    """防回归：seq 不严格递增（重放时出现乱序/重复）。"""
    drive(sess, *MAIN_PATH)
    got = seqs(sess)
    assert got == list(range(1, len(MAIN_PATH) + 1))
    assert all(b > a for a, b in zip(got, got[1:]))


def test_timestamp_is_float_and_positive(sess):
    """防回归：ts 字段缺失或非 float（断线重连重放时无法排序）。
    不依赖真实时钟的具体值，只断言类型与正性，保证确定性。"""
    drive(sess, Event.CREATE, Event.CONSENT_OK)
    for ev in sess.events:
        assert isinstance(ev.ts, float)
        assert ev.ts > 0


def test_state_event_is_frozen(sess):
    """防回归：StateEvent 变可写（审计日志可被事后篡改）。"""
    sess.apply(Event.CREATE)
    ev = sess.events[0]
    for attr, value in (
        ("seq", 99),
        ("to_state", State.ABORTED),
        ("event", Event.ABORT),
        ("payload", {}),
    ):
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(ev, attr, value)
    assert ev.seq == 1
    assert ev.to_state is State.SETUP


def test_state_event_equality_and_replace(sess):
    """防回归：StateEvent 值语义被破坏（无法用于 Golden Master 对比）。"""
    sess.apply(Event.CREATE)
    ev = sess.events[0]
    clone = dataclasses.replace(ev, payload={"a": 1})
    assert clone.seq == ev.seq
    assert clone.to_state is ev.to_state
    assert clone.payload == {"a": 1}
    assert ev.payload == {}


# ==========================================================================
# 6. payload / turns / meta
# ==========================================================================


def test_payload_defaults_to_empty_dict(sess):
    """防回归：缺省 payload 变成 None，下游 payload["k"] 直接崩。"""
    sess.apply(Event.CREATE)
    assert sess.events[0].payload == {}
    sess.apply(Event.CONSENT_OK, {})
    assert sess.events[1].payload == {}


def test_payload_is_passed_through(sess):
    """防回归：payload 未透传（追问理由、评分明细丢失）。"""
    payload = {"question_id": "q-7", "audio_ms": 4200}
    sess.apply(Event.CREATE, payload)
    assert sess.events[0].payload == {"question_id": "q-7", "audio_ms": 4200}
    assert sess.events[0].payload is payload


def test_payload_dicts_are_not_shared_between_events(sess):
    """防回归：可变默认值陷阱 —— 两个事件的 payload 指向同一个 dict。"""
    sess.apply(Event.CREATE)
    sess.apply(Event.CONSENT_OK)
    first, second = sess.events[0], sess.events[1]
    assert first.payload is not second.payload
    first.payload["k"] = 1
    assert second.payload == {}


def test_turns_and_meta_are_per_instance(make_sess):
    """防回归：turns/meta 被类级共享（一个会话的数据串到另一个会话）。"""
    a = make_sess(State.IDLE, "s-a")
    b = make_sess(State.IDLE, "s-b")
    a.turns.append({"turn": 1})
    a.meta["mode"] = "practice"
    assert b.turns == []
    assert b.meta == {}
    assert a.turns is not b.turns
    assert a.meta is not b.meta


def test_events_list_is_per_instance(make_sess):
    """防回归：events 列表被类级共享（日志跨会话串台）。"""
    a = make_sess(State.IDLE, "s-a")
    b = make_sess(State.IDLE, "s-b")
    a.apply(Event.CREATE)
    assert b.events == []
    assert b.state is State.IDLE


# ==========================================================================
# 7. 业务路径级回归
# ==========================================================================


def test_full_interview_main_path(sess):
    """防回归：主干链路被改断，或结束后未进入终态。"""
    drive(sess, *MAIN_PATH)
    assert sess.state is State.COMPLETED
    assert sess.is_terminal is True
    assert sess.visited(State.COMPLETED) is True
    assert len(sess.events) == 18
    assert seqs(sess) == list(range(1, 19))
    assert sess.history()[0] == (State.IDLE, Event.CREATE, State.SETUP)
    assert sess.history()[-1] == (State.REPORTING, Event.REPORT_DONE, State.COMPLETED)
    assert sess.count_event(Event.ANSWER_COMMIT) == 3
    assert sess.count_event(Event.EVAL_DONE) == 2
    assert sess.count_event(Event.EVAL_DEGRADED) == 1
    assert sess.visited(State.FOLLOWUP) is True
    assert sess.visited(State.CANDIDATE_QA) is True


def test_after_terminal_only_wildcard_allowed(sess):
    """防回归：会话结束后仍能被普通事件激活（重复出报告/重复计分）。"""
    assert sess.apply(Event.ABORT) is State.ABORTED
    log_len = len(sess.events)
    for event in ALL_EVENTS:
        if event in WILDCARD:
            continue
        with pytest.raises(InvalidTransition):
            sess.apply(event)
        assert sess.state is State.ABORTED
        assert len(sess.events) == log_len
    # 通配符仍可用（逃生阀）
    assert sess.apply(Event.FATAL) is State.FAILED
    assert sess.is_terminal is True


def test_abort_from_mid_interview_keeps_history(sess):
    """防回归：中止时清空历史（无法审计"进行到哪一步被中止"）。"""
    drive(sess, Event.CREATE, Event.CONSENT_OK, Event.GREETING_DONE, Event.TTS_DONE)
    assert sess.apply(Event.ABORT) is State.ABORTED
    assert len(sess.events) == 5
    assert sess.visited(State.LISTENING) is True
    assert sess.visited(State.COMPLETED) is False
    assert sess.count_event(Event.ABORT) == 1


def test_consent_denied_path(sess):
    """防回归：拒绝授权后仍继续（合规红线）。"""
    drive(sess, Event.CREATE, Event.CONSENT_DENIED)
    assert sess.state is State.ABORTED
    assert sess.is_terminal is True
    assert sess.visited(State.GREETING) is False
    assert sess.visited(State.ASKING) is False


def test_buffer_question_path_leaves_no_scoring_event(sess):
    """★ 防回归：缓冲题被计分 —— 断言整条缓冲路径
    ASKING →(INSERT_BUFFER) BUFFER →(TTS_DONE) BUFFER_LISTENING →(BUFFER_DONE) ASKING
    的事件日志里不含 EVAL_DONE / EVAL_DEGRADED（这是"不计分"的可审计证据）。"""
    drive(sess, Event.CREATE, Event.CONSENT_OK, Event.GREETING_DONE)
    assert sess.state is State.ASKING
    drive(sess, *BUFFER_ROUND)

    assert sess.state is State.ASKING
    assert sess.is_terminal is False
    fired = {e.event for e in sess.events}
    assert Event.EVAL_DONE not in fired
    assert Event.EVAL_DEGRADED not in fired
    assert sess.count_event(Event.EVAL_DONE) == 0
    assert sess.count_event(Event.EVAL_DEGRADED) == 0
    assert sess.count_event(Event.ANSWER_COMMIT) == 0
    assert sess.count_event(Event.BUFFER_DONE) == 1
    assert sess.visited(State.BUFFER) is True
    assert sess.visited(State.BUFFER_LISTENING) is True
    assert sess.visited(State.PROCESSING) is False, "缓冲回合进入了评分状态"
    assert sess.history()[-1] == (State.BUFFER_LISTENING, Event.BUFFER_DONE, State.ASKING)
    assert sess.history()[-3:] == [
        (State.ASKING, Event.INSERT_BUFFER, State.BUFFER),
        (State.BUFFER, Event.TTS_DONE, State.BUFFER_LISTENING),
        (State.BUFFER_LISTENING, Event.BUFFER_DONE, State.ASKING),
    ]


def test_buffer_round_variants_all_scoring_free(make_sess):
    """★ 防回归：缓冲回合的每一种收尾方式（答完 / 跳过 / 静默超时 / TTS 降级）
    都不产生评分事件。只测一条路径会漏掉后面加的旁路。"""
    variants = (
        (Event.INSERT_BUFFER, Event.TTS_DONE, Event.BUFFER_DONE),
        (Event.INSERT_BUFFER, Event.TTS_DONE, Event.SKIP),
        (Event.INSERT_BUFFER, Event.TTS_DONE, Event.SILENCE_TIMEOUT),
        (Event.INSERT_BUFFER, Event.TTS_FAILED, Event.BUFFER_DONE),
        (Event.INSERT_BUFFER, Event.TTS_DONE, Event.EXTEND_TIME, Event.BUFFER_DONE),
        (Event.INSERT_BUFFER, Event.TTS_DONE, Event.REPEAT_QUESTION, Event.TTS_DONE,
         Event.BUFFER_DONE),
    )
    for i, variant in enumerate(variants):
        target = make_sess(State.ASKING, f"s-buf-{i}")
        drive(target, *variant)
        assert target.state is State.ASKING, f"变体 {variant} 未回到 ASKING"
        for ev in SCORING_EVENTS:
            assert target.count_event(ev) == 0, f"变体 {variant} 产生了 {ev}"
        assert target.visited(State.PROCESSING) is False


def test_buffer_then_normal_question_mixes_correctly(sess):
    """防回归：缓冲题之后的正常题被误判成缓冲（计分丢失）；反之亦然。"""
    drive(sess, Event.CREATE, Event.CONSENT_OK, Event.GREETING_DONE)
    # 缓冲回合（不计分）
    drive(sess, *BUFFER_ROUND)
    # 正常回合（计分）
    drive(sess, Event.TTS_DONE, Event.ANSWER_COMMIT, Event.EVAL_DONE)
    assert sess.state is State.ASKING
    assert sess.count_event(Event.BUFFER_DONE) == 1
    assert sess.count_event(Event.EVAL_DONE) == 1
    assert sess.count_event(Event.ANSWER_COMMIT) == 1
    assert sess.visited(State.PROCESSING) is True
    assert sess.visited(State.BUFFER_LISTENING) is True


def test_skip_and_repeat_loop_without_scoring(sess):
    """防回归：跳过/重听被当成一次作答而进入评分。"""
    drive(sess, Event.CREATE, Event.CONSENT_OK, Event.GREETING_DONE)
    drive(sess, Event.TTS_DONE, Event.REPEAT_QUESTION, Event.TTS_DONE, Event.SKIP)
    assert sess.state is State.ASKING
    assert sess.count_event(Event.EVAL_DONE) == 0
    assert sess.count_event(Event.ANSWER_COMMIT) == 0
    assert sess.count_event(Event.REPEAT_QUESTION) == 1


def test_silence_timeout_still_requires_evaluation(sess):
    """防回归：静默超时绕过评分直接下一题（空白答案也能拿分）。"""
    drive(
        sess,
        Event.CREATE,
        Event.CONSENT_OK,
        Event.GREETING_DONE,
        Event.TTS_DONE,
        Event.SILENCE_TIMEOUT,
    )
    assert sess.state is State.PROCESSING
    assert sess.apply(Event.EVAL_DEGRADED) is State.ASKING


def test_session_can_be_constructed_at_any_state(make_sess):
    """防回归：从非 IDLE 状态恢复会话（断线重连）时构造失败。"""
    for state in ALL_STATES:
        target = make_sess(state, "s-resume")
        assert target.state is state
        assert target.events == []
        assert target.is_terminal is (state in TERMINAL)
        assert target._seq == 0


def test_can_is_consistent_with_session_apply():
    """防回归：上层用 can() 预判后调用 apply 却抛错（两套判定不一致）。"""
    for state, event in ALL_PAIRS:
        target = Session("s-c", state)
        if can(state, event):
            assert isinstance(target.apply(event), State)
        else:
            with pytest.raises(InvalidTransition):
                target.apply(event)
            assert target.state is state
