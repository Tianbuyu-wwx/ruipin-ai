"""领域层状态机的穷举/全覆盖测试（machine.py）。

覆盖策略：以 ALL_STATES × ALL_EVENTS = 15 × 23 = 345 个组合为穷举域
（合法 60 个 = 30 条显式边 + 15 状态 × 2 个通配符；非法 285 个），
先跑一遍拿到真实集合再写死断言。重点守护三条硬规则：

① 终态封闭（除 ABORT / FATAL 通配符逃生阀）；
② 无死状态、无孤立状态（每个非终态都有业务出边，每个状态都能从 IDLE 走到）；
③ ★ 缓冲回合内不存在评分时机：缓冲链是
   ASKING --INSERT_BUFFER--> BUFFER --TTS_DONE--> BUFFER_LISTENING --BUFFER_DONE--> ASKING，
   全程不经过 PROCESSING，因此 EVAL_DONE / EVAL_DEGRADED 无处可发。
   这条不是"靠调用方自觉"，而是结构保证 —— 本文件用**通用性质测试**守住它。

本文件不修改任何源码；如发现设计缺口，只在断言里如实记录当前行为。
"""

from __future__ import annotations

from itertools import product

import pytest

from ruipin.domain.errors import InvalidTransition
from ruipin.domain.machine import (
    TERMINAL,
    TRANSITIONS,
    WILDCARD,
    allowed_events,
    can,
    reachable_from,
    transition,
)
from ruipin.domain.states import ALL_EVENTS, ALL_STATES, Event, State

pytestmark = pytest.mark.unit

# ---------- 穷举域 ----------

ALL_PAIRS = list(product(ALL_STATES, ALL_EVENTS))
LEGAL_PAIRS = [(s, e) for (s, e) in ALL_PAIRS if (s, e) in TRANSITIONS or e in WILDCARD]
ILLEGAL_PAIRS = [(s, e) for (s, e) in ALL_PAIRS if (s, e) not in LEGAL_PAIRS]

NON_TERMINAL = tuple(s for s in ALL_STATES if s not in TERMINAL)

# 评分事件：一次"给分"必须经由其中之一。缓冲链路上它们必须无处可发。
SCORING_EVENTS = (Event.EVAL_DONE, Event.EVAL_DEGRADED)


# ---------- 图论小工具（供性质测试使用） ----------


def successors(state: State) -> dict[Event, State]:
    """某状态下所有合法出边的 {事件: 目标状态}。"""
    return {e: transition(state, e) for e in allowed_events(state)}


def predecessors(target: State) -> frozenset[State]:
    """所有能一步走到 target 的源状态集合。"""
    return frozenset(
        s for s in ALL_STATES if any(nxt is target for nxt in successors(s).values())
    )


def states_allowing(event: Event) -> frozenset[State]:
    """所有允许该事件发生的状态集合。"""
    return frozenset(s for s in ALL_STATES if can(s, event))


def reachable_avoiding(start: State, blocked) -> frozenset[State]:
    """从 start 出发、**不进入** blocked 中任何状态即可到达的状态集合（含自身）。

    与 reachable_from 的区别：一旦某条边的目标落在 blocked 里，就整条路剪掉，
    而不是穿过去继续走。这正是"抵达 ASKING 之前"这种时段性质的表达。
    """
    blocked = frozenset(blocked)
    assert start not in blocked, "起点自身被屏蔽则结果无意义"
    seen = {start}
    frontier = [start]
    while frontier:
        cur = frontier.pop()
        for e in ALL_EVENTS:
            if not can(cur, e):
                continue
            nxt = transition(cur, e)
            if nxt in blocked or nxt in seen:
                continue
            seen.add(nxt)
            frontier.append(nxt)
    return frozenset(seen)


def paths_until(start: State, stop) -> list[list[tuple[State, Event, State]]]:
    """从 start 出发、首次抵达 stop 中任一状态即停的所有"简单路径"。

    只走不重复状态的路径，因此环路（EXTEND_TIME 自环、REPEAT_QUESTION 回 BUFFER）
    不会导致无限展开，枚举必然终止。无法继续延伸且未抵达 stop 的前缀（如走进终态）
    也会原样记录，保证不漏掉任何一条逃逸路径。
    """
    stop = frozenset(stop)
    collected: list[list[tuple[State, Event, State]]] = []

    def walk(cur: State, visited: frozenset, path: list) -> None:
        if cur in stop:
            collected.append(list(path))
            return
        extended = False
        for e in ALL_EVENTS:
            if not can(cur, e):
                continue
            nxt = transition(cur, e)
            if nxt in visited:
                continue
            extended = True
            path.append((cur, e, nxt))
            walk(nxt, visited | {nxt}, path)
            path.pop()
        if not extended:
            collected.append(list(path))  # 死路：走到终态或再也走不动

    walk(start, frozenset({start}), [])
    return collected


# ==========================================================================
# 1. 表本身的结构性校验（防止枚举改名/新增状态后转换表没跟着改）
# ==========================================================================


def test_transition_table_keys_are_real_enum_members():
    """防回归：转换表里出现拼写错误/已删除的枚举成员时立刻报错（否则该边永远走不到）。"""
    for st, ev in TRANSITIONS:
        assert isinstance(st, State), f"非法 state: {st!r}"
        assert isinstance(ev, Event), f"非法 event: {ev!r}"
        assert st in ALL_STATES
        assert ev in ALL_EVENTS
    for target in TRANSITIONS.values():
        assert isinstance(target, State)
        assert target in ALL_STATES


def test_state_and_event_counts_are_pinned():
    """防回归：新增/删除状态或事件时，本文件的穷举域断言会静默失效（组合数漂移却没人发现）。
    实测：15 个状态 × 23 个事件 = 345 组合，其中 60 个合法。"""
    assert len(ALL_STATES) == 15
    assert len(ALL_EVENTS) == 23
    assert len(ALL_PAIRS) == 345
    assert len(LEGAL_PAIRS) == 60
    assert len(ILLEGAL_PAIRS) == 285
    assert len(TRANSITIONS) == 30
    assert len(LEGAL_PAIRS) == len(TRANSITIONS) + len(ALL_STATES) * len(WILDCARD)


def test_buffer_listening_state_is_declared_and_wired():
    """防回归：BUFFER_LISTENING 被删除或改名后，缓冲链退回"靠纪律保证不计分"。"""
    assert State.BUFFER_LISTENING.value == "buffer_listening"
    assert State.BUFFER_LISTENING in ALL_STATES
    # 必须是真的接进表里，而不是只声明不用（那样它是孤立状态）
    wired_as_source = [e for (s, e) in TRANSITIONS if s is State.BUFFER_LISTENING]
    wired_as_target = [s for (s, e), t in TRANSITIONS.items() if t is State.BUFFER_LISTENING]
    assert wired_as_source, "BUFFER_LISTENING 没有任何出边"
    assert wired_as_target, "BUFFER_LISTENING 没有任何入边"
    assert State.BUFFER_LISTENING in reachable_from(State.IDLE)


def test_terminal_members_are_real_states():
    """防回归：TERMINAL 里混入非 State（如裸字符串）会导致 is_terminal 永远为 False。"""
    assert isinstance(TERMINAL, frozenset)
    for st in TERMINAL:
        assert isinstance(st, State)
        assert st in ALL_STATES
    assert TERMINAL == frozenset({State.COMPLETED, State.ABORTED, State.FAILED})


def test_terminal_states_have_no_explicit_outgoing_edge():
    """防回归：终态若被加进 TRANSITIONS，就破坏了"终态不可迁出"的约束。"""
    for st, _ev in TRANSITIONS:
        assert st not in TERMINAL, f"终态 {st} 不应出现在 TRANSITIONS 中"


def test_wildcard_events_are_never_shadowed_by_table():
    """防回归：若某状态下 ABORT/FATAL 被写进 TRANSITIONS，通配符就会被静默覆盖。"""
    shadowed = [(s, e) for (s, e) in TRANSITIONS if e in WILDCARD]
    assert shadowed == []
    assert set(WILDCARD) == {Event.ABORT, Event.FATAL}


# ==========================================================================
# 2. 穷举：can() 与 transition() 必须完全一致
# ==========================================================================


@pytest.mark.property
@pytest.mark.parametrize(("state", "event"), ALL_PAIRS, ids=lambda v: v.name)
def test_can_matches_transition_for_every_pair(state, event):
    """防回归：can() 与 transition() 判定不一致（上层用 can 预判后 apply 却抛错）。"""
    result = can(state, event)
    assert isinstance(result, bool)
    assert result is ((state, event) in TRANSITIONS or event in WILDCARD)
    if result:
        assert isinstance(transition(state, event), State)
    else:
        with pytest.raises(InvalidTransition):
            transition(state, event)


@pytest.mark.property
@pytest.mark.parametrize(("state", "event"), ILLEGAL_PAIRS, ids=lambda v: v.name)
def test_illegal_pair_raises_invalid_transition(state, event):
    """防回归：非法组合被静默忽略（返回 None 或原状态）而不是抛错。"""
    with pytest.raises(InvalidTransition) as exc:
        transition(state, event)
    err = exc.value
    assert err.state is state
    assert err.event is event


@pytest.mark.property
@pytest.mark.parametrize(("state", "event"), LEGAL_PAIRS, ids=lambda v: v.name)
def test_legal_pair_returns_expected_state_member(state, event):
    """防回归：合法组合返回了字符串而非 State 枚举成员（下游 is 比较会失效）。"""
    new_state = transition(state, event)
    assert isinstance(new_state, State)
    assert new_state in ALL_STATES
    assert can(state, event) is True


def test_transition_and_can_never_raise_other_exception_types():
    """防回归：can() 只吞 InvalidTransition，其它异常（KeyError/TypeError）不能漏出去。"""
    for s, e in ALL_PAIRS:
        try:
            transition(s, e)
        except InvalidTransition:
            pass
        assert can(s, e) in (True, False)


# ==========================================================================
# 3. 通配符优先级（ABORT / FATAL 在任意状态都成立，含终态）
# ==========================================================================


def test_wildcard_holds_in_every_state():
    """防回归：通配符失效导致某个状态卡死无法中断（无法放弃面试）。"""
    for event, target in WILDCARD.items():
        for state in ALL_STATES:
            assert transition(state, event) is target, f"{state} --{event}--> ?"


def test_completed_abort_is_a_designed_escape_hatch():
    """防回归：COMPLETED --ABORT--> ABORTED 是设计内的逃生阀，不是 bug，别被"修好"。
    已完成的面试仍允许被运营侧强制中止/标记失败。"""
    assert transition(State.COMPLETED, Event.ABORT) is State.ABORTED
    assert transition(State.COMPLETED, Event.FATAL) is State.FAILED
    assert transition(State.ABORTED, Event.FATAL) is State.FAILED
    assert transition(State.FAILED, Event.ABORT) is State.ABORTED


def test_wildcard_event_is_legal_even_in_terminal_states():
    """防回归：终态把 ABORT/FATAL 也一并封死，导致无法记录事后失败。"""
    for state in TERMINAL:
        for event in WILDCARD:
            assert can(state, event) is True


def test_wildcard_reaches_aborted_and_failed_from_every_state():
    """防回归：新增状态后忘了通配符兜底（该状态无法被中止）。"""
    for state in ALL_STATES:
        assert State.ABORTED in reachable_from(state)
        assert State.FAILED in reachable_from(state)


# ==========================================================================
# 4. 终态封闭性 / 无死状态
# ==========================================================================


@pytest.mark.property
@pytest.mark.parametrize("state", list(TERMINAL), ids=lambda v: v.name)
def test_terminal_states_only_allow_wildcard(state):
    """防回归：终态出现新的出边，使"已结束"的会话被重新激活。"""
    allowed = set(allowed_events(state))
    assert allowed == set(WILDCARD)
    assert allowed <= set(WILDCARD)


@pytest.mark.property
@pytest.mark.parametrize("state", list(NON_TERMINAL), ids=lambda v: v.name)
def test_non_terminal_state_has_no_dead_end(state):
    """防回归：新增状态后忘记配出边，面试卡在该状态无法推进也无法放弃。"""
    allowed = allowed_events(state)
    assert isinstance(allowed, tuple)
    assert len(allowed) >= 1, f"{state} 是死状态，没有任何合法事件"
    # 至少有 ABORT/FATAL 兜底之外的一条业务出边
    assert set(allowed) - set(WILDCARD), f"{state} 只有通配符，没有业务出边"


def test_no_transition_leads_out_of_terminal_except_wildcard():
    """防回归：穷举确认没有任何 (终态, 非通配符事件) 组合是合法的。"""
    leaks = [
        (s, e)
        for s in TERMINAL
        for e in ALL_EVENTS
        if e not in WILDCARD and can(s, e)
    ]
    assert leaks == []


def test_every_state_has_a_predecessor_or_is_the_entry_point():
    """防回归：孤立状态（除 IDLE 这个唯一起点外没有任何入边，永远走不到）。"""
    for state in ALL_STATES:
        if state is State.IDLE:
            assert predecessors(state) == frozenset()
        else:
            assert predecessors(state), f"{state} 没有任何入边，是孤立状态"


# ==========================================================================
# 5. 可达性（真实算出的集合，写死断言）
# ==========================================================================


def test_reachable_from_idle_covers_every_state():
    """防回归：出现"声明了但走不到"的状态（现系统 CANDIDATE_QA 不可达就是这个缺陷）。
    实测 reachable_from(IDLE) == 全部 15 个状态。"""
    reach = reachable_from(State.IDLE)
    assert isinstance(reach, frozenset)
    assert reach == frozenset(ALL_STATES)
    assert len(reach) == 15
    assert State.BUFFER_LISTENING in reach


def test_every_non_terminal_state_is_reachable_from_idle():
    """防回归：孤立状态（没有任何路径能进入）。"""
    reach = reachable_from(State.IDLE)
    for state in NON_TERMINAL:
        assert state in reach, f"{state} 从 IDLE 不可达"
    assert State.CANDIDATE_QA in reach
    assert State.BUFFER in reach
    assert State.BUFFER_LISTENING in reach
    assert State.FOLLOWUP in reach


def test_reachable_from_buffer_and_buffer_listening():
    """防回归：缓冲链路被改断（缓冲题结束后回不到出题环节，面试卡死）。
    实测（两者相同，因为 BUFFER ↔ BUFFER_LISTENING 互相可达）：
      reachable_from(BUFFER) == reachable_from(BUFFER_LISTENING)
        == {BUFFER, BUFFER_LISTENING, ASKING, LISTENING, PROCESSING, FOLLOWUP,
            CANDIDATE_QA, CLOSING, REPORTING, COMPLETED, ABORTED, FAILED}  # 12 个
    """
    expected = frozenset(
        {
            State.BUFFER,
            State.BUFFER_LISTENING,
            State.ASKING,
            State.LISTENING,
            State.PROCESSING,
            State.FOLLOWUP,
            State.CANDIDATE_QA,
            State.CLOSING,
            State.REPORTING,
            State.COMPLETED,
            State.ABORTED,
            State.FAILED,
        }
    )
    assert reachable_from(State.BUFFER) == expected
    assert reachable_from(State.BUFFER_LISTENING) == expected
    assert len(expected) == 12
    # 缓冲回合结束后能回到正常出题流程（含计分路径）—— 这是"下一题照常计分"的保证
    assert State.ASKING in expected
    assert State.PROCESSING in expected
    # 但回到不了开场阶段
    assert State.IDLE not in expected
    assert State.SETUP not in expected
    assert State.GREETING not in expected


def test_reachable_from_asking_and_listening():
    """防回归：出题/作答环节的可达集合写死断言。
    实测 reachable_from(ASKING) == reachable_from(LISTENING) == reachable_from(PROCESSING)
        == reachable_from(FOLLOWUP) == reachable_from(BUFFER)（同一强连通块，12 个状态）。"""
    block = reachable_from(State.ASKING)
    for state in (
        State.LISTENING,
        State.PROCESSING,
        State.FOLLOWUP,
        State.BUFFER,
        State.BUFFER_LISTENING,
    ):
        assert reachable_from(state) == block, f"{state} 的可达集合与 ASKING 不一致"
    assert len(block) == 12


def test_reachable_from_terminal_states():
    """防回归：终态可达集合写死断言 —— 终态只能沿通配符互跳，回不到业务状态。
    实测：
      reachable_from(COMPLETED) == {COMPLETED, ABORTED, FAILED}
      reachable_from(ABORTED)   == {ABORTED, FAILED}
      reachable_from(FAILED)    == {ABORTED, FAILED}
    """
    assert reachable_from(State.COMPLETED) == frozenset(
        {State.COMPLETED, State.ABORTED, State.FAILED}
    )
    assert reachable_from(State.ABORTED) == frozenset({State.ABORTED, State.FAILED})
    assert reachable_from(State.FAILED) == frozenset({State.ABORTED, State.FAILED})
    # 终态不可回到任何业务状态
    for start in TERMINAL:
        assert reachable_from(start) <= TERMINAL | {start}
    assert State.ASKING not in reachable_from(State.ABORTED)
    assert State.COMPLETED not in reachable_from(State.ABORTED)
    assert State.BUFFER_LISTENING not in reachable_from(State.ABORTED)


def test_reachable_from_candidate_qa_and_later_stages():
    """防回归：收尾阶段意外回到出题环节（重复计分）。
    实测：
      reachable_from(CANDIDATE_QA) == {CANDIDATE_QA, CLOSING, REPORTING, COMPLETED, ABORTED, FAILED}
      reachable_from(CLOSING)      == {CLOSING, REPORTING, COMPLETED, ABORTED, FAILED}
      reachable_from(REPORTING)    == {REPORTING, COMPLETED, ABORTED, FAILED}
    """
    assert reachable_from(State.CANDIDATE_QA) == frozenset(
        {
            State.CANDIDATE_QA,
            State.CLOSING,
            State.REPORTING,
            State.COMPLETED,
            State.ABORTED,
            State.FAILED,
        }
    )
    assert reachable_from(State.CLOSING) == frozenset(
        {State.CLOSING, State.REPORTING, State.COMPLETED, State.ABORTED, State.FAILED}
    )
    assert reachable_from(State.REPORTING) == frozenset(
        {State.REPORTING, State.COMPLETED, State.ABORTED, State.FAILED}
    )
    # 收尾阶段回不到计分环节
    assert State.PROCESSING not in reachable_from(State.CANDIDATE_QA)
    assert State.BUFFER not in reachable_from(State.CANDIDATE_QA)


def test_reachable_from_includes_self_and_is_closed_under_steps():
    """防回归：reachable_from 漏算自身或漏算一步可达的状态。"""
    for state in ALL_STATES:
        reach = reachable_from(state)
        assert state in reach
        for event in ALL_EVENTS:
            if can(state, event):
                assert transition(state, event) in reach


# ==========================================================================
# 6. ★ 缓冲题不可计分（本项目最重要的硬规则）—— 结构保证的性质测试
# ==========================================================================

# 缓冲回合的"出口"：一旦回到 ASKING，缓冲回合就结束了，之后的计分与缓冲题无关。
BUFFER_ROUND_EXIT = frozenset({State.ASKING})


def test_buffer_round_has_no_scoring_moment_before_returning_to_asking():
    """★ 防回归（核心）：缓冲回合内出现评分时机，缓冲题被计分。

    不变量的正确表述**不是**"从 BUFFER 到不了 PROCESSING"（错：缓冲结束回到 ASKING
    后当然能到 PROCESSING），而是：

        从 BUFFER 出发、在抵达 ASKING 之前，所经过的任何状态都不允许 EVAL_DONE /
        EVAL_DEGRADED。

    这里用通用的"避开 ASKING 的可达性搜索"计算该集合，再对**每个**状态检查它是否
    允许评分事件。这样以后有人往缓冲链路上加一条边（比如
    BUFFER_LISTENING --ANSWER_COMMIT--> PROCESSING），测试会立刻失败，
    而不是靠维护一份硬编码的白名单。
    """
    during_buffer_round = reachable_avoiding(State.BUFFER, BUFFER_ROUND_EXIT)
    offenders = [
        (state, ev)
        for state in sorted(during_buffer_round, key=lambda s: s.name)
        for ev in SCORING_EVENTS
        if can(state, ev)
    ]
    assert offenders == [], f"缓冲回合内存在评分时机：{offenders}"
    # 更强的形式：连 PROCESSING（唯一允许评分的状态）都进不去
    assert State.PROCESSING not in during_buffer_round


def test_no_scoring_state_is_reachable_during_buffer_round():
    """★ 防回归：等价的第二种写法 —— 直接用"允许评分事件的状态集合"求交。
    两条写法互为交叉验证，避免其中一种实现写错时把缺陷放过去。"""
    during_buffer_round = reachable_avoiding(State.BUFFER, BUFFER_ROUND_EXIT)
    for ev in SCORING_EVENTS:
        assert during_buffer_round.isdisjoint(states_allowing(ev)), (
            f"{ev} 在缓冲回合内可发"
        )
    assert states_allowing(Event.EVAL_DONE) == frozenset({State.PROCESSING})
    assert states_allowing(Event.EVAL_DEGRADED) == frozenset({State.PROCESSING})


def test_every_path_from_buffer_to_asking_is_scoring_free():
    """★ 防回归：逐个路径枚举（而非只看可达集合）—— BUFFER 到 ASKING 的**每一条**
    简单路径上都不出现 EVAL_* 事件、也不经过 PROCESSING。
    可达集合只能证明"存在哪些状态"，路径级枚举还能证明"走到 ASKING 之前没有别的近路"。"""
    paths = paths_until(State.BUFFER, BUFFER_ROUND_EXIT)
    assert paths, "从 BUFFER 根本走不到 ASKING（缓冲回合无法结束）"
    reached_asking = [p for p in paths if p and p[-1][2] is State.ASKING]
    assert reached_asking, "没有任何一条路径能结束缓冲回合"

    for path in paths:
        for _src, ev, dst in path:
            assert ev not in SCORING_EVENTS, f"路径上出现评分事件 {ev}"
            assert dst is not State.PROCESSING, "路径上进入了 PROCESSING（评分状态）"
        # 每条路径要么以回到 ASKING 结束（正常结束），要么走进终态（ABORT/FATAL）
        if path:
            assert path[-1][2] is State.ASKING or path[-1][2] in TERMINAL

    # 结束缓冲回合至少要两步：BUFFER → BUFFER_LISTENING → ASKING
    assert all(len(p) >= 2 for p in reached_asking)
    # 且每一步都必须经过 BUFFER_LISTENING（不能一步从 BUFFER 直接跳回 ASKING）
    for path in reached_asking:
        assert any(dst is State.BUFFER_LISTENING for _src, _ev, dst in path)


def test_buffer_round_reachable_set_anchor():
    """★ 防回归：把上面性质测试算出的具体集合写死为回归锚点。
    实测：不经过 ASKING 时，从 BUFFER 可达集合 == {BUFFER, BUFFER_LISTENING, ABORTED, FAILED}。
    一旦有人往缓冲链路加边，本断言与上面的性质测试会同时失败。"""
    during = reachable_avoiding(State.BUFFER, BUFFER_ROUND_EXIT)
    assert during == frozenset(
        {State.BUFFER, State.BUFFER_LISTENING, State.ABORTED, State.FAILED}
    )
    assert len(during) == 4
    # BUFFER_LISTENING 出发同样是这个集合（它与 BUFFER 互相可达）
    assert reachable_avoiding(State.BUFFER_LISTENING, BUFFER_ROUND_EXIT) == during


def test_allowed_events_of_buffer_listening_anchor():
    """★ 防回归：BUFFER_LISTENING 的合法事件集合写死。
    实测 == {silence_timeout, skip, repeat_question, extend_time, buffer_done, abort, fatal}
    —— 注意**不含** answer_commit / eval_done / eval_degraded。"""
    allowed = set(allowed_events(State.BUFFER_LISTENING))
    assert allowed == {
        Event.SILENCE_TIMEOUT,
        Event.SKIP,
        Event.REPEAT_QUESTION,
        Event.EXTEND_TIME,
        Event.BUFFER_DONE,
        Event.ABORT,
        Event.FATAL,
    }
    assert len(allowed_events(State.BUFFER_LISTENING)) == 7
    for ev in SCORING_EVENTS:
        assert ev not in allowed
    assert Event.ANSWER_COMMIT not in allowed


def test_buffer_listening_rejects_scoring_and_commit_events():
    """★ 防回归：BUFFER_LISTENING 上允许 ANSWER_COMMIT / EVAL_* —— 缓冲答案被计分。"""
    for ev in (Event.ANSWER_COMMIT, Event.EVAL_DONE, Event.EVAL_DEGRADED):
        assert can(State.BUFFER_LISTENING, ev) is False, f"{ev} 在缓冲作答中竟然合法"
        with pytest.raises(InvalidTransition) as exc:
            transition(State.BUFFER_LISTENING, ev)
        assert exc.value.state is State.BUFFER_LISTENING
        assert exc.value.event is ev


def test_processing_rejects_buffer_done():
    """★ 防回归：(PROCESSING, BUFFER_DONE) 被重新加回表里 —— 会让"缓冲答案进入处理阶段"
    重新变成合法路径，缓冲题就可能被计分。源码已刻意删除该边。"""
    assert (State.PROCESSING, Event.BUFFER_DONE) not in TRANSITIONS
    assert can(State.PROCESSING, Event.BUFFER_DONE) is False
    with pytest.raises(InvalidTransition):
        transition(State.PROCESSING, Event.BUFFER_DONE)


def test_buffer_listening_done_returns_to_asking():
    """★ 防回归：缓冲回合的收尾事件 BUFFER_DONE 必须回到 ASKING（枚举身份用 is 断言）。"""
    assert transition(State.BUFFER_LISTENING, Event.BUFFER_DONE) is State.ASKING
    assert transition(State.BUFFER_LISTENING, Event.SILENCE_TIMEOUT) is State.ASKING
    assert transition(State.BUFFER_LISTENING, Event.SKIP) is State.ASKING
    assert transition(State.BUFFER_LISTENING, Event.EXTEND_TIME) is State.BUFFER_LISTENING
    assert transition(State.BUFFER_LISTENING, Event.REPEAT_QUESTION) is State.BUFFER


def test_buffer_state_only_leads_into_buffer_listening():
    """防回归：BUFFER 的出边被改成通往 LISTENING（缓冲作答混进正常作答，从而可计分）。"""
    assert transition(State.BUFFER, Event.TTS_DONE) is State.BUFFER_LISTENING
    assert transition(State.BUFFER, Event.TTS_FAILED) is State.BUFFER_LISTENING
    assert set(successors(State.BUFFER).values()) == {
        State.BUFFER_LISTENING,
        State.ABORTED,
        State.FAILED,
    }
    assert set(allowed_events(State.BUFFER)) == {
        Event.TTS_DONE,
        Event.TTS_FAILED,
        Event.ABORT,
        Event.FATAL,
    }
    for ev in SCORING_EVENTS + (Event.ANSWER_COMMIT,):
        assert can(State.BUFFER, ev) is False
    assert can(State.BUFFER, Event.SILENCE_TIMEOUT) is False
    assert can(State.BUFFER, Event.BUFFER_DONE) is False


def test_buffer_and_normal_question_diverge_at_listening():
    """防回归：缓冲题与正常题必须在 LISTENING / BUFFER_LISTENING 处就分流，
    而不是等到 PROCESSING 再靠事件区分 —— 分流越早，结构保证越强。"""
    assert transition(State.ASKING, Event.TTS_DONE) is State.LISTENING
    assert transition(State.BUFFER, Event.TTS_DONE) is State.BUFFER_LISTENING
    assert State.LISTENING is not State.BUFFER_LISTENING
    assert State.LISTENING.value != State.BUFFER_LISTENING.value
    # 正常作答才能进 PROCESSING；缓冲作答不能
    assert transition(State.LISTENING, Event.ANSWER_COMMIT) is State.PROCESSING
    assert can(State.BUFFER_LISTENING, Event.ANSWER_COMMIT) is False


def test_normal_round_scoring_path_still_works():
    """防回归：为了"缓冲不计分"把正常题的计分路径也一起砍掉（大面积漏计分）。"""
    assert transition(State.ASKING, Event.TTS_DONE) is State.LISTENING
    assert transition(State.LISTENING, Event.ANSWER_COMMIT) is State.PROCESSING
    assert transition(State.PROCESSING, Event.EVAL_DONE) is State.ASKING
    assert transition(State.LISTENING, Event.SILENCE_TIMEOUT) is State.PROCESSING
    assert transition(State.PROCESSING, Event.EVAL_DEGRADED) is State.ASKING
    assert states_allowing(Event.ANSWER_COMMIT) == frozenset({State.LISTENING})


def test_processing_is_only_entered_from_listening():
    """防回归：出现跳过作答直接进 PROCESSING 的捷径（没作答却给了分）。"""
    assert predecessors(State.PROCESSING) == frozenset({State.LISTENING})
    assert State.PROCESSING not in set(successors(State.BUFFER).values())
    assert State.PROCESSING not in set(successors(State.BUFFER_LISTENING).values())
    assert State.PROCESSING not in set(successors(State.ASKING).values())
    assert State.PROCESSING not in set(successors(State.FOLLOWUP).values())
    assert set(successors(State.PROCESSING).values()) == {
        State.ASKING,
        State.ABORTED,
        State.FAILED,
    }


def test_buffer_done_is_only_available_in_buffer_listening():
    """防回归：BUFFER_DONE 出现在别的阶段（可被用来绕过评分直接跳回出题）。"""
    assert states_allowing(Event.BUFFER_DONE) == frozenset({State.BUFFER_LISTENING})


def test_insert_buffer_is_only_available_in_asking():
    """防回归：缓冲题被插入到不该插入的阶段（如收尾期），打乱题目结构。"""
    assert states_allowing(Event.INSERT_BUFFER) == frozenset({State.ASKING})
    assert transition(State.ASKING, Event.INSERT_BUFFER) is State.BUFFER
    assert predecessors(State.BUFFER) == frozenset({State.ASKING, State.BUFFER_LISTENING})


def test_only_processing_allows_eval_events():
    """防回归：评分事件在别的时机被触发（如直接 LISTENING --EVAL_DONE--> ASKING），
    会产生"没有作答却给了分"的脏数据。"""
    assert states_allowing(Event.EVAL_DONE) == frozenset({State.PROCESSING})
    assert states_allowing(Event.EVAL_DEGRADED) == frozenset({State.PROCESSING})


# ==========================================================================
# 7. 确定性 / 纯函数
# ==========================================================================


def test_transition_is_deterministic_and_pure():
    """防回归：transition 带副作用或依赖外部时钟/随机，导致重放结果不一致。"""
    table_snapshot = dict(TRANSITIONS)
    for s, e in ALL_PAIRS:
        if can(s, e):
            first = transition(s, e)
            assert transition(s, e) is first
            assert transition(s, e) is first
    assert dict(TRANSITIONS) == table_snapshot
    assert len(TRANSITIONS) == len(table_snapshot)


def test_reachable_from_is_deterministic():
    """防回归：reachable_from 结果不确定（依赖集合迭代顺序给出不同答案）。"""
    for state in ALL_STATES:
        assert reachable_from(state) == reachable_from(state)
        assert reachable_from(state) == reachable_from(state)


def test_reachable_avoiding_is_deterministic_and_monotone():
    """防回归：自实现的"避开集合"搜索结果不稳定，导致核心性质测试时灵时不灵。"""
    for state in ALL_STATES:
        blockers = {State.ASKING, State.LISTENING} - {state}
        first = reachable_avoiding(state, blockers)
        assert first == reachable_avoiding(state, blockers)
        # 屏蔽集合越大，可达集合只会越小（单调性）
        wider = (blockers | {State.CANDIDATE_QA}) - {state}
        if wider != blockers:
            bigger = reachable_avoiding(state, wider)
            assert bigger <= first
        # 起点永远在结果里
        assert state in first


def test_allowed_events_is_deterministic_and_sorted_by_declaration():
    """防回归：allowed_events 顺序随枚举哈希抖动，导致上层取 [0] 时行为漂移。"""
    for state in ALL_STATES:
        first = allowed_events(state)
        assert first == allowed_events(state)
        order = [e for e in ALL_EVENTS if e in first]
        assert tuple(order) == first


# ==========================================================================
# 8. 具体业务路径（穷举之外的定向回归）
# ==========================================================================


def test_happy_path_skeleton():
    """防回归：主干链路被改断。"""
    assert transition(State.IDLE, Event.CREATE) is State.SETUP
    assert transition(State.SETUP, Event.CONSENT_OK) is State.GREETING
    assert transition(State.GREETING, Event.GREETING_DONE) is State.ASKING
    assert transition(State.ASKING, Event.TTS_DONE) is State.LISTENING
    assert transition(State.LISTENING, Event.ANSWER_COMMIT) is State.PROCESSING
    assert transition(State.PROCESSING, Event.EVAL_DONE) is State.ASKING
    assert transition(State.ASKING, Event.NO_MORE_QUESTIONS) is State.CANDIDATE_QA
    assert transition(State.CANDIDATE_QA, Event.QA_DONE) is State.CLOSING
    assert transition(State.CLOSING, Event.CLOSING_DONE) is State.REPORTING
    assert transition(State.REPORTING, Event.REPORT_DONE) is State.COMPLETED


def test_buffer_round_full_path():
    """防回归：整条缓冲路径 ASKING → BUFFER → BUFFER_LISTENING → ASKING 被改断。"""
    assert transition(State.ASKING, Event.INSERT_BUFFER) is State.BUFFER
    assert transition(State.BUFFER, Event.TTS_DONE) is State.BUFFER_LISTENING
    assert transition(State.BUFFER_LISTENING, Event.BUFFER_DONE) is State.ASKING


def test_consent_denied_goes_straight_to_aborted():
    """防回归：未授权仍进入面试流程（合规红线）。"""
    assert transition(State.SETUP, Event.CONSENT_DENIED) is State.ABORTED
    assert State.ABORTED in TERMINAL


def test_degraded_paths_bypass_audio_but_keep_flow():
    """防回归：TTS 失败时流程卡死（应有纯文本降级路径）。
    缓冲题的 TTS 失败同样只进入 BUFFER_LISTENING，绝不落到 LISTENING。"""
    assert transition(State.ASKING, Event.TTS_FAILED) is State.LISTENING
    assert transition(State.FOLLOWUP, Event.TTS_FAILED) is State.LISTENING
    assert transition(State.BUFFER, Event.TTS_FAILED) is State.BUFFER_LISTENING
    assert transition(State.PROCESSING, Event.EVAL_DEGRADED) is State.ASKING


def test_listening_self_loop_and_shortcuts():
    """防回归：延长时限被实现成"重新出题"（EXTEND_TIME 必须是自环）。"""
    assert transition(State.LISTENING, Event.EXTEND_TIME) is State.LISTENING
    assert transition(State.BUFFER_LISTENING, Event.EXTEND_TIME) is State.BUFFER_LISTENING
    assert transition(State.LISTENING, Event.SKIP) is State.ASKING
    assert transition(State.LISTENING, Event.REPEAT_QUESTION) is State.ASKING
    assert transition(State.LISTENING, Event.SILENCE_TIMEOUT) is State.PROCESSING
    assert transition(State.ASKING, Event.NEXT_QUESTION) is State.ASKING
    assert transition(State.ASKING, Event.FOLLOWUP_DECISION) is State.FOLLOWUP
    assert transition(State.CANDIDATE_QA, Event.SKIP) is State.CLOSING


def test_idle_has_no_way_back():
    """防回归：会话被"重置"回 IDLE（应为一次性起点，无入边）。"""
    assert predecessors(State.IDLE) == frozenset()
    assert State.IDLE not in reachable_from(State.SETUP)
    assert State.IDLE not in reachable_from(State.ASKING)
