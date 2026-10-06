"""声明式状态转换表（唯一真相源）。

所有状态迁移必须经过 transition()，禁止任何地方直接赋值 session.state
（现系统 app.py 直接改 session.status 是反面教材，自检 A5 已修正）。
"""

from __future__ import annotations

from .errors import InvalidTransition
from .states import ALL_EVENTS, ALL_STATES, Event, State

S = State
E = Event

# (当前状态, 事件) -> 新状态
TRANSITIONS: dict[tuple[State, Event], State] = {
    (S.IDLE, E.CREATE): S.SETUP,
    (S.SETUP, E.CONSENT_OK): S.GREETING,
    (S.SETUP, E.CONSENT_DENIED): S.ABORTED,
    (S.GREETING, E.GREETING_DONE): S.ASKING,

    # 出题与呈现
    (S.ASKING, E.TTS_DONE): S.LISTENING,
    (S.ASKING, E.TTS_FAILED): S.LISTENING,      # 降级为纯文本呈现，不放音
    (S.ASKING, E.INSERT_BUFFER): S.BUFFER,      # 适应性：插入缓冲题

    # 缓冲回合：BUFFER → BUFFER_LISTENING → ASKING。
    # ★ 缓冲回合**不存在 PROCESSING 阶段**，因此不存在任何评分时机——
    # 这是"适应性调节只能作用于不计分部分"（详设 §5.3.1 / 自检 A3）的结构保证，
    # 而不是靠调用方自觉。BUFFER_LISTENING 与 LISTENING 分离正是为此。
    (S.BUFFER, E.TTS_DONE): S.BUFFER_LISTENING,
    (S.BUFFER, E.TTS_FAILED): S.BUFFER_LISTENING,
    (S.BUFFER_LISTENING, E.EXTEND_TIME): S.BUFFER_LISTENING,
    (S.BUFFER_LISTENING, E.REPEAT_QUESTION): S.BUFFER,      # 重听缓冲题
    (S.BUFFER_LISTENING, E.BUFFER_DONE): S.ASKING,
    (S.BUFFER_LISTENING, E.SILENCE_TIMEOUT): S.ASKING,      # 缓冲题答不上来：直接过
    (S.BUFFER_LISTENING, E.SKIP): S.ASKING,

    # 作答
    (S.LISTENING, E.ANSWER_COMMIT): S.PROCESSING,
    (S.LISTENING, E.SILENCE_TIMEOUT): S.PROCESSING,
    (S.LISTENING, E.SKIP): S.ASKING,
    (S.LISTENING, E.REPEAT_QUESTION): S.ASKING,
    (S.LISTENING, E.EXTEND_TIME): S.LISTENING,  # 自环：仅延长时限

    # 评估
    (S.PROCESSING, E.EVAL_DONE): S.ASKING,
    (S.PROCESSING, E.EVAL_DEGRADED): S.ASKING,
    # 注意：此处**故意没有** (PROCESSING, BUFFER_DONE)。
    # 缓冲回合已在 BUFFER_LISTENING 处结束，不该有"缓冲答案进入 PROCESSING"的路径。

    # 推进
    (S.ASKING, E.FOLLOWUP_DECISION): S.FOLLOWUP,
    (S.FOLLOWUP, E.TTS_DONE): S.LISTENING,
    (S.FOLLOWUP, E.TTS_FAILED): S.LISTENING,
    (S.ASKING, E.NEXT_QUESTION): S.ASKING,
    (S.ASKING, E.NO_MORE_QUESTIONS): S.CANDIDATE_QA,

    # 收尾
    (S.CANDIDATE_QA, E.QA_DONE): S.CLOSING,
    (S.CANDIDATE_QA, E.SKIP): S.CLOSING,
    (S.CLOSING, E.CLOSING_DONE): S.REPORTING,
    (S.REPORTING, E.REPORT_DONE): S.COMPLETED,
}

# 通配符：任意状态都可发生
WILDCARD: dict[Event, State] = {
    E.ABORT: S.ABORTED,
    E.FATAL: S.FAILED,
}

# 终态：不可再迁出
TERMINAL = frozenset({S.COMPLETED, S.ABORTED, S.FAILED})


def transition(state: State, event: Event) -> State:
    """纯函数：返回新状态；非法转换抛 InvalidTransition。"""
    key = (state, event)
    if key in TRANSITIONS:
        return TRANSITIONS[key]
    if event in WILDCARD:
        return WILDCARD[event]
    raise InvalidTransition(state, event)


def can(state: State, event: Event) -> bool:
    try:
        transition(state, event)
        return True
    except InvalidTransition:
        return False


def allowed_events(state: State) -> tuple[Event, ...]:
    return tuple(e for e in ALL_EVENTS if can(state, e))


def reachable_from(state: State) -> frozenset[State]:
    """从给定状态出发可达的状态集合（含自身）。"""
    seen = {state}
    frontier = [state]
    while frontier:
        cur = frontier.pop()
        for e in ALL_EVENTS:
            if can(cur, e):
                nxt = transition(cur, e)
                if nxt not in seen:
                    seen.add(nxt)
                    frontier.append(nxt)
    return frozenset(seen)


__all__ = [
    "TRANSITIONS", "WILDCARD", "TERMINAL", "ALL_STATES", "ALL_EVENTS",
    "transition", "can", "allowed_events", "reachable_from",
]
