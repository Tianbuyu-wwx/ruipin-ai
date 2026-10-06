"""会话聚合：状态 + append-only 事件日志。

事件日志的作用：① 可回放审计 ② 断线重连重放 ③ 穷举测试能暴露"声明了但走不到"的缺陷
（现系统 CANDIDATE_QA 不可达正是在这种结构下会被直接测出）。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

from .errors import InvalidTransition
from .machine import TERMINAL, transition
from .states import Event, State


@dataclass(frozen=True)
class StateEvent:
    seq: int
    from_state: Optional[State]
    to_state: State
    event: Event
    ts: float
    payload: dict[str, Any] = field(default_factory=dict)


class Session:
    """会话聚合根。

    `state` 是**只读属性**：修改状态的唯一途径是 `apply(event)`。
    这不是风格偏好——现系统 `app.py` 到处直接改 `session.status`，正是这种
    写法让状态机形同虚设。把禁止写死在语言层面，比写在注释里可靠。
    """

    def __init__(self, session_id: str, state: State = State.IDLE):
        if not isinstance(state, State):
            raise TypeError(f"state 必须是 State 枚举，收到 {state!r}")
        self.session_id = session_id
        self._state = state
        self.events: list[StateEvent] = []
        self._seq = 0
        self.turns: list[dict[str, Any]] = []
        self.meta: dict[str, Any] = {}

    @property
    def state(self) -> State:
        return self._state

    def apply(self, event: Event, payload: Optional[dict[str, Any]] = None) -> State:
        """唯一的写入口。非法事件抛错，绝不静默吞掉。"""
        new_state = transition(self.state, event)
        self._seq += 1
        self.events.append(
            StateEvent(
                seq=self._seq,
                from_state=self.state,
                to_state=new_state,
                event=event,
                ts=time.time(),
                payload=payload or {},
            )
        )
        self._state = new_state
        return new_state

    def try_apply(self, event: Event, payload: Optional[dict] = None) -> Optional[State]:
        try:
            return self.apply(event, payload)
        except InvalidTransition:
            return None

    @property
    def is_terminal(self) -> bool:
        return self.state in TERMINAL

    def history(self) -> list[tuple[State, Event, State]]:
        return [(e.from_state, e.event, e.to_state) for e in self.events]

    def count_event(self, event: Event) -> int:
        return sum(1 for e in self.events if e.event == event)

    def visited(self, state: State) -> bool:
        return any(e.to_state == state for e in self.events)
