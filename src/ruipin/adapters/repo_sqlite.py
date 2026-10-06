"""存储适配器：用标准库 sqlite3 实现 `ruipin.ports.RepoPort`。

> **生产环境将替换为 SQLAlchemy + Alembic 适配器，接口不变。**
> 这里刻意只用标准库 sqlite3：零外部依赖、测试一定能跑起来、schema 自解释，
> 便于在 CI 里对"写入语义"（upsert / 事务）做白盒断言。

相对遗留系统的修正
------------------
1. **upsert，不是"删光重建"**。遗留系统 `database.py:286` 每次保存回合都先
   `delete()` 该会话的全部回合再 insert，写入量 O(N²)，且并发下会掉数据。
   本实现按 `turn_id` 做 `INSERT ... ON CONFLICT DO UPDATE`：
   重复保存同一 turn 只更新那一行，不产生新行、不影响其他行。
2. **单次保存是一个事务**：`BEGIN IMMEDIATE` → 写 → `COMMIT`，任一步失败即
   `ROLLBACK`，绝不留下"写了一部分"的中间状态。
3. **append-only 事件日志**：`events` 表只追加不改写，配合 `seq` 可回放审计。

约定
----
* `data` 列是 JSON 文本；`load_*` 返回的是**原样回读**的 dict，不做字段注入。
* `state` 列是 `data["state"]` 的镜像，只为可检索存在（接受 Enum，落库取 `.value`）。
* 时间戳一律为 REAL（epoch 秒），时钟可注入（默认 `time.time`），便于确定性测试。
* 外键在 DDL 中声明但**不强制** `PRAGMA foreign_keys`：编排层允许先写 turn 后
  落 session 快照，强制外键会把这种合法顺序变成运行时错误。
"""

from __future__ import annotations

import json
import sqlite3
import time
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

_SCHEMA: tuple[str, ...] = (
    # 会话快照
    """
    CREATE TABLE IF NOT EXISTS sessions (
        session_id TEXT PRIMARY KEY,
        state      TEXT,
        data       TEXT NOT NULL,
        created_at REAL NOT NULL,
        updated_at REAL NOT NULL
    )
    """,
    # 回合：按 turn_id upsert，绝不整表删除重建
    """
    CREATE TABLE IF NOT EXISTS turns (
        turn_id    TEXT PRIMARY KEY,
        session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
        turn_index INTEGER NOT NULL DEFAULT 0,
        data       TEXT NOT NULL,
        updated_at REAL NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id)",
    # 事件日志：append-only
    """
    CREATE TABLE IF NOT EXISTS events (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
        seq        INTEGER NOT NULL,
        data       TEXT NOT NULL,
        ts         REAL NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_events_session_seq ON events(session_id, seq)",
)

_UPSERT_SESSION = """
INSERT INTO sessions (session_id, state, data, created_at, updated_at)
VALUES (?, ?, ?, ?, ?)
ON CONFLICT(session_id) DO UPDATE SET
    state      = excluded.state,
    data       = excluded.data,
    updated_at = excluded.updated_at
"""

_UPSERT_TURN = """
INSERT INTO turns (turn_id, session_id, turn_index, data, updated_at)
VALUES (?, ?, ?, ?, ?)
ON CONFLICT(turn_id) DO UPDATE SET
    session_id = excluded.session_id,
    turn_index = excluded.turn_index,
    data       = excluded.data,
    updated_at = excluded.updated_at
"""


def _json_default(obj: Any) -> Any:
    """Enum（State / Event）按 `.value` 落库；其余类型一律报错，不静默兜底。"""
    if isinstance(obj, Enum):
        return obj.value
    raise TypeError(f"不可序列化的字段: {obj!r}")


def _dumps(data: Mapping[str, Any]) -> str:
    return json.dumps(dict(data), ensure_ascii=False, default=_json_default)


def _state_of(data: Mapping[str, Any]) -> Optional[str]:
    state = data.get("state")
    if state is None:
        return None
    return state.value if isinstance(state, Enum) else str(state)


def _turn_index_of(turn: Mapping[str, Any]) -> int:
    raw = turn.get("turn_index")
    if raw is None:
        return 0
    return int(raw)


class SqliteRepo:
    """`RepoPort` 的 sqlite3 实现。

    Args:
        db_path: 数据库路径，`:memory:` 表示内存库（默认）。
        clock: 时间源，返回 epoch 秒；注入可控时钟即可做确定性断言。
    """

    _closed = False

    def __init__(
        self,
        db_path: str | Path = ":memory:",
        *,
        clock: Optional[Callable[[], float]] = None,
    ) -> None:
        self.db_path = str(db_path)
        self._clock: Callable[[], float] = clock or time.time
        # isolation_level=None → 关闭 python 的隐式事务，由本类显式 BEGIN/COMMIT
        self._conn = sqlite3.connect(self.db_path, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        for ddl in _SCHEMA:
            self._conn.execute(ddl)

    # ---------- RepoPort ----------

    def save_session(self, session_id: str, data: dict[str, Any]) -> None:
        """保存/更新会话快照（upsert，按 session_id）。"""
        self._begin()
        try:
            payload = _dumps(data)
            now = self._clock()
            self._conn.execute(
                _UPSERT_SESSION,
                (session_id, _state_of(data), payload, now, now),
            )
        except Exception:
            self._rollback()
            raise
        self._commit()

    def save_turn(self, session_id: str, turn: dict[str, Any]) -> None:
        """保存/更新单个回合（upsert，按 turn_id）。"""
        self.save_turns(session_id, [turn])

    def save_turns(self, session_id: str, turns: list[dict[str, Any]]) -> None:
        """批量 upsert 回合：一个事务，要么全成功要么整体回滚。

        端口（`RepoPort.save_turn`）只定义了单个回合；批量接口是为"一轮结束写多个
        turn"与"重放整场会话"准备的，同时它是事务语义唯一可观测的入口。
        """
        self._begin()
        try:
            now = self._clock()
            for turn in turns:
                turn_id = turn.get("turn_id")
                if not turn_id:
                    raise ValueError(f"turn 缺少非空 turn_id: {turn!r}")
                self._conn.execute(
                    _UPSERT_TURN,
                    (str(turn_id), session_id, _turn_index_of(turn), _dumps(turn), now),
                )
        except Exception:
            self._rollback()
            raise
        self._commit()

    def load_session(self, session_id: str) -> Optional[dict[str, Any]]:
        row = self._conn.execute(
            "SELECT data FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        return json.loads(row["data"]) if row is not None else None

    def load_turns(self, session_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT data FROM turns WHERE session_id = ? ORDER BY turn_index, turn_id",
            (session_id,),
        ).fetchall()
        return [json.loads(r["data"]) for r in rows]

    def append_event(self, session_id: str, ev: dict[str, Any]) -> int:
        """追加一条事件（append-only）。返回落库的 seq。

        未显式给 `seq` 时按该会话当前最大 seq + 1 分配；回读的事件**一定带 `seq`**，
        使日志本身自描述、可直接回放，无需依赖表列。
        """
        seq = ev.get("seq")
        if seq is None:
            row = self._conn.execute(
                "SELECT MAX(seq) AS m FROM events WHERE session_id = ?", (session_id,)
            ).fetchone()
            seq = (row["m"] or 0) + 1
        seq = int(seq)
        self._begin()
        try:
            payload = dict(ev)
            payload.setdefault("seq", seq)
            self._conn.execute(
                "INSERT INTO events (session_id, seq, data, ts) VALUES (?, ?, ?, ?)",
                (session_id, seq, _dumps(payload), self._clock()),
            )
        except Exception:
            self._rollback()
            raise
        self._commit()
        return seq

    # ---------- 读扩展（非端口方法，但 events 表必须有读者才成立） ----------

    def load_events(self, session_id: str) -> list[dict[str, Any]]:
        """按 seq 升序回读事件日志，供回放 / 审计 / 断线重连使用。"""
        rows = self._conn.execute(
            "SELECT data FROM events WHERE session_id = ? ORDER BY seq, id",
            (session_id,),
        ).fetchall()
        return [json.loads(r["data"]) for r in rows]

    # ---------- 连接管理 ----------

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        self._closed = True
        self._conn.close()

    def __enter__(self) -> "SqliteRepo":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ---------- 内部 ----------

    def _begin(self) -> None:
        self._conn.execute("BEGIN IMMEDIATE")

    def _commit(self) -> None:
        self._conn.execute("COMMIT")

    def _rollback(self) -> None:
        if self._conn.in_transaction:
            self._conn.execute("ROLLBACK")
