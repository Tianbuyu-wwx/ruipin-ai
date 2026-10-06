"""存储适配器测试。

重点覆盖三类缺陷：
1. **upsert 语义**（对应遗留系统 `database.py:286` 的"删光重建"）：重复保存不增行、不误删兄弟行；
2. **事务**：一批写入要么全成功要么整体回滚；
3. **连接与模式**：`:memory:` 与临时文件两种模式、close / context manager。
"""

from __future__ import annotations

import sqlite3
from enum import Enum

import pytest

from ruipin.adapters.repo_sqlite import SqliteRepo
from ruipin.domain.states import State
from ruipin.ports import RepoPort


# ---------- helpers ----------


def _turn(turn_id: str, turn_index: int, **extra) -> dict:
    return {"turn_id": turn_id, "turn_index": turn_index, **extra}


def _count(repo: SqliteRepo, table: str) -> int:
    """表里实际行数（白盒：本测试专门验证"没有多写/没有误删"）。"""
    return repo._conn.execute(f"SELECT COUNT(*) AS c FROM {table}").fetchone()["c"]


def _count_turns(repo: SqliteRepo, session_id: str) -> int:
    return repo._conn.execute(
        "SELECT COUNT(*) AS c FROM turns WHERE session_id = ?", (session_id,)
    ).fetchone()["c"]


# ---------- 端口契约 ----------


def test_implements_repo_port(tmp_db):
    assert isinstance(tmp_db, RepoPort)


# ---------- session ----------


def test_session_roundtrip(tmp_db, session_id):
    data = {"session_id": session_id, "state": "asking", "candidate": "张三", "n": 1}
    tmp_db.save_session(session_id, data)

    assert tmp_db.load_session(session_id) == data


def test_load_unknown_session_returns_none(tmp_db):
    assert tmp_db.load_session("nope") is None


def test_save_session_twice_is_upsert_not_duplicate(tmp_db, session_id, clock):
    """重复保存会话：仍只有一行，created_at 不变、updated_at 前进。"""
    repo = SqliteRepo(":memory:", clock=clock.monotonic)
    try:
        clock.advance(10.0)
        repo.save_session(session_id, {"state": "setup", "v": 1})
        clock.advance(5.0)
        repo.save_session(session_id, {"state": "asking", "v": 2})

        assert _count(repo, "sessions") == 1
        assert repo.load_session(session_id) == {"state": "asking", "v": 2}

        row = repo._conn.execute(
            "SELECT created_at, updated_at FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        assert row["created_at"] == 10.0
        assert row["updated_at"] == 15.0
    finally:
        repo.close()


def test_state_column_mirrors_data_and_accepts_enum(tmp_db, session_id):
    tmp_db.save_session(session_id, {"state": State.ASKING})

    row = tmp_db._conn.execute(
        "SELECT state FROM sessions WHERE session_id = ?", (session_id,)
    ).fetchone()
    assert row["state"] == "asking"
    # 回读的 data 里 Enum 已按 .value 落库（不静默 str() 成 "State.ASKING"）
    assert tmp_db.load_session(session_id) == {"state": "asking"}


def test_non_str_enum_serialized_by_value(tmp_db, session_id):
    class Kind(Enum):  # 非 str 子类：必须走 default=，按 .value 落库
        TECHNICAL = "technical"

    tmp_db.save_session(session_id, {"kind": Kind.TECHNICAL})
    assert tmp_db.load_session(session_id) == {"kind": "technical"}


def test_save_session_rejects_unserializable_data(tmp_db, session_id):
    with pytest.raises(TypeError):
        tmp_db.save_session(session_id, {"bad": object()})
    assert tmp_db.load_session(session_id) is None
    assert _count(tmp_db, "sessions") == 0


# ---------- turns：upsert 语义 ----------


def test_save_turn_upsert_is_idempotent(tmp_db, session_id):
    """同一 turn_id 连续保存三次：仍是一行，内容是最后一次，且不删除别的行。"""
    tmp_db.save_turn(session_id, _turn("t1", 0, answer="v1"))
    tmp_db.save_turn(session_id, _turn("t1", 0, answer="v2"))
    tmp_db.save_turn(session_id, _turn("t1", 0, answer="v3"))

    turns = tmp_db.load_turns(session_id)
    assert len(turns) == 1
    assert turns[0]["answer"] == "v3"
    assert _count(tmp_db, "turns") == 1


def test_save_turn_does_not_delete_siblings(tmp_db, session_id):
    """针对遗留缺陷：保存 B 之后，之前保存的 A 必须还在（不是 DELETE 全部再 INSERT）。"""
    tmp_db.save_turn(session_id, _turn("t-a", 0, answer="A"))
    tmp_db.save_turn(session_id, _turn("t-b", 1, answer="B"))
    tmp_db.save_turn(session_id, _turn("t-a", 0, answer="A2"))  # 更新 A

    turns = tmp_db.load_turns(session_id)
    assert [t["turn_id"] for t in turns] == ["t-a", "t-b"]
    assert turns[0]["answer"] == "A2"
    assert turns[1]["answer"] == "B"
    assert _count_turns(tmp_db, session_id) == 2


def test_save_turns_bulk_upsert_all_present(tmp_db, session_id):
    tmp_db.save_turns(session_id, [_turn("t1", 0), _turn("t2", 1), _turn("t3", 2)])
    tmp_db.save_turns(session_id, [_turn("t2", 1, updated=True)])

    assert _count_turns(tmp_db, session_id) == 3
    assert tmp_db.load_turns(session_id)[1]["updated"] is True


def test_load_turns_sorted_by_turn_index(tmp_db, session_id):
    tmp_db.save_turns(
        session_id, [_turn("t3", 2), _turn("t1", 0), _turn("t2", 1)]
    )

    assert [t["turn_id"] for t in tmp_db.load_turns(session_id)] == ["t1", "t2", "t3"]


def test_load_turns_empty_for_unknown_session(tmp_db):
    assert tmp_db.load_turns("nope") == []


def test_turns_are_isolated_between_sessions(tmp_db):
    tmp_db.save_turn("s1", _turn("t1", 0))
    tmp_db.save_turn("s2", _turn("t2", 0))

    assert tmp_db.load_turns("s1") == [_turn("t1", 0)]
    assert tmp_db.load_turns("s2") == [_turn("t2", 0)]


def test_same_turn_id_moves_to_latest_session(tmp_db):
    """turn_id 是全局主键：同一 turn_id 后写会覆盖（含 session_id 归属）。"""
    tmp_db.save_turn("s1", _turn("t1", 0))
    tmp_db.save_turn("s2", _turn("t1", 0))

    assert tmp_db.load_turns("s1") == []
    assert tmp_db.load_turns("s2") == [_turn("t1", 0)]


def test_save_turn_requires_turn_id(tmp_db, session_id):
    with pytest.raises(ValueError):
        tmp_db.save_turn(session_id, {"turn_index": 0})
    assert _count(tmp_db, "turns") == 0


def test_turn_index_defaults_to_zero(tmp_db, session_id):
    tmp_db.save_turn(session_id, {"turn_id": "t1"})
    assert tmp_db.load_turns(session_id) == [{"turn_id": "t1"}]


# ---------- events ----------


def test_append_event_auto_seq_and_order(tmp_db, session_id):
    for i in range(3):
        tmp_db.append_event(session_id, {"kind": f"e{i}"})

    events = tmp_db.load_events(session_id)
    assert [e["kind"] for e in events] == ["e0", "e1", "e2"]
    assert [e["seq"] for e in events] == [1, 2, 3]


def test_append_event_keeps_explicit_seq(tmp_db, session_id):
    tmp_db.append_event(session_id, {"seq": 10, "kind": "late"})
    tmp_db.append_event(session_id, {"seq": 2, "kind": "early"})

    # load_events 按 seq 升序，写库顺序不影响回放顺序
    assert [e["kind"] for e in tmp_db.load_events(session_id)] == ["early", "late"]
    # 未指定 seq 时从当前最大 seq 继续
    assert tmp_db.append_event(session_id, {"kind": "auto"}) == 11


def test_events_isolated_and_empty(tmp_db, session_id):
    tmp_db.append_event(session_id, {"kind": "e"})
    assert tmp_db.load_events("other") == []
    assert len(tmp_db.load_events(session_id)) == 1


def test_event_roundtrip_preserves_payload_and_adds_seq(tmp_db, session_id):
    ev = {"event": "eval_done", "payload": {"score": 65.0, "provider": "fake"}}
    tmp_db.append_event(session_id, ev)
    assert tmp_db.load_events(session_id) == [{**ev, "seq": 1}]


def test_append_event_rolls_back_on_failure(tmp_db, session_id):
    tmp_db.append_event(session_id, {"kind": "ok"})
    with pytest.raises(TypeError):
        tmp_db.append_event(session_id, {"bad": object()})

    assert tmp_db.load_events(session_id) == [{"kind": "ok", "seq": 1}]


# ---------- 事务 ----------


def test_bulk_save_rolls_back_without_partial_data(tmp_db, session_id):
    """一批 3 条，第 3 条序列化失败 → 前两条也不落库。"""
    bad = _turn("t3", 2)
    bad["frame"] = object()  # json 无法序列化

    with pytest.raises(TypeError):
        tmp_db.save_turns(session_id, [_turn("t1", 0), _turn("t2", 1), bad])

    assert tmp_db.load_turns(session_id) == []
    assert _count(tmp_db, "turns") == 0


def test_rollback_does_not_poison_connection(tmp_db, session_id):
    with pytest.raises(ValueError):
        tmp_db.save_turns(session_id, [_turn("t1", 0), {"turn_index": 1}])

    tmp_db.save_turn(session_id, _turn("t9", 0, ok=True))
    assert tmp_db.load_turns(session_id) == [_turn("t9", 0, ok=True)]


# ---------- 连接与模式 ----------


def test_file_db_persists_across_reopen(tmp_path):
    path = tmp_path / "persist.sqlite3"
    repo = SqliteRepo(path)
    repo.save_session("s1", {"state": "completed"})
    repo.save_turn("s1", _turn("t1", 0, answer="hi"))
    repo.close()

    with SqliteRepo(path) as repo2:
        assert repo2.load_session("s1") == {"state": "completed"}
        assert repo2.load_turns("s1") == [_turn("t1", 0, answer="hi")]


def test_context_manager_closes(make_repo):
    with make_repo(":memory:") as repo:
        repo.save_session("s1", {"state": "idle"})
        assert repo.closed is False
    assert repo.closed is True
    with pytest.raises(sqlite3.ProgrammingError):
        repo.load_session("s1")


def test_memory_and_file_modes_are_independent(make_repo):
    mem = make_repo(":memory:")
    disk = make_repo("disk.sqlite3")
    mem.save_session("s1", {"where": "mem"})
    disk.save_session("s1", {"where": "disk"})

    assert mem.load_session("s1") == {"where": "mem"}
    assert disk.load_session("s1") == {"where": "disk"}


# ---------- schema ----------


def test_turns_has_index_on_session_id(tmp_db):
    rows = tmp_db._conn.execute("PRAGMA index_info(idx_turns_session)").fetchall()
    assert [r["name"] for r in rows] == ["session_id"]


def test_events_has_index_on_session_and_seq(tmp_db):
    rows = tmp_db._conn.execute("PRAGMA index_info(idx_events_session_seq)").fetchall()
    assert [r["name"] for r in rows] == ["session_id", "seq"]


def test_events_id_is_autoincrement(tmp_db, session_id):
    tmp_db.append_event(session_id, {"kind": "a"})
    tmp_db.append_event(session_id, {"kind": "b"})
    ids = [r["id"] for r in tmp_db._conn.execute("SELECT id FROM events").fetchall()]
    assert ids == [1, 2]
