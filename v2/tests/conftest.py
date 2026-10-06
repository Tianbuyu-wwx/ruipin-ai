"""共享 fixture。

注意：本文件会被**所有**测试（含其他模块并行编写的测试）加载，因此对尚在
开发中的包（`scoring` / `physio` / `perception` / `orchestrator` 等）一律用
try/except 保护 —— 缺失时跳过，绝不因 import 失败炸掉整个套件。
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Coroutine, TypeVar

import pytest

from ruipin.adapters.fakes import DeterministicClock, FakeEvaluator
from ruipin.adapters.repo_sqlite import SqliteRepo
from ruipin.domain.session import Session
from ruipin.ports import EvalRequest

T = TypeVar("T")


def run_async(coro: Coroutine[Any, Any, T]) -> T:
    """同步测试里跑协程的小助手（环境未装 pytest-asyncio）。"""
    return asyncio.run(coro)


# ---------- 可选依赖：并行开发中的包，缺失则跳过 ----------

try:
    from ruipin import scoring as _scoring
except Exception:  # noqa: BLE001 - 模块可能还在被创建
    _scoring = None

try:
    from ruipin import physio as _physio
except Exception:  # noqa: BLE001
    _physio = None

try:
    from ruipin import perception as _perception
except Exception:  # noqa: BLE001
    _perception = None

try:
    from ruipin import orchestrator as _orchestrator
except Exception:  # noqa: BLE001
    _orchestrator = None


def _optional_module(mod: object, name: str) -> object:
    if mod is None:
        pytest.skip(f"ruipin.{name} 尚未就绪（并行开发中）")
    return mod


@pytest.fixture
def scoring_module():
    return _optional_module(_scoring, "scoring")


@pytest.fixture
def physio_module():
    return _optional_module(_physio, "physio")


@pytest.fixture
def perception_module():
    return _optional_module(_perception, "perception")


@pytest.fixture
def orchestrator_module():
    return _optional_module(_orchestrator, "orchestrator")


# ---------- 存储 ----------


@pytest.fixture
def tmp_db() -> SqliteRepo:
    """内存版存储（每测试一个全新连接）。"""
    repo = SqliteRepo(":memory:")
    try:
        yield repo
    finally:
        repo.close()


@pytest.fixture
def tmp_db_path(tmp_path) -> str:
    """临时文件版存储的 db 路径。"""
    return str(tmp_path / "repo.sqlite3")


@pytest.fixture
def make_repo(tmp_path) -> Callable[[str], SqliteRepo]:
    """按路径构造存储；传 ':memory:' 得内存库。调用方负责 close。"""
    created: list[SqliteRepo] = []

    def _make(db_path: str = ":memory:") -> SqliteRepo:
        path = ":memory:" if db_path == ":memory:" else str(tmp_path / db_path)
        repo = SqliteRepo(path)
        created.append(repo)
        return repo

    try:
        yield _make
    finally:
        for repo in created:
            repo.close()


@pytest.fixture
def clock() -> DeterministicClock:
    return DeterministicClock()


# ---------- 领域 ----------


@pytest.fixture
def session() -> Session:
    """一个全新的 Session（IDLE）。"""
    return Session("s-fixture")


@pytest.fixture
def session_id() -> str:
    return "s-fixture"


# ---------- 评分替身 ----------


@pytest.fixture
def fake_evaluator() -> FakeEvaluator:
    return FakeEvaluator()


@pytest.fixture
def sample_eval_request() -> EvalRequest:
    """固定的 EvalRequest —— Golden Master / 确定性断言专用，勿改动字段。"""
    return EvalRequest(
        question="请介绍一个你主导过的项目，以及你在其中解决的关键问题。",
        answer="我主导了推荐系统的召回重构，用向量检索替换倒排，QPS 提升 3 倍。",
        question_type="technical",
        transcript="我主导了推荐系统的召回重构，用向量检索替换倒排，QPS 提升 3 倍。",
        has_audio=True,
        has_video=False,
        keywords=("推荐系统", "向量检索", "召回"),
    )
