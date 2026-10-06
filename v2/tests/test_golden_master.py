"""Golden Master 契约快照比对。

跑一场**固定输入**的面试，把对外可见的产物（状态迁移序列 / 每轮是否计分 /
总分 / 等级 / 维度 / 生效权重 / 说明文案 / 口径版本）规范化后与 `EXPECTED_SNAPSHOT`
逐键比对。不一致时打印结构化 diff，由人判断是"有意变更"还是"回归"。

⚠️ 测试**不会**自动更新 `EXPECTED_SNAPSHOT`（那样就永远通过、失去意义）。
更新方法见 `golden/contract_snapshot.py` 顶部 docstring。
"""

from __future__ import annotations

import pytest

from golden.contract_snapshot import (
    EXPECTED_SNAPSHOT,
    build_snapshot,
    diff_snapshot,
)

pytestmark = pytest.mark.integration

#: 快照里**不允许**出现的键：时间戳、会话/回合唯一 id、延迟与成本、provider 自由文本。
#: 出现任何一个都说明归一化漏了东西，快照会随环境漂移。
FORBIDDEN_KEYS = frozenset(
    {"ts", "session_id", "turn_id", "latency_ms", "cost_usd",
     "created_at", "updated_at", "feedback", "provider", "payload", "seq", "eval"}
)


def _all_keys(node, acc: set[str]) -> set[str]:
    if isinstance(node, dict):
        for key, value in node.items():
            acc.add(str(key))
            _all_keys(value, acc)
    elif isinstance(node, list):
        for item in node:
            _all_keys(item, acc)
    return acc


def test_golden_contract_matches_frozen_snapshot():
    """防回归：对外契约（状态序列 / 计分口径 / 分数 / 权重 / 文案）不漂移。"""
    actual = build_snapshot()
    diff = diff_snapshot(EXPECTED_SNAPSHOT, actual)
    if diff:
        pytest.fail(
            "Golden Master 契约快照与冻结值不一致（若确属有意变更，"
            "按 golden/contract_snapshot.py 顶部说明重新生成）：\n  "
            + "\n  ".join(diff)
        )


def test_snapshot_is_deterministic_across_runs():
    """防回归：同一份固定输入跑两次必须完全相同——否则它根本不能当快照用。"""
    assert build_snapshot() == build_snapshot()


def test_snapshot_contains_no_nondeterministic_field():
    """防回归：快照里不得混入时间戳 / 会话 id / 延迟 / provider 自由文本。"""
    leaked = _all_keys(build_snapshot(), set()) & FORBIDDEN_KEYS
    assert leaked == set(), f"快照混入了不确定字段: {sorted(leaked)}"


def test_rubric_version_is_wired_through_to_the_artifact():
    """防回归：`rubric_version` 必须真的串到产物里，而不是报告上写死的常量。"""
    actual = build_snapshot(rubric_version="rubric-v2-experiment")
    assert actual["report"]["rubric_version"] == "rubric-v2-experiment"

    diff = diff_snapshot(EXPECTED_SNAPSHOT, actual)
    assert len(diff) == 1, f"改口径版本不应牵动其他契约字段：{diff}"
    assert "rubric_version" in diff[0]
    assert "rubric-v1" in diff[0] and "rubric-v2-experiment" in diff[0]


def test_frozen_snapshot_shape_is_complete():
    """防回归：冻结值本身结构完整（防止手改常量时漏项却仍"通过"）。"""
    assert set(EXPECTED_SNAPSHOT) == {"transitions", "turns", "physio", "report"}
    assert set(EXPECTED_SNAPSHOT["report"]) == {
        "available", "reason", "score", "level", "dims", "effective_weights",
        "counterfactuals", "notes", "rubric_version", "n_scored_turns", "n_buffer_turns",
    }
    # 4 道核心题 × (出题 + 评分) + 4 道缓冲题 × 出题 → 8 个回合
    assert len(EXPECTED_SNAPSHOT["turns"]) == 8
    assert sum(1 for t in EXPECTED_SNAPSHOT["turns"] if t["scored"]) == 4
    assert sum(1 for t in EXPECTED_SNAPSHOT["turns"] if not t["scored"]) == 4
    assert EXPECTED_SNAPSHOT["transitions"][0] == ["idle", "create", "setup"]
    assert EXPECTED_SNAPSHOT["transitions"][-1] == ["reporting", "report_done", "completed"]
