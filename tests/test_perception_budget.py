"""会话预算硬顶（方案 §4.4 成本护栏）。"""

from __future__ import annotations

import pytest

from ruipin.domain.errors import BudgetExceeded, RuipinError
from ruipin.perception.budget import SessionBudget


def test_fresh_budget():
    b = SessionBudget(cap_usd=0.5)
    assert b.spent == 0.0
    assert b.remaining() == 0.5
    assert b.exhausted is False
    assert b.spend_log == []


def test_spend_deducts_and_logs():
    b = SessionBudget(cap_usd=0.5)
    assert b.spend(0.2, tag="vlm") == pytest.approx(0.3)
    assert b.spent == pytest.approx(0.2)
    assert b.remaining() == pytest.approx(0.3)
    assert b.spend_log == [("vlm", 0.2)]


def test_spend_log_accumulates_in_order():
    b = SessionBudget(cap_usd=1.0)
    b.spend(0.1, "vlm")
    b.spend(0.2, "llm_scoring")
    b.spend(0.05, "vlm")
    assert b.spend_log == [("vlm", 0.1), ("llm_scoring", 0.2), ("vlm", 0.05)]
    assert b.spent == pytest.approx(0.35)
    assert b.total_for("vlm") == pytest.approx(0.15)
    assert b.total_for("tts") == 0.0


def test_exactly_exhausting_spend_is_allowed():
    b = SessionBudget(cap_usd=0.5)
    b.spend(0.2)
    b.spend(0.3)
    assert b.remaining() == 0.0
    assert b.exhausted is True


def test_overspend_raises_budget_exceeded_and_changes_nothing():
    b = SessionBudget(cap_usd=0.5)
    b.spend(0.4, "vlm")
    with pytest.raises(BudgetExceeded):
        b.spend(0.2, "vlm")
    assert b.spent == pytest.approx(0.4)          # 失败不扣钱
    assert b.spend_log == [("vlm", 0.4)]          # 失败不入账
    assert b.remaining() == pytest.approx(0.1)


def test_budget_exceeded_is_ruipin_error():
    b = SessionBudget(cap_usd=0.1)
    with pytest.raises(RuipinError):
        b.spend(1.0)


def test_try_spend_returns_false_instead_of_raising():
    """编排器路径：优雅关闭非必要调用，不抛异常、不打断主链。"""
    b = SessionBudget(cap_usd=0.5)
    assert b.try_spend(0.3, "vlm") is True
    assert b.try_spend(0.3, "vlm_recheck") is False
    assert b.spent == pytest.approx(0.3)
    assert b.spend_log == [("vlm", 0.3)]          # 被拒的不入账
    assert b.try_spend(0.2, "vlm") is True        # 剩下的刚好还能花
    assert b.exhausted is True
    assert b.try_spend(0.0001, "vlm") is False


def test_try_spend_boundary_is_inclusive():
    b = SessionBudget(cap_usd=0.25)
    assert b.try_spend(0.25) is True
    assert b.exhausted is True


def test_zero_cap_is_exhausted_from_the_start():
    b = SessionBudget(cap_usd=0.0)
    assert b.exhausted is True
    assert b.usage_ratio == 0.0
    assert b.remaining() == 0.0
    assert b.try_spend(0.01) is False
    with pytest.raises(BudgetExceeded):
        b.spend(0.01)


def test_negative_amounts_rejected():
    b = SessionBudget(cap_usd=1.0)
    with pytest.raises(ValueError):
        b.spend(-0.1)
    with pytest.raises(ValueError):
        b.try_spend(-0.1)
    assert b.spent == 0.0


def test_negative_cap_rejected():
    with pytest.raises(ValueError):
        SessionBudget(cap_usd=-1.0)


def test_usage_ratio():
    b = SessionBudget(cap_usd=0.5)
    assert b.usage_ratio == 0.0
    b.spend(0.25)
    assert b.usage_ratio == pytest.approx(0.5)
    b.spend(0.25)
    assert b.usage_ratio == pytest.approx(1.0)


def test_float_accumulation_does_not_false_positive():
    """0.1 累加 5 次 == 0.5，不能因为浮点误差被判超顶。"""
    b = SessionBudget(cap_usd=0.5)
    for _ in range(5):
        assert b.try_spend(0.1, "vlm") is True
    assert b.spent == pytest.approx(0.5)
    assert b.exhausted is True
