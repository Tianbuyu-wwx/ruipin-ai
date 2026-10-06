"""会话预算硬顶（方案 §4.4 成本护栏）。

每会话预算硬顶（如 $0.5），到顶即关闭非必要调用并标注。
两档 API：
- `spend()`：账务路径，**超顶抛 BudgetExceeded**（硬失败，不许假装成功）
- `try_spend()`：编排器路径，返回 bool，用于优雅关闭"非必要调用"（如投机预生成、VLM 复核）
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..domain.errors import BudgetExceeded

_EPS = 1e-9


@dataclass
class SessionBudget:
    """一会话一实例。cost_usd 实时计入，管理端可见。"""

    cap_usd: float
    spent: float = 0.0
    spend_log: list[tuple[str, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.cap_usd < 0:
            raise ValueError(f"预算上限不能为负: {self.cap_usd}")

    def remaining(self) -> float:
        return max(0.0, self.cap_usd - self.spent)

    @property
    def exhausted(self) -> bool:
        """已到顶（含刚好花光）。到顶后所有 spend/try_spend 一律失败。"""
        return self.spent >= self.cap_usd - _EPS

    @property
    def usage_ratio(self) -> float:
        if self.cap_usd <= 0:
            return 1.0 if self.spent > 0 else 0.0
        return min(1.0, self.spent / self.cap_usd)

    def can_afford(self, amount: float) -> bool:
        return amount >= 0 and (self.spent + amount) <= self.cap_usd + _EPS

    def spend(self, amount: float, tag: str = "") -> float:
        """扣减并返回剩余额度；超顶抛 BudgetExceeded 且不产生日志。"""
        self._validate(amount)
        if not self.can_afford(amount):
            # 结构化抛错：告警侧能直接读到"超了多少"，不必回代码里推。
            raise BudgetExceeded(
                tag or "session_usd", limit=self.cap_usd, used=self.spent + amount
            )
        self.spent += amount
        self.spend_log.append((tag, amount))
        return self.remaining()

    def try_spend(self, amount: float, tag: str = "") -> bool:
        """优雅扣减：超顶返回 False，**不抛异常**，供编排器关闭非必要调用。"""
        self._validate(amount)
        if not self.can_afford(amount):
            return False
        self.spent += amount
        self.spend_log.append((tag, amount))
        return True

    def total_for(self, tag: str) -> float:
        """按 tag 汇总（如 'vlm' / 'llm_scoring'），用于成本归因。"""
        return sum(a for t, a in self.spend_log if t == tag)

    @staticmethod
    def _validate(amount: float) -> None:
        if amount < 0:
            raise ValueError(f"金额不能为负: {amount}")
