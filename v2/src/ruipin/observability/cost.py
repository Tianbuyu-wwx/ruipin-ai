"""成本治理：单价表 + 成本台账 + 硬顶（方案 §12.3 / §10「成本模型」）。

设计意图
--------
方案原稿只说"要控成本"，但没说清：单价从哪来、超了怎么办、一场到底多少钱。
本模块把它落成三件事：

1. `PriceTable` —— 单位价格表（每 1K token 的输入 / 输出单价），**未知
   provider / model 一律抛错**，绝不按 0 折算（按 0 算会让账单看起来很便宜，
   是最难被发现的一类 bug）。
2. `CostLedger` —— 成本台账：累计、分供应商累计、硬顶、按当前均值推算 N 场。
3. 硬顶语义 —— **失败不入账**：超顶抛 `BudgetExceeded` 且分文不记，绝不出现
   "抛了异常但账目已被污染"的情况（与 `perception.budget.SessionBudget` 同源纪律）。

关键纪律：测不准 = 不计入
-------------------------
* `remaining()` 在没有设 cap 时返回 `None`，不返回 `float("inf")`（无法 JSON
  序列化，也容易被当成"随便花"）。
* `project(n)` 在没有任何场次样本时返回 `None`，**不返回 0**——0 会被读成
  "跑 N 场不要钱"。
* 单价表默认**为空**：单价必须由配置注入（方案 §10 要求用 20 场实测标定），
  不内置任何"看起来很像真的"默认价。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from ruipin.domain.errors import BudgetExceeded, RuipinError
from ruipin.ports import Usage

#: 浮点累加的容差：0.1 连加 5 次必须仍然等于 0.5，不能被判成超顶。
_EPS = 1e-9

__all__ = [
    "Price",
    "PriceTable",
    "UnknownPriceError",
    "CostLedger",
]


class UnknownPriceError(RuipinError, KeyError):
    """单价表里查不到这个 provider / model。

    继承 `KeyError`（语义上就是查表未命中），但重写 `__str__` 以免被 KeyError
    默认的 repr 加引号。消息里带上已登记的清单，排障时不用回代码里翻。
    """

    def __init__(
        self, provider: str, model: str, known: tuple[tuple[str, str], ...] = ()
    ) -> None:
        self.provider = provider
        self.model = model
        self.known = known
        msg = f"单价表未登记: provider={provider!r} model={model!r}"
        if known:
            listed = ", ".join(f"{p}/{m}" for p, m in known)
            msg += f"（已登记: {listed}）"
        self._msg = msg
        super().__init__(msg)

    def __str__(self) -> str:  # KeyError 默认会把消息 repr 成 'msg'
        return self._msg


@dataclass(frozen=True)
class Price:
    """每 1K token 的单价（USD）。"""

    prompt_per_1k: float
    completion_per_1k: float

    def __post_init__(self) -> None:
        for field_name, value in (
            ("prompt_per_1k", self.prompt_per_1k),
            ("completion_per_1k", self.completion_per_1k),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{field_name} 必须是数值：{value!r}")
            if math.isnan(value) or math.isinf(value) or value < 0:
                raise ValueError(f"{field_name} 必须是非负有限数：{value!r}")


class PriceTable:
    """按 (provider, model) 查单价；未登记即抛错，绝不静默按 0 算。

    默认**空表** —— 单价必须由配置注入（`register` 或构造时传入），
    本模块不内置任何占位价，避免出现"看起来很像真的"默认单价。
    """

    def __init__(
        self, prices: Optional[Mapping[tuple[str, str], Price]] = None
    ) -> None:
        self._prices: dict[tuple[str, str], Price] = {}
        if prices:
            for (provider, model), price in prices.items():
                self.register(provider, model, price)

    def register(self, provider: str, model: str, price: Price) -> None:
        if not isinstance(price, Price):
            raise TypeError(f"price 必须是 Price 实例：{price!r}")
        self._prices[(str(provider), str(model))] = price

    def price_of(self, provider: str, model: str) -> Price:
        key = (str(provider), str(model))
        price = self._prices.get(key)
        if price is None:
            raise UnknownPriceError(key[0], key[1], self.known_keys())
        return price

    def known_keys(self) -> tuple[tuple[str, str], ...]:
        return tuple(sorted(self._prices))

    def __len__(self) -> int:
        return len(self._prices)

    def cost_of(
        self,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> float:
        """按 (provider, model) 单价算一次调用的成本（USD）。"""
        if prompt_tokens < 0 or completion_tokens < 0:
            raise ValueError(
                f"token 数不能为负: prompt={prompt_tokens} completion={completion_tokens}"
            )
        price = self.price_of(provider, model)
        return (
            prompt_tokens / 1000.0 * price.prompt_per_1k
            + completion_tokens / 1000.0 * price.completion_per_1k
        )


class CostLedger:
    """成本台账：累计 + 总硬顶 + 分供应商硬顶 + 场次推算。

    * `spend(usage, session_id=...)`：入账。**超顶抛 `BudgetExceeded` 且不入账**。
    * `session_id` 是可选的：只有传了它的开销才算进"单场成本"样本，避免把
      后台任务 / 预热调用算进单场成本里（那会让 `project()` 虚高）。
    * `warn_ratio`（默认 0.8）：达到即 `should_warn()` 为 True，**只告警不阻断**。
    """

    def __init__(
        self,
        cap_usd: Optional[float] = None,
        per_provider_caps: Optional[Mapping[str, float]] = None,
        warn_ratio: float = 0.8,
    ) -> None:
        if cap_usd is not None:
            if math.isnan(cap_usd) or cap_usd < 0:
                raise ValueError(f"cap_usd 必须是非负数或 None：{cap_usd!r}")
        if not 0.0 <= warn_ratio <= 1.0:
            raise ValueError(f"warn_ratio 必须落在 [0, 1]：{warn_ratio!r}")
        self._cap: Optional[float] = None if cap_usd is None else float(cap_usd)
        self._provider_caps: dict[str, float] = {}
        if per_provider_caps:
            for provider, cap in per_provider_caps.items():
                if math.isnan(cap) or cap < 0:
                    raise ValueError(
                        f"供应商 {provider!r} 的 cap 必须是非负数：{cap!r}"
                    )
                self._provider_caps[str(provider)] = float(cap)
        self._warn_ratio = float(warn_ratio)
        self._total = 0.0
        self._by_provider: dict[str, float] = {}
        self._by_session: dict[str, float] = {}
        self._n_calls = 0

    # ----- 入账 -----

    def spend(self, usage: Usage, session_id: Optional[str] = None) -> float:
        """记一笔开销，返回本次入账金额。

        硬顶检查在入账**之前**：超顶就抛 `BudgetExceeded`，且分文不记
        （失败不能污染账目）。先查供应商顶（归因更具体），再查总顶。
        """
        cost = float(usage.cost_usd)
        if math.isnan(cost) or math.isinf(cost):
            raise ValueError(f"cost_usd 必须是有限数：{usage.cost_usd!r}")
        if cost < 0:
            raise ValueError(f"cost_usd 不能为负：{usage.cost_usd!r}")
        provider = str(usage.provider)
        new_provider = self._by_provider.get(provider, 0.0) + cost
        new_total = self._total + cost
        provider_cap = self._provider_caps.get(provider)
        if provider_cap is not None and new_provider > provider_cap + _EPS:
            raise BudgetExceeded(
                f"provider_usd:{provider}", provider_cap, new_provider
            )
        if self._cap is not None and new_total > self._cap + _EPS:
            raise BudgetExceeded("total_usd", self._cap, new_total)
        self._total = new_total
        self._by_provider[provider] = new_provider
        self._n_calls += 1
        if session_id is not None:
            key = str(session_id)
            self._by_session[key] = self._by_session.get(key, 0.0) + cost
        return cost

    # ----- 查询 -----

    @property
    def spent_total(self) -> float:
        return self._total

    @property
    def by_provider(self) -> dict[str, float]:
        return dict(self._by_provider)

    @property
    def by_session(self) -> dict[str, float]:
        return dict(self._by_session)

    def spent_by_provider(self, provider: str) -> float:
        return self._by_provider.get(str(provider), 0.0)

    @property
    def n_calls(self) -> int:
        return self._n_calls

    @property
    def n_sessions(self) -> int:
        return len(self._by_session)

    @property
    def mean_per_session(self) -> Optional[float]:
        """单场均值；没有场次样本时返回 `None`（**不是 0**）。"""
        if not self._by_session:
            return None
        return sum(self._by_session.values()) / len(self._by_session)

    def project(self, n_sessions: int) -> Optional[float]:
        """按当前单场均值推算 N 场的成本；无场次样本返回 `None`。"""
        if n_sessions < 0:
            raise ValueError(f"n_sessions 不能为负：{n_sessions!r}")
        mean = self.mean_per_session
        if mean is None:
            return None
        return mean * n_sessions

    def remaining(self, provider: Optional[str] = None) -> Optional[float]:
        """剩余额度；没设对应 cap 时返回 `None`（不是 inf，也不是"无限"）。"""
        if provider is None:
            if self._cap is None:
                return None
            return max(0.0, self._cap - self._total)
        cap = self._provider_caps.get(str(provider))
        if cap is None:
            return None
        return max(0.0, cap - self._by_provider.get(str(provider), 0.0))

    def usage_ratio(self, provider: Optional[str] = None) -> Optional[float]:
        """已用 / 上限；没设 cap 返回 `None`。零预算（cap=0）按 1.0 计（即已满）。"""
        if provider is None:
            if self._cap is None:
                return None
            return self._ratio(self._total, self._cap)
        cap = self._provider_caps.get(str(provider))
        if cap is None:
            return None
        return self._ratio(self._by_provider.get(str(provider), 0.0), cap)

    def warn_targets(self) -> tuple[str, ...]:
        """已达到 `warn_ratio` 的预算项（`'total'` 或 provider 名）。"""
        targets = []
        if self._cap is not None:
            if self.usage_ratio() >= self._warn_ratio:
                targets.append("total")
        for provider in sorted(self._provider_caps):
            if self.usage_ratio(provider) >= self._warn_ratio:
                targets.append(provider)
        return tuple(targets)

    def should_warn(self) -> bool:
        """是否该打 warning。**只告警，不阻断**（阻断由 `spend` 的硬顶负责）。"""
        return bool(self.warn_targets())

    @staticmethod
    def _ratio(spent: float, cap: float) -> float:
        if cap <= 0:
            return 1.0
        return spent / cap

    def as_dict(self) -> dict[str, Any]:
        """JSON 安全的台账视图（用于日志 / 告警 payload）。"""
        return {
            "spent_total": self._total,
            "by_provider": dict(self._by_provider),
            "by_session": dict(self._by_session),
            "n_calls": self._n_calls,
            "n_sessions": len(self._by_session),
            "cap_usd": self._cap,
            "per_provider_caps": dict(self._provider_caps),
            "warn_ratio": self._warn_ratio,
            "should_warn": self.should_warn(),
            "mean_per_session": self.mean_per_session,
        }
