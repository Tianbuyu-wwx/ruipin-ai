"""成本治理（单价表 / 台账 / 硬顶）的单元测试。

防的回归：未知单价被静默按 0 折算（账单看起来很便宜）、超顶时先入账再抛异常
（账目被污染）、没有样本时 `project()` 返回 0 假装免费、warn 变成硬阻断。
"""

from __future__ import annotations

import json

import pytest

from ruipin.domain.errors import BudgetExceeded, RuipinError
from ruipin.observability.cost import (
    CostLedger,
    Price,
    PriceTable,
    UnknownPriceError,
)
from ruipin.ports import Usage

pytestmark = pytest.mark.unit


def usage(provider: str = "openai", cost: float = 0.1, **kw) -> Usage:
    return Usage(provider=provider, operation=kw.pop("operation", "score"),
                 cost_usd=cost, **kw)


# ---------- 单价表 ----------


def test_cost_of_matches_hand_computed_price():
    """防回归：单价 × token/1000 必须手算可核对，不能有任何隐含取整。"""
    table = PriceTable({("openai", "gpt"): Price(0.01, 0.03)})
    # 1000/1000*0.01 + 500/1000*0.03 = 0.01 + 0.015
    assert table.cost_of("openai", "gpt", 1000, 500) == pytest.approx(0.025)
    assert table.cost_of("openai", "gpt", 0, 0) == pytest.approx(0.0)
    assert table.cost_of("openai", "gpt", 2000, 1000) == pytest.approx(0.05)


def test_unknown_model_raises_and_never_returns_zero():
    """防回归：未登记的 model 必须抛错——按 0 算会让成本看起来免费。"""
    table = PriceTable({("openai", "gpt"): Price(0.01, 0.03)})
    with pytest.raises(UnknownPriceError) as exc:
        table.cost_of("openai", "unreleased-model", 1000, 1000)
    assert "unreleased-model" in str(exc.value)
    assert "openai/gpt" in str(exc.value)          # 消息要列出已登记项
    assert isinstance(exc.value, RuipinError)
    assert isinstance(exc.value, KeyError)         # 语义上就是查表未命中
    assert exc.value.provider == "openai"
    assert exc.value.model == "unreleased-model"


def test_unknown_provider_raises():
    """防回归：provider 拼错（大小写 / 别名）也必须当场炸，而不是按 0 记账。"""
    table = PriceTable({("openai", "gpt"): Price(0.01, 0.03)})
    with pytest.raises(UnknownPriceError):
        table.price_of("openai ", "gpt")
    with pytest.raises(KeyError):
        table.price_of("qwen", "gpt")


def test_default_table_is_empty_by_design():
    """防回归：不内置占位单价——任何单价都必须由配置注入（方案 §10 标定）。"""
    table = PriceTable()
    assert len(table) == 0
    assert table.known_keys() == ()
    with pytest.raises(UnknownPriceError):
        table.cost_of("openai", "gpt", 1, 1)


def test_register_then_lookup_and_known_keys():
    """防回归：register 后进表可查，且 known_keys 有序稳定（进告警文案）。"""
    table = PriceTable()
    table.register("qwen", "vl", Price(0.002, 0.006))
    table.register("openai", "gpt", Price(0.01, 0.03))
    assert len(table) == 2
    assert table.known_keys() == (("openai", "gpt"), ("qwen", "vl"))
    assert table.price_of("qwen", "vl").completion_per_1k == pytest.approx(0.006)
    with pytest.raises(TypeError):
        table.register("x", "y", (0.01, 0.03))


def test_cost_of_rejects_negative_tokens():
    """防回归：负 token 会把成本冲减成负数，掩盖真实开销。"""
    table = PriceTable({("openai", "gpt"): Price(0.01, 0.03)})
    with pytest.raises(ValueError):
        table.cost_of("openai", "gpt", -1, 0)
    with pytest.raises(ValueError):
        table.cost_of("openai", "gpt", 0, -1)


def test_price_rejects_nonsense_unit_prices():
    """防回归：负 / NaN / 非数值单价会让所有成本计算静默失真。"""
    with pytest.raises(ValueError):
        Price(-0.01, 0.03)
    with pytest.raises(ValueError):
        Price(float("nan"), 0.03)
    with pytest.raises(ValueError):
        Price(float("inf"), 0.03)
    with pytest.raises(TypeError):
        Price("0.01", 0.03)
    assert Price(0.0, 0.0).prompt_per_1k == 0.0     # 免费模型是合法的


# ---------- 台账：累计与查询 ----------


def test_ledger_accumulates_total_and_by_provider():
    """防回归：总账与供应商分账必须同时算对，成本归因靠它。"""
    ledger = CostLedger()
    ledger.spend(usage("openai", 0.10), session_id="s1")
    ledger.spend(usage("openai", 0.05), session_id="s1")
    ledger.spend(usage("qwen", 0.02), session_id="s2")
    assert ledger.spent_total == pytest.approx(0.17)
    assert ledger.spent_by_provider("openai") == pytest.approx(0.15)
    assert ledger.spent_by_provider("qwen") == pytest.approx(0.02)
    assert ledger.spent_by_provider("never-used") == 0.0
    assert ledger.n_calls == 3
    assert ledger.n_sessions == 2
    assert ledger.by_session == {"s1": pytest.approx(0.15), "s2": pytest.approx(0.02)}


def test_no_cap_means_remaining_is_none_not_infinity():
    """防回归：没设 cap 时 remaining/ratio 返回 None——不假装"无限"，也便于序列化。"""
    ledger = CostLedger()
    ledger.spend(usage("openai", 0.1))
    assert ledger.remaining() is None
    assert ledger.remaining("openai") is None
    assert ledger.usage_ratio() is None
    assert ledger.usage_ratio("openai") is None
    assert ledger.should_warn() is False


def test_remaining_and_ratio_with_cap():
    """防回归：剩余额度与占用比例要手算可核对。"""
    ledger = CostLedger(cap_usd=1.0)
    assert ledger.remaining() == pytest.approx(1.0)
    ledger.spend(usage("openai", 0.25))
    assert ledger.remaining() == pytest.approx(0.75)
    assert ledger.usage_ratio() == pytest.approx(0.25)


def test_per_provider_remaining_and_ratio():
    """防回归：供应商维度要能单独查剩余与占用。"""
    ledger = CostLedger(per_provider_caps={"openai": 0.5})
    ledger.spend(usage("openai", 0.1))
    assert ledger.remaining("openai") == pytest.approx(0.4)
    assert ledger.usage_ratio("openai") == pytest.approx(0.2)
    assert ledger.remaining("qwen") is None


# ---------- 硬顶 ----------


def test_exactly_using_up_the_cap_is_allowed():
    """防回归：刚好用满不算超顶（含浮点累加：0.1×5 必须仍等于 0.5）。"""
    ledger = CostLedger(cap_usd=0.5)
    for _ in range(5):
        assert ledger.spend(usage("openai", 0.1)) == pytest.approx(0.1)
    assert ledger.spent_total == pytest.approx(0.5)
    assert ledger.remaining() == pytest.approx(0.0)


def test_overspend_by_one_cent_raises_and_does_not_book():
    """防回归：超顶必须抛 BudgetExceeded，且**失败不入账**（账目不能被污染）。"""
    ledger = CostLedger(cap_usd=0.5)
    ledger.spend(usage("openai", 0.5), session_id="s1")
    with pytest.raises(BudgetExceeded) as exc:
        ledger.spend(usage("openai", 0.01), session_id="s1")
    assert exc.value.kind == "total_usd"
    assert exc.value.limit == pytest.approx(0.5)
    assert exc.value.used == pytest.approx(0.51)
    assert "0.51" in str(exc.value)
    assert ledger.spent_total == pytest.approx(0.5)          # 未变
    assert ledger.spent_by_provider("openai") == pytest.approx(0.5)
    assert ledger.n_calls == 1
    assert ledger.by_session == {"s1": pytest.approx(0.5)}   # 场次样本也没被污染


def test_per_provider_cap_is_independent():
    """防回归：A 供应商顶破不该影响 B 供应商，也不该动总账。"""
    ledger = CostLedger(cap_usd=10.0, per_provider_caps={"openai": 0.3, "qwen": 0.5})
    ledger.spend(usage("openai", 0.3))
    ledger.spend(usage("qwen", 0.1))
    with pytest.raises(BudgetExceeded) as exc:
        ledger.spend(usage("openai", 0.01))
    assert exc.value.kind == "provider_usd:openai"
    assert exc.value.limit == pytest.approx(0.3)
    assert exc.value.used == pytest.approx(0.31)
    assert ledger.spent_by_provider("openai") == pytest.approx(0.3)   # 未入账
    assert ledger.spent_total == pytest.approx(0.4)                   # 未入账
    assert ledger.spend(usage("qwen", 0.05)) == pytest.approx(0.05)   # qwen 仍可花
    assert ledger.spent_by_provider("qwen") == pytest.approx(0.15)    # 顶破的是 openai


def test_zero_cap_blocks_everything_and_warns():
    """防回归：零预算必须立刻告警且一分钱都花不出去。"""
    ledger = CostLedger(cap_usd=0.0)
    assert ledger.remaining() == 0.0
    assert ledger.usage_ratio() == 1.0
    assert ledger.should_warn() is True
    with pytest.raises(BudgetExceeded):
        ledger.spend(usage("openai", 0.0001))
    assert ledger.spent_total == 0.0


def test_budget_exceeded_is_ruipin_error():
    """防回归：预算异常必须走项目内错误体系，便于统一捕获与序列化。"""
    ledger = CostLedger(cap_usd=0.1)
    with pytest.raises(RuipinError):
        ledger.spend(usage("openai", 1.0))


def test_spend_rejects_bad_cost():
    """防回归：负 / NaN 成本会冲减账目，必须拒绝入账。"""
    ledger = CostLedger()
    with pytest.raises(ValueError):
        ledger.spend(usage("openai", -0.1))
    with pytest.raises(ValueError):
        ledger.spend(usage("openai", float("nan")))
    assert ledger.spent_total == 0.0
    assert ledger.n_calls == 0


# ---------- 推算 ----------


def test_project_is_none_without_session_samples():
    """防回归：没有场次样本时返回 None，绝不返回 0 假装免费。"""
    ledger = CostLedger()
    assert ledger.project(10) is None
    assert ledger.mean_per_session is None
    ledger.spend(usage("openai", 0.2))          # 没有 session_id，不算场次样本
    assert ledger.n_sessions == 0
    assert ledger.project(10) is None


def test_project_uses_mean_per_session():
    """防回归：推算值必须手算可核对 —— 均值 0.3 × 10 场 = 3.0。"""
    ledger = CostLedger()
    ledger.spend(usage("openai", 0.2), session_id="s1")
    ledger.spend(usage("openai", 0.6), session_id="s2")
    assert ledger.mean_per_session == pytest.approx(0.4)
    assert ledger.project(10) == pytest.approx(4.0)
    assert ledger.project(1) == pytest.approx(0.4)
    assert ledger.project(0) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        ledger.project(-1)


def test_same_session_id_accumulates_into_one_sample():
    """防回归：同一场的多次调用要合并成一个样本，否则单场成本被稀释。"""
    ledger = CostLedger()
    ledger.spend(usage("openai", 0.10), session_id="s1")
    ledger.spend(usage("qwen", 0.05), session_id="s1")
    ledger.spend(usage("openai", 0.10), session_id="s2")
    assert ledger.n_sessions == 2
    assert ledger.mean_per_session == pytest.approx(0.125)   # (0.15+0.10)/2


# ---------- warn_ratio ----------


def test_warn_at_eighty_percent_does_not_block():
    """防回归：到 80% 只告警不阻断——阻断是硬顶的事，warn 不许拦业务。"""
    ledger = CostLedger(cap_usd=0.5, warn_ratio=0.8)
    ledger.spend(usage("openai", 0.3))
    assert ledger.should_warn() is False
    assert ledger.warn_targets() == ()
    ledger.spend(usage("openai", 0.1))          # 累计 0.4 = 80%
    assert ledger.should_warn() is True
    assert ledger.warn_targets() == ("total",)
    assert ledger.spend(usage("openai", 0.05)) == pytest.approx(0.05)  # 不阻断
    assert ledger.spent_total == pytest.approx(0.45)


def test_warn_ratio_is_configurable_and_provider_aware():
    """防回归：warn 阈值可调，且供应商维度独立告警（归因要看得出是谁）。"""
    tight = CostLedger(cap_usd=1.0, warn_ratio=0.5)
    tight.spend(usage("openai", 0.5))
    assert tight.should_warn() is True
    loose = CostLedger(cap_usd=1.0, warn_ratio=0.9)
    loose.spend(usage("openai", 0.5))
    assert loose.should_warn() is False
    per_provider = CostLedger(per_provider_caps={"qwen": 1.0})
    per_provider.spend(usage("qwen", 0.85))
    assert per_provider.should_warn() is True
    assert per_provider.warn_targets() == ("qwen",)


def test_invalid_ledger_config_rejected():
    """防回归：非法 cap / warn_ratio 必须构造时就拒绝，而不是运行时静默失效。"""
    with pytest.raises(ValueError):
        CostLedger(cap_usd=-1.0)
    with pytest.raises(ValueError):
        CostLedger(cap_usd=float("nan"))
    with pytest.raises(ValueError):
        CostLedger(per_provider_caps={"openai": -0.5})
    with pytest.raises(ValueError):
        CostLedger(warn_ratio=1.5)
    with pytest.raises(ValueError):
        CostLedger(warn_ratio=-0.1)
    assert CostLedger(cap_usd=1.0, warn_ratio=1.0).should_warn() is False


def test_as_dict_is_json_serializable():
    """防回归：台账视图要能直接进日志 / 告警 payload（allow_nan=False）。"""
    ledger = CostLedger(cap_usd=1.0, per_provider_caps={"openai": 0.5})
    ledger.spend(usage("openai", 0.45), session_id="s1")
    payload = ledger.as_dict()
    text = json.dumps(payload, allow_nan=False, ensure_ascii=False)
    loaded = json.loads(text)
    assert loaded["spent_total"] == pytest.approx(0.45)
    assert loaded["cap_usd"] == 1.0
    assert loaded["per_provider_caps"] == {"openai": 0.5}
    assert loaded["should_warn"] is True          # 0.45/0.5=0.9 与 0.45/1.0 比较
    assert loaded["mean_per_session"] == pytest.approx(0.45)
