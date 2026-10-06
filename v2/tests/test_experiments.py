"""实验框架（`experiments/assign.py` / `registry.py` / `exposure.py`）的测试。

钉住的纪律（每条断言都对应一条不可违背的原则）：

1. **分桶确定且均匀** —— 同一 session 多次调用结果必须一致（否则用户会在变体间
   横跳）；1000 个 session 分 A/B 两桶各落在 40%–60%（否则实验组不可比）。
2. **改 salt 换桶** —— salt 是实验正交性的保证，失效要能被看见。
3. **"未纳入" ≠ "对照组"** —— 关闭实验 / 灰度外返回 `CONTROL`，且禁止真实变体占用它。
4. **关闭实验不抛错** —— 开关语义是"让实验消失"，不该制造线上失败。
5. **护栏优先于主指标** —— 主指标大涨但护栏被突破必须 `ROLLBACK`；多个护栏同时
   突破必须**全部**列出。
6. **样本不足不提前下结论** —— 主指标再好看，样本不够也一律 `INSUFFICIENT_DATA`。
7. **人工急停凌驾于统计** —— 急停后 `decide` 一律 `ROLLBACK`。
8. **数据缺失不放过** —— 注册了护栏却没上报值，`INSUFFICIENT_DATA` 而非"跳过"。
9. **曝光去重** —— `counts` 按去重 session 数、`raw_counts` 按原始条数，重复不覆盖。
10. **口径显式** —— 主指标判定按是否指定 `baseline_variant` 分派为绝对 / 相对口径；
    相对口径下基线缺失必须抛错（不退回绝对）、基线为 0 必须不可判定（不除零拿 inf）。
"""

from __future__ import annotations

import pickle

import pytest

from ruipin.experiments.assign import CONTROL, ExperimentSpec, assign
from ruipin.experiments.exposure import (
    ExposureRecord,
    ExposureSink,
    MemoryExposureLog,
)
from ruipin.experiments.registry import (
    GUARDRAIL_APPEAL_RATE,
    GUARDRAIL_CLOSED_RATE,
    Decision,
    ExperimentRegistration,
    ExperimentRegistry,
    GuardrailDirection,
    GuardrailMetric,
    MissingBaselineVariant,
    StopRule,
    UnknownExperiment,
    VariantObservation,
    Verdict,
)

VARIANTS = ("control", "treatment")


# ========== assign：确定性、均匀性、salt ==========


def test_assign_is_deterministic():
    """纪律：同一 session 多次调用必须同桶，否则用户会在口径间横跳。"""
    first = assign("exp", VARIANTS, "session-42", "salt-1")
    for _ in range(50):
        assert assign("exp", VARIANTS, "session-42", "salt-1") == first


def test_assign_uniform_across_1000_sessions():
    """纪律：不同 session 必须大致均匀落到两个变体（各 40%–60%），否则对照失效。"""
    counts = {"control": 0, "treatment": 0}
    for i in range(1000):
        counts[assign("exp", VARIANTS, f"session-{i}", "salt-1")] += 1
    assert counts["control"] + counts["treatment"] == 1000
    assert 400 <= counts["control"] <= 600, f"分桶偏斜：{counts}"
    assert 400 <= counts["treatment"] <= 600, f"分桶偏斜：{counts}"


def test_changing_salt_swaps_buckets():
    """纪律：salt 决定正交性，换 salt 必须换桶，否则两个实验其实是一个。"""
    salt_a = {f"s{i}": assign("exp", VARIANTS, f"s{i}", "salt-a") for i in range(1000)}
    salt_b = {f"s{i}": assign("exp", VARIANTS, f"s{i}", "salt-b") for i in range(1000)}
    assert salt_a != salt_b
    changed = sum(1 for k in salt_a if salt_a[k] != salt_b[k])
    assert changed > 100, f"换 salt 几乎没换桶，哈希没起作用：changed={changed}"


def test_rollout_zero_returns_control_for_all():
    """纪律：0% 灰度 = 谁都不参与，且返回的是"未纳入"而非某个变体。"""
    for i in range(50):
        assert assign("exp", VARIANTS, f"s{i}", "salt", rollout_pct=0) == CONTROL


def test_rollout_full_assigns_everyone():
    """纪律：100% 灰度下不应有人被排除。"""
    assigned = {assign("exp", VARIANTS, f"s{i}", "salt", rollout_pct=100) for i in range(200)}
    assert CONTROL not in assigned
    assert assigned <= set(VARIANTS)


def test_rollout_partial_excludes_some_but_not_all():
    """纪律：部分灰度要同时出现"纳入"和"未纳入"两种结果。"""
    results = {
        assign("exp", VARIANTS, f"s{i}", "salt", rollout_pct=50) for i in range(600)
    }
    assert CONTROL in results
    assert results - {CONTROL}, "50% 灰度不该一个都没纳入"


def test_disabled_returns_control_without_raising():
    """纪律：关闭实验一律 CONTROL，且即便配置非法也不得抛错（开关不该制造失败）。"""
    assert assign("exp", VARIANTS, "s1", "salt", enabled=False) == CONTROL
    # 关闭时即使 variants 非法，也必须安静返回 CONTROL。
    assert assign("exp", ("only-one",), "s1", "salt", enabled=False) == CONTROL
    assert assign("exp", VARIANTS, "s1", "salt", enabled=False, rollout_pct=999) == CONTROL


def test_assign_rejects_too_few_variants():
    """纪律：不足两个变体没有对照可言，必须抛错。"""
    for bad in ((), ("only",)):
        with pytest.raises(ValueError, match="至少需要 2 个变体"):
            assign("exp", bad, "s1", "salt")


def test_assign_rejects_control_sentinel_as_variant():
    """纪律：CONTROL 是哨兵，禁止真实变体占用，否则"未纳入"会被当对照组。"""
    with pytest.raises(ValueError, match="哨兵值"):
        assign("exp", ("control", CONTROL), "s1", "salt")


def test_assign_rejects_duplicate_variants():
    """纪律：重复变体让"分布是否均匀"失去意义，必须拒绝。"""
    with pytest.raises(ValueError, match="重复"):
        assign("exp", ("a", "a"), "s1", "salt")


def test_assign_rejects_empty_session():
    with pytest.raises(ValueError, match="session_id"):
        assign("exp", VARIANTS, "", "salt")


def test_assign_rejects_out_of_range_rollout():
    for bad in (-1, 101):
        with pytest.raises(ValueError, match="rollout_pct"):
            assign("exp", VARIANTS, "s1", "salt", rollout_pct=bad)


def test_spec_validates_fields():
    with pytest.raises(ValueError, match="experiment_id"):
        ExperimentSpec("", VARIANTS, "salt")
    with pytest.raises(ValueError, match="salt"):
        ExperimentSpec("exp", VARIANTS, "")
    with pytest.raises(ValueError, match="rollout_pct"):
        ExperimentSpec("exp", VARIANTS, "salt", rollout_pct=200)
    with pytest.raises(ValueError, match="至少需要 2 个变体"):
        ExperimentSpec("exp", ("x",), "salt")


def test_spec_assign_delegates():
    spec = ExperimentSpec("exp", VARIANTS, "salt", rollout_pct=100)
    expected = assign("exp", VARIANTS, "s1", "salt", rollout_pct=100)
    assert spec.assign("s1") == expected


# ========== registry：护栏与决策顺序 ==========


def _reg(
    guardrails=(),
    min_sample=5,
    target_lift=0.8,
    experiment_id="exp",
    primary_metric="quality",
    baseline_variant=None,
):
    return ExperimentRegistration(
        experiment_id=experiment_id,
        primary_metric=primary_metric,
        guardrails=guardrails,
        min_sample_per_variant=min_sample,
        stop_rule=StopRule(max_days=14, target_lift=target_lift),
        baseline_variant=baseline_variant,
    )


def _closed_rate(max_allowed=0.2):
    return GuardrailMetric("closed_rate", GuardrailDirection.MAX, max_allowed=max_allowed)


def test_guardrail_direction_is_normalized_from_string():
    """纪律：允许传字符串方向，但内部必须归一成枚举，比较才可靠。"""
    g = GuardrailMetric("x", "max", max_allowed=1.0)
    assert g.direction is GuardrailDirection.MAX


def test_guardrail_rejects_invalid_direction():
    with pytest.raises(ValueError):
        GuardrailMetric("x", "sideways", max_allowed=1.0)


def test_guardrail_max_requires_only_max_allowed():
    with pytest.raises(ValueError, match="只能填 max_allowed"):
        GuardrailMetric("x", GuardrailDirection.MAX, max_allowed=0.2, min_required=0.1)
    with pytest.raises(ValueError, match="只能填 max_allowed"):
        GuardrailMetric("x", GuardrailDirection.MAX)
    with pytest.raises(ValueError, match="只能填 max_allowed"):
        GuardrailMetric("x", GuardrailDirection.MAX, min_required=0.1)


def test_guardrail_min_requires_only_min_required():
    with pytest.raises(ValueError, match="只能填 min_required"):
        GuardrailMetric("x", GuardrailDirection.MIN, max_allowed=0.2, min_required=0.1)
    with pytest.raises(ValueError, match="只能填 min_required"):
        GuardrailMetric("x", GuardrailDirection.MIN)
    with pytest.raises(ValueError, match="只能填 min_required"):
        GuardrailMetric("x", GuardrailDirection.MIN, max_allowed=0.2)


def test_guardrail_name_required():
    with pytest.raises(ValueError, match="name 不能为空"):
        GuardrailMetric("", GuardrailDirection.MAX, max_allowed=1.0)


def test_guardrail_threshold_and_breach_semantics():
    """纪律：低于/高于阈值的判定方向不能反，超幅必须是正数。"""
    upper = _closed_rate(0.2)
    assert upper.threshold == 0.2
    assert upper.breached_by(0.25) == pytest.approx(0.05)
    assert upper.breached_by(0.2) is None
    assert upper.breached_by(0.1) is None

    lower = GuardrailMetric("completion", GuardrailDirection.MIN, min_required=0.6)
    assert lower.threshold == 0.6
    assert lower.breached_by(0.5) == pytest.approx(0.1)
    assert lower.breached_by(0.6) is None
    assert lower.breached_by(0.9) is None


def test_stop_rule_validation():
    with pytest.raises(ValueError, match="max_days"):
        StopRule(max_days=0, target_lift=0.5)
    with pytest.raises(ValueError, match="target_lift"):
        StopRule(max_days=7, target_lift=0.0)


def test_registration_requires_ids():
    with pytest.raises(ValueError, match="experiment_id"):
        _reg(experiment_id="")
    with pytest.raises(ValueError, match="primary_metric"):
        _reg(primary_metric="")


def test_registry_rejects_duplicate_registration():
    """纪律：预注册一次性，重复覆盖会让实验中途悄悄换口径。"""
    reg = ExperimentRegistry()
    reg.register(_reg())
    with pytest.raises(ValueError, match="已注册"):
        reg.register(_reg())


def test_registry_rejects_small_min_sample():
    reg = ExperimentRegistry()
    with pytest.raises(ValueError, match="min_sample_per_variant"):
        reg.register(_reg(min_sample=0))


def test_registry_rejects_duplicate_guardrail_names():
    reg = ExperimentRegistry()
    dup = (_closed_rate(0.2), _closed_rate(0.3))
    with pytest.raises(ValueError, match="重名"):
        reg.register(_reg(guardrails=dup))


def test_registry_get_unknown_raises_unknown_experiment():
    reg = ExperimentRegistry()
    reg.register(_reg())
    with pytest.raises(UnknownExperiment) as excinfo:
        reg.get("nope")
    assert "nope" in str(excinfo.value)
    assert isinstance(excinfo.value, KeyError), "查表未命中在语义上是 KeyError"


def test_get_unknown_on_empty_registry_has_no_known_list():
    """空注册表时异常消息不该硬拼一个空括号（known 为空这条分支要走到）。"""
    reg = ExperimentRegistry()
    with pytest.raises(UnknownExperiment) as excinfo:
        reg.get("nope")
    assert "已注册" not in str(excinfo.value)
    assert excinfo.value.known == ()


def test_unknown_experiment_pickles_roundtrip():
    """异常要能跨进程传播，pickle 往返后属性与消息不变。"""
    err = UnknownExperiment("exp-x", ("a", "b"))
    restored = pickle.loads(pickle.dumps(err))
    assert restored.experiment_id == "exp-x"
    assert restored.known == ("a", "b")
    assert str(restored) == str(err)


def test_has_guardrails_visible_when_empty():
    """纪律：空护栏合法，但必须能被看见（否则会等于"有护栏且全过"）。"""
    reg = ExperimentRegistry()
    reg.register(_reg(guardrails=()))  # 空护栏允许注册
    assert reg.has_guardrails("exp") is False
    assert reg.get("exp").has_guardrails is False

    reg.register(_reg(experiment_id="guarded", guardrails=(_closed_rate(),)))
    assert reg.has_guardrails("guarded") is True


def test_experiment_ids_preserves_order():
    reg = ExperimentRegistry()
    reg.register(_reg(experiment_id="a"))
    reg.register(_reg(experiment_id="b"))
    assert reg.experiment_ids == ("a", "b")


def test_decide_insufficient_data_even_if_primary_metric_is_great():
    """纪律：样本不足时主指标再好看也不提前下结论（不许被漂亮数字带跑）。"""
    reg = ExperimentRegistry()
    reg.register(_reg(min_sample=10, guardrails=(_closed_rate(),), target_lift=0.5))
    obs = {
        "control": VariantObservation(n=3, primary_metric=0.99, guardrails={"closed_rate": 0.0}),
        "treatment": VariantObservation(n=10, primary_metric=0.99, guardrails={"closed_rate": 0.0}),
    }
    decision = reg.decide("exp", obs)
    assert decision.verdict is Verdict.INSUFFICIENT_DATA
    assert decision.samples_needed == {"control": 7}, "必须写明还差多少样本"


def test_decide_rollback_wins_over_great_primary_metric():
    """纪律（核心）：主指标大涨但护栏被突破 → 必须 ROLLBACK，红线不能拿效果换。"""
    reg = ExperimentRegistry()
    reg.register(_reg(guardrails=(_closed_rate(0.2),), target_lift=0.8))
    obs = {
        "control": VariantObservation(n=10, primary_metric=0.9, guardrails={"closed_rate": 0.5}),
        "treatment": VariantObservation(n=10, primary_metric=0.95, guardrails={"closed_rate": 0.1}),
    }
    decision = reg.decide("exp", obs)
    assert decision.verdict is Verdict.ROLLBACK
    assert len(decision.triggered) == 1
    breach = decision.triggered[0]
    assert breach.variant == "control"
    assert breach.name == "closed_rate"
    assert breach.excess == pytest.approx(0.3)


def test_decide_lists_all_breached_guardrails_not_just_first():
    """纪律：多个护栏同时破要全部列出，只报第一条会让人修完又踩雷。"""
    reg = ExperimentRegistry()
    reg.register(
        _reg(
            guardrails=(
                _closed_rate(0.2),
                GuardrailMetric("appeal_rate", GuardrailDirection.MAX, max_allowed=0.05),
            )
        )
    )
    obs = {
        "control": VariantObservation(
            n=10,
            primary_metric=0.9,
            guardrails={"closed_rate": 0.5, "appeal_rate": 0.2},
        ),
        "treatment": VariantObservation(
            n=10,
            primary_metric=0.9,
            guardrails={"closed_rate": 0.5, "appeal_rate": 0.1},
        ),
    }
    decision = reg.decide("exp", obs)
    assert decision.verdict is Verdict.ROLLBACK
    names = {(b.variant, b.name) for b in decision.triggered}
    assert names == {
        ("control", "closed_rate"),
        ("control", "appeal_rate"),
        ("treatment", "closed_rate"),
        ("treatment", "appeal_rate"),
    }, f"必须列出全部被突破的护栏：{names}"


def test_decide_ship_when_all_guardrails_pass_and_target_met():
    reg = ExperimentRegistry()
    reg.register(_reg(guardrails=(_closed_rate(),), target_lift=0.8))
    obs = {
        "control": VariantObservation(n=10, primary_metric=0.7, guardrails={"closed_rate": 0.1}),
        "treatment": VariantObservation(n=10, primary_metric=0.85, guardrails={"closed_rate": 0.1}),
    }
    decision = reg.decide("exp", obs)
    assert decision.verdict is Verdict.SHIP
    assert decision.triggered == ()
    assert "treatment" in decision.reason
    assert "绝对口径" in decision.reason


def test_decide_continue_when_target_not_met():
    reg = ExperimentRegistry()
    reg.register(_reg(guardrails=(_closed_rate(),), target_lift=0.9))
    obs = {
        "control": VariantObservation(n=10, primary_metric=0.7, guardrails={"closed_rate": 0.1}),
        "treatment": VariantObservation(n=10, primary_metric=0.85, guardrails={"closed_rate": 0.1}),
    }
    decision = reg.decide("exp", obs)
    assert decision.verdict is Verdict.CONTINUE
    assert decision.samples_needed == {}
    assert "绝对口径" in decision.reason, "无基线时必须回显用的是绝对口径"


# ---------- 相对口径（baseline_variant 非空） ----------


def test_relative_lift_not_met_reports_baseline():
    """纪律：相对口径下未达目标要判 CONTINUE，且 reason 必须点明相对的是哪个基线。"""
    reg = ExperimentRegistry()
    reg.register(_reg(target_lift=0.10, baseline_variant="control"))
    obs = {
        "control": VariantObservation(n=10, primary_metric=0.60),
        "treatment": VariantObservation(n=10, primary_metric=0.63),
    }
    decision = reg.decide("exp", obs)
    assert decision.verdict is Verdict.CONTINUE
    assert "相对口径" in decision.reason and "control" in decision.reason


def test_relative_lift_met_ships():
    """纪律：相对抬升 (0.69-0.60)/0.60=0.15 ≥ 目标 0.10 → SHIP，且口径写明。"""
    reg = ExperimentRegistry()
    reg.register(_reg(target_lift=0.10, baseline_variant="control"))
    obs = {
        "control": VariantObservation(n=10, primary_metric=0.60),
        "treatment": VariantObservation(n=10, primary_metric=0.69),
    }
    decision = reg.decide("exp", obs)
    assert decision.verdict is Verdict.SHIP
    assert "相对口径" in decision.reason and "control" in decision.reason


def test_relative_uses_best_excluding_baseline():
    """纪律：相对口径的最优变体必须排除基线自身，否则基线会被拿去和自己比。"""
    reg = ExperimentRegistry()
    reg.register(_reg(target_lift=0.10, baseline_variant="control"))
    obs = {
        # 基线自身值最高，但它不能被当作"最优变体"去和它自己算 0 抬升。
        "control": VariantObservation(n=10, primary_metric=0.90),
        "treatment": VariantObservation(n=10, primary_metric=0.72),
    }
    decision = reg.decide("exp", obs)
    # 抬升 = (0.72-0.90)/0.90 < 0 → 未达标
    assert decision.verdict is Verdict.CONTINUE
    assert "treatment" in decision.reason


def test_relative_zero_baseline_is_insufficient_not_ship():
    """纪律：基线为 0 时相对抬升无定义，必须不可判定，绝不除零拿 inf 当达标。"""
    reg = ExperimentRegistry()
    reg.register(_reg(target_lift=0.10, baseline_variant="control"))
    obs = {
        "control": VariantObservation(n=10, primary_metric=0.0),
        "treatment": VariantObservation(n=10, primary_metric=0.9),
    }
    decision = reg.decide("exp", obs)
    assert decision.verdict is Verdict.INSUFFICIENT_DATA
    assert decision.verdict is not Verdict.SHIP
    assert "0" in decision.reason


def test_relative_with_only_baseline_is_insufficient():
    """纪律：只有基线、没有可比对象时无法计算相对抬升，不可判定。"""
    reg = ExperimentRegistry()
    reg.register(_reg(target_lift=0.10, baseline_variant="control"))
    obs = {"control": VariantObservation(n=10, primary_metric=0.6)}
    decision = reg.decide("exp", obs)
    assert decision.verdict is Verdict.INSUFFICIENT_DATA


def test_relative_missing_baseline_raises_not_falls_back():
    """纪律：基线不在观测里是配置错误，必须抛明确异常，不许悄悄退回绝对口径。"""
    reg = ExperimentRegistry()
    reg.register(_reg(target_lift=0.10, baseline_variant="control"))
    obs = {"treatment": VariantObservation(n=10, primary_metric=0.9)}
    with pytest.raises(MissingBaselineVariant) as excinfo:
        reg.decide("exp", obs)
    assert "control" in str(excinfo.value)
    assert excinfo.value.baseline_variant == "control"
    assert excinfo.value.observed == ("treatment",)


def test_missing_baseline_variant_pickles_roundtrip():
    err = MissingBaselineVariant("control", ("treatment",))
    restored = pickle.loads(pickle.dumps(err))
    assert restored.baseline_variant == "control"
    assert restored.observed == ("treatment",)
    assert str(restored) == str(err)


def test_relative_baseline_missing_when_no_observations_at_all():
    """空观测 + 相对口径：仍属基线缺失这一配置错误，异常里不带观测清单。"""
    reg = ExperimentRegistry()
    reg.register(_reg(target_lift=0.10, baseline_variant="control"))
    with pytest.raises(MissingBaselineVariant) as excinfo:
        reg.decide("exp", {})
    assert excinfo.value.observed == ()
    assert "当前观测变体" not in str(excinfo.value)


def test_registration_rejects_empty_baseline_string():
    """纪律：基线要么 None（绝对口径）、要么给出变体名，空串会让口径含糊。"""
    with pytest.raises(ValueError, match="baseline_variant"):
        _reg(baseline_variant="")


def test_decide_missing_guardrail_data_is_insufficient_not_pass():
    """纪律：缺测 ≠ 未超标；注册了护栏却没上报值必须不可判定，不能跳过。"""
    reg = ExperimentRegistry()
    reg.register(_reg(guardrails=(_closed_rate(),)))
    obs = {
        "control": VariantObservation(n=10, primary_metric=0.9, guardrails={}),
        "treatment": VariantObservation(n=10, primary_metric=0.9, guardrails={"closed_rate": 0.1}),
    }
    decision = reg.decide("exp", obs)
    assert decision.verdict is Verdict.INSUFFICIENT_DATA
    assert "closed_rate" in decision.reason


def test_missing_custom_guardrail_without_convenience_field():
    """纪律：非便捷名的护栏缺测同样是"缺测"，必须不可判定（不能当成 0）。"""
    reg = ExperimentRegistry()
    reg.register(
        _reg(guardrails=(GuardrailMetric("toxicity", GuardrailDirection.MAX, max_allowed=0.1),))
    )
    obs = {"control": VariantObservation(n=10, primary_metric=0.9)}
    decision = reg.decide("exp", obs)
    assert decision.verdict is Verdict.INSUFFICIENT_DATA
    assert "toxicity" in decision.reason


def test_decide_rejects_empty_observations():
    reg = ExperimentRegistry()
    reg.register(_reg())
    decision = reg.decide("exp", {})
    assert decision.verdict is Verdict.INSUFFICIENT_DATA


def test_decide_uses_convenience_closed_and_appeal_rate_fields():
    """便捷字段只在显式提供时兜底；这里验证两个便捷名都能被读到。"""
    reg = ExperimentRegistry()
    reg.register(
        _reg(
            guardrails=(
                GuardrailMetric(GUARDRAIL_CLOSED_RATE, GuardrailDirection.MAX, max_allowed=0.2),
                GuardrailMetric(GUARDRAIL_APPEAL_RATE, GuardrailDirection.MAX, max_allowed=0.05),
            )
        )
    )
    obs = {
        "control": VariantObservation(
            n=10, primary_metric=0.9, closed_rate=0.1, appeal_rate=0.02
        )
    }
    # 两护栏都满足 → 主指标达标 → SHIP
    assert reg.decide("exp", obs).verdict is Verdict.SHIP
    # 只把 appeal_rate 抬高 → 必须 ROLLBACK，证明便捷字段确实被读到了
    bad = {"control": VariantObservation(n=10, primary_metric=0.9, closed_rate=0.1, appeal_rate=0.5)}
    assert reg.decide("exp", bad).verdict is Verdict.ROLLBACK


def test_force_rollback_overrides_statistics():
    """纪律：人工急停优先于一切统计结论，之后 decide 一律 ROLLBACK。"""
    reg = ExperimentRegistry()
    reg.register(_reg(guardrails=(_closed_rate(),), target_lift=0.5))
    perfect = {
        "control": VariantObservation(n=10, primary_metric=0.99, guardrails={"closed_rate": 0.0})
    }
    assert reg.decide("exp", perfect).verdict is Verdict.SHIP
    reg.force_rollback("exp", "线上舆情告警")
    assert reg.is_forced_rollback("exp") is True
    assert reg.forced_rollback_reason("exp") == "线上舆情告警"
    decision = reg.decide("exp", perfect)
    assert decision.verdict is Verdict.ROLLBACK
    assert "人工急停" in decision.reason


def test_force_rollback_requires_known_experiment_and_reason():
    reg = ExperimentRegistry()
    reg.register(_reg())
    with pytest.raises(UnknownExperiment):
        reg.force_rollback("ghost", "whatever")
    with pytest.raises(ValueError, match="原因"):
        reg.force_rollback("exp", "")


def test_is_forced_rollback_false_by_default():
    reg = ExperimentRegistry()
    reg.register(_reg())
    assert reg.is_forced_rollback("exp") is False
    assert reg.forced_rollback_reason("exp") is None


def test_variant_observation_rejects_negative_n():
    with pytest.raises(ValueError, match="不能为负"):
        VariantObservation(n=-1, primary_metric=0.5)


# ========== exposure：去重计数 ==========


class _Clock:
    """确定性时钟：每次调用自增 1 秒（纪律：时间必须可注入、可复现）。"""

    def __init__(self, start: float = 0.0) -> None:
        self._t = start

    def __call__(self) -> float:
        self._t += 1.0
        return self._t


class _RecordingSink:
    def __init__(self) -> None:
        self.records: list[ExposureRecord] = []

    def record(self, record: ExposureRecord) -> None:
        self.records.append(record)


def test_exposure_uses_injected_clock():
    clock = _Clock(start=100.0)
    log = MemoryExposureLog(clock=clock)
    first = log.record("s1", "exp", "control")
    second = log.record("s2", "exp", "treatment")
    assert first.ts == 101.0
    assert second.ts == 102.0


def test_exposure_rejects_bad_clock():
    with pytest.raises(TypeError, match="clock"):
        MemoryExposureLog(clock=123)  # type: ignore[arg-type]


def test_counts_dedup_vs_raw_counts():
    """纪律：`counts` 按去重 session 数（正确口径），`raw_counts` 按原始条数（污染口径）。"""
    log = MemoryExposureLog(clock=_Clock())
    log.record("s1", "exp", "control")
    log.record("s1", "exp", "control")  # 重复曝光
    log.record("s2", "exp", "control")
    log.record("s3", "exp", "treatment")
    assert log.counts("exp") == {"control": 2, "treatment": 1}
    assert log.raw_counts("exp") == {"control": 3, "treatment": 1}


def test_duplicate_exposure_does_not_overwrite():
    """纪律：append-only，重复曝光不得覆盖既有记录，但重复事实要能被查到。"""
    log = MemoryExposureLog(clock=_Clock())
    log.record("s1", "exp", "control", metadata={"src": "first"})
    log.record("s1", "exp", "treatment", metadata={"src": "second"})
    records = log.for_session("s1")
    assert [r.variant for r in records] == ["control", "treatment"], "首次记录不能被覆盖"
    assert records[0].metadata == {"src": "first"}
    assert log.duplicate_exposures() == {("s1", "exp"): 1}


def test_duplicate_exposures_filter_by_experiment():
    log = MemoryExposureLog(clock=_Clock())
    log.record("s1", "exp-a", "control")
    log.record("s1", "exp-a", "control")
    log.record("s1", "exp-b", "x")
    log.record("s1", "exp-b", "x")
    assert log.duplicate_exposures("exp-a") == {("s1", "exp-a"): 1}
    assert log.duplicate_exposures() == {("s1", "exp-a"): 1, ("s1", "exp-b"): 1}


def test_exposure_filters_and_records_property():
    log = MemoryExposureLog(clock=_Clock())
    log.record("s1", "exp-a", "control")
    log.record("s2", "exp-b", "x")
    assert [r.experiment_id for r in log.for_session("s1")] == ["exp-a"]
    assert [r.session_id for r in log.for_experiment("exp-b")] == ["s2"]
    assert len(log.records) == 2


def test_exposure_rejects_empty_ids():
    log = MemoryExposureLog(clock=_Clock())
    with pytest.raises(ValueError, match="session_id"):
        log.record("", "exp", "control")
    with pytest.raises(ValueError, match="experiment_id"):
        log.record("s1", "", "control")


def test_exposure_metadata_is_copied_not_aliased():
    log = MemoryExposureLog(clock=_Clock())
    source = {"k": "v"}
    log.record("s1", "exp", "control", metadata=source)
    source["k"] = "mutated"
    assert log.records[0].metadata == {"k": "v"}


def test_exposure_forwards_to_sink():
    """纪律：真实部署靠 ExposureSink 落库，转发必须发生。"""
    sink = _RecordingSink()
    assert isinstance(sink, ExposureSink), "记录型 sink 应满足持久化端口"
    log = MemoryExposureLog(clock=_Clock(), sink=sink)
    log.record("s1", "exp", "control")
    assert len(sink.records) == 1
    assert sink.records[0].session_id == "s1"


def test_exposure_without_sink_still_works():
    log = MemoryExposureLog(clock=_Clock())
    log.record("s1", "exp", "control")
    assert len(log.records) == 1


def test_counts_empty_experiment_is_empty_mapping():
    """纪律：没有曝光就返回空映射，不合成任何默认值。"""
    log = MemoryExposureLog(clock=_Clock())
    assert log.counts("ghost") == {}
    assert log.raw_counts("ghost") == {}


def test_raw_counts_scoped_to_its_experiment():
    """纪律：两个计数口径都必须只统计本次实验，不得把别的实验算进来。"""
    log = MemoryExposureLog(clock=_Clock())
    log.record("s1", "exp-a", "control")
    log.record("s2", "exp-b", "control")
    log.record("s3", "exp-b", "control")
    assert log.raw_counts("exp-a") == {"control": 1}
    assert log.raw_counts("exp-b") == {"control": 2}
