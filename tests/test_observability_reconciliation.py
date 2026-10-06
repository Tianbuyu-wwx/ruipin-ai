"""每日对账（`observability/reconciliation.py`）的测试。

方案 §12.3「每日对账」：会话数 × 平均成本 vs 账单；偏差 >10% 告警。

本文件把三件最容易出错的事钉死：

1. **偏差带符号**：超收与少收方向相反，测试断言两者 deviation 符号相反——
   若有人日后加了 `abs()`，方向信息就丢了，这里会红。
2. **算不出来 ≠ 没偏差**：`billed≤0` / `sessions≤0` → `UNKNOWN`（不是 OK，也不是
   偏差 0），且 UNKNOWN 时 `deviation is None`。
3. **UNKNOWN 也要发告警**：对不上账但算不出偏差本身就该被人看到，不能静默。
"""

from __future__ import annotations

import json

import pytest

from ruipin.observability.alerts import MemoryAlertSink, Severity
from ruipin.observability.reconciliation import (
    DEFAULT_TOLERANCE,
    RULE_DAILY_RECONCILIATION,
    ReconcileReport,
    ReconcileVerdict,
    Reconciler,
    reconcile,
)


# ======================================================================
# 三态判定
# ======================================================================


def test_ok_when_within_tolerance():
    """防回归：偏差在容差内 → OK，且 deviation 如实给出（不许省掉）。"""
    report = reconcile(sessions=100, mean_cost_usd=1.0, billed_usd=100.0, ts=3.0)
    assert isinstance(report, ReconcileReport)
    assert report.verdict is ReconcileVerdict.OK
    assert report.expected_usd == pytest.approx(100.0)
    assert report.billed_usd == 100.0
    assert report.deviation == pytest.approx(0.0)
    assert report.tolerance == DEFAULT_TOLERANCE == 0.10
    assert report.ts == 3.0
    assert DEFAULT_TOLERANCE == 0.10


def test_deviation_positive_direction_underbilled():
    """★ 纪律：预期 > 账单 → deviation 为正（少收/少计），带符号。"""
    report = reconcile(sessions=100, mean_cost_usd=1.0, billed_usd=80.0)
    assert report.verdict is ReconcileVerdict.DEVIATION
    assert report.deviation is not None and report.deviation > 0
    assert report.deviation == pytest.approx(0.25)


def test_deviation_negative_direction_overbilled():
    """★ 纪律：账单 > 预期 → deviation 为负（多收/多计）。"""
    report = reconcile(sessions=100, mean_cost_usd=1.0, billed_usd=125.0)
    assert report.verdict is ReconcileVerdict.DEVIATION
    assert report.deviation is not None and report.deviation < 0
    assert report.deviation == pytest.approx(-0.2)


def test_deviation_sign_directions_are_opposite():
    """★ 纪律：两个方向的 deviation 必须符号相反——绝不能取绝对值把符号抹掉。

    少收要查计费遗漏、多收要查重复计费，动作不同；只报一个"差了 25%"没法行动。
    """
    under = reconcile(sessions=100, mean_cost_usd=1.0, billed_usd=80.0).deviation
    over = reconcile(sessions=100, mean_cost_usd=1.0, billed_usd=125.0).deviation
    assert under is not None and over is not None
    assert under > 0 > over
    # 同为 20% / 25% 量级的两个方向，若被 abs 化就会变成同一个正数
    assert under != -over


def test_exactly_tolerance_is_ok_not_deviation():
    """★ 边界口径钉死：正好 10% 不算偏差（严格 `>`）。"""
    report = reconcile(sessions=110, mean_cost_usd=1.0, billed_usd=100.0, tolerance=0.10)
    assert report.deviation == pytest.approx(0.10)
    assert report.verdict is ReconcileVerdict.OK


def test_just_over_tolerance_is_deviation():
    """防回归：刚过 10% 就要报（10.55% > 10%）。"""
    report = reconcile(
        sessions=110, mean_cost_usd=1.005, billed_usd=100.0, tolerance=0.10
    )
    assert report.deviation == pytest.approx(0.1055)
    assert report.verdict is ReconcileVerdict.DEVIATION


def test_billed_zero_is_unknown_without_division():
    """★ 纪律：账单为 0 → UNKNOWN，不除零、不拿 inf 当偏差、不读成"无偏差"。"""
    report = reconcile(sessions=100, mean_cost_usd=1.0, billed_usd=0.0)
    assert report.verdict is ReconcileVerdict.UNKNOWN
    assert report.deviation is None  # 不是 0！0 会被读成"完全对上账"
    assert "账单" in report.reason


def test_billed_negative_is_unknown():
    """防回归：负账单同样判不了（退款/冲正等异常形态不该被算成正常偏差）。"""
    report = reconcile(sessions=100, mean_cost_usd=1.0, billed_usd=-5.0)
    assert report.verdict is ReconcileVerdict.UNKNOWN
    assert report.deviation is None


def test_zero_sessions_is_unknown():
    """★ 纪律：会话数为 0 → 预期金额无意义 → UNKNOWN（不拿 0 场当"没花超"）。"""
    report = reconcile(sessions=0, mean_cost_usd=1.0, billed_usd=100.0)
    assert report.verdict is ReconcileVerdict.UNKNOWN
    assert report.deviation is None
    assert "会话数" in report.reason


def test_negative_sessions_is_unknown():
    """防回归：负会话数属于脏数据，同样判不了，不能算出负的预期金额。"""
    report = reconcile(sessions=-3, mean_cost_usd=1.0, billed_usd=100.0)
    assert report.verdict is ReconcileVerdict.UNKNOWN
    assert report.deviation is None


# ======================================================================
# 构造校验（越界即抛，绝不静默）
# ======================================================================


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.0, 1.5])
def test_tolerance_out_of_range_rejected(bad):
    """防回归：容差必须在 (0,1)——0 等于"永远告警"，≥1 等于"永不告警"。"""
    with pytest.raises(ValueError):
        reconcile(sessions=1, mean_cost_usd=1.0, billed_usd=1.0, tolerance=bad)


def test_negative_mean_cost_rejected():
    """★ 纪律：成本不可能为负；负均值会让少收与多收互相抵消，账面假健康。"""
    with pytest.raises(ValueError):
        reconcile(sessions=10, mean_cost_usd=-0.5, billed_usd=10.0)


def test_sessions_must_be_int():
    """防回归：会话数必须是 int（`True` 也不行，bool 是 int 的子类）。"""
    with pytest.raises(TypeError):
        reconcile(sessions=True, mean_cost_usd=1.0, billed_usd=1.0)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        reconcile(sessions="10", mean_cost_usd=1.0, billed_usd=1.0)  # type: ignore[arg-type]


def test_non_finite_inputs_rejected():
    """防回归：NaN/ inf 会让比较与除法产出无意义的数，必须在入口拦住。"""
    with pytest.raises(ValueError):
        reconcile(sessions=1, mean_cost_usd=float("nan"), billed_usd=1.0)
    with pytest.raises(ValueError):
        reconcile(sessions=1, mean_cost_usd=1.0, billed_usd=float("inf"))
    with pytest.raises(ValueError):
        reconcile(sessions=1, mean_cost_usd=1.0, billed_usd=1.0, ts=float("nan"))


def test_report_to_dict_is_json_safe():
    """防回归：UNKNOWN 的 deviation=None 必须真的是 None（严格 JSON 也要过）。"""
    ok = reconcile(sessions=100, mean_cost_usd=1.0, billed_usd=100.0).to_dict()
    assert ok["verdict"] == "ok"

    unknown = reconcile(sessions=100, mean_cost_usd=1.0, billed_usd=0.0).to_dict()
    assert unknown["verdict"] == "unknown"
    assert unknown["deviation"] is None
    json.dumps(unknown, ensure_ascii=False, allow_nan=False)


# ======================================================================
# Reconciler：结论 → 告警
# ======================================================================


def test_reconciler_deviation_emits_single_critical():
    """★ 纪律：DEVIATION → 恰好 1 条 CRITICAL 告警，且自带方向与数值。"""
    sink = MemoryAlertSink()
    rec = Reconciler(sink, tolerance=0.10)
    report = rec.run(sessions=100, mean_cost_usd=1.0, billed_usd=80.0, ts=9.0)

    assert report.verdict is ReconcileVerdict.DEVIATION
    assert sink.count == 1
    alert = sink.alerts[0]
    assert alert.rule_id == RULE_DAILY_RECONCILIATION == "daily_reconciliation"
    assert alert.severity is Severity.CRITICAL
    assert alert.value == pytest.approx(0.25)  # 带符号的偏差
    assert alert.threshold == 0.10
    assert alert.ts == 9.0
    json.dumps(alert.to_dict(), ensure_ascii=False, allow_nan=False)


def test_reconciler_unknown_emits_single_warning():
    """★ 纪律：UNKNOWN **也要**发一条 WARNING——对不上账但算不出来同样要被看到。"""
    sink = MemoryAlertSink()
    rec = Reconciler(sink)
    report = rec.run(sessions=100, mean_cost_usd=1.0, billed_usd=0.0)

    assert report.verdict is ReconcileVerdict.UNKNOWN
    assert sink.count == 1
    alert = sink.alerts[0]
    assert alert.severity is Severity.WARNING
    assert alert.rule_id == RULE_DAILY_RECONCILIATION
    # 偏差算不出来：value 是 NaN，to_dict 会转 None（也不能让 JSON 抛）
    assert alert.to_dict()["value"] is None
    json.dumps(alert.to_dict(), ensure_ascii=False, allow_nan=False)


def test_reconciler_ok_emits_nothing():
    """防回归：OK 时静默（不 emit），避免把正常账目也刷成告警。"""
    sink = MemoryAlertSink()
    rec = Reconciler(sink)
    report = rec.run(sessions=100, mean_cost_usd=1.0, billed_usd=100.0)

    assert report.verdict is ReconcileVerdict.OK
    assert sink.count == 0
    assert sink.alerts == ()


def test_reconciler_validates_tolerance():
    """防回归：Reconciler 构造期也要拦越界容差（与 reconcile 同一口径）。"""
    sink = MemoryAlertSink()
    with pytest.raises(ValueError):
        Reconciler(sink, tolerance=0.0)
    with pytest.raises(ValueError):
        Reconciler(sink, tolerance=1.0)
    assert Reconciler(sink, tolerance=0.2).tolerance == 0.2
