"""每日对账（方案 §12.3「每日对账」）。

原文："会话数 × 平均成本 vs 账单；偏差 >10% 告警。"

口径与三态结论
--------------
把"我们以为花了多少"和"账单说花了多少"对一遍：

* `expected_usd = sessions × mean_cost_usd` —— 由埋点口径推出的**预期**金额。
* `deviation = (expected_usd - billed_usd) / billed_usd` —— **带符号**：正数表示
  预期 > 账单（**少收 / 少计成本**），负数表示账单 > 预期（**多收 / 多计成本**）。
  两个方向的处理动作完全不同（少收要查计费遗漏，多收要查重复计费），所以绝不能
  先取绝对值再比——那会把方向信息丢掉，只剩一个"差了 12%"。
* `|deviation| > tolerance` → `DEVIATION`；否则 `OK`。边界：**正好等于 10% 不算偏差**
  （严格 `>`），有测试钉住。
* 账单为 0/负、或会话数为 0 → `UNKNOWN`。这三态是"判不了"的一等公民：
  **绝不**除零、**绝不**把 inf 当偏差、**绝不**把"没数据"读成"无偏差"。

`UNKNOWN` 也要发告警
--------------------
对不上账但算不出偏差，本身就是需要人看的信号（比如账单接口返回了 0，很可能是
对接挂了而不是"这月免费"）。所以 `Reconciler` 在 `UNKNOWN` 时**也**emit 一条
**WARNING**，`DEVIATION` 时 emit **CRITICAL**，`OK` 时静默。这点有测试。

不硬编码时间
------------
`ts` 由调用方传入（默认 0.0，仅作占位）；本模块不调用 `time.time()`。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from ruipin.observability.alerts import Alert, AlertSink, Severity, _check_finite, _check_ts

__all__ = [
    "ReconcileVerdict",
    "ReconcileReport",
    "Reconciler",
    "reconcile",
    "RULE_DAILY_RECONCILIATION",
    "DEFAULT_TOLERANCE",
]

#: 对账告警的规则 id（进告警路由，须稳定）。
RULE_DAILY_RECONCILIATION = "daily_reconciliation"
#: 方案 §12.3：偏差 >10% 告警。
DEFAULT_TOLERANCE = 0.10


class ReconcileVerdict(str, Enum):
    """对账结论。`UNKNOWN` 是**一等公民**：算不出来不等于没偏差。"""

    OK = "ok"
    DEVIATION = "deviation"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ReconcileReport:
    """一次对账的结果。

    `deviation` 在 `UNKNOWN` 时为 `None`（**不是 0**）——0 会被读成"完全对上账"，
    正好把"账单为 0、算不出来"这种故障伪装成健康状态。
    """

    verdict: ReconcileVerdict
    expected_usd: float
    billed_usd: float
    deviation: Optional[float]
    tolerance: float
    reason: str
    ts: float

    def to_dict(self) -> dict[str, Any]:
        """JSON 安全视图（`deviation=None` 会如实序列化成 `null`）。"""
        return {
            "verdict": self.verdict.value,
            "expected_usd": self.expected_usd,
            "billed_usd": self.billed_usd,
            "deviation": self.deviation,
            "tolerance": self.tolerance,
            "reason": self.reason,
            "ts": self.ts,
        }


def reconcile(
    *,
    sessions: int,
    mean_cost_usd: float,
    billed_usd: float,
    tolerance: float = DEFAULT_TOLERANCE,
    ts: float = 0.0,
) -> ReconcileReport:
    """对一次账，返回三态结论。

    * `tolerance` 越界（≤0 或 ≥1）抛 `ValueError`——容差本身就是比例，0 会让
      "任何偏差都算超"（等于永远告警），≥1 会让"偏差 100% 都算正常"（等于永不告警）。
    * `mean_cost_usd < 0` 抛 `ValueError`——成本不可能为负，负均值会让"少收"与
      "多收"在求和时互相抵消，账面看着没事，实际两个方向都错了。
    * `billed_usd ≤ 0` → `UNKNOWN`（不除零）；`sessions ≤ 0` → `UNKNOWN`。
    """
    tolerance_value = _check_finite(tolerance, "tolerance")
    if not 0.0 < tolerance_value < 1.0:
        raise ValueError(f"tolerance 必须落在 (0, 1)，实际: {tolerance!r}")
    mean_cost = _check_finite(mean_cost_usd, "mean_cost_usd")
    if mean_cost < 0:
        raise ValueError(f"mean_cost_usd 不能为负，实际: {mean_cost_usd!r}")
    billed = _check_finite(billed_usd, "billed_usd")
    now = _check_ts(ts)

    if isinstance(sessions, bool) or not isinstance(sessions, int):
        raise TypeError(f"sessions 必须是 int：{sessions!r}")

    expected = sessions * mean_cost

    # 判不了的第一类：账单不可用。先查这个——因为算偏差要以账单为分母。
    if billed <= 0:
        return ReconcileReport(
            verdict=ReconcileVerdict.UNKNOWN,
            expected_usd=expected,
            billed_usd=billed,
            deviation=None,
            tolerance=tolerance_value,
            reason=f"账单为 0 或负（billed_usd={billed!r}），无法计算偏差",
            ts=now,
        )
    # 判不了的又一 类：没有会话，预期金额无意义（不拿 0 场当"没花超"）。
    if sessions <= 0:
        return ReconcileReport(
            verdict=ReconcileVerdict.UNKNOWN,
            expected_usd=expected,
            billed_usd=billed,
            deviation=None,
            tolerance=tolerance_value,
            reason=f"会话数为 {sessions}（≤0），预期金额无意义，无法计算偏差",
            ts=now,
        )

    deviation = (expected - billed) / billed  # 带符号：正=少收，负=多收
    if deviation > 0:
        direction = "预期 > 账单（少收/少计）"
    elif deviation < 0:
        direction = "账单 > 预期（多收/多计）"
    else:
        direction = "预期与账单一致"
    if abs(deviation) > tolerance_value:  # 严格大于：正好等于容差不算偏差
        verdict = ReconcileVerdict.DEVIATION
        reason = (
            f"偏差 {deviation:+.2%} 超过容差 {tolerance_value:.2%}"
            f"（预期 ${expected:.4f} vs 账单 ${billed:.4f}，{direction}）"
        )
    else:
        verdict = ReconcileVerdict.OK
        reason = (
            f"偏差 {deviation:+.2%} 在容差 {tolerance_value:.2%} 以内"
            f"（预期 ${expected:.4f} vs 账单 ${billed:.4f}，{direction}）"
        )
    return ReconcileReport(
        verdict=verdict,
        expected_usd=expected,
        billed_usd=billed,
        deviation=deviation,
        tolerance=tolerance_value,
        reason=reason,
        ts=now,
    )


class Reconciler:
    """把对账结果接到告警出口。

    结论映射（三条都有测试）：

    * `DEVIATION` → 一条 **CRITICAL** 告警（偏差是可量化的，必须有人查账）。
    * `UNKNOWN` → 一条 **WARNING** 告警（算不出来也要被看见，不能静默）。
    * `OK` → 不 emit。
    """

    rule_id = RULE_DAILY_RECONCILIATION

    def __init__(
        self, sink: AlertSink, tolerance: float = DEFAULT_TOLERANCE
    ) -> None:
        tolerance_value = _check_finite(tolerance, "tolerance")
        if not 0.0 < tolerance_value < 1.0:
            raise ValueError(f"tolerance 必须落在 (0, 1)，实际: {tolerance!r}")
        self._sink = sink
        self._tolerance = tolerance_value

    @property
    def tolerance(self) -> float:
        return self._tolerance

    def run(
        self,
        *,
        sessions: int,
        mean_cost_usd: float,
        billed_usd: float,
        ts: float = 0.0,
    ) -> ReconcileReport:
        """对账并（按结论）emit 告警，返回报告。"""
        report = reconcile(
            sessions=sessions,
            mean_cost_usd=mean_cost_usd,
            billed_usd=billed_usd,
            tolerance=self._tolerance,
            ts=ts,
        )
        if report.verdict is ReconcileVerdict.DEVIATION:
            self._sink.emit(self._deviation_alert(report))
        elif report.verdict is ReconcileVerdict.UNKNOWN:
            self._sink.emit(self._unknown_alert(report))
        return report

    def _deviation_alert(self, report: ReconcileReport) -> Alert:
        return Alert(
            rule_id=self.rule_id,
            severity=Severity.CRITICAL,
            message=f"每日对账偏差超限：{report.reason}",
            value=float(report.deviation),  # DEVIATION 必有偏差（None 只出现在 UNKNOWN）
            threshold=report.tolerance,
            ts=report.ts,
            context={
                "verdict": report.verdict.value,
                "expected_usd": report.expected_usd,
                "billed_usd": report.billed_usd,
                "deviation": report.deviation,
                "tolerance": report.tolerance,
            },
        )

    def _unknown_alert(self, report: ReconcileReport) -> Alert:
        # 偏差算不出来："value" 无意义，用 NaN 表示；`Alert.to_dict()` 会把它转 None。
        return Alert(
            rule_id=self.rule_id,
            severity=Severity.WARNING,
            message=f"每日对账无法判定：{report.reason}",
            value=float("nan"),
            threshold=report.tolerance,
            ts=report.ts,
            context={
                "verdict": report.verdict.value,
                "expected_usd": report.expected_usd,
                "billed_usd": report.billed_usd,
                "deviation": report.deviation,
                "tolerance": report.tolerance,
            },
        )
