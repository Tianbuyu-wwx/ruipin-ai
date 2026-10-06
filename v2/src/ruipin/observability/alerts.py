"""告警规则（方案 §12.3「告警」；每日对账见 `reconciliation.py`）。

方案 §12.3 表里列了四条告警，本模块把它们逐条落成**看门对象**（watch），再加一个
统一评估器把一次采样的所有输入跑完、把结果 emit 到 sink：

======================  ======================================  ==============================
规则                     条件                                     本模块实现
======================  ======================================  ==============================
降级 Level ≥2 持续 5min  `degradation_sustained`                  `DegradationWatch`
worker 队列等待 p95 >10s `queue_wait_p95`                         `QueueWaitWatch`
单会话成本 >预算 80%      `session_cost_ratio`                     `SessionCostWatch`
HR 某群体得分偏移 >0.3SD  `group_score_drift`                      `GroupScoreDriftWatch`
======================  ======================================  ==============================

为什么不是"一次判断"而是"看门对象"
----------------------------------
三条规则里有一条（降级）天生是**有时序记忆**的：单次采到 L2 并不构成"持续 5min"。
如果写成无状态的纯函数，就得把"什么时候开始超阈值"塞进调用方，调用方一旦忘记
维护，就会变成"每个采样都报警"或"永远不报"。把计时状态封在看门对象里，语义
边界（何时重置、何时去重）只在一处定义，测试也只需盯这一处。

三条不可违背的纪律
------------------
1. **"判不了" ≠ "没问题"**：数据缺失 / 分母非正时，绝不静默算成"偏差为 0、无告警"。
   `GroupScoreDriftWatch` 在 `sd ≤ 0` / 群体为空时返回空列表，但通过 `last_undecidable`
   暴露原因；`AlertEvaluator` 把它记进 `undecidable`（"这批里哪几条判不了、为什么"），
   而不是假装"查过了、没问题"。
2. **报结论必报口径**：`AlertBatch` 同时携带 `checked_rules` 与 `undecidable`。只看到
   "0 条告警"而看不到"查了几条、几条判不了"，分不清"真过了"和"空过了"。
3. **阈值边界钉死**：队列 p95 用**严格大于**（正好 10000.0ms 不报），降级用**满时**
   （299.9s 不报、300.0s 报）。这些口径都进了测试，避免以后有人把它改成 `>=`。

不硬编码时间
------------
所有 `update` / `evaluate` 都要求调用方显式传入 `ts`（秒）。本模块**不**调用
`time.time()`——时钟必须由编排层注入，否则回放测试与时间旅行调试无从谈起。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional, Protocol, runtime_checkable

__all__ = [
    "Severity",
    "Alert",
    "AlertSink",
    "MemoryAlertSink",
    "AlertBatch",
    "AlertEvaluator",
    "DegradationWatch",
    "QueueWaitWatch",
    "SessionCostWatch",
    "GroupScoreDriftWatch",
    "RULE_DEGRADATION",
    "RULE_QUEUE_WAIT",
    "RULE_SESSION_COST",
    "RULE_GROUP_DRIFT",
    "DEFAULT_MIN_LEVEL",
    "DEFAULT_SUSTAINED_S",
    "DEFAULT_QUEUE_WAIT_MS",
    "DEFAULT_WARN_RATIO",
    "DEFAULT_MAX_SD",
]

# ---------- 规则 id（用常量而非散落的字面量，避免告警文案与路由键对不上）----------

RULE_DEGRADATION = "degradation_sustained"
RULE_QUEUE_WAIT = "queue_wait_p95"
RULE_SESSION_COST = "session_cost_ratio"
RULE_GROUP_DRIFT = "group_score_drift"

#: 方案 §12.3：降级 Level ≥2 持续 5min。
DEFAULT_MIN_LEVEL = 2
DEFAULT_SUSTAINED_S = 300.0
#: 方案 §12.3：worker 队列等待 p95 >10s。
DEFAULT_QUEUE_WAIT_MS = 10_000.0
#: 方案 §12.3：单会话成本 >预算 80%。
DEFAULT_WARN_RATIO = 0.8
#: 方案 §12.3：HR 模块某群体得分偏移 >0.3 SD。
DEFAULT_MAX_SD = 0.3


# ---------- 通用校验 ----------


def _check_finite(value: Any, field_name: str) -> float:
    """把数值输入校验成有限 float。

    字符串 / 布尔显式拒绝（`float("1200")` 会静默成功，把"传成字符串"这个集成
    bug 一路带到线上）；NaN / inf 也拒绝——NaN 参与比较恒为 False，会让"数据坏了"
    被读成"没越界"，正是本模块要消灭的那类静默。
    """
    if isinstance(value, bool) or isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{field_name} 必须是数值：{value!r}")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field_name} 必须是数值：{value!r}") from exc
    if math.isnan(number) or math.isinf(number):
        raise ValueError(f"{field_name} 必须是有限数：{value!r}")
    return number


def _check_ts(ts: Any) -> float:
    """时间戳（秒）必须是有限数。时钟注入不等于可以传垃圾进来。"""
    return _check_finite(ts, "ts")


def _json_safe(value: Any) -> Any:
    """递归把值整理成严格 JSON 可序列化的形态（`allow_nan=False` 也过）。

    NaN / inf 显式转 `None`：`json.dumps(..., allow_nan=False)` 遇到 NaN 会直接抛，
    很多日志管道就这么配的——告警发不出去比告警内容少一个字段严重得多。
    """
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


# ---------- 值对象 ----------


class Severity(str, Enum):
    """告警级别。`str` 混入是为了能直接进 JSON / 日志字段而不必先 `.value`。"""

    WARNING = "warning"
    CRITICAL = "critical"


@dataclass(frozen=True)
class Alert:
    """一条告警。`message` 是**中文**且**必须自带证据**（具体数值与阈值）。

    只报"队列等待过长"而不报实测值与阈值，收到告警的人还得回代码里推——这条
    纪律与 `slo.SLOReport.message` 一致。

    `value` / `threshold` 允许是 NaN：每日对账在"账单为 0、偏差算不出来"时会用 NaN
    表示"这个数没有意义"，`to_dict()` 会把它转成 `None`（见 `_json_safe`）。
    """

    rule_id: str
    severity: Severity
    message: str
    value: float
    threshold: float
    ts: float
    #: 结构化上下文（规则名、群体名、倍数……），进日志 / 告警 payload。
    context: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.rule_id, str) or not self.rule_id:
            raise ValueError(f"rule_id 必须是非空字符串：{self.rule_id!r}")
        if not isinstance(self.severity, Severity):
            raise ValueError(
                f"severity 必须是 Severity 枚举：{self.severity!r}（不许用裸字符串）"
            )
        if not isinstance(self.message, str) or not self.message:
            raise ValueError(f"message 必须是非空字符串：{self.message!r}")

    def to_dict(self) -> dict[str, Any]:
        """JSON 安全视图。NaN / inf 一律转 `None`（细节见 `_json_safe`）。"""
        return {
            "rule_id": self.rule_id,
            "severity": self.severity.value,
            "message": self.message,
            "value": _json_safe(self.value),
            "threshold": _json_safe(self.threshold),
            "ts": _json_safe(self.ts),
            "context": _json_safe(dict(self.context)),
        }


@runtime_checkable
class AlertSink(Protocol):
    """告警出口。生产里换成 webhook / 消息队列适配器时实现 `emit` 即可。"""

    def emit(self, alert: Alert) -> None: ...


class MemoryAlertSink:
    """内存告警出口 / 测试替身。

    **append-only**：只提供写入与读取，不提供删除 / 清空。告警记录是事后追责的
    凭据，允许"擦掉历史"会让它失去可信度（与 `session_trace` 的追加语义同源）。
    """

    def __init__(self) -> None:
        self._alerts: list[Alert] = []

    def emit(self, alert: Alert) -> None:
        if not isinstance(alert, Alert):
            raise TypeError(f"只能 emit Alert：{alert!r}")
        self._alerts.append(alert)

    @property
    def alerts(self) -> tuple[Alert, ...]:
        return tuple(self._alerts)

    @property
    def count(self) -> int:
        return len(self._alerts)

    def by_severity(self) -> Mapping[Severity, int]:
        """各级别计数。**两个级别都出现**（无则计 0），保证视图形状稳定——
        形状随内容变化的统计表，下游做面板时最容易漏读。"""
        return {
            Severity.WARNING: sum(
                1 for a in self._alerts if a.severity is Severity.WARNING
            ),
            Severity.CRITICAL: sum(
                1 for a in self._alerts if a.severity is Severity.CRITICAL
            ),
        }

    def fired_rule_ids(self) -> tuple[str, ...]:
        """触发过的规则 id，按**首次触发顺序**去重。"""
        seen: dict[str, None] = {}
        for alert in self._alerts:
            seen.setdefault(alert.rule_id, None)
        return tuple(seen)

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "by_severity": {s.value: n for s, n in self.by_severity().items()},
            "fired_rule_ids": list(self.fired_rule_ids()),
            "alerts": [a.to_dict() for a in self._alerts],
        }


# ---------- 看门对象 ----------


class DegradationWatch:
    """降级 Level ≥ `min_level` **持续** `sustained_s` 秒后告警。

    状态机（三个不变量，全部有测试钉住）：

    * 进入超阈值区间时记下起点 `_above_since`；单次采样**绝不**立即告警。
    * 中途掉回 `< min_level` → **重置计时**并把 `dip_count` 加一。抖动（一会儿
      L2、一会儿 L1）永远攒不满持续时间，也永远不会告警——这正是"持续"二字的
      用意；`dip_count` 暴露抖动次数，便于回看"这条规则是不是一直在被抖掉"。
    * 已报警时用 `firing` 去重，不再逐采样重复报；恢复（先掉下阈值）后可以**再次**
      触发。否则要么刷屏，要么一次告警后彻底哑火，两种都不可接受。
    """

    rule_id = RULE_DEGRADATION

    def __init__(
        self, min_level: int = DEFAULT_MIN_LEVEL, sustained_s: float = DEFAULT_SUSTAINED_S
    ) -> None:
        if isinstance(min_level, bool) or not isinstance(min_level, int):
            raise TypeError(f"min_level 必须是 int：{min_level!r}")
        if min_level < 1:
            raise ValueError(f"min_level 必须 ≥ 1：{min_level!r}")
        sustained = _check_finite(sustained_s, "sustained_s")
        if sustained <= 0:
            raise ValueError(f"sustained_s 必须 > 0，实际: {sustained_s!r}")
        self._min_level = min_level
        self._sustained_s = sustained
        self._above_since: Optional[float] = None
        self._firing = False
        self._dip_count = 0

    @property
    def dip_count(self) -> int:
        """计时被重置（掉回阈值以下）的次数。"""
        return self._dip_count

    @property
    def firing(self) -> bool:
        """当前是否处于"已报警、未恢复"状态（用于去重）。"""
        return self._firing

    def update(self, level: int, ts: float) -> Optional[Alert]:
        """喂一次降级等级采样；只有"持续满时"且未在报警中才返回 `Alert`。"""
        if isinstance(level, bool) or not isinstance(level, int):
            raise TypeError(f"level 必须是 int：{level!r}")
        now = _check_ts(ts)

        if level >= self._min_level:
            if self._above_since is None:
                self._above_since = now
            if self._firing:
                # 已在报警：不重复报（去重），计时保持不动。
                return None
            elapsed = now - self._above_since
            if elapsed < self._sustained_s:
                return None
            self._firing = True
            return Alert(
                rule_id=self.rule_id,
                severity=Severity.WARNING,
                message=(
                    f"降级 Level {level} 已持续 {elapsed:.1f}s（阈值 Level≥"
                    f"{self._min_level} 持续 {self._sustained_s:.1f}s），触发告警"
                ),
                value=float(level),
                threshold=float(self._min_level),
                ts=now,
                context={
                    "level": level,
                    "min_level": self._min_level,
                    "sustained_s": self._sustained_s,
                    "elapsed_s": elapsed,
                    "dip_count": self._dip_count,
                },
            )

        # 掉回阈值以下：重置计时（若在计时/报警中，记为一次抖动）。
        if self._above_since is not None:
            self._dip_count += 1
            self._above_since = None
        self._firing = False
        return None


class QueueWaitWatch:
    """worker 队列等待 p95 > `threshold_ms` 告警；≥ 2×阈值升 CRITICAL。

    边界口径：**严格大于**。正好等于 10_000.0ms **不报**（这是刻意的，见
    `tests/test_observability_alerts.py::test_queue_wait_exactly_threshold_not_fired`）；
    正好 2×阈值（20_000.0ms）算 CRITICAL。
    """

    rule_id = RULE_QUEUE_WAIT

    def __init__(self, threshold_ms: float = DEFAULT_QUEUE_WAIT_MS) -> None:
        threshold = _check_finite(threshold_ms, "threshold_ms")
        if threshold <= 0:
            raise ValueError(f"threshold_ms 必须 > 0，实际: {threshold_ms!r}")
        self._threshold_ms = threshold

    @property
    def threshold_ms(self) -> float:
        return self._threshold_ms

    def update(self, p95_ms: float, ts: float) -> Optional[Alert]:
        value = _check_finite(p95_ms, "p95_ms")
        now = _check_ts(ts)
        if value <= self._threshold_ms:  # 严格大于才算越界：正好等于阈值不报
            return None
        critical_at = 2.0 * self._threshold_ms
        if value >= critical_at:
            severity = Severity.CRITICAL
            relation = f"≥ {critical_at:.1f}ms（=2×阈值）"
        else:
            severity = Severity.WARNING
            relation = f"> {self._threshold_ms:.1f}ms"
        return Alert(
            rule_id=self.rule_id,
            severity=severity,
            message=(
                f"worker 队列等待 p95={value:.1f}ms {relation}，"
                f"阈值 {self._threshold_ms:.1f}ms"
            ),
            value=value,
            threshold=self._threshold_ms,
            ts=now,
            context={
                "p95_ms": value,
                "threshold_ms": self._threshold_ms,
                "critical_at_ms": critical_at,
            },
        )


class SessionCostWatch:
    """单会话成本占预算比例告警：≥ `warn_ratio` WARNING，≥ 1.0 CRITICAL。

    `budget_usd` 必须 > 0：预算为 0 意味着"任何花费都超顶"，比例无穷大，这种配置
    事故要在构造期拦下，而不是线上算出一个 inf 比例再到处传播。
    """

    rule_id = RULE_SESSION_COST

    def __init__(
        self, budget_usd: float, warn_ratio: float = DEFAULT_WARN_RATIO
    ) -> None:
        budget = _check_finite(budget_usd, "budget_usd")
        if budget <= 0:
            raise ValueError(f"budget_usd 必须 > 0，实际: {budget_usd!r}")
        ratio = _check_finite(warn_ratio, "warn_ratio")
        if not 0.0 < ratio <= 1.0:
            raise ValueError(f"warn_ratio 必须落在 (0, 1]，实际: {warn_ratio!r}")
        self._budget_usd = budget
        self._warn_ratio = ratio

    @property
    def budget_usd(self) -> float:
        return self._budget_usd

    @property
    def warn_ratio(self) -> float:
        return self._warn_ratio

    def update(self, spent_usd: float, ts: float) -> Optional[Alert]:
        spent = _check_finite(spent_usd, "spent_usd")
        now = _check_ts(ts)
        used = spent / self._budget_usd
        if used >= 1.0:
            severity = Severity.CRITICAL
            threshold = 1.0
            verdict = "已达/超过预算"
        elif used >= self._warn_ratio:
            severity = Severity.WARNING
            threshold = self._warn_ratio
            verdict = f"已达预算 {self._warn_ratio:.0%} 警戒线"
        else:
            return None
        return Alert(
            rule_id=self.rule_id,
            severity=severity,
            message=(
                f"单会话成本 ${spent:.4f} 占预算 ${self._budget_usd:.4f} 的 "
                f"{used:.1%}（{verdict}，阈值 {threshold:.0%}）"
            ),
            value=used,
            threshold=threshold,
            ts=now,
            context={
                "spent_usd": spent,
                "budget_usd": self._budget_usd,
                "usage_ratio": used,
                "warn_ratio": self._warn_ratio,
            },
        )


class GroupScoreDriftWatch:
    """HR 模块某群体得分偏移 > `max_sd` 个 SD 告警（**每个越界群体各一条**）。

    判定：对每个群体算 `|group_mean - overall_mean| / sd`，严格大于 `max_sd` 才越界。
    严格大于意味着"恰好 0.3"不报（边界口径有测试）。

    **判不了通道**：`sd ≤ 0` 或 `group_means` 为空时返回空列表，并把原因写进
    `last_undecidable`。这一步至关重要——如果直接返回空列表了事，"分母是 0、
    根本换算不出倍数"会被读成"所有群体偏移都是 0、很健康"，与"测不准 = 不计入"
    的纪律背道而驰。`AlertEvaluator` 会读走 `last_undecidable` 并记进 `undecidable`。
    """

    rule_id = RULE_GROUP_DRIFT

    def __init__(self, max_sd: float = DEFAULT_MAX_SD) -> None:
        threshold = _check_finite(max_sd, "max_sd")
        if threshold <= 0:
            raise ValueError(f"max_sd 必须 > 0，实际: {max_sd!r}")
        self._max_sd = threshold
        #: 上一次 `update` 若判不了，这里写明原因；判得了则为 `None`。
        self.last_undecidable: Optional[str] = None

    @property
    def max_sd(self) -> float:
        return self._max_sd

    def update(
        self,
        group_means: Mapping[str, float],
        overall_mean: float,
        sd: float,
        ts: float,
    ) -> list[Alert]:
        """为每个越界群体产出一条 Alert；判不了时返回空列表并记录原因。"""
        self.last_undecidable = None
        now = _check_ts(ts)
        if not group_means:
            self.last_undecidable = "group_means 为空：没有群体可比较（不是'无偏移'）"
            return []
        sd_value = _check_finite(sd, "sd")
        if sd_value <= 0:
            self.last_undecidable = (
                f"sd={sd_value!r} 非正：无法把偏移换算成 SD 倍数"
                f"（不是'无偏移'，也不能当 0 处理）"
            )
            return []
        overall = _check_finite(overall_mean, "overall_mean")

        alerts: list[Alert] = []
        # 按群体名排序，保证同一批输出顺序稳定（便于对拍与去重）。
        for group in sorted(group_means):
            mean = _check_finite(group_means[group], f"group_means[{group!r}]")
            multiples = abs(mean - overall) / sd_value
            if multiples <= self._max_sd:  # 严格大于：恰好等于阈值不报
                continue
            alerts.append(
                Alert(
                    rule_id=self.rule_id,
                    severity=Severity.WARNING,
                    message=(
                        f"群体 {group!r} 得分偏移 {multiples:.3f} SD（均值 {mean:.4f}，"
                        f"整体均值 {overall:.4f}，SD {sd_value:.4f}），"
                        f"超过阈值 {self._max_sd:.3f} SD"
                    ),
                    value=multiples,
                    threshold=self._max_sd,
                    ts=now,
                    context={
                        "group": group,
                        "group_mean": mean,
                        "overall_mean": overall,
                        "sd": sd_value,
                        "multiples": multiples,
                    },
                )
            )
        return alerts


# ---------- 统一评估器 ----------


@dataclass(frozen=True)
class AlertBatch:
    """一次评估的完整口径。

    * `alerts` —— 本次真正触发的告警（含同一规则的多条，如多群体越界）。
    * `checked_rules` —— 本次**输入齐备、真的跑了判定**的规则条数。
    * `undecidable` —— 规则名 → 为什么判不了；来源有两类：(a) 本次没提供该项输入，
      (b) 提供了输入但判定条件不足（如 `sd ≤ 0`）。"判不了"与"没问题"必须能分开。
    """

    alerts: tuple[Alert, ...]
    checked_rules: int
    undecidable: Mapping[str, str]

    def by_severity(self) -> Mapping[Severity, int]:
        return {
            Severity.WARNING: sum(
                1 for a in self.alerts if a.severity is Severity.WARNING
            ),
            Severity.CRITICAL: sum(
                1 for a in self.alerts if a.severity is Severity.CRITICAL
            ),
        }

    def summary(self) -> str:
        """一行口径汇总：`告警 2 条（W1/C1）, 已查 3 条规则, 2 条判不了`。"""
        counts = self.by_severity()
        return (
            f"告警 {len(self.alerts)} 条"
            f"（W{counts[Severity.WARNING]}/C{counts[Severity.CRITICAL]}）, "
            f"已查 {self.checked_rules} 条规则, {len(self.undecidable)} 条判不了"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary(),
            "checked_rules": self.checked_rules,
            "undecidable": dict(self.undecidable),
            "by_severity": {s.value: n for s, n in self.by_severity().items()},
            "alerts": [a.to_dict() for a in self.alerts],
        }


class AlertEvaluator:
    """一次跑完四条规则并 emit 到 sink。

    四条规则的输入彼此独立，**哪条没给数据就在 `undecidable` 里点名**，而不是
    悄悄跳过——跳过会让"0 条告警"看起来像"全绿"，实际是"只查了一部分"。

    各规则所需输入：

    * `degradation`：`level`
    * `queue_wait`：`queue_p95_ms`
    * `session_cost`：`session_spent_usd`
    * `group_score_drift`：`group_means` + `overall_mean` + `sd`

    看门对象可在构造时替换（如把降级持续时间调短做灰度）；不传则用方案默认值。

    `session_budget_usd` 是**必传**的：单会话预算没有"合理默认值"，随便给个 1.0
    会让 `session_cost` 规则在真实预算下彻底失真——这正是"不许合成兜底值"要防的。
    """

    def __init__(
        self,
        sink: AlertSink,
        *,
        session_budget_usd: float,
        warn_ratio: float = DEFAULT_WARN_RATIO,
        degradation: Optional[DegradationWatch] = None,
        queue_wait: Optional[QueueWaitWatch] = None,
        group_drift: Optional[GroupScoreDriftWatch] = None,
    ) -> None:
        self._sink = sink
        self._degradation = degradation or DegradationWatch()
        self._queue_wait = queue_wait or QueueWaitWatch()
        self._session_cost = SessionCostWatch(session_budget_usd, warn_ratio)
        self._group_drift = group_drift or GroupScoreDriftWatch()
        #: 本评估器覆盖的规则条数（用于"查了几条"的口径上界）。
        self.rule_count = 4

    @property
    def degradation(self) -> DegradationWatch:
        return self._degradation

    @property
    def queue_wait(self) -> QueueWaitWatch:
        return self._queue_wait

    @property
    def session_cost(self) -> SessionCostWatch:
        return self._session_cost

    @property
    def group_drift(self) -> GroupScoreDriftWatch:
        return self._group_drift

    def evaluate(
        self,
        *,
        ts: float,
        level: Optional[int] = None,
        queue_p95_ms: Optional[float] = None,
        session_spent_usd: Optional[float] = None,
        group_means: Optional[Mapping[str, float]] = None,
        overall_mean: Optional[float] = None,
        sd: Optional[float] = None,
    ) -> AlertBatch:
        """跑完四条规则，emit 所有告警，返回带口径的 `AlertBatch`。"""
        now = _check_ts(ts)
        alerts: list[Alert] = []
        undecidable: dict[str, str] = {}
        checked = 0

        # 1) 降级 Level ≥2 持续 5min
        if level is None:
            undecidable[self._degradation.rule_id] = "未提供 level（降级等级）"
        else:
            checked += 1
            alert = self._degradation.update(level, now)
            if alert is not None:
                alerts.append(alert)

        # 2) worker 队列等待 p95
        if queue_p95_ms is None:
            undecidable[self._queue_wait.rule_id] = "未提供 queue_p95_ms（队列等待 p95）"
        else:
            checked += 1
            alert = self._queue_wait.update(queue_p95_ms, now)
            if alert is not None:
                alerts.append(alert)

        # 3) 单会话成本
        if session_spent_usd is None:
            undecidable[self._session_cost.rule_id] = "未提供 session_spent_usd（单会话成本）"
        else:
            checked += 1
            alert = self._session_cost.update(session_spent_usd, now)
            if alert is not None:
                alerts.append(alert)

        # 4) 群体得分漂移（需要三个输入一起给，缺一即判不了）
        missing = [
            name
            for name, ok in (
                ("group_means", group_means is not None),
                ("overall_mean", overall_mean is not None),
                ("sd", sd is not None),
            )
            if not ok
        ]
        if missing:
            undecidable[self._group_drift.rule_id] = (
                f"缺少输入 {missing}，无法比较群体偏移"
            )
        else:
            checked += 1
            group_alerts = self._group_drift.update(
                group_means or {},  # type: ignore[arg-type]
                overall_mean,  # type: ignore[arg-type]
                sd,  # type: ignore[arg-type]
                now,
            )
            alerts.extend(group_alerts)
            if self._group_drift.last_undecidable is not None:
                undecidable[self._group_drift.rule_id] = (
                    self._group_drift.last_undecidable
                )

        for alert in alerts:
            self._sink.emit(alert)

        return AlertBatch(
            alerts=tuple(alerts),
            checked_rules=checked,
            undecidable=undecidable,
        )
