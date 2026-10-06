"""数据保留策略与删除执行（方案 §12.2）。

覆盖的数据类别与保留期（见 §12.2 表格）：

====================  ==========================  ==================
数据类别               保留期                       到期动作
====================  ==========================  ==================
原始音视频             **不落盘**                   —
HR 派生时序            与报告同生命周期（默认 12 月）  删除
评估报告               默认 12 月，租户可配（≤36）   删除并通知
结构化日志             90 天                        删除
Trace / 指标           30 天                        聚合后删除原始
审计日志               3 年（合规要求）             归档
====================  ==========================  ==================

核心纪律
--------
* **绝不把"没删干净"当成"删成功"**：`RetentionEnforcer.purge` 会把
  `PurgePort.delete` 的实际删除数与待删数逐一比对，不一致时产出
  `verified=False` 的凭证并写明差多少。方案要求"删除必须真删（含对象存储与备份），
  并提供删除凭证"——凭证的价值就在于它**可能**是 False。
* **"不落盘"与"保留期很长"是两件事**：`RAW_MEDIA` 从不落盘，没有"到期时刻"；
  某条规则也可以是"无限期保留"（`retain_days=None` 但动作不是 NONE）。
  两者 `due_at` 都返回 `None`，必须靠 `RetentionRule.never_stored` 区分语义——
  否则会把"根本没存过"误报成"永久保留"，合规口径直接错。
* **不硬编码时间**：时钟由构造参数注入（返回**秒**）。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Callable, Mapping, Optional, Protocol, runtime_checkable

#: 保留期以"日"为最小单位；月按 30 天换算（与方案里"12 个月"的粗略口径一致）。
MONTH_DAYS = 30

#: 一天的秒数。
DAY_S = 86_400

#: 审计日志保留 3 年（合规要求），按 365 天/年。
AUDIT_RETAIN_DAYS = 3 * 365

#: 报告默认保留 12 个月。
DEFAULT_REPORT_MONTHS = 12

#: 报告保留的合规上限：36 个月。超过即非法配置。
MAX_REPORT_MONTHS = 36


class RetentionAction(str, Enum):
    """到期动作。"""

    #: 不适用（从不落盘的数据没有到期动作）。
    NONE = "none"
    #: 直接删除。
    DELETE = "delete"
    #: 归档保留（审计日志）。
    ARCHIVE = "archive"
    #: 先聚合再删除原始明细（Trace/指标）。
    AGGREGATE_THEN_DELETE = "aggregate_then_delete"


class DataClass(str, Enum):
    """受保留策略管理的数据类别。"""

    RAW_MEDIA = "raw_media"
    HR_TIMESERIES = "hr_timeseries"
    REPORT = "report"
    STRUCTURED_LOG = "structured_log"
    TRACE_METRIC = "trace_metric"
    AUDIT_LOG = "audit_log"


@dataclass(frozen=True)
class RetentionRule:
    """单条保留规则。

    Args:
        retain_days: 保留天数。`None` 有两种含义，靠 `never_stored` 区分：
            * `action=NONE` → **从不落盘**（没有数据，无到期时刻）；
            * `action!=NONE` → **无限期保留**（有数据，但当前没有删除计划）。
        action: 到期动作。`NONE` 仅与 `retain_days=None` 搭配。
    """

    data_class: DataClass
    retain_days: Optional[int]
    action: RetentionAction

    def __post_init__(self) -> None:
        if not isinstance(self.data_class, DataClass):
            raise TypeError(f"data_class 必须是 DataClass，实际: {self.data_class!r}")
        if not isinstance(self.action, RetentionAction):
            raise TypeError(f"action 必须是 RetentionAction，实际: {self.action!r}")
        if self.action is RetentionAction.NONE:
            if self.retain_days is not None:
                raise ValueError(
                    "action=NONE 表示从不落盘，retain_days 必须为 None，"
                    f"实际: {self.retain_days}"
                )
        elif self.retain_days is not None and self.retain_days <= 0:
            raise ValueError(f"retain_days 必须为正，实际: {self.retain_days}")

    @property
    def never_stored(self) -> bool:
        """是否**从不落盘**（区别于"无限期保留"）。"""
        return self.action is RetentionAction.NONE and self.retain_days is None


class RetentionPolicy:
    """一组保留规则的不可变视图。

    用普通类而非 dataclass：内部字典要包成 `MappingProxyType` 冻结对外视图，
    且构造只应通过 `default()` / `for_tenant()`（保证合规校验一定被走到）。
    """

    def __init__(
        self,
        rules: Mapping[DataClass, RetentionRule],
        *,
        report_months: int = DEFAULT_REPORT_MONTHS,
    ) -> None:
        self._rules: Mapping[DataClass, RetentionRule] = MappingProxyType(dict(rules))
        self._report_months = int(report_months)

    # ---------- 构造 ----------

    @classmethod
    def default(cls) -> "RetentionPolicy":
        """标准策略（报告 12 个月）。"""
        return cls(cls._build_rules(DEFAULT_REPORT_MONTHS), report_months=DEFAULT_REPORT_MONTHS)

    @classmethod
    def for_tenant(cls, report_months: int) -> "RetentionPolicy":
        """按租户配置生成策略（HR 时序与报告同生命周期）。

        Raises:
            ValueError: `report_months` 超出 (0, 36] —— 36 个月是**合规硬上限**，
                超过即违规，必须显式报错而不是"悄悄截断到 36"。
        """
        if report_months <= 0:
            raise ValueError(f"report_months 必须为正，实际: {report_months}")
        if report_months > MAX_REPORT_MONTHS:
            raise ValueError(
                f"report_months 超过合规上限 {MAX_REPORT_MONTHS}，实际: {report_months}"
            )
        return cls(cls._build_rules(report_months), report_months=int(report_months))

    @staticmethod
    def _build_rules(report_months: int) -> dict[DataClass, RetentionRule]:
        """按 §12.2 表格构建规则表。"""
        report_days = report_months * MONTH_DAYS
        return {
            # 原始音视频从不落盘。
            DataClass.RAW_MEDIA: RetentionRule(
                DataClass.RAW_MEDIA, None, RetentionAction.NONE
            ),
            # HR 时序与报告同生命周期。
            DataClass.HR_TIMESERIES: RetentionRule(
                DataClass.HR_TIMESERIES, report_days, RetentionAction.DELETE
            ),
            DataClass.REPORT: RetentionRule(
                DataClass.REPORT, report_days, RetentionAction.DELETE
            ),
            DataClass.STRUCTURED_LOG: RetentionRule(
                DataClass.STRUCTURED_LOG, 90, RetentionAction.DELETE
            ),
            DataClass.TRACE_METRIC: RetentionRule(
                DataClass.TRACE_METRIC, 30, RetentionAction.AGGREGATE_THEN_DELETE
            ),
            DataClass.AUDIT_LOG: RetentionRule(
                DataClass.AUDIT_LOG, AUDIT_RETAIN_DAYS, RetentionAction.ARCHIVE
            ),
        }

    # ---------- 查询 ----------

    @property
    def report_months(self) -> int:
        return self._report_months

    @staticmethod
    def _coerce(data_class: object) -> DataClass:
        """把入参规范成 `DataClass`；无法识别时抛 `KeyError`（而非 ValueError）。

        统一成 `KeyError` 是因为调用方关心的是"这个类别在策略里找不到"，
        与查表未命中语义一致；`ValueError` 只用于"值是合法类别但规则非法"。
        """
        if isinstance(data_class, DataClass):
            return data_class
        try:
            return DataClass(data_class)
        except (ValueError, KeyError) as exc:
            raise KeyError(f"未知数据类别: {data_class!r}") from exc

    def rule(self, data_class: object) -> RetentionRule:
        """取某类别的规则。

        Raises:
            KeyError: 未知类别。
        """
        key = self._coerce(data_class)
        try:
            return self._rules[key]
        except KeyError as exc:
            known = [d.value for d in self._rules]
            raise KeyError(
                f"未知数据类别: {data_class!r}；已知类别: {known}"
            ) from exc

    def due_at(self, data_class: object, created_at: int) -> Optional[int]:
        """返回应删除/归档的时刻（Unix 秒）。

        `None` 表示**没有到期时刻**，须由调用方结合 `rule(...).never_stored`
        判断是"从不落盘"还是"无限期保留"（方案 §12.2 的区分要求）。
        """
        rule = self.rule(data_class)
        if rule.retain_days is None:
            return None
        return int(created_at) + rule.retain_days * DAY_S

    def is_due(self, data_class: object, created_at: int, now: int) -> bool:
        """是否已到期。从不落盘/无限期保留的类别恒为 False。"""
        due = self.due_at(data_class, created_at)
        return due is not None and now >= due


@dataclass(frozen=True)
class DeletionReceipt:
    """删除凭证（方案 §12.2："删除必须是真删……并提供删除凭证"）。

    `verified=False` 意味着**不能视为删除成功**：`evidence` 必须写明差多少，
    便于对账与人工补删。凭证还会作为"删除已执行"的审计依据，故字段校验从严。
    """

    data_class: DataClass
    subject_id: str
    deleted_at: int
    objects: tuple[str, ...]
    verified: bool
    evidence: str

    def __post_init__(self) -> None:
        if not isinstance(self.data_class, DataClass):
            raise TypeError(f"data_class 必须是 DataClass，实际: {self.data_class!r}")
        if not self.subject_id:
            raise ValueError("subject_id 不能为空：凭证必须能定位到被删除的主体")
        if not isinstance(self.objects, tuple):
            raise TypeError("objects 必须是 tuple（凭证须不可变，防止事后被篡改）")
        if not self.evidence:
            raise ValueError("evidence 不能为空：凭证必须说明删除依据")


@runtime_checkable
class PurgePort(Protocol):
    """删除端口（含对象存储与备份的具体实现藏在适配器后）。

    核心层不关心对象存在 S3、OSS 还是数据库；只要求实现能"列出某主体的某类对象"
    并"删除给定对象、返回实际成功删除的个数"。返回数是 `verified` 判定的唯一依据。
    """

    def list_objects(
        self, subject_id: str, data_class: DataClass
    ) -> tuple[str, ...]: ...

    def delete(self, object_ids: tuple[str, ...]) -> int: ...


class RetentionEnforcer:
    """按策略执行真删，并产出可核验的凭证。

    Args:
        policy: 保留策略。
        store: 删除端口（对象存储/DB/备份）。
        clock: 返回当前时间（**秒**）的可调用对象，用于凭证里的 `deleted_at`。
    """

    def __init__(
        self,
        policy: RetentionPolicy,
        store: PurgePort,
        clock: Callable[[], float],
    ) -> None:
        self._policy = policy
        self._store = store
        self._clock = clock

    def purge(self, data_class: object, subject_id: str) -> DeletionReceipt:
        """删除某主体在某类别下的全部对象，返回凭证。

        Raises:
            KeyError: 未知数据类别（由 `policy.rule` 抛出）。
            ValueError: `subject_id` 为空（凭证无法定位主体）。
        """
        rule = self._policy.rule(data_class)
        now = int(self._clock())

        # 先校验主体再动存储：空主体应立刻失败，而不是删完一批对象后才发现凭证无法定位主体
        #（那样存储已被改动，却拿不到可用凭证，反而造成"删了但说不清删了谁"）。
        if not subject_id:
            raise ValueError("subject_id 不能为空：凭证必须能定位到被删除的主体")

        # 从不落盘：无需（也无法）删除。这是**成功**——没有数据就没有泄露面，
        # 与"该删的没删掉"是两回事，故 verified=True。
        if rule.never_stored:
            return DeletionReceipt(
                data_class=rule.data_class,
                subject_id=subject_id,
                deleted_at=now,
                objects=(),
                verified=True,
                evidence=(
                    f"{rule.data_class.value} 的策略是'从不落盘'，"
                    "没有任何对象被保存，无需删除"
                ),
            )

        listed = tuple(self._store.list_objects(subject_id, rule.data_class))
        if not listed:
            return DeletionReceipt(
                data_class=rule.data_class,
                subject_id=subject_id,
                deleted_at=now,
                objects=(),
                verified=True,
                evidence=(
                    f"{rule.data_class.value} 未找到任何待删对象"
                    f"（subject={subject_id}），无需删除"
                ),
            )

        deleted = int(self._store.delete(listed))
        total = len(listed)
        verified = deleted == total
        if verified:
            evidence = (
                f"已删除 {deleted}/{total} 个对象，数量核对一致"
                "（真删：含对象存储；备份生命周期由存储方保证）"
            )
        elif deleted < total:
            evidence = (
                f"删除不完整：应删 {total} 个，实际删除 {deleted} 个，"
                f"少删 {total - deleted} 个；按纪律不视为删除成功，需重试并人工核查"
            )
        else:
            evidence = (
                f"删除数量异常：应删 {total} 个，端口返回 {deleted} 个，"
                f"多删 {deleted - total} 个（存储返回不可信）；不视为删除成功"
            )
        return DeletionReceipt(
            data_class=rule.data_class,
            subject_id=subject_id,
            deleted_at=now,
            objects=listed,
            verified=verified,
            evidence=evidence,
        )


__all__ = [
    "AUDIT_RETAIN_DAYS",
    "DAY_S",
    "DEFAULT_REPORT_MONTHS",
    "MAX_REPORT_MONTHS",
    "MONTH_DAYS",
    "DataClass",
    "DeletionReceipt",
    "PurgePort",
    "RetentionAction",
    "RetentionEnforcer",
    "RetentionPolicy",
    "RetentionRule",
]
