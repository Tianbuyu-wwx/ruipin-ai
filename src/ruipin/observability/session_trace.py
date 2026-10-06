"""会话级可观测性：单会话调试视图 + 评分归因（方案 §12.3）。

设计意图
--------
方案要两件在生产里真正救火的能力：

1. **单会话调试视图**：按 `session_id` 一次拉出完整事件日志、每步耗时、每次
   模型调用（prompt / 响应 / 成本 / 置信度）、降级记录。线上出问题第一句话
   永远是"把那个 session 的日志拉出来"，如果拉一次要拼四五个入口，排障就从
   五分钟变成五十分钟。
2. **评分归因**：任一维度可下钻到"哪个 provider、哪次调用、原始返回、如何聚合"。
   评分不可解释时，申诉无从回应、劣化无从定位。

关键纪律：排除必须留痕
----------------------
归因里被排除的贡献（置信度不足、测不准、供应商降级……）**必须写进 `explain()`**，
不能静默丢掉。静默丢弃会让"分数为什么变了"永远查不出来——贡献凭空消失，比数值
偏了更难排查。因此每个被排除的贡献都强制带 `excluded_reason`（构造期校验：要么
不排除，要么写明原因，不允许"排除但不说为什么"）。

另一个纪律：区分"没有这个会话"与"会话没有数据"
--------------------------------------------
`MemoryTraceStore.for_session` 对未知 session 返回 `None`，不返回一个空视图。空视图
会让"session 不存在"和"session 存在但确实什么都没发生"混为一谈——前者是 bug
（分桶错了 / 上报丢了），后者是正常，必须能分开。

时间
----
事件 / 降级记录的时间戳由调用方传入（调用方持有注入时钟），本模块不直接读系统时钟；
模型调用与步骤耗时用调用方上报的相对量（`cost_usd` / `latency_ms` / `started_ms`），
不做二次计时。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable

from ruipin.domain.errors import RuipinError

__all__ = [
    "Degradation",
    "DimensionAttribution",
    "DimensionContribution",
    "MemoryTraceStore",
    "ModelCall",
    "SessionDebugSnapshot",
    "SessionDebugView",
    "StepTrace",
    "TraceCollector",
    "TraceEvent",
    "UnknownDimension",
]


@dataclass(frozen=True)
class ModelCall:
    """一次模型调用的完整留痕。

    **失败调用也必须留痕**：`ok=False` 时 `error` 必须非空，`ok=True` 时 `error`
    必须为空。二者一致性能被构造期校验钉住——失败却没有原因，等于事后无从归因；
    成功却挂着 error，会让"这次到底成没成"变得含糊。
    """

    call_id: int
    turn_index: Optional[int]
    provider: str
    model: str
    prompt: str
    response: str
    cost_usd: float
    latency_ms: float
    confidence: Optional[float]
    ok: bool
    error: Optional[str] = None

    def __post_init__(self) -> None:
        if self.call_id < 0:
            raise ValueError(f"call_id 不能为负：{self.call_id!r}")
        if self.cost_usd < 0:
            raise ValueError(f"cost_usd 不能为负：{self.cost_usd!r}")
        if self.latency_ms < 0:
            raise ValueError(f"latency_ms 不能为负：{self.latency_ms!r}")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"置信度必须落在 [0, 1]：{self.confidence!r}")
        if self.ok:
            if self.error is not None:
                raise ValueError(
                    f"成功调用不应带 error：call_id={self.call_id}, error={self.error!r}"
                )
        elif not self.error:
            raise ValueError(
                f"失败调用必须写明 error（失败也要留痕）：call_id={self.call_id}"
            )


@dataclass(frozen=True)
class StepTrace:
    """一个编排步骤的耗时记录。"""

    step: str
    turn_index: Optional[int]
    started_ms: int
    ended_ms: int
    timed_out: bool
    note: str

    @property
    def duration_ms(self) -> int:
        """耗时（ms）。`ended_ms < started_ms` 说明时钟算错了，当场炸掉。

        不在这里钳成 0：负耗时是"时钟/单位写错了"的信号，静默归零会把它藏起来。
        """
        if self.ended_ms < self.started_ms:
            raise ValueError(
                f"步骤 {self.step!r} 的 ended_ms({self.ended_ms}) 早于 "
                f"started_ms({self.started_ms})，时钟算错了"
            )
        return self.ended_ms - self.started_ms


@dataclass(frozen=True)
class TraceEvent:
    """一条通用事件（自由文本 + 时间戳）。"""

    ts: float
    kind: str
    detail: str


@dataclass(frozen=True)
class Degradation:
    """一条降级记录。`level` 越大越严重（与 `domain.errors.Degraded` 同口径）。"""

    level: int
    reason: str
    ts: float


@dataclass(frozen=True)
class DimensionContribution:
    """某个维度在一次调用上的归一贡献（或"本应贡献但被排除"）。

    `excluded_reason` 非空即表示这条贡献**不参与聚合**，且原因已写明。
    """

    turn_index: int
    value: float
    weight: float
    confidence: float
    provider: str
    call_id: Optional[int] = None
    excluded_reason: Optional[str] = None

    def __post_init__(self) -> None:
        # 允许 None（未排除）；一旦给了原因就不能是空串——空串会让"排除"看起来像没排除。
        if self.excluded_reason is not None and not self.excluded_reason:
            raise ValueError("排除原因不能为空字符串：要么 None（未排除），要么写明原因")

    @property
    def excluded(self) -> bool:
        return self.excluded_reason is not None


@dataclass(frozen=True)
class DimensionAttribution:
    """一个维度的评分归因：若干贡献 + 最终值。

    本类只承载与下钻，不做聚合计算（聚合口径由 scoring 层负责，避免两处算法打架）。
    """

    dimension: str
    contributions: tuple[DimensionContribution, ...]
    final_value: Optional[float]

    def __post_init__(self) -> None:
        if not self.dimension:
            raise ValueError("dimension 不能为空")

    @property
    def excluded(self) -> tuple[DimensionContribution, ...]:
        """被排除的贡献（聚合时不计入，但归因必须展示）。"""
        return tuple(c for c in self.contributions if c.excluded)

    def explain(self) -> str:
        """生成人可读的中文归因说明，**含被排除的贡献及其原因**。"""
        lines = [f"维度 {self.dimension!r} 的评分归因："]
        if self.final_value is None:
            lines.append("  最终值：不可判定（无有效贡献）")
        else:
            lines.append(f"  最终值：{self.final_value:.4g}")

        included = tuple(c for c in self.contributions if not c.excluded)
        lines.append(f"  计入 {len(included)} 条贡献：")
        for c in included:
            lines.append(f"    · {self._describe(c)}")

        excluded = self.excluded
        lines.append(f"  排除 {len(excluded)} 条贡献（不计入聚合，但保留原因）：")
        for c in excluded:
            lines.append(f"    · {self._describe(c)} —— 原因：{c.excluded_reason}")
        return "\n".join(lines)

    @staticmethod
    def _describe(c: DimensionContribution) -> str:
        call = f"call#{c.call_id}" if c.call_id is not None else "无调用"
        return (
            f"第{c.turn_index}轮 {c.provider}({call}) value={c.value:.4g} "
            f"weight={c.weight:.4g} conf={c.confidence:.2f}"
        )


@dataclass(frozen=True)
class SessionDebugSnapshot:
    """一次会话的完整调试快照（`session_id` 一次拉全）。"""

    session_id: str
    events: tuple[TraceEvent, ...]
    steps: tuple[StepTrace, ...]
    calls: tuple[ModelCall, ...]
    degradations: tuple[Degradation, ...]
    attributions: tuple[DimensionAttribution, ...]

    @property
    def total_cost_usd(self) -> float:
        """本会话模型调用累计成本（USD）。"""
        return sum(c.cost_usd for c in self.calls)

    @property
    def failed_calls(self) -> tuple[ModelCall, ...]:
        """失败调用（`ok=False`），用于快速定位。"""
        return tuple(c for c in self.calls if not c.ok)


class UnknownDimension(RuipinError, KeyError):
    """按名字取归因时，该维度不存在。

    继承 `KeyError` 但重写 `__str__`（免受 repr 加引号），消息带已知维度清单。
    """

    def __init__(self, dimension: str, known: tuple[str, ...] = ()) -> None:
        self.dimension = dimension
        self.known = known
        msg = f"未知评分维度: {dimension!r}"
        if known:
            msg += f"（已知维度: {', '.join(known)}）"
        self._msg = msg
        super().__init__(msg)

    def __str__(self) -> str:
        return self._msg

    def __reduce__(self):
        return (self.__class__, (self.dimension, self.known))


class SessionDebugView:
    """单会话的调试视图（可写）。

    写入方法做**局部一致性校验**（call_id 唯一递增、ended≥started），因为这类
    错误一旦进入视图就会污染整条时间线，越早炸越好。
    `attributions` 是公开列表，由调用方直接填入——本类不做聚合计算。
    """

    def __init__(self, session_id: str) -> None:
        if not session_id:
            raise ValueError("session_id 不能为空")
        self._session_id = session_id
        self._events: list[TraceEvent] = []
        self._steps: list[StepTrace] = []
        self._calls: list[ModelCall] = []
        self._degradations: list[Degradation] = []
        #: 由调用方填入的维度归因（本类只承载）。
        self.attributions: list[DimensionAttribution] = []

    @property
    def session_id(self) -> str:
        return self._session_id

    def append_event(self, kind: str, detail: str, ts: float) -> TraceEvent:
        event = TraceEvent(ts=float(ts), kind=kind, detail=detail)
        self._events.append(event)
        return event

    def append_step(
        self,
        step: str,
        started_ms: int,
        ended_ms: int,
        *,
        turn_index: Optional[int] = None,
        timed_out: bool = False,
        note: str = "",
    ) -> StepTrace:
        """追加一步耗时。`ended_ms < started_ms` 直接抛错（时钟算错必须当场暴露）。"""
        if ended_ms < started_ms:
            raise ValueError(
                f"步骤 {step!r} 的 ended_ms({ended_ms}) 早于 started_ms({started_ms})"
            )
        trace = StepTrace(
            step=step,
            turn_index=turn_index,
            started_ms=started_ms,
            ended_ms=ended_ms,
            timed_out=timed_out,
            note=note,
        )
        self._steps.append(trace)
        return trace

    def append_call(self, call: ModelCall) -> None:
        """追加一次模型调用。`call_id` 必须**唯一递增**，重复或倒退都抛错。

        递增（而非仅唯一）是为了让快照天然按时间顺序排列，查询"最后一次调用"
        之类的问题不用再排序。
        """
        if self._calls and call.call_id <= self._calls[-1].call_id:
            raise ValueError(
                f"call_id 必须唯一递增：本条 {call.call_id} 不大于上一条 "
                f"{self._calls[-1].call_id}"
            )
        self._calls.append(call)

    def append_degradation(self, level: int, reason: str, ts: float) -> Degradation:
        if level < 1:
            raise ValueError(f"降级等级必须 ≥ 1：{level!r}")
        if not reason:
            raise ValueError("降级必须写明原因")
        degradation = Degradation(level=level, reason=reason, ts=float(ts))
        self._degradations.append(degradation)
        return degradation

    def dimension(self, name: str) -> DimensionAttribution:
        """按名字取维度归因；未知维度抛 `UnknownDimension`。"""
        for attribution in self.attributions:
            if attribution.dimension == name:
                return attribution
        raise UnknownDimension(name, tuple(a.dimension for a in self.attributions))

    def snapshot(self) -> SessionDebugSnapshot:
        """冻结成不可变快照（一次拿到本会话全部调试数据）。"""
        return SessionDebugSnapshot(
            session_id=self._session_id,
            events=tuple(self._events),
            steps=tuple(self._steps),
            calls=tuple(self._calls),
            degradations=tuple(self._degradations),
            attributions=tuple(self.attributions),
        )


@runtime_checkable
class TraceCollector(Protocol):
    """会话调试视图的收集端口。换成进程外存储时实现这两个方法即可。"""

    def add(self, view: SessionDebugView) -> None: ...

    def for_session(self, session_id: str) -> Optional[SessionDebugView]: ...


class MemoryTraceStore:
    """`TraceCollector` 的内存实现。

    对未知 session 返回 `None`（**不造空视图**），以区分"没有这个会话"与
    "这个会话没有任何数据"。同一 session 重复 `add` 时以最后一次为准。
    """

    def __init__(self) -> None:
        self._views: dict[str, SessionDebugView] = {}

    def add(self, view: SessionDebugView) -> None:
        self._views[view.session_id] = view

    def for_session(self, session_id: str) -> Optional[SessionDebugView]:
        return self._views.get(session_id)

    def sessions(self) -> tuple[str, ...]:
        """已登记的 session id（登记顺序）。"""
        return tuple(self._views)
