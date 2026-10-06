"""实验曝光日志（append-only）（方案 §12.5）。

设计意图
--------
方案把"曝光"单列，因为曝光重复是实验里最常见的污染源：一个 session 若被记了
两次曝光，它在指标里的权重就翻了倍，主指标会被少数重度用户带偏。本模块用一条
**只追加**的内存日志把这件事变得可见：

* 不提供删除 / 修改方法——append-only 意味着历史不可篡改；
* 同一 `(session_id, experiment_id)` 重复曝光**不覆盖**既有记录，但会被记入
  `duplicate_exposures`，让"重复"从一个隐形 bug 变成一个能查的数字；
* `counts` 按**去重后的 session 数**统计（口径正确），`raw_counts` 给出未去重的
  原始条数（口径污染）。两个数不一致，就说明有重复曝光需要排查。

时间一律来自注入的 `clock`（返回秒），本模块不直接调用 `time.time()`，否则测试
无法确定性断言、回放也无法复现。

真实部署需要一个持久化端口：本模块通过可选的 `sink`（`ExposureSink`）把每条
记录转发出去，内存实现只负责本地查询与去重统计。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Protocol, runtime_checkable

__all__ = [
    "ExposureRecord",
    "ExposureSink",
    "MemoryExposureLog",
]


@dataclass(frozen=True)
class ExposureRecord:
    """一条曝光记录。`ts` 是秒（来自注入时钟）。"""

    session_id: str
    experiment_id: str
    variant: str
    ts: float
    metadata: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class ExposureSink(Protocol):
    """曝光持久化端口。

    内存版 `MemoryExposureLog` 之外的真实后端（Postgres / Kinesis / 数仓）实现
    这一个方法即可：日志构建、去重统计仍由内存版负责，落库这步外置。
    """

    def record(self, record: ExposureRecord) -> None: ...


class MemoryExposureLog:
    """曝光日志的内存实现（append-only，非线程安全）。

    只适合本地排障、单测与回放；生产请配合 `ExposureSink` 落库。
    """

    def __init__(
        self, clock: Callable[[], float], sink: Optional[ExposureSink] = None
    ) -> None:
        if not callable(clock):
            raise TypeError("clock 必须是可调用对象且返回秒（如 lambda: clock.now()）")
        self._clock = clock
        self._sink = sink
        self._records: list[ExposureRecord] = []
        #: 已出现过的 (session_id, experiment_id)，用于识别重复曝光。
        self._seen: set[tuple[str, str]] = set()
        #: (session_id, experiment_id) → 重复次数（超出首次的部分）。
        self._duplicates: dict[tuple[str, str], int] = {}

    def record(
        self,
        session_id: str,
        experiment_id: str,
        variant: str,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> ExposureRecord:
        """追加一条曝光。重复曝光只记账、不覆盖既有记录。

        `ts` 由注入时钟提供，调用方不得自带时间戳——统一时间源才能保证日志
        与其它观测对齐。
        """
        if not session_id:
            raise ValueError("session_id 不能为空")
        if not experiment_id:
            raise ValueError("experiment_id 不能为空")
        key = (session_id, experiment_id)
        if key in self._seen:
            # append-only：既有记录保持原样，仅把重复事实记下来供排查。
            self._duplicates[key] = self._duplicates.get(key, 0) + 1
        else:
            self._seen.add(key)
        record = ExposureRecord(
            session_id=session_id,
            experiment_id=experiment_id,
            variant=variant,
            ts=float(self._clock()),
            metadata=dict(metadata) if metadata else {},
        )
        self._records.append(record)
        if self._sink is not None:
            self._sink.record(record)
        return record

    @property
    def records(self) -> tuple[ExposureRecord, ...]:
        """全部曝光记录（按写入顺序，含重复）。"""
        return tuple(self._records)

    def for_session(self, session_id: str) -> tuple[ExposureRecord, ...]:
        """某个 session 的全部曝光记录（可能不止一条，重复会被如实返回）。"""
        return tuple(r for r in self._records if r.session_id == session_id)

    def for_experiment(self, experiment_id: str) -> tuple[ExposureRecord, ...]:
        """某个实验的全部曝光记录。"""
        return tuple(r for r in self._records if r.experiment_id == experiment_id)

    def counts(self, experiment_id: str) -> Mapping[str, int]:
        """按**去重后的 session 数**统计每个变体的曝光量（正确口径）。"""
        seen_sessions: set[str] = set()
        out: dict[str, int] = {}
        for r in self._records:
            if r.experiment_id != experiment_id or r.session_id in seen_sessions:
                continue
            seen_sessions.add(r.session_id)
            out[r.variant] = out.get(r.variant, 0) + 1
        return out

    def raw_counts(self, experiment_id: str) -> Mapping[str, int]:
        """未去重的原始条数（污染口径）。

        与 `counts` 的差值就是重复曝光的规模；分析主指标时必须用 `counts`。
        """
        out: dict[str, int] = {}
        for r in self._records:
            if r.experiment_id == experiment_id:
                out[r.variant] = out.get(r.variant, 0) + 1
        return out

    def duplicate_exposures(
        self, experiment_id: Optional[str] = None
    ) -> Mapping[tuple[str, str], int]:
        """重复曝光计数，键为 `(session_id, experiment_id)`，值为重复次数。

        传 `experiment_id` 只看某个实验；不传看全部。
        """
        if experiment_id is None:
            return dict(self._duplicates)
        return {key: n for key, n in self._duplicates.items() if key[1] == experiment_id}
