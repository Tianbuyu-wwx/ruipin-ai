"""可观测性：内存版埋点收集器（方案 §12.3「可观测性与成本治理」）。

设计意图
--------
方案自检发现：原方案在"可观测性"一节只写了"埋点"两个字，不足以支撑生产——
没有延迟分位数就无法回答"用户实际体感多慢"，没有成本硬顶就无法回答"跑一场
多少钱"，没有 SLO 违约检测就无法在指标劣化时自动告警。本模块补的是第一块：
一个**零第三方依赖**的内存收集器，它同时扮演两个角色：

1. 生产里的兜底实现 —— 没有 Prometheus / OpenTelemetry 时也能出分位数
   （本地排障、离线回放、单测断言）。
2. 测试替身 —— 断言"编排器到底埋了哪些点、花了多少 token"，不必真起 exporter。

类 `InMemoryMeter` **严格实现** `ruipin.ports.Meter`（runtime_checkable，
可用 `isinstance(m, Meter)` 校验），换 Prometheus 实现时只改适配器。

关键纪律：测不准 = 不计入
-------------------------
样本为空时，分位数返回 `None`，**不是 0**。0 会被下游读成"快得不可思议"，
从而把 SLO 判成"达标"——这比不测更危险。这条纪律与 `physio` 的
"测不准即弃权"、与 `scoring` 的"不许合成分数"是同一条原则。

窗口限制（重要取舍）
--------------------
延迟样本只保留**最近 N 条**（`max_samples`，默认 1000），否则内存无界增长。
代价必须说清楚：样本数超过 N 之后，p95 是"最近 N 条的 p95"而不是"全场次的
p95"，早先的长尾事件会被挤出窗口。生产化时应换成分位数草图（t-digest /
DDSketch）或直方图导出，本版不做（要零依赖）。

并发
----
**非线程安全**。v2 编排层是 asyncio 单线程，收集器只在同一个事件循环内被写入，
因此本类不加锁。若将来引入线程池 / 多进程 worker，必须在外层加锁或改为
进程外聚合（Prometheus 推网关等），不要指望本类在多线程下正确。
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any, Mapping, Optional, Sequence

from ruipin.ports import Usage

#: 每条延迟序列默认保留的样本数（最近 N 条），见模块 docstring 的取舍说明。
DEFAULT_MAX_SAMPLES = 1000

__all__ = [
    "DEFAULT_MAX_SAMPLES",
    "InMemoryMeter",
    "linear_percentile",
    "weighted_percentile",
]


# ---------- 分位数 ----------


def linear_percentile(values: Sequence[float], p: float) -> Optional[float]:
    """线性插值分位数，与 `numpy.percentile(..., method="linear")`（默认）一致。

    不取"最近秩"：最近秩在小样本上会把 p95 压得过分乐观。
    空样本返回 `None`（**不是 0**，0 会被读成"极快"从而误判达标）。
    """
    if not 0.0 < p <= 1.0:
        raise ValueError(f"分位数 p 必须落在 (0, 1]：{p!r}")
    ordered = sorted(float(v) for v in values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    h = (len(ordered) - 1) * p
    lo = math.floor(h)
    hi = math.ceil(h)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (h - lo)


def weighted_percentile(
    pairs: Sequence[tuple[float, float]], p: float
) -> Optional[float]:
    """加权分位数，用于"加权 p95"这类按流量构成加权的口径。

    位置轴定义：把按值升序的第 i 个样本放在 `r_i = C_{i-1} + (w_i - 1) / 2`
    （`C_{i-1}` 是它之前的累计权重），再在位置轴上做与 `linear_percentile`
    相同的线性插值。**权重全为 1 时 `r_i = i`，本函数退化为 `linear_percentile`**
    ——这个性质可以直接对拍验证，避免两套分位数口径打架。

    空样本或权重合计 ≤ 0 时返回 `None`。
    """
    if not 0.0 < p <= 1.0:
        raise ValueError(f"分位数 p 必须落在 (0, 1]：{p!r}")
    ordered = sorted((float(v), float(w)) for v, w in pairs)
    if not ordered:
        return None
    for _v, w in ordered:
        if not w > 0.0:
            raise ValueError(f"权重必须为正数：{pairs!r}")
    if len(ordered) == 1:
        return ordered[0][0]
    positions: list[float] = []
    cum = 0.0
    for _v, w in ordered:
        positions.append(cum + (w - 1.0) / 2.0)
        cum += w
    target = positions[0] + p * (positions[-1] - positions[0])
    idx = 0
    while idx + 2 < len(positions) and target > positions[idx + 1]:
        idx += 1
    span = positions[idx + 1] - positions[idx]
    frac = (target - positions[idx]) / span
    return ordered[idx][0] + frac * (ordered[idx + 1][0] - ordered[idx][0])


# ---------- 内部小工具 ----------


def _check_ms(ms: Any) -> float:
    """延迟入样本前的校验：必须是有限、非负的**数值**。

    NaN 会污染分位数（sorted 后位置随机），负数说明上报方算错了时钟，
    两者都必须当场炸掉而不是记一笔脏数据。

    字符串/布尔**显式拒绝**：`float("1200")` 会静默成功，于是调用方把 `"1200"`
    当延迟传进来时不会报错，却让"延迟是字符串"这个集成 bug 一路潜伏到线上；
    `float(True) == 1.0` 同理，1ms 的延迟会伪装成一个快得离谱的好数字。
    """
    if isinstance(ms, bool) or isinstance(ms, (str, bytes, bytearray)):
        raise TypeError(f"延迟必须是数值、不接受字符串或布尔：{ms!r}")
    try:
        value = float(ms)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"延迟必须是数值：{ms!r}") from exc
    if math.isnan(value) or math.isinf(value):
        raise ValueError(f"延迟必须是有限数：{ms!r}")
    if value < 0:
        raise ValueError(f"延迟不能为负：{ms!r}")
    return value


def _check_tokens(value: Any, field: str) -> int:
    """token 计数校验：必须是非负整数。

    `int(1.7)` 会静默截断成 1，`int(-5)` 会把总量算成负数——两者都会让成本
    对不上账，却都不报错。所以这里要求**真的是整数**（float 仅接受等价值）。
    """
    if isinstance(value, bool) or isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{field} 必须是整数：{value!r}")
    if isinstance(value, int):
        result = value
    else:
        number = float(value)
        if not number.is_integer():
            raise ValueError(f"{field} 必须是整数，不能截断小数：{value!r}")
        result = int(number)
    if result < 0:
        raise ValueError(f"{field} 不能为负：{value!r}")
    return result


def _summarize(values: Sequence[float]) -> dict[str, Any]:
    """一条延迟序列的摘要（JSON 安全；空序列的数值字段为 None）。"""
    return {
        "count": len(values),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "mean": (sum(values) / len(values)) if values else None,
        "p50": linear_percentile(values, 0.5),
        "p95": linear_percentile(values, 0.95),
    }


def _frozen_tags(tags: Optional[Mapping[str, str]]) -> tuple[tuple[str, str], ...]:
    """把 tags 冻结成可哈希、且**与插入顺序无关**的键。"""
    if not tags:
        return ()
    return tuple(sorted((str(k), str(v)) for k, v in tags.items()))


def _counter_label(name: str, frozen: tuple[tuple[str, str], ...]) -> str:
    """计数器在快照里的扁平键：`name` 或 `name{k=v,k2=v2}`。"""
    if not frozen:
        return name
    return name + "{" + ",".join(f"{k}={v}" for k, v in frozen) + "}"


class _UsageBucket:
    """按 (provider, operation) 分桶的用量累计。"""

    __slots__ = (
        "provider",
        "operation",
        "calls",
        "cached_calls",
        "prompt_tokens",
        "completion_tokens",
        "cost_usd",
        "latency",
    )

    def __init__(self, provider: str, operation: str, max_samples: int) -> None:
        self.provider = provider
        self.operation = operation
        self.calls = 0
        self.cached_calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cost_usd = 0.0
        self.latency: deque[float] = deque(maxlen=max_samples)

    def add(
        self,
        *,
        cached: bool,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float,
        latency_ms: float,
    ) -> None:
        """累加一次**已校验**的用量。

        入参全部由 `InMemoryMeter.record_usage` 先校验完毕再传进来：校验与写入
        分开，是为了保证"校验失败时桶里不留半个副作用"（旧版把 `calls += 1`
        放在校验之前，于是被拒绝的那次调用仍会凭空多出一个空桶和一个调用次数）。
        """
        self.calls += 1
        if cached:
            self.cached_calls += 1
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.cost_usd += cost_usd
        self.latency.append(latency_ms)

    def as_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "operation": self.operation,
            "calls": self.calls,
            "cached_calls": self.cached_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
            "cost_usd": self.cost_usd,
            "latency": _summarize(list(self.latency)),
        }


# ---------- 收集器 ----------


class InMemoryMeter:
    """`ruipin.ports.Meter` 的内存实现 / 测试替身。

    分三条线收集：

    * `record_usage(Usage)` —— 按 (provider, operation) 分桶累计 token / 成本 /
      次数 / 延迟样本（`Usage` 由适配器如实上报，见 `ports.Usage`）。
    * `observe_ms(name, ms)` —— 具名延迟序列，供 SLO 分位数判定使用；
      `observe_weighted_ms` 是带权版本（加权 p95）。
    * `incr(name, value, tags=...)` —— 计数器（降级次数、VLM 帧数……）。

    **非线程安全**，见模块 docstring。延迟样本只保留最近 `max_samples` 条。
    """

    def __init__(self, max_samples: int = DEFAULT_MAX_SAMPLES) -> None:
        if not isinstance(max_samples, int) or isinstance(max_samples, bool):
            raise TypeError(f"max_samples 必须是 int：{max_samples!r}")
        if max_samples < 1:
            raise ValueError(f"max_samples 必须 ≥ 1：{max_samples!r}")
        self._max_samples = max_samples
        self._latency: dict[str, deque[float]] = {}
        self._weighted: dict[str, deque[tuple[float, float]]] = {}
        self._usage: dict[tuple[str, str], _UsageBucket] = {}
        self._counters: dict[tuple[str, tuple[tuple[str, str], ...]], int] = {}

    # ----- Meter 端口 -----

    def record_usage(self, usage: Usage) -> None:
        """按 (provider, operation) 分桶累计一次外部调用的用量。

        **先全部校验、再写入**：任何一项不合法都直接抛错，且账目与样本
        完全不变（不留半个副作用）。
        """
        cost = float(usage.cost_usd)
        if math.isnan(cost) or math.isinf(cost):
            raise ValueError(f"cost_usd 必须是有限数：{usage.cost_usd!r}")
        if cost < 0:
            raise ValueError(f"cost_usd 不能为负：{usage.cost_usd!r}")
        prompt_tokens = _check_tokens(usage.prompt_tokens, "prompt_tokens")
        completion_tokens = _check_tokens(usage.completion_tokens, "completion_tokens")
        latency_ms = _check_ms(usage.latency_ms)

        key = (str(usage.provider), str(usage.operation))
        bucket = self._usage.get(key)
        if bucket is None:
            bucket = _UsageBucket(key[0], key[1], self._max_samples)
            self._usage[key] = bucket
        bucket.add(
            cached=bool(usage.cached),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost,
            latency_ms=latency_ms,
        )

    def observe_ms(self, name: str, ms: float) -> None:
        """记录一条具名延迟样本（ms）。"""
        self._latency.setdefault(name, deque(maxlen=self._max_samples)).append(
            _check_ms(ms)
        )

    def incr(
        self, name: str, value: int = 1, *, tags: Optional[Mapping[str, str]] = None
    ) -> None:
        """计数器自增。`tags` 会被冻结成与插入顺序无关的可哈希键。"""
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"计数增量必须是 int：{value!r}")
        key = (str(name), _frozen_tags(tags))
        self._counters[key] = self._counters.get(key, 0) + value

    # ----- 加权延迟 -----

    def observe_weighted_ms(self, name: str, ms: float, weight: float = 1.0) -> None:
        """记录一条带权延迟样本（`weight > 0`），用于加权 p95。"""
        value = _check_ms(ms)
        w = float(weight)
        if math.isnan(w) or math.isinf(w) or w <= 0:
            raise ValueError(f"权重必须是正有限数：{weight!r}")
        self._weighted.setdefault(name, deque(maxlen=self._max_samples)).append(
            (value, w)
        )

    def count_weighted(self, name: str) -> int:
        """加权序列的**样本条数**（不是权重合计）。"""
        return len(self._weighted.get(name, ()))

    def total_weight(self, name: str) -> float:
        """加权序列的权重合计。"""
        return sum(w for _v, w in self._weighted.get(name, ()))

    def percentile_weighted(self, name: str, p: float) -> Optional[float]:
        """加权分位数；无样本返回 `None`。"""
        return weighted_percentile(list(self._weighted.get(name, ())), p)

    # ----- 查询 -----

    def samples(self, name: str) -> list[float]:
        """延迟样本快照（按到达顺序，最多最近 N 条）。"""
        return list(self._latency.get(name, ()))

    def count(self, name: str) -> int:
        return len(self._latency.get(name, ()))

    def percentile(self, name: str, p: float) -> Optional[float]:
        """分位数；无样本返回 `None`（绝不用 0 冒充"很快"）。"""
        return linear_percentile(self.samples(name), p)

    def mean(self, name: str) -> Optional[float]:
        series = self._latency.get(name)
        if not series:
            return None
        return sum(series) / len(series)

    def minimum(self, name: str) -> Optional[float]:
        series = self._latency.get(name)
        return min(series) if series else None

    def maximum(self, name: str) -> Optional[float]:
        series = self._latency.get(name)
        return max(series) if series else None

    def summary(self, name: str) -> dict[str, Any]:
        """一条延迟序列的摘要：count / min / max / mean / p50 / p95。"""
        return _summarize(self.samples(name))

    def counter(self, name: str, tags: Optional[Mapping[str, str]] = None) -> int:
        return self._counters.get((str(name), _frozen_tags(tags)), 0)

    def counters(self) -> dict[str, int]:
        """扁平化的计数器视图：`name` 或 `name{k=v}` → 值。"""
        return {_counter_label(n, t): v for (n, t), v in self._counters.items()}

    def usage_buckets(self) -> dict[str, dict[str, Any]]:
        """按 `"provider|operation"` 扁平化的用量视图。"""
        return {
            f"{provider}|{operation}": bucket.as_dict()
            for (provider, operation), bucket in self._usage.items()
        }

    def usage_totals(self) -> dict[str, Any]:
        """跨桶合计：调用次数 / cached 次数 / token / 成本。"""
        return {
            "calls": sum(b.calls for b in self._usage.values()),
            "cached_calls": sum(b.cached_calls for b in self._usage.values()),
            "prompt_tokens": sum(b.prompt_tokens for b in self._usage.values()),
            "completion_tokens": sum(
                b.completion_tokens for b in self._usage.values()
            ),
            "cost_usd": sum(b.cost_usd for b in self._usage.values()),
        }

    # ----- 快照 -----

    def snapshot(self) -> dict[str, Any]:
        """完整快照，**可 JSON 序列化**（`json.dumps(..., allow_nan=False)` 通过）。

        结构::

            {
              "max_samples": int,                       # 每条序列保留的样本上限
              "latency": {name: {"count","min","max","mean","p50","p95","samples"}},
              "weighted_latency": {name: {"count","total_weight","p50","p95",
                                          "samples": [[ms, weight], ...]}},
              "usage": {"provider|operation": {"provider","operation","calls",
                        "cached_calls","prompt_tokens","completion_tokens",
                        "total_tokens","cost_usd","latency": {...}}},
              "counters": {"name" 或 "name{k=v}": int},
              "totals": {"calls","cached_calls","prompt_tokens",
                         "completion_tokens","cost_usd"},
            }

        空序列的数值字段一律为 `None`（不是 0）。
        """
        latency: dict[str, Any] = {}
        for name, series in self._latency.items():
            item = _summarize(list(series))
            item["samples"] = list(series)
            latency[name] = item
        weighted: dict[str, Any] = {}
        for name, series in self._weighted.items():
            pairs = list(series)
            weighted[name] = {
                "count": len(pairs),
                "total_weight": sum(w for _v, w in pairs),
                "p50": weighted_percentile(pairs, 0.5),
                "p95": weighted_percentile(pairs, 0.95),
                "samples": [[v, w] for v, w in pairs],
            }
        return {
            "max_samples": self._max_samples,
            "latency": latency,
            "weighted_latency": weighted,
            "usage": self.usage_buckets(),
            "counters": self.counters(),
            "totals": self.usage_totals(),
        }
