"""通用聚合器（方案 N5：可靠性 → 权重 → 重分配）。

核心主张
--------
**测不准 = 不计入并把权重还回去，绝不给中间分。**

这与现系统的根本差别：现系统在崩溃时给全 60 分，与"真实 60 分"不可区分；
本聚合器在维度不可信时把它的权重**按比例还给其他可信维度**，总分仍是 100% 的加权和，
因此"关闭生理模块"不会导致总分被系统性压低（这是可写成自动化断言的红线）。

虽然是因生理维度（rPPG 可靠性 R）而起（方案 §5.6），但实现**不绑定任何具体维度**：
任何 provider、任何维度，只要给出 `confidence`，就走同一条链路。

链路
----
1. `gate(dimension)` → 0~1 的可信系数（默认线性门控 + degraded 折半）
2. `effective = weight × gate`
3. 被门控掉的那部分权重，按**其余有效维度的原权重比例**重新分配，保证和为 1.0
4. 全部维度有效权重为 0 → 抛 `Unavailable`（绝不给中间分）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from ..domain.errors import Unavailable
from ..ports import DimensionScore

# ---------- 默认门控参数（与方案 §5.6 gate(R) 完全一致） ----------

#: conf < 0.40 → 权重归零（测不准即不计入）
GATE_LOW = 0.40
#: conf ≥ 0.75 → 权重完全生效
GATE_HIGH = 0.75
#: 降级维度：结果仍可用，但证据质量打折（与 confidence.DEGRADED_FACTOR 同源）
DEGRADED_GATE_FACTOR = 0.5

#: 六维默认权重（和为 1.0）。技术/问题解决是面试的主要构念，故权重最高。
DEFAULT_WEIGHTS: dict[str, float] = {
    "technical": 0.28,
    "communication": 0.14,
    "completeness": 0.18,
    "problem_solving": 0.22,
    "teamwork": 0.09,
    "leadership": 0.09,
}


@dataclass(frozen=True)
class AggregateResult:
    """聚合结果。

    - effective_weights：实际生效权重，和恒为 1.0（浮点意义上，比较请用 approx）
    - redistributed：每个维度**额外分到**多少权重（0 表示没有分到；被门控的维度恒为 0）
    - counterfactuals：完全剔除某维度后的总分，用于"关闭该模块会怎样"的反事实报告；
      无法计算（剔除后无任何可用维度）时填 `float("nan")` —— 既不是 0 也不是中间分
    - notes：人读说明，可直接进报告的"评分归因"栏
    """

    score: float
    effective_weights: dict[str, float]
    redistributed: dict[str, float]
    counterfactuals: dict[str, float]
    notes: list[str] = field(default_factory=list)


def default_gate(d: DimensionScore) -> float:
    """默认门控：置信度线性门控，降级维度额外折半。

    conf < 0.40 → 0；0.40 ≤ conf < 0.75 → (conf-0.40)/0.35；conf ≥ 0.75 → 1。
    `degraded=True` 再乘 0.5（降级结果是打了折的证据，不能等权参与）。
    """
    conf = _clamp01(d.confidence)
    if conf < GATE_LOW:
        g = 0.0
    elif conf >= GATE_HIGH:
        g = 1.0
    else:
        g = (conf - GATE_LOW) / (GATE_HIGH - GATE_LOW)
    if d.degraded:
        g *= DEGRADED_GATE_FACTOR
    return g


def aggregate(
    dims: dict[str, DimensionScore],
    weights: dict[str, float],
    gate: Optional[Callable[[DimensionScore], float]] = None,
) -> AggregateResult:
    """把多个维度聚合成 0-100 的总分。

    参数
    ----
    dims: 维度名 → DimensionScore（含 value/confidence/degraded）
    weights: 维度名 → 原权重（**不必预先归一化**，内部会归一化；不在 dims 中的键被忽略）
    gate: 自定义门控函数，默认 `default_gate`

    抛出
    ----
    Unavailable("aggregate", ...)：所有维度有效权重都为 0（包括权重全为 0 的情形）。
    """
    gate_fn = gate if gate is not None else default_gate

    keys = [k for k in dims]
    if not keys:
        raise Unavailable("aggregate", "所有维度不可用：没有输入维度")

    # 1) 归一化原权重（防御：权重和 <= 0 时直接进入不可用分支，不产生除零）
    w = {k: max(0.0, float(weights.get(k, 0.0))) for k in keys}
    total_w = sum(w.values())
    if total_w <= 0.0:
        raise Unavailable("aggregate", "所有维度不可用：权重和为 0")
    w = {k: v / total_w for k, v in w.items()}

    # 2) 门控
    gates = {k: _clamp01(gate_fn(dims[k])) for k in keys}

    # 3) 生效权重 + 重分配
    eff = _distribute(keys, w, gates)
    if eff is None:
        raise Unavailable("aggregate", "所有维度不可用")

    score = sum(eff[k] * float(dims[k].value) for k in keys)
    redistributed = {k: eff[k] - w[k] * gates[k] for k in keys}

    # 4) 反事实：逐个剔除
    counterfactuals: dict[str, float] = {}
    for k in keys:
        cf_gates = dict(gates)
        cf_gates[k] = 0.0
        cf_eff = _distribute(keys, w, cf_gates)
        counterfactuals[k] = (
            float("nan")
            if cf_eff is None
            else sum(cf_eff[j] * float(dims[j].value) for j in keys)
        )

    return AggregateResult(
        score=score,
        effective_weights=eff,
        redistributed=redistributed,
        counterfactuals=counterfactuals,
        notes=_build_notes(keys, dims, w, gates, eff, redistributed),
    )


def _distribute(
    keys: list[str],
    w: dict[str, float],
    gates: dict[str, float],
) -> Optional[dict[str, float]]:
    """计算生效权重：weight×gate 后把缺口按其余维度的**原权重比例**补回去并归一化。

    返回 None 表示"没有任何可用维度"。
    """
    raw = {k: w[k] * gates[k] for k in keys}
    total = sum(raw.values())
    if total <= 0.0:
        return None

    deficit = 1.0 - total
    if deficit > 0.0:
        # 只补给"还活着"的维度；被门控归零的维度不得复活
        cands = [k for k in keys if raw[k] > 0.0]
        cand_w = sum(w[k] for k in cands)
        if cand_w > 0.0:
            for k in cands:
                raw[k] += deficit * (w[k] / cand_w)

    # 归一化：gate 已截断在 [0,1]，raw 非负且 total>0，故 s>0，不会除零
    s = sum(raw.values())
    return {k: raw[k] / s for k in keys}


def _build_notes(
    keys: list[str],
    dims: dict[str, DimensionScore],
    w: dict[str, float],
    gates: dict[str, float],
    eff: dict[str, float],
    redistributed: dict[str, float],
) -> list[str]:
    notes: list[str] = []
    gated_out: list[str] = []
    partial: list[str] = []

    for k in keys:
        d = dims[k]
        g = gates[k]
        if g <= 0.0:
            gated_out.append(k)
            notes.append(
                f"维度 {k} 可靠性不足（confidence={d.confidence:.2f}），"
                f"权重 {w[k]:.1%} 已全部重分配给其他维度，未计入总分"
            )
        elif g < 1.0:
            partial.append(k)
            tail = "（该维度处于降级状态，权重额外折半）" if d.degraded else ""
            notes.append(
                f"维度 {k} 可靠性不足（confidence={d.confidence:.2f}，门控系数 {g:.2f}），"
                f"权重由 {w[k]:.1%} 降至 {w[k] * g:.1%}{tail}"
            )
        elif d.degraded:
            notes.append(
                f"维度 {k} 处于降级状态（provider={d.provider}），"
                f"权重由 {w[k]:.1%} 折减至 {w[k] * g:.1%}"
            )

    receivers = [(k, v) for k, v in redistributed.items() if v > 1e-9]
    if receivers:
        detail = "、".join(f"{k} +{v:.1%}" for k, v in receivers)
        notes.append(f"权重重分配：{detail}")
        notes.append(
            "被重分配的权重按其余维度的原权重比例分摊，总分仍是 100% 加权和，"
            "不会因某维度未评估而被系统性压低"
        )
    else:
        notes.append("所有维度可靠性达标，未发生权重重分配")

    return notes


def _clamp01(x: float) -> float:
    v = float(x)
    if v != v:  # NaN → 视为完全不可信
        return 0.0
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


__all__ = [
    "AggregateResult", "DEFAULT_WEIGHTS",
    "GATE_LOW", "GATE_HIGH", "DEGRADED_GATE_FACTOR",
    "aggregate", "default_gate",
]
