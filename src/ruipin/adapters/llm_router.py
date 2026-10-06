"""多供应商评分路由：主供应商不可用时按序切换到备用供应商。

设计意图
--------
1. **全部失败就抛 `Unavailable`，绝不返回合成结果**（本项目第一纪律）。不拼默认
   分、不取历史均值、不用"上一题的分数"顶上——任何"看起来正常"的兜底分都与真实
   分不可区分，这正是遗留系统崩溃即全 60 分的问题所在。
2. **只认 `Unavailable` 作为切换信号**。适配器契约就是"失败抛 Unavailable"；
   其它异常（ValueError 之类）是代码缺陷，应当直接炸出来被看见，而不是被路由
   器悄悄吞掉后换一家（那会把缺陷伪装成"供应商不可用"）。
3. **切换是可观测的**：`fallback_reason` 保留主供应商的失败原因，供编排层做
   `degradation.changed` 下行；同时在结果上标注 `degraded`，让聚合器知道这份
   分数来自备用链路。
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Optional, Sequence

from ..domain.errors import Unavailable
from ..ports import EvalRequest, EvalResult, Evaluator

#: 计量前缀：实际服务的供应商被记到 `llm.used.<provider>`。
USED_COUNTER_PREFIX = "llm.used."


class EvaluatorRouter:
    """按序尝试多个 `Evaluator`，返回第一个成功的结果。

    Args:
        evaluators: 供应商列表，**顺序即优先级**（第一个是主供应商）。
        meter: 可观测端口；成功时 `incr(f"llm.used.{provider}")`。
        name: 路由自身名字（进异常 `provider` 字段）。
        mark_degraded_on_fallback: 走了备用供应商时是否把结果标注为 `degraded`
            （默认 True：备用链路的成本/质量与主链路不同，聚合器需要知道）。
    """

    def __init__(
        self,
        evaluators: Sequence[Evaluator],
        *,
        meter: Optional[Any] = None,
        name: str = "router",
        mark_degraded_on_fallback: bool = True,
    ) -> None:
        if not evaluators:
            raise ValueError("EvaluatorRouter 至少需要 1 个 Evaluator")
        self._evaluators: list[Evaluator] = list(evaluators)
        self._meter = meter
        self._mark_degraded_on_fallback = bool(mark_degraded_on_fallback)
        self.name = name
        self._fallback_reason: Optional[str] = None

    # ---------- Evaluator ----------

    async def evaluate(self, req: EvalRequest) -> EvalResult:
        """按序评分。**全失败 → 抛 `Unavailable`，返回值一定是真实评估结果。**"""
        self._fallback_reason = None
        failures: list[str] = []

        for ev in self._evaluators:
            try:
                res = await ev.evaluate(req)
            except Unavailable as exc:
                # 只把这一种异常当作"换下一家"的信号。
                failures.append(f"{exc.provider}: {exc.reason}")
                continue

            if self._meter is not None:
                self._meter.incr(f"{USED_COUNTER_PREFIX}{res.provider}")
            if failures:
                reason = "已切换备用供应商（" + "；".join(failures) + "）"
                self._fallback_reason = reason
                if self._mark_degraded_on_fallback:
                    res = _annotate_fallback(res, reason)
            return res

        raise Unavailable(
            self.name,
            "全部评分供应商不可用（" + "；".join(failures) + "）",
        )

    # ---------- 只读视图 ----------

    @property
    def providers(self) -> tuple[str, ...]:
        """供应商顺序表（第一个为主供应商）。"""
        return tuple(ev.name for ev in self._evaluators)

    @property
    def fallback_reason(self) -> Optional[str]:
        """上一次 `evaluate` 是否走了备用供应商；None = 主供应商直接成功或未调用。"""
        return self._fallback_reason


def _annotate_fallback(res: EvalResult, reason: str) -> EvalResult:
    """给走备用链路的结果打降级标注（保留原有的 degrade_reason）。"""
    merged = f"{res.degrade_reason}；{reason}" if res.degrade_reason else reason
    return replace(res, degraded=True, degrade_reason=merged)


__all__ = ["USED_COUNTER_PREFIX", "EvaluatorRouter"]
