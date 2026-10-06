"""每轮流水线调度（方案 §3.2）。

```
t=0   answer.commit
 ├─ ASR finalize        ≤1200ms  超时：用已收到的 partial
 ├─ [投机并行] 预生成下一题 ≤3000ms  只有不追问时才用
 └─ LLM 评分            ≤3500ms  失败重试 1 次（仅对可重试错误）→ 仍失败则降级
 ↓ 追问决策 ≤500ms
 ↓ TTS 首块 ≤600ms（由 gateway 负责，不在本模块）
```

对外目标（自检 A2 修正）：**p50 ≤4 s / p95 ≤8 s**。

纪律：
1. **不得无限等待** —— 每一步用注入的 `clock` 判定 deadline，超时即取消任务并走降级。
2. **禁止合成分数** —— 评分失败时 `TurnOutcome.eval is None`，
   绝不用 60 分之类默认值冒充真实评估（现系统的头号可信度缺陷）。

时钟与 sleep 均可注入：测试用虚拟时钟，不依赖真实 sleep。
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

from ..domain.errors import RuipinError, Unavailable
from ..ports import EvalRequest, EvalResult, Evaluator
from .degradation import DegradationTracker

# 评分最多重试 1 次（方案 §3.2）
MAX_SCORING_RETRIES = 1

ASRFn = Callable[["TurnContext"], Awaitable[Optional[str]]]
DeciderFn = Callable[["TurnContext", Optional[EvalResult]], Awaitable[bool]]
QuestionFn = Callable[["TurnContext"], Awaitable[Optional[str]]]


class StepTimeout(RuipinError):
    """单步超时。由 deadline 判定触发，任务会被取消，绝不无限等待。"""

    def __init__(self, step: str, deadline_ms: float):
        self.step = step
        self.deadline_ms = deadline_ms
        super().__init__(f"步骤超时: {step} 超过 {deadline_ms:.0f}ms")


@dataclass(frozen=True)
class StepDeadlines:
    """每一步的时间预算（ms），可按会话/实验注入覆盖。"""

    asr_ms: float = 1200.0
    scoring_ms: float = 3500.0
    retry_ms: float = 3500.0
    decision_ms: float = 500.0
    speculation_ms: float = 3000.0
    turn_ms: float = 15000.0  # 轮次级硬预算，超则 forced_advance


@dataclass
class TurnContext:
    turn_index: int
    question: str
    answer: str
    deadlines: StepDeadlines = field(default_factory=StepDeadlines)
    transcript: Optional[str] = None
    question_type: str = "technical"
    has_video: bool = False
    keywords: tuple[str, ...] = ()


@dataclass
class TurnOutcome:
    """一轮调度的结果。

    `eval is None` 表示**本轮没有真实评估**（降级/超时/失败），
    调用方必须据此展示"评估稍后生成"而不是塞一个数字进去。
    """

    turn_index: int
    eval: Optional[EvalResult] = None
    degraded: bool = False
    degrade_reason: Optional[str] = None
    speculated_question: Optional[str] = None
    speculation_hit: bool = False
    latency_ms: int = 0
    timed_out_steps: list[str] = field(default_factory=list)
    level: int = 0
    forced_advance: bool = False

    @property
    def ui_badge(self) -> str:
        """给前端的徽标文案（空串表示无需打扰用户）。"""
        from .degradation import badge_for

        return badge_for(self.level) if self.degraded else ""


class TurnScheduler:
    """一轮回答的编排器。

    evaluator 之外的一切（时钟、sleep、ASR、追问决策、下一题生成器）都可注入，
    以便在无网络、无真实时间的环境下确定性测试。
    """

    def __init__(
        self,
        evaluator: Evaluator,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
        asr: Optional[ASRFn] = None,
        decider: Optional[DeciderFn] = None,
        question_generator: Optional[QuestionFn] = None,
        tracker: Optional[DegradationTracker] = None,
        retry_backoff_s: float = 0.0,
        poll_interval_s: float = 0.01,
    ) -> None:
        self._evaluator = evaluator
        self._clock = clock
        self._sleeper = sleeper
        self._asr = asr
        self._decider = decider
        self._question_generator = question_generator
        self.tracker = tracker if tracker is not None else DegradationTracker()
        self._retry_backoff_s = retry_backoff_s
        self._poll_interval_s = poll_interval_s

    # ---------- 主流程 ----------

    async def run_turn(self, ctx: TurnContext) -> TurnOutcome:
        t0 = self._clock()
        timed_out: list[str] = []
        reasons: list[str] = []
        result: Optional[EvalResult] = None
        speculated: Optional[str] = None
        speculation_hit = False

        # 1) ASR finalize —— 超时用已收到的 partial
        transcript = await self._finalize_asr(ctx, timed_out, reasons)

        answer_text = (transcript or ctx.answer or "").strip()

        # 2) 评分（含 1 次重试）与投机预生成并发发起
        spec_task: Optional[asyncio.Future] = None
        if not answer_text:
            self.tracker.escalate("asr_failed", detail="无可用作答文本（ASR 与文本均为空）")
            reasons.append("asr_failed: 无可用作答文本")
        else:
            if self._question_generator is not None:
                spec_task = asyncio.ensure_future(self.speculative_next_question(ctx))
            result = await self._score(ctx, answer_text, transcript, timed_out, reasons)

        # 3) 追问决策（超时则放弃追问，安全推进）
        need_followup = await self._decide(ctx, result, timed_out)

        # 4) 投机结果：只有"不追问"时才用得上
        if spec_task is not None:
            if need_followup:
                spec_task.cancel()
                await self._drain_cancelled(spec_task)
            else:
                speculated = await self._drain(spec_task)
                speculation_hit = speculated is not None

        latency_ms = int(round((self._clock() - t0) * 1000.0))
        forced_advance = latency_ms > ctx.deadlines.turn_ms

        if result is not None and result.degraded and result.degrade_reason:
            # 评估器自己声明了降级（如 provider=rubric），必须显式落到徽标上
            self.tracker.escalate("rubric_rule", detail=result.degrade_reason)
            reasons.append(result.degrade_reason)
        degraded = bool(reasons) or (result is not None and result.degraded)

        return TurnOutcome(
            turn_index=ctx.turn_index,
            eval=result,
            degraded=degraded,
            degrade_reason="; ".join(reasons) if reasons else None,
            speculated_question=speculated,
            speculation_hit=speculation_hit,
            latency_ms=latency_ms,
            timed_out_steps=timed_out,
            level=self.tracker.level,
            forced_advance=forced_advance,
        )

    def run_turn_sync(self, ctx: TurnContext) -> TurnOutcome:
        """同步包装，便于测试与离线批处理。"""
        return asyncio.run(self.run_turn(ctx))

    # ---------- 投机预生成 ----------

    async def speculative_next_question(
        self,
        ctx: TurnContext,
        generator: Optional[QuestionFn] = None,
        deadline_ms: Optional[float] = None,
    ) -> Optional[str]:
        """预生成"通用下一题"，与评分并发（调用方用 asyncio.gather / ensure_future）。

        超时返回 None（投机失败不影响主链，不算入降级，因为主链会现算一题）。
        """
        gen = generator if generator is not None else self._question_generator
        if gen is None:
            return None
        dl = deadline_ms if deadline_ms is not None else ctx.deadlines.speculation_ms
        try:
            return await self._with_deadline("speculation", dl, gen(ctx))
        except StepTimeout:
            return None

    # ---------- 步骤实现 ----------

    async def _finalize_asr(
        self, ctx: TurnContext, timed_out: list[str], reasons: list[str]
    ) -> Optional[str]:
        if self._asr is None:
            return ctx.transcript
        try:
            return await self._with_deadline("asr", ctx.deadlines.asr_ms, self._asr(ctx))
        except StepTimeout as exc:
            timed_out.append("asr")
            self.tracker.escalate("asr_timeout", detail=str(exc))
            reasons.append(f"asr_timeout: {exc}")
            return ctx.transcript  # 退化为已收到的 partial

    async def _score(
        self,
        ctx: TurnContext,
        answer_text: str,
        transcript: Optional[str],
        timed_out: list[str],
        reasons: list[str],
    ) -> Optional[EvalResult]:
        req = EvalRequest(
            question=ctx.question,
            answer=answer_text,
            question_type=ctx.question_type,
            transcript=transcript,
            has_audio=transcript is not None,
            has_video=ctx.has_video,
            keywords=ctx.keywords,
        )

        last_exc: Optional[BaseException] = None
        retryable = True
        for attempt in range(MAX_SCORING_RETRIES + 1):
            step = "scoring" if attempt == 0 else "scoring_retry"
            deadline = ctx.deadlines.scoring_ms if attempt == 0 else ctx.deadlines.retry_ms
            try:
                result = await self._with_deadline(
                    step, deadline, self._evaluator.evaluate(req)
                )
                if result is None:
                    # 契约违背：评估器返回 None 而非抛 Unavailable
                    raise Unavailable(getattr(self._evaluator, "name", "evaluator"), "返回 None")
                return result
            except StepTimeout as exc:
                timed_out.append(step)
                last_exc = exc
            except Unavailable as exc:
                last_exc = exc
            except Exception as exc:  # 非可重试错误：立即降级，不重试
                last_exc = exc
                retryable = False

            if attempt < MAX_SCORING_RETRIES and retryable:
                if self._retry_backoff_s > 0:
                    self._sleeper(self._retry_backoff_s)
                continue
            break

        reason_code = "scoring_timeout" if isinstance(last_exc, StepTimeout) else (
            "scoring_unretryable" if not retryable else "scoring_failed"
        )
        self.tracker.escalate(reason_code, detail=str(last_exc))
        reasons.append(f"{reason_code}: {last_exc}")
        return None  # ★ 绝不在此处编造分数

    async def _decide(
        self,
        ctx: TurnContext,
        result: Optional[EvalResult],
        timed_out: list[str],
    ) -> bool:
        if self._decider is None:
            return False
        try:
            return bool(
                await self._with_deadline(
                    "decision", ctx.deadlines.decision_ms, self._decider(ctx, result)
                )
            )
        except StepTimeout:
            # 决策超时不影响分数可信度，只是放弃这次追问机会；记录即可。
            timed_out.append("decision")
            return False

    # ---------- 超时基础设施 ----------

    async def _with_deadline(self, step: str, deadline_ms: float, coro: Any) -> Any:
        """在 deadline 内等待 coro；超时取消任务并抛 StepTimeout。

        用注入的 clock 判定，因此虚拟时钟下也是确定性的；
        调度器本身不 sleep，只是轮询让出事件循环。
        """
        start = self._clock()
        task = asyncio.ensure_future(coro)
        while True:
            done, _ = await asyncio.wait({task}, timeout=self._poll_interval_s)
            elapsed_ms = (self._clock() - start) * 1000.0
            if task in done:
                err = task.exception()
                if elapsed_ms >= deadline_ms:
                    raise StepTimeout(step, deadline_ms)
                if err is not None:
                    raise err
                return task.result()
            if elapsed_ms >= deadline_ms:
                task.cancel()
                await self._drain_cancelled(task)
                raise StepTimeout(step, deadline_ms)

    @staticmethod
    async def _drain(task: "asyncio.Future") -> Any:
        await asyncio.wait({task})
        err = task.exception()
        if err is not None:
            raise err
        return task.result()

    @staticmethod
    async def _drain_cancelled(task: "asyncio.Future") -> None:
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass
