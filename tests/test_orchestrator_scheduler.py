"""每轮流水线调度（方案 §3.2）。

不用 pytest-asyncio：所有协程在同步测试里用 asyncio.run() 跑。
不依赖真实时间：时钟与 sleep 全部注入，慢评估器用虚拟时钟推进。
"""

from __future__ import annotations

import asyncio
import inspect
import re
import time

import pytest

from ruipin.domain.errors import Unavailable
from ruipin.orchestrator import degradation as degradation_mod
from ruipin.orchestrator import scheduler as scheduler_mod
from ruipin.orchestrator.scheduler import (
    MAX_SCORING_RETRIES,
    StepDeadlines,
    TurnContext,
    TurnOutcome,
    TurnScheduler,
)
from ruipin.ports import DIMENSIONS, EvalResult

FORBIDDEN_SCORE = 60.0  # 现系统"崩溃=60 分"的默认值，红线：绝不允许出现


# --------------------------------------------------------------------------
# 测试替身
# --------------------------------------------------------------------------


class VirtualClock:
    """虚拟单调时钟：只在显式推进时前进，测试不消耗真实时间。"""

    def __init__(self, t: float = 1000.0) -> None:
        self.t = t
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.t += seconds


class FakeEvaluator:
    """可注入延迟/异常/挂死的评分器。"""

    def __init__(
        self,
        result: EvalResult | None = None,
        exc: BaseException | None = None,
        delay_s: float = 0.0,
        clock: VirtualClock | None = None,
        chunks: int = 3,
        hang: bool = False,
        on_tick=None,
    ) -> None:
        self.name = "fake-llm"
        self.result = result
        self.exc = exc
        self.delay_s = delay_s
        self.clock = clock
        self.chunks = max(1, chunks)
        self.hang = hang
        self.on_tick = on_tick
        self.calls = 0
        self.requests: list = []
        self.cancelled = False
        self.finished = 0

    async def evaluate(self, req):  # noqa: D102
        self.calls += 1
        self.requests.append(req)
        if self.delay_s:
            for _ in range(self.chunks):
                self.clock.advance(self.delay_s / self.chunks)
                await asyncio.sleep(0)
                if self.on_tick is not None:
                    self.on_tick()
        if self.hang:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled = True
                raise
        if self.exc is not None:
            raise self.exc
        self.finished += 1
        return self.result


def make_result(score: float = 73.0, provider: str = "fake-llm", **kw) -> EvalResult:
    return EvalResult(
        dims={d: score for d in DIMENSIONS},
        score=score,
        feedback="结构清晰，举例充分",
        provider=provider,
        confidence=0.9,
        **kw,
    )


def make_ctx(**kw) -> TurnContext:
    base = dict(
        turn_index=1,
        question="请说明线程池的选型依据",
        answer="我用 ThreadPoolExecutor，核心线程数按 CPU 核数设定",
    )
    base.update(kw)
    return TurnContext(**base)


def numeric_values(outcome: TurnOutcome) -> list[float]:
    """把 outcome 里所有数值摊平（跳过索引/时延/等级这类记账字段）。"""
    skip = {"turn_index", "latency_ms", "level"}
    vals: list[float] = []
    for k, v in vars(outcome).items():
        if k in skip:
            continue
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            vals.append(float(v))
        elif isinstance(v, dict):
            vals += [
                float(x)
                for x in v.values()
                if isinstance(x, (int, float)) and not isinstance(x, bool)
            ]
    return vals


# --------------------------------------------------------------------------
# 预算常量
# --------------------------------------------------------------------------


def test_default_deadlines_match_corrected_budget():
    """方案 §3.2 + 自检 A2 修正后的预算。"""
    d = StepDeadlines()
    assert (d.asr_ms, d.scoring_ms, d.retry_ms) == (1200.0, 3500.0, 3500.0)
    assert (d.decision_ms, d.speculation_ms, d.turn_ms) == (500.0, 3000.0, 15000.0)


def test_only_one_scoring_retry():
    assert MAX_SCORING_RETRIES == 1


# --------------------------------------------------------------------------
# 正常路径
# --------------------------------------------------------------------------


def test_normal_path_returns_real_eval():
    clock = VirtualClock()
    ev = FakeEvaluator(result=make_result(73.0), clock=clock, delay_s=2.2)
    sched = TurnScheduler(ev, clock=clock, sleeper=clock.sleep)

    out = sched.run_turn_sync(make_ctx())

    assert out.turn_index == 1
    assert out.eval is not None
    assert out.eval.score == 73.0
    assert out.degraded is False
    assert out.degrade_reason is None
    assert out.timed_out_steps == []
    assert out.level == 0
    assert out.ui_badge == ""
    assert out.forced_advance is False
    assert out.latency_ms == pytest.approx(2200, abs=1)  # 虚拟时钟，确定性
    assert ev.calls == 1
    assert ev.requests[0].question == "请说明线程池的选型依据"
    assert ev.requests[0].answer.startswith("我用 ThreadPoolExecutor")


def test_latency_comes_from_injected_clock_not_wall_clock():
    clock = VirtualClock()
    ev = FakeEvaluator(result=make_result(), clock=clock, delay_s=0.0)
    sched = TurnScheduler(ev, clock=clock, sleeper=clock.sleep)
    wall0 = time.monotonic()
    out = sched.run_turn_sync(make_ctx())
    assert time.monotonic() - wall0 < 0.5  # 真实耗时可忽略
    assert out.latency_ms == 0  # 虚拟时钟没走


def test_transcript_is_used_as_answer_and_marks_audio():
    clock = VirtualClock()
    ev = FakeEvaluator(result=make_result(), clock=clock)
    sched = TurnScheduler(ev, clock=clock, sleeper=clock.sleep)
    out = sched.run_turn_sync(make_ctx(transcript="ASR 转写文本"))
    assert out.eval is not None
    assert ev.requests[0].answer == "ASR 转写文本"
    assert ev.requests[0].has_audio is True


# --------------------------------------------------------------------------
# 红线：禁止合成/default 分数
# --------------------------------------------------------------------------


def test_unavailable_yields_no_eval_and_no_default_score():
    clock = VirtualClock()
    ev = FakeEvaluator(exc=Unavailable("qwen-plus", "upstream 503"), clock=clock)
    sched = TurnScheduler(ev, clock=clock, sleeper=clock.sleep)

    out = sched.run_turn_sync(make_ctx())

    assert out.eval is None, "评估不可用必须是 None，不能塞默认分"
    assert out.degraded is True
    assert out.degrade_reason is not None
    assert "qwen-plus" in out.degrade_reason
    assert "upstream 503" in out.degrade_reason
    assert out.level == 2
    assert out.ui_badge == "本轮为规则评分，仅供参考"  # 可见，不静默
    assert out.timed_out_steps == []

    # 红线断言：结果里不存在任何 60 分（或任何数字型分数）
    assert FORBIDDEN_SCORE not in numeric_values(out)
    assert numeric_values(out) == []


def test_unavailable_retried_exactly_once():
    clock = VirtualClock()
    ev = FakeEvaluator(exc=Unavailable("qwen-plus", "boom"), clock=clock)
    sched = TurnScheduler(ev, clock=clock, sleeper=clock.sleep)
    out = sched.run_turn_sync(make_ctx())
    assert ev.calls == MAX_SCORING_RETRIES + 1 == 2
    assert out.eval is None


def test_retry_backoff_uses_injected_sleeper():
    clock = VirtualClock()
    ev = FakeEvaluator(exc=Unavailable("qwen-plus", "boom"), clock=clock)
    sched = TurnScheduler(ev, clock=clock, sleeper=clock.sleep, retry_backoff_s=0.25)
    out = sched.run_turn_sync(make_ctx())
    assert clock.sleeps == [0.25]  # 退避走注入的 sleeper，不消耗真实时间
    assert out.eval is None
    assert out.latency_ms == pytest.approx(250, abs=1)


def test_non_retryable_error_is_not_retried():
    clock = VirtualClock()
    ev = FakeEvaluator(exc=ValueError("schema 不合法"), clock=clock)
    sched = TurnScheduler(ev, clock=clock, sleeper=clock.sleep)
    out = sched.run_turn_sync(make_ctx())
    assert ev.calls == 1
    assert out.eval is None
    assert out.degraded is True
    assert "scoring_unretryable" in out.degrade_reason
    assert out.level == 2


def test_evaluator_returning_none_is_treated_as_unavailable():
    """契约违背：返回 None 而非抛 Unavailable，同样不许变成 0 分。"""
    clock = VirtualClock()
    ev = FakeEvaluator(result=None, clock=clock)
    sched = TurnScheduler(ev, clock=clock, sleeper=clock.sleep)
    out = sched.run_turn_sync(make_ctx())
    assert out.eval is None
    assert out.degraded is True
    assert FORBIDDEN_SCORE not in numeric_values(out)


def test_scheduler_source_contains_no_fabricated_score():
    """静态红线：编排器源码不得构造 EvalResult，也不得写死任何分数。"""
    src = inspect.getsource(scheduler_mod)
    assert "EvalResult(" not in src, "编排器不应自己构造评估结果"
    assert re.search(r"(?m)^\s*(default_score|fallback_score|score)\s*[:=]\s*[\d.]+", src) is None
    assert re.search(r"(?m)^\s*dims\s*[:=].*\d", src) is None


def test_degraded_result_from_evaluator_is_surfaced():
    """评估器自己标注 degraded（provider=rubric）时，编排器必须显式暴露徽标。"""
    clock = VirtualClock()
    degraded_result = make_result(
        score=55.0, provider="rubric", degraded=True, degrade_reason="rubric: LLM 不可用"
    )
    ev = FakeEvaluator(result=degraded_result, clock=clock)
    sched = TurnScheduler(ev, clock=clock, sleeper=clock.sleep)
    out = sched.run_turn_sync(make_ctx())
    assert out.eval is not None and out.eval.provider == "rubric"
    assert out.degraded is True
    assert "rubric" in out.degrade_reason
    assert "本轮为规则评分，仅供参考" in out.ui_badge


# --------------------------------------------------------------------------
# 超时：不得无限等待
# --------------------------------------------------------------------------


def test_slow_evaluator_times_out_and_is_cancelled():
    clock = VirtualClock()
    ev = FakeEvaluator(result=make_result(), clock=clock, delay_s=5.0, hang=True)
    sched = TurnScheduler(ev, clock=clock, sleeper=clock.sleep)

    wall0 = time.monotonic()
    out = sched.run_turn_sync(make_ctx())
    assert time.monotonic() - wall0 < 1.0, "超时判定不得依赖真实耗时"

    assert out.eval is None
    assert out.degraded is True
    assert out.timed_out_steps == ["scoring", "scoring_retry"]
    assert out.level == 2
    assert ev.cancelled is True, "超时任务必须被取消，不能泄漏"
    assert FORBIDDEN_SCORE not in numeric_values(out)


def test_slow_but_within_budget_is_not_a_timeout():
    clock = VirtualClock()
    ev = FakeEvaluator(result=make_result(), clock=clock, delay_s=3.4)
    sched = TurnScheduler(ev, clock=clock, sleeper=clock.sleep)
    out = sched.run_turn_sync(make_ctx())
    assert out.eval is not None
    assert out.timed_out_steps == []
    assert out.degraded is False


def test_turn_budget_forced_advance_is_flagged():
    clock = VirtualClock()
    ev = FakeEvaluator(result=make_result(), clock=clock, delay_s=2.2)
    sched = TurnScheduler(ev, clock=clock, sleeper=clock.sleep)
    out = sched.run_turn_sync(make_ctx(deadlines=StepDeadlines(turn_ms=1000.0)))
    assert out.eval is not None  # 分数仍是真的
    assert out.forced_advance is True
    assert out.latency_ms == pytest.approx(2200, abs=1)


# --------------------------------------------------------------------------
# ASR
# --------------------------------------------------------------------------


def test_asr_timeout_falls_back_to_partial_and_flags_degradation():
    clock = VirtualClock()

    async def slow_asr(ctx):
        clock.advance(5.0)
        await asyncio.Event().wait()

    ev = FakeEvaluator(result=make_result(), clock=clock, delay_s=1.0)
    sched = TurnScheduler(ev, clock=clock, sleeper=clock.sleep, asr=slow_asr)

    out = sched.run_turn_sync(make_ctx(transcript="已收到的 partial 转写"))

    assert out.timed_out_steps == ["asr"]
    assert out.degraded is True
    assert out.level == 3
    assert out.ui_badge == "未获取语音"
    assert "asr_timeout" in out.degrade_reason
    assert ev.requests[0].answer == "已收到的 partial 转写"  # 用 partial 继续评分
    assert out.eval is not None


def test_asr_timeout_without_any_text_skips_scoring():
    clock = VirtualClock()

    async def slow_asr(ctx):
        clock.advance(5.0)
        await asyncio.Event().wait()

    ev = FakeEvaluator(result=make_result(), clock=clock)
    sched = TurnScheduler(ev, clock=clock, sleeper=clock.sleep, asr=slow_asr)

    out = sched.run_turn_sync(make_ctx(answer="", transcript=None))

    assert out.timed_out_steps == ["asr"]
    assert out.eval is None
    assert out.degraded is True
    assert ev.calls == 0, "没有作答文本就不该评分"


def test_no_asr_and_empty_answer_is_degraded_level_3():
    clock = VirtualClock()
    ev = FakeEvaluator(result=make_result(), clock=clock)
    sched = TurnScheduler(ev, clock=clock, sleeper=clock.sleep)
    out = sched.run_turn_sync(make_ctx(answer="   ", transcript=None))
    assert out.eval is None
    assert out.degraded is True
    assert out.level == 3
    assert ev.calls == 0


def test_fast_asr_uses_finalized_transcript():
    clock = VirtualClock()

    async def asr(ctx):
        clock.advance(0.8)
        return "ASR 最终结果"

    ev = FakeEvaluator(result=make_result(), clock=clock, delay_s=1.5)
    sched = TurnScheduler(ev, clock=clock, sleeper=clock.sleep, asr=asr)
    out = sched.run_turn_sync(make_ctx())
    assert out.timed_out_steps == []
    assert out.degraded is False
    assert ev.requests[0].answer == "ASR 最终结果"
    assert ev.requests[0].has_audio is True
    assert out.latency_ms == pytest.approx(2300, abs=1)


# --------------------------------------------------------------------------
# 投机预生成
# --------------------------------------------------------------------------


def test_speculation_hit_when_no_followup():
    clock = VirtualClock()
    trace: list[str] = []

    async def gen(ctx):
        trace.append("gen:start")
        clock.advance(0.5)
        await asyncio.sleep(0)
        trace.append("gen:end")
        return "下一题：请谈谈 GC 调优"

    ev = FakeEvaluator(result=make_result(), clock=clock, delay_s=2.0, chunks=4,
                       on_tick=lambda: trace.append("eval:tick"))
    sched = TurnScheduler(
        ev, clock=clock, sleeper=clock.sleep, question_generator=gen,
        decider=lambda ctx, res: asyncio.sleep(0, result=False),
    )

    out = sched.run_turn_sync(make_ctx())

    assert out.speculation_hit is True
    assert out.speculated_question == "下一题：请谈谈 GC 调优"
    assert out.eval is not None
    # 真并发：投机任务在评分还在跑的时候就已经完成
    assert trace[0] == "gen:start"
    assert trace.index("gen:end") < (len(trace) - 1 - trace[::-1].index("eval:tick"))


def test_speculation_missed_when_followup_needed():
    clock = VirtualClock()
    calls = {"gen": 0}

    async def gen(ctx):
        calls["gen"] += 1
        return "下一题（不会被用）"

    ev = FakeEvaluator(result=make_result(), clock=clock, delay_s=1.0)
    sched = TurnScheduler(
        ev,
        clock=clock,
        sleeper=clock.sleep,
        question_generator=gen,
        decider=lambda ctx, res: asyncio.sleep(0, result=True),
    )
    out = sched.run_turn_sync(make_ctx())
    assert out.speculation_hit is False
    assert out.speculated_question is None
    assert calls["gen"] == 1  # 投机已发起，只是没用上


def test_speculation_timeout_is_best_effort():
    clock = VirtualClock()

    async def slow_gen(ctx):
        clock.advance(5.0)  # > speculation_ms=3000
        await asyncio.Event().wait()

    ev = FakeEvaluator(result=make_result(), clock=clock, delay_s=1.0)
    sched = TurnScheduler(ev, clock=clock, sleeper=clock.sleep, question_generator=slow_gen)
    out = sched.run_turn_sync(make_ctx())
    assert out.speculated_question is None
    assert out.speculation_hit is False
    assert out.eval is not None  # 主链不受投机失败影响
    assert "speculation" not in out.timed_out_steps  # 投机失败不算主链超时


def test_speculation_standalone_without_generator():
    clock = VirtualClock()
    ev = FakeEvaluator(result=make_result(), clock=clock)
    sched = TurnScheduler(ev, clock=clock, sleeper=clock.sleep)
    assert asyncio.run(sched.speculative_next_question(make_ctx())) is None


def test_speculation_error_is_not_swallowed():
    """投机生成器的非超时异常不静默吞掉：向上传播，由调用方决定。"""
    clock = VirtualClock()

    async def broken_gen(ctx):
        raise RuntimeError("出题服务 500")

    ev = FakeEvaluator(result=make_result(), clock=clock, delay_s=1.0)
    sched = TurnScheduler(ev, clock=clock, sleeper=clock.sleep, question_generator=broken_gen)
    with pytest.raises(RuntimeError, match="出题服务 500"):
        sched.run_turn_sync(make_ctx())


def test_decision_timeout_gives_up_followup_and_uses_speculation():
    clock = VirtualClock()

    async def slow_decider(ctx, res):
        clock.advance(1.0)  # > decision_ms=500
        await asyncio.Event().wait()

    async def gen(ctx):
        return "预生成的下一题"

    ev = FakeEvaluator(result=make_result(), clock=clock, delay_s=1.0)
    sched = TurnScheduler(
        ev, clock=clock, sleeper=clock.sleep, decider=slow_decider, question_generator=gen
    )
    out = sched.run_turn_sync(make_ctx())
    assert out.timed_out_steps == ["decision"]
    assert out.speculation_hit is True
    assert out.speculated_question == "预生成的下一题"
    assert out.eval is not None


def test_no_decider_means_no_followup():
    clock = VirtualClock()
    ev = FakeEvaluator(result=make_result(), clock=clock, delay_s=1.0)
    sched = TurnScheduler(ev, clock=clock, sleeper=clock.sleep)
    out = sched.run_turn_sync(make_ctx())
    assert out.timed_out_steps == []


# --------------------------------------------------------------------------
# 审计与跨轮累积
# --------------------------------------------------------------------------


def test_tracker_accumulates_across_turns():
    clock = VirtualClock()
    ev = FakeEvaluator(exc=Unavailable("qwen-plus", "down"), clock=clock)
    sched = TurnScheduler(ev, clock=clock, sleeper=clock.sleep)
    first = sched.run_turn_sync(make_ctx(turn_index=1))
    second = sched.run_turn_sync(make_ctx(turn_index=2))

    assert first.turn_index == 1 and second.turn_index == 2
    assert first.eval is None and second.eval is None
    assert sched.tracker.level == 2
    assert len(sched.tracker.events()) == 2
    assert set(sched.tracker.reasons()) == {"scoring_failed"}


def test_every_degraded_outcome_has_a_non_empty_badge():
    """禁止静默降级：任何降级结果都必须能拿到中文徽标。"""
    clock = VirtualClock()
    ev = FakeEvaluator(exc=Unavailable("qwen-plus", "down"), clock=clock)
    sched = TurnScheduler(ev, clock=clock, sleeper=clock.sleep)
    out = sched.run_turn_sync(make_ctx())
    assert out.degraded
    assert out.ui_badge
    assert out.ui_badge == degradation_mod.badge_for(out.level)
