"""面试编排服务：把状态机、每轮调度、评分聚合、持久化串成一条可回放的主链。

为什么需要这一层
----------------
状态机（`domain/machine.py`）只回答"这个事件合法吗"，调度器（`scheduler.py`）只回答
"这一轮怎么在预算内跑完"，聚合器（`scoring/aggregate.py`）只回答"几个维度怎么合成总分"。
但**谁来按什么顺序调它们、什么时候落库、什么时候允许出报告**，散落在调用方就会出现
现系统那种"到处直接改 session.status"的局面。本模块是这条主链的唯一编排者。

四条硬纪律（每一条都有对应的自动化测试，见 tests/test_e2e_interview.py）
----------------------------------------------------------------------
1. **绝不合成兜底分数**。一轮没有拿到真实评估时 `TurnOutcome.eval is None`，
   该轮就不进聚合；若整场没有任何真实评估，`Report.available=False` 且 `score=None`。
   这与现系统"崩溃即返回全 60 分"正相反——60 分与真实 60 分不可区分，是最大的可信度缺陷。
2. **缓冲题不计分**（自检 A3）。适应性调节只允许插入 `BUFFER` 状态的题目，
   `BUFFER` 回合走 `BUFFER_DONE` 回 `ASKING`，事件日志里**不会出现 EVAL_DONE**，
   因此"这一轮没有被评分"是可审计的事实，不是口头承诺。
3. **生理维度单独授权**（PIPL 第 28 条）。`Consent.base` 与 `Consent.physio` 是两个开关；
   未单独授权时生理数据**根本不进服务**（连事件都不收），而不是收了再弃用。
4. **测不准 = 不计入**。生理维度不可用时，它**不出现在 `dims` 里**，聚合器会把它的
   权重按其余维度的原比例还回去，总分仍是 100% 加权和——关闭生理模块不会系统性压低总分。

时钟、存储、评估器、调度器全部注入，因此整场面试可以在无网络、无真实时间的环境下
确定性回放（Golden Master 契约测试的前提）。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional

from ..domain.errors import InvalidTransition, Unavailable
from ..domain.session import Session
from ..domain.states import Event, State
from ..physio.regulation import (
    DEFAULT_WEIGHTS as PHYSIO_DEFAULT_WEIGHTS,
)
from ..physio.regulation import (
    W_MAX,
    evaluate_physio,
    redistribute,
)
from ..ports import (
    DIMENSIONS,
    DimensionScore,
    EvalResult,
    InterviewEvent,
    PHYSIO_DIM,
    RegulationResult,
    RepoPort,
)
from ..scoring.aggregate import AggregateResult, aggregate
from ..scoring.level import classify
from .scheduler import TurnContext, TurnOutcome, TurnScheduler

#: 评分口径版本。任何口径变更都要递增，并写入每条记录，保证历史可比（方案 §10 风险表）。
RUBRIC_VERSION = "rubric-v1"

#: 缓冲题插入的反应幅度阈值（归一化反应幅度 r_i = ΔHR/pre）。
#: 超过该阈值说明候选人在这一题上应激过强，插入缓冲题让他回稳。
DEFAULT_REACTIVITY_THRESHOLD = 0.25


@dataclass(frozen=True)
class Consent:
    """授权记录。

    `base`（参与面试）与 `physio`（生理信号分析）**必须分开取得**——
    生理数据属于敏感个人信息，PIPL 第 28 条要求单独同意，不能用一揽子授权带过。
    """

    base: bool = False
    physio: bool = False
    media_recording: bool = False

    @property
    def can_continue(self) -> bool:
        return self.base


@dataclass(frozen=True)
class PhysioQuality:
    """一场生理采集的质量汇总（端侧上报，服务端只做聚合与门控）。"""

    median_snr: float = 0.0
    reject_ratio: float = 0.0
    algo_agreement: float = 0.0
    ibis: tuple[float, ...] = ()


@dataclass(frozen=True)
class InterviewConfig:
    """一场面试的配置。**全部可注入**，测试里用固定值保证确定性。"""

    questions: tuple[str, ...] = ()
    #: 适应性缓冲题库。缓冲题不计分，只用于让候选人回稳（自检 A3）。
    buffer_questions: tuple[str, ...] = ()
    #: 维度权重。默认含生理维度 8%（上限 W_MAX），生理不可用时权重自动归还。
    weights: Mapping[str, float] = field(default_factory=lambda: dict(PHYSIO_DEFAULT_WEIGHTS))
    adaptive_enabled: bool = False
    physio_enabled: bool = False
    reactivity_threshold: float = DEFAULT_REACTIVITY_THRESHOLD
    rubric_version: str = RUBRIC_VERSION
    #: 出报告所需的最少真实评分数。达不到就不出分（宁可没有，不可编造）。
    min_scored_turns: int = 1


@dataclass(frozen=True)
class Report:
    """最终报告。

    `available=False` 时 `score` 必为 `None`——这是本模块对外的红线：
    **没有任何一种失败路径会产出一个"看起来正常"的分数**。
    """

    session_id: str
    available: bool
    reason: Optional[str]
    score: Optional[float] = None
    level: Optional[str] = None
    dims: dict[str, float] = field(default_factory=dict)
    effective_weights: dict[str, float] = field(default_factory=dict)
    redistributed: dict[str, float] = field(default_factory=dict)
    counterfactuals: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    physio: Optional[RegulationResult] = None
    rubric_version: str = RUBRIC_VERSION
    n_scored_turns: int = 0
    n_buffer_turns: int = 0

    def to_dict(self) -> dict[str, Any]:
        physio = None
        if self.physio is not None:
            p = self.physio
            physio = {
                "available": p.available,
                "reason": p.reason,
                "score": p.score,
                "reliability": p.reliability,
                "weight_applied": p.weight_applied,
                "n_valid": p.n_valid,
                "components": dict(p.components),
            }
        return {
            "session_id": self.session_id,
            "available": self.available,
            "reason": self.reason,
            "score": self.score,
            "level": self.level,
            "dims": dict(self.dims),
            "effective_weights": dict(self.effective_weights),
            "redistributed": dict(self.redistributed),
            # NaN 不能进 JSON（json.dumps 会写出 NaN，被严格解析器拒绝），转 None
            "counterfactuals": {
                k: (None if v != v else v) for k, v in self.counterfactuals.items()
            },
            "notes": list(self.notes),
            "physio": physio,
            "rubric_version": self.rubric_version,
            "n_scored_turns": self.n_scored_turns,
            "n_buffer_turns": self.n_buffer_turns,
        }


@dataclass
class TurnRecord:
    """一轮的落地记录（写库用，也是回放的最小单元）。"""

    turn_id: str
    turn_index: int
    question: str
    scored: bool
    eval: Optional[EvalResult] = None
    degrade_reason: Optional[str] = None
    latency_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        ev = None
        if self.eval is not None:
            e = self.eval
            ev = {
                "dims": dict(e.dims),
                "score": e.score,
                "feedback": e.feedback,
                "provider": e.provider,
                "confidence": e.confidence,
                "latency_ms": e.latency_ms,
                "cost_usd": e.cost_usd,
                "degraded": e.degraded,
                "degrade_reason": e.degrade_reason,
            }
        return {
            "turn_id": self.turn_id,
            "turn_index": self.turn_index,
            "question": self.question,
            "scored": self.scored,
            "eval": ev,
            "degrade_reason": self.degrade_reason,
            "latency_ms": self.latency_ms,
        }


class InterviewService:
    """一场面试的编排者。

    所有跨模块的顺序决策都在这里，其余模块各自只做一件事。
    外部（gateway / 测试）通过本类驱动整场面试，不直接碰状态机或调度器。
    """

    def __init__(
        self,
        repo: RepoPort,
        scheduler: TurnScheduler,
        config: InterviewConfig,
        *,
        session_id: str = "s-1",
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.repo = repo
        self.scheduler = scheduler
        self.config = config
        self.session_id = session_id
        self.session = Session(session_id)
        self._clock = clock

        self.consent: Optional[Consent] = None
        self._q_index = 0
        self._buffer_queue: list[str] = []
        self._buffer_active = False
        self._turns: list[TurnRecord] = []
        self._next_turn_index = 0

        # 生理
        self._physio_events: list[InterviewEvent] = []
        self._baseline_bpm: Optional[float] = None
        self._physio_quality = PhysioQuality()
        self._physio_dropped_reason: Optional[str] = None
        self._physio_forced_reason: Optional[str] = None

    # ---------- 生命周期 ----------

    def create(self, meta: Optional[dict[str, Any]] = None) -> State:
        """IDLE → SETUP。"""
        self._apply(Event.CREATE)
        if meta:
            self.session.meta.update(meta)
        self._persist()
        return self.session.state

    def record_consent(self, consent: Consent) -> State:
        """SETUP → GREETING（同意）或 → ABORTED（拒绝）。

        未单独授权时生理开关被强制关闭：不是"收了数据再弃用"，而是**根本不收**。
        """
        self.consent = consent
        if not consent.can_continue:
            self._apply(Event.CONSENT_DENIED)
            self._persist()
            return self.session.state
        if self.config.physio_enabled and not consent.physio:
            self._physio_dropped_reason = "未取得生理信号分析的单独授权"
        self._apply(Event.CONSENT_OK)
        self._persist()
        return self.session.state

    def greeting_done(self) -> State:
        """GREETING → ASKING。"""
        self._apply(Event.GREETING_DONE)
        self._persist()
        return self.session.state

    # ---------- 出题 ----------

    def ask_next(self, *, tts_ok: bool = True) -> Optional[str]:
        """呈现下一题。

        - 处于 `BUFFER`：呈现缓冲题（不计分）。
        - 处于 `ASKING`：呈现核心题；题已出完则进 `CANDIDATE_QA` 并返回 None。
        - `tts_ok=False`：走 `TTS_FAILED`，降级为纯文本呈现，流程继续（不卡死）。

        返回 None 只表示"核心题出完了"，不代表异常。
        """
        ev = Event.TTS_DONE if tts_ok else Event.TTS_FAILED

        if self.session.state == State.BUFFER:
            question = self._buffer_queue.pop(0) if self._buffer_queue else ""
            self._apply(ev, {"question": question, "scored": False})
            return question

        if self.session.state != State.ASKING:
            raise InvalidTransition(self.session.state, Event.TTS_DONE)

        if self._q_index >= len(self.config.questions):
            self._apply(Event.NO_MORE_QUESTIONS)
            self._persist()
            return None

        question = self.config.questions[self._q_index]
        # 题号在这里前进：repeat_question 通过回退 _q_index 来重播同一题，
        # skip 则保持已前进的题号（跳过的题不再出现）。
        self._q_index += 1
        self._apply(ev, {"question": question, "scored": True})
        self._persist()
        return question

    # ---------- 作答 ----------

    async def submit_answer(self, text: str) -> TurnOutcome:
        """提交一轮作答。

        缓冲回合：只记录不评分，走 `BUFFER_DONE` 回 `ASKING`。
        正常回合：交给调度器在预算内跑完评分，据结果走 `EVAL_DONE` / `EVAL_DEGRADED`。
        """
        turn_id = f"{self.session_id}-t{self._next_turn_index}"
        turn_index = self._next_turn_index
        self._next_turn_index += 1

        if self._buffer_active:
            # ★ 缓冲题不计分：BUFFER_LISTENING --BUFFER_DONE--> ASKING。
            # 缓冲回合**不进 PROCESSING**，因此状态机上就不存在评分时机，
            # 也不调用评估器（省一次 LLM 调用）。做题号不变——缓冲题不消耗核心题。
            self._turns.append(
                TurnRecord(
                    turn_id=turn_id,
                    turn_index=turn_index,
                    question=self._last_question,
                    scored=False,
                )
            )
            self._buffer_active = False
            self._apply(
                Event.BUFFER_DONE, {"turn_id": turn_id, "answered": bool(text.strip())}
            )
            self._persist_turn(self._turns[-1])
            return TurnOutcome(turn_index=turn_index)

        self._apply(Event.ANSWER_COMMIT)

        ctx = TurnContext(
            turn_index=turn_index,
            question=self._last_question,
            answer=text,
            question_type="technical",
        )
        outcome = await self.scheduler.run_turn(ctx)

        record = TurnRecord(
            turn_id=turn_id,
            turn_index=turn_index,
            question=self._last_question,
            scored=outcome.eval is not None,
            eval=outcome.eval,
            degrade_reason=outcome.degrade_reason,
            latency_ms=outcome.latency_ms,
        )
        self._turns.append(record)
        self._apply(
            Event.EVAL_DEGRADED if outcome.degraded else Event.EVAL_DONE,
            {"turn_id": turn_id, "level": outcome.level},
        )
        self._persist_turn(record)
        self._maybe_insert_buffer(record)
        return outcome

    def skip(self) -> State:
        """跳过当前题（LISTENING → ASKING，或 CANDIDATE_QA → CLOSING）。"""
        self._apply(Event.SKIP)
        self._persist()
        return self.session.state

    def repeat_question(self) -> State:
        """重听题目（LISTENING → ASKING，同一题重新呈现，不出新题）。"""
        if self._q_index > 0:
            self._q_index -= 1
        self._apply(Event.REPEAT_QUESTION)
        self._persist()
        return self.session.state

    def extend_time(self) -> State:
        """延长作答时限（LISTENING 自环，只改时限不改状态语义）。"""
        self._apply(Event.EXTEND_TIME)
        return self.session.state

    def silence_timeout(self) -> State:
        self._apply(Event.SILENCE_TIMEOUT)
        self._persist()
        return self.session.state

    def abort(self) -> State:
        self._apply(Event.ABORT)
        self._persist()
        return self.session.state

    def fatal(self, detail: str = "") -> State:
        self._apply(Event.FATAL, {"detail": detail})
        self._persist()
        return self.session.state

    # ---------- 收尾 ----------

    def end_questioning(self) -> State:
        """提前结束出题（ASKING → CANDIDATE_QA）。

        候选人点"结束面试"时用：不必把剩下的题硬出完，但状态序列仍然完整
        （`NO_MORE_QUESTIONS` 是既有事件），因此事件日志照样可回放。
        """
        if self.session.state is State.ASKING:
            self._apply(Event.NO_MORE_QUESTIONS)
            self._persist()
        return self.session.state

    def finish_qa(self) -> State:
        self._apply(Event.QA_DONE)
        self._persist()
        return self.session.state

    def close(self) -> State:
        self._apply(Event.CLOSING_DONE)
        self._persist()
        return self.session.state

    def finalize(self) -> Report:
        """REPORTING → COMPLETED，产出报告并落库。

        任何"拿不到真实评估"的情形都返回 `available=False` 且 `score=None`。
        """
        scored = [t for t in self._turns if t.scored and t.eval is not None]
        n_buffer = sum(1 for t in self._turns if not t.scored)

        dims = _merge_dims([t.eval for t in scored])

        physio = self._evaluate_physio()
        if physio is not None and physio.available and physio.score is not None:
            dims[PHYSIO_DIM] = DimensionScore(
                value=physio.score,
                # 可靠性 R 直接当置信度交给通用门控：R<0.40 → 权重归零，与 §4.7 gate 同构
                confidence=physio.reliability,
                provider="rppg",
                degraded=False,
            )

        report: Report
        if len(scored) < self.config.min_scored_turns:
            # ★ 红线：没有真实评估就出报告时，绝不给任何数字
            report = Report(
                session_id=self.session_id,
                available=False,
                reason=(
                    f"真实评估轮数 {len(scored)} < {self.config.min_scored_turns}，"
                    "不出分（宁缺毋滥，不生成兜底分数）"
                ),
                dims={k: v.value for k, v in dims.items()},
                physio=physio,
                rubric_version=self.config.rubric_version,
                n_scored_turns=len(scored),
                n_buffer_turns=n_buffer,
            )
        else:
            try:
                agg = aggregate(dims, self.config.weights)
            except Unavailable as exc:
                report = Report(
                    session_id=self.session_id,
                    available=False,
                    reason=f"聚合不可用：{exc.reason}",
                    dims={k: v.value for k, v in dims.items()},
                    physio=physio,
                    rubric_version=self.config.rubric_version,
                    n_scored_turns=len(scored),
                    n_buffer_turns=n_buffer,
                )
            else:
                report = _build_report(
                    self.session_id, agg, dims, physio, len(scored), n_buffer,
                    self.config.rubric_version,
                )

        self._apply(Event.REPORT_DONE, {"available": report.available})
        self.session.meta["report"] = report.to_dict()
        self._persist()
        return report

    # ---------- 生理 ----------

    def set_baseline(self, bpm: Optional[float]) -> None:
        """设置静息基线心率（答题前的静息期测得）。"""
        self._baseline_bpm = bpm

    def set_physio_quality(
        self,
        *,
        median_snr: float = 0.0,
        reject_ratio: float = 0.0,
        algo_agreement: float = 0.0,
        ibis: Optional[list[float]] = None,
    ) -> None:
        self._physio_quality = PhysioQuality(
            median_snr=median_snr,
            reject_ratio=reject_ratio,
            algo_agreement=algo_agreement,
            ibis=tuple(ibis) if ibis else (),
        )

    def set_physio_unavailable_reason(self, reason: Optional[str]) -> None:
        """登记"生理维度为何整场不可用"的**根因**文案。

        没有这个入口时，环境门控（静息期 SNR 中位数 < 0.4）会被降级成
        "静息基线心率缺失"——把"灯太暗"说成"没测基线"，候选人会往错误方向自救。
        """
        self._physio_forced_reason = reason

    def add_physio_event(self, ev: InterviewEvent) -> bool:
        """接收一个应激事件的生理观测。返回是否被接受。

        未单独授权时**直接拒收**（返回 False 并记录原因），不进内存、不进库。
        """
        if not (self.config.physio_enabled and self.consent is not None and self.consent.physio):
            self._physio_dropped_reason = (
                self._physio_dropped_reason or "未取得生理信号分析的单独授权"
            )
            return False
        self._physio_events.append(ev)
        return True

    @property
    def physio_events(self) -> tuple[InterviewEvent, ...]:
        return tuple(self._physio_events)

    # ---------- 查询 ----------

    @property
    def scored_turns(self) -> tuple[TurnRecord, ...]:
        return tuple(t for t in self._turns if t.scored and t.eval is not None)

    @property
    def buffer_turns(self) -> tuple[TurnRecord, ...]:
        return tuple(t for t in self._turns if not t.scored)

    @property
    def last_question(self) -> str:
        return self._last_question

    # ---------- 内部 ----------

    def _maybe_insert_buffer(self, record: TurnRecord) -> None:
        """适应性调节：检测到过强应激时插入缓冲题（自检 A3：缓冲题不计分）。

        触发条件三者同时满足：开关打开 → 有库存 → 最近一次事件的归一化反应幅度
        r_i = ΔHR/pre 超过阈值。只有**刚评完分的正常回合**之后才可能插入，
        缓冲回合本身不再触发（避免连续缓冲）。
        """
        if not self.config.adaptive_enabled:
            return
        if not self._buffer_queue_candidates():
            return
        if self.session.state != State.ASKING:
            return
        if not self._physio_events:
            return
        last = self._physio_events[-1]
        r = last.ratio
        if r is None or r < self.config.reactivity_threshold:
            return

        self._buffer_queue.append(self._buffer_queue_candidates()[0])
        self._buffer_active = True
        self._apply(Event.INSERT_BUFFER, {"reason": f"r={r:.3f} ≥ {self.config.reactivity_threshold}"})

    def _buffer_queue_candidates(self) -> tuple[str, ...]:
        return self.config.buffer_questions

    def _evaluate_physio(self) -> Optional[RegulationResult]:
        if not self.config.physio_enabled:
            return None
        if self._physio_forced_reason:
            # 根因优先：环境门控/心律不齐这类"整场关闭"的判定，文案比"基线缺失"更有信息量。
            # 直接构造不可用结果：分数 None、权重 0、W_MAX 全额归还其他维度。
            return RegulationResult(
                available=False,
                reason=self._physio_forced_reason,
                baseline_bpm=self._baseline_bpm,
                n_events=len(self._physio_events),
                n_valid=sum(1 for e in self._physio_events if e.valid),
                components={},
                score=None,
                reliability=0.0,
                weight_applied=0.0,
                redistributed_to=redistribute(self.config.weights, PHYSIO_DIM, W_MAX),
            )
        if self._physio_dropped_reason:
            return evaluate_physio(
                events=self._physio_events,
                baseline_bpm=None,  # 未授权：走"基线缺失"分支，产出 reason 与全额权重归还
                n_expected=len(self.config.questions),
                median_snr=0.0,
                reject_ratio=1.0,
                algo_agreement=0.0,
                authorized=False,
                weights=self.config.weights,
            )
        return evaluate_physio(
            events=list(self._physio_events),
            baseline_bpm=self._baseline_bpm,
            n_expected=float(len(self.config.questions)),
            median_snr=self._physio_quality.median_snr,
            reject_ratio=self._physio_quality.reject_ratio,
            algo_agreement=self._physio_quality.algo_agreement,
            authorized=True,
            ibis=list(self._physio_quality.ibis) or None,
            weights=self.config.weights,
        )

    @property
    def _last_question(self) -> str:
        """当前正在作答的题目文本。"""
        if self.session.events:
            for ev in reversed(self.session.events):
                if ev.event in (Event.TTS_DONE, Event.TTS_FAILED) and "question" in ev.payload:
                    return str(ev.payload["question"])
        return ""

    def _apply(self, event: Event, payload: Optional[dict[str, Any]] = None) -> State:
        new_state = self.session.apply(event, payload)
        self.repo.append_event(
            self.session_id,
            {
                "seq": self.session.events[-1].seq,
                "event": event.value,
                "from": self.session.events[-1].from_state.value
                if self.session.events[-1].from_state
                else None,
                "to": new_state.value,
                "payload": payload or {},
            },
        )
        return new_state

    def _persist(self) -> None:
        self.repo.save_session(
            self.session_id,
            {
                "session_id": self.session_id,
                "state": self.session.state,
                "meta": dict(self.session.meta),
                "consent": (
                    {
                        "base": self.consent.base,
                        "physio": self.consent.physio,
                        "media_recording": self.consent.media_recording,
                    }
                    if self.consent
                    else None
                ),
                "rubric_version": self.config.rubric_version,
                "n_turns": len(self._turns),
            },
        )

    def _persist_turn(self, record: TurnRecord) -> None:
        self.repo.save_turn(self.session_id, record.to_dict())
        self._persist()


# ---------- 合并与报告构造 ----------


def _merge_dims(results: list[EvalResult]) -> dict[str, DimensionScore]:
    """把多轮评估合并成"每维度一个 DimensionScore"。

    - 分值：按置信度加权平均（置信度高的轮次说话权更大，而不是简单算术平均）。
      置信度全为 0 时退化为算术平均——此时门控会把它归零，不会产生伪分数。
    - 置信度：各轮置信度的算术平均（反映"整体有多少把握"，不做乐观叠加）。
    - degraded：任一轮降级则整维度降级（证据质量取最差的那次）。

    只接受真实的 `EvalResult`；**不存在"没有结果就填默认值"的分支**——
    没有评估的维度根本不会出现在返回的 dict 里，权重由聚合器归还其他维度。
    """
    out: dict[str, DimensionScore] = {}
    for dim in DIMENSIONS:
        vals = [(r.dims[dim], r.confidence) for r in results if dim in r.dims]
        if not vals:
            continue
        total_c = sum(c for _, c in vals)
        value = (
            sum(v * c for v, c in vals) / total_c
            if total_c > 0
            else sum(v for v, _ in vals) / len(vals)
        )
        out[dim] = DimensionScore(
            value=value,
            confidence=sum(c for _, c in vals) / len(vals),
            provider=",".join(sorted({r.provider for r in results})),
            degraded=any(r.degraded for r in results),
        )
    return out


def _build_report(
    session_id: str,
    agg: AggregateResult,
    dims: dict[str, DimensionScore],
    physio: Optional[RegulationResult],
    n_scored: int,
    n_buffer: int,
    rubric_version: str,
) -> Report:
    notes = list(agg.notes)
    if physio is None:
        notes.append("生理维度未启用（本场不采集、不计入）")
    elif not physio.available:
        notes.append(f"生理维度不计入：{physio.reason}")
        notes.append(
            f"其权重上限 {W_MAX:.0%} 已全额归还其他维度，总分不受影响"
        )
    else:
        notes.append(
            f"生理维度已计入：可靠性 R={physio.reliability:.2f}，"
            f"实得权重 {physio.weight_applied:.2%}（上限 {W_MAX:.0%}）"
        )

    score = float(agg.score)
    return Report(
        session_id=session_id,
        available=True,
        reason=None,
        score=round(score, 2),
        level=classify(score).level,
        dims={k: round(float(v.value), 2) for k, v in dims.items()},
        effective_weights=dict(agg.effective_weights),
        redistributed={k: v for k, v in agg.redistributed.items()},
        counterfactuals=dict(agg.counterfactuals),
        notes=notes,
        physio=physio,
        rubric_version=rubric_version,
        n_scored_turns=n_scored,
        n_buffer_turns=n_buffer,
    )


__all__ = [
    "Consent", "InterviewConfig", "InterviewService", "PhysioQuality",
    "Report", "TurnRecord", "RUBRIC_VERSION", "DEFAULT_REACTIVITY_THRESHOLD",
]
