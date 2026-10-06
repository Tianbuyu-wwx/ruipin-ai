"""信令 ↔ 业务：把 WebSocket 信封翻译成 `InterviewService` 调用（P0 的接线）。

为什么单列一层
--------------
`Gateway`（传输层）明确"不认识业务"，`InterviewService`（编排层）明确"不认识协议"。
两边都对，但**中间必须有东西把它们接起来**，否则 P0 的"文本面试闭环"就没有闭环——
网关只知道收到了一帧 `answer.commit`，编排器只接受 `await submit_answer(text)`
和一个 `TurnOutcome`。这一层就是那根线。

它承担四件事：
1. **协议 ↔ 领域对象**：`Envelope.payload` ↔ `Consent` / `EventAnchor` / `HRBatch`。
2. **状态迁移的回放下行**：领域的事件日志是权威，但前端需要**逐条**收到
   `state.changed`（断线重连要靠它重建 UI）。本层用游标把新增的 `StateEvent`
   转成信令，保证"日志里有的迁移，客户端一定收到过"。
3. **异步边界的适配**：`Gateway` 的 handler 是**同步**的（`Envelope -> list[Envelope]`），
   而评分是异步的。见 `SyncHandler` 的说明——这里不假装同步等待，而是明确区分
   "立即帧"与"稍后推送帧"。
4. **语音下行**：出题时把题干交给 `TTSPort`，产出音频块（`question.tts_chunk`，
   opcode `0x11`）与口型时间轴（`avatar.viseme`，opcode `0x12`）两个二进制帧族。
   见 `_start_tts`：**合成在后台跑，不阻塞出题**。

纪律
----
* 任何异常都翻成结构化 `error` 帧（带 `code`），**不吞**。业务失败（如评估不可用）
  走 `degradation.changed` 显式下行，绝不静默。
* 未实现的功能（如 `control.barge_in`，方案标注 Phase 3）回 `unknown_type` +
  明确文案，而不是假装成功。
* 语音链路同一条：**TTS 挂了就明说（L4 `tts_unavailable`），绝不发静音冒充成功**。
"""

from __future__ import annotations

import asyncio
import json
import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Mapping, Optional, Sequence

from ..domain.errors import BudgetExceeded, InvalidTransition, RuipinError, Unavailable
from ..domain.states import State
from ..orchestrator.degradation import badge_for
from ..orchestrator.scheduler import TurnScheduler
from ..orchestrator.service import (
    RUBRIC_VERSION,
    Consent,
    InterviewConfig,
    InterviewService,
    Report,
)
from ..physio.collector import StressRegulationCollector
from ..physio.regulation import W_MAX
from ..ports import (
    EnvQuality,
    EventAnchor,
    HRBatch,
    HRSample,
    Meter,
    RepoPort,
    SpeechPlan,
    TTSPort,
    VisemeEvent,
)
from .protocol import (
    OPCODE_KEY,
    ClientType,
    Envelope,
    ErrorCode,
    Opcode,
    ServerType,
    encode_binary,
    encode_text,
    error_envelope,
    make_envelope,
    state_changed,
)

#: 下行信封里用 `opcode` 字段标记"这其实是二进制帧"（由服务层转成 `encode_binary`）。
#: 网关只搬信封，不关心字节；把编码推迟到最外层，是为了让本层可以被纯逻辑测试。
#: 该键的**定义**在 `protocol`（帧格式层），因为网关也要靠它区分文本帧与媒体帧；
#: 这里从 protocol 导入后继续导出，`from .bridge import OPCODE_KEY` 不受影响。

#: TTS 音频的下行分片大小（字节）。
#: 4 KiB 是"一块能立刻发出去"的量级：太小会把帧数顶到几千（每帧 5 字节头也是开销），
#: 太大则首块要等更久、且一旦单帧丢失整块音频都废。真实链路还会先做 Opus 编码，
#: 编码后的块远小于此值，这里的切分只是**上界约束**。
DEFAULT_TTS_CHUNK_BYTES: int = 4096

#: 终态：会话已经走完，任何"继续推进"的指令都必须显式回绝而非静默忽略。
_TERMINAL_STATES: frozenset[State] = frozenset(
    {State.COMPLETED, State.ABORTED, State.FAILED}
)


@dataclass(frozen=True)
class BridgeConfig:
    """一场面试的对外配置（由 `session.create` 的信封填充）。"""

    questions: tuple[str, ...]
    buffer_questions: tuple[str, ...] = ()
    weights: Optional[Mapping[str, float]] = None
    adaptive_enabled: bool = False
    physio_enabled: bool = False
    rubric_version: str = RUBRIC_VERSION

    def to_interview_config(self) -> InterviewConfig:
        kwargs: dict[str, Any] = {
            "questions": self.questions,
            "buffer_questions": self.buffer_questions,
            "adaptive_enabled": self.adaptive_enabled,
            "physio_enabled": self.physio_enabled,
            "rubric_version": self.rubric_version,
        }
        if self.weights is not None:
            kwargs["weights"] = dict(self.weights)
        return InterviewConfig(**kwargs)


def _bad_payload(message: str, detail: Optional[str] = None) -> Envelope:
    return error_envelope(ErrorCode.BAD_FRAME, message, detail=detail)


def encode_viseme_timeline(visemes: Sequence[VisemeEvent]) -> bytes:
    """把口型时间轴编码成 `avatar.viseme` 二进制帧的负载（UTF-8 JSON）。

    **格式是与客户端定死的契约**（`web/app` 的协议层按同一格式解码）：
    `{"visemes": [{"t_ms": int, "viseme": str, "weight": float}, ...]}`。

    两条硬约束，都不是风格问题：
    1. **整条时间轴一帧下发**。客户端收到新的 viseme 帧是**整体替换**而非追加，
       分帧发送等于只有最后一帧生效——中间的全被丢掉，且丢得毫无痕迹。
    2. **空时间轴也要发**（`{"visemes": []}`）。客户端的时间轴是**跨题保留**的，
       题目没有口型数据时如果什么都不发，上一题的口型会**残留**下来继续动，
       音画看起来"在动"，但动的是上一题。显式清空优于沉默。
    """
    return json.dumps(
        {
            "visemes": [
                {
                    "t_ms": int(v.t_ms),
                    "viseme": str(v.viseme),
                    "weight": float(v.weight),
                }
                for v in visemes
            ]
        },
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


class InterviewBridge:
    """一条连接 ↔ 一场面试。

    Args:
        repo: 存储端口。
        scheduler: 每轮调度器（含评估器）。
        config: 面试配置。
        session_id: 会话标识。
        clock: 毫秒时间源（`Callable[[], float]`，返回**秒**），默认 `time.monotonic`。
        meter: 可观测端口，可空。
        tts: 语音合成端口；`None` 表示**本部署未配置语音**（纯文本面试）。
            注意区分"未配置"（不降级，见 `_start_tts`）与"配置了但调用失败"
            （必须降级下行）——把两者混成一种会让"没配 TTS"的部署天天报故障。
        tts_voice: 合成音色名，透传给 `TTSPort.synth`。
        tts_chunk_bytes: 音频分片大小（字节），必须 >= 1。
        pusher: "稍后推送帧"的出口。`SyncHandler` 会接管它（见 `set_pusher`）；
            不绑定时产物落在 `take_pending()` 里，供纯逻辑测试直接取。

    用法::

        bridge = InterviewBridge(repo=repo, scheduler=sched, config=cfg, session_id="s-1")
        frames = await bridge.handle(envelope)      # 一帧进，多帧出
    """

    def __init__(
        self,
        *,
        repo: RepoPort,
        scheduler: TurnScheduler,
        config: BridgeConfig,
        session_id: str,
        clock: Callable[[], float] = time.monotonic,
        meter: Optional[Meter] = None,
        tts: Optional[TTSPort] = None,
        tts_voice: str = "default",
        tts_chunk_bytes: int = DEFAULT_TTS_CHUNK_BYTES,
        pusher: Optional[Callable[[list[Envelope]], None]] = None,
    ) -> None:
        if tts_chunk_bytes < 1:
            raise ValueError(f"tts_chunk_bytes 必须 >= 1，实际: {tts_chunk_bytes}")
        self.repo = repo
        self.scheduler = scheduler
        self.config = config
        self.session_id = session_id
        self._clock = clock
        self._meter = meter
        self._tts = tts
        self._tts_voice = tts_voice
        self._tts_chunk_bytes = int(tts_chunk_bytes)
        self._pusher = pusher
        self._pending: list[Envelope] = []
        self._tts_tasks: set[asyncio.Task[None]] = set()

        self.service = InterviewService(
            repo=repo,
            scheduler=scheduler,
            config=config.to_interview_config(),
            session_id=session_id,
            clock=clock,
        )
        self._state_cursor = 0
        self._pending_text = ""
        self._t_q_ms: Optional[int] = None
        self._t_a_ms: Optional[int] = None
        self._t_e_ms: Optional[int] = None
        self._turn_index = 0
        # 当前轮的标识。**出题时分配一次**，随后 `question.start` / `transcript.final`
        # / `eval.done` 都用这一对值——三轮的三种帧必须能被前端唯一地对应起来。
        self._current_turn_id = ""
        self._current_turn_index = -1

        self.collector: Optional[StressRegulationCollector] = (
            StressRegulationCollector(session_id, authorized=True)
            if config.physio_enabled
            else None
        )

    # ---------- 入口 ----------

    async def handle(self, env: Envelope) -> list[Envelope]:
        """处理一帧上行信封，返回要下行的信封列表。

        **不抛异常给调用方**：所有失败都翻成 `error` 帧（传输层不该被业务异常打断）。
        """
        handler = _ROUTES.get(env.type)
        if handler is None:
            return [
                error_envelope(
                    ErrorCode.UNKNOWN_TYPE,
                    f"未知消息类型：{env.type}",
                    seq=env.seq,
                    ts=self._now_ms(),
                    session_id=self.session_id,
                )
            ]
        try:
            return await handler(self, env)
        except InvalidTransition as exc:
            return [
                error_envelope(
                    ErrorCode.BAD_FRAME,
                    f"当前状态不允许该操作：{exc}",
                    seq=env.seq,
                    ts=self._now_ms(),
                    session_id=self.session_id,
                )
            ]
        except Unavailable as exc:
            # 能力不可用：显式降级下行，绝不静默（方案 §3.4）
            return [
                self._degradation_frame(1, f"{exc.provider} 不可用：{exc.reason}"),
                error_envelope(
                    ErrorCode.INTERNAL,
                    f"能力不可用：{exc.reason}",
                    seq=env.seq,
                    ts=self._now_ms(),
                    session_id=self.session_id,
                ),
            ]
        except BudgetExceeded as exc:
            return [
                self._degradation_frame(3, f"预算硬顶：{exc}"),
                error_envelope(
                    ErrorCode.INTERNAL, str(exc), seq=env.seq, ts=self._now_ms(),
                    session_id=self.session_id,
                ),
            ]
        except RuipinError as exc:
            return [
                error_envelope(
                    ErrorCode.INTERNAL, str(exc), seq=env.seq,
                    ts=self._now_ms(), session_id=self.session_id,
                )
            ]

    # ---------- 客户端消息 ----------

    async def on_session_create(self, env: Envelope) -> list[Envelope]:
        """`session.create`：建会话（IDLE → SETUP）。"""
        self.service.create({"candidate_id": env.payload.get("candidate_id")})
        return self._drain_states()

    async def on_consent_grant(self, env: Envelope) -> list[Envelope]:
        """`consent.grant`：分项授权 → GREETING → 自动进入第一题。

        拒绝参加（`base=False`）→ ABORTED，不再出题。
        """
        payload = env.payload
        consent = Consent(
            base=bool(payload.get("base")),
            # 兼容两种字段名：方案里写 physiology，前端可能写 physio
            physio=bool(payload.get("physiology", payload.get("physio", False))),
            media_recording=bool(payload.get("screen", False)),
        )
        self.service.record_consent(consent)
        out = self._drain_states()
        if not consent.can_continue:
            return out
        self.service.greeting_done()
        out += self._drain_states()
        out += self._advance()
        return out

    async def on_answer_text(self, env: Envelope) -> list[Envelope]:
        """`answer.text`：累积作答文本，回流式转写。

        第一步作答文本到达时记为 `t_a`（详设 §4.1 的锚点之一）。
        """
        text = env.payload.get("text")
        if not isinstance(text, str):
            return [_bad_payload("answer.text 的 payload.text 必须是字符串")]
        if self._t_a_ms is None:
            self._t_a_ms = self._now_ms()
        self._pending_text += text
        return [
            make_envelope(
                ServerType.TRANSCRIPT_PARTIAL,
                {"text": self._pending_text},
                ts=self._now_ms(),
                session_id=self.session_id,
            )
        ]

    async def on_answer_commit(self, env: Envelope) -> list[Envelope]:
        """`answer.commit`：结算本轮并推进到下一题。"""
        text = env.payload.get("text")
        answer = self._pending_text if text is None else str(text)
        out = await self._commit_turn(answer)
        out += self._advance()
        return out

    async def _commit_turn(self, answer: str) -> list[Envelope]:
        """把当前这轮跑完并产出相应下行帧（不推进到下一题）。"""
        self._t_e_ms = self._now_ms()
        outcome = await self.service.submit_answer(answer)
        # 生理锚点用**服务内部**的轮次号，保证与 Habituation 回归的自变量 i 对齐
        self._close_physio_event(outcome.turn_index)

        out = self._drain_states()
        out.append(
            make_envelope(
                ServerType.TRANSCRIPT_FINAL,
                {
                    "text": answer,
                    "turn_id": self._current_turn_id,
                    "turn_index": self._current_turn_index,
                },
                ts=self._now_ms(),
                session_id=self.session_id,
            )
        )
        if outcome.eval is not None:
            e = outcome.eval
            out.append(
                make_envelope(
                    ServerType.EVAL_DONE,
                    {
                        # 与 `question.start` 同一对标识：前端据此把评估挂回那一轮。
                        # 原先只给 `outcome.turn_index`（服务侧自己的计数），
                        # 前端拿不到 `turn_id` 就只能新建一个键为空的回合——
                        # 实测表现为逐轮评估里多出一条**没有题干的幽灵回合**，
                        # 而三条真评估全被挤到它上面。
                        "turn_id": self._current_turn_id,
                        "turn_index": self._current_turn_index,
                        "score": e.score,
                        "dims": dict(e.dims),
                        "feedback": e.feedback,
                        "provider": e.provider,
                        "confidence": e.confidence,
                        "degraded": e.degraded,
                    },
                    ts=self._now_ms(),
                    session_id=self.session_id,
                )
            )
        if outcome.degraded:
            out.append(
                self._degradation_frame(
                    outcome.level, outcome.degrade_reason or "本轮评估降级", outcome.ui_badge
                )
            )
        self._pending_text = ""
        self._t_a_ms = None
        return out

    async def on_control_skip(self, env: Envelope) -> list[Envelope]:
        """`control.skip`：跳题/跳过提问环节。"""
        self.service.skip()
        out = self._drain_states()
        if self._pending_text:
            self._pending_text = ""
            self._t_a_ms = None
        out += self._advance()
        return out

    async def on_control_end(self, env: Envelope) -> list[Envelope]:
        """`control.end`：收尾并出报告。

        候选人可能在任何状态点结束：
        * 正在作答（LISTENING）→ 先把这轮跑完，别丢掉已经说出口的内容；
        * 正在出题（ASKING）→ 直接转入提问环节，不把剩下的题硬出完；
        * 已在提问环节（CANDIDATE_QA）→ 直接收尾。

        三种情况都走既有的 FSM 事件，因此**状态序列始终完整、可回放**。

        已经结束的会话（completed/aborted/failed）再收到 `control.end` 时**显式回错**，
        而不是静默返回空帧——客户端会一直等一个永远不会再来的报告。
        """
        svc = self.service
        out: list[Envelope] = []
        if svc.session.state in _TERMINAL_STATES:
            return [
                error_envelope(
                    ErrorCode.BAD_FRAME,
                    f"会话已结束（{svc.session.state.value}），无需再次结束",
                    seq=env.seq,
                    ts=self._now_ms(),
                    session_id=self.session_id,
                )
            ]

        if svc.session.state is State.LISTENING:
            out += await self._commit_turn(self._pending_text)
        if svc.session.state is State.ASKING:
            svc.end_questioning()
        if svc.session.state is State.CANDIDATE_QA:
            svc.finish_qa()
        if svc.session.state is State.CLOSING:
            svc.close()

        out += self._drain_states()
        if svc.session.state is State.REPORTING:
            report = self._finalize()
            out += self._drain_states()
            out.append(self._report_frame(report))
        return out

    async def on_barge_in(self, env: Envelope) -> list[Envelope]:
        """`control.barge_in`：方案标注 Phase 3，本版明确回绝，不假装成功。"""
        return [
            error_envelope(
                ErrorCode.UNKNOWN_TYPE,
                "打断（barge_in）尚未实现（方案标注为 Phase 3）",
                seq=env.seq,
                ts=self._now_ms(),
                session_id=self.session_id,
            )
        ]

    async def on_physio_batch(self, env: Envelope) -> list[Envelope]:
        """`physio.batch`：端侧 rPPG 结果批量上行（详设 §9）。

        **只在已单独授权时才接收**；未授权时直接丢弃并回 `error`，
        数据连内存都不进（PIPL 第 28 条：不是"收了再弃用"）。
        """
        if self.collector is None:
            return [
                error_envelope(
                    ErrorCode.UNAUTHORIZED,
                    "本场未启用生理采集（需在 session.create 中开启 physio 并单独授权）",
                    seq=env.seq,
                    ts=self._now_ms(),
                    session_id=self.session_id,
                )
            ]
        consent = self.service.consent
        if consent is None or not consent.physio:
            return [
                error_envelope(
                    ErrorCode.UNAUTHORIZED,
                    "未取得生理信号分析的单独授权，数据已拒收",
                    seq=env.seq,
                    ts=self._now_ms(),
                    session_id=self.session_id,
                )
            ]

        payload = env.payload
        samples: list[HRSample] = []
        for raw in payload.get("samples", ()):
            if not isinstance(raw, dict) or "t_ms" not in raw:
                return [_bad_payload("physio.batch 的样本必须含 t_ms")]
            bpm = raw.get("bpm")
            samples.append(
                HRSample(
                    t_ms=int(raw["t_ms"]),
                    bpm=None if bpm is None else float(bpm),
                    snr=float(raw.get("snr", 0.0)),
                    algo_spread=float(raw.get("algo_spread", 0.0)),
                )
            )
        env_q = payload.get("env")
        batch = HRBatch(
            session_id=self.session_id,
            samples=tuple(samples),
            algo_agreement=float(payload.get("algo_agreement", 0.0)),
            rejected_windows=int(payload.get("rejected_windows", 0)),
            env=(
                EnvQuality(
                    fps=float(env_q.get("fps", 0.0)),
                    brightness=float(env_q.get("brightness", 0.0)),
                    face_ratio=float(env_q.get("face_ratio", 0.0)),
                    flicker=float(env_q.get("flicker", 0.0)),
                )
                if isinstance(env_q, dict)
                else None
            ),
        )
        self.collector.push_batch(batch)

        if payload.get("rest_end"):
            baseline = self.collector.baseline_from_rest()
            self.service.set_baseline(baseline)
            self.service.set_physio_unavailable_reason(self.collector.baseline_gate_reason)
            return [
                make_envelope(
                    ServerType.METRIC_HR,
                    {
                        "phase": "baseline",
                        "baseline_bpm": baseline,
                        "available": baseline is not None,
                        "reason": self.collector.baseline_gate_reason,
                    },
                    ts=self._now_ms(),
                    session_id=self.session_id,
                )
            ]
        return []

    async def on_media(self, env: Envelope) -> list[Envelope]:
        """`media.*` 文本形态：媒体应走二进制帧，这里只做明确回绝。

        静默接受会让前端以为上传成功，实际一个字节都没到——那是最坏的失败模式。
        """
        return [
            error_envelope(
                ErrorCode.BAD_FRAME,
                f"{env.type} 必须走二进制帧（opcode 0x0X），不接受文本帧",
                seq=env.seq,
                ts=self._now_ms(),
                session_id=self.session_id,
            )
        ]

    # ---------- 内部 ----------

    def _advance(self) -> list[Envelope]:
        """若当前可出题则出下一题（含题尽转入提问环节）。"""
        if self.service.session.state is not State.ASKING:
            return []
        question = self.service.ask_next()
        out = self._drain_states()
        if question is None:
            return out

        turn_index = self._turn_index
        turn_id = f"{self.session_id}-t{turn_index}"
        self._turn_index += 1
        self._current_turn_id = turn_id
        self._current_turn_index = turn_index
        self._t_q_ms = self._now_ms()
        self._t_a_ms = None
        self._pending_text = ""
        # `audio_sample_rate` 只在配置了 TTS 时才有意义；前端不读也不受影响。
        # 放在 `question.start` 而不是音频帧里：二进制帧只有 `[opcode][长度][负载]`，
        # 没有任何地方能放元数据（方案 §2.4 定死的线格式）。
        #
        # `turn_id` 与 `turn_index` 必须**成对下发**：前者是前端做关联的键，
        # 后者只用于显示"第 N 题"。只给 turn_id 时前端只能拿状态机里那个
        # 从不确定前进的轮次号当序号，实测表现为状态栏永远停在"第 1 题"。
        out.append(
            make_envelope(
                ServerType.QUESTION_START,
                {
                    "turn_id": turn_id,
                    "turn_index": turn_index,
                    "text": question,
                    "tts": self._tts is not None,
                    "audio_sample_rate": self._tts_sample_rate,
                },
                ts=self._now_ms(),
                session_id=self.session_id,
            )
        )
        self._start_tts(question, turn_id)
        return out

    # ---------- 语音下行 ----------

    @property
    def _tts_sample_rate(self) -> int:
        """本部署的合成采样率；未配置 TTS 时为 0（"不存在"而不是"未知的 24000"）。"""
        if self._tts is None:
            return 0
        rate = getattr(self._tts, "sample_rate", None)
        return int(rate) if isinstance(rate, int) and rate > 0 else 24000

    def set_pusher(self, pusher: Optional[Callable[[list[Envelope]], None]]) -> None:
        """绑定"稍后推送帧"的出口。

        `SyncHandler` 构造时把这一步做掉，于是 TTS 的产物直接进服务层的 outbox，
        由读循环在下一轮 poll 时发出去。**必须在 `SyncHandler.__init__` 里绑定**：
        晚绑定（例如第一次 `__call__` 时才绑）会让"绑定前就已完成的 TTS"落在
        `_pending` 里，而那批帧**永远不会再被取走**──表现是偶尔丢第一题的语音。
        """
        self._pusher = pusher

    def take_pending(self) -> list[Envelope]:
        """取走并清空本地待推送帧（未绑定 pusher 时的出口）。"""
        frames = list(self._pending)
        self._pending.clear()
        return frames

    def _push(self, frames: list[Envelope]) -> None:
        """把"稍后产生"的帧投到出口。"""
        if not frames:
            return
        if self._pusher is not None:
            self._pusher(frames)
        else:
            self._pending.extend(frames)

    @property
    def tts_tasks_pending(self) -> int:
        """在途 TTS 合成任务数（0 = 音频/口型已全部产出）。"""
        return sum(1 for t in self._tts_tasks if not t.done())

    async def joined(self) -> None:
        """等所有在途 TTS 合成任务结束。

        与 `SyncHandler.joined()` 是同一条纪律的一半：那条等"业务指令处理完"，
        这条等"语音产出完"。少了它，服务端会在音频还没推完时就关连接──
        客户端只会听到**半句话**，而服务端日志一切正常。
        """
        while True:
            tasks = [t for t in self._tts_tasks if not t.done()]
            if not tasks:
                return
            # 用 gather(return_exceptions=True)：任务内部已把异常翻成降级帧，
            # 这里的异常只可能是"真的没人接住"的编程错误，不能让它打断收尾。
            await asyncio.gather(*tasks, return_exceptions=True)

    def _start_tts(self, question: str, turn_id: str) -> None:
        """为刚出的这道题启动语音合成。**后台跑，不阻塞出题。**

        为什么是后台任务而不是在这里 `await`：方案 §6.3 要求首字延迟 ~150 ms，
        而题干文本必须**立刻**下发（前端先显示题目，再出声）。若在此 await，
        TTS 的整段延迟会直接变成"题目显示延迟"——网络稍抖就是几秒白屏。
        两者的关联靠 `turn_id`（文本帧里）+ **帧顺序**（二进制帧没有元数据字段）。

        `self._tts is None` 时**什么都不做、也不降级**：那是"本部署没接语音"
        的配置事实，不是故障。反之（配了但失败）必须降级，见 `_synth_and_push`。
        """
        if self._tts is None:
            return
        task = asyncio.ensure_future(self._synth_and_push(question, turn_id))
        self._tts_tasks.add(task)
        task.add_done_callback(self._tts_tasks.discard)

    async def _synth_and_push(self, question: str, turn_id: str) -> None:
        """合成一道题并推送音频块 / 口型 / 降级帧。**本协程绝不向外抛。**"""
        tts = self._tts
        if tts is None:  # pragma: no cover - 仅 _start_tts 会启动本协程
            return
        try:
            plan = await tts.synth(question, voice=self._tts_voice)
        except Unavailable as exc:
            self._push([self._degradation_frame(4, f"tts_unavailable: {exc.reason}")])
            return
        except asyncio.CancelledError:
            # 取消要原样外抛（吞掉等于拒绝被取消），交给 asyncio 收尾。
            raise
        except Exception as exc:  # noqa: BLE001 - 后台任务的异常最容易被丢掉
            # 这里的 `except Exception` 不是为了"容错"，而是因为后台任务的异常
            # 若逃出去，日志里只有一行 "Task exception was never retrieved"，
            # 业务侧完全看不到。宁可粗一点，也要把它变成用户可见的降级。
            self._push(
                [
                    self._degradation_frame(
                        4, f"tts_unavailable: {type(exc).__name__}: {exc}"
                    )
                ]
            )
            return

        if not plan.audio:
            # 空音频 = 静音冒充成功，本项目头号红线（见 adapters/tts.py 的红线一节）。
            self._push([self._degradation_frame(4, "tts_unavailable: 合成返回空音频")])
            return

        self._push(self._speech_frames(plan, turn_id))

    def _speech_frames(self, plan: SpeechPlan, turn_id: str) -> list[Envelope]:
        """把一次合成的产物切成下行帧（音频块在前，口型时间轴在后）。

        顺序不是随意的：口型帧携带**整条**时间轴，客户端拿到它就开始按时间轴驱动
        口型。先给音频再给时间轴，客户端至少有"音频已就绪"这个前提，
        不会出现"口型在动但一个字都没响"的观感。
        """
        ts = self._now_ms()
        frames: list[Envelope] = []
        size = self._tts_chunk_bytes
        total = max(1, math.ceil(len(plan.audio) / size))
        for i in range(total):
            frames.append(
                media_envelope(
                    ServerType.QUESTION_TTS_CHUNK,
                    Opcode.TTS_AUDIO,
                    plan.audio[i * size : (i + 1) * size],
                    ts=ts,
                    session_id=self.session_id,
                )
            )
        # 空时间轴也发：客户端的时间轴是跨题保留的，不发就等于让上一题的口型
        # 继续演这一题（见 `encode_viseme_timeline` 的说明）。
        frames.append(
            media_envelope(
                ServerType.AVATAR_VISEME,
                Opcode.VISEME,
                encode_viseme_timeline(plan.visemes),
                ts=ts,
                session_id=self.session_id,
            )
        )
        return frames

    def _drain_states(self) -> list[Envelope]:
        """把新增的领域状态迁移逐条翻成 `state.changed`。

        用游标而不是"只报当前状态"：断线重连要靠序列重建 UI，
        只报最终态会让中间过程永久丢失（也就无法回放）。
        """
        events = self.service.session.events
        fresh = events[self._state_cursor :]
        self._state_cursor = len(events)
        ts = self._now_ms()
        return [
            state_changed(
                ev.from_state.value if ev.from_state is not None else None,
                ev.to_state.value,
                ev.event.value,
                ts=ts,
                session_id=self.session_id,
            )
            for ev in fresh
        ]

    def _close_physio_event(self, turn_index: int) -> None:
        """本轮作答结束时登记生理锚点（t_q/t_a/t_e，详设 §4.1）。"""
        if self.collector is None or self._t_q_ms is None or self._t_e_ms is None:
            return
        self.collector.mark_event(
            EventAnchor(
                turn_index=turn_index,
                t_q_ms=self._t_q_ms,
                t_a_ms=self._t_a_ms if self._t_a_ms is not None else self._t_q_ms,
                t_e_ms=self._t_e_ms,
            )
        )

    def _finalize(self) -> Report:
        """把生理聚合结果交给服务，再出报告。"""
        if self.collector is not None:
            self.service.set_physio_quality(
                median_snr=self.collector.median_snr(),
                reject_ratio=self.collector.reject_ratio(),
                algo_agreement=self.collector.algo_agreement(),
            )
            if self.collector.baseline_gate_reason is not None:
                self.service.set_physio_unavailable_reason(
                    self.collector.baseline_gate_reason
                )
            for ev in self.collector.events:
                self.service.add_physio_event(ev)
        return self.service.finalize()

    def _report_frame(self, report: Report) -> Envelope:
        payload = report.to_dict()
        payload["physio_weight_cap"] = W_MAX
        return make_envelope(
            ServerType.REPORT_READY, payload, ts=self._now_ms(), session_id=self.session_id
        )

    def _degradation_frame(self, level: int, reason: str, badge: str = "") -> Envelope:
        """降级必须**显式**告知用户（方案 §3.4：禁止静默降级）。"""
        return make_envelope(
            ServerType.DEGRADATION_CHANGED,
            {"level": level, "reason": reason, "badge": badge or badge_for(level)},
            ts=self._now_ms(),
            session_id=self.session_id,
        )

    def _now_ms(self) -> int:
        return int(self._clock() * 1000.0)


#: 路由表：类型 → 处理函数。刻意用显式字典而不是 if/elif 链，
#: 这样"协议里有哪些类型"与"我们实现了哪些"可以直接对拍（见测试）。
_ROUTES: dict[str, Callable[[InterviewBridge, Envelope], Awaitable[list[Envelope]]]] = {
    str(ClientType.SESSION_CREATE): InterviewBridge.on_session_create,
    str(ClientType.CONSENT_GRANT): InterviewBridge.on_consent_grant,
    str(ClientType.ANSWER_TEXT): InterviewBridge.on_answer_text,
    str(ClientType.ANSWER_COMMIT): InterviewBridge.on_answer_commit,
    str(ClientType.CONTROL_SKIP): InterviewBridge.on_control_skip,
    str(ClientType.CONTROL_END): InterviewBridge.on_control_end,
    str(ClientType.CONTROL_BARGE_IN): InterviewBridge.on_barge_in,
    str(ClientType.PHYSIO_BATCH): InterviewBridge.on_physio_batch,
    str(ClientType.MEDIA_AUDIO): InterviewBridge.on_media,
    str(ClientType.MEDIA_VIDEO): InterviewBridge.on_media,
    str(ClientType.MEDIA_SCREEN): InterviewBridge.on_media,
}

ROUTES: Mapping[str, Callable[..., Any]] = _ROUTES


def split_outgoing(frames: list[Envelope]) -> tuple[list[str], list[bytes]]:
    """把下行信封拆成「文本帧」与「二进制帧」。

    带 `opcode` 字段的信封是媒体（如 TTS 音频块 `0x11`、viseme `0x12`），
    由最外层编码成二进制帧；其余走 JSON 文本帧。把这一步放在最外层，
    本层就能被纯逻辑测试（不需要真的连一条 WebSocket）。
    """
    texts: list[str] = []
    binaries: list[bytes] = []
    for env in frames:
        opcode = env.payload.get(OPCODE_KEY)
        if opcode is None:
            texts.append(encode_text(env))
            continue
        data = env.payload.get("data", b"")
        binaries.append(encode_binary(int(opcode), bytes(data)))
    return texts, binaries


def media_envelope(
    mtype: ServerType | str,
    opcode: Opcode | int,
    data: bytes,
    *,
    seq: int = 0,
    ts: int = 0,
    session_id: str = "",
    **extra: Any,
) -> Envelope:
    """构造一个"待编码成二进制帧"的信封（供 TTS / viseme 下发使用）。"""
    payload: dict[str, Any] = {OPCODE_KEY: int(opcode), "data": data}
    payload.update(extra)
    return make_envelope(mtype, payload, seq=seq, ts=ts, session_id=session_id)


class SyncHandler:
    """把 `InterviewBridge` 适配成 `Gateway` 需要的**同步** handler。

    `Gateway` 的 handler 签名是 `Envelope -> list[Envelope]`（同步），而评分是异步的。
    **不做假同步**（在事件循环里同步等异步必然死锁），而是把语义拆成两段：

    * **立即帧**：handler 当场返回的帧（`state.changed`、`question.start` 等）；
    * **推送帧**：异步完成后落在 `outbox` 里，由服务层的写循环在下一轮取走发送。
      两类异步产物共用这一个 outbox：评估结果，以及 TTS 的音频块 / 口型时间轴
      （桥接层通过 `set_pusher` 把出口接到这里）。于是**服务层的读循环完全不需要
      知道"语音"这回事**——它只搬 outbox。

    这是真实 WS 服务的常见形态（先 ack 再 push），也便于测试：
    `await asyncio.sleep(0)` 后 `drain()` 即可拿到推送帧。

    **同连接的指令严格串行**（踩过的坑，务必保留）
    ----------------------------------------------
    第一版实现给每一帧各开一个 task，结果同一连接的相邻两帧会**并发交错**：
    `answer.commit` 内部要 `await`（评分），一 `await` 就轮到下一帧开跑，
    于是 `control.end` 可能在 commit 尚未生效时就执行——实测会看到
    ①凭空冒出的"无可用作答文本"降级（`_pending_text` 被并发清空）
    ②会话停在 `candidate_qa`，报告再也出不来。
    而这一切在"逐帧 await 调用"的测试里**完全看不出来**，只有真实读循环才暴露。

    所以这里改成**队列 + 单工作协程**：同连接的帧先进先出、逐帧跑完再跑下一帧。
    这不是性能取舍，是语义要求——WebSocket 是有序流，业务是状态机，
    并发应用同一客户端的两条指令**按定义就是错的**。
    （方案 §11.1-2：面试是回合制，不做抢话打断，因此串行不牺牲任何体验。）

    Args:
        bridge: 被包装的桥接器。
        loop: 事件循环；必须与网关被调用时所在的循环一致。
    """

    def __init__(self, bridge: InterviewBridge, loop: asyncio.AbstractEventLoop) -> None:
        self.bridge = bridge
        self.loop = loop
        self.outbox: list[Envelope] = []
        self._queue: deque[Envelope] = deque()
        self._worker: Optional[asyncio.Task[None]] = None
        # 立刻接管桥接层的推送出口：TTS 的产物与业务帧共用同一个 outbox，
        # 服务层的读循环因此不需要知道"语音"这回事（它只搬 outbox）。
        bridge.set_pusher(self.outbox.extend)

    def __call__(self, env: Envelope) -> list[Envelope]:
        """入队并（必要时）唤醒工作协程；**立即返回空**（真结果稍后推送）。"""
        self._queue.append(env)
        if self._worker is None or self._worker.done():
            self._worker = self.loop.create_task(self._drain_queue())
        return []

    async def _drain_queue(self) -> None:
        """逐帧串行处理，直到队列空。空则退出，下次 `__call__` 再唤醒。"""
        while self._queue:
            env = self._queue.popleft()
            try:
                frames = await self.bridge.handle(env)
            except Exception as exc:  # noqa: BLE001 - 桥接层本不该抛，真抛了也不能吞
                frames = [
                    error_envelope(
                        ErrorCode.INTERNAL,
                        f"处理 {env.type} 失败：{type(exc).__name__}: {exc}",
                        seq=env.seq,
                    )
                ]
            self.outbox.extend(frames)

    @property
    def pending(self) -> int:
        """在途工作量 = 队列长度 + 工作协程 + 在途 TTS 合成（0 表示彻底静默）。

        把 TTS 算进来是必须的：读循环靠 `pending == 0` 判断"没事可做了"。
        漏算它，收尾时就会在音频推完之前判定空闲并关闭连接——
        副作用是"偶尔最后一道题的语音听不全"，而统计上一切正常。
        """
        running = 1 if (self._worker is not None and not self._worker.done()) else 0
        return len(self._queue) + running + self.bridge.tts_tasks_pending

    def drain(self) -> list[Envelope]:
        """取走并清空推送帧。"""
        frames = list(self.outbox)
        self.outbox.clear()
        return frames

    async def joined(self) -> None:
        """等所有在途任务完成（队列抽干 + 工作协程退出 + TTS 产出完）。

        服务端在**关闭连接前**必须调用它，并且**之后还要再 `drain()` 一次**——
        否则最后一条指令（常常正是 `control.end`）产生的报告会永远留在
        outbox 里，而日志上一切正常。
        """
        while self._worker is not None and not self._worker.done():
            await asyncio.gather(self._worker, return_exceptions=True)
        if self._queue:  # 异常路径下的兜底：worker 没把队列抽干
            await self._drain_queue()
        self._worker = None
        # 业务跑完了，最后一道题的语音可能还在合成中；不等它就会"说到一半断线"。
        await self.bridge.joined()


__all__ = [
    "DEFAULT_TTS_CHUNK_BYTES",
    "OPCODE_KEY",
    "ROUTES",
    "BridgeConfig",
    "InterviewBridge",
    "SyncHandler",
    "encode_viseme_timeline",
    "media_envelope",
    "split_outgoing",
]
