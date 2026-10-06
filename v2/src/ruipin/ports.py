"""端口（Port）定义 —— 核心层唯一允许 import 的外部契约。

所有外部能力（评分模型 / VLM / TTS / 形象 / 存储 / 生理）都藏在适配器后面，
核心只认这里的 Protocol。换实现 = 只改 adapter。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable

# 六个业务维度（顺序即报告呈现顺序）
DIMENSIONS: tuple[str, ...] = (
    "technical",
    "communication",
    "completeness",
    "problem_solving",
    "teamwork",
    "leadership",
)

# 生理维度（可选，权重受可靠性门控）
PHYSIO_DIM = "stress_regulation"


@dataclass(frozen=True)
class EvalRequest:
    question: str
    answer: str
    question_type: str = "technical"
    transcript: Optional[str] = None
    has_audio: bool = False
    has_video: bool = False
    keywords: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvalResult:
    """评估结果。**不允许**表示"崩溃后的兜底分数"。

    degraded=True 时必须给出 degrade_reason，且该结果只用于标注，
    不得被当作与真实评估同质的分数参与聚合（聚合器会对 degraded 降权）。
    """

    dims: dict[str, float]
    score: float
    feedback: str
    provider: str
    confidence: float
    latency_ms: int = 0
    cost_usd: float = 0.0
    degraded: bool = False
    degrade_reason: Optional[str] = None

    def __post_init__(self) -> None:
        missing = [d for d in DIMENSIONS if d not in self.dims]
        if missing:
            raise ValueError(f"EvalResult 缺少维度: {missing}")
        for k, v in self.dims.items():
            if not isinstance(v, (int, float)):
                raise TypeError(f"维度 {k} 非数值: {v!r}")
            if not (0.0 <= float(v) <= 100.0):
                raise ValueError(f"维度 {k} 越界 [0,100]: {v}")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence 越界 [0,1]: {self.confidence}")
        if self.degraded and not self.degrade_reason:
            raise ValueError("degraded=True 必须给出 degrade_reason")


@runtime_checkable
class Evaluator(Protocol):
    """评分器。失败时必须抛 Unavailable，禁止返回合成/default 分数。"""

    name: str

    async def evaluate(self, req: EvalRequest) -> EvalResult: ...


@dataclass(frozen=True)
class DimensionScore:
    """进入聚合器的单个维度。"""

    value: float
    confidence: float = 1.0
    provider: str = "unknown"
    degraded: bool = False


@dataclass(frozen=True)
class Usage:
    """一次外部调用的资源消耗。

    延迟、token、成本**必须**由适配器如实上报（估不出来就填 0 并注明），
    不允许为了"好看"而少报——成本模型要在 P1 前用 20 场实测标定（方案 §10）。
    """

    provider: str
    operation: str = "unknown"
    latency_ms: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    cached: bool = False

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@runtime_checkable
class Meter(Protocol):
    """可观测性端口：埋点、延迟观测、计数器。

    核心层只依赖这个 Protocol；具体实现可以是 Prometheus、OpenTelemetry、
    也可以是测试里的内存收集器。换实现 = 只改适配器。
    """

    def record_usage(self, usage: Usage) -> None: ...
    def observe_ms(self, name: str, ms: float) -> None: ...
    def incr(self, name: str, value: int = 1) -> None: ...


@runtime_checkable
class RepoPort(Protocol):
    def save_session(self, session_id: str, data: dict[str, Any]) -> None: ...
    def save_turn(self, session_id: str, turn: dict[str, Any]) -> None: ...
    def load_session(self, session_id: str) -> Optional[dict[str, Any]]: ...
    def load_turns(self, session_id: str) -> list[dict[str, Any]]: ...
    def append_event(self, session_id: str, ev: dict[str, Any]) -> None: ...


# ---------- 语音与形象 ----------


@dataclass(frozen=True)
class WordTiming:
    """一个词的时间边界（ms，相对音频起点）。用于驱动口型。"""

    word: str
    start_ms: int
    end_ms: int


@dataclass(frozen=True)
class VisemeEvent:
    """一个口型事件。

    `t_ms` 相对音频起点的**同一时钟**；`weight` ∈ [0,1] 是该口型的张开/融合强度。
    音频与形象必须共用这一个时间基准，否则一定音画不同步（方案 §6.5）。
    """

    t_ms: int
    viseme: str
    weight: float = 1.0


@dataclass(frozen=True)
class SpeechPlan:
    """一次合成的完整产物：音频 + 词时间轴 + 口型时间轴。"""

    audio: bytes
    words: tuple[WordTiming, ...] = ()
    visemes: tuple[VisemeEvent, ...] = ()
    sample_rate: int = 24000
    provider: str = "unknown"
    latency_ms: int = 0
    cost_usd: float = 0.0
    #: 词时间轴的来源。`native` = TTS 直接给出；`forced_alignment` = 对合成音频做
    # 强制对齐得到；`none` = 两者都没有（此时 `words` 必为空，口型不下发）。
    #: 2026-10 已证实：**自托管开源 TTS 权重没有一款原生输出词级时间戳**
    #: （见 `docs/开源TTS选型研究-2026-10.md`），因此生产路径恒为
    #: `forced_alignment`，实现在 `adapters/align_qwen3.py`。
    timing_source: str = "forced_alignment"


@runtime_checkable
class TTSPort(Protocol):
    """语音合成。**流式**：先出首块再出完整时间轴，避免为拿时间戳而牺牲首字延迟。"""

    name: str

    async def synth_stream(self, text: str, *, voice: str = "default") -> Any: ...

    async def synth(self, text: str, *, voice: str = "default") -> SpeechPlan: ...


@runtime_checkable
class ASRPort(Protocol):
    """语音识别。`partial` 允许为空串（还没识别出东西），但**不得**编造内容。"""

    name: str

    async def feed(self, chunk: bytes) -> Optional[str]: ...

    async def finalize(self) -> str: ...


# ---------- 云端视觉 ----------


@dataclass(frozen=True)
class VLMObservation:
    """一次云端视觉调用的结构化观察结果。

    `confidence` 低时上层应丢弃该观察而不是当证据用；`evidence_frames` 保留
    被采样的帧索引，便于事后复核（方案 §10：VLM 幻觉风险的兜底）。
    """

    summary: str
    confidence: float
    labels: tuple[str, ...] = ()
    evidence_frames: tuple[int, ...] = ()
    provider: str = "unknown"
    cost_usd: float = 0.0
    latency_ms: int = 0


@runtime_checkable
class VLMPort(Protocol):
    """云端多模态大模型。本地**不保留**任何视觉大模型（方案 §4）。"""

    name: str

    async def observe(
        self, frames: list[bytes], prompt: str, *, schema: Optional[dict] = None
    ) -> VLMObservation: ...


# ---------- 生理：压力调节 ----------


@dataclass(frozen=True)
class EnvQuality:
    """端侧环境自检（详设 §9）。用于判断"测不准是环境问题还是人"。

    `face_ratio` 是画面中脸部面积占比（过小说明离摄像头太远，ROI 里像素不够）；
    `flicker` 是光照闪烁强度（工频灯会造成周期性干扰，直接摧毁 rPPG）。
    """

    fps: float = 0.0
    brightness: float = 0.0
    face_ratio: float = 0.0
    flicker: float = 0.0


@dataclass(frozen=True)
class HRSample:
    """一个 rPPG 窗口的心率估计（详设 §9）。

    `bpm is None` 表示**弃权**——信号质量不足时正确行为是"不产出数值"，
    绝不是给一个猜的数值。低 SNR 出错误数值会对深肤色候选人系统性不利，
    因此这个 `None` 是公平性设计的一部分，不是容错细节。
    """

    t_ms: int
    bpm: Optional[float]
    snr: float = 0.0
    algo_spread: float = 0.0

    @property
    def rejected(self) -> bool:
        return self.bpm is None


@dataclass(frozen=True)
class HRBatch:
    """端侧批量上报（每 2 s 一条，详设 §9）。"""

    session_id: str
    samples: tuple[HRSample, ...] = ()
    algo_agreement: float = 0.0
    rejected_windows: int = 0
    env: Optional[EnvQuality] = None


@dataclass(frozen=True)
class EventAnchor:
    """一次应激事件的三个时间锚点（详设 §4.1）。

    `t_r`（恢复观测窗结束）不在这里——它由 `t_e + 45s` 推出，且可被下一题截断，
    由收集器按数据实际长度决定，不由调用方指定。
    """

    turn_index: int
    t_q_ms: int
    t_a_ms: Optional[int] = None
    t_e_ms: Optional[int] = None


@runtime_checkable
class StressRegulationPort(Protocol):
    """压力调节评估端口（详设 §9）。

    与详设的差异：原文写 `mark_event(e: InterviewEvent)`，但 `InterviewEvent`
    只有 `pre/peak/t50`，装不下 `t_q/t_a/t_e` 三个锚点，无法反推指标；故改为
    传入 `EventAnchor`，由收集器在数据到位后产出 `InterviewEvent`。
    """

    def push_batch(self, b: HRBatch) -> int: ...
    def mark_event(self, a: EventAnchor) -> None: ...
    def result(self) -> RegulationResult: ...


@dataclass(frozen=True)
class WindowQuality:
    """单个 rPPG 窗口的质量。snr<阈值即弃权（不产出数值）。"""

    bpm: Optional[float]
    snr: float
    algo_spread: float
    rejected: bool


@dataclass(frozen=True)
class InterviewEvent:
    """一次应激事件（一道题）的生理观测。"""

    turn_index: int
    pre_bpm: Optional[float]
    peak_bpm: Optional[float]
    t50_s: Optional[float]
    censored: bool = False
    valid: bool = True

    @property
    def delta(self) -> Optional[float]:
        if self.pre_bpm is None or self.peak_bpm is None:
            return None
        return self.peak_bpm - self.pre_bpm

    @property
    def ratio(self) -> Optional[float]:
        """归一化反应幅度 r_i = ΔHR_i / pre_i。

        自检 A4：绝对幅值跨个体/跨设备不可比，幅值类结论一律用它。
        """
        d = self.delta
        if d is None or not self.pre_bpm:
            return None
        return d / self.pre_bpm


@dataclass(frozen=True)
class RegulationResult:
    available: bool
    reason: Optional[str]
    baseline_bpm: Optional[float] = None
    n_events: int = 0
    n_valid: int = 0
    t50_median_s: Optional[float] = None
    habituation_slope: Optional[float] = None
    reactivity_ratio_mean: Optional[float] = None
    consistency_sigma: Optional[float] = None
    components: dict[str, Optional[float]] = field(default_factory=dict)
    score: Optional[float] = None
    reliability: float = 0.0
    weight_applied: float = 0.0
    redistributed_to: dict[str, float] = field(default_factory=dict)
