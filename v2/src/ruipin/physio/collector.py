"""端侧窗口流 → 事件指标 → 压力调节结果（详设 §4.1–4.4 的实现）。

这一层补的是原先的**断链**：`window.py` 会算单窗口 BPM、`metrics.py` 会做跨事件
回归、`regulation.py` 会标定与门控，但"一批批 2 秒粒度的窗口"如何变成
"每个题目的 pre/peak/T50"没有任何代码负责。没有这一层，整条心率链路就停在
"能测到心率"而永远进不了评分。

职责边界
--------
* `push_batch()` —— 吃端侧批量上报（`HRBatch`），窗口可能**乱序到达**（弱网重传），
  这里按 `t_ms` 归并排序；重复窗口按 `t_ms` 去重（同一时刻只保留先到的那条）。
* `mark_event()` —— 吃题目锚点（`t_q/t_a/t_e`），据此切出 `pre_i` / `peak_i` / `λ_i`。
* `result()` —— 交给 `regulation.evaluate_physio` 做标定、可靠性门控与权重重分配。

三条纪律
--------
1. **弃权优于猜测**：`bpm is None` 的窗口不进任何统计；拟合点不足时该事件的
   `valid=False`，直接不参与跨事件聚合，绝不用 0 或基线值补上。
2. **基线期 SNR 门控是整场开关**（详设 §4.2）：静息期 SNR 中位数 < 0.4 时，
   整块生理维度关闭并给出原因——环境不支持就该说"不做这项"，不是硬算出个数。
3. **单个事件永不进入评分**：本模块只产出 `InterviewEvent`，真正的判定权在
   `metrics`/`regulation`（≥8 个有效事件才放行）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional

from ..ports import (
    PHYSIO_DIM,
    EnvQuality,
    EventAnchor,
    HRBatch,
    HRSample,
    InterviewEvent,
    RegulationResult,
)
from .metrics import RECOVERY_WINDOW_S, fit_recovery
from .regulation import (
    DEFAULT_WEIGHTS,
    W_MAX,
    PhysioCalibration,
    redistribute,
)

#: `pre_i` 取值窗：题目下发前 3 秒（详设 §4.3 `median(HR over [t_q − 3s, t_q])`）。
PRE_WINDOW_MS = 3_000

#: `peak_i` 取值窗：开始作答后 20 秒（详设 §4.3 `max(HR over [t_a, t_a + 20s])`）。
PEAK_WINDOW_MS = 20_000

#: 恢复观测窗长度（详设 §4.1：`t_r = t_e + 45s`，可被下一题截断）。
RECOVERY_WINDOW_MS = int(RECOVERY_WINDOW_S * 1000)

#: 静息基线期：开场引导后 60 秒，其中**最后 30 秒**用于取中位数
#: （前 30 秒丢弃，让 rPPG 信号收敛；见详设 §4.2）。
BASELINE_REST_MS = 60_000
BASELINE_TAIL_MS = 30_000

#: 基线期 SNR 中位数低于此值 → 整场关闭本模块（详设 §4.2）。
BASELINE_MIN_SNR = 0.40

#: `ΔHR < 2 BPM` 视为"未激发"，不计入恢复统计（详设 §4.3）。常量在 metrics 里已定义，
#: 这里不重复定义以免两处口径漂移。
MIN_FIT_POINTS_PER_EVENT = 4


@dataclass
class _SampleBuffer:
    """按 `t_ms` 归并、去重的窗口缓冲。

    端侧批量上报在弱网下可能**乱序**甚至重复（重传）。若不排序，`pre`/`peak`
    的时间窗切片会错位；若不去重，同一秒的 BPM 被计两次会拉偏中位数。
    """

    _by_ts: dict[int, HRSample] = field(default_factory=dict)

    def merge(self, samples: tuple[HRSample, ...]) -> int:
        added = 0
        for s in samples:
            if s.t_ms not in self._by_ts:
                self._by_ts[s.t_ms] = s
                added += 1
        return added

    def window(self, start_ms: int, end_ms: int) -> list[HRSample]:
        """闭区间 `[start_ms, end_ms]` 内、**按时间升序**的窗口。"""
        if end_ms < start_ms:
            return []
        return [s for t, s in sorted(self._by_ts.items()) if start_ms <= t <= end_ms]

    def all(self) -> list[HRSample]:
        """全部窗口，按时间升序。"""
        return [s for _t, s in sorted(self._by_ts.items())]

    def __len__(self) -> int:
        return len(self._by_ts)


def _median(values: list[float]) -> Optional[float]:
    """中位数；空列表返回 None（**不是 0**，0 会被当成一个真实的心率）。"""
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _valid_bpms(samples: list[HRSample]) -> list[float]:
    """只取未被弃权的窗口——`bpm is None` 是"不知道"，不能当 0 参与统计。"""
    return [float(s.bpm) for s in samples if s.bpm is not None]


class StressRegulationCollector:
    """`StressRegulationPort` 的实现（详设 §9）。

    典型用法::

        c = StressRegulationCollector(session_id=..., authorized=True)
        c.push_batch(batch)                      # 端侧每 2s 一批
        c.baseline_from_rest(0, BASELINE_REST_MS)  # 静息期结束时刻
        c.mark_event(EventAnchor(0, t_q, t_a, t_e))
        ...
        result = c.result()                      # ≥8 有效事件才 available

    Args:
        session_id: 会话标识（用于拒绝串场的批次）。
        authorized: 是否取得生理分析的**单独授权**（PIPL 第 28 条）。False 时
            `result()` 直接产出"未授权"的不可用结果，权重全额归还其他维度。
        weights: 维度权重表，缺省含生理维度 8%。
        calibration: 分位数标定表，缺省用占位表（**上线前必须替换**）。
        expected_events: 期望的事件数（用于 `n_expected`）。缺省用实际锚点数。
    """

    def __init__(
        self,
        session_id: str,
        *,
        authorized: bool = True,
        weights: Optional[Mapping[str, float]] = None,
        calibration: Optional[PhysioCalibration] = None,
        expected_events: Optional[int] = None,
    ) -> None:
        self.session_id = session_id
        self.authorized = bool(authorized)
        self._weights: dict[str, float] = (
            dict(weights) if weights is not None else dict(DEFAULT_WEIGHTS)
        )
        self._calibration = calibration
        self._expected_events = expected_events

        self._buf = _SampleBuffer()
        self._anchors: list[EventAnchor] = []
        #: 事件指标缓存；锚点或窗口一变就失效（见 `_build_events`）。
        self._events_cache: Optional[list[InterviewEvent]] = None
        self._agreements: list[float] = []
        self._env: Optional[EnvQuality] = None
        self._batches = 0
        self._declared_rejected = 0
        self._baseline_bpm: Optional[float] = None
        self._baseline_gate_reason: Optional[str] = None
        self._ibis: list[float] = []

    # ---------- StressRegulationPort ----------

    def push_batch(self, b: HRBatch) -> int:
        """接收一批端侧窗口。返回**新增**（去重后）的窗口数。

        串场的批次（`session_id` 不一致且两边都非空）直接拒绝并返回 0——
        静默混入会把另一个人的心率算进本场报告，是最严重的数据污染。
        """
        if b.session_id and self.session_id and b.session_id != self.session_id:
            return 0
        self._batches += 1
        self._declared_rejected += max(0, int(b.rejected_windows))
        if 0.0 <= b.algo_agreement <= 1.0:
            self._agreements.append(float(b.algo_agreement))
        if b.env is not None:
            self._env = b.env
        added = self._buf.merge(b.samples)
        if added:
            self._events_cache = None
        return added

    def mark_event(self, a: EventAnchor) -> None:
        """登记一个事件锚点。

        **不在此刻切指标**：恢复观测窗的上界是 `min(t_e + 45s, 下一题 t_q)`，
        而登记第 i 题时下一题的 `t_q` 往往还没到（下一题是在候选人答完之后才下发的）。
        若此刻就固化，恢复窗永远截不断——第 i+1 题期间的心率会被算进第 i 题的回落里，
        让 `T50` 系统性偏大（看起来"恢复得更慢"）。因此事件在读取时才按**锚点全集**统一构建。
        """
        self._anchors.append(a)
        self._events_cache = None

    def result(self) -> RegulationResult:
        """产出压力调节结果。**不可用时 `score=None` 且权重全额归还。**"""
        from .regulation import evaluate_physio  # 局部导入避免循环引用

        events = self._build_events()
        # 判定顺序 = 根因顺序：授权 → 环境门控 → 有没有数据 → 有没有基线。
        # 环境门控排在"事件数"之前，是因为它是**根因**：环境不支持时后面那些
        # 指标缺失都只是后果，报"事件不足"会把候选人引向错误的自救方向
        #（他会以为多答几题就好，实际上是灯太暗）。
        if not self.authorized:
            return self._unavailable("未获得生理信号分析的单独授权")
        if self._baseline_gate_reason is not None:
            return self._unavailable(self._baseline_gate_reason)
        if not events:
            return self._unavailable("没有任何已完成的事件锚点，本项不计入")
        if self._baseline_bpm is None:
            return self._unavailable("静息基线心率缺失，本项不计入")

        n_expected = (
            float(self._expected_events)
            if self._expected_events is not None
            else float(len(self._anchors))
        )
        return evaluate_physio(
            events=events,
            baseline_bpm=self._baseline_bpm,
            n_expected=n_expected,
            median_snr=self.median_snr(),
            reject_ratio=self.reject_ratio(),
            algo_agreement=self.algo_agreement(),
            authorized=True,
            ibis=self._ibis or None,
            weights=self._weights,
            calibration=self._calibration,
        )

    # ---------- 基线 ----------

    def set_baseline(self, bpm: Optional[float]) -> None:
        """直接设定静息基线（由调用方在静息期结束后算出）。"""
        self._baseline_bpm = None if bpm is None else float(bpm)

    def baseline_from_rest(
        self,
        rest_start_ms: int = 0,
        rest_end_ms: int = BASELINE_REST_MS,
    ) -> Optional[float]:
        """从静息期窗口算基线，并做**整场 SNR 门控**（详设 §4.2）。

        `B = median(HR over 最后 30s)`（丢弃前 30 s 让信号收敛）。
        SNR 中位数 < 0.40 → 记下关闭原因，`result()` 会整场不计入。
        """
        tail_start = max(rest_start_ms, rest_end_ms - BASELINE_TAIL_MS)
        tail = self._buf.window(tail_start, rest_end_ms)

        snrs = [float(s.snr) for s in tail]
        median_snr = _median(snrs)
        if median_snr is None or median_snr < BASELINE_MIN_SNR:
            shown = "无数据" if median_snr is None else f"{median_snr:.2f}"
            self._baseline_gate_reason = (
                f"静息期 SNR 中位数 {shown} < {BASELINE_MIN_SNR:.2f}："
                "环境不支持本次生理评估，本项整场不计入"
            )
            self._baseline_bpm = None
            return None

        bpms = _valid_bpms(tail)
        baseline = _median(bpms)
        if baseline is None:
            self._baseline_gate_reason = (
                "静息期所有窗口均被弃权，无法建立基线，本项整场不计入"
            )
            self._baseline_bpm = None
            return None

        self._baseline_gate_reason = None
        self._baseline_bpm = baseline
        return baseline

    def set_ibis(self, ibis: list[float]) -> None:
        """提供逐拍间期用于心律不齐启发式检测（**非诊断**，仅决定是否关闭本项）。"""
        self._ibis = [float(x) for x in ibis]

    # ---------- 质量聚合 ----------

    def median_snr(self) -> float:
        """全窗口 SNR 中位数。无窗口时返回 0（可靠性 R 会因此判为不可信，是正确的）。"""
        value = _median([float(s.snr) for s in self._buf.all()])
        return 0.0 if value is None else value

    def reject_ratio(self) -> float:
        """弃权窗口占比 ∈ [0,1]。无窗口时返回 1.0（全弃权 = 完全不可信）。"""
        samples = self._buf.all()
        if not samples:
            return 1.0
        rejected = sum(1 for s in samples if s.bpm is None)
        return rejected / len(samples)

    def algo_agreement(self) -> float:
        """三算法一致率（端侧自报批次的均值）。无上报时 0。"""
        if not self._agreements:
            return 0.0
        return sum(self._agreements) / len(self._agreements)

    # ---------- 只读视图 ----------

    @property
    def n_samples(self) -> int:
        return len(self._buf)

    @property
    def n_batches(self) -> int:
        return self._batches

    @property
    def n_anchors(self) -> int:
        """已登记的事件锚点数（含尚未补齐 t_a/t_e 的）。"""
        return len(self._anchors)

    @property
    def n_events(self) -> int:
        """已切出指标的事件数。"""
        return len(self._build_events())

    def samples(self) -> tuple[HRSample, ...]:
        """全部窗口（按 t_ms 升序）—— 口径自检与离线复核用。"""
        return tuple(self._buf.all())

    @property
    def baseline_bpm(self) -> Optional[float]:
        return self._baseline_bpm

    @property
    def baseline_gate_reason(self) -> Optional[str]:
        return self._baseline_gate_reason

    @property
    def env(self) -> Optional[EnvQuality]:
        return self._env

    @property
    def events(self) -> tuple[InterviewEvent, ...]:
        return tuple(self._build_events())

    # ---------- 内部 ----------

    def _build_events(self) -> list[InterviewEvent]:
        """按锚点全集统一构建事件指标（带缓存）。

        必须看全集才能算"恢复窗被下一题截断"——逐条登记时下一题还没到（见 `mark_event`）。
        """
        if self._events_cache is None:
            self._events_cache = [
                self._build_event(a)
                for a in self._anchors
                if a.t_a_ms is not None and a.t_e_ms is not None
            ]
        return self._events_cache

    def _build_event(self, a: EventAnchor) -> InterviewEvent:
        """按详设 §4.3 切出单个事件的指标。

        `pre` = [t_q−3s, t_q] 的中位数；`peak` = [t_a, t_a+20s] 的最大值；
        `λ/T50` 由 `[peak, t_r]` 做对数线性拟合，`t_r = min(t_e+45s, 下一题 t_q)`。
        任一步拿不到足够数据 → `valid=False`（弃权），**不用替代值硬凑**。
        """
        pre_values = [
            float(s.bpm)
            for s in self._buf.window(a.t_q_ms - PRE_WINDOW_MS, a.t_q_ms)
            if s.bpm is not None
        ]
        pre = _median(pre_values)

        t_a = a.t_a_ms if a.t_a_ms is not None else a.t_q_ms
        peak_values = _valid_bpms(self._buf.window(t_a, t_a + PEAK_WINDOW_MS))
        peak = max(peak_values) if peak_values else None

        if pre is None or peak is None:
            return InterviewEvent(
                turn_index=a.turn_index, pre_bpm=pre, peak_bpm=peak,
                t50_s=None, censored=False, valid=False,
            )

        delta = peak - pre
        if delta < 2.0:
            # "未激发"：不是数据错误，但也不计入恢复统计（详设 §4.3）
            return InterviewEvent(
                turn_index=a.turn_index, pre_bpm=pre, peak_bpm=peak,
                t50_s=None, censored=False, valid=False,
            )

        # 找到峰值所在窗口，自峰值起算恢复窗
        t_peak = self._peak_time(t_a, t_a + PEAK_WINDOW_MS, peak)
        t_r = min(a.t_e_ms + RECOVERY_WINDOW_MS, self._next_question_ts(a.turn_index))
        recovery = self._buf.window(t_peak, t_r)

        times = [(s.t_ms - t_peak) / 1000.0 for s in recovery]
        hrs = [float(s.bpm) if s.bpm is not None else float("nan") for s in recovery]
        usable = [(t, h) for t, h in zip(times, hrs) if h == h]  # 丢掉 NaN（弃权窗）
        if len(usable) < MIN_FIT_POINTS_PER_EVENT:
            return InterviewEvent(
                turn_index=a.turn_index, pre_bpm=pre, peak_bpm=peak,
                t50_s=None, censored=False, valid=False,
            )

        fit = fit_recovery([t for t, _ in usable], [h for _, h in usable], pre)
        if fit.lam is None or fit.t50_s is None:
            return InterviewEvent(
                turn_index=a.turn_index, pre_bpm=pre, peak_bpm=peak,
                t50_s=None, censored=False, valid=False,
            )
        return InterviewEvent(
            turn_index=a.turn_index, pre_bpm=pre, peak_bpm=peak,
            t50_s=fit.t50_s, censored=fit.censored, valid=True,
        )

    def _peak_time(self, start_ms: int, end_ms: int, peak: float) -> int:
        """峰值所在窗口的时间戳（多个并列时取最早的一个）。"""
        for s in self._buf.window(start_ms, end_ms):
            if s.bpm is not None and float(s.bpm) == peak:
                return s.t_ms
        return start_ms

    def _next_question_ts(self, turn_index: int) -> int:
        """下一题的 `t_q`；没有下一题时用一个大值（恢复窗不被截断）。"""
        later = [a.t_q_ms for a in self._anchors if a.turn_index > turn_index]
        return min(later) if later else (1 << 62)

    def _unavailable(self, reason: str) -> RegulationResult:
        """不计入：分数 None、权重 0、并把生理维度的权重上限全额归还其他维度。"""
        events = self._build_events()
        return RegulationResult(
            available=False,
            reason=reason,
            baseline_bpm=self._baseline_bpm,
            n_events=len(events),
            n_valid=sum(1 for e in events if e.valid),
            components={},
            score=None,
            reliability=0.0,
            weight_applied=0.0,
            redistributed_to=redistribute(self._weights, PHYSIO_DIM, W_MAX),
        )


__all__ = [
    "BASELINE_MIN_SNR",
    "BASELINE_REST_MS",
    "BASELINE_TAIL_MS",
    "PEAK_WINDOW_MS",
    "PRE_WINDOW_MS",
    "RECOVERY_WINDOW_MS",
    "StressRegulationCollector",
]
