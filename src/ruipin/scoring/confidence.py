"""置信度：回答"这个分数有多可信"，而不是"分数有多高"。

设计要点
--------
1. 置信度只由**输入证据的充分度**决定，与分数高低**正交**。
   一个答得很差但录音清晰的回答，置信度是高的 —— 高分低置信 / 低分高置信都必须可能，
   否则置信度就退化成了分数的另一个名字。
2. 所有阈值都是**模块常量**且写明依据，禁止散落在代码里的魔法数。
3. `should_downgrade` 只回答"要不要在报告上打降级徽标"，不修改分数。
"""

from __future__ import annotations

# ---------- 长度证据 ----------

#: 低于此字符数视为"几乎没有内容"（约 5~8 个中文短句以下）
MIN_ANSWER_LEN = 20
#: 达到此字符数即视为"充分作答"，再长不再增加置信度（中文约 1 分钟口述量）
FULL_ANSWER_LEN = 160

# ---------- 模态证据 ----------

#: 有音频 = 能拿到停顿、语速、迟疑等超文本信号，是最重要的单一增量
W_AUDIO = 0.20
#: 有视频 = 能拿到表情/姿态，但语义贡献弱于音频
W_VIDEO = 0.10
#: 长度是最基础也最稳定的证据，给最大权重
W_LENGTH = 0.45
#: 独立证据源数量（文本/转写/关键词/音频/视频）
W_SIGNALS = 0.25

#: 达到此数量的独立信号即视为"证据源饱和"
N_SIGNALS_FULL = 4

#: provider 处于降级态时置信度折半（与 aggregate 里 degraded 的 0.5 折减同源）
DEGRADED_FACTOR = 0.5

# ---------- 降级阈值 ----------

#: ≥0.75：证据充分，与 aggregate 的 gate 上限对齐（方案 §5.6 gate(R) 的 R≥0.75）
CONF_GATE_HIGH = 0.75
#: ≥0.40：低于此值 aggregate 会把该维度权重归零（方案 §5.6 gate(R) 的下界）
CONF_GATE_LOW = 0.40
#: <0.50：证据不足一半，报告必须打"仅供参考"徽标（方案 §3.4 Level 2 的呈现口径）
CONF_DOWNGRADE = 0.50


def length_evidence(answer_len: int) -> float:
    """把作答长度归一化到 [0,1] 的证据强度（线性，两端截断）。

    只做截断不做衰减：超长回答不重复加分，也**不扣分**（啰嗦与否由 rubric 的结构分处理）。
    """
    n = max(0, int(answer_len))
    if n <= MIN_ANSWER_LEN:
        return 0.0
    if n >= FULL_ANSWER_LEN:
        return 1.0
    return (n - MIN_ANSWER_LEN) / (FULL_ANSWER_LEN - MIN_ANSWER_LEN)


def compute_confidence(
    answer_len: int,
    has_audio: bool = False,
    has_video: bool = False,
    n_signals: int = 1,
    provider_degraded: bool = False,
) -> float:
    """计算置信度，值域 [0,1]。

    组成（权重和为 1，各分量含义见常量处注释）：
    - 长度证据 0.45
    - 音频 0.20 / 视频 0.10
    - 独立证据源数量 0.25（饱和于 4 个）
    最后若 provider 降级，整体乘 0.5 —— 降级结果是"打了折的真实分数"，不是等权证据。

    单调性：对 answer_len / n_signals 单调不减，has_audio / has_video 取 True 不减，
    provider_degraded=True 严格不增。
    """
    parts = (
        W_LENGTH * length_evidence(answer_len),
        W_AUDIO * (1.0 if has_audio else 0.0),
        W_VIDEO * (1.0 if has_video else 0.0),
        W_SIGNALS * min(max(0, int(n_signals)) / N_SIGNALS_FULL, 1.0),
    )
    conf = sum(parts)
    if provider_degraded:
        conf *= DEGRADED_FACTOR
    return _clamp01(conf)


def should_downgrade(confidence: float) -> bool:
    """置信度是否低到需要在报告上标注"仅供参考"。

    阈值 0.50：低于一半证据量时，分数仍可展示但其解释力不足以支撑决策
    （方案 §3.4 Level 2 要求此类结果对用户可见地标注）。
    """
    return confidence < CONF_DOWNGRADE


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)


__all__ = [
    "MIN_ANSWER_LEN", "FULL_ANSWER_LEN",
    "W_AUDIO", "W_VIDEO", "W_LENGTH", "W_SIGNALS", "N_SIGNALS_FULL", "DEGRADED_FACTOR",
    "CONF_GATE_HIGH", "CONF_GATE_LOW", "CONF_DOWNGRADE",
    "length_evidence", "compute_confidence", "should_downgrade",
]
