"""规则评分器 `RuleEvaluator`（provider="rubric"）。

定位：**降级阶梯 Level 2 的真实评分器**，不是兜底。
- 它是降级项（方案 §3.4 L2），但输出的是**真实计算的分数**：关键词覆盖、作答长度、
  结构完整度，都是可复算、可解释的信号。
- 与"崩溃返回 60 分"的本质区别：本模块**没有失败路径**。它不依赖任何外部服务，
  因此不存在"适配器失败后返回合成值"的可能；真正无输入可评时它抛 `Unavailable`
  （空作答 = 没有证据 = 不产分数，与方案 §5.6 红线一致）。

关于六维中的弱证据维度
----------------------
技术题问不出团队协作，这是事实。做法是**先验收缩（shrinkage）**：
    dim = evidence · raw + (1 - evidence) · NEUTRAL_PRIOR
其中 `NEUTRAL_PRIOR = 50` 是**显式声明的先验**，不是崩溃兜底 —— 它是常量、写在代码里、
被 `confidence` 如实反映、并在聚合时被门控（证据不足 → 权重归零 → 不计入总分）。
"填 50 但权重为 0"与"填 60 且权重照常"是两件完全不同的事，后者才是不可接受的。

确定性
------
纯函数、无随机、无时间戳参与评分、无字典遍历顺序依赖 → 同样输入必然同样输出，
可直接做 Golden Master 测试。
"""

from __future__ import annotations

import re

from ..domain.errors import Unavailable
from ..ports import DIMENSIONS, DimensionScore, EvalRequest, EvalResult
from .aggregate import DEFAULT_WEIGHTS, aggregate
from .confidence import compute_confidence, length_evidence, should_downgrade

PROVIDER = "rubric"
NAME = "rubric"

#: 弱证据维度的先验收缩目标（见模块文档说明：显式先验，非崩溃兜底）
NEUTRAL_PRIOR = 50.0

#: 归一化长度（0~1，见 confidence.length_evidence）→ 长度分的锚点，分段线性插值
LENGTH_ANCHORS: tuple[tuple[float, float], ...] = (
    (0.00, 25.0),   # 极短
    (0.15, 45.0),   # 极短 → 短
    (0.45, 72.0),   # 短 → 中
    (0.85, 90.0),   # 中 → 长
    (1.00, 95.0),   # 长（饱和，超长不额外加分也不扣分）
)

#: 长度档位（上界, 名称）—— 与 LENGTH_ANCHORS 同一套标尺，仅用于人读反馈。
#: 最后一项取 1.0 是刻意的：归一化长度恰好为 1.0 时落进兜底分支，即"充分"。
LENGTH_BANDS: tuple[tuple[float, str], ...] = (
    (0.15, "极短"),
    (0.45, "偏短"),
    (0.85, "中等"),
    (1.00, "充分"),
)

# ---------- 结构信号词 ----------

_CONNECTIVES = (
    "首先", "其次", "然后", "最后", "第一", "第二", "第三", "一方面", "另一方面",
    "因此", "所以", "因为", "由于", "比如", "例如", "接下来", "综上", "总之",
)
_CLOSERS = ("总之", "综上", "总的来说", "最后", "因此", "所以")
_STAR = ("背景", "情境", "任务", "目标", "行动", "结果", "收获", "反思")
_SENT_SPLIT = re.compile(r"[。！？；!?;]")
_DIGIT = re.compile(r"\d")

# 结构分构成（和为 100）
_W_CONNECTIVE = 35.0
_W_DIGIT = 20.0
_W_MULTI_SENT = 25.0
_W_CLOSER = 20.0

# ---------- 题型档案 ----------

#: 每个题型的 (关键词覆盖, 长度, 结构) 三维混合权重，按维度不同；每个元组和为 1.0
_DIM_MIX: dict[str, dict[str, tuple[float, float, float]]] = {
    "technical": {
        "technical": (0.70, 0.15, 0.15),
        "communication": (0.15, 0.50, 0.35),
        "completeness": (0.40, 0.35, 0.25),
        "problem_solving": (0.55, 0.15, 0.30),
        "teamwork": (0.20, 0.40, 0.40),
        "leadership": (0.20, 0.40, 0.40),
    },
    "project": {
        "technical": (0.45, 0.20, 0.35),
        "communication": (0.15, 0.45, 0.40),
        "completeness": (0.35, 0.40, 0.25),
        "problem_solving": (0.50, 0.15, 0.35),
        "teamwork": (0.30, 0.35, 0.35),
        "leadership": (0.35, 0.30, 0.35),
    },
    "behavioral": {
        "technical": (0.25, 0.30, 0.45),
        "communication": (0.15, 0.45, 0.40),
        "completeness": (0.25, 0.40, 0.35),
        "problem_solving": (0.35, 0.25, 0.40),
        "teamwork": (0.45, 0.25, 0.30),
        "leadership": (0.45, 0.25, 0.30),
    },
    "self_intro": {
        "technical": (0.30, 0.30, 0.40),
        "communication": (0.15, 0.50, 0.35),
        "completeness": (0.25, 0.45, 0.30),
        "problem_solving": (0.30, 0.25, 0.45),
        "teamwork": (0.30, 0.35, 0.35),
        "leadership": (0.35, 0.30, 0.35),
    },
}

#: 证据强度 → 同时用于 ①先验收缩 ②DimensionScore.confidence（进而决定聚合权重）
_DIM_EVIDENCE: dict[str, dict[str, float]] = {
    "technical": {
        "technical": 0.90, "communication": 0.55, "completeness": 0.70,
        "problem_solving": 0.75, "teamwork": 0.30, "leadership": 0.25,
    },
    "project": {
        "technical": 0.70, "communication": 0.60, "completeness": 0.75,
        "problem_solving": 0.80, "teamwork": 0.65, "leadership": 0.60,
    },
    "behavioral": {
        "technical": 0.35, "communication": 0.75, "completeness": 0.60,
        "problem_solving": 0.60, "teamwork": 0.85, "leadership": 0.80,
    },
    "self_intro": {
        "technical": 0.40, "communication": 0.80, "completeness": 0.70,
        "problem_solving": 0.45, "teamwork": 0.55, "leadership": 0.55,
    },
}

DEFAULT_QUESTION_TYPE = "technical"

#: 关键词为空时的中性覆盖率（不算命中也不算未命中）
NEUTRAL_COVERAGE = 0.5


class RuleEvaluator:
    """规则评分器。实现 `ports.Evaluator` Protocol。

    无外部依赖 → 永不返回合成分数；无作答内容时抛 `Unavailable`。
    """

    name = NAME

    async def evaluate(self, req: EvalRequest) -> EvalResult:
        qtype = req.question_type if req.question_type in _DIM_MIX else DEFAULT_QUESTION_TYPE
        answer = (req.answer or "").strip()
        transcript = (req.transcript or "").strip()

        if not answer and not transcript:
            # 没有任何可评估内容：诚实做法是"未评估"，而不是给一组中间分
            raise Unavailable(PROVIDER, "无作答内容，无法评分（未生成任何分数）")

        text = answer or transcript
        n = len(text)

        # 空白关键词既不算命中也不算未命中：直接排除，不进分母
        kws = tuple(k.strip() for k in req.keywords if k and k.strip())
        kw_total = len(kws)
        kw_hit, kw_missed = _match_keywords(text, kws)
        coverage = (kw_hit / kw_total) if kw_total else NEUTRAL_COVERAGE

        len_norm = length_evidence(n)
        len_score = _length_score(len_norm)
        struct_score = _structure_score(text)

        mix = _DIM_MIX[qtype]
        evidence = _DIM_EVIDENCE[qtype]

        dims: dict[str, float] = {}
        dim_objs: dict[str, DimensionScore] = {}
        for d in DIMENSIONS:
            w_kw, w_len, w_struct = mix[d]
            raw = w_kw * (coverage * 100.0) + w_len * len_score + w_struct * struct_score
            e = evidence[d]
            dims[d] = _clamp100(e * raw + (1.0 - e) * NEUTRAL_PRIOR)
            dim_objs[d] = DimensionScore(value=dims[d], confidence=e, provider=PROVIDER)

        agg = aggregate(dim_objs, DEFAULT_WEIGHTS)

        # 证据源计数：作答文本 1 + 转写/关键词/音频/视频
        n_signals = 1 + bool(transcript) + bool(req.keywords) + bool(req.has_audio) + bool(req.has_video)
        conf = compute_confidence(n, req.has_audio, req.has_video, n_signals, False)

        return EvalResult(
            dims=dims,
            score=_clamp100(agg.score),
            feedback=_feedback(
            qtype, kw_hit, kw_total, kw_missed, coverage, _band(len_norm), conf, agg.notes
        ),
            provider=PROVIDER,
            confidence=conf,
            latency_ms=0,
            cost_usd=0.0,
            degraded=False,
            degrade_reason=None,
        )


# ---------- 信号计算（纯函数，可单测） ----------


def _match_keywords(text: str, keywords: tuple[str, ...]) -> tuple[int, list[str]]:
    """关键词大小写不敏感子串匹配。调用方需先过滤空白关键词（否则会污染分母）。"""
    hay = (text or "").lower()
    hit = 0
    missed: list[str] = []
    for kw in keywords:
        k = kw.strip().lower()
        if k in hay:
            hit += 1
        else:
            missed.append(kw)
    return hit, missed


def _length_score(len_norm: float) -> float:
    """归一化长度 → 0-100 的长度分（分段线性，单调不减，两端截断）。"""
    x = min(max(len_norm, 0.0), 1.0)
    pts = LENGTH_ANCHORS
    if x <= pts[0][0]:
        return pts[0][1]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return pts[-1][1]


def _band(len_norm: float) -> str:
    x = min(max(len_norm, 0.0), 1.0)
    for upper, name in LENGTH_BANDS:
        if x < upper:
            return name
    return LENGTH_BANDS[-1][1]


def _structure_score(text: str) -> float:
    """结构完整度 0-100：连接词 / 量化 / 多句展开 / 收尾。全部可复算。"""
    if not text:
        return 0.0
    s = 0.0
    if any(c in text for c in _CONNECTIVES):
        s += _W_CONNECTIVE
    if _DIGIT.search(text):
        s += _W_DIGIT
    if len([p for p in _SENT_SPLIT.split(text) if p.strip()]) >= 3:
        s += _W_MULTI_SENT
    if any(c in text for c in _CLOSERS):
        s += _W_CLOSER
    return s


def _feedback(
    qtype: str,
    kw_hit: int,
    kw_total: int,
    kw_missed: list[str],
    coverage: float,
    band: str,
    conf: float,
    notes: list[str],
) -> str:
    if kw_total:
        kw_part = f"关键词命中 {kw_hit}/{kw_total}（覆盖率 {coverage:.0%}）"
        if kw_missed:
            kw_part += "；未命中：" + "、".join(kw_missed[:3])
    else:
        kw_part = "本题未配置关键词，覆盖率按中性 50% 处理"
    head = f"[规则评分/{qtype}] {kw_part}；作答长度档次：{band}；置信度 {conf:.2f}"
    tail = "；证据不足，本轮结果仅供参考" if should_downgrade(conf) else ""
    return head + tail + "。权重归因：" + "；".join(notes[:2])


def _clamp100(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 100.0:
        return 100.0
    return float(x)


__all__ = ["RuleEvaluator", "PROVIDER", "NEUTRAL_PRIOR", "DEFAULT_QUESTION_TYPE"]
