"""单窗口质量门控：三算法投票 + SNR 阈值（详设 §3.3 / §3.8）。

纪律一：**弃权优于猜测**。算法分歧过大或 SNR 不足时，本窗口 `bpm=None`
（rejected=True），绝不输出一个"大概的数"。

窗口层不识人、不识题，只回答一件事：这一窗口的心率值能不能信。
"""

from __future__ import annotations

from typing import Optional

from ..ports import WindowQuality

# --- 投票阈值（BPM）--------------------------------------------------------
AGREE_TOL_BPM = 3.0  # 两两差 ≤3 → 一致
DISAGREE_TOL_BPM = 8.0  # 3–8 → 分歧（仍取中位数）；>8 → 弃权

# --- SNR 阈值 --------------------------------------------------------------
SNR_MIN = 0.35  # §3.8：SNR < 0.35 → 该窗口判无效

# --- SNR 分量归一化参考点 ---------------------------------------------------
# norm(v) = v / (v + ref)，即 v == ref 时归一化得 0.5（饱和映射，避免个别强峰一家独大）。
# SPEC_REF = 2.0 对应 §3.5 的"主峰需高出频带内均值 ≥3 dB"。
_SPEC_REF = 2.0
_HARM_REF = 1.0


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def _median(vals: list[float]) -> float:
    s = sorted(vals)
    n = len(s)
    mid = n // 2
    if n % 2:
        return float(s[mid])
    return (float(s[mid - 1]) + float(s[mid])) / 2.0


def _saturate(v: float, ref: float) -> float:
    """饱和归一化到 [0,1)：v == ref 时得 0.5。"""
    if v <= 0.0 or ref <= 0.0:
        return 0.0
    return v / (v + ref)


def vote(
    chrom: Optional[float],
    pos: Optional[float],
    ssr: Optional[float],
    *,
    snr: float = 0.0,
) -> WindowQuality:
    """CHROM / POS / SSR 三算法投票（§3.3）。

    规则：
    - 三者两两差 ≤3 BPM  → 取中位数，一致率高
    - 两两差 3–8 BPM     → 取中位数，标记 disagree（由 `algo_spread` 表达）
    - 任意两者差 >8 BPM  → **弃权**：`bpm=None, rejected=True`
    - 任一算法自身弃权（None）→ 按剩余算法处理；只剩一个算法时直接采信，
      但一致率降低（见 `agreement_score`）

    `snr` 为该窗口的 SNR（§3.8）；未给出时按 0 处理 → 直接弃权（纪律一）。

    一致/分歧不在返回结构体上单独打标，而是由 `algo_spread` 连续表达，
    供上层折算 `algo_agreement` 与诊断。
    """
    vals = [float(v) for v in (chrom, pos, ssr) if v is not None]
    if not vals:
        return WindowQuality(bpm=None, snr=snr, algo_spread=0.0, rejected=True)

    spread = max(vals) - min(vals) if len(vals) >= 2 else 0.0
    # >8 BPM → 弃权；否则（含 3–8 的分歧区间）取中位数
    bpm: Optional[float] = None if spread > DISAGREE_TOL_BPM else _median(vals)
    if is_rejected(bpm, snr, spread):
        # 弃权即不产出数值：rejected 与 bpm=None 永远绑定，不允许"带着数说不确定"
        return WindowQuality(bpm=None, snr=snr, algo_spread=spread, rejected=True)
    return WindowQuality(bpm=bpm, snr=snr, algo_spread=spread, rejected=False)


def compute_snr(
    peak_power: float,
    band_median_power: float,
    harmonic_power: float,
    continuity_ratio: float,
) -> float:
    """窗口 SNR ∈ [0,1]（§3.8）。

        SNR_spec   = 主峰功率 / 带内中位功率      # 频谱峰显著度
        SNR_harm   = 2·f0 处功率 / 带内中位功率   # 谐波存在性（心跳非纯正弦）
        continuity = 相邻估计变化 ≤8 BPM 的步数占比 ∈ [0,1]
        SNR        = 0.6·norm(SNR_spec) + 0.2·norm(SNR_harm) + 0.2·continuity

    两个功率比先经饱和归一化（v==ref → 0.5），再加权，避免原始功率比的量纲与
    离群值主导。SNR < 0.35 → 该窗口判无效。
    """
    if band_median_power <= 0.0:
        spec = harm = 0.0
    else:
        spec = _saturate(peak_power / band_median_power, _SPEC_REF)
        harm = _saturate(harmonic_power / band_median_power, _HARM_REF)
    return _clamp01(0.6 * spec + 0.2 * harm + 0.2 * _clamp01(continuity_ratio))


def is_rejected(bpm: Optional[float], snr: float, algo_spread: float) -> bool:
    """该窗口是否弃权。三条任一成立即弃权，无一是"给个中间值"。"""
    if bpm is None:  # 算法自身弃权或分歧过大
        return True
    if snr < SNR_MIN:  # 频谱质量不足
        return True
    if algo_spread > DISAGREE_TOL_BPM:  # 算法互不信
        return True
    return False


def agreement_score(spread: float, n_algos: int) -> float:
    """单窗口算法一致率 ∈ [0,1]，整场 `algo_agreement` 由它平均得到。

    - 3 算法且极差 ≤3 BPM → 1.0
    - 3 算法且极差 3–8 BPM → 0.55（分歧：采信但降权）
    - 极差 >8 BPM（已弃权）→ 0.0
    - 只剩 1 个算法 → 0.30（无人可对照，采信但一致率显著下降）
    """
    if n_algos <= 0:
        return 0.0
    if n_algos == 1:
        return 0.30
    if spread > DISAGREE_TOL_BPM:
        return 0.0
    if spread <= AGREE_TOL_BPM:
        return 1.0
    return 0.55
