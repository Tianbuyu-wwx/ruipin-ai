// quality.ts — 窗口质量门控（SNR / 三算法投票 / 弃权 / AE 漂移检测）
//
// 权威规范：docs/模块详设-心率与压力调节评估.md
//   §3.3 多算法投票    §3.5 峰值/连续性    §3.8 SNR 明确定义    §7 光照与 AE 漂移
//
// 第一纪律（文档 §3.3 与团队负责人反复强调）：**弃权优于猜测**。
// 测不准 = 输出"无信号"（bpm = null），绝不给一个错的值。

import { PPG, bandPowerAt, median } from './signal.ts';

/** 质量门控常量。阈值均标注来源。 */
export const QUALITY = {
  // ---- §3.8 SNR 定义 ----
  /** SNR 组合权重：0.6·norm(SNR_spec)。文档 §3.8。 */
  SNR_WEIGHT_SPEC: 0.6,
  /** SNR 组合权重：0.2·norm(SNR_harm)。文档 §3.8。 */
  SNR_WEIGHT_HARM: 0.2,
  /** SNR 组合权重：0.2·continuity。文档 §3.8。 */
  SNR_WEIGHT_CONT: 0.2,
  /** 弃权阈值：SNR < 0.35 → 该窗口判无效。文档 §3.8。 */
  SNR_REJECT_THRESHOLD: 0.35,
  /**
   * norm(SNR_spec) 的对数满量程参考。
   * 来源：**本实现补充**——文档 §3.8 给出了 SNR 公式但未定义 norm() 的具体映射。
   *      此处采用对数映射（峰值-中位数比跨越数个数量级，线性映射会立即饱和而失去分辨力）：
   *        normSpec = clamp01( (log10(spec) − log10(FLOOR)) / (log10(CEIL) − log10(FLOOR)) )
   *      即 spec = 3 时归零，spec ≥ 1e5 时饱和为 1。
   *      这两个常数**必须**在真实数据 POC 中重新标定（见 README「尚未验证」）。
   */
  SNR_SPEC_LOG_FLOOR_RATIO: 3,
  SNR_SPEC_LOG_CEIL_RATIO: 100000,
  /**
   * norm(SNR_harm) 的线性满量程参考。文档 §3.8 的 SNR_harm 中，比值 ≈1 表示
   * "该处只有噪声底、无谐波"；比值越大表示谐波越显著。
   * 来源：**本实现补充**，同样需 POC 重标定。
   */
  SNR_HARM_FLOOR_RATIO: 1,
  SNR_HARM_CEIL_RATIO: 10,
  /** 谐波不可评估时（2·f0 超出奈奎斯特）normHarm 取中性值 0.5。本实现补充。 */
  SNR_HARM_NEUTRAL: 0.5,

  // ---- §3.3 三算法投票 ----
  /** 两两差 ≤3 BPM → 取中位数，置信度高。文档 §3.3。 */
  VOTE_TIGHT_BPM: 3,
  /** 两两差 3–8 BPM → 取中位数，标记 disagree。文档 §3.3。 */
  VOTE_LOOSE_BPM: 8,

  // ---- §3.5 / §3.8 时间连续性 ----
  /** 相邻估计变化上限 8 BPM。文档 §3.5、§3.8。 */
  CONTINUITY_MAX_DELTA_BPM: 8,

  // ---- §7 AE 漂移 ----
  /** 亮度帧间突变阈值：>15% 视为光照突变。文档 §7。 */
  AE_BRIGHTNESS_STEP_RATIO: 0.15,
  /**
   * 亮度慢漂移阈值（窗口前后半均值相对变化）。
   * 来源：**本实现补充**——文档 §7 要求"软件侧检测 AE 漂移"，但未给阈值；
   *      该值必须在真实摄像头 POC 中重标定（见 README「尚未验证」）。
   */
  AE_BRIGHTNESS_SLOW_RATIO: 0.08,
  /** 饱和像素占比上限：>30% ROI → 窗口无效。文档 §7。 */
  SATURATION_MAX_RATIO: 0.3,
} as const;

function clamp01(x: number): number {
  if (!Number.isFinite(x)) return 0;
  if (x < 0) return 0;
  if (x > 1) return 1;
  return x;
}

// ---------------------------------------------------------------------------
// §3.8 SNR_spec / SNR_harm
//   SNR_spec = P(f0 ± 0.15Hz) / median(P(带内其他频率))
//   SNR_harm = P(2·f0 ± 0.15Hz) / median(P(带内))
// ---------------------------------------------------------------------------

/** P(f0 ± halfWidth) —— 该窄带内的平均功率。 */
function peakBandPower(freqs: Float64Array, psd: Float64Array, f0: number): number {
  return bandPowerAt(freqs, psd, f0, PPG.SNR_BAND_HZ);
}

/** 带内（0.7–4.0 Hz）PSD 的中位数。 */
function inBandMedian(freqs: Float64Array, psd: Float64Array): number {
  const vals: number[] = [];
  for (let k = 0; k < freqs.length; k++) {
    if (freqs[k] >= PPG.BAND_LOW_HZ && freqs[k] <= PPG.BAND_HIGH_HZ) vals.push(psd[k]);
  }
  const m = median(vals);
  return Number.isFinite(m) ? m : 0;
}

/**
 * SNR_spec 原始比值。分母为"带内其他频率"（排除 f0±0.15Hz 窄带）的中位数，文档 §3.8。
 */
export function snrSpecRatio(freqs: Float64Array, psd: Float64Array, f0Hz: number): number {
  const num = peakBandPower(freqs, psd, f0Hz);
  const others: number[] = [];
  for (let k = 0; k < freqs.length; k++) {
    const f = freqs[k];
    if (f >= PPG.BAND_LOW_HZ && f <= PPG.BAND_HIGH_HZ && Math.abs(f - f0Hz) > PPG.SNR_BAND_HZ) {
      others.push(psd[k]);
    }
  }
  const den = median(others);
  if (!(den > 0)) return num > 0 ? num / 1e-15 : 0;
  return num / den;
}

/**
 * SNR_harm 原始比值。若 2·f0 超出奈奎斯特频率则返回 null（不可评估）。
 */
export function snrHarmRatio(freqs: Float64Array, psd: Float64Array, f0Hz: number): number | null {
  const f2 = 2 * f0Hz;
  const nyq = freqs[freqs.length - 1];
  if (f2 > nyq) return null;
  const num = peakBandPower(freqs, psd, f2);
  const den = inBandMedian(freqs, psd);
  if (!(den > 0)) return num > 0 ? num / 1e-15 : 0;
  return num / den;
}

export interface NormalizedSnr {
  snr: number;
  normSpec: number;
  normHarm: number;
}

/** 归一化并加权组合 SNR（文档 §3.8：0.6·norm(SNR_spec)+0.2·norm(SNR_harm)+0.2·continuity）。 */
export function combineSnr(snrSpecRaw: number, snrHarmRaw: number | null, continuity: number): NormalizedSnr {
  const specFloor = Math.log10(QUALITY.SNR_SPEC_LOG_FLOOR_RATIO);
  const specCeil = Math.log10(QUALITY.SNR_SPEC_LOG_CEIL_RATIO);
  const spec = snrSpecRaw > 0 ? Math.log10(snrSpecRaw) : -Infinity;
  const normSpec = clamp01((spec - specFloor) / (specCeil - specFloor));
  const normHarm =
    snrHarmRaw === null
      ? QUALITY.SNR_HARM_NEUTRAL
      : clamp01(
          (snrHarmRaw - QUALITY.SNR_HARM_FLOOR_RATIO) /
            (QUALITY.SNR_HARM_CEIL_RATIO - QUALITY.SNR_HARM_FLOOR_RATIO),
        );
  const cont = clamp01(continuity);
  const snr =
    QUALITY.SNR_WEIGHT_SPEC * normSpec + QUALITY.SNR_WEIGHT_HARM * normHarm + QUALITY.SNR_WEIGHT_CONT * cont;
  return { snr, normSpec, normHarm };
}

/**
 * 时间连续性：满足 |HR_t − HR_{t−1}| ≤ 8 BPM 的步数占比（文档 §3.8）。
 * 输入为按时间排列的估计序列（可为 null）；仅统计两端均非 null 的相邻对。
 * 有效相邻对 <1 时返回 1（中性，无历史可判）。
 */
export function computeContinuity(
  history: (number | null)[],
  maxDelta: number = QUALITY.CONTINUITY_MAX_DELTA_BPM,
): number {
  let total = 0;
  let good = 0;
  for (let i = 1; i < history.length; i++) {
    const a = history[i - 1];
    const b = history[i];
    if (a === null || b === null || !Number.isFinite(a) || !Number.isFinite(b)) continue;
    total++;
    if (Math.abs(b - a) <= maxDelta) good++;
  }
  return total === 0 ? 1 : good / total;
}

// ---------------------------------------------------------------------------
// §3.3 三算法投票
// ---------------------------------------------------------------------------

export type VoteDecision = 'ok' | 'disagree' | 'abstain';

export interface VoteResult {
  /** 投票后的心率；弃权时为 null。 */
  bpm: number | null;
  /** 三算法极差（max − min，BPM）。 */
  spread: number;
  /** 一致率 [0,1]：两两差 ≤ VOTE_TIGHT_BPM 的对数占比。 */
  agreement: number;
  decision: VoteDecision;
  reason: string | null;
}

/**
 * 三算法投票（文档 §3.3）：
 *   两两差 ≤3 BPM → 取中位数（高置信）
 *   两两差 3–8 BPM → 取中位数（标记 disagree）
 *   任意两者差 >8 BPM → 弃权（bpm = null）
 * 若任一算法本窗口无有效估计 → 无法完成两两比较 → 弃权（弃权优于猜测）。
 */
export function voteAlgorithms(algoBpms: (number | null)[]): VoteResult {
  const valid = algoBpms.filter((v): v is number => v !== null && Number.isFinite(v));
  if (valid.length < 3) {
    return {
      bpm: null,
      spread: Number.POSITIVE_INFINITY,
      agreement: valid.length === 0 ? 0 : NaN,
      decision: 'abstain',
      reason: 'insufficient_algorithms',
    };
  }
  const min = Math.min(...valid);
  const max = Math.max(...valid);
  const spread = max - min;
  let tightPairs = 0;
  let pairs = 0;
  for (let i = 0; i < valid.length; i++) {
    for (let j = i + 1; j < valid.length; j++) {
      pairs++;
      if (Math.abs(valid[i] - valid[j]) <= QUALITY.VOTE_TIGHT_BPM) tightPairs++;
    }
  }
  const agreement = pairs > 0 ? tightPairs / pairs : 0;
  const med = median(valid);
  if (spread <= QUALITY.VOTE_TIGHT_BPM) {
    return { bpm: med, spread, agreement, decision: 'ok', reason: null };
  }
  if (spread <= QUALITY.VOTE_LOOSE_BPM) {
    return { bpm: med, spread, agreement, decision: 'disagree', reason: 'algo_disagree_medium' };
  }
  return { bpm: null, spread, agreement, decision: 'abstain', reason: 'algo_disagreement' };
}

// ---------------------------------------------------------------------------
// §7 AE 漂移 / 光照突变检测（软件侧，不依赖 exposureMode 锁定）
// ---------------------------------------------------------------------------

export interface AeDriftResult {
  drifted: boolean;
  /** 帧间最大相对突变（|Δb|/b）。 */
  maxJumpRatio: number;
  /** 发生最大突变处的帧索引；无则 -1。 */
  jumpIndex: number;
  /** 窗口前后半均值相对变化（慢漂移指标）。 */
  slowDriftRatio: number;
}

/**
 * ROI 亮度序列的 AE 漂移检测（文档 §7）：
 *   - 帧间亮度相对突变 > 15% → 判为光照突变；
 *   - 窗口前后半均值相对变化 > 阈值 → 判为慢漂移（AE/白平衡活跃）。
 * 任一命中即 drifted = true → 该窗口不可信。
 */
export function detectAeDrift(
  brightness: Float64Array,
  opts?: { jumpRatio?: number; slowRatio?: number },
): AeDriftResult {
  const jumpThr = opts?.jumpRatio ?? QUALITY.AE_BRIGHTNESS_STEP_RATIO;
  const slowThr = opts?.slowRatio ?? QUALITY.AE_BRIGHTNESS_SLOW_RATIO;
  const n = brightness.length;
  let maxJumpRatio = 0;
  let jumpIndex = -1;
  for (let i = 1; i < n; i++) {
    const prev = Math.abs(brightness[i - 1]);
    const denom = prev > 1e-6 ? prev : 1e-6;
    const ratio = Math.abs(brightness[i] - brightness[i - 1]) / denom;
    if (ratio > maxJumpRatio) {
      maxJumpRatio = ratio;
      jumpIndex = i;
    }
  }
  let slowDriftRatio = 0;
  if (n >= 4) {
    const half = n >> 1;
    let s1 = 0;
    for (let i = 0; i < half; i++) s1 += brightness[i];
    let s2 = 0;
    for (let i = n - half; i < n; i++) s2 += brightness[i];
    const m1 = s1 / half;
    const m2 = s2 / half;
    const ref = Math.abs(m1) > 1e-6 ? Math.abs(m1) : 1e-6;
    slowDriftRatio = Math.abs(m2 - m1) / ref;
  }
  const drifted = maxJumpRatio > jumpThr || slowDriftRatio > slowThr;
  return { drifted, maxJumpRatio, jumpIndex, slowDriftRatio };
}

// ---------------------------------------------------------------------------
// 窗口最终裁决：合并投票 / SNR / AE 漂移 / 饱和 → 是否弃权
// ---------------------------------------------------------------------------

export interface WindowDecisionInput {
  vote: VoteResult;
  snrSpecRaw: number;
  snrHarmRaw: number | null;
  continuity: number;
  aeDrift: AeDriftResult;
  /** 饱和像素占比（可选，>0.3 判无效）。 */
  saturationRatio?: number;
}

export interface WindowDecision {
  bpm: number | null;
  snr: number;
  normSpec: number;
  normHarm: number;
  rejected: boolean;
  reason: string | null;
}

/**
 * 窗口裁决。任一条件命中即弃权（bpm = null）：
 *   1) 三算法弃权（分歧过大 / 算法不足）；
 *   2) AE 漂移或光照突变；
 *   3) 饱和像素占比 > 30%；
 *   4) SNR < 0.35（文档 §3.8）。
 */
export function decideWindow(input: WindowDecisionInput): WindowDecision {
  const { snr, normSpec, normHarm } = combineSnr(input.snrSpecRaw, input.snrHarmRaw, input.continuity);
  let rejected = false;
  let reason: string | null = null;

  if (input.vote.decision === 'abstain') {
    rejected = true;
    reason = input.vote.reason ?? 'algo_disagreement';
  } else if (input.aeDrift.drifted) {
    rejected = true;
    reason = 'ae_drift';
  } else if ((input.saturationRatio ?? 0) > QUALITY.SATURATION_MAX_RATIO) {
    rejected = true;
    reason = 'saturation';
  } else if (snr < QUALITY.SNR_REJECT_THRESHOLD) {
    rejected = true;
    reason = 'low_snr';
  }

  return {
    bpm: rejected ? null : input.vote.bpm,
    snr,
    normSpec,
    normHarm,
    rejected,
    reason,
  };
}
