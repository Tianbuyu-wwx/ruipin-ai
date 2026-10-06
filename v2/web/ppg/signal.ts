// signal.ts — 端侧 rPPG 纯算法层（零依赖 / 零随机数 / 确定性）
//
// 权威规范：docs/模块详设-心率与压力调节评估.md
//   §3.4 滤波  §3.5 心率估计  §3.8 质量门控（SNR 定义见 quality.ts）
//
// 工程约束（来自团队负责人）：
//   - 目标运行环境：浏览器，最终将以 WASM/Worker 形式移植（见 README.md）。
//   - 本文件必须可被 `node --experimental-strip-types` 直接类型擦除运行：
//     不使用 enum / namespace / 参数属性；枚举一律用 `as const` 常量对象代替。
//   - 算法内不得引入随机数：所有迭代初值固定，结果可复现。
//
// 若本文件与实现者先验知识冲突，一律以文档为准，并在注释中标注 "文档优先"。

/** 算法常量。所有数值均标注来源（文档章节或外部论文）。 */
export const PPG = {
  /** 带通下限 (Hz)。文档 §3.4：0.7–4.0 Hz = 42–240 BPM。 */
  BAND_LOW_HZ: 0.7,
  /** 带通上限 (Hz)。文档 §3.4：0.7–4.0 Hz = 42–240 BPM。 */
  BAND_HIGH_HZ: 4.0,
  /** Butterworth 阶数。文档 §3.4。实现为 HP2×LP2 级联（合计 4 阶）。 */
  FILTER_ORDER: 4,
  /** 二阶 Butterworth 的 Q = 1/√2。 */
  BUTTER_Q: Math.SQRT1_2,
  /** 去趋势滑动窗口 = 1.6× 当前估计周期。文档 §3.4。 */
  DETREND_PERIODS: 1.6,
  /** Welch 分段重叠比。文档 §3.5。 */
  WELCH_OVERLAP: 0.75,
  /** SNR 峰带半宽 (Hz)：f0 ± 0.15Hz。文档 §3.8。 */
  SNR_BAND_HZ: 0.15,
  /** 峰值显著性门槛 (dB)：主峰需高出带内均值 ≥3 dB。文档 §3.5。 */
  PEAK_SIGNIFICANCE_DB: 3,
  /** 默认静息心率 (BPM)，仅用于冷启动时推导去趋势窗口长度；不影响估计结果。 */
  DEFAULT_BPM: 72,
  /**
   * 谐波校正：次级（更低）候选峰需达到主峰功率的比例才判定为主峰为谐波。
   * 数值来源：本实现补充（文档 §3.5 仅要求"2×f0 处应有能量"，未给出该比例）。
   */
  HARMONIC_FUNDAMENTAL_MIN_RATIO: 0.3,
} as const;

// ---------------------------------------------------------------------------
// 基础统计
// ---------------------------------------------------------------------------

export function mean(x: Float64Array): number {
  let s = 0;
  for (let i = 0; i < x.length; i++) s += x[i];
  return x.length ? s / x.length : 0;
}

export function std(x: Float64Array): number {
  const n = x.length;
  if (n === 0) return 0;
  const m = mean(x);
  let s = 0;
  for (let i = 0; i < n; i++) {
    const d = x[i] - m;
    s += d * d;
  }
  return Math.sqrt(s / n);
}

export function median(values: number[]): number {
  if (values.length === 0) return NaN;
  const a = values.slice().sort((p, q) => p - q);
  const mid = a.length >> 1;
  return a.length % 2 ? a[mid] : 0.5 * (a[mid - 1] + a[mid]);
}

// ---------------------------------------------------------------------------
// §3.4 去趋势：滑动窗口逐点减均值（抑制慢漂移）
// ---------------------------------------------------------------------------

/**
 * 滑动平均去趋势（居中窗口，边缘按有效样本数归一）。
 * 文档 §3.4：窗口 = 1.6× 当前估计周期。
 * 说明：该操作会部分衰减接近窗口长度的频率分量，但其频率响应在 f0 邻域单调，
 *      不改变功率谱主峰位置，因此不影响 BPM 估计，只抑制更低频的漂移。
 */
export function detrendMovingAverage(x: Float64Array, windowSamples: number): Float64Array {
  const n = x.length;
  const w = Math.max(1, Math.floor(windowSamples));
  const half = Math.floor(w / 2);
  // 前缀和 O(n)
  const ps = new Float64Array(n + 1);
  for (let i = 0; i < n; i++) ps[i + 1] = ps[i] + x[i];
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const lo = Math.max(0, i - half);
    const hi = Math.min(n - 1, i + half);
    const m = (ps[hi + 1] - ps[lo]) / (hi - lo + 1);
    out[i] = x[i] - m;
  }
  return out;
}

// ---------------------------------------------------------------------------
// §3.4 带通：Butterworth 4 阶，零相位（filtfilt / 前向+反向）
// ---------------------------------------------------------------------------

/** 归一化（a0 = 1）的双二阶（biquad）系数。 */
export interface BiquadCoeffs {
  b0: number;
  b1: number;
  b2: number;
  a1: number;
  a2: number;
}

function normalizeBiquad(
  b0: number,
  b1: number,
  b2: number,
  a0: number,
  a1: number,
  a2: number,
): BiquadCoeffs {
  return { b0: b0 / a0, b1: b1 / a0, b2: b2 / a0, a1: a1 / a0, a2: a2 / a0 };
}

/** 二阶低通（RBJ cookbook），fc 单位 Hz。 */
export function biquadLowpass(fc: number, fs: number, q: number): BiquadCoeffs {
  const w0 = (2 * Math.PI * fc) / fs;
  const cw = Math.cos(w0);
  const sw = Math.sin(w0);
  const alpha = sw / (2 * q);
  return normalizeBiquad((1 - cw) / 2, 1 - cw, (1 - cw) / 2, 1 + alpha, -2 * cw, 1 - alpha);
}

/** 二阶高通（RBJ cookbook），fc 单位 Hz。 */
export function biquadHighpass(fc: number, fs: number, q: number): BiquadCoeffs {
  const w0 = (2 * Math.PI * fc) / fs;
  const cw = Math.cos(w0);
  const sw = Math.sin(w0);
  const alpha = sw / (2 * q);
  return normalizeBiquad((1 + cw) / 2, -(1 + cw), (1 + cw) / 2, 1 + alpha, -2 * cw, 1 - alpha);
}

/** 单次前向 IIR（直接 II 型转置，零初值 — 确定性）。 */
function filterForward(x: Float64Array, c: BiquadCoeffs): Float64Array {
  const y = new Float64Array(x.length);
  let x1 = 0;
  let x2 = 0;
  let y1 = 0;
  let y2 = 0;
  for (let i = 0; i < x.length; i++) {
    const xi = x[i];
    const yi = c.b0 * xi + c.b1 * x1 + c.b2 * x2 - c.a1 * y1 - c.a2 * y2;
    y[i] = yi;
    x2 = x1;
    x1 = xi;
    y2 = y1;
    y1 = yi;
  }
  return y;
}

function reverse(x: Float64Array): Float64Array {
  const n = x.length;
  const y = new Float64Array(n);
  for (let i = 0; i < n; i++) y[i] = x[n - 1 - i];
  return y;
}

/** 镜像（不含端点重复）索引映射。 */
function reflectIndex(i: number, n: number): number {
  if (n <= 1) return 0;
  const period = 2 * (n - 1);
  let m = i % period;
  if (m < 0) m += period;
  return m < n ? m : period - m;
}

/**
 * 零相位滤波（filtfilt）：镜像补边 → 前向 → 反向 → 前向 → 反向 → 去补边。
 * 文档 §3.4 要求零相位，以免相位偏移影响事件打点时间对齐。
 */
export function filtfilt(x: Float64Array, c: BiquadCoeffs): Float64Array {
  const n = x.length;
  if (n === 0) return new Float64Array(0);
  const pad = Math.min(n - 1, 32);
  const m = n + 2 * pad;
  const padded = new Float64Array(m);
  for (let k = 0; k < n; k++) padded[pad + k] = x[k];
  for (let j = 0; j < pad; j++) {
    padded[pad - 1 - j] = x[reflectIndex(-1 - j, n)];
    padded[pad + n + j] = x[reflectIndex(n + j, n)];
  }
  const f1 = filterForward(padded, c);
  const f2 = filterForward(reverse(f1), c);
  const out = reverse(f2);
  return out.slice(pad, pad + n);
}

/**
 * 带通滤波 0.7–4.0 Hz（文档 §3.4）：二阶高通(0.7) + 二阶低通(4.0)，各自零相位，
 * 合计 4 阶。返回新数组，不修改输入。
 */
export function bandpassFilter(
  x: Float64Array,
  fs: number,
  low: number = PPG.BAND_LOW_HZ,
  high: number = PPG.BAND_HIGH_HZ,
): Float64Array {
  if (x.length === 0) return new Float64Array(0);
  const nyq = fs / 2;
  const hi = Math.min(high, nyq * 0.99);
  const lo = Math.min(low, hi * 0.5);
  const y = filtfilt(x, biquadHighpass(lo, fs, PPG.BUTTER_Q));
  return filtfilt(y, biquadLowpass(hi, fs, PPG.BUTTER_Q));
}

// ---------------------------------------------------------------------------
// §3.3 三种色度法：CHROM / POS / SSR
// 输入均为 ROI 内逐帧平均后的 R/G/B 时序（长度相同）。
// ---------------------------------------------------------------------------

export interface RgbTrace {
  r: Float64Array;
  g: Float64Array;
  b: Float64Array;
}

/** 按时间均值归一化三通道（r/chrom、pos 的标准前置步骤）。 */
function normalizeByMean(x: Float64Array): Float64Array {
  const m = mean(x);
  const out = new Float64Array(x.length);
  const inv = m !== 0 ? 1 / m : 0;
  for (let i = 0; i < x.length; i++) out[i] = x[i] * inv;
  return out;
}

/**
 * CHROM（de Haan & Jeanne 2013, IEEE TBME）。
 * 文档 §3.3 指定算法之一。
 *   Xs = 3Rn − 2Gn
 *   Ys = 1.5Rn + Gn − 1.5Bn
 *   α  = std(Xs)/std(Ys);  S = Xs − α·Ys
 */
export function chrom(rgb: RgbTrace): Float64Array {
  const n = rgb.r.length;
  const Rn = normalizeByMean(rgb.r);
  const Gn = normalizeByMean(rgb.g);
  const Bn = normalizeByMean(rgb.b);
  const Xs = new Float64Array(n);
  const Ys = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    Xs[i] = 3 * Rn[i] - 2 * Gn[i];
    Ys[i] = 1.5 * Rn[i] + Gn[i] - 1.5 * Bn[i];
  }
  const sy = std(Ys);
  const alpha = sy > 1e-12 ? std(Xs) / sy : 0;
  const S = new Float64Array(n);
  for (let i = 0; i < n; i++) S[i] = Xs[i] - alpha * Ys[i];
  return S;
}

/**
 * POS（Wang et al. 2016, IEEE TBME）。
 * 文档 §3.3 指定算法之一。投影矩阵 P = [[0,1,-1],[-2,1,1]]：
 *   S1 = Gn − Bn;  S2 = −2Rn + Gn + Bn;  α = std(S1)/std(S2);  h = S1 + α·S2
 */
export function pos(rgb: RgbTrace): Float64Array {
  const n = rgb.r.length;
  const Rn = normalizeByMean(rgb.r);
  const Gn = normalizeByMean(rgb.g);
  const Bn = normalizeByMean(rgb.b);
  const S1 = new Float64Array(n);
  const S2 = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    S1[i] = Gn[i] - Bn[i];
    S2[i] = -2 * Rn[i] + Gn[i] + Bn[i];
  }
  const s2 = std(S2);
  const alpha = s2 > 1e-12 ? std(S1) / s2 : 0;
  const h = new Float64Array(n);
  for (let i = 0; i < n; i++) h[i] = S1[i] + alpha * S2[i];
  return h;
}

/**
 * SSR（Wang et al. 2015, IEEE TBME "Algorithmic Principles of Remote PPG"）。
 * 文档 §3.3 指定算法之一。
 *
 * 说明（文档优先 + 工程简化，已注明）：原方法面向多像素/多子区域，在"皮肤平面"内
 * 做子空间旋转以找最大脉动方向。本实现输入为单 ROI 平均后的 3 通道时序，故采用
 * 其**退化形式**：
 *   1) 三通道按均值归一化后去均值，得 3×N 色度变化矩阵 ΔC；
 *   2) 对其 3×3 协方差做确定性幂迭代，取前两个特征向量张成的"变化平面"；
 *   3) 在该平面内以固定步长扫描旋转角 θ，取使带内功率占比最大的投影 p(θ)。
 * 全过程无随机数，旋转角网格固定。
 */
export function ssr(rgb: RgbTrace, fs: number): Float64Array {
  const n = rgb.r.length;
  const Rn = normalizeByMean(rgb.r);
  const Gn = normalizeByMean(rgb.g);
  const Bn = normalizeByMean(rgb.b);
  const dr = new Float64Array(n);
  const dg = new Float64Array(n);
  const db = new Float64Array(n);
  const mr = mean(Rn);
  const mg = mean(Gn);
  const mb = mean(Bn);
  for (let i = 0; i < n; i++) {
    dr[i] = Rn[i] - mr;
    dg[i] = Gn[i] - mg;
    db[i] = Bn[i] - mb;
  }
  // 3×3 协方差（对称）
  let c00 = 0;
  let c01 = 0;
  let c02 = 0;
  let c11 = 0;
  let c12 = 0;
  let c22 = 0;
  for (let i = 0; i < n; i++) {
    c00 += dr[i] * dr[i];
    c01 += dr[i] * dg[i];
    c02 += dr[i] * db[i];
    c11 += dg[i] * dg[i];
    c12 += dg[i] * db[i];
    c22 += db[i] * db[i];
  }
  const inv = n > 0 ? 1 / n : 0;
  c00 *= inv;
  c01 *= inv;
  c02 *= inv;
  c11 *= inv;
  c12 *= inv;
  c22 *= inv;

  const matVec = (v: number[]): number[] => [
    c00 * v[0] + c01 * v[1] + c02 * v[2],
    c01 * v[0] + c11 * v[1] + c12 * v[2],
    c02 * v[0] + c12 * v[1] + c22 * v[2],
  ];
  const norm3 = (v: number[]): number[] => {
    const L = Math.hypot(v[0], v[1], v[2]) || 1e-12;
    return [v[0] / L, v[1] / L, v[2] / L];
  };

  // 主特征向量（固定初值，确定性）
  let v1 = norm3([1, 1, 1]);
  for (let it = 0; it < 100; it++) v1 = norm3(matVec(v1));
  const mv1 = matVec(v1);
  const lambda1 = v1[0] * mv1[0] + v1[1] * mv1[1] + v1[2] * mv1[2];

  // 次特征向量：对 C − λ1·v1·v1ᵀ 幂迭代，初值取与 v1 最不正交的坐标轴
  let seed = [1, 0, 0];
  const abs1 = v1.map((x) => Math.abs(x));
  const kmin = abs1.indexOf(Math.min(abs1[0], abs1[1], abs1[2]));
  seed = [0, 0, 0];
  seed[kmin] = 1;
  seed = norm3(seed);
  const orth = (v: number[], u: number[]): number[] => {
    const d = v[0] * u[0] + v[1] * u[1] + v[2] * u[2];
    return norm3([v[0] - d * u[0], v[1] - d * u[1], v[2] - d * u[2]]);
  };
  let v2 = orth(seed, v1);
  for (let it = 0; it < 100; it++) {
    const mv = matVec(v2);
    const d = v1[0] * mv[0] + v1[1] * mv[1] + v1[2] * mv[2];
    const deflated = [mv[0] - lambda1 * d * v1[0], mv[1] - lambda1 * d * v1[1], mv[2] - lambda1 * d * v1[2]];
    v2 = orth(deflated, v1);
  }

  const s1 = new Float64Array(n);
  const s2 = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    s1[i] = v1[0] * dr[i] + v1[1] * dg[i] + v1[2] * db[i];
    s2[i] = v2[0] * dr[i] + v2[1] * dg[i] + v2[2] * db[i];
  }

  // 平面内旋转扫描，选带内功率占比最大者
  const STEPS = 180;
  let bestScore = -1;
  let best = new Float64Array(n);
  const varOf = (a: Float64Array): number => {
    const m = mean(a);
    let s = 0;
    for (let i = 0; i < a.length; i++) {
      const d = a[i] - m;
      s += d * d;
    }
    return s / (a.length || 1);
  };
  for (let s = 0; s < STEPS; s++) {
    const theta = (Math.PI * s) / STEPS;
    const ct = Math.cos(theta);
    const st = Math.sin(theta);
    const p = new Float64Array(n);
    for (let i = 0; i < n; i++) p[i] = ct * s1[i] + st * s2[i];
    const bp = bandpassFilter(p, fs);
    const total = varOf(p);
    const score = total > 1e-15 ? varOf(bp) / total : 0;
    if (score > bestScore) {
      bestScore = score;
      best = p;
    }
  }
  return best;
}

// ---------------------------------------------------------------------------
// FFT / Welch PSD（§3.5：Welch 功率谱，重叠 75%）
// ---------------------------------------------------------------------------

export function nextPow2(n: number): number {
  let p = 1;
  while (p < n) p <<= 1;
  return p;
}

/** 迭代式 radix-2 原地 FFT（确定性）。长度必须是 2 的幂。 */
export function fftRadix2(re: Float64Array, im: Float64Array): void {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      const tr = re[i];
      re[i] = re[j];
      re[j] = tr;
      const ti = im[i];
      im[i] = im[j];
      im[j] = ti;
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2 * Math.PI) / len;
    const wRe = Math.cos(ang);
    const wIm = Math.sin(ang);
    const half = len >> 1;
    for (let i = 0; i < n; i += len) {
      let curRe = 1;
      let curIm = 0;
      for (let j = 0; j < half; j++) {
        const uRe = re[i + j];
        const uIm = im[i + j];
        const vRe = re[i + j + half] * curRe - im[i + j + half] * curIm;
        const vIm = re[i + j + half] * curIm + im[i + j + half] * curRe;
        re[i + j] = uRe + vRe;
        im[i + j] = uIm + vIm;
        re[i + j + half] = uRe - vRe;
        im[i + j + half] = uIm - vIm;
        const nRe = curRe * wRe - curIm * wIm;
        curIm = curRe * wIm + curIm * wRe;
        curRe = nRe;
      }
    }
  }
}

export interface WelchResult {
  freqs: Float64Array;
  psd: Float64Array;
  /** 频率分辨率 (Hz) */
  df: number;
}

/**
 * Welch 功率谱（Hann 窗 + 平均周期图）。
 * 文档 §3.5：窗口 8 s，重叠 75%。本实现中 segmentLength 默认取整段（= 分析窗），
 * overlap 在 segmentLength < 数据长度 时生效；FFT 采用 2 的幂补零以提升峰值插值精度。
 */
export function welchPSD(
  x: Float64Array,
  fs: number,
  opts?: { segmentLength?: number; overlap?: number },
): WelchResult {
  const n = x.length;
  const seg = Math.max(4, Math.min(opts?.segmentLength ?? n, n));
  const overlap = opts?.overlap ?? PPG.WELCH_OVERLAP;
  const step = Math.max(1, Math.round(seg * (1 - overlap)));
  const nfft = nextPow2(seg);
  const half = nfft >> 1;

  const w = new Float64Array(seg);
  let sumW2 = 0;
  for (let i = 0; i < seg; i++) {
    w[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (seg - 1 || 1));
    sumW2 += w[i] * w[i];
  }

  const acc = new Float64Array(half + 1);
  let count = 0;
  for (let start = 0; start + seg <= n; start += step) {
    let m = 0;
    for (let i = 0; i < seg; i++) m += x[start + i];
    m /= seg;
    const re = new Float64Array(nfft);
    const im = new Float64Array(nfft);
    for (let i = 0; i < seg; i++) re[i] = (x[start + i] - m) * w[i];
    fftRadix2(re, im);
    for (let k = 0; k <= half; k++) {
      let p = re[k] * re[k] + im[k] * im[k];
      if (k !== 0 && k !== half) p *= 2; // 单边谱
      acc[k] += p;
    }
    count++;
  }
  if (count === 0) {
    // 数据短于一段：整段作为单段处理
    const re = new Float64Array(nfft);
    const im = new Float64Array(nfft);
    for (let i = 0; i < n; i++) re[i] = x[i] * w[i];
    fftRadix2(re, im);
    for (let k = 0; k <= half; k++) {
      let p = re[k] * re[k] + im[k] * im[k];
      if (k !== 0 && k !== half) p *= 2;
      acc[k] += p;
    }
    count = 1;
  }
  const scale = 1 / (fs * (sumW2 || 1));
  const psd = new Float64Array(half + 1);
  for (let k = 0; k <= half; k++) psd[k] = (acc[k] / count) * scale;
  const freqs = new Float64Array(half + 1);
  const df = fs / nfft;
  for (let k = 0; k <= half; k++) freqs[k] = k * df;
  return { freqs, psd, df };
}

// ---------------------------------------------------------------------------
// §3.5 心率估计：主峰 + 峰值插值 + 谐波校正 + 带外主峰拒绝
// ---------------------------------------------------------------------------

export interface HrEstimate {
  /** 估计心率 (BPM)。 */
  bpm: number;
  /** 基频 (Hz)。 */
  f0Hz: number;
  /** 主峰相对带内均值的 dB 值（显著性指标，§3.5 要求 ≥3 dB）。 */
  peakDb: number;
  /** 是否满足 §3.5 的峰值显著性要求。 */
  significant: boolean;
  /** 若发生了谐波校正，记录校正前的 bpm；否则 null。 */
  harmonicCorrectedFromBpm: number | null;
}

/** 守门功率谱：未经带通、仅去均值的全谱，用于"拒绝带外主峰"（文档 §3.4）。 */
export interface GuardSpectrum {
  freqs: Float64Array;
  psd: Float64Array;
}

/**
 * 带外主峰判定（文档 §3.4"拒绝带外主峰"）。
 *
 * 为什么需要额外守门谱：带通滤波器在截止频率附近并非砖墙。以 40 BPM(0.667Hz) 为例，
 * 0.7Hz 高通只衰减约 −7 dB，带通后其能量仍会以"带内边缘"的形态出现，仅凭带通谱
 * 无法识别。故额外观察未带通的全谱。
 *
 * 为避免把宽带 1/f 漂移误判为"带外主峰"，要求带外峰同时满足：比带内主峰更强
 * **且** 相对其所在频段中位数显著（≥3×），即必须是窄带谱峰而非宽带漂移。
 */
function hasDominantOutOfBandPeak(inBandPeak: number, guard: GuardSpectrum): boolean {
  const lo = PPG.BAND_LOW_HZ;
  const hi = PPG.BAND_HIGH_HZ;
  const { freqs, psd } = guard;
  const collect = (from: number, to: number) => {
    let max = 0;
    const vals: number[] = [];
    for (let k = 0; k < freqs.length; k++) {
      const f = freqs[k];
      if (f >= from && f < to) {
        if (psd[k] > max) max = psd[k];
        vals.push(psd[k]);
      }
    }
    const med = median(vals);
    return { max, med: Number.isFinite(med) ? med : 0 };
  };
  const below = collect(0.15, lo);
  const above = collect(hi, freqs[freqs.length - 1] + 1e-9);
  const narrow = (s: { max: number; med: number }) => s.max > inBandPeak && s.max > 3 * s.med && s.max > 0;
  return narrow(below) || narrow(above);
}

/** 指定中心与半宽内的平均功率。 */
export function bandPowerAt(
  freqs: Float64Array,
  psd: Float64Array,
  centerHz: number,
  halfWidthHz: number,
): number {
  let s = 0;
  let c = 0;
  for (let k = 0; k < freqs.length; k++) {
    if (Math.abs(freqs[k] - centerHz) <= halfWidthHz) {
      s += psd[k];
      c++;
    }
  }
  return c > 0 ? s / c : 0;
}

/**
 * 从 PSD 估计心率。
 * - 仅在通带 [0.7, 4.0] Hz 内取主峰；
 * - 若守门全谱显示通带外（更低或更高频）存在更强的窄带主峰 → 判为带外主峰，返回 null（文档 §3.4）；
 * - 对数功率抛物线插值提高频率分辨率；
 * - 谐波校正：若 f0/2 或 f0/3 处有足够能量，则判定当前主峰为谐波，改取基频（§3.5）。
 */
export function estimateHrFromPsd(
  freqs: Float64Array,
  psd: Float64Array,
  guard?: GuardSpectrum,
): HrEstimate | null {
  const lo = PPG.BAND_LOW_HZ;
  const hi = PPG.BAND_HIGH_HZ;
  const n = freqs.length;
  if (n < 3) return null;

  let bestK = -1;
  let bestV = 0;
  for (let k = 0; k < n; k++) {
    const f = freqs[k];
    const v = psd[k];
    if (f >= lo && f <= hi && v > bestV) {
      bestV = v;
      bestK = k;
    }
  }
  if (bestK < 0 || bestV <= 0) return null;
  // 带外主峰：更强的窄带能量在通带之外 → 弃权（拒绝带外主峰，文档 §3.4）
  if (guard && hasDominantOutOfBandPeak(bestV, guard)) return null;

  const df = freqs[1] - freqs[0];
  let f0 = freqs[bestK];
  // 对数功率抛物线插值
  if (bestK > 0 && bestK < n - 1 && psd[bestK - 1] > 0 && psd[bestK + 1] > 0) {
    const l0 = Math.log(psd[bestK - 1]);
    const l1 = Math.log(psd[bestK]);
    const l2 = Math.log(psd[bestK + 1]);
    const denom = l0 - 2 * l1 + l2;
    if (Math.abs(denom) > 1e-12) {
      const delta = (0.5 * (l0 - l2)) / denom;
      if (delta > -1 && delta < 1) f0 = (bestK + delta) * df;
    }
  }

  // 峰值显著性：主峰 vs 带内均值（dB）
  let sum = 0;
  let cnt = 0;
  for (let k = 0; k < n; k++) {
    if (freqs[k] >= lo && freqs[k] <= hi) {
      sum += psd[k];
      cnt++;
    }
  }
  const meanBand = cnt > 0 ? sum / cnt : 0;
  const peakDb = 10 * Math.log10(bestV / (meanBand || 1e-15));

  // 谐波校正（§3.5）
  const peakPow = bandPowerAt(freqs, psd, f0, PPG.SNR_BAND_HZ);
  let harmonicCorrectedFromBpm: number | null = null;
  for (const div of [2, 3]) {
    const fc = f0 / div;
    if (fc < lo) continue;
    const fp = bandPowerAt(freqs, psd, fc, PPG.SNR_BAND_HZ);
    if (fp >= PPG.HARMONIC_FUNDAMENTAL_MIN_RATIO * peakPow) {
      harmonicCorrectedFromBpm = f0 * 60;
      f0 = fc;
      break;
    }
  }

  if (f0 < lo || f0 > hi) return null;
  return {
    bpm: f0 * 60,
    f0Hz: f0,
    peakDb,
    significant: peakDb >= PPG.PEAK_SIGNIFICANCE_DB,
    harmonicCorrectedFromBpm,
  };
}

// ---------------------------------------------------------------------------
// 单算法完整链路：BVP 波形 → 去趋势 → 带通 → Welch → HR 估计
// ---------------------------------------------------------------------------

export type AlgoName = 'chrom' | 'pos' | 'ssr';

export interface ChannelAnalysis {
  algo: AlgoName;
  /** 估计心率 (BPM)；null = 该算法本窗口无有效带内主峰。 */
  bpm: number | null;
  estimate: HrEstimate | null;
  /** 带通后的 BVP 波形（供 SNR 与诊断使用）。 */
  bvp: Float64Array;
  psd: Float64Array;
  freqs: Float64Array;
}

export interface AnalyzeOptions {
  /** 去趋势滑动窗口长度（样本）。缺省按 DEFAULT_BPM 的 1.6 周期。 */
  detrendSamples?: number;
  /** Welch 分段长度（样本）。缺省 = 分析窗长度。 */
  welchSegmentLength?: number;
}

/** 由目标心率推导去趋势窗口样本数（窗口 = 1.6× 周期，文档 §3.4）。 */
export function detrendSamplesForBpm(bpm: number, fs: number): number {
  const b = bpm > 0 ? bpm : PPG.DEFAULT_BPM;
  return Math.max(3, Math.round((PPG.DETREND_PERIODS * 60 * fs) / b));
}

function analyzeOne(
  algo: AlgoName,
  raw: Float64Array,
  fs: number,
  opts?: AnalyzeOptions,
): ChannelAnalysis {
  const detrendSamples = opts?.detrendSamples ?? detrendSamplesForBpm(PPG.DEFAULT_BPM, fs);
  const detrended = detrendMovingAverage(raw, detrendSamples);
  const bvp = bandpassFilter(detrended, fs);
  const seg = opts?.welchSegmentLength ?? bvp.length;
  const { freqs, psd } = welchPSD(bvp, fs, { segmentLength: seg, overlap: PPG.WELCH_OVERLAP });
  // 守门谱：未带通、仅去时间均值的全谱（用于拒绝带外主峰，文档 §3.4）
  const rawCentered = new Float64Array(raw.length);
  const rawMean = mean(raw);
  for (let i = 0; i < raw.length; i++) rawCentered[i] = raw[i] - rawMean;
  const guardWelch = welchPSD(rawCentered, fs, { segmentLength: seg, overlap: PPG.WELCH_OVERLAP });
  const estimate = estimateHrFromPsd(freqs, psd, { freqs: guardWelch.freqs, psd: guardWelch.psd });
  return { algo, bpm: estimate ? estimate.bpm : null, estimate, bvp, psd, freqs };
}

/**
 * 同一窗口上并行运行 CHROM / POS / SSR 三个算法，各自输出 BVP 与 HR 估计。
 * 文档 §3.3。
 */
export function analyzeRgb(
  rgb: RgbTrace,
  fs: number,
  opts?: AnalyzeOptions,
): { chrom: ChannelAnalysis; pos: ChannelAnalysis; ssr: ChannelAnalysis } {
  return {
    chrom: analyzeOne('chrom', chrom(rgb), fs, opts),
    pos: analyzeOne('pos', pos(rgb), fs, opts),
    ssr: analyzeOne('ssr', ssr(rgb, fs), fs, opts),
  };
}
