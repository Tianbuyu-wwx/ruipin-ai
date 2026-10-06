// synth.ts — 测试用合成信号发生器（确定性，无第三方依赖）
//
// 用于在**没有摄像头**的条件下验证信号链：构造已知 BPM 的三通道血容积调制信号，
// 并可叠加噪声 / 运动伪影 / 谐波 / 亮度阶跃。
//
// 不是算法的一部分，不参与打包；文件名不匹配 *.test.ts，不会被测试运行器当作测试。

import type { PpgFrame } from './pipeline.ts';

export interface SynthOptions {
  fs?: number;
  seconds?: number;
  bpm?: number;
  /** 各通道加性高斯白噪标准差（与 AC 同量纲）。 */
  noise?: number;
  /** 直流偏置（典型肤色量级）。 */
  dc?: { r: number; g: number; b: number };
  /** 血容积 AC 幅度（绿通道吸收最强，故增益最大）。 */
  ac?: { r: number; g: number; b: number };
  /** 谐波分量：mult 倍基频、相对基频的幅度 amp。 */
  harmonics?: { mult: number; amp: number }[];
  /** 运动/光照伪影：单一频率，逐通道幅度可不同（不同幅度会诱发三算法分歧）。 */
  motion?: { freqHz: number; ampR: number; ampG: number; ampB: number };
  /** 亮度阶跃（模拟 AE/开关灯）：atSec 之后所有通道乘以 factor。 */
  brightnessStep?: { atSec: number; factor: number };
  /** 随机种子（仅影响噪声，保证可复现）。 */
  seed?: number;
}

/** mulberry32 —— 确定性 PRNG。 */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** 标准正态生成器（Box–Muller，确定性）。 */
export function makeRandn(seed: number): () => number {
  const rand = mulberry32(seed);
  let spare: number | null = null;
  return () => {
    if (spare !== null) {
      const s = spare;
      spare = null;
      return s;
    }
    let u = 0;
    let v = 0;
    let s = 0;
    do {
      u = rand() * 2 - 1;
      v = rand() * 2 - 1;
      s = u * u + v * v;
    } while (s === 0 || s >= 1);
    const mul = Math.sqrt((-2 * Math.log(s)) / s);
    spare = v * mul;
    return u * mul;
  };
}

/**
 * 合成脉搏信号。
 * 模型：c(t) = DC_c + AC_c·pulse(t) + motion_c(t) + noise_c(t)
 *   pulse(t) = sin(2π f t) + Σ h.amp·sin(2π f·h.mult·t)
 */
export function synthPulse(o: SynthOptions = {}): PpgFrame[] {
  const fs = o.fs ?? 30;
  const seconds = o.seconds ?? 20;
  const bpm = o.bpm ?? 72;
  const f = bpm / 60;
  const dc = o.dc ?? { r: 180, g: 140, b: 120 };
  const ac = o.ac ?? { r: 2.0, g: 4.0, b: 1.2 };
  const noiseStd = o.noise ?? 0;
  const harmonics = o.harmonics ?? [];
  const motion = o.motion ?? null;
  const step = o.brightnessStep ?? null;
  const randn = makeRandn(o.seed ?? 12345);
  const n = Math.round(fs * seconds);
  const frames: PpgFrame[] = [];
  for (let i = 0; i < n; i++) {
    const t = i / fs;
    let pulse = Math.sin(2 * Math.PI * f * t);
    for (const h of harmonics) pulse += h.amp * Math.sin(2 * Math.PI * f * h.mult * t);
    const mR = motion ? motion.ampR * Math.sin(2 * Math.PI * motion.freqHz * t) : 0;
    const mG = motion ? motion.ampG * Math.sin(2 * Math.PI * motion.freqHz * t) : 0;
    const mB = motion ? motion.ampB * Math.sin(2 * Math.PI * motion.freqHz * t) : 0;
    let k = 1;
    if (step && t >= step.atSec) k = step.factor;
    frames.push({
      t_ms: Math.round((i * 1000) / fs),
      r: k * (dc.r + ac.r * pulse + mR + noiseStd * randn()),
      g: k * (dc.g + ac.g * pulse + mG + noiseStd * randn()),
      b: k * (dc.b + ac.b * pulse + mB + noiseStd * randn()),
    });
  }
  return frames;
}

/** 常用合成参数：明亮静坐、无噪、典型肤色。 */
export const SYNTH_CLEAN = { fs: 30, seconds: 20, dc: { r: 180, g: 140, b: 120 }, ac: { r: 2.0, g: 4.0, b: 1.2 } } as const;
