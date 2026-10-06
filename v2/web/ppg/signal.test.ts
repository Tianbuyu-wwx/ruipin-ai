// signal.test.ts — 纯算法层单元测试
//
// 运行：node --experimental-strip-types --test "web/ppg/*.test.ts"
// 每条测试的中文注释说明它"防什么回归"。

import { test } from 'node:test';
import assert from 'node:assert/strict';

import {
  PPG,
  analyzeRgb,
  bandpassFilter,
  chrom,
  detrendMovingAverage,
  estimateHrFromPsd,
  fftRadix2,
  filtfilt,
  biquadLowpass,
  pos,
  ssr,
  welchPSD,
} from './signal.ts';
import { synthPulse } from './synth.ts';

function windowRgb(frames: ReturnType<typeof synthPulse>, n: number) {
  const r = new Float64Array(n);
  const g = new Float64Array(n);
  const b = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    r[i] = frames[i].r;
    g[i] = frames[i].g;
    b[i] = frames[i].b;
  }
  return { r, g, b };
}

test('FFT：与朴素 DFT 逐点一致（防 FFT 实现错误导致谱峰错位）', () => {
  const n = 16;
  const re = new Float64Array(n);
  const im = new Float64Array(n);
  const refRe = new Float64Array(n);
  const refIm = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    re[i] = Math.sin(0.3 * i) + 0.5 * Math.cos(0.7 * i);
    refRe[i] = re[i];
  }
  fftRadix2(re, im);
  for (let k = 0; k < n; k++) {
    let sr = 0;
    let si = 0;
    for (let i = 0; i < n; i++) {
      const ang = (-2 * Math.PI * k * i) / n;
      sr += refRe[i] * Math.cos(ang);
      si += refRe[i] * Math.sin(ang);
    }
    assert.ok(Math.abs(re[k] - sr) < 1e-9, `k=${k} re`);
    assert.ok(Math.abs(im[k] - si) < 1e-9, `k=${k} im`);
  }
});

test('Welch PSD：主峰落在输入正弦频率 ±1 个频率单元（防谱估计频率刻度错误）', () => {
  const fs = 30;
  const f = 1.2; // 72 BPM
  const n = 240; // 8 s
  const x = new Float64Array(n);
  for (let i = 0; i < n; i++) x[i] = Math.sin((2 * Math.PI * f * i) / fs);
  const { freqs, psd, df } = welchPSD(x, fs);
  let k = 1;
  for (let i = 2; i < psd.length; i++) if (psd[i] > psd[k]) k = i;
  assert.ok(Math.abs(freqs[k] - f) <= df, `peak=${freqs[k]}, f=${f}, df=${df}`);
});

test('零相位滤波：对称脉冲经带通后峰值位置不变（防相位偏移破坏事件打点对齐）', () => {
  const fs = 30;
  const n = 240;
  const center = 120;
  const x = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const d = i - center;
    x[i] = Math.exp(-(d * d) / (2 * 6 * 6)); // 关于 center 对称的宽带脉冲
  }
  const y = bandpassFilter(x, fs);
  let k = 0;
  for (let i = 1; i < y.length; i++) if (y[i] > y[k]) k = i;
  assert.ok(Math.abs(k - center) <= 1, `argmax=${k}, center=${center}`);
});

test('filtfilt 低通不改变常量的直流位置随频率单调衰减（防滤波器设计成高通/增益异常）', () => {
  const fs = 30;
  const coeffs = biquadLowpass(2, fs, PPG.BUTTER_Q);
  // 直流应保持（低通）
  const dc = new Float64Array(200).fill(1);
  const ydc = filtfilt(dc, coeffs);
  assert.ok(Math.abs(ydc[100] - 1) < 1e-6, `dc=${ydc[100]}`);
  // 高频应衰减
  const hi = new Float64Array(200);
  for (let i = 0; i < 200; i++) hi[i] = Math.sin((2 * Math.PI * 10 * i) / fs);
  const yhi = filtfilt(hi, coeffs);
  const ampIn = 1;
  let ampOut = 0;
  for (let i = 50; i < 150; i++) ampOut = Math.max(ampOut, Math.abs(yhi[i]));
  assert.ok(ampOut < 0.3 * ampIn, `10Hz 残留幅度=${ampOut.toFixed(4)}`);
});

test('去趋势：去除线性漂移分量（防慢漂移污染低频带）', () => {
  const n = 300;
  const x = new Float64Array(n);
  for (let i = 0; i < n; i++) x[i] = 100 + 0.5 * i; // 强线性漂移
  const y = detrendMovingAverage(x, 30);
  const mid = y.slice(100, 200);
  let maxAbs = 0;
  for (const v of mid) maxAbs = Math.max(maxAbs, Math.abs(v));
  assert.ok(maxAbs < 20, `去趋势后中段残留=${maxAbs.toFixed(2)}`);
});

test('三色度法在纯净脉搏上给出接近一致的 BVP（防某算法公式写错）', () => {
  const fs = 30;
  const frames = synthPulse({ bpm: 72, seconds: 8, noise: 0 });
  const rgb = windowRgb(frames, 240);
  const a = analyzeRgb(rgb, fs);
  const bpms = [a.chrom.bpm, a.pos.bpm, a.ssr.bpm];
  for (const v of bpms) {
    assert.ok(v !== null && Math.abs(v - 72) <= 1, `algo bpm=${v}`);
  }
  // BVP 波形非平凡（方差不退化）
  const variance = (arr: Float64Array) => {
    let m = 0;
    for (const v of arr) m += v;
    m /= arr.length;
    let s = 0;
    for (const v of arr) s += (v - m) * (v - m);
    return s / arr.length;
  };
  assert.ok(variance(chrom(rgb)) > 0, 'chrom 方差应 > 0');
  assert.ok(variance(pos(rgb)) > 0, 'pos 方差应 > 0');
  assert.ok(variance(ssr(rgb, fs)) > 0, 'ssr 方差应 > 0');
});

test('谐波校正：二次谐波更强时仍取基频（防把 2×f0 当成心率，60→120 错误）', () => {
  const fs = 30;
  // 基频 60 BPM(1Hz) 幅度 1，二次谐波 120 BPM(2Hz) 幅度 1.5（更强）
  const frames = synthPulse({ bpm: 60, seconds: 8, noise: 0, harmonics: [{ mult: 2, amp: 1.5 }] });
  const a = analyzeRgb(windowRgb(frames, 240), fs);
  assert.ok(a.chrom.bpm !== null && Math.abs(a.chrom.bpm - 60) <= 1, `chrom=${a.chrom.bpm}`);
  assert.ok(a.pos.bpm !== null && Math.abs(a.pos.bpm - 60) <= 1, `pos=${a.pos.bpm}`);
});

test('谐波校正不误伤：真实 120 BPM 且无基频时不得被折半到 60（防过度校正）', () => {
  const fs = 30;
  const frames = synthPulse({ bpm: 120, seconds: 8, noise: 0, harmonics: [{ mult: 2, amp: 0.5 }] });
  const a = analyzeRgb(windowRgb(frames, 240), fs);
  assert.ok(a.chrom.bpm !== null && Math.abs(a.chrom.bpm - 120) <= 1, `chrom=${a.chrom.bpm}`);
});

test('通带边界：40 BPM(低于 0.7Hz) 判无信号，260 BPM(高于 4Hz) 判无信号（防输出带外错误心率）', () => {
  const fs = 30;
  const low = analyzeRgb(windowRgb(synthPulse({ bpm: 40, seconds: 8, noise: 0 }), 240), fs);
  const high = analyzeRgb(windowRgb(synthPulse({ bpm: 260, seconds: 8, noise: 0 }), 240), fs);
  assert.equal(low.chrom.bpm, null, '40 BPM 应弃权');
  assert.equal(low.pos.bpm, null, '40 BPM 应弃权');
  assert.equal(high.chrom.bpm, null, '260 BPM 应弃权');
  assert.equal(high.pos.bpm, null, '260 BPM 应弃权');
});

test('通带内 200 BPM 是有效的（澄清：文档通带上限 240 BPM，200 在带内）', () => {
  const fs = 30;
  const a = analyzeRgb(windowRgb(synthPulse({ bpm: 200, seconds: 8, noise: 0 }), 240), fs);
  assert.ok(a.chrom.bpm !== null && Math.abs(a.chrom.bpm - 200) <= 2, `chrom=${a.chrom.bpm}`);
});

test('确定性：同一输入两次分析逐值相等（防引入随机数/不确定初始化）', () => {
  const fs = 30;
  const frames = synthPulse({ bpm: 84, seconds: 8, noise: 2.0, seed: 99 });
  const rgb = windowRgb(frames, 240);
  const a1 = analyzeRgb(rgb, fs);
  const a2 = analyzeRgb(rgb, fs);
  assert.equal(a1.chrom.bpm, a2.chrom.bpm);
  assert.equal(a1.pos.bpm, a2.pos.bpm);
  assert.equal(a1.ssr.bpm, a2.ssr.bpm);
  for (let i = 0; i < a1.chrom.psd.length; i++) {
    assert.equal(a1.chrom.psd[i], a2.chrom.psd[i]);
  }
});

test('estimateHrFromPsd：全零谱返回 null（防空输入崩溃或伪造峰值）', () => {
  const freqs = new Float64Array([0, 0.5, 1, 1.5, 2]);
  const psd = new Float64Array(5);
  assert.equal(estimateHrFromPsd(freqs, psd), null);
});
