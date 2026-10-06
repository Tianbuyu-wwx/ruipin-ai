// quality.test.ts — 质量门控单元测试（投票 / SNR / AE 漂移 / 连续性）
//
// 运行：node --experimental-strip-types --test "web/ppg/*.test.ts"

import { test } from 'node:test';
import assert from 'node:assert/strict';

import {
  QUALITY,
  combineSnr,
  computeContinuity,
  decideWindow,
  detectAeDrift,
  snrHarmRatio,
  snrSpecRatio,
  voteAlgorithms,
} from './quality.ts';
import { welchPSD } from './signal.ts';
import type { AeDriftResult, VoteResult } from './quality.ts';

function cleanAe(): AeDriftResult {
  return { drifted: false, maxJumpRatio: 0, jumpIndex: -1, slowDriftRatio: 0 };
}

test('投票：三者一致（差≤3）取中位数且 decision=ok（防一致时误弃权）', () => {
  const v = voteAlgorithms([70, 71, 72]);
  assert.equal(v.decision, 'ok');
  assert.equal(v.bpm, 71);
  assert.ok(v.spread <= QUALITY.VOTE_TIGHT_BPM);
});

test('投票：中等分歧（3–8 BPM）取中位数并标记 disagree（防中等分歧被当成一致）', () => {
  const v = voteAlgorithms([68, 72, 75]);
  assert.equal(v.decision, 'disagree');
  assert.equal(v.bpm, 72);
  assert.ok(v.spread > QUALITY.VOTE_TIGHT_BPM && v.spread <= QUALITY.VOTE_LOOSE_BPM);
});

test('投票：分歧 >8 BPM → 弃权 bpm=null（本模块第一纪律：分歧大就不给数）', () => {
  const v = voteAlgorithms([60, 75, 90]);
  assert.equal(v.decision, 'abstain');
  assert.equal(v.bpm, null);
  assert.ok(v.spread > QUALITY.VOTE_LOOSE_BPM);
});

test('投票：任一算法无估计 → 弃权（防在信息不全时臆测）', () => {
  const v = voteAlgorithms([60, null, 90]);
  assert.equal(v.decision, 'abstain');
  assert.equal(v.bpm, null);
});

test('decideWindow：分歧弃权即便 SNR 很高也不出数（防高 SNR 掩盖算法分歧）', () => {
  const vote: VoteResult = { bpm: null, spread: 30, agreement: 0, decision: 'abstain', reason: 'algo_disagreement' };
  const d = decideWindow({ vote, snrSpecRaw: 1e6, snrHarmRaw: 5, continuity: 1, aeDrift: cleanAe() });
  assert.equal(d.bpm, null);
  assert.equal(d.rejected, true);
  assert.equal(d.reason, 'algo_disagreement');
});

test('SNR_spec：纯净正弦的峰显著度远大于噪声（防 SNR 定义反了）', () => {
  const fs = 30;
  const n = 240;
  const sine = new Float64Array(n);
  const noise = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    sine[i] = Math.sin((2 * Math.PI * 1.2 * i) / fs);
    noise[i] = (Math.sin(7.1 * i) + Math.cos(3.3 * i)) * 1.5;
  }
  const s = welchPSD(sine, fs);
  const q = welchPSD(noise, fs);
  const specSine = snrSpecRatio(s.freqs, s.psd, 1.2);
  const specNoise = snrSpecRatio(q.freqs, q.psd, 1.2);
  assert.ok(specSine > 100 * specNoise, `sine=${specSine.toExponential(2)} noise=${specNoise.toExponential(2)}`);
});

test('SNR_harm：2·f0 超过奈奎斯特时返回 null（防空访问越界并允许中性处理）', () => {
  const fs = 30;
  const n = 240;
  const x = new Float64Array(n);
  for (let i = 0; i < n; i++) x[i] = Math.sin((2 * Math.PI * 1.0 * i) / fs);
  const { freqs, psd } = welchPSD(x, fs);
  // f0 = 8 Hz → 2f0 = 16 Hz > 15 Hz(Nyquist)
  assert.equal(snrHarmRatio(freqs, psd, 8), null);
  // f0 = 1 Hz → 2f0 = 2 Hz 在范围内
  assert.ok(snrHarmRatio(freqs, psd, 1) !== null);
});

test('combineSnr：低 SNR_spec + 低谐波 + 连续性 0 一定低于弃权阈值（防阈值不生效）', () => {
  const bad = combineSnr(1, 0.5, 0);
  assert.ok(bad.snr < QUALITY.SNR_REJECT_THRESHOLD, `snr=${bad.snr}`);
  const good = combineSnr(1e6, 8, 1);
  assert.ok(good.snr > 0.7, `snr=${good.snr}`);
  assert.ok(good.snr <= 1 && bad.snr >= 0, 'SNR 必须落在 [0,1]');
});

test('computeContinuity：超过 8 BPM 跳变被计入违反（防连续性指标恒为 1）', () => {
  assert.equal(computeContinuity([70, 71, 72]), 1);
  const c = computeContinuity([70, 80, 81]); // 第一对跳变 10 > 8，第二对 1 ≤ 8
  assert.ok(Math.abs(c - 0.5) < 1e-9, `c=${c}`);
  assert.equal(computeContinuity([null, null]), 1, '无有效对时中性为 1');
});

test('AE 漂移：亮度阶跃 20% 被判为不可信（防 AE 突变窗口污染心率）', () => {
  const n = 240;
  const b = new Float64Array(n);
  for (let i = 0; i < n; i++) b[i] = i < 180 ? 150 : 180; // +20% 阶跃
  const r = detectAeDrift(b);
  assert.equal(r.drifted, true);
  assert.ok(r.maxJumpRatio > QUALITY.AE_BRIGHTNESS_STEP_RATIO, `jump=${r.maxJumpRatio}`);
});

test('AE 漂移：稳定亮度不误报（防正常信号被当作漂移而全部弃权）', () => {
  const n = 240;
  const b = new Float64Array(n);
  for (let i = 0; i < n; i++) b[i] = 150 + 1.5 * Math.sin((2 * Math.PI * 1.2 * i) / 30); // 正常脉动 <2%
  const r = detectAeDrift(b);
  assert.equal(r.drifted, false);
});

test('decideWindow：AE 漂移窗口被拒（reason=ae_drift）', () => {
  const vote: VoteResult = { bpm: 72, spread: 1, agreement: 1, decision: 'ok', reason: null };
  const d = decideWindow({
    vote,
    snrSpecRaw: 1e6,
    snrHarmRaw: 5,
    continuity: 1,
    aeDrift: { drifted: true, maxJumpRatio: 0.2, jumpIndex: 10, slowDriftRatio: 0.2 },
  });
  assert.equal(d.rejected, true);
  assert.equal(d.reason, 'ae_drift');
  assert.equal(d.bpm, null);
});

test('decideWindow：低 SNR 窗口被拒（reason=low_snr）——测不准就说出"不知道"', () => {
  const vote: VoteResult = { bpm: 72, spread: 1, agreement: 1, decision: 'ok', reason: null };
  const d = decideWindow({ vote, snrSpecRaw: 4, snrHarmRaw: 0.5, continuity: 0, aeDrift: cleanAe() });
  assert.equal(d.rejected, true);
  assert.equal(d.reason, 'low_snr');
  assert.equal(d.bpm, null);
});
