// pipeline.test.ts — 端到端滑动窗口管线测试（无摄像头，全合成信号）
//
// 运行：node --experimental-strip-types --test "web/ppg/*.test.ts"

import { test } from 'node:test';
import assert from 'node:assert/strict';

import { PpgPipeline, runPipeline } from './pipeline.ts';
import { synthPulse } from './synth.ts';
import type { PpgWindowResult } from './pipeline.ts';

function medianOf(nums: number[]): number {
  const a = nums.slice().sort((x, y) => x - y);
  const m = a.length >> 1;
  return a.length % 2 ? a[m] : 0.5 * (a[m - 1] + a[m]);
}

function nonNullBpms(ws: readonly PpgWindowResult[]): number[] {
  return ws.map((w) => w.bpm).filter((b): b is number => b !== null);
}

// ---------------------------------------------------------------------------
// 1. 纯净正弦：已知 BPM 的合成脉搏，误差 ≤1 BPM
// ---------------------------------------------------------------------------
test('纯净正弦：60/72/90 BPM 估计误差 ≤1 BPM（防基本估计链路错误）', () => {
  for (const bpm of [60, 72, 90]) {
    const frames = synthPulse({ bpm, seconds: 20, noise: 0 });
    const ws = runPipeline(frames);
    const bpms = nonNullBpms(ws);
    assert.ok(bpms.length >= 5, `${bpm}: 有效窗口数=${bpms.length}`);
    for (const b of bpms) {
      assert.ok(Math.abs(b - bpm) <= 1, `${bpm} BPM 档：估计 ${b.toFixed(2)}，误差 ${Math.abs(b - bpm).toFixed(2)}`);
    }
    // 三算法极差应为 0（纯信号三法一致）
    for (const w of ws) assert.ok(w.algo_spread <= 0.5, `spread=${w.algo_spread}`);
  }
});

// ---------------------------------------------------------------------------
// 2. 含噪：SNR 随噪声增大单调下降；高噪声档弃权
// ---------------------------------------------------------------------------
test('含噪：SNR 随噪声增大单调下降（防 SNR 指标失去分辨力）', () => {
  const levels = [0, 0.5, 1.5, 4, 8, 15];
  const meds: number[] = [];
  for (const noise of levels) {
    const ws = runPipeline(synthPulse({ bpm: 72, seconds: 20, noise }));
    meds.push(medianOf(ws.map((w) => w.snr)));
  }
  for (let i = 1; i < meds.length; i++) {
    assert.ok(
      meds[i] < meds[i - 1],
      `噪声 ${levels[i]} 的 SNR 中位数 ${meds[i].toFixed(4)} 未低于上一档 ${meds[i - 1].toFixed(4)}`,
    );
  }
});

test('含噪：高噪声档触发弃权，bpm 全为 null（低 SNR 绝不出数）', () => {
  const ws = runPipeline(synthPulse({ bpm: 72, seconds: 20, noise: 15 }));
  assert.ok(ws.length > 0);
  for (const w of ws) {
    assert.equal(w.bpm, null, `高噪声窗口仍给出 bpm=${w.bpm}`);
    assert.equal(w.rejected, true);
  }
});

// ---------------------------------------------------------------------------
// 3. 运动伪影：硬断言——要么对，要么弃权
// ---------------------------------------------------------------------------
test('运动伪影（低频大幅度 / 带内单通道）：绝不输出远离真值的错误 BPM', () => {
  const TRUE = 72;
  const cases = [
    { name: '低频小幅度(0.25Hz, amp5)', o: { bpm: TRUE, seconds: 20, noise: 0, motion: { freqHz: 0.25, ampR: 5, ampG: 5, ampB: 5 } } },
    { name: '低频大幅度(0.25Hz, amp40)', o: { bpm: TRUE, seconds: 20, noise: 0, motion: { freqHz: 0.25, ampR: 40, ampG: 40, ampB: 40 } } },
    { name: '带内混合(1.5Hz, 通道不一致)', o: { bpm: TRUE, seconds: 20, noise: 0.5, motion: { freqHz: 1.5, ampR: 12, ampG: 2, ampB: 8 } } },
  ];
  for (const c of cases) {
    const ws = runPipeline(synthPulse(c.o));
    for (const w of ws) {
      // 硬断言：|输出−真值| ≤5 BPM，或输出 null（弃权）
      assert.ok(
        w.bpm === null || Math.abs(w.bpm - TRUE) <= 5,
        `${c.name} 窗口输出错误心率 bpm=${w.bpm}（真值 ${TRUE}）`,
      );
    }
    // 至少不能整段沉默（低频运动是可被带通抑制的，应能恢复）
    if (c.name.startsWith('低频小')) {
      const bpms = nonNullBpms(ws);
      assert.ok(bpms.some((b) => Math.abs(b - TRUE) <= 5), `${c.name} 应能恢复心率`);
    }
  }
});

// ---------------------------------------------------------------------------
// 4. 谐波：主频 60 但二次谐波更强 → 仍得 60 而非 120（端到端）
// ---------------------------------------------------------------------------
test('谐波端到端：二次谐波更强仍输出 60 BPM 而非 120（防谐波误判）', () => {
  const ws = runPipeline(synthPulse({ bpm: 60, seconds: 20, noise: 0, harmonics: [{ mult: 2, amp: 1.5 }] }));
  const bpms = nonNullBpms(ws);
  assert.ok(bpms.length > 0);
  for (const b of bpms) assert.ok(Math.abs(b - 60) <= 1, `估计 ${b.toFixed(2)} 而非 60`);
});

// ---------------------------------------------------------------------------
// 5. 通带边界：40 BPM（低于带）与 260 BPM（高于带）弃权；200 BPM 带内有效
// ---------------------------------------------------------------------------
test('通带边界：40 BPM 与 260 BPM 弃权，200 BPM（带内，文档上限 240）有效', () => {
  const low = runPipeline(synthPulse({ bpm: 40, seconds: 20, noise: 0 }));
  const high = runPipeline(synthPulse({ bpm: 260, seconds: 20, noise: 0 }));
  for (const w of low) assert.equal(w.bpm, null, `40 BPM 窗口给出 ${w.bpm}`);
  for (const w of high) assert.equal(w.bpm, null, `260 BPM 窗口给出 ${w.bpm}`);
  const mid = runPipeline(synthPulse({ bpm: 200, seconds: 20, noise: 0 }));
  for (const w of mid) assert.ok(w.bpm !== null && Math.abs(w.bpm - 200) <= 2, `200 BPM 估计 ${w.bpm}`);
});

// ---------------------------------------------------------------------------
// 6. AE 漂移：亮度阶跃 → 该窗口被标记不可信
// ---------------------------------------------------------------------------
test('AE 漂移：亮度阶跃后窗口 rejected 且 aeDrift=true（防 AE 污染窗口出数）', () => {
  const ws = runPipeline(synthPulse({ bpm: 72, seconds: 20, noise: 0, brightnessStep: { atSec: 13, factor: 1.2 } }));
  const after = ws.filter((w) => w.end_ms >= 14000);
  assert.ok(after.length > 0, '应存在阶跃后的窗口');
  for (const w of after) {
    assert.equal(w.aeDrift, true, `窗口 end=${w.end_ms} 未标记 AE 漂移`);
    assert.equal(w.rejected, true);
    assert.equal(w.bpm, null);
  }
  // 阶跃前的窗口应正常
  const before = ws.filter((w) => w.end_ms <= 12000);
  assert.ok(before.every((w) => w.bpm !== null), '阶跃前窗口不应被误伤');
});

// ---------------------------------------------------------------------------
// 7. 三算法分歧 → 弃权（端到端，由通道不一致的运动诱发）
// ---------------------------------------------------------------------------
test('三算法分歧端到端：分歧超阈值的窗口 bpm=null（防分歧时仍输出数值）', () => {
  const ws = runPipeline(
    synthPulse({ bpm: 72, seconds: 20, noise: 0.5, motion: { freqHz: 1.5, ampR: 12, ampG: 2, ampB: 8 } }),
  );
  const disagree = ws.filter((w) => w.algo_spread > 8);
  assert.ok(disagree.length > 0, '应存在分歧>8 的窗口');
  for (const w of disagree) {
    assert.equal(w.bpm, null, `spread=${w.algo_spread} 却给出 bpm=${w.bpm}`);
    assert.equal(w.rejected, true);
  }
});

// ---------------------------------------------------------------------------
// 8. 确定性：同输入跑两次逐值相等
// ---------------------------------------------------------------------------
test('确定性：同一输入两次管线输出逐值相等（防随机数/不确定状态）', () => {
  const frames = synthPulse({ bpm: 78, seconds: 20, noise: 2.0, seed: 7 });
  const a = runPipeline(frames);
  const b = runPipeline(frames);
  assert.equal(JSON.stringify(a), JSON.stringify(b));
});

// ---------------------------------------------------------------------------
// 9. 窗口流完整性：数量、时间戳间隔、事件打点归属
// ---------------------------------------------------------------------------
test('窗口流完整性：900 帧/8s 窗/2s 步长 → 12 窗口、间隔 2000ms、事件落入正确窗口', () => {
  const frames = synthPulse({ bpm: 72, seconds: 30, noise: 0 }); // 900 帧 @30fps
  const p = new PpgPipeline({ fs: 30, windowSec: 8, stepSec: 2 });
  p.markEvent({ id: 'q1-start', type: 'start', t_ms: 5000 });
  p.markEvent({ id: 'q1-end', type: 'end', t_ms: 20000 });
  p.pushFrames(frames);
  const ws = p.process();

  // (30 - 8)/2 + 1 = 12
  assert.equal(ws.length, 12, `窗口数=${ws.length}`);
  for (let i = 1; i < ws.length; i++) {
    assert.equal(ws[i].t_ms - ws[i - 1].t_ms, 2000, `窗口 ${i} 时间戳间隔错误`);
  }
  // 帧时间戳由 round(i*1000/fs) 生成，窗口末端落在第 239 / 899 帧附近
  assert.ok(ws[0].end_ms > 7900 && ws[0].end_ms <= 8000, `首窗口 end=${ws[0].end_ms}`);
  assert.ok(ws[ws.length - 1].end_ms > 29900 && ws[ws.length - 1].end_ms <= 30000, `末窗口 end=${ws[ws.length - 1].end_ms}`);

  const ev = p.eventWindow('q1-start');
  assert.ok(ev !== null, '事件应落入某窗口');
  assert.ok(ev!.start_ms <= 5000 && 5000 <= ev!.end_ms, `事件 5000ms 不在 [${ev!.start_ms},${ev!.end_ms}]`);

  // windowsCovering 覆盖查询
  const covering = p.windowsCovering(5000);
  assert.ok(covering.length >= 1);
  assert.ok(covering.every((w) => w.start_ms <= 5000 && 5000 <= w.end_ms));

  // hrStream 结构与 §9 对齐
  const stream = p.hrStream();
  assert.equal(stream.length, 12);
  assert.ok('t_ms' in stream[0] && 'bpm' in stream[0] && 'snr' in stream[0] && 'algo_spread' in stream[0]);
});

// ---------------------------------------------------------------------------
// 10. 性能 sanity：30 秒 @30fps（900 帧）处理 < 1 秒
// ---------------------------------------------------------------------------
test('性能 sanity：900 帧端到端处理 < 1000ms（防端侧算力不可用）', () => {
  const frames = synthPulse({ bpm: 72, seconds: 30, noise: 1.0 });
  const t0 = Date.now();
  const ws = runPipeline(frames);
  const dt = Date.now() - t0;
  assert.ok(ws.length === 12, `窗口数=${ws.length}`);
  assert.ok(dt < 1000, `耗时 ${dt}ms 超过 1000ms`);
});
