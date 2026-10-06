// pipeline.ts — 有状态滑动窗口管线：ROI RGB 时序 → 窗口 BPM 流 + 事件打点
//
// 权威规范：docs/模块详设-心率与压力调节评估.md
//   §3.5 分析窗 8s / 重叠 75%   §3.8 每窗口 SNR   §4.1 事件时间轴
//
// 输入为**已在别处完成 ROI 提取**的逐帧 R/G/B 均值（本模块不做人脸/ROI 检测，
// 见 §3.2；ROI 检测由 WASM FaceMesh 层负责，此处只消费其输出）。
//
// 输出结构对应 §9 的 HRSample：{ t_ms, bpm, snr, algo_spread }，额外带 rejected 与诊断字段。

import { analyzeRgb, detrendSamplesForBpm, PPG } from './signal.ts';
import type { ChannelAnalysis } from './signal.ts';
import {
  computeContinuity,
  decideWindow,
  detectAeDrift,
  snrHarmRatio,
  snrSpecRatio,
  voteAlgorithms,
} from './quality.ts';
import type { AeDriftResult, VoteResult } from './quality.ts';

/** 逐帧 ROI 平均颜色。t_ms 为相对会话开始的毫秒时间戳。 */
export interface PpgFrame {
  t_ms: number;
  r: number;
  g: number;
  b: number;
}

/** 事件打点（§4.1）：题目开始/结束等锚点。 */
export interface PpgEvent {
  id: string;
  type: 'start' | 'end';
  t_ms: number;
}

export interface PerAlgoBpm {
  chrom: number | null;
  pos: number | null;
  ssr: number | null;
}

/** 单窗口结果（§9 HRSample 的超集）。 */
export interface PpgWindowResult {
  index: number;
  start_ms: number;
  end_ms: number;
  /** 窗口时间戳（取窗口结束时刻，即该估计可用时刻）。 */
  t_ms: number;
  bpm: number | null;
  snr: number;
  /** 三算法极差（BPM），供诊断。 */
  algo_spread: number;
  agreement: number;
  rejected: boolean;
  reason: string | null;
  perAlgo: PerAlgoBpm;
  aeDrift: boolean;
}

export interface PpgPipelineOptions {
  /** 采样率（帧率），默认 30（文档 §3.1 目标 30fps）。 */
  fs?: number;
  /** 分析窗长度（秒），默认 8（文档 §3.8）。 */
  windowSec?: number;
  /** 滑动步长（秒），默认 2（文档 §3.5 重叠 75% / §9 每 2s 一条）。 */
  stepSec?: number;
  /** 连续性历史长度（窗口数），默认 5。 */
  continuityHistory?: number;
  /** Welch 分段长度（样本），默认 = 分析窗长度。 */
  welchSegmentLength?: number;
}

/**
 * 滑动窗口管线。
 * - 每接收一帧即尝试推进窗口；窗口结束索引按 step 递增，保证时间戳间隔 = step。
 * - 连续性使用"投票后心率"序列（即使最终因低 SNR 被弃权，仍保留原始投票值用于时间平滑度量）。
 * - 不做任何跨帧插值；缺失即缺失（§7 运动段处理原则）。
 */
export class PpgPipeline {
  readonly fs: number;
  readonly windowSec: number;
  readonly stepSec: number;
  readonly continuityHistory: number;
  readonly welchSegmentLength: number | undefined;
  readonly framesPerWindow: number;
  readonly stepFrames: number;

  private frames: PpgFrame[] = [];
  private endCursor = -1;
  private out: PpgWindowResult[] = [];
  private events: PpgEvent[] = [];
  private continuityBuf: (number | null)[] = [];

  constructor(opts?: PpgPipelineOptions) {
    this.fs = opts?.fs ?? 30;
    this.windowSec = opts?.windowSec ?? 8;
    this.stepSec = opts?.stepSec ?? 2;
    this.continuityHistory = opts?.continuityHistory ?? 5;
    this.welchSegmentLength = opts?.welchSegmentLength;
    this.framesPerWindow = Math.max(4, Math.round(this.windowSec * this.fs));
    this.stepFrames = Math.max(1, Math.round(this.stepSec * this.fs));
  }

  /** 推入一帧 ROI 平均颜色，并推进所有已就绪的窗口。 */
  pushFrame(frame: PpgFrame): void {
    this.frames.push(frame);
    this.computeDue();
  }

  pushFrames(frames: readonly PpgFrame[]): void {
    for (const f of frames) this.pushFrame(f);
  }

  /**
   * 事件打点（§4.1 的 t_q / t_e）。事件仅记录时间，不改动信号链路。
   */
  markEvent(ev: PpgEvent): void {
    this.events.push(ev);
  }

  /** 已计算出的窗口结果（按时间升序）。 */
  getWindows(): readonly PpgWindowResult[] {
    return this.out;
  }

  /** 以 {t_ms, bpm, snr, algo_spread, rejected} 形式返回 HR 流（§9）。 */
  hrStream(): { t_ms: number; bpm: number | null; snr: number; algo_spread: number; rejected: boolean }[] {
    return this.out.map((w) => ({
      t_ms: w.t_ms,
      bpm: w.bpm,
      snr: w.snr,
      algo_spread: w.algo_spread,
      rejected: w.rejected,
    }));
  }

  getEvents(): readonly PpgEvent[] {
    return this.events;
  }

  /** 覆盖某时间点的全部窗口（事件到窗口的归属查询）。 */
  windowsCovering(t_ms: number): PpgWindowResult[] {
    return this.out.filter((w) => t_ms >= w.start_ms && t_ms <= w.end_ms);
  }

  /** 某事件所落入的窗口（取第一个覆盖它的窗口）。 */
  eventWindow(id: string): PpgWindowResult | null {
    const ev = this.events.find((e) => e.id === id);
    if (!ev) return null;
    const covering = this.windowsCovering(ev.t_ms);
    return covering.length > 0 ? covering[0] : null;
  }

  /** 强制推进（pushFrame 已自动推进；此方法用于语义清晰的收尾）。 */
  process(): readonly PpgWindowResult[] {
    this.computeDue();
    return this.out;
  }

  reset(): void {
    this.frames = [];
    this.endCursor = -1;
    this.out = [];
    this.events = [];
    this.continuityBuf = [];
  }

  // ---- 内部 ----

  private computeDue(): void {
    const W = this.framesPerWindow;
    if (this.frames.length < W) return;
    if (this.endCursor < 0) this.endCursor = W - 1;
    while (this.endCursor <= this.frames.length - 1) {
      this.computeWindow(this.endCursor);
      this.endCursor += this.stepFrames;
    }
  }

  private computeWindow(endIdx: number): void {
    const startIdx = endIdx - this.framesPerWindow + 1;
    const n = this.framesPerWindow;
    const r = new Float64Array(n);
    const g = new Float64Array(n);
    const b = new Float64Array(n);
    const brightness = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const f = this.frames[startIdx + i];
      r[i] = f.r;
      g[i] = f.g;
      b[i] = f.b;
      brightness[i] = (f.r + f.g + f.b) / 3;
    }

    // 去趋势窗口长度由上一个有效心率推导（§3.4：1.6×周期）
    const prevBpm = this.lastVotedBpm();
    const detrendSamples = detrendSamplesForBpm(prevBpm ?? PPG.DEFAULT_BPM, this.fs);

    const analyses = analyzeRgb({ r, g, b }, this.fs, {
      detrendSamples,
      welchSegmentLength: this.welchSegmentLength,
    });

    const perAlgo: PerAlgoBpm = {
      chrom: analyses.chrom.bpm,
      pos: analyses.pos.bpm,
      ssr: analyses.ssr.bpm,
    };
    const vote: VoteResult = voteAlgorithms([perAlgo.chrom, perAlgo.pos, perAlgo.ssr]);

    // 选取参考分析用于 SNR：优先与投票结果最接近的算法
    const ref = this.pickReference(analyses, vote);

    let snrSpecRaw = 0;
    let snrHarmRaw: number | null = null;
    if (ref && ref.estimate) {
      snrSpecRaw = snrSpecRatio(ref.freqs, ref.psd, ref.estimate.f0Hz);
      snrHarmRaw = snrHarmRatio(ref.freqs, ref.psd, ref.estimate.f0Hz);
    }

    this.continuityBuf.push(vote.bpm);
    while (this.continuityBuf.length > this.continuityHistory) this.continuityBuf.shift();
    const continuity = computeContinuity(this.continuityBuf);

    const aeDrift: AeDriftResult = detectAeDrift(brightness);

    const decision = decideWindow({ vote, snrSpecRaw, snrHarmRaw, continuity, aeDrift });

    const start_ms = this.frames[startIdx].t_ms;
    const end_ms = this.frames[endIdx].t_ms;
    this.out.push({
      index: this.out.length,
      start_ms,
      end_ms,
      t_ms: end_ms,
      bpm: decision.bpm,
      snr: decision.snr,
      algo_spread: vote.spread,
      agreement: vote.agreement,
      rejected: decision.rejected,
      reason: decision.reason,
      perAlgo,
      aeDrift: aeDrift.drifted,
    });
  }

  private lastVotedBpm(): number | null {
    for (let i = this.continuityBuf.length - 1; i >= 0; i--) {
      const v = this.continuityBuf[i];
      if (v !== null && Number.isFinite(v)) return v;
    }
    return null;
  }

  private pickReference(
    analyses: { chrom: ChannelAnalysis; pos: ChannelAnalysis; ssr: ChannelAnalysis },
    vote: VoteResult,
  ): ChannelAnalysis | null {
    const all = [analyses.chrom, analyses.pos, analyses.ssr];
    const withEst = all.filter((a) => a.estimate !== null);
    if (withEst.length === 0) return null;
    if (vote.bpm !== null) {
      let best = withEst[0];
      let bestD = Number.POSITIVE_INFINITY;
      for (const a of withEst) {
        const d = Math.abs((a.bpm ?? Number.POSITIVE_INFINITY) - vote.bpm);
        if (d < bestD) {
          bestD = d;
          best = a;
        }
      }
      return best;
    }
    // 投票弃权时，取峰值显著性最高者用于诊断性 SNR
    let best = withEst[0];
    for (const a of withEst) {
      if ((a.estimate?.peakDb ?? -Infinity) > (best.estimate?.peakDb ?? -Infinity)) best = a;
    }
    return best;
  }
}

/** 便捷函数：一次性处理一段 RGB 帧序列，返回窗口结果。 */
export function runPipeline(frames: readonly PpgFrame[], opts?: PpgPipelineOptions): PpgWindowResult[] {
  const p = new PpgPipeline(opts);
  p.pushFrames(frames);
  return Array.from(p.process());
}
