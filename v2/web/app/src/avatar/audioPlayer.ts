/**
 * 下行音频播放器（方案 §2.4 / §6.5）。
 *
 * 三条硬要求：
 * 1. **统一时钟**：`now()` 返回 `AudioContext.currentTime`（音频时钟，ms），
 *    形象渲染与口型对齐都以它为基准——不用 `Date.now()`，否则缓冲抖动会累积成口型漂移。
 * 2. **抖动缓冲**：首块延后 `jitterMs`（方案要求 200–400ms）再起播，后续块紧接排队。
 * 3. **音频不可用必须退化**：拿不到 AudioContext 时 `textOnly=true`，入队变空操作、
 *    `now()` 退回注入的墙钟，**流程绝不因无音频而卡住**（对应"形态不可用即纯文本呈现"）。
 *
 * 说明：当前按 **16-bit 单声道 PCM** 解码（真实 TTS 音频格式确定前的最小实现），
 * 未接入真实 TTS 时不会伪造播放——`textOnly` 会如实置位。
 */

export interface AudioChunk {
  /** 原始 PCM 字节（16-bit little-endian 单声道）。 */
  data: Uint8Array;
  sampleRate?: number;
}

export interface AudioPlayerOptions {
  /** 首块抖动缓冲（ms），默认 300（方案区间 200–400）。 */
  jitterMs?: number;
  /** 默认采样率。 */
  sampleRate?: number;
  /** 无 AudioContext 时 `now()` 使用的墙钟（ms）。 */
  clock?: () => number;
  /** 注入 AudioContext 工厂；返回 null 表示环境不支持音频。 */
  audioContextFactory?: () => AudioContext | null;
}

function defaultAudioContextFactory(): AudioContext | null {
  if (typeof globalThis.AudioContext === "undefined") return null;
  return new globalThis.AudioContext();
}

export class AudioPlayer {
  readonly jitterMs: number;
  /** true = 无音频，UI 应走纯文本呈现。 */
  readonly textOnly: boolean;

  private readonly ctx: AudioContext | null;
  private readonly sampleRate: number;
  private readonly fallbackClock: () => number;
  /** 下一块的起播时间（音频时钟秒）；0 表示尚未排过队。 */
  private nextStartSec = 0;

  constructor(options: AudioPlayerOptions = {}) {
    this.jitterMs = options.jitterMs ?? 300;
    this.sampleRate = options.sampleRate ?? 24000;
    this.fallbackClock = options.clock ?? (() => Date.now());
    const factory = options.audioContextFactory ?? defaultAudioContextFactory;
    let ctx: AudioContext | null = null;
    try {
      ctx = factory();
    } catch {
      ctx = null;
    }
    this.ctx = ctx;
    this.textOnly = ctx === null;
    if (ctx && ctx.state === "suspended") {
      // 失败也不阻塞（浏览器策略需要用户手势）。
      void ctx.resume().catch(() => undefined);
    }
  }

  /** 统一时钟：音频时钟（ms）。无音频时退回墙钟，仍单调可用。 */
  now(): number {
    if (this.ctx) return this.ctx.currentTime * 1000;
    return this.fallbackClock();
  }

  /**
   * 入队一块音频并排程播放。
   * @returns true = 已排程；false = 处于纯文本降级（未播放，也未报错）。
   */
  enqueue(chunk: AudioChunk): boolean {
    const ctx = this.ctx;
    if (!ctx) return false;
    const samples = pcm16ToFloat32(chunk.data);
    if (samples.length === 0) return true;
    const rate = chunk.sampleRate ?? this.sampleRate;
    const buffer = ctx.createBuffer(1, samples.length, rate);
    buffer.getChannelData(0).set(samples);
    const source = ctx.createBufferSource();
    source.buffer = buffer;
    source.connect(ctx.destination);
    const earliest = ctx.currentTime + this.jitterMs / 1000;
    const startAt = Math.max(earliest, this.nextStartSec);
    source.start(startAt);
    this.nextStartSec = startAt + buffer.duration;
    return true;
  }

  /** 释放音频资源（幂等）。 */
  dispose(): void {
    if (this.ctx) void this.ctx.close().catch(() => undefined);
  }
}

/** 16-bit little-endian 单声道 PCM → [-1,1] 浮点。 */
export function pcm16ToFloat32(bytes: Uint8Array): Float32Array {
  const n = Math.floor(bytes.byteLength / 2);
  const out = new Float32Array(n);
  const view = new DataView(bytes.buffer, bytes.byteOffset, n * 2);
  for (let i = 0; i < n; i += 1) {
    out[i] = view.getInt16(i * 2, true) / 32768;
  }
  return out;
}
