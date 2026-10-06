/**
 * 单条 WebSocket 客户端（方案 §2.4）。
 *
 * 职责：一条连接上多路复用两类帧 + 传输层横切关注点：
 *   1. **文本帧**：JSON 信封（`codec.decodeText`）→ `EnvelopeEvent`。
 *   2. **二进制帧**：首字节 opcode（`codec.decodeBinary`）→ TTS 音频块（0x11）/
 *      viseme 时间轴（0x12）。与 Python `protocol.py` 的 `[opcode][4B 大端长度][payload]`
 *      完全一致。
 *   3. **信用窗口背压**：媒体上行占用信用，`media.ack` 到达后释放；窗口满则**暂停**发送
 *      并排队，等 ack 再续（对应后端 `Gateway` 的 CreditWindow）。
 *   4. **自动重连 + 事件重放**：非主动关闭时按指数退避重连；重连成功后调用
 *      `onReconnect`，由上层发送带 `resume` 的 `session.create` 请求服务端重放；
 *      重放事件到达后由 store 的 seq 去重保证幂等。
 *   5. **心跳（看门狗）**：周期性检查"最近一次收到帧"的时间，超时判死并强制重连。
 *      说明：本客户端不发明协议里没有的 ping 类型——保持与 Python 侧类型集合严格一致；
 *      需要服务端级 keep-alive 时可注入 `keepAlive`。
 *
 * 所有 IO/时钟/定时器/WebSocket 均**注入**，因此可在无浏览器的 node 环境下确定性单测。
 */

import { CreditWindow } from "./creditWindow";
import {
  decodeBinary,
  decodeText,
  encodeBinary,
  encodeText,
  type Envelope,
} from "../protocol/codec";
import {
  DEFAULT_ACK_EVERY,
  DEFAULT_CREDIT_CAPACITY,
  HEARTBEAT_INTERVAL_MS,
  Opcode,
  PROTOCOL_VERSION,
  RECONNECT_BASE_MS,
  RECONNECT_FACTOR,
  RECONNECT_MAX_MS,
  ServerType,
} from "../protocol/constants";
import type {
  EnvelopeEvent,
  ServerEvent,
  TtsChunkEvent,
  VisemeEvent,
  VisemeTimelineEvent,
} from "../store/types";

// ------------------------------------------------------------------ 可注入接口

/** 浏览器 `WebSocket` 的最小可用子集。 */
export interface WebSocketLike {
  binaryType: string;
  readyState: number;
  send(data: string | ArrayBufferLike | ArrayBufferView): void;
  close(code?: number, reason?: string): void;
  onopen: (() => void) | null;
  onmessage: ((ev: { data: unknown }) => void) | null;
  onclose: ((ev: { code: number; reason: string }) => void) | null;
  onerror: ((ev: unknown) => void) | null;
}

export interface TimerApi {
  setTimeout: (fn: () => void, ms: number) => number;
  clearTimeout: (id: number) => void;
  setInterval: (fn: () => void, ms: number) => number;
  clearInterval: (id: number) => void;
  now: () => number;
}

export type SocketStatus = "idle" | "connecting" | "open" | "reconnecting" | "closed";

export interface SocketClientOptions {
  url: string;
  sessionId?: string;
  socketFactory: (url: string) => WebSocketLike;
  timers?: TimerApi;
  /** 毫秒时钟（信封 ts 用）。默认取 `timers.now`。 */
  clock?: () => number;
  creditCapacity?: number;
  ackEvery?: number;
  /** 心跳巡检间隔（ms）；0 关闭。 */
  heartbeatMs?: number;
  /** 超过该时长未收到任何帧即判连接已死（ms）。默认 2×心跳间隔。 */
  heartbeatTimeoutMs?: number;
  reconnectBaseMs?: number;
  reconnectMaxMs?: number;
  reconnectFactor?: number;
  autoReconnect?: boolean;
  /** 媒体发送队列上限，超出丢最旧帧并报错。 */
  maxMediaQueue?: number;
  onEvent: (ev: ServerEvent) => void;
  onStatus?: (status: SocketStatus) => void;
  /**
   * 下行 TTS 音频原始字节（opcode 0x11）。`ServerEvent` 只记元信息（长度），
   * 原始字节不含在事件里，需真实播放时走这个回调交给 `AudioPlayer`。
   */
  onAudioChunk?: (bytes: Uint8Array) => void;
  /** 断线后重新建立连接时触发（区别于首次连接）。 */
  onReconnect?: () => void;
  onError?: (err: unknown) => void;
}

interface QueuedMedia {
  opcode: number;
  payload: Uint8Array;
}

// ------------------------------------------------------------------ 默认实现

const defaultTimers: TimerApi = {
  setTimeout: (fn, ms) => globalThis.setTimeout(fn, ms) as unknown as number,
  clearTimeout: (id) => globalThis.clearTimeout(id),
  setInterval: (fn, ms) => globalThis.setInterval(fn, ms) as unknown as number,
  clearInterval: (id) => globalThis.clearInterval(id),
  now: () => Date.now(),
};

// ------------------------------------------------------------------ 客户端

export class SocketClient {
  readonly credit: CreditWindow;

  private readonly opts: Required<
    Pick<
      SocketClientOptions,
      | "ackEvery"
      | "autoReconnect"
      | "creditCapacity"
      | "heartbeatMs"
      | "heartbeatTimeoutMs"
      | "maxMediaQueue"
      | "reconnectBaseMs"
      | "reconnectFactor"
      | "reconnectMaxMs"
    >
  > &
    SocketClientOptions;

  private readonly timers: TimerApi;
  private readonly clock: () => number;
  private socket: WebSocketLike | null = null;
  private status: SocketStatus = "idle";
  private outSeq = 0;
  private sessionId: string;
  private hasConnectedBefore = false;
  private intentionalClose = false;
  private attempt = 0;
  private reconnectTimer: number | null = null;
  private heartbeatTimer: number | null = null;
  private lastInboundAt = 0;
  /** 窗口满时暂存的媒体帧（FIFO）。 */
  private mediaQueue: QueuedMedia[] = [];
  /**
   * 连接就绪前暂存的**文本**消息（FIFO）。
   *
   * 文本和媒体不是一回事：媒体满了可以丢最旧的（背压），文本一条都不能丢 ——
   * 丢掉 `session.create` 的后果不是"少发一帧"，是整场面试根本没开始。
   */
  private textQueue: Envelope[] = [];

  constructor(options: SocketClientOptions) {
    this.opts = {
      reconnectBaseMs: RECONNECT_BASE_MS,
      reconnectMaxMs: RECONNECT_MAX_MS,
      reconnectFactor: RECONNECT_FACTOR,
      creditCapacity: DEFAULT_CREDIT_CAPACITY,
      ackEvery: DEFAULT_ACK_EVERY,
      heartbeatMs: HEARTBEAT_INTERVAL_MS,
      heartbeatTimeoutMs: (options.heartbeatMs ?? HEARTBEAT_INTERVAL_MS) * 2,
      autoReconnect: true,
      maxMediaQueue: 64,
      ...options,
    };
    this.timers = options.timers ?? defaultTimers;
    // 包一层箭头函数而不是直接取 `this.timers.now`：
    // 后者是**解绑的方法引用**，一旦传入的 `TimerApi` 是用类实现的（测试里就是），
    // 调用时 `this` 就丢了，`now()` 返回 undefined → 信封的 `ts` 变 undefined →
    // JSON 序列化把该字段整个丢掉 → 对端收到"缺 ts 字段"的坏帧。
    // 内置 `defaultTimers.now` 恰好是箭头函数，所以这个坑只在注入实现上出现。
    this.clock = options.clock ?? (() => this.timers.now());
    this.sessionId = options.sessionId ?? "";
    this.credit = new CreditWindow(this.opts.creditCapacity);
  }

  getStatus(): SocketStatus {
    return this.status;
  }

  get pendingMedia(): number {
    return this.mediaQueue.length;
  }

  /** 连接就绪前排队等发的文本消息条数。 */
  get pendingText(): number {
    return this.textQueue.length;
  }

  /** 建立连接（幂等：已连接时直接返回）。 */
  connect(): void {
    if (this.status === "open" || this.status === "connecting") return;
    this.intentionalClose = false;
    this.openSocket();
  }

  /** 主动关闭：不再重连，清理定时器。 */
  close(code = 1000, reason = "client close"): void {
    this.intentionalClose = true;
    this.clearReconnectTimer();
    this.stopHeartbeat();
    const sock = this.socket;
    this.socket = null;
    this.setStatus("closed");
    if (this.textQueue.length > 0) {
      // 主动关闭后不会再有人发它们。显式报出去 —— 悄悄清空等于假装发过了。
      const dropped = this.textQueue.length;
      this.textQueue = [];
      this.reportError(new Error(`连接已关闭，${dropped} 条未发出的文本消息被丢弃`));
    }
    if (sock) {
      try {
        sock.close(code, reason);
      } catch (err) {
        this.reportError(err);
      }
    }
  }

  // ------------------------------------------------------------ 上行

  /**
   * 发送一条文本信封（自动补 seq / ts / session_id）。
   *
   * **连接未就绪时不丢，改为入队。** 这不是"顺手加的容错"，是修一个真实事故：
   * `connect()` 内部会**同步**把状态置成 `connecting`，而调用方的写法必然是
   *
   *     socket.connect();
   *     socket.send(sessionCreate(...));   // 此刻状态还是 connecting
   *     socket.send(consentGrant(...));
   *
   * 早先这里 `return false` 把消息直接扔掉，返回值又没人检查 —— 现象是界面正常、
   * 控制台干净、服务端一条都没收到，然后永远等一个不会来的 `state.changed`。
   * 这类"静默丢弃"比抛异常难查得多。
   *
   * `seq` 在**入队那一刻**分配而不是补发时分配：否则补发顺序会覆盖调用顺序，
   * `consent.grant` 可能拿到比 `session.create` 更小的 seq。
   *
   * @returns 是否**已经写出**。入队待发返回 false（不要拿它当失败）。
   */
  send(msg: Envelope): boolean {
    const framed: Envelope = {
      v: msg.v || PROTOCOL_VERSION,
      type: msg.type,
      seq: msg.seq > 0 ? msg.seq : this.nextSeq(),
      ts: msg.ts > 0 ? msg.ts : this.clock(),
      session_id: msg.session_id || this.sessionId,
      payload: msg.payload,
    };
    const sock = this.socket;
    if (sock && this.status === "open" && sock.readyState === 1) {
      try {
        sock.send(encodeText(framed));
        return true;
      } catch (err) {
        this.reportError(err);
        return false;
      }
    }
    if (this.status === "idle" || this.status === "connecting" || this.status === "reconnecting") {
      this.textQueue.push(framed);
      return false;
    }
    // 已 closed 且不再重连：留着也没人会发，显式报出去而不是让它消失。
    this.reportError(new Error(`连接已关闭，${framed.type} 未发出`));
    return false;
  }

  /**
   * 发送一帧媒体（二进制）。占用一个信用；窗口满则**暂停**并入队，
   * 待 `media.ack` 释放信用后按序续发。
   * @returns true 表示已发出；false 表示已入队（背压）。
   */
  sendMedia(opcode: number, payload: Uint8Array): boolean {
    const sock = this.socket;
    const canWrite = sock !== null && this.status === "open" && sock.readyState === 1;
    if (!canWrite || !this.credit.tryAcquire(1)) {
      this.enqueueMedia({ opcode, payload });
      return false;
    }
    this.writeBinary(opcode, payload);
    return true;
  }

  // ------------------------------------------------------------ 内部：连接

  private openSocket(): void {
    this.setStatus(this.attempt > 0 ? "reconnecting" : "connecting");
    let sock: WebSocketLike;
    try {
      sock = this.opts.socketFactory(this.opts.url);
    } catch (err) {
      this.reportError(err);
      this.scheduleReconnect();
      return;
    }
    sock.binaryType = "arraybuffer";
    this.socket = sock;

    sock.onopen = () => {
      const wasReconnect = this.hasConnectedBefore;
      this.hasConnectedBefore = true;
      this.attempt = 0;
      this.setStatus("open");
      // 断线期间"在途未确认"的信用随连接丢失，一律归还；未发出的队列保留续发。
      this.credit.releaseAll();
      this.startHeartbeat();
      // 先补发积压的文本（它们 seq 更小、更早），再走重连回调，最后才是媒体。
      this.flushTextQueue();
      if (wasReconnect) {
        try {
          this.opts.onReconnect?.();
        } catch (err) {
          this.reportError(err);
        }
      }
      this.flushMediaQueue();
    };

    sock.onmessage = (ev) => {
      this.lastInboundAt = this.timers.now();
      this.handleIncoming(ev.data);
    };

    sock.onerror = (err) => this.reportError(err);

    sock.onclose = () => {
      this.stopHeartbeat();
      this.socket = null;
      this.setStatus("closed");
      if (!this.intentionalClose && this.opts.autoReconnect) {
        this.scheduleReconnect();
      }
    };
  }

  private scheduleReconnect(): void {
    if (this.reconnectTimer !== null) return;
    const delay = Math.min(
      this.opts.reconnectBaseMs * this.opts.reconnectFactor ** this.attempt,
      this.opts.reconnectMaxMs,
    );
    this.attempt += 1;
    this.setStatus("reconnecting");
    this.reconnectTimer = this.timers.setTimeout(() => {
      this.reconnectTimer = null;
      if (this.intentionalClose) return;
      this.openSocket();
    }, delay);
  }

  private clearReconnectTimer(): void {
    if (this.reconnectTimer !== null) {
      this.timers.clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
  }

  // ------------------------------------------------------------ 内部：下行

  private handleIncoming(data: unknown): void {
    if (typeof data === "string") {
      this.handleText(data);
      return;
    }
    if (data instanceof ArrayBuffer) {
      this.handleBinary(new Uint8Array(data));
      return;
    }
    if (ArrayBuffer.isView(data)) {
      this.handleBinary(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
      return;
    }
    if (typeof Blob !== "undefined" && data instanceof Blob) {
      data
        .arrayBuffer()
        .then((buf) => this.handleBinary(new Uint8Array(buf)))
        .catch((err) => this.reportError(err));
      return;
    }
    this.reportError(new Error(`不支持的 WS 消息类型: ${typeof data}`));
  }

  private handleText(raw: string): void {
    let env: Envelope;
    try {
      env = decodeText(raw);
    } catch (err) {
      this.reportError(err);
      return;
    }
    const event: EnvelopeEvent = {
      kind: "envelope",
      seq: env.seq,
      ts: env.ts,
      sessionId: env.session_id,
      type: env.type,
      payload: env.payload,
    };
    if (env.type === ServerType.MEDIA_ACK) {
      this.onMediaAck(env.payload);
    }
    this.opts.onEvent(event);
  }

  private handleBinary(raw: Uint8Array): void {
    let opcode: number;
    let payload: Uint8Array;
    try {
      ({ opcode, payload } = decodeBinary(raw));
    } catch (err) {
      this.reportError(err);
      return;
    }
    if (opcode === Opcode.TTS_AUDIO) {
      try {
        this.opts.onAudioChunk?.(payload);
      } catch (err) {
        this.reportError(err);
      }
      const event: TtsChunkEvent = {
        kind: "tts_chunk",
        seq: 0,
        ts: this.clock(),
        sessionId: this.sessionId,
        byteLength: payload.byteLength,
      };
      this.opts.onEvent(event);
      return;
    }
    if (opcode === Opcode.VISEME) {
      const visemes = parseVisemePayload(payload, (err) => this.reportError(err));
      const event: VisemeTimelineEvent = {
        kind: "viseme",
        seq: 0,
        ts: this.clock(),
        sessionId: this.sessionId,
        visemes,
      };
      this.opts.onEvent(event);
      return;
    }
    this.reportError(new Error(`未处理的下行 opcode: 0x${opcode.toString(16)}`));
  }

  private onMediaAck(payload: Record<string, unknown>): void {
    const frames = typeof payload["frames"] === "number" && payload["frames"] >= 1 ? payload["frames"] : 1;
    try {
      this.credit.release(frames);
    } catch (err) {
      // 重复 ack / 超量释放：显式暴露，但不炸连接。
      this.reportError(err);
      this.credit.releaseAll();
    }
    this.flushMediaQueue();
  }

  // ------------------------------------------------------------ 内部：文本队列

  /**
   * 把连接就绪前积压的文本按原顺序补发。
   *
   * 逐条走 `send()` 而不是直接 `sock.send()`：万一中途连接又断了，
   * `send()` 会把剩下的重新入队，顺序不变。直接写的话就会丢。
   */
  private flushTextQueue(): void {
    if (this.textQueue.length === 0) return;
    const queued = this.textQueue;
    this.textQueue = [];
    for (const env of queued) this.send(env);
  }

  // ------------------------------------------------------------ 内部：媒体队列

  private enqueueMedia(item: QueuedMedia): void {
    this.mediaQueue.push(item);
    while (this.mediaQueue.length > this.opts.maxMediaQueue) {
      this.mediaQueue.shift();
      this.reportError(new Error("媒体发送队列已满，丢弃最旧帧"));
    }
  }

  private flushMediaQueue(): void {
    const sock = this.socket;
    const canWrite = sock !== null && this.status === "open" && sock.readyState === 1;
    if (!canWrite) return;
    while (this.mediaQueue.length > 0) {
      if (!this.credit.tryAcquire(1)) return; // 窗口仍满：保持暂停
      const item = this.mediaQueue.shift() as QueuedMedia;
      this.writeBinary(item.opcode, item.payload);
    }
  }

  private writeBinary(opcode: number, payload: Uint8Array): void {
    const sock = this.socket;
    if (!sock) return;
    try {
      sock.send(encodeBinary(opcode, payload));
    } catch (err) {
      this.reportError(err);
    }
  }

  // ------------------------------------------------------------ 内部：心跳

  private startHeartbeat(): void {
    this.stopHeartbeat();
    this.lastInboundAt = this.timers.now();
    if (this.opts.heartbeatMs <= 0) return;
    this.heartbeatTimer = this.timers.setInterval(() => {
      const idle = this.timers.now() - this.lastInboundAt;
      if (idle > this.opts.heartbeatTimeoutMs) {
        // 判死：主动断开以触发重连。
        const sock = this.socket;
        this.socket = null;
        this.stopHeartbeat();
        this.reportError(new Error(`心跳超时：${idle}ms 未收到服务端帧，强制重连`));
        if (sock) {
          // close() 会触发 onclose，由它负责置 state 与安排重连（避免这里再覆写状态）。
          try {
            sock.close(4000, "heartbeat timeout");
          } catch (err) {
            this.reportError(err);
          }
        } else {
          this.setStatus("closed");
          if (!this.intentionalClose && this.opts.autoReconnect) this.scheduleReconnect();
        }
      }
    }, this.opts.heartbeatMs);
  }

  private stopHeartbeat(): void {
    if (this.heartbeatTimer !== null) {
      this.timers.clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
  }

  // ------------------------------------------------------------ 内部：杂项

  private nextSeq(): number {
    this.outSeq += 1;
    return this.outSeq;
  }

  private setStatus(status: SocketStatus): void {
    if (this.status === status) return;
    this.status = status;
    try {
      this.opts.onStatus?.(status);
    } catch (err) {
      this.reportError(err);
    }
  }

  private reportError(err: unknown): void {
    try {
      this.opts.onError?.(err);
    } catch {
      // onError 自身抛错时不再递归：静默到控制台。
    }
  }
}

// ------------------------------------------------------------------ 助手

/** viseme 二进制帧 payload 约定：UTF-8 JSON，形如 `{"visemes":[{t_ms,viseme,weight}]}` 或裸数组。 */
export function parseVisemePayload(
  payload: Uint8Array,
  onError?: (err: unknown) => void,
): VisemeEvent[] {
  const text = new TextDecoder().decode(payload);
  try {
    const data: unknown = JSON.parse(text);
    const list = Array.isArray(data)
      ? data
      : data !== null && typeof data === "object"
        ? ((data as Record<string, unknown>)["visemes"] ?? (data as Record<string, unknown>)["timeline"])
        : null;
    if (!Array.isArray(list)) return [];
    return list.map((v) => normalizeViseme(v));
  } catch (err) {
    onError?.(err);
    return [];
  }
}

function normalizeViseme(v: unknown): VisemeEvent {
  if (v !== null && typeof v === "object" && !Array.isArray(v)) {
    const o = v as Record<string, unknown>;
    const t = typeof o["t_ms"] === "number" ? o["t_ms"] : 0;
    const viseme = typeof o["viseme"] === "string" ? o["viseme"] : "rest";
    const weight = typeof o["weight"] === "number" ? o["weight"] : 1;
    return { t_ms: t, viseme, weight };
  }
  return { t_ms: 0, viseme: "rest", weight: 1 };
}
