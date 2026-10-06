/**
 * WS 客户端测试：帧解码、信用窗口背压、断线重连 + 事件重放、心跳看门狗。
 *
 * 全部 IO/时钟/定时器用假实现注入，无需真实网络或浏览器。
 */

import { beforeEach, describe, expect, it } from "vitest";
import { SocketClient, type TimerApi, type WebSocketLike } from "./socket";
import { decodeText, encodeBinary, encodeText, makeEnvelope } from "../protocol/codec";
import { Opcode, ServerType } from "../protocol/constants";
import { controlEnd, sessionCreate } from "../protocol/messages";
import { useInterviewStore } from "../store/interviewStore";
import type { ServerEvent } from "../store/types";

// ------------------------------------------------------------------ 假件

class FakeWS implements WebSocketLike {
  static instances: FakeWS[] = [];
  binaryType = "";
  readyState = 0;
  sent: unknown[] = [];
  onopen: (() => void) | null = null;
  onmessage: ((ev: { data: unknown }) => void) | null = null;
  onclose: ((ev: { code: number; reason: string }) => void) | null = null;
  onerror: ((ev: unknown) => void) | null = null;

  constructor(public url: string) {
    FakeWS.instances.push(this);
  }

  send(data: unknown): void {
    this.sent.push(data);
  }

  close(): void {
    this.readyState = 3;
    this.onclose?.({ code: 1000, reason: "" });
  }

  open(): void {
    this.readyState = 1;
    this.onopen?.();
  }

  deliver(data: unknown): void {
    this.onmessage?.({ data });
  }

  drop(): void {
    this.readyState = 3;
    this.onclose?.({ code: 1006, reason: "network" });
  }

  binarySends(): number {
    return this.sent.filter((s) => s instanceof Uint8Array).length;
  }
}

class FakeTimers implements TimerApi {
  private nextId = 1;
  private t = 0;
  private timeouts = new Map<number, { at: number; fn: () => void }>();
  private intervals = new Map<number, { at: number; every: number; fn: () => void }>();

  setTimeout(fn: () => void, ms: number): number {
    const id = this.nextId++;
    this.timeouts.set(id, { at: this.t + ms, fn });
    return id;
  }

  clearTimeout(id: number): void {
    this.timeouts.delete(id);
  }

  setInterval(fn: () => void, ms: number): number {
    const id = this.nextId++;
    this.intervals.set(id, { at: this.t + ms, every: ms, fn });
    return id;
  }

  clearInterval(id: number): void {
    this.intervals.delete(id);
  }

  now(): number {
    return this.t;
  }

  advance(ms: number): void {
    const target = this.t + ms;
    for (;;) {
      let nextAt = Infinity;
      let kind: "t" | "i" | null = null;
      let nextId = -1;
      for (const [id, e] of this.timeouts) {
        if (e.at <= target && e.at < nextAt) {
          nextAt = e.at;
          kind = "t";
          nextId = id;
        }
      }
      for (const [id, e] of this.intervals) {
        if (e.at <= target && e.at < nextAt) {
          nextAt = e.at;
          kind = "i";
          nextId = id;
        }
      }
      if (kind === null || nextId < 0) break;
      this.t = nextAt;
      if (kind === "t") {
        const e = this.timeouts.get(nextId);
        this.timeouts.delete(nextId);
        e?.fn();
      } else {
        const e = this.intervals.get(nextId);
        if (e) {
          e.at = this.t + e.every;
          e.fn();
        }
      }
    }
    this.t = target;
  }
}

// ------------------------------------------------------------------ 助手

function textFrame(type: string, payload: Record<string, unknown>, seq: number): string {
  return encodeText(makeEnvelope(type, payload, { seq, sessionId: "s-1" }));
}

function b64ToBytes(opcode: number, body: Uint8Array): ArrayBuffer {
  const framed = encodeBinary(opcode, body);
  return framed.buffer.slice(framed.byteOffset, framed.byteOffset + framed.byteLength) as ArrayBuffer;
}

beforeEach(() => {
  FakeWS.instances = [];
  useInterviewStore.getState().reset();
});

// ------------------------------------------------------------------ 文本发件箱

/**
 * `connect()` 内部会**同步**把状态置成 `connecting`，所以"连上之后马上发"这个
 * 写法必然撞上"还没 open"。下面钉住的是：那一刻消息必须入队，不能丢。
 *
 * 这一组对应一个真实事故：`App.tsx` 里 `connect()` 紧接着两次 `send()`，
 * 而 `send()` 当时是 `return false` 直接扔掉的 —— 服务端一条都没收到，
 * 界面却一切正常。
 */
describe("连接就绪前的文本消息：入队而不是丢弃", () => {
  it("connect() 之后立刻 send()：入队，open 时按原顺序补发且 seq 保持", () => {
    const client = new SocketClient({
      url: "ws://x/ws",
      socketFactory: (u) => new FakeWS(u),
      timers: new FakeTimers(),
      heartbeatMs: 0,
      onEvent: () => {},
    });

    client.connect();
    // 这就是 App.tsx 的写法：connect() 紧跟着 send()
    expect(client.send(sessionCreate({}))).toBe(false);
    expect(client.send(controlEnd())).toBe(false);

    expect(client.pendingText).toBe(2);
    expect(FakeWS.instances[0].sent).toEqual([]);

    FakeWS.instances[0].open();

    const out = FakeWS.instances[0].sent.map((s) => decodeText(s as string));
    expect(out.map((e) => e.type)).toEqual(["session.create", "control.end"]);
    // seq 在入队那一刻就定好，补发不会重排
    expect(out.map((e) => e.seq)).toEqual([1, 2]);
    expect(client.pendingText).toBe(0);
  });

  it("已 open 时 send() 立即写出，不进队列", () => {
    const client = new SocketClient({
      url: "ws://x/ws",
      socketFactory: (u) => new FakeWS(u),
      timers: new FakeTimers(),
      heartbeatMs: 0,
      onEvent: () => {},
    });
    client.connect();
    const ws = FakeWS.instances[0];
    ws.open();

    expect(client.send(sessionCreate({}))).toBe(true);
    expect(client.pendingText).toBe(0);
    expect(ws.sent).toHaveLength(1);
  });

  it("断线重连：积压的文本补发到新连接上，已发出的不重发", () => {
    const timers = new FakeTimers();
    const client = new SocketClient({
      url: "ws://x/ws",
      socketFactory: (u) => new FakeWS(u),
      timers,
      heartbeatMs: 0,
      onEvent: () => {},
    });
    client.connect();
    const first = FakeWS.instances[0];
    first.open();
    expect(client.send(sessionCreate({}))).toBe(true); // seq=1，已发出

    first.drop();
    timers.advance(10000); // 等退避重连

    const second = FakeWS.instances[1];
    expect(second).toBeDefined();
    expect(client.send(controlEnd())).toBe(false); // 新连接还没 open → 积压
    expect(client.pendingText).toBe(1);

    second.open();
    const out = second.sent.map((s) => decodeText(s as string));
    expect(out.map((e) => e.type)).toEqual(["control.end"]);
    expect(out[0].seq).toBe(2); // 接在已发出的 seq=1 之后，不回头
  });

  it("主动 close()：未发出的文本显式报错，不静默消失", () => {
    const errors: unknown[] = [];
    const client = new SocketClient({
      url: "ws://x/ws",
      socketFactory: (u) => new FakeWS(u),
      timers: new FakeTimers(),
      heartbeatMs: 0,
      onEvent: () => {},
      onError: (e) => errors.push(e),
    });
    client.connect();
    expect(client.send(sessionCreate({}))).toBe(false); // 入队

    client.close();

    expect(client.pendingText).toBe(0);
    expect(errors.some((e) => String(e).includes("未发出"))).toBe(true);
  });
});

// ------------------------------------------------------------------ 测试

describe("帧解码与多路复用", () => {
  it("文本帧 → EnvelopeEvent", () => {
    const events: ServerEvent[] = [];
    const timers = new FakeTimers();
    const client = new SocketClient({
      url: "ws://x/ws",
      socketFactory: (u) => new FakeWS(u),
      timers,
      heartbeatMs: 0,
      onEvent: (e) => events.push(e),
    });
    client.connect();
    const ws = FakeWS.instances[0];
    ws.open();
    ws.deliver(textFrame("state.changed", { from: "idle", to: "setup", reason: "create" }, 1));

    expect(events).toHaveLength(1);
    expect(events[0]).toMatchObject({ kind: "envelope", type: "state.changed", seq: 1 });
  });

  it("二进制 0x11 → tts_chunk 事件 + 原始字节回调", () => {
    const events: ServerEvent[] = [];
    const audio: Uint8Array[] = [];
    const client = new SocketClient({
      url: "ws://x/ws",
      socketFactory: (u) => new FakeWS(u),
      timers: new FakeTimers(),
      heartbeatMs: 0,
      onEvent: (e) => events.push(e),
      onAudioChunk: (b) => audio.push(b),
    });
    client.connect();
    const ws = FakeWS.instances[0];
    ws.open();
    ws.deliver(b64ToBytes(Opcode.TTS_AUDIO, new Uint8Array([9, 8, 7])));

    expect(audio[0]).toHaveLength(3);
    expect(events[0]).toMatchObject({ kind: "tts_chunk", byteLength: 3 });
  });

  it("二进制 0x12 → viseme 时间轴解析", () => {
    const events: ServerEvent[] = [];
    const client = new SocketClient({
      url: "ws://x/ws",
      socketFactory: (u) => new FakeWS(u),
      timers: new FakeTimers(),
      heartbeatMs: 0,
      onEvent: (e) => events.push(e),
    });
    client.connect();
    const ws = FakeWS.instances[0];
    ws.open();
    const body = new TextEncoder().encode(
      JSON.stringify({ visemes: [{ t_ms: 0, viseme: "rest" }, { t_ms: 120, viseme: "a", weight: 0.8 }] }),
    );
    ws.deliver(b64ToBytes(Opcode.VISEME, body));

    expect(events[0].kind).toBe("viseme");
    if (events[0].kind === "viseme") {
      expect(events[0].visemes).toHaveLength(2);
      expect(events[0].visemes[1]).toEqual({ t_ms: 120, viseme: "a", weight: 0.8 });
    }
  });
});

describe("信用窗口背压（上传）", () => {
  it("窗口满则暂停发送（入队），收到 media.ack 后按序续发", () => {
    const timers = new FakeTimers();
    const client = new SocketClient({
      url: "ws://x/ws",
      socketFactory: (u) => new FakeWS(u),
      timers,
      heartbeatMs: 0,
      creditCapacity: 2,
      onEvent: () => undefined,
    });
    client.connect();
    const ws = FakeWS.instances[0];
    ws.open();

    expect(client.sendMedia(Opcode.MEDIA_AUDIO, new Uint8Array([1]))).toBe(true);
    expect(client.sendMedia(Opcode.MEDIA_AUDIO, new Uint8Array([2]))).toBe(true);
    // 窗口满：第 3 帧入队，不发出。
    expect(client.credit.full).toBe(true);
    expect(client.sendMedia(Opcode.MEDIA_AUDIO, new Uint8Array([3]))).toBe(false);
    expect(client.pendingMedia).toBe(1);
    expect(ws.binarySends()).toBe(2);

    // 服务端回 ack（释放 1 个信用）→ 队列里的第 3 帧自动续发。
    ws.deliver(textFrame(ServerType.MEDIA_ACK, { frames: 1, consumed: 1, credits: 1 }, 5));
    expect(ws.binarySends()).toBe(3);
    expect(client.pendingMedia).toBe(0);
    expect(client.credit.full).toBe(true);
  });
});

describe("断线重连 + 事件重放", () => {
  it("重连后重放事件，store 状态与断线前一致（seq 去重）", () => {
    const timers = new FakeTimers();
    const store = useInterviewStore;
    const client = new SocketClient({
      url: "ws://x/ws",
      socketFactory: (u) => new FakeWS(u),
      timers,
      heartbeatMs: 0,
      onEvent: (e) => store.getState().appendEvent(e),
      onReconnect: () => {
        client.send(sessionCreate({ resume: true, lastSeq: store.getState().events.length }));
      },
    });
    client.connect();
    const ws0 = FakeWS.instances[0];
    ws0.open();

    const replay = [
      textFrame("state.changed", { from: "idle", to: "setup", reason: "create" }, 1),
      textFrame("question.start", { turn_id: "s-1-t0", turn_index: 0, text: "自我介绍", scored: true }, 2),
      textFrame("eval.done", { turn_id: "s-1-t0", score: 80, dims: {}, provider: "llm", confidence: 0.8 }, 3),
    ];
    for (const f of replay) ws0.deliver(f);
    const before = store.getState().derived;
    expect(store.getState().events).toHaveLength(3);

    // 断线 → 退避重连。
    ws0.drop();
    expect(client.getStatus()).toBe("reconnecting");
    timers.advance(2000);

    const ws1 = FakeWS.instances[1];
    expect(ws1).toBeDefined();
    ws1.open(); // 触发 onReconnect → 发送带 resume 的 session.create
    const resumeSent = ws1.sent.some(
      (s) => typeof s === "string" && s.includes('"resume":true'),
    );
    expect(resumeSent).toBe(true);
    expect(client.getStatus()).toBe("open");

    // 服务端重放同一批事件（重复 seq）。
    for (const f of replay) ws1.deliver(f);
    const after = store.getState().derived;
    expect(after).toEqual(before);
  });

  it("主动 close 不再重连", () => {
    const timers = new FakeTimers();
    const client = new SocketClient({
      url: "ws://x/ws",
      socketFactory: (u) => new FakeWS(u),
      timers,
      heartbeatMs: 0,
      onEvent: () => undefined,
    });
    client.connect();
    FakeWS.instances[0].open();
    client.close();
    timers.advance(20000);
    expect(FakeWS.instances).toHaveLength(1);
    expect(client.getStatus()).toBe("closed");
  });
});

describe("心跳看门狗", () => {
  it("长时间无入站帧 → 判死并强制重连", () => {
    const timers = new FakeTimers();
    const client = new SocketClient({
      url: "ws://x/ws",
      socketFactory: (u) => new FakeWS(u),
      timers,
      heartbeatMs: 1000,
      heartbeatTimeoutMs: 2500,
      onEvent: () => undefined,
    });
    client.connect();
    FakeWS.instances[0].open();
    expect(client.getStatus()).toBe("open");

    // 无任何入站帧，推进 3s → 看门狗判死。
    timers.advance(3000);
    expect(client.getStatus()).toBe("reconnecting");
    // 退避后建立新连接。
    timers.advance(1000);
    expect(FakeWS.instances.length).toBeGreaterThanOrEqual(2);
  });
});

describe("上行信封 seq 自动分配", () => {
  it("未写 seq 时自动补单调递增 seq，且带上 session_id", () => {
    const client = new SocketClient({
      url: "ws://x/ws",
      socketFactory: (u) => new FakeWS(u),
      timers: new FakeTimers(),
      heartbeatMs: 0,
      sessionId: "s-1",
      onEvent: () => undefined,
    });
    client.connect();
    const ws = FakeWS.instances[0];
    ws.open();
    client.send(sessionCreate({ position: "后端" }));
    client.send(sessionCreate({ position: "后端" }));
    const parsed = ws.sent
      .filter((s): s is string => typeof s === "string")
      .map((s) => JSON.parse(s) as Record<string, unknown>);
    expect(parsed[0].seq).toBe(1);
    expect(parsed[1].seq).toBe(2);
    expect(parsed[0].session_id).toBe("s-1");
  });

  it("显式 seq/ts/session_id 被保留", () => {
    const client = new SocketClient({
      url: "ws://x/ws",
      socketFactory: (u) => new FakeWS(u),
      timers: new FakeTimers(),
      heartbeatMs: 0,
      onEvent: () => undefined,
    });
    client.connect();
    const ws = FakeWS.instances[0];
    ws.open();
    client.send(makeEnvelope("answer.text", { text: "x" }, { seq: 9, ts: 5, sessionId: "z" }));
    const parsed = JSON.parse(ws.sent[0] as string) as Record<string, unknown>;
    expect(parsed.seq).toBe(9);
    expect(parsed.ts).toBe(5);
    expect(parsed.session_id).toBe("z");
  });
});

describe("错误与边界分支", () => {
  it("非法文本帧 → onError，不产生事件", () => {
    const events: ServerEvent[] = [];
    const errors: unknown[] = [];
    const client = new SocketClient({
      url: "ws://x/ws",
      socketFactory: (u) => new FakeWS(u),
      timers: new FakeTimers(),
      heartbeatMs: 0,
      onEvent: (e) => events.push(e),
      onError: (e) => errors.push(e),
    });
    client.connect();
    const ws = FakeWS.instances[0];
    ws.open();
    ws.deliver("{ this is not json");
    expect(events).toHaveLength(0);
    expect(errors).toHaveLength(1);
  });

  it("未知 opcode 的二进制帧 → onError", () => {
    const errors: unknown[] = [];
    const client = new SocketClient({
      url: "ws://x/ws",
      socketFactory: (u) => new FakeWS(u),
      timers: new FakeTimers(),
      heartbeatMs: 0,
      onEvent: () => undefined,
      onError: (e) => errors.push(e),
    });
    client.connect();
    const ws = FakeWS.instances[0];
    ws.open();
    ws.deliver(new Uint8Array([0x7f, 0, 0, 0, 0]).buffer);
    expect(errors).toHaveLength(1);
  });

  it("不支持的 WS 消息类型 → onError", () => {
    const errors: unknown[] = [];
    const client = new SocketClient({
      url: "ws://x/ws",
      socketFactory: (u) => new FakeWS(u),
      timers: new FakeTimers(),
      heartbeatMs: 0,
      onEvent: () => undefined,
      onError: (e) => errors.push(e),
    });
    client.connect();
    const ws = FakeWS.instances[0];
    ws.open();
    ws.deliver(12345);
    expect(errors).toHaveLength(1);
  });

  it("Uint8Array 视图（ArrayBufferView）同样被解码", () => {
    const events: ServerEvent[] = [];
    const client = new SocketClient({
      url: "ws://x/ws",
      socketFactory: (u) => new FakeWS(u),
      timers: new FakeTimers(),
      heartbeatMs: 0,
      onEvent: (e) => events.push(e),
    });
    client.connect();
    const ws = FakeWS.instances[0];
    ws.open();
    ws.deliver(encodeBinary(Opcode.TTS_AUDIO, new Uint8Array([1, 2])));
    expect(events[0]).toMatchObject({ kind: "tts_chunk", byteLength: 2 });
  });

  it("Blob 形式的二进制帧被异步解码", async () => {
    const events: ServerEvent[] = [];
    const client = new SocketClient({
      url: "ws://x/ws",
      socketFactory: (u) => new FakeWS(u),
      timers: new FakeTimers(),
      heartbeatMs: 0,
      onEvent: (e) => events.push(e),
    });
    client.connect();
    const ws = FakeWS.instances[0];
    ws.open();
    const framed = encodeBinary(Opcode.TTS_AUDIO, new Uint8Array([3, 4, 5]));
    ws.deliver(new Blob([framed.buffer as ArrayBuffer]));
    await new Promise((r) => setTimeout(r, 0));
    expect(events[0]).toMatchObject({ kind: "tts_chunk", byteLength: 3 });
  });

  it("连接未就绪时 send 返回 false 且不抛", () => {
    const client = new SocketClient({
      url: "ws://x/ws",
      socketFactory: (u) => new FakeWS(u),
      timers: new FakeTimers(),
      heartbeatMs: 0,
      onEvent: () => undefined,
    });
    expect(client.send(sessionCreate())).toBe(false);
    expect(client.getStatus()).toBe("idle");
  });

  it("未连接时 sendMedia 入队（背压）", () => {
    const client = new SocketClient({
      url: "ws://x/ws",
      socketFactory: (u) => new FakeWS(u),
      timers: new FakeTimers(),
      heartbeatMs: 0,
      onEvent: () => undefined,
    });
    expect(client.sendMedia(Opcode.MEDIA_AUDIO, new Uint8Array([1]))).toBe(false);
    expect(client.pendingMedia).toBe(1);
  });

  it("媒体队列超上限 → 丢最旧帧并报错", () => {
    const errors: unknown[] = [];
    const client = new SocketClient({
      url: "ws://x/ws",
      socketFactory: (u) => new FakeWS(u),
      timers: new FakeTimers(),
      heartbeatMs: 0,
      creditCapacity: 1,
      maxMediaQueue: 1,
      onEvent: () => undefined,
      onError: (e) => errors.push(e),
    });
    client.connect();
    const ws = FakeWS.instances[0];
    ws.open();
    client.sendMedia(Opcode.MEDIA_AUDIO, new Uint8Array([1])); // 占满窗口，发出
    client.sendMedia(Opcode.MEDIA_AUDIO, new Uint8Array([2])); // 入队
    client.sendMedia(Opcode.MEDIA_AUDIO, new Uint8Array([3])); // 队列满 → 丢最旧
    expect(client.pendingMedia).toBe(1);
    expect(errors.length).toBeGreaterThanOrEqual(1);
  });

  it("超量 media.ack 释放被显式暴露（不炸连接）", () => {
    const errors: unknown[] = [];
    const client = new SocketClient({
      url: "ws://x/ws",
      socketFactory: (u) => new FakeWS(u),
      timers: new FakeTimers(),
      heartbeatMs: 0,
      onEvent: () => undefined,
      onError: (e) => errors.push(e),
    });
    client.connect();
    const ws = FakeWS.instances[0];
    ws.open();
    client.sendMedia(Opcode.MEDIA_AUDIO, new Uint8Array([1]));
    // 只占了 1 个信用，却 ack 释放 5 个 → 抛错 → 被捕获并 releaseAll。
    ws.deliver(textFrame(ServerType.MEDIA_ACK, { frames: 5, consumed: 5, credits: 0 }, 3));
    expect(errors.length).toBeGreaterThanOrEqual(1);
    expect(client.credit.used).toBe(0);
  });

  it("socketFactory 抛错 → 报错并安排重连", () => {
    const errors: unknown[] = [];
    const client = new SocketClient({
      url: "ws://x/ws",
      socketFactory: () => {
        throw new Error("boom");
      },
      timers: new FakeTimers(),
      heartbeatMs: 0,
      onEvent: () => undefined,
      onError: (e) => errors.push(e),
    });
    client.connect();
    expect(errors.length).toBeGreaterThanOrEqual(1);
    expect(client.getStatus()).toBe("reconnecting");
  });

  it("重复 connect 幂等", () => {
    const client = new SocketClient({
      url: "ws://x/ws",
      socketFactory: (u) => new FakeWS(u),
      timers: new FakeTimers(),
      heartbeatMs: 0,
      onEvent: () => undefined,
    });
    client.connect();
    FakeWS.instances[0].open();
    client.connect();
    expect(FakeWS.instances).toHaveLength(1);
    expect(client.getStatus()).toBe("open");
  });
});
