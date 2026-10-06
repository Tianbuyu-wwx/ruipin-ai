/**
 * 跨语言线序契约（前端一侧）。
 *
 * 为什么要有这个文件
 * ------------------
 * 前端与后端各有一套帧编解码，两边各自的单元测试都只验证"自己跟自己一致"。
 * 于是编码器改了 4 字节头、viseme 的 JSON 键名改了、opcode 从 0x11 挪到 0x12
 * ——任一处改动，**两侧都不会红**，而用户看到的是"没声音"或"口型乱动"。
 *
 * 这里喂进去的字节**不是手写样例**，而是后端真实产物导出的夹具
 * （`wireFixtures.json`，由 `v2/tools/export_wire_fixtures.py` 生成，
 *  `v2/tests/test_wire_fixtures.py` 保证它不过期）。
 *
 * 读法：夹具过期 → 后端那侧先红；夹具更新后本文件红了 → 说明这次改动
 * 真的动了线序，前端必须同步跟进。两侧任一掉队都会立刻被发现。
 */

import { describe, expect, it } from "vitest";
import fixture from "./wireFixtures.json";
import { decodeBinary } from "./codec";
import { Opcode, ServerType } from "./constants";
import { SocketClient, parseVisemePayload, type WebSocketLike } from "../net/socket";
import type { ServerEvent } from "../store/types";

function hexToBytes(hex: string): Uint8Array {
  if (hex.length % 2 !== 0) throw new Error(`hex 长度必须是偶数：${hex.length}`);
  const out = new Uint8Array(hex.length / 2);
  for (let i = 0; i < out.length; i += 1) {
    out[i] = Number.parseInt(hex.slice(i * 2, i * 2 + 2), 16);
  }
  return out;
}

function bytesToHex(bytes: Uint8Array): string {
  return Array.from(bytes)
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

class FakeWS implements WebSocketLike {
  binaryType = "";
  readyState = 1;
  sent: unknown[] = [];
  onopen: (() => void) | null = null;
  onmessage: ((ev: { data: unknown }) => void) | null = null;
  onclose: ((ev: { code: number; reason: string }) => void) | null = null;
  onerror: ((ev: unknown) => void) | null = null;

  constructor(public url: string) {}

  send(data: unknown): void {
    this.sent.push(data);
  }

  close(): void {
    this.readyState = 3;
  }

  deliver(data: unknown): void {
    this.onmessage?.({ data });
  }
}

describe("opcode / 消息类型常量与后端一致", () => {
  it("TTS 音频与 viseme 的 opcode 值一致", () => {
    expect(Opcode.TTS_AUDIO).toBe(fixture.opcodes.TTS_AUDIO);
    expect(Opcode.VISEME).toBe(fixture.opcodes.VISEME);
    expect(Opcode.TTS_AUDIO).toBe(0x11);
    expect(Opcode.VISEME).toBe(0x12);
  });

  it("服务端消息类型字面量一致", () => {
    expect(ServerType.QUESTION_TTS_CHUNK).toBe(fixture.serverTypes.QUESTION_TTS_CHUNK);
    expect(ServerType.AVATAR_VISEME).toBe(fixture.serverTypes.AVATAR_VISEME);
  });
});

describe("后端导出的音频帧能被前端解码", () => {
  it("每一帧都能解出 opcode 0x11，且长度字段与实际负载相符", () => {
    for (const hex of fixture.ttsFramesHex) {
      const { opcode, payload } = decodeBinary(hexToBytes(hex));
      expect(opcode).toBe(Opcode.TTS_AUDIO);
      // 帧头 4 字节大端长度必须等于实际负载长度（长度不符会被 decodeBinary 拒绝，
      // 这里额外断言一次，避免"恰好没抛"掩盖了截断）
      expect(hexToBytes(hex).length).toBe(payload.byteLength + 5);
    }
  });

  it("逐帧拼接后的音频与后端整段音频逐字节一致", () => {
    const joined = fixture.ttsFramesHex
      .map((hex) => decodeBinary(hexToBytes(hex)).payload)
      .reduce((acc, cur) => {
        const merged = new Uint8Array(acc.byteLength + cur.byteLength);
        merged.set(acc, 0);
        merged.set(cur, acc.byteLength);
        return merged;
      }, new Uint8Array(0));
    expect(bytesToHex(joined)).toBe(fixture.audioHex);
    expect(joined.byteLength).toBeGreaterThan(0);
  });

  it("每帧负载都不超过后端声明的分片大小", () => {
    for (const hex of fixture.ttsFramesHex) {
      const { payload } = decodeBinary(hexToBytes(hex));
      expect(payload.byteLength).toBeLessThanOrEqual(fixture.ttsChunkBytes);
    }
  });
});

describe("后端导出的 viseme 帧能被前端解码", () => {
  it("解出的口型时间轴与后端逐字段一致", () => {
    const { opcode, payload } = decodeBinary(hexToBytes(fixture.visemeFrameHex));
    expect(opcode).toBe(Opcode.VISEME);
    expect(parseVisemePayload(payload)).toEqual(fixture.visemes);
  });

  it("时间轴单调不减（口型不能来回跳）", () => {
    const { payload } = decodeBinary(hexToBytes(fixture.visemeFrameHex));
    const times = parseVisemePayload(payload).map((v) => v.t_ms);
    expect(times).toEqual([...times].sort((a, b) => a - b));
  });

  it("损坏的负载不抛异常，而是报错并给空时间轴", () => {
    const errors: unknown[] = [];
    const bad = new TextEncoder().encode("{ 这不是 JSON");
    expect(parseVisemePayload(bad, (e) => errors.push(e))).toEqual([]);
    expect(errors).toHaveLength(1);
  });
});

describe("整帧走一遍真实客户端解码路径", () => {
  it("音频帧触发 onAudioChunk 且字节原样，viseme 帧产出时间轴事件", () => {
    const audioChunks: Uint8Array[] = [];
    const events: ServerEvent[] = [];
    const ws = new FakeWS("ws://x/ws");
    const client = new SocketClient({
      url: "ws://x/ws",
      socketFactory: () => ws,
      onEvent: (ev) => events.push(ev),
      onAudioChunk: (b) => audioChunks.push(b),
    });
    client.connect();

    for (const hex of fixture.ttsFramesHex) ws.deliver(hexToBytes(hex).buffer);
    ws.deliver(hexToBytes(fixture.visemeFrameHex).buffer);

    expect(audioChunks).toHaveLength(fixture.ttsFramesHex.length);
    const joined = audioChunks.reduce((acc, cur) => {
      const merged = new Uint8Array(acc.byteLength + cur.byteLength);
      merged.set(acc, 0);
      merged.set(cur, acc.byteLength);
      return merged;
    }, new Uint8Array(0));
    expect(bytesToHex(joined)).toBe(fixture.audioHex);

    const ttsEvents = events.filter((e) => e.kind === "tts_chunk");
    expect(ttsEvents).toHaveLength(fixture.ttsFramesHex.length);
    expect(
      ttsEvents.reduce(
        (n, e) => n + (e.kind === "tts_chunk" ? e.byteLength : 0),
        0,
      ),
    ).toBe(hexToBytes(fixture.audioHex).byteLength);

    const visemeEvents = events.filter((e) => e.kind === "viseme");
    expect(visemeEvents).toHaveLength(1);
    const first = visemeEvents[0];
    expect(first.kind === "viseme" ? first.visemes : null).toEqual(fixture.visemes);
  });

  it("未处理的 opcode 会被显式报错，而不是静默丢弃", () => {
    const errors: unknown[] = [];
    const ws = new FakeWS("ws://x/ws");
    const client = new SocketClient({
      url: "ws://x/ws",
      socketFactory: () => ws,
      onEvent: () => {},
      onError: (e) => errors.push(e),
    });
    client.connect();

    // 0x01 是**上行**媒体 opcode，服务端下行绝不该发它——收到就该报错
    const bogus = new Uint8Array([0x01, 0, 0, 0, 0]);
    ws.deliver(bogus.buffer);
    expect(errors).toHaveLength(1);
  });
});
