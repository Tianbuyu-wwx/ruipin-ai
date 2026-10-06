/**
 * 编解码与 Python `protocol.py` 的一致性测试。
 *
 * 断言重点：
 * - 信封字段名/线序与 Python `Envelope.to_dict()` 一致（v/type/seq/ts/session_id/payload）；
 * - 二进制帧布局 `[opcode][4B 大端长度][payload]`（HEADER_SIZE=5）；
 * - opcode 数值与 `Opcode` IntEnum 一致（0x01/0x02/0x03/0x11/0x12）；
 * - 结构性错误一律抛错，不静默兜底。
 */

import { describe, expect, it } from "vitest";
import {
  BadFrameError,
  UnsupportedVersionError,
  UnknownOpcodeError,
  decodeBinary,
  decodeText,
  encodeBinary,
  encodeText,
  makeEnvelope,
} from "./codec";
import { HEADER_SIZE, Opcode } from "./constants";

describe("文本帧编解码（信封）", () => {
  it("encode→decode 往返相等，且字段名与 Python 一致", () => {
    const env = makeEnvelope(
      "question.start",
      { turn_id: "t-7", text: "请介绍一下你自己" },
      { seq: 42, ts: 1770000000000, sessionId: "s-xxx" },
    );
    const wire = encodeText(env);
    // 线序与字段名严格对齐 Python `Envelope.to_dict()`。
    expect(Object.keys(JSON.parse(wire))).toEqual([
      "v",
      "type",
      "seq",
      "ts",
      "session_id",
      "payload",
    ]);
    // 中文保留原字符（ensure_ascii=False 口径）。
    expect(wire).toContain("请介绍一下你自己");
    expect(decodeText(wire)).toEqual(env);
  });

  it("未知 type 宽容放行（前端可能比后端新）", () => {
    const wire = encodeText(makeEnvelope("future.event", { a: 1 }, { seq: 1 }));
    const env = decodeText(wire);
    expect(env.type).toBe("future.event");
  });

  it("版本不在受支持集合内 → UnsupportedVersionError", () => {
    const bad = JSON.stringify({ v: 2, type: "x", seq: 1, ts: 0, session_id: "", payload: {} });
    expect(() => decodeText(bad)).toThrow(UnsupportedVersionError);
  });

  it("字段缺失 / 类型不对 → BadFrameError", () => {
    expect(() => decodeText("not json")).toThrow(BadFrameError);
    expect(() => decodeText(JSON.stringify({ v: 1, type: "x", ts: 0 }))).toThrow(BadFrameError); // 缺 seq
    expect(() =>
      decodeText(JSON.stringify({ v: 1, type: "x", seq: -1, ts: 0, session_id: "", payload: {} })),
    ).toThrow(BadFrameError);
    expect(() =>
      decodeText(JSON.stringify({ v: 1, type: "", seq: 1, ts: 0, session_id: "", payload: {} })),
    ).toThrow(BadFrameError);
  });
});

describe("二进制帧编解码", () => {
  it("opcode 数值与 Python Opcode 一致", () => {
    expect(Opcode.MEDIA_AUDIO).toBe(0x01);
    expect(Opcode.MEDIA_VIDEO).toBe(0x02);
    expect(Opcode.MEDIA_SCREEN).toBe(0x03);
    expect(Opcode.TTS_AUDIO).toBe(0x11);
    expect(Opcode.VISEME).toBe(0x12);
    expect(HEADER_SIZE).toBe(5);
  });

  it("encode→decode 往返相等，长度字段为大端", () => {
    const payload = new Uint8Array([1, 2, 3, 250]);
    const frame = encodeBinary(Opcode.TTS_AUDIO, payload);
    expect(frame.byteLength).toBe(HEADER_SIZE + payload.byteLength);
    expect(frame[0]).toBe(0x11);
    // 4 字节大端长度 = 4
    expect([frame[1], frame[2], frame[3], frame[4]]).toEqual([0, 0, 0, 4]);
    const { opcode, payload: back } = decodeBinary(frame);
    expect(opcode).toBe(Opcode.TTS_AUDIO);
    expect(Array.from(back)).toEqual([1, 2, 3, 250]);
  });

  it("未知 opcode → UnknownOpcodeError（先于长度校验）", () => {
    const frame = new Uint8Array([0x7f, 0, 0, 0, 0]);
    expect(() => decodeBinary(frame)).toThrow(UnknownOpcodeError);
  });

  it("长度字段与实际不符 → BadFrameError（截断/粘连不兜底）", () => {
    const payload = new Uint8Array([1, 2, 3]);
    const frame = encodeBinary(Opcode.VISEME, payload);
    const truncated = frame.subarray(0, frame.byteLength - 1);
    expect(() => decodeBinary(truncated)).toThrow(BadFrameError);
    expect(() => decodeBinary(new Uint8Array([0x11, 0]))).toThrow(BadFrameError); // 短于头部
  });

  it("encode 拒绝未知 opcode", () => {
    expect(() => encodeBinary(0x99, new Uint8Array([1]))).toThrow(UnknownOpcodeError);
  });
});
