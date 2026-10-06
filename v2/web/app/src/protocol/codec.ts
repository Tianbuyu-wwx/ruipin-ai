/**
 * 帧编解码（纯函数，零 IO、不读时钟）—— 与 Python `protocol.py` 逐一对应。
 *
 * 纪律：结构性问题一律抛 `ProtocolError` 子类，**绝不静默兜底**。
 * 唯一的宽容路径是"未知 type"：前端可能比后端新，解码放行，由上层提示。
 */

import {
  HEADER_SIZE,
  MAX_PAYLOAD_BYTES,
  OPCODE_TYPES,
  PROTOCOL_VERSION,
  SUPPORTED_VERSIONS,
} from "./constants";

export class ProtocolError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ProtocolError";
  }
}

/** 帧结构损坏：非 JSON / 字段缺失或类型不对 / 二进制长度不符。 */
export class BadFrameError extends ProtocolError {
  constructor(message: string) {
    super(message);
    this.name = "BadFrameError";
  }
}

/** 信封版本 v 不在受支持集合内。 */
export class UnsupportedVersionError extends ProtocolError {
  readonly version: number;
  constructor(version: number) {
    super(`不支持的协议版本: ${version}；受支持版本: ${SUPPORTED_VERSIONS.join(",")}`);
    this.name = "UnsupportedVersionError";
    this.version = version;
  }
}

/** 二进制帧首字节不在 OPCODE_TYPES 内。 */
export class UnknownOpcodeError extends ProtocolError {
  readonly opcode: number;
  constructor(opcode: number) {
    super(
      `未知 opcode: 0x${opcode.toString(16).toUpperCase().padStart(2, "0")}（${opcode}）；已知: ${Object.keys(
        OPCODE_TYPES,
      )
        .map((o) => `0x${Number(o).toString(16)}`)
        .join(",")}`,
    );
    this.name = "UnknownOpcodeError";
    this.opcode = opcode;
  }
}

/** 一条控制事件的信封（字段与 Python `Envelope.to_dict()` 线序一致）。 */
export interface Envelope {
  v: number;
  type: string;
  seq: number;
  ts: number;
  session_id: string;
  payload: Record<string, unknown>;
}

export function makeEnvelope(
  type: string,
  payload: Record<string, unknown> = {},
  opts: { seq?: number; ts?: number; sessionId?: string; v?: number } = {},
): Envelope {
  return {
    v: opts.v ?? PROTOCOL_VERSION,
    type,
    seq: opts.seq ?? 0,
    ts: opts.ts ?? 0,
    session_id: opts.sessionId ?? "",
    payload: { ...payload },
  };
}

/** 信封 → 文本帧（JSON，保留中文原字符，与 Python `encode_text` 同款紧凑输出）。 */
export function encodeText(msg: Envelope): string {
  try {
    return JSON.stringify({
      v: msg.v,
      type: msg.type,
      seq: msg.seq,
      ts: msg.ts,
      session_id: msg.session_id,
      payload: msg.payload,
    });
  } catch (exc) {
    throw new BadFrameError(`信封无法序列化为 JSON（type=${msg.type}）: ${String(exc)}`);
  }
}

/** 文本帧 → 信封。非 UTF-8 / 非 JSON / 字段缺失或类型不对 → BadFrameError。 */
export function decodeText(raw: string): Envelope {
  if (typeof raw !== "string") {
    throw new BadFrameError(`文本帧必须是 string，实际: ${typeof raw}`);
  }
  let data: unknown;
  try {
    data = JSON.parse(raw);
  } catch (exc) {
    throw new BadFrameError(`文本帧不是合法 JSON: ${String(exc)}`);
  }
  return envelopeFromObject(data);
}

function envelopeFromObject(data: unknown): Envelope {
  if (data === null || typeof data !== "object" || Array.isArray(data)) {
    throw new BadFrameError(`信封必须是 JSON 对象，实际: ${data === null ? "null" : typeof data}`);
  }
  const obj = data as Record<string, unknown>;

  const version = requireInt(obj, "v");
  if (!SUPPORTED_VERSIONS.includes(version)) {
    throw new UnsupportedVersionError(version);
  }

  const mtype = obj["type"];
  if (mtype === undefined || mtype === null) {
    throw new BadFrameError("信封缺少 type 字段");
  }
  if (typeof mtype !== "string" || mtype === "") {
    throw new BadFrameError(`type 必须是非空字符串，实际: ${JSON.stringify(mtype)}`);
  }

  const seq = requireInt(obj, "seq");
  if (seq < 0) {
    throw new BadFrameError(`seq 不能为负: ${seq}`);
  }

  const ts = requireNumber(obj, "ts");

  let sessionId = obj["session_id"];
  if (sessionId === undefined || sessionId === null) sessionId = "";
  if (typeof sessionId !== "string") {
    throw new BadFrameError(`session_id 必须是字符串，实际: ${JSON.stringify(sessionId)}`);
  }

  let payload = obj["payload"];
  if (payload === undefined || payload === null) payload = {};
  if (typeof payload !== "object" || Array.isArray(payload)) {
    throw new BadFrameError(`payload 必须是对象，实际: ${typeof payload}`);
  }

  return {
    v: version,
    type: mtype,
    seq,
    ts,
    session_id: sessionId,
    payload: payload as Record<string, unknown>,
  };
}

function isPlainInt(v: unknown): v is number {
  return typeof v === "number" && Number.isInteger(v);
}

function requireInt(obj: Record<string, unknown>, key: string): number {
  if (!(key in obj)) throw new BadFrameError(`信封缺少 ${key} 字段`);
  const val = obj[key];
  if (!isPlainInt(val)) {
    throw new BadFrameError(`${key} 必须是整数，实际: ${JSON.stringify(val)}`);
  }
  return val;
}

function requireNumber(obj: Record<string, unknown>, key: string): number {
  if (!(key in obj)) throw new BadFrameError(`信封缺少 ${key} 字段`);
  const val = obj[key];
  if (typeof val !== "number" || Number.isNaN(val)) {
    throw new BadFrameError(`${key} 必须是数字，实际: ${JSON.stringify(val)}`);
  }
  return Math.trunc(val);
}

/** 按 HEADER_SIZE 布局打包：`[opcode][4B 大端长度][payload]`。 */
export function encodeBinary(opcode: number, payload: Uint8Array): Uint8Array {
  if (!isPlainInt(opcode)) {
    throw new BadFrameError(`opcode 必须是整数，实际: ${JSON.stringify(opcode)}`);
  }
  if (!(opcode in OPCODE_TYPES)) {
    throw new UnknownOpcodeError(opcode);
  }
  if (!(payload instanceof Uint8Array)) {
    throw new BadFrameError(`payload 必须是 Uint8Array，实际: ${typeof payload}`);
  }
  if (payload.byteLength > MAX_PAYLOAD_BYTES) {
    throw new BadFrameError(`payload 过长: ${payload.byteLength} > MAX_PAYLOAD_BYTES`);
  }
  const out = new Uint8Array(HEADER_SIZE + payload.byteLength);
  out[0] = opcode & 0xff;
  // 4 字节大端无符号长度（对应 struct ">I"）
  out[1] = (payload.byteLength >>> 24) & 0xff;
  out[2] = (payload.byteLength >>> 16) & 0xff;
  out[3] = (payload.byteLength >>> 8) & 0xff;
  out[4] = payload.byteLength & 0xff;
  out.set(payload, HEADER_SIZE);
  return out;
}

/** 解包二进制帧，返回 `(opcode, payload)`。长度不符抛 BadFrameError。 */
export function decodeBinary(raw: Uint8Array): { opcode: number; payload: Uint8Array } {
  if (!(raw instanceof Uint8Array)) {
    throw new BadFrameError(`二进制帧必须是 Uint8Array，实际: ${typeof raw}`);
  }
  if (raw.byteLength < HEADER_SIZE) {
    throw new BadFrameError(`二进制帧过短: ${raw.byteLength} 字节 < 头部 ${HEADER_SIZE} 字节`);
  }
  const opVal = raw[0];
  if (!(opVal in OPCODE_TYPES)) {
    // 先于长度校验：身份不明时谈内容没有意义（与 Python 一致）
    throw new UnknownOpcodeError(opVal);
  }
  const length =
    ((raw[1] << 24) >>> 0) + ((raw[2] << 16) >>> 0) + ((raw[3] << 8) >>> 0) + raw[4];
  const body = raw.subarray(HEADER_SIZE);
  if (body.byteLength !== length) {
    throw new BadFrameError(
      `长度字段 ${length} 与实际负载 ${body.byteLength} 字节不符（帧被截断或粘连）`,
    );
  }
  return { opcode: opVal, payload: body };
}
