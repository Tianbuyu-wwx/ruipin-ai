/**
 * 协议常量 —— **唯一真相源**。
 *
 * 与 Python 侧 `src/ruipin/transport/protocol.py` 逐一对应（方案 §2.4）。
 * 修改本文件前必须先改 Python 侧，否则前后端必然解错帧。
 *
 * 对应关系：
 *   PROTOCOL_VERSION      ← protocol.PROTOCOL_VERSION            = 1
 *   SUPPORTED_VERSIONS    ← protocol.SUPPORTED_VERSIONS          = {1}
 *   HEADER_SIZE           ← protocol.HEADER_SIZE                 = 5
 *   MAX_PAYLOAD_BYTES     ← protocol.MAX_PAYLOAD_BYTES           = 0xFFFFFFFF
 *   SEQ_TOLERANCE         ← protocol.SEQ_TOLERANCE               = 8
 *   SEQ_RETAIN            ← protocol.SEQ_RETAIN                  = 4
 *   ClientType            ← protocol.ClientType (StrEnum)
 *   ServerType            ← protocol.ServerType (StrEnum)
 *   Opcode                ← protocol.Opcode (IntEnum)
 *   MEDIA_IN / MEDIA_OUT  ← protocol.MEDIA_IN / MEDIA_OUT
 *   ErrorCode             ← protocol.ErrorCode (StrEnum)
 *
 * 二进制帧布局（与 Python `encode_binary` / `decode_binary` 完全一致）：
 *
 *   offset  size   含义
 *   0       1      opcode（见 Opcode）
 *   1       4      length：payload 字节数，大端无符号整数（struct ">BI"）
 *   5       length payload：原始字节
 *
 *   ┌────────┬───────────────────────────────┐
 *   │ opcode │ length (4B, big-endian)       │  ← 5 字节定长头
 *   ├────────┼───────────────────────────────┤
 *   │ payload (length 字节)                   │
 *   └────────┴───────────────────────────────┘
 *
 * 解码时若实际负载长度 != length 字段 → 抛 BadFrameError（截断/粘连都算坏帧，
 * 不做"有多少用多少"的兜底，与 Python 一致）。
 */

export const PROTOCOL_VERSION = 1;
export const SUPPORTED_VERSIONS: readonly number[] = [1];

/** 二进制帧定长头：1 字节 opcode + 4 字节大端长度。 */
export const HEADER_SIZE = 5;
export const MAX_PAYLOAD_BYTES = 0xffffffff;

/** seq 容忍窗口与去重缓存深度（上行语义，见 Python module docstring）。 */
export const SEQ_TOLERANCE = 8;
export const SEQ_RETAIN = 4;

// ------------------------------------------------------------------ 消息类型

/** 客户端 → 服务端（对应 protocol.ClientType）。 */
export const ClientType = {
  SESSION_CREATE: "session.create",
  CONSENT_GRANT: "consent.grant",
  ANSWER_TEXT: "answer.text",
  ANSWER_COMMIT: "answer.commit",
  CONTROL_BARGE_IN: "control.barge_in",
  CONTROL_SKIP: "control.skip",
  CONTROL_END: "control.end",
  MEDIA_AUDIO: "media.audio",
  MEDIA_VIDEO: "media.video",
  MEDIA_SCREEN: "media.screen",
  /**
   * 端侧 rPPG 结果上行（每 2s 一批，走文本帧）。
   *
   * Python 侧 `protocol.ClientType.PHYSIO_BATCH` 一直有这一条，前端这张表却漏了
   * ——于是"心率进评分"那条通道在**类型表层面**根本不存在。生理信号在端侧算完
   * 才上传（服务端只做事件聚合与门控），所以这条不能少。
   * 注意：当前前端尚未实现端侧 rPPG，本项只是把类型表补齐、不再说谎。
   */
  PHYSIO_BATCH: "physio.batch",
} as const;
export type ClientType = (typeof ClientType)[keyof typeof ClientType];

/** 服务端 → 客户端（对应 protocol.ServerType）。 */
export const ServerType = {
  STATE_CHANGED: "state.changed",
  QUESTION_START: "question.start",
  QUESTION_TTS_CHUNK: "question.tts_chunk",
  AVATAR_VISEME: "avatar.viseme",
  TRANSCRIPT_PARTIAL: "transcript.partial",
  TRANSCRIPT_FINAL: "transcript.final",
  EVAL_DONE: "eval.done",
  METRIC_HR: "metric.hr",
  DEGRADATION_CHANGED: "degradation.changed",
  REPORT_READY: "report.ready",
  ERROR: "error",
  MEDIA_ACK: "media.ack",
  SESSION_TIMEOUT: "session.timeout",
} as const;
export type ServerType = (typeof ServerType)[keyof typeof ServerType];

export const CLIENT_TYPES: ReadonlySet<string> = new Set(Object.values(ClientType));
export const SERVER_TYPES: ReadonlySet<string> = new Set(Object.values(ServerType));
export const KNOWN_TYPES: ReadonlySet<string> = new Set([
  ...CLIENT_TYPES,
  ...SERVER_TYPES,
]);

// ------------------------------------------------------------------ opcode

/** 二进制帧首字节（对应 protocol.Opcode）。0x01–0x0F 上行，0x10–0x1F 下行。 */
export const Opcode = {
  MEDIA_AUDIO: 0x01,
  MEDIA_VIDEO: 0x02,
  MEDIA_SCREEN: 0x03,
  TTS_AUDIO: 0x11,
  VISEME: 0x12,
} as const;
export type Opcode = (typeof Opcode)[keyof typeof Opcode];

/** opcode → 消息类型（对应 protocol.MEDIA_IN）。 */
export const MEDIA_IN: Readonly<Record<number, string>> = {
  [Opcode.MEDIA_AUDIO]: ClientType.MEDIA_AUDIO,
  [Opcode.MEDIA_VIDEO]: ClientType.MEDIA_VIDEO,
  [Opcode.MEDIA_SCREEN]: ClientType.MEDIA_SCREEN,
};

/** opcode → 消息类型（对应 protocol.MEDIA_OUT）。 */
export const MEDIA_OUT: Readonly<Record<number, string>> = {
  [Opcode.TTS_AUDIO]: ServerType.QUESTION_TTS_CHUNK,
  [Opcode.VISEME]: ServerType.AVATAR_VISEME,
};

export const OPCODE_TYPES: Readonly<Record<number, string>> = {
  ...MEDIA_IN,
  ...MEDIA_OUT,
};

// ------------------------------------------------------------------ 错误码

/** `error` 帧的 code（对应 protocol.ErrorCode）。前端按 code 分支，不解析文案。 */
export const ErrorCode = {
  BAD_FRAME: "bad_frame",
  UNSUPPORTED_VERSION: "unsupported_version",
  UNKNOWN_TYPE: "unknown_type",
  UNAUTHORIZED: "unauthorized",
  RATE_LIMITED: "rate_limited",
  OUT_OF_ORDER: "out_of_order",
  DUPLICATE: "duplicate",
  SESSION_TIMEOUT: "session_timeout",
  CREDIT_EXHAUSTED: "credit_exhausted",
  INTERNAL: "internal",
} as const;
export type ErrorCode = (typeof ErrorCode)[keyof typeof ErrorCode];

// ------------------------------------------------------------------ 传输层默认值

/** 媒体上行信用窗口容量（对应 gateway.DEFAULT_CREDIT_CAPACITY）。 */
export const DEFAULT_CREDIT_CAPACITY = 16;
/** 服务端每消费多少帧回一个 media.ack（对应 gateway.DEFAULT_ACK_EVERY）。 */
export const DEFAULT_ACK_EVERY = 4;
/** 下行音频抖动缓冲（ms），方案 §2.4 要求 200–400ms。 */
export const JITTER_BUFFER_MS = 300;
/** 心跳间隔（ms）。 */
export const HEARTBEAT_INTERVAL_MS = 15000;
/** 重连退避：初始 / 最大 / 倍率。 */
export const RECONNECT_BASE_MS = 500;
export const RECONNECT_MAX_MS = 8000;
export const RECONNECT_FACTOR = 2;
