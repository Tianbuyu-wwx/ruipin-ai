"""传输层：WebSocket 帧的编解码与会话级网关（方案 §2.4）。

分四个模块：
- `protocol`：纯函数式编解码（信封 / 二进制帧 / opcode 表 / 信用窗口）。零状态、零时钟。
- `gateway`：会话级路由与传输层横切关注点（鉴权、限流、seq 去重、背压、超时、降级下行）。
- `bridge`：把信令翻译成 `InterviewService` 调用（唯一的接线层）。同时负责**语音下行**：
  出题后把题干交给 `TTSPort`，音频块（opcode `0x11`）与口型时间轴（`0x12`）走二进制帧下发。
- `session_server`：读写循环——读连接 → 喂网关 → 把下行帧写回连接（P0 闭环的最后一根线）。
  只依赖一个 `WSConnection` Protocol，不导入任何 WS 库，因此可被纯逻辑测试。

纪律：`gateway` **不认识业务**（不 import `InterviewService`），业务通过注入的
handler 回调处理；解析失败要么抛 `RuipinError` 子类，要么产出结构化 `error` 帧，
绝不静默返回半截数据。协议与业务之间的翻译集中在 `bridge`，不散落到两端。
"""

from .bridge import (
    DEFAULT_TTS_CHUNK_BYTES,
    OPCODE_KEY,
    BridgeConfig,
    InterviewBridge,
    SyncHandler,
    encode_viseme_timeline,
    media_envelope,
    split_outgoing,
)
from .gateway import DownlinkSequencer, Gateway
from .session_server import (
    DEFAULT_OUTBOX_POLL_S,
    ConnectionClosed,
    ConnectionStats,
    SessionServer,
    WSConnection,
)
from .protocol import (
    CLIENT_TYPES,
    HEADER_SIZE,
    KNOWN_TYPES,
    MAX_PAYLOAD_BYTES,
    MEDIA_IN,
    MEDIA_OUT,
    OPCODE_TYPES,
    PROTOCOL_VERSION,
    SERVER_TYPES,
    SEQ_RETAIN,
    SEQ_TOLERANCE,
    SUPPORTED_VERSIONS,
    BadFrameError,
    ClientType,
    CreditWindow,
    Envelope,
    ErrorCode,
    Opcode,
    ProtocolError,
    ServerType,
    UnknownOpcodeError,
    UnsupportedVersionError,
    decode_binary,
    decode_text,
    encode_binary,
    encode_text,
    error_envelope,
    make_envelope,
    state_changed,
)

__all__ = [
    "CLIENT_TYPES",
    "HEADER_SIZE",
    "KNOWN_TYPES",
    "MAX_PAYLOAD_BYTES",
    "MEDIA_IN",
    "MEDIA_OUT",
    "OPCODE_TYPES",
    "PROTOCOL_VERSION",
    "SEQ_RETAIN",
    "SEQ_TOLERANCE",
    "SERVER_TYPES",
    "SUPPORTED_VERSIONS",
    "BadFrameError",
    "ClientType",
    "CreditWindow",
    "DownlinkSequencer",
    "Envelope",
    "ErrorCode",
    "Gateway",
    "Opcode",
    "ProtocolError",
    "ServerType",
    "UnknownOpcodeError",
    "UnsupportedVersionError",
    "decode_binary",
    "decode_text",
    "encode_binary",
    "encode_text",
    "error_envelope",
    "make_envelope",
    "state_changed",
    # bridge
    "DEFAULT_TTS_CHUNK_BYTES",
    "OPCODE_KEY",
    "BridgeConfig",
    "InterviewBridge",
    "SyncHandler",
    "encode_viseme_timeline",
    "media_envelope",
    "split_outgoing",
    # session_server
    "DEFAULT_OUTBOX_POLL_S",
    "ConnectionClosed",
    "ConnectionStats",
    "SessionServer",
    "WSConnection",
]
