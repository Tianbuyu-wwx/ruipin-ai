"""WebSocket 帧协议：信封 + 二进制帧的编解码（零外部依赖，纯函数）。

对应方案 §2.4「通信协议」：单条 WebSocket 承载两类帧——
文本帧（JSON 控制事件）与二进制帧（首字节 opcode，同连接多路复用）。

设计要点
--------
1. **纯函数**：`encode_*` / `decode_*` 不持有状态、不读时钟、不做 IO，
   便于属性测试（穷举 opcode × 长度）。
2. **时钟不入本模块**：`ts` 由调用方传入（网关注入 `clock`，见 `gateway.py`）。
3. **绝不静默兜底**：结构性问题一律抛 `RuipinError` 子类（见下方异常族）。
   唯一的宽容路径是"未知 type"——前端可能比后端新，解码放行，
   由网关回 `error(code=unknown_type)`，而不是让连接炸掉。

二进制帧布局（本模块自定，唯一真相源）
--------------------------------------
    offset  size   含义
    0       1      opcode（见 MEDIA_IN / MEDIA_OUT）
    1       4      length：payload 字节数，大端无符号整数（struct ">BI"）
    5       length payload：原始字节

    ┌────────┬───────────────────────────────┐
    │ opcode │ length (4B, big-endian)       │  ← 5 字节定长头（HEADER_SIZE）
    ├────────┼───────────────────────────────┤
    │ payload (length 字节)                   │
    └────────┴───────────────────────────────┘

    解码时若实际负载长度 != length 字段，抛 `BadFrameError`（截断/粘连都算坏帧，
    不做"有多少用多少"的兜底）。

seq 语义（上行，每会话独立）
----------------------------
    SEQ_TOLERANCE = 8  容忍窗口：允许乱序到达的最大跨度
    SEQ_RETAIN    = 4  去重缓存保留的响应条数（内存有界）

    - 首帧：任意 seq 作为基线（客户端可能从任意值起）。
    - [last+1, last+SEQ_TOLERANCE]：正常接收，更新基线。
    - [last-SEQ_TOLERANCE, last] 且响应仍在缓存：duplicate → **回放上次响应**，
      不重复处理（弱网重传语义）。
    - [last-SEQ_TOLERANCE, last] 但缓存已淘汰：duplicate → `error(duplicate)`。
    - seq < last - SEQ_TOLERANCE：落后过多 → `error(out_of_order)`。
    - seq > last + SEQ_TOLERANCE：跳号过多（疑似丢包）→ `error(out_of_order)`。
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from types import MappingProxyType
from typing import Any, Mapping, Optional, Union

from ..domain.errors import RuipinError

# ---------------------------------------------------------------- 版本与常量

#: 当前写出的协议版本。
PROTOCOL_VERSION: int = 1
#: 服务端能接受的版本集合。不在其中的版本一律 `UnsupportedVersionError`。
SUPPORTED_VERSIONS: frozenset[int] = frozenset({1})

#: 二进制帧定长头：1 字节 opcode + 4 字节大端长度。
HEADER_SIZE: int = 5
#: 4 字节长度字段能表达的最大负载。超出必须报错，不能截尾写入。
MAX_PAYLOAD_BYTES: int = 0xFFFFFFFF

#: seq 容忍窗口与去重缓存深度（语义见模块 docstring）。
SEQ_TOLERANCE: int = 8
SEQ_RETAIN: int = 4


# ---------------------------------------------------------------- 消息类型


class ClientType(StrEnum):
    """客户端 → 服务端（方案 §2.4 表）。"""

    SESSION_CREATE = "session.create"
    CONSENT_GRANT = "consent.grant"
    ANSWER_TEXT = "answer.text"
    ANSWER_COMMIT = "answer.commit"
    CONTROL_BARGE_IN = "control.barge_in"
    CONTROL_SKIP = "control.skip"
    CONTROL_END = "control.end"
    MEDIA_AUDIO = "media.audio"
    MEDIA_VIDEO = "media.video"
    MEDIA_SCREEN = "media.screen"
    # 端侧 rPPG 结果上行（每 2s 一批）。方案 §2.4 的类型表**漏了这一条**，
    # 但模块详设 §9《数据结构与接口》明确要求「端侧 → 服务端（每 2 s 一条，批量上传）
    # HRBatch」。生理信号在端侧（WASM）算完才上传，服务端只做事件聚合与门控，
    # 所以必须有这条通道——否则"心率进评分"在协议层就断了。
    PHYSIO_BATCH = "physio.batch"


class ServerType(StrEnum):
    """服务端 → 客户端（方案 §2.4 表）。

    `media.ack` 与 `session.timeout` 是传输层补充：前者是信用窗口背压的确认，
    后者是空闲超时告知（方案要求"降级/异常必须显式告知用户"，超时同理）。
    """

    STATE_CHANGED = "state.changed"
    QUESTION_START = "question.start"
    QUESTION_TTS_CHUNK = "question.tts_chunk"
    AVATAR_VISEME = "avatar.viseme"
    TRANSCRIPT_PARTIAL = "transcript.partial"
    TRANSCRIPT_FINAL = "transcript.final"
    EVAL_DONE = "eval.done"
    METRIC_HR = "metric.hr"
    DEGRADATION_CHANGED = "degradation.changed"
    REPORT_READY = "report.ready"
    ERROR = "error"
    MEDIA_ACK = "media.ack"
    SESSION_TIMEOUT = "session.timeout"


# 用 str() 取字面值：StrEnum 成员的 hash 是 Enum 的 hash，直接放进集合会导致
# `"session.create" in CLIENT_TYPES` 判 False（踩过的坑），故显式转字符串。
CLIENT_TYPES: frozenset[str] = frozenset(str(m) for m in ClientType)
SERVER_TYPES: frozenset[str] = frozenset(str(m) for m in ServerType)
KNOWN_TYPES: frozenset[str] = CLIENT_TYPES | SERVER_TYPES


# ---------------------------------------------------------------- opcode


class Opcode(IntEnum):
    """二进制帧首字节。

    0x01–0x0F 客户端上传（`MEDIA_IN`）；0x10–0x1F 服务端下行（`MEDIA_OUT`）。
    """

    MEDIA_AUDIO = 0x01
    MEDIA_VIDEO = 0x02
    MEDIA_SCREEN = 0x03
    TTS_AUDIO = 0x11
    VISEME = 0x12


#: opcode → 对应的消息类型。只读映射，禁止运行时改动。
MEDIA_IN: Mapping[int, str] = MappingProxyType(
    {
        int(Opcode.MEDIA_AUDIO): str(ClientType.MEDIA_AUDIO),
        int(Opcode.MEDIA_VIDEO): str(ClientType.MEDIA_VIDEO),
        int(Opcode.MEDIA_SCREEN): str(ClientType.MEDIA_SCREEN),
    }
)
MEDIA_OUT: Mapping[int, str] = MappingProxyType(
    {
        int(Opcode.TTS_AUDIO): str(ServerType.QUESTION_TTS_CHUNK),
        int(Opcode.VISEME): str(ServerType.AVATAR_VISEME),
    }
)
OPCODE_TYPES: Mapping[int, str] = MappingProxyType({**MEDIA_IN, **MEDIA_OUT})

#: 信封 payload 里用这个键标记"这其实是一个二进制帧"（由最外层转成 `encode_binary`）。
#: 定义在帧格式层而不是桥接层：**网关也要靠它区分文本帧与媒体帧**
#: （例如下行编号只给文本帧——二进制帧根本没有 seq 字段）。
OPCODE_KEY = "opcode"


# ---------------------------------------------------------------- 错误码


class ErrorCode(StrEnum):
    """`error` 帧的结构化 code。前端按 code 分支，不解析中文文案。"""

    BAD_FRAME = "bad_frame"
    UNSUPPORTED_VERSION = "unsupported_version"
    UNKNOWN_TYPE = "unknown_type"
    UNAUTHORIZED = "unauthorized"
    RATE_LIMITED = "rate_limited"
    OUT_OF_ORDER = "out_of_order"
    DUPLICATE = "duplicate"
    SESSION_TIMEOUT = "session_timeout"
    CREDIT_EXHAUSTED = "credit_exhausted"
    INTERNAL = "internal"


# ---------------------------------------------------------------- 异常族


class ProtocolError(RuipinError):
    """传输层异常基类。"""


class BadFrameError(ProtocolError):
    """帧结构损坏：非 JSON / 字段缺失或类型不对 / 二进制长度不符。"""


class UnsupportedVersionError(ProtocolError):
    """信封版本 `v` 不在 `SUPPORTED_VERSIONS` 内。"""

    def __init__(self, version: int):
        self.version = version
        super().__init__(
            f"不支持的协议版本: {version}；受支持版本: {sorted(SUPPORTED_VERSIONS)}"
        )


class UnknownOpcodeError(ProtocolError):
    """二进制帧首字节不在 `OPCODE_TYPES` 内。"""

    def __init__(self, opcode: int):
        self.opcode = opcode
        super().__init__(
            f"未知 opcode: 0x{opcode:02X}（{opcode}）；已知: "
            f"{sorted(hex(o) for o in OPCODE_TYPES)}"
        )


# ---------------------------------------------------------------- 信封


@dataclass(frozen=True)
class Envelope:
    """一条控制事件的信封（方案 §2.4 示例字段，逐一对应）。

    Attributes:
        type: 消息类型，见 `ClientType` / `ServerType`。
        payload: 业务负载。媒体帧会带 `data: bytes`，仅供进程内 handler 使用，
            **不可**交给 `encode_text`（会抛 `BadFrameError`）。
        seq: 上行由客户端单调递增；下行由网关统一分配。
        ts: 毫秒时间戳，由注入的 clock 产生。
        session_id: 会话标识；鉴权前的帧为空串。
        v: 协议版本。
    """

    type: str
    payload: dict[str, Any] = field(default_factory=dict)
    seq: int = 0
    ts: int = 0
    session_id: str = ""
    v: int = PROTOCOL_VERSION

    @property
    def is_known_type(self) -> bool:
        """类型是否在已知集合内。False 时网关走 `unknown_type` 宽容路径。"""
        return self.type in KNOWN_TYPES

    def to_dict(self) -> dict[str, Any]:
        """线序字典：字段顺序与方案示例一致（v/type/seq/ts/session_id/payload）。"""
        return {
            "v": self.v,
            "type": self.type,
            "seq": self.seq,
            "ts": self.ts,
            "session_id": self.session_id,
            "payload": self.payload,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Envelope:
        """从已解析的 JSON 对象构造信封；校验失败抛 `ProtocolError` 子类。"""
        return _envelope_from_mapping(data)


# ---------------------------------------------------------------- 文本帧


def encode_text(msg: Envelope) -> str:
    """信封 → 文本帧（JSON 字符串，保留中文原字符）。

    Raises:
        BadFrameError: 负载含不可序列化对象（例如媒体帧的 `bytes`）。
    """
    try:
        return json.dumps(msg.to_dict(), ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError) as exc:  # 绝不静默：说清是谁序列化不了
        raise BadFrameError(f"信封无法序列化为 JSON（type={msg.type}）: {exc}") from exc


def decode_text(raw: Union[str, bytes, bytearray]) -> Envelope:
    """文本帧 → 信封。

    Raises:
        BadFrameError: 非 UTF-8 / 非 JSON / 非对象 / 字段缺失或类型不对。
        UnsupportedVersionError: `v` 不在受支持集合内。
    """
    if isinstance(raw, (bytes, bytearray)):
        try:
            raw = bytes(raw).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise BadFrameError(f"文本帧不是合法 UTF-8: {exc}") from exc
    if not isinstance(raw, str):
        raise BadFrameError(f"文本帧必须是 str/bytes，实际: {type(raw).__name__}")

    try:
        data = json.loads(raw)
    except ValueError as exc:  # json.JSONDecodeError 是 ValueError 子类
        raise BadFrameError(f"文本帧不是合法 JSON: {exc}") from exc
    return _envelope_from_mapping(data)


def _envelope_from_mapping(data: Any) -> Envelope:
    """字段级校验集中在此，供 `decode_text` 与 `Envelope.from_dict` 共用。"""
    if not isinstance(data, dict):
        raise BadFrameError(f"信封必须是 JSON 对象，实际: {type(data).__name__}")

    version = _require_int(data, "v")
    if version not in SUPPORTED_VERSIONS:
        raise UnsupportedVersionError(version)

    mtype = data.get("type")
    if mtype is None:
        raise BadFrameError("信封缺少 type 字段")
    if not isinstance(mtype, str) or not mtype:
        raise BadFrameError(f"type 必须是非空字符串，实际: {mtype!r}")

    seq = _require_int(data, "seq")
    if seq < 0:
        raise BadFrameError(f"seq 不能为负: {seq}")

    ts = _require_number(data, "ts")

    session_id = data.get("session_id", "")
    if session_id is None:
        session_id = ""
    if not isinstance(session_id, str):
        raise BadFrameError(f"session_id 必须是字符串，实际: {session_id!r}")

    payload = data.get("payload", {})
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        raise BadFrameError(f"payload 必须是对象，实际: {type(payload).__name__}")

    return Envelope(
        type=mtype,
        payload=payload,
        seq=seq,
        ts=ts,
        session_id=session_id,
        v=version,
    )


def _require_int(data: dict[str, Any], key: str) -> int:
    if key not in data:
        raise BadFrameError(f"信封缺少 {key} 字段")
    val = data[key]
    if isinstance(val, bool) or not isinstance(val, int):
        raise BadFrameError(f"{key} 必须是整数，实际: {val!r}")
    return val


def _require_number(data: dict[str, Any], key: str) -> int:
    if key not in data:
        raise BadFrameError(f"信封缺少 {key} 字段")
    val = data[key]
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        raise BadFrameError(f"{key} 必须是数字，实际: {val!r}")
    return int(val)


# ---------------------------------------------------------------- 二进制帧


def encode_binary(opcode: Union[int, Opcode], payload: bytes) -> bytes:
    """按 `HEADER_SIZE` 布局打包：`[opcode][4B 大端长度][payload]`。

    Raises:
        UnknownOpcodeError: opcode 不在 `OPCODE_TYPES` 内。
        BadFrameError: payload 不是字节流 / opcode 不是整数 / 超过 `MAX_PAYLOAD_BYTES`。
    """
    try:
        op = int(opcode)
    except (TypeError, ValueError) as exc:
        raise BadFrameError(f"opcode 必须是整数，实际: {opcode!r}") from exc
    if op not in OPCODE_TYPES:
        raise UnknownOpcodeError(op)
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise BadFrameError(f"payload 必须是 bytes，实际: {type(payload).__name__}")
    data = bytes(payload)
    if len(data) > MAX_PAYLOAD_BYTES:
        raise BadFrameError(
            f"payload 过长: {len(data)} > MAX_PAYLOAD_BYTES({MAX_PAYLOAD_BYTES})"
        )
    return struct.pack(">BI", op, len(data)) + data


def decode_binary(raw: bytes) -> tuple[Opcode, bytes]:
    """解包二进制帧，返回 `(opcode, payload)`。

    Raises:
        BadFrameError: 非字节流 / 短于 `HEADER_SIZE` / 长度字段与实际不符。
        UnknownOpcodeError: 首字节不在 `OPCODE_TYPES` 内（先于长度校验，
            身份不明时谈内容没有意义）。
    """
    if not isinstance(raw, (bytes, bytearray, memoryview)):
        raise BadFrameError(f"二进制帧必须是 bytes，实际: {type(raw).__name__}")
    data = bytes(raw)
    if len(data) < HEADER_SIZE:
        raise BadFrameError(
            f"二进制帧过短: {len(data)} 字节 < 头部 {HEADER_SIZE} 字节"
        )
    op_val = data[0]
    if op_val not in OPCODE_TYPES:
        raise UnknownOpcodeError(op_val)
    (length,) = struct.unpack(">I", data[1:HEADER_SIZE])
    body = data[HEADER_SIZE:]
    if len(body) != length:
        raise BadFrameError(
            f"长度字段 {length} 与实际负载 {len(body)} 字节不符（帧被截断或粘连）"
        )
    return Opcode(op_val), body


# ---------------------------------------------------------------- 帧构造助手


def make_envelope(
    mtype: Union[str, ClientType, ServerType],
    payload: Optional[dict[str, Any]] = None,
    *,
    seq: int = 0,
    ts: int = 0,
    session_id: str = "",
    v: int = PROTOCOL_VERSION,
) -> Envelope:
    """构造信封。`ts` 由调用方（网关注入的 clock）给出，本函数不读时钟。"""
    return Envelope(
        type=str(mtype),
        payload=dict(payload) if payload else {},
        seq=seq,
        ts=ts,
        session_id=session_id,
        v=v,
    )


def error_envelope(
    code: Union[ErrorCode, str],
    message: str,
    *,
    seq: int = 0,
    ts: int = 0,
    session_id: str = "",
    detail: Optional[str] = None,
) -> Envelope:
    """构造 `error` 帧：payload 含 `code`（机器可读）与 `message`（人可读）。"""
    payload: dict[str, Any] = {"code": str(code), "message": message}
    if detail is not None:
        payload["detail"] = detail
    return make_envelope(
        ServerType.ERROR, payload, seq=seq, ts=ts, session_id=session_id
    )


def state_changed(
    from_state: Optional[str],
    to_state: str,
    reason: str,
    *,
    seq: int = 0,
    ts: int = 0,
    session_id: str = "",
) -> Envelope:
    """构造 `state.changed` 帧（FSM 迁移，含 from/to/reason，方案 §2.4）。"""
    return make_envelope(
        ServerType.STATE_CHANGED,
        {"from": from_state, "to": to_state, "reason": reason},
        seq=seq,
        ts=ts,
        session_id=session_id,
    )


# ---------------------------------------------------------------- 信用窗口


class CreditWindow:
    """信用窗口（方案 §2.4 背压）：限制客户端"在途未消费"的媒体帧数。

    服务端每消费 N 帧回一个 `media.ack`，客户端窗口未满才继续发，防弱网堆积。
    窗口满时 `try_acquire()` 返回 False —— **由调用方决定**是丢弃、等待还是
    回 `credit_exhausted`，本类不替业务做兜底决策。

    Raises:
        ValueError: capacity < 1、n < 1、释放量超过在途量（绝不静默截断）。
    """

    __slots__ = ("_capacity", "_used")

    def __init__(self, capacity: int) -> None:
        if not isinstance(capacity, int) or isinstance(capacity, bool):
            raise ValueError(f"capacity 必须是整数，实际: {capacity!r}")
        if capacity < 1:
            raise ValueError(f"capacity 必须 >= 1，实际: {capacity}")
        self._capacity = capacity
        self._used = 0

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def used(self) -> int:
        """在途（已占用未释放）的帧数。"""
        return self._used

    @property
    def available(self) -> int:
        return self._capacity - self._used

    @property
    def full(self) -> bool:
        return self._used >= self._capacity

    def try_acquire(self, n: int = 1) -> bool:
        """尝试占用 n 个信用。窗口不足时返回 False 且**不占用任何**信用。"""
        if not isinstance(n, int) or isinstance(n, bool) or n < 1:
            raise ValueError(f"n 必须是 >= 1 的整数，实际: {n!r}")
        if self._used + n > self._capacity:
            return False
        self._used += n
        return True

    def release(self, n: int = 1) -> int:
        """释放 n 个信用（服务端已消费）。返回释放后的在途量。"""
        if not isinstance(n, int) or isinstance(n, bool) or n < 1:
            raise ValueError(f"n 必须是 >= 1 的整数，实际: {n!r}")
        if n > self._used:
            raise ValueError(
                f"释放量 {n} 超过在途量 {self._used}（重复消费，必须显式暴露）"
            )
        self._used -= n
        return self._used

    def release_all(self) -> int:
        """清空窗口，返回被释放的帧数。"""
        freed = self._used
        self._used = 0
        return freed

    def __repr__(self) -> str:  # pragma: no cover - 调试用途
        return f"CreditWindow(used={self._used}/{self._capacity})"
