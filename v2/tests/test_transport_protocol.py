"""传输层协议：信封与二进制帧的编解码（方案 §2.4）。

零外部依赖、纯函数，因此全部走穷举/边界断言；每个用例一句中文说明防的是什么回归。
"""

from __future__ import annotations

import json
import struct

import pytest

from ruipin.domain.errors import RuipinError
from ruipin.transport import protocol
from ruipin.transport.protocol import (
    CLIENT_TYPES,
    HEADER_SIZE,
    KNOWN_TYPES,
    MEDIA_IN,
    MEDIA_OUT,
    OPCODE_TYPES,
    PROTOCOL_VERSION,
    SEQ_RETAIN,
    SEQ_TOLERANCE,
    SERVER_TYPES,
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

# ---------------------------------------------------------------- 信封往返


def test_text_roundtrip_keeps_chinese_and_all_fields():
    """中文 payload 与全部字段必须逐字往返，防止编码/字段丢失。"""
    env = make_envelope(
        ServerType.QUESTION_START,
        {"turn_id": "t-7", "text": "请介绍一下你主导过的项目"},
        seq=42,
        ts=1770000000000,
        session_id="s-xxx",
    )
    raw = encode_text(env)
    back = decode_text(raw)
    assert back == env
    assert back.payload["text"] == "请介绍一下你主导过的项目"
    assert back.ts == 1770000000000
    assert back.seq == 42


def test_encode_text_emits_spec_field_order():
    """线序字典的键顺序与方案示例一致，防止下游按位置解析出错。"""
    env = make_envelope(ClientType.ANSWER_TEXT, {"a": 1}, seq=1, ts=2, session_id="s")
    assert list(json.loads(encode_text(env)).keys()) == [
        "v",
        "type",
        "seq",
        "ts",
        "session_id",
        "payload",
    ]


def test_decode_text_accepts_bytes_frame():
    """WS 框架常给 bytes，不能只认 str。"""
    raw = encode_text(make_envelope(ClientType.ANSWER_TEXT, {}, seq=1, ts=0))
    assert decode_text(raw.encode("utf-8")).type == "answer.text"


def test_decode_text_accepts_bytearray_frame():
    """bytearray 也是合法输入，不能因为类型判断过窄而误判坏帧。"""
    raw = encode_text(make_envelope(ClientType.ANSWER_TEXT, {}, seq=1, ts=0))
    assert decode_text(bytearray(raw.encode("utf-8"))).seq == 1


def test_decode_text_rejects_non_utf8_bytes():
    """非法 UTF-8 必须抛 BadFrameError，不能替换字符后继续解析。"""
    with pytest.raises(BadFrameError, match="UTF-8"):
        decode_text(b"\xff\xfe\x00")


def test_decode_text_rejects_non_string_input():
    """传 int/list 之类的类型必须报错，不能静默当空帧处理。"""
    with pytest.raises(BadFrameError, match="必须是 str/bytes"):
        decode_text(123)


def test_decode_text_rejects_broken_json():
    """半截 JSON 必须抛错，不能返回半截信封。"""
    with pytest.raises(BadFrameError, match="不是合法 JSON"):
        decode_text('{"v":1,"type":')


def test_decode_text_rejects_json_array():
    """JSON 数组不是信封，必须报错。"""
    with pytest.raises(BadFrameError, match="必须是 JSON 对象"):
        decode_text("[1,2,3]")


@pytest.mark.parametrize("missing", ["v", "type", "seq", "ts"])
def test_decode_text_requires_core_fields(missing):
    """四个核心字段缺一即坏帧，防止"缺字段就取默认值"的静默兜底。"""
    data = {"v": 1, "type": "answer.text", "seq": 1, "ts": 0}
    data.pop(missing)
    with pytest.raises(BadFrameError, match=f"缺少 {missing} 字段"):
        decode_text(json.dumps(data))


@pytest.mark.parametrize("bad_v", [True, "1", 1.0, None])
def test_decode_text_rejects_non_integer_version(bad_v):
    """v 必须是整数：True 是 int 子类但语义上是错的，必须挡住。"""
    data = {"v": bad_v, "type": "answer.text", "seq": 1, "ts": 0}
    with pytest.raises(BadFrameError, match="v 必须是整数"):
        decode_text(json.dumps(data))


@pytest.mark.parametrize("bad_version", [0, 2, 99])
def test_unsupported_version_raises(bad_version):
    """未知版本必须抛出明确异常（而不是被当成 v1 悄悄解析）。"""
    data = {"v": bad_version, "type": "answer.text", "seq": 1, "ts": 0}
    with pytest.raises(UnsupportedVersionError) as excinfo:
        decode_text(json.dumps(data))
    assert excinfo.value.version == bad_version
    assert str(bad_version) in str(excinfo.value)


def test_supported_versions_contains_current_protocol_version():
    """PROTCOL_VERSION 必须在受支持集合内，防止改版本号忘了登记。"""
    assert PROTOCOL_VERSION in SUPPORTED_VERSIONS
    assert SUPPORTED_VERSIONS == frozenset({1})


def test_unknown_type_is_tolerated_not_raised():
    """前端可能比后端新：未知类型解码放行，由网关回 error（宽容路径）。"""
    data = {"v": 1, "type": "question.hologram", "seq": 1, "ts": 0}
    env = decode_text(json.dumps(data))
    assert env.type == "question.hologram"
    assert env.is_known_type is False


def test_known_type_flag_for_spec_types():
    """方案表格里的类型必须都被判为已知，防止漏登记。"""
    assert make_envelope(ClientType.SESSION_CREATE).is_known_type is True
    assert make_envelope(ServerType.DEGRADATION_CHANGED).is_known_type is True


@pytest.mark.parametrize(
    "bad",
    [
        {"type": ""},
        {"type": 12},
        {"seq": True},
        {"seq": -1},
        {"seq": "1"},
        {"ts": "x"},
        {"ts": True},
        {"session_id": 5},
        {"payload": [1, 2]},
    ],
)
def test_decode_text_rejects_bad_field_types(bad):
    """字段类型不对一律坏帧；特别是 bool 冒充 int 与负 seq。"""
    data = {"v": 1, "type": "answer.text", "seq": 1, "ts": 0}
    data.update(bad)
    with pytest.raises(BadFrameError):
        decode_text(json.dumps(data))


def test_decode_text_defaults_for_optional_fields():
    """session_id / payload 可省略，缺省为空串与空 dict（不是 None）。"""
    env = decode_text(json.dumps({"v": 1, "type": "answer.text", "seq": 3, "ts": 0}))
    assert env.session_id == ""
    assert env.payload == {}


def test_decode_text_tolerates_explicit_null_optional_fields():
    """显式 null 的 session_id/payload 归一为空值，不让 None 渗进业务层。"""
    env = decode_text(
        json.dumps(
            {
                "v": 1,
                "type": "answer.text",
                "seq": 3,
                "ts": 0,
                "session_id": None,
                "payload": None,
            }
        )
    )
    assert env.session_id == ""
    assert env.payload == {}


def test_encode_text_rejects_media_envelope_with_bytes_payload():
    """媒体帧的 payload 含 bytes，不可走文本帧——必须报错而不是产出半截 JSON。"""
    env = make_envelope(ClientType.MEDIA_AUDIO, {"data": b"\x01\x02"})
    with pytest.raises(BadFrameError, match="无法序列化为 JSON"):
        encode_text(env)


def test_envelope_from_dict_equals_decode_text():
    """`Envelope.from_dict` 与 `decode_text` 必须走同一套校验，不能分叉。"""
    data = {"v": 1, "type": "answer.commit", "seq": 9, "ts": 5, "payload": {"a": 1}}
    assert Envelope.from_dict(data) == decode_text(json.dumps(data))


def test_envelope_to_dict_roundtrip_through_from_dict():
    """to_dict → from_dict 往返相等，保证下行帧可被回放/重放。"""
    env = make_envelope(ServerType.EVAL_DONE, {"score": 88.5}, seq=7, ts=1)
    assert Envelope.from_dict(env.to_dict()) == env


# ---------------------------------------------------------------- 消息类型表


def test_client_types_match_spec_table():
    """客户端→服务端类型表与方案 §2.4 完全一致，防止改名导致前后端对不上。"""
    assert CLIENT_TYPES == frozenset(
        {
            "session.create",
            "consent.grant",
            "answer.text",
            "answer.commit",
            "control.barge_in",
            "control.skip",
            "control.end",
            "media.audio",
            "media.video",
            "media.screen",
            # 方案 §2.4 的原表漏了端侧生理上行；模块详设 §9 的 HRBatch 要求
            # 「端侧 → 服务端（每 2 s 一条，批量上传）」才有这条。见 protocol.py 注释。
            "physio.batch",
        }
    )


def test_server_types_cover_spec_table_plus_transport_frames():
    """服务端类型表：方案表格 + 传输层补充（media.ack / session.timeout）。"""
    assert SERVER_TYPES >= frozenset(
        {
            "state.changed",
            "question.start",
            "question.tts_chunk",
            "avatar.viseme",
            "transcript.partial",
            "transcript.final",
            "eval.done",
            "metric.hr",
            "degradation.changed",
            "report.ready",
            "error",
        }
    )
    assert "media.ack" in SERVER_TYPES
    assert "session.timeout" in SERVER_TYPES


def test_known_types_is_union_and_disjoint_groups():
    """已知类型 = 上行 ∪ 下行；两组无交集，防止上下行类型串台。"""
    assert KNOWN_TYPES == CLIENT_TYPES | SERVER_TYPES
    assert not (CLIENT_TYPES & SERVER_TYPES)


# ---------------------------------------------------------------- 错误码


def test_error_code_enum_values_are_stable_wire_strings():
    """code 是前后端契约，改名即破坏前端分支，锁死字面值。"""
    assert [str(c) for c in ErrorCode] == [
        "bad_frame",
        "unsupported_version",
        "unknown_type",
        "unauthorized",
        "rate_limited",
        "out_of_order",
        "duplicate",
        "session_timeout",
        "credit_exhausted",
        "internal",
    ]


def test_transport_errors_inherit_ruipin_error():
    """传输层异常必须能被上层按 RuipinError 统一捕获，不能漏出裸 Exception。"""
    for exc in (
        BadFrameError("x"),
        UnsupportedVersionError(7),
        UnknownOpcodeError(0x7F),
    ):
        assert isinstance(exc, ProtocolError)
        assert isinstance(exc, RuipinError)


def test_error_envelope_payload_shape():
    """error 帧 payload 必须含机器可读 code + 人可读 message。"""
    env = error_envelope(ErrorCode.RATE_LIMITED, "太快了", seq=3, ts=10)
    assert env.type == "error"
    assert env.payload == {"code": "rate_limited", "message": "太快了"}
    assert env.seq == 3 and env.ts == 10


def test_error_envelope_appends_detail_only_when_given():
    """detail 是可选补充信息，缺省时不应出现空字段。"""
    assert "detail" not in error_envelope(ErrorCode.BAD_FRAME, "m").payload
    assert error_envelope(ErrorCode.BAD_FRAME, "m", detail="x").payload["detail"] == "x"


def test_state_changed_payload_has_from_to_reason():
    """FSM 迁移帧必须带 from/to/reason，否则前端无法呈现「为什么跳到这」。"""
    env = state_changed("LISTENING", "PROCESSING", "commit", seq=1, ts=2)
    assert env.type == "state.changed"
    assert env.payload == {"from": "LISTENING", "to": "PROCESSING", "reason": "commit"}


def test_make_envelope_defaults_and_copies_payload():
    """默认 seq/ts 为 0、payload 被拷贝，防止外部改 dict 污染信封（frozen 假象）。"""
    payload = {"a": 1}
    env = make_envelope("answer.text", payload)
    payload["b"] = 2
    assert env.seq == 0 and env.ts == 0 and env.session_id == "" and env.v == 1
    assert env.payload == {"a": 1}


# ---------------------------------------------------------------- opcode


def test_opcode_tables_values_and_direction():
    """opcode 表：0x01/0x02/0x03 上行，0x11/0x12 下行，两组不重叠。"""
    assert dict(MEDIA_IN) == {0x01: "media.audio", 0x02: "media.video", 0x03: "media.screen"}
    assert dict(MEDIA_OUT) == {0x11: "question.tts_chunk", 0x12: "avatar.viseme"}
    assert not (set(MEDIA_IN) & set(MEDIA_OUT))
    assert set(OPCODE_TYPES) == set(MEDIA_IN) | set(MEDIA_OUT)


def test_opcode_tables_are_read_only():
    """opcode 表是全局契约，运行期被改动会让编解码行为漂移。"""
    with pytest.raises(TypeError):
        MEDIA_IN[0x09] = "x"  # type: ignore[index]


def test_unknown_opcode_error_message_names_the_opcode():
    """未知 opcode 必须指名道姓，排查弱网乱码帧时才有价值。"""
    exc = UnknownOpcodeError(0x7F)
    assert exc.opcode == 0x7F
    assert "0x7F" in str(exc)


# ---------------------------------------------------------------- 二进制帧


@pytest.mark.property
@pytest.mark.parametrize("opcode", [0x01, 0x02, 0x03, 0x11, 0x12])
@pytest.mark.parametrize(
    "payload",
    [
        b"",
        b"\x00",
        b"\xff\xfe\xfd",
        "中文音频块".encode("utf-8"),
        bytes(range(256)),
        b"a" * 1024,
    ],
)
def test_binary_roundtrip_is_lossless(opcode, payload):
    """属性测试：任意 opcode × 任意负载（含 0 长度与非 ASCII）必须逐字节往返。"""
    frame = encode_binary(opcode, payload)
    got_op, got_payload = decode_binary(frame)
    assert int(got_op) == opcode
    assert got_payload == payload


def test_binary_frame_header_layout():
    """布局锁定：第 0 字节 opcode，1–4 字节大端长度，第 5 字节起是负载。"""
    frame = encode_binary(Opcode.MEDIA_SCREEN, b"abcd")
    assert len(frame) == HEADER_SIZE + 4
    assert frame[0] == 0x03
    assert struct.unpack(">I", frame[1:5])[0] == 4
    assert frame[5:] == b"abcd"


def test_binary_roundtrip_accepts_bytearray_and_memoryview():
    """编码器接受字节视图，解码器接受 bytearray——适配真实 WS 库的各种返回。"""
    assert decode_binary(bytearray(encode_binary(Opcode.TTS_AUDIO, b"z"))) == (
        Opcode.TTS_AUDIO,
        b"z",
    )
    assert encode_binary(Opcode.VISEME, memoryview(b"vm")) == encode_binary(
        Opcode.VISEME, b"vm"
    )


def test_encode_binary_rejects_unknown_opcode():
    """编码未知 opcode 必须报错，不能写出一个对端解不开的帧。"""
    with pytest.raises(UnknownOpcodeError) as excinfo:
        encode_binary(0x7F, b"x")
    assert excinfo.value.opcode == 0x7F


def test_encode_binary_rejects_non_integer_opcode():
    """字符串之类的 opcode 必须报 BadFrameError，而不是抛裸 TypeError。"""
    with pytest.raises(BadFrameError, match="opcode 必须是整数"):
        encode_binary("audio", b"x")  # type: ignore[arg-type]


def test_encode_binary_rejects_non_bytes_payload():
    """文本负载必须被挡住，防止把 str 悄悄 encode 成不确定的字节序。"""
    with pytest.raises(BadFrameError, match="payload 必须是 bytes"):
        encode_binary(Opcode.MEDIA_AUDIO, "abc")  # type: ignore[arg-type]


def test_encode_binary_rejects_oversized_payload(monkeypatch):
    """超过 4 字节长度上限必须报错，不能截尾写入导致对端长度校验失败。"""
    monkeypatch.setattr(protocol, "MAX_PAYLOAD_BYTES", 4)
    with pytest.raises(BadFrameError, match="过长"):
        encode_binary(Opcode.MEDIA_AUDIO, b"12345")


def test_decode_binary_rejects_non_bytes_input():
    """传 str 进来必须报错，不能当成空负载。"""
    with pytest.raises(BadFrameError, match="必须是 bytes"):
        decode_binary("not-bytes")  # type: ignore[arg-type]


def test_decode_binary_rejects_short_frame():
    """短于 5 字节头部的帧必须报错，不能按 0 长度解释。"""
    with pytest.raises(BadFrameError, match="过短"):
        decode_binary(b"\x01\x00\x00")


def test_decode_binary_rejects_length_mismatch():
    """长度字段与实际负载不符（截断/粘连）必须报错，不做「有多少用多少」。"""
    frame = encode_binary(Opcode.MEDIA_AUDIO, b"abcdef")
    with pytest.raises(BadFrameError, match="不符"):
        decode_binary(frame[:-1])


def test_decode_binary_rejects_unknown_opcode_before_length_check():
    """身份先于内容：未知 opcode 优先报 UnknownOpcodeError，便于定位对端版本。"""
    with pytest.raises(UnknownOpcodeError) as excinfo:
        decode_binary(struct.pack(">BI", 0x2A, 0))
    assert excinfo.value.opcode == 0x2A


def test_opcode_maps_to_message_type_for_gateway():
    """网关靠 OPCODE_TYPES 把二进制帧翻译成消息类型，映射必须完整。"""
    assert OPCODE_TYPES[0x01] == "media.audio"
    assert OPCODE_TYPES[0x11] == "question.tts_chunk"


# ---------------------------------------------------------------- 信用窗口


def test_credit_window_fresh_state():
    """初始窗口：在途 0、可用 = 容量。"""
    w = CreditWindow(4)
    assert w.capacity == 4 and w.used == 0 and w.available == 4
    assert w.full is False


def test_credit_window_acquire_until_full_then_refuses():
    """窗口满时必须拒绝（弱网背压的核心），且拒绝时不占用信用。"""
    w = CreditWindow(2)
    assert w.try_acquire() is True
    assert w.try_acquire() is True
    assert w.full is True
    assert w.try_acquire() is False
    assert w.used == 2


def test_credit_window_release_restores_capacity():
    """服务端消费后必须恢复可用额度，否则客户端会永久停摆。"""
    w = CreditWindow(2)
    w.try_acquire()
    w.try_acquire()
    assert w.release() == 1
    assert w.available == 1
    assert w.try_acquire() is True


def test_credit_window_acquire_batch_is_all_or_nothing():
    """批量占用不足时必须一个都不占，避免部分占用导致计数漂移。"""
    w = CreditWindow(3)
    assert w.try_acquire(2) is True
    assert w.try_acquire(2) is False
    assert w.used == 2
    assert w.available == 1


def test_credit_window_release_all_reports_freed_count():
    """清空窗口返回释放数量，供 ack 里告知客户端可继续发送。"""
    w = CreditWindow(5)
    w.try_acquire(3)
    assert w.release_all() == 3
    assert w.used == 0 and w.available == 5


@pytest.mark.parametrize("capacity", [0, -1])
def test_credit_window_rejects_non_positive_capacity(capacity):
    """容量 <= 0 是配置错误，必须显式报错而不是「永远满/永远空」。"""
    with pytest.raises(ValueError, match="capacity"):
        CreditWindow(capacity)


def test_credit_window_rejects_non_integer_capacity():
    """容量必须是整数，防止浮点容量让计数比较失真。"""
    with pytest.raises(ValueError, match="capacity 必须是整数"):
        CreditWindow(2.5)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_n", [0, -1, 1.5, True])
def test_credit_window_rejects_bad_n(bad_n):
    """占用/释放数量非法必须报错，绝不静默当 1 处理。"""
    w = CreditWindow(4)
    with pytest.raises(ValueError, match="n 必须是"):
        w.try_acquire(bad_n)
    with pytest.raises(ValueError, match="n 必须是"):
        w.release(bad_n)


def test_credit_window_rejects_over_release():
    """重复消费（释放超过在途）是计数 bug，必须炸出来而不是把窗口撑大。"""
    w = CreditWindow(4)
    w.try_acquire()
    with pytest.raises(ValueError, match="超过在途量"):
        w.release(2)
    assert w.used == 1


def test_seq_semantics_constants_are_consistent():
    """去重缓存必须比容忍窗口小，否则"缓存已淘汰"分支永远走不到。"""
    assert SEQ_TOLERANCE >= 1
    assert 0 <= SEQ_RETAIN < SEQ_TOLERANCE
