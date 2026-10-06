"""会话级网关：一条 WebSocket 上的帧路由 + 传输层横切关注点。

六边形架构定位：网关属于**传输层**，不认识 `InterviewService`、不碰状态机。
业务通过注入的回调处理：

    handler: Callable[[Envelope], list[Envelope]]   # 一帧进，多帧出

网关只负责：解码 → 鉴权 → 限流 → seq 去重/排序 → 信用窗口背压 → 派发 →
超时巡检 → 降级/错误下行。**任何一步失败都产出结构化 `error` 帧**，不静默吞。

横切策略（全部可注入/可配置）
------------------------------
- 鉴权：`token_verifier: Callable[[str], Optional[str]]`，返回 session_id 或 None。
  默认 `_deny_all`（fail-closed：没配校验器 = 谁也不放行）。
  需鉴权的类型前缀：`AUTH_REQUIRED_PREFIXES`（session./answer./media./consent.）。
  `control.*` 交给业务层按 session_id 校验——网关不认识会话语义，不越权判断。
- 限流：每 session 每秒最多 `rate_limit_per_sec` 条（1 秒滑动窗口），超出回
  `rate_limited`。未绑定会话的消息计入 `__anon__` 桶，与会话桶互不影响。
- seq：语义见 `protocol` 模块 docstring（容忍窗口 `SEQ_TOLERANCE`）。
  重复 seq **回放上次响应**（重传幂等），不重复派发给业务。
  下行编号由 `DownlinkSequencer` 统一分配：网关自产的帧在构造时编号，业务帧在
  `_invoke` 返回时编号（这样 `_replay` 里缓存的就是已编号的帧），推送帧则由服务层
  用同一个实例补号。
- 背压：`CreditWindow(capacity)`。`auto_consume=True` 时 handler 同步返回即视为
  已消费（释放信用并按需回 `media.ack`）；真实异步管线可设 False，由业务消费完
  调用 `consume(n)` 释放。
- 超时：`tick(now)` 巡检空闲会话，产出 `session.timeout` 并标记过期；过期会话
  后续帧一律回 `session_timeout`。
- 降级：`send_degradation(level, reason)` 产出 `degradation.changed`，文案取自
  `orchestrator.degradation.badge_for`（方案 §3.4：禁止静默降级，必须显式告知）。

时钟：所有时间戳来自注入的 `clock: Callable[[], float]`（默认 `time.time`），
测试注入 `DeterministicClock`。**本模块不直接调用 `time.time()`。**
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import replace
from typing import Callable, Optional, Union

from ..domain.errors import RuipinError
from ..orchestrator.degradation import badge_for
from .protocol import (
    OPCODE_KEY,
    OPCODE_TYPES,
    SEQ_RETAIN,
    SEQ_TOLERANCE,
    CreditWindow,
    Envelope,
    ErrorCode,
    ProtocolError,
    ServerType,
    UnsupportedVersionError,
    decode_binary,
    decode_text,
    error_envelope,
    make_envelope,
)

#: 业务回调：一帧进，多帧出。
Handler = Callable[[Envelope], list[Envelope]]
#: 令牌校验器：合法则返回 session_id，否则 None。
TokenVerifier = Callable[[str], Optional[str]]

#: 未绑定会话的消息使用的限流桶名（鉴权前的 `control.*` 会落到这里）。
ANON_BUCKET = "__anon__"

#: 需要鉴权才能处理的类型前缀。
#: consent.grant 也在其中——它作用于已建立的会话，未鉴权放行等于绕过鉴权。
AUTH_REQUIRED_PREFIXES: tuple[str, ...] = (
    "session.",
    "answer.",
    "media.",
    "consent.",
)

DEFAULT_RATE_LIMIT_PER_SEC: int = 30
DEFAULT_IDLE_TIMEOUT_S: float = 120.0
DEFAULT_CREDIT_CAPACITY: int = 16
DEFAULT_ACK_EVERY: int = 4

#: 限流窗口长度（秒）。
RATE_WINDOW_S: float = 1.0


def _deny_all(token: str) -> Optional[str]:
    """默认校验器：拒绝一切（fail-closed）。没配校验器就不该放行。"""
    return None


class DownlinkSequencer:
    """下行文本帧的统一编号器：一条连接一个实例，保证 seq **单调递增**。

    为什么必须存在（实测发现的问题）
    --------------------------------
    原先只有网关**自己产生**的帧带 seq（`error` / `media.ack` / `session.timeout` /
    `degradation.changed`）；业务帧（`question.start` / `report.ready` / ...）从桥接器
    出来时 seq 恒为 0。前端按「seq > 0 才参与去重」实现，于是**业务帧的去重完全
    失效**：一次重传回放就会让同一道题在页面上出现两次。

    为什么编号点选在「业务帧返回时」而不是「写连接时」
    --------------------------------------------------
    `Gateway._replay` 缓存的是 `_invoke` 的返回值。若等到写连接时才编号，重传回放
    拿到的是**新的** seq，前端认不出这是同一帧，去重依旧失效。所以业务帧必须在
    `_invoke` 返回处编号——缓存里存的就是已编号的帧。
    推送帧（异步评分 / TTS 落 outbox 的那些）不经过 `_invoke`，由服务层用**同一个**
    实例在写出前补号；两边共用水位，故先后都是单调的。

    为什么二进制帧不编号
    --------------------
    媒体帧（TTS 音频 / viseme）编成一个裸字节帧（`[opcode][len][payload]`），
    **没有 seq 字段**：编号既发不出去，又会在可观测的文本 seq 上留下空洞。
    所以 `stamp` 对带 `opcode` 的信封原样返回，且**不推进水位**。
    """

    def __init__(self, start: int = 0) -> None:
        if not isinstance(start, int) or isinstance(start, bool) or start < 0:
            raise ValueError(f"start 必须是 >= 0 的整数，实际: {start!r}")
        self._seq = start

    @property
    def last(self) -> int:
        """当前水位（最后分配出去的编号）。"""
        return self._seq

    def next(self) -> int:
        """分配下一个编号。"""
        self._seq += 1
        return self._seq

    def stamp(self, env: Envelope) -> Envelope:
        """给未编号的文本帧补号；已编号的**保留原号**并把水位抬到它之上。

        已有编号的帧必须保留原号：它可能已经进了 `_replay`，改写会让重传回放的
        seq 与首次下发对不上，前端的去重就白做了。
        """
        if env.payload.get(OPCODE_KEY) is not None:
            return env  # 媒体帧走二进制通道，没有 seq 字段
        if env.seq > 0:
            if env.seq > self._seq:
                self._seq = env.seq
            return env
        return replace(env, seq=self.next())


class Gateway:
    """一条 WebSocket 连接的会话级网关。

    Args:
        handler: 业务回调（必填）。网关不认识业务，只做转发。
        clock: 时间源，默认 `time.time`；测试注入 `DeterministicClock.now`。
        token_verifier: 令牌校验器，默认拒绝一切。
        rate_limit_per_sec: 每会话每秒消息上限。
        idle_timeout_s: 空闲超时阈值（秒），`tick()` 巡检用。
        credit_capacity: 媒体上行信用窗口容量（在途帧数）。
        ack_every: 每消费多少帧回一个 `media.ack`。
        auto_consume: True 时 handler 返回即释放信用（同步管线）。
        sequencer: 下行编号器；None 时本连接自建一个。服务层要为推送帧补号时，
            应当用 `Gateway.sequencer` 拿**同一个**实例，不要另起计数器。

    Raises:
        ValueError: 参数不合法（限流 < 1、超时 <= 0、ack_every < 1 等）。
    """

    def __init__(
        self,
        handler: Handler,
        *,
        clock: Callable[[], float] = time.time,
        token_verifier: Optional[TokenVerifier] = None,
        rate_limit_per_sec: int = DEFAULT_RATE_LIMIT_PER_SEC,
        idle_timeout_s: float = DEFAULT_IDLE_TIMEOUT_S,
        credit_capacity: int = DEFAULT_CREDIT_CAPACITY,
        ack_every: int = DEFAULT_ACK_EVERY,
        auto_consume: bool = True,
        sequencer: Optional[DownlinkSequencer] = None,
    ) -> None:
        if rate_limit_per_sec < 1:
            raise ValueError(f"rate_limit_per_sec 必须 >= 1，实际: {rate_limit_per_sec}")
        if idle_timeout_s <= 0:
            raise ValueError(f"idle_timeout_s 必须 > 0，实际: {idle_timeout_s}")
        if ack_every < 1:
            raise ValueError(f"ack_every 必须 >= 1，实际: {ack_every}")

        self._handler = handler
        self._clock = clock
        self._verify: TokenVerifier = token_verifier or _deny_all
        self._rate_limit = rate_limit_per_sec
        self._idle_timeout = float(idle_timeout_s)
        self._ack_every = ack_every
        self._auto_consume = auto_consume
        self._credit = CreditWindow(credit_capacity)

        self.authenticated: bool = False
        self.session_id: Optional[str] = None

        self._seq = sequencer if sequencer is not None else DownlinkSequencer()
        self._last_seq: Optional[int] = None
        self._replay: dict[int, list[Envelope]] = {}
        self._hits: dict[str, deque[float]] = {}
        self._last_active: dict[str, float] = {}
        self._expired: set[str] = set()
        self._media_total = 0
        self._media_since_ack = 0

    # ------------------------------------------------------------ 只读视图

    @property
    def credit(self) -> CreditWindow:
        """信用窗口（观测用：在途帧数 / 容量）。"""
        return self._credit

    @property
    def expired_sessions(self) -> frozenset[str]:
        """已被判定空闲超时的会话（桶名，可能是 `ANON_BUCKET`）。"""
        return frozenset(self._expired)

    @property
    def sequencer(self) -> DownlinkSequencer:
        """下行编号器。

        服务层的写循环必须用**同一个**实例给推送帧补号：业务帧不经过本网关的
        `_invoke`，若另起一个计数器，两者会撞号或序号倒退——而前端的丢帧判断
        正是基于 seq 的单调性。
        """
        return self._seq

    # ------------------------------------------------------------ 鉴权

    def authenticate(self, token: str) -> bool:
        """校验令牌并绑定会话。

        校验器必须返回**非空** session_id，否则视为失败（空串是"看起来通过
        其实没通过"的典型坑，一律拒绝）。失败不改变已认证状态之外的任何字段。
        """
        sid = self._verify(token)
        if not isinstance(sid, str) or not sid:
            self.authenticated = False
            return False
        self.authenticated = True
        self.session_id = sid
        self._touch(sid)
        return True

    # ------------------------------------------------------------ 上行入口

    def handle_text(self, raw: Union[str, bytes]) -> list[Envelope]:
        """处理一条上行文本帧，返回要下行的一帧或多帧。

        解析失败不会抛给调用方（WS 连接要活着），而是产出 `error` 帧：
        `unsupported_version` / `bad_frame` / `unknown_type`。
        """
        try:
            env = decode_text(raw)
        except UnsupportedVersionError as exc:  # 先于 ProtocolError：它是其子类
            return [self._error(ErrorCode.UNSUPPORTED_VERSION, str(exc))]
        except ProtocolError as exc:
            return [self._error(ErrorCode.BAD_FRAME, str(exc))]

        if not env.is_known_type:
            return [
                self._error(
                    ErrorCode.UNKNOWN_TYPE,
                    f"未知消息类型: {env.type}（服务端版本可能落后于客户端）",
                    detail=env.type,
                )
            ]
        return self._dispatch(env)

    def handle_binary(self, raw: bytes) -> list[Envelope]:
        """处理一条上行媒体帧：解码 → 鉴权 → 限流 → 占用信用 → 派发 → 回 ack。

        二进制帧布局无 seq 字段，故不做 seq 去重（媒体允许乱序/丢帧，靠
        信用窗口与 `media.ack` 兜底）。
        """
        try:
            opcode, payload = decode_binary(raw)
        except ProtocolError as exc:
            return [self._error(ErrorCode.BAD_FRAME, str(exc))]

        if not self.authenticated:
            return [self._error(ErrorCode.UNAUTHORIZED, "未认证的媒体帧已被拒绝")]

        bucket = self._bucket_for(self.session_id)
        if bucket in self._expired:
            return [self._error(ErrorCode.SESSION_TIMEOUT, f"会话 {bucket} 已超时")]
        if not self._allow(bucket):
            return [self._error(ErrorCode.RATE_LIMITED, _rate_msg(self._rate_limit))]

        if not self._credit.try_acquire():
            return [
                self._error(
                    ErrorCode.CREDIT_EXHAUSTED,
                    "媒体上行窗口已满，请等待 media.ack 后再发",
                )
            ]

        env = make_envelope(
            OPCODE_TYPES[int(opcode)],
            {"opcode": int(opcode), "size": len(payload), "data": payload},
            seq=0,
            ts=self._now_ms(),
            session_id=self.session_id or "",
        )
        frames = self._invoke(env)
        if self._auto_consume:
            frames = frames + self.consume(1)
        self._touch(bucket)
        return frames

    def _dispatch(self, env: Envelope) -> list[Envelope]:
        """文本帧的公共管线：鉴权 → 超时 → 限流 → seq → 派发。"""
        if _requires_auth(env.type) and not self.authenticated:
            return [self._error(ErrorCode.UNAUTHORIZED, f"{env.type} 需要认证")]

        bucket = self._bucket_for(env.session_id or self.session_id)
        if bucket in self._expired:
            return [self._error(ErrorCode.SESSION_TIMEOUT, f"会话 {bucket} 已超时")]
        if not self._allow(bucket):
            return [self._error(ErrorCode.RATE_LIMITED, _rate_msg(self._rate_limit))]

        replay = self._seq_frames(env)
        if replay is not None:
            return replay

        frames = self._invoke(env)
        self._remember(env.seq, frames)
        self._touch(bucket)
        return frames

    # ------------------------------------------------------------ 背压 / ack

    def consume(self, n: int = 1) -> list[Envelope]:
        """服务端消费掉 n 帧媒体：释放信用，并在满 `ack_every` 时回 `media.ack`。

        真实异步媒体管线用这个入口（`auto_consume=False`）；同步管线由
        `handle_binary` 自动代劳。
        """
        self._credit.release(n)
        self._media_total += n
        self._media_since_ack += n
        out: list[Envelope] = []
        while self._media_since_ack >= self._ack_every:
            self._media_since_ack -= self._ack_every
            out.append(
                make_envelope(
                    ServerType.MEDIA_ACK,
                    {
                        "frames": self._ack_every,
                        "consumed": self._media_total,
                        "credits": self._credit.available,
                    },
                    seq=self._next_seq(),
                    ts=self._now_ms(),
                    session_id=self.session_id or "",
                )
            )
        return out

    # ------------------------------------------------------------ 下行构造

    def send_degradation(self, level: int, reason: str) -> list[Envelope]:
        """降级必须显式下行（方案 §3.4）：产出 `degradation.changed` 帧。

        文案取自 `orchestrator.degradation.badge_for`；level 越界时该函数抛
        ValueError —— 不吞、不降级为 Level 0，让调用方立刻看见配置错误。
        """
        badge = badge_for(level)
        if level == 0:
            message = "服务已恢复正常，全功能运行"
        else:
            message = f"服务已降级至 L{level}：{badge}（原因：{reason}）"
        return [
            make_envelope(
                ServerType.DEGRADATION_CHANGED,
                {
                    "level": level,
                    "reason": reason,
                    "badge": badge,
                    "message": message,
                },
                seq=self._next_seq(),
                ts=self._now_ms(),
                session_id=self.session_id or "",
            )
        ]

    def tick(self, now: Optional[float] = None) -> list[Envelope]:
        """空闲巡检：为超过 `idle_timeout_s` 未活动的会话产出 `session.timeout`。

        Args:
            now: 当前时间（秒）。None 时取注入的 clock。

        同一会话只报一次超时（已过期的跳过）。
        """
        t = self._clock() if now is None else float(now)
        out: list[Envelope] = []
        for bucket, last in self._last_active.items():
            if bucket in self._expired:
                continue
            idle = t - last
            if idle >= self._idle_timeout:
                self._expired.add(bucket)
                out.append(
                    make_envelope(
                        ServerType.SESSION_TIMEOUT,
                        {
                            "code": str(ErrorCode.SESSION_TIMEOUT),
                            "message": (
                                f"会话空闲 {idle:.1f} 秒（阈值 "
                                f"{self._idle_timeout:.0f} 秒），已超时"
                            ),
                            "idle_s": round(idle, 3),
                            "session_id": bucket,
                        },
                        seq=self._next_seq(),
                        ts=self._now_ms(),
                        session_id=bucket if bucket != ANON_BUCKET else "",
                    )
                )
        return out

    # ------------------------------------------------------------ 内部

    def _invoke(self, env: Envelope) -> list[Envelope]:
        """派发给业务。业务异常不炸连接，转成 `internal` 错误帧（前端可见）。

        业务返回的帧在这里**统一补下行序号**（见 `DownlinkSequencer`）。放在这里
        而不是"发送时"，是因为回放缓存 `_replay` 存的就是本函数的返回值：若等到
        发送时才编号，重传回放的帧会拿到**新的** seq，前端按 seq 去重就失效了。
        """
        try:
            frames = self._handler(env)
        except RuipinError as exc:
            return [self._error(ErrorCode.INTERNAL, f"{type(exc).__name__}: {exc}")]
        except Exception as exc:  # noqa: BLE001 - 显式转为 error 帧，不静默吞
            return [self._error(ErrorCode.INTERNAL, f"{type(exc).__name__}: {exc}")]
        return [self._seq.stamp(f) for f in frames] if frames else []

    def _seq_frames(self, env: Envelope) -> Optional[list[Envelope]]:
        """seq 去重/排序。返回 None 表示放行；否则返回要直接下行的帧。"""
        if self._last_seq is None:  # 首帧：任意 seq 作为基线
            self._last_seq = env.seq
            return None
        if env.seq in self._replay:  # 重传：回放，不重复处理
            return list(self._replay[env.seq])
        if env.seq <= self._last_seq:
            if self._last_seq - env.seq > SEQ_TOLERANCE:
                return [
                    self._error(
                        ErrorCode.OUT_OF_ORDER,
                        f"seq {env.seq} 落后当前 {self._last_seq} 超过容忍窗口 "
                        f"{SEQ_TOLERANCE}",
                    )
                ]
            return [
                self._error(
                    ErrorCode.DUPLICATE,
                    f"seq {env.seq} 已处理过，但响应已淘汰无法回放",
                )
            ]
        if env.seq > self._last_seq + SEQ_TOLERANCE:
            return [
                self._error(
                    ErrorCode.OUT_OF_ORDER,
                    f"seq {env.seq} 跳号超过容忍窗口 {SEQ_TOLERANCE}"
                    f"（当前 {self._last_seq}）",
                )
            ]
        self._last_seq = env.seq
        return None

    def _remember(self, seq: int, frames: list[Envelope]) -> None:
        """缓存最近 `SEQ_RETAIN` 条响应，供重传回放（内存有界）。"""
        self._replay[seq] = list(frames)
        while len(self._replay) > SEQ_RETAIN:
            self._replay.pop(next(iter(self._replay)))

    def _allow(self, bucket: str) -> bool:
        """1 秒滑动窗口限流。超限返回 False 且不占用配额。"""
        now = self._clock()
        hits = self._hits.setdefault(bucket, deque())
        while hits and now - hits[0] >= RATE_WINDOW_S:
            hits.popleft()
        if len(hits) >= self._rate_limit:
            return False
        hits.append(now)
        return True

    def _touch(self, bucket: str) -> None:
        self._last_active[bucket] = self._clock()

    def _bucket_for(self, session_id: Optional[str]) -> str:
        return session_id if session_id else ANON_BUCKET

    def _now_ms(self) -> int:
        return int(self._clock() * 1000)

    def _next_seq(self) -> int:
        return self._seq.next()

    def _error(
        self,
        code: ErrorCode,
        message: str,
        *,
        detail: Optional[str] = None,
    ) -> Envelope:
        return error_envelope(
            code,
            message,
            seq=self._next_seq(),
            ts=self._now_ms(),
            session_id=self.session_id or "",
            detail=detail,
        )


# ---------------------------------------------------------------- 模块级助手


def _requires_auth(mtype: str) -> bool:
    return mtype.startswith(AUTH_REQUIRED_PREFIXES)


def _rate_msg(limit: int) -> str:
    return f"消息速率超过每会话 {limit} 条/秒"


__all__ = [
    "ANON_BUCKET",
    "AUTH_REQUIRED_PREFIXES",
    "DEFAULT_ACK_EVERY",
    "DEFAULT_CREDIT_CAPACITY",
    "DEFAULT_IDLE_TIMEOUT_S",
    "DEFAULT_RATE_LIMIT_PER_SEC",
    "Gateway",
    "Handler",
    "TokenVerifier",
]
