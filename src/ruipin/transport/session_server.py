"""会话服务端循环：把 `Gateway` + `InterviewBridge` 挂到一个**连接对象**上（P0 闭环的最后一根线）。

为什么这一层必须存在
--------------------
到此为止各层的边界都是干净的：
* `Gateway` 认识帧、不认识业务（`Envelope -> list[Envelope]`，且是**同步**的）；
* `InterviewBridge` 认识业务、不认识字节；
* `SyncHandler` 把异步评分拆成"立即帧 + 稍后推送帧"。

但**没有任何一个模块负责"读连接、喂网关、把出来的帧写回连接"**。缺了它，
前端能启动却永远连不上后端——P0"文本面试闭环"就还差一环。

为什么不用 `websockets` 之类的库
--------------------------------
本层**只依赖一个 `WSConnection` Protocol**（`recv` / `send`），不导入任何 WS 库：

* 核心与传输的接线逻辑要能被**纯逻辑测试**（注入假连接，零网络、零端口、零等待）；
* 具体用哪家 WS 库属于"换 X 只改适配器"的范畴（方案 §1.4），不该长在这里；
* 真实库的差异（是否支持 `recv` 取消、关闭握手细节）不应该渗进业务循环。

接真实库时只需写一个 ~20 行的适配器：

```python
class WebsocketsConnection:                     # adapters 层，不在核心
    def __init__(self, ws): self._ws = ws
    async def recv(self): return await self._ws.recv()
    async def send(self, data): await self._ws.send(data)
```

循环设计（两个坑，都踩过）
--------------------------
1. **不做"收到帧才去取推送帧"**：评分是异步的，它的结果可能在任何时刻就绪。
   如果只在收到上行帧时 `drain()`，那么"客户端沉默但服务端有话说"（例如
   长时间评估完成）就会一直压在 outbox 里发不出去。因此循环是
   **"等待输入 + 超时轮询 outbox"** 两条腿。
2. **不用 `asyncio.wait_for(conn.recv(), timeout)`**：超时会**取消**在途的
   `recv()`，而不同 WS 库对"取消 recv"的容忍度不同（有的会丢帧）。
   这里改成"保活一个 `recv` 任务 + `asyncio.wait(timeout=...)`"，
   与 `orchestrator.scheduler._with_deadline` 同一套写法，一是不取消在途读，
   二是无输入时也能周期性把推送帧发出去。

纪律：任何异常都翻成结构化下游帧或明确关闭原因，**不吞**；关闭原因必须可读。
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol, Union, runtime_checkable

from ..domain.errors import RuipinError
from .bridge import InterviewBridge, SyncHandler, split_outgoing
from .gateway import (
    DEFAULT_ACK_EVERY,
    DEFAULT_CREDIT_CAPACITY,
    DEFAULT_IDLE_TIMEOUT_S,
    DEFAULT_RATE_LIMIT_PER_SEC,
    DownlinkSequencer,
    Gateway,
)
from .protocol import ErrorCode, Envelope, encode_text, error_envelope

#: 无输入时轮询 outbox 的间隔（秒）。越小推送越及时，代价是空转。
DEFAULT_OUTBOX_POLL_S: float = 0.05


class ConnectionClosed(RuipinError):
    """对端已关闭连接（正常关闭也算）。

    由 `WSConnection.recv` 抛出；服务端据此收尾，**不当成故障**。
    """

    def __init__(self, reason: str = "对端关闭") -> None:
        self.reason = reason
        super().__init__(f"连接已关闭: {reason}")


@runtime_checkable
class WSConnection(Protocol):
    """最小连接契约。真实 WS 库写一个薄适配器即可满足。

    `send` 必须接受 `str`（文本帧）与 `bytes`（二进制帧）两种入参——
    这是本系统"同一连接多路复用"的前提（方案 §2.4）。
    """

    async def recv(self) -> Union[str, bytes]: ...

    async def send(self, data: Union[str, bytes]) -> None: ...


class _DeferredHandler:
    """延迟绑定的 handler：**鉴权完成前绝不派发业务**。

    `Gateway` 构造时就要一个 handler，而真正的桥接器必须等鉴权拿到 session_id
    之后才能建（一场面试 = 一条连接）。这里用"先占位、后绑定"解掉这个
    先有鸡先有蛋的问题。

    为什么不"先随便给个 handler，反正业务层会检查权限"：那样"未认证的连接
    不触达业务"就只是**调用方纪律**；而现在它在结构上根本不成立——
    没绑定就等于什么都不派发，不需要业务层配合。
    """

    def __init__(self) -> None:
        self._inner: Optional[Callable[[Envelope], list[Envelope]]] = None

    def bind(self, handler: Callable[[Envelope], list[Envelope]]) -> None:
        self._inner = handler

    @property
    def bound(self) -> bool:
        return self._inner is not None

    def __call__(self, env: Envelope) -> list[Envelope]:
        if self._inner is None:
            return []
        return self._inner(env)


def _note_close(stats: "ConnectionStats", reason: str) -> None:
    """记录关闭原因，**只保留第一条**。

    为什么是"第一条"而不是"最后一条"：对端断开时，写失败与读失败往往**同时**发生，
    顺序由事件循环决定，但**写失败才是根因**，读失败只是它的后果。
    如果后来者覆盖前者，日志里就会留下"脚本耗尽/对端关闭"这种没有信息量的结论，
    排查时得从头猜。先到先得，让根因留在案发现场。
    """
    if not stats.closed_reason:
        stats.closed_reason = reason


@dataclass
class ConnectionStats:
    """一条连接的生命周期统计（用于对账与排查，不是业务数据）。"""

    session_id: str = ""
    authenticated: bool = False
    frames_in: int = 0
    frames_out: int = 0
    bytes_out: int = 0
    text_out: int = 0
    binary_out: int = 0
    closed_reason: str = ""
    errors: list[str] = field(default_factory=list)

    @property
    def closed(self) -> bool:
        return bool(self.closed_reason)


class SessionServer:
    """一条连接 ↔ 一个 `InterviewBridge` 的读写循环。

    Args:
        build_bridge: `session_id -> InterviewBridge`。每条连接一份桥接器
            （一场面试），由调用方决定怎么装配（repo / scheduler / 配置）。
        token_verifier: 令牌校验器，语义同 `Gateway`（返回 session_id 或 None）。
        clock: 时间源（秒）。默认 `time.monotonic`；测试注入确定性时钟。
        outbox_poll_s: 无输入时轮询推送帧的间隔。
        max_frames: 单连接最多处理多少帧（防跑飞的测试/攻击）；None 表示不限。
        tick_idle: 是否在轮询时调用 `gateway.tick()` 巡检空闲超时。
        rate_limit_per_sec / idle_timeout_s / credit_capacity / ack_every:
            转发给每条连接的 `Gateway`（语义见 `Gateway` 文档）。不传就用网关默认值。

    用法::

        server = SessionServer(build_bridge=..., token_verifier=...)
        stats = await server.serve(conn, token="一次性令牌")
    """

    def __init__(
        self,
        *,
        build_bridge: Callable[[str], InterviewBridge],
        token_verifier: Callable[[str], Optional[str]],
        clock: Callable[[], float] = time.monotonic,
        outbox_poll_s: float = DEFAULT_OUTBOX_POLL_S,
        max_frames: Optional[int] = None,
        tick_idle: bool = True,
        rate_limit_per_sec: int = DEFAULT_RATE_LIMIT_PER_SEC,
        idle_timeout_s: float = DEFAULT_IDLE_TIMEOUT_S,
        credit_capacity: int = DEFAULT_CREDIT_CAPACITY,
        ack_every: int = DEFAULT_ACK_EVERY,
    ) -> None:
        if outbox_poll_s <= 0:
            raise ValueError(f"outbox_poll_s 必须 > 0，实际: {outbox_poll_s}")
        if max_frames is not None and max_frames < 1:
            raise ValueError(f"max_frames 必须 >= 1 或为 None，实际: {max_frames}")
        self._build = build_bridge
        self._verify = token_verifier
        self._clock = clock
        self._poll_s = float(outbox_poll_s)
        self._max_frames = max_frames
        self._tick_idle = tick_idle
        # 横切参数转发给每条连接新建的 `Gateway`。**原先不传**：`Gateway` 上的
        # `rate_limit_per_sec` / `idle_timeout_s` / `credit_capacity` / `ack_every`
        # 四个可调项在真实服务里恒等于模块默认值，部署侧没有任何入口能改它们——
        # 参数看着存在，实际上不可达。这里逐个转发并保留 `Gateway` 自己的校验。
        self._cross_cutting = {
            "rate_limit_per_sec": rate_limit_per_sec,
            "idle_timeout_s": idle_timeout_s,
            "credit_capacity": credit_capacity,
            "ack_every": ack_every,
        }

    async def serve(self, conn: WSConnection, *, token: str) -> ConnectionStats:
        """跑完一条连接，返回统计。**不抛异常给调用方**（连接级的失败要能收尾）。"""
        stats = ConnectionStats()
        deferred = _DeferredHandler()
        gw = Gateway(
            handler=deferred,  # 先占位：下面鉴权成功后才绑定真正的桥接器
            clock=self._clock,
            token_verifier=self._verify,
            **self._cross_cutting,
        )
        # 本连接唯一的编号器：网关自产的帧与 outbox 里的推送帧共用它，
        # 保证下行的 seq 单调递增（前端靠它判丢帧/去重）。
        seq = gw.sequencer
        if not gw.authenticate(token):
            _note_close(stats, "认证失败")
            stats.errors.append("认证失败：令牌无效或已过期")
            assert not deferred.bound, "认证失败时绝不能碰业务层"
            await self._safe_send(
                conn,
                encode_text(
                    seq.stamp(
                        error_envelope(
                            ErrorCode.UNAUTHORIZED,
                            "认证失败：令牌无效或已过期",
                            ts=int(self._clock() * 1000),
                        )
                    )
                ),
                stats,
            )
            return stats

        session_id = gw.session_id or ""
        stats.session_id = session_id
        stats.authenticated = True

        bridge = self._build(session_id)
        handler = SyncHandler(bridge, asyncio.get_running_loop())
        deferred.bind(handler)

        recv_task = asyncio.ensure_future(conn.recv())
        try:
            while True:
                done, _ = await asyncio.wait({recv_task}, timeout=self._poll_s)

                # (1) 无论有没有新输入，都先把已就绪的推送帧发出去
                await self._flush(handler, conn, stats, seq)
                if self._tick_idle:
                    await self._send_envelopes(gw.tick(), conn, stats, seq)

                if recv_task not in done:
                    continue

                try:
                    raw = recv_task.result()
                except ConnectionClosed as exc:
                    _note_close(stats, exc.reason)
                    break
                except asyncio.CancelledError:
                    # 这里必须区分两种"取消"，混为一谈会出两个错：
                    # ① **本协程**被取消 -> CancelledError 从上面的 `await asyncio.wait` 抛出，
                    #    根本不经过这里。这种取消必须继续往外抛（吞掉它等于拒绝被取消，
                    #    asyncio 会认为任务没停）。所以这一支**不能**写成 `raise`。
                    # ② **recv_task 自身**被取消 -> 才会走到这里。这正是 WS 库在关闭
                    #    连接时取消在途读取的常态，属于**正常关闭**，不是故障：
                    #    按对端关闭收尾、不记 error、不抛给调用方。
                    _note_close(stats, "对端关闭: 读取被取消")
                    break
                except Exception as exc:  # noqa: BLE001 - 对端异常也要收尾
                    _note_close(stats, f"接收失败: {type(exc).__name__}")
                    stats.errors.append(str(exc))
                    break

                recv_task = asyncio.ensure_future(conn.recv())
                stats.frames_in += 1

                # (2) 分派：二进制走媒体面（含信用窗口），其余走文本面
                try:
                    if isinstance(raw, (bytes, bytearray)):
                        await self._send_envelopes(gw.handle_binary(bytes(raw)), conn, stats, seq)
                    elif isinstance(raw, str):
                        await self._send_envelopes(gw.handle_text(raw), conn, stats, seq)
                    else:
                        stats.errors.append(f"不认识的帧类型: {type(raw).__name__}")
                        await self._safe_send(
                            conn,
                            encode_text(
                                seq.stamp(
                                    error_envelope(
                                        ErrorCode.BAD_FRAME,
                                        f"不认识的帧类型: {type(raw).__name__}",
                                        ts=int(self._clock() * 1000),
                                    )
                                )
                            ),
                            stats,
                        )
                finally:
                    # 文本帧可能触发了异步评分，这里再取一次推送帧
                    await self._flush(handler, conn, stats, seq)

                if self._max_frames is not None and stats.frames_in >= self._max_frames:
                    _note_close(stats, f"达到单连接帧数上限 {self._max_frames}")
                    break
        finally:
            recv_task.cancel()
            try:
                await recv_task
            except (asyncio.CancelledError, ConnectionClosed, Exception):  # noqa: BLE001
                pass
            # 收干净在途任务，并把它们**最后的推送发出去**。
            # 少了最后这一步，最后一条指令（常常正是 `control.end`）产生的帧会永远
            # 留在 outbox 里——报告出不来，而且日志上一切正常，极难发现。
            await handler.joined()
            await self._flush(handler, conn, stats, seq)
            # 防御性兜底：当前所有退出路径都已通过 `_note_close` 写过原因，
            # 这一行是为将来新增的 break 路径准备的——它保证 `serve` 返回后
            # `stats.closed` 一定为真，调用方不需要再判空。
            if not stats.closed_reason:  # pragma: no cover - 当前无可达路径
                stats.closed_reason = "循环结束"

        return stats

    # ---------- 内部 ----------

    async def _flush(
        self,
        handler: SyncHandler,
        conn: WSConnection,
        stats: ConnectionStats,
        seq: DownlinkSequencer,
    ) -> None:
        pending = handler.drain()
        if pending:
            await self._send_envelopes(pending, conn, stats, seq)

    async def _send_envelopes(
        self,
        frames: list[Envelope],
        conn: WSConnection,
        stats: ConnectionStats,
        seq: DownlinkSequencer,
    ) -> None:
        """把下行信封补号后按文本/二进制分流写出（带 `opcode` 的走二进制）。

        编号必须在这里也做一遍：**推送帧不经过网关**（异步评分、TTS 音频块都直接
        落 `SyncHandler.outbox`），只有走到这个唯一出口才有机会补号。补号用的是
        `Gateway` 那一个实例（`gw.sequencer`），所以与网关自产的帧共用一条水位线，
        顺序不会倒挂。
        """
        texts, binaries = split_outgoing([seq.stamp(f) for f in frames])
        for payload in texts:
            await self._safe_send(conn, payload, stats)
            stats.text_out += 1
        for payload in binaries:
            await self._safe_send(conn, payload, stats)
            stats.binary_out += 1

    async def _safe_send(
        self, conn: WSConnection, payload: Union[str, bytes], stats: ConnectionStats
    ) -> None:
        """写失败（对端已断）只记原因，不抛——**不能因为发不出去就丢掉业务状态**。"""
        try:
            await conn.send(payload)
        except Exception as exc:  # noqa: BLE001 - 对端断开是常态，不是故障
            stats.errors.append(f"发送失败: {type(exc).__name__}: {exc}")
            _note_close(stats, f"发送失败: {type(exc).__name__}")
            return
        stats.frames_out += 1
        stats.bytes_out += len(payload)


__all__ = [
    "DEFAULT_OUTBOX_POLL_S",
    "ConnectionClosed",
    "ConnectionStats",
    "SessionServer",
    "WSConnection",
]
