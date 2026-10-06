"""真实 WebSocket 库适配器 + 服务端入口（P0 闭环接上真实网络的最后一块）。

职责边界（刻意很窄）
--------------------
本模块**只做三件事**，任何一件越界都会把业务逻辑拖进传输层：

1. **连接适配**：把 `websockets` 的连接对象包成 `SessionServer` 要求的
   `WSConnection`（`recv` / `send`），并把库的 `ConnectionClosed` 翻译成
   我们自己的 `ConnectionClosed`。就这么多，没有重连、没有心跳业务、没有缓冲。
2. **握手取令牌**：从 HTTP 握手请求里取出一次性令牌。**不做鉴权**——
   鉴权只有一个点（`Gateway.authenticate`），在适配器里再判一次必然出现
   "两处判断不一致、其中一处先放行"的经典漏洞。
3. **装配与启动**：`serve_forever` / `start_server` 把 `SessionServer` 挂到端口上。

为什么延迟导入 `websockets`
---------------------------
本模块被 `adapters/__init__` 统一导出，而**导入包不该因为缺一个可选依赖就整体炸掉**。
所以顶部不 `import websockets`，改由 `_import_websockets()` 在真正要用时导入，
并把 `ImportError` 翻成带安装指引的 `Unavailable`——排查时省一次"翻代码找依赖名"。

版本与 API 形态（实测，不是照抄文档）
------------------------------------
`websockets 17.x` 只有新版 asyncio API（`websockets.asyncio.server`），
`websockets.legacy` **已被移除**。因此这里的写法在 13 以下的版本上不成立。
关键访问路径（实测确认）：
* `connection.request.path` → `'/ws?token=abc'`（**含 query 的原始路径**）
* `connection.request.headers` → 可用 `.get("Authorization")`
* `websockets.serve(handler, host, port, max_size=...)`，`Server.sockets[0].getsockname()`
* 异常 `websockets.exceptions.ConnectionClosed`，带 `.code` / `.reason`

红线
----
一个**开发用**的装配（`build_dev_server`）就写在本文件里，它用假评估器 + 内存存储。
启动时会把这三件事**打在日志里**（假评估器 / 内存存储 / 静态令牌），
因为"把开发服务当成能用的服务"是本项目最贵的一类误解。
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import secrets
import sys
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence, Union
from urllib.parse import parse_qs, urlsplit

from ..domain.errors import Unavailable
from ..orchestrator import TurnScheduler
from ..transport.bridge import BridgeConfig, InterviewBridge
from ..transport.gateway import (
    DEFAULT_ACK_EVERY,
    DEFAULT_CREDIT_CAPACITY,
    DEFAULT_IDLE_TIMEOUT_S,
    DEFAULT_RATE_LIMIT_PER_SEC,
)
from ..transport.session_server import (
    ConnectionClosed,
    ConnectionStats,
    SessionServer,
    WSConnection,
)

#: 可选依赖名（写进错误文案，避免"未安装 XXX"里 XXX 拼错）。
WEBSOCKETS_PACKAGE = "websockets"

#: 单条 WS 消息的字节上限（防对端一次塞爆内存）。
#:
#: **这是个待标定的数，不是推导出来的**：协议层的 `MAX_PAYLOAD_BYTES` 是 4 GiB
#: （纯整数溢出防护，等于没有约束），所以**第一道也是唯一一道实际闸门在这里**。
#: 取 1 MiB 是沿用 `websockets` 的默认值；真正的合理值取决于上行媒体帧的实测大小
#: （方案 §6.3 的 Opus 帧、端侧 rPPG 的上行数据），未标定前不要设成 `None`——
#: 那等于把这个限制交给对端决定。
DEFAULT_MAX_MESSAGE_BYTES: int = 1 << 20

DEFAULT_HOST = "127.0.0.1"  # 只绑本地：明文 ws 对外监听会把面试内容裸奔在局域网
#: 默认端口。**不用 8000**：8000 是最常被别的服务抢的端口之一（本机实测已被占用，
#: 撞上时表现为"服务起不来"或更糟的"连上了别人的服务"）。
#: 8787 不在任何常见默认清单里（8000/8080/3000/5000/9000/5173/4321），本机实测空闲。
#: 前端 `vite.config.ts` 的 proxy 与 `App.tsx` 的兜底 URL 都指向这个值，
#: `tests/test_adapters_ws_server.py` 有一处用例专门守住"两侧不许漂移"。
DEFAULT_PORT = 8787
DEFAULT_PATH = "/ws"  # 与前端 `web/app` 拼的 URL 对齐

#: 关帧状态码。1000/1001 是对端**正常**下线，其余按异常处理（方便排查时一眼分拣）。
_WS_NORMAL_CLOSE_CODES = frozenset({1000, 1001})

_LOGGER = logging.getLogger("ruipin.ws_server")


# ---------------------------------------------------------------- 依赖导入


def websockets_installed() -> bool:
    """`websockets` 是否可用。测试用它决定要不要跳过真实网络用例。"""
    try:
        import websockets  # noqa: F401
    except ImportError:
        return False
    return True


def _import_websockets() -> tuple[Any, Any, type]:
    """延迟导入，返回 `(websockets, serve, ConnectionClosed)`。

    Raises:
        Unavailable: 未安装。文案里带可直接执行的安装命令。
    """
    try:
        import websockets
        from websockets.asyncio.server import serve
        from websockets.exceptions import ConnectionClosed as WSConnectionClosed
    except ImportError as exc:  # pragma: no cover - 依赖已装时不走
        raise Unavailable(
            WEBSOCKETS_PACKAGE,
            f"未安装，请执行：pip install {WEBSOCKETS_PACKAGE}",
        ) from exc
    return websockets, serve, WSConnectionClosed


# ---------------------------------------------------------------- 握手取令牌


def token_from_path(raw_path: str) -> str:
    """从握手路径里取 `?token=`。取不到返回**空串**（不是 None）。

    空串会走 `Gateway.authenticate` 的"令牌无效"分支——于是"没带令牌"与
    "令牌错"共用同一个鉴权点，适配器里不需要再写一遍判空。
    """
    try:
        query = urlsplit(raw_path).query
    except ValueError:  # 畸形 URL（例如缺 scheme 的怪串）不该炸掉握手
        return ""
    values = parse_qs(query, keep_blank_values=False).get("token") or []
    return values[0] if values else ""


def token_from_request(request: Any) -> str:
    """取一次性令牌。**优先 `Authorization: Bearer`，其次 `?token=`**。

    为什么 header 优先：URL 会进反向代理日志、浏览器历史、Referer，
    令牌泄了等于会话被顶替（本项目令牌是一次性的，泄露即被消费）。
    query 参数留着只是为了让 `websocat 'ws://.../ws?token=x'` 这种命令行调试可行。
    """
    headers = getattr(request, "headers", None)
    if headers is not None:
        raw = headers.get("Authorization") or ""
        if isinstance(raw, str) and raw[:7].lower() == "bearer ":
            token = raw[7:].strip()
            if token:
                return token
    return token_from_path(getattr(request, "path", "") or "")


def path_matches(raw_path: str, expected: str) -> bool:
    """路径是否匹配（忽略 query）。"""
    try:
        return urlsplit(raw_path).path == expected
    except ValueError:
        return False


# ---------------------------------------------------------------- 连接适配


def _describe_close(exc: BaseException) -> str:
    """把库的关闭异常翻成**可读且可分拣**的一句中文。

    区分正常/异常关闭不是洁癖：面试中断的原因如果只记"连接已关闭"，
    排查时完全无法区分"候选人正常点了结束"与"网络断了"。
    """
    code = getattr(exc, "code", None)
    reason = getattr(exc, "reason", "") or ""
    if code is None:
        return "对端关闭"
    label = "对端正常关闭" if int(code) in _WS_NORMAL_CLOSE_CODES else "对端异常关闭"
    return f"{label}({int(code)}{': ' + str(reason) if reason else ''})"


class WebsocketsConnection:
    """`WSConnection` 的 `websockets` 实现（薄适配器）。

    Args:
        connection: `websockets` 的 `ServerConnection`（或其他同形的对象）。
        closed_types: 视为"对端关闭"的异常类型；默认从库里取。
            显式可注入是为了让本文件在**没装 `websockets`** 的环境下也能被测试。
    """

    def __init__(
        self,
        connection: Any,
        *,
        closed_types: Optional[tuple[type, ...]] = None,
    ) -> None:
        self._ws = connection
        # 注意这里**不能**写成 `(x if cond else y,)`：那个尾随逗号会把元组再包一层
        # （`((FakeClosed,),) `），于是 `except` 拿到的是元组而不是异常类，
        # 运行时报 `TypeError: catching classes that do not inherit from BaseException`。
        if closed_types is not None:
            self._closed_types: tuple[type, ...] = closed_types
        else:
            self._closed_types = (_import_websockets()[2],)
        self.closed_reason = ""

    async def recv(self) -> Union[str, bytes]:
        """收一帧；对端关闭时抛**我们自己的** `ConnectionClosed`（不是库的）。

        翻译而不是直接外抛库异常，是为了让 `SessionServer` 不必认识任何 WS 库
        （换库时这一行是唯一要改的地方）。
        """
        try:
            return await self._ws.recv()
        except self._closed_types as exc:
            self.closed_reason = _describe_close(exc)
            raise ConnectionClosed(self.closed_reason) from exc

    async def send(self, data: Union[str, bytes]) -> None:
        """发一帧。文本帧传 `str`，二进制帧传 `bytes`——同一个 `send` 多路复用。"""
        await self._ws.send(data)

    async def close(self, code: int = 1000, reason: str = "") -> None:
        """主动关闭（供拒绝握手等场景使用）。"""
        await self._ws.close(code, reason)


# ---------------------------------------------------------------- 服务端


async def _safe_close(connection: Any, code: int, reason: str) -> None:
    """尽力关闭对端；关不掉只记日志（此时连接已经坏了，抛出去没有意义）。"""
    try:
        await connection.close(code, reason)
    except Exception as exc:  # noqa: BLE001
        _LOGGER.warning("关闭连接失败（对端可能已消失）: %s: %s", type(exc).__name__, exc)


def _make_handler(
    server: SessionServer,
    *,
    path: str,
    on_stats: Optional[Callable[[ConnectionStats], None]],
    logger: logging.Logger,
    connection_factory: Callable[[Any], WSConnection] = WebsocketsConnection,
) -> Callable[[Any], Any]:
    """构造 `websockets` 需要的 handler 协程函数。"""

    async def handler(connection: Any) -> None:
        request = getattr(connection, "request", None)
        raw_path = getattr(request, "path", "") or ""
        if not path_matches(raw_path, path):
            # 不匹配就明确拒掉。静默接受等于把所有路径都当成了 /ws。
            logger.warning("拒绝未知路径的握手: %r（服务只挂在 %s）", raw_path, path)
            await _safe_close(connection, 1008, f"unknown path: {raw_path}")
            return
        token = token_from_request(request)
        conn = connection_factory(connection)
        try:
            stats = await server.serve(conn, token=token)
        except Exception as exc:  # noqa: BLE001 - 单连接失败不许拖垮整个服务
            logger.exception("会话循环异常退出: %s: %s", type(exc).__name__, exc)
            return
        logger.info(
            "会话结束 session=%s 收=%d 发=%d(文本 %d/二进制 %d) 关闭原因=%s",
            stats.session_id or "-",
            stats.frames_in,
            stats.frames_out,
            stats.text_out,
            stats.binary_out,
            stats.closed_reason or "-",
        )
        if on_stats is not None:
            try:
                on_stats(stats)
            except Exception as exc:  # noqa: BLE001 - 观测回调不该影响服务
                logger.exception("on_stats 回调失败: %s: %s", type(exc).__name__, exc)

    return handler


async def start_server(
    server: SessionServer,
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    path: str = DEFAULT_PATH,
    max_message_bytes: Optional[int] = DEFAULT_MAX_MESSAGE_BYTES,
    on_stats: Optional[Callable[[ConnectionStats], None]] = None,
    logger: Optional[logging.Logger] = None,
    connection_factory: Callable[[Any], WSConnection] = WebsocketsConnection,
    process_request: Optional[Callable[[Any, Any], Any]] = None,
    **serve_kwargs: Any,
) -> Any:
    """创建并启动监听，返回 `websockets` 的 `Server`（便于测试取端口）。

    与 `serve_forever` 分开是为了让测试能拿到真实端口（`port=0` 时由系统分配）——
    否则端到端用例只能硬编码端口，一撞上占用就随机失败。

    Args:
        max_message_bytes: 单帧上限，见 `DEFAULT_MAX_MESSAGE_BYTES` 的说明。
            `None` 表示不限（危险，仅在你清楚为什么时用）。
        process_request: 透传给 `websockets.serve` 的 HTTP 钩子（签名相同）。
            返回 `Response` 时该请求不走 WS 握手；返回 `None` 时照常升级。
            用于旁挂 `GET /health`、`GET /config/llm` 这类纯 HTTP 端点。
        serve_kwargs: 透传给 `websockets.serve`（如 `ping_interval`）。
    """
    _ws, ws_serve, _ = _import_websockets()
    log = logger or _LOGGER
    handler = _make_handler(
        server,
        path=path,
        on_stats=on_stats,
        logger=log,
        connection_factory=connection_factory,
    )
    kwargs: dict[str, Any] = dict(serve_kwargs)
    if process_request is not None:
        kwargs["process_request"] = process_request
    srv = await ws_serve(handler, host, port, max_size=max_message_bytes, **kwargs)
    bound = ", ".join(
        f"{s.getsockname()[0]}:{s.getsockname()[1]}" for s in (srv.sockets or ())
    )
    log.info("WebSocket 服务已启动：ws://%s%s", bound or f"{host}:{port}", path)
    return srv


async def serve_forever(
    server: SessionServer,
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    path: str = DEFAULT_PATH,
    max_message_bytes: Optional[int] = DEFAULT_MAX_MESSAGE_BYTES,
    on_stats: Optional[Callable[[ConnectionStats], None]] = None,
    on_listening: Optional[Callable[[Any], None]] = None,
    logger: Optional[logging.Logger] = None,
    connection_factory: Callable[[Any], WSConnection] = WebsocketsConnection,
    process_request: Optional[Callable[[Any, Any], Any]] = None,
    **serve_kwargs: Any,
) -> None:
    """启动并**一直服务**，直到被取消（Ctrl-C / 事件循环停止）。

    Args:
        on_listening: 监听已就绪时回调一次，参数是 `websockets` 的 `Server`。
            生产里用它注册信号处理 / 打启动日志 / 取真实端口；测试里用它抓
            `Server` 对象——否则 `port=0` 分配的端口根本拿不到，
            端到端用例就只能硬编码端口，一撞上占用就随机失败。
        process_request: 透传 `start_server`（HTTP 旁路端点，见其 docstring）。
    """
    srv = await start_server(
        server,
        host=host,
        port=port,
        path=path,
        max_message_bytes=max_message_bytes,
        on_stats=on_stats,
        logger=logger,
        connection_factory=connection_factory,
        process_request=process_request,
        **serve_kwargs,
    )
    if on_listening is not None:
        on_listening(srv)
    async with srv:
        await srv.serve_forever()


# ---------------------------------------------------------------- 开发装配


def make_llm_config_endpoint(
    evaluator_slot: list[Any],
    *,
    verify: Callable[[str], Optional[str]],
    apply_config: Callable[[Mapping[str, str]], tuple[bool, str]],
    logger: Optional[logging.Logger] = None,
) -> Callable[[Any, Any], Any]:
    """旁挂 HTTP 端点：`GET /health` 与 `GET /config/llm`（设置页用）。

    通道形态有个硬约束：websockets 的 `Request` **读不到 body**（库注释原话
    "body isn't useful in the context of this library"），所以配置走**请求头**：

        X-LLM-Base-URL / X-LLM-API-Key / X-LLM-Model

    key 只在内存里过一遍：不打印、不回显、不进日志（message 里只出现 model
    与 base_url）。鉴权与 WS 同源：同一个令牌，`Authorization: Bearer` 或
    `?token=` 二选一；对不上就 401，本机 dev 也不能裸奔。

    Args:
        evaluator_slot: 长度至少为 1 的可变槽（`build_dev_server` 挂在
            `server.evaluator_slot` 上）。替换 `slot[0]` 后，**新建立的面试**
            用新评估器；进行中的面试已持有旧对象引用，不受影响。
        verify: 令牌校验器（语义同 `make_static_token_verifier` 的返回值）。
        apply_config: 由装配层提供：收 `{base_url, api_key, model}`，
            返回 `(ok, message)`。构造评估器不是本函数的职责。
    """
    from urllib.parse import parse_qs, urlsplit

    log = logger or _LOGGER

    def _candidate_token(connection: Any) -> str:
        request = connection.request
        q = parse_qs(urlsplit(request.path).query).get("token", [""])
        if q and q[0]:
            return q[0]
        auth = request.headers.get("Authorization", "")
        return auth[7:] if auth.startswith("Bearer ") else auth

    def process_request(connection: Any, request_headers: Any) -> Any:
        path = urlsplit(connection.request.path).path
        if path == "/health":
            return connection.respond(200, '{"ok": true}')
        if path != "/config/llm":
            return None  # 其余请求交给 WS 握手（websockets 自己拒绝非法升级）

        if verify(_candidate_token(connection)) is None:
            return connection.respond(401, '{"ok": false, "message": "令牌缺失或不匹配"}')

        headers = connection.request.headers
        config = {
            "base_url": headers.get("X-LLM-Base-URL", ""),
            "api_key": headers.get("X-LLM-API-Key", ""),
            "model": headers.get("X-LLM-Model", ""),
        }
        try:
            ok, message = apply_config(config)
        except Exception as exc:  # noqa: BLE001 - 端点层兜底：任何装配异常都变 500 显式返回
            log.error("应用 LLM 配置失败：%s: %s", type(exc).__name__, exc)
            return connection.respond(500, '{"ok": false, "message": "应用配置失败，详见服务端日志"}')
        import json as _json

        body = _json.dumps({"ok": ok, "message": message}, ensure_ascii=False)
        return connection.respond(200 if ok else 400, body)

    return process_request


def make_static_token_verifier(
    token: str, *, session_prefix: str = "dev"
) -> Callable[[str], Optional[str]]:
    """构造"单一静态令牌"校验器。**仅供开发/联调**。

    生产必须换成一次性令牌（`ruipin.identity` 的令牌模块）——静态令牌可重放，
    泄漏一次，任何人都能建会话并拿到别人的面试报告。

    实现里有个必须写成这样的细节：比较走 `compare_digest` 的 **bytes** 重载。
    `compare_digest(str, str)` 在**任一参数含非 ASCII 字符时抛 `TypeError`**，
    而令牌来自 HTTP 握手，对端可以随便塞——`Bearer 中文` 就能把它炸成
    "会话循环异常退出"，连接莫名被关，日志里还看不出是攻击还是故障。
    先 `encode("utf-8")` 再比，这个入口就不存在了。

    Args:
        token: 期望的令牌，不能为空（空令牌会放过任何不带令牌的请求）。

    Raises:
        ValueError: 令牌为空。
    """
    if not token:
        raise ValueError("令牌不能为空：空令牌会让任何不带令牌的请求都通过")
    expected = token.encode("utf-8")

    def verify(candidate: str) -> Optional[str]:
        if not candidate:
            return None
        if not secrets.compare_digest(candidate.encode("utf-8"), expected):
            return None
        return f"{session_prefix}-{token[:8]}"

    return verify


def build_dev_server(
    *,
    questions: Sequence[str],
    token: str,
    repo: Any = None,
    evaluator: Any = None,
    clock: Optional[Callable[[], float]] = None,
    tts: Any = None,
    outbox_poll_s: float = 0.05,
    rate_limit_per_sec: int = DEFAULT_RATE_LIMIT_PER_SEC,
    idle_timeout_s: float = DEFAULT_IDLE_TIMEOUT_S,
    credit_capacity: int = DEFAULT_CREDIT_CAPACITY,
    ack_every: int = DEFAULT_ACK_EVERY,
) -> SessionServer:
    """开发/联调用的最小装配：**内存存储 + 假评估器 + 静态令牌**。

    这不是生产装配，差得还很远：
    * 评估器是假的（分数是写死的常量）→ 报告里的分数**没有任何意义**；
    * 存储是内存 SQLite → 进程一退，会话与报告全部消失；
    * 令牌是静态串 → 可重放，只能用于本机调试。

    将来接真实 LLM / 数据库时，本函数**不该被改造成生产装配**，而应由上层
    （部署编排）注入真实实现，本函数保持"小而假"的形态继续服务于联调。

    Args:
        token: 期望的令牌。请求里的 `Authorization: Bearer <token>` 或
            `?token=<token>` 必须与它相同（比较在 `Gateway.authenticate` 里做）。
    """
    from .fakes import FakeEvaluator  # 局部导入：开发装配的依赖不该污染模块导入期
    from .repo_sqlite import SqliteRepo

    import time as _time

    if not questions:
        raise ValueError("至少要给一道题：零题目会让每场面试都在第一轮直接结束")

    repo = repo if repo is not None else SqliteRepo(":memory:")
    clock = clock or _time.monotonic
    verify = make_static_token_verifier(token)

    # 评估器放可变槽而不是闭包直捕：设置页的 `GET /config/llm` 会热替换
    # slot[0]，新建立的面试即用新评估器。闭包直捕的话就焊死了。
    evaluator_slot: list[Any] = [evaluator if evaluator is not None else FakeEvaluator()]

    def build_bridge(session_id: str) -> InterviewBridge:
        return InterviewBridge(
            repo=repo,
            scheduler=TurnScheduler(evaluator_slot[0], clock=clock),
            config=BridgeConfig(questions=tuple(questions)),
            session_id=session_id,
            clock=clock,
            tts=tts,
        )

    server = SessionServer(
        build_bridge=build_bridge,
        token_verifier=verify,
        clock=clock,
        outbox_poll_s=outbox_poll_s,
        rate_limit_per_sec=rate_limit_per_sec,
        idle_timeout_s=idle_timeout_s,
        credit_capacity=credit_capacity,
        ack_every=ack_every,
    )
    server.evaluator_slot = evaluator_slot  # 供 make_llm_config_endpoint 使用
    return server


DEFAULT_DEV_QUESTIONS = (
    "请用两分钟介绍一下你自己和最近做过的项目。",
    "讲一个你解决过的技术难题，说明你的定位与取舍。",
    "你如何与团队协作推进一个有分歧的方案？",
)


@dataclass(frozen=True)
class DevPlan:
    """`main()` 的装配产物：一个已装好的服务端 + 监听参数。

    把"解析参数 + 装配"与"跑事件循环"分开，是因为**前者可以测、后者不能**
    （`asyncio.run` 会一直阻塞到进程结束）。混在一个函数里，命令行入口就只能靠
    手工试——而入口恰恰是"改一行参数就静默出错"的高发区。
    """

    server: SessionServer
    host: str
    port: int
    path: str
    token: str
    questions: tuple[str, ...]
    log_level: int = logging.INFO


def build_dev_plan(
    argv: Optional[Sequence[str]] = None, *, logger: Optional[logging.Logger] = None
) -> Optional[DevPlan]:
    """解析命令行并装配开发服务。参数不合法时返回 `None`（`main` 据此返回 2）。"""
    parser = argparse.ArgumentParser(
        prog="ruipin-ws-server",
        description="锐聘 AI 面试服务端（开发装配：假评估器 + 内存存储）",
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"监听地址（默认 {DEFAULT_HOST}）")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"端口（默认 {DEFAULT_PORT}）")
    parser.add_argument("--path", default=DEFAULT_PATH, help=f"WS 路径（默认 {DEFAULT_PATH}）")
    parser.add_argument("--token", default=None, help="一次性令牌；不传则随机生成并打印")
    parser.add_argument(
        "--questions",
        default=None,
        help="题目列表，逗号分隔；不传用内置示例题",
    )
    parser.add_argument("--log-level", default="INFO", help="日志级别（默认 INFO）")
    args = parser.parse_args(argv)

    log = logger or logging.getLogger("ruipin.ws_server")
    token = args.token or secrets.token_urlsafe(18)
    if args.questions is None:
        # 用 `is None` 而不是真值判断：`--questions ""` 是**显式给了空值**，
        # 与"没给这个参数"是两回事。用真值判断会把显式空值静默替换成默认题目——
        # 命令行上"我要求零题目"变成"起了 3 道题"，而且没有任何提示。
        questions = DEFAULT_DEV_QUESTIONS
    else:
        questions = tuple(q.strip() for q in str(args.questions).split(",") if q.strip())
    if not questions:
        log.error("题目列表为空，至少要给一道题")
        return None

    server = build_dev_server(questions=questions, token=token)
    return DevPlan(
        server=server,
        host=args.host,
        port=args.port,
        path=args.path,
        token=token,
        questions=questions,
        log_level=getattr(logging, str(args.log_level).upper(), logging.INFO),
    )


def main(
    argv: Optional[Sequence[str]] = None,
    *,
    runner: Optional[Callable[[DevPlan], int]] = None,
) -> int:
    """命令行入口：起一个开发用服务。

    用法::

        python -m ruipin.adapters.ws_server --port 8787 --token my-token

    Args:
        runner: 真正跑事件循环的那一步，可注入。默认实现会阻塞到进程结束，
            注入后可以在测试里验证"入口把参数一路传对了"而不真的起服务。
    """
    plan = build_dev_plan(argv)
    if plan is None:
        return 2

    logging.basicConfig(
        level=plan.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("ruipin.ws_server")
    log.warning("=" * 72)
    log.warning("开发装配：评估器是【假的】、存储是【内存】的、令牌是【静态】的。")
    log.warning("报告里的分数没有任何参考价值，仅用于验证链路是否通。")
    log.warning("=" * 72)
    log.info("令牌（放 Authorization: Bearer 或 ?token=）: %s", plan.token)
    log.info("题目 %d 道", len(plan.questions))

    run = runner or _run_blocking
    try:
        return run(plan)
    except KeyboardInterrupt:  # pragma: no cover - 交互式退出
        log.info("已停止")
        return 0
    except OSError as exc:
        # 端口被占是开发时最常见的一类失败。原始的 `OSError: [WinError 10048]`
        # 只说"地址已被使用"，不告诉人下一步该干什么，于是每次都有人去翻代码找默认端口。
        log.error(
            "无法监听 %s:%d（%s）。换一个端口重试：--port <空闲端口>",
            plan.host,
            plan.port,
            exc,
        )
        return 3


def _run_blocking(
    plan: DevPlan, *, serve: Callable[..., Any] = serve_forever
) -> int:
    """默认执行方式：起服务并一直跑（Ctrl-C 退出）。

    `serve` 可注入，理由与 `main` 的 `runner` 相同：这个函数正常**永远不返回**，
    不注入就没法验证"参数一路传对了"，只能靠手工试。注入后测试能让它正常返回。
    """
    asyncio.run(serve(plan.server, host=plan.host, port=plan.port, path=plan.path))
    return 0


if __name__ == "__main__":  # pragma: no cover - 入口
    sys.exit(main())
