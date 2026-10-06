"""HTTP 传输抽象：把"发一个 POST JSON 请求"关在可注入的 transport 后面。

设计意图
--------
1. **核心层与编排层永不直接 import httpx/aiohttp**。本机与 CI 没有第三方依赖
   也能跑完整测试套件：生产用 `HttpxTransport`（函数内惰性 import，缺失即抛
   `Unavailable`），测试用 `FakeTransport`（脚本化、零依赖、完全确定性）。
2. **非 2xx 不是异常**。状态码是业务信息——429 要退避重试、401 要立刻放弃。
   把状态塞进异常会让"可重试 vs 不可重试"的判定散落到每个调用点（遗留系统的
   通病：每个 provider 各写一套 `if "429" in str(e)`）。这里一律返回
   `HttpResponse`，判定集中在 `is_retryable_status` 一处。
3. 只有"请求根本没发出去"（DNS 失败 / 连接重置 / 超时）才抛异常；适配器把这类
   异常统一视为可重试。

超时一律由调用方以 `timeout_s` 显式传入，不存在"某个 client 里藏着的默认超时"。

两种传输
--------
* `HttpTransport.post_json`——**一次拿完**：评分、强制对齐、ASR 转写。
* `HttpStreamTransport.post_stream`——**边生成边拿**：TTS 音频块。语音的首字延迟
  完全由它决定，所以它必须在第一个字节到达时就交出，不得内部攒完再吐。
两者共用同一套"状态码是数据、只有请求没发出去才算异常"的约定。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    runtime_checkable,
)

from ..domain.errors import Unavailable

#: 默认超时（秒）。来源：方案 §10 的预算基线——单题 LLM 评分不得拖垮整场节奏。
DEFAULT_TIMEOUT_S: float = 30.0

#: 明确的"可重试"状态码。408 请求超时、429 限流；其余 5xx 由区间判定覆盖。
#: 400/401/403/404 属于"请求本身有问题"，重试一万次也不会变好，直接放弃。
RETRYABLE_STATUSES: frozenset[int] = frozenset({408, 429})

#: 流式响应出错时最多带回多少字节的错误说明。错误正文只用于排障，
#: 不该因为一个返回 10 MB HTML 错误页的网关把内存吃掉。
MAX_ERROR_BODY_BYTES: int = 2048


@dataclass(frozen=True)
class HttpResponse:
    """一次 HTTP 调用的原始结果。

    `body` 保持 bytes：字符集判定属于解析层的事，transport 不做猜测。
    """

    status: int
    body: bytes = b""
    headers: dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """2xx 即成功；**其余状态码不是异常**，交由上层按可重试性判定。"""
        return 200 <= self.status < 300

    @property
    def text(self) -> str:
        """按 UTF-8 解码；非法字节用替换字符，**不抛异常**（错误信息要能进日志）。"""
        return self.body.decode("utf-8", errors="replace")

    def json(self) -> Any:
        """解析响应体。非法 JSON 抛 `json.JSONDecodeError`（`ValueError` 子类）。"""
        return json.loads(self.text)


@runtime_checkable
class HttpTransport(Protocol):
    """HTTP 传输端口：只做一件事——发一个 JSON POST 并原样带回结果。"""

    async def post_json(
        self,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_s: float,
    ) -> HttpResponse: ...


@dataclass(frozen=True)
class HttpCall:
    """`FakeTransport` 记录的一次调用，供断言使用。"""

    url: str
    headers: Mapping[str, str]
    payload: Mapping[str, Any]
    timeout_s: float


class FakeTransport:
    """脚本化 transport：按调用次序返回预设响应（或抛预设异常），并记录每次调用。

    Args:
        script: 按次序消费的条目。`HttpResponse` 则原样返回，`Exception` 实例则
            抛出（模拟网络异常/超时）。"第 1 次 429、第 2 次 200" 就是
            `[HttpResponse(429, ...), HttpResponse(200, ...)]`。
        default: 脚本用尽后的兜底条目；为 None 时用尽即抛 `AssertionError`
            ——那是**测试的配方问题**，不是被测代码的问题，必须响亮地失败。
        on_call: 每次调用前触发，参数是 0-based 调用序号。典型用途是推进注入的
            `DeterministicClock`，让"延迟被观测"这件事可断言（不真的 sleep）。
    """

    def __init__(
        self,
        script: Sequence[Any] = (),
        *,
        default: Optional[Any] = None,
        on_call: Optional[Callable[[int], None]] = None,
    ) -> None:
        self.script: list[Any] = list(script)
        self.default = default
        self.on_call = on_call
        self.calls: list[HttpCall] = []

    @property
    def n_calls(self) -> int:
        """已发生的调用次数（重试次数断言的直接抓手）。"""
        return len(self.calls)

    async def post_json(
        self,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_s: float,
    ) -> HttpResponse:
        index = len(self.calls)
        self.calls.append(HttpCall(url, dict(headers), dict(payload), timeout_s))
        if self.on_call is not None:
            self.on_call(index)
        item = self._item_at(index)
        if isinstance(item, BaseException):
            raise item
        return item

    def reset(self) -> None:
        self.calls = []

    def _item_at(self, index: int) -> Any:
        if index < len(self.script):
            return self.script[index]
        if self.default is not None:
            return self.default
        raise AssertionError(
            f"FakeTransport 脚本已用尽：第 {index + 1} 次调用没有预设响应"
        )


class HttpxTransport:
    """真实 HTTP：`httpx` 的薄封装，**惰性 import**。

    httpx 未安装时不 import 失败，而是抛 `Unavailable("httpx", ...)`——把"依赖缺失"
    变成一条可观测、可换供应商的能力缺失事件，而不是启动期的 ImportError。
    """

    def __init__(self, *, timeout_s: float = DEFAULT_TIMEOUT_S) -> None:
        self.timeout_s = float(timeout_s)

    async def post_json(
        self,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_s: float,
    ) -> HttpResponse:
        try:
            import httpx  # 惰性：不在 import 期把 httpx 变成硬依赖
        except ImportError as exc:
            raise Unavailable(
                "httpx", f"未安装 httpx（pip install httpx）: {exc}"
            ) from exc

        async with httpx.AsyncClient(timeout=timeout_s) as client:
            resp = await client.post(url, headers=dict(headers), json=dict(payload))
        return HttpResponse(
            status=int(resp.status_code),
            body=bytes(resp.content),
            headers=dict(resp.headers),
        )


@dataclass(frozen=True)
class StreamEvent:
    """流式响应的一步。

    * **第一步一定是"响应头"**（`is_head` 为 True），携带状态码与响应头，`body` 通常为空；
    * 之后每一步只携带 `body` 字节；
    * **非 2xx 时，紧跟响应头的可能是"一步"错误说明（服务端返回的正文，已截断），
      随后流即结束**——调用方必须先判 `ok`，不能把错误正文当音频用。

    为什么不像 `HttpResponse` 那样"一次给完"：语音的首字延迟完全由"第一个字节何时
    到手"决定，攒完整个响应再返回就等于把流式退化成同步。
    为什么状态码仍是数据而不是异常：与 `HttpResponse` 同一个理由——"可重试 vs
    不可重试"的判定必须集中在一处，不能散落到每个调用点的 `if "429" in str(e)`。
    """

    body: bytes = b""
    status: int = 0
    headers: Mapping[str, str] = field(default_factory=dict)

    @property
    def is_head(self) -> bool:
        """是否是那一步"响应头"。状态码为 0 表示这是后续的 body 步。"""
        return self.status != 0

    @property
    def ok(self) -> bool:
        """2xx 即成功。**只在 `is_head` 上有意义**（body 步的 status 恒为 0）。"""
        return 200 <= self.status < 300


@runtime_checkable
class HttpStreamTransport(Protocol):
    """流式 HTTP 传输：发一个 JSON POST，把响应体**按到达顺序**分块交出来。

    与 `HttpTransport` 的分工：
    * `post_json` 用于"一次拿完"——评分、强制对齐、ASR 转写；
    * `post_stream` 用于"边生成边拿"——TTS 音频块。

    实现必须**在第一个字节到达时就交出**，不得内部攒完再吐；否则首字延迟与
    `post_json` 无异，加这一层就白加了。
    """

    def post_stream(
        self,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_s: float,
    ) -> AsyncIterator[StreamEvent]: ...


class FakeStreamTransport:
    """脚本化流式 transport：按调用次序产出预设块流（或抛预设异常），并记录每次调用。

    脚本条目有两种写法：
    * `Exception` 实例 → 直接抛出（模拟连接重置 / 超时）；
    * 可迭代的 `bytes`（或 `StreamEvent`）→ 依次交出。

    **非 2xx 要显式写进脚本**：`[StreamEvent(status=503), b"..."]`——这样才是在测
    "调用方有没有先判 `ok`"，而不是在测"传输层会不会自己抛"。
    """

    def __init__(
        self,
        script: Sequence[Any] = (),
        *,
        default: Optional[Any] = None,
        on_call: Optional[Callable[[int], None]] = None,
    ) -> None:
        self.script: list[Any] = list(script)
        self.default = default
        self.on_call = on_call
        self.calls: list[HttpCall] = []

    @property
    def n_calls(self) -> int:
        return len(self.calls)

    async def post_stream(
        self,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_s: float,
    ) -> AsyncIterator[StreamEvent]:
        index = len(self.calls)
        self.calls.append(HttpCall(url, dict(headers), dict(payload), timeout_s))
        if self.on_call is not None:
            self.on_call(index)
        item = self._item_at(index)
        if isinstance(item, BaseException):
            raise item
        for entry in item:
            yield entry if isinstance(entry, StreamEvent) else StreamEvent(body=entry)

    def reset(self) -> None:
        self.calls = []

    def _item_at(self, index: int) -> Any:
        if index < len(self.script):
            return self.script[index]
        if self.default is not None:
            return self.default
        raise AssertionError(
            f"FakeStreamTransport 脚本已用尽：第 {index + 1} 次调用没有预设响应"
        )


class HttpxStreamTransport:
    """真实流式 HTTP：`httpx` 的薄封装，**惰性 import**（与 `HttpxTransport` 同理）。

    超时是**逐块读超时**：连接建好之后，只要每两块之间的间隔不超过 `timeout_s`
    就不会被判超时。这对 TTS 是对的——一个长句合成时块与块之间本就有停顿，
    用"整段总时长"当超时会把它误杀。
    """

    def __init__(self, *, connect_timeout_s: float = 10.0) -> None:
        self.connect_timeout_s = float(connect_timeout_s)

    async def post_stream(
        self,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_s: float,
    ) -> AsyncIterator[StreamEvent]:
        try:
            import httpx  # 惰性：不在 import 期把 httpx 变成硬依赖
        except ImportError as exc:
            raise Unavailable(
                "httpx", f"未安装 httpx（pip install httpx）: {exc}"
            ) from exc

        timeout = httpx.Timeout(timeout_s, connect=min(timeout_s, self.connect_timeout_s))
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream(
                "POST", url, headers=dict(headers), json=dict(payload)
            ) as resp:
                yield StreamEvent(status=int(resp.status_code), headers=dict(resp.headers))
                if not 200 <= resp.status_code < 300:
                    # 非 2xx：把错误正文带一步出去（截断到 2 KB）再结束。
                    # 不带的话，一个 400（音色名写错）在日志里只剩"HTTP 400"，
                    # 排查得去翻服务端日志；带上就能一眼看出服务端说了什么。
                    detail = bytearray()
                    async for chunk in resp.aiter_bytes():
                        detail.extend(chunk)
                        if len(detail) >= MAX_ERROR_BODY_BYTES:
                            break
                    if detail:
                        yield StreamEvent(body=bytes(detail[:MAX_ERROR_BODY_BYTES]))
                    return
                async for chunk in resp.aiter_bytes():
                    if chunk:
                        yield StreamEvent(body=bytes(chunk))


def is_retryable_status(status: int) -> bool:
    """状态码是否值得重试。

    408/429 与全部 5xx 可重试（限流与服务端抖动会自愈）；
    其余一律不可重试（4xx 是请求本身的问题，重试只会浪费预算与时间）。
    """
    return status in RETRYABLE_STATUSES or 500 <= status < 600


__all__ = [
    "DEFAULT_TIMEOUT_S",
    "MAX_ERROR_BODY_BYTES",
    "RETRYABLE_STATUSES",
    "FakeStreamTransport",
    "FakeTransport",
    "HttpCall",
    "HttpResponse",
    "HttpStreamTransport",
    "HttpTransport",
    "HttpxStreamTransport",
    "HttpxTransport",
    "StreamEvent",
    "is_retryable_status",
]
