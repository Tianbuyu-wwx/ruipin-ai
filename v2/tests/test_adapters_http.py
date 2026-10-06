"""HTTP 传输抽象（`adapters/http.py`）的测试。

守的回归
--------
1. **非 2xx 不是异常**：状态码必须原样带回，否则上层无法区分"429 该重试"与
   "401 该放弃"，又会把判定逻辑抄回每个 provider 里。
2. `FakeTransport` 的脚本/记录语义：重试次数断言全靠它，它自己必须先是对的。
3. `HttpxTransport` 在无 httpx 环境下**惰性失败**成 `Unavailable` 而不是
   ImportError——依赖缺失要能被路由当成"这一家不可用"换到下一家。
"""

from __future__ import annotations

import asyncio
import sys
import types

import pytest

from ruipin.adapters.fakes import DeterministicClock
from ruipin.adapters.http import (
    DEFAULT_TIMEOUT_S,
    RETRYABLE_STATUSES,
    FakeTransport,
    HttpCall,
    HttpResponse,
    HttpTransport,
    HttpxTransport,
    is_retryable_status,
)
from ruipin.domain.errors import Unavailable

run = asyncio.run

_URL = "https://api.deepseek.com/v1/chat/completions"
_HEADERS = {"Authorization": "Bearer sk-test"}
_PAYLOAD = {"model": "deepseek-chat"}


# ---------- HttpResponse ----------


@pytest.mark.parametrize(
    "status, ok",
    [(199, False), (200, True), (204, True), (299, True), (300, False), (500, False)],
)
def test_ok_只认2xx其它状态码不是异常(status, ok):
    """防回归：把非 2xx 当异常会让 429 与 401 混为一谈，重试判定随之失效。"""
    assert HttpResponse(status=status, body=b"{}").ok is ok


def test_json_解析正常响应体():
    resp = HttpResponse(200, b'{"a": 1}', {"content-type": "application/json"})

    assert resp.json() == {"a": 1}
    assert resp.headers["content-type"] == "application/json"


def test_json_非法JSON抛ValueError供上层当失败处理():
    """防回归：非法响应体必须可判定为"本次调用失败"，不能静默成空结果。"""
    resp = HttpResponse(200, b"<html>502 Bad Gateway</html>")

    with pytest.raises(ValueError):
        resp.json()


def test_text_非法UTF8不抛异常便于进日志():
    resp = HttpResponse(200, b"\xff\xfe bad bytes")

    assert isinstance(resp.text, str)


# ---------- is_retryable_status ----------


@pytest.mark.parametrize(
    "status, retryable",
    [
        (200, False),
        (400, False),
        (401, False),
        (403, False),
        (404, False),
        (422, False),
        (408, True),
        (429, True),
        (500, True),
        (502, True),
        (503, True),
        (504, True),
        (599, True),
    ],
)
def test_可重试判定集中在一处(status, retryable):
    """防回归：4xx 重试一万次也不会变好，白烧预算；5xx/429/408 会自愈。"""
    assert is_retryable_status(status) is retryable


def test_可重试集合只含超时与限流():
    assert RETRYABLE_STATUSES == frozenset({408, 429})


# ---------- FakeTransport ----------


def test_按脚本次序返回响应并记录每次调用():
    transport = FakeTransport([HttpResponse(429, b"slow down"), HttpResponse(200, b"{}")])

    first = run(transport.post_json(_URL, _HEADERS, _PAYLOAD, 12.0))
    second = run(transport.post_json(_URL, _HEADERS, _PAYLOAD, 12.0))

    assert (first.status, second.status) == (429, 200)
    assert second.body == b"{}"
    assert transport.n_calls == 2
    assert transport.calls[0] == HttpCall(_URL, _HEADERS, _PAYLOAD, 12.0)


def test_脚本里的异常实例会被原样抛出():
    """防回归：网络异常/超时必须能被抛出，否则测不到"可重试"分支。"""
    boom = ConnectionResetError("connection reset by peer")
    transport = FakeTransport([boom, HttpResponse(200, b"{}")])

    with pytest.raises(ConnectionResetError) as exc:
        run(transport.post_json(_URL, _HEADERS, _PAYLOAD, 5.0))

    assert exc.value is boom
    assert transport.n_calls == 1
    assert run(transport.post_json(_URL, _HEADERS, _PAYLOAD, 5.0)).status == 200


def test_脚本用尽时响亮失败而不是静默返回假响应():
    """防回归：脚本配方写错必须立刻暴露，绝不能让被测代码拿到"凭空出现"的响应。"""
    transport = FakeTransport([HttpResponse(200, b"{}")])
    run(transport.post_json(_URL, _HEADERS, _PAYLOAD, 5.0))

    with pytest.raises(AssertionError, match="脚本已用尽"):
        run(transport.post_json(_URL, _HEADERS, _PAYLOAD, 5.0))


def test_脚本用尽后有default时走default():
    transport = FakeTransport(
        [], default=HttpResponse(503, b"always busy")
    )

    assert run(transport.post_json(_URL, _HEADERS, _PAYLOAD, 5.0)).status == 503
    assert run(transport.post_json(_URL, _HEADERS, _PAYLOAD, 5.0)).status == 503


def test_on_call钩子按序号触发可用于推进确定性时钟():
    """防回归：延迟断言不能靠 sleep；时钟推进必须发生在 transport 被调用的当时。"""
    clock = DeterministicClock()
    seen: list[int] = []
    transport = FakeTransport(
        [HttpResponse(200, b"{}"), HttpResponse(200, b"{}")],
        on_call=lambda i: (seen.append(i), clock.advance(0.25)),
    )

    run(transport.post_json(_URL, _HEADERS, _PAYLOAD, 5.0))
    run(transport.post_json(_URL, _HEADERS, _PAYLOAD, 5.0))

    assert seen == [0, 1]
    assert clock.t == pytest.approx(0.5)


def test_reset清空调用记录():
    transport = FakeTransport([HttpResponse(200, b"{}")])
    run(transport.post_json(_URL, _HEADERS, _PAYLOAD, 5.0))

    transport.reset()

    assert transport.n_calls == 0


def test_两个transport都满足HttpTransport协议():
    assert isinstance(FakeTransport(), HttpTransport)
    assert isinstance(HttpxTransport(), HttpTransport)


def test_默认超时是显式常量而非散落的魔法数():
    assert DEFAULT_TIMEOUT_S == 30.0


# ---------- HttpxTransport ----------


def test_httpx未安装时抛Unavailable而不是ImportError(monkeypatch):
    """防回归：依赖缺失要能被路由当成"这一家不可用"从而切下一家。"""
    monkeypatch.setitem(sys.modules, "httpx", None)
    transport = HttpxTransport()

    with pytest.raises(Unavailable) as exc:
        run(transport.post_json(_URL, _HEADERS, _PAYLOAD, 5.0))

    assert exc.value.provider == "httpx"
    assert "未安装 httpx" in exc.value.reason


def _install_httpx_stub(monkeypatch) -> dict:
    """塞一个最小 httpx 替身模块，断言超时与参数被如实透传。"""
    state: dict = {}

    class _Resp:
        status_code = 200
        content = b'{"ok": true}'
        headers = {"content-type": "application/json"}

    class _Client:
        def __init__(self, timeout=None):
            state["timeout"] = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, url, headers=None, json=None):
            state["url"] = url
            state["headers"] = headers
            state["json"] = json
            return _Resp()

    module = types.ModuleType("httpx")
    module.AsyncClient = _Client
    monkeypatch.setitem(sys.modules, "httpx", module)
    return state


def test_httpx透传超时与请求参数(monkeypatch):
    """防回归：超时必须来自调用方的 timeout_s，不能是某个 client 里的隐藏默认值。"""
    state = _install_httpx_stub(monkeypatch)
    transport = HttpxTransport()

    resp = run(transport.post_json(_URL, _HEADERS, _PAYLOAD, 7.5))

    assert resp.status == 200
    assert resp.json() == {"ok": True}
    assert state["timeout"] == 7.5
    assert state["url"] == _URL
    assert state["headers"] == _HEADERS
    assert state["json"] == _PAYLOAD


def test_httpx非2xx也原样返回不抛异常(monkeypatch):
    """防回归：429 要能一路带到上层做退避重试，不能在 transport 层就炸掉。"""
    state = _install_httpx_stub(monkeypatch)

    class _Resp429:
        status_code = 429
        content = b'{"error": "rate limited"}'
        headers = {"retry-after": "1"}

    class _Client:
        def __init__(self, timeout=None):
            state["timeout"] = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, url, headers=None, json=None):
            return _Resp429()

    module = types.ModuleType("httpx")
    module.AsyncClient = _Client
    monkeypatch.setitem(sys.modules, "httpx", module)

    resp = run(HttpxTransport().post_json(_URL, _HEADERS, _PAYLOAD, 3.0))

    assert resp.status == 429
    assert resp.ok is False
    assert is_retryable_status(resp.status) is True
    assert resp.headers["retry-after"] == "1"
