"""WebSocket 适配器与服务端入口（`adapters/ws_server.py`）的测试。

分两层，因为这两层的**证伪方式完全不同**：

* **握手与适配层**（取令牌、路径匹配、异常翻译）——纯逻辑，用假连接与假请求
  逐条钉死。这一层不需要 `websockets`，因此在任何环境下都能跑。
* **真实网络层**（`start_server` + 真实客户端）——**必须真的起一个服务、真的连一次**。
  前面所有层（`Gateway` / `SessionServer` / `SyncHandler` / 桥接层）都用假连接测过，
  但"假连接测过"与"真能连上"之间隔着整个 WS 库：协议版本、帧类型、
  二进制帧是否原样送达、关闭握手，一个都不在假连接的覆盖范围内。
  所以这里有一个端到端用例：起真服务 → 真实 `websockets` 客户端 →
  跑完一整场面试 → 断言报告出来了、TTS 的二进制帧也到了。

依赖缺失时真实网络用例 `skip`（并**在 skip 原因里写明缺什么**），
避免"没装库"被误读成"通过了"。
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Optional

import pytest

from ruipin.adapters.fakes import FakeEvaluator, FakeTTS
from ruipin.adapters.ws_server import (
    DEFAULT_HOST,
    DEFAULT_MAX_MESSAGE_BYTES,
    DEFAULT_PATH,
    DEFAULT_PORT,
    DevPlan,
    WebsocketsConnection,
    _describe_close,
    _import_websockets,
    _make_handler,
    _run_blocking,
    build_dev_plan,
    build_dev_server,
    main,
    make_static_token_verifier,
    path_matches,
    serve_forever,
    start_server,
    token_from_path,
    token_from_request,
    websockets_installed,
)
from ruipin.domain.errors import Unavailable
from ruipin.transport.session_server import SessionServer as _SessionServerForHttpTest
from ruipin.transport import (
    ClientType,
    Opcode,
    ServerType,
    decode_binary,
    decode_text,
    encode_text,
    make_envelope,
)
from ruipin.transport.session_server import ConnectionClosed, ConnectionStats

TOKEN = "tok-abc-123"
QUESTIONS = ("请介绍一下你自己", "讲一个你解决过的难题")


# ---------------------------------------------------------------- 假对象


class FakeClosed(Exception):
    """冒充 `websockets.exceptions.ConnectionClosed`。"""

    def __init__(self, code: Optional[int] = None, reason: str = "") -> None:
        self.code = code
        self.reason = reason
        super().__init__(f"closed {code} {reason}")


class FakeHeaders(dict):
    pass


class FakeRequest:
    def __init__(self, path: str = DEFAULT_PATH, headers: Optional[dict] = None) -> None:
        self.path = path
        self.headers = FakeHeaders(headers or {})


class FakeConn:
    """假的 `websockets` 连接对象。"""

    def __init__(
        self,
        incoming: Optional[list] = None,
        *,
        closed: Optional[FakeClosed] = None,
    ) -> None:
        self.incoming = list(incoming or [])
        self.closed_exc = closed
        self.sent: list = []
        self.closes: list[tuple[int, str]] = []
        self.request = FakeRequest()

    async def recv(self):
        if self.incoming:
            return self.incoming.pop(0)
        if self.closed_exc is not None:
            raise self.closed_exc
        raise FakeClosed(1000, "")

    async def send(self, data) -> None:
        self.sent.append(data)

    async def close(self, code: int = 1000, reason: str = "") -> None:
        self.closes.append((code, reason))


class RaisingConn(FakeConn):
    async def close(self, code: int = 1000, reason: str = "") -> None:
        raise RuntimeError("已经关掉了")


class RecordingServer:
    """冒充 `SessionServer`：只记录被怎么调用。"""

    def __init__(self) -> None:
        self.calls: list[tuple[Any, str]] = []

    async def serve(self, conn, *, token: str) -> ConnectionStats:
        self.calls.append((conn, token))
        return ConnectionStats(session_id="s-1", frames_in=2, closed_reason="测试")


class BoomServer(RecordingServer):
    async def serve(self, conn, *, token: str) -> ConnectionStats:
        raise RuntimeError("模拟会话循环崩溃")


class _Log:
    def __init__(self) -> None:
        self.records: list[tuple[str, str]] = []

    def info(self, fmt, *a):
        self.records.append(("info", fmt % a if a else fmt))

    def warning(self, fmt, *a):
        self.records.append(("warning", fmt % a if a else fmt))

    def exception(self, fmt, *a):
        self.records.append(("exception", fmt % a if a else fmt))


# ---------------------------------------------------------------- 取令牌


def test_token_from_path_reads_the_query_parameter():
    assert token_from_path("/ws?token=abc") == "abc"
    assert token_from_path("/ws?token=abc&x=1") == "abc"
    assert token_from_path("/ws") == ""
    assert token_from_path("/ws?x=1") == ""
    assert token_from_path("/ws?token=") == "", "空值等于没给，不能变成一个空串令牌"


def test_token_from_path_survives_a_malformed_url():
    # `http://[::1` 会让 urlsplit 抛 ValueError（IPv6 字面量没闭合）。
    # 握手请求的路径来自对端，畸形输入必须被吸收，不能让整个握手炸掉。
    assert token_from_path("http://[::1") == ""
    assert token_from_path("::::") == "", "这一串本身不抛异常，只是解不出 token"


def test_authorization_header_wins_over_the_query_parameter():
    """header 优先：URL 会进代理日志/浏览器历史，令牌泄了就是会话被顶替。"""
    req = FakeRequest("/ws?token=from-query", {"Authorization": "Bearer from-header"})
    assert token_from_request(req) == "from-header"


def test_bearer_prefix_is_case_insensitive_and_tolerates_spacing():
    assert token_from_request(FakeRequest(headers={"Authorization": "bearer  x  "})) == "x"
    assert token_from_request(FakeRequest(headers={"Authorization": "BEARER y"})) == "y"


@pytest.mark.parametrize(
    "header",
    ["", "Basic abc", "Bearer", "Bearer   ", "Token abc", None],
)
def test_a_header_that_is_not_a_bearer_token_falls_back_to_the_query(header):
    headers = {} if header is None else {"Authorization": header}
    req = FakeRequest("/ws?token=from-query", headers)
    assert token_from_request(req) == "from-query"


def test_token_from_request_without_headers_uses_the_path():
    class _Bare:
        path = "/ws?token=t"

    assert token_from_request(_Bare()) == "t"


def test_token_from_request_without_a_path_is_empty():
    assert token_from_request(object()) == ""


# ---------------------------------------------------------------- 路径匹配


def test_path_matches_ignores_query_and_rejects_other_paths():
    assert path_matches("/ws?token=x", "/ws") is True
    assert path_matches("/ws", "/ws") is True
    assert path_matches("/", "/ws") is False
    assert path_matches("/wsx", "/ws") is False, "前缀相同不算匹配（防路径穿越式误放行）"
    assert path_matches("::::", "/ws") is False
    assert path_matches("http://[::1", "/ws") is False, "畸形 URL 不能把握手炸掉"


# ---------------------------------------------------------------- 关闭原因


def test_describe_close_separates_normal_from_abnormal():
    """把"候选人正常下线"与"网络断了"混成一句话，排查时完全没法分拣。"""
    assert _describe_close(FakeClosed(1000, "")) == "对端正常关闭(1000)"
    assert _describe_close(FakeClosed(1001, "going away")) == "对端正常关闭(1001: going away)"
    assert _describe_close(FakeClosed(1011, "boom")) == "对端异常关闭(1011: boom)"
    assert _describe_close(FakeClosed(1006, "")) == "对端异常关闭(1006)"
    assert _describe_close(Exception("no code attr")) == "对端关闭"


# ---------------------------------------------------------------- 连接适配


def test_recv_translates_the_library_exception_into_ours():
    conn = FakeConn(closed=FakeClosed(1011, "internal"))
    ws = WebsocketsConnection(conn, closed_types=(FakeClosed,))
    with pytest.raises(ConnectionClosed) as ei:
        asyncio.run(ws.recv())
    assert "1011" in ei.value.reason
    assert ws.closed_reason == ei.value.reason, "关闭原因要留在连接对象上供排查"


def test_recv_passes_through_text_and_bytes():
    conn = FakeConn(["hi", b"\x00\x01"])
    ws = WebsocketsConnection(conn, closed_types=(FakeClosed,))
    assert asyncio.run(ws.recv()) == "hi"
    assert asyncio.run(ws.recv()) == b"\x00\x01"


def test_send_and_close_delegate_verbatim():
    conn = FakeConn()
    ws = WebsocketsConnection(conn, closed_types=(FakeClosed,))

    async def main():
        await ws.send("text")
        await ws.send(b"binary")
        await ws.close(1008, "bye")

    asyncio.run(main())
    assert conn.sent == ["text", b"binary"]
    assert conn.closes == [(1008, "bye")]


def test_closed_types_default_to_the_library_when_installed():
    if not websockets_installed():  # pragma: no cover - 本机已装
        pytest.skip("未安装 websockets：无法验证默认异常类型来自库")
    conn = FakeConn()
    ws = WebsocketsConnection(conn)
    assert len(ws._closed_types) == 1


# ---------------------------------------------------------------- handler


def _handler(server, **kwargs):
    return _make_handler(
        server,  # type: ignore[arg-type]
        path=DEFAULT_PATH,
        on_stats=kwargs.pop("on_stats", None),
        logger=kwargs.pop("logger", _Log()),
        **kwargs,
    )


def test_handler_rejects_a_wrong_path_without_touching_the_server():
    server = RecordingServer()
    conn = FakeConn(["hello"])
    conn.request = FakeRequest("/")  # 服务挂在 /ws，来的是 /
    log = _Log()
    asyncio.run(_handler(server, logger=log)(conn))
    assert conn.closes == [(1008, "unknown path: /")], "路径不匹配必须明确拒绝"
    assert server.calls == [], "被拒的握手绝不能触达会话服务"


def test_handler_rejects_a_wrong_path_even_if_closing_raises():
    """关不掉也不许把异常抛给 websockets 的调度器（那会变成一条无主崩溃）。"""
    server = RecordingServer()
    conn = RaisingConn()
    conn.request = FakeRequest("/nope")
    log = _Log()
    asyncio.run(_handler(server, logger=log)(conn))
    assert any(level == "warning" for level, _ in log.records)
    assert server.calls == []


def test_handler_forwards_the_token_and_reports_stats():
    server = RecordingServer()
    captured: list[ConnectionStats] = []
    conn = FakeConn()
    conn.request = FakeRequest(f"{DEFAULT_PATH}?token={TOKEN}")
    log = _Log()
    asyncio.run(_handler(server, on_stats=captured.append, logger=log)(conn))
    assert server.calls, "路径与令牌都对，必须真的进入会话服务"
    _, token = server.calls[0]
    assert token == TOKEN
    assert captured and captured[0].session_id == "s-1"
    assert any("会话结束" in msg for _, msg in log.records)


def test_handler_contains_a_crashing_session_loop():
    server = BoomServer()
    conn = FakeConn()
    captured: list[ConnectionStats] = []
    log = _Log()
    asyncio.run(_handler(server, on_stats=captured.append, logger=log)(conn))
    assert any(level == "exception" for level, _ in log.records)
    assert captured == [], "崩了就没有统计可报，不能伪造一条出来"


def test_handler_survives_a_raising_on_stats_callback():
    server = RecordingServer()
    conn = FakeConn()

    def boom(_stats):
        raise RuntimeError("观测回调炸了")

    asyncio.run(_handler(server, on_stats=boom, logger=_Log())(conn))
    # 走到这里不抛异常即通过：观测失败绝不能影响服务


def test_handler_uses_an_injected_connection_factory():
    server = RecordingServer()
    conn = FakeConn()
    seen: list[Any] = []

    def factory(raw):
        seen.append(raw)
        return WebsocketsConnection(raw, closed_types=(FakeClosed,))

    asyncio.run(_handler(server, connection_factory=factory)(conn))
    assert seen == [conn]


# ---------------------------------------------------------------- 静态令牌


def test_static_verifier_maps_a_good_token_to_a_session_id():
    verify = make_static_token_verifier(TOKEN)
    assert verify(TOKEN) == f"dev-{TOKEN[:8]}"
    assert verify("wrong") is None
    assert verify("") is None


def test_static_verifier_does_not_crash_on_a_non_ascii_token():
    """实测踩到的坑：`compare_digest(str, str)` 遇非 ASCII 会抛 TypeError。

    令牌来自 HTTP 握手，对端想塞什么就塞什么。若不先编码成 bytes，
    `Authorization: Bearer 中文` 就能把会话循环炸成"异常退出"，
    连接莫名被关、日志里还看不出是攻击还是故障。
    """
    verify = make_static_token_verifier(TOKEN)
    assert verify("中文令牌") is None
    assert verify("ＡＢＣ") is None
    assert verify(TOKEN) == f"dev-{TOKEN[:8]}"


def test_static_verifier_rejects_an_empty_expected_token():
    with pytest.raises(ValueError, match="令牌不能为空"):
        make_static_token_verifier("")


# ---------------------------------------------------------------- 开发装配


def test_build_dev_server_wires_a_working_session_server():
    server = build_dev_server(questions=QUESTIONS, token=TOKEN, repo=__import__(
        "ruipin.adapters.repo_sqlite", fromlist=["SqliteRepo"]
    ).SqliteRepo(":memory:"))
    assert server._verify(TOKEN) is not None  # noqa: SLF001 - 校验器就是这里要验的东西
    bridge = server._build(server._verify(TOKEN))  # noqa: SLF001
    assert bridge.session_id.startswith("dev-")
    assert bridge.config.questions == QUESTIONS


def test_build_dev_server_refuses_zero_questions():
    with pytest.raises(ValueError, match="至少要给一道题"):
        build_dev_server(questions=(), token=TOKEN)


def test_build_dev_server_accepts_injected_evaluator_and_tts():
    tts = FakeTTS()
    server = build_dev_server(
        questions=QUESTIONS, token=TOKEN, evaluator=FakeEvaluator(), tts=tts
    )
    bridge = server._build("s-x")  # noqa: SLF001
    assert bridge._tts is tts  # noqa: SLF001


# ---------------------------------------------------------------- 依赖缺失


def test_import_websockets_raises_unavailable_with_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "websockets", None)
    with pytest.raises(Unavailable) as ei:
        _import_websockets()
    assert "pip install websockets" in ei.value.reason, "错误文案必须给出可执行的下一步"


def test_websockets_installed_reports_false_when_the_library_is_absent(monkeypatch):
    monkeypatch.setitem(sys.modules, "websockets", None)
    assert websockets_installed() is False


def test_websockets_installed_reports_true_on_this_machine():
    if not websockets_installed():  # pragma: no cover - 本机已装
        pytest.skip("本机未安装 websockets")
    assert websockets_installed() is True


# ---------------------------------------------------------------- 命令行入口


def test_build_dev_plan_parses_every_flag():
    plan = build_dev_plan(
        [
            "--host", "0.0.0.0",
            "--port", "9123",
            "--path", "/interview",
            "--token", "tok",
            "--questions", "第一题, 第二题 ",
            "--log-level", "debug",
        ]
    )
    assert plan is not None
    assert (plan.host, plan.port, plan.path) == ("0.0.0.0", 9123, "/interview")
    assert plan.token == "tok"
    assert plan.questions == ("第一题", "第二题"), "逗号分隔要去空白、丢空项"
    assert plan.log_level == 10


def test_build_dev_plan_defaults_are_usable_without_any_flag():
    plan = build_dev_plan([])
    assert plan is not None
    assert plan.host == DEFAULT_HOST and plan.port == DEFAULT_PORT and plan.path == DEFAULT_PATH
    assert plan.questions, "不给题目时要退到内置示例题，不能起一个零题目的服务"
    assert plan.token, "不给令牌时要随机生成，不能是空串（空令牌等于不鉴权）"


def test_build_dev_plan_refuses_an_empty_question_list():
    assert build_dev_plan(["--questions", ",,  ,"]) is None
    assert build_dev_plan(["--questions", ""]) is None, (
        "显式给了空值必须报错：把它当成'没给参数'会静默换成默认题目"
    )


def test_main_returns_2_on_bad_arguments_and_never_starts_a_service():
    started: list = []
    assert main(["--questions", ""], runner=started.append) == 2
    assert started == [], "参数不合法时绝不能起服务"


def test_explicit_empty_questions_is_not_silently_replaced_by_defaults():
    """`--questions ""` 与"没给 `--questions`"必须走不同分支。"""
    assert len(build_dev_plan([]).questions) == 3  # noqa: SLF001
    assert build_dev_plan(["--questions", ""]) is None


def test_run_blocking_hands_the_plan_to_the_server(monkeypatch):
    """默认阻塞路径：正常情况下 `asyncio.run` 永远不返回，注入 serve 才能验通。"""
    calls: list = []

    async def fake_serve(server, *, host, port, path):
        calls.append((server, host, port, path))

    plan = DevPlan(
        server="S",  # type: ignore[arg-type]
        host="h",
        port=7,
        path="/p",
        token="t",
        questions=("q",),
    )
    assert _run_blocking(plan, serve=fake_serve) == 0
    assert calls == [("S", "h", 7, "/p")]


def test_main_passes_the_whole_plan_into_the_runner():
    seen: list = []

    def runner(plan):
        seen.append(plan)
        return 0

    assert main(["--port", "9124", "--token", "T", "--questions", "Q"], runner=runner) == 0
    assert len(seen) == 1
    plan = seen[0]
    assert plan.port == 9124 and plan.token == "T" and plan.questions == ("Q",)


def test_main_lets_keyboard_interrupt_end_cleanly():
    def runner(_plan):
        raise KeyboardInterrupt

    assert main(["--token", "T"], runner=runner) == 0, "Ctrl-C 是正常退出，不是错误"


def test_main_explains_a_taken_port_instead_of_dumping_a_traceback(caplog):
    """端口被占是开发时最常见的失败，必须给"下一步"，而不是只报 errno。

    断言走 `caplog` 而不是 `capsys`：这条消息是用 `logging` 打的，而
    `logging.basicConfig` 把 handler 挂在 **stderr** 上，`capsys` 默认根本抓不到它，
    会得到一个"测试总是失败"的假象（第一次就是这么栽的）。
    """

    def runner(_plan):
        raise OSError(10048, "address already in use")

    with caplog.at_level(logging.ERROR, logger="ruipin.ws_server"):
        assert main(["--token", "T"], runner=runner) == 3

    assert "--port" in caplog.text, "报错里必须含可执行的下一步（换个端口）"
    assert "8787" in caplog.text or "无法监听" in caplog.text, "要能看出是哪个端口失败了"


# ---------------------------------------------------------------- 默认值


def test_the_frontend_agrees_on_the_ws_endpoint():
    """端口与路径在前后端各写了一遍，是最容易漂移的常量。

    漂移的表现是"前端连不上"或"连到了别的服务"，而且**只在真的跑起来时才暴露**：
    typecheck 全绿、构建全过、两边的单元测试也全绿。
    所以这里直接读前端的源文件做机械比对，而不是再抄一个数字。

    （这也是"跨语言契约"思路的又一次应用：能被机械验证的，就不要靠口头约定。）
    """
    web = Path(__file__).resolve().parents[1] / "web" / "app"
    vite = (web / "vite.config.ts").read_text(encoding="utf-8")
    # 兜底 URL 的真相源在 settings/prefs.ts 的 resolveWsUrl（设置页引入后
    # 从 App.tsx 迁了过去），机械比对跟着真相源走。
    prefs = (web / "src" / "settings" / "prefs.ts").read_text(encoding="utf-8")

    assert f"ws://127.0.0.1:{DEFAULT_PORT}" in vite, (
        f"vite proxy 没指向后端默认端口 {DEFAULT_PORT}，前端会连到别的地方"
    )
    assert f'"{DEFAULT_PATH}"' in vite, f"vite proxy 的路径与后端 {DEFAULT_PATH} 不一致"
    assert f"ws://127.0.0.1:{DEFAULT_PORT}{DEFAULT_PATH}" in prefs, (
        f"prefs.ts 的兜底 URL 与后端默认端口 {DEFAULT_PORT} 不一致"
    )


def test_the_default_port_is_not_a_commonly_squatted_one():
    """默认端口要避开"最常被别人抢"的那几个。

    8000 就是这么被抢掉的：本机实测已被另一个服务占用。撞上时要么起不来，
    要么更糟——连上了别人的服务而自己毫无察觉。
    """
    squatted = {80, 443, 3000, 5000, 5173, 8000, 8080, 8081, 9000}
    assert DEFAULT_PORT not in squatted, f"{DEFAULT_PORT} 是常见默认端口，容易被占"
    assert 1024 < DEFAULT_PORT < 49152, "应在非特权端口区间内"


def test_defaults_are_internally_consistent():
    assert DEFAULT_HOST == "127.0.0.1", "默认只绑本地：明文 ws 对外监听会裸奔在局域网上"
    assert DEFAULT_PATH.startswith("/")
    assert DEFAULT_MAX_MESSAGE_BYTES > 0, "上限不能是不限（等于把内存交给对端）"


# ---------------------------------------------------------------- 真实网络端到端

pytestmark_e2e = pytest.mark.skipif(
    not websockets_installed(),
    reason="未安装 websockets：跳过真实网络端到端（请 pip install websockets）",
)


async def _collect(ws, bucket: list) -> None:
    """后台把收到的帧原样塞进 bucket，直到连接关闭。"""
    while True:
        try:
            bucket.append(await ws.recv())
        except Exception:  # noqa: BLE001 - 客户端侧收尾，关掉就是结束
            return


async def _wait_until(pred, *, timeout: float = 5.0) -> bool:
    """等到条件成立。**不用固定 sleep 赌时序**——那种写法在本机过、在慢 CI 随机挂。"""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        await asyncio.sleep(0.01)
    return False


def _envelope(mtype, payload: Optional[dict] = None, *, seq: int = 1) -> str:
    return encode_text(make_envelope(mtype, payload or {}, seq=seq))


class _WSClient:
    """最小客户端：**自己维护单调递增的 seq**。

    这不是仪式感：网关按 `seq` 去重/排序，seq 不变的话第二帧起会被判成
    `duplicate` 直接丢掉（实测：只有第一帧有响应，后面全无音信，
    看起来像"服务端卡住了"）。协议要求上行 seq 单调递增，真实客户端
    （`web/app` 的 `nextSeq()`）就是这么做的。
    """

    def __init__(self, ws) -> None:
        self._ws = ws
        self._seq = 0

    async def send(self, mtype, payload: Optional[dict] = None) -> None:
        self._seq += 1
        await self._ws.send(_envelope(mtype, payload, seq=self._seq))


@pytestmark_e2e
def test_end_to_end_interview_over_a_real_websocket():
    """真服务 + 真客户端跑完一整场：报告出来、TTS 二进制帧也到了。"""

    async def main():
        server = build_dev_server(questions=QUESTIONS, token=TOKEN, tts=FakeTTS())
        srv = await start_server(server, host="127.0.0.1", port=0)
        port = srv.sockets[0].getsockname()[1]
        received: list = []
        try:
            import websockets

            async with websockets.connect(
                f"ws://127.0.0.1:{port}{DEFAULT_PATH}",
                additional_headers={"Authorization": f"Bearer {TOKEN}"},
                proxy=None,
            ) as ws:
                reader = asyncio.create_task(_collect(ws, received))
                client = _WSClient(ws)
                await client.send(ClientType.SESSION_CREATE, {"candidate_id": "c-1"})
                await client.send(ClientType.CONSENT_GRANT, {"base": True})
                for _ in QUESTIONS:
                    await client.send(ClientType.ANSWER_COMMIT, {"text": "我的回答"})
                await client.send(ClientType.CONTROL_END, {})

                ok = await _wait_until(
                    lambda: any(
                        isinstance(m, str) and ServerType.REPORT_READY in m for m in received
                    )
                )
                assert ok, f"5 秒内没等到报告，收到的是：{[type(m).__name__ for m in received]}"
                reader.cancel()
        finally:
            srv.close()
            await srv.wait_closed()
        return received

    received = asyncio.run(main())

    texts = [decode_text(m) for m in received if isinstance(m, str)]
    types = [e.type for e in texts]
    assert ServerType.REPORT_READY in types
    assert types.count(ServerType.QUESTION_START) == len(QUESTIONS)
    assert ServerType.DEGRADATION_CHANGED not in types, "正常链路不该出现降级"

    report = next(e for e in texts if e.type == ServerType.REPORT_READY)
    assert report.payload["available"] is True

    binaries = [m for m in received if isinstance(m, bytes)]
    assert binaries, "TTS 音频必须以**二进制帧**上网（不是 JSON 里的 base64）"
    decoded = [decode_binary(b) for b in binaries]
    ops = {op for op, _ in decoded}
    assert Opcode.TTS_AUDIO in ops
    assert Opcode.VISEME in ops
    audio = b"".join(p for op, p in decoded if op is Opcode.TTS_AUDIO)
    assert audio, "音频负载不能是空的"
    # 口型是**一题一帧整条时间轴**，所以这里要逐帧解，不能把多帧拼起来当一条 JSON
    # （拼起来会得到 "Extra data" —— 那说明确实发了两帧，是期望的行为）。
    timelines = [
        json.loads(p.decode("utf-8")) for op, p in decoded if op is Opcode.VISEME
    ]
    assert len(timelines) == len(QUESTIONS), "每道题一帧口型时间轴"
    assert all(t["visemes"] for t in timelines), "口型时间轴不能是空的"


@pytestmark_e2e
def test_a_bad_token_is_refused_over_a_real_connection():
    async def main():
        server = build_dev_server(questions=QUESTIONS, token=TOKEN)
        srv = await start_server(server, host="127.0.0.1", port=0)
        port = srv.sockets[0].getsockname()[1]
        try:
            import websockets

            async with websockets.connect(
                f"ws://127.0.0.1:{port}{DEFAULT_PATH}",
                additional_headers={"Authorization": "Bearer wrong"},
                proxy=None,
            ) as ws:
                raw = await asyncio.wait_for(ws.recv(), timeout=5)
                return raw
        finally:
            srv.close()
            await srv.wait_closed()

    raw = asyncio.run(main())
    env = decode_text(raw)
    assert env.type == ServerType.ERROR
    assert env.payload["code"] == "unauthorized"


@pytestmark_e2e
def test_a_wrong_path_is_closed_with_1008_over_a_real_connection():
    async def main():
        server = build_dev_server(questions=QUESTIONS, token=TOKEN)
        srv = await start_server(server, host="127.0.0.1", port=0)
        port = srv.sockets[0].getsockname()[1]
        try:
            import websockets
            from websockets.exceptions import ConnectionClosed as WSClosed

            async with websockets.connect(
                f"ws://127.0.0.1:{port}/not-ws",
                additional_headers={"Authorization": f"Bearer {TOKEN}"},
                proxy=None,
            ) as ws:
                with pytest.raises(WSClosed) as ei:
                    await asyncio.wait_for(ws.recv(), timeout=5)
                return ei.value.code
        finally:
            srv.close()
            await srv.wait_closed()

    assert asyncio.run(main()) == 1008


@pytestmark_e2e
def test_a_non_ascii_bearer_token_does_not_crash_the_server():
    """端到端复现"非 ASCII 令牌"：必须得到一条 unauthorized，而不是连接崩掉。

    注意这里**只能走 query 参数**：`websockets` 客户端在设置 `Authorization`
    头时自己就会拒绝非 ASCII（`InvalidHeaderValue`），所以 header 这条路在
    Python 客户端上根本发不出去。但**服务端不能依赖这个巧合**——curl、Go、
    手写 HTTP 的客户端都不做这个校验，`Bearer 中文` 是能真的打到服务端的。
    query 会被 percent-encode，服务端 `parse_qs` 解出来正好是非 ASCII 串，
    等价于复现了那条路径。
    """
    from urllib.parse import quote

    async def main():
        server = build_dev_server(questions=QUESTIONS, token=TOKEN)
        srv = await start_server(server, host="127.0.0.1", port=0)
        port = srv.sockets[0].getsockname()[1]
        try:
            import websockets

            url = f"ws://127.0.0.1:{port}{DEFAULT_PATH}?token={quote('中文令牌')}"
            async with websockets.connect(url, proxy=None) as ws:
                return await asyncio.wait_for(ws.recv(), timeout=5)
        finally:
            srv.close()
            await srv.wait_closed()

    env = decode_text(asyncio.run(main()))
    assert env.type == ServerType.ERROR
    assert env.payload["code"] == "unauthorized"


@pytestmark_e2e
def test_serve_forever_accepts_a_connection_through_on_listening():
    """`serve_forever` 是真正的生产入口（`main` 走的就是它），必须真的能服务。"""

    async def main_():
        server = build_dev_server(questions=QUESTIONS, token=TOKEN)
        box: list = []
        ready = asyncio.Event()

        def on_listening(srv):
            box.append(srv)
            ready.set()

        task = asyncio.create_task(
            serve_forever(server, host="127.0.0.1", port=0, on_listening=on_listening)
        )
        await asyncio.wait_for(ready.wait(), timeout=5)
        srv = box[0]
        port = srv.sockets[0].getsockname()[1]
        import websockets

        try:
            async with websockets.connect(
                f"ws://127.0.0.1:{port}{DEFAULT_PATH}?token={TOKEN}", proxy=None
            ) as ws:
                await _WSClient(ws).send(ClientType.SESSION_CREATE, {"candidate_id": "c"})
                return decode_text(await asyncio.wait_for(ws.recv(), timeout=5))
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

    env = asyncio.run(main_())
    assert env.type == ServerType.STATE_CHANGED


@pytestmark_e2e
def test_serve_forever_works_without_the_on_listening_hook():
    """不传 `on_listening` 的那条路径同样要能起来并服务（分支不能只测一半）。"""
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]  # 先探一个空闲端口；这里不传 on_listening，拿不到 srv

    async def main_():
        import websockets

        server = build_dev_server(questions=QUESTIONS, token=TOKEN)
        task = asyncio.create_task(serve_forever(server, host="127.0.0.1", port=port))
        try:
            for _ in range(100):
                try:
                    async with websockets.connect(
                        f"ws://127.0.0.1:{port}{DEFAULT_PATH}?token={TOKEN}", proxy=None
                    ) as ws:
                        await _WSClient(ws).send(
                            ClientType.SESSION_CREATE, {"candidate_id": "c"}
                        )
                        return decode_text(await asyncio.wait_for(ws.recv(), timeout=5))
                except Exception as exc:  # noqa: BLE001 - 服务还在启动，重试
                    last = exc
                    await asyncio.sleep(0.02)
            raise AssertionError(f"服务一直没起来：{last!r}")
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

    assert asyncio.run(main_()).type == ServerType.STATE_CHANGED


@pytestmark_e2e
def test_token_can_be_passed_as_a_query_parameter():
    async def main():
        server = build_dev_server(questions=QUESTIONS, token=TOKEN)
        srv = await start_server(server, host="127.0.0.1", port=0)
        port = srv.sockets[0].getsockname()[1]
        try:
            import websockets

            async with websockets.connect(
                f"ws://127.0.0.1:{port}{DEFAULT_PATH}?token={TOKEN}", proxy=None
            ) as ws:
                await _WSClient(ws).send(ClientType.SESSION_CREATE, {"candidate_id": "c"})
                return await asyncio.wait_for(ws.recv(), timeout=5)
        finally:
            srv.close()
            await srv.wait_closed()

    env = decode_text(asyncio.run(main()))
    assert env.type == ServerType.STATE_CHANGED, "query 令牌必须也能建会话"


# ---------------------------------------------------------------- HTTP 旁路端点


class _FakeConnection:
    """make_llm_config_endpoint 的最小连接桩：只暴露 request + respond。"""

    def __init__(self, path: str, headers: Optional[dict] = None):
        class _Req:
            pass

        self.request = _Req()
        self.request.path = path
        self.request.headers = headers or {}
        self.responded: Optional[tuple] = None

    def respond(self, status, text):
        self.responded = (status, text)
        return (status, text)


def _noop_verify(candidate: str):
    return "s-1" if candidate == "good-token" else None


def _endpoint(slot=None, *, verify=None, apply=None):
    from ruipin.adapters.ws_server import make_llm_config_endpoint

    return make_llm_config_endpoint(
        slot if slot is not None else [object()],
        verify=verify or _noop_verify,
        apply_config=apply or (lambda cfg: (True, "ok")),
        logger=logging.getLogger("test"),
    )


def test_health_endpoint_answers_ok():
    conn = _FakeConnection("/health")
    assert _endpoint()(conn, None) == (200, '{"ok": true}')
    assert conn.responded[0] == 200


def test_non_target_paths_fall_through_to_ws_handshake():
    conn = _FakeConnection("/ws?token=good-token")
    assert _endpoint()(conn, None) is None
    conn2 = _FakeConnection("/")
    assert _endpoint()(conn2, None) is None


def test_config_llm_rejects_missing_or_wrong_token():
    bad = _endpoint()(_FakeConnection("/config/llm", {"Authorization": "Bearer wrong"}), None)
    assert bad[0] == 401
    none = _endpoint()(_FakeConnection("/config/llm"), None)
    assert none[0] == 401


def test_config_llm_accepts_token_via_query_and_header():
    seen = []

    def apply(cfg):
        seen.append(cfg)
        return True, "ok"

    a = _endpoint(apply=apply)(_FakeConnection("/config/llm?token=good-token"), None)
    assert a[0] == 200
    b = _endpoint(apply=apply)(
        _FakeConnection("/config/llm", {"Authorization": "Bearer good-token"}), None
    )
    assert b[0] == 200
    assert len(seen) == 2


def test_config_llm_passes_headers_and_replaces_slot():
    slot = [object()]
    applied = {}

    def apply(cfg):
        applied.update(cfg)
        slot[0] = "NEW"
        return True, "AI 评分已启用"

    resp = _endpoint(slot=slot, apply=apply)(
        _FakeConnection(
            "/config/llm?token=good-token",
            {"X-LLM-Base-URL": "https://api.example.com", "X-LLM-API-Key": "sk-x", "X-LLM-Model": "m-1"},
        ),
        None,
    )
    assert resp[0] == 200
    assert applied == {"base_url": "https://api.example.com", "api_key": "sk-x", "model": "m-1"}
    assert slot[0] == "NEW"
    # 响应体不回显 key
    assert "sk-x" not in resp[1]


def test_config_llm_partial_config_is_rejected_with_reason():
    resp = _endpoint(apply=lambda cfg: (False, "配置不完整"))(
        _FakeConnection("/config/llm?token=good-token", {"X-LLM-Base-URL": "https://x"}), None
    )
    assert resp[0] == 400
    assert "配置不完整" in resp[1]


def test_config_llm_apply_exception_becomes_500():
    def boom(cfg):
        raise RuntimeError("装配炸了")

    resp = _endpoint(apply=boom)(_FakeConnection("/config/llm?token=good-token"), None)
    assert resp[0] == 500
    assert "炸了" not in resp[1], "异常细节不能回显给客户端"


def test_health_and_config_llm_over_real_server():
    """集成：真起一个服务，HTTP 旁路端点与 WS 握手共存。"""
    httpx = pytest.importorskip("httpx")

    async def _scenario():
        def build_bridge(session_id):
            raise AssertionError("本用例不应建立 WS 会话")

        server = _SessionServerForHttpTest(
            build_bridge=build_bridge, token_verifier=lambda c: "s" if c == "good-token" else None
        )
        server.evaluator_slot = [FakeEvaluator()]

        def apply(cfg):
            server.evaluator_slot[0] = "REPLACED"
            return True, "ok"

        srv = await start_server(
            server,
            host="127.0.0.1",
            port=0,
            process_request=_endpoint(slot=server.evaluator_slot, apply=apply),
        )
        port = srv.sockets[0].getsockname()[1]
        try:
            async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}") as client:
                r = await client.get("/health")
                assert r.status_code == 200 and r.json() == {"ok": True}
                r2 = await client.get(
                    "/config/llm?token=good-token",
                    headers={"X-LLM-Base-URL": "https://a", "X-LLM-API-Key": "k", "X-LLM-Model": "m"},
                )
                assert r2.status_code == 200 and r2.json()["ok"] is True
                assert server.evaluator_slot[0] == "REPLACED"
                r3 = await client.get("/config/llm")  # 无令牌
                assert r3.status_code == 401
        finally:
            srv.close()
            await srv.wait_closed()

    asyncio.run(_scenario())
