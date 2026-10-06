"""跨语言线序契约的**后端一侧**：保证夹具不过期。

夹具（`web/app/src/protocol/wireFixtures.json`）是前端的输入，但它的来源是后端。
如果只在后端改代码、不重新导出，前端那份测试会**继续对着旧夹具全绿**——
这种"假绿"比没有测试更危险，因为它让人以为契约被守住了。

所以这里做一件事：**重新跑一遍后端链路，逐字节比对磁盘上的夹具**。
不一致就红，并直接给出修复命令。

相关的另一侧在 `web/app/src/protocol/wireContract.test.ts`。
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_V2_ROOT = Path(__file__).resolve().parents[1]
_TOOL_PATH = _V2_ROOT / "tools" / "export_wire_fixtures.py"


def _load_tool():
    """按文件路径加载导出工具（`tools/` 不在 pytest 的 pythonpath 里）。"""
    spec = importlib.util.spec_from_file_location("_export_wire_fixtures", _TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_tool = _load_tool()
FIXTURE_PATH: Path = _tool.FIXTURE_PATH


def test_the_fixture_file_is_committed_where_the_frontend_can_import_it():
    assert FIXTURE_PATH.exists(), (
        f"线序契约夹具缺失：{FIXTURE_PATH}；"
        "请运行 python tools/export_wire_fixtures.py"
    )
    assert "web" in FIXTURE_PATH.parts and "app" in FIXTURE_PATH.parts, (
        "夹具必须落在前端源码树里，否则前端 import 不到"
    )


def test_the_committed_fixture_is_not_stale():
    """夹具必须与**当前后端**逐字节一致。过期夹具会让前端测试变成假绿。"""
    on_disk = FIXTURE_PATH.read_text(encoding="utf-8")
    fresh = _tool.dump(_tool.build_fixture())
    if on_disk != fresh:
        pytest.fail(
            "夹具已过期：web/app/src/protocol/wireFixtures.json 与当前后端产物不一致。\n"
            "修复：python tools/export_wire_fixtures.py\n"
            "（若前端因格式变化而失败，说明这次改动真的动了线序契约。）"
        )


def test_the_fixture_carries_everything_the_frontend_needs():
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    for key in (
        "opcodes",
        "serverTypes",
        "audioHex",
        "ttsFramesHex",
        "visemeFrameHex",
        "visemes",
    ):
        assert key in fixture, f"夹具缺字段 {key}（前端会读不到）"
    assert fixture["ttsFramesHex"], "至少要有一帧音频，否则前端测的是一条空路径"
    assert fixture["visemes"], "口型时间轴不能是空的"


def test_the_fixture_matches_the_python_side_decoding():
    """把夹具解一遍，确认它确实能被后端自己的解码器读回（自洽性）。"""
    from ruipin.transport import Opcode, decode_binary

    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    audio = b""
    for raw_hex in fixture["ttsFramesHex"]:
        opcode, payload = decode_binary(bytes.fromhex(raw_hex))
        assert opcode is Opcode.TTS_AUDIO
        audio += payload
    assert audio.hex() == fixture["audioHex"], "逐帧拼接必须等于整段音频"

    opcode, payload = decode_binary(bytes.fromhex(fixture["visemeFrameHex"]))
    assert opcode is Opcode.VISEME
    assert json.loads(payload.decode("utf-8"))["visemes"] == fixture["visemes"]


def test_the_tool_can_check_without_writing(tmp_path, capsys):
    """`--check` 是 CI 用的入口，必须"只读"且能报出过期。"""
    assert _tool.main(["--check"]) == 0

    original = FIXTURE_PATH.read_text(encoding="utf-8")
    try:
        FIXTURE_PATH.write_text(original.replace("请介绍一下你自己", "被改过的题目"), encoding="utf-8")
        assert _tool.main(["--check"]) == 1, "过期夹具必须让 --check 以非零码退出"
    finally:
        FIXTURE_PATH.write_text(original, encoding="utf-8")
        assert _tool.main(["--check"]) == 0


def test_the_tool_reports_a_missing_fixture(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(_tool, "FIXTURE_PATH", tmp_path / "nope.json")
    assert _tool.main(["--check"]) == 1
