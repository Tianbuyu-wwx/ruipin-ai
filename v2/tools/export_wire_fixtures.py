"""导出**跨语言线序契约夹具**：后端真实产物 → 前端可读的 JSON。

为什么需要这个文件
------------------
后端（Python）与前端（TypeScript）各自实现了一套帧编解码，两边都测得很认真，
但**两套实现之间没有任何测试**。这类"两侧各自全绿、中间对不上"的接缝，
恰恰是最容易出事的：编码器改了 4 字节头、viseme 的 JSON 键名从 `visemes`
改成 `timeline`、opcode 从 0x11 改成 0x12——任一处改动，两侧的单元测试
**都不会红**，而用户看到的是"没有声音/口型乱动"。

于是这里把后端的**真实产物**（不是手写的样例字节）落成夹具：
`web/app/src/protocol/wireFixtures.json`。前端有测试直接消费它，
`tests/test_wire_fixtures.py` 则保证夹具**没有过期**（与当前后端产物逐字节一致）。

两者合起来形成一条闭环：后端改了 → Python 契约测试先红（提醒重新导出）
→ 导出后前端测试再红（提醒前端同步）。任何一侧掉队都会被立刻发现。

用法::

    python tools/export_wire_fixtures.py          # 导出
    python tools/export_wire_fixtures.py --check  # 只校验，不写盘（CI 用）

夹具必须**完全确定**：固定的题干、固定的替身 TTS、固定的时钟。
二进制帧本身不含时间戳，所以夹具里也没有会漂移的字段。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from ruipin.adapters.fakes import FakeEvaluator, FakeTTS  # noqa: E402
from ruipin.adapters.repo_sqlite import SqliteRepo  # noqa: E402
from ruipin.orchestrator import TurnScheduler  # noqa: E402
from ruipin.transport import (  # noqa: E402
    BridgeConfig,
    ClientType,
    InterviewBridge,
    Opcode,
    ServerType,
    decode_binary,
    make_envelope,
)

#: 夹具落盘位置（前端源码树内，便于前端直接 import）。
FIXTURE_PATH = _REPO_ROOT / "web" / "app" / "src" / "protocol" / "wireFixtures.json"

#: 固定题干。**改这里就等于改了契约**，前后端都要跟着动。
FIXTURE_QUESTION = "请介绍一下你自己"

#: 固定音频分片大小（取小值，好让夹具里出现多帧）。
FIXTURE_CHUNK_BYTES = 16


class _FixedClock:
    """固定时钟：夹具里不能有会漂移的字段。"""

    def __init__(self, t: float = 1_700_000_000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def build_fixture() -> dict[str, Any]:
    """跑一遍**真实的**后端链路，产出确定性夹具字典。"""

    async def go() -> list:
        bridge = InterviewBridge(
            repo=SqliteRepo(":memory:"),
            scheduler=TurnScheduler(FakeEvaluator(), clock=_FixedClock()),
            config=BridgeConfig(questions=(FIXTURE_QUESTION,)),
            session_id="s-fixture",
            clock=_FixedClock(),
            tts=FakeTTS(chunk_size=4, n_chunks=3),
            tts_chunk_bytes=FIXTURE_CHUNK_BYTES,
        )
        await bridge.handle(make_envelope(ClientType.SESSION_CREATE, {}))
        immediate = await bridge.handle(make_envelope(ClientType.CONSENT_GRANT, {"base": True}))
        await bridge.joined()
        return immediate + bridge.take_pending()

    frames = asyncio.run(go())

    tts_frames: list[bytes] = []
    viseme_frames: list[bytes] = []
    from ruipin.transport import split_outgoing

    _, binaries = split_outgoing(frames)
    for raw in binaries:
        opcode, payload = decode_binary(raw)
        if opcode is Opcode.TTS_AUDIO:
            tts_frames.append(raw)
        elif opcode is Opcode.VISEME:
            viseme_frames.append(raw)

    if not tts_frames or len(viseme_frames) != 1:
        raise AssertionError(
            f"夹具假设每道题产出多个音频帧 + 恰好一帧口型，实际 "
            f"音频 {len(tts_frames)} / 口型 {len(viseme_frames)}"
        )

    audio = b"".join(decode_binary(f)[1] for f in tts_frames)
    viseme_payload = decode_binary(viseme_frames[0])[1]
    visemes = json.loads(viseme_payload.decode("utf-8"))["visemes"]

    return {
        "note": (
            "由 tools/export_wire_fixtures.py 从后端真实产物导出，请勿手工编辑。"
            "前端 wireContract.test.ts 消费本文件；tests/test_wire_fixtures.py 保证它不过期。"
        ),
        "generatedBy": "tools/export_wire_fixtures.py",
        "question": FIXTURE_QUESTION,
        "ttsChunkBytes": FIXTURE_CHUNK_BYTES,
        "opcodes": {
            "TTS_AUDIO": int(Opcode.TTS_AUDIO),
            "VISEME": int(Opcode.VISEME),
            "MEDIA_AUDIO": int(Opcode.MEDIA_AUDIO),
        },
        "serverTypes": {
            "QUESTION_TTS_CHUNK": str(ServerType.QUESTION_TTS_CHUNK),
            "AVATAR_VISEME": str(ServerType.AVATAR_VISEME),
        },
        "audioHex": audio.hex(),
        "ttsFramesHex": [f.hex() for f in tts_frames],
        "visemeFrameHex": viseme_frames[0].hex(),
        "visemes": visemes,
    }


def dump(fixture: dict[str, Any]) -> str:
    """序列化成稳定的文本（键序固定 + 尾随换行），便于 diff 与逐字节比对。"""
    return json.dumps(fixture, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="只校验磁盘上的夹具是否与当前后端一致（不写盘）；不一致时返回 1",
    )
    args = parser.parse_args(argv)

    text = dump(build_fixture())
    if args.check:
        if not FIXTURE_PATH.exists():
            print(f"[夹具缺失] {FIXTURE_PATH}", file=sys.stderr)
            return 1
        if FIXTURE_PATH.read_text(encoding="utf-8") != text:
            print(
                f"[夹具过期] {FIXTURE_PATH} 与当前后端产物不一致；"
                "请运行 python tools/export_wire_fixtures.py 重新导出",
                file=sys.stderr,
            )
            return 1
        print(f"[夹具最新] {FIXTURE_PATH}")
        return 0

    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE_PATH.write_text(text, encoding="utf-8")
    print(f"[已导出] {FIXTURE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
