"""适配器层：端口（Port）的具体实现。换实现 = 只改这里，核心层不动。

本包统一收口"外部世界长什么样"：**核心层（`domain` / `orchestrator` / `scoring`）
只认识 `ruipin.ports` 里的 Protocol，任何真实依赖都长在这一层。**

命名约定
--------
* `Http*` 前缀 / 后缀 → 真实网络适配器（走 `http.HttpTransport`，可注入替身）
* `Fake*` / `Deterministic*` → 测试替身，确定性、零外部依赖
* `Sqlite*` → 本地存储适配器（生产将替换为 SQLAlchemy 版本，端口不变）

为什么不用 `from .x import *` 逐个星号导出
------------------------------------------
`OPS_LATENCY_KEY` 这个名字在 `asr` / `tts` / `vlm` / `llm_openai` 四个模块里**各有一个**。
星号导入会互相覆盖，最后到底是哪一个取决于导入顺序——这种"看起来能用、
换一行导入顺序就变"的歧义在观测埋点上是会出事故的。
因此本文件**显式列出**每个要公开的名字，重复的常量一律不导出，
需要时按模块取（如 `from ruipin.adapters.tts import OPS_LATENCY_KEY`）。

所有适配器都只依赖标准库（HTTP 走 `http.HttpTransport` 抽象），
因此本包可以在没有网络、没有密钥的环境下被完整导入——这是"能跑测试"的前提。
"""

from __future__ import annotations

from . import (  # noqa: F401 - 子模块本身也是公开面（按模块取重名常量）
    align_qwen3,
    asr,
    avatar,
    fakes,
    http,
    llm_openai,
    llm_router,
    repo_sqlite,
    tts,
    tts_qwen3,
    vlm,
    ws_server,
)
from .align_qwen3 import Qwen3ForcedAligner, words_span_ms
from .asr import ASRClient, HttpASR
from .avatar import RigFrame, RigTimeline, SyncChecker, SyncReport, build_timeline
from .fakes import (
    DEFAULT_DIMS,
    DeterministicClock,
    FakeEvaluator,
    FakeTTS,
    FakeVLMEvaluator,
)
from .http import (
    FakeStreamTransport,
    FakeTransport,
    HttpResponse,
    HttpStreamTransport,
    HttpTransport,
    HttpxStreamTransport,
    HttpxTransport,
    StreamEvent,
)
from .llm_openai import OpenAICompatEvaluator, api_key_from_env
from .llm_router import EvaluatorRouter
from .repo_sqlite import SqliteRepo
from .tts import Aligner, HttpTTS, TermDictionary, TTSChunk, TTSRequestError
from .tts_qwen3 import Qwen3TTSClient, WavStreamSplitter, parse_wav_header
from .vlm import CloudVLM, VLMClient, VLMResponse, estimate_vlm_cost_usd
from .ws_server import (
    DEFAULT_MAX_MESSAGE_BYTES,
    DEFAULT_PATH,
    DEFAULT_PORT,
    WebsocketsConnection,
    build_dev_server,
    serve_forever,
    start_server,
    token_from_request,
    websockets_installed,
)

__all__ = [
    # 子模块（用于取重名常量，如 adapters.tts.OPS_LATENCY_KEY）
    "align_qwen3",
    "asr",
    "avatar",
    "fakes",
    "http",
    "llm_openai",
    "llm_router",
    "repo_sqlite",
    "tts",
    "tts_qwen3",
    "vlm",
    "ws_server",
    # 存储
    "SqliteRepo",
    # 测试替身
    "DEFAULT_DIMS",
    "DeterministicClock",
    "FakeEvaluator",
    "FakeTTS",
    "FakeVLMEvaluator",
    # 评估（LLM）
    "OpenAICompatEvaluator",
    "EvaluatorRouter",
    "api_key_from_env",
    # 感知（ASR / VLM）
    "ASRClient",
    "HttpASR",
    "CloudVLM",
    "VLMClient",
    "VLMResponse",
    "estimate_vlm_cost_usd",
    # 面试官输出（TTS / 形象）
    "Aligner",
    "HttpTTS",
    "TTSChunk",
    "TTSRequestError",
    "TermDictionary",
    "Qwen3TTSClient",
    "Qwen3ForcedAligner",
    "WavStreamSplitter",
    "parse_wav_header",
    "words_span_ms",
    "RigFrame",
    "RigTimeline",
    "SyncChecker",
    "SyncReport",
    "build_timeline",
    # HTTP 传输（所有真实适配器的共同底座）
    "HttpTransport",
    "HttpResponse",
    "HttpStreamTransport",
    "HttpxTransport",
    "HttpxStreamTransport",
    "StreamEvent",
    "FakeTransport",
    "FakeStreamTransport",
    # WebSocket 传输（真实网络的接线；导入本包**不会**触发 websockets 的导入）
    "WebsocketsConnection",
    "build_dev_server",
    "serve_forever",
    "start_server",
    "token_from_request",
    "websockets_installed",
    "DEFAULT_MAX_MESSAGE_BYTES",
    "DEFAULT_PORT",
    "DEFAULT_PATH",
]
