"""语音识别（ASR）适配器：流式累积 + 可复位。

方案出处（`docs/锐聘AI-v2-技术方案.md`）
--------------------------------------
* **§7.1** 一轮回答的时序里，候选人语音经 ASR 得到转写文本，供评分链路使用。
* 本模块只做**接口编排**：把音频块累积起来交给注入的 `ASRClient`，把结果透传。

红线：**绝不编造转写文本**
--------------------------
识别不出东西时返回**空串**，而不是 `"嗯"` / `"未知"` / 上一块的残留。
`finalize()` 返回的也必须是引擎真实给出的文本；引擎给空就是空。任何"猜一个像样的
答案填进去"的做法都会污染后续评分——比空转写更糟（空转写可被上层显式当"本轮未走
语音"处理）。

缓冲语义
--------
`feed` 累积音频并返回当前 partial（允许空串）；`finalize` 交出最终文本并把缓冲
清空，因此同一实例可直接用于下一轮（复位）。空缓冲直接返回空串，不发起无意义调用。
"""

from __future__ import annotations

import time
from typing import Any, Callable, Optional, Protocol, runtime_checkable

from ..domain.errors import Unavailable
from ..ports import Usage
from .http import DEFAULT_TIMEOUT_S

#: 计量埋点名。
OPS_LATENCY_KEY = "asr.transcribe"


@runtime_checkable
class ASRClient(Protocol):
    """低层 ASR 客户端（生产实现负责真实端点）。

    无状态：每次收到**到目前为止累积的**音频与 `is_final` 标志，返回当前最佳文本。
    """

    async def transcribe(self, audio: bytes, *, is_final: bool) -> str: ...


class HttpASR:
    """`ASRPort` 的实现：累积音频、透传转写、复位与计量。

    Args:
        name: provider 名（进计量）。
        client: 低层 ASR 客户端（见 `ASRClient`）。
        meter: 可观测端口；None 则跳过计量。
        timeout_s: 单次请求超时（透传给客户端）。
        clock: 注入式时钟，保证测试确定性。
    """

    def __init__(
        self,
        name: str,
        client: ASRClient,
        *,
        meter: Optional[Any] = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.name = name
        self._client = client
        self.meter = meter
        self.timeout_s = float(timeout_s)
        self._clock = clock
        self._buffer = bytearray()
        self._n_chunks = 0

    # ---------- ASRPort ----------

    @property
    def n_bytes(self) -> int:
        """当前缓冲的音频字节数（供测试 / 诊断）。"""
        return len(self._buffer)

    @property
    def n_chunks(self) -> int:
        """已累积的音频块数（空块不计）。"""
        return self._n_chunks

    async def feed(self, chunk: bytes) -> Optional[str]:
        """累积一块音频并返回当前 partial（可能为空串）。"""
        if chunk:
            self._buffer.extend(chunk)
            self._n_chunks += 1
        if not self._buffer:
            # 还没有任何音频：返回空串，不发起调用、更不编造内容。
            return ""
        return await self._call(bytes(self._buffer), is_final=False)

    async def finalize(self) -> str:
        """返回最终文本并复位。空缓冲直接返回 `""`（不是填充词）。"""
        if not self._buffer:
            self._reset()
            return ""
        audio = bytes(self._buffer)
        try:
            return await self._call(audio, is_final=True)
        finally:
            self._reset()

    def reset(self) -> None:
        """丢弃当前缓冲，显式复位（下一轮复用同一实例）。"""
        self._reset()

    # ---------- 内部 ----------

    async def _call(self, audio: bytes, *, is_final: bool) -> str:
        start = self._clock()
        try:
            text = await self._client.transcribe(audio, is_final=is_final)
        except Unavailable:
            raise
        except Exception as exc:  # 连接重置 / DNS / 超时 / 客户端内部错误
            self._incr("asr.error")
            raise Unavailable(self.name, f"ASR 请求失败 {type(exc).__name__}: {exc}") from exc
        latency_ms = int(round((self._clock() - start) * 1000.0))
        clean = self._clean(text)
        self._report(latency_ms)
        return clean

    def _clean(self, text: Any) -> str:
        if not isinstance(text, str):
            raise Unavailable(self.name, f"客户端返回非字符串转写: {type(text).__name__}")
        return text.strip()

    def _reset(self) -> None:
        self._buffer = bytearray()
        self._n_chunks = 0

    def _incr(self, name: str) -> None:
        if self.meter is None:
            return
        self.meter.incr(name)

    def _report(self, latency_ms: int) -> None:
        if self.meter is None:
            return
        self.meter.record_usage(
            Usage(provider=self.name, operation=OPS_LATENCY_KEY, latency_ms=latency_ms)
        )
        self.meter.observe_ms(OPS_LATENCY_KEY, float(latency_ms))


__all__ = ["OPS_LATENCY_KEY", "ASRClient", "HttpASR"]
