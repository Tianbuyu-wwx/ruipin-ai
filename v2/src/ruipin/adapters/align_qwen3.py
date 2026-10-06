"""Qwen3 强制对齐器：给**已合成的音频**补词级时间轴，驱动口型。

为什么强制对齐是主方案
----------------------
`docs/开源TTS选型研究-2026-10.md` 的结论：**自托管开源 TTS 权重里没有任何一款
原生输出词级时间戳**。官方托管 API 有（`word_timestamp_enabled`），但那是 API
不是权重，与"可自托管"冲突。所以词时间轴由本模块独立产出，`HttpTTS` 会把
`timing_source` 标成 `forced_alignment`。这与方案 §6.5 的预判一致。

`Qwen3-ForcedAligner-0.6B` 与阿里云托管 API 同门，输入 (音频, 文本) 输出逐词起止
时间。vLLM-Omni 也可以把它接进 TTS 的流式前向（`--forced-aligner`），走同一条链路。

**单次对齐的音频上限是 180 秒（3 分钟）**，取自官方 `qwen-asr` 0.0.6
`inference/utils.py` 的 `MAX_FORCE_ALIGN_INPUT_SECONDS = 180`（实测取证）。
超过上限的音频必须**在调用方按静音切段后分别对齐**，不要指望服务端分块——
服务端不会替你做，它只会报错或截断。这也顺带说明：一轮对话的应答音频
（几秒到几十秒）远在上限之内，不必为此加切段逻辑。

**为什么是 async**
-------------------
一次强制对齐要 100–300 ms。如果写成同步阻塞调用，这 300 ms 会把整个事件循环
按在地上——同一进程里其他会话的心跳、评估推送、音频下行全部一起卡住。
所以这个适配器是异步的，`HttpTTS` 也相应地支持 await（见 `tts.py::_resolve_timings`）。

解析纪律：宁可响亮地失败，也不要产出一份看起来正常的错时间轴
--------------------------------------------------------------
**单位歧义是本模块的头号风险。** 若服务端给的是秒而被当成毫秒，口型会以
**1000 倍**速错位；反之则慢 1000 倍。两种都是"图能画出来、看起来正常、但完全错
的"输出，而且从画面上看不出来——所以这里**不做单位猜测**，只接受两种"单位写在
字段名里"的方言，每种方言的单位是固定的：

* `{"word": str, "start_ms": int, "end_ms": int}` —— **毫秒**（本项目自有服务契约）
* `{"text": str, "start_time": number, "end_time": number}` —— **秒**
  （官方 `qwen-asr` 包 `Qwen3ForcedAligner.align()` 的原生输出）

秒方言的来历（2026-10 实测取证，非文档推测）：读 `qwen_asr` 0.0.6 的
`inference/qwen3_forced_aligner.py` 确认，`parse_timestamp()` 产出
`{"text", "start_time", "end_time"}`，随后 `align()` 里显式做
`round(it['start_time'] / 1000.0, 3)` —— **单位是秒、精度到毫秒**。
同时确认**中文按单字切分**（`tokenize_chinese_mixed` 对每个 CJK 字符单独出一项），
所以中文下 `granularity` 实际是字级，`GRANULARITY_WORD` 只是请求侧提示。

两种方言的字段名集合不重叠（`word` vs `text`、`start_ms` vs `start_time`），
混用（例如 `word` 配 `start_time`）一律抛错，而不是"挑一个能用的"。
一份响应里所有条目必须是同一种方言。容器可以是顶层数组 / `{"words": [...]}` /
`{"segments": [{"words": [...]}, ...]}`——这三种只是"把同一种条目装在哪里"，
与单位无关。

条目字段自相矛盾（`end < start`、负数、`end_time` 早于 `start_time`）一律抛错
而不是"修正"。

⚠️ **仍未证实的部分**：vLLM-Omni 的 `/v1/audio/align` 实际返回形状没验过
（本机 WSL 被安全策略禁用，跑不了 vLLM）。上面第二种方言是**官方推理包**的形状，
不是 vLLM-Omni 的形状。生产若改走 vLLM-Omni，这一处需要重新取证。
"""

from __future__ import annotations

import base64
import json
from typing import Any, Mapping, Optional, Sequence

from ..domain.errors import Unavailable
from ..ports import WordTiming
from .http import DEFAULT_TIMEOUT_S, HttpTransport, HttpResponse

#: vLLM-Omni 强制对齐后端的默认端口（见选型研究附注）。
DEFAULT_BASE_URL = "http://127.0.0.1:8024"

#: 对齐端点。
DEFAULT_ALIGN_PATH = "/v1/audio/align"

#: provider 名。进 `Unavailable.provider`，看板按它聚合。
PROVIDER = "qwen3-forced-aligner"

#: 粒度。词级即可驱动口型；字级对中文更细，需要时再切。
GRANULARITY_WORD = "word"
GRANULARITY_CHAR = "char"


#: 条目方言：`(词键, 起始键, 结束键, 秒→毫秒因子)`。因子为 1 表示本来就是毫秒。
#: 两种方言的键名集合不重叠，因此"混用"是可检测的，不需要按值域猜单位。
_DIALECT_MS: tuple[str, str, str, float] = ("word", "start_ms", "end_ms", 1.0)
_DIALECT_SECONDS: tuple[str, str, str, float] = ("text", "start_time", "end_time", 1000.0)
_DIALECTS = (_DIALECT_MS, _DIALECT_SECONDS)


def _detect_dialect(items: Sequence[Any]) -> tuple[str, str, str, float]:
    """按**第一项**的键名判定整份响应用哪种方言，并校验其余项一致。

    为什么按第一项判定而不是逐项自适应：逐项自适应会让一份"半秒半毫秒"的脏响应
    被拼成一条看似完整的时间轴。按第一项定调、其余不一致就抛，脏数据会立刻暴露。

    Raises:
        ValueError: 无一方言匹配，或各项方言不一致。
    """
    if not items:
        raise ValueError("响应里没有任何条目，无法判定方言（空时间轴请返回空数组）")
    first = items[0]
    if not isinstance(first, Mapping):
        raise ValueError(f"第 0 项不是对象：{type(first).__name__}")
    for word_key, start_key, end_key, factor in _DIALECTS:
        if word_key in first or start_key in first or end_key in first:
            chosen = (word_key, start_key, end_key, factor)
            break
    else:
        raise ValueError(
            f"第 0 项的字段不认识，期望 {'/'.join(d[0] for d in _DIALECTS)} 之一："
            f"{dict(first)!r}"
        )
    word_key, start_key, end_key, _ = chosen
    for i, item in enumerate(items):
        if not isinstance(item, Mapping):
            raise ValueError(f"第 {i} 项不是对象：{type(item).__name__}")
        present = {k for k in (word_key, start_key, end_key) if k in item}
        if present != {word_key, start_key, end_key}:
            raise ValueError(
                f"第 {i} 项的字段与首项方言（{word_key}/{start_key}/{end_key}）不一致："
                f"{dict(item)!r}"
            )
    return chosen


def _parse_words(data: Any) -> list[WordTiming]:
    """把对齐服务的响应体解析成词时间轴。**形状/单位不符就抛，不猜。**

    Raises:
        ValueError: 容器形状不认识，或某一条目的字段缺失/类型不对/自相矛盾。
    """
    items = _collect_items(data)
    if not items:
        return []
    word_key, start_key, end_key, factor = _detect_dialect(items)
    out: list[WordTiming] = []
    for i, item in enumerate(items):
        word = item[word_key]
        if not isinstance(word, str):
            raise ValueError(f"第 {i} 项的 {word_key} 不是字符串：{word!r}")
        start = item[start_key]
        end = item[end_key]
        # bool 是 int 的子类，会在这里伪装成数值，必须先挡掉。
        if isinstance(start, bool) or isinstance(end, bool):
            raise ValueError(f"第 {i} 项的 {start_key}/{end_key} 是布尔值：{dict(item)!r}")
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            raise ValueError(
                f"第 {i} 项的 {start_key}/{end_key} 不是数值：{dict(item)!r}"
            )
        start_ms = int(round(start * factor))
        end_ms = int(round(end * factor))
        if start_ms < 0:
            raise ValueError(f"第 {i} 项的 {start_key} 为负：{start!r}")
        if end_ms < start_ms:
            raise ValueError(
                f"第 {i} 项的 {end_key}({end!r}) 早于 {start_key}({start!r})"
            )
        out.append(WordTiming(word=word, start_ms=start_ms, end_ms=end_ms))
    return out


def _collect_items(data: Any) -> list[Any]:
    """把三种"同一种条目装在不同的容器里"的形状归一成条目列表。

    顶层数组 /
    `{"words": [...]}` /
    `{"segments": [{"words": [...]}, ...]}`（展平）
    """
    if isinstance(data, list):
        return data
    if isinstance(data, Mapping):
        if isinstance(data.get("words"), list):
            return list(data["words"])
        segments = data.get("segments")
        if isinstance(segments, list):
            flat: list[Any] = []
            for seg in segments:
                if not isinstance(seg, Mapping) or not isinstance(seg.get("words"), list):
                    raise ValueError(f"segments 里的元素不含 words 数组：{seg!r}")
                flat.extend(seg["words"])
            return flat
    raise ValueError(
        "对齐响应形状不认识（期望顶层数组 / {'words': [...]} / {'segments': [...]}）："
        f"{type(data).__name__}"
    )


class Qwen3ForcedAligner:
    """`tts.Aligner` 的实现：POST 到强制对齐服务，返回词时间轴。

    实例本身是**可调用对象**（`__call__` 是 async），因此可以直接当
    `HttpTTS(aligner=...)` 的实参用，不需要额外的包装函数。

    Args:
        transport: HTTP 传输（`http.HttpTransport`；测试用 `FakeTransport`）。
        base_url: 对齐服务根地址。
        language: 语言提示（服务端支持时提升准确率）。
        granularity: `"word"` 或 `"char"`。
        api_key: 非空时加 `Authorization: Bearer`。
        timeout_s: 单次对齐超时（透传给传输层，不藏在 transport 里）。
        extra: 额外透传字段；**核心字段优先级更高**（防止被静默改掉）。
    """

    def __init__(
        self,
        transport: HttpTransport,
        *,
        base_url: str = DEFAULT_BASE_URL,
        language: str = "",
        granularity: str = GRANULARITY_WORD,
        api_key: str = "",
        timeout_s: float = DEFAULT_TIMEOUT_S,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if granularity not in (GRANULARITY_WORD, GRANULARITY_CHAR):
            raise ValueError(
                f"granularity 只支持 {GRANULARITY_WORD!r} / {GRANULARITY_CHAR!r}，"
                f"实际: {granularity!r}"
            )
        if not isinstance(timeout_s, (int, float)) or timeout_s <= 0:
            raise ValueError(f"timeout_s 必须为正数，实际: {timeout_s!r}")
        self.transport = transport
        self.base_url = base_url.rstrip("/")
        self.language = language
        self.granularity = granularity
        self.api_key = api_key
        self.timeout_s = float(timeout_s)
        self.extra = dict(extra or {})

    @property
    def url(self) -> str:
        return f"{self.base_url}{DEFAULT_ALIGN_PATH}"

    def build_request(
        self, audio: bytes, text: str, timeout_s: float
    ) -> tuple[str, dict[str, str], dict[str, Any], float]:
        """构造 (url, headers, payload, timeout)。

        音频走 base64 内联而不是 multipart：`HttpTransport` 的契约是"发一个 JSON
        POST"，走 multipart 就得给它加第二种请求形状——为了一处调用点把公共底座
        撑大不划算。代价是 base64 有 ~33% 体积膨胀，对齐用的短音频（一句话）可接受。
        """
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload: dict[str, Any] = dict(self.extra)
        payload.update(
            {
                "audio": base64.b64encode(audio).decode("ascii"),
                "text": text,
                "granularity": self.granularity,
            }
        )
        if self.language:
            payload["language"] = self.language
        return self.url, headers, payload, timeout_s

    async def __call__(self, audio: bytes, text: str) -> list[WordTiming]:
        """对齐 `audio` 与 `text`。

        Returns:
            词时间轴。**合法但为空**（例如音频是静音）时返回 `[]`——这不是失败，
            而是"对齐跑了、确实没有词"。

        Raises:
            Unavailable: 请求失败 / 非 2xx / 响应不可解析 / 字段自相矛盾。
                **绝不返回"看起来合理"的时间轴来兜底**。
        """
        if not audio:
            raise Unavailable(PROVIDER, "音频为空，拒绝对齐（空音频无法与文本对齐）")
        if not text:
            raise Unavailable(PROVIDER, "文本为空，拒绝对齐")

        url, headers, payload, timeout_s = self.build_request(
            audio, text, self.timeout_s
        )
        try:
            resp = await self.transport.post_json(url, headers, payload, timeout_s)
        except Unavailable:
            raise
        except Exception as exc:  # 连接重置 / DNS / 超时
            raise Unavailable(
                PROVIDER, f"对齐请求失败 {type(exc).__name__}: {exc}"
            ) from exc

        return self._interpret(resp)

    def _interpret(self, resp: HttpResponse) -> list[WordTiming]:
        """把 HTTP 结果翻译成词时间轴（或抛 `Unavailable`）。独立成方法便于单测。"""
        if not resp.ok:
            detail = resp.text.strip()[:300]
            raise Unavailable(PROVIDER, f"对齐服务返回 HTTP {resp.status}: {detail}")
        try:
            data = json.loads(resp.text)
        except ValueError as exc:
            raise Unavailable(PROVIDER, f"对齐响应不是合法 JSON: {exc}") from exc
        try:
            return _parse_words(data)
        except ValueError as exc:
            raise Unavailable(PROVIDER, f"对齐响应不可解析：{exc}") from exc


def words_span_ms(words: Sequence[WordTiming]) -> int:
    """时间轴覆盖的时长（ms）。空时间轴返回 0。供验证工具与诊断使用。"""
    if not words:
        return 0
    return max(int(w.end_ms) for w in words)


__all__ = [
    "DEFAULT_ALIGN_PATH",
    "DEFAULT_BASE_URL",
    "GRANULARITY_CHAR",
    "GRANULARITY_WORD",
    "PROVIDER",
    "Qwen3ForcedAligner",
    "words_span_ms",
]
