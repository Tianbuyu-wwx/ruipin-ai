"""云端视觉（VLM）适配器：多帧 + prompt → 结构化观察，带**成本护栏**。

方案出处（`docs/锐聘AI-v2-技术方案.md`）
--------------------------------------
* **§4.1**：**本地不保留任何视觉大模型**。本地/浏览器只做轻量 ROI 检测与裁剪，
  **语义理解全部上云**。本模块即"上云"那一步，本地无任何视觉模型权重。
* **§4.2**：**不要逐帧调用**——每轮把 8–12 帧**打包成一次多图请求**，一次输出
  结构化 JSON；否则 prompt token 成本翻 N 倍。故 `max_frames` 是硬闸。
* **§4.3**：强制 `response_format=json_schema`，输出 schema 固定；每个判断带
  `confidence` 与"证据帧序号"，便于抽样复核与争议回溯。
* **§4.4**：**成本护栏**——每会话预算硬顶，到顶即关闭并标注。

红线
----
1. **预算不足 → 抛 `BudgetExceeded`**，且**不发起网络调用**（不静默跳过、不假装成功）。
2. **低 `confidence` 原样返回**，不擅自抬高——低置信观察本就该被上层丢弃，篡改会让
   幻觉证据混进评分。
3. 响应不符合 schema / 解析失败 → 抛 `Unavailable`，**绝不合成**一个观察结果。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence, runtime_checkable

from ..domain.errors import BudgetExceeded, Unavailable
from ..ports import Usage, VLMObservation
from .http import DEFAULT_TIMEOUT_S

#: 计量埋点名。
OPS_LATENCY_KEY = "vlm.observe"

#: 每轮帧数硬闸。来源：方案 §4.2「每轮上限 12 帧」。
DEFAULT_MAX_FRAMES = 12

#: 单次调用固定成本与单帧图像成本（USD）。来源：方案 §4.4 的量级示例；**须在
#: 签约后按实际单价替换**，此处仅用于调用前的预算预检（estimate），不是账单。
DEFAULT_PER_CALL_USD = 0.0008
DEFAULT_PER_FRAME_USD = 0.0004

#: 视觉观察输出的 JSON Schema（下发为 `response_format=json_schema`）。
DEFAULT_VLM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "title": "VisionObservation",
    "additionalProperties": False,
    "required": ["summary", "confidence", "labels", "evidence_frames"],
    "properties": {
        "summary": {
            "type": "string",
            "description": "对候选人表情/姿态/环境的简要描述（简体中文）",
        },
        "confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
            "description": "本次观察的把握；证据不足时取低值",
        },
        "labels": {
            "type": "array",
            "items": {"type": "string"},
            "description": "结构化标签，如 neutral / tense / camera / slouch",
        },
        "evidence_frames": {
            "type": "array",
            "items": {"type": "integer", "minimum": 0},
            "description": "支撑该判断的帧序号（0-based，便于事后复核）",
        },
    },
}


def estimate_vlm_cost_usd(
    n_frames: int,
    *,
    per_call_usd: float = DEFAULT_PER_CALL_USD,
    per_frame_usd: float = DEFAULT_PER_FRAME_USD,
) -> float:
    """调用前的成本预检估计（粗略，仅用于预算闸门，不做账）。"""
    return float(per_call_usd) + float(per_frame_usd) * max(0, int(n_frames))


@dataclass(frozen=True)
class VLMResponse:
    """低层客户端返回的原始结果：已解析的结构化 payload + 成本/延迟元信息。"""

    payload: Mapping[str, Any]
    cost_usd: float = 0.0
    latency_ms: int = 0
    raw: str = ""


@runtime_checkable
class VLMClient(Protocol):
    """低层视觉客户端（生产实现负责真实云端端点）。

    只做"发帧 + 拿结构化 payload"；预算、schema 校验、字段解析都在 `CloudVLM`。
    """

    async def observe(
        self,
        frames: Sequence[bytes],
        prompt: str,
        schema: Mapping[str, Any],
        timeout_s: float,
    ) -> VLMResponse: ...


class CloudVLM:
    """`VLMPort` 的云端实现：多帧打包 → 结构化观察，带预算与帧数护栏。

    Args:
        name: provider 名（进 `VLMObservation.provider` 与计量）。
        client: 低层视觉客户端（见 `VLMClient`）。
        budget: 预算端口，需实现 `can_afford(amount) -> bool` 与
            `spend(amount, tag) -> float`（如 `perception.budget.SessionBudget`）。
        meter: 可观测端口；None 则跳过计量。
        max_frames: 每轮帧数硬闸（默认 12，见 §4.2）。
        schema: JSON Schema；调用时未显式传入则用它。
        cost_estimator: `n_frames -> 预估 USD`；用于调用前预算预检。
        timeout_s: 单次请求超时。
        clock: 注入式时钟，保证测试确定性。
    """

    def __init__(
        self,
        name: str,
        client: VLMClient,
        budget: Any,
        *,
        meter: Optional[Any] = None,
        max_frames: int = DEFAULT_MAX_FRAMES,
        schema: Optional[Mapping[str, Any]] = None,
        cost_estimator: Callable[[int], float] = estimate_vlm_cost_usd,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.name = name
        self._client = client
        self.budget = budget
        self.meter = meter
        self.max_frames = int(max_frames)
        self.schema: Mapping[str, Any] = schema if schema is not None else DEFAULT_VLM_SCHEMA
        self.cost_estimator = cost_estimator
        self.timeout_s = float(timeout_s)
        self._clock = clock

    # ---------- VLMPort ----------

    async def observe(
        self, frames: list[bytes], prompt: str, *, schema: Optional[dict] = None
    ) -> VLMObservation:
        """观察多帧。**预算/帧数不通过时抛错且不发请求；解析失败抛 `Unavailable`。**"""
        if not frames:
            raise ValueError("frames 不能为空：没有可观察的帧")
        if len(frames) > self.max_frames:
            raise BudgetExceeded("vlm_frames", limit=self.max_frames, used=len(frames))

        estimate = float(self.cost_estimator(len(frames)))
        if not self.budget.can_afford(estimate):
            # 硬失败：不发请求、不降级为"跳过"，让"预算不足"必须被显式处理。
            raise self._budget_exceeded("vlm_usd", estimate)

        effective_schema = schema if schema is not None else self.schema
        start = self._clock()
        try:
            resp = await self._client.observe(
                list(frames), prompt, effective_schema, self.timeout_s
            )
        except Unavailable:
            raise
        except Exception as exc:
            self._incr("vlm.error")
            raise Unavailable(self.name, f"VLM 请求失败 {type(exc).__name__}: {exc}") from exc
        latency_ms = int(round((self._clock() - start) * 1000.0))

        summary, confidence, labels, evidence = _parse_observation(resp.payload, self.name)
        cost = float(resp.cost_usd)
        # 账务扣减走硬顶路径：真实成本超过上限同样抛 BudgetExceeded（不静默）。
        self.budget.spend(cost, "vlm")

        observation = VLMObservation(
            summary=summary,
            confidence=confidence,
            labels=labels,
            evidence_frames=evidence,
            provider=self.name,
            cost_usd=round(cost, 6),
            latency_ms=latency_ms,
        )
        self._report(latency_ms, cost)
        return observation

    # ---------- 内部 ----------

    def _budget_exceeded(self, kind: str, estimate: float) -> BudgetExceeded:
        cap = getattr(self.budget, "cap_usd", None)
        spent = getattr(self.budget, "spent", None)
        limit = float(cap) if isinstance(cap, (int, float)) else None
        used = float(spent) + estimate if isinstance(spent, (int, float)) else None
        return BudgetExceeded(kind, limit=limit, used=used)

    def _incr(self, name: str) -> None:
        if self.meter is None:
            return
        self.meter.incr(name)

    def _report(self, latency_ms: int, cost_usd: float) -> None:
        if self.meter is None:
            return
        self.meter.record_usage(
            Usage(
                provider=self.name,
                operation=OPS_LATENCY_KEY,
                latency_ms=latency_ms,
                cost_usd=cost_usd,
                cached=False,
            )
        )
        self.meter.observe_ms(OPS_LATENCY_KEY, float(latency_ms))


# ---------- 解析：任何一步不满足契约都抛 Unavailable（绝不合成观察） ----------


def _parse_observation(
    payload: Mapping[str, Any], provider: str
) -> tuple[str, float, tuple[str, ...], tuple[int, ...]]:
    if not isinstance(payload, Mapping):
        raise Unavailable(provider, f"VLM payload 不是对象: {type(payload).__name__}")

    summary = payload.get("summary")
    if not isinstance(summary, str):
        raise Unavailable(provider, "VLM 响应缺少字符串 summary")

    confidence = _confidence(payload.get("confidence"), provider)
    labels = _str_tuple(payload.get("labels"), "labels", provider)
    evidence = _int_tuple(payload.get("evidence_frames"), "evidence_frames", provider)
    return summary, confidence, labels, evidence


def _confidence(value: Any, provider: str) -> float:
    # bool 是 int 子类，必须显式排除，否则 `true` 会被当成 1.0。
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Unavailable(provider, f"VLM 响应 confidence 非数值: {value!r}")
    confidence = float(value)
    if not (0.0 <= confidence <= 1.0):
        raise Unavailable(provider, f"VLM 响应 confidence 越界 [0,1]: {confidence}")
    return confidence


def _str_tuple(value: Any, label: str, provider: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise Unavailable(provider, f"VLM 响应 {label} 非数组: {type(value).__name__}")
    out: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise Unavailable(provider, f"VLM 响应 {label} 元素非字符串: {item!r}")
        out.append(item)
    return tuple(out)


def _int_tuple(value: Any, label: str, provider: str) -> tuple[int, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise Unavailable(provider, f"VLM 响应 {label} 非数组: {type(value).__name__}")
    out: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise Unavailable(provider, f"VLM 响应 {label} 元素非整数: {item!r}")
        if item < 0:
            raise Unavailable(provider, f"VLM 响应 {label} 元素为负: {item}")
        out.append(item)
    return tuple(out)


__all__ = [
    "DEFAULT_MAX_FRAMES",
    "DEFAULT_PER_CALL_USD",
    "DEFAULT_PER_FRAME_USD",
    "DEFAULT_VLM_SCHEMA",
    "OPS_LATENCY_KEY",
    "CloudVLM",
    "VLMClient",
    "VLMResponse",
    "estimate_vlm_cost_usd",
]
