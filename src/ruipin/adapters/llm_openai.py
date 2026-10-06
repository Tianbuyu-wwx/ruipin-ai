"""OpenAI 兼容 `chat/completions` 评分适配器（OpenAI / DeepSeek / 通义通用）。

设计意图
--------
1. **绝不产出合成分数**（本项目第一纪律）。解析失败 / 缺维度 / 维度越界 /
   confidence 越界，一律视为"这一次调用失败"：可重试的重试一次，重试仍失败就抛
   `Unavailable`。**不存在** "解析不出来就给 60 分"、"缺维度就补均值" 这类兜底
   路径——遗留系统的头号缺陷就是崩溃即返回全 60 分，与真实 60 分不可区分。
2. **用 JSON Schema 强约束输出**（方案 §10：防格式漂移）。自由文本解析是评分
   链路最大的不确定性来源；`response_format: json_schema` 把格式责任交给供应商，
   适配器只做"校验"，不做"猜测"。
3. **密钥不落代码**。`api_key` 只由调用方注入或经 `api_key_from_env` 从环境读取，
   本模块不提供任何默认值。遗留 `configs/config.py` 把密钥明文写进
   `os.environ.get(...)` 的默认参数是 P0 事故，此处绝不复现。
4. **可观测**：每次调用如实上报 `Usage`（token / 成本 / 延迟）与 `observe_ms`；
   没有 meter 时静默跳过（可观测性是可选的，评分正确性不是）。

degraded 语义
-------------
只有"结果确实拿到了、但质量有折损"才置 `degraded=True`：输出被截断
（`finish_reason == "length"`）、或供应商没给 feedback。**失败就是失败**，
不允许把失败伪装成降级（那会让上层把一次不可用当成弱证据用）。
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any, Awaitable, Callable, Mapping, Optional

from ..domain.errors import Unavailable
from ..ports import DIMENSIONS, EvalRequest, EvalResult, Usage
from .http import DEFAULT_TIMEOUT_S, HttpResponse, HttpTransport, is_retryable_status

#: 计量埋点名。图/告警按这个 key 聚合，改名要同步改看板。
OPS_LATENCY_KEY = "llm.eval"

#: 六维在提示词里的口径说明。模型按这个打分，改文案等于改评分标准，需重跑标定。
DIM_DESCRIPTIONS: dict[str, str] = {
    "technical": "技术深度：概念是否准确，是否给出可验证的技术细节",
    "communication": "表达清晰：结构是否清楚，术语使用是否得当",
    "completeness": "完整度：是否覆盖问题的关键方面，是否有量化结果",
    "problem_solving": "问题解决：定位思路、权衡与取舍是否讲清楚",
    "teamwork": "团队协作：角色分工、沟通与冲突处理是否具体",
    "leadership": "领导力：推动过程、决策依据与复盘是否有主线索",
}

#: `finish_reason` 里意味着"这次输出不可用"的取值（不是降级，是失败）。
_BAD_FINISH_REASONS: frozenset[str] = frozenset({"content_filter"})

_SYSTEM_TEMPLATE = """你是资深技术面试官的评分引擎，只依据给定的回答内容打分，不得臆造事实。

评分维度（每维 0-100，数值越高越好）：
{dim_lines}

硬性要求：
1. 只输出一个 JSON 对象，不要任何解释文字、Markdown 代码块或前后缀。
2. `score` 是综合分（0-100）；`confidence` 是你对本次评分的把握（0-1）。
3. `feedback` 用简体中文，2-4 句，指出一个具体优点和一个可改进点。
4. 回答为空、离题或信息不足以评判时，给低分并把 `confidence` 降到 0.3 以下，
   **不要给中间值敷衍**——中间值与"评不出来"必须可区分。"""

_USER_TEMPLATE = """题目类型：{question_type}
题目：{question}
参考答案关键词：{keywords}
候选人回答：{answer}

请按六个维度打分，只输出 JSON。"""


def build_eval_schema() -> dict[str, Any]:
    """评分输出的 JSON Schema（给 `response_format` 用）。

    维度由 `DIMENSIONS` 推导而非手写，避免"核心层加了维度、提示词没跟上"的漂移。
    """
    properties: dict[str, Any] = {
        d: {
            "type": "number",
            "minimum": 0,
            "maximum": 100,
            "description": DIM_DESCRIPTIONS[d],
        }
        for d in DIMENSIONS
    }
    properties["score"] = {
        "type": "number",
        "minimum": 0,
        "maximum": 100,
        "description": "综合分（0-100）",
    }
    properties["feedback"] = {
        "type": "string",
        "description": "简体中文评语：一个具体优点 + 一个可改进点",
    }
    properties["confidence"] = {
        "type": "number",
        "minimum": 0,
        "maximum": 1,
        "description": "评分把握（0-1）；信息不足时取低值",
    }
    return {
        "type": "object",
        "title": "InterviewEvaluation",
        "additionalProperties": False,
        "required": [*DIMENSIONS, "score", "feedback", "confidence"],
        "properties": properties,
    }


#: 模块级常量：测试与文档都引用它，保证"发出去的 schema"与"文档里的 schema"一致。
EVAL_JSON_SCHEMA: dict[str, Any] = build_eval_schema()


def build_system_prompt() -> str:
    return _SYSTEM_TEMPLATE.format(
        dim_lines="\n".join(
            f"- {d}：{DIM_DESCRIPTIONS[d]}" for d in DIMENSIONS
        )
    )


SYSTEM_PROMPT: str = build_system_prompt()


def build_user_prompt(req: EvalRequest) -> str:
    """把 `EvalRequest` 的四个评分依据拼进用户消息（transcript 不参与评分）。"""
    keywords = "、".join(req.keywords) if req.keywords else "（未提供）"
    return _USER_TEMPLATE.format(
        question_type=req.question_type,
        question=req.question,
        keywords=keywords,
        answer=req.answer,
    )


def api_key_from_env(
    env_var: str, environ: Optional[Mapping[str, str]] = None
) -> Optional[str]:
    """从环境变量读密钥。**没有默认值**，缺失返回 None（由 `evaluate` 抛 Unavailable）。

    Args:
        env_var: 变量名，由调用方指定（本模块不猜名字，更不内置密钥）。
        environ: 变量表；默认 `os.environ`，测试可注入 dict。
    """
    env = os.environ if environ is None else environ
    raw = env.get(env_var)
    if raw is None or not raw.strip():
        return None
    return raw.strip()


class _BadResponse(Exception):
    """本次响应不可用（可重试）。消息即失败原因，会进 `Unavailable.reason`。"""


class OpenAICompatEvaluator:
    """`Evaluator` 端口的云端实现：任何 OpenAI 兼容端点都能接。

    Args:
        name: provider 名（进 `EvalResult.provider` 与计量）。
        model / base_url: 模型名与端点前缀（会拼上 `/chat/completions`）。
        api_key: 密钥；**None 时 `evaluate` 直接抛 Unavailable 且不发请求**。
        transport: HTTP 传输（见 `adapters.http`）。
        meter: 可观测端口；None 则完全跳过计量（不报错）。
        pricing: `{provider: {"prompt_per_1k": x, "completion_per_1k": y}}`，
            未命中 provider 时回退 `"default"`；为 None 则成本记 0。
        max_retries: 额外尝试次数（总次数 = `max_retries + 1`）。
        timeout_s: 单次请求超时。
        clock / sleeper: 注入式时钟与退避等待，保证测试确定性（不真睡）。
        backoff_base_s / backoff_factor: 第 n 次退避 = base * factor ** n。
        strict_schema: 是否下发 `strict: true`（OpenAI 严格模式；部分国产端点
            不支持，可关）。
    """

    def __init__(
        self,
        name: str,
        model: str,
        base_url: str,
        api_key: Optional[str],
        transport: HttpTransport,
        *,
        meter: Optional[Any] = None,
        pricing: Optional[Mapping[str, Mapping[str, float]]] = None,
        max_retries: int = 1,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Optional[Callable[[float], Awaitable[None]]] = None,
        backoff_base_s: float = 0.5,
        backoff_factor: float = 2.0,
        strict_schema: bool = True,
    ) -> None:
        self.name = name
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.transport = transport
        self.meter = meter
        self.pricing = pricing
        self.max_retries = max(0, int(max_retries))
        self.timeout_s = float(timeout_s)
        self._clock = clock
        self._sleeper: Callable[[float], Awaitable[None]] = sleeper or asyncio.sleep
        self.backoff_base_s = float(backoff_base_s)
        self.backoff_factor = float(backoff_factor)
        self.strict_schema = bool(strict_schema)

    # ---------- Evaluator ----------

    async def evaluate(self, req: EvalRequest) -> EvalResult:
        """评分。**成功才返回 `EvalResult`，失败一律抛 `Unavailable`。**"""
        if not (self.api_key or "").strip():
            # 不发请求、不重试、不兜底：配置缺失是运维问题，不是评分问题。
            # 空白串（" " / "\t"）也算未配置——它通常来自 CI 变量没赋值，
            # 若放行就会带着一个假密钥发出请求，然后收到 401 并浪费一次重试。
            raise Unavailable(self.name, "未配置 API key（禁止硬编码默认值）")

        payload = self._payload(req)
        url = f"{self.base_url}/chat/completions"
        headers = self._headers()
        last_reason = ""

        for attempt in range(self.max_retries + 1):
            t0 = self._clock()
            net_error: Optional[str] = None
            resp: Optional[HttpResponse] = None
            try:
                resp = await self.transport.post_json(
                    url, headers, payload, self.timeout_s
                )
            except Unavailable:
                raise  # 依赖缺失等结构性故障：重试无意义，立即上报
            except Exception as exc:  # 连接重置 / DNS / 超时：可重试
                net_error = f"网络异常 {type(exc).__name__}: {exc}"
            latency_ms = int(round((self._clock() - t0) * 1000.0))

            if net_error is not None:
                last_reason = net_error
            elif not resp.ok and not is_retryable_status(resp.status):
                raise Unavailable(
                    self.name, f"HTTP {resp.status}（不可重试）: {_brief(resp)}"
                )
            elif not resp.ok:
                last_reason = f"HTTP {resp.status}: {_brief(resp)}"
            else:
                try:
                    data = _json_of(resp)
                    content, finish = _content_and_finish(data)
                    dims, score, feedback, confidence = _scores_of(content)
                    degraded, degrade_reason = _degradation_of(finish, feedback)
                except _BadResponse as bad:
                    # 格式漂移 / 越界 / 截断到无法解析 = 这次调用失败，不是降级。
                    last_reason = str(bad)
                else:
                    usage = _usage_of(
                        data,
                        provider=self.name,
                        latency_ms=latency_ms,
                        pricing=self.pricing,
                    )
                    self._report(usage, latency_ms)
                    return EvalResult(
                        dims=dims,
                        score=score,
                        feedback=feedback,
                        provider=self.name,
                        confidence=confidence,
                        latency_ms=latency_ms,
                        cost_usd=usage.cost_usd,
                        degraded=degraded,
                        degrade_reason=degrade_reason,
                    )

            if attempt < self.max_retries:
                await self._sleeper(
                    self.backoff_base_s * (self.backoff_factor**attempt)
                )

        raise Unavailable(
            self.name,
            f"{self.max_retries + 1} 次尝试均失败（最后一次原因: {last_reason}）",
        )

    # ---------- 内部 ----------

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _payload(self, req: EvalRequest) -> dict[str, Any]:
        json_schema: dict[str, Any] = {
            "name": "interview_evaluation",
            "schema": EVAL_JSON_SCHEMA,
        }
        if self.strict_schema:
            json_schema["strict"] = True
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(req)},
            ],
            # 评分要可复现：温度固定为 0，不随调用方偏好漂移。
            "temperature": 0.0,
            "response_format": {"type": "json_schema", "json_schema": json_schema},
        }

    def _report(self, usage: Usage, latency_ms: int) -> None:
        """上报计量。没有 meter 时静默跳过——可观测性不得成为评分的前置条件。"""
        if self.meter is None:
            return
        self.meter.record_usage(usage)
        self.meter.observe_ms(OPS_LATENCY_KEY, float(latency_ms))


# ---------- 响应解析：任何一步不满足契约都抛 _BadResponse（可重试） ----------


def _brief(resp: HttpResponse, limit: int = 200) -> str:
    """截断响应体用于错误信息——全量回显会把整段回答灌进日志。"""
    return resp.text[:limit]


def _json_of(resp: HttpResponse) -> dict[str, Any]:
    try:
        data = resp.json()
    except ValueError as exc:
        raise _BadResponse(f"响应体不是合法 JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise _BadResponse(f"响应 JSON 顶层不是对象: {type(data).__name__}")
    return data


def _content_and_finish(data: Mapping[str, Any]) -> tuple[str, str]:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise _BadResponse("响应缺少 choices")
    first = choices[0]
    if not isinstance(first, dict):
        raise _BadResponse("choices[0] 不是对象")

    message = first.get("message")
    message = message if isinstance(message, dict) else {}
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise _BadResponse("响应缺少 message.content")
    finish = first.get("finish_reason")
    return content, finish if isinstance(finish, str) else ""


def _scores_of(content: str) -> tuple[dict[str, float], float, str, float]:
    """解析模型输出。**缺字段/非数值/越界一律失败，不做任何填补。**"""
    try:
        obj = json.loads(content)
    except ValueError as exc:
        raise _BadResponse(f"模型输出不是合法 JSON: {exc}") from exc
    if not isinstance(obj, dict):
        raise _BadResponse(f"模型输出顶层不是对象: {type(obj).__name__}")

    dims: dict[str, float] = {}
    for dim in DIMENSIONS:
        if dim not in obj:
            raise _BadResponse(f"缺少维度 {dim}")
        dims[dim] = _number(obj[dim], f"维度 {dim}", 0.0, 100.0)
    score = _number(obj.get("score"), "score", 0.0, 100.0)
    confidence = _number(obj.get("confidence"), "confidence", 0.0, 1.0)

    feedback = obj.get("feedback")
    if feedback is None:
        raise _BadResponse("缺少字段 feedback")
    if not isinstance(feedback, str):
        raise _BadResponse(f"feedback 不是字符串: {type(feedback).__name__}")
    return dims, score, feedback, confidence


def _number(value: Any, label: str, low: float, high: float) -> float:
    if value is None:
        raise _BadResponse(f"缺少字段 {label}")
    # bool 是 int 的子类，必须显式排除，否则 `true` 会被当成 1 分。
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _BadResponse(f"{label} 非数值: {value!r}")
    num = float(value)
    if not (low <= num <= high):
        raise _BadResponse(f"{label} 越界 [{low},{high}]: {num}")
    return num


def _degradation_of(finish: str, feedback: str) -> tuple[bool, Optional[str]]:
    """判定降级。**只判定"结果拿到了但折损"，不把失败粉饰成降级。**"""
    if finish in _BAD_FINISH_REASONS:
        raise _BadResponse(f"输出不可用（finish_reason={finish}）")
    reasons: list[str] = []
    if finish == "length":
        reasons.append("供应商输出被截断（finish_reason=length），内容可能不完整")
    if not feedback.strip():
        reasons.append("供应商未给出 feedback")
    if not reasons:
        return False, None
    return True, "；".join(reasons)


def _rates_for(
    pricing: Optional[Mapping[str, Mapping[str, float]]], provider: str
) -> Optional[Mapping[str, float]]:
    if not pricing:
        return None
    rates = pricing.get(provider)
    if rates is None:
        rates = pricing.get("default")
    return rates


def _usage_of(
    data: Mapping[str, Any],
    *,
    provider: str,
    latency_ms: int,
    pricing: Optional[Mapping[str, Mapping[str, float]]],
) -> Usage:
    """从响应里读 token 并按定价算成本；读不到就如实记 0（不估算、不编造）。"""
    raw = data.get("usage")
    raw = raw if isinstance(raw, dict) else {}
    prompt_tokens = _nonneg_int(raw.get("prompt_tokens"))
    completion_tokens = _nonneg_int(raw.get("completion_tokens"))

    rates = _rates_for(pricing, provider)
    cost = 0.0
    if rates is not None:
        cost = round(
            prompt_tokens / 1000.0 * float(rates.get("prompt_per_1k", 0.0))
            + completion_tokens / 1000.0 * float(rates.get("completion_per_1k", 0.0)),
            6,
        )
    return Usage(
        provider=provider,
        operation=OPS_LATENCY_KEY,
        latency_ms=latency_ms,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cost_usd=cost,
        # 供应商未上报缓存命中时记 False：宁可少报收益，不可虚报。
        cached=False,
    )


def _nonneg_int(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    return max(0, int(value))


__all__ = [
    "DIM_DESCRIPTIONS",
    "EVAL_JSON_SCHEMA",
    "OPS_LATENCY_KEY",
    "OpenAICompatEvaluator",
    "SYSTEM_PROMPT",
    "api_key_from_env",
    "build_eval_schema",
    "build_system_prompt",
    "build_user_prompt",
]
