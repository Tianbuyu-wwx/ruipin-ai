"""多供应商评分路由（`adapters/llm_router.py`）的测试。

这个文件补的是一个**真空**：`llm_router.py` 是"两家供应商可切换"这条方案承诺的
落点，但测试一度为零——零覆盖时"能切换"只是纸面说法。这里把三类红线钉住：

1. **全失败必须抛 Unavailable**，不许拼默认分（本项目第一纪律）。
2. **只把 `Unavailable` 当切换信号**；其它异常是代码缺陷，必须冒泡而不是伪装成
   "供应商不可用"被静默换家。
3. **切换必须可观测**：结果打 `degraded` 标注、`fallback_reason` 保留主供应商
   的原始失败原因，且已有的 `degrade_reason` 不被覆盖。
"""

from __future__ import annotations

import asyncio

import pytest

from ruipin.adapters.fakes import FakeEvaluator
from ruipin.adapters.llm_router import USED_COUNTER_PREFIX, EvaluatorRouter
from ruipin.domain.errors import Unavailable
from ruipin.observability.metrics import InMemoryMeter
from ruipin.ports import DIMENSIONS, EvalRequest

REQ = EvalRequest(
    question="请介绍你主导过的一个项目。",
    answer="我主导了召回重构，用向量检索替换倒排，QPS 提升 3 倍。",
    question_type="technical",
    transcript="我主导了召回重构，用向量检索替换倒排，QPS 提升 3 倍。",
    has_audio=True,
    keywords=("召回", "向量检索"),
)


def _run(coro):
    return asyncio.run(coro)


class _Boom(Exception):
    """非 `Unavailable` 的异常——代表代码缺陷，不该被路由吞掉。"""


class _RaisingEvaluator:
    """抛指定异常的假评估器（`FakeEvaluator` 只会抛 Unavailable）。"""

    def __init__(self, name: str, exc: BaseException) -> None:
        self.name = name
        self._exc = exc
        self.calls = 0

    async def evaluate(self, req: EvalRequest):
        self.calls += 1
        raise self._exc


class _CountingEvaluator(FakeEvaluator):
    """在 `FakeEvaluator` 之上记录调用次数，用于断言"短路后不再调用"。"""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.n_calls = 0

    async def evaluate(self, req: EvalRequest):
        self.n_calls += 1
        return await super().evaluate(req)


# ---------- 构造校验与只读视图 ----------


def test_router_requires_at_least_one_evaluator():
    """防回归：空列表会让 evaluate 直接抛 Unavailable，掩盖配置错误——构造期就该炸。"""
    with pytest.raises(ValueError):
        EvaluatorRouter([])


def test_providers_preserve_priority_order():
    """防回归：顺序即优先级，被排序或去重都会改变"谁是主供应商"。"""
    r = EvaluatorRouter(
        [FakeEvaluator(provider="primary"), FakeEvaluator(provider="backup")]
    )
    assert r.providers == ("primary", "backup")


# ---------- 正常路径 ----------


def test_primary_success_returns_primary_result_and_records_counter():
    """防回归：主供应商成功时不应触发任何降级标注，且计量要记到主供应商名下。"""
    meter = InMemoryMeter()
    primary = _CountingEvaluator(provider="deepseek")
    backup = _CountingEvaluator(provider="qwen")
    r = EvaluatorRouter([primary, backup], meter=meter)

    res = _run(r.evaluate(REQ))

    assert res.provider == "deepseek"
    assert res.degraded is False
    assert res.degrade_reason is None
    assert r.fallback_reason is None
    assert meter.counter(f"{USED_COUNTER_PREFIX}deepseek") == 1
    assert meter.counter(f"{USED_COUNTER_PREFIX}qwen") == 0


def test_short_circuits_after_first_success():
    """防回归：主成功就不得再打备用——多打一家既是成本也是延迟。"""
    primary = _CountingEvaluator(provider="a")
    backup = _CountingEvaluator(provider="b")
    r = EvaluatorRouter([primary, backup])

    _run(r.evaluate(REQ))

    assert primary.n_calls == 1
    assert backup.n_calls == 0


def test_router_without_meter_does_not_fail():
    """防回归：meter 是可选依赖，缺了不能把评分链路带崩。"""
    r = EvaluatorRouter([FakeEvaluator(provider="a")])
    assert _run(r.evaluate(REQ)).provider == "a"


# ---------- 切换路径 ----------


def test_fallback_marks_degraded_and_keeps_primary_failure_reason():
    """防回归：走备用链路必须(a)标注 degraded，(b)把主供应商的失败原因带到结果上。

    这两点都直接影响聚合器：不标注就等于把备用链路的分数当主链路的用。
    """
    primary = FakeEvaluator(fail=True, provider="deepseek")
    backup = FakeEvaluator(provider="qwen")
    r = EvaluatorRouter([primary, backup])

    res = _run(r.evaluate(REQ))

    assert res.provider == "qwen"
    assert res.degraded is True
    assert res.degrade_reason is not None
    assert "deepseek" in res.degrade_reason
    assert "已切换备用供应商" in res.degrade_reason
    assert r.fallback_reason is not None
    assert "deepseek" in r.fallback_reason


def test_fallback_merges_existing_degrade_reason_instead_of_overwriting():
    """防回归：备用评估器自己已声明降级原因时，不能被路由的说明覆盖掉。

    覆盖会造成"为什么降级"的信息丢失：读者只看到"换了供应商"，
    看不到"结果是截断的"这类更重要的原因。
    """
    primary = FakeEvaluator(fail=True, provider="deepseek")
    backup = FakeEvaluator(
        provider="qwen", degraded=True, degrade_reason="响应被 max_tokens 截断"
    )
    r = EvaluatorRouter([primary, backup])

    res = _run(r.evaluate(REQ))

    assert res.degraded is True
    assert "响应被 max_tokens 截断" in res.degrade_reason
    assert "已切换备用供应商" in res.degrade_reason


def test_mark_degraded_disabled_still_records_fallback_reason():
    """防回归：关闭标注开关只影响结果字段，不应让"发生过切换"这件事变得不可见。"""
    primary = FakeEvaluator(fail=True, provider="deepseek")
    backup = FakeEvaluator(provider="qwen")
    r = EvaluatorRouter([primary, backup], mark_degraded_on_fallback=False)

    res = _run(r.evaluate(REQ))

    assert res.provider == "qwen"
    assert res.degraded is False
    assert r.fallback_reason is not None


def test_three_providers_accumulate_all_failures():
    """防回归：链路上每一家的失败原因都要留下，便于定位"是不是某一类错误全灭"。"""
    evs = [FakeEvaluator(fail=True, provider=p) for p in ("a", "b")]
    evs.append(FakeEvaluator(provider="c"))
    meter = InMemoryMeter()
    r = EvaluatorRouter(evs, meter=meter)

    res = _run(r.evaluate(REQ))

    assert res.provider == "c"
    assert "a" in r.fallback_reason and "b" in r.fallback_reason
    assert meter.counter(f"{USED_COUNTER_PREFIX}c") == 1
    assert meter.counter(f"{USED_COUNTER_PREFIX}a") == 0


def test_fallback_reason_resets_between_calls():
    """防回归：上一次的切换原因不能粘到下一次调用上（会让告警看起来永远在降级）。"""
    primary = FakeEvaluator(fail=True, provider="deepseek")
    backup = FakeEvaluator(provider="qwen")
    r = EvaluatorRouter([primary, backup])

    _run(r.evaluate(REQ))
    assert r.fallback_reason is not None

    primary.fail = False
    res = _run(r.evaluate(REQ))

    assert res.provider == "deepseek"
    assert r.fallback_reason is None


# ---------- 失败红线 ----------


def test_all_providers_failing_raises_unavailable_and_never_synthesizes():
    """★ 红线：全部供应商失败 → 抛 Unavailable，**绝不**返回兜底分数。"""
    r = EvaluatorRouter(
        [FakeEvaluator(fail=True, provider="deepseek"), FakeEvaluator(fail=True, provider="qwen")]
    )

    with pytest.raises(Unavailable) as exc:
        _run(r.evaluate(REQ))

    assert "deepseek" in exc.value.reason
    assert "qwen" in exc.value.reason
    assert exc.value.provider == "router"


def test_non_unavailable_exception_propagates_and_does_not_switch():
    """★ 红线：非 Unavailable 的异常是代码缺陷，必须冒泡，不许被伪装成"换一家"。

    如果路由吞掉 ValueError 换备用供应商，缺陷就会被掩盖成"供应商抖动"，
    真正的问题会一直潜伏——这正是遗留系统 `except Exception` 泛滥的翻版。
    """
    boom = _RaisingEvaluator("broken", _Boom("维度键名写错了"))
    backup = _CountingEvaluator(provider="qwen")
    r = EvaluatorRouter([boom, backup])

    with pytest.raises(_Boom):
        _run(r.evaluate(REQ))

    assert boom.calls == 1
    assert backup.n_calls == 0  # 没有偷偷切走


def test_router_satisfies_evaluator_shape():
    """防回归：路由自身要能被当作 `Evaluator` 再嵌一层（组合而非特例）。"""
    r = EvaluatorRouter([FakeEvaluator(provider="a")])
    assert isinstance(r.name, str) and r.name
    assert callable(r.evaluate)
    outer = EvaluatorRouter([r, FakeEvaluator(provider="z")])
    assert _run(outer.evaluate(REQ)).provider == "a"


def test_all_six_dimensions_survive_the_router():
    """防回归：路由不得削掉任何维度（曾见过"转发时只挑几个字段"的写法）。"""
    r = EvaluatorRouter([FakeEvaluator(fail=True, provider="x"), FakeEvaluator(provider="y")])
    res = _run(r.evaluate(REQ))
    assert set(res.dims) == set(DIMENSIONS)
