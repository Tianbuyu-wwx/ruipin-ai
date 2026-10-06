"""领域层错误类型的全覆盖测试（errors.py）。

按**源码实际定义**写断言（先读后写）：
- Unavailable(provider, reason)  -> "{provider} unavailable: {reason}"
- Degraded(reason, level=1)      -> "degraded(L{level}): {reason}"，level 有默认值 → 两分支
- InvalidTransition(state, event)-> "invalid transition: {state.value} --{event.value}--> ?"
- BudgetExceeded(kind, limit=None, used=None)
                                 -> "budget exceeded: {kind}" + 可选的"（已用 x，上限 y）"

可序列化是硬要求：异常要跨进程边界（Redis Streams 解耦的 worker 池、任务队列、
日志采集）传播，因此每个子类都实现了 `__reduce__`。本文件对**每一个**异常类断言
`pickle` 往返后 type / str / 关键属性全部不变——这是跨进程传播的前提，务必测。

核心纪律：适配器失败必须抛 Unavailable，禁止返回合成/default 分数。
"""

from __future__ import annotations

import pickle

import pytest

from ruipin.domain.errors import (
    BudgetExceeded,
    Degraded,
    InvalidTransition,
    RuipinError,
    Unavailable,
)
from ruipin.domain.states import Event, State

pytestmark = pytest.mark.unit

ALL_ERRORS = (RuipinError, Unavailable, Degraded, InvalidTransition, BudgetExceeded)

# 每个异常类各构造一个"有代表性"的实例，供跨类穷举使用
SAMPLES = {
    RuipinError: lambda: RuipinError("base"),
    Unavailable: lambda: Unavailable("asr", "down"),
    Degraded: lambda: Degraded("down", 2),
    InvalidTransition: lambda: InvalidTransition(State.BUFFER, Event.EVAL_DONE),
    BudgetExceeded: lambda: BudgetExceeded("vlm", 0.5, 0.7),
}


# ==========================================================================
# 1. 构造方式：位置参数 / 关键字参数
# ==========================================================================


def test_unavailable_positional_and_keyword():
    """防回归：Unavailable 构造签名变更导致适配器调用点全部 TypeError。"""
    a = Unavailable("asr", "timeout")
    b = Unavailable(provider="asr", reason="timeout")
    assert a.provider == b.provider == "asr"
    assert a.reason == b.reason == "timeout"
    assert str(a) == str(b)


def test_degraded_positional_and_keyword():
    """防回归：Degraded 的 level 默认值（分支 1）与显式传值（分支 2）行为不一致。"""
    defaulted = Degraded("asr 慢")
    assert defaulted.reason == "asr 慢"
    assert defaulted.level == 1

    positional = Degraded("asr 慢", 2)
    assert positional.level == 2

    keyword = Degraded(reason="asr 慢", level=3)
    assert keyword.reason == "asr 慢"
    assert keyword.level == 3
    assert isinstance(keyword.level, int)


def test_invalid_transition_positional_and_keyword():
    """防回归：InvalidTransition 丢掉 state/event 上下文（定位不到出错的迁移）。"""
    a = InvalidTransition(State.ASKING, Event.ABORT)
    b = InvalidTransition(state=State.ASKING, event=Event.ABORT)
    assert a.state is State.ASKING
    assert a.event is Event.ABORT
    assert b.state is a.state
    assert b.event is a.event


def test_invalid_transition_accepts_plain_values():
    """防回归：state/event 只接受枚举（错误信息里塞自定义对象时崩）。
    `_val()` 对非枚举走 getattr(x, "value", x) 的兜底分支，这里专门覆盖它。"""
    err = InvalidTransition("idle", "create")
    assert err.state == "idle"
    assert err.event == "create"
    assert "idle" in str(err)
    assert "create" in str(err)
    assert str(err) == "invalid transition: idle --create--> ?"


def test_ruipin_error_base_positional():
    """防回归：基类被改成必须带参数（现有 raise RuipinError() 全部崩）。"""
    assert str(RuipinError("boom")) == "boom"
    assert str(RuipinError()) == ""
    assert RuipinError("a", "b").args == ("a", "b")


# ==========================================================================
# 2. BudgetExceeded：三种构造形态
# ==========================================================================


def test_budget_exceeded_kind_only():
    """防回归：只给 kind 时 limit/used 应为 None，且消息不带括号部分。"""
    err = BudgetExceeded("vlm")
    assert err.kind == "vlm"
    assert err.limit is None
    assert err.used is None
    assert str(err) == "budget exceeded: vlm"
    assert "（" not in str(err)


def test_budget_exceeded_kind_limit_used():
    """防回归：三个参数齐备时，消息必须带上"已用/上限"的具体数值 ——
    告警要能直接看出超了多少，不能让人回代码里推。"""
    err = BudgetExceeded("vlm", 0.5, 0.7)
    assert err.kind == "vlm"
    assert err.limit == 0.5
    assert err.used == 0.7
    text = str(err)
    assert text == "budget exceeded: vlm（已用 0.7，上限 0.5）"
    assert "0.7" in text
    assert "0.5" in text
    assert "vlm" in text


def test_budget_exceeded_kind_limit_only():
    """防回归：只给 limit 时走 elif 分支，消息形如"（上限 0.5）"。"""
    err = BudgetExceeded("llm_usd", 100.0)
    assert err.kind == "llm_usd"
    assert err.limit == 100.0
    assert err.used is None
    assert str(err) == "budget exceeded: llm_usd（上限 100.0）"
    assert "100.0" in str(err)
    assert "已用" not in str(err)


def test_budget_exceeded_used_without_limit_reports_usage():
    """防回归：只给 used 不给 limit 时，消息里**仍要显示已用量**。

    早期版本的分支只判断 `limit is not None`，于是 `BudgetExceeded("tts_chars",
    None, 12000)` 的消息里一个数字都没有，告警等于让人回代码里猜超了多少。
    已补上 `elif used is not None` 分支（见 errors.py）。"""
    err = BudgetExceeded("tts_chars", None, 12000)
    assert err.kind == "tts_chars"
    assert err.limit is None
    assert err.used == 12000
    assert str(err) == "budget exceeded: tts_chars（已用 12000）"


def test_budget_exceeded_positional_and_keyword():
    """防回归：BudgetExceeded 有了自定义 __init__ 后，关键字参数应当可用
    （早期沿用 Exception 的 *args 时不支持关键字参数）。"""
    a = BudgetExceeded("vlm", 0.5, 0.7)
    b = BudgetExceeded(kind="vlm", limit=0.5, used=0.7)
    assert (a.kind, a.limit, a.used) == (b.kind, b.limit, b.used)
    assert str(a) == str(b)
    with pytest.raises(TypeError):
        BudgetExceeded()  # kind 是必填
    with pytest.raises(TypeError):
        BudgetExceeded("vlm", 0.5, 0.7, "多余参数")


# ==========================================================================
# 3. 继承链
# ==========================================================================


@pytest.mark.parametrize("cls", ALL_ERRORS, ids=lambda c: c.__name__)
def test_every_error_is_a_ruipin_error(cls):
    """防回归：某个异常忘记继承 RuipinError，统一兜底捕获（except RuipinError）漏掉它。"""
    assert issubclass(cls, RuipinError)
    assert issubclass(cls, Exception)


def test_inheritance_hierarchy_is_flat():
    """防回归：异常层级被无意加深/改乱（except 顺序敏感）。"""
    for cls in (Unavailable, Degraded, InvalidTransition, BudgetExceeded):
        assert cls.__bases__ == (RuipinError,)
    assert RuipinError.__bases__ == (Exception,)


def test_catch_all_via_ruipin_error():
    """防回归：无法用 except RuipinError 一把兜住项目内异常。"""
    instances = [
        Unavailable("llm", "429"),
        Degraded("慢", 2),
        InvalidTransition(State.IDLE, Event.ABORT),
        BudgetExceeded("超预算"),
    ]
    for exc in instances:
        with pytest.raises(RuipinError):
            raise exc
        with pytest.raises(Exception):
            raise exc


def test_builtin_exceptions_are_not_ruipin_errors():
    """防回归：except RuipinError 误捕了 ValueError/KeyError 等外部异常。"""
    assert not issubclass(ValueError, RuipinError)
    assert not issubclass(KeyError, RuipinError)
    assert not issubclass(TypeError, RuipinError)


# ==========================================================================
# 4. str() 携带关键上下文（断言具体子串，不只断言非空）
# ==========================================================================


def test_unavailable_message_contains_provider_and_reason():
    """防回归：错误信息丢掉 provider/reason（线上无法判断是哪个适配器挂了）。"""
    err = Unavailable("whisper", "连接超时 5s")
    text = str(err)
    assert text == "whisper unavailable: 连接超时 5s"
    assert "whisper" in text
    assert "unavailable" in text
    assert "连接超时 5s" in text
    assert text.strip() != ""


def test_degraded_message_contains_level_and_reason():
    """防回归：降级信息丢掉 level（无法区分轻度/重度降级）。"""
    err = Degraded("评分降级为关键词匹配", 2)
    assert str(err) == "degraded(L2): 评分降级为关键词匹配"
    assert "L2" in str(err)
    assert "评分降级为关键词匹配" in str(err)
    assert str(Degraded("x")) == "degraded(L1): x"
    assert "L1" in str(Degraded("x"))


def test_invalid_transition_message_uses_enum_values():
    """防回归：消息用 `str(Enum)` 拼装 —— 其输出随 Python 版本变化
    （State.BUFFER 可能印成 `State.BUFFER` 也可能印成 `buffer`），而这条消息会进
    日志检索与断言，必须稳定。源码改用 `.value`，这里按 `.value` 拼出期望串比对。"""
    err = InvalidTransition(State.BUFFER, Event.EVAL_DONE)
    expected = (
        f"invalid transition: {State.BUFFER.value}"
        f" --{Event.EVAL_DONE.value}--> ?"
    )
    assert str(err) == expected
    assert str(err) == "invalid transition: buffer --eval_done--> ?"
    assert "invalid transition" in str(err)
    assert "?" in str(err)


def test_invalid_transition_message_for_every_state_event_pair():
    """防回归：逐个组合校验消息格式（避免只有 BUFFER/EVAL_DONE 这一对被验证）。"""
    for state in State:
        for event in Event:
            err = InvalidTransition(state, event)
            assert str(err) == f"invalid transition: {state.value} --{event.value}--> ?"


@pytest.mark.parametrize("cls", ALL_ERRORS, ids=lambda c: c.__name__)
def test_str_is_never_empty_for_constructed_errors(cls):
    """防回归：某个异常 str(e) 为空串（日志里只剩一行类名，等于没有信息）。"""
    err = SAMPLES[cls]() if cls in SAMPLES else cls("base")
    assert isinstance(str(err), str)
    assert str(err).strip() != ""


@pytest.mark.parametrize("cls", ALL_ERRORS, ids=lambda c: c.__name__)
def test_repr_contains_class_name(cls):
    """防回归：repr 不含类名导致日志/堆栈难读。"""
    err = SAMPLES[cls]() if cls in SAMPLES else cls("base")
    assert type(err).__name__ in repr(err)


# ==========================================================================
# 5. ★ 可序列化：pickle 往返（跨进程传播的前提）
# ==========================================================================


@pytest.mark.parametrize("cls", ALL_ERRORS, ids=lambda c: c.__name__)
def test_pickle_roundtrip_preserves_type_and_message(cls):
    """★ 防回归：异常跨进程/队列传递后类型或消息丢失。
    子类 __init__ 的参数列表与 Exception.args 不同，基类 args 无法直接重建子类，
    所以每个子类都显式实现了 __reduce__；本用例就是这条保证的守卫。"""
    err = SAMPLES[cls]() if cls in SAMPLES else cls("base")
    restored = pickle.loads(pickle.dumps(err))
    assert type(restored) is type(err)
    assert str(restored) == str(err)
    assert isinstance(restored, RuipinError)


def test_pickle_roundtrip_preserves_unavailable_fields():
    """★ 防回归：跨进程后 provider/reason 丢失（无法判断是哪个适配器挂了）。"""
    err = Unavailable("asr", "down")
    restored = pickle.loads(pickle.dumps(err))
    assert type(restored) is Unavailable
    assert restored.provider == "asr"
    assert restored.reason == "down"
    assert str(restored) == "asr unavailable: down"


def test_pickle_roundtrip_preserves_degraded_fields():
    """★ 防回归：跨进程后 reason/level 丢失（无法区分降级等级）。"""
    for err in (Degraded("down"), Degraded("down", 3)):
        restored = pickle.loads(pickle.dumps(err))
        assert type(restored) is Degraded
        assert restored.reason == err.reason
        assert restored.level == err.level
        assert str(restored) == str(err)


def test_pickle_roundtrip_preserves_invalid_transition_fields():
    """★ 防回归：跨进程后 state/event 丢失（定位不到出错的迁移）。
    枚举身份也要保持（用 is 比对），否则下游 `err.state is State.BUFFER` 会失效。"""
    err = InvalidTransition(State.BUFFER_LISTENING, Event.EVAL_DONE)
    restored = pickle.loads(pickle.dumps(err))
    assert type(restored) is InvalidTransition
    assert restored.state is State.BUFFER_LISTENING
    assert restored.event is Event.EVAL_DONE
    assert str(restored) == "invalid transition: buffer_listening --eval_done--> ?"


def test_pickle_roundtrip_preserves_budget_exceeded_fields():
    """★ 防回归：跨进程后 kind/limit/used 丢失（告警看不出超了多少）。"""
    for err in (
        BudgetExceeded("vlm"),
        BudgetExceeded("vlm", 0.5),
        BudgetExceeded("vlm", 0.5, 0.7),
    ):
        restored = pickle.loads(pickle.dumps(err))
        assert type(restored) is BudgetExceeded
        assert restored.kind == err.kind
        assert restored.limit == err.limit
        assert restored.used == err.used
        assert str(restored) == str(err)


def test_pickle_roundtrip_survives_when_raised_and_caught():
    """★ 防回归：序列化后重新 raise，仍能被原类型与 RuipinError 同时捕获
    （worker 侧抛、主进程侧 except 的真实链路）。"""
    err = BudgetExceeded("vlm", 0.5, 0.7)
    restored = pickle.loads(pickle.dumps(err))
    with pytest.raises(BudgetExceeded) as excinfo:
        raise restored
    assert excinfo.value.kind == "vlm"
    assert excinfo.value.limit == 0.5
    assert excinfo.value.used == 0.7


def test_pickle_roundtrip_drops_cause_chain():
    """★ 如实记录当前行为（已上报，未修）：跨进程后 `__cause__` 会丢失。

    原因是子类 `__reduce__` 只返回 `(cls, args)` 两元组，不带异常链；
    BaseException 原生的 `__reduce__` 会额外带上 `__cause__`，
    这里被子类覆盖掉了。影响：worker 侧 `raise Degraded(...) from Unavailable(...)`
    传到主进程后只剩最外层，根因追溯断链。

    若将来修复（在 `__reduce__` 里补上 `__cause__`，或返回三元组带 state），
    本用例会失败，请按新行为改写。
    """
    try:
        try:
            raise Unavailable("llm", "429")
        except Unavailable as exc:
            raise Degraded("回退到关键词评分", 2) from exc
    except Degraded as outer:
        restored = pickle.loads(pickle.dumps(outer))
    assert isinstance(restored, Degraded)
    assert restored.__cause__ is None, "若此行失败，说明 __cause__ 已能跨进程保留"
    # 外层自身的信息仍然完整（这才是最关键的）
    assert restored.reason == "回退到关键词评分"
    assert restored.level == 2
    assert str(restored) == "degraded(L2): 回退到关键词评分"


# ==========================================================================
# 6. raise ... from 链式抛出与属性可读取
# ==========================================================================


def test_raise_from_chain_preserves_cause():
    """防回归：适配器失败被转成 Degraded 时丢失原始原因（无法追溯根因）。"""
    with pytest.raises(Degraded) as excinfo:
        try:
            raise Unavailable("llm", "429 too many requests")
        except Unavailable as exc:
            raise Degraded("回退到关键词评分", 2) from exc
    err = excinfo.value
    assert isinstance(err.__cause__, Unavailable)
    assert err.__cause__.provider == "llm"
    assert err.__cause__.reason == "429 too many requests"
    assert err.reason == "回退到关键词评分"
    assert err.level == 2
    assert err.__context__ is err.__cause__


def test_unavailable_attributes_are_readable():
    """防回归：provider/reason 属性名被改（上层读 exc.provider 时 AttributeError）。"""
    err = Unavailable("tts", "quota exceeded")
    assert err.provider == "tts"
    assert err.reason == "quota exceeded"
    assert isinstance(err.provider, str)
    assert isinstance(err.reason, str)


def test_invalid_transition_attributes_are_readable():
    """防回归：state/event 属性名被改（状态机错误处理读不到上下文）。"""
    err = InvalidTransition(State.BUFFER_LISTENING, Event.ANSWER_COMMIT)
    assert err.state is State.BUFFER_LISTENING
    assert err.event is Event.ANSWER_COMMIT
    assert err.state in tuple(State)
    assert err.event in tuple(Event)
    assert str(err) == "invalid transition: buffer_listening --answer_commit--> ?"


def test_budget_exceeded_is_catchable_and_chainable():
    """防回归：预算硬顶异常无法被捕获（预算击穿后静默继续烧钱）。"""
    with pytest.raises(BudgetExceeded):
        try:
            raise Unavailable("llm", "budget")
        except Unavailable as exc:
            raise BudgetExceeded("vlm", 100.0, 130.5) from exc
    try:
        raise BudgetExceeded("vlm", 100.0, 130.5)
    except RuipinError as exc:
        assert "vlm" in str(exc)
        assert "130.5" in str(exc)
        assert "100.0" in str(exc)


def test_reraise_preserves_original_type_and_message():
    """防回归：异常在适配层被"翻译成别的类型"，导致纪律（失败即 Unavailable）失效。"""
    try:
        try:
            raise Unavailable("asr", "模型未加载")
        except Unavailable:
            raise
    except Unavailable as exc:
        assert exc.provider == "asr"
        assert exc.reason == "模型未加载"
        assert str(exc) == "asr unavailable: 模型未加载"


def test_no_error_class_swallows_its_message_on_empty_input():
    """防回归：空字符串入参时 str(e) 退化为只剩标点（日志无法区分"空原因"与"没填"）。"""
    assert str(Unavailable("", "")) == " unavailable: "
    assert str(Degraded("")) == "degraded(L1): "
    assert "invalid transition" in str(InvalidTransition("", ""))
    assert str(BudgetExceeded("")) == "budget exceeded: "
