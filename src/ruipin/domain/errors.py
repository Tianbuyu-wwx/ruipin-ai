"""错误类型。

核心纪律：适配器失败时必须抛 Unavailable，**禁止返回合成/default 分数**。
这是 v2 相对现系统的第一原则（现系统崩溃即返回全 60 分，与真实 60 分不可区分）。

可序列化
--------
异常要能跨进程边界（Redis Streams 解耦的 worker 池、任务队列、日志采集）传播，
因此每个子类都实现了 `__reduce__`，保证 `pickle.loads(pickle.dumps(e))` 后
**属性与消息完全不变**。基类的 `args` 是派生出来的字符串，不能直接拿来重建子类
（子类 `__init__` 的参数列表与 `Exception.args` 不同），所以显式声明重建元组。
"""


class RuipinError(Exception):
    """项目内所有异常的基类。"""


class Unavailable(RuipinError):
    """能力不可用。适配器必须抛此异常，不得返回兜底分数。"""

    def __init__(self, provider: str, reason: str):
        self.provider = provider
        self.reason = reason
        super().__init__(f"{provider} unavailable: {reason}")

    def __reduce__(self):
        return (self.__class__, (self.provider, self.reason))


class Degraded(RuipinError):
    """降级：能力退化但仍可用，携带降级原因。"""

    def __init__(self, reason: str, level: int = 1):
        self.reason = reason
        self.level = level
        super().__init__(f"degraded(L{level}): {reason}")

    def __reduce__(self):
        return (self.__class__, (self.reason, self.level))


class InvalidTransition(RuipinError):
    """非法的状态转换。

    消息里用枚举的 `.value` 而不是枚举本身——`str(Enum)` 的输出随 Python 版本变化
    （3.11 起 `Enum.__str__` 改了，`State.BUFFER` 可能印成 `State.BUFFER` 或 `buffer`），
    而这条消息会进日志检索与断言，必须稳定。
    """

    def __init__(self, state, event):
        self.state = state
        self.event = event
        super().__init__(
            f"invalid transition: {_val(state)} --{_val(event)}--> ?"
        )

    def __reduce__(self):
        return (self.__class__, (self.state, self.event))


class BudgetExceeded(RuipinError):
    """预算硬顶被突破。

    `kind` 说明被顶破的是哪一类预算（如 `vlm_frames` / `llm_usd` / `tts_chars`）。
    `limit` / `used` 可选，但**有值时必须如实填**——告警要能直接看出超了多少，
    不能让人回代码里推。
    """

    def __init__(self, kind: str, limit: float | None = None, used: float | None = None):
        self.kind = kind
        self.limit = limit
        self.used = used
        msg = f"budget exceeded: {kind}"
        if limit is not None and used is not None:
            msg += f"（已用 {used}，上限 {limit}）"
        elif limit is not None:
            msg += f"（上限 {limit}）"
        elif used is not None:
            # 只给了用量没给上限时也要把数值说出来——告警里出现一个光秃秃的
            # kind 而没有任何数字，等于让人自己去代码里猜。
            msg += f"（已用 {used}）"
        super().__init__(msg)

    def __reduce__(self):
        return (self.__class__, (self.kind, self.limit, self.used))


def _val(x) -> str:
    """取枚举的值；非枚举原样转字符串。"""
    return str(getattr(x, "value", x))
