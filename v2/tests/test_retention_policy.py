"""`ruipin.retention.policy` 的分支/边界测试。

覆盖的纪律：
- 保留表与方案 §12.2 对齐（每类的天数与动作）。
- 报告保留 >36 个月必须显式报错（合规硬上限，不许静默截断）。
- **"从不落盘"与"无限期保留"必须可区分**（都 `due_at=None`，靠 `never_stored` 分辨）。
- 删除必须**真删**：端口返回数与待删数不一致 → `verified=False` 的凭证并写明差多少。
"""

from __future__ import annotations

import pytest

from ruipin.adapters.fakes import DeterministicClock
from ruipin.retention import (
    AUDIT_RETAIN_DAYS,
    DAY_S,
    DEFAULT_REPORT_MONTHS,
    MAX_REPORT_MONTHS,
    MONTH_DAYS,
    DataClass,
    DeletionReceipt,
    PurgePort,
    RetentionAction,
    RetentionEnforcer,
    RetentionPolicy,
    RetentionRule,
)

pytestmark = pytest.mark.unit


# ---------- 测试替身 ----------


class FakePurge:
    """`PurgePort` 的确定性替身。

    `delete_delta` 用来构造"少删/多删"：实际返回 `len(objects) + delete_delta`（下限 0）。
    记录调用以便断言"从不落盘时根本不该碰存储"。
    """

    def __init__(self, objects=(), delete_delta: int = 0) -> None:
        self.objects = tuple(objects)
        self.delete_delta = delete_delta
        self.list_calls: list[tuple[str, DataClass]] = []
        self.delete_calls: list[tuple[str, ...]] = []

    def list_objects(self, subject_id, data_class):
        self.list_calls.append((subject_id, data_class))
        return self.objects

    def delete(self, object_ids):
        self.delete_calls.append(tuple(object_ids))
        return max(0, len(object_ids) + self.delete_delta)


@pytest.fixture
def clk() -> DeterministicClock:
    return DeterministicClock(1_000_000.0)


# ==========================================================================
# 1. 缺省策略表（§12.2）
# ==========================================================================


@pytest.mark.parametrize(
    "data_class,retain_days,action",
    [
        (DataClass.RAW_MEDIA, None, RetentionAction.NONE),
        (DataClass.HR_TIMESERIES, DEFAULT_REPORT_MONTHS * MONTH_DAYS, RetentionAction.DELETE),
        (DataClass.REPORT, DEFAULT_REPORT_MONTHS * MONTH_DAYS, RetentionAction.DELETE),
        (DataClass.STRUCTURED_LOG, 90, RetentionAction.DELETE),
        (DataClass.TRACE_METRIC, 30, RetentionAction.AGGREGATE_THEN_DELETE),
        (DataClass.AUDIT_LOG, AUDIT_RETAIN_DAYS, RetentionAction.ARCHIVE),
    ],
)
def test_default_policy_matches_spec(data_class, retain_days, action):
    """★ 防回归：缺省保留表与方案 §12.2 漂移（合规口径错）。"""
    rule = RetentionPolicy.default().rule(data_class)
    assert rule.retain_days == retain_days, f"{data_class.value} 保留天数不符"
    assert rule.action is action, f"{data_class.value} 到期动作不符"


def test_default_report_months():
    """防回归：缺省报告保留月数被改。"""
    assert RetentionPolicy.default().report_months == DEFAULT_REPORT_MONTHS == 12


# ==========================================================================
# 2. 租户定制与合规上限
# ==========================================================================


@pytest.mark.parametrize("months", [1, 6, 12, MAX_REPORT_MONTHS])
def test_for_tenant_sets_report_and_hr_days(months):
    """防回归：租户定制未同时作用于报告与 HR 时序（两者生命周期脱钩）。"""
    policy = RetentionPolicy.for_tenant(months)
    expected = months * MONTH_DAYS
    assert policy.rule(DataClass.REPORT).retain_days == expected, "报告期必须等于配置月数"
    assert policy.rule(DataClass.HR_TIMESERIES).retain_days == expected, (
        "HR 时序必须与报告同生命周期"
    )
    assert policy.report_months == months, "月数须如实记录"


def test_for_tenant_rejects_over_compliance_limit():
    """★ 防回归：报告保留 >36 个月被接受（违规；也不许静默截断到 36）。"""
    with pytest.raises(ValueError) as e:
        RetentionPolicy.for_tenant(MAX_REPORT_MONTHS + 1)
    assert "合规上限" in str(e.value), "必须报合规上限而非静默截断"
    assert "37" in str(e.value), "消息须带上实际值 37"


@pytest.mark.parametrize("months", [0, -1, -12])
def test_for_tenant_rejects_non_positive(months):
    """防回归：零/负月数被接受（产生无意义甚至反向的到期时刻）。"""
    with pytest.raises(ValueError) as e:
        RetentionPolicy.for_tenant(months)
    assert "必须为正" in str(e.value), "非正月数必须显式拒绝"


# ==========================================================================
# 3. rule() 查表与未知类别
# ==========================================================================


def test_rule_accepts_data_class_and_string_value():
    """防回归：只接受枚举导致字符串调用点 KeyError（调用方体验差）。"""
    policy = RetentionPolicy.default()
    assert policy.rule(DataClass.REPORT).data_class is DataClass.REPORT
    assert policy.rule("report").data_class is DataClass.REPORT, "字符串值也应可查"


@pytest.mark.parametrize("bad", ["nope", None, 123, object()])
def test_rule_unknown_class_raises_keyerror(bad):
    """★ 防回归：未知类别被静默映射到某个默认规则（返回错误策略）。"""
    with pytest.raises(KeyError) as e:
        RetentionPolicy.default().rule(bad)
    assert "未知数据类别" in str(e.value), "未知类别必须 KeyError 且说明"


def test_rule_known_enum_but_missing_from_custom_policy():
    """防回归：策略缺某合法类别时静默返回错误规则（必须查表未命中报错）。"""
    partial = RetentionPolicy({DataClass.REPORT: RetentionRule(DataClass.REPORT, 30, RetentionAction.DELETE)})
    with pytest.raises(KeyError) as e:
        partial.rule(DataClass.RAW_MEDIA)
    assert "已知类别" in str(e.value), "未命中应列出已知类别"


# ==========================================================================
# 4. due_at / is_due 边界
# ==========================================================================


def test_due_at_computes_seconds():
    """防回归：到期时刻换算错误（多/少一天）。"""
    policy = RetentionPolicy.default()
    assert policy.due_at(DataClass.REPORT, 0) == DEFAULT_REPORT_MONTHS * MONTH_DAYS * DAY_S
    assert policy.due_at(DataClass.STRUCTURED_LOG, 0) == 90 * DAY_S
    assert policy.due_at(DataClass.TRACE_METRIC, 100) == 100 + 30 * DAY_S
    assert policy.due_at(DataClass.AUDIT_LOG, 0) == AUDIT_RETAIN_DAYS * DAY_S


def test_due_at_raw_media_is_none_and_never_stored():
    """★ 防回归："不落盘"的类别被算出一个到期时刻（对根本不存在的数据发起删除）。"""
    policy = RetentionPolicy.default()
    assert policy.due_at(DataClass.RAW_MEDIA, 0) is None, "不落盘类别没有到期时刻"
    assert policy.rule(DataClass.RAW_MEDIA).never_stored is True, "never_stored 必须为真"


def test_never_stored_vs_indefinite_retention_are_distinguished():
    """★ 防回归：把"无限期保留"当成"从不落盘"（合规台账语义错）。

    两者 `due_at` 都是 None，必须能用 `never_stored` 区分：
    - 从不落盘：action=NONE 且 retain_days=None → never_stored=True
    - 无限期保留：action!=NONE 且 retain_days=None → never_stored=False
    """
    never = RetentionRule(DataClass.RAW_MEDIA, None, RetentionAction.NONE)
    forever = RetentionRule(DataClass.STRUCTURED_LOG, None, RetentionAction.DELETE)
    policy = RetentionPolicy(
        {DataClass.RAW_MEDIA: never, DataClass.STRUCTURED_LOG: forever}
    )
    assert never.never_stored is True, "从不落盘应为 True"
    assert forever.never_stored is False, "无限期保留应为 False"
    assert policy.due_at(DataClass.RAW_MEDIA, 0) is None
    assert policy.due_at(DataClass.STRUCTURED_LOG, 0) is None, "无限期保留也没有到期时刻"
    assert policy.is_due(DataClass.STRUCTURED_LOG, 0, 10**12) is False, (
        "无限期保留永远不到期"
    )


def test_is_due_boundaries():
    """防回归：到期判定边界口径错（提前一天或晚一天删除）。"""
    policy = RetentionPolicy.default()
    due = policy.due_at(DataClass.TRACE_METRIC, 0)
    assert due == 30 * DAY_S
    assert policy.is_due(DataClass.TRACE_METRIC, 0, due) is True, "恰好到期应判为宜删"
    assert policy.is_due(DataClass.TRACE_METRIC, 0, due - 1) is False, "差一秒不算到期"
    assert policy.is_due(DataClass.TRACE_METRIC, 0, due + 1) is True


def test_is_due_false_for_never_stored():
    """★ 防回归：从不落盘的类别被判为"应删除"（对不存在的数据报删除任务）。"""
    policy = RetentionPolicy.default()
    assert policy.is_due(DataClass.RAW_MEDIA, 0, 10**12) is False


# ==========================================================================
# 5. RetentionRule / DeletionReceipt 校验
# ==========================================================================


def test_retention_rule_never_stored_ok():
    """防回归：合法的"从不落盘"规则无法构造。"""
    rule = RetentionRule(DataClass.RAW_MEDIA, None, RetentionAction.NONE)
    assert rule.never_stored is True


def test_retention_rule_rejects_none_action_with_days():
    """防回归：action=NONE 却带保留天数（自相矛盾，语义不明）。"""
    with pytest.raises(ValueError) as e:
        RetentionRule(DataClass.RAW_MEDIA, 30, RetentionAction.NONE)
    assert "必须为 None" in str(e.value), "NONE 动作不得配 retain_days"


@pytest.mark.parametrize("days", [0, -5])
def test_retention_rule_rejects_non_positive_days(days):
    """防回归：非正保留天数被接受。"""
    with pytest.raises(ValueError) as e:
        RetentionRule(DataClass.REPORT, days, RetentionAction.DELETE)
    assert "必须为正" in str(e.value), "retain_days 必须为正"


def test_retention_rule_rejects_bad_types():
    """防回归：data_class/action 用错类型（枚举分支判断全部失效）。"""
    with pytest.raises(TypeError) as e1:
        RetentionRule("report", 30, RetentionAction.DELETE)  # type: ignore[arg-type]
    assert "data_class" in str(e1.value), "data_class 必须是 DataClass"
    with pytest.raises(TypeError) as e2:
        RetentionRule(DataClass.REPORT, 30, "delete")  # type: ignore[arg-type]
    assert "action" in str(e2.value), "action 必须是 RetentionAction"


def test_deletion_receipt_valid():
    """防回归：合法凭证无法构造。"""
    r = DeletionReceipt(DataClass.REPORT, "sub-1", 5, ("a", "b"), True, "ok")
    assert r.verified is True and r.objects == ("a", "b")


@pytest.mark.parametrize(
    "kwargs,frag",
    [
        ({"subject_id": ""}, "subject_id"),
        ({"evidence": ""}, "evidence"),
    ],
)
def test_deletion_receipt_rejects_empty_strings(kwargs, frag):
    """防回归：凭证缺主体/缺依据（无法定位、无法追溯）。"""
    base = dict(
        data_class=DataClass.REPORT,
        subject_id="s",
        deleted_at=1,
        objects=(),
        verified=True,
        evidence="e",
    )
    base.update(kwargs)
    with pytest.raises(ValueError) as e:
        DeletionReceipt(**base)
    assert frag in str(e.value), f"{frag} 不能为空"


def test_deletion_receipt_rejects_mutable_objects():
    """防回归：凭证用可变列表存对象（事后可被篡改）。"""
    with pytest.raises(TypeError) as e:
        DeletionReceipt(DataClass.REPORT, "s", 1, ["a"], True, "e")  # type: ignore[arg-type]
    assert "tuple" in str(e.value), "objects 必须是 tuple"


def test_deletion_receipt_rejects_bad_data_class():
    """防回归：凭证的数据类别类型错误。"""
    with pytest.raises(TypeError) as e:
        DeletionReceipt("report", "s", 1, (), True, "e")  # type: ignore[arg-type]
    assert "data_class" in str(e.value), "data_class 必须是 DataClass"


# ==========================================================================
# 6. RetentionEnforcer：真删与凭证
# ==========================================================================


def test_fake_purge_satisfies_protocol():
    """防回归：端口协议形同虚设（实现不匹配时无法被 isinstance 识别）。"""
    assert isinstance(FakePurge(), PurgePort), "替身应满足 PurgePort 协议"


def test_purge_never_stored_skips_store(clk):
    """★ 防回归：对"从不落盘"的类别仍去调存储（对不存在的数据操作）。"""
    store = FakePurge(objects=("x",))
    enforcer = RetentionEnforcer(RetentionPolicy.default(), store, clk.now)
    receipt = enforcer.purge(DataClass.RAW_MEDIA, "sub-1")
    assert receipt.objects == (), "从不落盘应无删除对象"
    assert receipt.verified is True, "从不落盘是成功（无数据即无泄露面）"
    assert "从不落盘" in receipt.evidence, "凭证须说明无需删除的理由"
    assert store.list_calls == [], "不得触碰存储的 list"
    assert store.delete_calls == [], "不得触碰存储的 delete"


def test_purge_exact_delete_verified(clk):
    """防回归：正常删除被误判失败。"""
    store = FakePurge(objects=("a", "b", "c"))
    enforcer = RetentionEnforcer(RetentionPolicy.default(), store, clk.now)
    receipt = enforcer.purge(DataClass.REPORT, "sub-1")
    assert receipt.objects == ("a", "b", "c"), "凭证应记录被删对象"
    assert receipt.verified is True, "全部删除应为已核验"
    assert "一致" in receipt.evidence, "凭证须说明数量核对一致"
    assert receipt.deleted_at == int(clk.now()), "deleted_at 必须取注入时钟"


def test_purge_partial_delete_not_verified(clk):
    """★ 防回归：少删一个对象却记为成功（数据残留但凭证说没问题——最危险）。"""
    store = FakePurge(objects=("a", "b", "c"), delete_delta=-1)
    enforcer = RetentionEnforcer(RetentionPolicy.default(), store, clk.now)
    receipt = enforcer.purge(DataClass.REPORT, "sub-1")
    assert receipt.verified is False, "少删必须 verified=False"
    assert "少删 1 个" in receipt.evidence, "凭证须说明差多少"
    assert receipt.objects == ("a", "b", "c"), "凭证应记录全部待删对象，便于补删"


def test_purge_over_delete_not_verified(clk):
    """★ 防回归：端口返回数多于待删数却记为成功（存储返回不可信）。"""
    store = FakePurge(objects=("a", "b"), delete_delta=2)
    enforcer = RetentionEnforcer(RetentionPolicy.default(), store, clk.now)
    receipt = enforcer.purge(DataClass.STRUCTURED_LOG, "sub-1")
    assert receipt.verified is False, "数量异常必须 verified=False"
    assert "多删" in receipt.evidence, "凭证须说明返回数异常"


def test_purge_empty_listing_verified_without_delete(clk):
    """防回归：没有待删对象仍去调用 delete（无意义副作用）。"""
    store = FakePurge(objects=())
    enforcer = RetentionEnforcer(RetentionPolicy.default(), store, clk.now)
    receipt = enforcer.purge(DataClass.STRUCTURED_LOG, "sub-1")
    assert receipt.objects == () and receipt.verified is True, "无对象应为成功空删"
    assert "未找到" in receipt.evidence, "凭证须说明未找到对象"
    assert store.delete_calls == [], "没有对象不应调用 delete"


def test_purge_rejects_empty_subject_before_touching_store(clk):
    """★ 防回归：空主体先删后报错（存储被改却拿不到可用凭证）。"""
    store = FakePurge(objects=("a",))
    enforcer = RetentionEnforcer(RetentionPolicy.default(), store, clk.now)
    with pytest.raises(ValueError) as e:
        enforcer.purge(DataClass.REPORT, "")
    assert "subject_id" in str(e.value), "空主体必须显式报错"
    assert store.list_calls == [] and store.delete_calls == [], "失败前不得触碰存储"


def test_purge_unknown_data_class_raises_keyerror(clk):
    """防回归：未知类别进入删除流程（策略未定义却动手删）。"""
    enforcer = RetentionEnforcer(RetentionPolicy.default(), FakePurge(), clk.now)
    with pytest.raises(KeyError) as e:
        enforcer.purge("not-a-class", "sub-1")
    assert "未知数据类别" in str(e.value), "未知类别必须报错"


def test_purge_uses_tenant_policy_days(clk):
    """防回归：执行器绕过策略自算天数（与租户配置脱钩）。"""
    policy = RetentionPolicy.for_tenant(3)
    enforcer = RetentionEnforcer(policy, FakePurge(objects=("a",)), clk.now)
    receipt = enforcer.purge(DataClass.HR_TIMESERIES, "sub-1")
    assert receipt.data_class is DataClass.HR_TIMESERIES, "凭证应记录实际类别"
    assert receipt.verified is True
