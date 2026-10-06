"""告警规则（`observability/alerts.py`）的测试。

这个文件补的是方案 §12.3「告警」一栏里四条规则从"有阈值"到"真能报"之间的空档。
每条规则都把**边界口径**钉成断言，理由写在用例注释里——边界是最容易被后人无意识
改掉的地方（`>` 改 `>=`、`dip_count` 忘了加、`sd=0` 静默返回空表），而它们恰好
决定了"该报的不报"还是"不该报的乱报"。

三条纪律在测试里各有一组：
1. "判不了" ≠ "没问题"：`sd≤0` / 群体为空 → 空告警 + `undecidable` 点名。
2. 报结论必报口径：`AlertBatch.checked_rules` 与 `undecidable` 都要能查到具体值。
3. 边界钉死：队列 p95 严格大于；降级正好 300.0s 才报；群体偏移恰好 0.3 不报。
"""

from __future__ import annotations

import json
import math

import pytest

from ruipin.observability.alerts import (
    DEFAULT_MAX_SD,
    DEFAULT_MIN_LEVEL,
    DEFAULT_QUEUE_WAIT_MS,
    DEFAULT_SUSTAINED_S,
    DEFAULT_WARN_RATIO,
    RULE_DEGRADATION,
    RULE_GROUP_DRIFT,
    RULE_QUEUE_WAIT,
    RULE_SESSION_COST,
    Alert,
    AlertBatch,
    AlertEvaluator,
    AlertSink,
    DegradationWatch,
    GroupScoreDriftWatch,
    MemoryAlertSink,
    QueueWaitWatch,
    SessionCostWatch,
    Severity,
)


# ======================================================================
# Severity / Alert / AlertSink / MemoryAlertSink
# ======================================================================


def test_severity_values_are_stable_strings():
    """防回归：级别进日志字段，其字符串值必须稳定（便于检索/路由）。"""
    assert Severity.WARNING.value == "warning"
    assert Severity.CRITICAL.value == "critical"
    assert isinstance(Severity.WARNING, str)  # str 混入：可直接进 JSON


def test_alert_rejects_empty_rule_id():
    """防回归：没有规则 id 的告警无法路由，也无人认领。"""
    with pytest.raises(ValueError):
        Alert(
            rule_id="",
            severity=Severity.WARNING,
            message="x",
            value=1.0,
            threshold=0.0,
            ts=0.0,
        )


def test_alert_rejects_bare_string_severity():
    """防回归：severity 必须是枚举——裸字符串会绕过枚举定义、埋下拼写错误。"""
    with pytest.raises(ValueError):
        Alert(
            rule_id="r",
            severity="warning",  # type: ignore[arg-type]
            message="x",
            value=1.0,
            threshold=0.0,
            ts=0.0,
        )


def test_alert_rejects_empty_message():
    """防回归：空消息的告警等于"出事了你猜"，必须拦住。"""
    with pytest.raises(ValueError):
        Alert(
            rule_id="r",
            severity=Severity.WARNING,
            message="",
            value=1.0,
            threshold=0.0,
            ts=0.0,
        )


def test_alert_context_defaults_to_empty_mapping():
    """防回归：不传 context 不应炸；且默认值不能是共享可变对象。"""
    a1 = Alert("r", Severity.WARNING, "m", 1.0, 0.0, 0.0)
    a2 = Alert("r", Severity.WARNING, "m", 1.0, 0.0, 0.0)
    assert dict(a1.context) == {}
    assert a1.context is not a2.context


def test_alert_to_dict_converts_nan_to_none():
    """★ 纪律：to_dict 必须 JSON 安全——NaN 显式转 None。

    很多日志管道用 `json.dumps(..., allow_nan=False)`，一旦出现 NaN 会直接抛，
    告警就发不出去了；把 NaN 留成 None 至少保证告警本身能落地。
    """
    a = Alert(
        rule_id="r",
        severity=Severity.CRITICAL,
        message="m",
        value=float("nan"),
        threshold=0.1,
        ts=float("inf"),
        context={"k": float("nan"), "t": (1.0, 2.0)},
    )
    d = a.to_dict()

    assert d["value"] is None
    assert d["ts"] is None
    assert d["context"]["k"] is None
    assert d["context"]["t"] == [1.0, 2.0]  # tuple 也整理成 list
    # 严格 JSON：allow_nan=False 也不能抛
    json.dumps(d, ensure_ascii=False, allow_nan=False)


def test_memory_sink_emit_rejects_non_alert():
    """防回归：sink 只接受 Alert，别的对象一律拒绝（否则下游解析必崩）。"""
    sink = MemoryAlertSink()
    with pytest.raises(TypeError):
        sink.emit("not an alert")  # type: ignore[arg-type]


def test_memory_sink_is_append_only_and_queryable():
    """★ 纪律：append-only + 口径可查（by_severity / fired_rule_ids / count）。

    append-only 的理由：告警是事后追责凭证，能被删除就不可信。这里没有 delete /
    clear 方法本身就是断言的一部分（`hasattr` 检查）。
    """
    sink = MemoryAlertSink()
    assert isinstance(sink, AlertSink)  # 满足 Protocol
    assert sink.count == 0
    assert sink.alerts == ()
    assert sink.by_severity() == {Severity.WARNING: 0, Severity.CRITICAL: 0}
    assert sink.fired_rule_ids() == ()

    sink.emit(Alert(RULE_QUEUE_WAIT, Severity.WARNING, "w1", 1.0, 0.5, 0.0))
    sink.emit(Alert(RULE_QUEUE_WAIT, Severity.WARNING, "w2", 2.0, 0.5, 1.0))
    sink.emit(Alert(RULE_SESSION_COST, Severity.CRITICAL, "c1", 3.0, 0.5, 2.0))

    assert sink.count == 3
    assert len(sink.alerts) == 3
    assert sink.by_severity() == {Severity.WARNING: 2, Severity.CRITICAL: 1}
    # 去重且保持首次触发顺序
    assert sink.fired_rule_ids() == (RULE_QUEUE_WAIT, RULE_SESSION_COST)
    assert not hasattr(sink, "delete")
    assert not hasattr(sink, "clear")
    json.dumps(sink.to_dict(), ensure_ascii=False, allow_nan=False)


def test_protocol_rejects_object_without_emit():
    """防回归：runtime_checkable 的 Protocol 要能识别"没有 emit"的假 sink。"""

    class _NoEmit:
        pass

    assert not isinstance(_NoEmit(), AlertSink)


# ======================================================================
# 规则 1：降级 Level ≥2 持续 5min
# ======================================================================


def test_degradation_defaults_match_plan():
    """防回归：默认参数直接来自方案 §12.3（Level≥2 持续 5min）。"""
    watch = DegradationWatch()
    assert (DEFAULT_MIN_LEVEL, DEFAULT_SUSTAINED_S) == (2, 300.0)
    assert watch.rule_id == RULE_DEGRADATION


def test_degradation_rejects_bad_min_level():
    """防回归：Level 从 1 起算，min_level<1 等于"任何等级都算降级"，是配置事故。"""
    with pytest.raises(ValueError):
        DegradationWatch(min_level=0)
    with pytest.raises(TypeError):
        DegradationWatch(min_level=2.5)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        DegradationWatch(min_level=True)  # type: ignore[arg-type]


def test_degradation_rejects_non_positive_sustained():
    """防回归：可持续 0 秒 = 立即告警，与"持续"二字直接矛盾。"""
    with pytest.raises(ValueError):
        DegradationWatch(sustained_s=0.0)
    with pytest.raises(ValueError):
        DegradationWatch(sustained_s=-1.0)


def test_degradation_rejects_bad_level_and_ts_types():
    """防回归：level 必须 int；ts 必须是有限数（时钟注入不等于能传垃圾）。"""
    watch = DegradationWatch()
    with pytest.raises(TypeError):
        watch.update(2.5, 0.0)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        watch.update(2, "0")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        watch.update(2, float("nan"))


def test_degradation_single_sample_does_not_fire():
    """★ 纪律：单次采样到 L2 绝不立即告警——否则一条抖动就会刷屏。"""
    watch = DegradationWatch(min_level=2, sustained_s=300.0)
    assert watch.update(2, 0.0) is None
    assert watch.update(3, 1.0) is None
    assert watch.firing is False


def test_degradation_fires_only_after_sustained():
    """★ 纪律：必须"持续满时"才报（300.0s 才触发）。"""
    watch = DegradationWatch(min_level=2, sustained_s=300.0)
    watch.update(2, 1000.0)  # 起点
    assert watch.update(2, 1299.9) is None  # 299.9s < 300s
    alert = watch.update(2, 1300.0)  # 恰好 300.0s
    assert alert is not None
    assert alert.rule_id == RULE_DEGRADATION
    assert alert.severity is Severity.WARNING
    assert alert.value == 2.0
    assert alert.threshold == 2.0
    assert alert.context["elapsed_s"] == pytest.approx(300.0)
    assert "持续" in alert.message and "300.0" in alert.message


def test_degradation_boundary_299_9_vs_300_0():
    """★ 边界口径钉死：299.9s 不报、300.0s 报（是 `>=` 不是 `>`）。"""
    w_under = DegradationWatch(min_level=2, sustained_s=300.0)
    w_under.update(2, 0.0)
    assert w_under.update(2, 299.9) is None

    w_exact = DegradationWatch(min_level=2, sustained_s=300.0)
    w_exact.update(2, 0.0)
    assert w_exact.update(2, 300.0) is not None


def test_degradation_dip_resets_timer_and_counts():
    """★ 纪律：中途掉回 <min_level 要重置计时，并用 dip_count 暴露抖动。

    若忘了重置，"断断续续 L2"会被误判成"持续 L2"；dip_count 让抖动变得可见。
    """
    watch = DegradationWatch(min_level=2, sustained_s=300.0)
    watch.update(2, 0.0)
    watch.update(2, 200.0)
    assert watch.update(1, 250.0) is None  # 掉回 L1 → 重置
    assert watch.dip_count == 1
    # 重置后重新计时：从 250.0 起，到 549.9 只有 299.9s，不报
    watch.update(2, 250.0)
    assert watch.update(2, 549.9) is None
    assert watch.update(2, 550.0) is not None
    assert watch.dip_count == 1


def test_degradation_below_threshold_from_start_is_not_a_dip():
    """防回归：一开始就在阈值以下不算"抖掉"，dip_count 不应无谓增长。"""
    watch = DegradationWatch(min_level=2, sustained_s=300.0)
    assert watch.update(1, 0.0) is None
    assert watch.update(0, 10.0) is None
    assert watch.dip_count == 0


def test_degradation_does_not_refire_while_firing():
    """★ 纪律：已在报警状态时不逐采样重复报（firing 去重）。"""
    watch = DegradationWatch(min_level=2, sustained_s=300.0)
    watch.update(2, 0.0)
    assert watch.update(2, 300.0) is not None
    assert watch.firing is True
    # 继续 L2 / 甚至 L3，都不应再报
    assert watch.update(2, 301.0) is None
    assert watch.update(3, 400.0) is None
    assert watch.firing is True


def test_degradation_can_refire_after_recovery():
    """★ 纪律：恢复（先掉下阈值）后能再次触发——一次告警后哑火同样不可接受。"""
    watch = DegradationWatch(min_level=2, sustained_s=300.0)
    watch.update(2, 0.0)
    assert watch.update(2, 300.0) is not None  # 第一次
    assert watch.update(1, 310.0) is None  # 恢复
    assert watch.dip_count == 1
    watch.update(2, 400.0)  # 重新计时
    assert watch.update(2, 699.9) is None
    assert watch.update(2, 700.0) is not None  # 第二次
    assert watch.dip_count == 1


# ======================================================================
# 规则 2：worker 队列等待 p95 > 10s
# ======================================================================


def test_queue_wait_default_and_validation():
    """防回归：默认阈值来自方案 §12.3（>10s）；非正阈值是配置事故。"""
    watch = QueueWaitWatch()
    assert DEFAULT_QUEUE_WAIT_MS == 10_000.0
    assert watch.threshold_ms == 10_000.0
    assert watch.rule_id == RULE_QUEUE_WAIT
    with pytest.raises(ValueError):
        QueueWaitWatch(threshold_ms=0.0)
    with pytest.raises(ValueError):
        QueueWaitWatch(threshold_ms=-5.0)


def test_queue_wait_exactly_threshold_not_fired():
    """★ 边界口径钉死：阈值比较用**严格大于** —— 正好 10_000.0ms 不报。

    这条注释就是口径本身：将来若有人把它改成 `>=`，这里会红，逼他去确认
    "到底是 > 还是 >="（方案原文是 `>10s`）。
    """
    watch = QueueWaitWatch(threshold_ms=10_000.0)
    assert watch.update(10_000.0, 0.0) is None
    assert watch.update(9_999.9, 0.0) is None


def test_queue_wait_just_over_threshold_warns():
    """防回归：刚过阈值即 WARNING（10_000.1 > 10_000.0）。"""
    watch = QueueWaitWatch(threshold_ms=10_000.0)
    alert = watch.update(10_000.1, 5.0)
    assert alert is not None
    assert alert.severity is Severity.WARNING
    assert alert.value == pytest.approx(10_000.1)
    assert alert.threshold == 10_000.0
    assert alert.ts == 5.0
    assert "10000.1ms" in alert.message and "10000.0ms" in alert.message


def test_queue_wait_double_threshold_is_critical():
    """防回归：≥ 2×阈值（20_000.0ms）升 CRITICAL。"""
    watch = QueueWaitWatch(threshold_ms=10_000.0)
    alert = watch.update(20_000.0, 0.0)
    assert alert is not None
    assert alert.severity is Severity.CRITICAL
    with pytest.raises(ValueError):
        watch.update(float("nan"), 0.0)


def test_queue_wait_ts_must_be_finite():
    """防回归：ts 必须有限。"""
    watch = QueueWaitWatch()
    with pytest.raises(ValueError):
        watch.update(11_000.0, float("inf"))


# ======================================================================
# 规则 3：单会话成本 > 预算 80%
# ======================================================================


def test_session_cost_default_and_validation():
    """防回归：预算必须 >0；warn_ratio 必须落在 (0,1]。"""
    watch = SessionCostWatch(budget_usd=2.0)
    assert watch.budget_usd == 2.0
    assert watch.warn_ratio == DEFAULT_WARN_RATIO == 0.8
    assert watch.rule_id == RULE_SESSION_COST
    with pytest.raises(ValueError):
        SessionCostWatch(budget_usd=0.0)
    with pytest.raises(ValueError):
        SessionCostWatch(budget_usd=-1.0)
    with pytest.raises(ValueError):
        SessionCostWatch(budget_usd=1.0, warn_ratio=0.0)
    with pytest.raises(ValueError):
        SessionCostWatch(budget_usd=1.0, warn_ratio=1.5)


def test_session_cost_below_warn_ratio_not_fired():
    """防回归：79.9% 不报（阈值是 80%）。"""
    watch = SessionCostWatch(budget_usd=10.0)
    assert watch.update(7.99, 0.0) is None


def test_session_cost_at_warn_ratio_warns():
    """防回归：正好 80% 报 WARNING（用 `>=`，与队列的严格大于不同，口径各自钉死）。"""
    watch = SessionCostWatch(budget_usd=10.0)
    alert = watch.update(8.0, 0.0)
    assert alert is not None
    assert alert.severity is Severity.WARNING
    assert alert.value == pytest.approx(0.8)
    assert alert.threshold == 0.8
    assert "80%" in alert.message


def test_session_cost_at_budget_is_critical():
    """防回归：100% 报 CRITICAL（已达/超预算）。"""
    watch = SessionCostWatch(budget_usd=10.0)
    alert = watch.update(10.0, 0.0)
    assert alert is not None
    assert alert.severity is Severity.CRITICAL
    assert alert.threshold == 1.0

    over = watch.update(12.5, 1.0)
    assert over is not None and over.severity is Severity.CRITICAL


def test_session_cost_rejects_non_finite_spent():
    """防回归：NaN 花费会让比较恒 False（静默不报），必须显式拒绝。"""
    watch = SessionCostWatch(budget_usd=1.0)
    with pytest.raises(ValueError):
        watch.update(float("nan"), 0.0)


# ======================================================================
# 规则 4：HR 某群体得分偏移 > 0.3 SD
# ======================================================================


def test_group_drift_default_and_validation():
    """防回归：默认阈值 0.3 SD；非正阈值无意义。"""
    watch = GroupScoreDriftWatch()
    assert DEFAULT_MAX_SD == 0.3
    assert watch.max_sd == 0.3
    assert watch.rule_id == RULE_GROUP_DRIFT
    with pytest.raises(ValueError):
        GroupScoreDriftWatch(max_sd=0.0)
    with pytest.raises(ValueError):
        GroupScoreDriftWatch(max_sd=-0.1)


def test_group_drift_lists_every_offending_group():
    """★ 纪律：多群体同时越界要**全部列出**，不许只报第一个。

    只报第一个会让后续群体长期无人处理——告警的完整性在这里。
    """
    watch = GroupScoreDriftWatch(max_sd=0.3)
    alerts = watch.update(
        group_means={"A": 0.5, "B": -0.4, "C": 0.1},
        overall_mean=0.0,
        sd=1.0,
        ts=7.0,
    )
    assert len(alerts) == 2  # C 未越界
    groups = {a.context["group"] for a in alerts}
    assert groups == {"A", "B"}
    for a in alerts:
        assert a.rule_id == RULE_GROUP_DRIFT
        assert a.severity is Severity.WARNING
        assert a.threshold == 0.3
        assert a.ts == 7.0
        # 消息必须自带证据：群体名、该群体均值、整体均值、SD、倍数、阈值
        assert a.context["group"] in a.message
        assert f"{a.context['multiples']:.3f}" in a.message
        assert "0.300" in a.message
    assert watch.last_undecidable is None


def test_group_drift_exactly_threshold_not_fired():
    """★ 边界口径钉死：恰好 0.3 SD 不报（严格大于）。"""
    watch = GroupScoreDriftWatch(max_sd=0.3)
    assert watch.update({"A": 0.3}, overall_mean=0.0, sd=1.0, ts=0.0) == []
    assert watch.update({"A": 0.3001}, overall_mean=0.0, sd=1.0, ts=0.0) != []


def test_group_drift_sd_zero_is_undecidable_not_healthy():
    """★ 纪律：sd=0 判不了，返回空列表并写明原因——绝不能读成"无偏移"。"""
    watch = GroupScoreDriftWatch(max_sd=0.3)
    alerts = watch.update({"A": 5.0, "B": -5.0}, overall_mean=0.0, sd=0.0, ts=0.0)
    assert alerts == []
    assert watch.last_undecidable is not None
    assert "sd" in watch.last_undecidable
    assert "不是'无偏移'" in watch.last_undecidable


def test_group_drift_negative_sd_is_undecidable():
    """防回归：负 SD 同样判不了（物理上不可能，但它绝不能等于"无偏移"）。"""
    watch = GroupScoreDriftWatch(max_sd=0.3)
    assert watch.update({"A": 5.0}, overall_mean=0.0, sd=-1.0, ts=0.0) == []
    assert watch.last_undecidable is not None


def test_group_drift_empty_groups_is_undecidable():
    """★ 纪律：没有群体可比 = 判不了，不是"无偏移"。"""
    watch = GroupScoreDriftWatch(max_sd=0.3)
    assert watch.update({}, overall_mean=0.0, sd=1.0, ts=0.0) == []
    assert watch.last_undecidable is not None
    assert "group_means 为空" in watch.last_undecidable


def test_group_drift_rejects_non_finite_means():
    """防回归：NaN 均值参与比较恒 False，会静默漏报，必须显式拒绝。"""
    watch = GroupScoreDriftWatch(max_sd=0.3)
    with pytest.raises(ValueError):
        watch.update({"A": float("nan")}, overall_mean=0.0, sd=1.0, ts=0.0)
    with pytest.raises(ValueError):
        watch.update({"A": 1.0}, overall_mean=float("inf"), sd=1.0, ts=0.0)


# ======================================================================
# AlertBatch / AlertEvaluator
# ======================================================================


def _evaluator(sink: MemoryAlertSink, **kwargs) -> AlertEvaluator:
    return AlertEvaluator(sink, session_budget_usd=10.0, **kwargs)


def test_evaluator_all_inputs_checks_all_rules():
    """★ 纪律：输入齐备时 checked_rules=4 且 undecidable 为空——"真跑了 4 条"。"""
    sink = MemoryAlertSink()
    ev = _evaluator(sink)
    batch = ev.evaluate(
        ts=0.0,
        level=2,  # 单次 L2：不报（持续规则）
        queue_p95_ms=11_000.0,  # WARNING
        session_spent_usd=8.5,  # 85% > 80% → WARNING
        group_means={"A": 0.5},  # 0.5 SD > 0.3 → WARNING
        overall_mean=0.0,
        sd=1.0,
    )
    assert isinstance(batch, AlertBatch)
    assert batch.checked_rules == 4
    assert batch.undecidable == {}
    assert len(batch.alerts) == 3
    assert sink.count == 3
    assert set(sink.fired_rule_ids()) == {RULE_QUEUE_WAIT, RULE_SESSION_COST, RULE_GROUP_DRIFT}
    assert batch.by_severity()[Severity.WARNING] == 3
    assert batch.by_severity()[Severity.CRITICAL] == 0


def test_evaluator_missing_inputs_go_to_undecidable():
    """★ 纪律：缺输入不是"跳过"，而是进 undecidable 点名——"没给数据"≠"数据正常"。"""
    sink = MemoryAlertSink()
    ev = _evaluator(sink)
    batch = ev.evaluate(ts=0.0, level=2)  # 只给了降级等级

    assert batch.checked_rules == 1
    assert set(batch.undecidable) == {RULE_QUEUE_WAIT, RULE_SESSION_COST, RULE_GROUP_DRIFT}
    assert "queue_p95_ms" in batch.undecidable[RULE_QUEUE_WAIT]
    assert batch.alerts == ()
    assert sink.count == 0


def test_evaluator_group_drift_needs_all_three_inputs():
    """防回归：群体规则需要 group_means / overall_mean / sd 三个一起给，缺一即判不了。"""
    sink = MemoryAlertSink()
    ev = _evaluator(sink)
    batch = ev.evaluate(ts=0.0, group_means={"A": 0.9}, overall_mean=0.0)  # 缺 sd
    assert RULE_GROUP_DRIFT in batch.undecidable
    assert "sd" in batch.undecidable[RULE_GROUP_DRIFT]


def test_evaluator_group_drift_sd_zero_is_checkable_but_undecidable():
    """★ 纪律：sd=0 时规则**被查了**（输入齐备，计入 checked_rules），
    但结论是"判不了"，同时进 undecidable——两种口径必须并存可见。"""
    sink = MemoryAlertSink()
    ev = _evaluator(sink)
    batch = ev.evaluate(
        ts=0.0,
        level=1,  # 未达 L2
        queue_p95_ms=1_000.0,  # 未越界，无告警（覆盖"给了输入但不触发"的分支）
        session_spent_usd=1.0,  # 10% 预算
        group_means={"A": 0.9},
        overall_mean=0.0,
        sd=0.0,  # 判不了
    )
    assert batch.checked_rules == 4  # 四条输入都齐了
    assert RULE_GROUP_DRIFT in batch.undecidable
    assert batch.alerts == ()
    assert sink.count == 0


def test_evaluator_emits_all_group_alerts():
    """防回归：评估器要把同一规则的**多条**告警全部 emit，不能只留第一条。"""
    sink = MemoryAlertSink()
    ev = _evaluator(sink)
    batch = ev.evaluate(
        ts=0.0,
        group_means={"A": 0.5, "B": 0.6},
        overall_mean=0.0,
        sd=1.0,
    )
    assert len(batch.alerts) == 2
    assert sink.count == 2


def test_evaluator_degradation_fires_across_calls():
    """防回归：降级看门对象跨多次 evaluate 保持计时状态（状态封在 watch 里）。"""
    sink = MemoryAlertSink()
    ev = _evaluator(sink, degradation=DegradationWatch(min_level=2, sustained_s=100.0))
    ev.evaluate(ts=0.0, level=2)
    batch = ev.evaluate(ts=100.0, level=2)
    degrading = [a for a in batch.alerts if a.rule_id == RULE_DEGRADATION]
    assert len(degrading) == 1
    assert degrading[0].severity is Severity.WARNING


def test_evaluator_summary_and_to_dict_are_json_safe():
    """★ 纪律：给出结论必给口径——summary 必须同时含告警数、已查规则数、判不了数。"""
    sink = MemoryAlertSink()
    ev = _evaluator(sink)
    batch = ev.evaluate(
        ts=0.0,
        level=1,  # 未达 L2，无告警
        queue_p95_ms=25_000.0,  # CRITICAL
        session_spent_usd=1.0,  # 10% 预算，无告警
        group_means={"A": 0.9},
        overall_mean=0.0,
        sd=0.0,  # 判不了
    )
    text = batch.summary()
    assert "告警 1 条" in text
    assert "已查 4 条规则" in text
    assert "1 条判不了" in text

    payload = json.dumps(batch.to_dict(), ensure_ascii=False, allow_nan=False)
    back = json.loads(payload)
    assert back["checked_rules"] == 4
    assert back["undecidable"][RULE_GROUP_DRIFT]
    assert back["alerts"][0]["severity"] == "critical"


def test_evaluator_rejects_non_finite_ts():
    """防回归：批级 ts 也要有限。"""
    sink = MemoryAlertSink()
    ev = _evaluator(sink)
    with pytest.raises(ValueError):
        ev.evaluate(ts=float("nan"), level=2)


def test_evaluator_exposes_component_watches():
    """防回归：评估器要能取回各看门对象（排查"这条规则到底用没用自己的阈值"）。"""
    sink = MemoryAlertSink()
    ev = _evaluator(sink)
    assert isinstance(ev.degradation, DegradationWatch)
    assert isinstance(ev.queue_wait, QueueWaitWatch)
    assert isinstance(ev.session_cost, SessionCostWatch)
    assert isinstance(ev.group_drift, GroupScoreDriftWatch)
    assert ev.rule_count == 4


def test_numeric_helper_rejects_non_numeric_object():
    """防回归：无法转 float 的对象要抛 TypeError，而不是 TypeError 之外的怪错。"""
    watch = DegradationWatch()
    with pytest.raises(TypeError):
        watch.update(2, object())  # type: ignore[arg-type]
