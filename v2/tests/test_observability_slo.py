"""SLO 定义与违约检测（`observability/slo.py`）的测试。

这个文件补的也是一个真空：`slo.py` 一度只有 52% 覆盖，而它承载的是"延迟劣化能
不能自动被发现"。这里把两条纪律钉住：

1. **样本不足 = `unknown`，绝不报 ok**。样本不够就宣称达标，比不测更危险——
   它会让告警静音，掩盖真实的长尾。
2. **meter 不具备分位数查询能力时抛 `TypeError`**，不许静默判 ok 或崩在 None 上。

同时把三条预置 SLO 的阈值写成回归锚点：它们直接对应方案自检 A2 修订后的口径
（单轮 p50 ≤ 4000ms、p95 ≤ 8000ms、加权 p95 ≤ 7000ms），被谁改动了要被测试看见。
"""

from __future__ import annotations

import json

import pytest

from ruipin.observability.metrics import InMemoryMeter
from ruipin.observability.slo import (
    DEFAULT_MIN_SAMPLES,
    DEFAULT_SLOS,
    HUMAN_INTERVIEWER_GAP_MS,
    METRIC_TURN_LATENCY_MS,
    METRIC_TURN_WEIGHTED_LATENCY_MS,
    SLO_TURN_LATENCY_P50,
    SLO_TURN_LATENCY_P95,
    SLO_TURN_WEIGHTED_LATENCY_P95,
    SLO,
    SLOBatch,
    SLOStatus,
    check_all,
    evaluate_slo,
)

#: 样本数达到 min_samples 的最小充分量，避免每个用例都灌 20 条。
MIN = 5


def _meter_with(values: list[float]) -> InMemoryMeter:
    m = InMemoryMeter()
    for v in values:
        m.observe_ms(METRIC_TURN_LATENCY_MS, v)
    return m


# ---------- SLO 构造校验 ----------


def test_slo_rejects_invalid_percentile():
    """防回归：p=0 或 p>1 会让分位数查询直接抛错，构造期拦住更好定位。"""
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            SLO(name="x", metric="m", threshold_ms=100.0, percentile=bad)


def test_slo_rejects_non_positive_threshold():
    """防回归：阈值 ≤ 0 会让任何实测值都"违约"，属于配置事故而非真实劣化。"""
    with pytest.raises(ValueError):
        SLO(name="x", metric="m", threshold_ms=0.0, percentile=0.5)
    with pytest.raises(ValueError):
        SLO(name="x", metric="m", threshold_ms=-1.0, percentile=0.5)


def test_slo_rejects_invalid_min_samples():
    """防回归：min_samples < 1 意味着"零样本也算达标"，与 unknown 纪律直接冲突。"""
    with pytest.raises(ValueError):
        SLO(name="x", metric="m", threshold_ms=100.0, percentile=0.5, min_samples=0)


def test_percentile_label_formatting():
    """防回归：告警文案里的分位标签必须是人读的 p50/p95，不是 0.5/0.95。"""
    assert SLO_TURN_LATENCY_P50.label == "p50"
    assert SLO_TURN_LATENCY_P95.label == "p95"


# ---------- 预置 SLO 的回归锚点 ----------


def test_preset_slo_thresholds_are_anchored_to_plan():
    """防回归：三条阈值来自方案自检 A2 的修订口径，被改动必须被看见。"""
    assert (SLO_TURN_LATENCY_P50.threshold_ms, SLO_TURN_LATENCY_P50.percentile) == (4000.0, 0.50)
    assert (SLO_TURN_LATENCY_P95.threshold_ms, SLO_TURN_LATENCY_P95.percentile) == (8000.0, 0.95)
    assert (
        SLO_TURN_WEIGHTED_LATENCY_P95.threshold_ms,
        SLO_TURN_WEIGHTED_LATENCY_P95.percentile,
        SLO_TURN_WEIGHTED_LATENCY_P95.weighted,
    ) == (7000.0, 0.95, True)
    assert SLO_TURN_LATENCY_P50.metric == METRIC_TURN_LATENCY_MS
    assert SLO_TURN_WEIGHTED_LATENCY_P95.metric == METRIC_TURN_WEIGHTED_LATENCY_MS
    assert len(DEFAULT_SLOS) == 3


def test_plan_source_is_recorded_on_every_preset_slo():
    """防回归：每条 SLO 都要能回答"这个阈值哪来的"——否则告警无法回溯。"""
    for slo in DEFAULT_SLOS:
        assert slo.source, f"{slo.name} 缺少方案出处"
        assert "方案" in slo.source


def test_human_interviewer_gap_is_the_experience_anchor():
    """防回归：体验基准是"真人间隔 1–3 秒"，不是"越快越好"（自检 A2）。"""
    assert HUMAN_INTERVIEWER_GAP_MS == (1000, 3000)
    assert DEFAULT_MIN_SAMPLES == 20


# ---------- unknown：样本不足是一等公民 ----------


def test_insufficient_samples_yields_unknown_not_ok():
    """★ 红线：样本不足必须 unknown，且 message 里报出"几条/需要几条"的口径。

    这条与"报 0 项问题必须同时报口径"是同一个道理：只说 unknown 而不说
    样本数，分不清是"真不够"还是"根本没采到"。
    """
    meter = _meter_with([1000.0, 1000.0])  # 2 条，远低于 20
    report = evaluate_slo(meter, SLO_TURN_LATENCY_P50)

    assert report.status is SLOStatus.UNKNOWN
    assert report.observed_ms is None
    assert report.samples == 2
    assert report.min_samples == DEFAULT_MIN_SAMPLES
    assert "样本不足" in report.message
    assert "2/20" in report.message
    assert "UNKNOWN" in report.message


def test_unknown_when_observed_is_none_despite_enough_samples():
    """防回归：样本计数够但分位数查不出来（如上游导出缺失）时，仍是 unknown。

    若这里回落到 ok，就会把"指标没采到"读成"性能很好"。
    """

    class _HalfMeter:
        def count(self, name: str) -> int:
            return 999  # 假装样本很多

        def percentile(self, name: str, p: float):
            return None  # 但根本算不出来

    report = evaluate_slo(_HalfMeter(), SLO_TURN_LATENCY_P50)  # type: ignore[arg-type]

    assert report.status is SLOStatus.UNKNOWN
    assert report.observed_ms is None
    assert report.samples == 999


def test_min_samples_override_is_respected():
    """防回归：调用方可按场景收紧/放宽样本门槛，覆盖应生效且进 message。"""
    meter = _meter_with([10.0, 20.0, 30.0])
    report = evaluate_slo(meter, SLO_TURN_LATENCY_P50, min_samples=3)

    assert report.status is SLOStatus.OK
    assert report.min_samples == 3
    assert "样本 3 条" in report.message


def test_invalid_min_samples_override_rejected():
    """防回归：把门槛设成 0 等于取消样本要求，必须拒绝。"""
    with pytest.raises(ValueError):
        evaluate_slo(_meter_with([1.0]), SLO_TURN_LATENCY_P50, min_samples=0)


# ---------- ok / breached ----------


def test_breached_reports_actual_threshold_and_samples():
    """★ 告警文案必须自带证据：实际值、阈值、样本数缺一不可。

    只报"延迟超标"而不报这三个数，收到告警的人还得自己去查。
    """
    meter = _meter_with([9000.0] * MIN)
    report = evaluate_slo(meter, SLO_TURN_LATENCY_P50, min_samples=MIN)

    assert report.status is SLOStatus.BREACHED
    assert report.observed_ms == pytest.approx(9000.0)
    assert report.samples == MIN
    assert "BREACHED" in report.message
    assert "9000.0ms" in report.message
    assert "4000.0ms" in report.message
    assert ">" in report.message
    assert f"样本 {MIN} 条" in report.message


def test_ok_reports_evidence_too():
    """防回归：达标也要带证据——否则无法区分"确实达标"与"没测"。
    （后者由 status=unknown 承担，两者口径不同，不能混。）"""
    meter = _meter_with([1000.0] * MIN)
    report = evaluate_slo(meter, SLO_TURN_LATENCY_P50, min_samples=MIN)

    assert report.status is SLOStatus.OK
    assert "OK" in report.message
    assert "1000.0ms" in report.message
    assert "≤" in report.message
    assert f"样本 {MIN} 条" in report.message


def test_threshold_boundary_is_inclusive():
    """防回归：恰好等于阈值判达标（`≤` 而非 `<`）。

    边界口径写错会造成"同一份数据在两侧被判成不同状态"，是最难排查的一类抖动。
    """
    meter = _meter_with([4000.0] * MIN)
    report = evaluate_slo(meter, SLO_TURN_LATENCY_P50, min_samples=MIN)

    assert report.observed_ms == pytest.approx(4000.0)
    assert report.status is SLOStatus.OK


def test_just_over_threshold_is_breached():
    """防回归：刚过阈值必须违约，不能被浮点比较糊过去。"""
    meter = _meter_with([4000.1] * MIN)
    assert evaluate_slo(meter, SLO_TURN_LATENCY_P50, min_samples=MIN).status is SLOStatus.BREACHED


# ---------- 加权路径 ----------


def test_weighted_slo_uses_weighted_series():
    """防回归：加权 SLO 必须读加权序列，不能回落到未加权分位数。

    "按题型加权"正是为了防"某类题型常年很慢却被总体分位数平均掉"，
    读错序列会让这条保护完全失效。
    """
    meter = InMemoryMeter()
    for ms, w in ((1000.0, 1.0), (9000.0, 19.0)):
        meter.observe_weighted_ms(METRIC_TURN_WEIGHTED_LATENCY_MS, ms, w)

    report = evaluate_slo(meter, SLO_TURN_WEIGHTED_LATENCY_P95, min_samples=2)

    assert report.status in (SLOStatus.OK, SLOStatus.BREACHED)  # 有实测值即可
    assert report.observed_ms is not None
    # 加权样本只落在加权序列里：未加权序列应为空，证明没有读错序列
    assert meter.count(METRIC_TURN_WEIGHTED_LATENCY_MS) == 0


def test_weighted_slo_reports_unknown_when_weighted_series_empty():
    """防回归：加权序列为空时即使未加权序列有数据也必须 unknown（不许跨序列借数）。"""
    meter = InMemoryMeter()
    for _ in range(30):
        meter.observe_ms(METRIC_TURN_WEIGHTED_LATENCY_MS, 500.0)

    report = evaluate_slo(meter, SLO_TURN_WEIGHTED_LATENCY_P95)

    assert report.status is SLOStatus.UNKNOWN
    assert report.samples == 0
    assert report.observed_ms is None


def test_meter_lacking_percentile_capability_raises_type_error():
    """★ 红线：meter 不支持分位数查询时抛 TypeError，不许静默判 ok。"""

    class _NoPercentile:
        def count(self, name: str) -> int:
            return 100

    with pytest.raises(TypeError) as exc:
        evaluate_slo(_NoPercentile(), SLO_TURN_LATENCY_P50)  # type: ignore[arg-type]

    assert "percentile" in str(exc.value)
    assert "不许静默判达标" in str(exc.value)


def test_meter_lacking_weighted_capability_raises_type_error():
    """防回归：只实现了未加权接口的 meter 不能拿来评加权 SLO。"""

    class _UnweightedOnly:
        def count(self, name: str) -> int:
            return 100

        def percentile(self, name: str, p: float):
            return 1.0

    with pytest.raises(TypeError) as exc:
        evaluate_slo(_UnweightedOnly(), SLO_TURN_WEIGHTED_LATENCY_P95)  # type: ignore[arg-type]

    assert "percentile_weighted" in str(exc.value)


# ---------- 批量 ----------


def test_check_all_mixes_all_three_statuses():
    """防回归：三条 SLO 同时出现 ok / breached / unknown 时，汇总必须分别计数。"""
    meter = InMemoryMeter()
    # 未加权序列：灌 20 条 9000ms → p50 与 p95 都超 4000/8000 ⇒ 两条 breached
    for _ in range(DEFAULT_MIN_SAMPLES):
        meter.observe_ms(METRIC_TURN_LATENCY_MS, 9000.0)
    # 加权序列一条都不灌 → 加权 p95 样本不足 ⇒ unknown

    batch = check_all(meter)

    assert isinstance(batch, SLOBatch)
    assert len(batch.reports) == 3
    assert batch.any_breached is True
    assert batch.any_unknown is True
    assert len(batch.breached()) == 2
    assert batch.summary() == "SLO: 2/3 违约, 1/3 样本不足"


def test_check_all_all_ok():
    """防回归：全达标时 any_breached / any_unknown 都应为 False。"""
    meter = InMemoryMeter()
    for _ in range(DEFAULT_MIN_SAMPLES):
        meter.observe_ms(METRIC_TURN_LATENCY_MS, 1500.0)
        meter.observe_weighted_ms(METRIC_TURN_WEIGHTED_LATENCY_MS, 1500.0, 1.0)

    batch = check_all(meter)

    assert batch.any_breached is False
    assert batch.any_unknown is False
    assert batch.summary() == "SLO: 0/3 违约, 0/3 样本不足"


def test_check_all_empty_list_is_not_breached():
    """防回归：空清单时"没有任何违约"成立，但也不能被读成"全部达标"——
    所以只能断言 any_breached=False，同时 summary 明确显示 0/0。"""
    batch = check_all(InMemoryMeter(), slos=())

    assert batch.any_breached is False
    assert batch.any_unknown is False
    assert batch.reports == ()
    assert batch.summary() == "SLO: 0/0 违约, 0/0 样本不足"


def test_check_all_with_custom_slo_list():
    """防回归：应支持只判定指定子集（灰度期可能只看一两条）。"""
    meter = _meter_with([500.0] * DEFAULT_MIN_SAMPLES)
    batch = check_all(meter, slos=(SLO_TURN_LATENCY_P50,))

    assert len(batch.reports) == 1
    assert batch.reports[0].status is SLOStatus.OK


# ---------- 序列化 ----------


def test_reports_are_json_safe_with_strict_floats():
    """防回归：告警要能被严格 JSON 序列化——`observed_ms=None` 必须真的是 None，
    不能是 NaN（`allow_nan=False` 会直接抛，很多日志管道就是这么配的）。"""
    meter = _meter_with([9000.0] * MIN)
    batch = check_all(meter, slos=DEFAULT_SLOS, min_samples=MIN)

    payload = json.dumps(batch.as_dict(), ensure_ascii=False, allow_nan=False)
    back = json.loads(payload)

    assert back["any_breached"] is True
    assert back["summary"].startswith("SLO:")
    unknown = [r for r in back["reports"] if r["status"] == "unknown"]
    assert unknown and all(r["observed_ms"] is None for r in unknown)


def test_unknown_report_dict_keeps_samples_and_min_samples():
    """防回归：unknown 的字段不能因为"没值"就把口径信息也省掉。"""
    meter = _meter_with([1000.0])
    d = evaluate_slo(meter, SLO_TURN_LATENCY_P50).as_dict()

    assert d["status"] == "unknown"
    assert d["observed_ms"] is None
    assert d["samples"] == 1
    assert d["min_samples"] == DEFAULT_MIN_SAMPLES
    assert d["source"]
