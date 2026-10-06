"""端侧窗口流 → 事件指标 → 压力调节结果（`physio/collector.py`）的测试。

这一层的价值在于**闭合断链**：没有它，"摄像头测到心率"与"心率进入评分"之间
没有任何代码负责。所以这里的用例不是形式化凑覆盖率，而是逐条钉住详设
§4.1–4.4 的公式口径，以及三条纪律：

1. **弃权优于猜测**：`bpm is None` 的窗口不能当 0 参与统计。
2. **基线期 SNR 门控是整场开关**：环境不支持就说"不做这项"，不硬算。
3. **单个事件永不进入评分**：本层只产出 `InterviewEvent`，放行权在 metrics/regulation。

合成信号用**已知真值**构造（T50=16s、ΔHR 逐题递减以产生习惯化斜率），
因此能对拍出"算得对不对"，而不是只看"跑没跑过"。
"""

from __future__ import annotations

import math

import pytest

from ruipin.physio.collector import (
    BASELINE_MIN_SNR,
    BASELINE_REST_MS,
    PEAK_WINDOW_MS,
    PRE_WINDOW_MS,
    StressRegulationCollector,
)
from ruipin.physio.regulation import W_MAX, gate
from ruipin.ports import (
    EnvQuality,
    EventAnchor,
    HRBatch,
    HRSample,
    StressRegulationPort,
)

# ---------- 合成信号的真值 ----------

BASE_HR = 70.0
SNR = 0.8
T50_TRUE_S = 16.0
EVENT_STEP_MS = 100_000
T_Q0 = 100_000
T_A_OFFSET_MS = 5_000
T_E_OFFSET_MS = 35_000
SAMPLE_STEP_MS = 2_000


def _delta_of(i: int) -> float:
    """第 i 题的绝对反应幅度（BPM）。

    刻意**逐题递减**（24, 23, 22, …）：习惯化斜率 β_hab 是对 `r_i` 的回归斜率，
    若每题幅度相同则自变量无变异、无法回归，`evaluate_physio` 会因"跨事件统计量
    不可用"而整项不计入。真实面试里适应过程本来就会让反应幅度收窄。
    """
    return 24.0 - 1.0 * i


def _anchor(i: int) -> EventAnchor:
    t_q = T_Q0 + i * EVENT_STEP_MS
    return EventAnchor(
        turn_index=i,
        t_q_ms=t_q,
        t_a_ms=t_q + T_A_OFFSET_MS,
        t_e_ms=t_q + T_E_OFFSET_MS,
    )


def _rest_samples(bpm: float = BASE_HR, snr: float = SNR) -> list[HRSample]:
    return [
        HRSample(t_ms=t, bpm=bpm, snr=snr)
        for t in range(0, BASELINE_REST_MS + SAMPLE_STEP_MS, SAMPLE_STEP_MS)
    ]


def _event_samples(i: int, *, t50_s: float = T50_TRUE_S, snr: float = SNR) -> list[HRSample]:
    """第 i 题的窗口序列：静息前窗 + 峰值 + 指数回落。

    `HR(t) = pre + Δ·exp(−λ(t − t_peak))`，`λ = ln2 / T50`（详设 §4.3）。
    """
    a = _anchor(i)
    delta = _delta_of(i)
    lam = math.log(2.0) / t50_s
    out: list[HRSample] = []

    # pre 窗：[t_q − 3s, t_q]，每 1s 一条
    for t in range(a.t_q_ms - PRE_WINDOW_MS, a.t_q_ms + 1, 1_000):
        out.append(HRSample(t_ms=t, bpm=BASE_HR, snr=snr))

    # [t_a, t_e]：峰值在 t_a，之后按 λ 回落
    for t in range(a.t_a_ms, a.t_e_ms + 1, SAMPLE_STEP_MS):
        dt = (t - a.t_a_ms) / 1000.0
        out.append(
            HRSample(t_ms=t, bpm=round(BASE_HR + delta * math.exp(-lam * dt), 4), snr=snr)
        )
    return out


def _collector(n_events: int = 8, **kwargs) -> StressRegulationCollector:
    """构造一个已完成静息期 + n 个事件的收集器。"""
    c = StressRegulationCollector("s-test", **kwargs)
    c.push_batch(
        HRBatch(
            session_id="s-test",
            samples=tuple(_rest_samples()),
            algo_agreement=1.0,
            env=EnvQuality(fps=30.0, brightness=0.5, face_ratio=0.25, flicker=0.02),
        )
    )
    c.baseline_from_rest()
    for i in range(n_events):
        c.push_batch(
            HRBatch(
                session_id="s-test",
                samples=tuple(_event_samples(i)),
                algo_agreement=1.0,
            )
        )
        c.mark_event(_anchor(i))
    return c


# ---------- 端口契约 ----------


def test_collector_satisfies_stress_regulation_port():
    """防回归：收集器必须可被当作 `StressRegulationPort` 注入（换实现只改适配器）。"""
    assert isinstance(_collector(2), StressRegulationPort)


# ---------- 批次归并与污染防护 ----------


def test_out_of_order_batches_are_merged_by_timestamp():
    """防回归：弱网重传会让批次乱序到达，不按 t_ms 归并会让 pre/peak 的切片错位。"""
    c = StressRegulationCollector("s1")
    c.push_batch(HRBatch("s1", samples=(HRSample(3000, 80.0), HRSample(1000, 70.0))))
    c.push_batch(HRBatch("s1", samples=(HRSample(2000, 75.0),)))

    assert [s.t_ms for s in c.samples()] == [1000, 2000, 3000]


def test_duplicate_timestamps_are_deduplicated():
    """防回归：同一时刻的窗口被计两次会拉偏中位数与覆盖率。"""
    c = StressRegulationCollector("s1")
    assert c.push_batch(HRBatch("s1", samples=(HRSample(1000, 70.0),))) == 1
    assert c.push_batch(HRBatch("s1", samples=(HRSample(1000, 200.0),))) == 0
    assert c.n_samples == 1
    # 先到的那条被保留（后到的重传不改写已成立的事实）
    assert c.samples()[0].bpm == 70.0


def test_foreign_session_batch_is_rejected():
    """★ 红线：串场批次必须拒收——把别人的心率算进本场报告是最严重的数据污染。"""
    c = StressRegulationCollector("s1")
    assert c.push_batch(HRBatch("s-other", samples=(HRSample(1000, 70.0),))) == 0
    assert c.n_samples == 0
    assert c.n_batches == 0  # 拒收的批次不应计入统计


# ---------- 基线 ----------


def test_baseline_uses_only_the_last_30_seconds():
    """防回归：详设 §4.2 明确"丢弃前 30s 让信号收敛"，基线只取最后 30s 的中位数。

    构造：前 30s 心率 100（未收敛），后 30s 心率 70。基线必须是 70，不是 85。
    """
    c = StressRegulationCollector("s1")
    first_half = [HRSample(t, 100.0, 0.9) for t in range(0, 30_000, SAMPLE_STEP_MS)]
    second_half = [HRSample(t, 70.0, 0.9) for t in range(30_000, BASELINE_REST_MS + 1, SAMPLE_STEP_MS)]
    c.push_batch(HRBatch("s1", samples=tuple(first_half + second_half)))

    assert c.baseline_from_rest() == pytest.approx(70.0)


def test_low_snr_rest_period_disables_the_whole_module():
    """★ 红线：静息期 SNR 中位数 < 0.4 → 整场关闭，并给出**可读原因**。

    环境不支持时的正确行为是"不做这项评估"（权重归零、归还其他维度），
    而不是硬算一个数字出来——后者会变成对拍摄条件差的人的隐性惩罚。
    """
    c = _collector(8)
    low = StressRegulationCollector("s2")
    low.push_batch(
        HRBatch("s2", samples=tuple(_rest_samples(snr=0.2)))
    )

    assert low.baseline_from_rest() is None
    assert low.baseline_gate_reason is not None
    assert f"{BASELINE_MIN_SNR:.2f}" in low.baseline_gate_reason
    assert "环境不支持" in low.baseline_gate_reason

    result = low.result()
    assert result.available is False
    assert result.score is None
    assert result.weight_applied == 0.0
    assert "环境不支持" in result.reason


def test_all_rest_windows_rejected_gives_explicit_reason():
    """防回归：静息期全弃权（bpm 全 None）与"SNR 低"是两种成因，原因文案要能区分。"""
    c = StressRegulationCollector("s1")
    c.push_batch(HRBatch("s1", samples=tuple(HRSample(t, None, 0.9) for t in range(0, 60_001, 2000))))

    assert c.baseline_from_rest() is None
    assert "均被弃权" in c.baseline_gate_reason


def test_baseline_can_be_set_directly():
    """防回归：允许调用方直接给基线（离线回放/人工校准场景）。"""
    c = StressRegulationCollector("s1")
    c.set_baseline(72.5)
    assert c.baseline_bpm == pytest.approx(72.5)
    c.set_baseline(None)
    assert c.baseline_bpm is None


# ---------- 单事件指标 ----------


def test_single_event_metrics_match_the_documented_formulas():
    """防回归：pre = [t_q−3s, t_q] 中位数、peak = [t_a, t_a+20s] 最大值、T50 ≈ 真值。"""
    c = _collector(1)
    ev = c.events[0]

    assert ev.pre_bpm == pytest.approx(BASE_HR)
    assert ev.peak_bpm == pytest.approx(BASE_HR + _delta_of(0))
    assert ev.delta == pytest.approx(_delta_of(0))
    assert ev.valid is True
    assert ev.censored is False
    # 合成信号的真值 T50 = 16s；2s 采样 + 对数线性拟合，容差给 3s
    assert ev.t50_s == pytest.approx(T50_TRUE_S, abs=3.0)


def test_unexcited_event_is_marked_invalid_not_guessed():
    """防回归：ΔHR < 2 BPM 属"未激发"，valid=False 排除出恢复统计——不是补一个值。"""
    a = _anchor(0)
    c = StressRegulationCollector("s1")
    # 作答期心率几乎不变（只抬高 1 BPM）
    flat = [HRSample(t, BASE_HR, 0.8) for t in range(a.t_q_ms - 3_000, a.t_q_ms + 1, 1_000)]
    flat += [
        HRSample(t, BASE_HR + 1.0, 0.8)
        for t in range(a.t_a_ms, a.t_e_ms + 1, SAMPLE_STEP_MS)
    ]
    c.push_batch(HRBatch("s1", samples=tuple(flat)))
    c.mark_event(a)

    ev = c.events[0]
    assert ev.delta == pytest.approx(1.0)
    assert ev.valid is False
    assert ev.t50_s is None


def test_rejected_windows_are_excluded_from_pre_and_peak():
    """★ 红线：弃权窗口（bpm=None）不得参与 pre/peak —— 不能当 0 或忽略标记直接用。

    这里把 pre 窗与作答期全部置为弃权，事件必须 valid=False，
    绝不允许因为"窗口数够"就凭空产出一个心率。
    """
    a = _anchor(0)
    c = StressRegulationCollector("s1")
    samples = [HRSample(t, None, 0.1) for t in range(a.t_q_ms - 3_000, a.t_e_ms + 1, 1_000)]
    c.push_batch(HRBatch("s1", samples=tuple(samples), rejected_windows=len(samples)))
    c.mark_event(a)

    ev = c.events[0]
    assert ev.pre_bpm is None
    assert ev.peak_bpm is None
    assert ev.valid is False


#: 第 1 题的下发时刻（紧跟第 0 题作答结束，间隔 3s）。
NEXT_Q_MS = T_Q0 + T_E_OFFSET_MS + 3_000


def _contaminated_tail() -> list[HRSample]:
    """下一题**已经开始**之后的心率：重新抬高（新的应激）。

    这些窗口落在第 0 题的 `t_e + 45s` 之内。若不按"下一题 t_q"截断，它们会被
    算进第 0 题的回落曲线，把 `T50` 拉大——表现为"这个人恢复得很慢"，而实际上
    是下一题又把他激起来了。这是**跨题借数据**，必须由截断挡住。
    """
    return [
        HRSample(t, BASE_HR + 15.0 + (t - NEXT_Q_MS) / 5000.0, SNR)
        for t in range(NEXT_Q_MS + 2_000, T_Q0 + T_E_OFFSET_MS + 45_000, SAMPLE_STEP_MS)
    ]


def test_recovery_window_is_truncated_by_next_question():
    """★ 防回归：恢复窗必须被下一题 `t_q` 截断（详设 §4.1），不得跨题借数据。

    对照组设计（同一批数据、只差一个锚点）：
    * 登记了下一题的 `t_q=140s` → 恢复窗截到 140s，尾部污染被排除，T50 ≈ 真值 16s；
    * 没有下一题锚点 → 恢复窗拉到 `t_e+45s`，尾部污染进入拟合，T50 明显偏大。
    两条一起看，才能证明"截断这件事真的在起作用"，而不是碰巧结果一样。
    """
    a = _anchor(0)
    samples = tuple(_event_samples(0) + _contaminated_tail())

    truncated = StressRegulationCollector("s1")
    truncated.push_batch(HRBatch("s1", samples=samples))
    truncated.mark_event(a)
    truncated.mark_event(EventAnchor(turn_index=1, t_q_ms=NEXT_Q_MS))  # 下一题下发

    uncontaminated = StressRegulationCollector("s2")
    uncontaminated.push_batch(HRBatch("s2", samples=samples))
    uncontaminated.mark_event(a)  # 故意不登记下一题 → 恢复窗不被截断

    ev_truncated = truncated.events[0]
    ev_contaminated = uncontaminated.events[0]

    assert ev_truncated.valid is True
    assert ev_truncated.t50_s == pytest.approx(T50_TRUE_S, abs=3.0)
    # 对照：污染进入拟合后 T50 必须明显变大（含右删失被钉到 45s 的情形）
    assert ev_contaminated.t50_s is None or ev_contaminated.t50_s > 25.0, (
        f"尾部污染没有影响拟合，说明这个对照没有区分力：{ev_contaminated.t50_s}"
    )


def test_insufficient_fit_points_yields_invalid_event():
    """防回归：拟合点太少时弃权（`metrics.MIN_FIT_POINTS` 的存在理由）。"""
    a = _anchor(0)
    c = StressRegulationCollector("s1")
    # 极稀疏采样：只有 3 个恢复点
    samples = [HRSample(t, BASE_HR, 0.8) for t in (97_000, 100_000)]
    samples += [
        HRSample(a.t_a_ms, BASE_HR + 20.0, 0.8),
        HRSample(a.t_a_ms + 6_000, BASE_HR + 15.0, 0.8),
        HRSample(a.t_a_ms + 12_000, BASE_HR + 11.0, 0.8),
    ]
    c.push_batch(HRBatch("s1", samples=tuple(samples)))
    c.mark_event(a)

    assert c.events[0].valid is False


def test_anchor_without_answer_marks_is_not_emitted():
    """防回归：只有 t_q 的锚点不足以切出指标，不应产出半个事件。"""
    c = StressRegulationCollector("s1")
    c.mark_event(EventAnchor(turn_index=0, t_q_ms=100_000))

    assert c.n_anchors == 1
    assert c.n_events == 0


# ---------- 质量聚合 ----------


def test_quality_aggregates_are_computed_over_all_windows():
    """防回归：SNR 取中位数、弃权率按窗口数、一致率取批次均值。"""
    c = StressRegulationCollector("s1")
    c.push_batch(
        HRBatch("s1", samples=(HRSample(0, 70.0, 0.9), HRSample(2000, None, 0.3), HRSample(4000, 75.0, 0.6)),
                algo_agreement=0.8)
    )
    c.push_batch(HRBatch("s1", samples=(HRSample(6000, 72.0, 0.7),), algo_agreement=1.0))

    assert c.median_snr() == pytest.approx(0.65)  # median(0.9,0.3,0.6,0.7) 取中间两数均值
    assert c.reject_ratio() == pytest.approx(0.25)  # 1/4
    assert c.algo_agreement() == pytest.approx(0.9)  # mean(0.8,1.0)


def test_empty_collector_reports_pessimistic_quality():
    """防回归：没有任何窗口时，SNR=0、弃权率=1 —— 而不是 0 弃权（那会显得信号很好）。"""
    c = StressRegulationCollector("s1")
    assert c.median_snr() == 0.0
    assert c.reject_ratio() == 1.0
    assert c.algo_agreement() == 0.0


def test_out_of_range_algo_agreement_is_ignored():
    """防回归：一致率越界（>1 或 <0）属脏数据，不能污染可靠性 R。"""
    c = StressRegulationCollector("s1")
    c.push_batch(HRBatch("s1", samples=(HRSample(0, 70.0, 0.8),), algo_agreement=1.7))
    assert c.algo_agreement() == 0.0


# ---------- 结果 ----------


def test_full_session_produces_an_available_score_with_full_weight():
    """★ 主线：8 个高质量事件 → 生理维度**计入**，R ≥ 0.75 拿到满额权重 8%。"""
    c = _collector(8)
    r = c.result()

    assert r.available is True
    assert r.reason is None
    assert r.score is not None and 0.0 <= r.score <= 100.0
    assert r.n_events == 8
    assert r.n_valid == 8
    assert r.baseline_bpm == pytest.approx(BASE_HR)
    assert r.reliability >= 0.75
    assert gate(r.reliability) == 1.0
    assert r.weight_applied == pytest.approx(W_MAX)
    assert set(r.components) == {"S_recovery", "S_habituation", "S_consistency"}


def test_habituation_and_recovery_components_are_present_and_bounded():
    """防回归：三个分量都必须在 0–100 之间（分位数映射把表外值夹到端点）。"""
    r = _collector(8).result()
    for name, value in r.components.items():
        assert value is not None, name
        assert 0.0 <= value <= 100.0, name


def test_fewer_than_eight_valid_events_is_unavailable():
    """★ 红线：有效事件 < 8 时整项不计入（纪律二：禁止对单次事件下判断）。"""
    r = _collector(5).result()

    assert r.available is False
    assert r.score is None
    assert r.weight_applied == 0.0
    assert "有效事件" in r.reason
    assert "不计入" in r.reason


def test_unauthorized_session_returns_unavailable_and_frees_weight():
    """★ 红线：未取得单独授权 → 不计入，且权重**全额归还**其他维度。

    注意是"归还"而不是"总分被压低"：这正是"关闭生理模块不会系统性降低总分"
    这条可写成自动化断言的承诺。
    """
    c = _collector(8, authorized=False)
    r = c.result()

    assert r.available is False
    assert r.score is None
    assert r.weight_applied == 0.0
    assert "授权" in r.reason
    # 归还明细：六维各自分到 W_MAX/6
    assert len(r.redistributed_to) >= 6
    assert sum(r.redistributed_to.values()) == pytest.approx(1.0 - 0.0)


def test_unauthorized_check_precedes_everything_else():
    """防回归：授权检查必须是第一道闸——没授权就不该碰任何数据。"""
    c = StressRegulationCollector("s1", authorized=False)
    r = c.result()

    assert r.available is False
    assert "授权" in r.reason
    assert r.n_events == 0
    assert r.reliability == 0.0


def test_result_without_anchors_is_unavailable():
    """防回归：一场没有任何事件锚点的会话不能出结果（原因要与"事件不足"区分）。"""
    c = StressRegulationCollector("s1")
    c.push_batch(HRBatch("s1", samples=tuple(_rest_samples())))
    c.baseline_from_rest()

    r = c.result()
    assert r.available is False
    assert "事件锚点" in r.reason


def test_missing_baseline_is_unavailable():
    """防回归：没有静息基线就没有"反应幅度"可言，整项不计入。"""
    c = StressRegulationCollector("s1")
    for i in range(8):
        c.push_batch(HRBatch("s1", samples=tuple(_event_samples(i))))
        c.mark_event(_anchor(i))

    r = c.result()
    assert r.available is False
    assert "基线" in r.reason


def test_arrhythmia_heuristic_closes_the_module():
    """防回归：检出心律不规则 → 自动关闭本项（医学排除，详设 §6.0）。

    只是工程启发式，不构成任何健康判断——文案里也必须这么说。
    """
    c = _collector(8)
    c.set_ibis([0.8, 0.4, 1.2, 0.5, 1.1, 0.35, 1.3, 0.6] * 4)

    r = c.result()
    assert r.available is False
    assert "心律不规则" in r.reason
    assert "非医学诊断" in r.reason


def test_expected_events_can_be_pinned_to_shorten_coverage():
    """防回归：`n_expected` 可显式指定；期望 20 题只完成 8 题时覆盖率下降、R 降低。

    这防的是"提前结束的面试反而拿到更高的可靠性分"。
    """
    short = _collector(8).result()
    pinned = _collector(8, expected_events=20).result()

    assert pinned.available is True
    assert pinned.reliability < short.reliability


def test_read_only_views_track_inputs():
    """防回归：只读视图要与输入一致（截图/报告要靠它做口径说明）。"""
    c = _collector(8)
    assert c.n_samples == len(c.samples())
    assert c.n_batches == 9  # 1 个静息批 + 8 个事件批
    assert c.n_events == 8
    assert len(c.events) == 8
    assert c.env is not None and c.env.fps == pytest.approx(30.0)
    assert c.baseline_gate_reason is None
