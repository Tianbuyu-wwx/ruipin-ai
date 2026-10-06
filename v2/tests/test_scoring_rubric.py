"""rubric 规则评分器测试。

重点：确定性（同输入同输出）、六维齐全、值域、关键词与题型的区分度、
空作答必须抛 Unavailable（不得产出合成分数）。
"""

from __future__ import annotations

import asyncio

import pytest

from ruipin.domain.errors import Unavailable
from ruipin.ports import DIMENSIONS, EvalRequest, Evaluator
from ruipin.scoring.confidence import should_downgrade
from ruipin.scoring.rubric import RuleEvaluator

# 覆盖率 100%、含连接词/量化/收尾的标准作答
GOOD_ANSWER = (
    "Redis 持久化主要有 RDB 和 AOF 两种。首先 RDB 是内存快照，恢复快但可能丢失最后一次快照之后的数据；"
    "其次 AOF 记录每一条写命令，数据更安全但文件更大。例如我在订单服务里使用 AOF 每秒刷盘，"
    "重启恢复时间从 30 秒降到 5 秒。因此我倾向于混合开启。"
)
HIT_KEYWORDS = ("RDB", "AOF", "持久化")
MISS_KEYWORDS = ("Kafka", "ZooKeeper", "一致性")


def run(**kw) -> object:
    """同步跑一次 evaluate（避免依赖 pytest-asyncio 插件）。"""
    base = {
        "question": "请介绍 Redis 持久化机制",
        "answer": GOOD_ANSWER,
        "keywords": HIT_KEYWORDS,
        "has_audio": True,
    }
    base.update(kw)
    return asyncio.run(RuleEvaluator().evaluate(EvalRequest(**base)))


# ---------- 契约 ----------


def test_implements_evaluator_protocol():
    ev = RuleEvaluator()
    assert isinstance(ev, Evaluator)
    assert ev.name == "rubric"


def test_result_flags_are_real_not_fallback():
    r = run()
    assert r.provider == "rubric"
    assert r.degraded is False
    assert r.degrade_reason is None


# ---------- 确定性 ----------


def test_deterministic_same_input_same_output():
    a = run()
    b = run()
    assert a.dims == b.dims
    assert a.score == b.score
    assert a.confidence == b.confidence
    assert a.feedback == b.feedback


def test_deterministic_across_instances():
    a = asyncio.run(RuleEvaluator().evaluate(
        EvalRequest(question="q", answer=GOOD_ANSWER, keywords=HIT_KEYWORDS)))
    b = asyncio.run(RuleEvaluator().evaluate(
        EvalRequest(question="q", answer=GOOD_ANSWER, keywords=HIT_KEYWORDS)))
    assert a.dims == b.dims and a.score == b.score


# ---------- 维度与值域 ----------


def test_all_six_dimensions_present_and_in_range():
    r = run()
    assert tuple(r.dims.keys()) == DIMENSIONS
    for d in DIMENSIONS:
        assert 0.0 <= r.dims[d] <= 100.0
    assert 0.0 <= r.score <= 100.0
    assert 0.0 <= r.confidence <= 1.0


def test_golden_values_full_hit_technical():
    """Golden Master：标准作答 + 全命中 + technical 的具体数值。"""
    r = run()
    assert r.score == pytest.approx(86.923, abs=0.01)
    assert r.dims["technical"] == pytest.approx(93.563, abs=0.01)
    assert r.dims["communication"] == pytest.approx(74.573, abs=0.01)
    assert r.dims["completeness"] == pytest.approx(82.392, abs=0.01)
    assert r.dims["problem_solving"] == pytest.approx(86.303, abs=0.01)
    assert r.dims["teamwork"] == pytest.approx(63.723, abs=0.01)
    assert r.dims["leadership"] == pytest.approx(61.436, abs=0.01)


# ---------- 关键词区分度 ----------


def test_keyword_hit_beats_miss():
    hit = run(keywords=HIT_KEYWORDS)
    miss = run(keywords=MISS_KEYWORDS)
    assert hit.score == pytest.approx(86.923, abs=0.01)
    assert miss.score == pytest.approx(43.923, abs=0.01)
    # 技术维度对关键词最敏感，差距必须显著（远超浮点噪声）
    assert hit.dims["technical"] - miss.dims["technical"] == pytest.approx(63.0, abs=0.01)
    assert hit.score - miss.score > 40.0


def test_no_keywords_is_neutral_between_hit_and_miss():
    hit = run(keywords=HIT_KEYWORDS)
    miss = run(keywords=MISS_KEYWORDS)
    neutral = run(keywords=())
    assert miss.score < neutral.score < hit.score
    assert "中性" in neutral.feedback
    assert "本题未配置关键词" in neutral.feedback


def test_blank_keywords_are_ignored_not_counted_as_miss():
    with_blank = run(keywords=("RDB", "", "  ", "AOF"))
    exact = run(keywords=("RDB", "AOF"))
    assert with_blank.score == exact.score
    assert "关键词命中 2/2" in with_blank.feedback


def test_feedback_reports_hit_count():
    assert "关键词命中 3/3" in run(keywords=HIT_KEYWORDS).feedback
    assert "关键词命中 0/3" in run(keywords=MISS_KEYWORDS).feedback


# ---------- 题型区分度 ----------


@pytest.mark.parametrize(
    "qtype,expected_score",
    [
        ("technical", 86.923),
        ("project", 83.724),
        ("behavioral", 82.570),
        ("self_intro", 79.114),
    ],
)
def test_question_type_changes_score(qtype, expected_score):
    r = run(question_type=qtype)
    assert r.score == pytest.approx(expected_score, abs=0.01)


def test_weak_evidence_dim_follows_question_type():
    tech = run(question_type="technical")
    behav = run(question_type="behavioral")
    # 技术题问不出团队协作，行为题才是团队协作的主场
    assert tech.dims["technical"] > tech.dims["teamwork"]
    assert behav.dims["teamwork"] > behav.dims["technical"]
    assert behav.dims["teamwork"] > tech.dims["teamwork"]
    assert tech.dims["technical"] > behav.dims["technical"]


def test_unknown_question_type_falls_back_to_default():
    unknown = run(question_type="瞎写的题型")
    default = run(question_type="technical")
    assert unknown.dims == default.dims
    assert unknown.score == default.score


# ---------- 边界输入 ----------


@pytest.mark.parametrize("empty", ["", "   ", "\n\t "])
def test_empty_answer_raises_unavailable(empty):
    """空作答 = 没有证据 = 不产分数。绝不能返回一组中间分。"""
    with pytest.raises(Unavailable) as ei:
        asyncio.run(RuleEvaluator().evaluate(
            EvalRequest(question="q", answer=empty, keywords=HIT_KEYWORDS)))
    assert ei.value.provider == "rubric"


def test_transcript_is_used_when_answer_empty():
    r = asyncio.run(RuleEvaluator().evaluate(
        EvalRequest(question="q", answer="", transcript=GOOD_ANSWER, keywords=HIT_KEYWORDS)))
    assert 0.0 <= r.score <= 100.0
    assert r.provider == "rubric"


def test_very_short_answer_low_score_and_low_confidence():
    r = asyncio.run(RuleEvaluator().evaluate(
        EvalRequest(question="q", answer="不知道", keywords=("RDB",))))
    assert r.score == pytest.approx(15.484, abs=0.01)
    assert r.confidence == pytest.approx(0.125, abs=1e-9)
    assert should_downgrade(r.confidence) is True


def test_very_long_answer_saturates_and_stays_in_range():
    long_answer = GOOD_ANSWER * 30
    r = asyncio.run(RuleEvaluator().evaluate(
        EvalRequest(question="q", answer=long_answer, keywords=("RDB", "AOF"), has_audio=True)))
    assert len(long_answer) > 4000
    assert r.score == pytest.approx(87.864, abs=0.01)
    for d in DIMENSIONS:
        assert 0.0 <= r.dims[d] <= 100.0
    assert "充分" in r.feedback  # 长度档次封顶在"充分"，超长不再加分


def test_length_monotonic_score():
    short = asyncio.run(RuleEvaluator().evaluate(
        EvalRequest(question="q", answer=GOOD_ANSWER[:40], keywords=HIT_KEYWORDS)))
    full = asyncio.run(RuleEvaluator().evaluate(
        EvalRequest(question="q", answer=GOOD_ANSWER, keywords=HIT_KEYWORDS)))
    longer = asyncio.run(RuleEvaluator().evaluate(
        EvalRequest(question="q", answer=GOOD_ANSWER * 2, keywords=HIT_KEYWORDS)))
    assert short.score < full.score <= longer.score


# ---------- 置信度与分数正交 ----------


def test_audio_changes_confidence_only_not_score():
    with_audio = run(has_audio=True)
    without_audio = run(has_audio=False)
    assert with_audio.score == without_audio.score  # 分数只由文本证据决定
    assert with_audio.confidence > without_audio.confidence
    # 音频同时贡献两项：模态权重 0.20 + 证据源计数 +0.25/4=0.0625
    assert with_audio.confidence - without_audio.confidence == pytest.approx(0.2625, abs=1e-9)


def test_low_confidence_answer_is_marked_but_still_real():
    r = run(has_audio=False, has_video=False, transcript=None)
    assert r.degraded is False  # rubric 本身没降级，只是证据少
    assert r.confidence == pytest.approx(0.5011, abs=1e-4)
