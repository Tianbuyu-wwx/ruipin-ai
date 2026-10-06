"""题库治理与偏差审计测试（`ruipin.safety.bank`）。

重点钉住：
- `revise` 后新版本**必须**回到 `PENDING_REVIEW`，旧版本转 `RETIRED`（治理核心）；
- 非法审校流转抛 `InvalidReviewTransition`，`DRAFT→APPROVED` 必须被拒；
- `selectable` 只出 `APPROVED` 的当前版本；
- 生理/健康题的整类拒绝；
- `BiasAudit` 样本不足**不得**用 0.0 冒充通过率。
"""

from __future__ import annotations

import pickle

import pytest

from ruipin.domain.errors import RuipinError, Unavailable
from ruipin.safety.bank import (
    MIN_GROUP_SAMPLE,
    BiasAudit,
    InvalidReviewTransition,
    QuestionBank,
    QuestionBankItem,
    ReviewStatus,
)

RS = ReviewStatus


class _Clock:
    """确定性时钟：每次调用 +1，便于断言 created_at 不是硬编码的常量。"""

    def __init__(self, start: float = 1000.0) -> None:
        self.t = start

    def __call__(self) -> float:
        self.t += 1.0
        return self.t


def make_item(
    qid: str = "q1",
    *,
    text: str = "请介绍一个你主导过的项目。",
    version: int = 1,
    status: ReviewStatus = RS.DRAFT,
    job_family: str = "backend",
    difficulty: float = 0.5,
    tags: frozenset[str] = frozenset({"project"}),
    created_at: int = 1,
    author: str = "alice",
) -> QuestionBankItem:
    return QuestionBankItem(
        question_id=qid,
        text=text,
        version=version,
        author=author,
        review_status=status,
        job_family=job_family,
        difficulty=difficulty,
        tags=tags,
        created_at=created_at,
    )


@pytest.fixture
def bank() -> QuestionBank:
    return QuestionBank(_Clock())


# ---------- QuestionBankItem 值校验 ----------


@pytest.mark.parametrize("bad", [-0.01, 1.01, -5.0, 2.0])
def test_difficulty_out_of_range_rejected(bad):
    with pytest.raises(ValueError, match="difficulty"):
        make_item(difficulty=bad)


@pytest.mark.parametrize("ok", [0.0, 1.0, 0.5])
def test_difficulty_boundary_accepted(ok):
    assert make_item(difficulty=ok).difficulty == ok


def test_difficulty_non_numeric_rejected():
    with pytest.raises(TypeError, match="difficulty"):
        make_item(difficulty="0.5")  # type: ignore[arg-type]


def test_version_must_be_positive():
    with pytest.raises(ValueError, match="version"):
        make_item(version=0)


@pytest.mark.parametrize("qid", ["", "   "])
def test_empty_question_id_rejected(qid):
    with pytest.raises(ValueError, match="question_id"):
        make_item(qid=qid)


@pytest.mark.parametrize("text", ["", "  \n "])
def test_empty_text_rejected(text):
    with pytest.raises(ValueError, match="正文"):
        make_item(text=text)


def test_item_is_frozen():
    it = make_item()
    with pytest.raises(Exception):
        it.text = "改一下"  # type: ignore[misc]


# ---------- add ----------


def test_add_then_get_and_len(bank):
    bank.add(make_item("q1"))
    assert len(bank) == 1
    assert bank.get("q1").question_id == "q1"  # type: ignore[union-attr]
    assert bank.get("nope") is None


def test_add_duplicate_id_rejected(bank):
    bank.add(make_item("q1"))
    with pytest.raises(ValueError, match="已存在"):
        bank.add(make_item("q1"))


def test_add_non_item_rejected(bank):
    with pytest.raises(TypeError, match="QuestionBankItem"):
        bank.add(object())  # type: ignore[arg-type]


def test_add_accepts_draft_and_pending_review(bank):
    bank.add(make_item("q1", status=RS.DRAFT))
    bank.add(make_item("q2", status=RS.PENDING_REVIEW))
    assert bank.get("q1").review_status is RS.DRAFT  # type: ignore[union-attr]
    assert bank.get("q2").review_status is RS.PENDING_REVIEW  # type: ignore[union-attr]


@pytest.mark.parametrize("status", [RS.APPROVED, RS.REJECTED, RS.RETIRED])
def test_add_rejects_already_reviewed_status(bank, status):
    """结构保证：add() 不能把已审题直接塞进库，否则"APPROVED 必来自 PENDING_REVIEW"只是口头纪律。"""
    with pytest.raises(ValueError) as ei:
        bank.add(make_item("q1", status=status))
    msg = str(ei.value)
    assert "transition" in msg and "seed_approved" in msg
    assert status.value in msg
    assert bank.get("q1") is None  # 拒绝即不落库


def test_add_rejection_message_mentions_reason_requirement(bank):
    with pytest.raises(ValueError, match="reason"):
        bank.add(make_item("q1", status=RS.APPROVED))


def test_bank_rejects_non_callable_clock():
    with pytest.raises(TypeError, match="clock"):
        QuestionBank("not-callable")  # type: ignore[arg-type]


# ---------- seed_approved：历史迁移的唯一入口 ----------


def test_seed_approved_makes_question_selectable(bank):
    """迁移路径必须真的打通：seed 后能被 selectable() 取到，而不是"存进去取不出来"。"""
    n = bank.seed_approved(
        [make_item("q1", status=RS.APPROVED)], actor="migrator", reason="迁移单 MIG-42"
    )
    assert n == 1
    assert [it.question_id for it in bank.selectable()] == ["q1"]


def test_seed_approved_records_actor_and_reason_for_audit(bank):
    bank.seed_approved(
        [make_item("q1", status=RS.APPROVED, tags=frozenset({"project"}))],
        actor="migrator",
        reason="迁移单 MIG-42",
    )
    tags = bank.get("q1").tags  # type: ignore[union-attr]
    assert "seeded:by=migrator" in tags
    assert "seeded:reason=迁移单 MIG-42" in tags
    assert "project" in tags  # 原有业务标签不被覆盖


def test_seed_approved_trace_survives_revision(bank):
    """留痕必须随版本继承——否则改一次版就再也看不出这题是被灌进来的。"""
    bank.seed_approved([make_item("q1", status=RS.APPROVED)], actor="m", reason="MIG-1")
    new = bank.revise("q1", text="改写", author="bob")
    assert "seeded:by=m" in new.tags and "seeded:reason=MIG-1" in new.tags


def test_seed_approved_returns_count_and_handles_empty(bank):
    assert bank.seed_approved([], actor="m", reason="MIG-1") == 0
    n = bank.seed_approved(
        [
            make_item("q1", status=RS.APPROVED),
            make_item("q2", status=RS.APPROVED),
        ],
        actor="m",
        reason="MIG-1",
    )
    assert n == 2
    assert len(bank) == 2


@pytest.mark.parametrize("bad_reason", ["", "   ", "\n\t"])
def test_seed_approved_rejects_blank_reason(bank, bad_reason):
    """迁移必须有据可查，不允许匿名灌数据。"""
    with pytest.raises(ValueError, match="reason"):
        bank.seed_approved([make_item("q1", status=RS.APPROVED)], actor="m", reason=bad_reason)


@pytest.mark.parametrize("bad_actor", ["", "   "])
def test_seed_approved_rejects_blank_actor(bank, bad_actor):
    with pytest.raises(ValueError, match="actor"):
        bank.seed_approved([make_item("q1", status=RS.APPROVED)], actor=bad_actor, reason="MIG-1")


def test_seed_approved_does_not_bypass_duplicate_id_check(bank):
    bank.add(make_item("q1", status=RS.DRAFT))
    with pytest.raises(ValueError, match="已存在"):
        bank.seed_approved([make_item("q1", status=RS.APPROVED)], actor="m", reason="MIG-1")


def test_seed_approved_does_not_bypass_difficulty_validation(bank):
    """构造期偷渡一个越界难度（绕过 __post_init__），seed 时因 replace 重跑校验必须被拦。"""
    rogue = object.__new__(QuestionBankItem)
    object.__setattr__(rogue, "question_id", "q1")
    object.__setattr__(rogue, "text", "越界难度的题")
    object.__setattr__(rogue, "version", 1)
    object.__setattr__(rogue, "author", "a")
    object.__setattr__(rogue, "review_status", RS.APPROVED)
    object.__setattr__(rogue, "job_family", "backend")
    object.__setattr__(rogue, "difficulty", 2.0)  # 越界
    object.__setattr__(rogue, "tags", frozenset())
    object.__setattr__(rogue, "created_at", 1)
    with pytest.raises(ValueError, match="difficulty"):
        bank.seed_approved([rogue], actor="m", reason="MIG-1")


def test_seed_approved_rejects_non_approved_item(bank):
    """seed_approved 只用于迁移已审题；借它塞草稿等于换了名字的 add()。"""
    with pytest.raises(ValueError, match="只迁移已审"):
        bank.seed_approved([make_item("q1", status=RS.DRAFT)], actor="m", reason="MIG-1")


def test_seed_approved_rejects_non_item(bank):
    with pytest.raises(TypeError, match="QuestionBankItem"):
        bank.seed_approved([object()], actor="m", reason="MIG-1")  # type: ignore[list-item]


def test_seed_approved_failure_leaves_bank_unchanged_on_duplicate(bank):
    """批量迁移里出现的重复 id 应报错，且此前条目的状态必须一致可解释。"""
    bank.seed_approved([make_item("q1", status=RS.APPROVED)], actor="m", reason="MIG-1")
    with pytest.raises(ValueError, match="已存在"):
        bank.seed_approved(
            [make_item("q2", status=RS.APPROVED), make_item("q1", status=RS.APPROVED)],
            actor="m",
            reason="MIG-2",
        )


# ---------- revise：治理核心 ----------


def test_revise_returns_pending_review_and_retires_old(bank):
    bank.add(make_item("q1", status=RS.PENDING_REVIEW, version=1))
    bank.transition("q1", RS.APPROVED, "reviewer")
    new = bank.revise("q1", text="改写后的题目", author="bob", difficulty=0.7)

    assert new.version == 2
    assert new.review_status is RS.PENDING_REVIEW  # 改过内容 → 必须重审
    assert new.text == "改写后的题目"
    assert new.difficulty == 0.7
    assert new.author == "bob"

    hist = bank.history("q1")
    assert [h.version for h in hist] == [1, 2]  # 升序
    assert hist[0].review_status is RS.RETIRED  # 旧版本归档
    assert bank.get("q1").version == 2  # type: ignore[union-attr]


def test_revise_keeps_untouched_fields(bank):
    bank.add(make_item("q1", job_family="data", tags=frozenset({"sql"}), difficulty=0.3))
    new = bank.revise("q1", author="bob")
    assert new.text == "请介绍一个你主导过的项目。"
    assert new.difficulty == 0.3
    assert new.job_family == "data"
    assert new.tags == frozenset({"sql"})


def test_revise_stamps_created_at_from_injected_clock(bank):
    bank.add(make_item("q1"))
    a = bank.revise("q1", text="v2", author="bob")
    b = bank.revise("q1", text="v3", author="bob")
    assert b.created_at > a.created_at  # 时钟注入生效，不是硬编码常量


def test_revise_removes_from_selectable_until_reapproved(bank):
    bank.add(make_item("q1", status=RS.PENDING_REVIEW))
    bank.transition("q1", RS.APPROVED, "reviewer")
    assert len(bank.selectable()) == 1
    bank.revise("q1", text="改写", author="bob")
    assert bank.selectable() == ()  # 重审前不得再面向候选人
    bank.transition("q1", RS.APPROVED, "carol")
    assert len(bank.selectable()) == 1


def test_revise_unknown_id_rejected(bank):
    with pytest.raises(ValueError, match="不存在"):
        bank.revise("nope", text="x", author="bob")


def test_revise_invalid_difficulty_rejected(bank):
    bank.add(make_item("q1"))
    with pytest.raises(ValueError, match="difficulty"):
        bank.revise("q1", difficulty=1.5, author="bob")


def test_history_unknown_id_rejected(bank):
    with pytest.raises(ValueError, match="不存在"):
        bank.history("nope")


# ---------- transition ----------


def test_transition_legal_path_draft_to_approved(bank):
    bank.add(make_item("q1", status=RS.DRAFT))
    bank.transition("q1", RS.PENDING_REVIEW, "alice")
    approved = bank.transition("q1", RS.APPROVED, "reviewer")
    assert approved.review_status is RS.APPROVED
    assert bank.history("q1")[0].review_status is RS.APPROVED


def test_transition_pending_to_rejected_then_back(bank):
    bank.add(make_item("q1", status=RS.PENDING_REVIEW))
    bank.transition("q1", RS.REJECTED, "reviewer")
    bank.transition("q1", RS.PENDING_REVIEW, "author")
    assert bank.get("q1").review_status is RS.PENDING_REVIEW  # type: ignore[union-attr]


def test_transition_approved_to_retired(bank):
    bank.add(make_item("q1", status=RS.PENDING_REVIEW))
    bank.transition("q1", RS.APPROVED, "reviewer")
    bank.transition("q1", RS.RETIRED, "ops")
    assert bank.get("q1").review_status is RS.RETIRED  # type: ignore[union-attr]


def test_transition_draft_to_approved_is_rejected(bank):
    """纪律：APPROVED 只能来自 PENDING_REVIEW，不允许 DRAFT→APPROVED 捷径。"""
    bank.add(make_item("q1", status=RS.DRAFT))
    with pytest.raises(InvalidReviewTransition) as ei:
        bank.transition("q1", RS.APPROVED, "sneaky")
    assert "draft" in str(ei.value) and "approved" in str(ei.value)


def test_transition_approved_to_pending_is_rejected(bank):
    bank.add(make_item("q1", status=RS.PENDING_REVIEW))
    bank.transition("q1", RS.APPROVED, "reviewer")
    with pytest.raises(InvalidReviewTransition):
        bank.transition("q1", RS.PENDING_REVIEW, "x")


def test_transition_from_retired_is_terminal(bank):
    bank.add(make_item("q1", status=RS.PENDING_REVIEW))
    bank.transition("q1", RS.APPROVED, "reviewer")
    bank.transition("q1", RS.RETIRED, "ops")
    with pytest.raises(InvalidReviewTransition):
        bank.transition("q1", RS.APPROVED, "x")


def test_transition_unknown_id_rejected(bank):
    with pytest.raises(ValueError, match="不存在"):
        bank.transition("nope", RS.APPROVED, "x")


def test_transition_unknown_status_rejected(bank):
    bank.add(make_item("q1", status=RS.DRAFT))
    with pytest.raises(ValueError, match="未知的审校状态"):
        bank.transition("q1", "published", "x")  # type: ignore[arg-type]


def test_invalid_review_transition_is_ruipin_error_and_picklable():
    err = InvalidReviewTransition("q1", RS.DRAFT, RS.APPROVED, "alice")
    assert isinstance(err, RuipinError)
    assert err.question_id == "q1" and err.actor == "alice"
    restored = pickle.loads(pickle.dumps(err))
    assert isinstance(restored, InvalidReviewTransition)
    assert restored.from_status is RS.DRAFT
    assert restored.to_status is RS.APPROVED
    assert str(restored) == str(err)


# ---------- selectable ----------


def test_selectable_only_returns_approved(bank):
    bank.add(make_item("q2", status=RS.DRAFT))
    bank.add(make_item("q3", status=RS.PENDING_REVIEW))
    bank.add(make_item("q4", status=RS.PENDING_REVIEW))
    bank.transition("q4", RS.REJECTED, "reviewer")
    bank.add(make_item("q5", status=RS.PENDING_REVIEW))
    bank.transition("q5", RS.APPROVED, "reviewer")
    bank.transition("q5", RS.RETIRED, "ops")
    bank.seed_approved(
        [make_item("q1", status=RS.APPROVED)], actor="migrator", reason="迁移单 MIG-42"
    )
    ids = [it.question_id for it in bank.selectable()]
    assert ids == ["q1"]


def test_selectable_sorted_for_determinism(bank):
    for qid in ("qc", "qa", "qb"):
        bank.add(make_item(qid, status=RS.PENDING_REVIEW))
        bank.transition(qid, RS.APPROVED, "reviewer")
    assert [it.question_id for it in bank.selectable()] == ["qa", "qb", "qc"]


def test_selectable_filters_job_family_and_difficulty(bank):
    bank.add(make_item("q1", status=RS.PENDING_REVIEW, job_family="backend", difficulty=0.3))
    bank.transition("q1", RS.APPROVED, "reviewer")
    bank.add(make_item("q2", status=RS.PENDING_REVIEW, job_family="data", difficulty=0.6))
    bank.transition("q2", RS.APPROVED, "reviewer")
    assert [it.question_id for it in bank.selectable(job_family="data")] == ["q2"]
    assert [it.question_id for it in bank.selectable(max_difficulty=0.4)] == ["q1"]
    assert bank.selectable(job_family="nope") == ()


# ---------- 生理/健康题检测 ----------


@pytest.mark.parametrize(
    "text",
    [
        "你是否有病史？",
        "你得过什么病",
        "请介绍你的健康状况",
        "说说你最近一次生病的情况",
        "你有高血压吗",
        "你的身体怎么样",
        "你做过什么手术",
    ],
)
def test_physio_question_detected(text):
    bank = QuestionBank(_Clock())
    bank.add(make_item("bad", text=text))
    with pytest.raises(ValueError, match="bad"):
        bank.assert_no_physio_question()


@pytest.mark.parametrize(
    "text",
    [
        "请介绍一个你主导过的项目。",
        "如何设计健康数据的存储方案？",  # how-to 技术题，非问候选人本人
        "你如何保证用户健康数据的隐私？",
        "谈谈你对高并发系统的理解。",
        "你在这段经历里最大的收获是什么？",
    ],
)
def test_non_physio_question_not_flagged(text):
    bank = QuestionBank(_Clock())
    bank.add(make_item("ok1", text=text))
    bank.assert_no_physio_question()  # 不应抛


def test_physio_check_lists_all_hits_sorted():
    bank = QuestionBank(_Clock())
    bank.add(make_item("q2", text="你是否有病史？"))
    bank.add(make_item("q1", text="你有高血压吗"))
    bank.add(make_item("q3", text="请介绍你的项目"))
    with pytest.raises(ValueError) as ei:
        bank.assert_no_physio_question()
    msg = str(ei.value)
    assert "q1" in msg and "q2" in msg and "q3" not in msg
    assert msg.index("q1") < msg.index("q2")  # 排序稳定


# ---------- BiasAudit ----------


def _recs(group: str, n_true: int, n_false: int) -> list[tuple[str, bool]]:
    return [(group, True)] * n_true + [(group, False)] * n_false


def test_pass_rate_for_sufficient_group():
    audit = BiasAudit()
    rates = audit.pass_rate_by_group(_recs("A", 7, 3))
    assert rates["A"] == pytest.approx(0.7)


def test_insufficient_group_rate_is_none_not_zero():
    """纪律：样本不足不得用 0.0 冒充通过率。"""
    audit = BiasAudit()
    rates = audit.pass_rate_by_group(_recs("small", 0, 3))
    assert rates["small"] is None
    stat = audit.stats[0]
    assert stat.group == "small"
    assert stat.n == 3
    assert stat.passed == 0
    assert stat.sufficient is False
    assert stat.pass_rate is None


def test_min_group_sample_boundary():
    audit = BiasAudit()
    rates = audit.pass_rate_by_group(_recs("edge", MIN_GROUP_SAMPLE, 0))
    assert rates["edge"] == 1.0  # 恰好达到最小样本量 → 可给通过率


def test_stats_property_sorted_and_sufficient_flag():
    audit = BiasAudit()
    audit.pass_rate_by_group(_recs("B", 6, 4) + _recs("A", 8, 2) + _recs("C", 1, 1))
    assert [s.group for s in audit.stats] == ["A", "B", "C"]
    assert audit.stats[0].sufficient is True
    assert audit.stats[2].sufficient is False


def test_divergence_between_two_sufficient_groups():
    audit = BiasAudit()
    audit.pass_rate_by_group(_recs("A", 7, 3) + _recs("B", 4, 6))
    assert audit.divergence("A", "B") == pytest.approx(0.3)
    assert audit.divergence("B", "A") == pytest.approx(0.3)  # 对称


def test_divergence_with_insufficient_group_raises_not_zero():
    """差异"算不出来"时抛 Unavailable，不得返回 0.0（会被误读为"两组一致"）。"""
    audit = BiasAudit()
    audit.pass_rate_by_group(_recs("A", 7, 3) + _recs("small", 2, 0))
    with pytest.raises(Unavailable) as ei:
        audit.divergence("A", "small")
    assert ei.value.provider == "bias-audit"


def test_divergence_unknown_group_rejected():
    audit = BiasAudit()
    audit.pass_rate_by_group(_recs("A", 7, 3))
    with pytest.raises(ValueError, match="不存在"):
        audit.divergence("A", "ghost")


def test_is_biased_true_when_pair_exceeds_threshold():
    audit = BiasAudit()
    audit.pass_rate_by_group(_recs("A", 9, 1) + _recs("B", 3, 7))
    assert audit.is_biased(max_abs_diff=0.15) is True


def test_is_biased_false_when_all_pairs_within_threshold():
    audit = BiasAudit()
    audit.pass_rate_by_group(_recs("A", 7, 3) + _recs("B", 6, 4))
    assert audit.is_biased(max_abs_diff=0.15) is False


def test_is_biased_ignores_insufficient_groups():
    """样本不足的组不参与两两比较——否则小样本噪声会假报偏差。"""
    audit = BiasAudit()
    audit.pass_rate_by_group(_recs("A", 7, 3) + _recs("B", 6, 4) + _recs("small", 0, 1))
    assert audit.is_biased() is False


def test_is_biased_with_fewer_than_two_sufficient_groups_raises():
    """有效分组 < 2 时不得返回 False（那等于宣称"已确认无偏差"）。"""
    audit = BiasAudit()
    audit.pass_rate_by_group(_recs("A", 7, 3))
    with pytest.raises(Unavailable, match="有效分组"):
        audit.is_biased()


def test_is_biased_negative_threshold_rejected():
    audit = BiasAudit()
    audit.pass_rate_by_group(_recs("A", 7, 3) + _recs("B", 6, 4))
    with pytest.raises(ValueError, match="max_abs_diff"):
        audit.is_biased(max_abs_diff=-0.1)


def test_pass_rate_rejects_malformed_records():
    audit = BiasAudit()
    with pytest.raises(ValueError, match="二元组"):
        audit.pass_rate_by_group([("A",)] )  # type: ignore[list-item]
    with pytest.raises(ValueError, match="分组名"):
        audit.pass_rate_by_group([("   ", True)])
    with pytest.raises(TypeError, match="bool"):
        audit.pass_rate_by_group([("A", 1)])  # type: ignore[list-item]


def test_pass_rate_empty_records_returns_empty_mapping():
    audit = BiasAudit()
    assert audit.pass_rate_by_group([]) == {}
