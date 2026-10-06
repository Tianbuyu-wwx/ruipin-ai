"""受限追问模板 + 安全网关测试（`ruipin.safety.selector`）。

重点钉住：
- **不自由生成**：只能从受限模板渲染，产出被改写必须报错；
- **安全网关**：渲染后命中或分类器不可用 → 丢弃并退回题库题（并标记回退）；
- **不凭空编题**：题库无可选题时抛 `Unavailable`。
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from ruipin.domain.errors import Unavailable
from ruipin.safety.bank import QuestionBank, QuestionBankItem, ReviewStatus
from ruipin.safety.classifier import (
    AttributeCommentClassifier,
    LexiconSafetyClassifier,
    SafetyCategory,
    SafetyVerdict,
)
from ruipin.safety.selector import FollowupTemplate, TemplateSelector


def _ok_classifier() -> LexiconSafetyClassifier:
    """永不命中的分类器（词表里放一个不可能出现的词）。"""
    return LexiconSafetyClassifier({SafetyCategory.POLITICAL: frozenset({"zzzz-never"})})


class _AlwaysHit:
    name = "always-hit"

    def classify(self, text: str) -> SafetyVerdict:
        return SafetyVerdict(category=SafetyCategory.DISCRIMINATION, hit=True, detail="强制命中")


class _Unavailable:
    name = "down"

    def classify(self, text: str) -> SafetyVerdict:
        raise Unavailable("safety-classifier", "端点不可达")


def _bank(*items: QuestionBankItem) -> QuestionBank:
    """搭建题库。

    注意 `add()` 只接受未审题（DRAFT/PENDING_REVIEW），已审题必须走
    `seed_approved()`、已下架题必须由 transition() 流转产生——这里按状态分流，
    顺便证明"迁移路径真的可用"。
    """
    b = QuestionBank(lambda: 1000.0)
    approved = [it for it in items if it.review_status is ReviewStatus.APPROVED]
    for it in items:
        if it.review_status is ReviewStatus.APPROVED:
            continue
        if it.review_status is ReviewStatus.RETIRED:
            b.add(replace(it, review_status=ReviewStatus.PENDING_REVIEW))
            b.transition(it.question_id, ReviewStatus.APPROVED, "test")
            b.transition(it.question_id, ReviewStatus.RETIRED, "test")
        else:
            b.add(it)
    if approved:
        b.seed_approved(approved, actor="test-fixture", reason="测试迁移")
    return b


def _approved(qid: str, text: str) -> QuestionBankItem:
    return QuestionBankItem(
        question_id=qid,
        text=text,
        version=1,
        author="alice",
        review_status=ReviewStatus.APPROVED,
        job_family="backend",
        difficulty=0.5,
        tags=frozenset(),
        created_at=1,
    )


TEMPLATES = (
    FollowupTemplate(
        template_id="t.expand",
        pattern="你提到{keyword}，能再展开说一下当时的具体做法吗？",
        category="technical",
    ),
    FollowupTemplate(
        template_id="t.example",
        pattern="关于{keyword}，可以举一个具体例子吗？",
        category="behavioural",
    ),
)


# ---------- FollowupTemplate 校验 ----------


def test_template_requires_placeholder():
    with pytest.raises(ValueError, match="占位符"):
        FollowupTemplate(template_id="t", pattern="没有占位符的句子", category="technical")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"template_id": "", "pattern": "{keyword}好", "category": "x"},
        {"template_id": "t", "pattern": "   ", "category": "x"},
        {"template_id": "t", "pattern": "{keyword}好", "category": "  "},
    ],
)
def test_template_rejects_empty_fields(kwargs):
    with pytest.raises(ValueError):
        FollowupTemplate(**kwargs)


def test_template_rejects_stray_brace():
    with pytest.raises(ValueError, match="格式非法"):
        FollowupTemplate(template_id="t", pattern="{keyword}然后{未闭合", category="x")


# ---------- TemplateSelector 构造校验 ----------


def test_selector_requires_templates():
    with pytest.raises(ValueError, match="templates"):
        TemplateSelector((), _ok_classifier(), _bank())


def test_selector_rejects_duplicate_template_id():
    dup = (
        FollowupTemplate("t", "{keyword}一", "x"),
        FollowupTemplate("t", "{keyword}二", "x"),
    )
    with pytest.raises(ValueError, match="重复"):
        TemplateSelector(dup, _ok_classifier(), _bank())


# ---------- render ----------


def test_render_substitutes_keyword():
    sel = TemplateSelector(TEMPLATES, _ok_classifier(), _bank())
    out = sel.render("t.expand", "向量检索")
    assert out == "你提到向量检索，能再展开说一下当时的具体做法吗？"


def test_render_unknown_template_rejected():
    sel = TemplateSelector(TEMPLATES, _ok_classifier(), _bank())
    with pytest.raises(ValueError, match="不存在"):
        sel.render("t.ghost", "x")


def test_render_non_str_keyword_rejected():
    sel = TemplateSelector(TEMPLATES, _ok_classifier(), _bank())
    with pytest.raises(TypeError, match="str"):
        sel.render("t.expand", 123)  # type: ignore[arg-type]


# ---------- safe_render ----------


def test_safe_render_passes_when_clean():
    sel = TemplateSelector(TEMPLATES, _ok_classifier(), _bank(_approved("q1", "备用题")))
    text, fell_back = sel.safe_render("t.expand", "缓存")
    assert fell_back is False
    assert text == "你提到缓存，能再展开说一下当时的具体做法吗？"


def test_safe_render_falls_back_when_hit():
    sel = TemplateSelector(TEMPLATES, _AlwaysHit(), _bank(_approved("q1", "备用题：请介绍你的项目。")))
    text, fell_back = sel.safe_render("t.expand", "缓存")
    assert fell_back is True
    assert text == "备用题：请介绍你的项目。"


def test_safe_render_falls_back_when_classifier_unavailable():
    """fail-closed：拿不到判定 == 不可放行 → 退回题库题，而不放行生成内容。"""
    sel = TemplateSelector(TEMPLATES, _Unavailable(), _bank(_approved("q1", "备用题")))
    text, fell_back = sel.safe_render("t.expand", "缓存")
    assert fell_back is True
    assert text == "备用题"


def test_safe_render_without_selectable_question_raises():
    """题库无可选题时抛 Unavailable——绝不凭空编一道题。"""
    sel = TemplateSelector(TEMPLATES, _AlwaysHit(), _bank())
    with pytest.raises(Unavailable) as ei:
        sel.safe_render("t.expand", "缓存")
    assert ei.value.provider == "question-bank"


def test_safe_render_uses_only_approved_current_question():
    """回退题必须是已审校的当前版本；草稿/已下架题不得出现在候选人面前。"""
    bank = _bank(
        _approved("q1", "已审校题"),
        QuestionBankItem("q2", "草稿题", 1, "a", ReviewStatus.DRAFT, "backend", 0.5, frozenset(), 1),
        QuestionBankItem("q3", "下架题", 1, "a", ReviewStatus.RETIRED, "backend", 0.5, frozenset(), 1),
    )
    sel = TemplateSelector(TEMPLATES, _AlwaysHit(), bank)
    text, fell_back = sel.safe_render("t.expand", "缓存")
    assert fell_back is True and text == "已审校题"


def test_safe_render_integration_with_attribute_classifier():
    """真实分类器：模板内容若触发属性评论红线，也必须回退。"""
    tpl = (FollowupTemplate("t.bad", "你提到{keyword}，不过你的语速有点太快了。", "x"),)
    sel = TemplateSelector(
        tpl, AttributeCommentClassifier(), _bank(_approved("q1", "备用题"))
    )
    text, fell_back = sel.safe_render("t.bad", "并发")
    assert fell_back is True and text == "备用题"


# ---------- assert_templated ----------


def test_assert_templated_accepts_matching_output():
    sel = TemplateSelector(TEMPLATES, _ok_classifier(), _bank())
    out = sel.render("t.expand", "分库分表")
    sel.assert_templated(out, "t.expand")  # 不应抛


def test_assert_templated_accepts_regex_special_keyword():
    """校验是拿模板生成正则（而非拿渲染文本），keyword 含正则元字符不应误报。"""
    sel = TemplateSelector(TEMPLATES, _ok_classifier(), _bank())
    out = sel.render("t.example", "C++ (*a.b*)")
    sel.assert_templated(out, "t.example")


def test_assert_templated_rejects_modified_output():
    """核心：产出被事后改写（哪怕加一个字）必须报错——拦截自由生成。"""
    sel = TemplateSelector(TEMPLATES, _ok_classifier(), _bank())
    out = sel.render("t.expand", "缓存") + "（悄悄多加的一句）"
    with pytest.raises(ValueError, match="不一致"):
        sel.assert_templated(out, "t.expand")


def test_assert_templated_unknown_template_rejected():
    sel = TemplateSelector(TEMPLATES, _ok_classifier(), _bank())
    with pytest.raises(ValueError, match="不存在"):
        sel.assert_templated("任意", "t.ghost")


def test_assert_templated_non_str_rejected():
    sel = TemplateSelector(TEMPLATES, _ok_classifier(), _bank())
    with pytest.raises(TypeError, match="str"):
        sel.assert_templated(123, "t.expand")  # type: ignore[arg-type]


def test_assert_templated_does_not_cross_match_templates():
    """来自 A 模板的产出不能冒充 B 模板。"""
    sel = TemplateSelector(TEMPLATES, _ok_classifier(), _bank())
    out_a = sel.render("t.expand", "缓存")
    with pytest.raises(ValueError, match="不一致"):
        sel.assert_templated(out_a, "t.example")
