"""安全分类器测试（`ruipin.safety.classifier`）。

重点钉住三条纪律：
1. **fail-closed**：组合分类器遇子分类器异常必须抛 `Unavailable`，不得当 OK 放行；
2. **不硬编码词表**：词表由外部注入，且不得包含 `SafetyCategory.OK`；
3. **属性评论 vs 正常反馈/指令**：正例必须命中，反例必须判 OK（边界最易写错）。
"""

from __future__ import annotations

import pytest

from ruipin.domain.errors import Unavailable
from ruipin.safety.classifier import (
    SAFETY_PRIORITY,
    AttributeCommentClassifier,
    CompositeSafetyClassifier,
    LexiconSafetyClassifier,
    SafetyCategory,
    SafetyClassifierPort,
    SafetyVerdict,
    severity_of,
)

CAT = SafetyCategory


# ---------- 值对象 ----------


def test_verdict_ok_is_not_hit_and_allowed():
    v = SafetyVerdict.ok()
    assert v.category is CAT.OK
    assert v.hit is False
    assert v.allowed is True
    assert v.spans == () and v.detail == ""


def test_verdict_hit_is_not_allowed():
    v = SafetyVerdict(category=CAT.POLITICAL, hit=True)
    assert v.allowed is False


def test_verdict_is_frozen():
    v = SafetyVerdict.ok()
    with pytest.raises(Exception):
        v.category = CAT.SEXUAL  # type: ignore[misc]


def test_severity_priority_order():
    """危害优先级排序必须稳定：涉政/涉黄 > 歧视 > 人身评价 > 医疗 > 人口学 > 外貌 > 口音。"""
    order = [
        CAT.POLITICAL,
        CAT.SEXUAL,
        CAT.DISCRIMINATION,
        CAT.PERSONAL_JUDGEMENT,
        CAT.MEDICAL_ADVICE,
        CAT.DEMOGRAPHIC_COMMENT,
        CAT.APPEARANCE_COMMENT,
        CAT.SPEECH_COMMENT,
        CAT.OK,
    ]
    sev = [severity_of(c) for c in order]
    assert sev == sorted(sev, reverse=True)
    assert severity_of(CAT.OK) == 0


def test_severity_unknown_category_defaults_to_max():
    """未知类别保守按最高危害处理——不能让新类别悄悄变成"低危"。"""
    assert severity_of("not-a-real-category") == max(SAFETY_PRIORITY.values())  # type: ignore[arg-type]


def test_lexicon_implements_protocol():
    clf = LexiconSafetyClassifier({CAT.POLITICAL: frozenset({"x"})})
    assert isinstance(clf, SafetyClassifierPort)


# ---------- 词表分类器：构造校验 ----------


def test_lexicon_rejects_ok_category():
    """纪律：OK 语义是"无命中"，放进词表会让"命中"与"通过"同义。"""
    with pytest.raises(ValueError, match="OK"):
        LexiconSafetyClassifier({CAT.OK: frozenset({"任意"})})


def test_lexicon_rejects_ok_via_plain_string_key():
    """str 枚举下 "ok" == SafetyCategory.OK，必须显式转换后仍被拦截。"""
    with pytest.raises(ValueError, match="OK"):
        LexiconSafetyClassifier({"ok": frozenset({"任意"})})


def test_lexicon_rejects_unknown_category():
    with pytest.raises(ValueError, match="未知的安全类别"):
        LexiconSafetyClassifier({"bogus": frozenset({"x"})})


def test_lexicon_accepts_plain_string_category_key():
    """合法的裸字符串键应被规整为枚举而不是报错。"""
    clf = LexiconSafetyClassifier({"political": frozenset({"fubar"})})
    assert clf.classify("fubar").category is CAT.POLITICAL


def test_lexicon_rejects_non_mapping():
    with pytest.raises(TypeError, match="Mapping"):
        LexiconSafetyClassifier([("political", {"x"})])  # type: ignore[arg-type]


def test_lexicon_rejects_non_str_term():
    with pytest.raises(TypeError, match="词项"):
        LexiconSafetyClassifier({CAT.POLITICAL: [123]})  # type: ignore[list-item]


def test_lexicon_rejects_blank_term():
    with pytest.raises(ValueError, match="空"):
        LexiconSafetyClassifier({CAT.POLITICAL: ["  "]})


# ---------- 词表分类器：匹配 ----------


def test_lexicon_empty_or_whitespace_is_ok():
    clf = LexiconSafetyClassifier({CAT.POLITICAL: frozenset({"bad"})})
    assert clf.classify("").hit is False
    assert clf.classify("   \n\t ").hit is False


def test_lexicon_non_str_raises_type_error():
    """非 str 不得静默当空串——那会把"类型错了"误判成"安全"。"""
    clf = LexiconSafetyClassifier({CAT.POLITICAL: frozenset({"bad"})})
    with pytest.raises(TypeError, match="str"):
        clf.classify(None)  # type: ignore[arg-type]


def test_lexicon_picks_highest_severity_across_categories():
    clf = LexiconSafetyClassifier(
        {CAT.SPEECH_COMMENT: frozenset({"低危"}), CAT.POLITICAL: frozenset({"高危"})}
    )
    # 低危在前、高危在后，仍应返回高危（按危害而非出现顺序）
    v1 = clf.classify("低危然后高危")
    assert v1.category is CAT.POLITICAL
    # 调换顺序，结论不变
    v2 = clf.classify("高危然后低危")
    assert v2.category is CAT.POLITICAL


def test_lexicon_same_span_not_double_counted():
    """同一 span 只归一个类别：同一段文字含两类别同词时只计 1 处命中。"""
    clf = LexiconSafetyClassifier(
        {CAT.DISCRIMINATION: frozenset({"敏感"}), CAT.POLITICAL: frozenset({"敏感"})}
    )
    v = clf.classify("敏感")
    assert v.hit is True
    assert v.category is CAT.POLITICAL  # 优先级更高者胜出
    assert len(v.spans) == 1  # 不重复计数
    assert "1 处" in v.detail


def test_lexicon_longest_match_first():
    """最长词优先：短词是高危、长词是低危时，长词仍先被消费。"""
    clf = LexiconSafetyClassifier(
        {CAT.POLITICAL: frozenset({"测试"}), CAT.SEXUAL: frozenset({"测试词"})}
    )
    v = clf.classify("测试词")
    assert v.category is CAT.SEXUAL
    assert v.spans == ((0, 3),)


def test_lexicon_records_all_spans_of_winner():
    clf = LexiconSafetyClassifier({CAT.POLITICAL: frozenset({"甲", "乙"})})
    v = clf.classify("甲xx乙")
    assert v.spans == ((0, 1), (3, 4))


def test_lexicon_spans_are_non_overlapping_and_sorted():
    clf = LexiconSafetyClassifier({CAT.POLITICAL: frozenset({"aa"})})
    v = clf.classify("aaaa")
    assert v.spans == ((0, 2), (2, 4))


def test_lexicon_detail_mentions_other_categories():
    """多类别命中时 detail 要列出涉及类别，便于运营排查，但不能漏掉返回的类别。"""
    clf = LexiconSafetyClassifier(
        {CAT.POLITICAL: frozenset({"甲"}), CAT.SEXUAL: frozenset({"乙"})}
    )
    v = clf.classify("甲乙")
    assert "涉政" in v.detail and "涉黄" in v.detail


# ---------- 组合分类器 ----------


class _Stub:
    """按固定 verdict 返回的假分类器。"""

    name = "stub"

    def __init__(self, verdict: SafetyVerdict) -> None:
        self._verdict = verdict

    def classify(self, text: str) -> SafetyVerdict:
        return self._verdict


class _Boom:
    name = "boom"

    def classify(self, text: str) -> SafetyVerdict:
        raise RuntimeError("模型超时")


def test_composite_requires_at_least_one_classifier():
    """空组合等于"永远返回通过"的静默放行，必须拒绝构造。"""
    with pytest.raises(ValueError, match="子分类器"):
        CompositeSafetyClassifier([])


def test_composite_all_ok_returns_ok():
    c = CompositeSafetyClassifier([_Stub(SafetyVerdict.ok()), _Stub(SafetyVerdict.ok())])
    assert c.classify("无害文本").allowed is True


def test_composite_returns_highest_severity_hit():
    low = _Stub(SafetyVerdict(category=CAT.SPEECH_COMMENT, hit=True))
    high = _Stub(SafetyVerdict(category=CAT.DISCRIMINATION, hit=True))
    c = CompositeSafetyClassifier([low, high])
    assert c.classify("x").category is CAT.DISCRIMINATION


def test_composite_ignores_lower_severity_after_higher():
    """先命中的低危不能被后到的更低危覆盖，只保留最高危。"""
    high = _Stub(SafetyVerdict(category=CAT.POLITICAL, hit=True))
    low = _Stub(SafetyVerdict(category=CAT.SPEECH_COMMENT, hit=True))
    c = CompositeSafetyClassifier([high, low])
    assert c.classify("x").category is CAT.POLITICAL


def test_composite_fail_closed_on_exception():
    """最关键的一条：子分类器异常 → 抛 Unavailable，绝不降级为 OK 放行。"""
    c = CompositeSafetyClassifier([_Stub(SafetyVerdict.ok()), _Boom()])
    with pytest.raises(Unavailable) as ei:
        c.classify("任意内容")
    assert ei.value.provider == "safety-classifier"
    assert "boom" in ei.value.reason


def test_composite_fail_closed_wraps_unavailable_from_sub():
    class _Unavail:
        name = "down"

        def classify(self, text: str) -> SafetyVerdict:
            raise Unavailable("some-model", "端点 500")

    c = CompositeSafetyClassifier([_Unavail()])
    with pytest.raises(Unavailable) as ei:
        c.classify("x")
    assert ei.value.provider == "safety-classifier"


def test_composite_with_real_lexicon_and_attribute_classifier():
    lex = LexiconSafetyClassifier({CAT.POLITICAL: frozenset({"甲"})})
    comp = CompositeSafetyClassifier([lex, AttributeCommentClassifier()])
    assert comp.classify("你的口音比较重").category is CAT.SPEECH_COMMENT
    assert comp.classify("甲").category is CAT.POLITICAL
    assert comp.classify("请介绍一下你的项目").allowed is True


# ---------- 属性评论：正例（必须命中） ----------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("你的口音比较重", CAT.SPEECH_COMMENT),
        ("你说话普通话不太标准", CAT.SPEECH_COMMENT),
        ("你的语速太快了", CAT.SPEECH_COMMENT),
        ("很重的口音", CAT.SPEECH_COMMENT),  # 左侧前置定语
        ("你看起来年纪偏大", CAT.DEMOGRAPHIC_COMMENT),
        ("你的年龄偏大", CAT.DEMOGRAPHIC_COMMENT),
        ("你的长相一般", CAT.APPEARANCE_COMMENT),
        ("你的仪表太差", CAT.APPEARANCE_COMMENT),
    ],
)
def test_attribute_comment_positives(text, expected):
    assert AttributeCommentClassifier().classify(text).category is expected


def test_attribute_comment_trailing_suggestion_does_not_whitewash():
    """尾随的"建议…"是补充指令，不能因此放过前面的属性评价。"""
    v = AttributeCommentClassifier().classify("你的口音有点重，建议多练习")
    assert v.category is CAT.SPEECH_COMMENT


# ---------- 属性评论：反例（必须 OK） ----------


@pytest.mark.parametrize(
    "text",
    [
        "请介绍一下你的项目",  # 正常指令
        "你说得比较清楚",  # 对作答内容的反馈（无属性指称）
        "请放慢一点语速",  # 指令：请 + 语速
        "请用普通话回答这个问题",  # 指令：请 + 普通话
        "你的项目做得很扎实",  # 评价的是项目，不是候选人属性
        "你的家乡在哪里，大学在哪读的",  # 含"家乡"但与"大"距离远，非谓词
        "请说明一下你解决过的最难的技术问题",
        "能具体讲讲你在团队里的角色吗",
    ],
)
def test_attribute_comment_negatives_are_ok(text):
    v = AttributeCommentClassifier().classify(text)
    assert v.hit is False, f"正常话术被误判为属性评论：{text} -> {v.category}"
    assert v.category is CAT.OK


def test_attribute_comment_reports_span_of_referent():
    v = AttributeCommentClassifier().classify("你的口音比较重")
    assert v.spans == ((2, 4),)  # "口音"


def test_attribute_comment_empty_is_ok():
    assert AttributeCommentClassifier().classify("").hit is False
    assert AttributeCommentClassifier().classify("  \n ").hit is False


def test_attribute_comment_non_str_raises():
    with pytest.raises(TypeError, match="str"):
        AttributeCommentClassifier().classify(42)  # type: ignore[arg-type]


def test_attribute_comment_picks_highest_severity():
    """同一句里人口学评论比口音评论危害高，应返回人口学。"""
    v = AttributeCommentClassifier().classify("你的口音比较重，而且年纪偏大")
    assert v.category is CAT.DEMOGRAPHIC_COMMENT
