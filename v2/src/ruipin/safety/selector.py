"""受限追问模板 + 安全网关（方案 §12.4）。

方案要求"追问从**受限模板库**选取，不自由生成"。原因：自由生成的追问是安全与
公平风险最集中的出口——它既可能夹带违规内容，也可能临时编出一句对被试者属性的
评头论足。本模块把出口收窄成两件事：
  1. 只能从登记在册的模板渲染（`assert_templated` 会校验产出确实来自模板）；
  2. 渲染结果必须过一遍安全分类，**命中就丢弃并退回题库题**。

安全分类器不可用怎么办
----------------------
`safe_render` 把分类器的 `Unavailable` 与"命中"等同处理——都退回题库题。
本项目纪律是"拿不到判定 == 不可放行"：分类器挂了绝不放行生成内容，
退回已审校的题库题是代价最小、后果最可控的退路。
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from ..domain.errors import Unavailable
from .bank import QuestionBank, QuestionBankItem
from .classifier import SafetyClassifierPort

#: 模板占位符。渲染用 `{keyword}`，校验正则时替换为惰性通配。
_PLACEHOLDER = "{keyword}"
_KEYWORD_GROUP = r"[\s\S]+?"


@dataclass(frozen=True)
class FollowupTemplate:
    """一个受限追问模板。

    `pattern` 必须含且只含 `{keyword}` 这一个占位符，其余字符原样输出——
    模板本身是"白名单内容"，不允许有第二个可注入点。
    """

    template_id: str
    pattern: str
    category: str

    def __post_init__(self) -> None:
        if not self.template_id or not self.template_id.strip():
            raise ValueError("template_id 不能为空")
        if not self.pattern or not self.pattern.strip():
            raise ValueError("pattern 不能为空")
        if not self.category or not self.category.strip():
            raise ValueError("category 不能为空")
        if _PLACEHOLDER not in self.pattern:
            raise ValueError(f"模板 {self.template_id} 必须包含占位符 {_PLACEHOLDER}")
        try:
            self.pattern.format(**{_PLACEHOLDER.strip("{}"): "x"})
        except (KeyError, IndexError, ValueError) as exc:
            # 模板里若有未转义的花括号，format 会炸；提前在构造期暴露，
            # 免得等到线上渲染时才失败（那时已进入对话流程）。
            raise ValueError(f"模板 {self.template_id} 格式非法：{exc}") from exc


class TemplateSelector:
    """受限追问模板选择器 + 安全网关。"""

    def __init__(
        self,
        templates: Sequence[FollowupTemplate],
        safety: SafetyClassifierPort,
        fallback_bank: QuestionBank,
    ) -> None:
        templates = tuple(templates)
        if not templates:
            raise ValueError("templates 不能为空")
        by_id: dict[str, FollowupTemplate] = {}
        regex_by_id: dict[str, re.Pattern[str]] = {}
        for tpl in templates:
            if tpl.template_id in by_id:
                raise ValueError(f"模板 id 重复：{tpl.template_id}")
            by_id[tpl.template_id] = tpl
            regex_by_id[tpl.template_id] = _template_regex(tpl.pattern)
        self._templates = by_id
        self._regexes = regex_by_id
        self._safety = safety
        self._fallback_bank = fallback_bank

    # ---------- 渲染 ----------

    def render(self, template_id: str, keyword: str) -> str:
        """按模板渲染一句追问。未知模板 id 报错（不允许凭空造模板）。"""
        tpl = self._template(template_id)
        if not isinstance(keyword, str):
            raise TypeError(f"keyword 必须是 str，收到 {type(keyword).__name__}")
        return tpl.pattern.format(keyword=keyword)

    def safe_render(self, template_id: str, keyword: str) -> tuple[str, bool]:
        """渲染并过安全网关，返回 `(文本, 是否发生回退)`。

        命中或分类器不可用 → 退回题库题，第二项为 `True`。
        题库无可选题时抛 `Unavailable`——**不凭空编一道题**。
        """
        text = self.render(template_id, keyword)
        try:
            verdict = self._safety.classify(text)
        except Unavailable:
            # 分类器不可用 = 无法判定 = 不放行（fail-closed），退回题库题。
            return self._fallback_text(), True
        if verdict.hit:
            return self._fallback_text(), True
        return text, False

    def assert_templated(self, text: str, template_id: str) -> None:
        """校验 `text` 确实来自 `template_id` 的模板（方案：不得自由生成）。

        做法：把模板转成正则（占位符 → 惰性通配）后全匹配。若产出被事后改写
        （哪怕多一个字），fullmatch 失败并抛 `ValueError`——这正是要拦的场景：
        有人在渲染后手工拼接了额外内容绕开模板白名单。
        """
        if not isinstance(text, str):
            raise TypeError(f"text 必须是 str，收到 {type(text).__name__}")
        regex = self._regexes.get(template_id)
        if regex is None:
            raise ValueError(f"模板不存在：{template_id}")
        if not regex.fullmatch(text):
            raise ValueError(
                f"产出与模板 {template_id} 不一致（疑似渲染后被改写，违反“追问须来自受限模板库”）"
            )

    # ---------- 内部 ----------

    def _template(self, template_id: str) -> FollowupTemplate:
        tpl = self._templates.get(template_id)
        if tpl is None:
            raise ValueError(f"模板不存在：{template_id}")
        return tpl

    def _fallback_item(self) -> QuestionBankItem:
        options = self._fallback_bank.selectable()
        if not options:
            raise Unavailable("question-bank", "无可用已审校题目")
        return options[0]

    def _fallback_text(self) -> str:
        return self._fallback_item().text


def _template_regex(pattern: str) -> re.Pattern[str]:
    """把模板 pattern 转成全匹配正则：占位符 → 惰性通配，其余字符字面量。"""
    escaped = re.escape(pattern).replace(re.escape(_PLACEHOLDER), _KEYWORD_GROUP)
    return re.compile(escaped)


__all__ = ["FollowupTemplate", "TemplateSelector"]
