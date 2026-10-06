"""生成内容安全过滤（方案 §12.4）。

为什么默认拒绝（fail-closed）
----------------------------
面试系统会把模型生成的追问/题目**直接展示给候选人**。一旦分类器不可用却被当成
"通过"，涉政/涉黄/歧视性内容就会当场推给候选人与企业客户，事后无法撤回。
因此本模块的态度是：**拿不到判定 == 不可放行**。`CompositeSafetyClassifier`
在任一子分类器抛异常时一律抛 `Unavailable`（而不是降级为 OK），由上层退回题库题——
宁可少问一句、回到已审校的题库，也绝不放行未经判定的生成内容。

为什么词表必须外部注入
----------------------
源码里一旦写死真实敏感词，仓库本身就成了敏感词清单（传播、误用、无法审计），
且不同客户/地区的词表本就不同。所以 `LexiconSafetyClassifier` 只接受外部注入的词表，
并在构造时校验它**不得包含** `SafetyCategory.OK`——OK 的语义是"无命中"，
把它放进词表会让"命中"与"通过"同义，判定瞬间失效。

为什么要专门拦"属性评论"
------------------------
方案红线：**禁止对候选人的外貌/口音/年龄/性别/地域做任何评论**。这类表述往往不含
任何敏感词，靠词表拦不住，只能用句式规则。关键判据是——
**这是对"被试者属性"下判断，还是对"作答内容"的反馈/指令**：
  - "你的口音比较重" —— 对属性（口音）下判断 → 拦截；
  - "你说得比较清楚" —— 对作答内容（清晰度）的反馈，未指向任何属性指称 → 放行；
  - "请放慢一点语速" —— 指令（带"请"前缀，重定向的是行为） → 放行。
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

from ..domain.errors import Unavailable


class SafetyCategory(str, Enum):
    """生成内容的安全类别。`OK` 表示"无命中"，**不得**出现在注入词表里。"""

    OK = "ok"
    POLITICAL = "political"
    SEXUAL = "sexual"
    DISCRIMINATION = "discrimination"
    PERSONAL_JUDGEMENT = "personal_judgement"
    MEDICAL_ADVICE = "medical_advice"
    APPEARANCE_COMMENT = "appearance_comment"
    SPEECH_COMMENT = "speech_comment"
    DEMOGRAPHIC_COMMENT = "demographic_comment"


#: 危害优先级（数值越大越严重）。
#:
#: 排序理由：涉政/涉黄属于合规红线，最高；歧视直接伤害群体且可能触发法律风险；
#: 人身评价与医疗建议会误导候选人；属性评论（外貌/口音/人口学）伤害相对可控但
#: 属于方案明确禁止的红线，放在其后。**同一 span 只归一个类别**，多类别命中时
#: 用这张表选出要返回的那一个类别，避免"一次命中被数成好几条"。
SAFETY_PRIORITY: dict[SafetyCategory, int] = {
    SafetyCategory.POLITICAL: 80,
    SafetyCategory.SEXUAL: 80,
    SafetyCategory.DISCRIMINATION: 70,
    SafetyCategory.PERSONAL_JUDGEMENT: 60,
    SafetyCategory.MEDICAL_ADVICE: 50,
    SafetyCategory.DEMOGRAPHIC_COMMENT: 40,
    SafetyCategory.APPEARANCE_COMMENT: 30,
    SafetyCategory.SPEECH_COMMENT: 20,
    SafetyCategory.OK: 0,
}

#: 类别的中文名，用于人读的 detail（日志/告警里出现英文枚举不便于运营排查）。
_CATEGORY_CN: dict[SafetyCategory, str] = {
    SafetyCategory.OK: "通过",
    SafetyCategory.POLITICAL: "涉政",
    SafetyCategory.SEXUAL: "涉黄",
    SafetyCategory.DISCRIMINATION: "歧视",
    SafetyCategory.PERSONAL_JUDGEMENT: "人身评价",
    SafetyCategory.MEDICAL_ADVICE: "医疗建议",
    SafetyCategory.APPEARANCE_COMMENT: "外貌评论",
    SafetyCategory.SPEECH_COMMENT: "口音/表达评论",
    SafetyCategory.DEMOGRAPHIC_COMMENT: "人口学评论",
}


def severity_of(category: SafetyCategory) -> int:
    """取类别的危害优先级；未知类别保守地按最高危害处理（fail-closed）。"""
    return SAFETY_PRIORITY.get(category, max(SAFETY_PRIORITY.values()))


@dataclass(frozen=True)
class SafetyVerdict:
    """一次安全判定。

    `hit=True` 即"不可放行"。`spans` 是命中的字符区间（左闭右开），只记录被采纳
    那个类别的命中位置——保持 `category` 与 `spans` 语义一致，避免让人误解成
    "这些位置属于别的高危类别"。
    """

    category: SafetyCategory
    hit: bool
    spans: tuple[tuple[int, int], ...] = ()
    detail: str = ""

    @classmethod
    def ok(cls) -> "SafetyVerdict":
        """构造"通过"判定（无命中、无 span）。"""
        return cls(category=SafetyCategory.OK, hit=False, spans=(), detail="")

    @property
    def allowed(self) -> bool:
        """是否放行。只有未命中才放行。"""
        return not self.hit


@runtime_checkable
class SafetyClassifierPort(Protocol):
    """安全分类端口。实现方**不得**在无法判定时返回 OK（那等于放行）。"""

    def classify(self, text: str) -> SafetyVerdict: ...


# ---------- 词表分类器 ----------


def _coerce_category(raw: object) -> SafetyCategory:
    """把词表键规整成枚举。

    注意 `SafetyCategory` 是 `str` 枚举，`SafetyCategory.OK == "ok"` 为真，
    所以必须显式转换，否则外部传入裸字符串 "ok" 会绕过"不得包含 OK"的校验。
    """
    if isinstance(raw, SafetyCategory):
        return raw
    try:
        return SafetyCategory(raw)  # type: ignore[arg-type]
    except ValueError as exc:
        raise ValueError(f"未知的安全类别：{raw!r}") from exc


class LexiconSafetyClassifier:
    """基于外部注入词表的分类器。

    命中策略：**最长词优先**的非重叠贪心扫描。这样做的直接收益是
    "同一个 span 只归一个类别"——扫描时一旦消费掉 [i, i+len) 就跳过该区间，
    不会出现一个词被两个类别各记一次（进而把一次命中数成两条告警）。
    """

    name = "lexicon"

    def __init__(self, lexicon: Mapping[SafetyCategory, Iterable[str]]) -> None:
        if not isinstance(lexicon, Mapping):
            raise TypeError(f"lexicon 必须是 Mapping，收到 {type(lexicon).__name__}")
        terms: list[tuple[str, SafetyCategory]] = []
        for raw_cat, raw_terms in lexicon.items():
            cat = _coerce_category(raw_cat)
            if cat is SafetyCategory.OK:
                raise ValueError(
                    "词表不得包含 SafetyCategory.OK：OK 表示“无命中”，"
                    "放进词表会让每次判定都变成“命中即通过”"
                )
            for raw_term in raw_terms:
                if not isinstance(raw_term, str):
                    raise TypeError(f"词项必须是 str，收到 {type(raw_term).__name__}")
                term = raw_term.strip()
                if not term:
                    raise ValueError("词项不能为空或纯空白（空词会命中任意位置）")
                terms.append((term, cat))
        # 最长优先；同长度按危害优先级降序，再按字典序 —— 全序保证结果确定。
        terms.sort(key=lambda p: (-len(p[0]), -severity_of(p[1]), p[0]))
        self._terms: tuple[tuple[str, SafetyCategory], ...] = tuple(terms)

    def classify(self, text: str) -> SafetyVerdict:
        if not isinstance(text, str):
            # 不静默当空串：静默会把"类型错了"误判成"安全"，属于放行。
            raise TypeError(f"classify 需要 str，收到 {type(text).__name__}")
        if not text.strip():
            return SafetyVerdict.ok()

        hits: dict[SafetyCategory, list[tuple[int, int]]] = {}
        i = 0
        n = len(text)
        while i < n:
            chosen: tuple[str, SafetyCategory] | None = None
            for term, cat in self._terms:
                if text.startswith(term, i):
                    chosen = (term, cat)
                    break
            if chosen is None:
                i += 1
                continue
            term, cat = chosen
            hits.setdefault(cat, []).append((i, i + len(term)))
            i += len(term)  # 跳过整个命中区间 → 同一 span 不会二次归属

        if not hits:
            return SafetyVerdict.ok()

        winner = max(hits, key=lambda c: (severity_of(c), -min(p for p, _ in hits[c])))
        total = sum(len(v) for v in hits.values())
        covered = "、".join(sorted(_CATEGORY_CN[c] for c in hits))
        return SafetyVerdict(
            category=winner,
            hit=True,
            spans=tuple(sorted(hits[winner])),
            detail=f"词表命中 {total} 处，涉及类别：{covered}；返回最高危害：{_CATEGORY_CN[winner]}",
        )


# ---------- 属性评论分类器 ----------

#: 属性指称（每种属性一组）。**只看方案明确点名的属性**，不无节制扩张——
#: 指称集合越大，误伤正常面试话术的概率越高。
_APPEARANCE_TERMS: tuple[str, ...] = ("长相", "外貌", "颜值", "形象", "穿着", "仪表")
_SPEECH_TERMS: tuple[str, ...] = ("口音", "方言", "普通话", "吐字", "语速")
_DEMOGRAPHIC_TERMS: tuple[str, ...] = ("年龄", "年纪", "岁数", "性别", "地域", "籍贯", "家乡")

_ATTRIBUTE_TERMS: tuple[tuple[str, SafetyCategory], ...] = tuple(
    sorted(
        [(t, SafetyCategory.APPEARANCE_COMMENT) for t in _APPEARANCE_TERMS]
        + [(t, SafetyCategory.SPEECH_COMMENT) for t in _SPEECH_TERMS]
        + [(t, SafetyCategory.DEMOGRAPHIC_COMMENT) for t in _DEMOGRAPHIC_TERMS],
        key=lambda p: (-len(p[0]), p[0]),
    )
)

#: 程度副词（可叠用，如"有点偏"）。用于吸收"偏大/比较重/太慢"这类结构。
_DEGREE_TOKENS: tuple[str, ...] = (
    "比较", "非常", "特别", "相当", "有点", "稍微", "略微", "格外", "极其",
    "过于", "愈发", "更加", "最为", "不太", "不很", "很", "太", "挺", "最", "较", "更", "偏",
)
#: 评价谓词。含"不大/不好/较多/较少/不够/之间"这类带间隔的否定/比较形式，
#: 它们本身就是完整谓词，允许在没有程度副词时独立成立。
_CORE_TOKENS: tuple[str, ...] = (
    "不标准", "标准", "糟糕", "不行", "不错", "一般", "明显", "清楚", "含糊",
    "不好", "不大", "较多", "较少", "不够", "之间",
    "好", "差", "烂", "重", "轻", "大", "小", "快", "慢", "高", "低", "强", "弱", "浓", "淡",
)


def _alternation(tokens: tuple[str, ...]) -> str:
    """构造正则或分支；长词在前，避免"标准"抢先匹配掉"不标准"。"""
    return "|".join(re.escape(t) for t in sorted(tokens, key=len, reverse=True))


_DEGREE = f"(?:{_alternation(_DEGREE_TOKENS)})"
_CORE = f"(?:{_alternation(_CORE_TOKENS)})"

#: 属性指称**右侧**紧邻的谓词（允许一个连接助词 + 至多两个程度副词）。
#: 用 `^` 锚定在指称结尾，是为了避免把"家乡在哪里，**大**学在哪读的"这种
#: 距离较远的字误当成谓词——谓词必须紧贴指称才构成评价式表述。
_RIGHT_PRED_RE = re.compile(rf"^[的得是有]?\s*{_DEGREE}{{0,2}}{_CORE}")
#: 属性指称**左侧**的谓词（"很重 的 口音"这类前置定语）。
_LEFT_PRED_RE = re.compile(rf"{_DEGREE}{{0,2}}{_CORE}[的得]?$")

#: 指令前缀：出现在属性指称之前时，说明这是"让人怎么做"而不是"评价这个人"。
_INSTRUCTION_RE = re.compile(r"(?:请你|麻烦你|请|麻烦|能否|可否|建议|希望|务必)")

#: 谓词窗口（方案 §12.4：指称前后各 8 个字符内）。
_PRED_WINDOW = 8
#: 指令前缀回看窗口（"请……语速"中间可能夹若干字）。
_INSTRUCTION_LOOKBACK = 12


class AttributeCommentClassifier:
    """用句式规则检测"对候选人某种属性下判断"，**不依赖词表**。

    只覆盖 `APPEARANCE_COMMENT`/`SPEECH_COMMENT`/`DEMOGRAPHIC_COMMENT`：
    这两类最容易以"善意提醒"的形式出现，却不含任何敏感词，词表拦不住。
    """

    name = "attribute-comment"

    def classify(self, text: str) -> SafetyVerdict:
        if not isinstance(text, str):
            raise TypeError(f"classify 需要 str，收到 {type(text).__name__}")
        if not text.strip():
            return SafetyVerdict.ok()

        hits: dict[SafetyCategory, list[tuple[int, int]]] = {}
        for term, cat in _ATTRIBUTE_TERMS:
            start = 0
            while True:
                pos = text.find(term, start)
                if pos < 0:
                    break
                end = pos + len(term)
                if self._is_comment(text, pos, end):
                    hits.setdefault(cat, []).append((pos, end))
                start = end

        if not hits:
            return SafetyVerdict.ok()

        winner = max(hits, key=lambda c: (severity_of(c), -min(p for p, _ in hits[c])))
        return SafetyVerdict(
            category=winner,
            hit=True,
            spans=tuple(sorted(hits[winner])),
            detail=f"检测到对候选人{_CATEGORY_CN[winner]}的评价式表述（属性判据，非作答反馈）",
        )

    @staticmethod
    def _is_comment(text: str, start: int, end: int) -> bool:
        """判定指称 [start, end) 是否处在"评价式"语境里。

        三步：
        1. 指称前若出现指令词（请/麻烦/建议…），是在下指令，不是评价 → 不算命中。
        2. 指称右侧紧贴谓词 → 是评价（"语速太快""普通话不够标准"）。
        3. 指称左侧紧贴谓词 → 是前置定语评价（"很重的口音"）。
        """
        prefix = text[max(0, start - _INSTRUCTION_LOOKBACK):start]
        if _INSTRUCTION_RE.search(prefix):
            return False
        if _RIGHT_PRED_RE.match(text[end:end + _PRED_WINDOW]):
            return True
        return bool(_LEFT_PRED_RE.search(text[max(0, start - _PRED_WINDOW):start]))


# ---------- 组合分类器 ----------


class CompositeSafetyClassifier:
    """按序调用子分类器，返回危害优先级最高的命中；全部 OK 才 OK。

    失败语义（本模块最关键的一条纪律）：**任一子分类器抛异常 = 整条判定不可用**，
    直接抛 `Unavailable`，由调用方退回题库题。绝不 catch 后当成 OK——
    那正是"分类器挂了就悄悄放行"的经典事故路径。
    """

    name = "composite"

    def __init__(self, classifiers: Sequence[SafetyClassifierPort]) -> None:
        classifiers = tuple(classifiers)
        if not classifiers:
            # 空组合等于"永不放行判定"却"永远返回通过"，是静默放行，必须拒绝构造。
            raise ValueError("CompositeSafetyClassifier 至少需要一个子分类器")
        self._classifiers = classifiers

    def classify(self, text: str) -> SafetyVerdict:
        best = SafetyVerdict.ok()
        for clf in self._classifiers:
            try:
                verdict = clf.classify(text)
            except Exception as exc:  # noqa: BLE001 - 任何失败都按不可用处理（fail-closed）
                sub_name = getattr(clf, "name", type(clf).__name__)
                raise Unavailable(
                    "safety-classifier",
                    f"子分类器 {sub_name} 调用失败，按“不可放行”处理：{exc!r}",
                ) from exc
            if verdict.hit and severity_of(verdict.category) > severity_of(best.category):
                best = verdict
        return best


__all__ = [
    "SafetyCategory",
    "SafetyVerdict",
    "SafetyClassifierPort",
    "LexiconSafetyClassifier",
    "AttributeCommentClassifier",
    "CompositeSafetyClassifier",
    "SAFETY_PRIORITY",
    "severity_of",
]
