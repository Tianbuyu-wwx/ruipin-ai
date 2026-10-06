"""题库治理（方案 §12.4）。

两条贯穿本模块的纪律
--------------------
1. **改过内容的题不能沿用旧的已审状态**。`revise` 产出 `version+1` 的新条目，
   状态强制回到 `PENDING_REVIEW`，旧版本转 `RETIRED`。否则"审校"形同虚设：
   审完再悄悄改字，候选人看到的仍是那份盖过章的内容。
2. **样本不足时不得用 0.0 冒充通过率**。`BiasAudit` 对样本数 < `MIN_GROUP_SAMPLE`
   的组显式返回 `None`（并在 `GroupStat.sufficient` 标出），调用方必须自己决定
   怎么处理"测不准"，而不是被一个假的 0.0 骗着下"该组通过率极低"的结论。

时间来源
--------
`created_at` 一律取自注入的 `clock`（返回秒），**不调用 `time.time()`**，
否则同一份输入在不同时刻会产出不同结果，测试与审计都失去确定性。
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from typing import Optional

from ..domain.errors import RuipinError, Unavailable

#: 通过率统计的最小样本量；低于此值一律视为"无法给出通过率"。
MIN_GROUP_SAMPLE = 10


class ReviewStatus(str, Enum):
    """题目的审校状态。"""

    DRAFT = "draft"
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    RETIRED = "retired"


#: 只允许以下流转。关键约束：`APPROVED` **只能**来自 `PENDING_REVIEW`
#: （不存在 DRAFT→APPROVED 的捷径，否则"先审后发"就不成立了）；
#: `RETIRED` 是终态——下架的题不会再被自动放回。
_ALLOWED_TRANSITIONS: dict[ReviewStatus, frozenset[ReviewStatus]] = {
    ReviewStatus.DRAFT: frozenset({ReviewStatus.PENDING_REVIEW}),
    ReviewStatus.PENDING_REVIEW: frozenset({ReviewStatus.APPROVED, ReviewStatus.REJECTED}),
    ReviewStatus.REJECTED: frozenset({ReviewStatus.PENDING_REVIEW}),
    ReviewStatus.APPROVED: frozenset({ReviewStatus.RETIRED}),
    ReviewStatus.RETIRED: frozenset(),
}

#: `add()` 允许的入库状态。
#:
#: 为什么不接受 `APPROVED`：`transition()` 已经保证"APPROVED 必来自 PENDING_REVIEW"，
#: 但若 `add()` 能直接把 APPROVED 塞进库，这条不变量就退化成了**口头纪律**——
#: 任何人都能用一个 `add()` 绕开审校。历史迁移是**唯一**的合法例外，故单独开
#: `seed_approved()` 这个显式入口，并要求填 reason 留痕（见该方法 docstring）。
#: `REJECTED`/`RETIRED` 同样不放行：它们是流转的**结果**，不是初始状态。
_ADD_ALLOWED_STATUSES: frozenset[ReviewStatus] = frozenset(
    {ReviewStatus.DRAFT, ReviewStatus.PENDING_REVIEW}
)


class InvalidReviewTransition(RuipinError):
    """非法的审校状态流转。

    定义在 `safety.bank` 而非 `domain/errors.py`：它是本模块的局部契约，
    放在紧邻使用点处更易审计；同时仍是 `RuipinError` 子类，`isinstance` 判断不受影响。

    带结构化字段（题号/来源/目标/操作人），并实现 `__reduce__`，
    以便跨进程（worker 池、任务队列）传播后属性不丢。
    """

    def __init__(
        self,
        question_id: str,
        from_status: ReviewStatus,
        to_status: ReviewStatus,
        actor: str,
    ) -> None:
        self.question_id = question_id
        self.from_status = from_status
        self.to_status = to_status
        self.actor = actor
        # 用枚举的 .value 而不是枚举本身：`str(Enum)` 的输出随 Python 版本变化，
        # 而这条消息会进日志检索与断言，必须稳定。
        super().__init__(
            f"非法审校流转：{from_status.value} → {to_status.value}"
            f"（题号 {question_id}，操作人 {actor}）"
        )

    def __reduce__(self):
        return (self.__class__, (self.question_id, self.from_status, self.to_status, self.actor))


@dataclass(frozen=True)
class QuestionBankItem:
    """一道题的某个版本。

    `version` 从 1 起；同一道题的新版本使用**同一** `question_id` + 递增 `version`，
    靠 `QuestionBank.revise` 维护，不要用新 `question_id` 表示"同一题的新版本"，
    否则历史链会断（`history()` 查不到）。
    """

    question_id: str
    text: str
    version: int
    author: str
    review_status: ReviewStatus
    job_family: str
    difficulty: float
    tags: frozenset[str]
    created_at: int

    def __post_init__(self) -> None:
        if not self.question_id or not self.question_id.strip():
            raise ValueError("question_id 不能为空")
        if not self.text or not self.text.strip():
            raise ValueError("题目正文不能为空")
        if self.version < 1:
            raise ValueError(f"version 必须 ≥ 1，收到 {self.version}")
        if not isinstance(self.difficulty, (int, float)):
            raise TypeError(f"difficulty 必须为数值，收到 {type(self.difficulty).__name__}")
        if not (0.0 <= float(self.difficulty) <= 1.0):
            raise ValueError(f"difficulty 越界 [0,1]：{self.difficulty}")


class QuestionBank:
    """题库。持有每道题的**当前版本**，并保留全部历史版本。"""

    def __init__(self, clock: Callable[[], float]) -> None:
        if not callable(clock):
            raise TypeError("clock 必须是可调用对象（返回秒）")
        self._clock = clock
        self._current: dict[str, QuestionBankItem] = {}
        self._history: dict[str, list[QuestionBankItem]] = {}

    # ---------- 增删改 ----------

    def add(self, item: QuestionBankItem) -> None:
        """入库一道**尚未审定**的题。

        只接受 `DRAFT` / `PENDING_REVIEW`。传 `APPROVED`/`REJECTED`/`RETIRED` 一律报错：
        前者会绕开"先审后发"，后者是流转结果而非初始状态。历史已审题库的迁移请走
        `seed_approved()`，普通入库后走 `transition()` 逐级流转。
        """
        if not isinstance(item, QuestionBankItem):
            raise TypeError(f"item 必须是 QuestionBankItem，收到 {type(item).__name__}")
        if item.question_id in self._current:
            raise ValueError(
                f"题号已存在：{item.question_id}（同一题的新版本请用 revise，而不是换题号）"
            )
        if item.review_status not in _ADD_ALLOWED_STATUSES:
            raise ValueError(
                f"已审状态的题不能直接入库（题号 {item.question_id}，"
                f"状态 {item.review_status.value}）：已审题必须由 transition() 逐级流转产生，"
                "历史迁移请用 seed_approved() 并填写 reason"
            )
        self._current[item.question_id] = item
        self._history[item.question_id] = [item]

    def seed_approved(
        self,
        items: Sequence[QuestionBankItem],
        *,
        actor: str,
        reason: str,
    ) -> int:
        """批量迁移**历史已审**题库（唯一的 APPROVED 直接入库通道）。

        为什么单独开这个入口而不是给 `add()` 加开关：非法路径与合法例外必须长得不一样。
        `add(APPROVED)` 被拒、`seed_approved(...)` 放行，这条差异本身就是审计线索；
        如果只是给 add 加个 `allow_approved=True` 参数，绕开审校的成本和正常入库一样低。

        为什么要求 `reason` 非空：迁移是"给库里灌一批没人重新审过的题"，必须留下依据
        （迁移单号、来源系统、审批记录），否则事后无法解释这些题凭什么直接是已审态。

        留痕方式：把 `seeded:by=<actor>` 与 `seeded:reason=<reason>` 追加进 `tags`。
        选 tags 而非新增字段，是因为 tags 已是 `frozenset[str]` 的可扩展多值集合，
        不改变值对象签名、不影响既有构造点，且天然随 `revise()` 继承到新版本——
        审计时从任意版本都能回溯到"这题是灌进来的"。代价是 tags 里混入了非业务标签，
        故统一加 `seeded:` 前缀以便过滤。

        校验一律不放宽：仍走 `QuestionBankItem` 的难度范围校验（`replace` 会重跑
        `__post_init__`），并逐个检查 `question_id` 重复。`items` 为空返回 0（不是异常）。
        返回成功导入的条数。
        """
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(
                "seed_approved 需要非空的 reason：历史迁移必须有据可查，不允许匿名灌数据"
            )
        if not isinstance(actor, str) or not actor.strip():
            raise ValueError("seed_approved 需要非空的 actor：迁移操作人不可为空，否则无法追溯")

        items = tuple(items)
        if not items:
            return 0

        seeded_count = 0
        for item in items:
            if not isinstance(item, QuestionBankItem):
                raise TypeError(f"item 必须是 QuestionBankItem，收到 {type(item).__name__}")
            if item.question_id in self._current:
                raise ValueError(f"题号已存在：{item.question_id}（seed_approved 不覆盖既有题）")
            if item.review_status is not ReviewStatus.APPROVED:
                raise ValueError(
                    f"seed_approved 只迁移已审题目；题号 {item.question_id} "
                    f"状态为 {item.review_status.value}"
                )
            seeded = replace(
                item,
                tags=item.tags
                | {f"seeded:by={actor}", f"seeded:reason={reason}"},
            )
            self._current[item.question_id] = seeded
            self._history[item.question_id] = [seeded]
            seeded_count += 1
        return seeded_count

    def revise(
        self,
        question_id: str,
        *,
        text: Optional[str] = None,
        difficulty: Optional[float] = None,
        author: str,
    ) -> QuestionBankItem:
        """产出 `version+1` 的**新条目**并替换当前版本。

        两条不可省略的后果：
        - 新版本状态强制 `PENDING_REVIEW`——改过内容就必须重审；
        - 旧版本转 `RETIRED`，仍可在 `history()` 里查到（审计要求可追溯）。
        """
        cur = self._require(question_id)
        retired = replace(cur, review_status=ReviewStatus.RETIRED)
        self._history[question_id][-1] = retired

        new_item = QuestionBankItem(
            question_id=question_id,
            text=cur.text if text is None else text,
            version=cur.version + 1,
            author=author,
            review_status=ReviewStatus.PENDING_REVIEW,
            job_family=cur.job_family,
            difficulty=cur.difficulty if difficulty is None else difficulty,
            tags=cur.tags,
            created_at=int(self._clock()),
        )
        self._current[question_id] = new_item
        self._history[question_id].append(new_item)
        return new_item

    def transition(self, question_id: str, to: ReviewStatus, actor: str) -> QuestionBankItem:
        """按 `_ALLOWED_TRANSITIONS` 流转审校状态；非法流转抛 `InvalidReviewTransition`。"""
        cur = self._require(question_id)
        if not isinstance(to, ReviewStatus):
            try:
                to = ReviewStatus(to)  # type: ignore[arg-type]
            except ValueError as exc:
                raise ValueError(f"未知的审校状态：{to!r}") from exc
        if to not in _ALLOWED_TRANSITIONS[cur.review_status]:
            raise InvalidReviewTransition(question_id, cur.review_status, to, actor)
        updated = replace(cur, review_status=to)
        self._current[question_id] = updated
        self._history[question_id][-1] = updated
        return updated

    # ---------- 查询 ----------

    def get(self, question_id: str) -> Optional[QuestionBankItem]:
        """取当前版本；不存在返回 `None`。"""
        return self._current.get(question_id)

    def history(self, question_id: str) -> tuple[QuestionBankItem, ...]:
        """全部版本，按 version 升序。"""
        versions = self._history.get(question_id)
        if versions is None:
            raise ValueError(f"题号不存在：{question_id}")
        return tuple(sorted(versions, key=lambda it: it.version))

    def selectable(
        self,
        job_family: Optional[str] = None,
        max_difficulty: Optional[float] = None,
    ) -> tuple[QuestionBankItem, ...]:
        """可供候选人使用的题目。

        只出 `APPROVED` 的**当前版本**：`RETIRED`/`DRAFT`/`PENDING_REVIEW`/`REJECTED`
        一律不返回——它们是"审校中"或"已下架"的内部状态，不应出现在面试里。
        默认按 question_id 升序，保证抽取结果可复现。
        """
        out: list[QuestionBankItem] = []
        for item in self._current.values():
            if item.review_status is not ReviewStatus.APPROVED:
                continue
            if job_family is not None and item.job_family != job_family:
                continue
            if max_difficulty is not None and item.difficulty > max_difficulty:
                continue
            out.append(item)
        out.sort(key=lambda it: it.question_id)
        return tuple(out)

    def assert_no_physio_question(self) -> None:
        """断言题库中不存在直接询问生理/健康/病史的题。

        为什么整类禁掉：这类题同时踩中"健康隐私"与"医疗建议"两条线，风险不可控，
        所以不做逐题判断，直接拒绝入库。命中时抛出并**列出全部**命中题号，
        方便一次性清理而不是反复试错。
        """
        flagged = sorted(it.question_id for it in self._current.values() if _looks_physio_probing(it.text))
        if flagged:
            raise ValueError(
                "题库含直接询问生理/健康/病史的题目，必须移除或改写："
                + "、".join(flagged)
            )

    def __len__(self) -> int:
        return len(self._current)

    def _require(self, question_id: str) -> QuestionBankItem:
        cur = self._current.get(question_id)
        if cur is None:
            raise ValueError(f"题号不存在：{question_id}")
        return cur


# ---------- 生理/健康题检测 ----------

#: 生理/健康类词。**故意保守**：宁可多拦（人工再确认），不可漏放（提给候选人）。
_PHYSIO_TERM_RE = re.compile(
    r"病史|疾病|病症|生病|病|手术|服药|吃药|用药|药物|过敏|遗传|传染|残疾|"
    r"体检|血压|心率|睡眠|精神|心理|抑郁|焦虑|怀孕|生育|身体|健康"
)

#: "问向个人"的探测装置：是非问、经历问、以及明确的提问/讲述动词。
_PHYSIO_PROBE_RE = re.compile(
    r"是否|有没有|有没|得过|患有|患过|做过|吃过|服过|受过|"
    r"吗|？|\?|说明|介绍|说说|谈谈|描述|聊聊|分享|告诉|何时|多久|怎么样|感觉如何"
)

#: "怎么设计/实现"类的技术题即便出现 health 词，也不是在问候选人本人——
#: 例如"如何设计健康数据的存储"。除非它同时直接指向第二人称的健康状况。
_PHYSIO_HOWTO_RE = re.compile(r"如何|怎么|怎样|设计|架构|方案|实现|原理|算法|流程|系统|技术|标准")

#: 第二人称 + 健康/身体 —— 出现即视为在问候选人本人，how-to 排除不再适用。
_PHYSIO_PERSONAL_RE = re.compile(
    r"(?:你|您)的?(?:病史|疾病|病症|健康|身体|血压|心率|睡眠|精神|心理|抑郁|焦虑|怀孕|生育|残疾|过敏)"
)


def _looks_physio_probing(text: str) -> bool:
    """判定一道题是否在直接询问生理/健康/病史。

    规则（写清以便复核）：含生理/健康词，且含"问向个人"的探测装置；
    但若整体是 how-to/设计类技术题且未直接指向第二人称健康，则不算命中。
    """
    if not _PHYSIO_TERM_RE.search(text):
        return False
    if _PHYSIO_HOWTO_RE.search(text) and not _PHYSIO_PERSONAL_RE.search(text):
        return False
    return bool(_PHYSIO_PROBE_RE.search(text))


# ---------- 偏差审计 ----------


@dataclass(frozen=True)
class GroupStat:
    """单个分组的通过率统计。

    `pass_rate=None` 表示**样本不足**，不是一个"算出来恰好为 0"的通过率。
    这是本模块最容易被写错的地方，故用独立字段把它显式化。
    """

    group: str
    n: int
    passed: int
    pass_rate: Optional[float]
    sufficient: bool


class BiasAudit:
    """按分组统计通过率并做偏差检测。

    样本不足时的行为：`pass_rate_by_group` 返回的该组值为 `None`；
    `divergence` / `is_biased` 直接抛 `Unavailable`，**不返回 0.0 冒充结论**。
    """

    def __init__(self) -> None:
        self._stats: dict[str, GroupStat] = {}

    @property
    def stats(self) -> tuple[GroupStat, ...]:
        return tuple(self._stats[g] for g in sorted(self._stats))

    def pass_rate_by_group(
        self, records: Sequence[tuple[str, bool]]
    ) -> Mapping[str, Optional[float]]:
        """统计每个分组的通过率。

        `records` 为 `(group, passed)` 序列。样本数 < `MIN_GROUP_SAMPLE` 的组
        通过率返回 `None`（详见 `GroupStat`），调用方不得把 `None` 当 0 用。
        """
        counts: dict[str, list[int]] = {}
        for rec in records:
            if not isinstance(rec, (tuple, list)) or len(rec) != 2:
                raise ValueError(f"record 必须是 (group, passed) 二元组，收到 {rec!r}")
            group, passed = rec[0], rec[1]
            if not isinstance(group, str) or not group.strip():
                raise ValueError(f"分组名必须是非空字符串，收到 {group!r}")
            if not isinstance(passed, bool):
                # bool 是 int 的子类，这里显式拒绝 1/0：把"通过=1"混进来会让
                # 数据源把缺失值编码成 0 时被静默计入分母（假阴性来源）。
                raise TypeError(f"通过标记必须是 bool，收到 {type(passed).__name__}")
            bucket = counts.setdefault(group, [0, 0])
            bucket[0] += 1
            bucket[1] += 1 if passed else 0

        self._stats = {}
        for group, (n, passed) in counts.items():
            sufficient = n >= MIN_GROUP_SAMPLE
            self._stats[group] = GroupStat(
                group=group,
                n=n,
                passed=passed,
                pass_rate=(passed / n if sufficient else None),
                sufficient=sufficient,
            )
        return {g: s.pass_rate for g, s in sorted(self._stats.items())}

    def divergence(self, baseline_group: str, other: str) -> float:
        """两组通过率之差的绝对值。

        任一组样本不足（或不存在）→ 抛 `Unavailable`：差异是"算不出来"，
        而不是"算出来等于 0"。返回 0.0 会被下游误读成"两组完全一致"。
        """
        base = self._rate(baseline_group)
        other_rate = self._rate(other)
        return abs(base - other_rate)

    def is_biased(self, max_abs_diff: float = 0.15) -> bool:
        """是否存在任一对有效分组的通过率差超过 `max_abs_diff`。

        有效分组（样本足够）少于 2 个时无法比较 → 抛 `Unavailable`，
        不允许返回 `False`（那等于声称"已确认无偏差"）。
        """
        if max_abs_diff < 0:
            raise ValueError(f"max_abs_diff 不能为负：{max_abs_diff}")
        sufficient = [s for s in self._stats.values() if s.sufficient]
        if len(sufficient) < 2:
            raise Unavailable(
                "bias-audit",
                f"有效分组不足（{len(sufficient)} < 2），无法判定偏差；"
                "不得据此宣称“无偏差”",
            )
        rates = [s.pass_rate for s in sufficient]
        for i in range(len(rates)):
            for j in range(i + 1, len(rates)):
                if abs(rates[i] - rates[j]) > max_abs_diff:
                    return True
        return False

    def _rate(self, group: str) -> float:
        stat = self._stats.get(group)
        if stat is None:
            raise ValueError(f"分组不存在或尚未统计：{group}")
        if not stat.sufficient or stat.pass_rate is None:
            raise Unavailable(
                "bias-audit",
                f"分组 {group} 样本不足（n={stat.n} < {MIN_GROUP_SAMPLE}），无法给出通过率差异",
            )
        return stat.pass_rate


__all__ = [
    "ReviewStatus",
    "QuestionBankItem",
    "QuestionBank",
    "InvalidReviewTransition",
    "BiasAudit",
    "GroupStat",
    "MIN_GROUP_SAMPLE",
]
