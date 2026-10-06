"""实验预注册、护栏指标与停止规则（方案 §12.5）。

设计意图
--------
方案要求实验必须**预注册**：主指标、护栏指标、最小样本量、停止规则都要在开跑
之前写死，且"**护栏指标不满足即回滚，不论主指标多好**"。这条是本模块的核心
安全语义，落实为一个不可颠倒的判定顺序：

1. 样本量不足 → 不可判定（`INSUFFICIENT_DATA`）。样本不够时哪怕主指标"已经
   大涨"也不许下结论——那不是效果，是噪声。
2. 任一护栏被突破 → 回滚（`ROLLBACK`）。护栏是红线（关闭率、申诉率），
   主指标涨得再多也不能拿红线换。**必须列出全部被突破的护栏**，不能只报第一条，
   否则修完第一条又踩第二条。
3. 主指标达到目标 → 上线（`SHIP`）。目标的口径由是否指定基线决定：无基线用
   绝对目标值，有基线用相对该基线的抬升比例（见 `ExperimentRegistration`）。
4. 否则 → 继续观察（`CONTINUE`）。

为什么"护栏优先"：主指标是"我想要的"，护栏是"我绝对不能付出的代价"。两者不是
同一个维度的加减法，不能相抵；一旦允许"主指标好就吞掉护栏告警"，护栏就形同虚设。

另一个纪律：数据缺失不放过
--------------------------
注册了某个护栏，但观测里没有它的值 → `INSUFFICIENT_DATA`，而不是"跳过这条护栏"。
跳过会让"没上报"和"没超标"变得无法区分，是最隐蔽的一类静默失效。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Optional

from ruipin.domain.errors import RuipinError

__all__ = [
    "GUARDRAIL_APPEAL_RATE",
    "GUARDRAIL_CLOSED_RATE",
    "Decision",
    "ExperimentRegistration",
    "ExperimentRegistry",
    "GuardrailBreach",
    "GuardrailDirection",
    "GuardrailMetric",
    "MissingBaselineVariant",
    "StopRule",
    "UnknownExperiment",
    "VariantObservation",
    "Verdict",
]

#: 便捷护栏名：会话关闭率（越低越好）。
GUARDRAIL_CLOSED_RATE = "closed_rate"
#: 便捷护栏名：申诉率（越低越好）。
GUARDRAIL_APPEAL_RATE = "appeal_rate"


class GuardrailDirection(str, Enum):
    """护栏方向：越低越好（MAX，给上限）还是越高越好（MIN，给下限）。"""

    MAX = "max"
    MIN = "min"


@dataclass(frozen=True)
class GuardrailMetric:
    """一条护栏指标。

    方向与阈值字段必须一一对应：`MAX` 只填 `max_allowed`，`MIN` 只填
    `min_required`。两项都填或都不填都抛错——"两条阈值并存"会让"到底以哪条为准"
    变得不可判定，而 NaN 式配置正是事故温床。
    """

    name: str
    direction: GuardrailDirection
    max_allowed: Optional[float] = None
    min_required: Optional[float] = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("护栏指标 name 不能为空")
        # 归一化：允许传字符串 "max"/"min"，但内部一律以枚举存储，比较才可靠。
        object.__setattr__(self, "direction", GuardrailDirection(self.direction))
        if self.direction is GuardrailDirection.MAX:
            if self.max_allowed is None or self.min_required is not None:
                raise ValueError(
                    f"MAX 方向护栏 {self.name!r} 必须且只能填 max_allowed："
                    f"max_allowed={self.max_allowed!r}, min_required={self.min_required!r}"
                )
        else:
            if self.min_required is None or self.max_allowed is not None:
                raise ValueError(
                    f"MIN 方向护栏 {self.name!r} 必须且只能填 min_required："
                    f"min_required={self.min_required!r}, max_allowed={self.max_allowed!r}"
                )

    @property
    def threshold(self) -> float:
        """当前方向对应的阈值。"""
        if self.direction is GuardrailDirection.MAX:
            return self.max_allowed  # type: ignore[return-value]
        return self.min_required  # type: ignore[return-value]

    def breached_by(self, value: float) -> Optional[float]:
        """返回突破阈值的幅度（恒为正数）；未突破返回 `None`。

        幅度有用：告警里直接写"超了多少"比"已突破"信息量大，省得回代码里推。
        """
        if self.direction is GuardrailDirection.MAX:
            if value > self.threshold:
                return value - self.threshold
            return None
        if value < self.threshold:
            return self.threshold - value
        return None


@dataclass(frozen=True)
class StopRule:
    """停止规则：最多观察多少天、主指标要抬到多少才值得上线。

    `target_lift` 的语义**取决于注册是否指定了基线**（见
    `ExperimentRegistration.baseline_variant`）：无基线时是主指标的绝对目标值，
    有基线时是相对该基线的抬升比例。这是全模块最容易误用的一处，故 `decide()`
    会在 `reason` 里回显本次用的是哪种口径。
    """

    max_days: int
    target_lift: float

    def __post_init__(self) -> None:
        if self.max_days < 1:
            raise ValueError(f"max_days 必须 ≥ 1：{self.max_days!r}")
        if not self.target_lift > 0:
            raise ValueError(f"target_lift 必须为正数：{self.target_lift!r}")


@dataclass(frozen=True)
class ExperimentRegistration:
    """一份实验的预注册记录。

    `guardrails` **允许为空**——有些实验确实没有可设护栏的指标。但空护栏意味着
    "这个实验没有红线保护"，调用方必须能看见这一点，所以提供 `has_guardrails`，
    而不是让空护栏偷偷等于"有护栏且全过"。

    `baseline_variant` 决定主指标的口径（最容易误用的一处，故在此写清）：

    * 为 `None` —— 绝对口径：`StopRule.target_lift` 是主指标要**达到的目标值**，
      最优变体 `primary_metric >= target_lift` 即 SHIP。
    * 非 `None` —— 相对口径：`StopRule.target_lift` 是**相对该基线变体的抬升比例**，
      用"除基线外最优变体"计算 `(best - base) / abs(base)`。相对 lift 才是 A/B 的
      标准口径，但"相对谁"必须显式指定——本字段把它从隐含歧义变成显式声明。

    该字段是单个 `Optional[str]`，因此"一个实验只能有一个基线"由类型天然保证；
    基线是否真的出现在观测里，在 `decide()` 运行时校验（注册时注册表还不知道
    变体集合）。
    """

    experiment_id: str
    primary_metric: str
    guardrails: tuple[GuardrailMetric, ...]
    min_sample_per_variant: int
    stop_rule: StopRule
    baseline_variant: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.experiment_id:
            raise ValueError("experiment_id 不能为空")
        if not self.primary_metric:
            raise ValueError("primary_metric 不能为空")
        if self.baseline_variant is not None and not self.baseline_variant:
            raise ValueError("baseline_variant 不能为空字符串：要么 None（绝对口径），要么给出变体名")

    @property
    def has_guardrails(self) -> bool:
        """是否配置了护栏。空护栏是合法状态，但必须显式可见。"""
        return len(self.guardrails) > 0


@dataclass(frozen=True)
class VariantObservation:
    """某个变体到当前为止的观测汇总。

    `guardrails` 是按护栏名的实测值映射。`closed_rate` / `appeal_rate` 是对两个
    常用护栏名的便捷写法——**默认 `None`（未提供）而不是 0.0**：0.0 会被读成
    "关闭率为零，好极了"，属于用合成值冒充成功，违反第一纪律。
    """

    n: int
    primary_metric: float
    guardrails: Mapping[str, float] = field(default_factory=dict)
    closed_rate: Optional[float] = None
    appeal_rate: Optional[float] = None

    def __post_init__(self) -> None:
        if self.n < 0:
            raise ValueError(f"样本数 n 不能为负：{self.n!r}")


class UnknownExperiment(RuipinError, KeyError):
    """查询/判定了一个未预注册的实验。

    继承 `KeyError`（语义就是查表未命中），但重写 `__str__` 以免被 KeyError
    默认的 repr 加引号；消息里带上已注册清单，排障不用回代码翻。
    """

    def __init__(self, experiment_id: str, known: tuple[str, ...] = ()) -> None:
        self.experiment_id = experiment_id
        self.known = known
        msg = f"实验未预注册: {experiment_id!r}"
        if known:
            msg += f"（已注册: {', '.join(known)}）"
        self._msg = msg
        super().__init__(msg)

    def __str__(self) -> str:
        return self._msg

    def __reduce__(self):
        return (self.__class__, (self.experiment_id, self.known))


class MissingBaselineVariant(RuipinError):
    """相对口径下，指定的基线变体没有出现在本次观测里。

    这是**调用方/配置错误**（观测集不完整或基线名拼错），必须显式抛出，绝不悄悄
    退回绝对口径——那样结论会用一个没人指定过的口径算出来，事后无从追溯。
    """

    def __init__(self, baseline_variant: str, observed: tuple[str, ...] = ()) -> None:
        self.baseline_variant = baseline_variant
        self.observed = observed
        msg = f"基线变体 {baseline_variant!r} 未出现在观测里"
        if observed:
            msg += f"（当前观测变体: {', '.join(observed)}）"
        super().__init__(msg)

    def __reduce__(self):
        return (self.__class__, (self.baseline_variant, self.observed))


class Verdict(str, Enum):
    """实验判定结论。`INSUFFICIENT_DATA` 是一等公民：样本不足绝不是"继续观察"。"""

    SHIP = "ship"
    ROLLBACK = "rollback"
    CONTINUE = "continue"
    INSUFFICIENT_DATA = "insufficient_data"


@dataclass(frozen=True)
class GuardrailBreach:
    """一条被突破的护栏。`excess` 是突破幅度（正数），方便告警直接引用。"""

    variant: str
    name: str
    direction: GuardrailDirection
    value: float
    threshold: float
    excess: float


@dataclass(frozen=True)
class Decision:
    """一次判定的结果。

    `samples_needed` 只在 `INSUFFICIENT_DATA` 且是样本量不足时非空，形如
    `{变体名: 还差的样本数}`；其余结论为空映射。
    """

    verdict: Verdict
    triggered: tuple[GuardrailBreach, ...]
    reason: str
    samples_needed: Mapping[str, int] = field(default_factory=dict)


def _guardrail_value(obs: VariantObservation, name: str) -> Optional[float]:
    """从观测里取护栏值；取不到返回 `None`（表示缺测，不是 0）。"""
    value = obs.guardrails.get(name)
    if value is not None:
        return value
    # 便捷字段只在**显式提供**时兜底；None 表示"没上报"，继续往下返回 None。
    if name == GUARDRAIL_CLOSED_RATE:
        return obs.closed_rate
    if name == GUARDRAIL_APPEAL_RATE:
        return obs.appeal_rate
    return None


class ExperimentRegistry:
    """实验注册表 + 判定器。

    非线程安全（与 v2 其他内存实现一致：编排层单事件循环内使用）。
    """

    def __init__(self) -> None:
        self._registrations: dict[str, ExperimentRegistration] = {}
        #: 人工急停：experiment_id → 原因。一旦置入，decide 一律 ROLLBACK。
        self._forced_rollbacks: dict[str, str] = {}

    # ----- 注册 -----

    def register(self, reg: ExperimentRegistration) -> None:
        """登记一份预注册。

        预注册是**一次性的**：重复 experiment_id 会被拒绝，避免"改了配置再注册一遍"
        把先前的口径悄悄覆盖掉——那样实验中途换口径，前后数据不可比却无人知晓。
        """
        if reg.experiment_id in self._registrations:
            raise ValueError(
                f"实验 {reg.experiment_id!r} 已注册，禁止覆盖预注册（改口径需新建实验）"
            )
        if reg.min_sample_per_variant < 1:
            raise ValueError(
                f"min_sample_per_variant 必须 ≥ 1：{reg.min_sample_per_variant!r}"
            )
        names = [g.name for g in reg.guardrails]
        if len(set(names)) != len(names):
            raise ValueError(f"护栏指标重名，无法区分：{sorted(names)!r}")
        self._registrations[reg.experiment_id] = reg

    def get(self, experiment_id: str) -> ExperimentRegistration:
        """取预注册；未注册抛 `UnknownExperiment`。"""
        try:
            return self._registrations[experiment_id]
        except KeyError as exc:
            raise UnknownExperiment(experiment_id, tuple(self._registrations)) from exc

    def has_guardrails(self, experiment_id: str) -> bool:
        """该实验是否配置了护栏（空护栏是合法但需要被看见的状态）。"""
        return self.get(experiment_id).has_guardrails

    @property
    def experiment_ids(self) -> tuple[str, ...]:
        """已注册实验的 id（登记顺序）。"""
        return tuple(self._registrations)

    # ----- 人工急停 -----

    def force_rollback(self, experiment_id: str, reason: str) -> None:
        """人工急停：之后无论统计结论多好，`decide` 一律 ROLLBACK。

        急停优先于一切统计结论——统计有滞后、有噪声，而人也可能已经看到了图上
        还没体现出来的线下事故（舆情、投诉）。重复调用保留最新原因。
        """
        self.get(experiment_id)  # 未注册直接抛 UnknownExperiment
        if not reason:
            raise ValueError("人工急停必须写明原因（否则无人知道为什么被停）")
        self._forced_rollbacks[experiment_id] = reason

    def is_forced_rollback(self, experiment_id: str) -> bool:
        return experiment_id in self._forced_rollbacks

    def forced_rollback_reason(self, experiment_id: str) -> Optional[str]:
        return self._forced_rollbacks.get(experiment_id)

    # ----- 判定 -----

    def decide(
        self, experiment_id: str, observations: Mapping[str, VariantObservation]
    ) -> Decision:
        """按"样本 → 护栏 → 主指标 → 继续"的固定顺序给出结论。

        顺序本身就是安全语义，不可重排（见模块 docstring）。

        相对口径（`baseline_variant` 非空）下，基线必须出现在 `observations` 里，
        否则抛 `MissingBaselineVariant`——这是配置/调用错误，不退回绝对口径。
        """
        reg = self.get(experiment_id)

        # 人工急停凌驾于一切统计之上，先于任何计算返回。
        forced = self._forced_rollbacks.get(experiment_id)
        if forced is not None:
            return Decision(
                verdict=Verdict.ROLLBACK,
                triggered=(),
                reason=f"人工急停优先于统计结论，直接回滚：{forced}",
            )

        # 基线缺失是配置错误，先于统计判定拦下（不得悄悄退回绝对口径）。
        baseline = reg.baseline_variant
        if baseline is not None and baseline not in observations:
            raise MissingBaselineVariant(baseline, tuple(observations))

        if not observations:
            return Decision(
                verdict=Verdict.INSUFFICIENT_DATA,
                triggered=(),
                reason="没有任何变体的观测，无法判定",
            )

        # 1) 样本量：不足即不可判定，主指标再好看也不提前下结论。
        samples_needed = {
            variant: reg.min_sample_per_variant - obs.n
            for variant, obs in observations.items()
            if obs.n < reg.min_sample_per_variant
        }
        if samples_needed:
            detail = "，".join(
                f"{variant} 还差 {need} 例" for variant, need in sorted(samples_needed.items())
            )
            return Decision(
                verdict=Verdict.INSUFFICIENT_DATA,
                triggered=(),
                reason=(
                    f"样本不足（每个变体至少 {reg.min_sample_per_variant} 例）：{detail}；"
                    "样本不够一律不可判定，主指标再好看也不提前下结论"
                ),
                samples_needed=samples_needed,
            )

        # 1b) 数据缺失：注册了护栏却没上报其值，同样不可判定（绝不跳过）。
        missing = sorted(
            f"{variant}.{g.name}"
            for variant, obs in observations.items()
            for g in reg.guardrails
            if _guardrail_value(obs, g.name) is None
        )
        if missing:
            return Decision(
                verdict=Verdict.INSUFFICIENT_DATA,
                triggered=(),
                reason=f"护栏指标缺测，不能判定（缺测 ≠ 未超标）：{'，'.join(missing)}",
            )

        # 2) 护栏：全部收集，不被第一条短路。
        breaches = self._collect_breaches(reg, observations)
        if breaches:
            return Decision(
                verdict=Verdict.ROLLBACK,
                triggered=breaches,
                reason="护栏被突破，必须回滚（护栏优先于主指标）："
                + "；".join(self._describe_breach(b) for b in breaches),
            )

        # 3)+4) 主指标：按是否指定基线，分别用绝对 / 相对口径判定。
        return self._evaluate_primary(reg, observations)

    @staticmethod
    def _evaluate_primary(
        reg: ExperimentRegistration, observations: Mapping[str, VariantObservation]
    ) -> Decision:
        """主指标判定：口径由 `baseline_variant` 分派，`reason` 必写明用的是哪种。"""
        target = reg.stop_rule.target_lift
        baseline = reg.baseline_variant

        if baseline is None:
            # 绝对口径：最优变体的主指标要直接达到预注册目标值。
            best_variant, best_obs = max(
                observations.items(), key=lambda item: item[1].primary_metric
            )
            caliber = (
                f"绝对口径：最优变体 {best_variant}={best_obs.primary_metric:.4g}，"
                f"目标值 {target:.4g}"
            )
            if best_obs.primary_metric >= target:
                return Decision(
                    verdict=Verdict.SHIP,
                    triggered=(),
                    reason=f"主指标 {reg.primary_metric} 达标（{caliber}）",
                )
            return Decision(
                verdict=Verdict.CONTINUE,
                triggered=(),
                reason=(
                    f"未达停止规则（{caliber}），继续观察"
                    f"（上限 {reg.stop_rule.max_days} 天）"
                ),
            )

        # 相对口径：排除基线自身，用"其余变体里最优的那个"算相对抬升。
        candidates = {v: o for v, o in observations.items() if v != baseline}
        if not candidates:
            return Decision(
                verdict=Verdict.INSUFFICIENT_DATA,
                triggered=(),
                reason=(
                    f"相对口径：除基线 {baseline!r} 外没有其它变体，无法计算相对抬升"
                ),
            )
        best_variant, best_obs = max(
            candidates.items(), key=lambda item: item[1].primary_metric
        )
        base_value = observations[baseline].primary_metric
        if base_value == 0:
            # 除零在数学上无定义；用 inf 冒充"达标"正是最危险的一类合成值。
            return Decision(
                verdict=Verdict.INSUFFICIENT_DATA,
                triggered=(),
                reason=(
                    f"相对口径：基线 {baseline!r} 的主指标为 0，相对抬升无定义"
                    f"（拒绝用 inf 冒充达标）"
                ),
            )
        lift = (best_obs.primary_metric - base_value) / abs(base_value)
        caliber = (
            f"相对口径：相对基线 {baseline!r} 的抬升 {lift:.4g}，"
            f"最优变体 {best_variant}={best_obs.primary_metric:.4g}，"
            f"基线值 {base_value:.4g}，目标抬升 {target:.4g}"
        )
        if lift >= target:
            return Decision(
                verdict=Verdict.SHIP,
                triggered=(),
                reason=f"主指标 {reg.primary_metric} 达标（{caliber}）",
            )
        return Decision(
            verdict=Verdict.CONTINUE,
            triggered=(),
            reason=(
                f"未达停止规则（{caliber}），继续观察（上限 {reg.stop_rule.max_days} 天）"
            ),
        )

    @staticmethod
    def _collect_breaches(
        reg: ExperimentRegistration, observations: Mapping[str, VariantObservation]
    ) -> tuple[GuardrailBreach, ...]:
        """收集**全部**被突破的护栏（跨变体、跨指标）。"""
        breaches: list[GuardrailBreach] = []
        for variant, obs in observations.items():
            for g in reg.guardrails:
                value = _guardrail_value(obs, g.name)
                # 走到这里已保证 value 非空（缺失在 decide 里提前拦下）。
                excess = g.breached_by(value)  # type: ignore[arg-type]
                if excess is not None:
                    breaches.append(
                        GuardrailBreach(
                            variant=variant,
                            name=g.name,
                            direction=g.direction,
                            value=value,  # type: ignore[arg-type]
                            threshold=g.threshold,
                            excess=excess,
                        )
                    )
        return tuple(breaches)

    @staticmethod
    def _describe_breach(breach: GuardrailBreach) -> str:
        relation = "≤" if breach.direction is GuardrailDirection.MAX else "≥"
        return (
            f"{breach.variant}.{breach.name}={breach.value:.4g} "
            f"（应 {relation} {breach.threshold:.4g}，超出 {breach.excess:.4g}）"
        )
