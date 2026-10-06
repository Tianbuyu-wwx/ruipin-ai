"""实验分桶与灰度（方案 §12.5「A/B 实验与灰度发布」）。

设计意图
--------
方案要求会话级稳定分桶：同一个 `session_id` 在整场面试里必须始终落在同一变体，
否则用户会在"新评分口径 / 旧评分口径"之间反复横跳，实验结论和用户体验都会碎掉。
因此分桶必须是**纯函数**：只取决于 (salt, experiment_id, session_id)，不含随机数、
不含时间、不含进程状态。这样多进程 worker 池、重放、回归对拍才能得到同一结果。

用 `hashlib.sha256` 而不是内置 `hash()`：后者在 Python 3.3+ 默认按进程随机加盐
（`PYTHONHASHSEED`），进程重启就换桶，实验数据会被打散到两个变体上。

关键纪律：区分"未纳入实验"与"对照组"
------------------------------------
关闭实验、或 session 落在灰度之外时，本模块返回哨兵 `CONTROL`。它表示"这个
session 压根没被随机化纳入实验"，**不是**实验内的对照组。二者必须区分：

* 实验内的"对照组"是一个真实变体，它与处理组之间经过了随机化，可以比较；
* "未纳入"的流量是自选的（比如只放量给新用户），拿它当对照组等于用非随机样本
  做因果推断，会产生**选择偏差**——而且往往是系统性偏差，不是噪声。

所以 `CONTROL` 不允许被任何真实变体占用（构造期校验），避免"未纳入"的脏数据
被当对照组喂进统计。

另一个纪律：关闭实验**不得**让请求失败
--------------------------------------
`enabled=False` 时一律返回 `CONTROL`，即便 `variants` 配置非法也不抛异常——关闭
开关的本意就是"让这个实验彻底消失"，它不该成为线上请求的失败源。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Sequence

__all__ = [
    "CONTROL",
    "ExperimentSpec",
    "assign",
]

#: "未纳入实验"的哨兵变体。
#:
#: 语义是"这个 session 没有被随机化纳入实验"（实验关闭 / 落在灰度之外），
#: **不是**实验内对照组。理由见模块 docstring：把未纳入的流量当对照组会让
#: 因果比较建立在非随机样本上。取值用不可能与业务命名撞车的下划线串。
CONTROL = "__not_enrolled__"

#: 参与判定的灰度位宽（百分比取模）。抽成常量而不是散落的字面量 100，
#: 免得有人改了一处忘了另一处。
_ROLLOUT_MODULUS = 100


def _validate_variants(variants: Sequence[str]) -> None:
    """校验变体清单。独立成函数，供 `ExperimentSpec` 与 `assign` 共用。

    至少两个变体才有对照可言；`CONTROL` 是哨兵不是变体，禁止占用；
    重复变体会让"分布是否均匀"的统计失去意义，也一律拒绝。
    """
    if len(variants) < 2:
        raise ValueError(f"variants 至少需要 2 个变体才能做对照：{tuple(variants)!r}")
    if CONTROL in variants:
        raise ValueError(
            f"variants 不得包含哨兵值 {CONTROL!r}——它表示"
            f"「未纳入实验」，与实验内的对照组是两回事"
        )
    if len(set(variants)) != len(variants):
        raise ValueError(f"variants 存在重复项：{tuple(variants)!r}")


@dataclass(frozen=True)
class ExperimentSpec:
    """一个实验的预注册描述（分桶所需的全部信息）。

    `rollout_pct` 是**进入灰度的流量百分比**（0–100），不是"给处理组的比例"；
    进入灰度的 session 再按哈希在 `variants` 间均分。
    """

    experiment_id: str
    variants: tuple[str, ...]
    salt: str
    enabled: bool = True
    rollout_pct: int = 100

    def __post_init__(self) -> None:
        if not self.experiment_id:
            raise ValueError("experiment_id 不能为空")
        _validate_variants(self.variants)
        if not self.salt:
            # 空 salt 会让不同实验的哈希键退化成 (experiment_id, session_id)，
            # 同一批 session 在不同实验里高度相关，实验之间不独立。
            raise ValueError("salt 不能为空（空 salt 会让不同实验相互串桶）")
        if not 0 <= self.rollout_pct <= _ROLLOUT_MODULUS:
            raise ValueError(
                f"rollout_pct 必须落在 [0, {_ROLLOUT_MODULUS}]：{self.rollout_pct!r}"
            )

    def assign(self, session_id: str) -> str:
        """按本规格给一个 session 分桶（委托给模块级纯函数）。"""
        return assign(
            self.experiment_id,
            self.variants,
            session_id,
            self.salt,
            enabled=self.enabled,
            rollout_pct=self.rollout_pct,
        )


def _digest(salt: str, experiment_id: str, session_id: str) -> bytes:
    """稳定哈希键的唯一构造点。

    键里带 `salt`：不同实验即使 experiment_id 相同也能互不相关；
    带 `experiment_id`：同一 salt 下不同实验互不相关。
    """
    key = f"{salt}:{experiment_id}:{session_id}"
    return hashlib.sha256(key.encode("utf-8")).digest()


def _bucket(digest: bytes, offset: int) -> int:
    """从摘要里取 8 字节无符整数。

    用同一摘要的不同字节窗口分别做"是否进入灰度"和"分到哪个变体"两件事，
    避免两次取模用到同一个整数而在低流量下产生相关。
    """
    return int.from_bytes(digest[offset : offset + 8], "big")


def assign(
    experiment_id: str,
    variants: Sequence[str],
    session_id: str,
    salt: str,
    *,
    enabled: bool = True,
    rollout_pct: int = 100,
) -> str:
    """把一个 session 确定性地分到某个变体，返回变体名或 `CONTROL`。

    参数顺序与方案一致：`(experiment_id, variants, session_id, salt)`；
    `enabled` / `rollout_pct` 为关键字参数，默认"开启且全量"。

    返回 `CONTROL` 的两种情况（都不是实验内对照组）：
    实验被关闭；或该 session 的灰度位落在 `rollout_pct` 之外。
    """
    # 关闭实验优先且不抛错：开关的语义是"让实验消失"，不该制造线上失败。
    if not enabled:
        return CONTROL
    if not 0 <= rollout_pct <= _ROLLOUT_MODULUS:
        raise ValueError(
            f"rollout_pct 必须落在 [0, {_ROLLOUT_MODULUS}]：{rollout_pct!r}"
        )
    # 0% 灰度 = 谁都不参与；提前返回可以省掉后面的校验（但 variants 合法时也一样）。
    if rollout_pct <= 0:
        return CONTROL
    _validate_variants(variants)
    if not session_id:
        raise ValueError("session_id 不能为空")

    digest = _digest(salt, experiment_id, session_id)
    if _bucket(digest, 0) % _ROLLOUT_MODULUS >= rollout_pct:
        # 未进入灰度：显式"未纳入"，绝不误当成对照组。
        return CONTROL
    index = _bucket(digest, 8) % len(variants)
    return variants[index]
