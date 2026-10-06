"""角色与权限矩阵（方案 §12.1 的落地）。

这一层回答的问题是："**谁**可以对这个**会话/租户/数据类别**做**什么**"。
它是六边形架构里的核心域逻辑——不依赖任何适配器，只依赖 `domain.errors`。

安全边界（为什么这样设计）
--------------------------
1. **默认拒绝**：`authorize()` 只有在*全部*条件通过时才返回 `None`，任何一项不满足
   都抛 `Forbidden`。权限表按"白名单"枚举，未列出的权限一律视为没有——
   漏配一个权限的后果是"访问被拒"，而不是"数据泄露"。
2. **多租户隔离是硬线**：只要请求带了与主体不一致的 `tenant_id`，一律拒绝，
   且这条检查对**所有**权限生效。跨租户绝不能靠调用方自觉。
3. **生理数据三重门**（§12.1："生理数据单独授权位：即使招聘方也默认不可见，
   需候选人二次授权"）：读 `PHYSIO_READ` 必须同时满足
   ① 该角色本身拥有该权限（管理员**没有**，所以管理员无论如何都读不到）、
   ② 候选人已二次授权（`physio_second_grant=True`）、
   ③ 请求方与数据**同租户**。
   这里"安全优先于可用"：缺任何一条都直接拒绝，且错误信息明确点名缺的是哪一条，
   避免排障时看不到根因。放弃"先给个默认值/给个空结果"的所谓容错——那等于绕过门禁。
4. **原始媒体另设权限位**（`RAW_MEDIA_READ`）：管理员和招聘方都没有这个位。
   方案明说"无候选人原始媒体访问权"，因此不靠"用途审查"来约束，直接不给权。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping, Optional

from ..domain.errors import RuipinError


class Role(str, Enum):
    """四类主体（方案 §12.1 的表格）。

    继承 `str` 是为了让角色值能直接进 JSON/JWT 载荷而不必额外转码；
    这里用 `Role.CANDIDATE.value` 而非 `str(Role.CANDIDATE)` 拼装对外文本——
    `str(Enum)` 的输出随 Python 版本变化，不能进日志检索或断言。
    """

    CANDIDATE = "candidate"
    RECRUITER = "recruiter"
    ADMIN = "admin"
    SYSTEM = "system"


# ---------- 权限常量 ----------
#
# 用 `资源:动作` 的字符串而非裸字符串散落各处：拼写错误会立刻变成 KeyError/不匹配，
# 而不是两个看起来不同的字符串被当成同一权限。

SESSION_READ = "session:read"
SESSION_WRITE = "session:write"
REPORT_READ = "report:read"
#: 读生理（压力调节）派生数据。**额外受三重门约束**，见 `authorize`。
PHYSIO_READ = "physio:read"
#: 读候选人原始音视频/媒体。方案要求"不落盘"，此位仅为防御性存在——没有任何角色持有。
RAW_MEDIA_READ = "raw_media:read"
CONFIG_WRITE = "config:write"
BANK_WRITE = "bank:write"
AUDIT_READ = "audit:read"
DATA_DELETE = "data:delete"
CONSENT_REVOKE = "consent:revoke"

#: 全部已定义权限（用于测试穷举与自检，不用于放行判断）。
ALL_PERMISSIONS: frozenset[str] = frozenset(
    {
        SESSION_READ,
        SESSION_WRITE,
        REPORT_READ,
        PHYSIO_READ,
        RAW_MEDIA_READ,
        CONFIG_WRITE,
        BANK_WRITE,
        AUDIT_READ,
        DATA_DELETE,
        CONSENT_REVOKE,
    }
)

#: 权限矩阵。改动此表即改动授权语义，务必同步 §12.1 与测试。
#:
#: - 候选人：仅本人会话 + 撤回生理授权 + 删除本人数据（报告只读自己的）。
#: - 招聘方：本企业会话与报告 + 生理（**受三重门**，默认不可见）。
#: - 管理员：配置/题库/审计；**无** `PHYSIO_READ`、**无** `RAW_MEDIA_READ`。
#: - 系统：服务间调用的最小权限，只够跑一场会话，读不到报告/生理/审计。
_ROLE_PERMISSIONS: dict[Role, frozenset[str]] = {
    Role.CANDIDATE: frozenset(
        {SESSION_READ, SESSION_WRITE, REPORT_READ, DATA_DELETE, CONSENT_REVOKE}
    ),
    Role.RECRUITER: frozenset({SESSION_READ, REPORT_READ, PHYSIO_READ}),
    Role.ADMIN: frozenset({CONFIG_WRITE, BANK_WRITE, AUDIT_READ}),
    Role.SYSTEM: frozenset({SESSION_READ, SESSION_WRITE}),
}

#: 对外暴露的**冻结**视图。用 `MappingProxyType` 而不是直接给 dict：
#: 任何 `ROLE_PERMISSIONS[Role.ADMIN] = ...` 都会立即 `TypeError`，
#: 杜绝运行时被悄悄提权（权限表必须是编译期常量级的不变量）。
ROLE_PERMISSIONS: Mapping[Role, frozenset[str]] = MappingProxyType(_ROLE_PERMISSIONS)


class Forbidden(RuipinError):
    """授权失败。

    `reason` 必须说清"缺的是哪一条"（权限位 / 租户 / 会话 / 三重门），
    `permission` / `role` / `target_tenant` 为可选结构化字段，便于审计检索。
    带结构化字段的异常要实现 `__reduce__`，否则跨进程（worker 池/日志采集）会丢字段。
    """

    def __init__(
        self,
        reason: str,
        permission: Optional[str] = None,
        role: Optional[str] = None,
        target_tenant: Optional[str] = None,
    ) -> None:
        self.reason = reason
        self.permission = permission
        self.role = role
        self.target_tenant = target_tenant
        msg = f"forbidden: {reason}"
        if permission is not None:
            msg += f"（权限 {permission}）"
        super().__init__(msg)

    def __reduce__(self):
        return (
            self.__class__,
            (self.reason, self.permission, self.role, self.target_tenant),
        )


@dataclass(frozen=True)
class Principal:
    """已认证主体。

    `session_id is None` 表示"该主体不绑定单个会话"（如招聘方跨本企业多场会话），
    而不是"未知会话"。区分这两者是会话归属检查的前提：
    只有**绑定了会话**的主体（候选人）才会被强制要求目标会话与其一致。
    """

    principal_id: str
    tenant_id: str
    role: Role
    session_id: Optional[str] = None
    scopes: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if not self.principal_id:
            raise ValueError("principal_id 不能为空（审计日志必须能定位到人/服务）")
        if not self.tenant_id:
            raise ValueError(
                "tenant_id 不能为空：多租户隔离要求每个请求都携带租户上下文"
            )
        if not isinstance(self.role, Role):
            raise TypeError(f"role 必须是 Role，实际: {self.role!r}")
        if self.session_id is not None and not self.session_id:
            raise ValueError("session_id 不可为空串；不绑定会话请传 None")


def has_permission(role: Role, permission: str) -> bool:
    """该角色是否拥有权限。**只查权限位**，不做租户/会话/三重门判断。"""
    return permission in ROLE_PERMISSIONS.get(role, frozenset())


def authorize(
    principal: Principal,
    permission: str,
    *,
    session_id: Optional[str] = None,
    tenant_id: Optional[str] = None,
    physio_second_grant: bool = False,
) -> None:
    """授权判定。通过返回 `None`，否则抛 `Forbidden`。

    检查顺序 = 根因顺序（角色权限 → 生理三重门 → 租户隔离 → 会话归属），
    这样错误信息总是指向"最上游的那条不满足"，而不是被下游检查抢先报一个次因。

    Args:
        principal: 已认证主体。
        permission: 目标权限常量。
        session_id: 目标会话；None 表示不针对具体会话（如按租户批量操作）。
        tenant_id: 目标租户；None 表示调用方未声明目标租户（**生理读取不允许**这种情况）。
        physio_second_grant: 候选人是否已对生理数据分析做过**二次授权**。

    Raises:
        Forbidden: 任一门未通过。
    """
    # 门①：角色是否本来就有该权限（白名单）。管理员在这条就被挡在 PHYSIO_READ 之外。
    if not has_permission(principal.role, permission):
        raise Forbidden(
            f"角色 {principal.role.value} 不具备权限 {permission}",
            permission=permission,
            role=principal.role.value,
        )

    # 生理数据三重门：在通用租户检查**之前**做，以便给出"缺哪一门"的明确信息。
    if permission == PHYSIO_READ:
        # 门②：候选人二次授权。默认不可见，缺此位即拒。
        if not physio_second_grant:
            raise Forbidden(
                "生理数据被拒绝：缺少候选人二次授权（三重门之② physio_second_grant=False）",
                permission=permission,
                role=principal.role.value,
            )
        # 门③：必须证明同租户。未提供目标租户 = 无法证明，按拒绝处理（安全优先）。
        if tenant_id is None:
            raise Forbidden(
                "生理数据被拒绝：未提供目标租户上下文，无法证明同租户（三重门之③）",
                permission=permission,
                role=principal.role.value,
            )
        if tenant_id != principal.tenant_id:
            raise Forbidden(
                "生理数据被拒绝：跨租户访问（三重门之③）"
                f"，主体租户 {principal.tenant_id}，目标租户 {tenant_id}",
                permission=permission,
                role=principal.role.value,
                target_tenant=tenant_id,
            )

    # 多租户隔离：对所有权限生效的硬线。跨租户一律拒绝。
    if tenant_id is not None and tenant_id != principal.tenant_id:
        raise Forbidden(
            "跨租户访问被拒绝："
            f"主体租户 {principal.tenant_id}，目标租户 {tenant_id}",
            permission=permission,
            role=principal.role.value,
            target_tenant=tenant_id,
        )

    # 会话归属：仅对**绑定会话**的主体（候选人）强制。绑定会话者不得访问他人会话。
    if (
        session_id is not None
        and principal.session_id is not None
        and principal.session_id != session_id
    ):
        raise Forbidden(
            "会话不匹配："
            f"主体绑定会话 {principal.session_id}，目标会话 {session_id}",
            permission=permission,
            role=principal.role.value,
        )


def revoke_consent(
    principal: Principal,
    *,
    session_id: Optional[str] = None,
    tenant_id: Optional[str] = None,
) -> None:
    """撤回生理数据授权的**授权判定**（不含存储实现）。

    只有候选人本人可撤回：`CONSENT_REVOKE` 权限仅在候选人角色中，且候选人主体
    绑定了 `session_id`，`authorize` 的会话归属检查会强制其只能撤回**本人**会话。
    招聘方/管理员/系统即便伪造会话也拿不到该权限位，直接在这里被拒。
    """
    authorize(
        principal,
        CONSENT_REVOKE,
        session_id=session_id,
        tenant_id=tenant_id,
    )


__all__ = [
    "ALL_PERMISSIONS",
    "AUDIT_READ",
    "BANK_WRITE",
    "CONFIG_WRITE",
    "CONSENT_REVOKE",
    "DATA_DELETE",
    "Forbidden",
    "PHYSIO_READ",
    "Principal",
    "RAW_MEDIA_READ",
    "REPORT_READ",
    "ROLE_PERMISSIONS",
    "Role",
    "SESSION_READ",
    "SESSION_WRITE",
    "authorize",
    "has_permission",
    "revoke_consent",
]
