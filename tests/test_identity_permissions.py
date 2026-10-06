"""`ruipin.identity.permissions` 的分支/边界测试。

覆盖的纪律：
- 默认拒绝：未列的权限一律不放行。
- 多租户隔离：跨租户访问一律拒绝，且对**所有**权限生效。
- 生理三重门：①角色本有权 ②候选人二次授权 ③同租户，缺一即拒并点名缺哪条。
- 管理员**读不到**生理数据，且**没有**原始媒体访问权。
- 候选人只能删/撤回**本人**会话的数据。
"""

from __future__ import annotations

import pickle

import pytest

from ruipin.domain.errors import RuipinError
from ruipin.identity.permissions import (
    ALL_PERMISSIONS,
    AUDIT_READ,
    BANK_WRITE,
    CONFIG_WRITE,
    CONSENT_REVOKE,
    DATA_DELETE,
    PHYSIO_READ,
    RAW_MEDIA_READ,
    REPORT_READ,
    ROLE_PERMISSIONS,
    SESSION_READ,
    SESSION_WRITE,
    Forbidden,
    Principal,
    Role,
    authorize,
    has_permission,
    revoke_consent,
)

pytestmark = pytest.mark.unit


# ---------- 构造助手 ----------


def candidate(session_id="s-1", tenant_id="t-1"):
    return Principal("cand-1", tenant_id, Role.CANDIDATE, session_id=session_id)


def recruiter(tenant_id="t-1"):
    return Principal("rec-1", tenant_id, Role.RECRUITER)


def admin(tenant_id="t-1"):
    return Principal("adm-1", tenant_id, Role.ADMIN)


def system(tenant_id="t-1"):
    return Principal("sys-1", tenant_id, Role.SYSTEM)


# ==========================================================================
# 1. 角色枚举与权限矩阵
# ==========================================================================


def test_role_values_are_stable_strings():
    """防回归：角色值与方案 §12.1 表格漂移（对外协议不兼容）。"""
    assert Role.CANDIDATE.value == "candidate"
    assert Role.RECRUITER.value == "recruiter"
    assert Role.ADMIN.value == "admin"
    assert Role.SYSTEM.value == "system"
    assert {r.value for r in Role} == {"candidate", "recruiter", "admin", "system"}


def test_role_permissions_is_immutable():
    """★ 防回归：权限表运行期被改写（悄然提权）。"""
    with pytest.raises(TypeError):
        ROLE_PERMISSIONS[Role.ADMIN] = frozenset({PHYSIO_READ})  # type: ignore[index]
    # 赋值失败后，管理员矩阵必须原样（未被污染）
    assert ROLE_PERMISSIONS[Role.ADMIN] == frozenset(
        {CONFIG_WRITE, BANK_WRITE, AUDIT_READ}
    ), "被拒绝的写入不得改变原矩阵"


def test_every_permission_in_matrix_is_defined():
    """防回归：矩阵里出现未定义的权限常量（拼写错误被静默接受）。"""
    for perms in ROLE_PERMISSIONS.values():
        assert perms <= ALL_PERMISSIONS, "矩阵不得包含未定义权限"


@pytest.mark.parametrize(
    "role,permission,expected",
    [
        (Role.CANDIDATE, SESSION_READ, True),
        (Role.CANDIDATE, CONSENT_REVOKE, True),
        (Role.CANDIDATE, AUDIT_READ, False),
        (Role.RECRUITER, SESSION_READ, True),
        (Role.RECRUITER, PHYSIO_READ, True),
        (Role.RECRUITER, RAW_MEDIA_READ, False),
        (Role.ADMIN, CONFIG_WRITE, True),
        (Role.ADMIN, BANK_WRITE, True),
        (Role.ADMIN, AUDIT_READ, True),
        (Role.ADMIN, PHYSIO_READ, False),
        (Role.ADMIN, RAW_MEDIA_READ, False),
        (Role.SYSTEM, SESSION_WRITE, True),
        (Role.SYSTEM, REPORT_READ, False),
        (Role.SYSTEM, AUDIT_READ, False),
    ],
)
def test_permission_matrix_matches_policy(role, permission, expected):
    """★ 防回归：权限矩阵与 §12.1 语义不一致（越权或功能不可用）。"""
    assert has_permission(role, permission) is expected, f"{role.value} 对 {permission} 的预期"


def test_has_permission_unknown_role_returns_false():
    """防回归：未知角色被当成有权限（默认拒绝原则）。"""
    assert has_permission("unknown", SESSION_READ) is False  # type: ignore[arg-type]


def test_no_role_can_read_raw_media():
    """★ 防回归：任何角色拿到原始媒体权限（方案要求"不落盘"，此位为防御性存在）。"""
    assert all(RAW_MEDIA_READ not in perms for perms in ROLE_PERMISSIONS.values()), (
        "没有任何角色应持有 RAW_MEDIA_READ"
    )


# ==========================================================================
# 2. Principal 校验
# ==========================================================================


def test_principal_accepts_valid():
    """防回归：合法主体构造失败（把所有调用点拖崩）。"""
    p = Principal("p", "t", Role.CANDIDATE, session_id="s", scopes=frozenset({"x"}))
    assert p.role is Role.CANDIDATE and p.session_id == "s"


@pytest.mark.parametrize("field,value", [("principal_id", ""), ("tenant_id", "")])
def test_principal_rejects_empty_ids(field, value):
    """防回归：空 id/租户被接受（无法归属/无法隔离）。"""
    kwargs = {"principal_id": "p", "tenant_id": "t", "role": Role.CANDIDATE}
    kwargs[field] = value
    with pytest.raises(ValueError) as e:
        Principal(**kwargs)
    assert "不能为空" in str(e.value), f"{field} 不能为空"


def test_principal_rejects_non_role():
    """防回归：role 用裸字符串冒充（角色分支比较全部失效）。"""
    with pytest.raises(TypeError) as e:
        Principal("p", "t", "candidate")  # type: ignore[arg-type]
    assert "Role" in str(e.value), "role 必须是 Role"


def test_principal_rejects_empty_session_string():
    """防回归：空串 session 与 None 被混为一谈（会话归属检查口径错）。"""
    with pytest.raises(ValueError) as e:
        Principal("p", "t", Role.CANDIDATE, session_id="")
    assert "session_id" in str(e.value), "空串 session 必须报错"


def test_principal_without_session_is_none():
    """防回归：不绑定会话的主体被强塞一个默认 session。"""
    assert Principal("p", "t", Role.RECRUITER).session_id is None, "招聘方不应绑定单会话"


# ==========================================================================
# 3. authorize：基础放行与拒绝
# ==========================================================================


def test_authorize_allows_candidate_read_own_session():
    """防回归：候选人读本人会话被误拒（正常功能不可用）。"""
    authorize(candidate("s-1"), SESSION_READ, session_id="s-1", tenant_id="t-1")


def test_authorize_rejects_role_without_permission():
    """★ 防回归：默认拒绝失效——系统角色拿到了题库写权限。"""
    with pytest.raises(Forbidden) as e:
        authorize(system(), BANK_WRITE)
    assert "不具备权限" in str(e.value), "缺权限必须明确报出"
    assert e.value.permission == BANK_WRITE, "结构化字段须带 permission"
    assert e.value.role == "system", "结构化字段须带 role"


def test_authorize_rejects_cross_tenant():
    """★ 防回归：跨租户访问被放行（多租户隔离破裂，最严重的越权）。"""
    with pytest.raises(Forbidden) as e:
        authorize(recruiter("t-1"), SESSION_READ, tenant_id="t-2")
    text = str(e.value)
    assert "跨租户" in text, "必须报跨租户"
    assert "t-1" in text and "t-2" in text, "消息须带上两个租户名"
    assert e.value.target_tenant == "t-2", "结构化字段须带目标租户"


def test_authorize_allows_same_tenant():
    """防回归：同租户被误判跨租户（隔离过严导致不可用）。"""
    authorize(recruiter("t-1"), SESSION_READ, tenant_id="t-1")


def test_authorize_rejects_session_mismatch():
    """★ 防回归：绑定了会话的主体访问**他人**会话（候选人互相查看）。"""
    with pytest.raises(Forbidden) as e:
        authorize(candidate("s-1"), SESSION_READ, session_id="s-2", tenant_id="t-1")
    assert "会话不匹配" in str(e.value), "必须报会话不匹配"


def test_authorize_allows_when_target_session_not_specified():
    """防回归：未指定目标会话时误触会话检查（按租户批量操作不可用）。"""
    authorize(candidate("s-1"), SESSION_READ, tenant_id="t-1")


def test_authorize_session_check_ignored_for_unbound_principal():
    """防回归：不绑定会话的招聘方被要求匹配某会话（无法跨会话工作）。"""
    authorize(recruiter("t-1"), SESSION_READ, session_id="s-whatever", tenant_id="t-1")


# ==========================================================================
# 4. 生理三重门
# ==========================================================================


def test_admin_cannot_read_physio_gate1():
    """★ 防回归：管理员读到生理数据（门①：管理员本无此权限）。"""
    with pytest.raises(Forbidden) as e:
        authorize(admin("t-1"), PHYSIO_READ, tenant_id="t-1", physio_second_grant=True)
    assert "不具备权限" in str(e.value), "管理员在门①就被挡下"


def test_recruiter_physio_without_second_grant_rejected_gate2():
    """★ 防回归：招聘方无二次授权就读生理数据（默认可见 = 违规）。"""
    with pytest.raises(Forbidden) as e:
        authorize(recruiter("t-1"), PHYSIO_READ, tenant_id="t-1", physio_second_grant=False)
    assert "二次授权" in str(e.value), "门②：必须点名缺候选人二次授权"


def test_recruiter_physio_without_tenant_rejected_gate3():
    """★ 防回归：无法证明同租户仍放行（门③可被绕过）。"""
    with pytest.raises(Forbidden) as e:
        authorize(recruiter("t-1"), PHYSIO_READ, tenant_id=None, physio_second_grant=True)
    assert "无法证明同租户" in str(e.value), "门③：缺租户上下文必须拒绝"


def test_recruiter_physio_cross_tenant_rejected_gate3():
    """★ 防回归：跨租户读生理数据被放行。"""
    with pytest.raises(Forbidden) as e:
        authorize(recruiter("t-1"), PHYSIO_READ, tenant_id="t-9", physio_second_grant=True)
    assert "跨租户" in str(e.value), "门③：跨租户必须拒绝"
    assert e.value.target_tenant == "t-9", "结构化字段须带目标租户"


def test_recruiter_physio_all_gates_pass():
    """防回归：三条件齐备仍被拒（合规链正常路径不可用）。"""
    authorize(recruiter("t-1"), PHYSIO_READ, tenant_id="t-1", physio_second_grant=True)


def test_second_grant_does_not_unlock_raw_media():
    """★ 防回归：二次生理授权被误当成原始媒体通行证。"""
    with pytest.raises(Forbidden) as e:
        authorize(
            recruiter("t-1"),
            RAW_MEDIA_READ,
            tenant_id="t-1",
            physio_second_grant=True,
        )
    assert "不具备权限" in str(e.value), "招聘方无原始媒体权限，二次授权也不解锁"


# ==========================================================================
# 5. 删除与撤回：只能动本人的
# ==========================================================================


def test_candidate_can_delete_own_data():
    """防回归：候选人删本人数据被误拒。"""
    authorize(candidate("s-1"), DATA_DELETE, session_id="s-1", tenant_id="t-1")


def test_candidate_cannot_delete_others_data():
    """★ 防回归：候选人删除**他人**会话数据（越权删除）。"""
    with pytest.raises(Forbidden) as e:
        authorize(candidate("s-1"), DATA_DELETE, session_id="s-2", tenant_id="t-1")
    assert "会话不匹配" in str(e.value), "候选人不得删除他人会话数据"


def test_recruiter_cannot_delete_data():
    """★ 防回归：招聘方拥有删除权（数据被草率销毁）。"""
    with pytest.raises(Forbidden) as e:
        authorize(recruiter("t-1"), DATA_DELETE, tenant_id="t-1")
    assert "不具备权限" in str(e.value), "招聘方不应有删除权"


def test_revoke_consent_candidate_own_session():
    """防回归：候选人撤回本人生理授权被误拒。"""
    revoke_consent(candidate("s-1"), session_id="s-1", tenant_id="t-1")


def test_revoke_consent_candidate_other_session_rejected():
    """★ 防回归：候选人撤回**他人**会话的授权。"""
    with pytest.raises(Forbidden) as e:
        revoke_consent(candidate("s-1"), session_id="s-2", tenant_id="t-1")
    assert "会话不匹配" in str(e.value), "只能撤回本人会话授权"


@pytest.mark.parametrize("principal", [recruiter("t-1"), admin("t-1"), system("t-1")])
def test_revoke_consent_non_candidate_rejected(principal):
    """★ 防回归：非候选人撤回授权（权限位只在候选人角色上）。"""
    with pytest.raises(Forbidden) as e:
        revoke_consent(principal, tenant_id="t-1")
    assert "不具备权限" in str(e.value), "只有候选人可撤回授权"


def test_revoke_consent_cross_tenant_rejected():
    """★ 防回归：候选人跨租户撤回（隔离失效）。"""
    with pytest.raises(Forbidden) as e:
        revoke_consent(candidate("s-1"), session_id="s-1", tenant_id="t-2")
    assert "跨租户" in str(e.value), "撤回也必须守租户边界"


# ==========================================================================
# 6. Forbidden：字段、消息、可序列化
# ==========================================================================


def test_forbidden_fields_and_message():
    """防回归：Forbidden 丢失结构化字段（审计检索失效）。"""
    err = Forbidden("x", permission=PHYSIO_READ, role="admin", target_tenant="t-9")
    assert err.reason == "x"
    assert err.permission == PHYSIO_READ
    assert err.role == "admin"
    assert err.target_tenant == "t-9"
    assert str(err) == "forbidden: x（权限 physio:read）", "消息格式须稳定"
    assert isinstance(err, RuipinError), "Forbidden 必须属于项目异常族"


def test_forbidden_optional_fields_default_none():
    """防回归：可选字段默认值被写错（默认写成空串而非 None）。"""
    err = Forbidden("x")
    assert err.permission is None and err.role is None and err.target_tenant is None
    assert str(err) == "forbidden: x", "无权限字段时不应拼括号"


def test_forbidden_pickle_roundtrip():
    """★ 防回归：跨进程后 Forbidden 字段丢失（worker 侧拒绝、主进程侧看不清原因）。"""
    err = Forbidden("x", permission=PHYSIO_READ, role="admin", target_tenant="t-9")
    restored = pickle.loads(pickle.dumps(err))
    assert type(restored) is Forbidden
    assert restored.reason == err.reason
    assert restored.permission == err.permission
    assert restored.role == err.role
    assert restored.target_tenant == err.target_tenant
    assert str(restored) == str(err)


def test_forbidden_message_never_empty():
    """防回归：Forbidden 消息退化（日志里没信息）。"""
    assert str(Forbidden("")).strip() != "", "消息不应为空"
    assert "forbidden" in str(Forbidden("")), "至少带统一前缀"
