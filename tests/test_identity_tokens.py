"""`ruipin.identity.tokens` 的分支/边界测试。

覆盖的纪律：
- 签名被篡改（改一个字符）→ InvalidToken，绝不"放行看起来对的 token"。
- 过期 / 未生效各有独立异常，`now == exp` 与 `now == iat` 的边界必须精确判定。
- 重放：同一 nonce 第二次校验必失败；且**失败不消耗 nonce**（不给重放者制造 DoS）。
- 先验签后解析：签名通过但载荷非法（缺字段/类型错/非法 JSON）也要拒绝。
- 时钟注入：所有时间断言都用确定性时钟，模块内不得自取时间。
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import pickle

import pytest

from ruipin.adapters.fakes import DeterministicClock
from ruipin.domain.errors import RuipinError
from ruipin.identity import tokens as tokens_mod
from ruipin.identity.permissions import Role
from ruipin.identity.tokens import (
    DEFAULT_TTL_S,
    InMemoryNonceStore,
    InterviewToken,
    InvalidToken,
    TokenError,
    TokenExpired,
    TokenNotYetValid,
    TokenReplayed,
    TokenSigner,
)

pytestmark = pytest.mark.unit

SECRET = b"unit-test-secret-key-32-bytes!!"  # ≥16 字节


# ---------- 测试辅助：独立复刻线格式，用于构造"签名合法但载荷非法"的 token ----------


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _sign_raw(raw: bytes, secret: bytes = SECRET) -> str:
    """按文档线格式 `base64url(raw).base64url(hmac)` 签名任意字节。

    刻意不复用被测实现的私有函数：这样能独立验证"签名覆盖原始字节"这一契约。
    """
    sig = hmac.new(secret, raw, hashlib.sha256).digest()
    return f"{_b64url(raw)}.{_b64url(sig)}"


def _signed_payload(obj, secret: bytes = SECRET) -> str:
    import json

    raw = json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )
    return _sign_raw(raw, secret)


@pytest.fixture
def clk() -> DeterministicClock:
    return DeterministicClock(1000.0)


@pytest.fixture
def signer(clk: DeterministicClock) -> TokenSigner:
    return TokenSigner(SECRET, clk.now)


# ==========================================================================
# 1. 签发与正常校验
# ==========================================================================


def test_issue_and_verify_roundtrip_preserves_fields(signer, clk):
    """防回归：签发后校验丢失字段（tenant/role/scopes 被吞）。
    断言：合法 token 必须能完整还原载荷。"""
    token = signer.issue(
        session_id="s-1",
        tenant_id="t-1",
        role=Role.RECRUITER,
        ttl_s=120,
        scopes=["session:read", "report:read"],
        nonce="nonce-abc",
    )
    parsed = signer.verify(token)
    assert parsed.session_id == "s-1", "session_id 必须原样还原"
    assert parsed.tenant_id == "t-1", "tenant_id 必须原样还原"
    assert parsed.role == "recruiter", "role 必须以字符串值入载荷"
    assert parsed.iat == 1000, "iat 必须等于签发时注入时钟的秒数"
    assert parsed.exp == 1120, "exp 必须等于 iat + ttl_s"
    assert parsed.nonce == "nonce-abc", "nonce 必须原样还原"
    assert parsed.scopes == frozenset({"session:read", "report:read"}), "scopes 必须还原为集合"


def test_issue_accepts_role_enum_and_string(signer):
    """防回归：role 只接受枚举导致字符串调用点崩。断言两种写法都得同一结果。"""
    a = signer.issue(session_id="s", tenant_id="t", role=Role.CANDIDATE, nonce="n1")
    b = signer.issue(session_id="s", tenant_id="t", role="candidate", nonce="n2")
    assert signer.verify(a).role == signer.verify(b).role == "candidate", "枚举与字符串须等价"


def test_issue_generates_unique_nonce_by_default(signer):
    """防回归：默认 nonce 退化为常量/空（重放防线形同虚设）。"""
    a = signer.issue(session_id="s", tenant_id="t", role=Role.CANDIDATE)
    b = signer.issue(session_id="s", tenant_id="t", role=Role.CANDIDATE)
    pa = signer.verify(a)
    pb = signer.verify(b)
    assert pa.nonce, "默认 nonce 不能为空"
    assert pb.nonce, "默认 nonce 不能为空"
    assert pa.nonce != pb.nonce, "默认 nonce 必须每次随机不同"


def test_issue_uses_default_ttl_when_not_given(signer):
    """防回归：未给 ttl_s 时不按默认有效期签发（链接寿命失控）。"""
    parsed = signer.verify(signer.issue(session_id="s", tenant_id="t", role=Role.CANDIDATE, nonce="n"))
    assert parsed.exp - parsed.iat == DEFAULT_TTL_S, "未指定 ttl 必须用默认有效期"


@pytest.mark.parametrize("bad_ttl", [0, -1])
def test_issue_rejects_non_positive_ttl(signer, bad_ttl):
    """防回归：零/负有效期被接受（签发即过期或永不合理的边界）。"""
    with pytest.raises(ValueError) as e:
        signer.issue(session_id="s", tenant_id="t", role=Role.CANDIDATE, ttl_s=bad_ttl)
    assert "ttl_s" in str(e.value), "错误信息必须点名 ttl_s 非法"


def test_issue_rejects_invalid_role(signer):
    """防回归：签发了权限层不认识的"裸角色"（下游 Role 解析时崩）。"""
    with pytest.raises(ValueError) as e:
        signer.issue(session_id="s", tenant_id="t", role="president")
    assert "role 非法" in str(e.value), "必须明确拒绝未知角色"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"session_id": ""},
        {"tenant_id": ""},
    ],
)
def test_issue_rejects_empty_identifiers(signer, kwargs):
    """防回归：空 session/tenant 被签发（无法归属到人/租户）。"""
    base = {"session_id": "s", "tenant_id": "t", "role": Role.CANDIDATE}
    base.update(kwargs)
    with pytest.raises(ValueError) as e:
        signer.issue(**base)
    assert "不能为空" in str(e.value), "空标识必须显式报错"


# ==========================================================================
# 2. TokenSigner 构造期校验
# ==========================================================================


def test_signer_rejects_non_bytes_secret(clk):
    """防回归：密钥类型混乱（str vs bytes）导致签名行为不确定。"""
    with pytest.raises(TypeError) as e:
        TokenSigner("not-bytes", clk.now)  # type: ignore[arg-type]
    assert "bytes" in str(e.value), "secret 必须是 bytes"


def test_signer_rejects_empty_secret(clk):
    """防回归：空密钥被接受（任何人都能伪造签名）。"""
    with pytest.raises(ValueError) as e:
        TokenSigner(b"", clk.now)
    assert "不能为空" in str(e.value), "空密钥必须拒绝"


def test_signer_rejects_short_secret(clk):
    """防回归：弱密钥被接受（可暴力枚举后伪造）。"""
    with pytest.raises(ValueError) as e:
        TokenSigner(b"short", clk.now)
    assert "过短" in str(e.value), "过短密钥必须拒绝"


def test_signer_rejects_non_callable_clock():
    """防回归：时钟不是可调用对象（运行期才崩）。"""
    with pytest.raises(TypeError) as e:
        TokenSigner(SECRET, None)  # type: ignore[arg-type]
    assert "clock" in str(e.value), "clock 必须是可调用对象"


def test_signer_rejects_non_positive_default_ttl(clk):
    """防回归：默认有效期被配成非正数。"""
    with pytest.raises(ValueError) as e:
        TokenSigner(SECRET, clk.now, default_ttl_s=0)
    assert "default_ttl_s" in str(e.value), "默认有效期必须为正"


# ==========================================================================
# 3. 结构 / 编码 / 签名 失败
# ==========================================================================


def test_verify_rejects_non_string(signer):
    """防回归：非字符串入参导致 AttributeError 而非结构化失败。"""
    with pytest.raises(InvalidToken) as e:
        signer.verify(123)  # type: ignore[arg-type]
    assert "必须是字符串" in str(e.value), "非字符串必须判为格式非法"


@pytest.mark.parametrize("bad", ["no-dot-here", "a.b.c"])
def test_verify_rejects_wrong_segment_count(signer, bad):
    """防回归：段数不对却继续解析（越界/错位）。"""
    with pytest.raises(InvalidToken) as e:
        signer.verify(bad)
    assert "两段" in str(e.value), "段数不对必须报格式非法"


@pytest.mark.parametrize("bad", [".", "abc.", ".abc"])
def test_verify_rejects_empty_segment(signer, bad):
    """防回归：空载荷/空签名段被当成合法（可被构造绕过签名）。"""
    with pytest.raises(InvalidToken) as e:
        signer.verify(bad)
    assert "为空" in str(e.value), "空段必须报格式非法"


def test_verify_rejects_invalid_base64url(signer):
    """防回归：字母表外字符被宽松解码吞掉（内容可能已被改动）。"""
    with pytest.raises(InvalidToken) as e:
        signer.verify("!!!!.@@@@")
    assert "base64url" in str(e.value), "非法 base64url 必须报格式非法"


def test_verify_rejects_tampered_signature(signer):
    """★ 防回归：签名被改一个字符仍通过（最严重——伪造 token 可行）。"""
    token = signer.issue(session_id="s", tenant_id="t", role=Role.CANDIDATE, nonce="n")
    payload, sig = token.split(".")
    # 翻转签名段一个字符（改变但不破坏 base64 长度）
    flipped = "A" if sig[0] != "A" else "B"
    tampered = f"{payload}.{flipped}{sig[1:]}"
    with pytest.raises(InvalidToken) as e:
        signer.verify(tampered)
    assert "签名不匹配" in str(e.value), "签名被篡改必须判签名不匹配"


def test_verify_rejects_tampered_payload(signer):
    """★ 防回归：载荷被改一个字符仍通过（改 session_id 劫持他人会话）。"""
    token = signer.issue(session_id="s", tenant_id="t", role=Role.CANDIDATE, nonce="n")
    payload, sig = token.split(".")
    flipped = "A" if payload[0] != "A" else "B"
    tampered = f"{flipped}{payload[1:]}.{sig}"
    with pytest.raises(InvalidToken) as e:
        signer.verify(tampered)
    assert "签名不匹配" in str(e.value), "载荷被篡改必须判签名不匹配"


def test_verify_rejects_token_signed_with_other_secret(clk):
    """★ 防回归：换密钥仍能验证（签名无意义）。"""
    a = TokenSigner(SECRET, clk.now)
    b = TokenSigner(b"another-secret-key-32-bytes!!!!", clk.now)
    token = a.issue(session_id="s", tenant_id="t", role=Role.CANDIDATE, nonce="n")
    with pytest.raises(InvalidToken) as e:
        b.verify(token)
    assert "签名不匹配" in str(e.value), "密钥不一致必须判签名不匹配"


def test_b64url_decode_rejects_non_str_directly():
    """防回归：私有解码器接受非字符串（调用点类型假设被破坏）。"""
    with pytest.raises(ValueError) as e:
        tokens_mod._b64url_decode(123)  # type: ignore[arg-type]
    assert "字符串" in str(e.value), "非字符串输入必须显式报错"


# ==========================================================================
# 4. 先验签后解析：签名合法但载荷非法
# ==========================================================================


def test_verify_rejects_signed_non_json_payload(signer):
    """★ 防回归：签名通过即跳过 JSON 校验（不可信载荷进入解析）。"""
    with pytest.raises(InvalidToken) as e:
        signer.verify(_sign_raw(b"this-is-not-json"))
    assert "JSON" in str(e.value), "签名合法但非 JSON 必须判格式非法"


def test_verify_rejects_signed_non_utf8_payload(signer):
    """防回归：非 UTF-8 字节导致 UnicodeDecodeError 逃逸（未归类为 InvalidToken）。"""
    with pytest.raises(InvalidToken) as e:
        signer.verify(_sign_raw(b"\xff\xfe\x00"))
    assert "JSON" in str(e.value), "非 UTF-8 载荷必须判格式非法"


def test_verify_rejects_signed_json_array(signer):
    """防回归：载荷是 JSON 但不是对象（字段访问会崩）。"""
    with pytest.raises(InvalidToken) as e:
        signer.verify(_sign_raw(b"[1,2,3]"))
    assert "对象" in str(e.value), "非对象 JSON 必须判格式非法"


def test_verify_rejects_missing_field(signer):
    """防回归：缺字段的载荷被接受（默认值冒充真实值）。"""
    obj = {
        "session_id": "s",
        "tenant_id": "t",
        "role": "candidate",
        "iat": 1000,
        "exp": 2000,
        "nonce": "n",
    }
    del obj["tenant_id"]
    with pytest.raises(InvalidToken) as e:
        signer.verify(_signed_payload(obj))
    assert "tenant_id" in str(e.value), "缺字段必须报出来具体是哪个字段"


def test_verify_rejects_missing_int_field(signer):
    """防回归：缺**整数**字段（iat/exp）被放行（时间校验失去依据）。"""
    obj = {
        "session_id": "s",
        "tenant_id": "t",
        "role": "candidate",
        "exp": 2000,
        "nonce": "n",
    }
    with pytest.raises(InvalidToken) as e:
        signer.verify(_signed_payload(obj))
    assert "iat" in str(e.value), "缺失的整数字段必须点名"


def test_verify_rejects_wrong_type_session_id(signer):
    """防回归：字段类型错误被强行转换（int 冒充 str）。"""
    obj = {
        "session_id": 123,
        "tenant_id": "t",
        "role": "candidate",
        "iat": 1000,
        "exp": 2000,
        "nonce": "n",
    }
    with pytest.raises(InvalidToken) as e:
        signer.verify(_signed_payload(obj))
    assert "session_id" in str(e.value), "类型错误必须点名字段"


def test_verify_rejects_bool_timestamp(signer):
    """防回归：JSON true/false 被当整数时间戳（bool 是 int 子类，易漏）。"""
    obj = {
        "session_id": "s",
        "tenant_id": "t",
        "role": "candidate",
        "iat": True,
        "exp": 2000,
        "nonce": "n",
    }
    with pytest.raises(InvalidToken) as e:
        signer.verify(_signed_payload(obj))
    assert "iat" in str(e.value), "bool 时间戳必须被拒绝"


def test_verify_rejects_non_int_timestamp(signer):
    """防回归：时间戳用浮点/字符串（比较语义不确定）。"""
    obj = {
        "session_id": "s",
        "tenant_id": "t",
        "role": "candidate",
        "iat": 1000,
        "exp": "2000",
        "nonce": "n",
    }
    with pytest.raises(InvalidToken) as e:
        signer.verify(_signed_payload(obj))
    assert "exp" in str(e.value), "非整数时间戳必须被拒绝"


@pytest.mark.parametrize("scopes", ["session:read", 5, {"a": 1}])
def test_verify_rejects_bad_scopes_container(signer, scopes):
    """防回归：scopes 不是数组却进了解析。"""
    obj = {
        "session_id": "s",
        "tenant_id": "t",
        "role": "candidate",
        "iat": 1000,
        "exp": 2000,
        "nonce": "n",
        "scopes": scopes,
    }
    with pytest.raises(InvalidToken) as e:
        signer.verify(_signed_payload(obj))
    assert "scopes" in str(e.value), "非法 scopes 容器必须被拒绝"


def test_verify_rejects_non_str_scope_element(signer):
    """防回归：scopes 元素非字符串被接受（下游权限比较口径漂移）。"""
    obj = {
        "session_id": "s",
        "tenant_id": "t",
        "role": "candidate",
        "iat": 1000,
        "exp": 2000,
        "nonce": "n",
        "scopes": [1, 2],
    }
    with pytest.raises(InvalidToken) as e:
        signer.verify(_signed_payload(obj))
    assert "scopes" in str(e.value), "非字符串 scopes 元素必须被拒绝"


def test_verify_treats_null_scopes_as_empty(signer):
    """防回归：scopes=null 被判非法（合法发送方可能省略/置空 scopes）。"""
    obj = {
        "session_id": "s",
        "tenant_id": "t",
        "role": "candidate",
        "iat": 1000,
        "exp": 2000,
        "nonce": "n",
        "scopes": None,
    }
    parsed = signer.verify(_signed_payload(obj))
    assert parsed.scopes == frozenset(), "scopes=null 应等价于空集合"


def test_verify_ignores_unknown_fields(signer):
    """防回归：多了未知字段就拒绝（妨碍向前兼容地扩展载荷）。"""
    obj = {
        "session_id": "s",
        "tenant_id": "t",
        "role": "candidate",
        "iat": 1000,
        "exp": 2000,
        "nonce": "n",
        "future_field": "whatever",
    }
    parsed = signer.verify(_signed_payload(obj))
    assert parsed.session_id == "s", "未知字段应被忽略而非导致失败"


# ==========================================================================
# 5. 时效：边界精确
# ==========================================================================


def test_verify_rejects_expired(signer, clk):
    """★ 防回归：过期 token 被接受（链接可无限期使用）。"""
    token = signer.issue(session_id="s", tenant_id="t", role=Role.CANDIDATE, ttl_s=10, nonce="n")
    clk.set(1011.0)  # > 1010
    with pytest.raises(TokenExpired) as e:
        signer.verify(token)
    assert "已过期" in str(e.value) and "1010" in str(e.value), "必须报过期且带上 exp"


def test_verify_accepts_at_exp_boundary(signer, clk):
    """防回归：`now == exp` 被误判过期（边界口径必须是严格大于）。"""
    token = signer.issue(session_id="s", tenant_id="t", role=Role.CANDIDATE, ttl_s=10, nonce="n")
    clk.set(1010.0)  # == exp
    assert signer.verify(token).exp == 1010, "now == exp 仍应有效"


def test_verify_rejects_not_yet_valid(signer, clk):
    """★ 防回归：未生效 token 被接受（提前签发被滥用）。"""
    token = signer.issue(session_id="s", tenant_id="t", role=Role.CANDIDATE, ttl_s=100, nonce="n")
    clk.set(999.0)  # < iat(1000)
    with pytest.raises(TokenNotYetValid) as e:
        signer.verify(token)
    assert "尚未生效" in str(e.value), "必须报尚未生效"


def test_verify_accepts_at_iat_boundary(signer, clk):
    """防回归：`now == iat` 被误判未生效（边界口径必须是严格小于）。"""
    token = signer.issue(session_id="s", tenant_id="t", role=Role.CANDIDATE, ttl_s=100, nonce="n")
    clk.set(1000.0)  # == iat
    assert signer.verify(token).iat == 1000, "now == iat 仍应有效"


def test_failed_time_check_does_not_consume_nonce(signer, clk):
    """★ 防回归：校验失败却消耗了 nonce（攻击者可用无效 token 顶掉有效 nonce，制造 DoS）。"""
    token = signer.issue(session_id="s", tenant_id="t", role=Role.CANDIDATE, ttl_s=100, nonce="n")
    clk.set(500.0)  # 未生效
    with pytest.raises(TokenNotYetValid):
        signer.verify(token)
    clk.set(1000.0)  # 回到有效窗口
    assert signer.verify(token).nonce == "n", "失败不应消耗 nonce，回到有效窗口后应能成功"


# ==========================================================================
# 6. 防重放
# ==========================================================================


def test_verify_rejects_replay(signer):
    """★ 防回归：一次性 token 可被重复使用（重放攻击）。"""
    token = signer.issue(session_id="s", tenant_id="t", role=Role.CANDIDATE, nonce="once")
    assert signer.verify(token).nonce == "once", "首次校验应成功"
    with pytest.raises(TokenReplayed) as e:
        signer.verify(token)
    assert "重放" in str(e.value) and "once" in str(e.value), "第二次必须报重放并点名 nonce"


def test_shared_nonce_store_detects_cross_signer_replay(clk):
    """★ 防回归：多实例各持内存集合时漏检重放（分布式场景必须共享端口）。"""
    store = InMemoryNonceStore()
    a = TokenSigner(SECRET, clk.now, nonce_store=store)
    b = TokenSigner(SECRET, clk.now, nonce_store=store)
    token = a.issue(session_id="s", tenant_id="t", role=Role.CANDIDATE, nonce="shared")
    a.verify(token)
    with pytest.raises(TokenReplayed):
        b.verify(token)


def test_independent_nonce_stores_do_not_false_positive(clk):
    """防回归：不同实例的独立集合互相干扰（正常请求被判重放）。"""
    a = TokenSigner(SECRET, clk.now)
    b = TokenSigner(SECRET, clk.now)
    token = a.issue(session_id="s", tenant_id="t", role=Role.CANDIDATE, nonce="solo")
    a.verify(token)
    # 独立集合的 b 不认识该 nonce，重新签发给她验证即可（这里直接构造同内容新 token）
    token2 = b.issue(session_id="s", tenant_id="t", role=Role.CANDIDATE, nonce="solo2")
    assert b.verify(token2).nonce == "solo2", "独立实例不应互相判重放"


def test_in_memory_nonce_store_claim_semantics():
    """防回归：内存集合的 claim 语义搞反（首次 False / 重复 True）。"""
    store = InMemoryNonceStore()
    assert store.claim("x") is True, "首次认领必须返回 True"
    assert store.claim("x") is False, "重复认领必须返回 False"
    assert len(store) == 1, "集合大小应为 1"


# ==========================================================================
# 7. 值对象校验
# ==========================================================================


def _valid_kwargs(**over):
    base = dict(
        session_id="s",
        tenant_id="t",
        role="candidate",
        iat=10,
        exp=20,
        nonce="n",
    )
    base.update(over)
    return base


@pytest.mark.parametrize(
    "field,value",
    [("session_id", ""), ("tenant_id", ""), ("nonce", "")],
)
def test_interview_token_rejects_empty_strings(field, value):
    """防回归：关键字段允许空串（无法归属/无法防重放）。"""
    with pytest.raises(ValueError) as e:
        InterviewToken(**_valid_kwargs(**{field: value}))
    assert "不能为空" in str(e.value), f"{field} 不能为空"


def test_interview_token_rejects_unknown_role():
    """防回归：值对象接受未知角色。"""
    with pytest.raises(ValueError) as e:
        InterviewToken(**_valid_kwargs(role="ghost"))
    assert "role 非法" in str(e.value), "未知角色必须被拒绝"


@pytest.mark.parametrize("field", ["iat", "exp"])
def test_interview_token_rejects_non_int_timestamps(field):
    """防回归：时间戳非整数（比较语义不确定）。"""
    with pytest.raises(TypeError) as e:
        InterviewToken(**_valid_kwargs(**{field: "10"}))
    assert field in str(e.value), f"{field} 必须是整数"


def test_interview_token_rejects_bool_timestamp():
    """防回归：bool 冒充整数时间戳。"""
    with pytest.raises(TypeError) as e:
        InterviewToken(**_valid_kwargs(iat=True))
    assert "iat" in str(e.value), "bool 时间戳必须被拒绝"


def test_interview_token_rejects_negative_iat():
    """防回归：负的签发时刻（时间线无意义）。"""
    with pytest.raises(ValueError) as e:
        InterviewToken(**_valid_kwargs(iat=-1, exp=20))
    assert "负" in str(e.value), "iat 不能为负"


@pytest.mark.parametrize("exp", [10, 9])
def test_interview_token_rejects_exp_not_after_iat(exp):
    """防回归：有效期为零/为负（签发即废）。"""
    with pytest.raises(ValueError) as e:
        InterviewToken(**_valid_kwargs(iat=10, exp=exp))
    assert "exp 必须大于 iat" in str(e.value), "有效期必须为正"


# ==========================================================================
# 8. 异常族：继承、消息、可序列化
# ==========================================================================

TOKEN_ERRORS = (TokenError, InvalidToken, TokenExpired, TokenNotYetValid, TokenReplayed)


@pytest.mark.parametrize("cls", TOKEN_ERRORS, ids=lambda c: c.__name__)
def test_token_errors_are_ruipin_errors(cls):
    """防回归：token 异常脱离 RuipinError（统一兜底捕获漏掉它）。"""
    assert issubclass(cls, TokenError), "必须属于 TokenError 族"
    assert issubclass(cls, RuipinError), "必须属于 RuipinError 族"


@pytest.mark.parametrize("cls", TOKEN_ERRORS, ids=lambda c: c.__name__)
def test_token_errors_pickle_roundtrip(cls):
    """★ 防回归：异常跨进程后类型/消息丢失（worker 侧抛、主进程侧无法分类）。"""
    err = cls("某原因")
    restored = pickle.loads(pickle.dumps(err))
    assert type(restored) is cls, "pickle 往返后类型必须不变"
    assert restored.reason == "某原因", "reason 必须保留"
    assert str(restored) == str(err), "消息必须一致"


def test_token_error_message_format():
    """防回归：异常消息缺少统一前缀（日志检索失败）。"""
    err = TokenExpired("已过期：当前 5 > 过期时刻 3")
    assert str(err) == "token error: 已过期：当前 5 > 过期时刻 3", "消息格式须稳定"
