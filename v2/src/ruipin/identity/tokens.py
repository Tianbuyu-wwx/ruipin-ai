"""一次性面试链接的签名 token（方案 §12.1："一次性面试链接（带签名 token + 有效期）"）。

用途：候选人点开邮件/短信里的一次性链接进入面试。这个 token 承载
`session_id / tenant_id / role / iat / exp / scopes / nonce`，用 HMAC-SHA256 签名，
服务器无需落库即可验证真伪与时效。

线格式
------
``base64url(json_payload) + "." + base64url(hmac_sha256(secret, json_payload))``

签名覆盖**原始 JSON 字节**而不是"解析后再序列化"的语义：JSON 里键顺序、空白、
数字格式在往返后可能变化，若先解析再重签，签名对不上就会把合法 token 判成伪造。
因此签发与验证都以同一段字节为准，payload 段里不会出现 `.`（base64url 字母表不含 `.`），
按 `.` 切两段是无歧义的。

安全边界（为什么这样设计）
--------------------------
1. **先验签，后解析**：`verify` 在 `json.loads` **之前**用常量时间比较验签
   （`hmac.compare_digest`）。绝不先信任未认证的载荷——这是所有签名 token
   （JWT 的 alg 混淆等）类漏洞的根因。普通 `==` 比较会因提前返回而泄漏时序，
   故一律用 `compare_digest`。
2. **失败必须分类显式**：格式坏 / 签名不符 / 过期 / 未生效 / 重放，各抛不同异常。
   "静默返回 None/空"会让调用方分不清"链接没用"是过期、伪造还是被用过，
   排障与安全告警都会失明。
3. **一次性（防重放）**：每次成功验证都通过 `NonceStore` 端口"认领" nonce，已被认领
   即 `TokenReplayed`。**只有全部检查通过才认领**：否则攻击者连续重放一个过期 token
   就能把别人的有效 nonce 顶掉（拒绝服务）。分布式部署必须注入共享存储
   （Redis 等）；只注入内存实现时，防重放仅在单进程内成立。
4. **时钟注入**：`clock` 由构造参数传入（返回**秒**），模块内不调用 `time.time()`，
   测试才能用确定性时钟精确卡边界（如 `now == exp` 仍未过期）。
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import secrets
from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Protocol, runtime_checkable

from ..domain.errors import RuipinError
from .permissions import Role

#: HMAC 密钥最小长度。SHA-256 的安全强度上限即密钥长度，短密钥可被暴力枚举后伪造 token。
MIN_SECRET_BYTES = 16

#: 默认有效期：1 小时。一次性面试链接不需要长有效期，越短攻击窗口越小。
DEFAULT_TTL_S = 3600

#: 合法的角色值（取自权限矩阵，避免 token 里出现权限层不认识的"裸角色"）。
_ROLE_VALUES: frozenset[str] = frozenset(r.value for r in Role)


# ==========================================================================
# 异常族：让调用方能用 except 精确区分失败类型
# ==========================================================================


class TokenError(RuipinError):
    """token 相关异常的基类。`reason` 必须说清是哪一种失败。"""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"token error: {reason}")

    def __reduce__(self):
        # 子类 __init__ 签名与基类一致，故用 self.__class__ 可直接重建为正确子类，
        # 保证跨进程（worker/队列/日志）传播后类型与消息不变。
        return (self.__class__, (self.reason,))


class InvalidToken(TokenError):
    """格式非法或签名不匹配。"""


class TokenExpired(TokenError):
    """已过期（`clock() > exp`）。"""


class TokenNotYetValid(TokenError):
    """尚未生效（`clock() < iat`）。时钟漂移或提前签发都会触发。"""


class TokenReplayed(TokenError):
    """重放：同一 nonce 第二次出现。一次性 token 只能成功验证一次。"""


# ==========================================================================
# 值对象
# ==========================================================================


@dataclass(frozen=True)
class InterviewToken:
    """已验签通过的 token 载荷。

    Args:
        iat: 签发时刻（Unix 秒）。
        exp: 过期时刻（Unix 秒），必须严格大于 `iat`（零/负有效期等于签发即废）。
        nonce: 一次性随机串，防重放的关键；不能为空。
        scopes: 细粒度授权范围（如 `session:read`）。用 `frozenset` 保证不可变。
    """

    session_id: str
    tenant_id: str
    role: str
    iat: int
    exp: int
    nonce: str
    scopes: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if not self.session_id:
            raise ValueError("session_id 不能为空")
        if not self.tenant_id:
            raise ValueError("tenant_id 不能为空")
        if not self.nonce:
            raise ValueError("nonce 不能为空（一次性防重放依赖它）")
        if self.role not in _ROLE_VALUES:
            raise ValueError(
                f"role 非法: {self.role!r}，必须是 {sorted(_ROLE_VALUES)} 之一"
            )
        if not isinstance(self.iat, int) or isinstance(self.iat, bool):
            raise TypeError(f"iat 必须是整数秒，实际: {self.iat!r}")
        if not isinstance(self.exp, int) or isinstance(self.exp, bool):
            raise TypeError(f"exp 必须是整数秒，实际: {self.exp!r}")
        if self.iat < 0:
            raise ValueError(f"iat 不能为负，实际: {self.iat}")
        if self.exp <= self.iat:
            raise ValueError(f"exp 必须大于 iat，实际: {self.exp} <= {self.iat}")


# ==========================================================================
# 防重放端口
# ==========================================================================


@runtime_checkable
class NonceStore(Protocol):
    """已用 nonce 集合端口。

    `claim(nonce)` 必须**原子**地判断并标记：首次出现返回 True 并记为已用，
    已出现过返回 False。生产环境应实现为 Redis `SET NX`（跨进程/跨节点），
    否则多 worker 场景下防重放会退化。
    """

    def claim(self, nonce: str) -> bool: ...


class InMemoryNonceStore:
    """单进程内存实现。

    仅用于单进程/测试。它是**有界失效**的：进程重启后集合清空，已用 nonce 可被再次
    接受；多 worker 也各持一份。生产必须换共享存储——本类刻意不假装自己是完整方案。
    """

    def __init__(self) -> None:
        self._seen: set[str] = set()

    def claim(self, nonce: str) -> bool:
        if nonce in self._seen:
            return False
        self._seen.add(nonce)
        return True

    def __len__(self) -> int:
        return len(self._seen)


# ==========================================================================
# 签发与校验
# ==========================================================================


def _b64url_encode(data: bytes) -> str:
    """标准 base64url，去掉 `=` 填充（避免填充长度歧义，且 `=` 不在 URL 安全集里）。"""
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(text: str) -> bytes:
    """严格 base64url 解码：`validate=True` 拒绝字母表外的字符。

    宽松解码会把非法字符静默丢弃，导致"看起来能解但内容被改过"的 token 通过结构检查，
    必须用严格模式把这类输入挡在 `InvalidToken` 里。
    """
    if not isinstance(text, str):
        raise ValueError(f"base64url 输入必须是字符串，实际: {type(text).__name__}")
    pad = (-len(text)) % 4
    # b64decode 的 altchars 把 `-`/`_` 映射到 `+`/`/`，与签发端一致。
    return base64.b64decode(text + "=" * pad, altchars=b"-_", validate=True)


def _as_str(obj: dict, key: str) -> str:
    if key not in obj:
        raise KeyError(f"缺少字段 {key}")
    value = obj[key]
    if not isinstance(value, str):
        raise TypeError(f"字段 {key} 应为字符串，实际: {type(value).__name__}")
    return value


def _as_int(obj: dict, key: str) -> int:
    if key not in obj:
        raise KeyError(f"缺少字段 {key}")
    value = obj[key]
    # bool 是 int 子类，但 JSON 里 true/false 显然不是时间戳，需显式排除。
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"字段 {key} 应为整数，实际: {type(value).__name__}")
    return value


def _as_scopes(obj: dict) -> frozenset[str]:
    value = obj.get("scopes", [])
    if value is None:
        return frozenset()
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"字段 scopes 应为数组，实际: {type(value).__name__}")
    for item in value:
        if not isinstance(item, str):
            raise TypeError(f"scopes 元素应为字符串，实际: {type(item).__name__}")
    return frozenset(value)


class TokenSigner:
    """签发并校验一次性面试 token。

    Args:
        secret: HMAC 密钥（≥16 字节）。为空/过短直接 `ValueError`——
            弱密钥会让签名形同虚设，宁可在构造期就炸掉。
        clock: 返回当前时间（**秒**）的可调用对象。不注入就用不了确定性测试。
        nonce_store: 防重放端口。不传则退化为**本实例内**的内存集合
            （同实例防重放仍成立）；跨实例/多 worker 必须注入共享实现。
        default_ttl_s: `issue` 未显式给 `ttl_s` 时的默认有效期（秒）。
    """

    def __init__(
        self,
        secret: bytes,
        clock: Callable[[], float],
        *,
        nonce_store: Optional[NonceStore] = None,
        default_ttl_s: int = DEFAULT_TTL_S,
    ) -> None:
        if not isinstance(secret, (bytes, bytearray)):
            raise TypeError(f"secret 必须是 bytes，实际: {type(secret).__name__}")
        if not secret:
            raise ValueError("secret 不能为空：空密钥任何人都能伪造签名")
        if len(secret) < MIN_SECRET_BYTES:
            raise ValueError(
                f"secret 过短（{len(secret)} 字节），至少 {MIN_SECRET_BYTES} 字节"
            )
        if not callable(clock):
            raise TypeError("clock 必须是可调用对象（返回秒）")
        if int(default_ttl_s) <= 0:
            raise ValueError(f"default_ttl_s 必须为正，实际: {default_ttl_s}")

        self._secret = bytes(secret)
        self._clock = clock
        self._nonces: NonceStore = (
            nonce_store if nonce_store is not None else InMemoryNonceStore()
        )
        self._default_ttl = int(default_ttl_s)

    # ---------- 签发 ----------

    def issue(
        self,
        *,
        session_id: str,
        tenant_id: str,
        role: Role | str,
        ttl_s: Optional[int] = None,
        scopes: Iterable[str] = (),
        nonce: Optional[str] = None,
    ) -> str:
        """签发一个新的 token 字符串。

        `nonce` 缺省用 `secrets.token_urlsafe` 生成（密码学随机，不可预测）；
        显式传入只用于测试或调用方自行保证唯一性的场景。

        Raises:
            ValueError: 参数非法（会被 `InterviewToken` 的校验捕获）。
        """
        ttl = self._default_ttl if ttl_s is None else int(ttl_s)
        if ttl <= 0:
            raise ValueError(f"ttl_s 必须为正，实际: {ttl}")
        now = int(self._clock())
        role_value = role.value if isinstance(role, Role) else str(role)
        token = InterviewToken(
            session_id=session_id,
            tenant_id=tenant_id,
            role=role_value,
            iat=now,
            exp=now + ttl,
            nonce=nonce if nonce is not None else secrets.token_urlsafe(16),
            scopes=frozenset(scopes),
        )
        return self._encode(token)

    def _encode(self, token: InterviewToken) -> str:
        """把载荷序列化为确定性的原始字节再签名。

        `sort_keys=True` + 固定分隔符让同一载荷的字节表示唯一，签名因此可复现；
        `ensure_ascii=False` 保留中文原样（跨语言解析方也能读），UTF-8 编码后再签。
        """
        body = {
            "session_id": token.session_id,
            "tenant_id": token.tenant_id,
            "role": token.role,
            "iat": token.iat,
            "exp": token.exp,
            "nonce": token.nonce,
            "scopes": sorted(token.scopes),
        }
        raw = json.dumps(
            body, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
        sig = hmac.new(self._secret, raw, hashlib.sha256).digest()
        return f"{_b64url_encode(raw)}.{_b64url_encode(sig)}"

    # ---------- 校验 ----------

    def verify(self, token: str) -> InterviewToken:
        """校验 token 并返回载荷。

        Raises:
            InvalidToken: 结构/编码/JSON/字段非法，**或签名不匹配**。
            TokenNotYetValid: 尚未生效（`clock() < iat`）。
            TokenExpired: 已过期（`clock() > exp`）。
            TokenReplayed: nonce 已被使用过。
        """
        # 1) 结构：必须是恰好两段。
        if not isinstance(token, str):
            raise InvalidToken(f"格式非法：token 必须是字符串，实际: {type(token).__name__}")
        parts = token.split(".")
        if len(parts) != 2:
            raise InvalidToken(
                f"格式非法：应为 base64url(payload).base64url(sig) 两段，实际 {len(parts)} 段"
            )
        payload_b64, sig_b64 = parts
        if not payload_b64 or not sig_b64:
            raise InvalidToken("格式非法：载荷段或签名段为空")

        # 2) 解码（严格 base64url；binascii.Error 是 ValueError 子类，一并捕获）。
        try:
            raw = _b64url_decode(payload_b64)
            given_sig = _b64url_decode(sig_b64)
        except (binascii.Error, ValueError) as exc:
            raise InvalidToken(f"格式非法：不是合法的 base64url 编码（{exc}）") from exc

        # 3) ★ 先验签（常量时间），再解析不可信载荷。
        expected_sig = hmac.new(self._secret, raw, hashlib.sha256).digest()
        if not hmac.compare_digest(expected_sig, given_sig):
            raise InvalidToken("签名不匹配：token 已被篡改或密钥不一致")

        # 4) JSON 解析（签名已通过，此处字段仍可能非法——如版本漂移）。
        try:
            obj = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise InvalidToken(f"格式非法：载荷不是合法 JSON（{exc}）") from exc
        if not isinstance(obj, dict):
            raise InvalidToken(
                f"格式非法：载荷 JSON 应为对象，实际: {type(obj).__name__}"
            )

        # 5) 构造值对象（字段缺失/类型错/语义错都归为 InvalidToken）。
        try:
            parsed = InterviewToken(
                session_id=_as_str(obj, "session_id"),
                tenant_id=_as_str(obj, "tenant_id"),
                role=_as_str(obj, "role"),
                iat=_as_int(obj, "iat"),
                exp=_as_int(obj, "exp"),
                nonce=_as_str(obj, "nonce"),
                scopes=_as_scopes(obj),
            )
        except (KeyError, ValueError, TypeError) as exc:
            raise InvalidToken(f"格式非法：载荷字段非法（{exc}）") from exc

        # 6) 时效。边界：now == iat 视为已生效；now == exp 视为未过期（过期是严格大于）。
        now = self._clock()
        if now < parsed.iat:
            raise TokenNotYetValid(
                f"尚未生效：当前 {now} < 签发时刻 {parsed.iat}"
            )
        if now > parsed.exp:
            raise TokenExpired(f"已过期：当前 {now} > 过期时刻 {parsed.exp}")

        # 7) 防重放：**通过全部检查后才认领**，失败不消耗 nonce（见模块 docstring 第 3 条）。
        if not self._nonces.claim(parsed.nonce):
            raise TokenReplayed(
                f"重放：nonce {parsed.nonce!r} 已被使用过（一次性 token 只能验证一次）"
            )
        return parsed


__all__ = [
    "DEFAULT_TTL_S",
    "MIN_SECRET_BYTES",
    "InMemoryNonceStore",
    "InterviewToken",
    "InvalidToken",
    "NonceStore",
    "TokenError",
    "TokenExpired",
    "TokenNotYetValid",
    "TokenReplayed",
    "TokenSigner",
]
