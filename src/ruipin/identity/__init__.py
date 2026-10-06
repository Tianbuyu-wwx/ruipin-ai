"""身份与权限（方案 §12.1）。

对外只暴露这一个入口：

* `tokens` —— 一次性面试链接的签名 token（签发/校验/防重放）。
* `permissions` —— 角色-权限矩阵、多租户隔离、生理数据三重门。

两者都不依赖任何适配器；存储/时钟/防重放集合都通过端口与构造参数注入。
"""

from __future__ import annotations

from .permissions import (
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
from .tokens import (
    DEFAULT_TTL_S,
    MIN_SECRET_BYTES,
    InMemoryNonceStore,
    InterviewToken,
    InvalidToken,
    NonceStore,
    TokenError,
    TokenExpired,
    TokenNotYetValid,
    TokenReplayed,
    TokenSigner,
)

__all__ = [
    "ALL_PERMISSIONS",
    "AUDIT_READ",
    "BANK_WRITE",
    "CONFIG_WRITE",
    "CONSENT_REVOKE",
    "DATA_DELETE",
    "DEFAULT_TTL_S",
    "Forbidden",
    "InMemoryNonceStore",
    "InterviewToken",
    "InvalidToken",
    "MIN_SECRET_BYTES",
    "NonceStore",
    "PHYSIO_READ",
    "Principal",
    "RAW_MEDIA_READ",
    "REPORT_READ",
    "ROLE_PERMISSIONS",
    "Role",
    "SESSION_READ",
    "SESSION_WRITE",
    "TokenError",
    "TokenExpired",
    "TokenNotYetValid",
    "TokenReplayed",
    "TokenSigner",
    "authorize",
    "has_permission",
    "revoke_consent",
]
