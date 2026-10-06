"""
锐聘AI - 安全性模块
提供输入验证、API密钥加密、XSS防护等安全功能
"""

import os
import re
import hashlib
import secrets
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime, timedelta

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

from .logger import logger


class SecureConfig:
    """安全配置管理器"""
    
    def __init__(self):
        self._encryption_key = self._get_or_create_encryption_key()
        self._cipher = Fernet(self._encryption_key)
    
    def _get_or_create_encryption_key(self) -> bytes:
        """获取或创建加密密钥"""
        key_file = os.path.join(os.path.dirname(__file__), '..', '.secret_key')
        
        try:
            if os.path.exists(key_file):
                with open(key_file, 'rb') as f:
                    key = f.read()
                # 验证密钥格式
                if len(key) != 44:  # Fernet密钥长度
                    raise ValueError("密钥格式无效")
                return key
            else:
                # 生成新密钥
                key = Fernet.generate_key()
                # 确保目录存在
                os.makedirs(os.path.dirname(key_file), exist_ok=True)
                with open(key_file, 'wb') as f:
                    f.write(key)
                # 跨平台权限设置
                try:
                    os.chmod(key_file, 0o600)
                except Exception:
                    pass  # Windows可能不支持
                logger.info("[OK] 生成新的加密密钥")
                return key
        except Exception as e:
            logger.error(f"密钥管理失败: {e}")
            # 使用临时内存密钥（不推荐用于生产环境）
            return Fernet.generate_key()
    
    def encrypt_api_key(self, api_key: str) -> str:
        """加密API密钥"""
        return self._cipher.encrypt(api_key.encode()).decode()
    
    def decrypt_api_key(self, encrypted_key: str) -> str:
        """解密API密钥"""
        return self._cipher.decrypt(encrypted_key.encode()).decode()
    
    def get_encrypted_env(self, key: str, default: str = "") -> str:
        """从环境变量获取加密值并解密"""
        encrypted = os.environ.get(f"ENCRYPTED_{key}", "")
        if encrypted:
            try:
                return self.decrypt_api_key(encrypted)
            except Exception:
                pass
        return os.environ.get(key, default)


class InputValidator:
    """输入验证器"""
    
    # 危险字符模式
    DANGEROUS_PATTERNS = [
        r'<script[^>]*>.*?</script>',  # XSS脚本
        r'javascript:',  # JS协议
        r'on\w+\s*=',  # 事件处理器
        r'data:text/html',  # data URI
        r'\.\./',  # 目录遍历
        r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]',  # 控制字符
    ]
    
    # SQL注入关键词
    SQL_KEYWORDS = [
        'SELECT', 'INSERT', 'UPDATE', 'DELETE', 'DROP',
        'UNION', 'ALTER', 'CREATE', 'EXEC', 'EXECUTE'
    ]
    
    @classmethod
    def validate_text(cls, text: str, max_length: int = 10000, field_name: str = "文本") -> Dict[str, Any]:
        """
        验证文本输入
        
        Returns:
            {"valid": bool, "error": str or None, "sanitized": str}
        """
        result = {"valid": True, "error": None, "sanitized": text}
        
        if not text:
            return result
        
        # 检查长度
        if len(text) > max_length:
            result["valid"] = False
            result["error"] = f"{field_name}过长，最大允许{max_length}字符"
            return result
        
        # 检查危险模式
        for pattern in cls.DANGEROUS_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE | re.DOTALL):
                result["valid"] = False
                result["error"] = f"{field_name}包含非法字符或脚本"
                # 清理危险内容
                result["sanitized"] = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
                return result
        
        # 检查SQL注入
        upper_text = text.upper()
        for keyword in cls.SQL_KEYWORDS:
            if keyword in upper_text:
                # 检查是否是合法的SQL关键词使用（如在代码示例中）
                if not cls._is_safe_sql_usage(text, keyword):
                    result["valid"] = False
                    result["error"] = f"{field_name}可能包含SQL注入攻击"
                    result["sanitized"] = cls._sanitize_sql(text)
                    return result
        
        return result
    
    @classmethod
    def _is_safe_sql_usage(cls, text: str, keyword: str) -> bool:
        """判断SQL关键词是否是安全使用"""
        # 如果在代码块中，通常是安全的
        if '```' in text or '`' in text:
            return True
        # 如果在引号中，可能是示例
        if f'"{keyword}"' in text or f"'{keyword}'" in text:
            return True
        return False
    
    @classmethod
    def _sanitize_sql(cls, text: str) -> str:
        """清理SQL注入"""
        for keyword in cls.SQL_KEYWORDS:
            text = re.sub(rf'\b{keyword}\b', f'[{keyword}]', text, flags=re.IGNORECASE)
        return text
    
    @classmethod
    def validate_candidate_name(cls, name: str) -> Dict[str, Any]:
        """验证候选人姓名"""
        if not name or not name.strip():
            return {"valid": False, "error": "姓名不能为空", "sanitized": ""}
        
        if len(name) > 50:
            return {"valid": False, "error": "姓名过长", "sanitized": name[:50]}
        
        # 只允许中文、英文、空格、常见分隔符
        if not re.match(r'^[\u4e00-\u9fa5a-zA-Z\s\-\.]+$', name):
            return {"valid": False, "error": "姓名包含非法字符", "sanitized": re.sub(r'[^\u4e00-\u9fa5a-zA-Z\s\-\.]', '', name)}
        
        return {"valid": True, "error": None, "sanitized": name.strip()}
    
    @classmethod
    def validate_session_id(cls, session_id: str) -> bool:
        """验证会话ID格式"""
        if not session_id:
            return False
        # UUID格式验证
        return bool(re.match(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', session_id))
    
    @classmethod
    def sanitize_html(cls, text: str) -> str:
        """清理HTML标签"""
        # 允许的基本标签
        allowed_tags = ['b', 'i', 'u', 'strong', 'em', 'br', 'p']
        
        # 移除所有标签，然后重新添加允许的
        text = re.sub(r'<[^>]+>', '', text)
        
        return text


class RateLimiter:
    """速率限制器 - 带自动清理机制"""
    
    def __init__(self):
        self._requests: Dict[str, List[datetime]] = {}
        self._limits = {
            "submit_answer": {"max": 10, "window": 60},  # 每分钟10次
            "start_interview": {"max": 5, "window": 300},  # 每5分钟5次
            "api_call": {"max": 100, "window": 60},  # 每分钟100次API调用
        }
        self._last_cleanup = datetime.now()
        self._cleanup_interval = 300  # 每5分钟清理一次
    
    def _cleanup_expired(self):
        """清理所有过期的请求记录，防止内存泄漏"""
        now = datetime.now()
        # 只在间隔时间到达时清理
        if (now - self._last_cleanup).total_seconds() < self._cleanup_interval:
            return
        
        self._last_cleanup = now
        max_window = max(l["window"] for l in self._limits.values())
        cutoff = now - timedelta(seconds=max_window * 2)  # 保留2倍最大窗口
        
        expired_keys = []
        for key, timestamps in self._requests.items():
            valid = [t for t in timestamps if t > cutoff]
            if valid:
                self._requests[key] = valid
            else:
                expired_keys.append(key)
        
        # 删除完全过期的键
        for key in expired_keys:
            del self._requests[key]
        
        if expired_keys:
            logger.debug(f"速率限制器清理: 移除 {len(expired_keys)} 个过期客户端")
    
    def is_allowed(self, client_id: str, action: str) -> bool:
        """
        检查是否允许执行操作
        
        Args:
            client_id: 客户端标识（IP或用户ID）
            action: 操作类型
            
        Returns:
            是否允许
        """
        # 定期清理
        self._cleanup_expired()
        
        if action not in self._limits:
            return True
        
        limit = self._limits[action]
        key = f"{client_id}:{action}"
        now = datetime.now()
        window_start = now - timedelta(seconds=limit["window"])
        
        # 清理过期记录
        if key in self._requests:
            self._requests[key] = [
                t for t in self._requests[key] 
                if t > window_start
            ]
        else:
            self._requests[key] = []
        
        # 检查是否超过限制
        if len(self._requests[key]) >= limit["max"]:
            logger.warning(f"速率限制触发: {client_id} - {action}")
            return False
        
        # 记录本次请求
        self._requests[key].append(now)
        return True
    
    def get_remaining(self, client_id: str, action: str) -> int:
        """获取剩余请求次数"""
        if action not in self._limits:
            return -1  # 无限制
        
        limit = self._limits[action]
        key = f"{client_id}:{action}"
        now = datetime.now()
        window_start = now - timedelta(seconds=limit["window"])
        
        if key in self._requests:
            recent = [t for t in self._requests[key] if t > window_start]
            return max(0, limit["max"] - len(recent))
        
        return limit["max"]


class AuditLogger:
    """审计日志器"""
    
    @staticmethod
    def log_security_event(event_type: str, details: Dict[str, Any]):
        """记录安全事件"""
        logger.warning(f"[SECURITY] {event_type}: {details}")
    
    @staticmethod
    def log_api_access(client_id: str, endpoint: str, status: str, duration_ms: float):
        """记录API访问"""
        logger.info(f"[API] {client_id} -> {endpoint} | {status} | {duration_ms}ms")
    
    @staticmethod
    def log_data_access(user_id: str, action: str, data_type: str, record_id: str):
        """记录数据访问"""
        logger.info(f"[DATA] {user_id} {action} {data_type}/{record_id}")


# 全局安全实例
secure_config = SecureConfig()
input_validator = InputValidator()
rate_limiter = RateLimiter()
audit_logger = AuditLogger()


def validate_and_sanitize_answer(answer: str, max_length: int = 10000) -> tuple:
    """
    验证并清理回答内容
    
    Returns:
        (is_valid, sanitized_text, error_message)
    """
    result = InputValidator.validate_text(answer, max_length, "回答")
    return result["valid"], result["sanitized"], result["error"]


def check_rate_limit(client_id: str, action: str = "submit_answer") -> tuple:
    """
    检查速率限制
    
    Returns:
        (is_allowed, remaining, reset_time)
    """
    allowed = rate_limiter.is_allowed(client_id, action)
    remaining = rate_limiter.get_remaining(client_id, action)
    
    limit = rate_limiter._limits.get(action, {})
    reset_time = datetime.now() + timedelta(seconds=limit.get("window", 60))
    
    return allowed, remaining, reset_time



