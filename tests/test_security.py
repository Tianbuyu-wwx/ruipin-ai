"""
安全模块测试
"""

import pytest
from src.security import (
    InputValidator,
    RateLimiter,
    validate_and_sanitize_answer,
    check_rate_limit,
    secure_config,
)


class TestInputValidator:
    """测试输入验证器"""

    def test_validate_text_normal(self):
        """测试正常文本验证"""
        text = "这是一个正常的回答，包含Java、Spring等技术词汇。"
        result = InputValidator.validate_text(text)
        assert result["valid"] is True
        assert result["error"] is None
        assert result["sanitized"] == text

    def test_validate_text_xss(self):
        """测试XSS攻击检测"""
        text = "<script>alert('xss')</script>正常内容"
        result = InputValidator.validate_text(text)
        assert result["valid"] is False
        assert "非法字符" in result["error"]
        assert "<script>" not in result["sanitized"]

    def test_validate_text_sql_injection(self):
        """测试SQL注入检测"""
        text = "DROP TABLE users; SELECT * FROM admin"
        result = InputValidator.validate_text(text)
        assert result["valid"] is False
        assert "SQL注入" in result["error"]

    def test_validate_text_sql_in_code_block(self):
        """测试代码块中的SQL关键词应允许"""
        text = "```sql\nSELECT * FROM table\n```"
        result = InputValidator.validate_text(text)
        assert result["valid"] is True

    def test_validate_text_too_long(self):
        """测试超长文本"""
        text = "A" * 10001
        result = InputValidator.validate_text(text, max_length=10000)
        assert result["valid"] is False
        assert "过长" in result["error"]

    def test_validate_text_directory_traversal(self):
        """测试目录遍历攻击"""
        text = "../../../etc/passwd"
        result = InputValidator.validate_text(text)
        assert result["valid"] is False

    def test_validate_candidate_name_valid(self):
        """测试有效姓名"""
        result = InputValidator.validate_candidate_name("张三")
        assert result["valid"] is True
        assert result["sanitized"] == "张三"

    def test_validate_candidate_name_invalid_chars(self):
        """测试包含非法字符的姓名"""
        result = InputValidator.validate_candidate_name("张三<script>")
        assert result["valid"] is False
        assert "非法字符" in result["error"]

    def test_validate_candidate_name_too_long(self):
        """测试超长姓名"""
        result = InputValidator.validate_candidate_name("A" * 51)
        assert result["valid"] is False
        assert "过长" in result["error"]

    def test_validate_session_id_valid(self):
        """测试有效UUID"""
        valid_uuid = "550e8400-e29b-41d4-a716-446655440000"
        assert InputValidator.validate_session_id(valid_uuid) is True

    def test_validate_session_id_invalid(self):
        """测试无效UUID"""
        assert InputValidator.validate_session_id("invalid-id") is False
        assert InputValidator.validate_session_id("") is False


class TestRateLimiter:
    """测试速率限制器"""

    def test_rate_limit_allowed(self):
        """测试正常请求应被允许"""
        limiter = RateLimiter()
        client_id = "test_client_1"

        for i in range(5):
            assert limiter.is_allowed(client_id, "submit_answer") is True

    def test_rate_limit_blocked(self):
        """测试超出限制应被阻止"""
        limiter = RateLimiter()
        client_id = "test_client_2"
        action = "submit_answer"
        max_requests = limiter._limits[action]["max"]

        for i in range(max_requests):
            assert limiter.is_allowed(client_id, action) is True

        assert limiter.is_allowed(client_id, action) is False

    def test_rate_limit_remaining(self):
        """测试剩余请求数计算"""
        limiter = RateLimiter()
        client_id = "test_client_3"
        action = "submit_answer"
        max_requests = limiter._limits[action]["max"]

        remaining = limiter.get_remaining(client_id, action)
        assert remaining == max_requests

        limiter.is_allowed(client_id, action)
        remaining = limiter.get_remaining(client_id, action)
        assert remaining == max_requests - 1

    def test_rate_limit_different_actions(self):
        """测试不同操作独立计数"""
        limiter = RateLimiter()
        client_id = "test_client_4"

        assert limiter.is_allowed(client_id, "submit_answer") is True
        assert limiter.is_allowed(client_id, "start_interview") is True

    def test_rate_limit_cleanup(self):
        """测试过期清理"""
        limiter = RateLimiter()
        client_id = "test_client_5"

        limiter.is_allowed(client_id, "submit_answer")
        assert len(limiter._requests) >= 0


class TestSecureConfig:
    """测试安全配置"""

    def test_encrypt_decrypt_api_key(self):
        """测试API密钥加密解密"""
        api_key = "sk-test-api-key-12345"
        encrypted = secure_config.encrypt_api_key(api_key)
        decrypted = secure_config.decrypt_api_key(encrypted)
        assert decrypted == api_key
        assert encrypted != api_key


class TestUtilityFunctions:
    """测试便捷函数"""

    def test_validate_and_sanitize_answer(self):
        """测试回答验证函数"""
        is_valid, sanitized, error = validate_and_sanitize_answer("正常回答")
        assert is_valid is True
        assert error is None
        assert sanitized == "正常回答"

    def test_check_rate_limit(self):
        """测试速率限制检查函数"""
        allowed, remaining, reset_time = check_rate_limit("test_client", "start_interview")
        assert isinstance(allowed, bool)
        assert isinstance(remaining, int)
        assert reset_time is not None
