"""
错误处理模块测试
"""

import pytest
from unittest.mock import patch, MagicMock
from src.error_handler import (
    ErrorSeverity,
    InterviewException,
    APIException,
    ValidationException,
    DatabaseException,
    DefaultEvaluationFallback,
    RetryPolicy,
    with_retry,
    with_fallback,
    safe_execute,
    error_handler,
    ErrorHandler,
)


class TestExceptions:
    """测试异常类"""

    def test_interview_exception(self):
        """测试基础异常"""
        exc = InterviewException("测试错误", error_code="TEST_ERROR")
        assert str(exc) == "测试错误"
        assert exc.error_code == "TEST_ERROR"
        assert exc.severity == ErrorSeverity.MEDIUM

    def test_api_exception(self):
        """测试API异常"""
        exc = APIException("API调用失败", api_name="deepseek", status_code=500)
        assert exc.api_name == "deepseek"
        assert exc.status_code == 500
        assert exc.severity == ErrorSeverity.HIGH

    def test_validation_exception(self):
        """测试验证异常"""
        exc = ValidationException("字段错误", field="name")
        assert exc.field == "name"
        assert exc.severity == ErrorSeverity.LOW

    def test_database_exception(self):
        """测试数据库异常"""
        exc = DatabaseException("连接失败", operation="query")
        assert exc.operation == "query"
        assert exc.severity == ErrorSeverity.HIGH


class TestFallbackStrategies:
    """测试降级策略"""

    def test_default_evaluation_fallback_short_answer(self):
        """测试短回答降级"""
        fallback = DefaultEvaluationFallback()
        result = fallback.execute(answer="短")
        assert result["score"] == 40
        assert "简短" in result["feedback"]
        assert result["fallback"] is True

    def test_default_evaluation_fallback_medium_answer(self):
        """测试中长回答降级"""
        fallback = DefaultEvaluationFallback()
        result = fallback.execute(answer="这是一个中等长度的回答内容。")
        assert 40 <= result["score"] <= 70

    def test_default_evaluation_fallback_long_answer(self):
        """测试长回答降级"""
        fallback = DefaultEvaluationFallback()
        long_answer = "这是一个非常详细的回答。" * 20
        result = fallback.execute(answer=long_answer)
        assert result["score"] == 80
        assert "详细完整" in result["feedback"]


class TestRetryPolicy:
    """测试重试策略"""

    def test_calculate_delay(self):
        """测试延迟计算"""
        policy = RetryPolicy(base_delay=1.0, exponential_base=2.0)
        assert policy.calculate_delay(0) == 1.0
        assert policy.calculate_delay(1) == 2.0
        assert policy.calculate_delay(2) == 4.0

    def test_calculate_delay_max_cap(self):
        """测试延迟上限"""
        policy = RetryPolicy(base_delay=1.0, max_delay=5.0, exponential_base=10.0)
        assert policy.calculate_delay(1) == 5.0

    def test_should_retry(self):
        """测试重试判断"""
        policy = RetryPolicy(max_retries=3)
        assert policy.should_retry(APIException("test"), 0) is True
        assert policy.should_retry(APIException("test"), 3) is False


class TestDecorators:
    """测试装饰器"""

    def test_with_retry_success(self):
        """测试重试装饰器成功"""
        call_count = 0

        @with_retry(RetryPolicy(max_retries=2, base_delay=0.01))
        def success_after_fail():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise APIException("临时错误")
            return {"score": 100}

        result = success_after_fail()
        assert result["score"] == 100
        assert call_count == 2

    def test_with_retry_exhausted(self):
        """测试重试耗尽"""
        @with_retry(RetryPolicy(max_retries=1, base_delay=0.01))
        def always_fail():
            raise APIException("持续错误")

        with pytest.raises(APIException):
            always_fail()

    def test_with_fallback(self):
        """测试降级装饰器"""
        fallback = DefaultEvaluationFallback()

        @with_fallback(fallback)
        def failing_function():
            raise Exception("模拟失败")

        result = failing_function()
        assert "score" in result
        assert result["fallback"] is True

    def test_safe_execute_success(self):
        """测试安全执行成功"""
        result = safe_execute(lambda: 42, default_return=0)
        assert result == 42

    def test_safe_execute_failure(self):
        """测试安全执行失败"""
        result = safe_execute(lambda: 1/0, default_return=0, error_message="除零错误")
        assert result == 0


class TestErrorHandler:
    """测试错误处理器"""

    def test_handle_error(self):
        """测试错误处理"""
        handler = ErrorHandler()
        exc = InterviewException("测试错误")
        context = handler.handle_error(exc, "test_function")

        assert context.error_type == "InterviewException"
        assert context.function_name == "test_function"
        assert context.severity == ErrorSeverity.MEDIUM

    def test_error_stats(self):
        """测试错误统计"""
        handler = ErrorHandler()
        handler.handle_error(InterviewException("错误1"), "func1")
        handler.handle_error(APIException("错误2"), "func2")

        stats = handler.get_error_stats()
        assert stats["total_errors"] == 2
        assert "InterviewException" in stats["error_counts"]
        assert "APIException" in stats["error_counts"]

    def test_is_healthy(self):
        """测试健康检查"""
        handler = ErrorHandler()
        assert handler.is_healthy() is True

        for i in range(5):
            handler.handle_error(
                InterviewException(f"致命错误{i}"),
                "func",
            )

        assert handler.is_healthy() is True
