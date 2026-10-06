"""
锐聘AI - 错误处理和降级机制
提供统一的异常处理、重试逻辑和降级策略
"""

import time
import functools
import traceback
from typing import Callable, Any, Optional, Type, Dict, Tuple
from dataclasses import dataclass
from enum import Enum

from .logger import logger


class ErrorSeverity(Enum):
    """错误严重级别"""
    LOW = "low"           # 轻微错误，可忽略
    MEDIUM = "medium"     # 中等错误，需要记录
    HIGH = "high"         # 严重错误，需要告警
    CRITICAL = "critical" # 致命错误，服务可能不可用


@dataclass
class ErrorContext:
    """错误上下文"""
    error_type: str
    error_message: str
    severity: ErrorSeverity
    timestamp: float
    function_name: str
    retry_count: int = 0
    max_retries: int = 0
    fallback_used: bool = False


class InterviewException(Exception):
    """面试系统基础异常"""
    def __init__(self, message: str, error_code: str = "INTERVIEW_ERROR", details: Optional[Dict] = None):
        super().__init__(message)
        self.error_code = error_code
        self.details = details or {}
        self.severity = ErrorSeverity.MEDIUM


class APIException(InterviewException):
    """API调用异常"""
    def __init__(self, message: str, api_name: str = "", status_code: int = 0):
        super().__init__(message, error_code="API_ERROR")
        self.api_name = api_name
        self.status_code = status_code
        self.severity = ErrorSeverity.HIGH


class ValidationException(InterviewException):
    """数据验证异常"""
    def __init__(self, message: str, field: str = ""):
        super().__init__(message, error_code="VALIDATION_ERROR")
        self.field = field
        self.severity = ErrorSeverity.LOW


class DatabaseException(InterviewException):
    """数据库异常"""
    def __init__(self, message: str, operation: str = ""):
        super().__init__(message, error_code="DATABASE_ERROR")
        self.operation = operation
        self.severity = ErrorSeverity.HIGH


class FallbackStrategy:
    """降级策略基类"""
    
    def execute(self, *args, **kwargs) -> Any:
        """执行降级逻辑"""
        raise NotImplementedError


class DefaultEvaluationFallback(FallbackStrategy):
    """默认评估降级策略"""
    
    def execute(self, answer: str = "", **kwargs) -> Dict[str, Any]:
        """返回默认评估结果"""
        logger.warning("[FALLBACK] 使用默认评估策略")
        
        length = len(answer) if answer else 0
        if length < 20:
            score = 40
            feedback = "回答过于简短，请详细说明。"
        elif length < 50:
            score = 55
            feedback = "回答尚可，但可以更加详细。"
        elif length < 100:
            score = 70
            feedback = "回答不错，涵盖了主要内容。"
        else:
            score = 80
            feedback = "回答详细完整，表达清晰。"
        
        return {
            "score": float(score),
            "technical": score,
            "communication": score,
            "completeness": score,
            "problem_solving": score,
            "teamwork": score,
            "leadership": score,
            "feedback": feedback + "（备用评估）",
            "fallback": True
        }


class CachedEvaluationFallback(FallbackStrategy):
    """缓存评估降级策略"""
    
    def __init__(self, cache: Optional[Dict] = None):
        self.cache = cache or {}
    
    def execute(self, cache_key: str = "", **kwargs) -> Optional[Dict[str, Any]]:
        """从缓存获取评估结果"""
        if cache_key and cache_key in self.cache:
            logger.info(f"[FALLBACK] 使用缓存结果: {cache_key}")
            return self.cache[cache_key]
        return None


class RetryPolicy:
    """重试策略"""
    
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        exponential_base: float = 2.0,
        retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,)
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.retryable_exceptions = retryable_exceptions
    
    def calculate_delay(self, attempt: int) -> float:
        """计算重试延迟"""
        delay = self.base_delay * (self.exponential_base ** attempt)
        return min(delay, self.max_delay)
    
    def should_retry(self, exception: Exception, attempt: int) -> bool:
        """判断是否应该重试"""
        if attempt >= self.max_retries:
            return False
        return isinstance(exception, self.retryable_exceptions)


# 默认重试策略
DEFAULT_RETRY_POLICY = RetryPolicy(
    max_retries=3,
    base_delay=1.0,
    max_delay=10.0,
    retryable_exceptions=(APIException, ConnectionError, TimeoutError)
)


def with_retry(
    retry_policy: Optional[RetryPolicy] = None,
    fallback: Optional[FallbackStrategy] = None
):
    """
    重试装饰器
    
    Args:
        retry_policy: 重试策略，默认使用DEFAULT_RETRY_POLICY
        fallback: 降级策略
    """
    policy = retry_policy or DEFAULT_RETRY_POLICY
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(policy.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                    
                except Exception as e:
                    last_exception = e
                    
                    if not policy.should_retry(e, attempt):
                        logger.error(f"[RETRY] 不可重试的错误: {e}")
                        break
                    
                    delay = policy.calculate_delay(attempt)
                    logger.warning(
                        f"[RETRY] 第{attempt + 1}次重试 {func.__name__} "
                        f"({type(e).__name__}: {e})，{delay:.1f}秒后重试..."
                    )
                    time.sleep(delay)
            
            # 所有重试失败，使用降级策略
            if fallback:
                logger.warning(f"[FALLBACK] 使用降级策略: {type(fallback).__name__}")
                try:
                    result = fallback.execute(*args, **kwargs)
                    if result is not None:
                        return result
                except Exception as fallback_error:
                    logger.error(f"[FALLBACK] 降级策略也失败: {fallback_error}")
            
            # 降级也失败，抛出原始异常
            raise last_exception
        
        return wrapper
    return decorator


def with_fallback(fallback_strategy: FallbackStrategy):
    """
    降级装饰器
    
    当主逻辑失败时，自动使用降级策略
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"[FALLBACK] {func.__name__} 失败: {e}，使用降级策略")
                try:
                    return fallback_strategy.execute(*args, **kwargs)
                except Exception as fallback_error:
                    logger.error(f"[FALLBACK] 降级策略失败: {fallback_error}")
                    raise
        return wrapper
    return decorator


def safe_execute(
    func: Callable,
    *args,
    default_return: Any = None,
    error_message: str = "执行失败",
    **kwargs
) -> Any:
    """
    安全执行函数
    
    捕获所有异常，返回默认值
    
    Args:
        func: 要执行的函数
        default_return: 失败时的默认返回值
        error_message: 错误日志消息
        
    Returns:
        函数返回值或默认值
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger.error(f"[SAFE_EXECUTE] {error_message}: {e}")
        return default_return


class ErrorHandler:
    """全局错误处理器"""
    
    def __init__(self):
        self.error_counts: Dict[str, int] = {}
        self.error_history: list = []
        self.max_history = 100
    
    def handle_error(self, error: Exception, context: str = "") -> ErrorContext:
        """
        处理错误
        
        Args:
            error: 异常对象
            context: 错误上下文
            
        Returns:
            错误上下文对象
        """
        error_type = type(error).__name__
        error_message = str(error)
        
        # 确定严重级别
        if isinstance(error, InterviewException):
            severity = error.severity
        elif isinstance(error, (ConnectionError, TimeoutError)):
            severity = ErrorSeverity.HIGH
        else:
            severity = ErrorSeverity.MEDIUM
        
        # 记录错误统计
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        # 创建错误上下文
        error_context = ErrorContext(
            error_type=error_type,
            error_message=error_message,
            severity=severity,
            timestamp=time.time(),
            function_name=context
        )
        
        # 添加到历史
        self.error_history.append(error_context)
        if len(self.error_history) > self.max_history:
            self.error_history.pop(0)
        
        # 记录日志
        log_message = f"[{severity.value.upper()}] {context}: {error_type} - {error_message}"
        if severity in (ErrorSeverity.HIGH, ErrorSeverity.CRITICAL):
            logger.error(log_message, exc_info=True)
        elif severity == ErrorSeverity.MEDIUM:
            logger.warning(log_message)
        else:
            logger.info(log_message)
        
        return error_context
    
    def get_error_stats(self) -> Dict[str, Any]:
        """获取错误统计"""
        return {
            "total_errors": sum(self.error_counts.values()),
            "error_counts": self.error_counts.copy(),
            "recent_errors": [
                {
                    "type": e.error_type,
                    "severity": e.severity.value,
                    "time": e.timestamp,
                    "function": e.function_name
                }
                for e in self.error_history[-10:]
            ]
        }
    
    def is_healthy(self) -> bool:
        """检查系统健康状态"""
        # 如果最近有致命错误，认为不健康
        recent_critical = [
            e for e in self.error_history[-10:]
            if e.severity == ErrorSeverity.CRITICAL
        ]
        return len(recent_critical) < 3


# 全局错误处理器
error_handler = ErrorHandler()


def handle_errors(context: str = "", severity: ErrorSeverity = ErrorSeverity.MEDIUM):
    """
    错误处理装饰器
    
    自动捕获并记录错误
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_handler.handle_error(e, context or func.__name__)
                raise
        return wrapper
    return decorator



