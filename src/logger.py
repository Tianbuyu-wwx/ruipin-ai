"""
锐聘AI - 日志配置模块
"""

import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logger(name: str = "ruipin_ai", log_dir: str = "logs") -> logging.Logger:
    """
    设置日志记录器
    
    Args:
        name: 日志记录器名称
        log_dir: 日志文件存放目录
        
    Returns:
        配置好的日志记录器
    """
    # 创建日志目录
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    # 创建日志记录器
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # 如果已经配置过处理器，直接返回
    if logger.handlers:
        return logger
    
    # 日志格式 - 使用ASCII字符避免编码问题
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 控制台处理器 - 使用UTF-8编码
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件处理器 - 按日期分割
    current_date = datetime.now().strftime("%Y-%m-%d")
    log_file = log_path / f"ruipin_ai_{current_date}.log"
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # 错误日志单独记录
    error_log_file = log_path / f"ruipin_ai_error_{current_date}.log"
    error_handler = logging.FileHandler(error_log_file, encoding='utf-8')
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    logger.addHandler(error_handler)
    
    logger.info(f"[OK] 日志系统初始化完成 | 日志目录: {log_path.absolute()}")
    return logger


# 全局日志记录器
logger = setup_logger()


class LogContext:
    """日志上下文管理器，用于记录函数执行时间和异常"""
    
    def __init__(self, logger: logging.Logger, operation: str, log_params: bool = False):
        self.logger = logger
        self.operation = operation
        self.log_params = log_params
        self.start_time = None
        
    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.info(f"[START] {self.operation}")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        if exc_type is None:
            self.logger.info(f"[DONE] {self.operation} | 耗时: {duration:.3f}秒")
        else:
            self.logger.error(
                f"[ERROR] {self.operation} | 耗时: {duration:.3f}秒 | "
                f"异常类型: {exc_type.__name__} | 异常信息: {exc_val}",
                exc_info=True
            )
        
        # 返回 False 表示不抑制异常
        return False


def log_function_call(logger: logging.Logger = None):
    """
    函数调用日志装饰器
    
    用法:
        @log_function_call()
        def my_function():
            pass
    """
    if logger is None:
        logger = logging.getLogger("ruipin_ai")
        
    def decorator(func):
        def wrapper(*args, **kwargs):
            func_name = func.__name__
            module_name = func.__module__
            
            logger.debug(f"[CALL] {module_name}.{func_name}")
            
            try:
                with LogContext(logger, f"{module_name}.{func_name}"):
                    result = func(*args, **kwargs)
                    return result
            except Exception as e:
                logger.error(
                    f"[FUNC_ERROR] {module_name}.{func_name} | "
                    f"异常: {type(e).__name__}: {str(e)}",
                    exc_info=True
                )
                raise
                
        return wrapper
    return decorator


# 性能日志记录器
class PerformanceLogger:
    """性能日志记录器，用于记录系统性能指标"""
    
    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger("ruipin_ai")
        self.metrics = {}
        
    def record(self, metric_name: str, value: float, unit: str = ""):
        """记录性能指标"""
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        self.metrics[metric_name].append(value)
        
        unit_str = f" {unit}" if unit else ""
        self.logger.debug(f"[PERF] {metric_name}: {value:.3f}{unit_str}")
        
    def get_average(self, metric_name: str) -> float:
        """获取性能指标平均值"""
        if metric_name in self.metrics and self.metrics[metric_name]:
            return sum(self.metrics[metric_name]) / len(self.metrics[metric_name])
        return 0.0
        
    def log_summary(self):
        """记录性能摘要"""
        self.logger.info("=" * 50)
        self.logger.info("性能统计摘要")
        self.logger.info("=" * 50)
        
        for metric_name, values in self.metrics.items():
            if values:
                avg = sum(values) / len(values)
                min_val = min(values)
                max_val = max(values)
                count = len(values)
                
                self.logger.info(
                    f"{metric_name}: 平均={avg:.3f}, 最小={min_val:.3f}, "
                    f"最大={max_val:.3f}, 次数={count}"
                )
        
        self.logger.info("=" * 50)


# 创建性能日志记录器实例
performance_logger = PerformanceLogger()
