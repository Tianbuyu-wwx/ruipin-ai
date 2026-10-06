"""
锐聘AI - 资源监控模块

监控系统资源指标，为降级决策提供数据支持
"""

import time
import threading
import psutil
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Callable
from collections import deque

from ..logger import logger


@dataclass
class ResourceMetrics:
    """系统资源监控指标"""

    timestamp: float = field(default_factory=time.time)

    api_response_time: float = 0.0
    api_error_rate: float = 0.0
    api_availability: bool = True
    api_consecutive_failures: int = 0

    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    disk_usage: float = 0.0
    gpu_memory: float = 0.0
    gpu_utilization: float = 0.0

    model_loaded: bool = False
    whisper_available: bool = False
    database_connected: bool = False

    queue_depth: int = 0
    active_sessions: int = 0
    requests_per_minute: int = 0

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "api": {
                "response_time_ms": round(self.api_response_time, 2),
                "error_rate": round(self.api_error_rate, 4),
                "availability": self.api_availability,
                "consecutive_failures": self.api_consecutive_failures,
            },
            "system": {
                "cpu_usage": round(self.cpu_usage, 4),
                "memory_usage": round(self.memory_usage, 4),
                "disk_usage": round(self.disk_usage, 4),
                "gpu_memory": round(self.gpu_memory, 4),
                "gpu_utilization": round(self.gpu_utilization, 4),
            },
            "services": {
                "model_loaded": self.model_loaded,
                "whisper_available": self.whisper_available,
                "database_connected": self.database_connected,
            },
            "business": {
                "queue_depth": self.queue_depth,
                "active_sessions": self.active_sessions,
                "requests_per_minute": self.requests_per_minute,
            },
        }


class ResourceMonitor:
    """资源监控器 - 持续收集系统资源指标"""

    def __init__(self, history_size: int = 100):
        self.history: deque = deque(maxlen=history_size)
        self.current_metrics = ResourceMetrics()
        self._lock = threading.RLock()
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._interval = 5.0

        self._api_response_times: deque = deque(maxlen=60)
        self._api_errors: deque = deque(maxlen=60)
        self._request_times: deque = deque(maxlen=60)

        self._custom_collectors: List[Callable[[], Dict]] = []

        logger.info("[OK] 资源监控器初始化完成")

    def register_custom_collector(self, collector: Callable[[], Dict]):
        self._custom_collectors.append(collector)
        name = getattr(collector, '__name__', str(collector))
        logger.info(f"注册自定义指标收集器: {name}")

    def start(self):
        if self._running:
            return
        self._running = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("[OK] 资源监控线程已启动")

    def stop(self):
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2)
        logger.info("[OK] 资源监控线程已停止")

    def _monitor_loop(self):
        while self._running:
            try:
                metrics = self.collect_metrics()
                with self._lock:
                    self.current_metrics = metrics
                    self.history.append(metrics)
                time.sleep(self._interval)
            except Exception as e:
                logger.error(f"监控循环异常: {e}")
                time.sleep(self._interval)

    def collect_metrics(self) -> ResourceMetrics:
        metrics = ResourceMetrics()
        metrics.timestamp = time.time()

        try:
            metrics.cpu_usage = psutil.cpu_percent(interval=0.1) / 100.0
        except Exception as e:
            logger.warning(f"CPU监控失败: {e}")

        try:
            mem = psutil.virtual_memory()
            metrics.memory_usage = mem.percent / 100.0
        except Exception as e:
            logger.warning(f"内存监控失败: {e}")

        try:
            disk = psutil.disk_usage("/")
            metrics.disk_usage = disk.percent / 100.0
        except Exception as e:
            logger.warning(f"磁盘监控失败: {e}")

        try:
            metrics.gpu_memory = self._get_gpu_memory_usage()
            metrics.gpu_utilization = self._get_gpu_utilization()
        except Exception as e:
            logger.debug(f"GPU监控失败: {e}")

        with self._lock:
            if self._api_response_times:
                metrics.api_response_time = sum(self._api_response_times) / len(self._api_response_times)
            if self._api_errors:
                metrics.api_error_rate = sum(self._api_errors) / len(self._api_errors)

        for collector in self._custom_collectors:
            try:
                custom_data = collector()
                for key, value in custom_data.items():
                    if hasattr(metrics, key):
                        setattr(metrics, key, value)
            except Exception as e:
                logger.warning(f"自定义收集器 {collector.__name__} 失败: {e}")

        return metrics

    def _get_gpu_memory_usage(self) -> float:
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return info.used / info.total if info.total > 0 else 0.0
        except Exception:
            return 0.0

    def _get_gpu_utilization(self) -> float:
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return util.gpu / 100.0
        except Exception:
            return 0.0

    def record_api_call(self, response_time_ms: float, success: bool):
        with self._lock:
            self._api_response_times.append(response_time_ms)
            self._api_errors.append(0.0 if success else 1.0)
            if success:
                self.current_metrics.api_consecutive_failures = 0
            else:
                self.current_metrics.api_consecutive_failures += 1
            if self._api_response_times:
                self.current_metrics.api_response_time = sum(self._api_response_times) / len(self._api_response_times)
            if self._api_errors:
                self.current_metrics.api_error_rate = sum(self._api_errors) / len(self._api_errors)
            self.history.append(self.current_metrics)

    def record_request(self):
        with self._lock:
            self._request_times.append(time.time())
            now = time.time()
            recent = [t for t in self._request_times if now - t < 60]
            self.current_metrics.requests_per_minute = len(recent)

    def update_service_status(self, model_loaded: Optional[bool] = None,
                              whisper_available: Optional[bool] = None,
                              database_connected: Optional[bool] = None):
        with self._lock:
            if model_loaded is not None:
                self.current_metrics.model_loaded = model_loaded
            if whisper_available is not None:
                self.current_metrics.whisper_available = whisper_available
            if database_connected is not None:
                self.current_metrics.database_connected = database_connected

    def update_business_metrics(self, queue_depth: Optional[int] = None,
                                 active_sessions: Optional[int] = None):
        with self._lock:
            if queue_depth is not None:
                self.current_metrics.queue_depth = queue_depth
            if active_sessions is not None:
                self.current_metrics.active_sessions = active_sessions

    def get_current_metrics(self) -> ResourceMetrics:
        with self._lock:
            return self.current_metrics

    def get_metrics_history(self, count: int = 10) -> List[ResourceMetrics]:
        with self._lock:
            return list(self.history)[-count:]

    def get_average_metrics(self, duration_seconds: int = 60) -> Optional[ResourceMetrics]:
        with self._lock:
            cutoff = time.time() - duration_seconds
            recent = [m for m in self.history if m.timestamp >= cutoff]
            if not recent:
                recent = list(self.history)
            if not recent:
                return None

            avg = ResourceMetrics()
            avg.timestamp = time.time()
            n = len(recent)

            avg.api_response_time = sum(m.api_response_time for m in recent) / n
            avg.api_error_rate = sum(m.api_error_rate for m in recent) / n
            avg.cpu_usage = sum(m.cpu_usage for m in recent) / n
            avg.memory_usage = sum(m.memory_usage for m in recent) / n
            avg.gpu_memory = sum(m.gpu_memory for m in recent) / n

            avg.api_availability = all(m.api_availability for m in recent)
            avg.model_loaded = recent[-1].model_loaded
            avg.whisper_available = recent[-1].whisper_available
            avg.database_connected = recent[-1].database_connected

            return avg
