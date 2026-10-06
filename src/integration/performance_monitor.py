"""
锐聘AI - 性能监控器

监控系统性能指标，识别性能瓶颈
"""

import time
import threading
import psutil
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque

from ..logger import logger


@dataclass
class PerformanceMetrics:
    """性能指标"""
    timestamp: float = field(default_factory=time.time)

    # 响应时间
    api_latency_ms: float = 0.0
    evaluation_latency_ms: float = 0.0
    total_processing_time_ms: float = 0.0

    # 吞吐量
    requests_per_second: float = 0.0
    evaluations_per_minute: int = 0

    # 资源使用
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_mb: float = 0.0
    gpu_memory_mb: float = 0.0
    gpu_utilization: float = 0.0

    # 队列状态
    queue_depth: int = 0
    active_threads: int = 0
    pending_evaluations: int = 0

    # 错误率
    error_rate: float = 0.0
    timeout_rate: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "timestamp": datetime.fromtimestamp(self.timestamp).isoformat(),
            "latency": {
                "api_ms": round(self.api_latency_ms, 2),
                "evaluation_ms": round(self.evaluation_latency_ms, 2),
                "total_ms": round(self.total_processing_time_ms, 2)
            },
            "throughput": {
                "rps": round(self.requests_per_second, 2),
                "epm": self.evaluations_per_minute
            },
            "resources": {
                "cpu_percent": round(self.cpu_percent, 2),
                "memory_percent": round(self.memory_percent, 2),
                "memory_mb": round(self.memory_mb, 2),
                "gpu_memory_mb": round(self.gpu_memory_mb, 2),
                "gpu_utilization": round(self.gpu_utilization, 2)
            },
            "queue": {
                "depth": self.queue_depth,
                "active_threads": self.active_threads,
                "pending_evaluations": self.pending_evaluations
            },
            "errors": {
                "error_rate": round(self.error_rate, 4),
                "timeout_rate": round(self.timeout_rate, 4)
            }
        }


class PerformanceMonitor:
    """性能监控器"""

    # 性能阈值
    THRESHOLDS = {
        "api_latency_p95_ms": 5000,
        "evaluation_latency_p95_ms": 10000,
        "cpu_percent": 80.0,
        "memory_percent": 85.0,
        "error_rate": 0.05,
        "timeout_rate": 0.02
    }

    def __init__(self, history_size: int = 1000):
        self.history: deque = deque(maxlen=history_size)
        self.current_metrics = PerformanceMetrics()
        self._lock = threading.RLock()
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._interval = 1.0

        # 统计计数器
        self._request_count = 0
        self._request_times: deque = deque(maxlen=100)
        self._evaluation_count = 0
        self._evaluation_times: deque = deque(maxlen=100)
        self._error_count = 0
        self._timeout_count = 0
        self._total_requests = 0

        # 告警回调
        self._alert_callbacks: List[Callable[[str, Dict], None]] = []

        logger.info("[OK] 性能监控器初始化完成")

    def start(self):
        """启动性能监控"""
        if self._running:
            return
        self._running = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("[OK] 性能监控线程已启动")

    def stop(self):
        """停止性能监控"""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2)
        logger.info("[OK] 性能监控线程已停止")

    def _monitor_loop(self):
        """监控主循环"""
        while self._running:
            try:
                metrics = self.collect_metrics()
                with self._lock:
                    self.current_metrics = metrics
                    self.history.append(metrics)

                # 检查阈值并触发告警
                self._check_thresholds(metrics)

                time.sleep(self._interval)
            except Exception as e:
                logger.error(f"性能监控异常: {e}")
                time.sleep(self._interval)

    def collect_metrics(self) -> PerformanceMetrics:
        """收集性能指标"""
        metrics = PerformanceMetrics()
        metrics.timestamp = time.time()

        # 系统资源
        try:
            metrics.cpu_percent = psutil.cpu_percent(interval=0.1)
            mem = psutil.virtual_memory()
            metrics.memory_percent = mem.percent
            metrics.memory_mb = mem.used / (1024 * 1024)
        except Exception as e:
            logger.debug(f"系统资源监控失败: {e}")

        # GPU资源
        try:
            metrics.gpu_memory_mb = self._get_gpu_memory()
            metrics.gpu_utilization = self._get_gpu_utilization()
        except Exception:
            pass

        # 计算延迟统计
        with self._lock:
            if self._request_times:
                metrics.api_latency_ms = sum(self._request_times) / len(self._request_times)
            if self._evaluation_times:
                metrics.evaluation_latency_ms = sum(self._evaluation_times) / len(self._evaluation_times)

            # 计算吞吐量
            now = time.time()
            recent_requests = [t for t in self._request_times if now - t < 1.0]
            metrics.requests_per_second = len(recent_requests)

            recent_evals = [t for t in self._evaluation_times if now - t < 60.0]
            metrics.evaluations_per_minute = len(recent_evals)

            # 错误率
            if self._total_requests > 0:
                metrics.error_rate = self._error_count / self._total_requests
                metrics.timeout_rate = self._timeout_count / self._total_requests

        return metrics

    def _get_gpu_memory(self) -> float:
        """获取GPU显存使用"""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return info.used / (1024 * 1024)
        except Exception:
            return 0.0

    def _get_gpu_utilization(self) -> float:
        """获取GPU利用率"""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return util.gpu
        except Exception:
            return 0.0

    def record_api_call(self, latency_ms: float, success: bool = True,
                        is_timeout: bool = False):
        """记录API调用"""
        with self._lock:
            self._request_times.append(latency_ms)
            self._request_count += 1
            self._total_requests += 1

            if not success:
                self._error_count += 1
            if is_timeout:
                self._timeout_count += 1

    def record_evaluation(self, latency_ms: float):
        """记录评估耗时"""
        with self._lock:
            self._evaluation_times.append(latency_ms)
            self._evaluation_count += 1

    def record_queue_state(self, depth: int = 0, active_threads: int = 0,
                           pending: int = 0):
        """记录队列状态"""
        with self._lock:
            self.current_metrics.queue_depth = depth
            self.current_metrics.active_threads = active_threads
            self.current_metrics.pending_evaluations = pending

    def _check_thresholds(self, metrics: PerformanceMetrics):
        """检查性能阈值"""
        alerts = []

        if metrics.api_latency_ms > self.THRESHOLDS["api_latency_p95_ms"]:
            alerts.append({
                "type": "latency",
                "message": f"API延迟过高: {metrics.api_latency_ms:.0f}ms",
                "value": metrics.api_latency_ms,
                "threshold": self.THRESHOLDS["api_latency_p95_ms"]
            })

        if metrics.cpu_percent > self.THRESHOLDS["cpu_percent"]:
            alerts.append({
                "type": "cpu",
                "message": f"CPU使用率过高: {metrics.cpu_percent:.1f}%",
                "value": metrics.cpu_percent,
                "threshold": self.THRESHOLDS["cpu_percent"]
            })

        if metrics.memory_percent > self.THRESHOLDS["memory_percent"]:
            alerts.append({
                "type": "memory",
                "message": f"内存使用率过高: {metrics.memory_percent:.1f}%",
                "value": metrics.memory_percent,
                "threshold": self.THRESHOLDS["memory_percent"]
            })

        if metrics.error_rate > self.THRESHOLDS["error_rate"]:
            alerts.append({
                "type": "error",
                "message": f"错误率过高: {metrics.error_rate:.2%}",
                "value": metrics.error_rate,
                "threshold": self.THRESHOLDS["error_rate"]
            })

        if metrics.timeout_rate > self.THRESHOLDS["timeout_rate"]:
            alerts.append({
                "type": "timeout",
                "message": f"超时率过高: {metrics.timeout_rate:.2%}",
                "value": metrics.timeout_rate,
                "threshold": self.THRESHOLDS["timeout_rate"]
            })

        for alert in alerts:
            logger.warning(f"[性能告警] {alert['message']}")
            for callback in self._alert_callbacks:
                try:
                    callback(alert["type"], alert)
                except Exception as e:
                    logger.warning(f"告警回调失败: {e}")

    def register_alert_callback(self, callback: Callable[[str, Dict], None]):
        """注册告警回调"""
        self._alert_callbacks.append(callback)

    def get_current_metrics(self) -> PerformanceMetrics:
        """获取当前指标"""
        with self._lock:
            return self.current_metrics

    def get_metrics_history(self, count: int = 100) -> List[PerformanceMetrics]:
        """获取历史指标"""
        with self._lock:
            return list(self.history)[-count:]

    def get_statistics(self, duration_seconds: int = 300) -> Dict:
        """获取统计信息"""
        with self._lock:
            cutoff = time.time() - duration_seconds
            recent = [m for m in self.history if m.timestamp >= cutoff]

            if not recent:
                return {}

            api_latencies = [m.api_latency_ms for m in recent if m.api_latency_ms > 0]
            eval_latencies = [m.evaluation_latency_ms for m in recent if m.evaluation_latency_ms > 0]

            return {
                "period_seconds": duration_seconds,
                "sample_count": len(recent),
                "api_latency": {
                    "avg_ms": round(sum(api_latencies) / len(api_latencies), 2) if api_latencies else 0,
                    "max_ms": round(max(api_latencies), 2) if api_latencies else 0,
                    "min_ms": round(min(api_latencies), 2) if api_latencies else 0
                },
                "evaluation_latency": {
                    "avg_ms": round(sum(eval_latencies) / len(eval_latencies), 2) if eval_latencies else 0,
                    "max_ms": round(max(eval_latencies), 2) if eval_latencies else 0,
                    "min_ms": round(min(eval_latencies), 2) if eval_latencies else 0
                },
                "resource_usage": {
                    "avg_cpu": round(sum(m.cpu_percent for m in recent) / len(recent), 2),
                    "avg_memory": round(sum(m.memory_percent for m in recent) / len(recent), 2)
                },
                "throughput": {
                    "avg_rps": round(sum(m.requests_per_second for m in recent) / len(recent), 2),
                    "avg_epm": round(sum(m.evaluations_per_minute for m in recent) / len(recent), 2)
                }
            }

    def get_bottleneck_analysis(self) -> Dict:
        """获取瓶颈分析"""
        stats = self.get_statistics(duration_seconds=60)

        bottlenecks = []
        recommendations = []

        if stats.get("api_latency", {}).get("avg_ms", 0) > 3000:
            bottlenecks.append("API响应延迟过高")
            recommendations.append("考虑启用异步处理或缓存")

        if stats.get("resource_usage", {}).get("avg_cpu", 0) > 70:
            bottlenecks.append("CPU使用率过高")
            recommendations.append("考虑增加计算资源或优化算法")

        if stats.get("resource_usage", {}).get("avg_memory", 0) > 80:
            bottlenecks.append("内存使用率过高")
            recommendations.append("考虑增加内存或优化内存使用")

        if stats.get("throughput", {}).get("avg_rps", 0) < 1:
            bottlenecks.append("吞吐量过低")
            recommendations.append("考虑优化处理流程或增加并发")

        return {
            "bottlenecks": bottlenecks,
            "recommendations": recommendations,
            "severity": "high" if len(bottlenecks) >= 2 else "medium" if bottlenecks else "low"
        }
