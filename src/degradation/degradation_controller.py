"""
锐聘AI - 智能降级决策中心

单例模式实现，负责监控系统状态并执行平滑降级策略
"""

import time
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum

from ..logger import logger
from .resource_monitor import ResourceMonitor, ResourceMetrics
from .health_checker import HealthChecker, HealthStatus
from .smooth_transition import SmoothTransition


class DegradationLevel(Enum):
    """降级级别枚举"""
    LEVEL_1 = 1
    LEVEL_2 = 2
    LEVEL_3 = 3
    LEVEL_4 = 4
    LEVEL_5 = 5


@dataclass
class LevelConfig:
    """降级级别配置"""
    level: int
    name: str
    description: str
    triggers: Dict[str, Any]
    actions: Dict[str, Any]
    evaluators: Dict[str, str]


class DegradationController:
    """降级决策中心 - 单例模式"""

    _instance: Optional["DegradationController"] = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        self.current_level = 1
        self.target_level = 1
        self._level_configs: Dict[int, LevelConfig] = {}
        self._init_level_configs()

        self.resource_monitor = ResourceMonitor()
        self.health_checker = HealthChecker()
        self.smooth_transition = SmoothTransition()

        self._running = False
        self._decision_thread: Optional[threading.Thread] = None
        self._decision_interval = 5.0
        self._recovery_cooldown = 300.0
        self._stability_check_duration = 60.0

        self._status_change_callbacks: List[Callable[[int, int], None]] = []
        self._degradation_history: List[Dict] = []

        logger.info("=" * 60)
        logger.info("[OK] 降级决策中心初始化完成")
        logger.info(f"  - 当前级别: Level {self.current_level}")
        logger.info(f"  - 监控间隔: {self._decision_interval}s")
        logger.info(f"  - 恢复冷却: {self._recovery_cooldown}s")
        logger.info("=" * 60)

    def _init_level_configs(self):
        """初始化五级降级配置"""
        self._level_configs = {
            1: LevelConfig(
                level=1,
                name="完全服务",
                description="所有功能全开，使用最高质量模型",
                triggers={},
                actions={
                    "streaming": True,
                    "cache": True,
                    "async_preload": True,
                },
                evaluators={
                    "primary": "deepseek_api",
                    "multimodal": "qwen3vl_local",
                    "voice": "full_analysis",
                    "whisper": "api_mode",
                }
            ),
            2: LevelConfig(
                level=2,
                name="性能降级",
                description="API降级为异步模式，启用缓存优先",
                triggers={
                    "api_p95_latency_ms": 3000,
                    "api_error_rate": 0.02,
                    "consecutive_timeouts": 2,
                    "cpu_usage": 0.70,
                },
                actions={
                    "switch_to_async_api": True,
                    "increase_cache_ttl": 3600,
                    "enable_batch_processing": True,
                    "reduce_max_tokens_ratio": 0.8,
                },
                evaluators={
                    "primary": "deepseek_async",
                    "multimodal": "qwen3vl_local",
                    "voice": "full_analysis",
                    "whisper": "api_mode",
                }
            ),
            3: LevelConfig(
                level=3,
                name="模型降级",
                description="切换到本地轻量级模型，关闭非核心功能",
                triggers={
                    "api_availability": False,
                    "gpu_memory": 0.90,
                    "local_model_load_failed": True,
                    "api_error_rate": 0.20,
                },
                actions={
                    "switch_to_local_model": "qwen3vl_4bit",
                    "disable_streaming": True,
                    "disable_voice_analysis": False,
                    "reduce_batch_size_ratio": 0.5,
                    "enable_request_queue": True,
                },
                evaluators={
                    "primary": "qwen3vl_local_4bit",
                    "multimodal": "qwen3vl_local_4bit",
                    "voice": "basic_analysis",
                    "whisper": "local_model",
                }
            ),
            4: LevelConfig(
                level=4,
                name="功能降级",
                description="仅保留核心评估功能，使用规则引擎",
                triggers={
                    "all_models_unavailable": True,
                    "cpu_usage": 0.95,
                    "memory_usage": 0.95,
                },
                actions={
                    "disable_multimodal": True,
                    "disable_voice_analysis": True,
                    "enable_rule_based": True,
                    "use_keyword_matching": True,
                    "reduce_question_complexity": True,
                },
                evaluators={
                    "primary": "rule_based",
                    "multimodal": "disabled",
                    "voice": "disabled",
                    "whisper": "disabled",
                }
            ),
            5: LevelConfig(
                level=5,
                name="基础服务",
                description="仅保留面试流程，评估简化为通过/不通过",
                triggers={
                    "system_critical": True,
                    "db_connection_failed": True,
                },
                actions={
                    "disable_all_ai": True,
                    "use_static_questions": True,
                    "manual_evaluation_mode": True,
                    "notify_admin": True,
                },
                evaluators={
                    "primary": "pass_fail_rule",
                    "multimodal": "disabled",
                    "voice": "disabled",
                    "whisper": "disabled",
                }
            ),
        }

    def start(self):
        """启动降级决策循环"""
        if self._running:
            return
        self._running = True

        self.resource_monitor.start()
        self._decision_thread = threading.Thread(target=self._decision_loop, daemon=True)
        self._decision_thread.start()

        logger.info("[OK] 降级决策循环已启动")

    def stop(self):
        """停止降级决策循环"""
        self._running = False
        self.resource_monitor.stop()
        if self._decision_thread:
            self._decision_thread.join(timeout=5)
        logger.info("[OK] 降级决策循环已停止")

    def _decision_loop(self):
        """决策主循环"""
        while self._running:
            try:
                self.evaluate_and_act()
                time.sleep(self._decision_interval)
            except Exception as e:
                logger.error(f"决策循环异常: {e}", exc_info=True)
                time.sleep(self._decision_interval)

    def evaluate_and_act(self):
        """评估系统状态并执行降级/恢复"""
        metrics = self.resource_monitor.get_current_metrics()
        health_status = self.health_checker.get_overall_status()

        new_level = self._calculate_target_level(metrics, health_status)

        if new_level > self.current_level:
            reason = self._build_degradation_reason(new_level, metrics, health_status)
            logger.warning(f"触发降级: Level {self.current_level} -> Level {new_level} | {reason}")
            success = self.smooth_transition.execute_degradation(
                from_level=self.current_level,
                to_level=new_level,
                apply_config_fn=self._apply_level_config,
                reason=reason
            )
            if success:
                self._set_level(new_level)

        elif new_level < self.current_level:
            if self._can_recover():
                reason = self._build_recovery_reason(new_level, metrics, health_status)
                logger.info(f"触发恢复: Level {self.current_level} -> Level {new_level} | {reason}")
                success = self.smooth_transition.execute_recovery(
                    from_level=self.current_level,
                    to_level=new_level,
                    apply_config_fn=self._apply_level_config,
                    reason=reason
                )
                if success:
                    self._set_level(new_level)
            else:
                logger.debug(f"满足恢复条件，但冷却期未过，保持 Level {self.current_level}")

    def _calculate_target_level(self, metrics: ResourceMetrics,
                                 health_status: HealthStatus) -> int:
        """计算目标降级级别"""
        if health_status == HealthStatus.UNHEALTHY:
            if not metrics.database_connected:
                return 5
            if metrics.cpu_usage > 0.95 or metrics.memory_usage > 0.95:
                return 4

        if metrics.api_error_rate > 0.20 or metrics.api_consecutive_failures >= 5:
            return 3

        if (metrics.api_response_time > 3000 or
            metrics.api_error_rate > 0.02 or
            metrics.cpu_usage > 0.70):
            return 2

        if health_status in (HealthStatus.HEALTHY, HealthStatus.DEGRADED):
            return 1

        return self.current_level

    def _can_recover(self) -> bool:
        """检查是否可以恢复"""
        if not self.smooth_transition.can_transition(self._recovery_cooldown):
            return False
        return self._check_metrics_stable(self._stability_check_duration)

    def _check_metrics_stable(self, duration_seconds: float) -> bool:
        """检查指标是否持续稳定"""
        history = self.resource_monitor.get_metrics_history(count=20)
        if len(history) < 3:
            return True

        cutoff = time.time() - duration_seconds
        recent = [m for m in history if m.timestamp >= cutoff]
        if len(recent) < 3:
            return False

        api_ok = all(m.api_error_rate < 0.01 and m.api_response_time < 2000
                     for m in recent)
        system_ok = all(m.cpu_usage < 0.60 and m.memory_usage < 0.70
                        for m in recent)

        return api_ok and system_ok

    def _apply_level_config(self, level: int) -> bool:
        """应用指定级别的配置"""
        config = self._level_configs.get(level)
        if not config:
            logger.error(f"未找到级别配置: Level {level}")
            return False

        logger.info(f"应用配置: Level {level} - {config.name}")
        logger.info(f"  评估器配置: {config.evaluators}")
        logger.info(f"  动作配置: {config.actions}")

        return True

    def _set_level(self, level: int):
        """设置当前级别并触发回调"""
        old_level = self.current_level
        self.current_level = level

        self._degradation_history.append({
            "timestamp": time.time(),
            "from_level": old_level,
            "to_level": level,
        })

        for callback in self._status_change_callbacks:
            try:
                callback(old_level, level)
            except Exception as e:
                logger.warning(f"状态变更回调失败: {e}")

        logger.warning(f"当前降级级别: Level {level} ({self._level_configs[level].name})")

    def _build_degradation_reason(self, target_level: int,
                                   metrics: ResourceMetrics,
                                   health_status: HealthStatus) -> str:
        """构建降级原因说明"""
        reasons = []
        config = self._level_configs.get(target_level)
        if not config:
            return "未知原因"

        triggers = config.triggers
        if "api_p95_latency_ms" in triggers and metrics.api_response_time > triggers["api_p95_latency_ms"]:
            reasons.append(f"API响应时间高({metrics.api_response_time:.0f}ms)")
        if "api_error_rate" in triggers and metrics.api_error_rate > triggers["api_error_rate"]:
            reasons.append(f"API错误率高({metrics.api_error_rate:.2%})")
        if "cpu_usage" in triggers and metrics.cpu_usage > triggers["cpu_usage"]:
            reasons.append(f"CPU使用率高({metrics.cpu_usage:.1%})")
        if "memory_usage" in triggers and metrics.memory_usage > triggers["memory_usage"]:
            reasons.append(f"内存使用率高({metrics.memory_usage:.1%})")
        if health_status == HealthStatus.UNHEALTHY:
            reasons.append("服务健康状态异常")

        return "; ".join(reasons) if reasons else "触发条件满足"

    def _build_recovery_reason(self, target_level: int,
                                metrics: ResourceMetrics,
                                health_status: HealthStatus) -> str:
        """构建恢复原因说明"""
        reasons = []
        if metrics.api_error_rate < 0.01:
            reasons.append("API错误率正常")
        if metrics.api_response_time < 2000:
            reasons.append("API响应时间正常")
        if metrics.cpu_usage < 0.60:
            reasons.append("CPU使用率正常")
        if health_status == HealthStatus.HEALTHY:
            reasons.append("服务健康状态良好")

        return "; ".join(reasons) if reasons else "指标恢复正常"

    def get_current_level(self) -> int:
        return self.current_level

    def get_level_config(self, level: Optional[int] = None) -> Optional[LevelConfig]:
        level = level or self.current_level
        return self._level_configs.get(level)

    def get_current_evaluators(self) -> Dict[str, str]:
        config = self.get_level_config()
        return config.evaluators if config else {}

    def register_status_change_callback(self, callback: Callable[[int, int], None]):
        self._status_change_callbacks.append(callback)
        logger.info("注册状态变更回调")

    def get_status(self) -> Dict[str, Any]:
        config = self.get_level_config()
        return {
            "current_level": self.current_level,
            "level_name": config.name if config else "未知",
            "level_description": config.description if config else "",
            "evaluators": config.evaluators if config else {},
            "metrics": self.resource_monitor.get_current_metrics().to_dict(),
            "health": self.health_checker.get_all_health(),
            "transition_history": self.smooth_transition.get_transition_history(count=5),
        }

    def force_level(self, level: int, reason: str = "manual") -> bool:
        """强制设置降级级别（用于管理操作）"""
        if level < 1 or level > 5:
            logger.error(f"无效的降级级别: {level}")
            return False

        if level > self.current_level:
            success = self.smooth_transition.execute_degradation(
                from_level=self.current_level,
                to_level=level,
                apply_config_fn=self._apply_level_config,
                reason=f"[强制] {reason}"
            )
        else:
            success = self.smooth_transition.execute_recovery(
                from_level=self.current_level,
                to_level=level,
                apply_config_fn=self._apply_level_config,
                reason=f"[强制] {reason}"
            )

        if success:
            self._set_level(level)
        return success

    def is_feature_available(self, feature: str) -> bool:
        """检查指定功能在当前级别是否可用"""
        config = self.get_level_config()
        if not config:
            return False

        feature_map = {
            "streaming": 2,
            "multimodal": 3,
            "voice_analysis": 3,
            "whisper": 3,
            "rule_based": 4,
        }

        max_level = feature_map.get(feature, 5)
        return self.current_level <= max_level

    @classmethod
    def reset_instance(cls):
        """重置单例实例（仅用于测试）"""
        if cls._instance:
            cls._instance.stop()
        cls._instance = None
