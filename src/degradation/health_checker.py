"""
锐聘AI - 健康检查模块

检查各服务组件的健康状态
"""

import time
import asyncio
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum

from ..logger import logger


class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    """组件健康状态"""
    name: str
    status: HealthStatus = HealthStatus.UNKNOWN
    last_check: float = field(default_factory=time.time)
    response_time_ms: float = 0.0
    error_message: Optional[str] = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "status": self.status.value,
            "last_check": self.last_check,
            "response_time_ms": round(self.response_time_ms, 2),
            "error_message": self.error_message,
            "consecutive_failures": self.consecutive_failures,
            "consecutive_successes": self.consecutive_successes,
            "metadata": self.metadata,
        }


class HealthChecker:
    """健康检查器 - 检查各服务组件状态"""

    def __init__(self):
        self.components: Dict[str, ComponentHealth] = {}
        self._checkers: Dict[str, Callable[[], bool]] = {}
        self._async_checkers: Dict[str, Callable[[], asyncio.Future]] = {}
        self._last_check_time: float = 0
        self._check_interval = 10.0

        logger.info("[OK] 健康检查器初始化完成")

    def register_checker(self, name: str, checker: Callable[[], bool],
                         metadata: Optional[Dict] = None):
        self._checkers[name] = checker
        self.components[name] = ComponentHealth(
            name=name,
            metadata=metadata or {}
        )
        logger.info(f"注册健康检查器: {name}")

    def register_async_checker(self, name: str,
                                checker: Callable[[], asyncio.Future],
                                metadata: Optional[Dict] = None):
        self._async_checkers[name] = checker
        self.components[name] = ComponentHealth(
            name=name,
            metadata=metadata or {}
        )
        logger.info(f"注册异步健康检查器: {name}")

    def check(self, name: str) -> ComponentHealth:
        if name not in self.components:
            return ComponentHealth(name=name, status=HealthStatus.UNKNOWN)

        component = self.components[name]

        if name in self._checkers:
            start = time.time()
            try:
                result = self._checkers[name]()
                component.response_time_ms = (time.time() - start) * 1000
                component.last_check = time.time()

                if result:
                    component.consecutive_successes += 1
                    component.consecutive_failures = 0
                    component.error_message = None
                    component.status = HealthStatus.HEALTHY
                else:
                    component.consecutive_failures += 1
                    component.consecutive_successes = 0
                    component.error_message = "健康检查返回False"

                    if component.consecutive_failures >= 3:
                        component.status = HealthStatus.UNHEALTHY
                    else:
                        component.status = HealthStatus.DEGRADED

            except Exception as e:
                component.response_time_ms = (time.time() - start) * 1000
                component.last_check = time.time()
                component.consecutive_failures += 1
                component.consecutive_successes = 0
                component.error_message = str(e)

                if component.consecutive_failures >= 3:
                    component.status = HealthStatus.UNHEALTHY
                else:
                    component.status = HealthStatus.DEGRADED

        return component

    def check_all(self) -> Dict[str, ComponentHealth]:
        results = {}
        for name in self.components:
            results[name] = self.check(name)
        self._last_check_time = time.time()
        return results

    def get_overall_status(self) -> HealthStatus:
        if not self.components:
            return HealthStatus.UNKNOWN

        statuses = [c.status for c in self.components.values()]

        if any(s == HealthStatus.UNHEALTHY for s in statuses):
            return HealthStatus.UNHEALTHY
        if any(s == HealthStatus.DEGRADED for s in statuses):
            return HealthStatus.DEGRADED
        if all(s == HealthStatus.HEALTHY for s in statuses):
            return HealthStatus.HEALTHY

        return HealthStatus.UNKNOWN

    def is_healthy(self, name: str) -> bool:
        if name not in self.components:
            return False
        return self.components[name].status == HealthStatus.HEALTHY

    def is_unhealthy(self, name: str) -> bool:
        if name not in self.components:
            return True
        return self.components[name].status == HealthStatus.UNHEALTHY

    def get_component_health(self, name: str) -> Optional[ComponentHealth]:
        return self.components.get(name)

    def get_all_health(self) -> Dict[str, Dict]:
        return {name: comp.to_dict() for name, comp in self.components.items()}

    def get_unhealthy_components(self) -> List[str]:
        return [
            name for name, comp in self.components.items()
            if comp.status in (HealthStatus.UNHEALTHY, HealthStatus.DEGRADED)
        ]
