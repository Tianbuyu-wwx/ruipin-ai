"""
锐聘AI - 平滑过渡模块

实现降级和恢复的平滑过渡，避免系统性能骤降
"""

import time
import threading
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from ..logger import logger


class TransitionType(Enum):
    DEGRADE = "degrade"
    RECOVER = "recover"


@dataclass
class TransitionRecord:
    """过渡记录"""
    timestamp: float
    from_level: int
    to_level: int
    transition_type: TransitionType
    duration_ms: float
    success: bool
    reason: str


class SmoothTransition:
    """平滑过渡控制器"""

    def __init__(self):
        self.transition_history: List[TransitionRecord] = []
        self._lock = threading.RLock()
        self._transition_callbacks: Dict[int, List[Callable]] = {}
        self._step_interval = 5.0
        self._max_transition_time = 30.0

        logger.info("[OK] 平滑过渡控制器初始化完成")

    def register_callback(self, level: int, callback: Callable[[int, int], None]):
        if level not in self._transition_callbacks:
            self._transition_callbacks[level] = []
        self._transition_callbacks[level].append(callback)
        logger.info(f"注册降级回调: level={level}")

    def execute_degradation(self, from_level: int, to_level: int,
                            apply_config_fn: Callable[[int], bool],
                            reason: str = "") -> bool:
        if from_level >= to_level:
            logger.warning(f"无需降级: from={from_level}, to={to_level}")
            return True

        logger.warning(f"开始平滑降级: Level {from_level} -> Level {to_level} | 原因: {reason}")
        start_time = time.time()
        current = from_level

        try:
            while current < to_level:
                next_level = current + 1
                logger.info(f"执行降级步骤: Level {current} -> Level {next_level}")

                success = apply_config_fn(next_level)
                if not success:
                    logger.error(f"降级步骤失败: Level {current} -> Level {next_level}")
                    self._record_transition(from_level, current, TransitionType.DEGRADE,
                                            start_time, False, reason)
                    return False

                self._trigger_callbacks(current, next_level)
                current = next_level

                if current < to_level:
                    elapsed = time.time() - start_time
                    if elapsed >= self._max_transition_time:
                        logger.warning(f"降级超时，强制完成: 当前Level={current}")
                        break
                    logger.info(f"等待 {self._step_interval}s 后执行下一步降级...")
                    time.sleep(self._step_interval)

            self._record_transition(from_level, current, TransitionType.DEGRADE,
                                    start_time, True, reason)
            logger.warning(f"降级完成: Level {from_level} -> Level {current}")
            return True

        except Exception as e:
            logger.error(f"降级过程异常: {e}", exc_info=True)
            self._record_transition(from_level, current, TransitionType.DEGRADE,
                                    start_time, False, f"{reason}; 异常: {str(e)}")
            return False

    def execute_recovery(self, from_level: int, to_level: int,
                         apply_config_fn: Callable[[int], bool],
                         reason: str = "") -> bool:
        if from_level <= to_level:
            logger.warning(f"无需恢复: from={from_level}, to={to_level}")
            return True

        logger.info(f"开始平滑恢复: Level {from_level} -> Level {to_level} | 原因: {reason}")
        start_time = time.time()
        current = from_level

        try:
            while current > to_level:
                next_level = current - 1
                logger.info(f"执行恢复步骤: Level {current} -> Level {next_level}")

                success = apply_config_fn(next_level)
                if not success:
                    logger.error(f"恢复步骤失败: Level {current} -> Level {next_level}")
                    self._record_transition(from_level, current, TransitionType.RECOVER,
                                            start_time, False, reason)
                    return False

                self._trigger_callbacks(current, next_level)
                current = next_level

                if current > to_level:
                    elapsed = time.time() - start_time
                    if elapsed >= self._max_transition_time:
                        logger.warning(f"恢复超时，强制完成: 当前Level={current}")
                        break
                    logger.info(f"等待 {self._step_interval}s 后执行下一步恢复...")
                    time.sleep(self._step_interval)

            self._record_transition(from_level, current, TransitionType.RECOVER,
                                    start_time, True, reason)
            logger.info(f"恢复完成: Level {from_level} -> Level {current}")
            return True

        except Exception as e:
            logger.error(f"恢复过程异常: {e}", exc_info=True)
            self._record_transition(from_level, current, TransitionType.RECOVER,
                                    start_time, False, f"{reason}; 异常: {str(e)}")
            return False

    def _trigger_callbacks(self, from_level: int, to_level: int):
        callbacks = self._transition_callbacks.get(to_level, [])
        for callback in callbacks:
            try:
                callback(from_level, to_level)
            except Exception as e:
                logger.warning(f"过渡回调执行失败: {e}")

    def _record_transition(self, from_level: int, to_level: int,
                           transition_type: TransitionType,
                           start_time: float, success: bool, reason: str):
        record = TransitionRecord(
            timestamp=time.time(),
            from_level=from_level,
            to_level=to_level,
            transition_type=transition_type,
            duration_ms=(time.time() - start_time) * 1000,
            success=success,
            reason=reason
        )
        with self._lock:
            self.transition_history.append(record)

    def get_transition_history(self, count: int = 10) -> List[Dict]:
        with self._lock:
            records = list(self.transition_history)[-count:]
            return [
                {
                    "timestamp": datetime.fromtimestamp(r.timestamp).isoformat(),
                    "from_level": r.from_level,
                    "to_level": r.to_level,
                    "type": r.transition_type.value,
                    "duration_ms": round(r.duration_ms, 2),
                    "success": r.success,
                    "reason": r.reason,
                }
                for r in records
            ]

    def get_last_transition_time(self) -> Optional[float]:
        with self._lock:
            if not self.transition_history:
                return None
            return self.transition_history[-1].timestamp

    def can_transition(self, cooldown_seconds: float = 300.0) -> bool:
        last_time = self.get_last_transition_time()
        if last_time is None:
            return True
        elapsed = time.time() - last_time
        return elapsed > cooldown_seconds
