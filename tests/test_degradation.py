"""
锐聘AI - 降级策略单元测试

测试降级决策中心、资源监控、健康检查和平滑过渡模块
"""

import time
import threading
import pytest
from unittest.mock import Mock, patch, MagicMock

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from src.degradation import (
    DegradationController,
    ResourceMonitor,
    ResourceMetrics,
    HealthChecker,
    HealthStatus,
    SmoothTransition,
)
from src.degradation.health_checker import ComponentHealth


class TestResourceMonitor:
    """资源监控模块测试"""

    def test_metrics_initialization(self):
        metrics = ResourceMetrics()
        assert metrics.timestamp > 0
        assert metrics.cpu_usage == 0.0
        assert metrics.api_availability is True
        assert metrics.api_consecutive_failures == 0

    def test_metrics_to_dict(self):
        metrics = ResourceMetrics()
        metrics.cpu_usage = 0.5
        metrics.api_response_time = 1500.0
        metrics.api_error_rate = 0.02

        d = metrics.to_dict()
        assert d["system"]["cpu_usage"] == 0.5
        assert d["api"]["response_time_ms"] == 1500.0
        assert d["api"]["error_rate"] == 0.02
        assert d["api"]["availability"] is True

    def test_monitor_start_stop(self):
        monitor = ResourceMonitor()
        monitor.start()
        assert monitor._running is True
        time.sleep(0.1)
        monitor.stop()
        assert monitor._running is False

    def test_record_api_call(self):
        monitor = ResourceMonitor()
        monitor.record_api_call(1000.0, True)
        monitor.record_api_call(2000.0, False)

        metrics = monitor.get_current_metrics()
        assert metrics.api_consecutive_failures == 1

        monitor.record_api_call(1500.0, True)
        metrics = monitor.get_current_metrics()
        assert metrics.api_consecutive_failures == 0

    def test_update_service_status(self):
        monitor = ResourceMonitor()
        monitor.update_service_status(model_loaded=True, database_connected=True)

        metrics = monitor.get_current_metrics()
        assert metrics.model_loaded is True
        assert metrics.database_connected is True
        assert metrics.whisper_available is False

    def test_get_average_metrics(self):
        monitor = ResourceMonitor()
        monitor.record_api_call(1000.0, True)
        monitor.record_api_call(2000.0, True)

        avg = monitor.get_average_metrics(duration_seconds=60)
        assert avg is not None
        assert avg.api_response_time == 1500.0


class TestHealthChecker:
    """健康检查模块测试"""

    def test_register_checker(self):
        checker = HealthChecker()
        mock_fn = Mock(return_value=True)
        checker.register_checker("deepseek_api", mock_fn, {"timeout": 5})

        assert "deepseek_api" in checker.components
        assert checker.components["deepseek_api"].metadata == {"timeout": 5}

    def test_check_healthy(self):
        checker = HealthChecker()
        mock_fn = Mock(return_value=True)
        checker.register_checker("api", mock_fn)

        result = checker.check("api")
        assert result.status == HealthStatus.HEALTHY
        assert result.consecutive_successes == 1

        result = checker.check("api")
        assert result.status == HealthStatus.HEALTHY
        assert result.consecutive_successes == 2

    def test_check_unhealthy(self):
        checker = HealthChecker()
        mock_fn = Mock(return_value=False)
        checker.register_checker("api", mock_fn)

        checker.check("api")
        checker.check("api")
        result = checker.check("api")

        assert result.status == HealthStatus.UNHEALTHY
        assert result.consecutive_failures == 3

    def test_check_exception(self):
        checker = HealthChecker()
        mock_fn = Mock(side_effect=Exception("Connection refused"))
        checker.register_checker("api", mock_fn)

        checker.check("api")
        checker.check("api")
        result = checker.check("api")

        assert result.status == HealthStatus.UNHEALTHY
        assert "Connection refused" in result.error_message

    def test_overall_status(self):
        checker = HealthChecker()
        checker.register_checker("api", Mock(return_value=True))
        checker.register_checker("db", Mock(return_value=True))

        checker.check_all()
        assert checker.get_overall_status() == HealthStatus.HEALTHY

        checker.register_checker("cache", Mock(return_value=False))
        checker.check("cache")
        checker.check("cache")
        checker.check("cache")

        assert checker.get_overall_status() == HealthStatus.UNHEALTHY

    def test_unhealthy_components(self):
        checker = HealthChecker()
        checker.register_checker("api", Mock(return_value=True))
        checker.register_checker("db", Mock(return_value=False))
        checker.check("db")
        checker.check("db")
        checker.check("db")

        unhealthy = checker.get_unhealthy_components()
        assert "db" in unhealthy
        assert "api" not in unhealthy


class TestSmoothTransition:
    """平滑过渡模块测试"""

    def test_degradation_step_by_step(self):
        st = SmoothTransition()
        st._step_interval = 0.01

        apply_fn = Mock(return_value=True)
        success = st.execute_degradation(1, 3, apply_fn, reason="test")

        assert success is True
        assert apply_fn.call_count == 2
        apply_fn.assert_any_call(2)
        apply_fn.assert_any_call(3)

    def test_recovery_step_by_step(self):
        st = SmoothTransition()
        st._step_interval = 0.01

        apply_fn = Mock(return_value=True)
        success = st.execute_recovery(3, 1, apply_fn, reason="test")

        assert success is True
        assert apply_fn.call_count == 2
        apply_fn.assert_any_call(2)
        apply_fn.assert_any_call(1)

    def test_degradation_failure(self):
        st = SmoothTransition()
        apply_fn = Mock(side_effect=[True, False])

        success = st.execute_degradation(1, 3, apply_fn, reason="test")
        assert success is False

    def test_transition_history(self):
        st = SmoothTransition()
        st._step_interval = 0.01

        apply_fn = Mock(return_value=True)
        st.execute_degradation(1, 2, apply_fn, reason="test_degrade")
        st.execute_recovery(2, 1, apply_fn, reason="test_recover")

        history = st.get_transition_history(count=10)
        assert len(history) == 2
        assert history[0]["type"] == "degrade"
        assert history[1]["type"] == "recover"

    def test_cooldown(self):
        st = SmoothTransition()
        st._step_interval = 0.01

        apply_fn = Mock(return_value=True)
        st.execute_degradation(1, 2, apply_fn, reason="test")

        assert st.can_transition(cooldown_seconds=1.0) is False
        time.sleep(0.01)
        assert st.can_transition(cooldown_seconds=0.005) is True

    def test_callback(self):
        st = SmoothTransition()
        st._step_interval = 0.01

        callback = Mock()
        st.register_callback(2, callback)

        apply_fn = Mock(return_value=True)
        st.execute_degradation(1, 2, apply_fn, reason="test")

        callback.assert_called_once_with(1, 2)


class TestDegradationController:
    """降级决策中心测试"""

    def setup_method(self):
        DegradationController.reset_instance()

    def teardown_method(self):
        DegradationController.reset_instance()

    def test_singleton(self):
        c1 = DegradationController()
        c2 = DegradationController()
        assert c1 is c2

    def test_initial_level(self):
        controller = DegradationController()
        assert controller.get_current_level() == 1

    def test_level_configs(self):
        controller = DegradationController()
        config = controller.get_level_config(1)
        assert config.name == "完全服务"
        assert config.evaluators["primary"] == "deepseek_api"

        config = controller.get_level_config(3)
        assert config.name == "模型降级"
        assert config.evaluators["primary"] == "qwen3vl_local_4bit"

    def test_calculate_target_level_healthy(self):
        controller = DegradationController()
        metrics = ResourceMetrics()
        metrics.api_response_time = 1000.0
        metrics.api_error_rate = 0.0
        metrics.cpu_usage = 0.3

        level = controller._calculate_target_level(metrics, HealthStatus.HEALTHY)
        assert level == 1

    def test_calculate_target_level_performance_degrade(self):
        controller = DegradationController()
        metrics = ResourceMetrics()
        metrics.api_response_time = 4000.0
        metrics.api_error_rate = 0.03
        metrics.cpu_usage = 0.5

        level = controller._calculate_target_level(metrics, HealthStatus.DEGRADED)
        assert level == 2

    def test_calculate_target_level_model_degrade(self):
        controller = DegradationController()
        metrics = ResourceMetrics()
        metrics.api_error_rate = 0.25
        metrics.api_consecutive_failures = 5

        level = controller._calculate_target_level(metrics, HealthStatus.DEGRADED)
        assert level == 3

    def test_calculate_target_level_critical(self):
        controller = DegradationController()
        metrics = ResourceMetrics()
        metrics.cpu_usage = 0.96
        metrics.memory_usage = 0.97
        metrics.database_connected = True

        level = controller._calculate_target_level(metrics, HealthStatus.UNHEALTHY)
        assert level == 4

    def test_calculate_target_level_db_failed(self):
        controller = DegradationController()
        metrics = ResourceMetrics()
        metrics.database_connected = False
        metrics.cpu_usage = 0.5

        level = controller._calculate_target_level(metrics, HealthStatus.UNHEALTHY)
        assert level == 5

    def test_feature_availability(self):
        controller = DegradationController()
        assert controller.is_feature_available("streaming") is True
        assert controller.is_feature_available("multimodal") is True

        controller.current_level = 2
        assert controller.is_feature_available("streaming") is True
        assert controller.is_feature_available("multimodal") is True

        controller.current_level = 3
        assert controller.is_feature_available("streaming") is False
        assert controller.is_feature_available("multimodal") is True
        assert controller.is_feature_available("voice_analysis") is True

        controller.current_level = 4
        assert controller.is_feature_available("multimodal") is False
        assert controller.is_feature_available("rule_based") is True

    def test_force_level(self):
        controller = DegradationController()
        controller.smooth_transition._step_interval = 0.01

        success = controller.force_level(3, reason="test")
        assert success is True
        assert controller.get_current_level() == 3

        success = controller.force_level(1, reason="test")
        assert success is True
        assert controller.get_current_level() == 1

    def test_status_change_callback(self):
        controller = DegradationController()
        controller.smooth_transition._step_interval = 0.01

        callback = Mock()
        controller.register_status_change_callback(callback)

        controller.force_level(2, reason="test")
        callback.assert_called_once_with(1, 2)

    def test_get_status(self):
        controller = DegradationController()
        status = controller.get_status()

        assert status["current_level"] == 1
        assert status["level_name"] == "完全服务"
        assert "metrics" in status
        assert "health" in status
        assert "evaluators" in status

    def test_apply_level_config(self):
        controller = DegradationController()
        assert controller._apply_level_config(1) is True
        assert controller._apply_level_config(3) is True
        assert controller._apply_level_config(99) is False

    def test_build_degradation_reason(self):
        controller = DegradationController()
        metrics = ResourceMetrics()
        metrics.api_response_time = 4000.0
        metrics.api_error_rate = 0.03
        metrics.cpu_usage = 0.75

        reason = controller._build_degradation_reason(2, metrics, HealthStatus.DEGRADED)
        assert "API响应时间高" in reason
        assert "API错误率高" in reason
        assert "CPU使用率高" in reason

    def test_build_recovery_reason(self):
        controller = DegradationController()
        metrics = ResourceMetrics()
        metrics.api_error_rate = 0.005
        metrics.api_response_time = 1500.0
        metrics.cpu_usage = 0.5

        reason = controller._build_recovery_reason(1, metrics, HealthStatus.HEALTHY)
        assert "API错误率正常" in reason
        assert "API响应时间正常" in reason
        assert "CPU使用率正常" in reason

    def test_can_recover_with_cooldown(self):
        controller = DegradationController()
        controller._recovery_cooldown = 0.001
        controller.smooth_transition.transition_history.append(
            MagicMock(timestamp=time.time() - 0.1)
        )

        with patch.object(controller, '_check_metrics_stable', return_value=True):
            assert controller._can_recover() is True

    def test_check_metrics_stable(self):
        controller = DegradationController()
        monitor = controller.resource_monitor

        metrics = ResourceMetrics()
        metrics.timestamp = time.time()
        metrics.api_error_rate = 0.005
        metrics.api_response_time = 1500.0
        metrics.cpu_usage = 0.5
        metrics.memory_usage = 0.6
        monitor.history.append(metrics)
        monitor.history.append(metrics)
        monitor.history.append(metrics)

        assert controller._check_metrics_stable(60.0) is True

        metrics2 = ResourceMetrics()
        metrics2.timestamp = time.time()
        metrics2.api_error_rate = 0.05
        metrics2.api_response_time = 1500.0
        metrics2.cpu_usage = 0.5
        metrics2.memory_usage = 0.6
        monitor.history.append(metrics2)

        assert controller._check_metrics_stable(60.0) is False


class TestIntegration:
    """集成测试"""

    def setup_method(self):
        DegradationController.reset_instance()

    def teardown_method(self):
        DegradationController.reset_instance()

    def test_full_degradation_flow(self):
        controller = DegradationController()
        controller.smooth_transition._step_interval = 0.001
        controller._recovery_cooldown = 0.001
        controller._decision_interval = 0.001

        monitor = controller.resource_monitor
        monitor.update_service_status(model_loaded=True, whisper_available=True)

        health = controller.health_checker
        health.register_checker("api", Mock(return_value=True))
        health.register_checker("db", Mock(return_value=True))
        health.check_all()

        assert controller.get_current_level() == 1

        metrics = ResourceMetrics()
        metrics.timestamp = time.time()
        metrics.api_response_time = 4000.0
        metrics.api_error_rate = 0.03
        metrics.cpu_usage = 0.75
        monitor.current_metrics = metrics
        monitor.history.append(metrics)
        monitor.history.append(metrics)
        monitor.history.append(metrics)

        controller.evaluate_and_act()
        assert controller.get_current_level() == 2

        metrics2 = ResourceMetrics()
        metrics2.timestamp = time.time()
        metrics2.api_error_rate = 0.25
        metrics2.api_consecutive_failures = 5
        metrics2.cpu_usage = 0.5
        monitor.current_metrics = metrics2
        monitor.history = [metrics2, metrics2, metrics2]

        controller.evaluate_and_act()
        assert controller.get_current_level() == 3

    def test_monitor_with_custom_collector(self):
        monitor = ResourceMonitor()
        custom_collector = Mock(return_value={"queue_depth": 10, "active_sessions": 5})
        monitor.register_custom_collector(custom_collector)

        metrics = monitor.collect_metrics()
        assert metrics.queue_depth == 10
        assert metrics.active_sessions == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
