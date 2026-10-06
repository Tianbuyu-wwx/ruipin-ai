"""
锐聘AI - InterviewEngine 降级策略集成测试

测试面试引擎在降级时的评估器自动切换功能
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import pytest
from unittest.mock import Mock, patch, MagicMock

from src.degradation import DegradationController
from src.interview_engine import InterviewEngine


class TestInterviewEngineDegradation:
    """InterviewEngine 降级集成测试"""

    def setup_method(self):
        DegradationController.reset_instance()

    def teardown_method(self):
        DegradationController.reset_instance()

    def test_engine_initializes_with_degradation(self):
        """测试面试引擎初始化时集成降级决策中心"""
        engine = InterviewEngine()

        assert engine.degradation_controller is not None
        assert engine._degradation_level == 1

    def test_evaluator_selection_level_1(self):
        """测试 Level 1 使用 DeepSeek API"""
        engine = InterviewEngine()

        evaluator = engine._get_evaluator_for_current_level()

        # Level 1 应该返回 model_processor（DeepSeek）
        assert evaluator is not None
        assert engine._degradation_level == 1

    def test_evaluator_selection_after_degradation(self):
        """测试降级后评估器切换"""
        engine = InterviewEngine()

        # 强制降级到 Level 3
        engine.degradation_controller.force_level(3, reason="test")

        # 验证回调已触发
        assert engine._degradation_level == 3

        # Level 3 应该返回本地模型（如果可用）或 None
        evaluator = engine._get_evaluator_for_current_level()
        # 本地模型可用时返回 MultimodalEvaluator，否则返回 None
        if engine.multimodal_evaluator and engine.multimodal_evaluator.is_available():
            assert evaluator is not None
            assert type(evaluator).__name__ == "MultimodalEvaluator"
        else:
            assert evaluator is None

    def test_evaluator_selection_level_4(self):
        """测试 Level 4 使用规则引擎"""
        engine = InterviewEngine()

        engine.degradation_controller.force_level(4, reason="test")

        assert engine._degradation_level == 4

        evaluator = engine._get_evaluator_for_current_level()
        assert evaluator is None

    def test_degradation_callback(self):
        """测试降级回调功能"""
        engine = InterviewEngine()

        # 强制降级
        engine.degradation_controller.force_level(2, reason="test")

        assert engine._degradation_level == 2

        # 再降级
        engine.degradation_controller.force_level(3, reason="test")

        assert engine._degradation_level == 3

    def test_evaluate_with_degradation_level_1(self):
        """测试 Level 1 正常评估流程"""
        engine = InterviewEngine()
        session = engine.create_session("Java后端开发", "测试用户")
        engine.start_interview()

        # 模拟回答
        result = engine.process_answer("我是测试用户，有3年Java开发经验。")

        assert "error" not in result
        assert engine._degradation_level == 1

    def test_evaluate_with_degradation_level_4(self):
        """测试 Level 4 使用规则引擎评估"""
        engine = InterviewEngine()

        # 强制降级到 Level 4
        engine.degradation_controller.force_level(4, reason="test")

        session = engine.create_session("Java后端开发", "测试用户")
        engine.start_interview()

        # 模拟回答
        result = engine.process_answer("我是测试用户，有3年Java开发经验。")

        assert "error" not in result
        assert engine._degradation_level == 4

    def test_feature_availability_during_degradation(self):
        """测试降级期间功能可用性检查"""
        engine = InterviewEngine()

        # Level 1: streaming 可用
        assert engine.degradation_controller.is_feature_available("streaming")

        # 降级到 Level 3
        engine.degradation_controller.force_level(3, reason="test")

        # Level 3: streaming 不可用
        assert not engine.degradation_controller.is_feature_available("streaming")

        # Level 3: multimodal 仍可用
        assert engine.degradation_controller.is_feature_available("multimodal")

    def test_health_check_registration(self):
        """测试健康检查注册"""
        engine = InterviewEngine()

        health = engine.degradation_controller.health_checker
        assert "deepseek_api" in health.components


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
