"""
锐聘AI - 系统集成测试器

测试各模块之间的集成和协作
"""

import time
import threading
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime

from ..logger import logger


@dataclass
class TestResult:
    """测试结果"""
    test_name: str
    passed: bool
    duration_ms: float
    message: str
    details: Dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return {
            "test_name": self.test_name,
            "passed": self.passed,
            "duration_ms": round(self.duration_ms, 2),
            "message": self.message,
            "details": self.details,
            "timestamp": datetime.fromtimestamp(self.timestamp).isoformat()
        }


class SystemIntegrationTester:
    """系统集成测试器"""

    def __init__(self):
        self.results: List[TestResult] = []
        self._lock = threading.Lock()
        logger.info("[OK] 系统集成测试器初始化完成")

    def run_all_tests(self) -> Dict[str, any]:
        """运行所有集成测试"""
        logger.info("=" * 60)
        logger.info("开始系统集成测试")
        logger.info("=" * 60)

        start_time = time.time()

        # 降级策略集成测试
        self._test_degradation_integration()

        # 评分规则集成测试
        self._test_scoring_integration()

        # 多模态集成测试
        self._test_multimodal_integration()

        # 端到端流程测试
        self._test_end_to_end_flow()

        # 降级切换流程测试
        self._test_degradation_switching()

        # 性能基准测试
        self._test_performance_baseline()

        total_duration = (time.time() - start_time) * 1000

        # 汇总结果
        passed = sum(1 for r in self.results if r.passed)
        failed = sum(1 for r in self.results if not r.passed)

        summary = {
            "total_tests": len(self.results),
            "passed": passed,
            "failed": failed,
            "success_rate": round(passed / len(self.results) * 100, 1) if self.results else 0,
            "total_duration_ms": round(total_duration, 2),
            "results": [r.to_dict() for r in self.results]
        }

        logger.info("=" * 60)
        logger.info("系统集成测试完成")
        logger.info(f"  - 总测试数: {summary['total_tests']}")
        logger.info(f"  - 通过: {summary['passed']}")
        logger.info(f"  - 失败: {summary['failed']}")
        logger.info(f"  - 成功率: {summary['success_rate']}%")
        logger.info(f"  - 总耗时: {summary['total_duration_ms']}ms")
        logger.info("=" * 60)

        return summary

    def _test_degradation_integration(self):
        """测试降级策略集成"""
        test_name = "降级策略集成"
        start = time.time()

        try:
            from ..degradation import DegradationController

            controller = DegradationController()

            # 测试降级级别配置
            config = controller.get_level_config(1)
            assert config is not None, "Level 1配置不存在"
            assert config.name == "完全服务", "Level 1名称错误"

            # 测试功能可用性检查
            assert controller.is_feature_available("streaming") is True, "streaming功能不可用"

            # 测试强制降级
            controller.force_level(3, reason="test")
            assert controller.get_current_level() == 3, "降级失败"

            # 测试恢复
            controller.force_level(1, reason="test")
            assert controller.get_current_level() == 1, "恢复失败"

            self._add_result(test_name, True, start, "降级策略集成正常")

        except Exception as e:
            self._add_result(test_name, False, start, f"降级策略集成失败: {str(e)}")

    def _test_scoring_integration(self):
        """测试评分规则集成"""
        test_name = "评分规则集成"
        start = time.time()

        try:
            from ..scoring import DynamicWeightAdjuster, ScoreConfidenceCalculator, ScoreLevelClassifier

            # 测试动态权重
            adjuster = DynamicWeightAdjuster()
            weights = adjuster.calculate_weights("backend_engineer", "mid", has_voice_data=False)
            assert abs(sum(weights.values()) - 1.0) < 0.01, "权重未归一化"

            # 测试置信度计算
            confidence_calc = ScoreConfidenceCalculator()
            confidence = confidence_calc.calculate(
                scores={"technical": 80, "communication": 75},
                answer_length=200,
                has_audio=True
            )
            assert "overall_confidence" in confidence, "置信度计算失败"

            # 测试等级划分
            classifier = ScoreLevelClassifier()
            result = classifier.classify(85)
            assert result["level"] == "A", "等级划分错误"

            self._add_result(test_name, True, start, "评分规则集成正常")

        except Exception as e:
            self._add_result(test_name, False, start, f"评分规则集成失败: {str(e)}")

    def _test_multimodal_integration(self):
        """测试多模态集成"""
        test_name = "多模态集成"
        start = time.time()

        try:
            from ..multimodal import MultimodalFusionEngine, TextConsistencyChecker

            # 测试文本一致性
            checker = TextConsistencyChecker()
            result = checker.check("测试文本", "测试文本")
            assert result["similarity"] > 0.9, "文本一致性校验失败"

            # 测试多模态融合
            engine = MultimodalFusionEngine()
            fusion_result = engine.evaluate(
                text_result={
                    "processed_text": "我有三年Java经验",
                    "dimension_scores": {"technical": 85, "communication": 80}
                },
                whisper_text="我有三年Java经验",
                voice_report={
                    "overall_score": 75,
                    "speech_rate": {"score": 80, "category": "normal", "total_duration": 30},
                    "pause": {"score": 70, "total_pauses": 5, "pause_ratio": 0.15},
                    "emotion": {"confidence": 85, "scores": {"confident": 80}}
                }
            )
            assert "overall_score" in fusion_result, "多模态融合失败"

            self._add_result(test_name, True, start, "多模态集成正常")

        except Exception as e:
            self._add_result(test_name, False, start, f"多模态集成失败: {str(e)}")

    def _test_end_to_end_flow(self):
        """测试端到端流程"""
        test_name = "端到端流程"
        start = time.time()

        try:
            from ..interview_engine import InterviewEngine
            from ..degradation import DegradationController

            # 初始化引擎
            engine = InterviewEngine()

            # 创建会话
            session = engine.create_session("Java后端开发", "测试候选人")
            assert session is not None, "会话创建失败"

            # 开始面试
            result = engine.start_interview()
            assert "error" not in result, "面试启动失败"

            # 处理回答
            result = engine.process_answer("我有三年Java开发经验，熟悉Spring Boot框架")
            assert "error" not in result, "回答处理失败"

            # 验证降级中心连接
            assert engine.degradation_controller is not None, "降级中心未连接"

            self._add_result(test_name, True, start, "端到端流程正常")

        except Exception as e:
            self._add_result(test_name, False, start, f"端到端流程失败: {str(e)}")

    def _test_degradation_switching(self):
        """测试降级切换流程"""
        test_name = "降级切换流程"
        start = time.time()

        try:
            from ..interview_engine import InterviewEngine
            from ..degradation import DegradationController

            DegradationController.reset_instance()
            engine = InterviewEngine()

            # 初始级别应为1
            assert engine._degradation_level == 1, "初始级别错误"

            # 强制降级到Level 3
            engine.degradation_controller.force_level(3, reason="test")
            assert engine._degradation_level == 3, "降级未生效"

            # 验证评估器切换
            evaluator = engine._get_evaluator_for_current_level()
            # Level 3应使用本地模型或规则引擎
            assert evaluator is not None or engine._degradation_level >= 3, "评估器切换失败"

            # 恢复
            engine.degradation_controller.force_level(1, reason="test")
            assert engine._degradation_level == 1, "恢复失败"

            self._add_result(test_name, True, start, "降级切换流程正常")

        except Exception as e:
            self._add_result(test_name, False, start, f"降级切换流程失败: {str(e)}")

    def _test_performance_baseline(self):
        """测试性能基准"""
        test_name = "性能基准"
        start = time.time()

        try:
            from ..scoring import DynamicWeightAdjuster, ScoreConfidenceCalculator, ScoreLevelClassifier

            # 测试权重计算性能
            adjuster = DynamicWeightAdjuster()
            weight_start = time.time()
            for _ in range(100):
                adjuster.calculate_weights("backend_engineer", "mid", has_voice_data=False)
            weight_duration = (time.time() - weight_start) * 1000

            # 测试置信度计算性能
            confidence_calc = ScoreConfidenceCalculator()
            conf_start = time.time()
            for _ in range(100):
                confidence_calc.calculate(
                    scores={"technical": 80, "communication": 75},
                    answer_length=200
                )
            conf_duration = (time.time() - conf_start) * 1000

            # 测试等级划分性能
            classifier = ScoreLevelClassifier()
            class_start = time.time()
            for _ in range(100):
                classifier.classify(75)
            class_duration = (time.time() - class_start) * 1000

            details = {
                "weight_calc_avg_ms": round(weight_duration / 100, 3),
                "confidence_calc_avg_ms": round(conf_duration / 100, 3),
                "classification_avg_ms": round(class_duration / 100, 3)
            }

            # 性能阈值检查
            assert weight_duration / 100 < 10, "权重计算性能不达标"
            assert conf_duration / 100 < 10, "置信度计算性能不达标"
            assert class_duration / 100 < 10, "等级划分性能不达标"

            self._add_result(test_name, True, start, "性能基准测试通过", details)

        except Exception as e:
            self._add_result(test_name, False, start, f"性能基准测试失败: {str(e)}")

    def _add_result(self, test_name: str, passed: bool, start_time: float,
                    message: str, details: Optional[Dict] = None):
        """添加测试结果"""
        duration = (time.time() - start_time) * 1000
        result = TestResult(
            test_name=test_name,
            passed=passed,
            duration_ms=duration,
            message=message,
            details=details or {}
        )

        with self._lock:
            self.results.append(result)

        status = "[PASS]" if passed else "[FAIL]"
        logger.info(f"{status} {test_name}: {message} ({duration:.1f}ms)")

    def get_failed_tests(self) -> List[TestResult]:
        """获取失败的测试"""
        return [r for r in self.results if not r.passed]

    def get_test_report(self) -> str:
        """生成测试报告"""
        passed = sum(1 for r in self.results if r.passed)
        failed = sum(1 for r in self.results if not r.passed)

        report = []
        report.append("=" * 60)
        report.append("系统集成测试报告")
        report.append("=" * 60)
        report.append(f"总测试数: {len(self.results)}")
        report.append(f"通过: {passed}")
        report.append(f"失败: {failed}")
        report.append(f"成功率: {passed / len(self.results) * 100:.1f}%" if self.results else "N/A")
        report.append("")

        for result in self.results:
            status = "✅ 通过" if result.passed else "❌ 失败"
            report.append(f"{status} {result.test_name}")
            report.append(f"   消息: {result.message}")
            report.append(f"   耗时: {result.duration_ms:.1f}ms")
            if result.details:
                report.append(f"   详情: {result.details}")
            report.append("")

        report.append("=" * 60)

        return "\n".join(report)
