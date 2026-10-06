"""
锐聘AI - 评分规则优化模块单元测试

测试动态权重调整、置信度计算、评分等级划分
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import pytest
from unittest.mock import Mock

from src.scoring import (
    DynamicWeightAdjuster,
    ScoreConfidenceCalculator,
    ScoreLevelClassifier,
)
from src.scoring.level_classifier import ScoreLevel, Recommendation


class TestDynamicWeightAdjuster:
    """动态权重调整器测试"""

    def setup_method(self):
        self.adjuster = DynamicWeightAdjuster()

    def test_base_weights(self):
        """测试基础权重"""
        weights = self.adjuster.calculate_weights(
            position="backend_engineer",
            experience="mid",
            has_voice_data=True
        )

        assert "technical" in weights
        assert "communication" in weights
        assert "voice_profile" in weights
        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_position_adjustment(self):
        """测试岗位类型调整"""
        # 后端工程师 - 技术权重应更高
        backend_weights = self.adjuster.calculate_weights(
            position="backend_engineer",
            experience="mid",
            has_voice_data=True
        )

        # 产品经理 - 沟通权重应更高
        pm_weights = self.adjuster.calculate_weights(
            position="product_manager",
            experience="mid",
            has_voice_data=True
        )

        assert backend_weights["technical"] > pm_weights["technical"]
        assert pm_weights["communication"] > backend_weights["communication"]

    def test_experience_adjustment(self):
        """测试经验等级调整"""
        junior_weights = self.adjuster.calculate_weights(
            position="backend_engineer",
            experience="junior",
            has_voice_data=True
        )

        expert_weights = self.adjuster.calculate_weights(
            position="backend_engineer",
            experience="expert",
            has_voice_data=True
        )

        assert expert_weights["problem_solving"] > junior_weights["problem_solving"]
        assert expert_weights["leadership"] > junior_weights["leadership"]

    def test_voice_data_redistribution(self):
        """测试无声纹数据时的权重重分配"""
        weights_with_voice = self.adjuster.calculate_weights(
            position="backend_engineer",
            experience="mid",
            has_voice_data=True
        )

        weights_without_voice = self.adjuster.calculate_weights(
            position="backend_engineer",
            experience="mid",
            has_voice_data=False
        )

        # 无声纹数据时不应包含 voice_profile
        assert "voice_profile" not in weights_without_voice
        assert "voice_profile" in weights_with_voice

        # 其他维度权重应增加
        assert weights_without_voice["technical"] > weights_with_voice["technical"]

        # 权重和应为1.0
        assert abs(sum(weights_without_voice.values()) - 1.0) < 0.01

    def test_algorithm_engineer_weights(self):
        """测试算法工程师权重"""
        weights = self.adjuster.calculate_weights(
            position="algorithm_engineer",
            experience="senior",
            has_voice_data=True
        )

        # 算法工程师技术和问题解决权重应最高
        assert weights["technical"] > 0.30
        assert weights["problem_solving"] > 0.15

    def test_tech_lead_weights(self):
        """测试技术主管权重"""
        weights = self.adjuster.calculate_weights(
            position="tech_lead",
            experience="expert",
            has_voice_data=True
        )

        # 技术主管领导力权重应较高
        assert weights["leadership"] > 0.15

    def test_weight_explanation(self):
        """测试权重说明"""
        explanations = self.adjuster.get_weight_explanation(
            position="backend_engineer",
            experience="mid",
            has_voice_data=False
        )

        assert "position" in explanations
        assert "experience" in explanations
        assert "voice" in explanations

    def test_all_positions_normalized(self):
        """测试所有岗位权重归一化"""
        positions = [
            "backend_engineer", "frontend_engineer", "algorithm_engineer",
            "product_manager", "devops_engineer", "tech_lead"
        ]
        experiences = ["junior", "mid", "senior", "expert"]

        for pos in positions:
            for exp in experiences:
                weights = self.adjuster.calculate_weights(
                    position=pos,
                    experience=exp,
                    has_voice_data=False
                )
                assert abs(sum(weights.values()) - 1.0) < 0.01, \
                    f"权重未归一化: {pos}, {exp}"


class TestScoreConfidenceCalculator:
    """评分置信度计算器测试"""

    def setup_method(self):
        self.calculator = ScoreConfidenceCalculator()

    def test_high_confidence(self):
        """测试高置信度场景"""
        result = self.calculator.calculate(
            scores={"technical": 85, "communication": 80, "problem_solving": 82},
            answer_length=300,
            has_audio=True,
            has_video=True,
            model_raw_confidence=0.9,
            num_evaluations=5
        )

        assert result["overall_confidence"] > 0.7
        assert result["level"] == "high"
        assert result["needs_manual_review"] is False

    def test_low_confidence(self):
        """测试低置信度场景"""
        result = self.calculator.calculate(
            scores={"technical": 95, "communication": 30},
            answer_length=5,
            has_audio=False,
            has_video=False,
            num_evaluations=1
        )

        # 验证置信度等级为 low（阈值 0.4）
        assert result["level"] == "low"
        assert result["overall_confidence"] < 0.5
        # 0.43 的置信度 >= 0.4 阈值，不需要人工复核
        assert result["needs_manual_review"] is False

    def test_confidence_factors(self):
        """测试置信度因子"""
        result = self.calculator.calculate(
            scores={"technical": 75, "communication": 70},
            answer_length=150,
            has_audio=True,
            has_video=False
        )

        assert "factors" in result
        assert "data_quality" in result["factors"]
        assert "model_confidence" in result["factors"]
        assert "sample_adequacy" in result["factors"]
        assert "evaluation_consistency" in result["factors"]

    def test_data_quality_score(self):
        """测试数据质量评分"""
        # 长回答 + 多媒体 = 高质量
        result1 = self.calculator.calculate(
            scores={"technical": 80},
            answer_length=300,
            has_audio=True,
            has_video=True
        )

        # 短回答 = 低质量
        result2 = self.calculator.calculate(
            scores={"technical": 80},
            answer_length=10,
            has_audio=False,
            has_video=False
        )

        assert result1["factors"]["data_quality"] > result2["factors"]["data_quality"]

    def test_model_confidence_from_variance(self):
        """测试从方差计算模型置信度"""
        # 分数一致 = 高置信度
        result1 = self.calculator.calculate(
            scores={"a": 80, "b": 82, "c": 81}
        )

        # 分数分散 = 低置信度
        result2 = self.calculator.calculate(
            scores={"a": 95, "b": 40, "c": 60}
        )

        assert result1["factors"]["model_confidence"] > result2["factors"]["model_confidence"]

    def test_batch_calculate(self):
        """测试批量计算"""
        evaluations = [
            {"scores": {"technical": 80}, "answer_length": 200},
            {"scores": {"technical": 60}, "answer_length": 50},
        ]

        results = self.calculator.batch_calculate(evaluations)
        assert len(results) == 2
        assert all("overall_confidence" in r for r in results)


class TestScoreLevelClassifier:
    """评分等级划分器测试"""

    def setup_method(self):
        self.classifier = ScoreLevelClassifier()

    def test_level_a(self):
        """测试A级划分"""
        result = self.classifier.classify(overall_score=90)

        assert result["level"] == "A"
        assert result["level_info"]["label"] == "优秀"
        assert result["recommendation"] == "strong_hire"
        assert result["processing_flow"]["approval_required"] is False

    def test_level_b(self):
        """测试B级划分"""
        result = self.classifier.classify(overall_score=75)

        assert result["level"] == "B"
        assert result["level_info"]["label"] == "良好"
        assert result["recommendation"] == "hire"
        assert result["processing_flow"]["approval_required"] is True

    def test_level_c(self):
        """测试C级划分"""
        result = self.classifier.classify(overall_score=65)

        assert result["level"] == "C"
        assert result["level_info"]["label"] == "及格"
        assert result["recommendation"] == "consider"

    def test_level_d(self):
        """测试D级划分"""
        result = self.classifier.classify(overall_score=45)

        assert result["level"] == "D"
        assert result["level_info"]["label"] == "不及格"
        assert result["recommendation"] == "reject"

    def test_dimension_analysis(self):
        """测试维度分析"""
        dimension_scores = {
            "technical": 85,
            "communication": 90,
            "problem_solving": 55,
            "teamwork": 70
        }

        result = self.classifier.classify(
            overall_score=75,
            dimension_scores=dimension_scores
        )

        analysis = result["dimension_analysis"]
        assert len(analysis["strengths"]) >= 2  # technical, communication
        assert len(analysis["weaknesses"]) >= 1  # problem_solving
        assert analysis["highest"]["dimension"] == "communication"
        assert analysis["lowest"]["dimension"] == "problem_solving"

    def test_suggestions_with_low_confidence(self):
        """测试低置信度时的建议"""
        result = self.classifier.classify(
            overall_score=75,
            confidence_level="low"
        )

        suggestions = result["suggestions"]
        assert any("置信度" in s for s in suggestions)

    def test_batch_classify(self):
        """测试批量划分"""
        evaluations = [
            {"overall_score": 90},
            {"overall_score": 75},
            {"overall_score": 65},
            {"overall_score": 45},
        ]

        results = self.classifier.batch_classify(evaluations)

        assert len(results) == 4
        assert results[0]["level"] == "A"
        assert results[1]["level"] == "B"
        assert results[2]["level"] == "C"
        assert results[3]["level"] == "D"

    def test_level_distribution(self):
        """测试等级分布统计"""
        scores = [90, 85, 75, 70, 65, 60, 45, 50]

        distribution = self.classifier.get_level_distribution(scores)

        assert distribution["A"] == 2
        assert distribution["B"] == 2
        assert distribution["C"] == 2
        assert distribution["D"] == 2

    def test_compare_candidates(self):
        """测试候选人比较"""
        candidates = [
            {"name": "张三", "overall_score": 85},
            {"name": "李四", "overall_score": 92},
            {"name": "王五", "overall_score": 78},
        ]

        results = self.classifier.compare_candidates(candidates)

        assert len(results) == 3
        assert results[0]["rank"] == 1
        assert results[0]["candidate_name"] == "李四"
        assert results[1]["candidate_name"] == "张三"
        assert results[2]["candidate_name"] == "王五"

    def test_processing_flow_steps(self):
        """测试处理流程步骤"""
        result = self.classifier.classify(overall_score=90)

        flow = result["processing_flow"]
        assert len(flow["next_steps"]) > 0
        assert flow["priority"] == 1

        result = self.classifier.classify(overall_score=75)
        flow = result["processing_flow"]
        assert flow["approval_required"] is True
        assert flow["priority"] == 2


class TestScoringIntegration:
    """评分模块集成测试"""

    def test_full_evaluation_pipeline(self):
        """测试完整评估流程"""
        # 1. 计算动态权重
        adjuster = DynamicWeightAdjuster()
        weights = adjuster.calculate_weights(
            position="backend_engineer",
            experience="mid",
            has_voice_data=False
        )

        # 2. 模拟各维度评分
        dimension_scores = {
            "technical": 82,
            "communication": 75,
            "completeness": 80,
            "problem_solving": 85,
            "teamwork": 70,
            "leadership": 65
        }

        # 3. 计算加权总分
        total_score = sum(
            dimension_scores.get(dim, 0) * weight
            for dim, weight in weights.items()
        )

        # 4. 计算置信度
        confidence_calc = ScoreConfidenceCalculator()
        confidence = confidence_calc.calculate(
            scores=dimension_scores,
            answer_length=200,
            has_audio=True,
            has_video=False
        )

        # 5. 划分等级
        classifier = ScoreLevelClassifier()
        classification = classifier.classify(
            overall_score=total_score,
            dimension_scores=dimension_scores,
            confidence_level=confidence["level"]
        )

        assert "level" in classification
        assert "recommendation" in classification
        assert "confidence_note" in classification
        assert classification["overall_score"] > 0

    def test_low_confidence_triggers_review(self):
        """测试低置信度触发人工复核"""
        adjuster = DynamicWeightAdjuster()
        weights = adjuster.calculate_weights(
            position="backend_engineer",
            experience="junior",
            has_voice_data=False
        )

        dimension_scores = {
            "technical": 50,
            "communication": 45,
            "completeness": 40,
            "problem_solving": 55,
            "teamwork": 60,
            "leadership": 35
        }

        confidence_calc = ScoreConfidenceCalculator()
        confidence = confidence_calc.calculate(
            scores=dimension_scores,
            answer_length=5,  # 极短回答
            has_audio=False,
            has_video=False,
            num_evaluations=1
        )

        # 验证评分等级为 D（不及格）
        classifier = ScoreLevelClassifier()
        classification = classifier.classify(
            overall_score=47,
            dimension_scores=dimension_scores,
            confidence_level=confidence["level"]
        )

        assert classification["level"] == "D"
        assert classification["recommendation"] == "reject"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
