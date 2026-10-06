"""
锐聘AI - 多模态评价整合模块单元测试

测试文本一致性校验、声纹分析整合、多模态融合决策
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import pytest
from unittest.mock import Mock

from src.multimodal import (
    TextConsistencyChecker,
    VoiceAnalysisIntegration,
    MultimodalFusionEngine,
)


class TestTextConsistencyChecker:
    """文本一致性校验器测试"""

    def setup_method(self):
        self.checker = TextConsistencyChecker()

    def test_identical_texts(self):
        """测试完全相同的文本"""
        result = self.checker.check(
            "我有三年Java开发经验",
            "我有三年Java开发经验"
        )

        assert result["similarity"] > 0.95
        assert result["level"] == "high"
        assert result["is_consistent"] is True

    def test_similar_texts(self):
        """测试相似文本"""
        result = self.checker.check(
            "我有三年Java开发经验",
            "我有3年Java开发经验"
        )

        assert result["similarity"] > 0.5
        assert result["is_consistent"] is True

    def test_different_texts(self):
        """测试不同文本"""
        result = self.checker.check(
            "我有三年Java开发经验",
            "今天天气很好"
        )

        assert result["similarity"] < 0.3
        assert result["level"] in ("low", "none")
        assert result["is_consistent"] is False

    def test_empty_text(self):
        """测试空文本"""
        result = self.checker.check("", "测试文本")

        assert result["similarity"] == 0.0
        assert result["level"] == "none"

    def test_mismatched_segments(self):
        """测试差异片段提取"""
        result = self.checker.check(
            "我有三年Java开发经验",
            "我有五年Python开发经验"
        )

        assert result["diff_count"] > 0
        assert len(result["mismatched_segments"]) > 0

    def test_batch_check(self):
        """测试批量校验"""
        pairs = [
            ("文本一", "文本一"),
            ("文本A", "文本B"),
            ("相同文本", "相同文本"),
        ]

        results = self.checker.batch_check(pairs)

        assert len(results) == 3
        assert results[0]["similarity"] > 0.9
        assert results[2]["similarity"] > 0.9


class TestVoiceAnalysisIntegration:
    """声纹分析整合器测试"""

    def setup_method(self):
        self.integrator = VoiceAnalysisIntegration()

    def test_valid_voice_report(self):
        """测试有效声纹报告"""
        report = {
            "overall_score": 75.0,
            "speech_rate": {
                "score": 80.0,
                "wcpm": 150.0,
                "category": "normal",
                "total_duration": 30.0
            },
            "pause": {
                "score": 70.0,
                "total_pauses": 5,
                "pause_ratio": 0.15
            },
            "emotion": {
                "primary_emotion": "confident",
                "confidence": 85.0,
                "intensity": 70.0,
                "scores": {
                    "confident": 80.0,
                    "nervous": 20.0,
                    "neutral": 50.0
                }
            }
        }

        result = self.integrator.integrate(report)

        assert result["has_voice_data"] is True
        assert result["overall_voice_score"] > 0
        assert "dimension_scores" in result
        assert "communication" in result["dimension_scores"]
        assert "confidence" in result["dimension_scores"]
        assert "fluency" in result["dimension_scores"]
        assert "stability" in result["dimension_scores"]
        assert result["reliability"] > 0

    def test_empty_report(self):
        """测试空报告"""
        result = self.integrator.integrate({})

        assert result["has_voice_data"] is False
        assert result["overall_voice_score"] == 0.0

    def test_error_report(self):
        """测试错误报告"""
        report = {
            "overall_score": 0,
            "error": "分析失败",
            "speech_rate": {"score": 0},
            "pause": {"score": 0},
            "emotion": {"confidence": 0}
        }

        result = self.integrator.integrate(report)

        # overall_score=0 被视为无效报告
        assert result["has_voice_data"] is False
        assert result["reliability"] == 0.0

    def test_merge_with_text_scores(self):
        """测试与文本评分融合"""
        voice_scores = {
            "communication": 80.0,
            "confidence": 75.0,
            "fluency": 70.0,
            "stability": 85.0
        }

        text_scores = {
            "technical": 82.0,
            "communication": 78.0,
            "problem_solving": 80.0,
            "teamwork": 75.0
        }

        merged = self.integrator.merge_with_text_scores(
            voice_scores, text_scores, voice_weight=0.15
        )

        assert "communication" in merged
        assert "technical" in merged
        # 沟通维度应该受到声纹影响
        assert merged["communication"] != text_scores["communication"]

    def test_dimension_mapping(self):
        """测试维度映射"""
        report = {
            "overall_score": 80.0,
            "speech_rate": {
                "score": 85.0,
                "category": "normal"
            },
            "pause": {
                "score": 75.0,
                "total_pauses": 6,
                "pause_ratio": 0.2
            },
            "emotion": {
                "confidence": 90.0,
                "scores": {
                    "confident": 85.0,
                    "nervous": 15.0,
                    "neutral": 55.0
                }
            }
        }

        result = self.integrator.integrate(report)
        scores = result["dimension_scores"]

        # 各维度应该有合理评分
        assert 0 <= scores["communication"] <= 100
        assert 0 <= scores["confidence"] <= 100
        assert 0 <= scores["fluency"] <= 100
        assert 0 <= scores["stability"] <= 100


class TestMultimodalFusionEngine:
    """多模态融合决策引擎测试"""

    def setup_method(self):
        self.engine = MultimodalFusionEngine()

    def test_high_text_consistency(self):
        """测试高文本一致性场景"""
        text_result = {
            "processed_text": "我有三年Java开发经验",
            "dimension_scores": {
                "technical": 85,
                "communication": 80,
                "problem_solving": 82
            },
            "confidence": 0.9
        }

        whisper_text = "我有三年Java开发经验"

        voice_report = {
            "overall_score": 75.0,
            "speech_rate": {"score": 80, "category": "normal", "total_duration": 30},
            "pause": {"score": 70, "total_pauses": 5, "pause_ratio": 0.15},
            "emotion": {
                "confidence": 85,
                "scores": {"confident": 80, "nervous": 20, "neutral": 50}
            }
        }

        result = self.engine.evaluate(text_result, whisper_text, voice_report)

        assert result["fusion_strategy"] == "confidence_based"
        assert result["text_consistency"]["level"] == "high"
        assert result["overall_score"] > 0
        assert "recommendation" in result

    def test_low_text_consistency(self):
        """测试低文本一致性场景"""
        text_result = {
            "processed_text": "我有三年Java开发经验",
            "dimension_scores": {
                "technical": 85,
                "communication": 80
            }
        }

        whisper_text = "今天天气很好，我喜欢编程"

        voice_report = {
            "overall_score": 75.0,
            "speech_rate": {"score": 80, "category": "normal", "total_duration": 30},
            "pause": {"score": 70, "total_pauses": 5, "pause_ratio": 0.15},
            "emotion": {
                "confidence": 85,
                "scores": {"confident": 80, "nervous": 20, "neutral": 50}
            }
        }

        result = self.engine.evaluate(text_result, whisper_text, voice_report)

        assert result["fusion_strategy"] == "weighted_average"
        assert result["text_consistency"]["level"] in ("low", "none")
        # 低一致性时文本权重应降低（<= 0.3）
        assert result["weights"]["text"] <= 0.3

    def test_without_video(self):
        """测试无视频数据场景"""
        text_result = {
            "processed_text": "测试文本",
            "dimension_scores": {"technical": 80}
        }

        whisper_text = "测试文本"

        voice_report = {
            "overall_score": 70.0,
            "speech_rate": {"score": 75, "category": "normal", "total_duration": 20},
            "pause": {"score": 65, "total_pauses": 4, "pause_ratio": 0.2},
            "emotion": {
                "confidence": 80,
                "scores": {"confident": 75, "nervous": 25, "neutral": 50}
            }
        }

        result = self.engine.evaluate(text_result, whisper_text, voice_report)

        # 无视频时，视频权重应为0
        assert "video" not in result["weights"] or result["weights"]["video"] == 0
        assert result["overall_score"] > 0

    def test_empty_inputs(self):
        """测试空输入场景"""
        text_result = {"processed_text": "", "dimension_scores": {}}
        whisper_text = ""
        voice_report = {}

        result = self.engine.evaluate(text_result, whisper_text, voice_report)

        assert result["overall_score"] == 0.0
        assert result["text_consistency"]["level"] == "none"

    def test_strategy_selection(self):
        """测试策略选择逻辑"""
        # 高一致性 -> confidence_based
        strategy, weights = self.engine._select_fusion_strategy(
            0.9, 0.8, True
        )
        assert strategy.value == "confidence_based"
        assert weights["text"] > weights["voice"]

        # 中等一致性 -> adaptive_weighting
        strategy, weights = self.engine._select_fusion_strategy(
            0.6, 0.8, True
        )
        assert strategy.value == "adaptive_weighting"

        # 低一致性 -> weighted_average
        strategy, weights = self.engine._select_fusion_strategy(
            0.3, 0.8, True
        )
        assert strategy.value == "weighted_average"
        assert weights["text"] < weights["voice"]

    def test_low_voice_reliability(self):
        """测试低声纹可靠性场景"""
        strategy, weights = self.engine._select_fusion_strategy(
            0.9, 0.3, True
        )

        # 声纹可靠性低时，声纹权重应降低
        assert weights["voice"] < 0.2

    def test_batch_evaluate(self):
        """测试批量评估"""
        evaluations = [
            {
                "text_result": {
                    "processed_text": "文本1",
                    "dimension_scores": {"technical": 80}
                },
                "whisper_text": "文本1",
                "voice_report": {
                    "overall_score": 70,
                    "speech_rate": {"score": 75, "category": "normal", "total_duration": 20},
                    "pause": {"score": 65, "total_pauses": 4, "pause_ratio": 0.2},
                    "emotion": {"confidence": 80, "scores": {"confident": 75}}
                }
            },
            {
                "text_result": {
                    "processed_text": "文本2",
                    "dimension_scores": {"technical": 90}
                },
                "whisper_text": "文本2",
                "voice_report": {
                    "overall_score": 80,
                    "speech_rate": {"score": 85, "category": "normal", "total_duration": 25},
                    "pause": {"score": 75, "total_pauses": 3, "pause_ratio": 0.15},
                    "emotion": {"confidence": 85, "scores": {"confident": 80}}
                }
            }
        ]

        results = self.engine.batch_evaluate(evaluations)

        assert len(results) == 2
        assert all("overall_score" in r for r in results)


class TestMultimodalIntegration:
    """多模态集成测试"""

    def test_full_pipeline(self):
        """测试完整多模态评估流程"""
        # 1. 文本一致性校验
        checker = TextConsistencyChecker()
        consistency = checker.check(
            "我有三年Java开发经验，熟悉Spring Boot框架",
            "我有三年Java开发经验，熟悉Spring Boot框架"
        )

        # 2. 声纹整合
        integrator = VoiceAnalysisIntegration()
        voice_report = {
            "overall_score": 78.0,
            "speech_rate": {
                "score": 82.0,
                "wcpm": 155.0,
                "category": "normal",
                "total_duration": 45.0
            },
            "pause": {
                "score": 75.0,
                "total_pauses": 6,
                "pause_ratio": 0.18
            },
            "emotion": {
                "primary_emotion": "confident",
                "confidence": 88.0,
                "intensity": 72.0,
                "scores": {
                    "confident": 85.0,
                    "nervous": 15.0,
                    "neutral": 55.0
                }
            }
        }
        voice_result = integrator.integrate(voice_report)

        # 3. 多模态融合
        engine = MultimodalFusionEngine()
        text_result = {
            "processed_text": "我有三年Java开发经验，熟悉Spring Boot框架",
            "dimension_scores": {
                "technical": 85.0,
                "communication": 80.0,
                "problem_solving": 82.0,
                "teamwork": 75.0
            },
            "confidence": 0.9
        }

        fusion_result = engine.evaluate(
            text_result=text_result,
            whisper_text="我有三年Java开发经验，熟悉Spring Boot框架",
            voice_report=voice_report
        )

        # 验证结果
        assert fusion_result["overall_score"] > 0
        assert fusion_result["text_consistency"]["is_consistent"] is True
        assert fusion_result["voice_integration"]["has_voice_data"] is True
        assert "final_scores" in fusion_result
        assert "recommendation" in fusion_result

    def test_voice_text_merge(self):
        """测试声纹与文本评分融合"""
        integrator = VoiceAnalysisIntegration()

        voice_scores = {
            "communication": 85.0,
            "confidence": 80.0,
            "fluency": 75.0,
            "stability": 82.0
        }

        text_scores = {
            "technical": 88.0,
            "communication": 78.0,
            "problem_solving": 85.0,
            "teamwork": 72.0,
            "leadership": 70.0
        }

        merged = integrator.merge_with_text_scores(
            voice_scores, text_scores, voice_weight=0.15
        )

        # 验证融合后的评分
        assert len(merged) == len(text_scores)
        # 技术维度不受声纹影响（无声纹对应维度）
        assert merged["technical"] == text_scores["technical"]
        # 沟通维度受声纹影响
        assert merged["communication"] != text_scores["communication"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
