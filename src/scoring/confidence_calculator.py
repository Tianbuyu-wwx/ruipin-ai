"""
锐聘AI - 评分置信度计算器

基于数据质量、模型置信度、样本充分性计算评分置信度
低置信度时触发人工复核
"""

import math
from typing import Dict, List, Optional
from dataclasses import dataclass

from ..logger import logger


@dataclass
class ConfidenceFactors:
    """置信度因子"""
    data_quality: float = 0.0      # 数据质量 (0-1)
    model_confidence: float = 0.0   # 模型置信度 (0-1)
    sample_adequacy: float = 0.0    # 样本充分性 (0-1)
    evaluation_consistency: float = 0.0  # 评估一致性 (0-1)


class ScoreConfidenceCalculator:
    """评分置信度计算器"""

    # 置信度阈值
    CONFIDENCE_THRESHOLDS = {
        "high": 0.8,      # 高置信度
        "medium": 0.6,    # 中等置信度
        "low": 0.4        # 低置信度（触发人工复核）
    }

    def __init__(self):
        logger.info("[OK] 评分置信度计算器初始化完成")

    def calculate(self,
                  scores: Dict[str, float],
                  answer_length: int = 0,
                  has_audio: bool = False,
                  has_video: bool = False,
                  model_raw_confidence: Optional[float] = None,
                  num_evaluations: int = 1,
                  score_variance: Optional[float] = None) -> Dict[str, any]:
        """
        计算评分置信度

        Args:
            scores: 各维度评分
            answer_length: 回答长度
            has_audio: 是否有音频
            has_video: 是否有视频
            model_raw_confidence: 模型原始置信度
            num_evaluations: 评估次数
            score_variance: 评分方差

        Returns:
            置信度结果字典
        """
        factors = ConfidenceFactors()

        # 1. 数据质量评估
        factors.data_quality = self._calculate_data_quality(
            answer_length, has_audio, has_video, scores
        )

        # 2. 模型置信度
        factors.model_confidence = self._calculate_model_confidence(
            model_raw_confidence, scores
        )

        # 3. 样本充分性
        factors.sample_adequacy = self._calculate_sample_adequacy(
            answer_length, num_evaluations
        )

        # 4. 评估一致性
        factors.evaluation_consistency = self._calculate_consistency(
            scores, score_variance
        )

        # 计算综合置信度（加权平均）
        overall_confidence = (
            factors.data_quality * 0.30 +
            factors.model_confidence * 0.25 +
            factors.sample_adequacy * 0.25 +
            factors.evaluation_consistency * 0.20
        )

        # 确定置信度等级
        level = self._get_confidence_level(overall_confidence)

        # 确定是否需要人工复核
        needs_review = overall_confidence < self.CONFIDENCE_THRESHOLDS["low"]

        result = {
            "overall_confidence": round(overall_confidence, 4),
            "level": level,
            "needs_manual_review": needs_review,
            "factors": {
                "data_quality": round(factors.data_quality, 4),
                "model_confidence": round(factors.model_confidence, 4),
                "sample_adequacy": round(factors.sample_adequacy, 4),
                "evaluation_consistency": round(factors.evaluation_consistency, 4)
            },
            "thresholds": self.CONFIDENCE_THRESHOLDS,
            "recommendation": self._get_recommendation(overall_confidence, needs_review)
        }

        logger.info(f"置信度计算完成: {result['overall_confidence']:.2f} ({level})")
        if needs_review:
            logger.warning(f"置信度过低，建议人工复核!")

        return result

    def _calculate_data_quality(self, answer_length: int,
                                 has_audio: bool, has_video: bool,
                                 scores: Dict[str, float]) -> float:
        """计算数据质量分数"""
        quality = 0.5  # 基础分

        # 回答长度评分
        if answer_length >= 200:
            quality += 0.20
        elif answer_length >= 100:
            quality += 0.15
        elif answer_length >= 50:
            quality += 0.10
        elif answer_length >= 20:
            quality += 0.05
        else:
            quality -= 0.10

        # 多媒体数据加分
        if has_audio:
            quality += 0.10
        if has_video:
            quality += 0.10

        # 评分合理性检查
        avg_score = sum(scores.values()) / len(scores) if scores else 0
        if 30 <= avg_score <= 90:
            quality += 0.05
        elif avg_score < 20 or avg_score > 95:
            quality -= 0.05  # 极端分数可能表示数据问题

        return min(1.0, max(0.0, quality))

    def _calculate_model_confidence(self,
                                     model_raw_confidence: Optional[float],
                                     scores: Dict[str, float]) -> float:
        """计算模型置信度"""
        if model_raw_confidence is not None:
            return min(1.0, max(0.0, model_raw_confidence))

        # 无原始置信度时，基于评分离散度估算
        if not scores:
            return 0.5

        values = list(scores.values())
        if len(values) < 2:
            return 0.6

        # 计算标准差
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        std = math.sqrt(variance)

        # 标准差越小，置信度越高
        if std < 10:
            return 0.85
        elif std < 20:
            return 0.70
        elif std < 30:
            return 0.55
        else:
            return 0.40

    def _calculate_sample_adequacy(self, answer_length: int,
                                    num_evaluations: int) -> float:
        """计算样本充分性"""
        adequacy = 0.5

        # 回答长度充分性
        if answer_length >= 300:
            adequacy += 0.25
        elif answer_length >= 150:
            adequacy += 0.15
        elif answer_length >= 50:
            adequacy += 0.05

        # 评估次数充分性
        if num_evaluations >= 5:
            adequacy += 0.25
        elif num_evaluations >= 3:
            adequacy += 0.15
        elif num_evaluations >= 2:
            adequacy += 0.05

        return min(1.0, adequacy)

    def _calculate_consistency(self, scores: Dict[str, float],
                                score_variance: Optional[float]) -> float:
        """计算评估一致性"""
        if not scores:
            return 0.5

        if score_variance is not None:
            # 使用提供的方差
            if score_variance < 50:
                return 0.90
            elif score_variance < 100:
                return 0.75
            elif score_variance < 200:
                return 0.60
            else:
                return 0.40

        # 计算维度间一致性
        values = list(scores.values())
        if len(values) < 2:
            return 0.6

        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        std = math.sqrt(variance)

        # 标准差越小越一致
        if std < 5:
            return 0.95
        elif std < 10:
            return 0.85
        elif std < 15:
            return 0.70
        elif std < 25:
            return 0.55
        else:
            return 0.35

    def _get_confidence_level(self, confidence: float) -> str:
        """获取置信度等级"""
        if confidence >= self.CONFIDENCE_THRESHOLDS["high"]:
            return "high"
        elif confidence >= self.CONFIDENCE_THRESHOLDS["medium"]:
            return "medium"
        elif confidence >= self.CONFIDENCE_THRESHOLDS["low"]:
            return "low"
        else:
            return "very_low"

    def _get_recommendation(self, confidence: float,
                            needs_review: bool) -> str:
        """获取建议"""
        if needs_review:
            return "置信度过低，必须人工复核"
        elif confidence < self.CONFIDENCE_THRESHOLDS["medium"]:
            return "置信度较低，建议人工复核"
        elif confidence < self.CONFIDENCE_THRESHOLDS["high"]:
            return "置信度中等，可作为参考"
        else:
            return "置信度高，结果可信"

    def batch_calculate(self, evaluations: List[Dict]) -> List[Dict]:
        """批量计算置信度"""
        results = []
        for eval_data in evaluations:
            result = self.calculate(
                scores=eval_data.get("scores", {}),
                answer_length=eval_data.get("answer_length", 0),
                has_audio=eval_data.get("has_audio", False),
                has_video=eval_data.get("has_video", False),
                model_raw_confidence=eval_data.get("model_confidence"),
                num_evaluations=eval_data.get("num_evaluations", 1),
                score_variance=eval_data.get("score_variance")
            )
            results.append(result)
        return results
