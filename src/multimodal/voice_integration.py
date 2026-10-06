"""
锐聘AI - 声纹分析整合器

将声纹分析结果映射到综合评分维度，作为独立评价维度纳入评分体系
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass

from ..logger import logger


@dataclass
class VoiceDimensionScores:
    """声纹维度评分"""
    communication: float = 0.0
    confidence: float = 0.0
    fluency: float = 0.0
    stability: float = 0.0
    overall: float = 0.0


class VoiceAnalysisIntegration:
    """声纹分析整合器"""

    # 声纹指标到评分维度的映射权重
    DIMENSION_MAPPING = {
        "communication": {
            "speech_rate_score": 0.40,
            "fluency_score": 0.35,
            "emotion_confidence": 0.25
        },
        "confidence": {
            "emotion_confidence": 0.50,
            "emotion_stability": 0.30,
            "speech_rate_consistency": 0.20
        },
        "fluency": {
            "fluency_score": 0.60,
            "pause_quality": 0.25,
            "speech_rate_score": 0.15
        },
        "stability": {
            "emotion_stability": 0.40,
            "pause_consistency": 0.35,
            "rate_variance": 0.25
        }
    }

    def __init__(self):
        logger.info("[OK] 声纹分析整合器初始化完成")

    def integrate(self, voice_report: Dict[str, Any]) -> Dict[str, any]:
        """
        将声纹分析报告整合为评分维度

        Args:
            voice_report: 声纹分析报告（来自VoiceAnalysisManager）

        Returns:
            整合后的评分结果
        """
        if not voice_report or voice_report.get("overall_score", 0) == 0:
            logger.warning("声纹分析报告为空或无效")
            return {
                "has_voice_data": False,
                "dimension_scores": {},
                "overall_voice_score": 0.0,
                "reliability": 0.0,
                "feedback": "无声纹数据"
            }

        # 提取声纹指标
        metrics = self._extract_metrics(voice_report)

        # 计算各维度评分
        dimension_scores = self._calculate_dimension_scores(metrics)

        # 计算综合声纹评分
        overall = self._calculate_overall_score(dimension_scores)

        # 计算可靠性
        reliability = self._calculate_reliability(voice_report)

        result = {
            "has_voice_data": True,
            "dimension_scores": dimension_scores,
            "overall_voice_score": round(overall, 1),
            "reliability": round(reliability, 2),
            "feedback": self._generate_feedback(dimension_scores, metrics),
            "raw_metrics": metrics
        }

        logger.info(f"声纹分析整合完成: 综合评分={result['overall_voice_score']}, 可靠性={reliability}")

        return result

    def _extract_metrics(self, report: Dict[str, Any]) -> Dict[str, float]:
        """从报告中提取关键指标"""
        metrics = {}

        # 语速指标
        speech_rate = report.get("speech_rate", {})
        metrics["speech_rate_score"] = speech_rate.get("score", 50)
        metrics["speech_rate_wcpm"] = speech_rate.get("wcpm", 0)
        metrics["speech_rate_category"] = speech_rate.get("category", "normal")

        # 停顿指标
        pause = report.get("pause", {})
        metrics["fluency_score"] = pause.get("score", 50)
        metrics["total_pauses"] = pause.get("total_pauses", 0)
        metrics["pause_ratio"] = pause.get("pause_ratio", 0)

        # 情绪指标
        emotion = report.get("emotion", {})
        metrics["emotion_confidence"] = emotion.get("confidence", 50)
        metrics["emotion_intensity"] = emotion.get("intensity", 50)
        metrics["primary_emotion"] = emotion.get("primary_emotion", "neutral")

        # 情绪分数
        scores = emotion.get("scores", {})
        metrics["confident_score"] = scores.get("confident", 50)
        metrics["nervous_score"] = scores.get("nervous", 50)
        metrics["neutral_score"] = scores.get("neutral", 50)

        # 计算情绪稳定性
        metrics["emotion_stability"] = self._calculate_emotion_stability(scores)

        # 计算语速一致性（基于语速类别）
        metrics["speech_rate_consistency"] = 80 if metrics["speech_rate_category"] == "normal" else 60

        # 停顿质量（停顿比例适中为佳）
        pause_ratio = metrics["pause_ratio"]
        if 0.1 <= pause_ratio <= 0.3:
            metrics["pause_quality"] = 85
        elif 0.05 <= pause_ratio < 0.1 or 0.3 < pause_ratio <= 0.5:
            metrics["pause_quality"] = 70
        else:
            metrics["pause_quality"] = 50

        # 停顿一致性（停顿数量适中）
        total_pauses = metrics["total_pauses"]
        if 3 <= total_pauses <= 10:
            metrics["pause_consistency"] = 80
        elif 1 <= total_pauses < 3 or 10 < total_pauses <= 15:
            metrics["pause_consistency"] = 65
        else:
            metrics["pause_consistency"] = 45

        # 语速变化率（假设值，实际可从rate_curve计算）
        metrics["rate_variance"] = 75  # 默认中等变化

        return metrics

    def _calculate_emotion_stability(self, scores: Dict[str, float]) -> float:
        """计算情绪稳定性"""
        if not scores:
            return 50.0

        # 情绪稳定性 = 100 - 情绪分数的标准差
        values = list(scores.values())
        if len(values) < 2:
            return 50.0

        import statistics
        try:
            std = statistics.stdev(values)
            stability = max(0, 100 - std * 2)
            return stability
        except statistics.StatisticsError:
            return 50.0

    def _calculate_dimension_scores(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """计算各维度评分"""
        scores = {}

        for dimension, weights in self.DIMENSION_MAPPING.items():
            score = 0.0
            total_weight = 0.0

            for metric_name, weight in weights.items():
                if metric_name in metrics:
                    score += metrics[metric_name] * weight
                    total_weight += weight

            if total_weight > 0:
                scores[dimension] = round(score / total_weight, 1)
            else:
                scores[dimension] = 50.0

        return scores

    def _calculate_overall_score(self, dimension_scores: Dict[str, float]) -> float:
        """计算综合声纹评分"""
        if not dimension_scores:
            return 0.0

        # 加权平均
        weights = {
            "communication": 0.30,
            "confidence": 0.25,
            "fluency": 0.25,
            "stability": 0.20
        }

        total = 0.0
        weight_sum = 0.0

        for dim, score in dimension_scores.items():
            w = weights.get(dim, 0.25)
            total += score * w
            weight_sum += w

        return total / weight_sum if weight_sum > 0 else 0.0

    def _calculate_reliability(self, report: Dict[str, Any]) -> float:
        """计算声纹分析的可靠性"""
        reliability = 0.7  # 基础可靠性

        # 有错误报告时降低可靠性
        if "error" in report:
            reliability -= 0.3

        # 根据音频时长调整（假设有duration信息）
        speech_rate = report.get("speech_rate", {})
        duration = speech_rate.get("total_duration", 0)
        if duration < 10:
            reliability -= 0.1
        elif duration > 60:
            reliability += 0.1

        # 根据情绪置信度调整
        emotion = report.get("emotion", {})
        conf = emotion.get("confidence", 50)
        if conf > 80:
            reliability += 0.1
        elif conf < 50:
            reliability -= 0.1

        return min(1.0, max(0.0, reliability))

    def _generate_feedback(self, dimension_scores: Dict[str, float],
                           metrics: Dict[str, float]) -> str:
        """生成反馈建议"""
        feedback_parts = []

        # 根据各维度表现生成反馈
        if dimension_scores.get("communication", 0) >= 80:
            feedback_parts.append("沟通表达能力优秀")
        elif dimension_scores.get("communication", 0) < 60:
            feedback_parts.append("沟通表达能力有待提升")

        if dimension_scores.get("confidence", 0) >= 80:
            feedback_parts.append("表现自信从容")
        elif dimension_scores.get("confidence", 0) < 60:
            feedback_parts.append("建议增强自信心")

        if dimension_scores.get("fluency", 0) >= 80:
            feedback_parts.append("表达流畅自然")
        elif dimension_scores.get("fluency", 0) < 60:
            feedback_parts.append("表达流畅度需要改善")

        if dimension_scores.get("stability", 0) >= 80:
            feedback_parts.append("情绪状态稳定")
        elif dimension_scores.get("stability", 0) < 60:
            feedback_parts.append("情绪稳定性有待加强")

        return "；".join(feedback_parts) if feedback_parts else "声纹特征正常"

    def merge_with_text_scores(self, voice_scores: Dict[str, float],
                               text_scores: Dict[str, float],
                               voice_weight: float = 0.15) -> Dict[str, float]:
        """
        将声纹评分与文本评分融合

        Args:
            voice_scores: 声纹维度评分
            text_scores: 文本维度评分
            voice_weight: 声纹权重（默认15%）

        Returns:
            融合后的评分
        """
        merged = {}

        # 映射声纹维度到文本维度
        dimension_map = {
            "communication": "communication",
            "confidence": "communication",
            "fluency": "communication",
            "stability": "teamwork"
        }

        text_weight = 1.0 - voice_weight

        for dim, text_score in text_scores.items():
            voice_score = 0.0
            voice_count = 0

            # 找到对应的声纹维度
            for v_dim, t_dim in dimension_map.items():
                if t_dim == dim and v_dim in voice_scores:
                    voice_score += voice_scores[v_dim]
                    voice_count += 1

            if voice_count > 0:
                voice_avg = voice_score / voice_count
                merged[dim] = round(text_score * text_weight + voice_avg * voice_weight, 1)
            else:
                merged[dim] = text_score

        return merged
