"""
锐聘AI - 多模态融合决策引擎

整合API文本处理结果、Whisper转写文本、声纹分析数据和视频分析结果
进行综合分析并生成最终评判结果
"""

from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum

from ..logger import logger
from .text_consistency import TextConsistencyChecker
from .voice_integration import VoiceAnalysisIntegration


class FusionStrategy(Enum):
    """融合策略"""
    WEIGHTED_AVERAGE = "weighted_average"
    CONFIDENCE_BASED = "confidence_based"
    ADAPTIVE_WEIGHTING = "adaptive_weighting"


@dataclass
class ModalityResult:
    """模态结果"""
    scores: Dict[str, float]
    confidence: float
    reliability: float


class WeightedAverageFusion:
    """加权平均融合"""

    def fuse(self, text_scores: Dict, voice_scores: Dict,
             video_scores: Dict, weights: Dict[str, float]) -> Dict:
        """执行加权平均融合"""
        fused_scores = {}
        all_dims = set(text_scores.keys()) | set(voice_scores.keys()) | set(video_scores.keys())

        for dim in all_dims:
            total = 0.0
            weight_sum = 0.0

            for modality, weight in weights.items():
                scores = {"text": text_scores, "voice": voice_scores, "video": video_scores}[modality]
                if dim in scores:
                    total += scores[dim] * weight
                    weight_sum += weight

            if weight_sum > 0:
                fused_scores[dim] = round(total / weight_sum, 1)

        return {
            "scores": fused_scores,
            "confidence": 0.7,
            "strategy": "weighted_average"
        }


class ConfidenceBasedFusion:
    """基于置信度的融合"""

    def fuse(self, text_scores: Dict, voice_scores: Dict,
             video_scores: Dict, weights: Dict[str, float]) -> Dict:
        """执行置信度融合"""
        # 这里简化处理，实际应根据各模态置信度动态调整权重
        return WeightedAverageFusion().fuse(text_scores, voice_scores, video_scores, weights)


class AdaptiveWeightingFusion:
    """自适应加权融合"""

    def fuse(self, text_scores: Dict, voice_scores: Dict,
             video_scores: Dict, weights: Dict[str, float]) -> Dict:
        """执行自适应加权融合"""
        # 根据数据质量动态调整权重
        adjusted_weights = weights.copy()

        # 如果某模态数据缺失，降低其权重
        if not text_scores:
            adjusted_weights["text"] = 0.0
        if not voice_scores:
            adjusted_weights["voice"] = 0.0
        if not video_scores:
            adjusted_weights["video"] = 0.0

        # 归一化
        total = sum(adjusted_weights.values())
        if total > 0:
            adjusted_weights = {k: v / total for k, v in adjusted_weights.items()}

        return WeightedAverageFusion().fuse(text_scores, voice_scores, video_scores, adjusted_weights)


class MultimodalFusionEngine:
    """多模态融合决策引擎"""

    def __init__(self):
        self.text_checker = TextConsistencyChecker()
        self.voice_integrator = VoiceAnalysisIntegration()

        self.fusion_strategies = {
            FusionStrategy.WEIGHTED_AVERAGE: WeightedAverageFusion(),
            FusionStrategy.CONFIDENCE_BASED: ConfidenceBasedFusion(),
            FusionStrategy.ADAPTIVE_WEIGHTING: AdaptiveWeightingFusion()
        }

        logger.info("[OK] 多模态融合决策引擎初始化完成")

    def evaluate(self,
                 text_result: Dict,      # API文本处理结果
                 whisper_text: str,       # Whisper转写文本
                 voice_report: Dict,      # 声纹分析报告
                 video_analysis: Dict = None  # 视频多模态分析（可选）
                 ) -> Dict:
        """
        多模态融合评估

        融合策略:
        1. 文本一致性校验 (API结果 vs Whisper转写)
        2. 声纹-文本一致性校验
        3. 视频-声纹一致性校验
        4. 综合决策

        Args:
            text_result: API文本处理结果
            whisper_text: Whisper转写文本
            voice_report: 声纹分析报告
            video_analysis: 视频分析结果（可选）

        Returns:
            融合评估结果
        """
        logger.info("=" * 60)
        logger.info("开始多模态融合评估")
        logger.info("=" * 60)

        # 1. 文本一致性校验
        api_text = text_result.get("processed_text", "")
        text_consistency = self.text_checker.check(api_text, whisper_text)
        logger.info(f"[融合] 文本一致性: {text_consistency['similarity']:.2%} ({text_consistency['level']})")

        # 2. 声纹分析整合
        voice_integration = self.voice_integrator.integrate(voice_report)
        logger.info(f"[融合] 声纹整合: 综合评分={voice_integration['overall_voice_score']}, 可靠性={voice_integration['reliability']}")

        # 3. 提取各模态评分
        text_scores = text_result.get("dimension_scores", {})
        voice_scores = voice_integration.get("dimension_scores", {})
        video_scores = video_analysis.get("dimension_scores", {}) if video_analysis else {}

        # 4. 根据一致性选择融合策略
        strategy, weights = self._select_fusion_strategy(
            text_consistency["similarity"],
            voice_integration["reliability"],
            video_analysis is not None
        )

        logger.info(f"[融合] 选择策略: {strategy.value}, 权重: {weights}")

        # 5. 执行融合
        fusion_result = self.fusion_strategies[strategy].fuse(
            text_scores, voice_scores, video_scores, weights
        )

        # 6. 生成融合报告
        result = {
            "final_scores": fusion_result["scores"],
            "overall_score": self._calculate_overall_score(fusion_result["scores"]),
            "confidence": fusion_result["confidence"],
            "fusion_strategy": strategy.value,
            "weights": weights,
            "text_consistency": text_consistency,
            "voice_integration": voice_integration,
            "modality_reliability": {
                "text": text_result.get("confidence", 0.8),
                "voice": voice_integration.get("reliability", 0.7),
                "video": video_analysis.get("reliability", 0.75) if video_analysis else 0.0
            },
            "recommendation": self._generate_recommendation(
                fusion_result["scores"],
                text_consistency,
                voice_integration
            )
        }

        logger.info("=" * 60)
        logger.info("多模态融合评估完成")
        logger.info(f"[融合] 综合评分: {result['overall_score']}")
        logger.info(f"[融合] 置信度: {result['confidence']}")
        logger.info(f"[融合] 策略: {result['fusion_strategy']}")
        logger.info("=" * 60)

        return result

    def _select_fusion_strategy(self, text_consistency: float,
                                 voice_reliability: float,
                                 has_video: bool) -> tuple:
        """选择融合策略"""
        if text_consistency > 0.8:
            # 文本一致性高，信任文本评估
            strategy = FusionStrategy.CONFIDENCE_BASED
            weights = {"text": 0.5, "voice": 0.2, "video": 0.3}
        elif text_consistency > 0.5:
            # 文本一致性中等，降低文本权重
            strategy = FusionStrategy.ADAPTIVE_WEIGHTING
            weights = {"text": 0.3, "voice": 0.3, "video": 0.4}
        else:
            # 文本一致性低，主要依赖声纹和视频
            strategy = FusionStrategy.WEIGHTED_AVERAGE
            weights = {"text": 0.15, "voice": 0.35, "video": 0.5}

        # 如果没有视频数据，重新分配权重
        if not has_video:
            video_weight = weights.pop("video")
            total = sum(weights.values())
            for k in weights:
                weights[k] += video_weight * (weights[k] / total)

        # 如果声纹可靠性低，降低声纹权重
        if voice_reliability < 0.5:
            voice_weight = weights.get("voice", 0.2)
            weights["voice"] = voice_weight * 0.5
            # 重新归一化
            total = sum(weights.values())
            weights = {k: v / total for k, v in weights.items()}

        return strategy, weights

    def _calculate_overall_score(self, scores: Dict[str, float]) -> float:
        """计算综合评分"""
        if not scores:
            return 0.0

        # 简单平均
        return round(sum(scores.values()) / len(scores), 1)

    def _generate_recommendation(self, scores: Dict[str, float],
                                  text_consistency: Dict,
                                  voice_integration: Dict) -> str:
        """生成录用建议"""
        overall = self._calculate_overall_score(scores)

        if overall >= 85:
            return "强烈推荐录用"
        elif overall >= 70:
            return "推荐录用"
        elif overall >= 60:
            return "考虑录用"
        else:
            return "不建议录用"

    def batch_evaluate(self, evaluations: List[Dict]) -> List[Dict]:
        """批量评估"""
        results = []
        for eval_data in evaluations:
            result = self.evaluate(
                text_result=eval_data.get("text_result", {}),
                whisper_text=eval_data.get("whisper_text", ""),
                voice_report=eval_data.get("voice_report", {}),
                video_analysis=eval_data.get("video_analysis")
            )
            results.append(result)
        return results
