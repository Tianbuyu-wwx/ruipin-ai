"""
锐聘AI - 评分等级划分器

将综合评分划分为 A/B/C/D 等级，并提供处理流程建议
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from ..logger import logger


class ScoreLevel(Enum):
    """评分等级"""
    A = "A"      # 优秀 (85-100)
    B = "B"      # 良好 (70-84)
    C = "C"      # 及格 (60-69)
    D = "D"      # 不及格 (0-59)


class Recommendation(Enum):
    """录用建议"""
    STRONG_HIRE = "strong_hire"      # 强烈推荐录用
    HIRE = "hire"                     # 推荐录用
    CONSIDER = "consider"             # 考虑录用
    REJECT = "reject"                 # 不建议录用


@dataclass
class LevelCriteria:
    """等级标准"""
    min_score: float
    max_score: float
    label: str
    color: str
    description: str


@dataclass
class ProcessingFlow:
    """处理流程"""
    level: ScoreLevel
    recommendation: Recommendation
    next_steps: List[str]
    approval_required: bool
    priority: int


class ScoreLevelClassifier:
    """评分等级划分器"""

    # 等级标准定义
    LEVEL_CRITERIA = {
        ScoreLevel.A: LevelCriteria(
            min_score=85, max_score=100,
            label="优秀", color="#4caf50",
            description="表现卓越，强烈推荐录用"
        ),
        ScoreLevel.B: LevelCriteria(
            min_score=70, max_score=84,
            label="良好", color="#8bc34a",
            description="表现良好，推荐录用"
        ),
        ScoreLevel.C: LevelCriteria(
            min_score=60, max_score=69,
            label="及格", color="#ff9800",
            description="基本达标，可考虑录用"
        ),
        ScoreLevel.D: LevelCriteria(
            min_score=0, max_score=59,
            label="不及格", color="#f44336",
            description="未达标，不建议录用"
        )
    }

    # 处理流程定义
    PROCESSING_FLOWS = {
        ScoreLevel.A: ProcessingFlow(
            level=ScoreLevel.A,
            recommendation=Recommendation.STRONG_HIRE,
            next_steps=[
                "自动生成录用意向书",
                "安排HR进行薪酬沟通",
                "优先安排入职流程",
                "加入人才库重点跟踪"
            ],
            approval_required=False,
            priority=1
        ),
        ScoreLevel.B: ProcessingFlow(
            level=ScoreLevel.B,
            recommendation=Recommendation.HIRE,
            next_steps=[
                "安排二面或技术总监面试",
                "进行背景调查",
                "评估薪资期望",
                "1-2个工作日内给出反馈"
            ],
            approval_required=True,
            priority=2
        ),
        ScoreLevel.C: ProcessingFlow(
            level=ScoreLevel.C,
            recommendation=Recommendation.CONSIDER,
            next_steps=[
                "安排补充面试或笔试",
                "评估具体能力短板",
                "考虑培训后录用",
                "3-5个工作日内给出反馈"
            ],
            approval_required=True,
            priority=3
        ),
        ScoreLevel.D: ProcessingFlow(
            level=ScoreLevel.D,
            recommendation=Recommendation.REJECT,
            next_steps=[
                "发送感谢信",
                "提供改进建议",
                "保留简历6个月",
                "加入人才库（低优先级）"
            ],
            approval_required=False,
            priority=4
        )
    }

    def __init__(self):
        logger.info("[OK] 评分等级划分器初始化完成")

    def classify(self, overall_score: float,
                 dimension_scores: Optional[Dict[str, float]] = None,
                 confidence_level: Optional[str] = None) -> Dict[str, any]:
        """
        划分评分等级

        Args:
            overall_score: 综合评分 (0-100)
            dimension_scores: 各维度评分
            confidence_level: 置信度等级

        Returns:
            等级划分结果
        """
        # 确定等级
        level = self._determine_level(overall_score)
        criteria = self.LEVEL_CRITERIA[level]
        flow = self.PROCESSING_FLOWS[level]

        # 分析维度表现
        dimension_analysis = self._analyze_dimensions(dimension_scores)

        # 生成综合建议
        suggestions = self._generate_suggestions(
            level, dimension_analysis, confidence_level
        )

        result = {
            "overall_score": round(overall_score, 1),
            "level": level.value,
            "level_info": {
                "label": criteria.label,
                "color": criteria.color,
                "description": criteria.description,
                "score_range": f"{criteria.min_score}-{criteria.max_score}"
            },
            "recommendation": flow.recommendation.value,
            "recommendation_label": self._get_recommendation_label(flow.recommendation),
            "processing_flow": {
                "next_steps": flow.next_steps,
                "approval_required": flow.approval_required,
                "priority": flow.priority
            },
            "dimension_analysis": dimension_analysis,
            "suggestions": suggestions,
            "confidence_note": self._get_confidence_note(confidence_level)
        }

        logger.info(f"评分等级划分完成: {overall_score:.1f} -> {level.value}级 ({criteria.label})")

        return result

    def _determine_level(self, score: float) -> ScoreLevel:
        """确定评分等级"""
        if score >= 85:
            return ScoreLevel.A
        elif score >= 70:
            return ScoreLevel.B
        elif score >= 60:
            return ScoreLevel.C
        else:
            return ScoreLevel.D

    def _analyze_dimensions(self,
                            dimension_scores: Optional[Dict[str, float]]) -> Dict:
        """分析各维度表现"""
        if not dimension_scores:
            return {}

        analysis = {
            "strengths": [],
            "weaknesses": [],
            "average": 0.0,
            "highest": None,
            "lowest": None
        }

        if not dimension_scores:
            return analysis

        # 计算平均分
        avg = sum(dimension_scores.values()) / len(dimension_scores)
        analysis["average"] = round(avg, 1)

        # 找出最高和最低维度
        sorted_dims = sorted(dimension_scores.items(), key=lambda x: x[1], reverse=True)

        if sorted_dims:
            analysis["highest"] = {
                "dimension": sorted_dims[0][0],
                "score": sorted_dims[0][1]
            }
            analysis["lowest"] = {
                "dimension": sorted_dims[-1][0],
                "score": sorted_dims[-1][1]
            }

        # 识别优势和短板
        for dim, score in dimension_scores.items():
            if score >= 80:
                analysis["strengths"].append({
                    "dimension": dim,
                    "score": score,
                    "level": "优秀"
                })
            elif score < 60:
                analysis["weaknesses"].append({
                    "dimension": dim,
                    "score": score,
                    "level": "待提升"
                })

        return analysis

    def _generate_suggestions(self, level: ScoreLevel,
                              dimension_analysis: Dict,
                              confidence_level: Optional[str]) -> List[str]:
        """生成综合建议"""
        suggestions = []

        # 基于等级的建议
        if level == ScoreLevel.A:
            suggestions.append("候选人表现卓越，建议尽快安排录用")
        elif level == ScoreLevel.B:
            suggestions.append("候选人整体表现良好，建议安排后续面试")
        elif level == ScoreLevel.C:
            suggestions.append("候选人基本达标，建议评估具体能力短板")
        else:
            suggestions.append("候选人未达标，建议婉拒")

        # 基于维度分析的建议
        strengths = dimension_analysis.get("strengths", [])
        weaknesses = dimension_analysis.get("weaknesses", [])

        if strengths:
            dim_names = [s["dimension"] for s in strengths[:2]]
            suggestions.append(f"优势维度: {', '.join(dim_names)}")

        if weaknesses:
            dim_names = [w["dimension"] for w in weaknesses[:2]]
            suggestions.append(f"待提升维度: {', '.join(dim_names)}")

        # 基于置信度的建议
        if confidence_level == "low" or confidence_level == "very_low":
            suggestions.append("置信度较低，建议人工复核评估结果")

        return suggestions

    def _get_recommendation_label(self,
                                   recommendation: Recommendation) -> str:
        """获取建议标签"""
        labels = {
            Recommendation.STRONG_HIRE: "强烈推荐录用",
            Recommendation.HIRE: "推荐录用",
            Recommendation.CONSIDER: "考虑录用",
            Recommendation.REJECT: "不建议录用"
        }
        return labels.get(recommendation, "未知")

    def _get_confidence_note(self,
                              confidence_level: Optional[str]) -> str:
        """获取置信度说明"""
        if confidence_level is None:
            return "未提供置信度信息"

        notes = {
            "high": "评估结果可信度高",
            "medium": "评估结果可信度中等，建议参考",
            "low": "评估结果可信度较低，建议复核",
            "very_low": "评估结果可信度很低，必须人工复核"
        }
        return notes.get(confidence_level, "未知置信度等级")

    def batch_classify(self, evaluations: List[Dict]) -> List[Dict]:
        """批量划分等级"""
        results = []
        for eval_data in evaluations:
            result = self.classify(
                overall_score=eval_data.get("overall_score", 0),
                dimension_scores=eval_data.get("dimension_scores"),
                confidence_level=eval_data.get("confidence_level")
            )
            results.append(result)
        return results

    def get_level_distribution(self, scores: List[float]) -> Dict[str, int]:
        """获取等级分布统计"""
        distribution = {"A": 0, "B": 0, "C": 0, "D": 0}

        for score in scores:
            level = self._determine_level(score)
            distribution[level.value] += 1

        return distribution

    def compare_candidates(self, candidates: List[Dict]) -> List[Dict]:
        """比较多个候选人"""
        # 按分数排序
        sorted_candidates = sorted(
            candidates,
            key=lambda x: x.get("overall_score", 0),
            reverse=True
        )

        results = []
        for i, candidate in enumerate(sorted_candidates):
            classification = self.classify(
                overall_score=candidate.get("overall_score", 0),
                dimension_scores=candidate.get("dimension_scores"),
                confidence_level=candidate.get("confidence_level")
            )

            classification["rank"] = i + 1
            classification["candidate_name"] = candidate.get("name", "未知")

            results.append(classification)

        return results
