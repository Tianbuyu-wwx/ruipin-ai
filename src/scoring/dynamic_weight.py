"""
锐聘AI - 动态权重调整器

根据岗位类型、经验等级动态调整评分维度权重
支持声纹数据缺失时的权重重分配
"""

from typing import Dict, Optional
from dataclasses import dataclass

from ..logger import logger


@dataclass
class WeightConfig:
    """权重配置"""
    technical: float = 0.25
    communication: float = 0.20
    completeness: float = 0.15
    problem_solving: float = 0.15
    teamwork: float = 0.10
    leadership: float = 0.10
    voice_profile: float = 0.05


class DynamicWeightAdjuster:
    """动态权重调整器"""

    # 基础权重模板
    BASE_WEIGHTS = {
        "technical": 0.25,
        "communication": 0.20,
        "completeness": 0.15,
        "problem_solving": 0.15,
        "teamwork": 0.10,
        "leadership": 0.10,
        "voice_profile": 0.05
    }

    # 岗位类型权重调整
    POSITION_ADJUSTMENTS = {
        "backend_engineer": {
            "technical": +0.10,
            "problem_solving": +0.05,
            "communication": -0.05,
            "teamwork": -0.05,
            "voice_profile": 0.00
        },
        "frontend_engineer": {
            "technical": +0.05,
            "communication": +0.05,
            "problem_solving": 0.00,
            "teamwork": -0.05,
            "voice_profile": -0.05
        },
        "algorithm_engineer": {
            "technical": +0.15,
            "problem_solving": +0.10,
            "communication": -0.10,
            "teamwork": -0.10,
            "voice_profile": -0.05
        },
        "product_manager": {
            "technical": -0.10,
            "communication": +0.10,
            "problem_solving": +0.05,
            "teamwork": +0.05,
            "leadership": +0.05,
            "voice_profile": -0.05
        },
        "devops_engineer": {
            "technical": +0.05,
            "problem_solving": +0.10,
            "communication": -0.05,
            "teamwork": -0.05,
            "voice_profile": -0.05
        },
        "tech_lead": {
            "technical": +0.05,
            "communication": +0.05,
            "problem_solving": +0.05,
            "leadership": +0.10,
            "teamwork": -0.05,
            "voice_profile": -0.10
        }
    }

    # 经验等级权重调整
    EXPERIENCE_ADJUSTMENTS = {
        "junior": {        # 0-2年
            "technical": -0.05,
            "completeness": +0.05,
            "problem_solving": -0.05,
            "leadership": -0.05,
            "voice_profile": +0.02
        },
        "mid": {           # 3-5年
            "technical": 0.00,
            "problem_solving": +0.05,
            "teamwork": +0.02,
            "leadership": 0.00,
            "voice_profile": -0.02
        },
        "senior": {        # 6-8年
            "technical": +0.02,
            "problem_solving": +0.05,
            "leadership": +0.05,
            "teamwork": +0.02,
            "voice_profile": -0.04
        },
        "expert": {        # 9年以上
            "technical": +0.05,
            "problem_solving": +0.10,
            "leadership": +0.10,
            "communication": +0.02,
            "voice_profile": -0.07
        }
    }

    def __init__(self):
        logger.info("[OK] 动态权重调整器初始化完成")

    def calculate_weights(self, position: str, experience: str,
                         has_voice_data: bool = False) -> Dict[str, float]:
        """
        计算动态权重

        Args:
            position: 岗位类型代码
            experience: 经验等级 (junior/mid/senior/expert)
            has_voice_data: 是否有声纹数据

        Returns:
            归一化后的权重字典
        """
        weights = self.BASE_WEIGHTS.copy()

        # 应用岗位调整
        if position in self.POSITION_ADJUSTMENTS:
            for dim, delta in self.POSITION_ADJUSTMENTS[position].items():
                weights[dim] += delta
                logger.debug(f"岗位调整: {dim} += {delta}")

        # 应用经验调整
        if experience in self.EXPERIENCE_ADJUSTMENTS:
            for dim, delta in self.EXPERIENCE_ADJUSTMENTS[experience].items():
                weights[dim] += delta
                logger.debug(f"经验调整: {dim} += {delta}")

        # 无声纹数据时重新分配权重
        if not has_voice_data:
            voice_weight = weights.pop("voice_profile")
            total_other = sum(weights.values())
            if total_other > 0:
                for dim in weights:
                    weights[dim] += voice_weight * (weights[dim] / total_other)
            logger.info(f"无声纹数据，权重重新分配: voice_profile({voice_weight:.2f}) -> 其他维度")

        # 归一化
        total = sum(weights.values())
        normalized = {k: round(v / total, 4) for k, v in weights.items()}

        logger.info(f"动态权重计算完成: {normalized}")
        return normalized

    def get_weight_explanation(self, position: str, experience: str,
                               has_voice_data: bool = False) -> Dict[str, str]:
        """获取权重调整说明"""
        explanations = {}

        if position in self.POSITION_ADJUSTMENTS:
            pos_name = {
                "backend_engineer": "后端工程师",
                "frontend_engineer": "前端工程师",
                "algorithm_engineer": "算法工程师",
                "product_manager": "产品经理",
                "devops_engineer": "运维工程师",
                "tech_lead": "技术主管"
            }.get(position, position)
            explanations["position"] = f"岗位类型: {pos_name}"

        if experience in self.EXPERIENCE_ADJUSTMENTS:
            exp_name = {
                "junior": "初级(0-2年)",
                "mid": "中级(3-5年)",
                "senior": "高级(6-8年)",
                "expert": "专家(9年+)"
            }.get(experience, experience)
            explanations["experience"] = f"经验等级: {exp_name}"

        if not has_voice_data:
            explanations["voice"] = "无声纹数据，voice_profile权重分配至其他维度"

        return explanations

    def validate_weights(self, weights: Dict[str, float]) -> bool:
        """验证权重是否合法"""
        if not weights:
            return False

        # 检查所有维度
        required_dims = set(self.BASE_WEIGHTS.keys())
        if not has_voice_data:
            required_dims.discard("voice_profile")

        if not required_dims.issubset(set(weights.keys())):
            return False

        # 检查权重和是否为1.0（允许小误差）
        total = sum(weights.values())
        return abs(total - 1.0) < 0.01


# 兼容旧接口
has_voice_data = False
