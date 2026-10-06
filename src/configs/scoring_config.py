"""
评分规则配置
支持动态加载和岗位定制
"""

from typing import Dict, List
from dataclasses import dataclass, field


@dataclass
class DimensionScoringConfig:
    """单个维度的评分配置"""
    base_score: float = 50.0
    max_score: float = 100.0
    min_score: float = 0.0

    # 关键词加分配置: {关键词: 加分值}
    keywords: Dict[str, float] = field(default_factory=dict)
    # 是否只匹配一次（True=只加一次分，False=可多次加分）
    keyword_match_once: bool = True

    # 回答长度加分阈值
    length_bonus_threshold: int = 100
    length_bonus: float = 5.0

    # 音视频加分
    audio_bonus: float = 3.0
    video_bonus: float = 3.0

    # 空回答默认分（无音视频）
    empty_score_no_media: float = 30.0
    # 空回答默认分（有音视频）
    empty_score_with_media: float = 55.0


# 默认评分配置
DEFAULT_SCORING_CONFIG = {
    "technical": DimensionScoringConfig(
        base_score=50.0,
        keywords={
            "架构": 8.0, "设计": 8.0, "优化": 8.0, "性能": 8.0,
            "并发": 8.0, "分布式": 8.0, "微服务": 8.0, "高可用": 8.0,
            "原理": 6.0, "机制": 6.0, "源码": 6.0, "底层": 6.0,
            "经验": 5.0, "实践": 5.0, "项目": 5.0, "实际": 5.0,
        },
        keyword_match_once=False,
        length_bonus_threshold=80,
        length_bonus=5.0,
    ),
    "communication": DimensionScoringConfig(
        base_score=55.0,
        keywords={
            "清晰": 8.0, "逻辑": 8.0, "结构": 8.0, "条理": 8.0,
            "表达": 6.0, "阐述": 6.0, "说明": 6.0, "解释": 6.0,
            "因为": 5.0, "所以": 5.0, "首先": 5.0, "然后": 5.0,
        },
        keyword_match_once=False,
        length_bonus_threshold=100,
        length_bonus=5.0,
    ),
    "completeness": DimensionScoringConfig(
        base_score=50.0,
        keywords={
            "背景": 8.0, "目标": 8.0, "实现": 8.0, "结果": 8.0,
            "挑战": 6.0, "解决": 6.0, "成果": 6.0, "数据": 6.0,
        },
        keyword_match_once=False,
        length_bonus_threshold=120,
        length_bonus=5.0,
    ),
    "problem_solving": DimensionScoringConfig(
        base_score=45.0,
        keywords={
            "分析": 8.0, "排查": 8.0, "定位": 8.0, "诊断": 8.0, "调研": 8.0,
            "解决": 8.0, "方案": 8.0, "优化": 8.0, "改进": 8.0, "修复": 8.0,
            "步骤": 10.0, "流程": 10.0, "首先": 10.0,
            "验证": 5.0, "测试": 5.0, "确认": 5.0, "效果": 5.0, "结果": 5.0,
            "根本原因": 8.0, "底层": 8.0, "原理": 8.0, "机制": 8.0, "架构": 8.0,
        },
        keyword_match_once=True,
        length_bonus_threshold=100,
        length_bonus=5.0,
    ),
    "teamwork": DimensionScoringConfig(
        base_score=45.0,
        keywords={
            "团队": 10.0, "合作": 10.0, "协作": 10.0, "沟通": 10.0, "配合": 10.0, "同事": 10.0,
            "协调": 10.0, "分工": 10.0,
            "冲突": 8.0, "分歧": 8.0, "协商": 8.0, "调解": 8.0, "共识": 8.0,
            "分享": 5.0, "帮助": 5.0, "支持": 5.0, "指导": 5.0, "学习": 5.0,
        },
        keyword_match_once=True,
        length_bonus_threshold=80,
        length_bonus=5.0,
    ),
    "leadership": DimensionScoringConfig(
        base_score=40.0,
        keywords={
            "负责": 12.0, "主导": 12.0, "带领": 12.0, "组织": 12.0, "统筹": 12.0,
            "决策": 8.0, "决定": 8.0, "选择": 8.0, "评估": 8.0, "权衡": 8.0,
            "培养": 8.0, "激励": 8.0, "成长": 8.0, "提升": 8.0, "发展": 8.0,
        },
        keyword_match_once=True,
        length_bonus_threshold=80,
        length_bonus=5.0,
    ),
}

# 岗位定制评分配置
POSITION_SCORING_OVERRIDES = {
    "algorithm_engineer": {
        "technical": DimensionScoringConfig(
            base_score=55.0,
            keywords={
                "模型": 8.0, "算法": 8.0, "训练": 8.0, "调优": 8.0,
                "特征": 8.0, "损失函数": 8.0, "梯度": 8.0, "过拟合": 8.0,
                "准确率": 6.0, "召回率": 6.0, "F1": 6.0, "AUC": 6.0,
                "TensorFlow": 5.0, "PyTorch": 5.0, "sklearn": 5.0,
            },
            keyword_match_once=False,
        ),
    },
    "devops_engineer": {
        "technical": DimensionScoringConfig(
            base_score=50.0,
            keywords={
                "Docker": 8.0, "Kubernetes": 8.0, "K8s": 8.0, "容器": 8.0,
                "CI/CD": 8.0, "Jenkins": 8.0, "GitLab": 8.0, "流水线": 8.0,
                "监控": 6.0, "Prometheus": 6.0, "Grafana": 6.0, "日志": 6.0,
                "自动化": 6.0, "脚本": 6.0, "Shell": 6.0, "Python": 6.0,
            },
            keyword_match_once=False,
        ),
    },
}


def get_scoring_config(position: str = "", dimension: str = "") -> DimensionScoringConfig:
    """
    获取评分配置
    
    Args:
        position: 岗位类型，为空则使用默认配置
        dimension: 维度名称
    
    Returns:
        DimensionScoringConfig 配置对象
    """
    # 获取岗位定制配置
    position_config = POSITION_SCORING_OVERRIDES.get(position, {})
    
    # 获取维度配置（优先使用岗位定制，否则使用默认）
    if dimension in position_config:
        return position_config[dimension]
    elif dimension in DEFAULT_SCORING_CONFIG:
        return DEFAULT_SCORING_CONFIG[dimension]
    else:
        # 返回默认配置
        return DimensionScoringConfig()


def calculate_dimension_score(
    answer: str,
    dimension: str,
    has_audio: bool = False,
    has_video: bool = False,
    position: str = ""
) -> float:
    """
    基于配置计算维度评分
    
    Args:
        answer: 候选人回答
        dimension: 评分维度
        has_audio: 是否有音频
        has_video: 是否有视频
        position: 岗位类型
    
    Returns:
        评分 (0-100)
    """
    config = get_scoring_config(position, dimension)
    
    # 空回答处理
    if not answer or not answer.strip():
        if has_audio or has_video:
            return config.empty_score_with_media
        return config.empty_score_no_media
    
    # 基础分
    score = config.base_score
    
    # 关键词加分
    if config.keyword_match_once:
        # 只匹配一次，取最高分
        max_bonus = 0.0
        for keyword, bonus in config.keywords.items():
            if keyword in answer:
                max_bonus = max(max_bonus, bonus)
        score += max_bonus
    else:
        # 可多次加分，但有上限
        for keyword, bonus in config.keywords.items():
            if keyword in answer:
                score += bonus
    
    # 回答长度加分
    if len(answer) >= config.length_bonus_threshold:
        score += config.length_bonus
    
    # 音视频加分
    if has_audio:
        score += config.audio_bonus
    if has_video:
        score += config.video_bonus
    
    # 限制在有效范围内
    return min(config.max_score, max(config.min_score, score))
