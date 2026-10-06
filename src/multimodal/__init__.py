"""
锐聘AI - 多模态评价整合模块

提供文本一致性校验、声纹分析整合、多模态融合决策功能
"""

from .text_consistency import TextConsistencyChecker
from .voice_integration import VoiceAnalysisIntegration
from .fusion_engine import MultimodalFusionEngine

__all__ = [
    "TextConsistencyChecker",
    "VoiceAnalysisIntegration",
    "MultimodalFusionEngine",
]
