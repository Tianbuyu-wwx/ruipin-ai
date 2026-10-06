"""
锐聘AI - 声纹分析模块

提供语音的多维度分析能力:
1. 语速分析 - WCPM计算、语速变化曲线
2. 停顿分析 - 短/中/长停顿检测与分类
3. 情绪识别 - 紧张/自信等情绪状态识别

用法:
    from src.voice_analysis import VoiceAnalysisManager
    
    manager = VoiceAnalysisManager()
    report = manager.analyze(audio_array, sample_rate, transcribed_text)
"""

from .speech_rate_analyzer import SpeechRateAnalyzer, SpeechRateMetrics
from .pause_analyzer import PauseAnalyzer, PauseMetrics, PauseType
from .emotion_recognizer import EmotionRecognizer, EmotionResult
from .voice_analysis_manager import VoiceAnalysisManager

__all__ = [
    'SpeechRateAnalyzer',
    'SpeechRateMetrics',
    'PauseAnalyzer',
    'PauseMetrics',
    'PauseType',
    'EmotionRecognizer',
    'EmotionResult',
    'VoiceAnalysisManager',
]
