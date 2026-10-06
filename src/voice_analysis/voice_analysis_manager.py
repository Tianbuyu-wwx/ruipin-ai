"""
锐聘AI - 声纹分析管理器

整合语速分析、停顿分析和情绪识别三个模块，
提供统一的声纹分析接口和综合报告生成。
"""

import numpy as np
from typing import Dict, Any, Optional
from pathlib import Path

from ..logger import logger, LogContext, log_function_call

from .speech_rate_analyzer import SpeechRateAnalyzer, SpeechRateMetrics
from .pause_analyzer import PauseAnalyzer, PauseMetrics
from .emotion_recognizer import EmotionRecognizer, EmotionResult


class VoiceAnalysisManager:
    """
    声纹分析管理器
    
    整合三个分析器，提供一站式声纹分析服务
    """
    
    def __init__(self, language: str = "zh"):
        """
        Args:
            language: 语言代码 (zh/en/ja/ko 等)
        """
        logger.info("=" * 60)
        logger.info("初始化声纹分析管理器")
        logger.info("=" * 60)
        
        self.language = language
        
        # 初始化三个分析器
        self.speech_rate_analyzer = SpeechRateAnalyzer(language=language)
        self.pause_analyzer = PauseAnalyzer()
        self.emotion_recognizer = EmotionRecognizer()
        
        logger.info("[VoiceAnalysis] 声纹分析管理器初始化完成")
    
    @log_function_call()
    def analyze(
        self,
        audio: np.ndarray,
        sr: int = 16000,
        transcribed_text: str = ""
    ) -> Dict[str, Any]:
        """
        执行完整的声纹分析
        
        Args:
            audio: 音频数据（numpy 数组，float32，范围 [-1, 1]）
            sr: 采样率
            transcribed_text: 转录文字（用于语速计算）
            
        Returns:
            综合分析报告
            {
                "overall_score": float,           # 综合评分 (0-100)
                "speech_rate": {                  # 语速分析结果
                    "score": float,
                    "wcpm": float,
                    "category": str,
                    "total_words": int,
                    "rate_curve": List[Tuple[float, float]],
                    "feedback": str
                },
                "pause": {                        # 停顿分析结果
                    "score": float,
                    "total_pauses": int,
                    "pause_breakdown": Dict,
                    "average_pause_duration": float,
                    "pause_frequency": float,
                    "feedback": str,
                    "distribution": Dict
                },
                "emotion": {                      # 情绪识别结果
                    "primary_emotion": str,
                    "confidence": float,
                    "intensity": float,
                    "scores": Dict[str, float],
                    "feedback": str,
                    "distribution": Dict
                },
                "summary": str                    # 综合摘要
            }
        """
        total_duration = len(audio) / sr
        
        logger.info("=" * 60)
        logger.info("开始完整声纹分析")
        logger.info("=" * 60)
        logger.info(f"[VoiceAnalysis] 音频长度: {len(audio)} 样本 | "
                   f"采样率: {sr}Hz | 时长: {total_duration:.2f}s | 转录文字: {len(transcribed_text)} 字符")
        logger.info(f"[VoiceAnalysis] 语言设置: {self.language}")
        
        try:
            with LogContext(logger, "声纹分析"):
                # 1. 语速分析
                logger.info("[VoiceAnalysis] [模块 1/3] 开始语速分析...")
                speech_rate = self.speech_rate_analyzer.analyze(audio, sr, transcribed_text)
                logger.info(f"[VoiceAnalysis] [模块 1/3_OK] 语速分析完成: WCPM={speech_rate.wcpm:.1f}, 类别={speech_rate.rate_category}")
                
                # 2. 停顿分析
                logger.info("[VoiceAnalysis] [模块 2/3] 开始停顿分析...")
                pauses = self.pause_analyzer.analyze(audio, sr)
                logger.info(f"[VoiceAnalysis] [模块 2/3_OK] 停顿分析完成: 总停顿={pauses.total_pauses}, 流畅度={pauses.fluency_score:.1f}")
                
                # 3. 情绪识别（依赖语速和停顿结果）
                logger.info("[VoiceAnalysis] [模块 3/3] 开始情绪识别...")
                emotion = self.emotion_recognizer.recognize(
                    audio, sr, speech_rate, pauses
                )
                logger.info(f"[VoiceAnalysis] [模块 3/3_OK] 情绪识别完成: {emotion.primary_emotion} (置信度={emotion.confidence:.1f}%)")
                
                # 4. 生成综合报告
                logger.info("[VoiceAnalysis] 生成综合报告...")
                report = self._generate_report(speech_rate, pauses, emotion)
                
                logger.info("=" * 60)
                logger.info("声纹分析完成")
                logger.info(f"[VoiceAnalysis] 综合评分: {report['overall_score']}")
                logger.info(f"[VoiceAnalysis] 语速评分: {report['speech_rate']['score']} | 流畅度: {report['pause']['score']} | 情绪稳定性: {emotion.emotion_scores.get('confident', 50):.1f}")
                logger.info("=" * 60)
                
                return report
                
        except Exception as e:
            logger.error(f"[VoiceAnalysis] 声纹分析失败: {e}", exc_info=True)
            return self._generate_error_report(str(e))
    
    def analyze_from_file(
        self,
        audio_path: str,
        transcribed_text: str = ""
    ) -> Dict[str, Any]:
        """
        从音频文件执行声纹分析
        
        Args:
            audio_path: 音频文件路径
            transcribed_text: 转录文字
            
        Returns:
            综合分析报告
        """
        import wave
        
        logger.info("=" * 60)
        logger.info("从文件加载音频并分析")
        logger.info("=" * 60)
        logger.info(f"[VoiceAnalysis] 音频文件路径: {audio_path}")
        logger.info(f"[VoiceAnalysis] 转录文字长度: {len(transcribed_text)} 字符")
        
        try:
            with wave.open(audio_path, 'rb') as wav_file:
                sr = wav_file.getframerate()
                n_channels = wav_file.getnchannels()
                n_frames = wav_file.getnframes()
                sample_width = wav_file.getsampwidth()
                
                logger.info(f"[VoiceAnalysis] [File] WAV信息: 采样率={sr}Hz, 通道={n_channels}, 采样位宽={sample_width*8}bit, 帧数={n_frames}")
                
                # 读取音频数据
                raw_data = wav_file.readframes(n_frames)
                audio = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
                
                logger.info(f"[VoiceAnalysis] [File] 原始音频数据: {len(audio)} 样本, 时长={len(audio)/sr:.2f}s")
                
                # 如果是立体声，转换为单声道
                if n_channels == 2:
                    audio = audio.reshape(-1, 2).mean(axis=1)
                    logger.info(f"[VoiceAnalysis] [File] 立体声转单声道: {len(audio)} 样本")
                
                logger.info(f"[VoiceAnalysis] [File] 音频加载完成，开始分析...")
                
                return self.analyze(audio, sr, transcribed_text)
                
        except Exception as e:
            logger.error(f"[VoiceAnalysis] 加载音频文件失败: {e}", exc_info=True)
            return self._generate_error_report(f"音频加载失败: {e}")
    
    def _generate_report(
        self,
        speech_rate: SpeechRateMetrics,
        pauses: PauseMetrics,
        emotion: EmotionResult
    ) -> Dict[str, Any]:
        """
        生成综合报告
        
        Args:
            speech_rate: 语速指标
            pauses: 停顿指标
            emotion: 情绪识别结果
            
        Returns:
            综合报告字典
        """
        logger.info("[VoiceAnalysis] [Report] 开始生成综合报告...")
        
        # 计算各维度评分
        speech_rate_score = self._score_speech_rate(speech_rate.wcpm)
        fluency_score = pauses.fluency_score
        
        # 情绪稳定性评分（自信越高越好）
        emotion_stability_score = emotion.emotion_scores.get('confident', 50)
        
        logger.info(f"[VoiceAnalysis] [Report] 各维度评分: 语速={speech_rate_score:.1f}, 流畅度={fluency_score:.1f}, 情绪稳定性={emotion_stability_score:.1f}")
        
        # 综合评分（加权平均）
        overall_score = (
            speech_rate_score * 0.3 +
            fluency_score * 0.35 +
            emotion_stability_score * 0.35
        )
        logger.info(f"[VoiceAnalysis] [Report] 综合评分计算: {speech_rate_score:.1f}*0.3 + {fluency_score:.1f}*0.35 + {emotion_stability_score:.1f}*0.35 = {overall_score:.1f}")
        
        return {
            "overall_score": round(overall_score, 1),
            "speech_rate": {
                "score": round(speech_rate_score, 1),
                "wcpm": round(speech_rate.wcpm, 1),
                "category": speech_rate.rate_category,
                "total_words": speech_rate.total_words,
                "speech_duration": round(speech_rate.speech_duration, 2),
                "total_duration": round(speech_rate.total_duration, 2),
                "rate_curve": speech_rate.rate_curve,
                "feedback": self.speech_rate_analyzer.get_rate_feedback(speech_rate)
            },
            "pause": {
                "score": round(fluency_score, 1),
                "total_pauses": pauses.total_pauses,
                "pause_breakdown": {
                    "short": pauses.short_pauses,
                    "medium": pauses.medium_pauses,
                    "long": pauses.long_pauses
                },
                "average_pause_duration": round(pauses.average_pause_duration, 2),
                "pause_frequency": round(pauses.pause_frequency, 1),
                "pause_ratio": round(pauses.pause_ratio, 3),
                "feedback": self.pause_analyzer.get_pause_feedback(pauses),
                "distribution": self.pause_analyzer.get_pause_distribution(pauses)
            },
            "emotion": {
                "primary_emotion": emotion.primary_emotion,
                "confidence": round(emotion.confidence, 1),
                "intensity": round(emotion.intensity, 1),
                "scores": {k: round(v, 1) for k, v in emotion.emotion_scores.items()},
                "features": {k: round(v, 3) for k, v in emotion.features.items()},
                "feedback": self.emotion_recognizer.get_emotion_feedback(emotion),
                "distribution": self.emotion_recognizer.get_emotion_distribution(emotion)
            },
            "summary": self._generate_summary(speech_rate, pauses, emotion)
        }
    
    def _score_speech_rate(self, wcpm: float) -> float:
        """
        语速评分
        
        中文最佳语速: 120-200 WCPM
        英文最佳语速: 100-160 WCPM
        
        Args:
            wcpm: 每分钟字数
            
        Returns:
            0-100 的评分
        """
        if self.language == "zh":
            optimal = 160
            min_good = 120
            max_good = 200
        else:
            optimal = 130
            min_good = 100
            max_good = 160
        
        logger.info(f"[VoiceAnalysis] [Score] 语速评分 | 语言={self.language}, WCPM={wcpm:.1f}, 最佳范围=[{min_good}, {max_good}], 最优={optimal}")
        
        if min_good <= wcpm <= max_good:
            # 在最佳范围内
            score = 100 - abs(wcpm - optimal) * 0.5
            logger.info(f"[VoiceAnalysis] [Score] 语速在最佳范围内: 100 - abs({wcpm:.1f} - {optimal}) * 0.5 = {score:.1f}")
            return score
        elif wcpm < min_good:
            # 偏慢
            score = max(60, 100 - (min_good - wcpm) * 1.5)
            logger.info(f"[VoiceAnalysis] [Score] 语速偏慢: max(60, 100 - ({min_good} - {wcpm:.1f}) * 1.5) = {score:.1f}")
            return score
        else:
            # 偏快
            score = max(60, 100 - (wcpm - max_good) * 1.5)
            logger.info(f"[VoiceAnalysis] [Score] 语速偏快: max(60, 100 - ({wcpm:.1f} - {max_good}) * 1.5) = {score:.1f}")
            return score
    
    def _generate_summary(
        self,
        speech_rate: SpeechRateMetrics,
        pauses: PauseMetrics,
        emotion: EmotionResult
    ) -> str:
        """
        生成综合摘要
        
        Args:
            speech_rate: 语速指标
            pauses: 停顿指标
            emotion: 情绪识别结果
            
        Returns:
            摘要文字
        """
        parts = []
        
        # 语速摘要
        if speech_rate.rate_category == 'fast':
            parts.append(f"语速偏快（{speech_rate.wcpm:.0f} 字/分钟），建议适当放慢节奏")
        elif speech_rate.rate_category == 'slow':
            parts.append(f"语速偏慢（{speech_rate.wcpm:.0f} 字/分钟），可以适当加快")
        else:
            parts.append(f"语速适中（{speech_rate.wcpm:.0f} 字/分钟）")
        
        # 停顿摘要
        if pauses.long_pauses >= 3:
            parts.append(f"出现 {pauses.long_pauses} 次较长停顿，可能影响表达流畅度")
        elif pauses.total_pauses > 15:
            parts.append(f"停顿较多（{pauses.total_pauses} 次），注意控制节奏")
        
        # 情绪摘要
        if emotion.primary_emotion == 'nervous':
            parts.append(f"情绪偏紧张（强度: {emotion.intensity:.0f}/100），建议放松心态")
        elif emotion.primary_emotion == 'confident':
            parts.append(f"表现自信（强度: {emotion.intensity:.0f}/100），状态良好")
        
        # 流畅度总评
        if pauses.fluency_score >= 85:
            parts.append("整体表达流畅")
        elif pauses.fluency_score >= 70:
            parts.append("表达基本流畅，仍有提升空间")
        else:
            parts.append("表达流畅度有待提升")
        
        return "；".join(parts) if parts else "声纹特征正常"
    
    def _generate_error_report(self, error_message: str) -> Dict[str, Any]:
        """
        生成错误报告
        
        Args:
            error_message: 错误信息
            
        Returns:
            错误报告
        """
        return {
            "overall_score": 0,
            "speech_rate": {
                "score": 0,
                "wcpm": 0,
                "category": "unknown",
                "total_words": 0,
                "speech_duration": 0,
                "total_duration": 0,
                "rate_curve": [],
                "feedback": f"分析失败: {error_message}"
            },
            "pause": {
                "score": 0,
                "total_pauses": 0,
                "pause_breakdown": {"short": 0, "medium": 0, "long": 0},
                "average_pause_duration": 0,
                "pause_frequency": 0,
                "pause_ratio": 0,
                "feedback": f"分析失败: {error_message}",
                "distribution": {}
            },
            "emotion": {
                "primary_emotion": "unknown",
                "confidence": 0,
                "intensity": 0,
                "scores": {"nervous": 0, "confident": 0, "neutral": 0},
                "features": {},
                "feedback": f"分析失败: {error_message}",
                "distribution": {}
            },
            "summary": f"声纹分析失败: {error_message}",
            "error": error_message
        }
    
    def get_overall_assessment(self, report: Dict[str, Any]) -> str:
        """
        获取整体评估文字
        
        Args:
            report: 分析报告
            
        Returns:
            评估文字
        """
        score = report["overall_score"]
        
        if score >= 85:
            return "优秀"
        elif score >= 75:
            return "良好"
        elif score >= 60:
            return "合格"
        else:
            return "待提升"
