"""
锐聘AI - 情绪识别模块

功能:
- 基于声学特征识别说话人的情绪状态
- 支持"紧张"和"自信"两种基础情绪分类
- 提供情绪强度量化指标（0-100分）

特征:
- 基频变化（pitch variability）
- 语速变化（speech rate variability）
- 能量波动（energy variability）
- 停顿比例（pause ratio）
- 声音颤抖（voice tremor / jitter）
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from enum import Enum

from ..logger import logger, log_function_call


class EmotionType(Enum):
    """情绪类型"""
    NERVOUS = "nervous"
    CONFIDENT = "confident"
    NEUTRAL = "neutral"


@dataclass
class EmotionResult:
    """情绪识别结果"""
    primary_emotion: str           # 主要情绪: 'nervous', 'confident', 'neutral'
    confidence: float              # 主要情绪的置信度 (0-100)
    emotion_scores: Dict[str, float]  # 各情绪得分
    features: Dict[str, float]     # 原始特征值
    intensity: float               # 情绪强度 (0-100)


class EmotionRecognizer:
    """
    语音情绪识别器
    
    基于声学特征识别紧张/自信等情绪状态
    """
    
    # 特征权重（用于计算紧张得分）
    FEATURE_WEIGHTS = {
        'pitch_variability': 0.25,       # 基频变化
        'speech_rate_variability': 0.20,  # 语速不稳
        'energy_variability': 0.20,       # 能量波动
        'pause_ratio': 0.15,              # 停顿比例
        'voice_tremor': 0.20              # 声音颤抖
    }
    
    def __init__(self):
        logger.info("[Emotion] 初始化情绪识别器")
    
    @log_function_call()
    def recognize(
        self,
        audio: np.ndarray,
        sr: int = 16000,
        speech_rate_metrics=None,
        pause_metrics=None
    ) -> EmotionResult:
        """
        识别语音情绪
        
        Args:
            audio: 音频数据
            sr: 采样率
            speech_rate_metrics: 语速指标（可选）
            pause_metrics: 停顿指标（可选）
            
        Returns:
            EmotionResult
        """
        total_duration = len(audio) / sr
        
        logger.info("=" * 60)
        logger.info("开始情绪识别")
        logger.info("=" * 60)
        logger.info(f"[Emotion] 输入音频长度: {len(audio)} 样本 | 采样率: {sr}Hz | 总时长: {total_duration:.2f}s")
        if speech_rate_metrics:
            logger.info(f"[Emotion] 语速指标: WCPM={speech_rate_metrics.wcpm:.1f}, 类别={speech_rate_metrics.rate_category}")
        if pause_metrics:
            logger.info(f"[Emotion] 停顿指标: 总停顿={pause_metrics.total_pauses}, 长停顿={pause_metrics.long_pauses}, 占比={pause_metrics.pause_ratio*100:.1f}%")
        
        # 1. 提取声学特征
        logger.info("[Emotion] [STEP 1/5] 提取声学特征...")
        features = self._extract_features(audio, sr)
        logger.info(f"[Emotion] [STEP 1_OK] 提取到 {len(features)} 个声学特征")
        for key, value in features.items():
            logger.info(f"[Emotion]   特征 {key}: {value:.4f}")
        
        # 2. 结合语速和停顿特征
        logger.info("[Emotion] [STEP 2/5] 结合语速和停顿特征...")
        if speech_rate_metrics and speech_rate_metrics.rate_curve:
            features['speech_rate_variability'] = self._calculate_rate_variability(
                speech_rate_metrics.rate_curve
            )
            logger.info(f"[Emotion] [STEP 2_OK] 语速变化度: {features['speech_rate_variability']:.4f}")
        else:
            logger.info("[Emotion] [STEP 2_OK] 无语速曲线数据，跳过语速变化度计算")
        
        if pause_metrics:
            features['pause_ratio'] = pause_metrics.pause_ratio
            features['long_pause_ratio'] = (
                pause_metrics.long_pauses / pause_metrics.total_pauses
                if pause_metrics.total_pauses > 0 else 0
            )
            logger.info(f"[Emotion] [STEP 2_OK] 停顿比例: {features['pause_ratio']:.4f}, 长停顿比例: {features['long_pause_ratio']:.4f}")
        else:
            logger.info("[Emotion] [STEP 2_OK] 无停顿指标数据，跳过停顿特征计算")
        
        # 3. 计算情绪得分
        logger.info("[Emotion] [STEP 3/5] 计算情绪得分...")
        emotion_scores = self._calculate_emotion_scores(features)
        logger.info(f"[Emotion] [STEP 3_OK] 情绪得分: 紧张={emotion_scores['nervous']:.1f}, 自信={emotion_scores['confident']:.1f}, 中性={emotion_scores['neutral']:.1f}")
        
        # 4. 确定主要情绪
        logger.info("[Emotion] [STEP 4/5] 确定主要情绪...")
        primary_emotion = max(emotion_scores, key=emotion_scores.get)
        confidence = emotion_scores[primary_emotion]
        logger.info(f"[Emotion] [STEP 4_OK] 主要情绪: {primary_emotion} (置信度: {confidence:.1f}%)")
        
        # 5. 计算情绪强度
        logger.info("[Emotion] [STEP 5/5] 计算情绪强度...")
        intensity = self._calculate_intensity(features, primary_emotion)
        logger.info(f"[Emotion] [STEP 5_OK] 情绪强度: {intensity:.1f}/100")
        
        logger.info("=" * 60)
        logger.info("情绪识别完成")
        logger.info(f"[Emotion] 最终结果: 情绪={primary_emotion}, 置信度={confidence:.1f}%, 强度={intensity:.1f}")
        logger.info("=" * 60)
        
        return EmotionResult(
            primary_emotion=primary_emotion,
            confidence=confidence,
            emotion_scores=emotion_scores,
            features=features,
            intensity=intensity
        )
    
    def _extract_features(self, audio: np.ndarray, sr: int) -> Dict[str, float]:
        """
        提取声学特征
        
        Returns:
            {
                'pitch_mean': float,
                'pitch_std': float,
                'pitch_variability': float,
                'energy_mean': float,
                'energy_std': float,
                'energy_variability': float,
                'voice_tremor': float,
                'zcr_mean': float
            }
        """
        logger.info("[Emotion] [Features] 开始提取声学特征...")
        features = {}
        
        # 1. 基频 (F0) 及其变化
        logger.info("[Emotion] [Features] [1/4] 提取基频 (F0)...")
        try:
            # 使用自相关法估算基频
            f0_values = self._estimate_f0(audio, sr)
            
            if len(f0_values) > 0:
                features['pitch_mean'] = float(np.mean(f0_values))
                features['pitch_std'] = float(np.std(f0_values))
                features['pitch_variability'] = (
                    features['pitch_std'] / features['pitch_mean']
                    if features['pitch_mean'] > 0 else 0
                )
                logger.info(f"[Emotion] [Features] [1/4_OK] 基频: 均值={features['pitch_mean']:.2f}Hz, 标准差={features['pitch_std']:.2f}Hz, 变异系数={features['pitch_variability']:.4f} (有效帧数: {len(f0_values)})")
            else:
                features['pitch_mean'] = 0.0
                features['pitch_std'] = 0.0
                features['pitch_variability'] = 0.0
                logger.warning("[Emotion] [Features] [1/4_OK] 未检测到有效基频，设为0")
        except Exception as e:
            logger.warning(f"[Emotion] [Features] [1/4_ERR] 基频提取失败: {e}")
            features['pitch_mean'] = 0.0
            features['pitch_std'] = 0.0
            features['pitch_variability'] = 0.0
        
        # 2. 能量特征
        logger.info("[Emotion] [Features] [2/4] 提取能量特征...")
        frame_length = int(0.02 * sr)
        hop_length = int(0.01 * sr)
        
        rms_values = []
        for i in range(0, len(audio) - frame_length, hop_length):
            frame = audio[i:i + frame_length]
            rms = np.sqrt(np.mean(frame ** 2))
            rms_values.append(rms)
        
        if rms_values:
            features['energy_mean'] = float(np.mean(rms_values))
            features['energy_std'] = float(np.std(rms_values))
            features['energy_variability'] = (
                features['energy_std'] / features['energy_mean']
                if features['energy_mean'] > 0 else 0
            )
            logger.info(f"[Emotion] [Features] [2/4_OK] 能量: 均值={features['energy_mean']:.6f}, 标准差={features['energy_std']:.6f}, 变异系数={features['energy_variability']:.4f} (帧数: {len(rms_values)})")
        else:
            features['energy_mean'] = 0.0
            features['energy_std'] = 0.0
            features['energy_variability'] = 0.0
            logger.warning("[Emotion] [Features] [2/4_OK] 未计算出能量值，设为0")
        
        # 3. 声音颤抖（jitter 近似）
        logger.info("[Emotion] [Features] [3/4] 计算声音颤抖 (jitter)...")
        if len(f0_values) > 1:
            jitter = np.mean(np.abs(np.diff(f0_values))) / np.mean(f0_values)
            features['voice_tremor'] = min(float(jitter * 100), 1.0)
            logger.info(f"[Emotion] [Features] [3/4_OK] 声音颤抖: jitter={jitter:.4f}, tremor={features['voice_tremor']:.4f}")
        else:
            features['voice_tremor'] = 0.0
            logger.info("[Emotion] [Features] [3/4_OK] 基频数据不足，声音颤抖设为0")
        
        # 4. 过零率
        logger.info("[Emotion] [Features] [4/4] 计算过零率 (ZCR)...")
        zcr_values = []
        for i in range(0, len(audio) - frame_length, hop_length):
            frame = audio[i:i + frame_length]
            zcr = np.mean(np.abs(np.diff(np.sign(frame)))) / 2
            zcr_values.append(zcr)
        
        features['zcr_mean'] = float(np.mean(zcr_values)) if zcr_values else 0.0
        logger.info(f"[Emotion] [Features] [4/4_OK] 过零率: 均值={features['zcr_mean']:.4f} (帧数: {len(zcr_values)})")
        
        logger.info("[Emotion] [Features] 特征提取完成")
        return features
    
    def _estimate_f0(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        使用自相关法估算基频
        
        Returns:
            基频数组
        """
        logger.info("[Emotion] [F0] 开始基频估算...")
        frame_length = int(0.04 * sr)   # 40ms
        hop_length = int(0.01 * sr)     # 10ms
        
        logger.info(f"[Emotion] [F0] 帧长: {frame_length} 样本 ({frame_length/sr*1000:.1f}ms) | "
                   f"帧移: {hop_length} 样本 ({hop_length/sr*1000:.1f}ms)")
        
        f0_values = []
        valid_frames = 0
        rejected_by_peak = 0
        rejected_by_range = 0
        
        min_lag = int(sr / 500)   # 对应 500Hz
        max_lag = int(sr / 50)    # 对应 50Hz
        logger.info(f"[Emotion] [F0] 滞后范围: {min_lag}-{max_lag} 样本 (对应 50-500Hz)")
        
        for i in range(0, len(audio) - frame_length, hop_length):
            frame = audio[i:i + frame_length]
            
            # 计算自相关
            autocorr = np.correlate(frame, frame, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            
            # 寻找峰值（排除零滞后）
            if len(autocorr) > max_lag:
                peak_lag = min_lag + np.argmax(autocorr[min_lag:max_lag])
                
                # 检查峰值是否显著
                if autocorr[peak_lag] > autocorr[0] * 0.3:
                    f0 = sr / peak_lag
                    if 50 <= f0 <= 500:  # 合理的基频范围
                        f0_values.append(f0)
                        valid_frames += 1
                    else:
                        rejected_by_range += 1
                else:
                    rejected_by_peak += 1
        
        logger.info(f"[Emotion] [F0] 基频估算完成: 有效帧={valid_frames}, 因峰值不显著拒绝={rejected_by_peak}, 因频率范围拒绝={rejected_by_range}")
        logger.info(f"[Emotion] [F0] 基频范围: {min(f0_values):.1f}Hz - {max(f0_values):.1f}Hz (若有效)")
        
        return np.array(f0_values)
    
    def _calculate_rate_variability(self, rate_curve: List[Tuple[float, float]]) -> float:
        """
        计算语速变化程度
        
        Args:
            rate_curve: [(time, wcpm), ...]
            
        Returns:
            变化系数
        """
        if not rate_curve or len(rate_curve) < 2:
            return 0.0
        
        rates = [rate for _, rate in rate_curve]
        mean_rate = np.mean(rates)
        std_rate = np.std(rates)
        
        return float(std_rate / mean_rate) if mean_rate > 0 else 0.0
    
    def _calculate_emotion_scores(self, features: Dict[str, float]) -> Dict[str, float]:
        """
        计算各情绪得分
        
        紧张特征:
        - 基频变化大（pitch_variability 高）
        - 语速不稳定（speech_rate_variability 高）
        - 能量波动大（energy_variability 高）
        - 停顿多（pause_ratio 高）
        - 声音颤抖（voice_tremor 高）
        
        自信特征: 以上特征相反
        
        Returns:
            {'nervous': score, 'confident': score, 'neutral': score}
        """
        logger.info("[Emotion] [Scores] 开始计算情绪得分...")
        logger.info(f"[Emotion] [Scores] 特征权重: {self.FEATURE_WEIGHTS}")
        
        # 归一化特征到 0-1 范围
        normalized = {
            'pitch_variability': min(features.get('pitch_variability', 0) * 5, 1.0),
            'speech_rate_variability': min(features.get('speech_rate_variability', 0) * 3, 1.0),
            'energy_variability': min(features.get('energy_variability', 0) * 3, 1.0),
            'pause_ratio': min(features.get('pause_ratio', 0) * 2, 1.0),
            'voice_tremor': features.get('voice_tremor', 0)
        }
        
        logger.info("[Emotion] [Scores] 归一化特征值:")
        for key, value in normalized.items():
            raw = features.get(key, 0)
            logger.info(f"[Emotion] [Scores]   {key}: 原始={raw:.4f} -> 归一化={value:.4f}")
        
        # 计算紧张得分（特征越高越紧张）
        nervous_score = (
            normalized['pitch_variability'] * self.FEATURE_WEIGHTS['pitch_variability'] +
            normalized['speech_rate_variability'] * self.FEATURE_WEIGHTS['speech_rate_variability'] +
            normalized['energy_variability'] * self.FEATURE_WEIGHTS['energy_variability'] +
            normalized['pause_ratio'] * self.FEATURE_WEIGHTS['pause_ratio'] +
            normalized['voice_tremor'] * self.FEATURE_WEIGHTS['voice_tremor']
        ) * 100
        
        logger.info(f"[Emotion] [Scores] 紧张得分计算: "
                   f"({normalized['pitch_variability']:.3f}*{self.FEATURE_WEIGHTS['pitch_variability']} + "
                   f"{normalized['speech_rate_variability']:.3f}*{self.FEATURE_WEIGHTS['speech_rate_variability']} + "
                   f"{normalized['energy_variability']:.3f}*{self.FEATURE_WEIGHTS['energy_variability']} + "
                   f"{normalized['pause_ratio']:.3f}*{self.FEATURE_WEIGHTS['pause_ratio']} + "
                   f"{normalized['voice_tremor']:.3f}*{self.FEATURE_WEIGHTS['voice_tremor']}) * 100 = {nervous_score:.2f}")
        
        # 自信得分与紧张相反
        confident_score = 100 - nervous_score
        logger.info(f"[Emotion] [Scores] 自信得分: 100 - {nervous_score:.2f} = {confident_score:.2f}")
        
        # 中性得分（中间状态）
        # 当紧张和自信都较低时，偏向中性
        neutral_score = max(0, 100 - abs(nervous_score - 50) * 2)
        logger.info(f"[Emotion] [Scores] 中性得分: max(0, 100 - abs({nervous_score:.2f} - 50) * 2) = {neutral_score:.2f}")
        
        result = {
            'nervous': min(max(nervous_score, 0), 100),
            'confident': min(max(confident_score, 0), 100),
            'neutral': min(max(neutral_score, 0), 100)
        }
        
        logger.info(f"[Emotion] [Scores] 最终得分: 紧张={result['nervous']:.1f}, 自信={result['confident']:.1f}, 中性={result['neutral']:.1f}")
        return result
    
    def _calculate_intensity(self, features: Dict[str, float], primary_emotion: str) -> float:
        """
        计算情绪强度
        
        Args:
            features: 特征值
            primary_emotion: 主要情绪
            
        Returns:
            强度 (0-100)
        """
        logger.info("[Emotion] [Intensity] 开始计算情绪强度...")
        
        # 基于特征的综合强度
        intensity_factors = [
            min(features.get('pitch_variability', 0) * 5, 1.0),
            min(features.get('energy_variability', 0) * 3, 1.0),
            features.get('voice_tremor', 0)
        ]
        
        logger.info(f"[Emotion] [Intensity] 强度因子: pitch_var={intensity_factors[0]:.4f}, energy_var={intensity_factors[1]:.4f}, tremor={intensity_factors[2]:.4f}")
        
        avg_intensity = np.mean(intensity_factors) * 100
        logger.info(f"[Emotion] [Intensity] 平均强度因子: {avg_intensity:.2f}")
        
        if primary_emotion == 'nervous':
            # 紧张强度直接关联
            intensity = min(max(avg_intensity, 0), 100)
            logger.info(f"[Emotion] [Intensity] 紧张情绪: 强度={intensity:.1f} (直接关联波动程度)")
            return intensity
        elif primary_emotion == 'confident':
            # 自信强度与低波动关联
            intensity = min(max(100 - avg_intensity, 0), 100)
            logger.info(f"[Emotion] [Intensity] 自信情绪: 强度={intensity:.1f} (100 - {avg_intensity:.2f})")
            return intensity
        else:
            # 中性居中
            logger.info(f"[Emotion] [Intensity] 中性情绪: 强度=50.0 (固定居中)")
            return 50.0
    
    def get_emotion_feedback(self, result: EmotionResult) -> str:
        """
        获取情绪反馈建议
        
        Args:
            result: 情绪识别结果
            
        Returns:
            反馈文字
        """
        emotion = result.primary_emotion
        confidence = result.confidence
        intensity = result.intensity
        
        if emotion == 'nervous':
            if intensity > 70:
                return (
                    f"情绪明显紧张（强度: {intensity:.0f}/100），"
                    f"建议深呼吸放松，放慢语速，保持眼神交流"
                )
            elif intensity > 40:
                return (
                    f"情绪略显紧张（强度: {intensity:.0f}/100），"
                    f"可以尝试放慢节奏，增强自信"
                )
            else:
                return (
                    f"轻微紧张（强度: {intensity:.0f}/100），"
                    f"整体表现尚可"
                )
        
        elif emotion == 'confident':
            if intensity > 70:
                return (
                    f"表现非常自信（强度: {intensity:.0f}/100），"
                    f"状态良好，继续保持"
                )
            elif intensity > 40:
                return (
                    f"表现自信（强度: {intensity:.0f}/100），"
                    f"状态不错"
                )
            else:
                return (
                    f"较为平静（强度: {intensity:.0f}/100），"
                    f"可以适当增强表达力度"
                )
        
        else:  # neutral
            return (
                f"情绪状态平稳（强度: {intensity:.0f}/100），"
                f"保持自然即可"
            )
    
    def get_emotion_distribution(self, result: EmotionResult) -> Dict[str, Any]:
        """
        获取情绪分布数据（用于图表展示）
        
        Returns:
            {
                "labels": ["紧张", "自信", "中性"],
                "values": [score1, score2, score3]
            }
        """
        return {
            "labels": ["紧张", "自信", "中性"],
            "values": [
                round(result.emotion_scores.get('nervous', 0), 1),
                round(result.emotion_scores.get('confident', 0), 1),
                round(result.emotion_scores.get('neutral', 0), 1)
            ]
        }
