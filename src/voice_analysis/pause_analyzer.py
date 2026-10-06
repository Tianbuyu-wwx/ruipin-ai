"""
锐聘AI - 停顿分析模块

功能:
- 检测并分类不同时长的停顿
- 短停顿 (<0.5秒): 正常呼吸、思考
- 中停顿 (0.5-2秒): 较明显的思考或犹豫
- 长停顿 (>2秒): 明显的卡壳、紧张或组织语言

输出:
- 停顿统计信息
- 停顿分布
- 流畅度评分
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

from ..logger import logger, log_function_call


class PauseType(Enum):
    """停顿类型"""
    SHORT = "short"      # < 0.5s - 正常呼吸/思考
    MEDIUM = "medium"    # 0.5s - 2s - 较明显的思考
    LONG = "long"        # > 2s - 明显的卡壳/紧张


@dataclass
class Pause:
    """单个停顿记录"""
    start_time: float      # 停顿开始时间（秒）
    end_time: float        # 停顿结束时间（秒）
    duration: float        # 停顿时长（秒）
    type: PauseType        # 停顿类型
    preceding_words: int = 0   # 停顿前说了多少字
    following_words: int = 0   # 停顿后说了多少字


@dataclass
class PauseMetrics:
    """停顿指标"""
    total_pauses: int                  # 总停顿数
    short_pauses: int                  # 短停顿数
    medium_pauses: int                 # 中停顿数
    long_pauses: int                   # 长停顿数
    total_pause_time: float            # 总停顿时长（秒）
    average_pause_duration: float      # 平均停顿时长（秒）
    pause_frequency: float             # 每分钟停顿次数
    pause_ratio: float                 # 停顿时间占比
    pauses: List[Pause]                # 停顿详情列表
    
    # 流畅度相关
    fluency_score: float = 0.0         # 流畅度评分 (0-100)
    

class PauseAnalyzer:
    """
    停顿分析器
    
    通过检测语音和非语音段，分析停顿模式
    """
    
    # 停顿时长阈值（秒）
    THRESHOLDS = {
        PauseType.SHORT: (0.0, 0.5),
        PauseType.MEDIUM: (0.5, 2.0),
        PauseType.LONG: (2.0, float('inf'))
    }
    
    # 流畅度评分参数
    FLUENCY_PARAMS = {
        'ideal_short_ratio': 0.7,      # 理想短停顿比例
        'ideal_medium_ratio': 0.25,    # 理想中停顿比例
        'max_acceptable_long': 3,      # 可接受的最大长停顿数
        'ideal_pause_frequency': 10    # 理想每分钟停顿次数
    }
    
    def __init__(
        self,
        min_pause_duration: float = 0.15,   # 最小停顿时长（秒）
        energy_threshold_ratio: float = 0.3  # 能量阈值比例
    ):
        """
        Args:
            min_pause_duration: 小于此值的停顿不视为有效停顿
            energy_threshold_ratio: 能量阈值相对于平均能量的比例
        """
        self.min_pause_duration = min_pause_duration
        self.energy_threshold_ratio = energy_threshold_ratio
        
        logger.info(f"[Pause] 初始化停顿分析器 | "
                   f"最小停顿: {min_pause_duration}s | "
                   f"能量阈值比例: {energy_threshold_ratio}")
    
    @log_function_call()
    def analyze(
        self,
        audio: np.ndarray,
        sr: int = 16000,
        word_timings: Optional[List[Tuple[float, float]]] = None
    ) -> PauseMetrics:
        """
        分析停顿模式
        
        Args:
            audio: 音频数据
            sr: 采样率
            word_timings: 每个字/词的时间范围 [(start, end), ...]
            
        Returns:
            PauseMetrics
        """
        total_duration = len(audio) / sr
        
        logger.info("=" * 60)
        logger.info("开始停顿分析")
        logger.info("=" * 60)
        logger.info(f"[Pause] 输入音频长度: {len(audio)} 样本 | 采样率: {sr}Hz | 总时长: {total_duration:.2f}s")
        if word_timings:
            logger.info(f"[Pause] 输入字词时间戳: {len(word_timings)} 个")
        
        # 1. 检测语音段
        logger.info("[Pause] [STEP 1/5] 检测语音活动段...")
        speech_segments = self._detect_speech_segments(audio, sr)
        logger.info(f"[Pause] [STEP 1_OK] 检测到 {len(speech_segments)} 个语音段")
        for i, (start, end) in enumerate(speech_segments):
            logger.info(f"[Pause]   语音段 {i+1}: {start:.2f}s - {end:.2f}s (时长: {end-start:.2f}s)")
        
        # 2. 从语音段之间提取停顿
        logger.info("[Pause] [STEP 2/5] 提取停顿...")
        pauses = self._extract_pauses(speech_segments, total_duration)
        logger.info(f"[Pause] [STEP 2_OK] 提取到 {len(pauses)} 个潜在停顿")
        for i, (start, end) in enumerate(pauses):
            logger.info(f"[Pause]   潜在停顿 {i+1}: {start:.2f}s - {end:.2f}s (时长: {end-start:.2f}s)")
        
        # 3. 分类停顿
        logger.info("[Pause] [STEP 3/5] 分类停顿...")
        classified_pauses = self._classify_pauses(pauses)
        short = sum(1 for p in classified_pauses if p.type == PauseType.SHORT)
        medium = sum(1 for p in classified_pauses if p.type == PauseType.MEDIUM)
        long = sum(1 for p in classified_pauses if p.type == PauseType.LONG)
        logger.info(f"[Pause] [STEP 3_OK] 分类完成: 短={short}, 中={medium}, 长={long}")
        for pause in classified_pauses:
            logger.info(f"[Pause]   {pause.type.value}停顿: {pause.start_time:.2f}s - {pause.end_time:.2f}s (时长: {pause.duration:.2f}s)")
        
        # 4. 计算指标
        logger.info("[Pause] [STEP 4/5] 计算停顿指标...")
        metrics = self._calculate_metrics(classified_pauses, total_duration)
        logger.info(f"[Pause] [STEP 4_OK] 停顿指标: 总停顿={metrics.total_pauses}, 总时长={metrics.total_pause_time:.2f}s, 频率={metrics.pause_frequency:.1f}次/分钟, 占比={metrics.pause_ratio*100:.1f}%")
        
        # 5. 计算流畅度评分
        logger.info("[Pause] [STEP 5/5] 计算流畅度评分...")
        metrics.fluency_score = self._calculate_fluency_score(metrics)
        logger.info(f"[Pause] [STEP 5_OK] 流畅度评分: {metrics.fluency_score:.1f}")
        
        logger.info("=" * 60)
        logger.info("停顿分析完成")
        logger.info("=" * 60)
        
        return metrics
    
    def _detect_speech_segments(
        self,
        audio: np.ndarray,
        sr: int
    ) -> List[Tuple[float, float]]:
        """
        检测语音活动段
        
        Returns:
            [(start_time, end_time), ...]
        """
        logger.info("[Pause] [VAD] 开始语音活动检测...")
        logger.info(f"[Pause] [VAD] 音频长度: {len(audio)} 样本 | 采样率: {sr}Hz")
        
        frame_length = int(0.02 * sr)   # 20ms 帧
        hop_length = int(0.01 * sr)     # 10ms 步长
        
        logger.info(f"[Pause] [VAD] 帧长: {frame_length} 样本 ({frame_length/sr*1000:.1f}ms) | "
                   f"帧移: {hop_length} 样本 ({hop_length/sr*1000:.1f}ms)")
        
        # 计算每帧能量
        logger.info("[Pause] [VAD] 计算每帧能量...")
        energies = []
        for i in range(0, len(audio) - frame_length, hop_length):
            frame = audio[i:i + frame_length]
            energy = np.sqrt(np.mean(frame ** 2))
            energies.append(energy)
        
        logger.info(f"[Pause] [VAD] 总帧数: {len(energies)}")
        
        if not energies:
            logger.warning("[Pause] [VAD] 未计算出任何帧能量，返回空语音段")
            return []
        
        # 统计能量信息
        energy_array = np.array(energies)
        logger.info(f"[Pause] [VAD] 能量统计: 均值={np.mean(energy_array):.6f}, "
                   f"标准差={np.std(energy_array):.6f}, 最大={np.max(energy_array):.6f}, "
                   f"最小={np.min(energy_array):.6f}")
        
        # 自适应阈值
        mean_energy = np.mean(energies)
        threshold = max(0.005, mean_energy * self.energy_threshold_ratio)
        logger.info(f"[Pause] [VAD] 能量阈值: {threshold:.6f} (max(0.005, 均值*{self.energy_threshold_ratio}))")
        
        # 检测语音段
        logger.info("[Pause] [VAD] 开始检测语音段...")
        segments = []
        in_speech = False
        speech_start = 0
        
        for i, energy in enumerate(energies):
            time_pos = i * hop_length / sr
            
            if energy > threshold and not in_speech:
                in_speech = True
                speech_start = time_pos
                logger.debug(f"[Pause] [VAD] 语音开始 @ {time_pos:.2f}s (能量={energy:.6f})")
            elif energy < threshold * 0.5 and in_speech:
                in_speech = False
                duration = time_pos - speech_start
                if duration >= 0.1:  # 至少 100ms
                    segments.append((speech_start, time_pos))
                    logger.debug(f"[Pause] [VAD] 语音结束 @ {time_pos:.2f}s, 时长={duration:.2f}s")
                else:
                    logger.debug(f"[Pause] [VAD] 忽略过短语音段 @ {speech_start:.2f}s-{time_pos:.2f}s (时长={duration:.2f}s < 0.1s)")
        
        # 处理结尾
        if in_speech:
            final_time = len(audio) / sr
            duration = final_time - speech_start
            if duration >= 0.1:
                segments.append((speech_start, final_time))
                logger.debug(f"[Pause] [VAD] 结尾语音段 {speech_start:.2f}s - {final_time:.2f}s")
        
        logger.info(f"[Pause] [VAD] 检测完成，共 {len(segments)} 个语音段")
        return segments
    
    def _extract_pauses(
        self,
        speech_segments: List[Tuple[float, float]],
        total_duration: float
    ) -> List[Tuple[float, float]]:
        """
        从语音段之间提取停顿
        
        Returns:
            [(start_time, end_time), ...]
        """
        logger.info(f"[Pause] [Extract] 开始提取停顿 | 语音段数: {len(speech_segments)} | 总时长: {total_duration:.2f}s | 最小停顿时长: {self.min_pause_duration}s")
        pauses = []
        
        # 开头停顿（从0到第一个语音段）
        if speech_segments and speech_segments[0][0] > self.min_pause_duration:
            start, end = 0.0, speech_segments[0][0]
            pauses.append((start, end))
            logger.info(f"[Pause] [Extract] 开头停顿: {start:.2f}s - {end:.2f}s (时长: {end-start:.2f}s)")
        elif speech_segments and speech_segments[0][0] > 0:
            logger.info(f"[Pause] [Extract] 开头静音 {speech_segments[0][0]:.2f}s，小于最小停顿时长，忽略")
        
        # 段间停顿
        for i in range(len(speech_segments) - 1):
            pause_start = speech_segments[i][1]
            pause_end = speech_segments[i + 1][0]
            duration = pause_end - pause_start
            
            if duration >= self.min_pause_duration:
                pauses.append((pause_start, pause_end))
                logger.info(f"[Pause] [Extract] 段间停顿 {i+1}: {pause_start:.2f}s - {pause_end:.2f}s (时长: {duration:.2f}s)")
            else:
                logger.debug(f"[Pause] [Extract] 段间间隙 {i+1}: {pause_start:.2f}s - {pause_end:.2f}s (时长: {duration:.2f}s < {self.min_pause_duration}s，忽略)")
        
        # 结尾停顿
        if speech_segments:
            tail_silence = total_duration - speech_segments[-1][1]
            if tail_silence > self.min_pause_duration:
                start, end = speech_segments[-1][1], total_duration
                pauses.append((start, end))
                logger.info(f"[Pause] [Extract] 结尾停顿: {start:.2f}s - {end:.2f}s (时长: {end-start:.2f}s)")
            elif tail_silence > 0:
                logger.info(f"[Pause] [Extract] 结尾静音 {tail_silence:.2f}s，小于最小停顿时长，忽略")
        
        logger.info(f"[Pause] [Extract] 提取完成，共 {len(pauses)} 个有效停顿")
        return pauses
    
    def _classify_pauses(self, pauses: List[Tuple[float, float]]) -> List[Pause]:
        """
        分类停顿
        
        Args:
            pauses: [(start, end), ...]
            
        Returns:
            List[Pause]
        """
        logger.info(f"[Pause] [Classify] 开始分类 {len(pauses)} 个停顿...")
        logger.info(f"[Pause] [Classify] 分类阈值: 短停顿 < {self.THRESHOLDS[PauseType.SHORT][1]}s | "
                   f"中停顿 {self.THRESHOLDS[PauseType.SHORT][1]}-{self.THRESHOLDS[PauseType.MEDIUM][1]}s | "
                   f"长停顿 > {self.THRESHOLDS[PauseType.MEDIUM][1]}s")
        
        classified = []
        
        for start, end in pauses:
            duration = end - start
            
            # 确定类型
            if duration < self.THRESHOLDS[PauseType.SHORT][1]:
                pause_type = PauseType.SHORT
            elif duration < self.THRESHOLDS[PauseType.MEDIUM][1]:
                pause_type = PauseType.MEDIUM
            else:
                pause_type = PauseType.LONG
            
            classified.append(Pause(
                start_time=start,
                end_time=end,
                duration=duration,
                type=pause_type
            ))
            logger.info(f"[Pause] [Classify] 停顿 {start:.2f}s-{end:.2f}s (时长{duration:.2f}s) -> {pause_type.value}")
        
        # 统计
        short = sum(1 for p in classified if p.type == PauseType.SHORT)
        medium = sum(1 for p in classified if p.type == PauseType.MEDIUM)
        long = sum(1 for p in classified if p.type == PauseType.LONG)
        logger.info(f"[Pause] [Classify] 分类完成: 短={short}, 中={medium}, 长={long}")
        
        return classified
    
    def _calculate_metrics(
        self,
        pauses: List[Pause],
        total_duration: float
    ) -> PauseMetrics:
        """
        计算停顿指标
        
        Args:
            pauses: 停顿列表
            total_duration: 总时长（秒）
            
        Returns:
            PauseMetrics
        """
        logger.info(f"[Pause] [Metrics] 开始计算停顿指标 | 停顿数: {len(pauses)} | 总时长: {total_duration:.2f}s")
        
        # 分类统计
        short = sum(1 for p in pauses if p.type == PauseType.SHORT)
        medium = sum(1 for p in pauses if p.type == PauseType.MEDIUM)
        long = sum(1 for p in pauses if p.type == PauseType.LONG)
        logger.info(f"[Pause] [Metrics] 分类统计: 短={short}, 中={medium}, 长={long}")
        
        # 时长统计
        total_pause_time = sum(p.duration for p in pauses)
        avg_duration = total_pause_time / len(pauses) if pauses else 0
        logger.info(f"[Pause] [Metrics] 时长统计: 总停顿时长={total_pause_time:.2f}s, 平均停顿时长={avg_duration:.2f}s")
        
        # 频率统计
        frequency = len(pauses) / (total_duration / 60) if total_duration > 0 else 0
        logger.info(f"[Pause] [Metrics] 频率统计: {len(pauses)} / ({total_duration:.2f}/60) = {frequency:.1f} 次/分钟")
        
        # 停顿占比
        ratio = total_pause_time / total_duration if total_duration > 0 else 0
        logger.info(f"[Pause] [Metrics] 停顿占比: {total_pause_time:.2f} / {total_duration:.2f} = {ratio*100:.1f}%")
        
        return PauseMetrics(
            total_pauses=len(pauses),
            short_pauses=short,
            medium_pauses=medium,
            long_pauses=long,
            total_pause_time=total_pause_time,
            average_pause_duration=avg_duration,
            pause_frequency=frequency,
            pause_ratio=ratio,
            pauses=pauses
        )
    
    def _calculate_fluency_score(self, metrics: PauseMetrics) -> float:
        """
        计算流畅度评分
        
        评分标准:
        - 基础分: 80
        - 短停顿比例合适: +10
        - 中停顿比例合适: +5
        - 长停顿过多: -10/个
        - 停顿频率合适: +5
        
        Returns:
            0-100 的流畅度评分
        """
        logger.info("[Pause] [Fluency] 开始计算流畅度评分...")
        logger.info(f"[Pause] [Fluency] 评分参数: 理想短停顿比例={self.FLUENCY_PARAMS['ideal_short_ratio']}, "
                   f"理想中停顿比例={self.FLUENCY_PARAMS['ideal_medium_ratio']}, "
                   f"最大可接受长停顿={self.FLUENCY_PARAMS['max_acceptable_long']}, "
                   f"理想停顿频率={self.FLUENCY_PARAMS['ideal_pause_frequency']}次/分钟")
        
        if metrics.total_pauses == 0:
            logger.info("[Pause] [Fluency] 无停顿，返回基础分 85")
            return 85  # 无停顿，可能过于流畅
        
        score = 80.0
        logger.info(f"[Pause] [Fluency] 基础分: {score}")
        
        # 短停顿比例
        short_ratio = metrics.short_pauses / metrics.total_pauses
        if 0.6 <= short_ratio <= 0.8:
            score += 10
            logger.info(f"[Pause] [Fluency] 短停顿比例 {short_ratio:.2f} 在理想范围 [0.6, 0.8]，+10 分 -> {score}")
        elif short_ratio < 0.5:
            score -= 5
            logger.info(f"[Pause] [Fluency] 短停顿比例 {short_ratio:.2f} 过低 (<0.5)，-5 分 -> {score}")
        else:
            logger.info(f"[Pause] [Fluency] 短停顿比例 {short_ratio:.2f} 不在理想范围，不加分 -> {score}")
        
        # 中停顿比例
        medium_ratio = metrics.medium_pauses / metrics.total_pauses
        if 0.15 <= medium_ratio <= 0.3:
            score += 5
            logger.info(f"[Pause] [Fluency] 中停顿比例 {medium_ratio:.2f} 在理想范围 [0.15, 0.3]，+5 分 -> {score}")
        else:
            logger.info(f"[Pause] [Fluency] 中停顿比例 {medium_ratio:.2f} 不在理想范围 [0.15, 0.3]，不加分 -> {score}")
        
        # 长停顿惩罚
        if metrics.long_pauses > self.FLUENCY_PARAMS['max_acceptable_long']:
            excess = metrics.long_pauses - self.FLUENCY_PARAMS['max_acceptable_long']
            penalty = excess * 10
            score -= penalty
            logger.info(f"[Pause] [Fluency] 长停顿 {metrics.long_pauses} 超过阈值 {self.FLUENCY_PARAMS['max_acceptable_long']}，"
                       f"超额 {excess} 个，惩罚 -{penalty} 分 -> {score}")
        else:
            logger.info(f"[Pause] [Fluency] 长停顿 {metrics.long_pauses} 未超过阈值 {self.FLUENCY_PARAMS['max_acceptable_long']}，无惩罚 -> {score}")
        
        # 停顿频率
        if 5 <= metrics.pause_frequency <= 15:
            score += 5
            logger.info(f"[Pause] [Fluency] 停顿频率 {metrics.pause_frequency:.1f} 在理想范围 [5, 15] 次/分钟，+5 分 -> {score}")
        elif metrics.pause_frequency > 25:
            score -= 10
            logger.info(f"[Pause] [Fluency] 停顿频率 {metrics.pause_frequency:.1f} 过高 (>25 次/分钟)，-10 分 -> {score}")
        else:
            logger.info(f"[Pause] [Fluency] 停顿频率 {metrics.pause_frequency:.1f} 不在理想范围，不加分 -> {score}")
        
        # 停顿占比
        if metrics.pause_ratio > 0.4:
            score -= 15
            logger.info(f"[Pause] [Fluency] 停顿占比 {metrics.pause_ratio*100:.1f}% 过高 (>40%)，-15 分 -> {score}")
        else:
            logger.info(f"[Pause] [Fluency] 停顿占比 {metrics.pause_ratio*100:.1f}% 正常 (<=40%)，不扣分 -> {score}")
        
        final_score = max(0, min(100, score))
        logger.info(f"[Pause] [Fluency] 最终流畅度评分: {final_score:.1f} (原始={score:.1f}, 限制在 [0, 100])")
        
        return final_score
    
    def get_pause_feedback(self, metrics: PauseMetrics) -> str:
        """
        获取停顿反馈建议
        
        Args:
            metrics: 停顿指标
            
        Returns:
            反馈文字
        """
        feedback_parts = []
        
        # 长停顿反馈
        if metrics.long_pauses >= 3:
            feedback_parts.append(
                f"出现 {metrics.long_pauses} 次较长停顿（>2秒），"
                f"可能存在紧张或思路中断，建议提前准备回答框架"
            )
        elif metrics.long_pauses >= 1:
            feedback_parts.append(
                f"出现 {metrics.long_pauses} 次较长停顿，注意保持思路连贯"
            )
        
        # 中停顿反馈
        if metrics.medium_pauses >= 5:
            feedback_parts.append(
                f"中等停顿较多（{metrics.medium_pauses} 次），"
                f"可以适当减少思考时间，增强表达流畅度"
            )
        
        # 停顿频率反馈
        if metrics.pause_frequency > 20:
            feedback_parts.append(
                f"停顿过于频繁（{metrics.pause_frequency:.1f} 次/分钟），"
                f"建议减少不必要的停顿，保持表达连贯"
            )
        elif metrics.pause_frequency < 3:
            feedback_parts.append(
                "停顿较少，表达较为流畅"
            )
        
        # 流畅度总评
        if metrics.fluency_score >= 85:
            feedback_parts.append("整体表达流畅，节奏控制良好")
        elif metrics.fluency_score >= 70:
            feedback_parts.append("表达基本流畅，仍有提升空间")
        else:
            feedback_parts.append("表达流畅度有待提升，建议多加练习")
        
        return "；".join(feedback_parts) if feedback_parts else "停顿模式正常"
    
    def get_pause_distribution(self, metrics: PauseMetrics) -> Dict[str, Any]:
        """
        获取停顿分布数据（用于图表展示）
        
        Returns:
            {
                "labels": ["短停顿", "中停顿", "长停顿"],
                "values": [count1, count2, count3],
                "percentages": [pct1, pct2, pct3]
            }
        """
        total = metrics.total_pauses
        if total == 0:
            return {
                "labels": ["短停顿(<0.5s)", "中停顿(0.5-2s)", "长停顿(>2s)"],
                "values": [0, 0, 0],
                "percentages": [0, 0, 0]
            }
        
        return {
            "labels": ["短停顿(<0.5s)", "中停顿(0.5-2s)", "长停顿(>2s)"],
            "values": [
                metrics.short_pauses,
                metrics.medium_pauses,
                metrics.long_pauses
            ],
            "percentages": [
                round(metrics.short_pauses / total * 100, 1),
                round(metrics.medium_pauses / total * 100, 1),
                round(metrics.long_pauses / total * 100, 1)
            ]
        }
