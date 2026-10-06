"""
锐聘AI - 语速分析模块

功能:
- WCPM (Words/Characters Per Minute) 计算
- 语速变化曲线
- 语速分类（慢/正常/快）
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

from ..logger import logger, log_function_call


class SpeechRateCategory(Enum):
    """语速分类"""
    SLOW = "slow"
    NORMAL = "normal"
    FAST = "fast"


@dataclass
class SpeechRateMetrics:
    """语速指标"""
    wcpm: float                           # 每分钟字数
    total_words: int                      # 总字数
    total_duration: float                 # 总时长（秒）
    speech_duration: float                # 纯语音时长（秒，去除停顿）
    rate_curve: List[Tuple[float, float]] # 语速变化曲线 [(时间, WCPM), ...]
    rate_category: str                    # 语速分类
    avg_chars_per_second: float           # 平均每秒字数


class SpeechRateAnalyzer:
    """
    语速分析器
    
    支持中文和英文的语速计算:
    - 中文: 按字符数计算
    - 英文: 按单词数计算
    """
    
    # WCPM 参考标准（中文）
    CHINESE_THRESHOLDS = {
        'slow': (0, 120),
        'normal': (120, 200),
        'fast': (200, 400)
    }
    
    # WCPM 参考标准（英文）
    ENGLISH_THRESHOLDS = {
        'slow': (0, 100),
        'normal': (100, 160),
        'fast': (160, 300)
    }
    
    def __init__(
        self,
        window_size: float = 5.0,     # 计算WCPM的窗口大小（秒）
        hop_size: float = 1.0,        # 窗口滑动步长（秒）
        language: str = "zh"
    ):
        """
        Args:
            window_size: 语速计算窗口大小
            hop_size: 窗口滑动步长
            language: 语言代码 (zh/en)
        """
        self.window_size = window_size
        self.hop_size = hop_size
        self.language = language
        self.thresholds = (self.CHINESE_THRESHOLDS if language == "zh" 
                          else self.ENGLISH_THRESHOLDS)
        
        logger.info(f"[SpeechRate] 初始化语速分析器 | 语言: {language} | "
                   f"窗口: {window_size}s | 步长: {hop_size}s")
    
    @log_function_call()
    def analyze(
        self,
        audio: np.ndarray,
        sr: int = 16000,
        transcribed_text: str = ""
    ) -> SpeechRateMetrics:
        """
        分析语速
        
        Args:
            audio: 音频数据
            sr: 采样率
            transcribed_text: 转录文字
            
        Returns:
            SpeechRateMetrics
        """
        logger.info("=" * 60)
        logger.info("开始语速分析")
        logger.info("=" * 60)
        logger.info(f"[SpeechRate] 输入音频长度: {len(audio)} 样本 | 采样率: {sr}Hz | "
                   f"时长: {len(audio)/sr:.2f}s")
        logger.info(f"[SpeechRate] 输入文字: '{transcribed_text[:50]}...' ({len(transcribed_text)} 字符)")
        
        # 1. 检测语音活动区域
        logger.info("[SpeechRate] [STEP 1/6] 检测语音活动段...")
        speech_segments = self._detect_speech_segments(audio, sr)
        logger.info(f"[SpeechRate] [STEP 1_OK] 检测到 {len(speech_segments)} 个语音段")
        for i, (start, end) in enumerate(speech_segments):
            logger.info(f"[SpeechRate]   语音段 {i+1}: {start:.2f}s - {end:.2f}s (时长: {end-start:.2f}s)")
        
        # 2. 计算时长
        logger.info("[SpeechRate] [STEP 2/6] 计算时长...")
        total_duration = len(audio) / sr
        speech_duration = sum(end - start for start, end in speech_segments)
        silence_duration = total_duration - speech_duration
        
        logger.info(f"[SpeechRate] [STEP 2_OK] 总时长: {total_duration:.2f}s | "
                   f"语音时长: {speech_duration:.2f}s | 静音时长: {silence_duration:.2f}s | "
                   f"语音占比: {speech_duration/total_duration*100:.1f}%")
        
        # 3. 计算字数
        logger.info("[SpeechRate] [STEP 3/6] 计算字数...")
        total_words = self._count_words(transcribed_text)
        logger.info(f"[SpeechRate] [STEP 3_OK] 从文字统计字数: {total_words}")
        
        if total_words == 0 and speech_duration > 0:
            # 无转录文字时，基于时长估算
            logger.warning("[SpeechRate] [STEP 3_WARN] 无转录文字，将基于语音时长估算字数")
            total_words = self._estimate_word_count(speech_duration)
            logger.info(f"[SpeechRate] [STEP 3_OK] 估算字数: {total_words} "
                       f"(基于 {speech_duration:.2f}s * {3.5 if self.language=='zh' else 2.5} 字/秒)")
        elif total_words == 0 and speech_duration == 0:
            logger.error("[SpeechRate] [STEP 3_FAIL] 无文字且无语音，无法计算语速")
        
        logger.info(f"[SpeechRate] [STEP 3_OK] 最终字数: {total_words}")
        
        # 4. 计算 WCPM
        logger.info("[SpeechRate] [STEP 4/6] 计算 WCPM...")
        if speech_duration > 0:
            wcpm = (total_words / speech_duration) * 60
            avg_chars_per_second = total_words / speech_duration
            logger.info(f"[SpeechRate] [STEP 4_OK] WCPM 计算公式: ({total_words} / {speech_duration:.2f}) * 60 = {wcpm:.1f}")
            logger.info(f"[SpeechRate] [STEP 4_OK] 每秒字数: {avg_chars_per_second:.2f}")
        else:
            wcpm = 0
            avg_chars_per_second = 0
            logger.warning("[SpeechRate] [STEP 4_WARN] 语音时长为0，WCPM设为0")
        
        # 5. 计算语速变化曲线
        logger.info("[SpeechRate] [STEP 5/6] 计算语速变化曲线...")
        rate_curve = self._calculate_rate_curve(
            audio, sr, speech_segments, transcribed_text, total_words
        )
        logger.info(f"[SpeechRate] [STEP 5_OK] 语速曲线点: {len(rate_curve)}")
        if rate_curve:
            wcpm_values = [w for _, w in rate_curve]
            logger.info(f"[SpeechRate] [STEP 5_OK] 曲线 WCPM 范围: {min(wcpm_values):.1f} - {max(wcpm_values):.1f}")
            logger.info(f"[SpeechRate] [STEP 5_OK] 曲线 WCPM 均值: {np.mean(wcpm_values):.1f}")
        else:
            logger.info("[SpeechRate] [STEP 5_OK] 语速曲线为空（音频过短或无足够数据）")
        
        # 6. 判断语速类别
        logger.info("[SpeechRate] [STEP 6/6] 判断语速类别...")
        rate_category = self._categorize_rate(wcpm)
        logger.info(f"[SpeechRate] [STEP 6_OK] WCPM={wcpm:.1f} 属于类别: {rate_category}")
        logger.info(f"[SpeechRate] [STEP 6_OK] 阈值参考: 慢<{self.thresholds['normal'][0]}, "
                   f"正常{self.thresholds['normal'][0]}-{self.thresholds['normal'][1]}, "
                   f"快>{self.thresholds['normal'][1]}")
        
        logger.info("=" * 60)
        logger.info("语速分析完成")
        logger.info(f"[SpeechRate] 最终结果: WCPM={wcpm:.1f}, 类别={rate_category}, 字数={total_words}")
        logger.info("=" * 60)
        
        return SpeechRateMetrics(
            wcpm=wcpm,
            total_words=total_words,
            total_duration=total_duration,
            speech_duration=speech_duration,
            rate_curve=rate_curve,
            rate_category=rate_category,
            avg_chars_per_second=avg_chars_per_second
        )
    
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
        logger.info("[SpeechRate] [VAD] 开始语音活动检测...")
        logger.info(f"[SpeechRate] [VAD] 音频长度: {len(audio)} 样本 | 采样率: {sr}Hz")
        
        # 使用能量阈值进行简单的 VAD
        frame_length = int(0.02 * sr)   # 20ms 帧
        hop_length = int(0.01 * sr)     # 10ms 步长
        
        logger.info(f"[SpeechRate] [VAD] 帧长: {frame_length} 样本 ({frame_length/sr*1000:.1f}ms) | "
                   f"帧移: {hop_length} 样本 ({hop_length/sr*1000:.1f}ms)")
        
        # 计算每帧能量
        logger.info("[SpeechRate] [VAD] 计算每帧能量...")
        energies = []
        for i in range(0, len(audio) - frame_length, hop_length):
            frame = audio[i:i + frame_length]
            energy = np.sqrt(np.mean(frame ** 2))
            energies.append(energy)
        
        logger.info(f"[SpeechRate] [VAD] 总帧数: {len(energies)}")
        
        if not energies:
            logger.warning("[SpeechRate] [VAD_WARN] 无有效帧，返回空语音段")
            return []
        
        # 统计能量信息
        energy_array = np.array(energies)
        logger.info(f"[SpeechRate] [VAD] 能量统计: 均值={np.mean(energy_array):.6f}, "
                   f"标准差={np.std(energy_array):.6f}, 最大={np.max(energy_array):.6f}, "
                   f"最小={np.min(energy_array):.6f}")
        
        # 自适应阈值
        threshold = max(0.01, np.mean(energies) * 0.3)
        logger.info(f"[SpeechRate] [VAD] 能量阈值: {threshold:.6f} (max(0.01, 均值*0.3))")
        
        # 检测语音段
        segments = []
        in_speech = False
        speech_start = 0
        
        logger.info("[SpeechRate] [VAD] 开始检测语音起止...")
        for i, energy in enumerate(energies):
            time_pos = i * hop_length / sr
            
            if energy > threshold and not in_speech:
                in_speech = True
                speech_start = time_pos
                logger.debug(f"[SpeechRate] [VAD] 语音开始 @ {time_pos:.3f}s (能量: {energy:.6f})")
            elif energy < threshold * 0.6 and in_speech:
                in_speech = False
                duration = time_pos - speech_start
                if duration >= 0.1:  # 至少 100ms
                    segments.append((speech_start, time_pos))
                    logger.debug(f"[SpeechRate] [VAD] 语音结束 @ {time_pos:.3f}s (时长: {duration:.3f}s)")
                else:
                    logger.debug(f"[SpeechRate] [VAD] 丢弃过短语音段: {duration:.3f}s")
        
        # 处理结尾
        if in_speech:
            final_time = len(audio) / sr
            duration = final_time - speech_start
            if duration >= 0.1:
                segments.append((speech_start, final_time))
                logger.debug(f"[SpeechRate] [VAD] 末尾语音段: {speech_start:.3f}s - {final_time:.3f}s")
        
        logger.info(f"[SpeechRate] [VAD_OK] 检测完成，找到 {len(segments)} 个语音段")
        
        return segments
    
    def _count_words(self, text: str) -> int:
        """
        计算字数
        
        中文: 字符数
        英文: 单词数
        """
        logger.info(f"[SpeechRate] [CountWords] 开始字数统计 | 语言: {self.language}")
        logger.info(f"[SpeechRate] [CountWords] 输入文字: '{text[:80]}...' ({len(text)} 字符)")
        
        if not text:
            logger.info("[SpeechRate] [CountWords] 文字为空，返回0")
            return 0
        
        text = text.strip()
        
        if self.language == "zh":
            # 中文字符数（去除空格和标点）
            import re
            chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
            count = len(chinese_chars)
            logger.info(f"[SpeechRate] [CountWords] 中文字符数: {count} (从 {len(text)} 字符中匹配)")
        else:
            # 英文单词数
            words = text.split()
            count = len(words)
            logger.info(f"[SpeechRate] [CountWords] 英文单词数: {count}")
        
        return count
    
    def _estimate_word_count(self, speech_duration: float) -> int:
        """
        基于语音时长估算字数
        
        中文平均语速: 3-4 字/秒
        英文平均语速: 2-3 词/秒
        """
        rate = 3.5 if self.language == "zh" else 2.5
        estimated = int(speech_duration * rate)
        logger.info(f"[SpeechRate] [Estimate] 估算字数: {estimated} = {speech_duration:.2f}s * {rate} 字/秒")
        return estimated
    
    def _calculate_rate_curve(
        self,
        audio: np.ndarray,
        sr: int,
        speech_segments: List[Tuple[float, float]],
        transcribed_text: str,
        total_words: int
    ) -> List[Tuple[float, float]]:
        """
        计算语速变化曲线
        
        将文字按时间窗口分配，计算每个窗口的 WCPM
        
        Returns:
            [(window_center_time, wcpm), ...]
        """
        logger.info("[SpeechRate] [Curve] 开始计算语速变化曲线...")
        total_duration = len(audio) / sr
        
        logger.info(f"[SpeechRate] [Curve] 总时长: {total_duration:.2f}s | 总字数: {total_words} | "
                   f"窗口: {self.window_size}s | 步长: {self.hop_size}s")
        
        if total_words == 0:
            logger.warning("[SpeechRate] [Curve_WARN] 总字数为0，无法计算语速曲线")
            return []
        
        if total_duration < self.window_size:
            logger.warning(f"[SpeechRate] [Curve_WARN] 音频时长({total_duration:.2f}s) < 窗口大小({self.window_size}s)，无法计算曲线")
            return []
        
        # 计算窗口数量
        num_windows = max(1, int((total_duration - self.window_size) / self.hop_size) + 1)
        logger.info(f"[SpeechRate] [Curve] 窗口数量: {num_windows}")
        
        # 每个窗口的字数（均匀分配）
        words_per_window = total_words / num_windows if num_windows > 0 else 0
        logger.info(f"[SpeechRate] [Curve] 每窗口字数: {words_per_window:.2f}")
        
        curve = []
        
        for i in range(num_windows):
            window_start = i * self.hop_size
            window_end = window_start + self.window_size
            window_center = window_start + self.window_size / 2
            
            # 计算窗口内的语音时长
            window_speech_time = 0
            for seg_start, seg_end in speech_segments:
                overlap_start = max(seg_start, window_start)
                overlap_end = min(seg_end, window_end)
                if overlap_end > overlap_start:
                    window_speech_time += overlap_end - overlap_start
            
            # 计算该窗口的 WCPM
            if window_speech_time > 0.5:  # 至少 0.5 秒语音
                window_wcpm = (words_per_window / window_speech_time) * 60
                curve.append((round(window_center, 1), round(window_wcpm, 1)))
                logger.debug(f"[SpeechRate] [Curve] 窗口 {i+1}: "
                           f"{window_start:.1f}s-{window_end:.1f}s | "
                           f"语音: {window_speech_time:.2f}s | WCPM: {window_wcpm:.1f}")
            else:
                logger.debug(f"[SpeechRate] [Curve] 窗口 {i+1}: "
                           f"{window_start:.1f}s-{window_end:.1f}s | "
                           f"语音不足({window_speech_time:.2f}s < 0.5s)，跳过")
        
        logger.info(f"[SpeechRate] [Curve_OK] 语速曲线计算完成，共 {len(curve)} 个点")
        
        return curve
    
    def _categorize_rate(self, wcpm: float) -> str:
        """判断语速类别"""
        logger.info(f"[SpeechRate] [Category] 判断语速类别 | WCPM={wcpm:.1f}")
        logger.info(f"[SpeechRate] [Category] 阈值: 慢<{self.thresholds['normal'][0]}, "
                   f"正常{self.thresholds['normal'][0]}-{self.thresholds['normal'][1]}, "
                   f"快>{self.thresholds['normal'][1]}")
        
        if wcpm < self.thresholds['normal'][0]:
            category = SpeechRateCategory.SLOW.value
            logger.info(f"[SpeechRate] [Category_OK] WCPM={wcpm:.1f} < {self.thresholds['normal'][0]} -> 慢")
        elif wcpm < self.thresholds['normal'][1]:
            category = SpeechRateCategory.NORMAL.value
            logger.info(f"[SpeechRate] [Category_OK] {self.thresholds['normal'][0]} <= WCPM={wcpm:.1f} < {self.thresholds['normal'][1]} -> 正常")
        else:
            category = SpeechRateCategory.FAST.value
            logger.info(f"[SpeechRate] [Category_OK] WCPM={wcpm:.1f} >= {self.thresholds['normal'][1]} -> 快")
        
        return category
    
    def get_rate_feedback(self, metrics: SpeechRateMetrics) -> str:
        """
        获取语速反馈建议
        
        Args:
            metrics: 语速指标
            
        Returns:
            反馈文字
        """
        wcpm = metrics.wcpm
        category = metrics.rate_category
        
        if category == SpeechRateCategory.SLOW.value:
            if wcpm < 80:
                return f"语速过慢（{wcpm:.0f} 字/分钟），建议适当加快节奏，保持面试官注意力"
            else:
                return f"语速偏慢（{wcpm:.0f} 字/分钟），可以适当提高语速"
        elif category == SpeechRateCategory.FAST.value:
            if wcpm > 250:
                return f"语速过快（{wcpm:.0f} 字/分钟），建议放慢节奏，确保表达清晰"
            else:
                return f"语速偏快（{wcpm:.0f} 字/分钟），注意控制节奏"
        else:
            return f"语速适中（{wcpm:.0f} 字/分钟），表达节奏良好"
