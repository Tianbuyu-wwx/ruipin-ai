"""
锐聘AI - 智能面试系统
"""

import os
import sys
import json
import time
import threading
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from pathlib import Path

import gradio as gr
import numpy as np

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

# 加载配置
from configs.config import GRADIO_CONFIG, INTERVIEW_CONFIG

from src.interview_engine import InterviewEngine, InterviewStatus
from src.logger import logger, LogContext, log_function_call, performance_logger

# 导入多模态评估器（用于硬件检测和推理模式切换）
try:
    from src.multimodal_evaluator import MultimodalEvaluator
    MULTIMODAL_AVAILABLE = True
except ImportError as e:
    logger.warning(f"多模态评估器导入失败: {e}")
    MULTIMODAL_AVAILABLE = False
    MultimodalEvaluator = None

# 导入新模块
try:
    from src.security import validate_and_sanitize_answer, check_rate_limit
    from src.database import db_manager
    SECURITY_AVAILABLE = True
except ImportError as e:
    logger.warning(f"安全模块导入失败: {e}")
    SECURITY_AVAILABLE = False
    validate_and_sanitize_answer = None
    check_rate_limit = None
    db_manager = None

# 导入音频处理模块
try:
    from src.audio_processor import AudioProcessorManager
    AUDIO_PROCESSOR_AVAILABLE = True
except ImportError as e:
    logger.warning(f"音频处理模块导入失败: {e}")
    AUDIO_PROCESSOR_AVAILABLE = False
    AudioProcessorManager = None

# 导入实时转录模块
try:
    from src.realtime_transcription import (
        RealtimeTranscriptionManager,
        init_rt_manager,
        FASTAPI_AVAILABLE as RT_FASTAPI_AVAILABLE
    )
    REALTIME_TRANSCRIPTION_AVAILABLE = True
except ImportError as e:
    logger.warning(f"实时转录模块导入失败: {e}")
    REALTIME_TRANSCRIPTION_AVAILABLE = False
    RealtimeTranscriptionManager = None
    init_rt_manager = None
    RT_FASTAPI_AVAILABLE = False

# 导入声纹分析模块
try:
    from src.voice_analysis import VoiceAnalysisManager
    VOICE_ANALYSIS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"声纹分析模块导入失败: {e}")
    VOICE_ANALYSIS_AVAILABLE = False
    VoiceAnalysisManager = None


def perform_hardware_check():
    """
    启动时自动执行硬件检测和推理模式选择
    显示系统硬件信息和推荐的推理配置
    """
    logger.info("=" * 70)
    logger.info("系统启动 - 硬件检测与推理模式选择")
    logger.info("=" * 70)

    try:
        if MULTIMODAL_AVAILABLE and MultimodalEvaluator:
            # 创建临时评估器实例进行硬件检测
            # 注意：这里不加载模型，只检测硬件
            evaluator = MultimodalEvaluator.__new__(MultimodalEvaluator)
            evaluator.model_path = "base_model"

            # 执行硬件检测
            hardware_info = evaluator._detect_hardware()

            # 执行模式选择
            evaluator._auto_select_mode()
            mode_info = evaluator.get_current_mode()

            logger.info("=" * 70)
            logger.info("硬件检测完成 - 系统配置摘要")
            logger.info("=" * 70)

            # 显示硬件信息
            if hardware_info["cuda_available"]:
                logger.info(f"GPU: {hardware_info['gpus'][0]['name']}")
                logger.info(f"总显存: {hardware_info['total_vram_gb']:.2f} GB")
                logger.info(f"可用显存: {hardware_info['free_vram_gb']:.2f} GB")
                logger.info(f"CUDA 版本: {hardware_info['cuda_version']}")
            else:
                logger.info("GPU: 未检测到 CUDA 设备")

            # 显示选择的推理模式
            logger.info(f"推理模式: {mode_info['description']}")
            logger.info(f"模式ID: {mode_info['mode']}")

            if mode_info['config']:
                config = mode_info['config']
                logger.info(f"量化: {'启用' if config.get('load_in_4bit') else '禁用'}")
                logger.info(f"数据类型: {config.get('torch_dtype', 'float32')}")
                logger.info(f"最大图像尺寸: {config.get('max_image_size', '未限制')}")
                logger.info(f"最大回答长度: {config.get('max_answer_len', '未限制')}")

            logger.info("=" * 70)
            logger.info("[OK] 硬件检测和推理模式选择完成")
            logger.info("=" * 70)

            return {
                "hardware": hardware_info,
                "mode": mode_info
            }
        else:
            logger.warning("多模态评估器不可用，跳过硬件检测")
            return None

    except Exception as e:
        logger.error(f"硬件检测失败: {e}", exc_info=True)
        return None


class InterviewWebApp:
    """面试Web应用"""
    
    def __init__(self):
        logger.info("=" * 70)
        logger.info("初始化锐聘AI Web应用")
        logger.info("=" * 70)

        # 启动时自动执行硬件检测
        self.hardware_info = None
        self.inference_mode = None
        try:
            hardware_result = perform_hardware_check()
            if hardware_result:
                self.hardware_info = hardware_result.get("hardware")
                self.inference_mode = hardware_result.get("mode")
        except Exception as e:
            logger.warning(f"启动时硬件检测失败: {e}")

        try:
            from configs.config import MULTIMODAL_CONFIG
            model_path = MULTIMODAL_CONFIG.get("qwen3vl_model_path")
            logger.info(f"模型路径: {model_path}")

            logger.info("正在初始化面试引擎...")
            with LogContext(logger, "面试引擎初始化"):
                self.engine = InterviewEngine(model_path)
            logger.info("[OK] 面试引擎初始化完成")
            
            self.session_id: Optional[str] = None
            
            self.is_running = False
            self.is_waiting_for_answer = False
            self.start_time = None
            self.elapsed_seconds = 0
            self.timer_thread = None
            
            self.chat_history: List[Tuple[Optional[str], Optional[str]]] = []
            self.ui_components = {}
            
            # 初始化音频处理器
            self.audio_processor = None
            if AUDIO_PROCESSOR_AVAILABLE and AudioProcessorManager:
                try:
                    logger.info("正在初始化音频处理器...")
                    self.audio_processor = AudioProcessorManager()
                    if self.audio_processor.is_available():
                        logger.info("[OK] 音频处理器初始化成功")
                    else:
                        logger.warning("音频处理器不可用")
                        self.audio_processor = None
                except Exception as e:
                    logger.warning(f"音频处理器初始化失败: {e}")
                    self.audio_processor = None
            
            # 初始化实时转录管理器
            self.rt_manager = None
            if REALTIME_TRANSCRIPTION_AVAILABLE and init_rt_manager:
                try:
                    logger.info("正在初始化实时转录管理器...")
                    # 如果有音频处理器，共享 Whisper 实例
                    whisper_processor = None
                    if self.audio_processor and self.audio_processor.whisper:
                        whisper_processor = self.audio_processor.whisper
                    
                    init_rt_manager(whisper_processor)
                    logger.info("[OK] 实时转录管理器初始化成功")
                except Exception as e:
                    logger.warning(f"实时转录管理器初始化失败: {e}")
            
            # 初始化声纹分析器
            self.voice_analyzer = None
            self.voice_analysis_results = {}  # 存储每回合的声纹分析结果
            if VOICE_ANALYSIS_AVAILABLE and VoiceAnalysisManager:
                try:
                    logger.info("正在初始化声纹分析器...")
                    self.voice_analyzer = VoiceAnalysisManager(language="zh")
                    logger.info("[OK] 声纹分析器初始化成功")
                except Exception as e:
                    logger.warning(f"声纹分析器初始化失败: {e}")
                    self.voice_analyzer = None
            
            # 实时转录状态
            self.realtime_transcription_enabled = True
            self.current_transcription = ""
            self.ws_connection_status = "disconnected"
            
            # 心率监测状态
            self.hr_monitoring = False
            self.hr_thread = None
            self.hr_data = []  # 心率历史数据 [(timestamp, heart_rate), ...]
            self.hr_start_time = None
            
            logger.info("[OK] Web应用初始化完成")
            
        except Exception as e:
            logger.error(f"[ERROR] Web应用初始化失败: {e}", exc_info=True)
            raise
    
    def _extract_frame_from_video(self, video_path: str) -> Optional[np.ndarray]:
        """从视频文件中提取一帧图像"""
        try:
            import cv2
            import os

            logger.info("=" * 60)
            logger.info("开始从视频中提取帧")
            logger.info("=" * 60)
            logger.info(f"视频路径: {video_path}")

            # 检查文件是否存在
            if not os.path.exists(video_path):
                logger.error(f"[ERROR] 视频文件不存在: {video_path}")
                return None

            # 安全检查：文件大小限制（100MB）
            max_size_mb = 100
            file_size = os.path.getsize(video_path)
            file_size_mb = file_size / (1024 * 1024)
            logger.info(f"视频文件大小: {file_size_mb:.2f}MB")

            if file_size > max_size_mb * 1024 * 1024:
                logger.error(f"[ERROR] 视频文件过大: {file_size_mb:.1f}MB > {max_size_mb}MB")
                return None
            logger.info("[OK] 文件大小检查通过")

            # 打开视频文件
            logger.info("正在打开视频文件...")
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"[ERROR] 无法打开视频文件: {video_path}")
                return None
            logger.info("[OK] 视频文件打开成功")

            # 获取视频信息
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = total_frames / fps if fps > 0 else 0

            logger.info(f"视频信息:")
            logger.info(f"  - 总帧数: {total_frames}")
            logger.info(f"  - 帧率: {fps:.2f} FPS")
            logger.info(f"  - 分辨率: {width}x{height}")
            logger.info(f"  - 时长: {duration:.2f}秒")

            if total_frames <= 0:
                logger.error("[ERROR] 视频文件无有效帧")
                cap.release()
                return None

            # 读取中间帧
            middle_frame = total_frames // 2
            logger.info(f"准备读取第 {middle_frame}/{total_frames} 帧（中间帧）...")
            cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)

            ret, frame = cap.read()
            cap.release()

            if ret:
                # 转换 BGR 到 RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                logger.info(f"[OK] 成功提取视频帧")
                logger.info(f"  - 帧形状: {frame_rgb.shape}")
                logger.info(f"  - 数据类型: {frame_rgb.dtype}")
                logger.info(f"  - 像素范围: [{frame_rgb.min()}, {frame_rgb.max()}]")
                return frame_rgb
            else:
                logger.error("[ERROR] 无法读取视频帧（cap.read() 返回 False）")
                return None

        except Exception as e:
            logger.error(f"[ERROR] 提取视频帧时发生错误: {e}", exc_info=True)
            return None
    
    def _generate_heart_rate_data(self):
        """生成模拟心率数据（后台线程）"""
        import random
        import time
        import math
        
        logger.info("[OK] 心率监测线程启动")
        
        # 初始化心率参数，使数据更真实
        current_hr = 72  # 起始心率
        target_hr = 72   # 目标心率
        trend = 0        # 趋势
        
        while self.hr_monitoring:
            try:
                # 根据面试状态确定基础心率范围
                if self.is_running:
                    # 面试进行中，心率偏高且有波动
                    # 模拟紧张时的生理反应：心率逐渐上升，偶尔有波动
                    interview_duration = time.time() - self.hr_start_time if self.hr_start_time else 0
                    
                    # 面试初期心率上升，后期趋于稳定或略微下降（适应）
                    if interview_duration < 60:
                        # 前1分钟：心率快速上升期
                        target_hr = random.randint(82, 95)
                    elif interview_duration < 180:
                        # 1-3分钟：心率高峰期
                        target_hr = random.randint(88, 102)
                    else:
                        # 3分钟后：适应期，心率略微下降但仍高于正常
                        target_hr = random.randint(78, 92)
                else:
                    # 面试未开始，正常静息心率
                    target_hr = random.randint(68, 76)
                
                # 使用平滑过渡，模拟真实心率变化（不是瞬间跳变）
                # 向目标心率靠拢，每次变化不超过3 BPM
                diff = target_hr - current_hr
                change = int(diff * 0.3) + random.randint(-2, 2)
                change = max(-3, min(3, change))  # 限制单次变化幅度
                
                current_hr += change
                
                # 添加呼吸相关的周期性波动（呼吸性窦性心律不齐）
                # 心率随呼吸周期轻微波动（约4-6秒一个周期）
                breath_cycle = math.sin(time.time() * 1.5) * 2  # 呼吸周期波动 ±2 BPM
                heart_rate = int(current_hr + breath_cycle)
                
                # 偶尔添加小的随机波动（模拟身体微小变化）
                if random.random() < 0.15:  # 15%概率
                    heart_rate += random.randint(-2, 2)
                
                # 确保心率在合理范围内（正常成年人静息心率60-100，运动时可达120）
                heart_rate = max(58, min(115, heart_rate))
                
                # 记录时间戳（相对于开始监测的时间）
                if self.hr_start_time:
                    elapsed = time.time() - self.hr_start_time
                else:
                    elapsed = 0
                
                self.hr_data.append((round(elapsed, 1), heart_rate))
                
                # 只保留最近 5 分钟的数据
                if len(self.hr_data) > 300:
                    self.hr_data.pop(0)
                
                logger.debug(f"心率数据: {heart_rate} BPM, 时间: {elapsed:.1f}s")
                
                # 每秒更新一次
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"心率监测线程错误: {e}")
                break
        
        logger.info("[OK] 心率监测线程停止")
    
    def start_heart_rate_monitoring(self):
        """开始心率监测"""
        try:
            if self.hr_monitoring:
                return "已在监测中", self._get_hr_plot_data()
            
            self.hr_monitoring = True
            self.hr_data = []
            self.hr_start_time = time.time()
            
            # 启动监测线程
            self.hr_thread = threading.Thread(target=self._generate_heart_rate_data, daemon=True)
            self.hr_thread.start()
            
            logger.info("[OK] 心率监测已启动")
            return "监测中", self._get_hr_plot_data()
            
        except Exception as e:
            logger.error(f"启动心率监测失败: {e}")
            return "启动失败", self._get_hr_plot_data()
    
    def stop_heart_rate_monitoring(self):
        """停止心率监测"""
        try:
            self.hr_monitoring = False
            if self.hr_thread:
                self.hr_thread.join(timeout=2)
            
            logger.info("[OK] 心率监测已停止")
            return "已停止", self._get_hr_plot_data()
            
        except Exception as e:
            logger.error(f"停止心率监测失败: {e}")
            return "停止失败", self._get_hr_plot_data()
    
    def get_current_heart_rate(self):
        """获取当前心率数据"""
        try:
            if not self.hr_data:
                return 0, "未开始", self._get_hr_plot_data()
            
            current_hr = self.hr_data[-1][1] if self.hr_data else 0
            
            # 根据心率判断状态
            if current_hr < 60:
                status = "偏低"
            elif current_hr < 80:
                status = "正常"
            elif current_hr < 100:
                status = "略高"
            else:
                status = "偏高"
            
            if not self.hr_monitoring:
                status = "已停止"
            
            return current_hr, status, self._get_hr_plot_data()
            
        except Exception as e:
            logger.error(f"获取心率数据失败: {e}")
            return 0, "错误", self._get_hr_plot_data()
    
    def _get_hr_plot_data(self):
        """获取用于图表显示的心率数据"""
        import pandas as pd
        
        if not self.hr_data:
            # 返回空数据
            return pd.DataFrame({"时间": [0], "心率": [0]})
        
        times = [d[0] for d in self.hr_data]
        rates = [d[1] for d in self.hr_data]
        
        return pd.DataFrame({"时间": times, "心率": rates})
    
    def create_interface(self) -> gr.Blocks:
        """创建Gradio界面"""
        logger.info("创建Gradio界面...")
        
        try:
            with gr.Blocks(
                title="锐聘AI - 智能面试系统",
                css="""
                /* 全局样式 - 居中显示 */
                .gradio-container { 
                    max-width: 1200px !important; 
                    margin: 0 auto !important;
                }
                
                /* 主容器居中 */
                .main-container {
                    display: flex !important;
                    justify-content: center !important;
                    width: 100% !important;
                }
                
                /* 响应式布局 */
                @media (max-width: 768px) {
                    .main-container { flex-direction: column !important; }
                    .sidebar { order: 1 !important; }
                    .main-content { order: 2 !important; }
                    .input-section { margin-bottom: 10px !important; }
                }
                
                /* 自定义样式 */
                .input-section { border: 2px solid #e0e0e0; border-radius: 10px; padding: 15px; margin: 10px 0; transition: all 0.3s ease; }
                .input-section:hover { border-color: #2196f3; box-shadow: 0 2px 8px rgba(33, 150, 243, 0.1); }
                
                .video-container { border-radius: 10px; overflow: hidden; }
                
                .status-bar { background: #f5f5f5; padding: 10px; border-radius: 5px; font-weight: bold; transition: all 0.3s ease; }
                .status-bar.ready { background: #e8f5e8; color: #2e7d32; }
                .status-bar.running { background: #e3f2fd; color: #1565c0; }
                .status-bar.completed { background: #fff3e0; color: #ef6c00; }
                .status-bar.error { background: #ffebee; color: #c62828; }
                
                .question-box { background: #e3f2fd; padding: 15px; border-radius: 10px; border-left: 4px solid #2196f3; transition: all 0.3s ease; }
                .question-box:hover { box-shadow: 0 2px 8px rgba(33, 150, 243, 0.15); }
                
                .evaluation-box { background: #f3e5f5; padding: 10px; border-radius: 8px; margin-top: 10px; transition: all 0.3s ease; }
                .evaluation-box:hover { box-shadow: 0 2px 8px rgba(156, 39, 176, 0.15); }
                
                /* 按钮样式 */
                .gr-button { transition: all 0.3s ease !important; }
                .gr-button:hover { transform: translateY(-2px) !important; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1) !important; }
                
                /* 滑块样式 */
                .gr-slider input[type="range"] { accent-color: #2196f3 !important; }
                

                
                /* 心率监测样式 */
                .heart-rate-display input { 
                    font-size: 28px !important; 
                    font-weight: bold !important; 
                    color: #e53935 !important;
                    text-align: center !important;
                    border: 2px solid #ffcdd2 !important;
                    border-radius: 10px !important;
                    background: linear-gradient(135deg, #ffebee 0%, #fff 100%) !important;
                }
                .heart-rate-normal { color: #4caf50 !important; }
                .heart-rate-high { color: #ff9800 !important; }
                .heart-rate-very-high { color: #e53935 !important; }
                
                @keyframes heartbeat {
                    0% { transform: scale(1); }
                    50% { transform: scale(1.1); }
                    100% { transform: scale(1); }
                }
                .heart-icon {
                    animation: heartbeat 1s ease-in-out infinite;
                    display: inline-block;
                }
                
                /* 输入区域样式优化 */
                .input-section { 
                    border: 2px solid #e0e0e0; 
                    border-radius: 12px; 
                    padding: 16px; 
                    margin: 12px 0; 
                    transition: all 0.3s ease;
                    background: #fafafa;
                }
                .input-section:hover { 
                    border-color: #2196f3; 
                    box-shadow: 0 4px 12px rgba(33, 150, 243, 0.15);
                    background: #fff;
                }
                
                /* 输入模式选择器样式 */
                .input-mode-selector {
                    background: linear-gradient(135deg, #f5f7fa 0%, #fff 100%);
                    border: 2px solid #e0e0e0;
                    border-radius: 12px;
                    padding: 16px;
                }
                
                /* 按钮样式优化 */
                .gr-button { 
                    transition: all 0.3s ease !important;
                    border-radius: 8px !important;
                    font-weight: 500 !important;
                }
                .gr-button:hover { 
                    transform: translateY(-2px) !important; 
                    box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15) !important;
                }
                .gr-button-primary {
                    background: linear-gradient(135deg, #2196f3 0%, #1976d2 100%) !important;
                }
                .gr-button-secondary {
                    background: linear-gradient(135deg, #f5f5f5 0%, #e0e0e0 100%) !important;
                }
                
                /* 整体布局优化 */
                .main-container { gap: 20px; }
                .sidebar { 
                    background: linear-gradient(180deg, #f8f9fa 0%, #fff 100%);
                    border-radius: 16px;
                    padding: 20px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
                }
                .main-content {
                    background: #fff;
                    border-radius: 16px;
                    padding: 20px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
                }
                
                /* 问题框样式优化 */
                .question-box { 
                    background: linear-gradient(135deg, #e3f2fd 0%, #f3e5f5 100%); 
                    padding: 20px; 
                    border-radius: 12px; 
                    border-left: 5px solid #2196f3;
                    font-size: 16px;
                    line-height: 1.6;
                    box-shadow: 0 2px 8px rgba(33, 150, 243, 0.1);
                }
                
                /* 评估反馈框 */
                .evaluation-box { 
                    background: linear-gradient(135deg, #f3e5f5 0%, #e8f5e8 100%); 
                    padding: 16px; 
                    border-radius: 12px; 
                    border-left: 4px solid #9c27b0;
                    box-shadow: 0 2px 8px rgba(156, 39, 176, 0.1);
                }
                
                /* 状态栏样式 */
                .status-bar { 
                    background: #f5f5f5; 
                    padding: 12px 16px; 
                    border-radius: 8px; 
                    font-weight: 600; 
                    font-size: 14px;
                    transition: all 0.3s ease;
                    border: 2px solid transparent;
                }
                .status-bar.ready { 
                    background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%); 
                    color: #2e7d32;
                    border-color: #81c784;
                }
                .status-bar.running { 
                    background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%); 
                    color: #1565c0;
                    border-color: #64b5f6;
                }
                .status-bar.completed { 
                    background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%); 
                    color: #ef6c00;
                    border-color: #ffb74d;
                }
                .status-bar.error { 
                    background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%); 
                    color: #c62828;
                    border-color: #ef5350;
                }
                
                /* 动画效果 */
                @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
                .fade-in { animation: fadeIn 0.5s ease forwards; }
                
                /* 加载动画 */
                .loading-spinner { display: inline-block; width: 20px; height: 20px; border: 2px solid #f3f3f3; border-top: 2px solid #2196f3; border-radius: 50%; animation: spin 1s linear infinite; }
                @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
                
                /* 标签页样式 */
                .gr-tabitem {
                    border-radius: 8px 8px 0 0 !important;
                }
                
                /* 滑块样式 */
                .gr-slider input[type="range"] { 
                    accent-color: #2196f3 !important;
                    height: 8px !important;
                    border-radius: 4px !important;
                }
                
                /* 视频容器 */
                .video-container { 
                    border-radius: 12px; 
                    overflow: hidden;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                }
                
                /* 报告页面样式 */
                .score-display input {
                    font-size: 48px !important;
                    font-weight: bold !important;
                    text-align: center !important;
                    color: #2196f3 !important;
                    background: linear-gradient(135deg, #e3f2fd 0%, #f3e5f5 100%) !important;
                    border: 3px solid #2196f3 !important;
                    border-radius: 16px !important;
                    padding: 20px !important;
                }
                
                .recommendation-display input {
                    font-size: 24px !important;
                    font-weight: bold !important;
                    text-align: center !important;
                    border-radius: 12px !important;
                    padding: 15px !important;
                }
                
                .strengths-box {
                    background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%) !important;
                    border-left: 5px solid #4caf50 !important;
                    border-radius: 12px !important;
                    padding: 20px !important;
                    font-size: 14px !important;
                    line-height: 1.8 !important;
                }
                
                .improvements-box {
                    background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%) !important;
                    border-left: 5px solid #ff9800 !important;
                    border-radius: 12px !important;
                    padding: 20px !important;
                    font-size: 14px !important;
                    line-height: 1.8 !important;
                }
                
                .report-json {
                    background: #f8f9fa !important;
                    border-radius: 12px !important;
                    padding: 15px !important;
                }
                
                /* 维度分数颜色 */
                .score-excellent { color: #4caf50 !important; font-weight: bold; }
                .score-good { color: #8bc34a !important; font-weight: bold; }
                .score-pass { color: #ff9800 !important; font-weight: bold; }
                .score-fail { color: #f44336 !important; font-weight: bold; }
                """
            ) as app:
                
                gr.Markdown("""
                # 🤖 锐聘AI - 智能面试系统
                ## 完整面试流程：AI提问 → 你回答 → 实时评估 → 多轮对话 → 最终评分
                """)
                
                # 状态栏
                with gr.Row():
                    status_text = gr.Textbox(
                        value="● 就绪 | 请点击「开始面试」",
                        label="系统状态",
                        interactive=False,
                        elem_classes=["status-bar", "ready"]
                    )
                    self.ui_components['status'] = status_text
                    
                    progress_text = gr.Textbox(
                        value="未开始",
                        label="面试进度",
                        interactive=False
                    )
                    self.ui_components['progress'] = progress_text
                    
                    time_text = gr.Textbox(
                        value="00:00",
                        label="面试时长",
                        interactive=False
                    )
                    self.ui_components['time'] = time_text
                
                with gr.Tabs() as tabs:
                    # ========== 面试界面 ==========  
                    with gr.TabItem("🎤 面试"):
                        with gr.Row(elem_classes=["main-container"]):
                            # 左侧：配置和输入区域
                            with gr.Column(scale=1, elem_classes=["sidebar"]):
                                gr.Markdown("### ⚙️ 面试配置")
                                
                                with gr.Group(elem_classes=["input-section"]):
                                    gr.Markdown("👤 **候选人信息**")
                                    candidate_name = gr.Textbox(
                                        label="姓名",
                                        placeholder="请输入您的姓名",
                                        value=""
                                    )
                                    
                                    position = gr.Dropdown(
                                        label="面试岗位",
                                        choices=[
                                            "Java后端开发",
                                            "Web前端开发",
                                            "Python算法工程师"
                                        ],
                                        value="Java后端开发"
                                    )
                                
                                with gr.Group(elem_classes=["input-section"]):
                                    gr.Markdown("📋 **题目设置**")
                                    with gr.Row():
                                        with gr.Column(scale=1):
                                            num_technical = gr.Slider(
                                                label="技术",
                                                minimum=1, maximum=8, value=3, step=1
                                            )
                                        with gr.Column(scale=1):
                                            num_project = gr.Slider(
                                                label="项目",
                                                minimum=0, maximum=4, value=1, step=1
                                            )
                                        with gr.Column(scale=1):
                                            num_behavioral = gr.Slider(
                                                label="行为",
                                                minimum=0, maximum=4, value=1, step=1
                                            )
                                    
                                    enable_follow_up = gr.Checkbox(
                                        label="启用智能追问",
                                        value=True
                                    )
                                
                                # 控制按钮
                                with gr.Row():
                                    start_btn = gr.Button("▶️ 开始面试", variant="primary", size="lg")
                                    end_btn = gr.Button("⏹️ 结束面试", variant="stop", size="lg")
                            
                            # 中间：对话区域
                            with gr.Column(scale=2, elem_classes=["main-content"]):
                                gr.Markdown("### 💭 面试对话")
                                
                                # 当前问题显示
                                current_question = gr.Textbox(
                                    label="当前问题",
                                    value="请点击「开始面试」开始",
                                    interactive=False,
                                    lines=3,
                                    elem_classes=["question-box"]
                                )
                                self.ui_components['current_question'] = current_question
                                
                                # 对话历史
                                chat_history_text = gr.Textbox(
                                    label="对话记录",
                                    interactive=False,
                                    lines=20
                                )
                                self.ui_components['chat_history_text'] = chat_history_text
                                
                                # 评估反馈
                                with gr.Group(elem_classes=["evaluation-box"]):
                                    evaluation_feedback = gr.Textbox(
                                        label="💡 评估反馈",
                                        value="",
                                        interactive=False,
                                        lines=2
                                    )
                                    self.ui_components['evaluation_feedback'] = evaluation_feedback
                            
                            # 右侧：多模态输入和评估
                            with gr.Column(scale=1):
                                gr.Markdown("### 📥 回答输入")
                                
                                # 输入模式选择
                                with gr.Group(elem_classes=["input-section"]):
                                    gr.Markdown("🎙️ **输入模式**")
                                    input_mode = gr.Radio(
                                        label="选择输入方式",
                                        choices=[
                                            "仅文字",
                                            "仅音频",
                                            "视频+音频(同步)"
                                        ],
                                        value="仅文字",
                                        info="选择视频会自动同步录制音频"
                                    )
                                
                                # 视频输入（选择视频模式时显示）
                                with gr.Group(elem_classes=["input-section"], visible=False) as video_group:
                                    gr.Markdown("📹 **视频**（含音频同步）")
                                    video_input = gr.Video(
                                        label="摄像头录制",
                                        sources=["webcam"],
                                        format="mp4",
                                        height=240,
                                        elem_classes=["video-container"]
                                    )
                                
                                # 音频输入（选择音频/视频模式时显示）
                                with gr.Group(elem_classes=["input-section"], visible=False) as audio_group:
                                    gr.Markdown("🎙️ **音频**")
                                    audio_input = gr.Audio(
                                        label="麦克风",
                                        sources=["microphone"],
                                        type="filepath"
                                    )
                                
                                # 文本输入（始终显示，选填）
                                with gr.Group(elem_classes=["input-section"]):
                                    gr.Markdown("📝 **文字回答**（选填）")
                                    text_input = gr.Textbox(
                                        label="请输入回答（可选）",
                                        placeholder="可在此输入文字补充说明，也可仅使用音视频...",
                                        lines=4
                                    )
                                
                                # 提交按钮
                                submit_btn = gr.Button(
                                    "💬 提交回答",
                                    variant="primary",
                                    size="lg",
                                    interactive=False
                                )
                                
                                clear_btn = gr.Button(
                                    "🔄 清空输入",
                                    variant="secondary"
                                )
                                
                                gr.Markdown("---")
                                
                                # 实时转录面板
                                with gr.Group(elem_classes=["input-section"]):
                                    gr.Markdown("🎤 **实时转录**")
                                    
                                    rt_status = gr.Textbox(
                                        label="连接状态",
                                        value="未连接",
                                        interactive=False
                                    )
                                    
                                    rt_transcription = gr.Textbox(
                                        label="实时字幕",
                                        value="",
                                        placeholder="开始录音后将显示实时转录...",
                                        interactive=False,
                                        lines=3
                                    )
                                    
                                    with gr.Row():
                                        rt_start_btn = gr.Button(
                                            "🎙️ 开始实时转录",
                                            variant="secondary",
                                            size="sm"
                                        )
                                        rt_stop_btn = gr.Button(
                                            "⏹️ 停止转录",
                                            variant="secondary",
                                            size="sm"
                                        )
                                    
                                    gr.Markdown("""
                                    <div style="font-size: 12px; color: #666; margin-top: 8px;">
                                    💡 基于 WebSocket 的实时语音转文字，支持断点续传。
                                    </div>
                                    """)
                                
                                gr.Markdown("---")
                                
                                # 心率监测面板
                                with gr.Group(elem_classes=["input-section"]):
                                    gr.Markdown("❤️ **心率监测**")
                                    
                                    with gr.Row():
                                        heart_rate_display = gr.Number(
                                            label="当前心率",
                                            value=0,
                                            interactive=False,
                                            elem_classes=["heart-rate-display"]
                                        )
                                        heart_rate_status = gr.Textbox(
                                            label="状态",
                                            value="未开始",
                                            interactive=False
                                        )
                                    
                                    # 心率历史图表
                                    heart_rate_plot = gr.LinePlot(
                                        label="心率变化趋势",
                                        x="时间",
                                        y="心率",
                                        height=150,
                                        interactive=False
                                    )
                                    
                                    with gr.Row():
                                        start_hr_btn = gr.Button(
                                            "▶️ 开始监测",
                                            variant="secondary",
                                            size="sm"
                                        )
                                        stop_hr_btn = gr.Button(
                                            "⏹️ 停止监测",
                                            variant="secondary",
                                            size="sm"
                                        )
                                    
                                    gr.Markdown("""
                                    <div style="font-size: 12px; color: #666; margin-top: 8px;">
                                    💡 实时监测候选人面试过程中的心率变化，辅助评估紧张程度。
                                    </div>
                                    """)
                                
                                gr.Markdown("---")
                                
                                # 评估反馈区域（仅显示文字反馈，不显示实时分数）
                                with gr.Group(elem_classes=["input-section"]):
                                    gr.Markdown("### 💬 评估反馈")
                                    
                                    evaluation_feedback = gr.Textbox(
                                        label="",
                                        value="等待回答后开始评估...",
                                        interactive=False,
                                        lines=3
                                    )
                    
                    # ========== 面试报告标签页 ==========  
                    with gr.TabItem("📊 面试报告"):
                        # 报告头部 - 总体评分
                        with gr.Row():
                            with gr.Column(scale=2):
                                gr.Markdown("## 📊 面试评估报告")
                                
                                with gr.Row():
                                    with gr.Column():
                                        final_score = gr.Number(
                                            label="综合得分",
                                            value=0,
                                            interactive=False,
                                            elem_classes=["score-display"]
                                        )
                                    with gr.Column():
                                        recommendation = gr.Textbox(
                                            label="推荐等级",
                                            interactive=False,
                                            elem_classes=["recommendation-display"]
                                        )
                                    with gr.Column():
                                        report_status = gr.Textbox(
                                            label="面试状态",
                                            value="未开始",
                                            interactive=False
                                        )
                            
                            with gr.Column(scale=1):
                                # 面试统计信息
                                with gr.Group():
                                    gr.Markdown("### 📈 面试统计")
                                    interview_duration = gr.Textbox(
                                        label="面试时长",
                                        value="-",
                                        interactive=False
                                    )
                                    questions_count = gr.Textbox(
                                        label="问题数量",
                                        value="-",
                                        interactive=False
                                    )
                                    follow_up_count = gr.Textbox(
                                        label="追问次数",
                                        value="-",
                                        interactive=False
                                    )
                        
                        gr.Markdown("---")
                        
                        # 各维度得分雷达图/柱状图
                        with gr.Row():
                            with gr.Column():
                                gr.Markdown("### 📊 能力维度分析")
                                dimension_chart = gr.BarPlot(
                                    label="各维度得分",
                                    x="维度",
                                    y="得分",
                                    height=300,
                                    interactive=False
                                )
                            
                            with gr.Column():
                                gr.Markdown("### 📝 维度详情")
                                dimension_details = gr.Dataframe(
                                    headers=["维度", "得分", "等级", "权重"],
                                    label="详细评分",
                                    interactive=False
                                )
                        
                        gr.Markdown("---")
                        
                        # 评价详情
                        with gr.Row():
                            with gr.Column():
                                gr.Markdown("### ✅ 主要优点")
                                strengths = gr.Textbox(
                                    label="",
                                    lines=6,
                                    interactive=False,
                                    elem_classes=["strengths-box"]
                                )
                            
                            with gr.Column():
                                gr.Markdown("### 💡 改进建议")
                                improvements = gr.Textbox(
                                    label="",
                                    lines=6,
                                    interactive=False,
                                    elem_classes=["improvements-box"]
                                )
                        
                        gr.Markdown("---")
                        
                        # 完整报告数据和操作
                        with gr.Row():
                            with gr.Column(scale=2):
                                gr.Markdown("### 📋 完整报告数据")
                                report_json = gr.JSON(
                                    label="",
                                    elem_classes=["report-json"]
                                )
                            
                            with gr.Column(scale=1):
                                gr.Markdown("### 💾 操作")
                                with gr.Group():
                                    export_btn = gr.Button(
                                        "📥 导出面试报告",
                                        variant="primary",
                                        size="lg"
                                    )
                                    export_file = gr.File(
                                        label="下载报告文件",
                                        interactive=False
                                    )
                                    
                                    gr.Markdown("""
                                    <div style="margin-top: 10px; padding: 10px; background: #f5f5f5; border-radius: 8px; font-size: 12px; color: #666;">
                                    💡 导出报告包含完整的面试数据、评分详情和建议，可用于存档或分享。
                                    </div>
                                    """)
                    
                    # ========== 使用说明 ==========  
                    with gr.TabItem("❓ 使用说明"):
                        gr.Markdown("""
                        ## 使用指南
                        
                        ### 面试流程
                        1. **配置面试**：选择岗位、设置问题数量、输入姓名
                        2. **开始面试**：点击「开始面试」按钮
                        3. **回答问题**：
                           - 查看左侧显示的当前问题
                           - 通过文字、音频、视频任意一种或多种方式回答
                           - 点击「提交回答」
                        4. **查看反馈**：系统会实时评估并给出反馈
                        5. **继续对话**：根据面试官的下一个问题继续回答
                        6. **结束面试**：完成所有问题后查看最终报告
                        
                        ### 面试阶段
                        - **自我介绍**：开场介绍
                        - **技术问答**：核心技术问题
                        - **项目经验**：项目相关问题
                        - **行为面试**：软技能问题
                        - **候选人提问**：你可以向面试官提问
                        
                        ### 评分维度
                        - **技术能力**：回答的技术准确性
                        - **表达能力**：语言组织和清晰度
                        - **回答完整度**：内容是否全面
                        - **问题解决**：分析和解决问题的能力
                        - **团队协作**：团队合作和沟通能力
                        - **领导力**：领导能力和项目管理能力
                        """)
                
                # ========== 事件绑定 ==========  
                
                # 输入模式切换
                def on_input_mode_change(mode):
                    """根据输入模式显示/隐藏相应的输入组件"""
                    if mode == "仅文字":
                        return gr.Group(visible=False), gr.Group(visible=False)
                    elif mode == "仅音频":
                        return gr.Group(visible=False), gr.Group(visible=True)
                    elif mode == "视频+音频(同步)":
                        return gr.Group(visible=True), gr.Group(visible=True)
                    return gr.Group(visible=False), gr.Group(visible=False)
                
                input_mode.change(
                    fn=on_input_mode_change,
                    inputs=[input_mode],
                    outputs=[video_group, audio_group]
                )
                
                # 开始面试
                start_btn.click(
                    fn=self.start_interview,
                    inputs=[candidate_name, position, num_technical, num_project, num_behavioral, enable_follow_up],
                    outputs=[chat_history_text, current_question, progress_text, status_text, time_text, submit_btn]
                )
                
                # 提交回答 - 根据输入模式处理
                def submit_with_mode(text, video_path, audio_path, mode):
                    """根据输入模式提交回答"""
                    # 根据模式设置音频/视频标志
                    if mode == "仅文字":
                        # 仅文字模式：视频和音频都为空
                        return self.process_answer(text, None, None)
                    elif mode == "仅音频":
                        # 仅音频模式：视频为空，音频根据输入
                        has_audio = audio_path is not None and audio_path != ""
                        return self.process_answer(text, None, audio_path if has_audio else None)
                    elif mode == "视频+音频(同步)":
                        # 视频模式：视频和音频同步（视频包含音频）
                        has_video = video_path is not None and video_path != ""
                        # 如果开启了视频，自动标记为有音频
                        return self.process_answer(text, video_path, audio_path if has_video else None)
                    else:
                        return self.process_answer(text, video_path, audio_path)
                
                submit_btn.click(
                    fn=submit_with_mode,
                    inputs=[text_input, video_input, audio_input, input_mode],
                    outputs=[chat_history_text, current_question, evaluation_feedback, progress_text,
                            status_text, time_text, text_input, audio_input, tabs]
                )
                
                # 清空输入 - 同时清空所有输入
                def clear_all_inputs():
                    return "", None, None
                
                clear_btn.click(
                    fn=clear_all_inputs,
                    outputs=[text_input, audio_input, video_input]
                )
                
                # 结束面试
                end_btn.click(
                    fn=self.end_interview,
                    outputs=[chat_history_text, current_question, final_score, recommendation, report_status,
                            strengths, improvements, report_json, status_text, progress_text, time_text, tabs]
                )
                
                # 导出报告
                export_btn.click(fn=self.export_report, outputs=[export_file])
                
                # 实时转录控制
                rt_start_btn.click(
                    fn=self.start_realtime_transcription,
                    outputs=[rt_status, rt_transcription]
                )
                
                rt_stop_btn.click(
                    fn=self.stop_realtime_transcription,
                    outputs=[rt_status, rt_transcription]
                )
                
                # 心率监测控制
                start_hr_btn.click(
                    fn=self.start_heart_rate_monitoring,
                    outputs=[heart_rate_status, heart_rate_plot]
                )
                
                stop_hr_btn.click(
                    fn=self.stop_heart_rate_monitoring,
                    outputs=[heart_rate_status, heart_rate_plot]
                )
                
                # 定时更新心率显示（每 2 秒更新一次）
                def update_hr_display():
                    hr, status, plot_data = self.get_current_heart_rate()
                    return hr, status, plot_data
                
                # 使用 gr.Timer 定期更新（Gradio 5.x 支持）
                try:
                    timer = gr.Timer(2, active=True)
                    timer.tick(
                        fn=update_hr_display,
                        outputs=[heart_rate_display, heart_rate_status, heart_rate_plot]
                    )
                except:
                    # 如果 Timer 不可用，使用按钮手动刷新
                    refresh_hr_btn = gr.Button("🔄 刷新心率", size="sm", variant="secondary")
                    refresh_hr_btn.click(
                        fn=update_hr_display,
                        outputs=[heart_rate_display, heart_rate_status, heart_rate_plot]
                    )
            
            logger.info("[OK] Gradio界面创建完成")
            return app
            
        except Exception as e:
            logger.error(f"创建Gradio界面时发生错误: {e}", exc_info=True)
            raise
    
    @log_function_call()
    def start_interview(self, candidate_name, position, num_technical, num_project, num_behavioral, enable_follow_up):
        """开始面试"""
        try:
            logger.info(f"开始面试: 候选人={candidate_name or '匿名'}, 岗位={position}")
            
            self.engine.create_session(
                position=position, candidate_name=candidate_name or "匿名候选人",
                num_technical=num_technical, num_project=num_project,
                num_behavioral=num_behavioral, enable_follow_up=enable_follow_up
            )
            
            result = self.engine.start_interview()
            
            if "error" in result:
                logger.error(f"开始面试失败: {result['error']}")
                status_msg = f"● 错误: {result['error'][:50]}"
                status_class = "status-bar error"
                return self._format_chat_history(), result['error'], "错误", gr.Textbox(value=status_msg, elem_classes=[status_class]), "00:00", gr.Button(interactive=False)
            
            self.is_running = True
            self.is_waiting_for_answer = True
            self.start_time = datetime.now()
            self.elapsed_seconds = 0
            self.chat_history = []
            
            self.chat_history.append((None, result['full_message']))
            
            self._start_timer()
            
            status_msg = f"● 面试进行中 | 等待回答..."
            status_class = "status-bar running"
            
            logger.info(f"[OK] 面试开始成功: {result['session_id']}")
            
            return self._format_chat_history(), result['full_message'], result['progress'], gr.Textbox(value=status_msg, elem_classes=[status_class]), "00:00", gr.Button(interactive=True)
            
        except Exception as e:
            logger.error(f"开始面试时发生错误: {e}", exc_info=True)
            status_msg = f"● 错误: {str(e)[:50]}"
            status_class = "status-bar error"
            return f"开始面试失败: {str(e)}", f"错误: {str(e)}", "错误", gr.Textbox(value=status_msg, elem_classes=[status_class]), "00:00", gr.Button(interactive=False)
    
    @log_function_call()
    def process_answer(self, text, video_path, audio_path):
        """处理候选人回答 - 增强版（含安全验证）"""
        try:
            if not self.is_running:
                logger.warning("面试未开始，无法处理回答")
                status_msg = "● 就绪 | 请点击「开始面试」"
                status_class = "status-bar ready"
                return self._format_chat_history(), "请先点击「开始面试」", "", "未开始", gr.Textbox(value=status_msg, elem_classes=[status_class]), "00:00", text, audio_path, gr.Tabs(selected="🎤 面试")
            
            # 速率限制检查
            if SECURITY_AVAILABLE and check_rate_limit:
                client_id = self.engine.session.session_id if self.engine.session else "unknown"
                allowed, remaining, reset_time = check_rate_limit(client_id, "submit_answer")
                if not allowed:
                    logger.warning(f"速率限制触发: {client_id}")
                    status_msg = f"● 面试进行中 | 提交过于频繁，请稍后再试"
                    status_class = "status-bar running"
                    return self._format_chat_history(), "提交过于频繁，请稍后再试", "", "等待", gr.Textbox(value=status_msg, elem_classes=[status_class]), self._format_time(self.elapsed_seconds), text, audio_path, gr.Tabs(selected="🎤 面试")
            
            # 检查是否有任何输入（文字、音频或视频）
            has_audio = audio_path is not None and audio_path != ""
            has_video = video_path is not None and video_path != ""
            has_text = text and text.strip()
            
            if not has_text and not has_audio and not has_video:
                logger.warning("回答为空")
                status_msg = f"● 面试进行中 | 时间: {self._format_time(self.elapsed_seconds)} | 请输入回答或上传音视频"
                status_class = "status-bar running"
                return self._format_chat_history(), "请输入回答内容或上传音视频", "", "未开始", gr.Textbox(value=status_msg, elem_classes=[status_class]), self._format_time(self.elapsed_seconds), text, audio_path, gr.Tabs(selected="🎤 面试")
            
            # 输入安全验证
            if SECURITY_AVAILABLE and validate_and_sanitize_answer and has_text:
                is_valid, sanitized_text, error = validate_and_sanitize_answer(text)
                if not is_valid:
                    logger.warning(f"输入验证失败: {error}")
                    # 记录安全事件但不阻止（避免误杀正常输入）
                else:
                    text = sanitized_text
                    logger.info("[OK] 输入验证通过")
            
            logger.info(f"处理回答: 文字={has_text}, 视频={has_video}, 音频={has_audio}")

            # ========== Whisper 音频转文字 ==========
            transcribed_text = ""
            transcription_info = None
            if (has_audio or has_video) and self.audio_processor and self.audio_processor.is_available():
                logger.info("检测到音频/视频输入，开始 Whisper 转录...")
                try:
                    audio_result = self.audio_processor.process_audio_input(
                        audio_path=audio_path if has_audio else None,
                        video_path=video_path if has_video else None,
                        text_input=text if has_text else None,
                        language="zh"
                    )
                    if audio_result["has_transcription"]:
                        transcribed_text = audio_result["transcribed_text"]
                        transcription_info = audio_result["transcription_info"]
                        logger.info(f"[OK] Whisper 转录完成: {len(transcribed_text)} 字符")
                    else:
                        logger.warning("Whisper 转录未返回结果")
                except Exception as e:
                    logger.error(f"Whisper 转录失败: {e}", exc_info=True)

            # 合并文字：优先使用转录文字，如果没有则使用用户输入
            final_text = transcribed_text if transcribed_text else (text or "")

            # 从视频中提取一帧用于评估
            video_frame = None
            if has_video:
                video_frame = self._extract_frame_from_video(video_path)

            # 构建输入摘要（显示在对话历史中）
            input_summary = final_text if final_text else "[仅音视频输入]"
            if has_audio:
                input_summary += " [含音频]"
            if has_video:
                input_summary += " [含视频]"
            if transcribed_text:
                input_summary += " [已转录]"

            self.chat_history.append((input_summary, None))

            # 显示加载状态
            status_msg = f"● 面试进行中 | 时间: {self._format_time(self.elapsed_seconds)} | 评估中..."
            status_class = "status-bar running"

            # ========== 声纹分析 ==========
            voice_analysis_report = None
            if (has_audio or has_video) and self.voice_analyzer:
                logger.info("开始声纹分析...")
                try:
                    import wave
                    # 读取音频数据进行声纹分析
                    actual_audio_path = audio_path if has_audio else video_path
                    if actual_audio_path and Path(actual_audio_path).exists():
                        with wave.open(actual_audio_path, 'rb') as wav_file:
                            audio_sr = wav_file.getframerate()
                            n_channels = wav_file.getnchannels()
                            n_frames = wav_file.getnframes()
                            raw_data = wav_file.readframes(n_frames)
                            audio_data = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
                            if n_channels == 2:
                                audio_data = audio_data.reshape(-1, 2).mean(axis=1)
                        
                        voice_analysis_report = self.voice_analyzer.analyze(
                            audio=audio_data,
                            sr=audio_sr,
                            transcribed_text=final_text
                        )
                        
                        # 保存声纹分析结果
                        current_turn_idx = len(self.engine.session.dialogue_turns) - 1 if self.engine.session else 0
                        self.voice_analysis_results[current_turn_idx] = voice_analysis_report
                        
                        logger.info(f"[OK] 声纹分析完成 | 综合评分: {voice_analysis_report.get('overall_score', 0):.1f}")
                        
                        # 将声纹分析反馈追加到评估反馈
                        if voice_analysis_report:
                            va_summary = voice_analysis_report.get('summary', '')
                            if va_summary:
                                feedback_extra = f"\n\n🎤 声纹分析: {va_summary}"
                                logger.info(f"[VoiceAnalysis] {va_summary}")
                except Exception as e:
                    logger.error(f"声纹分析失败: {e}", exc_info=True)

            with LogContext(logger, "处理回答"):
                result = self.engine.process_answer(
                    answer=final_text,
                    has_audio=has_audio,
                    has_video=has_video,
                    video_frame=video_frame,
                    audio_path=audio_path,
                    transcribed_text=transcribed_text,
                    transcription_info=transcription_info
                )
            
            if "error" in result:
                logger.error(f"处理回答失败: {result['error']}")
                status_msg = f"● 错误: {result['error'][:50]}"
                status_class = "status-bar error"
                return self._format_chat_history(), f"处理失败: {result['error']}", "", "0/0", gr.Textbox(value=status_msg, elem_classes=[status_class]), self._format_time(self.elapsed_seconds), text, audio_path, gr.Tabs(selected="🎤 面试")
            
            evaluation = result.get('evaluation', {})
            feedback = evaluation.get('feedback', '')
            
            # 追加声纹分析反馈
            if voice_analysis_report:
                va_summary = voice_analysis_report.get('summary', '')
                if va_summary and va_summary not in feedback:
                    feedback += f"\n\n🎤 声纹分析: {va_summary}"
            
            if result.get('type') == 'closing':
                self.chat_history.append((None, result.get('message', '面试结束')))
                self.is_running = False
                self._stop_timer()
                logger.info("面试结束")
                return self._prepare_report_result(result, feedback)
            else:
                self.chat_history.append((None, result.get('question', '请继续')))
            
            status_msg = f"● 面试进行中 | 时间: {self._format_time(self.elapsed_seconds)} | 等待回答..."
            
            logger.info(f"[OK] 回答处理完成: 进度={result.get('progress', '0/0')}")
            
            return (self._format_chat_history(), result.get('question', ''), feedback, 
                   result.get('progress', '0/0'),
                   gr.Textbox(value=status_msg, elem_classes=[status_class]), self._format_time(self.elapsed_seconds), 
                   "", None, gr.Tabs(selected="🎤 面试"))
            
        except Exception as e:
            logger.error(f"处理回答时发生错误: {e}", exc_info=True)
            status_msg = f"● 错误: {str(e)[:50]}"
            status_class = "status-bar error"
            return (self._format_chat_history(), "处理失败，请重试", f"错误: {str(e)[:100]}", 
                   "0/0", gr.Textbox(value=status_msg, elem_classes=[status_class]), 
                   self._format_time(self.elapsed_seconds), text, audio_path, gr.Tabs(selected="🎤 面试"))
    
    def _prepare_report_result(self, result, feedback):
        """准备报告结果 - 增强版"""
        try:
            status_msg = "● 面试已完成"
            status_class = "status-bar completed"
            
            logger.info("准备报告结果: 面试已完成")
            
            # 生成完整报告
            report = self.engine.generate_report()
            
            # 准备维度图表数据
            dimension_data = []
            for dim, score in report.get('dimension_scores', {}).items():
                dim_name = {
                    'technical': '技术能力',
                    'communication': '表达能力',
                    'completeness': '完整度',
                    'problem_solving': '问题解决',
                    'teamwork': '团队协作',
                    'leadership': '领导力'
                }.get(dim, dim)
                dimension_data.append({'维度': dim_name, '得分': score})
            
            # 准备维度详情数据
            dimension_details_data = []
            dim_levels = report.get('dimension_levels', {})
            weights = report.get('weights', {})
            for dim, score in report.get('dimension_scores', {}).items():
                dim_name = {
                    'technical': '技术能力',
                    'communication': '表达能力',
                    'completeness': '完整度',
                    'problem_solving': '问题解决',
                    'teamwork': '团队协作',
                    'leadership': '领导力'
                }.get(dim, dim)
                level_info = dim_levels.get(dim, {})
                weight = weights.get(dim, 0)
                dimension_details_data.append([
                    dim_name,
                    f"{score:.1f}",
                    level_info.get('level', '-'),
                    f"{weight*100:.0f}%"
                ])
            
            return {
                'chat_history': self._format_chat_history(),
                'next_question': "面试已结束，请查看报告标签页",
                'feedback': feedback,
                'progress': "完成",
                'status': gr.Textbox(value=status_msg, elem_classes=[status_class]),
                'time': self._format_time(self.elapsed_seconds),
                'text_input': "",
                'audio_input': None,
                'tabs': gr.Tabs(selected="📊 面试报告"),
                # 报告数据
                'final_score': report.get('overall_score', 0),
                'recommendation': report.get('recommendation', '-'),
                'report_status': "已完成",
                'interview_duration': report.get('duration', '-'),
                'questions_count': str(report.get('total_questions', 0)),
                'follow_up_count': str(report.get('total_follow_ups', 0)),
                'dimension_chart': dimension_data,
                'dimension_details': dimension_details_data,
                'strengths': "\n".join(report.get('strengths', [])),
                'improvements': "\n".join(report.get('improvements', [])),
                'report_json': report
            }
        except Exception as e:
            logger.error(f"准备报告结果时发生错误: {e}", exc_info=True)
            return (self._format_chat_history(), "报告生成失败", "", "错误",
                   gr.Textbox(value="● 错误", elem_classes=["status-bar error"]), "00:00",
                   "", None, gr.Tabs(selected="🎤 面试"))
    
    @log_function_call()
    def end_interview(self):
        """结束面试"""
        try:
            if not self.is_running or not self.engine.session:
                logger.warning("面试未开始，无法结束")
                status_msg = "● 就绪 | 请点击「开始面试」"
                status_class = "status-bar ready"
                return self._format_chat_history(), "面试未开始", 0, "-", "未开始", "", "", {}, gr.Textbox(value=status_msg, elem_classes=[status_class]), "未开始", "00:00", gr.Tabs(selected="🎤 面试")
            
            logger.info("手动结束面试")
            
            result = self.engine.process_answer("结束面试")
            
            if result.get('type') != 'closing':
                self.engine.session.status = InterviewStatus.COMPLETED
                report = self.engine.generate_report()
                result = {'type': 'closing', 'message': '面试已手动结束', 'report': report}
            
            self.chat_history.append((None, result.get('message', '面试结束')))
            self.is_running = False
            self._stop_timer()
            
            report = result.get('report', {})
            status_msg = "● 面试已完成"
            status_class = "status-bar completed"
            
            logger.info(f"[OK] 面试结束成功: 综合得分={report.get('overall_score', 0):.1f}")
            
            return (self._format_chat_history(), "面试已结束", report.get('overall_score', 0),
                   report.get('recommendation', '-'), "已完成", "\n".join(report.get('strengths', [])),
                   "\n".join(report.get('improvements', [])), report, gr.Textbox(value=status_msg, elem_classes=[status_class]), 
                   "完成", self._format_time(self.elapsed_seconds), gr.Tabs(selected="📊 面试报告"))
            
        except Exception as e:
            logger.error(f"结束面试时发生错误: {e}", exc_info=True)
            status_msg = f"● 错误: {str(e)[:50]}"
            status_class = "status-bar error"
            return (self._format_chat_history(), f"结束失败: {str(e)}", 0, "错误", "错误", "", "", {},
                   gr.Textbox(value=status_msg, elem_classes=[status_class]), "错误", "00:00", gr.Tabs(selected="🎤 面试"))
    
    @log_function_call()
    def export_report(self):
        """导出报告"""
        try:
            if not self.engine.session:
                logger.warning("无法导出报告: 面试会话不存在")
                return None
            
            logger.info("导出面试报告...")
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            candidate = self.engine.session.candidate_name or "anonymous"
            filename = f"interview_report_{candidate}_{timestamp}.json"
            
            reports_dir = Path("reports")
            reports_dir.mkdir(exist_ok=True)
            
            filepath = reports_dir / filename
            report = self.engine.generate_report()
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            logger.info(f"[OK] 报告导出成功: {filepath}")
            
            return str(filepath)
            
        except Exception as e:
            logger.error(f"导出报告失败: {e}", exc_info=True)
            return None
    
    def _start_timer(self):
        """启动计时器"""
        def timer_loop():
            while self.is_running:
                time.sleep(1)
                self.elapsed_seconds += 1
                # 更新时间显示
                if 'time' in self.ui_components:
                    try:
                        self.ui_components['time'].update(value=self._format_time(self.elapsed_seconds))
                    except Exception:
                        pass
        
        self.timer_thread = threading.Thread(target=timer_loop, daemon=True)
        self.timer_thread.start()
        logger.debug("计时器启动")
    
    def _stop_timer(self):
        """停止计时器"""
        if self.timer_thread:
            self.timer_thread.join(timeout=1)
            logger.debug("计时器停止")
    
    def _format_time(self, seconds):
        """格式化时间"""
        minutes = seconds // 60
        secs = seconds % 60
        return f"{minutes:02d}:{secs:02d}"
    
    def _format_chat_history(self):
        """格式化对话历史"""
        lines = []
        for user_msg, assistant_msg in self.chat_history:
            if user_msg:
                lines.append(f"👤 候选人: {user_msg}")
            if assistant_msg:
                lines.append(f"🤖 面试官: {assistant_msg}")
            lines.append("-" * 40)
        return "\n".join(lines)

    def get_system_info(self):
        """获取系统硬件和推理模式信息"""
        info_lines = ["=" * 50, "系统信息", "=" * 50]

        if self.hardware_info:
            hw = self.hardware_info
            info_lines.append(f"CUDA 可用: {'是' if hw.get('cuda_available') else '否'}")

            if hw.get("cuda_available") and hw.get("gpus"):
                gpu = hw["gpus"][0]
                info_lines.append(f"GPU: {gpu.get('name', '未知')}")
                info_lines.append(f"总显存: {gpu.get('total_memory_gb', 0):.2f} GB")
                info_lines.append(f"可用显存: {gpu.get('free_memory_gb', 0):.2f} GB")
                info_lines.append(f"计算能力: {gpu.get('compute_capability', '未知')}")
        else:
            info_lines.append("硬件信息: 未获取")

        if self.inference_mode:
            mode = self.inference_mode
            info_lines.append(f"推理模式: {mode.get('description', '未知')}")

            config = mode.get("config", {})
            if config:
                info_lines.append(f"量化: {'启用' if config.get('load_in_4bit') else '禁用'}")
                info_lines.append(f"数据类型: {config.get('torch_dtype', 'float32')}")
                info_lines.append(f"最大图像: {config.get('max_image_size', '未限制')}")
                info_lines.append(f"最大回答: {config.get('max_answer_len', '未限制')}")
        else:
            info_lines.append("推理模式: 未配置")

        info_lines.append("=" * 50)
        return "\n".join(info_lines)

    def start_realtime_transcription(self):
        """开始实时转录"""
        try:
            if not REALTIME_TRANSCRIPTION_AVAILABLE:
                return "实时转录模块不可用", ""

            logger.info("[RT_UI] 用户请求开始实时转录")

            # 生成会话ID
            import uuid
            rt_session_id = f"rt_{uuid.uuid4().hex[:8]}"

            # 注意：实际的 WebSocket 连接需要前端 JavaScript 支持
            # 这里返回状态提示用户
            self.ws_connection_status = "connecting"

            return (
                "正在连接...",
                f"实时转录会话已启动: {rt_session_id}\n"
                "请使用支持 WebSocket 的客户端连接 ws://localhost:7872/ws/transcribe\n"
                "或在浏览器中启用实时录音功能。"
            )

        except Exception as e:
            logger.error(f"[RT_UI] 启动实时转录失败: {e}")
            return f"启动失败: {str(e)}", ""

    def stop_realtime_transcription(self):
        """停止实时转录"""
        try:
            if not REALTIME_TRANSCRIPTION_AVAILABLE:
                return "实时转录模块不可用", ""

            logger.info("[RT_UI] 用户请求停止实时转录")
            self.ws_connection_status = "disconnected"
            self.current_transcription = ""

            return "已断开", ""

        except Exception as e:
            logger.error(f"[RT_UI] 停止实时转录失败: {e}")
            return f"停止失败: {str(e)}", ""


def create_app():
    """创建应用实例"""
    try:
        logger.info("创建应用实例...")
        app = InterviewWebApp()
        logger.info("[OK] 应用实例创建完成")
        return app.create_interface()
    except Exception as e:
        logger.error(f"创建应用实例失败: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("锐聘AI - 智能面试系统")
    logger.info("=" * 70)

    try:
        # 先执行硬件检测
        logger.info("正在执行启动前硬件检测...")
        hardware_result = perform_hardware_check()

        # 创建应用实例
        demo = create_app()

        # 显示系统信息
        if hasattr(demo, 'get_system_info'):
            logger.info("\n" + demo.get_system_info())

        logger.info("启动Gradio服务器...")
        demo.launch(**GRADIO_CONFIG)
    except Exception as e:
        logger.error(f"应用启动失败: {e}", exc_info=True)
        sys.exit(1)
