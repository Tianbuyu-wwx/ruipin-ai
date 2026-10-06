"""
锐聘AI - 多模态面试评估器
基于 Qwen3-VL 本地模型，支持文本+图像联合评估
分析表情、眼神、肢体语言等非语言因素
"""

import torch
import numpy as np
from PIL import Image
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
import hashlib
import time
import re
import json

from .logger import logger, log_function_call


class MultimodalEvaluator:
    """
    多模态面试评估器
    整合文本回答质量和视频画面表现进行综合评估
    支持根据硬件条件自动切换推理模式
    """

    # 推理模式配置
    INFERENCE_MODES = {
        "full_4bit": {
            "description": "4-bit 量化模式",
            "min_vram_gb": 6,
            "load_in_4bit": True,
            "torch_dtype": "float16",
            "max_image_size": (448, 448),
            "max_answer_len": 2000,
            "enable_gradient_checkpointing": True
        },
        "full_fp16": {
            "description": "FP16 半精度模式",
            "min_vram_gb": 10,
            "load_in_4bit": False,
            "torch_dtype": "float16",
            "max_image_size": (448, 448),
            "max_answer_len": 2000,
            "enable_gradient_checkpointing": True
        },
        "cpu_offload": {
            "description": "CPU 卸载模式",
            "min_vram_gb": 4,
            "load_in_4bit": True,
            "torch_dtype": "float16",
            "device_map": "auto",
            "max_image_size": (336, 336),
            "max_answer_len": 1500,
            "enable_gradient_checkpointing": True
        },
        "cpu_only": {
            "description": "纯 CPU 模式",
            "min_vram_gb": 0,
            "load_in_4bit": False,
            "torch_dtype": "float32",
            "max_image_size": (336, 336),
            "max_answer_len": 1000,
            "enable_gradient_checkpointing": False
        }
    }

    def __init__(self, model_path: str = "base_model"):
        """初始化多模态评估器 - 自动检测硬件并选择最优推理模式"""
        logger.info("=" * 60)
        logger.info("初始化多模态面试评估器")
        logger.info("=" * 60)

        self.model_path = model_path
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 缓存
        self.evaluation_cache = {}
        self.cache_size = 500
        self.cache_hits = 0
        self.cache_misses = 0

        # 性能统计
        self.inference_times = []

        # 当前推理模式
        self.current_mode = None
        self.mode_config = None

        # 硬件信息
        self.hardware_info = self._detect_hardware()

        logger.info(f"模型路径: {self.model_path}")
        logger.info(f"设备: {self.device}")

        # 自动选择并加载最优推理模式
        self._auto_select_mode()
        self._load_model()

    def __del__(self):
        """析构函数：确保对象销毁时释放显存"""
        try:
            if self.model is not None or self.processor is not None or self.tokenizer is not None:
                logger.info("MultimodalEvaluator 对象销毁，清理模型资源...")
                self._unload_model()
        except Exception:
            pass

    def _detect_hardware(self) -> Dict[str, Any]:
        """检测硬件信息：显卡型号、显存、CUDA版本等"""
        info = {
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "gpu_count": 0,
            "gpus": [],
            "total_vram_gb": 0,
            "free_vram_gb": 0,
            "recommended_mode": "cpu_only"
        }

        if torch.cuda.is_available():
            info["gpu_count"] = torch.cuda.device_count()

            for i in range(info["gpu_count"]):
                props = torch.cuda.get_device_properties(i)
                total_memory = props.total_memory / 1024**3
                free_memory = (props.total_memory - torch.cuda.memory_allocated(i)) / 1024**3

                gpu_info = {
                    "index": i,
                    "name": props.name,
                    "total_memory_gb": round(total_memory, 2),
                    "free_memory_gb": round(free_memory, 2),
                    "compute_capability": f"{props.major}.{props.minor}",
                    "multi_processor_count": props.multi_processor_count
                }
                info["gpus"].append(gpu_info)
                info["total_vram_gb"] += total_memory
                info["free_vram_gb"] += free_memory

            logger.info("=" * 60)
            logger.info("硬件检测报告")
            logger.info("=" * 60)
            logger.info(f"CUDA 版本: {info['cuda_version']}")
            logger.info(f"GPU 数量: {info['gpu_count']}")

            for gpu in info["gpus"]:
                logger.info(f"GPU {gpu['index']}: {gpu['name']}")
                logger.info(f"  总显存: {gpu['total_memory_gb']:.2f} GB")
                logger.info(f"  可用显存: {gpu['free_memory_gb']:.2f} GB")
                logger.info(f"  计算能力: {gpu['compute_capability']}")

            logger.info(f"总显存: {info['total_vram_gb']:.2f} GB")
            logger.info(f"可用显存: {info['free_vram_gb']:.2f} GB")
        else:
            logger.warning("[WARN] CUDA 不可用，将使用 CPU 模式")

        return info

    def _auto_select_mode(self):
        """根据硬件条件自动选择最优推理模式"""
        # 确保 hardware_info 已设置
        if not hasattr(self, 'hardware_info') or self.hardware_info is None:
            self.hardware_info = self._detect_hardware()

        if not self.hardware_info["cuda_available"]:
            new_mode = "cpu_only"
        else:
            free_vram = self.hardware_info["free_vram_gb"]
            gpu_name = self.hardware_info["gpus"][0]["name"].lower() if self.hardware_info["gpus"] else ""

            # 检查是否支持 4-bit 量化 (需要 compute capability >= 6.0)
            supports_4bit = True
            if self.hardware_info["gpus"]:
                major = int(self.hardware_info["gpus"][0]["compute_capability"].split(".")[0])
                if major < 6:
                    supports_4bit = False
                    logger.warning(f"[WARN] GPU 计算能力过低 ({major}.x)，不支持 4-bit 量化")

            # 根据显存大小选择模式
            if free_vram >= 10:
                new_mode = "full_fp16"
            elif free_vram >= 6 and supports_4bit:
                new_mode = "full_4bit"
            elif free_vram >= 4 and supports_4bit:
                new_mode = "cpu_offload"
            else:
                new_mode = "cpu_only"

        # 如果模型已经加载，且新模式与当前模式不同，需要先卸载旧模型
        current_model = getattr(self, 'model', None)
        current_mode = getattr(self, 'current_mode', None)
        if current_model is not None and current_mode != new_mode:
            logger.info(f"[AUTO] 检测到模式变更: {current_mode} -> {new_mode}，卸载旧模型...")
            self._unload_model()

        self.current_mode = new_mode
        self.mode_config = self.INFERENCE_MODES[new_mode]

        if new_mode == "cpu_only" and not self.hardware_info["cuda_available"]:
            logger.info("[AUTO] 选择推理模式: CPU 模式 (无 GPU)")
        elif new_mode == "full_fp16":
            logger.info(f"[AUTO] 选择推理模式: FP16 半精度模式 (可用显存 {free_vram:.1f}GB >= 10GB)")
        elif new_mode == "full_4bit":
            logger.info(f"[AUTO] 选择推理模式: 4-bit 量化模式 (可用显存 {free_vram:.1f}GB >= 6GB)")
        elif new_mode == "cpu_offload":
            logger.info(f"[AUTO] 选择推理模式: CPU 卸载模式 (可用显存 {free_vram:.1f}GB >= 4GB)")
        else:
            logger.info(f"[AUTO] 选择推理模式: 纯 CPU 模式 (可用显存 {free_vram:.1f}GB < 4GB 或不支持量化)")

        logger.info(f"[AUTO] 模式配置: {self.mode_config['description']}")
        logger.info(f"[AUTO] 最大图像尺寸: {self.mode_config['max_image_size']}")
        logger.info(f"[AUTO] 最大回答长度: {self.mode_config['max_answer_len']}")

    def get_hardware_info(self) -> Dict[str, Any]:
        """获取硬件信息"""
        return self.hardware_info

    def get_current_mode(self) -> Dict[str, Any]:
        """获取当前推理模式信息"""
        return {
            "mode": self.current_mode,
            "description": self.mode_config["description"] if self.mode_config else "未初始化",
            "config": self.mode_config
        }

    def _unload_model(self):
        """卸载当前加载的模型，释放显存和内存"""
        if self.model is not None:
            logger.info("正在卸载当前模型...")
            try:
                # 将模型移到 CPU 以便释放显存
                if hasattr(self.model, 'cpu'):
                    self.model.cpu()
                # 删除模型引用
                del self.model
                self.model = None
            except Exception as e:
                logger.warning(f"卸载模型时出错: {e}")

        if self.processor is not None:
            try:
                del self.processor
                self.processor = None
            except Exception as e:
                logger.warning(f"卸载 processor 时出错: {e}")

        if self.tokenizer is not None:
            try:
                del self.tokenizer
                self.tokenizer = None
            except Exception as e:
                logger.warning(f"卸载 tokenizer 时出错: {e}")

        # 强制 Python 垃圾回收
        import gc
        gc.collect()

        # 清理 CUDA 缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            freed = torch.cuda.memory_allocated() / 1024**3
            logger.info(f"[OK] 模型已卸载，显存已清理 | 当前已分配显存: {freed:.2f} GB")

    def switch_mode(self, new_mode: str):
        """切换推理模式：先卸载旧模型，再加载新模型"""
        if new_mode not in self.INFERENCE_MODES:
            raise ValueError(f"不支持的推理模式: {new_mode}，支持的模式: {list(self.INFERENCE_MODES.keys())}")

        if self.current_mode == new_mode:
            logger.info(f"当前已经是 {new_mode} 模式，无需切换")
            return

        logger.info("=" * 60)
        logger.info(f"切换推理模式: {self.current_mode} -> {new_mode}")
        logger.info("=" * 60)

        # 1. 卸载旧模型
        self._unload_model()

        # 2. 更新模式配置
        self.current_mode = new_mode
        self.mode_config = self.INFERENCE_MODES[new_mode]

        # 3. 重新检测硬件（获取最新显存信息）
        self.hardware_info = self._detect_hardware()

        # 4. 加载新模型
        self._load_model()

        logger.info("=" * 60)
        logger.info(f"[OK] 推理模式切换完成: {self.mode_config['description']}")
        logger.info("=" * 60)

    def _load_model(self):
        """加载 Qwen3-VL 模型 - 智能版：根据选择的推理模式加载"""
        try:
            if not Path(self.model_path).exists():
                logger.error(f"模型路径不存在: {self.model_path}")
                raise FileNotFoundError(f"模型路径不存在: {self.model_path}")

            logger.info("开始加载 Qwen3-VL 模型...")
            logger.info(f"当前推理模式: {self.mode_config['description']}")
            start_time = time.time()

            from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, AutoTokenizer

            # 根据模式配置加载模型
            load_in_4bit = self.mode_config.get("load_in_4bit", False)
            torch_dtype_str = self.mode_config.get("torch_dtype", "float16")
            torch_dtype = torch.float16 if torch_dtype_str == "float16" else torch.float32
            device_map = self.mode_config.get("device_map", None)
            enable_gradient_checkpointing = self.mode_config.get("enable_gradient_checkpointing", False)

            if torch.cuda.is_available() and self.current_mode != "cpu_only":
                if load_in_4bit:
                    # 使用 4-bit 量化加载
                    try:
                        from transformers import BitsAndBytesConfig
                        quantization_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=torch_dtype,
                            bnb_4bit_use_double_quant=True,
                            bnb_4bit_quant_type="nf4"
                        )

                        load_kwargs = {
                            "pretrained_model_name_or_path": self.model_path,
                            "torch_dtype": torch_dtype,
                            "quantization_config": quantization_config,
                            "trust_remote_code": True
                        }

                        # CPU 卸载模式使用 device_map="auto"
                        if device_map == "auto":
                            load_kwargs["device_map"] = "auto"

                        self.model = Qwen3VLForConditionalGeneration.from_pretrained(**load_kwargs)
                        logger.info("[OK] 4-bit 量化模型加载成功")
                    except ImportError:
                        logger.warning("bitsandbytes 未安装，回退到 FP16 加载")
                        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                            self.model_path,
                            torch_dtype=torch_dtype,
                            trust_remote_code=True
                        ).to(self.device)
                else:
                    # FP16 模式
                    self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                        self.model_path,
                        torch_dtype=torch_dtype,
                        trust_remote_code=True
                    ).to(self.device)
                    logger.info("[OK] FP16 模型加载成功")
            else:
                # CPU 模式
                self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                    self.model_path,
                    torch_dtype=torch_dtype,
                    trust_remote_code=True
                )
                logger.info("[OK] CPU 模型加载成功")

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )

            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )

            # 启用梯度检查点（如果配置启用）
            if enable_gradient_checkpointing and hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
                logger.info("[OK] 梯度检查点已启用")

            load_time = time.time() - start_time
            logger.info(f"[OK] 模型加载完成，耗时: {load_time:.2f}秒")

            # 记录显存使用情况
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                logger.info(f"[INFO] GPU 显存使用: 已分配={allocated:.2f}GB, 预留={reserved:.2f}GB")

        except Exception as e:
            logger.error(f"[ERROR] 模型加载失败: {e}", exc_info=True)
            self.model = None
            self.processor = None
            self.tokenizer = None

    def is_available(self) -> bool:
        """检查模型是否可用"""
        return self.model is not None and self.processor is not None

    @log_function_call()
    def evaluate(
        self,
        answer: str,
        question: str,
        video_frame: Optional[np.ndarray] = None,
        has_audio: bool = False
    ) -> Dict[str, Any]:
        """
        执行多模态评估

        Args:
            answer: 候选人文字回答
            question: 面试问题
            video_frame: 视频帧 (numpy数组)
            has_audio: 是否有音频输入

        Returns:
            包含文本评估和视频评估的综合结果
        """
        try:
            logger.info("=" * 60)
            logger.info("开始多模态评估")
            logger.info("=" * 60)
            logger.info(f"回答长度: {len(answer) if answer else 0} 字符")
            logger.info(f"有视频帧: {video_frame is not None}")
            logger.info(f"有音频: {has_audio}")

            # 检查缓存
            cache_key = self._generate_cache_key(answer, video_frame)
            cached = self._check_cache(cache_key)
            if cached:
                logger.info("[OK] 缓存命中，返回缓存结果")
                return cached

            # 1. 文本评估
            text_evaluation = self._evaluate_text(answer, question)

            # 2. 视频评估（如果有视频帧）
            video_evaluation = None
            if video_frame is not None and self.is_available():
                video_evaluation = self._evaluate_video_frame(video_frame, answer)

            # 3. 综合评估
            final_evaluation = self._merge_evaluations(
                text_evaluation,
                video_evaluation,
                has_audio
            )

            # 更新缓存
            self._update_cache(cache_key, final_evaluation)

            logger.info("[OK] 多模态评估完成")
            return final_evaluation

        except Exception as e:
            logger.error(f"多模态评估失败: {e}", exc_info=True)
            return self._get_default_evaluation()

    def _evaluate_text(self, answer: str, question: str) -> Dict[str, Any]:
        """评估文本回答质量（优化版）- 限制输入长度并清理显存"""
        try:
            # 始终计算规则评估分数（作为基础）
            rule_based = self._rule_based_text_evaluation(answer)

            if not self.is_available():
                logger.info("模型不可用，使用规则评估")
                return rule_based

            # 限制输入长度，避免显存溢出（使用模式配置的最大长度）
            max_answer_len = self.mode_config.get("max_answer_len", 2000) if self.mode_config else 2000
            max_question_len = 500
            truncated_answer = answer[:max_answer_len] if len(answer) > max_answer_len else answer
            truncated_question = question[:max_question_len] if len(question) > max_question_len else question

            if len(answer) > max_answer_len:
                logger.warning(f"回答长度超过限制({len(answer)}>{max_answer_len})，已截断")

            # 尝试使用模型评估
            prompt = f"""你是一位专业的技术面试官。请评估候选人的面试回答。

面试问题：{truncated_question}

候选人回答：{truncated_answer}

请从以下六个维度进行评分（0-100分），并给出详细反馈：

1. 技术能力：技术概念理解、实践经验、技术深度
2. 表达能力：逻辑清晰、表达流畅、条理分明
3. 回答完整度：内容全面、覆盖要点、无重大遗漏
4. 问题解决：分析思路、解决方案、应变能力
5. 团队协作：合作意识、沟通技巧、冲突处理
6. 领导力：项目把控、团队管理、决策能力

请严格按以下JSON格式返回，不要包含任何其他文字：
{{"technical": <分数>, "communication": <分数>, "completeness": <分数>, "problem_solving": <分数>, "teamwork": <分数>, "leadership": <分数>, "feedback": "详细反馈"}}"""

            inputs = self.processor(text=prompt, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = inputs.to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=400,
                    do_sample=False,
                    repetition_penalty=1.1,
                    num_return_sequences=1
                )

            response = self.processor.decode(outputs[0], skip_special_tokens=True)

            # 立即清理推理占用的显存
            del inputs, outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # 清理响应，移除prompt内容
            response = self._clean_response(response, prompt)
            model_eval = self._parse_json_evaluation(response)

            # 融合模型评估和规则评估（50% 模型 + 50% 规则）
            # 这样既能利用模型的理解能力，又能保证规则评估的稳定性
            merged = {}
            for key in ["technical", "communication", "completeness", "problem_solving", "teamwork", "leadership", "score"]:
                model_val = model_eval.get(key, 60)
                rule_val = rule_based.get(key, 60)
                # 如果模型分数异常（所有维度相同），更信任规则评估
                if self._is_uniform_score(model_eval):
                    merged[key] = round(rule_val * 0.7 + model_val * 0.3, 1)
                else:
                    merged[key] = round(model_val * 0.5 + rule_val * 0.5, 1)

            # 反馈优先使用模型的
            merged["feedback"] = model_eval.get("feedback", rule_based.get("feedback", ""))

            logger.info(f"文本评估融合: 模型={model_eval.get('score')}, 规则={rule_based.get('score')}, 融合={merged.get('score')}")
            return merged

        except Exception as e:
            logger.error(f"文本评估失败: {e}")
            return self._rule_based_text_evaluation(answer)

    def _is_uniform_score(self, evaluation: Dict[str, Any]) -> bool:
        """检查评估分数是否过于均匀（可能是模型未正确理解）"""
        scores = [evaluation.get(k, 0) for k in ["technical", "communication", "completeness", "problem_solving", "teamwork", "leadership"]]
        if len(scores) < 2:
            return False
        return max(scores) - min(scores) < 5  # 所有维度分数差异小于5分

    def _evaluate_video_frame(self, video_frame: np.ndarray, answer: str) -> Dict[str, Any]:
        """
        评估视频帧中的非语言表现 - 优化版：图像压缩和显存清理
        分析表情、眼神、肢体语言等
        """
        try:
            logger.info("=" * 60)
            logger.info("开始视频帧评估")
            logger.info("=" * 60)

            if not self.is_available():
                logger.warning("[WARN] 模型不可用，跳过视频帧评估")
                return None

            # 检查输入数据
            logger.info(f"输入视频帧信息:")
            logger.info(f"  - 类型: {type(video_frame)}")
            logger.info(f"  - 形状: {video_frame.shape}")
            logger.info(f"  - 数据类型: {video_frame.dtype}")

            # 转换 numpy 数组为 PIL Image
            logger.info("正在转换图像格式 (numpy -> PIL)...")
            if isinstance(video_frame, np.ndarray):
                if video_frame.shape[-1] == 3:  # RGB
                    image = Image.fromarray(video_frame.astype(np.uint8))
                else:
                    image = Image.fromarray(video_frame.astype(np.uint8)).convert('RGB')
            else:
                image = video_frame

            # 压缩图像以减少显存占用（使用模式配置的最大尺寸）
            max_image_size = self.mode_config.get("max_image_size", (448, 448)) if self.mode_config else (448, 448)
            if image.size[0] > max_image_size[0] or image.size[1] > max_image_size[1]:
                logger.info(f"图像尺寸过大({image.size})，压缩到{max_image_size}")
                image = image.resize(max_image_size, Image.Resampling.LANCZOS)
                logger.info("[OK] 图像压缩完成")

            logger.info(f"PIL图像信息: 尺寸={image.size}, 模式={image.mode}")

            # 构建视觉评估提示
            visual_prompt = """你是一位专业的面试行为分析师。请仔细观察候选人的视频画面，分析其非语言表现。

请从以下四个维度进行评分（0-100分），并给出详细反馈：

1. 表情管理：面部表情是否自信、自然、积极
   - 观察：是否微笑、是否紧张、表情是否僵硬
   - 高分：表情自然、自信、有亲和力
   - 低分：表情僵硬、紧张、不自然

2. 眼神交流：是否有良好的眼神接触
   - 观察：是否直视镜头、眼神是否游离
   - 高分：眼神坚定、与镜头有良好交流
   - 低分：眼神游离、回避镜头

3. 肢体语言：姿态是否端正、手势是否得当
   - 观察：坐姿/站姿、手势、头部动作
   - 高分：姿态端正、手势自然、有适当动作
   - 低分：姿态懒散、无手势或动作过多

4. 整体形象：着装、精神面貌
   - 观察：着装是否得体、精神面貌是否良好
   - 高分：着装得体、精神饱满
   - 低分：着装随意、精神萎靡

请严格按以下JSON格式返回：
{
    "expression": <表情管理分数>,
    "eye_contact": <眼神交流分数>,
    "body_language": <肢体语言分数>,
    "overall_image": <整体形象分数>,
    "visual_feedback": "详细反馈，描述观察到的具体表现和改进建议"
}"""

            # 使用 Qwen3-VL 处理图像+文本
            logger.info("正在构建多模态输入...")
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": visual_prompt}
                ]
            }]

            logger.info("应用对话模板...")
            text_input = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            logger.info(f"[OK] 对话模板应用完成，文本长度: {len(text_input)} 字符")

            logger.info("正在预处理输入数据...")
            inputs = self.processor(text=[text_input], images=[image], return_tensors="pt", padding=True)
            logger.info(f"[OK] 输入预处理完成")
            logger.info(f"  - 输入键: {list(inputs.keys())}")
            for key, value in inputs.items():
                if hasattr(value, 'shape'):
                    logger.info(f"  - {key} 形状: {value.shape}")

            if torch.cuda.is_available():
                logger.info("将输入数据移至GPU...")
                inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
                logger.info(f"[OK] 数据已移至 {self.device}")

            # 模型推理
            logger.info("开始模型推理（视觉评估）...")
            inference_start = time.time()

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=400,
                    do_sample=False,
                    temperature=0.3,
                    repetition_penalty=1.1,
                    num_return_sequences=1
                )

            inference_time = time.time() - inference_start
            logger.info(f"[OK] 模型推理完成，耗时: {inference_time:.2f}秒")

            # 立即清理推理占用的显存
            del inputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("解码模型输出...")
            response = self.processor.decode(outputs[0], skip_special_tokens=True)

            # 清理输出张量
            del outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info(f"[OK] 解码完成，响应长度: {len(response)} 字符")
            logger.info(f"原始响应前200字符: {response[:200]}")

            # 清理响应
            response = self._clean_response(response)

            # 提取JSON部分
            logger.info("正在解析评估结果...")
            visual_eval = self._parse_json_evaluation(response)

            logger.info("=" * 60)
            logger.info("视频帧评估结果")
            logger.info("=" * 60)
            logger.info(f"  - 表情管理: {visual_eval.get('expression', 'N/A')}")
            logger.info(f"  - 眼神交流: {visual_eval.get('eye_contact', 'N/A')}")
            logger.info(f"  - 肢体语言: {visual_eval.get('body_language', 'N/A')}")
            logger.info(f"  - 整体形象: {visual_eval.get('overall_image', 'N/A')}")
            logger.info(f"  - 视觉反馈: {visual_eval.get('visual_feedback', 'N/A')[:100]}...")
            logger.info("=" * 60)

            return visual_eval

        except Exception as e:
            logger.error(f"[ERROR] 视频帧评估失败: {e}", exc_info=True)
            return None

    def _merge_evaluations(
        self,
        text_eval: Dict[str, Any],
        video_eval: Optional[Dict[str, Any]],
        has_audio: bool
    ) -> Dict[str, Any]:
        """合并文本评估和视频评估"""
        result = dict(text_eval)

        # 如果有视频评估，合并视觉维度
        if video_eval:
            result.update({
                "expression": video_eval.get("expression", 60),
                "eye_contact": video_eval.get("eye_contact", 60),
                "body_language": video_eval.get("body_language", 60),
                "overall_image": video_eval.get("overall_image", 60),
                "visual_feedback": video_eval.get("visual_feedback", "")
            })

            # 调整表达能力分数（结合语言表达和非语言表达）
            verbal_comm = text_eval.get("communication", 60)
            non_verbal = (
                video_eval.get("expression", 60) +
                video_eval.get("eye_contact", 60) +
                video_eval.get("body_language", 60)
            ) / 3

            # 表达能力 = 70% 语言 + 30% 非语言
            result["communication"] = round(verbal_comm * 0.7 + non_verbal * 0.3, 1)

            # 调整完整度分数
            result["completeness"] = min(100, result.get("completeness", 60) + 5)

            # 综合反馈
            text_feedback = text_eval.get("feedback", "")
            visual_feedback = video_eval.get("visual_feedback", "")
            result["feedback"] = f"【回答内容评估】\n{text_feedback}\n\n【非语言表现评估】\n{visual_feedback}"

        else:
            # 无视频评估时的默认值
            result.update({
                "expression": None,
                "eye_contact": None,
                "body_language": None,
                "overall_image": None,
                "visual_feedback": "未提供视频，无法评估非语言表现"
            })

        # 音频加分
        if has_audio:
            result["communication"] = min(100, result.get("communication", 60) + 3)
            result["completeness"] = min(100, result.get("completeness", 60) + 2)

        # 重新计算综合得分（包含视觉维度）
        scores = [
            result.get("technical", 60),
            result.get("communication", 60),
            result.get("completeness", 60),
            result.get("problem_solving", 60),
            result.get("teamwork", 60),
            result.get("leadership", 60)
        ]

        # 如果有视觉评估，加入计算
        if video_eval:
            visual_scores = [
                result.get("expression", 60),
                result.get("eye_contact", 60),
                result.get("body_language", 60),
                result.get("overall_image", 60)
            ]
            # 综合得分 = 70% 文本维度 + 30% 视觉维度
            text_avg = sum(scores) / len(scores)
            visual_avg = sum(visual_scores) / len(visual_scores)
            result["score"] = round(text_avg * 0.7 + visual_avg * 0.3, 1)
        else:
            result["score"] = round(sum(scores) / len(scores), 1)

        return result

    def _rule_based_text_evaluation(self, answer: str) -> Dict[str, Any]:
        """基于规则的文本评估（优化版）- 更敏感地反映内容质量差异"""
        if not answer:
            return {
                "score": 0,
                "technical": 0,
                "communication": 0,
                "completeness": 0,
                "problem_solving": 0,
                "teamwork": 0,
                "leadership": 0,
                "feedback": "未提供回答。"
            }

        length = len(answer)

        # ===== 基础分计算（基于长度） =====
        if length < 10:
            base_score = 15
        elif length < 30:
            base_score = 30
        elif length < 60:
            base_score = 45
        elif length < 120:
            base_score = 60
        elif length < 200:
            base_score = 70
        else:
            base_score = 78

        # ===== 内容质量加分 =====
        quality_bonus = 0

        # 1. 技术深度加分
        tech_indicators = [
            '原理', '机制', '底层', '源码', '算法', '架构', '设计模式',
            '优化', '性能', '并发', '分布式', '微服务', '容器',
            'JVM', 'GC', '线程池', '索引', '事务', '锁',
            'Redis', 'Kafka', 'Docker', 'Kubernetes', 'Spring',
            'MySQL', 'PostgreSQL', 'MongoDB', 'Elasticsearch'
        ]
        tech_count = sum(1 for indicator in tech_indicators if indicator in answer)
        quality_bonus += min(15, tech_count * 3)

        # 2. 实践经验加分
        exp_indicators = [
            '项目', '实际', '生产环境', '线上', '部署', '运维',
            '排查', '定位', '解决', '处理过', '遇到过', '经验'
        ]
        exp_count = sum(1 for indicator in exp_indicators if indicator in answer)
        quality_bonus += min(10, exp_count * 2)

        # 3. 结构化表达加分
        structure_indicators = [
            '首先', '其次', '然后', '最后', '第一', '第二', '第三',
            '1.', '2.', '3.', '①', '②', '③',
            '总结', '综上所述', '因此', '所以'
        ]
        structure_count = sum(1 for indicator in structure_indicators if indicator in answer)
        quality_bonus += min(8, structure_count * 2)

        # 4. 举例说明加分
        example_indicators = ['例如', '比如', '举个例子', '如', 'instance', 'case']
        has_example = any(indicator in answer for indicator in example_indicators)
        if has_example:
            quality_bonus += 5

        # 5. 数据量化加分
        import re
        has_numbers = bool(re.search(r'\d+', answer))
        has_percentage = bool(re.search(r'\d+%', answer))
        if has_numbers:
            quality_bonus += 3
        if has_percentage:
            quality_bonus += 3

        # 6. 问题分析加分
        analysis_indicators = ['分析', '排查', '定位', '原因', '根因', '思路']
        has_analysis = any(indicator in answer for indicator in analysis_indicators)
        if has_analysis:
            quality_bonus += 5

        # 7. 解决方案加分
        solution_indicators = ['方案', '解决', '优化', '改进', '措施', '策略']
        has_solution = any(indicator in answer for indicator in solution_indicators)
        if has_solution:
            quality_bonus += 5

        # 计算最终分数
        final_score = min(100, base_score + quality_bonus)

        # 根据内容质量调整各维度分数
        technical = min(100, final_score * 0.9 + (10 if tech_count > 2 else 0))
        communication = min(100, final_score * 0.8 + (15 if structure_count > 1 else 5))
        completeness = min(100, base_score + (10 if length > 100 else 0))
        problem_solving = min(100, final_score * 0.85 + (10 if has_analysis and has_solution else 0))
        teamwork = min(100, final_score * 0.7 + 15)  # 团队协作默认较高
        leadership = min(100, final_score * 0.7 + 10)  # 领导力默认中等

        # 生成反馈
        if final_score >= 85:
            feedback = "回答优秀！技术深度好，表达清晰，有实际经验支撑。"
        elif final_score >= 70:
            feedback = "回答良好。涵盖了主要内容，但可以更深入一些。"
        elif final_score >= 55:
            feedback = "回答一般。基本理解了问题，但缺乏深度和细节。"
        elif final_score >= 40:
            feedback = "回答较浅。需要更多技术细节和实际经验。"
        else:
            feedback = "回答过于简短或缺乏实质内容。请详细说明。"

        return {
            "score": round(final_score, 1),
            "technical": round(technical, 1),
            "communication": round(communication, 1),
            "completeness": round(completeness, 1),
            "problem_solving": round(problem_solving, 1),
            "teamwork": round(teamwork, 1),
            "leadership": round(leadership, 1),
            "feedback": feedback
        }

    def _clean_response(self, response: str, prompt: str = None) -> str:
        """清理模型响应，移除prompt内容和特殊标记"""
        cleaned = response.strip()

        # 如果响应包含prompt内容，提取最后的部分（模型生成的内容）
        if prompt and prompt in cleaned:
            parts = cleaned.split(prompt)
            if len(parts) > 1:
                cleaned = parts[-1].strip()

        # 移除常见的角色标记
        for prefix in ["assistant", "user", "model", "AI:", "Assistant:"]:
            if cleaned.lower().startswith(prefix.lower()):
                cleaned = cleaned[len(prefix):].strip()
                if cleaned.startswith(":"):
                    cleaned = cleaned[1:].strip()

        # 移除markdown代码块
        if cleaned.startswith('```json'):
            cleaned = cleaned[7:]
        if cleaned.startswith('```'):
            cleaned = cleaned[3:]
        if cleaned.endswith('```'):
            cleaned = cleaned[:-3]

        return cleaned.strip()

    def _parse_json_evaluation(self, response: str) -> Dict[str, Any]:
        """从模型响应中解析JSON评估结果"""
        try:
            cleaned = self._clean_response(response)

            # 尝试直接解析JSON
            try:
                result = json.loads(cleaned)
                return result
            except json.JSONDecodeError:
                pass

            # 尝试从文本中提取JSON（匹配嵌套的大括号）
            json_pattern = r'\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}'
            json_matches = re.findall(json_pattern, cleaned, re.DOTALL)

            for match in json_matches:
                try:
                    result = json.loads(match)
                    if any(k in result for k in ["technical", "expression", "score"]):
                        return result
                except:
                    continue

            # 尝试正则提取分数
            result = {}
            for key in ["technical", "communication", "completeness", "problem_solving",
                       "teamwork", "leadership", "expression", "eye_contact",
                       "body_language", "overall_image"]:
                patterns = [
                    rf'["\']?{key}["\']?\s*[:=]\s*(\d+(?:\.\d+)?)',
                    rf'{key}\s*[:=]\s*(\d+(?:\.\d+)?)',
                ]
                for pattern in patterns:
                    match = re.search(pattern, cleaned, re.IGNORECASE)
                    if match:
                        result[key] = float(match.group(1))
                        break

            # 提取反馈
            feedback_patterns = [
                r'["\']?feedback["\']?\s*[:=]\s*["\']([^"\']+)',
                r'["\']?visual_feedback["\']?\s*[:=]\s*["\']([^"\']+)',
            ]
            for pattern in feedback_patterns:
                feedback_match = re.search(pattern, cleaned, re.IGNORECASE)
                if feedback_match:
                    result["feedback"] = feedback_match.group(1).strip()
                    break

            if "feedback" not in result:
                result["feedback"] = cleaned[:200]

            if not result:
                logger.warning(f"无法从响应中提取评估数据，原始响应: {cleaned[:200]}")
                raise ValueError("无法从响应中提取评估数据")

            return result

        except Exception as e:
            logger.error(f"解析评估响应失败: {e}")
            logger.debug(f"原始响应: {response[:500]}")
            return self._get_default_evaluation()

    def _get_default_evaluation(self) -> Dict[str, Any]:
        """获取默认评估结果"""
        return {
            "score": 60.0,
            "technical": 60,
            "communication": 60,
            "completeness": 60,
            "problem_solving": 60,
            "teamwork": 60,
            "leadership": 60,
            "expression": None,
            "eye_contact": None,
            "body_language": None,
            "overall_image": None,
            "feedback": "评估失败，使用默认评分",
            "visual_feedback": ""
        }

    def _generate_cache_key(self, text: str, image: Optional[np.ndarray] = None) -> str:
        """生成缓存键"""
        hash_obj = hashlib.md5()
        hash_obj.update(text.encode('utf-8'))
        if image is not None:
            hash_obj.update(f"{image.shape}_{image.dtype}".encode('utf-8'))
        return hash_obj.hexdigest()

    def _check_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """检查缓存"""
        if cache_key in self.evaluation_cache:
            self.cache_hits += 1
            return self.evaluation_cache[cache_key]
        self.cache_misses += 1
        return None

    def _update_cache(self, cache_key: str, evaluation: Dict[str, Any]):
        """更新缓存"""
        if len(self.evaluation_cache) >= self.cache_size:
            oldest_key = next(iter(self.evaluation_cache))
            del self.evaluation_cache[oldest_key]
        self.evaluation_cache[cache_key] = evaluation

    def get_memory_usage(self) -> Dict[str, float]:
        """获取当前显存/内存使用情况"""
        usage = {
            "gpu_allocated_gb": 0.0,
            "gpu_reserved_gb": 0.0,
            "gpu_max_allocated_gb": 0.0,
        }

        if torch.cuda.is_available():
            usage["gpu_allocated_gb"] = torch.cuda.memory_allocated() / 1024**3
            usage["gpu_reserved_gb"] = torch.cuda.memory_reserved() / 1024**3
            usage["gpu_max_allocated_gb"] = torch.cuda.max_memory_allocated() / 1024**3

        return usage

    def cleanup_memory(self):
        """手动清理显存和缓存"""
        # 清空评估缓存
        cache_size = len(self.evaluation_cache)
        self.evaluation_cache.clear()
        logger.info(f"[OK] 已清空评估缓存 ({cache_size} 条)")

        # 清理 PyTorch 缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.info("[OK] GPU 显存缓存已清理")

        # 记录清理后的显存使用
        usage = self.get_memory_usage()
        logger.info(f"[INFO] 清理后显存: 已分配={usage['gpu_allocated_gb']:.2f}GB, "
                   f"预留={usage['gpu_reserved_gb']:.2f}GB")
