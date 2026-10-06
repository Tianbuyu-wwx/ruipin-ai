"""
锐聘AI - 音频处理模块
基于 OpenAI Whisper 模型，支持音频转文字
"""

import os
import time
import tempfile
import math
from pathlib import Path
from typing import Dict, Any, Optional, Union

from .logger import logger, LogContext, log_function_call


class WhisperProcessor:
    """
    Whisper 音频转文字处理器
    支持本地模型和 API 两种模式
    """

    def __init__(
        self,
        model_size: str = "base",
        device: Optional[str] = None,
        api_key: Optional[str] = None,
        use_api: bool = False
    ):
        """
        初始化 Whisper 处理器

        Args:
            model_size: 本地模型尺寸 (tiny/base/small/medium/large)
            device: 计算设备 (cuda/cpu)，None 则自动选择
            api_key: OpenAI API 密钥（API 模式使用）
            use_api: 是否使用 OpenAI API 而非本地模型
        """
        logger.info("=" * 60)
        logger.info("初始化 Whisper 音频处理器")
        logger.info("=" * 60)
        logger.info(f"[CONFIG] 模型尺寸: {model_size}")
        logger.info(f"[CONFIG] 指定设备: {device or '自动选择'}")
        logger.info(f"[CONFIG] 使用模式: {'API模式' if use_api else '本地模型模式'}")

        self.model_size = model_size
        self.device = device
        self.api_key = api_key
        self.use_api = use_api

        self.model = None
        self._is_available = False

        if use_api:
            self._init_api_mode()
        else:
            self._init_local_mode()

    def _init_api_mode(self):
        """初始化 API 模式"""
        logger.info("[INIT] 开始初始化 Whisper API 模式...")
        try:
            from openai import OpenAI

            if not self.api_key:
                logger.warning("[INIT_FAIL] Whisper API 模式需要 api_key，未提供")
                return

            self.client = OpenAI(api_key=self.api_key)
            self._is_available = True
            logger.info("[INIT_OK] Whisper API 模式初始化成功")

        except ImportError:
            logger.error("[INIT_FAIL] openai 库未安装，无法使用 Whisper API 模式")
            logger.error("[INIT_FAIL] 请运行: pip install openai>=1.0.0")
        except Exception as e:
            logger.error(f"[INIT_FAIL] Whisper API 模式初始化失败: {e}", exc_info=True)

    def _init_local_mode(self):
        """初始化本地模型模式"""
        logger.info("[INIT] 开始初始化 Whisper 本地模型模式...")
        try:
            import whisper

            logger.info(f"[INIT] 正在加载 Whisper 本地模型: {self.model_size}")
            logger.info(f"[INIT] 模型将下载到: {os.path.expanduser('~/.cache/whisper')}")
            start_time = time.time()

            self.model = whisper.load_model(
                self.model_size,
                device=self.device
            )

            load_time = time.time() - start_time
            logger.info(f"[INIT_OK] Whisper 模型加载完成，耗时: {load_time:.2f}秒")
            self._is_available = True

        except ImportError:
            logger.error("[INIT_FAIL] whisper 库未安装，无法使用本地模型。")
            logger.error("[INIT_FAIL] 请运行: pip install openai-whisper")
        except Exception as e:
            logger.error(f"[INIT_FAIL] Whisper 本地模型加载失败: {e}", exc_info=True)

    def is_available(self) -> bool:
        """检查处理器是否可用"""
        return self._is_available

    @log_function_call()
    def transcribe(
        self,
        audio_path: Union[str, Path],
        language: Optional[str] = "zh",
        prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        将音频转录为文字

        Args:
            audio_path: 音频文件路径
            language: 音频语言代码 (zh/en/ja 等)，None 则自动检测
            prompt: 提示词，帮助模型理解上下文（如专业术语）

        Returns:
            {
                "text": "转录文字",
                "language": "检测到的语言",
                "confidence": 置信度分数,
                "segments": [段落详情],
                "duration": 音频时长(秒),
                "success": True/False,
                "error": 错误信息（如有）
            }
        """
        audio_path = Path(audio_path)

        logger.info("=" * 60)
        logger.info("开始音频转录")
        logger.info("=" * 60)
        logger.info(f"[INPUT] 音频文件路径: {audio_path}")
        logger.info(f"[INPUT] 指定语言: {language or '自动检测'}")
        logger.info(f"[INPUT] 提示词: {prompt or '无'}")

        # 检查文件是否存在
        if not audio_path.exists():
            logger.error(f"[CHECK_FAIL] 音频文件不存在: {audio_path}")
            logger.error(f"[CHECK_FAIL] 绝对路径: {audio_path.absolute()}")
            return {
                "text": "",
                "language": "",
                "confidence": 0.0,
                "segments": [],
                "duration": 0.0,
                "success": False,
                "error": f"音频文件不存在: {audio_path}"
            }

        # 检查文件大小
        try:
            file_size = audio_path.stat().st_size
            file_size_mb = file_size / (1024 * 1024)
            logger.info(f"[CHECK_OK] 音频文件存在，大小: {file_size_mb:.2f} MB ({file_size} 字节)")
            if file_size == 0:
                logger.error("[CHECK_FAIL] 音频文件大小为 0，可能是空文件")
                return {
                    "text": "",
                    "language": "",
                    "confidence": 0.0,
                    "segments": [],
                    "duration": 0.0,
                    "success": False,
                    "error": "音频文件为空"
                }
        except Exception as e:
            logger.error(f"[CHECK_FAIL] 无法读取文件信息: {e}")

        # 检查处理器是否可用
        if not self._is_available:
            logger.error("[CHECK_FAIL] Whisper 处理器不可用，可能初始化失败")
            return {
                "text": "",
                "language": "",
                "confidence": 0.0,
                "segments": [],
                "duration": 0.0,
                "success": False,
                "error": "Whisper 处理器未初始化"
            }

        logger.info(f"[CHECK_OK] 所有前置检查通过，开始转录...")
        logger.info(f"[CHECK_OK] 使用模式: {'API' if self.use_api else '本地模型'}")

        try:
            with LogContext(logger, "Whisper 音频转录"):
                transcribe_start = time.time()

                if self.use_api:
                    logger.info("[TRANSCRIBE] 调用 OpenAI API 进行转录...")
                    result = self._transcribe_api(audio_path, language, prompt)
                else:
                    logger.info("[TRANSCRIBE] 调用本地 Whisper 模型进行转录...")
                    result = self._transcribe_local(audio_path, language, prompt)

                transcribe_time = time.time() - transcribe_start
                logger.info(f"[TRANSCRIBE] 转录耗时: {transcribe_time:.2f}秒")

            if result["success"]:
                logger.info(f"[TRANSCRIBE_OK] 转录成功")
                logger.info(f"[TRANSCRIBE_OK] 文字长度: {len(result['text'])} 字符")
                logger.info(f"[TRANSCRIBE_OK] 检测语言: {result['language']}")
                logger.info(f"[TRANSCRIBE_OK] 音频时长: {result['duration']:.2f}秒")
                logger.info(f"[TRANSCRIBE_OK] 置信度: {result['confidence']:.2f}%")
                logger.info(f"[TRANSCRIBE_OK] 段落数: {len(result['segments'])}")
                logger.info(f"[TRANSCRIBE_OK] 前100字: {result['text'][:100]}...")
            else:
                logger.error(f"[TRANSCRIBE_FAIL] 转录失败: {result.get('error', '未知错误')}")

            return result

        except Exception as e:
            logger.error(f"[TRANSCRIBE_FAIL] 音频转录时发生未捕获异常: {e}", exc_info=True)
            return {
                "text": "",
                "language": "",
                "confidence": 0.0,
                "segments": [],
                "duration": 0.0,
                "success": False,
                "error": str(e)
            }

    def _transcribe_local(
        self,
        audio_path: Path,
        language: Optional[str],
        prompt: Optional[str]
    ) -> Dict[str, Any]:
        """使用本地模型转录"""
        import whisper

        logger.info("[LOCAL] 准备本地模型转录参数...")
        options = {
            "language": language,
            "task": "transcribe",
            "verbose": False
        }
        if prompt:
            options["initial_prompt"] = prompt
            logger.info(f"[LOCAL] 使用提示词: {prompt}")

        logger.info(f"[LOCAL] 转录选项: {options}")
        logger.info(f"[LOCAL] 开始调用 model.transcribe()...")

        try:
            result = self.model.transcribe(str(audio_path), **options)
            logger.info("[LOCAL] model.transcribe() 调用完成")
        except Exception as e:
            logger.error(f"[LOCAL_FAIL] model.transcribe() 调用失败: {e}", exc_info=True)
            return {
                "text": "",
                "language": "",
                "confidence": 0.0,
                "segments": [],
                "duration": 0.0,
                "success": False,
                "error": f"本地模型转录失败: {str(e)}"
            }

        segments = result.get("segments", [])
        logger.info(f"[LOCAL] 获取到 {len(segments)} 个段落")

        avg_confidence = 0.0
        if segments:
            avg_logprob = sum(
                seg.get("avg_logprob", 0) for seg in segments
            ) / len(segments)
            avg_confidence = math.exp(avg_logprob)
            logger.info(f"[LOCAL] 平均 logprob: {avg_logprob:.4f}, 转换后置信度: {avg_confidence:.4f}")
        else:
            logger.warning("[LOCAL] 未获取到任何段落，可能音频无内容")

        detected_language = result.get("language", language or "unknown")
        logger.info(f"[LOCAL] 检测到的语言: {detected_language}")

        return {
            "text": result.get("text", "").strip(),
            "language": detected_language,
            "confidence": min(avg_confidence * 100, 100.0),
            "segments": [
                {
                    "start": seg.get("start", 0),
                    "end": seg.get("end", 0),
                    "text": seg.get("text", "").strip()
                }
                for seg in segments
            ],
            "duration": segments[-1]["end"] if segments else 0.0,
            "success": True,
            "error": None
        }

    def _transcribe_api(
        self,
        audio_path: Path,
        language: Optional[str],
        prompt: Optional[str]
    ) -> Dict[str, Any]:
        """使用 OpenAI API 转录"""
        logger.info("[API] 准备 OpenAI API 转录参数...")

        try:
            with open(audio_path, "rb") as audio_file:
                kwargs = {
                    "model": "whisper-1",
                    "file": audio_file,
                    "response_format": "verbose_json"
                }
                if language:
                    kwargs["language"] = language
                    logger.info(f"[API] 指定语言: {language}")
                if prompt:
                    kwargs["prompt"] = prompt
                    logger.info(f"[API] 使用提示词: {prompt}")

                logger.info("[API] 发送请求到 OpenAI API...")
                api_start = time.time()
                response = self.client.audio.transcriptions.create(**kwargs)
                api_time = time.time() - api_start
                logger.info(f"[API] API 响应耗时: {api_time:.2f}秒")

            segments = getattr(response, "segments", [])
            logger.info(f"[API] 获取到 {len(segments)} 个段落")

            avg_confidence = 0.0
            if segments:
                avg_logprob = sum(
                    getattr(seg, "avg_logprob", 0) for seg in segments
                ) / len(segments)
                avg_confidence = math.exp(avg_logprob)
                logger.info(f"[API] 平均 logprob: {avg_logprob:.4f}, 转换后置信度: {avg_confidence:.4f}")
            else:
                logger.warning("[API] 未获取到任何段落")

            detected_language = getattr(response, "language", language or "unknown")
            logger.info(f"[API] 检测到的语言: {detected_language}")

            return {
                "text": getattr(response, "text", "").strip(),
                "language": detected_language,
                "confidence": min(avg_confidence * 100, 100.0),
                "segments": [
                    {
                        "start": getattr(seg, "start", 0),
                        "end": getattr(seg, "end", 0),
                        "text": getattr(seg, "text", "").strip()
                    }
                    for seg in segments
                ],
                "duration": segments[-1].end if segments else 0.0,
                "success": True,
                "error": None
            }

        except Exception as e:
            logger.error(f"[API_FAIL] OpenAI API 转录失败: {e}", exc_info=True)
            return {
                "text": "",
                "language": "",
                "confidence": 0.0,
                "segments": [],
                "duration": 0.0,
                "success": False,
                "error": f"API 转录失败: {str(e)}"
            }

    @log_function_call()
    def extract_audio_from_video(
        self,
        video_path: Union[str, Path],
        output_audio_path: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
        """
        从视频文件中提取音频

        Args:
            video_path: 视频文件路径
            output_audio_path: 输出音频路径，None 则使用临时文件

        Returns:
            {
                "audio_path": "提取的音频路径",
                "success": True/False,
                "error": 错误信息
            }
        """
        video_path = Path(video_path)

        logger.info("=" * 60)
        logger.info("开始从视频提取音频")
        logger.info("=" * 60)
        logger.info(f"[INPUT] 视频文件: {video_path}")
        logger.info(f"[INPUT] 指定输出路径: {output_audio_path or '自动创建临时文件'}")

        # 检查视频文件是否存在
        if not video_path.exists():
            logger.error(f"[CHECK_FAIL] 视频文件不存在: {video_path}")
            logger.error(f"[CHECK_FAIL] 绝对路径: {video_path.absolute()}")
            return {
                "audio_path": None,
                "success": False,
                "error": f"视频文件不存在: {video_path}"
            }

        # 检查视频文件大小
        try:
            file_size = video_path.stat().st_size
            file_size_mb = file_size / (1024 * 1024)
            logger.info(f"[CHECK_OK] 视频文件存在，大小: {file_size_mb:.2f} MB")
            if file_size == 0:
                logger.error("[CHECK_FAIL] 视频文件大小为 0")
                return {
                    "audio_path": None,
                    "success": False,
                    "error": "视频文件为空"
                }
        except Exception as e:
            logger.error(f"[CHECK_FAIL] 无法读取视频文件信息: {e}")

        try:
            # 确定输出路径
            if output_audio_path is None:
                temp_dir = Path(tempfile.gettempdir()) / "ruipin_ai_audio"
                temp_dir.mkdir(exist_ok=True)
                output_audio_path = temp_dir / f"{video_path.stem}.wav"
                logger.info(f"[CONFIG] 使用临时目录: {temp_dir}")
            else:
                output_audio_path = Path(output_audio_path)

            logger.info(f"[CONFIG] 输出音频路径: {output_audio_path}")

            # 使用 ffmpeg 提取音频
            import subprocess

            cmd = [
                "ffmpeg",
                "-y",
                "-i", str(video_path),
                "-vn",
                "-acodec", "pcm_s16le",
                "-ar", "16000",
                "-ac", "1",
                str(output_audio_path)
            ]

            logger.info(f"[FFMPEG] 执行命令: {' '.join(cmd)}")
            logger.info("[FFMPEG] 开始提取音频...")

            ffmpeg_start = time.time()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            ffmpeg_time = time.time() - ffmpeg_start
            logger.info(f"[FFMPEG] ffmpeg 执行耗时: {ffmpeg_time:.2f}秒")
            logger.info(f"[FFMPEG] 返回码: {result.returncode}")

            if result.returncode != 0:
                logger.error(f"[FFMPEG_FAIL] ffmpeg 提取音频失败")
                logger.error(f"[FFMPEG_FAIL] 标准错误输出: {result.stderr}")
                logger.error(f"[FFMPEG_FAIL] 标准输出: {result.stdout}")
                return {
                    "audio_path": None,
                    "success": False,
                    "error": f"ffmpeg 错误 (返回码 {result.returncode}): {result.stderr[:500]}"
                }

            # 检查输出文件
            if not output_audio_path.exists():
                logger.error(f"[FFMPEG_FAIL] ffmpeg 执行成功但输出文件不存在: {output_audio_path}")
                return {
                    "audio_path": None,
                    "success": False,
                    "error": "ffmpeg 输出文件未生成"
                }

            output_size = output_audio_path.stat().st_size
            output_size_mb = output_size / (1024 * 1024)
            logger.info(f"[FFMPEG_OK] 音频提取成功")
            logger.info(f"[FFMPEG_OK] 输出文件: {output_audio_path}")
            logger.info(f"[FFMPEG_OK] 输出大小: {output_size_mb:.2f} MB ({output_size} 字节)")

            return {
                "audio_path": str(output_audio_path),
                "success": True,
                "error": None
            }

        except FileNotFoundError:
            logger.error("[FFMPEG_FAIL] ffmpeg 未安装，无法提取音频")
            logger.error("[FFMPEG_FAIL] 请安装 ffmpeg 并确保它在系统 PATH 中")
            logger.error("[FFMPEG_FAIL] Windows: https://ffmpeg.org/download.html")
            logger.error("[FFMPEG_FAIL] Ubuntu/Debian: sudo apt install ffmpeg")
            logger.error("[FFMPEG_FAIL] macOS: brew install ffmpeg")
            return {
                "audio_path": None,
                "success": False,
                "error": "ffmpeg 未安装，请安装 ffmpeg 后重试"
            }
        except subprocess.TimeoutExpired:
            logger.error("[FFMPEG_FAIL] ffmpeg 执行超时 (超过30秒)")
            return {
                "audio_path": None,
                "success": False,
                "error": "ffmpeg 提取音频超时"
            }
        except Exception as e:
            logger.error(f"[FFMPEG_FAIL] 提取音频时发生未捕获异常: {e}", exc_info=True)
            return {
                "audio_path": None,
                "success": False,
                "error": str(e)
            }


class AudioProcessorManager:
    """
    音频处理器管理器
    封装 Whisper 转录，提供统一的音频处理接口
    """

    def __init__(self):
        logger.info("=" * 60)
        logger.info("初始化音频处理器管理器")
        logger.info("=" * 60)
        self.whisper = None
        self._init_whisper()

    def _init_whisper(self):
        """初始化 Whisper 处理器"""
        logger.info("[INIT] 开始初始化 Whisper 处理器...")
        try:
            logger.info("[INIT] 尝试加载 Whisper 本地模型 (base)...")
            self.whisper = WhisperProcessor(
                model_size="base",
                use_api=False
            )

            if self.whisper.is_available():
                logger.info("[INIT_OK] Whisper 处理器初始化成功且可用")
            else:
                logger.warning("[INIT_FAIL] Whisper 处理器初始化完成但不可用")
                self.whisper = None

        except Exception as e:
            logger.warning(f"[INIT_FAIL] Whisper 初始化失败: {e}", exc_info=True)
            self.whisper = None

    def is_available(self) -> bool:
        """检查音频处理是否可用"""
        available = self.whisper is not None and self.whisper.is_available()
        logger.debug(f"[CHECK] 音频处理器可用状态: {available}")
        return available

    def process_audio_input(
        self,
        audio_path: Optional[Union[str, Path]] = None,
        video_path: Optional[Union[str, Path]] = None,
        text_input: Optional[str] = None,
        language: str = "zh"
    ) -> Dict[str, Any]:
        """
        处理音频/视频输入，返回转录文字

        Args:
            audio_path: 音频文件路径
            video_path: 视频文件路径（会自动提取音频）
            text_input: 用户已输入的文字（作为补充）
            language: 音频语言

        Returns:
            {
                "transcribed_text": "转录的文字内容",
                "combined_text": "合并后的完整文字（转录+用户输入）",
                "has_transcription": 是否成功转录,
                "transcription_info": 转录详情,
                "success": True/False,
                "error": 错误信息
            }
        """
        logger.info("=" * 60)
        logger.info("处理音频/视频输入")
        logger.info("=" * 60)
        logger.info(f"[INPUT] 音频路径: {audio_path}")
        logger.info(f"[INPUT] 视频路径: {video_path}")
        logger.info(f"[INPUT] 用户文字: {'有' if text_input else '无'} ({len(text_input or '')} 字符)")
        logger.info(f"[INPUT] 指定语言: {language}")
        logger.info(f"[INPUT] 音频处理器可用: {self.is_available()}")

        transcribed_text = ""
        transcription_info = None
        has_transcription = False
        errors = []

        # 1. 如果有视频，先提取音频
        actual_audio_path = audio_path
        if video_path and not audio_path:
            logger.info("[STEP_1] 检测到视频输入且无音频输入，开始从视频提取音频...")
            if self.whisper:
                extract_result = self.whisper.extract_audio_from_video(video_path)
                if extract_result["success"]:
                    actual_audio_path = extract_result["audio_path"]
                    logger.info(f"[STEP_1_OK] 从视频提取音频成功: {actual_audio_path}")
                else:
                    error_msg = extract_result.get("error", "未知错误")
                    logger.warning(f"[STEP_1_FAIL] 从视频提取音频失败: {error_msg}")
                    errors.append(f"视频提取音频失败: {error_msg}")
            else:
                logger.warning("[STEP_1_SKIP] Whisper 处理器不可用，跳过视频提取")
                errors.append("Whisper 处理器不可用")
        elif video_path and audio_path:
            logger.info("[STEP_1] 同时提供了视频和音频，优先使用音频文件")
        else:
            logger.info("[STEP_1] 无需从视频提取音频")

        # 2. 转录音频
        if actual_audio_path:
            logger.info(f"[STEP_2] 准备转录音频: {actual_audio_path}")
            if self.is_available():
                logger.info("[STEP_2] 音频处理器可用，开始转录...")
                result = self.whisper.transcribe(
                    audio_path=actual_audio_path,
                    language=language
                )

                if result["success"]:
                    transcribed_text = result["text"]
                    has_transcription = True
                    transcription_info = {
                        "language": result["language"],
                        "confidence": result["confidence"],
                        "duration": result["duration"],
                        "segments_count": len(result["segments"])
                    }
                    logger.info(f"[STEP_2_OK] 音频转录成功")
                    logger.info(f"[STEP_2_OK] 转录文字: {len(transcribed_text)} 字符")
                    logger.info(f"[STEP_2_OK] 语言: {result['language']}, 置信度: {result['confidence']:.1f}%")
                else:
                    error_msg = result.get("error", "未知错误")
                    logger.warning(f"[STEP_2_FAIL] 音频转录失败: {error_msg}")
                    errors.append(f"转录失败: {error_msg}")
            else:
                logger.warning("[STEP_2_SKIP] 音频处理器不可用，跳过转录")
                errors.append("音频处理器不可用")
        else:
            logger.info("[STEP_2_SKIP] 无音频文件需要转录")

        # 3. 合并文字（转录文字 + 用户输入文字）
        logger.info("[STEP_3] 开始合并文字...")
        combined_text = self._merge_texts(transcribed_text, text_input or "")
        logger.info(f"[STEP_3_OK] 文字合并完成")

        # 最终结果汇总
        logger.info("=" * 60)
        logger.info("音频/视频处理结果汇总")
        logger.info("=" * 60)
        logger.info(f"[RESULT] 转录成功: {has_transcription}")
        logger.info(f"[RESULT] 转录文字: {len(transcribed_text)} 字符")
        logger.info(f"[RESULT] 用户输入: {len(text_input or '')} 字符")
        logger.info(f"[RESULT] 合并文字: {len(combined_text)} 字符")
        if transcription_info:
            logger.info(f"[RESULT] 转录详情: {transcription_info}")
        if errors:
            logger.warning(f"[RESULT] 处理过程中的错误: {errors}")

        return {
            "transcribed_text": transcribed_text,
            "combined_text": combined_text,
            "has_transcription": has_transcription,
            "transcription_info": transcription_info,
            "success": True,
            "errors": errors if errors else None
        }

    def _merge_texts(self, transcribed: str, user_input: str) -> str:
        """合并转录文字和用户输入文字"""
        transcribed = transcribed.strip()
        user_input = user_input.strip()

        logger.debug(f"[MERGE] 转录文字: {len(transcribed)} 字符")
        logger.debug(f"[MERGE] 用户输入: {len(user_input)} 字符")

        if not transcribed and not user_input:
            logger.debug("[MERGE] 两者均为空，返回空字符串")
            return ""

        if not transcribed:
            logger.debug("[MERGE] 无转录文字，使用用户输入")
            return user_input

        if not user_input:
            logger.debug("[MERGE] 无用户输入，使用转录文字")
            return transcribed

        # 如果用户输入是转录文字的子集或相似，优先使用转录
        if user_input in transcribed or transcribed in user_input:
            chosen = transcribed if len(transcribed) > len(user_input) else user_input
            logger.debug(f"[MERGE] 检测到包含关系，选择较长者 ({len(chosen)} 字符)")
            return chosen

        # 否则合并两者
        merged = f"{transcribed}\n\n[补充说明]: {user_input}"
        logger.debug(f"[MERGE] 合并两者，总长度: {len(merged)} 字符")
        return merged
