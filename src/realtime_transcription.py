"""
锐聘AI - 实时流式转录系统
基于 WebSocket 的音频流接收与实时转录

核心功能:
1. WebSocket 连接管理（多会话并发）
2. 音频流环形缓冲区（支持 VAD）
3. 断点续传机制（序列号确认+重传）
4. 实时 Whisper 转录（后台异步处理）
"""

import asyncio
import json
import time
import tempfile
import wave
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Any, Tuple
from pathlib import Path

import numpy as np

from .logger import logger, LogContext, log_function_call


# ==================== 数据模型 ====================

class ConnectionState(Enum):
    """WebSocket 连接状态"""
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    DISCONNECTED = "disconnected"
    ERROR = "error"


@dataclass
class AudioChunk:
    """音频数据块"""
    sequence_number: int          # 序列号，用于断点续传
    timestamp: float              # 客户端发送时间戳
    data: bytes                   # 音频数据
    is_final: bool = False        # 是否为最后一块
    received_at: float = field(default_factory=time.time)  # 服务端接收时间


@dataclass
class TranscriptionResult:
    """转录结果"""
    type: str                     # 'partial' | 'final'
    text: str                     # 完整文本
    delta: str                    # 增量文本
    language: str                 # 检测到的语言
    confidence: float             # 置信度
    timestamp: float              # 结果生成时间
    sequence_start: int = 0       # 对应的音频起始序列号
    sequence_end: int = 0         # 对应的音频结束序列号


# ==================== 音频流缓冲区 ====================

class AudioStreamBuffer:
    """
    音频流环形缓冲区
    
    功能:
    - 接收乱序到达的音频 chunk，按序列号排序
    - 检测缺失的序列号，触发重传请求
    - VAD 语音活动检测，提取有效语音段
    - 支持断点续传后的数据拼接
    """
    
    def __init__(
        self,
        session_id: str,
        max_duration: float = 60.0,      # 最大缓冲时长（秒）
        sample_rate: int = 16000,
        vad_threshold: float = 0.01,      # VAD 能量阈值
        min_speech_duration: float = 0.3  # 最小语音段时长
    ):
        self.session_id = session_id
        self.sample_rate = sample_rate
        self.vad_threshold = vad_threshold
        self.min_speech_duration = min_speech_duration
        
        # 音频数据存储（序列号 -> AudioChunk）
        self.chunks: Dict[int, AudioChunk] = {}
        self.max_buffered_chunks = int(max_duration * 2)  # 500ms per chunk
        
        # 序列号管理
        self.expected_sequence = 0        # 期望接收的下一个序列号
        self.last_contiguous_sequence = -1  # 最后连续的序列号
        
        # 已处理标记
        self.processed_sequences: Set[int] = set()
        
        # 音频数据缓存（用于转录）
        self.audio_buffer = bytearray()
        self.max_buffer_bytes = int(max_duration * sample_rate * 2)  # 16-bit PCM
        
        # 统计
        self.total_chunks_received = 0
        self.total_chunks_missing = 0
        self.total_bytes_received = 0
        
        logger.info(f"[Buffer] 初始化音频缓冲区 | 会话: {session_id} | "
                   f"采样率: {sample_rate}Hz | 最大时长: {max_duration}s")
    
    def add_chunk(self, chunk: AudioChunk) -> Dict[str, Any]:
        """
        添加音频 chunk 到缓冲区
        
        Returns:
            {
                "status": "ok" | "duplicate" | "out_of_order" | "buffer_full",
                "missing_sequences": List[int],  # 检测到缺失的序列号
                "is_contiguous": bool            # 是否连续
            }
        """
        result = {
            "status": "ok",
            "missing_sequences": [],
            "is_contiguous": False
        }
        
        seq = chunk.sequence_number
        
        # 检查是否重复
        if seq in self.chunks:
            logger.debug(f"[Buffer] 重复 chunk，序列号: {seq}")
            result["status"] = "duplicate"
            return result
        
        # 检查缓冲区是否已满
        if len(self.chunks) >= self.max_buffered_chunks:
            logger.warning(f"[Buffer] 缓冲区已满，丢弃旧数据 | 会话: {self.session_id}")
            self._evict_old_chunks()
            result["status"] = "buffer_full"
        
        # 存储 chunk
        self.chunks[seq] = chunk
        self.total_chunks_received += 1
        self.total_bytes_received += len(chunk.data)
        
        # 检查是否有缺失的序列号
        if seq > self.expected_sequence:
            missing = list(range(self.expected_sequence, seq))
            result["missing_sequences"] = missing
            self.total_chunks_missing += len(missing)
            logger.warning(f"[Buffer] 检测到缺失 chunk | 会话: {self.session_id} | "
                          f"缺失: {missing} | 期望: {self.expected_sequence} | 收到: {seq}")
        
        # 更新期望序列号
        if seq == self.expected_sequence:
            # 找到连续的最大序列号
            while self.expected_sequence in self.chunks:
                self.expected_sequence += 1
            self.last_contiguous_sequence = self.expected_sequence - 1
            result["is_contiguous"] = True
            
            # 追加到音频缓冲区
            self._append_to_audio_buffer(seq)
        
        return result
    
    def _append_to_audio_buffer(self, up_to_sequence: int):
        """将连续的 chunk 追加到音频缓冲区"""
        sequences_to_append = []
        current_seq = up_to_sequence
        
        # 找到所有未处理的连续序列号
        while current_seq in self.chunks and current_seq not in self.processed_sequences:
            sequences_to_append.append(current_seq)
            current_seq += 1
        
        # 按顺序追加
        for seq in sorted(sequences_to_append):
            chunk = self.chunks[seq]
            self.audio_buffer.extend(chunk.data)
            self.processed_sequences.add(seq)
        
        # 限制缓冲区大小
        if len(self.audio_buffer) > self.max_buffer_bytes:
            excess = len(self.audio_buffer) - self.max_buffer_bytes
            self.audio_buffer = self.audio_buffer[excess:]
            logger.debug(f"[Buffer] 截断音频缓冲区 | 截断: {excess} bytes")
    
    def _evict_old_chunks(self):
        """淘汰旧 chunk"""
        if not self.chunks:
            return
        
        # 保留最近 80% 的 chunk
        sorted_seqs = sorted(self.chunks.keys())
        keep_count = int(self.max_buffered_chunks * 0.8)
        to_remove = sorted_seqs[:-keep_count] if len(sorted_seqs) > keep_count else []
        
        for seq in to_remove:
            del self.chunks[seq]
            self.processed_sequences.discard(seq)
        
        logger.info(f"[Buffer] 淘汰旧 chunk | 淘汰数量: {len(to_remove)}")
    
    def get_audio_for_transcription(self, min_duration: float = 1.0) -> Optional[np.ndarray]:
        """
        获取用于转录的音频数据
        
        使用 VAD 提取有效语音段
        
        Args:
            min_duration: 最小转录音频时长（秒）
            
        Returns:
            音频数组或 None
        """
        if len(self.audio_buffer) < int(min_duration * self.sample_rate * 2):
            return None
        
        # 转换为 numpy 数组
        audio = np.frombuffer(self.audio_buffer, dtype=np.int16).astype(np.float32) / 32768.0
        
        # VAD 检测
        speech_segments = self._vad_detect(audio)
        
        if not speech_segments:
            return None
        
        # 获取最新的语音段
        start, end = speech_segments[-1]
        speech_audio = audio[start:end]
        
        if len(speech_audio) < int(min_duration * self.sample_rate):
            return None
        
        return speech_audio
    
    def _vad_detect(self, audio: np.ndarray) -> List[Tuple[int, int]]:
        """
        简单的能量阈值 VAD
        
        Returns:
            [(start_sample, end_sample), ...]
        """
        frame_size = int(0.02 * self.sample_rate)  # 20ms
        hop_size = int(0.01 * self.sample_rate)    # 10ms
        
        # 计算每帧能量
        frames = []
        for i in range(0, len(audio) - frame_size, hop_size):
            frame = audio[i:i+frame_size]
            energy = np.sqrt(np.mean(frame ** 2))
            frames.append((i, energy))
        
        # 自适应阈值
        energies = [e for _, e in frames]
        if not energies:
            return []
        
        threshold = max(self.vad_threshold, np.mean(energies) * 0.3)
        
        # 检测语音段
        segments = []
        in_speech = False
        speech_start = 0
        
        for i, (sample_pos, energy) in enumerate(frames):
            if energy > threshold and not in_speech:
                in_speech = True
                speech_start = sample_pos
            elif energy < threshold * 0.6 and in_speech:
                in_speech = False
                duration = (sample_pos - speech_start) / self.sample_rate
                if duration >= self.min_speech_duration:
                    segments.append((speech_start, sample_pos))
        
        if in_speech:
            duration = (len(audio) - speech_start) / self.sample_rate
            if duration >= self.min_speech_duration:
                segments.append((speech_start, len(audio)))
        
        return segments
    
    def get_missing_sequences(self, max_lookback: int = 10) -> List[int]:
        """获取最近缺失的序列号"""
        if self.expected_sequence == 0:
            return []
        
        missing = []
        start = max(0, self.expected_sequence - max_lookback)
        
        for seq in range(start, self.expected_sequence):
            if seq not in self.chunks:
                missing.append(seq)
        
        return missing
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓冲区统计信息"""
        return {
            "total_chunks_received": self.total_chunks_received,
            "total_chunks_missing": self.total_chunks_missing,
            "total_bytes_received": self.total_bytes_received,
            "buffered_chunks": len(self.chunks),
            "expected_sequence": self.expected_sequence,
            "last_contiguous_sequence": self.last_contiguous_sequence,
            "audio_buffer_duration": len(self.audio_buffer) / (self.sample_rate * 2)
        }
    
    def clear(self):
        """清空缓冲区"""
        self.chunks.clear()
        self.processed_sequences.clear()
        self.audio_buffer = bytearray()
        self.expected_sequence = 0
        self.last_contiguous_sequence = -1
        logger.info(f"[Buffer] 缓冲区已清空 | 会话: {self.session_id}")


# ==================== 断点续传管理器 ====================

class ResumableTransferManager:
    """
    断点续传管理器
    
    功能:
    - 跟踪每个会话的接收状态
    - 检测缺失的数据块
    - 向客户端发送重传请求
    - 维护会话的断点信息（用于完全断线后的恢复）
    """
    
    def __init__(self, max_retransmit_requests: int = 3):
        self.max_retransmit = max_retransmit_requests
        
        # 会话状态
        self.session_states: Dict[str, Dict[str, Any]] = {}
        
        # 重传请求记录
        self.retransmit_requests: Dict[str, Dict[int, int]] = {}  # session -> {seq: count}
        
        logger.info("[Resumable] 初始化断点续传管理器")
    
    def register_session(self, session_id: str):
        """注册新会话"""
        self.session_states[session_id] = {
            "state": ConnectionState.CONNECTED,
            "connected_at": time.time(),
            "last_activity": time.time(),
            "total_chunks_expected": 0,
            "total_chunks_received": 0,
            "breakpoints": []  # 断点记录
        }
        self.retransmit_requests[session_id] = {}
        logger.info(f"[Resumable] 注册会话: {session_id}")
    
    def unregister_session(self, session_id: str):
        """注销会话"""
        if session_id in self.session_states:
            # 保存断点信息（用于后续恢复）
            state = self.session_states[session_id]
            state["state"] = ConnectionState.DISCONNECTED
            state["disconnected_at"] = time.time()
            logger.info(f"[Resumable] 注销会话: {session_id} | "
                       f"总接收: {state['total_chunks_received']}")
    
    def record_chunk_received(self, session_id: str, sequence: int):
        """记录收到 chunk"""
        if session_id in self.session_states:
            self.session_states[session_id]["last_activity"] = time.time()
            self.session_states[session_id]["total_chunks_received"] += 1
    
    def request_retransmit(self, session_id: str, missing_sequences: List[int]) -> List[int]:
        """
        请求重传
        
        Returns:
            需要请求重传的序列号列表（过滤掉已超时的）
        """
        if session_id not in self.retransmit_requests:
            return missing_sequences
        
        requests = self.retransmit_requests[session_id]
        to_request = []
        
        for seq in missing_sequences:
            count = requests.get(seq, 0)
            if count < self.max_retransmit:
                requests[seq] = count + 1
                to_request.append(seq)
            else:
                logger.warning(f"[Resumable] 序列号 {seq} 重传次数超限，放弃")
        
        return to_request
    
    def create_checkpoint(self, session_id: str, sequence: int):
        """创建断点检查点"""
        if session_id in self.session_states:
            self.session_states[session_id]["breakpoints"].append({
                "sequence": sequence,
                "timestamp": time.time()
            })
    
    def get_resume_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        获取断线恢复信息
        
        Returns:
            {
                "can_resume": bool,
                "last_sequence": int,
                "breakpoints": List[Dict]
            }
        """
        state = self.session_states.get(session_id)
        if not state:
            return None
        
        breakpoints = state.get("breakpoints", [])
        last_sequence = breakpoints[-1]["sequence"] if breakpoints else 0
        
        return {
            "can_resume": state["state"] == ConnectionState.DISCONNECTED,
            "last_sequence": last_sequence,
            "breakpoints": breakpoints
        }
    
    def cleanup_old_sessions(self, max_age: float = 3600):
        """清理过期会话"""
        now = time.time()
        to_remove = []
        
        for session_id, state in self.session_states.items():
            if state.get("disconnected_at"):
                if now - state["disconnected_at"] > max_age:
                    to_remove.append(session_id)
        
        for session_id in to_remove:
            del self.session_states[session_id]
            del self.retransmit_requests[session_id]
        
        if to_remove:
            logger.info(f"[Resumable] 清理过期会话: {len(to_remove)} 个")


# ==================== 实时转录管理器 ====================

class RealtimeTranscriptionManager:
    """
    实时转录管理器
    
    核心职责:
    1. 管理 WebSocket 连接生命周期
    2. 接收音频流并维护缓冲区
    3. 后台异步执行 Whisper 转录
    4. 推送转录结果到客户端
    5. 处理断点续传
    """
    
    def __init__(self, whisper_processor=None):
        """
        Args:
            whisper_processor: WhisperProcessor 实例
        """
        self.whisper = whisper_processor
        
        # WebSocket 连接管理
        self.connections: Dict[str, Any] = {}  # session_id -> websocket
        
        # 音频缓冲区
        self.buffers: Dict[str, AudioStreamBuffer] = {}
        
        # 断点续传
        self.resumable = ResumableTransferManager()
        
        # 转录结果
        self.transcription_results: Dict[str, str] = {}  # session_id -> accumulated text
        self.partial_results: Dict[str, str] = {}        # session_id -> last partial
        
        # 后台处理任务
        self.processing_tasks: Dict[str, asyncio.Task] = {}
        self.task_running: Dict[str, bool] = {}
        
        # 统计
        self.stats = {
            "total_sessions": 0,
            "active_sessions": 0,
            "total_transcriptions": 0
        }
        
        logger.info("=" * 60)
        logger.info("初始化实时转录管理器")
        logger.info("=" * 60)
    
    async def connect(self, websocket, session_id: str) -> bool:
        """
        建立 WebSocket 连接
        
        Args:
            websocket: WebSocket 对象
            session_id: 会话ID
            
        Returns:
            是否成功
        """
        try:
            logger.info(f"[RT] 新连接请求 | 会话: {session_id}")
            
            # 检查是否是断线恢复
            resume_info = self.resumable.get_resume_info(session_id)
            if resume_info and resume_info["can_resume"]:
                logger.info(f"[RT] 检测到断线恢复 | 会话: {session_id} | "
                           f"上次序列: {resume_info['last_sequence']}")
            
            # 存储连接
            self.connections[session_id] = websocket
            
            # 创建音频缓冲区
            self.buffers[session_id] = AudioStreamBuffer(session_id)
            
            # 注册断点续传
            self.resumable.register_session(session_id)
            
            # 初始化转录结果
            if session_id not in self.transcription_results:
                self.transcription_results[session_id] = ""
                self.partial_results[session_id] = ""
            
            # 启动后台处理任务
            self.task_running[session_id] = True
            self.processing_tasks[session_id] = asyncio.create_task(
                self._processing_loop(session_id)
            )
            
            # 更新统计
            self.stats["total_sessions"] += 1
            self.stats["active_sessions"] += 1
            
            logger.info(f"[RT] 连接成功 | 会话: {session_id} | "
                       f"活跃会话: {self.stats['active_sessions']}")
            
            # 发送连接确认
            await websocket.send_json({
                "type": "connected",
                "session_id": session_id,
                "timestamp": time.time(),
                "resume_info": resume_info
            })
            
            return True
            
        except Exception as e:
            logger.error(f"[RT] 连接失败 | 会话: {session_id} | 错误: {e}")
            return False
    
    async def disconnect(self, session_id: str, reason: str = ""):
        """断开连接"""
        logger.info(f"[RT] 断开连接 | 会话: {session_id} | 原因: {reason}")
        
        # 停止后台任务
        self.task_running[session_id] = False
        if session_id in self.processing_tasks:
            self.processing_tasks[session_id].cancel()
            try:
                await self.processing_tasks[session_id]
            except asyncio.CancelledError:
                pass
            del self.processing_tasks[session_id]
        
        # 注销断点续传
        self.resumable.unregister_session(session_id)
        
        # 清理连接
        if session_id in self.connections:
            del self.connections[session_id]
        
        # 更新统计
        self.stats["active_sessions"] = max(0, self.stats["active_sessions"] - 1)
        
        logger.info(f"[RT] 连接已清理 | 会话: {session_id} | "
                   f"剩余活跃: {self.stats['active_sessions']}")
    
    async def receive_audio_chunk(self, session_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        接收音频 chunk
        
        Args:
            session_id: 会话ID
            data: {
                "sequence_number": int,
                "timestamp": float,
                "data": List[int],  # bytes as list
                "is_final": bool
            }
            
        Returns:
            处理结果
        """
        if session_id not in self.buffers:
            return {"status": "error", "message": "Session not found"}
        
        try:
            # 解码音频数据
            audio_bytes = bytes(data["data"])
            
            # 创建 chunk
            chunk = AudioChunk(
                sequence_number=data["sequence_number"],
                timestamp=data.get("timestamp", time.time()),
                data=audio_bytes,
                is_final=data.get("is_final", False)
            )
            
            # 添加到缓冲区
            buffer = self.buffers[session_id]
            result = buffer.add_chunk(chunk)
            
            # 记录收到 chunk
            self.resumable.record_chunk_received(session_id, chunk.sequence_number)
            
            # 如果有缺失，请求重传
            if result["missing_sequences"]:
                to_request = self.resumable.request_retransmit(
                    session_id, result["missing_sequences"]
                )
                
                if to_request and session_id in self.connections:
                    await self.connections[session_id].send_json({
                        "type": "retransmit_request",
                        "sequences": to_request,
                        "timestamp": time.time()
                    })
            
            # 如果是最后一块，标记结束
            if chunk.is_final:
                logger.info(f"[RT] 收到最后一块音频 | 会话: {session_id} | "
                           f"序列: {chunk.sequence_number}")
            
            return {
                "status": result["status"],
                "received_sequence": chunk.sequence_number,
                "expected_sequence": buffer.expected_sequence,
                "missing_count": len(result["missing_sequences"])
            }
            
        except Exception as e:
            logger.error(f"[RT] 处理 chunk 失败 | 会话: {session_id} | 错误: {e}")
            return {"status": "error", "message": str(e)}
    
    async def _processing_loop(self, session_id: str):
        """
        后台处理循环
        每 500ms 检查一次是否有新的语音需要转录
        """
        logger.info(f"[RT] 启动处理循环 | 会话: {session_id}")
        
        last_processed_sequence = -1
        
        while self.task_running.get(session_id, False):
            try:
                await asyncio.sleep(0.5)  # 500ms 处理间隔
                
                if session_id not in self.buffers:
                    continue
                
                buffer = self.buffers[session_id]
                
                # 检查是否有新的连续数据
                if buffer.last_contiguous_sequence <= last_processed_sequence:
                    continue
                
                # 获取音频进行转录
                audio = buffer.get_audio_for_transcription(min_duration=1.0)
                
                if audio is not None and len(audio) > 0:
                    # 执行转录
                    result = await self._transcribe_audio(session_id, audio)
                    
                    if result:
                        # 发送结果
                        await self._send_transcription_result(session_id, result)
                        
                        # 更新处理进度
                        last_processed_sequence = buffer.last_contiguous_sequence
                        
                        # 创建检查点
                        self.resumable.create_checkpoint(
                            session_id, last_processed_sequence
                        )
                
            except asyncio.CancelledError:
                logger.info(f"[RT] 处理循环取消 | 会话: {session_id}")
                break
            except Exception as e:
                logger.error(f"[RT] 处理循环错误 | 会话: {session_id} | 错误: {e}")
        
        logger.info(f"[RT] 处理循环结束 | 会话: {session_id}")
    
    async def _transcribe_audio(self, session_id: str, audio: np.ndarray) -> Optional[TranscriptionResult]:
        """
        转录音频
        
        Args:
            session_id: 会话ID
            audio: 音频数据
            
        Returns:
            TranscriptionResult 或 None
        """
        if not self.whisper or not self.whisper.is_available():
            logger.warning("[RT] Whisper 不可用，跳过转录")
            return None
        
        try:
            # 保存临时文件
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                # 写入 WAV 文件
                with wave.open(tmp.name, 'wb') as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(self.buffers[session_id].sample_rate)
                    wav_file.writeframes((audio * 32767).astype(np.int16).tobytes())
                
                tmp_path = tmp.name
            
            # 执行转录
            start_time = time.time()
            result = self.whisper.transcribe(
                audio_path=tmp_path,
                language=None  # 自动检测语言
            )
            transcribe_time = time.time() - start_time
            
            # 清理临时文件
            Path(tmp_path).unlink(missing_ok=True)
            
            if not result["success"]:
                logger.warning(f"[RT] 转录失败 | 会话: {session_id} | "
                              f"错误: {result.get('error', 'unknown')}")
                return None
            
            # 计算增量文本
            current_text = result["text"].strip()
            previous_text = self.partial_results.get(session_id, "")
            
            if current_text.startswith(previous_text):
                delta = current_text[len(previous_text):]
            else:
                delta = current_text
            
            # 更新部分结果
            self.partial_results[session_id] = current_text
            self.transcription_results[session_id] = current_text
            
            # 更新统计
            self.stats["total_transcriptions"] += 1
            
            logger.info(f"[RT] 转录完成 | 会话: {session_id} | "
                       f"耗时: {transcribe_time:.2f}s | "
                       f"文字: {current_text[:50]}...")
            
            return TranscriptionResult(
                type="partial",
                text=current_text,
                delta=delta,
                language=result.get("language", "unknown"),
                confidence=result.get("confidence", 0),
                timestamp=time.time()
            )
            
        except Exception as e:
            logger.error(f"[RT] 转录异常 | 会话: {session_id} | 错误: {e}")
            return None
    
    async def _send_transcription_result(self, session_id: str, result: TranscriptionResult):
        """发送转录结果到客户端"""
        if session_id not in self.connections:
            return
        
        websocket = self.connections[session_id]
        
        message = {
            "type": result.type,
            "text": result.text,
            "delta": result.delta,
            "language": result.language,
            "confidence": result.confidence,
            "timestamp": result.timestamp
        }
        
        try:
            await websocket.send_json(message)
            logger.debug(f"[RT] 发送结果 | 会话: {session_id} | "
                        f"增量: {result.delta[:30]}...")
        except Exception as e:
            logger.error(f"[RT] 发送结果失败 | 会话: {session_id} | 错误: {e}")
    
    async def finalize(self, session_id: str) -> Optional[str]:
        """
        结束转录，返回最终结果
        
        Args:
            session_id: 会话ID
            
        Returns:
            最终转录文本
        """
        logger.info(f"[RT] 结束转录 | 会话: {session_id}")
        
        # 停止后台任务
        self.task_running[session_id] = False
        if session_id in self.processing_tasks:
            self.processing_tasks[session_id].cancel()
            try:
                await self.processing_tasks[session_id]
            except asyncio.CancelledError:
                pass
            del self.processing_tasks[session_id]
        
        # 获取最终结果
        final_text = self.transcription_results.get(session_id, "")
        
        # 发送最终结果
        if session_id in self.connections:
            try:
                await self.connections[session_id].send_json({
                    "type": "final",
                    "text": final_text,
                    "language": "unknown",
                    "confidence": 0,
                    "timestamp": time.time()
                })
            except Exception as e:
                logger.error(f"[RT] 发送最终结果失败 | 会话: {session_id} | 错误: {e}")
        
        # 清理
        if session_id in self.buffers:
            self.buffers[session_id].clear()
        
        logger.info(f"[RT] 转录完成 | 会话: {session_id} | "
                   f"最终文字: {final_text[:100]}...")
        
        return final_text
    
    def get_session_stats(self, session_id: str) -> Optional[Dict[str, Any]]:
        """获取会话统计"""
        if session_id not in self.buffers:
            return None
        
        buffer_stats = self.buffers[session_id].get_stats()
        
        return {
            "session_id": session_id,
            "buffer_stats": buffer_stats,
            "transcribed_text": self.transcription_results.get(session_id, ""),
            "is_processing": self.task_running.get(session_id, False)
        }
    
    def get_global_stats(self) -> Dict[str, Any]:
        """获取全局统计"""
        return {
            **self.stats,
            "active_connections": len(self.connections),
            "active_buffers": len(self.buffers)
        }


# ==================== FastAPI WebSocket 路由 ====================

# 全局管理器实例
_rt_manager: Optional[RealtimeTranscriptionManager] = None


def get_rt_manager() -> RealtimeTranscriptionManager:
    """获取全局实时转录管理器"""
    global _rt_manager
    if _rt_manager is None:
        from .audio_processor import WhisperProcessor
        whisper = WhisperProcessor()
        _rt_manager = RealtimeTranscriptionManager(whisper)
    return _rt_manager


def init_rt_manager(whisper_processor=None):
    """初始化全局管理器"""
    global _rt_manager
    _rt_manager = RealtimeTranscriptionManager(whisper_processor)
    logger.info("[RT] 全局实时转录管理器已初始化")


# 尝试导入 FastAPI
# 注意：这个模块可以在没有 FastAPI 的情况下独立使用（仅管理器部分）
# WebSocket 路由需要 FastAPI 环境
try:
    from fastapi import APIRouter, WebSocket, WebSocketDisconnect
    FASTAPI_AVAILABLE = True
    
    router = APIRouter()
    
    @router.websocket("/ws/transcribe")
    async def websocket_transcribe(websocket: WebSocket):
        """
        WebSocket 实时转录端点
        
        协议:
        1. 客户端连接后发送初始化消息: {"type": "init", "session_id": "xxx"}
        2. 服务端确认连接: {"type": "connected", ...}
        3. 客户端发送音频: {"type": "audio_chunk", "sequence_number": N, "data": [...]}
        4. 服务端返回转录结果: {"type": "partial", "text": "...", "delta": "..."}
        5. 客户端发送结束: {"type": "finalize"}
        6. 服务端返回最终结果: {"type": "final", "text": "..."}
        """
        session_id = None
        manager = get_rt_manager()
        
        try:
            # 等待连接
            await websocket.accept()
            logger.info("[WS] WebSocket 连接已接受")
            
            # 等待初始化消息
            init_data = await websocket.receive_json()
            
            if init_data.get("type") != "init":
                await websocket.send_json({
                    "type": "error",
                    "message": "Expected init message"
                })
                await websocket.close(code=4001)
                return
            
            session_id = init_data.get("session_id")
            if not session_id:
                await websocket.send_json({
                    "type": "error",
                    "message": "Missing session_id"
                })
                await websocket.close(code=4002)
                return
            
            # 建立连接
            success = await manager.connect(websocket, session_id)
            if not success:
                await websocket.close(code=4003)
                return
            
            # 接收消息循环
            while True:
                try:
                    data = await websocket.receive_json()
                    msg_type = data.get("type")
                    
                    if msg_type == "audio_chunk":
                        # 处理音频 chunk
                        result = await manager.receive_audio_chunk(session_id, data)
                        
                        # 发送确认
                        await websocket.send_json({
                            "type": "ack",
                            "sequence_number": data.get("sequence_number"),
                            "status": result["status"],
                            "expected_sequence": result.get("expected_sequence")
                        })
                    
                    elif msg_type == "finalize":
                        # 结束转录
                        final_text = await manager.finalize(session_id)
                        break
                    
                    elif msg_type == "ping":
                        # 心跳
                        await websocket.send_json({
                            "type": "pong",
                            "timestamp": time.time()
                        })
                    
                    elif msg_type == "retransmit":
                        # 客户端重传
                        chunk = AudioChunk(
                            sequence_number=data["sequence_number"],
                            timestamp=data.get("timestamp", time.time()),
                            data=bytes(data["data"])
                        )
                        if session_id in manager.buffers:
                            manager.buffers[session_id].add_chunk(chunk)
                    
                    else:
                        logger.warning(f"[WS] 未知消息类型: {msg_type}")
                
                except Exception as e:
                    logger.error(f"[WS] 消息处理错误 | 会话: {session_id} | 错误: {e}")
                    break
        
        except WebSocketDisconnect:
            logger.info(f"[WS] 客户端断开连接 | 会话: {session_id}")
        except Exception as e:
            logger.error(f"[WS] WebSocket 错误 | 会话: {session_id} | 错误: {e}")
        finally:
            if session_id:
                await manager.disconnect(session_id, "connection_closed")
    
    logger.info("[WS] FastAPI WebSocket 路由已注册")

except ImportError:
    FASTAPI_AVAILABLE = False
    router = None
    logger.warning("[WS] FastAPI 未安装，WebSocket 路由不可用")
    logger.warning("[WS] 如需使用 WebSocket，请安装: pip install fastapi uvicorn websockets")
