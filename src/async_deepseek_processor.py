"""
锐聘AI - 异步DeepSeek API处理器（基于线程池）
解决同步API调用阻塞主线程的问题

使用方式:
    1. 直接替换 DeepSeekProcessor
    2. 或者作为独立组件使用

特性:
    - 基于 concurrent.futures.ThreadPoolExecutor 实现异步
    - 支持回调函数处理评估结果
    - 支持超时控制
    - 兼容现有 DeepSeekProcessor 接口
"""

import os
import json
import hashlib
import time
import re
import threading
from typing import Dict, Any, Optional, Union, Callable, List
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from queue import Queue

from .deepseek_processor import DeepSeekProcessor
from .logger import logger


class AsyncDeepSeekProcessor:
    """
    异步DeepSeek处理器 - 基于线程池
    
    将同步的API调用转换为异步执行，避免阻塞主线程。
    适用于Gradio等需要保持UI响应的场景。
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-chat",
        timeout: float = 10.0,
        max_retries: int = 3,
        max_workers: int = 3,
        queue_size: int = 100
    ):
        """
        初始化异步处理器
        
        Args:
            api_key: DeepSeek API密钥
            base_url: API基础URL
            model: 模型名称
            timeout: 请求超时时间（秒）
            max_retries: 最大重试次数
            max_workers: 线程池最大工作线程数（默认3，控制并发API调用数）
            queue_size: 任务队列大小
        """
        logger.info("=" * 60)
        logger.info("初始化异步DeepSeek处理器（线程池模式）")
        logger.info("=" * 60)
        
        # 创建同步处理器实例（在线程中使用）
        self._processor = DeepSeekProcessor(
            api_key=api_key,
            base_url=base_url,
            model=model,
            timeout=timeout,
            max_retries=max_retries
        )
        
        # 线程池配置
        self.max_workers = max_workers
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="deepseek_worker"
        )
        
        # 任务队列和回调
        self._pending_tasks: Dict[str, Future] = {}
        self._callbacks: Dict[str, Callable] = {}
        self._task_lock = threading.Lock()
        self._task_counter = 0
        
        # 统计信息
        self._stats = {
            "submitted": 0,
            "completed": 0,
            "failed": 0,
            "cancelled": 0
        }
        
        logger.info(f"[OK] 异步处理器初始化完成")
        logger.info(f"[OK] 线程池大小: {max_workers}")
        logger.info(f"[OK] 任务队列大小: {queue_size}")
    
    def _generate_task_id(self) -> str:
        """生成唯一任务ID"""
        with self._task_lock:
            self._task_counter += 1
            return f"task_{self._task_counter}_{hashlib.md5(str(time.time()).encode()).hexdigest()[:6]}"
    
    def evaluate_answer_async(
        self,
        answer: str,
        question: Optional[str] = None,
        has_audio: bool = False,
        has_video: bool = False,
        image: Optional[Any] = None,
        callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        error_callback: Optional[Callable[[Exception], None]] = None
    ) -> str:
        """
        异步评估回答 - 非阻塞
        
        Args:
            answer: 候选人的回答文本
            question: 面试问题
            has_audio: 是否包含音频
            has_video: 是否包含视频
            image: 视频帧图像
            callback: 成功回调函数，接收评估结果
            error_callback: 错误回调函数，接收异常对象
            
        Returns:
            task_id: 任务ID，可用于查询状态或取消任务
        """
        task_id = self._generate_task_id()
        
        logger.info(f"[ASYNC-{task_id}] 提交异步评估任务")
        logger.info(f"[ASYNC-{task_id}] 回答长度: {len(answer) if answer else 0} 字符")
        
        # 存储回调
        if callback:
            self._callbacks[task_id] = callback
        
        # 提交任务到线程池
        future = self._executor.submit(
            self._evaluate_worker,
            task_id,
            answer,
            question,
            has_audio,
            has_video,
            image
        )
        
        # 添加完成回调
        future.add_done_callback(
            lambda f, tid=task_id, err_cb=error_callback: 
            self._on_task_complete(tid, f, err_cb)
        )
        
        with self._task_lock:
            self._pending_tasks[task_id] = future
            self._stats["submitted"] += 1
        
        logger.info(f"[ASYNC-{task_id}] 任务已提交到线程池")
        logger.info(f"[ASYNC-{task_id}] 当前待处理任务数: {len(self._pending_tasks)}")
        
        return task_id
    
    def _evaluate_worker(
        self,
        task_id: str,
        answer: str,
        question: Optional[str],
        has_audio: bool,
        has_video: bool,
        image: Optional[Any]
    ) -> Dict[str, Any]:
        """
        工作线程函数 - 在线程池中执行同步API调用
        """
        logger.info(f"[ASYNC-{task_id}] 工作线程开始执行")
        start_time = time.time()
        
        try:
            # 调用同步处理器的评估方法
            result = self._processor.evaluate_answer(
                answer=answer,
                question=question,
                has_audio=has_audio,
                has_video=has_video,
                image=image
            )
            
            elapsed = time.time() - start_time
            logger.info(f"[ASYNC-{task_id}] 工作线程完成，耗时: {elapsed:.3f}秒")
            
            return result
            
        except Exception as e:
            logger.error(f"[ASYNC-{task_id}] 工作线程异常: {e}")
            raise
    
    def _on_task_complete(
        self,
        task_id: str,
        future: Future,
        error_callback: Optional[Callable[[Exception], None]] = None
    ):
        """
        任务完成回调 - 在主线程中执行
        """
        try:
            # 获取结果（如果异常会在这里抛出）
            result = future.result()
            
            with self._task_lock:
                self._stats["completed"] += 1
                if task_id in self._pending_tasks:
                    del self._pending_tasks[task_id]
            
            logger.info(f"[ASYNC-{task_id}] 任务完成，得分: {result.get('score', 0)}")
            
            # 执行用户回调
            if task_id in self._callbacks:
                callback = self._callbacks.pop(task_id)
                try:
                    callback(result)
                    logger.info(f"[ASYNC-{task_id}] 用户回调执行成功")
                except Exception as e:
                    logger.error(f"[ASYNC-{task_id}] 用户回调异常: {e}")
            
        except Exception as e:
            with self._task_lock:
                self._stats["failed"] += 1
                if task_id in self._pending_tasks:
                    del self._pending_tasks[task_id]
            
            logger.error(f"[ASYNC-{task_id}] 任务失败: {e}")
            
            # 执行错误回调
            if error_callback:
                try:
                    error_callback(e)
                except Exception as cb_err:
                    logger.error(f"[ASYNC-{task_id}] 错误回调异常: {cb_err}")
            
            # 清理回调
            if task_id in self._callbacks:
                del self._callbacks[task_id]
    
    def get_result(self, task_id: str, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """
        阻塞等待任务结果
        
        Args:
            task_id: 任务ID
            timeout: 超时时间（秒），None表示无限等待
            
        Returns:
            评估结果，超时返回None
        """
        with self._task_lock:
            future = self._pending_tasks.get(task_id)
        
        if not future:
            logger.warning(f"[ASYNC-{task_id}] 任务不存在或已完成")
            return None
        
        try:
            logger.info(f"[ASYNC-{task_id}] 等待任务结果，超时: {timeout}秒")
            result = future.result(timeout=timeout)
            logger.info(f"[ASYNC-{task_id}] 获取结果成功")
            return result
        except TimeoutError:
            logger.warning(f"[ASYNC-{task_id}] 等待结果超时")
            return None
        except Exception as e:
            logger.error(f"[ASYNC-{task_id}] 获取结果失败: {e}")
            return None
    
    def cancel_task(self, task_id: str) -> bool:
        """
        取消任务
        
        Args:
            task_id: 任务ID
            
        Returns:
            是否成功取消
        """
        with self._task_lock:
            future = self._pending_tasks.get(task_id)
        
        if not future:
            return False
        
        cancelled = future.cancel()
        if cancelled:
            with self._task_lock:
                self._stats["cancelled"] += 1
                if task_id in self._pending_tasks:
                    del self._pending_tasks[task_id]
                if task_id in self._callbacks:
                    del self._callbacks[task_id]
            logger.info(f"[ASYNC-{task_id}] 任务已取消")
        else:
            logger.warning(f"[ASYNC-{task_id}] 任务取消失败（可能已在运行）")
        
        return cancelled
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._task_lock:
            stats = self._stats.copy()
            stats["pending"] = len(self._pending_tasks)
            stats["thread_pool_size"] = self.max_workers
        return stats
    
    def shutdown(self, wait: bool = True):
        """
        关闭处理器
        
        Args:
            wait: 是否等待所有任务完成
        """
        logger.info("关闭异步处理器...")
        
        # 取消所有待处理任务
        with self._task_lock:
            for task_id, future in list(self._pending_tasks.items()):
                future.cancel()
            self._pending_tasks.clear()
            self._callbacks.clear()
        
        # 关闭线程池
        self._executor.shutdown(wait=wait)
        
        logger.info("[OK] 异步处理器已关闭")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown(wait=True)


class BatchAsyncProcessor:
    """
    批量异步处理器
    
    用于批量评估多个回答，自动管理并发和结果收集
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        max_workers: int = 3,
        max_concurrent: int = 3
    ):
        self.processor = AsyncDeepSeekProcessor(
            api_key=api_key,
            max_workers=max_workers
        )
        self.max_concurrent = max_concurrent
    
    def evaluate_batch(
        self,
        items: List[Dict[str, Any]],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[Dict[str, Any]]:
        """
        批量评估 - 并发执行，收集所有结果
        
        Args:
            items: 评估项列表，每项包含 answer, question 等字段
            progress_callback: 进度回调，参数(已完成数, 总数)
            
        Returns:
            评估结果列表（与输入顺序一致）
        """
        total = len(items)
        logger.info(f"[BATCH] 开始批量评估: {total} 项")
        
        # 存储结果和同步对象
        results = [None] * total
        completed_count = [0]
        lock = threading.Lock()
        event = threading.Event()
        
        def on_result(index: int, result: Dict[str, Any]):
            results[index] = result
            with lock:
                completed_count[0] += 1
                current = completed_count[0]
            
            logger.info(f"[BATCH] 进度: {current}/{total}")
            
            if progress_callback:
                try:
                    progress_callback(current, total)
                except Exception as e:
                    logger.error(f"[BATCH] 进度回调异常: {e}")
            
            if current >= total:
                event.set()
        
        def on_error(index: int, error: Exception):
            logger.error(f"[BATCH] 第 {index} 项评估失败: {error}")
            # 使用默认评估结果
            results[index] = {
                "score": 60,
                "technical": 60,
                "communication": 60,
                "completeness": 60,
                "problem_solving": 60,
                "teamwork": 60,
                "leadership": 60,
                "feedback": f"评估失败: {str(error)}"
            }
            with lock:
                completed_count[0] += 1
                current = completed_count[0]
            
            if current >= total:
                event.set()
        
        # 提交所有任务
        for i, item in enumerate(items):
            self.processor.evaluate_answer_async(
                answer=item.get("answer", ""),
                question=item.get("question"),
                has_audio=item.get("has_audio", False),
                has_video=item.get("has_video", False),
                image=item.get("image"),
                callback=lambda r, idx=i: on_result(idx, r),
                error_callback=lambda e, idx=i: on_error(idx, e)
            )
        
        # 等待所有任务完成
        logger.info("[BATCH] 等待所有任务完成...")
        event.wait()
        
        logger.info("[BATCH] 批量评估完成")
        return results
    
    def shutdown(self):
        """关闭处理器"""
        self.processor.shutdown()



