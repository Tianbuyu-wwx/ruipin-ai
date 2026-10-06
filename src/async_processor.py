"""
锐聘AI - 异步DeepSeek API处理器
支持异步评估和流式响应
"""

import os
import json
import hashlib
import time
import asyncio
from typing import Dict, Any, Optional, AsyncGenerator
from concurrent.futures import ThreadPoolExecutor

from openai import AsyncOpenAI

from .logger import logger
from .deepseek_processor import DeepSeekProcessor


class AsyncDeepSeekProcessor:
    """异步DeepSeek API处理器"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-chat",
        timeout: float = 10.0,
        max_retries: int = 3
    ):
        """
        初始化异步处理器
        
        Args:
            api_key: DeepSeek API密钥
            base_url: API基础URL
            model: 模型名称
            timeout: 请求超时时间
            max_retries: 最大重试次数
        """
        logger.info("=" * 60)
        logger.info("初始化异步DeepSeek API处理器")
        logger.info("=" * 60)
        
        # API配置
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        
        # 异步客户端
        self.client = None
        
        # 线程池（用于同步到异步的桥接）
        self.executor = ThreadPoolExecutor(max_workers=5)
        
        # 缓存和统计
        self.evaluation_cache = {}
        self.cache_size = 1000
        self.api_call_times = []
        self.total_calls = 0
        self.failed_calls = 0
        
        # 初始化客户端
        self._init_client()
    
    def _init_client(self) -> bool:
        """初始化异步客户端"""
        try:
            if not self.api_key:
                logger.warning("未设置API密钥")
                return False
            
            self.client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
                max_retries=self.max_retries
            )
            
            logger.info("[OK] 异步DeepSeek客户端初始化成功")
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] 异步客户端初始化失败: {e}")
            return False
    
    def _generate_cache_key(self, text: str, question: Optional[str] = None) -> str:
        """生成缓存键"""
        try:
            hash_obj = hashlib.md5()
            if question:
                hash_obj.update(question.encode('utf-8'))
            hash_obj.update(text.encode('utf-8'))
            return hash_obj.hexdigest()
        except Exception:
            return f"error_{time.time()}"
    
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
            "feedback": "评估解析失败，使用默认评分"
        }
    
    def _build_evaluation_prompt(
        self,
        answer: str,
        question: Optional[str] = None,
        has_audio: bool = False,
        has_video: bool = False
    ) -> str:
        """构建评估提示"""
        system_prompt = """你是一位专业的技术面试官，擅长评估候选人的技术能力和综合素质。

你的任务是根据候选人的回答，从以下六个维度进行专业评估：
1. 技术能力（0-100分）：回答的技术准确性和专业性
2. 表达能力（0-100分）：语言组织和清晰度
3. 回答完整度（0-100分）：内容是否全面
4. 问题解决（0-100分）：分析和解决问题的能力
5. 团队协作（0-100分）：团队合作和沟通能力
6. 领导力（0-100分）：领导能力和项目管理能力

请给出客观、公正的评分，并提供具体、建设性的反馈意见。"""

        evaluation_template = """请评估以下面试回答：

{question_section}
回答内容：{answer}
{media_section}

请按以下JSON格式返回评估结果：
{{
    "technical": <技术能力评分0-100>,
    "communication": <表达能力评分0-100>,
    "completeness": <回答完整度评分0-100>,
    "problem_solving": <问题解决评分0-100>,
    "teamwork": <团队协作评分0-100>,
    "leadership": <领导力评分0-100>,
    "feedback": "详细的评估反馈，包括优点和改进建议"
}}

注意：
1. 所有评分必须是0-100之间的整数
2. feedback字段必须是中文，不少于50字
3. 只返回JSON格式，不要其他内容"""
        
        if question:
            question_section = f"面试问题：{question}\n"
        else:
            question_section = ""
        
        media_parts = []
        if has_video:
            media_parts.append("视频")
        if has_audio:
            media_parts.append("音频")
        
        if media_parts:
            media_section = f"\n输入类型：候选人通过{'和'.join(media_parts)}进行了回答"
        else:
            media_section = ""
        
        return evaluation_template.format(
            question_section=question_section,
            answer=answer or "（候选人未提供文字回答）",
            media_section=media_section
        )
    
    def _parse_evaluation_response(self, response_text: str) -> Dict[str, Any]:
        """解析API响应"""
        try:
            try:
                result = json.loads(response_text)
            except json.JSONDecodeError:
                import re
                json_match = re.search(r'\{[\s\S]*\}', response_text)
                if json_match:
                    result = json.loads(json_match.group())
                else:
                    raise ValueError("无法从响应中提取JSON")
            
            scores = {
                "technical": int(result.get("technical", 60)),
                "communication": int(result.get("communication", 60)),
                "completeness": int(result.get("completeness", 60)),
                "problem_solving": int(result.get("problem_solving", 60)),
                "teamwork": int(result.get("teamwork", 60)),
                "leadership": int(result.get("leadership", 60)),
                "feedback": result.get("feedback", "评估完成")
            }
            
            for key in ["technical", "communication", "completeness", 
                       "problem_solving", "teamwork", "leadership"]:
                scores[key] = max(0, min(100, scores[key]))
            
            scores["score"] = round(sum([
                scores["technical"], scores["communication"],
                scores["completeness"], scores["problem_solving"],
                scores["teamwork"], scores["leadership"]
            ]) / 6, 1)
            
            return scores
            
        except Exception as e:
            logger.error(f"解析评估响应失败: {e}")
            return self._get_default_evaluation()
    
    async def evaluate_answer_async(
        self,
        answer: str,
        question: Optional[str] = None,
        has_audio: bool = False,
        has_video: bool = False
    ) -> Dict[str, Any]:
        """
        异步评估面试回答
        
        Returns:
            评估结果字典
        """
        request_id = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
        
        try:
            logger.info(f"[REQ-{request_id}] 异步评估开始")
            
            if not self.client:
                logger.error(f"[REQ-{request_id}] 客户端未初始化")
                return self._get_default_evaluation()
            
            # 检查缓存
            cache_key = self._generate_cache_key(answer or "", question)
            if cache_key in self.evaluation_cache:
                logger.info(f"[REQ-{request_id}] 缓存命中")
                return self.evaluation_cache[cache_key]
            
            # 构建提示
            prompt = self._build_evaluation_prompt(answer, question, has_audio, has_video)
            
            # 异步调用API
            start_time = time.time()
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._build_evaluation_prompt.__doc__ or ""},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=800,
                response_format={"type": "json_object"}
            )
            
            api_time = time.time() - start_time
            self.api_call_times.append(api_time)
            self.total_calls += 1
            
            # 解析响应
            response_text = response.choices[0].message.content
            evaluation = self._parse_evaluation_response(response_text)
            
            logger.info(f"[REQ-{request_id}] 异步评估完成: {evaluation['score']:.1f}分, 耗时: {api_time:.3f}秒")
            
            # 更新缓存
            self.evaluation_cache[cache_key] = evaluation
            
            return evaluation
            
        except Exception as e:
            self.failed_calls += 1
            logger.error(f"[REQ-{request_id}] 异步评估失败: {e}")
            return self._get_default_evaluation()
    
    async def evaluate_answer_stream(
        self,
        answer: str,
        question: Optional[str] = None,
        has_audio: bool = False,
        has_video: bool = False
    ) -> AsyncGenerator[str, None]:
        """
        流式评估面试回答
        
        Yields:
            流式响应片段
        """
        request_id = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
        
        try:
            logger.info(f"[REQ-{request_id}] 流式评估开始")
            
            if not self.client:
                yield json.dumps(self._get_default_evaluation())
                return
            
            prompt = self._build_evaluation_prompt(answer, question, has_audio, has_video)
            
            # 流式调用API
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是专业的技术面试官"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=800,
                stream=True
            )
            
            full_response = ""
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    yield content
            
            # 解析完整响应
            evaluation = self._parse_evaluation_response(full_response)
            logger.info(f"[REQ-{request_id}] 流式评估完成: {evaluation['score']:.1f}分")
            
        except Exception as e:
            logger.error(f"[REQ-{request_id}] 流式评估失败: {e}")
            yield json.dumps(self._get_default_evaluation())
    
    async def evaluate_batch(
        self,
        answers: list,
        question: Optional[str] = None
    ) -> list:
        """
        批量异步评估
        
        Args:
            answers: 回答列表
            question: 面试问题
            
        Returns:
            评估结果列表
        """
        logger.info(f"批量评估: {len(answers)} 个回答")
        
        tasks = [
            self.evaluate_answer_async(answer, question)
            for answer in answers
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理异常结果
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"批量评估中的错误: {result}")
                processed_results.append(self._get_default_evaluation())
            else:
                processed_results.append(result)
        
        return processed_results
    
    def run_sync(self, coro):
        """运行异步协程（同步接口）"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 如果在运行中的事件循环中，使用线程池
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, coro)
                    return future.result()
            else:
                return loop.run_until_complete(coro)
        except RuntimeError:
            # 没有事件循环，创建新的
            return asyncio.run(coro)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "total_calls": self.total_calls,
            "failed_calls": self.failed_calls,
            "success_rate": round((self.total_calls - self.failed_calls) / self.total_calls * 100, 2) 
                           if self.total_calls > 0 else 100,
            "avg_api_time": round(sum(self.api_call_times) / len(self.api_call_times), 3) 
                           if self.api_call_times else 0,
            "model": self.model,
            "available": self.client is not None
        }


class AsyncHybridProcessor:
    """异步混合处理器"""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        api_key: Optional[str] = None,
        prefer_online: bool = True
    ):
        """初始化异步混合处理器"""
        logger.info("=" * 60)
        logger.info("初始化异步混合评估处理器")
        logger.info("=" * 60)
        
        self.prefer_online = prefer_online
        
        # 初始化异步DeepSeek处理器
        self.async_processor = AsyncDeepSeekProcessor(api_key=api_key)
        
        # 初始化同步本地处理器（备用）
        self.local_processor = None
        if model_path:
            try:
                from .model_processor import Qwen3VLProcessor
                self.local_processor = Qwen3VLProcessor(model_path)
                logger.info("[OK] 本地处理器初始化成功")
            except Exception as e:
                logger.warning(f"本地处理器初始化失败: {e}")
        
        logger.info(f"首选方案: {'DeepSeek API' if prefer_online else '本地模型'}")
    
    async def evaluate_answer(
        self,
        answer: str,
        question: Optional[str] = None,
        has_audio: bool = False,
        has_video: bool = False
    ) -> Dict[str, Any]:
        """
        异步评估 - 自动选择最佳方案
        """
        # 尝试异步DeepSeek API
        if self.prefer_online and self.async_processor.client:
            try:
                result = await self.async_processor.evaluate_answer_async(
                    answer=answer,
                    question=question,
                    has_audio=has_audio,
                    has_video=has_video
                )
                
                if result.get("feedback") != "评估解析失败，使用默认评分":
                    logger.info("使用异步DeepSeek API评估")
                    return result
                    
            except Exception as e:
                logger.warning(f"异步DeepSeek评估失败: {e}")
        
        # 降级到本地模型
        if self.local_processor:
            try:
                logger.info("降级到本地模型评估")
                return self.local_processor.evaluate_answer(
                    answer=answer,
                    question=question
                )
            except Exception as e:
                logger.error(f"本地模型评估也失败: {e}")
        
        # 最终备用方案
        logger.warning("使用默认规则评估")
        return self._fallback_evaluation(answer)
    
    def _fallback_evaluation(self, answer: str) -> Dict[str, Any]:
        """最终备用评估"""
        length = len(answer) if answer else 0
        base_score = 65
        
        if length < 20:
            score = base_score - 20
            feedback = "回答过于简短，请详细说明。"
        elif length < 50:
            score = base_score
            feedback = "回答尚可，但可以更加详细。"
        elif length < 100:
            score = base_score + 15
            feedback = "回答不错，涵盖了主要内容。"
        else:
            score = base_score + 25
            feedback = "回答详细完整，表达清晰。"
        
        score = max(40, min(95, score))
        
        return {
            "score": float(score),
            "technical": score,
            "communication": score,
            "completeness": score,
            "problem_solving": score,
            "teamwork": score,
            "leadership": score,
            "feedback": feedback + "（备用评估）"
        }



