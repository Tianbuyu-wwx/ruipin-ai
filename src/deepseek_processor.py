"""
锐聘AI - DeepSeek API处理器
负责使用DeepSeek API进行面试评估
集成API监控中间件，自动记录调用指标到降级决策中心
"""

import os
import json
import hashlib
import time
import re
import functools
from typing import Dict, Any, Optional, Union
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from .logger import logger, LogContext, log_function_call, performance_logger


def api_monitor(method_name: str = "api_call"):
    """
    API监控装饰器
    
    自动记录API调用指标到降级决策中心的资源监控器：
    - 响应时间
    - 成功/失败状态
    - 连续失败次数
    
    使用方式:
        @api_monitor("deepseek_evaluate")
        def evaluate_answer(self, ...):
            ...
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            start_time = time.time()
            success = False
            response_time_ms = 0.0
            
            try:
                result = func(self, *args, **kwargs)
                success = True
                return result
            except Exception as e:
                success = False
                raise
            finally:
                response_time_ms = (time.time() - start_time) * 1000
                
                try:
                    from .degradation import DegradationController
                    controller = DegradationController()
                    controller.resource_monitor.record_api_call(
                        response_time_ms=response_time_ms,
                        success=success
                    )
                    logger.debug(
                        f"[API监控] {method_name} | "
                        f"耗时: {response_time_ms:.1f}ms | "
                        f"状态: {'成功' if success else '失败'}"
                    )
                except Exception as monitor_error:
                    logger.debug(f"API监控记录失败: {monitor_error}")
        
        return wrapper
    return decorator


class DeepSeekProcessor:
    """DeepSeek API处理器 - 首选评估方案"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-chat",
        timeout: float = 10.0,
        max_retries: int = 3
    ):
        """
        初始化DeepSeek处理器
        
        Args:
            api_key: DeepSeek API密钥，默认从环境变量DEEPSEEK_API_KEY获取
            base_url: API基础URL
            model: 模型名称 (deepseek-chat 或 deepseek-reasoner)
            timeout: 请求超时时间（秒）
            max_retries: 最大重试次数
        """
        logger.info("=" * 60)
        logger.info("初始化DeepSeek API处理器")
        logger.info("=" * 60)
        
        # API配置 - 优先使用传入的api_key，其次环境变量，最后配置文件
        self.api_key = api_key
        if not self.api_key:
            self.api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        if not self.api_key:
            # 尝试从配置文件读取
            try:
                import sys
                from pathlib import Path
                sys.path.insert(0, str(Path(__file__).parent.parent))
                from configs.config import EVALUATION_CONFIG
                if EVALUATION_CONFIG:
                    self.api_key = EVALUATION_CONFIG.get("deepseek", {}).get("api_key", "")
            except Exception:
                pass
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        
        # 客户端
        self.client = None
        
        # 缓存机制
        self.evaluation_cache = {}
        self.cache_size = 1000
        self.cache_hits = 0
        self.cache_misses = 0
        
        # 性能统计
        self.api_call_times = []
        self.total_calls = 0
        self.failed_calls = 0
        
        # 评估提示模板
        self._init_prompt_templates()
        
        logger.info(f"模型: {self.model}")
        logger.info(f"API地址: {self.base_url}")
        logger.info(f"超时: {self.timeout}秒")
        logger.info(f"最大重试: {self.max_retries}次")
        logger.info(f"缓存大小: {self.cache_size}")
        
        # 初始化客户端
        self._init_client()
    
    def _init_client(self) -> bool:
        """初始化OpenAI客户端"""
        try:
            if not self.api_key:
                logger.warning("未设置API密钥，请设置DEEPSEEK_APIKEY环境变量")
                return False
            
            if OpenAI is None:
                logger.error("未安装openai库，请运行: pip install openai")
                return False
            
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
                max_retries=self.max_retries
            )
            
            logger.info("[OK] DeepSeek客户端初始化成功")
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] 客户端初始化失败: {e}")
            self.client = None
            return False
    
    def _init_prompt_templates(self):
        """初始化评估提示模板"""
        self.system_prompt = """你是一位专业的技术面试官，擅长评估候选人的技术能力和综合素质。

你的任务是根据候选人的回答，从以下六个维度进行专业评估：
1. 技术能力（0-100分）：回答的技术准确性和专业性
2. 表达能力（0-100分）：语言组织和清晰度
3. 回答完整度（0-100分）：内容是否全面
4. 问题解决（0-100分）：分析和解决问题的能力
5. 团队协作（0-100分）：团队合作和沟通能力
6. 领导力（0-100分）：领导能力和项目管理能力

评估要求：
- 技术能力：考察技术概念理解、实践经验、技术深度
- 表达能力：考察逻辑清晰、表达流畅、条理分明
- 回答完整度：考察内容全面、覆盖要点、无重大遗漏
- 问题解决：考察分析思路、解决方案、应变能力
- 团队协作：考察合作意识、沟通技巧、冲突处理
- 领导力：考察项目把控、团队管理、决策能力

请给出客观、公正的评分，并提供具体、建设性的反馈意见。"""

        self.evaluation_template = """请评估以下面试回答：

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
    
    def _build_evaluation_prompt(
        self,
        answer: str,
        question: Optional[str] = None,
        has_audio: bool = False,
        has_video: bool = False
    ) -> str:
        """构建评估提示"""
        # 问题部分
        if question:
            question_section = f"面试问题：{question}\n"
        else:
            question_section = ""
        
        # 媒体输入说明
        media_parts = []
        if has_video:
            media_parts.append("视频")
        if has_audio:
            media_parts.append("音频")
        
        if media_parts:
            media_section = f"\n输入类型：候选人通过{'和'.join(media_parts)}进行了回答"
        else:
            media_section = ""
        
        return self.evaluation_template.format(
            question_section=question_section,
            answer=answer or "（候选人未提供文字回答）",
            media_section=media_section
        )
    
    def _generate_cache_key(self, text: str, question: Optional[str] = None) -> str:
        """生成缓存键"""
        try:
            hash_obj = hashlib.md5()
            if question:
                hash_obj.update(question.encode('utf-8'))
            hash_obj.update(text.encode('utf-8'))
            return hash_obj.hexdigest()
        except Exception as e:
            logger.error(f"生成缓存键失败: {e}")
            return f"error_{time.time()}"
    
    def _check_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """检查缓存"""
        try:
            if cache_key in self.evaluation_cache:
                self.cache_hits += 1
                hit_rate = self._get_cache_hit_rate()
                logger.debug(f"缓存命中: {self.cache_hits}/{self.cache_hits + self.cache_misses} ({hit_rate:.1f}%)")
                return self.evaluation_cache[cache_key]
            self.cache_misses += 1
            return None
        except Exception as e:
            logger.error(f"检查缓存失败: {e}")
            return None
    
    def _update_cache(self, cache_key: str, evaluation: Dict[str, Any]):
        """更新缓存"""
        try:
            if len(self.evaluation_cache) >= self.cache_size:
                oldest_key = next(iter(self.evaluation_cache))
                del self.evaluation_cache[oldest_key]
                logger.debug(f"缓存已满，删除最旧项: {oldest_key}")
            
            self.evaluation_cache[cache_key] = evaluation
            logger.debug(f"缓存更新: {cache_key}")
        except Exception as e:
            logger.error(f"更新缓存失败: {e}")
    
    def _get_cache_hit_rate(self) -> float:
        """获取缓存命中率"""
        total = self.cache_hits + self.cache_misses
        return (self.cache_hits / total * 100) if total > 0 else 0
    
    def _parse_evaluation_response(self, response_text: str) -> Dict[str, Any]:
        """解析API响应"""
        try:
            # 尝试直接解析JSON
            try:
                result = json.loads(response_text)
            except json.JSONDecodeError:
                # 尝试从文本中提取JSON
                json_match = re.search(r'\{[\s\S]*\}', response_text)
                if json_match:
                    result = json.loads(json_match.group())
                else:
                    raise ValueError("无法从响应中提取JSON")
            
            # 验证并提取评分
            scores = {
                "technical": int(result.get("technical", 60)),
                "communication": int(result.get("communication", 60)),
                "completeness": int(result.get("completeness", 60)),
                "problem_solving": int(result.get("problem_solving", 60)),
                "teamwork": int(result.get("teamwork", 60)),
                "leadership": int(result.get("leadership", 60)),
                "feedback": result.get("feedback", "评估完成")
            }
            
            # 确保评分在有效范围内
            for key in ["technical", "communication", "completeness", 
                       "problem_solving", "teamwork", "leadership"]:
                scores[key] = max(0, min(100, scores[key]))
            
            # 计算综合得分
            scores["score"] = round(sum([
                scores["technical"],
                scores["communication"],
                scores["completeness"],
                scores["problem_solving"],
                scores["teamwork"],
                scores["leadership"]
            ]) / 6, 1)
            
            return scores
            
        except Exception as e:
            logger.error(f"解析评估响应失败: {e}")
            logger.debug(f"原始响应: {response_text[:200]}...")
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
            "feedback": "评估解析失败，使用默认评分"
        }
    
    @api_monitor("deepseek_evaluate")
    @log_function_call()
    def evaluate_answer(
        self,
        answer: str,
        question: Optional[str] = None,
        has_audio: bool = False,
        has_video: bool = False,
        image: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        评估面试回答
        
        Args:
            answer: 候选人回答内容
            question: 面试问题（可选）
            has_audio: 是否包含音频输入
            has_video: 是否包含视频输入
            image: 图像数据（当前不支持直接上传，需预处理）
            
        Returns:
            评估结果字典
        """
        evaluation_start_time = time.time()
        request_id = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
        
        try:
            logger.info(f"[REQ-{request_id}] DeepSeek评估开始 | "
                         f"回答长度={len(answer) if answer else 0}字符 | "
                         f"音频={has_audio} | 视频={has_video} | 图像={image is not None}")
            
            # 检查客户端
            if not self.client:
                logger.error(f"[REQ-{request_id}] DeepSeek客户端未初始化，无法进行评估")
                return self._get_default_evaluation()
            
            # 生成缓存键
            cache_key = self._generate_cache_key(answer or "", question)
            cached_result = self._check_cache(cache_key)
            if cached_result:
                logger.info(f"[REQ-{request_id}] 缓存命中 | 得分={cached_result.get('score', 0)}")
                return cached_result

            logger.debug(f"[REQ-{request_id}] 缓存未命中，准备调用API")
            
            # 构建提示
            prompt_start = time.time()
            prompt = self._build_evaluation_prompt(answer, question, has_audio, has_video)
            prompt_time = time.time() - prompt_start
            logger.debug(f"[REQ-{request_id}] 提示构建完成 | 耗时={prompt_time:.3f}秒 | 长度={len(prompt)}字符")

            api_start_time = time.time()
            logger.info(f"[REQ-{request_id}] 调用DeepSeek API | 模型={self.model}")
            
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=800,
                    response_format={"type": "json_object"}
                )
                
                api_time = time.time() - api_start_time
                self.api_call_times.append(api_time)
                if len(self.api_call_times) > 10:
                    self.api_call_times.pop(0)
                
                self.total_calls += 1
                
                logger.info(f"[REQ-{request_id}] API调用成功 | 响应时间={api_time:.3f}秒 | "
                             f"token={response.usage.total_tokens if response.usage else 'N/A'}")

                if not response or not response.choices:
                    logger.error(f"[REQ-{request_id}] API响应为空或choices为空")
                    raise ValueError("API响应格式异常")
                
                # 解析响应
                parse_start = time.time()
                response_text = response.choices[0].message.content
                logger.debug(f"[REQ-{request_id}] 响应内容长度: {len(response_text)} 字符")

                evaluation = self._parse_evaluation_response(response_text)
                parse_time = time.time() - parse_start

                logger.info(f"[REQ-{request_id}] 评估完成 | 得分={evaluation['score']} | "
                             f"技术={evaluation['technical']} | 表达={evaluation['communication']} | "
                             f"完整度={evaluation['completeness']} | 问题解决={evaluation['problem_solving']} | "
                             f"团队={evaluation['teamwork']} | 领导力={evaluation['leadership']} | "
                             f"解析耗时={parse_time:.3f}秒")
                
                avg_time = sum(self.api_call_times) / len(self.api_call_times)
                total_time = time.time() - evaluation_start_time
                
                performance_logger.record("deepseek_api_time", api_time, "秒")
                self._update_cache(cache_key, evaluation)
                logger.debug(f"[REQ-{request_id}] 结果已缓存 | 总耗时={total_time:.3f}秒 | 平均API={avg_time:.3f}秒 | 调用次数={self.total_calls}")
                
                return evaluation
                
            except Exception as e:
                self.failed_calls += 1
                error_time = time.time() - api_start_time
                logger.error(f"[REQ-{request_id}] DeepSeek API调用失败!")
                logger.error(f"[REQ-{request_id}] 错误类型: {type(e).__name__}")
                logger.error(f"[REQ-{request_id}] 错误信息: {str(e)}")
                logger.error(f"[REQ-{request_id}] 失败前耗时: {error_time:.3f}秒")
                
                # 判断是否为超时错误
                error_str = str(e).lower()
                if any(keyword in error_str for keyword in ['timeout', 'timed out', '连接超时', 'read timeout']):
                    logger.error(f"[REQ-{request_id}] ⚠️ 检测到API超时错误!")
                    logger.error(f"[REQ-{request_id}] 建议:")
                    logger.error(f"[REQ-{request_id}]   1. 检查网络连接")
                    logger.error(f"[REQ-{request_id}]   2. 增加超时时间 (当前: {self.timeout}秒)")
                    logger.error(f"[REQ-{request_id}]   3. 检查DeepSeek服务状态")
                    logger.error(f"[REQ-{request_id}]   4. 考虑启用降级到本地模型")
                elif any(keyword in error_str for keyword in ['connection', 'connect', '网络', 'dns']):
                    logger.error(f"[REQ-{request_id}] ⚠️ 检测到网络连接错误!")
                    logger.error(f"[REQ-{request_id}] 建议检查网络连接和DNS配置")
                elif any(keyword in error_str for keyword in ['rate limit', 'too many', '429']):
                    logger.error(f"[REQ-{request_id}] ⚠️ 检测到API限流错误!")
                    logger.error(f"[REQ-{request_id}] 建议降低请求频率或升级API套餐")
                elif any(keyword in error_str for keyword in ['auth', 'key', 'unauthorized', '401']):
                    logger.error(f"[REQ-{request_id}] ⚠️ 检测到API认证错误!")
                    logger.error(f"[REQ-{request_id}] 建议检查API密钥是否正确")
                
                logger.info(f"[REQ-{request_id}] 当前失败次数: {self.failed_calls}/{self.total_calls}")
                logger.info(f"[REQ-{request_id}] 成功率: {(self.total_calls - self.failed_calls) / self.total_calls * 100:.1f}%")
                
                raise
                
        except Exception as e:
            total_time = time.time() - evaluation_start_time
            logger.error(f"[REQ-{request_id}] DeepSeek评估失败!")
            logger.error(f"[REQ-{request_id}] 总耗时: {total_time:.3f}秒")
            logger.error(f"[REQ-{request_id}] 错误详情:", exc_info=True)
            logger.error(f"[REQ-{request_id}] DeepSeek评估请求失败 | 错误={e}")
            return self._get_default_evaluation()
    
    def is_available(self) -> bool:
        """检查处理器是否可用"""
        return self.client is not None and bool(self.api_key)
    
    def register_health_check(self):
        """向降级决策中心注册健康检查"""
        try:
            from .degradation import DegradationController
            controller = DegradationController()
            
            def check_deepseek_health() -> bool:
                """检查DeepSeek API健康状态"""
                if not self.client or not self.api_key:
                    return False
                try:
                    self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": "hi"}],
                        max_tokens=5,
                        timeout=5.0
                    )
                    return True
                except Exception:
                    return False
            
            controller.health_checker.register_checker(
                "deepseek_api",
                check_deepseek_health,
                metadata={"model": self.model, "base_url": self.base_url}
            )
            logger.info("[OK] DeepSeek API健康检查已注册到降级决策中心")
            
        except Exception as e:
            logger.warning(f"注册健康检查失败: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取处理器统计信息"""
        return {
            "total_calls": self.total_calls,
            "failed_calls": self.failed_calls,
            "success_rate": round((self.total_calls - self.failed_calls) / self.total_calls * 100, 2) 
                           if self.total_calls > 0 else 100,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": round(self._get_cache_hit_rate(), 2),
            "avg_api_time": round(sum(self.api_call_times) / len(self.api_call_times), 3) 
                           if self.api_call_times else 0,
            "model": self.model,
            "available": self.is_available()
        }
    
    def clear_cache(self):
        """清除缓存"""
        try:
            self.evaluation_cache.clear()
            self.cache_hits = 0
            self.cache_misses = 0
            logger.info("[OK] DeepSeek缓存已清除")
        except Exception as e:
            logger.error(f"清除缓存失败: {e}")


class HybridProcessor:
    """混合处理器 - DeepSeek API优先，本地模型备用"""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        api_key: Optional[str] = None,
        prefer_online: bool = True
    ):
        """
        初始化混合处理器
        
        Args:
            model_path: 本地模型路径
            api_key: DeepSeek API密钥
            prefer_online: 是否优先使用在线API
        """
        logger.info("=" * 60)
        logger.info("初始化混合评估处理器")
        logger.info("=" * 60)
        
        self.prefer_online = prefer_online
        
        # 初始化DeepSeek处理器
        self.deepseek_processor = DeepSeekProcessor(api_key=api_key)
        
        # 初始化本地处理器（备用）
        self.local_processor = None
        if model_path:
            try:
                from .model_processor import Qwen3VLProcessor
                self.local_processor = Qwen3VLProcessor(model_path)
                logger.info("[OK] 本地处理器初始化成功")
            except Exception as e:
                logger.warning(f"本地处理器初始化失败: {e}")
        
        logger.info(f"首选方案: {'DeepSeek API' if prefer_online else '本地模型'}")
        logger.info(f"DeepSeek可用: {self.deepseek_processor.is_available()}")
        logger.info(f"本地模型可用: {self.local_processor is not None}")
    
    @log_function_call()
    def evaluate_answer(
        self,
        answer: str,
        question: Optional[str] = None,
        has_audio: bool = False,
        has_video: bool = False,
        image: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        评估面试回答 - 自动选择最佳方案
        
        策略：
        1. 如果prefer_online=True且DeepSeek可用，优先使用API
        2. 如果API调用失败或超时，降级到本地模型
        3. 如果本地模型不可用，使用规则评估
        """
        # 尝试DeepSeek API
        if self.prefer_online and self.deepseek_processor.is_available():
            try:
                result = self.deepseek_processor.evaluate_answer(
                    answer=answer,
                    question=question,
                    has_audio=has_audio,
                    has_video=has_video,
                    image=image
                )
                
                # 检查是否为默认失败结果
                if result.get("feedback") != "评估解析失败，使用默认评分":
                    logger.info("使用DeepSeek API评估")
                    return result
                    
            except Exception as e:
                logger.warning(f"DeepSeek评估失败，尝试降级: {e}")
        
        # 降级到本地模型
        if self.local_processor:
            try:
                logger.info("降级到本地模型评估")
                return self.local_processor.evaluate_answer(
                    answer=answer,
                    question=question,
                    image=image
                )
            except Exception as e:
                logger.error(f"本地模型评估也失败: {e}")
        
        # 最终备用方案
        logger.warning("使用默认规则评估")
        return self._fallback_evaluation(answer, has_audio, has_video)
    
    def _fallback_evaluation(
        self,
        answer: str,
        has_audio: bool = False,
        has_video: bool = False
    ) -> Dict[str, Any]:
        """最终备用评估方案"""
        try:
            length = len(answer) if answer else 0
            
            # 基础分数
            base_score = 65 if (has_audio or has_video) else 60
            
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
            
        except Exception as e:
            logger.error(f"备用评估失败: {e}")
            return {
                "score": 60.0,
                "technical": 60,
                "communication": 60,
                "completeness": 60,
                "problem_solving": 60,
                "teamwork": 60,
                "leadership": 60,
                "feedback": "评估失败，使用默认评分"
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = {
            "prefer_online": self.prefer_online,
            "deepseek_available": self.deepseek_processor.is_available(),
            "local_available": self.local_processor is not None
        }
        
        if self.deepseek_processor:
            stats["deepseek"] = self.deepseek_processor.get_stats()
        
        return stats


if __name__ == "__main__":
    # 测试代码
    logger.info("=" * 60)
    logger.info("运行DeepSeek处理器测试")
    logger.info("=" * 60)
    
    try:
        # 测试DeepSeek处理器
        processor = DeepSeekProcessor()
        
        if processor.is_available():
            # 测试评估
            test_answer = """Java中的多线程可以通过继承Thread类或实现Runnable接口来实现。
            继承Thread类需要重写run()方法，而实现Runnable接口需要实现run()方法并将Runnable实例传给Thread构造函数。
            另外，还可以使用ExecutorService线程池来管理线程，这种方式更加灵活和高效。"""
            
            test_question = "请介绍Java中实现多线程的几种方式"
            
            logger.info(f"测试回答: {test_answer[:100]}...")
            
            result = processor.evaluate_answer(
                answer=test_answer,
                question=test_question
            )
            
            logger.info(f"评估结果:")
            logger.info(f"  综合得分: {result['score']}")
            logger.info(f"  技术能力: {result['technical']}")
            logger.info(f"  表达能力: {result['communication']}")
            logger.info(f"  反馈意见: {result['feedback'][:100]}...")
            
            # 测试缓存
            logger.info("测试缓存...")
            result_cached = processor.evaluate_answer(
                answer=test_answer,
                question=test_question
            )
            
            # 打印统计
            stats = processor.get_stats()
            logger.info(f"统计信息: {stats}")
            
            logger.info("[OK] 测试完成")
        else:
            logger.warning("DeepSeek API不可用，请检查API密钥")
        
    except Exception as e:
        logger.error(f"测试过程中发生错误: {e}", exc_info=True)
