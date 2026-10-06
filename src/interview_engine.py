"""
锐聘AI - 面试引擎核心
"""

import json
import uuid
import random
import re
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from .model_processor import Qwen3VLProcessor
from .deepseek_processor import DeepSeekProcessor, HybridProcessor
from .multimodal_evaluator import MultimodalEvaluator
from .logger import logger, LogContext, log_function_call, performance_logger

# 导入评分配置
try:
    from .configs.scoring_config import calculate_dimension_score, get_scoring_config
    SCORING_CONFIG_AVAILABLE = True
except ImportError:
    try:
        from configs.scoring_config import calculate_dimension_score, get_scoring_config
        SCORING_CONFIG_AVAILABLE = True
    except ImportError:
        SCORING_CONFIG_AVAILABLE = False
        calculate_dimension_score = None
        get_scoring_config = None

# 导入降级决策中心
try:
    from .degradation import DegradationController
    DEGRADATION_AVAILABLE = True
except ImportError as e:
    logger.warning(f"降级决策中心导入失败: {e}")
    DEGRADATION_AVAILABLE = False
    DegradationController = None

# 导入新模块
try:
    from .database import db_manager
    from .cache_manager import CacheManager, CacheConfig
    from .security import validate_and_sanitize_answer
    from .error_handler import with_retry, DefaultEvaluationFallback, RetryPolicy
    NEW_MODULES_AVAILABLE = True
except ImportError as e:
    logger.warning(f"新模块导入失败: {e}")
    NEW_MODULES_AVAILABLE = False
    db_manager = None
    CacheManager = None
    CacheConfig = None
    validate_and_sanitize_answer = None
    with_retry = None

# 导入异步处理器
try:
    from .async_deepseek_processor import AsyncDeepSeekProcessor, BatchAsyncProcessor
    ASYNC_AVAILABLE = True
except ImportError as e:
    logger.warning(f"异步处理器导入失败: {e}")
    ASYNC_AVAILABLE = False
    AsyncDeepSeekProcessor = None
    BatchAsyncProcessor = None

# 导入配置
try:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from configs.config import EVALUATION_CONFIG, MULTIMODAL_CONFIG
except ImportError:
    EVALUATION_CONFIG = None
    MULTIMODAL_CONFIG = None


class InterviewStatus(Enum):
    """面试状态"""
    CREATED = "created"
    GREETING = "greeting"
    IN_PROGRESS = "in_progress"
    FOLLOW_UP = "follow_up"
    EVALUATING = "evaluating"
    COMPLETED = "completed"
    ABORTED = "aborted"


class InterviewPhase(Enum):
    """面试阶段"""
    SELF_INTRO = "self_intro"
    TECHNICAL = "technical"
    PROJECT = "project"
    BEHAVIORAL = "behavioral"
    CANDIDATE_QA = "candidate_qa"
    CLOSING = "closing"


@dataclass
class DialogueTurn:
    """对话回合"""
    turn_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    interviewer_question: str = ""
    question_id: Optional[str] = None
    question_type: str = ""
    difficulty: int = 1
    
    candidate_answer: str = ""
    has_audio: bool = False
    has_video: bool = False
    video_frame: Any = None  # 视频帧数据
    audio_path: Optional[str] = None  # 音频文件路径
    
    evaluation: Dict[str, Any] = field(default_factory=dict)
    scores: Dict[str, float] = field(default_factory=dict)
    
    is_follow_up: bool = False
    follow_up_count: int = 0
    parent_turn_id: Optional[str] = None  # 父回合ID（用于追问关联）


@dataclass
class InterviewSession:
    """面试会话"""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    position: str = ""
    candidate_name: str = ""
    status: InterviewStatus = InterviewStatus.CREATED
    current_phase: InterviewPhase = InterviewPhase.SELF_INTRO
    
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    
    dialogue_history: List[DialogueTurn] = field(default_factory=list)
    interview_plan: Dict[str, List[Dict]] = field(default_factory=dict)
    asked_question_ids: List[str] = field(default_factory=list)
    
    num_technical: int = 5
    num_project: int = 2
    num_behavioral: int = 2
    enable_follow_up: bool = True
    max_follow_up_depth: int = 2
    max_total_follow_ups: int = 10  # 全局追问上限，防止死循环

    used_transitions: List[str] = field(default_factory=list)  # 已使用的过渡语，避免重复

    total_questions_asked: int = 0
    total_follow_ups: int = 0

    # 超时配置（单位：秒）
    max_interview_duration_seconds: int = 3600  # 面试总时长限制：60分钟
    max_answer_wait_seconds: int = 300  # 单题回答等待限制：5分钟
    max_report_generation_seconds: int = 30  # 报告生成限制：30秒
    
    def get_current_turn(self) -> Optional[DialogueTurn]:
        if self.dialogue_history:
            return self.dialogue_history[-1]
        return None
    
    def get_progress(self) -> str:
        total = self.num_technical + self.num_project + self.num_behavioral + 1
        return f"{self.total_questions_asked}/{total}"

    def get_detailed_progress(self) -> Dict[str, Any]:
        """获取详细的面试进度信息"""
        plan = self.interview_plan
        asked_ids = set(self.asked_question_ids)

        # 计算各阶段进度
        phase_progress = {}
        for phase_type in ["technical", "project", "behavioral"]:
            questions = plan.get(phase_type, [])
            total = len(questions)
            asked = len([q for q in questions if q.get("id") in asked_ids])
            phase_progress[phase_type] = {
                "asked": asked,
                "total": total,
                "completed": asked >= total if total > 0 else True
            }

        # 计算总体进度
        total_planned = sum(len(plan.get(pt, [])) for pt in ["technical", "project", "behavioral"])
        total_asked = sum(phase_progress[pt]["asked"] for pt in ["technical", "project", "behavioral"])

        # 估算剩余时间（假设每题平均3分钟）
        avg_minutes_per_question = 3
        remaining_questions = max(0, total_planned - total_asked)
        estimated_remaining_minutes = remaining_questions * avg_minutes_per_question

        return {
            "current_phase": self.current_phase.value,
            "phase_progress": phase_progress,
            "overall": {
                "asked": total_asked,
                "total": total_planned,
                "percentage": round(total_asked / total_planned * 100, 1) if total_planned > 0 else 0
            },
            "follow_ups": {
                "total": self.total_follow_ups,
                "current_question": self.dialogue_history[-1].follow_up_count if self.dialogue_history else 0
            },
            "estimated_remaining_minutes": estimated_remaining_minutes
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典，用于数据库持久化"""
        return {
            "session_id": self.session_id,
            "candidate_name": self.candidate_name,
            "position": self.position,
            "status": self.status.value if isinstance(self.status, Enum) else str(self.status),
            "current_phase": self.current_phase.value if isinstance(self.current_phase, Enum) else str(self.current_phase),
            "created_at": self.created_at,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "total_questions_asked": self.total_questions_asked,
            "total_follow_ups": self.total_follow_ups,
            "dialogue_history": [
                {
                    "turn_id": turn.turn_id,
                    "interviewer_question": turn.interviewer_question,
                    "candidate_answer": turn.candidate_answer,
                    "question_id": turn.question_id,
                    "question_type": turn.question_type,
                    "difficulty": turn.difficulty,
                    "has_audio": turn.has_audio,
                    "has_video": turn.has_video,
                    "is_follow_up": turn.is_follow_up,
                    "follow_up_count": turn.follow_up_count,
                    "parent_turn_id": turn.parent_turn_id,
                    "scores": turn.scores,
                    "evaluation": turn.evaluation,
                    "timestamp": turn.timestamp
                }
                for turn in self.dialogue_history
            ]
        }


class QuestionBankManager:
    """题库管理器"""
    
    def __init__(self, knowledge_base_path: Optional[str] = None):
        logger.info("初始化题库管理器...")
        
        if knowledge_base_path is None:
            self.knowledge_base_path = Path(__file__).parent.parent / "knowledge_base" / "questions"
        else:
            self.knowledge_base_path = Path(knowledge_base_path)
        
        logger.info(f"题库路径: {self.knowledge_base_path}")
        
        self.question_banks: Dict[str, Dict] = {}
        self._load_all_question_banks()
    
    def _load_all_question_banks(self):
        """加载所有题库"""
        try:
            if not self.knowledge_base_path.exists():
                logger.error(f"题库路径不存在: {self.knowledge_base_path}")
                return
            
            json_files = list(self.knowledge_base_path.glob("*.json"))
            logger.info(f"发现 {len(json_files)} 个题库文件")
            
            for json_file in json_files:
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        position_code = data.get('position_code', json_file.stem)
                        self.question_banks[position_code] = data
                        question_count = len(data.get('questions', []))
                        logger.info(f"[OK] 加载题库: {data.get('position', position_code)} ({question_count}题)")
                except json.JSONDecodeError as e:
                    logger.error(f"[ERROR] 题库文件格式错误 {json_file}: {e}")
                except Exception as e:
                    logger.error(f"[ERROR] 加载题库失败 {json_file}: {e}")
            
            logger.info(f"题库加载完成，共 {len(self.question_banks)} 个岗位")
            
        except Exception as e:
            logger.error(f"加载题库过程中发生错误: {e}", exc_info=True)
    
    def get_position_code(self, position_name: str) -> Optional[str]:
        """获取岗位代码"""
        try:
            position_mapping = {
                "Java后端开发": "java_backend",
                "Java后端开发工程师": "java_backend",
                "Web前端开发": "web_frontend",
                "Web前端开发工程师": "web_frontend",
                "Python算法工程师": "python_algorithm",
            }
            
            if position_name in self.question_banks:
                return position_name
            
            code = position_mapping.get(position_name)
            if code and code in self.question_banks:
                return code
            
            for code, bank in self.question_banks.items():
                if position_name in bank.get('position', ''):
                    return code
            
            logger.warning(f"未找到岗位映射: {position_name}")
            return None
            
        except Exception as e:
            logger.error(f"获取岗位代码时发生错误: {e}")
            return None
    
    def get_questions(self, position: str, question_type: Optional[str] = None, 
                     difficulty: Optional[str] = None, count: int = 10) -> List[Dict]:
        """获取问题列表"""
        try:
            position_code = self.get_position_code(position)
            if not position_code or position_code not in self.question_banks:
                logger.warning(f"未找到岗位题库: {position}")
                return []
            
            bank = self.question_banks[position_code]
            questions = bank.get('questions', [])
            
            filtered = questions
            if question_type:
                filtered = [q for q in filtered if q.get('type') == question_type]
            if difficulty:
                filtered = [q for q in filtered if q.get('difficulty') == difficulty]
            
            if len(filtered) > count:
                filtered = random.sample(filtered, count)
            
            logger.debug(f"获取问题: 岗位={position}, 类型={question_type}, 数量={len(filtered)}")
            return filtered
            
        except Exception as e:
            logger.error(f"获取问题时发生错误: {e}")
            return []
    
    def generate_interview_plan(self, position: str, num_technical: int = 5, 
                               num_project: int = 2, num_behavioral: int = 2) -> Dict[str, List[Dict]]:
        """生成面试计划"""
        try:
            plan = {
                'technical': self.get_questions(position=position, question_type='technical', count=num_technical),
                'project': self.get_questions(position=position, question_type='project', count=num_project),
                'behavioral': self.get_questions(position=position, question_type='behavioral', count=num_behavioral)
            }
            
            total_questions = sum(len(v) for v in plan.values())
            logger.info(f"生成面试计划: {position} | 技术题={len(plan['technical'])}, "
                       f"项目题={len(plan['project'])}, 行为题={len(plan['behavioral'])}, "
                       f"总计={total_questions}")
            
            return plan
            
        except Exception as e:
            logger.error(f"生成面试计划时发生错误: {e}")
            return {'technical': [], 'project': [], 'behavioral': []}
    
    def evaluate_answer(self, position: str, question_id: str, answer: str) -> Dict[str, Any]:
        """评估回答"""
        try:
            position_code = self.get_position_code(position)
            if not position_code or position_code not in self.question_banks:
                logger.warning(f"无法评估: 未找到岗位题库 {position}")
                return {'score': 50, 'feedback': '无法评估'}
            
            bank = self.question_banks[position_code]
            questions = bank.get('questions', [])
            
            question = None
            for q in questions:
                if q.get('id') == question_id:
                    question = q
                    break
            
            if not question:
                logger.warning(f"无法评估: 未找到题目 {question_id}")
                return {'score': 60, 'feedback': '未找到题目'}
            
            keywords = question.get('keywords', [])
            if not keywords:
                logger.warning(f"无法评估: 题目 {question_id} 没有关键词")
                return {'score': 60, 'feedback': '暂无评估标准'}
            
            answer_lower = answer.lower()
            matched = [k for k in keywords if k.lower() in answer_lower]
            missed = [k for k in keywords if k.lower() not in answer_lower]
            
            match_ratio = len(matched) / len(keywords) if keywords else 0
            score = min(100, match_ratio * 100 + min(10, len(answer) / 50))
            
            if score >= 80:
                feedback = "回答优秀，涵盖了关键知识点。"
            elif score >= 60:
                feedback = f"回答良好，但缺少以下要点: {', '.join(missed[:3])}"
            else:
                feedback = f"回答不够完整，建议补充: {', '.join(missed[:3])}"
            
            logger.debug(f"评估结果: 题目={question_id}, 得分={score:.1f}, 匹配关键词={len(matched)}/{len(keywords)}")
            
            return {
                'score': round(score, 1),
                'matched_keywords': matched,
                'missed_keywords': missed,
                'feedback': feedback
            }
            
        except Exception as e:
            logger.error(f"评估回答时发生错误: {e}", exc_info=True)
            return {'score': 60, 'feedback': '评估过程发生错误'}


class InterviewEngine:
    """面试引擎 - 锐聘AI核心"""
    
    def __init__(self, model_path: Optional[str] = None):
        logger.info("初始化面试引擎...")
        
        self.question_bank = QuestionBankManager()
        self.session: Optional[InterviewSession] = None
        self.model_processor: Optional[Any] = None
        self.multimodal_evaluator: Optional[MultimodalEvaluator] = None

        # 初始化缓存管理器
        self.cache_manager = None
        if NEW_MODULES_AVAILABLE and CacheManager:
            try:
                cache_config = CacheConfig(backend="memory", default_ttl=3600)
                self.cache_manager = CacheManager(cache_config)
                logger.info("[OK] 缓存管理器初始化成功")
            except Exception as e:
                logger.warning(f"缓存管理器初始化失败: {e}")

        # 根据配置初始化评估处理器
        self._init_evaluation_processor(model_path)

        # 初始化多模态评估器（本地模型）
        self._init_multimodal_evaluator()

        # 过渡语模板
        self.transitions = {
            "to_technical": [
                "好的，了解了。接下来我们进入技术问答环节。",
                "谢谢你的介绍。下面我们来聊一些技术方面的问题。",
                "明白了。那接下来我想了解一下你的技术能力。"
            ],
            "to_project": [
                "技术问题聊得差不多了，我们来谈谈你的项目经验吧。",
                "好的。接下来想了解一下你在实际项目中的应用。",
                "了解了。下面能分享一些你参与的项目吗？"
            ],
            "to_behavioral": [
                "技术能力很不错。接下来我们聊一些行为方面的问题。",
                "好的。除了技术，我也想了解一下你的软技能。",
                "技术部分到此为止。我们来谈谈工作方式和团队协作。"
            ],
            "to_closing": [
                "面试快结束了，你有什么问题想问我的吗？",
                "好的，我的问题问完了。你有什么想了解的吗？",
                "面试基本结束了。你有什么问题想问我吗？"
            ]
        }

        # 追问模板
        self.follow_up_templates = {
            "low_score": [
                "这个回答还可以更详细一些。你能具体说说吗？",
                "我对这部分还挺感兴趣的，能再展开讲讲吗？",
                "你提到了这一点，能举个例子说明一下吗？"
            ],
            "medium_score": [
                "回答得不错。那在实际应用中有什么需要注意的吗？",
                "了解了。那这个技术和其他方案相比有什么优势？",
                "好的。如果遇到性能问题，你会怎么优化？"
            ],
            "high_score": [
                "回答得很专业！那深入问一下，底层原理是什么？",
                "很全面的回答。那如果让你设计这个系统，你会怎么设计？",
                "不错。那在分布式环境下有什么挑战？"
            ]
        }

        # 初始化异步处理器（可选）
        self.async_processor = None
        if ASYNC_AVAILABLE and AsyncDeepSeekProcessor:
            try:
                ds_config = EVALUATION_CONFIG.get("deepseek", {}) if EVALUATION_CONFIG else {}
                self.async_processor = AsyncDeepSeekProcessor(
                    api_key=ds_config.get("api_key"),
                    base_url=ds_config.get("base_url", "https://api.deepseek.com"),
                    model=ds_config.get("model", "deepseek-chat"),
                    timeout=ds_config.get("timeout", 10.0),
                    max_workers=3  # 线程池大小
                )
                logger.info("[OK] 异步处理器初始化成功")
            except Exception as e:
                logger.warning(f"异步处理器初始化失败: {e}")

        # 初始化降级决策中心
        self.degradation_controller = None
        self._degradation_level = 1
        if DEGRADATION_AVAILABLE and DegradationController:
            try:
                self.degradation_controller = DegradationController()
                self.degradation_controller.register_status_change_callback(
                    self._on_degradation_level_change
                )
                # 注册DeepSeek健康检查
                if hasattr(self.model_processor, 'register_health_check'):
                    self.model_processor.register_health_check()
                self._degradation_level = self.degradation_controller.get_current_level()
                logger.info(f"[OK] 降级决策中心集成完成 | 当前级别: Level {self._degradation_level}")
            except Exception as e:
                logger.warning(f"降级决策中心集成失败: {e}")

        logger.info("面试引擎初始化完成")

    def _on_degradation_level_change(self, old_level: int, new_level: int):
        """降级级别变更回调"""
        self._degradation_level = new_level
        logger.warning(f"[降级通知] 面试引擎感知到级别变更: Level {old_level} -> Level {new_level}")

        config = self.degradation_controller.get_level_config(new_level)
        if config:
            logger.info(f"  - 级别名称: {config.name}")
            logger.info(f"  - 评估器配置: {config.evaluators}")
            logger.info(f"  - 动作配置: {config.actions}")

            # 如果降级到 Level 3+ 且当前使用本地模型，切换到 4-bit 模式以节省显存
            if new_level >= 3 and self.multimodal_evaluator:
                try:
                    current_mode = self.multimodal_evaluator.current_mode
                    if current_mode not in ("full_4bit", "cpu_offload", "cpu_only"):
                        logger.warning(f"[降级] 降级到 Level {new_level}，切换模型到 4-bit 量化模式")
                        self.multimodal_evaluator.switch_mode("full_4bit")
                except Exception as e:
                    logger.error(f"[降级] 切换模型模式失败: {e}")

    def _get_evaluator_for_current_level(self):
        """根据当前降级级别获取合适的评估器"""
        if not self.degradation_controller:
            return self.model_processor

        level = self.degradation_controller.get_current_level()
        evaluators = self.degradation_controller.get_current_evaluators()
        primary = evaluators.get("primary", "deepseek_api")

        logger.info(f"[降级感知] 当前级别: Level {level}, 主评估器: {primary}")

        # Level 1-2: 使用DeepSeek API（同步或异步）
        if primary in ("deepseek_api", "deepseek_async"):
            if primary == "deepseek_async" and self.async_processor:
                logger.info("  -> 使用异步DeepSeek处理器")
                return self.async_processor
            elif self.model_processor and hasattr(self.model_processor, 'is_available') and self.model_processor.is_available():
                logger.info("  -> 使用同步DeepSeek处理器")
                return self.model_processor
            else:
                logger.warning("  -> DeepSeek不可用，降级到本地模型")
                return self.multimodal_evaluator

        # Level 3: 使用本地模型（4bit量化）
        elif primary in ("qwen3vl_local", "qwen3vl_local_4bit"):
            if self.multimodal_evaluator and self.multimodal_evaluator.is_available():
                logger.info("  -> 使用本地多模态评估器")
                return self.multimodal_evaluator
            else:
                logger.warning("  -> 本地模型不可用，降级到规则评估")
                return None

        # Level 4-5: 使用规则引擎
        elif primary in ("rule_based", "pass_fail_rule"):
            logger.info("  -> 使用规则引擎评估")
            return None

        # 默认回退
        logger.warning(f"  -> 未知评估器类型 '{primary}'，使用默认处理器")
        return self.model_processor
    
    def _init_evaluation_processor(self, model_path: Optional[str] = None):
        """根据配置初始化评估处理器"""
        try:
            # 获取配置
            provider = "hybrid"  # 默认使用混合模式
            if EVALUATION_CONFIG:
                provider = EVALUATION_CONFIG.get("provider", "hybrid")
                logger.info(f"评估提供商: {provider}")
            
            if provider == "deepseek":
                # 仅使用DeepSeek API
                logger.info("初始化DeepSeek API处理器...")
                ds_config = EVALUATION_CONFIG.get("deepseek", {}) if EVALUATION_CONFIG else {}
                self.model_processor = DeepSeekProcessor(
                    api_key=ds_config.get("api_key"),
                    base_url=ds_config.get("base_url", "https://api.deepseek.com"),
                    model=ds_config.get("model", "deepseek-chat"),
                    timeout=ds_config.get("timeout", 10.0),
                    max_retries=ds_config.get("max_retries", 3)
                )
                
                if self.model_processor.is_available():
                    logger.info("[OK] DeepSeek API处理器初始化成功")
                else:
                    logger.warning("DeepSeek API不可用，将使用备用评估")
                    
            elif provider == "local":
                # 仅使用本地模型
                logger.info("初始化本地模型处理器...")
                if model_path:
                    self.model_processor = Qwen3VLProcessor(model_path)
                    logger.info("[OK] 本地模型处理器初始化成功")
                else:
                    logger.warning("未提供模型路径，无法初始化本地处理器")
                    
            else:  # hybrid 或其他
                # 使用混合处理器（DeepSeek优先，本地备用）
                logger.info("初始化混合评估处理器（DeepSeek API优先）...")
                ds_config = EVALUATION_CONFIG.get("deepseek", {}) if EVALUATION_CONFIG else {}
                hybrid_config = EVALUATION_CONFIG.get("hybrid", {}) if EVALUATION_CONFIG else {}
                
                local_path = model_path
                if not local_path and MULTIMODAL_CONFIG:
                    local_path = str(MULTIMODAL_CONFIG.get("qwen3vl_model_path", ""))
                
                self.model_processor = HybridProcessor(
                    model_path=local_path if local_path else None,
                    api_key=ds_config.get("api_key"),
                    prefer_online=hybrid_config.get("prefer_online", True)
                )
                logger.info("[OK] 混合处理器初始化成功")
                
        except Exception as e:
            logger.error(f"初始化评估处理器失败: {e}", exc_info=True)
            logger.warning("将使用备用评估方案")
            self.model_processor = None

    def _init_multimodal_evaluator(self):
        """初始化多模态评估器（本地Qwen3-VL模型）"""
        try:
            logger.info("初始化多模态评估器...")

            # 候选路径列表（按优先级排序）
            candidate_paths = []

            # 1. 从配置获取路径
            if MULTIMODAL_CONFIG:
                config_path = MULTIMODAL_CONFIG.get("qwen3vl_model_path", "")
                if config_path:
                    # 如果是文件路径，取父目录；如果是目录路径，直接使用
                    path = Path(config_path)
                    if path.is_file() or path.suffix:
                        candidate_paths.append(str(path.parent))
                    else:
                        candidate_paths.append(str(path))

                # 2. 从多模态配置获取路径
                multimodal_path = EVALUATION_CONFIG.get("multimodal", {}).get("model_path", "") if EVALUATION_CONFIG else ""
                if multimodal_path and multimodal_path not in candidate_paths:
                    candidate_paths.append(multimodal_path)

            # 3. 默认路径
            for default_path in ["base_model", "models/qwen3-vl", "models"]:
                if default_path not in candidate_paths:
                    candidate_paths.append(default_path)

            # 查找第一个存在的有效模型目录
            model_path = None
            for candidate in candidate_paths:
                if Path(candidate).exists():
                    # 检查目录中是否有模型文件
                    model_files = list(Path(candidate).glob("*.safetensors")) + \
                                  list(Path(candidate).glob("pytorch_model.bin")) + \
                                  list(Path(candidate).glob("*.json"))
                    if model_files:
                        model_path = candidate
                        logger.info(f"找到有效模型路径: {model_path} ({len(model_files)} 个模型文件)")
                        break

            if not model_path:
                logger.warning(f"未找到有效的多模态模型路径，候选路径: {candidate_paths}，多模态评估将不可用")
                self.multimodal_evaluator = None
                return

            self.multimodal_evaluator = MultimodalEvaluator(model_path=model_path)

            if self.multimodal_evaluator.is_available():
                logger.info("[OK] 多模态评估器初始化成功")
            else:
                logger.warning("多模态评估器初始化失败，将仅使用文本评估")
                self.multimodal_evaluator = None

        except Exception as e:
            logger.error(f"初始化多模态评估器失败: {e}", exc_info=True)
            self.multimodal_evaluator = None
    
    @log_function_call()
    def create_session(self, position: str, candidate_name: str = "", 
                      num_technical: int = 5, num_project: int = 2, 
                      num_behavioral: int = 2, enable_follow_up: bool = True) -> InterviewSession:
        """创建面试会话"""
        try:
            logger.info(f"创建面试会话: 岗位={position}, 候选人={candidate_name or '匿名'}")
            
            # 输入验证
            if NEW_MODULES_AVAILABLE and validate_and_sanitize_answer:
                is_valid, sanitized_name, error = validate_and_sanitize_answer(candidate_name, max_length=50)
                if not is_valid:
                    logger.warning(f"候选人姓名验证失败: {error}")
                else:
                    candidate_name = sanitized_name
            
            interview_plan = self.question_bank.generate_interview_plan(
                position=position, num_technical=num_technical, 
                num_project=num_project, num_behavioral=num_behavioral
            )
            
            self.session = InterviewSession(
                position=position, candidate_name=candidate_name or "匿名候选人",
                interview_plan=interview_plan, num_technical=num_technical,
                num_project=num_project, num_behavioral=num_behavioral,
                enable_follow_up=enable_follow_up
            )
            
            # 保存到数据库
            if NEW_MODULES_AVAILABLE and db_manager:
                try:
                    db_manager.save_interview(self.session.to_dict())
                    logger.info(f"[OK] 面试会话已保存到数据库")
                except Exception as e:
                    logger.warning(f"保存面试会话到数据库失败: {e}")
            
            logger.info(f"[OK] 面试会话创建成功: {self.session.session_id}")
            return self.session
            
        except Exception as e:
            logger.error(f"创建面试会话时发生错误: {e}", exc_info=True)
            raise
    
    @log_function_call()
    def start_interview(self) -> Dict[str, Any]:
        """开始面试"""
        try:
            if not self.session:
                logger.error("面试会话不存在")
                return {"error": "请先创建面试会话", "error_code": "SES_001"}
            
            logger.info(f"开始面试: {self.session.session_id}")
            
            self.session.status = InterviewStatus.GREETING
            self.session.started_at = datetime.now().isoformat()
            
            # 生成个性化问候语
            name = self.session.candidate_name
            position = self.session.position
            
            hour = datetime.now().hour
            if 9 <= hour < 12:
                time_greeting = "上午好"
            elif 12 <= hour < 14:
                time_greeting = "中午好"
            elif 14 <= hour < 18:
                time_greeting = "下午好"
            else:
                time_greeting = "您好"
            
            greetings = [
                f"{name}，{time_greeting}！欢迎参加{position}的面试。我是今天的面试官。",
                f"{time_greeting}，{name}！感谢你抽出时间参加{position}的面试。",
                f"你好，{name}！很高兴见到你。今天我们将进行{position}的面试。"
            ]
            
            greeting = random.choice(greetings)
            first_question = "请先简单介绍一下自己，包括你的教育背景和工作经历。"
            full_message = f"{greeting}\n\n{first_question}"
            
            # 创建第一回合
            turn = DialogueTurn(
                interviewer_question=full_message,
                question_type="greeting",
                difficulty=1
            )
            self.session.dialogue_history.append(turn)
            
            logger.info(f"[OK] 面试开始成功: {self.session.session_id}")
            
            return {
                "session_id": self.session.session_id,
                "full_message": full_message,
                "phase": self.session.current_phase.value,
                "progress": self.session.get_progress()
            }
            
        except Exception as e:
            logger.error(f"开始面试时发生错误: {e}", exc_info=True)
            return {"error": "开始面试失败，请稍后重试", "error_code": "START_001"}
    
    def _check_interview_timeout(self) -> bool:
        """检查面试是否超时"""
        if not self.session or not self.session.started_at:
            return False
        try:
            start_time = datetime.fromisoformat(self.session.started_at)
            elapsed = (datetime.now() - start_time).total_seconds()
            if elapsed > self.session.max_interview_duration_seconds:
                logger.warning(f"面试超时 | 已用时{elapsed:.0f}秒 | 限制{self.session.max_interview_duration_seconds}秒")
                return True
        except Exception as e:
            logger.error(f"检查超时时发生错误: {e}")
        return False

    def _validate_answer(self, answer: str) -> Dict[str, Any]:
        """验证候选人回答的有效性

        检查项：
        1. 最大长度限制（防止恶意输入）
        2. 重复字符比例（防止填充内容）
        3. 空内容或仅空白字符
        """
        result = {"valid": True, "error": "", "error_code": ""}

        # 1. 空内容检查
        if not answer or not answer.strip():
            result["valid"] = False
            result["error"] = "回答内容不能为空"
            result["error_code"] = "VAL_001"
            return result

        # 2. 最大长度检查（10000字符）
        MAX_ANSWER_LENGTH = 10000
        if len(answer) > MAX_ANSWER_LENGTH:
            result["valid"] = False
            result["error"] = f"回答内容过长，请控制在{MAX_ANSWER_LENGTH}字符以内"
            result["error_code"] = "VAL_002"
            return result

        # 3. 重复字符比例检查（防止"aaaaaa"或"111111"等填充内容）
        if len(answer) >= 20:
            unique_chars = len(set(answer))
            unique_ratio = unique_chars / len(answer)
            if unique_ratio < 0.1:  # 重复字符比例超过90%
                result["valid"] = False
                result["error"] = "回答内容重复度过高，请提供更有意义的回答"
                result["error_code"] = "VAL_003"
                return result

        # 4. 连续重复字符检查（防止"aaaaaaaa"）
        import re
        if re.search(r'(.)\1{10,}', answer):  # 同一字符连续出现10次以上
            result["valid"] = False
            result["error"] = "回答内容包含过多连续重复字符"
            result["error_code"] = "VAL_004"
            return result

        return result

    @log_function_call()
    def process_answer(self, answer: str, has_audio: bool = False,
                      has_video: bool = False, video_frame=None, audio_path=None,
                      transcribed_text: str = "", transcription_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """处理候选人回答"""
        try:
            logger.info("开始处理候选人回答")

            if not self.session:
                logger.error("[ERROR] 面试会话不存在")
                return {"error": "面试会话不存在", "error_code": "SES_002"}

            # 检查面试超时
            if self._check_interview_timeout():
                logger.warning("面试已超时，强制结束")
                return self.end_interview()

            if self.session.status == InterviewStatus.COMPLETED:
                logger.warning("[WARN] 面试已结束，无法处理回答")
                return {"error": "面试已结束", "error_code": "SES_003"}

            # 输入验证和清理
            if NEW_MODULES_AVAILABLE and validate_and_sanitize_answer:
                is_valid, sanitized_answer, error = validate_and_sanitize_answer(answer)
                if not is_valid:
                    logger.warning(f"[WARN] 回答验证失败: {error}")
                    # 不过滤，只记录日志，继续处理
                else:
                    answer = sanitized_answer

            # 额外的输入验证：长度检查和重复内容检查
            validation_result = self._validate_answer(answer)
            if not validation_result["valid"]:
                logger.warning(f"[WARN] 回答验证失败: {validation_result['error']}")
                return {
                    "error": validation_result["error"],
                    "error_code": validation_result["error_code"],
                    "type": "validation_error"
                }

            logger.info(f"处理回答 | 会话={self.session.session_id} | "
                         f"长度={len(answer)}字符 | 音频={has_audio} | 视频={has_video} | "
                         f"视频帧={'有' if video_frame is not None else '无'} | "
                         f"转录={len(transcribed_text)}字符")

            # 更新当前回合
            current_turn = self.session.get_current_turn()
            if current_turn:
                logger.info("更新当前回合信息...")
                current_turn.candidate_answer = answer
                current_turn.has_audio = has_audio
                current_turn.has_video = has_video
                current_turn.video_frame = video_frame
                current_turn.audio_path = audio_path
                # 保存转录信息到回合中
                if transcribed_text:
                    current_turn.evaluation["transcribed_text"] = transcribed_text
                    current_turn.evaluation["transcription_info"] = transcription_info or {}
                logger.info("[OK] 回合信息更新完成")

            # 评估回答
            logger.info("开始评估回答...")
            with LogContext(logger, "回答评估"):
                evaluation = self._evaluate_current_answer(video_frame=video_frame)

            if current_turn:
                current_turn.evaluation = evaluation

                # 记录评估来源和详细分数（脱敏：不记录原始评估内容中的敏感信息）
                logger.info("=" * 60)
                logger.info("评估结果处理")
                logger.info("=" * 60)
                # 只记录评估的关键指标，不记录可能包含候选人信息的原始内容
                eval_summary = {k: v for k, v in evaluation.items() if k in ["score", "technical", "communication", "completeness", "problem_solving", "teamwork", "leadership"]}
                logger.info(f"评估摘要: {eval_summary}")

                # 检查是否有AI模型返回的详细维度分数（DeepSeek API返回的格式）
                has_dimension_scores = all(key in evaluation for key in ["technical", "communication", "completeness", "problem_solving", "teamwork", "leadership"])

                if has_dimension_scores:
                    # 使用AI模型返回的详细维度分数
                    current_turn.scores = {
                        "technical": evaluation.get("technical", 60),
                        "communication": evaluation.get("communication", 60),
                        "completeness": evaluation.get("completeness", 60),
                        "problem_solving": evaluation.get("problem_solving", 60),
                        "teamwork": evaluation.get("teamwork", 60),
                        "leadership": evaluation.get("leadership", 60)
                    }
                    logger.info("[OK] 使用AI模型详细评分")
                    logger.info(f"  - 技术能力: {current_turn.scores['technical']}")
                    logger.info(f"  - 表达能力: {current_turn.scores['communication']}")
                    logger.info(f"  - 完整度: {current_turn.scores['completeness']}")
                    logger.info(f"  - 问题解决: {current_turn.scores['problem_solving']}")
                    logger.info(f"  - 团队协作: {current_turn.scores['teamwork']}")
                    logger.info(f"  - 领导力: {current_turn.scores['leadership']}")
                else:
                    # 使用规则评估的分数（优化版）
                    base_score = evaluation.get("score", 60)

                    # 获取问题类型，用于动态调整评分权重
                    question_type = current_turn.question_type

                    # 优化后的完整度评分
                    completeness = self._evaluate_completeness(answer, has_audio, has_video)

                    # 根据问题类型调整各维度评分
                    current_turn.scores = self._calculate_dimension_scores(
                        base_score=base_score,
                        answer=answer,
                        question_type=question_type,
                        has_audio=has_audio,
                        has_video=has_video,
                        evaluation=evaluation
                    )
                    logger.info("[OK] 使用规则评估评分（优化版）")
                    logger.info(f"  - 基础得分: {base_score}")
                    logger.info(f"  - 问题类型: {question_type}")
                    logger.info(f"  - 各维度评分: {current_turn.scores}")
                
                # 缓存评估结果
                if self.cache_manager:
                    try:
                        self.cache_manager.set_evaluation(
                            answer=answer,
                            evaluation=current_turn.scores,
                            question=current_turn.interviewer_question
                        )
                        logger.info("[OK] 评估结果已缓存")
                    except Exception as e:
                        logger.warning(f"缓存评估结果失败: {e}")
                
                # 保存到数据库
                if NEW_MODULES_AVAILABLE and db_manager:
                    try:
                        db_manager.save_interview(self.session.to_dict())
                        logger.info("[OK] 会话状态已保存到数据库")
                    except Exception as e:
                        logger.warning(f"保存会话到数据库失败: {e}")
            
            # 更新状态
            self.session.status = InterviewStatus.IN_PROGRESS
            
            # 决定是否追问
            if self._should_follow_up():
                logger.info("生成追问问题")
                return self._generate_follow_up(evaluation)
            
            # 进入下一阶段或下一题
            return self._advance_to_next(evaluation)
            
        except Exception as e:
            logger.error(f"处理回答时发生错误: {e}", exc_info=True)
            return {"error": "系统处理异常，请稍后重试", "error_code": "PROC_001"}
    
    def _evaluate_current_answer(self, video_frame=None) -> Dict[str, Any]:
        """评估当前回答 - 集成降级策略"""
        eval_start_time = time.time()
        logger.debug("开始评估当前回答")

        try:
            if not self.session:
                logger.error("评估失败: 面试会话不存在")
                return {}

            current_turn = self.session.get_current_turn()
            if not current_turn:
                logger.error("评估失败: 当前回合不存在")
                return {}

            answer = current_turn.candidate_answer
            question_id = current_turn.question_id
            question = current_turn.interviewer_question
            has_audio = current_turn.has_audio
            has_video = current_turn.has_video

            logger.info(f"评估回答 | 会话={self.session.session_id} | 回合={current_turn.turn_id} | "
                         f"级别=Level{self._degradation_level} | 回答长度={len(answer)}字符 | "
                         f"音频={has_audio} | 视频={has_video}")

            # ===== 降级感知评估器选择 =====
            evaluator = self._get_evaluator_for_current_level()

            # Level 1-2: 优先使用多模态评估（如果有视频帧）
            if self._degradation_level <= 2:
                if video_frame is not None and self.multimodal_evaluator and self.multimodal_evaluator.is_available():
                    logger.info("[Level 1-2] 优先使用多模态评估器（文本+视频）...")
                    try:
                        with LogContext(logger, "多模态评估"):
                            evaluation = self.multimodal_evaluator.evaluate(
                                answer=answer,
                                question=question or "",
                                video_frame=video_frame,
                                has_audio=has_audio
                            )

                        eval_time = time.time() - eval_start_time
                        logger.info(f"多模态评估完成 | 得分={evaluation.get('score', 0):.1f} | 耗时={eval_time:.3f}秒")
                        return evaluation

                    except Exception as e:
                        logger.error(f"多模态评估失败: {e}")
                        logger.info("将尝试降级评估方案...")

            # Level 1-3: 使用AI模型评估（同步或异步）
            if self._degradation_level <= 3 and evaluator is not None:
                # 异步模式（Level 2性能降级时启用）
                if (self._degradation_level == 2 and
                    self.async_processor and
                    hasattr(self.async_processor, '_processor') and
                    self.async_processor._processor.is_available()):
                    logger.info("[Level 2] 使用异步处理器进行评估（非阻塞模式）...")
                    try:
                        task_id = self.async_processor.evaluate_answer_async(
                            answer=answer,
                            question=question,
                            has_audio=has_audio,
                            has_video=has_video,
                            image=video_frame
                        )

                        evaluation = self.async_processor.get_result(task_id, timeout=30)

                        if evaluation:
                            eval_time = time.time() - eval_start_time
                            logger.info(f"异步评估完成 | 得分={evaluation.get('score', 0):.1f} | 耗时={eval_time:.3f}秒")
                            return evaluation
                        else:
                            logger.warning("异步评估超时，将使用同步评估...")
                    except Exception as e:
                        logger.error(f"异步评估失败: {e}")
                        logger.info("将使用同步评估...")

                # 同步模式
                logger.info(f"[Level {self._degradation_level}] 使用AI模型处理器进行评估...")
                try:
                    with LogContext(logger, "AI模型评估"):
                        logger.info(f"调用评估处理器:")
                        logger.info(f"  - 处理器: {type(evaluator).__name__}")
                        logger.info(f"  - 问题: {question[:50] + '...' if question and len(question) > 50 else question}")

                        evaluation = evaluator.evaluate_answer(
                            answer=answer,
                            question=question,
                            has_audio=has_audio,
                            has_video=has_video,
                            image=video_frame
                        )

                    eval_time = time.time() - eval_start_time
                    logger.info(f"AI模型评估完成 | 得分={evaluation.get('score', 0):.1f} | "
                                 f"技术={evaluation.get('technical', 0)} | 表达={evaluation.get('communication', 0)} | "
                                 f"耗时={eval_time:.3f}秒")
                    return evaluation

                except Exception as e:
                    logger.error(f"AI模型评估失败: {e}")
                    logger.error(f"错误类型: {type(e).__name__}")
                    logger.info("将尝试备用评估方案...")
            else:
                logger.warning(f"[Level {self._degradation_level}] AI评估不可用，将使用备用评估")

            # Level 4-5: 规则引擎评估
            logger.info(f"[Level {self._degradation_level}] 使用规则引擎评估...")
            if question_id:
                logger.info(f"使用题库评估: 问题ID={question_id}")
                result = self.question_bank.evaluate_answer(
                    position=self.session.position,
                    question_id=question_id,
                    answer=answer
                )
                logger.info(f"题库评估完成: 得分={result.get('score', 0)}")
                return result

            # 通用评估（自我介绍等）
            logger.info("使用通用规则评估...")
            length = len(answer)
            if length < 20:
                score, feedback = 40, "回答过于简短，请详细说明。"
            elif length < 50:
                score, feedback = 60, "回答尚可，但可以更加详细。"
            elif length < 100:
                score, feedback = 75, "回答不错，涵盖了主要内容。"
            else:
                score, feedback = 85, "回答详细完整，表达清晰。"

            eval_time = time.time() - eval_start_time
            logger.info(f"通用评估完成 | 得分={score} | 耗时={eval_time:.3f}秒")
            return {"score": score, "feedback": feedback}

        except Exception as e:
            eval_time = time.time() - eval_start_time
            logger.error(f"评估失败 | 类型={type(e).__name__} | 耗时={eval_time:.3f}秒 | 错误={e}", exc_info=True)
            return {"score": 60, "feedback": "评估过程发生错误"}
    
    def _evaluate_completeness(self, answer: str, has_audio: bool = False, has_video: bool = False) -> float:
        """评估回答完整度（优化版）"""
        try:
            if not answer or not answer.strip():
                if has_audio and has_video:
                    return 75.0
                elif has_audio or has_video:
                    return 60.0
                else:
                    return 0.0

            length = len(answer)

            # 基础分根据回答长度计算（非线性，避免短回答分数过低）
            if length < 10:
                base = 20.0
            elif length < 30:
                base = 35.0
            elif length < 60:
                base = 50.0
            elif length < 120:
                base = 65.0
            elif length < 200:
                base = 75.0
            else:
                base = 85.0

            # 内容质量加分
            bonus = 0.0
            sentences = answer.count('。') + answer.count('.') + answer.count('；') + 1
            if sentences >= 3:
                bonus += 5.0
            if sentences >= 5:
                bonus += 5.0

            # 结构化表达加分
            structure_markers = ['首先', '其次', '然后', '最后', '第一', '第二', '第三',
                                '1.', '2.', '3.', '①', '②', '③']
            for marker in structure_markers:
                if marker in answer:
                    bonus += 3.0
                    break

            # 举例说明加分
            example_markers = ['例如', '比如', '举个例子', '如', 'instance']
            for marker in example_markers:
                if marker in answer:
                    bonus += 3.0
                    break

            # 音视频额外加分
            if has_audio:
                bonus += 2.0
            if has_video:
                bonus += 2.0

            return min(100.0, base + bonus)
        except Exception as e:
            logger.error(f"评估完整度时发生错误: {e}")
            return 50.0

    def _evaluate_communication(self, answer: str, has_audio: bool = False, has_video: bool = False) -> float:
        """评估沟通能力（优化版）"""
        try:
            if not answer or not answer.strip():
                if has_audio:
                    return 70.0
                if has_video:
                    return 65.0
                return 30.0

            # 基础分根据表达清晰度
            score = 50.0

            # 逻辑连接词加分
            connectors = ['首先', '其次', '最后', '总结', '因为', '所以', '例如', '但是', '因此']
            conn_count = sum(1 for c in connectors if c in answer)
            score += min(15.0, conn_count * 3.0)

            # 段落结构加分
            if '\n' in answer:
                score += 5.0

            # 回答长度适中加分（避免过短或过长）
            length = len(answer)
            if 50 <= length <= 300:
                score += 10.0
            elif 30 <= length < 50:
                score += 5.0

            # 专业术语使用加分
            tech_terms = ['通过', '使用', '实现', '设计', '优化', '架构', '方案']
            term_count = sum(1 for t in tech_terms if t in answer)
            score += min(10.0, term_count * 2.0)

            # 音视频额外加分
            if has_audio:
                score += 3.0
            if has_video:
                score += 3.0

            return min(100.0, score)
        except Exception as e:
            logger.error(f"评估沟通能力时发生错误: {e}")
            return 50.0

    def _evaluate_problem_solving(self, answer: str, has_audio: bool = False, has_video: bool = False) -> float:
        """评估问题解决能力（优化版）"""
        try:
            if not answer or not answer.strip():
                if has_audio or has_video:
                    return 55.0
                return 25.0

            score = 45.0

            # 分析方法加分
            analysis_words = ['分析', '排查', '定位', '诊断', '调研']
            for word in analysis_words:
                if word in answer:
                    score += 8.0
                    break

            # 解决方案加分
            solution_words = ['解决', '方案', '优化', '改进', '修复']
            for word in solution_words:
                if word in answer:
                    score += 8.0
                    break

            # 步骤描述加分
            if '步骤' in answer or '流程' in answer or '首先' in answer:
                score += 10.0

            # 结果验证加分
            result_words = ['验证', '测试', '确认', '效果', '结果']
            for word in result_words:
                if word in answer:
                    score += 5.0
                    break

            # 深度思考加分
            deep_words = ['根本原因', '底层', '原理', '机制', '架构']
            for word in deep_words:
                if word in answer:
                    score += 8.0
                    break

            return min(100.0, score)
        except Exception as e:
            logger.error(f"评估问题解决能力时发生错误: {e}")
            return 45.0

    def _evaluate_teamwork(self, answer: str, has_audio: bool = False, has_video: bool = False) -> float:
        """评估团队协作能力（优化版）"""
        try:
            if not answer or not answer.strip():
                if has_audio or has_video:
                    return 55.0
                return 30.0

            score = 45.0

            # 团队相关词汇加分
            teamwork_words = ['团队', '合作', '协作', '沟通', '配合', '同事']
            for word in teamwork_words:
                if word in answer:
                    score += 10.0
                    break

            # 协调能力加分
            if '协调' in answer or '分工' in answer or '协作' in answer:
                score += 10.0

            # 冲突处理加分
            conflict_words = ['冲突', '分歧', '协商', '调解', '共识']
            for word in conflict_words:
                if word in answer:
                    score += 8.0
                    break

            # 分享互助加分
            share_words = ['分享', '帮助', '支持', '指导', '学习']
            for word in share_words:
                if word in answer:
                    score += 5.0
                    break

            return min(100.0, score)
        except Exception as e:
            logger.error(f"评估团队协作能力时发生错误: {e}")
            return 45.0

    def _evaluate_leadership(self, answer: str, has_audio: bool = False, has_video: bool = False) -> float:
        """评估领导力（优化版）"""
        try:
            if not answer or not answer.strip():
                if has_audio or has_video:
                    return 50.0
                return 25.0

            score = 40.0

            # 责任承担加分
            responsibility_words = ['负责', '主导', '带领', '组织', '统筹']
            for word in responsibility_words:
                if word in answer:
                    score += 12.0
                    break

            # 决策能力加分
            decision_words = ['决策', '决定', '选择', '评估', '权衡']
            for word in decision_words:
                if word in answer:
                    score += 8.0
                    break

            # 项目管理加分
            if '项目' in answer and ('负责' in answer or '管理' in answer):
                score += 10.0

            # 团队建设加分
            team_build_words = ['培养', '激励', '成长', '提升', '发展']
            for word in team_build_words:
                if word in answer:
                    score += 8.0
                    break

            return min(100.0, score)
        except Exception as e:
            logger.error(f"评估领导力时发生错误: {e}")
            return 40.0

    def _calculate_dimension_scores(self, base_score: float, answer: str, question_type: str,
                                    has_audio: bool, has_video: bool, evaluation: Dict) -> Dict[str, float]:
        """
        根据问题类型动态计算各维度评分（优化版）

        不同问题类型有不同的评分权重：
        - technical: 技术能力权重高
        - project: 问题解决和领导力权重高
        - behavioral: 团队协作和沟通权重高
        - greeting/self_intro: 沟通和完整度权重高
        - candidate_qa: 沟通权重高
        """
        try:
            # 基础维度评分
            communication = self._evaluate_communication(answer, has_audio, has_video)
            completeness = self._evaluate_completeness(answer, has_audio, has_video)
            problem_solving = self._evaluate_problem_solving(answer, has_audio, has_video)
            teamwork = self._evaluate_teamwork(answer, has_audio, has_video)
            leadership = self._evaluate_leadership(answer, has_audio, has_video)

            # 提取多模态评估的视觉维度分数（如果有）
            visual_scores = {
                "expression": evaluation.get("expression"),
                "eye_contact": evaluation.get("eye_contact"),
                "body_language": evaluation.get("body_language"),
                "overall_image": evaluation.get("overall_image")
            }

            # 计算视觉平均分（用于调整沟通分数）
            visual_avg = None
            valid_visual = [v for v in visual_scores.values() if v is not None and v > 0]
            if valid_visual:
                visual_avg = sum(valid_visual) / len(valid_visual)

            # 根据问题类型调整各维度权重和分数
            if question_type == "technical":
                # 技术问题：技术能力权重最高
                technical = base_score * 0.7 + problem_solving * 0.3
                # 技术问题中沟通和团队协作权重降低
                communication = communication * 0.6 + 20  # 降低沟通分权重
                teamwork = teamwork * 0.5 + 25
                leadership = leadership * 0.5 + 25
            elif question_type == "project":
                # 项目问题：技术、问题解决、领导力权重高
                technical = base_score * 0.4 + problem_solving * 0.4 + leadership * 0.2
                problem_solving = problem_solving * 0.8 + 15
                leadership = min(100, leadership * 1.1 + 5)
                teamwork = teamwork * 0.7 + 15
            elif question_type == "behavioral":
                # 行为问题：团队协作和沟通权重高，技术权重低
                technical = base_score * 0.2 + communication * 0.4 + teamwork * 0.4
                teamwork = min(100, teamwork * 1.1 + 5)
                communication = min(100, communication * 1.1 + 5)
                leadership = leadership * 0.8 + 10
            elif question_type in ["greeting", "self_intro"]:
                # 自我介绍：沟通和完整度权重最高
                technical = base_score * 0.2 + communication * 0.4 + completeness * 0.4
                communication = min(100, communication * 1.15 + 3)
                completeness = min(100, completeness * 1.1 + 2)
                problem_solving = problem_solving * 0.5 + 25
                teamwork = teamwork * 0.5 + 25
                leadership = leadership * 0.5 + 25
            elif question_type == "candidate_qa":
                # 候选人提问：主要评估沟通
                technical = base_score * 0.2 + communication * 0.5 + problem_solving * 0.3
                communication = min(100, communication * 1.1 + 5)
            else:
                # 默认情况：均衡权重
                technical = base_score * 0.5 + problem_solving * 0.3 + communication * 0.2

            # 如果有有效的视觉评分，调整沟通分数（非语言沟通）
            if visual_avg is not None and visual_avg > 0:
                # 视觉评分占沟通分数的15%
                communication = communication * 0.85 + visual_avg * 0.15
                logger.info(f"  - 视觉评分融入: 视觉平均分={visual_avg:.1f}, 调整后沟通分={communication:.1f}")

            # 确保所有分数在合理范围内
            return {
                "technical": round(min(100, max(0, technical)), 1),
                "communication": round(min(100, max(0, communication)), 1),
                "completeness": round(min(100, max(0, completeness)), 1),
                "problem_solving": round(min(100, max(0, problem_solving)), 1),
                "teamwork": round(min(100, max(0, teamwork)), 1),
                "leadership": round(min(100, max(0, leadership)), 1)
            }

        except Exception as e:
            logger.error(f"计算维度分数时发生错误: {e}")
            # 返回基础分数
            return {
                "technical": round(base_score, 1),
                "communication": 50.0,
                "completeness": 50.0,
                "problem_solving": 50.0,
                "teamwork": 50.0,
                "leadership": 50.0
            }
    
    def _should_follow_up(self) -> bool:
        """判断是否需要追问"""
        try:
            logger.debug("判断是否需要追问")
            
            if not self.session:
                logger.info("不追问: 会话不存在")
                return False
            
            if not self.session.enable_follow_up:
                logger.info("不追问: 追问功能已禁用")
                return False
            
            current_turn = self.session.get_current_turn()
            if not current_turn:
                logger.info("不追问: 当前回合不存在")
                return False
            
            # 自我介绍阶段不追问
            if self.session.current_phase == InterviewPhase.SELF_INTRO:
                logger.info("不追问: 自我介绍阶段")
                return False
            
            # 面试结束阶段不追问
            if self.session.current_phase == InterviewPhase.CLOSING:
                logger.info("不追问: 面试已结束")
                return False
            
            # 候选人提问阶段不追问（该阶段是候选人向面试官提问）
            if self.session.current_phase == InterviewPhase.CANDIDATE_QA:
                logger.info("不追问: 候选人提问阶段")
                return False
            
            # 检查全局追问上限（防止死循环）
            global_follow_up_count = len([t for t in self.session.dialogue_history if t.is_follow_up])
            if global_follow_up_count >= self.session.max_total_follow_ups:
                logger.info(f"不追问: 已达到全局追问上限 ({global_follow_up_count}/{self.session.max_total_follow_ups})")
                return False
            
            # 找到当前问题链的原始问题（非追问）和追问次数
            # 当前回合可能是追问，需要找到它关联的原始问题
            original_question_id = current_turn.question_id
            current_follow_up_count = current_turn.follow_up_count
            
            # 如果当前回合是追问，检查是否已达最大追问深度
            if current_turn.is_follow_up:
                # 追问回合的 follow_up_count 表示该问题链已追问的次数
                # 当前回合本身就是追问，所以已追问次数就是 follow_up_count
                if current_follow_up_count >= self.session.max_follow_up_depth:
                    logger.info(f"不追问: 追问回合已达最大追问深度 ({current_follow_up_count}/{self.session.max_follow_up_depth})")
                    return False
                logger.info(f"当前是追问回合，已追问 {current_follow_up_count} 次，未达最大深度 {self.session.max_follow_up_depth}")
            else:
                # 原始问题回合
                logger.info(f"当前是原始问题回合，已追问 {current_follow_up_count} 次")
            
            # 检查是否有评估结果
            evaluation = current_turn.evaluation
            if not evaluation:
                logger.warning("无评估结果，使用默认分数判断")
                score = 60
            else:
                score = evaluation.get("score", 60)
            
            answer_length = len(current_turn.candidate_answer) if current_turn.candidate_answer else 0
            
            logger.info(f"追问判断条件:")
            logger.info(f"  - 当前阶段: {self.session.current_phase.value}")
            logger.info(f"  - 当前问题ID: {original_question_id}")
            logger.info(f"  - 是否追问回合: {current_turn.is_follow_up}")
            logger.info(f"  - 已追问次数: {current_follow_up_count}")
            logger.info(f"  - 评估得分: {score}")
            logger.info(f"  - 回答长度: {answer_length}字符")
            
            # 如果当前回合已经是追问且得分仍然很低，不再继续追问（防止无限循环）
            if current_turn.is_follow_up and score < 60:
                logger.info(f"不追问: 追问后得分仍低于60，进入下一题")
                return False
            
            # 低分一定追问（仅对原始问题）
            if score < 60 and not current_turn.is_follow_up:
                logger.info("决定追问: 原始问题得分低于60，需要深入考察")
                return True
            
            # 中等分数50%概率追问（仅对原始问题）
            if 60 <= score < 75 and not current_turn.is_follow_up:
                should_follow = random.random() < 0.5
                logger.info(f"决定{'追问' if should_follow else '不追问'}: 中等得分({score})，概率50% -> {should_follow}")
                return should_follow
            
            # 高分但内容简短，可能值得追问（仅对原始问题）
            if score >= 75 and answer_length < 50 and not current_turn.is_follow_up:
                should_follow = random.random() < 0.3
                logger.info(f"决定{'追问' if should_follow else '不追问'}: 高分({score})但回答简短({answer_length}字符)，概率30% -> {should_follow}")
                return should_follow
            
            logger.info(f"不追问: 得分{score}良好且回答完整({answer_length}字符)")
            return False
            
        except Exception as e:
            logger.error(f"判断追问时发生错误: {e}", exc_info=True)
            return False
    
    def _generate_follow_up(self, evaluation: Dict) -> Dict[str, Any]:
        """生成追问 - 基于当前问题内容生成相关追问"""
        try:
            score = evaluation.get("score", 60)
            current_turn = self.session.get_current_turn()

            if not current_turn:
                logger.error("无法生成追问：当前回合不存在")
                return self._advance_to_next(evaluation)

            # 找到原始问题回合（非追问回合）
            # 如果当前回合是追问，需要追溯到原始父回合
            original_turn = current_turn
            if current_turn.is_follow_up and current_turn.parent_turn_id:
                for turn in reversed(self.session.dialogue_history):
                    if turn.turn_id == current_turn.parent_turn_id:
                        original_turn = turn
                        break

            # 获取当前问题的信息
            current_question = current_turn.interviewer_question
            current_question_id = current_turn.question_id
            current_question_type = current_turn.question_type
            candidate_answer = current_turn.candidate_answer
            follow_up_count = original_turn.follow_up_count

            logger.info(f"生成追问: 当前问题ID={current_question_id}, 类型={current_question_type}, 已追问={follow_up_count}次")
            logger.info(f"当前问题: {current_question[:80]}...")
            logger.info(f"候选人回答长度: {len(candidate_answer) if candidate_answer else 0}字符")

            # 基于当前问题内容生成相关追问
            question = self._generate_contextual_follow_up(
                current_question=current_question,
                current_question_id=current_question_id,
                current_question_type=current_question_type,
                candidate_answer=candidate_answer,
                score=score,
                follow_up_count=follow_up_count,
                evaluation=evaluation
            )

            # 原子操作：先创建回合（在新回合中记录追问次数），再更新计数
            new_follow_up_count = original_turn.follow_up_count + 1
            turn = DialogueTurn(
                interviewer_question=question,
                question_id=current_question_id,
                question_type=current_question_type,
                difficulty=current_turn.difficulty,
                is_follow_up=True,
                parent_turn_id=original_turn.turn_id,  # 始终关联到原始回合
                follow_up_count=new_follow_up_count  # 在新回合中记录追问深度
            )

            # 一次性更新：追加回合、更新原始回合计数、更新全局计数
            self.session.dialogue_history.append(turn)
            original_turn.follow_up_count = new_follow_up_count
            self.session.total_follow_ups += 1
            
            self.session.status = InterviewStatus.FOLLOW_UP
            
            logger.info(f"[OK] 生成追问完成: 得分={score:.1f}, 追问次数={current_turn.follow_up_count}")
            logger.info(f"追问内容: {question}")
            
            return {
                "type": "follow_up",
                "question": question,
                "evaluation": evaluation,
                "phase": self.session.current_phase.value,
                "progress": self.session.get_detailed_progress(),
                "feedback": evaluation.get("feedback", "")
            }
            
        except Exception as e:
            logger.error(f"生成追问时发生错误: {e}", exc_info=True)
            logger.info("追问生成失败，进入下一题")
            return self._advance_to_next(evaluation)
    
    def _generate_contextual_follow_up(
        self,
        current_question: str,
        current_question_id: Optional[str],
        current_question_type: str,
        candidate_answer: str,
        score: float,
        follow_up_count: int,
        evaluation: Dict
    ) -> str:
        """
        基于上下文生成相关追问
        
        策略：
        1. 如果有题库中的追问问题，优先使用
        2. 根据回答内容提取关键词生成追问
        3. 根据得分选择不同深度的追问
        """
        try:
            # 尝试从题库获取追问问题
            if current_question_id and self.question_bank:
                follow_up_questions = self._get_follow_up_from_bank(
                    current_question_id, follow_up_count, score
                )
                if follow_up_questions:
                    question = random.choice(follow_up_questions)
                    logger.info(f"从题库获取追问: {question[:60]}...")
                    return question
            
            # 根据回答内容生成上下文相关的追问
            contextual_question = self._generate_follow_up_from_answer(
                current_question=current_question,
                candidate_answer=candidate_answer or "",
                score=score,
                follow_up_count=follow_up_count,
                evaluation=evaluation
            )
            
            if contextual_question:
                logger.info(f"生成上下文追问: {contextual_question[:60]}...")
                return contextual_question
            
            # 回退到通用追问模板（但会添加上下文提示）
            generic_question = self._get_generic_follow_up(score, follow_up_count)
            logger.info(f"使用通用追问: {generic_question[:60]}...")
            return generic_question
            
        except Exception as e:
            logger.error(f"生成上下文追问失败: {e}")
            return self._get_generic_follow_up(score, follow_up_count)
    
    def _get_follow_up_from_bank(
        self,
        question_id: str,
        follow_up_count: int,
        score: float
    ) -> List[str]:
        """从题库获取追问问题"""
        try:
            # 查找当前问题的追问列表
            for position_data in self.question_bank.question_banks.values():
                questions = position_data.get("questions", [])
                for q in questions:
                    if q.get("id") == question_id:
                        follow_ups = q.get("follow_ups", [])
                        if follow_ups and follow_up_count < len(follow_ups):
                            return [follow_ups[follow_up_count]]
            return []
        except Exception as e:
            logger.error(f"从题库获取追问失败: {e}")
            return []
    
    def _generate_follow_up_from_answer(
        self,
        current_question: str,
        candidate_answer: str,
        score: float,
        follow_up_count: int,
        evaluation: Dict
    ) -> Optional[str]:
        """基于回答内容生成追问"""
        try:
            # 提取回答中的关键词
            keywords = self._extract_keywords(candidate_answer)
            feedback = evaluation.get("feedback", "")
            
            # 根据追问深度和得分生成不同追问
            if follow_up_count == 0:
                # 第一次追问：深入回答中的某个点
                if score < 60:
                    # 低分：要求补充细节
                    if keywords:
                        return f"你提到了'{keywords[0]}'，能详细解释一下这个概念吗？"
                    return "你能更详细地说明一下你的思路吗？"
                else:
                    # 高分：要求深入原理
                    if keywords:
                        return f"你提到了'{keywords[0]}'，它的底层原理是什么？"
                    return "能深入讲讲背后的实现原理吗？"
            
            elif follow_up_count == 1:
                # 第二次追问：实际应用或对比
                if score < 70:
                    return "在实际项目中，你是如何应用这个技术的？"
                else:
                    return "这个方案和其他替代方案相比有什么优缺点？"
            
            # 根据反馈内容生成追问
            if "详细" in feedback or "简略" in feedback:
                return "能举个例子来说明你的观点吗？"
            elif "原理" in feedback or "深入" in feedback:
                return "如果让你来优化这个实现，你会怎么做？"
            elif "经验" in feedback:
                return "在实际项目中遇到过什么相关问题吗？"
            
            return None
            
        except Exception as e:
            logger.error(f"从回答生成追问失败: {e}")
            return None
    
    def _extract_keywords(self, text: str) -> List[str]:
        """从文本中提取技术关键词"""
        try:
            if not text:
                return []
            
            # 常见的技术关键词模式
            tech_patterns = [
                r'\b[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*\b',  # 驼峰命名
                r'\b[a-z]+[A-Z][a-zA-Z]*\b',  # 小驼峰
                r'\b(?:Spring|Java|Python|React|Vue|Node|Docker|K8s|Kubernetes|Redis|MySQL|MongoDB|Elasticsearch|Kafka|RabbitMQ)\b',
                r'\b(?:多线程|并发|异步|同步|缓存|索引|事务|锁|队列|栈|链表|树|图)\b',
            ]
            
            keywords = []
            for pattern in tech_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                keywords.extend(matches)
            
            # 去重并过滤短词
            keywords = list(set([k for k in keywords if len(k) > 2]))
            
            # 按在文本中出现的位置排序
            keywords.sort(key=lambda x: text.lower().find(x.lower()))
            
            return keywords[:3]  # 返回前3个关键词
            
        except Exception as e:
            logger.error(f"提取关键词失败: {e}")
            return []
    
    def _get_generic_follow_up(self, score: float, follow_up_count: int) -> str:
        """获取通用追问模板"""
        try:
            # 根据追问深度选择不同模板
            if follow_up_count == 0:
                templates = {
                    "low": [
                        "这个回答还可以更详细一些。你能具体说说吗？",
                        "我对这部分还挺感兴趣的，能再展开讲讲吗？",
                        "你提到了这一点，能举个例子说明一下吗？"
                    ],
                    "medium": [
                        "回答得不错。那在实际应用中有什么需要注意的吗？",
                        "了解了。那这个技术和其他方案相比有什么优势？",
                        "好的。如果遇到性能问题，你会怎么优化？"
                    ],
                    "high": [
                        "回答得很专业！那深入问一下，底层原理是什么？",
                        "很全面的回答。那如果让你设计这个系统，你会怎么设计？",
                        "不错。那在分布式环境下有什么挑战？"
                    ]
                }
            else:
                # 第二次追问
                templates = {
                    "low": [
                        "能再详细说明一下实现细节吗？",
                        "实际项目中是怎么用的？",
                        "有没有遇到过什么坑？"
                    ],
                    "medium": [
                        "性能方面有什么考虑？",
                        "如何保证高可用？",
                        "有没有更好的替代方案？"
                    ],
                    "high": [
                        "如果数据量很大，怎么优化？",
                        "在微服务架构中怎么部署？",
                        "如何监控和排查问题？"
                    ]
                }
            
            # 根据得分选择模板组
            if score < 60:
                group = "low"
            elif score < 80:
                group = "medium"
            else:
                group = "high"
            
            return random.choice(templates[group])
            
        except Exception as e:
            logger.error(f"获取通用追问失败: {e}")
            return "能再详细说说吗？"
    
    def _advance_to_next(self, evaluation: Dict) -> Dict[str, Any]:
        """进入下一题或下一阶段"""
        try:
            old_phase = self.session.current_phase
            
            # 更新阶段
            self._update_phase()
            new_phase = self.session.current_phase
            
            # 检查是否结束
            if self._should_end_interview():
                return self._end_interview()
            
            # 获取下一个问题
            next_question = self._get_next_question()
            
            # 添加过渡语（如果阶段变化）
            question_text = next_question["question"]
            if old_phase != new_phase:
                transition = self._get_transition(old_phase, new_phase)
                if transition:
                    question_text = f"{transition}\n\n{question_text}"
            
            # 创建新回合
            turn = DialogueTurn(
                interviewer_question=question_text,
                question_id=next_question.get("id"),
                question_type=next_question.get("type", "technical"),
                difficulty=next_question.get("difficulty", 2)
            )
            self.session.dialogue_history.append(turn)
            self.session.total_questions_asked += 1
            
            if next_question.get("id"):
                self.session.asked_question_ids.append(next_question["id"])
            
            logger.info(f"进入下一题: 阶段={new_phase.value}, 进度={self.session.get_progress()}")
            
            return {
                "type": "next_question",
                "question": question_text,
                "question_type": next_question.get("type", "technical"),
                "difficulty": next_question.get("difficulty", 2),
                "evaluation": evaluation,
                "phase": self.session.current_phase.value,
                "progress": self.session.get_detailed_progress(),
                "feedback": evaluation.get("feedback", "")
            }
            
        except Exception as e:
            logger.error(f"进入下一题时发生错误: {e}", exc_info=True)
            return {"error": "进入下一题失败，请稍后重试", "error_code": "ADV_001"}
    
    def _update_phase(self):
        """更新面试阶段 - 改进版：同步面试计划，避免跳过空阶段"""
        try:
            current = self.session.current_phase
            plan = self.session.interview_plan

            def _get_next_phase(after_phase: InterviewPhase) -> InterviewPhase:
                """根据面试计划获取下一个非空阶段"""
                phase_order = [
                    InterviewPhase.SELF_INTRO,
                    InterviewPhase.TECHNICAL,
                    InterviewPhase.PROJECT,
                    InterviewPhase.BEHAVIORAL,
                    InterviewPhase.CANDIDATE_QA,
                    InterviewPhase.CLOSING
                ]
                try:
                    idx = phase_order.index(after_phase)
                except ValueError:
                    return InterviewPhase.CLOSING

                for phase in phase_order[idx + 1:]:
                    phase_key = phase.value
                    # CANDIDATE_QA 是候选人提问环节，不需要在计划中配置问题
                    if phase == InterviewPhase.CANDIDATE_QA:
                        return phase
                    # 检查该阶段在面试计划中是否有问题，或是否配置了题目数量
                    has_questions = bool(plan.get(phase_key, []))
                    has_configured_count = getattr(self.session, f"num_{phase_key}", 0) > 0
                    if has_questions or has_configured_count:
                        return phase
                return InterviewPhase.CLOSING

            if current == InterviewPhase.SELF_INTRO:
                # 自我介绍完成后，按顺序找到第一个非空阶段
                next_phase = _get_next_phase(InterviewPhase.SELF_INTRO)
                self.session.current_phase = next_phase

            elif current == InterviewPhase.TECHNICAL:
                technical_asked = len([t for t in self.session.dialogue_history
                                       if t.question_type == "technical" and not t.is_follow_up])
                technical_planned = len(plan.get("technical", []))
                # 只有当技术题问完（以实际计划和已问数量中较大者为准）才切换
                if technical_asked >= max(self.session.num_technical, technical_planned):
                    self.session.current_phase = _get_next_phase(InterviewPhase.TECHNICAL)

            elif current == InterviewPhase.PROJECT:
                project_asked = len([t for t in self.session.dialogue_history
                                     if t.question_type == "project" and not t.is_follow_up])
                project_planned = len(plan.get("project", []))
                if project_asked >= max(self.session.num_project, project_planned):
                    self.session.current_phase = _get_next_phase(InterviewPhase.PROJECT)

            elif current == InterviewPhase.BEHAVIORAL:
                behavioral_asked = len([t for t in self.session.dialogue_history
                                        if t.question_type == "behavioral" and not t.is_follow_up])
                behavioral_planned = len(plan.get("behavioral", []))
                if behavioral_asked >= max(self.session.num_behavioral, behavioral_planned):
                    self.session.current_phase = _get_next_phase(InterviewPhase.BEHAVIORAL)

            elif current == InterviewPhase.CANDIDATE_QA:
                self.session.current_phase = InterviewPhase.CLOSING

        except Exception as e:
            logger.error(f"更新面试阶段时发生错误: {e}")
    
    def _get_transition(self, old_phase: InterviewPhase, new_phase: InterviewPhase) -> Optional[str]:
        """获取阶段过渡语 - 改进版：避免重复使用相同的过渡语"""
        try:
            if old_phase == InterviewPhase.SELF_INTRO and new_phase == InterviewPhase.TECHNICAL:
                candidates = self.transitions["to_technical"]
            elif old_phase == InterviewPhase.TECHNICAL and new_phase == InterviewPhase.PROJECT:
                candidates = self.transitions["to_project"]
            elif old_phase == InterviewPhase.PROJECT and new_phase == InterviewPhase.BEHAVIORAL:
                candidates = self.transitions["to_behavioral"]
            elif new_phase == InterviewPhase.CANDIDATE_QA:
                candidates = self.transitions["to_closing"]
            else:
                return None

            # 过滤掉最近使用过的过渡语（保留最近3次的历史）
            recent_used = set(self.session.used_transitions[-3:])
            available = [t for t in candidates if t not in recent_used]

            # 如果所有过渡语都最近用过，则使用全部候选
            if not available:
                available = candidates

            transition = random.choice(available)

            # 记录使用的过渡语
            self.session.used_transitions.append(transition)
            # 限制历史记录长度，避免无限增长
            if len(self.session.used_transitions) > 20:
                self.session.used_transitions = self.session.used_transitions[-10:]

            return transition
        except Exception as e:
            logger.error(f"获取过渡语时发生错误: {e}")
            return None
    
    def _get_next_question(self) -> Dict[str, Any]:
        """获取下一个问题"""
        try:
            phase = self.session.current_phase
            plan = self.session.interview_plan
            asked = self.session.asked_question_ids
            
            if phase == InterviewPhase.TECHNICAL:
                questions = plan.get("technical", [])
            elif phase == InterviewPhase.PROJECT:
                questions = plan.get("project", [])
            elif phase == InterviewPhase.BEHAVIORAL:
                questions = plan.get("behavioral", [])
            elif phase == InterviewPhase.CANDIDATE_QA:
                return {"question": "面试快结束了，你有什么问题想问我的吗？", "type": "candidate_qa", "difficulty": 1}
            else:
                questions = []
            
            # 找到未问过的问题
            for q in questions:
                if q.get("id") not in asked:
                    return q
            
            # 如果都问过了，返回通用问题
            return {
                "question": self._generate_generic_question(phase),
                "type": phase.value,
                "difficulty": 2
            }
        except Exception as e:
            logger.error(f"获取下一问题时发生错误: {e}")
            return {"question": "请继续。", "type": "general", "difficulty": 2}
    
    def _generate_generic_question(self, phase: InterviewPhase) -> str:
        """生成通用问题 - 改进版：基于已回答内容生成个性化延伸问题"""
        try:
            # 首先尝试基于历史回答生成个性化问题
            personalized = self._generate_personalized_question(phase)
            if personalized:
                return personalized

            # 回退到通用问题模板
            generic_questions = {
                InterviewPhase.TECHNICAL: [
                    "能谈谈你在技术方面的其他经验吗？",
                    "你在技术学习中遇到过什么挑战？",
                    "你平时如何保持技术能力的提升？"
                ],
                InterviewPhase.PROJECT: [
                    "能分享另一个你参与的项目吗？",
                    "在项目中你是如何与团队合作的？",
                    "你遇到过最难的技术问题是什么？"
                ],
                InterviewPhase.BEHAVIORAL: [
                    "你如何处理工作中的压力？",
                    "描述一次你解决冲突的经历。",
                    "你认为自己最大的优点是什么？"
                ]
            }

            questions = generic_questions.get(phase, ["请继续。"])
            return random.choice(questions)
        except Exception as e:
            logger.error(f"生成通用问题时发生错误: {e}")
            return "请继续。"

    def _generate_personalized_question(self, phase: InterviewPhase) -> Optional[str]:
        """基于历史回答生成个性化延伸问题"""
        try:
            # 获取当前阶段的历史回答
            phase_history = [
                t for t in self.session.dialogue_history
                if t.question_type == phase.value and not t.is_follow_up and t.candidate_answer
            ]

            if not phase_history:
                return None

            # 获取最近一个回答的内容
            last_turn = phase_history[-1]
            last_answer = last_turn.candidate_answer
            last_question = last_turn.interviewer_question

            # 基于回答内容提取关键词生成延伸问题
            # 这里使用简单的启发式规则，实际可以接入LLM生成更个性化的问题
            answer_lower = last_answer.lower()

            # 如果回答提到了具体技术，追问相关细节
            tech_keywords = ["java", "python", "spring", "redis", "mysql", "docker", "k8s", "kubernetes"]
            mentioned_tech = [tech for tech in tech_keywords if tech in answer_lower]

            if phase == InterviewPhase.TECHNICAL and mentioned_tech:
                tech = mentioned_tech[0]
                return f"你提到了{tech}，能详细说说你在实际项目中是如何使用它的吗？"

            # 如果回答提到了团队或合作，追问团队协作细节
            if phase == InterviewPhase.PROJECT and any(kw in answer_lower for kw in ["团队", "合作", "协作", "同事"]):
                return "你提到了团队合作，能具体描述一下你在团队中承担的角色和贡献吗？"

            # 如果回答提到了挑战或困难，追问解决过程
            if any(kw in answer_lower for kw in ["挑战", "困难", "问题", "bug", "故障"]):
                return "你提到了遇到的挑战，能详细说说你是如何分析和解决这个问题的吗？"

            # 如果回答比较简短，鼓励候选人展开
            if len(last_answer) < 50:
                return f"关于'{last_question[:20]}...'，你能再详细展开说说吗？"

            return None
        except Exception as e:
            logger.error(f"生成个性化问题时发生错误: {e}")
            return None
    
    def _get_default_weights(self) -> Dict[str, float]:
        """获取默认评分权重"""
        return {
            "technical": 0.3,
            "communication": 0.2,
            "completeness": 0.15,
            "problem_solving": 0.15,
            "teamwork": 0.1,
            "leadership": 0.1
        }

    def _should_end_interview(self) -> bool:
        """判断是否应该结束面试 - 改进版：添加超时检查和完成度检查"""
        try:
            # 阶段检查
            if self.session.current_phase == InterviewPhase.CLOSING:
                logger.info("面试阶段为CLOSING，准备结束")
                return True

            # 超时检查
            if self._check_interview_timeout():
                logger.info("面试超时，强制结束")
                return True

            # 完成度检查
            if self._check_interview_completeness():
                logger.info("面试已完成所有必要环节，准备结束")
                return True

            return False
        except Exception as e:
            logger.error(f"判断面试结束时发生错误: {e}")
            return False

    def _check_interview_completeness(self) -> bool:
        """检查面试是否完成所有必要环节"""
        try:
            plan = self.session.interview_plan
            asked = self.session.asked_question_ids

            # 检查每个阶段是否至少问了一个问题（如果该阶段有计划问题）
            for phase_type in ["technical", "project", "behavioral"]:
                questions = plan.get(phase_type, [])
                if questions:
                    # 至少有一个该阶段的问题被问过
                    phase_asked = [q for q in questions if q.get("id") in asked]
                    if not phase_asked:
                        logger.debug(f"面试未完成: {phase_type} 阶段尚未开始")
                        return False

            # 检查是否已进入候选人提问或结束阶段
            if self.session.current_phase in (InterviewPhase.CANDIDATE_QA, InterviewPhase.CLOSING):
                logger.debug("面试已完成所有必要环节")
                return True

            return False
        except Exception as e:
            logger.error(f"检查面试完成度时发生错误: {e}")
            return False
    
    def _end_interview(self) -> Dict[str, Any]:
        """结束面试"""
        try:
            self.session.status = InterviewStatus.COMPLETED
            self.session.ended_at = datetime.now().isoformat()
            self.session.current_phase = InterviewPhase.CLOSING
            
            # 生成最终报告
            report = self.generate_report()
            
            # 生成个性化结束语
            name = self.session.candidate_name
            avg_score = report.get('overall_score', 60)
            
            if avg_score >= 80:
                performance_comment = "你的表现非常出色，技术能力和沟通表达都给我们留下了深刻印象。"
            elif avg_score >= 70:
                performance_comment = "你的表现很不错，展现了良好的技术基础和沟通能力。"
            elif avg_score >= 60:
                performance_comment = "你的表现还可以，在某些方面还有提升空间。"
            else:
                performance_comment = "感谢你的参与，希望这次面试对你有所帮助。"
            
            closing = f"""感谢{name}参加今天的面试！

{performance_comment}

我们已经完成了所有问题的讨论。面试结果将在3个工作日内通过邮件通知你。

再次感谢你的时间和配合，祝你有个愉快的一天！"""
            
            logger.info(f"面试结束: {self.session.session_id}, 最终得分: {avg_score:.1f}")
            
            return {
                "type": "closing",
                "message": closing,
                "report": report,
                "phase": "closing",
                "progress": "完成"
            }
            
        except Exception as e:
            logger.error(f"结束面试时发生错误: {e}", exc_info=True)
            return {"error": "结束面试失败，请稍后重试", "error_code": "END_001"}
    
    @log_function_call()
    def generate_report(self) -> Dict[str, Any]:
        """生成面试报告 - 增强版"""
        try:
            if not self.session:
                logger.error("无法生成报告: 面试会话不存在")
                return {}
            
            logger.info("=" * 60)
            logger.info(f"生成面试报告: {self.session.session_id}")
            logger.info("=" * 60)
            
            # 计算各维度平均分
            dimension_scores = {
                "technical": [],
                "communication": [],
                "completeness": [],
                "problem_solving": [],
                "teamwork": [],
                "leadership": []
            }
            
            # 记录每个回合的详细评分
            turn_details = []
            
            for i, turn in enumerate(self.session.dialogue_history):
                if turn.scores:
                    turn_info = {
                        "turn_id": turn.turn_id,
                        "question_type": turn.question_type,
                        "is_follow_up": turn.is_follow_up,
                        "scores": turn.scores.copy(),
                        "evaluation": turn.evaluation
                    }
                    turn_details.append(turn_info)
                    
                    for dim, score in turn.scores.items():
                        if dim in dimension_scores:
                            dimension_scores[dim].append(score)
            
            # 计算平均分
            avg_scores = {}
            for dim, scores in dimension_scores.items():
                avg_scores[dim] = round(sum(scores) / len(scores), 1) if scores else 0
            
            # 计算加权总分 - 使用动态权重（如果可用）
            if SCORING_CONFIG_AVAILABLE and 'DynamicWeightAdjuster' in globals():
                try:
                    from src.scoring.dynamic_weight import DynamicWeightAdjuster
                    adjuster = DynamicWeightAdjuster()
                    # 检查是否有声纹数据
                    has_voice = any(t.audio_path for t in self.session.dialogue_history)
                    weights = adjuster.calculate_weights(
                        position=self.session.position,
                        experience="mid",  # 默认中级经验
                        has_voice_data=has_voice
                    )
                    # 移除 voice_profile（报告只展示6个核心维度）
                    weights.pop("voice_profile", None)
                    # 重新归一化
                    total = sum(weights.values())
                    weights = {k: round(v / total, 4) for k, v in weights.items()}
                    logger.info(f"[OK] 使用动态权重: {weights}")
                except Exception as e:
                    logger.warning(f"动态权重计算失败，使用默认权重: {e}")
                    weights = self._get_default_weights()
            else:
                weights = self._get_default_weights()

            weighted_score = 0
            for dim, weight in weights.items():
                weighted_score += avg_scores.get(dim, 0) * weight

            overall_score = round(weighted_score, 1)
            
            # 计算各维度等级
            def get_score_level(score):
                if score >= 85:
                    return "优秀", "#4caf50"
                elif score >= 70:
                    return "良好", "#8bc34a"
                elif score >= 60:
                    return "及格", "#ff9800"
                else:
                    return "待提升", "#f44336"
            
            dimension_levels = {}
            for dim, score in avg_scores.items():
                level, color = get_score_level(score)
                dimension_levels[dim] = {"level": level, "color": color, "score": score}
            
            # 生成详细建议
            strengths = []
            improvements = []
            
            # 技术能力建议
            tech_score = avg_scores.get("technical", 0)
            if tech_score >= 80:
                strengths.append(f"[OK] 技术能力优秀 ({tech_score}分)：专业知识扎实，技术深度和广度都很好")
            elif tech_score >= 65:
                strengths.append(f"[OK] 技术能力良好 ({tech_score}分)：具备基本的技术知识")
                improvements.append(f"[TIP] 技术提升建议 ({tech_score}分)：建议深入学习核心技术原理和最佳实践")
            else:
                improvements.append(f"[WARN] 技术能力待提升 ({tech_score}分)：建议系统学习相关技术栈，加强基础知识")
            
            # 表达能力建议
            comm_score = avg_scores.get("communication", 0)
            if comm_score >= 80:
                strengths.append(f"[OK] 表达能力出色 ({comm_score}分)：逻辑清晰，表达流畅")
            elif comm_score >= 65:
                strengths.append(f"[OK] 表达能力良好 ({comm_score}分)：能够清晰表达观点")
                improvements.append(f"[TIP] 表达提升建议 ({comm_score}分)：建议加强结构化表达能力")
            else:
                improvements.append(f"[WARN] 表达能力待提升 ({comm_score}分)：建议多练习技术演讲和文档写作")
            
            # 完整度建议
            comp_score = avg_scores.get("completeness", 0)
            if comp_score >= 80:
                strengths.append(f"[OK] 回答完整度高 ({comp_score}分)：内容全面，覆盖要点")
            elif comp_score >= 65:
                improvements.append(f"[TIP] 回答深度建议 ({comp_score}分)：建议回答更加详细和全面")
            else:
                improvements.append(f"[WARN] 回答完整度待提升 ({comp_score}分)：建议扩展回答内容，补充细节")
            
            # 问题解决建议
            prob_score = avg_scores.get("problem_solving", 0)
            if prob_score >= 80:
                strengths.append(f"[OK] 问题解决能力强 ({prob_score}分)：分析思路清晰，解决方案合理")
            elif prob_score >= 65:
                improvements.append(f"[TIP] 问题解决提升 ({prob_score}分)：建议多练习复杂问题的分析和解决")
            else:
                improvements.append(f"[WARN] 问题解决能力待提升 ({prob_score}分)：建议学习系统分析和调试技巧")
            
            # 团队协作建议
            team_score = avg_scores.get("teamwork", 0)
            if team_score >= 80:
                strengths.append(f"[OK] 团队协作优秀 ({team_score}分)：合作意识强，善于沟通")
            elif team_score >= 65:
                strengths.append(f"[OK] 团队协作良好 ({team_score}分)：具备基本的团队协作能力")
            else:
                improvements.append(f"[TIP] 团队协作建议 ({team_score}分)：建议加强沟通技巧和团队意识")
            
            # 领导力建议
            lead_score = avg_scores.get("leadership", 0)
            if lead_score >= 80:
                strengths.append(f"[OK] 领导力突出 ({lead_score}分)：能够带领团队，把控项目")
            elif lead_score >= 65:
                strengths.append(f"[OK] 领导力良好 ({lead_score}分)：具备一定的领导潜质")
            
            # 推荐等级
            if overall_score >= 85:
                recommendation = "强烈推荐 [A级]"
                recommendation_level = "A"
            elif overall_score >= 70:
                recommendation = "推荐 [B级]"
                recommendation_level = "B"
            elif overall_score >= 60:
                recommendation = "考虑 [C级]"
                recommendation_level = "C"
            else:
                recommendation = "不推荐 [D级]"
                recommendation_level = "D"
            
            # 计算面试时长
            try:
                if self.session.started_at:
                    start_time = datetime.fromisoformat(self.session.started_at)
                    end_time = datetime.now()
                    duration = (end_time - start_time).total_seconds()
                    duration_str = f"{int(duration // 60)}分{int(duration % 60)}秒"
                else:
                    duration_str = "未知"
            except:
                duration_str = "未知"
            
            # 计算评分置信度
            try:
                from src.scoring.confidence_calculator import ScoreConfidenceCalculator
                confidence_calc = ScoreConfidenceCalculator()

                # 收集所有回答的总长度
                total_answer_length = sum(
                    len(t.candidate_answer) for t in self.session.dialogue_history if t.candidate_answer
                )
                # 检查是否有音视频数据
                has_audio = any(t.audio_path for t in self.session.dialogue_history)
                has_video = any(t.video_frame is not None for t in self.session.dialogue_history)
                # 计算评分方差
                score_values = list(avg_scores.values())
                if len(score_values) >= 2:
                    mean_score = sum(score_values) / len(score_values)
                    score_variance = sum((s - mean_score) ** 2 for s in score_values) / len(score_values)
                else:
                    score_variance = None

                confidence_result = confidence_calc.calculate(
                    scores=avg_scores,
                    answer_length=total_answer_length,
                    has_audio=has_audio,
                    has_video=has_video,
                    num_evaluations=len(turn_details),
                    score_variance=score_variance
                )
                logger.info(f"[OK] 评分置信度: {confidence_result['overall_confidence']:.2f} ({confidence_result['level']})")
            except Exception as e:
                logger.warning(f"置信度计算失败: {e}")
                confidence_result = None

            # 构建完整报告
            report = {
                "session_id": self.session.session_id,
                "candidate_name": self.session.candidate_name,
                "position": self.session.position,
                "interview_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "duration": duration_str,
                "overall_score": overall_score,
                "recommendation": recommendation,
                "recommendation_level": recommendation_level,
                "dimension_scores": avg_scores,
                "dimension_levels": dimension_levels,
                "weights": weights,
                "strengths": strengths,
                "improvements": improvements,
                "total_questions": self.session.total_questions_asked,
                "total_follow_ups": self.session.total_follow_ups,
                "turn_details": turn_details,
                "score_distribution": {
                    "excellent": len([s for s in avg_scores.values() if s >= 85]),
                    "good": len([s for s in avg_scores.values() if 70 <= s < 85]),
                    "pass": len([s for s in avg_scores.values() if 60 <= s < 70]),
                    "fail": len([s for s in avg_scores.values() if s < 60])
                },
                "confidence": confidence_result  # 添加置信度信息
            }
            
            logger.info(f"[OK] 面试报告生成成功")
            logger.info(f"  - 综合得分: {overall_score:.1f}")
            logger.info(f"  - 推荐等级: {recommendation}")
            logger.info(f"  - 各维度分数: {avg_scores}")
            logger.info("=" * 60)
            
            return report
            
        except Exception as e:
            logger.error(f"生成面试报告时发生错误: {e}", exc_info=True)
            return {}
    
    @log_function_call()
    def end_interview(self) -> Dict[str, Any]:
        """结束面试"""
        try:
            if not self.session:
                logger.error("面试会话不存在")
                return {"error": "面试会话不存在", "error_code": "SES_004"}

            if self.session.status == InterviewStatus.COMPLETED:
                logger.warning("面试已结束")
                return {"type": "closing", "message": "面试已结束"}
            
            logger.info(f"结束面试: {self.session.session_id}")
            
            self.session.status = InterviewStatus.COMPLETED
            self.session.ended_at = datetime.now().isoformat()
            
            # 生成结束语
            name = self.session.candidate_name
            closing_messages = [
                f"{name}，今天的面试就到这里。感谢你的参与，我们会尽快给你反馈。",
                f"谢谢{name}抽出时间参加面试。我们会综合评估后通知你结果。",
                f"面试结束了，{name}。很高兴认识你，祝你好运！"
            ]
            
            closing_message = random.choice(closing_messages)
            
            # 添加结束回合
            turn = DialogueTurn(
                interviewer_question=closing_message,
                question_type="closing",
                difficulty=1
            )
            self.session.dialogue_history.append(turn)
            
            logger.info(f"[OK] 面试结束: {self.session.session_id}")
            
            return {
                "type": "closing",
                "message": closing_message,
                "session_id": self.session.session_id,
                "status": "completed"
            }
            
        except Exception as e:
            logger.error(f"结束面试时发生错误: {e}", exc_info=True)
            return {"error": "结束面试失败，请稍后重试", "error_code": "END_002"}


if __name__ == "__main__":
    # 测试代码
    logger.info("=" * 60)
    logger.info("运行面试引擎测试")
    logger.info("=" * 60)
    
    try:
        engine = InterviewEngine()
        session = engine.create_session("Java后端开发", "测试用户")
        result = engine.start_interview()
        logger.info(f"面试开始结果: {result}")
        
        # 模拟回答
        test_answer = "我是测试用户，有3年Java开发经验。"
        result = engine.process_answer(test_answer)
        logger.info(f"回答处理结果: {result}")
        
        # 生成报告
        report = engine.generate_report()
        logger.info(f"面试报告: {report}")
        
        logger.info("[OK] 测试完成")
        
    except Exception as e:
        logger.error(f"测试过程中发生错误: {e}", exc_info=True)
