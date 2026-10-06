"""
锐聘AI - 数据库持久化层
使用SQLAlchemy实现面试数据的持久化存储
"""

import json
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path

from sqlalchemy import (
    create_engine, Column, String, Float, DateTime,
    Text, Integer, Boolean, JSON, ForeignKey, event, func
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session

from .logger import logger

# 数据库路径
DB_PATH = Path(__file__).parent.parent / "data" / "interviews.db"
DB_PATH.parent.mkdir(exist_ok=True)

# 创建引擎 - 配置连接池优化性能
engine = create_engine(
    f"sqlite:///{DB_PATH}",
    echo=False,
    connect_args={"check_same_thread": False},
    pool_pre_ping=True,  # 连接前ping检查，避免使用失效连接
    pool_recycle=3600,   # 1小时后回收连接
    max_overflow=10,     # 超出pool_size时允许的最大连接数
    pool_size=5          # 连接池大小
)

# 会话工厂
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 基类
Base = declarative_base()


class InterviewRecord(Base):
    """面试记录表"""
    __tablename__ = "interviews"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String(36), unique=True, index=True)
    candidate_name = Column(String(100), nullable=False)
    position = Column(String(100), nullable=False)
    status = Column(String(20), default="created")
    
    # 时间戳
    created_at = Column(DateTime, default=datetime.now)
    started_at = Column(DateTime, nullable=True)
    ended_at = Column(DateTime, nullable=True)
    
    # 配置
    num_technical = Column(Integer, default=5)
    num_project = Column(Integer, default=2)
    num_behavioral = Column(Integer, default=2)
    enable_follow_up = Column(Boolean, default=True)
    max_follow_up_depth = Column(Integer, default=2)
    
    # 统计
    total_questions_asked = Column(Integer, default=0)
    total_follow_ups = Column(Integer, default=0)
    
    # 评分
    overall_score = Column(Float, nullable=True)
    recommendation = Column(String(50), nullable=True)
    recommendation_level = Column(String(10), nullable=True)
    
    # 维度分数
    technical_score = Column(Float, nullable=True)
    communication_score = Column(Float, nullable=True)
    completeness_score = Column(Float, nullable=True)
    problem_solving_score = Column(Float, nullable=True)
    teamwork_score = Column(Float, nullable=True)
    leadership_score = Column(Float, nullable=True)
    
    # 详细数据（JSON存储）
    dimension_scores = Column(JSON, default=dict)
    dimension_levels = Column(JSON, default=dict)
    strengths = Column(JSON, default=list)
    improvements = Column(JSON, default=list)
    interview_plan = Column(JSON, default=dict)
    
    # 关系
    dialogue_turns = relationship("DialogueTurnRecord", back_populates="interview", cascade="all, delete-orphan")
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "candidate_name": self.candidate_name,
            "position": self.position,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "overall_score": self.overall_score,
            "recommendation": self.recommendation,
            "recommendation_level": self.recommendation_level,
            "dimension_scores": self.dimension_scores or {},
            "dimension_levels": self.dimension_levels or {},
            "strengths": self.strengths or [],
            "improvements": self.improvements or [],
            "total_questions": self.total_questions_asked,
            "total_follow_ups": self.total_follow_ups,
        }


class DialogueTurnRecord(Base):
    """对话回合记录表"""
    __tablename__ = "dialogue_turns"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    interview_id = Column(String(36), ForeignKey("interviews.id"), index=True)
    turn_id = Column(String(36), index=True)
    
    # 问题信息
    interviewer_question = Column(Text, default="")
    question_id = Column(String(50), nullable=True)
    question_type = Column(String(50), default="")
    difficulty = Column(Integer, default=1)
    
    # 回答信息
    candidate_answer = Column(Text, default="")
    has_audio = Column(Boolean, default=False)
    has_video = Column(Boolean, default=False)
    
    # 评估
    evaluation = Column(JSON, default=dict)
    scores = Column(JSON, default=dict)
    
    # 追问
    is_follow_up = Column(Boolean, default=False)
    follow_up_count = Column(Integer, default=0)
    parent_turn_id = Column(String(36), nullable=True)
    
    # 时间戳
    timestamp = Column(DateTime, default=datetime.now)
    
    # 关系
    interview = relationship("InterviewRecord", back_populates="dialogue_turns")
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "turn_id": self.turn_id,
            "interviewer_question": self.interviewer_question,
            "question_id": self.question_id,
            "question_type": self.question_type,
            "candidate_answer": self.candidate_answer,
            "evaluation": self.evaluation or {},
            "scores": self.scores or {},
            "is_follow_up": self.is_follow_up,
            "follow_up_count": self.follow_up_count,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }


class DatabaseManager:
    """数据库管理器"""
    
    def __init__(self):
        self.engine = engine
        self.SessionLocal = SessionLocal
        self._init_database()
    
    def _init_database(self):
        """初始化数据库"""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info(f"[OK] 数据库初始化完成: {DB_PATH}")
        except Exception as e:
            logger.error(f"[ERROR] 数据库初始化失败: {e}")
            raise
    
    def get_session(self) -> Session:
        """获取数据库会话"""
        return self.SessionLocal()
    
    def save_interview(self, session_data: Dict[str, Any]) -> str:
        """
        保存面试会话到数据库
        
        Args:
            session_data: 面试会话数据字典
            
        Returns:
            记录ID
        """
        db = self.get_session()
        try:
            # 检查是否已存在
            existing = db.query(InterviewRecord).filter(
                InterviewRecord.session_id == session_data.get("session_id")
            ).first()
            
            if existing:
                # 更新现有记录
                record = self._update_interview_record(existing, session_data)
            else:
                # 创建新记录
                record = self._create_interview_record(session_data)
                db.add(record)
            
            db.commit()
            db.refresh(record)
            
            # 保存对话回合
            self._save_dialogue_turns(db, record.id, session_data.get("dialogue_history", []))
            
            logger.info(f"[OK] 面试记录已保存: {record.session_id}")
            return record.id
            
        except Exception as e:
            db.rollback()
            logger.error(f"[ERROR] 保存面试记录失败: {e}")
            raise
        finally:
            db.close()
    
    def _create_interview_record(self, data: Dict[str, Any]) -> InterviewRecord:
        """创建面试记录"""
        return InterviewRecord(
            session_id=data.get("session_id", str(uuid.uuid4())),
            candidate_name=data.get("candidate_name", ""),
            position=data.get("position", ""),
            status=data.get("status", "created"),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now(),
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
            ended_at=datetime.fromisoformat(data["ended_at"]) if data.get("ended_at") else None,
            num_technical=data.get("num_technical", 5),
            num_project=data.get("num_project", 2),
            num_behavioral=data.get("num_behavioral", 2),
            enable_follow_up=data.get("enable_follow_up", True),
            max_follow_up_depth=data.get("max_follow_up_depth", 2),
            total_questions_asked=data.get("total_questions_asked", 0),
            total_follow_ups=data.get("total_follow_ups", 0),
        )
    
    def _update_interview_record(self, record: InterviewRecord, data: Dict[str, Any]) -> InterviewRecord:
        """更新面试记录"""
        record.status = data.get("status", record.status)
        record.ended_at = datetime.fromisoformat(data["ended_at"]) if data.get("ended_at") else record.ended_at
        record.total_questions_asked = data.get("total_questions_asked", record.total_questions_asked)
        record.total_follow_ups = data.get("total_follow_ups", record.total_follow_ups)
        
        # 更新评分（如果有）
        if "overall_score" in data:
            record.overall_score = data["overall_score"]
        if "recommendation" in data:
            record.recommendation = data["recommendation"]
        if "recommendation_level" in data:
            record.recommendation_level = data["recommendation_level"]
        
        # 更新维度分数
        dimension_scores = data.get("dimension_scores", {})
        if dimension_scores:
            record.dimension_scores = dimension_scores
            record.technical_score = dimension_scores.get("technical")
            record.communication_score = dimension_scores.get("communication")
            record.completeness_score = dimension_scores.get("completeness")
            record.problem_solving_score = dimension_scores.get("problem_solving")
            record.teamwork_score = dimension_scores.get("teamwork")
            record.leadership_score = dimension_scores.get("leadership")
        
        if "dimension_levels" in data:
            record.dimension_levels = data["dimension_levels"]
        if "strengths" in data:
            record.strengths = data["strengths"]
        if "improvements" in data:
            record.improvements = data["improvements"]
        
        return record
    
    def _save_dialogue_turns(self, db: Session, interview_id: str, turns: List[Dict[str, Any]]):
        """保存对话回合"""
        try:
            # 删除旧记录
            db.query(DialogueTurnRecord).filter(
                DialogueTurnRecord.interview_id == interview_id
            ).delete()
            
            # 添加新记录
            for turn_data in turns:
                turn = DialogueTurnRecord(
                    interview_id=interview_id,
                    turn_id=turn_data.get("turn_id", str(uuid.uuid4())),
                    interviewer_question=turn_data.get("interviewer_question", ""),
                    question_id=turn_data.get("question_id"),
                    question_type=turn_data.get("question_type", ""),
                    difficulty=turn_data.get("difficulty", 1),
                    candidate_answer=turn_data.get("candidate_answer", ""),
                    has_audio=turn_data.get("has_audio", False),
                    has_video=turn_data.get("has_video", False),
                    evaluation=turn_data.get("evaluation", {}),
                    scores=turn_data.get("scores", {}),
                    is_follow_up=turn_data.get("is_follow_up", False),
                    follow_up_count=turn_data.get("follow_up_count", 0),
                    parent_turn_id=turn_data.get("parent_turn_id"),
                    timestamp=datetime.fromisoformat(turn_data["timestamp"]) if "timestamp" in turn_data else datetime.now(),
                )
                db.add(turn)
            
            db.commit()
            logger.info(f"[OK] 保存了 {len(turns)} 个对话回合")
            
        except Exception as e:
            db.rollback()
            logger.error(f"[ERROR] 保存对话回合失败: {e}")
            raise
    
    def get_interview(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        获取面试记录
        
        Args:
            session_id: 会话ID
            
        Returns:
            面试记录字典，不存在返回None
        """
        db = self.get_session()
        try:
            record = db.query(InterviewRecord).filter(
                InterviewRecord.session_id == session_id
            ).first()
            
            if not record:
                return None
            
            result = record.to_dict()
            result["dialogue_turns"] = [turn.to_dict() for turn in record.dialogue_turns]
            
            return result
            
        except Exception as e:
            logger.error(f"[ERROR] 获取面试记录失败: {e}")
            return None
        finally:
            db.close()
    
    def list_interviews(
        self, 
        position: Optional[str] = None,
        candidate_name: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        列出面试记录
        
        Args:
            position: 岗位筛选
            candidate_name: 候选人姓名筛选
            status: 状态筛选
            limit: 返回数量限制
            offset: 偏移量
            
        Returns:
            面试记录列表
        """
        db = self.get_session()
        try:
            query = db.query(InterviewRecord)
            
            if position:
                query = query.filter(InterviewRecord.position == position)
            if candidate_name:
                query = query.filter(InterviewRecord.candidate_name.contains(candidate_name))
            if status:
                query = query.filter(InterviewRecord.status == status)
            
            records = query.order_by(InterviewRecord.created_at.desc()).offset(offset).limit(limit).all()
            
            return [record.to_dict() for record in records]
            
        except Exception as e:
            logger.error(f"[ERROR] 列出面试记录失败: {e}")
            return []
        finally:
            db.close()
    
    def delete_interview(self, session_id: str) -> bool:
        """
        删除面试记录
        
        Args:
            session_id: 会话ID
            
        Returns:
            是否成功删除
        """
        db = self.get_session()
        try:
            record = db.query(InterviewRecord).filter(
                InterviewRecord.session_id == session_id
            ).first()
            
            if record:
                db.delete(record)
                db.commit()
                logger.info(f"[OK] 删除面试记录: {session_id}")
                return True
            
            return False
            
        except Exception as e:
            db.rollback()
            logger.error(f"[ERROR] 删除面试记录失败: {e}")
            return False
        finally:
            db.close()
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取面试统计信息
        
        Returns:
            统计数据字典
        """
        db = self.get_session()
        try:
            total_interviews = db.query(InterviewRecord).count()
            completed_interviews = db.query(InterviewRecord).filter(
                InterviewRecord.status == "completed"
            ).count()
            
            avg_score = db.query(InterviewRecord).filter(
                InterviewRecord.overall_score.isnot(None)
            ).with_entities(
                func.avg(InterviewRecord.overall_score)
            ).scalar()
            
            return {
                "total_interviews": total_interviews,
                "completed_interviews": completed_interviews,
                "in_progress_interviews": total_interviews - completed_interviews,
                "average_score": round(float(avg_score), 2) if avg_score else 0,
            }
            
        except Exception as e:
            logger.error(f"[ERROR] 获取统计信息失败: {e}")
            return {}
        finally:
            db.close()


# 全局数据库管理器实例
db_manager = DatabaseManager()


def save_interview_to_db(session_data: Dict[str, Any]) -> str:
    """便捷函数：保存面试到数据库"""
    return db_manager.save_interview(session_data)


def get_interview_from_db(session_id: str) -> Optional[Dict[str, Any]]:
    """便捷函数：从数据库获取面试"""
    return db_manager.get_interview(session_id)



