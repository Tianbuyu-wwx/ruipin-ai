"""
锐聘AI - 面试会话管理
"""

import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


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
    video_frame: Any = None
    audio_path: Optional[str] = None
    
    evaluation: Dict[str, Any] = field(default_factory=dict)
    scores: Dict[str, float] = field(default_factory=dict)
    
    is_follow_up: bool = False
    follow_up_count: int = 0
    parent_turn_id: Optional[str] = None


class InterviewSession:
    """面试会话"""
    
    def __init__(
        self,
        position: str,
        candidate_name: str,
        num_technical: int = 5,
        num_project: int = 2,
        num_behavioral: int = 2,
        enable_follow_up: bool = True,
        max_follow_up_depth: int = 2
    ):
        self.session_id = str(uuid.uuid4())
        self.created_at = datetime.now().isoformat()
        self.started_at: Optional[str] = None
        self.ended_at: Optional[str] = None
        
        self.position = position
        self.candidate_name = candidate_name
        
        self.num_technical = num_technical
        self.num_project = num_project
        self.num_behavioral = num_behavioral
        self.enable_follow_up = enable_follow_up
        self.max_follow_up_depth = max_follow_up_depth
        
        self.dialogue_history: List[DialogueTurn] = []
        self.total_questions_asked = 0
        self.total_follow_ups = 0
        
        self.status = "created"
        self.current_phase = None
        self.interview_plan: List[Dict] = []
        self.current_plan_index = 0
    
    def get_current_turn(self) -> Optional[DialogueTurn]:
        """获取当前回合"""
        if self.dialogue_history:
            return self.dialogue_history[-1]
        return None
    
    def get_progress(self) -> str:
        """获取进度"""
        total = len(self.interview_plan)
        if total == 0:
            return "准备中"
        current = min(self.current_plan_index + 1, total)
        return f"{current}/{total}"
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "session_id": self.session_id,
            "candidate_name": self.candidate_name,
            "position": self.position,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "total_questions": self.total_questions_asked,
            "total_follow_ups": self.total_follow_ups,
            "dialogue_history": [
                {
                    "turn_id": turn.turn_id,
                    "question": turn.interviewer_question,
                    "answer": turn.candidate_answer,
                    "scores": turn.scores,
                    "evaluation": turn.evaluation
                }
                for turn in self.dialogue_history
            ]
        }
