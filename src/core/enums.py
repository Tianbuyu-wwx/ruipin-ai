"""
锐聘AI - 枚举定义
"""

from enum import Enum


class InterviewStatus(Enum):
    """面试状态"""
    CREATED = "created"
    GREETING = "greeting"
    IN_PROGRESS = "in_progress"
    FOLLOW_UP = "follow_up"
    EVALUATING = "evaluating"
    COMPLETED = "completed"


class InterviewPhase(Enum):
    """面试阶段"""
    SELF_INTRO = "self_intro"
    TECHNICAL = "technical"
    PROJECT = "project"
    BEHAVIORAL = "behavioral"
    CANDIDATE_QUESTIONS = "candidate_questions"
    ENDING = "ending"
