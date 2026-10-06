"""
数据库模块测试
"""

import pytest
import json
import tempfile
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, MagicMock
from src.database import DatabaseManager


class TestDatabaseManager:
    """测试面试数据库"""

    def test_init_default(self):
        """测试默认初始化"""
        db = DatabaseManager()
        assert db.engine is not None
        assert db.SessionLocal is not None

    def test_save_interview(self):
        """测试保存面试记录"""
        db = DatabaseManager()

        dialogue_history = [
            {
                "turn_id": "t001",
                "timestamp": datetime.now().isoformat(),
                "interviewer_question": "问题1",
                "candidate_answer": "回答1",
                "question_type": "technical",
                "difficulty": 2,
                "scores": {"technical": 80},
                "evaluation": {"score": 80},
                "is_follow_up": False,
                "follow_up_count": 0,
            }
        ]

        session_data = {
            "session_id": "test-session-001",
            "candidate_name": "测试候选人",
            "position": "Java后端开发",
            "status": "completed",
            "created_at": datetime.now().isoformat(),
            "started_at": datetime.now().isoformat(),
            "ended_at": datetime.now().isoformat(),
            "num_technical": 3,
            "num_project": 1,
            "num_behavioral": 1,
            "enable_follow_up": True,
            "max_follow_up_depth": 2,
            "total_questions_asked": 5,
            "total_follow_ups": 2,
            "dialogue_history": dialogue_history,
            "overall_score": 80,
            "recommendation": "推荐",
            "report_data": {"overall_score": 80, "recommendation": "推荐"}
        }

        result = db.save_interview(session_data)
        assert result is not None
        assert isinstance(result, str)

    def test_get_interview(self):
        """测试获取面试记录"""
        db = DatabaseManager()

        dialogue_history = [
            {
                "turn_id": "t001",
                "timestamp": datetime.now().isoformat(),
                "interviewer_question": "问题1",
                "candidate_answer": "回答1",
                "question_type": "technical",
                "difficulty": 2,
                "scores": {"technical": 80},
                "evaluation": {"score": 80},
                "is_follow_up": False,
                "follow_up_count": 0,
            }
        ]

        session_data = {
            "session_id": "test-session-002",
            "candidate_name": "测试候选人2",
            "position": "Python后端开发",
            "status": "completed",
            "created_at": datetime.now().isoformat(),
            "started_at": datetime.now().isoformat(),
            "ended_at": datetime.now().isoformat(),
            "num_technical": 3,
            "num_project": 1,
            "num_behavioral": 1,
            "enable_follow_up": True,
            "max_follow_up_depth": 2,
            "total_questions_asked": 5,
            "total_follow_ups": 2,
            "dialogue_history": dialogue_history,
            "overall_score": 85,
            "recommendation": "强烈推荐",
            "report_data": {"overall_score": 85, "recommendation": "强烈推荐"}
        }

        db.save_interview(session_data)

        record = db.get_interview("test-session-002")
        assert record is not None
        assert record["candidate_name"] == "测试候选人2"
        assert record["position"] == "Python后端开发"

    def test_get_interview_not_found(self):
        """测试获取不存在的记录"""
        db = DatabaseManager()
        record = db.get_interview("nonexistent")
        assert record is None

    def test_list_interviews_by_position(self):
        """测试按岗位查询"""
        db = DatabaseManager()

        dialogue_history = [
            {
                "turn_id": "t001",
                "timestamp": datetime.now().isoformat(),
                "interviewer_question": "问题1",
                "candidate_answer": "回答1",
                "question_type": "technical",
                "difficulty": 2,
                "scores": {"technical": 80},
                "evaluation": {"score": 80},
                "is_follow_up": False,
                "follow_up_count": 0,
            }
        ]

        session_data_a = {
            "session_id": "test-session-003",
            "candidate_name": "候选人A",
            "position": "Java后端开发",
            "status": "completed",
            "created_at": datetime.now().isoformat(),
            "started_at": datetime.now().isoformat(),
            "ended_at": datetime.now().isoformat(),
            "num_technical": 3,
            "num_project": 1,
            "num_behavioral": 1,
            "enable_follow_up": True,
            "max_follow_up_depth": 2,
            "total_questions_asked": 5,
            "total_follow_ups": 2,
            "dialogue_history": dialogue_history,
            "overall_score": 80,
            "recommendation": "推荐",
            "report_data": {}
        }

        session_data_b = {
            "session_id": "test-session-004",
            "candidate_name": "候选人B",
            "position": "Java后端开发",
            "status": "completed",
            "created_at": datetime.now().isoformat(),
            "started_at": datetime.now().isoformat(),
            "ended_at": datetime.now().isoformat(),
            "num_technical": 3,
            "num_project": 1,
            "num_behavioral": 1,
            "enable_follow_up": True,
            "max_follow_up_depth": 2,
            "total_questions_asked": 5,
            "total_follow_ups": 2,
            "dialogue_history": dialogue_history,
            "overall_score": 90,
            "recommendation": "强烈推荐",
            "report_data": {}
        }

        db.save_interview(session_data_a)
        db.save_interview(session_data_b)

        results = db.list_interviews(position="Java后端开发")
        assert len(results) >= 2
        assert all(r["position"] == "Java后端开发" for r in results)

    def test_list_interviews_by_date_range(self):
        """测试按日期范围查询"""
        db = DatabaseManager()

        dialogue_history = [
            {
                "turn_id": "t001",
                "timestamp": datetime.now().isoformat(),
                "interviewer_question": "问题1",
                "candidate_answer": "回答1",
                "question_type": "technical",
                "difficulty": 2,
                "scores": {"technical": 80},
                "evaluation": {"score": 80},
                "is_follow_up": False,
                "follow_up_count": 0,
            }
        ]

        session_data = {
            "session_id": "test-session-005",
            "candidate_name": "候选人C",
            "position": "前端开发",
            "status": "completed",
            "created_at": datetime.now().isoformat(),
            "started_at": datetime.now().isoformat(),
            "ended_at": datetime.now().isoformat(),
            "num_technical": 3,
            "num_project": 1,
            "num_behavioral": 1,
            "enable_follow_up": True,
            "max_follow_up_depth": 2,
            "total_questions_asked": 5,
            "total_follow_ups": 2,
            "dialogue_history": dialogue_history,
            "overall_score": 75,
            "recommendation": "推荐",
            "report_data": {}
        }

        db.save_interview(session_data)

        results = db.list_interviews()
        assert len(results) >= 1

    def test_get_statistics(self):
        """测试获取统计信息"""
        db = DatabaseManager()
        stats = db.get_statistics()

        assert isinstance(stats, dict)

    def test_delete_interview(self):
        """测试删除记录"""
        db = DatabaseManager()

        dialogue_history = [
            {
                "turn_id": "t001",
                "timestamp": datetime.now().isoformat(),
                "interviewer_question": "问题1",
                "candidate_answer": "回答1",
                "question_type": "technical",
                "difficulty": 2,
                "scores": {"technical": 80},
                "evaluation": {"score": 80},
                "is_follow_up": False,
                "follow_up_count": 0,
            }
        ]

        session_data = {
            "session_id": "test-session-007",
            "candidate_name": "候选人E",
            "position": "运维工程师",
            "status": "completed",
            "created_at": datetime.now().isoformat(),
            "started_at": datetime.now().isoformat(),
            "ended_at": datetime.now().isoformat(),
            "num_technical": 3,
            "num_project": 1,
            "num_behavioral": 1,
            "enable_follow_up": True,
            "max_follow_up_depth": 2,
            "total_questions_asked": 5,
            "total_follow_ups": 2,
            "dialogue_history": dialogue_history,
            "overall_score": 78,
            "recommendation": "推荐",
            "report_data": {}
        }

        db.save_interview(session_data)

        result = db.delete_interview("test-session-007")
        assert result is True

        record = db.get_interview("test-session-007")
        assert record is None

    def test_delete_interview_not_found(self):
        """测试删除不存在的记录"""
        db = DatabaseManager()
        result = db.delete_interview("nonexistent")
        assert result is False
