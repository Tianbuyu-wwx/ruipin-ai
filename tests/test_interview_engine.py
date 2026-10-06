"""
面试引擎测试
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
from src.interview_engine import (
    InterviewStatus,
    InterviewPhase,
    DialogueTurn,
    InterviewSession,
    QuestionBankManager,
    InterviewEngine,
)


class TestDialogueTurn:
    """测试对话回合"""

    def test_default_creation(self):
        """测试默认创建"""
        turn = DialogueTurn()
        assert turn.turn_id is not None
        assert len(turn.turn_id) == 8
        assert turn.timestamp is not None
        assert turn.interviewer_question == ""
        assert turn.candidate_answer == ""
        assert turn.is_follow_up is False
        assert turn.follow_up_count == 0

    def test_custom_values(self):
        """测试自定义值"""
        turn = DialogueTurn(
            interviewer_question="测试问题",
            question_type="technical",
            difficulty=3,
            is_follow_up=True,
            follow_up_count=1
        )
        assert turn.interviewer_question == "测试问题"
        assert turn.question_type == "technical"
        assert turn.difficulty == 3
        assert turn.is_follow_up is True
        assert turn.follow_up_count == 1


class TestInterviewSession:
    """测试面试会话"""

    def test_creation(self):
        """测试会话创建"""
        session = InterviewSession(
            position="Java后端开发",
            candidate_name="测试候选人",
            num_technical=3,
            num_project=1,
            num_behavioral=1
        )
        assert session.session_id is not None
        assert session.position == "Java后端开发"
        assert session.candidate_name == "测试候选人"
        assert session.num_technical == 3
        assert session.num_project == 1
        assert session.num_behavioral == 1
        assert session.status.value == "created"
        assert len(session.dialogue_history) == 0

    def test_get_current_turn_empty(self):
        """测试空历史获取当前回合"""
        session = InterviewSession("Java", "测试")
        assert session.get_current_turn() is None

    def test_get_current_turn(self):
        """测试获取当前回合"""
        session = InterviewSession("Java", "测试")
        turn = DialogueTurn(interviewer_question="问题1")
        session.dialogue_history.append(turn)
        assert session.get_current_turn() == turn

    def test_get_progress_no_plan(self):
        """测试无计划时的进度"""
        session = InterviewSession("Java", "测试")
        progress = session.get_progress()
        assert progress == "准备中" or progress == "0/10"

    def test_to_dict(self):
        """测试转换为字典"""
        session = InterviewSession("Java", "测试")
        data = session.to_dict()
        assert "session_id" in data
        assert "candidate_name" in data
        assert "position" in data
        assert "status" in data


class TestQuestionBankManager:
    """测试题库管理器"""

    def test_load_question_bank(self, temp_dir, sample_question_bank):
        """测试加载题库"""
        bank_path = temp_dir / "questions"
        bank_path.mkdir()

        with open(bank_path / "java_backend.json", "w", encoding="utf-8") as f:
            json.dump(sample_question_bank, f)

        manager = QuestionBankManager(str(bank_path))
        assert "java_backend" in manager.question_banks
        assert len(manager.question_banks["java_backend"]["questions"]) == 4

    def test_get_position_code(self, temp_dir, sample_question_bank):
        """测试获取岗位代码"""
        bank_path = temp_dir / "questions"
        bank_path.mkdir()

        with open(bank_path / "java_backend.json", "w", encoding="utf-8") as f:
            json.dump(sample_question_bank, f)

        manager = QuestionBankManager(str(bank_path))
        assert manager.get_position_code("Java后端开发") == "java_backend"
        assert manager.get_position_code("Java后端开发工程师") == "java_backend"
        assert manager.get_position_code("不存在") is None

    def test_get_questions(self, temp_dir, sample_question_bank):
        """测试获取问题"""
        bank_path = temp_dir / "questions"
        bank_path.mkdir()

        with open(bank_path / "java_backend.json", "w", encoding="utf-8") as f:
            json.dump(sample_question_bank, f)

        manager = QuestionBankManager(str(bank_path))
        questions = manager.get_questions("Java后端开发", question_type="technical", count=10)
        assert len(questions) == 2
        assert all(q["type"] == "technical" for q in questions)

    def test_generate_interview_plan(self, temp_dir, sample_question_bank):
        """测试生成面试计划"""
        bank_path = temp_dir / "questions"
        bank_path.mkdir()

        with open(bank_path / "java_backend.json", "w", encoding="utf-8") as f:
            json.dump(sample_question_bank, f)

        manager = QuestionBankManager(str(bank_path))
        plan = manager.generate_interview_plan("Java后端开发", num_technical=2, num_project=1, num_behavioral=1)
        assert len(plan["technical"]) == 2
        assert len(plan["project"]) == 1
        assert len(plan["behavioral"]) == 1

    def test_evaluate_answer(self, temp_dir, sample_question_bank):
        """测试评估回答"""
        bank_path = temp_dir / "questions"
        bank_path.mkdir()

        with open(bank_path / "java_backend.json", "w", encoding="utf-8") as f:
            json.dump(sample_question_bank, f)

        manager = QuestionBankManager(str(bank_path))
        result = manager.evaluate_answer("Java后端开发", "java-001", "使用Thread和Runnable实现多线程")
        assert "score" in result
        assert "feedback" in result
        assert "matched_keywords" in result

    def test_evaluate_answer_no_keywords(self, temp_dir):
        """测试无关键词题目的评估"""
        bank_path = temp_dir / "questions"
        bank_path.mkdir()

        bank_data = {
            "position": "测试岗位",
            "position_code": "test",
            "questions": [
                {"id": "test-001", "type": "technical", "difficulty": 1, "question": "测试问题", "keywords": []}
            ]
        }

        with open(bank_path / "test.json", "w", encoding="utf-8") as f:
            json.dump(bank_data, f)

        manager = QuestionBankManager(str(bank_path))
        result = manager.evaluate_answer("测试岗位", "test-001", "回答")
        assert result["score"] == 60


class TestInterviewEngine:
    """测试面试引擎"""

    @patch("src.interview_engine.QuestionBankManager")
    @patch("src.interview_engine.HybridProcessor")
    def test_init(self, mock_processor, mock_bank):
        """测试初始化"""
        mock_bank.return_value.question_banks = {}
        mock_processor.return_value = None

        engine = InterviewEngine()
        assert engine.question_bank is not None
        assert engine.session is None

    @patch("src.interview_engine.QuestionBankManager")
    @patch("src.interview_engine.HybridProcessor")
    def test_create_session(self, mock_processor, mock_bank):
        """测试创建会话"""
        mock_bank.return_value.question_banks = {}
        mock_bank.return_value.generate_interview_plan.return_value = {
            "technical": [], "project": [], "behavioral": []
        }
        mock_processor.return_value = None

        engine = InterviewEngine()
        session = engine.create_session("Java后端开发", "测试候选人", num_technical=2, num_project=1, num_behavioral=1)

        assert session is not None
        assert session.position == "Java后端开发"
        assert session.candidate_name == "测试候选人"
        assert engine.session == session

    @patch("src.interview_engine.QuestionBankManager")
    @patch("src.interview_engine.HybridProcessor")
    def test_start_interview(self, mock_processor, mock_bank):
        """测试开始面试"""
        mock_bank.return_value.question_banks = {}
        mock_bank.return_value.generate_interview_plan.return_value = {
            "technical": [], "project": [], "behavioral": []
        }
        mock_processor.return_value = None

        engine = InterviewEngine()
        engine.create_session("Java后端开发", "测试候选人")
        result = engine.start_interview()

        assert "error" not in result
        assert "session_id" in result
        assert "full_message" in result
        assert engine.session.status == InterviewStatus.GREETING
        assert engine.session.started_at is not None

    @patch("src.interview_engine.QuestionBankManager")
    @patch("src.interview_engine.HybridProcessor")
    def test_start_interview_no_session(self, mock_processor, mock_bank):
        """测试无会话时开始面试"""
        mock_bank.return_value.question_banks = {}
        mock_processor.return_value = None

        engine = InterviewEngine()
        result = engine.start_interview()
        assert "error" in result

    @patch("src.interview_engine.QuestionBankManager")
    @patch("src.interview_engine.HybridProcessor")
    def test_process_answer(self, mock_processor, mock_bank):
        """测试处理回答"""
        mock_bank.return_value.question_banks = {}
        mock_bank.return_value.generate_interview_plan.return_value = {
            "technical": [
                {"id": "q1", "type": "technical", "difficulty": 2, "question": "问题1"}
            ],
            "project": [],
            "behavioral": []
        }
        mock_processor.return_value = None

        engine = InterviewEngine()
        engine.create_session("Java后端开发", "测试候选人", num_technical=1, num_project=0, num_behavioral=0)
        engine.start_interview()

        result = engine.process_answer("这是我的回答")
        assert "error" not in result
        assert "type" in result

    @patch("src.interview_engine.QuestionBankManager")
    @patch("src.interview_engine.HybridProcessor")
    def test_generate_report(self, mock_processor, mock_bank):
        """测试生成报告"""
        mock_bank.return_value.question_banks = {}
        mock_bank.return_value.generate_interview_plan.return_value = {
            "technical": [], "project": [], "behavioral": []
        }
        mock_processor.return_value = None

        engine = InterviewEngine()
        engine.create_session("Java后端开发", "测试候选人")
        engine.start_interview()

        turn = DialogueTurn(
            interviewer_question="问题",
            candidate_answer="回答",
            scores={
                "technical": 80,
                "communication": 75,
                "completeness": 70,
                "problem_solving": 78,
                "teamwork": 72,
                "leadership": 68
            }
        )
        engine.session.dialogue_history.append(turn)
        engine.session.started_at = engine.session.created_at

        report = engine.generate_report()
        assert "overall_score" in report
        assert "recommendation" in report
        assert "dimension_scores" in report
        assert "strengths" in report
        assert "improvements" in report

    @patch("src.interview_engine.QuestionBankManager")
    @patch("src.interview_engine.HybridProcessor")
    def test_generate_report_empty(self, mock_processor, mock_bank):
        """测试空会话生成报告"""
        mock_bank.return_value.question_banks = {}
        mock_processor.return_value = None

        engine = InterviewEngine()
        report = engine.generate_report()
        assert report == {}

    @patch("src.interview_engine.QuestionBankManager")
    @patch("src.interview_engine.HybridProcessor")
    def test_end_interview(self, mock_processor, mock_bank):
        """测试结束面试"""
        mock_bank.return_value.question_banks = {}
        mock_bank.return_value.generate_interview_plan.return_value = {
            "technical": [], "project": [], "behavioral": []
        }
        mock_processor.return_value = None

        engine = InterviewEngine()
        engine.create_session("Java后端开发", "测试候选人")
        engine.start_interview()

        result = engine.end_interview()
        assert "error" not in result
        assert result["type"] == "closing"
        assert engine.session.status == InterviewStatus.COMPLETED
        assert engine.session.ended_at is not None
