"""
P1 修复验证测试
- 验证面试结束判断完善
- 验证评分权重动态化
- 验证异常信息脱敏
- 验证面试进度反馈
"""

import pytest
from unittest.mock import patch, MagicMock
from src.interview_engine import (
    InterviewPhase,
    InterviewStatus,
    DialogueTurn,
    InterviewSession,
    InterviewEngine,
)


class TestInterviewEndCondition:
    """测试面试结束判断完善 (P1-4)"""

    @patch("src.interview_engine.QuestionBankManager")
    @patch("src.interview_engine.HybridProcessor")
    def test_should_end_when_closing_phase(self, mock_processor, mock_bank):
        """测试CLOSING阶段正确结束"""
        mock_bank.return_value.question_banks = {}
        mock_bank.return_value.generate_interview_plan.return_value = {
            "technical": [], "project": [], "behavioral": []
        }
        mock_processor.return_value = None

        engine = InterviewEngine()
        engine.create_session("Java后端开发", "测试候选人", num_technical=0, num_project=0, num_behavioral=0)
        engine.session.current_phase = InterviewPhase.CLOSING

        assert engine._should_end_interview() is True

    @patch("src.interview_engine.QuestionBankManager")
    @patch("src.interview_engine.HybridProcessor")
    def test_should_not_end_when_incomplete(self, mock_processor, mock_bank):
        """测试未完成时不应结束"""
        mock_bank.return_value.question_banks = {}
        mock_bank.return_value.generate_interview_plan.return_value = {
            "technical": [{"id": "t1", "type": "technical"}],
            "project": [], "behavioral": []
        }
        mock_processor.return_value = None

        engine = InterviewEngine()
        engine.create_session("Java后端开发", "测试候选人", num_technical=1, num_project=0, num_behavioral=0)
        engine.session.current_phase = InterviewPhase.TECHNICAL
        engine.session.interview_plan = {
            "technical": [{"id": "t1", "type": "technical"}],
            "project": [], "behavioral": []
        }

        # 技术题还没问完
        assert engine._should_end_interview() is False

    @patch("src.interview_engine.QuestionBankManager")
    @patch("src.interview_engine.HybridProcessor")
    def test_should_end_when_all_phases_completed(self, mock_processor, mock_bank):
        """测试所有阶段完成后应结束"""
        mock_bank.return_value.question_banks = {}
        mock_bank.return_value.generate_interview_plan.return_value = {
            "technical": [{"id": "t1", "type": "technical"}],
            "project": [], "behavioral": []
        }
        mock_processor.return_value = None

        engine = InterviewEngine()
        engine.create_session("Java后端开发", "测试候选人", num_technical=1, num_project=0, num_behavioral=0)
        engine.session.current_phase = InterviewPhase.CANDIDATE_QA
        engine.session.interview_plan = {
            "technical": [{"id": "t1", "type": "technical"}],
            "project": [], "behavioral": []
        }
        # 技术题已问完
        engine.session.asked_question_ids = ["t1"]

        assert engine._check_interview_completeness() is True


class TestErrorCodeSanitization:
    """测试异常信息脱敏 (P1-6)"""

    @patch("src.interview_engine.QuestionBankManager")
    @patch("src.interview_engine.HybridProcessor")
    def test_error_response_has_error_code(self, mock_processor, mock_bank):
        """测试错误响应包含 error_code"""
        mock_bank.return_value.question_banks = {}
        mock_bank.return_value.generate_interview_plan.return_value = {
            "technical": [], "project": [], "behavioral": []
        }
        mock_processor.return_value = None

        engine = InterviewEngine()

        # 测试未创建会话时的错误
        result = engine.start_interview()
        assert "error_code" in result
        assert result["error_code"] == "SES_001"
        assert "请稍后重试" in result["error"] or "请先创建" in result["error"]

    @patch("src.interview_engine.QuestionBankManager")
    @patch("src.interview_engine.HybridProcessor")
    def test_error_response_no_internal_details(self, mock_processor, mock_bank):
        """测试错误响应不包含内部异常详情"""
        mock_bank.return_value.question_banks = {}
        mock_bank.return_value.generate_interview_plan.return_value = {
            "technical": [], "project": [], "behavioral": []
        }
        mock_processor.return_value = None

        engine = InterviewEngine()
        engine.create_session("Java后端开发", "测试候选人", num_technical=0, num_project=0, num_behavioral=0)
        engine.start_interview()

        # 模拟一个会导致异常的回答处理（通过mock评估器）
        with patch.object(engine, '_evaluate_current_answer', side_effect=Exception("内部数据库连接失败")):
            result = engine.process_answer("测试回答")
            assert "error_code" in result
            # 错误消息不应包含原始异常信息
            assert "数据库连接失败" not in result["error"]
            assert "请稍后重试" in result["error"]


class TestProgressFeedback:
    """测试面试进度实时反馈 (P1-7)"""

    @patch("src.interview_engine.QuestionBankManager")
    @patch("src.interview_engine.HybridProcessor")
    def test_progress_contains_detailed_info(self, mock_processor, mock_bank):
        """测试进度信息包含详细字段"""
        mock_bank.return_value.question_banks = {}
        mock_bank.return_value.generate_interview_plan.return_value = {
            "technical": [
                {"id": "t1", "type": "technical", "difficulty": 2, "question": "技术问题1"},
                {"id": "t2", "type": "technical", "difficulty": 2, "question": "技术问题2"}
            ],
            "project": [],
            "behavioral": []
        }
        mock_processor.return_value = None

        engine = InterviewEngine()
        engine.create_session("Java后端开发", "测试候选人", num_technical=2, num_project=0, num_behavioral=0)
        engine.start_interview()

        # 获取详细进度
        progress = engine.session.get_detailed_progress()

        assert "current_phase" in progress
        assert "phase_progress" in progress
        assert "overall" in progress
        assert "follow_ups" in progress
        assert "estimated_remaining_minutes" in progress

        # 验证 overall 结构
        assert "asked" in progress["overall"]
        assert "total" in progress["overall"]
        assert "percentage" in progress["overall"]

        # 验证 phase_progress 结构
        assert "technical" in progress["phase_progress"]
        assert "asked" in progress["phase_progress"]["technical"]
        assert "total" in progress["phase_progress"]["technical"]
        assert "completed" in progress["phase_progress"]["technical"]

    @patch("src.interview_engine.QuestionBankManager")
    @patch("src.interview_engine.HybridProcessor")
    def test_progress_updates_after_question(self, mock_processor, mock_bank):
        """测试回答问题后进度更新"""
        mock_bank.return_value.question_banks = {}
        mock_bank.return_value.generate_interview_plan.return_value = {
            "technical": [
                {"id": "t1", "type": "technical", "difficulty": 2, "question": "技术问题1"}
            ],
            "project": [],
            "behavioral": []
        }
        mock_processor.return_value = None

        engine = InterviewEngine()
        engine.create_session("Java后端开发", "测试候选人", num_technical=1, num_project=0, num_behavioral=0)
        engine.start_interview()

        # 回答第一个问题
        result = engine.process_answer("测试回答")

        # 验证返回中包含进度信息
        assert "progress" in result
        progress = result["progress"]
        assert isinstance(progress, dict)
        assert progress["overall"]["asked"] >= 1

    @patch("src.interview_engine.QuestionBankManager")
    @patch("src.interview_engine.HybridProcessor")
    def test_progress_percentage_calculation(self, mock_processor, mock_bank):
        """测试进度百分比计算正确"""
        mock_bank.return_value.question_banks = {}
        mock_bank.return_value.generate_interview_plan.return_value = {
            "technical": [
                {"id": "t1", "type": "technical"},
                {"id": "t2", "type": "technical"}
            ],
            "project": [
                {"id": "p1", "type": "project"}
            ],
            "behavioral": []
        }
        mock_processor.return_value = None

        engine = InterviewEngine()
        engine.create_session("Java后端开发", "测试候选人", num_technical=2, num_project=1, num_behavioral=0)
        engine.session.interview_plan = {
            "technical": [{"id": "t1"}, {"id": "t2"}],
            "project": [{"id": "p1"}],
            "behavioral": []
        }

        # 问完1/3的题目
        engine.session.asked_question_ids = ["t1"]
        progress = engine.session.get_detailed_progress()

        assert progress["overall"]["asked"] == 1
        assert progress["overall"]["total"] == 3
        assert progress["overall"]["percentage"] == 33.3
