"""
P2 修复验证测试
- 验证过渡语去重
- 验证通用问题个性化
- 验证报告置信度
- 验证输入验证完善
- 验证日志脱敏
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


class TestTransitionDeduplication:
    """测试过渡语去重 (P2-8)"""

    @patch("src.interview_engine.QuestionBankManager")
    @patch("src.interview_engine.HybridProcessor")
    def test_transition_not_repeated(self, mock_processor, mock_bank):
        """测试相同过渡语不会连续重复"""
        mock_bank.return_value.question_banks = {}
        mock_bank.return_value.generate_interview_plan.return_value = {
            "technical": [], "project": [], "behavioral": []
        }
        mock_processor.return_value = None

        engine = InterviewEngine()
        engine.create_session("Java后端开发", "测试候选人", num_technical=0, num_project=0, num_behavioral=0)

        # 模拟多次获取过渡语（候选池只有3个，测试相邻不重复即可）
        transitions = []
        for _ in range(3):
            transition = engine._get_transition(
                InterviewPhase.SELF_INTRO,
                InterviewPhase.TECHNICAL
            )
            if transition:
                transitions.append(transition)

        # 检查相邻过渡语不重复（由于过滤最近3次，相邻的一定不同）
        for i in range(len(transitions) - 1):
            # 相邻两次不应相同
            assert transitions[i] != transitions[i + 1], \
                f"过渡语重复: '{transitions[i]}' 在位置 {i} 和 {i+1}"

    @patch("src.interview_engine.QuestionBankManager")
    @patch("src.interview_engine.HybridProcessor")
    def test_transition_history_tracked(self, mock_processor, mock_bank):
        """测试过渡语历史被正确记录"""
        mock_bank.return_value.question_banks = {}
        mock_bank.return_value.generate_interview_plan.return_value = {
            "technical": [], "project": [], "behavioral": []
        }
        mock_processor.return_value = None

        engine = InterviewEngine()
        engine.create_session("Java后端开发", "测试候选人", num_technical=0, num_project=0, num_behavioral=0)

        # 获取过渡语
        transition = engine._get_transition(
            InterviewPhase.SELF_INTRO,
            InterviewPhase.TECHNICAL
        )

        # 验证历史记录中有该过渡语
        assert transition in engine.session.used_transitions


class TestPersonalizedQuestion:
    """测试通用问题个性化 (P2-9)"""

    @patch("src.interview_engine.QuestionBankManager")
    @patch("src.interview_engine.HybridProcessor")
    def test_personalized_question_from_tech_keywords(self, mock_processor, mock_bank):
        """测试根据技术关键词生成个性化问题"""
        mock_bank.return_value.question_banks = {}
        mock_bank.return_value.generate_interview_plan.return_value = {
            "technical": [], "project": [], "behavioral": []
        }
        mock_processor.return_value = None

        engine = InterviewEngine()
        engine.create_session("Java后端开发", "测试候选人", num_technical=0, num_project=0, num_behavioral=0)

        # 添加一个包含技术关键词的历史回答
        turn = DialogueTurn(
            interviewer_question="你用过哪些技术？",
            question_id="t1",
            question_type="technical",
            is_follow_up=False
        )
        turn.candidate_answer = "我主要使用Java和Spring框架开发后端服务"
        engine.session.dialogue_history.append(turn)

        # 生成个性化问题
        question = engine._generate_personalized_question(InterviewPhase.TECHNICAL)

        # 应该生成包含"java"或"spring"的个性化问题
        assert question is not None
        assert "java" in question.lower() or "spring" in question.lower()

    @patch("src.interview_engine.QuestionBankManager")
    @patch("src.interview_engine.HybridProcessor")
    def test_personalized_question_from_short_answer(self, mock_processor, mock_bank):
        """测试根据简短回答生成鼓励展开的问题"""
        mock_bank.return_value.question_banks = {}
        mock_bank.return_value.generate_interview_plan.return_value = {
            "technical": [], "project": [], "behavioral": []
        }
        mock_processor.return_value = None

        engine = InterviewEngine()
        engine.create_session("Java后端开发", "测试候选人", num_technical=0, num_project=0, num_behavioral=0)

        # 添加一个简短回答
        turn = DialogueTurn(
            interviewer_question="请介绍你的项目经验",
            question_id="p1",
            question_type="project",
            is_follow_up=False
        )
        turn.candidate_answer = "做过一个项目"
        engine.session.dialogue_history.append(turn)

        # 生成个性化问题
        question = engine._generate_personalized_question(InterviewPhase.PROJECT)

        # 应该生成鼓励展开的问题
        assert question is not None
        assert "详细" in question or "展开" in question


class TestInputValidation:
    """测试输入验证完善 (P2-11)"""

    @patch("src.interview_engine.QuestionBankManager")
    @patch("src.interview_engine.HybridProcessor")
    def test_empty_answer_rejected(self, mock_processor, mock_bank):
        """测试空回答被拒绝"""
        mock_bank.return_value.question_banks = {}
        mock_bank.return_value.generate_interview_plan.return_value = {
            "technical": [], "project": [], "behavioral": []
        }
        mock_processor.return_value = None

        engine = InterviewEngine()
        engine.create_session("Java后端开发", "测试候选人", num_technical=0, num_project=0, num_behavioral=0)
        engine.start_interview()

        result = engine._validate_answer("")
        assert result["valid"] is False
        assert result["error_code"] == "VAL_001"

    @patch("src.interview_engine.QuestionBankManager")
    @patch("src.interview_engine.HybridProcessor")
    def test_long_answer_rejected(self, mock_processor, mock_bank):
        """测试超长回答被拒绝"""
        mock_bank.return_value.question_banks = {}
        mock_bank.return_value.generate_interview_plan.return_value = {
            "technical": [], "project": [], "behavioral": []
        }
        mock_processor.return_value = None

        engine = InterviewEngine()
        engine.create_session("Java后端开发", "测试候选人", num_technical=0, num_project=0, num_behavioral=0)

        # 生成超过10000字符的回答
        long_answer = "a" * 10001
        result = engine._validate_answer(long_answer)
        assert result["valid"] is False
        assert result["error_code"] == "VAL_002"

    @patch("src.interview_engine.QuestionBankManager")
    @patch("src.interview_engine.HybridProcessor")
    def test_repetitive_answer_rejected(self, mock_processor, mock_bank):
        """测试重复内容过多的回答被拒绝"""
        mock_bank.return_value.question_banks = {}
        mock_bank.return_value.generate_interview_plan.return_value = {
            "technical": [], "project": [], "behavioral": []
        }
        mock_processor.return_value = None

        engine = InterviewEngine()
        engine.create_session("Java后端开发", "测试候选人", num_technical=0, num_project=0, num_behavioral=0)

        # 生成重复字符过多的回答
        repetitive_answer = "a" * 100
        result = engine._validate_answer(repetitive_answer)
        assert result["valid"] is False
        assert result["error_code"] == "VAL_003"

    @patch("src.interview_engine.QuestionBankManager")
    @patch("src.interview_engine.HybridProcessor")
    def test_valid_answer_accepted(self, mock_processor, mock_bank):
        """测试有效回答被接受"""
        mock_bank.return_value.question_banks = {}
        mock_bank.return_value.generate_interview_plan.return_value = {
            "technical": [], "project": [], "behavioral": []
        }
        mock_processor.return_value = None

        engine = InterviewEngine()
        engine.create_session("Java后端开发", "测试候选人", num_technical=0, num_project=0, num_behavioral=0)

        valid_answer = "我在之前的项目中主要负责后端开发，使用Java和Spring Boot框架，实现了用户认证、订单管理等功能。"
        result = engine._validate_answer(valid_answer)
        assert result["valid"] is True


class TestLogSanitization:
    """测试日志脱敏 (P2-12)"""

    @patch("src.interview_engine.QuestionBankManager")
    @patch("src.interview_engine.HybridProcessor")
    def test_log_does_not_contain_full_answer(self, mock_processor, mock_bank, caplog):
        """测试日志不包含完整回答内容"""
        mock_bank.return_value.question_banks = {}
        mock_bank.return_value.generate_interview_plan.return_value = {
            "technical": [], "project": [], "behavioral": []
        }
        mock_processor.return_value = None

        engine = InterviewEngine()
        engine.create_session("Java后端开发", "测试候选人", num_technical=0, num_project=0, num_behavioral=0)
        engine.start_interview()

        # 处理回答
        with caplog.at_level("INFO"):
            engine.process_answer("这是我的敏感个人信息和项目经验")

        # 检查日志中不包含完整回答内容
        for record in caplog.records:
            assert "敏感个人信息" not in record.message
