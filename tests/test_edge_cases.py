"""
边界场景测试补充
覆盖 interview_engine.py 中的未覆盖边界条件
"""

import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock
from src.interview_engine import (
    InterviewPhase,
    InterviewStatus,
    DialogueTurn,
    InterviewSession,
    InterviewEngine,
)


class TestInputValidationEdgeCases:
    """输入验证边界场景"""

    @patch("src.interview_engine.MultimodalEvaluator")
    @patch("src.interview_engine.QuestionBankManager")
    @patch("src.interview_engine.HybridProcessor")
    def test_unicode_and_special_characters(self, mock_processor, mock_bank, mock_multimodal):
        """测试Unicode和特殊字符处理"""
        mock_bank.return_value.question_banks = {}
        mock_bank.return_value.generate_interview_plan.return_value = {
            "technical": [], "project": [], "behavioral": []
        }
        mock_processor.return_value = None

        engine = InterviewEngine()
        engine.create_session("Java后端开发", "测试候选人", num_technical=0, num_project=0, num_behavioral=0)

        # 测试各种特殊字符
        special_inputs = [
            "Hello\x00World",  # 空字符
            "Test\x01\x02\x03",  # 控制字符
            "🎉🎊🎁",  # Emoji
            "<script>alert('xss')</script>",  # XSS尝试
            "'; DROP TABLE users; --",  # SQL注入尝试
            "A" * 9999,  # 边界长度
        ]

        for input_text in special_inputs:
            result = engine._validate_answer(input_text)
            # 验证系统不会崩溃，返回有效结果
            assert isinstance(result, dict)
            assert "valid" in result

    @patch("src.interview_engine.MultimodalEvaluator")
    @patch("src.interview_engine.QuestionBankManager")
    @patch("src.interview_engine.HybridProcessor")
    def test_whitespace_only_answer(self, mock_processor, mock_bank, mock_multimodal):
        """测试仅空白字符的回答"""
        mock_bank.return_value.question_banks = {}
        mock_bank.return_value.generate_interview_plan.return_value = {
            "technical": [], "project": [], "behavioral": []
        }
        mock_processor.return_value = None

        engine = InterviewEngine()
        engine.create_session("Java后端开发", "测试候选人", num_technical=0, num_project=0, num_behavioral=0)

        # 各种空白字符组合
        whitespace_inputs = ["", "   ", "\t\n\r", "  \t  "]

        for ws in whitespace_inputs:
            result = engine._validate_answer(ws)
            assert result["valid"] is False
            assert result["error_code"] == "VAL_001"


class TestPhaseTransitionEdgeCases:
    """阶段切换边界场景"""

    @patch("src.interview_engine.MultimodalEvaluator")
    @patch("src.interview_engine.QuestionBankManager")
    @patch("src.interview_engine.HybridProcessor")
    def test_multiple_phases_empty(self, mock_processor, mock_bank, mock_multimodal):
        """测试多个阶段同时为空时的切换"""
        mock_bank.return_value.question_banks = {}
        mock_bank.return_value.generate_interview_plan.return_value = {
            "technical": [], "project": [], "behavioral": []
        }
        mock_processor.return_value = None

        engine = InterviewEngine()
        engine.create_session("Java后端开发", "测试候选人", num_technical=0, num_project=0, num_behavioral=0)
        engine.session.interview_plan = {
            "technical": [], "project": [], "behavioral": []
        }

        # 从自我介绍阶段切换
        engine.session.current_phase = InterviewPhase.SELF_INTRO
        engine._update_phase()

        # 所有阶段为空，应该进入候选人提问阶段
        assert engine.session.current_phase == InterviewPhase.CANDIDATE_QA

    @patch("src.interview_engine.MultimodalEvaluator")
    @patch("src.interview_engine.QuestionBankManager")
    @patch("src.interview_engine.HybridProcessor")
    def test_phase_transition_with_no_plan(self, mock_processor, mock_bank, mock_multimodal):
        """测试面试计划为空时的阶段切换"""
        mock_bank.return_value.question_banks = {}
        mock_bank.return_value.generate_interview_plan.return_value = {
            "technical": [], "project": [], "behavioral": []
        }
        mock_processor.return_value = None

        engine = InterviewEngine()
        engine.create_session("Java后端开发", "测试候选人", num_technical=0, num_project=0, num_behavioral=0)
        engine.session.interview_plan = {}

        # 从自我介绍阶段切换
        engine.session.current_phase = InterviewPhase.SELF_INTRO
        engine._update_phase()

        # 计划为空，应该进入结束阶段
        assert engine.session.current_phase == InterviewPhase.CANDIDATE_QA


class TestReportGenerationEdgeCases:
    """报告生成边界场景"""

    @patch("src.interview_engine.MultimodalEvaluator")
    @patch("src.interview_engine.QuestionBankManager")
    @patch("src.interview_engine.HybridProcessor")
    def test_empty_dialogue_history(self, mock_processor, mock_bank, mock_multimodal):
        """测试空对话历史时的报告生成"""
        mock_bank.return_value.question_banks = {}
        mock_bank.return_value.generate_interview_plan.return_value = {
            "technical": [], "project": [], "behavioral": []
        }
        mock_processor.return_value = None

        engine = InterviewEngine()
        engine.create_session("Java后端开发", "测试候选人", num_technical=0, num_project=0, num_behavioral=0)
        engine.start_interview()

        # 清空调话历史
        engine.session.dialogue_history = []

        # 生成报告不应崩溃
        report = engine.generate_report()
        assert report is not None
        assert "session_id" in report

    @patch("src.interview_engine.MultimodalEvaluator")
    @patch("src.interview_engine.QuestionBankManager")
    @patch("src.interview_engine.HybridProcessor")
    def test_report_with_no_evaluations(self, mock_processor, mock_bank, mock_multimodal):
        """测试没有评估数据的报告生成"""
        mock_bank.return_value.question_banks = {}
        mock_bank.return_value.generate_interview_plan.return_value = {
            "technical": [], "project": [], "behavioral": []
        }
        mock_processor.return_value = None

        engine = InterviewEngine()
        engine.create_session("Java后端开发", "测试候选人", num_technical=0, num_project=0, num_behavioral=0)
        engine.start_interview()

        # 添加一个没有评估的回合
        turn = DialogueTurn(
            interviewer_question="测试问题",
            question_id="q1",
            question_type="technical",
            candidate_answer="测试回答"
        )
        # 不设置 evaluation
        engine.session.dialogue_history.append(turn)

        # 生成报告不应崩溃
        report = engine.generate_report()
        assert report is not None
        assert "overall_score" in report


class TestFollowUpEdgeCases:
    """追问边界场景"""

    @patch("src.interview_engine.MultimodalEvaluator")
    @patch("src.interview_engine.QuestionBankManager")
    @patch("src.interview_engine.HybridProcessor")
    def test_follow_up_with_missing_parent(self, mock_processor, mock_bank, mock_multimodal):
        """测试父回合不存在时的追问处理"""
        mock_bank.return_value.question_banks = {}
        mock_bank.return_value.generate_interview_plan.return_value = {
            "technical": [{"id": "q1", "type": "technical"}],
            "project": [], "behavioral": []
        }
        mock_processor.return_value = None

        engine = InterviewEngine()
        engine.create_session("Java后端开发", "测试候选人", num_technical=1, num_project=0, num_behavioral=0)

        # 添加一个追问回合，但父回合不存在
        follow_up_turn = DialogueTurn(
            interviewer_question="追问",
            question_id="q1",
            question_type="technical",
            is_follow_up=True,
            parent_turn_id="non-existent-id",
            follow_up_count=1
        )
        follow_up_turn.candidate_answer = "回答"
        follow_up_turn.evaluation = {"score": 50}
        engine.session.dialogue_history.append(follow_up_turn)

        # 尝试生成追问，不应崩溃
        result = engine._generate_follow_up({"score": 50})
        # 应该返回追问或进入下一题
        assert result is not None
        assert isinstance(result, dict)

    @patch("src.interview_engine.MultimodalEvaluator")
    @patch("src.interview_engine.QuestionBankManager")
    @patch("src.interview_engine.HybridProcessor")
    def test_deep_follow_up_chain(self, mock_processor, mock_bank, mock_multimodal):
        """测试深度追问链（超过正常深度）"""
        mock_bank.return_value.question_banks = {}
        mock_bank.return_value.generate_interview_plan.return_value = {
            "technical": [{"id": "q1", "type": "technical"}],
            "project": [], "behavioral": []
        }
        mock_processor.return_value = None

        engine = InterviewEngine()
        engine.create_session("Java后端开发", "测试候选人", num_technical=1, num_project=0, num_behavioral=0)

        # 创建原始回合
        original_turn = DialogueTurn(
            interviewer_question="原始问题",
            question_id="q1",
            question_type="technical",
            is_follow_up=False,
            follow_up_count=0
        )
        engine.session.dialogue_history.append(original_turn)

        # 创建深度追问链（5层，超过默认的2层限制）
        parent_id = original_turn.turn_id
        for i in range(5):
            follow_up = DialogueTurn(
                interviewer_question=f"追问{i+1}",
                question_id="q1",
                question_type="technical",
                is_follow_up=True,
                parent_turn_id=parent_id,
                follow_up_count=i+1
            )
            follow_up.candidate_answer = f"回答{i+1}"
            follow_up.evaluation = {"score": 50}
            engine.session.dialogue_history.append(follow_up)
            parent_id = follow_up.turn_id

        # 设置当前回合为最后一个追问
        engine.session.dialogue_history[-1].candidate_answer = "最终回答"
        engine.session.dialogue_history[-1].evaluation = {"score": 50}

        # 尝试生成追问，应该被拒绝（超过深度限制）
        should_follow = engine._should_follow_up()
        assert should_follow is False


class TestSessionEdgeCases:
    """会话边界场景"""

    @patch("src.interview_engine.MultimodalEvaluator")
    @patch("src.interview_engine.QuestionBankManager")
    @patch("src.interview_engine.HybridProcessor")
    def test_session_with_extreme_timeouts(self, mock_processor, mock_bank, mock_multimodal):
        """测试极端超时配置"""
        mock_bank.return_value.question_banks = {}
        mock_bank.return_value.generate_interview_plan.return_value = {
            "technical": [], "project": [], "behavioral": []
        }
        mock_processor.return_value = None

        engine = InterviewEngine()
        engine.create_session("Java后端开发", "测试候选人", num_technical=0, num_project=0, num_behavioral=0)

        # 设置极端超时值
        engine.session.max_interview_duration_seconds = 0  # 立即超时
        engine.session.max_answer_wait_seconds = 0
        engine.session.started_at = datetime.now().isoformat()
        engine.session.status = InterviewStatus.IN_PROGRESS

        # 处理回答应该立即超时
        result = engine.process_answer("测试")
        assert engine.session.status == InterviewStatus.COMPLETED

    @patch("src.interview_engine.MultimodalEvaluator")
    @patch("src.interview_engine.QuestionBankManager")
    @patch("src.interview_engine.HybridProcessor")
    def test_concurrent_operations(self, mock_processor, mock_bank, mock_multimodal):
        """测试并发操作安全性"""
        mock_bank.return_value.question_banks = {}
        mock_bank.return_value.generate_interview_plan.return_value = {
            "technical": [{"id": "q1", "type": "technical"}],
            "project": [], "behavioral": []
        }
        mock_processor.return_value = None

        engine = InterviewEngine()
        engine.create_session("Java后端开发", "测试候选人", num_technical=1, num_project=0, num_behavioral=0)
        engine.start_interview()

        # 模拟并发追问计数更新
        import threading

        def add_follow_up():
            turn = DialogueTurn(
                interviewer_question="追问",
                question_id="q1",
                question_type="technical",
                is_follow_up=True
            )
            engine.session.dialogue_history.append(turn)
            engine.session.total_follow_ups += 1

        threads = []
        for _ in range(10):
            t = threading.Thread(target=add_follow_up)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # 验证计数一致性
        assert engine.session.total_follow_ups == 10
        follow_up_turns = [t for t in engine.session.dialogue_history if t.is_follow_up]
        assert len(follow_up_turns) == 10


class TestErrorHandlingEdgeCases:
    """错误处理边界场景"""

    @patch("src.interview_engine.MultimodalEvaluator")
    @patch("src.interview_engine.QuestionBankManager")
    @patch("src.interview_engine.HybridProcessor")
    def test_process_answer_with_none_input(self, mock_processor, mock_bank, mock_multimodal):
        """测试None输入处理"""
        mock_bank.return_value.question_banks = {}
        mock_bank.return_value.generate_interview_plan.return_value = {
            "technical": [], "project": [], "behavioral": []
        }
        mock_processor.return_value = None

        engine = InterviewEngine()
        engine.create_session("Java后端开发", "测试候选人", num_technical=0, num_project=0, num_behavioral=0)
        engine.start_interview()

        # None输入应该被处理
        result = engine.process_answer(None)
        assert result is not None
        assert isinstance(result, dict)

    @patch("src.interview_engine.MultimodalEvaluator")
    @patch("src.interview_engine.QuestionBankManager")
    @patch("src.interview_engine.HybridProcessor")
    def test_database_failure_handling(self, mock_processor, mock_bank, mock_multimodal):
        """测试数据库失败时的处理"""
        mock_bank.return_value.question_banks = {}
        mock_bank.return_value.generate_interview_plan.return_value = {
            "technical": [], "project": [], "behavioral": []
        }
        mock_processor.return_value = None

        engine = InterviewEngine()
        engine.create_session("Java后端开发", "测试候选人", num_technical=0, num_project=0, num_behavioral=0)
        engine.start_interview()

        # 模拟数据库保存失败
        with patch('src.interview_engine.db_manager.save_interview', side_effect=Exception("DB Error")):
            # 处理回答不应崩溃
            result = engine.process_answer("测试回答")
            assert result is not None
