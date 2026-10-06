"""
P0 修复验证测试
- 验证追问竞态条件修复
- 验证阶段切换跳过问题修复
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


class TestFollowUpRaceCondition:
    """测试追问竞态条件修复 (P0-1)"""

    @patch("src.interview_engine.QuestionBankManager")
    @patch("src.interview_engine.HybridProcessor")
    def test_follow_up_atomic_update(self, mock_processor, mock_bank):
        """测试追问计数和回合创建是原子操作"""
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

        # 模拟一个原始问题回合
        original_turn = DialogueTurn(
            interviewer_question="原始问题",
            question_id="q1",
            question_type="technical",
            difficulty=2,
            is_follow_up=False,
            follow_up_count=0
        )
        original_turn.candidate_answer = "测试回答"
        original_turn.evaluation = {"score": 50, "feedback": "需要追问"}
        engine.session.dialogue_history.append(original_turn)
        engine.session.total_questions_asked = 1

        # 记录生成追问前的状态
        prev_follow_up_count = engine.session.total_follow_ups
        prev_turn_count = len(engine.session.dialogue_history)
        prev_original_follow_up_count = original_turn.follow_up_count

        # 生成追问
        result = engine._generate_follow_up({"score": 50, "feedback": "需要追问"})

        # 验证：新回合已创建
        assert len(engine.session.dialogue_history) == prev_turn_count + 1
        # 验证：全局追问计数已更新
        assert engine.session.total_follow_ups == prev_follow_up_count + 1
        # 验证：原始回合计数已更新
        assert original_turn.follow_up_count == prev_original_follow_up_count + 1
        # 验证：新回合的 follow_up_count 正确记录了追问深度
        new_turn = engine.session.dialogue_history[-1]
        assert new_turn.is_follow_up is True
        assert new_turn.follow_up_count == 1
        assert new_turn.parent_turn_id == original_turn.turn_id
        assert result["type"] == "follow_up"

    @patch("src.interview_engine.QuestionBankManager")
    @patch("src.interview_engine.HybridProcessor")
    def test_follow_up_count_consistency(self, mock_processor, mock_bank):
        """测试多次追问后计数保持一致"""
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

        # 创建原始问题回合
        original_turn = DialogueTurn(
            interviewer_question="原始问题",
            question_id="q1",
            question_type="technical",
            difficulty=2,
            is_follow_up=False,
            follow_up_count=0
        )
        original_turn.candidate_answer = "回答1"
        original_turn.evaluation = {"score": 50}
        engine.session.dialogue_history.append(original_turn)
        engine.session.total_questions_asked = 1

        # 生成第一次追问
        engine._generate_follow_up({"score": 50})
        assert original_turn.follow_up_count == 1
        assert engine.session.total_follow_ups == 1

        # 模拟回答第一次追问（得分 >= 60 才能继续追问，避免无限循环保护）
        follow_up_turn_1 = engine.session.dialogue_history[-1]
        follow_up_turn_1.candidate_answer = "回答2"
        follow_up_turn_1.evaluation = {"score": 65}

        # 生成第二次追问
        engine._generate_follow_up({"score": 65})
        assert original_turn.follow_up_count == 2
        assert engine.session.total_follow_ups == 2

        # 验证所有追问回合的 follow_up_count 递增
        follow_up_turns = [t for t in engine.session.dialogue_history if t.is_follow_up]
        assert len(follow_up_turns) == 2
        assert follow_up_turns[0].follow_up_count == 1
        assert follow_up_turns[1].follow_up_count == 2


class TestPhaseTransition:
    """测试阶段切换修复 (P0-2)"""

    @patch("src.interview_engine.QuestionBankManager")
    @patch("src.interview_engine.HybridProcessor")
    def test_phase_skips_empty_phases(self, mock_processor, mock_bank):
        """测试空阶段被正确跳过"""
        mock_bank.return_value.question_banks = {}
        # 面试计划中只有行为题，没有技术题和项目题
        mock_bank.return_value.generate_interview_plan.return_value = {
            "technical": [],
            "project": [],
            "behavioral": [
                {"id": "b1", "type": "behavioral", "difficulty": 2, "question": "行为问题1"}
            ]
        }
        mock_processor.return_value = None

        engine = InterviewEngine()
        engine.create_session(
            "Java后端开发", "测试候选人",
            num_technical=0, num_project=0, num_behavioral=1
        )
        engine.session.current_phase = InterviewPhase.SELF_INTRO
        engine.session.interview_plan = {
            "technical": [],
            "project": [],
            "behavioral": [{"id": "b1", "type": "behavioral"}]
        }

        # 从自我介绍阶段切换
        engine._update_phase()

        # 应该跳过技术和项目阶段，直接进入行为阶段
        assert engine.session.current_phase == InterviewPhase.BEHAVIORAL

    @patch("src.interview_engine.QuestionBankManager")
    @patch("src.interview_engine.HybridProcessor")
    def test_phase_respects_plan_not_config_count(self, mock_processor, mock_bank):
        """测试阶段切换基于面试计划而非仅配置数量"""
        mock_bank.return_value.question_banks = {}
        mock_bank.return_value.generate_interview_plan.return_value = {
            "technical": [
                {"id": "t1", "type": "technical", "difficulty": 2, "question": "技术问题1"},
                {"id": "t2", "type": "technical", "difficulty": 2, "question": "技术问题2"},
            ],
            "project": [],
            "behavioral": []
        }
        mock_processor.return_value = None

        engine = InterviewEngine()
        engine.create_session(
            "Java后端开发", "测试候选人",
            num_technical=1, num_project=0, num_behavioral=0
        )
        engine.session.current_phase = InterviewPhase.TECHNICAL
        engine.session.interview_plan = {
            "technical": [
                {"id": "t1", "type": "technical"},
                {"id": "t2", "type": "technical"}
            ],
            "project": [],
            "behavioral": []
        }

        # 只问了一个技术题（计划中有2个，但 num_technical=1）
        turn = DialogueTurn(
            interviewer_question="技术问题",
            question_id="t1",
            question_type="technical",
            is_follow_up=False
        )
        engine.session.dialogue_history.append(turn)

        # 更新阶段
        engine._update_phase()

        # 应该继续留在技术阶段，因为计划中还有题没问
        # 实际逻辑使用 max(num_technical, planned_count) = max(1, 2) = 2
        # 只问了1个，所以不应该切换
        assert engine.session.current_phase == InterviewPhase.TECHNICAL

    @patch("src.interview_engine.QuestionBankManager")
    @patch("src.interview_engine.HybridProcessor")
    def test_phase_transition_to_closing_when_all_empty(self, mock_processor, mock_bank):
        """测试所有阶段都为空时直接切换到结束"""
        mock_bank.return_value.question_banks = {}
        mock_bank.return_value.generate_interview_plan.return_value = {
            "technical": [],
            "project": [],
            "behavioral": []
        }
        mock_processor.return_value = None

        engine = InterviewEngine()
        engine.create_session(
            "Java后端开发", "测试候选人",
            num_technical=0, num_project=0, num_behavioral=0
        )
        engine.session.current_phase = InterviewPhase.SELF_INTRO
        engine.session.interview_plan = {
            "technical": [],
            "project": [],
            "behavioral": []
        }

        engine._update_phase()

        # 所有阶段都为空，但仍进入候选人提问阶段（给候选人提问机会）
        # 然后再次调用 _update_phase 才会进入结束阶段
        assert engine.session.current_phase == InterviewPhase.CANDIDATE_QA

        # 从候选人提问阶段切换到结束
        engine._update_phase()
        assert engine.session.current_phase == InterviewPhase.CLOSING

    @patch("src.interview_engine.QuestionBankManager")
    @patch("src.interview_engine.HybridProcessor")
    def test_phase_normal_transition(self, mock_processor, mock_bank):
        """测试正常阶段切换流程"""
        mock_bank.return_value.question_banks = {}
        mock_bank.return_value.generate_interview_plan.return_value = {
            "technical": [
                {"id": "t1", "type": "technical", "difficulty": 2, "question": "技术问题1"}
            ],
            "project": [
                {"id": "p1", "type": "project", "difficulty": 2, "question": "项目问题1"}
            ],
            "behavioral": [
                {"id": "b1", "type": "behavioral", "difficulty": 2, "question": "行为问题1"}
            ]
        }
        mock_processor.return_value = None

        engine = InterviewEngine()
        engine.create_session(
            "Java后端开发", "测试候选人",
            num_technical=1, num_project=1, num_behavioral=1
        )
        engine.session.interview_plan = {
            "technical": [{"id": "t1"}],
            "project": [{"id": "p1"}],
            "behavioral": [{"id": "b1"}]
        }

        # 自我介绍 -> 技术
        engine.session.current_phase = InterviewPhase.SELF_INTRO
        engine._update_phase()
        assert engine.session.current_phase == InterviewPhase.TECHNICAL

        # 技术 -> 项目（技术题已问完）
        turn = DialogueTurn(
            interviewer_question="技术问题",
            question_id="t1",
            question_type="technical",
            is_follow_up=False
        )
        engine.session.dialogue_history.append(turn)
        engine._update_phase()
        assert engine.session.current_phase == InterviewPhase.PROJECT

        # 项目 -> 行为（项目题已问完）
        turn = DialogueTurn(
            interviewer_question="项目问题",
            question_id="p1",
            question_type="project",
            is_follow_up=False
        )
        engine.session.dialogue_history.append(turn)
        engine._update_phase()
        assert engine.session.current_phase == InterviewPhase.BEHAVIORAL

        # 行为 -> 候选人提问（行为题已问完）
        turn = DialogueTurn(
            interviewer_question="行为问题",
            question_id="b1",
            question_type="behavioral",
            is_follow_up=False
        )
        engine.session.dialogue_history.append(turn)
        engine._update_phase()
        assert engine.session.current_phase == InterviewPhase.CANDIDATE_QA

        # 候选人提问阶段直接调用 _update_phase 会切换到结束
        # 这是正常流程：候选人在 CANDIDATE_QA 阶段提问后，面试官调用 _update_phase 结束面试
        engine._update_phase()
        assert engine.session.current_phase == InterviewPhase.CLOSING
