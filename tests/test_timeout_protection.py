"""
超时保护机制测试套件
验证面试总时长限制、单题回答限制和极端边界条件
"""

import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from src.interview_engine import InterviewEngine, InterviewSession, InterviewStatus


class TestTimeoutProtection:
    """超时保护机制测试"""

    def setup_method(self):
        """每个测试前初始化"""
        self.engine = InterviewEngine()
        self.session = self.engine.create_session(
            position="java_backend",
            candidate_name="超时测试候选人"
        )
        # 设置极短的超时时间以便测试
        self.session.max_interview_duration_seconds = 2  # 2秒超时
        self.session.max_answer_wait_seconds = 1  # 1秒等待

    def test_interview_timeout_triggers_end(self):
        """测试面试总时长超时自动结束"""
        # 模拟面试已开始3秒前
        self.session.started_at = (datetime.now() - timedelta(seconds=3)).isoformat()
        self.session.status = InterviewStatus.IN_PROGRESS
        self.engine.session = self.session

        # 处理回答时应触发超时
        result = self.engine.process_answer("测试回答")

        # 验证面试已结束（超时后状态应为COMPLETED）
        assert self.session.status == InterviewStatus.COMPLETED
        # 超时结束的消息可能包含"面试结束"或"超时"
        result_str = str(result)
        assert "面试结束" in result_str or "超时" in result_str or result.get("type") == "closing"

    def test_interview_not_timeout_when_within_limit(self):
        """测试面试未超时时正常处理"""
        # 模拟面试刚开始
        self.session.started_at = datetime.now().isoformat()
        self.session.status = InterviewStatus.IN_PROGRESS
        self.engine.session = self.session

        # 先开始面试
        self.engine.start_interview()

        # 处理回答时不应触发超时
        result = self.engine.process_answer("这是一个正常回答")

        # 验证面试仍在进行中（不是错误返回）
        assert "error" not in result or result.get("error") != "面试已结束"

    def test_extremely_long_answer_handling(self):
        """测试超长回答处理"""
        self.session.started_at = datetime.now().isoformat()
        self.session.status = InterviewStatus.IN_PROGRESS
        self.engine.session = self.session

        # 构造10000字符的超长回答
        long_answer = "A" * 10000

        # 处理超长回答
        result = self.engine.process_answer(long_answer)

        # 验证系统没有崩溃，返回了结果
        assert result is not None
        assert isinstance(result, dict)

    def test_rapid_fire_answers_timeout(self):
        """测试快速连续提交不会误触发超时"""
        self.session.started_at = datetime.now().isoformat()
        self.session.status = InterviewStatus.IN_PROGRESS
        self.engine.session = self.session

        # 先开始面试
        self.engine.start_interview()

        # 快速提交多个回答
        results = []
        for i in range(5):
            result = self.engine.process_answer(f"快速回答{i}")
            results.append(result)
            time.sleep(0.1)

        # 所有请求都应被处理（没有超时）
        assert len(results) == 5
        # 验证没有因为超时而返回错误
        assert not any(r.get("error_code") == "SES_003" for r in results if isinstance(r, dict))

    def test_timeout_with_no_start_time(self):
        """测试无开始时间时的超时检查"""
        self.session.started_at = None
        self.session.status = InterviewStatus.IN_PROGRESS
        self.engine.session = self.session

        # 不应触发超时
        result = self.engine.process_answer("测试回答")

        # 验证正常处理（不是超时结束）
        assert "面试已结束" not in str(result)

    def test_timeout_boundary_exactly_at_limit(self):
        """测试刚好在超时边界的情况"""
        # 先开始面试
        self.engine.start_interview()

        # 设置开始时间为2秒前（刚好在边界）
        self.session.started_at = (datetime.now() - timedelta(seconds=2)).isoformat()
        self.session.status = InterviewStatus.IN_PROGRESS
        self.engine.session = self.session

        result = self.engine.process_answer("边界测试")

        # 刚好在边界，应该触发超时
        assert self.session.status == InterviewStatus.COMPLETED

    def test_timeout_with_invalid_start_time(self):
        """测试无效开始时间格式"""
        self.session.started_at = "invalid-datetime"
        self.session.status = InterviewStatus.IN_PROGRESS
        self.engine.session = self.session

        # 不应崩溃，应正常处理
        result = self.engine.process_answer("测试回答")

        # 验证没有崩溃
        assert result is not None

    def test_session_none_timeout_check(self):
        """测试session为None时的超时检查"""
        self.engine.session = None

        # 不应崩溃
        result = self.engine.process_answer("测试回答")

        assert "error" in result

    def test_extreme_boundary_conditions(self):
        """测试极端边界条件组合"""
        # 设置极端配置
        self.session.max_interview_duration_seconds = 0  # 立即超时
        self.session.max_total_follow_ups = 0  # 不允许追问
        self.session.max_follow_up_depth = 0
        self.session.started_at = datetime.now().isoformat()
        self.session.status = InterviewStatus.IN_PROGRESS
        self.engine.session = self.session

        # 空回答
        result1 = self.engine.process_answer("")
        assert result1 is not None

        # 超长回答
        result2 = self.engine.process_answer("X" * 50000)
        assert result2 is not None

        # 特殊字符
        result3 = self.engine.process_answer("\x00\x01\x02\x03")
        assert result3 is not None

    def test_timeout_preserves_interview_data(self):
        """测试超时后面试数据是否保留"""
        # 先添加一些对话历史
        from src.interview_engine import DialogueTurn
        turn = DialogueTurn(
            turn_id="test-001",
            question_id="q-001",
            interviewer_question="测试问题",
            candidate_answer="测试回答",
            scores={"technical": 80}
        )
        self.session.dialogue_history.append(turn)

        # 设置超时
        self.session.started_at = (datetime.now() - timedelta(seconds=10)).isoformat()
        self.session.status = InterviewStatus.IN_PROGRESS
        self.engine.session = self.session

        # 触发超时
        result = self.engine.process_answer("另一个回答")

        # 验证历史数据保留
        assert len(self.session.dialogue_history) >= 1
        assert self.session.dialogue_history[0].candidate_answer == "测试回答"

    def test_concurrent_timeout_scenario(self):
        """测试并发场景下的超时处理"""
        self.session.started_at = datetime.now().isoformat()
        self.session.status = InterviewStatus.IN_PROGRESS
        self.engine.session = self.session

        # 模拟时间流逝导致超时
        with patch('src.interview_engine.datetime') as mock_datetime:
            # 设置当前时间比开始时间晚10秒
            future_time = datetime.now() + timedelta(seconds=10)
            mock_datetime.now.return_value = future_time
            mock_datetime.fromisoformat = datetime.fromisoformat

            result = self.engine.process_answer("并发测试")

            # 应该触发超时
            assert self.session.status == InterviewStatus.COMPLETED


class TestTimeoutConfiguration:
    """超时配置测试"""

    def test_default_timeout_values(self):
        """测试默认超时配置值"""
        session = InterviewSession(
            session_id="test",
            position="java_backend",
            candidate_name="测试"
        )

        assert session.max_interview_duration_seconds == 3600  # 60分钟
        assert session.max_answer_wait_seconds == 300  # 5分钟
        assert session.max_report_generation_seconds == 30  # 30秒

    def test_custom_timeout_values(self):
        """测试自定义超时配置"""
        session = InterviewSession(
            session_id="test",
            position="java_backend",
            candidate_name="测试",
            max_interview_duration_seconds=1800,  # 30分钟
            max_answer_wait_seconds=600,  # 10分钟
        )

        assert session.max_interview_duration_seconds == 1800
        assert session.max_answer_wait_seconds == 600

    def test_timeout_values_persistence(self):
        """测试超时配置在会话中的持久性"""
        engine = InterviewEngine()
        session = engine.create_session(
            position="java_backend",
            candidate_name="测试"
        )

        # 修改配置
        session.max_interview_duration_seconds = 600

        # 验证配置保持
        assert engine.session.max_interview_duration_seconds == 600


class TestFollowUpLimitIntegration:
    """追问限制与超时集成测试"""

    def setup_method(self):
        self.engine = InterviewEngine()
        self.session = self.engine.create_session(
            position="java_backend",
            candidate_name="追问限制测试"
        )
        self.session.max_total_follow_ups = 3  # 最多3次追问
        self.session.max_follow_up_depth = 1  # 每问题最多1次追问
        self.session.started_at = datetime.now().isoformat()
        self.session.status = InterviewStatus.IN_PROGRESS
        self.engine.session = self.session

    def test_global_follow_up_limit_enforced(self):
        """测试全局追问限制执行"""
        from src.interview_engine import DialogueTurn

        # 添加3个追问回合（达到上限）
        for i in range(3):
            turn = DialogueTurn(
                turn_id=f"follow-up-{i}",
                question_id="q-001",
                interviewer_question=f"追问{i}",
                candidate_answer="回答",
                is_follow_up=True
            )
            self.session.dialogue_history.append(turn)

        # 添加一个原始问题回合
        original_turn = DialogueTurn(
            turn_id="original-001",
            question_id="q-001",
            interviewer_question="原始问题",
            candidate_answer="原始回答"
        )
        self.session.dialogue_history.append(original_turn)

        # 尝试触发追问（应该被拒绝）
        should_follow_up = self.engine._should_follow_up()
        assert should_follow_up is False

    def test_follow_up_limit_with_timeout(self):
        """测试追问限制与超时同时触发"""
        from src.interview_engine import DialogueTurn

        # 设置即将超时
        self.session.started_at = (datetime.now() - timedelta(seconds=3599)).isoformat()
        self.session.max_interview_duration_seconds = 3600  # 1小时

        # 添加追问回合
        for i in range(2):
            turn = DialogueTurn(
                turn_id=f"follow-up-{i}",
                question_id="q-001",
                interviewer_question=f"追问{i}",
                candidate_answer="回答",
                is_follow_up=True
            )
            self.session.dialogue_history.append(turn)

        # 添加原始问题回合
        original_turn = DialogueTurn(
            turn_id="original-001",
            question_id="q-001",
            interviewer_question="原始问题",
            candidate_answer="原始回答"
        )
        self.session.dialogue_history.append(original_turn)

        # 此时既达到追问上限又即将超时
        should_follow_up = self.engine._should_follow_up()
        assert should_follow_up is False

    def test_no_follow_up_when_timeout_imminent(self):
        """测试即将超时时不进行追问"""
        from src.interview_engine import DialogueTurn

        # 设置即将超时（只剩1秒）
        self.session.started_at = (datetime.now() - timedelta(seconds=3599)).isoformat()
        self.session.max_interview_duration_seconds = 3600

        # 添加原始问题回合（未达到追问上限）
        original_turn = DialogueTurn(
            turn_id="original-001",
            question_id="q-001",
            interviewer_question="原始问题",
            candidate_answer="原始回答",
            evaluation={"score": 50}  # 低分，通常会触发追问
        )
        self.session.dialogue_history.append(original_turn)

        # 即将超时，不应追问
        should_follow_up = self.engine._should_follow_up()
        assert should_follow_up is False
