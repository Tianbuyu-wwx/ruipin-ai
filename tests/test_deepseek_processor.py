"""
DeepSeek处理器测试
"""

import pytest
import json
from unittest.mock import patch, MagicMock, mock_open
from src.deepseek_processor import DeepSeekProcessor, HybridProcessor


class TestDeepSeekProcessor:
    """测试DeepSeek处理器"""

    @patch("src.deepseek_processor.OpenAI")
    def test_init_with_api_key(self, mock_openai_class):
        """测试使用API密钥初始化"""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        processor = DeepSeekProcessor(api_key="sk-test-key")
        assert processor.api_key == "sk-test-key"
        assert processor.client is not None

    @patch("src.deepseek_processor.OpenAI")
    def test_init_without_api_key(self, mock_openai_class):
        """测试无API密钥初始化"""
        mock_openai_class.return_value = None

        processor = DeepSeekProcessor(api_key=None)
        assert processor.client is None
        assert processor.is_available() is False

    @patch("src.deepseek_processor.OpenAI")
    def test_is_available(self, mock_openai_class):
        """测试可用性检查"""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        processor = DeepSeekProcessor(api_key="sk-test")
        assert processor.is_available() is True

    def test_generate_cache_key(self):
        """测试缓存键生成"""
        processor = DeepSeekProcessor(api_key="sk-test")
        key1 = processor._generate_cache_key("回答1", "问题1")
        key2 = processor._generate_cache_key("回答1", "问题1")
        key3 = processor._generate_cache_key("回答2", "问题1")

        assert key1 == key2
        assert key1 != key3
        assert len(key1) == 32

    def test_check_cache_hit(self):
        """测试缓存命中"""
        processor = DeepSeekProcessor(api_key="sk-test")
        processor.evaluation_cache["test_key"] = {"score": 85}

        result = processor._check_cache("test_key")
        assert result == {"score": 85}
        assert processor.cache_hits == 1

    def test_check_cache_miss(self):
        """测试缓存未命中"""
        processor = DeepSeekProcessor(api_key="sk-test")

        result = processor._check_cache("nonexistent")
        assert result is None
        assert processor.cache_misses == 1

    def test_update_cache(self):
        """测试更新缓存"""
        processor = DeepSeekProcessor(api_key="sk-test")
        processor._update_cache("key1", {"score": 90})

        assert "key1" in processor.evaluation_cache
        assert processor.evaluation_cache["key1"]["score"] == 90

    def test_update_cache_overflow(self):
        """测试缓存溢出"""
        processor = DeepSeekProcessor(api_key="sk-test")
        processor.cache_size = 2

        processor._update_cache("key1", {"score": 1})
        processor._update_cache("key2", {"score": 2})
        processor._update_cache("key3", {"score": 3})

        assert len(processor.evaluation_cache) <= 2

    def test_parse_evaluation_response_valid_json(self):
        """测试解析有效JSON响应"""
        processor = DeepSeekProcessor(api_key="sk-test")
        response = json.dumps({
            "technical": 80,
            "communication": 75,
            "completeness": 70,
            "problem_solving": 78,
            "teamwork": 72,
            "leadership": 68,
            "feedback": "测试反馈"
        })

        result = processor._parse_evaluation_response(response)
        assert result["technical"] == 80
        assert result["communication"] == 75
        assert result["score"] == 73.8
        assert result["feedback"] == "测试反馈"

    def test_parse_evaluation_response_json_in_text(self):
        """测试从文本中提取JSON"""
        processor = DeepSeekProcessor(api_key="sk-test")
        response = "一些说明文字\n```json\n{\"technical\": 80, \"communication\": 75, \"completeness\": 70, \"problem_solving\": 78, \"teamwork\": 72, \"leadership\": 68, \"feedback\": \"测试\"}\n```"

        result = processor._parse_evaluation_response(response)
        assert result["technical"] == 80

    def test_parse_evaluation_response_invalid(self):
        """测试解析无效响应"""
        processor = DeepSeekProcessor(api_key="sk-test")
        result = processor._parse_evaluation_response("无效响应")
        assert result["score"] == 60.0
        assert "评估解析失败" in result["feedback"]

    def test_get_default_evaluation(self):
        """测试获取默认评估"""
        processor = DeepSeekProcessor(api_key="sk-test")
        result = processor._get_default_evaluation()

        assert result["score"] == 60.0
        assert result["technical"] == 60
        assert result["communication"] == 60
        assert "评估解析失败" in result["feedback"]

    def test_build_evaluation_prompt(self):
        """测试构建评估提示"""
        processor = DeepSeekProcessor(api_key="sk-test")
        prompt = processor._build_evaluation_prompt(
            answer="测试回答",
            question="测试问题",
            has_audio=True,
            has_video=True
        )

        assert "测试回答" in prompt
        assert "测试问题" in prompt
        assert "视频" in prompt
        assert "音频" in prompt

    def test_get_stats(self):
        """测试获取统计信息"""
        processor = DeepSeekProcessor(api_key="sk-test")
        processor.total_calls = 10
        processor.failed_calls = 2
        processor.cache_hits = 5
        processor.cache_misses = 5
        processor.api_call_times = [1.0, 2.0, 3.0]

        stats = processor.get_stats()
        assert stats["total_calls"] == 10
        assert stats["failed_calls"] == 2
        assert stats["success_rate"] == 80.0
        assert stats["cache_hit_rate"] == 50.0
        assert stats["avg_api_time"] == 2.0

    def test_clear_cache(self):
        """测试清除缓存"""
        processor = DeepSeekProcessor(api_key="sk-test")
        processor.evaluation_cache["key"] = {"score": 80}
        processor.cache_hits = 5
        processor.cache_misses = 3

        processor.clear_cache()
        assert len(processor.evaluation_cache) == 0
        assert processor.cache_hits == 0
        assert processor.cache_misses == 0

    @patch("src.deepseek_processor.OpenAI")
    def test_evaluate_answer_with_cache(self, mock_openai_class):
        """测试带缓存的评估"""
        processor = DeepSeekProcessor(api_key="sk-test")
        processor.evaluation_cache[processor._generate_cache_key("回答", "问题")] = {"score": 85}

        result = processor.evaluate_answer(answer="回答", question="问题")
        assert result["score"] == 85


class TestHybridProcessor:
    """测试混合处理器"""

    @patch("src.deepseek_processor.DeepSeekProcessor")
    def test_init(self, mock_deepseek):
        """测试初始化"""
        mock_deepseek.return_value.is_available.return_value = True

        processor = HybridProcessor(api_key="sk-test", prefer_online=True)
        assert processor.prefer_online is True
        assert processor.deepseek_processor is not None

    @patch("src.deepseek_processor.DeepSeekProcessor")
    def test_evaluate_answer_online_preferred(self, mock_deepseek_class):
        """测试优先使用在线API"""
        mock_deepseek = MagicMock()
        mock_deepseek.is_available.return_value = True
        mock_deepseek.evaluate_answer.return_value = {
            "score": 85,
            "feedback": "在线评估结果"
        }
        mock_deepseek_class.return_value = mock_deepseek

        processor = HybridProcessor(api_key="sk-test", prefer_online=True)
        result = processor.evaluate_answer(answer="测试")

        assert result["score"] == 85
        mock_deepseek.evaluate_answer.assert_called_once()

    @patch("src.deepseek_processor.DeepSeekProcessor")
    def test_evaluate_answer_fallback(self, mock_deepseek_class):
        """测试降级到备用评估"""
        mock_deepseek = MagicMock()
        mock_deepseek.is_available.return_value = True
        mock_deepseek.evaluate_answer.return_value = {
            "feedback": "评估解析失败，使用默认评分",
            "score": 60
        }
        mock_deepseek_class.return_value = mock_deepseek

        processor = HybridProcessor(api_key="sk-test", prefer_online=True)
        result = processor.evaluate_answer(answer="测试")

        assert "score" in result

    @patch("src.deepseek_processor.DeepSeekProcessor")
    def test_fallback_evaluation(self, mock_deepseek_class):
        """测试最终降级评估"""
        mock_deepseek = MagicMock()
        mock_deepseek.is_available.return_value = False
        mock_deepseek_class.return_value = mock_deepseek

        processor = HybridProcessor(api_key="sk-test", prefer_online=True)
        result = processor._fallback_evaluation(answer="测试回答", has_audio=False, has_video=False)

        assert "score" in result
        assert "feedback" in result
        assert "备用评估" in result["feedback"]

    def test_get_stats(self):
        """测试获取统计信息"""
        with patch("src.deepseek_processor.DeepSeekProcessor") as mock_deepseek_class:
            mock_deepseek = MagicMock()
            mock_deepseek.is_available.return_value = True
            mock_deepseek.get_stats.return_value = {"total_calls": 10}
            mock_deepseek_class.return_value = mock_deepseek

            processor = HybridProcessor(api_key="sk-test")
            stats = processor.get_stats()

            assert "prefer_online" in stats
            assert "deepseek_available" in stats
            assert stats["deepseek_available"] is True
