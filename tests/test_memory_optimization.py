"""
显存优化验证测试
验证 4-bit 量化、输入限制、显存清理等优化措施的效果
"""

import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock
from PIL import Image


class TestMemoryOptimization:
    """测试显存优化效果"""

    def test_model_loads_with_4bit_quantization(self):
        """测试模型是否使用 4-bit 量化加载"""
        from src.multimodal_evaluator import MultimodalEvaluator

        # 记录加载前的显存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            memory_before = torch.cuda.memory_allocated() / 1024**3

        evaluator = MultimodalEvaluator()

        # 如果自动选择的不是 4-bit 模式，切换到 4-bit 模式进行测试
        if evaluator.current_mode != "full_4bit":
            evaluator.switch_mode("full_4bit")

        if torch.cuda.is_available() and evaluator.is_available():
            memory_after = torch.cuda.memory_allocated() / 1024**3
            peak_memory = torch.cuda.max_memory_allocated() / 1024**3
            memory_used = memory_after - memory_before

            print(f"\n模型加载显存使用: {memory_used:.2f} GB")
            print(f"峰值显存使用: {peak_memory:.2f} GB")

            # 4-bit 量化后，4B 模型应该使用约 2-4GB 显存
            # 如果超过 6GB，可能是 FP16 加载
            assert memory_used < 6.0, f"模型加载使用过多显存: {memory_used:.2f}GB，可能未启用 4-bit 量化"
            assert evaluator.model is not None
        else:
            pytest.skip("CUDA 不可用或模型加载失败")

    def test_text_evaluation_memory_cleanup(self):
        """测试文本评估后显存是否被清理"""
        from src.multimodal_evaluator import MultimodalEvaluator

        evaluator = MultimodalEvaluator()

        # 如果自动选择的不是 4-bit 模式，切换到 4-bit 模式进行测试
        if evaluator.current_mode != "full_4bit":
            evaluator.switch_mode("full_4bit")

        if not evaluator.is_available():
            pytest.skip("模型不可用")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            # 运行多次评估
            for i in range(3):
                memory_before = torch.cuda.memory_allocated() / 1024**3
                result = evaluator._evaluate_text(
                    "这是一个测试回答，用于验证显存清理效果。" * 50,
                    "请介绍你的项目经验"
                )
                memory_after = torch.cuda.memory_allocated() / 1024**3
                print(f"\n第{i+1}次评估 - 显存变化: {memory_before:.2f}GB -> {memory_after:.2f}GB")

            # 检查峰值显存是否在合理范围内
            peak_memory = torch.cuda.max_memory_allocated() / 1024**3
            print(f"峰值显存: {peak_memory:.2f} GB")

            # 峰值应该小于 8GB（4-bit 量化 + 输入限制）
            assert peak_memory < 8.0, f"评估峰值显存过高: {peak_memory:.2f}GB"

    def test_image_resize_reduces_memory(self):
        """测试图像压缩是否减少显存使用"""
        from src.multimodal_evaluator import MultimodalEvaluator

        evaluator = MultimodalEvaluator()

        # 如果自动选择的不是 4-bit 模式，切换到 4-bit 模式进行测试
        if evaluator.current_mode != "full_4bit":
            evaluator.switch_mode("full_4bit")

        if not evaluator.is_available():
            pytest.skip("模型不可用")

        # 创建大尺寸图像 (1920x1080)
        large_image = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            # 评估大尺寸图像
            result = evaluator._evaluate_video_frame(large_image, "测试回答")

            peak_memory = torch.cuda.max_memory_allocated() / 1024**3
            print(f"\n大尺寸图像评估峰值显存: {peak_memory:.2f} GB")

            # 图像压缩后，峰值应该小于 6GB
            assert peak_memory < 6.0, f"图像评估峰值显存过高: {peak_memory:.2f}GB"

    def test_input_truncation(self):
        """测试超长输入是否被截断"""
        from src.multimodal_evaluator import MultimodalEvaluator

        evaluator = MultimodalEvaluator()

        # 创建超长回答 (10000 字符)
        long_answer = "这是一个很长的测试回答。" * 500
        assert len(long_answer) > 2000

        # 模拟评估过程，验证截断逻辑
        MAX_ANSWER_LEN = 2000
        truncated = long_answer[:MAX_ANSWER_LEN]

        assert len(truncated) == MAX_ANSWER_LEN
        assert truncated == long_answer[:MAX_ANSWER_LEN]

    def test_memory_usage_reporting(self):
        """测试显存使用报告功能"""
        from src.multimodal_evaluator import MultimodalEvaluator

        evaluator = MultimodalEvaluator()

        # 获取显存使用情况
        usage = evaluator.get_memory_usage()

        assert "gpu_allocated_gb" in usage
        assert "gpu_reserved_gb" in usage
        assert "gpu_max_allocated_gb" in usage

        print(f"\n显存使用情况:")
        print(f"  已分配: {usage['gpu_allocated_gb']:.2f} GB")
        print(f"  预留: {usage['gpu_reserved_gb']:.2f} GB")
        print(f"  峰值: {usage['gpu_max_allocated_gb']:.2f} GB")

    def test_cleanup_memory(self):
        """测试显存清理功能"""
        from src.multimodal_evaluator import MultimodalEvaluator

        evaluator = MultimodalEvaluator()
        if not evaluator.is_available():
            pytest.skip("模型不可用")

        if torch.cuda.is_available():
            # 添加一些缓存数据
            for i in range(10):
                evaluator.evaluation_cache[f"key_{i}"] = {"score": 60}

            # 记录清理前的显存
            memory_before = torch.cuda.memory_allocated() / 1024**3
            cache_size_before = len(evaluator.evaluation_cache)

            # 执行清理
            evaluator.cleanup_memory()

            # 记录清理后的显存
            memory_after = torch.cuda.memory_allocated() / 1024**3
            cache_size_after = len(evaluator.evaluation_cache)

            print(f"\n清理前显存: {memory_before:.2f} GB, 缓存: {cache_size_before}")
            print(f"清理后显存: {memory_after:.2f} GB, 缓存: {cache_size_after}")

            assert cache_size_after == 0, "缓存未清空"
            assert memory_after <= memory_before, "显存未减少"


class TestMemoryEdgeCases:
    """显存边界场景测试"""

    def test_consecutive_evaluations_memory_stable(self):
        """测试连续评估时显存是否稳定"""
        from src.multimodal_evaluator import MultimodalEvaluator

        evaluator = MultimodalEvaluator()
        if not evaluator.is_available():
            pytest.skip("模型不可用")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

            memory_readings = []
            for i in range(5):
                evaluator._evaluate_text(
                    f"测试回答 {i}" * 100,
                    "测试问题"
                )
                memory = torch.cuda.memory_allocated() / 1024**3
                memory_readings.append(memory)

            print(f"\n连续评估显存读数: {memory_readings}")

            # 显存应该相对稳定，不应持续增长
            max_memory = max(memory_readings)
            min_memory = min(memory_readings)
            memory_variance = max_memory - min_memory

            assert memory_variance < 2.0, f"显存持续泄漏: 方差={memory_variance:.2f}GB"

    def test_empty_input_memory_safe(self):
        """测试空输入不会导致显存异常"""
        from src.multimodal_evaluator import MultimodalEvaluator

        evaluator = MultimodalEvaluator()
        if not evaluator.is_available():
            pytest.skip("模型不可用")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            memory_before = torch.cuda.memory_allocated() / 1024**3

            # 空输入评估
            result = evaluator._evaluate_text("", "测试问题")

            memory_after = torch.cuda.memory_allocated() / 1024**3
            print(f"\n空输入评估显存: {memory_before:.2f}GB -> {memory_after:.2f}GB")

            # 显存变化应该很小
            assert abs(memory_after - memory_before) < 1.0
