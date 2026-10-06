"""
端到端测试：验证模型精度切换后的显存占用是否正常
"""

import pytest
import torch
import gc
from src.multimodal_evaluator import MultimodalEvaluator


class TestEndToEndModeSwitchMemory:
    """端到端测试：验证切换精度后显存是否正常释放"""

    def test_fp16_to_4bit_memory_release(self):
        """测试从 FP16 切换到 4-bit 后显存是否正常释放"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA 不可用")

        # 清理初始显存
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
        initial_memory = torch.cuda.memory_allocated() / 1024**3
        print(f"\n初始显存占用: {initial_memory:.2f} GB")

        # 1. 加载 FP16 模型
        print("\n=== 步骤 1: 加载 FP16 模型 ===")
        evaluator = MultimodalEvaluator()
        fp16_memory = torch.cuda.memory_allocated() / 1024**3
        print(f"FP16 模型加载后显存: {fp16_memory:.2f} GB")
        assert evaluator.current_mode == "full_fp16", f"期望 FP16 模式，实际是 {evaluator.current_mode}"

        # 2. 执行一次推理
        print("\n=== 步骤 2: FP16 模式下执行推理 ===")
        result = evaluator._evaluate_text(
            "这是一个测试回答，用于验证显存清理效果。" * 20,
            "请介绍你的项目经验"
        )
        assert result is not None
        fp16_after_inference = torch.cuda.memory_allocated() / 1024**3
        print(f"FP16 推理后显存: {fp16_after_inference:.2f} GB")

        # 3. 切换到 4-bit 模式
        print("\n=== 步骤 3: 切换到 4-bit 模式 ===")
        evaluator.switch_mode("full_4bit")
        torch.cuda.synchronize()
        gc.collect()

        # 验证模式已切换
        assert evaluator.current_mode == "full_4bit", f"期望 4-bit 模式，实际是 {evaluator.current_mode}"

        # 验证显存已释放（应该显著降低）
        memory_after_switch = torch.cuda.memory_allocated() / 1024**3
        print(f"切换到 4-bit 后显存: {memory_after_switch:.2f} GB")

        # 显存应该显著降低（从 ~8GB 降到 ~3GB）
        memory_reduction = fp16_memory - memory_after_switch
        print(f"显存减少: {memory_reduction:.2f} GB")
        assert memory_reduction > 4.0, f"显存释放不足: 只减少了 {memory_reduction:.2f} GB，期望至少减少 4GB"

        # 4. 在 4-bit 模式下执行推理
        print("\n=== 步骤 4: 4-bit 模式下执行推理 ===")
        result = evaluator._evaluate_text(
            "这是另一个测试回答，用于验证 4-bit 模式下的推理效果。" * 20,
            "请描述你遇到的最大挑战"
        )
        assert result is not None
        bit4_after_inference = torch.cuda.memory_allocated() / 1024**3
        print(f"4-bit 推理后显存: {bit4_after_inference:.2f} GB")

        # 5. 切换回 FP16 模式
        print("\n=== 步骤 5: 切换回 FP16 模式 ===")
        evaluator.switch_mode("full_fp16")
        torch.cuda.synchronize()
        gc.collect()

        assert evaluator.current_mode == "full_fp16", f"期望 FP16 模式，实际是 {evaluator.current_mode}"

        memory_back_to_fp16 = torch.cuda.memory_allocated() / 1024**3
        print(f"切换回 FP16 后显存: {memory_back_to_fp16:.2f} GB")

        # 6. 清理
        print("\n=== 步骤 6: 清理所有资源 ===")
        del evaluator
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        final_memory = torch.cuda.memory_allocated() / 1024**3
        print(f"最终显存占用: {final_memory:.2f} GB")

        # 验证最终显存接近初始值
        memory_leak = final_memory - initial_memory
        print(f"显存泄漏: {memory_leak:.2f} GB")
        assert memory_leak < 1.0, f"检测到显存泄漏: {memory_leak:.2f} GB"

        print("\n=== 测试通过！显存切换正常 ===")

    def test_multiple_mode_switches_memory_stable(self):
        """测试多次切换模式后显存是否稳定"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA 不可用")

        # 清理初始显存
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
        initial_memory = torch.cuda.memory_allocated() / 1024**3
        print(f"\n初始显存占用: {initial_memory:.2f} GB")

        evaluator = MultimodalEvaluator()
        base_memory = torch.cuda.memory_allocated() / 1024**3
        print(f"初始模型加载后显存: {base_memory:.2f} GB")

        # 多次切换模式
        modes_to_test = ["full_4bit", "full_fp16", "full_4bit", "full_fp16"]
        memory_readings = []

        for i, mode in enumerate(modes_to_test):
            print(f"\n=== 切换轮次 {i+1}: -> {mode} ===")
            evaluator.switch_mode(mode)
            torch.cuda.synchronize()
            gc.collect()

            current_memory = torch.cuda.memory_allocated() / 1024**3
            memory_readings.append((mode, current_memory))
            print(f"当前显存: {current_memory:.2f} GB")

        # 验证显存稳定
        print("\n=== 显存使用汇总 ===")
        fp16_readings = [m for mode, m in memory_readings if mode == "full_fp16"]
        bit4_readings = [m for mode, m in memory_readings if mode == "full_4bit"]

        if len(fp16_readings) >= 2:
            fp16_variance = max(fp16_readings) - min(fp16_readings)
            print(f"FP16 模式显存波动: {fp16_variance:.2f} GB")
            assert fp16_variance < 1.0, f"FP16 模式显存不稳定: 波动 {fp16_variance:.2f} GB"

        if len(bit4_readings) >= 2:
            bit4_variance = max(bit4_readings) - min(bit4_readings)
            print(f"4-bit 模式显存波动: {bit4_variance:.2f} GB")
            assert bit4_variance < 1.0, f"4-bit 模式显存不稳定: 波动 {bit4_variance:.2f} GB"

        # 清理
        del evaluator
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        final_memory = torch.cuda.memory_allocated() / 1024**3
        memory_leak = final_memory - initial_memory
        print(f"\n最终显存泄漏: {memory_leak:.2f} GB")
        assert memory_leak < 1.0, f"检测到显存泄漏: {memory_leak:.2f} GB"

        print("\n=== 测试通过！多次切换后显存稳定 ===")

    def test_degradation_triggers_mode_switch(self):
        """测试降级机制触发时是否正确切换模型模式"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA 不可用")

        from src.interview_engine import InterviewEngine
        from unittest.mock import patch, MagicMock

        # 创建模拟配置
        mock_config = MagicMock()
        mock_config.phases = []
        mock_config.questions_per_phase = 3
        mock_config.time_limit_minutes = 30
        mock_config.enable_voice_analysis = True
        mock_config.enable_follow_up = True
        mock_config.follow_up_threshold = 0.6
        mock_config.follow_up_max_count = 2
        mock_config.enable_report = True
        mock_config.report_template = "default"

        # Mock 数据库
        mock_db = MagicMock()
        mock_db.get_interview_record.return_value = None

        # Mock 问题库
        with patch("src.interview_engine.QuestionBankManager") as mock_bank, \
             patch("src.interview_engine.HybridProcessor") as mock_processor, \
             patch("src.interview_engine.db_manager", mock_db):

            mock_bank_instance = MagicMock()
            mock_bank_instance.get_questions_for_phase.return_value = [
                {"id": "q1", "content": "测试问题1", "type": "technical", "difficulty": 3}
            ]
            mock_bank.return_value = mock_bank_instance

            mock_processor_instance = MagicMock()
            mock_processor_instance.evaluate_text.return_value = {
                "score": 75,
                "feedback": "测试反馈",
                "dimensions": {"technical": 80}
            }
            mock_processor.return_value = mock_processor_instance

            # 创建面试引擎
            engine = InterviewEngine(mock_config)

            # 验证初始模式
            initial_mode = engine.multimodal_evaluator.current_mode
            print(f"\n初始推理模式: {initial_mode}")

            # 模拟降级到 Level 3
            print("\n=== 模拟降级到 Level 3 ===")
            engine.degradation_controller.force_level(3, reason="显存压力测试")

            # 给降级回调一点时间执行
            import time
            time.sleep(0.5)

            # 验证模式是否切换
            current_mode = engine.multimodal_evaluator.current_mode
            print(f"降级后推理模式: {current_mode}")

            # 如果初始是 FP16，降级后应该是 4-bit
            if initial_mode == "full_fp16":
                assert current_mode == "full_4bit", f"降级后应切换到 4-bit 模式，实际是 {current_mode}"
                print("[OK] 降级正确触发了模型模式切换")

            # 清理
            del engine
            gc.collect()
            torch.cuda.empty_cache()

            print("\n=== 测试通过！降级机制正确触发模式切换 ===")
