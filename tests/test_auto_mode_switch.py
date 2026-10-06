"""
推理模式自动切换测试
验证硬件检测和自动模式选择功能
"""

import pytest
import torch
from unittest.mock import patch, MagicMock
from src.multimodal_evaluator import MultimodalEvaluator


class TestHardwareDetection:
    """测试硬件检测功能"""

    def test_detect_hardware_with_cuda(self):
        """测试 CUDA 可用时的硬件检测"""
        evaluator = MultimodalEvaluator.__new__(MultimodalEvaluator)
        evaluator.model_path = "base_model"

        info = evaluator._detect_hardware()

        assert "cuda_available" in info
        assert "gpu_count" in info
        assert "gpus" in info
        assert "total_vram_gb" in info
        assert "free_vram_gb" in info

        if torch.cuda.is_available():
            assert info["cuda_available"] is True
            assert info["gpu_count"] > 0
            assert len(info["gpus"]) > 0

            for gpu in info["gpus"]:
                assert "name" in gpu
                assert "total_memory_gb" in gpu
                assert "free_memory_gb" in gpu
                assert "compute_capability" in gpu
                assert gpu["total_memory_gb"] > 0

    def test_detect_hardware_without_cuda(self):
        """测试 CUDA 不可用时的硬件检测"""
        with patch('torch.cuda.is_available', return_value=False):
            evaluator = MultimodalEvaluator.__new__(MultimodalEvaluator)
            evaluator.model_path = "base_model"

            info = evaluator._detect_hardware()

            assert info["cuda_available"] is False
            assert info["gpu_count"] == 0
            assert len(info["gpus"]) == 0
            assert info["total_vram_gb"] == 0
            assert info["free_vram_gb"] == 0


class TestAutoModeSelection:
    """测试自动模式选择功能"""

    def test_select_cpu_only_mode(self):
        """测试选择 CPU 模式"""
        evaluator = MultimodalEvaluator.__new__(MultimodalEvaluator)
        evaluator.model_path = "base_model"
        evaluator.hardware_info = {
            "cuda_available": False,
            "free_vram_gb": 0
        }

        evaluator._auto_select_mode()

        assert evaluator.current_mode == "cpu_only"
        assert evaluator.mode_config["load_in_4bit"] is False
        assert evaluator.mode_config["torch_dtype"] == "float32"

    def test_select_full_4bit_mode(self):
        """测试选择 4-bit 量化模式"""
        evaluator = MultimodalEvaluator.__new__(MultimodalEvaluator)
        evaluator.model_path = "base_model"
        evaluator.hardware_info = {
            "cuda_available": True,
            "free_vram_gb": 8,
            "gpus": [{"name": "RTX 3060", "compute_capability": "8.6"}]
        }

        evaluator._auto_select_mode()

        assert evaluator.current_mode == "full_4bit"
        assert evaluator.mode_config["load_in_4bit"] is True
        assert evaluator.mode_config["max_image_size"] == (448, 448)
        assert evaluator.mode_config["max_answer_len"] == 2000

    def test_select_full_fp16_mode(self):
        """测试选择 FP16 模式"""
        evaluator = MultimodalEvaluator.__new__(MultimodalEvaluator)
        evaluator.model_path = "base_model"
        evaluator.hardware_info = {
            "cuda_available": True,
            "free_vram_gb": 12,
            "gpus": [{"name": "RTX 3090", "compute_capability": "8.6"}]
        }

        evaluator._auto_select_mode()

        assert evaluator.current_mode == "full_fp16"
        assert evaluator.mode_config["load_in_4bit"] is False
        assert evaluator.mode_config["torch_dtype"] == "float16"

    def test_select_cpu_offload_mode(self):
        """测试选择 CPU 卸载模式"""
        evaluator = MultimodalEvaluator.__new__(MultimodalEvaluator)
        evaluator.model_path = "base_model"
        evaluator.hardware_info = {
            "cuda_available": True,
            "free_vram_gb": 5,
            "gpus": [{"name": "GTX 1060", "compute_capability": "6.1"}]
        }

        evaluator._auto_select_mode()

        assert evaluator.current_mode == "cpu_offload"
        assert evaluator.mode_config["load_in_4bit"] is True
        assert evaluator.mode_config["device_map"] == "auto"
        assert evaluator.mode_config["max_image_size"] == (336, 336)

    def test_low_compute_capability_fallback(self):
        """测试低计算能力 GPU 回退到 FP16"""
        evaluator = MultimodalEvaluator.__new__(MultimodalEvaluator)
        evaluator.model_path = "base_model"
        evaluator.hardware_info = {
            "cuda_available": True,
            "free_vram_gb": 8,
            "gpus": [{"name": "GTX 750 Ti", "compute_capability": "5.0"}]
        }

        evaluator._auto_select_mode()

        # 计算能力 5.0 不支持 4-bit 量化，应回退
        assert evaluator.current_mode in ["full_fp16", "cpu_only"]


class TestModeConfiguration:
    """测试模式配置功能"""

    def test_get_hardware_info(self):
        """测试获取硬件信息"""
        evaluator = MultimodalEvaluator.__new__(MultimodalEvaluator)
        evaluator.model_path = "base_model"
        evaluator.hardware_info = {
            "cuda_available": True,
            "gpu_count": 1,
            "gpus": [{"name": "RTX 3060"}],
            "total_vram_gb": 12,
            "free_vram_gb": 8
        }

        info = evaluator.get_hardware_info()

        assert info["gpu_count"] == 1
        assert info["total_vram_gb"] == 12
        assert info["free_vram_gb"] == 8

    def test_get_current_mode(self):
        """测试获取当前模式"""
        evaluator = MultimodalEvaluator.__new__(MultimodalEvaluator)
        evaluator.model_path = "base_model"
        evaluator.current_mode = "full_4bit"
        evaluator.mode_config = MultimodalEvaluator.INFERENCE_MODES["full_4bit"]

        mode_info = evaluator.get_current_mode()

        assert mode_info["mode"] == "full_4bit"
        assert mode_info["description"] == "4-bit 量化模式"
        assert "config" in mode_info


class TestDynamicConfigUsage:
    """测试动态配置在推理中的使用"""

    @patch("src.interview_engine.QuestionBankManager")
    @patch("src.interview_engine.HybridProcessor")
    def test_mode_config_affects_image_size(self, mock_processor, mock_bank):
        """测试模式配置影响图像尺寸限制"""
        mock_bank.return_value.question_banks = {}
        mock_bank.return_value.generate_interview_plan.return_value = {
            "technical": [], "project": [], "behavioral": []
        }
        mock_processor.return_value = None

        evaluator = MultimodalEvaluator()
        if not evaluator.is_available():
            pytest.skip("模型不可用")

        # 验证模式配置已加载
        assert evaluator.mode_config is not None
        assert "max_image_size" in evaluator.mode_config

        # 验证图像尺寸限制生效
        max_size = evaluator.mode_config["max_image_size"]
        assert isinstance(max_size, tuple)
        assert len(max_size) == 2

    @patch("src.interview_engine.QuestionBankManager")
    @patch("src.interview_engine.HybridProcessor")
    def test_mode_config_affects_answer_length(self, mock_processor, mock_bank):
        """测试模式配置影响回答长度限制"""
        mock_bank.return_value.question_banks = {}
        mock_bank.return_value.generate_interview_plan.return_value = {
            "technical": [], "project": [], "behavioral": []
        }
        mock_processor.return_value = None

        evaluator = MultimodalEvaluator()
        if not evaluator.is_available():
            pytest.skip("模型不可用")

        # 验证模式配置已加载
        assert evaluator.mode_config is not None
        assert "max_answer_len" in evaluator.mode_config

        # 验证回答长度限制生效
        max_len = evaluator.mode_config["max_answer_len"]
        assert isinstance(max_len, int)
        assert max_len > 0
