"""
主入口集成测试
验证启动时硬件检测和推理模式自动切换功能
"""

import pytest
from unittest.mock import patch, MagicMock


class TestHardwareCheckIntegration:
    """测试硬件检测集成"""

    def test_perform_hardware_check_with_cuda(self):
        """测试 CUDA 环境下的硬件检测"""
        from app import perform_hardware_check

        result = perform_hardware_check()

        # 验证返回了硬件信息
        assert result is not None
        assert "hardware" in result
        assert "mode" in result

        hardware = result["hardware"]
        assert "cuda_available" in hardware
        assert "gpu_count" in hardware
        assert "gpus" in hardware

        mode = result["mode"]
        assert "mode" in mode
        assert "description" in mode
        assert "config" in mode

    def test_perform_hardware_check_without_multimodal(self):
        """测试多模态模块不可用时的硬件检测"""
        from app import perform_hardware_check

        with patch('app.MULTIMODAL_AVAILABLE', False):
            result = perform_hardware_check()
            assert result is None


class TestInterviewWebAppInitialization:
    """测试 InterviewWebApp 初始化"""

    @patch("app.InterviewEngine")
    @patch("app.perform_hardware_check")
    def test_app_initialization_with_hardware_check(self, mock_hardware_check, mock_engine):
        """测试应用初始化时执行硬件检测"""
        from app import InterviewWebApp

        # Mock 硬件检测结果
        mock_hardware_check.return_value = {
            "hardware": {
                "cuda_available": True,
                "gpu_count": 1,
                "gpus": [{"name": "RTX 3060", "total_memory_gb": 12, "free_memory_gb": 8}],
                "total_vram_gb": 12,
                "free_vram_gb": 8
            },
            "mode": {
                "mode": "full_4bit",
                "description": "4-bit 量化模式",
                "config": {
                    "load_in_4bit": True,
                    "torch_dtype": "float16",
                    "max_image_size": (448, 448),
                    "max_answer_len": 2000
                }
            }
        }

        # Mock 引擎
        mock_engine_instance = MagicMock()
        mock_engine.return_value = mock_engine_instance

        # 创建应用实例
        app = InterviewWebApp()

        # 验证硬件检测被调用
        mock_hardware_check.assert_called_once()

        # 验证硬件信息被保存
        assert app.hardware_info is not None
        assert app.hardware_info["cuda_available"] is True

        # 验证推理模式被保存
        assert app.inference_mode is not None
        assert app.inference_mode["mode"] == "full_4bit"

    @patch("app.InterviewEngine")
    def test_app_get_system_info(self, mock_engine):
        """测试获取系统信息功能"""
        from app import InterviewWebApp

        mock_engine_instance = MagicMock()
        mock_engine.return_value = mock_engine_instance

        app = InterviewWebApp()

        # 设置模拟硬件信息
        app.hardware_info = {
            "cuda_available": True,
            "gpus": [{
                "name": "RTX 3060",
                "total_memory_gb": 12.0,
                "free_memory_gb": 8.0,
                "compute_capability": "8.6"
            }]
        }
        app.inference_mode = {
            "mode": "full_4bit",
            "description": "4-bit 量化模式",
            "config": {
                "load_in_4bit": True,
                "torch_dtype": "float16",
                "max_image_size": (448, 448),
                "max_answer_len": 2000
            }
        }

        # 获取系统信息
        info = app.get_system_info()

        # 验证信息包含关键内容
        assert "系统信息" in info
        assert "RTX 3060" in info
        assert "4-bit 量化模式" in info
        assert "启用" in info
        assert "float16" in info

    @patch("app.InterviewEngine")
    def test_app_get_system_info_no_hardware(self, mock_engine):
        """测试无硬件信息时的系统信息"""
        from app import InterviewWebApp

        mock_engine_instance = MagicMock()
        mock_engine.return_value = mock_engine_instance

        app = InterviewWebApp()

        # 不设置硬件信息
        app.hardware_info = None
        app.inference_mode = None

        # 获取系统信息
        info = app.get_system_info()

        # 验证信息包含默认值
        assert "系统信息" in info
        assert "硬件信息: 未获取" in info
        assert "推理模式: 未配置" in info


class TestMainEntryPoint:
    """测试主入口点"""

    def test_main_entry_hardware_check(self):
        """测试主入口执行硬件检测"""
        from app import perform_hardware_check

        # 验证硬件检测函数存在且可调用
        result = perform_hardware_check()
        # 结果可能为 None（如果多模态不可用）或包含硬件信息
        assert result is None or (isinstance(result, dict) and "hardware" in result and "mode" in result)
