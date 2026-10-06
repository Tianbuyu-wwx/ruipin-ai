"""
面试系统配置文件
"""

import os
from pathlib import Path

# 项目根目录
BASE_DIR = Path(__file__).parent.parent

# 面试系统配置
INTERVIEW_CONFIG = {
    "default_position": "Java后端开发",
    "default_num_technical": 3,
    "default_num_project": 1,
    "default_num_behavioral": 1,
    "enable_follow_up": True,
    "max_follow_up_depth": 2,
}

# 题库路径
QUESTION_BANK_PATH = os.environ.get(
    "QUESTION_BANK_PATH",
    BASE_DIR / "knowledge_base" / "questions"
)

# 报告输出路径
REPORTS_PATH = os.environ.get(
    "REPORTS_PATH",
    BASE_DIR / "reports"
)

# Gradio配置
GRADIO_CONFIG = {
    "server_name": "0.0.0.0",
    "server_port": 7872,
    "share": False,
    "show_error": True,
}

# 多模态模型配置
MULTIMODAL_CONFIG = {
    "enable_video": True,
    "enable_audio": True,
    "video_model_path": os.environ.get("VIDEO_MODEL_PATH", ""),
    "audio_model_path": os.environ.get("AUDIO_MODEL_PATH", ""),
    "qwen3vl_model_path": os.environ.get("QWEN3VL_MODEL_PATH", "models/qwen3-vl-4B.pth"),
}

# 评估模型配置 - 支持本地模型和DeepSeek API
EVALUATION_CONFIG = {
    # 评估提供商: "deepseek" | "local" | "hybrid" | "multimodal"
    # "multimodal" - 优先使用本地Qwen3-VL进行多模态评估（文本+视频）
    "provider": os.environ.get("EVALUATION_PROVIDER", "deepseek"),
    
    # DeepSeek API配置
    "deepseek": {
        "api_key": os.environ.get("DEEPSEEK_API_KEY", ""),
        "base_url": os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        "model": os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),  # 或 "deepseek-reasoner"
        "timeout": float(os.environ.get("DEEPSEEK_TIMEOUT", "10.0")),
        "max_retries": int(os.environ.get("DEEPSEEK_MAX_RETRIES", "3")),
    },
    
    # 本地模型配置
    "local": {
        "model_path": os.environ.get("LOCAL_MODEL_PATH", "models/qwen3-vl-4B.pth"),
        "enable": True,
    },
    
    # 混合模式配置
    "hybrid": {
        "prefer_online": os.environ.get("HYBRID_PREFER_ONLINE", "true").lower() == "true",
        "fallback_on_failure": True,
    },

    # 多模态评估配置
    "multimodal": {
        "enable_visual_eval": True,  # 启用视觉评估（表情、眼神、肢体语言）
        "visual_weight": 0.3,  # 视觉维度在综合得分中的权重
        "text_weight": 0.7,  # 文本维度在综合得分中的权重
        "model_path": os.environ.get("MULTIMODAL_MODEL_PATH", "base_model"),
    },
    
    # 缓存配置
    "cache": {
        "enabled": True,
        "size": 1000,
    },
    
    # 降级配置
    "fallback": {
        "enabled": True,  # API失败时是否使用备用评估
        "timeout_seconds": 3.0,  # API超时时间
    },
}
