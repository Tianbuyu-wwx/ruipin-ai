"""多模态感知层（方案 §4）。

本层只做**取景与门控决策**：ROI 裁剪、缩放、变化门控、帧预算硬顶。
语义理解全部上云（VLM），本地不留视觉大模型。

不依赖 numpy/PIL：pHash 以十六进制字符串由调用方注入，
所有门控逻辑都是纯函数/纯计数，可在无图像库环境下确定性测试。
"""

from .budget import SessionBudget
from .sampling import (
    DEFAULT_CHANGE_THRESHOLD,
    DEFAULT_PER_TURN_CAP,
    DEFAULT_RATE_S,
    DEFAULT_SESSION_CAP,
    MAX_LONG_EDGE,
    ROI_EXPAND,
    FrameSampler,
    expand_roi,
    hamming,
    resize_long_edge,
)

__all__ = [
    "SessionBudget",
    "FrameSampler",
    "hamming",
    "expand_roi",
    "resize_long_edge",
    "DEFAULT_RATE_S",
    "DEFAULT_CHANGE_THRESHOLD",
    "DEFAULT_PER_TURN_CAP",
    "DEFAULT_SESSION_CAP",
    "MAX_LONG_EDGE",
    "ROI_EXPAND",
]
