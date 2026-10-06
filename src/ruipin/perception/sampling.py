"""帧采样与变化门控（方案 §4.2）—— 成本控制的核心。

参数（方案 §4.2 表）：
- 采样率 1 帧 / 4 s（答题期）
- 变化门控：与上一帧 pHash 汉明距离 <6 则丢弃（候选人静止可省 40–60%）
- 每轮上限 12 帧、每场上限 150 帧（硬闸）
- ROI 裁剪：人脸框外扩 1.4×
- 分辨率：最长边 ≤512 px

**不依赖真实图像库**：pHash 由上游（客户端 WASM / 本地检测器）以十六进制
字符串注入，本模块只做门控决策 + 计数，因此完全可测。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

DEFAULT_RATE_S = 4.0
DEFAULT_CHANGE_THRESHOLD = 6
DEFAULT_PER_TURN_CAP = 12
DEFAULT_SESSION_CAP = 150

MAX_LONG_EDGE = 512
ROI_EXPAND = 1.4

# 半字节（0–15）的 popcount 表
_POPCOUNT = tuple(bin(i).count("1") for i in range(16))


def hamming(a: str, b: str) -> int:
    """两个十六进制 pHash 字符串的汉明距离（不同比特数）。

    逐字符比较而非整体转 int，避免超长 pHash 的平台差异；
    长度不一致直接报错（说明上下游 hash 方案不一致，必须显式修复而非静默截断）。
    """
    if len(a) != len(b):
        raise ValueError(f"pHash 长度不一致: {len(a)} vs {len(b)}")
    dist = 0
    for ca, cb in zip(a, b):
        dist += _POPCOUNT[int(ca, 16) ^ int(cb, 16)]
    return dist


def expand_roi(
    box: tuple[int, int, int, int],
    scale: float = ROI_EXPAND,
    bounds: Optional[tuple[int, int]] = None,
) -> tuple[int, int, int, int]:
    """人脸框外扩（默认 1.4×），以中心为基准，越界裁剪到画面内。

    box/bounds 均为 (x, y, w, h) / (W, H)。返回整数框。
    """
    if scale < 1.0:
        raise ValueError(f"外扩系数必须 ≥1.0，收到 {scale}")
    x, y, w, h = box
    if w <= 0 or h <= 0:
        raise ValueError(f"ROI 尺寸非法: {box}")
    cx, cy = x + w / 2.0, y + h / 2.0
    nw, nh = w * scale, h * scale
    nx, ny = cx - nw / 2.0, cy - nh / 2.0
    if bounds is not None:
        bw, bh = bounds
        nx = max(0.0, min(nx, float(bw)))
        ny = max(0.0, min(ny, float(bh)))
        nw = max(1.0, min(nw, float(bw) - nx))
        nh = max(1.0, min(nh, float(bh) - ny))
    return (int(round(nx)), int(round(ny)), int(round(nw)), int(round(nh)))


def resize_long_edge(
    width: int, height: int, max_edge: int = MAX_LONG_EDGE
) -> tuple[int, int]:
    """把最长边压到 ≤max_edge，保持宽高比；不放大。"""
    if width <= 0 or height <= 0:
        raise ValueError(f"尺寸非法: {width}x{height}")
    if max_edge <= 0:
        raise ValueError(f"max_edge 非法: {max_edge}")
    longest = max(width, height)
    if longest <= max_edge:
        return (width, height)
    ratio = max_edge / float(longest)
    return (max(1, int(round(width * ratio))), max(1, int(round(height * ratio))))


@dataclass
class FrameSampler:
    """帧采样门控器。

    三个闸门依次判定，任一不通过即丢弃并计数：
    ① 采样率（距上次采样 ≥rate_s） ② 变化门控（汉明距离 ≥threshold）
    ③ 上限（每轮 per_turn_cap / 每场 session_cap）。

    计数即成本证据：sampled 是真正要付费上传的帧。
    """

    rate_s: float = DEFAULT_RATE_S
    change_threshold: int = DEFAULT_CHANGE_THRESHOLD
    per_turn_cap: int = DEFAULT_PER_TURN_CAP
    session_cap: int = DEFAULT_SESSION_CAP

    sampled: int = 0
    turn_sampled: int = 0
    dropped_by_rate: int = 0
    dropped_by_change: int = 0
    dropped_by_cap: int = 0

    # ---- 上限 ----

    @property
    def turn_exhausted(self) -> bool:
        return self.turn_sampled >= self.per_turn_cap

    @property
    def exhausted(self) -> bool:
        """每场上限打满（触发即关门并标注）。"""
        return self.sampled >= self.session_cap

    @property
    def turn_remaining(self) -> int:
        return max(0, self.per_turn_cap - self.turn_sampled)

    @property
    def remaining(self) -> int:
        return max(0, self.session_cap - self.sampled)

    # ---- 门控 ----

    def should_sample(
        self,
        t_s: float,
        last_sent_t: Optional[float],
        phash: str,
        last_sent_phash: Optional[str],
    ) -> bool:
        """是否应当采样这一帧。

        last_sent_t / last_sent_phash 为 None 表示本轮尚无已发送帧，
        此时不做速率与变化门控（首帧必采，保证每轮至少有一帧证据）。
        """
        if self.exhausted or self.turn_exhausted:
            self.dropped_by_cap += 1
            return False
        if last_sent_t is not None and (t_s - last_sent_t) < self.rate_s:
            self.dropped_by_rate += 1
            return False
        if last_sent_phash is not None:
            if hamming(phash, last_sent_phash) < self.change_threshold:
                self.dropped_by_change += 1
                return False
        return True

    def record_sent(self, n: int = 1) -> None:
        """确认 n 帧已打包上传。"""
        if n < 0:
            raise ValueError(f"n 不能为负: {n}")
        self.sampled += n
        self.turn_sampled += n

    def begin_turn(self) -> None:
        """开始新一轮：只重置轮内配额，场次配额与统计累计。"""
        self.turn_sampled = 0

    def reset(self) -> None:
        """整场重置（一般只在新建会话时调用）。"""
        self.sampled = 0
        self.turn_sampled = 0
        self.dropped_by_rate = 0
        self.dropped_by_change = 0
        self.dropped_by_cap = 0

    def stats(self) -> dict[str, int]:
        return {
            "sampled": self.sampled,
            "turn_sampled": self.turn_sampled,
            "dropped_by_rate": self.dropped_by_rate,
            "dropped_by_change": self.dropped_by_change,
            "dropped_by_cap": self.dropped_by_cap,
        }
