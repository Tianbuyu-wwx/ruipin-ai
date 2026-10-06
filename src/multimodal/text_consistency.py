"""
锐聘AI - 文本一致性校验器

校验API文本处理结果与Whisper转写文本的一致性
"""

import difflib
from typing import Dict, List, Tuple
from dataclasses import dataclass

from ..logger import logger


@dataclass
class ConsistencyResult:
    """一致性校验结果"""
    similarity: float = 0.0
    is_consistent: bool = False
    diff_count: int = 0
    common_words: int = 0
    total_words: int = 0
    mismatched_segments: List[Tuple[str, str]] = None

    def __post_init__(self):
        if self.mismatched_segments is None:
            self.mismatched_segments = []


class TextConsistencyChecker:
    """文本一致性校验器"""

    # 一致性阈值
    CONSISTENCY_THRESHOLDS = {
        "high": 0.80,
        "medium": 0.50,
        "low": 0.30
    }

    def __init__(self):
        logger.info("[OK] 文本一致性校验器初始化完成")

    def check(self, api_text: str, whisper_text: str) -> Dict[str, any]:
        """
        校验两段文本的一致性

        Args:
            api_text: API文本处理结果
            whisper_text: Whisper转写文本

        Returns:
            一致性校验结果
        """
        if not api_text or not whisper_text:
            logger.warning("文本为空，无法校验一致性")
            return {
                "similarity": 0.0,
                "level": "none",
                "is_consistent": False,
                "diff_count": 0,
                "common_words": 0,
                "total_words": 0,
                "mismatched_segments": [],
                "recommendation": "文本缺失，无法校验"
            }

        # 预处理
        api_clean = self._preprocess(api_text)
        whisper_clean = self._preprocess(whisper_text)

        # 计算相似度
        similarity = self._calculate_similarity(api_clean, whisper_clean)

        # 找出差异片段
        mismatches = self._find_mismatches(api_clean, whisper_clean)

        # 统计信息
        api_words = set(api_clean.split())
        whisper_words = set(whisper_clean.split())
        common = len(api_words & whisper_words)
        total = len(api_words | whisper_words)

        # 确定一致性等级
        level = self._get_consistency_level(similarity)
        is_consistent = similarity >= self.CONSISTENCY_THRESHOLDS["medium"]

        result = {
            "similarity": round(similarity, 4),
            "level": level,
            "is_consistent": is_consistent,
            "diff_count": len(mismatches),
            "common_words": common,
            "total_words": total,
            "mismatched_segments": mismatches[:5],  # 最多返回5个差异
            "recommendation": self._get_recommendation(similarity, level)
        }

        logger.info(f"文本一致性校验完成: 相似度={similarity:.2%}, 等级={level}")

        return result

    def _preprocess(self, text: str) -> str:
        """预处理文本"""
        # 去除多余空白
        text = " ".join(text.split())
        # 转为小写
        text = text.lower()
        # 去除标点
        import re
        text = re.sub(r'[^\w\s]', '', text)
        return text.strip()

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """计算两段文本的相似度"""
        # 使用SequenceMatcher计算相似度
        similarity = difflib.SequenceMatcher(None, text1, text2).ratio()

        # 同时计算词级别的Jaccard相似度
        words1 = set(text1.split())
        words2 = set(text2.split())

        if not words1 or not words2:
            return 0.0

        jaccard = len(words1 & words2) / len(words1 | words2)

        # 综合两种相似度
        combined = similarity * 0.6 + jaccard * 0.4

        return combined

    def _find_mismatches(self, text1: str, text2: str) -> List[Tuple[str, str]]:
        """找出文本差异片段"""
        mismatches = []

        # 使用difflib找出差异
        diff = list(difflib.ndiff(text1.split(), text2.split()))

        i = 0
        while i < len(diff):
            line = diff[i]
            if line.startswith('- '):
                # text1中有但text2中没有
                word1 = line[2:]
                if i + 1 < len(diff) and diff[i + 1].startswith('+ '):
                    word2 = diff[i + 1][2:]
                    mismatches.append((word1, word2))
                    i += 2
                else:
                    mismatches.append((word1, ""))
                    i += 1
            elif line.startswith('+ '):
                # text2中有但text1中没有
                word2 = line[2:]
                mismatches.append(("", word2))
                i += 1
            else:
                i += 1

        return mismatches

    def _get_consistency_level(self, similarity: float) -> str:
        """获取一致性等级"""
        if similarity >= self.CONSISTENCY_THRESHOLDS["high"]:
            return "high"
        elif similarity >= self.CONSISTENCY_THRESHOLDS["medium"]:
            return "medium"
        elif similarity >= self.CONSISTENCY_THRESHOLDS["low"]:
            return "low"
        else:
            return "none"

    def _get_recommendation(self, similarity: float, level: str) -> str:
        """获取建议"""
        if level == "high":
            return "文本一致性高，可信任文本评估结果"
        elif level == "medium":
            return "文本一致性中等，建议结合其他模态评估"
        elif level == "low":
            return "文本一致性低，主要依赖声纹和视频评估"
        else:
            return "文本严重不一致，需人工复核"

    def batch_check(self, pairs: List[Tuple[str, str]]) -> List[Dict]:
        """批量校验"""
        results = []
        for api_text, whisper_text in pairs:
            result = self.check(api_text, whisper_text)
            results.append(result)
        return results
