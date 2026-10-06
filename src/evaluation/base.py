"""
锐聘AI - 评估器基类
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class BaseEvaluator(ABC):
    """评估器基类"""
    
    @abstractmethod
    def evaluate_answer(
        self,
        answer: str,
        question: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        评估面试回答
        
        Args:
            answer: 候选人回答
            question: 面试问题
            **kwargs: 额外参数
            
        Returns:
            评估结果字典
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """检查评估器是否可用"""
        pass
    
    def get_default_evaluation(self) -> Dict[str, Any]:
        """获取默认评估结果"""
        return {
            "score": 60.0,
            "technical": 60,
            "communication": 60,
            "completeness": 60,
            "problem_solving": 60,
            "teamwork": 60,
            "leadership": 60,
            "feedback": "默认评估"
        }
