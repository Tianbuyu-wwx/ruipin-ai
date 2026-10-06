"""
锐聘AI - DeepSeek评估器
"""

import os
import json
import hashlib
import time
from typing import Dict, Any, Optional

from openai import OpenAI

from .base import BaseEvaluator
from ..logger import logger


class DeepSeekEvaluator(BaseEvaluator):
    """DeepSeek API评估器"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-chat",
        timeout: float = 10.0
    ):
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        self.client = None
        
        self._init_client()
        self._init_prompts()
    
    def _init_client(self):
        """初始化客户端"""
        try:
            if self.api_key:
                self.client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                    timeout=self.timeout
                )
                logger.info("[OK] DeepSeek评估器初始化成功")
        except Exception as e:
            logger.error(f"[ERROR] DeepSeek评估器初始化失败: {e}")
    
    def _init_prompts(self):
        """初始化提示模板"""
        self.system_prompt = """你是一位专业的技术面试官..."""
        self.evaluation_template = """请评估以下面试回答..."""
    
    def evaluate_answer(
        self,
        answer: str,
        question: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """评估回答"""
        if not self.client:
            return self.get_default_evaluation()
        
        try:
            prompt = self._build_prompt(answer, question)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=800,
                response_format={"type": "json_object"}
            )
            
            return self._parse_response(response.choices[0].message.content)
            
        except Exception as e:
            logger.error(f"DeepSeek评估失败: {e}")
            return self.get_default_evaluation()
    
    def _build_prompt(self, answer: str, question: Optional[str]) -> str:
        """构建提示"""
        return f"问题: {question}\n回答: {answer}"
    
    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """解析响应"""
        try:
            result = json.loads(response_text)
            return {
                "score": result.get("score", 60),
                "technical": result.get("technical", 60),
                "communication": result.get("communication", 60),
                "completeness": result.get("completeness", 60),
                "problem_solving": result.get("problem_solving", 60),
                "teamwork": result.get("teamwork", 60),
                "leadership": result.get("leadership", 60),
                "feedback": result.get("feedback", "")
            }
        except Exception:
            return self.get_default_evaluation()
    
    def is_available(self) -> bool:
        """检查是否可用"""
        return self.client is not None
