"""
测试夹具配置
"""

import pytest
import tempfile
import shutil
from pathlib import Path


@pytest.fixture
def temp_dir():
    """提供临时目录"""
    tmp = tempfile.mkdtemp()
    yield Path(tmp)
    shutil.rmtree(tmp)


@pytest.fixture
def sample_question_bank():
    """提供示例题库数据"""
    return {
        "position": "Java后端开发",
        "position_code": "java_backend",
        "questions": [
            {
                "id": "java-001",
                "type": "technical",
                "difficulty": 2,
                "question": "请解释Java中的多线程实现方式",
                "keywords": ["Thread", "Runnable", "ExecutorService", "线程池", "并发"],
                "follow_ups": [
                    "线程池的核心参数有哪些？",
                    "如何优雅地关闭线程池？"
                ]
            },
            {
                "id": "java-002",
                "type": "technical",
                "difficulty": 3,
                "question": "Spring Boot的核心注解有哪些？",
                "keywords": ["@SpringBootApplication", "@Controller", "@Service", "@Repository", "@Autowired"]
            },
            {
                "id": "java-003",
                "type": "project",
                "difficulty": 3,
                "question": "你在项目中如何处理数据库事务？",
                "keywords": ["事务隔离级别", "事务传播行为", "@Transactional", "回滚", "提交"]
            },
            {
                "id": "java-004",
                "type": "behavioral",
                "difficulty": 2,
                "question": "你如何解决团队中的技术分歧？",
                "keywords": ["沟通", "协商", "技术评估", "团队合作", "解决方案"]
            }
        ]
    }


@pytest.fixture
def sample_interview_config():
    """提供示例面试配置"""
    return {
        "position": "Java后端开发",
        "candidate_name": "测试候选人",
        "num_technical": 2,
        "num_project": 1,
        "num_behavioral": 1,
        "enable_follow_up": True,
        "max_follow_up_depth": 2
    }


@pytest.fixture
def mock_deepseek_response():
    """提供Mock的DeepSeek API响应"""
    return {
        "technical": 80,
        "communication": 75,
        "completeness": 70,
        "problem_solving": 78,
        "teamwork": 72,
        "leadership": 68,
        "feedback": "回答良好，技术基础扎实，但可以更深入地讲解底层原理。"
    }
