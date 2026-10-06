"""
锐聘AI - 全面模拟数据生成器
覆盖正常场景、异常场景、边界值场景及特殊场景
"""

import uuid
import random
import string
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field


@dataclass
class TestScenario:
    """测试场景定义"""
    name: str
    category: str  # normal, abnormal, boundary, special
    description: str
    data: Dict[str, Any]
    expected_result: str
    expected_exception: Optional[type] = None


class MockDataGenerator:
    """模拟数据生成器"""

    # 正常岗位类型
    NORMAL_POSITIONS = [
        "java_backend",
        "python_backend",
        "web_frontend",
        "algorithm_engineer",
        "data_engineer",
        "devops_engineer",
        "mobile_ios",
        "mobile_android",
        "fullstack",
    ]

    # 经验等级
    EXPERIENCE_LEVELS = ["junior", "mid", "senior", "expert"]

    # 正常候选人姓名
    NORMAL_NAMES = [
        "张三", "李四", "王五", "赵六", "陈七",
        "Alice Zhang", "Bob Li", "Charlie Wang",
        "Emma Liu", "David Chen", "Sophie Wu"
    ]

    # 正常技术回答样本
    NORMAL_ANSWERS = [
        "我使用Spring Boot和MySQL开发了电商后台系统，实现了订单管理、库存管理和用户认证功能。",
        "在项目中我负责设计RESTful API，使用Redis做缓存，RabbitMQ处理异步消息。",
        "我熟悉Java并发编程，使用过线程池、CountDownLatch和CompletableFuture。",
        "前端使用React和TypeScript，配合Redux进行状态管理，实现了组件化开发。",
        "我使用Python和TensorFlow训练了图像分类模型，准确率达到了95%。",
    ]

    # 技术关键词
    TECH_KEYWORDS = [
        "Java", "Python", "Spring", "MySQL", "Redis", "Docker", "Kubernetes",
        "React", "Vue", "Node.js", "TensorFlow", "PyTorch", "Elasticsearch",
        "Kafka", "RabbitMQ", "MongoDB", "PostgreSQL", "Git", "Jenkins"
    ]

    @classmethod
    def generate_normal_scenarios(cls) -> List[TestScenario]:
        """生成正常场景数据"""
        scenarios = []

        # 场景1: 标准Java后端面试
        scenarios.append(TestScenario(
            name="标准Java后端面试",
            category="normal",
            description="完整的Java后端开发面试流程",
            data={
                "candidate_name": "张三",
                "position": "java_backend",
                "experience": "senior",
                "answers": [
                    {
                        "question": "请介绍你的技术栈",
                        "answer": "我使用Spring Boot和MySQL开发了电商后台系统，实现了订单管理、库存管理和用户认证功能。",
                        "has_audio": True,
                        "has_video": False,
                    },
                    {
                        "question": "如何处理高并发场景？",
                        "answer": "使用Redis缓存热点数据，数据库读写分离，引入消息队列削峰填谷，必要时使用分布式锁。",
                        "has_audio": True,
                        "has_video": False,
                    },
                ],
                "config": {
                    "num_technical": 5,
                    "num_project": 2,
                    "num_behavioral": 2,
                    "enable_follow_up": True,
                }
            },
            expected_result="success"
        ))

        # 场景2: 算法工程师面试
        scenarios.append(TestScenario(
            name="算法工程师面试",
            category="normal",
            description="算法岗位面试，包含ML项目经验",
            data={
                "candidate_name": "Alice Zhang",
                "position": "algorithm_engineer",
                "experience": "mid",
                "answers": [
                    {
                        "question": "介绍一个你参与的机器学习项目",
                        "answer": "我使用Python和TensorFlow训练了图像分类模型，使用ResNet50作为骨干网络，准确率达到了95%。",
                        "has_audio": False,
                        "has_video": False,
                    },
                ],
                "config": {
                    "num_technical": 6,
                    "num_project": 1,
                    "num_behavioral": 2,
                    "enable_follow_up": True,
                }
            },
            expected_result="success"
        ))

        # 场景3: 前端工程师面试（含视频）
        scenarios.append(TestScenario(
            name="前端工程师面试含视频",
            category="normal",
            description="前端岗位，启用视频分析",
            data={
                "candidate_name": "李四",
                "position": "web_frontend",
                "experience": "junior",
                "answers": [
                    {
                        "question": "请介绍Vue.js的生命周期",
                        "answer": "Vue组件有beforeCreate、created、beforeMount、mounted、beforeUpdate、updated、beforeDestroy、destroyed等生命周期钩子。",
                        "has_audio": True,
                        "has_video": True,
                    },
                ],
                "config": {
                    "num_technical": 4,
                    "num_project": 2,
                    "num_behavioral": 2,
                    "enable_follow_up": False,
                }
            },
            expected_result="success"
        ))

        # 场景4: 资深专家面试
        scenarios.append(TestScenario(
            name="资深专家面试",
            category="normal",
            description="专家级别，全面评估",
            data={
                "candidate_name": "王五",
                "position": "devops_engineer",
                "experience": "expert",
                "answers": [
                    {
                        "question": "如何设计一个高可用微服务架构？",
                        "answer": "采用Kubernetes进行容器编排，Istio做服务网格，Prometheus+Grafana监控，ELK收集日志，多活部署保证可用性。",
                        "has_audio": True,
                        "has_video": True,
                    },
                ],
                "config": {
                    "num_technical": 5,
                    "num_project": 3,
                    "num_behavioral": 3,
                    "enable_follow_up": True,
                }
            },
            expected_result="success"
        ))

        return scenarios

    @classmethod
    def generate_abnormal_scenarios(cls) -> List[TestScenario]:
        """生成异常场景数据"""
        scenarios = []

        # 场景1: XSS攻击输入
        scenarios.append(TestScenario(
            name="XSS攻击输入",
            category="abnormal",
            description="候选人回答包含XSS脚本",
            data={
                "candidate_name": "张三",
                "position": "java_backend",
                "experience": "mid",
                "answers": [
                    {
                        "question": "自我介绍",
                        "answer": "<script>alert('xss')</script>我喜欢编程",
                        "has_audio": False,
                        "has_video": False,
                    },
                ],
                "config": {"num_technical": 3, "num_project": 1, "num_behavioral": 1}
            },
            expected_result="sanitized"
        ))

        # 场景2: SQL注入攻击
        scenarios.append(TestScenario(
            name="SQL注入攻击",
            category="abnormal",
            description="回答中包含SQL注入语句",
            data={
                "candidate_name": "李四",
                "position": "python_backend",
                "experience": "junior",
                "answers": [
                    {
                        "question": "如何查询数据库？",
                        "answer": "SELECT * FROM users WHERE id = 1; DROP TABLE users; --",
                        "has_audio": False,
                        "has_video": False,
                    },
                ],
                "config": {"num_technical": 3, "num_project": 1, "num_behavioral": 1}
            },
            expected_result="sanitized"
        ))

        # 场景3: 空回答
        scenarios.append(TestScenario(
            name="完全空回答",
            category="abnormal",
            description="候选人未作任何回答",
            data={
                "candidate_name": "王五",
                "position": "web_frontend",
                "experience": "senior",
                "answers": [
                    {
                        "question": "请介绍你的项目经验",
                        "answer": "",
                        "has_audio": False,
                        "has_video": False,
                    },
                ],
                "config": {"num_technical": 3, "num_project": 1, "num_behavioral": 1}
            },
            expected_result="low_score"
        ))

        # 场景4: 乱码输入
        scenarios.append(TestScenario(
            name="乱码和特殊字符输入",
            category="abnormal",
            description="回答包含乱码和控制字符",
            data={
                "candidate_name": "赵六",
                "position": "algorithm_engineer",
                "experience": "mid",
                "answers": [
                    {
                        "question": "介绍你的技术背景",
                        "answer": "我使用\x00\x01\x02Python\x7f开发\xff\xfe",
                        "has_audio": False,
                        "has_video": False,
                    },
                ],
                "config": {"num_technical": 3, "num_project": 1, "num_behavioral": 1}
            },
            expected_result="sanitized"
        ))

        # 场景5: 不存在的岗位
        scenarios.append(TestScenario(
            name="不存在的岗位类型",
            category="abnormal",
            description="使用系统中未定义的岗位",
            data={
                "candidate_name": "陈七",
                "position": "blockchain_developer",
                "experience": "senior",
                "answers": [
                    {
                        "question": "自我介绍",
                        "answer": "我是区块链开发工程师",
                        "has_audio": False,
                        "has_video": False,
                    },
                ],
                "config": {"num_technical": 3, "num_project": 1, "num_behavioral": 1}
            },
            expected_result="fallback"
        ))

        return scenarios

    @classmethod
    def generate_boundary_scenarios(cls) -> List[TestScenario]:
        """生成边界值场景数据"""
        scenarios = []

        # 场景1: 最大长度回答
        scenarios.append(TestScenario(
            name="最大长度回答",
            category="boundary",
            description="回答长度刚好达到最大限制",
            data={
                "candidate_name": "张三",
                "position": "java_backend",
                "experience": "mid",
                "answers": [
                    {
                        "question": "详细描述你的项目",
                        "answer": "A" * 10000,  # 刚好达到最大长度
                        "has_audio": False,
                        "has_video": False,
                    },
                ],
                "config": {"num_technical": 1, "num_project": 1, "num_behavioral": 1}
            },
            expected_result="success"
        ))

        # 场景2: 超长姓名
        scenarios.append(TestScenario(
            name="超长候选人姓名",
            category="boundary",
            description="姓名长度超过限制",
            data={
                "candidate_name": "张" * 100,
                "position": "python_backend",
                "experience": "junior",
                "answers": [
                    {
                        "question": "自我介绍",
                        "answer": "我是Python开发工程师",
                        "has_audio": False,
                        "has_video": False,
                    },
                ],
                "config": {"num_technical": 1, "num_project": 1, "num_behavioral": 1}
            },
            expected_result="truncated"
        ))

        # 场景3: 最小配置面试
        scenarios.append(TestScenario(
            name="最小配置面试",
            category="boundary",
            description="最少问题数量配置",
            data={
                "candidate_name": "李四",
                "position": "web_frontend",
                "experience": "junior",
                "answers": [],
                "config": {
                    "num_technical": 0,
                    "num_project": 0,
                    "num_behavioral": 0,
                    "enable_follow_up": False,
                }
            },
            expected_result="minimal"
        ))

        # 场景4: 最大配置面试
        scenarios.append(TestScenario(
            name="最大配置面试",
            category="boundary",
            description="最多问题数量配置",
            data={
                "candidate_name": "王五",
                "position": "java_backend",
                "experience": "expert",
                "answers": [
                    {
                        "question": f"技术问题{i}",
                        "answer": f"技术回答{i} " + "Java Spring MySQL Redis ",
                        "has_audio": False,
                        "has_video": False,
                    }
                    for i in range(20)
                ],
                "config": {
                    "num_technical": 10,
                    "num_project": 5,
                    "num_behavioral": 5,
                    "enable_follow_up": True,
                    "max_follow_up_depth": 5,
                }
            },
            expected_result="success"
        ))

        # 场景5: 分数边界值
        scenarios.append(TestScenario(
            name="分数边界值",
            category="boundary",
            description="各种分数边界情况",
            data={
                "candidate_name": "赵六",
                "position": "data_engineer",
                "experience": "mid",
                "answers": [
                    {
                        "question": "评分测试",
                        "answer": "测试回答",
                        "has_audio": False,
                        "has_video": False,
                        "forced_scores": {
                            "exactly_60": 60.0,  # 及格线
                            "exactly_70": 70.0,  # 中等线
                            "exactly_85": 85.0,  # 良好线
                            "exactly_100": 100.0,  # 满分
                            "exactly_0": 0.0,  # 零分
                        }
                    },
                ],
                "config": {"num_technical": 1, "num_project": 1, "num_behavioral": 1}
            },
            expected_result="success"
        ))

        return scenarios

    @classmethod
    def generate_special_scenarios(cls) -> List[TestScenario]:
        """生成特殊场景数据"""
        scenarios = []

        # 场景1: 纯英文面试
        scenarios.append(TestScenario(
            name="纯英文面试",
            category="special",
            description="全英文回答",
            data={
                "candidate_name": "David Chen",
                "position": "fullstack",
                "experience": "senior",
                "answers": [
                    {
                        "question": "Tell me about yourself",
                        "answer": "I am a full-stack developer with 8 years of experience. I specialize in React, Node.js, and cloud infrastructure. I have built scalable systems serving millions of users.",
                        "has_audio": True,
                        "has_video": False,
                    },
                ],
                "config": {"num_technical": 3, "num_project": 1, "num_behavioral": 1}
            },
            expected_result="success"
        ))

        # 场景2: 混合中英文
        scenarios.append(TestScenario(
            name="混合中英文回答",
            category="special",
            description="回答中混合使用中英文",
            data={
                "candidate_name": "Sophie Wu",
                "position": "mobile_ios",
                "experience": "mid",
                "answers": [
                    {
                        "question": "介绍你的iOS开发经验",
                        "answer": "我使用Swift和SwiftUI开发了多个iOS app，包括一个social media应用。使用了MVVM架构，Combine处理异步，Core Data做本地存储。",
                        "has_audio": True,
                        "has_video": False,
                    },
                ],
                "config": {"num_technical": 3, "num_project": 1, "num_behavioral": 1}
            },
            expected_result="success"
        ))

        # 场景3: 包含代码块的回答
        scenarios.append(TestScenario(
            name="包含代码块的回答",
            category="special",
            description="回答中包含代码示例",
            data={
                "candidate_name": "Charlie Wang",
                "position": "python_backend",
                "experience": "senior",
                "answers": [
                    {
                        "question": "写一个单例模式",
                        "answer": """```python
class Singleton:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
```
这是线程安全的单例模式实现。""",
                        "has_audio": False,
                        "has_video": False,
                    },
                ],
                "config": {"num_technical": 3, "num_project": 1, "num_behavioral": 1}
            },
            expected_result="success"
        ))

        # 场景4: 降级场景测试
        scenarios.append(TestScenario(
            name="系统降级场景",
            category="special",
            description="模拟系统高负载下的降级处理",
            data={
                "candidate_name": "测试用户",
                "position": "java_backend",
                "experience": "mid",
                "answers": [
                    {
                        "question": "降级测试问题",
                        "answer": "这是一个测试回答",
                        "has_audio": False,
                        "has_video": False,
                    },
                ],
                "config": {"num_technical": 1, "num_project": 1, "num_behavioral": 1},
                "degradation_level": 4,  # 强制降级到Level 4
            },
            expected_result="degraded"
        ))

        # 场景5: 多模态完整流程
        scenarios.append(TestScenario(
            name="多模态完整评估",
            category="special",
            description="音频+视频+文本全模态评估",
            data={
                "candidate_name": "综合测试",
                "position": "algorithm_engineer",
                "experience": "expert",
                "answers": [
                    {
                        "question": "介绍你的深度学习项目",
                        "answer": "我使用PyTorch实现了Transformer模型，用于自然语言处理任务。模型在GLUE基准上达到了SOTA效果。",
                        "has_audio": True,
                        "has_video": True,
                        "audio_path": "tests/mock_audio/test_speech_5s.wav",
                    },
                ],
                "config": {"num_technical": 2, "num_project": 1, "num_behavioral": 1}
            },
            expected_result="multimodal"
        ))

        # 场景6: 快速连续提交（速率限制测试）
        scenarios.append(TestScenario(
            name="速率限制测试",
            category="special",
            description="快速连续提交多个回答",
            data={
                "candidate_name": "速率测试",
                "position": "java_backend",
                "experience": "junior",
                "answers": [
                    {
                        "question": f"快速问题{i}",
                        "answer": f"快速回答{i}",
                        "has_audio": False,
                        "has_video": False,
                    }
                    for i in range(15)  # 超过速率限制
                ],
                "config": {"num_technical": 5, "num_project": 2, "num_behavioral": 2}
            },
            expected_result="rate_limited"
        ))

        return scenarios

    @classmethod
    def generate_all_scenarios(cls) -> List[TestScenario]:
        """生成所有测试场景"""
        all_scenarios = []
        all_scenarios.extend(cls.generate_normal_scenarios())
        all_scenarios.extend(cls.generate_abnormal_scenarios())
        all_scenarios.extend(cls.generate_boundary_scenarios())
        all_scenarios.extend(cls.generate_special_scenarios())
        return all_scenarios

    @classmethod
    def generate_voice_analysis_data(cls) -> List[Dict[str, Any]]:
        """生成声纹分析测试数据"""
        return [
            {
                "name": "正常语速",
                "speech_rate": 150,  # 词/分钟
                "pause_count": 3,
                "pause_duration": 0.5,
                "emotion": "neutral",
                "confidence": 0.9,
            },
            {
                "name": "过快语速",
                "speech_rate": 250,
                "pause_count": 1,
                "pause_duration": 0.1,
                "emotion": "nervous",
                "confidence": 0.7,
            },
            {
                "name": "过慢语速",
                "speech_rate": 80,
                "pause_count": 8,
                "pause_duration": 2.0,
                "emotion": "uncertain",
                "confidence": 0.6,
            },
            {
                "name": "边界-零停顿",
                "speech_rate": 120,
                "pause_count": 0,
                "pause_duration": 0,
                "emotion": "neutral",
                "confidence": 0.8,
            },
            {
                "name": "边界-极大停顿",
                "speech_rate": 100,
                "pause_count": 20,
                "pause_duration": 5.0,
                "emotion": "confused",
                "confidence": 0.5,
            },
        ]

    @classmethod
    def generate_degradation_test_data(cls) -> List[Dict[str, Any]]:
        """生成降级策略测试数据"""
        return [
            {
                "level": 1,
                "name": "完全服务",
                "expected_features": ["streaming", "multimodal", "voice_analysis", "deepseek_api"],
            },
            {
                "level": 2,
                "name": "性能降级",
                "expected_features": ["streaming", "multimodal", "voice_analysis", "deepseek_api"],
            },
            {
                "level": 3,
                "name": "模型降级",
                "expected_features": ["streaming", "multimodal", "deepseek_api"],
                "disabled_features": ["voice_analysis"],
            },
            {
                "level": 4,
                "name": "功能降级",
                "expected_features": ["deepseek_api"],
                "disabled_features": ["streaming", "multimodal", "voice_analysis"],
            },
            {
                "level": 5,
                "name": "基础服务",
                "expected_features": [],
                "disabled_features": ["streaming", "multimodal", "voice_analysis", "deepseek_api"],
            },
        ]

    @classmethod
    def generate_scoring_test_data(cls) -> List[Dict[str, Any]]:
        """生成评分规则测试数据"""
        return [
            {
                "name": "优秀候选人",
                "position": "java_backend",
                "experience": "expert",
                "dimension_scores": {
                    "technical": 92,
                    "communication": 88,
                    "completeness": 90,
                    "problem_solving": 95,
                    "teamwork": 85,
                    "leadership": 88,
                },
                "expected_level": "excellent",
                "expected_recommendation": "强烈推荐",
            },
            {
                "name": "良好候选人",
                "position": "web_frontend",
                "experience": "senior",
                "dimension_scores": {
                    "technical": 78,
                    "communication": 82,
                    "completeness": 75,
                    "problem_solving": 80,
                    "teamwork": 85,
                    "leadership": 70,
                },
                "expected_level": "good",
                "expected_recommendation": "推荐",
            },
            {
                "name": "中等候选人",
                "position": "python_backend",
                "experience": "mid",
                "dimension_scores": {
                    "technical": 65,
                    "communication": 70,
                    "completeness": 68,
                    "problem_solving": 72,
                    "teamwork": 75,
                    "leadership": 60,
                },
                "expected_level": "average",
                "expected_recommendation": "考虑",
            },
            {
                "name": "及格候选人",
                "position": "algorithm_engineer",
                "experience": "junior",
                "dimension_scores": {
                    "technical": 55,
                    "communication": 60,
                    "completeness": 58,
                    "problem_solving": 62,
                    "teamwork": 65,
                    "leadership": 50,
                },
                "expected_level": "pass",
                "expected_recommendation": "待定",
            },
            {
                "name": "不及格候选人",
                "position": "data_engineer",
                "experience": "junior",
                "dimension_scores": {
                    "technical": 45,
                    "communication": 50,
                    "completeness": 48,
                    "problem_solving": 52,
                    "teamwork": 55,
                    "leadership": 40,
                },
                "expected_level": "fail",
                "expected_recommendation": "不推荐",
            },
            {
                "name": "边界-刚好及格",
                "position": "java_backend",
                "experience": "mid",
                "dimension_scores": {
                    "technical": 60,
                    "communication": 60,
                    "completeness": 60,
                    "problem_solving": 60,
                    "teamwork": 60,
                    "leadership": 60,
                },
                "expected_level": "pass",
                "expected_recommendation": "待定",
            },
            {
                "name": "边界-刚好良好",
                "position": "web_frontend",
                "experience": "senior",
                "dimension_scores": {
                    "technical": 70,
                    "communication": 70,
                    "completeness": 70,
                    "problem_solving": 70,
                    "teamwork": 70,
                    "leadership": 70,
                },
                "expected_level": "good",
                "expected_recommendation": "推荐",
            },
            {
                "name": "边界-刚好优秀",
                "position": "devops_engineer",
                "experience": "expert",
                "dimension_scores": {
                    "technical": 85,
                    "communication": 85,
                    "completeness": 85,
                    "problem_solving": 85,
                    "teamwork": 85,
                    "leadership": 85,
                },
                "expected_level": "excellent",
                "expected_recommendation": "强烈推荐",
            },
        ]
