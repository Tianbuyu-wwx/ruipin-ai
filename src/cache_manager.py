"""
锐聘AI - 缓存管理器
支持内存缓存和Redis分布式缓存
"""

import json
import hashlib
import time
import pickle
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta

from .logger import logger


# 尝试导入redis
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("未安装redis库，将使用内存缓存")


@dataclass
class CacheConfig:
    """缓存配置"""
    backend: str = "memory"  # "memory" 或 "redis"
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    default_ttl: int = 3600  # 默认过期时间（秒）
    max_memory_size: int = 1000  # 内存缓存最大条目数


class MemoryCache:
    """内存缓存实现 - 带LRU淘汰和过期清理"""
    
    def __init__(self, max_size: int = 1000):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
        self._access_count = 0  # 用于LRU
        self._cleanup_threshold = max_size * 2  # 触发清理的阈值
    
    def _cleanup_expired(self):
        """清理过期条目"""
        now = time.time()
        expired = [k for k, v in self.cache.items() if v["expires_at"] <= now]
        for k in expired:
            del self.cache[k]
        if expired:
            logger.debug(f"内存缓存清理: 移除 {len(expired)} 个过期条目")
    
    def _evict_lru(self, count: int = 1):
        """LRU淘汰最久未使用的条目"""
        if not self.cache:
            return
        # 按访问计数排序，淘汰最小的
        sorted_items = sorted(self.cache.items(), key=lambda x: x[1]["access_count"])
        for i in range(min(count, len(sorted_items))):
            del self.cache[sorted_items[i][0]]
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        if key in self.cache:
            item = self.cache[key]
            # 检查是否过期
            if item["expires_at"] > time.time():
                self.hits += 1
                self._access_count += 1
                item["access_count"] = self._access_count
                return item["value"]
            else:
                # 过期删除
                del self.cache[key]
        
        self.misses += 1
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """设置缓存值"""
        # 如果缓存过大，先清理过期和淘汰旧条目
        if len(self.cache) >= self._cleanup_threshold:
            self._cleanup_expired()
        if len(self.cache) >= self.max_size and key not in self.cache:
            # 删除最久未使用的条目（LRU）
            self._evict_lru(max(1, len(self.cache) // 10))  # 淘汰10%
        
        self._access_count += 1
        expires_at = time.time() + (ttl or 3600)
        self.cache[key] = {
            "value": value,
            "created_at": time.time(),
            "expires_at": expires_at,
            "access_count": self._access_count
        }
    
    def delete(self, key: str) -> bool:
        """删除缓存值"""
        if key in self.cache:
            del self.cache[key]
            return True
        return False
    
    def clear(self):
        """清空缓存"""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
        self._access_count = 0
    
    def keys(self, pattern: str = "*") -> List[str]:
        """获取匹配的键"""
        import fnmatch
        return [k for k in self.cache.keys() if fnmatch.fnmatch(k, pattern)]
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        total = self.hits + self.misses
        return {
            "size": len(self.cache),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total * 100, 2) if total > 0 else 0,
            "max_size": self.max_size
        }


class RedisCache:
    """Redis缓存实现"""
    
    def __init__(self, config: CacheConfig):
        if not REDIS_AVAILABLE:
            raise ImportError("未安装redis库，请运行: pip install redis")
        
        self.client = redis.Redis(
            host=config.redis_host,
            port=config.redis_port,
            db=config.redis_db,
            password=config.redis_password,
            decode_responses=False  # 使用二进制序列化
        )
        
        # 测试连接
        try:
            self.client.ping()
            logger.info("[OK] Redis连接成功")
        except Exception as e:
            logger.error(f"[ERROR] Redis连接失败: {e}")
            raise
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        try:
            data = self.client.get(key)
            if data:
                return pickle.loads(data)
            return None
        except Exception as e:
            logger.error(f"Redis获取失败: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """设置缓存值"""
        try:
            serialized = pickle.dumps(value)
            if ttl:
                self.client.setex(key, ttl, serialized)
            else:
                self.client.set(key, serialized)
        except Exception as e:
            logger.error(f"Redis设置失败: {e}")
    
    def delete(self, key: str) -> bool:
        """删除缓存值"""
        try:
            return self.client.delete(key) > 0
        except Exception as e:
            logger.error(f"Redis删除失败: {e}")
            return False
    
    def clear(self):
        """清空缓存"""
        try:
            self.client.flushdb()
        except Exception as e:
            logger.error(f"Redis清空失败: {e}")
    
    def keys(self, pattern: str = "*") -> List[str]:
        """获取匹配的键"""
        try:
            return [k.decode() if isinstance(k, bytes) else k for k in self.client.keys(pattern)]
        except Exception as e:
            logger.error(f"Redis获取键失败: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        try:
            info = self.client.info()
            return {
                "used_memory": info.get("used_memory_human", "N/A"),
                "connected_clients": info.get("connected_clients", 0),
                "total_keys": self.client.dbsize(),
                "uptime": info.get("uptime_in_seconds", 0)
            }
        except Exception as e:
            logger.error(f"Redis获取统计失败: {e}")
            return {}


class CacheManager:
    """缓存管理器 - 统一接口"""
    
    def __init__(self, config: Optional[CacheConfig] = None):
        """
        初始化缓存管理器
        
        Args:
            config: 缓存配置，默认使用内存缓存
        """
        self.config = config or CacheConfig()
        self.backend = None
        
        # 初始化后端
        if self.config.backend == "redis":
            try:
                self.backend = RedisCache(self.config)
                logger.info("[OK] 使用Redis缓存")
            except Exception as e:
                logger.warning(f"Redis初始化失败，回退到内存缓存: {e}")
                self.backend = MemoryCache(self.config.max_memory_size)
        else:
            self.backend = MemoryCache(self.config.max_memory_size)
            logger.info("[OK] 使用内存缓存")
    
    def _make_key(self, prefix: str, *args, **kwargs) -> str:
        """生成缓存键"""
        key_data = f"{prefix}:{json.dumps(args, sort_keys=True)}:{json.dumps(kwargs, sort_keys=True)}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        return self.backend.get(key)
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """设置缓存值"""
        self.backend.set(key, value, ttl or self.config.default_ttl)
    
    def delete(self, key: str) -> bool:
        """删除缓存值"""
        return self.backend.delete(key)
    
    def clear(self):
        """清空缓存"""
        self.backend.clear()
    
    def get_evaluation(self, answer: str, question: Optional[str] = None) -> Optional[Dict]:
        """
        获取评估缓存
        
        Args:
            answer: 候选人回答
            question: 面试问题
            
        Returns:
            缓存的评估结果或None
        """
        key = self._make_key("eval", answer, question)
        return self.get(key)
    
    def set_evaluation(self, answer: str, evaluation: Dict, question: Optional[str] = None, ttl: Optional[int] = None):
        """
        设置评估缓存
        
        Args:
            answer: 候选人回答
            evaluation: 评估结果
            question: 面试问题
            ttl: 过期时间
        """
        key = self._make_key("eval", answer, question)
        self.set(key, evaluation, ttl)
    
    def get_question(self, position: str, question_type: str, difficulty: int) -> Optional[Dict]:
        """获取问题缓存"""
        key = self._make_key("question", position, question_type, difficulty)
        return self.get(key)
    
    def set_question(self, position: str, question_type: str, difficulty: int, question: Dict, ttl: Optional[int] = None):
        """设置问题缓存"""
        key = self._make_key("question", position, question_type, difficulty)
        self.set(key, question, ttl)
    
    def get_report(self, session_id: str) -> Optional[Dict]:
        """获取报告缓存"""
        key = f"report:{session_id}"
        return self.get(key)
    
    def set_report(self, session_id: str, report: Dict, ttl: Optional[int] = None):
        """设置报告缓存"""
        key = f"report:{session_id}"
        self.set(key, report, ttl)
    
    def invalidate_session(self, session_id: str):
        """使会话相关缓存失效"""
        # 删除报告缓存
        self.delete(f"report:{session_id}")
        
        # 删除相关的评估缓存
        for key in self.backend.keys(f"*eval*{session_id}*"):
            self.delete(key)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        stats = self.backend.get_stats()
        stats["backend_type"] = type(self.backend).__name__
        return stats


# 全局缓存管理器实例
cache_manager = CacheManager()


def get_cache() -> CacheManager:
    """获取全局缓存管理器"""
    return cache_manager


if __name__ == "__main__":
    # 测试缓存管理器
    print("测试缓存管理器...")
    
    # 测试内存缓存
    cache = CacheManager(CacheConfig(backend="memory", default_ttl=60))
    
    # 测试基本操作
    cache.set("test_key", {"score": 85, "feedback": "测试"})
    result = cache.get("test_key")
    print(f"基本操作: {result}")
    
    # 测试评估缓存
    cache.set_evaluation("回答内容", {"score": 90}, "问题内容")
    eval_result = cache.get_evaluation("回答内容", "问题内容")
    print(f"评估缓存: {eval_result}")
    
    # 测试统计
    stats = cache.get_stats()
    print(f"缓存统计: {stats}")
    
    print("测试完成!")
