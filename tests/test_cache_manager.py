"""
缓存管理器测试
"""

import pytest
import time
from src.cache_manager import MemoryCache, CacheConfig, CacheManager


class TestMemoryCache:
    """测试内存缓存"""

    def test_basic_set_get(self):
        """测试基本存取"""
        cache = MemoryCache(max_size=100)
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

    def test_get_nonexistent(self):
        """测试获取不存在的键"""
        cache = MemoryCache(max_size=100)
        assert cache.get("nonexistent") is None

    def test_ttl_expiration(self):
        """测试TTL过期"""
        cache = MemoryCache(max_size=100)
        cache.set("key1", "value1", ttl=1)
        assert cache.get("key1") == "value1"
        time.sleep(1.1)
        assert cache.get("key1") is None

    def test_lru_eviction(self):
        """测试LRU淘汰"""
        cache = MemoryCache(max_size=3)
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")

        cache.get("key1")
        cache.set("key4", "value4")

        assert cache.get("key1") == "value1"
        assert cache.get("key2") is None

    def test_delete(self):
        """测试删除"""
        cache = MemoryCache(max_size=100)
        cache.set("key1", "value1")
        assert cache.delete("key1") is True
        assert cache.get("key1") is None
        assert cache.delete("nonexistent") is False

    def test_clear(self):
        """测试清空"""
        cache = MemoryCache(max_size=100)
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.clear()
        assert cache.get("key1") is None
        assert cache.get("key2") is None
        assert cache.get_stats()["size"] == 0

    def test_stats(self):
        """测试统计信息"""
        cache = MemoryCache(max_size=100)
        cache.set("key1", "value1")
        cache.get("key1")
        cache.get("nonexistent")

        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["size"] == 1
        assert stats["hit_rate"] == 50.0

    def test_keys_pattern(self):
        """测试键模式匹配"""
        cache = MemoryCache(max_size=100)
        cache.set("prefix_key1", "value1")
        cache.set("prefix_key2", "value2")
        cache.set("other_key", "value3")

        keys = cache.keys("prefix_*")
        assert len(keys) == 2
        assert "prefix_key1" in keys
        assert "prefix_key2" in keys


class TestCacheManager:
    """测试缓存管理器"""

    def test_memory_backend(self):
        """测试内存后端"""
        config = CacheConfig(backend="memory", default_ttl=3600)
        manager = CacheManager(config)
        assert "MemoryCache" in manager.get_stats()["backend_type"]

    def test_evaluation_cache(self):
        """测试评估缓存"""
        config = CacheConfig(backend="memory")
        manager = CacheManager(config)

        evaluation = {"score": 85, "feedback": "良好"}
        manager.set_evaluation("回答内容", evaluation, "问题内容")

        cached = manager.get_evaluation("回答内容", "问题内容")
        assert cached == evaluation

    def test_report_cache(self):
        """测试报告缓存"""
        config = CacheConfig(backend="memory")
        manager = CacheManager(config)

        report = {"overall_score": 80, "recommendation": "推荐"}
        manager.set_report("session-001", report)

        cached = manager.get_report("session-001")
        assert cached == report

    def test_invalidate_session(self):
        """测试会话缓存失效"""
        config = CacheConfig(backend="memory")
        manager = CacheManager(config)

        manager.set_report("session-002", {"score": 90})
        manager.invalidate_session("session-002")

        assert manager.get_report("session-002") is None
