"""
锐聘AI - 性能基准测试套件

提供系统性能基准测试和对比分析
"""

import time
import threading
import statistics
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime

from ..logger import logger


@dataclass
class BenchmarkResult:
    """基准测试结果"""
    benchmark_name: str
    iterations: int
    total_duration_ms: float
    avg_duration_ms: float
    min_duration_ms: float
    max_duration_ms: float
    median_duration_ms: float
    p95_duration_ms: float
    p99_duration_ms: float
    throughput_per_second: float
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return {
            "benchmark_name": self.benchmark_name,
            "iterations": self.iterations,
            "total_duration_ms": round(self.total_duration_ms, 2),
            "avg_duration_ms": round(self.avg_duration_ms, 3),
            "min_duration_ms": round(self.min_duration_ms, 3),
            "max_duration_ms": round(self.max_duration_ms, 3),
            "median_duration_ms": round(self.median_duration_ms, 3),
            "p95_duration_ms": round(self.p95_duration_ms, 3),
            "p99_duration_ms": round(self.p99_duration_ms, 3),
            "throughput_per_second": round(self.throughput_per_second, 2),
            "timestamp": datetime.fromtimestamp(self.timestamp).isoformat()
        }


class BenchmarkSuite:
    """性能基准测试套件"""

    # 性能目标
    PERFORMANCE_TARGETS = {
        "weight_calculation": {"avg_ms": 5, "p95_ms": 10},
        "confidence_calculation": {"avg_ms": 5, "p95_ms": 10},
        "level_classification": {"avg_ms": 3, "p95_ms": 5},
        "text_consistency": {"avg_ms": 10, "p95_ms": 20},
        "voice_integration": {"avg_ms": 10, "p95_ms": 20},
        "multimodal_fusion": {"avg_ms": 20, "p95_ms": 50},
        "degradation_evaluation": {"avg_ms": 5, "p95_ms": 10}
    }

    def __init__(self):
        self.results: List[BenchmarkResult] = []
        self._lock = threading.Lock()
        logger.info("[OK] 性能基准测试套件初始化完成")

    def run_all_benchmarks(self, iterations: int = 100) -> Dict[str, any]:
        """运行所有基准测试"""
        logger.info("=" * 60)
        logger.info("开始性能基准测试")
        logger.info("=" * 60)

        start_time = time.time()

        # 动态权重计算基准
        self._benchmark_weight_calculation(iterations)

        # 置信度计算基准
        self._benchmark_confidence_calculation(iterations)

        # 等级划分基准
        self._benchmark_level_classification(iterations)

        # 文本一致性基准
        self._benchmark_text_consistency(iterations)

        # 声纹整合基准
        self._benchmark_voice_integration(iterations)

        # 多模态融合基准
        self._benchmark_multimodal_fusion(iterations)

        # 降级评估基准
        self._benchmark_degradation_evaluation(iterations)

        total_duration = (time.time() - start_time) * 1000

        # 汇总结果
        summary = {
            "total_benchmarks": len(self.results),
            "total_duration_ms": round(total_duration, 2),
            "results": [r.to_dict() for r in self.results],
            "performance_summary": self._get_performance_summary()
        }

        logger.info("=" * 60)
        logger.info("性能基准测试完成")
        logger.info(f"  - 总测试数: {summary['total_benchmarks']}")
        logger.info(f"  - 总耗时: {summary['total_duration_ms']}ms")
        logger.info("=" * 60)

        return summary

    def _run_benchmark(self, name: str, func: Callable, iterations: int) -> BenchmarkResult:
        """运行单个基准测试"""
        durations = []

        # 预热
        for _ in range(min(10, iterations)):
            func()

        # 正式测试
        start = time.time()
        for _ in range(iterations):
            iter_start = time.time()
            func()
            durations.append((time.time() - iter_start) * 1000)
        total_duration = (time.time() - start) * 1000

        # 计算统计量
        sorted_durations = sorted(durations)
        avg = statistics.mean(durations)
        median = statistics.median(durations)
        p95_idx = int(len(sorted_durations) * 0.95)
        p99_idx = int(len(sorted_durations) * 0.99)

        result = BenchmarkResult(
            benchmark_name=name,
            iterations=iterations,
            total_duration_ms=total_duration,
            avg_duration_ms=avg,
            min_duration_ms=min(durations),
            max_duration_ms=max(durations),
            median_duration_ms=median,
            p95_duration_ms=sorted_durations[min(p95_idx, len(sorted_durations) - 1)],
            p99_duration_ms=sorted_durations[min(p99_idx, len(sorted_durations) - 1)],
            throughput_per_second=iterations / (total_duration / 1000) if total_duration > 0 else float('inf')
        )

        with self._lock:
            self.results.append(result)

        logger.info(f"[基准] {name}: 平均={result.avg_duration_ms:.3f}ms, "
                   f"P95={result.p95_duration_ms:.3f}ms, "
                   f"吞吐量={result.throughput_per_second:.1f}/s")

        return result

    def _benchmark_weight_calculation(self, iterations: int):
        """动态权重计算基准"""
        from ..scoring import DynamicWeightAdjuster

        adjuster = DynamicWeightAdjuster()

        def test():
            adjuster.calculate_weights("backend_engineer", "mid", has_voice_data=False)

        self._run_benchmark("weight_calculation", test, iterations)

    def _benchmark_confidence_calculation(self, iterations: int):
        """置信度计算基准"""
        from ..scoring import ScoreConfidenceCalculator

        calculator = ScoreConfidenceCalculator()

        def test():
            calculator.calculate(
                scores={"technical": 80, "communication": 75, "problem_solving": 82},
                answer_length=200,
                has_audio=True
            )

        self._run_benchmark("confidence_calculation", test, iterations)

    def _benchmark_level_classification(self, iterations: int):
        """等级划分基准"""
        from ..scoring import ScoreLevelClassifier

        classifier = ScoreLevelClassifier()

        def test():
            classifier.classify(75, dimension_scores={"technical": 80, "communication": 75})

        self._run_benchmark("level_classification", test, iterations)

    def _benchmark_text_consistency(self, iterations: int):
        """文本一致性基准"""
        from ..multimodal import TextConsistencyChecker

        checker = TextConsistencyChecker()
        text1 = "我有三年Java开发经验，熟悉Spring Boot框架"
        text2 = "我有3年Java开发经验，熟悉SpringBoot框架"

        def test():
            checker.check(text1, text2)

        self._run_benchmark("text_consistency", test, iterations)

    def _benchmark_voice_integration(self, iterations: int):
        """声纹整合基准"""
        from ..multimodal import VoiceAnalysisIntegration

        integrator = VoiceAnalysisIntegration()
        report = {
            "overall_score": 75,
            "speech_rate": {"score": 80, "category": "normal", "total_duration": 30},
            "pause": {"score": 70, "total_pauses": 5, "pause_ratio": 0.15},
            "emotion": {
                "confidence": 85,
                "scores": {"confident": 80, "nervous": 20, "neutral": 50}
            }
        }

        def test():
            integrator.integrate(report)

        self._run_benchmark("voice_integration", test, iterations)

    def _benchmark_multimodal_fusion(self, iterations: int):
        """多模态融合基准"""
        from ..multimodal import MultimodalFusionEngine

        engine = MultimodalFusionEngine()
        text_result = {
            "processed_text": "我有三年Java经验",
            "dimension_scores": {"technical": 85, "communication": 80}
        }
        whisper_text = "我有三年Java经验"
        voice_report = {
            "overall_score": 75,
            "speech_rate": {"score": 80, "category": "normal", "total_duration": 30},
            "pause": {"score": 70, "total_pauses": 5, "pause_ratio": 0.15},
            "emotion": {"confidence": 85, "scores": {"confident": 80}}
        }

        def test():
            engine.evaluate(text_result, whisper_text, voice_report)

        self._run_benchmark("multimodal_fusion", test, iterations)

    def _benchmark_degradation_evaluation(self, iterations: int):
        """降级评估基准"""
        from ..degradation import DegradationController

        controller = DegradationController()

        def test():
            controller.evaluate_and_act()

        self._run_benchmark("degradation_evaluation", test, iterations)

    def _get_performance_summary(self) -> Dict:
        """获取性能汇总"""
        summary = {
            "meets_targets": 0,
            "misses_targets": 0,
            "details": []
        }

        for result in self.results:
            target = self.PERFORMANCE_TARGETS.get(result.benchmark_name, {})
            avg_target = target.get("avg_ms", float('inf'))
            p95_target = target.get("p95_ms", float('inf'))

            meets_avg = result.avg_duration_ms <= avg_target
            meets_p95 = result.p95_duration_ms <= p95_target

            detail = {
                "benchmark": result.benchmark_name,
                "avg_ms": round(result.avg_duration_ms, 3),
                "avg_target_ms": avg_target,
                "avg_meets_target": meets_avg,
                "p95_ms": round(result.p95_duration_ms, 3),
                "p95_target_ms": p95_target,
                "p95_meets_target": meets_p95
            }

            summary["details"].append(detail)

            if meets_avg and meets_p95:
                summary["meets_targets"] += 1
            else:
                summary["misses_targets"] += 1

        return summary

    def get_comparison_report(self, baseline_results: Optional[List[BenchmarkResult]] = None) -> str:
        """生成对比报告"""
        report = []
        report.append("=" * 60)
        report.append("性能基准测试对比报告")
        report.append("=" * 60)
        report.append("")

        for result in self.results:
            report.append(f"【{result.benchmark_name}】")
            report.append(f"  迭代次数: {result.iterations}")
            report.append(f"  平均耗时: {result.avg_duration_ms:.3f}ms")
            report.append(f"  P95耗时:  {result.p95_duration_ms:.3f}ms")
            report.append(f"  P99耗时:  {result.p99_duration_ms:.3f}ms")
            report.append(f"  最小耗时: {result.min_duration_ms:.3f}ms")
            report.append(f"  最大耗时: {result.max_duration_ms:.3f}ms")
            report.append(f"  吞吐量:   {result.throughput_per_second:.1f}/s")

            # 与基线对比
            if baseline_results:
                baseline = next((b for b in baseline_results if b.benchmark_name == result.benchmark_name), None)
                if baseline:
                    avg_change = ((result.avg_duration_ms - baseline.avg_duration_ms) / baseline.avg_duration_ms) * 100
                    report.append(f"  平均耗时变化: {avg_change:+.1f}%")

            # 与目标对比
            target = self.PERFORMANCE_TARGETS.get(result.benchmark_name, {})
            if target:
                avg_target = target.get("avg_ms")
                if avg_target:
                    status = "✅" if result.avg_duration_ms <= avg_target else "❌"
                    report.append(f"  目标检查: {status} 平均<= {avg_target}ms")

            report.append("")

        report.append("=" * 60)

        return "\n".join(report)

    def get_bottleneck_report(self) -> Dict:
        """获取瓶颈报告"""
        bottlenecks = []
        recommendations = []

        for result in self.results:
            target = self.PERFORMANCE_TARGETS.get(result.benchmark_name, {})
            avg_target = target.get("avg_ms", float('inf'))
            p95_target = target.get("p95_ms", float('inf'))

            if result.avg_duration_ms > avg_target * 2:
                bottlenecks.append({
                    "benchmark": result.benchmark_name,
                    "issue": "平均耗时严重超标",
                    "actual_ms": result.avg_duration_ms,
                    "target_ms": avg_target
                })
                recommendations.append(f"优化 {result.benchmark_name} 的平均性能")

            if result.p95_duration_ms > p95_target * 2:
                bottlenecks.append({
                    "benchmark": result.benchmark_name,
                    "issue": "P95耗时严重超标",
                    "actual_ms": result.p95_duration_ms,
                    "target_ms": p95_target
                })
                recommendations.append(f"优化 {result.benchmark_name} 的尾部延迟")

        return {
            "bottlenecks": bottlenecks,
            "recommendations": recommendations,
            "severity": "high" if len(bottlenecks) >= 3 else "medium" if bottlenecks else "low"
        }
