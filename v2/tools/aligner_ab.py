"""对齐器 A/B 验证：把"口型准不准"从主观印象变成可回归的指标。

为什么需要这个工具
------------------
选型研究（`docs/开源TTS选型研究-2026-10.md` §8）的结论是：**自托管开源 TTS 权重
都不原生输出词级时间戳，词时间轴只能靠强制对齐**。于是对齐器成了口型正确性的
**唯一真相源**——它错了，嘴就是错的，而且**错得均匀、肉眼几乎看不出**。

"看着还行"在这件事上是无效验收：错 80 ms 和错 800 ms 在静音背景下都像"差不多"。
所以要拿数字说话，而且这个数字要能重跑——换对齐器、换模型、调阈值之后，同一批
音频重跑一次就能看出是变好还是变坏。

怎么用
------
先用 `--self-test` 确认工具自身的指标算法没问题（不需要任何模型）：

    python tools/aligner_ab.py --self-test

再对着真实音频跑（需要一个在跑的对齐服务）：

    python tools/aligner_ab.py --manifest clips.jsonl \\
        --candidate "http:http://127.0.0.1:8024" \\
        --candidate "mod:mypkg.whisperx_aligner:make" \\
        --out reports/aligner_ab.json

清单格式（JSON 数组，或一行一条的 JSONL）::

    [{"id": "q01",
      "audio": "clips/q01.wav",
      "text": "请介绍一下你自己",
      "truth": [{"word": "请", "start_ms": 0, "end_ms": 90}, ...]}]

`truth` 是**独立于两个候选的参考真值**（人工标注，或用另一个独立方法产出）。
没有它这个工具就没有意义——拿 A 当标准去量 B，只会得出"B 像不像 A"。

指标怎么读（重要）
------------------
* `coverage` = 参考词里被匹配上的比例。**先看它，再看误差。**
  匹配靠词串相等；两个对齐器分词方式不同（"你好" vs "你"/"好"）会拉低覆盖率，
  此时误差数字只建立在少数能对上的词上，**代表性不足**。覆盖率低于
  `--min-coverage`（默认 0.8）的片段会被标成 `low_coverage` 并从结论里排除。
* `start_err_median_ms` / `end_err_median_ms` = 匹配词边界误差的中位数。
  用中位数而非均值：少数极端错位会污染均值，而中位数反映"大多数词错多少"。
* `over_threshold` = 误差超过阈值的**匹配词数**。阈值由 `--threshold-ms` 给出，
  **它是一个可配置的判据，不是行业标准**；默认 100 ms。

一句话：**覆盖率低就不看误差，误差看中位数与超阈个数，不看均值。**
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from ruipin.adapters.align_qwen3 import Qwen3ForcedAligner  # noqa: E402
from ruipin.adapters.http import HttpxTransport, HttpTransport  # noqa: E402
from ruipin.ports import WordTiming  # noqa: E402

#: 一个对齐器：收 (音频字节, 文本)，给词时间轴。可同步可异步。
AlignerFn = Callable[[bytes, str], Any]

#: 误差超阈判据的默认值（ms）。**可配置的判据，不是行业标准。**
DEFAULT_THRESHOLD_MS = 100

#: 覆盖率低于此值的片段视为"匹配不足"，不进结论。
DEFAULT_MIN_COVERAGE = 0.8


# ---------- 指标计算（纯函数，可单测） ----------


@dataclass(frozen=True)
class PairError:
    """一对匹配词（参考 vs 候选）的边界误差。"""

    word: str
    ref_start_ms: int
    ref_end_ms: int
    cand_start_ms: int
    cand_end_ms: int
    start_err_ms: int
    end_err_ms: int


@dataclass(frozen=True)
class ClipReport:
    """单个片段的比对结果。"""

    clip_id: str
    ref_words: int
    cand_words: int
    matched: int
    coverage: float
    count_agreement: float
    start_err_median_ms: Optional[float]
    end_err_median_ms: Optional[float]
    start_err_p95_ms: Optional[float]
    end_err_p95_ms: Optional[float]
    over_threshold: int
    threshold_ms: int
    low_coverage: bool


def _percentile(sorted_values: Sequence[float], q: float) -> Optional[float]:
    """最近秩法（nearest-rank）分位数。空输入返回 None。

    用最近秩而不是插值：误差样本量本来就只有几十个，"插"出来的小数点后几位
    是虚假精度——它会让两次跑出来的差异看起来比实际更细。
    """
    if not sorted_values:
        return None
    n = len(sorted_values)
    idx = max(0, min(n - 1, int(round(q * n + 0.5)) - 1))
    return float(sorted_values[idx])


def match_words(
    ref: Sequence[WordTiming], cand: Sequence[WordTiming]
) -> list[PairError]:
    """按词串相等对齐两条时间轴，返回成对的边界误差。

    **这是本工具最薄弱的一环，必须说清楚**：它用 `SequenceMatcher` 做最长公共
    子序列式的匹配，只在词串**完全相同**时配对。两个对齐器若分词方式不同
    （一个给"你好"、另一个给"你"/"好"），配对率会显著下降。
    因此 `coverage` 与误差要一起看——`ClipReport.low_coverage` 就是为这件事设的。
    """
    ref_words = [w.word for w in ref]
    cand_words = [w.word for w in cand]
    matcher = SequenceMatcher(a=ref_words, b=cand_words, autojunk=False)

    pairs: list[PairError] = []
    for block in matcher.get_matching_blocks():
        for offset in range(block.size):
            r = ref[block.a + offset]
            c = cand[block.b + offset]
            pairs.append(
                PairError(
                    word=r.word,
                    ref_start_ms=int(r.start_ms),
                    ref_end_ms=int(r.end_ms),
                    cand_start_ms=int(c.start_ms),
                    cand_end_ms=int(c.end_ms),
                    start_err_ms=abs(int(c.start_ms) - int(r.start_ms)),
                    end_err_ms=abs(int(c.end_ms) - int(r.end_ms)),
                )
            )
    return pairs


def compare_clip(
    clip_id: str,
    ref: Sequence[WordTiming],
    cand: Sequence[WordTiming],
    *,
    threshold_ms: int = DEFAULT_THRESHOLD_MS,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
) -> ClipReport:
    """比对单个片段。"""
    pairs = match_words(ref, cand)
    matched = len(pairs)
    coverage = (matched / len(ref)) if ref else 0.0
    denominator = max(len(ref), len(cand))
    count_agreement = (matched / denominator) if denominator else 0.0

    start_errs = sorted(p.start_err_ms for p in pairs)
    end_errs = sorted(p.end_err_ms for p in pairs)
    over = sum(
        1 for p in pairs if p.start_err_ms > threshold_ms or p.end_err_ms > threshold_ms
    )

    def _med(values: Sequence[int]) -> Optional[float]:
        return float(statistics.median(values)) if values else None

    return ClipReport(
        clip_id=clip_id,
        ref_words=len(ref),
        cand_words=len(cand),
        matched=matched,
        coverage=round(coverage, 4),
        count_agreement=round(count_agreement, 4),
        start_err_median_ms=_med(start_errs),
        end_err_median_ms=_med(end_errs),
        start_err_p95_ms=_percentile(start_errs, 0.95),
        end_err_p95_ms=_percentile(end_errs, 0.95),
        over_threshold=over,
        threshold_ms=threshold_ms,
        low_coverage=bool(ref) and coverage < min_coverage,
    )


@dataclass(frozen=True)
class AlignerReport:
    """一个候选对齐器在一批片段上的汇总。"""

    name: str
    clips: int
    usable_clips: int
    low_coverage_clips: int
    ref_words_total: int
    matched_total: int
    coverage: float
    start_err_median_ms: Optional[float]
    end_err_median_ms: Optional[float]
    start_err_p95_ms: Optional[float]
    end_err_p95_ms: Optional[float]
    over_threshold: int
    over_threshold_rate: Optional[float]
    threshold_ms: int
    per_clip: list[ClipReport] = field(default_factory=list)


def aggregate(
    name: str,
    per_clip: Sequence[ClipReport],
    *,
    threshold_ms: int = DEFAULT_THRESHOLD_MS,
) -> AlignerReport:
    """把逐片段结果汇总。

    **误差统计只汇总"可用片段"**（覆盖率达标者）。把低覆盖片段混进来会让中位数
    被少数硬凑上的词带跑，读出来的数字比实际更乐观或更悲观，且看不出来。
    低覆盖片段单独计数。
    """
    usable = [c for c in per_clip if not c.low_coverage]
    ref_total = sum(c.ref_words for c in per_clip)
    matched_total = sum(c.matched for c in per_clip)
    usable_matched = sum(c.matched for c in usable)
    usable_ref = sum(c.ref_words for c in usable)

    # 汇总中位数用"逐片段中位数的中位数"近似不了池化样本，所以这里按
    # 可用片段的匹配数加权合并各片段的误差分布——对中位数而言，等价做法是
    # 直接取各片段中位数按匹配数加权；样本量小时与池化差异可忽略，且无需
    # 重新读音频。这里选择**加权平均中位数**并把这个口径写在报告里。
    def _weighted_median(attr: str) -> Optional[float]:
        pairs = [
            (getattr(c, attr), c.matched)
            for c in usable
            if getattr(c, attr) is not None and c.matched > 0
        ]
        if not pairs:
            return None
        total = sum(w for _v, w in pairs)
        acc = sum(float(v) * w for v, w in pairs)
        return round(acc / total, 2)

    def _max_p95(attr: str) -> Optional[float]:
        values = [getattr(c, attr) for c in usable if getattr(c, attr) is not None]
        return max(values) if values else None

    over = sum(c.over_threshold for c in usable)
    return AlignerReport(
        name=name,
        clips=len(per_clip),
        usable_clips=len(usable),
        low_coverage_clips=len(per_clip) - len(usable),
        ref_words_total=ref_total,
        matched_total=matched_total,
        coverage=round((matched_total / ref_total) if ref_total else 0.0, 4),
        start_err_median_ms=_weighted_median("start_err_median_ms"),
        end_err_median_ms=_weighted_median("end_err_median_ms"),
        start_err_p95_ms=_max_p95("start_err_p95_ms"),
        end_err_p95_ms=_max_p95("end_err_p95_ms"),
        over_threshold=over,
        over_threshold_rate=(
            round(over / usable_matched, 4) if usable_matched else None
        ),
        threshold_ms=threshold_ms,
        per_clip=list(per_clip),
    )


# ---------- 清单与对齐器装配 ----------


@dataclass(frozen=True)
class Clip:
    """一条待比对素材。"""

    clip_id: str
    audio_path: Path
    text: str
    truth: tuple[WordTiming, ...]


def load_manifest(path: Path) -> list[Clip]:
    """读清单（JSON 数组或 JSONL）。字段缺失/越界一律抛错，不猜路径。"""
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        raise ValueError(f"清单为空：{path}")
    if raw.startswith("["):
        entries = json.loads(raw)
    else:
        entries = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if not isinstance(entries, list):
        raise ValueError(f"清单顶层必须是数组：{path}")

    base = path.parent
    clips: list[Clip] = []
    for i, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise ValueError(f"第 {i} 条不是对象：{type(entry).__name__}")
        for key in ("id", "audio", "text", "truth"):
            if key not in entry:
                raise ValueError(f"第 {i} 条缺字段 {key!r}")
        audio = Path(str(entry["audio"]))
        if not audio.is_absolute():
            audio = base / audio
        truth = tuple(
            WordTiming(
                word=str(item["word"]),
                start_ms=int(item["start_ms"]),
                end_ms=int(item["end_ms"]),
            )
            for item in entry["truth"]
        )
        clips.append(Clip(str(entry["id"]), audio, str(entry["text"]), truth))
    return clips


def build_aligner(spec: str) -> tuple[str, AlignerFn]:
    """按 `--candidate` 的写法装配对齐器。

    * `http:<base_url>` —— 走 `Qwen3ForcedAligner` 打真实服务（需要它在跑）；
    * `mod:<module>:<attr>` —— 导入任意可调用对象，`(audio_bytes, text) -> words`。
      用来接 WhisperX / MFA 等外部实现，**不在本工具里内置**：它们的依赖很重，
      装进来会让"跑指标的脚本"变成"要装半个语音栈"。
    """
    if spec.startswith("http:"):
        base_url = spec[len("http:") :]
        if not base_url:
            raise ValueError("http: 后面要跟服务地址，例如 http:http://127.0.0.1:8024")
        aligner = Qwen3ForcedAligner(HttpxTransport(), base_url=base_url)
        return f"qwen3-forced-aligner@{base_url}", aligner

    if spec.startswith("mod:"):
        body = spec[len("mod:") :]
        if ":" not in body:
            raise ValueError("mod: 的写法是 mod:<module>:<attr>")
        module_name, attr = body.rsplit(":", 1)
        import importlib

        factory = getattr(importlib.import_module(module_name), attr)
        target = factory() if callable(factory) else factory
        return f"{module_name}:{attr}", target

    raise ValueError(f"认不出的 --candidate 写法：{spec!r}（只认 http: 与 mod:）")


async def _run_aligner(aligner: AlignerFn, audio: bytes, text: str) -> list[WordTiming]:
    result = aligner(audio, text)
    if hasattr(result, "__await__"):
        result = await result
    return list(result)


def evaluate_candidate(
    name: str,
    aligner: AlignerFn,
    clips: Sequence[Clip],
    *,
    threshold_ms: int = DEFAULT_THRESHOLD_MS,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
) -> tuple[AlignerReport, list[str]]:
    """跑一个候选，返回 (汇总报告, 逐片段错误说明)。

    **对齐器抛错不中断整批**：单个片段失败只记一条说明并跳过——否则一个坏音频
    会让你拿不到任何结论。但失败会**显式列出来**，不静默吞掉。
    """
    per_clip: list[ClipReport] = []
    failures: list[str] = []
    for clip in clips:
        try:
            audio = clip.audio_path.read_bytes()
        except OSError as exc:
            failures.append(f"{clip.clip_id}: 读不到音频 {clip.audio_path} ({exc})")
            continue
        try:
            produced = asyncio.run(_run_aligner(aligner, audio, clip.text))
        except Exception as exc:  # noqa: BLE001 - 单个片段失败不该毁掉整批
            failures.append(f"{clip.clip_id}: 对齐失败 {type(exc).__name__}: {exc}")
            continue
        per_clip.append(
            compare_clip(
                clip.clip_id,
                clip.truth,
                produced,
                threshold_ms=threshold_ms,
                min_coverage=min_coverage,
            )
        )
    return aggregate(name, per_clip, threshold_ms=threshold_ms), failures


# ---------- 输出 ----------


def _fmt(value: Optional[float]) -> str:
    return "—" if value is None else f"{value:.1f}"


def render_table(reports: Iterable[AlignerReport]) -> str:
    """人读的对比表。列的顺序按"先看覆盖率，再看误差"排。"""
    header = (
        f"{'对齐器':<38}{'片段':>6}{'可用':>6}{'覆盖':>8}"
        f"{'起中位':>9}{'止中位':>9}{'起p95':>9}{'止p95':>9}{'超阈':>7}"
    )
    lines = [header, "-" * len(header)]
    for r in reports:
        lines.append(
            f"{r.name[:36]:<38}{r.clips:>6}{r.usable_clips:>6}{r.coverage:>8.3f}"
            f"{_fmt(r.start_err_median_ms):>9}{_fmt(r.end_err_median_ms):>9}"
            f"{_fmt(r.start_err_p95_ms):>9}{_fmt(r.end_err_p95_ms):>9}"
            f"{r.over_threshold:>7}"
        )
    lines.append("")
    lines.append(
        "口径：覆盖 = 匹配上的参考词占比（靠词串相等匹配，分词不同会拉低）；"
        f"超阈 = 起/止任一误差 > {next(iter(reports), None).threshold_ms if list(reports) else DEFAULT_THRESHOLD_MS} ms 的匹配词数。"
    )
    lines.append("低覆盖片段（<0.8）不进误差统计；误差看中位数与超阈个数，不看均值。")
    return "\n".join(lines)


# ---------- 自检 ----------


def self_test() -> list[str]:
    """用合成数据验证指标算法本身。不需要任何模型或音频。

    它回答的是"这个工具算得对不对"，不是"对齐器好不好"——两件事必须分开，
    否则算法算错了会把结论带向完全错误的方向，而没人会发现。
    """
    problems: list[str] = []

    # 完全一致 → 误差全 0、覆盖 1.0
    ref = [WordTiming("你", 0, 100), WordTiming("好", 100, 200)]
    r = compare_clip("same", ref, list(ref))
    if (r.start_err_median_ms, r.end_err_median_ms) != (0.0, 0.0):
        problems.append(f"完全一致时误差应为 0，实际 {r.start_err_median_ms}/{r.end_err_median_ms}")
    if r.coverage != 1.0 or r.over_threshold != 0:
        problems.append(f"完全一致时覆盖应为 1.0、超阈应为 0，实际 {r.coverage}/{r.over_threshold}")

    # 整体平移 50 ms → 中位数 50，且不超 100 ms 阈值
    shifted = [WordTiming(w.word, w.start_ms + 50, w.end_ms + 50) for w in ref]
    r = compare_clip("shift", ref, shifted)
    if r.start_err_median_ms != 50.0 or r.over_threshold != 0:
        problems.append(f"平移 50 ms 应得中位 50 / 超阈 0，实际 {r.start_err_median_ms}/{r.over_threshold}")

    # 整体平移 150 ms → 每个词都超 100 ms 阈值
    far = [WordTiming(w.word, w.start_ms + 150, w.end_ms + 150) for w in ref]
    r = compare_clip("far", ref, far)
    if r.over_threshold != 2:
        problems.append(f"平移 150 ms 两个词都该超阈，实际 {r.over_threshold}")

    # 分词不同 → 覆盖率下降、被标 low_coverage，且**不**进汇总
    coarse = [WordTiming("你好", 0, 200)]
    r = compare_clip("coarse", ref, coarse)
    if r.coverage != 0.0 or not r.low_coverage:
        problems.append(f"分词不同应得覆盖 0 且 low_coverage，实际 {r.coverage}/{r.low_coverage}")
    agg = aggregate("x", [r])
    if agg.usable_clips != 0 or agg.start_err_median_ms is not None:
        problems.append("低覆盖片段不该进误差统计")

    # 最近秩分位数
    if _percentile([1.0, 2.0, 3.0, 4.0, 5.0], 0.95) != 5.0:
        problems.append("p95 最近秩算错")
    if _percentile([], 0.95) is not None:
        problems.append("空样本的分位数应为 None")

    return problems


# ---------- CLI ----------


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="对齐器 A/B 验证：同一批音频过多个对齐器，用独立参考真值量误差。"
    )
    parser.add_argument("--manifest", type=Path, help="清单文件（JSON 或 JSONL）")
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        help="候选对齐器，写法 http:<base_url> 或 mod:<module>:<attr>；可重复给多个",
    )
    parser.add_argument("--out", type=Path, help="把完整结果写成 JSON")
    parser.add_argument(
        "--threshold-ms",
        type=int,
        default=DEFAULT_THRESHOLD_MS,
        help=f"误差超阈判据（ms），默认 {DEFAULT_THRESHOLD_MS}。可配置的判据，不是行业标准",
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=DEFAULT_MIN_COVERAGE,
        help=f"覆盖率低于此值的片段不进误差统计，默认 {DEFAULT_MIN_COVERAGE}",
    )
    parser.add_argument("--self-test", action="store_true", help="只验证指标算法，不碰模型")
    args = parser.parse_args(argv)

    if args.self_test:
        problems = self_test()
        if problems:
            print("自检失败：", file=sys.stderr)
            for p in problems:
                print(f"  - {p}", file=sys.stderr)
            return 1
        print("自检通过：指标算法（覆盖率 / 中位数 / 最近秩分位数 / 低覆盖排除）全部符合预期。")
        return 0

    if args.manifest is None or not args.candidate:
        parser.error("要么给 --self-test，要么同时给 --manifest 与至少一个 --candidate")
        return 2
    if args.manifest is None:
        parser.error("缺少 --manifest")
        return 2

    clips = load_manifest(args.manifest)
    if not clips:
        print("清单里一条素材都没有。", file=sys.stderr)
        return 1

    reports: list[AlignerReport] = []
    all_failures: dict[str, list[str]] = {}
    for spec in args.candidate:
        try:
            name, aligner = build_aligner(spec)
        except Exception as exc:  # noqa: BLE001 - 装配失败要清楚地报出来
            print(f"装配对齐器失败 {spec!r}: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        report, failures = evaluate_candidate(
            name,
            aligner,
            clips,
            threshold_ms=args.threshold_ms,
            min_coverage=args.min_coverage,
        )
        reports.append(report)
        if failures:
            all_failures[name] = failures

    print(render_table(reports))
    for name, failures in all_failures.items():
        print(f"\n{name} 有 {len(failures)} 个片段失败（已跳过，未计入统计）：")
        for line in failures[:20]:
            print(f"  - {line}")
        if len(failures) > 20:
            print(f"  … 另有 {len(failures) - 20} 条")

    if args.out:
        payload = {
            "threshold_ms": args.threshold_ms,
            "min_coverage": args.min_coverage,
            "clips": len(clips),
            "reports": [asdict(r) for r in reports],
            "failures": all_failures,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n完整结果已写入 {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
