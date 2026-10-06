"""`tools/aligner_ab.py` 的测试：保证**指标算法本身**是对的。

为什么这个测试很重要
--------------------
A/B 工具的输出会被用来决定"用哪个对齐器"，而它算错了不会报错——它只会给出一个
数字。**算错的指标比没有指标更危险**，因为它会带着权威感把人带向错误结论。

所以这里逐条钉死算法的行为：完全一致 → 误差 0；整体平移 50 ms → 中位 50 且不超阈；
平移 150 ms → 全超阈；分词不同 → 覆盖率下降并被排除出统计；分位数按最近秩。

工具脚本按文件路径加载（`tools/` 不在 pytest 的 pythonpath 里），与
`tests/test_wire_fixtures.py` 同一套做法。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from ruipin.ports import WordTiming

_V2_ROOT = Path(__file__).resolve().parents[1]
_TOOL_PATH = _V2_ROOT / "tools" / "aligner_ab.py"


def _load_tool():
    """按文件路径加载工具。

    注意 `sys.modules` 那一步**不能省**：工具里有 frozen dataclass，而 Python 3.13
    的 `dataclasses` 会通过 `sys.modules[cls.__module__]` 去解析类型注解；模块没注册
    就会炸在 `AttributeError: 'NoneType' object has no attribute '__dict__'`，
    报错位置还落在标准库里，看不出真正原因。
    """
    spec = importlib.util.spec_from_file_location("_aligner_ab", _TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(spec.name, None)
        raise
    return module


ab = _load_tool()


def W(word: str, start: int, end: int) -> WordTiming:
    return WordTiming(word=word, start_ms=start, end_ms=end)


REF = [W("你", 0, 100), W("好", 100, 200), W("吗", 200, 300)]


# ---------- 指标算法 ----------


def test_identical_timelines_have_zero_error_and_full_coverage():
    r = ab.compare_clip("c", REF, list(REF))
    assert (r.start_err_median_ms, r.end_err_median_ms) == (0.0, 0.0)
    assert r.coverage == 1.0
    assert r.count_agreement == 1.0
    assert r.over_threshold == 0
    assert not r.low_coverage


def test_uniform_shift_reports_median_and_stays_under_threshold():
    """整体平移 50 ms：中位数必须是 50（不是均值的近似），且不超 100 ms 阈值。"""
    shifted = [W(w.word, w.start_ms + 50, w.end_ms + 50) for w in REF]
    r = ab.compare_clip("c", REF, shifted)
    assert r.start_err_median_ms == 50.0
    assert r.end_err_median_ms == 50.0
    assert r.over_threshold == 0


def test_uniform_shift_beyond_threshold_counts_every_word():
    far = [W(w.word, w.start_ms + 150, w.end_ms + 150) for w in REF]
    r = ab.compare_clip("c", REF, far, threshold_ms=100)
    assert r.over_threshold == 3


def test_threshold_is_configurable_and_actually_used():
    """阈值是判据不是常量：调大之后同一份数据超阈数必须变。"""
    shifted = [W(w.word, w.start_ms + 120, w.end_ms + 120) for w in REF]
    assert ab.compare_clip("c", REF, shifted, threshold_ms=100).over_threshold == 3
    assert ab.compare_clip("c", REF, shifted, threshold_ms=200).over_threshold == 0


def test_different_tokenization_lowers_coverage_and_flags_low_coverage():
    """分词方式不同（下划线 vs 细粒度）时覆盖率必须掉下来并被标记。

    这是工具最薄弱的一环，必须**看见**它：低覆盖片段要排除出误差统计，
    否则误差会建立在少数硬凑上的词上，代表性不足却看不出来。
    """
    coarse = [W("你好吗", 0, 300)]
    r = ab.compare_clip("c", REF, coarse)
    assert r.coverage == 0.0
    assert r.low_coverage is True
    assert r.start_err_median_ms is None


def test_partial_match_computes_error_only_on_matched_words():
    """只匹配上中间那个词时，误差只能来自它，且覆盖率如实反映只匹配了 1/3。"""
    partial = [W("好", 130, 230)]
    r = ab.compare_clip("c", REF, partial)
    assert r.matched == 1
    assert r.coverage == pytest.approx(1 / 3, abs=1e-4)  # 报告里四舍五入到 4 位
    assert r.start_err_median_ms == 30.0
    assert r.end_err_median_ms == 30.0


def test_empty_reference_is_not_treated_as_perfect():
    """空参考真值不是"完美"，是"没得比"——覆盖率为 0 且不算低覆盖（没有参考可言）。"""
    r = ab.compare_clip("c", [], list(REF))
    assert r.coverage == 0.0
    assert r.low_coverage is False
    assert r.start_err_median_ms is None


def test_out_of_order_candidate_uses_sequence_matching():
    """候选顺序被打乱时按子序列匹配，匹配数下降而不是错配出一堆假误差。"""
    shuffled = [W("吗", 0, 50), W("你", 60, 120), W("好", 120, 200)]
    r = ab.compare_clip("c", REF, shuffled)
    # 最长公共子序列是 ["你", "好"]（长度 2）——不会把"吗"错配给"你"而造出假误差
    assert r.matched == 2
    assert r.coverage == pytest.approx(2 / 3, abs=1e-4)  # 同上：报告值保留 4 位
    # 2/3 < 0.8，因此它应被判低覆盖并排除出误差统计
    assert r.low_coverage is True


def test_low_coverage_clip_is_excluded_from_error_aggregate():
    """低覆盖片段的误差**不许**进汇总——否则中位数会被少数硬凑上的词带跑。"""
    good = ab.compare_clip("good", REF, list(REF))
    bad = ab.compare_clip("bad", REF, [W("完全不同的词", 0, 10)])
    agg = ab.aggregate("x", [good, bad])
    assert agg.clips == 2
    assert agg.usable_clips == 1
    assert agg.low_coverage_clips == 1
    assert agg.start_err_median_ms == 0.0  # 只来自 good
    assert agg.over_threshold == 0


def test_percentile_uses_nearest_rank():
    assert ab._percentile([1.0, 2.0, 3.0, 4.0, 5.0], 0.95) == 5.0
    assert ab._percentile([7.0], 0.95) == 7.0
    assert ab._percentile([], 0.95) is None


def test_aggregate_reports_over_threshold_rate_over_usable_words():
    shifted = [W(w.word, w.start_ms + 150, w.end_ms + 150) for w in REF]
    r = ab.compare_clip("c", REF, shifted, threshold_ms=100)
    agg = ab.aggregate("x", [r], threshold_ms=100)
    assert agg.matched_total == 3
    assert agg.over_threshold == 3
    assert agg.over_threshold_rate == 1.0


# ---------- 清单加载 ----------


def _write_manifest(tmp_path: Path, payload) -> Path:
    p = tmp_path / "clips.json"
    p.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return p


def test_load_manifest_parses_json_array(tmp_path: Path):
    p = _write_manifest(
        tmp_path,
        [
            {
                "id": "q01",
                "audio": "a.wav",
                "text": "你好吗",
                "truth": [
                    {"word": "你", "start_ms": 0, "end_ms": 100},
                    {"word": "好", "start_ms": 100, "end_ms": 200},
                ],
            }
        ],
    )
    clips = ab.load_manifest(p)
    assert len(clips) == 1
    assert clips[0].clip_id == "q01"
    assert clips[0].audio_path == (tmp_path / "a.wav").resolve()
    assert clips[0].truth == (W("你", 0, 100), W("好", 100, 200))


def test_load_manifest_parses_jsonl(tmp_path: Path):
    p = tmp_path / "clips.jsonl"
    lines = [
        {"id": "a", "audio": "a.wav", "text": "x", "truth": [{"word": "x", "start_ms": 0, "end_ms": 1}]},
        {"id": "b", "audio": "b.wav", "text": "y", "truth": [{"word": "y", "start_ms": 0, "end_ms": 1}]},
    ]
    p.write_text("\n".join(json.dumps(line) for line in lines), encoding="utf-8")
    assert [c.clip_id for c in ab.load_manifest(p)] == ["a", "b"]


def test_load_manifest_rejects_missing_field(tmp_path: Path):
    """缺字段要抛，不能猜一个默认值——猜出来的路径会静默指向错误的音频。"""
    p = _write_manifest(tmp_path, [{"id": "a", "audio": "a.wav", "text": "x"}])
    with pytest.raises(ValueError, match="truth"):
        ab.load_manifest(p)


def test_load_manifest_rejects_empty_file(tmp_path: Path):
    p = tmp_path / "empty.json"
    p.write_text("   ", encoding="utf-8")
    with pytest.raises(ValueError, match="空"):
        ab.load_manifest(p)


# ---------- 对齐器装配 ----------


def test_build_aligner_rejects_unknown_scheme():
    with pytest.raises(ValueError, match="认不出"):
        ab.build_aligner("ftp://somewhere")


def test_build_aligner_http_requires_address():
    with pytest.raises(ValueError, match="服务地址"):
        ab.build_aligner("http:")


def test_build_aligner_http_builds_qwen3():
    name, aligner = ab.build_aligner("http:http://127.0.0.1:8024")
    assert name == "qwen3-forced-aligner@http://127.0.0.1:8024"
    assert aligner.url.endswith("/v1/audio/align")


# ---------- 逐片段评估：失败隔离 ----------


def _make_clip(tmp_path: Path, clip_id: str, truth=None) -> "ab.Clip":
    audio = tmp_path / f"{clip_id}.wav"
    audio.write_bytes(b"FAKE-WAV-BYTES")
    return ab.Clip(clip_id, audio, "你好吗", tuple(truth or REF))


def test_evaluate_candidate_runs_sync_aligner(tmp_path: Path):
    clip = _make_clip(tmp_path, "c1")

    def aligner(audio: bytes, text: str):
        assert audio == b"FAKE-WAV-BYTES"
        return list(REF)

    report, failures = ab.evaluate_candidate("sync", aligner, [clip])
    assert failures == []
    assert report.clips == 1 and report.usable_clips == 1
    assert report.start_err_median_ms == 0.0


def test_evaluate_candidate_runs_async_aligner(tmp_path: Path):
    clip = _make_clip(tmp_path, "c1")

    async def aligner(audio: bytes, text: str):
        return list(REF)

    report, failures = ab.evaluate_candidate("async", aligner, [clip])
    assert failures == []
    assert report.start_err_median_ms == 0.0


def test_evaluate_candidate_isolates_single_clip_failure(tmp_path: Path):
    """一个片段炸了不该毁掉整批——但必须**列出来**，不能静默吞掉。"""
    good = _make_clip(tmp_path, "good")
    bad = _make_clip(tmp_path, "bad")

    def flaky(audio: bytes, text: str):
        raise RuntimeError("Aligner exploded")

    report, failures = ab.evaluate_candidate("flaky", flaky, [good, bad])
    assert report.clips == 0  # 没有片段成功
    assert len(failures) == 2
    assert all("RuntimeError" in f for f in failures)
    assert all("Aligner exploded" in f for f in failures)


def test_evaluate_candidate_reports_missing_audio_file(tmp_path: Path):
    clip = ab.Clip("gone", tmp_path / "nope.wav", "x", tuple(REF))
    report, failures = ab.evaluate_candidate("x", lambda a, t: list(REF), [clip])
    assert report.clips == 0
    assert len(failures) == 1 and "读不到音频" in failures[0]


def test_evaluate_candidate_mixes_success_and_failure(tmp_path: Path):
    ok_clip = _make_clip(tmp_path, "ok")
    bad_clip = _make_clip(tmp_path, "bad")
    calls = {"n": 0}

    def sometimes(audio: bytes, text: str):
        calls["n"] += 1
        if calls["n"] == 2:
            raise ValueError("second one fails")
        return list(REF)

    report, failures = ab.evaluate_candidate("mixed", sometimes, [ok_clip, bad_clip])
    assert report.clips == 1 and report.usable_clips == 1
    assert len(failures) == 1 and "ValueError" in failures[0]


# ---------- CLI ----------


def test_cli_self_test_passes(capsys):
    assert ab.main(["--self-test"]) == 0
    out = capsys.readouterr().out
    assert "自检通过" in out


def test_cli_requires_manifest_or_self_test():
    with pytest.raises(SystemExit) as ei:
        ab.main([])
    assert ei.value.code == 2


def test_cli_end_to_end_with_plugin_aligner(tmp_path: Path, capsys, monkeypatch):
    """端到端：临时插件模块当对齐器，跑完整 CLI 并写 JSON 报告。"""
    manifest = _write_manifest(
        tmp_path,
        [
            {
                "id": "q01",
                "audio": "q01.wav",
                "text": "你好吗",
                "truth": [
                    {"word": "你", "start_ms": 0, "end_ms": 100},
                    {"word": "好", "start_ms": 100, "end_ms": 200},
                ],
            }
        ],
    )
    (tmp_path / "q01.wav").write_bytes(b"AUDIO")
    plugin = tmp_path / "ab_plugin.py"
    plugin.write_text(
        "from ruipin.ports import WordTiming\n"
        "def make():\n"
        "    def aligner(audio, text):\n"
        "        return [WordTiming('你', 10, 110), WordTiming('好', 110, 210)]\n"
        "    return aligner\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    out_path = tmp_path / "report.json"

    code = ab.main(
        [
            "--manifest",
            str(manifest),
            "--candidate",
            "mod:ab_plugin:make",
            "--out",
            str(out_path),
        ]
    )
    assert code == 0
    printed = capsys.readouterr().out
    assert "覆盖" in printed and "口径" in printed

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["clips"] == 1
    report = payload["reports"][0]
    assert report["name"] == "ab_plugin:make"
    assert report["coverage"] == 1.0
    assert report["start_err_median_ms"] == 10.0
    assert report["end_err_median_ms"] == 10.0
    assert payload["failures"] == {}


def test_cli_reports_unbuildable_aligner(tmp_path: Path, capsys):
    manifest = _write_manifest(
        tmp_path,
        [{"id": "a", "audio": "a.wav", "text": "x", "truth": [{"word": "x", "start_ms": 0, "end_ms": 1}]}],
    )
    assert ab.main(["--manifest", str(manifest), "--candidate", "bogus:thing"]) == 1
    assert "装配对齐器失败" in capsys.readouterr().err


def test_render_table_mentions_reading_discipline():
    """表尾必须带上口径说明：不看口径的数字会被当成可直接横比的标准。"""
    r = ab.compare_clip("c", REF, list(REF))
    table = ab.render_table([ab.aggregate("a", [r])])
    assert "覆盖" in table and "中位数" in table
    assert "低覆盖片段" in table
