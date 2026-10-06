"""Golden Master 契约快照：把"对外契约"冻结成一个字面量常量。

它防什么
--------
它防的是**无意识的口径漂移**：换模型、调权重、改评分链路、重构状态机时，
如果对外可见的产物（状态迁移序列 / 每轮是否计分 / 总分 / 维度 / 生效权重 / 说明文案）
悄悄变了，这里会立刻失败，并打印结构化 diff 告诉你是哪一项变了。

它不是"不许变"——口径当然可以变，但必须**人为确认后**改这个常量，
并在提交信息里写明改了哪条口径（见下方"重新生成快照"）。

被剔除的不确定项（快照里**不出现**）
------------------------------------
1. `session_id` / `turn_id`：业务唯一标识，含会话 id，跨环境不恒定；
   回合用 `turn_index`（会话内序号，语义等价且恒定）标识。
2. 任何时间戳：`StateEvent.ts`、repo 的 `created_at/updated_at/ts`——
   依赖真实时钟，且 `Session.apply()` 内部写死 `time.time()`，无法注入。
3. 任何延迟/成本：`latency_ms`（含 `turn.latency_ms` 与 `eval.latency_ms`）、
   `cost_usd`——依赖真实调度耗时，非契约的一部分。
4. 事件 payload 里的浮点原因串（如 `r=0.320 ≥ 0.25`）——只保留事件名序列，
   payload 只保留"是否计分"这类语义字段。
5. 评估器的 `feedback` 文案——属 provider 自由文本，不属契约。

其余数值一律**显式定标**后入库：`score/dims` 保留 2 位，`effective_weights`
与 `counterfactuals` 保留 6 / 4 位，生理分量保留 4 位——避免浮点末位噪声
把"没变"误报成"变了"，同时不会把真实变更抹平（量级远大于定标精度）。

重新生成快照（仅在人为确认口径变更后才允许）
--------------------------------------------
本模块**绝不会**自动覆写 `EXPECTED_SNAPSHOT`：自动覆写等于让测试永远通过，
Golden Master 就失去意义了。确认变更是有意的之后，手工执行：

    cd E:/项目/锐聘AI/v2
    PYTHONPATH=src ./.venv/Scripts/python.exe tests/golden/contract_snapshot.py

它会打印当前实现产出的规范化快照 JSON。逐项核对确属"有意变更"后，
把打印结果**整段粘贴**覆盖 `EXPECTED_SNAPSHOT`，并在 commit message 里
写明变更了哪条口径、影响哪些历史报告的可比性（口径变更通常需要递增
`rubric_version`，历史报告靠它区分）。
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Optional

from ruipin.adapters.fakes import DeterministicClock, FakeEvaluator
from ruipin.adapters.repo_sqlite import SqliteRepo
from ruipin.domain.states import State
from ruipin.orchestrator import InterviewConfig, InterviewService, TurnScheduler
from ruipin.orchestrator.service import RUBRIC_VERSION, Consent
from ruipin.physio.regulation import DEFAULT_WEIGHTS as PHYSIO_DEFAULT_WEIGHTS
from ruipin.ports import InterviewEvent

# ---------- 固定输入（改动任何一项都会让快照变化，属"有意变更"）----------

SESSION_ID = "golden-1"

QUESTIONS: tuple[str, ...] = (
    "A1：请介绍一个你主导过的项目，以及你在其中解决的关键问题。",
    "A2：线上服务 P99 延迟突然翻倍，你的排查顺序是什么？",
    "A3：描述一次你与同事的技术路线分歧，最后如何收敛。",
    "A4：你如何保证多人协作的模块在迭代中不退化？",
)

BUFFER_QUESTIONS: tuple[str, ...] = ("B1：换个轻松的话题，你最近在读什么书？",)

ANSWERS: tuple[str, ...] = (
    "我主导了推荐召回重构，用向量检索替换倒排，QPS 提升 3 倍，耗时两周。",
    "先按入口、网关、服务、依赖四层看监控，再看 P99 分布与慢查询，最后灰度回滚验证。",
    "我们把分歧拆成两个可验证的假设，用一周的灰度数据决定，选了延迟更低的方案。",
    "靠三层保障：接口契约测试、关键路径基准压测、发版前回归清单。",
)

BUFFER_ANSWER = "最近在读《数据密集型应用系统设计》，刚看完一致性那几章。"

#: 固定六维（刻意互不相同，便于发现维度串位/被抹平）；均值 65.0
EVAL_DIMS: dict[str, float] = {
    "technical": 70.0,
    "communication": 65.0,
    "completeness": 60.0,
    "problem_solving": 68.0,
    "teamwork": 72.0,
    "leadership": 55.0,
}

EVAL_PROVIDER = "golden-fake"
EVAL_CONFIDENCE = 0.9

#: 固定权重：六维等分 0.92 + 生理维度 0.08（W_MAX）
WEIGHTS: dict[str, float] = dict(PHYSIO_DEFAULT_WEIGHTS)

REACTIVITY_THRESHOLD = 0.25

BASELINE_BPM = 68.0
PHYSIO_QUALITY = {"median_snr": 0.9, "reject_ratio": 0.05, "algo_agreement": 1.0}

#: 10 个固定生理事件：(turn_index, pre_bpm, peak_bpm, t50_s)
#: r_i = (peak-pre)/pre 从 0.50 线性降到 0.32，**全部 ≥ 阈值 0.25**，
#: 因此每轮评分后都会插入一次缓冲题（4 轮 → 4 个缓冲回合）。
PHYSIO_EVENT_SPEC: tuple[tuple[int, float, float, float], ...] = tuple(
    (i, 70.0, 70.0 * (1.0 + (0.50 - 0.02 * i)), 10.0 + 0.5 * i) for i in range(10)
)


def physio_events() -> list[InterviewEvent]:
    return [
        InterviewEvent(turn_index=i, pre_bpm=pre, peak_bpm=peak, t50_s=t50)
        for i, pre, peak, t50 in PHYSIO_EVENT_SPEC
    ]


# ---------- 跑一场固定面试 ----------


def _drive(svc: InterviewService) -> None:
    """按固定脚本把整场面试跑完（含缓冲回合）。"""

    async def go() -> None:
        core_i = 0
        while True:
            is_buffer = svc.session.state == State.BUFFER
            question = svc.ask_next()
            if question is None:
                break
            if is_buffer:
                await svc.submit_answer(BUFFER_ANSWER)
            else:
                await svc.submit_answer(ANSWERS[core_i])
                core_i += 1

    asyncio.run(go())


def run_golden_interview(rubric_version: str = RUBRIC_VERSION) -> tuple[SqliteRepo, InterviewService, Any]:
    """跑完整场，返回 (repo, svc, report)。调用方负责 `repo.close()`。"""
    clock = DeterministicClock()
    repo = SqliteRepo(":memory:", clock=clock.now)
    evaluator = FakeEvaluator(
        dims=dict(EVAL_DIMS),
        provider=EVAL_PROVIDER,
        confidence=EVAL_CONFIDENCE,
        feedback="golden feedback",
    )
    scheduler = TurnScheduler(evaluator, clock=clock.monotonic, sleeper=lambda _s: None)
    config = InterviewConfig(
        questions=QUESTIONS,
        buffer_questions=BUFFER_QUESTIONS,
        weights=dict(WEIGHTS),
        adaptive_enabled=True,
        physio_enabled=True,
        reactivity_threshold=REACTIVITY_THRESHOLD,
        rubric_version=rubric_version,
        min_scored_turns=1,
    )
    svc = InterviewService(repo, scheduler, config, session_id=SESSION_ID, clock=clock.now)

    svc.create(meta={"source": "golden-master"})
    svc.record_consent(Consent(base=True, physio=True, media_recording=False))
    svc.greeting_done()
    svc.set_baseline(BASELINE_BPM)
    svc.set_physio_quality(**PHYSIO_QUALITY)
    for ev in physio_events():
        svc.add_physio_event(ev)

    _drive(svc)
    svc.finish_qa()
    svc.close()
    return repo, svc, svc.finalize()


# ---------- 规范化 ----------


def _r(value: Optional[float], nd: int) -> Optional[float]:
    return None if value is None else round(float(value), nd)


def _normalize(repo: SqliteRepo, svc: InterviewService, report: Any) -> dict[str, Any]:
    """把一场面试的对外可见产物规范化成可比较的 dict（剔除不确定项）。"""
    transitions = [
        [f.value if f is not None else None, e.value, t.value]
        for f, e, t in svc.session.history()
    ]

    turns = [
        {
            "turn_index": int(t["turn_index"]),
            "question": str(t["question"]),
            "scored": bool(t["scored"]),
            "degrade_reason": t["degrade_reason"],
        }
        for t in repo.load_turns(SESSION_ID)
    ]

    physio = None
    if report.physio is not None:
        p = report.physio
        physio = {
            "available": bool(p.available),
            "n_valid": int(p.n_valid),
            "reliability": _r(p.reliability, 4),
            "weight_applied": _r(p.weight_applied, 6),
            "score": _r(p.score, 4),
            "components": {k: _r(v, 4) for k, v in sorted(p.components.items())},
        }

    return {
        "transitions": transitions,
        "turns": turns,
        "physio": physio,
        "report": {
            "available": bool(report.available),
            "reason": report.reason,
            "score": _r(report.score, 2),
            "level": report.level,
            "dims": {k: _r(v, 2) for k, v in sorted(report.dims.items())},
            "effective_weights": {
                k: _r(v, 6) for k, v in sorted(report.effective_weights.items())
            },
            "counterfactuals": {
                # NaN 转 None：json.dumps(allow_nan=False) 会拒绝 NaN
                k: (None if v != v else _r(v, 4))
                for k, v in sorted(report.counterfactuals.items())
            },
            "notes": list(report.notes),
            "rubric_version": report.rubric_version,
            "n_scored_turns": int(report.n_scored_turns),
            "n_buffer_turns": int(report.n_buffer_turns),
        },
    }


def build_snapshot(rubric_version: str = RUBRIC_VERSION) -> dict[str, Any]:
    """跑一场固定面试并产出规范化快照。"""
    repo, svc, report = run_golden_interview(rubric_version)
    try:
        return _normalize(repo, svc, report)
    finally:
        repo.close()


# ---------- 结构化 diff ----------


def diff_snapshot(
    expected: Any, actual: Any, path: str = ""
) -> list[str]:
    """逐键比对两份快照，返回人类可读的差异行（空列表表示一致）。"""
    out: list[str] = []
    if isinstance(expected, dict) and isinstance(actual, dict):
        for key in sorted(set(expected) | set(actual)):
            sub = f"{path}/{key}" if path else f"/{key}"
            if key not in expected:
                out.append(f"+ 实得多出键 {sub} = {actual[key]!r}")
            elif key not in actual:
                out.append(f"- 实得缺失键 {sub}（期望 {expected[key]!r}）")
            else:
                out.extend(diff_snapshot(expected[key], actual[key], sub))
        return out

    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            out.append(f"~ {path or '/'}: 列表长度 {len(expected)} → {len(actual)}")
        for i, (a, b) in enumerate(zip(expected, actual)):
            out.extend(diff_snapshot(a, b, f"{path}[{i}]"))
        return out

    if expected != actual:
        out.append(f"~ {path or '/'}: 期望 {expected!r} → 实得 {actual!r}")
    return out


# ---------- 冻结的期望值 ----------
# ⚠️ 人工确认口径变更后方可更新；更新方法见本文件顶部 docstring。

EXPECTED_SNAPSHOT: dict[str, Any] = {
    "physio": {
        "available": True,
        "components": {
            "S_consistency": 95.0,
            "S_habituation": 95.0,
            "S_recovery": 68.75,
        },
        "n_valid": 10,
        "reliability": 0.9575,
        "score": 79.25,
        "weight_applied": 0.08,
    },
    "report": {
        "available": True,
        "counterfactuals": {
            "communication": 66.3465,
            "completeness": 67.252,
            "leadership": 68.1575,
            "problem_solving": 65.8031,
            "stress_regulation": 65.0,
            "technical": 65.4409,
            "teamwork": 65.0787,
        },
        "dims": {
            "communication": 65.0,
            "completeness": 60.0,
            "leadership": 55.0,
            "problem_solving": 68.0,
            "stress_regulation": 79.25,
            "teamwork": 72.0,
            "technical": 70.0,
        },
        "effective_weights": {
            "communication": 0.153333,
            "completeness": 0.153333,
            "leadership": 0.153333,
            "problem_solving": 0.153333,
            "stress_regulation": 0.08,
            "teamwork": 0.153333,
            "technical": 0.153333,
        },
        "level": "C",
        "notes": [
            "所有维度可靠性达标，未发生权重重分配",
            "生理维度已计入：可靠性 R=0.96，实得权重 8.00%（上限 8%）",
        ],
        "n_buffer_turns": 4,
        "n_scored_turns": 4,
        "reason": None,
        "rubric_version": "rubric-v1",
        "score": 66.14,
    },
    "transitions": [
        ["idle", "create", "setup"],
        ["setup", "consent_ok", "greeting"],
        ["greeting", "greeting_done", "asking"],
        ["asking", "tts_done", "listening"],
        ["listening", "answer_commit", "processing"],
        ["processing", "eval_done", "asking"],
        ["asking", "insert_buffer", "buffer"],
        ["buffer", "tts_done", "buffer_listening"],
        ["buffer_listening", "buffer_done", "asking"],
        ["asking", "tts_done", "listening"],
        ["listening", "answer_commit", "processing"],
        ["processing", "eval_done", "asking"],
        ["asking", "insert_buffer", "buffer"],
        ["buffer", "tts_done", "buffer_listening"],
        ["buffer_listening", "buffer_done", "asking"],
        ["asking", "tts_done", "listening"],
        ["listening", "answer_commit", "processing"],
        ["processing", "eval_done", "asking"],
        ["asking", "insert_buffer", "buffer"],
        ["buffer", "tts_done", "buffer_listening"],
        ["buffer_listening", "buffer_done", "asking"],
        ["asking", "tts_done", "listening"],
        ["listening", "answer_commit", "processing"],
        ["processing", "eval_done", "asking"],
        ["asking", "insert_buffer", "buffer"],
        ["buffer", "tts_done", "buffer_listening"],
        ["buffer_listening", "buffer_done", "asking"],
        ["asking", "no_more_questions", "candidate_qa"],
        ["candidate_qa", "qa_done", "closing"],
        ["closing", "closing_done", "reporting"],
        ["reporting", "report_done", "completed"],
    ],
    "turns": [
        {
            "degrade_reason": None,
            "question": "A1：请介绍一个你主导过的项目，以及你在其中解决的关键问题。",
            "scored": True,
            "turn_index": 0,
        },
        {
            "degrade_reason": None,
            "question": "B1：换个轻松的话题，你最近在读什么书？",
            "scored": False,
            "turn_index": 1,
        },
        {
            "degrade_reason": None,
            "question": "A2：线上服务 P99 延迟突然翻倍，你的排查顺序是什么？",
            "scored": True,
            "turn_index": 2,
        },
        {
            "degrade_reason": None,
            "question": "B1：换个轻松的话题，你最近在读什么书？",
            "scored": False,
            "turn_index": 3,
        },
        {
            "degrade_reason": None,
            "question": "A3：描述一次你与同事的技术路线分歧，最后如何收敛。",
            "scored": True,
            "turn_index": 4,
        },
        {
            "degrade_reason": None,
            "question": "B1：换个轻松的话题，你最近在读什么书？",
            "scored": False,
            "turn_index": 5,
        },
        {
            "degrade_reason": None,
            "question": "A4：你如何保证多人协作的模块在迭代中不退化？",
            "scored": True,
            "turn_index": 6,
        },
        {
            "degrade_reason": None,
            "question": "B1：换个轻松的话题，你最近在读什么书？",
            "scored": False,
            "turn_index": 7,
        },
    ],
}


if __name__ == "__main__":
    print(
        json.dumps(
            build_snapshot(),
            ensure_ascii=False,
            indent=4,
            sort_keys=True,
        )
    )
