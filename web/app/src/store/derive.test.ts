/**
 * `derive(events)` 的确定性与去重测试（方案 §2.3 事件溯源）。
 */

import { describe, expect, it } from "vitest";
import { derive, emptyDerivedState } from "./derive";
import { FsmState, type ServerEvent } from "./types";

function env(seq: number, type: string, payload: Record<string, unknown>, ts = seq * 1000): ServerEvent {
  return { kind: "envelope", seq, ts, sessionId: "s-1", type, payload };
}

const sampleEvents: ServerEvent[] = [
  env(1, "state.changed", { from: "idle", to: "setup", reason: "create" }),
  env(2, "state.changed", { from: "setup", to: "greeting", reason: "consent_ok" }),
  env(3, "question.start", { turn_id: "s-1-t0", turn_index: 0, text: "自我介绍", scored: true }),
  env(4, "transcript.partial", { text: "我叫" }),
  env(5, "transcript.final", { turn_id: "s-1-t0", text: "我叫小明，做后端" }),
  env(6, "eval.done", {
    turn_id: "s-1-t0",
    dims: { technical: 80, communication: 88, completeness: 75, problem_solving: 82, teamwork: 70, leadership: 65 },
    score: 79.3,
    feedback: "结构清晰",
    provider: "llm",
    confidence: 0.82,
    degraded: false,
    latency_ms: 2100,
  }),
  env(7, "metric.hr", { ts: 7000, bpm: 76, confidence: 0.9 }),
  env(8, "metric.hr", { ts: 8000, bpm: null, confidence: 0.2, reason: "信号不足" }),
];

describe("derive 确定性", () => {
  it("同一序列两次 derive 结果 deepEqual", () => {
    expect(derive(sampleEvents)).toEqual(derive(sampleEvents));
  });

  it("空事件 → 初始态", () => {
    expect(derive([])).toEqual(emptyDerivedState());
  });
});

describe("seq 去重（断线重放幂等）", () => {
  it("重复 seq 被去重，不破坏状态", () => {
    const once = derive(sampleEvents);
    // 用"原序列 + 完整重放一遍"模拟重连重放。
    const replayed = derive([...sampleEvents, ...sampleEvents]);
    expect(replayed).toEqual(once);
  });

  it("重放后仍需处理新增事件（不误伤新 seq）", () => {
    const once = derive(sampleEvents);
    const after = derive([...sampleEvents, ...sampleEvents, env(9, "state.changed", { from: "asking", to: "listening", reason: "tts_done" })]);
    expect(after.fsm.state).toBe(FsmState.LISTENING);
    expect(after.fsm.history.length).toBe(once.fsm.history.length + 1);
  });
});

describe("轮次归属与序号（踩过的坑）", () => {
  it("连出三题：状态栏序号必须逐题前进，不能停在「第 1 题」", () => {
    // 回归：`question.start` 缺 `turn_index` 时，兜底值原先取 `s.fsm.turnIdx`，
    // 而它只在收到 question.start 时才被写——于是每题都算出同一个号。
    const d = derive([
      env(1, "question.start", { turn_id: "s-1-t0", text: "题一", scored: true }),
      env(2, "question.start", { turn_id: "s-1-t1", text: "题二", scored: true }),
      env(3, "question.start", { turn_id: "s-1-t2", text: "题三", scored: true }),
    ]);
    expect(Object.values(d.turns).map((t) => t.turnIndex)).toEqual([0, 1, 2]);
    expect(d.question?.turnIndex).toBe(2);
  });

  it("eval.done 只带 turn_index 时按序号挂回对应回合（不造幽灵回合）", () => {
    const d = derive([
      env(1, "question.start", { turn_id: "s-1-t0", turn_index: 0, text: "题一" }),
      env(2, "question.start", { turn_id: "s-1-t1", turn_index: 1, text: "题二" }),
      env(3, "eval.done", { turn_index: 1, score: 80, dims: {}, feedback: "ok" }),
    ]);
    expect(Object.keys(d.turns)).toHaveLength(2);
    expect(d.turns["s-1-t1"].eval?.score).toBe(80);
    expect(d.turns["s-1-t0"].eval).toBeNull();
  });

  it("eval.done 无法归属到任何一轮：显式记错，绝不新建空题干回合", () => {
    // 凭空插一条键为空的回合，会让逐轮评估多出一行没有题目的幽灵记录，
    // 真正的评估全被挤到它上面——这比丢一条评估视图更坏。
    const d = derive([env(1, "eval.done", { score: 80, dims: {}, feedback: "ok" })]);
    expect(Object.keys(d.turns)).toHaveLength(0);
    expect(d.errors[d.errors.length - 1]?.message).toContain("无法归属");
  });
});

describe("派生态内容", () => {
  it("fsm 迁移被记录，question/turn/eval 正确派生", () => {
    const d = derive(sampleEvents);
    expect(d.sessionId).toBe("s-1");
    expect(d.fsm.state).toBe(FsmState.GREETING);
    expect(d.question?.text).toBe("自我介绍");
    expect(d.turns["s-1-t0"].eval?.score).toBeCloseTo(79.3);
    expect(d.turns["s-1-t0"].finalTranscript).toBe("我叫小明，做后端");
  });

  it("转写 partial→final 覆盖", () => {
    const d = derive(sampleEvents);
    expect(d.transcript.partial).toBe("");
    expect(d.transcript.finals).toEqual(["我叫小明，做后端"]);
  });

  it("心率 low-confidence 的 bpm=null 被保留（测不准 ≠ 0）", () => {
    const d = derive(sampleEvents);
    expect(d.hr.samples).toHaveLength(2);
    expect(d.hr.latest?.bpm).toBeNull();
    expect(d.hr.samples[0].bpm).toBe(76);
  });

  it("降级取已发生事件的最高等级", () => {
    const d = derive([
      env(1, "degradation.changed", { level: 2, reason: "scoring_failed", badge: "规则评分", message: "已降级" }),
      env(2, "degradation.changed", { level: 1, reason: "vision_timeout", badge: "无视觉", message: "视觉关闭" }),
    ]);
    expect(d.degradation.level).toBe(2);
    expect(d.degradation.current?.level).toBe(2);
  });

  it("会话超时被标记并记入错误", () => {
    const d = derive([env(1, "session.timeout", { code: "session_timeout", message: "空闲超时" })]);
    expect(d.timedOut).toBe(true);
    expect(d.errors[0].code).toBe("session_timeout");
  });

  it("二进制 TTS 块与 viseme 事件计入派生", () => {
    const d = derive([
      { kind: "tts_chunk", seq: 0, ts: 1, sessionId: "s-1", byteLength: 1024 },
      {
        kind: "viseme",
        seq: 0,
        ts: 2,
        sessionId: "s-1",
        visemes: [
          { t_ms: 0, viseme: "rest", weight: 1 },
          { t_ms: 120, viseme: "a", weight: 0.9 },
        ],
      },
    ]);
    expect(d.tts).toEqual({ chunks: 1, bytes: 1024 });
    expect(d.visemes).toHaveLength(2);
    expect(d.visemes[1].viseme).toBe("a");
  });
});
