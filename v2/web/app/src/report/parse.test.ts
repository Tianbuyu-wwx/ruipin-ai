/**
 * 报告解析与"生理不可用不得出现数字"红线测试。
 */

import { describe, expect, it } from "vitest";
import {
  counterfactualText,
  hasDigits,
  parseReport,
  physioScoreText,
  reportScoreText,
} from "./parse";

const baseReport = {
  session_id: "s-1",
  available: true,
  reason: null,
  score: 78.5,
  level: "B",
  dims: {
    technical: 80,
    communication: 88,
    completeness: 75,
    problem_solving: 82,
    teamwork: 70,
    leadership: 65,
  },
  effective_weights: { technical: 0.28, communication: 0.14 },
  redistributed: { technical: 0.01 },
  counterfactuals: { stress_regulation: 77.9, technical: 76.1 },
  notes: ["所有维度可靠性达标，未发生权重重分配"],
  physio: {
    available: true,
    reason: null,
    score: 66.2,
    reliability: 0.81,
    weight_applied: 0.08,
    n_valid: 9,
    components: { S_recovery: 70, S_habituation: 60, S_consistency: 55 },
  },
  rubric_version: "rubric-v1",
  n_scored_turns: 4,
  n_buffer_turns: 1,
};

describe("parseReport", () => {
  it("snake_case → camelCase 映射正确", () => {
    const r = parseReport(baseReport as unknown as Record<string, unknown>);
    expect(r.sessionId).toBe("s-1");
    expect(r.effectiveWeights.technical).toBeCloseTo(0.28);
    expect(r.counterfactuals.stress_regulation).toBeCloseTo(77.9);
    expect(r.rubricVersion).toBe("rubric-v1");
    expect(r.nScoredTurns).toBe(4);
    expect(r.nBufferTurns).toBe(1);
    expect(r.physio?.weightApplied).toBeCloseTo(0.08);
    expect(r.physio?.nValid).toBe(9);
  });

  it("available=false ⇒ score 强制为 null（红线兜底）", () => {
    const dirty = { ...baseReport, available: false, score: 88 };
    const r = parseReport(dirty as unknown as Record<string, unknown>);
    expect(r.available).toBe(false);
    expect(r.score).toBeNull();
    expect(hasDigits(reportScoreText(r))).toBe(false);
  });

  it("兼容 payload 嵌套在 report 字段下", () => {
    const r = parseReport({ report: baseReport } as unknown as Record<string, unknown>);
    expect(r.sessionId).toBe("s-1");
  });
});

describe("生理维度红线", () => {
  it("available=false 时呈现文案不含任何数字", () => {
    const r = parseReport({
      ...baseReport,
      physio: { available: false, reason: "可靠性 R 不足：信号质量差", score: null, reliability: 0, weight_applied: 0, n_valid: 2, components: {} },
    } as unknown as Record<string, unknown>);
    const text = physioScoreText(r.physio, true);
    expect(text).toContain("未计入");
    expect(hasDigits(text)).toBe(false);
    // score 被收敛为 null。
    expect(r.physio?.score).toBeNull();
  });

  it("physio 为 null（本场未启用）也不出现数字", () => {
    expect(hasDigits(physioScoreText(null, false))).toBe(false);
    expect(hasDigits(physioScoreText(null, true))).toBe(false);
  });

  it("可用时才显示分数", () => {
    const r = parseReport(baseReport as unknown as Record<string, unknown>);
    const text = physioScoreText(r.physio, true);
    expect(hasDigits(text)).toBe(true);
    expect(text).toContain("66.2");
  });

  it("反事实 NaN（后端写 null）显示'无法计算'而非数字", () => {
    expect(hasDigits(counterfactualText(null))).toBe(false);
    expect(counterfactualText(undefined)).toContain("无法计算");
    expect(counterfactualText(77.9)).toContain("77.9");
  });
});
