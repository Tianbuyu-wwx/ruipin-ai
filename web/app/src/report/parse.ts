/**
 * 报告解析与"生理红线"呈现逻辑（纯函数）。
 *
 * 后端 `orchestrator/service.py::Report.to_dict()` 输出 **snake_case**，
 * 前端统一转 camelCase 视图。两条红线在这里被物理保证：
 *   1. `available=false` ⇒ `score` 必为 null（后端如此，前端再兜一道，防止脏 payload 造数）；
 *   2. 生理维度 `available=false` ⇒ **绝不产出任何数字**，只产出"未计入（原因：…）"文案。
 *      （对应后端"测不准 = 不计入而非给低分"的口径。）
 */

import type { PhysioResult, ReportView } from "../store/types";

// ------------------------------------------------------------------ 取值助手

function isObj(v: unknown): v is Record<string, unknown> {
  return v !== null && typeof v === "object" && !Array.isArray(v);
}

function numOrNull(v: unknown): number | null {
  if (typeof v === "number" && Number.isFinite(v)) return v;
  return null;
}

function numOrZero(v: unknown): number {
  return numOrNull(v) ?? 0;
}

function strOrNull(v: unknown): string | null {
  return typeof v === "string" ? v : null;
}

function boolOr(v: unknown, dflt: boolean): boolean {
  return typeof v === "boolean" ? v : dflt;
}

/** `Record<string, number>`：保留有限数值；非数值以 0 计入（仅用于展示，不作为分数）。 */
function numMap(v: unknown): Record<string, number> {
  const out: Record<string, number> = {};
  if (!isObj(v)) return out;
  for (const [k, val] of Object.entries(v)) out[k] = numOrZero(val);
  return out;
}

/** 反事实表：`NaN`（后端写 null）原样保留为 null，不填 0。 */
function cfMap(v: unknown): Record<string, number | null> {
  const out: Record<string, number | null> = {};
  if (!isObj(v)) return out;
  for (const [k, val] of Object.entries(v)) out[k] = numOrNull(val);
  return out;
}

function strList(v: unknown): string[] {
  if (!Array.isArray(v)) return [];
  return v.filter((x): x is string => typeof x === "string");
}

// ------------------------------------------------------------------ physio

export function parsePhysio(v: unknown): PhysioResult | null {
  if (!isObj(v)) return null;
  const available = boolOr(v["available"], false);
  const rawScore = numOrNull(v["score"]);
  // 红线：不可用即不携带分数。
  const score = available ? rawScore : null;
  return {
    available,
    reason: strOrNull(v["reason"]),
    score,
    reliability: numOrZero(v["reliability"]),
    weightApplied: numOrZero(v["weight_applied"]),
    nValid: numOrZero(v["n_valid"]),
    baselineBpm: numOrNull(v["baseline_bpm"]),
    t50MedianS: numOrNull(v["t50_median_s"]),
    habituationSlope: numOrNull(v["habituation_slope"]),
    reactivityRatioMean: numOrNull(v["reactivity_ratio_mean"]),
    consistencySigma: numOrNull(v["consistency_sigma"]),
    components: numMap(v["components"]),
  };
}

// ------------------------------------------------------------------ report

/**
 * 解析 `report.ready` 的 payload → `ReportView`。
 *
 * 兼容两种形状：payload 即报告字典，或 `payload.report` 为报告字典。
 */
export function parseReport(payload: Record<string, unknown>): ReportView {
  const src = isObj(payload["report"]) ? (payload["report"] as Record<string, unknown>) : payload;
  const available = boolOr(src["available"], false);
  const rawScore = numOrNull(src["score"]);
  return {
    sessionId: strOrNull(src["session_id"]) ?? "",
    available,
    reason: strOrNull(src["reason"]),
    // 红线：不可用 ⇒ 不给分（后端已保证，这里再兜一道）。
    score: available ? rawScore : null,
    level: strOrNull(src["level"]),
    dims: numMap(src["dims"]),
    effectiveWeights: numMap(src["effective_weights"]),
    redistributed: numMap(src["redistributed"]),
    counterfactuals: cfMap(src["counterfactuals"]),
    notes: strList(src["notes"]),
    physio: parsePhysio(src["physio"]),
    rubricVersion: strOrNull(src["rubric_version"]) ?? "",
    nScoredTurns: numOrZero(src["n_scored_turns"]),
    nBufferTurns: numOrZero(src["n_buffer_turns"]),
  };
}

// ------------------------------------------------------------------ 呈现文案

/** 文字是否携带数字（用于"生理不可用时不得出现数字"的断言与自检）。 */
export function hasDigits(s: string): boolean {
  return /[0-9]/.test(s);
}

/**
 * 生理维度的呈现文案。**不可用时绝不出现任何数字**。
 *
 * @param physio 解析后的生理结果（可为 null = 本场未启用）
 * @param enabled 本场是否启用了生理模块（用于区分"未启用"与"启用但未计入"）
 */
export function physioScoreText(physio: PhysioResult | null, enabled: boolean): string {
  if (physio === null) {
    return enabled ? "未计入（原因：未采集到生理数据）" : "未启用（本场未采集生理信号）";
  }
  if (!physio.available || physio.score === null) {
    const reason = physio.reason ?? "信号质量不足";
    return `未计入（原因：${reason}）`;
  }
  return `已计入：${physio.score.toFixed(1)} 分（可靠性 R=${physio.reliability.toFixed(2)}，实得权重 ${(
    physio.weightApplied * 100
  ).toFixed(1)}%）`;
}

/** 总分呈现：不可用时明确"不出分"，不显示数字。 */
export function reportScoreText(report: ReportView): string {
  if (!report.available || report.score === null) {
    return `本次未出分：${report.reason ?? "无有效评估"}`;
  }
  return `${report.score.toFixed(1)} 分`;
}

/**
 * 反事实呈现："若完全剔除某维度，总分将为 X"。
 * 值为 null（后端 NaN）时显示"无法计算"，**不显示数字**。
 */
export function counterfactualText(value: number | null | undefined): string {
  if (value === null || value === undefined) return "无法计算（剔除后无可用维度）";
  return `${value.toFixed(1)} 分`;
}
