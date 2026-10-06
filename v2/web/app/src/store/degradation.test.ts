/**
 * 降级横幅"不静默"测试（方案 §3.4 红线：降级必须让用户看见）。
 */

import { describe, expect, it } from "vitest";
import { derive } from "./derive";
import { degradationBanner } from "./selectors";
import type { ServerEvent } from "./types";

function env(seq: number, type: string, payload: Record<string, unknown>): ServerEvent {
  return { kind: "envelope", seq, ts: seq * 1000, sessionId: "s-1", type, payload };
}

describe("degradationBanner", () => {
  it("无降级事件时不可见且无文案", () => {
    const view = degradationBanner(derive([]));
    expect(view.visible).toBe(false);
    expect(view.text).toBe("");
  });

  it("收到 degradation.changed 后 store 出现可见降级状态且文案非空", () => {
    const d = derive([
      env(1, "degradation.changed", {
        level: 2,
        reason: "scoring_failed",
        badge: "本轮为规则评分，仅供参考",
        message: "服务已降级至 L2：本轮为规则评分，仅供参考（原因：scoring_failed）",
      }),
    ]);
    const view = degradationBanner(d);
    expect(view.visible).toBe(true);
    expect(view.level).toBe(2);
    expect(view.text.trim().length).toBeGreaterThan(0);
    expect(view.text).toContain("L2");
    expect(view.badges).toContain("本轮为规则评分，仅供参考");
  });

  it("服务端未给 message 时本地兜底文案仍非空（杜绝静默）", () => {
    const d = derive([env(1, "degradation.changed", { level: 4, reason: "tts_unavailable", badge: "", message: "" })]);
    const view = degradationBanner(d);
    expect(view.visible).toBe(true);
    expect(view.text.trim().length).toBeGreaterThan(0);
    expect(view.text).toContain("L4");
  });
});
