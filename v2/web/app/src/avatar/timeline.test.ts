/**
 * viseme 时间轴 → 口型帧映射测试，含三条边界。
 */

import { describe, expect, it } from "vitest";
import { REST_VISEME, sortTimeline, visemeAt } from "./timeline";
import type { VisemeEvent } from "../store/types";

const timeline: VisemeEvent[] = [
  { t_ms: 100, viseme: "rest", weight: 0 },
  { t_ms: 200, viseme: "a", weight: 1 },
  { t_ms: 400, viseme: "o", weight: 0.6 },
];

describe("visemeAt 边界", () => {
  it("空时间轴 → 待机口型", () => {
    expect(visemeAt([], 0)).toEqual({ viseme: REST_VISEME, weight: 1 });
  });

  it("t 早于首帧 → 待机（rest）", () => {
    expect(visemeAt(timeline, 0)).toEqual({ viseme: REST_VISEME, weight: 1 });
    expect(visemeAt(timeline, 99)).toEqual({ viseme: REST_VISEME, weight: 1 });
  });

  it("两帧之间 → 保持前一帧 viseme，权重线性插值", () => {
    // t=150 位于 [100,200] 中段 → 权重 0 + (1-0)*0.5 = 0.5
    expect(visemeAt(timeline, 150)).toEqual({ viseme: "rest", weight: 0.5 });
    // t=300 位于 [200,400] 中段 → 权重 1 + (0.6-1)*0.5 = 0.8
    const mid = visemeAt(timeline, 300);
    expect(mid.viseme).toBe("a");
    expect(mid.weight).toBeCloseTo(0.8);
  });

  it("命中帧首 → 取该帧", () => {
    expect(visemeAt(timeline, 200)).toEqual({ viseme: "a", weight: 1 });
  });

  it("t 超出末帧 → 保持末帧", () => {
    expect(visemeAt(timeline, 5000)).toEqual({ viseme: "o", weight: 0.6 });
  });

  it("权重被夹到 [0,1]", () => {
    const bad: VisemeEvent[] = [
      { t_ms: 0, viseme: "a", weight: 2 },
      { t_ms: 100, viseme: "o", weight: -1 },
    ];
    const at = visemeAt(bad, 50);
    expect(at.weight).toBeGreaterThanOrEqual(0);
    expect(at.weight).toBeLessThanOrEqual(1);
  });

  it("sortTimeline 不改入参并升序", () => {
    const unsorted: VisemeEvent[] = [
      { t_ms: 300, viseme: "o", weight: 1 },
      { t_ms: 100, viseme: "a", weight: 1 },
    ];
    const sorted = sortTimeline(unsorted);
    expect(sorted.map((v) => v.t_ms)).toEqual([100, 300]);
    expect(unsorted[0].t_ms).toBe(300); // 原数组未被改动
  });
});
