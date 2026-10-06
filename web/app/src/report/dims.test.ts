/**
 * 维度定义测试（顺序即报告呈现顺序，与 Python ports.py 一致）。
 */

import { describe, expect, it } from "vitest";
import { DIMENSIONS, dimLabel, DIM_LABEL, PHYSIO_DIM } from "./dims";

describe("评分维度", () => {
  it("六维顺序与 Python ports.DIMENSIONS 一致", () => {
    expect([...DIMENSIONS]).toEqual([
      "technical",
      "communication",
      "completeness",
      "problem_solving",
      "teamwork",
      "leadership",
    ]);
  });

  it("生理维度名与 PHYSIO_DIM 一致", () => {
    expect(PHYSIO_DIM).toBe("stress_regulation");
    expect(DIM_LABEL[PHYSIO_DIM]).toContain("生理");
  });

  it("dimLabel 有中文名，未知维度回退为原名", () => {
    expect(dimLabel("technical")).toBe("技术能力");
    expect(dimLabel("unknown_dim")).toBe("unknown_dim");
  });
});
