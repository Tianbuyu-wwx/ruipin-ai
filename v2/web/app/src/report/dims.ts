/**
 * 评分维度定义（与 Python `ports.py::DIMENSIONS` / `PHYSIO_DIM` 一致）。
 * 顺序即报告呈现顺序——后端聚合器的权重重分配也依赖此顺序。
 */

export const DIMENSIONS = [
  "technical",
  "communication",
  "completeness",
  "problem_solving",
  "teamwork",
  "leadership",
] as const;

export type Dimension = (typeof DIMENSIONS)[number];

/** 生理维度名（`ports.py::PHYSIO_DIM`），权重上限 8%。 */
export const PHYSIO_DIM = "stress_regulation";

export const DIM_LABEL: Record<string, string> = {
  technical: "技术能力",
  communication: "沟通表达",
  completeness: "完整度",
  problem_solving: "问题解决",
  teamwork: "团队协作",
  leadership: "领导力",
  [PHYSIO_DIM]: "应激恢复（生理）",
};

export function dimLabel(key: string): string {
  return DIM_LABEL[key] ?? key;
}
