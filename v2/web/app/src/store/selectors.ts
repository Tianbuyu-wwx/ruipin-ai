/**
 * store 派生状态的只读选择器 / 文案构造（纯函数，供 UI 与测试共用）。
 *
 * 这里承担一条硬要求：**降级必须让用户看得见**（方案 §3.4 红线）。
 * `degradationBanner` 保证只要 `degradation.level > 0`，就一定产出一段**非空**文案，
 * 不存在"降级了但界面没话说"的静默路径。
 */

import type { DerivedState } from "./types";

export interface DegradationBannerView {
  visible: boolean;
  level: number;
  /** 保证非空（visible 时）：来自服务端 message，缺失则本地兜底生成。 */
  text: string;
  /** 各级徽标文案（去重，按等级升序）。 */
  badges: string[];
}

export function degradationBanner(d: DerivedState): DegradationBannerView {
  const level = d.degradation.level;
  const badges: string[] = [];
  const seen = new Set<string>();
  for (const b of [...d.degradation.badges].sort((a, b) => a.level - b.level)) {
    if (b.badge && !seen.has(b.badge)) {
      seen.add(b.badge);
      badges.push(b.badge);
    }
  }
  if (level <= 0) {
    return { visible: false, level: 0, text: "", badges };
  }
  const cur = d.degradation.current;
  const text =
    (cur?.message && cur.message.trim()) ||
    (badges.length > 0 ? badges.join("；") : `服务已降级（等级 L${level}）`);
  return { visible: true, level, text, badges };
}

/** 当前题目是否有计时/是否为缓冲题等展示所需的信息汇总。 */
export function questionSummary(d: DerivedState): { text: string; scored: boolean } | null {
  if (!d.question) return null;
  return { text: d.question.text, scored: d.question.scored };
}
