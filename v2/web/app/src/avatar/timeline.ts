/**
 * viseme 时间轴 → 口型帧映射（方案 §6.5）。
 *
 * 纯函数、不读时钟：给定时间轴与"当前音频时间" `tMs`（来自 `AudioContext.currentTime`
 * 的统一音频时钟，非墙钟），返回该时刻应呈现的口型与张开强度。
 *
 * 边界语义（单测覆盖三条边界）：
 *   - t 早于首帧：待机口型（REST，嘴闭合），而不是"没有口型"；
 *   - 两帧之间：保持前一帧的 viseme，权重在前后帧之间**线性插值**（避免机械跳变，
 *     方案要求 60–120ms 过渡）；
 *   - t 超出末帧：保持末帧（句子结束后停在最后一个口型，下一次时间轴到达时替换）。
 */

import type { VisemeEvent } from "../store/types";

/** 待机口型（嘴自然闭合）。 */
export const REST_VISEME = "rest";

export interface VisemeFrame {
  viseme: string;
  weight: number;
}

/** 按 t_ms 升序排序（返回新数组，不改入参）。 */
export function sortTimeline(timeline: readonly VisemeEvent[]): VisemeEvent[] {
  return [...timeline].sort((a, b) => a.t_ms - b.t_ms);
}

/**
 * 求 `tMs` 时刻的口型帧。
 *
 * @param timeline 已按 t_ms 升序的 viseme 时间轴
 * @param tMs 当前音频时间（ms，相对音频起点）
 */
export function visemeAt(timeline: readonly VisemeEvent[], tMs: number): VisemeFrame {
  if (timeline.length === 0) return { viseme: REST_VISEME, weight: 1 };

  const first = timeline[0];
  if (tMs < first.t_ms) return { viseme: REST_VISEME, weight: 1 };

  const last = timeline[timeline.length - 1];
  if (tMs >= last.t_ms) return { viseme: last.viseme, weight: clamp01(last.weight ?? 1) };

  // 二分/线性查找：最后一个 t_ms <= tMs 的帧。
  for (let i = 0; i < timeline.length - 1; i += 1) {
    const cur = timeline[i];
    const nxt = timeline[i + 1];
    if (tMs >= cur.t_ms && tMs < nxt.t_ms) {
      const span = nxt.t_ms - cur.t_ms;
      const frac = span <= 0 ? 0 : (tMs - cur.t_ms) / span;
      const w0 = clamp01(cur.weight ?? 1);
      const w1 = clamp01(nxt.weight ?? 1);
      return { viseme: cur.viseme, weight: w0 + (w1 - w0) * frac };
    }
  }

  // 理论不可达（上面已处理 t >= last），兜底返回末帧。
  return { viseme: last.viseme, weight: clamp01(last.weight ?? 1) };
}

function clamp01(x: number): number {
  if (!Number.isFinite(x)) return 0;
  if (x < 0) return 0;
  if (x > 1) return 1;
  return x;
}
