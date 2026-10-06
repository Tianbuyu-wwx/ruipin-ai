/**
 * 单一 Zustand store（方案 §2.3）：**只由 WS 事件驱动**。
 *
 * 纪律：store 保存 **append-only 的原始事件数组**，UI 读的 `derived` 一律由
 * 纯函数 `derive(events)` 现算——不在 store 里做"就地改字段"的增量写。
 * 这样断线重连把重放事件再灌一遍，去重后状态与断线前完全一致（见 derive 的 seq 去重）。
 */

import { create } from "zustand";
import { derive, emptyDerivedState } from "./derive";
import type { DerivedState, ServerEvent } from "./types";

export interface InterviewStore {
  /** 追加式事件日志（原始，可能含重放重复；去重发生在 derive 内）。 */
  events: ServerEvent[];
  /** 派生视图（UI 只读此处）。 */
  derived: DerivedState;
  /** 追加一条事件并重算派生状态。 */
  appendEvent: (ev: ServerEvent) => void;
  /** 批量追加（重连重放用）。 */
  appendEvents: (evs: readonly ServerEvent[]) => void;
  /** 清空（新会话）。 */
  reset: () => void;
}

export const useInterviewStore = create<InterviewStore>((set) => ({
  events: [],
  derived: emptyDerivedState(),
  appendEvent: (ev) =>
    set((s) => {
      const events = [...s.events, ev];
      return { events, derived: derive(events) };
    }),
  appendEvents: (evs) =>
    set((s) => {
      const events = [...s.events, ...evs];
      return { events, derived: derive(events) };
    }),
  reset: () => set({ events: [], derived: emptyDerivedState() }),
}));

/** 非 React 环境（net 层、测试）读取当前派生状态的便捷入口。 */
export function currentDerived(): DerivedState {
  return useInterviewStore.getState().derived;
}
