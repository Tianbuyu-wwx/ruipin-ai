// index.ts — 端侧 rPPG 信号链统一导出入口
//
// 分层：
//   signal.ts   纯算法（去趋势 / 带通 / CHROM·POS·SSR / Welch PSD / 心率估计）
//   quality.ts  窗口质量（SNR / 三算法投票 / 弃权 / AE 漂移）
//   pipeline.ts 有状态滑动窗口管线 + 事件打点
//
// 规范：docs/模块详设-心率与压力调节评估.md §3、§7。

export * from './signal.ts';
export * from './quality.ts';
export * from './pipeline.ts';
