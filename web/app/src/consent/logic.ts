/**
 * 授权（同意）逻辑 —— 纯函数，独立于 DOM，便于单测。
 *
 * 合规红线（PIPL 第 28 条 + 方案 §12.6）：
 * 1. **生理信号属敏感个人信息，必须单独同意**——`physiology` 是独立开关，
 *    不能用"同意参加面试"或"同意用摄像头"一揽子带过。
 * 2. **拒绝任何一项都不能剥夺参加面试的权利**——只要 `base`（参加面试）为真即可开始；
 *    摄像头/生理/屏幕都是可选增强，对应"权重重分配"（未采集维度不计入、不压低总分）。
 * 3. 生理信号靠 rPPG 从摄像头画面提取，**camera 是物理前提**：
 *    拒绝 camera ⇒ physiology 自动关闭，并给出可读原因（不是静默置灰）。
 */

import type { ConsentGrantPayload } from "../protocol/messages";

export type ConsentKey = "base" | "camera" | "physiology" | "screen";

export interface ConsentFlags {
  /** 参加面试本身（唯一必需项）。 */
  base: boolean;
  /** 摄像头采集（可选增强；rPPG 的物理前提）。 */
  camera: boolean;
  /** 生理信号分析（敏感个人信息，需单独同意）。 */
  physiology: boolean;
  /** 屏幕共享（可选增强）。 */
  screen: boolean;
}

export const DEFAULT_CONSENT: ConsentFlags = {
  base: false,
  camera: false,
  physiology: false,
  screen: false,
};

/**
 * 切换某一项授权，并施加依赖约束：
 * - 关闭 camera ⇒ 强制关闭 physiology（失去物理前提）；
 * - 开启 physiology ⇒ 自动开启 camera（生理分析需要画面），避免出现"想要但不成立"的状态。
 */
export function applyConsentToggle(flags: ConsentFlags, key: ConsentKey, value: boolean): ConsentFlags {
  const next: ConsentFlags = { ...flags, [key]: value };
  if (key === "camera" && !value) next.physiology = false;
  if (key === "physiology" && value) next.camera = true;
  return next;
}

/** 生理项当前是否真正生效（需要 camera 且显式同意）。 */
export function isPhysiologyActive(flags: ConsentFlags): boolean {
  return flags.camera && flags.physiology;
}

/** 生理项不可用/被禁用时给用户看的原因；可用时返回 null。 */
export function physiologyConstraint(flags: ConsentFlags): string | null {
  if (!flags.camera) {
    return "未开启摄像头：生理信号（rPPG）从面部画面提取，缺少摄像头时不可用。不开摄像头也可完成面试，分数照常可比。";
  }
  return null;
}

/** 是否可以开始面试（只取决于 base）。 */
export function canStartInterview(flags: ConsentFlags): boolean {
  return flags.base;
}

/** 转成发往服务端的 `consent.grant` payload（生理项按"真正生效"口径收敛）。 */
export function toConsentPayload(
  flags: ConsentFlags,
  extra: { mediaRecording?: boolean } = {},
): ConsentGrantPayload {
  return {
    base: flags.base,
    camera: flags.camera,
    physiology: isPhysiologyActive(flags),
    screen: flags.screen,
    media_recording: extra.mediaRecording ?? false,
  };
}
