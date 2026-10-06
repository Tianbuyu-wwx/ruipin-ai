/**
 * 授权逻辑测试（PIPL 第 28 条：生理单独同意；拒绝任一增强不剥夺面试权利）。
 */

import { describe, expect, it } from "vitest";
import {
  applyConsentToggle,
  canStartInterview,
  DEFAULT_CONSENT,
  isPhysiologyActive,
  physiologyConstraint,
  toConsentPayload,
} from "./logic";

describe("分项授权与依赖约束", () => {
  it("拒绝 camera ⇒ physiology 被强制禁用并给出可读原因", () => {
    let flags = applyConsentToggle(DEFAULT_CONSENT, "camera", true);
    flags = applyConsentToggle(flags, "physiology", true);
    expect(isPhysiologyActive(flags)).toBe(true);

    flags = applyConsentToggle(flags, "camera", false);
    expect(flags.physiology).toBe(false);
    expect(isPhysiologyActive(flags)).toBe(false);
    expect(physiologyConstraint(flags)).not.toBeNull();
    expect(physiologyConstraint(flags)).toContain("摄像头");
  });

  it("开启 physiology 会自动开启 camera（rPPG 需要画面）", () => {
    const flags = applyConsentToggle(DEFAULT_CONSENT, "physiology", true);
    expect(flags.camera).toBe(true);
    expect(isPhysiologyActive(flags)).toBe(true);
  });

  it("只同意 base 也能开始面试", () => {
    const flags = applyConsentToggle(DEFAULT_CONSENT, "base", true);
    expect(canStartInterview(flags)).toBe(true);
    const payload = toConsentPayload(flags);
    expect(payload.base).toBe(true);
    expect(payload.physiology).toBe(false);
    expect(payload.camera).toBe(false);
  });

  it("未同意 base 不能开始", () => {
    const flags = applyConsentToggle(DEFAULT_CONSENT, "screen", true);
    expect(canStartInterview(flags)).toBe(false);
  });

  it("payload 中 physiology 按'真正生效'口径收敛（需 camera）", () => {
    // 直接构造一个不一致的 flags（camera=false, physiology=true）模拟脏输入。
    const dirty = { ...DEFAULT_CONSENT, base: true, camera: false, physiology: true };
    expect(toConsentPayload(dirty).physiology).toBe(false);
  });
});
