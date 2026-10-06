/**
 * 信用窗口测试（方案 §2.4 背压）。对应 Python `protocol.CreditWindow`。
 */

import { describe, expect, it } from "vitest";
import { CreditWindow } from "./creditWindow";

describe("CreditWindow", () => {
  it("窗口满时 tryAcquire 失败且不消耗信用", () => {
    const w = new CreditWindow(3);
    expect(w.tryAcquire()).toBe(true);
    expect(w.tryAcquire()).toBe(true);
    expect(w.tryAcquire()).toBe(true);
    expect(w.full).toBe(true);
    expect(w.available).toBe(0);
    // 满窗再取：失败，且不改动 used。
    expect(w.tryAcquire()).toBe(false);
    expect(w.used).toBe(3);
  });

  it("收到 ack 释放后恢复可发送", () => {
    const w = new CreditWindow(2);
    w.tryAcquire();
    w.tryAcquire();
    expect(w.full).toBe(true);
    expect(w.release(1)).toBe(1);
    expect(w.full).toBe(false);
    expect(w.available).toBe(1);
    expect(w.tryAcquire()).toBe(true);
    expect(w.full).toBe(true);
  });

  it("批量占用：不足以容纳 n 时一个都不占", () => {
    const w = new CreditWindow(3);
    expect(w.tryAcquire(2)).toBe(true);
    expect(w.tryAcquire(2)).toBe(false);
    expect(w.used).toBe(2);
  });

  it("超量释放显式抛错（重复消费必须暴露）", () => {
    const w = new CreditWindow(4);
    w.tryAcquire(2);
    expect(() => w.release(3)).toThrow();
  });

  it("releaseAll 清空在途（断线重连用）", () => {
    const w = new CreditWindow(4);
    w.tryAcquire();
    w.tryAcquire();
    expect(w.releaseAll()).toBe(2);
    expect(w.used).toBe(0);
  });

  it("非法容量/增量抛错", () => {
    expect(() => new CreditWindow(0)).toThrow();
    const w = new CreditWindow(2);
    expect(() => w.tryAcquire(0)).toThrow();
  });
});
