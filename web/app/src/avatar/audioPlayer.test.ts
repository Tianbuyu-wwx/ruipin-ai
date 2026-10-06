/**
 * 音频不可用退化测试：拿不到 AudioContext 时必须走纯文本，不抛异常、不卡流程。
 */

import { describe, expect, it } from "vitest";
import { AudioPlayer, pcm16ToFloat32 } from "./audioPlayer";

describe("AudioPlayer 退化路径", () => {
  it("无 AudioContext ⇒ textOnly=true，enqueue 不抛且返回 false", () => {
    const player = new AudioPlayer({ audioContextFactory: () => null, clock: () => 12345 });
    expect(player.textOnly).toBe(true);
    expect(player.enqueue({ data: new Uint8Array([1, 2, 3, 4]) })).toBe(false);
    // now() 退回注入的墙钟，仍可用（统一时钟契约不因无音频而失效）。
    expect(player.now()).toBe(12345);
  });

  it("工厂抛异常同样退化为纯文本", () => {
    const player = new AudioPlayer({
      audioContextFactory: () => {
        throw new Error("no audio");
      },
    });
    expect(player.textOnly).toBe(true);
    expect(player.enqueue({ data: new Uint8Array() })).toBe(false);
    expect(() => player.dispose()).not.toThrow();
  });
});

describe("pcm16ToFloat32", () => {
  it("16-bit LE → [-1,1] 浮点", () => {
    const buf = new Uint8Array([0x00, 0x00, 0x00, 0x40, 0x00, 0xc0]); // 0, 16384, -16384
    const out = pcm16ToFloat32(buf);
    expect(out).toHaveLength(3);
    expect(out[0]).toBeCloseTo(0);
    expect(out[1]).toBeCloseTo(0.5);
    expect(out[2]).toBeCloseTo(-0.5);
  });
});
