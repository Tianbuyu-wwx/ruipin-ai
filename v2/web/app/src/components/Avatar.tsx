/**
 * 虚拟面试官形象 —— **方案 A：2D 骨骼 rig**（方案 §6.4：60FPS、CPU 可跑、零 GPU，
 * 神经说话头实时性不达标，Phase 1–3 一律用 2D rig）。
 *
 * 口型由 `avatar.viseme` 时间轴驱动：以 `AudioPlayer.now()`（AudioContext 音频时钟）
 * 为统一基准，逐帧用纯函数 `visemeAt` 求口型（方案 §6.5 要求音画同源时钟）。
 * 音频不可用时（`player.textOnly`）：口型停待机位，并在画面下方以**字幕**呈现文本，
 * 流程照常推进，不卡。
 *
 * 无障碍：`role="img"` + `aria-label`；字幕同时是"面试官语音必须有字幕"的落点（§12.6）。
 */

import { useEffect, useRef } from "react";
import type { AudioPlayer } from "../avatar/audioPlayer";
import { REST_VISEME, visemeAt } from "../avatar/timeline";
import type { VisemeEvent } from "../store/types";

/** 各口型的"张嘴度"基准（0=闭合，1=全开）。未知口型取中间值。 */
const VISEME_OPENNESS: Record<string, number> = {
  rest: 0.08,
  sil: 0.05,
  m: 0.04,
  fv: 0.16,
  i: 0.34,
  u: 0.3,
  e: 0.6,
  o: 0.72,
  a: 1,
  aa: 1,
};

export interface AvatarProps {
  timeline: readonly VisemeEvent[];
  /** 音频播放器（提供统一时钟）；为 null 时用本地 rAF 时钟且视为纯文本。 */
  player?: AudioPlayer | null;
  /** 是否正在说话；false 时口型停待机。 */
  speaking?: boolean;
  /** 字幕文本（当前题目/播报内容）。 */
  caption?: string;
  width?: number;
  height?: number;
}

function opennessOf(viseme: string, weight: number): number {
  const base = VISEME_OPENNESS[viseme] ?? 0.5;
  const w = Number.isFinite(weight) ? Math.max(0, Math.min(1, weight)) : 1;
  return Math.max(0, Math.min(1, base * w));
}

export function Avatar({
  timeline,
  player = null,
  speaking = false,
  caption = "",
  width = 220,
  height = 220,
}: AvatarProps) {
  const mouthRef = useRef<SVGEllipseElement | null>(null);
  const browRef = useRef<SVGPathElement | null>(null);

  useEffect(() => {
    let raf = 0;
    const base = typeof performance !== "undefined" ? performance.now() : Date.now();
    const tick = () => {
      const mouth = mouthRef.current;
      if (mouth) {
        let open = 0.08;
        if (speaking) {
          const t = player ? player.now() : safePerf() - base;
          const frame = visemeAt(timeline, t);
          open = opennessOf(frame.viseme, frame.weight);
        } else if (timeline.length > 0) {
          open = opennessOf(REST_VISEME, 1);
        }
        mouth.setAttribute("ry", String(4 + open * 18));
        mouth.setAttribute("rx", String(20 - open * 4));
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [timeline, player, speaking]);

  return (
    <figure className="avatar-stage">
      <svg
        role="img"
        aria-label="虚拟面试官形象，口型随语音同步"
        width={width}
        height={height}
        viewBox="0 0 200 200"
      >
        <ellipse cx="100" cy="96" rx="58" ry="66" fill="#e8eef7" stroke="#94a3b8" strokeWidth="2" />
        <circle cx="78" cy="84" r="6" fill="#334155" />
        <circle cx="122" cy="84" r="6" fill="#334155" />
        <path ref={browRef} d="M64 66 Q78 58 92 66 M108 66 Q122 58 136 66" stroke="#334155" strokeWidth="3" fill="none" />
        <ellipse ref={mouthRef} cx="100" cy="128" rx="20" ry="6" fill="#7c3aed" />
      </svg>
      {caption ? (
        <figcaption className="avatar-caption" aria-live="polite">
          {caption}
        </figcaption>
      ) : null}
    </figure>
  );
}

function safePerf(): number {
  return typeof performance !== "undefined" ? performance.now() : Date.now();
}
