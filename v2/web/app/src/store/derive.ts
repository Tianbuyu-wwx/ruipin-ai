/**
 * 事件溯源核心：`derive(events) → DerivedState`（**纯函数**，不读时钟、无副作用）。
 *
 * 设计要点（方案 §2.1 / §2.3）：
 * 1. store 只追加保存**原始事件**（append-only），UI 状态全部由本函数派生——
 *    因此可回放、可时间旅行调试。
 * 2. **seq 去重**：断线重连时服务端会重放事件，重复投递是常态。信封事件按 `seq`
 *    去重（下行 seq 由网关 `_next_seq` 单调分配，恒 ≥ 1）。二进制媒体事件无 seq，
 *    按到达顺序处理（媒体本身允许丢帧/乱序，靠信用窗口兜底）。
 * 3. 确定性：同一事件序列两次 derive 结果 `deepEqual`（不含随机/时间因素）。
 */

import { parseReport } from "../report/parse";
import {
  type DegradationBadge,
  type DerivedState,
  type ErrorItem,
  type HrSample,
  type ServerEvent,
  type TurnEvalView,
  type TurnView,
  type VisemeEvent,
  FsmState,
  MAX_DEGRADATION_LEVEL,
  MAX_HR_SAMPLES,
} from "./types";

// ------------------------------------------------------------------ 取值助手

function str(v: unknown, dflt = ""): string {
  return typeof v === "string" ? v : dflt;
}

function strOrNull(v: unknown): string | null {
  return typeof v === "string" ? v : null;
}

function num(v: unknown, dflt = 0): number {
  return typeof v === "number" && Number.isFinite(v) ? v : dflt;
}

function numOrNull(v: unknown): number | null {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

function bool(v: unknown, dflt = false): boolean {
  return typeof v === "boolean" ? v : dflt;
}

function numMap(v: unknown): Record<string, number> {
  const out: Record<string, number> = {};
  if (v !== null && typeof v === "object" && !Array.isArray(v)) {
    for (const [k, val] of Object.entries(v as Record<string, unknown>)) {
      out[k] = num(val, 0);
    }
  }
  return out;
}

// ------------------------------------------------------------------ 初始态

export function emptyDerivedState(): DerivedState {
  return {
    sessionId: "",
    fsm: { state: FsmState.IDLE, from: null, reason: "", turnIdx: 0, history: [] },
    question: null,
    turns: {},
    transcript: { partial: "", finals: [] },
    hr: { latest: null, samples: [] },
    degradation: { level: 0, current: null, badges: [] },
    report: null,
    errors: [],
    tts: { chunks: 0, bytes: 0 },
    visemes: [],
    mediaAck: null,
    timedOut: false,
  };
}

// ------------------------------------------------------------------ 派生主函数

export function derive(events: readonly ServerEvent[]): DerivedState {
  const s = emptyDerivedState();
  // 信封事件去重集合（seq > 0）。一次 derive 内有效。
  const seenSeq = new Set<number>();

  for (const ev of events) {
    if (ev.kind === "envelope") {
      if (ev.seq > 0) {
        if (seenSeq.has(ev.seq)) continue; // 重放重复投递：跳过，不破坏状态
        seenSeq.add(ev.seq);
      }
      if (ev.sessionId) s.sessionId = s.sessionId || ev.sessionId;
      applyEnvelope(s, ev.type, ev.payload, ev.seq, ev.ts, ev.sessionId);
    } else if (ev.kind === "tts_chunk") {
      s.tts.chunks += 1;
      s.tts.bytes += ev.byteLength;
    } else if (ev.kind === "viseme") {
      s.visemes = ev.visemes.map(cloneViseme);
    }
  }

  return s;
}

function applyEnvelope(
  s: DerivedState,
  type: string,
  payload: Record<string, unknown>,
  seq: number,
  ts: number,
  sessionId: string,
): void {
  switch (type) {
    case "state.changed": {
      const from = strOrNull(payload["from"]);
      const to = str(payload["to"], s.fsm.state);
      const reason = str(payload["reason"]);
      s.fsm.from = from as DerivedState["fsm"]["from"];
      s.fsm.state = to as DerivedState["fsm"]["state"];
      s.fsm.reason = reason;
      s.fsm.history.push({ from, to, reason, seq, ts });
      break;
    }

    case "question.start": {
      const turnId = str(payload["turn_id"], `turn-${Object.keys(s.turns).length}`);
      // 序号优先取服务端下发的 `turn_index`；缺失时用"已建回合数"兜底。
      // 原先兜底走的是 `s.fsm.turnIdx`，而那个值只在收到 question.start 时才被写进
      // 下面的 `s.fsm.turnIdx = turnIndex`——于是第二题起算出的还是同一个号，
      // 实测表现为状态栏永远停在「第 1 题」。
      const turnIndex = num(payload["turn_index"], Object.keys(s.turns).length);
      const text = str(payload["text"]);
      const scored = bool(payload["scored"], true);
      const questionType = strOrNull(payload["question_type"]);
      s.question = { turnId, turnIndex, text, scored, questionType: questionType ?? undefined };
      s.fsm.turnIdx = turnIndex;
      ensureTurn(s, turnId, turnIndex, text, scored);
      break;
    }

    case "transcript.partial": {
      s.transcript.partial = str(payload["text"]);
      break;
    }

    case "transcript.final": {
      const text = str(payload["text"]);
      s.transcript.finals.push(text);
      s.transcript.partial = "";
      const turn = turnForPayload(s, payload);
      if (turn) turn.finalTranscript = text;
      break;
    }

    case "eval.done":
    case "eval.degraded": {
      const evaluated = parseEval(payload);
      const turn = turnForPayload(s, payload);
      if (turn) {
        turn.eval = evaluated;
        break;
      }
      // 对不上任何一轮就**不新建回合**：凭空插一条键为空的回合，会让逐轮评估里
      // 多出一行没有题干的幽灵记录，而真正的评估全被挤到它上面（实测踩过）。
      // 宁可把这次协议不一致显式报出来，也不伪造一条记录。
      s.errors.push({
        seq,
        ts,
        code: "internal",
        message: "收到无法归属到任何一轮的评估结果（没有可匹配的 turn_id / turn_index）",
        detail: null,
      });
      break;
    }

    case "metric.hr": {
      const sample: HrSample = {
        ts: num(payload["ts"], ts),
        bpm: numOrNull(payload["bpm"]),
        confidence: num(payload["confidence"], 0),
        reason: strOrNull(payload["reason"]) ?? undefined,
      };
      s.hr.samples.push(sample);
      if (s.hr.samples.length > MAX_HR_SAMPLES) {
        s.hr.samples.splice(0, s.hr.samples.length - MAX_HR_SAMPLES);
      }
      s.hr.latest = sample;
      break;
    }

    case "degradation.changed": {
      const level = num(payload["level"], 0);
      const badge: DegradationBadge = {
        seq,
        ts,
        level,
        reason: str(payload["reason"]),
        badge: str(payload["badge"]),
        message: str(payload["message"]),
      };
      s.degradation.badges.push(badge);
      // 取已发生事件的最高等级（与后端 DegradationTracker 同口径：不因后续恢复被抹掉）。
      if (level > s.degradation.level) s.degradation.level = level;
      // current 指向当前最高等级对应的徽标（同级取最新）。
      if (level === s.degradation.level) s.degradation.current = badge;
      break;
    }

    case "report.ready": {
      s.report = parseReport(payload);
      break;
    }

    case "error": {
      const item: ErrorItem = {
        seq,
        ts,
        code: str(payload["code"], "internal"),
        message: str(payload["message"]),
        detail: strOrNull(payload["detail"]),
      };
      s.errors.push(item);
      break;
    }

    case "media.ack": {
      s.mediaAck = { consumed: num(payload["consumed"]), credits: num(payload["credits"]) };
      break;
    }

    case "session.timeout": {
      s.timedOut = true;
      s.errors.push({
        seq,
        ts,
        code: "session_timeout",
        message: str(payload["message"], "会话已空闲超时"),
        detail: null,
      });
      break;
    }

    case "question.tts_chunk": {
      // 文本形态的 TTS 块（正常走二进制 0x11），此处只为兼容与统计。
      s.tts.chunks += 1;
      s.tts.bytes += num(payload["size"], 0);
      break;
    }

    case "avatar.viseme": {
      const list = payload["visemes"] ?? payload["timeline"];
      if (Array.isArray(list)) {
        s.visemes = list.map((v) => cloneViseme(normalizeViseme(v)));
      }
      break;
    }

    default:
      // 未知类型：宽容放行（前端可能比后端新），不改变状态。
      break;
  }

  // 会话 id 兜底：事件带 sessionId 时以它为准。
  if (sessionId && !s.sessionId) s.sessionId = sessionId;
}

// ------------------------------------------------------------------ 局部构造

function ensureTurn(
  s: DerivedState,
  turnId: string,
  turnIndex: number,
  question: string,
  scored: boolean,
): TurnView {
  const existing = s.turns[turnId];
  if (existing) {
    if (question) existing.question = question;
    return existing;
  }
  const turn: TurnView = {
    turnId,
    turnIndex,
    question,
    scored,
    finalTranscript: "",
    eval: null,
  };
  s.turns[turnId] = turn;
  return turn;
}

/**
 * 找到 payload 所指的那一轮。
 *
 * 三条线索按可靠性降序：`turn_id`（精确）→ `turn_index`（精确）→ 最后一个已建回合
 * （对 `transcript.final` / `eval.done` 来说，"刚提交的那一轮"就是它）。
 * **找不到时返回 null，绝不新建**：凭空造一个键为空的回合会在逐轮评估里多出一条
 * 没有题目的幽灵行（实测踩过，且真正的评估全被挤到那一条上）。
 */
function turnForPayload(s: DerivedState, payload: Record<string, unknown>): TurnView | null {
  const turnId = str(payload["turn_id"]);
  if (turnId && s.turns[turnId]) return s.turns[turnId];
  const idx = numOrNull(payload["turn_index"]);
  if (idx !== null) {
    const byIdx = Object.values(s.turns).find((t) => t.turnIndex === idx);
    if (byIdx) return byIdx;
  }
  // 未带任何可用标识：落到最后一个已建回合。
  const ids = Object.keys(s.turns);
  return ids.length > 0 ? s.turns[ids[ids.length - 1]] : null;
}

function parseEval(payload: Record<string, unknown>): TurnEvalView {
  return {
    dims: numMap(payload["dims"]),
    score: numOrNull(payload["score"]),
    feedback: str(payload["feedback"]),
    provider: str(payload["provider"], "unknown"),
    confidence: num(payload["confidence"], 0),
    degraded: bool(payload["degraded"]),
    degradeReason: strOrNull(payload["degrade_reason"]),
    latencyMs: num(payload["latency_ms"]),
  };
}

function normalizeViseme(v: unknown): VisemeEvent {
  if (v !== null && typeof v === "object" && !Array.isArray(v)) {
    const o = v as Record<string, unknown>;
    return {
      t_ms: num(o["t_ms"]),
      viseme: str(o["viseme"], "rest"),
      weight: typeof o["weight"] === "number" ? o["weight"] : 1,
    };
  }
  return { t_ms: 0, viseme: "rest", weight: 1 };
}

function cloneViseme(v: VisemeEvent): VisemeEvent {
  return { t_ms: v.t_ms, viseme: v.viseme, weight: v.weight ?? 1 };
}

/** 降级等级上限（供 UI 校验，避免越界渲染）。 */
export const DEGRADATION_MAX = MAX_DEGRADATION_LEVEL;
