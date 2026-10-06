/**
 * 上行消息构造器：把领域意图打包成标准信封。
 *
 * **payload 字段名是与 Python 侧的契约**，集中在此处，禁止在组件里手写字符串：
 *   session.create ← orchestrator/service.py `create(meta)`（meta 原样进入 session.meta）
 *   consent.grant  ← orchestrator/service.py `Consent`（base/physio/media_recording）
 *   answer.text / answer.commit / control.* ← ClientType（方案 §2.4）
 *
 * `encode` 不做校验以外的事情，`seq/ts/session_id` 由调用方注入（SocketClient 负责），
 * 保持本模块为纯函数、可单测。
 */

import { ClientType } from "./constants";
import { type Envelope, makeEnvelope } from "./codec";

/**
 * `session.create` 的 payload。
 *
 * ⚠️ 现状：后端 `bridge.py::on_session_create` **只读 `candidate_id`**，
 * 下列字段目前发出去无人接收（`position` 不会影响出题、`n_questions` 不会改题量、
 * `adaptive` / `physio_enabled` 不会改能力开关）。
 * 之所以仍然照发：这些是商定的线格式，服务端补齐后无需改前端；
 * 但**不要据此认为功能已经生效**。重连重放目前由网关按上行 seq 做（`Gateway._replay`），
 * 与本结构里的 `resume` / `last_seq` 无关。
 */
export interface SessionCreatePayload {
  /** 应聘岗位（进入 `session.meta`）。当前后端不读。 */
  position?: string;
  /** 题量。当前后端不读（题量由服务端启动参数决定）。 */
  nQuestions?: number;
  /** 能力开关。当前后端不读。 */
  adaptive?: boolean;
  physioEnabled?: boolean;
  /** 重连恢复标记与已收到的最大 seq。当前后端不读。 */
  resume?: boolean;
  lastSeq?: number;
}

/**
 * `consent.grant` 的 payload。
 *
 * PIPL 第 28 条：生理属敏感个人信息，`physiology` 必须**单独**取得同意，
 * 不能用一揽子 `camera` 带过。`camera` 还是 rPPG 的物理前提，故 camera 拒绝时
 * `physiology` 必为 false（由 `consent/logic.ts` 保证）。
 *
 * ⚠️ 与后端 `Consent(base, physio, media_recording)` 的**实际**映射
 * （见 `transport/bridge.py::on_consent_grant`，照抄事实，不是照抄愿望）：
 *   base       → Consent.base
 *   physiology → Consent.physio
 *   screen     → Consent.media_recording   ← 注意：屏幕共享被记成"媒体录制"
 *   camera     → **后端不读**（域对象里没有这一项；它在前端是 rPPG 的物理前提）
 *   media_recording → **后端不读**（本字段目前发出去等于白发）
 *
 * 也就是说：界面上没有"录制"开关，服务端报告里的 `media_recording` 实际记录的是
 * "用户是否同意屏幕共享"。域对象补齐 `camera` / 真正的录制同意之前，
 * 别把 `media_recording` 当成"录制授权"来解读。
 */
export interface ConsentGrantPayload {
  base: boolean;
  camera: boolean;
  physiology: boolean;
  screen: boolean;
  media_recording: boolean;
}

export function sessionCreate(
  payload: SessionCreatePayload = {},
  opts: { seq?: number; ts?: number; sessionId?: string } = {},
): Envelope {
  return makeEnvelope(
    ClientType.SESSION_CREATE,
    {
      position: payload.position ?? "",
      n_questions: payload.nQuestions ?? 0,
      adaptive: payload.adaptive ?? false,
      physio_enabled: payload.physioEnabled ?? false,
      resume: payload.resume ?? false,
      last_seq: payload.lastSeq ?? 0,
    },
    opts,
  );
}

export function consentGrant(
  payload: ConsentGrantPayload,
  opts: { seq?: number; ts?: number; sessionId?: string } = {},
): Envelope {
  return makeEnvelope(ClientType.CONSENT_GRANT, { ...payload }, opts);
}

export function answerText(
  text: string,
  opts: { seq?: number; ts?: number; sessionId?: string } = {},
): Envelope {
  return makeEnvelope(ClientType.ANSWER_TEXT, { text }, opts);
}

export function answerCommit(
  opts: { seq?: number; ts?: number; sessionId?: string } = {},
): Envelope {
  return makeEnvelope(ClientType.ANSWER_COMMIT, {}, opts);
}

export function controlBargeIn(
  opts: { seq?: number; ts?: number; sessionId?: string } = {},
): Envelope {
  return makeEnvelope(ClientType.CONTROL_BARGE_IN, {}, opts);
}

export function controlSkip(
  opts: { seq?: number; ts?: number; sessionId?: string } = {},
): Envelope {
  return makeEnvelope(ClientType.CONTROL_SKIP, {}, opts);
}

export function controlEnd(
  opts: { seq?: number; ts?: number; sessionId?: string } = {},
): Envelope {
  return makeEnvelope(ClientType.CONTROL_END, {}, opts);
}
