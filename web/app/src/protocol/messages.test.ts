/**
 * 上行消息构造器测试：类型与 payload 字段名是与 Python 侧的契约。
 */

import { describe, expect, it } from "vitest";
import { ClientType } from "./constants";
import {
  answerCommit,
  answerText,
  consentGrant,
  controlBargeIn,
  controlEnd,
  controlSkip,
  sessionCreate,
} from "./messages";

describe("客户端消息构造", () => {
  it("session.create 字段名与默认值", () => {
    const e = sessionCreate();
    expect(e.type).toBe(ClientType.SESSION_CREATE);
    expect(e.payload).toEqual({
      position: "",
      n_questions: 0,
      adaptive: false,
      physio_enabled: false,
      resume: false,
      last_seq: 0,
    });
  });

  it("session.create 传入参数被映射", () => {
    const e = sessionCreate({ position: "后端", nQuestions: 5, adaptive: true, physioEnabled: true, resume: true, lastSeq: 42 });
    expect(e.payload).toMatchObject({
      position: "后端",
      n_questions: 5,
      adaptive: true,
      physio_enabled: true,
      resume: true,
      last_seq: 42,
    });
  });

  it("consent.grant 携带分项开关", () => {
    const e = consentGrant({ base: true, camera: false, physiology: false, screen: false, media_recording: false });
    expect(e.type).toBe(ClientType.CONSENT_GRANT);
    expect(e.payload).toEqual({ base: true, camera: false, physiology: false, screen: false, media_recording: false });
  });

  it("answer.text 携带文本，answer.commit 为空负载", () => {
    expect(answerText("你好").type).toBe(ClientType.ANSWER_TEXT);
    expect(answerText("你好").payload).toEqual({ text: "你好" });
    expect(answerCommit().type).toBe(ClientType.ANSWER_COMMIT);
    expect(answerCommit().payload).toEqual({});
  });

  it("control.* 三个控制消息类型正确", () => {
    expect(controlBargeIn().type).toBe(ClientType.CONTROL_BARGE_IN);
    expect(controlSkip().type).toBe(ClientType.CONTROL_SKIP);
    expect(controlEnd().type).toBe(ClientType.CONTROL_END);
  });

  it("opts 透传 seq/ts/sessionId", () => {
    const e = answerText("x", { seq: 7, ts: 123, sessionId: "s-1" });
    expect(e.seq).toBe(7);
    expect(e.ts).toBe(123);
    expect(e.session_id).toBe("s-1");
  });
});
