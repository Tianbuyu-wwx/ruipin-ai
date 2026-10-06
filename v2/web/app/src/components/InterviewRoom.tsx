/**
 * 面试主界面（方案 §2.2 组件拼装 + §12.6 无障碍）。
 *
 * 组成：形象舞台 / 题目卡 / 作答区（文本 + 可选录音）/ 转写流 / 逐轮评估 / 降级横幅。
 *
 * 两条硬要求：
 * - **降级横幅显式可见**：`role="status" aria-live="polite"`，颜色之外还有文字与图标，
 *   不把颜色作为唯一信息载体（§12.6）。
 * - **键盘可达**：所有交互均可用键盘完成；作答框支持 Ctrl/⌘+Enter 提交。
 */

import { type KeyboardEvent, useState } from "react";
import type { AudioPlayer } from "../avatar/audioPlayer";
import type { SocketStatus } from "../net/socket";
import { degradationBanner } from "../store/selectors";
import { type DerivedState, FSM_STATE_LABEL, type VisemeEvent } from "../store/types";
import { Avatar } from "./Avatar";

const STATUS_TEXT: Record<SocketStatus, string> = {
  idle: "未连接",
  connecting: "连接中…",
  open: "已连接",
  reconnecting: "重连中…",
  closed: "连接已断开",
};

export interface InterviewRoomProps {
  derived: DerivedState;
  status: SocketStatus;
  timeline: readonly VisemeEvent[];
  player?: AudioPlayer | null;
  onCommitAnswer: (text: string) => void;
  onSkip: () => void;
  onEnd: () => void;
  /** 录音是否可用（MediaRecorder 未接时为 false，界面显式说明）。 */
  recordingAvailable?: boolean;
  recording?: boolean;
  onToggleRecording?: () => void;
}

export function InterviewRoom({
  derived,
  status,
  timeline,
  player = null,
  onCommitAnswer,
  onSkip,
  onEnd,
  recordingAvailable = false,
  recording = false,
  onToggleRecording,
}: InterviewRoomProps) {
  const [draft, setDraft] = useState("");
  const banner = degradationBanner(derived);
  const question = derived.question;
  const turns = Object.values(derived.turns).sort((a, b) => a.turnIndex - b.turnIndex);

  const submit = () => {
    const text = draft.trim();
    onCommitAnswer(text);
    setDraft("");
  };

  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
      e.preventDefault();
      submit();
    }
  };

  const canSubmit = question !== null && !derived.timedOut;

  return (
    <div className="interview-room">
      {/* 状态栏：连接状态 + FSM 状态 + 降级徽标 */}
      <header className="status-bar" role="status" aria-live="polite">
        <span className={`conn conn-${status}`}>{STATUS_TEXT[status]}</span>
        <span className="fsm-state">阶段：{FSM_STATE_LABEL[derived.fsm.state]}</span>
        {question ? (
          <span className="q-index">{question.scored ? `第 ${question.turnIndex + 1} 题` : "缓冲题（不计分）"}</span>
        ) : null}
        {derived.timedOut ? <span className="timeout">会话已超时</span> : null}
      </header>

      {/* 降级横幅：显式、可见、非静默 */}
      <div
        className={`degradation-banner ${banner.visible ? "visible" : "hidden"}`}
        role="status"
        aria-live="polite"
        data-degraded={banner.visible ? "true" : "false"}
      >
        {banner.visible ? (
          <>
            <span aria-hidden="true">⚠</span>
            <strong>降级中（L{banner.level}）</strong>
            <span>{banner.text}</span>
            {banner.badges.length > 0 ? <span className="badges">{banner.badges.join(" / ")}</span> : null}
          </>
        ) : null}
      </div>

      <main className="room-main">
        <section className="stage" aria-label="虚拟面试官">
          <Avatar
            timeline={timeline}
            player={player}
            speaking={derived.fsm.state === "asking" || derived.fsm.state === "greeting"}
            caption={question?.text ?? ""}
          />
        </section>

        <section className="question-card" aria-labelledby="question-heading">
          <h2 id="question-heading">当前题目</h2>
          {question ? (
            <>
              <p className="question-text">{question.text || "（题目文本为空，请重听）"}</p>
              {!question.scored ? (
                <p className="question-tag" role="note">
                  缓冲题：仅用于帮助你调整状态，<strong>不计分</strong>。
                </p>
              ) : null}
            </>
          ) : (
            <p className="question-text muted">等待服务端下发题目…</p>
          )}
        </section>

        <section className="answer-area" aria-labelledby="answer-heading">
          <h2 id="answer-heading">你的作答</h2>
          <label htmlFor="answer-input" className="visually-hidden">
            作答输入框
          </label>
          <textarea
            id="answer-input"
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={onKeyDown}
            rows={5}
            placeholder="在此输入你的回答（Ctrl/⌘ + Enter 提交）"
            disabled={!canSubmit}
          />
          <div className="answer-actions">
            <button type="button" className="primary" onClick={submit} disabled={!canSubmit}>
              提交作答
            </button>
            {recordingAvailable ? (
              <button type="button" onClick={() => onToggleRecording?.()} aria-pressed={recording}>
                {recording ? "停止录音" : "开始录音"}
              </button>
            ) : (
              <span className="muted" role="note">
                录音未接入（当前仅文本作答）。
              </span>
            )}
            <button type="button" onClick={onSkip}>
              跳过本题
            </button>
            <button type="button" onClick={onEnd}>
              结束面试
            </button>
          </div>
        </section>

        <section className="transcript" aria-labelledby="transcript-heading" aria-live="polite">
          <h2 id="transcript-heading">转写</h2>
          {derived.transcript.finals.length === 0 && !derived.transcript.partial ? (
            <p className="muted">（暂无转写）</p>
          ) : (
            <ol>
              {derived.transcript.finals.map((t, i) => (
                <li key={`f-${i}`}>{t}</li>
              ))}
            </ol>
          )}
          {derived.transcript.partial ? <p className="partial">…{derived.transcript.partial}</p> : null}
        </section>

        <section className="evals" aria-labelledby="eval-heading">
          <h2 id="eval-heading">逐轮评估</h2>
          {turns.length === 0 ? (
            <p className="muted">（暂无评估）</p>
          ) : (
            <ul>
              {turns.map((t) => (
                <li key={t.turnId}>
                  <span className="turn-q">{t.question}</span>
                  {t.eval ? (
                    <span className="turn-eval">
                      {t.eval.score === null ? "无分" : `${t.eval.score.toFixed(1)} 分`}
                      （provider={t.eval.provider}，confidence={t.eval.confidence.toFixed(2)}
                      {t.eval.degraded ? `，降级：${t.eval.degradeReason ?? ""}` : ""}）
                      {t.eval.feedback ? <em className="feedback"> {t.eval.feedback}</em> : null}
                    </span>
                  ) : (
                    <span className="muted"> 未评分</span>
                  )}
                </li>
              ))}
            </ul>
          )}
        </section>

        {derived.errors.length > 0 ? (
          <section className="errors" role="alert">
            <h2>错误</h2>
            <ul>
              {derived.errors.map((e, i) => (
                <li key={`e-${e.seq}-${i}`}>
                  [{e.code}] {e.message}
                  {e.detail ? `（${e.detail}）` : ""}
                </li>
              ))}
            </ul>
          </section>
        ) : null}
      </main>
    </div>
  );
}
