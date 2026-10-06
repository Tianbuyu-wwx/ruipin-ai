/**
 * 面试主界面（mockup 舞台 3 的产品化）。
 *
 * 版式：room-top（进度/计时/侧栏开关）+ room-main（信号台/题目卡 + 侧栏三面板）
 * + room-bar（控制带，按轮次身份切换可见控件）+ 降级徽标。
 *
 * 数据全部来自 store 派生：转写（transcript）、逐轮评估（turns）、
 * 日志（events 原始流）。没有的数据不摆拍：总题数协议未下发，
 * 进度条只画已出现的轮次。
 */

import { type KeyboardEvent, useEffect, useState } from "react";
import type { AudioPlayer } from "../avatar/audioPlayer";
import type { SocketStatus } from "../net/socket";
import { degradationBanner } from "../store/selectors";
import type { DerivedState, ServerEvent, VisemeEvent } from "../store/types";
import { Avatar } from "./Avatar";

/** FSM 状态 → 轮次身份（驱动波形动画与控制带可见性）。 */
function turnOf(state: DerivedState["fsm"]["state"]): "speaking" | "yours" | "processing" | "idle" {
  switch (state) {
    case "asking":
    case "greeting":
    case "followup":
    case "buffer":
      return "speaking";
    case "listening":
    case "buffer_listening":
    case "candidate_qa":
      return "yours";
    case "processing":
      return "processing";
    default:
      return "idle";
  }
}

const TURN_LABEL: Record<string, string> = {
  speaking: "面试官在说",
  yours: "轮到你了",
  processing: "正在评估",
  idle: "等待中",
};

export interface InterviewRoomProps {
  derived: DerivedState;
  status: SocketStatus;
  timeline: readonly VisemeEvent[];
  player?: AudioPlayer | null;
  /** 原始事件流（侧栏日志面板用）。 */
  events?: readonly ServerEvent[];
  /** 会话开始时刻（秒）；计时器从这里走。 */
  sessionStartedAt?: number | null;
  onCommitAnswer: (text: string) => void;
  onSkip: () => void;
  onEnd: () => void;
  onInterrupt?: () => void;
  recordingAvailable?: boolean;
  recording?: boolean;
  onToggleRecording?: () => void;
}

function useElapsed(startedAt: number | null): string {
  const [, tick] = useState(0);
  useEffect(() => {
    if (startedAt === null) return;
    const id = window.setInterval(() => tick((n) => n + 1), 1000);
    return () => window.clearInterval(id);
  }, [startedAt]);
  if (startedAt === null) return "00:00";
  const s = Math.max(0, Math.floor(Date.now() / 1000 - startedAt));
  const mm = String(Math.floor(s / 60)).padStart(2, "0");
  const ss = String(s % 60).padStart(2, "0");
  return `${mm}:${ss}`;
}

const WAVE_BARS = 30;

export function InterviewRoom({
  derived,
  status: _status,
  timeline,
  player = null,
  events = [],
  sessionStartedAt = null,
  onCommitAnswer,
  onSkip,
  onEnd,
  onInterrupt,
}: InterviewRoomProps) {
  const [draft, setDraft] = useState("");
  const [tab, setTab] = useState<"transcript" | "evals" | "log">("transcript");
  const [railOpen, setRailOpen] = useState(true);
  const [degradeOpen, setDegradeOpen] = useState(false);

  const banner = degradationBanner(derived);
  const question = derived.question;
  const turn = turnOf(derived.fsm.state);
  const turns = Object.values(derived.turns).sort((a, b) => a.turnIndex - b.turnIndex);
  const elapsed = useElapsed(sessionStartedAt);

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

  const canSubmit = question !== null && !derived.timedOut && turn === "yours";

  return (
    <div className="room" data-turn={turn} data-degrade={banner.visible ? "1" : "0"}>
      <header className="room-top">
        <div className="brand">
          <span className="mark">RP</span>锐聘 AI
        </div>
        <div className="spacer"></div>
        <div className="progress">
          <span>
            第 <span className="num">{question ? question.turnIndex + 1 : 1}</span> 题
            {!question?.scored && question ? "（缓冲）" : ""}
          </span>
          <span className="bars" aria-hidden="true">
            {turns.map((t, i) => (
              <b key={t.turnId} data-done={i < turns.length - 1 ? "" : undefined} data-now={i === turns.length - 1 ? "" : undefined}></b>
            ))}
          </span>
        </div>
        <div className="timer num">{elapsed}</div>
        <button
          type="button"
          className="btn btn-plain btn-sm"
          aria-expanded={railOpen}
          onClick={() => setRailOpen((v) => !v)}
        >
          {railOpen ? "收起侧栏" : "展开侧栏"}
        </button>
      </header>

      <div className="room-main">
        <section className="stage" aria-label="虚拟面试官">
          <div className="signal">
            <span className="state">
              <span className="dot"></span>
              {TURN_LABEL[turn]}
              {_status === "reconnecting" ? "（重连中…）" : ""}
            </span>
            <div className="wave" aria-hidden="true">
              {Array.from({ length: WAVE_BARS }, (_, i) => (
                <b key={i} style={{ "--peak": 2.5 + ((i * 13) % 7) * 0.6 } as React.CSSProperties} />
              ))}
            </div>
            {/* 口型时间线驱动的小型形象（可用时）；不可用则只留波形，不摆拍假人 */}
            <Avatar
              timeline={timeline}
              player={player}
              speaking={turn === "speaking"}
              caption={question?.text ?? ""}
            />
          </div>

          <div className="prompt">
            <div className="who">{question ? `面试官 · 第 ${question.turnIndex + 1} 题` : "面试官"}</div>
            <p>{question ? question.text || "（题目文本为空，请重听）" : "等待服务端下发题目…"}</p>
            {question && !question.scored ? (
              <div className="buf">
                <span className="dot"></span>这是一道缓冲题，不记分
              </div>
            ) : null}
          </div>
        </section>

        <aside className="rail" data-open={railOpen ? "true" : "false"}>
          <div className="rail-tabs" role="tablist">
            {(
              [
                ["transcript", "转写"],
                ["evals", "逐轮评估"],
                ["log", "日志"],
              ] as const
            ).map(([key, label]) => (
              <button
                key={key}
                type="button"
                role="tab"
                aria-selected={tab === key}
                onClick={() => setTab(key)}
              >
                {label}
              </button>
            ))}
          </div>
          <div className="rail-body">
            {tab === "transcript" ? (
              <div role="tabpanel" aria-label="语音转写">
                <h4>语音转写</h4>
                {derived.transcript.finals.length === 0 && !derived.transcript.partial ? (
                  <div className="line">
                    <span className="live">（暂无转写）</span>
                  </div>
                ) : (
                  derived.transcript.finals.map((t, i) => (
                    <div className="line" key={`f-${i}`}>
                      <span className="a">{t}</span>
                    </div>
                  ))
                )}
                {derived.transcript.partial ? (
                  <div className="line">
                    <span className="live">…{derived.transcript.partial}</span>
                    <span className="caret"></span>
                  </div>
                ) : null}
              </div>
            ) : null}

            {tab === "evals" ? (
              <div role="tabpanel" aria-label="逐轮评估">
                <h4>逐轮评估</h4>
                {turns.length === 0 ? (
                  <div className="line">
                    <span className="live">（还没有完成的轮次）</span>
                  </div>
                ) : (
                  turns.map((t) => (
                    <div className="line" key={t.turnId}>
                      <span className="q">{t.question || `第 ${t.turnIndex + 1} 题`}</span>
                      {t.eval ? (
                        <div className="score-row">
                          <span className="n">{t.eval.score === null ? "无分" : t.eval.score.toFixed(1)}</span>
                          <span style={{ fontSize: 12.5, color: "var(--faint)" }}>
                            置信度 {t.eval.confidence.toFixed(2)}
                            {t.eval.provider ? ` · ${t.eval.provider}` : ""}
                          </span>
                        </div>
                      ) : (
                        <div className="score-row">
                          <span className="n idle">这轮还没答完</span>
                        </div>
                      )}
                    </div>
                  ))
                )}
              </div>
            ) : null}

            {tab === "log" ? (
              <div role="tabpanel" aria-label="会话日志">
                <h4>会话日志</h4>
                {events.length === 0 ? (
                  <div className="line logline">（暂无事件）</div>
                ) : (
                  events.slice(-40).map((ev, i) => {
                    const label =
                      ev.kind === "envelope"
                        ? ev.type
                        : ev.kind === "tts_chunk"
                          ? `tts_chunk (${ev.byteLength}B)`
                          : `avatar.viseme (${ev.visemes.length})`;
                    return (
                      <div className="logline" key={`${ev.seq}-${i}`}>
                        seq {ev.seq} · {label}
                      </div>
                    );
                  })
                )}
                {derived.errors.map((e, i) => (
                  <div className="line" key={`err-${e.seq}-${i}`}>
                    <span className="errtext">
                      [{e.code}] {e.message}
                      {e.detail ? `（${e.detail}）` : ""}
                    </span>
                  </div>
                ))}
              </div>
            ) : null}
          </div>
        </aside>
      </div>

      <footer className="room-bar">
        <div className="turn" aria-live="polite">
          <span className="dot"></span>
          {TURN_LABEL[turn]}
        </div>

        <div className="compose">
          <label htmlFor="answer" style={{ position: "absolute", width: 1, height: 1, overflow: "hidden", clip: "rect(0 0 0 0)" }}>
            作答输入
          </label>
          <textarea
            id="answer"
            placeholder="在这里打字，或者直接开口说。Ctrl/⌘ 加 Enter 提交"
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={onKeyDown}
            disabled={turn !== "yours"}
          ></textarea>
          <div className="bar-actions only-yours">
            <button type="button" className="btn btn-primary btn-sm" disabled={!canSubmit} onClick={submit}>
              提交
            </button>
            <button type="button" className="btn btn-quiet btn-sm" onClick={onSkip}>
              跳过
            </button>
          </div>
        </div>

        <p className="hint-idle">面试官在说话，可以打断</p>

        <div className="bar-actions only-speak">
          {onInterrupt ? (
            <button type="button" className="btn btn-quiet btn-sm" onClick={onInterrupt}>
              打断
            </button>
          ) : null}
        </div>

        <div className="bar-actions">
          <button type="button" className="btn btn-plain btn-sm" onClick={onEnd}>
            结束面谈
          </button>
        </div>
      </footer>

      {banner.visible ? (
        <div className="degrade" data-open={degradeOpen ? "true" : "false"}>
          <button
            type="button"
            className="degrade-pill"
            role="status"
            aria-live="polite"
            aria-expanded={degradeOpen}
            onClick={() => setDegradeOpen((v) => !v)}
          >
            <span className="dot"></span>
            {banner.badges[0] ?? `降级中（L${banner.level}）`}
          </button>
          <div className="degrade-panel">
            <h4>这一轮哪里不一样</h4>
            <div className="row">
              <b>{banner.text}</b>
              {banner.badges.length > 0 ? banner.badges.join("；") : ""}
            </div>
            <div className="row">
              <b>不用你做什么</b>
              接着答完就行。报告里会标出哪几轮是规则算法算的。
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
}
