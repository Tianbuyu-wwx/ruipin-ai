/**
 * 应用容器（方案 §2.2 `<InterviewApp>` 的最小实现）。
 *
 * 装配：ConsentDialog → InterviewRoom → ReportView，并把 SocketClient / AudioPlayer
 * 与事件溯源 store 连起来。**不造假数据**：拿不到后端就如实显示"未连接/无题目"，
 * 音频不可用就走纯文本（`AudioPlayer.textOnly`）。
 */

import { useEffect, useMemo, useRef, useState } from "react";
import { AudioPlayer } from "./avatar/audioPlayer";
import { ConsentDialog } from "./components/ConsentDialog";
import { InterviewRoom } from "./components/InterviewRoom";
import { ReportView } from "./components/ReportView";
import { SettingsPanel } from "./components/SettingsPanel";
import { type ConsentFlags, DEFAULT_CONSENT, toConsentPayload } from "./consent/logic";
import { SocketClient, type SocketStatus } from "./net/socket";
import { answerCommit, answerText, consentGrant, controlEnd, controlSkip, sessionCreate } from "./protocol/messages";
import { type Prefs, loadPrefs, resolveWsUrl } from "./settings/prefs";
import { useInterviewStore } from "./store/interviewStore";

/** 取本机存储；被禁用（隐私模式）或不存在时返回 null，设置层会回落默认。 */
function safeStorage(): Storage | null {
  try {
    return typeof localStorage !== "undefined" ? localStorage : null;
  } catch {
    return null;
  }
}

/**
 * 拼出 WS 地址：本机设置（设置页可改）优先，其次构建期 env，最后跟随当前页面域名。
 *
 * 令牌只能走 query（`?token=`）：浏览器 `WebSocket` 构造函数**不允许自定义 header**，
 * 所以后端的 `Authorization: Bearer` 那条路只有非浏览器客户端能用。
 * 后端两种都收（header 优先，其次 query），这里走 query。
 */
function buildWsUrl(prefs: Prefs): string {
  const fromEnv = import.meta.env.VITE_WS_URL as string | undefined;
  const page =
    typeof location !== "undefined"
      ? { protocol: location.protocol, host: location.host }
      : undefined;
  // URL query 上的令牌优先于本机设置：便于临时换令牌而不动配置。
  const fromQuery =
    typeof location !== "undefined" ? new URLSearchParams(location.search).get("token") : null;
  const effective = fromQuery ? { ...prefs, token: fromQuery } : prefs;
  return resolveWsUrl(effective, fromEnv, page);
}

export function App() {
  const derived = useInterviewStore((s) => s.derived);
  const appendEvent = useInterviewStore((s) => s.appendEvent);
  const [phase, setPhase] = useState<"consent" | "interview" | "settings">("consent");
  const [status, setStatus] = useState<SocketStatus>("idle");
  const [position, setPosition] = useState("");
  const [consent, setConsent] = useState<ConsentFlags | null>(null);
  // 本机偏好：挂载时读一次；设置页保存并返回后立即重读（改动当场生效）。
  const [prefs, setPrefs] = useState<Prefs>(() => loadPrefs(safeStorage()));

  const socketRef = useRef<SocketClient | null>(null);
  const playerRef = useRef<AudioPlayer | null>(null);
  const lastSeqRef = useRef(0);

  useEffect(() => {
    return () => {
      socketRef.current?.close();
      playerRef.current?.dispose();
    };
  }, []);

  const startSession = (flags: ConsentFlags) => {
    setConsent(flags);
    const player = new AudioPlayer();
    playerRef.current = player;
    const socket = new SocketClient({
      url: buildWsUrl(prefs),
      socketFactory: (u) => new WebSocket(u) as unknown as import("./net/socket").WebSocketLike,
      onEvent: (ev) => {
        if (ev.seq > lastSeqRef.current) lastSeqRef.current = ev.seq;
        appendEvent(ev);
      },
      onAudioChunk: (bytes) => {
        player.enqueue({ data: bytes });
      },
      onStatus: setStatus,
      onReconnect: () => {
        // 断线重连：请求服务端从 last_seq 之后重放；store 侧 seq 去重保证幂等。
        socket.send(sessionCreate({ resume: true, lastSeq: lastSeqRef.current }));
      },
      onError: () => {
        /* 传输层错误已通过 status / error 帧暴露，此处不再重复提示 */
      },
    });
    socketRef.current = socket;
    socket.connect();
    socket.send(sessionCreate({ position, adaptive: false, physioEnabled: flags.camera && flags.physiology }));
    socket.send(consentGrant(toConsentPayload(flags)));
    setPhase("interview");
  };

  const timeline = useMemo(() => derived.visemes, [derived.visemes]);

  if (phase === "settings") {
    return (
      <div className="app">
        <SettingsPanel
          initial={prefs}
          storage={safeStorage()}
          onBack={() => {
            setPrefs(loadPrefs(safeStorage()));
            setPhase("consent");
          }}
        />
      </div>
    );
  }

  if (phase === "consent") {
    return (
      <div className="app">
        <ConsentDialog
          initial={{
            ...DEFAULT_CONSENT,
            camera: prefs.presetCamera || prefs.presetPhysiology,
            physiology: prefs.presetPhysiology,
            screen: prefs.presetScreen,
          }}
          onGrant={startSession}
          onDecline={() => undefined}
        />
        <label className="position-input">
          应聘岗位（可选）：
          <input value={position} onChange={(e) => setPosition(e.target.value)} placeholder="如：后端工程师" />
        </label>
        <button type="button" className="settings-entry" onClick={() => setPhase("settings")}>
          设置
        </button>
      </div>
    );
  }

  return (
    <div className="app">
      <InterviewRoom
        derived={derived}
        status={status}
        timeline={timeline}
        player={playerRef.current}
        recordingAvailable={false}
        onCommitAnswer={(text) => {
          const socket = socketRef.current;
          if (!socket) return;
          if (text) socket.send(answerText(text));
          socket.send(answerCommit());
        }}
        onSkip={() => socketRef.current?.send(controlSkip())}
        onEnd={() => socketRef.current?.send(controlEnd())}
      />
      {derived.report ? (
        <div className="report-overlay">
          <ReportView report={derived.report} />
        </div>
      ) : null}
      {consent && !consent.camera ? (
        <p className="muted consent-echo" role="note">
          本次未开启摄像头：视频/生理维度不计入，其余维度按权重重分配，总分可比。
        </p>
      ) : null}
    </div>
  );
}
