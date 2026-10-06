/**
 * 应用容器（mockup 五舞台的产品化装配）。
 *
 * 舞台：landing → prepare → room → closing → report，settings 可从 landing /
 * prepare 进入。SocketClient / AudioPlayer / 事件溯源 store 的接线保持不变——
 * 只换了视图层，协议与状态逻辑不动。
 *
 * **不造假数据**：拿不到后端就如实显示"未连接/无题目"，音频不可用就走纯文本。
 */

import { useEffect, useMemo, useRef, useState } from "react";
import { AudioPlayer } from "./avatar/audioPlayer";
import { Closing } from "./components/Closing";
import { InterviewRoom } from "./components/InterviewRoom";
import { Landing } from "./components/Landing";
import { Prepare } from "./components/Prepare";
import { ReportView } from "./components/ReportView";
import { SettingsPanel } from "./components/SettingsPanel";
import { type ConsentFlags, toConsentPayload } from "./consent/logic";
import { SocketClient, type SocketStatus } from "./net/socket";
import {
  answerCommit,
  answerText,
  consentGrant,
  controlBargeIn,
  controlEnd,
  controlSkip,
  sessionCreate,
} from "./protocol/messages";
import { type Prefs, loadPrefs } from "./settings/prefs";
import { useInterviewStore } from "./store/interviewStore";

type Stage = "landing" | "prepare" | "settings" | "room" | "closing" | "report";

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
  const override = effective.wsUrl.trim();
  const base =
    override || fromEnv?.trim() || (page ? `ws://${page.host}/ws` : "ws://127.0.0.1:8787/ws");
  const token = effective.token.trim();
  if (!token) return base;
  return `${base}${base.includes("?") ? "&" : "?"}token=${encodeURIComponent(token)}`;
}

export function App() {
  const events = useInterviewStore((s) => s.events);
  const derived = useInterviewStore((s) => s.derived);
  const appendEvent = useInterviewStore((s) => s.appendEvent);
  const [stage, setStage] = useState<Stage>("landing");
  const [status, setStatus] = useState<SocketStatus>("idle");
  const [sessionStartedAt, setSessionStartedAt] = useState<number | null>(null);

  const socketRef = useRef<SocketClient | null>(null);
  const playerRef = useRef<AudioPlayer | null>(null);
  const lastSeqRef = useRef(0);
  // 本机偏好：挂载时读一次；设置页保存并返回后立即重读（改动当场生效）。
  const [prefs, setPrefs] = useState<Prefs>(() => loadPrefs(safeStorage()));

  useEffect(() => {
    return () => {
      socketRef.current?.close();
      playerRef.current?.dispose();
    };
  }, []);

  const endSession = () => {
    socketRef.current?.close();
    socketRef.current = null;
    playerRef.current?.dispose();
    playerRef.current = null;
    setSessionStartedAt(null);
  };

  const startSession = (flags: ConsentFlags, pos: string) => {
    useInterviewStore.getState().reset();
    lastSeqRef.current = 0;
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
    socket.send(
      sessionCreate({ position: pos, adaptive: false, physioEnabled: flags.camera && flags.physiology }),
    );
    socket.send(consentGrant(toConsentPayload(flags)));
    setSessionStartedAt(Date.now() / 1000);
    setStage("room");
  };

  const timeline = useMemo(() => derived.visemes, [derived.visemes]);

  if (stage === "landing") {
    return (
      <div className="app-landing">
        <Landing onStart={() => setStage("prepare")} onOpenSettings={() => setStage("settings")} />
      </div>
    );
  }

  if (stage === "settings") {
    return (
      <div className="app-settings">
        <SettingsPanel
          initial={prefs}
          storage={safeStorage()}
          onBack={() => {
            setPrefs(loadPrefs(safeStorage()));
            setStage("landing");
          }}
        />
      </div>
    );
  }

  if (stage === "prepare") {
    return (
      <div className="app-prepare">
        <Prepare
          initial={{
            base: false,
            camera: prefs.presetCamera || prefs.presetPhysiology,
            physiology: prefs.presetPhysiology,
            screen: prefs.presetScreen,
          }}
          initialPosition={prefs.position}
          onStart={startSession}
          onBack={() => setStage("landing")}
          onOpenSettings={() => setStage("settings")}
        />
      </div>
    );
  }

  if (stage === "closing") {
    return (
      <div className="app-closing">
        <Closing
          nTurns={Object.keys(derived.turns).length}
          nBufferTurns={Object.values(derived.turns).filter((t) => !t.scored).length}
          onOpenReport={() => setStage("report")}
          onExit={() => {
            endSession();
            setStage("landing");
          }}
        />
      </div>
    );
  }

  if (stage === "report") {
    return (
      <div className="app-report">
        <ReportView report={derived.report} />
      </div>
    );
  }

  // stage === "room"
  return (
    <div className="app-room">
      <InterviewRoom
        derived={derived}
        status={status}
        timeline={timeline}
        player={playerRef.current}
        events={events}
        sessionStartedAt={sessionStartedAt}
        onCommitAnswer={(text) => {
          const socket = socketRef.current;
          if (!socket) return;
          if (text) socket.send(answerText(text));
          socket.send(answerCommit());
        }}
        onSkip={() => socketRef.current?.send(controlSkip())}
        onEnd={() => {
          socketRef.current?.send(controlEnd());
          setStage("closing");
        }}
        onInterrupt={() => socketRef.current?.send(controlBargeIn())}
      />
    </div>
  );
}
