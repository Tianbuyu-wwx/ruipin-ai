/**
 * 前端领域类型。
 *
 * 与 Python 侧的对应：
 *   FsmState            ← domain/states.py `State`（str Enum，字面值逐一对应）
 *   DegradationLevel    ← orchestrator/degradation.py `LEVELS`
 *   ReportView          ← orchestrator/service.py `Report.to_dict()`
 *   PhysioResult        ← physio/regulation.py `RegulationResult`
 *   VisemeEvent         ← ports.py `VisemeEvent`
 */

// ------------------------------------------------------------------ 状态机

/** 面试状态机状态（字面值与 `domain/states.py::State` 一致）。 */
export const FsmState = {
  IDLE: "idle",
  SETUP: "setup",
  GREETING: "greeting",
  ASKING: "asking",
  BUFFER: "buffer",
  BUFFER_LISTENING: "buffer_listening",
  LISTENING: "listening",
  PROCESSING: "processing",
  FOLLOWUP: "followup",
  CANDIDATE_QA: "candidate_qa",
  CLOSING: "closing",
  REPORTING: "reporting",
  COMPLETED: "completed",
  ABORTED: "aborted",
  FAILED: "failed",
} as const;
export type FsmState = (typeof FsmState)[keyof typeof FsmState];

/** 每个状态的中文展示文案（覆盖 states.py 的全部 15 个状态）。 */
export const FSM_STATE_LABEL: Record<FsmState, string> = {
  idle: "待开始",
  setup: "准备中（等待授权）",
  greeting: "开场问候",
  asking: "面试官提问中",
  buffer: "缓冲题呈现（不计分）",
  buffer_listening: "缓冲题作答中（不计分）",
  listening: "请作答",
  processing: "评估中",
  followup: "追问中",
  candidate_qa: "候选人提问",
  closing: "结束语",
  reporting: "生成报告中",
  completed: "已完成",
  aborted: "已中止",
  failed: "异常终止（可重试/可导出作答）",
};

/** 状态机事件（字面值与 `domain/states.py::Event` 一致，用于展示 reason）。 */
export const FsmEvent = {
  CREATE: "create",
  CONSENT_OK: "consent_ok",
  CONSENT_DENIED: "consent_denied",
  GREETING_DONE: "greeting_done",
  TTS_DONE: "tts_done",
  TTS_FAILED: "tts_failed",
  ANSWER_COMMIT: "answer_commit",
  SILENCE_TIMEOUT: "silence_timeout",
  SKIP: "skip",
  REPEAT_QUESTION: "repeat_question",
  EXTEND_TIME: "extend_time",
  INSERT_BUFFER: "insert_buffer",
  BUFFER_DONE: "buffer_done",
  EVAL_DONE: "eval_done",
  EVAL_DEGRADED: "eval_degraded",
  FOLLOWUP_DECISION: "followup_decision",
  NEXT_QUESTION: "next_question",
  NO_MORE_QUESTIONS: "no_more_questions",
  QA_DONE: "qa_done",
  CLOSING_DONE: "closing_done",
  REPORT_DONE: "report_done",
  ABORT: "abort",
  FATAL: "fatal",
} as const;
export type FsmEvent = (typeof FsmEvent)[keyof typeof FsmEvent];

export const FSM_EVENT_LABEL: Record<FsmEvent, string> = {
  create: "创建会话",
  consent_ok: "授权完成",
  consent_denied: "授权被拒绝",
  greeting_done: "开场结束",
  tts_done: "语音播报结束",
  tts_failed: "语音播报失败",
  answer_commit: "提交作答",
  silence_timeout: "静音超时",
  skip: "跳题",
  repeat_question: "重复题目",
  extend_time: "延长作答时间",
  insert_buffer: "插入缓冲题",
  buffer_done: "缓冲题结束",
  eval_done: "评估完成",
  eval_degraded: "评估降级完成",
  followup_decision: "追问决策",
  next_question: "下一题",
  no_more_questions: "无更多题目",
  qa_done: "提问结束",
  closing_done: "结束语结束",
  report_done: "报告生成完成",
  abort: "中止",
  fatal: "致命错误",
};

// ------------------------------------------------------------------ 降级

export interface DegradationLevelView {
  level: number;
  name: string;
  description: string;
  /** 中文徽标文案；Level 0 为空串（正常态，不需要打扰用户）。 */
  uiBadge: string;
}

/** 与 `orchestrator/degradation.py::LEVELS` 一致（level/name/description/ui_badge）。 */
export const DEGRADATION_LEVELS: readonly DegradationLevelView[] = [
  { level: 0, name: "normal", description: "全功能运行", uiBadge: "" },
  { level: 1, name: "no_vision", description: "关闭视觉维度，仅文本评估", uiBadge: "本轮未做视觉分析" },
  { level: 2, name: "rubric_rule", description: "评分降级为规则评分（标注 provider=rubric）", uiBadge: "本轮为规则评分，仅供参考" },
  { level: 3, name: "asr_failed", description: "ASR 失败，仅文本评估并标注无语音", uiBadge: "未获取语音" },
  { level: 4, name: "tts_unavailable", description: "TTS 不可用，纯文本呈现不放音", uiBadge: "无声，文本照常" },
  { level: 5, name: "avatar_failed", description: "形象渲染失败，回退静态形象", uiBadge: "静态形象" },
  { level: 6, name: "fatal", description: "严重故障：只保存答案，事后补评估", uiBadge: "评估稍后生成" },
];

export const MAX_DEGRADATION_LEVEL = 6;

export interface DegradationBadge {
  seq: number;
  ts: number;
  level: number;
  reason: string;
  badge: string;
  message: string;
}

// ------------------------------------------------------------------ 事件

/** 服务端文本帧事件（信封解出来的领域事件）。 */
export interface EnvelopeEvent {
  kind: "envelope";
  seq: number;
  ts: number;
  sessionId: string;
  type: string;
  payload: Record<string, unknown>;
}

/** TTS 音频块（二进制 opcode 0x11）。只记元信息入日志，音频字节交给播放器。 */
export interface TtsChunkEvent {
  kind: "tts_chunk";
  seq: number;
  ts: number;
  sessionId: string;
  byteLength: number;
  sampleRate?: number;
}

/** viseme 时间轴（文本帧 avatar.viseme 或二进制 opcode 0x12）。 */
export interface VisemeTimelineEvent {
  kind: "viseme";
  seq: number;
  ts: number;
  sessionId: string;
  visemes: VisemeEvent[];
}

export type ServerEvent = EnvelopeEvent | TtsChunkEvent | VisemeTimelineEvent;

/** 与 `ports.py::VisemeEvent` 一致：t_ms 相对音频起点，与音频同源时钟。 */
export interface VisemeEvent {
  t_ms: number;
  viseme: string;
  weight?: number;
}

// ------------------------------------------------------------------ 派生状态

export interface Transition {
  from: string | null;
  to: string;
  reason: string;
  seq: number;
  ts: number;
}

export interface QuestionView {
  turnId: string;
  turnIndex: number;
  text: string;
  /** 缓冲题不计分（states.py：BUFFER 系列无通往评分的路径）。 */
  scored: boolean;
  questionType?: string;
}

export interface TurnView {
  turnId: string;
  turnIndex: number;
  question: string;
  scored: boolean;
  /** 已 final 的转写文本；为空表示本轮未走语音。 */
  finalTranscript: string;
  eval: TurnEvalView | null;
}

export interface TurnEvalView {
  dims: Record<string, number>;
  score: number | null;
  feedback: string;
  provider: string;
  confidence: number;
  degraded: boolean;
  degradeReason: string | null;
  latencyMs: number;
}

/** 心率样本。`bpm` 为 null 表示"测不准"——与后端 NoSignal/低置信口径一致。 */
export interface HrSample {
  ts: number;
  bpm: number | null;
  confidence: number;
  reason?: string;
}

export interface MediaStatus {
  cam: "off" | "ready" | "denied" | "error";
  mic: "off" | "ready" | "denied" | "error";
  screen: "off" | "ready" | "denied" | "error";
}

export interface PhysioResult {
  available: boolean;
  reason: string | null;
  score: number | null;
  reliability: number;
  weightApplied: number;
  nValid: number;
  baselineBpm: number | null;
  t50MedianS: number | null;
  habituationSlope: number | null;
  reactivityRatioMean: number | null;
  consistencySigma: number | null;
  components: Record<string, number | null>;
}

export interface ReportView {
  sessionId: string;
  /** false 时 score 必为 null：没有任何失败路径会产出"看起来正常"的分数。 */
  available: boolean;
  reason: string | null;
  score: number | null;
  level: string | null;
  dims: Record<string, number>;
  effectiveWeights: Record<string, number>;
  redistributed: Record<string, number>;
  /** 反事实：完全剔除某维度后的总分。NaN 在后端被写成 null。 */
  counterfactuals: Record<string, number | null>;
  notes: string[];
  physio: PhysioResult | null;
  rubricVersion: string;
  nScoredTurns: number;
  nBufferTurns: number;
}

export interface ErrorItem {
  seq: number;
  ts: number;
  code: string;
  message: string;
  detail: string | null;
}

export interface DerivedState {
  sessionId: string;
  fsm: {
    state: FsmState;
    from: FsmState | null;
    reason: string;
    turnIdx: number;
    history: Transition[];
  };
  question: QuestionView | null;
  turns: Record<string, TurnView>;
  transcript: { partial: string; finals: string[] };
  hr: { latest: HrSample | null; samples: HrSample[] };
  degradation: {
    /** 取已发生事件的**最高等级**（与 DegradationTracker 同口径：不因后续恢复被抹掉）。 */
    level: number;
    current: DegradationBadge | null;
    badges: DegradationBadge[];
  };
  report: ReportView | null;
  errors: ErrorItem[];
  tts: { chunks: number; bytes: number };
  visemes: VisemeEvent[];
  mediaAck: { consumed: number; credits: number } | null;
  timedOut: boolean;
}

/** 心率样本保留上限（有界内存；确定性截断，不影响重放结果）。 */
export const MAX_HR_SAMPLES = 200;
