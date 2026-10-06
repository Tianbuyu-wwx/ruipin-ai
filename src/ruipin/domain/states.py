"""面试状态机的状态与事件定义（唯一真相源）。

设计要点：
1. 状态/事件只在这里定义一次，禁止散落赋值（现系统的反面教材）。
2. BUFFER 是独立状态：缓冲题不计分，且从 BUFFER 不存在通往评分的路径
   —— 这把"适应性只能作用于不计分部分"的硬规则编码进了状态机本身
   （对应详设 §5.3.1 / 自检 A3）。
"""

from enum import Enum


class State(str, Enum):
    IDLE = "idle"
    SETUP = "setup"
    GREETING = "greeting"
    ASKING = "asking"
    BUFFER = "buffer"          # 缓冲题呈现中（不计分）
    # 缓冲题作答中。**刻意不与 LISTENING 合并**：合并后 BUFFER→LISTENING→PROCESSING
    # →EVAL_DONE 会成为一条合法路径，"缓冲题不计分"就退化成口头纪律而非结构保证。
    # 独立出来之后，从 BUFFER 出发**根本到不了 PROCESSING**，也就没有任何评分时机。
    BUFFER_LISTENING = "buffer_listening"
    LISTENING = "listening"
    PROCESSING = "processing"
    FOLLOWUP = "followup"
    CANDIDATE_QA = "candidate_qa"
    CLOSING = "closing"
    REPORTING = "reporting"
    COMPLETED = "completed"
    ABORTED = "aborted"
    FAILED = "failed"


class Event(str, Enum):
    CREATE = "create"
    CONSENT_OK = "consent_ok"
    CONSENT_DENIED = "consent_denied"
    GREETING_DONE = "greeting_done"

    TTS_DONE = "tts_done"
    TTS_FAILED = "tts_failed"

    ANSWER_COMMIT = "answer_commit"
    SILENCE_TIMEOUT = "silence_timeout"
    SKIP = "skip"
    REPEAT_QUESTION = "repeat_question"
    EXTEND_TIME = "extend_time"

    # 适应性调节（自检 A5 补充）
    INSERT_BUFFER = "insert_buffer"
    BUFFER_DONE = "buffer_done"

    EVAL_DONE = "eval_done"
    EVAL_DEGRADED = "eval_degraded"

    FOLLOWUP_DECISION = "followup_decision"
    NEXT_QUESTION = "next_question"
    NO_MORE_QUESTIONS = "no_more_questions"

    QA_DONE = "qa_done"
    CLOSING_DONE = "closing_done"
    REPORT_DONE = "report_done"

    ABORT = "abort"
    FATAL = "fatal"


ALL_STATES = tuple(State)
ALL_EVENTS = tuple(Event)
