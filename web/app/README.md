# 锐聘 AI 模拟面试系统 v2 —— 前端（最小可跑版）

React 18 + TypeScript + Vite + Zustand 的最小可跑前端：单条 WebSocket + 事件溯源 store +
2D rig 形象 + 分项授权 + 报告页。**与后端 `v2/src/ruipin` 共用同一份 WS 协议**
（`transport/protocol.py`），协议常量与帧布局以 Python 侧为唯一真相源。

## 快速开始

```bash
cd v2/web/app
npm install          # 首次
npm run dev          # 起 dev server（默认 http://localhost:5173）
npm run build        # tsc --noEmit && vite build（产物在 dist/）
npx vitest run       # 单元测试
npx vitest run --coverage   # 覆盖率报告（html 在 coverage/）
```

### 连接后端 WebSocket

- 前端默认连 `ws://<当前 host>/ws`，由 Vite dev proxy 转发到后端网关
  `ws://127.0.0.1:8787`（见 `vite.config.ts` 的 `server.proxy`）。
- 覆盖地址：启动时设环境变量
  ```bash
  VITE_WS_URL=ws://127.0.0.1:9000/ws npm run dev
  ```
- 端口在前后端各写了一遍（这里是 `vite.config.ts` + `App.tsx` 的兜底 URL，
  后端是 `adapters/ws_server.py` 的 `DEFAULT_PORT`）。`v2/tests/test_adapters_ws_server.py`
  有一处用例**读这两个前端源文件**并机械比对，改了单边就会红。

后端这样起（开发装配：假评估器 + 内存存储 + 静态令牌）：

```bash
cd v2
./.venv/Scripts/python.exe -m ruipin.adapters.ws_server --port 8787 --token dev-token
```

## 代码结构

```
src/
  protocol/   constants.ts（协议常量·唯一真相源）codec.ts（帧编解码）messages.ts（上行构造）
  net/        creditWindow.ts（信用窗口）socket.ts（WS 客户端：多路复用/重连/背压/心跳）
  store/      types.ts（领域类型）derive.ts（事件溯源纯函数）interviewStore.ts（Zustand）selectors.ts
  avatar/     timeline.ts（viseme→口型）audioPlayer.ts（AudioContext 时钟+抖动缓冲+纯文本退化）
  consent/    logic.ts（camera/physiology/screen 分项与依赖约束）
  report/     parse.ts（report.ready 解析 + 生理红线）dims.ts（六维 + 生理维）
  components/ ConsentDialog / Avatar / InterviewRoom / ReportView
  App.tsx main.tsx index.css
```

## 与后端协议的对应关系

- 二进制帧布局：`[opcode(1B)][length(4B 大端)][payload]`，`HEADER_SIZE=5`
  （见 `src/protocol/constants.ts` 顶部注释，与 `protocol.py` 逐项对应）。
- opcode：`0x01` audio / `0x02` video / `0x03` screen（上行）、`0x11` TTS / `0x12` viseme（下行）。
- 信封字段与线序：`v / type / seq / ts / session_id / payload`（同 `Envelope.to_dict()`）。
- 信用窗口：默认容量 16、每消费 4 帧回一个 `media.ack`（同 `gateway.DEFAULT_*`）。
- 状态与事件枚举：字面值与 `domain/states.py` 的 15 个 `State` / 23 个 `Event` 一致。

## 已实现

- **事件溯源**：store 只追加原始事件，`derive(events)` 现算 UI 状态；按 `seq` 去重，
  断线重放幂等。
- **WS 客户端**：文本/二进制多路复用、自动重连（指数退避）、重连后发送 `resume` 请求重放、
  信用窗口背压（窗口满暂停入队、`media.ack` 后按序续发）、心跳看门狗（长时间无入站帧强制重连）。
- **授权**：camera / physiology / screen 分项；生理需单独同意（PIPL 第 28 条）；
  拒绝 camera ⇒ 生理自动禁用并给出原因；只同意"参加面试"即可开始。
- **形象**：SVG 2D rig，按 `avatar.viseme` 时间轴以 `AudioContext.currentTime` 统一时钟驱动口型；
  音频不可用时口型停待机并以字幕呈现文本，流程不卡。
- **降级**：`degradation.changed` 触发 `role="status" aria-live` 横幅，文案保证非空（不静默）。
- **报告**：六维 + 总分 + 等级 + 生理维度；生理不可用时显示"未计入（原因：…）"且**不出现数字**；
  展示 `counterfactuals`（关闭某维度后的总分）。
- **无障碍**：交互可键盘操作（作答框 Ctrl/⌘+Enter 提交）、关键区域 `role=status`/`aria-live`、
  颜色不作为唯一信息载体。

## 尚未接入（需后端联调 / 后续）

1. **录音作答**：界面标注"录音未接入"，仅文本作答（`recordingAvailable=false`）。
2. **真实 TTS 音频**：`socket.onAudioChunk` 已把 0x11 原始字节交给 `AudioPlayer`，
   但 `AudioPlayer` 目前按 **16-bit 单声道 PCM** 解码，且真正的音频编码/采样率需与
   `tts-worker` 对齐；未对齐前音频按纯文本降级呈现（不伪造播放）。
3. **viseme 二进制 payload 格式**：`avatar.viseme` 文本帧与 0x12 二进制帧均按
   `{"visemes":[{t_ms,viseme,weight}]}`（或裸数组）解析，需与 `avatar-worker` 确认。
4. **组件级 DOM 测试**：当前只测纯逻辑（store/net/protocol/avatar/consent/report），
   未引入 jsdom；如需渲染测试再加 `jsdom`。
5. **后端起服务**：需要把 `Gateway` 挂到真实 WS server 并桥接 `InterviewService` 下行。

## 环境注意（Windows）

Vitest 在 Windows 上会并发把转译模块写入临时目录的同一缓存文件，触发 `EPERM` 使
`vitest run` 以非零码退出。已在 `vite.config.ts` 的 `test` 中用
`pool: "threads" + singleThread + fileParallelism: false` 规避；本项目用例量小，无并行损失。
