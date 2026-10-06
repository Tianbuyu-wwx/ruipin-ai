# 锐聘 AI

AI 模拟面试系统：真人节奏的语音问答、规则/LLM 双路评分、可解释的面试报告。

后端是一条 WebSocket 实时链路（协议网关 → 会话编排 → 评分聚合 → 数字人语音），前端是一个 React 单页应用（授权 → 面试 → 报告）。**没有配置 LLM 时整套系统照常跑**：评分自动走规则引擎并在报告里写明"非 AI 评分"，TTS 不可用时退回纯文本——每一处降级都显式可见，不静默。

> 早期 Gradio 版本（`app.py`、`src/` 等）已从工作区移除，完整代码在 git 历史的首个提交里，需要时可 `git show` 查看。

## 功能

- **实时面试链路**：文本帧 JSON + 二进制音频/口型帧复用一条 WebSocket；下行帧统一 seq 编号，断线重连按 `last_seq` 重放，前端按 seq 去重。
- **评分**：LLM（OpenAI 兼容接口）优先，失败该轮自动降级规则评分（provider=rubric），降级原因随帧下发并在界面挂横幅；置信度不足的维度在聚合时权重归零，报告只给"可以当参考，不适合排名"口径的结论，不出伪精确排名。
- **数字人语音**：TTS 走 Qwen3-TTS 12Hz（HTTP 整段 WAV），词级时间戳来自 Qwen3-ForcedAligner；口型时间线按 viseme 二进制帧下发。对齐失败时音频照常交付、报告标注 `timing_source=none`。
- **端侧 rPPG**：摄像头画面在浏览器本地提取脉搏相关统计量（`v2/web/ppg/`，零运行时依赖），原始视频不上传；生理信号属敏感个人信息，单独勾选同意。
- **授权与合规**：摄像头 / 生理信号 / 屏幕共享分项同意（PIPL 口径），拒绝任何可选项都不影响面试与总分可比性（未采集维度按权重重分配）。
- **设置页**：服务地址、令牌、授权预设、AI 评分配置集中管理，localStorage 持久化，只存本机。

## 快速开始

依赖：Python 3.12+、Node 22+。后端与前端是两个进程。

### 1. 启动后端

```bash
cd v2
python -m venv .venv
.venv/Scripts/pip install websockets httpx pytest   # Linux/macOS 用 .venv/bin/

PYTHONPATH=src .venv/Scripts/python -m ruipin.adapters.deploy --port 8787 --token my-token
```

启动横幅会打印当前装配事实（评分来源、TTS 地址、令牌）。`--token` 不传则随机生成并打印。

### 2. 启动前端

```bash
cd v2/web/app
npm ci
npm run dev
```

浏览器打开 http://localhost:5173 ，在页面左下角"设置"里填入令牌（或直接用 `http://localhost:5173/?token=my-token` 进入），勾选授权项即可开始一场模拟面试。

### 3. 接入 AI 评分（可选）

两条路，任选：

- **设置页**（推荐）：打开"设置"，填 AI 评分的服务地址 / API Key / 模型名（OpenAI 兼容，如 DeepSeek），保存时自动推送给正在运行的服务端，对下一场面试生效。Key 只存本机 localStorage，推送走请求头，不进日志。
- **环境变量**：启动前设 `RUIPIN_LLM_BASE_URL` / `RUIPIN_LLM_API_KEY` / `RUIPIN_LLM_MODEL`（三项要么齐设要么全不设）。

### 4. 接入 TTS 语音（可选）

默认指向本机 `http://127.0.0.1:8091` 的 Qwen3-TTS 服务（vLLM-Omni 部署，选型与实测文档在本地 `docs/` 维护）。没有 TTS 服务时面试自动退回纯文本模式。环境变量 `RUIPIN_TTS_BASE_URL` / `RUIPIN_TTS_MODEL` / `RUIPIN_TTS_VOICE` 可覆盖，设为空串显式关闭。

## 架构

```
浏览器 (React)                         服务端 (Python, websockets)
┌─────────────────┐   ws://…/ws      ┌──────────────────────────────┐
│ ConsentDialog    │── 文本帧 JSON ──▶│ ws_server → session_server   │
│ InterviewRoom    │◀─ 事件/评估帧 ── │   → gateway(seq 编号/重放)   │
│ ReportView       │◀═ 二进制音频/口型 │   → bridge → InterviewService│
│ SettingsPanel    │                  │   → TurnScheduler → 评估器   │
└─────────────────┘                   └──────────┬───────────────────┘
        │ 本地 rPPG(摄像头逐帧)                  │ HTTP
        ▼                                       ▼
   生理统计量(不上传视频)               LLM(OpenAI 兼容) / Qwen3-TTS
```

协议要点：上行帧必须带 `v:1` 与单调 `seq`；二进制帧为 `[1B opcode][4B 大端长度][payload]`（0x11=TTS 音频、0x12=viseme）；`consent.grant` 不带 `base:true` 直接 ABORTED（设计如此）。服务端在 WS 端口上旁挂两个 HTTP 端点：`GET /health` 与 `GET /config/llm`（设置页热更新 AI 评分，需令牌）。

评估链是一个路由器：云 LLM 在前、规则评分在后，任何一轮 LLM 失败该轮自动落到规则评分并标降级；聚合时置信度低于阈值的维度权重归零（"宁缺毋滥"：一段 20 字的敷衍作答出不了分，是设计语义不是 bug）。

## 目录结构

```
v2/
├─ src/ruipin/
│  ├─ transport/        网关(seq/重放/限流)、会话循环、桥接
│  ├─ orchestrator/     轮次调度（每题一回合）
│  ├─ scoring/          规则评分(rubric)、聚合门控
│  ├─ adapters/         LLM/TTS/装配：deploy 入口、OpenAI 兼容评估、Qwen3-TTS
│  ├─ perception/ …     感知、生理、安全、留存等域模块
├─ tests/               后端测试（pytest，3200+ 用例，含 WS 契约与端点集成）
├─ web/app/             前端（React 18 + Vite + zustand，自写 CSS）
├─ web/ppg/             端侧 rPPG（TypeScript，Node 内置 test runner）
├─ web/mockup/          纯前端示意图（非产品代码）
└─ tools/               对齐器验收等开发工具
```

## 配置

环境变量（后端）：

| 变量 | 说明 |
| --- | --- |
| `RUIPIN_LLM_BASE_URL` / `RUIPIN_LLM_API_KEY` / `RUIPIN_LLM_MODEL` | AI 评分（OpenAI 兼容）。三项齐全启用，缺项整体走规则评分 |
| `RUIPIN_TTS_BASE_URL` / `RUIPIN_TTS_MODEL` / `RUIPIN_TTS_VOICE` | TTS（默认 `http://127.0.0.1:8091`），设空串显式关闭 |

前端本机偏好（服务地址覆盖、令牌、授权预设、AI 评分）在页面"设置"里改，存 localStorage。

本机开发约定：后端默认端口 **8787**（8000 在本机被其他服务占用）；`vite.config.ts` 的代理目标可用 `VITE_WS_TARGET` 覆盖（端到端探针靠它把后端起在别的端口）。

## 测试

```bash
# 后端全量（须带 PYTHONPATH=src）
cd v2 && PYTHONPATH=src python -m pytest

# 前端类型检查 + 单测
cd v2/web/app
npm run typecheck && npm test

# 端侧 rPPG
cd v2/web && node --experimental-strip-types --test "ppg/*.test.ts"
```

CI（GitHub Actions）跑以上三组，见 `.github/workflows/ci.yml`。

## 入库范围

仓库只收代码与 CI 配置。以下内容**不入库**（.gitignore 已排除）：`base_model/`（模型权重）、`data/*.db`（用户数据）、`.secret_key`（本机测试密钥）、`.workbuddy/`、日志、覆盖率产物、`v2/.venv`、`node_modules`、`docs/`（设计文档留在本地）。

## 状态

个人项目，持续开发中。WS 协议与报告格式在 v2 稳定前可能调整。
