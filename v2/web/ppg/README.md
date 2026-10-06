# web/ppg — 摄像头心率（rPPG）端侧信号链

> 面向 AI 模拟面试系统的**浏览器端** rPPG 信号链，零依赖 TypeScript。
> 权威规范：`docs/模块详设-心率与压力调节评估.md` 第 3 章「信号链路详细设计」与第 7 章「光照与运动鲁棒性」。
> **凡本文与规范冲突之处，以规范为准；规范未定义之处，本文显式标注为「本实现补充」。**

---

## ⚠️ 尚未验证 / 需要真实数据（**必读，不得对外称"已完成"**）

**本模块目前只在合成信号上通过了单元/集成测试，尚未经过任何真实摄像头与真人验证。**
以下工作**全部未做**，且都是上线前的强制前置：

- [ ] **30 人金标准 POC**：与指夹式脉搏血氧仪同步对比，覆盖 Fitzpatrick I–VI、戴镜/不戴镜、明亮/侧光/逆光/频闪 × 静坐/说话/转头（规范 §10.1）。放行线：明亮静坐 MAE ≤5 BPM；最差条件 MAE ≤10 BPM 且弃权率 ≥40%。
- [ ] **MMPD 多肤色数据集验证**：rPPG-Toolbox 的 TS-CAN 在 PURE 上 MAE 1.07、在 MMPD 上 **12.59**（差 10 倍）。**只报 PURE/UBFC 成绩的验证是自欺**（规范 §2.1）。
- [ ] **60–90 人分群公平性验证**：按肤色/性别/年龄/眼镜/妆容/光照/残障与面部差异逐群检验（规范 §6.1–6.2）。**结果无偏（Levene + Kruskal-Wallis，p>0.05）才准上线**。
- [ ] **真实摄像头 AE 漂移阈值标定**：`detectAeDrift` 的 15% / 8% 阈值来自文档与先验，**未在真实设备上验证**；`exposureMode:'manual'` 仅 Chrome 支持、Safari 不支持、Firefox 未实现 `getCapabilities`，锁定失败是常态（规范 §7）。
- [ ] **SNR 归一化常数标定**：`SNR_SPEC_LOG_FLOOR_RATIO` / `CEIL` / `SNR_HARM_*` 为**本实现补充**（规范 §3.8 只给了公式，未定义 `norm()`），必须在真实数据上重标定。
- [ ] **端侧算力验证**：本仓库在桌面 Node 上 900 帧耗时 ~47ms，**不代表手机浏览器 WASM 表现**；低端机降采样档（20fps）需实测。
- [ ] **真实部署精度预期**：合成信号上 MAE ≈0.06 BPM 是**链路自洽性**的证明，**不是**真实精度。真实部署预期为 **MAE 3–8 BPM**（规范 §1.1、§3 立场），请勿据此得出任何真实精度结论。

> **纪律**：测不准 = 输出"无信号"（`bpm = null`），绝不给错的值。合并前请确认这一点未被破坏（见 `pipeline.test.ts` 的"运动伪影硬断言"与"高噪声弃权"用例）。

---

## 文件结构

| 文件 | 职责 |
|---|---|
| `signal.ts` | 纯算法：去趋势、Butterworth 4 阶零相位带通、CHROM/POS/SSR、Welch PSD、峰值插值、含谐波校正的心率估计、带外主峰拒绝 |
| `quality.ts` | 窗口质量：SNR 三项（spec/harm/continuity）、三算法投票与弃权、AE 漂移/光照突变检测、窗口裁决 |
| `pipeline.ts` | 有状态滑动窗口管线：ROI RGB 时序 → 窗口 BPM 流；事件打点与窗口归属查询 |
| `index.ts` | 统一导出 |
| `synth.ts` | 测试用合成信号发生器（不参与打包，文件名不匹配 `*.test.ts`） |
| `*.test.ts` | Node 内置测试运行器用例 |

---

## 运行测试

```bash
# 项目根 E:/项目/锐聘AI/v2
"C:/Users/Tianbuyu/.workbuddy/binaries/node/versions/22.22.2-6/node.exe" \
  --experimental-strip-types --test "web/ppg/*.test.ts"
```

**注意**：
- 必须用 **glob 模式**（`"web/ppg/*.test.ts"`）。该 Node 版本把目录参数（`"web/ppg/"`）当作模块路径，会报 `MODULE_NOT_FOUND`，`.ts` 测试文件不会被收集。
- `web/ppg/` 目录下**不要放 `package.json`**，否则会被当成独立 npm 包。当前依赖 Node 的默认模块语法探测（Node ≥22.7 默认开启）把含 `import/export` 的 `.ts` 视作 ESM。
- `--experimental-strip-types` 只做类型擦除：**禁用 `enum` / `namespace` / 参数属性**；本仓库用 `as const` 常量对象代替枚举，跨文件类型用 `import type`。

---

## 与规范的对应关系

| 规范 | 实现位置 | 说明 |
|---|---|---|
| §3.4 去趋势（窗口 = 1.6×周期） | `detrendMovingAverage` | 前缀和 O(n)；窗口样本数由上一窗口心率推导 |
| §3.4 带通 0.7–4.0Hz，Butterworth 4 阶，零相位 | `bandpassFilter` / `filtfilt` | 二阶高通(0.7) × 二阶低通(4.0)，各自 filtfilt（镜像补边，前向+反向），合 4 阶 |
| §3.4 拒绝带外主峰 | `estimateHrFromPsd` + `hasDominantOutOfBandPeak` | 见下文「关键设计决定」 |
| §3.3 CHROM / POS / SSR | `chrom` / `pos` / `ssr` | 公式见 `signal.ts` 内注释（de Haan 2013 / Wang 2016 / Wang 2015） |
| §3.5 Welch PSD + 峰值插值 | `welchPSD` + `estimateHrFromPsd` | Hann 窗、重叠 75%、对数功率抛物线插值 |
| §3.5 谐波校验 | `estimateHrFromPsd` | 2·f0/3·f0 更强时回落到基频 |
| §3.3 三算法投票 | `voteAlgorithms` | ≤3 / 3–8 / >8 BPM 三档；任一算法无估计即弃权 |
| §3.8 SNR | `snrSpecRatio` / `snrHarmRatio` / `combineSnr` | 见下文公式对照 |
| §7 AE 漂移 / 光照突变 | `detectAeDrift` | 帧间突变 >15%、窗口前后半均值慢漂移 >8% |
| §4.1 事件时间轴 | `PpgPipeline.markEvent` / `eventWindow` / `windowsCovering` | 事件仅打点，不改动信号链路 |

---

## SNR 实现与规范 §3.8 的逐项对照

规范 §3.8 原文：

```
SNR_spec   = P(f0 ± 0.15Hz) / median(P(带内其他频率))
SNR_harm   = P(2·f0 ± 0.15Hz) / median(P(带内))
continuity = 满足 |HR_t − HR_{t−1}| ≤ 8 BPM 的步数占比
SNR        = 0.6·norm(SNR_spec) + 0.2·norm(SNR_harm) + 0.2·continuity   ∈ [0,1]
SNR < 0.35 → 该窗口判无效（弃权）
```

| 项 | 本实现 | 是否与规范一致 |
|---|---|---|
| `SNR_spec` 分子 | `P(f0±0.15Hz)` 取该窄带内 PSD 的**平均**（`bandPowerAt`） | ✅ 定义一致（"P"取均值 vs 积分，属等价量纲选择） |
| `SNR_spec` 分母 | 带内 `[0.7,4.0]` 排除 `f0±0.15` 后 PSD 的**中位数** | ✅ 一致 |
| `SNR_harm` 分子/分母 | `P(2f0±0.15)` / 带内 PSD 中位数；**2·f0 > 奈奎斯特时返回 `null`** | ✅ 一致，额外处理了规范未定义的越界情形（`null` → 中性 0.5） |
| `continuity` | 滚动历史（默认 5 窗）中 `|Δ|≤8 BPM` 的相邻非空对占比；无有效对时中性为 1 | ✅ 一致；"无历史"的处理为**本实现补充** |
| `norm()` | **规范未定义 → 本实现补充**：`normSpec` 用对数映射（比值跨数量级，线性会立即饱和）；`normHarm` 用线性映射 | ⚠️ **补充项，必须 POC 重标定** |
| 权重 0.6/0.2/0.2 | 与规范一致 | ✅ |
| 弃权阈值 0.35 | `QUALITY.SNR_REJECT_THRESHOLD = 0.35` | ✅ |

**归一化常数（本实现补充，需重标定）**：
`normSpec = clamp01((log10(spec) − log10(3)) / (log10(1e5) − log10(3)))`，
`normHarm = clamp01((harm − 1) / (10 − 1))`。

---

## 关键设计决定（含与文档的张力，均已注明）

1. **"窗口 8s，重叠 75%"的解读**：§3.5（Welch 重叠 75%）与 §3.8（每 8s 窗口）之间存在张力。本实现取 **分析窗 8s、相邻窗口步长 2s（= 75% 重叠）**；当分析窗显著长于 8s 时，`welchPSD` 的 `segmentLength` 才会产生多段平均。该解读使 §3.5、§3.8、§9（每 2s 一条）三者自洽。
2. **拒绝带外主峰需"守门谱"**：带通在截止附近并非砖墙——40 BPM(0.667Hz) 经 0.7Hz 高通仅衰减约 −7 dB，其能量会以"带内边缘"形态残留。故额外对**未带通、仅去均值**的全谱做带外峰检测，并要求带外峰既强于带内主峰、又相对其频段中位数显著（≥3×，排除宽带 1/f 漂移误判）。
3. **谐波校正的基频判据**：`f0/2` 或 `f0/3` 处平均功率 ≥ 主峰的 `HARMONIC_FUNDAMENTAL_MIN_RATIO`(=0.3，**本实现补充**) 时回落基频。已测两向：60BPM+更强二次谐波 → 60；真实 120BPM 无误折半到 60。
4. **"200 BPM 高于通带"是误述**：文档 §3.4 通带为 0.7–4.0 Hz = **42–240 BPM**，故 200 BPM(3.33Hz) **在带内**且可正常估计；真正越界的是 <42 或 >240。测试同时覆盖 40 / 200 / 260 BPM。
5. **SSR 的退化形式**：原 SSR 面向多像素/多子区域。本实现输入为单 ROI 平均的 3 通道序列，故对 3×3 协方差做确定性幂迭代取变化平面，再固定网格扫描旋转角取带内功率占比最大者。全过程无随机数。

---

## 合成信号实测数据（**链路自洽性证明，非真实精度**）

运行 `*.test.ts` 中的用例复现。以下为 Node v22.22.2 桌面环境实测（合成信号）：

| 真值 BPM | 有效窗 | MAE | 最大误差 | 中位 SNR |
|---|---|---|---|---|
| 60 | 7/7 | 0.057 | 0.059 | 0.810 |
| 72 | 7/7 | 0.083 | 0.087 | 0.800 |
| 90 | 7/7 | 0.085 | 0.086 | 0.800 |

| 噪声档 | 中位 SNR | 弃权率 | 有效窗 MAE |
|---|---|---|---|
| 0 | 0.800 | 0/7 | 0.08 |
| 0.5 | 0.531 | 0/7 | 0.12 |
| 1.5 | 0.471 | 0/7 | 0.23 |
| 4 | 0.364 | 3/7 | 0.27 |
| 8 | 0.316 | 7/7 | — |
| 15 | 0.300 | 7/7 | — |

- 带外：40 BPM、260 BPM 全部 `null`（弃权）。
- 谐波：60BPM + 二次谐波×1.5 → 输出 60。
- 性能：900 帧（30s@30fps）端到端 **~47ms**（桌面 Node，非 WASM）。

> ⚠️ 表中 MAE 是**合成信号**上的链路误差，**不能**与真实部署的 3–8 BPM 预期混淆。

---

## WASM / Worker 移植说明

本库刻意保持**纯函数 + 单类**、无 DOM 依赖、无随机数、无第三方包，便于直接编译到 WASM 或包进 Worker：

1. **编译路径**：`signal.ts` / `quality.ts` / `pipeline.ts` 为纯 TS。可用 `Emscripten`（C++ 重写热点）或 `AssemblyScript`（TS 子集）编译。**热点是 FFT 与 SSR 的旋转扫描**（每窗口 180 次带通），优先移植这两处。
2. **Worker 化**：`PpgPipeline` 无全局状态、`pushFrame` 只依赖入参，可直接放入 Web Worker：主线程只做 ROI 提取（FaceMesh/WASM）后 `postMessage({t_ms,r,g,b})`，Worker 回传 `hrStream()`。
3. **零相位滤波的补边**：`filtfilt` 用镜像补边（`pad = min(n-1, 32)`）。移植到 C/WASM 时保持相同补边策略，否则结果不再逐值一致（会破坏与 JS 基准的回归对比）。
4. **内存**：每帧 3 个 `f64`；环形缓冲 10s@30fps ≈ 900 帧 ≈ 21.6 KB，可在 Worker 内常驻。窗口分析临时数组每窗约 O(几 KB)，无长生命周期分配。
5. **确定性**：不引入 `Math.random`；`fftRadix2` 为定点顺序的迭代实现；SSR 幂迭代初值固定。移植时若用 SIMD/多线程，请注意浮点归约顺序变化会破坏"逐值一致"的确定性测试，如需保留确定性请固定归约顺序。
6. **降级档**：低端机可降采样到 20fps（`fs=20`）或增大 `stepSec`；`PpgPipeline` 已参数化。文档 §3.1：**fps<25 直接判不可用**（HRV 场景更高），此判断应在采集层而非本库完成。
7. **接口契约**：上游只需提供逐帧 ROI 平均 RGB（`{t_ms,r,g,b}`）；ROI 选取（FaceMesh 468 点、额头+双颊、饱和像素剔除、平滑跟踪）属 §3.2，由 WASM FaceMesh 层负责，**不在本库范围**。

---

## 使用示例

```ts
import { PpgPipeline } from './ppg/index.ts';

const p = new PpgPipeline({ fs: 30, windowSec: 8, stepSec: 2 });
p.markEvent({ id: 'q1-start', type: 'start', t_ms: 5000 });
p.markEvent({ id: 'q1-end', type: 'end', t_ms: 20000 });

// 采集循环：每帧 ROI 平均颜色
p.pushFrame({ t_ms: 0, r: 180, g: 140, b: 120 });
// ...
for (const s of p.hrStream()) {
  // s = { t_ms, bpm: number|null, snr, algo_spread, rejected }
  // bpm === null 即"本窗口无信号（弃权）"——请勿用任何默认值替代。
}

const evWin = p.eventWindow('q1-start'); // 事件落入的窗口
```
