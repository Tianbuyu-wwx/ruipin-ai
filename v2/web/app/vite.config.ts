import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

/**
 * 后端 WS 网关的开发期代理目标。
 *
 * 为什么允许从环境变量注入：这个地址原先**硬编码**在配置里，于是它同时存在于
 * 三处（本文件、`App.tsx` 的兜底 URL、后端 `--port`）。一旦后端换端口，
 * 代理会连到一个没人监听的端口上，而浏览器只报「握手前连接就被关闭」——
 * 看不出是端口对不上。端到端探针需要把后端起在别的端口上（避开残留实例），
 * 没有这个开关就只能去改源码。
 *
 * 这里不发 `process.env`：本项目**没装 `@types/node`**（`src/` 里不需要 Node 全局），
 * 而 `vite.config.ts` 是被 `tsconfig` 一起检查的。用 `globalThis` 上的一次窄化读取
 * 拿到同一个值，就不用为了这一行往 devDependencies 里加 `@types/node`。
 */
const nodeEnv = (globalThis as { process?: { env?: Record<string, string | undefined> } }).process?.env;
const WS_TARGET = nodeEnv?.VITE_WS_TARGET || "ws://127.0.0.1:8787";
// HTTP 旁路端点（/health、/config/llm）与 WS 同端口；http-proxy 的 target 要 http:// 形态。
const HTTP_TARGET = WS_TARGET.replace(/^ws(s?):/, "http$1:");

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // dev 期用 vite proxy 转发 /ws，避免跨域与协议差异。
    // /health 与 /config/llm 是后端在 WS 端口上旁挂的 HTTP 端点（设置页用）。
    proxy: {
      "/ws": {
        target: WS_TARGET,
        ws: true,
        changeOrigin: true,
      },
      "/config": {
        target: HTTP_TARGET,
        changeOrigin: true,
      },
      "/health": {
        target: HTTP_TARGET,
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: "dist",
    sourcemap: true,
  },
  test: {
    environment: "node",
    include: ["src/**/*.test.ts", "src/**/*.test.tsx"],
    // Windows 上 Vitest 会并发把转译后的模块写入临时目录的同一缓存文件，
    // 触发 EPERM（文件被占用）并让 `vitest run` 以非零码退出。单线程 + 关闭文件级
    // 并行可稳定规避该竞态；本项目用例量小，无并行收益损失。
    pool: "threads",
    poolOptions: {
      threads: { singleThread: true },
    },
    fileParallelism: false,
    coverage: {
      provider: "v8",
      include: ["src/store/**/*.ts", "src/net/**/*.ts", "src/protocol/**/*.ts", "src/avatar/**/*.ts", "src/consent/**/*.ts", "src/report/**/*.ts"],
      reporter: ["text", "html"],
    },
  },
});
