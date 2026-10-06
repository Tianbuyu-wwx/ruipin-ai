/**
 * 设置同步单测：fetch 注入内存桩，覆盖成功/HTTP 失败/网络不可达/坏 JSON。
 */

import { describe, expect, it } from "vitest";
import { DEFAULT_PREFS, type Prefs } from "./prefs";
import { pushLlmConfig } from "./sync";

function prefs(over: Partial<Prefs> = {}): Prefs {
  return { ...DEFAULT_PREFS, ...over };
}

function jsonRes(status: number, body: unknown): Response {
  return {
    ok: status < 400,
    status,
    json: async () => body,
  } as unknown as Response;
}

describe("pushLlmConfig", () => {
  it("带上三个请求头与令牌 query", async () => {
    let captured: { url: string; headers?: Record<string, string> } | null = null;
    const res = await pushLlmConfig(
      prefs({ token: "t 1", llmBaseUrl: "https://a", llmApiKey: "k", llmModel: "m" }),
      async (url, init) => {
        captured = { url, headers: init?.headers };
        return jsonRes(200, { ok: true, message: "AI 评分已启用" });
      },
    );
    expect((captured as { url: string } | null)!.url).toBe("/config/llm?token=t%201");
    expect((captured as { headers: Record<string, string> } | null)!.headers["X-LLM-Base-URL"]).toBe("https://a");
    expect((captured as { headers: Record<string, string> } | null)!.headers["X-LLM-API-Key"]).toBe("k");
    expect(res).toEqual({ ok: true, message: "AI 评分已启用" });
  });

  it("无令牌时不带 query", async () => {
    let url = "";
    await pushLlmConfig(prefs(), async (u) => {
      url = u;
      return jsonRes(200, { ok: true, message: "x" });
    });
    expect(url).toBe("/config/llm");
  });

  it("HTTP 400 解析服务端消息", async () => {
    const res = await pushLlmConfig(prefs(), async () => jsonRes(400, { ok: false, message: "配置不完整" }));
    expect(res).toEqual({ ok: false, message: "配置不完整" });
  });

  it("网络不可达返回 null（调用方须显式告知未同步）", async () => {
    const res = await pushLlmConfig(prefs(), async () => {
      throw new Error("refused");
    });
    expect(res).toBeNull();
  });

  it("非 JSON 响应用默认消息且按 ok=false 处理", async () => {
    const bad = {
      ok: false,
      status: 500,
      json: async () => {
        throw new Error("not json");
      },
    } as unknown as Response;
    const res = await pushLlmConfig(prefs(), async () => bad);
    expect(res).toEqual({ ok: false, message: "服务端返回 500" });
  });
});
