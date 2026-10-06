/**
 * 设置数据层单测：往返持久化、坏数据兜底、URL 校验与解析优先级。
 * storage 一律注入内存实现，不碰真实 localStorage。
 */

import { describe, expect, it } from "vitest";
import {
  DEFAULT_PREFS,
  type Prefs,
  type PrefsStorage,
  loadPrefs,
  resolveWsUrl,
  savePrefs,
  validateLlmConfig,
  validateWsUrl,
} from "./prefs";

function memoryStorage(): PrefsStorage & { map: Map<string, string> } {
  const map = new Map<string, string>();
  return {
    map,
    getItem: (k) => (map.has(k) ? (map.get(k) as string) : null),
    setItem: (k, v) => void map.set(k, v),
  };
}

/** 构造一个会抛错的 storage（隐私模式模拟）。 */
function throwingStorage(): PrefsStorage {
  return {
    getItem: () => {
      throw new Error("denied");
    },
    setItem: () => {
      throw new Error("denied");
    },
  };
}

describe("loadPrefs", () => {
  it("空 storage 回到全默认", () => {
    expect(loadPrefs(memoryStorage())).toEqual(DEFAULT_PREFS);
  });

  it("null storage（SSR/异常环境）回到全默认", () => {
    expect(loadPrefs(null)).toEqual(DEFAULT_PREFS);
  });

  it("savePrefs 后 loadPrefs 完整往返", () => {
    const store = memoryStorage();
    const prefs: Prefs = {
      wsUrl: "ws://192.168.1.9:8787/ws",
      token: "dev-token-1",
      presetCamera: true,
      presetPhysiology: true,
      presetScreen: false,
      llmBaseUrl: "https://api.example.com",
      llmApiKey: "sk-roundtrip",
      llmModel: "deepseek-chat",
      position: "后端工程师",
    };
    savePrefs(store, prefs);
    expect(loadPrefs(store)).toEqual(prefs);
  });

  it("JSON 损坏回落默认，不抛错", () => {
    const store = memoryStorage();
    store.setItem("ruipin.prefs.v1", "{not json");
    expect(loadPrefs(store)).toEqual(DEFAULT_PREFS);
  });

  it("非对象 JSON（数字/字符串）回落默认", () => {
    const store = memoryStorage();
    store.setItem("ruipin.prefs.v1", "42");
    expect(loadPrefs(store)).toEqual(DEFAULT_PREFS);
  });

  it("缺字段与默认值合并（旧版本数据不整体作废）", () => {
    const store = memoryStorage();
    store.setItem("ruipin.prefs.v1", JSON.stringify({ wsUrl: "ws://10.0.0.2:8787/ws", token: 123 }));
    expect(loadPrefs(store)).toEqual({
      ...DEFAULT_PREFS,
      wsUrl: "ws://10.0.0.2:8787/ws",
      token: "",
    });
  });

  it("storage 抛错时回落默认且不外泄异常", () => {
    expect(loadPrefs(throwingStorage())).toEqual(DEFAULT_PREFS);
  });
});

describe("savePrefs", () => {
  it("storage 抛错时静默放弃，不外泄异常", () => {
    expect(() => savePrefs(throwingStorage(), DEFAULT_PREFS)).not.toThrow();
  });

  it("null storage 不写入也不抛错", () => {
    expect(() => savePrefs(null, DEFAULT_PREFS)).not.toThrow();
  });
});

describe("validateWsUrl", () => {
  it("空串合法（跟随默认）", () => {
    expect(validateWsUrl("")).toBeNull();
    expect(validateWsUrl("   ")).toBeNull();
  });

  it("ws:// 与 wss:// 合法", () => {
    expect(validateWsUrl("ws://127.0.0.1:8787/ws")).toBeNull();
    expect(validateWsUrl("wss://example.com/ws")).toBeNull();
  });

  it("http/https 与裸域名给出原因", () => {
    expect(validateWsUrl("http://127.0.0.1:8787/ws")).toMatch(/ws:\/\//);
    expect(validateWsUrl("127.0.0.1:8787")).toMatch(/ws:\/\//);
  });
});

describe("resolveWsUrl", () => {
  const page = { protocol: "https:", host: "demo.example.com" };

  it("用户设置优先级最高", () => {
    const prefs: Prefs = { ...DEFAULT_PREFS, wsUrl: "ws://192.168.1.9:8787/ws" };
    expect(resolveWsUrl(prefs, "wss://env.example.com/ws", page)).toBe("ws://192.168.1.9:8787/ws");
  });

  it("无用户设置时用 env，再无 env 回落页面域名", () => {
    expect(resolveWsUrl(DEFAULT_PREFS, "wss://env.example.com/ws", page)).toBe("wss://env.example.com/ws");
    expect(resolveWsUrl(DEFAULT_PREFS, undefined, page)).toBe("ws://demo.example.com/ws");
  });

  it("https 页面不带 env 时不再出现（回落 host 即可，协议由 override/env 决定）", () => {
    const url = resolveWsUrl(DEFAULT_PREFS, undefined, page);
    expect(url.startsWith("ws://demo.example.com")).toBe(true);
  });

  it("令牌拼到 query，且地址已有 query 时用 & 连接", () => {
    const withToken = { ...DEFAULT_PREFS, token: "abc 123" };
    expect(resolveWsUrl(withToken, undefined, page)).toBe("ws://demo.example.com/ws?token=abc%20123");
    const presetQuery = { ...withToken, wsUrl: "ws://x.example/ws?room=1" };
    expect(resolveWsUrl(presetQuery, undefined, page)).toBe("ws://x.example/ws?room=1&token=abc%20123");
  });

  it("无页面环境（非浏览器）回落本机默认端口 8787", () => {
    expect(resolveWsUrl(DEFAULT_PREFS, undefined, undefined)).toBe("ws://127.0.0.1:8787/ws");
  });
});

describe("validateLlmConfig", () => {
  it("三项全空或全填都合法", () => {
    expect(validateLlmConfig("", "", "")).toBeNull();
    expect(validateLlmConfig("https://a", "sk-1", "m")).toBeNull();
    expect(validateLlmConfig("  ", "  ", " ")).toBeNull();
  });

  it("半套给出缺哪一项", () => {
    expect(validateLlmConfig("https://a", "", "m")).toMatch(/API Key/);
    expect(validateLlmConfig("", "sk-1", "m")).toMatch(/服务地址/);
    expect(validateLlmConfig("https://a", "sk-1", "")).toMatch(/模型名/);
  });
});
