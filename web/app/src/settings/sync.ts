/**
 * 设置到服务端的同步 —— AI 评分配置推给后端热更新。
 *
 * 通道：`GET /config/llm` + 自定义请求头（websockets 库读不到 POST body，
 * 服务端只能从头里收）。key 只经内存请求头，不进 URL、不进日志。
 * 开发环境下 `/config` 由 vite 代理转发到后端；服务端不可达时返回 null，
 * 调用方要显式告知用户"只存了本机"，不能装作同步成功。
 */

import type { Prefs } from "./prefs";

export interface SyncResult {
  ok: boolean;
  message: string;
}

type FetchLike = (input: string, init?: { headers?: Record<string, string> }) => Promise<Response>;

/**
 * 推送 AI 评分配置。任何网络层失败都收敛为 null（服务端不可达），
 * HTTP 层的失败（401/400/500）解析出服务端消息。
 */
export async function pushLlmConfig(
  prefs: Prefs,
  fetchImpl: FetchLike = fetch as unknown as FetchLike,
): Promise<SyncResult | null> {
  const token = prefs.token.trim();
  const url = token ? `/config/llm?token=${encodeURIComponent(token)}` : "/config/llm";
  let res: Response;
  try {
    res = await fetchImpl(url, {
      headers: {
        "X-LLM-Base-URL": prefs.llmBaseUrl.trim(),
        "X-LLM-API-Key": prefs.llmApiKey.trim(),
        "X-LLM-Model": prefs.llmModel.trim(),
      },
    });
  } catch {
    return null;
  }
  let message = `服务端返回 ${res.status}`;
  try {
    const body = (await res.json()) as { ok?: boolean; message?: string };
    if (typeof body.message === "string" && body.message) message = body.message;
    if (res.ok && typeof body.ok === "boolean") {
      return { ok: body.ok, message };
    }
  } catch {
    /* 响应不是 JSON：用默认 message */
  }
  return { ok: false, message };
}
