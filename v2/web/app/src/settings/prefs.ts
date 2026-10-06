/**
 * 用户偏好（设置页数据层）—— 纯函数，独立于 DOM，便于单测。
 *
 * 职责：
 * 1. `Prefs` 的读取/写入（storage 由外部注入，测试传内存实现）；
 * 2. 坏数据兜底：JSON 损坏、缺字段、类型不对都回落默认值，绝不抛错；
 * 3. WS 地址解析：用户设置优先于构建期 env，最后回落到当前页面域名。
 *
 * 隐私边界：令牌只存本机 localStorage，不上传（它本来就只在本页与后端之间用）。
 */

/** 存储接口（localStorage 的最小子集，测试可注入内存实现）。 */
export interface PrefsStorage {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
}

export interface Prefs {
  /** 覆盖 WS 地址（ws:// 或 wss://）；留空表示跟随页面域名 / 构建期配置。 */
  wsUrl: string;
  /** 开发期令牌（拼到 ?token= 上）。 */
  token: string;
  /** 授权页"摄像头"初始勾选。 */
  presetCamera: boolean;
  /** 授权页"生理信号"初始勾选（依赖摄像头，勾上会自动带起摄像头）。 */
  presetPhysiology: boolean;
  /** 授权页"共享屏幕"初始勾选。 */
  presetScreen: boolean;
  /** AI 评分服务地址（OpenAI 兼容）。留空 = 用规则评分。 */
  llmBaseUrl: string;
  /** AI 评分 API Key。只存本机，保存时经请求头推给服务端，不落日志。 */
  llmApiKey: string;
  /** AI 评分模型名。 */
  llmModel: string;
  /** 应聘岗位（会话创建时带给服务端，影响出题方向）。 */
  position: string;
}

export const DEFAULT_PREFS: Prefs = {
  wsUrl: "",
  token: "",
  presetCamera: false,
  presetPhysiology: false,
  presetScreen: false,
  llmBaseUrl: "",
  llmApiKey: "",
  llmModel: "",
  position: "",
};

export const PREFS_STORAGE_KEY = "ruipin.prefs.v1";

/** 解析条件：主字段存在且类型正确。 */
function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

/** 字符串字段：非字符串或缺失回落默认。 */
function str(source: Record<string, unknown>, key: keyof Prefs, fallback: string): string {
  const raw = source[key];
  return typeof raw === "string" ? raw : fallback;
}

/** 布尔字段：非布尔或缺失回落默认。 */
function bool(source: Record<string, unknown>, key: keyof Prefs, fallback: boolean): boolean {
  const raw = source[key];
  return typeof raw === "boolean" ? raw : fallback;
}

/**
 * 从 storage 读偏好。任何异常（storage 拒绝访问、JSON 损坏）都回落默认值。
 * 缺字段的旧版本数据与默认值合并，不做整体作废。
 */
export function loadPrefs(storage: PrefsStorage | null): Prefs {
  if (!storage) return { ...DEFAULT_PREFS };
  let raw: string | null = null;
  try {
    raw = storage.getItem(PREFS_STORAGE_KEY);
  } catch {
    return { ...DEFAULT_PREFS };
  }
  if (!raw) return { ...DEFAULT_PREFS };
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch {
    return { ...DEFAULT_PREFS };
  }
  if (!isRecord(parsed)) return { ...DEFAULT_PREFS };
  return {
    wsUrl: str(parsed, "wsUrl", DEFAULT_PREFS.wsUrl),
    token: str(parsed, "token", DEFAULT_PREFS.token),
    presetCamera: bool(parsed, "presetCamera", DEFAULT_PREFS.presetCamera),
    presetPhysiology: bool(parsed, "presetPhysiology", DEFAULT_PREFS.presetPhysiology),
    presetScreen: bool(parsed, "presetScreen", DEFAULT_PREFS.presetScreen),
    llmBaseUrl: str(parsed, "llmBaseUrl", DEFAULT_PREFS.llmBaseUrl),
    llmApiKey: str(parsed, "llmApiKey", DEFAULT_PREFS.llmApiKey),
    llmModel: str(parsed, "llmModel", DEFAULT_PREFS.llmModel),
    position: str(parsed, "position", DEFAULT_PREFS.position),
  };
}

/** 写入 storage。storage 不可用时静默放弃（设置丢就丢，不能挡住面试主流程）。 */
export function savePrefs(storage: PrefsStorage | null, prefs: Prefs): void {
  if (!storage) return;
  try {
    storage.setItem(PREFS_STORAGE_KEY, JSON.stringify(prefs));
  } catch {
    /* 存储被禁用（隐私模式/配额满）：保持现状即可 */
  }
}

/**
 * 校验用户填写的 WS 地址。合法返回 null，不合法返回可读原因。
 * 空串合法（表示"跟随默认"）。
 */
export function validateWsUrl(value: string): string | null {
  const trimmed = value.trim();
  if (!trimmed) return null;
  if (!/^wss?:\/\//.test(trimmed)) return "地址要以 ws:// 或 wss:// 开头";
  return null;
}

/**
 * 校验 AI 评分三项。合法返回 null；三项要么都填要么都空，
 * 半套会让"到底用没用真 AI"变得不可判断，直接拒掉。
 */
export function validateLlmConfig(baseUrl: string, apiKey: string, model: string): string | null {
  const filled = [baseUrl.trim(), apiKey.trim(), model.trim()].map((s) => s.length > 0);
  if (filled.every(Boolean) || filled.every((b) => !b)) return null;
  const names = ["服务地址", "API Key", "模型名"];
  const missing = names.filter((_, i) => !filled[i]).join("、");
  return `AI 评分配置不完整：还差${missing}。三项要么都填，要么都留空`;
}

/** resolveWsUrl 的页面环境（生产里传 location，测试传字面量）。 */
export interface PageEnv {
  protocol: string;
  host: string;
}

/**
 * 拼最终 WS 地址。优先级：用户设置 > 构建期 env（VITE_WS_URL）> 当前页面域名。
 * 令牌追加在 query 上（浏览器 WebSocket 不能自定义 header，后端 query 也收）。
 */
export function resolveWsUrl(
  prefs: Prefs,
  envUrl: string | undefined,
  page: PageEnv | undefined,
): string {
  const override = prefs.wsUrl.trim();
  const base = override || envUrl?.trim() || (page ? `ws://${page.host}/ws` : "ws://127.0.0.1:8787/ws");
  const token = prefs.token.trim();
  if (!token) return base;
  return `${base}${base.includes("?") ? "&" : "?"}token=${encodeURIComponent(token)}`;
}
