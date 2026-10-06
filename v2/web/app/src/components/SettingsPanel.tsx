/**
 * 设置页 —— 集中管理本机可调配置。
 *
 * 三组设置：
 *  1. 连接：WS 地址与令牌（留空即跟随默认；令牌只存本机）；
 *  2. 授权预设：下次进入授权页时各可选项的初始勾选（不代替勾选，只省一次点击）；
 *  3. 说明：设置保存到哪、什么时候生效。
 *
 * 纪律：
 * - 改动不即时生效，显式点"保存"才落盘；保存结果用 role=status 明说，不静默；
 * - 地址校验不过就禁用保存，错误原因可读；
 * - 生理信号预设与摄像头预设的依赖关系复用 consent 逻辑（勾生理必须带摄像头）。
 */

import { useState, type FormEvent } from "react";
import { applyConsentToggle } from "../consent/logic";
import {
  type Prefs,
  savePrefs,
  validateLlmConfig,
  validateWsUrl,
} from "../settings/prefs";
import { pushLlmConfig } from "../settings/sync";

export interface SettingsPanelProps {
  initial: Prefs;
  /** 存储实现；测试注入内存版。生产由 App 传 window.localStorage。 */
  storage: Parameters<typeof savePrefs>[0];
  onBack: () => void;
}

interface Draft {
  wsUrl: string;
  token: string;
  presetCamera: boolean;
  presetPhysiology: boolean;
  presetScreen: boolean;
  llmBaseUrl: string;
  llmApiKey: string;
  llmModel: string;
}

export function SettingsPanel({ initial, storage, onBack }: SettingsPanelProps) {
  const [draft, setDraft] = useState<Draft>({ ...initial });
  const [saved, setSaved] = useState<string | null>(null);
  const [syncing, setSyncing] = useState(false);

  const wsError = validateWsUrl(draft.wsUrl);
  const llmError = validateLlmConfig(draft.llmBaseUrl, draft.llmApiKey, draft.llmModel);
  const canSave = wsError === null && llmError === null;

  const set = (patch: Partial<Draft>) => {
    setSaved(null);
    setDraft((prev) => ({ ...prev, ...patch }));
  };

  // 授权预设之间复用 consent 的依赖规则：关摄像头必须关生理，开生理必须带摄像头。
  const togglePreset = (key: "presetCamera" | "presetPhysiology" | "presetScreen", value: boolean) => {
    setSaved(null);
    setDraft((prev) => {
      const merged = applyConsentToggle(
        { base: true, camera: prev.presetCamera, physiology: prev.presetPhysiology, screen: prev.presetScreen },
        key === "presetCamera" ? "camera" : key === "presetPhysiology" ? "physiology" : "screen",
        value,
      );
      return {
        ...prev,
        presetCamera: merged.camera,
        presetPhysiology: merged.physiology,
        presetScreen: merged.screen,
      };
    });
  };

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    if (!canSave || syncing) return;
    const next: Prefs = {
      wsUrl: draft.wsUrl.trim(),
      token: draft.token.trim(),
      presetCamera: draft.presetCamera,
      presetPhysiology: draft.presetPhysiology,
      presetScreen: draft.presetScreen,
      llmBaseUrl: draft.llmBaseUrl.trim(),
      llmApiKey: draft.llmApiKey.trim(),
      llmModel: draft.llmModel.trim(),
    };
    savePrefs(storage, next);
    setSyncing(true);
    const sync = await pushLlmConfig(next);
    setSyncing(false);
    if (sync === null) {
      setSaved("已保存到本机。服务端现在连不上，AI 评分配置没有同步；服务端起来后再保存一次即可。");
      return;
    }
    setSaved(`已保存到本机。服务端答复：${sync.message}（对下一场面试生效）`);
  };

  const reset = () => {
    setDraft({
      wsUrl: "",
      token: "",
      presetCamera: false,
      presetPhysiology: false,
      presetScreen: false,
      llmBaseUrl: "",
      llmApiKey: "",
      llmModel: "",
    });
    setSaved(null);
  };

  return (
    <section className="settings" aria-labelledby="settings-title">
      <h1 id="settings-title">设置</h1>
      <p className="settings-intro">这里改的是本机偏好，改动不影响其他人，也不会上传。</p>

      <form onSubmit={submit}>
        <fieldset>
          <legend>连接</legend>
          <label className="settings-field">
            <span>服务地址</span>
            <input
              type="text"
              value={draft.wsUrl}
              onChange={(e) => set({ wsUrl: e.target.value })}
              placeholder="留空则自动（跟随页面域名或构建期配置）"
              spellCheck={false}
            />
          </label>
          {wsError ? (
            <p className="settings-error" role="alert">
              {wsError}
            </p>
          ) : null}
          <label className="settings-field">
            <span>令牌（可选）</span>
            <input
              type="password"
              value={draft.token}
              onChange={(e) => set({ token: e.target.value })}
              placeholder="开发期连接令牌；只保存在本机"
              autoComplete="off"
              spellCheck={false}
            />
          </label>
        </fieldset>

        <fieldset>
          <legend>授权预设（下次进授权页时的初始勾选）</legend>
          <label className="settings-check">
            <input
              type="checkbox"
              checked={draft.presetCamera}
              onChange={(e) => togglePreset("presetCamera", e.target.checked)}
            />
            <span>摄像头</span>
          </label>
          <label className="settings-check">
            <input
              type="checkbox"
              checked={draft.presetPhysiology}
              onChange={(e) => togglePreset("presetPhysiology", e.target.checked)}
            />
            <span>生理信号分析（会自动带起摄像头）</span>
          </label>
          <label className="settings-check">
            <input
              type="checkbox"
              checked={draft.presetScreen}
              onChange={(e) => togglePreset("presetScreen", e.target.checked)}
            />
            <span>共享屏幕</span>
          </label>
          <p className="settings-note">
            预设只是替你先把框勾好，到了授权页仍可以逐项改。参加面试那一项在这里管不了，必须本人当场勾。
          </p>
        </fieldset>

        <fieldset>
          <legend>AI 评分（可选）</legend>
          <label className="settings-field">
            <span>服务地址（OpenAI 兼容）</span>
            <input
              type="text"
              value={draft.llmBaseUrl}
              onChange={(e) => set({ llmBaseUrl: e.target.value })}
              placeholder="如 https://api.deepseek.com"
              spellCheck={false}
            />
          </label>
          <label className="settings-field">
            <span>API Key</span>
            <input
              type="password"
              value={draft.llmApiKey}
              onChange={(e) => set({ llmApiKey: e.target.value })}
              placeholder="保存时推送给服务端；只存本机，不进日志"
              autoComplete="off"
              spellCheck={false}
            />
          </label>
          <label className="settings-field">
            <span>模型名</span>
            <input
              type="text"
              value={draft.llmModel}
              onChange={(e) => set({ llmModel: e.target.value })}
              placeholder="如 deepseek-chat"
              spellCheck={false}
            />
          </label>
          <p className="settings-note">
            三项都填才启用 AI 评分，都留空则用规则评分。保存时会推送给正在运行的服务端，
            对下一场面试生效；AI 评分失败时该轮自动降级为规则评分，报告里会写明。
          </p>
          {llmError ? (
            <p className="settings-error" role="alert">
              {llmError}
            </p>
          ) : null}
        </fieldset>

        <div className="settings-actions">
          <button type="submit" className="primary" disabled={!canSave || syncing}>
            {syncing ? "保存中…" : "保存"}
          </button>
          <button type="button" onClick={reset}>
            还原默认
          </button>
          <button type="button" onClick={onBack}>
            返回
          </button>
        </div>
        {saved ? (
          <p className="settings-saved" role="status">
            {saved}
          </p>
        ) : null}
      </form>
    </section>
  );
}
