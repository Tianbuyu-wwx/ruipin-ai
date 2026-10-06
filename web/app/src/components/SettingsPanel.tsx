/**
 * 设置页（mockup 设计语言）。
 *
 * 四组设置：
 *  1. 连接：WS 地址与令牌（留空即跟随默认；令牌只存本机）；
 *  2. 面试选项：应聘岗位默认值、下次进授权页的授权预设；
 *  3. AI 评分：OpenAI 兼容三项，保存时推送给运行中的服务端；
 *  4. 说明：数据存在哪、什么时候生效。
 *
 * 纪律：显式保存才落盘；校验不过禁保存；服务端不可达时明说"未同步"。
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
  position: string;
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
      position: draft.position.trim(),
    };
    savePrefs(storage, next);
    setSyncing(true);
    const sync = await pushLlmConfig(next);
    setSyncing(false);
    if (sync === null) {
      setSaved("已保存到本机。服务端现在连不上，AI 评分配置没有同步；服务端起来后再保存一次即可。");
      return;
    }
    setSaved(`已保存到本机。服务端答复：${sync.message}`);
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
      position: "",
    });
    setSaved(null);
  };

  return (
    <div className="wrap">
      <header className="settings-head">
        <h1>设置</h1>
        <p className="lede">
          这里改的是本机偏好。AI 评分配置保存时会推给正在运行的服务端，其余改动回准备页即生效。
        </p>
      </header>

      <form onSubmit={submit}>
        <div className="settings-body">
          <div className="card">
            <h3>连接</h3>
            <div className="field">
              <label htmlFor="set-wsurl">服务地址</label>
              <input
                id="set-wsurl"
                type="text"
                value={draft.wsUrl}
                onChange={(e) => set({ wsUrl: e.target.value })}
                placeholder="留空则自动（跟随页面域名或构建期配置）"
                spellCheck={false}
              />
            </div>
            {wsError ? (
              <p className="settings-error" role="alert">
                {wsError}
              </p>
            ) : null}
            <div className="field">
              <label htmlFor="set-token">令牌（可选）</label>
              <input
                id="set-token"
                type="password"
                value={draft.token}
                onChange={(e) => set({ token: e.target.value })}
                placeholder="开发期连接令牌；只保存在本机"
                autoComplete="off"
                spellCheck={false}
              />
            </div>
          </div>

          <div className="card">
            <h3>面试选项</h3>
            <div className="field" style={{ marginTop: 12 }}>
              <label htmlFor="set-position">应聘岗位（默认值）</label>
              <input
                id="set-position"
                type="text"
                value={draft.position}
                onChange={(e) => set({ position: e.target.value })}
                placeholder="例如：后端工程师"
                spellCheck={false}
              />
              <p className="hint">开始面试前还可以在准备页临时改。</p>
            </div>
            <div style={{ marginTop: 8 }}>
              <div className="check-row">
                <span className="lbl">摄像头预设</span>
                <button
                  type="button"
                  className="switch"
                  role="switch"
                  aria-checked={draft.presetCamera}
                  aria-label="摄像头预设"
                  onClick={() => togglePreset("presetCamera", !draft.presetCamera)}
                >
                  <i></i>
                </button>
              </div>
              <div className="check-row">
                <span className="lbl">生理信号预设（会带起摄像头）</span>
                <button
                  type="button"
                  className="switch"
                  role="switch"
                  aria-checked={draft.presetPhysiology}
                  aria-label="生理信号预设"
                  onClick={() => togglePreset("presetPhysiology", !draft.presetPhysiology)}
                >
                  <i></i>
                </button>
              </div>
              <div className="check-row">
                <span className="lbl">共享屏幕预设</span>
                <button
                  type="button"
                  className="switch"
                  role="switch"
                  aria-checked={draft.presetScreen}
                  aria-label="共享屏幕预设"
                  onClick={() => togglePreset("presetScreen", !draft.presetScreen)}
                >
                  <i></i>
                </button>
              </div>
            </div>
            <p className="settings-note">
              预设只是替你先把授权项勾好，到了准备页仍可以逐项改。参加面试那一项管不了，必须本人当场确认。
            </p>
          </div>

          <div className="card">
            <h3>AI 评分（可选）</h3>
            <div className="field" style={{ marginTop: 12 }}>
              <label htmlFor="set-llmurl">服务地址（OpenAI 兼容）</label>
              <input
                id="set-llmurl"
                type="text"
                value={draft.llmBaseUrl}
                onChange={(e) => set({ llmBaseUrl: e.target.value })}
                placeholder="如 https://api.deepseek.com"
                spellCheck={false}
              />
            </div>
            <div className="field">
              <label htmlFor="set-llmkey">API Key</label>
              <input
                id="set-llmkey"
                type="password"
                value={draft.llmApiKey}
                onChange={(e) => set({ llmApiKey: e.target.value })}
                placeholder="保存时推送给服务端；只存本机，不进日志"
                autoComplete="off"
                spellCheck={false}
              />
            </div>
            <div className="field">
              <label htmlFor="set-llmmodel">模型名</label>
              <input
                id="set-llmmodel"
                type="text"
                value={draft.llmModel}
                onChange={(e) => set({ llmModel: e.target.value })}
                placeholder="如 deepseek-chat"
                spellCheck={false}
              />
            </div>
            <p className="settings-note">
              三项都填才启用 AI 评分，都留空则用规则评分。AI 评分失败时该轮自动降级为规则评分，报告里会写明。
            </p>
            {llmError ? (
              <p className="settings-error" role="alert">
                {llmError}
              </p>
            ) : null}
          </div>
        </div>

        <div className="settings-actions">
          <button type="submit" className="btn btn-primary" disabled={!canSave || syncing}>
            {syncing ? "保存中…" : "保存"}
          </button>
          <button type="button" className="btn btn-quiet" onClick={reset}>
            还原默认
          </button>
          <button type="button" className="btn btn-plain" onClick={onBack}>
            返回
          </button>
          {saved ? (
            <p className="settings-saved" role="status">
              {saved}
            </p>
          ) : null}
        </div>
      </form>
    </div>
  );
}
