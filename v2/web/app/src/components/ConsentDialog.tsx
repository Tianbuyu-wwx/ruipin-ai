/**
 * 授权门（方案 §2.2 `<ConsentGate />` + PIPL 第 28 条）。
 *
 * 三/四项**分开勾选**：
 *   - 参加面试（base）：唯一必需项；
 *   - 摄像头（camera）：可选增强；
 *   - 生理信号（physiology）：**敏感个人信息，单独同意**，有独立说明，且明确
 *     "拒绝也可继续面试"；未开摄像头时该项禁用并说明原因（rPPG 依赖画面）；
 *   - 屏幕共享（screen）：可选增强。
 *
 * 无障碍：`fieldset/legend` 分组、每项 `aria-describedby` 关联说明、可全键盘操作。
 */

import { type ChangeEvent, useId, useState } from "react";
import {
  applyConsentToggle,
  canStartInterview,
  type ConsentFlags,
  type ConsentKey,
  DEFAULT_CONSENT,
  physiologyConstraint,
} from "../consent/logic";

export interface ConsentDialogProps {
  initial?: ConsentFlags;
  onGrant: (flags: ConsentFlags) => void;
  onDecline?: () => void;
}

interface ItemDef {
  key: ConsentKey;
  label: string;
  desc: string;
}

const ITEMS: ItemDef[] = [
  { key: "base", label: "参加本次模拟面试", desc: "必需。不勾选则无法开始面试。" },
  {
    key: "camera",
    label: "开启摄像头（行程/表情分析，可选）",
    desc: "可选增强。拒绝即可，面试与评分照常进行（相应维度按权重重分配，不会压低总分）。",
  },
  {
    key: "physiology",
    label: "生理信号分析（心率/应激恢复，可选）",
    desc: "敏感个人信息，需单独同意。我们仅从画面提取脉搏相关统计量，不保存原始视频。拒绝也可继续面试。",
  },
  {
    key: "screen",
    label: "共享屏幕（可选）",
    desc: "可选增强，用于演示类题目。拒绝不影响其余流程。",
  },
];

export function ConsentDialog({ initial = DEFAULT_CONSENT, onGrant, onDecline }: ConsentDialogProps) {
  const [flags, setFlags] = useState<ConsentFlags>(initial);
  const baseId = useId();
  const constraint = physiologyConstraint(flags);
  const canStart = canStartInterview(flags);

  const toggle = (key: ConsentKey) => (e: ChangeEvent<HTMLInputElement>) => {
    setFlags((prev) => applyConsentToggle(prev, key, e.target.checked));
  };

  return (
    <section className="consent-gate" aria-labelledby={`${baseId}-title`}>
      <h1 id={`${baseId}-title`}>开始前请阅读并选择授权项</h1>
      <p className="consent-intro">
        各项授权相互独立。除"参加面试"外均可拒绝，拒绝不会导致面试无法进行。
      </p>

      <fieldset>
        <legend>授权项（分项勾选）</legend>
        {ITEMS.map((item) => {
          const descId = `${baseId}-${item.key}-desc`;
          const isPhysio = item.key === "physiology";
          const disabled = isPhysio && !flags.camera;
          return (
            <div className="consent-item" key={item.key}>
              <label>
                <input
                  type="checkbox"
                  checked={flags[item.key]}
                  disabled={disabled}
                  onChange={toggle(item.key)}
                  aria-describedby={descId}
                />
                <span>{item.label}</span>
              </label>
              <p id={descId} className="consent-desc">
                {item.desc}
              </p>
              {isPhysio && disabled && constraint ? (
                <p className="consent-note" role="note">
                  {constraint}
                </p>
              ) : null}
            </div>
          );
        })}
      </fieldset>

      <div className="consent-actions">
        <button type="button" className="primary" disabled={!canStart} onClick={() => onGrant(flags)}>
          同意并开始面试
        </button>
        <button type="button" onClick={() => onDecline?.()}>
          暂不参加
        </button>
      </div>
      {!canStart ? (
        <p className="consent-note" role="status">
          请勾选"参加本次模拟面试"后再开始。
        </p>
      ) : null}
    </section>
  );
}
