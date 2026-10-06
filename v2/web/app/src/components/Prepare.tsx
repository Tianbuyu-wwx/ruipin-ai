/**
 * 准备页（mockup 舞台 2 的产品化）。
 *
 * 组成：事实行 + 四项授权（复用 consent/logic 的纯逻辑）+ 设备自检 + 岗位 + 吸底动作条。
 * 设备自检**真探测**：`getUserMedia` 拿一次流立即释放，探测结果只有
 * 「可用 / 不可用 / 未检测」三态，绝不显示假的"有信号"。
 */

import { useCallback, useEffect, useState } from "react";
import {
  applyConsentToggle,
  canStartInterview,
  type ConsentFlags,
  type ConsentKey,
  physiologyConstraint,
} from "../consent/logic";

export interface PrepareProps {
  initial: ConsentFlags;
  initialPosition: string;
  onStart: (flags: ConsentFlags, position: string) => void;
  onBack: () => void;
  onOpenSettings: () => void;
}

type CheckState = "untested" | "ok" | "unavailable";

export function Prepare({ initial, initialPosition, onStart, onBack, onOpenSettings }: PrepareProps) {
  // 参加面试是必选项且恒为开：用户主动进到这一页、点"开始面谈"，就是参加；
  // 不想参加走"暂不参加"。这一项不需要（也不应该）做成可拨的开关。
  const [flags, setFlags] = useState<ConsentFlags>({ ...initial, base: true });
  const [position, setPosition] = useState(initialPosition);
  const [mic, setMic] = useState<CheckState>("untested");
  const [camera, setCamera] = useState<CheckState>("untested");

  // 真探测：授权过的设备枚举一次。拒绝授权时显式标"不可用"，
  // 不冒充"有信号"（mockup 里的三绿是演示摆拍，产品不做）。
  useEffect(() => {
    let cancelled = false;
    const probe = async (video: boolean) => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video });
        stream.getTracks().forEach((t) => t.stop());
        return "ok" as const;
      } catch {
        return "unavailable" as const;
      }
    };
    void (async () => {
      const m = await probe(false);
      const c = await probe(true);
      if (!cancelled) {
        setMic(m);
        setCamera(c);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const toggle = useCallback((key: ConsentKey, value: boolean) => {
    setFlags((prev) => applyConsentToggle(prev, key, value));
  }, []);

  const constraint = physiologyConstraint(flags);
  const canStart = canStartInterview(flags);

  const items: Array<{ key: ConsentKey; title: React.ReactNode; desc: string; why: string; locked?: boolean }> = [
    {
      key: "base",
      title: (
        <>
          参加本次模拟面试 <span className="tag tag-accent">必需</span>
        </>
      ),
      desc: "没法跳过。面试流程和最后那份报告都建立在这一项上。",
      why: flags.base ? "已同意，进入准备流程。" : "还没有同意，勾上才能开始。",
      locked: true,
    },
    {
      key: "camera",
      title: <>开启摄像头</>,
      desc: "用来取画面，并从画面里读脉搏信号。原始视频不留存，只留下统计量。",
      why: flags.camera ? "已开启。生理维度参与评分，权重最多占 8%。" : "已关闭。视频与生理维度不计入，其余按权重重分配。",
    },
    {
      key: "physiology",
      title: (
        <>
          生理信号分析 <span className="tag">敏感信息 · 单独授权</span>
        </>
      ),
      desc: "从画面里读心率和紧张之后的恢复速度。这属于敏感个人信息，得单独同意一次，勾选摄像头不能代替这一项。",
      why: flags.physiology ? "已授权。会计算紧张之后恢复得多快。" : "未授权。这一维不参与算分。",
    },
    {
      key: "screen",
      title: <>共享屏幕</>,
      desc: "只有演示类题目会用到。不开也能把整场面试走完。",
      why: flags.screen ? "已开启。演示类题目可以共享画面。" : "已关闭。演示类题目改成口述。",
    },
  ];

  const tagFor = (state: CheckState) =>
    state === "ok" ? (
      <span className="tag tag-live">可用</span>
    ) : state === "unavailable" ? (
      <span className="tag tag-warn">不可用</span>
    ) : (
      <span className="tag">未检测</span>
    );

  return (
    <div className="wrap prepare">
      <header className="prepare-head">
        <h1>确认授权和设备，然后开始。</h1>
        <p className="lede">
          四项授权各自独立。除了参加面试，其余都能关。关了只是让对应维度不参与算分，不会让总分变低。
        </p>

        <div className="facts">
          <div className="fact">
            <div className="k">随时可中止</div>
            <div className="v">答过的会留着</div>
          </div>
          <div className="fact">
            <div className="k">作答方式</div>
            <div className="v">打字或开口说</div>
          </div>
          <div className="fact">
            <div className="k">评分口径</div>
            <div className="v">报告里写明来源</div>
          </div>
        </div>
      </header>

      <div className="prepare-body">
        <div>
          <div className="consent-list">
            {items.map((item) => {
              const disabled = item.locked || (item.key === "physiology" && !flags.camera);
              return (
                <div
                  className="consent"
                  key={item.key}
                  data-on={flags[item.key] ? "true" : "false"}
                  data-locked={item.locked ? "true" : "false"}
                >
                  <div className="txt">
                    <div className="t">{item.title}</div>
                    <p className="d">{item.desc}</p>
                    {item.key === "physiology" && !flags.camera && constraint ? (
                      <p className="why" role="note">
                        {constraint}
                      </p>
                    ) : (
                      <p className="why" data-role="consequence">
                        {item.why}
                      </p>
                    )}
                  </div>
                  <button
                    type="button"
                    className="switch"
                    role="switch"
                    aria-checked={flags[item.key]}
                    aria-label={typeof item.title === "string" ? item.title : item.key}
                    disabled={disabled}
                    onClick={() => toggle(item.key, !flags[item.key])}
                  >
                    <i></i>
                  </button>
                </div>
              );
            })}
          </div>
        </div>

        <aside className="prepare-side">
          <div className="card">
            <h3>设备自检</h3>
            <div style={{ marginTop: 10 }}>
              <div className="check-row">
                <span className="lbl">
                  <span className="dot" style={{ color: mic === "ok" ? "var(--live)" : "var(--warn)" }}></span>
                  麦克风
                </span>
                {tagFor(mic)}
              </div>
              <div className="check-row">
                <span className="lbl">
                  <span className="dot" style={{ color: camera === "ok" ? "var(--live)" : "var(--warn)" }}></span>
                  摄像头
                </span>
                {tagFor(camera)}
              </div>
            </div>
            <p className="hint" style={{ marginTop: 12, fontSize: 12.5, color: "var(--faint)" }}>
              设备不可用也能继续：题目会完整显示成文字，作答和评分都不受影响，对应维度不计入。
            </p>
          </div>

          <div className="card">
            <div className="field" style={{ marginTop: 0 }}>
              <label htmlFor="position">应聘岗位</label>
              <input
                id="position"
                placeholder="例如：后端工程师"
                value={position}
                onChange={(e) => setPosition(e.target.value)}
              />
              <p className="hint">填了会按岗位出题，不填就用通用题。也可以在设置页里存成默认值。</p>
            </div>
          </div>

          <div className="card">
            <h3>服务与密钥</h3>
            <div className="check-row">
              <span className="lbl">连接、令牌、AI 评分</span>
              <button type="button" className="btn btn-quiet btn-sm" onClick={onOpenSettings}>
                打开设置
              </button>
            </div>
          </div>
        </aside>
      </div>

      <div className="prepare-actions">
        <button
          type="button"
          className="btn btn-primary"
          disabled={!canStart}
          onClick={() => onStart(flags, position)}
        >
          开始面谈
        </button>
        <button type="button" className="btn btn-plain" onClick={onBack}>
          暂不参加
        </button>
        <span style={{ fontSize: 13, color: "var(--faint)" }}>点了开始才会请求麦克风</span>
      </div>
    </div>
  );
}
