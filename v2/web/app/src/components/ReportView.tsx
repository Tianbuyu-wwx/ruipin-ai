/**
 * 报告页（mockup 舞台 5 的产品化）。
 *
 * 红线不变：
 * - `report.available=false` ⇒ **不出分**，verdict 只有解释文字，页面不出现任何分数数字。
 * - 生理维度不可用 ⇒ "未评"条形 + 原因文字（"测不准 = 不计入"，不扣分）。
 * - 反事实（去掉某一维的总分）用 note 呈现，数字来自服务端，不本地造。
 */

import { counterfactualText, physioScoreText, reportScoreText } from "../report/parse";
import { DIMENSIONS, dimLabel, PHYSIO_DIM } from "../report/dims";
import type { ReportView as ReportViewData } from "../store/types";

export interface ReportViewProps {
  report: ReportViewData | null;
}

export function ReportView({ report }: ReportViewProps) {
  if (!report) {
    return (
      <div className="wrap report">
        <div className="report-body">
          <p className="rep-empty" role="status">
            报告尚未生成。
          </p>
        </div>
      </div>
    );
  }

  const physioEnabled = report.physio !== null;
  const dimKeys = [...DIMENSIONS, PHYSIO_DIM].filter(
    (k) => k in report.dims || (k === PHYSIO_DIM && physioEnabled),
  );
  const counterfactualKeys = Object.keys(report.counterfactuals);
  const scoreText = reportScoreText(report);

  return (
    <div className="wrap report">
      <nav className="toc">
        <div className="t">目录</div>
        <a href="#rep-head">总览</a>
        <a href="#rep-dims">各维度</a>
        <a href="#rep-physio">生理维度</a>
        {counterfactualKeys.length > 0 ? <a href="#rep-counter">去掉某一维</a> : null}
        <div className="meta">
          评分版本 {report.rubricVersion}
          <br />
          计分 {report.nScoredTurns} 轮 / 缓冲 {report.nBufferTurns} 轮
        </div>
      </nav>

      <div className="report-body">
        <header className="rep-head" id="rep-head">
          <h1>这份报告能信几分</h1>
          <p className="sub">
            {report.available
              ? "结论来自下面这些计分轮次。每一项分数怎么来的、哪几项没算，都在各节里写明。"
              : "这次没有给出总分。原因写在下面。"}
          </p>
        </header>

        <div className="verdict">
          <div>
            <div className="grade" data-unavailable={report.available ? "false" : "true"}>
              {report.available ? (
                <>
                  可以当参考 <span>不适合排名</span>
                </>
              ) : (
                "这次不出分"
              )}
            </div>
            <p className="expl">{scoreText}</p>
            {!report.available && report.reason ? (
              <p className="expl">原因：{report.reason}</p>
            ) : null}
          </div>
          {report.available ? (
            <div className="total">
              <span className="n">{report.score?.toFixed(1) ?? "—"}</span>
              <div className="cap">计分轮次总分{report.level ? ` · 等级 ${report.level}` : ""}</div>
            </div>
          ) : null}
        </div>

        <section className="block" id="rep-dims">
          <h2>各维度</h2>
          <p className="intro">条越长分越高。灰色的项这轮没算，不加分也不扣分。</p>
          {dimKeys.map((k) => {
            const scored = k in report.dims;
            const value = scored ? report.dims[k] : null;
            return (
              <div className="dim" key={k} data-muted={scored ? "false" : "true"}>
                <span className="nm">{dimLabel(k)}</span>
                <span className="bar">
                  <i style={{ width: `${Math.max(0, Math.min(100, value ?? 0))}%` }}></i>
                </span>
                <span className="sc">{value === null ? "未评" : value.toFixed(1)}</span>
              </div>
            );
          })}
        </section>

        <section className="block" id="rep-physio">
          <h2>生理维度</h2>
          <div className={report.physio?.available ? "note" : "note note-warn"}>
            <h3>{report.physio?.available ? "生理维度参与了评分" : "这一轮没有可用的生理信号"}</h3>
            {physioScoreText(report.physio, physioEnabled)}
            {report.physio && report.physio.available ? (
              <>
                {" "}有效事件 {report.physio.nValid} 个，权重上限 8%，实得{" "}
                {(report.physio.weightApplied * 100).toFixed(1)}%。
              </>
            ) : null}
          </div>
        </section>

        {counterfactualKeys.length > 0 ? (
          <section className="block" id="rep-counter">
            <h2>去掉某一维，总分变成多少</h2>
            <p className="intro">把某一维整个去掉再算一遍，看总分差多少。差得越多，这一维对这次结果影响越大。</p>
            {counterfactualKeys.map((k) => {
              const total = report.counterfactuals[k];
              return (
                <div className="note" key={k}>
                  <h3>
                    去掉{dimLabel(k)}，总分变成{" "}
                    <span className="fig">{total === null ? "（算不出来）" : total.toFixed(1)}</span>
                  </h3>
                  {counterfactualText(total)}
                </div>
              );
            })}
          </section>
        ) : null}

        {report.notes.length > 0 ? (
          <section className="block" id="rep-limit">
            <h2>口径说明</h2>
            {report.notes.map((n, i) => (
              <div className="note" key={`n-${i}`}>
                {n}
              </div>
            ))}
          </section>
        ) : null}

        <div className="rep-foot">
          评分版本 {report.rubricVersion}；计分 {report.nScoredTurns} 轮，缓冲 {report.nBufferTurns} 轮
          <br />
          分数只用来做参考。哪几轮被规则算法算的、哪些维度没算，上面各节都有标注。
        </div>
      </div>
    </div>
  );
}
