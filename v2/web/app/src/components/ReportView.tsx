/**
 * 报告页（方案 §3.4 红线 + §12.6）。
 *
 * 关键约束：
 * - `report.available=false` ⇒ **不出分**，只给原因（`reportScoreText` 保证不出现数字）。
 * - 生理维度不可用 ⇒ 显示"未计入（原因：…）"，**绝不显示任何数字**
 *   （对应后端"测不准 = 不计入"）；并展示 `counterfactuals` 的"关闭生理模块后总分是多少"。
 * - 权重重分配可视化：让用户看到"某维度没算，权重还给了其他维度，总分未被压低"。
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
      <section className="report" aria-labelledby="report-heading">
        <h1 id="report-heading">面试报告</h1>
        <p role="status">报告尚未生成。</p>
      </section>
    );
  }

  const physioEnabled = report.physio !== null;
  const dimKeys = [...DIMENSIONS, PHYSIO_DIM].filter(
    (k) => k in report.dims || (k === PHYSIO_DIM && physioEnabled),
  );
  const counterfactualKeys = Object.keys(report.counterfactuals);

  return (
    <section className="report" aria-labelledby="report-heading">
      <h1 id="report-heading">面试报告</h1>

      <div className={`report-score ${report.available ? "ok" : "unavailable"}`}>
        <div className="total" role="status" aria-live="polite">
          {reportScoreText(report)}
        </div>
        {report.available && report.level ? <div className="level">等级：{report.level}</div> : null}
      </div>

      <h2>维度得分</h2>
      <table className="dims">
        <caption className="visually-hidden">各评分维度得分与生效权重</caption>
        <thead>
          <tr>
            <th scope="col">维度</th>
            <th scope="col">得分</th>
            <th scope="col">生效权重</th>
          </tr>
        </thead>
        <tbody>
          {dimKeys.map((k) => (
            <tr key={k}>
              <th scope="row">{dimLabel(k)}</th>
              <td>{k in report.dims ? report.dims[k].toFixed(1) : "未计入"}</td>
              <td>{((report.effectiveWeights[k] ?? 0) * 100).toFixed(1)}%</td>
            </tr>
          ))}
        </tbody>
      </table>

      <h2>生理维度</h2>
      <p className="physio-status" data-available={report.physio?.available ? "true" : "false"}>
        {physioScoreText(report.physio, physioEnabled)}
      </p>
      {report.physio && report.physio.available ? (
        <p className="muted">
          有效事件 {report.physio.nValid} 个；权重上限 8%，实得权重{" "}
          {(report.physio.weightApplied * 100).toFixed(1)}%。
        </p>
      ) : null}

      {counterfactualKeys.length > 0 ? (
        <>
          <h2>反事实（关闭某维度后的总分）</h2>
          <ul className="counterfactuals">
            {counterfactualKeys.map((k) => (
              <li key={k}>
                若剔除「{dimLabel(k)}」：{counterfactualText(report.counterfactuals[k])}
              </li>
            ))}
          </ul>
        </>
      ) : null}

      {report.notes.length > 0 ? (
        <>
          <h2>评分归因</h2>
          <ul className="notes">
            {report.notes.map((n, i) => (
              <li key={`n-${i}`}>{n}</li>
            ))}
          </ul>
        </>
      ) : null}

      <p className="muted">
        口径版本：{report.rubricVersion}；计分轮次 {report.nScoredTurns}，缓冲轮次 {report.nBufferTurns}。
      </p>
    </section>
  );
}
