/**
 * 收尾页（mockup 舞台 4 的产品化）。
 *
 * 报告生成中。report 一到就由 App 切到报告页；这里不显示假的进度条。
 */

export interface ClosingProps {
  onOpenReport: () => void;
  onExit: () => void;
  nTurns?: number;
  nBufferTurns?: number;
}

export function Closing({ onOpenReport, onExit, nTurns, nBufferTurns }: ClosingProps) {
  return (
    <div className="wrap closing">
      <div className="closing-inner">
        <h1>回答都保存好了。</h1>
        <p>报告正在生成，好了会自动显示。可以先等着，不用做任何操作。</p>
        <div className="acts">
          <button type="button" className="btn btn-primary" onClick={onOpenReport}>
            看看出来了没
          </button>
          <button type="button" className="btn btn-quiet" onClick={onExit}>
            回首页
          </button>
        </div>
        {typeof nTurns === "number" && nTurns > 0 ? (
          <p className="meta">
            一共 <span className="num">{nTurns}</span> 轮
            {typeof nBufferTurns === "number" && nBufferTurns > 0 ? (
              <>
                ，其中 <span className="num">{nBufferTurns}</span> 轮是缓冲题，不记分
              </>
            ) : null}
          </p>
        ) : null}
      </div>
    </div>
  );
}
