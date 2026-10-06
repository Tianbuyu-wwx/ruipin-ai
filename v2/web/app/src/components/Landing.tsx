/**
 * 产品首页（mockup 舞台 1 的产品化）。
 *
 * 纯展示层：CTA 进入准备页。文案里的数字与事实（题量、授权规则、降级行为）
 * 必须与服务端真实行为一致—— mockup 里"一共五道题"是演示口径，产品里
 * 题量由服务端装配决定，这里不写死数字。
 */

export interface LandingProps {
  onStart: () => void;
  onOpenSettings: () => void;
}

export function Landing({ onStart, onOpenSettings }: LandingProps) {
  return (
    <div className="wrap">
      <nav className="nav">
        <div className="wordmark">
          <span className="mark">RP</span>锐聘 AI
        </div>
        <button type="button" className="btn btn-plain" onClick={onOpenSettings}>
          设置
        </button>
      </nav>

      <header className="hero">
        <div>
          <h1>
            把面试练一遍，
            <br />
            看看哪里讲不清楚。
          </h1>
          <p className="lede">
            面试官逐题提问，答得笼统就接着问刚才那句里的细节。答完出一份报告，
            分数怎么算、哪几项不该信，都写在里面。
          </p>
          <div className="cta-row">
            <button type="button" className="btn btn-primary" onClick={onStart}>
              开始面试
            </button>
            <button type="button" className="btn btn-plain" onClick={onOpenSettings}>
              配置服务与密钥
            </button>
          </div>
          <p className="reassure">
            不用注册。摄像头和生理信号可以分开拒绝，拒绝不会让分数变低。
          </p>
        </div>

        <figure className="preview">
          <div className="preview-bar">
            <i></i>
            <i></i>
            <i></i>
            <span style={{ marginLeft: 6 }}>面试间</span>
          </div>
          <div className="preview-body">
            <span className="tag tag-live">
              <span className="dot"></span>面试官在说
            </span>
            <p className="preview-q">
              请讲一个你主导过、但结果不如预期的项目。你当时是怎么判断问题出在哪里的？
            </p>
            <div className="preview-wave" aria-hidden="true">
              {Array.from({ length: 26 }, (_, i) => (
                <b key={i} style={{ "--peak": 2 + ((i * 7) % 5) } as React.CSSProperties} />
              ))}
            </div>
            <div className="preview-foot">
              <span>追问取决于你刚才说了什么</span>
            </div>
          </div>
        </figure>
      </header>

      <div className="bento" id="bento">
        <div className="cell cell-a">
          <div>
            <h3>心率测不准就不算这一项</h3>
            <p>
              画面里的脉搏信号信噪比不够的时候，这一轮不计入生理维度，剩下几项按比例重新分权重。
            </p>
          </div>
          <div className="cell-figure">
            <div className="dim" style={{ gridTemplateColumns: "92px minmax(0,1fr) 56px", padding: "7px 0" }}>
              <span className="nm">技术能力</span>
              <span className="bar">
                <i style={{ width: "84%" }}></i>
              </span>
              <span className="sc num">84.0</span>
            </div>
            <div className="dim" data-muted="true" style={{ gridTemplateColumns: "92px minmax(0,1fr) 56px", padding: "7px 0" }}>
              <span className="nm">生理维度</span>
              <span className="bar">
                <i style={{ width: "0%" }}></i>
              </span>
              <span className="sc">未评</span>
            </div>
          </div>
        </div>

        <div className="cell cell-b">
          <div>
            <h3>哪一环没跑起来，角落会挂个标</h3>
            <p>语音识别、形象渲染、评分，有一个没跑起来，角落就多一个徽标。</p>
          </div>
          <div>
            <span className="tag tag-warn">这轮分数是规则算法算的</span>
          </div>
        </div>

        <div className="cell cell-c">
          <div>
            <h3>追问看你刚说了什么</h3>
            <p>连着两轮答不上来，中间插一道不计分的缓冲题。</p>
          </div>
          <div>
            <span className="tag">缓冲题不计分</span>
          </div>
        </div>

        <div className="cell cell-d">
          <div>
            <h3>报告里会写清楚哪些分数不能信</h3>
            <p>六个维度的分数之外，还有两行：某一维去掉之后总分变成多少，这次有几轮没进评分。</p>
          </div>
          <div>
            <span className="tag tag-accent">去掉一项，总分 71.2</span>
          </div>
        </div>
      </div>

      <section className="lede-block">
        <h2>没接 AI 评分的时候，分数只用来验链路。</h2>
        <p>
          评分来源是规则算法还是模型，报告里会写明。规则口径的分数只用来验证整条链路通不通，
          拿去判断一个人够不够格是不行的。
        </p>
      </section>

      <footer className="foot">
        <span>锐聘 AI · 模拟面试系统</span>
        <span>生理数据单独授权，原始视频不留存</span>
      </footer>
    </div>
  );
}
