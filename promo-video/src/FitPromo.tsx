import React from 'react';
import {
  AbsoluteFill,
  Easing,
  Img,
  Sequence,
  interpolate,
  spring,
  staticFile,
  useCurrentFrame,
  useVideoConfig,
} from 'remotion';

const C = {
  bg: '#050505',
  panel: '#101112',
  panel2: '#17181A',
  ink: '#F4F1E8',
  muted: '#9A9CA1',
  grid: '#2C2E32',
  orange: '#FF5A1F',
  blue: '#5593FF',
};

const font = "'Noto Sans CJK SC', 'Noto Sans SC', 'Noto Sans', sans-serif";

const tiers = [
  '7G', '7.5G', '8G', '8.5G', '9G', '9.5G', '10G',
  '10.5G', '11G', '11.5G', '12G', '12.5G', '13G', '13.5G',
];

const cutCard: React.CSSProperties = {
  background: C.panel,
  border: `1px solid ${C.grid}`,
  clipPath: 'polygon(18px 0, 100% 0, 100% calc(100% - 18px), calc(100% - 18px) 100%, 0 100%, 0 18px)',
};

const ease = Easing.bezier(0.16, 1, 0.3, 1);

const sceneOpacity = (frame: number, duration: number) =>
  interpolate(frame, [0, 18, duration - 18, duration], [0, 1, 1, 0], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });

const rise = (frame: number, delay = 0, distance = 42) => {
  const progress = interpolate(frame, [delay, delay + 28], [0, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
    easing: ease,
  });
  return {opacity: progress, transform: `translateY(${(1 - progress) * distance}px)`};
};

const Label: React.FC<{children: React.ReactNode}> = ({children}) => (
  <div style={{fontSize: 22, letterSpacing: 5, color: C.orange, fontWeight: 700}}>{children}</div>
);

const CornerBrand: React.FC = () => (
  <div style={{position: 'absolute', right: 76, top: 58, fontSize: 22, letterSpacing: 3, color: C.orange, fontWeight: 800}}>
    FIT-GGUF
  </div>
);

const Background: React.FC = () => {
  const frame = useCurrentFrame();
  return (
    <AbsoluteFill style={{backgroundColor: C.bg, overflow: 'hidden', fontFamily: font, color: C.ink}}>
      <div
        style={{
          position: 'absolute', inset: -180,
          backgroundImage: `linear-gradient(${C.grid}55 1px, transparent 1px), linear-gradient(90deg, ${C.grid}55 1px, transparent 1px)`,
          backgroundSize: '96px 96px',
          opacity: 0.28,
          transform: `perspective(900px) rotateX(67deg) translateY(${230 + frame * 0.12}px) scale(1.3)`,
          transformOrigin: 'center center',
        }}
      />
      <div style={{position: 'absolute', inset: 0, background: 'radial-gradient(circle at 50% 44%, transparent 0%, #05050544 48%, #050505 84%)'}} />
    </AbsoluteFill>
  );
};

const Intro: React.FC<{duration: number}> = ({duration}) => {
  const frame = useCurrentFrame();
  const reveal = interpolate(frame, [12, 70], [0, 100], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp', easing: ease});
  return (
    <AbsoluteFill style={{opacity: sceneOpacity(frame, duration), justifyContent: 'center', alignItems: 'center'}}>
      <div style={{width: 1220, clipPath: `inset(0 ${100 - reveal}% 0 0)`}}>
        <Img src={staticFile('fit-logo.svg')} style={{width: '100%'}} />
      </div>
      <div style={{...rise(frame, 70), marginTop: 44, fontSize: 34, letterSpacing: 8, color: C.muted}}>FIT-TO-SIZE INTELLIGENT TENSOR QUANTIZATION</div>
    </AbsoluteFill>
  );
};

const PresetProblem: React.FC<{duration: number}> = ({duration}) => {
  const frame = useCurrentFrame();
  const cards = [
    {name: 'IQ2_S', size: '8.72 GiB', state: '没有用满'},
    {name: 'IQ2_M', size: '9.32 GiB', state: '超出预算'},
    {name: 'Q2_K', size: '9.98 GiB', state: '差得更多'},
  ];
  return (
    <AbsoluteFill style={{opacity: sceneOpacity(frame, duration), padding: '120px 140px'}}>
      <CornerBrand />
      <div style={rise(frame, 0)}><Label>01 / 预设之间的空白</Label></div>
      <div style={{...rise(frame, 8), marginTop: 22, fontSize: 76, lineHeight: 1.12, fontWeight: 750}}>你的硬件，<br/><span style={{color: C.orange}}>不是预设档位。</span></div>
      <div style={{display: 'flex', gap: 28, marginTop: 78}}>
        {cards.map((card, i) => {
          const p = spring({frame: frame - 34 - i * 8, fps: 30, config: {damping: 18, stiffness: 110}});
          return (
            <div key={card.name} style={{...cutCard, width: 350, padding: '34px 38px', opacity: p, transform: `translateY(${(1 - p) * 55}px)`}}>
              <div style={{fontSize: 26, color: C.muted}}>{card.name}</div>
              <div style={{fontSize: 54, marginTop: 18, fontWeight: 700}}>{card.size}</div>
              <div style={{marginTop: 28, height: 8, background: C.grid}}><div style={{width: `${58 + i * 20}%`, height: '100%', background: i === 0 ? C.muted : C.blue}} /></div>
              <div style={{fontSize: 22, color: i === 0 ? C.muted : C.blue, marginTop: 18}}>{card.state}</div>
            </div>
          );
        })}
        <div style={{...cutCard, flex: 1, padding: 38, borderColor: C.orange}}>
          <div style={{fontSize: 24, color: C.orange}}>你的目标</div>
          <div style={{fontSize: 82, marginTop: 24, fontWeight: 800}}>9.00 GiB</div>
          <div style={{fontSize: 25, color: C.muted, marginTop: 24}}>预算固定。预设却只给你几个选项。</div>
        </div>
      </div>
    </AbsoluteFill>
  );
};

const SizeSlider: React.FC<{duration: number}> = ({duration}) => {
  const frame = useCurrentFrame();
  const steps = [
    {value: '0.5 GiB', label: '首发下载间隔', note: '14 个预制产物', color: C.muted},
    {value: '100 MiB', label: '本地自定义', note: '更贴近硬件预算', color: C.orange},
    {value: '10 MiB', label: '本地自定义', note: '进一步细化目标', color: C.orange},
  ];
  return (
    <AbsoluteFill style={{opacity: sceneOpacity(frame, duration), padding: '120px 140px'}}>
      <CornerBrand />
      <div style={rise(frame)}><Label>02 / 自定义目标精度</Label></div>
      <div style={{...rise(frame, 6), marginTop: 22, fontSize: 78, fontWeight: 760}}><span style={{color: C.muted}}>0.5 GiB</span>，只是首发采样。</div>
      <div style={{...rise(frame, 14), marginTop: 10, fontSize: 70, fontWeight: 760}}><span style={{color: C.orange}}>100 MiB。甚至 10 MiB。</span> 由你定。</div>
      <div style={{display: 'flex', gap: 28, marginTop: 76}}>
        {steps.map((step, i) => {
          const p = spring({frame: frame - 38 - i * 22, fps: 30, config: {damping: 18, stiffness: 105}});
          return (
            <div key={step.value} style={{...cutCard, flex: 1, padding: '34px 38px', borderColor: i === 0 ? C.grid : C.orange, opacity: p, transform: `translateY(${(1 - p) * 48}px)`}}>
              <div style={{fontSize: 21, letterSpacing: 3, color: step.color}}>{step.label}</div>
              <div style={{fontSize: 64, fontWeight: 820, marginTop: 20, color: i === 0 ? C.muted : C.ink}}>{step.value}</div>
              <div style={{height: 5, background: C.grid, marginTop: 24}}><div style={{height: '100%', width: `${100 / (i + 1)}%`, background: step.color}} /></div>
              <div style={{fontSize: 23, color: C.muted, marginTop: 20}}>{step.note}</div>
            </div>
          );
        })}
      </div>
      <div style={{...rise(frame, 118), display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: 42}}>
        <div style={{fontSize: 29}}>CLI 接受任意 <span style={{fontFamily: 'monospace', color: C.orange}}>--target-bytes</span></div>
        <div style={{fontSize: 22, color: C.muted}}>实际配方受离散 Tensor/qtype 步长约束</div>
      </div>
    </AbsoluteFill>
  );
};

const Pipeline: React.FC<{duration: number}> = ({duration}) => {
  const frame = useCurrentFrame();
  const stages = [
    {n: '01', title: '分析', en: 'ANALYZE', body: '读取源模型、imatrix 与有效预设配方'},
    {n: '02', title: '规划', en: 'PLAN', body: '在字节预算内分配安全的精度升级'},
    {n: '03', title: '量化', en: 'QUANTIZE', body: '执行配方，验证文件尺寸并记录 SHA-256'},
  ];
  return (
    <AbsoluteFill style={{opacity: sceneOpacity(frame, duration), padding: '105px 130px'}}>
      <CornerBrand />
      <div style={rise(frame)}><Label>03 / 可复现流程</Label></div>
      <div style={{...rise(frame, 5), marginTop: 18, fontSize: 72, fontWeight: 760}}>分析。规划。量化。</div>
      <div style={{display: 'flex', gap: 28, marginTop: 68}}>
        {stages.map((stage, i) => {
          const p = spring({frame: frame - 25 - i * 14, fps: 30, config: {damping: 20, stiffness: 100}});
          return (
            <React.Fragment key={stage.n}>
              <div style={{...cutCard, width: 470, minHeight: 340, padding: 40, opacity: p, transform: `translateY(${(1 - p) * 60}px)`}}>
                <div style={{display: 'flex', justifyContent: 'space-between'}}><span style={{color: C.orange, fontSize: 24}}>{stage.n}</span><span style={{color: C.muted, letterSpacing: 3}}>{stage.en}</span></div>
                <div style={{fontSize: 64, fontWeight: 800, marginTop: 42}}>{stage.title}</div>
                <div style={{fontSize: 25, color: C.muted, lineHeight: 1.55, marginTop: 30}}>{stage.body}</div>
                <div style={{display: 'grid', gridTemplateColumns: 'repeat(8, 1fr)', gap: 7, marginTop: 35}}>
                  {Array.from({length: 24}).map((_, j) => <div key={j} style={{height: 10, background: j < Math.max(0, (frame - 50 - i * 15) / 3) ? (j % 5 === 0 ? C.orange : C.ink) : C.grid}} />)}
                </div>
              </div>
              {i < 2 ? <div style={{alignSelf: 'center', fontSize: 46, color: C.orange}}>→</div> : null}
            </React.Fragment>
          );
        })}
      </div>
    </AbsoluteFill>
  );
};

const TierGrid: React.FC<{duration: number}> = ({duration}) => {
  const frame = useCurrentFrame();
  return (
    <AbsoluteFill style={{opacity: sceneOpacity(frame, duration), padding: '105px 130px'}}>
      <CornerBrand />
      <div style={rise(frame)}><Label>04 / 下载样例 ≠ 工具上限</Label></div>
      <div style={{...rise(frame, 5), marginTop: 18, fontSize: 70, fontWeight: 760}}><span style={{color: C.orange}}>14 个首发档位</span>，不是 FIT 的上限。</div>
      <div style={{display: 'grid', gridTemplateColumns: 'repeat(7, 1fr)', gap: 18, marginTop: 56}}>
        {tiers.map((tier, i) => {
          const p = spring({frame: frame - 24 - i * 4, fps: 30, config: {damping: 17, stiffness: 130}});
          return (
            <div key={tier} style={{...cutCard, padding: '22px 22px', opacity: p, transform: `scale(${0.88 + p * 0.12})`, borderColor: tier === '12G' ? C.orange : C.grid}}>
              <div style={{fontSize: 20, color: tier === '12G' ? C.orange : C.muted}}>FIT</div>
              <div style={{fontSize: 43, fontWeight: 800, marginTop: 8}}>{tier}</div>
            </div>
          );
        })}
      </div>
      <div style={{...rise(frame, 95), ...cutCard, borderColor: C.orange, padding: '24px 30px', marginTop: 32, display: 'flex', justifyContent: 'space-between', alignItems: 'center'}}>
        <div style={{fontSize: 27}}>0.5 GiB 网格，只是因为无法制作并上传数百个 GGUF。</div>
        <div style={{fontSize: 25, color: C.orange, fontWeight: 750}}>本地运行 FIT → 输入你的字节预算</div>
      </div>
    </AbsoluteFill>
  );
};

const Accuracy: React.FC<{duration: number}> = ({duration}) => {
  const frame = useCurrentFrame();
  const p = interpolate(frame, [24, 118], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp', easing: ease});
  return (
    <AbsoluteFill style={{opacity: sceneOpacity(frame, duration), padding: '110px 140px'}}>
      <CornerBrand />
      <div style={rise(frame)}><Label>05 / 尺寸验证</Label></div>
      <div style={{display: 'flex', alignItems: 'flex-end', gap: 72, marginTop: 60}}>
        <div style={{fontSize: 190, lineHeight: 0.9, fontWeight: 850, color: C.orange}}>14<span style={{color: C.muted}}>/14</span></div>
        <div style={{fontSize: 58, fontWeight: 720, lineHeight: 1.25}}>实际产物字节数<br/>等于最终 oracle 预测</div>
      </div>
      <div style={{...cutCard, marginTop: 100, padding: 48}}>
        <div style={{display: 'flex', justifyContent: 'space-between', fontSize: 23, color: C.muted}}><span>请求目标</span><span>100%</span></div>
        <div style={{height: 24, background: C.grid, marginTop: 20, position: 'relative'}}>
          <div style={{height: '100%', width: `${99.427 * p}%`, background: `linear-gradient(90deg, ${C.orange}, #FF8A42)`}} />
          <div style={{position: 'absolute', left: '99.427%', top: -14, bottom: -14, width: 2, background: C.blue}} />
        </div>
        <div style={{display: 'flex', justifyContent: 'space-between', marginTop: 24}}><span style={{fontSize: 28}}>目标利用率：99.427%–99.998%</span><span style={{fontSize: 23, color: C.blue}}>11.5G 的 67.5 MiB 空隙已记录</span></div>
      </div>
      <div style={{...rise(frame, 128), fontSize: 25, color: C.muted, marginTop: 36}}>预制档与自定义目标使用同一套 oracle 预测、量化与核验流程。</div>
    </AbsoluteFill>
  );
};

const Quality: React.FC<{duration: number}> = ({duration}) => {
  const frame = useCurrentFrame();
  const p = interpolate(frame, [16, 58], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp', easing: ease});
  return (
    <AbsoluteFill style={{opacity: sceneOpacity(frame, duration), padding: '70px 95px'}}>
      <CornerBrand />
      <div style={{...cutCard, position: 'absolute', left: 74, top: 65, width: 1250, height: 915, overflow: 'hidden', opacity: p, transform: `scale(${0.96 + 0.04 * p})`}}>
        <Img src={staticFile('kl-curve.png')} style={{width: '100%', height: '100%', objectFit: 'contain'}} />
      </div>
      <div style={{position: 'absolute', right: 95, top: 170, width: 430}}>
        <Label>06 / 实测质量</Label>
        <div style={{fontSize: 72, fontWeight: 820, marginTop: 34, color: C.orange}}>FIT-12G</div>
        <div style={{fontSize: 26, color: C.muted, marginTop: 8}}>五域宏平均</div>
        <div style={{...cutCard, marginTop: 35, padding: 30}}><div style={{fontSize: 22, color: C.muted}}>Macro KL ↓</div><div style={{fontSize: 62, fontWeight: 800, marginTop: 10}}>0.1227</div></div>
        <div style={{...cutCard, marginTop: 18, padding: 30}}><div style={{fontSize: 22, color: C.muted}}>Same-top ↑</div><div style={{fontSize: 62, fontWeight: 800, marginTop: 10}}>90.3%</div></div>
        <div style={{fontSize: 21, color: C.muted, lineHeight: 1.55, marginTop: 28}}>固定五域、512-token 协议下的观测值。不是通用能力排行榜。</div>
      </div>
    </AbsoluteFill>
  );
};

const HonestBoundary: React.FC<{duration: number}> = ({duration}) => {
  const frame = useCurrentFrame();
  return (
    <AbsoluteFill style={{opacity: sceneOpacity(frame, duration), padding: '115px 140px'}}>
      <CornerBrand />
      <div style={rise(frame)}><Label>07 / 边界同样重要</Label></div>
      <div style={{...rise(frame, 5), fontSize: 76, fontWeight: 780, marginTop: 22}}>尺寸可控，<span style={{color: C.orange}}>不等于万能最优。</span></div>
      <div style={{display: 'flex', gap: 34, marginTop: 84}}>
        <div style={{...cutCard, flex: 1, padding: 48, borderColor: C.orange}}>
          <div style={{fontSize: 24, letterSpacing: 4, color: C.orange}}>FIT 已解决</div>
          <div style={{fontSize: 52, fontWeight: 760, marginTop: 32}}>确定性的目标尺寸规划</div>
          <div style={{fontSize: 25, color: C.muted, lineHeight: 1.55, marginTop: 26}}>配方可复现。产物可验证。尺寸预测有明确适用边界。</div>
        </div>
        <div style={{...cutCard, flex: 1, padding: 48}}>
          <div style={{fontSize: 24, letterSpacing: 4, color: C.blue}}>仍待解决</div>
          <div style={{fontSize: 52, fontWeight: 760, marginTop: 32}}>跨模型的全局最优分配</div>
          <div style={{fontSize: 25, color: C.muted, lineHeight: 1.55, marginTop: 26}}>本模型 14 档曲线已修复为单调；跨模型最优仍未证实。</div>
        </div>
      </div>
    </AbsoluteFill>
  );
};

const Outro: React.FC<{duration: number}> = ({duration}) => {
  const frame = useCurrentFrame();
  const p = spring({frame: frame - 8, fps: 30, config: {damping: 20, stiffness: 90}});
  return (
    <AbsoluteFill style={{opacity: sceneOpacity(frame, duration), justifyContent: 'center', alignItems: 'center'}}>
      <Img src={staticFile('fit-logo.svg')} style={{width: 980, opacity: p, transform: `scale(${0.92 + p * 0.08})`}} />
      <div style={{...rise(frame, 48), fontSize: 60, fontWeight: 760, marginTop: 76}}>下载成品。或输入预算，<span style={{color: C.orange}}>自己 FIT。</span></div>
      <div style={{...rise(frame, 72), fontSize: 24, color: C.muted, letterSpacing: 4, marginTop: 30}}>14 PREBUILT TIERS · ARBITRARY TARGET-BYTES</div>
    </AbsoluteFill>
  );
};

export const FitPromo: React.FC = () => {
  return (
    <AbsoluteFill style={{fontFamily: font, backgroundColor: C.bg, color: C.ink}}>
      <Background />
      <Sequence from={0} durationInFrames={150}><Intro duration={150} /></Sequence>
      <Sequence from={120} durationInFrames={240}><PresetProblem duration={240} /></Sequence>
      <Sequence from={330} durationInFrames={240}><SizeSlider duration={240} /></Sequence>
      <Sequence from={540} durationInFrames={270}><Pipeline duration={270} /></Sequence>
      <Sequence from={780} durationInFrames={270}><TierGrid duration={270} /></Sequence>
      <Sequence from={1020} durationInFrames={240}><Accuracy duration={240} /></Sequence>
      <Sequence from={1230} durationInFrames={240}><Quality duration={240} /></Sequence>
      <Sequence from={1440} durationInFrames={240}><HonestBoundary duration={240} /></Sequence>
      <Sequence from={1650} durationInFrames={210}><Outro duration={210} /></Sequence>
    </AbsoluteFill>
  );
};
