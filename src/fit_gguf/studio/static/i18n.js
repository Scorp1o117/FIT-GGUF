'use strict';
// Only application copy is localized. Paths, model names and raw logs stay intact.
const studioEnglish={
  'FIT Studio 首页':'FIT Studio home','工作空间':'Workspace','主导航':'Main navigation',
  '指定质量档位':'Quality target','指定体积':'Size target','开始':'Start',
  '辅助工具':'Tools','硬件与预算':'Hardware & budget','模型适配':'Model fit',
  '任务记录':'Task history','校准注册表':'Calibration registry','本地运行':'Running locally',
  '模型文件留在你的机器上。':'Model files stay on your machine.',
  '按预算规划，以实际产物校验。':'Plan within budget. Verify the actual artifact.',
  '正在检测硬件…':'Detecting hardware…','刷新硬件':'Refresh hardware',
  '输入会自动保存到本地工作目录':'Edits are saved in the local workspace',
  '清除已存草稿':'Clear saved draft','重试保存':'Retry save',
  '正在保存草稿…':'Saving draft…','草稿保存失败 · 请重试':'Draft save failed. Please retry.',
  '草稿已保存 · 重开后可继续填写':'Draft saved. Continue after reopening.',
  '有修改，正在等待保存…':'Changes pending save…',
  '已清除草稿 · 当前输入保留，下次修改会重新保存':'Draft cleared. Inputs remain; editing saves again.',
  '清除失败 · 点击按钮重试':'Clear failed. Click to retry.',
  '草稿读取失败 · 任务记录仍可使用':'Draft could not be read. Task history is available.',
  '已恢复上次草稿 · 结果可从任务记录查看':'Draft restored. Results are in task history.',
  '你想控制什么？':'What would you like to control?',
  '选一种目标，剩下的交给 FIT。':'Choose a target. FIT handles the workflow.',
  '给定文件预算，分配每个张量的精度。':'Set a file budget and allocate tensor precision.',
  '选定质量目标，搜索通过评测的更小文件。':'Set a quality target and search for smaller verified files.',
  '例如：生成不超过 8 GiB 的 GGUF':'Example: a GGUF within an 8 GiB budget',
  '例如：满足 Balanced 档位的 KL 门槛':'Example: meet the Balanced KL threshold',
  '按体积量化':'Quantize to size','按质量搜索':'Search by quality',
  '体积流程校验实际字节；质量流程按冻结的 eval-v1 协议评测。两种结果分别记录。':'Size runs verify actual bytes; quality runs use frozen eval-v1 evaluation. Results are recorded separately.',
  '给定预算，生成可复现的量化方案。':'Create a reproducible quantization plan within your budget.',
  '目标文件体积 / GiB':'Target file size / GiB','源模型上限':'Source model limit',
  '先填预算并分析源模型，FIT 会显示这个预设区间能达到的体积。':'Enter a budget and analyze the source to see the achievable preset interval.',
  '模型与分析':'Model & analysis','执行与校验':'Run & verify','规划精度分配':'Allocate precision',
  '分析模型':'Analyze model','分析候选集':'Analyze candidates','生成量化方案':'Generate plan',
  '规划':'Plan','输入':'Input','确定性':'Deterministic','实际产物':'Actual artifact',
  '需要源模型 GGUF、重要性矩阵，以及支持 dry-run 的 llama.cpp。':'Requires source GGUF, importance matrix and a dry-run capable llama.cpp.',
  '源模型 GGUF':'Source GGUF','重要性矩阵 GGUF':'Importance matrix GGUF',
  'llama.cpp 运行时目录':'llama.cpp runtime directory','下界预设':'Lower preset','上界预设':'Upper preset',
  '桌面版选择文件':'Select a file in desktop mode','桌面版选择目录':'Select a directory in desktop mode',
  '分析记录':'Analysis record','方案记录':'Plan record','读取已有分析':'Load existing analysis',
  '分析完成后自动填入；也可载入已有 analysis.json':'Filled after analysis, or load an existing analysis.json',
  '规划完成后自动填入；也可选择已有 FIT-plan.json':'Filled after planning, or select an existing FIT-plan.json',
  '先完成分析，查看可规划的预算区间。':'Analyze first to see the valid budget interval.',
  'GiB 目标文件大小':'GiB target file size','目标字节预算':'Target byte budget','精确目标 / bytes':'Exact target / bytes',
  '分配策略':'Allocation policy','按区块均衡':'Balanced across blocks','贪心分配':'Greedy allocation',
  '预算必须位于所选预设区间内。张量升级是离散的，最终文件可能略小于目标。':'Budget must be in the preset interval. Discrete tensor upgrades may leave some budget unused.',
  '将执行 llama-quantize。每次任务写入独立目录，成功必须通过字节校验。':'Runs llama-quantize in an independent task directory. Success requires byte verification.',
  '量化并验证':'Quantize & verify','当前结果':'Current result','等待任务':'Waiting for a task',
  '还没有量化方案':'No plan yet','先分析你的源模型。完成后，区间、方案和实际校验结果会显示在这里。':'Analyze the source first. The interval, plan and verified result will appear here.',
  '任务日志':'Task log','取消任务':'Cancel task','任务启动后会显示日志。':'Logs appear after a task starts.',
  '按顺序运行一个任务；历史记录与产物都保存在本地。':'One task runs at a time. History and artifacts stay local.',
  '固定 KL 目标，在评测通过的候选中搜索更小的 GGUF。':'Fix a KL threshold and search passing candidates for a smaller GGUF.',
  '选择质量档位':'Choose a quality tier','模型与参考':'Model & references','源浮点 GGUF':'Source floating-point GGUF',
  '对应的 imatrix GGUF':'Matching imatrix GGUF','包含 llama-quantize 和 llama-perplexity':'Contains llama-quantize and llama-perplexity',
  '评测参考文件':'Evaluation references','参考 KLD 目录':'Reference KLD directory',
  '评测语料目录':'Evaluation corpus directory','冻结协议 FREEZE.json':'Frozen protocol FREEZE.json',
  '参考清单 reference-manifest.json':'Reference manifest reference-manifest.json',
  '质量评测需要与源权重对应的五域参考。可以使用 fit calibrate 生成的参考与清单。':'Quality evaluation requires five-domain references bound to the source weights. Use references and manifests from fit calibrate.',
  '包含五域 kl-eval 文本':'Contains five-domain kl-eval texts','绑定源权重、参考和语料哈希':'Binds source weights, references and corpus hashes',
  '包含 bf16-<domain>.kld':'Contains bf16-<domain>.kld','CPU 评测线程':'CPU evaluation threads',
  '默认 CPU 评测、单任务执行；最多 8 次新搜索评测，最终产物另行复测。源模型仍受 5B 上限约束。':'CPU evaluation, one task at a time, up to 8 fresh search evaluations plus final re-evaluation. The 5B source limit applies.',
  '搜索并验证质量档位':'Search & verify quality','质量搜索结果':'Quality search result',
  '先选择你要的档位':'Choose a tier first','只有通过实际 KL 评测并复测的文件才会作为产物交付。':'Only files passing actual KL evaluation and final re-evaluation are delivered.',
  '硬件建议是估算 · 文件产物以实际校验为准':'Hardware recommendations are estimates. Actual artifacts are verified.',
  '模型或预设已修改，请重新分析。历史产物仍保留在任务记录中。':'Model or presets changed. Analyze again; historical artifacts remain in task history.',
  '输入已修改 · 请重新规划':'Inputs changed. Generate a new plan.',
  '需要重新分析':'Analysis required',
  '草稿中的分析记录已不可用，请重新分析或载入有效的 analysis.json。':'Draft analysis is unavailable. Analyze again or load a valid analysis.json.',
  '已读取分析记录。':'Analysis loaded.','已恢复评测参考 · 展开修改':'References restored. Expand to edit.',
  '等待启动':'Queued','运行中':'Running','已完成':'Completed','正在取消':'Cancelling','已取消':'Cancelled','已中断':'Interrupted','失败':'Failed',
  '复制路径':'Copy path','打开目录':'Open folder','任务已启动。':'Task started.',
  '任务正在启动…':'Starting task…','正在处理当前任务':'Processing task',
  '完成校验后会显示本次结果。':'Results appear after verification.',
  '本次任务未交付结果':'This task did not deliver a result','量化校验':'Quantization verification','质量档位搜索':'Quality tier search',
  '量化参数精度分布':'Parameter precision distribution','下界文件大小':'Lower file size','上界文件大小':'Upper file size',
  '目标预算':'Target budget','预测文件大小':'Predicted file size','实际文件大小':'Actual file size',
  '字节校验':'Byte verification','实际字节数':'Actual bytes','预测字节数':'Predicted bytes',
  '候选张量升级':'Candidate tensor upgrades','预设区间':'Preset interval','可分配差值':'Allocatable difference',
  '预算剩余':'Unused budget','已选择升级':'Selected upgrades','主要量化类型':'Dominant quantization type',
  '运行时确认轮次':'Runtime confirmation rounds','实际与预测差值':'Actual minus predicted bytes',
  '这是大小方案，不代表通过了量化质量评测。':'This is a size plan, not a quality evaluation pass.',
  '交付状态':'Delivery status','验证通过':'Verified pass','未交付':'Not delivered','最终文件体积':'Final file size',
  '搜索状态':'Search status','最终实测 macro KL':'Final measured macro KL','最终实测 Same-top':'Final measured Same-top',
  '新搜索评测':'Fresh search evaluations','停止条件':'Stopping condition','参考指标':'Reference metric',
  '本地硬件检测':'Local hardware detection','llmfit 已连接':'llmfit connected',
  '质量门槛越低，允许的分布偏差越小；档位不是通用准确率。':'Lower thresholds allow less distribution divergence. Tiers are not universal accuracy scores.',
  '宽松门槛，允许搜索更多压缩候选。':'A relaxed threshold allows more compressed candidates.',
  '偏向压缩，仍需通过冻结评测。':'Favors compression, subject to frozen evaluation.',
  '默认折中，先从这里比较实际结果。':'Default tradeoff. Start here and compare measured results.',
  '更严格的分布偏差门槛。':'A stricter distribution divergence threshold.',
  '最严格门槛，可能需要更大体积。':'Strictest threshold; may require larger files.',
};
const textSources=new WeakMap(), attributeSources=new WeakMap();
const translationKeys=Object.keys(studioEnglish).sort((a,b)=>b.length-a.length);
const translationExpression=new RegExp(translationKeys.map(key=>key.replace(/[.*+?^${}()|[\]\\]/g,'\\$&')).join('|'),'g');
let studioLanguage='zh-CN';
function translateStudioText(source) {
  if(studioLanguage!=='en')return source;
  // A single replacement pass prevents translated fragments being replaced again.
  return source.replace(translationExpression,key=>studioEnglish[key]);
}
function localizeStudio(root=document.body) {
  const nodes=document.createTreeWalker(root,NodeFilter.SHOW_TEXT);
  while(nodes.nextNode()) {
    const node=nodes.currentNode;
    if(node.parentElement?.closest('script,style,pre,.mono,.result-artifact,.model-card h3,.model-meta,.result-details dd,#globalError'))continue;
    let record=textSources.get(node);
    if(!record||node.textContent!==record.rendered)record={source:node.textContent};
    const rendered=translateStudioText(record.source);
    if(node.textContent!==rendered)node.textContent=rendered;
    record.rendered=rendered;textSources.set(node,record);
  }
  for(const node of root.querySelectorAll('[title],[placeholder],[aria-label]')) {
    const records=attributeSources.get(node)||{};
    for(const name of ['title','placeholder','aria-label']) {
      if(!node.hasAttribute(name))continue;
      const value=node.getAttribute(name);
      let record=records[name];
      if(!record||record.rendered!==value)record={source:value};
      const rendered=translateStudioText(record.source);
      if(value!==rendered)node.setAttribute(name,rendered);
      record.rendered=rendered;records[name]=record;
    }
    attributeSources.set(node,records);
  }
}
window.FitI18n={setLanguage(language){studioLanguage=language==='en'?'en':'zh-CN';document.documentElement.lang=studioLanguage;document.title=studioLanguage==='en'?'FIT Studio · GGUF quantization':'FIT Studio · 精确量化工作台';localizeStudio();},translate:translateStudioText};
let localizationPending=false;
new MutationObserver(()=> {
  if(localizationPending)return;
  localizationPending=true;queueMicrotask(()=>{localizationPending=false;localizeStudio();});
}).observe(document.body,{childList:true,subtree:true,characterData:true});
