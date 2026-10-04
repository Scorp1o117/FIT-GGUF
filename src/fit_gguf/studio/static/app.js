'use strict';
const $ = selector => document.querySelector(selector);
const GIB = 1024 ** 3;
const state = {system:null, budget:null, analysis:null, jobs:[], models:[], modelLimit:40, active:null, lastResult:null, polling:false};
const labels = {start:'开始',overview:'硬件与预算',workspace:'指定体积',quality:'指定质量档位',models:'模型适配',jobs:'任务记录',registry:'校准注册表'};
const statuses = {queued:'等待启动',running:'运行中',succeeded:'已完成',failed:'失败',cancelled:'已取消',cancelling:'正在取消',interrupted:'已中断'};
const actions = {analyze:'分析模型',plan:'规划精度',quantize:'量化校验',quality:'质量档位搜索'};
const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const giB = value => value == null ? '—' : (Number(value)/GIB).toFixed(Number(value)/GIB<.1?4:2);
const num = value => Number(value).toLocaleString('zh-CN');
const detail = (key,value) => `<div><dt>${esc(key)}</dt><dd>${esc(value)}</dd></div>`;
const artifactPath = (path,jobId) => `<div class="artifact-location"><div class="result-artifact">${esc(path)}</div><button type="button" class="text-button" data-copy-path="${esc(path)}">复制路径</button><button type="button" class="text-button open-directory" data-open-job="${esc(jobId)}">打开目录</button></div>`;
const draftForms=['analyzeForm','planForm','quantizeForm','qualityForm'];
const draftSettings=['desiredGiB','reserve','overhead','runMode','maxParams','context','language'];
function updateTierHelp() {
  const descriptions={mini:'宽松门槛，允许搜索更多压缩候选。',compact:'偏向压缩，仍需通过冻结评测。',balanced:'默认折中，先从这里比较实际结果。',quality:'更严格的分布偏差门槛。',reference:'最严格门槛，可能需要更大体积。'};
  $('#tierHelp').textContent=descriptions[$('#qualityForm [name=tier]:checked')?.value]||descriptions.balanced;
}
let draftReady=false, draftTimer, draftQueue=Promise.resolve();
function collectDraft() {
  const fields={};
  for(const id of draftForms) for(const [name,value] of new FormData($('#'+id))) fields[`${id}.${name}`]=String(value);
  for(const id of draftSettings) fields[id]=$('#'+id).value;
  return {fields};
}
function applyDraft(draft) {
  for(const [key,value] of Object.entries(draft.fields||{})) {
    const [form,name]=key.split('.');
    const elements=name&&draftForms.includes(form)?$('#'+form).querySelectorAll(`[name="${name}"]`):draftSettings.includes(key)?[$('#'+key)]:[];
    for(const input of elements) {
      if(input.type==='radio') input.checked=input.value===value;
      else if(input.tagName!=='SELECT'||Array.from(input.options).some(option=>option.value===value)) input.value=value;
    }
  }
  $('#maxParams').value=Math.min(Number($('#maxParams').value)||5,Number($('#maxParams').max));
  $('#targetLabel').textContent=Number($('#targetBytes').value)>0?giB($('#targetBytes').value):'—';
  updateTierHelp();window.FitI18n.setLanguage($('#language').value);
}
function saveDraft() {
  if(!draftReady)return draftQueue;
  clearTimeout(draftTimer);
  const draft=collectDraft();
  $('#retryDraft').classList.add('hidden');
  $('#draftStatus').textContent='正在保存草稿…';
  draftQueue=draftQueue.then(()=>api('/api/draft',draft)).then(()=> {
    $('#draftStatus').textContent='草稿已保存 · 重开后可继续填写';
  }).catch(()=> {$('#draftStatus').textContent='草稿保存失败 · 请重试';$('#retryDraft').classList.remove('hidden');});
  return draftQueue;
}
function scheduleDraft() {
  if(!draftReady)return;
  $('#draftStatus').textContent='有修改，正在等待保存…';
  clearTimeout(draftTimer); draftTimer=setTimeout(saveDraft,400);
}
function precisionChart(shares) {
  const rows=Object.entries(shares||{}).filter(([,value])=>Number.isFinite(value)&&value>0).sort((a,b)=>b[1]-a[1]);
  const colors=['#a6f0d0','#a69df5','#76aace','#edc181','#e3a2b3','#7bb8a5'];
  return rows.length?`<div class="precision-chart"><p class="caption">量化参数精度分布</p><div class="precision-track">${rows.map(([name,value],i)=>`<span title="${esc(name.toUpperCase())} ${(value*100).toFixed(1)}%" style="width:${Math.min(value*100,100)}%;background:${colors[i%colors.length]}"></span>`).join('')}</div><div class="precision-legend">${rows.map(([name,value],i)=>`<span><i style="background:${colors[i%colors.length]}"></i>${esc(name.toUpperCase())} ${(value*100).toFixed(1)}%</span>`).join('')}</div></div>`:'';
}
const stat = (name,value,unit,note,icon) => `<article class="stat"><div class="stat-label"><span>${esc(name)}</span><span>${icon}</span></div><div class="stat-value">${esc(value)}<small>${esc(unit)}</small></div><div class="stat-note">${esc(note)}</div></article>`;
async function api(path, payload) {
  const response = await fetch(path, payload === undefined ? {} : {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `请求失败 (${response.status})`);
  return data;
}
function error(message) { $('#globalError').textContent=message; $('#globalError').classList.remove('hidden'); }
function clearError() { $('#globalError').classList.add('hidden'); }
function setBusy(busy) { for(const selector of ['#analyzeForm','#planForm','#quantizeForm','#qualityForm']) $(selector).querySelector('button[type=submit]').disabled=busy; }
function sizeStep(id) {
  for(const form of ['analyzeForm','planForm','quantizeForm']) $('#'+form).classList.toggle('hidden',form!==id);
  document.querySelectorAll('[data-size-step]').forEach(button=>{button.classList.toggle('selected',button.dataset.sizeStep===id);button.setAttribute('aria-current',button.dataset.sizeStep===id?'step':'false');});
}
function syncModelFields(form) {
  for(const field of ['source','imatrix','runtime']) {
    const value=form.querySelector(`[name=${field}]`).value;
    for(const other of ['#analyzeForm','#qualityForm']) $(other).querySelector(`[name=${field}]`).value=value;
  }
}
function invalidatePlan() {
  if($('#planPath').value)$('#resultStatus').textContent='输入已修改 · 请重新规划';
  $('#planPath').value='';
}
function invalidateAnalysis() {
  state.analysis=null;$('#analysisPath').value='';invalidatePlan();
  $('#targetSlider').disabled=true;
  $('#targetBytes').min=1;$('#targetBytes').removeAttribute('max');
  $('#interval').textContent='模型或预设已修改，请重新分析。历史产物仍保留在任务记录中。';
  $('#resultStatus').textContent='需要重新分析';
}
let toastTimer;
function toast(message) { $('#toast').textContent=message; $('#toast').classList.remove('hidden'); clearTimeout(toastTimer); toastTimer=setTimeout(()=>$('#toast').classList.add('hidden'),3500); }
function go(page) {
  document.querySelectorAll('.page').forEach(node=>node.classList.toggle('hidden',node.id!==page));
  document.querySelectorAll('.nav').forEach(node=>node.classList.toggle('active',node.dataset.page===page));
  $('#pageName').textContent=labels[page];
  document.querySelectorAll('[data-page]').forEach(button=>button.setAttribute('aria-current',button.dataset.page===page?'page':'false'));
  if (page==='registry') loadRegistry().catch(e=>error(e.message));
  if (page==='jobs') refreshJobs().catch(e=>error(e.message));
  window.scrollTo({top:0,behavior:'smooth'});
}
async function refreshHardware() {
  $('#refresh').disabled=true;
  try {
    const data=await api('/api/system'); state.system=data.system;
    $('#integration').textContent=data.llmfit_available?'llmfit 已连接':'本地硬件检测';
    $('#hardwareSource').textContent=data.provider;
    const system=data.system, gpus=system.gpus||[], gpu=gpus.reduce((best,row)=>(row.free_vram_gb||0)>(best?.free_vram_gb||0)?row:best,gpus[0]);
    const available=gpu?.free_vram_gb;
    $('#hardwareCards').innerHTML=stat('GPU · 空闲显存',available==null?'—':available.toFixed(1),'GiB',gpu?`${gpu.name} · 总显存 ${Number(gpu.vram_gb).toFixed(1)} GiB`:'未获取空闲 GPU 显存','◈')+
      stat('内存 · 当前可用',system.available_ram_gb==null?'—':Number(system.available_ram_gb).toFixed(1),'GiB',`物理内存 ${system.total_ram_gb==null?'未知':Number(system.total_ram_gb).toFixed(1)+' GiB'} · 系统和应用已占用部分内存`,'◫')+
      stat('CPU · 逻辑核心',system.cpu_cores||system.total_cpu_cores||'—','cores',system.cpu_name||'CPU','⌘');
    await updateBudget();
    if(data.warnings?.length) toast('部分硬件检测不可用，已使用本地检测。');
  } finally { $('#refresh').disabled=false; }
}
async function updateBudget() {
  state.budget=await api('/api/budget',{reserve_gb:Number($('#reserve').value),overhead_gb:Number($('#overhead').value),mode:$('#runMode').value});
  $('#budgetValue').textContent=giB(state.budget.target_bytes);
  const ratio=state.budget.available_gb?state.budget.target_bytes/GIB/state.budget.available_gb:0;
  $('#budgetFill').style.width=`${Math.max(0,Math.min(100,ratio*100))}%`;
  $('#useBudget').disabled=!(state.budget.target_bytes>0);
}
function setTarget(value) {
  $('#targetBytes').value=Math.round(value);
  $('#desiredGiB').value=String(Math.round(value)/GIB);
  $('#targetLabel').textContent=giB(value);
  if (state.analysis) $('#targetSlider').value=value;
  scheduleDraft();
}
function acceptAnalysis(path,result) {
  if($('#analysisPath').value!==path) $('#planPath').value='';
  state.analysis=result; $('#analysisPath').value=path;
  for(const field of ['source','imatrix']) $('#analyzeForm').querySelector(`[name=${field}]`).value=result[field].path;
  $('#analyzeForm [name=runtime]').value=result.runtime.dir;
  $('#lowerPreset').value=result.presets.lower.name;$('#upperPreset').value=result.presets.upper.name;
  syncModelFields($('#analyzeForm'));
  const lower=result.presets.lower.predicted_size_bytes, upper=result.presets.upper.predicted_size_bytes;
  const slider=$('#targetSlider'); slider.disabled=false; slider.min=lower; slider.max=upper; slider.step=1;
  $('#interval').textContent=`${result.presets.lower.name} (${giB(lower)} GiB) → ${result.presets.upper.name} (${giB(upper)} GiB) · ${num(lower)}–${num(upper)} bytes`;
  $('#targetBytes').min=lower; $('#targetBytes').max=upper;
  const candidate=Number($('#targetBytes').value)||Math.round((lower+upper)/2);
  setTarget(Math.min(upper,candidate));
  if(candidate<lower) error(`目标体积低于当前区间下限 ${giB(lower)} GiB。请选择更低的预设重新分析，或主动调整预算。`);
}
async function loadAnalysis() {
  const data=await api('/api/analysis',{path:$('#analysisPath').value});
  invalidatePlan();
  acceptAnalysis(data.path,data.result); toast('已读取分析记录。');
}
async function restoreJob(job, syncInputs=true) {
  if(job.action==='quality') {
    for(const field of ['source','imatrix','runtime','refs_dir','eval_data_dir','freeze','reference_manifest','threads']) {
      const flag='--'+field.replaceAll('_','-'), index=job.command.indexOf(flag);
      if(index>=0) $('#qualityForm').querySelector(`[name=${field}]`).value=job.command[index+1];
    }
    const tierIndex=job.command.indexOf('--tier');
    if(tierIndex>=0) $('#qualityForm').querySelector(`[value=${job.command[tierIndex+1]}]`).checked=true;
    updateTierHelp();
    const references=$('#qualityForm .quality-inputs');
    references.open=false;
    references.querySelector('summary').textContent='已恢复评测参考 · 展开修改';
    if(syncInputs)syncModelFields($('#qualityForm'));renderResult(job);return;
  }
  const path=job.action==='analyze'?job.artifacts.analysis:job.result?.analysis_path;
  if(job.status==='succeeded'&&path) {
    const analysis=await api('/api/analysis',{path}); acceptAnalysis(analysis.path,analysis.result);
    for(const field of ['source','imatrix']) $('#analyzeForm').querySelector(`[name=${field}]`).value=analysis.result[field].path;
    $('#analyzeForm').querySelector('[name=runtime]').value=analysis.result.runtime.dir;
    if(syncInputs)syncModelFields($('#analyzeForm'));
    $('#lowerPreset').value=analysis.result.presets.lower.name; $('#upperPreset').value=analysis.result.presets.upper.name;
    if(job.action==='plan') {setTarget(job.result.target_bytes);$('#planForm').querySelector('[name=policy]').value=job.result.policy;}
    if(job.action==='quantize') {
      $('#planPath').value='';
      for(const row of state.jobs.filter(row=>row.action==='plan'&&row.status==='succeeded')) {
        const plan=await api(`/api/jobs/${row.id}`);
        if(plan.result.analysis_path===path&&plan.result.predicted_size_bytes===job.result.expect_bytes&&plan.result.tensor_types_sha256===job.result.tensor_types_sha256) {
          $('#planPath').value=plan.artifacts.plan;setTarget(plan.result.target_bytes);$('#planForm').querySelector('[name=policy]').value=plan.result.policy;break;
        }
      }
    }
  }
  renderResult(job);
}
function renderResult(job) {
  const quality=job.action==='quality';
  $(quality?'#qualityStatus':'#resultStatus').textContent=statuses[job.status]||job.status;
  if(!job.result) {
    const result=$(quality?'#qualityResult':'#resultSummary');
    const running=['queued','running','cancelling'].includes(job.status);
    result.classList.add('empty');
    result.innerHTML=`<h3>${running?'正在处理当前任务':'本次任务未交付结果'}</h3><p>${esc(running?'完成校验后会显示本次结果。':job.error||'请查看任务状态和日志；未校验的部分文件保留在任务目录中。')}</p>`;
    return;
  }
  if(job.action==='quality') {
    $('#qualityStatus').textContent=statuses[job.status]||job.status;
    if(!job.result) return;
    const r=job.result, metrics=r.artifact?.verified_metrics, delivered=job.status==='succeeded'&&r.status==='verified_pass'&&r.artifact;
    $('#qualityResult').classList.remove('empty');
    $('#qualityResult').innerHTML=`<div class="result-stats"><div class="result-stat"><small>交付状态</small><strong>${delivered?'验证通过':'未交付'}</strong></div><div class="result-stat"><small>最终文件体积</small><strong>${delivered?giB(r.artifact.size_bytes)+' GiB':'—'}</strong></div></div><dl class="result-details">${detail('搜索状态',r.product_status||r.status)}${detail('最终实测 macro KL',metrics?.macro_kl==null?'—':Number(metrics.macro_kl).toFixed(6))}${detail('最终实测 Same-top',metrics?.same_top_pct==null?'—':Number(metrics.same_top_pct).toFixed(2)+'% · 参考指标')}${detail('新搜索评测',`${r.fresh_evals??'—'} / ${r.budget??'—'}`)}${detail('停止条件',r.active_constraint||'—')}</dl>${delivered?`${artifactPath(r.artifact.path,job.id)}`:''}<p class="caption">${esc(r.note||'KL 是档位门槛；Same-top 仅作参考。仅在搜索窗口和预算内寻找通过验证的更小产物。')}</p>`;
    return;
  }
  $('#resultStatus').textContent=statuses[job.status]||job.status;
  if (!job.result) return;
  $('#resultSummary').classList.remove('empty');
  const r=job.result, a=job.artifacts;
  if(job.action==='analyze') {
    acceptAnalysis(a.analysis,r);
    sizeStep('planForm');
    $('#resultSummary').innerHTML=`<div class="result-stats"><div class="result-stat"><small>下界文件大小</small><strong>${giB(r.presets.lower.predicted_size_bytes)} GiB</strong></div><div class="result-stat"><small>上界文件大小</small><strong>${giB(r.presets.upper.predicted_size_bytes)} GiB</strong></div></div><dl class="result-details">${detail('候选张量升级',num(r.candidate_count))}${detail('预设区间',r.presets.lower.name+' → '+r.presets.upper.name)}${detail('可分配差值',num(r.net_preset_gap_bytes)+' bytes')}</dl>${artifactPath(a.analysis,job.id)}`;
  } else if(job.action==='plan') {
    sizeStep('quantizeForm');
    $('#planPath').value=a.plan; $('#analysisPath').value=r.analysis_path;
    $('#resultSummary').innerHTML=`<div class="result-stats"><div class="result-stat"><small>目标预算</small><strong>${giB(r.target_bytes)} GiB</strong></div><div class="result-stat"><small>预测文件大小</small><strong>${giB(r.predicted_size_bytes)} GiB</strong></div></div>${precisionChart(r.qtype_parameter_shares)}<dl class="result-details">${detail('预测字节数',num(r.predicted_size_bytes))}${detail('预算剩余',num(r.unused_bytes)+' bytes')}${detail('已选择升级',num(r.selected_count)+' 个')}${detail('主要量化类型',String(r.dominant_qtype||'—').toUpperCase())}${detail('分配策略',r.policy)}${detail('运行时确认轮次',r.oracle_iterations)}</dl>${artifactPath(a.plan,job.id)}<p class="caption">这是大小方案，不代表通过了量化质量评测。</p>`;
  } else {
    $('#analysisPath').value=r.analysis_path;
    const pass=r.size_matches_refinalization&&r.size_matches_expectation;
    $('#resultSummary').innerHTML=`<div class="result-stats"><div class="result-stat"><small>实际文件大小</small><strong>${giB(r.size_bytes)} GiB</strong></div><div class="result-stat"><small>字节校验</small><strong>${pass?'PASS':'FAIL'}</strong></div></div><dl class="result-details">${detail('实际字节数',num(r.size_bytes))}${detail('实际与预测差值',num(r.size_bytes-r.refinalized_expected_bytes)+' bytes')}${detail('SHA-256',r.sha256)}</dl>${artifactPath(a.model,job.id)}<p class="caption">文件大小校验通过不代表量化质量已评测。质量评测使用 FIT 的校准与保真度流程。</p>`;
  }
}
async function watchJob(id) {
  const job=await api(`/api/jobs/${id}`);
  const quality=job.action==='quality';
  $(quality?'#qualityLog':'#liveLog').textContent=job.log_text||job.error||'任务运行中。底层量化日志可能在当前步骤结束后写入。';
  $(quality?'#qualityStatus':'#resultStatus').textContent=statuses[job.status]||job.status;
  const running=['queued','running','cancelling'].includes(job.status);
  $('#cancelJob').classList.toggle('hidden',!running||quality);
  $('#cancelQuality').classList.toggle('hidden',!running||!quality);
  if (!running) {
    if(state.active===id) state.active=null;
    if(state.lastResult!==id) {
      state.lastResult=id;
      if(job.status==='succeeded') { renderResult(job); scheduleDraft(); toast(`${actions[job.action]}完成。`); }
      else {renderResult(job);if(job.status==='failed')error(job.error||'任务未能交付产物，请查看结果和日志。');}
    }
  }
  setBusy(Boolean(state.active));
  return job;
}
async function submit(form,action) {
  clearError();
  setBusy(true);
  try {
    const data=Object.fromEntries(new FormData(form)); data.action=action;
    if(action==='plan') data.target_bytes=Number(data.target_bytes);
    if(action==='quality') {data.threads=Number(data.threads);syncModelFields(form);}
    if(action==='analyze') syncModelFields(form);
    if(action==='quantize') data.analysis=$('#analysisPath').value;
    if(['plan','quantize'].includes(action)) {
      data.model_context=Object.fromEntries(new FormData($('#analyzeForm')));
      if(action==='quantize') data.plan_context={target_bytes:Number($('#targetBytes').value),policy:$('#planForm [name=policy]').value};
    }
    await saveDraft();
    const job=await api('/api/jobs',data);
    state.active=job.id; state.lastResult=null;
    const result=$(action==='quality'?'#qualityResult':'#resultSummary');
    result.classList.add('empty');
    result.innerHTML='<h3>正在处理当前任务</h3><p>完成校验后会显示本次结果。</p>';
    $(action==='quality'?'#qualityLog':'#liveLog').textContent='任务正在启动…'; $(action==='quality'?'#qualityStatus':'#resultStatus').textContent='等待启动';
    await refreshJobs(); await watchJob(job.id);
    toast('任务已启动。');
  } finally { setBusy(Boolean(state.active)); }
}
async function refreshJobs() {
  state.jobs=await api('/api/jobs'); $('#jobCount').textContent=state.jobs.length;
  if(!state.active) state.active=state.jobs.find(j=>['queued','running','cancelling'].includes(j.status))?.id||null;
  if(!state.jobs.length) { $('#jobsList').innerHTML='<div class="empty">还没有任务。去工作台创建第一个分析任务。</div>'; return; }
  $('#jobsList').innerHTML=state.jobs.map(job=>`<div class="job-row"><div><h3>${esc(actions[job.action])} <span class="mono muted">${esc(job.id.slice(0,8))}</span></h3><p>${esc(new Date(job.created_at).toLocaleString('zh-CN'))}</p></div><div><span class="status ${esc(job.status)}">${esc(statuses[job.status]||job.status)}</span> <button class="text-button" data-job="${esc(job.id)}">查看 →</button></div></div>`).join('');
}
function renderModels() {
  const needle=$('#modelSearch').value.toLowerCase();
  const rows=state.models.filter(row=>[row.name,row.provider,row.use_case].join(' ').toLowerCase().includes(needle));
  $('#modelRows').innerHTML=rows.length?rows.slice(0,state.modelLimit).map(row=> {
    const safeRepo=/^[\w.-]+\/[\w.-]+$/.test(row.name);
    return `<article class="model-card"><div class="model-top"><span class="tag">${esc(row.fit_level)}</span><span class="model-score">${Number(row.score).toFixed(0)}<small>/100 适配评分</small></span></div><h3>${esc(row.name)}</h3><div class="model-meta">${esc(row.parameter_count)} · ${esc(row.provider)} · ${esc(row.use_case)}</div><div class="result-stats"><div class="result-stat"><small>内存需求估算</small><strong>${Number(row.memory_required_gb).toFixed(2)} GiB</strong></div><div class="result-stat"><small>建议量化</small><strong>${esc(row.best_quant)}</strong></div></div><dl class="result-details">${detail('运行方式',row.run_mode)}${detail('上下文',num(row.effective_context_length||row.context_length||0)+' tokens')}${detail('估算速度',Number(row.estimated_tps).toFixed(1)+' tok/s · 非本机实测')}</dl>${safeRepo?`<a class="model-link" href="https://huggingface.co/${encodeURI(row.name)}" target="_blank" rel="noopener noreferrer">查看源模型 ↗</a>`:''}</article>`;
  }).join('')+(rows.length>state.modelLimit?`<button type="button" class="secondary" id="moreModels">继续显示（已显示 ${state.modelLimit} / ${rows.length}）</button>`:''):'<div class="empty">没有匹配的模型。可调整搜索条件或上下文长度。</div>';
}
async function loadModels() {
  const button=$('#modelsForm button'); button.disabled=true; clearError();
  $('#modelsNote').textContent='正在读取 llmfit 适配结果…';
  try {
    const data=await api(`/api/models?max_params=${encodeURIComponent($('#maxParams').value)}&context=${encodeURIComponent($('#context').value)}`);
    state.models=data.models; state.modelLimit=40; renderModels();
    $('#modelsNote').textContent=data.available?`${data.models.length} 个模型 · ≤ ${data.max_params}B · ${data.context} tokens · 按当前空闲内存扣除 2 GiB 预留评估；适配与速度为估算，未进行下载或推理。`:data.note;
  } finally { button.disabled=false; }
}
async function loadRegistry() {
  const data=await api('/api/registry');
  $('#registryRows').innerHTML=(data.entries||[]).map(row=>`<article class="model-card"><span class="tag">${esc(row.status)}</span><h3>${esc(row.model_id)}</h3><p class="mono">${esc(row.source_weights_sha256)}</p><p class="caption">仅适用于这个源权重哈希。查看详细校准记录可使用 fit registry show。</p></article>`).join('')||'<div class="empty">注册表暂时没有记录。</div>';
}
document.querySelectorAll('[data-page]').forEach(b=>b.addEventListener('click',()=>go(b.dataset.page)));
document.querySelectorAll('[data-go]').forEach(b=>b.addEventListener('click',()=>go(b.dataset.go)));
document.querySelectorAll('[data-size-step]').forEach(b=>b.addEventListener('click',()=>sizeStep(b.dataset.sizeStep)));
document.querySelectorAll('[data-page]').forEach(b=>{b.title=labels[b.dataset.page];b.setAttribute('aria-label',labels[b.dataset.page]);});
document.addEventListener('input',event=>{if(event.target.matches('input,select'))scheduleDraft();});
document.addEventListener('change',event=>{if(event.target.matches('input,select'))scheduleDraft();});
$('#retryDraft').addEventListener('click',()=>saveDraft());
$('#language').addEventListener('change',()=>window.FitI18n.setLanguage($('#language').value));
$('#qualityForm').addEventListener('change',updateTierHelp);
$('#clearDraft').addEventListener('click',async()=> {
  clearTimeout(draftTimer);
  const button=$('#clearDraft');button.disabled=true;
  // Clear follows all earlier saves so an in-flight request cannot recreate it.
  draftQueue=draftQueue.then(()=>api('/api/draft',{clear:true})).then(()=> {
    $('#draftStatus').textContent='已清除草稿 · 当前输入保留，下次修改会重新保存';
    $('#retryDraft').classList.add('hidden');
  }).catch(()=> {$('#draftStatus').textContent='清除失败 · 点击按钮重试';});
  await draftQueue;button.disabled=false;
});
$('#desiredGiB').addEventListener('input',event=> {
  invalidatePlan();
  const bytes=Math.round(Number(event.target.value)*GIB);
  $('#targetBytes').value=Number.isSafeInteger(bytes)&&bytes>0?bytes:'';
  $('#targetLabel').textContent=bytes>0?giB(bytes):'—';
  if(state.analysis)$('#targetSlider').value=bytes;
});
for(const form of ['#analyzeForm','#qualityForm']) for(const field of ['source','imatrix','runtime']) for(const event of ['input','change']) $(form).querySelector(`[name=${field}]`).addEventListener(event,()=>{syncModelFields($(form));invalidateAnalysis();});
for(const selector of ['#lowerPreset','#upperPreset']) $(selector).addEventListener('change',invalidateAnalysis);
$('#analysisPath').addEventListener('input',()=>{state.analysis=null;invalidatePlan();$('#targetSlider').disabled=true;$('#targetBytes').min=1;$('#targetBytes').removeAttribute('max');});
$('#planForm [name=policy]').addEventListener('change',invalidatePlan);
$('#refresh').addEventListener('click',()=>refreshHardware().catch(e=>error(e.message)));
for(const selector of ['#reserve','#overhead','#runMode']) $(selector).addEventListener('change',()=>updateBudget().catch(e=>error(e.message)));
$('#useBudget').addEventListener('click',()=> {
  let target=state.budget.target_bytes;
  if(state.analysis) {
    if(target<state.analysis.presets.lower.predicted_size_bytes) {error('硬件预算低于当前预设区间，请选择更低的预设重新分析。');go('workspace');return;}
    target=Math.min(state.analysis.presets.upper.predicted_size_bytes,target);
  }
  invalidatePlan();setTarget(target); go('workspace'); toast(target===state.budget.target_bytes?'已带入文件预算；完成分析后会限制到可规划区间。':'硬件预算超出当前预设区间，已限制到区间内；可重新选择预设分析。');
});
$('#targetSlider').addEventListener('input',e=>{invalidatePlan();setTarget(Number(e.target.value));});
$('#targetBytes').addEventListener('input',e=> {
  invalidatePlan();
  const bytes=Number(e.target.value);
  $('#targetLabel').textContent=giB(bytes); $('#targetSlider').value=e.target.value;
  if(Number.isFinite(bytes)&&bytes>0)$('#desiredGiB').value=String(bytes/GIB);
});
$('#loadAnalysis').addEventListener('click',()=>loadAnalysis().catch(e=>error(e.message)));
$('#qualityForm').addEventListener('invalid',event=> {
  const references=event.target.closest('details');
  if(references)references.open=true;
},true);
document.addEventListener('click',async event=> {
  const button=event.target.closest('[data-copy-path]');
  if(!button)return;
  try {await navigator.clipboard.writeText(button.dataset.copyPath);toast('已复制产物路径。');}
  catch {error('无法访问剪贴板，请直接选择并复制显示的路径。');}
});
window.addEventListener('pywebviewready',()=>document.documentElement.classList.add('native'));
if(window.pywebview?.api)document.documentElement.classList.add('native');
document.addEventListener('click',async event=> {
  const button=event.target.closest('[data-open-job]');
  if(!button)return;
  try {await window.pywebview.api.open_output(button.dataset.openJob);}
  catch(e){error(e.message||'无法打开产物目录，请复制路径后手动打开。');}
});
for(const [selector,action] of [['#analyzeForm','analyze'],['#planForm','plan'],['#quantizeForm','quantize'],['#qualityForm','quality']]) $(selector).addEventListener('submit',event=>{ event.preventDefault(); submit(event.target,action).catch(e=>error(e.message)); });
$('#modelsForm').addEventListener('submit',event=> { event.preventDefault(); loadModels().catch(e=> {error(e.message);$('#modelsNote').textContent='读取失败，可重试或检查 llmfit 安装。';}); });
$('#modelSearch').addEventListener('input',()=>{state.modelLimit=40;renderModels();});
$('#modelRows').addEventListener('click',event=>{if(event.target.id==='moreModels'){state.modelLimit+=40;renderModels();}});
$('#jobsList').addEventListener('click',event=> { const button=event.target.closest('[data-job]'); if(button) {const row=state.jobs.find(j=>j.id===button.dataset.job);go(row?.action==='quality'?'quality':'workspace'); watchJob(button.dataset.job).then(async job=> {await restoreJob(job);if(['running','queued','cancelling'].includes(job.status))state.active=job.id;}).catch(e=>error(e.message));} });
$('#cancelJob').addEventListener('click',()=> { if(state.active) api(`/api/jobs/${state.active}/cancel`,{}).then(()=>toast('已请求取消任务。')).catch(e=>error(e.message)); });
$('#cancelQuality').addEventListener('click',()=> { if(state.active) api(`/api/jobs/${state.active}/cancel`,{}).then(()=>toast('已请求取消质量搜索。')).catch(e=>error(e.message)); });
document.querySelectorAll('.pick').forEach(button=>button.addEventListener('click',async()=> {
  if(!window.pywebview?.api) {toast('浏览器版请粘贴完整路径；桌面版支持文件选择。');return;}
  try {
    const path=await window.pywebview.api.pick_path(button.dataset.kind);
    if(path) {
      const input=button.parentElement.querySelector('input');
      input.value=path; input.dispatchEvent(new Event('change',{bubbles:true}));
    }
  } catch(e) {error(e.message);}
}));
async function init() {
  const info=await api('/api/info'); $('#workspacePath').textContent=info.workspace;
  $('.version').textContent=`FIT-GGUF ${info.version} · LOCAL STUDIO`;
  $('#modelLimit').textContent=`源模型上限 ${info.max_model_params}B`;
  $('#maxParams').max=info.max_model_params; $('#maxParams').value=Math.min(5,info.max_model_params);
  for(const selector of ['#lowerPreset','#upperPreset']) $(selector).innerHTML=info.presets.map(p=>`<option>${esc(p)}</option>`).join('');
  $('#lowerPreset').value='IQ3_M'; $('#upperPreset').value='IQ4_XS';
  let draft=null;
  try {draft=(await api('/api/draft')).draft;}
  catch {$('#draftStatus').textContent='草稿读取失败 · 任务记录仍可使用';}
  if(draft) {
    applyDraft(draft);
    if($('#analysisPath').value) {
      try {const data=await api('/api/analysis',{path:$('#analysisPath').value});acceptAnalysis(data.path,data.result);}
      catch {error('草稿中的分析记录已不可用，请重新分析或载入有效的 analysis.json。');}
      applyDraft(draft); // Keep the user's exact budget instead of clamping it during restore.
      if(state.analysis)$('#targetSlider').value=$('#targetBytes').value;
    }
    $('#draftStatus').textContent='已恢复上次草稿 · 结果可从任务记录查看';
  }
  refreshHardware().catch(e=>error(e.message));
  await refreshJobs();
  const completed=state.jobs.filter(job=>job.status==='succeeded');
  for(const last of (draft?[]:[completed.find(job=>job.action!=='quality'),completed.find(job=>job.action==='quality')].filter(Boolean))) {
    const job=await api(`/api/jobs/${last.id}`);
    await restoreJob(job,false);
    $(job.action==='quality'?'#qualityLog':'#liveLog').textContent=job.log_text||'已恢复上次完成的任务。';
  }
  state.lastResult=completed[0]?.id||null;
  if(state.active) await watchJob(state.active);
  draftReady=true;
  updateTierHelp();window.FitI18n.setLanguage($('#language').value);
  go('start');if(draft)sizeStep('analyzeForm');
}
setInterval(async()=> { if(state.polling||!state.active)return; state.polling=true;try{await watchJob(state.active);await refreshJobs();}catch(e){error(e.message);}finally{state.polling=false;} },1500);
init().catch(e=>error(e.message));

