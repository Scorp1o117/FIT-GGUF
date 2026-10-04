// Run with node --test tests/studio_i18n.test.cjs (no browser dependency).
const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const path=require('node:path');
function runtime(skip=false) {
  const text={textContent:'指定体积',parentElement:{closest:()=>skip?'pre':null}};
  const body={querySelectorAll:()=>[]};
  const context={window:{},document:{body,documentElement:{lang:'zh-CN'},
    createTreeWalker:()=>{let used=false;return {nextNode(){if(used)return false;used=true;this.currentNode=text;return true;}};}},
    NodeFilter:{SHOW_TEXT:4},MutationObserver:class {observe(){}},queueMicrotask};
  vm.createContext(context);
  vm.runInContext(fs.readFileSync(path.join(__dirname,'../src/fit_gguf/studio/static/i18n.js'),'utf8'),context);
  return {context,text,i18n:context.window.FitI18n};
}
test('switching back restores the original DOM text without translation drift',()=> {
  const {i18n,text,context}=runtime();
  for(let i=0;i<10;i++) {
    i18n.setLanguage('en');assert.equal(text.textContent,'Size target');assert.equal(context.document.documentElement.lang,'en');
    i18n.setLanguage('zh-CN');assert.equal(text.textContent,'指定体积');
  }
});
test('new dynamic text replaces the cached source and placeholders remain intact',()=> {
  const {i18n,text}=runtime();i18n.setLanguage('en');
  text.textContent='目标预算 0.023456789 GiB';i18n.setLanguage('en');
  assert.equal(text.textContent,'Target budget 0.023456789 GiB');
  i18n.setLanguage('zh-CN');assert.equal(text.textContent,'目标预算 0.023456789 GiB');
});
test('core workflow, states and longer matching phrases are translated',()=> {
  const {i18n}=runtime();i18n.setLanguage('en');
  assert.equal(i18n.translate('指定质量档位'),'Quality target');
  assert.equal(i18n.translate('正在取消 / 已取消'),'Cancelling / Cancelled');
  assert.equal(i18n.translate('质量档位搜索'),'Quality tier search');
  assert.equal(i18n.translate('llama.cpp 运行时目录'),'llama.cpp runtime directory');
});
test('raw evidence text is excluded from localization',()=> {
  const {i18n,text}=runtime(true);text.textContent='D:\\模型\\指定体积.gguf';
  i18n.setLanguage('en');assert.equal(text.textContent,'D:\\模型\\指定体积.gguf');
});
