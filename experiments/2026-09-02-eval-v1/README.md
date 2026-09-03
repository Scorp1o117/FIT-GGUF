# eval-v1 — FIT Evaluator Contract v1（规范 + 参考实现）

日期：2026-09-02。状态：**rc1，待真实 runtime repeat 烟测后终冻结**。
- `rc0` digest `7d987e47989d8bb6fe38a9a298998b21ca4f7aa2dc7c846f10777947abde4783`（GPT 评审前候选，保留作审计记录）
- **`rc1` digest `ddac5c3ae47d687b7d4b46d0ecb94f44b008918c2f553cfcd9ce3b2dfce68c6c`**（四项 blocker 修复后）
契约冻结体：`evaluator-contract-eval-v1.json`（canonical JSON）。

## rc0 → rc1 修订（响应 GPT 评审四 blocker + 三偏差意见）

1. **Blocker A（切分表述）**：实测确认切片为 65,536 **字节**（chinese 域
   65,534 字节 = 61,792 码点）；契约改为"fixed 65,536-BYTE UTF-8 file
   slices (historical label: M9 64k)，corpus files authoritative by
   sha256，no reslicing"，码点/字节数下放到 reference manifest 记录。
2. **Blocker B（runtime 溯源分层）**：契约拆成 `semantics_pin`
   （llama.cpp source revision b10666 + 算法路径 + flags）与
   `execution_provenance_required`（binary sha256/编译信息/ROCm/GPU，
   属执行 manifest，不进语义身份——换平台不被迫升 eval-v2）。
3. **Blocker C（参照闭环）**：新增 `reference_manifest` 定义（六钉：
   BF16 GGUF sha / tokenizer hash / corpus domain sha / reference .kld
   sha / contract hash / runtime provenance + expected_valid_tokens）；
   结果文档增加 `reference_manifest_hash`。
4. **Blocker D（regressed_domains 语义）**：基础 evaluator 不再输出
   regression 判定（KL≥0 时无意义），改为 `comparison: null`；新增
   comparison 层 `compare_results(baseline, candidate, hash)`，仅当
   显式提供 baseline 结果时计算。
5. **偏差①补充**：canonicalization 算法本身写进契约（utf-8、字典序
   键、separators、ensure_ascii=false、无尾换行、implementation 字串）。
6. **偏差③补充（源码考古）**：从 b10666 tag 源码逐行核对后钉死三条
   语义：KL 求和含 **−16 nat 参照截断**（`p_log_base > -16.f`）；参照
   .kld 是 **f16 仿射压缩 log-prob**（`scale·f16+min`），非原始 logits；
   Δp 精确公式 = 目标 token 概率差 `p_quant(τ) − p_ref(τ)`，RMS ×100
   显示。PPL 措辞改为"ground-truth next-token targets of the frozen
   corpus"。内核 `kl_divergence` 已实现 −16 截断并有专门测试（截断位
   候选 −inf → 有限；截断上候选 −inf → +inf）。
7. **非阻塞采纳**：结果文档独立 schema id `fit.eval_result.v1`；
   expected_valid_tokens 进 reference manifest（重生成后钉）。

## 0. 定位

eval-v1 是 v0.2.1 §9 的统一评测协议：Fidelity Contract 的每一个数字只在
"eval-v1 之下"才有意义（`fit.fidelity.evaluator_contract = "eval-v1"`）。
本目录同时交付参考实现（`src/fit_gguf/eval/`），其指标内核是纯 Python、
零依赖、由手算闭式夹具钉死——改代码弯不了尺子。

## 1. 与 GPT 六项交付清单的对应

| 项 | 交付 |
|---|---|
| A. contract 全冻结 | `contract.py` 的 `EVAL_V1`（runtime pin b10666、`-ngl 99 -t 16 -c 512 -b 512 --kl-divergence --kl-divergence-base`、五域 64KiB 切片 + SHA-256（M9 集永不重切）、等权 0.2、KL 方向、数值策略）；canonical JSON + `contract_hash()` |
| B. same-top 边角规则 | 并列取最低词表下标；EOS 计入不 mask；raw-text 无 padding；词表不匹配 = fatal 拒评（绝不静默重映射）；NaN/Inf 位置剔除并计数，域剔除率 > 0.1% 判 FAIL |
| C. 两级聚合 | 域内 token 算术均值 → 五域等权 macro（**contract primary**）；token 池化 micro 仅诊断输出，禁用于合同判定（测试里有 macro≠micro 的反例钉住） |
| D. ref logits 一等 artifact | 契约冻结参照定义（BF16 同源、位置对齐 teacher-forced、.kld base 文件）；每个结果文档强制携带 `model_hash / reference_logits_hash / candidate_plan_hash / evaluator_contract_hash` |
| E. determinism + synthetic fixture | `synthetic.py`：vocab=4、ln 整数比 logits，KL/same-top/NLL 全部有手算闭式（不经过被测代码路径）；方向反转有独立闭式（D_KL(p‖q) ≠ D_KL(q‖p) 被测试钉住）；结果序列化字节级确定 |
| F. 输出 schema | `results.build_eval_result`：`evaluator_contract(+hash) / model_hash / reference_logits_hash / candidate_plan_hash / total_tokens / macro_kl / macro_same_top / micro_kl / worst_domain(+kl) / regressed_domains / domains{tokens,kl,same_top,ppl,rms,excluded}` |

## 2. 语义继承声明（不另起炉灶）

eval-v1 的数值语义 = FIT v0.1 已发布 14 档曲线所用的
`llama-perplexity --kl-divergence --kl-divergence-base` 协议（M9 预注册，
b10666 运行时）：

- KL 方向：**D_KL(P_ref ‖ P_quant)**（llama.cpp 同款：参照在前），
  全词表 softmax、温度 1、逐 teacher-forced 位置；
- same-top：与工具日志中的 `Same top p` 同义；
- 日志解析：`results.parse_llama_kl_log` 以真实 P4 日志校验
  （FIT-12G wiki_test → mean KLD 0.052627 / same-top 90.968%）。

## 3. 与清单的已知偏差（需评审确认）

1. **canonical JSON 而非 YAML**：仓库零依赖约束（pyproject `dependencies=[]`）；
   内容等价，哈希取自 canonical 形式。
2. **参照 .kld 哈希未进契约**：参照文件按（模型×域）成对存在，属结果级
   provenance（`reference_logits_hash`），契约只冻结其生成定义；BF16 .kld
   重生成时必须留 SHA-256 记录。
3. **PPL/RMS 定义**：PPL = exp(域内平均 token NLL)（候选招生自身目标），
   Δp RMS 以百分比报告——与 llama.cpp 输出口径一致，数值仅供诊断。

## 4. 测试与验证状态

`tests/test_eval_v1.py` 11 用例全过；全仓 85 测试无回归。关键钉子：

- KL 闭式：`(8/15)ln(22/15) + (7/15)ln(11/15)`（position A）与 0（B）；
- 方向反转独立闭式；
- tie → 最低下标；vocab mismatch fatal；NaN 剔除 + 0.1% 预算边界
  （恰 0.1% 过、0.5% 拒）；
- macro（0.255）≠ micro（被大域拉低）反例；
- 契约哈希确定性；权重和 ≠1 即拒；结果序列化字节级一致；
- 真实日志解析抽查。

## 5. 冻结流程下一步（§9 余款）

1. 策划侧评审本契约（digest `7d987e47…` 为锚）；
2. BF16 参照 .kld 五域重生成 + SHA-256 存档（b10666 pin 下）；
3. `Top|KL boundary` 分布在 DEV 模型上采样（Candidate Contract
   93/91/88/85 的 P5/P10 校准输入）；
4. Fidelity Contract v1 冻结（contract + evaluator_contract 双哈希入 metadata）。
