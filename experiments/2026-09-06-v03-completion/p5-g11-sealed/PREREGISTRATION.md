# P5 — G11 Independent-Family Sealed Validation：预注册 v1.2（PROTOCOL-FREEZE-CANDIDATE）

日期：2026-09-06 · 状态：**协议条款冻结候选——family 暂不锁定**（GPT 裁定 planner-verdict-g11-protocol.md）
上位：planner-verdict-v03-preview.md §九 + planner-verdict-g11-protocol.md（六条款有条件批准）
执行顺序（GPT 批准）：Spark Gate B → Qwen3-4B DEV → 修问题 → **P2/A2/tests FROZEN** → G11 eligibility preflight → final prereg + SOURCE_LOCK → one-shot。

## 1. 目的

用一个**从未进入 development loop 的独立家族**，一次性验证 v0.3 生产线的泛化性：陌生模型 → `fit calibrate` → Guard → 四档 Fidelity Search，全程可审计、无手工干预。G11 验证的是「冻结的产品能否在没拿该模型调过自己的前提下完成 onboarding」，不是「跑个没见过的模型看看」。

## 2. 禁选清单（GPT 裁定，全部排除）

Qwen（含 Qwen3-4B，P4 用）、Granite、gemma、BailingMoE/Ling、Spark、Ornith、PRISM、DeepSeek、OrcaRouter。

## 3. Family 优先级（GPT v1.2 裁定，预检后锁定）

1. **首选：allenai/OLMo-2-1124-7B Base** —— Apache-2.0、独立 lineage、llama.cpp 明确支持；有 imatrix NaN 历史 bug（llama.cpp#11764），故必须先过 eligibility preflight，工具链不兼容在「未进入 sealed evaluation」阶段判 fail 换备选，不算 FIT sealed failure。
2. **备选 1：tiiuae/Falcon3-3B-Base** —— 技术批准；**许可为 TII Falcon-LLM License 2.0（非 Apache-2.0）**，v1.1 草案理由作废。
3. **备选 2：HuggingFaceTB/SmolLM2-1.7B** —— Apache-2.0；1.7B 太小，低 bit cliff 概率高，信息量较低。

锁定时点：P4 完成 + P2/A2 冻结后，对首选做 preflight，通过即锁 OLMo，失败按序转 Falcon3。

## 4. 条款 A — Eligibility Preflight（不消耗 one-shot）

允许检查：HF revision 可取 → converter 可转换 → BF16 可 load → tokenizer 正常 → imatrix 能生成 → coverage 可解析 → 单个 preset 能 quantize → llama-perplexity 能执行 → reference write/read → G2 dry-run/predict。
**禁止**：五域 eval-v1 正式指标、Top|KL curve、Guard floor、tier window、gap probe、Fidelity Search。工具链 smoke 用**非 eval-v1 corpus**。
**一旦看到任何正式五域质量指标 = SEALED RUN STARTED**，此后不得换模型。

## 5. 条款 B — SOURCE_LOCK 两阶段钉定

1. G11-PREREG：HF repo + revision、converter commit + args、runtime、contracts、execution config。
2. eligibility + 确定性转换 → **SOURCE_LOCK.json**：safetensors hashes、tokenizer SHA、BF16 GGUF SHA、imatrix corpus SHA、eval corpus SHA。
3. SOURCE_LOCK 写入后不得重转另一份 BF16 作同一 G11 source；之后才正式 calibrate。

## 6. 条款 C — 有效记录 ≠ PASS

- **G11 PASS** 至少满足：`fit calibrate` 机械完成 AND Guard=validated AND provenance 完整 AND registry candidate entry verifies → 然后才进 Search。
- Search 可接受终态：verified_pass；no_pass / NOT_REACHABLE（健康前沿被完整关闭）。
- **G11 FAIL（但 sealed result 仍有效）**：Guard=candidate、INSUFFICIENT_WINDOW、异常、provenance/G2 failure、**budget_exhausted**。
- sealed result validity ≠ release gate verdict。

## 7. 条款 D — candidate Guard 分支

Guard=validated → 四档 Fidelity Search；Guard=candidate → 产品按 resolver fail-closed 硬拒绝（CALIBRATION_NOT_VALIDATED）→ G11 sealed FAIL。**禁止** require_eval_provenance=False 或任何 override。

## 8. 条款 E — 基础设施恢复

允许：hardware reboot / process crash / disk I/O 中断后，用同一 config 从最后 SHA 验证的 checkpoint 恢复。
禁止：看结果后改 target / preset / probe budget / floor / runtime semantics / imatrix / BF16。
**Infrastructure retry is allowed; metric-informed adaptation is not.**

## 9. 条款 F — 失败即永久 DEV

G11 run 发现 bug 且结果用于修改 FIT → 该 family 永久入 DEV（sealed_eligible=false）+ G11 FAIL。再次 sealed PASS 必须换全新 unseen family + 新 prereg。

## 10. 最终 prereg 钉定清单（执行前哈希）

preregistration_id、selected_family、HF repo + revision SHA、converter commit + args、source safetensors hashes、BF16 GGUF SHA、tokenizer SHA、FIT code commit、A3 registry schema/digest、fidelity-calibration-v1 SHA（455338c5…）、eval-v1 SHA（5ce78dee…）、imatrix corpus SHA + config、五 eval corpus SHAs、llama.cpp source revision、binary SHA、compiler/build/backend/GPU、execution profile、standard preset ladder、poison set、probe budget、Fidelity Search profile（fresh-eval budget、tolerance）、允许的 infra recovery、禁止的 adaptations、PASS criteria、FAIL criteria、post-run DEV transition。

## 11. one-shot 纪律

下载 → 标准转换 BF16（SOURCE_LOCK）→ **单次** `fit calibrate`（标准梯 + ≤4 探针/档）→ 单次四档 Fidelity Search（seeds=校准观测）。run 后该 family 入 DEV。

## 12. 执行记录

（批准后填写）
