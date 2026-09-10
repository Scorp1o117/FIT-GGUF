# GPT 裁定 — G11 Sealed Protocol（2026-09-06）

来源：ChatGPT 会话 6a966e2d（思考 3m27s，含 web 核实），ZCode 归档。
状态：**G11 sealed protocol：CONDITIONAL APPROVED ✅**；具体 sealed family **暂不锁定**。

## 裁定表（原文要点）

| 问题 | 裁定 |
|---|---|
| Falcon3-3B sealed | 🟡 技术批准但不建议首选；**不是 Apache-2.0**（TII Falcon-LLM License 2.0） |
| 首选 family | **OLMo-2-1124-7B Base**（allenai/OLMo-2-1124-7B） |
| 备选 | Falcon3-3B-Base → SmolLM2-1.7B |
| Eligibility preflight | **必须增加** |
| candidate 算有效 sealed record | ✅ |
| candidate 算 G11 PASS | ❌ |
| budget_exhausted 算 G11 PASS | ❌ |
| candidate 后强跑 Search | **禁止**（resolver fail-closed，不得 require_eval_provenance=False 绕过） |
| infra-only resume | ✅ |
| metric-informed rerun | ❌ |
| 执行顺序 | Spark Gate B → Qwen3-4B DEV → 修问题 → **P2/A2/tests FROZEN** → G11 eligibility preflight → final prereg + SOURCE_LOCK → one-shot |

## 六条新增协议条款

- **A. Eligibility Preflight**（seal 前、不消耗 one-shot）：允许查 HF revision/converter/BF16 load/tokenizer/imatrix 生成与 coverage/单 preset quantize/llama-perplexity/reference 读写/G2 dry-run；**禁止**五域 eval-v1 正式指标、Top|KL curve、Guard floor、tier window、gap probe、Fidelity Search。工具链 smoke 用非 eval-v1 corpus。一旦看到任何正式五域质量指标 = SEALED RUN STARTED，不得换模型。动机：OLMo-2 有 llama.cpp imatrix NaN 历史 bug（ggml-org/llama.cpp#11764），须在 eligibility 阶段判 fail 换 Falcon，不算 FIT sealed failure。
- **B. SOURCE_LOCK 两阶段钉定**：G11-PREREG（HF repo+revision、converter commit+args、runtime、contracts、execution config）→ eligibility + 确定性转换 → SOURCE_LOCK.json（safetensors hashes、tokenizer SHA、BF16 GGUF SHA、imatrix corpus SHA、eval corpus SHA）→ 之后才 calibrate。SOURCE_LOCK 写入后不得重转另一份 BF16 作同一 G11 source。
- **C. 有效记录 ≠ PASS**：G11 PASS 至少 = fit calibrate 机械完成 AND Guard=validated AND provenance 完整 AND registry candidate entry verifies，然后才进 Search。Search 可接受终态：verified_pass；no_pass/NOT_REACHABLE（健康前沿被完整关闭也可）。FAIL（但 sealed result 有效）：Guard=candidate、INSUFFICIENT_WINDOW、异常、provenance/G2 failure、budget_exhausted。sealed result validity ≠ release gate verdict。
- **D. candidate Guard 分支**：Guard=validated → 四档 Search；Guard=candidate → 产品按合同输出 CALIBRATION_NOT_VALIDATED 硬拒绝 → G11 sealed FAIL，不得 override。
- **E. 基础设施恢复定义**：允许 hardware reboot/process crash/disk I/O 中断后用同一 config 从最后 SHA 验证的 checkpoint 恢复；禁止看结果后改 target/preset/probe budget/floor/runtime semantics/imatrix/BF16。"Infrastructure retry is allowed; metric-informed adaptation is not."
- **F. sealed failure 用于修代码 → family 永久 DEV + G11 FAIL**；要再次 sealed PASS 必须换全新 unseen family + 新 prereg。

## 最终 prereg 钉定清单

preregistration_id、selected_family、HF repo+revision SHA、converter commit+args、source safetensors hashes、BF16 GGUF SHA、tokenizer SHA、FIT code commit、A3 registry schema/digest、fidelity-calibration-v1 SHA、eval-v1 SHA、imatrix corpus SHA+config、五 eval corpus SHAs、llama.cpp source revision、binary SHA、compiler/build/backend/GPU、execution profile、standard preset ladder、poison set、probe budget、Fidelity Search profile、search fresh-eval budget、search tolerance、允许的 infra recovery、禁止的 adaptations、PASS criteria、FAIL criteria、post-run DEV transition。

## 当前文档状态

本目录 PREREGISTRATION.md 更新为 **G11-PROTOCOL-FREEZE-CANDIDATE v1.2**（family 未锁，协议六条款并入）。最终 prereg SHA 在 P4 完成且 P2/A2 冻结后才落。
