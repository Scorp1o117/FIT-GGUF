# Planner Verdict — G2 closure + Gemma-4-E4B substitution (GPT, 2026-09-02 晚)

来源：GPT 对 G2 落地汇报的完整回复（签字版）。上接 planner-verdict-m2.md。

## 1. G2 exact-size repair：正式验收 PASS ✅

GPT 状态表：
```
G2 Exact Size ✅ PASS
Tensor-info trailing-dim normalization ✅
Dynamic imatrix metadata finalization ✅
Effective recipe re-resolution ✅
Post-quant exact-byte gate ✅
BailingMoE 6-plan reproduction ✅ 6/6 exact
Q5_0 / Q5_1 byte-model coverage ✅
BailingMoE exact-size support ✅ VALIDATED
G2 qtype coverage issue: CLOSED ✅
```

- 架构收益升级为 v0.2 正式定义：区分 **plan-time estimate** 与
  **invocation-finalized exact prediction**；CLI 明确输出
  `Plan target / Finalized prediction / Actual GGUF / Delta / G2: PASS`（已实现）。
- "重终结门"被确认为正确设计：quantize.imatrix.file 属 invocation-dependent
  serialized metadata，不应被 analyze 阶段永久冻结。

## 2. 第三 dense 家族换人：APPROVED ✅

```
Mistral-7B-v0.3 → (owner decision) → google/gemma-4-E4B
```
- dense 三族齐：Qwen3.8-27B（Qwen 系）/ Granite-4.2-8B（IBM 系）/ gemma-4-E4B（Gemma 系）。
- 若 E4B 仍支持 93/90/88/85，dense-v1 说服力强于旧 Mistral。
- **执行前硬门（ruling）**：正式 M2 采样前必须产出
  **M2-Gemma Text-Tower Eligibility Record**：main GGUF inventory 中
  vision tensor count = 0、audio tensor count = 0、text tensor count = expected、
  mmproj externalized/absent。**只要主 GGUF 混入视觉/音频权重即暂停 M2**——
  否则 Size→Fidelity 关系被不参与 forward 的张量污染。
  → 已执行：PASS（text=720 / vision=0 / audio=0，mmproj 不存在），
  记录：results-gemma/text-tower-eligibility.json。
- MatFormer/effective-size 不是 blocker，但报告须写全：
  `Family: Gemma 4 / Architecture class: dense text tower / Model scale: ~8B stored / E4B effective configuration`。

## 3. G2 fixture 设计：主体 PASS ✅ + 三补（已全部实现）

GPT 建议（非阻塞）→ 状态：
- A. 127B 截断边界（126/127/128/>127，mid-codepoint 场景）→ `test_kv_string_truncation_boundary_bytes`；
  顺带修复真实边界 bug：`_truncate_kv_string` 由 decode-ignore 改 surrogateescape（否则多字节切断
  时少算 1-3 字节，exact-size 会破），`_encoded_field_size` 同步 surrogatepass 字节语义。
- B. UTF-8 字节长度（字符数≠字节数）→ `test_utf8_path_byte_length_not_char_length`。
- C. 尾维归一规则单测（(4,1,2048,1)→3、(4,1,2048)→3、(4,2048)→2、最低维语义）→
  `test_trailing_singleton_normalization_rule`。
- 审计链（ruling）：manifest 永远保留 raw 证据 + 解析产物哈希
  `oracle_log_sha256 / parsed_recipe_sha256 / actual_gguf_sha256 / expected_actual_bytes`，
  证据链 oracle log → parser → recipe → predictor → actual artifact 可独立复核
  → fixture JSON 已含四类字段。
- 套件：95 passed。

## 4. 队列（GPT 批准版）

```
G2 Exact Size ── ✅ PASS / CLOSED
M2 Dense Calibration
  ├── Qwen3.8-27B ✅
  ├── Granite-4.2-8B ✅
  └── Gemma-4-E4B 🚧 NEXT   └── text-tower eligibility gate first ✅ PASS
M2 BailingMoE ── ✅ candidate calibration complete
M3 Fidelity Contract v1 ── ⏳ blocked by Gemma dense calibration
M10 Ornith ── 🔒 SEALED
```

- 终局交付（orcarouter Qwen3.8-27B-Uncensored 四档 × 预设/FIT v0.1/FIT v0.2）放
  **M3 之后**，让 FIT v0.2 列真正使用冻结后的 eval-v1 + Contract v1 + Refine。
- GPT 预告届时给两个主表：**Quality @ Fixed Size** 与 **Size @ Fixed Fidelity**，
  外加一张"四档三方对决"总表作为 v0.2 README 核心图。

最终签字：1) G2 PASS ✅ 2) Gemma-4-E4B substitution APPROVED ✅
3) fixture design PASS ✅ + 3 non-blocking boundary fixtures（已补）
「可以进入 Gemma M2 了。」
