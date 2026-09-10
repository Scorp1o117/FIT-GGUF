# P1 — Calibration Contract v1（fidelity-calibration-v1）预注册 v1.1

日期：2026-09-06 · 作者：ZCode（GLM-5.3-Flash）执笔 · 状态：**GPT DESIGN APPROVED / FREEZE CANDIDATE——本 v1.1 已并入全部 4 项 normative 修订与 §13 裁定，待 contract replay 后签 FROZEN**
裁定：results/planner-verdict-p1.md（2026-09-06）。上位：A3 FROZEN（fit.fidelity_registry.v1）。
**机读合同**：`src/fit_gguf/contracts/fidelity-calibration-v1.json`（A3 canonical JSON v1；其 canonical SHA = `calibration_contract_sha256`，见 §14）。本 MD 为 normative design rationale / preregistration archive，头部记录 machine_contract_sha256。

本稿回答："什么样的 calibration evidence，有资格生成一个 Registry 可以接纳的 validated exact-model Guard？"

## 0. 合同身份与范围

- 合同 ID：`fidelity-calibration-v1`；canonical SHA 见机读合同。
- **管**：Calibration Bundle 生产规则——参照生成、预设阶梯、窗口与补点、地板推导、witness、验证状态与晋升、记录与打包、失败状态枚举。
- **不管**：Fidelity Search 内部（v1 队列）、fit calibrate 工程实现（P2）、family/arch promotion（research-only）、eval-v1 语义（永久冻结，只消费）。
- **祖辈条款**：`legacy_bootstrap_v0` admission 永久关闭（仅 orcarouter/Spark）。

## 1. 固定输入（SHA 钉定，漂移即硬失败）

| 输入 | 钉定 |
|---|---|
| BF16 源 GGUF | source_weights_sha256 |
| imatrix 语料 | 文件 SHA + chunk 规格（默认 APEX-imatrix-Small.txt 500×c512） |
| eval-data 五域切片 | corpus_sha256 == eval-v1 冻结值（漂移 = CORPUS_DRIFT） |
| 运行时 | 构建来源 + llama-perplexity binary sha 前缀 + backend/GPU；非钉定构建须附 perplexity 源码与 eval-v1 钉定构建逐字节一致 diff 证据，否则 REF_RUNTIME_UNVERIFIED |
| eval-v1 digest | live contract_digest()（rc2 5ce78dee…） |

## 2. 参照生成（reference bundle）

1. 五域 `.kld` 由 BF16 走 **write path**（`--kl-divergence-base` only），协议 `-ngl 99 -t 16 -c 512 -b 512`（-ngl/-t 属 Execution Profile，§9）。
2. 每参照过 **ref_ok** 行数完整算术校验（防 ENOSPC 静默截断）。
3. **reference-manifest**（`fit.eval_reference_manifest.v1`）钉：evaluator_contract_hash、source_bf16_gguf_sha256、tokenizer、每域 corpus_sha256 + reference_kld_sha256 + raw_bytes + unicode_codepoints、**expected_valid_tokens（authoritative/normative——evaluator/参照本体导出，参与 alignment 验证）**、`token_count_estimate`（**optional informational**：{value, method}，如 tokenizers 法；绝不与 expected 混用）、runtime_provenance。
4. 大 `.kld` 为本地/transport 资产；canonical manifest 原件入实验记录，逐字节副本进 `registry/manifests/`。

## 3. 预设阶梯（preset ladder）

- **标准梯（12 点，写死）**：IQ2_XXS IQ2_XS IQ2_M IQ3_XXS IQ3_XS IQ3_M IQ4_XS Q3_K_M Q4_K_M Q5_K_M Q6_K Q8_0；允许 --extra-presets 增补（记录理由，不得跳过标准梯）。
- 每点 = 原始 llama-quantize --imatrix + 五域 eval；记录 name/size_bytes/artifact_sha256/per-domain KL·top/macro。
- **毒 set（v1 写死）**：{Q3_K_S, IQ2_XS}——永不入窗、永不作种子。修改毒 set = contract semantics change → 新 digest（amendment）或 v2，禁止同 SHA 静默追加。
- **imatrix 覆盖检查强制**：entries ↔ 源张量差集；norm/embd 豁免；可量化矩阵缺条目 → 该点 IMATRIX_PARTIAL，低 bit 类型命中缺失张量必须 --tensor-type-file 提升并记录，否则该点 NOT VALIDATED。
- 产出 **curve-points.jsonl**。

## 4. 档位窗口（boundary locality，修订 A）

**W(K) = [0.85K, 1.15K]**（macro KL，双端闭）：

| tier | K | 窗口 |
|---|---|---|
| quality | 0.05 | [0.0425, 0.0575] |
| balanced | 0.10 | [0.0850, 0.1150] |
| compact | 0.15 | [0.1275, 0.1725] |
| mini | 0.20 | [0.1700, 0.2300] |

- 这是 **boundary calibration**：远离锚点的点（如 KL 0.007）不得计入 quality 样本，防止 floor 被高质量远点抬高。
- 窗内 n<3 → **gap probes 补点**，不拿远处预设凑数。
- 补点方式：普通 fit plan（无 tier 门）+ balanced 策略，目标尺寸落在窗内；**预算 ≤4 fresh probe/tier**（只计 artifact 成功产出且 eval-v1 实际执行的；cache hit/重试不计）；耗尽仍不足 → INSUFFICIENT_WINDOW → candidate，禁止自动加预算。
- 窗口几何的 fit-analyze（预设对）属 P2/search 资产，非本合同强制。

## 5. Same-top 地板推导（精确定义）

5.1 **样本值**：直接取 eval-v1 result 的 `macro_same_top` **存储全精度值**——Calibration 不重新实现 aggregation（eval-v1 result is authority；禁止对已 round 的显示值再平均）。
5.2 **截断规则**：`trunc4(v) = Decimal(canonical_decimal_string(v)).quantize(Decimal("0.0001"), rounding=ROUND_DOWN)`——Decimal 语义，**禁止裸 binary-float `floor(v*10000)`**（ε 会破坏 canonical replay）；same-top 非负，ROUND_DOWN 即向下。边界 fixture 必备：0.9350 / 0.9349999… / 0.8503 / 0.8894。
5.3 **P5 定义**：线性插值分位数，`k = (n-1) × 0.05`，相邻次序统计量内插。
5.4 **推导规则（冻结）**：
- `n ≥ 3`：`floor = trunc4(P5(samples))`。
- `n ≤ 2`：`floor = trunc4(min(samples))`，但该档 `validation_status = candidate`。
- 地板只写 Guard Profile YAML；registry entry 永不复制 floors。
- **样本冻结时点**：校准期观测（阶梯 + 空洞探针）；Search 观测**永不静默回填**；重校 = 新 calibration revision（新 record → 新 guard hash → 更新 registry entry），旧记录保留。
5.5 **Unique observations（修订 D）**：`sample_count` 只计**独立校准观测**，dedup key = `artifact_sha256`——同 artifact 重复评测 n+=0；同尺寸不同配方 = 独立观测（保留 plan_hash/recipe_hash 审计）。
5.6 **Calibration witness（裁定 1）**：每档除 n≥3 外，必须存在 ≥1 个校准观测满足 `macro_kl ≤ K ∧ same_top ≥ floor`（字段 witness: {point_id, macro_kl, macro_same_top, pass: true}）；缺失 → 该档 candidate。语义：证明合同没有把自己的校准证据全部挡在门外。

## 6. 验证状态与晋升（candidate → validated）

- **validated（整体）**：四档全部 `sample_count ≥ 3` **且** 每档 witness 存在，加 guard YAML 合法、calibration record 绑齐 §8、无未关闭失败状态。
- **candidate**：任一档 n<3 或 witness 缺失或带未关闭失败状态。可入 registry（可审计），resolver 拒出正式 tier。
- **晋升 = 重新校准**（补点后重跑推导），无运行中转正。
- 逐档字段：`floor_method: empirical_p5 | min_fallback`、`sample_count`、`witness`、`validation_status`；profile 级 `confidence: high | low`。
- v1 不引入 per-tier registry status / validated_partial（不重开 A3；未来需要 → Registry v2 / Guard schema v2）。

## 7. 数值精度与序列化（冻结）

- 地板 4 位小数 Decimal ROUND_DOWN（§5.2）；Guard YAML `safe_dump(sort_keys=False)`；profile_hash = `fidelity.profile_hash`。
- registry entry/索引：A3 canonical JSON v1。
- curve-points/record 存双精度全值；显示精度 ≠ 存储精度。

## 8. Calibration record（canonical audit root）

`fit.calibration_record.v1` 绑齐：
1. 输入区：source_sha、imatrix sha+规格、五域 corpus_sha、runtime provenance、eval-v1 digest、calibration_contract_sha256；
2. 过程区：curve-points 摘要（含 artifact_sha256/plan_hash/recipe_hash）、空洞探针 plan 记录、imatrix 覆盖检查结果；
3. 推导区：逐档窗口样本明细、dedup 说明、floor_method、trunc4 地板、sample_count、witness、validation_status；
4. 产物区：guard sha、reference manifest sha、bundle 文件清单 + SHA256SUMS；
5. 失败区：全部未关闭失败状态及处理；
6. 治理区：contract_sha256、DEV 教训引用（Spark run-1/run-2 = normative fixtures）。

## 9. Execution Profile（与 A2 边界）

- 默认：一次 calibrate session 内参照与候选用同一 execution profile。
- 跨机器：official 参照 bundle 只需 provenance RECORDED + 语义等价 VERIFIED（KL 语义 diff 证据）；不强制同 -ngl/backend。
- eval-v1 语义量（-c 512 -b 512、KL 算法、语料、指标）不可配置。

## 10. 失败状态枚举（fail-closed）

| 状态 | 触发 | 后果 |
|---|---|---|
| CORPUS_DRIFT | 切片 SHA ≠ 冻结值 | 硬失败 |
| REF_GENERATION_FAILED | 参照重试后仍无有效 .kld | 硬失败 |
| REF_RUNTIME_UNVERIFIED | 运行时无语义一致性证据 | 硬失败 |
| IMATRIX_COVERAGE_MISSING | 可量化矩阵缺条目且未按规提升 | 点 NOT VALIDATED；低档无替代 → INSUFFICIENT_WINDOW |
| INSUFFICIENT_WINDOW | 补点预算耗尽后 n<3 或 witness 缺失 | Guard=candidate，exit 非零 |
| NOT_REACHABLE | 某档锚点无 healthy 窗口可覆盖 | candidate + unreachable 标记 |
| BUDGET_EXHAUSTED | search 层预算耗尽 | 按 search 状态传播 |
| INPUT_DRIFT | BF16/imatrix 与声明 SHA 不符 | 硬失败 |

## 11. Determinism / canonical replay（A1 验收门）

同 input + 同合同 → canonical 输出逐字段一致：floors（Decimal trunc4）、sample_count、floor_method、validation_status、witness、curve 点集（name/size/artifact_sha/kl/top）、manifest SHA、guard profile_hash、record SHA。非 canonical 字段（时间戳/run-id/路径/耗时/host）不参与。**P2 回归测试：Spark 全量重放 == 现行 guard v3 语义值。**

## 12. DEV 教训的规范地位（normative fixtures）

1. 地板向上 round gate 掉来源点（Q6_K 93.4952% vs round 0.9350）→ §5.2。
2. 小样本 min-地板使最差诊断点变 PASS 且无相邻 FAIL（FS01）→ §5.4 + §6。
3. （v1.1 新增，GPT 增补 5 夹具）window locality / duplicate observation / witness 缺失 / expected vs estimate / decimal truncation 边界。

## 13. §13 裁定记录（GPT，2026-09-06）

| 裁定点 | 结论 |
|---|---|
| promotion 追加 Search PASS | 不需要 ✅（正交）；新增 calibration-layer witness（§5.6） |
| 空洞探针 ≤4/档 | ✅ 预算语义钉死（§4） |
| 地板样本冻结校准期 | ✅ 严格执行（§5.4） |
| expected_valid_tokens informational | ❌ authoritative 保留；estimate 另字段（§2.3） |
| 合同本体 experiments/MD | ❌ 包内 machine contract JSON + MD archive（§14） |

normative 修订：A 窗口局部性 W(K)=[0.85K,1.15K]（blocker）/ B macro_same_top 直接消费 eval-v1（blocker）/ C Decimal trunc4（blocker）/ D unique observations（blocker）——全部已并入本 v1.1。

## 14. 机读合同（canonical）

`src/fit_gguf/contracts/fidelity-calibration-v1.json`：A3 canonical JSON v1 序列化；`contract_sha256` = sha256(去自身 contract_sha256 字段后的 canonical bytes)；experiments 的 CONTRACT.md/本 MD 记录 machine_contract_sha256。runtime/registry admission 只信 JSON 与其 SHA，不信 MD 排版。
