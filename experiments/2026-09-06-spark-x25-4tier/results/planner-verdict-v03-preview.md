# Planner verdict — v0.3 功能更新预览评审（2026-09-06）

来源：ChatGPT 策划师会话（FIT质量下降阈值），对 ZCode 提交的 v0.3 预览（~/下载/FIT-GGUF-v0.3-功能更新预览.md）+ Spark 第二模型接入汇报的裁定。本文件为 ZCode 誊录的裁定要点，原文在会话内。

## 总裁定

**v0.3 的 release-critical 主线改为：Multi-model Fidelity Infrastructure。关键路径 = A3 → A1 → A2-lite → fresh DEV onboarding → G11 sealed。** Structural Refine（B1/B2）继续研发但**不再阻塞 v0.3 GA**（成熟可作为 experimental 附加，否则顺延）。

- `fit calibrate`：**正式立项，v0.3 核心功能**。
- large-model：**进 v0.3，但改名收缩为 Execution Profile + on-disk support**，不承诺未经验证的 "70B fully supported"。
- Spark：**DEV architecture-transfer / onboarding case study**——不是 dense candidate，绝不能算 G11 sealed。

## 一、推进顺序（Phase 0 = v0.3 G0 Scope Freeze）

先冻结 v0.3 边界：release-critical = A3 Trust Registry / A1 Calibration Protocol + fit calibrate / A2 Execution Profile·On-disk / Fresh DEV onboarding / C1 G11 unseen-family validation / CLI·docs·registry promotion；non-blocking research = B1 Structural Refine、B2 Refine benchmark、B3 capacity hypothesis、C2 family/arch guard promotion。（防止重演 v0.2 后期范围膨胀。）

## 二、第一优先级是 A3（Registry schema），不是 A1

**三层拆分**（GPT 正式批准）：
- eval-v1 = HOW TO MEASURE（永久冻结，FREEZE.json **不再修改**——orcarouter manifest 前缀保留为 v0.2 历史 bootstrap provenance，不搞 rc3）
- Calibration Contract v1（fidelity-calibration-v1）= HOW TO CREATE A MODEL GUARD
- Fidelity Registry v1 = WHICH MODEL-SPECIFIC BUNDLES ARE TRUSTED

否决预览里两个原始方案（每模型改 FREEZE / FREEZE 钉 registry 目录）。新结构：`registry/fidelity-registry-v1.json` + `registry/entries/<source_sha>.json`；entry 字段含 source_weights_sha256 / tokenizer_sha256 / evaluator_contract(+sha) / reference_manifest_sha256 / guard_profile_sha256 / calibration_contract(+sha) / scope / status / entry_sha256。**lookup 键 = source_weights_sha256 + eval-v1 digest，不是模型名**。Trust root = 当前 FIT release / git commit / package 自身（trust chain: FIT release → official registry snapshot → full-SHA 引用 manifest+guard → 都绑 eval-v1 final digest）。加模型 = registry += entry，不碰 evaluator contract。

## 三、Calibration Contract v1（fidelity-calibration-v1，A1 开工前必须冻结）

冻结内容：preset ladder 选择、healthy/poison 资格、gap 检测、local target windows、FIT fill-point 生成、same-top floor 推导、floor 数值表示、样本数规则、profile confidence、candidate→validated promotion、source SHA 绑定、参照 bundle 生成、失败/NOT VALIDATED 条件。Spark 两条发现（floor 不许向上 round、P5/min 规则）作为 DEV evidence 进 contract。

**GPT 修正 n≤2 规则**：n≥3 → eligible for validated；**n≤2 → candidate / low-confidence，不得自动 validated**（防 A1 一键化后批量生产证据过薄的 Guard）。profile YAML 增加 floor_method / sample_count / confidence / validation_status；sample_count<3 时 A1 自动继续 fill-window，预算耗尽仍不足 → Guard = candidate。

## 四、A1 fit calibrate

产出 **Calibration Bundle** 而非直接改官方 registry：calibration-record.json / reference-manifest.json / references/*.kld / curve-points.jsonl / guard-profile.yaml / profile-report.md / registry-entry.json（status=candidate|validated）/ SHA256SUMS。registry-entry 只是"可被注册的 bundle"；官方 promotion 走 `fit registry validate/add` 或维护者人工 review——区分 local self-calibration 与 officially trusted model support。

**A1 验收改为 canonical replay**：同 input + 同 protocol → 同 canonical profile digest（floors、source SHA、reference hashes、curve metrics、scope、evaluator digest、status、profile hash 必须一致；timestamp/run-id/临时路径/耗时/host 不要求）。

## 五、A2 = Execution Profile / Out-of-core Calibration（改名）

可配置：--n-gpu-layers、--device/backend、--threads、--workdir、--on-disk。eval-v1 语义（-c 512 -b 512、KL semantics、cutoff、corpus、metric 定义）**不可开放改变**。

**GPT 否决"参照与候选必须同 execution config"作为 contract 硬要求**（破坏 portability）：应为 same evaluator semantics REQUIRED + execution provenance RECORDED + execution equivalence VERIFIED。默认行为 = 一次 calibrate session 内参照与候选同 execution profile（保守默认）；官方参照 bundle 跨机器可用（runtime provenance verifier 判 semantics compatible），否则不可分发。"70B 3–7h / 9–24h" 是估算不是测量 → 只能进 design note，不得作为产品 SLA；宣称 "70B supported" 前至少拿一个真实 >RAM/>VRAM 模型 smoke。

## 六/七、B1/B2 = 并行研究线（不阻塞 GA）

B1 需完整 G0（baseline allocator、band-swap 动作空间、paired 降/升级规则、字节中性/修复策略、proposal score、chain model、DEV 数据集、excluded sealed、regression fixtures、compute budget、fixed-size gates、fixed-fidelity gates、成功标准）。**GPT 修正预览 gate**：对比基线从 v0.1 改为 **v0.2 current allocator**（v0.1 历史化）；第一阶段成功门 = Fixed Fidelity: Size_structural ≤ Size_v0.2 于 ≥3/4 档且无档 >+0.5%；Fixed Size: 无系统性 KL 回退且 ≥2/4 档可测改善；compute 在预注册预算内。B2 依附 B1 的 G0（验证泛化，不是重新无穷拟合 C_role）。DEV universe：Qwen/Granite/Ling/Gemma/Spark/PRISM 全部永久 DEV。

## 八、B3 Qwen3-4B：建议跑，双价值

① H-M3-02（Qwen lineage 大→小 Top|KL 随 capacity 下移？）② **A1 的 fresh DEV onboarding validation**——Spark 是 A1 的 replay fixture（A1 因它而生），Qwen3-4B 验证"没为模型写专用脚本也能跑通"。A1 实现后立即排队；跑完即 DEV，不得用于 G11。

## 九、C1 G11：最后，顺序钉死

A3 schema FROZEN → Calibration Contract FROZEN → A1 FROZEN → A2 semantics FROZEN →（若 GA 含 B1 则 B1 algorithm FROZEN）→ G11 预注册（选 unseen family → 全部 hash → one-shot）。若 GA 不含 Structural Refine，G11 不验证 B1。G11 禁选 family：Qwen/Granite/Gemma/BailingMoE-Ling/Spark/Ornith/PRISM 全部禁。

## 十、C2 dense-v1/moe-v1：从 v0.3 release scope 降级（research-only）

Spark 进一步证明 same-top 不是 dense/MoE 二分类（现有排序：Gemma dense ≈ Ling MoE < Granite dense < Qwen dense）。v0.3 保持 exact-model Guard 为默认成熟层级；family scope promotion 需 ≥2 独立 calibration models + 1 个未参与拟合的 transfer validation；architecture scope 更严格。

## 十一/十二、Spark 治理定位（正式）

**Spark = DEV architecture-transfer + onboarding case study**（标签：split=dev / role=architecture_transfer+onboarding_case+calibration_protocol_fixture+multi_model_product_validation / architecture_family=spark2_5_hybrid / **sealed_eligible=false**）。理由：已进 development loop（暴露 UTF-8 bug、改变 floor derivation discipline、推动 A1/A3、影响 v0.3 设计）。可以做：A1 replay fixture、A3 registry fixture、B1/B2 DEV、v0.3 regression fixture。不能做：G11、independent-family final generalization claim。**其 exact-model Guard v3 可继续 validated/dev_calibration**——Guard validated（对这个 exact model 完成了 onboarding）与 DEV（不能证明跨 family 泛化）是两个正交维度。

## 十三、v0.3 主题句修改

新主题句："**FIT-GGUF v0.3：从单模型 Fidelity 验证走向可复用的多模型 Fidelity 基础设施——自动校准、可信注册、可扩展执行；Structural Refine 作为并行质量研究线推进。**" 产品句："**Calibrate once. Search safely. Reuse everywhere.**"（或 "Bring Fidelity Search to any supported model."）

## 十四、优先级表

| 优先级 | 工作项 | v0.3 GA blocker | 需要 G0 |
|---|---|---|---|
| P0 | A3 Fidelity Registry v1 | ✅ | Trust/schema freeze（非 Refine G0） |
| P1 | Calibration Contract v1 | ✅ | ✅ Calibration prereg |
| P2 | A1 fit calibrate | ✅ | 依赖 Calibration Contract |
| P3 | A2 Execution Profile / on-disk | ✅ Lite scope | Engineering spec，不需 full G0 |
| P4 | Qwen3-4B fresh DEV onboarding | ✅（建议作为 A1 acceptance） | Hypothesis prereg |
| P5 | C1 G11 unseen-family sealed | ✅ | ✅ Full sealed prereg |
| Parallel | B1 Structural Refine | ❌ | ✅ Full G0 |
| Parallel | B2 benchmark/profile | ❌ | 跟随 B1 G0 |
| Later | C2 family/arch Guard promotion | ❌ | Promotion criteria prereg |
| Final | C3 naming/docs | ✅ | 不需 G0 |

## 结语（GPT 原文要点）

"这次 Spark 给项目带来的最大结论其实很清楚：**v0.2 已经证明 Fidelity Search 能工作；v0.3 真正要解决的是'如何让一个陌生模型以可审计、可重复、无需手搓脚本的方式进入这个系统'。** 这个问题解决后，FIT 才真正从'主人能维护的项目'变成'别人也能自己 onboarding 模型的工具'。"
