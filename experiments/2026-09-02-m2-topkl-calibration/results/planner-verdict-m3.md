# Planner Verdict — M3 Fidelity Contract v1 架构裁定（GPT，2026-09-03）

来源：GPT 对 gemma M3 汇报的完整回复（签字版）。上接 planner-verdict-gemma-g2.md。

## 0. 总方向

**M3 选 C 为主、A 为实现机制、B 只保留为研究假设。**
Fidelity Contract v1 冻结「全局 KL Core + Guard Policy」，**不冻结任何全局 Same-top 数字**；
93/90/88/85 降级为 `historical dense-candidate-r1 — REJECTED as global guard`（保留实验史，移出正式合同）。

## 1. Gemma violated 定性

- 正式结论：**Top|KL 不具备跨 family/architecture 的统一映射**（四族证据充分）。
- 关键排序：Granite(dense 8B) > Ling(MoE 7.9B) ≈ Gemma(dense E4B) → **dense/MoE 不是决定性分类变量**；但规模/有效容量仍是强 confounder。
- 假设登记：`H-M3-01: Top|KL is model/family/architecture dependent — SUPPORTED`；
  `H-M3-02: Effective capacity contributes materially to guard stiffness — OPEN`（不贴 family/scale 二选一标签）。
- Gemma 数据**接受**用于否决 global dense guard（四档同方向 2.3-3.8pp 系统偏移，非单点抖动）；
  未来若冻结 gemma4-e profile，G-A1（blk.24-41 attn_k imatrix missing workaround）必须写入 provenance。

## 2. Fidelity Contract v1 架构（✅ READY TO FREEZE）

```
Evaluator: eval-v1
Global KL Core: Quality <= .05 / Balanced <= .10 / Compact <= .15 / Mini <= .20
Same-top: REQUIRED — threshold resolved by validated Guard Profile
Unknown profile → Fidelity Tier NOT VALIDATED → onboarding required
```

数学：KL ≤ K_t 且 SameTop ≥ G(p, t)；**不再存在 G(t) 全局阈值**。

## 3. Guard Profile = 独立 YAML artifact（scope 三级）

```
guard_profile_id / guard_profile_version / evaluator_contract: eval-v1
scope: {type: exact_model | family | architecture, identifier}
calibration_models / calibration_manifest_hashes
tiers: {quality|balanced|compact|mini: {kl_anchor, same_top_floor}}
confidence / status: candidate | validated / profile_hash
```
- Level 1 exact-model（如 orcarouter 单模型）→ Level 2 family → Level 3 architecture
  （**不要轻易晋升 Level 3**）。

## 4. 新 family onboarding（方向对，修正取数法）

- 不机械"取 P5"——n=2 的 P5/P10 是"穿西装的 minimum"。
- onboarding = **pre-registered local boundary calibration**：保留 Top@KL crossing、
  local-window observations、P5/P10 diagnostic、sampling density、confidence。
- profile freeze 门槛：**每 tier ≥3 个有效 near-boundary points**，preset cliff 掏窗则强制 FIT fill-window。

## 5. 未验证架构的 CLI 行为（硬拒绝）

```
Fidelity validation unavailable: No validated Same-top Guard Profile for this model/family.
Options: --calibrate-fidelity --experimental-fidelity
```
- 不许静默用 90%，不许偷 fallback 到 Qwen。
- 正式命名 tier（如 FIT-Balanced）仅当 **KL Core PASS + Validated Same-top Guard PASS**。

## 6. B（容量分层）明确反对进 Contract v1

数据远不够（Gemma 低≠4B→low）；规模改为 **profile confidence feature**，不是 contract branching dimension。

## 7. 第 4 family：要跑，不阻塞 M3；选择改为 **Qwen3-4B > Llama-3.2-3B**

- 理由：Llama 3B 同时变 family+scale 两个变量；Qwen3-4B 提供 Qwen lineage 大→小的对照。
  若 Qwen3-4B ≈ Gemma 4B → 容量假设增强；若接近 Qwen27/Granite → 家族效应增强。
- 定性：H-M3-02 研究任务，**不是 M3 freeze blocker**。

## 8. orcarouter 四档三方对比：重要纠正

- **当前 FIT v0.2 Quality(KL .0523>.05) 与 Compact(.1746>.15) 不达标** → 不能叫正式
  "FIT v0.2 Quality/Compact" tier；Balanced(.0976)✅、Mini(.1924)✅。
- 现表降级为 **Refine Bootstrap Snapshot**（开发记录）：证明 bootstrap-v0 尚未在所有区间
  兑现 Refine 收益（Quality 差 .0023、Compact 真回退 .1746 vs .1439）。
- **正式三方 benchmark 等 chain-conditioned marginal model 或至少 Refine v1 allocator 到位后再做**，
  否则是拿 bootstrap 代表 FIT v0.2 最终算法，不公平。

## 9. 双主表设计（正式格式）

**主表 A — Quality @ Fixed Size**：用 **native preset 实际尺寸作公共 byte anchor**
（orcarouter: Q5_K_M 17.91G / IQ4_XS 14.05G / IQ3_S 11.57G / IQ3_XXS 10.42G），
Native / FIT v0.1 / FIT v0.2 三方法全部做**完全相同 target bytes**，
报 KL + Top → 回答"同样 GiB 谁质量最好"（真正测 Refine）。

**主表 B — Size @ Fixed Fidelity**：三方法各自求 **minimum passing byte size**
（KL ≤ K_t 且 Top ≥ G_t），报 Min Size / Active Constraint / Saving vs Preset。
Preset 必须扫完整 ladder 取最小达标档（不许人为选高档）；FIT 两版也找最小达标点——三方完全公平。

## 10. 状态牌

```
M1 eval-v1 ✅ FROZEN
M2 Candidate Calibration ✅ PASS（Qwen/Granite/BailingMoE/Gemma E4B）
  Global Same-top hypothesis ❌ REJECTED；Dense Same-top hypothesis ❌ REJECTED
M3 Fidelity Contract Architecture ✅ READY TO FREEZE
  ├── global KL Core └── profile-resolved Same-top Guard
Guard Profiles 🚧 separate artifacts
H-M3-02 Capacity hypothesis 🚧（Qwen3-4B recommended next）
M6 Refine final allocator 🚧
M10 Ornith 🔒 / M11 independent family 🔒
```

## 二轮回执裁定（同日，双主表回执后）

1. Guard Profile orcarouter exact-v1 **VALIDATED ✅**（补 validation_basis/generalization_scope 字段；resolver 应绑权重 hash——待办）。
2. 双主表纠正：A 表 byte delta≠0 → 改名 **Near-Preset Development Snapshot**，正式 Quality @ Fixed Size 须 byte delta=0 待 Refine v1；B 表改 **Smallest Observed Passing Artifact**（Compact 行按模板标注 true crossing 未搜索）。
3. **P0 = M6b band-conditional Refine**（C(role, layer_band, src→dst, arch)）；M6 gate：Fixed Size 下 Refine v1 KL ≤ bootstrap KL（Compact/Mini 重点、无明显回退）+ Fixed Fidelity 下 Size_refine_v1 ≤ Size_v0.1 于 ≥2/3 档（Quality 不要求首轮就赢）。
4. M6 拆分：M6a bootstrap ✅ → M6b band-conditional 🚧 NEXT → M6c chain-conditioned ⏳。
5. Qwen3-4B = P3 研究任务非阻塞（GPU 空闲可并行）；诚实公开结论：orcarouter 27B native preset 强，FIT 主价值 = 任意 byte budget。

## 三轮 amendment（2026-09-03 上午，M6b gate 后）：Compact 行撤销"真回退"

**INVALID COMPARATOR — Q3_K_S toxic-window contamination**（GPT 三轮裁定，全文见 planner-verdict-m6b.md）：

- 本文件/双主表中 "Compact 档 v0.2 真回退（12.95G 才达标）" 的对照不成立：
  v0.1 参照 FIT-11.5G 窗口 = **IQ3_XS→IQ3_S**，v0.2 Compact 窗口误用**毒预设 Q3_K_S**
  做下界（P4 fixture：Q3_K_S 11.24G KL 0.2148 离群）。
- 同目标字节 12,277,279,936 换 IQ3_XS 窗口后：v2b 0.1465 @ 11.37G ≤ 0.15 锚。
- 修正结论：**公平窗口下 v0.2b 与 v0.1 曲线基本等价，且以 ~57MB 更小尺寸落在 Compact
  contract 区域**。Gate B 判定 Compact PASS。
- 连带处置：Q3_K_S 升级为 planner 禁毒下界（known_outlier=true；不作 interpolation
  anchor / bracket，除非显式强制），与 IQ2_XS poison 一起由 frontier builder 自动排除。
- M6b 总判定（GPT）：PASS —— implementation ✅ / late-ssm_out reversal ✅ / no-regression ✅ /
  Fixed Fidelity Gate 3/3 ✅ / material fixed-size uplift ❌ not demonstrated。
  v0.2 定位改判 C：Fidelity-tier exact-size allocator（Refine 降级为 quality/safety layer，
  M6c + band-swap 并入 v0.3）。
