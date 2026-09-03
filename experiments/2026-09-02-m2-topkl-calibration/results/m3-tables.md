# M3 表 — orcarouter-Qwen3.8-27B-Uncensored（2026-09-03；GPT 二轮裁定后定名，见文末）

方法分组：preset = 原生预设（M2A/M2B 归 FIT v0.1，它们是 FIT 工艺产物）；
FIT v0.1 = balanced 分配器（P4/M2 时代 + Table A v01）；
FIT v0.2 = balanced + Refine C_role bootstrap-v0（V2 四档 + Table A v02）。
全部 eval-v1 口径；FIT 全部经 G2 重终结门（12/12 delta+0）。

## Near-Preset Development Snapshot（原名主表 A；FIT 实际落点与 anchor 差 0~0.65GiB，byte delta ≠ 0，故不可称 Quality @ Fixed Size——正式版须三方法 byte delta=0，待 Refine v1 后重建）

| Fixed Size | Native | KL | Top% | FIT v0.1 KL | Top% | FIT v0.2 KL | Top% |
|---|---|---|---|---|---|---|---|
| 17.91G (Q5_K_M) | Q5_K_M | 0.0414 | 95.69 | 0.0468 @17.26G | 95.24 | 0.0468 @17.26G | 95.24 |
| 14.05G (IQ4_XS) | IQ4_XS | 0.0624 | 93.79 | 0.0694 @13.96G | 93.54 | 0.0694 @13.96G | 93.54 |
| 11.57G (IQ3_S) | IQ3_S | 0.1424 | 89.53 | 0.1592 @11.56G | 88.43 | 0.1592 @11.56G | 88.43 |
| 10.42G (IQ3_XXS) | IQ3_XXS | 0.1946 | 87.12 | 0.1955 @10.42G | 86.18 | 0.1923 @10.42G | 86.40 |

诚实读法：本模型上 native preset 在四个 anchor 全部 ≥ FIT（Q5_K_M/IQ4_XS/IQ3_S 尤其明显；
IQ3_XXS 档 FIT-v02 以 0.1923 vs 0.1946 反超）。这与 P5 已知结论一致——该 27B 模型的
preset 谱系强，FIT 的价值在"任意目标字节"而非"同尺寸反超"。

## Minimum Verified Size @ Fixed Fidelity（v0.2 README headline 主表；Fidelity Search v1 + preset 邻域视图，2026-09-03）

| Tier | Smaller healthy preset | FIT v0.2 FS | Larger healthy preset | Saving vs larger | Active |
|---|---|---|---|---|---|
| Quality | Q4_K_M 15.41G 0.0658/93.79 FAIL-BOTH | **16.25G 0.0497/94.88 PASS** | Q5_K_S 17.40G 0.0455/95.59 PASS | −6.6% | kl |
| Balanced | IQ3_M 11.72G 0.1445/89.38 FAIL-BOTH | **12.84G 0.0997/91.57 PASS** | IQ4_XS 14.05G 0.0624/93.79 PASS | −8.6% | kl |
| Compact | IQ3_XS 11.15G 0.1512/89.05 FAIL-KL | **11.17G 0.1486/89.09 PASS‡** | IQ3_S 11.57G 0.1424/89.53 PASS | −3.4% | kl |
| Mini | Q2_K 9.98G 0.2439/84.08 FAIL-BOTH | **10.35G 0.1924/86.31 PASS** | IQ3_XXS 10.42G 0.1946/87.12 PASS | −0.6% | kl |

- 邻域 = actual bytes 机械解析的 healthy preset frontier（毒预设不列为邻居）；
  cell 判据 = KL ≤ Tier KL AND Top ≥ Guard floor（PASS/FAIL-KL/FAIL-TOP/FAIL-BOTH）。
- FIT v0.2 列 = Fidelity Search minimum verified PASS（bracket ≤128MiB，G2 delta+0，
  交付物本体 verify eval）。
- 脚注 †（Mini）：Search stopped at the validated healthy-frontier boundary; lower-size
  region lacks a valid interpolation window（Q2_K 9.98G FAIL 为边界证据）。
- 数据源：P4 发布批次 p4-results.json（IQ 系+Q2 系）+ M2 manifest/logs（K 系高段），
  全部 eval-v1 口径；P1 窗口地板探针与 P4 数字逐位互证。
- Compact 行（2026-09-04 采纳更新，GPT 裁定 planner-verdict-r2.md）：B_FS 由 11.37G
  采纳为 **11.17G**——R2 独立 sweep 在 11.37G−128MiB 线下发现两个 healthy PASS
  （11,991,706,848 = 0.1486/89.09 与 12,023,164,128 = 0.1493/89.10），fidelity-search
  重跑纳入全部证据后以 **noise_inversion / smallest verified PASS** 状态 0 fresh evals
  收敛于 11.17G（bracket [12,076,248,288 FAIL, 11,991,706,848 PASS]）。穿越区配方阶梯
  质量非单调（0.1486 PASS / 0.1519 FAIL / 0.1493 PASS / 0.1502 FAIL / 0.1539 FAIL /
  0.1582 FAIL，六观测跨 ~105MB）——**"更大不保证更好"在 ~100MB 尺度上真实存在**。
  主动约束 = **kl**（不再是 Same-top active case）。‡ = manual_verified_promotion：
  CLI 对 noise_inversion 保持 fail-closed，11.17G 经存档 recipe 重建（G2 delta+0）→
  本体 verify eval 通过后由裁定晋升出货。案例口径 = "Why FIT verifies the final
  artifact instead of trusting size monotonicity"（非单调配方区 + 最终工件验证），
  双硬门放 Fidelity Contract 部分讲，不包装成 Same-top 案例。R2 状态 =
  INITIAL FAIL → CORRECTIVE ACTION TAKEN → AMENDED RELEASE VALIDATION PASS
  （历史永久保留；按 Codex 审查口径，修正后的 PASS 不是 untouched independent
  sweep，而是 evidence-incorporating correction 后的发布验证）。

## 结论（对照 GPT 预期）

1. **v0.1 vs preset**：Balanced +7.5%、Quality +4.1%、Compact +1.1%、Mini ±0——FIT 价值成立。
2. **v0.2 (bootstrap-v0) vs v0.1**：Balanced +0.13pp top 同尺寸微优；Compact 区间真回退；
   整体未兑现——证实 GPT 判断：bootstrap-v0 不能代表 Refine v1，正式 benchmark 等
   chain-conditioned marginal model。
3. Guard Profile（orcarouter exact-v1, validated）+ CLI 未验证架构硬拒绝已实现（103 测试绿）。


## GPT 二轮裁定要点（2026-09-03，已执行）

1. **orcarouter exact-v1 Guard Profile：VALIDATED ✅**——补 governance 字段
   `validation_basis: dev_calibration`、`generalization_scope: exact_model_only`（已加）；
   exact-model resolver 建议绑定权重 hash 而非仅 repo 名（待办）。
2. 主表 A 改名 **Near-Preset Development Snapshot**；正式 Quality @ Fixed Size 须
   byte delta = 0（三方法 exact same bytes），Refine v1 后重建。诚实结论可公开：
   orcarouter 27B native preset 本身很强，FIT 主价值 = 填充任意 byte budget。
3. 主表 B 改名 **Smallest Observed Passing Artifact**；Compact 行按模板呈现
   （"11.56G candidate failed KL; true crossing 未搜索"）；正式 minimum passing size
   等 Fidelity Search（coarse→bracket→fine→verify）。
4. **优先级 P0 = M6b band-conditional Refine**（C(role, layer_band, src→dst, arch)；
   验证案例 = late ssm_out；gate：同字节 Refine v1 KL ≤ bootstrap KL（Compact/Mini 重点）
   + Fixed Fidelity 下 Size_refine_v1 ≤ Size_v0.1 于 ≥2/3 档）；M6c chain-conditioned 后续；
   **Qwen3-4B = P3 研究任务非阻塞**（GPU 空闲可并行）。
5. 状态牌：M3 Contract Architecture PASS/FREEZE；Guard Profile orcarouter exact-v1 VALIDATED；
   M6b NEXT。
