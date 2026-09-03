# GPT 三轮裁定 — M6b gate 结果 + v0.2 目标定位（2026-09-03 上午）

提问背景：M6b 实现完成（112 测试绿），gate B 3/3 PASS 但与 v0.1 拉不开质量差距；
Gate A parity；Compact"回退"确认为 Q3_K_S 毒窗口混杂。问：v0.2 要以 refine 质量为目标，
还是收缩为 fidelity-tier 档位 allocator？

## 最终裁定

1. **v0.2 定位 = C**："FIT-GGUF v0.2 = Fidelity-aware exact-size GGUF allocator"。
   主卖点 = 用户给 Fidelity Tier，FIT 自动找到满足质量合同的最小安全尺寸并生成字节精确 GGUF。
   Refine 保留但重新定位为 "allocation quality / safety improvement layer"，
   不承诺 "显著超越 v0.1"。发布口径（GPT 原文）：
   "v0.2 allocator quality: non-regressive against v0.1 under fair window matching"。
2. **B（结构性 band-swap 优化器）→ v0.3 主研究线**；M6c 不删，迁移 v0.3 与 B 合流；
   A 不再作为 v0.2 blocker。理由：决定收益上限的是 action space（1-3% 边际字节重排 vs
   PRISM 级整 band 结构搬移），不是评分器聪明程度。
3. **--fidelity-tier 直出最终 artifact**：`fit quantize model.gguf --fidelity-tier balanced`
   = 生成满足 Balanced Contract 的最小尺寸 GGUF（自动链：Resolve contract → Guard Profile →
   Load tier → Search size → Plan → Quantize → eval-v1 → PASS/FAIL → refine bracket →
   minimum verified PASS → G2 finalize）。产物命名 `<Model>-FIT-Balanced-<Size>-<PrimaryQType>.gguf`。
4. **Tier 语义：双硬门**。Tier 身份由 Global KL Core 定义（K_t）；Guard Profile 决定该模型
   能否安全获得该 tier（Same-top ≥ G(p,t)）。Pass_t(B) = [KL ≤ K_t] ∧ [Top ≥ G(p,t)]。
   不能说 tier "锚定 Guard floor"，也不能只说 "KL≤0.10"。
   Guard Profile 缺失 → --fidelity-tier 硬拒绝（已实现，维持）；experimental 产物必须命名
   `FIT-Balanced-Candidate` 防止合同语义稀释。
5. **Fidelity Search v1 必须进 v0.2**（不能停在 Smallest Observed Passing Artifact）：
   coarse → bracket → active-constraint detection → fine → verify；
   预算 Normal ≤ 8 evals / Precise ≤ 16。
   最终产物必须是 **minimum verified PASS**（区分 Observed PASS / Search estimate / Verified minimum PASS）。
   Active constraint 指标保留：M_KL = K_t − KL、M_Top = Top − G(p,t)，归一化取 min → 自动识别
   KL/Same-top/Joint 主约束。
6. **M6b 判定：PASS，但 material fixed-size uplift 未证明**（implementation ✅ / late-ssm_out
   reversal ✅ / no-regression ✅ / Fixed Fidelity Gate 3/3 ✅ / fixed-size uplift ❌）。
   不是 Gate FAIL——因为 v0.2 目标已重定义。价值=证明 band-aware prior 可用且无害，
   同时实验性确认了狭窄动作空间限制收益。
7. **Compact 历史结论 amendment：同意**。旧 "v0.2 Compact 真回退" 撤销，标记
   `INVALID COMPARATOR — Q3_K_S toxic-window contamination`。修正结论：公平窗口下 v0.2b 与
   v0.1 曲线基本等价，且以 ~57MB 更小尺寸落在 Compact contract 区域。
   **必须写进 planner-verdict-m3.md amendment，不能只在新报告提一嘴。**
8. **Q3_K_S 升级为 planner 禁毒下界**（两次污染实证：PRISM/G7 离群 + M3 Compact 窗口）：
   known_outlier=true → 不得作 interpolation anchor、不得作 bracket 上下界（除非显式强制）；
   可评测记录但不参与健康 frontier。与 IQ2_XS poison 一起由 Fidelity Search frontier builder
   自动排除——"不是只找最小尺寸，而是在 healthy frontier 上找"。
9. **双主表**：表 A（Quality @ Fixed Size）重建为 non-regression / allocator characterization
   表（byte delta=0；preset ≥ FIT ≈ v0.1 就照实发，反而强化真实定位）。
   表 B（Size @ Fixed Fidelity）升级为 **主表中的主表（README headline）**：三方
   （Native preset / FIT v0.1 / FIT v0.2）在同一合同（KL Core + orcarouter exact-v1 Guard）下
   各自找 min Size。
10. **v0.2 Release Gates 重写**（删除 "Refine 平均 KL 提升 ≥3%" 硬门）：
    - R1 Fidelity correctness：四档最终 artifact KL PASS + Guard PASS
    - R2 Search correctness：距 dense sweep 最小 PASS ≤0.5% 或 ≤64-128 MiB
    - R3 Search budget：Normal ≤8 / Precise ≤16 evals
    - R4 G2：predicted == actual
    - R5 Non-regression：公平窗口/同尺寸下不系统性显著差于 v0.1
    - R6 Reproducibility：同输入 → same plan / same artifact / same metrics
11. **新优先级队列**：
    ```
    P0 Fidelity Search v1（coarse→bracket→fine→verify + healthy-frontier filtering + active constraint）
    P1 orcarouter 四档真正 minimum PASS → 重建 Size @ Fixed Fidelity 主表
    P2 最终 exact-byte Fixed Size 主表
    P3 v0.2 release gates / docs
    v0.3 Structural Refine（band-swap action space + paired moves + free-slot + ΔU(a_t|S_t)）
    ```
    Qwen3-4B（H-M3-02）继续 research backlog 不挡发布。

## GPT 收束语

"不是 C 不够聪明，是它根本没有权限做 PRISM 那些真正值钱的大动作。那就别让 v0.2 为错误的
动作空间继续延期。先把 eval-v1 + Guard + G2 + Fidelity Search 做成完整产品；v0.3 再把手伸进
band-swap 这个真正的大金矿里。"
