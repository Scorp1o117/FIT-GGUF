# GPT 四轮裁定 — P1 Fidelity Search 旗舰验证（2026-09-03 下午）

## 最终裁定

**P1 Fidelity Search：PASS ✅ —— "Fidelity Search v1 flagship validation PASS"**

1. **主表 B 正式批准为 v0.2 README headline**，标题：
   **"Minimum Size at Fixed Fidelity — Qwen3.8-27B-Uncensored"**；
   副注（原文）："FIT v0.2 searches for the smallest verified GGUF that satisfies both
   eval-v1 KL and the validated model-specific Same-top guard. Search tolerance: <=128 MiB."
   - 总口径 = "minimum verified PASS within the validated search frontier"，
     不用无限定的数学全局最小值。
   - **不要泛化**成"所有模型都省 8.6%"——这是 exact-model flagship case study。
   - **Mini 加脚注 †**："Search stopped at the validated healthy-frontier boundary;
     lower-size region lacks a valid interpolation window."（Q2_K 0.2439 FAIL 可作边界证据，
     不暗示 9.xG 不可能偶然 PASS。）
2. **R2 需要补，但重定义为 Independent Local Reference Sweep**（不跑完整 dense curve）：
   - 四个最终 crossing 各预注册固定 byte grid：step = 64MiB，
     range = final PASS − 256MiB → + 64MiB；
   - 已有观测点 cache hit，只补缺失点；
   - 求 smallest passing grid point，验收 |B_FS − B_grid| ≤ 128MiB 或相对 ≤0.5%；
   - Mini 只在 healthy frontier 内 sweep（"Search matches the independent minimum
     within the healthy frontier"）。
   - 理由：bracket 收敛证明的是搜索器自知收敛；independent sweep 证明的是没有因
     策略/缓存/路由/局部非单调漏掉更小的健康 PASS——Compact 的 0.1502/0.1539 噪声
     反转正是活例子。
3. **Compact 小案例进 README 小框**：
   "Compact target: KL <= .15, Same-top >= 88.94%; 11.37G: KL .1465 PASS,
   Same-top 89.05 PASS, only +0.11pp margin → active constraint = Same-top" +
   "A KL-only search would have continued shrinking the model, while FIT v0.2 stops
   when decision fidelity becomes the tighter constraint."
4. **Compact 重放 smoke（可选加分项）**：从空 search cache 重放一次 Compact search，
   确认最终 bracket / selected artifact / metrics 一致（最复杂 case 的端到端冒烟，
   不是新 gate）。
5. **队列调整为 P3a → P2 → R2 → P3b**：
   - **P3a** 固化公开 CLI 产品路径：`fit quantize model --fidelity-tier balanced` 真走
     resolve Guard → Fidelity Search → quantize → eval → verify → G2 finalize → artifact；
     钉住公共语义：Normal ≤8 / Precise ≤16 / cache / run-id / active constraint /
     healthy frontier / NOT REACHABLE。
   - **P2** 用最终 CLI 重建 Fixed Size 表 A（README benchmark 应由用户可调用的生产路径
     生成，而非 development harness；表 A 降为 Secondary Table — Quality @ Exact Fixed
     Size，byte delta = 0，"Native >= v0.2 ~= v0.1" 也照实发）。
   - **R2** 独立 local sweep 补正式 release evidence。
   - **P3b** README / release docs / metadata 最后冻结。
6. **R1-R6 正式签字**：R1 ✅ PASS（4/4 KL+Guard）/ R2 🟡 PROVISIONAL（bracket 收敛 PASS，
   independent sweep pending）/ R3 ✅（11 evals = 1/4/3/3，全 ≤8）/ R4 ✅（全 G2 delta=0）/
   R5 ✅（4/4 Size ≤ v0.1，公平前沿对照）/ R6 ✅（确定性 manifest/pipeline）。

## 产品一句话（GPT：现在不是宣传话术，是四档全跑通的实验事实）

"Choose the fidelity you want; FIT finds the smallest verified GGUF that safely meets it."
