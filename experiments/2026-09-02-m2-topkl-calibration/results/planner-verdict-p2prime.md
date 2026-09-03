# GPT 六轮裁定 — P2 取消、Tier 邻域视图（2026-09-03 晚）

用户方向：v0.2 主要是加功能，不必刻意与 v0.1 对比（质量相近已有 Gate 数据支撑）；
每档只和预设的最近两档（高、低）对比。

## 最终裁定

1. **原 P2（三方法 exact-byte Size Mode 表 A）正式取消 ❌**。v0.2 README 不再背着
   "Native vs FIT v0.1 vs FIT v0.2"证明 allocator 变聪明了多少。产品故事 =
   "Native presets 是离散台阶；FIT v0.2 在台阶之间自动寻找满足 Fidelity Contract 的
   最小 verified GGUF"。v0.1 从 README 主证据链退出，只在 changelog/history 说明。
2. **邻域视图并入主表 B ✅**（不再有 A/B 两张表），六列：
   Tier / Smaller healthy preset / FIT v0.2 / Larger healthy preset / Saving vs larger /
   Active constraint。
3. **邻域定义 = actual bytes 机械解析**（不是"印象里对应的前后两个 preset"）：
   healthy preset frontier 内最大 B_preset < B_FIT 与最小 B_preset > B_FIT；
   毒预设（Q3_K_S/IQ2_XS）不允许作为邻居。
   - 我提的 Quality 邻域（Q5_K_S/Q5_K_M）被纠正：两个都比 FIT 大。机械解析后
     Quality lower = **Q4_K_M**（metadata 查询确认）。
4. **PASS/FAIL 标注规则**：邻域 cell 的 PASS = KL ≤ Tier KL AND Top ≥ exact-model
   Guard floor；cell 显示 PASS / FAIL-KL / FAIL-TOP / FAIL-BOTH。
5. **IQ3_M eval 批准**但附机械检查前提——执行时发现 P4 发布批次已有全预设梯队
   eval-v1 数据（IQ3_M 11.72G 0.1445/89.38），且 P1 窗口地板探针（IQ3_XS/Q2_K）
   与 P4 数字逐位一致（管线确定性互证）→ **零新增 eval**。
6. R2 不受影响：五点 grid −256/−192/−128/−64/+64MiB，FAIL 判据 = 发现
   < B_FS−128MiB 的 healthy PASS。P2' 是 presentation evidence，R2 是 search
   correctness evidence，职责不同。

## 机械解析结果（全表零新增 eval）

| Tier | Smaller healthy preset | FIT v0.2 FS | Larger healthy preset | Saving | Active |
|---|---|---|---|---|---|
| Quality | Q4_K_M 15.41G 0.0658/93.79 FAIL-BOTH | **16.25G 0.0497/94.88 PASS** | Q5_K_S 17.40G 0.0455/95.59 PASS | −6.6% | kl |
| Balanced | IQ3_M 11.72G 0.1445/89.38 FAIL-BOTH | **12.84G 0.0997/91.57 PASS** | IQ4_XS 14.05G 0.0624/93.79 PASS | −8.6% | kl |
| Compact | IQ3_XS 11.15G 0.1512/89.05 FAIL-KL | **11.37G 0.1465/89.05 PASS** | IQ3_S 11.57G 0.1424/89.53 PASS | −1.7% | same_top |
| Mini | Q2_K 9.98G 0.2439/84.08 FAIL-BOTH | **10.35G 0.1924/86.31 PASS** | IQ3_XXS 10.42G 0.1946/87.12 PASS | −0.6% | kl |

数据源：P4 发布批次 p4-results.json（IQ 系+Q2 系，eval-v1 口径）+ M2 manifest/logs
（K 系高段 Q4_K_S/Q4_K_M/Q5_K_S/Q5_K_M）。Compact 行是全表最漂亮的一行
（IQ3_XS 差 0.0012 过不了；IQ3_S 能过但贵 200MB；FIT 自动落在中间）——README 教科书展示。

产品一句话（GPT）："Presets give you steps. FIT gives you the space between them—and
stops at the smallest verified point that still meets your fidelity target."
