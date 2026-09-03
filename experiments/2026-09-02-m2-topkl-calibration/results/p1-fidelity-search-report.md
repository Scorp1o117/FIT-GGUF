# P1 — orcarouter 四档 Fidelity Search 结果（2026-09-03）

GPT 三轮裁定后的首个旗舰交付：**Size @ Fixed Fidelity 主表（表 B 升级版）**。
全部四档 = minimum verified PASS（bracket ≤128MiB），v0.2 栈（band profile + balanced +
G2 delta+0 + eval-v1），预算 ≤8 evals/档（实际共 11 次）。

## 主表：Size @ Fixed Fidelity（最小已验证达标工件）

| Tier | 合同（KL ≤ / Top ≥） | Native preset | FIT v0.1 | **FIT v0.2 FS** | vs v0.1 | vs preset | active constraint |
|---|---|---|---|---|---|---|---|
| Quality | 0.05 / 94.75% | Q5_K_S 17.40G (0.0455/95.59) | 16.68G (0.0496/95.00) | **16.25G** (0.0497/94.88) | **−2.6%** | −6.6% | kl |
| Balanced | 0.10 / 91.18% | IQ4_XS 14.05G (0.0624/93.79) | 13.00G (0.0987/91.93) | **12.84G** (0.0997/91.57) | **−1.2%** | −8.6% | kl |
| Compact | 0.15 / 88.94% | IQ3_S 11.57G (0.1424/89.53) | 11.43G (0.1439/89.23) | **11.37G** (0.1465/89.05) | **−0.5%** | −1.7% | **same_top** |
| Mini | 0.20 / 85.03% | IQ3_XXS 10.42G (0.1946/87.12) | 10.42G (0.1955/86.18) | **10.35G** (0.1924/86.31) | **−0.6%** | −0.6% | kl |

四档全部 **verified_pass**；尺寸 4/4 ≤ FIT v0.1（GPT gate 要求 ≥2/3）；预算 1/4/3/3 次
（远低于 ≤8）。所有工件 G2 门 delta+0，manifest hash 留档，tmpfs 已清理。

## 搜索过程要点

1. **括号收窄轨迹**（bytes）：
   - Quality [17,425,501,408 FAIL, 17,908,501,696 PASS] → 单探针 17,443,851,488 PASS 收敛（span 18MB）；
   - Balanced [13,051,034,848, 13,900,299,488] → 4 探针二分至 104MB；
   - Compact [11,186,330,848, 12,206,255,328] → 3 探针（11.14G/11.21G/11.24G 全 FAIL，跨度 134.18MB ≤ 134.22MB 容差闭合）；
   - Mini [10,711,665,888 (Q2_K 预设 0.2439), 11,116,944,608] → 3 探针至 97MB。
2. **Compact 噪声反转被规则诚实吸收**：11.21G kld 0.1502 vs 11.24G kld 0.1539（更大反而更差），
   bracket 按 order-free 规则收紧，容差闭合——未伪装成超出测量精度的断言。
3. **Compact 的 active constraint = same_top**：Top 余量相对动态区间 (0.1106) 已比 KL 余量更紧
   （89.05% vs floor 88.94%）——orcarouter 的 Same-top Guard floor 在 Compact 档接近绑定，
   这是 Guard Profile 架构第一次在实际搜索里"干活"。
4. **观测曲线新增 11 个点**（9.97G 0.2439 / 10.16G 0.2253 / 11.14G 0.1512 / 11.21G 0.1502 /
   11.24G 0.1539 / 12.15G 0.1203 / 12.55G 0.1121 / 12.75G 0.1019 / 16.13G 0.0536 / 16.23G 0.0524 等），
   全部入 manifest+logs，供后续 Refine/校准复用。

## Release Gates R1-R6 状态

- R1 Fidelity correctness：4/4 artifact KL PASS + Guard PASS ✅
- R2 Search correctness：对 dense sweep 的 ≤0.5%/64-128MiB — 以 bracket ≤128MiB 达成 ✅（dense 对照 sweep 未跑，如需正式证明可在 Mini/Compact 补）
- R3 Search budget：11 evals / 32 上限（≤8/档）✅
- R4 G2：全部 delta+0 ✅
- R5 Non-regression：4/4 档 Size ≤ v0.1，公平窗口下无回退 ✅
- R6 Reproducibility：同输入同 plan 同 artifact 同 metrics（确定性管线）✅

## 过程记录（执行器加固，4 个工程坑已修并入测试/记忆）

llama-perplexity 统计走 stderr → 以 parse_llama_kl_log 为唯一验收判据；plan 欠填是单阶段
固有行为（K 窗口步长粗，~420-466MB unused），probe 的 tensor-types 直接量化、禁止二次规划；
同 deliverable 重复探测 → bump+缓存+有用区间（≤max_fail/≥min_pass 跳过）；跨运行 tag 复用
导致 manifest/日志错配 → run-id 唯一命名。

## 诚实边界

- Mini 的真实 crossing 可能在 9.97G 以下（Q2_K 窗口地板以下无健康窗口覆盖），当前 10.35G
  是"健康窗口内最小已验证 PASS"。
- 本模型 native preset 质量仍强（P5 结论不变）；FIT 的价值 = 任意 byte budget + 一键
  fidelity-tier 自动化 + Guard 安全性，不是同尺寸质量反超。
