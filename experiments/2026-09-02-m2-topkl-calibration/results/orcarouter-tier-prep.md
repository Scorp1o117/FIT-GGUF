# orcarouter 四档位三方对比 — 数据齐备（2026-09-03）

四档目标 = 校准曲线交叉尺寸；FIT v0.2 = v0.2 栈构建（Refine C_role bootstrap-v0 + balanced 分配器 + G2 重终结门 delta+0 PASS ×4 + eval-v1）。

| Tier | KL 目标 | 方法 | 产物 | size GiB | macro_kld | top% |
|---|---|---|---|---|---|---|
| Quality | 0.05 | FIT v0.2 | FIT-V2-Quality | 16.19 | 0.0523 | 94.90 |
| Quality | 0.05 | FIT v0.1 | FIT-M2B | 16.68 | 0.0496 | 95.00 |
| Quality | 0.05 | preset | Q5_K_M | 17.91 | 0.0414 | 95.69 |
| Balanced | 0.10 | FIT v0.2 | FIT-V2-Balanced | 12.95 | 0.0976 | 92.06 |
| Balanced | 0.10 | FIT v0.1 | FIT-13G | 13.00 | 0.0987 | 91.93 |
| Balanced | 0.10 | preset | IQ4_XS | 14.05 | 0.0624 | 93.79 |
| Compact | 0.15 | FIT v0.2 | FIT-V2-Compact | 11.42 | 0.1746 | 88.04 |
| Compact | 0.15 | FIT v0.1 | FIT-11.5G | 11.43 | 0.1439 | 89.23 |
| Compact | 0.15 | preset | IQ3_S | 11.57 | 0.1424 | 89.53 |
| Mini | 0.20 | FIT v0.2 | FIT-V2-Mini | 10.35 | 0.1924 | 86.31 |
| Mini | 0.20 | FIT v0.1 | FIT-10.5G | 10.50 | 0.1873 | 87.38 |
| Mini | 0.20 | preset | IQ3_XXS | 10.42 | 0.1946 | 87.12 |

诚实观察（待 GPT 判读）：
- Balanced 档 v0.2 同尺寸微优于 v0.1（92.06 vs 91.93）。
- Compact/Mini 档 v0.2 略差于 v0.1（kld +0.031/+0.005）——bootstrap-v0 的 C_role 是
  proposal 级修正，Compact 档把 byte 从 IQ3_S 系挪到 attn_v 系的收益未兑现。
- Quality 档 v0.2 因 oracle 粒度落在 16.19 GiB（低于交叉目标），与 M2B(16.68) 非严格同尺寸。
- 三方对比的核心信息不变：preset 悬崖（Q3_K_S 离群、IQ4_XS 断档）在 11-14G 区间
  让"同尺寸最优"只有 FIT 能做到；v0.1/v0.2 差异 << preset vs FIT 差异。

执行事故记录：v0.2 构建初期 eval 全败根因是脚本内模型名拼写错误（orcorouter 双 o
vs orcarouter），且 /dev/shm 曾被 m2-ling 残留 32G 撑爆导致参照 ENOSPC 静默截断
（行数完整校验已固化进 analyze_m2 与驱动脚本）。
