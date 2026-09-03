# M3 汇报草稿（gemma-4-E4B dense 校准完成）— 2026-09-03 凌晨

（此文件为向 GPT 汇报的留档草稿；浏览器 webview 未附着时暂存，发送后以 verdict 文件为准。）

## 一、执行摘要
20 点（12 preset + 8 FIT 填窗）0 flag；Text-Tower 硬门 PASS（text=720/vision=0/audio=0）；**机械判定：Quality/Balanced/Compact/Mini 四档全部 violated**。dense-candidate-r1 被第三族否决，same-top-guard-dense-v1 无法按原计划冻结。

## 二、gemma-4-E4B 机械判定表
| target | window | n(win) | Top@t | P5 | P10 | candidate | verdict |
|---|---|---|---|---|---|---|---|
| 0.05 | [0.0425,0.0575] | 2 | 90.71 | 90.68 | 90.69 | 93 | violated |
| 0.10 | [0.0850,0.1150] | 2 | 86.95 | 86.48 | 86.48 | 90 | violated |
| 0.15 | [0.1275,0.1725] | 2 | 84.17 | 84.11 | 84.12 | 88 | violated |
| 0.20 | [0.1700,0.2300] | 2 | 81.71 | 82.16 | 82.17 | 85 | violated |

缺口 −2.3/−3.0/−3.8/−3.3pp——四档一致低于 dense 候选，且比 Ling(bailingmoe MoE) 还低 ~1pp。"dense class 内部族间差异"大于"dense vs MoE"分类差。

## 三、执行口径与偏差（PREREG-GEMMA.md，修正案 G-A1）
1. 硬门：Eligibility Record PASS；BF16=master 转换器 7798007a（720 张量纯文本塔）。
2. 12 preset 梯 + Q3_K_M 毒离群复现（kld 1.82 vs IQ3_M 0.47）。
3. G-A1：b10666 imatrix 采集对 gemma4 确定性缺 blk.{24..41}.attn_k.weight（两次复现）；受影响五个低_bit 预设点对 18 张量 --tensor-type-file 提升 Q4_K，其余 9 点纯 preset。
4. FIT 填窗 8 点（两轮）：第一轮 3 点因曲线强非线性落窗外，按 17 点实测曲线重新锚定后全部命中；最终每窗 n=2。

## 四、G2 重终结门生产验证
8/8 FIT 点在全新架构上首次尝试即 delta +0 PASS（三组 preset pair）。G2 修复+fixture+生产三重闭环。

## 五、工具链观察
1. /dev/shm ENOSPC 静默截断（m2-ling 残留 32G）：新增 _logits_ 行数完整算术校验（20B 头+tokens+每 chunk 255 行×(n_vocab+4)u16）+ 空间门槛 + 每家族清 tmpfs 纪律。
2. b10666 对 gemma4 多 batch 前向（n_batch<n_ctx）产出垃圾 logits（-b 256 PPL 867）；口径 -b 512 单 batch 正确（自 KL≈0.001/top 100%）。

## 六、四族证据结构（同 KL Top 排序）
- orcarouter(Qwen 27B)：94.97/91.84/89.11/87.03 — supported ×4
- Granite-4.2-8B：93.81/90.37/88.39/85.58 — 0.10 violated / 0.15 at-risk
- ling(bailingmoe MoE)：0.10/0.15/0.20 violated
- gemma-4-E4B：四档 violated
排序 ≈ orcarouter > Granite > ling ≈ gemma。候选解释：①有效容量效应 ②gemma 谱系配方 ③family-specific 量化敏感性。

## 七、M3 裁定请求
A. 家族专属 Guard Profile 细分（谱系级 guard）；B. 容量/规模维度分层（需第 4 个 3B 级 dense family 区分家族效应 vs 规模效应）；C. 候选降级（合同 v1 只冻结 KL Core + 未验证架构拒绝 + 家族按需 onboarding=M8 校准触发）。
执行人倾向 C+B 混合。请裁定：1) violated 定性；2) 合同结构；3) 第 4 family 排队与否。

## 八、追加（2026-09-03 凌晨）：orcarouter 四档位三方对比数据已备

四档目标 = 校准曲线交叉尺寸；FIT v0.2 = v0.2 栈（Refine C_role bootstrap-v0 + balanced + G2 门 delta+0 ×4 + eval-v1）。

| Tier | KL | 方法 | size GiB | macro_kld | top% |
|---|---|---|---|---|---|
| Quality | 0.05 | FIT v0.2 | 16.19 | 0.0523 | 94.90 |
| Quality | 0.05 | FIT v0.1 | 16.68 | 0.0496 | 95.00 |
| Quality | 0.05 | preset Q5_K_M | 17.91 | 0.0414 | 95.69 |
| Balanced | 0.10 | FIT v0.2 | 12.95 | 0.0976 | 92.06 |
| Balanced | 0.10 | FIT v0.1 | 13.00 | 0.0987 | 91.93 |
| Balanced | 0.10 | preset IQ4_XS | 14.05 | 0.0624 | 93.79 |
| Compact | 0.15 | FIT v0.2 | 11.42 | 0.1746 | 88.04 |
| Compact | 0.15 | FIT v0.1 | 11.43 | 0.1439 | 89.23 |
| Compact | 0.15 | preset IQ3_S | 11.57 | 0.1424 | 89.53 |
| Mini | 0.20 | FIT v0.2 | 10.35 | 0.1924 | 86.31 |
| Mini | 0.20 | FIT v0.1 | 10.50 | 0.1873 | 87.38 |
| Mini | 0.20 | preset IQ3_XXS | 10.42 | 0.1946 | 87.12 |

诚实观察：Balanced 档 v0.2 微优于 v0.1；Compact/Mini 档 v0.2 略差（bootstrap-v0 的 C_role 是 proposal 级修正，收益未兑现）；Quality 档 v0.2 因 oracle 粒度落在 16.19 GiB 非严格同尺寸。核心信息不变：preset 悬崖（Q3_K_S 离群、IQ4_XS 断档）让 11-14G 区间"同尺寸最优"只有 FIT 能做到；v0.1/v0.2 差异 << preset vs FIT 差异。请把这份对比的正式表格设计（两个主表）一并在裁定中给出。
