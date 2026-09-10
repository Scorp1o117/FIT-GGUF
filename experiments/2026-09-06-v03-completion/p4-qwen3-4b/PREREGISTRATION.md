# P4 — Qwen3-4B fresh DEV onboarding：hypothesis 预注册

日期：2026-09-06 · 状态：预注册（执行前落盘 + SHA 钉定）· 依据：planner-verdict-v03-preview.md §八
Hypothesis：**H-M3-02**（Qwen lineage 大→小，Top|KL 关系是否随 capacity 下移）

## 1. 目的（双价值）

1. 验证 H-M3-02：Qwen3 系 27B(orcarouter) → 4B 的容量下移是否系统性压低 same-top（M3 容量假设）。
2. **A1 acceptance**：`fit calibrate` 在**没有为该模型写任何专用脚本**的前提下完成全流程——这是 P2 生产线"别人也能用"的直接证明。

## 2. 固定输入（执行前钉定）

- 模型：`Qwen/Qwen3-4B`（HF，Apache-2.0，标准 Qwen3 架构，text-only）safetensors → BF16 GGUF（PR 分支 convert_hf_to_gguf.py，标准转换无定制）
- 语料：APEX-imatrix-Small.txt（同族先例）500×c512
- 运行时：tools/llama-b10666-rocm（eval-v1 钉定构建；Qwen3 是标准架构无需 PR 分支）
- 合同：fidelity-calibration-v1（455338c5…）；registry：fit.fidelity_registry.v1
- 下载与转换记录：safetensors 与 BF16 GGUF 全 SHA 钉定

## 3. 预注册预期（执行前声明）

- 主预期：Qwen3-4B 的 same-top 地板（若 validated）**低于** orcarouter 27B 的对应地板（容量假设方向性预测）；具体数值不做点预测，只记录方向。
- 流程预期：四档窗口在标准梯+≤4探针/档下可填满（Qwen3-4B 无 abliteration、KL 曲线预计较 Spark 平缓）；若出现 INSUFFICIENT_WINDOW → 诚实 candidate，同样是有效结果。
- **DEV 后果（不可撤销）**：本模型从进入校准起即为 DEV（split=dev, sealed_eligible=false），永久不得用于 G11。

## 4. 成功标准

1. `fit calibrate` 单命令完成（无模型专用脚本、无手工干预）；
2. 产出合法 bundle（admission 通过的 candidate 或 validated guard）；
3. 失败必须是合同枚举内的失败状态（诚实失败也算通过——衡量的是生产线，不是模型）；
4. 地板/曲线数据归档供 M3 容量假设分析。

## 5. 执行记录

执行：2026-09-10 13:46 → 14:24（约 38 分钟）· 单条 `fit calibrate` 命令，无模型专用脚本、无手工干预

- **bundle**：`experiments/2026-09-06-v03-completion/p4-qwen3-4b/run/`（八件套齐全）
- **overall status**：`validated`，open failures: none，`fit registry validate` → admissible
- **imatrix coverage**：252 entries, 0 missing matrices
- **阶梯**：12 档标准梯全部完成；gap probes balanced×3 / compact×2 / mini×3 / quality×3（均 ≤4/档）
- **各档 n / floor / witness**：

| tier | window | n | floor | method | witness |
|---|---|---|---|---|---|
| quality | [0.0425, 0.0575] | 3 | 0.9119 | empirical_p5 | IQ4_XS ✓ |
| balanced | [0.085, 0.115] | 3 | 0.8812 | empirical_p5 | probe-balanced-1 ✓ |
| compact | [0.1275, 0.1725] | 4 | 0.8470 | empirical_p5 | probe-compact-1 ✓ |
| mini | [0.17, 0.23] | 3 | 0.8360 | empirical_p5 | probe-mini-1 ✓ |

- **registry entry**：已入册 `f3e9d463…`（v0.3.0，status=validated），`fit registry verify` 3 entries OK

### 结果 1 — 生产线的价值被证实（A1 acceptance）

`fit calibrate` 在一个**从未为它写过任何脚本、且从未进入过 dev loop** 的模型上，用一条命令完成了
imatrix → 五域参照 → 12 档梯 → 空洞探针 → 地板 → 八件套 bundle 的全流程，且诚实产出
`validated`（不是靠复用他人地板）。成功标准 1–3 全部满足。

### 结果 2 — Contract v2（只看 KL）下 Qwen3-4B 的每档最小体积

| 档位 | KL 硬门槛 | 最小点 | 体积 | top-1（参考） |
|---|---|---|---|---|
| quality | ≤ 0.05 | IQ4_XS | 2.115 G | 92.28% |
| balanced | ≤ 0.10 | probe-balanced-1 | 1.974 G | 88.53% |
| compact | ≤ 0.15 | probe-compact-1 | 1.742 G | 86.08% |
| mini | ≤ 0.20 | probe-mini-3 | 1.581 G | 83.56% |

### 结果 3 — H-M3-02 容量假设：方向性成立

Qwen lineage 27B（orcarouter）→ 4B 的容量下移，**在 4 个档位全部**压低 Same-top 地板：

| 档位 | 27B 地板 | 4B 地板 | 差 |
|---|---|---|---|
| quality | 94.75% | 91.19% | −3.56 pt |
| balanced | 91.18% | 88.12% | −3.06 pt |
| compact | 88.94% | 84.70% | −4.24 pt |
| mini | 85.03% | 83.60% | −1.43 pt |

预注册只声明方向、不做点预测 —— 方向 4/4 命中。**注意**：自 Contract v2 起该地板只是
参考值，不再是门槛，因此这是"同 KL 下 top-1 一致率随容量下降"的观测结论，
而不是一个判定条件。

### 执行说明（运维）

首轮执行（2026-09-06 22:21 起）实为 **ntfs3 内核 panic** 打断，非进程被杀 —— 原委与修复见
`docs/execution-profile.md` §5 与 CHANGELOG 的 v0.3.0 ntfs3 条目。本次重跑全程热循环在
tmpfs（源权重、参照、日志、工件），NTFS 仅只读与收尾批量拷贝。
