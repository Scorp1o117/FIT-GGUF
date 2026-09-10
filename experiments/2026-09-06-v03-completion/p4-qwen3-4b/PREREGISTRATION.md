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

（执行后填写：bundle 路径、overall status、各档 n/floor/witness、耗时、registry entry）
