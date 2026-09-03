# Prereg — M2 第三 dense 家族校准：google/gemma-4-E4B（2026-09-03）

承接 V0.2 主预注册（2026-09-02）与 planner-verdict-gemma-g2.md 的签字链。
本文件冻结 gemma 家族执行口径；偏差一律走修正案条目，不许事后改。

## 1. 家族与签字链

- 主人裁定（2026-09-02）："Mistral 太老"，第三 dense 家族 = gemma-4-E4B 或 E2B；
  执行人核查后选 E4B，GPT 补签 APPROVED（planner-verdict-gemma-g2.md §2）。
- 执行前硬门（GPT ruling）：**Text-Tower Eligibility Record** —— 主 GGUF 必须
  text-only。已过：`results-gemma/text-tower-eligibility.json`
  （text=720 / vision=0 / audio=0 / mmproj 不存在）。
- BF16 来源：google/gemma-4-E4B（base 非 it，Apache-2.0，非 gated），
  safetensors 15,992,595,884B（content-length 精确一致 + 头自洽 2130 张量全 BF16）；
  master 转换器 7798007a（conversion/gemma.py）→ `gemma-4-E4B-bf16.gguf`
  15,053,078,400B（720 张量）。

## 2. 冻结评估口径（与 eval-v1 一致，工具链不变）

- 工具链：tools/llama-b10666-rocm（冻结 b10666，不变）。
- 参照：BF16 五域 `--kl-divergence-base`（写路径，-ngl 99 -t 16 -c 512 -b 512）；
  五域 = wiki_test / wiki_valid / chinese / code / agent_chat（eval-data 65,536B 切片）。
- 参照有效性判据（新增，ENOSPC 事故教训）：文件必须满足
  `size == 20 + n_chunk·n_ctx·4 + n_chunk·255·(n_vocab+4)·2` 的行数完整算术。
- 点位：preset 粗曲线 12 点 =
  IQ2_XXS IQ2_XS IQ2_M IQ3_XXS IQ3_XS IQ3_M IQ4_XS Q3_K_M Q4_K_M Q5_K_M Q6_K Q8_0；
  每点五域 KL eval（读路径，同一冻结旗标）。FIT 填窗点随后按 M2 amendment 流程
  （fit analyze/plan/quantize，G2 重终结门）。
- imatrix：与 granite/ling 同源 APEX 语料（APEX-imatrix-Small.txt，500 chunks × c512），
  llama-imatrix（b10666），GGUF 格式。

## 3. 修正案 G-A1（执行前）：低_bit 预设的 attn_k 提升偏差

- 事实（两次独立采集复现）：b10666 的 imatrix 采集对 gemma4 **确定性地**缺失
  `blk.{24..41}.attn_k.weight` 共 18 条（其余 342 条齐全；500 chunks 跑满、
  PPL 4.90 正常收敛；缺失集与 sliding/full 层分布无关，为顺序截断模式）。
- 影响面：`tensor_requires_imatrix` 清单 = {IQ3_XXS, IQ2_XXS, IQ2_XS, IQ2_S,
  IQ1_M, IQ1_S}（llama-quant.cpp:803）；本梯子中 IQ2_XXS / IQ2_XS / IQ3_XXS
  三点会因缺条目拒答（"Missing importance matrix … bailing out"，实测复现）。
- 处置：三点量化时 `--tensor-type-file` 把缺失条目的 18 个 attn_k.weight
  提升为 **Q4_K**（脚本按 imatrix 实际缺失集动态生成；这是 llama.cpp 对
  无 imatrix 张量的同款回退方向）。其余 9 点为纯 preset。
- 记录义务：manifest 行不变名；分析输出表对三点加脚注；GPT 汇报必须携带本条。

## 4. 事故记录（预防性纪律）

- `/dev/shm` 残留 m2-ling 32G 导致 gmma 参照 ENOSPC 静默截断（文件页对齐、
  进程仍打印 Final estimate —— 写路径死而输出未死）。已清 32G 并立规矩：
  每家族跑完立即清 tmpfs；参照一律过行数完整校验；脚本带空间门槛
  （BF16 未驻留需 40G，驻留后需 24G）。
- 观察项（不影响本校准）：b10666 对 gemma4 的多 batch 前向（n_batch < n_ctx）
  产出垃圾 logits（-b 256 时 PPL 867）；本口径全程 -b 512 = n_ctx 单 batch，
  自 KL 测试 KLD≈0.001 / top 命中 100% 佐证正确性。

## 5. 交付物

- 三模型 dense 对比表（GPT 模板）：
  `Tier | Top@KL | local n | P5 | P10 | dense guard`
  Tier=Quality/Balanced/Compact/Mini，guard=93/90/88/85；
  家族行：Family: Gemma 4 / Architecture class: dense text tower /
  Model scale: ~8B stored / E4B effective configuration。
- 若四档不打穿 → dense-candidate 升 same-top-guard-dense-v1，进 M3 决策。
