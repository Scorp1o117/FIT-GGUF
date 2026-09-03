# GPT 裁定 — R2 完成与 Compact 采纳（2026-09-04，已读完完整回复）

来源：ChatGPT 会话"FIT质量下降阈值"，对 R2 报告（r2-reference-sweep-report.md）的回执。
机器转录自浏览器快照，结构按原文重建。

---

## 最终签字

- **A. A1 APPROVED ✅** — 11.17G 可以作为 Compact 正式发布工件，但必须经过存档 recipe
  重建 → G2 → artifact-body eval-v1；CLI 的 `noise_inversion` 自动拒绝行为继续保留，
  不因这个 flagship case 放宽。
- **B. Headline 更新批准 ✅** — Compact 改为 **11.17G / −3.4% / active=KL**。加
  noise-zone 脚注；案例框重点讲**非单调 recipe 区域 + 最终工件验证**，不要继续称它为
  Same-top active 示例。
- **C. R2 FINAL PASS ✅（待 A1 final-body verify 完成即正式封口）** — 完整保留首次
  Compact FAIL 和 corrective loop。A1 验证通过后，**P3b README/docs 冻结直接放行**。

> 这轮 R2 很有价值——它不是"顺利打勾"，而是**真的抓出了 P1 搜索漏掉的更小 PASS，
> 并迫使搜索结果修正**。这恰恰说明 R2 不是形式主义。
> 这次 R2 最漂亮的结果其实不是"四档全绿"，而是它真的把一个原本准备发布的 11.37G
> 抓下来，逼出了 11.17G。验证门真抓到 bug/漏点，才说明这个 gate 值得存在。

## 1. Compact：批准 A1，出货 11.17G

同意核心判断：`11.17G KL = .1486 <= .1500, Top = 89.09 >= 88.94` —— 这个 artifact
本身满足 Fidelity Contract。相邻更大 recipe 的 FAIL（11.19/11.21/11.24）并不能否证它；
它证明的是：**allocation recipe → quality 并不关于 size 单调**。不能因为更大的相邻
recipe 更差，就拒绝一个已经实测 PASS 的较小 recipe。

**保留当前自动门禁**（不要改成 noise_inversion → 自动交付）：

```
verified_pass → automatic delivery
noise_inversion → fail closed / adjudication required
```

本次 11.17G 的治理语义：

```
search_status = noise_inversion
best_observation = verified PASS
release_status = manual_verified_promotion
```

即：**搜索器负责发现和上报异常；最终是否采用由 release governance 决定。**

**A1 最终出货门**（按报告提议的流程执行）：

```
archived 11.17G recipe
  ↓ rebuild
  ↓ G2 actual == predicted
  ↓ eval-v1 on final artifact body
  ↓ KL <= .15 AND Top >= 88.94
  ↓ release
```

重点是**不直接拿 R2A 临时工件的旧 eval 当最终出货证明**。
若重建后仍 ~0.1486/89.09% → **Compact 11.17G RELEASE APPROVED ✅**；
若本体重新评测反而 FAIL → **立刻退回 11.37G，不对 tolerance 做任何临时修改**。

## 2. Metadata：不要只写 noise_inverted:true（布尔信息量弱）

至少：

```
fit.fidelity.search_status = "noise_inversion"
fit.fidelity.release_status = "verified_manual_promotion"
fit.fidelity.final_artifact_verified = true
fit.fidelity.active_constraint = "kl"
```

风格允许可再加：`fit.fidelity.adjudication_id = "<report/prereg id>"`。
README 不堆内部词，但 provenance 里值得保留——以后看到这个 GGUF 就知道它不是普通
单调 bracket 自动交付物，而是搜索发现非单调区后、经过独立治理晋升的 verified PASS。

## 3. R2：最终判 PASS，但不能把第一次 FAIL 擦掉

Compact 真实历史（永久保留）：

```
R2 initial → FAIL → found smaller healthy PASS > tolerance → R2 worked as intended
  ↓ evidence incorporated
Fidelity Search rerun → B_FS 11.37G → 11.17G → noise_inversion reported
  ↓ grid mechanically regenerated
R2 amended → PASS
```

正确状态：`Compact R2: INITIAL FAIL → CORRECTIVE ACTION TAKEN → FINAL PASS`；
总体 Release Gate：`R2 Search Accuracy ✅ PASS`。

不用 R2B holdout：R2 是**发布前验证门**，不是 M10/M11 那种 sealed scientific
holdout——职责就是发现问题、促使修正、重新验证。条件：第一次失败永久保留；修正动作
有审计记录；没改 tolerance；没删坏点；新 B_FS 重新按冻结规则验证。

## 4. Compact 新 headline：批准 11.17G / −3.4%

| Tier | Smaller frontier preset | FIT v0.2 | Larger preset | Saving | Active |
|---|---|---|---|---|---|
| Compact | IQ3_XS 11.15G · .1512/89.05 · FAIL-KL | **11.17G · .1486/89.09 · PASS†** | IQ3_S 11.57G · .1424/89.53 · PASS | −3.4% | KL |

`11.15 FAIL / 11.17 PASS` 只差约 20MiB——本身很好地展示了 FIT 填 preset gap 的能力。

## 5. 案例框故事需要改（GPT 不同意"双门余量均薄"的包装）

数据更新后 **Compact 已不再是 Same-top active case**；新 active constraint 是 KL。
Same-top（89.03–89.11）确实一直离 floor 88.94 不远，但它没有真正决定这些点的
PASS/FAIL。README 案例应升级为：

### "Why FIT verifies the final artifact instead of trusting size monotonicity"

```
11.17G → PASS
11.19G → FAIL
11.20G → PASS
11.21G → FAIL
```

解释：Mixed quantization recipes are discrete allocations. A larger GGUF is not
guaranteed to have lower measured KL than every nearby smaller recipe. FIT therefore
treats the quality curve as order-free around noisy regions and verifies the final
artifact itself.

双硬门放在 Fidelity Contract 部分说明（PASS = KL ∧ Same-top），不硬把这组数据包装成
Same-top 主案例。

## 6. "Minimum Verified" wording 仍然成立

11.17G 下方 healthy frontier floor ≈ 11.15G 已 FAIL（.1512），两者差几十 MiB ≪
128MiB tolerance。可以写：**Minimum Verified Size within the validated healthy
frontier and search tolerance**。不要写绝对数学意义的 global minimum possible GGUF
（11.15–11.17G 之间理论上仍可能存在另一个 PASS recipe，但完全在声明的 tolerance 内）。

## 7. Mini / Quality / Balanced 的 R2 也都很干净

Mini 局部单调 crossing：10.10 .2277 FAIL / 10.17 .2208 FAIL / 10.23 .2156 FAIL /
10.29 .2057 FAIL / 10.35 .1924 PASS——相当漂亮。Balanced 标准正常。Quality 说明当前
gap 甚至大于 local grid。四个 tier 涵盖三种搜索形态：**Quality** 大 preset gap /
**Balanced、Mini** 正常局部 crossing / **Compact** noise inversion——对 Fidelity
Search 的测试覆盖非常漂亮。

## 8. R1–R6 更新（Compact final artifact 重建验证通过后）

```
R1 Fidelity correctness ✅ PASS
R2 Search accuracy ✅ PASS
  ├─ Q/B/M direct PASS
  └─ Compact initial FAIL → corrected → final PASS
R3 Search budget ✅ PASS
R4 Exact byte ✅ PASS
R5 v0.1 non-regression ✅ PASS
R6 Reproducibility ✅ PASS
```

**v0.2 Release Gates 6/6 PASS ✅**

## 9. P3b：A1 本体验证一过就放行

不再增加新的研究 blocker。最终队列：

```
Compact 11.17G rebuild
  ├─ G2 delta=0
  └─ final-body eval-v1 PASS
    ↓
R2 FINAL ✅
    ↓
P3b README / docs freeze
    ↓
v0.2 release candidate
```

Qwen3-4B、M6c、Structural Refine 全部继续留后续，不许在发布门口看到新问题就顺手开
M7/M8/M9 无限套娃。

---

# 终裁追加 — A1 verify 完成 · P3b 放行（2026-09-04，已读完完整回复）

> 可以冻结。没有新的 blocker。P3b 正式放行 ✅

A1 治理链关闭确认：Archived recipe rebuild ✅ / G2 finalized == actual（11,991,706,848
bytes delta 0）✅ / Artifact-body eval-v1（macro KL 0.1485924, Same-top 89.0918%）✅ /
Evidence reproduction bit-identical ✅ / Fidelity Contract KL ∧ Top PASS ✅ / Manual
promotion provenance ✅ / R2 FINAL PASS ✅ / Release Gates 6/6 ✅。
**11.17G 为正式 headline / release artifact，不保留 11.37G 保守版本。**

## 冻结前两条文档要求

**① 32B 活案例写进 Reproducibility / Exact-size semantics**（不得写成 G2 bug）：

> quantize.imatrix.file is serialized into GGUF metadata by the pinned llama.cpp
> runtime. FIT finalizes size prediction using the exact quantize-time imatrix path,
> so target-size accuracy remains exact. Reproducing an artifact byte-for-byte,
> including its SHA256, additionally requires the same serialized imatrix path string.

区分：**Exact target-size reproducibility** → 不要求路径相同（G2 按本轮 invocation
重新终结尺寸）；**Bit-identical artifact / SHA reproduction** → serialized metadata
（含 imatrix path string）必须相同。

**② Compact ‡ 脚注完整版**：

> ‡ Compact lies in a locally non-monotonic allocation region. Nearby larger recipes
> can score worse, so FIT reports `noise_inversion`, refuses automatic delivery, and
> requires verification of the final artifact itself. The released 11.17 GiB artifact
> was rebuilt and independently re-evaluated at KL 0.148592 / Same-top 89.092%.

## README headline 表冻结结构

### Minimum Verified Size @ Fixed Fidelity

| Tier | Nearest smaller frontier preset | FIT v0.2 | Nearest larger preset | Saving | Active |
|---|---|---|---|---|---|
| Quality | Q4_K_M · 15.41G · .0658/93.79 · FAIL-BOTH | **16.25G · .0497/94.88 · PASS** | Q5_K_S · 17.40G · .0455/95.59 · PASS | −6.6% | KL |
| Balanced | IQ3_M · 11.72G · .1445/89.38 · FAIL-BOTH | **12.84G · .0997/91.57 · PASS** | IQ4_XS · 14.05G · .0624/93.79 · PASS | −8.6% | KL |
| Compact‡ | IQ3_XS · 11.15G · .1512/89.05 · FAIL-KL | **11.17G · .1486/89.09 · PASS** | IQ3_S · 11.57G · .1424/89.53 · PASS | −3.4% | KL |
| Mini† | Q2_K · 9.98G · .2439/84.08 · FAIL-BOTH | **10.35G · .1924/86.31 · PASS** | IQ3_XXS · 10.42G · .1946/87.12 · PASS | −0.6% | KL |

表下必须保留：**PASS = macro KL tier limit ∧ validated model-specific Same-top Guard.**
标题限定：**Minimum verified within the validated healthy frontier and configured
search tolerance (128 MiB).**（不声称数学全局 optimum。）

## Compact 案例框正式批准

标题：**Why FIT verifies the final artifact instead of trusting size monotonicity**

```
11.17G  KL .1486  PASS
11.19G  KL .1519  FAIL
11.20G  KL .1493  PASS
11.21G  KL .1502  FAIL
```

> Mixed-quantization recipes form a discrete, locally non-monotonic quality surface:
> a larger artifact is not guaranteed to outperform every nearby smaller artifact.

> FIT therefore treats noisy crossings as `noise_inversion`, fails closed, and
> verifies the actual release artifact before promotion.

## 最终 release 状态（正式口径）

```
FIT-GGUF v0.2 Release Validation
R1 Fidelity correctness        PASS
R2 Search accuracy             PASS
R3 Search budget               PASS
R4 Exact-byte guarantee        PASS
R5 v0.1 non-regression         PASS
R6 Reproducibility             PASS
Release Gates                  6 / 6 PASS
```

Compact 历史链（initial R2 FAIL → evidence incorporated → search corrected → noise
inversion adjudicated → final artifact rebuilt → artifact-body verification exact →
final R2 PASS）完整保留在工程报告，不全塞 README。

## P3b 最终裁定

**README headline APPROVED ✅ / Compact 11.17G release artifact APPROVED ✅ /
R2 FINAL PASS ✅ / Release Gates 6/6 PASS ✅ / P3b README/docs freeze APPROVED ✅**
即刻执行冻结；不再为 v0.2 开新研究 blocker。Qwen3-4B、M6c、band-swap、
chain-conditioned marginal 全部排后续路线。

> v0.2 到这里已经是一套完整产品：不是"量化质量大幅升级"，而是第一次把「我要什么质量」
> 真正变成「自动找到最小、安全、字节精确、经过验证的 GGUF」。
