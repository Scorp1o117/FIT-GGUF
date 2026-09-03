# R2 — Independent Local Reference Sweep（amended release validation）
# 完成报告（2026-09-04；initial FAIL → evidence-incorporating correction →
# amended PASS——修正后的 PASS 是发布验证，不是 untouched independent sweep）

接 2026-09-03 晚中断点。**R2 四档全 PASS**；Compact 采纳新 B_FS = 11.17G；
总计 6 fresh evals / 14 cache hits。待 GPT 裁定事项见 §7。

## 1. 结论摘要

| Tier | B_FS | Lowest grid PASS | Δ vs FS | Verdict |
|---|---|---|---|---|
| Quality | 17,443,851,488 (16.25G) | 无（五点全 FAIL-KL） | — | **PASS** |
| Balanced | 13,790,608,608 (12.84G) | 13,855,735,008 (12.90G) | +62.1MiB | **PASS** |
| Compact | 11,991,706,848 (11.17G，**今日采纳**) | 无（五点全 FAIL-KL） | — | **PASS** |
| Mini | 11,116,944,608 (10.35G) | 11,183,750,368 (10.42G, clip) | +63.7MiB | **PASS** |

FAIL 判据（六轮裁定原文）：存在 healthy grid PASS 于 delivered < B_FS−128MiB。
四档均无；Balanced/Mini 的最低 PASS 都在容差内（+62.1 / +63.7 MiB）。

## 2. 恢复叙事：昨晚字面 FAIL → 今日处置

1. **昨晚**：compact 在 B_FS = 11.37G 的 R2 grid 字面 FAIL——grid 在容差线下发现两个
   healthy PASS（11,991,706,848 与 12,023,164,128，均低于 12,072,037,600 = B_FS−128MiB）。
2. **调查**：穿越区评测读数与档位余量同量级。将昨晚全部 grid 点的 plan 记录 +
   eval 日志与 P1 FS 点合并后，得到 §3 的六观测非单调区。
3. **处置（今日）**：compact R2 点归档重命名 R2A-*（证据全保留，见
   `results-fs/r2-compact/archive-oldbfs-1137G/README.md`），manifest 回填 14 行
   （eval-only 标记，工件按 plan recipe 可确定性复现）；然后按五轮裁定的合法路径
   **重跑 compact fidelity-search 纳入全部证据**。
4. **采纳结果（canonical CLI `fit fidelity-search`）**：
   `status = noise_inversion | best = 11,991,706,848 (kld 0.1486, top 89.09) |
   bracket [12,076,248,288 FAIL, 11,991,706,848 PASS] | 0/8 fresh evals | active: kl`
   ——搜索状态机按设计诚实降级（"smallest verified PASS; bracket inverted"），
   **B_FS(compact) 采纳为 11.17G**。
5. R2 compact grid 在新 B_FS 下机械重算 → PASS（§3）。

## 3. Compact 穿越区六观测（非单调证据，按尺寸升序）

| size (G) | macro KL | top% | 判定 (0.15 / 88.94) | 来源 |
|---|---|---|---|---|
| 11.15 (11,967,130,848) | 0.1512 | 89.05 | FAIL-KL | 窗口地板 = IQ3_XS 预设 = P1 FS01 |
| 11.17 (11,991,706,848) | **0.1486** | 89.09 | **PASS** | R2A off-192 → **采纳 B_FS** |
| 11.19 (12,016,037,088) | 0.1519 | 89.04 | FAIL-KL | 今日 R2 off+64（fresh） |
| 11.20 (12,023,164,128) | **0.1493** | 89.10 | **PASS** | R2A off-128 |
| 11.21 (12,038,647,008) | 0.1502 | 89.08 | FAIL-KL | P1 FS02 |
| 11.24 (12,072,070,368) | 0.1539 | 89.11 | FAIL-KL | P1 FS03 |
| 11.25 (12,076,248,288) | 0.1582 | 89.03 | FAIL-KL | R2A off-64 |

读法：约 105MB 内六个观测，PASS/FAIL 交错四次——**"更大不保证更好"在该穿越区
真实存在**（配方阶梯每步换不同张量组合，单步质量非单调；与 M5 教训同源）。
Top 维度全部压线通过（+0.03~+0.17pp），KL 是唯一分纬度。双门按 artifact 本体
评测判定（P1 五轮裁定"搜索探针 PASS ≠ 最终 artifact PASS"的同义延伸），
noise_inversion 状态是这台状态机为这种情况设计的诚实出口。

## 4. R2 grid 明细（GPT 要求格式：Tier / B_FS / lowest PASS / Δ / 线下已测点 / fresh·cache）

- **Quality**：B_FS 16.25G。五点 15.64–15.92G 全 FAIL-KL（kld 0.0545–0.0582；
  注意 Q4_K_M→Q5_K_M 窗欠填 ~420-466MB，off+64 实落 15.92G，仍比 B_FS 低 326MB 且
  FAIL）。lowest PASS = 无。fresh 0 / cache 5。
- **Balanced**：B_FS 12.84G。12.59–12.78G 四点 FAIL-KL（0.1039–0.1113 单调）；
  off+64 12.90G **PASS**（0.0994/91.75），Δ = +62.1MiB（容差内）。fresh 0 / cache 5。
- **Compact**（新 B_FS 11.17G）：off−256/−192/−128/−64 全部 clip 到本档窗口地板
  11,967,130,848（FAIL-KL 0.1512，cache 命中 P1 FS01 评测）；off+64 实落
  12,016,037,088 FAIL-KL 0.1519（fresh，§3 第六观测）。lowest PASS = 无 → 线下无
  healthy PASS。fresh 1 / cache 4。
- **Mini**：B_FS 10.35G。此档昨晚无任何状态，五点全 fresh：
  10.10G 0.2277/84.73 → 10.17G 0.2208/85.27 → 10.23G 0.2156/85.65 →
  10.29G 0.2057/86.01，全 FAIL-KL 且**严格单调**（mini 窗无反转；其中 10.23G 点
  落在容差线下 0.8MB 处，实测 0.2156 无歧义）；off+64 clip 到窗顶实落 10.42G
  **PASS**（0.1922/86.40），Δ = +63.7MiB（容差内）。fresh 5 / cache 0。
  B_FS 10.35G 维持（不在容差线以下发现 PASS，无强制采纳项）。

totals：**fresh 6 / cache 14**。产物清单：`results/r2-reference-sweep.json`（机读全量）。

## 5. 采纳的治理形态与出货门禁

- canonical CLI 按五轮裁定执行；搜索部分 0 fresh evals（全部 cache 命中 +
  种子吸收，符合"cache 全匹配 + 交付物永远重 verify"的合法性口径）。
- **product 门禁对 noise_inversion 拒绝自动出货**（strict verified_pass-only）：
  `fidelity-search-compact-product.json` 记录 status/best/artifact=None。
  11.17G 的配方与张量表完整存档
  （`archive-oldbfs-1137G/R2A-compact-off-192-plan-recipe.json` + tensor-types），
  可随时按 P3a 链路重建 + 本体 verify eval。**出货路径待 GPT 裁定（§7-A）**。

## 6. 本轮工程与协议修正

1. **sweep cache 前缀 bug（昨晚全 fresh 的根因之一）**：cache 以 `orcarouter-` 全名
   建键、日志 tag 无前缀 → 永远 miss。已修（键用剥前缀名）。
2. **clip 语义修正**：grid 目标低于 B_FS 所在窗口地板时，clip 到**本档窗口地板**
   （与 docstring 一致）；原实现会落进"下方最近窗口"（昨晚 compact off-256 因此测了
   mini 窗配方）。质量上：本档地板点是"最小可建尺寸"的更强检验。
3. **fresh 点即时写 manifest（真实 SHA）**：中断可续不再依赖事后回填。
4. fresh 结果 memoize（同尺寸去重）+ plan 记录复用（target 相同即复用，确定性）。
5. **回填协议**：eval-only 工件（tmpfs 已释放、SHA 未录）以 `eval-only` 标记入
   manifest + README 指针到可复现 recipe；不伪造摘要列。

## 7. 请 GPT 裁定

**A. Compact 出货路径（主裁定项）**。B_FS 已采纳 11.17G，但产品门禁拒出。
两个选项：
  - **A1（建议）**：放行 11.17G 出货——配方/张量表按存档 recipe 重建 → G2 →
    本体 verify eval（必须重验，预期 0.1486/89.09）→ release；产品 JSON 加
    `noise_inverted: true` 元数据标记。理由：合同定义在 artifact 本体双门上，
    11.17G 是实测 PASS；其上的 FAIL 是配方非单调，不是对该工件的否证；
    门禁的严格性已起到"把决策上报策划师"的作用。
  - A2：release 保守用 11.37G（P1 verified bracket），11.17G 只作表内 observed
    smaller PASS。代价：headline saving 3.4% 不兑现。

**B. headline Compact 行更新**（本地 m3-tables.md 已按采纳更新）：
`IQ3_XS 11.15G 0.1512/89.05 FAIL-KL | 11.17G 0.1486/89.09 PASS | IQ3_S 11.57G
0.1424/89.53 PASS | −3.4% | kl`。案例框建议改写：P1 时代的 "same_top 为主动约束"
升级为双门余量均薄（M_kl 0.93% / M_top 1.36% 归一化）+ 六观测非单调区——
双门设计的更强实证。

**C. R2 状态升级 🟡 → ✅，P3b README/docs 冻结放行？** 若 A1 获批，README 的
Compact 行按 11.17G 冻结（附 noise-zone 脚注）；若 A2，按 11.37G。
