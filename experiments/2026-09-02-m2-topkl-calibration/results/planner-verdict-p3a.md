# GPT 五轮裁定 — P3a CLI 产品路径（2026-09-03 傍晚）

## 最终裁定

**P3a PASS ✅ —— `fit fidelity-search` 定为 v0.2 canonical CLI**（优于原设想的
`fit quantize --fidelity-tier`：Fidelity Search 本质是"多次 quantize + 多次 eval +
括号收敛 + 最终再生 + 交付物验证"，独立子命令让用户对计算成本和行为预期更清楚）。

**Compact 空 cache 冒烟全链打穿**：validated Guard resolution ✅ / historical seed
reuse ✅ / 0 fresh eval legal ✅ / healthy frontier ✅ / order-free bracket ✅ /
exact-byte final planning ✅ / G2 delta=0 ✅ / artifact-body eval ✅ / contract
verification ✅ / final delivery ✅。

关键确认：**0 fresh evals 是合法行为**——只要 cache key/provenance 全匹配，且最终交付
工件永远重新 verify，搜索阶段完全可以用历史点闭合括号（这正是 cache 的价值）。
"搜索探针 PASS ≠ 最终 artifact PASS"——交付 GGUF 自己重跑 eval-v1 才是真闭环。

## 唯一加固项：poison taint 自动传播（fail-closed）

工程发现：名字完全正常的 FIT artifact 也可能因生成窗口用了毒 preset 而不适合做
healthy-frontier 种子。正确性不能依赖用户手工 `--exclude-seed`：

```
seed provenance → source window (lower/upper anchor)
  → known_outlier / poison_transition / unhealthy_frontier
  → tainted seed → NOT ADMITTED BY DEFAULT
```

冻结状态：P3a implementation ✅ / CLI command surface ✅ FROZEN / CLI search
semantics ✅ FROZEN / **seed provenance taint default ⚠️ MUST be fail-closed**。
（已实现：executor 逐工件写 seed-provenance.jsonl，load_seeds 自动排除毒窗产物；
历史 FIT-A-IQ3S-v01/v02、FIT-V2-Compact 已回填 provenance。--exclude-seed 保留为人工补充。）

## P2 批准（两处修改）

1. **Quality anchor 改用 Q5_K_S**（不是 Q5_K_M）——四个 anchor 与主表 B 的四档 native
   baseline 对齐：Q5_K_S / IQ4_XS / IQ3_S / IQ3_XXS。A/B 两表天然对应：
   B 表 = "达到该档至少需要多少"；A 表 = "在 Native 达到该档的字节预算上三方质量对比"。
2. **表 A 纯 Size Mode**：`fit plan --target-bytes <EXACT_NATIVE_BYTES> --policy balanced
   --refine-profile band`，不让 --fidelity-tier / Guard floor / Fidelity Search 参与
   allocation；Tier 只用于事后报告"该工件满足哪档"。
   Table A = Size Mode；Table B = Fidelity Mode——两表展示两个产品入口。
3. **v0.1 也必须 exact-byte**：历史 P4 工件 actual_bytes ≠ native anchor bytes 就不能用，
   必须用冻结的 v0.1 allocator 重新生成 exact-target 工件。byte delta = 0，一字节不能差。

## R2 批准（grid 修改）

- 固定五点：**B_FS − 256 / −192 / −128 / −64 / +64 MiB**（B_FS 自身不算 grid 点，
  它已由 final verification 证明）。
- 判定：存在 healthy PASS 且 B < B_FS − 128MiB → **R2 FAIL**；否则 PASS
  （B_FS−64MiB 偶然 PASS 仍 PASS，在搜索精度内）。
- cache hit 可用（前提 eval-v1/guard/model/imatrix/provenance hash 全兼容）。
- Mini：B_FS−256MiB 掉出 healthy frontier 就 clip 到边界，报告
  "R2 validated within healthy frontier"；不跑毒窗口。
- Compact：sweep 必须同时查 KL + Same-top 两门。

## 队列

```
P1 Fidelity Search flagship ✅ PASS
P3a Production CLI ✅ PASS（poison-taint propagation 🔧 已完成实现+回填）
P2 Exact Fixed-Size Table A 🚧 NEXT（anchors Q5_K_S/IQ4_XS/IQ3_S/IQ3_XXS）
R2 Independent Local Sweep ⏳ AFTER P2
P3b Release docs / README ⏳
```
