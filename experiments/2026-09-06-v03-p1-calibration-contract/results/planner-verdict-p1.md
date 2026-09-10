# Planner verdict — P1 fidelity-calibration-v1 预注册评审（2026-09-06）

来源：ChatGPT 策划师会话，对 P1 预注册稿（ZCode 执笔）的裁定。本文件为誊录要点，原文在会话内。

## 总结论

**P1 DESIGN APPROVED ✅ / FREEZE CANDIDATE — FROZEN pending 4 项 normative 修订。** 修完用现有 Spark/orcarouter 数据做一次**零新增 eval 的 contract replay**（能生成合法 calibration record、窗口合规、地板符合新合同、无意外 candidate）→ 正式签 **P1 FROZEN** → 立即进 P2 fit calibrate，不再插研究项。

## §13 五裁定点

1. **Promotion 不追加 Search verified PASS ✅**（防循环依赖：Guard validated ≠ 四档 reachable；tier 后来 NOT_REACHABLE 是量化前沿事实，不否定 Guard）。**但新增 calibration-layer witness**：每档除 n≥3 外必须 ≥1 个校准观测满足 `KL ≤ anchor ∧ same-top ≥ floor`（字段 witness {point_id, macro_kl, macro_same_top, pass}），否则该档 candidate——"证明合同没把自己的校准证据全挡在门外"。
2. **空洞探针 ≤4/档 ✅**，预算语义钉死：只计"artifact 成功产生 + eval-v1 实际执行"的 fresh probe；cache hit/重试不计；耗尽 → INSUFFICIENT_WINDOW → candidate，禁止自动加预算。
3. **地板样本冻结于校准期 ✅ 严格执行**：ladder + gap probes → floor set FROZEN → derive Guard → Search；Search 点**永不静默回填**；重校 = 新 calibration revision（新 record → 新 guard hash → 更新 registry entry），旧记录保留。
4. **expected_valid_tokens informational：否决 ❌（freeze blocker）**——该字段在已冻结 eval-v1 里有强语义（observed vs expected 不一致 → FAIL）。保留为 **authoritative evaluator-derived pin**；tokenizers 估算另开 `token_count_estimate {value, method, informational: true}`。
5. **合同本体不放 experiments/MD：否决 ❌**——machine-readable canonical contract 放 `src/fit_gguf/contracts/fidelity-calibration-v1.json`（A3 canonical JSON v1，其 SHA = calibration_contract_sha256）；experiments 的 CONTRACT.md 保留为 rationale/prereg archive，写明 machine_contract_sha256。

## 四项 normative 修订（freeze blocker）

- **A. 窗口局部性**：我的四段 bin（(0,.055] 等）不是 boundary calibration——Quality 的 KL .007/.015/.030 全被算进样本，floor 会被远离边界的高质量点抬高。**恢复 M2 规则 W(K)=[0.85K, 1.15K]**：Quality [.0425,.0575] / Balanced [.0850,.1150] / Compact [.1275,.1725] / Mini [.1700,.2300]；窗内不足 n=3 → gap probes，不拿远处预设凑数。
- **B. Same-top 样本值 = eval-v1 的 macro_same_top 存储全精度值**：Calibration 不重新实现 aggregation（eval-v1 result is authority），防显示 round 后再平均的微漂移。
- **C. trunc4 用 Decimal 语义**：`Decimal(canonical_decimal_string(v)).quantize(Decimal("0.0001"), ROUND_DOWN)`——裸 `floor(v*10000)` 会被 binary float ε 多降一格，破坏 canonical replay。边界 fixture：0.9350 / 0.9349999… / 0.8503 / 0.8894。
- **D. n 计数 unique calibration observations**：dedup key = artifact_sha256；同 artifact 重评 n+=0；同尺寸不同配方 = 独立观测（留 plan_hash/recipe_hash 审计）。防 A1 自动化后 cache/replay 虚增 n。

## 其他裁定

- **毒 set 版本纪律**：v1 frozen poison set = {Q3_K_S, IQ2_XS} 写死；改毒 set = contract semantics change → 新 digest/amendment 或 v2，不得同 SHA 静默追加。
- **§6 整体 status 暂批准**（四档全 n≥3 → validated，否则 candidate；不引入 per-tier registry status，不重开 A3）。
- **测试夹具增 5 个**：window locality（KL .01 不计入 Quality 窗、.049 计入）/ duplicate observation（同 SHA×3 → n=1）/ calibration witness（n=3 无 PASS 点 → candidate）/ expected_valid_tokens（authoritative mismatch→FAIL，estimate→informational）/ decimal truncation 边界。

## 冻结路径

4 修订并入 → contract replay（零新增 eval）→ **P1 FROZEN** → 立即 P2 fit calibrate。

## P1 核心问题（GPT 表述）

"什么样的 calibration evidence，有资格生成一个 Registry 可以接纳的 validated exact-model Guard？"
