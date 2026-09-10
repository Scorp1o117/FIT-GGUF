# Planner verdict — A3 Fidelity Registry v1 设计评审（2026-09-06）

来源：ChatGPT 策划师会话，对 A3 设计冻结提案 v1.0（ZCode 执笔）的裁定。本文件为 ZCode 誊录要点，原文在会话内。

## 总结论

**A3 Registry v1：DESIGN APPROVED / FREEZE PENDING 6 项修订。** 修完可直接宣告 `fit.fidelity_registry.v1 FROZEN`，无需再开架构讨论。大方向（三层信任架构）完全符合前一轮裁定；"真正需要修的不是大架构，而是把'官方 registry 已经接受什么''runtime resolver 信任什么''大文件从哪里拿'三件事彻底拆开。"

## 四个裁定点

1. **Bootstrap calibration：批准但切换语义改**——否决"P1 冻结后 bootstrap entry fail-closed"（会让 orcarouter/Spark 两个已 validated entry 突然失效）。原则：**admission-time governance ≠ runtime trust**；bootstrap = **永久 grandfather 的 legacy_bootstrap_v0**（仅限 orcarouter+Spark 两个 source SHA；未来 add 不许声明该 basis）。Registry 必须快照确定性——entry 有效性不随 P1 是否冻结变化。Schema：`calibration: {basis, contract, contract_sha256, record:{path,sha256}}`。
2. **Guard Profile：外部 {path, sha256} 引用，不内联** ✅。硬规则：entry 不得复制 authoritative floor 数据（floors 只在 Guard YAML）；registry verify 必须做语义交叉验证（entry source SHA == guard source SHA == manifest source_bf16_gguf_sha256；tokenizer、evaluator digest 三方一致；guard 绑 manifest 时也钉）。
3. **`fit registry add --official`：否决 ❌**——public trust-elevation flag 制造"本地加 flag 即官方"的错觉。官方 trust root = FIT release 本身，不是一个 CLI flag。v0.3 CLI 只保留 list/show/verify/validate；官方 registry 修改 = maintainer workflow（scripts/add_registry_entry.py → tests → commit → release）；**review + merged release content 才形成信任**。用户自有本地 registry 的 add 可以后做。
4. **.kld 不进 release** ✅——TRUST ≠ TRANSPORT。Registry 只内置小型 manifest；大文件走独立外部 reference-bundle channel（倾向单独 HF Dataset/LFS 仓，evaluator asset 不进模型仓）；下载后一律 SHA 验证。**Locator 不进 trusted entry**（不写 download_url，否则换镜像 URL 就得改 entry SHA）——放非信任层 `registry/locations/reference-locations-v1.json`；`fit refs fetch` 留 A1/后续。

## Schema 六项修订

1. **Bootstrap = immutable grandfather basis**（legacy_bootstrap_v0）；P1 冻结只影响新 entry admission policy，不影响旧 entry runtime validity。
2. **Reference manifest 必须是 release runtime asset**：`registry/manifests/<source_sha>.json`（发布时与实验目录 manifest 逐字节复制）；不引用 experiments/ 路径（PyPI wheel 未必有）；guard 仍在 profiles/guard/，package data 明确包含。
3. **Calibration evidence 绑 canonical record SHA**：entry 只绑一个 `registry/calibration/<source_sha>.json` audit root，record 内部列证据 hashes。
4. **Cross-object semantic validation**：即使所有单文件 SHA 正确，entry/guard/manifest 的 source SHA、tokenizer、evaluator digest 交叉不一致即 FAIL（防"model A 的 entry + model B 的 guard + model C 的 manifest 且哈希都对"的现实维护事故）。这才是 bundle closure。
5. **Canonical JSON 完整定义**：`sort_keys=True, ensure_ascii=False, separators=(",",":"), allow_nan=False` + 无 BOM + 尾随换行不参与 + 移除 entry_sha 后 canonicalize + schema 尽量避免浮点。暂不上 RFC 8785。
6. **官方性与完整性区分**：`registry verify` = structural/content integrity verification，非 official authenticity attestation——自洽仿冒 registry 纯内容无法证伪，官方性来自 PyPI hash/signature、GitHub release/tag、git commit。README 不得声称 verify 可认证官方身份。

## 测试计划增补（v1.0 六组之外）

- **Cross-object mismatch**（最重要：比改一个字节更现实的维护事故）
- Duplicate/identity invariants：索引同 SHA 两次 FAIL、entry 文件名≠SHA FAIL、索引 SHA≠entry SHA FAIL、deterministic sort
- Path safety：拒绝绝对路径/`..`/root 逃逸（防第三方 bundle 任意文件读取）
- Bootstrap grandfather 双向：现有官方 legacy entry 永久 PASS；新 source + legacy basis → admission FAIL
- **Package-content smoke**：clean wheel/sdist 安装后 `fit registry verify`（v0.2.0 被 guard gitignore 咬过的教训）
- model_id ambiguity：同名多 SHA 返回全部候选或要求 SHA，绝不取第一个
- Candidate semantics：resolver 拒绝 candidate 但 list/show 仍显示（审计意义）

## A3 后续

**直接进 P1（fidelity-calibration-v1 prereg），不插别的功能。** 顺序：schema amendments → implementation/tests → orcarouter+Spark migration → clean package registry verify → A3 FROZEN → P1 prereg。A3 已定义 fit calibrate 必须产出的四类 canonical objects：reference manifest / guard profile / calibration record / candidate registry entry。

## 结语（GPT 原文要点）

"这三层一拆清楚，A1 后面就不会再发生'为了支持第三个模型，又得去改 eval-v1 FREEZE'的事情了——这才是 Registry v1 最核心的价值。"
