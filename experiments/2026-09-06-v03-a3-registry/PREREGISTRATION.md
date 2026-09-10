# A3 — Fidelity Registry v1：设计冻结提案 v1.1（修订版）

日期：2026-09-06 · 作者：ZCode（GLM-5.3-Flash）执笔 · 状态：**GPT DESIGN APPROVED / FREEZE PENDING implementation**
裁定：planner-verdict-a3.md（DESIGN APPROVED / FREEZE PENDING 6 项修订；本 v1.1 已全部并入——按 GPT 原话"修完下面这些，不需要再开新一轮架构讨论，可以直接宣告 fit.fidelity_registry.v1 FROZEN"）
v1.0→v1.1 变更：①bootstrap 改 immutable grandfather basis ②manifest 移入 registry runtime assets ③calibration evidence 绑 canonical record SHA ④cross-object semantic validation ⑤canonical JSON 完整定义 ⑥官方性与完整性区分。另：否决 `fit registry add --official`；新增 locations 非信任层；测试计划增补 7 组。

## 0. 本提案冻结什么 / 不冻结什么

**冻结**：Fidelity Registry v1 的文件布局、entry schema、canonical JSON 序列化、lookup 语义、trust root、fail-closed 行为、与 v0.2 遗留路径的兼容规则、迁移测试验收标准。
**不冻结**：Calibration Contract v1 内容（P1）、fit calibrate 实现（P2）、新模型校准数据。

## 1. 三层信任架构

```
eval-v1（experiments/2026-09-02-eval-v1/FREEZE.json, rc2 5ce78dee…）
  └─ HOW TO MEASURE —— 永久冻结；FREEZE.json 自本提案起不再修改，
     manifest_sha256_prefix=631fc09f…（orcarouter）保留为 v0.2 历史 bootstrap provenance。
Calibration Contract v1（fidelity-calibration-v1，P1）
  └─ HOW TO CREATE A MODEL GUARD
Fidelity Registry v1（本提案）
  └─ WHICH MODEL-SPECIFIC BUNDLES ARE TRUSTED
```

Trust root = **当前 FIT release 本身**（git commit / PyPI 包 / GitHub Release）。信任链：
`FIT release → 官方 registry 快照（随 release 分发）→ entry（全 SHA 引用）→ 均绑定 eval-v1 final digest`。
加模型 = registry 增加一个 entry，不触碰 evaluator contract。

## 2. 文件布局（随 release 分发，git 追踪，package data 必须包含）

```
registry/
├── fidelity-registry-v1.json            # 索引（entry 摘要 + digest）
├── entries/<source_weights_sha256>.json # 主键即文件名
├── manifests/<source_weights_sha256>.json  # reference manifest 的 runtime 副本（与实验目录逐字节相同）
├── calibration/<source_weights_sha256>.json # canonical calibration record（audit root）
└── locations/reference-locations-v1.json    # 非信任层 transport locator（可更新/加镜像，不影响 entry trust identity）
```

### 2.1 索引 `fidelity-registry-v1.json`

```json
{
  "registry_schema": "fit.fidelity_registry.v1",
  "registry_id": "fit-official-registry",
  "created": "2026-09-06",
  "evaluator_contract": "eval-v1",
  "evaluator_contract_sha256": "5ce78dee9d11e6dfe83416628d0459d462719c0ecddf194c53ea9629db243d7c",
  "entries": [
    { "source_weights_sha256": "<64hex>", "entry_sha256": "<64hex>",
      "model_id": "…", "status": "validated", "scope": "exact_model" }
  ]
}
```

### 2.2 Entry `entries/<source_weights_sha256>.json`

```json
{
  "registry_schema": "fit.fidelity_registry.v1",
  "model_id": "spark-x25-4b-abliterated",
  "source_weights_sha256": "<64hex>",
  "tokenizer_sha256": "<64hex>",
  "evaluator_contract": "eval-v1",
  "evaluator_contract_sha256": "5ce78dee…",
  "reference_manifest": { "path": "registry/manifests/<source_sha>.json", "sha256": "<64hex>" },
  "guard_profile": { "path": "profiles/guard/guard-….yaml", "sha256": "<64hex>" },
  "calibration": {
    "basis": "legacy_bootstrap_v0",
    "contract": null,
    "contract_sha256": null,
    "record": { "path": "registry/calibration/<source_sha>.json", "sha256": "<64hex>" }
  },
  "scope": "exact_model",
  "status": "validated",
  "added_in": "v0.2.1",
  "entry_sha256": "<64hex>"
}
```

**规则**：
- **Canonical JSON（修订 5，完整定义）**：`json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")`；无 BOM；尾随换行不参与 canonical bytes；`entry_sha256` 字段移除后再 canonicalize；schema 尽量避免浮点（floor 数值只存在于 Guard Profile，不进 entry）。
- **主键** = `source_weights_sha256`：一个权重 digest 至多一个 entry；同名不同权不继承。
- **Bootstrap grandfather（修订 1）**：`calibration.basis = "legacy_bootstrap_v0"` **仅限** orcarouter、Spark 两个已 grandfather 的 source SHA，**runtime resolver 永久接受**（registry 是快照确定性的——entry 有效性不随 P1 是否冻结而变化）；P1 冻结只影响 **admission policy**：此后 `fit registry validate/add` 对新模型拒绝 legacy basis。新体系 entry：`basis="fidelity-calibration-v1"` + contract_sha256（P1 冻结后）。
- **Calibration record（修订 3）**：entry 只绑一个 canonical provenance root（`registry/calibration/<sha>.json`），record 内部再列证据 hashes（阶梯 summary、manifests、审计文件）。v0.2/v0.3 的实验记录保持原位，record 引用其 SHA。
- **Guard Profile（裁定 2）**：外部 `{path, sha256}` 引用，不内联；`profiles/guard/` 保持唯一事实源；**entry 不得复制 authoritative floor 数据**（floors 只存在于 Guard YAML）。索引里的 model_id/status/scope 仅为 UX summary。
- **status ∈ {validated, candidate}**；`scope` v0.3 只用 `exact_model`（family/architecture 为保留字，resolver 硬拒绝）。

## 3. Lookup 语义与验证（fail-closed，任何失败抛 RegistryError/GuardProfileError）

1. 计算 `source_sha256` → 索引精确匹配（无 → NOT VALIDATED，沿用现有 REFUSE_MESSAGE 语义）。
2. 读 entry 验 `entry_sha256`；读 guard/manifest/record 文件并逐个验 SHA。
3. **Cross-object semantic validation（修订 4，bundle closure）**——除文件哈希外必须交叉一致：
   - `entry.source_weights_sha256 == guard.source_sha256 == manifest.source_bf16_gguf_sha256`
   - `entry.tokenizer_sha256 == manifest.tokenizer_sha256`
   - `entry.evaluator_contract_sha256 == guard.evaluator_contract_sha256 == manifest.evaluator_contract_sha256 == live eval-v1 digest`
   - guard 若绑定 reference manifest：`guard.reference_manifest_sha256 == entry.reference_manifest.sha256`
4. `status == "validated"`（candidate 不能出正式 tier；但 `list/show` 必须仍显示 candidate——审计意义）。
5. 取 `floors[tier]` 组装 TierContract（与现有 resolve_contract 同构）。模型名仅 UX 标识。

**`fit registry verify` 语义（修订 6）**：= structural/content integrity verification（结构+哈希+交叉语义），**不是 official authenticity attestation**——自洽的本地仿冒 registry 无法被纯内容证伪，官方性只来自 PyPI wheel hash/signature、GitHub release/tag、git commit 等分发信任。README 不得声称 verify 可认证官方身份。

## 4. Transport（TRUST ≠ TRANSPORT，裁定 4）

Registry 只内置小型 manifest；大型 `.kld` **不进 release/wheel/git**，走独立外部 reference-bundle channel（倾向单独 HF Dataset/LFS 仓——evaluator asset，不进模型仓）。下载后一律 SHA 验证（match → usable / mismatch → reject）。locator 只放 `registry/locations/reference-locations-v1.json`（非信任层，type/repo_id/revision/subdir），可更新加镜像而不改 entry trust identity；`fit refs fetch <model>`（locator→下载→验 manifest→验每个 .kld SHA→装缓存）留 A1 或后续。

## 5. CLI 面（v0.3 范围，裁定 3）

- 产品 CLI 仅：`fit registry list` / `show <source_sha|model_id>` / `verify`（全量结构+哈希+交叉语义）/ `validate <bundle>`（只读，校验 bundle 可注册性）。
- **无 `add --official`**（否决：public trust-elevation flag 制造"本地加 flag 即官方"的错觉）。官方 registry 修改 = maintainer workflow：`bundle → fit registry validate → maintainer review → repo mutation script（scripts/add_registry_entry.py，maintainer-only）→ tests → commit → release`。**review + merged release content 才形成信任**（PR review 本身不是 runtime trust root）。用户自有本地 registry 的 add 可以后做，不进 v0.3。
- `show` 按 model_id 查询遇同名多 SHA → 返回全部候选或要求 SHA，绝不取第一个；source SHA 永远是唯一解析键。

## 6. 测试计划（§6 v1.0 六组 + GPT 增补七组）

1. registry+entry 往返 / canonical hash（含：无 BOM、无尾随换行、pretty-print 不影响 digest）
2. 篡改检测（entry / index / guard / manifest 任一字节）
3. 兼容双通道：orcarouter+spark 新旧通道 TierContract 逐字段一致
4. fail-closed（缺 entry / 任一 SHA 不匹配 / evaluator 漂移 / scope 保留字）
5. bootstrap grandfather 双向：**现有官方 legacy entry 永久 PASS**；新 source 声明 legacy basis → admission FAIL（不是"P1 后全 FAIL"）
6. 现有 152 测试零回归
7. **Cross-object mismatch**（最重要）：所有单文件 SHA 正确但 entry/guard/manifest 的 source SHA、tokenizer、evaluator 不一致 → FAIL
8. Duplicate/identity invariants：索引同 SHA 两次 / 文件名≠SHA / 索引 SHA≠entry SHA → FAIL；索引 deterministic sort
9. Path safety：拒绝绝对路径、`..`、registry root 逃逸（防未来第三方 bundle 任意文件读取）
10. **Package-content smoke**：clean wheel/sdist 安装后 `fit registry verify` 通过——钉死 registry/ + profiles/guard/ + manifests/ 真的被打包（v0.2.0 guard 被 gitignore 咬过的教训）
11. model_id ambiguity：同名多 SHA 返回多候选
12. Candidate semantics：resolver 拒绝但 list/show 可见

## 7. 首批 entry（迁移）

- orcarouter-Qwen3.8-27B-Uncensored（f9545645… / guard-orcarouter-…-exact-v1 / manifest e8d10125…）
- spark-x25-4b-abliterated（aa73aeb4… / guard-spark-…-exact-v1 v3 / manifest 按实验记录 refs+语料 SHA 正式打包，纯复制工作）

## 8. 裁定记录（GPT，2026-09-06）

| 裁定项 | 结论 |
|---|---|
| Bootstrap calibration | ✅ 批准，改为**永久 grandfather** 的 legacy_bootstrap_v0；P1 后只禁止**新增** legacy entry |
| Guard Profile | ✅ 外部 {path, sha256}，不内联；entry 不得复制 floors |
| registry add --official | ❌ 取消 public trust-elevation flag；官方 registry 只经 maintainer/release workflow |
| Reference .kld | ✅ release 不携带；manifest 内置，大文件走独立外部 reference-bundle channel |
| Registry schema | 🟡 FREEZE PENDING 6 amendments（本 v1.1 已并入） |
| Tests | 🟡 补 cross-object、identity、path、grandfather、package smoke、model_id、candidate |
| A3 后续 | ✅ 完成即进 P1 Calibration Contract v1（不插别的功能） |

**A3 done 路径（GPT 钉死顺序）**：schema amendments（本 v1.1）→ implementation/tests → orcarouter+Spark migration → clean package registry verify → **A3 FROZEN** → P1 fidelity-calibration-v1 prereg。A3 已告知 P1：fit calibrate 必须产出 reference manifest / guard profile / calibration record / candidate registry entry 四类 canonical objects。
