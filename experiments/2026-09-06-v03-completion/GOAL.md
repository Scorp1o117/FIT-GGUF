# v0.3 完成目标（Definition of Done）

设立：2026-09-06 · 修订：2026-09-10（瘦身版）

> **2026-09-10 修订说明**：原计划要求每段 GPT 策划师签字、全密封一次性验证（P5 G11）与
> hypothesis 预注册。经项目负责人裁定：**这是程序，不是科研项目** —— 治理流程按工程口径
> 重写：砍掉「证明我们是对的」的外部仪式，保留「保证程序不炸」的工程卫生（测试、
> fail-closed 语义、数据完整性）。原有的 PREREGISTRATION.md / FREEZE.json /
> planner-verdict-*.md 作为历史记录保留在 experiments/ 下，但**不再是流程要求**。

## 目标声明

**v0.3.0 发布**：在已完成的 A3 / P1 / P2 / P3 之上，跑通一次新模型验收（P4 冒烟测试），
完成命名与文档定稿（C3），以 v0.3.0 发布收口。

## 判定口径（工程）

| 手段 | 定位 |
|---|---|
| 单元测试 `pytest tests/` | ✅ 质量门，必须全绿 |
| fail-closed 拒绝语义（unvalidated 模型拒绝、candidate 不冒充 validated） | ✅ 程序行为，必须有 |
| registry SHA / manifest 交叉校验 | ✅ 数据完整性，必须有 |
| 大文件 & 敏感文件体检（`.gitignore` / `git check-ignore`） | ✅ 发布前必查 |
| GPT 策划师签字 / 预注册 / FREEZE / sealed one-shot | ❌ 不做 |

## 各段 DoD 与状态

### A3 — Fidelity Registry v1 ✅ 完成
- [x] `registry.py` + 2 个 exact-model entry（orcarouter-27B / spark-x25-4b，均 validated）
- [x] `fit registry list/show/verify/validate`
- [x] guard profiles 进包（修掉 v0.2「repo 里有、wheel 里没有」的坑）

### P1 — fidelity-calibration-v1 合同 ✅ 完成
- [x] 机读合同 `src/fit_gguf/contracts/fidelity-calibration-v1.json`
- [x] 合同库 `calibration.py`（W(K) 窗口 / Decimal trunc4 / unique-observation dedup / witness / 失败状态枚举）

### P2 — fit calibrate 生产线 ✅ 完成
- [x] CLI：输入钉定 → imatrix → 参照生成 → 标准梯(+extras) → gap probes → 地板推导 → Guard YAML → Bundle 八件套
- [x] `--replay-existing` 零评测复现通道
- [x] Spark fresh onboarding 产出完整八件套 `experiments/2026-09-06-v03-completion/p2-spark-fresh/`
- [x] normative fixtures 进 tests/

### P3 — A2 Execution Profile ✅ 完成
- [x] `--n-gpu-layers / --threads / --workdir / --on-disk` 贯穿 calibrate 与执行器
- [x] `docs/execution-profile.md`

### P4 — 新模型验收（冒烟测试）✅ 通过
- [x] 单条 `fit calibrate` 命令跑通全流程（无模型专用脚本、无手工干预，耗时 ~38 分钟）
- [x] 产出合法 bundle：`overall=validated`、open failures none、`fit registry validate` admissible
- [x] 已入册 registry（`f3e9d463…`，v0.3.0，status=validated）
- 结果与结论见 `p4-qwen3-4b/PREREGISTRATION.md` §5：生产线价值证实（A1 acceptance）；
  H-M3-02 容量假设方向性 4/4 命中（同 KL 下 top-1 一致率随容量下移）
- 说明：hypothesis 预注册 / DEV-sealed 标记等研究属性取消；这里只验「新模型开箱能不能用」。

### P5 — G11 unseen-family sealed ❌ 取消
- 理由：产品泛化性由 P4 回答；一次性密封协议属科研仪式。
- 后续若想看陌生模型，随手跑一遍即可，不需要 sealed 框架。

### C3 + 发布
- [x] 命名规范定稿（已随 `fit plan` 的 `dominant_qtype` / `suggested_filename` 字段落地）
- [x] README 补 `fit calibrate` / `fit registry` 用法（EN/zh-CN）
- [x] `pyproject.toml` → 0.3.0 + CHANGELOG
- [ ] commit + tag `v0.3.0` + GitHub Release

## 非目标

- B1/B2 Structural Refine（并行研究线，不阻塞发布）
- C2 family/arch guard promotion
- 任何未实测的性能声明（如「70B supported」）

## 状态日志

- 2026-09-06：目标设立。A3 ✅、P1 ✅ 达成。
- 2026-09-06 22:46：P2 Gate B ✅（Spark fresh 八件套产出）；P4 首次尝试中断。
- 2026-09-10：v0.3 工作从「无 commit 工作区」落为本地 4 个 commit（保命）；治理流程瘦身（本文件修订）；P4 重启。
- 2026-09-10：查明 P4 首次失败实为 **ntfs3 内核 panic**（非进程被杀），修复热循环写入路径后重跑。
- 2026-09-10：**Fidelity Contract v2** —— 档位硬门槛改为只看 KL，Same-top 降为参考值。
- 2026-09-10：**P4 ✅ 通过**（validated + 入册）；C3 完成，待发布收口。
