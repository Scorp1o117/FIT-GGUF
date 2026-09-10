# v0.3 完成目标（Definition of Done）

设立：2026-09-06 · 执行：ZCode（GLM-5.3-Flash）· 治理：GPT 策划师（每段 verdict 归档）
队列依据：planner-verdict-v03-preview.md 优先级表 + planner-verdict-a3-done.md / planner-verdict-p1-frozen.md 的顺序钉死。

## 目标声明

**v0.3.0 GA**：A3 ✅ 与 P1 ✅ 两个地基之上，交付 `fit calibrate` 生产线（P2）、Execution Profile（P3）、Qwen3-4B fresh DEV onboarding（P4）、G11 unseen-family sealed 验证（P5）、命名与文档定稿（C3），全部按已冻结合同执行、每段 GPT 签字，最后以 v0.3.0 发布收口。

## 各段 DoD

### P2 — fit calibrate（🚧 当前）
- [ ] 合同库 `src/fit_gguf/calibration.py`：机读合同加载+digest 校验、W(K) 窗口、Decimal trunc4、P5、unique-observation dedup、witness、逐档/整体状态、失败状态枚举（§10 全表）
- [ ] `fit calibrate` CLI：输入钉定→imatrix 覆盖检查→参照生成→标准梯(+extras)→窗口分析→gap probes→地板推导→Guard YAML→Bundle 八件套（calibration-record / reference-manifest / curve-points.jsonl / guard-profile.yaml / profile-report.md / registry-entry(candidate) / SHA256SUMS / references 清单）
- [ ] normative fixtures（合同 5 + Spark 2）进 tests/
- [ ] **Gate P2-A**：`--replay-existing` 逐字段复现 P1 replay（quality n=0 / balanced n=0 / compact 0.8359 validated / mini 0.8145 candidate / overall candidate / INSUFFICIENT_WINDOW）
- [ ] **Gate P2-B**：Spark fresh onboarding ≤4 probe/档补窗，产出**新** fidelity-calibration-v1 guard（validated 或诚实 candidate），绝不复用 grandfather floor
- [ ] GPT 签字

### P3 — A2 Execution Profile（lite scope）
- [ ] --n-gpu-layers / --threads / --workdir / --on-disk 贯穿 calibrate 与执行器
- [ ] 工程 spec 文档（无需 full G0）；70B 声明不做（无实测）

### P4 — Qwen3-4B fresh DEV onboarding
- [ ] hypothesis prereg（H-M3-02 容量假设）hash 后落盘
- [ ] `fit calibrate` 无模型专用脚本跑通全流程（下载→bundle→registry entry candidate/validated）
- [ ] 从入册起标记 DEV（sealed_eligible=false）

### P5 — C1 G11 unseen-family sealed
- [ ] 选定从未进 dev loop 的 family（排除 Qwen/Granite/Gemma/Ling/Spark/Ornith/PRISM；候选：Falcon3-3B / SmolLM2 / OLMo-2，Apache-2.0 可获取优先）
- [ ] 全密封预注册（family、模型、协议、预算、成功标准，全 hash）先于任何执行
- [ ] one-shot：fit calibrate + 四档 search → sealed 验证报告
- [ ] GPT 签字

### C3 + 发布
- [ ] 命名规范定稿（tier+size+dominant；v1 冻结）
- [ ] README/docs 更新（registry、calibrate、contract 语义）
- [ ] v0.3.0：版本号、CHANGELOG、commit+tag+GitHub Release（GPT 终审后）

## 非目标（明确不做）

- B1/B2 Structural Refine（并行研究线，不阻塞 GA）
- C2 family/arch guard promotion（research-only）
- 任何未实测的性能声明（如"70B supported"）

## 状态日志

- 2026-09-06：目标设立。A3 ✅ FROZEN、P1 ✅ FROZEN（455338c5…）已达成。
