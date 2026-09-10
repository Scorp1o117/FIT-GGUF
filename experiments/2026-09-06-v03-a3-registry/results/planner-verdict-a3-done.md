# Planner verdict — A3 完成确认（2026-09-06）

来源：ChatGPT 策划师会话，对 A3 实现完成汇报的答复（raw=planner-verdict-a3-done-raw.md）。

## 签字

- **A3 Fidelity Registry v1：FROZEN ✅**（Design/Schema、Canonical JSON v1、Trust-root semantics、Lookup/fail-closed、Legacy migration、Package distribution、Cross-object closure 全部 PASS）
- **A3 implementation & migration：PASS ✅**，无剩余 blocker
- **批准立即进入 P1 fidelity-calibration-v1 预注册起草**（不插 A2/Refine/Registry 功能）

## 复核通过的要点

1. legacy bootstrap 治理正确（有限 grandfather，非时间性失效）。
2. canonical runtime assets 移进 package（src/fit_gguf/registry + profiles/guard）+ clean wheel venv verify 实测——v0.2.0"repo 里有、包里没"的坑被结构性堵死。
3. cross-object semantic closure 达标：entry source SHA ↕ guard ↕ manifest ↕ live eval-v1 digest + corpus drift check——"这些正确的文件确实属于同一个受信任模型 bundle"。
4. evaluator triple（entry==live==manifest；guard 只存契约名）接受，非 blocker——未来 P1 Guard schema 可自描述 digest，但"不为了这个去改 Registry v1"。

## 归档动作（已落实）

experiments/2026-09-06-v03-a3-registry/FREEZE.json（invariants 全表 + change_policy：改任何 invariant = fit.fidelity_registry.v2，不是顺手改）。

## P1 起草钉死清单（GPT 列出）

preset ladder/healthy-frontier/poison 规则；四档 local-window 生成与补点；n≥3 最低 validation evidence；P5 与向下截断精确定义；n<3 → candidate fail-closed；floor 数值精度与 canonical serialization；reference manifest 生成与 runtime provenance；calibration-record 完整 evidence roots；candidate→validated promotion 条件；fit calibrate 允许的失败状态（INSUFFICIENT_WINDOW / NOT_REACHABLE 等，绝不为"一键完成"放松门槛）。

## 队列

A3 ✅ FROZEN → P1 fidelity-calibration-v1 🚧 NEXT → P2 fit calibrate → A2 execution/on-disk → Qwen3-4B fresh DEV → G11 sealed
