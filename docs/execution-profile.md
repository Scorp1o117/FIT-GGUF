# A2 — Execution Profile / Out-of-core Calibration（工程规范）

状态：engineering spec（planner 裁定：lite scope，不需 full G0）· 2026-09-06 · v0.3

## 1. 定位

Execution Profile 回答"**在哪跑、怎么跑**"；eval-v1 与 fidelity-calibration-v1 回答"**测什么、怎么测**"。三层纪律：

- **semantics REQUIRED**：`-c 512 -b 512`、KL 算法、五域语料、指标定义——不可配置，动即新合同/新 eval 版本。
- **provenance RECORDED**：`-ngl`、threads、backend、binary SHA、GPU——全部进 manifests/calibration record。
- **equivalence VERIFIED**：非钉定运行时必须附 perplexity 源码与钉定构建逐字节一致的 diff 证据。

## 2. 参数面（已实现）

| 参数 | 去处 | 默认 |
|---|---|---|
| `--n-gpu-layers` | calibrate / fidelity-search / RunnerConfig | 99 |
| `--threads` | 同上 | 16 |
| `--workdir` | calibrate（scratch） | tmpfs `/dev/shm/cal-<model>` |
| `--on-disk` | calibrate（禁 tmpfs，scratch 落 out_dir/work） | off |

`fit calibrate` 默认一次 session 内参照与候选用同一 execution profile（保守默认，GPT 裁定）；official 参照 bundle 跨机器可用（provenance recorded + semantics verified）。

## 3. Out-of-core 原则

llama.cpp 全链 mmap 流式：BF16 超过 RAM/VRAM 时用 `--on-disk` + 调低 `--n-gpu-layers`（甚至 0）即可运行，代价是时间。**不做**未实测的性能声明；宣称"X B supported"前必须以真实 >RAM/>VRAM 模型 smoke 一次并记录 wall-time。

## 4. 失败语义

执行层失败（OOM/设备缺失/scratch 不足）→ `INPUT_DRIFT` 之外的执行态错误，fail-closed，禁止降级语义参数重试。
