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
| `--workdir` | calibrate（scratch） | `FIT_CALIBRATE_TMP` → tmpfs `/dev/shm/cal-<model>` → 平台临时目录 |
| `--on-disk` | calibrate（禁 tmpfs，scratch 落 out_dir/work） | off |

`fit calibrate` 默认一次 session 内参照与候选用同一 execution profile（保守默认，GPT 裁定）；official 参照 bundle 跨机器可用（provenance recorded + semantics verified）。

## 3. Out-of-core 原则

llama.cpp 全链 mmap 流式：BF16 超过 RAM/VRAM 时用 `--on-disk` + 调低 `--n-gpu-layers`（甚至 0）即可运行，代价是时间。**不做**未实测的性能声明；宣称"X B supported"前必须以真实 >RAM/>VRAM 模型 smoke 一次并记录 wall-time。

## 4. 失败语义

执行层失败（OOM/设备缺失/scratch 不足）→ `INPUT_DRIFT` 之外的执行态错误，fail-closed，禁止降级语义参数重试。

## 5. 宿主文件系统约束（硬性红线）

`calibrate._run` 把子进程的 stdout/stderr **文件描述符直接交给 llama.cpp**。llama.cpp
的日志走**无缓冲 stderr**，fd 直写会在目标文件上产生大量**短、非对齐、跨页**的缓冲写。

目标落在 `ntfs3` 上时，这会踩中内核 BUG 并**整机 panic**：

```
kernel BUG at fs/iomap/buffered-io.c:1061!
RIP: iomap_write_end+0x1e0/0x1f0
  iomap_write_iter → iomap_file_buffered_write
  ntfs_file_write_iter [ntfs3] → vfs_write → ksys_write
```

实测三次（2026-09-06 ×2、2026-09-10 ×1），写入进程分别是 `llama-quantize` 与
`llama-perplexity`；kdump 转储在 `/var/crash/<YYYYMMDDHHMM>/dmesg.*`。
panic 还会冲掉该卷上的未落盘写入（曾丢失 git index 与分支引用）。

**规则：**

- 热循环写入目标（子进程日志、参照 logits、候选工件）**必须是 tmpfs**。
  `--workdir` 默认已满足；不要把 `--on-disk` 指向 `ntfs3`。
  默认 scratch 由 `calibrate.default_scratch_root()` 解析：`FIT_CALIBRATE_TMP`
  优先，其次 `/dev/shm`（存在时），最后落到平台临时目录 —— Windows 上没有
  `/dev/shm`，字面路径会静默变成当前盘符下的 `\dev\shm`。
- 从 bundle **读取**参照不受影响；从 tmpfs 向 bundle 的**批量拷贝**
  （`cp` / `shutil.copyfile`）也不受影响 —— 只有「子进程 fd 直写」会触发。
- `assert_hot_loop_fs_safe()` 会对此 fail-closed 拒绝；`FIT_ALLOW_UNSAFE_FS=1`
  可显式覆盖，仅在完全知情时使用。
- 大模型建议连**源权重**也放 tmpfs（`--source /dev/shm/...`）：否则每个档位的
  quantize 都要从慢卷重读一次完整源权重。bundle 只记录源权重 SHA-256，不记路径，
  因此这样做不影响记录的可信度。

**为什么 `pipeline.py` 不受影响**：`fit analyze/plan/quantize/fidelity-search` 用
`subprocess.run(capture_output=True)` 捕获后在**本进程内**一次性写出日志，子进程
从不持有文件描述符。两个模块若将来统一实现，应统一到 capture 形态，而不是反过来。

**平台范围**：本节约束针对 Linux 的 `ntfs3` 驱动；Windows 的 NTFS 是另一套内核
代码路径，没有该写入 bug。`mount_fs_type()` 在没有 `/proc/mounts` 的平台返回
`None`（未知），因此该 guard 在 Windows 上不会误报。
