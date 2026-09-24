# 消融方法论 playbook（v3 输入门控）—— 可复用版

**来源**：occamy-1.0（Qwen3.6-35B-A3B，256 路由专家 MoE）消融实践，2026-09-21 ~ 09-22
**结论**：`拒答 99/100 → 8/100`（成品实测 10/100），`3-token KL = 0.0844`（对比此前最好 0.2196，**降低 62%**）

---

## 0. 一句话回答「能不能复用」

**能，而且复用价值很大 —— 但复用的不是「一次运行」，而是「一条流水线 + 一套诊断方法」。**

新模型上，真正需要重跑的只有两步（约 30 分钟 + 一次 study）；其余全部是环境无关的代码和方法。

| 环节 | 新模型要做什么 | 成本 |
|---|---|---|
| 引擎补丁（门控 / rank-k / 导出重放） | **直接复用**，env 门控默认关闭，零回归 | 0 |
| 门控采集 + 拟合 | **必须重跑**（模型特异） | ~30 分钟 |
| 交叉验证方向搜索 | **必须重跑**（依赖上面的统计） | ~15 分钟 |
| KL 组件归因 | 可直接算（需要 study 的 trial 数据） | 0 |
| study（贝叶斯搜索） | **必须重跑** | ~3.5 小时（20 trial） |
| 导出 / 核验 / 成品实测 | **直接复用**（脚本已参数化） | ~1 小时 |

**效率提升体现在三处**：
1. **不再走已被证伪的轴**（rank-k、rdo 的收益上限已探明）；
2. **不再踩已知的坑**（末位 token 采集、导出静默丢失、journal 冻结设置…）；
3. **搜索空间已收窄**（知道 `mlp.down_proj` 这种纯成本组件要允许归零、注意力是收益不是成本）。

---

## 1. 核心认知：为什么旧方法 KL 这么贵

**两种编辑在数学上是同一个家族的。** 设 `v` 为拒答方向（残差流），`W` 为某个专家的 down_proj：

```
经典编辑 : ΔW = -w · v (vᵀW)        输入侧向量 q₀ = vᵀW
门控编辑 : ΔW = -w · v  gᵀ          输入侧向量 g
```

两者都是 **rank-1、输出方向同为 `v`**，唯一区别是**输入侧那个向量**。

**而所有 vector_method 最后都做 `F.normalize(..., p=2)`，所以 `‖v‖ = 1`**，于是：

```
vᵀ W' = (1 - w) · vᵀ W
```

- **`w = 1` 是精确归零**
- **`w > 1` 是把拒答分量反向灌注**

**occamy 的旧冠军用的是 `w = 5.13`** —— 也就是把拒答分量反推到 **−4.13 倍**，而且对**每一个输入**（无害输入也一样）一致生效。

> **这才是 KL 账单的大头，不是「移除拒答的必然代价」。**

想买更多压制，正确做法是**换更好的输入向量**（把编辑精准打到该打的地方），不是把一个方向反复过推。

---

## 2. 方法：输入门控消融

### 2.1 目标函数

编辑对模型的扰动是 `-w·(gᵀx)·v`，能量 ∝ `E[(gᵀx)²]`；而拒答压制 ∝ `gᵀμ_target`。所以：

```
maximize  (gᵀ μ_r)² / (gᵀ S_h g)      ⇒      g ∝ S_h⁻¹ μ_r
```

- `μ_r` = 该专家输入空间里**有害 prompt 的均值**
- `S_h` = **无害 prompt 的二阶矩**（白化矩阵）

**用无害协方差白化**是关键：它把编辑推离无害流量真正占据的子空间。用 pooled 协方差反而更差。

> 注意：理论最优是**白化目标均值** `S_h⁻¹μ_r`，**不是**白化差值 `S_h⁻¹(μ_r−μ_h)`。实测两者打平（0.362 vs 0.363），但前者有推导支撑。

### 2.2 实测效果（交叉验证，2425 对）

| 候选方向 | 中位 win | 优于现有 | 优于 2× |
|---|---|---|---|
| `mean_target` | 0.705 | 2071/2425 | 613/2425 |
| `lda`（未白化） | 0.611 | 1922/2425 | 843/2425 |
| `fisher_h = S_h⁻¹(μ_r−μ_h)` | 0.363 | 2377/2425 | 1926/2425 |
| **`whiten_mean = S_h⁻¹μ_r`（采用）** | **0.362** | **2377/2425** | **1928/2425** |

**在体对照**（同 study 家族、编辑激进度最接近的一组）：

| | n_suppress | w | 拒答 | KL3 |
|---|---|---|---|---|
| 无门控 | 64 | 7.98 | 8 | 0.4795 |
| **有门控** | 56 | 8.25 | **7** | **0.1143** |

**同拒答、同编辑量级，KL 降 4.2 倍。**

### 2.3 归一化（保持 `w` 语义可比）

```
g_e = u_e · E_r|q₀ᵀx| / E_r|u_eᵀx|        （q₀ = vᵀW，即旧编辑的输入向量）
```

这样 `w = 1` 时门控编辑在**拒答侧的扰动量**与旧编辑 `w = 1` 相同 —— **两次实验的 `expert_ablation_weight` 才可比**。按**绝对值**配平（不用符号均值），避免 `E_r[r] ≈ 0` 时退化。

---

## 3. 可复用流水线（7 步）

### Step 1 —— 便宜预检：先问「方向存在吗」

**不要直接开长跑。** 先做分离度预检：

```bash
python pretest_input_gating.py     # 末位 token 版，快速看 Cohen's d
```

- **Cohen's d 中位数 > 1** → 有害/无害在专家输入空间可分，门控有戏
- **d ≈ 0.2** → 分不开，直接换轴，别浪费时间

occamy 实测 **d = 5.1**（724/724 全过 1.0）。

### Step 2 —— 全位置采集（**最容易踩的坑**）

```bash
PRETEST_TOP_EXPERTS=64 PRETEST_N=800 PRETEST_BATCH=32 \
PRETEST_STATS=/run/media/s117/OS/tmp/input-gating-stats-v2.pt \
python pretest_input_gating_v2.py
```

**为什么必须全位置**：safex 排序用的是**全部 token 位置**的路由率（`safex.py` 注册 router hook 记录整条序列）。只抓末位 token 的话：

> occamy 实测：study 选的 top-10 safex 专家里 **53/60 在末位 token 上零激活** → 门控只覆盖 **17.8%** 的编辑，第一次跑 82% 走的还是旧路径，**12 小时白费**。

采集方式：monkeypatch `Qwen3_5MoeExperts.forward`（纯 Python 循环），精确拿到 `x = silu(gate) * up`。约 6 分钟。

### Step 3 —— 交叉验证方向搜索 + 建门控

```bash
OMP_NUM_THREADS=1 PRETEST_STATS=<stats.pt> python analyse_input_gating_v3.py
```

**必须交叉验证。** 512 维空间里几十个样本，**样本内拟合会给出 `win = 0.0000` 的"完美"方向**（Fisher 退化 / 数值优化都能找到与样本正交的方向）——那是背样本，不是信号。

**判定真信号的方法**：闭式 Fisher 白化**与**梯度优化**两条独立路线收敛到同一个数字**（occamy: 0.1515 vs 0.1785）。两条独立路径一致 = 真信号。

输出 `input_gate_u.pt`（occamy: 2490 个向量，40 层，覆盖 study 可选的 top-64 共 100%）。

### Step 4 —— 引擎补丁（**已实现，直接复用**）

`src/abliterix/core/steering.py`：

- `_load_input_gate()` / `_load_component_input_gate()` —— env 门控的加载器
- `_apply_moe_steering` 里：`vTW = gate_vec` 替代 `v_dev @ W`
- **打印门控命中率**（`input gate: N/M expert edits gated (X%)`）—— 让静默回退无处藏身

```bash
export ABLITERIX_EXPERT_INPUT_GATE=/path/to/input_gate_u.pt
```

**默认关闭，rank-1 路径逐位不变**（冒烟测试 `max_err = 0.00e+00`）→ 历史结果零影响。

### Step 5 —— 跑 study

**搜索空间设计的四条经验**：

1. **允许组件归零**。旧配置 `strength_range = [0.5, 4.0]` 的下限**强制每个组件常开** —— 而 `mlp.down_proj` 在 LoRA 模式下只解析到**共享专家**（融合专家的 down_proj 是 Parameter 不是 Module），受控 A/B 证明共享专家单独消融 → **98/100，对拒答零贡献**。开放到 0 后搜索立刻归零它：**KL 0.1035 → 0.0844**。
2. **`w` 范围要给足过推空间**（`[0, 12]`），因为 `w=1` 不足以消除行为。
3. **n_suppress 上限要与门控覆盖对齐**（否则超出部分静默回退）。
4. **trial 数按时间预算定**：一个 trial ≈ 13 分钟（生成式评测是瓶颈）。

### Step 6 —— KL 组件归因（诊断纯成本组件）

用已有 trial 数据做标准化多元回归，**固定在拒答带内**（否则 KL 只是在追踪拒答水平）：

```python
X = [attn.o_proj.max_weight, mlp.down_proj.max_weight, moe.n_suppress, moe.expert_ablation_weight]
y = kl_divergence        # 限制在 refusal ∈ [8, 25]
```

occamy 结果：

| 组件 | 系数 | 解读 |
|---|---|---|
| `moe.n_suppress` | **+0.546** | KL 主宰 —— 编辑多少专家最贵 |
| `mlp.down_proj` | **+0.181** | **纯成本**，对拒答零贡献 |
| `attn.o_proj` | **−0.169** | **收益** —— 它买到的压制否则得由更贵的专家消融来买 |
| `moe.expert_ablation_weight` | +0.039 | 次要 |

**负系数 ≠ 无用**：注意力和专家是**互相替代**的，回归在固定拒答数下比较，所以"注意力系数为负"意味着它是**效率杠杆**，应当加码而非砍掉。

### Step 7 —— 导出 / 核验 / 实测

```bash
python export_occamy_best.py          # 复现 Pareto 索引 + 喂 stdin + 自动核验
python verify_occamy_export.py <dir>  # 逐字节比对
```

**这一步有一个会毁掉交付的坑**（见 §5.2）。

---

## 4. 已验证的边界（不要再走的轴）

| 轴 | 结论 | 证据 |
|---|---|---|
| **rank-k 子空间消融** | ❌ **证伪** | rank-4 在每个同拒答档位差 **2.4–3.3 倍**，且够不到 ≤15 拒答。原因：`_extract_multi_directions` 的第 1..k−1 个方向是**中心化逐样本差异的 SVD 分量** —— 刻画的是「prompt 之间的变化」，不是拒答信号，削掉=纯损伤 |
| **`rdo` 方向提取** | ⚠️ 上限有限 | rank-1 线性编辑的**输出方向必须取 `v`**：任何别的方向 `u` 的有效分量只有 `vᵀu`，纯属浪费。且 `mean`/`sra` 两种独立提取器已收敛到 **1.4%** 以内 |
| **router 操作** | ❌ 有害 | 关掉 `gate.weight[eid] *= scale` 后，同参数下 KL 从 1.54 → 0.2228（**6.3× 改善**） |
| **纯注意力 LoRA（不动专家）** | ❌ 无效 | LoRA 模式下 `mlp.down_proj` 只打到共享专家；共享专家单独消融 → 98/100 |
| **多编辑轴 × MoE 专家** | ❌ 工具不支持 | abliterix 显式 raise（本次已通过补丁实现 rank-k，但方法本身被证伪） |

**编辑的「形式」已经没有剩余空间**：rank-1 线性编辑在输入侧取 `S_h⁻¹μ_r`、输出侧取 `v`，**是可证最优的**。剩下的杠杆只有「编辑哪些/多少组件」。

---

## 5. 工程坑清单（复用时可省下大量时间）

### 5.1 采集与拟合

| 坑 | 症状 | 解法 |
|---|---|---|
| **末位 token 采集** | 门控覆盖率仅 17.8%，效果看不出 | 全位置累加 moments |
| **样本内拟合** | 出现 `win = 0.0000` 的"完美"方向 | 交叉验证（按 prompt 奇偶分半） |
| **`float(负数) ** 0.5`** | Python 返回**复数**，污染所有后续比较 | `max(x, 0.0) ** 0.5`（二阶矩算出的 `qᵀSq` 在 float64 下可能微负） |
| **PyTorch 多线程拖垮小算子** | 512×512 算子烧 2973% CPU 却十分钟零进展 | `torch.set_num_threads(1)` |
| **设备不匹配** | 累加器在 CPU、`x` 在 GPU | 累加器建在 `x.device`，存盘前再 `.cpu()` |
| **4296 维的 gram 太贵** | o_proj 门控每批 130 GFLOP/层 | token 抽稀（stride 8）—— 4096 维里 2.4 万样本足够定协方差 |

### 5.2 导出（**会静默毁掉交付**）

`_apply_moe_steering` 是**原地改基座权重**（融合专家是 Parameter，LoRA adapter 完全代表不了它）。而 `SteeringEngine.export_merged()` 的非量化分支会 **丢掉设备模型 → 从磁盘重载干净基座**。

> **后果：导出的模型没有任何专家消融，而所有其它产物看起来都正常。** occamy 的拒答全在路由专家里 → 成品会退化成 99/100。

**`export_adapter()` 会大声拒绝**（`_expert_deltas` 非空就 raise），**只有 merged 路径静默出错**。

**已实现的修复**：
- `_snapshot_base_weight_edits()` —— 丢掉设备模型**之前**快照 `(w, v, vᵀW)` delta（几 MB，而非 20 GB 整张量）
- `_replay_base_weight_edits(base, snap)` —— 在 CPU 基座上重放，放在 `get_peft_model` **之前**
- CLI / 交互路径都写 `base_weight_edits.pt` sidecar
- **`verify_occamy_export.py`** 从**原始 checkpoint + delta** 重推期望值，与成品**逐字节比对**（键名按后缀匹配，兼容不同前缀）

### 5.3 运行时

| 坑 | 解法 |
|---|---|
| **`uma_guard` 默认值太激进** | 会在 `avail=38.6G` 的健康状态自杀。用 `ABLITERIX_UMA_MIN_FREE_GB=6 ABLITERIX_UMA_MAX_SWAP_GB=24 ABLITERIX_UMA_PSI_SOME_AVG10=50` |
| **裸 `from_pretrained(device_map="auto")` 双份拷贝** | 峰值 rss 46.7G + cuda 65.4G。用引擎的 fast load：`low_cpu_mem_usage=True, disable_mmap=True` + `move_parameters_skipping_ngram` |
| **`python -m abliterix.cli` 静默空转** | 该模块**没有 `__main__` 入口**，退出 0 什么都不做。必须用控制台脚本 `heretic-env/bin/abliterix` |
| **续跑会恢复 journal 里冻结的设置** | `_handle_existing_checkpoint` 从 `user_attr['settings']` 恢复配置，且只接受**更大**的 `num_trials` → **改配置文件无效**。想提前结束 study 出菜单，须改写 journal 里的 `settings` 记录（**先备份 `.jsonl`**） |
| **`pgrep -f` 自匹配** | 等待循环的 pattern 会匹配到自身 bash -c 文本 → 用「文件存在 + size 连续两次采样不变」 |
| **llama.cpp 架构注册已拆包** | 新版模型类在 `conversion/` 子包，`convert_hf_to_gguf.py` 只剩 307 行；直接 grep 主脚本找不到架构支持 |
| **GGUF 转换的 MTP 断言** | `_Qwen35MRopeMixin.__init__` 在 `no_mtp=False` 且 config 无 `mtp_num_hidden_layers` 时硬断言 → **必须加 `--no-mtp`** |
| **`llama-cli -no-cnv` 已废弃** | 改用 `-st` / `--single-turn` |

---

## 6. 复用到新模型的清单

```bash
# 0. 引擎补丁：直接复用（src/abliterix/core/{steering,engine}.py + cli.py + interactive.py）

# 1. 预检（~10 分钟）
python pretest_input_gating.py

# 2. 全位置采集（~30 分钟）
PRETEST_TOP_EXPERTS=64 PRETEST_N=800 PRETEST_BATCH=32 \
PRETEST_STATS=/tmp/<model>-stats.pt python pretest_input_gating_v2.py

# 3. 交叉验证 + 建门控（~15 分钟）
OMP_NUM_THREADS=1 PRETEST_STATS=/tmp/<model>-stats.pt python analyse_input_gating_v3.py
#    → 看 median win，< 0.6 才继续

# 4. 配 study：<model>_gate.toml
#    - max_suppress 与门控覆盖对齐
#    - strength_range / component ranges 全部允许 0
#    - ablation_weight_range = [0, 12]
#    - 紧 prune 门（不改评测参数）

# 5. 跑 study（~3.5 小时 / 20 trial）
ABLITERIX_EXPERT_INPUT_GATE=<gate.pt> ./run-<model>.sh

# 6. 组件归因 → 找出纯成本组件（免费，用 trial 数据）

# 7. 导出 + 核验 + 成品实测
python export_occamy_best.py --out <dir>
python verify_occamy_export.py <dir>
#    再用 configs/occamy-measure-delivered.toml 的零 steering 配置读成品拒答数
```

**判断「值不值得做」的前置条件**：
1. 拒答行为是否落在**容易编辑的组件**里（先做受控 A/B：共享专家 / 路由专家 / 注意力，各自单独消融看拒答变化）
2. 预检的 **Cohen's d** 是否够大
3. 交叉验证的 **median win** 是否 < 0.6

**任何一步不通过就立刻换轴** —— 这是本项目最省钱的一条纪律。

---

## 7. 已知未达成项

**目标 KL ≤ 0.05 未达到**（停在 0.0844，验收线 0.10 已过）。

**下一杠杆已备好**：给 `attn.o_proj` 做**组件输入门控**。
- 依据：归因显示注意力是**收益**（−0.169），让它变便宜就能把预算释放给专家侧
- 可行性已确认：LoRA 路径 `lora_A = V @ W` 与专家侧 `vᵀW` 是**同一个 rank-1 槽位**
- 代码已实现（`ABLITERIX_COMPONENT_INPUT_GATE`）+ 采集脚本已写好（`build_component_gate.py`，含 4096 维的 token 抽稀）
- **尚未运行**（需要 GPU 空闲）
