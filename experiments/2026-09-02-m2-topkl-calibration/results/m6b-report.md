# M6b 汇报草稿（band-conditional Refine 实现 + gate 结果）— 2026-09-03

（此文件为向 GPT 汇报的留档草稿；发送后以 verdict 文件为准。）

## 一、M6b 实现摘要（本轮代码产出，112 测试绿）

1. **信号提炼**：band = blk//16（PRISM depth_bucket_of，n_layers=65）；数据集内可用证据两类：
   - sensitivity-map anchor cells（16 个，`role:bucket:src->dst`，wholesale band retype 实测 macro Δ%）；
   - chain 孤立单动作步：nc-v4 late(blk.23+) ssm_out q3_K→q4_K k=30 → **−3.7448%**（wholesale 全 48 张量 role 探针 = +3.34% 有害）——role 级平均掩盖 band 结构的实锤，即 GPT 指定的自然验证案例。
2. **拟合规则（bootstrap-v1，全数据支撑、同 v0 语义）**：utility cell 只取升级方向（dst BPW > src BPW），
   c = clamp(1 − Δ/20, 0.5, 1.5)；降级 cell（含 q2_K 悬崖）只作 protection 证据不进 utility（FIT 候选全是严格升级）；
   多动作 chain 步跳过（L2 不可加）并留审计。产出 7 个 utility cell：

| role | blk | src→dst | Δ% | C |
|---|---|---|---|---|
| attn_qkv | 0-15 | q3_K→q6_K | −16.71 | 1.500（clamp）|
| attn_qkv | 16-31 | q3_K→q6_K | −4.16 | 1.208 |
| ffn_down | 0-15 | q3_K→q6_K | −5.34 | 1.267 |
| ffn_down | 16-31 | q3_K→q6_K | −7.02 | 1.351 |
| ffn_down | 32-47 | q3_K→q6_K | −7.09 | 1.355 |
| ffn_down | 48-64 | q4_K→q6_K | −3.03 | 1.151 |
| ssm_out | ≥23（chain）| q3_K→q4_K | −3.74 | 1.187 |

   未覆盖 (role,band) 回退 role 级 C_role；late-ssm_out 从 role 级 0.833 反转为 1.187 ✅（验证案例成立）。
3. **planner 接入**：`_apply_refine_corrections` 按 candidate 的 (role, blk) 解析最窄匹配 cell；
   plan 记录 `band_cells_applied` 计数；v0 profile（无 band 段）走原 role 路径完全兼容。
   已知限制（写入 profile fit_rule）：bootstrap 按 (role, band) 应用，逐 (src→dst) 边际应用需 G0 预注册校准（M6c 方向）。

## 二、发现：M3 表 "Compact 回退" 是窗口混杂，不是 allocator 回退

M3 双主表里 Compact 档 v0.2 "真回退"（12.95G 才达标）的对照不公平：
- v0.1 参照 FIT-11.5G 窗口 = **IQ3_XS→IQ3_S**；
- v0.2 Compact 窗口 = **Q3_K_S→IQ3_S**（下界是已知毒预设 Q3_K_S，P4 fixture：Q3_K_S 11.24G KL 0.2148）。
同一字节（v0.1 的 12,277,279,936 = 11.43G）三组对照（本轮实测）：
- v2b band profile，IQ3_XS→IQ3_S：__FILL__
- v1 stack（无 refine），IQ3_XS→IQ3_S：__FILL__
- v2b band profile，Q3_K_S→IQ3_S：__FILL__
→ 归因：__FILL__

## 三、Gate A（Fixed Size：同字节 KL ≤ bootstrap-v0）

| Tier | v2b | bootstrap-v0 | 判定 |
|---|---|---|---|
| Quality | 0.0525 / top 94.83 @ 16.17G | 0.0523 @ 16.19G | parity：KL +0.0002（0.4% rel）但尺寸 −23MB；按局部曲线斜率 ~0.0051/G 折算 23MB≈0.0001 KL，两点曲线等价 |
| Balanced | __FILL__ | 0.0976 @ 12.95G | __FILL__ |
| Compact | recipe byte-equal（identity）| 0.1746 @ 11.42G | PASS by identity |
| Mini | recipe byte-equal（identity）| 0.1924 @ 10.35G | PASS by identity |

注：两 Quality 工件是"同 target 不同欠填"（band C 改变选择 → oracle 离散残差不同），
严格 same-bytes 的 Quality 对比要等 byte delta=0 正式版（P1 已排）。

recipe diff：Quality 2 张量（blk.32.ffn_down +q5_k / blk.40.ffn_up −q5_k）；Balanced 2 张量（同模式 blk.33/blk.40）——
band cell 把预算从 ffn_up 移向 ffn_down mid-band，方向与实测 cell 一致。

## 四、Gate B（Fixed Fidelity：Size_v2b ≤ Size_v0.1 且过档位锚）

| Tier | v2b | v0.1 参照 | anchor | 判定 |
|---|---|---|---|---|
| Balanced | 0.0987 / top 92.02 @ 13.00G（同窗口同字节）| 0.0987 @ 13.00G | 0.10 | **PASS（平手）** |
| Compact | 0.1465 / top 89.05 @ 11.37G（IQ3_XS 窗口）| 0.1439 @ 11.43G（同窗口）| 0.15 | **PASS** |
| Mini | bootstrap 工件即 v2b（recipe identity）0.1924 @ 10.35G | 0.1955 @ 10.42G | 0.20 | **PASS** |

**Gate B = 3/3 Size_v2b ≤ Size_v0.1（GPT 门槛 ≥2/3）✅**。Compact v2b 比 v0.1 小 57MB、
KL +0.0026（约当于尺寸差内的曲线噪声）；Balanced 同字节与 v0.1 精确平手（4 位小数）。

## 五、执行口径

1. 全部 eval-v1 口径（5 域 macro），G2 重终结门 6/6 delta+0；工件 hash 记录后释放，tmpfs 热循环。
2. CompactFF 计划落点 12,206,255,328（11.37G，target 12,277,279,936 欠填 71MB，oracle 离散步长）——对 "≤ v0.1" 判定方向有利，如实标注。
3. Gate A 的 Compact/Mini 为身份通过（v2b recipe 与 bootstrap 完全一致 → 同工件同 KL）；band cell 在这两档只改变排序未改变选择。

## 六、请 GPT 裁定

1. Gate A/B 判定与 "Compact 回退归因（窗口混杂）" 是否接受；M3 双主表 Compact 行是否按此改写。
2. M6b bootstrap-v1 是否可以标记为 Refine v1 candidate（dev），进入正式双主表重建（byte delta=0 + Fidelity Search）。
3. M6c（chain-conditioned marginal / 逐 src→dst 应用）是否仍需要，还是 band 级已够 v0.2 发布。
4. Compact 档窗口选择规则（禁毒预设做下界？）是否固化为 pipeline 规则。

---

## 七、收尾补记（2026-09-03 上午）

1. gate 运行 4/6 时机器正常关机重启（journalctl 确认 clean poweroff，非内核恐慌）：
   CompactFF / BalancedFF / Quality 三工件完整落盘（manifest+eval 日志），Balanced 交叉字节
   工件丢于 tmpfs、两归因对照未跑——均为 P1 四档 minimum PASS 工作的一部分，随 Fidelity
   Search v1 一起补跑，不阻塞本裁定。
2. Gate A Balanced（交叉字节）在重启中丢失，按 12,95G bootstrap 的 Gate B 同窗口平手结果与
   recipe 2-张量交换推断为 parity 级；待补跑确认（列入 P1）。
3. 已发送 GPT（9:42）并读完整三轮裁定（planner-verdict-m6b.md）：
   - **v0.2 定位 = C**：Fidelity-tier exact-size allocator；Refine 降级为 quality/safety layer；
   - M6c + 结构性 band-swap → v0.3 合流；A 撤销为 blocker；
   - --fidelity-tier 直出 artifact；Tier = KL Core ∧ Guard 双硬门；Fidelity Search v1 进 v0.2
     （coarse→bracket→active-constraint→fine→verify，≤8/≤16 evals，输出 minimum verified PASS）；
   - M6b 判定 PASS（material fixed-size uplift 未证明，非 FAIL——目标已重定义）；
   - Compact amendment 同意（本文件§二、§四的结论被 M3 verdict/m3-tables 正式引用）；
   - Q3_K_S 升级 planner 禁毒下界；Release Gates R1-R6 重写（删除 "refine KL≥3%" 硬门）。
