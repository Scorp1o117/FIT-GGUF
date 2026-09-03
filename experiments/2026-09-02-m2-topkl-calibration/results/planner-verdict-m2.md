# M2 Planner Verdict (GPT, 2026-09-02)

Source: ChatGPT 会话「FIT质量下降阈值」收尾回复。「这轮 M2 数据已经足够让我做出明确策划裁定」。

## 状态
- **M2 Candidate Calibration：PASS ✅**
- **G3 / M3 Fidelity Contract v1：仍不冻结**（等 ≥1 个第三 DEV family）。
- 阶段牌：M1 ✅ FROZEN / M2 ✅ PASS / M3 ⏳ NOT FROZEN (blocked by third DEV family) /
  M5 ✅ / M6 🚧 / M10 🔒 / M11 🔒。

## Candidate Contract v0.2（内部名 candidate-contract-m2 / fidelity-contract-candidate-r2，NOT v1）
| Tier | KL | Same-Top | M2 decision |
|---|---|---|---|
| Quality | ≤ 0.05 | ≥ 93 % | ✅ Supported |
| Balanced | ≤ 0.10 | ≥ 90 %（原 91） | ❌ Too strict → REVISED |
| Compact | ≤ 0.15 | ≥ 88 % | ⚠️ At-risk，暂不动 |
| Mini | ≤ 0.20 | ≥ 85 % | ✅ Supported |

## 各档依据
- **Balanced 91 → 90**：Granite Top@.10=90.37 / P5=90.31 / P10=90.32 三读法都稳定 ~90.3，
  91 对 Granite 已是正常 healthy curve 的主动约束而非 pathological guard。**拒绝 90.5**：
  两 family 下写 0.5pp 会制造虚假统计精度。90 给 Granite 留 +0.37pp crossing margin。
- **Compact 88 保留 AT-RISK**：Granite 0.15 窗 n=4 全 FIT，P5/P10 实为近-min 统计，
  不具真分位含义；直接证据仍是 Top@.15（Qwen 89.10 / Granite 88.39）双支持 88。
  第三 family 若 Top@.15≈87.x 再降 87。
- **Quality 93 / Mini 85**：均 SUPPORTED（M2A/M2B 把 0.05 crossing 精确夹住是关键加分）。

## 长期原则（GPT 拍板）
- **TopFloor ≈ P5(Top | KL≈target, healthy)**；P10 作敏感性分析 / conservative diagnostic。
- 当前小 n（窗口 2~5 点）下 P5/P10 仅 diagnostic，不作统计冻结依据——冻 Same-top floor 前必须补 ≥1 个独立 DEV family。
- **数据单元 = model × tier**（本轮 2×4 = 8 个 crossing observation），不是 61 个 point。
- **不要固定 tier→controller 映射**（KL 主控 / Top 主控是模型×allocation×size 的性质）；保持动态 Kl|Same-top|Joint。
- Granite Balanced 是"必须双约束（KL+Same-top）"的教科书反例 → 双约束设计被数据验证。

## 下一动作（GPT 排序）
1. 补 ≥1 个第三 DEV family：7B~12B、非 Qwen / 非 Granite、标准 dense decoder
   （Gemma / Llama / Mistral 系任选工具链最方便的）；选定即永久 DEV，绝不挪作 M11 sealed。
2. 在 4 model 上重跑 Top@KL crossing（model×tier 核心表：Qwen / Granite / C / D）。
3. 重新审视 Compact 88（第三 family 若 Top@.15≈87.x 则降 87）。
4. 然后才谈冻结 Same-top floor + Fidelity Contract v1。
---

# Amendment-2 / Ling Window Fill — GPT 签字（2026-09-02 晚）

**M2 Amendment-2 PASS ✅**（preset-cliff mitigation PASS / 预钉目标纪律 PASS / family-dependence 证据 PASS / Quality 窗 n=1 PARTIAL 接受 / Compact crossing 置信提升 / 追加微批 NOT REQUIRED）

- **0.05 窗封存**：Quality=91.52%（confidence=LOW，近锚 KL0.0434/Top92.09，local_window_n=1）；不再为 n=1→n≥2 烧点；未来升级为 validated profile 时才按 below/near/above 正式三近邻重做。
- **bailingmoe-candidate 研究记录（冻结为研究记录，不冻结合同）**：91.52 / 87.91 / 85.28 / 82.76 vs dense-candidate-r1 = −1.48 / −2.09 / −2.72 / −2.24pp。Global Same-top Floor Hypothesis = REJECTED；H-M2-01 = SUPPORTED（family/architecture dependent，不判"generic MoE floor"也不判"MoE 固定低 2.2pp"）。Compact 一档因 LB1 贴脸真实点（0.1537/85.15）成为最硬证据。
- **新 OPEN 问题（优先级高于补点）**：
  1. ⚠️ G2 exact-size bug：`bailingmoe-exact-size-offset-480` —— 6/6 plan actual = predicted−480（恒定）。定性：BailingMoE calibration data VALID for research；BailingMoE FIT exact-size support NOT VALIDATED。要求做差分表（tensor payload/alignment/KV/header 分列）定位是容器项还是 tensor 记账；修好后同批 6 plan 即成 G2 regression fixture。生产代码禁止写死 `if bailingmoe: size -= 480`。
  2. ⚠️ planner coverage gap：Q5_0/Q5_1 不在 v0.1 尺寸预测器目标宇宙 → 应在 analyze/profile 阶段提前报 unsupported transition，不能等 exact-size 求解中途炸。
- **优先级队列**：① 定位/修 −480B ② 固化 G2 regression ③ 补第三个 dense DEV family ④ 冻结 dense Same-top Guard v1 ⑤ 视情况开第二 MoE 家族校准线。
- **M10/M11 提醒**：sealed 跑时结果 JSON/原始 evaluator log/prereg/执行 manifest 必须持久化，大 GGUF 可评完即删。
