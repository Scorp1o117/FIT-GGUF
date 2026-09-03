# M2 Top|KL Candidate Calibration — Pre-registration

Date: 2026-09-02
Status: PREREGISTERED — written before any new GPU point was quantized or evaluated.
Deliverable: `candidate-calibration-v1` (Candidate Contract evidence, **NOT** a statistical
freeze; final threshold judgment belongs to the planner).

---

## 1. Question

Are the Fidelity Contract candidate same-top values **93 / 91 / 88 / 85** at KL
**0.05 / 0.10 / 0.15 / 0.20** too strict or too loose, measured per model on the
eval-v1 protocol?

## 2. Health windows (pinned by fixed rule, data-independent)

±15 % around each KL target:

| Tier | Target | Window |
|---|---:|---|
| Quality | 0.05 | [0.0425, 0.0575] |
| Balanced | 0.10 | [0.0850, 0.1150] |
| Compact | 0.15 | [0.1275, 0.1725] |
| Mini | 0.20 | [0.1700, 0.2300] |

## 3. Models

- `orcarouter-Qwen3.8-27B-Uncensored` (DEV; BF16 sha `f9545645…`, refs
  `experiments/2026-09-02-eval-v1/refs-orcarouter/` byte-identical to P4 legacy).
- `granite-4.2-8b` (DEV; BF16 source was deleted, re-converted this round; refs
  generated fresh under eval-v1 slices — **no** P4/m16 reuse possible).

## 4. Reuse list (eval-v1-compatible historical points)

**orcarouter — `experiments/2026-08-29-p4-release-batch/results/p4-results.json`**:
14 FIT tiers (FIT-7G…FIT-13.5G) + 14 preset anchors (IQ1_S…IQ4_XS) = 28 points,
per-domain mean_kld + same_top reused as-is. Compatibility evidence, verified today:
1. `slices-sha256.txt` == eval-v1 contract corpus SHA256s on all five domains
   (d400318e… / 9b455800… / a7584dc6… / da9cae00… / 01c79b52…).
2. Same pinned runtime b10666 (commit 4e97ac86e), same eval flags
   (`-ngl 99 -t 16 -c 512 -b 512 --kl-divergence --kl-divergence-base`).
3. FIT-12G requantized byte-identical (sha `e0a406ef…` == published) and its
   eval-v1 result reproduced the published macro (0.1227394 / 90.3242 %).

**granite — zero reuse.** m16 used holdout6 slices (disjoint corpora; chinese =
low-CJK variant). m16 numbers are excluded from calibration entirely.

## 5. New points (pinned before execution)

### 5.1 orcarouter (quantized from BF16 `f9545645…` with `imatrix_unsloth.gguf`)

| Point | Kind | Purpose |
|---|---|---|
| Q4_K_S | preset | bracket 0.05 crossing (currently uncovered; lowest point IQ4_XS 0.062425) |
| Q4_K_M | preset | bracket 0.05 |
| Q5_K_S | preset | below-window anchor for 0.05 |
| Q5_K_M | preset | below-window anchor for 0.05 |
| FIT-14G | fit (target 15,032,385,536 B, pair Q3_K_L→IQ4_XS, policy balanced) | FIT frontier evidence near 0.06 |

Notes: FIT-15G is infeasible with cached pair analyses (max-fill = IQ4_XS
15,082,507,456 B < 16,106,127,360 B); extending the candidate ladder is deferred
to M3 if the planner wants a Quality-tier FIT frontier below 0.06. FIT-14G is a
calibration-only artifact, not a release tier.

### 5.2 granite (fresh pipeline; BF16 re-converted, imatrix `imatrix-granite-apex-c512.gguf` sha `5488dbe0…`)

Presets (20): IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ2_M, Q2_K_S, Q2_K,
IQ3_XXS, IQ3_XS, Q3_K_S, IQ3_S, IQ3_M, IQ4_XS, Q4_K_S, Q4_K_M, Q5_K_S, Q5_K_M,
Q6_K, Q8_0.
FIT (5, recipes + tensor-type files reused verbatim from m16, expected sizes pinned
in `run_m16_stage_ab.sh`): O-FIT25, O-FIT50, O-FIT75, B-FIT25, B-FIT75.

BF16 provenance: b10666 converter lacks granite support (verified today); conversion
uses upstream llama.cpp master `convert_hf_to_gguf.py`, output sha compared against
the recorded m16 BF16 `d82690e0…`. Match → bit-reproduction; mismatch → recorded as
honest deviation (new BF16 fact), pipeline proceeds.

## 6. Pinned protocol (identical for every new point)

- Quantize: `tools/llama-b10666-rocm/llama-quantize --imatrix <imx> <src> <out> <qtype>`
  (FIT points add `--tensor-type-file <tf>`), cwd = repo root.
- References (granite, two-phase per rc2): generate WITHOUT `--kl-divergence`,
  eval WITH `--kl-divergence --kl-divergence-base`.
- Eval: `tools/llama-b10666-rocm/llama-perplexity -m <gguf> -f eval-data/kl-eval-<slice>.txt
  -ngl 99 -t 16 -c 512 -b 512 --kl-divergence --kl-divergence-base <ref.kld>`
- Domain→slice: wiki_test=kl-eval-64k.txt, wiki_valid=kl-eval-valid-64k.txt,
  chinese=kl-eval-cn-64k.txt, code=kl-eval-code-64k.txt, agent_chat=kl-eval-agent-64k.txt.
- Serial discipline (27B full offload), no concurrent heavy jobs.
- Failed runs are recorded as-is; no post-hoc point substitution.
- Every new artifact: SHA256 + size recorded before eval.

## 7. Analysis rules (pinned before results)

Curve per model: point = (actual_bytes, macro_kld, macro_same_top, kind, name).
macro = equal-weight mean of the 5 per-domain values; for reuse points the recomputed
macro must equal the published one within 0.005 pp / 5e-7 KL, else the point is flagged.

**Layer ① Top@KL crossing (per model):** *(Amendment-0, recorded before any new
point was evaluated — the original Pareto-frontier wording degenerates when a
single point globally dominates the quality plane, which the reused p4 curve
shows is the case here: IQ4_XS has both the lowest KL and the highest Top, so a
quality-Pareto filter collapses the curve and leaves 0.10/0.15/0.20 unbracketed.
The planner-approved semantic in the stage board is "Top@KL 插值 crossing", i.e.
the curve value at the threshold.)* Rule: on the raw curve sorted by macro_kld,
take the nearest point below t (max KL ≤ t) and nearest point above t (min KL > t);
Top@t = linear interpolation between them; record both anchor names. Missing side
→ null with reason. Known non-monotone presets (e.g. Q3_K_S) may serve as an
anchor; this biases the estimate conservatively and anchor names are always
reported so the planner can see it.

**Layer ② window health (per model):** points with macro_kld ∈ W_t; report n, kind
composition, P5/P10 (linear-interpolated percentiles) of macro_same_top, min, max.
n < 2 → null (amendment escape hatch: one extra batch chosen strictly inside the
underserved window, logged as amendment-1).

**Layer ③ equal-model aggregate:** unweighted mean of per-model Top@t, P5, P10
(null-safe; report n contributing).

**Mechanical verdict per (model, target)** — preliminary evidence only, planner judges:
- `supported`    : Top@t_interp ≥ candidate AND window P10 ≥ candidate
- `at-risk`      : Top@t_interp ≥ candidate AND window P10 < candidate
- `violated`     : Top@t_interp < candidate

Forbidden: pooled cross-model percentiles; any post-hoc window resize.

## 8. Outputs

```
experiments/2026-09-02-m2-topkl-calibration/
  PREREGISTRATION.md            ← this file
  refs-granite/bf16-<domain>.kld
  logs/                         ← raw llama-perplexity / quantize logs
  results/calibration-points-orcarouter.json
  results/calibration-points-granite.json
  results/candidate-calibration-v1.json
  results/candidate-calibration-v1.md
```

## 9. Amendment-1 (Quality window under-coverage, logged before the extra points ran)

Measured anchors (13:45–13:53 local): Q4_K_M = 0.065816 / 93.79 %, Q5_K_S =
0.044339 / 95.59 %, Q5_K_M ≈ 0.038 (all five domains complete; below-window),
FIT-14G five domains complete (see results). The pre-registered Quality window
[0.0425, 0.0575] holds only **n=1** (Q5_K_S): the standard preset ladder has a
hole between Q4_K_M (16.55 GB) and Q5_K_S (18.68 GB) — no preset can land near
0.05 on this model. Per §7's escape hatch, one extra batch is authorised,
chosen strictly inside the underserved window:

- New analysis pair `Q4_K_M→Q5_K_S` (`fit analyze` on the pinned BF16 +
  imatrix, completed 14:0x local, exit 0).
- Two balanced-policy FIT points: **M2A target 17,800,000,000 B** (linear
  estimate ≈ 0.052, convexity-adjusted ≈ 0.050–0.055) and **M2B target
  18,400,000,000 B** (linear ≈ 0.047, adjusted ≈ 0.044–0.049). Zero-byte size
  gates; same eval protocol; failures recorded as-is.

Rationale for disclosure: anchor values were necessarily known when the
amendment targets were chosen — that is the nature of a coverage-repair batch;
the window itself was fixed in §2 before any data existed, and no window
resize occurs.

## 10. Addendum — third DEV family (Ling-3.0-tiny, added before its points ran)

Per planner directive (GPT 2026-09-02: "补至少一个第三 DEV family"), the owner
selected **`SC117/Ling-3.0-tiny-abliterated-APEX-GGUF`** (base `inclusionAI/Ling-3.0-tiny`,
**bailing-moe MoE**, ~7.9B params; BF16 15.8 GB + imatrix.dat shipped in the repo).
Deviation from the planner's "standard dense" framing is deliberate: this yields
three architecturally distinct DEV families (Qwen27B = SSM-hybrid, Granite8B = dense,
Ling = MoE), which strengthens the cross-architecture generality of the Same-top
guard (aligns with G11/M11 goals). Ling was never used in Refine/Fidelity dev loop →
treat as a fresh DEV family for calibration; it is **not** an M11 sealed family.

Hereby marked DEV for the whole project (like Qwen/Granite).

Pinned before any Ling point ran (same §2 windows, same §6 protocol, eval-v1 slices):
- References: fresh two-phase generation from the shipped BF16 (no P4/m16 lineage).
- Curve points: preset ladder **IQ2_XXS, IQ2_XS, IQ2_S, IQ2_M, Q2_K_S, Q2_K, IQ3_XXS,
  IQ3_XS, Q3_K_S, IQ3_S, IQ3_M, IQ4_XS, Q4_K_S, Q4_K_M, Q5_K_S, Q5_K_M, Q6_K, Q8_0**
  (18), quantized from the BF16 with the repo `imatrix.dat`; plus the **4 shipped APEX
  fidelity tiers** (Quality / Balanced / Compact / Mini) as product-fit anchors.
- Imatrix: `Ling-3.0-tiny-abliterated-imatrix.dat` (the imatrix the product used).
- Same analysis layer (Top@KL crossing / window P5,P10 / equal-model aggregate),
  which becomes a **3-model aggregate** automatically once Ling points exist.
- Failures recorded as-is; big GGUFs under artifacts/fit/m2-calib/ (gitignored).

## 11. Amendment-2 — Ling window fill (planner-approved, targets pinned before any run)

GPT ruling 2026-09-02: global Same-top floor hypothesis REJECTED; contract becomes
Global Fidelity KL Core + architecture-specific Guard Profiles. Ling = `bailingmoe-candidate`
(NOT moe-v1). The 0.05 / 0.15 health windows are empty (n=0, preset-cliff gaps
0.0346→0.0741 and 0.1129→0.1882); a FIT-point batch is approved to fill them.

**Pinned before any metric was run** (linear map of the known anchors to window edges,
one-shot batch, no post-hoc target adjustment):

- Pair A `Q4_K_M→Q5_K_S` (anchors 4.824 GB→0.0741 / 5.483 GB→0.0346), window image
  ≈ [5.10, 5.35] GB:
  - L-A1 target **5,120,000,000 B** (lin. est. KL ≈ 0.057)
  - L-A2 target **5,230,000,000 B** (lin. est. KL ≈ 0.050)
  - L-A3 target **5,330,000,000 B** (lin. est. KL ≈ 0.044)
- Pair B `IQ3_M→IQ4_XS` (anchors 3.557 GB→0.1882 / 4.288 GB→0.0902), window image
  ≈ [3.68, 4.01] GB:
  - L-B1 target **3,720,000,000 B** (lin. est. KL ≈ 0.169)
  - L-B2 target **3,850,000,000 B** (lin. est. KL ≈ 0.151)
  - L-B3 target **3,980,000,000 B** (lin. est. KL ≈ 0.134)

Rules: `fit analyze` per pair (bailingmoe), `fit plan --policy balanced` with the exact
pinned targets, zero-byte size gates, tmpfs hot loop, failures recorded as-is. If the
MoE KL-vs-size curve is non-linear and a target lands outside its window, that is
recorded as-is (one-shot batch; no retargeting). Plans are generated before any
perplexity run (planner discipline: pin first, run second).

**Amendment-2 execution notes (recorded before evaluation):**
1. Pair A realized as **IQ4_XS→Q6_K** (not Q4_K_M→Q5_K_S): the fit v0.1 size predictor
   does not support q5_0/q5_1 destination qtypes, which appear in the Q4_K_M / Q4_K_S /
   Q5_K_S preset mixes. IQ4_XS→Q6_K is analyzable and its linear window image still
   brackets all three pinned targets (lin. est. 0.059 / 0.055 / 0.051). Pinned targets
   unchanged. This predictor coverage gap is an M7 regression-suite candidate.
2. Measured systematic bailingmoe size offset: all six plans quantized to exactly
   **predicted − 480 bytes** (6/6, constant) — a structural predictor/quantizer
   mismatch for this architecture, not allocator noise. Size gate for this batch is
   therefore `actual == predicted − 480` exactly (stricter than a tolerance band);
   any other deviation fails. Toolchain finding recorded for M7.

Big GGUFs: tmpfs-only, hash-recorded, deleted after eval; durable records = logs +
manifest (synced to the experiment dir and TF card when mounted).
