# M15 Matched-Random Baseline at FIT-25 and FIT-75 — Preregistration + Protocol Lock

Date: 2026-08-29
Status: PREREGISTERED AND LOCKED — this document, the seeds, the generation
method, the gates, and the holdout-5 slice hashes below were committed before
any M15 plan was generated, before any reference logits were built, and
before any evaluation ran. No element may be changed after results are seen.

## Purpose

M14 (D-0020) rejected the positional mechanism; attribution of v0.1b's
confirmed FIT-50 gain is unresolved. Two open questions drive this milestone:

1. **H75 — allocation sensitivity collapse.** FIT-75's original/v0.1b
   difference is a practical tie on three independent holdout sets. If fresh
   matched random seeds land at the same level, high-budget insensitivity is
   confirmed and the product conclusion becomes "high budgets do not need a
   clever allocator".
2. **H25 — original advantage exceeds random variance.** M11's warning signal
   (one random seed nearly tied v0.1b at FIT-50) makes it necessary to show
   the original utility's FIT-25 advantage is larger than sampled random
   variance.

## Frozen design

- **Paired random trajectories:** each seed first fixes one deterministic
  SHA-256 priority order over all safe promotion candidates
  (`optimize_random(seed=…)`); the FIT-25 plan is the greedy prefix consumed
  at the FIT-25 target, and the FIT-75 plan continues the same order at the
  FIT-75 target, so plan75 ⊇ plan25 by construction. The generator asserts
  the superset property.
- **Seeds (exact strings):** `m15-v1`, `m15-v2`, `m15-v3`.
- **Targets:** FIT-25 = 13,206,283,232 bytes; FIT-75 = 14,457,099,232 bytes
  (the M9 positions). Lower preset IQ3_M, upper IQ4_XS, canonical imatrix,
  pinned runtime — all as in every prior milestone.
- **Artifacts:** six new random GGUFs (r1/r2/r3 × 25/75). Reference points
  O25/B25/O75/B75 (existing artifacts, hashes already recorded) are
  re-evaluated on the same holdout — all statistics are relative to the same
  holdout's O/B; no absolute KL value is preregistered anywhere.

## Holdout-5 slices (frozen)

Five new 64 KiB slices from the same five sources, disjoint from the M9, M11,
M13, and M14 slices. SHA-256 values are locked in `holdout5-slices.json`:

```text
wiki_test   d1254e2d1df0745a…  offset 458,752   (prior: 0 / 655,360 / 983,040 / 327,680)
wiki_valid  33319c28128daa2f…  offset 458,752   (prior: 0 / 655,360 / 983,040 / 327,680)
chinese     5a4a0af546d3f365…  offset 8,800,000 (prior: 3,173,914 / 4,000,000 / 6,500,000 / 7,700,000)
code        86c89aa2119f2d9c…  offset 33,500,000 (prior: 12,536,883 / 18,000,000 / 28,000,000 / 21,000,000)
agent_chat  5d35521a42e8ab65…  offset 1,500,000 (prior: 998,087 / 400,000 / 1,200,000 / 1,400,000)
```

Fresh BF16 references (source SHA-256 `8a033407…`, protocol
`-ngl 99 -t 16 -c 512 -b 512`). Ten artifacts × five domains = 50 runs.

## Preregistered hypotheses and gates

delta δ = 1% relative macro-KL ROPE (frozen; same convention as M13/M14).

**H75 — sensitivity collapse (all three must hold):**

```text
H75  = (KL_O75 + KL_B75) / 2                      (macro over five domains)
G75a: |KL_O75 - KL_B75| / H75        ≤ 1%
G75b: |KL_randomMean75 - H75|        ≤ 1%
G75c: (max - min)(random macro KL75) / H75 ≤ 2%
```

Guardrail: no random artifact worse than the worse of O75/B75 by more than
25% in any domain. Secondary (reported, not gated): normalized random spread
at 75 < at 25 (paired seeds).

**H25 — original advantage exceeds random variance (all three must hold):**

```text
H25   = KL_O25 (macro)
G25a: KL_randomMean25 - H25 ≥ 1% of H25
G25b: original beats at least 2 of 3 random seeds on macro KL
G25c: any seed that beats original stays within the 1% ROPE
```

Guardrail: no random artifact worse than the worse of O25/B25 by more than
25% in any domain. If original beats 3/3 seeds and G25a holds, record as
strong support.

**Decision rule (frozen):**

- H75 passes → high-budget allocation insensitivity is accepted; the product
  may use a trivial fill strategy at high budgets.
- H25 passes → the original utility's FIT-25 advantage is accepted as real
  (beyond one-seed luck).
- Then write D-0021 (allocator + budget-policy freeze for the first
  cross-model validation) and only then reveal Granite. No new
  original-model experiments after M15 unless a preregistered gate triggers
  its written failure branch; a Granite failure is recorded as a failure.

## Execution notes

- Quantization: `llama-quantize --imatrix imatrix_unsloth.gguf
  --tensor-type-file <file> BF16 OUT IQ3_M`, exact predicted sizes verified.
- 50 evaluation runs; raw logs local under `artifacts/logs/`; parsed metrics
  to `m15-results.json`; verdict computed mechanically by
  `scripts/evaluate_m15_gate.py`.
- Elapsed seconds are wall-clock from log file timestamps; max RSS is not
  instrumented.
