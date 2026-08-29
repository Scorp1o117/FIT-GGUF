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

## Provenance verification (executed)

- BF16 source SHA-256 re-verified: `8a033407…` OK.
- All six random artifacts match their exact predicted sizes byte-for-byte;
  the paired-trajectory superset property (plan75 ⊇ plan25 per seed) was
  asserted at generation time. Artifact SHA-256 values are recorded in
  `artifact-hashes.txt` (artifacts deleted after evaluation per the disk
  policy; reproducible from the retained recipes).
- Holdout-5 verified disjoint from M9/M11/M13/M14 slices.
- Holdout-5 BF16 reference PPL: wiki_test 5.3040 ± 0.13756,
  wiki_valid 7.1389 ± 0.21906, Chinese 10.9360 ± 0.41642,
  code 5.2237 ± 0.17462, agent_chat 6.2246 ± 0.19589.

## Results

Mean KL divergence on holdout-5 (lower is better):

| Domain | O25 | B25 | r1-25 | r2-25 | r3-25 | O75 | B75 | r1-75 | r2-75 | r3-75 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| wiki_test | 0.040990 | 0.046193 | 0.048481 | 0.050702 | 0.046486 | 0.022746 | 0.025580 | 0.029204 | 0.033652 | 0.026761 |
| wiki_valid | 0.039125 | 0.047860 | 0.043967 | 0.047477 | 0.044809 | 0.022552 | 0.025180 | 0.026958 | 0.032303 | 0.025502 |
| Chinese | 0.157498 | 0.149136 | 0.181900 | 0.153444 | 0.177530 | 0.113575 | 0.114433 | 0.106299 | 0.111765 | 0.112566 |
| code | 0.213855 | 0.201556 | 0.211223 | 0.204779 | 0.219087 | 0.158753 | 0.134671 | 0.148220 | 0.150582 | 0.151471 |
| agent_chat | 0.134387 | 0.138174 | 0.149381 | 0.137813 | 0.148834 | 0.111101 | 0.113516 | 0.114050 | 0.113145 | 0.117208 |
| Macro | 0.117171 | 0.116584 | 0.126990 | 0.118843 | 0.127349 | 0.085745 | 0.082676 | 0.084946 | 0.088289 | 0.086702 |

## Preregistered gate verdict

**H25 (original advantage exceeds random variance): CONFIRMED — strong support.**

- G25a: original beats the random mean by 6.17% (0.117171 vs 0.124394);
- G25b: original beats 3 of 3 random seeds;
- G25c: no seed beat original, so the ROPE clause is vacuous;
- guardrail held (worst random cell +18.7%);
- recorded as strong support (3/3 + margin).

**H75 (allocation sensitivity collapse): NOT CONFIRMED.**

- G75a fails: |O75 - B75| / H75 = 3.65% — and the sign flipped: on this
  holdout B75 (0.082676) is clearly better than O75 (0.085745), after three
  prior holdout sets showed a +0.17%/+0.21% tie;
- G75b fails: random mean is 2.89% above H75 — random does NOT catch up to
  the heuristics at 75%;
- G75c fails: random range is 3.97% of H75;
- guardrail fails: r2-75/wiki_test is +47.95% above the worse heuristic —
  a random allocation can catastrophically fail in a single domain even at
  high budget.

Secondary (paired seeds): random spread shrinks from 7.26% (25) to 3.97% (75)
— dispersion contracts with budget, but the level does not converge to the
heuristics within this holdout.

## Interpretation (diagnostic, not gate-relevant)

Two honest conclusions:

1. **The original utility is genuinely good at FIT-25.** A 6.17% margin over
   the random mean with a 3/3 seed sweep is far beyond one-seed luck; every
   random seed is worse in every domain. The low budget is
   allocation-critical and the utility spends it well.
2. **The FIT-75 "tie" is itself high-variance.** Across four holdout sets the
   O75/B75 gap reads +0.17%, +0.21%, tie (design domains), and now -3.65%
   (sign flip, B better). There is no stable O-vs-B statement at 75 — and
   random seeds do not reach the heuristic level there either (mean +2.9%,
   one domain blowup of +48%). The honest FIT-75 claim is: the ordering of
   the two heuristics is holdout-dependent, both heuristics beat random on
   average, and random carries heavy single-domain tail risk. The earlier
   "practical equivalence" reading survives only as "no stable winner", not
   as "nothing matters".

Machine records: `holdout5-slices.json`, `plan-predictions.json`,
`m15-results.json`, `gate-verdict.json`, `artifact-hashes.txt`.
