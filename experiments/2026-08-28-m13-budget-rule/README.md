# M13 Budget-Conditional Allocator Rule — Preregistration

Date: 2026-08-28
Status: PREREGISTERED — written and committed before the third holdout set was
created, before any holdout-3 reference logits were generated, and before any
M13 evaluation ran. The threshold, gates, and failure branch below are frozen.

## Purpose

M11/M12 established (D-0018) that the allocation trade-off is
budget-dependent: the original utility wins at FIT-25 while frozen
block-balanced v0.1b wins at FIT-50 (confirmed on untouched holdout) and
FIT-75 (design domains). This milestone turns that observation into a single
deployable rule and validates it independently on a third, fresh holdout set.

## Frozen decision rule

```text
budget ratio r = (target_bytes - lower_preset_bytes) / (upper_preset_bytes - lower_preset_bytes)

r <  0.50  ->  original M7 utility allocation
r >= 0.50  ->  block-balanced v0.1b allocation
```

At the M9 curve positions this selects: FIT-25 -> original, FIT-50 -> v0.1b,
FIT-75 -> v0.1b. The 0.50 threshold is frozen. Post-hoc threshold movement is
forbidden regardless of outcome.

## Frozen artifacts (all six already exist; no requantization)

| Variant | SHA-256 |
| --- | --- |
| original FIT-25 | `d757266985bdbfe8a2df7a1d6f209effaf192036ece6be9b3391cb3c2dcef4e2` |
| original FIT-50 | `e4fe1c46ab89c8b6343203168ebeec699372c2fb21f411c61c47edc2e1f33306` |
| original FIT-75 | `4a05fccd1b0f77c51ff8d7f4be43663e85adc07f7f6ab1eece0ce5f14f00fad1` |
| v0.1b FIT-25 | `191781ca4d12eaec7c0704828d695c722f302d011c028d6b813df1ffb1df35a7` |
| v0.1b FIT-50 | `7cfa1b91600115c046cb9afcae8347adc7de5a77b0d880b47a956cb8a4799a07` |
| v0.1b FIT-75 | `6023d58ab00876671af1eee9ece957fdea7b69d38541de88e7664824bd4f950e` |

Quarter weights, utility coefficients, and recipes are frozen. No new
quantization is allowed in M13.

## Holdout set 3

Five new 64 KiB slices (65,536 characters, UTF-8), from the same five source
files as M9/M11, at offsets disjoint from BOTH earlier slice sets:

| Domain | Source | M9 offset | M11 offset | M13 offset |
| --- | --- | ---: | ---: | ---: |
| wiki_test | wiki.test.raw | 0 | 655,360 | 983,040 |
| wiki_valid | wiki.valid.raw | 0 | 655,360 | 983,040 |
| Chinese | combined_cn_medium.parquet | 3,173,914 | 4,000,000 | 6,500,000 |
| code | code_medium.parquet | 12,536,883 | 18,000,000 | 28,000,000 |
| agent_chat | agentworld_clean_quick.txt | 998,087 | 400,000 | 1,200,000 |

The slice generator asserts arithmetic disjointness from the two documented
offset ranges and that sampled windows of the M9 and M11 slices are absent
from each new slice. SHA-256 values are recorded in `holdout3-slices.json`.
BF16 references are regenerated on these slices from the same BF16 source
(SHA-256 `8a033407…`), protocol `-ngl 99 -t 16 -c 512 -b 512`.

## Preregistered gates

Primary metric: mean KL on holdout set 3, evaluated for all six artifacts
(3 budgets x 2 allocators) under the identical protocol.

- **Gate 1 (direction reproduction):** at every budget point the rule-selected
  policy has the lower macro KL across the five domains: at FIT-25 original <
  v0.1b; at FIT-50 v0.1b < original; at FIT-75 v0.1b < original.
- **Gate 2 (composite win):** the rule's macro KL over all 15 budget×domain
  cells is strictly lower than both pure strategies' 15-cell macro KL
  ("all original" and "all v0.1b").
- **Gate 3 (no catastrophic per-cell regression):** in no budget×domain cell
  does the selected policy's KL exceed the unselected policy's KL by more
  than 25%.
- Same-top and PPL are reported as diagnostics only and gate nothing.

Decision rule (frozen):

- Gates 1 ∧ 2 ∧ 3 -> the budget-conditional rule is frozen as the first
  accepted deployment policy for this model; proceed to M14 (matched random
  seeds at FIT-25/FIT-75) and then M15 (second model family).
- Otherwise -> the ONLY permitted next step is the role-matched early/late
  block swap ablation for attribution. Moving the 0.50 threshold, re-tuning
  quarter weights, or redesigning gates after seeing results is forbidden.

## Execution notes

- 6 artifacts × 5 domains = 30 evaluation runs; raw logs stay local under
  `artifacts/logs/`, parsed metrics go to `m13-results.json`, and the gate
  verdict is computed mechanically by `scripts/evaluate_m13_gate.py`.
- Elapsed seconds are wall-clock from log file timestamps; max RSS is not
  instrumented (same convention as M11/M12).

## Provenance verification (executed)

- BF16 source SHA-256 re-verified: `8a033407…` OK.
- All six frozen artifacts re-verified against the preregistered SHA-256
  table: OK.
- Holdout-3 BF16 reference PPL (fresh slices, not comparable to M9/M11
  values): wiki_test 7.5783 ± 0.22911, wiki_valid 8.1562 ± 0.24990,
  Chinese 9.5948 ± 0.35326, code 5.1106 ± 0.14331,
  agent_chat 6.3469 ± 0.19118.

## Results

Mean KL divergence on holdout set 3 (lower is better):

| Domain | orig-25 | v0.1b-25 | orig-50 | v0.1b-50 | orig-75 | v0.1b-75 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| wiki_test | 0.040599 | 0.053018 | 0.033985 | 0.042304 | 0.024281 | 0.027223 |
| wiki_valid | 0.046274 | 0.055791 | 0.036951 | 0.044391 | 0.026791 | 0.029583 |
| Chinese | 0.162215 | 0.170637 | 0.158703 | 0.145871 | 0.125438 | 0.115889 |
| code | 0.156527 | 0.158683 | 0.152543 | 0.133206 | 0.117676 | 0.121291 |
| agent_chat | 0.082629 | 0.086228 | 0.076241 | 0.064414 | 0.055351 | 0.056293 |
| Macro | 0.097649 | 0.104871 | 0.091685 | 0.086037 | 0.069907 | 0.070056 |

## Preregistered gate verdict

- **Gate 1 (direction reproduction): FAIL.** FIT-25 reproduces (original wins,
  -6.89% macro); FIT-50 reproduces (v0.1b wins, -6.16% macro); **FIT-75 does
  not reproduce**: original 0.069907 vs v0.1b 0.070056 — v0.1b is 0.21% worse,
  statistically a tie (per-domain KL uncertainties are ~1-2% relative), but
  the preregistered gate required a v0.1b win and did not get one. The
  FIT-75 v0.1b advantage seen in M12 on the design domains does not transfer
  to untouched data.
- **Gate 2 (composite win): PASS.** The rule's 15-cell macro KL 0.084581
  beats both pure strategies: all-original 0.086414 (-2.12%) and all-v0.1b
  0.086988 (-2.77%).
- **Gate 3 (per-cell guard): PASS.** Worst selected-policy cell is
  fit50/wiki_test at +24.48% versus the unselected policy, inside the
  preregistered 25% bound (note: this is close to the bound).

**Decision: NOT ACCEPTED**, exactly per the frozen rule (all three gates were
required). The failure branch is the role-matched early/late block swap
ablation. The 0.50 threshold stays where it is; no quarter-weight or gate
changes are permitted in response to this result.

## Interpretation (diagnostic, not gate-relevant)

The budget-conditional rule is directionally right where it matters most
(FIT-25 and FIT-50 reproduce cleanly, and the composite rule beats both pure
strategies on fresh data — real deployable value). The instability is confined
to FIT-75, where the two allocators are within noise of each other on
untouched data (0.21% macro apart while M12 showed -4.75% on the design
domains). The honest reading: v0.1b's FIT-75 advantage was partly adaptive,
and near the upper budget the allocator choice barely matters because both
recipes sit close to the IQ4_XS ceiling. Attribution of where v0.1b's gains
actually come from is now the blocking question — hence the preregistered
role-matched swap ablation.

Machine records: `holdout3-slices.json`, `m13-results.json`,
`gate-verdict.json`.
