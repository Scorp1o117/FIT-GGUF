# M14 Role-Matched Early/Late Crossover Ablation — Preregistration

Date: 2026-08-28
Status: PREREGISTERED — written and committed before any swap recipe was
generated, before the fourth holdout set was created, and before any M14
evaluation ran. The statistic, delta, gates, and failure interpretation below
are frozen.

## Background (free diagnostic, run before this preregistration)

`scripts/diagnose_recipe_overlap.py` (`recipe-overlap.json`) measured the
byte-weighted overlap of the original-utility and v0.1b upgrade sets:

```text
FIT-25 Jaccard 0.328  (orig E/L 49M/574M, v0.1b E/L 308M/316M)
FIT-50 Jaccard 0.464  (orig E/L 172M/1,079M, v0.1b E/L 628M/620M)
FIT-75 Jaccard 0.689  (orig E/L 634M/1,241M, v0.1b E/L 934M/940M)
```

Overlap rises monotonically with budget, supporting the high-budget
saturation hypothesis behind the M13 FIT-75 tie (D-0019). Early/late halves
are blocks 0-31 and 32-63 throughout.

## Design: bidirectional transition-matched crossover

Matching unit: (role, from_qtype->to_qtype) class, byte-exact. A tensor may
only be exchanged with a tensor of the same role and the same qtype
transition; a class exchanges equal byte totals or nothing. No filler tensors
are allowed. Unmatched classes stay at their skeleton assignment.

Frozen artifacts (all four at FIT-50, primary budget):

- **O** = original FIT-50 (`e4fe1c46…`), skeleton;
- **B** = v0.1b FIT-50 (`7cfa1b91…`), skeleton;
- **O→E** = original skeleton with a byte-exact subset of its late-block
  upgrades (per class) replaced by v0.1b's early-block upgrades in the same
  class; total, per-role, and per-transition upgrade bytes unchanged;
- **B→L** = v0.1b skeleton with a byte-exact subset of its early-block
  upgrades replaced by original's late-block upgrades in the same class;
- **SHUF-50** = negative control derived from the O→E exchange volumes: same
  skeleton, same per-class exchange byte totals, same early/late distribution,
  but both the removed (late) and added (early) tensors are chosen by SHA-256
  priority (seed `m14-shuffle-50`) within the same class pools instead of by
  the planners' preferences. Classes without an alternative byte-exact
  subset keep the O→E choice.

Secondary budget (saturation diagnostic, ungated):

- **O→E-75** and **B→L-75** built by the same rule at the FIT-75 position.

Predicted sizes of every new artifact must equal their skeleton's predicted
size exactly (per-class byte-exact exchanges); a mismatch blocks execution.

## Deterministic exchange construction (frozen before any generation)

Per (role, transition) class:

1. Removal pool = skeleton's late-block upgrades in the class (O→E) or
   skeleton's early-block upgrades (B→L); addition pool = the other plan's
   opposite-half upgrades in the same class whose tensors are NOT upgraded in
   the skeleton.
2. Find the maximum total byte sum s achievable simultaneously by a subset of
   the removal pool and a subset of the addition pool (exact subset-sum
   equality; integer DP on gcd-scaled values). Classes with s = 0 are skipped.
3. Subsets are reconstructed greedily over items ordered by tensor name
   (canonical plans) or by SHA-256 of `m14-shuffle-50:<tensor>` (the shuffle
   control), both ascending. If both orders yield the same unique subset, the
   control keeps it — the control differs only where alternatives exist.

The O→E and B→L per-class exchange totals are those maximum values; SHUF-50
reuses exactly O→E's per-class totals s. Predicted artifact size must equal
the skeleton's predicted size byte-for-byte.

## Data: fourth holdout set (untouched)

Five new 64 KiB slices from the same sources, disjoint from the M9, M11, and
M13 slices (arithmetically and by sampled-window checks):

| Domain | Source | M9 | M11 | M13 | M14 |
| --- | --- | ---: | ---: | ---: | ---: |
| wiki_test | wiki.test.raw | 0 | 655,360 | 983,040 | 327,680 |
| wiki_valid | wiki.valid.raw | 0 | 655,360 | 983,040 | 327,680 |
| Chinese | combined_cn_medium.parquet | 3,173,914 | 4,000,000 | 6,500,000 | 7,700,000 |
| code | code_medium.parquet | 12,536,883 | 18,000,000 | 28,000,000 | 21,000,000 |
| agent_chat | agentworld_clean_quick.txt | 998,087 | 400,000 | 1,200,000 | 1,400,000 |

Fresh BF16 references are generated on these slices (source SHA-256
`8a033407…`, protocol `-ngl 99 -t 16 -c 512 -b 512`). All five new artifacts
(O→E, B→L, SHUF-50 at FIT-50; O→E-75, B→L-75 at FIT-75) are evaluated on all
five domains: 25 runs. O and B results come from the same runs' sibling
artifacts — note O and B at FIT-50/75 are NOT re-evaluated here; the
crossover statistics use O and B measured on the SAME holdout-4 slices, so
O50, B50, O75, B75 are added to the run matrix: **9 artifacts × 5 domains =
45 runs** (O/B exist already; only their evaluation is new).

## Statistics

With delta δ = 1% relative macro-KL (frozen practical-equivalence ROPE):

```text
term(O→E) = (KL(O50) - KL(O→E50)) / KL(O50)
term(B→L) = (KL(B→L50) - KL(B50)) / KL(B50)
S50 = (term(O→E) + term(B→L)) / 2        # > 0 means early placement helps
```

Per-domain synthetic effect `e_d` is the same average computed per domain.
FIT-75 statistics (S75) are computed identically for the saturation
diagnostic.

## Preregistered gates

- **Gate 1 (mechanism, primary):** S50 ≥ 1% AND min(term(O→E), term(B→L)) ≥ -1%
  (one arm may sit inside the ROPE, but neither may be clearly reversed).
- **Gate 2 (domain robustness):** e_d ≥ -1% in at least 4 of 5 domains, and no
  swap artifact exceeds its skeleton's KL by more than 25% in any cell.
- **Gate 3 (negative control):** (KL(SHUF-50) - KL(O→E50)) / KL(O→E50) ≥ 1% —
  the real swap must beat the matched shuffle clearly, otherwise the gain is
  not attributable to early/late position.

Secondary, ungated preregistered predictions: S75 within ±1% of zero
(saturation); S50 > S75; recipe overlap J75 > J50 > J25 (already observed).

## Decision rule (frozen)

- Gates 1 ∧ 2 ∧ 3 -> early-vs-late position is accepted as a causal component
  of v0.1b's confirmed FIT-50 gain; proceed to matched random seeds at
  FIT-25/75 (variance baseline), then the D-0020 allocator freeze and M15
  cross-model validation (Granite).
- Otherwise -> record that early/late position is NOT the confirmed mechanism.
  No allocator promotion, no threshold or quarter-weight changes; proceed to
  the matched random-seed baseline with attribution unresolved, and treat any
  future allocator redesign as requiring a new preregistered gate.
- FIT-75 results gate nothing and must not trigger design changes.

## Execution notes

- New quantizations: O→E50, B→L50, SHUF-50, O→E-75, B→L-75 (five), each with
  `--imatrix imatrix_unsloth.gguf --tensor-type-file <file> IQ3_M`, sizes
  verified against skeleton predictions with zero-byte error.
- Raw logs stay local under `artifacts/logs/`; parsed metrics go to
  `m14-results.json`; the verdict is computed mechanically by
  `scripts/evaluate_m14_gate.py`. Elapsed seconds are wall-clock from log
  timestamps; max RSS is not instrumented.
