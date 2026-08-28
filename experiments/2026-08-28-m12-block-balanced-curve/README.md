# M12 Block-Balanced v0.1b at FIT-25 and FIT-75

Date: 2026-08-28

## Scope

M11 confirmed the frozen block-balanced v0.1b allocator on untouched holdout
data at FIT-50 (gates A/B/C all passed). Per the preregistered M11 decision
rule, this milestone extends the same frozen allocator — no quarter-weight,
coefficient, or rule changes — to the other two M9 curve positions and
evaluates the full v0.1b curve against the M9 references on the original five
M9 slices, using the pinned protocol `-ngl 99 -t 16 -c 512 -b 512`.

Plan generation used `scripts/generate_m12_block_balanced.py`. The driver was
validated against known ground truth before use: it reproduces the retained
M10 v0.1b FIT-50 plan with byte-identical tensor-type files and the exact
13,828,987,872-byte prediction.

## Size results

| Variant | Target bytes | Predicted bytes | Actual bytes | Unused bytes | Overrides | Time (s) | SHA-256 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| bb FIT-25 | 13,206,283,232 | 13,205,208,032 | 13,205,208,032 | 1,075,200 | 252 | 260.4 | `191781ca4d12eaec7c0704828d695c722f302d011c028d6b813df1ffb1df35a7` |
| bb FIT-75 | 14,457,099,232 | 14,454,856,672 | 14,454,856,672 | 2,242,560 | 411 | 252.7 | `6023d58ab00876671af1eee9ece957fdea7b69d38541de88e7664824bd4f950e` |

Both actual sizes match predictions exactly (zero-byte error). Both are under
target and meet the M11 fine-fill bound of `unused <= max(16 MiB, 0.1%)`
(0.008% and 0.016% of target).

## Quality results (M9 slices, M9 BF16 references)

Mean KL divergence, original M9 curve versus v0.1b curve (lower is better):

| Domain | FIT-25 orig | FIT-25 v0.1b | FIT-50 orig | FIT-50 v0.1b | FIT-75 orig | FIT-75 v0.1b |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| wiki_test | 0.038943 | 0.047161 | 0.031555 | 0.037874 | 0.022408 | 0.025852 |
| wiki_valid | 0.037375 | 0.041870 | 0.030332 | 0.035114 | 0.021560 | 0.023862 |
| Chinese | 0.193422 | 0.190217 | 0.181044 | 0.159104 | 0.144548 | 0.130690 |
| code | 0.153504 | 0.165253 | 0.138484 | 0.142717 | 0.132087 | 0.121871 |
| agent_chat | 0.108142 | 0.115107 | 0.105204 | 0.092399 | 0.077999 | 0.077396 |
| Macro mean | 0.106277 | 0.111922 | 0.097324 | 0.093442 | 0.079720 | 0.075934 |

v0.1b macro change versus the original curve at matched size:

- FIT-25: **+5.31% (worse)** — v0.1b loses on 4 of 5 domains; only Chinese
  improves (-1.66%).
- FIT-50: **-3.99% (better)** — the M11-confirmed point; Chinese and
  agent_chat improve ~12.1%, both wiki domains regress.
- FIT-75: **-4.75% (better)** — Chinese -9.59%, code -7.73%, agent_chat
  -0.77%, wiki_test +15.37%, wiki_valid +10.68%.

Same-top differences stay below 0.6 percentage points everywhere and do not
contradict the KL ordering. Machine record:
`block-balanced-curve-results.json`.

## Interpretation

The block-balanced trade-off is **budget-dependent**. At a tight budget the
original utility's late-block concentration is worth more than balance (the
imatrix says the late blocks carry the most recoverable quality per byte, and
with only 625 MB of upgrades you cannot afford to spread evenly); at 50-75%
budgets, balance wins on macro KL and non-wiki domains. Both curves remain
strictly monotonic in size (all KL steps decrease from IQ3_M to IQ4_XS for
their respective allocators).

This is a diagnosis, not a promotion: no single allocator is accepted across
the curve, and the original utility still owns FIT-25. Any budget-conditional
allocator selection (or quarter-weight adaptation by budget) is a new
hypothesis that would need its own preregistered gate and untouched holdout
validation — that is optimizer-v2 territory and remains blocked by D-0017
until explicitly re-scoped.

## Acceptance

M12 evaluation is complete and recorded. The M11 positive-allocation claim is
now: at matched FIT-50 budget, the frozen v0.1b allocation beats both the
original utility and the random-seed mean on untouched data, while the
original utility remains better at FIT-25 and v0.1b better at FIT-75 on the
design domains.
