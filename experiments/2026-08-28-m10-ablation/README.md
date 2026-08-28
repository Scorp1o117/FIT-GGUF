# M10 FIT-50 Random Allocation Ablation

Date: 2026-08-28

## Question

Does the M7 imatrix-guided FIT-50 allocation outperform random promotion at
the same IQ3_M lower baseline and effectively identical byte budget?

## Controls

`optimize_random()` assigns every safe promotion a SHA-256 priority derived
from a recorded seed, tensor name, and destination qtype. It never reads the
candidate utility. Greedy packing then consumes the same FIT-50 target budget.

| Variant | Selected | Actual bytes | Difference from FIT-50 | SHA-256 |
| --- | ---: | ---: | ---: | --- |
| FIT-50 | 323 | 13,831,486,432 | 0 | see M9 |
| random v1 | 209 | 13,831,672,672 | +186,240 | `dba2ec218beeed0cc47fc54bbc76c9ae97d4e24298931d940da29b7365f94892` |
| random v2 | 256 | 13,831,684,832 | +198,400 | `2fd1a30b1b671cc223085d920d88f3239b9580d7076852ade74ebab9fcfcfeb4` |
| random v3 | 258 | 13,831,684,832 | +198,400 | `c0a119b8961bdec5402f41ee29fede3659e2005c90dc02434191dbafbd7dce40` |

Every random artifact has zero predicted-size error, 851 matching tensors, and
zero qtype mismatches. The random artifacts are 0.00135%-0.00143% larger than
FIT-50, so a FIT win cannot be attributed to extra bytes. Seeds are
`m10-fit50-v1`, `m10-fit50-v2`, and `m10-fit50-v3`.

## KL results

| Domain | FIT-50 | Random mean | Random range | FIT change vs random mean |
| --- | ---: | ---: | ---: | ---: |
| wiki_test | 0.031555 | 0.037187 | 0.034097-0.040089 | -15.14% |
| wiki_valid | 0.030332 | 0.036838 | 0.035910-0.038671 | -17.66% |
| Chinese | 0.181044 | 0.170672 | 0.153582-0.181432 | +6.08% |
| code | 0.138484 | 0.148607 | 0.142650-0.156387 | -6.81% |
| agent_chat | 0.105204 | 0.105055 | 0.096606-0.120448 | +0.14% |
| Macro mean | 0.097324 | 0.099672 | 0.094051-0.107405 | -2.36% |

Negative change means FIT is better. FIT macro KL beats random v1 by 9.39%
and v2 by 0.24%, but loses to v3 by 3.48%. Across the 15 domain/seed pairs,
FIT has the lower KL in 11. The strongest stable gains are the two wiki domains;
Chinese is worse than the random mean and random v3 is materially better.

FIT Same-top beats random in 13 of 15 domain/seed pairs. Its domain-level
advantages over the random means are +0.375, +0.586, +0.448, +0.491, and
+0.321 percentage points for wiki_test, wiki_valid, Chinese, code, and
agent_chat respectively. This is encouraging, but it does not cancel the KL
counterexample.

Complete parsed measurements and timing are in `random-fit50-results.json`;
raw logs are preserved locally under `artifacts/logs/`.

## Allocation diagnosis

FIT-50 spends 620,733,440 bytes in blocks 48-63 and only 47,047,680 bytes in
blocks 0-15. Random v3 is much more balanced, spending 298,005,760,
305,763,200, 327,612,800, and 319,427,840 bytes across the four 16-block
quarters.

The equal-cost FIT-only versus v3-only difference is more extreme:

- FIT-only: 583,317,760 bytes, including 301,305,600 in blocks 48-63;
- v3-only: 583,516,160 bytes, including 270,786,560 in blocks 0-15 and none in
  blocks 48-63.

The FIT-only median role-relative importance is 1.272 versus 0.350 for v3-only,
so the optimizer is following its configured proxy. The observed Chinese
counterexample means that proxy is not sufficiently domain-robust. This is a
correlation, not yet proof that block position is causal.

## Block-balanced diagnostic

The targeted follow-up assigned approximately one quarter of the upgrade-byte
budget to each 16-block range, retained the same imatrix utility ordering
within each range, and used the global utility order only to fill quota
fragments.

| Field | Value |
| --- | ---: |
| Selected upgrades | 336 |
| Target bytes | 13,831,691,232 |
| Predicted / actual bytes | 13,828,987,872 |
| Unused bytes | 2,703,360 |
| Quantization time | 264.84 s |
| SHA-256 | `7cfa1b91600115c046cb9afcae8347adc7de5a77b0d880b47a956cb8a4799a07` |

The actual output has zero predicted-size error and zero qtype mismatches over
851 tensors. Quarter costs are 312,427,520, 315,458,560, 311,895,040, and
308,331,520 bytes.

| Domain | Original FIT-50 KL | Balanced KL | Change |
| --- | ---: | ---: | ---: |
| wiki_test | 0.031555 | 0.037874 | +20.03% |
| wiki_valid | 0.030332 | 0.035114 | +15.77% |
| Chinese | 0.181044 | 0.159104 | -12.12% |
| code | 0.138484 | 0.142717 | +3.06% |
| agent_chat | 0.105204 | 0.092399 | -12.17% |
| Macro mean | 0.097324 | 0.093442 | -3.99% |

Negative change is better. Block balancing sacrifices the original FIT's
strong wiki behavior but materially improves Chinese and agent_chat. Its macro
KL beats all three random-seed macros and the random mean by 6.25%. It beats
the per-domain random mean in four of five domains; wiki_test remains worse.
Full parsed results are in `block-balanced-fit50-results.json`.

This confirms that late-block concentration is a real cross-domain allocation
trade-off. It does not independently validate block balancing, because the
strategy was designed after inspecting the same five domains used to evaluate
it. Treat it as the frozen v0.1b hypothesis and confirm on untouched holdout
slices before promoting it.

## Conclusion and handoff

M10 diagnosis succeeded, but M10 acceptance and positive allocation evidence
remain open. Original FIT is strongly wiki-oriented; deterministic block
balancing improves macro and non-wiki behavior but needs fresh validation.

Next: freeze v0.1b, generate untouched holdout slices, compare original FIT-50,
block-balanced FIT-50, and at least three random seeds, then extend the winning
policy to FIT-25/FIT-75. No optimizer-v2 work is justified yet.
