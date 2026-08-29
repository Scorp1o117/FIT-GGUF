# P5: K-Free Redo of FIT-12.5G / 13G / 13.5G (Owner Directive)

Date: 2026-08-30. Status: preregistered before any execution step of this
experiment.

## Trigger and observed failure

Owner spotted that added size bought (almost) no quality, and directed:
"Q4KM以下把K系列移除试试，用IQ3-IQ4XS重做一次12.5 / 13 / 13.5".

Measured baseline (P4 `results/p4-results.json`, M9 macro):

| Tier | Pair | Actual GiB | Macro KLD | Macro Same-top % |
| --- | --- | ---: | ---: | ---: |
| FIT-12G | IQ3_M->IQ4_XS | 12.000 | 0.122739 | 90.324 |
| FIT-12.5G | Q3_K_M->Q3_K_L | 12.498 | 0.131559 | 90.102 |
| FIT-13G | Q3_K_M->Q3_K_L | 12.996 | 0.130385 | 90.074 |
| FIT-13.5G | Q3_K_L->IQ4_XS | 13.498 | 0.116350 | 90.771 |

FIT-12.5G is strictly WORSE than the 0.5 GiB cheaper FIT-12G, and 12.5G vs
13G is quality-flat. Diagnosis: the (Q3_K_M -> Q3_K_L) span pins all 353 bulk
matrices at q3_k and offers only the ~138 auxiliary q4_k -> q5_k transitions
as degrees of freedom, which barely move KL; q3_k as a bulk type is
inefficient per byte on this model (Q3_K_S 0.2148 @ 11.245G vs IQ3_XS
0.1512 @ 11.145G). This is a P4 span-selection error, recorded as such.

## Change (frozen before execution)

Redo the three tiers with K-series excluded from the candidate space. Per the
P4 span rule (largest preset <= target -> smallest preset > target) applied
to the K-free IQ ladder, all three tiers select the SAME pair:

| Tier | Pair (lower -> upper) | Target bytes |
| --- | --- | ---: |
| FIT-12.5G | IQ3_M -> IQ4_XS | 13,421,772,800 |
| FIT-13G | IQ3_M -> IQ4_XS | 13,958,643,712 |
| FIT-13.5G | IQ3_M -> IQ4_XS | 14,495,514,624 |

Endpoint recipes (P2 ladder): IQ3_M = iq3_s x409 + q4_k x88 + q6_k x1;
IQ4_XS = iq4_xs x433 + q5_k x64 + q6_k x1. The interpolated candidate space
therefore contains ZERO q2_k/q3_k tensors. The q4_k/q5_k/q6_k auxiliary
positions (token_embd/output/attn aux) are llama.cpp's own IQ-preset
assignments and remain - the owner-directed ban targets the K bulk
interpolation that caused the regression. P4 tier-selection scope: exactly
the three tiers named by the owner; 9.5G/10G keep their Q2_K-touching pairs.

Everything else stays frozen: balanced allocator (v0.1b), imatrix_unsloth.gguf,
pinned runtime tools/llama-b10666-rocm, oracle planning, zero-byte quantize
gate, M9 KL protocol (five fixed 64 KiB slices, BF16 references reused).
Analysis cache reuse: `2026-08-29-p4-release-batch/tiers/_analysis/
IQ3_M-IQ4_XS/analysis.json` (identical source/imatrix/runtime/pair; cached
per pair by P4 design). Baseline snapshot of the K-based results is taken
before any results file is overwritten.

## Gates (frozen before execution)

- K1 plan gate: 3 plans, policy balanced, predicted <= target, model_name
  Qwen3.8-27B-Uncensored, suggested filename label exactly 12.5G / 13G /
  13.5G, dominant qtype recorded and in {iq3_s, iq4_xs}.
- K2 quantize gate: 3 artifacts with size == predicted bytes (zero
  tolerance), SHA-256 recorded.
- K3 K-free gate: per-plan qtype distribution contains zero q2_k and q3_k
  tensors.
- K4 adoption gate: each new tier's macro KLD strictly below the K-based
  tier it replaces (0.131559 / 0.130385 / 0.116350). On pass, the three
  tiers replace the K-based ones in the release (P4 tiers.csv rows, tier
  plan dirs with old files archived under `k-based/`, bundle GGUFs and
  fit-plans swapped, tables/curves re-rendered, P4 gates re-run). On fail,
  record as failure and keep the K-based release.
- K5 curve-sanity gate: macro KLD non-increasing along
  FIT-12G -> FIT-12.5G -> FIT-13G -> FIT-13.5G. A violation is investigated
  and recorded before any release swap is finalized.

Execution order: plans -> quantizes -> evals -> gates -> (on K4 pass) release
swap -> re-summarize -> P4 gate re-run. Failure handling: any gate failure is
recorded as-is; fixes are code/script changes followed by a full re-run of
the affected stage.

## Amendment 1 (2026-08-30, run-1 gate verdict invalid, recorded before any re-run)

Run 1 completed plans and all three quantizes (K1/K2/K3 pass, zero-byte),
then STOPPED before the release swap with K4/K5 false - correctly per the
preregistered failure path. The verdict is however INVALID as a quality
measurement: the eval-log skip guard keyed on the tier label, which this
experiment reuses from the K-based run, so all 15 eval logs were stale
K-based measurements (the "new" macro KLDs printed by the gate evaluator
are bit-identical to the K-based baselines). The K4/K5 failure is therefore
recorded as a measurement-infrastructure failure, not a quality result.

Fix before any affected step (stage C) re-runs: the eval guard now keys on
the evaluated artifact's SHA-256 (sentinel file `eval-<tier>-<domain>.sha`
next to each log); stale logs are invalidated automatically. The invalid
verdict is preserved as `results/gate-verdict-run1-stale-logs.json`. The
affected stage (C: evals + gate evaluation) is re-run in full; plans and
quantizes are untouched (their artifacts are hash-verified on resume).
