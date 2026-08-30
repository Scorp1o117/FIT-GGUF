# P6: IQ2 Span Fix (8G / 8.5G) + Q3_K-Free 9.5G / 10G (Owner Directive)

Date: 2026-08-30. Status: preregistered before any execution step of this
experiment.

## Trigger

Owner flagged FIT-8G as fully dominated by a native preset (8.000G, macro
KLD 0.5664, Same-top 75.53 vs IQ2_XXS 7.854G, 0.5403, 75.56) and approved
testing the Q3_K-free route for FIT-9.5G / FIT-10G.

## Diagnosis (from P4 reference measurements)

Per-byte KL deltas between adjacent IQ2 presets on this model:

| Preset delta | Bytes | dKLD | dKLD/GiB |
| --- | ---: | ---: | ---: |
| IQ2_XXS -> IQ2_XS | +0.612G | +0.0635 | +0.1037 (TOXIC) |
| IQ2_XS -> IQ2_S | +0.254G | -0.0334 | -0.1318 |
| IQ2_S -> IQ2_M | +0.598G | -0.1094 | -0.1831 (golden) |

iq2_xs as a tensor type is the poison: the IQ2_XS preset is itself an
outlier above the frontier (same phenomenon class as Q3_K_S). FIT-8G's span
(IQ2_XXS -> IQ2_XS) fills through xxs->xs upgrades, hence the regression;
FIT-8.5G (IQ2_XS -> IQ2_S) is dominated by IQ2_XXS as well (0.5752 @ 8.499G)
and would become the new non-monotonic point if 8G alone moved.

## Change (frozen before execution)

All four tiers move to native-preset endpoints that route fill through
measured-good upgrade paths; candidate override sets contain zero q3_k:

| Tier | Old pair | New pair | Target bytes |
| --- | --- | --- | ---: |
| FIT-8G | IQ2_XXS->IQ2_XS | IQ2_XXS->IQ2_M | 8,589,934,592 |
| FIT-8.5G | IQ2_XS->IQ2_S | IQ2_XXS->IQ2_M | 9,126,805,504 |
| FIT-9.5G | IQ2_M->Q2_K | IQ2_M->Q2_K_S | 10,200,547,328 |
| FIT-10G | Q2_K->IQ3_XXS | Q2_K_S->IQ3_XXS | 10,737,418,240 |

FIT-8G fill ~10% of span, FIT-8.5G ~41%, FIT-9.5G ~81%, FIT-10G ~52%.
Q2_K_S (10,248,326,336) and IQ2_M (10,004,593,856) bracket both 9.5G and
10G targets correctly. Untouched: FIT-7.5G (on frontier), FIT-9G
(IQ2_S->IQ2_M already the golden path, 0.4589 beats IQ2_M), all other
tiers. Everything else frozen: balanced allocator, imatrix_unsloth.gguf,
pinned runtime, oracle planning, zero-byte gate, M9 KL protocol, BF16
references reused. New analyses required for the three new pairs
(IQ2_XXS-IQ2_M, IQ2_M-Q2_K_S, Q2_K_S-IQ3_XXS).

Infra note (from P5 amendment 1): eval logs are keyed by tier label which
this experiment reuses; the eval guard keys on the artifact SHA-256
(sentinel per log). Additionally, stale quantize records from the original
P4 run live in `artifacts/fit/release/` (checked before `_record_path`
lookups) - the pre-move step retires them alongside the old bundle
artifacts before any new quantize, so record lookups cannot hit stale
files even when a new artifact keeps an old dominant qtype in its name.

## Gates (frozen before execution)

- G1 plan gate: 4 plans, policy balanced, predicted <= target, model_name
  Qwen3.8-27B-Uncensored, suggested filename label exactly 8G / 8.5G /
  9.5G / 10G, dominant qtype recorded.
- G2 quantize gate: 4 artifacts with size == predicted bytes (zero
  tolerance), SHA-256 recorded.
- G3 K-free gate: zero q3_k tensors in every override set (q2_k allowed
  where the native Q2_K_S endpoint assigns it).
- G4 adoption gate: each new tier's macro KLD strictly below the tier it
  replaces (0.566418 / 0.575169 / 0.309781 / 0.239082). On pass the four
  tiers replace the old ones in the release (P4 tiers.csv, tier plan dirs
  archived under `prev-span/`, bundle GGUFs/fit-plans swapped, tables and
  curves re-rendered, P4 gates re-run). On fail: record, keep old release.
- G5 curve gate: macro KLD non-increasing across all 14 FIT tiers ordered
  by target size.
- G6 anti-domination gate: FIT-8G and FIT-8.5G macro KLD strictly below
  the IQ2_XXS reference (0.540252) - the owner-flagged condition.

Execution order: analyses -> plans -> retire old artifacts/records ->
quantizes -> evals -> gates -> (on pass) release swap -> re-summarize ->
P4 gate re-run. Failure handling: any gate failure is recorded as-is; the
retired artifacts stay recoverable under `artifacts/fit/release/
retired-p6/` until the swap succeeds.

## Amendment 2 (2026-08-30, run-1 G5 failure root-caused; FIT-9G joins the redo)

Run 1: all four tiers quantized zero-byte; G1-G4 and G6 pass with large
improvements (8G 0.5664->0.4838, 8.5G 0.5752->0.4527, 9.5G 0.3098->0.2737,
10G 0.2391->0.2299); G5 (whole-curve monotonicity) FAILS: new FIT-8.5G
(0.4527) beats FIT-9G (0.4589, unchanged). Correctly stopped before the
swap; verdict recorded.

Root cause: the run-1 measurements expose the IQ2_XXS->IQ2_M fill
trajectory as non-monotone in fill fraction - 0% 0.5403, 10% 0.4838, 44%
0.4527, 100% (the IQ2_M preset itself) 0.4609. FIT-9G sits in the IQ2
dead zone between the 8.5G mix and the Q2_K_S step: no IQ2-family recipe
at 9.0G can beat the 8.5G mix (every native preset in 8.7-9.3G is
dominated by it). Leaving 9G unchanged reintroduces exactly the
size-up-quality-down defect this program is eliminating.

Change: FIT-9G moves to the span IQ2_XXS -> Q2_K_S (target 9,663,676,416;
fill ~68%, bulk upgrades iq2_xxs -> q2_k, zero q3_k by endpoint recipes -
Q2_K_S contains none). Consistent with run-1's 9.5G result (0.2737 beats
the Q2_K_S preset itself) showing q2_k-directed fill is the strong path
in this region. Everything else from run 1 is kept (artifacts hash-pinned;
eval sentinels make re-evaluation a no-op for them).

Gates: G4 for FIT-9G = strictly below 0.458876 (its replaced value); G5
now requires monotonicity with the new 9G value; G3 unchanged (no q3_k in
any override set). Runner hardened: the retire step skips tiers whose P6
artifact is already hash-recorded, so a resume never retires completed
artifacts.
