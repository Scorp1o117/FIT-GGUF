# P4: Release Batch (0.5-GiB Tier Grid, Balanced Allocator, KL Curves)

Date: 2026-08-29

## Scope

Owner-directed first-release batch for `orcarouter/Qwen3.8-27B-Uncensored`:
every 0.5 GiB from 7G to 12G (11 tiers), allocator balanced (v0.1b, owner
decision), each tier KL-evaluated alongside the surrounding llama.cpp default
presets as reference points. Above-12G tiers are explicitly deferred (owner:
"12G 以上的这次就不做了").

This is release documentation, not a new research claim: the KL protocol and
slices are the established M9 set (five 64 KiB domains), reused unchanged.
Quality results are recorded as observations for the model card; the IQ1/IQ2
region is expected to be rough and is reported as measured.

## Frozen tier definitions (exact byte targets, G = GiB)

| Tier | Pair (lower -> upper) | Target bytes |
| --- | --- | ---: |
| FIT-7G | IQ1_S -> IQ1_M | 7,516,192,768 |
| FIT-7.5G | IQ1_M -> IQ2_XXS | 8,053,063,680 |
| FIT-8G | IQ2_XXS -> IQ2_XS | 8,589,934,592 |
| FIT-8.5G | IQ2_XS -> IQ2_S | 9,126,805,504 |
| FIT-9G | IQ2_S -> IQ2_M | 9,663,676,416 |
| FIT-9.5G | IQ2_M -> Q2_K | 10,200,547,328 |
| FIT-10G | Q2_K -> IQ3_XXS | 10,737,418,240 |
| FIT-10.5G | IQ3_XXS -> IQ3_XS | 11,274,289,152 |
| FIT-11G | IQ3_XXS -> IQ3_XS | 11,811,160,064 |
| FIT-11.5G | IQ3_XS -> IQ3_S | 12,348,030,976 |
| FIT-12G | IQ3_M -> IQ4_XS | 12,884,901,888 |

All pairs passed the preregistered health check (target bracketed, candidate
budget >= fill need; recorded in the P4 proposal). Imatrix:
`imatrix_unsloth.gguf` (P2 amendment 1). Every plan uses
`--policy balanced --target-bytes <exact>`; analyses are cached per pair.

Reference presets for the KL comparison: IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS,
IQ2_S, IQ2_M, Q2_K_S, Q2_K, IQ3_XXS, IQ3_XS, Q3_K_S, IQ3_S, IQ3_M, IQ4_XS
(14 artifacts; IQ4_XS is the upper context anchor for FIT-12G and is not a
release candidate this round).

## Gates (frozen before execution)

- R1 plan gate: 11 plans with the balanced allocator; prediction <= target;
  suggested filename renders the tier label exactly (7G, 7.5G, ... 12G per
  P3 amendment 1); dominant qtype and shares recorded per tier.
- R2 quantize gate: 11 FIT artifacts with size == prediction, zero tolerance;
  SHA-256 recorded per artifact.
- R3 reference gate: 14 preset artifacts with actual size == the E3 ladder
  predicted size (`preset-ladder.json`), zero tolerance.
- R4 KL gate: five BF16 reference KLDs generated from the converted BF16
  source on the fixed M9 slices
  (`/run/media/s117/OS/Models/eval-data/kl-eval-*.txt`, SHA-256 recorded);
  then 25 artifacts x 5 domains with the pinned protocol
  (`llama-perplexity -ngl 99 -t 16 -c 512 -b 512 --kl-divergence
  --kl-divergence-base`); every run parsed successfully; per-artifact
  per-domain mean KL, Same-top, and PPL recorded in `results/p4-results.json`.
- R5 reporting gate: comparison table (per tier: pair, target, actual bytes,
  dominant qtype, suggested filename, per-domain and macro KL, macro Same-top)
  and a line chart (x = artifact size GiB, y = macro mean KL; FIT tiers
  connected as a line, preset references as points), rendered to
  `results/kl-curve.png` and written as `results/comparison-table.md`.
- R6 suite gate: full test suite passes.

Failure handling: any gate failure is recorded as-is; fixes are code or
script changes followed by a re-run of the affected stage. Stage scripts
skip already-completed work (hash-checked) so a failure resumes cleanly.

## Disk and provenance

FIT release artifacts (~107 GB) are retained under `artifacts/fit/release/`
for upload; the 14 reference artifacts (~147 GB) are deleted after R4
(reproducible in minutes from presets). Disk was 313 GB free at batch start.

## Amendment 1 (2026-08-29, R2 failure on FIT-11.5G, root-caused before any re-run)

FIT-11.5G failed its zero-byte gate: actual 12,277,279,936 vs predicted
12,344,126,656 (-66,846,720 bytes = 16 x 4,177,920, exact). All 110 planned
overrides WERE honored in the artifact; the divergence is in 16 non-override
tensors: blk.56-63 ffn_gate/ffn_up, planned iq3_s, actual iq3_xxs.

Root cause (confirmed in pinned src/llama-quant.cpp): for the IQ3_XS ftype,
`llama_tensor_get_type_impl` forces ffn_gate/ffn_up tensors whose
counter-based layer index lands in [n_layer/8, 7*n_layer/8) down to IQ3_XXS,
and the counters `i_ffn_gate/i_ffn_up` are only incremented inside that
function - which is SKIPPED for tensors matched by a `--tensor-type-file`
override (`if (!manual ...)`). The analyze-time dry-run runs WITHOUT
overrides, so its captured base recipe cannot see this effect: with 40
ffn_gate and 38 ffn_up overrides all inside blocks 8-55, the non-overridden
blocks 56-63 receive counter indices 16-23, inside the forced window. The
same skip applies to the counter-based ffn_down first-eighth rules of the
IQ1/IQ2 ftypes. Pairs whose override sets touch no counter-ruled category
(or only touch categories outside the windows) are unaffected - which is why
the other ten tiers and every historical artifact (whose upgrades were
ffn_down-only with no overlapping window) matched exactly.

Fix (llama.cpp-oracle planning): after writing the tensor-type file, `fit
plan` re-runs the lower-preset dry-run WITH that file and adopts the parsed
effective recipe as the prediction oracle. If the effective prediction
exceeds the target, the planner re-selects with a reduced target
(overshoot + 1 MiB margin, at most 3 oracle iterations). The planned
override set semantics are unchanged; only the prediction basis becomes the
oracle. Slack left by counter shifts is reported as unused bytes.

FIT-11.5G under the fix: effective prediction 12,277,279,936 (<= target
12,348,030,976), slack 70,751,040 bytes (0.55% of target) - accepted and
documented; the tier is re-quantized against the oracle prediction. The ten
already-quantized tiers keep their artifacts; their plans are re-run through
the oracle (expected no-op, since their actual outputs already matched).

## Honest scope notes

- Size control per artifact is self-proving (R2/R3).
- Quality numbers are protocol-scoped observations (512-token context,
  KL/Same-top vs aligned BF16, five fixed slices), intended for the model
  card and the comparison chart - not new allocator or generalization claims.
- Allocator research stays frozen (D-0022/D-0023): balanced is used as the
  documented release choice, with the M11/M16 caveats cited in the model card.

## Results (2026-08-29)

All six gates passed (`gate-verdict.json`): 11 balanced plans with exact tier
names, 11 FIT artifacts and 14 reference presets at zero-byte size error,
25 x 5-domain KL evaluations parsed, tables and curves rendered, 61 tests
green. See `results/comparison-table.md`, `results/kl-curve.png`,
`results/sametop-curve.png`, `results/p4-results.json`.

Key observations (protocol-scoped, M9 slices):

- The FIT tier line threads the preset reference points smoothly across the
  whole 7-12 GiB range; every tier's macro KL sits at or between its
  bracketing presets.
- Q3_K_S is a visible outlier ABOVE the frontier (macro KL 0.2148 at
  11.24 GiB - worse than the cheaper IQ3_XS at 11.15, macro 0.1512); FIT-11G
  and FIT-11.5G dominate it outright.
- FIT-12G (12.0 GiB, macro KL 0.1227, Same-top 90.32%) beats both IQ3_S
  (11.57 GiB, 0.1424) and IQ3_M (11.72 GiB, 0.1445) at less than +0.3 GiB
  over IQ3_M - the mixed-quant allocation advantage, on this architecture
  where the allocator evidence directly applies.
- The IQ1 region is as rough as expected (macro KL ~1.13) and is reported
  as measured; the curve turns sharply at the IQ2 boundary (7.5G).

The 14 reference artifacts were deleted after evaluation per the disk policy
(reproducible in minutes); the 11 FIT release artifacts are retained under
`artifacts/fit/release/` for upload.

## Amendment 2 (2026-08-29, owner-directed release naming)

Release filenames drop the source-org prefix: the released artifacts are
`Qwen3.8-27B-Uncensored-FIT-<size>-<qtype>.gguf` (plans re-run with
`--model-name Qwen3.8-27B-Uncensored`; predictions and artifacts unchanged,
11/11 size-verified after rename). The complete release bundle is assembled
at `/Qwen3.8-27B-Uncensored-FIT-GGUF/` (gitignored staging folder): the 11
GGUFs, `results/` (comparison table, KL and Same-top curves, machine
records), `fit-plans/` (per-tier plan/recipe/tensor-type provenance), and a
README draft. The GGUFs live in the bundle; the quantization workspace
under `artifacts/fit/release/` keeps the quantize records.

## Amendment 3 (2026-08-29, owner-directed grid extension)

Three tiers added to fill the 12-14 GiB gap: FIT-12.5G and FIT-13G on the
(Q3_K_M -> Q3_K_L) pair (the immediate bracket; 138 positive-gain candidates,
budget 0.97 GiB >= both fill needs - the Q3_K_L attention-piece transitions
avoid the zero-gain trap), and FIT-13.5G on (Q3_K_L -> IQ4_XS, exact target 14,495,514,624 = 13.5 GiB; an initial off-by-1.1 MB target of 14,496,634,368 rendered "14G" and was corrected before release). Same frozen
settings as the original eleven: balanced allocator, oracle planning,
zero-byte quantize gate, 5-domain KL evaluation.

Final release state after this amendment: 14 FIT artifacts and 14 reference
artifacts are present in `results/p4-results.json`; all 28 have five-domain
measurements, and all 14 released FIT artifacts match their final post-oracle
predicted byte sizes exactly. This final count supersedes the original
11-FIT/25-evaluation counts in the pre-extension Results section above while
preserving that section as the contemporaneous record of the first run.

## Amendment 4 (2026-08-30, owner-directed K-free redo of FIT-12.5G/13G/13.5G)

The owner spotted that the 12.5-13G tiers bought no (indeed negative)
quality: macro KL 0.122739 (FIT-12G) -> 0.131559 (12.5G) -> 0.130385 (13G).
Diagnosis: the amendment-3 pair (Q3_K_M -> Q3_K_L) pins all 353 bulk matrices
at q3_k and offers only ~138 auxiliary q4_k -> q5_k transitions, which barely
move KL - while q3_k bulk is per-byte inefficient on this model (Q3_K_S
0.2148 @ 11.245G vs IQ3_XS 0.1512 @ 11.145G). A P4 span-selection error,
recorded as such.

Per the owner directive ("Q4KM以下把K系列移除试试，用IQ3-IQ4XS重做一次
12.5 / 13 / 13.5"), the three tiers.csv rows move to the K-free IQ ladder:
all three select (IQ3_M -> IQ4_XS), whose candidate space contains zero
q2_k/q3_k tensors. Everything else frozen (balanced, imatrix_unsloth, oracle
planning, zero-byte gate, M9 KL protocol). Execution and gates K1-K5 are
preregistered in `experiments/2026-08-30-p5-kfree-12-13.5/`; old plan files
are archived under `tiers/<tier>/k-based/`.

New results (all gates pass, release swapped, this file's Results counts
superseded for these three tiers):

| Tier | Pair | Actual GiB | Macro KLD | Macro Same-top % |
| --- | --- | ---: | ---: | ---: |
| FIT-12.5G | IQ3_M -> IQ4_XS | 12.497 | 0.111572 | 91.045 |
| FIT-13G | IQ3_M -> IQ4_XS | 12.998 | 0.098702 | 91.931 |
| FIT-13.5G | IQ3_M -> IQ4_XS | 13.498 | 0.083768 | 92.737 |

Improvements over the K-based tiers: -0.0200 / -0.0317 / -0.0326 macro KL;
the FIT curve is now monotonically improving from FIT-12G through FIT-13.5G
(0.1227 -> 0.1116 -> 0.0987 -> 0.0838), removing the pair-boundary
regression. P4 gates R1-R6 re-run ALL_PASS after the swap.

## Amendment 5 (2026-08-30, owner-directed P6: IQ2 span fix + Q3_K-free 9.5G/10G)

The owner flagged FIT-8G as fully dominated by the IQ2_XXS preset (8.000G
macro KL 0.5664 vs 0.5403 at 7.854G). Reference deltas identified iq2_xs as
a toxic tensor type on this model (IQ2_XXS->IQ2_XS: +0.1037 KLD/GiB; the
IQ2_XS preset is an outlier like Q3_K_S), while IQ2_S->IQ2_M is the golden
path (-0.1831/G). FIT-8.5G was dominated as well. Per preregistered
experiment `experiments/2026-08-30-p6-iq2-span-fix/` (amendments 1-4
documented there: 9G joined after run-1 G5 monotonicity failure; oracle
loop convergence rule fixed), five tiers moved to new native-preset spans:

| Tier | New pair | Actual GiB | Macro KLD | Macro Same-top % |
| --- | --- | ---: | ---: | ---: |
| FIT-8G | IQ2_XXS -> IQ2_M | 7.999 | 0.483774 | 76.897 |
| FIT-8.5G | IQ2_XXS -> IQ2_M | 8.499 | 0.452663 | 78.416 |
| FIT-9G | IQ2_XXS -> Q2_K_S | 8.999 | 0.336343 | 80.824 |
| FIT-9.5G | IQ2_M -> Q2_K_S | 9.500 | 0.273662 | 83.059 |
| FIT-10G | Q2_K_S -> IQ3_XXS | 9.999 | 0.229923 | 84.598 |

Improvements over the replaced tiers: -0.083 / -0.123 / -0.123 / -0.036 /
-0.009 macro KL. The full 14-tier FIT curve is now strictly monotone in
measured quality, and in the 8-10 GiB region every FIT tier beats its
surrounding native presets (8G/8.5G beat IQ2_XXS, 9.5G beats Q2_K_S, 10G
beats Q2_K). All override sets are q3_k-free. Old plan files archived
under `tiers/<tier>/prev-span/`. Gates G1-G6 pass; P4 gates R1-R6 re-run
ALL_PASS after the swap.
