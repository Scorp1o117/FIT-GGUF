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
