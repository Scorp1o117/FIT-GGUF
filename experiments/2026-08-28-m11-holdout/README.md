# M11 Untouched Holdout Validation of Block-Balanced v0.1b

Date: 2026-08-28
Status: PREREGISTERED — this gate was written before any holdout slice was
created, before any holdout reference logits were generated, and before any
holdout evaluation was run.

## Purpose

The block-balanced v0.1b allocator (D-0017) was designed after inspecting the
five M9/M10 evaluation domains, so its 3.99% macro-KL win is adaptive, not
confirmed. This milestone tests the frozen v0.1b recipe on fresh, untouched
holdout slices from the same five domains.

## Frozen inputs (no tuning allowed)

- v0.1b recipe: `experiments/2026-08-28-m10-ablation/block-balanced-fit50-tensor-types.txt`
  and `block-balanced-fit50-recipe.json` (frozen; no quarter-weight changes).
- Original FIT-50: M9 artifact, SHA-256 `e4fe1c46ab89c8b6343203168ebeec699372c2fb21f411c61c47edc2e1f33306`.
- Random v1/v2/v3: rebuilt from the retained M10 tensor-type files; rebuilt
  artifacts must reproduce the M10 SHA-256 values
  (`dba2ec21…`, `2fd1a30b…`, `c0a119b8…`). A hash mismatch is a reproducibility
  failure and blocks interpretation until resolved.
- BF16 source: `artifacts/source/Huihui-Qwen3.8-27B-abliterated-BF16.gguf`,
  SHA-256 `8a033407c8f58d43102aade25b973cc6d2f2ce5c5cbf4dc75a2cdb60b9e33cbc`.
- Evaluation protocol: `llama-perplexity -ngl 99 -t 16 -c 512 -b 512`, KL and
  Same-top against fresh BF16 reference logits on each holdout slice.

## Holdout slices

Five new 64 KiB slices (65,536 characters, UTF-8 encoded), each cut from the
same source file as its M9/M10 counterpart at a documented, disjoint offset:

| Domain | Source | Original offset | Holdout offset |
| --- | --- | ---: | ---: |
| wiki_test | `eval-data/wikitext-2-raw/wiki.test.raw` | 0 | 655,360 |
| wiki_valid | `eval-data/wikitext-2-raw/wiki.valid.raw` | 0 | 655,360 |
| Chinese | `imatrix-calibration/combined_cn_medium.parquet` | 3,173,914 | 4,000,000 |
| code | `imatrix-calibration/code_medium.parquet` | 12,536,883 | 18,000,000 |
| agent_chat | `imatrix-calibration/agentworld_clean_quick.txt` | 998,087 | 400,000 |

Holdout offsets were chosen before evaluation to (a) not overlap the original
slice ranges, and (b) match the original slice's domain density as closely as
possible (Chinese: CJK fraction 0.027 in both regions; code: heuristic code
density ~4.2 in both; agent_chat: ~2.4 vs 2.73). Slice construction, SHA-256
values, and non-overlap checks are recorded by `scripts/make_holdout_slices.py`
in `holdout-slices.json`.

## Preregistered decision gate

Primary metric: five-domain macro mean KL on the holdout slices (lower is
better), all five variants evaluated under the identical protocol.

- **Gate A (confirmatory win):** v0.1b macro KL < original FIT-50 macro KL.
- **Gate B (beats random):** v0.1b macro KL < the three-seed random mean macro
  KL, and v0.1b KL is below the per-domain random mean in at least 4 of 5
  domains.
- **Guard C (no catastrophic domain regression vs original FIT-50):** in no
  domain does v0.1b KL exceed original FIT-50 KL by more than 25%. The 25%
  bound was fixed now, before evaluation, based on the already-observed adaptive
  trade-off (+20.0% wiki_test, +15.8% wiki_valid on the design domains); it is
  intentionally not relaxed after results.

Decision rule:

- A ∧ B ∧ C → v0.1b is confirmed at FIT-50. Extend: generate and evaluate
  block-balanced FIT-25 and FIT-75 against the M9 curve points (matched-size
  protocol). Only then consider quarter-weight tuning or optimizer v2.
- Otherwise → perform the role-matched early/late block swap ablation before
  any allocator change. Optimizer v2 remains blocked either way until this
  attribution settles.

Secondary diagnostics (reported, not gated): per-domain KL table, Same-top,
PPL, and uncertainty ranges.

## Execution notes

- Reference generation: 5 BF16 runs on the holdout slices.
- Rebuild 3 random GGUFs from retained tensor-type files; verify hashes.
- Evaluation: 5 variants × 5 domains = 25 runs, parsed by the deterministic
  M9 summarizer protocol into `holdout-results.json`.
- Raw logs and KLD files are preserved locally under `artifacts/` and excluded
  from Git.

## Provenance verification (executed)

- BF16 source SHA-256 re-verified before reference generation: `8a033407…` OK.
- All three random GGUFs were rebuilt from their retained tensor-type files and
  reproduced the M10 SHA-256 values exactly (`dba2ec21…`, `2fd1a30b…`,
  `c0a119b8…`), confirming quantization determinism of the pinned runtime.
- Retained original FIT-50 (`e4fe1c46…`) and block-balanced v0.1b (`7cfa1b91…`)
  artifacts re-verified OK.
- Holdout BF16 reference PPL (fresh slices, not comparable to M9 values):
  wiki_test 6.9127 ± 0.19683, wiki_valid 8.4965 ± 0.25445,
  Chinese 8.3366 ± 0.29977, code 4.3074 ± 0.12175,
  agent_chat 6.9905 ± 0.23109.
- Timing note: M11 logs were not instrumented; `elapsed_seconds` in
  `holdout-results.json` is wall-clock from log file birth→mtime and
  `max_rss_kb` is null.

## Results

Mean KL divergence on the five untouched holdout slices (lower is better):

| Domain | FIT-50 | v0.1b | random v1 | random v2 | random v3 | random mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| wiki_test | 0.031732 | 0.036882 | 0.036279 | 0.032757 | 0.033457 | 0.034164 |
| wiki_valid | 0.033148 | 0.035963 | 0.040468 | 0.033870 | 0.035652 | 0.036663 |
| Chinese | 0.156337 | 0.142490 | 0.155902 | 0.163236 | 0.139773 | 0.152970 |
| code | 0.111544 | 0.102015 | 0.102071 | 0.104958 | 0.103460 | 0.103496 |
| agent_chat | 0.109829 | 0.100818 | 0.109252 | 0.109701 | 0.106147 | 0.108367 |
| Macro mean | 0.088518 | 0.083634 | 0.088794 | 0.088904 | 0.083698 | 0.087132 |

Same-top percentage:

| Domain | FIT-50 | v0.1b | random v1 | random v2 | random v3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| wiki_test | 92.367 | 91.877 | 91.933 | 92.143 | 92.297 |
| wiki_valid | 91.541 | 90.980 | 90.798 | 91.625 | 91.835 |
| Chinese | 91.395 | 90.924 | 90.857 | 90.846 | 91.137 |
| code | 92.636 | 92.603 | 91.950 | 92.277 | 92.407 |
| agent_chat | 91.337 | 91.634 | 90.778 | 90.481 | 91.206 |

## Preregistered gate verdict

- **Gate A (confirmatory win): PASS.** v0.1b macro KL 0.083634 vs original
  FIT-50 0.088518, a 5.52% improvement on untouched data.
- **Gate B (beats random): PASS.** v0.1b macro KL beats the random-seed mean
  (0.087132) and is below the per-domain random mean in 4 of 5 domains
  (wiki_test is the exception at +7.95%).
- **Gate C (no catastrophic regression): PASS.** The worst domain regression
  versus original FIT-50 is wiki_test at +16.23%, inside the preregistered 25%
  bound; wiki_valid regresses +8.49% while Chinese, code, and agent_chat
  improve 8.86%, 8.54%, and 8.20%.

**Decision: CONFIRMED — extend v0.1b to FIT-25 and FIT-75** (M12), per the
preregistered rule.

## Honest qualifications

- The macro win over random is aggregate-driven: rebuilt random v3 alone is
  statistically tied with v0.1b on macro KL (0.083698 vs 0.083634, a 0.08%
  v0.1b edge). v0.1b's advantage is its cross-domain profile — it beats the
  random mean in 4 of 5 domains and the original FIT-50 macro by 5.52% — not a
  large uniform margin over every seed.
- The wiki trade-off reproduces qualitatively on untouched data (v0.1b gives up
  wiki KL for Chinese/code/agent_chat gains), so the trade-off is a property of
  the allocation strategy, not an artifact of the design domains.
- Same-top differences are small (< 0.6 pp everywhere) and do not contradict
  the KL ordering.

Machine records: `holdout-slices.json` (slice provenance and SHA-256),
`holdout-results.json` (parsed metrics), `gate-verdict.json` (gate evaluation).
