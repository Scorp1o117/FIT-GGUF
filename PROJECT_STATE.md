# FIT-GGUF Project State

## Goal

Build and empirically validate a continuous-size GGUF quantization prototype
that uses one canonical imatrix to generate deterministic, size-safe recipes
between standard llama.cpp presets.

## Current milestone

M13 complete - the preregistered budget-conditional rule (r<0.5 original,
r>=0.5 v0.1b) was NOT accepted on the third holdout set: FIT-25 and FIT-50
directions reproduced and the composite rule beat both pure strategies, but
the FIT-75 direction flipped to a statistical tie. The preregistered failure
branch (role-matched early/late block swap ablation) is the only permitted
next step.

## Verified facts

- The repository began as an asset-only directory and was initialized as a Git
  repository on 2026-08-28.
- The local runtime is llama.cpp build 10666, commit `4e97ac86e`, built with GNU
  15.2.0 for Linux x86_64 and ROCm/HIP libraries.
- `llama-quantize` exposes `--dry-run`, `--tensor-type`, and
  `--tensor-type-file`.
- `--dry-run` uses the same preset, override, and shape-fallback path as real
  quantization and prints each resolved tensor qtype, but its final size total
  contains only unpadded tensor payload bytes.
- Manual tensor override patterns use ordered ECMAScript `regex_search` with
  first-match precedence; a match bypasses preset mixture logic before the
  final shape fallback.
- The runtime supports the required development presets IQ3_S, IQ3_M, and
  IQ4_XS.
- The source model is stored as 18 safetensors shards, totaling 55,563,006,216
  bytes (51.747 GiB).
- Seven reference-logit files total 42,300,194,332 bytes (39.395 GiB).
- The calibration dataset contains 299 Parquet files. Its nested repository is
  at commit `e87ed55dcba9d9c3a3e41539f3e728e981b1daa4` and has local hydrated-LFS and
  added-file changes; it must not be modified by FIT.
- The canonical imatrix SHA-256 is
  `0ee5b10bd0c2fa2127c6f4b43dbfe1efd71e383b63217af9dade1de36599f1c1`.
- The canonical imatrix is GGUF v3 with 992 tensors, 32-byte alignment, dataset
  `unsloth_calibration_dataset`, 1,251 chunks, and chunk size 8,192.
- The current Huihui weights are newer than the weights used by the preserved
  PRISM plans and reference artifacts. Cross-version results are not valid FIT
  baselines without explicit provenance checks.
- A matching-converter BF16 GGUF now exists locally with 851 tensors and
  SHA-256
  `8a033407c8f58d43102aade25b973cc6d2f2ce5c5cbf4dc75a2cdb60b9e33cbc`.

## Current implementation

- Repository layout and project records exist.
- Large local assets and the prebuilt runtime are excluded from Git.
- `docs/llama-integration.md` records the reviewed M1 source audit.
- `src/fit_gguf/dry_run.py` strictly parses build-10666 dry-run tensor
  assignments and payload summaries into immutable typed records.
- Parser validation rejects malformed candidates, duplicate names/ordinals,
  incomplete tensor sequences, inconsistent totals, conflicting summaries, and
  aggregate differences beyond display-rounding tolerance.
- The parser, size predictor, profiler, planner, candidate generator, optimizer,
  and llama.cpp recipe writer have 43 passing unit tests, including Qwen3.5
  attention, FFN, and SSM tensor names. They deliberately treat printed MiB values as rounded
  display measurements rather than exact file-size predictions.
- IQ3_S, IQ3_M, and IQ4_XS dry-run assignments were compared with three full
  real quantizations of the current 27B model. All three had 851/851 matching
  tensor names and zero destination-qtype mismatches.
- The M2 evidence, commands, hashes, sizes, qtype distributions, and timings
  are recorded in `experiments/2026-08-28-m2-effective-recipes/README.md`.
- `src/fit_gguf/gguf.py` reads GGUF v3 layout without tensor data and predicts
  exact single-file size from a source layout and effective qtype recipe.
- M3 matches IQ3_S, IQ3_M, IQ4_XS, and a targeted Q5_K tensor override with
  zero-byte size error and zero actual-qtype mismatches. Evidence is recorded
  in `experiments/2026-08-28-m3-exact-size/README.md`.
- `src/fit_gguf/imatrix.py` reproduces build-10666 imatrix normalization and
  emits deterministic schema-v1 tensor profiles.
- The canonical profile covers all 496 expected layer matrices with zero
  missing or extra names. Its diagnostics and provenance are recorded in
  `experiments/2026-08-28-m4-imatrix-profile/README.md`.
- Baseline selection uses exact byte size rather than preset-name ordering.
- Candidate generation rejects 24 IQ3_M-to-IQ4_XS tensor transitions that
  would reduce encoded precision below the lower recipe.
- A deterministic greedy 13 GiB plan selected 338 profiled upgrades and left
  2,556,960 bytes unused.
- The real 13 GiB FIT artifact is exactly 13,956,086,752 bytes, matches its
  prediction with zero-byte error, and has zero qtype mismatches. Planning and
  integration evidence is in `experiments/2026-08-28-m7-greedy/README.md`.
- Formal FIT-25, FIT-50, and FIT-75 artifacts were generated between IQ3_M and
  IQ4_XS. All three are under target, meet the M11 fine-fill bound already,
  match exact predicted size with zero-byte error, and have zero qtype
  mismatches across 851 tensors. Evidence is in
  `experiments/2026-08-28-m9-fit-curve/README.md`.
- Fresh current-weight BF16 reference logits and 25 controlled KL/PPL runs
  cover wiki_test, wiki_valid, Chinese, code, and agent_chat. Mean KL strictly
  decreases and Same-top strictly increases at every size step in all five
  domains. The deterministic machine-readable record is
  `experiments/2026-08-28-m9-fit-curve/curve-results.json`.
- `optimize_random()` provides a Python-hash-independent SHA-256 priority
  baseline. Three matched-budget FIT-50 random seeds were real-quantized and
  evaluated across all five domains. FIT beats the three-seed random mean on
  macro KL by only 2.36%; one random seed beats FIT on macro KL, and random is
  better on Chinese KL on average. Positive allocation evidence is therefore
  not accepted. Evidence is in
  `experiments/2026-08-28-m10-ablation/README.md`.
- `optimize_block_balanced()` implements the targeted M10 diagnosis: equal
  byte quotas across four 16-block ranges with unchanged within-range imatrix
  ranking. Its real FIT-50 artifact has zero size/qtype error and improves
  five-domain macro KL by 3.99% versus the original FIT-50, driven by 12.1%
  gains on Chinese and agent_chat while both wiki domains regress. Because the
  ablation was designed after inspecting these same domains, it is a promising
  v0.1b candidate rather than independent validation.
- M11 (untouched holdout validation, preregistered in
  `experiments/2026-08-28-m11-holdout/README.md` before execution): five fresh
  64 KiB slices from the same five domain sources at disjoint offsets, all
  SHA-256 recorded. All three deleted random GGUFs were rebuilt from retained
  tensor-type files and reproduced their M10 SHA-256 hashes exactly. On
  holdout, v0.1b macro KL 0.083634 beats original FIT-50 0.088518 (-5.52%) and
  the random-seed mean 0.087132, with 4-of-5 per-domain wins over the random
  mean and a worst-domain regression of +16.23% (wiki_test) inside the
  preregistered 25% guard. All three preregistered gates passed.
  Random v3 alone is statistically tied with v0.1b on macro KL (0.083698);
  the v0.1b advantage is its cross-domain profile, not a uniform margin.
- M12 (v0.1b curve extension): the frozen allocator was applied at the M9
  FIT-25 and FIT-75 positions with a driver first validated against the
  retained M10 ground truth (byte-identical plan). Both artifacts match their
  exact predicted sizes with zero-byte error. On the original M9 slices,
  v0.1b is worse at FIT-25 (macro +5.31%, loses 4 of 5 domains) and better at
  FIT-75 (macro -4.75%, wins Chinese/code/agent_chat). The allocation
  trade-off is therefore budget-dependent; the original utility still owns
  FIT-25.
- M13 (preregistered budget-rule validation, third holdout set): rule
  `r<0.5 -> original, r>=0.5 -> v0.1b` with gates frozen and committed before
  execution. FIT-25 direction reproduced (original -6.89%), FIT-50 direction
  reproduced (v0.1b -6.16%), and the composite 15-cell rule macro beat both
  pure strategies (0.084581 vs 0.086414 all-original and 0.086988 all-v0.1b),
  but the FIT-75 direction did NOT reproduce: original 0.069907 vs v0.1b
  0.070056 (+0.21%, statistical tie). Gate 1 failed, so the rule was NOT
  accepted; per the frozen failure branch the role-matched early/late block
  swap ablation is the only permitted next step, and the 0.50 threshold is
  untouchable.

## Experimental results

M2 produced three validated preset artifacts. Actual sizes were
12,419,328,992 bytes (IQ3_S), 12,580,875,232 bytes (IQ3_M), and
15,082,507,232 bytes (IQ4_XS). These validate recipe assignment only; no model
quality claim has been made.

M3 predicts all three preset sizes plus one custom mixed recipe with zero-byte
error. This validates size accounting, not generation quality.

M4 produced a 496-entry canonical tensor profile. It confirms strong role and
block variation but makes no causal quality claim.

M8 produced the first arbitrary-size FIT GGUF at a 13 GiB target. It uses
99.9817% of the budget and matches the exact predictor, but quality remains
unevaluated.

The formal M9 size curve is IQ3_M / FIT-25 / FIT-50 / FIT-75 / IQ4_XS. Its
three intermediate artifacts have exact sizes 13,203,487,712, 13,831,486,432,
and 14,456,126,432 bytes. Across five domains, macro mean KL is 0.141098,
0.106277, 0.097324, 0.079720, and 0.057874 respectively. Macro Same-top is
89.456%, 91.175%, 91.910%, 93.121%, and 93.865%. PPL point estimates are not
strictly monotonic in Chinese and code, but the fluctuations overlap their
reported uncertainty while KL and Same-top retain consistent direction.

## Known issues

- A clean detached llama.cpp source checkout is available at
  `third_party/llama.cpp`, commit
  `4e97ac86ebe2c4cb8212d98d2641ad6768810896`, matching the runtime.
- The development GGUF intentionally excludes the auxiliary NextN/MTP head;
  this scope must remain explicit when comparing against external artifacts.
- Existing reference logits may correspond to older weights and require
  provenance classification before reuse.

## Decisions

- Use baseline-anchored promotion and never intentionally downgrade below the
  effective lower preset.
- Preserve local test assets while excluding them from the project repository.
- Treat the supplied build-10666 ROCm bundle as the initial execution runtime.
- Use Anti-Gravity CLI with Gemini 3.7 Flash High for the initial read-only M1
  investigation; all claims will be independently checked against source.

## Rejected approaches

- Reusing the unrelated dirty llama.cpp checkout found elsewhere on disk: it
  contains thousands of working-tree changes and cannot provide clean
  provenance.
- Deleting prior plans, logs, reference logits, or calibration Git metadata as
  cosmetic cleanup: these may be needed for provenance and controlled
  comparisons.

## Current task

Record the M13 failure and prepare the preregistered failure branch: the
role-matched early/late block swap ablation for attribution. The
budget-conditional rule is not accepted; the 0.50 threshold and all recipes
stay frozen.

## Next task

Design, preregister, and run the role-matched early/late block swap ablation
(the only permitted next step after the M13 gate failure). Only after that
attribution settles may the project revisit allocator design or move to M14/M15
(more random seeds; second model family).

## Acceptance status

Not accepted. M0-M12 complete; M13 preregistered rule validation FAILED gate 1
(FIT-75 direction). Positive allocation evidence now stands as: v0.1b
confirmed at FIT-50 on untouched data (M11); the composite budget rule beats
both pure strategies on fresh data but is not accepted because the FIT-75
direction is unstable. Attribution work (role-matched swap) is required
before any deployment claim.
