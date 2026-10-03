# Changelog

All notable changes to FIT-GGUF. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions use
[Semantic Versioning](https://semver.org/).

## [0.5.0] — 2026-10-04

### Added

- **FIT Studio**, a local browser or Windows desktop application with two
  primary entries: specify file size, or specify a quality tier. The size
  workflow exposes analyze, plan and quantize one step at a time, including
  precision distribution, actual-byte checks, hashes and persisted history.
- **Quality tier search in the UI** calls the real `fit fidelity-search`
  product flow with source-bound frozen five-domain references, a bounded
  search budget and final artifact re-evaluation. No-pass reports remain
  visible without being presented as successful deliveries.
- Optional **llmfit** integration through its executable JSON API, live free
  NVIDIA VRAM and available RAM budgets. Single-card GPU budgets do not add
  system RAM or sum independent GPUs. Default source experiments are limited
  to 5B tensor parameters; recommendations filter total MoE parameters.
- Windows application builder with a bundled Python worker, llmfit and license
  notices; real source/frozen worker smoke tools; Windows/Linux test matrix.

### Fixed

- Small-model quantizer fallback to Q4_0/Q4_1 is now supported by the exact
  GGUF size predictor, with traits checked against upstream b10666 headers.
- CLI/package/persisted-record versions now share one version source. The
  previous CLI still reported 0.3.3 while package metadata was 0.4.0.
- Absolute imatrix paths survive independent worker directories; task history
  sorts by creation time and persists interrupted status across reopening.
- Failed llmfit detection is reported as a local fallback. A file budget below
  the analysis interval is not silently increased.
- UTF-8 quality logs are read consistently under legacy Windows locales;
  runtime paths use the target platform's separators.
- Desktop quantization/evaluation subprocesses stay hidden and isolate their
  DLL search from the frozen Python runtime. Starting a job clears stale
  results; both workflows offer a copyable artifact path.

## [0.4.0] — 2026-09-24

### Added

- **`fit tier-search` — solve each tier for the smallest artifact that reaches
  its KL anchor, instead of shipping the preset the window happened to contain.**

  `fit calibrate` fills each tier's window by probing the **largest uncovered KL
  gap** inside it (`_largest_uncovered_gap`). That serves the *floor derivation* —
  it wants enough samples in the window to take a stable P5 — and it is blind to
  what the tier is actually for. A tier's product is the smallest artifact that
  reaches the anchor, and that artifact generally lies **between** two ladder
  presets.

  On `occamy-1.0-abliterated` the quality window held `IQ4_XS` (17.44 GiB, KL
  0.0503 — a FAIL) and `Q4_K_M` (19.71 GiB, KL 0.0471). The answer lives in the KL
  gap `0.0471..0.0503`; the largest gap was `0.0503..0.0575`, so the single probe
  went **above** `IQ4_XS` and produced a *smaller* artifact with a *worse* KL
  (17.27 GiB @ 0.0555), failing the anchor outright. The tier then fell back to
  the `Q4_K_M` preset — 1.71 GiB larger than the minimum PASS on the same curve.
  Across the four tiers the gap was **6.2 GiB (9.8%)**.

  Two entry points:

  * `fit calibrate --tier-search` — finishing stage. It runs after the floors are
    frozen and **before** `stage_emit`, and that ordering is load-bearing: the
    search appends its probes to `curve-points.jsonl`, whose digest `stage_emit`
    records in `calibration-record.json`. A post-emit run would leave a stale
    digest (the bundle still validates — nothing re-hashes the curve — but the
    record would lie).
  * `fit tier-search` — standalone re-run against an emitted bundle, for
    tightening a tier later without redoing the calibration. It reseals the
    record, the ladder seed material, the registry pin and `SHA256SUMS`
    afterwards: rewriting the record moves its digest, which moves the registry
    entry's pin, which moves the entry digest, and `validate_bundle` fails on the
    first stale pin.

  Floors are frozen *before* the search runs, deliberately: the artifacts it
  selects must not feed back into the floor they are judged against. Neither the
  search nor the reseal touches the guard profile or the floor derivation.

  The gate is **KL alone**, per v0.3 (`fidelity_search.TierContract.passes`). The
  model's calibrated same-top floor is carried through the report as
  `same_top_reference` / `clears_floor` — the same "reference, not gate" role the
  tier contract already gives it — but it never decides a verdict. This is not
  cosmetic: on occamy the mini floor sat **0.0003** above an 11.27 GiB point that
  clears the 0.20 anchor, and the balanced floor left only **0.0012** of headroom,
  so a dual gate both hides a smaller valid artifact and can steer a
  KL-clearing probe *upward* when its same-top dips.

- **Precision floors for the tensors the candidate set cannot reach**
  (`fit plan --always-active-floors`).

  `generate_upgrade_candidates` admits **only positive-size lower-to-upper
  transitions**, so a tensor typed *identically* in both bracketing presets never
  becomes a candidate — it is unreachable at any budget, under any policy. That
  is not an edge case: it is the always-active machinery. In an `IQ2_XS`→`IQ3_XXS`
  window the attention, SSM and embedding tensors are typed the same at both ends,
  so no plan can protect them however much budget it has.

  `ALWAYS_ACTIVE_FLOORS` (`ffn_{gate,up,down}_shexp` → `Q6_K`, `attn_q/k/gate`,
  `ssm_{alpha,beta,out}`, `token_embd.weight` → `Q4_K`) is a *floor*, not an
  override: it is billed as ordinary spending, and the oracle loop re-selects on
  measured overshoot, so the budget is **redirected, never increased**. The
  llama.cpp native heuristic already lifts `attn_qkv`, `attn_v` and `output.weight`
  one step in every preset, so those are deliberately not floored.

  At a fixed size on occamy the floors are worth **−16.4% macro KL**
  (14.90 GiB @ 0.0835 with, vs 14.89 GiB @ 0.0999 without) — and they are what
  makes a lower base preset safe to use at all.

- **`fit tier-search --restart-tiers <tier,...>`** — re-solve a tier from the
  ladder instead of from its own history. A policy change moves **every** point,
  and because the search ranks by size, a marginally smaller older point can
  outrank a much better allocation and freeze the tier before the new one is ever
  measured. `--restart-tiers` drops that tier's own probes and sweeps, leaving the
  ladder presets — the bounds no policy can move — as the bracket.

- **A lower base preset per tier** (`bracketing_pair(steps=2)`). A plan can only
  spend `target − lower_preset`, so anchoring one rung lower buys a full ladder
  step of headroom at no cost to the target: `balanced` moved from an
  `IQ3_M`-based window to `IQ3_XS`/`IQ3_XXS` and found 13.80 GiB where the
  adjacent pair had bottomed out at 14.89. The rule is not free — a lower base is
  a *wider* window with a coarser candidate set, and on occamy the `IQ3_XXS`-based
  probes at 13.64/13.71 GiB both failed where the `IQ3_XS`-based sweep at
  13.80 passed — so it is a widening, not a guarantee.

- **The Reference tier, and capitalized tier names.** `KL_ANCHORS` gains
  `reference: 0.02` — near-reference fidelity at the smallest artifact that
  reaches it — and `TIER_DISPLAY` gives every tier the name the product uses
  (`Mini`, `Compact`, `Balanced`, `Quality`, `Reference`). A **product tier need
  not be a calibrated one**: the guard profile carries the tiers a calibration
  derived floors for, while the product offers the tiers in `KL_ANCHORS`, so a
  wanted tier that the profile does not carry is planned with `floor: None` (the
  same-top reference is informational anyway) instead of being refused.

  **A tier's window decides what the tier can be, and the candidate palette is
  the union of the two endpoint recipes' types.** On occamy the search bracketed
  the 0.02 anchor with `Q4_K_M`–`Q6_K` — a window whose every tensor offers
  exactly one transition, `q4_k → q6_k`. Its probes plateaued at 26.21 / 26.34 /
  26.40 GiB with KL 0.0224 / 0.0220 / 0.0217, never reaching the anchor, so the
  tier fell back to the `Q6_K` preset at 26.56 GiB. A `Q5_K_M`–`Q8_0` window has
  **432 candidates** across every role, and one probe at 23.6 GiB reached
  **23.55 GiB @ KL 0.0195** — 3.00 GiB under the preset, and ahead of the
  external `APEX-I-Balanced` recipe (23.60 GiB @ 0.0197) that had prompted the
  question. The window is not a bracket, it is the *palette*: a window whose
  endpoints differ in one type can interpolate, and only interpolate.

- **`scripts/sweep_sizes.py`** — plan → quantize → evaluate one or more exact
  sizes from a frozen analysis. `fit tier-search` bisects for the smallest
  artifact that passes a gate, which cannot answer *"same bytes, different
  distribution, which KL?"*: the old point is still the smallest PASS, so the
  search stops and the new allocation is never measured. An unreachable size is a
  **result**, not a crash — floors are mandatory spending, so every window has a
  minimum achievable size above its lower preset.

  `--curve-bundle` registers each measured point into the bundle's curve as it
  goes — plan and recipe included, because a ledger entry that cannot be rebuilt
  is not a ledger entry — and **`scripts/register_curve_points.py`** ingests
  sweeps that ran before that existed. A measurement the search cannot see does
  not exist: on occamy a paid-for 13.80 GiB / 0.0959 floor artifact sat in a
  sweep report while the balanced tier reported a worse, larger point as its
  winner.

  A sweep also decouples *measuring* from *bracketing*: once a sweep has measured
  a passing point, `fit tier-search --budget 0` records the verdict and pays for
  no probes, which is how occamy's reference tier was re-solved in seconds after
  its window changed.

- **`scripts/emit_tier_artifacts.py`** — turns a tier-search report into the
  shipped files. `fit tier-search` answers *which* artifact each tier ships and
  leaves its recipe behind; nothing emitted it. The script quantizes from the
  recorded analysis + recipe, names the result
  `<model>-FIT-<TIER>-<size>G-<type>.gguf` (suffix kept a nameable preset — the
  Hugging Face model page drops files whose suffix it cannot match), and
  **re-evaluates the shipped bytes** so the report carries the file's own measured
  KL rather than a promise inherited from the probe.

### Fixed

- **A curve point records which planning POLICY produced it, not just whether
  floors were on.**

  The regime boolean answers "were floors active". It cannot answer "which table,
  applied how" — and those move a plan too. When the floor-application bug above
  was fixed, every point in the ledger still claimed `always_active_floors: true`
  while four of five tiers' winners had been built by the old semantics, so the
  search would have kept defeated-floor artifacts as their tiers' winners and the
  fix would have changed nothing that ships.

  Points now carry `floor_policy`: a digest of the floor table plus
  `FLOOR_SEMANTICS_VERSION`, which is bumped when the *meaning* of the table
  changes while its text does not. `reusable_points` requires both the regime and
  the policy to match, so editing the table or fixing how a floor is applied
  invalidates exactly the points that are no longer comparable — automatically
  for table edits, and by one integer for semantic ones.

  This is the third instance of the same class in one project: a policy moved, the
  ledger did not notice, and because the search ranks by size a stale point did
  not add noise — it won.

- **A precision floor applies only where the candidate set cannot reach.**

  The table exists because the candidate set is *blind*: a tensor both bracketing
  presets type identically never becomes a candidate, so no budget can move it.
  Where the set **can** reach the tensor, a floor is not a guarantee — it is a
  constraint on an optimizer that could have spent there by itself.

  Measured on occamy by counting which floored roles each window's candidate set
  covers:

  | window | floored roles reached | what the floors were worth |
  | --- | --- | --- |
  | `IQ3_XS`–`IQ3_M` | 5 of 10 | **−20.8% macro KL** at a fixed 14.10 GiB |
  | `Q3_K_M`–`Q4_K_M` | 9 of 10 | about **+1.2 GiB**, no gain |
  | `IQ4_XS`–`Q4_K_M` | 9 of 10 | about **+1.2 GiB**, no gain |

  The same table was rescuing the low tiers and taxing the high ones.
  `floor_overrides` now skips a tensor the candidate set already covers, and
  `FLOOR_SEMANTICS_VERSION` moves to 3 so every point planned under the old
  semantics stops being reusable — which is exactly what the policy id is for.

- **"Smallest" is resolved to within a megabyte before KL is consulted.**

  A pure size ordering is not stable at the resolution the search decides at. The
  quality tier had a passing point **295 KiB** (0.0016%) under its incumbent at
  **1.7% worse macro KL**, and size alone picked it.

  The window is deliberately far below the bisection tolerance (128 MiB): that gap
  is the search's own resolution, and a tie window that wide would quietly
  overturn a tier's definition — on the mini tier it would have traded 113 MiB
  (1% of the file) for 2% of KL, which is a product decision wearing the clothes
  of rounding.

- **`emit_tier_artifacts.py` merges its report instead of replacing it.**
  Re-emitting one tier is a normal operation — a tier was re-solved, or a single
  artifact was rebuilt — and overwriting `emit-report.json` would erase the tiers
  the run did not touch. This mirrors the merge `fit tier-search` already does for
  its own report, for the same reason.

- **A precision floor could be silently defeated by the plan it was correcting.**

  `llama-quantize` resolves a tensor-type file by **first match**: a second line
  for the same tensor is not a merge, it is dead text, and the earlier one takes
  the tensor. `apply_floors` *appended* its raise to the optimizer's selection
  instead of replacing the pick for that tensor, so any tensor the optimizer had
  also chosen produced two lines and the floor lost the tie.

  Auditing the five shipped occamy tiers against their own recipes: Reference 0
  defeated, Quality **100**, Balanced **85**, Compact **258**, Mini **260** — the
  shared experts shipping at the base preset's type where the floor said `Q6_K`,
  and on Compact/Mini `attn_gate` and `ssm_alpha` below their floors as well. The
  measured KL in those tiers stays honest (the eval ran on the bytes that
  shipped), but the *policy* was not the one the release described.

  The same bug produced the quieter failure that found it: a fixed-size A/B of
  two candidate floor rules reproduced the incumbent artifact **byte for byte**
  (identical sha256) — a comparison that cannot change the bytes cannot measure
  anything.

  `apply_floors` now replaces the pick, and `write_tensor_type_file`
  deduplicates as a second lock, keeping the highest type, so the file cannot
  carry a duplicate however a caller assembled the plan. The cost arithmetic is
  unchanged and now correct by construction: the floor's delta is measured from
  the pick it replaces, so the pick's delta plus the floor's compose to a single
  preset-to-floor step.

- **A curve point is only reusable in the floor regime it was planned under.**

  Turning precision floors on is a policy change, but nothing in a tier name says
  so — and the search ranks by size, so a stale point does not merely add noise,
  it *wins*. On occamy the pre-floor `tier-balanced-s3` (14.89 GiB, KL 0.0999) sat
  just under the floor artifact at 14.90 GiB, so the search selected it and then
  read its own bracket as "under tolerance"; the 13.80 GiB / 0.0959 floor
  artifact that actually wins the tier was never looked at. `quality` lost the
  same way: 19.13 GiB reported, 17.50 GiB / 0.0464 available.

  Points now carry `always_active_floors`, and `reusable_points` refuses to mix
  populations — a point measured under another policy is not a bound, it is a
  different experiment that happens to share a name. Resolution order is the
  point's own field, then the probe block, then the archived plan record, then
  `None`; **`None` is never reused**, because a pre-floor plan predates the field
  entirely and guessing would silently merge two incomparable populations. Ladder
  presets are exempt *by name* rather than by default: they are uniform
  quantizations that never consult the table, so they are the one thing a regime
  change cannot move.

  `--restart-tiers` remains the explicit "forget this tier's history" knob; the
  regime filter is not a substitute for it, it is what makes it safe to forget
  less. Corrected occamy result: `balanced` 13.80 GiB @ 0.0959 (was 14.89 @
  0.0999), `quality` 17.50 GiB @ 0.0464 (was 19.13 @ 0.0490).

- **A tier answered by a ladder preset is reported, not duplicated.**
  `emit_tier_artifacts.py` reproduced every winner from
  `probes/<point>-plan.json`, which a preset does not have — it errored out
  instead. A preset has no recipe of its own: the preset *is* the recipe. The
  tier is now recorded with its size and measurement and `preset_fallback: true`,
  and nothing is written, because copying 26 GiB of a standard file so that a
  lineup looks complete is a worse lie than the missing file. The chart reads the
  row either way.

- **A failed probe quantize no longer takes the whole calibration down.** On
  occamy the balanced-tier probe aborted an entire run with `EDQUOT` — a partial
  ladder-`Q8_0` artifact had exhausted the scratch quota — while 11 evaluated
  points sat in the bundle. A probe that cannot be written now logs, unlinks its
  partial artifact, and is skipped; the bracket is unchanged, so it costs one
  budget slot and nothing else. A failed ladder quantize likewise unlinks the
  partial file it leaves behind, which is what starved the next stage.

## [0.3.3] — 2026-09-10

A released file name has to be readable by the tools that read file names.

### Fixed

- **`…-Q4_K.gguf` is not a name any GGUF reader recognises, and the release
  path could produce it.** `Q4_K`, `Q3_K` and `Q5_K` are *tensor types*, not
  presets — llama.cpp ships only the `Q4_K_S` / `Q4_K_M` / `Q4_K_L` (and `Q3_K_*`,
  `Q5_K_*`) variants. A FIT recipe whose element-weighted dominant type landed on
  one of those three was named after it, producing a file name with no
  recognisable quantisation token at all. The observed consequence is not
  cosmetic: **Hugging Face's model page drops such a file from its
  quantisation-variant panel entirely** — a repository with five GGUFs listed
  four, silently omitting `…-Q4_K.gguf`, and reported "We're not able to
  determine the quantization variants."

  `primary_type_from_plan()` now guarantees its return value is a **nameable
  preset** (`PRESET_FILE_TYPES`), which it did not before:

  | Recipe | Suffix |
  | --- | --- |
  | overrides no tensor | the window's lower preset (`Q6_K`, `IQ3_M`, …) |
  | overridden, dominant type is a preset name | that dominant type (`IQ4_XS`, `IQ3_S`, …) |
  | overridden, dominant type is a bare tensor type | the recipe's **base preset** |

  The third case is not a guess: `general.file_type` in the artifact's own
  metadata already carries exactly that preset (analysis.json records
  `PRESET_FILE_TYPES[lower_preset]`), so the name now agrees with the file
  instead of contradicting it. On the MiniCPM5-2B-abliterated batch this
  renamed QUALITY `…-Q4_K.gguf` → `…-Q4_K_M.gguf` (file_type 15) and COMPACT
  `…-Q3_K.gguf` → `…-Q3_K_M.gguf` (file_type 12); bytes unchanged, and both
  files were missing from the model page's variant list before the rename.

  As a side effect the zero-override branch is now strict: a plan that records
  no `lower_preset` returns None and the product path refuses, rather than
  falling back to the dominant type — that fallback is exactly the IQ3_M →
  IQ3_S misdescription the rule exists to prevent.

### Notes

- **235 tests pass, 1 skipped.** Naming coverage pins the invariant directly:
  every case the rule can take must return a member of `PRESET_FILE_TYPES`,
  and the three bare tensor types are asserted *not* to be preset names. A
  regression here is invisible in the artifact and visible only on a model page,
  so it is pinned rather than commented.

## [0.3.2] — 2026-09-10

One freeze, every model; no silent CPU evaluation; and a release name that
says what the file is.

### Fixed

- **Evaluation no longer silently falls back to CPU.** A llama.cpp Windows
  CUDA release ships `ggml-cuda.dll` inside the binary directory but keeps the
  CUDA runtime it links against (`cudart64_*.dll`, `cublas*_*.dll`) in a
  separate `cudart-*` directory, and the release notes put "put cudart next to
  the binaries" on the user. When the two directories sit side by side — which
  is exactly the layout the llama.cpp-hub launcher produces — `ggml-cuda.dll`
  fails to load and llama.cpp **quietly evaluates on CPU**. Nothing errors, the
  numbers are still valid, and the only symptom is time: measured on a 2.5B BF16
  model at the evaluator's own `-b 512`, **150 t/s on CPU versus 11,779 t/s on
  CUDA (78x)**. A full 12-preset ladder plus probes went from hours to minutes.

  `llama_integration.cuda_runtime_siblings()` finds such a sibling (any
  directory beside the runtime that carries `cudart64_*.dll` / `cublas64_*.dll`)
  and `llama_integration.runtime_env()` puts it, plus the runtime directory
  itself, on the loader path — `PATH` on Windows, `LD_LIBRARY_PATH` elsewhere.
  Both evaluation hot loops (`calibrate._run`, `fidelity_runner._eval_domains`)
  now spawn `llama-perplexity` with that environment instead of inheriting one
  that may or may not be configured correctly.

- **Delivered tiers carry their primary type in the file name.** The product
  path emitted `<model>-FIT-<TIER>-<size>GiB.gguf` and dropped the type, even
  though the record written beside it advertised `…-<size>-<primary-qtype>`:
  the name and the promise disagreed. Tiers now ship as
  `<model>-FIT-<TIER>-<size>GiB-<type>.gguf`. `<type>` is the window's **lower
  preset** (`Q4_K_M`, `Q6_K`, …) when the recipe overrides no tensors, because
  those bytes *are* that preset, and the element-weighted **dominant type**
  (`Q4_K`, `IQ4_XS`, …) when it does, because then no native preset produces
  those bytes. Naming the zero-override case for its dominant type would
  misdescribe the file it ships — the IQ3_M preset's dominant type is IQ3_S.

  `pipeline.primary_type_from_plan()` owns the rule; the search records each
  deliverable's plan under `artifact_plans`, and summaries written before that
  field existed are recovered from the plan record the search already writes
  beside the tensor-types file. A deliverable whose primary type cannot be
  established now fails instead of shipping under a name that does not describe
  it.

- **A preset that lands on a window boundary is reproduced from the side that
  can reach it.** When a tier's answer is a native preset whose size is exactly
  a boundary shared by two adjacent windows, `_best_analysis` took the first
  window containing it — the one where that size is the **upper** bound.
  Planning is upgrade-only from the window's lower preset, so the same preset is
  exactly reproducible as a lower bound (recipe = that preset, zero upgrades)
  but as an upper bound needs every upgrade in the window gap to fit, and the
  tail of a gap usually admits none: the MINI tier failed with `exact size
  1,226,310,752 is not deliverable by this window's candidate ladder` — 7,077,888
  bytes short. `_best_analysis` now prefers the window whose **lower** bound
  equals the answer. Interior points have exactly one containing window, so
  their behaviour is unchanged.

### Changed

- **The eval-v1 freeze is model-independent again.** A freeze states *how to
  measure*, so it must cover any model, but `verify_eval_v1_provenance` also
  enforced `freeze_conditions.reference_regeneration.manifest_sha256_prefix` —
  a pin on the literal bytes of *one* reference manifest. That made a freeze
  valid for exactly one model and forced per-model surgery on a release
  document: onboarding a second model meant hand-building a new freeze whose
  prefix matched its manifest, or bypassing provenance entirely — which is what
  the Spark-X2.5 onboarding had to do, recorded there as an
  `M2-style model calibration`.

  The prefix is now read as a historical v0.2 bootstrap record and **not
  enforced**. The bindings that actually constrain a manifest are unchanged and
  still fail closed: the frozen contract digest, the manifest's
  `evaluator_contract_hash` against it, the manifest's
  `source_bf16_gguf_sha256` against the weights being quantized, and every
  per-domain reference/corpus SHA-256. Per-model trust is the Fidelity
  Registry's job — it already pins each released model's manifest by **full**
  SHA-256, keyed by `source_weights_sha256`, which is a strictly stronger pin
  than a 16-hex prefix and does not need to live in a frozen document.

- **Reference manifests are discovered per model.**
  `discover_reference_manifest()` looks for the manifest beside the references
  it describes — `references/reference-manifest.json`, then
  `reference-manifest.json` one level up, which is the layout `fit calibrate`
  writes for every model. The v0.2 freeze-adjacent `reference-manifest-*.json`
  glob survives as a last-resort fallback, and now reports ambiguity instead of
  silently picking one. `fit fidelity-search` therefore needs no per-model
  arguments beyond the model's own inputs.

- **A calibration bundle feeds the search directly.** `fit calibrate` spends
  five-domain evals on every standard-ladder preset, and the product search
  could not use a single one of them: bracket seeds are admitted from a
  `<name> <size> <sha256>` manifest plus an attested provenance sidecar, and the
  calibration emitted neither. The bundle now carries both
  (`state-artifact-manifest.txt`, `seed-provenance.jsonl`), so pointing
  `--manifest`/`--logs-dir` at a bundle turns its ladder into budget-free
  bracket evidence. Native-preset points name their preset on both window
  anchors, which is what keeps a poison preset (IQ2_XS) out of bracket evidence
  downstream; probe points carry no anchors because they are planned inside
  healthy windows by construction.

- **`--seed-prefix` defaults to a match-all.** The manifest and log directory
  already scope a search to one model's bundle, so requiring a name prefix on
  top only forced every model to be spelled identically in three places just to
  reuse its own calibration. Admission control is unchanged and does the real
  work: a seed is admitted only when its provenance sidecar attests the live
  frozen contract *and* the exact reference-manifest file in use, and when its
  name appears in the given size manifest.

### Notes

- 233 tests pass, 1 skipped. New coverage pins the generalization: one freeze
  accepts a second model's manifest while still refusing foreign weights,
  manifest discovery prefers the model bundle / falls back to the bootstrap
  layout / errors on absence or ambiguity, and a calibration bundle's own
  manifest + sidecar round-trip back through `load_seeds` into admissible seeds
  (poison preset excluded, wrong reference manifest rejected). Naming is pinned
  on both sides of the rule — a zero-override plan names its preset, an
  overridden plan names its dominant type — together with the delivered-name
  format and the plan-record lookup, including summaries written before
  `artifact_plans` existed.

## [0.3.1] — 2026-09-10

Windows support, and a checkout that reproduces the bytes it pins.

### Fixed

- **The product CLI runs on Windows.** Every llama.cpp binary was resolved as
  ``runtime_dir / "llama-quantize"`` (and `llama-imatrix` / `llama-perplexity`),
  an extensionless name that only exists in POSIX builds, so a correct Windows
  runtime directory failed with `llama-quantize not found at …` before any work
  started. Eight call sites are now platform-aware via
  `llama_integration.resolve_runtime_binary()`: `.exe` is preferred on Windows
  and the extensionless name on POSIX, while every shipped form is tried on
  every platform. `.cmd`/`.bat` shims are accepted on Windows too, because
  CreateProcess launches those directly. The recorded `analysis.json` runtime
  path is now the *resolved* one, and `quantize` re-resolves from the recorded
  directory, so an analysis written before this change (a Linux-produced
  analysis replayed on Windows, say) still replays without re-analysis.
- **A fresh clone verifies its own digests.** The Fidelity Registry pins SHA-256
  over the exact bytes of checked-in files, but the repository had no
  `.gitattributes` and Git for Windows defaults to `core.autocrlf=true`, so a
  checkout rewrote LF to CRLF and `fit registry verify` failed every entry with
  `guard_profile sha256 mismatch` — 3 tests' worth of false alarm on an
  unmodified clone. A repository-wide `* -text` now disables end-of-line
  translation; the fix is deliberately not narrowed to specific paths, since any
  hashed text file would otherwise be a silent integrity hole.
- **`fit calibrate`'s default scratch no longer resolves inside the source
  drive on Windows.** `Path("/dev/shm")` is a real tmpfs on Linux but silently
  becomes `\dev\shm` on the current drive on Windows.
  `calibrate.default_scratch_root()` prefers `FIT_CALIBRATE_TMP`, then `/dev/shm`
  when it exists, then the platform temp directory.
- `_runtime_env()` sets `PATH` for the runtime directory on Windows, matching
  what `LD_LIBRARY_PATH` does on POSIX, so a runtime's own shared libraries
  resolve the same way on both platforms.

### Changed

- **The test suite is portable.** The fake llama.cpp runtime was a `bash` script
  and `chmod 0o755` only applies on POSIX, so 7 tests died with
  `OSError: [WinError 193]` on Windows. `tests/stub_runtime.py` now writes a
  `.cmd` launcher on Windows and an `sh` script elsewhere, both exec'ing one
  Python implementation, so the end-to-end tests still exercise a real
  subprocess on both platforms. The stub also emits its output as explicit UTF-8
  bytes: the fake KL log contains `±`, and a redirected child on a non-UTF-8
  Windows code page (cp936, cp932) would otherwise encode it locally while the
  parent decodes as UTF-8. Also fixed: the wheel smoke test assumed a venv's
  `bin/` (`Scripts/` on Windows), and `test_mount_fs_type_resolves_real_mounts`
  no longer requires `/dev/shm`.
- `setuptools` and `pip` joined the `test` extra: the opt-in wheel smoke test
  builds with `--no-build-isolation`, so both the build backend and the build
  frontend must be importable from the test environment. A stdlib
  `python -m venv` ships pip, `uv venv` does not, so declaring it keeps both
  documented setup paths equivalent. That test also no longer swallows its
  subprocess stderr behind `check=True`: a missing dependency used to surface as
  a bare `returned non-zero exit status 1`.

### Notes

- 216 tests pass, 0 skipped in a full development checkout on Windows (216
  collected). A bare clone reports 214 passed / 2 skipped: the two
  `test_gguf_traits_source.py` cases re-parse a pinned llama.cpp checkout at the
  gitignored `third_party/llama.cpp/`. The 7 tests that used to require POSIX now
  pass on both platforms, and `tests/test_runtime_binary.py` pins the candidate
  order for both platforms via the `_is_windows` probe.
- Windows NTFS is unaffected by the `ntfs3` hot-loop guard: that bug is in the
  Linux driver, and `mount_fs_type()` correctly reports unknown without
  `/proc/mounts`.

## [0.3.0] — 2026-09-10

Onboard a new model with one command, and make the trust root inspectable.

### Added

- **`fit calibrate`** — the calibration production line. Pins its inputs,
  generates or reuses an imatrix, builds five aligned BF16 references, walks a
  standard preset ladder, spends at most **4 gap probes per tier**, derives
  Same-top floors inside the tier windows `W(K) = [0.85K, 1.15K]`, and emits a
  **Calibration Bundle**: `calibration-record.json`, `reference-manifest.json`,
  `curve-points.jsonl`, `guard-profile.yaml`, `profile-report.md`, a candidate
  `registry-entry.json`, `SHA256SUMS` and the bundled `references/`.
- **`fit registry`** — `list` / `show` / `verify` / `validate`. Entries are keyed
  by exact source-weights SHA-256; `verify` performs structural, hash and
  cross-object closure checks.
- **`src/fit_gguf/contracts/fidelity-calibration-v1.json`** — machine-readable
  calibration contract: window rule, n ≥ 3 sample minimum, artifact-SHA dedup
  key, P5 floor derivation with decimal truncation, promotion rule and
  failure-state enumeration.
- **Fidelity Registry v1** trust root under `src/fit_gguf/registry/`, with three
  entries: `orcarouter-Qwen3.8-27B-Uncensored`, `spark-x25-4b-abliterated` and —
  onboarded end-to-end by a single `fit calibrate` command as the v0.3
  acceptance run — `Qwen3-4B` (source `f3e9d463…`, ~38 min, 12-preset ladder,
  gap probes within the 4-per-tier budget, `overall=validated`). That run also
  confirms the H-M3-02 capacity expectation directionally: at equal KL, top-1
  agreement sits below the 27B sibling in all four tiers
  (quality −3.56 pt, balanced −3.06 pt, compact −4.24 pt, mini −1.43 pt).
- **A2 execution profile** — `--n-gpu-layers`, `--threads`, `--workdir` and
  `--on-disk` across `calibrate` and the search executor; documented in
  `docs/execution-profile.md`.
- `fit calibrate --replay-existing` — zero-evaluation re-derivation from recorded
  observations.

### Changed

- **The fidelity tier gate is now KL only (Fidelity Contract v2).**
  `PASS = macro KL ≤ tier anchor`; the anchors stay the frozen global constants
  (`Quality` 0.05 / `Balanced` 0.10 / `Compact` 0.15 / `Mini` 0.20), so a tier is
  one fixed, comparable, model-independent target. Same-top agreement is still
  measured, reported and archived on every point — and still shown against the
  model's calibrated floor when the registry has one, now named
  `same_top_reference` — but it never changes a verdict. Consequences:
  - Naming a tier no longer requires a validated Guard Profile, so
    `fit plan --fidelity-tier X` and `fit fidelity-search --tier X` work on a
    model with no registry entry at all. `--guard-registry` is optional and, when
    a profile does cover the exact weights, supplies the informational reference.
  - On the two calibrated models the change is narrow: it moves the answer in
    **1 of 4 tiers**. Spark-X2.5-4B `Quality` becomes 2.85 GiB instead of
    3.15 GiB (−9.5%) with top-1 agreement 91.34% instead of 93.50% (−2.16 pt);
    `Balanced`, `Compact` and `Mini` are unchanged.
  - v0.2 tier results and their PASS/FAIL labels were produced under the v0.2
    dual gate and remain as historical release records.
  - Because the gate no longer rides on a Guard that pins weights, the
    source-GGUF ↔ reference-manifest binding in the product path is now checked
    **unconditionally**: quantizing weights that differ from the ones the
    references were built from fails closed instead of silently producing
    meaningless KL numbers.
- Guard Profiles moved from the repo-level `profiles/guard/` into the package
  (`src/fit_gguf/profiles/guard/`) and are declared as package data, so a built
  wheel ships them. The v0.2 wheel shipped without them.
- `.gitignore` hardened: reference logits (`*.kld`) are now excluded everywhere,
  and the duplicated `experiments/**` rule block was collapsed.

### Fixed

- **`fit calibrate` no longer panics the host on `ntfs3`.** `_run` handed the
  subprocess's stdout/stderr file descriptor straight to llama.cpp; with the log
  file on an `ntfs3` mount, llama.cpp's unbuffered stderr becomes thousands of
  short, unaligned, page-spanning buffered writes, which trip
  `kernel BUG at fs/iomap/buffered-io.c:1061` in `iomap_write_end` and take the
  machine down (three kdump captures: 2026-09-06 ×2, 2026-09-10; the writing
  process was `llama-quantize` or `llama-perplexity` every time). Subprocess logs
  and reference logits are now staged on the scratch volume and bulk-copied into
  the bundle afterwards, and `assert_hot_loop_fs_safe()` refuses a hot-loop
  destination on `ntfs3` unless `FIT_ALLOW_UNSAFE_FS=1` is set. `fit analyze`,
  `fit plan`, `fit quantize` and `fit fidelity-search` were never affected —
  `pipeline.py` captures subprocess output with `capture_output=True` and writes
  the log itself.
- `fit plan` resolves the default Guard registry from inside the package instead
  of assuming a repository-relative `profiles/guard` path.
- The calibration contract now ships in the wheel. `default_contract_path()`
  resolves to `<package>/contracts/fidelity-calibration-v1.json`, but
  `package-data` declared only `registry/` and `profiles/guard/`, so an installed
  package failed at `fit calibrate` and `fit registry validate`. `tests/test_packaging.py`
  now expands the declared `package-data` globs and fails if any runtime-loaded
  asset is not shipped — this is the second release in a row to hit a
  "present in the repo, missing from the wheel" defect (v0.2 shipped without the
  Guard Profiles), and the first case where a test guards the class.

### Notes

- Fail-closed semantics are unchanged and extended: a tier whose window cannot be
  filled reports `INSUFFICIENT_WINDOW` and stays `candidate`; floors are never
  borrowed across models, the sample minimum is never relaxed, and search
  observations are never back-filled into the floor set.
- 203 tests pass, 1 skipped (204 collected).

## [0.2.1] — 2026-09-06

### Added

- Second-model onboarding: `Spark-X2.5-4B-abliterated` calibration and fidelity
  tiers.

### Fixed

- Tolerate non-UTF-8 bytes in llama.cpp subprocess output.

### Changed

- Backfilled the v0.2.0 release record (planner release verdict + v0.2 chart
  renderer).

## [0.2.0] — 2026-09-03

### Added

- **Fidelity tiers** — `Quality` ≤ 0.05, `Balanced` ≤ 0.10, `Compact` ≤ 0.15,
  `Mini` ≤ 0.20 macro KL, each gated together with a model-specific validated
  Same-top Guard floor.
- **`fit fidelity-search`** — walks the healthy preset frontier, brackets the
  crossing and returns the **minimum verified PASS** rather than an extrapolation.
  Re-evaluates the final artifact itself, reports `noise_inversion` and fails
  closed in locally non-monotonic regions.
- Release-gate hardening: the frozen eval-v1 provenance closure is enforced
  before any evaluation runs.
- Bilingual (EN / zh-CN) release charts.

### Release gates

`R1 Fidelity correctness`, `R2 Search accuracy`, `R3 Search budget`,
`R4 Exact-byte guarantee`, `R5 v0.1 non-regression`, `R6 Reproducibility` —
6 / 6 PASS on the flagship model.

## [0.1.0] — 2026-08-30

Initial public release.

### Added

- `fit analyze` / `fit plan` / `fit quantize` — exact-size planning for GGUF
  quantization: anchor, measure, allocate, oracle-replay, verify.
- Prediction-exact byte targeting with `--expect-bytes` enforcement and output
  SHA-256 recording.
- Five frozen 64 KiB KL evaluation slices with provenance
  (`eval-data/PROVENANCE.md`).
- MIT license, bilingual README and branding assets.
