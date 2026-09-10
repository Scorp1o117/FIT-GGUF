# Changelog

All notable changes to FIT-GGUF. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions use
[Semantic Versioning](https://semver.org/).

## [Unreleased]

One freeze, every model; and no silent CPU evaluation.

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

- 221 tests pass, 1 skipped. New coverage pins the generalization: one freeze
  accepts a second model's manifest while still refusing foreign weights,
  manifest discovery prefers the model bundle / falls back to the bootstrap
  layout / errors on absence or ambiguity, and a calibration bundle's own
  manifest + sidecar round-trip back through `load_seeds` into admissible seeds
  (poison preset excluded, wrong reference manifest rejected).

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
