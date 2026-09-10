# Changelog

All notable changes to FIT-GGUF. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions use
[Semantic Versioning](https://semver.org/).

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
- **Fidelity Registry v1** trust root under `src/fit_gguf/registry/`, seeded with
  two entries: `orcarouter-Qwen3.8-27B-Uncensored` and
  `spark-x25-4b-abliterated`.
- **A2 execution profile** — `--n-gpu-layers`, `--threads`, `--workdir` and
  `--on-disk` across `calibrate` and the search executor; documented in
  `docs/execution-profile.md`.
- `fit calibrate --replay-existing` — zero-evaluation re-derivation from recorded
  observations.

### Changed

- Guard Profiles moved from the repo-level `profiles/guard/` into the package
  (`src/fit_gguf/profiles/guard/`) and are declared as package data, so a built
  wheel ships them. The v0.2 wheel shipped without them.
- `.gitignore` hardened: reference logits (`*.kld`) are now excluded everywhere,
  and the duplicated `experiments/**` rule block was collapsed.

### Fixed

- `fit plan` resolves the default Guard registry from inside the package instead
  of assuming a repository-relative `profiles/guard` path.

### Notes

- Fail-closed semantics are unchanged and extended: a tier whose window cannot be
  filled reports `INSUFFICIENT_WINDOW` and stays `candidate`; floors are never
  borrowed across models, the sample minimum is never relaxed, and search
  observations are never back-filled into the floor set.
- 189 tests pass, 1 skipped.

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
