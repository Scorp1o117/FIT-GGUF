# Spark-X2.5-4B-abliterated (T615): second-model onboarding — calibration + four fidelity tiers

**Date**: 2026-09-06 · **Runtime**: llama.cpp @ PR [#27868](https://github.com/ggml-org/llama.cpp/pull/27868) head (ae320b1, Spark2_5 support; its `perplexity.cpp` is byte-identical to the pinned eval-v1 build b10666) · **Status**: all four tiers **verified_pass**

The first end-to-end onboarding of a second, external-architecture model after the v0.2.0 release. Everything here was produced by the record scripts, on this model, with the eval-v1 five-domain protocol (64 KiB slices, 512 ctx, macro KL + same-top against the model's own BF16).

## Result

| Tier | Size (bytes) | Recipe | Macro KL | Same-top | Gates (KL ≤ / top ≥) |
|---|---|---|---|---|---|
| QUALITY | 3,379,173,056 | Q6_K preset | 0.0289 | 93.50% | 0.05 / 93.16% |
| BALANCED | 2,877,626,816 | Q5_K_S preset | 0.0739 | 89.15% | 0.10 / 89.14% |
| COMPACT | 2,611,714,496 | FIT tensor-level recipe (Q4_K-dominant) | 0.1458 | 84.88% | 0.15 / 84.88% |
| MINI | 2,426,554,816 | Q4_K_S preset | 0.1834 | 82.61% | 0.20 / 82.61% |

All four tiers are **same-top-bound** (KL has slack; the guard floors decide the sizes). The preset ladder is strong on this model, so the minimum verified PASS at Quality/Balanced/Mini is the native preset itself; FIT fills the Compact gap between Q5_K_S (0.074) and Q4_K_M (0.170) where no preset exists. 2-bit classes collapse on this model (macro KL 0.8–3.9) and are excluded.

Artifacts are published at [SC117/Spark-X2.5-4B-abliterated-FIT-GGUF](https://huggingface.co/SC117/Spark-X2.5-4B-abliterated-FIT-GGUF).

## Method

1. **BF16 source** — the locally abliterated weights (abliterix mean-LoRA, Optuna trial T615), converted to GGUF BF16 (sha256 `aa73aeb4…`).
2. **imatrix** — APEX calibration corpus, 500 chunks × 512 ctx (`scripts/run_spark_calibration.sh`, tmpfs hot-loop discipline).
3. **References** — five-domain `.kld` from the BF16 with row-completeness arithmetic checks (`state/artifact-manifest.txt` records sizes and hashes; the 11 GB `refs/` tree stays local by gitignore).
4. **Preset ladder** — 15 points × 5 domains (`logs/`, sizes in `state/artifact-manifest.txt`).
5. **Gap probes** — the compact KL window is empty on this model; two plain `fit plan` probes (2.65 / 2.75 GB targets) populate it (`probes/`).
6. **Guard Profile** — `profiles/guard/guard-spark-x25-4b-abliterated-exact-v1.yaml` (v3): floors = per-tier window P5 (n ≥ 3) or window minimum (n ≤ 2), truncated **down**; `validation_basis: dev_calibration`, exact-model scope, source-SHA bound.
7. **Fidelity Search v1** per tier (`scripts/run_fidelity_search_spark.py`, `results-fs/<tier>/`): healthy windows only, budget ≤ 8 fresh evals, 128 MiB tolerance, seeds from all 25 observed points. Final artifacts re-quantized from recorded recipes and re-evaluated as shipped.

## Floor-derivation discipline (paid for in this run)

- A floor **rounded** above its own generating point gates that point out of its tier (run 1 bit Q6_K / Q5_K_S by ≤ 0.005 pp) — floors are truncated down.
- At n = 3 a P5 floor correctly excludes the worst diagnostic point (FS01: same-top 93.13%); a minimum-based floor would have made that FAIL point a PASS with no adjacent FAIL to verify against.

## Governance note

This run is an **M2-style model self-calibration** (`validation_basis: dev_calibration`): the search machinery ran via `RunnerConfig(require_eval_provenance=False)`. The product CLI's eval-v1 freeze pins the orcarouter reference-manifest SHA prefix, which cannot cover a second model; extending the freeze to a per-model manifest registry is a planner decision, deliberately not taken unilaterally here.

## Toolchain fix shipped in v0.2.1

llama.cpp metadata dumps truncate token-array previews mid-codepoint (spark2_5 fullwidth special tokens), producing invalid UTF-8 on stderr; three `subprocess` call sites (`pipeline.run_dry_run`, `pipeline.quantize`, `fidelity_runner.SearchExecutor._eval_domains`) decoded strictly and crashed. Fixed to bytes + `errors="replace"` (152 tests green).

## Reproducibility

`scripts/run_spark_calibration.sh` → `scripts/summarize_spark_ladder.py` → `fit analyze` (windows) → `scripts/run_fidelity_search_spark.py`. The pipeline is deterministic; the COMPACT recipe re-quantizes byte-for-byte (verified). Evaluation slices ship in the repo under `eval-data/`.
