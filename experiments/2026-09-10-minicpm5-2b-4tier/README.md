# MiniCPM5-2B-abliterated: third-model onboarding — calibration + four fidelity tiers

**Date**: 2026-09-10 · **Runtime**: llama.cpp b10690 (Windows x86_64, CUDA 13.3) · **Status**: all four tiers **verified_pass**

The first onboarding of an **upstream, unmodified-architecture** model (`llama` — no PR branch needed, unlike Spark's `spark2_5`), and the first run where the **calibration bundle fed the search directly** as budget-free bracket evidence: all four tiers were answered from calibration points, `fresh_evals: 0` on every one.

Artifacts are published at [SC117/MiniCPM5-2B-abliterated-FIT-GGUF](https://huggingface.co/SC117/MiniCPM5-2B-abliterated-FIT-GGUF).

## Result

| Tier | Size (bytes) | Recipe | Macro KL | Same-top | Gates (KL ≤ / top ≥) |
|---|---|---|---|---|---|
| QUALITY | 1,566,057,568 | FIT tensor-level recipe (Q4_K-dominant, 67 overrides) | 0.0493 | 89.90% | 0.05 / 89.52% |
| BALANCED | 1,378,067,552 | FIT tensor-level recipe (IQ4_XS-dominant, 174 overrides) | 0.0929 | 86.27% | 0.10 / 85.48% |
| COMPACT | 1,301,390,432 | FIT tensor-level recipe (Q3_K-dominant, 94 overrides) | 0.1495 | 83.02% | 0.15 / 82.42% |
| MINI | 1,226,310,752 | **native IQ3_M preset** (0 overrides) | 0.1855 | 81.13% | 0.20 / 79.55% |

All four tiers are **KL-bound** — unlike Spark, where the floors decided the sizes. All four carry **G2 delta +0** (delivered bytes equal the re-finalized prediction exactly).

The 2-bit region collapses on this model and is excluded by design: IQ2_XXS 3.9211, IQ2_XS 1.9325, IQ2_M 0.8140, IQ3_XXS 0.3981.

Two findings that generalise beyond this model:

- **MINI's answer is a native preset sitting exactly on a shared window boundary.** `_best_analysis` picked the window where that size is the *upper* bound, but planning is upgrade-only from the lower preset, so the boundary preset is only reproducible from the side where it *is* the lower bound — the search failed with `exact size 1,226,310,752 is not deliverable` (7,077,888 bytes short). Fixed in `fit_gguf.product._best_analysis` (commit `29994b5`).
- **The preset ladder is the binding quality story at this size**, not the tiers: Q6_K (1.93 GiB) is 6× cleaner than QUALITY (1.46 GiB) and Q8_0 (2.50 GiB) 38× cleaner. The tiers answer "smallest file at a given quality", not "best quality".

## Method

1. **Source** — locally abliterated weights (abliterix mean-LoRA, Optuna trial #68 of 80), converted to BF16 GGUF (sha256 `f161afcf…c6e12`; pinned in `state/bf16-gguf.sha`).
2. **imatrix** — APEX-imatrix-Small.txt, 500 chunks × 512 ctx, 294 entries (sha256 in `state/imatrix.gguf.sha`). The imatrix path string is embedded in GGUF metadata, so it affects artifact *byte counts* — see the reproducibility note below.
3. **Calibration** (`calibration/`) — 12-point standard preset ladder + 10 gap probes, five domains each, 22 curve points (`curve-points.jsonl`). Guard Profile `guard-minicpm5-2b-abliterated-exact-v1`: `exact_model` scope, floors by `empirical_p5` (n = 3), source-SHA bound. `overall_status: validated`, no failures.
4. **Bundle → seeds** — `state-artifact-manifest.txt` + `seed-provenance.jsonl` made the whole ladder admissible as bracket evidence, which is why all four searches spent **zero fresh evals**. This is the payoff of the v0.3 "a calibration bundle feeds the search" change.
5. **Fidelity Search** per tier (`fs/<tier>/`) — healthy windows only, budget ≤ 8, 128 MiB tolerance. Final artifacts re-quantized from recorded recipes (`final-plan-*`) and re-evaluated on their own bytes (`*.quantize-record.json`).

Note the delivered sizes are unchanged from the pre-0.3.2 runs; only the *names* gained their primary type (see the release record). `selected_count == 0` is what marks MINI as the native preset — and is exactly the signal the naming rule uses.

## Known issue: the calibration bundle mixes two measurement backends

The bundle's own logs show it (the eval logs stay local — this repo ignores `*.log`, as it did for the Spark run), and this record keeps the census itself:

| Records | Backend | Count |
|---|---|---|
| 12-point preset ladder | **CPU** | 60 / 60 |
| 10 gap probes | CPU 6 / CUDA 4 | 30 + 20 |
| Four-tier final-verify | **CUDA** | 20 / 20 |
| BF16 reference logits | CPU | 5 |

Cause: calibration started **before** the CUDA-runtime-loading fix (`8e544be`, "never evaluate on CPU because the CUDA runtime was not on PATH") and the tier searches ran **after** it. `fit calibrate` resumes by skipping already-recorded presets (`ladder <preset>: already recorded, skipping`), so it never re-measured the ladder — both measurement eras ended up in one bundle. Of the four shipped answer points, only QUALITY is CUDA-basis.

The evaluation itself is sound. **Within one backend it is bit-reproducible**: a fresh CUDA re-run returned the delivery values to the last digit (BALANCED 0.104014, COMPACT 0.171645 on wiki_test — identical to final-verify). Across backends it is not:

| Tier | Recorded | CUDA re-check |
|---|---|---|
| QUALITY | 0.049281 | 0.049281 |
| BALANCED | 0.092908 | 0.091799 |
| COMPACT | 0.149528 | 0.148408 |
| MINI | 0.185472 | 0.183088 |

Largest single-domain gap: COMPACT · wiki_test, CPU 0.161609 → CUDA 0.171645 (6%). **All four PASS either way.**

The published charts plot the recorded values and carry a footnote stating the mixed basis; `results/four-tier-results.json` records the census machine-readably under `measurement_basis`. The calibration bundle was deliberately **not** regenerated: `registry-entry.json` pins `calibration-record.json` by sha256, so re-running would invalidate the guard-profile binding — a separate, planned piece of work.

Re-check data: `re-eval-check.json` in the published release.

## Sizes in the file names are measured bytes

The 0.3.2 naming rule: an overridden recipe is named for its element-weighted dominant type, a zero-override recipe for the window's lower preset. The IQ3_M preset's dominant type is IQ3_S — naming MINI `IQ3_S` would have described a recipe the file does not use. `fit_gguf.pipeline.primary_type_from_plan` owns the rule; a deliverable whose primary type cannot be established now fails rather than shipping under a name that does not describe it.

## Reproducibility

Regenerable from what is here plus the published BF16:

- references — five-domain `.kld` are **not** in this directory (9.6 GiB); regenerate from the published BF16 GGUF, then check against `calibration/reference-manifest.json`, which pins each domain's `reference_kld_sha256`, corpus sha and byte count.
- the imatrix gguf is likewise not stored (path embedded in metadata ⇒ its string affects artifact bytes); `state/imatrix.gguf.sha` pins the exact file used.
- the ladder / probe plans and recipes are in `calibration/probes/` and `fs/<tier>/analyses/`.

Exact-size behaviour is scoped to the recorded source metadata and the recorded llama.cpp build; changing the converter, runtime, source layout or metadata requires revalidation.
