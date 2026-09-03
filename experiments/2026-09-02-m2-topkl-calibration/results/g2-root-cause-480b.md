# G2 Root Cause — bailingmoe −480B exact-size discrepancy (2026-09-02)

Fixture: L-A1 (plan L-A1, pair IQ4_XS→Q6_K, base IQ4_XS), 526 tensors.
Predicted 5,090,966,048 vs actual 5,090,965,568, Δ = −480 (was 6/6 constant across all
amendment-2 plans).

## Ground truth established first

The ACTUAL file is internally exact: all 526 tensor byte spans match the standard GGUF
payload formula ((ne0/bs)·ts·rows, 32-aligned) with **zero** mismatches, and
metadata (6,516,160 aligned) + payload (5,084,449,408) = actual size. The quantizer is
correct; the entire −480 lives in `predict_output_metadata_size` (fit_gguf/gguf.py).

## Decomposition (exact, byte-level)

| component | model | actual | delta |
|---|---:|---:|---:|
| tensor_info_bytes | 33,727 (modeled from SOURCE layout) | 33,295 | **−432** |
| quantize.imatrix.file | 140 (analyze-time `--imatrix-arg`, 99-char abs path) | 74 (quantize-time path, 33 chars) | **−66** |
| everything else (all KVs, entries/chunks/dataset, header 24) | — | — | 0 |
| alignment slack after corrections | | | −18 (lands 18B below the 32-boundary) |
| **total** | | | **−480** |

Corrected model = 6,516,640 − 66 − 432 = 6,516,142 → aligned 6,516,160 = actual, byte-exact.

## Root causes

1. **Dims normalization on write (−432, deterministic, architecture-driven).**
   b10666 drops trailing singleton dims when writing output tensor infos. Ling
   (bailingmoe) has 54 SSM conv1d tensors stored 4-D `(4,1,2048,1)` in the converted
   BF16; the output writes them 3-D `(4,1,2048)` — 8 bytes saved per tensor info.
   The predictor models tensor infos from the SOURCE layout (keeps 4-D) → overcounts.
   Granite/orcarouter never hit this because none of their tensors carry trailing
   singleton dims. Fix: normalize dims (drop trailing 1s, keep ≥1 dim) when modeling
   output tensor infos.

2. **imatrix path string embedded in the artifact (−66, provenance-dependent).**
   b10666 writes `quantize.imatrix.file` = the imatrix path **as passed on the CLI**.
   The predictor freezes the analyze-time string; any later quantize with a different
   path/string shifts the artifact size by the encoded-length delta. Artifact size is
   therefore **path-length dependent** — a latent hazard on ALL architectures
   (granite/orcarouter matched only because analyze and quantize used the identical
   string). Fix options: (a) pipeline quantize re-derives the prediction from its own
   `--imatrix` string and gates against that; (b) pin a canonical relative basename
   for `--imatrix-arg` at analyze time and require the same string at quantize;
   (c) model the dataset/chunks KVs from the imatrix file itself (the analyzer already
   parses them — values matched exactly here).

## Fix plan (G2)

1. `predict_output_metadata_size`: normalize tensor dims (trailing-1 drop) for the
   tensor-info size model; keep source KVs model as-is (it is exact).
2. Provenance contract: quantize-time gate must re-predict with quantize-time
   `--imatrix` string (or enforce string identity with analyze).
3. Regression fixture: the 6 Ling amendment-2 plans + diff-LA1 artifact → assert
   byte-exact totals after the fix (covers MoE expert layout + 4-D tensors + path
   provenance).
4. Alignment-slack note: with both components modeled, the residual is the usual
   32B alignment of the data section — already modeled via `_align`.

Status: calibration data unaffected (KL/Top measured on real artifacts); the bug only
mis-predicted sizes. Production FIT for bailingmoe stays NOT VALIDATED until this fix
lands with the G2 fixture green.
