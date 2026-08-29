# P2: Full-Envelope Extension (IQ1_S .. Q8_0)

Date: 2026-08-29

## Scope

Owner-directed envelope extension: FIT must cover every quantization size
llama.cpp build 10666 supports, from IQ1_S up to Q8_0 (user: "覆盖 llama.cpp
支持的所有量化尺寸"). This is engineering, not allocator research: no optimizer,
threshold, or policy behavior changes. All changes are additive tables.

## 1. Type-trait extension (code, frozen by this document)

`GGML_TYPE_TRAITS` gains ten entries, with (block_size, type_size) taken
verbatim from the pinned source `ggml/src/ggml-common.h` static_assert
expressions (QK_K=256, QK4_NL=32, QK8_0=32, K_SCALE_SIZE=12, IQ3S_N_SCALE=4):

| qtype | block | size | BPW |
| --- | ---: | ---: | ---: |
| iq1_s | 256 | 50 | 1.5625 |
| iq1_m | 256 | 56 | 1.7500 |
| iq2_xxs | 256 | 66 | 2.0625 |
| iq2_xs | 256 | 74 | 2.3125 |
| iq2_s | 256 | 82 | 2.5625 |
| iq3_xxs | 256 | 98 | 3.0625 |
| q2_k | 256 | 84 | 2.6250 |
| q3_k | 256 | 110 | 3.4375 |
| q8_0 | 32 | 34 | 8.5000 |
| iq4_nl | 32 | 18 | 4.5000 |

Existing six entries (iq3_s 110, iq4_xs 136, q4_k 144, q5_k 176, q6_k 210,
bf16/f32) are unchanged and their derivation is reconfirmed by the same
assert expressions.

`PRESET_FILE_TYPES` gains the pinned quantize.cpp preset names with their
llama.h LLAMA_FTYPE values: IQ1_S 24, IQ1_M 31, IQ2_XXS 19, IQ2_XS 20,
IQ2_S 28, IQ2_M 29, IQ3_XXS 23, IQ3_XS 22, IQ4_NL 25, Q2_K 10, Q2_K_S 21,
Q3_K_S 11, Q3_K_M 12, Q3_K_L 13, Q4_K_S 14, Q4_K_M 15, Q5_K_S 16, Q5_K_M 17,
Q6_K 18, Q8_0 7 (existing IQ3_S 26, IQ3_M 27, IQ4_XS 30 unchanged). As in
P1, only each file_type KV's presence and encoded size matter; values are
recorded for fidelity.

A unit test re-parses the pinned `ggml-common.h` static_assert expressions
and asserts every trait entry against it (skipped if third_party is absent).

## 2. Envelope validation on the first release source

Source: `orcarouter/Qwen3.8-27B-Uncensored` safetensors (Apache-2.0, 18
shards, 51.75 GiB, arch Qwen3_5ForConditionalGeneration, 64 layers, vocab
248,320, MTP head present, vision tower present), converted with the pinned
third_party converter to BF16 GGUF using the research-consistent
`--no-nextn` (D-0008). The MTP-in-release question is a separate experiment
and is not gated here.

Steps and gates (all frozen before execution):

- E1 Conversion gate: converter completes; the BF16 GGUF's tensor set,
  SHA-256, and size are recorded. Sanity: tensor count within the 851-region
  count reported by the pinned converter for qwen35 without the MTP head.
- E2 imatrix: built from `APEX-imatrix-Small.txt` at chunk 512 (the owner's
  standing calibration directive; proven on Granite in M16) with pinned
  llama-imatrix, GPU offload allowed. Imatrix SHA-256 and chunk/entry counts
  are recorded before any quantization.
- E3 Preset ladder sweep: one `--dry-run --imatrix` per accepted FIT preset
  (IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ2_M, IQ3_XXS, IQ3_XS, IQ3_S,
  IQ3_M, IQ4_NL, IQ4_XS, Q2_K, Q2_K_S, Q3_K_S, Q3_K_M, Q3_K_L, Q4_K_S,
  Q4_K_M, Q5_K_S, Q5_K_M, Q6_K; Q8_0 included as ladder top). Gate: every
  dry-run parses strictly and predicted preset sizes are strictly increasing
  in the source-defined BPW order of each preset's dominant type.
- E4 Probe quantization gates: three FIT plans at preregistered positions,
  planned with the frozen original allocator, each quantized and verified
  byte-exact (output size == prediction, zero tolerance):
  - probe-low: IQ1_M-anchored envelope, --fit 0.5;
  - probe-mid: IQ4_XS-anchored pair to Q4_K_M... replaced by exact pair from
    the E3 ladder: the adjacent preset pair bracketing 2x IQ4_XS size,
    --fit 0.5 (near Q6_K region only if such a pair exists; otherwise the
    largest adjacent pair below Q8_0);
  - probe-top: the adjacent pair bracketing the largest gap below Q8_0,
    --fit 0.5.
  Gate: 3/3 zero-byte.
- E5 Regression gate: the full 53-test suite passes unchanged after the
  additive table edits; `git diff` shows no edits outside the trait/preset
  tables, their tests, and new P2 records.

Failure handling: any gate failure is recorded as-is; fixes are code changes
followed by a full re-run of E3-E5. No gate relaxation.

## Amendment 1 (2026-08-29, owner directive, before E2-E5 execution)

The owner directed reuse of `imatrix_unsloth.gguf` (the Huihui canonical
imatrix, SHA-256 `0ee5b10bd0c2fa2127c6f4b43dbfe1efd71e383b63217af9dade1de36599f1c1`)
instead of building a new imatrix for the orcarouter source. The in-progress
self-built APEX run was stopped mid-computation and its partial output
deleted; no E2/E3/E4 step had been evaluated. E2 is replaced by:

- E2': verify `imatrix_unsloth.gguf` by SHA-256, confirm all 496 profiled
  entries cover the quantizable matrices of the new BF16 (same qwen35
  architecture, 851 no-MTP tensors, identical tensor names and shapes), and
  record the provenance: the importance values derive from
  Huihui-Qwen3.8-27B-abliterated activations, not from this model's own
  activations. Quantized outputs will embed
  `quantize.imatrix.file = "imatrix_unsloth.gguf"` and dataset
  `unsloth_calibration_dataset`.

E1 stands as completed (BF16 SHA-256 recorded in `bf16-sha256.txt`). E3-E5
are unchanged and have not been executed.

## Amendment 2 (2026-08-29, before E3 execution)

E3 gate refinement (agreed with the owner):

- Payload-consistency check: for every preset, the predicted payload sum over
  the recipe's quantized tensors must match the dry-run's own displayed
  `quant size` total within display-rounding tolerance
  (`(n_quantized + 1) * 5243` bytes). This makes the sweep exercise the trait
  table of every type that appears in any preset recipe, not only the probe
  artifacts.
- Ladder monotonicity: predicted preset sizes are non-decreasing in the
  source-defined BPW order of each preset's dominant type, and strictly
  increasing between preset groups whose dominant types have different BPW
  (iq3_s and q3_k share 3.4375 BPW; ties inside a group may order either
  way).

E4 probe pairs are frozen from the E3 ladder immediately after E3 completes
and committed before any probe quantization:

- probe-low: (IQ1_M, IQ3_XXS), `--fit 0.5` - bottom-envelope types.
- probe-mid: the adjacent preset pair whose sizes bracket 1.25x the IQ4_XS
  size (fallback if no bracket exists: the pair immediately above IQ4_XS),
  `--fit 0.5` - upper-middle K-quant region.
- probe-top: the adjacent pair with the largest size gap ending at Q8_0,
  `--fit 0.5`.

## Amendment 3 (2026-08-29, after E2' first run, before E3 execution)

The initial E2' assertion required every quantized tensor to have an imatrix
entry and failed on `output.weight` and `token_embd.weight`. Source evidence:
the Huihui M2 dry-run (experiments/2026-08-28-m2-effective-recipes) shows the
identical assignments for this imatrix (output.weight -> q6_K,
token_embd.weight -> iq3_s) - the imatrix never carried entries for these two
matrices, the quantizer's fallback is deterministic, and every M2/M3/M9
zero-byte prediction was achieved under exactly this condition. The orcarouter
IQ3_M assignment is byte-identical to Huihui's on these tensors. E2' is
corrected to assert (a) imatrix SHA-256, (b) complete coverage of the 496
layer matrices, and (c) recorded fallback behavior for token_embd/output
(assignment oracle is the dry-run, per D-0005/M2). No evaluation result was
observed before this correction; E3 had not run.

## 3. Honest scope notes

- Size control per artifact remains self-proving (predict -> quantize ->
  byte check); the E4 probes extend that proof to both envelope extremes.
- Quality claims do not extend with the envelope: the monotonic-curve
  observation was recorded on IQ3_M..IQ4_XS only. IQ1/IQ2-region quality is
  expected to be rough and is not claimed here.
- IQ1/IQ2 presets require an imatrix; the pipeline always passes one.
- The allocator used for probes is the frozen original (v0.1a) purely for
  envelope validation; per-tier allocator choice for the release remains an
  open product decision.
