# M4 Canonical Imatrix Profile

Date: 2026-08-28

## Provenance

- Input: `imatrix_unsloth.gguf`
- Input SHA-256:
  `0ee5b10bd0c2fa2127c6f4b43dbfe1efd71e383b63217af9dade1de36599f1c1`
- Dataset: `unsloth_calibration_dataset`
- Chunks: 1,251
- Chunk size: 8,192
- Raw GGUF tensors: 992, comprising 496 matched `in_sum2`/`counts`
  pairs
- Generated profile: `profile.json`
- Profile schema: 1
- Profile SHA-256:
  `6a5afeca40b54dd4a170f8ef0c4a29161be6f33d6030bc44c57b2fc91774ca55`

## Normalization

The profiler reproduces the pinned llama.cpp GGUF imatrix semantics in
`tools/quantize/quantize.cpp:183-242`: each `in_sum2` segment is divided by its
corresponding rounded positive count; a non-positive count falls back to ones.

For each tensor it records normalized mean, RMS, standard deviation, min,
P50/P95/P99/max, nonzero fraction, counts, block, and role. Tensor means also
receive global and within-role median ratios and percentile ranks.

The scalar tensor mean is a planning feature, not a standalone quality oracle.
The full channel vector still controls llama.cpp's actual importance-aware
quantization.

## Coverage

The IQ3_S effective recipe contains 498 converted tensors. Excluding the
unprofiled token embedding and output head leaves 496 layer matrices. The
profile name set matches those 496 matrices exactly: zero missing and zero
extra.

All 496 entries have one count value equal to 1,149,614.

| Role | Count | Minimum mean | Median mean | Maximum mean |
| --- | ---: | ---: | ---: | ---: |
| attn_gate | 48 | 0.131423 | 0.795993 | 1.355680 |
| attn_k | 16 | 0.668654 | 1.008985 | 1.693357 |
| attn_output | 16 | 0.000343 | 0.038786 | 0.932923 |
| attn_q | 16 | 0.668654 | 1.008985 | 1.693357 |
| attn_qkv | 48 | 0.131423 | 0.795993 | 1.355680 |
| attn_v | 16 | 0.668654 | 1.008985 | 1.693357 |
| ffn_down | 64 | 0.000090 | 0.013424 | 2.450404 |
| ffn_gate | 64 | 0.008654 | 0.391116 | 1.532337 |
| ffn_up | 64 | 0.008654 | 0.391116 | 1.532337 |
| ssm_alpha | 48 | 0.131423 | 0.795993 | 1.355680 |
| ssm_beta | 48 | 0.131423 | 0.795993 | 1.355680 |
| ssm_out | 48 | 0.000179 | 0.013244 | 0.803229 |

## Diagnostics

Highest tensor means include `blk.63.ffn_down.weight` (2.450404), the block-3
full-attention Q/K/V inputs (1.693357), and block-61 FFN gate/up inputs
(1.532337). The lowest include early FFN-down and SSM-output tensors.

The largest profiled tensors are the 178,257,920-byte BF16 FFN down/gate/up
matrices. Size alone and importance alone therefore disagree in several early
blocks, which is precisely the tradeoff the later utility-per-byte planner must
make explicit.

Several tensors share identical activation statistics because they consume the
same layer input: full-attention Q/K/V, recurrent attention gate/QKV/SSM
alpha/beta, and FFN gate/up pairs. These are valid separate upgrade candidates,
but their identical imatrix evidence must not be treated as independent
corroborating samples.

## Acceptance

M4 passes for the canonical GGUF imatrix. The profile is deterministic,
versioned, fully name-aligned with the development model, and covered by unit
tests. M5 may use predicted exact preset sizes; M6 must keep role normalization
and shared-input structure visible when defining candidate utility.
