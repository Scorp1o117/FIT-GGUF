# llama.cpp Integration at build 10666

This document records verified behavior at llama.cpp commit
`4e97ac86ebe2c4cb8212d98d2641ad6768810896`. The initial investigation was
delegated read-only through Anti-Gravity CLI using Gemini 3.7 Flash High and was
then checked directly against the pinned source and matching binary runtime.

## Executive finding

The supplied runtime already contains the two interfaces FIT needs most:

- `llama-quantize --dry-run` executes llama.cpp's real effective tensor-type
  selection and prints each tensor's resolved target type and payload size.
- `--tensor-type` and `--tensor-type-file` apply ordered regex overrides before
  the final shape-compatibility fallback.

M2 therefore does not need to reproduce preset logic or patch the quantizer.
FIT can parse a deterministic dry-run dump. M3 cannot treat the final dry-run
total as complete file size because it excludes GGUF metadata and alignment.

## Imatrix loading

The CLI parses `--imatrix`, optional include/exclude substring filters, and
tensor overrides in
[`tools/quantize/quantize.cpp`](../third_party/llama.cpp/tools/quantize/quantize.cpp#L395).
`prepare_imatrix()` calls `load_imatrix()` and normalizes the loaded values
before constructing the null-terminated `llama_model_imatrix_data` array passed
through `llama_model_quantize_params` (lines 183-303 and 496-540).

`common_imatrix_load()` in
[`common/imatrix-loader.cpp`](../third_party/llama.cpp/common/imatrix-loader.cpp#L82)
first attempts GGUF and falls back to the legacy binary stream format only when
GGUF initialization fails.

The GGUF imatrix schema is:

- `imatrix.datasets`: string array;
- `imatrix.chunk_count`: `uint32`;
- `imatrix.chunk_size`: `uint32`;
- `<tensor>.in_sum2`: F32 activation-square sums;
- `<tensor>.counts`: F32 per-expert counts.

The loader requires paired sums/count tensors and F32 types (lines 118-168).
The quantize CLI divides each expert slice by its corresponding count; a zero
count produces importance values of 1 (quantize.cpp lines 195-245).

The canonical local imatrix was verified as GGUF v3 with:

```text
general.type: imatrix
imatrix.datasets: [unsloth_calibration_dataset]
imatrix.chunk_count: 1251
imatrix.chunk_size: 8192
tensor_count: 992
alignment: 32
```

During quantization, `llama_model_quantize_impl()` validates all imatrix floats
as finite, looks up the possibly layer-remapped tensor name, and requires the
entry length to equal `ne[0] * ne[2]`. Non-token-embedding mismatches abort.
Expert slices receive separate imatrix offsets (llama-quant.cpp lines 949-956,
1076-1091, and 1222-1305).

Important distinction: IQ3_S, IQ3_M, and IQ4_XS do not intrinsically require an
imatrix in `tensor_requires_imatrix()`. FIT still always supplies the canonical
imatrix because it is the ranking input and because the presence of an imatrix
changes some standard-preset mixture decisions, including IQ4_XS FFN-down
handling.

## Effective preset selection

All effective choices flow through
[`llama_tensor_get_type()`](../third_party/llama.cpp/src/llama-quant.cpp#L683):

1. Reject tensors that are not quantizable.
2. Apply dedicated token-embedding or output flags if set.
3. Apply the first matching manual tensor regex.
4. If no manual match and `--pure` is not set, execute
   `llama_tensor_get_type_impl()`.
5. Apply `tensor_type_fallback()` for incompatible first dimensions.

The default qtypes are defined by `llama_ftype_get_default_type()` (lines
828-871):

| Preset | Default tensor qtype |
| --- | --- |
| IQ3_S | IQ3_S |
| IQ3_M | IQ3_S |
| IQ4_XS | IQ4_XS |

IQ3_M is therefore a mixture recipe, not an all-IQ3_M tensor type.

Key mixture rules at this pinned commit include:

- Output or tied token embeddings normally become Q6_K for these three
  presets, with Q8_0 used for Falcon or incompatible dimensions.
- IQ3_M upgrades attention-V-like tensors and attention output to Q4_K.
- IQ3_M upgrades selected FFN-down layers to Q4_K.
- IQ3_S upgrades attention-V-like tensors to Q4_K when GQA is at least 4.
- IQ4_XS upgrades attention-V-like tensors to Q5_K when GQA is at least 4.
- Eight-expert architectures receive additional Q5_K/Q8_0 special cases.
- 70B attention-V-like tensors receive a Q5_K special case.

The complete role logic is in
[`llama_tensor_get_type_impl()`](../third_party/llama.cpp/src/llama-quant.cpp#L428).
FIT must never encode this matrix independently.

For the development Qwen3.5 model, `ssm_alpha`, `ssm_beta`, and `ssm_out` fall
into the generic `OTHER` category and normally retain the preset default;
`ssm_conv1d` is explicitly excluded from quantization. Attention and FFN
tensors use the broad role rules above. Actual assignments will be captured
from dry-run rather than inferred from names.

`tensor_type_fallback()` (lines 372-425) resolves incompatible shapes after all
preset or manual decisions. IQ3_S and IQ4_XS fall back to IQ4_NL; Q4_K to Q5_0;
Q5_K to Q5_1; Q6_K to Q8_0. If the first dimension is still not divisible by
the 32-element fallback block, it uses F16.

## Tensor override interface

`parse_tensor_type()` and `parse_tensor_type_file()` are in
[`tools/quantize/quantize.cpp`](../third_party/llama.cpp/tools/quantize/quantize.cpp#L305).

- Syntax is `<ECMAScript-regex>=<ggml_type>`.
- Type matching is case-insensitive.
- The regex text is lowercased by the CLI.
- A file is whitespace-tokenized, so each rule must contain no whitespace.
- Regexes are compiled once in `quantize_state_impl`.
- Matching uses `std::regex_search`, not a required full-string match.
- Rules have first-match precedence.
- A manual match bypasses preset mixture logic for that tensor.
- Final shape fallback still applies.
- Dedicated `--token-embedding-type` and `--output-tensor-type` can preempt
  ordinary tensor rules.

FIT recipes should emit escaped, full-name regexes such as
`^blk\.7\.ffn_down\.weight$` and order specific rules before any broad rules.
The initial recipe implementation should emit only exact tensor names to avoid
regex collisions.

Base-preset plus upgrades already works without a llama.cpp patch:

```bash
llama-quantize \
  --imatrix imatrix_unsloth.gguf \
  --tensor-type-file fit-overrides.txt \
  source-bf16.gguf output-fit.gguf IQ3_S
```

## Effective recipe dump (M2)

The minimum implementation is to invoke:

```bash
llama-quantize --dry-run --imatrix IMATRIX SOURCE_GGUF PRESET
```

The dry-run path computes cached metadata and target types using the same
`tensor_allows_quantization()`, `llama_tensor_get_type()`, and fallback logic as
real quantization. It prints tensor name, shape, original type/size, resolved
type, and resolved payload size (llama-quant.cpp lines 1030-1093 and
1174-1206). It writes no output GGUF.

M2 should parse this output into a versioned, deterministically sorted record.
The parser must fail loudly when a tensor line cannot be parsed or when the
subprocess exits nonzero.

The staging API in `src/llama-ext.h` and the existing
`tests/test-quant-type-selection.cpp` provide an alternative in-process route,
but dry-run is preferable initially because it operates on the exact source
GGUF metadata and tensor list.

## Encoded size and M3

For target qtype `t`, llama.cpp calculates tensor payload bytes as:

```text
rows = ne[1] * ne[2] * ne[3]
row_bytes = ne[0] / block_size(t) * type_size(t)
payload_bytes = rows * row_bytes
```

Dry-run sums only these unpadded payload bytes. It does not include:

- the GGUF header, KV metadata, or tensor descriptors;
- metadata-to-data alignment (already included in `gguf_get_meta_size()` when
  using llama.cpp's writer);
- per-tensor padding to the GGUF alignment;
- repeated metadata when retaining split shards.

For one output file, the source-verified physical-size model is:

```text
gguf_get_meta_size(output_context)
+ sum(GGML_PAD(payload_bytes_i, output_alignment))
```

`gguf_get_meta_size()` includes the padding between metadata and tensor data
([`ggml/src/gguf.cpp`](../third_party/llama.cpp/ggml/src/gguf.cpp#L1628)). The
quantizer writes padding after every tensor (llama-quant.cpp lines 1313-1318).

M3 must therefore augment the dry-run tensor results with exact output metadata
size and alignment, then validate against real files. The printed dry-run total
is a useful payload oracle, not the final GGUF-size oracle.

## Risks and open checks

- No source BF16/F16 GGUF is present yet, so effective assignments for this
  exact Qwen3.5 weight set have not been dumped.
- The current imatrix covers 496 logical tensor names (992 sum/count tensors),
  but exact name/shape alignment with the newly converted source GGUF remains a
  mandatory gate.
- Manual overrides bypass preset-specific sensitivity rules; FIT must only emit
  upgrades from the already captured effective lower recipe.
- Override regex first-match semantics make stable ordering part of the recipe
  contract.
- `--keep-split` requires per-shard metadata accounting and is deferred until
  the single-file size predictor is exact.
- Existing PRISM plans and reference logits may target older weights and cannot
  establish FIT quality without provenance alignment.
