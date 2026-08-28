# M3 Exact Size Predictor Validation

Date: 2026-08-28

## Result

The pure-Python predictor matches four complete 27B GGUF artifacts with zero
byte error: three standard presets and one targeted tensor override.

## Algorithm

For each tensor with destination qtype `t`:

```text
row_bytes     = ne0 / block_size[t] * type_size[t]
payload_bytes = row_bytes * ne1 * ne2 * ne3
padded_bytes  = align(payload_bytes, 32)
```

The final single-file size is:

```text
output_metadata_bytes + sum(padded_tensor_bytes)
```

`output_metadata_bytes` is reconstructed from the source GGUF metadata and
tensor descriptors. The predictor applies the same build-10666 transformations
used here: remove split keys; set fixed-width quantization version and file type;
and add the four imatrix provenance keys.

The implementation is in `src/fit_gguf/gguf.py`. It reads GGUF v3 metadata and
tensor descriptors using only the Python standard library and never reads the
large tensor-data section.

## Pinned source evidence

- `ggml/src/ggml.c:1329-1343` defines block size, type size, and exact row size.
- `ggml/src/ggml.c:783-874` maps the M2 qtypes to pinned block structures.
- `ggml/src/ggml-common.h:89-90,326-367,411-460` fixes block element counts and
  encoded structure sizes.
- `src/llama-quant.cpp:1189-1206` computes dry-run payloads from row size and
  row count.
- `src/llama-quant.cpp:1313-1318` pads every real tensor to GGUF alignment.
- `src/llama-quant.cpp:963-978` copies/transforms output metadata and removes
  split keys.
- `tools/quantize/quantize.cpp:511-541` adds imatrix file, dataset, entry-count,
  and chunk-count metadata.
- `ggml/src/gguf.cpp:1628-1646,1695-1705` serializes and sizes output metadata.

## Validation

Common source and imatrix provenance are identical to the M2 experiment.

| Recipe | Exact tensor payload | Metadata | Predicted | Actual | Error |
| --- | ---: | ---: | ---: | ---: | ---: |
| IQ3_S | 12,408,334,336 | 10,994,656 | 12,419,328,992 | 12,419,328,992 | 0 |
| IQ3_M | 12,569,880,576 | 10,994,656 | 12,580,875,232 | 12,580,875,232 | 0 |
| IQ4_XS | 15,071,512,576 | 10,994,656 | 15,082,507,232 | 15,082,507,232 | 0 |
| IQ3_S + `blk.0.ffn_down.weight=Q5_K` | 12,431,312,896 | 10,994,656 | 12,442,307,552 | 12,442,307,552 | 0 |

The override artifact SHA-256 is
`0b88b79a96d16c3324f6cdf80cdbbef193ba5766d78dfa86b2fc5b173785a758`.
Its real quantization took 287.040 seconds and its 851 output tensor qtypes had
zero mismatches against the effective dry-run recipe.

The dry-run summary cannot supply these exact payloads because its per-tensor
and aggregate MiB values are display-rounded. Relative to exact payloads, the
three displayed aggregate conversions differed by +246, -2,376, and -4,669
bytes respectively.

## Review note

The Anti-Gravity M3 read-only investigation correctly identified the row-size
and padding formula, but assumed output metadata stayed identical to the source
and therefore cited the source data offset of 10,994,432 bytes. Real artifacts
contain four additional `quantize.imatrix.*` fields and use a 10,994,656-byte
metadata section. The implemented predictor includes this 224-byte change.

## Scope

M3 is accepted for pinned build 10666, GGUF v3, single-file output, no layer
pruning, no split retention, and the supported qtypes in the implementation.
Arbitrary `--override-kv`, sharded output, new qtypes, or a llama.cpp revision
change require explicit support and revalidation.
