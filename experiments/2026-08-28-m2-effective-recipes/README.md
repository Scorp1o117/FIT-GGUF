# M2 Effective Recipe Validation

Date: 2026-08-28

## Question

Can pinned llama.cpp build 10666 expose the effective tensor qtype recipe for
IQ3_S, IQ3_M, and IQ4_XS, and does that recipe match the tensors written by
real quantization?

## Provenance

- Runtime: llama.cpp build 10666, commit `4e97ac86e`.
- Matching source: commit
  `4e97ac86ebe2c4cb8212d98d2641ad6768810896`.
- Source HF model: `Huihui-Qwen3.8-27B-abliterated`.
- Development GGUF: BF16, converted with the matching source converter and
  `--no-nextn`.
- BF16 GGUF SHA-256:
  `8a033407c8f58d43102aade25b973cc6d2f2ce5c5cbf4dc75a2cdb60b9e33cbc`.
- BF16 GGUF size: 53,808,282,368 bytes.
- BF16 GGUF structure: GGUF v3, alignment 32, 45 metadata entries, 851
  tensors, data offset 10,994,432.
- Canonical imatrix SHA-256:
  `0ee5b10bd0c2fa2127c6f4b43dbfe1efd71e383b63217af9dade1de36599f1c1`.
- Imatrix coverage: 496 matrix names, represented by 992 `in_sum2`/`counts`
  tensors, over 1,251 calibration chunks.

The one-layer NextN/MTP head was excluded because it is an auxiliary
speculative head, is not covered by the canonical imatrix, and is not part of
the ordinary next-token quality target. It can be exported and evaluated as a
separate artifact later.

## Commands

The source conversion was:

```text
python3 third_party/llama.cpp/convert_hf_to_gguf.py \
  test-Models/Huihui-Qwen3.8-27B-abliterated \
  --outtype bf16 \
  --outfile artifacts/source/Huihui-Qwen3.8-27B-abliterated-BF16.gguf \
  --no-nextn
```

Each preset was first inspected and then actually quantized:

```text
tools/llama-b10666-rocm/llama-quantize \
  --dry-run --imatrix imatrix_unsloth.gguf SOURCE.gguf PRESET

tools/llama-b10666-rocm/llama-quantize \
  --imatrix imatrix_unsloth.gguf SOURCE.gguf OUTPUT.gguf PRESET
```

Full local logs are preserved under `artifacts/logs/` and intentionally
excluded from Git.

## Results

| Preset | Dry payload bytes | Actual file bytes | File minus displayed payload | BPW | Real time | SHA-256 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| IQ3_S | 12,408,334,582 | 12,419,328,992 | 10,994,410 | 3.69 | 278.373 s | `e2cfeab212781d5530e3988ded332765cbf244cf3302c8547e111b5c3b646b10` |
| IQ3_M | 12,569,878,200 | 12,580,875,232 | 10,997,032 | 3.74 | 279.163 s | `a569f12c5f63af0c47e55c97f93bf53add96d2c059ba1b143b058b94f9dd847a` |
| IQ4_XS | 15,071,507,907 | 15,082,507,232 | 10,999,325 | 4.48 | 279.138 s | `262f8714ca315c3bda278aeb39993394e6c3fcaccde7e4fd9e2accf67ac257b3` |

All three dry runs contained 851 unique, complete tensor assignments: 498
converted tensors and 353 unchanged F32 tensors.

| Preset | Effective destination-type counts |
| --- | --- |
| IQ3_S | 353 F32, 433 IQ3_S, 64 Q4_K, 1 Q6_K |
| IQ3_M | 353 F32, 409 IQ3_S, 88 Q4_K, 1 Q6_K |
| IQ4_XS | 353 F32, 433 IQ4_XS, 64 Q5_K, 1 Q6_K |

For every preset, the actual output GGUF had exactly the same 851 tensor names
as the dry run and zero destination-qtype mismatches.

## Conclusion

M2 passes for the pinned runtime and current 27B source model. The strict
dry-run parser exposes the effective preset recipe, and three complete real
quantizations confirm that every reported destination qtype is written as
reported.

The displayed dry-run total is not an exact file-size oracle. Its MiB values
are rounded and it excludes output metadata/alignment. M3 must compute exact
encoded row sizes plus the GGUF metadata and per-tensor padding rather than
learning a constant correction from these three files.
