# Local Asset Inventory

Verified on 2026-08-28. Paths are relative to the repository root unless noted.

## Canonical imatrix

- Path: `imatrix_unsloth.gguf`
- Size: 13,642,656 bytes
- SHA-256: `0ee5b10bd0c2fa2127c6f4b43dbfe1efd71e383b63217af9dade1de36599f1c1`

The extension does not establish that this is a normal model GGUF; it is treated
as an opaque llama.cpp imatrix input until M1 verifies the file format.

## Development model

- Path: `test-Models/Huihui-Qwen3.8-27B-abliterated/`
- 18 safetensors shards
- Total shard bytes: 55,563,006,216 (51.747 GiB)
- `config.json` SHA-256:
  `191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab`
- `model.safetensors.index.json` SHA-256:
  `77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df`

`VERSION-NOTES.md` states that these weights use the newer layers-18-through-51
abliteration layout. Preserved PRISM artifacts refer to an older weight version
unless individually proven otherwise.

## Reference logits and prior plans

- Seven `reference-logits*.bin` files total 42,300,194,332 bytes (39.395 GiB).
- Three `.ttf` files are ASCII tensor-type assignment tables, not font files.
- JSON plans and five-domain logs are preserved as prior experimental evidence,
  not accepted as FIT baselines.

## Calibration dataset

- Path: `test-Models/imatrix-calibration/`
- 299 Parquet files
- Nested Git commit: `e87ed55dcba9d9c3a3e41539f3e728e981b1daa4`
- Remote: `https://huggingface.co/datasets/eaddario/imatrix-calibration`

The nested worktree contains hydrated LFS content and local additions. FIT must
read it without changing or normalizing that repository.

## llama.cpp runtime

- Path: `tools/llama-b10666-rocm/`
- Previous path:
  `/home/s117/llama.cpp-hub/llama.cpp-hub-v0.9.5.3-b9837-linux-rocm-7.2/llamacpp/llama-b10666-bin-ubuntu-rocm-10.0-x64/`
- Version: `0.3.0-dev`, build 10666, commit `4e97ac86e`
- Compiler: GNU 15.2.0, Linux x86_64
- `llama-quantize` SHA-256:
  `66e85cd13cfb2a7a1c864c3316e05c4c20fa137008790332f2bbd089862cc042`

The runtime is a local binary dependency and is excluded from Git. It includes
ROCm/HIP libraries and the tools required for conversion-adjacent inspection,
quantization, imatrix generation, perplexity, GGUF inspection, and tests.

## Matching llama.cpp source

- Path: `third_party/llama.cpp/`
- Commit: `4e97ac86ebe2c4cb8212d98d2641ad6768810896`
- State: clean detached checkout from `https://github.com/ggml-org/llama.cpp.git`

The source checkout is excluded from the FIT repository and will remain
read-only during M1.

## Cleanup record

- Removed a 12 KiB model download cache.
- Removed `test-Models/imatrix_unsloth.gguf` after byte comparison and SHA-256
  verification proved it was an independent duplicate of the canonical root
  imatrix.
