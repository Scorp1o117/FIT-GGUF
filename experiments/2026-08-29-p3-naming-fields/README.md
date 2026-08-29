# P3: Naming Fields in fit plan (DominantQType + Suggested Filename)

Date: 2026-08-29

## Scope

Owner-directed CLI increment implementing the frozen release naming
convention. Purely additive output; no planner, allocator, size, or recipe
behavior changes. MTP stays excluded from the release source (owner decision:
"保留移除状态").

## Feature specification (frozen before implementation)

`fit plan` record gains four fields, derived only from the final applied
recipe and the analysis:

- `qtype_parameter_shares`: distribution of the destination qtypes over the
  recipe's QUANTIZED tensors, weighted by parameter element count
  (`sum(prod(shape))` per tensor, grouped by lowercase qtype). Unchanged
  (non-quantized) tensors are excluded: the label describes the quantized
  parameters. Shares are exact float fractions.
- `dominant_qtype`: the qtype with the largest parameter element count;
  ties break alphabetically for determinism. Element weighting is the
  convention's explicit requirement: tensor COUNT must not be used (the
  motivating trap is 100 small IQ4_XS tensors versus 30 large IQ3_M tensors).
- `model_name`: from the analysis source file name with the `.gguf` suffix
  and a trailing `-BF16` marker removed, unless overridden by a new
  `--model-name` plan option.
- `suggested_filename`: `{model_name}-FIT-{G}G-{DOMINANT_QTYPE}.gguf` where
  `{G}` is the target size in GiB (1024^3) rounded half-up to the nearest
  integer (minimum 1) and the qtype is the canonical uppercase form
  (`IQ4_XS`, never `IQ4XS`). The promise encoded in the name is the target,
  matching the v0.1 positioning (file size ≈ target).

The CLI prints the dominant qtype with its share and the suggested filename.
`write_fit_recipe` and `write_tensor_type_file` are untouched.

## Acceptance gates (frozen before execution)

- A1 unit tests: element-weighted dominance defeats tensor-count intuition
  (constructed case); shares sum to 1; GiB rounding is half-up; the `-BF16`
  strip and `--model-name` override behave as specified; uppercase qtype
  formatting.
- A2 zero-drift replay: re-running `fit plan` on each of the three committed
  P2 probe analyses must reproduce the committed plan's
  `recipe_sha256`, `tensor_types_sha256`, `predicted_size_bytes`, and
  `target_bytes` exactly; the new record differs only by the added fields.
  Recorded in `replay-results.json`.
- A3 CLI output shows the dominant qtype line and suggested filename.

Failure handling: any gate failure is recorded as-is; fixes are code changes
followed by a full re-run of the affected gates.

## Amendment 1 (2026-08-29, owner-directed 0.5-GiB tier grid, before P4)

The release tier grid is every 0.5 GiB from 7G to 12G. Integer-GiB rounding
would collide half-GiB tiers (7.5G and 8G would both render "8G"). The
suggested-filename rule is extended: exact half-GiB targets render with one
decimal (`7.5G`), every other target keeps integer rounding (`7G`, `9G` from
8.76 GiB, so the recorded P2/P3 replay filenames stay valid). "G" in FIT
filenames means GiB (1024^3 bytes); the model card states this explicitly.
Tier targets are exact byte targets (`--target-bytes`), never rounded.

## Results (2026-08-29)

- A1: 4 new unit tests pass (element-weighted dominance defeats tensor-count
  intuition; shares sum to 1; GiB half-up rounding; -BF16 strip and
  --model-name override). Suite total 59/59.
- A2: zero-drift replay on all three P2 probe analyses
  (`replay-results.json`): recipe_sha256, tensor_types_sha256,
  predicted_size_bytes, target_bytes, and selected_count are byte-identical
  to the committed P2 plans; only the additive naming fields differ.
  Observed suggested filenames: FIT-9G-IQ3_XXS (probe-low, dominant is the
  upgraded FFN mass - element weighting working as specified),
  FIT-18G-Q5_K (87.6% of quantized parameters), FIT-24G-Q6_K.
- A3: CLI prints the dominant qtype with its share and the suggested
  filename.
