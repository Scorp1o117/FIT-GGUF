# P1: fit CLI and Regression Replay

Date: 2026-08-29

## Scope

Productization phase (engineering, not a hypothesis test): wrap the validated
M0-M16 pipeline into a user-facing three-command CLI and prove zero behavioral
drift by replaying two historical ground-truth points through it end to end.

The research program is frozen at M16 (D-0022). This phase changes no allocator,
threshold, or evaluation behavior. Its only allowed claim: the CLI reproduces
the exact plans and artifacts the experiment scripts produced.

## CLI contract (frozen before replay)

### `fit analyze`

```
fit analyze --source SRC.gguf --imatrix IMX.gguf --runtime DIR \
            [--lower IQ3_M] [--upper IQ4_XS] --out-dir DIR [--skip-hash]
```

Runs one `llama-quantize --dry-run --imatrix <arg>` per preset against the
source, parses both logs strictly, reads the source GGUF layout, derives the
quantization metadata from the imatrix GGUF itself (see derivation below),
loads the imatrix profile, generates the upgrade candidate set, and writes
`analysis.json`, `profile.json`, and both dry-run logs into the out-dir.

Metadata derivation replaces every hand-copied M12/M16 constant:

- `quantize.imatrix.file` = the imatrix path string exactly as passed on the
  command line, truncated to 127 characters (pinned quantize.cpp behavior).
- `quantize.imatrix.dataset` = first entry of the imatrix GGUF
  `imatrix.datasets` array, truncated to 127 characters; omitted when absent.
- `quantize.imatrix.entries_count` = number of sums/counts pairs.
- `quantize.imatrix.chunks_count` = the imatrix `imatrix.chunk_count` value;
  omitted when zero.
- `general.file_type` = per-preset constant (IQ3_M 27, IQ4_XS 30, from pinned
  include/llama.h). Only the KV's encoded size matters for prediction, not the
  value.

`analysis.json` records source/imatrix paths, sizes, SHA-256 hashes (unless
`--skip-hash`), predicted preset sizes, per-preset qtype histograms, the full
candidate and rejected lists, the derived metadata, and
`block_span_auto = max_block // 4 + 1` from the profile.

### `fit plan`

```
fit plan --analysis DIR/analysis.json (--fit F | --target-bytes N) \
         (--policy original | --policy balanced | --policy random --seed S) \
         [--block-span auto | N] --out-prefix PREFIX
```

- Target: `lower + Fraction(F) numerator arithmetic` on the predicted preset
  gap (`--fit 0.5` means `lower + gap // 2`, reproducing the M9/M16 integer
  semantics exactly); or an explicit byte target.
- Policies map to the frozen M16 implementations: `original` =
  `optimize_greedy`, `balanced` = `optimize_block_balanced` (v0.1b,
  `--block-span auto` default), `random` = `optimize_random`.
- Rebuilds the lower-preset recipe with overrides applied, predicts the exact
  output size, fails if the prediction exceeds the target, and writes
  `PREFIX-plan.json`, `PREFIX-recipe.json`, and `PREFIX-tensor-types.txt`.

### `fit quantize`

```
fit quantize --analysis DIR/analysis.json --tensor-types FILE \
             --out OUT.gguf [--expect-bytes N]
```

Runs `llama-quantize --imatrix <arg> --tensor-type-file FILE SRC OUT LOWER`
(flags before positionals), verifies the output size against the prediction
(and `--expect-bytes` when given), and writes `OUT.quantize-record.json` with
size and SHA-256.

## Replay gates (frozen acceptance; no tolerance, no post-hoc adjustment)

Any gate failure is a replay failure. Fixes may only be code changes, followed
by a full replay re-run of all gates.

- G1 Huihui analyze: source SHA-256
  `8a033407c8f58d43102aade25b973cc6d2f2ce5c5cbf4dc75a2cdb60b9e33cbc`;
  predicted IQ3_M = 12,580,875,232 and IQ4_XS = 15,082,507,232; profile
  entries = 496; derived provenance equals the hand META
  (`file="imatrix_unsloth.gguf"`, `dataset="unsloth_calibration_dataset"`,
  `entries_count=496`, `chunks_count=1251`).
- G2 Huihui plan original at `--fit 0.5`: tensor-types file byte-identical to
  `experiments/2026-08-28-m7-greedy/tensor-types-FIT50.txt`
  (SHA-256 `d7e662980a5a28f4a586f06cb78f11b313a1439b89303d69e9f183f5e96a238c`);
  target = 13,831,691,232; predicted = 13,831,486,432; unused = 204,800.
- G3 Huihui plan balanced at `--fit 0.5`: byte-identical to
  `experiments/2026-08-28-m10-ablation/block-balanced-fit50-tensor-types.txt`
  (SHA-256 `0f1e30a0d63ff726370c5443014c35a89002603a2112b16fc8ad669d6fcaba02`);
  predicted = 13,828,987,872.
- G4 Huihui quantize: original artifact SHA-256
  `e4fe1c46ab89c8b6343203168ebeec699372c2fb21f411c61c47edc2e1f33306` (M9);
  balanced artifact SHA-256
  `7cfa1b91600115c046cb9afcae8347adc7de5a77b0d880b47a956cb8a4799a07` (M10).
- G5 Granite analyze: predicted IQ3_M = 4,089,184,640 and IQ4_XS =
  4,820,287,872; profile entries = 280; derived provenance equals the M16 META
  (`file="imatrix-granite-apex-c512.gguf"`,
  `dataset="/run/media/s117/OS/Models/imatrix-calibration/APEX-imatrix-Small.txt"`,
  `entries_count=280`, `chunks_count=1250`); imatrix SHA-256
  `5488dbe0391dd8e54b1404cc14d805bd92ea2bfe09eb14be5794bbd0894ce18e`.
- G6 Granite plan original at `--fit 0.5`: byte-identical to
  `experiments/2026-08-29-m16-granite-reveal/o-fit50-tensor-types.txt`
  (SHA-256 `f5e8781793c0641256c2e68bcb6cfb8377c031aba6fda6683f20f5da1007ccff`);
  target = 4,454,736,256; predicted = 4,454,351,232.
- G7 Granite plan balanced at `--fit 0.5`: byte-identical to
  `experiments/2026-08-29-m16-granite-reveal/b-fit50-tensor-types.txt`
  (SHA-256 `1ddfa270f824b8f73bcb77d5df8befb19c2a5a8eafb20a38381192bfd4909a22`);
  predicted = 4,454,564,224.
- G8 Granite quantize: original artifact SHA-256
  `09ca3d8505ed728b1c1a2202d7a9793268f66d3ee8ad5001f5b76993f1d1023e`; balanced
  artifact SHA-256
  `17660767b5967a08112c27f04ad95b67278e9c372afde2cbd6e37ba81c123e21`
  (M16 artifact-hashes.txt).

Imatrix SHA-256 provenance for Huihui:
`0ee5b10bd0c2fa2127c6f4b43dbfe1efd71e383b63217af9dade1de36599f1c1`.

## Non-goals

No quality evaluation (M9-M16 already recorded it), no allocator or threshold
changes, no new model, no tuning on either model.

## Amendment (2026-08-29, before any replay execution)

A pre-run derivation check showed the Granite imatrix carries
`imatrix.chunk_count = 3394`, while the M16 hand META recorded
`chunks_count = 1250`. quantize.cpp writes chunks_count as a 4-byte integer
KV, so its value cannot affect size prediction; only the KV's presence does.
G5 is corrected to require the derived provenance to match the hand META on
the size-relevant fields (file and dataset strings, exactly) and to report
`entries_count = 280` with `chunks_count > 0` present. The observed
3394-vs-1250 value difference is recorded as a provenance note in the replay
results. No size gate is relaxed. The Huihui derivation matches the hand META
exactly (496 / 1251 / `unsloth_calibration_dataset`, block span 16).

