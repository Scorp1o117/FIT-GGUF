# FIT-GGUF

FIT-GGUF (Fit-to-Size Intelligent Tensor Quantization) is an experimental
continuous-size planning layer for GGUF quantization.

Given a source model, one canonical importance matrix, and a target byte
budget, FIT starts from the largest standard quantization preset that fits and
spends the remaining bytes on deterministic tensor precision upgrades, landing
at exactly the predicted size.

**v0.1 headline** (see `FINAL_REPORT.md`): exact deterministic size control
transfers across models (22/22 artifacts at zero-byte error, two
architectures); imatrix-driven allocation value is validated only on the
development model and is reported honestly as not yet generalizing.

## The fit CLI

```text
fit analyze --source SRC.gguf --imatrix IMX.gguf --runtime DIR --out-dir DIR
fit plan    --analysis DIR/analysis.json --fit 0.5 --policy original|balanced|random [--seed S] --out-prefix PREFIX
fit quantize --analysis DIR/analysis.json --tensor-types PREFIX-tensor-types.txt --out OUT.gguf --expect-bytes PREDICTED
```

- `analyze` runs pinned `llama-quantize --dry-run` for both presets, predicts
  exact output sizes (metadata, padding, and imatrix KVs included), profiles
  the imatrix, and freezes the upgrade candidate set into `analysis.json`.
  Every metadata constant is derived from the imatrix GGUF itself.
- `plan` picks the exact byte target (`--fit` uses exact integer fraction
  arithmetic), packs upgrades under the budget with a frozen policy, and
  writes `-plan.json`, `-recipe.json`, and `-tensor-types.txt` with SHA-256
  provenance.
- `quantize` executes `llama-quantize --imatrix --tensor-type-file` from the
  lower preset and verifies the output size against the prediction exactly.

Policies: `original` = v0.1a greedy imatrix utility (owns FIT-25 on the
development model), `balanced` = v0.1b block-balanced (confirmed at FIT-50 on
the development model), `random` = SHA-256-priority baseline.

Install: `pip install -e .` (or run `PYTHONPATH=src python -m fit_gguf.cli`).
Tests: `python3 -m pytest tests/`.

## Evidence

- `FINAL_REPORT.md` — the v0.1 final report with all validated and rejected
  claims.
- `experiments/` — preregistered experiment records M0-M16 and P1, each with
  frozen gates committed before execution.
- 53 unit tests cover the parser, size predictor, profiler, planner,
  candidates, optimizer, pipeline, and CLI.

## Project records

- `PROJECT_STATE.md` is the current source of truth.
- `DECISIONS.md` records accepted and rejected design decisions.
- `HANDOFF.md` is the current entry point for a new maintainer.

Large model, calibration, and prebuilt llama.cpp files are kept locally and
excluded from Git; their provenance is recorded in `docs/asset-inventory.md`.
