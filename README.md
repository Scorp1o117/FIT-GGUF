# FIT-GGUF

FIT-GGUF (Fit-to-Size Intelligent Tensor Quantization) is an experimental
continuous-size planning layer for GGUF quantization.

Given a source model, one canonical importance matrix, and a target byte
budget, FIT aims to start from the largest standard quantization preset that
fits and spend the remaining bytes on deterministic tensor precision upgrades.

The project has validated effective-recipe capture and exact byte-size
prediction against four real 27B quantizations on pinned llama.cpp build 10666.
The canonical imatrix is fully profiled, exact baseline planning and
recipe-driven quantization are implemented, and six arbitrary-size artifacts
including the formal FIT-25/FIT-50/FIT-75 curve points match their predictions
exactly. The formal five-point curve has also been evaluated against freshly
generated current-weight BF16 logits across five domains; KL improves
monotonically in every domain.

## Local development assets

Large model, calibration, reference-logit, and prebuilt llama.cpp files are
kept locally and intentionally excluded from Git. Their verified inventory and
provenance are recorded in `docs/asset-inventory.md`.

## Planned workflow

```text
fit analyze MODEL --imatrix IMATRIX
fit plan --profile PROFILE --target-size SIZE
fit quantize MODEL --profile PROFILE --target-size SIZE
```

The CLI above is a target interface, not yet an implemented contract.

## Project records

- `PROJECT_STATE.md` is the current source of truth.
- `DECISIONS.md` records accepted and rejected design decisions.
- `experiments/` will contain reproducible experiment records.
- `docs/` contains integration and algorithm documentation.
