# FIT-GGUF Handoff

Date: 2026-08-28 (updated after M11/M12 by the GLM-5.3-Flash session)

## Read this first

The repository is past M12. M0-M12 are complete. Do not restart source
conversion, imatrix profiling, preset quantization, the five-domain M9 curve,
or the M11 holdout validation. The current open question is the
budget-dependent allocation trade-off recorded in D-0018: the frozen
block-balanced v0.1b allocator is confirmed at FIT-50 on untouched holdout
data and wins at FIT-75 on the design domains, while the original utility
remains better at FIT-25.

Canonical status and evidence:

- `PROJECT_STATE.md` - concise source of truth;
- `DECISIONS.md` - decisions D-0001 through D-0018;
- `experiments/2026-08-28-m9-fit-curve/README.md` - accepted formal curve;
- `experiments/2026-08-28-m10-ablation/README.md` - random and block diagnosis;
- `experiments/2026-08-28-m11-holdout/README.md` - preregistered holdout
  confirmation of v0.1b at FIT-50 (gates A/B/C all passed);
- `experiments/2026-08-28-m12-block-balanced-curve/README.md` - v0.1b curve
  extension showing the budget-dependent trade-off.

## Environment and provenance

- Workspace: `/run/media/s117/OS/FIT-GGUF`
- Runtime: `tools/llama-b10666-rocm`
- llama.cpp: build 10666, commit `4e97ac86e`
- Matching source: `third_party/llama.cpp`, detached commit
  `4e97ac86ebe2c4cb8212d98d2641ad6768810896`
- BF16 source: `artifacts/source/Huihui-Qwen3.8-27B-abliterated-BF16.gguf`
- BF16 SHA-256:
  `8a033407c8f58d43102aade25b973cc6d2f2ce5c5cbf4dc75a2cdb60b9e33cbc`
- Canonical imatrix: `imatrix_unsloth.gguf`
- Imatrix SHA-256:
  `0ee5b10bd0c2fa2127c6f4b43dbfe1efd71e383b63217af9dade1de36599f1c1`
- Development conversion excludes the auxiliary NextN/MTP head.
- Do not reuse the older preserved reference logits; current-weight references
  are under `experiments/2026-08-28-m9-fit-curve/artifacts/reference-logits/`
  (original slices) and
  `experiments/2026-08-28-m11-holdout/artifacts/reference-logits/` (holdout
  slices).

Evaluation parameters are fixed to:

```text
llama-perplexity -ngl 99 -t 16 -c 512 -b 512
```

Original five eval slices live in `/run/media/s117/OS/Models/eval-data/`
(see `scripts/run_m12_stage_c.sh` for the exact file-name mapping). Holdout
slices and their SHA-256 values are in
`experiments/2026-08-28-m11-holdout/holdout-slices.json` and
`/run/media/s117/OS/Models/eval-data/holdout-m11/`.

## Implemented code

- strict build-10666 dry-run parser;
- pure-stdlib GGUF v3 layout reader and exact size predictor;
- canonical imatrix normalization/profile loader;
- exact-size baseline planner and safe promotion candidate generator;
- deterministic imatrix greedy, SHA-256 random, and block-balanced optimizers;
- exact-name llama.cpp tensor-type-file writer;
- deterministic M9/M11 log summarizers;
- preregistered M11 gate evaluator (`scripts/evaluate_m11_gate.py`);
- holdout slice generator (`scripts/make_holdout_slices.py`);
- M12 plan generator with ground-truth self-check
  (`scripts/generate_m12_block_balanced.py`).

Current verification:

```bash
python -m pytest -q
# 43 passed
git diff --check
```

## Accepted results

Formal curve: IQ3_M / FIT-25 / FIT-50 / FIT-75 / IQ4_XS.

- All three intermediate outputs are below target.
- Predicted versus actual size error is zero bytes.
- All have 851 tensors and zero qtype mismatches.
- Across five domains, all 20 adjacent KL steps decrease and all 20 adjacent
  Same-top steps increase.
- Macro KL: `0.141098 / 0.106277 / 0.097324 / 0.079720 / 0.057874`.

M9 is accepted only for this model/source/imatrix/runtime/data protocol.

## M11 result (holdout confirmation)

Preregistered gates all passed on untouched slices: v0.1b macro KL 0.083634
beats original FIT-50 0.088518 (-5.52%) and the random-seed mean 0.087132,
with 4-of-5 per-domain wins over the random mean and a +16.23% worst-domain
regression (wiki_test) inside the preregistered 25% guard. Honest caveat:
rebuilt random v3 alone ties v0.1b on macro KL (0.083698); v0.1b's advantage
is its cross-domain profile, not a uniform margin over every seed.

## M12 result (budget dependence)

The frozen v0.1b allocator applied at all three curve positions:
macro +5.31% (worse) at FIT-25, -3.99% at FIT-50, -4.75% at FIT-75 versus the
original curve on the M9 slices. The trade-off is budget-dependent; FIT-25
remains owned by the original utility. Both new artifacts match exact
predicted sizes with zero-byte error. See D-0018.

## Exact next step

1. Choose and preregister ONE next investigation before running it:
   a. a budget-conditional allocator-selection rule validated on fresh
      holdouts (new slices, new offsets, new SHA-256 values);
   b. the role-matched early/late block swap ablation for sharper attribution;
   c. a second model family for generalization.
2. Do not tune quarter weights, do not change the optimizer family, and do not
   begin optimizer v2 without a new preregistered gate.
3. Keep using the retained M9/M11 reference logits only with their matching
   slices; never mix.

## Storage and reproducibility

Large GGUFs, KLD files, and raw logs are intentionally ignored by Git. Recipes,
hashes, parsed JSON, and reports are tracked. The random GGUFs are fully
reproducible from their retained tensor-type files: the M11 rebuild reproduced
all three M10 SHA-256 values exactly. Retained artifacts: BF16 source, M9
references, original FIT-25/50/75, block-balanced FIT-50/25/75, and the three
rebuilt random FIT-50 GGUFs (about 24 GB free-headroom change; check `df`
before adding artifacts).

The initial Git commit was created after M11/M12; subsequent records should be
committed incrementally.
