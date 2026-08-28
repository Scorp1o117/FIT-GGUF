# FIT-GGUF Handoff

Date: 2026-08-28 (updated after M13 by the GLM-5.3-Flash session)

## Read this first

The repository is past M13. M0-M13 are complete. Do not restart source
conversion, imatrix profiling, preset quantization, the five-domain M9 curve,
the M11 holdout validation, or the M13 budget-rule test. The current open
question, per D-0019: the preregistered budget-conditional rule was REJECTED
on the third holdout set (FIT-75 direction flipped to a statistical tie). The
preregistered failure branch - the role-matched early/late block swap
ablation - is the only permitted next step. The 0.50 threshold, quarter
weights, and all recipes are frozen; v0.1b is confirmed only at FIT-50
(M11), and the composite rule's fresh-data win does not override the failed
direction gate.

Canonical status and evidence:

- `PROJECT_STATE.md` - concise source of truth;
- `DECISIONS.md` - decisions D-0001 through D-0019;
- `experiments/2026-08-28-m9-fit-curve/README.md` - accepted formal curve;
- `experiments/2026-08-28-m10-ablation/README.md` - random and block diagnosis;
- `experiments/2026-08-28-m11-holdout/README.md` - preregistered holdout
  confirmation of v0.1b at FIT-50 (gates A/B/C all passed);
- `experiments/2026-08-28-m12-block-balanced-curve/README.md` - v0.1b curve
  extension showing the budget-dependent trade-off;
- `experiments/2026-08-28-m13-budget-rule/README.md` - preregistered
  budget-rule validation and its rejection (D-0019).

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

## M13 result (budget rule rejected)

Preregistered rule `r<0.5 -> original, r>=0.5 -> v0.1b` on a third, disjoint
holdout set: FIT-25 and FIT-50 directions reproduced; the composite rule beat
both pure strategies (0.084581 vs 0.086414/0.086988); but FIT-75 flipped to a
+0.21% tie and failed preregistered gate 1. NOT ACCEPTED per the frozen rule;
failure branch is the role-matched early/late block swap ablation. See
D-0019.

## Exact next step

1. Preregister and run the role-matched early/late block swap ablation - the
   frozen M13 failure branch and the only permitted next step. It must
   attribute WHERE v0.1b's confirmed FIT-50 gain comes from (which block
   roles/positions) before any deployment or promotion claim.
2. Do not move the 0.50 threshold, do not tune quarter weights, do not change
   the optimizer family, and do not begin optimizer v2 without a new
   preregistered gate.
3. Only after the attribution settles: M14 (matched random seeds at
   FIT-25/FIT-75) and M15 (second model family - the user plans to download
   ibm-granite/granite-4.2-8b into a models/ folder for cross-model testing).
4. Keep using the retained M9/M11/M13 reference logits only with their
   matching slices; never mix.

## Storage and reproducibility

Large GGUFs, KLD files, and raw logs are intentionally ignored by Git. Recipes,
hashes, parsed JSON, and reports are tracked. The random GGUFs are fully
reproducible from their retained tensor-type files: the M11 rebuild reproduced
all three M10 SHA-256 values exactly. Retained artifacts: BF16 source, M9
references, original FIT-25/50/75, block-balanced FIT-50/25/75, and the three
rebuilt random FIT-50 GGUFs (about 24 GB free-headroom change; check `df`
before adding artifacts).

The initial Git commit (a16d7ab) covered M0-M12; M13 preregistration
(f5f93cc) and M13 results are committed incrementally.
