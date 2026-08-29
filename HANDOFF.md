# FIT-GGUF Handoff

Date: 2026-08-29 (updated after P1 by the GLM-5.3-Flash session)

## Read this first

The repository is past P1. M0-M16 (research) and P1 (productization) are
complete; the project is versioned **FIT-GGUF v0.1** and frozen. The final
evidence state, per D-0022 and D-0023: FIT's exact deterministic size control
and monotonic quality-vs-budget curves transfer to a second model family
(granite-4.2-8b, 11/11 zero-byte artifacts); the imatrix-allocation value does
NOT transfer (original utility loses to random at FIT-50 on Granite; v0.1b
shows no advantage there). On the Huihui development model, v0.1b is confirmed
at FIT-50 (M11) and the original utility at FIT-25 (M15). The `fit
analyze/plan/quantize` CLI replays both models' historical FIT-50 ground truth
byte-identically (preregistered gates G1-G8, all passed). Every milestone is
preregistered, executed, and recorded; nothing is pending.

Canonical status and evidence:

- `PROJECT_STATE.md` - concise source of truth;
- `DECISIONS.md` - decisions D-0001 through D-0022;
- `experiments/2026-08-28-m9-fit-curve/README.md` - accepted formal curve;
- `experiments/2026-08-28-m10-ablation/README.md` - random and block diagnosis;
- `experiments/2026-08-28-m11-holdout/README.md` - preregistered holdout
  confirmation of v0.1b at FIT-50 (gates A/B/C all passed);
- `experiments/2026-08-28-m12-block-balanced-curve/README.md` - v0.1b curve
  extension showing the budget-dependent trade-off;
- `experiments/2026-08-28-m13-budget-rule/README.md` - preregistered
  budget-rule validation and its rejection (D-0019);
- `experiments/2026-08-28-m14-swap-ablation/README.md` - crossover ablation
  rejecting the positional mechanism (D-0020), with the recipe-overlap
  diagnostic (byte-Jaccard 0.328/0.464/0.689 at 25/50/75);
- `experiments/2026-08-29-m15-random-baseline/README.md` - matched random
  baseline: H25 strong support, H75 collapse rejected, D-0021 freeze;
- `experiments/2026-08-29-m16-granite-reveal/README.md` - cross-model reveal:
  size control transfers, allocator value does not (D-0022).
- `experiments/2026-08-29-p1-cli/README.md` - preregistered productization
  replay: CLI reproduces both models' FIT-50 ground truth byte-identically
  (D-0023).
- `FINAL_REPORT.md` - the closing v0.1 report: what transfers, what is
  model-specific, what stays open.

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
  (`scripts/generate_m12_block_balanced.py`);
- the `fit analyze/plan/quantize` CLI (`src/fit_gguf/cli.py` +
  `src/fit_gguf/pipeline.py`) with derived metadata, auto block span, exact
  integer fraction targets, and SHA-256-provenanced plan/quantize records
  (`scripts/run_p1_replay.sh` + `scripts/evaluate_p1_replay.py`).

Current verification:

```bash
python -m pytest -q
# 53 passed
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

## M14 result (positional mechanism rejected)

The preregistered bidirectional crossover on a fourth holdout set: S50 = +3.70%
but carried entirely by the harmful B->L arm (+7.56%); O->E is macro-neutral
(-0.17%) while trading wiki against non-wiki; the matched shuffle lands at the
harmful arm's level; both wiki domains fail the domain gate (3-of-5 non-negative,
needing 4). NOT ACCEPTED per the frozen rule. Secondary predictions confirmed:
S75 = -0.46% within the +-1% ROPE, S50 > S75. See D-0020.

## Exact next step

None scheduled. M0-M16 and P1 are complete and FIT-GGUF is versioned v0.1
(FINAL_REPORT.md, README). Research is frozen per D-0022/D-0023: any future
allocator experiment (v0.2) requires a new preregistered design - conditional
marginal utility per D-0020 - on a fresh third model; do not tune on Huihui
or Granite, which are development data only.

## Storage and reproducibility

Large GGUFs, KLD files, and raw logs are intentionally ignored by Git. Recipes,
hashes, parsed JSON, and reports are tracked. Quantized GGUFs are fully
reproducible from their retained tensor-type files: the M11 rebuild reproduced
all three M10 SHA-256 values exactly, and the P1 replay reproduced four more
historical hashes bit-for-bit. As of the post-freeze cleanup no quantized GGUF
or BF16 GGUF remains under `artifacts/`; see the disk policy below.

Disk policy (owner-directed, 2026-08-29): after the v0.1 freeze the owner
directed deletion of ALL remaining GGUF artifacts - the six retained curve
artifacts, the M2 preset artifacts, and both BF16 sources (23 GGUFs, about
239 GB freed; disk 89% -> 75%). Everything deleted is reproducible: the
quantized artifacts from their retained tensor-type files (bit-exact
reproduction proven in M11 and again in the P1 replay), and the BF16 sources
from the safetensors trees under `/run/media/s117/OS/Models/`
(`Huihui-Qwen3.8-27B-abliterated/`, `granite-4.2-8b/`) with the documented
conversions. Full provenance hashes live in the experiment records: M2/M9/M10
/M12 README SHA tables, M16 `artifact-hashes.txt`, Huihui BF16
`8a033407...` in PROJECT_STATE, and the now-complete Granite BF16 hash in
`experiments/2026-08-29-m16-granite-reveal/granite-bf16-sha256.txt`
(`d82690e0dc827f2c43effeb3d489f572afbe2541fa8b4895b0d958ce473925a6`). The
safetensors sources and the calibration dataset are project inputs. On
2026-08-29 the duplicate copies under `test-Models/` were removed and replaced
with symlinks to the canonical copies under
`/run/media/s117/OS/Models/`. The first release source,
`orcarouter/Qwen3.8-27B-Uncensored`, is being fetched into
`/run/media/s117/OS/Models/orcarouter-Qwen3.8-27B-Uncensored/`. Check `df`
before adding artifacts; about 406 GB was free after the post-freeze cleanup.

Commits: a16d7ab (M0-M12), f5f93cc (M13 prereg), 414aaa6 (M13 results),
5836443 (M14 prereg + overlap diagnostic), then M14 results; incremental.
