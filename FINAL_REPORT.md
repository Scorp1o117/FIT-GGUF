# FIT-GGUF v0.1 Final Report

**Continuous-Size Quantization Works; Utility Allocation Does Not Yet
Generalize**

Date: 2026-08-29
Status: v0.1 research freeze (M16, D-0022) plus productization replay (P1)

## 1. Summary

FIT-GGUF set out to answer one product question: can a GGUF model be
quantized to *any* target size between two standard llama.cpp presets, with
exact deterministic size control and quality that degrades gracefully?

After the full M0-M16 program on two models, the answer splits cleanly in
two, and the final claim is deliberately worded to stay inside the evidence:

1. **Exact-size control works.** Across the two tested model families, all
   evaluated target recipes matched their predicted output sizes exactly —
   22/22 targets achieved zero-byte prediction error under the tested
   toolchain — and the productized CLI reproduced four historical reference
   GGUFs bit-for-bit. "Continuous" here means near-continuous: arbitrary
   target sizes within the representable recipe space (existing GGUF
   qtypes/presets plus tensor overrides); the smallest granularity is still
   the discrete byte change of a tensor/qtype transition.

2. **The imatrix-guided utility allocator does not demonstrate cross-model
   generalization.** It was beneficial on the development model (multiple
   holdouts and matched random baselines), but failed to outperform matched
   random allocation on granite-4.2-8b at FIT-50, and v0.1b showed no
   advantage there. Both were recorded as preregistered failures, not tuned
   away. FIT v0.1 therefore claims reliable target-size control, not
   universally superior tensor allocation.

In one sentence: **FIT-GGUF v0.1 is accepted as a product prototype; the
allocator research is frozen with a negative transfer result.**

## 2. Method

FIT plans a quantization at a byte target T between a lower preset L (IQ3_M)
and an upper preset U (IQ4_XS):

1. `analyze`: one pinned `llama-quantize --dry-run` per preset captures the
   effective per-tensor recipe; a strict stdlib GGUF reader predicts exact
   output sizes (metadata, padding, and imatrix KVs included); the imatrix is
   profiled with llama.cpp normalization semantics into per-tensor
   role-relative importances.
2. `plan`: safe upgrade candidates (strict positive encoded-BPW and byte
   deltas only; 24 negative transitions rejected, D-0011) are ranked by the
   frozen policy and greedily packed under the exact residual budget
   `T - size(L)`; the recipe is rebuilt and its size re-predicted exactly.
3. `quantize`: `llama-quantize --imatrix --tensor-type-file` executes the
   recipe from the lower preset; the output is verified against the
   prediction byte-for-byte in size and hashed.

All planning is deterministic; all quantization reproduces SHA-256-identical
artifacts on re-run.

## 3. What was validated

### Size control (transfers across both tested model families; product-ready)

- 22/22 evaluated targets achieved zero-byte prediction error under the
  tested toolchain: 4/4 preset predictions and 18/18 FIT-plan predictions
  across both models (M2, M3, M7, M9, M10, M12, M14-M16), 11/11 zero-byte on
  Granite (M16). This is a statement about the pinned build-10666 toolchain,
  not about every llama.cpp build or platform.
- Re-quantization from retained tensor-type files reproduces artifact
  SHA-256 exactly (M11 rebuild of three deleted random artifacts). The P1
  hashes prove that the CLI reproduces the *frozen research pipeline*
  bit-for-bit on this toolchain; they do not claim universal determinism
  across compilers, upstream versions, or metadata changes.
- The `fit` CLI replays both historical ground-truth points (Huihui and
  Granite FIT-50, original and block-balanced) with byte-identical recipes
  and identical artifact hashes (P1, see section 6).

### Quality-vs-budget behavior (observations, not guarantees)

On the development model (M9), across five domains (wiki_test, wiki_valid,
Chinese, code, agent_chat), macro KL improved monotonically at every size
step of IQ3_M → FIT-25 → FIT-50 → FIT-75 → IQ4_XS:
0.141098 → 0.106277 → 0.097324 → 0.079720 → 0.057874, with Same-top strictly
increasing 89.456% → 93.865%. All 20 adjacent transitions in all five domains
moved in the expected direction.

On Granite (M16 diagnostics, ungated), the monotone structure was observed as
well: KL decreased strictly from IQ3_M through every FIT point to IQ4_XS in
every evaluated domain. These are empirical observations under the tested
protocol — the planner's size constraint is enforced, but quality
monotonicity is not an algorithmic guarantee.

### v0.1b block balancing at FIT-50 on the development model (M11)

On a preregistered untouched holdout, block-balanced FIT-50 beat the original
ordering by 5.52% macro KL, beat the three-seed random mean, and passed all
three frozen gates (worst-domain regression +16.23% inside the 25% guard).

### Original utility owns FIT-25 (M15)

Original ordering beat all three matched random seeds at FIT-25 on holdout
data, by 6.17% over the random mean, with every seed worse in every domain.

## 4. What was rejected (all preregistered, recorded as failures)

| Hypothesis | Milestone | Outcome |
| --- | --- | --- |
| Positional mechanism (late-block concentration is what helps) | M14 crossover ablation | Rejected: order-fixed equal-cost swap improved the metric S50 = +3.70% (worse), gates not met (D-0020) |
| Budget-conditional allocator rule (r<0.5 → original, else v0.1b) | M13 holdout | Rejected: FIT-75 direction did not reproduce (+0.21%, tie); composite rule still beat both pure strategies but Gate 1 failed (D-0019) |
| FIT-75 allocation-sensitivity collapse | M15 paired random baseline | Not confirmed: O/B gap swung to -3.65% after three ties; random spread 3.97% |
| Original utility transfers cross-model | M16 Granite reveal | Rejected: original FIT-50 loses to the random mean by 1.36%; best variant is random seed r1 (macro 0.165320) (D-0022) |
| v0.1b balancing transfers cross-model | M16 Granite reveal | Rejected: B50 within the ±1% ROPE of O50 (+0.38%) (D-0022) |

The M16 verdict is the honest headline: **G-size PASS, G-util FAIL,
G-bal FAIL.** Positive allocation evidence exists only on the development
model and only at some budgets.

## 5. Scope and limitations

- Pinned runtime: llama.cpp build 10666 (`4e97ac86e`), ROCm, Linux x86_64.
  Size prediction is exact for this build's metadata behavior; other builds
  require revalidation. All quality results are protocol-scoped
  (512-token context, fixed 64 KiB slices, KL/Same-top against aligned BF16
  references; short-corpus PPL point estimates are diagnostics only, D-0014).
- Envelope: IQ3_M ↔ IQ4_XS with upgrade-only transitions from the lower
  preset (D-0001); no downgrades, no unrestricted search. Targets are
  near-continuous within this representable recipe space, not mathematically
  continuous.
- The auxiliary NextN/MTP head is excluded from targets and budgets (D-0008).
- The scalar imatrix utility is a search proxy, not a quality model (D-0012).
  The negative result is about cross-model generalization of *this
  heuristic*, not about imatrix utility being useless everywhere.
- Documented deviation kept for reproducibility: Granite's BF16 conversion
  used the upstream master `conversion/` package because the pinned converter
  lacked Granite support, while the evaluation/quantization runtime remained
  the pinned b10666 build.

## 6. P1 productization replay

The `fit` CLI (`fit analyze` / `fit plan` / `fit quantize`,
`src/fit_gguf/pipeline.py` + `cli.py`) reimplements the experiment pipeline
with every hand-copied constant replaced by derivation (metadata provenance
from the imatrix GGUF, block span from the profile, targets by exact integer
fraction arithmetic). Preregistered replay gates, all frozen in
`experiments/2026-08-29-p1-cli/README.md` before execution:

| Gate | Requirement | Result |
| --- | --- | --- |
| G1 | Huihui analyze: source hash, preset predictions, provenance = hand META | PASS |
| G2 | Huihui plan original: tensor-types byte-identical to M7, sizes exact | PASS |
| G3 | Huihui plan balanced: tensor-types byte-identical to M10 | PASS |
| G4 | Huihui quantize: artifacts hash = M9/M10 SHA-256 | PASS |
| G5 | Granite analyze: preset predictions, provenance (see amendment) | PASS |
| G6 | Granite plan original: tensor-types byte-identical to M16 | PASS |
| G7 | Granite plan balanced: tensor-types byte-identical to M16 | PASS |
| G8 | Granite quantize: artifacts hash = M16 SHA-256 | PASS |

All eight gates passed on the first full run: both Huihui plans and both
Granite plans produced tensor-type files byte-identical to their M7/M10/M16
ground truth, and all four re-quantized artifacts reproduced their historical
SHA-256 hashes exactly (`e4fe1c46...`, `7cfa1b91...`, `09ca3d85...`,
`17660767...`). Zero behavioral drift. Mechanical verdict recorded in
`experiments/2026-08-29-p1-cli/gate-verdict.json`. One provenance note per
the prereg amendment: the Granite imatrix's `imatrix.chunk_count` (3394)
differs from the M16 hand-recorded 1250; the KV is a 4-byte integer, so the
value cannot affect size prediction.

## 7. Version and deliverables

- Version: **v0.1** (not v1.0 — the allocation claim is model-scoped).
- Deliverables: `fit` CLI, 53 unit tests, this report, updated
  PROJECT_STATE/HANDOFF/DECISIONS (D-0023), preregistered experiment record
  M0-M16 and P1.
- Research freeze: M16 closed allocator research. A future v0.2, if reopened,
  should use the conditional-marginal-utility design (local marginal probe →
  context features → pairwise) on a third, officially released sparse MoE
  validation model, with Huihui and Granite as development data only.

## 8. Closing

FIT solved "9 GB for a 9 GB slot": exact, deterministic, reproducible sizes
between presets, with monotonic quality. What it did not solve - and honestly
reports not solving - is *where the bytes go best* in a model it has never
seen. That allocation problem stays open, and the negative results here are
the map of where not to dig next.
