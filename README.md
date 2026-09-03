<p align="center">
  <span>English</span> · <a href="README.zh-CN.md">简体中文</a>
</p>

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="branding/fit-gguf-logo-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="branding/fit-gguf-logo-light.svg">
    <img alt="FIT-GGUF" src="branding/fit-gguf-logo-light.svg" width="760">
  </picture>
</p>

<p align="center"><strong>Fit-to-Size Intelligent Tensor Quantization for GGUF</strong></p>

<p align="center">
  A deterministic planning layer that turns llama.cpp quantization presets into a near-continuous model-size control.
</p>

## The short version

Standard GGUF quantization asks you to choose a preset. FIT-GGUF starts from
the largest supported preset below a requested byte budget, then spends the
remaining bytes on deterministic tensor-level precision upgrades.

> **Traditional GGUF gives you presets. FIT gives you a size slider.**

FIT-GGUF v0.1 has two deliberately separate conclusions:

| Claim | Status | Scope |
| --- | --- | --- |
| Deterministic size prediction and recipe execution | **Validated** | 22/22 evaluated targets across two tested model families had zero-byte actual-vs-predicted error under the pinned toolchain; the current 14-tier release also matched its post-oracle predictions exactly. |
| Universally optimal tensor allocation | **Not established** | The imatrix-guided allocator helped on the development family but did not outperform matched random allocation on the second family. FIT claims precise size control, not a universal quality optimum. |

“Continuous” means arbitrary targets within the representable GGUF recipe
space. Tensor/qtype transitions are discrete, so a small amount of target
slack may remain. The predictor is required to match the resulting artifact;
it does not pretend that every byte target is exactly representable.

**v0.2 extends the slider into a quality dial: choose the fidelity you want;
FIT finds the smallest verified GGUF that safely meets it.** See
[Fidelity tiers](#v02-fidelity-tiers) below.

## First release: Qwen3.8-27B-Uncensored

The first public batch contains **14 FIT tiers from 7 GiB to 13.5 GiB in
0.5-GiB steps**, produced from
[`orcarouter/Qwen3.8-27B-Uncensored`](https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored).
Every tier includes its plan, effective recipe, tensor override file, measured
five-domain KL/Same-top results, and SHA-256 provenance.

![FIT-GGUF quality curve](docs/assets/kl-curve-en.png)

[English chart](docs/assets/kl-curve-en.png) · [中文图表](docs/assets/kl-curve-zh.png)

The curve is an empirical observation for this model and protocol, not a
general guarantee. P5 replaced the K-spanning 12.5-13.5G recipes with the
IQ3_M-to-IQ4_XS span; P6 repaired the 8-10G spans. The final measured FIT
curve is monotonic across all 14 release tiers, while the superseded recipes
remain retained as experiment evidence.

Additional release charts:

- Same-top agreement: [English](docs/assets/sametop-curve-en.png) · [中文](docs/assets/sametop-curve-zh.png)
- Old-to-final allocation comparison: [English](docs/assets/strategy-improvement-en.png) · [中文](docs/assets/strategy-improvement-zh.png)
- Target-budget utilization: [English](docs/assets/target-utilization-en.png) · [中文](docs/assets/target-utilization-zh.png)

## v0.2: Fidelity tiers

v0.2 adds **fidelity tiers** on top of exact-size planning. A tier is a dual
hard gate:

**PASS = macro KL ≤ tier limit ∧ Same-top ≥ validated model-specific Guard.**

- Tier KL limits come from a frozen Global KL Core: `Quality` ≤ 0.05,
  `Balanced` ≤ 0.10, `Compact` ≤ 0.15, `Mini` ≤ 0.20, measured under the
  frozen eval-v1 protocol.
- The Same-top floor is resolved from a **Guard Profile** validated for the
  exact model. With no validated profile, the CLI refuses to emit an official
  tier instead of borrowing a floor from another model.

`fit fidelity-search` then walks the healthy preset frontier (poison presets
excluded), brackets the crossing, and returns the **minimum verified PASS** —
not an extrapolation.

### Minimum Verified Size @ Fixed Fidelity

Flagship case study — [`orcarouter/Qwen3.8-27B-Uncensored`](https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored) (27B MoE; exact-model scope, not a cross-model claim).

| Tier | Nearest smaller frontier preset | FIT v0.2 | Nearest larger preset | Saving | Active |
| --- | --- | --- | --- | --- | --- |
| Quality | Q4_K_M · 15.41G · .0658 / 93.79 · FAIL-BOTH | **16.25G · .0497 / 94.88 · PASS** | Q5_K_S · 17.40G · .0455 / 95.59 · PASS | −6.6% | KL |
| Balanced | IQ3_M · 11.72G · .1445 / 89.38 · FAIL-BOTH | **12.84G · .0997 / 91.57 · PASS** | IQ4_XS · 14.05G · .0624 / 93.79 · PASS | −8.6% | KL |
| Compact‡ | IQ3_XS · 11.15G · .1512 / 89.05 · FAIL-KL | **11.17G · .1486 / 89.09 · PASS** | IQ3_S · 11.57G · .1424 / 89.53 · PASS | −3.4% | KL |
| Mini† | Q2_K · 9.98G · .2439 / 84.08 · FAIL-BOTH | **10.35G · .1924 / 86.31 · PASS** | IQ3_XXS · 10.42G · .1946 / 87.12 · PASS | −0.6% | KL |

**Minimum verified within the validated healthy frontier and configured search
tolerance (128 MiB).** Sizes are GiB; quality columns are macro KL / Same-top %
under the frozen eval-v1 protocol.

`†` Mini: the search stopped at the validated healthy-frontier boundary; the
region below it lacks a valid interpolation window.

`‡` Compact lies in a locally non-monotonic allocation region. Nearby larger
recipes can score worse, so FIT reports `noise_inversion`, refuses automatic
delivery, and requires verification of the final artifact itself. The released
11.17 GiB artifact was rebuilt and independently re-evaluated at KL 0.148592 /
Same-top 89.092%.

### Why FIT verifies the final artifact instead of trusting size monotonicity

```
11.17G  KL .1486  PASS
11.19G  KL .1519  FAIL
11.20G  KL .1493  PASS
11.21G  KL .1502  FAIL
```

Mixed-quantization recipes form a discrete, locally non-monotonic quality
surface: a larger artifact is not guaranteed to outperform every nearby
smaller artifact. FIT therefore treats noisy crossings as `noise_inversion`,
fails closed, and verifies the actual release artifact before promotion.

### v0.2 release gates (flagship model)

```
R1 Fidelity correctness        PASS
R2 Search accuracy             PASS
R3 Search budget               PASS
R4 Exact-byte guarantee        PASS
R5 v0.1 non-regression         PASS
R6 Reproducibility             PASS
Release Gates                  6 / 6 PASS
```

## Install

FIT-GGUF requires Python 3.11+ and a compatible llama.cpp runtime containing
`llama-quantize`.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python -m pytest tests/
```

The Python package's only runtime dependency is PyYAML (Guard Profile
parsing). Real analysis and quantization use the supplied llama.cpp binary.

## CLI workflow

### 1. Analyze a source/preset interval

```bash
fit analyze \
  --source model-BF16.gguf \
  --imatrix imatrix.gguf \
  --runtime /path/to/llama.cpp/bin \
  --lower IQ3_M \
  --upper IQ4_XS \
  --out-dir work/analysis
```

`analyze` captures the effective preset recipes with pinned
`llama-quantize --dry-run`, profiles the imatrix, derives GGUF metadata and
freezes the candidate set.

### 2. Plan an explicit byte target

```bash
fit plan \
  --analysis work/analysis/analysis.json \
  --target-bytes 12884901888 \
  --policy balanced \
  --model-name MyModel \
  --out-prefix work/MyModel-FIT-12G
```

This writes a plan record, effective recipe and `--tensor-type-file`. Planning
is deterministic for fixed inputs, policy and toolchain.

### 3. Quantize and enforce the prediction

```bash
fit quantize \
  --analysis work/analysis/analysis.json \
  --tensor-types work/MyModel-FIT-12G-tensor-types.txt \
  --out MyModel-FIT-12G.gguf \
  --expect-bytes PREDICTED_BYTES_FROM_THE_PLAN
```

The command rejects a size mismatch and records the output SHA-256.

### 4. Or search by fidelity (v0.2)

```bash
fit fidelity-search \
  --source model-BF16.gguf \
  --imatrix imatrix.gguf \
  --runtime /path/to/llama.cpp/bin \
  --refs-dir refs/bf16 \
  --eval-data-dir eval-slices \
  --guard-registry profiles/guard \
  --tier compact \
  --preset-ladder IQ2_XXS,IQ2_M,IQ3_XXS,IQ3_XS,IQ3_S,IQ3_M,IQ4_XS \
  --manifest work/manifest.txt \
  --logs-dir work/logs \
  --out-dir out/compact \
  --work-dir /dev/shm/fit-compact
```

This resolves the tier contract (KL limit + validated Guard floor), searches
the healthy preset frontier for the minimum verified PASS, then builds the
exact-size artifact and re-evaluates **the final artifact itself** against the
tier contract. `--preset-ladder` auto-analyzes adjacent preset pairs; pass
frozen `--analysis` directories instead when you need byte-stable
reproducibility. Search budgets: `--profile normal` ≤ 8 fresh evaluations,
`--profile precise` ≤ 16. If the crossing lies in a locally non-monotonic
region the search reports `noise_inversion` and fails closed rather than
delivering automatically.

## How it works

1. **Anchor** — select the largest supported lower preset whose predicted
   artifact fits the budget.
2. **Measure** — derive tensor shapes, effective qtypes, encoded byte deltas
   and imatrix statistics from the actual source/toolchain.
3. **Allocate** — rank safe positive-precision transitions and pack them under
   the remaining byte budget using a frozen policy.
4. **Ask the oracle** — replay the proposed tensor override file through
   llama.cpp dry-run so counter-based preset rules are reflected in the final
   effective recipe.
5. **Verify** — quantize, check the artifact byte size against the oracle
   prediction, and hash the result.

This architecture matters because llama.cpp presets are recipes, not a single
qtype applied uniformly to every tensor.

## Evidence and reproducibility

- `FINAL_REPORT.md` — v0.1 conclusions, accepted claims and rejected claims.
- `experiments/` — frozen experiment inputs, gates, logs and machine-readable
  results from M0–M16 and productization stages.
- `experiments/2026-08-29-p4-release-batch/` — the first 14-tier release batch.
- `src/fit_gguf/` — parser, GGUF size model, imatrix profiler, planner,
  optimizer and CLI.
- `tests/` — deterministic unit and integration-level coverage.

The published measurements use llama.cpp b10666 (`4e97ac86e`), Linux x86_64,
ROCm, 512-token context, 512 batch size, aligned BF16 references and five fixed
64 KiB slices (`wiki_test`, `wiki_valid`, Chinese, code and `agent_chat`). KL
and Same-top are primary directional metrics; short-corpus PPL is diagnostic.
Macro KL is the two-level domain mean (per-domain mean first, then across
domains) under the frozen eval-v1 evaluator contract.

Exact-size semantics: `quantize.imatrix.file` is serialized into GGUF metadata
by the pinned llama.cpp runtime. FIT finalizes size prediction using the exact
quantize-time imatrix path, so target-size accuracy remains exact. Reproducing
an artifact byte-for-byte, including its SHA-256, additionally requires the
same serialized imatrix path string.

## Boundaries

- A FIT filename such as `FIT-12G` describes the requested **main GGUF file
  budget**, not total RAM or VRAM usage. KV cache, compute buffers, runtime
  overhead and any multimodal projector are separate.
- Exact prediction is toolchain-scoped. Revalidate after changing llama.cpp,
  converter behavior, metadata, source layout or platform.
- Quality monotonicity is observed on tested curves, not enforced by the
  planner. Pair-boundary regressions can occur and must remain visible.
- The balanced v0.1b allocator is a documented release choice, not a
  cross-model theorem. Its transfer failure is retained in `FINAL_REPORT.md`.
- The current optimizer is upgrade-only within a selected lower/upper preset
  interval; it is not an unrestricted global qtype search.

## Project records

- `DECISIONS.md` — accepted and rejected design decisions (D-0001..D-0024).
- `FINAL_REPORT.md` — the v0.1 research report and its validated claims.
- `docs/llama-integration.md` — reviewed integration path through llama.cpp.
- `eval-data/PROVENANCE.md` — sources, offsets and SHA-256 of the five
  preregistered KL evaluation slices.
- `experiments/` — preregistered experiment records (gates frozen before
  execution, results recorded as measured).

## License

MIT — see [LICENSE](LICENSE). The released base model
(`orcarouter/Qwen3.8-27B-Uncensored`) carries its own license terms; the
released FIT quantizations inherit the model's usage restrictions.
