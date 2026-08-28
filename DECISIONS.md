# FIT-GGUF Decision Log

## D-0001: Use baseline-anchored promotion

Reason:
Starting from llama.cpp's effective lower-preset recipe provides a conservative
quality floor and makes arbitrary-size planning an upgrade-only problem.

Alternatives:
Unrestricted tensor/qtype search, or allowing downgrades to fund other
upgrades.

Evidence:
The project specification makes the nearest smaller standard preset the minimum
quality baseline. Unrestricted search is not needed to test the core product
claim.

Status:
Accepted.

## D-0005: Use llama-quantize dry-run as the initial M2 recipe source

Reason:
Dry-run uses the exact source GGUF tensor set and the same upstream preset,
override, architecture, and shape-fallback logic as actual quantization.

Alternatives:
Reimplement preset rules in Python, or immediately add a custom llama.cpp dump
tool around the staging C++ API.

Evidence:
Pinned-source review of `llama_model_quantize_impl` shows target types are
computed before the dry-run branch and printed per tensor. The matching binary
accepts all required development presets.

Status:
Accepted for M2. An in-process tool remains an option only if text parsing is
shown to be unstable.

## D-0006: Do not use the dry-run total as the M3 file-size oracle

Reason:
The dry-run total sums unpadded tensor payload bytes and omits GGUF metadata and
alignment overhead.

Alternatives:
Treat the printed total as final file size, or use parameters-times-bits
estimates.

Evidence:
Pinned source lines 1189-1206 sum `ggml_nrows * ggml_row_size`, while actual
writing reserves `gguf_get_meta_size` and pads every tensor to the GGUF
alignment.

Status:
Accepted. M3 will add exact output metadata and padding, then validate against
real GGUF files.

## D-0007: Parse dry-run output strictly at the process boundary

Reason:
Build 10666 already exposes the effective per-tensor qtype selected after
preset rules, manual overrides, and shape fallback. A small strict parser keeps
M2 tied to that behavior without duplicating llama.cpp policy in Python.

Alternatives:
Reimplement every preset rule, or immediately maintain a custom C++ dump tool.

Evidence:
The parser accepts both unchanged and converted build-10666 tensor lines,
requires a complete unique ordinal/name set, validates payload summaries within
their printed-MiB rounding bounds, and passes 18 focused tests.

Status:
Accepted for the pinned runtime format. Real IQ3_S, IQ3_M, and IQ4_XS logs must
still be compared with actual quantization before M2 is complete.

## D-0008: Exclude the auxiliary NextN head from the development target GGUF

Reason:
The one-layer NextN/MTP head is an auxiliary speculative-decoding component,
not part of the ordinary target-model next-token path, and its matrices are not
covered by the canonical imatrix.

Alternatives:
Bundle the unprofiled head into every FIT budget, or export it as a separate
draft artifact.

Evidence:
The matching converter reports 866 tensors with the head and 851 with
`--no-nextn`. The canonical imatrix exactly covers the 496 quantizable
main-model matrices, while the extra head has no entries.

Status:
Accepted for development and target-model quality evaluation. Separate MTP
support remains possible after the core FIT claim is validated.

## D-0009: Use a strict stdlib GGUF layout reader for exact size prediction

Reason:
Exact size needs tensor dimensions, pinned qtype block traits, metadata
transformation sizes, and per-tensor alignment, but not tensor values. A small
read-only parser avoids a runtime dependency on the much larger llama.cpp
Python conversion package.

Alternatives:
Estimate from printed MiB, bind the C++ library, or maintain a custom helper
binary.

Evidence:
The implementation predicts three standard 27B artifacts and one targeted
mixed-qtype artifact with zero-byte error. It also caught the 224-byte imatrix
metadata increase omitted by the delegated M3 source report.

Status:
Accepted for pinned build 10666, GGUF v3, single-file output, and explicitly
supported qtypes. Revision or format changes require revalidation.

## D-0010: Preserve raw and role-normalized imatrix summaries

Reason:
Raw mean activation energy varies by orders of magnitude across roles, while
within-role block variation is also substantial. Keeping both avoids silently
turning a scale artifact into an importance ranking.

Alternatives:
Use only a global mean, only a role percentile, or treat every channel vector
as an independent tensor-level observation.

Evidence:
The 496-entry canonical profile shows FFN-down means from 0.000090 to 2.450404
and repeated identical statistics for tensors sharing the same layer input.

Status:
Accepted as profiling output. The final candidate utility still requires real
quality validation and must not claim that a scalar imatrix score is causal.

## D-0011: Reject non-positive lower-to-upper tensor transitions

Reason:
A larger preset file is not necessarily a tensorwise precision superset. FIT's
lower-baseline guarantee applies to the effective recipe, not the preset label.

Alternatives:
Copy every upper-preset qtype or assume preset size implies per-tensor quality.

Evidence:
IQ3_M to IQ4_XS contains 24 `Q4_K -> IQ4_XS` transitions with negative byte
cost. Keeping Q4_K makes the safe candidate envelope 38,010,880 bytes larger
than the net preset-size gap.

Status:
Accepted. Candidate generation requires a strict positive encoded-BPW and byte
delta.

## D-0012: Treat imatrix utility as a provisional search proxy

Reason:
The greedy optimizer needs deterministic ordering before real quality labels
exist, but scalar imatrix summaries do not prove causal quality gain.

Alternatives:
Claim the proxy is a quality model, randomize allocation, or block all progress
until exhaustive ablations exist.

Evidence:
The first 13 GiB plan uses 99.9817% of its target and quantizes exactly as
planned. M10 quality and M11 ablation results are still required to accept or
replace the utility formula.

Status:
Accepted only as the v0.1 search heuristic, not as a quality conclusion.

## D-0013: Regenerate quality references for the current weights

Reason:
The preserved reference logits were generated before layers 18-51 of the
Huihui source were replaced. KL comparisons require token-, vocabulary-, and
weight-aligned reference logits.

Alternatives:
Reuse the old logits because the model name is unchanged, report PPL only, or
compare against an unrelated external reference.

Evidence:
The model version notes identify the weight replacement, while the current
BF16 GGUF has SHA-256
`8a033407c8f58d43102aade25b973cc6d2f2ce5c5cbf4dc75a2cdb60b9e33cbc`.

Status:
Accepted. M9 quality evaluation must generate fresh BF16 reference logits from
this exact GGUF and record evaluation-data hashes and runtime parameters.

## D-0014: Accept M9 on KL and Same-top, retain PPL diagnostics

Reason:
KL directly compares the full quantized distribution with the aligned BF16
reference, while Same-top checks agreement of the highest-probability token.
Short-corpus PPL point estimates can fluctuate within their reported
uncertainty even when distribution agreement improves.

Alternatives:
Require every PPL point estimate to be monotonic, average unlike domain PPLs,
or select only the metrics that favor FIT.

Evidence:
Across five domains and five size points, all 20 adjacent KL transitions
strictly decrease and all 20 adjacent Same-top transitions strictly increase.
Chinese and code contain non-monotonic PPL point estimates whose changes are
comparable to their uncertainty.

Status:
Accepted for M9. Report all three metrics and their uncertainty. Do not claim
positive imatrix-allocation evidence until matched-size M10 ablations pass.

## D-0015: Use SHA-256 priorities for reproducible random baselines

Reason:
Random promotion is a required research baseline, but ordinary language-level
hashes and shuffle implementation details can make a supposedly fixed seed
non-reproducible across processes or runtimes.

Alternatives:
Use Python `hash()`, rely on `random.shuffle`, or store only the final tensor
list without a generative rule.

Evidence:
`optimize_random()` derives a stable priority from the recorded seed, tensor
name, and destination qtype. Unit tests verify input-order independence and
that the baseline does not follow utility ordering.

Status:
Accepted for controlled random ablations. Multiple seeds are mandatory before
making a reliability claim.

## D-0016: Do not accept positive allocation evidence after FIT-50 random trials

Reason:
An allocation heuristic must beat random reliably across domains, not only on
an aggregate or a favorable seed.

Alternatives:
Accept the 2.36% macro improvement over the three-seed mean, cite the 13/15
Same-top wins alone, or discard the unfavorable random seed.

Evidence:
FIT wins macro KL against two random seeds but loses to the third. It beats the
random mean strongly on both wiki domains, loses on Chinese, and is effectively
tied on agent_chat. Random v3 materially beats FIT on Chinese KL and slightly
on Chinese Same-top.

Status:
Rejected as positive allocation proof. Continue M10 with a matched-budget
block-balanced imatrix ablation before changing the optimizer family.

## D-0017: Freeze block-balanced v0.1b pending untouched holdout validation

Reason:
The original imatrix utility concentrates upgrade bytes in late blocks and
performs strongly on wiki domains but weakly on Chinese. Equal block-quarter
byte quotas directly test that positional concentration without changing the
within-quarter utility rule.

Alternatives:
Immediately replace the original planner, tune quarter weights on the five
observed domains, or add optimizer-v2 complexity.

Evidence:
At matched FIT-50 budget, block balancing improves Chinese and agent_chat KL by
12.1% each and macro KL by 3.99%, while wiki_test and wiki_valid regress by
20.0% and 15.8%. It beats all three random seeds on macro KL. The hypothesis
was chosen after observing these domains, so the same results are adaptive,
not untouched confirmation.

Status:
Accepted as a frozen v0.1b research candidate only. Require fresh holdout
confirmation before promotion or quarter-weight tuning.

## D-0002: Use the supplied build-10666 ROCm runtime

Reason:
The supplied runtime is small, versioned, executable on the target platform,
and already exposes selective tensor qtypes plus dry-run size calculation.

Alternatives:
Use the unrelated dirty source checkout, or build an arbitrary newer revision
before inspecting the supplied runtime.

Evidence:
`llama-cli --version` reports build 10666 and commit `4e97ac86e`.
`llama-quantize --help` confirms `--dry-run`, `--tensor-type`, and
`--tensor-type-file`.

Status:
Accepted for initial development. Matching source must still be acquired and
audited.

## D-0003: Preserve large local assets but exclude them from Git

Reason:
The weights, imatrix, calibration data, prior plans, and reference logits are
potential experimental inputs or provenance evidence, but committing roughly
95 GiB into the FIT repository would be unsafe and unnecessary.

Alternatives:
Delete the assets, commit them, or relocate all of them before investigation.

Evidence:
The workspace audit found valid model shards, calibration data, prior tensor
recipes, and multi-domain reference artifacts. Only a 12 KiB model-download
cache was unambiguously disposable and was removed.

Status:
Accepted.

## D-0004: Delegate M1 through Anti-Gravity CLI in read-only mode

Reason:
The project specification explicitly assigns the initial llama.cpp source audit
to Gemini 3.7 Flash while retaining review responsibility in the primary agent.

Alternatives:
Use a Codex subagent or allow the delegated agent to edit source during M1.

Evidence:
The installed `agy` CLI exposes `gemini-3.7-flash-high`, plan mode, sandboxing,
and non-interactive output.

Status:
Accepted.

## D-0018: Record budget-dependent allocation after M11/M12

Reason:
Positive allocation evidence must state where an allocator wins, not only
whether it wins somewhere. M11 confirmed frozen v0.1b on untouched holdout
data at FIT-50 under the preregistered gates; M12 showed the same frozen
allocator loses to the original utility at FIT-25 and wins at FIT-75, so the
allocation trade-off is budget-dependent rather than a uniform improvement.

Alternatives:
Promote v0.1b as the universal planner, tune quarter weights per budget on
already-seen domains, or hide the FIT-25 regression behind macro averages.

Evidence:
M11 holdout macro KL: v0.1b 0.083634 vs original FIT-50 0.088518 and random
mean 0.087132, with 4-of-5 per-domain wins over the random mean and a +16.23%
worst-domain regression inside the preregistered guard; random v3 alone ties
v0.1b on macro (0.083698). M12 on the original M9 slices: v0.1b macro
+5.31% at FIT-25 (worse on 4 of 5 domains), -3.99% at FIT-50, -4.75% at
FIT-75. All artifacts match exact predicted sizes with zero-byte error.

Status:
Accepted as a recorded finding. The FIT-25 position remains owned by the
original utility; v0.1b is supported only at FIT-50/FIT-75. Any
budget-conditional selection rule, quarter-weight adaptation, or optimizer-v2
work requires a new preregistered gate with untouched holdout validation
before acceptance.
