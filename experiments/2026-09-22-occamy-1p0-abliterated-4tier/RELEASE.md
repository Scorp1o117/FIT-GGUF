# occamy-1.0-abliterated — FIT tiers

Five FIT (Fidelity-Interpolated Tensor) tiers for `occamy-1.0-abliterated`,
produced with [FIT-GGUF](https://github.com/Scorp1o117/FIT-GGUF) v0.3.

Each tier is the **smallest measured artifact that reaches its KL anchor** —
solved by bisecting the size bracket, not read off a preset ladder. Together they
are **78.55 GiB** where the smallest passing standard preset in each tier would be
**89.89 GiB**: **11.34 GiB (12.6%) smaller**.

| Tier | File | Size | macro KL | Gate | Same-top |
| --- | --- | --- | --- | --- | --- |
| Reference | `…-FIT-REFERENCE-24G-Q5_K_M.gguf` | **23.55 GiB** | **0.0195** | ≤ 0.02 ✅ | 95.40% |
| Quality | `…-FIT-QUALITY-17G-IQ4_XS.gguf` | **17.50 GiB** | **0.0464** | ≤ 0.05 ✅ | 92.26% |
| Balanced | `…-FIT-BALANCED-14G-IQ3_S.gguf` | **13.80 GiB** | **0.0959** | ≤ 0.10 ✅ | 88.83% |
| Compact | `…-FIT-COMPACT-13G-IQ3_XXS.gguf` | **12.76 GiB** | **0.1358** | ≤ 0.15 ✅ | 86.15% |
| Mini | `…-FIT-MINI-11G-IQ2_S.gguf` | **10.94 GiB** | **0.1992** | ≤ 0.20 ✅ | 83.18% |

**The gate is KL alone.** Since v0.3 a tier is one fixed, model-independent
number; same-top agreement is measured, archived and reported, but it never
decides a verdict. It is shown here without a floor column because the shipped
tiers are **not policy-uniform** — see *Planning policy* below — and a floor
reading would silently compare artifacts built different ways.

Sizes are the exact byte counts of the files below; every one matched its
re-quantization prediction byte for byte (the G2 exact-size gate).

## Measured five-domain results (eval-v1, frozen protocol)

Each column is `mean KL / same-top %`.

| Tier | wiki_test | wiki_valid | chinese | code | agent_chat | macro KL |
| --- | --- | --- | --- | --- | --- | --- |
| Reference | .0160 / 95.14 | .0116 / 95.36 | .0204 / 94.90 | .0354 / 95.97 | .0140 / 95.63 | **.0195** |
| Quality | .0425 / 91.58 | .0347 / 92.18 | .0502 / 91.94 | .0668 / 93.43 | .0378 / 92.17 | **.0464** |
| Balanced | .0934 / 87.97 | .0749 / 88.25 | .1002 / 88.25 | .1268 / 90.37 | .0841 / 89.34 | **.0959** |
| Compact | .1406 / 84.57 | .1124 / 85.36 | .1427 / 85.23 | .1583 / 88.57 | .1249 / 87.05 | **.1358** |
| Mini | .2257 / 81.28 | .1987 / 81.37 | .1940 / 82.48 | .2123 / 86.19 | .1652 / 84.59 | **.1992** |

These are the numbers for **the files that ship**: each artifact was
re-evaluated on its own bytes after quantization, and every one reproduced the
search-time KL exactly.

## Provenance

- Source: `occamy-1.0-abliterated-BF16.gguf`,
  sha256 `b77f117553d1106e56fd14cbc95f6a92022f98bb0bd385735805412ef8f78856`
- Calibration contract: `fidelity-calibration-v1`; evaluator: `eval-v1`
- imatrix: `occamy-1.0-abliterated-BF16-imatrix.gguf`, 500 chunks, 510 entries

| Tier | sha256 | Recipe point |
| --- | --- | --- |
| Reference | `52f04da488c27b7d26c16830d8bcfc194d9d319ac93cf04bc04bcbf933f0e6e0` | `refwin-23.6G` |
| Quality | `65e30676451f99cf1db6b2514f64643e9faad4edf5e04109339e1a75d751cf4a` | `floorq-17.5G` |
| Balanced | `80a512a7bad6b7acc75207e961f20dc68cf7ff6e20aff86efe127519804dd7ce` | `floor5-13.8G` |
| Compact | `a6eefab58c674c129dca5cf27eb9f1a3b26d2d5611ad43ed309f2e4796daaac5` | `tier-compact-s1` |
| Mini | `e55732e59372e9b55ce6434aa18c189cb6accc47e2a09ac6416a2f0c8849e7b7` | `tier-mini-s3` |

Every `.gguf` ships with three sidecars:

- `.plan.json` — the size-exact plan (target, predicted size, dominant type)
- `.tensor-types.txt` — the effective recipe, the thing that reproduces the file
- `.quantize-record.json` — the exact `llama-quantize` invocation, the imatrix
  argument it used, and the re-finalized prediction it matched

`emit-report.json` collects all of the above, including the per-domain metrics.

### Planning policy

The ledger records, per point, whether precision floors were active **and** which
floor policy produced it (a digest of the table plus a semantics version). The
shipped tiers were not all planned the same way, and that is the intended
behaviour rather than an oversight:

> a tier's product is the smallest artifact that reaches its anchor **and can be
> rebuilt from the bundle**. That artifact does not care which policy planned it.
> Policy governs how *probes* are planned — where the search looks next — and the
> search filters its bracket by it, because points planned by different rules do
> not share a curve.

Filtering *selection* by policy instead silently discarded smaller passing
artifacts: with floors on, mini and compact were both reported about 0.2 GiB
larger than passing artifacts already in the ledger, purely because those had
been planned before the floors existed.

### Reproducing a tier

```bash
fit quantize \
  --analysis <bundle>/analysis/Q5_K_M-Q8_0/analysis.json \
  --tensor-types occamy-1.0-abliterated-FIT-REFERENCE-24G-Q5_K_M.gguf.tensor-types.txt \
  --out reference.gguf
```

`fit quantize` re-finalizes its own size prediction from the invocation it is
about to run and refuses to return if the output misses it.

### A note on the byte counts

The search measured a probe artifact built with the search's own imatrix
argument string; these files let `analysis.json` supply it. llama.cpp stores
`quantize.imatrix.file` verbatim, so the two differ by up to 32 bytes of
metadata — the quantized tensors are identical, which the exact KL agreement
above confirms, and each shipped file was separately re-measured and passed its
own exact-size gate.

## How these five were solved

`fit calibrate` fills each tier's window by probing the **largest uncovered KL
gap** in it. That serves the *floor derivation* — it wants enough samples to take
a stable P5 — and it is blind to what the tier is actually for. A tier's product
is the smallest artifact that reaches the anchor, and that artifact generally
lies **between** two ladder presets.

`fit tier-search` bisects each tier's size bracket instead, evaluates the
candidates, and keeps the smallest PASS with its recipe.

### The window is a palette, not a bracket

**A candidate set is the union of the two endpoint recipes' types.** A tensor
typed *identically* at both ends never becomes a candidate, and one transition
per tensor is all a window can ever offer — so a window whose endpoints differ in
one type can only *interpolate* between them. It cannot re-allocate.

The Reference tier is where that stopped being an abstraction. Bracketing the
0.02 anchor gave `Q4_K_M`–`Q6_K`, whose entire candidate set is 371 tensors with
exactly one transition each, `q4_k → q6_k`:

| Probe | Size | macro KL | Verdict |
| --- | --- | --- | --- |
| `tier-reference-s4` | 26.21 GiB | 0.0224 | FAIL |
| `tier-reference-s5` | 26.34 GiB | 0.0220 | FAIL |
| `tier-reference-s6` | 26.40 GiB | 0.0217 | FAIL |
| `Q6_K` (preset) | 26.56 GiB | 0.0175 | PASS |

Three probes and the anchor never came into reach; the tier fell back to the
ladder's own `Q6_K`. A `Q5_K_M`–`Q8_0` window offers **432 candidates across
every role**, and one probe reached **23.55 GiB @ 0.0195** — 3.00 GiB under the
preset, from a different window and nothing else.

### Precision floors: where the candidate set is blind, and only there

`generate_upgrade_candidates` admits **only positive-size lower-to-upper
transitions**, so a tensor typed identically at both ends of a window is
unreachable at any budget, under any policy. In an `IQ2_XS`→`IQ3_XXS` window that
is the always-active machinery: attention, SSM and embedding tensors are typed
the same at both ends, so no plan can protect them however much budget it has.

`--always-active-floors` forces a floor on exactly those tensors. It is billed as
ordinary spending, and the oracle loop re-selects on measured overshoot, so the
budget is **redirected, never increased**.

Where the candidate set **can** reach a tensor, a floor is not a guarantee — it is
a constraint on an optimizer that could have spent there by itself. Counting the
floored roles each window's candidate set covers:

| window | floored roles reached | what the floors were worth |
| --- | --- | --- |
| `IQ3_XS`–`IQ3_M` | 5 of 10 | **−20.8% macro KL** at a fixed 14.10 GiB |
| `Q3_K_M`–`Q4_K_M` | 9 of 10 | about **+1.2 GiB**, no gain |
| `IQ4_XS`–`Q4_K_M` | 9 of 10 | about **+1.2 GiB**, no gain |

The same table was rescuing the low tiers and taxing the high ones, so a floor now
applies only where the blindness it exists for is real.

### Two floors were lifted from a published, hand-audited recipe

IsValorum's APEX-I-MiniPlus-V2.1 publishes a tensor-by-tensor map. Compared
against our own solve, six of its eight rules were already matched or exceeded
(shared experts `Q6_K` against its `Q5_K`; `attn_qkv`/`ssm_beta`/`ssm_out` at
`Q4_K` against its `Q3_K`; norms, routers and `output.weight` identical), and its
**core-versus-edge expert tiering** — higher precision on layers 10–29 — is what
our own oracle had already discovered independently. Two rules sat above us and
were cheap enough to be free: the DeltaNet recurrence decay `ssm_alpha` at
**F32** (+0.2 MiB per layer) and the attention gates `attn_gate` at **Q8_0**
(+4 MiB per layer), about **126 MiB** across the model.

### A floor used to be silently defeated by the plan it was correcting

`llama-quantize` resolves a tensor-type file by **first match**: a second line for
the same tensor is not a merge, it is dead text, and the earlier one takes the
tensor. The floor application *appended* its raise instead of replacing the
optimizer's pick, so any tensor the optimizer had also chosen produced two lines
and the floor lost.

Auditing the tiers that had shipped before the fix, against their own recipes:

| Tier | tensors where the floor lost | shipped instead of |
| --- | --- | --- |
| Reference | 0 | — |
| Quality | 100 | `ffn_*_shexp` at `q4_k` instead of `q6_k` |
| Balanced | 85 | `ffn_*_shexp` at `iq3_s` instead of `q6_k` |
| Compact | 258 | `shexp`/`attn_gate`/`ssm_alpha` at `iq3_s` |
| Mini | 260 | `shexp`/`attn_gate` at `iq3_xxs` |

The bug is fixed (a floor replaces the pick, and the writer deduplicates), and the
re-solve above is the result. The measured KL of the *earlier* files was never
wrong — every artifact is evaluated on the bytes that ship — only the policy
description was.

### "Smallest" is resolved to within a megabyte

A pure size ordering is not stable at the resolution the search decides at. The
quality tier had a passing point **295 KiB** (0.0016%) under its incumbent at
**1.7% worse macro KL**, and size alone picked it. The tie window is one
megabyte — deliberately far below the bisection tolerance of 128 MiB, because a
window that wide would quietly overturn a tier's definition: on the mini tier it
would have traded 113 MiB (1% of the file) for 2% of KL.
