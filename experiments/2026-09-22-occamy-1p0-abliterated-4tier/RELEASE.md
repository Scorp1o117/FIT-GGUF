# occamy-1.0-abliterated — FIT tiers

Five FIT (Fidelity-Interpolated Tensor) tiers for `occamy-1.0-abliterated`,
produced with [FIT-GGUF](https://github.com/Scorp1o117/FIT-GGUF) v0.3.

Each tier is the **smallest artifact that reaches its KL anchor** — solved by
bisecting the size bracket, not read off a preset ladder. Together they are
**78.81 GiB** where the smallest passing standard preset in each tier would be
**89.89 GiB**: **11.08 GiB (12.3%) smaller**.

| Tier | File | Size | macro KL | Gate | Same-top | Floor ref. |
| --- | --- | --- | --- | --- | --- | --- |
| Reference | `…-FIT-REFERENCE-24G-Q5_K_M.gguf` | **23.55 GiB** | **0.0195** | ≤ 0.02 ✅ | 95.40% | — |
| Quality | `…-FIT-QUALITY-17G-IQ4_XS.gguf` | **17.50 GiB** | **0.0464** | ≤ 0.05 ✅ | 92.26% | 91.27% ✅ |
| Balanced | `…-FIT-BALANCED-14G-IQ3_S.gguf` | **13.80 GiB** | **0.0959** | ≤ 0.10 ✅ | 88.83% | 88.03% ✅ |
| Compact | `…-FIT-COMPACT-13G-IQ3_XXS.gguf` | **12.78 GiB** | **0.1406** | ≤ 0.15 ✅ | 85.71% | 84.42% ✅ |
| Mini | `…-FIT-MINI-11G-IQ2_XS.gguf` | **11.18 GiB** | **0.1990** | ≤ 0.20 ✅ | 83.22% | 83.56% ⚠️ |

**The gate is KL alone.** Since v0.3 a tier is one fixed, model-independent
number; same-top agreement is measured, archived and shown against this model's
calibrated floor, but it never decides a verdict. Mini is the honest case: it
clears the 0.20 KL anchor and misses the model's 83.56% floor by 0.34 points —
reported, not hidden, and not a gate. Reference has no calibrated floor because
this model's four-tier calibration predates the tier; it is planned with
`floor: None` and reports `—`.

Sizes are the exact byte counts of the files below; every one matched its
re-quantization prediction byte for byte (the G2 exact-size gate).

## Measured five-domain results (eval-v1, frozen protocol)

Each column is `mean KL / same-top %`.

| Tier | wiki_test | wiki_valid | chinese | code | agent_chat | macro KL |
| --- | --- | --- | --- | --- | --- | --- |
| Reference | .0160 / 95.14 | .0116 / 95.36 | .0204 / 94.90 | .0354 / 95.97 | .0140 / 95.63 | **.0195** |
| Quality | .0425 / 91.58 | .0347 / 92.18 | .0502 / 91.94 | .0668 / 93.43 | .0378 / 92.17 | **.0464** |
| Balanced | .0934 / 87.97 | .0749 / 88.25 | .1002 / 88.25 | .1268 / 90.37 | .0841 / 89.34 | **.0959** |
| Compact | .1477 / 84.35 | .1231 / 84.77 | .1522 / 84.71 | .1568 / 88.46 | .1234 / 86.26 | **.1406** |
| Mini | .2277 / 81.37 | .1892 / 82.00 | .1914 / 82.48 | .2184 / 85.81 | .1680 / 84.42 | **.1990** |

These are the numbers for **the files that ship**: each artifact was
re-evaluated on its own bytes after quantization, and every one reproduced the
search-time KL exactly.

## Provenance

- Source: `occamy-1.0-abliterated-BF16.gguf`,
  sha256 `b77f117553d1106e56fd14cbc95f6a92022f98bb0bd385735805412ef8f78856`
- Calibration contract: `fidelity-calibration-v1`; evaluator: `eval-v1`
- imatrix: `occamy-1.0-abliterated-BF16-imatrix.gguf`, 500 chunks, 510 entries
- Precision floors: **on** for every tier; the same-top floors of the four
  calibrated tiers are the reference values above, not gates

| Tier | sha256 | Recipe point |
| --- | --- | --- |
| Reference | `52f04da488c27b7d26c16830d8bcfc194d9d319ac93cf04bc04bcbf933f0e6e0` | `refwin-23.6G` |
| Quality | `65e30676451f99cf1db6b2514f64643e9faad4edf5e04109339e1a75d751cf4a` | `floorq-17.5G` |
| Balanced | `80a512a7bad6b7acc75207e961f20dc68cf7ff6e20aff86efe127519804dd7ce` | `floor5-13.8G` |
| Compact | `e11f2ddd442399c199de1b8c7d02feee2a9f25b16e975c6a1f6749fb9aac3fe5` | `tier-compact-s4` |
| Mini | `c0ffc5694eca5c9482f04491e28579ac72fccec09589f0b75292f59995a84e20` | `tier-mini-s5` |

Every `.gguf` ships with three sidecars:

- `.plan.json` — the size-exact plan (target, predicted size, dominant type)
- `.tensor-types.txt` — the effective recipe, the thing that reproduces the file
- `.quantize-record.json` — the exact `llama-quantize` invocation, the imatrix
  argument it used, and the re-finalized prediction it matched

`emit-report.json` collects all of the above, including the per-domain metrics.

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
exactly one transition each, `q4_k → q6_k`. Its probes:

| Probe | Size | macro KL | Verdict |
| --- | --- | --- | --- |
| `tier-reference-s4` | 26.21 GiB | 0.0224 | FAIL |
| `tier-reference-s5` | 26.34 GiB | 0.0220 | FAIL |
| `tier-reference-s6` | 26.40 GiB | 0.0217 | FAIL |
| `Q6_K` (preset) | 26.56 GiB | 0.0175 | PASS |

Three probes, 0.35 GiB of budget, and the anchor never came into reach — the
window's best move is to downgrade whole tensors from `Q6_K` to `Q4_K`, and that
costs more than it saves. The tier fell back to the ladder's own `Q6_K`.

A `Q5_K_M`–`Q8_0` window offers **432 candidates across every role**. One probe:

| Probe | Size | macro KL | Verdict |
| --- | --- | --- | --- |
| `refwin-23.6G` | **23.55 GiB** | **0.0195** | **PASS** |

**3.00 GiB under the preset, from a different window and nothing else** — same
model, same imatrix, same evaluator, same budget. The tier is a real FIT
artifact rather than a preset with a FIT label.

The window is also how this release answers the question that started it: the
external `APEX-I-Balanced` recipe reaches 23.60 GiB @ 0.0197 with a hand-written
palette (`Q5_K` experts, `Q8_0` shared experts, `Q6_K` attention/SSM). FIT's
`Q5_K_M`–`Q8_0` window found **23.55 GiB @ 0.0195** — 0.05 GiB smaller and 0.0002
lower KL. The comparison is close enough that it should be read as a tie between
two different searches, not a win; what it does show is that the earlier 26.56
GiB result was a property of the window, not of the model.

```bash
# the window, and the one measurement that decided the tier
fit analyze --source occamy-1.0-abliterated-BF16.gguf --imatrix …-imatrix.gguf \
  --runtime tools/llama-b10666-rocm --lower Q5_K_M --upper Q8_0 \
  --out-dir <bundle>/analysis/Q5_K_M-Q8_0
python3 scripts/sweep_sizes.py --analysis <bundle>/analysis/Q5_K_M-Q8_0/analysis.json \
  --runtime tools/llama-b10666-rocm --eval-data eval-data --refs-dir <refs> \
  --sizes 23.6 --model-name occamy-1.0-abliterated --imatrix-arg "<imatrix arg>" \
  --dest <out> --floors --tag refwin --curve-bundle <bundle>

# record the verdict without paying for probes
fit tier-search --bundle <bundle> --tiers reference --budget 0 --always-active-floors
```

### Precision floors

`generate_upgrade_candidates` admits **only positive-size lower-to-upper
transitions**, so a tensor typed identically at both ends of a window is
unreachable at any budget, under any policy. In an `IQ2_XS`→`IQ3_XXS` window
that is the always-active machinery: attention, SSM and embedding tensors are
typed the same at both ends, so no plan can protect them however much budget it
has.

`--always-active-floors` forces a floor on exactly those tensors (`ffn_*_shexp`
→ `Q6_K`; `attn_q/k/gate`, `ssm_{alpha,beta,out}`, `token_embd.weight` →
`Q4_K`). It is billed as ordinary spending, and the oracle loop re-selects on
measured overshoot, so the budget is **redirected, never increased**. At a fixed
size on this model the floors are worth **−16.4% macro KL** (14.90 GiB @ 0.0835
with, vs 14.89 GiB @ 0.0999 without).

They are not free at the bottom of the ladder: Mini is 11.18 GiB under the floors
where the pre-floor solve reached 10.94 GiB at the same KL, and Compact is 12.78
against 12.76. The floors buy protection the five-domain KL does not fully price
— they are what makes a lower base preset safe to use at all — and the trade is
recorded here rather than netted out silently.

### A lower base preset

A plan can only spend `target − lower_preset`, so anchoring one rung lower buys a
full ladder step of headroom at no cost to the target. Balanced moved from an
`IQ3_M`-based window to `IQ3_XS`/`IQ3_XXS` and found **13.80 GiB** where the
adjacent pair had bottomed out at 14.89.

The rule is a widening, not a guarantee: a lower base is a *wider* window with a
coarser candidate set, and on this model the `IQ3_XXS`-based probes at 13.64 and
13.71 GiB both failed (0.1174, 0.1161) where the `IQ3_XS`-based sweep at 13.80
GiB passed (0.0959). The tier was won by the sweep, not by the search's own
probes.

### Why the balanced tier had to be re-solved

The first five-tier pass reported balanced at 14.89 GiB @ 0.0999 and quality at
19.13 GiB @ 0.0490. Both were wrong, and both were wrong the same way: a curve
point measured *before* the floors existed was still in the ledger, and because
the search ranks by size, the pre-floor point outranked the floor artifact that
actually wins. Balanced's 14.89 GiB point sat 0.01 GiB under a 14.90 GiB floor
point that was 0.0054 KL better, and the search then read its own bracket as
"under tolerance" and stopped.

Points now carry the floor regime they were planned under, and a point measured
under another policy is not a bound — it is a different experiment that happens
to share a name. Re-solved: balanced **13.80 GiB @ 0.0959** (1.09 GiB smaller),
quality **17.50 GiB @ 0.0464** (1.63 GiB smaller).
