# occamy-1.0-abliterated — FIT tiers

Four FIT (Fidelity-Interpolated Tensor) tiers for
`occamy-1.0-abliterated`, produced with
[FIT-GGUF](https://github.com/Scorp1o117/FIT-GGUF) v0.3.

Each tier is the **smallest artifact that reaches its KL anchor** — solved by
bisecting the size bracket, not read off a preset ladder. Together they are
**56.09 GiB** where the smallest passing standard preset in each tier would be
**63.33 GiB**: **7.24 GiB (11.4%) smaller**.

| Tier | File | Size | macro KL | Gate | Same-top | Floor ref. |
| --- | --- | --- | --- | --- | --- | --- |
| Quality | `…-FIT-QUALITY-18G-IQ4_XS.gguf` | **17.50 GiB** | **0.0463** | ≤ 0.05 ✅ | 92.25% | 91.27% ✅ |
| Balanced | `…-FIT-BALANCED-15G-IQ3_S.gguf` | **14.89 GiB** | **0.0999** | ≤ 0.10 ✅ | 88.33% | 88.03% ✅ |
| Compact | `…-FIT-COMPACT-13G-IQ3_XXS.gguf` | **12.76 GiB** | **0.1358** | ≤ 0.15 ✅ | 86.15% | 84.42% ✅ |
| Mini | `…-FIT-MINI-11G-IQ2_S.gguf` | **10.94 GiB** | **0.1992** | ≤ 0.20 ✅ | 83.18% | 83.56% ⚠️ |

**The gate is KL alone.** Since v0.3 a tier is one fixed, model-independent
number; same-top agreement is measured, archived and shown against this model's
calibrated floor, but it never decides a verdict. Mini is the honest case: it
clears the 0.20 KL anchor and misses the model's 83.56% floor by 0.38 points —
reported, not hidden, and not a gate.

Sizes are the exact byte counts of the files below; every one matched its
re-quantization prediction byte for byte (the G2 exact-size gate).

## Measured five-domain results (eval-v1, frozen protocol)

Each column is `mean KL / same-top %`.

| Tier | wiki_test | wiki_valid | chinese | code | agent_chat | macro KL |
| --- | --- | --- | --- | --- | --- | --- |
| Quality | .0424 / 91.33 | .0354 / 92.00 | .0485 / 91.86 | .0666 / 93.54 | .0387 / 92.51 | **.0463** |
| Balanced | .0952 / 87.75 | .0823 / 87.55 | .1098 / 87.29 | .1259 / 90.16 | .0862 / 88.91 | **.0999** |
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
| Quality | `4374bbcc0633a5931f781627c74131c2cf25f7fcadd11506289fe1188ba48d89` | `tier-quality-s5` |
| Balanced | `528eecb3634831114c872e407e572e873a06daa61262770fedf24c8ed03032b1` | `tier-balanced-s3` |
| Compact | `a6eefab58c674c129dca5cf27eb9f1a3b26d2d5611ad43ed309f2e4796daaac5` | `tier-compact-s1` |
| Mini | `e55732e59372e9b55ce6434aa18c189cb6accc47e2a09ac6416a2f0c8849e7b7` | `tier-mini-s3` |

Every `.gguf` ships with three sidecars:

- `.plan.json` — the size-exact plan (target, predicted size, dominant type)
- `.tensor-types.txt` — the effective recipe, the thing that reproduces the file
- `.quantize-record.json` — the exact `llama-quantize` invocation, the imatrix
  argument it used, and the re-finalized prediction it matched

`emit-report.json` collects all of the above, including the per-domain metrics.

### Reproducing a tier

```bash
fit quantize \
  --analysis <bundle>/analysis/IQ4_XS-Q4_K_M/analysis.json \
  --tensor-types occamy-1.0-abliterated-FIT-QUALITY-18G-IQ4_XS.gguf.tensor-types.txt \
  --out quality.gguf
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

## How these four were solved

`fit calibrate` fills each tier's window by probing the **largest uncovered KL
gap** in it. That serves the *floor derivation* — it wants enough samples to take
a stable P5 — and it is blind to what the tier is actually for. A tier's product
is the smallest artifact that reaches the anchor, and that artifact generally
lies **between** two ladder presets.

For Quality the window held `IQ4_XS` (17.44 GiB, KL 0.0503 — a FAIL) and
`Q4_K_M` (19.71 GiB, KL 0.0471). The answer lives in the KL gap
`0.0471..0.0503`; the largest gap was `0.0503..0.0575`, so the single window
probe went **above** `IQ4_XS`, produced a *smaller* artifact with a *worse* KL
(17.27 GiB @ 0.0555), failed the anchor, and the tier fell back to the
`Q4_K_M` preset — 1.71 GiB larger than the minimum PASS on the same curve.

`fit tier-search` (v0.3) bisects each tier's size bracket instead, evaluates the
candidates, and keeps the smallest PASS with its recipe. It landed Quality at
**17.50 GiB, KL 0.0463** — 2.21 GiB under that preset.
