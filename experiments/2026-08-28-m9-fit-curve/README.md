# M9 First Real FIT Curve

Date: 2026-08-28

## Scope

The formal curve uses one current-weight BF16 source, one canonical imatrix,
and pinned llama.cpp build 10666 for every point:

```text
IQ3_M / FIT-25 / FIT-50 / FIT-75 / IQ4_XS
```

FIT targets are the exact 25%, 50%, and 75% byte positions between the actual
IQ3_M and IQ4_XS artifact sizes. The plans use the deterministic M7 utility
rule and retain IQ3_M qtypes for all rejected non-promotion transitions.

## Size and quantization results

| Variant | Target bytes | Predicted bytes | Actual bytes | Unused bytes | Overrides | Time (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| IQ3_M | 12,580,875,232 | 12,580,875,232 | 12,580,875,232 | 0 | 0 | 279.163 |
| FIT-25 | 13,206,283,232 | 13,203,487,712 | 13,203,487,712 | 2,795,520 | 235 | 265.200 |
| FIT-50 | 13,831,691,232 | 13,831,486,432 | 13,831,486,432 | 204,800 | 323 | 272.160 |
| FIT-75 | 14,457,099,232 | 14,456,126,432 | 14,456,126,432 | 972,800 | 404 | 270.250 |
| IQ4_XS | 15,082,507,232 | 15,082,507,232 | 15,082,507,232 | 0 | preset | 279.138 |

Every intermediate artifact is below target and already meets the M11
fine-fill target of `unused <= max(16 MiB, 0.1%)`. Each actual size equals its
prediction exactly. Dry-run and output GGUF comparison found 851/851 matching
tensor names and zero qtype mismatches at every intermediate point.

## Artifact hashes

- FIT-25: `d757266985bdbfe8a2df7a1d6f209effaf192036ece6be9b3391cb3c2dcef4e2`
- FIT-50: `e4fe1c46ab89c8b6343203168ebeec699372c2fb21f411c61c47edc2e1f33306`
- FIT-75: `4a05fccd1b0f77c51ff8d7f4be43663e85adc07f7f6ab1eece0ce5f14f00fad1`

Recipe JSON and tensor override files are tracked in the M7 experiment
directory. Full dry-run and quantization logs are preserved locally there
under `artifacts/logs/` and excluded from Git.

## Quality protocol

Existing reference logits were rejected because they predate the current
Huihui weights. Five fresh references were generated from the exact BF16 GGUF
with SHA-256
`8a033407c8f58d43102aade25b973cc6d2f2ce5c5cbf4dc75a2cdb60b9e33cbc`.

Every reference and comparison used pinned `llama-perplexity` build 10666 with
`-ngl 99 -t 16 -c 512 -b 512`. The five input files are fixed 64 KiB slices
covering wiki_test, wiki_valid, Chinese, code, and agent_chat. Their hashes and
all parsed measurements are preserved in `curve-results.json`; full raw logs
and large KLD files remain local under `artifacts/`.

| Domain | BF16 PPL | KLD SHA-256 |
| --- | ---: | --- |
| wiki_test | 5.6136 +/- 0.14999 | `81afde2b749b13255ef42afb64fbb23d0812714223867faebb73838ac75ad266` |
| wiki_valid | 5.4626 +/- 0.14533 | `128022a840f3fb79739c218b4448a06dee70a5e91a8442bd6bab67839a67f4d6` |
| Chinese | 10.6144 +/- 0.42124 | `f3355aae5950a915765d7530586fe7be2637c58e315eff699328dae19946ab90` |
| code | 4.9591 +/- 0.15244 | `9b4eaeaedac9fa5556f3a2e265e522fcbeab630e582b96cd283a1154b6323651` |
| agent_chat | 6.7943 +/- 0.21524 | `007eedc7dc859910f24ee3054ae8b42ae7799da0ca317eab5fa29a11a2dfe856` |

## Quality results

Mean KL divergence (lower is better):

| Domain | IQ3_M | FIT-25 | FIT-50 | FIT-75 | IQ4_XS |
| --- | ---: | ---: | ---: | ---: | ---: |
| wiki_test | 0.060566 | 0.038943 | 0.031555 | 0.022408 | 0.017959 |
| wiki_valid | 0.057070 | 0.037375 | 0.030332 | 0.021560 | 0.016231 |
| Chinese | 0.252486 | 0.193422 | 0.181044 | 0.144548 | 0.108121 |
| code | 0.190477 | 0.153504 | 0.138484 | 0.132087 | 0.093160 |
| agent_chat | 0.144890 | 0.108142 | 0.105204 | 0.077999 | 0.053897 |
| Macro mean | 0.141098 | 0.106277 | 0.097324 | 0.079720 | 0.057874 |

Same-top percentage (higher is better):

| Domain | IQ3_M | FIT-25 | FIT-50 | FIT-75 | IQ4_XS |
| --- | ---: | ---: | ---: | ---: | ---: |
| wiki_test | 89.956 | 91.701 | 92.397 | 93.801 | 94.333 |
| wiki_valid | 90.487 | 92.473 | 92.865 | 93.650 | 94.497 |
| Chinese | 87.229 | 88.693 | 90.078 | 91.529 | 92.523 |
| code | 90.208 | 91.507 | 92.292 | 93.186 | 93.971 |
| agent_chat | 89.400 | 91.503 | 91.919 | 93.440 | 93.999 |
| Macro mean | 89.456 | 91.175 | 91.910 | 93.121 | 93.865 |

Mean quantized PPL:

| Domain | IQ3_M | FIT-25 | FIT-50 | FIT-75 | IQ4_XS |
| --- | ---: | ---: | ---: | ---: | ---: |
| wiki_test | 5.863777 | 5.744649 | 5.730317 | 5.667801 | 5.669252 |
| wiki_valid | 5.648473 | 5.568069 | 5.544453 | 5.465418 | 5.496144 |
| Chinese | 11.687543 | 10.322706 | 10.997307 | 10.836222 | 10.999479 |
| code | 5.291320 | 4.881605 | 5.042097 | 5.181511 | 5.186846 |
| agent_chat | 7.404830 | 6.810028 | 7.015584 | 6.878413 | 6.772951 |

## Interpretation

All five domains show strictly decreasing mean KL and strictly increasing
Same-top at every size step. The macro KL improves by 24.7% from IQ3_M to
FIT-25, 8.4% from FIT-25 to FIT-50, 18.1% from FIT-50 to FIT-75, and 27.4% from
FIT-75 to IQ4_XS. This accepts the M9 claim that additional FIT budget produces
a broadly monotonic empirical quality curve without domain regression on the
chosen primary distribution metrics.

PPL point estimates are not strictly monotonic in Chinese and code. Those
changes overlap the reported per-run uncertainty, and the corresponding KL and
Same-top values retain the expected direction. They remain explicit M10
diagnostic targets rather than evidence for or against the allocation proxy.

Reference generation took 343.96 seconds and the 25 comparison runs took
1,676.97 seconds, excluding model quantization. M9 does not establish that the
imatrix allocation beats a matched-size naive or random allocation; controlled
M10 ablations are required before making that claim.

## Acceptance

M9 is accepted for this model, source hash, canonical imatrix, pinned runtime,
five fixed data slices, and the stated parameters. It demonstrates exact size
control and a five-domain monotonic quality curve. Generalization to another
model family and positive allocation evidence remain open.
