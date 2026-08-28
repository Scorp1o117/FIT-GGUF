# M5-M8 Planning and First FIT Artifact

Date: 2026-08-28

## Baseline selection

The planner sorts exact artifact sizes, not preset names or caller order.

| Preset | Exact bytes |
| --- | ---: |
| IQ3_S | 12,419,328,992 |
| IQ3_M | 12,580,875,232 |
| IQ4_XS | 15,082,507,232 |

For a 13 GiB target, M5 selected IQ3_M as lower, IQ4_XS as upper, and
1,377,768,480 bytes of extra budget.

## Candidate safety

M6 found 497 qtype differences between the two effective recipes. It retained
473 strict encoded-precision promotions and rejected 24 `Q4_K -> IQ4_XS`
transitions whose encoded size decreases. This preserves the lower baseline
tensor by tensor even though the upper preset is larger overall.

The 473 candidates contain 409 `IQ3_S -> IQ4_XS` and 64 `Q4_K -> Q5_K`
transitions. Of these, 472 have canonical imatrix profiles. The unprofiled
token embedding receives zero provisional utility and is considered last.

Candidate `expected_gain` is currently the within-role imatrix mean ratio
multiplied by encoded BPW gain. It is explicitly a deterministic search proxy,
not a measured quality improvement.

## Greedy plan

M7 selected 338 candidates with an exact aggregate cost of 1,375,211,520
bytes. The schema-v1 record is `fit-recipe-13GiB.json`; the exact-name
llama.cpp override file is `tensor-types-13GiB.txt`.

| Field | Bytes |
| --- | ---: |
| Target | 13,958,643,712 |
| Lower IQ3_M | 12,580,875,232 |
| Selected upgrades | 1,375,211,520 |
| Predicted output | 13,956,086,752 |
| Unused | 2,556,960 |

Unused budget is about 0.0183% of the target.

## Real quantization

M8 dry-run reproduced all 338 intended overrides with zero qtype mismatches.
The real output then matched the predicted byte size exactly and had 851/851
matching tensor names/qtypes.

- Artifact: `Huihui-Qwen3.8-27B-FIT-13GiB.gguf`
- Actual bytes: 13,956,086,752
- Size error versus predictor: 0
- SHA-256:
  `6f2a45fdcb616b714992a40b7f3391e0e35ba811f1d5f904aeff567549e849d9`
- Quantization time: 260.503 seconds
- Reported payload BPW: 4.15

Full dry-run and quantization logs are preserved locally under
`artifacts/logs/` and excluded from Git.

## Acceptance

M5-M8 pass for the first 13 GiB target. Budget selection is size-derived,
candidate generation never downgrades below the effective lower recipe, the
optimizer is deterministic and budget-safe, and the minimal tensor-type-file
integration produces the predicted real artifact.

No quality claim is made yet. M9 must generate more arbitrary-size points and
M10 must test whether the proxy ordering produces a useful quality curve.
