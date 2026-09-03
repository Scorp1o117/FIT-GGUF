# M2 candidate-calibration-v1 — per-model tables

Verdicts are mechanical prereg §7 evidence; final judgment: planner.

## orcarouter  (points 47/47 valid, 0 flagged)

| target | window | n(win) | Top@t interp | P5 | P10 | candidate | verdict |
|---|---|---:|---:|---:|---:|---:|---|
| 0.05 | [0.0425, 0.0575] | 6 | 94.987 | 94.765 | 94.811 | 93 | supported |
| 0.10 | [0.0850, 0.1150] | 4 | 91.842 | 91.178 | 91.311 | 91 | supported |
| 0.15 | [0.1275, 0.1725] | 7 | 89.105 | 88.434 | 88.434 | 88 | supported |
| 0.20 | [0.1700, 0.2300] | 8 | 86.320 | 85.153 | 85.707 | 85 | supported |

## granite  (points 26/26 valid, 0 flagged)

| target | window | n(win) | Top@t interp | P5 | P10 | candidate | verdict |
|---|---|---:|---:|---:|---:|---:|---|
| 0.05 | [0.0425, 0.0575] | 2 | 93.814 | 93.771 | 93.783 | 93 | supported |
| 0.10 | [0.0850, 0.1150] | 3 | 90.374 | 90.306 | 90.324 | 91 | violated |
| 0.15 | [0.1275, 0.1725] | 4 | 88.389 | 87.143 | 87.177 | 88 | at-risk |
| 0.20 | [0.1700, 0.2300] | 3 | 85.578 | 85.860 | 85.905 | 85 | supported |

## ling  (points 28/28 valid, 0 flagged)

| target | window | n(win) | Top@t interp | P5 | P10 | candidate | verdict |
|---|---|---:|---:|---:|---:|---:|---|
| 0.05 | [0.0425, 0.0575] | 1 | 91.519 | — | — | 93 | indeterminate (window n<2) |
| 0.10 | [0.0850, 0.1150] | 2 | 87.913 | 87.287 | 87.348 | 91 | violated |
| 0.15 | [0.1275, 0.1725] | 2 | 85.283 | 85.178 | 85.205 | 88 | violated |
| 0.20 | [0.1700, 0.2300] | 2 | 82.758 | 82.197 | 82.245 | 85 | violated |

## gemma  (points 20/20 valid, 0 flagged)

| target | window | n(win) | Top@t interp | P5 | P10 | candidate | verdict |
|---|---|---:|---:|---:|---:|---:|---|
| 0.05 | [0.0425, 0.0575] | 2 | 90.710 | 90.676 | 90.689 | 93 | violated |
| 0.10 | [0.0850, 0.1150] | 2 | 86.949 | 86.483 | 86.484 | 91 | violated |
| 0.15 | [0.1275, 0.1725] | 2 | 84.170 | 84.106 | 84.116 | 88 | violated |
| 0.20 | [0.1700, 0.2300] | 2 | 81.714 | 82.157 | 82.169 | 85 | violated |

## Layer 3 — equal-model aggregate

| target | candidate | mean Top@t | mean P5 | mean P10 | n models (top/p10) |
|---|---:|---:|---:|---:|---|
| 0.05 | 93 | 92.757 | 93.070 | 93.094 | 4/3 |
| 0.10 | 91 | 89.270 | 88.814 | 88.867 | 4/4 |
| 0.15 | 88 | 86.737 | 86.215 | 86.233 | 4/4 |
| 0.20 | 85 | 84.092 | 83.842 | 84.006 | 4/4 |
