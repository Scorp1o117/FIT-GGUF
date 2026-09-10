# Calibration report — spark-x25-4b-abliterated

- contract: fidelity-calibration-v1 (455338c52de7aa1b…)
- source: aa73aeb45870f7eb…
- overall: **candidate**
- open failures: ['INSUFFICIENT_WINDOW', 'WITNESS_MISSING']

| tier | window | n | floor | method | witness | status |
|---|---|---|---|---|---|---|
| quality | [0.0425, 0.0575] | 3 | 0.9057 | empirical_p5 | ✓ | validated |
| balanced | [0.085, 0.115] | 1 | 0.8699 | min_fallback | ✗ | candidate |
| compact | [0.1275, 0.1725] | 3 | 0.8346 | empirical_p5 | ✓ | validated |
| mini | [0.17, 0.23] | 2 | 0.8145 | min_fallback | ✗ | candidate |
