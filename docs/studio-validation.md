# FIT Studio validation · 2026-10-04

Local development build on Windows, Python 3.13.13, PyWebView 6.2.1 and
llmfit 1.1.16. Hardware: RTX 5080 Laptop GPU (16,303 MiB reported VRAM),
32 GiB physical RAM, Ryzen 9 7945HX. No model over 5B was downloaded or run.

## Automated checks

- Full Python suite: **307 passed, 3 skipped**.
- JavaScript syntax check: `node --check src/fit_gguf/studio/static/app.js`.
- `git diff --check` passed.
- Wheel inspected for Studio Python modules and all three static assets.
- Windows directory build contains its own Python runtime, worker entry point,
  llmfit binary, static assets and dependency license notices.

## Real model smoke

Used `ggml-org/models-moved` TinyStories `tinyllamas/stories15M.gguf`,
approximately 15M parameters, with the pinned source revision recorded locally.
Generated an imatrix with official llama.cpp **b10666**, CPU only, four threads,
128-token context and four chunks of the included KL evaluation corpus.

The source worker, browser workflow and packaged executable worker each
completed analyze → plan → quantize with Q4_K_M / Q8_0 endpoints:

| Measurement | Bytes |
| --- | ---: |
| Requested target | 23,829,056 |
| Predicted final size | 23,153,408 |
| Actual final size | 23,153,408 |
| Actual minus prediction | 0 |

All produced the same SHA-256:
`3b54e056a348b6c70dd7718e9df30078505db174c798d71450d073f97c9faf91`.
The resulting GGUF loaded in `llama-cli` and generated 32 tokens on CPU.
This establishes file compatibility and exact-size behavior on this small
fixture; it is **not** a quality evaluation or a performance benchmark.
The small tensor shapes required quantizer fallback for three tensors.

Reports and logs remain local under `work/real-studio-smoke`,
`work/frozen-final-smoke` and `work/studio`. Downloaded binaries and model data
are excluded from Git and the application archive.

## UI checks and limits

Browser controls exercised against the running local application: hardware
overview, model recommendation/search, analyze, plan, quantize, history,
precision distribution and restoration of the previous successful run.
Screenshots are retained locally at `work/studio-overview.png` and
`work/studio-workspace.png`.

The final Windows executable opens its native WebView2 window, loads its
bundled UI and llmfit integration, and restores the shared task history.
Native content was verified through the accessibility tree. The computer-use
capture tool returned black native screenshots and could not activate the
window for clicks, so native picker clicks were not verified end to end.
The picker bridge has an automated API test. Browser interaction checks and
the packaged worker smoke are separate from this native interaction limit.

This is an unsigned local build. Calibration, fidelity search and tier search
remain CLI workflows in the first Studio version.
