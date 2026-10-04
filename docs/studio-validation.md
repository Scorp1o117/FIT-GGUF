# FIT Studio validation · 2026-10-04

Local development build on Windows, Python 3.13.13, PyWebView 6.2.1 and
llmfit 1.1.16. Hardware: RTX 5080 Laptop GPU (16,303 MiB reported VRAM),
32 GiB physical RAM, Ryzen 9 7945HX. No model over 5B was downloaded or run.

## Automated checks

- Full Python suite after the two-entry redesign and desktop runtime fixes:
  **320 passed, 1 skipped**.
  Upstream b10666 header tests were enabled by downloading the two headers to
  the ignored reference directory.
- JavaScript syntax check: `node --check src/fit_gguf/studio/static/app.js`.
- `git diff --check` passed.
- Wheel inspected for Studio Python modules and all three static assets.
- Windows directory build contains its own Python runtime, worker entry point,
  llmfit binary, static assets and dependency license notices.
- GitHub Actions passed on Windows and Ubuntu with Python 3.11 and 3.13.
  UTF-8 log admission is also checked under a simulated legacy Windows locale.
  External runtime processes suppress Windows console creation, and frozen
  workers restore their DLL directory even if a subprocess fails.

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
An F16 source converted from the same fixture also passed the complete volume
workflow with the same predicted and actual file size. Preserved F16 tensors
have an explicit payload/alignment regression test.
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
Both workflows restore their own last successful inputs and results. An actual
quality task was cancelled through the UI; its Python/llama.cpp process tree
terminated and the UI showed cancelled with no delivered result. An invalid
FREEZE document failed before quantization and also showed no delivered result.
Screenshots are retained locally at `work/studio-overview.png` and
`work/studio-workspace.png`.

The final Windows executable opens its native WebView2 window, loads its
bundled UI and llmfit integration, and restores the shared task history.
Native content was verified through the accessibility tree. The computer-use
capture tool returned black native screenshots and could not activate the
window for clicks, so native picker clicks were not verified end to end.
Picker and completed-output folder actions have automated bridge tests.
Native folder-button clicks share the input-tool limit above. Browser checks and
the packaged worker smoke are separate from this native interaction limit.

## Real quality entry

Generated all five reference KLD files with the same TinyStories source using
the frozen 512-context evaluation protocol, CPU four threads. References occupy
approximately 2.97 GB of disk and were processed sequentially.

The source worker, packaged executable worker and browser quality-entry submission ran the
real fidelity-search product chain. Balanced was selected (macro KL ≤ 0.10).
Seven fresh search evaluations followed by re-evaluation of the final file
delivered:

- actual size **26,239,616 bytes**, with **zero** G2 prediction delta;
- final macro KL **0.018522**, Same-top **88.6446%** (reference metric);
- SHA-256 `a4ec8f09956db1e6095d45c0d91e254570ccf261ab50640955635ef1a9d1e8dd`.

This is a real workflow smoke using a small F32 source, not a recommendation or
quality benchmark for this model. The default 128 MiB tolerance is wider than
this tiny model's search bracket; it does not establish the global minimum
size. Reports are under `work/quality-real-smoke` and
`work/studio/runs/eef4d3fc9f36493f93d0cbd30bb6e644`; UI evidence is at
`work/studio-two-entries.png` and `work/studio-quality-verified.png`.

The experiment exposed and fixed missing Q4_0/Q4_1 destination traits when
llama.cpp falls back on small tensor shapes. The type sizes were checked
against the official b10666 `ggml-common.h` static assertions.

This is an unsigned local build. Calibration and calibration-bundle tier search
remain CLI workflows; the quality entry now invokes real fidelity search.
# Post-v0.5.0 draft and usability validation (2026-10-04)

- Full Python suite: **328 passed, 1 skipped**. Draft/Studio subset: **33 passed**.
- Draft API requires the same local session as other Studio APIs. Tests cover
  Unicode paths, restart-independent workspace persistence, clear without task
  submission, malformed values preserving the old draft, concurrent atomic
  writes, temporary-file cleanup and recovery from corrupt JSON.
- Browser input/reload check: `0.023456789` GiB remains intact and corresponds
  to **25,186,535 bytes**; both survive reload alongside source paths, presets
  and the Quality tier. Size and quality source fields synchronize during input.
- Clear removes the saved file while retaining visible inputs. A deliberately
  overlong path caused a refused save; replacing it and using the retry button
  produced the saved confirmation. No model job was created by these checks.
- A 390px viewport produced a 375px document client width and the same scroll
  width (no horizontal overflow). Temporary viewport override was reset.
- `node --check` and `git diff --check` pass. Native file picker and frozen
  Windows packaging were not revalidated in this UI/draft update.
