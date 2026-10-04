# FIT Studio

FIT Studio offers two main entries: **specify file size** and **specify quality
tier**. The same UI runs in a normal browser and an optional PyWebView desktop
window. Hardware budgets, optional llmfit recommendations, task history and
registry evidence are supporting tools.

## Language and workflow guidance

Use **Language** to switch the home page, size/quality forms, navigation and
common task states between Simplified Chinese and English. The choice is saved
with the workspace draft. Raw logs, model names, artifact paths and upstream
errors retain their original text. Auxiliary hardware/model details currently
have partial English coverage.

The quality entry explains the selected tier's tradeoff. Lower KL thresholds
allow less distribution divergence against the frozen reference; they are not
universal accuracy percentages. Every delivered quality artifact still needs
actual evaluation and final re-evaluation.

Editing source, imatrix, runtime or presets invalidates the current analysis and
plan references. Editing size or allocation policy invalidates the plan. Existing
artifacts stay in history, but a new plan is required for the changed inputs.
Loading an analysis binds the model fields to that record. Planning/quantizing
requests also send the visible model context; quantization includes the visible
budget and policy. The server refuses contexts that differ from the saved
analysis/plan before launching a task. Existing file/hash checks still apply.
Older API clients that omit context remain supported through those record checks.

## Local form drafts

Studio saves form edits after a short pause to `studio-draft.json` inside the
selected workspace. Source/runtime/reference paths, exact size budget, quality
tier and hardware budget settings survive a restart, including a different
local server port. Paths are stored as plain text locally; model files are not
copied. Each workspace has one shared draft; the most recent save wins if several
windows edit it.

A saved draft takes priority over automatically restoring completed tasks.
Results remain in task history. Restoring a draft never submits a task. An
analysis path is re-read to restore slider bounds; missing analysis files show
an actionable error without discarding other inputs. Budget values remain exact
and still have to pass the normal planning validation before submission.

The save indicator confirms when edits reach disk; wait for it before closing
the window. Failed saves expose a retry button. **Clear saved draft** removes the
saved file while keeping current inputs and all task records. Editing again
creates a new draft. An unreadable draft can also be cleared from this control.

## Start from source

Python 3.11+, with an activated virtual environment:

```sh
python -m venv .venv
# Windows: .venv/Scripts/Activate.ps1 ; POSIX: source .venv/bin/activate
python -m pip install -e '.[desktop,hardware,test]'
fit desktop
```

On Windows, `Start-FIT-Studio.ps1` opens the desktop app and creates a virtual
environment on first use. For browser mode (no desktop dependency required):

```sh
python -m pip install -e '.[hardware]'
fit gui
fit gui --no-open --port 8766 --workspace work/studio
```

Open the session URL printed in the terminal. The link establishes a local
cookie and redirects to a URL without the token. Use the exact `127.0.0.1`
hostname. The server listens only on loopback and rejects foreign hosts and
origins. It is a local application, not a multi-user network service.

The desktop window requires the Microsoft Edge WebView2 runtime on Windows.
Browser mode remains available without WebView2 or PyWebView. Existing CLI
commands remain usable without any GUI dependency.

## Workflow

1. **Hardware and budget:** inspect current free VRAM, available RAM and CPU cores. Choose
   one GPU or CPU/RAM, reserve memory for other applications, and estimate KV
   and compute overhead. Bring the resulting file budget into the workspace.
2. **Analyze:** select a source floating-point GGUF, imatrix GGUF and compatible
   llama.cpp runtime directory. Select two supported bracketing presets. Studio
   passes the absolute imatrix path so execution is independent of the task's
   working directory. Desktop `…` buttons open native file/folder pickers;
   in the browser, paste full local paths.
3. **Plan:** load the completed or an existing `analysis.json`, choose a target
   within its interval with the slider or exact byte input, and select balanced
   or original allocation. Saves the plan, recipe and tensor-type overrides.
4. **Quantize:** select the plan and execute the quantizer. Studio checks that
   it belongs to the analysis, verifies analysis/tensor-file hashes and any
   recorded source/imatrix hashes, and reserves 256 MiB disk headroom.
   A successful job must pass FIT's actual-byte gates. The result displays
   actual size, prediction difference and SHA-256.
5. **History:** each run has its own directory, job record and log. Tasks run
   one at a time, can be cancelled, and are cancelled when the server closes.
   A task left running by a crashed process becomes `interrupted` on reopening.
   Partial output is retained for inspection, never labelled successful.

The size entry shows one step at a time. Set the requested GiB budget, analyze
the model, plan, then execute. A requested budget below the preset interval is
not silently raised: select lower presets or explicitly change the budget.

## Specify a quality tier

Choose Mini, Compact, Balanced, Quality or Reference. The global macro KL
thresholds are respectively 0.20, 0.15, 0.10, 0.05 and 0.02. Studio invokes the
real `fit fidelity-search` product flow, not a preset renamed as a quality tier.

Supply the source GGUF, its imatrix and llama.cpp runtime, then the five-domain
reference KLD directory, fixed evaluation corpus directory, eval-v1 `FREEZE.json`
and the source-bound `reference-manifest.json`. A `fit calibrate` reference
bundle can provide the references and manifest. FIT verifies the source,
reference/corpus hashes and frozen evaluator contract before expensive work.

The Studio execution profile uses CPU evaluation, four threads by default
(configurable from 1 to 16), the normal search budget of eight fresh evaluations,
and final evaluation of the delivered artifact. Every run has independent
scratch, logs, manifest and output paths; input reference directories stay
read-only. A failed/no-pass search exposes its report without claiming a
successful artifact. The default search tolerance is 128 MiB, so the result is
the smallest verified PASS found within that bracket and budget, not a proof of
the global minimum. Same-top is informational; KL decides the tier.

The default source-model safety limit is **5B tensor parameters**. Change it
deliberately with `fit desktop --max-model-params 3` (or a larger limit on
another machine). This guards parameter count; it does not guarantee a job
fits memory. Existing analyses created with `--skip-hash` have no source
digest to verify; re-analyze with hashing for reproducible execution.

## Hardware and llmfit integration

Install the `hardware` extra or supply an existing executable:

```sh
fit gui --llmfit /path/to/llmfit
```

Studio calls `llmfit system --json` and `llmfit --max-context … fit --json`,
filters by **total** parameters (not active MoE parameters), and shows llama.cpp
recommendations. Where live telemetry is available, recommendations use free
memory minus a 2 GiB reserve. Browsing does not download, serve, benchmark or
upload a model. The JSON integration was exercised with llmfit **1.1.16**.

An absent or broken llmfit falls back to local CPU/RAM/NVIDIA detection; model
recommendations explain their availability. NVIDIA free VRAM is sampled with
`nvidia-smi`; unknown free VRAM stays unknown. The file-budget estimator uses
the largest individual GPU pool and never silently adds RAM to VRAM or sums
independent cards. Unified memory is one pool. CPU budgets use available RAM.

llmfit scores and throughput figures are **estimates**. A recommendation is not
an existing FIT calibration, an exact size prediction, a guaranteed compatible
download, or a measured benchmark. Registry entries are shown separately,
bound to exact source hashes. `calibrate` and bundle `tier-search` remain CLI
operations; Studio's quality entry executes `fidelity-search`. Size verification
alone does not establish quantization quality.

## Build a Windows application

```powershell
uv pip install --python .venv/Scripts/python.exe -e '.[desktop,hardware]' pyinstaller
./tools/build_studio.ps1
```

Run `dist/FIT-Studio/FIT-Studio.exe`, keeping its whole directory together.
Python and application dependencies are bundled; llama.cpp and models are
supplied separately. llmfit is bundled if installed in the build environment.
The workspace defaults to `%LOCALAPPDATA%/FIT-Studio/workspace`; `--workspace`
overrides it. `-OneFile` builds a self-extracting executable instead.

Frozen jobs launch the application's own CLI worker, with no system Python
requirement. `THIRD_PARTY_NOTICES.md` and `licenses/` accompany the build.

## Validation

```sh
python -m pytest tests/test_studio.py
python tools/smoke_studio.py --source model.gguf --imatrix model.imatrix.gguf \
  --runtime /path/to/llama.cpp --workspace work/studio-smoke
```

The smoke command uses a supplied <=5B model, runs the actual subprocess chain
and writes `smoke-report.json`. It does not download any inputs.
