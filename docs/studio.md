# FIT Studio

FIT Studio puts FIT's **analyze → plan → quantize** workflow in a local graphical
application. The same UI runs in a normal browser and an optional PyWebView
desktop window. It adds live hardware budgets and optional llmfit model-fit
recommendations without changing FIT's allocation or exact-size gates.

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

1. **Overview:** inspect current free VRAM, available RAM and CPU cores. Choose
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
bound to exact source hashes. FIT's existing `calibrate`, `fidelity-search`
and `tier-search` remain CLI operations in this first version. Size
verification alone does not establish quantization quality.

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
