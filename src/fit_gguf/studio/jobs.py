"""Sequential subprocess jobs with persisted records and process-tree cancellation."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import uuid

from fit_gguf.gguf import read_gguf_layout
from fit_gguf.pipeline import PRESET_FILE_TYPES, load_analysis


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def input_path(value: object, *, directory: bool = False) -> Path:
    if not isinstance(value, str) or not value.strip() or "\0" in value:
        raise ValueError("Enter an existing local path")
    path = Path(value.strip()).expanduser().resolve()
    if not (path.is_dir() if directory else path.is_file()):
        raise ValueError(f"{'Directory' if directory else 'File'} not found: {path}")
    return path


def read_record(path: Path) -> dict:
    if path.stat().st_size > 32 * 1024 * 1024:
        raise ValueError("JSON record exceeds 32 MiB")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON object")
    return payload


def summarize_analysis(path: Path) -> dict:
    record = load_analysis(path)
    return {key: record.get(key) for key in ("source", "imatrix", "runtime", "presets",
                                             "candidate_count", "net_preset_gap_bytes")}


class JobManager:
    def __init__(self, workspace: Path, max_model_params: float = 5, worker_executable: Path | None = None):
        self.workspace = workspace.resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.limit = max_model_params
        self.worker_executable = worker_executable.resolve() if worker_executable else None
        self.lock = threading.RLock()
        self.process: subprocess.Popen | None = None
        self.active: str | None = None
        self.jobs: dict[str, dict] = {}
        for path in sorted(self.workspace.glob("runs/*/job.json")):
            try:
                record = read_record(path)
                if record["status"] in ("running", "queued", "cancelling"):
                    record.update(status="interrupted", finished_at=now())
                    temporary = path.with_suffix(".tmp")
                    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
                    temporary.replace(path)
                self.jobs[record["id"]] = record
            except (OSError, ValueError, KeyError):
                continue

    def _check_source(self, path: Path) -> None:
        layout = read_gguf_layout(path)
        parameters = sum(math.prod(tensor.shape) for tensor in layout.tensors)
        if parameters > self.limit * 1e9:
            raise ValueError(f"Source has {parameters / 1e9:.2f}B tensor parameters; this Studio is limited to {self.limit:g}B")

    def _check_analysis_inputs(self, record: dict) -> None:
        for key in ("source", "imatrix"):
            path = input_path(record[key]["path"])
            expected = record[key].get("sha256")
            if expected:
                digest = hashlib.sha256()
                with path.open("rb") as stream:
                    while chunk := stream.read(1024 * 1024):
                        digest.update(chunk)
                if digest.hexdigest() != expected:
                    raise ValueError(f"Analysis input changed: {path}")

    def _command(self, action: str, payload: dict, folder: Path) -> tuple[list[str], dict]:
        arguments = ([str(self.worker_executable or sys.executable), "--fit-worker", action]
                     if self.worker_executable or getattr(sys, "frozen", False)
                     else [sys.executable, "-u", "-m", "fit_gguf.cli", action])
        artifacts = {}
        if action == "analyze":
            source = input_path(payload.get("source"))
            self._check_source(source)
            imatrix = input_path(payload.get("imatrix"))
            runtime = input_path(payload.get("runtime"), directory=True)
            lower, upper = payload.get("lower", "IQ3_M"), payload.get("upper", "IQ4_XS")
            if lower not in PRESET_FILE_TYPES or upper not in PRESET_FILE_TYPES or lower == upper:
                raise ValueError("Select two different supported presets")
            arguments += ["--source", str(source), "--imatrix", str(imatrix), "--runtime", str(runtime),
                          "--imatrix-arg", str(imatrix), "--lower", lower, "--upper", upper,
                          "--out-dir", str(folder / "analysis")]
            artifacts["analysis"] = str(folder / "analysis" / "analysis.json")
        elif action == "plan":
            analysis = input_path(payload.get("analysis"))
            record = load_analysis(analysis)
            from fit_gguf.studio.context import check_model_context
            check_model_context(payload, record)
            self._check_source(input_path(record["source"]["path"]))
            self._check_analysis_inputs(record)
            target = payload.get("target_bytes")
            if isinstance(target, bool) or not isinstance(target, int) or target <= 0:
                raise ValueError("Target must be a positive integer byte count")
            lower = record["presets"]["lower"]["predicted_size_bytes"]
            upper = record["presets"]["upper"]["predicted_size_bytes"]
            if not lower <= target <= upper:
                raise ValueError(f"Target must lie in this preset interval: {lower:,}–{upper:,} bytes")
            policy = payload.get("policy", "balanced")
            if policy not in ("original", "balanced"):
                raise ValueError("Choose original or balanced policy")
            prefix = folder / "FIT"
            arguments += ["--analysis", str(analysis), "--target-bytes", str(target),
                          "--policy", policy, "--out-prefix", str(prefix)]
            artifacts.update(plan=str(folder / "FIT-plan.json"), recipe=str(folder / "FIT-recipe.json"),
                             tensor_types=str(folder / "FIT-tensor-types.txt"))
        elif action == "quantize":
            analysis = input_path(payload.get("analysis"))
            record = load_analysis(analysis)
            from fit_gguf.studio.context import check_model_context, check_plan_context
            check_model_context(payload, record)
            self._check_source(input_path(record["source"]["path"]))
            self._check_analysis_inputs(record)
            plan_file = input_path(payload.get("plan"))
            plan = read_record(plan_file)
            check_plan_context(payload, plan)
            if Path(plan["analysis_path"]).resolve() != analysis:
                raise ValueError("This plan belongs to a different analysis")
            types = input_path(plan["tensor_types_path"])
            for path, digest_key in ((analysis, "analysis_sha256"), (types, "tensor_types_sha256")):
                if hashlib.sha256(path.read_bytes()).hexdigest() != plan.get(digest_key):
                    raise ValueError(f"Plan input changed: {path}")
            output = folder / "FIT.gguf"
            expected = int(plan["predicted_size_bytes"])
            import shutil
            if shutil.disk_usage(self.workspace).free < expected + 256 * 1024 ** 2:
                raise ValueError("Not enough disk space for output and 256 MiB headroom")
            arguments += ["--analysis", str(analysis), "--tensor-types", str(types),
                          "--expect-bytes", str(expected), "--out", str(output)]
            artifacts.update(model=str(output), quantize=str(output) + ".quantize-record.json")
        elif action == "quality":
            source = input_path(payload.get("source"))
            self._check_source(source)
            imatrix = input_path(payload.get("imatrix"))
            runtime = input_path(payload.get("runtime"), directory=True)
            refs = input_path(payload.get("refs_dir"), directory=True)
            corpus = input_path(payload.get("eval_data_dir"), directory=True)
            freeze = input_path(payload.get("freeze"))
            manifest = input_path(payload.get("reference_manifest"))
            from fit_gguf.fidelity import KL_ANCHORS
            tier = payload.get("tier", "balanced")
            if tier not in KL_ANCHORS:
                raise ValueError("Select a supported quality tier")
            threads = payload.get("threads", 4)
            if isinstance(threads, bool) or not isinstance(threads, int) or not 1 <= threads <= 16:
                raise ValueError("Evaluator threads must be an integer between 1 and 16")
            # Keep user reference/corpus directories read-only. Every plan,
            # probe, log and delivered model belongs to this independent run.
            arguments[-1] = "fidelity-search"
            arguments += ["--source", str(source), "--imatrix", str(imatrix), "--runtime", str(runtime),
                          "--refs-dir", str(refs), "--eval-data-dir", str(corpus), "--freeze", str(freeze),
                          "--reference-manifest", str(manifest), "--tier", tier,
                          "--preset-ladder", "IQ2_XXS,IQ3_XXS,IQ3_M,IQ4_XS,Q4_K_M,Q5_K_M,Q6_K,Q8_0",
                          "--out-dir", str(folder / "quality"), "--work-dir", str(folder / "scratch"),
                          "--logs-dir", str(folder / "logs"), "--manifest", str(folder / "artifacts.sha256"),
                          "--threads", str(threads), "--n-gpu-layers", "0", "--profile", "normal"]
            artifacts["quality"] = str(folder / "quality" / f"fidelity-search-{tier}-product.json")
        else:
            raise ValueError("Unsupported task; choose analyze, plan, quantize or quality")
        return arguments, artifacts

    def submit(self, action: str, payload: dict) -> dict:
        with self.lock:
            if self.active:
                raise ValueError("A task is already running. Wait or cancel it first.")
            identifier = uuid.uuid4().hex
            folder = self.workspace / "runs" / identifier
            command, artifacts = self._command(action, payload, folder)
            folder.mkdir(parents=True)
            if action == "quality":
                for name in ("quality", "scratch", "logs"):
                    (folder / name).mkdir()
                (folder / "artifacts.sha256").touch()
            job = {"id": identifier, "action": action, "status": "queued", "created_at": now(),
                   "command": command, "artifacts": artifacts, "log": str(folder / "task.log")}
            self.jobs[identifier] = job
            self.active = identifier
            self._save(job)
            threading.Thread(target=self._run, args=(identifier,), daemon=True).start()
            return dict(job)

    def _save(self, job: dict) -> None:
        destination = self.workspace / "runs" / job["id"] / "job.json"
        temporary = destination.with_suffix(".tmp")
        temporary.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(destination)

    def _run(self, identifier: str) -> None:
        job = self.jobs[identifier]
        completion = {}
        try:
            with Path(job["log"]).open("w", encoding="utf-8") as log:
                environment = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
                environment["FIT_STUDIO_WORKER_LOG"] = job["log"]
                with self.lock:
                    if job["status"] == "cancelling":
                        completion = {"status": "cancelled", "finished_at": now()}
                        return
                    self.process = subprocess.Popen(job["command"], stdout=log, stderr=subprocess.STDOUT,
                                                    env=environment, cwd=self.workspace,
                                                    creationflags=(subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0),
                                                    start_new_session=os.name != "nt")
                    job.update(status="running", started_at=now())
                    self._save(job)
                    process = self.process
                code = process.wait()
                completion = {"status": "succeeded" if code == 0 else "failed",
                              "exit_code": code, "finished_at": now()}
        except Exception as error:
            completion = {"status": "failed", "error": str(error), "finished_at": now()}
        finally:
            with self.lock:
                if job["status"] == "cancelling":
                    completion["status"] = "cancelled"
                job.update(completion)
                self._save(job)
                self.process = None
                self.active = None

    def cancel(self, identifier: str) -> dict:
        with self.lock:
            if identifier not in self.jobs:
                raise KeyError(identifier)
            job = self.jobs[identifier]
            if self.active == identifier:
                job["status"] = "cancelling"
                self._save(job)
                if self.process and self.process.poll() is None:
                    if os.name == "nt":
                        subprocess.run(["taskkill", "/PID", str(self.process.pid), "/T", "/F"],
                                       capture_output=True, timeout=10, creationflags=subprocess.CREATE_NO_WINDOW)
                    else:
                        os.killpg(self.process.pid, signal.SIGTERM)
            return dict(job)

    def snapshot(self, identifier: str | None = None) -> dict | list:
        with self.lock:
            if identifier:
                job = dict(self.jobs[identifier])
                log_path = Path(job["log"])
                if log_path.is_file():
                    with log_path.open("rb") as stream:
                        stream.seek(max(0, log_path.stat().st_size - 65536))
                        job["log_text"] = stream.read().decode("utf-8", errors="replace")
                if job["status"] in ("succeeded", "failed"):
                    for name in ("analysis", "plan", "quantize", "quality"):
                        if job["status"] != "succeeded" and name != "quality":
                            continue
                        if name in job["artifacts"]:
                            path = Path(job["artifacts"][name])
                            if path.is_file():
                                job["result"] = summarize_analysis(path) if name == "analysis" else read_record(path)
                return job
            return [dict(job) for job in sorted(self.jobs.values(), key=lambda job: job.get("created_at", ""), reverse=True)]

    def close(self) -> None:
        if self.active:
            self.cancel(self.active)

