"""Read-only llmfit integration and conservative, live memory budgets.

llmfit stays an optional executable, not a forked scoring engine. Its scores
and throughput figures are estimates; FIT's exact-byte checks stay separate.
"""
from __future__ import annotations

import csv
import io
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sysconfig
import sys

GIB = 1024 ** 3


def run_json(command: list[str], timeout: int = 30) -> dict:
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8",
                            errors="replace", timeout=timeout, creationflags=(
                                subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0))
    if result.returncode:
        raise ValueError((result.stderr or result.stdout or "llmfit failed")[-2000:])
    payload = json.loads(result.stdout)
    if not isinstance(payload, dict) or "error" in payload:
        raise ValueError(f"Unexpected llmfit response: {str(payload)[:300]}")
    return payload


def find_llmfit(explicit: str | None = None) -> str | None:
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"llmfit executable not found: {path}")
        return str(path)
    executable = "llmfit.exe" if os.name == "nt" else "llmfit"
    if getattr(sys, "frozen", False):
        candidate = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent)) / executable
        if candidate.is_file():
            return str(candidate)
    candidate = Path(sysconfig.get_path("scripts")) / executable
    return str(candidate) if candidate.is_file() else shutil.which(executable)


def live_system() -> dict:
    system = {"cpu_name": platform.processor() or platform.machine(),
              "cpu_cores": os.cpu_count(), "total_ram_gb": None,
              "available_ram_gb": None, "gpus": [], "unified_memory": False}
    try:
        import psutil
        memory = psutil.virtual_memory()
        system.update(total_ram_gb=memory.total / GIB, available_ram_gb=memory.available / GIB)
    except ImportError:
        if os.name == "nt":
            import ctypes
            class MemoryStatus(ctypes.Structure):
                _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                    (key, ctypes.c_ulonglong) for key in
                    ("total", "available", "page_total", "page_available", "virtual_total",
                     "virtual_available", "extended_available")]
            status = MemoryStatus()
            status.length = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                system.update(total_ram_gb=status.total / GIB, available_ram_gb=status.available / GIB)
        elif Path("/proc/meminfo").is_file():
            values = {line.split(":")[0]: int(line.split()[1]) * 1024
                      for line in Path("/proc/meminfo").read_text().splitlines()}
            system.update(total_ram_gb=values["MemTotal"] / GIB,
                          available_ram_gb=values.get("MemAvailable", 0) / GIB)
    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            output = subprocess.run([smi, "--query-gpu=name,memory.total,memory.free",
                                     "--format=csv,noheader,nounits"], capture_output=True,
                                    text=True, timeout=5, creationflags=(
                                        subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0))
            if output.returncode == 0:
                system["gpus"] = [{"name": row[0].strip(), "vram_gb": float(row[1]) / 1024,
                                   "free_vram_gb": float(row[2]) / 1024, "backend": "CUDA"}
                                  for row in csv.reader(io.StringIO(output.stdout)) if len(row) == 3]
        except (OSError, ValueError, subprocess.TimeoutExpired):
            pass
    return system


def memory_budget(system: dict, reserve_gb: float = 2, overhead_gb: float = 1,
                  mode: str = "gpu") -> dict:
    """One conservative pool; do not add RAM to VRAM or sum independent GPUs.

    A file budget is pool minus user reserve minus KV/compute estimate. Unknown
    free memory is not treated as total/free; unified memory uses one RAM pool.
    """
    if any(not math.isfinite(v) or v < 0 for v in (reserve_gb, overhead_gb)):
        raise ValueError("Reserve and runtime overhead must be finite and nonnegative")
    if mode not in ("gpu", "cpu"):
        raise ValueError("Mode must be gpu or cpu")
    available = system.get("available_ram_gb")
    if mode == "gpu" and not system.get("unified_memory"):
        pools = [gpu.get("free_vram_gb") for gpu in system.get("gpus", [])]
        pools = [p for p in pools if isinstance(p, (float, int)) and math.isfinite(p)]
        available = max(pools) if pools else None
    if available is None:
        return {"available_gb": None, "target_bytes": None, "note": "Free memory unavailable; enter a file budget manually."}
    target = max(0, float(available) - reserve_gb - overhead_gb)
    return {"available_gb": available, "target_bytes": int(target * GIB),
            "note": "Estimate only. GGUF file size excludes KV cache, compute buffers and other processes."}


def parameter_billions(value: object) -> float | None:
    # Total parameters, not active MoE parameters (e.g. 30B-A3B is NOT 3B).
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([BMT])(?:\s*[-/]\s*A?\d+(?:\.\d+)?[BM])?\s*",
                         str(value), re.IGNORECASE)
    return float(match[1]) * {"B": 1, "M": .001, "T": 1000}[match[2].upper()] if match else None


class LlmfitAdapter:
    def __init__(self, executable: str | None = None):
        self.executable = find_llmfit(executable)

    def system(self) -> dict:
        local = live_system()
        warnings = []
        provider = "local"
        if self.executable:
            try:
                remote = run_json([self.executable, "system", "--json"])["system"]
                # llmfit detects non-NVIDIA backends; our nvidia-smi sample adds
                # live free VRAM absent from older llmfit system JSON schemas.
                for key in ("total_ram_gb", "available_ram_gb", "cpu_cores"):
                    if local.get(key) is not None:
                        remote[key] = local[key]
                if local["gpus"] and not remote.get("unified_memory"):
                    remote["gpus"] = local["gpus"]
                local = remote
                provider = "llmfit + live telemetry"
            except (OSError, ValueError, KeyError, subprocess.TimeoutExpired) as error:
                warnings.append(f"llmfit unavailable; local detection used: {error}")
        return {"system": local, "provider": provider, "llmfit_available": bool(self.executable),
                "warnings": warnings, "budget": memory_budget(local)}

    def models(self, max_params: float = 5, context: int = 4096) -> dict:
        if not 0 < max_params <= 1000 or not 128 <= context <= 131072:
            raise ValueError("Invalid parameter or context limit")
        if not self.executable:
            return {"models": [], "available": False,
                    "note": "Install fit-gguf[hardware] or pass --llmfit to enable llmfit recommendations."}
        local = live_system()
        command = [self.executable, "--max-context", str(context)]
        gpu_free = [gpu.get("free_vram_gb") for gpu in local.get("gpus", [])]
        gpu_free = [value for value in gpu_free if isinstance(value, (int, float))]
        if gpu_free:
            command += ["--memory", f"{max(.01, max(gpu_free) - 2):.4f}G"]
        if local.get("available_ram_gb") is not None:
            command += ["--ram", f"{max(.01, local['available_ram_gb'] - 2):.4f}G"]
        payload = run_json(command + ["fit", "--json"], timeout=60)
        rows = []
        for model in payload.get("models", []):
            count = model.get("params_b")
            if not isinstance(count, (int, float)):
                count = parameter_billions(model.get("parameter_count"))
            # Reject obvious catalog/name conflicts conservatively. Some
            # pre-quantized 8B catalog entries infer <5B from compressed files.
            named_sizes = [float(value) for value in re.findall(
                r"(?<![A-Za-z0-9])(\d+(?:\.\d+)?)B(?=[^A-Za-z0-9]|$)",
                str(model.get("name", "")), re.IGNORECASE)]
            if named_sizes and max(named_sizes) > max_params:
                continue
            if count is not None and math.isfinite(count) and 0 < count <= max_params and str(model.get("runtime", "llamacpp")).lower() in ("llamacpp", "llama.cpp"):
                rows.append(model)
        return {"models": rows, "available": True, "max_params": max_params,
                "context": context, "note": "llmfit estimates against live available memory minus 2 GiB reserve when detectable; no models are downloaded or executed."}
