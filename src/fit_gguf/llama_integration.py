"""Recipe-file generation and platform-aware llama.cpp binary lookup."""

import os
from pathlib import Path
import re
import subprocess
import sys

from fit_gguf.candidates import GGML_TYPE_TRAITS
from fit_gguf.optimizer import OptimizationPlan

# llama.cpp release archives ship extensionless executables on POSIX and
# ``.exe`` on Windows, so a runtime directory is not portable by name alone.
# Windows users also wrap the real binary in a ``.cmd``/``.bat`` shim to pin
# CUDA paths or extra env vars; CreateProcess runs those directly (no shell),
# so they are accepted as ordinary runtime binaries. Every candidate is tried on
# every platform — the platform's native form just comes first — so a runtime
# directory built for the other OS does not silently resolve to nothing.
NATIVE_WINDOWS_SUFFIXES = (".exe", ".cmd", ".bat")


def _is_windows() -> bool:
    """Platform probe, isolated so both candidate orders are testable anywhere."""
    return os.name == "nt"


def run_runtime(command, **kwargs):
    """Run an external llama.cpp process without desktop console flashes.

    Frozen Windows applications alter the DLL search directory. External
    runtimes must load their own libraries, then the application's directory
    is restored for subsequent Python extension imports.
    """
    if os.name != "nt":
        return subprocess.run(command, **kwargs)
    kwargs["creationflags"] = kwargs.get("creationflags", 0) | subprocess.CREATE_NO_WINDOW
    if not getattr(sys, "frozen", False):
        return subprocess.run(command, **kwargs)
    import ctypes
    kernel = ctypes.windll.kernel32
    previous = ctypes.create_unicode_buffer(32768)
    kernel.GetDllDirectoryW(len(previous), previous)
    kernel.SetDllDirectoryW(None)
    try:
        return subprocess.run(command, **kwargs)
    finally:
        kernel.SetDllDirectoryW(previous.value or None)


def binary_candidate_names(name: str) -> tuple[str, ...]:
    """Ordered file names to try for a llama.cpp binary called ``name``."""
    if _is_windows():
        return tuple(f"{name}{suffix}" for suffix in NATIVE_WINDOWS_SUFFIXES) + (name,)
    return (name, f"{name}.exe")


def resolve_runtime_binary(runtime_dir: str | Path, name: str) -> Path:
    """Resolve ``name`` inside ``runtime_dir`` for the current platform.

    Returns the first candidate that exists. When none does, the platform's
    preferred name is returned anyway: callers own the error, so a missing
    binary stays distinguishable from a resolution bug without exceptions
    escaping pure path logic.
    """
    directory = Path(runtime_dir)
    candidates = [directory / candidate for candidate in binary_candidate_names(name)]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def binary_not_found_message(runtime_dir: str | Path, name: str) -> str:
    """Error text naming every candidate that was tried."""
    tried = ", ".join(binary_candidate_names(name))
    return f"{name} not found in {runtime_dir} (tried: {tried})"


def cuda_runtime_siblings(runtime_dir: str | Path) -> list[Path]:
    """Sibling directories carrying the CUDA runtime a GPU backend links against.

    A llama.cpp Windows CUDA release ships ``ggml-cuda.dll`` *inside* the binary
    directory but keeps the CUDA runtime it links against (``cudart64_*.dll``,
    ``cublas*_*.dll``) in a separate ``cudart-*`` directory, and the release
    notes put "copy cudart next to the binaries" on the user. If the user keeps
    them side by side instead — the layout the llama.cpp-hub launcher itself
    produces — ``ggml-cuda.dll`` fails to load and llama.cpp **silently falls
    back to CPU**: the run still succeeds, only enormously slower, which is the
    worst failure mode there is (measured: 150 t/s vs 11,779 t/s at pp512 on a
    2.5B BF16 model, so 78x).
    """
    directory = Path(runtime_dir)
    try:
        siblings = sorted(entry for entry in directory.parent.iterdir() if entry.is_dir())
    except OSError:
        return []
    found: list[Path] = []
    for sibling in siblings:
        if sibling == directory:
            continue
        if any(sibling.glob("cudart64_*.dll")) or any(sibling.glob("cublas64_*.dll")):
            found.append(sibling)
    return found


def runtime_env(runtime_dir: str | Path, base: dict | None = None) -> dict:
    """Environment for a runtime binary, with its libraries discoverable.

    POSIX loaders search ``LD_LIBRARY_PATH``; Windows resolves DLLs from the
    executable's own directory first and then along ``PATH``, which is the
    equivalent knob there. Both the runtime directory and any sibling CUDA
    runtime directory are added — see :func:`cuda_runtime_siblings` for why
    leaving the sibling out is a silent 78x slowdown rather than an error.
    """
    directory = Path(runtime_dir)
    ordered = [directory, *cuda_runtime_siblings(directory)]
    windows = _is_windows()
    separator = ";" if windows else ":"
    prefix = separator.join(str(entry) for entry in ordered)
    env = dict(os.environ if base is None else base)
    if windows:
        env["PATH"] = prefix + separator + env.get("PATH", "")
    else:
        env["LD_LIBRARY_PATH"] = prefix + separator + env.get("LD_LIBRARY_PATH", "")
    return env


def write_tensor_type_file(plan: OptimizationPlan, path: str | Path) -> None:
    """Write exact-name llama.cpp tensor overrides in deterministic order.

    **One line per tensor, and the highest type wins.** ``llama-quantize``
    resolves this file by *first match*: a second line for the same tensor is not
    a merge, it is dead text, and the earlier one takes the tensor.  So a
    duplicate is never harmless — it is a silent downgrade.  Occamy shipped four
    of five tiers with their precision floors evaporated exactly this way, which
    is why the deduplication lives here and not only in the caller that was
    supposed to avoid it.
    """
    best: dict[str, str] = {}
    for tensor, qtype in plan.overrides:
        current = best.get(tensor)
        if current is None or _bits_per_weight(qtype) > _bits_per_weight(current):
            best[tensor] = qtype
    lines = [f"^{re.escape(tensor)}$={qtype}" for tensor, qtype in sorted(best.items())]
    Path(path).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _bits_per_weight(qtype: str) -> float:
    """Encoded bits per weight for a GGML type name, 0.0 when unknown.

    An unknown name sorts below every known one rather than raising: this runs
    while writing a file, and a name this table has never seen is a reason to
    keep the other candidate, not to abort a plan.
    """
    traits = GGML_TYPE_TRAITS.get(qtype.lower())
    if traits is None:
        return 0.0
    block_size, type_size = traits
    return type_size * 8.0 / block_size
