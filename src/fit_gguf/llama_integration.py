"""Recipe-file generation and platform-aware llama.cpp binary lookup."""

from pathlib import Path
import os
import re

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


def write_tensor_type_file(plan: OptimizationPlan, path: str | Path) -> None:
    """Write exact-name llama.cpp tensor overrides in deterministic order."""
    lines = [
        f"^{re.escape(tensor)}$={qtype}"
        for tensor, qtype in sorted(plan.overrides, key=lambda override: override[0])
    ]
    Path(path).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
