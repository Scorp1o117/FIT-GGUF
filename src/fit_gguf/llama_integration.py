"""Minimal recipe-file integration with pinned llama-quantize."""

from pathlib import Path
import re

from fit_gguf.optimizer import OptimizationPlan


def write_tensor_type_file(plan: OptimizationPlan, path: str | Path) -> None:
    """Write exact-name llama.cpp tensor overrides in deterministic order."""
    lines = [
        f"^{re.escape(tensor)}$={qtype}"
        for tensor, qtype in sorted(plan.overrides, key=lambda override: override[0])
    ]
    Path(path).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
