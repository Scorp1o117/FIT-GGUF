#!/usr/bin/env python3
"""Summarize M15 holdout-5 eval logs into a deterministic JSON record."""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from summarize_m13 import DOMAINS, parse_eval_log, parse_reference_log

VARIANTS = ("o25", "b25", "o75", "b75", "r1-25", "r2-25", "r3-25", "r1-75", "r2-75", "r3-75")

def main() -> None:
    experiment_dir = Path(sys.argv[1]); output = Path(sys.argv[2])
    logs = experiment_dir / "artifacts" / "logs"
    payload = {
        "schema_version": 1,
        "parameters": {"n_ctx": 512, "batch_size": 512, "gpu_layers": 99, "threads": 16},
        "references": {d: parse_reference_log(logs / f"ref-holdout5-bf16-{d}.log") for d in DOMAINS},
        "results": {d: {v: parse_eval_log(logs / f"eval-{v}-{d}.log") for v in VARIANTS} for d in DOMAINS},
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {output}")

if __name__ == "__main__":
    main()
