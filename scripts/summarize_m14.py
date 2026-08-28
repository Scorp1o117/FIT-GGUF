#!/usr/bin/env python3
"""Summarize M14 holdout-4 eval logs into a deterministic JSON record."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from summarize_m13 import DOMAINS, parse_eval_log, parse_reference_log  # noqa: E402

VARIANTS = (
    "orig-fit50", "v01b-fit50", "oe-fit50", "bl-fit50", "shuf-fit50",
    "orig-fit75", "v01b-fit75", "oe-fit75", "bl-fit75",
)


def main() -> None:
    experiment_dir = Path(sys.argv[1])
    output = Path(sys.argv[2])
    logs = experiment_dir / "artifacts" / "logs"
    payload = {
        "schema_version": 1,
        "parameters": {"n_ctx": 512, "batch_size": 512, "gpu_layers": 99, "threads": 16},
        "references": {
            domain: parse_reference_log(logs / f"ref-holdout4-bf16-{domain}.log")
            for domain in DOMAINS
        },
        "results": {
            domain: {
                variant: parse_eval_log(logs / f"eval-{variant}-{domain}.log")
                for variant in VARIANTS
            }
            for domain in DOMAINS
        },
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
