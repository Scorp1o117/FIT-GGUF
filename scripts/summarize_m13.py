#!/usr/bin/env python3
"""Summarize M13 holdout-3 eval logs into a deterministic JSON record."""

from __future__ import annotations

import json
import sys
from pathlib import Path
import re

sys.path.insert(0, str(Path(__file__).parent))
from summarize_m11 import PAIR_PATTERNS, wall_seconds  # noqa: E402

DOMAINS = ("wiki_test", "wiki_valid", "chinese", "code", "agent_chat")
VARIANTS = ("orig-fit25", "orig-fit50", "orig-fit75",
            "v01b-fit25", "v01b-fit50", "v01b-fit75")
REFERENCE_PPL_PATTERN = re.compile(r"Final estimate: PPL = ([0-9.]+) \+/- ([0-9.]+)", re.MULTILINE)


def parse_eval_log(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8")
    result: dict[str, object] = {}
    for name, pattern in PAIR_PATTERNS.items():
        match = pattern.search(text)
        if match is None:
            raise ValueError(f"Missing {name} in {path}")
        result[name] = float(match.group(1))
        result[f"{name}_uncertainty"] = float(match.group(2))
    result["elapsed_seconds"] = wall_seconds(path)
    result["max_rss_kb"] = None
    return result


def parse_reference_log(path: Path) -> dict[str, float | int | None]:
    text = path.read_text(encoding="utf-8")
    ppl = REFERENCE_PPL_PATTERN.search(text)
    if ppl is None:
        raise ValueError(f"Incomplete reference log: {path}")
    return {
        "ppl": float(ppl.group(1)),
        "ppl_uncertainty": float(ppl.group(2)),
        "elapsed_seconds": wall_seconds(path),
        "max_rss_kb": None,
    }


def main() -> None:
    experiment_dir = Path(sys.argv[1])
    output = Path(sys.argv[2])
    logs = experiment_dir / "artifacts" / "logs"
    payload = {
        "schema_version": 1,
        "parameters": {"n_ctx": 512, "batch_size": 512, "gpu_layers": 99, "threads": 16},
        "references": {
            domain: parse_reference_log(logs / f"ref-holdout3-bf16-{domain}.log")
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
