#!/usr/bin/env python3
"""Summarize pinned llama-perplexity M9 logs into a deterministic JSON record."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re


PAIR_PATTERNS = {
    "ppl": re.compile(r"^Mean PPL\(Q\)\s+:\s+([0-9.]+) ±\s+([0-9.]+)$", re.MULTILINE),
    "base_ppl": re.compile(r"^Mean PPL\(base\)\s+:\s+([0-9.]+) ±\s+([0-9.]+)$", re.MULTILINE),
    "ppl_delta": re.compile(
        r"^Mean PPL\(Q\)-PPL\(base\)\s+:\s+(-?[0-9.]+) ±\s+([0-9.]+)$",
        re.MULTILINE,
    ),
    "mean_kld": re.compile(r"^Mean\s+KLD:\s+([0-9.]+) ±\s+([0-9.]+)$", re.MULTILINE),
    "same_top_percent": re.compile(
        r"^Same top p:\s+([0-9.]+) ±\s+([0-9.]+) %$", re.MULTILINE
    ),
}
TIME_PATTERN = re.compile(r"^elapsed_seconds=([0-9.]+) max_rss_kb=(\d+)$", re.MULTILINE)
REFERENCE_PPL_PATTERN = re.compile(
    r"Final estimate: PPL = ([0-9.]+) \+/- ([0-9.]+)", re.MULTILINE
)


def parse_eval_log(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8")
    result: dict[str, object] = {}
    for name, pattern in PAIR_PATTERNS.items():
        match = pattern.search(text)
        if match is None:
            raise ValueError(f"Missing {name} in {path}")
        result[name] = float(match.group(1))
        result[f"{name}_uncertainty"] = float(match.group(2))
    timing = TIME_PATTERN.search(text)
    if timing is None:
        raise ValueError(f"Missing timing in {path}")
    result["elapsed_seconds"] = float(timing.group(1))
    result["max_rss_kb"] = int(timing.group(2))
    return result


def parse_reference_log(path: Path) -> dict[str, float | int]:
    text = path.read_text(encoding="utf-8")
    ppl = REFERENCE_PPL_PATTERN.search(text)
    timing = TIME_PATTERN.search(text)
    if ppl is None or timing is None:
        raise ValueError(f"Incomplete reference log: {path}")
    return {
        "ppl": float(ppl.group(1)),
        "ppl_uncertainty": float(ppl.group(2)),
        "elapsed_seconds": float(timing.group(1)),
        "max_rss_kb": int(timing.group(2)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    logs = args.experiment_dir / "artifacts" / "logs"
    domains = ("wiki_test", "wiki_valid", "chinese", "code", "agent_chat")
    variants = ("lower", "fit25", "fit50", "fit75", "upper")
    payload = {
        "schema_version": 1,
        "parameters": {"n_ctx": 512, "batch_size": 512, "gpu_layers": 99, "threads": 16},
        "references": {
            domain: parse_reference_log(logs / f"ref-bf16-{domain}.log")
            for domain in domains
        },
        "results": {
            domain: {
                variant: parse_eval_log(logs / f"eval-{variant}-{domain}.log")
                for variant in variants
            }
            for domain in domains
        },
    }
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
