#!/usr/bin/env python3
"""Summarize M11 holdout eval logs into a deterministic JSON record.

Self-contained on purpose: the M9 summarizer expects an externally appended
timing line that M11 logs do not carry. Elapsed time is taken from the log's
own llama_perf "total time"; max RSS was not instrumented in M11.
"""

from __future__ import annotations

import json
import sys
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
    "same_top_percent": re.compile(r"^Same top p:\s+([0-9.]+) ±\s+([0-9.]+) %$", re.MULTILINE),
}
TOTAL_TIME_NOTE = (
    "elapsed_seconds is wall-clock from log file birth->mtime (M11 did not "
    "instrument the runs); max_rss_kb was not instrumented."
)
REFERENCE_PPL_PATTERN = re.compile(r"Final estimate: PPL = ([0-9.]+) \+/- ([0-9.]+)", re.MULTILINE)

DOMAINS = ("wiki_test", "wiki_valid", "chinese", "code", "agent_chat")
VARIANTS = ("fit50", "block-balanced-fit50", "random-v1", "random-v2", "random-v3")


def wall_seconds(path: Path) -> float:
    st = path.stat()
    elapsed = st.st_mtime - st.st_ctime if st.st_ctime else 0.0
    # ntfs3 exposes creation time via statx birth time; fall back to ctime.
    import subprocess

    out = subprocess.run(["stat", "-c", "%W", str(path)], capture_output=True, text=True)
    try:
        birth = int(out.stdout.strip())
    except ValueError:
        birth = 0
    if birth > 0:
        elapsed = st.st_mtime - birth
    return round(max(elapsed, 0.0), 2)


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
            domain: parse_reference_log(logs / f"ref-holdout-bf16-{domain}.log")
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
