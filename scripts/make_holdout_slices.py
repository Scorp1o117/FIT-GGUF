#!/usr/bin/env python3
"""Create the M11 untouched holdout slices documented in
experiments/2026-08-28-m11-holdout/README.md.

Each holdout slice is cut from the same source file as its M9/M10 counterpart
at a documented disjoint offset, with the same 65,536-character construction.
Outputs land in eval-data/holdout-m11/ (outside the Git repository, like the
original eval slices) and a machine-readable provenance record is written to
the M11 experiment directory.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pyarrow.parquet as pq

CALIB = Path("/run/media/s117/OS/Models/imatrix-calibration")
EVAL_DATA = Path("/run/media/s117/OS/Models/eval-data")
OUT_DIR = EVAL_DATA / "holdout-m11"
RECORD = Path(
    "/run/media/s117/OS/FIT-GGUF/experiments/2026-08-28-m11-holdout/holdout-slices.json"
)

SLICE_CHARS = 65_536

# domain -> (source path, original slice offset in chars, holdout offset in chars)
# Offsets were fixed in the preregistration README before any evaluation.
SOURCES = {
    "wiki_test": (
        EVAL_DATA / "wikitext-2-raw/wiki.test.raw",
        0,
        655_360,
    ),
    "wiki_valid": (
        EVAL_DATA / "wikitext-2-raw/wiki.valid.raw",
        0,
        655_360,
    ),
    "chinese": (
        CALIB / "combined_cn_medium.parquet",
        3_173_914,
        4_000_000,
    ),
    "code": (
        CALIB / "code_medium.parquet",
        12_536_883,
        18_000_000,
    ),
    "agent_chat": (
        CALIB / "agentworld_clean_quick.txt",
        998_087,
        400_000,
    ),
}

# Original eval slices, used for the disjointness check.
ORIGINAL_FILES = {
    "wiki_test": "kl-eval-64k.txt",
    "wiki_valid": "kl-eval-valid-64k.txt",
    "chinese": "kl-eval-cn-64k.txt",
    "code": "kl-eval-code-64k.txt",
    "agent_chat": "kl-eval-agent-64k.txt",
}

CJK = re.compile(r"[\u4e00-\u9fff]")


def load_source(path: Path) -> str:
    if path.suffix == ".parquet":
        return pq.read_table(path, columns=["content"]).column("content")[0].as_py() or ""
    return path.read_text(encoding="utf-8", errors="replace")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def disjointness_windows(text: str) -> list[str]:
    step = len(text) // 8
    return [text[i * step : i * step + 100] for i in range(8) if text[i * step : i * step + 100]]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    record: dict[str, object] = {"schema_version": 1, "slice_chars": SLICE_CHARS, "slices": {}}

    for domain, (src, orig_off, hold_off) in SOURCES.items():
        text = load_source(src)
        assert hold_off + SLICE_CHARS <= len(text), f"{domain}: holdout window exceeds source"
        orig_slice = load_source(EVAL_DATA / ORIGINAL_FILES[domain])
        hold = text[hold_off : hold_off + SLICE_CHARS]

        # Non-overlap: the original slice's start/end and sampled windows must be
        # absent from the holdout text.
        leaked = [w for w in disjointness_windows(orig_slice) if w in hold]
        assert not leaked, f"{domain}: holdout overlaps original slice content"

        out_path = OUT_DIR / f"holdout-{domain}-64k.txt"
        out_path.write_text(hold, encoding="utf-8", newline="")

        record["slices"][domain] = {
            "source": str(src),
            "original_offset_chars": orig_off,
            "holdout_offset_chars": hold_off,
            "chars": len(hold),
            "bytes": out_path.stat().st_size,
            "sha256": sha256_file(out_path),
            "cjk_fraction": round(len(CJK.findall(hold)) / len(hold), 4),
            "original_windows_found_in_holdout": 0,
        }
        print(f"{domain}: {out_path} bytes={record['slices'][domain]['bytes']} sha={record['slices'][domain]['sha256'][:16]}…")

    RECORD.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"provenance record: {RECORD}")


if __name__ == "__main__":
    main()
