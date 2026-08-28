#!/usr/bin/env python3
"""Create the M13 (third) holdout slices documented in
experiments/2026-08-28-m13-budget-rule/README.md.

Same sources and 65,536-character construction as M9/M11, at offsets disjoint
from BOTH earlier slice sets. Disjointness is asserted two ways: arithmetic
offset ranges, and absence of sampled windows of the M9 and M11 slices in the
new slice text.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pyarrow.parquet as pq

CALIB = Path("/run/media/s117/OS/Models/imatrix-calibration")
EVAL_DATA = Path("/run/media/s117/OS/Models/eval-data")
OUT_DIR = EVAL_DATA / "holdout-m13"
RECORD = Path(
    "/run/media/s117/OS/FIT-GGUF/experiments/2026-08-28-m13-budget-rule/holdout3-slices.json"
)
SLICE_CHARS = 65_536

# domain -> (source, [(label, offset), ...] prior sets, m13 offset)
SOURCES = {
    "wiki_test": (
        EVAL_DATA / "wikitext-2-raw/wiki.test.raw",
        {"m9": 0, "m11": 655_360},
        983_040,
    ),
    "wiki_valid": (
        EVAL_DATA / "wikitext-2-raw/wiki.valid.raw",
        {"m9": 0, "m11": 655_360},
        983_040,
    ),
    "chinese": (
        CALIB / "combined_cn_medium.parquet",
        {"m9": 3_173_914, "m11": 4_000_000},
        6_500_000,
    ),
    "code": (
        CALIB / "code_medium.parquet",
        {"m9": 12_536_883, "m11": 18_000_000},
        28_000_000,
    ),
    "agent_chat": (
        CALIB / "agentworld_clean_quick.txt",
        {"m9": 998_087, "m11": 400_000},
        1_200_000,
    ),
}

PRIOR_FILES = {
    "m9": {
        "wiki_test": "kl-eval-64k.txt",
        "wiki_valid": "kl-eval-valid-64k.txt",
        "chinese": "kl-eval-cn-64k.txt",
        "code": "kl-eval-code-64k.txt",
        "agent_chat": "kl-eval-agent-64k.txt",
    },
    "m11": {
        "wiki_test": "holdout-m11/holdout-wiki_test-64k.txt",
        "wiki_valid": "holdout-m11/holdout-wiki_valid-64k.txt",
        "chinese": "holdout-m11/holdout-chinese-64k.txt",
        "code": "holdout-m11/holdout-code-64k.txt",
        "agent_chat": "holdout-m11/holdout-agent_chat-64k.txt",
    },
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


def windows(text: str, count: int = 12) -> list[str]:
    step = len(text) // count
    return [text[i * step : i * step + 100] for i in range(count) if text[i * step : i * step + 100]]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    record: dict[str, object] = {"schema_version": 1, "slice_chars": SLICE_CHARS, "slices": {}}

    for domain, (src, prior_offsets, m13_off) in SOURCES.items():
        text = load_source(src)
        assert m13_off + SLICE_CHARS <= len(text), f"{domain}: window exceeds source"
        for label, off in prior_offsets.items():
            prior_end = off + SLICE_CHARS
            assert m13_off >= prior_end or m13_off + SLICE_CHARS <= off, (
                f"{domain}: M13 window overlaps {label} arithmetic range"
            )
        hold = text[m13_off : m13_off + SLICE_CHARS]

        leaked = []
        for label, rel in PRIOR_FILES.items():
            prior = load_source(EVAL_DATA / rel[domain])
            leaked += [f"{label}:{w[:24]!r}" for w in windows(prior) if w in hold]
        assert not leaked, f"{domain}: content overlap detected {leaked}"

        out_path = OUT_DIR / f"holdout3-{domain}-64k.txt"
        out_path.write_text(hold, encoding="utf-8", newline="")

        record["slices"][domain] = {
            "source": str(src),
            "prior_offsets_chars": prior_offsets,
            "m13_offset_chars": m13_off,
            "chars": len(hold),
            "bytes": out_path.stat().st_size,
            "sha256": sha256_file(out_path),
            "cjk_fraction": round(len(CJK.findall(hold)) / len(hold), 4),
            "prior_windows_found": 0,
        }
        print(f"{domain}: bytes={record['slices'][domain]['bytes']} sha={record['slices'][domain]['sha256'][:16]}…")

    RECORD.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"provenance record: {RECORD}")


if __name__ == "__main__":
    main()
