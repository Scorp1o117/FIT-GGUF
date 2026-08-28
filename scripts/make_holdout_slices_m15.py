#!/usr/bin/env python3
"""Create the M14 (fourth) holdout slices: disjoint from M9, M11, and M13."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pyarrow.parquet as pq

CALIB = Path("/run/media/s117/OS/Models/imatrix-calibration")
EVAL_DATA = Path("/run/media/s117/OS/Models/eval-data")
OUT_DIR = EVAL_DATA / "holdout-m15"
RECORD = Path(
    "/run/media/s117/OS/FIT-GGUF/experiments/2026-08-29-m15-random-baseline/holdout5-slices.json"
)
SLICE_CHARS = 65_536

SOURCES = {
    "wiki_test": (EVAL_DATA / "wikitext-2-raw/wiki.test.raw", {"m9": 0, "m11": 655_360, "m13": 983_040, "m14": 327_680}, 458_752),
    "wiki_valid": (EVAL_DATA / "wikitext-2-raw/wiki.valid.raw", {"m9": 0, "m11": 655_360, "m13": 983_040, "m14": 327_680}, 458_752),
    "chinese": (CALIB / "combined_cn_medium.parquet", {"m9": 3_173_914, "m11": 4_000_000, "m13": 6_500_000, "m14": 7_700_000}, 8_800_000),
    "code": (CALIB / "code_medium.parquet", {"m9": 12_536_883, "m11": 18_000_000, "m13": 28_000_000, "m14": 21_000_000}, 33_500_000),
    "agent_chat": (CALIB / "agentworld_clean_quick.txt", {"m9": 998_087, "m11": 400_000, "m13": 1_200_000, "m14": 1_400_000}, 1_500_000),
}
PRIOR_FILES = {
    "m9": {"wiki_test": "kl-eval-64k.txt", "wiki_valid": "kl-eval-valid-64k.txt", "chinese": "kl-eval-cn-64k.txt", "code": "kl-eval-code-64k.txt", "agent_chat": "kl-eval-agent-64k.txt"},
    "m11": {d: f"holdout-m11/holdout-{d}-64k.txt" for d in SOURCES},
    "m13": {d: f"holdout-m13/holdout3-{d}-64k.txt" for d in SOURCES},
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
    for domain, (src, prior_offsets, m15_off) in SOURCES.items():
        text = load_source(src)
        assert m15_off + SLICE_CHARS <= len(text)
        for label, off in prior_offsets.items():
            assert m15_off >= off + SLICE_CHARS or m15_off + SLICE_CHARS <= off, f"{domain} overlaps {label}"
        hold = text[m15_off : m15_off + SLICE_CHARS]
        leaked = [
            f"{label}:{w[:24]!r}"
            for label, rel in PRIOR_FILES.items()
            for w in windows(load_source(EVAL_DATA / rel[domain]))
            if w in hold
        ]
        assert not leaked, f"{domain}: overlap {leaked}"
        out_path = OUT_DIR / f"holdout5-{domain}-64k.txt"
        out_path.write_text(hold, encoding="utf-8", newline="")
        record["slices"][domain] = {
            "source": str(src),
            "prior_offsets_chars": prior_offsets,
            "m15_offset_chars": m15_off,
            "chars": len(hold),
            "bytes": out_path.stat().st_size,
            "sha256": sha256_file(out_path),
            "cjk_fraction": round(len(CJK.findall(hold)) / len(hold), 4),
        }
        print(f"{domain}: bytes={record['slices'][domain]['bytes']} sha={record['slices'][domain]['sha256'][:16]}…")
    RECORD.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"provenance record: {RECORD}")


if __name__ == "__main__":
    main()
