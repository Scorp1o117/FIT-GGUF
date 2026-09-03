#!/usr/bin/env python3
"""Assemble the P1 main table from the four tier search summaries."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
M2 = REPO / "experiments" / "2026-09-02-m2-topkl-calibration"
RESULTS = M2 / "results-fs"
GI = 1024 ** 3

# v0.1 / preset reference points (m3-tables.md, eval-v1 口径)
V01 = {"quality": (16.68, 0.0496, 95.00), "balanced": (13.00, 0.0987, 91.93),
       "compact": (11.43, 0.1439, 89.23), "mini": (10.42, 0.1955, 86.18)}
PRESET = {"quality": (17.40, 0.0455, 95.59), "balanced": (14.05, 0.0624, 93.79),
          "compact": (11.57, 0.1424, 89.53), "mini": (10.42, 0.1946, 87.12)}


def main() -> int:
    rows = []
    for tier in ("quality", "balanced", "compact", "mini"):
        path = RESULTS / tier / f"fidelity-search-{tier}-summary.json"
        if not path.is_file():
            rows.append((tier, "PENDING", None, None, None, "", ""))
            continue
        s = json.loads(path.read_text())
        best = s.get("best")
        if best:
            rows.append((
                tier,
                s["status"],
                best["size_bytes"] / GI,
                best["macro_kl"],
                best["same_top"] * 100,
                s["active_constraint"] or "",
                f"{s['fresh_evals']}/{s['budget']} evals" + (f"; {s['note']}" if s.get("note") else ""),
            ))
        else:
            rows.append((tier, s["status"], None, None, None, "", s["guarantee"]))

    print("| Tier | 方法 | 最小达标 artifact | size GiB | kld | top% | vs v0.1 | active |")
    print("|---|---|---|---|---|---|---|---|")
    for tier, status, size, kl, top, active, note in rows:
        if size is None:
            print(f"| {tier} | FIT v0.2 | {status} | — | — | — | — | {note} |")
            continue
        v01_size, v01_kl, v01_top = V01[tier]
        saving = (v01_size - size) / v01_size * 100
        print(f"| {tier} | FIT v0.2 FS | {status} | {size:.2f} | {kl:.4f} | {top:.2f} | "
              f"{saving:+.1f}% size | {active} |")
        print(f"| {tier} | FIT v0.1 | | {v01_size:.2f} | {v01_kl:.4f} | {v01_top:.2f} | — | |")
        p_size, p_kl, p_top = PRESET[tier]
        print(f"| {tier} | preset | | {p_size:.2f} | {p_kl:.4f} | {p_top:.2f} | — | |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
