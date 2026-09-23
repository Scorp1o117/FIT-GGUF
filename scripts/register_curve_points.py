#!/usr/bin/env python3
"""Ingest already-paid sweep measurements into a calibration bundle's curve.

A sweep writes a report next to its artifacts and moves on.  ``tier-search``
reads only ``curve-points.jsonl``, so unless the sweep also registered its rows
the search cannot see them — the artifact is on disk, the five-domain eval was
paid for, and the tier still reports a worse point as its winner.

This tool closes that gap for sweeps that ran before registration existed.  It
is evidence-driven: every field comes from the sweep's own report, and the floor
regime comes from the ``floors`` flag the sweep recorded per row, never from a
guess about what the run "must" have used.

    python scripts/register_curve_points.py \\
        --bundle experiments/<run>/calibration \\
        --report /path/to/floor5-report.json [--report ...] [--tier <tier>]

Idempotent: a point already in the ledger is skipped, so re-running after adding
a new report cannot double-count anything.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from fit_gguf.calibrate import register_curve_point  # noqa: E402

GIB = 2**30


def point_id_for(report: Path, row: dict, tier: str | None) -> str:
    """``<tag>-<size>G`` — the tag is the report's own name, the size the request.

    The requested size, not the realized one: that is the name the eval log and
    the artifact already carry, and a point id that disagrees with them would
    make the ledger unjoinable to its own evidence.
    """
    tag = report.name[: -len("-report.json")] if report.name.endswith("-report.json") else report.stem
    stem = f"{tag}-{tier}-{row['requested_gib']:g}G" if tier else f"{tag}-{row['requested_gib']:g}G"
    return stem


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path, required=True, help="Calibration bundle")
    ap.add_argument("--report", type=Path, action="append", required=True,
                    help="Sweep report JSON; repeatable")
    ap.add_argument("--tier", default=None,
                    help="Optional tier to scope the points to (makes them restartable)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    total = added = skipped = 0
    for report in args.report:
        payload = json.loads(report.read_text(encoding="utf-8"))
        for row in payload["rows"]:
            total += 1
            if row.get("status") != "ok":
                print(f"  skip  {report.name} {row['requested_gib']:g}G — {row.get('status')}")
                skipped += 1
                continue
            pid = point_id_for(report, row, args.tier)
            if args.dry_run:
                print(f"  would add {pid:28} {row['actual_size_bytes'] / GIB:6.2f} GiB "
                      f"KL={row['macro_kl']:.4f} floors={row['floors']}")
                added += 1
                continue
            ok = register_curve_point(
                args.bundle, pid,
                {
                    "size_bytes": row["actual_size_bytes"],
                    "artifact_sha256": row["artifact_sha256"],
                    "macro_kl": row["macro_kl"],
                    "same_top": row["same_top"],
                    "per_domain": row.get("per_domain"),
                },
                always_active_floors=bool(row["floors"]),
                tier=args.tier,
            )
            if ok:
                added += 1
                print(f"  add   {pid:28} {row['actual_size_bytes'] / GIB:6.2f} GiB "
                      f"KL={row['macro_kl']:.4f} floors={row['floors']}")
            else:
                skipped += 1
                print(f"  have  {pid:28} already registered")
    verb = "would add" if args.dry_run else "added"
    print(f"\n[register] {total} row(s): {verb} {added}, skipped {skipped}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
