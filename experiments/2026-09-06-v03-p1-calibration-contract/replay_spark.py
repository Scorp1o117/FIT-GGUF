#!/usr/bin/env python3
"""P1 contract replay: apply fidelity-calibration-v1 to the existing Spark
calibration observations with ZERO new evals (planner freeze requirement).

Consumes experiments/2026-09-06-spark-x25-4tier/results/spark-ladder-summary.json
(point -> size_bytes/macro_kl/same_top) + state/artifact-manifest.txt (point ->
artifact sha, dedup key), derives floors per the machine contract, emits the
replay report + a draft calibration record.
"""

from __future__ import annotations

import hashlib
import json
import sys
from decimal import ROUND_DOWN, Decimal
from pathlib import Path

REPO = Path("/run/media/s117/OS/FIT-GGUF")
sys.path.insert(0, str(REPO / "src"))
from fit_gguf.registry import canonical_json_bytes, sha256_file  # noqa: E402

CONTRACT_PATH = REPO / "src/fit_gguf/contracts/fidelity-calibration-v1.json"
EXP = REPO / "experiments/2026-09-06-spark-x25-4tier"
RESULTS = EXP / "results"
MP = "spark-x25-4b-abliterated-"


def trunc4(v: float) -> float:
    """§5.2: Decimal ROUND_DOWN at 4 decimals (never binary-float floor)."""
    return float(Decimal(repr(v)).quantize(Decimal("0.0001"), rounding=ROUND_DOWN))


def p05(values: list[float]) -> float:
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * 0.05
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def main() -> int:
    contract = json.loads(CONTRACT_PATH.read_text())
    contract_sha = contract["contract_sha256"]
    payload = {k: v for k, v in contract.items() if k != "contract_sha256"}
    assert hashlib.sha256(canonical_json_bytes(payload)).hexdigest() == contract_sha, \
        "machine contract fails its own digest"

    summary = json.loads((RESULTS / "spark-ladder-summary.json").read_text())
    manifest_shas = {}
    for line in (EXP / "state" / "artifact-manifest.txt").read_text().splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[0].startswith(MP):
            manifest_shas[parts[0][len(MP):]] = parts[2]

    anchors = contract["tier_kl_anchors"]
    lo, hi = contract["window_rule"]["low_factor"], contract["window_rule"]["high_factor"]
    min_n = contract["validation_min_samples"]

    # unique observations by artifact sha (§5.5)
    seen_art: set[str] = set()
    points = []
    for p in summary["points"]:
        art = manifest_shas.get(p["point"])
        if art is None or art in seen_art:
            continue
        seen_art.add(art)
        points.append(p)

    tiers = {}
    for tier, anchor in anchors.items():
        w_lo, w_hi = anchor * lo, anchor * hi
        samples = [p for p in points if w_lo <= p["macro_kl"] <= w_hi]
        n = len(samples)
        if n == 0:
            tiers[tier] = {
                "window": [round(w_lo, 6), round(w_hi, 6)], "sample_count": 0,
                "floor": None, "floor_method": None, "witness": None,
                "validation_status": "candidate",
                "failure": "INSUFFICIENT_WINDOW",
            }
            continue
        tops = [p["same_top"] for p in samples]
        if n >= min_n:
            floor = trunc4(p05(tops))
            method = "empirical_p5"
        else:
            floor = trunc4(min(tops))
            method = "min_fallback"
        witness_pts = [p for p in samples
                       if p["macro_kl"] <= anchor and p["same_top"] >= floor]
        has_witness = len(witness_pts) > 0
        ok = n >= min_n and has_witness
        tiers[tier] = {
            "window": [round(w_lo, 6), round(w_hi, 6)],
            "sample_count": n,
            "samples": [{"point": p["point"], "macro_kl": p["macro_kl"],
                         "same_top": p["same_top"],
                         "artifact_sha256": manifest_shas[p["point"]][:16] + "…"}
                        for p in sorted(samples, key=lambda p: p["macro_kl"])],
            "floor": floor, "floor_method": method,
            "witness": ({"point": witness_pts[0]["point"],
                         "macro_kl": witness_pts[0]["macro_kl"],
                         "macro_same_top": witness_pts[0]["same_top"],
                         "pass": True} if has_witness else None),
            "validation_status": "validated" if ok else "candidate",
            "failure": None if ok else ("INSUFFICIENT_WINDOW" if n < min_n
                                        else "WITNESS_MISSING"),
        }

    all_ok = all(t["validation_status"] == "validated" for t in tiers.values())
    overall = "validated" if all_ok else "candidate"
    failures = sorted({f for t in tiers.values() if t.get("failure")
                       for f in [t["failure"]]})

    report = {
        "replay_schema": "fit.calibration_contract_replay.v1",
        "contract_id": contract["contract_id"],
        "calibration_contract_sha256": contract_sha,
        "input": "experiments/2026-09-06-spark-x25-4tier/results/spark-ladder-summary.json (zero new evals)",
        "dedup_key": "artifact_sha256",
        "overall_status": overall,
        "open_failures": failures,
        "tiers": tiers,
        "note": "legacy grandfather entry (A3) — this replay validates the CONTRACT "
        "machinery on real data; Spark does not re-onboard under v1 windows. "
        "Empty quality/balanced windows under [0.85K,1.15K] are the expected "
        "boundary-locality behavior, not an incident.",
    }
    (RESULTS / "contract-replay-spark.json").write_text(
        json.dumps(report, indent=1) + "\n", encoding="utf-8")

    print(f"contract {contract_sha[:16]}…  replay overall: {overall}  failures: {failures}")
    for tier, t in tiers.items():
        w = t["window"]
        print(f"  {tier:<9} window[{w[0]},{w[1]}] n={t['sample_count']} "
              f"floor={t['floor']} method={t['floor_method']} "
              f"witness={'Y' if t['witness'] else 'N'} -> {t['validation_status']}"
              + (f" [{t['failure']}]" if t["failure"] else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
