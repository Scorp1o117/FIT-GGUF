#!/usr/bin/env python3
"""Mechanically evaluate the preregistered P2 gates (E1-E5)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

REPO = Path("/run/media/s117/OS/FIT-GGUF")
P2 = REPO / "experiments/2026-08-29-p2-full-envelope"
PROBES = P2 / "probes"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def check(gates: dict, gate: str, description: str, *conditions: bool) -> None:
    gates[gate] = {
        "description": description,
        "pass": all(conditions),
        "conditions": [bool(c) for c in conditions],
    }


def main() -> int:
    gates: dict[str, object] = {}

    e1 = (P2 / "bf16-sha256.txt").read_text(encoding="utf-8").splitlines()
    sha_line = e1[0].split()[0]
    size_line = [line for line in e1 if line.startswith("size=")][0]
    check(
        gates, "E1", "BF16 conversion recorded",
        len(sha_line) == 64,
        size_line == "size=53808282624",
    )

    e2 = json.loads((P2 / "e2prime-record.json").read_text(encoding="utf-8"))
    check(
        gates, "E2prime", "imatrix provenance and coverage",
        e2["imatrix_sha256"] == "0ee5b10bd0c2fa2127c6f4b43dbfe1efd71e383b63217af9dade1de36599f1c1",
        e2["imatrix_entries"] == 496,
        e2["layer_matrix_coverage"] == "complete (496/496)",
        e2["fallback_without_imatrix_entry"] == ["output.weight", "token_embd.weight"],
        e2["fallback_assignments"] == {"output.weight": "q6_K", "token_embd.weight": "iq3_s"},
    )

    e3 = json.loads((P2 / "preset-ladder.json").read_text(encoding="utf-8"))
    check(
        gates, "E3", "preset ladder sweep",
        e3["pass"] is True,
        e3["accepted_count"] == 23,
        e3["all_payload_ok"] is True,
        e3["rejected"] == [],
        e3["inversions"] == [],
    )

    e4_details = {}
    e4_conds = []
    for name in ("probe-low", "probe-mid", "probe-top"):
        plan = json.loads((PROBES / name / "fit50-plan.json").read_text(encoding="utf-8"))
        record = json.loads(
            (REPO / "artifacts/fit/p2-probes" / f"{name}-FIT50.gguf.quantize-record.json")
            .read_text(encoding="utf-8")
        )
        artifact_sha = (PROBES / name / "artifact-sha256.txt").read_text(encoding="utf-8").split()[0]
        ok = (
            record["size_matches_expectation"] is True
            and record["size_bytes"] == plan["predicted_size_bytes"]
            and record["sha256"] == artifact_sha
            and plan["unused_bytes"] >= 0
        )
        e4_details[name] = {
            "target": plan["target_bytes"],
            "predicted": plan["predicted_size_bytes"],
            "actual": record["size_bytes"],
            "unused": plan["unused_bytes"],
            "selected": plan["selected_count"],
            "sha256": record["sha256"],
            "ok": ok,
        }
        e4_conds.append(ok)
    check(gates, "E4", "three zero-byte probe quantizes", *e4_conds)

    pytest_run = subprocess.run(
        ["python3", "-m", "pytest", "tests/", "-q", "--tb=no"],
        cwd=REPO, capture_output=True, text=True,
    )
    pytest_tail = pytest_run.stdout.strip().splitlines()[-1]
    check(
        gates, "E5", "test suite unchanged and green",
        pytest_run.returncode == 0,
        "passed" in pytest_tail and "failed" not in pytest_tail and "error" not in pytest_tail,
    )
    e5_detail = pytest_tail

    verdict = {
        "schema_version": 1,
        "gates": gates,
        "all_pass": all(g["pass"] for g in gates.values()),  # type: ignore[index]
        "e4_details": e4_details,
        "e5_pytest": e5_detail,
    }
    (P2 / "gate-verdict.json").write_text(
        json.dumps(verdict, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({name: g["pass"] for name, g in gates.items()}))  # type: ignore[index]
    print(json.dumps(e4_details, indent=2))
    print(f"ALL_PASS={verdict['all_pass']}")
    return 0 if verdict["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
