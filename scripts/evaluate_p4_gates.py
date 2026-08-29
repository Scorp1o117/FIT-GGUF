#!/usr/bin/env python3
"""Mechanically evaluate the preregistered P4 gates (R1-R6)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

REPO = Path("/run/media/s117/OS/FIT-GGUF")
P4 = REPO / "experiments/2026-08-29-p4-release-batch"
RELEASE = REPO / "Qwen3.8-27B-Uncensored-FIT-GGUF"  # release bundle holds the GGUFs
MODEL = "orcarouter-Qwen3.8-27B-Uncensored"
DOMAINS = ("wiki_test", "wiki_valid", "chinese", "code", "agent_chat")


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
    tiers = [
        line.split(",")
        for line in (P4 / "tiers.csv").read_text(encoding="utf-8").splitlines()[1:]
    ]
    presets = [
        line.strip()
        for line in (P4 / "refs.csv").read_text(encoding="utf-8").splitlines()[1:]
    ]

    # R1: plans
    r1_ok = True
    for tier, _lower, _upper, target in tiers:
        plan = json.loads((P4 / f"tiers/{tier}/fit-plan.json").read_text(encoding="utf-8"))
        expected_label = tier.removeprefix("FIT-")
        expected_name = f"{MODEL}-FIT-{expected_label}-"
        ok = (
            plan["policy"] == "balanced"
            and plan["predicted_size_bytes"] <= int(target)
            and plan["target_bytes"] == int(target)
            and str(plan["suggested_filename"]).startswith(expected_name)
            and plan["dominant_qtype"] is not None
        )
        r1_ok = r1_ok and ok
    check(gates, "R1", "11 balanced plans with exact tier names", r1_ok and len(tiers) == 11)

    # R2: FIT quantizes zero-byte
    r2_ok = True
    for tier, *_ in tiers:
        plan = json.loads((P4 / f"tiers/{tier}/fit-plan.json").read_text(encoding="utf-8"))
        artifact = RELEASE / str(plan["suggested_filename"])
        record = json.loads(
            Path(str(artifact) + ".quantize-record.json").read_text(encoding="utf-8")
        )
        recorded = (P4 / f"tiers/{tier}/artifact-sha256.txt").read_text(encoding="utf-8").split()[0]
        ok = (
            record["size_matches_expectation"] is True
            and record["size_bytes"] == plan["predicted_size_bytes"]
            and record["sha256"] == recorded == sha256(artifact)
        )
        r2_ok = r2_ok and ok
    check(gates, "R2", "11 FIT artifacts at zero-byte error", r2_ok)

    # R3: preset references vs ladder predictions
    ladder = json.loads(
        (REPO / "experiments/2026-08-29-p2-full-envelope/preset-ladder.json")
        .read_text(encoding="utf-8")
    )
    r3_ok = True
    for preset in presets:
        artifact = RELEASE / "refs" / f"{preset}.gguf"
        ok = (
            artifact.stat().st_size == ladder["ladder"][preset]["predicted_size_bytes"]
            and (P4 / f"refs/{preset}-sha256.txt").read_text(encoding="utf-8").split()[0]
            == sha256(artifact)
        )
        r3_ok = r3_ok and ok
    check(gates, "R3", "14 reference presets match ladder predictions", r3_ok)

    # R4: KL evaluations all parsed
    results = json.loads((P4 / "results/p4-results.json").read_text(encoding="utf-8"))
    artifacts = results["artifacts"]
    complete = len(artifacts) == 25 and all(
        set(a["domains"]) == set(DOMAINS) for a in artifacts.values()
    )
    refs_ok = len(results["bf16_references"]) == 5
    slices_recorded = (P4 / "slices-sha256.txt").is_file()
    check(gates, "R4", "25 artifacts x 5 domains evaluated", complete and refs_ok and slices_recorded)

    # R5: reporting outputs
    check(
        gates, "R5", "comparison table and curves rendered",
        (P4 / "results/comparison-table.md").is_file(),
        (P4 / "results/kl-curve.png").is_file(),
        (P4 / "results/sametop-curve.png").is_file(),
    )

    # R6: test suite
    run = subprocess.run(
        ["python3", "-m", "pytest", "tests/", "-q", "--tb=no"],
        cwd=REPO, capture_output=True, text=True,
    )
    tail = run.stdout.strip().splitlines()[-1]
    check(
        gates, "R6", "test suite green",
        run.returncode == 0,
        "passed" in tail and "failed" not in tail and "error" not in tail,
    )

    verdict = {
        "schema_version": 1,
        "gates": gates,
        "all_pass": all(g["pass"] for g in gates.values()),  # type: ignore[index]
    }
    (P4 / "gate-verdict.json").write_text(
        json.dumps(verdict, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({name: g["pass"] for name, g in gates.items()}))  # type: ignore[index]
    print(f"ALL_PASS={verdict['all_pass']}")
    return 0 if verdict["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
