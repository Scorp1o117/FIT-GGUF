#!/usr/bin/env python3
"""Mechanically evaluate the preregistered P6 gates (G1-G6)."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

REPO = Path("/run/media/s117/OS/FIT-GGUF")
P6 = REPO / "experiments/2026-08-30-p6-iq2-span-fix"
P4 = REPO / "experiments/2026-08-29-p4-release-batch"
BUNDLE = REPO / "Qwen3.8-27B-Uncensored-FIT-GGUF"
MODEL = "Qwen3.8-27B-Uncensored"
DOMAINS = ("wiki_test", "wiki_valid", "chinese", "code", "agent_chat")

OLD_KLD = {"FIT-8G": 0.566418, "FIT-8.5G": 0.575169, "FIT-9.5G": 0.309781, "FIT-10G": 0.239082}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def parse_kld(path: Path) -> float:
    match = re.search(r"^Mean\s+KLD:\s+([0-9.]+)", path.read_text(encoding="utf-8"), re.MULTILINE)
    if match is None:
        raise ValueError(f"missing Mean KLD in {path}")
    return float(match.group(1))


def macro_kld(tier: str) -> float:
    return sum(parse_kld(P4 / f"artifacts/logs/eval-{tier}-{d}.log") for d in DOMAINS) / len(DOMAINS)


def main() -> int:
    gates: dict[str, object] = {}
    tiers = [
        line.split(",")
        for line in (P6 / "tiers.csv").read_text(encoding="utf-8").splitlines()[1:]
    ]
    baseline = json.loads(
        (P6 / "results/baseline/p4-results.json").read_text(encoding="utf-8")
    )["artifacts"]

    # G1: plans
    g1 = True
    plans: dict[str, dict] = {}
    for tier, _lower, _upper, target in tiers:
        plan = json.loads((P6 / f"tiers/{tier}/fit-plan.json").read_text(encoding="utf-8"))
        plans[tier] = plan
        expected_label = tier.removeprefix("FIT-")
        ok = (
            plan["policy"] == "balanced"
            and plan["predicted_size_bytes"] <= int(target)
            and plan["target_bytes"] == int(target)
            and str(plan["suggested_filename"]).startswith(f"{MODEL}-FIT-{expected_label}-")
            and plan["dominant_qtype"]
            and plan["model_name"] == MODEL
        )
        g1 = g1 and ok
    gates["G1"] = {
        "description": "4 balanced plans with exact tier labels",
        "pass": g1 and len(tiers) == 4,
        "conditions": [g1, len(tiers) == 4],
    }

    # G2: zero-byte quantizes
    g2 = True
    for tier, *_ in tiers:
        plan = plans[tier]
        artifact = BUNDLE / str(plan["suggested_filename"])
        record = json.loads(
            (BUNDLE / (plan["suggested_filename"] + ".quantize-record.json")).read_text(encoding="utf-8")
        )
        recorded = (P6 / f"tiers/{tier}/artifact-sha256.txt").read_text(encoding="utf-8").split()[0]
        ok = (
            record["size_matches_expectation"] is True
            and record["size_bytes"] == plan["predicted_size_bytes"]
            and record["sha256"] == recorded == sha256(artifact)
        )
        g2 = g2 and ok
    gates["G2"] = {
        "description": "4 FIT artifacts at zero-byte size error",
        "pass": g2,
        "conditions": [g2],
    }

    # G3: q3_k-free override sets
    g3 = True
    g3_counts: dict[str, dict[str, int]] = {}
    for tier, *_ in tiers:
        types = (P6 / f"tiers/{tier}/fit-tensor-types.txt").read_text(encoding="utf-8")
        used = [line.rsplit("=", 1)[1] for line in types.splitlines() if "=" in line]
        counts = {k: used.count(k) for k in sorted(set(used))}
        g3_counts[tier] = counts
        g3 = g3 and "q3_k" not in counts
    gates["G3"] = {
        "description": "no q3_k tensors in any override set",
        "pass": g3,
        "conditions": [g3],
        "qtype_counts": g3_counts,
    }

    # G4: adoption (strictly better macro KLD than the replaced tier)
    new_kld = {tier: macro_kld(tier) for tier, *_ in tiers}
    g4_details = {}
    g4 = True
    for tier, *_ in tiers:
        old = float(baseline[tier]["macro_kld"])
        g4_details[tier] = {"old": old, "new": new_kld[tier]}
        g4 = g4 and new_kld[tier] < old
    gates["G4"] = {
        "description": "each new tier strictly beats the tier it replaces",
        "pass": g4,
        "conditions": [g4],
        "detail": g4_details,
    }

    # G5: whole-curve monotonicity (14 FIT tiers by target size)
    all_tiers = [
        line.split(",")
        for line in (P4 / "tiers.csv").read_text(encoding="utf-8").splitlines()[1:]
    ]
    seq_pairs = []
    for tier, _lo, _up, target in all_tiers:
        if tier in new_kld:
            seq_pairs.append((int(target), new_kld[tier]))
        else:
            seq_pairs.append((int(target), float(baseline[tier]["macro_kld"])))
    seq_pairs.sort()
    seq = [k for _, k in seq_pairs]
    g5 = all(a >= b for a, b in zip(seq, seq[1:]))
    gates["G5"] = {
        "description": "macro KLD non-increasing across all 14 FIT tiers",
        "pass": g5,
        "conditions": [seq],
        "sequence": seq,
    }

    # G6: anti-domination for 8G / 8.5G vs the IQ2_XXS reference
    ref = float(baseline["IQ2_XXS"]["macro_kld"])
    g6 = new_kld["FIT-8G"] < ref and new_kld["FIT-8.5G"] < ref
    gates["G6"] = {
        "description": "8G and 8.5G beat the IQ2_XXS reference",
        "pass": g6,
        "conditions": [new_kld["FIT-8G"] < ref, new_kld["FIT-8.5G"] < ref],
        "reference": ref,
    }

    verdict = {
        "schema_version": 1,
        "gates": gates,
        "all_pass": all(g["pass"] for g in gates.values()),  # type: ignore[index]
    }
    (P6 / "results/gate-verdict.json").write_text(
        json.dumps(verdict, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({name: g["pass"] for name, g in gates.items()}))  # type: ignore[index]
    print("new macro KLD:", json.dumps(new_kld))
    print(f"ALL_PASS={verdict['all_pass']}")
    return 0 if verdict["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
