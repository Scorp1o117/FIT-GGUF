#!/usr/bin/env python3
"""Mechanically evaluate the preregistered P5 gates (K1-K5)."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

REPO = Path("/run/media/s117/OS/FIT-GGUF")
P5 = REPO / "experiments/2026-08-30-p5-kfree-12-13.5"
P4 = REPO / "experiments/2026-08-29-p4-release-batch"
BUNDLE = REPO / "Qwen3.8-27B-Uncensored-FIT-GGUF"
MODEL = "Qwen3.8-27B-Uncensored"
DOMAINS = ("wiki_test", "wiki_valid", "chinese", "code", "agent_chat")

Kinds = {"q2_k", "q3_k"}


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
        for line in (P5 / "tiers.csv").read_text(encoding="utf-8").splitlines()[1:]
    ]
    baseline = json.loads(
        (P5 / "results/baseline-k-based/p4-results.json").read_text(encoding="utf-8")
    )["artifacts"]

    # K1: plans
    k1 = True
    plans: dict[str, dict] = {}
    for tier, _lower, _upper, target in tiers:
        plan = json.loads((P5 / f"tiers/{tier}/fit-plan.json").read_text(encoding="utf-8"))
        plans[tier] = plan
        expected_label = tier.removeprefix("FIT-")
        ok = (
            plan["policy"] == "balanced"
            and plan["predicted_size_bytes"] <= int(target)
            and plan["target_bytes"] == int(target)
            and str(plan["suggested_filename"]).startswith(f"{MODEL}-FIT-{expected_label}-")
            and plan["dominant_qtype"] in ("iq3_s", "iq4_xs")
            and plan["model_name"] == MODEL
        )
        k1 = k1 and ok
    gates["K1"] = {
        "description": "3 balanced K-free plans with exact tier labels",
        "pass": k1 and len(tiers) == 3,
        "conditions": [k1, len(tiers) == 3],
    }

    # K2: zero-byte quantizes
    k2 = True
    for tier, *_ in tiers:
        plan = plans[tier]
        artifact = BUNDLE / str(plan["suggested_filename"])
        record = json.loads(
            (BUNDLE / (plan["suggested_filename"] + ".quantize-record.json")).read_text(encoding="utf-8")
        )
        recorded = (P5 / f"tiers/{tier}/artifact-sha256.txt").read_text(encoding="utf-8").split()[0]
        ok = (
            record["size_matches_expectation"] is True
            and record["size_bytes"] == plan["predicted_size_bytes"]
            and record["sha256"] == recorded == sha256(artifact)
        )
        k2 = k2 and ok
    gates["K2"] = {
        "description": "3 FIT artifacts at zero-byte size error",
        "pass": k2,
        "conditions": [k2],
    }

    # K3: K-free candidate space (tensor-types file has no q2_k/q3_k targets)
    k3 = True
    k3_counts: dict[str, dict[str, int]] = {}
    for tier, *_ in tiers:
        types = (P5 / f"tiers/{tier}/fit-tensor-types.txt").read_text(encoding="utf-8")
        used = [line.rsplit("=", 1)[1] for line in types.splitlines() if "=" in line]
        counts = {k: used.count(k) for k in sorted(set(used))}
        k3_counts[tier] = counts
        k3 = k3 and not (set(used) & Kinds)
    gates["K3"] = {
        "description": "no q2_k/q3_k tensors in the override set",
        "pass": k3,
        "conditions": [k3],
        "qtype_counts": k3_counts,
    }

    # K4: adoption (strictly better macro KLD than the K-based tier)
    new_kld = {tier: macro_kld(tier) for tier, *_ in tiers}
    k4_details = {}
    k4 = True
    for tier, *_ in tiers:
        old = float(baseline[tier]["macro_kld"])
        k4_details[tier] = {"old_k_based": old, "new_k_free": new_kld[tier]}
        k4 = k4 and new_kld[tier] < old
    gates["K4"] = {
        "description": "each K-free tier strictly beats its K-based baseline",
        "pass": k4,
        "conditions": [k4],
        "detail": k4_details,
    }

    # K5: curve sanity (macro KLD non-increasing 12G -> 12.5G -> 13G -> 13.5G)
    kld12g = float(baseline["FIT-12G"]["macro_kld"])
    seq = [kld12g, new_kld["FIT-12.5G"], new_kld["FIT-13G"], new_kld["FIT-13.5G"]]
    gates["K5"] = {
        "description": "macro KLD non-increasing across FIT-12G..13.5G",
        "pass": all(a >= b for a, b in zip(seq, seq[1:])),
        "conditions": [seq],
        "sequence": seq,
    }

    verdict = {
        "schema_version": 1,
        "gates": gates,
        "all_pass": all(g["pass"] for g in gates.values()),  # type: ignore[index]
    }
    (P5 / "results/gate-verdict.json").write_text(
        json.dumps(verdict, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({name: g["pass"] for name, g in gates.items()}))  # type: ignore[index]
    print("new macro KLD:", json.dumps(new_kld))
    print(f"ALL_PASS={verdict['all_pass']}")
    return 0 if verdict["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
