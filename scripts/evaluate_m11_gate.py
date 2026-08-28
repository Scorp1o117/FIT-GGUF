#!/usr/bin/env python3
"""Evaluate the preregistered M11 gate on holdout-results.json.

Gate (fixed in experiments/2026-08-28-m11-holdout/README.md before evaluation):
  A: block-balanced macro KL < fit50 macro KL
  B: block-balanced macro KL < random mean macro KL, and beats the per-domain
     random mean in >= 4 of 5 domains
  C: no domain where block-balanced KL > fit50 KL by more than 25%
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

DOMAINS = ("wiki_test", "wiki_valid", "chinese", "code", "agent_chat")


def macro(domain_map: dict[str, float]) -> float:
    return sum(domain_map[d] for d in DOMAINS) / len(DOMAINS)


def main() -> None:
    results = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["results"]

    kl = {
        v: {d: results[d][v]["mean_kld"] for d in DOMAINS}
        for v in ("fit50", "block-balanced-fit50", "random-v1", "random-v2", "random-v3")
    }
    random_mean = {d: sum(kl[v][d] for v in ("random-v1", "random-v2", "random-v3")) / 3 for d in DOMAINS}
    macro_kl = {v: macro(kl[v]) for v in kl}
    macro_random_mean = macro(random_mean)

    table = {d: {"fit50": kl["fit50"][d], "v01b": kl["block-balanced-fit50"][d],
                 "random_mean": random_mean[d]} for d in DOMAINS}
    for d in DOMAINS:
        row = table[d]
        row["v01b_vs_fit50"] = f"{(row['v01b'] / row['fit50'] - 1) * 100:+.2f}%"
        row["v01b_vs_random_mean"] = f"{(row['v01b'] / row['random_mean'] - 1) * 100:+.2f}%"

    gate_a = macro_kl["block-balanced-fit50"] < macro_kl["fit50"]
    wins_vs_random = sum(kl["block-balanced-fit50"][d] < random_mean[d] for d in DOMAINS)
    gate_b = macro_kl["block-balanced-fit50"] < macro_random_mean and wins_vs_random >= 4
    regressions = {
        d: (kl["block-balanced-fit50"][d] / kl["fit50"][d] - 1) * 100
        for d in DOMAINS
    }
    gate_c = all(r <= 25.0 for r in regressions.values())

    verdict = {
        "macro_kl": {**macro_kl, "random_mean": macro_random_mean},
        "per_domain": table,
        "gate_A_confirmatory_win": gate_a,
        "gate_B_beats_random": gate_b,
        "gate_B_domain_wins": wins_vs_random,
        "gate_C_max_regression_percent_vs_fit50": max(regressions.values()),
        "gate_C_pass": gate_c,
        "decision": "CONFIRMED extend to FIT-25/FIT-75" if (gate_a and gate_b and gate_c)
                    else "NOT CONFIRMED -> role-matched early/late block swap ablation",
    }
    print(json.dumps(verdict, indent=2, ensure_ascii=False))
    out = Path(sys.argv[1]).parent / "gate-verdict.json"
    out.write_text(json.dumps(verdict, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
