#!/usr/bin/env python3
"""Mechanically evaluate the preregistered M13 gate on m13-results.json.

Gate (frozen in experiments/2026-08-28-m13-budget-rule/README.md):
  1 (direction): FIT-25 original wins, FIT-50 v0.1b wins, FIT-75 v0.1b wins
                 (macro KL over five domains per budget).
  2 (composite): rule's 15-cell macro KL < "all original" and < "all v0.1b".
  3 (per-cell guard): no selected-policy cell exceeds the unselected policy's
                      KL by more than 25%.
Failure branch: role-matched early/late block swap ablation only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

DOMAINS = ("wiki_test", "wiki_valid", "chinese", "code", "agent_chat")
BUDGETS = ("fit25", "fit50", "fit75")


def macro(values: dict[str, float]) -> float:
    return sum(values[d] for d in DOMAINS) / len(DOMAINS)


def main() -> None:
    results = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["results"]

    kl = {
        (b, a): {d: results[d][f"{a}-{b}"]["mean_kld"] for d in DOMAINS}
        for b in BUDGETS
        for a in ("orig", "v01b")
    }
    macro_kl = {f"{b}/{a}": macro(kl[(b, a)]) for b in BUDGETS for a in ("orig", "v01b")}

    selection = {"fit25": "orig", "fit50": "v01b", "fit75": "v01b"}
    unselected = {b: ("v01b" if a == "orig" else "orig") for b, a in selection.items()}

    gate1 = {
        "fit25": macro_kl["fit25/orig"] < macro_kl["fit25/v01b"],
        "fit50": macro_kl["fit50/v01b"] < macro_kl["fit50/orig"],
        "fit75": macro_kl["fit75/v01b"] < macro_kl["fit75/orig"],
    }

    rule_cells = [kl[(b, selection[b])][d] for b in BUDGETS for d in DOMAINS]
    all_orig = [kl[(b, "orig")][d] for b in BUDGETS for d in DOMAINS]
    all_v01b = [kl[(b, "v01b")][d] for b in BUDGETS for d in DOMAINS]
    rule_macro = sum(rule_cells) / 15
    orig_macro = sum(all_orig) / 15
    v01b_macro = sum(all_v01b) / 15
    gate2 = rule_macro < orig_macro and rule_macro < v01b_macro

    regressions = {
        f"{b}/{d}": (kl[(b, selection[b])][d] / kl[(b, unselected[b])][d] - 1) * 100
        for b in BUDGETS
        for d in DOMAINS
    }
    worst_cell = max(regressions, key=regressions.get)
    gate3 = regressions[worst_cell] <= 25.0

    verdict = {
        "macro_kl_per_budget": macro_kl,
        "selection": selection,
        "gate1_direction": gate1,
        "gate2_composite": {
            "rule_15cell_macro": rule_macro,
            "all_original_macro": orig_macro,
            "all_v01b_macro": v01b_macro,
            "pass": gate2,
        },
        "gate3_worst_cell": {
            "cell": worst_cell,
            "regression_percent": regressions[worst_cell],
            "pass": gate3,
        },
        "all_pass": all(gate1.values()) and gate2 and gate3,
        "decision": (
            "ACCEPTED: budget-conditional rule frozen as first deployment policy"
            if (all(gate1.values()) and gate2 and gate3)
            else "NOT ACCEPTED: role-matched early/late block swap ablation is the only permitted next step"
        ),
    }
    print(json.dumps(verdict, indent=2, ensure_ascii=False))
    out = Path(sys.argv[1]).parent / "gate-verdict.json"
    out.write_text(json.dumps(verdict, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
