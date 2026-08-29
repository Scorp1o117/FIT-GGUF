#!/usr/bin/env python3
"""Mechanically evaluate the preregistered M15 gates on m15-results.json.

Frozen in experiments/2026-08-29-m15-random-baseline/README.md; delta = 1%.
H75 (collapse):  G75a |O-B|/H75<=1%; G75b |Rmean-H75|/H75<=1%;
                 G75c random range/H75<=2%; guardrail 25%.
H25 (advantage): G25a mean-random minus O >= 1% of O; G25b O beats >=2/3 seeds;
                 G25c any winning seed within 1% ROPE; guardrail 25%.
Secondary: paired random spread at 75 < at 25 (reported, not gated).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

DOMAINS = ("wiki_test", "wiki_valid", "chinese", "code", "agent_chat")
DELTA = 0.01


def main() -> None:
    results = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["results"]

    def macro(v: str) -> float:
        return sum(results[d][v]["mean_kld"] for d in DOMAINS) / len(DOMAINS)

    def worst_vs_guard(variants: tuple[str, ...], floor: str) -> tuple[str, float]:
        cells = {
            f"{v}/{d}": (results[d][v]["mean_kld"] / results[d][floor]["mean_kld"] - 1) * 100
            for v in variants
            for d in DOMAINS
        }
        worst = max(cells, key=cells.get)
        return worst, cells[worst]

    # ---- H75 ----
    o75, b75 = macro("o75"), macro("b75")
    r75 = [macro(f"r{i}-75") for i in (1, 2, 3)]
    h75 = (o75 + b75) / 2
    g75a = abs(o75 - b75) / h75 <= DELTA
    g75b = abs(sum(r75) / 3 - h75) / h75 <= DELTA
    g75c = (max(r75) - min(r75)) / h75 <= 2 * DELTA
    w75, w75pct = worst_vs_guard(("r1-75", "r2-75", "r3-75"), "b75")  # worse of O/B checked below
    w75b, w75bpct = worst_vs_guard(("r1-75", "r2-75", "r3-75"), "o75")
    guard75 = max(w75pct, w75bpct) <= 25.0
    h75_pass = g75a and g75b and g75c and guard75

    # ---- H25 ----
    o25, b25 = macro("o25"), macro("b25")
    r25 = [macro(f"r{i}-25") for i in (1, 2, 3)]
    mean25 = sum(r25) / 3
    g25a = (mean25 - o25) / o25 >= DELTA
    seeds_beaten = sum(1 for r in r25 if o25 < r)
    g25b = seeds_beaten >= 2
    over = [r for r in r25 if r < o25]
    g25c = all((o25 - r) / o25 <= DELTA for r in over)
    w25, w25pct = worst_vs_guard(("r1-25", "r2-25", "r3-25"), "b25")
    w25b, w25bpct = worst_vs_guard(("r1-25", "r2-25", "r3-25"), "o25")
    guard25 = max(w25pct, w25bpct) <= 25.0
    h25_pass = g25a and g25b and g25c and guard25
    strong25 = seeds_beaten == 3 and g25a

    # ---- paired secondary: spread shrink ----
    spread25 = (max(r25) - min(r25)) / o25
    spread75 = (max(r75) - min(r75)) / h75
    spread_shrinks = spread75 < spread25

    verdict = {
        "delta_rope": DELTA,
        "fit75_collapse": {
            "O75": round(o75, 6), "B75": round(b75, 6), "H75": round(h75, 6),
            "random_macros_75": [round(r, 6) for r in r75],
            "G75a_OB_gap": round(abs(o75 - b75) / h75, 5),
            "G75b_mean_gap": round(abs(sum(r75) / 3 - h75) / h75, 5),
            "G75c_range": round((max(r75) - min(r75)) / h75, 5),
            "guard_worst_cell": max(w75, w75b), "guard_worst_percent": round(max(w75pct, w75bpct), 2),
            "g75a": g75a, "g75b": g75b, "g75c": g75c, "guard75": guard75,
            "h75_collapse_confirmed": h75_pass,
        },
        "fit25_advantage": {
            "O25": round(o25, 6), "B25": round(b25, 6),
            "random_macros_25": [round(r, 6) for r in r25],
            "random_mean_25": round(mean25, 6),
            "G25a_margin": round((mean25 - o25) / o25, 5),
            "seeds_beaten": seeds_beaten,
            "g25a": g25a, "g25b": g25b, "g25c": g25c, "guard25": guard25,
            "strong_support": strong25,
            "h25_advantage_confirmed": h25_pass,
        },
        "secondary_paired_spread": {
            "spread25": round(spread25, 5), "spread75": round(spread75, 5),
            "spread_shrinks": spread_shrinks,
        },
        "h75_confirmed": h75_pass,
        "h25_confirmed": h25_pass,
        "decision": (
            " | ".join(filter(None, [
                "H75: allocation insensitivity at high budget ACCEPTED" if h75_pass else "H75: collapse NOT confirmed",
                "H25: original FIT-25 advantage beyond random variance ACCEPTED" if h25_pass else "H25: advantage NOT beyond random variance",
            ])) + " -> proceed to D-0021 freeze and Granite reveal"
        ),
    }
    print(json.dumps(verdict, indent=2, ensure_ascii=False))
    out = Path(sys.argv[1]).parent / "gate-verdict.json"
    out.write_text(json.dumps(verdict, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
