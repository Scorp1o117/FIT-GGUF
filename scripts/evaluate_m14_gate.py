#!/usr/bin/env python3
"""Mechanically evaluate the preregistered M14 gate on m14-results.json.

Frozen in experiments/2026-08-28-m14-swap-ablation/README.md:
  delta = 1% relative macro-KL ROPE.
  term(OE) = (KL(O50)-KL(OE50))/KL(O50); term(BL) = (KL(BL50)-KL(B50))/KL(B50)
  S50 = mean of the two terms.
  Gate 1 (mechanism): S50 >= 1% and min(term) >= -1%.
  Gate 2 (domain robustness): per-domain synthetic effect >= -1% in >=4/5
          domains; no swap artifact worse than its skeleton by >25% in any cell.
  Gate 3 (negative control): (KL(SHUF50)-KL(OE50))/KL(OE50) >= 1%.
  Secondary (ungated): S75 and its terms; predicted within +-1% and < S50.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

DOMAINS = ("wiki_test", "wiki_valid", "chinese", "code", "agent_chat")
DELTA = 0.01


def rel(new: float, old: float) -> float:
    return (new - old) / old


def main() -> None:
    results = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["results"]

    def kl(variant: str) -> dict[str, float]:
        return {d: results[d][variant]["mean_kld"] for d in DOMAINS}

    def macro(values: dict[str, float]) -> float:
        return sum(values.values()) / len(DOMAINS)

    O50, B50 = kl("orig-fit50"), kl("v01b-fit50")
    OE50, BL50, SH50 = kl("oe-fit50"), kl("bl-fit50"), kl("shuf-fit50")
    O75, B75 = kl("orig-fit75"), kl("v01b-fit75")
    OE75, BL75 = kl("oe-fit75"), kl("bl-fit75")

    # Preregistered formula: S = [(KL(O)-KL(OE))/KL(O) + (KL(BL)-KL(B))/KL(B)] / 2
    # First term positive when O->E improves over O; second term positive when
    # removing early placement from B hurts (i.e. early placement helps).
    term_oe = (macro(O50) - macro(OE50)) / macro(O50)
    term_bl = (macro(BL50) - macro(B50)) / macro(B50)
    s50 = (term_oe + term_bl) / 2
    gate1 = s50 >= DELTA and min(term_oe, term_bl) >= -DELTA

    e_d = {
        d: (
            (O50[d] - OE50[d]) / O50[d] + (BL50[d] - B50[d]) / B50[d]
        ) / 2
        for d in DOMAINS
    }
    domain_ok = sum(v >= -DELTA for v in e_d.values()) >= 4
    regressions = {}
    for arm, skel in (("oe-fit50", O50), ("bl-fit50", B50), ("oe-fit75", O75), ("bl-fit75", B75)):
        arm_kl = kl(arm)
        for d in DOMAINS:
            regressions[f"{arm}/{d}"] = (arm_kl[d] / skel[d] - 1) * 100
    worst = max(regressions, key=regressions.get)
    gate2 = domain_ok and regressions[worst] <= 25.0

    gate3 = rel(macro(SH50), macro(OE50)) >= DELTA  # positive = shuffle worse than O->E

    term_oe75 = (macro(O75) - macro(OE75)) / macro(O75)
    term_bl75 = (macro(BL75) - macro(B75)) / macro(B75)
    s75 = (term_oe75 + term_bl75) / 2

    all_pass = gate1 and gate2 and gate3
    verdict = {
        "delta_rope": DELTA,
        "fit50": {
            "term_oe": round(term_oe, 5),
            "term_bl": round(term_bl, 5),
            "S50": round(s50, 5),
            "per_domain_effect": {d: round(v, 5) for d, v in e_d.items()},
            "gate1_mechanism": gate1,
            "gate2_domain_robustness": gate2,
            "gate2_worst_regression_cell": worst,
            "gate2_worst_regression_percent": round(regressions[worst], 2),
            "gate3_negative_control": gate3,
            "macro_kl": {
                "O": round(macro(O50), 6), "B": round(macro(B50), 6),
                "OE": round(macro(OE50), 6), "BL": round(macro(BL50), 6),
                "SHUF": round(macro(SH50), 6),
            },
        },
        "fit75_secondary": {
            "term_oe": round(term_oe75, 5),
            "term_bl": round(term_bl75, 5),
            "S75": round(s75, 5),
            "within_rope_prediction": abs(s75) < DELTA,
            "s50_gt_s75_prediction": s50 > s75,
        },
        "all_pass": all_pass,
        "decision": (
            "ACCEPTED: early-vs-late position is a causal component of v0.1b's FIT-50 gain"
            if all_pass
            else "NOT ACCEPTED: early/late position is not the confirmed mechanism"
        ),
    }
    print(json.dumps(verdict, indent=2, ensure_ascii=False))
    out = Path(sys.argv[1]).parent / "gate-verdict.json"
    out.write_text(json.dumps(verdict, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
