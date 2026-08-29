#!/usr/bin/env python3
"""Mechanically evaluate the preregistered M16 gates on m16-results.json.

Frozen in experiments/2026-08-29-m16-granite-reveal/README.md; delta = 1%.
G-size is verified by the quantization stage (zero-byte errors recorded there).
G-util: O50 beats random mean >= 1% and >= 2/3 seeds (winner within ROPE).
G-bal: B50 beats O50 >= 1%.
Guardrail: no artifact worse than worse(O50,B50) by > 25% in any domain.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

DOMAINS = ("wiki_test", "wiki_valid", "chinese", "code", "agent_chat")
DELTA = 0.01

def main() -> None:
    results = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["results"]

    def macro(v: str) -> float:
        return sum(results[d][v]["mean_kld"] for d in DOMAINS) / len(DOMAINS)

    o50, b50 = macro("o50"), macro("b50")
    r50 = [macro(f"r{i}-50") for i in (1, 2, 3)]
    mean50 = sum(r50) / 3

    g_util_a = (mean50 - o50) / o50 >= DELTA
    beaten = sum(1 for r in r50 if o50 < r)
    g_util_b = beaten >= 2
    over = [r for r in r50 if r < o50]
    g_util_c = all((o50 - r) / o50 <= DELTA for r in over)
    g_util = g_util_a and g_util_b and g_util_c

    g_bal = (b50 - o50) / o50 <= -DELTA

    cells = {}
    for v in ("r1-50", "r2-50", "r3-50", "o50", "b50"):
        for d in DOMAINS:
            floor = min(results[d]["o50"]["mean_kld"], results[d]["b50"]["mean_kld"])
            cells[f"{v}/{d}"] = (results[d][v]["mean_kld"] / floor - 1) * 100
    worst = max(cells, key=cells.get)
    guard = cells[worst] <= 25.0

    verdict = {
        "delta_rope": DELTA,
        "macro_kl": {v: round(macro(v), 6) for v in
                     ("iq3m", "o25", "b25", "o50", "b50", "o75", "b75", "iq4xs",
                      "r1-50", "r2-50", "r3-50")},
        "random_mean_50": round(mean50, 6),
        "G_util": {
            "margin_over_mean": round((mean50 - o50) / o50, 5),
            "seeds_beaten": beaten,
            "g_util_a": g_util_a, "g_util_b": g_util_b, "g_util_c": g_util_c,
            "pass": g_util,
        },
        "G_bal": {
            "b_vs_o": round((b50 - o50) / o50, 5),
            "pass": g_bal,
        },
        "guardrail": {"worst_cell": worst, "percent": round(cells[worst], 2), "pass": guard},
        "transfer_verdict": {
            "size_control": "PASS (zero-byte errors at quantization stage)" ,
            "utility_transfers": g_util,
            "v01b_balancing_transfers": g_bal,
            "overall": (
                "FULL TRANSFER" if (g_util and g_bal) else
                "PARTIAL TRANSFER (utility only)" if g_util else
                "NO TRANSFER"
            ),
        },
    }
    print(json.dumps(verdict, indent=2, ensure_ascii=False))
    out = Path(sys.argv[1]).parent / "gate-verdict.json"
    out.write_text(json.dumps(verdict, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out}")

if __name__ == "__main__":
    main()
