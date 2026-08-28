#!/usr/bin/env python3
"""Free diagnostic (no quantization): weighted recipe overlap between the
original utility and v0.1b at FIT-25/50/75.

Reports, per budget:
- upgrade set sizes and upgrade bytes per plan;
- byte-weighted Jaccard of the upgrade sets;
- early/late (blocks 0-31 / 32-63) upgrade byte split per plan;
- per-(role, transition) late-of-original vs early-of-v0.1b byte availability,
  which bounds the exact-byte crossover exchange used by M14.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path("/run/media/s117/OS/FIT-GGUF")

M7 = REPO / "experiments/2026-08-28-m7-greedy"
M10 = REPO / "experiments/2026-08-28-m10-ablation"
M12 = REPO / "experiments/2026-08-28-m12-block-balanced-curve"
OUT = REPO / "experiments/2026-08-28-m14-swap-ablation/recipe-overlap.json"

PLANS = {
    ("orig", "fit25"): M7 / "fit-recipe-FIT25.json",
    ("orig", "fit50"): M7 / "fit-recipe-FIT50.json",
    ("orig", "fit75"): M7 / "fit-recipe-FIT75.json",
    ("v01b", "fit25"): M12 / "block-balanced-fit25-recipe.json",
    ("v01b", "fit50"): M10 / "block-balanced-fit50-recipe.json",
    ("v01b", "fit75"): M12 / "block-balanced-fit75-recipe.json",
}

EARLY_MAX_BLOCK = 31  # blocks 0-31 = early half, 32-63 = late half


def load_upgrades(path: Path) -> dict[str, dict]:
    recipe = json.loads(path.read_text(encoding="utf-8"))
    return {
        o["tensor"]: {
            "to_qtype": o["to_qtype"],
            "from_qtype": o["from_qtype"],
            "block": o["block"],
            "delta_bytes": o["delta_bytes"],
            "role": o["role"],
        }
        for o in recipe["overrides"]
    }


def main() -> None:
    report: dict[str, object] = {"schema_version": 1, "early_max_block": EARLY_MAX_BLOCK, "budgets": {}}
    for budget in ("fit25", "fit50", "fit75"):
        u_o = load_upgrades(PLANS[("orig", budget)])
        u_b = load_upgrades(PLANS[("v01b", budget)])
        inter = sum(u["delta_bytes"] for t, u in u_o.items() if t in u_b)
        union = sum(u["delta_bytes"] for u in u_o.values()) + sum(
            u["delta_bytes"] for t, u in u_b.items() if t not in u_o
        )
        split = {}
        for label, u in (("orig", u_o), ("v01b", u_b)):
            early = sum(x["delta_bytes"] for x in u.values() if x["block"] is not None and x["block"] <= EARLY_MAX_BLOCK)
            late = sum(x["delta_bytes"] for x in u.values() if x["block"] is not None and x["block"] > EARLY_MAX_BLOCK)
            split[label] = {"upgrades": len(u), "upgrade_bytes": early + late,
                            "early_bytes": early, "late_bytes": late}
        # (role, transition) class availability for the two crossover directions
        oe: dict[tuple[str, str], dict[str, int]] = {}
        bl: dict[tuple[str, str], dict[str, int]] = {}
        for label, u in (("orig", u_o), ("v01b", u_b)):
            for x in u.values():
                if x["block"] is None:
                    continue
                key = f"{x['role']}/{x['from_qtype']}->{x['to_qtype']}"
                early_side = x["block"] <= EARLY_MAX_BLOCK
                if label == "orig":
                    if not early_side:
                        oe.setdefault(key, {"orig_late": 0, "v01b_early": 0})["orig_late"] += x["delta_bytes"]
                    else:
                        bl.setdefault(key, {"v01b_late": 0, "orig_early": 0})["orig_early"] += x["delta_bytes"]
                else:
                    if early_side:
                        oe.setdefault(key, {"orig_late": 0, "v01b_early": 0})["v01b_early"] += x["delta_bytes"]
                    else:
                        bl.setdefault(key, {"v01b_late": 0, "orig_early": 0})["v01b_late"] += x["delta_bytes"]
        report["budgets"][budget] = {
            "split": split,
            "byte_jaccard": round(inter / union, 4) if union else 0.0,
            "intersection_bytes": inter,
            "union_bytes": union,
        }
        report.setdefault("exchange_availability", {})[budget] = {
            "OE_direction": dict(sorted(oe.items())),
            "BL_direction": dict(sorted(bl.items())),
        }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for b, v in report["budgets"].items():
        s = v["split"]
        print(
            f"{b}: jaccard={v['byte_jaccard']:.4f} | orig {s['orig']['early_bytes']:,}/{s['orig']['late_bytes']:,} "
            f"(E/L) | v01b {s['v01b']['early_bytes']:,}/{s['v01b']['late_bytes']:,} (E/L)"
        )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
