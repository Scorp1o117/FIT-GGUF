#!/usr/bin/env python3
"""Generate the M14 crossover artifacts' recipes and tensor-type files.

Implements the deterministic construction frozen in
experiments/2026-08-28-m14-swap-ablation/README.md: per (role, transition)
class, exchange the maximum byte-exact subset between the skeleton's
opposite-half upgrades and the donor plan's upgrades; the SHUF-50 control
reuses O->E's per-class totals with SHA-256-ordered selection.

Every generated plan must predict to exactly its skeleton's size.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

REPO = Path("/run/media/s117/OS/FIT-GGUF")
M7 = REPO / "experiments/2026-08-28-m7-greedy"
M10 = REPO / "experiments/2026-08-28-m10-ablation"
M12 = REPO / "experiments/2026-08-28-m12-block-balanced-curve"
M14 = REPO / "experiments/2026-08-28-m14-swap-ablation"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from fit_gguf import parse_dry_run, predict_quantized_size, read_gguf_layout  # noqa: E402
from fit_gguf.gguf import ImatrixProvenance, QuantizationMetadata  # noqa: E402
from generate_m12_block_balanced import M2  # noqa: E402
from generate_m12_block_balanced import BF16, IMATRIX  # noqa: E402

EARLY_MAX_BLOCK = 31
SHUFFLE_SEED = "m14-shuffle-50"

PLANS = {
    ("orig", "fit50"): M7 / "fit-recipe-FIT50.json",
    ("orig", "fit75"): M7 / "fit-recipe-FIT75.json",
    ("v01b", "fit50"): M10 / "block-balanced-fit50-recipe.json",
    ("v01b", "fit75"): M12 / "block-balanced-fit75-recipe.json",
}
SKELETON_PREDICTED = {  # retained ground-truth predicted sizes
    ("orig", "fit50"): 13_831_486_432,
    ("v01b", "fit50"): 13_828_987_872,
    ("orig", "fit75"): 14_456_126_432,
    ("v01b", "fit75"): 14_454_856_672,
}

META = QuantizationMetadata(
    file_type=27,
    imatrix=ImatrixProvenance(
        file="imatrix_unsloth.gguf",
        dataset="unsloth_calibration_dataset",
        entries_count=496,
        chunks_count=1251,
    ),
)


def load_upgrades(path: Path) -> dict[str, dict]:
    recipe = json.loads(path.read_text(encoding="utf-8"))
    return {o["tensor"]: o for o in recipe["overrides"]}


def order_key(name: str, use_hash: bool) -> str:
    if not use_hash:
        return name
    return hashlib.sha256(f"{SHUFFLE_SEED}:{name}".encode()).hexdigest()


def max_exchange(
    removals: list[dict], additions: list[dict], use_hash: bool
) -> tuple[set[str], set[str], int]:
    """Byte-exact subset pair with maximum total; deterministic reconstruction.

    Selection order (tensor name, or SHA-256 for the shuffle control) is fixed
    BEFORE the DP so the reconstruction masks match the greedy walk.
    """
    if not removals or not additions:
        return set(), set(), 0
    g = math.gcd(*(x["delta_bytes"] for x in removals + additions))

    def solve_side(items: list[dict]) -> tuple[set[str], list[int]]:
        ordered = sorted(items, key=lambda x: order_key(x["tensor"], use_hash))
        values = [x["delta_bytes"] // g for x in ordered]
        masks = [1]
        r = 1
        for v in values:
            r |= r << v
            masks.append(r)
        return ordered, masks

    ordered_r, masks_r = solve_side(removals)
    ordered_a, masks_a = solve_side(additions)
    common = masks_r[-1] & masks_a[-1]
    s = common.bit_length() - 1  # highest achievable common sum (scaled)
    if s == 0:
        return set(), set(), 0

    def take_subset(ordered: list[dict], masks: list[int], target: int) -> set[str]:
        values = [x["delta_bytes"] // g for x in ordered]
        chosen: set[str] = set()
        for k in range(len(ordered) - 1, -1, -1):
            if (masks[k] >> target) & 1:
                continue  # achievable without item k
            chosen.add(ordered[k]["tensor"])
            target -= values[k]
        assert target == 0
        return chosen

    take_r = take_subset(ordered_r, masks_r, s)
    take_a = take_subset(ordered_a, masks_a, s)
    return take_r, take_a, s * g


def build_crossover(
    budget: str, skeleton_label: str, donor_label: str, direction: str, use_hash: bool
) -> tuple[dict[str, str], dict[str, object]]:
    """Return (final tensor->qtype map, stats). direction: 'OE' or 'BL'."""
    skel = load_upgrades(PLANS[(skeleton_label, budget)])
    donor = load_upgrades(PLANS[(donor_label, budget)])

    classes: dict[tuple[str, str], dict[str, list[dict]]] = {}
    for t, o in skel.items():
        late = o["block"] is not None and o["block"] > EARLY_MAX_BLOCK
        early = o["block"] is not None and o["block"] <= EARLY_MAX_BLOCK
        if direction == "OE" and late:
            classes.setdefault((o["role"], o["from_qtype"] + "->" + o["to_qtype"]), {}).setdefault("removals", []).append(o)
        if direction == "BL" and early:
            classes.setdefault((o["role"], o["from_qtype"] + "->" + o["to_qtype"]), {}).setdefault("removals", []).append(o)
    for t, o in donor.items():
        if t in skel:
            continue
        early = o["block"] is not None and o["block"] <= EARLY_MAX_BLOCK
        late = o["block"] is not None and o["block"] > EARLY_MAX_BLOCK
        if direction == "OE" and early:
            classes.setdefault((o["role"], o["from_qtype"] + "->" + o["to_qtype"]), {}).setdefault("additions", []).append(o)
        if direction == "BL" and late:
            classes.setdefault((o["role"], o["from_qtype"] + "->" + o["to_qtype"]), {}).setdefault("additions", []).append(o)

    final = {t: o["to_qtype"] for t, o in skel.items()}
    removed_bytes = added_bytes = 0
    exchanged_classes = 0
    per_class: dict[str, int] = {}
    for key, pools in sorted(classes.items()):
        removals = pools.get("removals", [])
        additions = pools.get("additions", [])
        take_r, take_a, s = max_exchange(removals, additions, use_hash)
        if s == 0:
            continue
        for name in take_r:
            del final[name]
        for name in take_a:
            final[name] = donor[name]["to_qtype"]
        removed_bytes += sum(x["delta_bytes"] for x in removals if x["tensor"] in take_r)
        added_bytes += sum(x["delta_bytes"] for x in additions if x["tensor"] in take_a)
        per_class[f"{key[0]}/{key[1]}"] = s
        exchanged_classes += 1
        assert removed_bytes >= added_bytes - s and s >= 0
    stats = {
        "exchanged_classes": exchanged_classes,
        "removed_bytes": removed_bytes,
        "added_bytes": added_bytes,
        "per_class": per_class,
    }
    assert removed_bytes == added_bytes
    return final, stats


def predict(final: dict[str, str], skeleton_predicted: int) -> int:
    lower_recipe = parse_dry_run((M2 / "iq3_m-dry-run.log").read_text(encoding="utf-8"))
    from fit_gguf.models import DryRunResult, DryRunTensorAssignment

    tensors = tuple(
        DryRunTensorAssignment(
            ordinal=t.ordinal,
            total_tensors=t.total_tensors,
            name=t.name,
            shape=t.shape,
            src_type=t.src_type,
            dst_type=final.get(t.name, t.dst_type),
            is_quantized=t.is_quantized,
            orig_bytes=t.orig_bytes,
            new_bytes=t.new_bytes,
        )
        for t in lower_recipe.tensors
    )
    recipe = DryRunResult(
        tensors=tensors,
        total_tensors=lower_recipe.total_tensors,
        reported_orig_bytes=lower_recipe.reported_orig_bytes,
        reported_new_bytes=lower_recipe.reported_new_bytes,
    )
    layout = read_gguf_layout(BF16)
    prediction = predict_quantized_size(layout, recipe, META).total_bytes
    assert prediction == skeleton_predicted, f"prediction {prediction:,} != skeleton {skeleton_predicted:,}"
    return prediction


def write_outputs(name: str, final: dict[str, str], skeleton: dict[str, dict], stats: dict[str, object]) -> None:
    M14.mkdir(parents=True, exist_ok=True)
    overrides = []
    for t, q in final.items():
        if t in skeleton and skeleton[t]["to_qtype"] == q:
            continue  # unchanged from skeleton
        base = skeleton.get(t) or {}
        overrides.append({
            "tensor": t,
            "from_qtype": base.get("from_qtype", "iq3_m"),
            "to_qtype": q,
            "block": base.get("block"),
            "delta_bytes": base.get("delta_bytes"),
            "role": base.get("role"),
            "origin": "swap",
        })
    recipe = {
        "schema_version": 1,
        "kind": "m14-crossover",
        "name": name,
        "stats": stats,
        "overrides": overrides,
    }
    (M14 / f"{name}-recipe.json").write_text(json.dumps(recipe, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [f"^{t}$={q}" for t, q in sorted(final.items())]
    (M14 / f"{name}-tensor-types.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{name}: {len(final)} assignments, {stats['exchanged_classes']} classes, {stats['removed_bytes']:,} bytes exchanged")


def main() -> None:
    jobs = [
        ("m14-oe-fit50", "fit50", "orig", "v01b", "OE", False, ("orig", "fit50")),
        ("m14-bl-fit50", "fit50", "v01b", "orig", "BL", False, ("v01b", "fit50")),
        ("m14-shuf-fit50", "fit50", "orig", "v01b", "OE", True, ("orig", "fit50")),
        ("m14-oe-fit75", "fit75", "orig", "v01b", "OE", False, ("orig", "fit75")),
        ("m14-bl-fit75", "fit75", "v01b", "orig", "BL", False, ("v01b", "fit75")),
    ]
    for name, budget, skel_label, donor_label, direction, use_hash, skel_key in jobs:
        final, stats = build_crossover(budget, skel_label, donor_label, direction, use_hash)
        predict(final, SKELETON_PREDICTED[skel_key])
        skeleton = load_upgrades(PLANS[skel_key])
        write_outputs(name, final, skeleton, stats)
    print("M14 recipes generated")


if __name__ == "__main__":
    main()
