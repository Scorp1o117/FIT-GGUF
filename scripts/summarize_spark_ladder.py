#!/usr/bin/env python3
"""Summarize the Spark-X2.5-4B-abliterated preset ladder and derive
per-tier Same-top floors (P5 over each tier's KL window), emitting a
Guard Profile YAML draft (exact_model scope, source-SHA bound).

Windows follow the M2 calibration convention: a tier window is the set of
observed points with macro KL <= anchor + 0.005 (and above the previous
anchor), i.e. the near-boundary population whose Same-top behavior the
floor must guard. TopFloor~P5 principle (GPT, M2 ruling).
"""

import json
import sys
from pathlib import Path

REPO = Path("/run/media/s117/OS/FIT-GGUF")
sys.path.insert(0, str(REPO / "src"))

from fit_gguf.eval.results import parse_llama_kl_log  # noqa: E402
from fit_gguf.eval.contract import DOMAINS  # noqa: E402
from fit_gguf.fidelity import KL_ANCHORS, profile_hash  # noqa: E402

EXP = REPO / "experiments" / "2026-09-06-spark-x25-4tier"
MP = "spark-x25-4b-abliterated-"
SOURCE_SHA_FILE = EXP / "state" / "bf16-sha.txt"

TIER_BANDS = {  # (low_exclusive, high_inclusive)
    "quality": (0.0, 0.055),
    "balanced": (0.055, 0.105),
    "compact": (0.105, 0.155),
    "mini": (0.155, 0.205),
}
POISON = ("Q3_K_S", "IQ2_XS")


def percentile(values: list[float], p: float) -> float:
    """Linear-interpolated percentile (numpy default method)."""
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * p
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def main() -> int:
    logs = EXP / "logs"
    manifest = EXP / "state" / "artifact-manifest.txt"
    if not manifest.is_file():
        manifest = EXP / "logs" / "artifact-manifest.txt"

    sizes: dict[str, int] = {}
    for line in manifest.read_text().splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[0].startswith(MP):
            sizes[parts[0][len(MP):]] = int(parts[1])

    by_point: dict[str, dict[str, dict]] = {}
    for entry in sorted(logs.iterdir()):
        name = entry.name
        if not (name.startswith("eval-") and name.endswith(".log")):
            continue
        body = name[len("eval-"):-len(".log")]
        if not body.startswith(MP):
            continue
        dom = None
        for d in DOMAINS:
            if body.endswith("-" + d):
                dom = d
                break
        if dom is None:
            continue
        point = body[: -len(dom) - 1][len(MP):]
        try:
            metrics = parse_llama_kl_log(entry.read_text(errors="replace"))
        except Exception:
            continue
        by_point.setdefault(point, {})[dom] = metrics

    rows = []
    for point, domains in sorted(by_point.items(), key=lambda kv: -(sizes.get(kv[0], 0))):
        if len(domains) != len(DOMAINS) or point not in sizes:
            continue
        kl = sum(domains[d]["mean_kld"] for d in DOMAINS) / len(DOMAINS)
        top = sum(domains[d]["same_top_pct"] for d in DOMAINS) / len(DOMAINS) / 100.0
        rows.append((point, sizes[point], kl, top))

    print(f"{'point':<28} {'size':>12} {'GiB':>7} {'macro KL':>9} {'same-top':>9}")
    for point, size, kl, top in rows:
        print(f"{point:<28} {size:>12,} {size / 2**30:>7.3f} {kl:>9.4f} {top*100:>8.2f}%")

    floors = {}
    print("\nper-tier windows (floor = window minimum; P5 degenerates at n<=2):")
    for tier, (lo, hi) in TIER_BANDS.items():
        pts = [r for r in rows if lo < r[2] <= hi and not any(p in r[0] for p in POISON)]
        if not pts:
            print(f"  {tier}: EMPTY window [{lo}, {hi}] — need ladder points!")
            floors[tier] = None
            continue
        tops = [r[3] for r in pts]
        # TopFloor~P5 principle (M2 ruling). Truncate DOWN so the floor never
        # rounds above the value it is derived from (run 1 bug: Q6_K gated out
        # by its own rounded floor). n<=2: P5 is a "minimum in a suit" — use
        # the window minimum instead.
        import math

        def _trunc4(v: float) -> float:
            return math.floor(v * 10000) / 10000

        floor = _trunc4(percentile(tops, 0.05)) if len(tops) >= 3 else _trunc4(min(tops))
        floors[tier] = floor
        detail = ", ".join(f"{r[0].split('-')[-1]} {r[2]:.4f}/{r[3]*100:.2f}" for r in pts)
        print(f"  {tier}: n={len(pts)} floor={floor:.4f} (P5={percentile(tops, 0.05):.4f})   [{detail}]")

    out = EXP / "results" / "spark-ladder-summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "points": [{"point": p, "size_bytes": s, "macro_kl": k, "same_top": t} for p, s, k, t in rows],
        "floors": floors,
    }, indent=1) + "\n")
    print(f"\nwrote {out}")

    if all(v is not None for v in floors.values()) and SOURCE_SHA_FILE.is_file():
        source_sha = SOURCE_SHA_FILE.read_text().strip()
        profile = {
            "calibration_manifest_hashes": [
                {"path": "experiments/2026-09-06-spark-x25-4tier/results/spark-ladder-summary.json",
                 "sha256": __import__("hashlib").sha256(out.read_bytes()).hexdigest()},
            ],
            "calibration_models": ["spark-x25-4b-abliterated"],
            "confidence": {
                "known_caveats": [
                    "calibration-run floors derived from the 15-point preset ladder "
                    "plus 2 compact-gap calibration probes; floor = per-tier window "
                    "minimum (P5 degenerates at n<=2; TopFloor~P5 principle)",
                    "product-CLI provenance (freeze manifest prefix) is orcarouter-"
                    "pinned and does not cover this model",
                ],
                "note": "model-calibrated on the eval-v1 five-domain ladder; floors = per-tier window minimum (TopFloor~P5 principle)",
                "window_n": {t: len([r for r in rows if TIER_BANDS[t][0] < r[2] <= TIER_BANDS[t][1]
                                     and not any(p in r[0] for p in POISON)]) for t in TIER_BANDS},
            },
            "evaluator_contract": "eval-v1",
            "generalization_scope": "exact_model_only",
            "guard_profile_id": "guard-spark-x25-4b-abliterated-exact-v1",
            "guard_profile_version": 3,
            "scope": {"identifier": "spark-x25-4b-abliterated", "type": "exact_model"},
            "source_sha256": source_sha,
            "status": "validated",
            "tiers": {t: {"kl_anchor": KL_ANCHORS[t], "same_top_floor": floors[t]} for t in
                      ("quality", "balanced", "compact", "mini")},
            "validation_basis": "dev_calibration",
        }
        profile["profile_hash"] = profile_hash(profile)
        yaml_path = REPO / "profiles" / "guard" / "guard-spark-x25-4b-abliterated-exact-v1.yaml"
        try:
            import yaml
            yaml_path.write_text(yaml.safe_dump(profile, sort_keys=False, allow_unicode=True))
            print(f"wrote {yaml_path}")
        except ImportError:
            (yaml_path.with_suffix(".json")).write_text(json.dumps(profile, indent=1))
            print("PyYAML missing — wrote JSON instead")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
