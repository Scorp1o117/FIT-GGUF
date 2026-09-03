#!/usr/bin/env python3
"""Fit the first development Refine profiles from PRISM refine-dataset-v1.

Local development tool (not part of the public release). Emits
``fit.refine_profile.v1`` JSON files under profiles/ with full provenance
(dataset digest, fitter version). Profiles are DEV artifacts: G0
preregistration is required before any formal validation use.
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from fit_gguf.refine import fit_profile, load_refine_dataset, save_profile  # noqa: E402


DEFAULT_DATASET = Path("/run/media/s117/OS/Models/prism/experiments/refine-dataset-v1")
CREATED = "2026-09-02"
BAND_CREATED = "2026-09-03"

SCOPE = {
    "model_family": "qwen",
    "architecture": "qwen3.5-hybrid (16 attn + 48 linear-attn/SSM, 64 blk)",
    "evidence_models": ["Qwen3.8-27B", "Huihui-Qwen3.8-27B-abliterated"],
    "note": "same-architecture cross-weight evidence only; not validated on dense non-hybrid or MoE families",
}


def main() -> int:
    dataset = load_refine_dataset(DEFAULT_DATASET)
    print(f"dataset: {dataset.dataset_id} digest={dataset.digest[:16]}… "
          f"chains={len(dataset.chains)} probes={len(dataset.probes)} curve={len(dataset.curve_points)}")
    print(f"splits: {dataset.split_summary()}")

    profile = fit_profile(
        dataset, "qwen-hybrid-dev1", scope=SCOPE, created=CREATED, with_band_cells=False
    )
    out = save_profile(profile, REPO / "profiles" / "refine-profile-qwen-hybrid-dev1.json")
    print(f"profile -> {out}")

    band = fit_profile(dataset, "qwen-hybrid-band-v1", scope=SCOPE, created=BAND_CREATED)
    band_out = save_profile(band, REPO / "profiles" / "refine-profile-qwen-hybrid-band-v1.json")
    print(f"band profile -> {band_out}")

    print("\nC_role (ordinal v0):")
    for role, value in sorted(profile["role_correction"].items(), key=lambda kv: -kv[1]):
        evidence = profile["role_evidence"][role]
        up = evidence["upgrade_delta_pct"]
        down = evidence["downgrade_delta_pct"]
        print(f"  {role:12s} C={value:.3f}  upgrade={up!r:>8}  downgrade={down!r}")
    print(f"\ntransition_prior cells: {len(profile['transition_prior'])}")
    print(f"measured cliffs: {len(profile['known_risks']['measured_cliffs'])}")

    cells = band["band_correction"]["cells"]
    print(f"\nband utility cells ({len(cells)}):")
    for cell in cells:
        print(
            f"  {cell['role']:10s} blk[{cell['min_block']:>2},{cell['max_block']:>2}] "
            f"{cell['src_qtype']}->{cell['dst_qtype']}  delta={cell['delta_macro_kl_pct']:+7.2f}%  "
            f"C={cell['c']:.3f}  [{cell['evidence']}]"
        )
    prot = band["band_correction"]["protection_cells"]
    print(f"band protection cells (downgrade evidence, not applied): {len(prot)}")
    for item in band["band_correction"]["skipped_evidence"]:
        print(f"  skipped: {item}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
