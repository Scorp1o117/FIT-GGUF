#!/usr/bin/env python3
"""Generate the M15 paired random plans: three seeds, each consumed greedily
at the FIT-25 target (prefix) and continued to the FIT-75 target, per the
protocol lock in experiments/2026-08-29-m15-random-baseline/README.md.

Asserts the paired-trajectory superset property (plan75 selected ⊇ plan25
selected for every seed) and records exact predicted sizes for the
quantization stage to verify.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path("/run/media/s117/OS/FIT-GGUF")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from fit_gguf import optimize_random, predict_quantized_size  # noqa: E402
from generate_m12_block_balanced import (  # noqa: E402
    build_candidate_set,
    plan_to_recipe,
    M2,
)
from fit_gguf import parse_dry_run, read_gguf_layout  # noqa: E402
from fit_gguf.gguf import ImatrixProvenance, QuantizationMetadata  # noqa: E402

M15 = REPO / "experiments/2026-08-29-m15-random-baseline"
M15.mkdir(parents=True, exist_ok=True)

LOWER_BYTES = 12_580_875_232
UPPER_BYTES = 15_082_507_232
TARGET_25 = LOWER_BYTES + (UPPER_BYTES - LOWER_BYTES) // 4   # 13,206,283,232
TARGET_75 = LOWER_BYTES + 3 * (UPPER_BYTES - LOWER_BYTES) // 4  # 14,457,099,232
SEEDS = ("m15-v1", "m15-v2", "m15-v3")

# Same frozen metadata as every imatrix-quantized artifact in this project.
META = QuantizationMetadata(
    file_type=27,
    imatrix=ImatrixProvenance(
        file="imatrix_unsloth.gguf",
        dataset="unsloth_calibration_dataset",
        entries_count=496,
        chunks_count=1251,
    ),
)


def main() -> None:
    candidate_set, layout, _ = build_candidate_set()
    plans: dict[str, dict[str, int]] = {}
    for seed in SEEDS:
        plan25 = optimize_random(TARGET_25, candidate_set, seed=seed)
        plan75 = optimize_random(TARGET_75, candidate_set, seed=seed)
        sel25 = {c.tensor for c in plan25.selected}
        sel75 = {c.tensor for c in plan75.selected}
        assert sel25 <= sel75, f"{seed}: paired-trajectory prefix property violated"
        for budget, plan, target in (("fit25", plan25, TARGET_25), ("fit75", plan75, TARGET_75)):
            prediction = predict_quantized_size(layout, plan_to_recipe(plan, budget), META).total_bytes
            assert prediction <= target, f"{seed}/{budget}: prediction {prediction:,} exceeds target {target:,}"
            from fit_gguf import write_fit_recipe, write_tensor_type_file
            write_fit_recipe(plan, M15 / f"{seed}-{budget}-recipe.json",
                             lower_preset="IQ3_M", upper_preset="IQ4_XS")
            write_tensor_type_file(plan, M15 / f"{seed}-{budget}-tensor-types.txt")
            plans[f"{seed}-{budget}"] = prediction
            print(f"{seed}/{budget}: selected={len(plan.selected)} predicted={prediction:,} "
                  f"target={target:,} unused={plan.unused_bytes:,}")
    (M15 / "plan-predictions.json").write_text(
        json.dumps({"schema_version": 1, "targets": {"fit25": TARGET_25, "fit75": TARGET_75},
                    "predicted": plans}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("M15 plans generated")


if __name__ == "__main__":
    main()
