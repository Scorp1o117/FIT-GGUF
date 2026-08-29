#!/usr/bin/env python3
"""Generate all M16 Granite plans: presets validation, O/B at 25/50/75,
three random seeds at FIT-50. Freezes exact predicted sizes for the
quantization stage.

Granite adaptations per the preregistration: 40 blocks -> block_span=10 for
the block-balanced rule; imatrix profile from imatrix-granite-apex-c512.gguf;
dry-runs parsed from the granite dry-run logs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path("/run/media/s117/OS/FIT-GGUF")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from fit_gguf import (  # noqa: E402
    load_imatrix_profile,
    optimize_block_balanced,
    optimize_greedy,
    optimize_random,
    parse_dry_run,
    predict_quantized_size,
    read_gguf_layout,
    write_fit_recipe,
    write_tensor_type_file,
)
from fit_gguf.candidates import generate_upgrade_candidates  # noqa: E402
from fit_gguf.gguf import ImatrixProvenance, QuantizationMetadata  # noqa: E402

M16 = REPO / "experiments/2026-08-29-m16-granite-reveal"
LOGS = M16 / "artifacts/logs"
BF16 = REPO / "artifacts/source/granite-4.2-8b-BF16.gguf"
IMATRIX = REPO / "imatrix-granite-apex-c512.gguf"
BLOCK_SPAN = 10  # 40 blocks / 4 quarters

META = QuantizationMetadata(
    file_type=27,
    imatrix=ImatrixProvenance(
        file="imatrix-granite-apex-c512.gguf",
        dataset="/run/media/s117/OS/Models/imatrix-calibration/APEX-imatrix-Small.txt",
        entries_count=280,
        chunks_count=1250,
    ),
)

SEEDS = ("m16-v1", "m16-v2", "m16-v3")


def main() -> None:
    lower_recipe = parse_dry_run((LOGS / "granite-IQ3_M-dry-run.log").read_text(encoding="utf-8"))
    upper_recipe = parse_dry_run((LOGS / "granite-IQ4_XS-dry-run.log").read_text(encoding="utf-8"))
    layout = read_gguf_layout(BF16)
    lower_size = predict_quantized_size(layout, lower_recipe, META).total_bytes
    upper_size = predict_quantized_size(layout, upper_recipe, META).total_bytes
    print(f"IQ3_M predicted {lower_size:,}; IQ4_XS predicted {upper_size:,}")

    profile = load_imatrix_profile(IMATRIX)
    candidate_set = generate_upgrade_candidates(
        lower_recipe, upper_recipe,
        predict_quantized_size(layout, lower_recipe, META),
        predict_quantized_size(layout, upper_recipe, META),
        profile,
    )
    gap = upper_size - lower_size
    targets = {
        "fit25": lower_size + gap // 4,
        "fit50": lower_size + gap // 2,
        "fit75": lower_size + 3 * gap // 4,
    }

    outputs = {}
    for budget, target in targets.items():
        o = optimize_greedy(target, candidate_set)
        outputs[f"o-{budget}"] = (o, target)
        b = optimize_block_balanced(target, candidate_set, block_span=BLOCK_SPAN)
        outputs[f"b-{budget}"] = (b, target)
    for i, seed in enumerate(SEEDS, 1):
        outputs[f"r{i}-fit50"] = (optimize_random(targets["fit50"], candidate_set, seed=seed), targets["fit50"])

    from fit_gguf.models import DryRunResult, DryRunTensorAssignment

    def granite_plan_to_recipe(plan):
        """Rebuild the lower-preset recipe with this plan's overrides applied
        (granite baseline, NOT the hardcoded Huihui one in generate_m12)."""
        overrides = {c.tensor: c.to_qtype for c in plan.selected}
        tensors = tuple(
            DryRunTensorAssignment(
                ordinal=t.ordinal,
                total_tensors=t.total_tensors,
                name=t.name,
                shape=t.shape,
                src_type=t.src_type,
                dst_type=overrides.get(t.name, t.dst_type),
                is_quantized=t.is_quantized,
                orig_bytes=t.orig_bytes,
                new_bytes=t.new_bytes,
            )
            for t in lower_recipe.tensors
        )
        return DryRunResult(
            tensors=tensors,
            total_tensors=lower_recipe.total_tensors,
            reported_orig_bytes=lower_recipe.reported_orig_bytes,
            reported_new_bytes=lower_recipe.reported_new_bytes,
        )

    predictions = {}
    for name, (plan, target) in outputs.items():
        recipe = granite_plan_to_recipe(plan)
        prediction = predict_quantized_size(layout, recipe, META).total_bytes
        assert prediction <= target, f"{name}: prediction {prediction:,} exceeds target {target:,}"
        predictions[name] = prediction
        write_fit_recipe(plan, M16 / f"{name}-recipe.json", lower_preset="IQ3_M", upper_preset="IQ4_XS")
        write_tensor_type_file(plan, M16 / f"{name}-tensor-types.txt")
        print(f"{name}: selected={len(plan.selected)} predicted={prediction:,} target={target:,} unused={plan.unused_bytes:,}")

    predictions["iq3_m_preset"] = lower_size
    predictions["iq4_xs_preset"] = upper_size
    (M16 / "plan-predictions.json").write_text(
        json.dumps({"schema_version": 1, "targets": targets, "predicted": predictions},
                   indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("M16 plans generated")


if __name__ == "__main__":
    main()
