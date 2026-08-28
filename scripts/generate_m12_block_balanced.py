#!/usr/bin/env python3
"""Generate block-balanced (frozen v0.1b) FIT-25/FIT-75 plans.

The allocator is the frozen M10 v0.1b rule: equal byte quotas across 16-block
ranges, unchanged within-range imatrix ranking, global fill for fragments.
No quarter-weight or coefficient changes are allowed here (D-0017).

The driver is validated against the known M10 ground truth first: reproducing
the block-balanced FIT-50 plan must yield the exact retained recipe bytes.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fit_gguf import (
    load_imatrix_profile,
    optimize_block_balanced,
    parse_dry_run,
    predict_quantized_size,
    read_gguf_layout,
    write_fit_recipe,
    write_tensor_type_file,
)
from fit_gguf.candidates import generate_upgrade_candidates
from fit_gguf.gguf import ImatrixProvenance, QuantizationMetadata

REPO = Path("/run/media/s117/OS/FIT-GGUF")
M2 = REPO / "experiments/2026-08-28-m2-effective-recipes/artifacts/logs"
M10 = REPO / "experiments/2026-08-28-m10-ablation"
M12 = REPO / "experiments/2026-08-28-m12-block-balanced-curve"
BF16 = REPO / "artifacts/source/Huihui-Qwen3.8-27B-abliterated-BF16.gguf"
IMATRIX = REPO / "imatrix_unsloth.gguf"

LOWER_BYTES = 12_580_875_232  # actual IQ3_M artifact size (M9)
UPPER_BYTES = 15_082_507_232  # actual IQ4_XS artifact size (M9)
FIT50_ACTUAL = 13_828_987_872  # retained v0.1b FIT-50 artifact (M10)

M12.mkdir(parents=True, exist_ok=True)


def build_candidate_set() -> "tuple[object, object, object]":
    lower_recipe = parse_dry_run((M2 / "iq3_m-dry-run.log").read_text(encoding="utf-8"))
    upper_recipe = parse_dry_run((M2 / "iq4_xs-dry-run.log").read_text(encoding="utf-8"))
    layout = read_gguf_layout(BF16)

    # Metadata must byte-match the real llama-quantize output: file_type 27
    # (IQ3_M family) and the relative --imatrix path used in every M2-M10 run,
    # read back from the retained block-balanced artifact.
    meta = QuantizationMetadata(
        file_type=27,
        imatrix=ImatrixProvenance(
            file="imatrix_unsloth.gguf",
            dataset="unsloth_calibration_dataset",
            entries_count=496,
            chunks_count=1251,
        ),
    )
    lower_size = predict_quantized_size(layout, lower_recipe, meta)
    upper_size = predict_quantized_size(layout, upper_recipe, meta)
    assert lower_size.total_bytes == LOWER_BYTES, f"IQ3_M prediction {lower_size.total_bytes}"
    assert upper_size.total_bytes == UPPER_BYTES, f"IQ4_XS prediction {upper_size.total_bytes}"

    profile = load_imatrix_profile(IMATRIX)
    candidate_set = generate_upgrade_candidates(
        lower_recipe, upper_recipe, lower_size, upper_size, profile
    )
    return candidate_set, layout, meta


def emit(plan, name: str, layout, meta) -> None:
    recipe_path = M12 / f"block-balanced-{name}-recipe.json"
    types_path = M12 / f"block-balanced-{name}-tensor-types.txt"
    write_fit_recipe(plan, recipe_path, lower_preset="IQ3_M", upper_preset="IQ4_XS")
    write_tensor_type_file(plan, types_path)
    prediction = predict_quantized_size(
        layout, plan_to_recipe(plan, name), meta
    )
    print(
        f"{name}: selected={len(plan.selected)} target={plan.target_bytes:,} "
        f"predicted={prediction.total_bytes:,} unused={plan.unused_bytes:,} "
        f"prediction_delta_vs_target={prediction.total_bytes - plan.target_bytes:,}"
    )


def plan_to_recipe(plan, name: str):
    """Rebuild a DryRunResult-shaped recipe: lower assignments with overrides applied."""
    lower_recipe = parse_dry_run((M2 / "iq3_m-dry-run.log").read_text(encoding="utf-8"))
    from fit_gguf.models import DryRunResult, DryRunTensorAssignment

    overrides = {candidate.tensor: candidate.to_qtype for candidate in plan.selected}
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


def main() -> None:
    candidate_set, layout, meta = build_candidate_set()

    # Driver validation: reproduce the frozen M10 v0.1b FIT-50 plan exactly.
    fit50_target = LOWER_BYTES + (UPPER_BYTES - LOWER_BYTES) // 2
    plan50 = optimize_block_balanced(fit50_target, candidate_set)
    plan50_prediction = predict_quantized_size(layout, plan_to_recipe(plan50, "fit50"), meta)
    retained50 = (M10 / "block-balanced-fit50-recipe.json").read_text(encoding="utf-8")
    assert plan50_prediction.total_bytes == FIT50_ACTUAL, (
        f"FIT-50 prediction {plan50_prediction.total_bytes:,} != retained artifact {FIT50_ACTUAL:,}"
    )
    print(
        f"driver check fit50: predicted={plan50_prediction.total_bytes:,} "
        f"selected={len(plan50.selected)} unused={plan50.unused_bytes:,}"
    )
    assert plan50.target_bytes == fit50_target

    # M12 extensions at the preregistered M9 curve positions.
    fit25_target = LOWER_BYTES + (UPPER_BYTES - LOWER_BYTES) // 4
    fit75_target = LOWER_BYTES + 3 * (UPPER_BYTES - LOWER_BYTES) // 4
    plan25 = optimize_block_balanced(fit25_target, candidate_set)
    plan75 = optimize_block_balanced(fit75_target, candidate_set)
    emit(plan25, "fit25", layout, meta)
    emit(plan75, "fit75", layout, meta)
    print("M12 plans generated")


if __name__ == "__main__":
    main()
