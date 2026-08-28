"""Deterministic budget-safe greedy FIT optimizer."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from fit_gguf.candidates import CandidateSet, UpgradeCandidate


class OptimizationError(ValueError):
    """Raised when a FIT optimization request is internally invalid."""


@dataclass(frozen=True, slots=True)
class OptimizationPlan:
    schema_version: int
    target_bytes: int
    lower_size_bytes: int
    predicted_size_bytes: int
    unused_bytes: int
    selected: tuple[UpgradeCandidate, ...]
    skipped_count: int

    @property
    def selected_cost_bytes(self) -> int:
        return self.predicted_size_bytes - self.lower_size_bytes

    @property
    def overrides(self) -> tuple[tuple[str, str], ...]:
        return tuple((candidate.tensor, candidate.to_qtype) for candidate in self.selected)


def optimize_greedy(target_bytes: int, candidate_set: CandidateSet) -> OptimizationPlan:
    """Select the highest-utility candidates that fit the exact byte budget."""
    lower_size = candidate_set.lower_size_bytes
    if target_bytes < lower_size:
        raise OptimizationError(
            f"Target {target_bytes} bytes is below lower baseline {lower_size} bytes"
        )
    names = [candidate.tensor for candidate in candidate_set.candidates]
    if len(names) != len(set(names)):
        raise OptimizationError("Candidate tensor names must be unique")
    if any(candidate.delta_bytes <= 0 for candidate in candidate_set.candidates):
        raise OptimizationError("All candidate costs must be positive")

    ordered = sorted(
        candidate_set.candidates,
        key=lambda candidate: (
            -candidate.utility_per_byte,
            -candidate.expected_gain,
            candidate.delta_bytes,
            candidate.tensor,
            candidate.to_qtype,
        ),
    )
    remaining = target_bytes - lower_size
    selected: list[UpgradeCandidate] = []
    for candidate in ordered:
        if candidate.delta_bytes <= remaining:
            selected.append(candidate)
            remaining -= candidate.delta_bytes

    predicted_size = target_bytes - remaining
    return OptimizationPlan(
        schema_version=1,
        target_bytes=target_bytes,
        lower_size_bytes=lower_size,
        predicted_size_bytes=predicted_size,
        unused_bytes=remaining,
        selected=tuple(selected),
        skipped_count=len(ordered) - len(selected),
    )


def optimize_random(
    target_bytes: int,
    candidate_set: CandidateSet,
    *,
    seed: str | int,
) -> OptimizationPlan:
    """Create a reproducible random-priority budget baseline.

    SHA-256 priorities avoid dependence on Python's randomized hash seed or the
    implementation details of ``random.shuffle``. Candidate utility is never
    consulted.
    """
    lower_size = candidate_set.lower_size_bytes
    if target_bytes < lower_size:
        raise OptimizationError(
            f"Target {target_bytes} bytes is below lower baseline {lower_size} bytes"
        )
    names = [candidate.tensor for candidate in candidate_set.candidates]
    if len(names) != len(set(names)):
        raise OptimizationError("Candidate tensor names must be unique")
    if any(candidate.delta_bytes <= 0 for candidate in candidate_set.candidates):
        raise OptimizationError("All candidate costs must be positive")

    seed_text = str(seed)
    ordered = sorted(
        candidate_set.candidates,
        key=lambda candidate: (
            hashlib.sha256(
                f"{seed_text}\0{candidate.tensor}\0{candidate.to_qtype}".encode("utf-8")
            ).digest(),
            candidate.tensor,
            candidate.to_qtype,
        ),
    )
    remaining = target_bytes - lower_size
    selected: list[UpgradeCandidate] = []
    for candidate in ordered:
        if candidate.delta_bytes <= remaining:
            selected.append(candidate)
            remaining -= candidate.delta_bytes

    return OptimizationPlan(
        schema_version=1,
        target_bytes=target_bytes,
        lower_size_bytes=lower_size,
        predicted_size_bytes=target_bytes - remaining,
        unused_bytes=remaining,
        selected=tuple(selected),
        skipped_count=len(ordered) - len(selected),
    )


def optimize_block_balanced(
    target_bytes: int,
    candidate_set: CandidateSet,
    *,
    block_span: int = 16,
) -> OptimizationPlan:
    """Balance byte quotas across block ranges, then rank by utility within each.

    This is an ablation strategy for diagnosing block-position bias. Unprofiled
    candidates and quota leftovers participate only in the final global fill.
    """
    if block_span <= 0:
        raise OptimizationError("block_span must be positive")
    lower_size = candidate_set.lower_size_bytes
    if target_bytes < lower_size:
        raise OptimizationError(
            f"Target {target_bytes} bytes is below lower baseline {lower_size} bytes"
        )
    names = [candidate.tensor for candidate in candidate_set.candidates]
    if len(names) != len(set(names)):
        raise OptimizationError("Candidate tensor names must be unique")
    if any(candidate.delta_bytes <= 0 for candidate in candidate_set.candidates):
        raise OptimizationError("All candidate costs must be positive")

    profiled_groups: dict[int, list[UpgradeCandidate]] = {}
    for candidate in candidate_set.candidates:
        if candidate.block is not None:
            profiled_groups.setdefault(candidate.block // block_span, []).append(candidate)
    if not profiled_groups:
        raise OptimizationError("Block-balanced optimization requires profiled block candidates")

    def utility_order(candidates: list[UpgradeCandidate]) -> list[UpgradeCandidate]:
        return sorted(
            candidates,
            key=lambda candidate: (
                -candidate.utility_per_byte,
                -candidate.expected_gain,
                candidate.delta_bytes,
                candidate.tensor,
                candidate.to_qtype,
            ),
        )

    budget = target_bytes - lower_size
    group_ids = sorted(profiled_groups)
    quota, quota_remainder = divmod(budget, len(group_ids))
    selected: list[UpgradeCandidate] = []
    selected_names: set[str] = set()
    for position, group_id in enumerate(group_ids):
        group_remaining = quota + (quota_remainder if position == len(group_ids) - 1 else 0)
        for candidate in utility_order(profiled_groups[group_id]):
            if candidate.delta_bytes <= group_remaining:
                selected.append(candidate)
                selected_names.add(candidate.tensor)
                group_remaining -= candidate.delta_bytes

    remaining = budget - sum(candidate.delta_bytes for candidate in selected)
    leftovers = [
        candidate
        for candidate in candidate_set.candidates
        if candidate.tensor not in selected_names
    ]
    for candidate in utility_order(leftovers):
        if candidate.delta_bytes <= remaining:
            selected.append(candidate)
            remaining -= candidate.delta_bytes

    return OptimizationPlan(
        schema_version=1,
        target_bytes=target_bytes,
        lower_size_bytes=lower_size,
        predicted_size_bytes=target_bytes - remaining,
        unused_bytes=remaining,
        selected=tuple(selected),
        skipped_count=len(candidate_set.candidates) - len(selected),
    )


def write_fit_recipe(
    plan: OptimizationPlan,
    path: str | Path,
    *,
    lower_preset: str,
    upper_preset: str | None,
) -> None:
    """Write a deterministic schema-v1 FIT recipe record."""
    payload = {
        "schema_version": plan.schema_version,
        "target_bytes": plan.target_bytes,
        "lower_preset": lower_preset,
        "upper_preset": upper_preset,
        "lower_size_bytes": plan.lower_size_bytes,
        "predicted_size_bytes": plan.predicted_size_bytes,
        "unused_bytes": plan.unused_bytes,
        "selected_cost_bytes": plan.selected_cost_bytes,
        "selected_count": len(plan.selected),
        "skipped_count": plan.skipped_count,
        "overrides": [
            {
                "tensor": candidate.tensor,
                "from_qtype": candidate.from_qtype,
                "to_qtype": candidate.to_qtype,
                "delta_bytes": candidate.delta_bytes,
                "importance": candidate.importance,
                "raw_importance": candidate.raw_importance,
                "expected_gain": candidate.expected_gain,
                "utility_per_byte": candidate.utility_per_byte,
                "profiled": candidate.profiled,
                "block": candidate.block,
                "role": candidate.role,
            }
            for candidate in plan.selected
        ],
    }
    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
