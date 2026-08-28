"""Conservative lower-to-upper tensor upgrade candidate generation."""

from dataclasses import dataclass

from fit_gguf.gguf import GGML_TYPE_TRAITS, GGUFSizePrediction
from fit_gguf.imatrix import ImatrixProfile
from fit_gguf.models import DryRunResult


class CandidateGenerationError(ValueError):
    """Raised when lower/upper recipes cannot produce safe candidates."""


@dataclass(frozen=True, slots=True)
class UpgradeCandidate:
    tensor: str
    from_qtype: str
    to_qtype: str
    delta_bytes: int
    importance: float
    raw_importance: float
    expected_gain: float
    utility_per_byte: float
    profiled: bool
    block: int | None
    role: str


@dataclass(frozen=True, slots=True)
class RejectedTransition:
    tensor: str
    from_qtype: str
    to_qtype: str
    delta_bytes: int
    reason: str


@dataclass(frozen=True, slots=True)
class CandidateSet:
    candidates: tuple[UpgradeCandidate, ...]
    rejected: tuple[RejectedTransition, ...]
    lower_size_bytes: int
    upper_size_bytes: int

    @property
    def candidate_budget_bytes(self) -> int:
        return sum(candidate.delta_bytes for candidate in self.candidates)

    @property
    def net_preset_gap_bytes(self) -> int:
        return self.upper_size_bytes - self.lower_size_bytes


def _bpw(qtype: str) -> float:
    try:
        block_size, type_size = GGML_TYPE_TRAITS[qtype.lower()]
    except KeyError as exc:
        raise CandidateGenerationError(f"Unsupported qtype in candidate transition: {qtype}") from exc
    return type_size * 8.0 / block_size


def generate_upgrade_candidates(
    lower_recipe: DryRunResult,
    upper_recipe: DryRunResult,
    lower_size: GGUFSizePrediction,
    upper_size: GGUFSizePrediction,
    profile: ImatrixProfile,
) -> CandidateSet:
    """Generate only positive-size lower-to-upper transitions.

    `importance` is the within-role median ratio from M4. `expected_gain` is an
    explicitly provisional proxy: importance multiplied by encoded BPW gain.
    It is a deterministic search feature, not an empirical quality estimate.
    """
    lower_assignments = lower_recipe.tensor_map
    upper_assignments = upper_recipe.tensor_map
    if lower_assignments.keys() != upper_assignments.keys():
        raise CandidateGenerationError("Lower and upper recipe tensor sets differ")
    lower_sizes = {tensor.name: tensor for tensor in lower_size.tensors}
    upper_sizes = {tensor.name: tensor for tensor in upper_size.tensors}
    if lower_sizes.keys() != lower_assignments.keys() or upper_sizes.keys() != upper_assignments.keys():
        raise CandidateGenerationError("Recipe and size-prediction tensor sets differ")

    profiles = profile.entry_map
    candidates: list[UpgradeCandidate] = []
    rejected: list[RejectedTransition] = []
    for tensor in lower_recipe.tensors:
        upper = upper_assignments[tensor.name]
        from_qtype = tensor.dst_type.lower()
        to_qtype = upper.dst_type.lower()
        if from_qtype == to_qtype:
            continue
        delta_bytes = (
            upper_sizes[tensor.name].padded_bytes - lower_sizes[tensor.name].padded_bytes
        )
        bpw_gain = _bpw(to_qtype) - _bpw(from_qtype)
        if delta_bytes <= 0 or bpw_gain <= 0:
            rejected.append(
                RejectedTransition(
                    tensor=tensor.name,
                    from_qtype=from_qtype,
                    to_qtype=to_qtype,
                    delta_bytes=delta_bytes,
                    reason="not_a_strict_encoded_precision_promotion",
                )
            )
            continue

        tensor_profile = profiles.get(tensor.name)
        profiled = tensor_profile is not None
        importance = tensor_profile.role_relative_mean if tensor_profile else 0.0
        raw_importance = tensor_profile.mean if tensor_profile else 0.0
        expected_gain = importance * bpw_gain
        candidates.append(
            UpgradeCandidate(
                tensor=tensor.name,
                from_qtype=from_qtype,
                to_qtype=to_qtype,
                delta_bytes=delta_bytes,
                importance=importance,
                raw_importance=raw_importance,
                expected_gain=expected_gain,
                utility_per_byte=expected_gain / delta_bytes,
                profiled=profiled,
                block=tensor_profile.block if tensor_profile else None,
                role=tensor_profile.role if tensor_profile else "unprofiled",
            )
        )

    return CandidateSet(
        candidates=tuple(sorted(candidates, key=lambda candidate: candidate.tensor)),
        rejected=tuple(sorted(rejected, key=lambda transition: transition.tensor)),
        lower_size_bytes=lower_size.total_bytes,
        upper_size_bytes=upper_size.total_bytes,
    )
