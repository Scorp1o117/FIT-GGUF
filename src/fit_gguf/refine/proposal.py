"""Transition proposal scoring for FIT Refine (v0.2.1 §13/§15).

Scores are **proposal scores only**: they rank candidate tensor/qtype
transitions for the exact-byte solver. They are not quality estimates —
PRISM's L2 law (non-additivity of near-lossless slots) means single-action
scores must never be read as predicted KL deltas.
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_C_ROLE = 1.0
_MIN_BYTES = 1
_MIN_IMPORTANCE = 1e-12

# Roles whose fitted C_role sank to this level or below show network-hostile
# upgrade behavior (e.g. attn_gate). Transitions on such roles are flagged
# caution — "this action is suspect", NOT "this role is unimportant".
_ROLE_HOSTILE_C = 0.85

# Cell-delta keys look like "ffn_down:0:q3_K->q2_K".
_BAND_SEPARATOR = ":"


@dataclass(frozen=True, slots=True)
class TransitionProposal:
    tensor: str
    role: str
    band: int | None
    src_qtype: str
    dst_qtype: str
    delta_bytes: int
    importance: float
    c_role: float
    transition_gain: float
    caution: bool
    reason: str | None
    score: float


def resolve_band_key(role: str, band: int | None, src_qtype: str, dst_qtype: str) -> str:
    suffix = f"{src_qtype}->{dst_qtype}"
    if band is not None:
        return f"{role}:{band}:{suffix}"
    return f"{role}:{suffix}"


class ProposalScorer:
    """Rank qtype transitions using a fitted refine profile.

    score = (importance * c_role * transition_gain) / delta_bytes

    ``transition_gain`` is 1 + (-measured_delta_pct / 100) when the profile
    carries a measured cell for the (role, band, src->dst) transition, so a
    measured-improving retype scores above an unmeasured one and a measured
    cliff scores below it. Blocked destinations and cliffs are flagged
    ``caution`` for the planner to reject or demote.
    """

    def __init__(self, profile: dict) -> None:
        self.profile = profile
        self.role_correction: dict[str, float] = profile.get("role_correction", {})
        self.transition_prior: dict[str, float] = profile.get("transition_prior", {})
        risks = profile.get("known_risks", {})
        self.blocked_dst: dict[str, str] = risks.get("blocked_dst_qtypes", {})
        self.cliffs: dict[str, float] = {
            key: value for key, value in risks.get("measured_cliffs", {}).items() if value > 0
        }

    def c_role(self, role: str) -> float:
        return float(self.role_correction.get(role, DEFAULT_C_ROLE))

    def _cell_delta(self, role: str, band: int | None, src: str, dst: str) -> tuple[float | None, str | None]:
        if band is not None:
            key = f"{role}:{band}:{src}->{dst}"
            if key in self.transition_prior:
                return float(self.transition_prior[key]), key
        key = f"{role}:{src}->{dst}"
        if key in self.transition_prior:
            return float(self.transition_prior[key]), key
        return None, None

    def score(
        self,
        tensor: str,
        role: str,
        src_qtype: str,
        dst_qtype: str,
        delta_bytes: int,
        importance: float,
        band: int | None = None,
    ) -> TransitionProposal:
        if delta_bytes <= 0:
            raise ValueError(f"{tensor}: delta_bytes must be positive, got {delta_bytes}")

        c_role = self.c_role(role)
        cell_delta, cell_key = self._cell_delta(role, band, src_qtype, dst_qtype)
        transition_gain = 1.0
        reason: str | None = None
        caution = False

        if dst_qtype in self.blocked_dst:
            caution = True
            reason = f"blocked dst qtype {dst_qtype}: {self.blocked_dst[dst_qtype]}"
        elif cell_delta is not None:
            transition_gain = 1.0 - cell_delta / 100.0
            if cell_key in self.cliffs:
                caution = True
                reason = f"measured cliff {cell_key}: +{self.cliffs[cell_key]:+.2f}% KL"
        elif c_role <= _ROLE_HOSTILE_C:
            caution = True
            reason = (
                f"role {role} is upgrade-hostile (c_role={c_role:.2f}); "
                "proposal calibration bootstrap flag, not an importance verdict"
            )

        score = (max(importance, _MIN_IMPORTANCE) * c_role * transition_gain) / max(delta_bytes, _MIN_BYTES)
        return TransitionProposal(
            tensor=tensor,
            role=role,
            band=band,
            src_qtype=src_qtype,
            dst_qtype=dst_qtype,
            delta_bytes=delta_bytes,
            importance=importance,
            c_role=c_role,
            transition_gain=transition_gain,
            caution=caution,
            reason=reason,
            score=score,
        )


__all__ = ["ProposalScorer", "TransitionProposal", "resolve_band_key"]
