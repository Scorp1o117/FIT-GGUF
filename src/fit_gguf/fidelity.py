"""Guard Profile artifacts for the Fidelity Contract v1 Same-top policy.

Contract v1 (planner-verdict-m3.md): the global KL Core is universal, but the
Same-top threshold is resolved by a validated Guard Profile scoped to an exact
model, a family, or an architecture. A model without a validated profile must
not emit official Fidelity tiers — it requires onboarding calibration.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
import hashlib
import json
from pathlib import Path

import yaml

TIERS = ("quality", "balanced", "compact", "mini")
KL_ANCHORS = {"quality": 0.05, "balanced": 0.10, "compact": 0.15, "mini": 0.20}
SCOPES = ("exact_model", "family", "architecture")
_STATUSES = ("candidate", "validated")

REFUSE_MESSAGE = (
    "Fidelity validation unavailable: No validated Same-top Guard Profile "
    "for this model/family. Options: --calibrate-fidelity --experimental-fidelity"
)


class GuardProfileError(ValueError):
    """Raised when a Guard Profile artifact is invalid or cannot be resolved."""


@dataclass(frozen=True, slots=True)
class GuardProfile:
    profile_id: str
    version: int
    scope_type: str
    scope_identifier: str
    status: str
    kl_anchors: dict[str, float]
    floors: dict[str, float]
    source_sha256: str | None
    raw: dict

    def floor_for(self, tier: str) -> float:
        return self.floors[tier]


def _canonical_content(profile: dict) -> str:
    content = {k: v for k, v in profile.items() if k != "profile_hash"}
    return json.dumps(content, sort_keys=True, ensure_ascii=False)


def profile_hash(profile: dict) -> str:
    return hashlib.sha256(_canonical_content(profile).encode("utf-8")).hexdigest()


def validate_guard_profile(profile: dict) -> None:
    if profile.get("evaluator_contract") != "eval-v1":
        raise GuardProfileError("guard profile must pin evaluator_contract: eval-v1")
    if profile.get("scope", {}).get("type") not in SCOPES:
        raise GuardProfileError(f"scope.type must be one of {SCOPES}")
    if not profile.get("scope", {}).get("identifier"):
        raise GuardProfileError("scope.identifier is required")
    source_sha = profile.get("source_sha256")
    if source_sha is not None and (
        not isinstance(source_sha, str) or len(source_sha) != 64
        or any(c not in "0123456789abcdef" for c in source_sha.lower())
    ):
        raise GuardProfileError("source_sha256 must be a 64-hex lowercase digest")
    if profile.get("status") not in _STATUSES:
        raise GuardProfileError(f"status must be one of {_STATUSES}")
    tiers = profile.get("tiers")
    if not isinstance(tiers, dict) or set(tiers) != set(TIERS):
        raise GuardProfileError(f"tiers must define exactly {sorted(TIERS)}")
    for tier, spec in tiers.items():
        if abs(float(spec.get("kl_anchor", -1)) - KL_ANCHORS[tier]) > 1e-9:
            raise GuardProfileError(f"tiers.{tier}.kl_anchor must be {KL_ANCHORS[tier]}")
        floor = float(spec.get("same_top_floor", -1))
        if not 0.0 < floor <= 1.0:
            raise GuardProfileError(f"tiers.{tier}.same_top_floor out of range: {floor}")
    if not profile.get("guard_profile_id"):
        raise GuardProfileError("guard_profile_id is required")
    if "profile_hash" in profile and not profile["profile_hash"]:
        raise GuardProfileError("profile_hash must be non-empty when present")


def load_guard_profile(path: str | Path) -> GuardProfile:
    profile = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(profile, dict):
        raise GuardProfileError(f"guard profile {path} is not a mapping")
    validate_guard_profile(profile)
    expected = profile.get("profile_hash")
    if expected and expected != profile_hash(profile):
        raise GuardProfileError(f"guard profile {path} fails its own profile_hash")
    return GuardProfile(
        profile_id=profile["guard_profile_id"],
        version=int(profile.get("guard_profile_version", 1)),
        scope_type=profile["scope"]["type"],
        scope_identifier=profile["scope"]["identifier"],
        status=profile["status"],
        kl_anchors={t: float(s["kl_anchor"]) for t, s in profile["tiers"].items()},
        floors={t: float(s["same_top_floor"]) for t, s in profile["tiers"].items()},
        source_sha256=(profile.get("source_sha256") or None),
        raw=profile,
    )


def resolve_guard_profile(
    model_name: str,
    registry_dir: str | Path,
    source_sha256: str | None = None,
) -> GuardProfile | None:
    """Find a validated profile covering model_name; None when unvalidated.

    A profile that pins ``source_sha256`` covers the model only when the
    caller supplies the SAME weights digest — a same-named model with
    different weights must not inherit the validated floors. Supplying the
    digest is mandatory for such profiles: without it the model is treated
    as unvalidated.
    """
    registry = Path(registry_dir)
    if not registry.is_dir():
        return None
    matches: list[GuardProfile] = []
    for path in sorted(registry.glob("*.yaml")) + sorted(registry.glob("*.yml")):
        profile = load_guard_profile(path)
        if profile.status != "validated":
            continue
        if profile.source_sha256 is not None:
            if source_sha256 is None or source_sha256.lower() != profile.source_sha256:
                continue
        if profile.scope_type == "exact_model" and profile.scope_identifier == model_name:
            matches.append(profile)
        elif profile.scope_type == "family" and model_name.startswith(profile.scope_identifier):
            matches.append(profile)
    if not matches:
        return None
    # most specific scope wins; ties broken by highest version then id
    rank = {"exact_model": 0, "family": 1, "architecture": 2}
    return sorted(matches, key=lambda p: (rank[p.scope_type], -p.version, p.profile_id))[0]


def require_guard_profile(
    model_name: str,
    tier: str,
    registry_dir: str | Path,
    source_sha256: str | None = None,
) -> GuardProfile:
    """Raise the contract refusal when no validated profile covers the model."""
    tier_key = tier.strip().lower()
    if tier_key not in TIERS:
        raise GuardProfileError(f"unknown fidelity tier: {tier!r} (expected {TIERS})")
    profile = resolve_guard_profile(model_name, registry_dir, source_sha256)
    if profile is None:
        raise GuardProfileError(REFUSE_MESSAGE)
    return profile
