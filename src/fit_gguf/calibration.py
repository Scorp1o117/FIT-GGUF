"""fidelity-calibration-v1 contract library (P1 FROZEN, SHA 455338c5…).

Executes the frozen Calibration Contract semantics over calibration
observations: boundary-local windows W(K)=[0.85K,1.15K], unique-observation
dedup by artifact_sha256, Decimal ROUND_DOWN floor truncation, linear
P5 interpolation, calibration witnesses, and the fail-closed status/failure
enumeration. The contract JSON is authoritative — this module must bend to
it, never the reverse (planner discipline, 2026-09-06).
"""

from __future__ import annotations

import hashlib
import json
from decimal import ROUND_DOWN, Decimal
from pathlib import Path
from typing import Any

import fit_gguf
from fit_gguf.registry import canonical_json_bytes

CONTRACT_ID = "fidelity-calibration-v1"
TIERS = ("quality", "balanced", "compact", "mini")

FAILURE_STATES = (
    "CORPUS_DRIFT",
    "REF_GENERATION_FAILED",
    "REF_RUNTIME_UNVERIFIED",
    "IMATRIX_COVERAGE_MISSING",
    "INSUFFICIENT_WINDOW",
    "NOT_REACHABLE",
    "BUDGET_EXHAUSTED",
    "INPUT_DRIFT",
)


class CalibrationError(RuntimeError):
    """Raised on contract violations or hard calibration failures."""


def default_contract_path() -> Path:
    return Path(fit_gguf.__file__).resolve().parent / "contracts" / f"{CONTRACT_ID}.json"


def load_contract(path: str | Path | None = None) -> tuple[dict, str]:
    """Load the machine contract and verify its self-digest.

    Returns ``(contract, contract_sha256)`` where the digest is computed over
    the canonical bytes of the contract minus its own ``contract_sha256``
    field (A3 canonical JSON v1).
    """
    path = Path(path) if path else default_contract_path()
    try:
        contract = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CalibrationError(f"cannot read calibration contract {path}: {error}") from error
    if contract.get("contract_id") != CONTRACT_ID:
        raise CalibrationError(f"{path}: unexpected contract_id")
    digest = contract.get("contract_sha256")
    payload = {k: v for k, v in contract.items() if k != "contract_sha256"}
    actual = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    if digest != actual:
        raise CalibrationError(f"{path}: contract fails its own contract_sha256")
    return contract, digest


def trunc4(v: float) -> float:
    """§5.2: Decimal ROUND_DOWN at 4 decimals — never binary-float floor."""
    return float(Decimal(repr(v)).quantize(Decimal("0.0001"), rounding=ROUND_DOWN))


def p05(values: list[float]) -> float:
    """§5.3: linear-interpolated 5th percentile (default method)."""
    xs = sorted(values)
    if not xs:
        raise CalibrationError("p05 of empty sample set")
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * 0.05
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def window_bounds(anchor: float, contract: dict) -> tuple[float, float]:
    """§4: W(K) = [0.85K, 1.15K], both ends closed."""
    rule = contract.get("window_rule", {})
    if rule.get("type") != "multiplicative_band":
        raise CalibrationError(f"unsupported window rule: {rule!r}")
    lo_factor = float(rule.get("low_factor", 0.85))
    hi_factor = float(rule.get("high_factor", 1.15))
    return anchor * lo_factor, anchor * hi_factor


def unique_observations(observations: list[dict]) -> list[dict]:
    """§5.5: dedup by artifact_sha256 — repeated evals of the same artifact
    count once; distinct artifacts at the same size are distinct observations.
    Order preserved (first occurrence wins)."""
    seen: set[str] = set()
    out = []
    for obs in observations:
        sha = obs.get("artifact_sha256")
        if not sha:
            raise CalibrationError(
                f"observation {obs.get('point_id')!r} missing artifact_sha256"
            )
        if sha in seen:
            continue
        seen.add(sha)
        out.append(obs)
    return out


def window_samples(
    observations: list[dict], anchor: float, contract: dict
) -> list[dict]:
    lo, hi = window_bounds(anchor, contract)
    return [o for o in observations if lo <= float(o["macro_kl"]) <= hi]


def derive_tier(
    tier: str,
    anchor: float,
    observations: list[dict],
    contract: dict,
    open_failures: list[str] | None = None,
) -> dict:
    """§5.4–5.6: floor derivation, witness, and per-tier validation status."""
    open_failures = open_failures or []
    min_n = int(contract.get("validation_min_samples", 3))
    samples = window_samples(observations, anchor, contract)
    n = len(samples)
    tier_result: dict[str, Any] = {
        "tier": tier,
        "anchor": anchor,
        "window": [round(anchor * float(contract["window_rule"]["low_factor"]), 6),
                   round(anchor * float(contract["window_rule"]["high_factor"]), 6)],
        "sample_count": n,
        "samples": [
            {
                "point_id": s["point_id"],
                "size_bytes": int(s["size_bytes"]),
                "macro_kl": float(s["macro_kl"]),
                "macro_same_top": float(s["same_top"]),
                "artifact_sha256": s["artifact_sha256"],
            }
            for s in sorted(samples, key=lambda s: float(s["macro_kl"]))
        ],
        "floor": None,
        "floor_method": None,
        "witness": None,
        "validation_status": "candidate",
        "failure": None,
    }
    if n == 0:
        tier_result["failure"] = "INSUFFICIENT_WINDOW"
        return tier_result
    tops = [float(s["same_top"]) for s in samples]
    if n >= min_n:
        floor = trunc4(p05(tops))
        method = "empirical_p5"
    else:
        floor = trunc4(min(tops))
        method = "min_fallback"
    tier_result["floor"] = floor
    tier_result["floor_method"] = method
    witnesses = [
        s for s in samples
        if float(s["macro_kl"]) <= anchor and float(s["same_top"]) >= floor
    ]
    if witnesses:
        w = sorted(witnesses, key=lambda s: float(s["macro_kl"]))[0]
        tier_result["witness"] = {
            "point_id": w["point_id"],
            "macro_kl": float(w["macro_kl"]),
            "macro_same_top": float(w["same_top"]),
            "pass": True,
        }
    else:
        tier_result["failure"] = "WITNESS_MISSING"
    if n < min_n and tier_result["failure"] is None:
        tier_result["failure"] = "INSUFFICIENT_WINDOW"
    if open_failures:
        tier_result["failure"] = tier_result["failure"] or open_failures[0]
    tier_result["validation_status"] = (
        "validated" if n >= min_n and tier_result["witness"] and not open_failures
        else "candidate"
    )
    return tier_result


def evaluate_observations(
    observations: list[dict],
    contract: dict,
    extra_failures: list[str] | None = None,
) -> dict:
    """Full contract evaluation: dedup → per-tier derivation → overall status.

    ``extra_failures`` carries session-level open failures (e.g.
    INSUFFICIENT_WINDOW propagated by the caller is derived here; caller-level
    hard failures like CORPUS_DRIFT abort before this call).
    """
    contract_sha = contract.get("contract_sha256")
    if not contract_sha:
        raise CalibrationError("contract missing contract_sha256")
    unique = unique_observations(observations)
    tiers = {}
    for tier in TIERS:
        anchor = float(contract["tier_kl_anchors"][tier])
        tiers[tier] = derive_tier(tier, anchor, unique, contract)
    # a session-level hard failure (e.g. IMATRIX coverage left open) demotes
    # every tier — fail-closed, never silently repaired
    if extra_failures:
        for tier in tiers.values():
            tier["failure"] = tier["failure"] or extra_failures[0]
            tier["validation_status"] = "candidate"
    all_ok = all(t["validation_status"] == "validated" for t in tiers.values())
    failures = sorted({t["failure"] for t in tiers.values() if t["failure"]})
    if extra_failures:
        failures = sorted(set(failures) | set(extra_failures))
    return {
        "contract_id": CONTRACT_ID,
        "calibration_contract_sha256": contract_sha,
        "observation_count": len(unique),
        "tiers": tiers,
        "overall_status": "validated" if all_ok else "candidate",
        "open_failures": failures,
    }
