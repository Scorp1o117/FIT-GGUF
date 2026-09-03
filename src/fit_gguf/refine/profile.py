"""Fit a ``fit.refine_profile.v1`` from a refine dataset (v0.2.1 M6).

The v0 fit is deliberately ordinal and fully data-backed:

* ``role_correction`` (C_role) is derived from PRISM's network-measured
  bidirectional role perturbations (wholesale qtype upgrades/downgrades).
  Roles whose upgrade *improves* macro KL get C_role > 1 (they deserve more
  bits than their imatrix importance suggests); roles whose upgrade hurts
  get C_role < 1. This is a rank/ordinal correction, not the calibrated
  network/proxy ratio — that form requires paired proxy-vs-network
  measurements and must go through a G0 preregistered Refine Calibration.
* ``transition_prior`` embeds the measured ``role:band:src->dst`` cell
  deltas (KL % change when retyping).
* ``known_risks`` carries the G7 regression fixtures (maxfree collapse,
  Q3_K_S outlier, IQ2_XS poison) and measured retyping cliffs.

Profiles fitted here carry ``split: dev`` and a null preregistration_id:
they are development artifacts. Per G0, formal validation of any profile
requires a frozen preregistration first.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from fit_gguf.gguf import GGML_TYPE_TRAITS
from fit_gguf.refine.dataset import RefineDataset

PROFILE_SCHEMA = "fit.refine_profile.v1"
FITTER_VERSION = "0.2.0"

# C_role = clamp(1 - upgrade_delta_pct / _C_SCALE, _C_MIN, _C_MAX) with the
# direction flipped for downgrade probes. _C_SCALE spreads the observed
# -5%..+12% macro deltas over the [0.5, 1.5] correction band.
_C_SCALE = 20.0
_C_MIN = 0.5
_C_MAX = 1.5

_PROFILE_ID_SUFFIX_RE = re.compile(r"^role-(upgrade|downgrade)-(?P<role>.+)$")

# Band -> inclusive block range, mirroring PRISM depth_bucket_of(layer, 65, 4)
# (bucket = blk // 16; the MTP blk.64 folds into the late band).
_BAND_BLOCK_RANGES = {0: (0, 15), 1: (16, 31), 2: (32, 47), 3: (48, 64)}

# Chain layer_band labels: "late(blk.N+)" is a block predicate, not a bucket.
_LATE_BAND_RE = re.compile(r"^late\(blk\.(\d+)\s*\+\)$")


def _bpw(qtype: str) -> float:
    """Encoded bits-per-weight from the pinned b10666 type traits."""
    block_size, type_size = GGML_TYPE_TRAITS[qtype.lower()]
    return type_size * 8.0 / block_size

_REGRESSION_FIXTURES = (
    {"fixture": "maxfree", "kind": "interaction-collapse", "observed": "+10.8% macro on composition of near-lossless slots",
     "source": "prism frontier-macro0-v1 (L2 law)"},
    {"fixture": "Q3_K_S", "kind": "preset-outlier", "observed": "11.24G KL 0.2148 worse than cheaper IQ3_XS 0.1512",
     "source": "fit-gguf P4 (Huihui tiers)"},
    {"fixture": "IQ2_XS", "kind": "poison-transition", "observed": "IQ2_XXS->IQ2_XS +0.1037 KLD/G vs golden xs->s -0.1831/G",
     "source": "fit-gguf P6 (IQ2 span fix)"},
    {"fixture": "ffn_down-under-allocation", "kind": "proxy-bias", "observed": "imatrix proxy systematically underestimates ffn_down network sensitivity",
     "source": "prism FINAL_REPORT §2.2-1"},
)


class ProfileFitError(ValueError):
    """Raised when a Refine profile cannot be fitted or validated."""


@dataclass(frozen=True, slots=True)
class FittedRole:
    role: str
    c_role: float
    upgrade_delta_pct: float | None
    downgrade_delta_pct: float | None
    directions: int


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def fit_role_corrections(dataset: RefineDataset) -> tuple[dict[str, float], dict[str, FittedRole]]:
    """Fit ordinal C_role per tensor role from bidirectional perturbations.

    Only G0-cleared records (``dataset.records_for_fitting``) are consumed;
    excluded sealed records can never influence the fit.
    """
    allowed, _ = dataset.records_for_fitting()
    upgrades: dict[str, float] = {}
    downgrades: dict[str, float] = {}
    for record in allowed:
        probe_id = str(record.get("probe_id", ""))
        if not probe_id.startswith("role-"):
            continue
        match = _PROFILE_ID_SUFFIX_RE.match(probe_id)
        if match is None:
            continue
        delta = record.get("delta_macro_kl_pct")
        if not isinstance(delta, (int, float)):
            continue
        role = match.group("role")
        if match.group(1) == "upgrade":
            upgrades[role] = float(delta)
        else:
            downgrades[role] = float(delta)

    if not upgrades and not downgrades:
        raise ProfileFitError("dataset contains no role perturbation probes")

    correction: dict[str, float] = {}
    fitted: dict[str, FittedRole] = {}
    for role in sorted(set(upgrades) | set(downgrades)):
        up = upgrades.get(role)
        down = downgrades.get(role)
        estimates = []
        if up is not None:
            estimates.append(_clamp(1.0 - up / _C_SCALE, _C_MIN, _C_MAX))
        if down is not None:
            estimates.append(_clamp(1.0 + down / _C_SCALE, _C_MIN, _C_MAX))
        c_role = sum(estimates) / len(estimates)
        correction[role] = round(_clamp(c_role, _C_MIN, _C_MAX), 4)
        fitted[role] = FittedRole(
            role=role,
            c_role=correction[role],
            upgrade_delta_pct=up,
            downgrade_delta_pct=down,
            directions=len(estimates),
        )
    return correction, fitted


def _transition_prior(dataset: RefineDataset) -> dict[str, float]:
    cells = dataset.priors.get("role_layer_transition_cells", {}).get("cell_deltas", {})
    return {str(key): round(float(value), 4) for key, value in sorted(cells.items())}


def _band_range_from_label(label: str) -> tuple[int, int] | None:
    """Map a dataset band label to an inclusive block range."""
    label = str(label).strip()
    if label.isdigit() and int(label) in _BAND_BLOCK_RANGES:
        return _BAND_BLOCK_RANGES[int(label)]
    late = _LATE_BAND_RE.match(label)
    if late:
        return int(late.group(1)), 64
    return None


def _parse_cell_key(key: str) -> tuple[str, str, str] | None:
    """Split "role:band:src->dst" -> (role, band_label, "src->dst")."""
    parts = key.split(":", 2)
    if len(parts) != 3 or "->" not in parts[2]:
        return None
    return parts[0], parts[1], parts[2]


def fit_band_cells(dataset: RefineDataset) -> dict:
    """Fit band-conditional correction cells (M6b, bootstrap-v1).

    Two evidence classes, both wholesale-retypes on a base plan (same
    semantic as the role probes, so the same delta->C mapping applies):

    * sensitivity-map anchor cells (``role:bucket:src->dst``): a bucket is
      mapped to its block range. Only *upgrade* cells (dst BPW > src BPW)
      become utility cells — FIT candidates are strict lower->upper
      promotions, so a downgrade response cannot reweight one. Downgrade
      cells stay as protection evidence (they duplicate known_risks cliffs).
    * isolated single-action chain steps whose action carries a parseable
      layer_band (e.g. "late(blk.23+)") and a measured delta: the natural
      validation case is nc-v4 late-ssm_out (k=30, -3.74%) which the
      role-level probe (+3.34% over all 48 tensors) inverted.

    Multi-action chain steps are skipped (non-additive composition, L2) and
    recorded in ``skipped_evidence`` for audit.
    """
    cells: list[dict] = []
    protection: list[dict] = []
    skipped: list[dict] = []

    smap = dataset.priors.get("role_layer_transition_cells", {})
    for key, delta in sorted(smap.get("cell_deltas", {}).items()):
        parsed = _parse_cell_key(str(key))
        if parsed is None:
            continue
        role, band_label, transition = parsed
        src, dst = (part.strip() for part in transition.split("->", 1))
        try:
            band_range = _band_range_from_label(band_label)
            upgrade = _bpw(dst) > _bpw(src)
        except KeyError:
            skipped.append({"cell_id": str(key), "reason": "unknown qtype in transition"})
            continue
        if band_range is None:
            skipped.append({"cell_id": str(key), "reason": f"unparseable band {band_label!r}"})
            continue
        delta = float(delta)
        cell = {
            "cell_id": str(key),
            "role": role,
            "band_label": band_label,
            "min_block": band_range[0],
            "max_block": band_range[1],
            "direction": "upgrade" if upgrade else "downgrade",
            "src_qtype": src,
            "dst_qtype": dst,
            "delta_macro_kl_pct": round(delta, 4),
            "evidence": "sensitivity-map anchor (wholesale band retype)",
            "provenance": str(smap.get("source", "experiments/sensitivity-map.json")),
        }
        if upgrade:
            cell["c"] = round(_clamp(1.0 - delta / _C_SCALE, _C_MIN, _C_MAX), 4)
            cells.append(cell)
        else:
            protection.append(cell)

    allowed, _ = dataset.records_for_fitting()
    for record in allowed:
        if not str(record.get("chain_id", "")):
            continue
        actions = record.get("actions")
        delta = record.get("delta_macro_kl_pct")
        if not isinstance(actions, list) or len(actions) != 1:
            if isinstance(actions, list) and len(actions) > 1:
                skipped.append({
                    "chain_step": f"{record.get('chain_id')}:{record.get('step_label')}",
                    "reason": f"multi-action step ({len(actions)} actions), non-additive",
                })
            continue
        if not isinstance(delta, (int, float)):
            continue
        action = actions[0]
        band_range = _band_range_from_label(str(action.get("layer_band", "")))
        if band_range is None:
            skipped.append({
                "chain_step": f"{record.get('chain_id')}:{record.get('step_label')}",
                "reason": f"unparseable band {action.get('layer_band')!r}",
            })
            continue
        src = str(action.get("src_qtype", ""))
        dst = str(action.get("dst_qtype", ""))
        try:
            upgrade = _bpw(dst) > _bpw(src)
        except KeyError:
            skipped.append({
                "chain_step": f"{record.get('chain_id')}:{record.get('step_label')}",
                "reason": "unknown qtype in transition",
            })
            continue
        delta = float(delta)
        cell = {
            "cell_id": f"{record.get('chain_id')}:{record.get('step_label')}:{action.get('role')}",
            "role": str(action.get("role", "")),
            "band_label": str(action.get("layer_band")),
            "min_block": band_range[0],
            "max_block": band_range[1],
            "direction": "upgrade" if upgrade else "downgrade",
            "src_qtype": src,
            "dst_qtype": dst,
            "delta_macro_kl_pct": round(delta, 4),
            "tensor_count": action.get("tensor_count"),
            "evidence": "isolated single-action chain step (wholesale band retype)",
            "provenance": f"chains.jsonl {record.get('chain_id')} step {record.get('step_index')}",
        }
        if upgrade:
            cell["c"] = round(_clamp(1.0 - delta / _C_SCALE, _C_MIN, _C_MAX), 4)
            cells.append(cell)
        else:
            protection.append(cell)

    return {
        "cells": sorted(cells, key=lambda cell: (cell["role"], cell["min_block"], cell["cell_id"])),
        "protection_cells": sorted(protection, key=lambda cell: (cell["role"], cell["min_block"])),
        "skipped_evidence": skipped,
        "fallback": "role_correction",
        "fit_rule": (
            "utility cells = upgrade-direction wholesale-retypes: "
            "c = clamp(1 - delta_macro_kl_pct/20, 0.5, 1.5); downgrade cells are "
            "protection evidence only (FIT candidates are strict promotions); "
            "application is per (role, block range) — per-(src->dst) marginal "
            "application requires a G0-preregistered calibration (bootstrap-v1)"
        ),
    }


def resolve_band_correction(
    profile: dict, role: str, block: int | None
) -> tuple[float, str | None]:
    """Resolve C for one candidate: narrowest matching utility cell, else C_role.

    ``block`` is the candidate tensor's layer index (None for non-block
    tensors — they can never fall inside a band cell and always take the
    role-level fallback).
    """
    cells = (profile.get("band_correction") or {}).get("cells") or []
    if block is not None:
        matches = [
            cell
            for cell in cells
            if cell.get("role") == role
            and cell.get("direction") == "upgrade"
            and int(cell["min_block"]) <= block <= int(cell["max_block"])
        ]
        if matches:
            best = min(matches, key=lambda cell: int(cell["max_block"]) - int(cell["min_block"]))
            return float(best["c"]), str(best["cell_id"])
    return float(profile.get("role_correction", {}).get(role, 1.0)), None


def _known_risks(dataset: RefineDataset) -> dict:
    cells = dataset.priors.get("role_layer_transition_cells", {}).get("cell_deltas", {})
    cliffs = {
        str(key): round(float(value), 4)
        for key, value in cells.items()
        if float(value) >= 2.0
    }
    return {
        "measured_cliffs": cliffs,
        "blocked_dst_qtypes": {"IQ2_XS": "poison transition on qwen hybrid (fit P6); profile-scoped"},
        "regression_fixtures": list(_REGRESSION_FIXTURES),
    }


def fit_profile(
    dataset: RefineDataset,
    profile_id: str,
    scope: dict,
    created: str = "1970-01-01",
    with_band_cells: bool = True,
) -> dict:
    """Fit a complete refine profile dict from a loaded dataset.

    ``with_band_cells=False`` emits the frozen v0 role-only form (no
    band_correction section) — used to reproduce the bootstrap-v0 baseline.
    """
    allowed, excluded_sealed = dataset.records_for_fitting()
    role_correction, role_evidence = fit_role_corrections(dataset)
    laws = dataset.priors.get("generic_laws", {})
    band = fit_band_cells(dataset) if with_band_cells else None
    c_role_form = (
        "band-conditional-bootstrap-v1"
        if band and band["cells"]
        else "proposal-calibration-bootstrap-v0"
    )
    profile = {
        "schema": PROFILE_SCHEMA,
        "profile_id": profile_id,
        "version": FITTER_VERSION,
        "created": created,
        "split": "dev",
        "preregistration_id": None,
        "preregistration_required": True,
        "source": {
            "dataset_id": dataset.dataset_id,
            "dataset_schema": dataset.priors.get("schema"),
            "dataset_digest": dataset.digest,
            "fitter": "fit_gguf.refine.profile",
            "fitter_version": FITTER_VERSION,
        },
        "scope": scope,
        "governance": {
            "fit_split_policy": "dataset-keyed allowlist, fail-closed for unknown datasets (G0)",
            "fitting_records": len(allowed),
            "included_probe_record_ids": sorted(
                str(r.get("probe_id")) for r in allowed if r.get("probe_id")
            ),
            "excluded_sealed_record_ids": sorted(
                f"{r.get('chain_id') or r.get('probe_id')}:{r.get('split')}" for r in excluded_sealed
            ),
        },
        "role_correction": role_correction,
        "role_evidence": {
            role: {
                "c_role": fitted.c_role,
                "upgrade_delta_pct": fitted.upgrade_delta_pct,
                "downgrade_delta_pct": fitted.downgrade_delta_pct,
                "directions": fitted.directions,
            }
            for role, fitted in sorted(role_evidence.items())
        },
        "band_correction": band,
        "transition_prior": _transition_prior(dataset),
        "known_risks": _known_risks(dataset),
        "generic_laws": {
            "laws": laws.get("laws", []),
            "note": laws.get("note", ""),
        },
        "calibration_status": {
            "c_role_form": c_role_form,
            "calibrated_ratio": False,
            "next_step": "G0-preregistered Refine Calibration (paired proxy/network measurements)",
        },
    }
    if band is None:
        del profile["band_correction"]
    return profile


def validate_profile(profile: dict) -> None:
    """Structural validation for loaded or fitted profiles."""
    if profile.get("schema") != PROFILE_SCHEMA:
        raise ProfileFitError(f"schema {profile.get('schema')!r} != {PROFILE_SCHEMA!r}")
    for key in ("profile_id", "split", "source", "role_correction"):
        if key not in profile:
            raise ProfileFitError(f"missing profile field: {key}")
    for role, value in profile["role_correction"].items():
        if not isinstance(value, (int, float)) or not (_C_MIN <= float(value) <= _C_MAX):
            raise ProfileFitError(f"role_correction[{role!r}] out of range: {value!r}")
    band = profile.get("band_correction")
    if band is not None:
        cells = band.get("cells")
        if not isinstance(cells, list):
            raise ProfileFitError("band_correction.cells must be a list")
        for cell in cells:
            for key in ("role", "cell_id", "min_block", "max_block", "c", "direction"):
                if cell.get(key) is None:
                    raise ProfileFitError(f"band cell {cell!r} missing field {key!r}")
            if not (_C_MIN <= float(cell["c"]) <= _C_MAX):
                raise ProfileFitError(f"band cell {cell['cell_id']!r} c out of range: {cell['c']!r}")
            if int(cell["min_block"]) > int(cell["max_block"]):
                raise ProfileFitError(f"band cell {cell['cell_id']!r} inverted block range")


def load_profile(path: str | Path) -> dict:
    profile = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_profile(profile)
    return profile


def save_profile(profile: dict, path: str | Path) -> Path:
    validate_profile(profile)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(profile, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return out
