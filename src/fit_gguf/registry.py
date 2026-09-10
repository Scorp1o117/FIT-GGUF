"""Fidelity Registry v1 (``fit.fidelity_registry.v1``) — the A3 trust layer.

Planner architecture (ruling 2026-09-06): three layers, each with one job.

* ``eval-v1`` (experiments/2026-09-02-eval-v1/FREEZE.json) — HOW TO MEASURE.
  Permanently frozen; the orcarouter ``manifest_sha256_prefix`` stays as the
  v0.2 historical bootstrap provenance. This module never modifies it.
* ``fidelity-calibration-v1`` (P1) — HOW TO CREATE A MODEL GUARD.
* ``Fidelity Registry v1`` (this module) — WHICH MODEL-SPECIFIC BUNDLES ARE
  TRUSTED. Lookup key is ``source_weights_sha256`` + the live eval-v1 digest,
  never a model name.

Trust root is the FIT release itself (git commit / wheel / GitHub release).
``verify`` here is structural/content integrity verification — it is NOT an
official-authenticity attestation: a locally consistent counterfeit registry
cannot be refuted from content alone. Large ``.kld`` references are transport
assets (TRUST ≠ TRANSPORT); only their manifests ship in the registry.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fit_gguf
from fit_gguf.fidelity import (
    GuardProfileError,
    KL_ANCHORS,
    load_guard_profile,
)
from fit_gguf.eval.contract import contract_digest

REGISTRY_SCHEMA = "fit.fidelity_registry.v1"
EVAL_CONTRACT = "eval-v1"
LEGACY_BASIS = "legacy_bootstrap_v0"
CALIBRATION_CONTRACT = "fidelity-calibration-v1"

# immutable grandfather set: the only source digests that may carry (or keep)
# calibration.basis = legacy_bootstrap_v0. Admission of NEW legacy entries is
# refused forever; runtime resolution of these two official bootstrap entries
# is accepted forever (registry entries are snapshot-deterministic).
GRANDFATHERED_SOURCE_SHAS = frozenset(
    {
        "f95456457fededfaf9f51cd4739a00aada0795be372882844071cc19146634cd",  # orcarouter
        "aa73aeb45870f7ebb9e5d523323b88468a2b3b613918e941bda439d8c1b59d42",  # spark-x25-4b-abliterated
    }
)

INDEX_NAME = "fidelity-registry-v1.json"


class RegistryError(RuntimeError):
    """Raised when the registry fails structural, hash, or semantic checks."""


def canonical_json_bytes(obj: Any) -> bytes:
    """Canonical serialization (registry v1): sorted keys, compact separators,
    no NaN/Inf, UTF-8, no BOM. Callers strip the digest field themselves and
    exclude trailing newlines from canonical bytes."""
    return json.dumps(
        obj,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def entry_digest(entry: dict) -> str:
    """SHA-256 over the canonical bytes of the entry minus its own digest."""
    payload = {k: v for k, v in entry.items() if k != "entry_sha256"}
    return sha256_bytes(canonical_json_bytes(payload))


def find_package_dir(explicit: str | Path | None = None) -> Path:
    """Locate the package directory that contains ``registry/`` and
    ``profiles/guard/``: explicit argument, then the packaged ``fit_gguf``
    directory (works from a source checkout via PYTHONPATH=src and from an
    installed wheel alike). All registry pin paths are relative to it."""
    if explicit is not None:
        root = Path(explicit)
        if not root.is_dir():
            raise RegistryError(f"package directory not found: {root}")
        return root
    root = Path(fit_gguf.__file__).resolve().parent
    if not (root / "registry" / INDEX_NAME).is_file():
        raise RegistryError(
            "packaged registry not found — the wheel/sdist must ship "
            "fit_gguf/registry (package smoke: fit registry verify)"
        )
    return root


def default_guard_registry() -> Path:
    """Packaged Guard Profile directory (single source of truth for floors)."""
    root = find_package_dir() / "profiles" / "guard"
    if not root.is_dir():
        raise GuardProfileError(
            "packaged guard registry not found — fit_gguf/profiles/guard must "
            "ship with the package"
        )
    return root


def _safe_relative(root: Path, rel: str) -> Path:
    """Resolve ``rel`` under ``root``; refuse absolute paths and escapes."""
    if not isinstance(rel, str) or not rel:
        raise RegistryError(f"invalid registry path: {rel!r}")
    candidate = Path(rel)
    if candidate.is_absolute():
        raise RegistryError(f"registry paths must be relative, got: {rel}")
    resolved = (root / candidate).resolve()
    root_resolved = root.resolve()
    if resolved != root_resolved and root_resolved not in resolved.parents:
        raise RegistryError(f"registry path escapes the registry root: {rel}")
    return resolved


def _is_64hex(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value.lower())
    )


def load_index(root: Path) -> dict:
    """``root`` is the package directory containing ``registry/``."""
    index_path = root / "registry" / INDEX_NAME
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RegistryError(f"cannot read registry index {index_path}: {error}") from error
    if index.get("registry_schema") != REGISTRY_SCHEMA:
        raise RegistryError(f"{index_path}: unexpected registry_schema")
    if index.get("evaluator_contract") != EVAL_CONTRACT:
        raise RegistryError(f"{index_path}: evaluator_contract must be {EVAL_CONTRACT}")
    if not _is_64hex(index.get("evaluator_contract_sha256")):
        raise RegistryError(f"{index_path}: evaluator_contract_sha256 must be 64-hex")
    if index.get("evaluator_contract_sha256") != contract_digest():
        raise RegistryError(
            f"{index_path}: evaluator_contract_sha256 does not match the live "
            f"eval-v1 digest — the registry targets a different evaluator closure"
        )
    entries = index.get("entries")
    if not isinstance(entries, list):
        raise RegistryError(f"{index_path}: entries must be a list")
    seen: set[str] = set()
    keys = []
    for row in entries:
        sha = row.get("source_weights_sha256")
        if not _is_64hex(sha):
            raise RegistryError(f"{index_path}: entry with invalid source sha: {sha!r}")
        if sha in seen:
            raise RegistryError(f"{index_path}: duplicate source sha in index: {sha}")
        seen.add(sha)
        if not _is_64hex(row.get("entry_sha256")):
            raise RegistryError(f"{index_path}: entry {sha} missing entry_sha256")
        keys.append([sha, row.get("model_id") or "", row.get("status") or "", row.get("scope") or ""])
    if keys != sorted(keys):
        raise RegistryError(f"{index_path}: entries must be deterministically sorted")
    return index


def load_entry(root: Path, source_sha: str, index: dict | None = None) -> dict:
    if not _is_64hex(source_sha):
        raise RegistryError(f"invalid source sha: {source_sha!r}")
    index = index or load_index(root)
    row = next(
        (r for r in index["entries"] if r.get("source_weights_sha256") == source_sha),
        None,
    )
    if row is None:
        raise GuardProfileError(
            "Fidelity validation unavailable: no registry entry for these source "
            "weights. Options: fit calibrate, or --guard-registry with a local "
            "profile for offline use."
        )
    entry_path = _safe_relative(root, f"registry/entries/{source_sha}.json")
    try:
        entry = json.loads(entry_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RegistryError(f"cannot read entry {entry_path}: {error}") from error
    if entry.get("registry_schema") != REGISTRY_SCHEMA:
        raise RegistryError(f"{entry_path}: unexpected registry_schema")
    if entry.get("source_weights_sha256") != source_sha:
        raise RegistryError(
            f"{entry_path}: entry source_weights_sha256 does not match its file name"
        )
    if row.get("entry_sha256") != entry.get("entry_sha256"):
        raise RegistryError(f"{entry_path}: index entry_sha256 does not match the entry")
    if entry_digest(entry) != entry.get("entry_sha256"):
        raise RegistryError(f"{entry_path}: entry fails its own entry_sha256")
    return entry


def _load_pinned_json(root: Path, pin: dict, what: str) -> dict:
    path = _safe_relative(root, pin.get("path"))
    if not path.is_file():
        raise RegistryError(f"{what} file missing: {pin.get('path')}")
    if sha256_file(path) != pin.get("sha256"):
        raise RegistryError(f"{what} sha256 mismatch: {pin.get('path')}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RegistryError(f"cannot read {what} {path}: {error}") from error


def _check_calibration(entry: dict, root: Path) -> dict:
    cal = entry.get("calibration")
    if not isinstance(cal, dict):
        raise RegistryError("entry.calibration must be an object")
    basis = cal.get("basis")
    record_pin = cal.get("record")
    if not isinstance(record_pin, dict) or not _is_64hex(record_pin.get("sha256")):
        raise RegistryError("entry.calibration.record must pin a sha256")
    _load_pinned_json(root, record_pin, "calibration record")
    if basis == LEGACY_BASIS:
        if entry.get("source_weights_sha256") not in GRANDFATHERED_SOURCE_SHAS:
            raise RegistryError(
                "legacy_bootstrap_v0 is a closed grandfather set: new entries may "
                "not declare it"
            )
        if cal.get("contract") is not None or cal.get("contract_sha256") is not None:
            raise RegistryError("legacy bootstrap entries must leave contract null")
    elif basis == CALIBRATION_CONTRACT:
        if not _is_64hex(cal.get("contract_sha256")):
            raise RegistryError(
                "fidelity-calibration-v1 entries must pin contract_sha256"
            )
    else:
        raise RegistryError(f"unknown calibration basis: {basis!r}")
    return cal


def verify_registry(root: str | Path) -> dict:
    """Full structural + hash + cross-object verification of the registry.

    ``root`` is the package directory containing ``registry/`` and
    ``profiles/guard/``. Returns a report dict; raises RegistryError /
    GuardProfileError on any failure. Semantic bundle closure is enforced:
    entry/guard/manifest must agree on source weights, tokenizer (when pinned
    as 64-hex), and the evaluator contract digest.
    """
    root = Path(root)
    index = load_index(root)
    live_digest = contract_digest()
    verified = []
    for row in index["entries"]:
        sha = row["source_weights_sha256"]
        entry = load_entry(root, sha, index)
        if entry.get("entry_sha256") != row.get("entry_sha256"):
            raise RegistryError(f"entries/{sha}: index/entry digest drift")

        guard_pin = entry.get("guard_profile")
        if not isinstance(guard_pin, dict):
            raise RegistryError(f"entries/{sha}: guard_profile must be a pin object")
        guard_path = _safe_relative(root, guard_pin.get("path"))
        if sha256_file(guard_path) != guard_pin.get("sha256"):
            raise RegistryError(f"entries/{sha}: guard_profile sha256 mismatch")
        guard = load_guard_profile(guard_path)

        manifest_pin = entry.get("reference_manifest")
        if not isinstance(manifest_pin, dict):
            raise RegistryError(f"entries/{sha}: reference_manifest must be a pin object")
        manifest = _load_pinned_json(root, manifest_pin, "reference manifest")

        # ---- cross-object bundle closure (amendment 4) ----
        if guard.source_sha256 != sha:
            raise RegistryError(
                f"entries/{sha}: guard source_sha256 {guard.source_sha256} does not "
                "bind these source weights"
            )
        manifest_source = manifest.get("source_bf16_gguf_sha256")
        if manifest_source != sha:
            raise RegistryError(
                f"entries/{sha}: manifest source_bf16_gguf_sha256 {manifest_source} "
                "does not bind these source weights"
            )
        manifest_tokenizer = manifest.get("tokenizer_sha256")
        entry_tokenizer = entry.get("tokenizer_sha256")
        if _is_64hex(manifest_tokenizer) and manifest_tokenizer != entry_tokenizer:
            raise RegistryError(
                f"entries/{sha}: tokenizer_sha256 disagrees between entry and manifest"
            )
        if entry.get("evaluator_contract_sha256") != live_digest:
            raise RegistryError(
                f"entries/{sha}: evaluator_contract_sha256 does not match the live "
                "eval-v1 digest"
            )
        if manifest.get("evaluator_contract_hash") not in (None, live_digest):
            raise RegistryError(
                f"entries/{sha}: manifest evaluator_contract_hash does not match "
                "the live eval-v1 digest"
            )
        if manifest.get("evaluator_contract_hash") is None:
            raise RegistryError(
                f"entries/{sha}: manifest must pin evaluator_contract_hash"
            )
        if guard_pin.get("sha256") and not _is_64hex(guard_pin.get("sha256")):
            raise RegistryError(f"entries/{sha}: guard pin sha must be 64-hex")

        _check_calibration(entry, root)

        if entry.get("scope") != "exact_model":
            raise RegistryError(
                f"entries/{sha}: scope {entry.get('scope')!r} is not resolvable in "
                "registry v1 (family/architecture are reserved)"
            )
        if entry.get("status") not in ("validated", "candidate"):
            raise RegistryError(f"entries/{sha}: invalid status {entry.get('status')!r}")
        verified.append(
            {
                "source_weights_sha256": sha,
                "model_id": entry.get("model_id"),
                "status": entry.get("status"),
                "floors": guard.floors,
            }
        )
    return {
        "registry_schema": REGISTRY_SCHEMA,
        "evaluator_contract_digest": live_digest,
        "entries_verified": verified,
        "note": "structural/content integrity only — NOT an official-authenticity "
        "attestation (official trust root = the FIT release)",
    }


@dataclass(frozen=True, slots=True)
class ResolvedModel:
    """A validated registry entry with its bound Guard Profile."""

    entry: dict
    guard: Any  # fit_gguf.fidelity.GuardProfile
    registry_root: Path

    @property
    def model_id(self) -> str:
        return str(self.entry["model_id"])

    def floor_for(self, tier: str) -> float:
        return self.guard.floor_for(tier)


def resolve_source(
    source_sha256: str, package_dir: str | Path | None = None
) -> ResolvedModel:
    """Resolve a validated registry entry by source-weights digest.

    ``package_dir`` defaults to the packaged fit_gguf directory. Raises
    GuardProfileError (not validated / no entry) or RegistryError (integrity
    failure) — the same fail-closed contract as the legacy ``resolve_contract``
    path.
    """
    root = find_package_dir(package_dir)
    index = load_index(root)
    entry = load_entry(root, source_sha256.lower(), index)
    if entry.get("status") != "validated":
        raise GuardProfileError(
            f"Registry entry for these source weights is "
            f"{entry.get('status')!r}, not validated — refusing to emit official "
            "fidelity tiers."
        )
    _check_calibration(entry, root)
    guard_pin = entry.get("guard_profile")
    guard_path = _safe_relative(root, guard_pin.get("path"))
    if sha256_file(guard_path) != guard_pin.get("sha256"):
        raise RegistryError("guard_profile sha256 mismatch")
    guard = load_guard_profile(guard_path)
    if guard.source_sha256 != source_sha256.lower():
        raise RegistryError("guard profile does not bind these source weights")
    manifest_pin = entry.get("reference_manifest")
    manifest = _load_pinned_json(root, manifest_pin, "reference manifest")
    if manifest.get("evaluator_contract_hash") != contract_digest():
        raise RegistryError("manifest evaluator contract drift")
    return ResolvedModel(entry=entry, guard=guard, registry_root=root)


def resolve_tier_contract(
    source_sha256: str, tier: str, package_dir: str | Path | None = None
):
    """Registry-channel equivalent of ``fidelity_runner.resolve_contract``."""
    from fit_gguf.fidelity_search import TierContract

    tier_key = tier.strip().lower()
    if tier_key not in KL_ANCHORS:
        raise GuardProfileError(f"unknown fidelity tier: {tier!r}")
    resolved = resolve_source(source_sha256, package_dir)
    return TierContract(
        tier=tier_key,
        kl_anchor=KL_ANCHORS[tier_key],
        same_top_floor=resolved.floor_for(tier_key),
    )


def validate_bundle(bundle_dir: str | Path) -> dict:
    """Read-only admission check for a Calibration Bundle (A1 output shape).

    Checks the draft registry entry's structure, digest, calibration basis
    admission rules, and that the referenced guard/manifest/record files exist
    and hash-match. Does NOT write to the official registry.
    """
    bundle = Path(bundle_dir)
    entry_path = bundle / "registry-entry.json"
    try:
        entry = json.loads(entry_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RegistryError(f"cannot read {entry_path}: {error}") from error
    if entry.get("registry_schema") != REGISTRY_SCHEMA:
        raise RegistryError(f"{entry_path}: unexpected registry_schema")
    problems: list[str] = []
    sha = entry.get("source_weights_sha256")
    if not _is_64hex(sha):
        problems.append("source_weights_sha256 must be 64-hex")
    if entry.get("entry_sha256") != entry_digest(entry):
        problems.append("entry fails its own entry_sha256")
    if entry.get("calibration", {}).get("basis") == LEGACY_BASIS and (
        sha not in GRANDFATHERED_SOURCE_SHAS
    ):
        problems.append("legacy_bootstrap_v0 admission is closed to new sources")
    if entry.get("calibration", {}).get("basis") == CALIBRATION_CONTRACT:
        if not _is_64hex(entry["calibration"].get("contract_sha256")):
            problems.append("calibration contract_sha256 must be 64-hex")
    for key in ("guard_profile", "reference_manifest", "calibration"):
        pin = entry.get(key, {}).get("record") if key == "calibration" else entry.get(key)
        if isinstance(pin, dict):
            target = bundle / str(pin.get("path", ""))
            if not target.is_file() or sha256_file(target) != pin.get("sha256"):
                problems.append(f"{key}: referenced file missing or hash mismatch")
    if problems:
        raise RegistryError("bundle rejected: " + "; ".join(problems))
    return {
        "admissible": True,
        "model_id": entry.get("model_id"),
        "source_weights_sha256": sha,
        "status": entry.get("status"),
        "note": "bundle is admissible — official promotion is a maintainer/release "
        "workflow, not a CLI flag",
    }
