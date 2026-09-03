"""Tests for Guard Profile artifacts and the Fidelity Contract refusal."""

from pathlib import Path

import pytest
import yaml

from fit_gguf.fidelity import (
    GuardProfileError,
    load_guard_profile,
    profile_hash,
    require_guard_profile,
    resolve_guard_profile,
)


def _profile_dict(**overrides) -> dict:
    profile = {
        "guard_profile_id": "guard-test-exact-v1",
        "guard_profile_version": 1,
        "evaluator_contract": "eval-v1",
        "scope": {"type": "exact_model", "identifier": "test-model"},
        "calibration_models": ["test-model"],
        "calibration_manifest_hashes": [],
        "tiers": {
            "quality": {"kl_anchor": 0.05, "same_top_floor": 0.94},
            "balanced": {"kl_anchor": 0.10, "same_top_floor": 0.91},
            "compact": {"kl_anchor": 0.15, "same_top_floor": 0.88},
            "mini": {"kl_anchor": 0.20, "same_top_floor": 0.85},
        },
        "confidence": {},
        "status": "validated",
        "profile_hash": "PENDING",
    }
    profile.update(overrides)
    profile["profile_hash"] = profile_hash(profile)
    return profile


def _profile(tmp_path: Path, **overrides) -> Path:
    path = tmp_path / "guard-test.yaml"
    path.write_text(yaml.safe_dump(_profile_dict(**overrides), sort_keys=True), encoding="utf-8")
    return path


def test_resolve_prefers_exact_model_over_family(tmp_path: Path):
    _profile(tmp_path)
    family = _profile_dict(
        guard_profile_id="guard-family-v1",
        scope={"type": "family", "identifier": "test"},
    )
    (tmp_path / "guard-family.yaml").write_text(
        yaml.safe_dump(family, sort_keys=True), encoding="utf-8"
    )
    resolved = resolve_guard_profile("test-model", tmp_path)
    assert resolved is not None
    assert resolved.scope_type == "exact_model"


def test_load_and_self_hash(tmp_path: Path):
    path = _profile(tmp_path)
    profile = load_guard_profile(path)
    assert profile.profile_id == "guard-test-exact-v1"
    assert profile.floor_for("balanced") == 0.91
    assert profile.kl_anchors["mini"] == 0.20


def test_tampered_profile_rejected(tmp_path: Path):
    path = _profile(tmp_path)
    profile = yaml.safe_load(path.read_text())
    profile["tiers"]["mini"]["same_top_floor"] = 0.99
    path.write_text(yaml.safe_dump(profile), encoding="utf-8")
    with pytest.raises(GuardProfileError, match="profile_hash"):
        load_guard_profile(path)


def test_require_refuses_unvalidated_model(tmp_path: Path):
    with pytest.raises(GuardProfileError) as exc:
        require_guard_profile("unknown-model", "balanced", tmp_path)
    assert "Fidelity validation unavailable" in str(exc.value)
    assert "--calibrate-fidelity" in str(exc.value)


def test_require_refuses_candidate_status(tmp_path: Path):
    _profile(tmp_path, status="candidate")
    with pytest.raises(GuardProfileError, match="Fidelity validation unavailable"):
        require_guard_profile("test-model", "balanced", tmp_path)


def test_require_passes_validated(tmp_path: Path):
    _profile(tmp_path)
    profile = require_guard_profile("test-model", "compact", tmp_path)
    assert profile.floor_for("compact") == 0.88


def test_require_rejects_unknown_tier(tmp_path: Path):
    _profile(tmp_path)
    with pytest.raises(GuardProfileError, match="unknown fidelity tier"):
        require_guard_profile("test-model", "ultra", tmp_path)
