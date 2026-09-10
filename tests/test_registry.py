"""Fidelity Registry v1 (A3) tests: canonical digests, tamper detection,
cross-object closure, grandfather semantics, path safety, CLI behavior.

Real-entry integration tests cover orcarouter + spark through BOTH channels
(registry resolver vs legacy guard registry) and must stay byte-identical on
TierContract fields. The package smoke is opt-in via RUN_PACKAGE_SMOKE=1.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from fit_gguf.eval.contract import contract_digest
from fit_gguf.fidelity import GuardProfileError, profile_hash
from fit_gguf.fidelity_runner import resolve_contract
from fit_gguf import registry as reg

REPO = Path(__file__).resolve().parent.parent
SPARK_SHA = "aa73aeb45870f7ebb9e5d523323b88468a2b3b613918e941bda439d8c1b59d42"
ORCA_SHA = "f95456457fededfaf9f51cd4739a00aada0795be372882844071cc19146634cd"
TIERS = ("quality", "balanced", "compact", "mini")


# ---------------------------------------------------------------- fixtures


def _guard_yaml(source_sha: str, floors: dict[str, float], identifier: str) -> str:
    profile = {
        "evaluator_contract": "eval-v1",
        "generalization_scope": "exact_model_only",
        "guard_profile_id": f"guard-{identifier}-test",
        "guard_profile_version": 1,
        "scope": {"identifier": identifier, "type": "exact_model"},
        "source_sha256": source_sha,
        "status": "validated",
        "tiers": {
            t: {"kl_anchor": a, "same_top_floor": floors[t]}
            for t, a in zip(TIERS, (0.05, 0.10, 0.15, 0.20))
        },
        "validation_basis": "dev_calibration",
    }
    profile["profile_hash"] = profile_hash(profile)
    import yaml

    return yaml.safe_dump(profile, sort_keys=False)


def _build_registry(
    tmp: Path,
    *,
    source_sha: str = SPARK_SHA,
    model_id: str = "model-a",
    floors: dict[str, float] | None = None,
    status: str = "validated",
    tokenizer_sha: str | None = "b" * 64,
    scope: str = "exact_model",
    mutate_entry=None,
    mutate_index=None,
    mutate_guard=None,
    mutate_manifest=None,
) -> Path:
    """Build a minimal synthetic package dir: registry/ + profiles/guard/."""
    root = tmp
    (root / "registry").mkdir(parents=True)
    (root / "registry" / "entries").mkdir()
    (root / "registry" / "manifests").mkdir()
    (root / "registry" / "calibration").mkdir()
    (root / "profiles" / "guard").mkdir(parents=True)

    floors = floors or {"quality": 0.93, "balanced": 0.89, "compact": 0.85, "mini": 0.82}
    guard_path = root / "profiles" / "guard" / "guard-test.yaml"
    guard_text = _guard_yaml(source_sha, floors, model_id)
    if mutate_guard is not None:
        guard_text = mutate_guard(guard_text)
    guard_path.write_text(guard_text, encoding="utf-8")

    manifest = {
        "manifest_schema": "fit.eval_reference_manifest.v1",
        "evaluator_contract": "eval-v1",
        "evaluator_contract_hash": contract_digest(),
        "source_bf16_gguf_sha256": source_sha,
        "tokenizer_sha256": tokenizer_sha,
        "domains": {},
    }
    if mutate_manifest is not None:
        manifest = mutate_manifest(dict(manifest))
    manifest_path = root / "registry" / "manifests" / f"{source_sha}.json"
    manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")

    record = {
        "schema": "fit.calibration_record.v1",
        "model_id": model_id,
        "basis": "legacy_bootstrap_v0",
        "evidence": [],
    }
    record_path = root / "registry" / "calibration" / f"{source_sha}.json"
    record_path.write_text(json.dumps(record, indent=1), encoding="utf-8")

    entry = {
        "registry_schema": reg.REGISTRY_SCHEMA,
        "model_id": model_id,
        "source_weights_sha256": source_sha,
        "tokenizer_sha256": tokenizer_sha,
        "evaluator_contract": "eval-v1",
        "evaluator_contract_sha256": contract_digest(),
        "reference_manifest": {
            "path": f"registry/manifests/{source_sha}.json",
            "sha256": reg.sha256_file(manifest_path),
        },
        "guard_profile": {
            "path": "profiles/guard/guard-test.yaml",
            "sha256": reg.sha256_file(guard_path),
        },
        "calibration": {
            "basis": "legacy_bootstrap_v0",
            "contract": None,
            "contract_sha256": None,
            "record": {
                "path": f"registry/calibration/{source_sha}.json",
                "sha256": reg.sha256_file(record_path),
            },
        },
        "scope": scope,
        "status": status,
        "added_in": "test",
    }
    if source_sha not in reg.GRANDFATHERED_SOURCE_SHAS:
        entry["calibration"] = {
            "basis": "fidelity-calibration-v1",
            "contract": "fidelity-calibration-v1",
            "contract_sha256": "c" * 64,
            "record": entry["calibration"]["record"],
        }
    if mutate_entry is not None:
        entry = mutate_entry(dict(entry))
    entry["entry_sha256"] = reg.entry_digest(entry)
    (root / "registry" / "entries" / f"{source_sha}.json").write_bytes(
        reg.canonical_json_bytes(entry)
    )

    index = {
        "registry_schema": reg.REGISTRY_SCHEMA,
        "registry_id": "fit-official-registry",
        "created": "2026-09-06",
        "evaluator_contract": "eval-v1",
        "evaluator_contract_sha256": contract_digest(),
        "entries": [
            {
                "source_weights_sha256": source_sha,
                "entry_sha256": entry["entry_sha256"],
                "model_id": model_id,
                "status": status,
                "scope": scope,
            }
        ],
    }
    if mutate_index is not None:
        index = mutate_index(dict(index))
    (root / "registry" / "fidelity-registry-v1.json").write_bytes(
        reg.canonical_json_bytes(index)
    )
    return root


# ------------------------------------------------- canonicalization basics


def test_canonical_bytes_are_stable_and_compact():
    obj = {"b": 1, "a": "中文"}
    a = reg.canonical_json_bytes(obj)
    b = reg.canonical_json_bytes({"a": "中文", "b": 1})
    assert a == b
    assert b" " not in a and b"\n" not in a
    assert "中文".encode("utf-8") in a


def test_entry_digest_ignores_its_own_field_and_formatting():
    entry = {"a": 1, "entry_sha256": "0" * 64}
    d1 = reg.entry_digest(entry)
    pretty = json.dumps({k: v for k, v in entry.items() if k != "entry_sha256"}, indent=4)
    reloaded = json.loads(pretty)
    assert reg.entry_digest(reloaded) == d1


# ------------------------------------------------------ real registry verify


def test_real_registry_verifies_and_resolves_spark():
    root = reg.find_package_dir(None)
    report = reg.verify_registry(root)
    ids = {r["model_id"] for r in report["entries_verified"]}
    assert {"spark-x25-4b-abliterated", "orcarouter-Qwen3.8-27B-Uncensored"} <= ids
    assert "NOT" in report["note"]
    resolved = reg.resolve_source(SPARK_SHA)
    assert resolved.floor_for("balanced") == pytest.approx(0.8914)


def test_real_dual_channel_contract_equality():
    """A3 acceptance: registry channel and legacy guard channel must produce
    identical TierContract fields for orcarouter and spark, all four tiers.

    Since Contract v2 the gate is KL-only; ``same_top_reference`` is the
    informational field both channels must still agree on.
    """
    for sha, name in ((SPARK_SHA, "spark-x25-4b-abliterated"),
                      (ORCA_SHA, "orcarouter-Qwen3.8-27B-Uncensored")):
        for tier in TIERS:
            via_registry = reg.resolve_tier_contract(sha, tier)
            via_legacy = resolve_contract(name, tier, reg.default_guard_registry(), sha)
            assert via_registry.kl_anchor == via_legacy.kl_anchor
            assert via_registry.tier == via_legacy.tier
            assert via_registry.same_top_reference == via_legacy.same_top_reference
            assert via_registry.same_top_reference is not None


def test_unregistered_source_still_gets_a_usable_tier_contract():
    """v0.3: naming a tier no longer requires a validated Guard Profile.

    The hard gate is the global KL anchor, so an unknown model gets a real
    contract — just with no Same-top reference to report against.
    """
    contract = reg.resolve_tier_contract("f" * 64, "balanced")
    assert contract.tier == "balanced"
    assert contract.kl_anchor == 0.10
    assert contract.same_top_reference is None


def test_unknown_source_is_refused():
    with pytest.raises(GuardProfileError):
        reg.resolve_source("e" * 64)


# --------------------------------------------------------- synthetic tamper


def test_tampered_entry_fails_digest(tmp_path):
    root = _build_registry(tmp_path)
    path = root / "registry" / "entries" / f"{SPARK_SHA}.json"
    entry = json.loads(path.read_text())
    entry["model_id"] = "tampered"
    path.write_bytes(reg.canonical_json_bytes(entry))
    with pytest.raises(reg.RegistryError, match="entry_sha256"):
        reg.verify_registry(root)


def test_tampered_guard_fails_hash(tmp_path):
    root = _build_registry(tmp_path)
    guard = root / "profiles" / "guard" / "guard-test.yaml"
    guard.write_text(guard.read_text() + "\n# touched\n", encoding="utf-8")
    with pytest.raises(reg.RegistryError, match="guard_profile sha256 mismatch"):
        reg.verify_registry(root)


def test_tampered_manifest_fails_hash(tmp_path):
    root = _build_registry(tmp_path)
    m = root / "registry" / "manifests" / f"{SPARK_SHA}.json"
    data = json.loads(m.read_text())
    data["source_bf16_gguf_sha256"] = "f" * 64
    m.write_text(json.dumps(data, indent=1))
    with pytest.raises(reg.RegistryError, match="reference manifest sha256 mismatch"):
        reg.verify_registry(root)


def test_duplicate_index_sha_fails(tmp_path):
    def dup(index):
        index["entries"] = index["entries"] + [dict(index["entries"][0])]
        return index
    root = _build_registry(tmp_path, mutate_index=dup)
    with pytest.raises(reg.RegistryError, match="duplicate source sha"):
        reg.verify_registry(root)


def _write_second_entry(root: Path, sha: str, model_id: str) -> dict:
    """Add a fully consistent second entry under `sha` (calibration-contract basis)."""
    (root / "registry" / "manifests" / f"{sha}.json").write_text(
        json.dumps({
            "manifest_schema": "fit.eval_reference_manifest.v1",
            "evaluator_contract_hash": contract_digest(),
            "source_bf16_gguf_sha256": sha,
            "tokenizer_sha256": "b" * 64,
            "domains": {},
        }, indent=1))
    (root / "registry" / "calibration" / f"{sha}.json").write_text("{}")
    guard = root / "profiles" / "guard" / f"guard-{model_id}.yaml"
    guard.write_text(_guard_yaml(sha, {"quality": 0.9, "balanced": 0.88, "compact": 0.86, "mini": 0.84}, model_id))
    entry = {
        "registry_schema": reg.REGISTRY_SCHEMA,
        "model_id": model_id,
        "source_weights_sha256": sha,
        "tokenizer_sha256": "b" * 64,
        "evaluator_contract": "eval-v1",
        "evaluator_contract_sha256": contract_digest(),
        "reference_manifest": {"path": f"registry/manifests/{sha}.json",
                               "sha256": reg.sha256_file(root / "registry" / "manifests" / f"{sha}.json")},
        "guard_profile": {"path": f"profiles/guard/guard-{model_id}.yaml",
                          "sha256": reg.sha256_file(guard)},
        "calibration": {"basis": "fidelity-calibration-v1", "contract": "fidelity-calibration-v1",
                        "contract_sha256": "c" * 64,
                        "record": {"path": f"registry/calibration/{sha}.json",
                                   "sha256": reg.sha256_file(root / "registry" / "calibration" / f"{sha}.json")}},
        "scope": "exact_model", "status": "validated", "added_in": "test",
    }
    entry["entry_sha256"] = reg.entry_digest(entry)
    (root / "registry" / "entries" / f"{sha}.json").write_bytes(reg.canonical_json_bytes(entry))
    return entry


def test_unsorted_index_fails(tmp_path):
    root = _build_registry(tmp_path)
    entry_b = _write_second_entry(root, "9" * 64, "model-b")
    index = json.loads((root / "registry" / "fidelity-registry-v1.json").read_text())
    index["entries"].append({
        "source_weights_sha256": "9" * 64,
        "entry_sha256": entry_b["entry_sha256"],
        "model_id": "model-b", "status": "validated", "scope": "exact_model",
    })
    (root / "registry" / "fidelity-registry-v1.json").write_bytes(
        reg.canonical_json_bytes(index))
    with pytest.raises(reg.RegistryError, match="deterministically sorted"):
        reg.verify_registry(root)


def test_filename_mismatch_fails(tmp_path):
    root = _build_registry(tmp_path)
    wrong = "d" * 64
    shutil.copy(
        root / "registry" / "entries" / f"{SPARK_SHA}.json",
        root / "registry" / "entries" / f"{wrong}.json",
    )
    index = json.loads((root / "registry" / "fidelity-registry-v1.json").read_text())
    index["entries"].append({
        "source_weights_sha256": wrong,
        "entry_sha256": index["entries"][0]["entry_sha256"],
        "model_id": "model-c",
        "status": "validated",
        "scope": "exact_model",
    })
    (root / "registry" / "fidelity-registry-v1.json").write_bytes(
        reg.canonical_json_bytes(index)
    )
    with pytest.raises(reg.RegistryError):
        reg.verify_registry(root)


def test_cross_object_source_mismatch_fails(tmp_path):
    """Every file hash is correct — only the semantic closure catches that the
    guard profile binds a DIFFERENT model's weights."""
    root = _build_registry(tmp_path)
    guard = root / "profiles" / "guard" / "guard-test.yaml"
    guard.write_text(_guard_yaml("9" * 64, {"quality": 0.9, "balanced": 0.88,
                                            "compact": 0.86, "mini": 0.84}, "model-a"))
    entry_path = root / "registry" / "entries" / f"{SPARK_SHA}.json"
    entry = json.loads(entry_path.read_text())
    entry["guard_profile"]["sha256"] = reg.sha256_file(guard)
    entry.pop("entry_sha256")
    entry["entry_sha256"] = reg.entry_digest(entry)
    entry_path.write_bytes(reg.canonical_json_bytes(entry))
    index = json.loads((root / "registry" / "fidelity-registry-v1.json").read_text())
    index["entries"][0]["entry_sha256"] = entry["entry_sha256"]
    (root / "registry" / "fidelity-registry-v1.json").write_bytes(
        reg.canonical_json_bytes(index)
    )
    with pytest.raises(reg.RegistryError, match="does not bind"):
        reg.verify_registry(root)


def test_tokenizer_cross_object_mismatch_fails(tmp_path):
    def manifest_drift(manifest):
        manifest["tokenizer_sha256"] = "f" * 64
        return manifest
    root = _build_registry(tmp_path, mutate_manifest=manifest_drift)
    with pytest.raises(reg.RegistryError, match="tokenizer_sha256 disagree"):
        reg.verify_registry(root)


def test_evaluator_drift_fails(tmp_path):
    def drift(entry):
        entry["evaluator_contract_sha256"] = "e" * 64
        return entry
    root = _build_registry(tmp_path, mutate_entry=drift)
    with pytest.raises(reg.RegistryError, match="live eval-v1"):
        reg.verify_registry(root)


def test_family_scope_reserved(tmp_path):
    root = _build_registry(tmp_path, scope="family")
    with pytest.raises(reg.RegistryError, match="reserved"):
        reg.verify_registry(root)


def test_candidate_not_resolvable(tmp_path):
    root = _build_registry(tmp_path, status="candidate")
    with pytest.raises(GuardProfileError, match="candidate"):
        reg.resolve_source(SPARK_SHA, root)


def test_path_safety(tmp_path):
    root = _build_registry(tmp_path)
    entry_path = root / "registry" / "entries" / f"{SPARK_SHA}.json"
    entry = json.loads(entry_path.read_text())
    entry["guard_profile"]["path"] = "../../etc/passwd"
    entry["guard_profile"]["sha256"] = "a" * 64
    entry.pop("entry_sha256")
    entry["entry_sha256"] = reg.entry_digest(entry)
    entry_path.write_bytes(reg.canonical_json_bytes(entry))
    index = json.loads((root / "registry" / "fidelity-registry-v1.json").read_text())
    index["entries"][0]["entry_sha256"] = entry["entry_sha256"]
    (root / "registry" / "fidelity-registry-v1.json").write_bytes(
        reg.canonical_json_bytes(index))
    with pytest.raises(reg.RegistryError, match="escapes|relative"):
        reg.verify_registry(root)


# ------------------------------------------------------------ grandfather


def test_grandfather_new_source_admission_refused(tmp_path):
    entry = {
        "registry_schema": reg.REGISTRY_SCHEMA,
        "model_id": "new-model",
        "source_weights_sha256": "c" * 64,
        "calibration": {"basis": reg.LEGACY_BASIS, "contract": None,
                        "contract_sha256": None},
    }
    entry["entry_sha256"] = reg.entry_digest(entry)
    (tmp_path / "registry-entry.json").write_bytes(reg.canonical_json_bytes(entry))
    with pytest.raises(reg.RegistryError, match="closed to new sources"):
        reg.validate_bundle(tmp_path)


def test_grandfathered_source_keeps_legacy_basis(tmp_path):
    root = _build_registry(tmp_path, source_sha=ORCA_SHA, model_id="orcarouter-Qwen3.8-27B-Uncensored")
    report = reg.verify_registry(root)
    assert report["entries_verified"][0]["status"] == "validated"


# ------------------------------------------------------------------- CLI


def _cli(capsys, *argv):
    from fit_gguf.cli import main

    code = main(list(argv))
    out = capsys.readouterr().out
    return code, out


def test_cli_list_shows_candidates_and_entries(capsys):
    """The list must agree with the index — not with a frozen entry count.

    Onboarding a model is a release workflow, so pinning "2 entries" here would
    mean every future onboarding breaks this test. Assert the invariants that
    matter instead: every index entry is listed, and the count matches.
    """
    code, out = _cli(capsys, "registry", "list")
    assert code == 0

    index = reg.load_index(reg.find_package_dir(None))
    expected = {row["model_id"] for row in index["entries"]}
    assert {"spark-x25-4b-abliterated", "orcarouter-Qwen3.8-27B-Uncensored"} <= expected
    assert "spark-x25-4b-abliterated" in out
    assert "orcarouter-Qwen3.8-27B-Uncensored" in out
    assert f"{len(index['entries'])} entries" in out
    for row in index["entries"]:
        assert row["source_weights_sha256"] in out


def test_cli_show_by_model_id(capsys):
    code, out = _cli(capsys, "registry", "show", "spark-x25-4b-abliterated")
    assert code == 0
    assert SPARK_SHA in out


def test_cli_verify_note(capsys):
    code, out = _cli(capsys, "registry", "verify")
    assert code == 0
    assert "NOT an official-authenticity attestation" in out


# --------------------------------------------------------- package smoke


def test_package_data_config_lists_registry_assets():
    pyproject = (REPO / "pyproject.toml").read_text()
    assert "registry/**/*.json" in pyproject
    assert "profiles/guard/*.yaml" in pyproject
    pkg = reg.find_package_dir(None)
    assert (pkg / "registry" / "fidelity-registry-v1.json").is_file()
    assert (pkg / "profiles" / "guard" / "guard-spark-x25-4b-abliterated-exact-v1.yaml").is_file()


@pytest.mark.skipif(
    os.environ.get("RUN_PACKAGE_SMOKE") != "1",
    reason="builds a wheel and a venv; opt in with RUN_PACKAGE_SMOKE=1",
)
def test_clean_wheel_registry_verify():
    """GPT-required package smoke: install the wheel into a fresh venv and run
    `fit registry verify` — registry/, profiles/guard/ and manifests must ship."""
    import tempfile
    import venv

    with tempfile.TemporaryDirectory() as td:
        wheel_dir = Path(td) / "wheel"
        subprocess.run(
            [sys.executable, "-m", "pip", "wheel", ".", "--no-deps",
             "--no-build-isolation", "-w", str(wheel_dir)],
            cwd=REPO, check=True, capture_output=True,
        )
        wheel = next(wheel_dir.glob("*.whl"))
        env_dir = Path(td) / "venv"
        venv.create(env_dir, with_pip=True, system_site_packages=True)
        pip = env_dir / "bin" / "pip"
        subprocess.run([str(pip), "install", "--no-deps", str(wheel)],
                       check=True, capture_output=True)
        py = env_dir / "bin" / "python"
        result = subprocess.run(
            [str(py), "-c",
             "import sys; sys.path = [p for p in sys.path if 'src' not in p];"
             "from fit_gguf.cli import main; sys.exit(main(['registry', 'verify']))"],
            capture_output=True, text=True, cwd=td,
        )
        assert result.returncode == 0, result.stderr
        assert "spark-x25-4b-abliterated" in result.stdout
