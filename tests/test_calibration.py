"""fidelity-calibration-v1 contract library + fit calibrate tests.

Covers the planner-mandated normative fixtures (window locality, duplicate
observation, calibration witness, expected-vs-estimate tokens, decimal
truncation boundaries) plus Gate P2-A: the zero-eval replay of the archived
P1 Spark contract replay.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fit_gguf import calibration as cal
from fit_gguf import registry as reg
from fit_gguf.calibration import CalibrationError
from fit_gguf.eval import contract_digest
from fit_gguf.fidelity_runner import load_seeds

REPO = Path(__file__).resolve().parent.parent
SPARK_SHA = "aa73aeb45870f7ebb9e5d523323b88468a2b3b613918e941bda439d8c1b59d42"
CONTRACT, CONTRACT_SHA = cal.load_contract(None)


def _obs(point, kl, top, sha=None, size=1_000_000):
    return {"point_id": point, "macro_kl": kl, "same_top": top,
            "artifact_sha256": sha or hashlib_sha(point), "size_bytes": size}


def hashlib_sha(name: str) -> str:
    import hashlib
    return hashlib.sha256(name.encode()).hexdigest()


# ------------------------------------------------------------ normative 1


def test_window_locality():
    """KL .01 must NOT count for the quality window; .049 must."""
    anchor = CONTRACT["tier_kl_anchors"]["quality"]
    lo, hi = cal.window_bounds(anchor, CONTRACT)
    assert lo == pytest.approx(0.0425)
    assert hi == pytest.approx(0.0575)
    near = _obs("near", 0.049, 0.940)
    far = _obs("far", 0.010, 0.970)
    result = cal.evaluate_observations([near, far], CONTRACT)
    q = result["tiers"]["quality"]
    assert [s["point_id"] for s in q["samples"]] == ["near"]


# ------------------------------------------------------------ normative 2


def test_duplicate_observations_dedup_by_artifact_sha():
    a = _obs("p1", 0.14, 0.85)
    dup = _obs("p1-re-eval", 0.14, 0.85, sha=a["artifact_sha256"])
    third = _obs("p1-again", 0.14, 0.85, sha=a["artifact_sha256"])
    result = cal.evaluate_observations([a, dup, third], CONTRACT)
    assert result["tiers"]["compact"]["sample_count"] == 1


# ------------------------------------------------------------ normative 3


def test_witness_missing_tier_is_candidate():
    # three in-window samples, all failing the derived floor gate
    obs = [_obs("w1", 0.155, 0.80), _obs("w2", 0.16, 0.81), _obs("w3", 0.17, 0.82)]
    result = cal.evaluate_observations(obs, CONTRACT)
    compact = result["tiers"]["compact"]
    assert compact["sample_count"] == 3
    assert compact["witness"] is None
    assert compact["validation_status"] == "candidate"
    assert compact["failure"] == "WITNESS_MISSING"
    assert result["overall_status"] == "candidate"


# ------------------------------------------------- normative 4 (tokens)


def test_expected_tokens_authoritative_estimate_informational():
    # contract semantics: token estimates never masquerade as the pin
    assert CONTRACT["expected_valid_tokens"] == "authoritative_evaluator_derived"
    assert CONTRACT["token_count_estimate_informational"] is True


# --------------------------------------------------------- normative 5


def test_trunc4_decimal_boundaries():
    assert cal.trunc4(0.9350) == 0.935
    assert cal.trunc4(0.9349999999999999) == 0.9349  # binary-ε must not round up
    assert cal.trunc4(0.8503) == 0.8503
    assert cal.trunc4(0.8894) == 0.8894
    assert cal.trunc4(0.8145999999999999) == 0.8145


def test_p05_linear_interpolation():
    assert cal.p05([0.90]) == 0.90
    assert cal.p05([0.8345, 0.8488, 0.8600]) == pytest.approx(0.83593)
    assert cal.p05([0.80, 0.90]) == pytest.approx(0.805)


# --------------------------------------------------------- contract integrity


def test_contract_loads_and_digest_binds():
    contract, sha = cal.load_contract(None)
    assert contract["contract_id"] == "fidelity-calibration-v1"
    assert sha == "455338c52de7aa1b4d826ae759a69f8837476ffdb1c16e2d007a289a03c54d47"
    tampered = dict(contract)
    tampered["probe_budget_per_tier"] = 400
    import hashlib
    from fit_gguf.registry import canonical_json_bytes
    payload = {k: v for k, v in tampered.items() if k != "contract_sha256"}
    assert hashlib.sha256(canonical_json_bytes(payload)).hexdigest() != sha


def test_unknown_contract_file_rejected(tmp_path):
    bad = tmp_path / "contract.json"
    bad.write_text(json.dumps({"contract_id": "fidelity-calibration-v1", "nope": 1}))
    with pytest.raises(CalibrationError):
        cal.load_contract(bad)


# ------------------------------------------------------- n<=2 min fallback


def test_two_samples_min_fallback_candidate():
    obs = [_obs("m1", 0.19, 0.8145999999999999), _obs("m2", 0.22, 0.83)]
    result = cal.evaluate_observations(obs, CONTRACT)
    mini = result["tiers"]["mini"]
    assert mini["sample_count"] == 2
    assert mini["floor_method"] == "min_fallback"
    assert mini["floor"] == pytest.approx(0.8145)  # Decimal truncation of stored value
    assert mini["validation_status"] == "candidate"
    assert mini["failure"] == "INSUFFICIENT_WINDOW"


def test_short_tier_does_not_demote_healthy_tiers():
    """Tier-local INSUFFICIENT_WINDOW must stay tier-local: with no session
    hard failures, a healthy tier stays validated (Gate B regression — the
    caller once passed the unresolved set as extra_failures and demoted
    quality/compact alongside the short tiers)."""
    obs = [
        _obs("q1", 0.0458, 0.9134), _obs("q2", 0.0511, 0.9086),
        _obs("q3", 0.0450, 0.9120),          # quality window, 3 samples
        _obs("b1", 0.1129, 0.8699),          # balanced: 1 sample only
    ]
    result = cal.evaluate_observations(obs, CONTRACT)
    assert result["tiers"]["quality"]["validation_status"] == "validated"
    assert result["tiers"]["quality"]["failure"] is None
    assert result["tiers"]["balanced"]["validation_status"] == "candidate"
    # sample sits above the anchor, so the witness check fires first
    assert result["tiers"]["balanced"]["failure"] == "WITNESS_MISSING"
    assert result["overall_status"] == "candidate"


def test_session_hard_failure_demotes_every_tier():
    """extras stay reserved for session-hard failures (imatrix coverage...):
    those must demote even otherwise-validated tiers — fail-closed."""
    obs = [
        _obs("q1", 0.0458, 0.9134), _obs("q2", 0.0511, 0.9086),
        _obs("q3", 0.0450, 0.9120),
    ]
    result = cal.evaluate_observations(obs, CONTRACT, extra_failures=["IMATRIX_COVERAGE_OPEN"])
    assert result["tiers"]["quality"]["validation_status"] == "candidate"
    assert result["overall_status"] == "candidate"
    assert "IMATRIX_COVERAGE_OPEN" in result["open_failures"]


# ------------------------------------------------------------- Gate P2-A


def test_gate_p2a_replay_matches_frozen_p1_replay():
    """Gate P2-A (planner verdict p1-frozen): the implementation must reproduce
    the archived P1 replay field-for-field from the recorded observations."""
    archived = json.loads(
        (REPO / "experiments/2026-09-06-spark-x25-4tier/results/contract-replay-spark.json")
        .read_text()
    )
    summary = REPO / "experiments/2026-09-06-spark-x25-4tier/results/spark-ladder-summary.json"
    manifest = REPO / "experiments/2026-09-06-spark-x25-4tier/state/artifact-manifest.txt"
    cfg_out = Path(pytest.tmp_path) if hasattr(pytest, "tmp_path") else None  # unused
    from fit_gguf.calibrate import CalibrateConfig, replay_existing

    cfg = CalibrateConfig(
        source=Path("/dev/null"), imatrix_corpus=Path("/dev/null"),
        runtime_dir=Path("/dev/null"), eval_data_dir=REPO / "eval-data",
        out_dir=REPO / "experiments/2026-09-06-v03-p1-calibration-contract/results",
        model_id="spark-x25-4b-abliterated",
    )
    report = replay_existing(cfg, summary, manifest)
    assert report["overall_status"] == archived["overall_status"] == "candidate"
    assert report["open_failures"] == archived["open_failures"] == ["INSUFFICIENT_WINDOW"]
    for tier in ("quality", "balanced", "compact", "mini"):
        mine, theirs = report["tiers"][tier], archived["tiers"][tier]
        assert mine["sample_count"] == theirs["sample_count"], tier
        assert mine["floor"] == theirs["floor"], tier
        assert mine["floor_method"] == theirs["floor_method"], tier
        assert mine["validation_status"] == theirs["validation_status"], tier
        assert bool(mine["witness"]) == bool(theirs["witness"]), tier
    # exact expected values from the frozen replay
    assert report["tiers"]["compact"]["floor"] == 0.8359
    assert report["tiers"]["mini"]["floor"] == 0.8145


def test_validate_bundle_accepts_draft_entry(tmp_path):
    entry = {
        "registry_schema": reg.REGISTRY_SCHEMA,
        "model_id": "fresh-model",
        "source_weights_sha256": "c" * 64,
        "tokenizer_sha256": None,
        "evaluator_contract": "eval-v1",
        "evaluator_contract_sha256": CONTRACT["evaluator_contract_sha256"],
        "reference_manifest": {"path": "reference-manifest.json", "sha256": "a" * 64},
        "guard_profile": {"path": "guard-profile.yaml", "sha256": "b" * 64},
        "calibration": {
            "basis": "fidelity-calibration-v1",
            "contract": "fidelity-calibration-v1",
            "contract_sha256": CONTRACT_SHA,
            "record": {"path": "calibration-record.json", "sha256": "d" * 64},
        },
        "scope": "exact_model",
        "status": "candidate",
        "added_in": "PENDING",
    }
    entry["entry_sha256"] = reg.entry_digest(entry)
    (tmp_path / "registry-entry.json").write_bytes(reg.canonical_json_bytes(entry))
    # referenced files absent -> admission must refuse on pins, not on basis
    from fit_gguf.registry import RegistryError
    with pytest.raises(RegistryError, match="referenced file missing"):
        reg.validate_bundle(tmp_path)


# --------------------------------------------------- gap probes (Gate P2-B)


def test_gap_probes_never_anchor_on_probe_observations(tmp_path, monkeypatch):
    """Regression (Spark Gate B crash): probe observations carry non-preset
    point_ids; they are window evidence but must never be passed to
    pipeline_analyze as a bracketing preset pair. Also: probe budget counts
    probes already spent for the tier (resume semantics)."""
    from fit_gguf import pipeline as pipeline_mod
    from fit_gguf.calibrate import CalibrateConfig, stage_gap_probes

    def sha(name):
        import hashlib
        return hashlib.sha256(name.encode()).hexdigest()

    ladder = [
        # mini window (anchor .20): [0.17, 0.23]
        _obs("Q4_K_M", 0.15, 0.88, sha("Q4_K_M"), size=4_000_000),
        _obs("Q3_K_M", 0.10, 0.86, sha("Q3_K_M"), size=3_000_000),
        _obs("IQ2_XS", 0.25, 0.80, sha("IQ2_XS"), size=2_000_000),  # poison
        _obs("IQ2_XXS", 0.30, 0.78, sha("IQ2_XXS"), size=1_000_000),
    ]
    # a probe left over from an interrupted run: closest KL below the window,
    # but its point_id is not a preset
    stale_probe = _obs("probe-mini-1", 0.165, 0.87, sha("probe-mini-1"),
                       size=3_500_000)
    stale_probe["probe"] = {"tier": "mini", "target_bytes": 3_400_000}
    observations = ladder + [stale_probe]

    analyzed = []

    def fake_analyze(source, imx, runtime, out_dir, *, lower_preset,
                     upper_preset, imatrix_arg=None):
        analyzed.append((lower_preset, upper_preset))
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "analysis.json").write_text("{}")

    def fake_plan(analysis_path, out_prefix, *, target_bytes=None, policy=None,
                  model_name=None, **kw):
        prefix = Path(str(out_prefix))
        prefix.parent.mkdir(parents=True, exist_ok=True)
        Path(str(prefix) + "-plan.json").write_text("{}")
        Path(str(prefix) + "-tensor-types.txt").write_text("blk.0=Q4_K\n")
        return {"target_bytes": target_bytes}

    def fake_quantize(analysis_path, types_path, out_path, **kw):
        Path(out_path).write_bytes(b"gguf")

    eval_calls = {"n": 0}

    def fake_eval(cfg, artifact, refs_dir, point_id, env):
        eval_calls["n"] += 1
        return {"point_id": point_id,
                "size_bytes": 4_500_000,
                "artifact_sha256": sha(f"probe-artifact-{eval_calls['n']}"),
                "macro_kl": 0.18, "same_top": 0.88, "per_domain": {}}

    monkeypatch.setattr(pipeline_mod, "analyze", fake_analyze)
    monkeypatch.setattr(pipeline_mod, "plan", fake_plan)
    monkeypatch.setattr(pipeline_mod, "quantize", fake_quantize)
    monkeypatch.setattr("fit_gguf.calibrate.eval_artifact", fake_eval)

    cfg = CalibrateConfig(
        source=tmp_path / "src.gguf", imatrix_corpus=tmp_path / "corpus.txt",
        runtime_dir=tmp_path / "bin", eval_data_dir=tmp_path / "eval",
        out_dir=tmp_path / "out", model_id="test-model", workdir=tmp_path,
    )
    mini_only = dict(CONTRACT)
    mini_only["tier_kl_anchors"] = {"mini": CONTRACT["tier_kl_anchors"]["mini"]}
    new_obs, unresolved = stage_gap_probes(cfg, {}, tmp_path / "imx",
                                           tmp_path / "refs", observations,
                                           mini_only, [])
    # every analyze bracket was a real preset pair, never a probe tag
    assert analyzed, "expected at least one gap probe"
    from fit_gguf.pipeline import PRESET_FILE_TYPES
    for lo_p, hi_p in analyzed:
        assert lo_p in PRESET_FILE_TYPES and hi_p in PRESET_FILE_TYPES
    # stale probe consumed one unit of the per-tier budget (4): 3 new probes
    assert len(new_obs) == 3
    assert unresolved == []
    assert all(o["point_id"].startswith("probe-mini-") for o in new_obs)


# ------------------------------------------- seed material for the search


def _observation(point_id, size, sha, kl):
    return {
        "point_id": point_id,
        "size_bytes": size,
        "artifact_sha256": sha,
        "macro_kl": kl,
        "same_top": 0.9,
    }


def test_calibration_bundle_feeds_the_search_as_seeds(tmp_path):
    """The end-to-end point of the seed material: a ladder arrives as evidence.

    `fit calibrate` spends five-domain evals on every ladder preset. This test
    drives the exact pair of files it emits back through `load_seeds` — the
    search's own admission path — and requires the ladder to come out the far
    side as budget-free bracket evidence, with the poison preset excluded.
    """
    from fit_gguf.calibrate import write_seed_material

    bundle = tmp_path / "bundle"
    logs = bundle / "logs"
    logs.mkdir(parents=True)
    observations = [
        _observation("IQ3_M", 1_100_000_000, "a" * 64, 0.30),
        _observation("IQ4_XS", 1_300_000_000, "b" * 64, 0.12),
        _observation("Q4_K_M", 1_560_000_000, "c" * 64, 0.07),
        # poison: in the standard ladder, must never become bracket evidence
        _observation("IQ2_XS", 800_000_000, "d" * 64, 1.93),
    ]
    for obs in observations:
        for domain in ("wiki_test", "wiki_valid", "chinese", "code", "agent_chat"):
            (logs / f"eval-{obs['point_id']}-{domain}.log").write_text(
                "====== KL divergence statistics ======\n"
                f"Mean KLD: {obs['macro_kl']:.6f} \u00b1 0.010000\n"
                "Same top p: 90.0000 \u00b1 0.1000 %\n",
                encoding="utf-8",
            )
    manifest_sha = "e" * 64
    write_seed_material(
        bundle,
        observations,
        set(CONTRACT["ladder_standard_presets"]),
        reference_manifest_sha256=manifest_sha,
        evaluator_contract_sha256=contract_digest(),
    )

    seeds = load_seeds(
        bundle / "state-artifact-manifest.txt",
        logs,
        "",
        require_seed_provenance=True,
        reference_manifest_sha256=manifest_sha,
    )
    by_size = {seed.size_bytes: seed for seed in seeds}
    assert set(by_size) == {1_100_000_000, 1_300_000_000, 1_560_000_000}
    assert 800_000_000 not in by_size, "IQ2_XS is poison and must not be a seed"
    assert by_size[1_560_000_000].macro_kl == pytest.approx(0.07, abs=1e-6)


def test_seed_material_is_rejected_against_another_models_manifest(tmp_path):
    """Admission control is the reference-manifest binding, not the names."""
    from fit_gguf.calibrate import write_seed_material

    bundle = tmp_path / "bundle"
    logs = bundle / "logs"
    logs.mkdir(parents=True)
    obs = _observation("IQ4_XS", 1_300_000_000, "b" * 64, 0.12)
    for domain in ("wiki_test", "wiki_valid", "chinese", "code", "agent_chat"):
        (logs / f"eval-IQ4_XS-{domain}.log").write_text(
            "====== KL divergence statistics ======\n"
            "Mean KLD: 0.120000 \u00b1 0.010000\nSame top p: 90.0000 \u00b1 0.1000 %\n",
            encoding="utf-8",
        )
    write_seed_material(
        bundle, [obs], set(CONTRACT["ladder_standard_presets"]),
        reference_manifest_sha256="e" * 64,
        evaluator_contract_sha256=contract_digest(),
    )

    assert load_seeds(
        bundle / "state-artifact-manifest.txt", logs, "",
        require_seed_provenance=True, reference_manifest_sha256="e" * 64,
    ), "matching manifest must be admitted"
    assert not load_seeds(
        bundle / "state-artifact-manifest.txt", logs, "",
        require_seed_provenance=True, reference_manifest_sha256="f" * 64,
    ), "a different model's reference manifest must reject every seed"
