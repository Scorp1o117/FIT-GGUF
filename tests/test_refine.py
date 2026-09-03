"""Tests for the refine dataset loader, profile fitting, and proposal scoring."""

import json

import pytest

from fit_gguf.refine import (
    ProfileFitError,
    ProposalScorer,
    RefineDatasetError,
    fit_profile,
    fit_role_corrections,
    load_refine_dataset,
    save_profile,
    validate_profile,
)

DATASET_SCHEMA = "fit.refine_dataset.v1"


def _record(**overrides):
    base = {
        "schema": DATASET_SCHEMA,
        "dataset_id": "refine-dataset-v1",
        "exporter": "test",
        "exporter_version": "t",
        "exported_at": "1970-01-01",
        "evaluator_version": "test-evaluator",
    }
    base.update(overrides)
    return base


def _chain(**overrides):
    return _record(
        record_type="chain_step",
        chain_id="c1",
        step_index=0,
        step_label="s0",
        suite={"suite_id": "dev"},
        split="dev",
        actions=[],
        post_state={"macro_kl": 0.1},
        **overrides,
    )


def _probe(probe_id="role-upgrade-ffn_down", delta=-4.8, split="dev", **overrides):
    return _record(
        record_type="single_action_probe",
        probe_id=probe_id,
        suite={"suite_id": "dev"},
        split=split,
        action={"role": "ffn_down"},
        post_state={"macro_kl": 0.1},
        delta_macro_kl_pct=delta,
        **overrides,
    )


def _curve(**overrides):
    return _record(
        record_type="curve_point",
        point_id="p0",
        suite={"suite_id": "dev"},
        size_gib=12.0,
        macro_kl=0.12,
        **overrides,
    )


def _write_dataset(tmp_path, *, chains=None, probes=None, curve=None, priors=None):
    root = tmp_path / "refine-dataset-v1"
    root.mkdir()
    defaults = {
        "chains.jsonl": chains if chains is not None else [_chain()],
        "single_action_probes.jsonl": probes if probes is not None else [_probe()],
        "curve_points.jsonl": curve if curve is not None else [_curve()],
        "probe_index.jsonl": [],
    }
    for name, records in defaults.items():
        with (root / name).open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")
    priors = priors if priors is not None else {"dataset_id": "refine-dataset-v1"}
    (root / "priors.json").write_text(json.dumps(priors), encoding="utf-8")
    return root


def _priors():
    return {
        "dataset_id": "refine-dataset-v1",
        "role_layer_transition_cells": {
            "cell_deltas": {
                "ffn_down:0:q3_K->q2_K": 10.5,
                "ffn_down:0:q3_K->q6_K": -5.3,
                "ffn_down:3:q4_K->q3_K": 5.8,
            }
        },
        "generic_laws": {"laws": ["L1", "L2"], "note": "n"},
    }


def test_load_rejects_missing_directory(tmp_path):
    with pytest.raises(RefineDatasetError):
        load_refine_dataset(tmp_path / "nope")


def test_load_rejects_bad_schema(tmp_path):
    bad = _chain(schema="other.schema")
    root = _write_dataset(tmp_path, chains=[bad])
    with pytest.raises(RefineDatasetError, match="schema"):
        load_refine_dataset(root)


def test_load_rejects_unknown_split(tmp_path):
    root = _write_dataset(tmp_path, probes=[_probe(split="mystery")])
    with pytest.raises(RefineDatasetError, match="unknown split"):
        load_refine_dataset(root)


def test_load_rejects_missing_fields(tmp_path):
    incomplete = _probe()
    del incomplete["post_state"]
    root = _write_dataset(tmp_path, probes=[incomplete])
    with pytest.raises(RefineDatasetError, match="missing fields"):
        load_refine_dataset(root)


def test_role_correction_orders_roles_by_network_evidence(tmp_path):
    probes = [
        _probe("role-upgrade-ffn_down", delta=-4.8),   # upgrading helps -> C > 1
        _probe("role-upgrade-attn_gate", delta=12.4),  # upgrading hurts -> C = floor
        _probe("role-downgrade-attn_v", delta=10.4),   # downgrading hurts -> C > 1
    ]
    dataset = load_refine_dataset(_write_dataset(tmp_path, probes=probes, priors=_priors()))
    correction, fitted = fit_role_corrections(dataset)

    assert correction["ffn_down"] > 1.0
    assert correction["ffn_down"] == pytest.approx(1.0 - (-4.8) / 20.0)
    assert correction["attn_gate"] == pytest.approx(0.5)  # clamped floor
    assert correction["attn_v"] == pytest.approx(1.5)  # 1.52 clamped to ceiling
    assert fitted["attn_v"].directions == 1
    # ffn_down must outrank attn_gate: proxy bias story preserved
    assert correction["ffn_down"] > correction["attn_gate"]


def test_fit_profile_is_dev_and_preregistration_pending(tmp_path):
    dataset = load_refine_dataset(_write_dataset(tmp_path, priors=_priors()))
    profile = fit_profile(dataset, "test-profile", scope={"model_family": "qwen"})

    validate_profile(profile)
    assert profile["split"] == "dev"
    assert profile["preregistration_id"] is None
    assert profile["preregistration_required"] is True
    assert profile["source"]["dataset_id"] == "refine-dataset-v1"
    # the fixture carries an upgrade cell (ffn_down:0:q3_K->q6_K), so the
    # fitted form is band-conditional (M6b); the frozen v0 form is covered by
    # test_refine_band::test_fit_profile_without_band_cells_is_frozen_v0_form
    assert profile["calibration_status"]["c_role_form"] == "band-conditional-bootstrap-v1"
    assert profile["band_correction"]["fallback"] == "role_correction"
    assert profile["transition_prior"]["ffn_down:3:q4_K->q3_K"] == 5.8
    fixtures = profile["known_risks"]["regression_fixtures"]
    assert {f["fixture"] for f in fixtures} == {"maxfree", "Q3_K_S", "IQ2_XS", "ffn_down-under-allocation"}


def test_governance_policy_keys_on_dataset_identity(tmp_path):
    probes = [
        _probe("role-upgrade-ffn_down", delta=-4.8, split="dev"),
        _probe("role-upgrade-ssm_out", delta=3.3, split="sealed"),
    ]
    dataset = load_refine_dataset(_write_dataset(tmp_path, probes=probes, priors=_priors()))

    # refine-dataset-v1's sealed splits are historically opened -> fit as dev
    allowed, excluded = dataset.records_for_fitting()
    assert sorted(r["probe_id"] for r in allowed if "probe_id" in r) == [
        "role-upgrade-ffn_down",
        "role-upgrade-ssm_out",
    ]
    assert excluded == ()

    profile = fit_profile(dataset, "gov", scope={})
    governance = profile["governance"]
    assert governance["fitting_records"] == len(dataset.chains + dataset.probes)
    assert governance["excluded_sealed_record_ids"] == []
    assert "role-upgrade-ssm_out" in governance["included_probe_record_ids"]


def test_governance_fails_closed_for_unknown_dataset(tmp_path):
    priors = _priors()
    priors["dataset_id"] = "future-native-sealed-set"
    dataset = load_refine_dataset(_write_dataset(tmp_path, priors=priors))

    allowed, excluded = dataset.records_for_fitting()
    assert allowed == ()
    assert len(excluded) == len(dataset.chains + dataset.probes)

    with pytest.raises(ProfileFitError, match="no role perturbation"):
        fit_profile(dataset, "sealed-profile", scope={})


def test_profile_roundtrip(tmp_path):
    dataset = load_refine_dataset(_write_dataset(tmp_path, priors=_priors()))
    profile = fit_profile(dataset, "rt", scope={})
    out = save_profile(profile, tmp_path / "sub" / "profile.json")

    from fit_gguf.refine import load_profile

    loaded = load_profile(out)
    assert loaded["profile_id"] == "rt"

    profile["role_correction"]["ffn_down"] = 99.0
    with pytest.raises(ProfileFitError, match="out of range"):
        save_profile(profile, tmp_path / "bad.json")


def _profile_with_risks():
    return {
        "schema": "fit.refine_profile.v1",
        "profile_id": "t",
        "split": "dev",
        "role_correction": {"ffn_down": 1.24, "attn_gate": 0.5},
        "transition_prior": {
            "ffn_down:0:q3_K->q6_K": -5.3,
            "ffn_down:3:q4_K->q3_K": 5.8,
        },
        "known_risks": {
            "measured_cliffs": {"ffn_down:3:q4_K->q3_K": 5.8},
            "blocked_dst_qtypes": {"IQ2_XS": "poison"},
        },
    }


def test_proposal_scores_rank_by_importance_and_bytes():
    scorer = ProposalScorer(_profile_with_risks())

    high = scorer.score("blk.9.ffn_down.weight", "ffn_down", "q3_K", "q6_K", 1000, 2.0, band=0)
    low = scorer.score("blk.9.ffn_down.weight", "ffn_down", "q3_K", "q6_K", 8000, 0.5, band=0)

    assert high.score > low.score
    assert high.transition_gain == pytest.approx(1.053)
    assert high.c_role == pytest.approx(1.24)
    assert not high.caution


def test_proposal_flags_cliff_and_blocked_dst():
    scorer = ProposalScorer(_profile_with_risks())

    cliff = scorer.score("blk.9.ffn_down.weight", "ffn_down", "q4_K", "q3_K", 1000, 1.0, band=3)
    poison = scorer.score("blk.9.ffn_up.weight", "ffn_up", "iq2_xxs", "IQ2_XS", 1000, 1.0)

    assert cliff.caution
    assert "cliff" in cliff.reason
    assert cliff.transition_gain < 1.0
    assert poison.caution and "poison" in poison.reason


def test_proposal_flags_hostile_role_as_caution():
    scorer = ProposalScorer(_profile_with_risks())

    hostile = scorer.score("blk.9.attn_gate.weight", "attn_gate", "q3_K", "q4_K", 1000, 1.0)
    assert hostile.caution
    assert "upgrade-hostile" in hostile.reason

    # roles without fitted evidence are NOT hostile by default
    unknown = scorer.score("blk.9.ffn_up.weight", "ffn_up", "q3_K", "q4_K", 1000, 1.0)
    assert not unknown.caution


def test_proposal_falls_back_to_role_level_and_default_c():
    scorer = ProposalScorer(_profile_with_risks())

    unmeasured = scorer.score("blk.9.ffn_up.weight", "ffn_up", "q3_K", "q4_K", 1000, 1.0)
    assert unmeasured.c_role == pytest.approx(1.0)  # default for unknown role
    assert unmeasured.transition_gain == pytest.approx(1.0)
    assert not unmeasured.caution

    with pytest.raises(ValueError, match="delta_bytes"):
        scorer.score("t", "ffn_up", "q3_K", "q4_K", 0, 1.0)


def test_apply_role_corrections_reweights_utility():
    """v0.2 plan-time re-weighting: I* = I x C_role; unknown roles stay neutral."""
    from dataclasses import replace as _replace

    from fit_gguf.candidates import CandidateSet, UpgradeCandidate
    from fit_gguf.pipeline import _apply_role_corrections

    base = UpgradeCandidate(
        tensor="blk.0.a", from_qtype="iq3_s", to_qtype="iq4_xs",
        delta_bytes=100, importance=1.0, raw_importance=1.0,
        expected_gain=0.5, utility_per_byte=0.005, profiled=True,
        block=0, role="attn_v",
    )
    cand = base
    other = _replace(base, tensor="blk.0.b", role="ffn_gate", utility_per_byte=0.010)
    cs = CandidateSet(candidates=(cand, other), rejected=(), lower_size_bytes=10, upper_size_bytes=20)

    out = _apply_role_corrections(cs, {"attn_v": 1.2646, "ffn_gate": 0.8412})
    assert out.candidates[0].utility_per_byte == pytest.approx(0.005 * 1.2646)
    assert out.candidates[1].utility_per_byte == pytest.approx(0.010 * 0.8412)
    # sizes and rejected list pass through untouched
    assert out.lower_size_bytes == 10 and out.upper_size_bytes == 20
    assert out.rejected == ()

    # roles outside the profile stay neutral (1.0)
    out2 = _apply_role_corrections(cs, {"attn_gate": 0.5})
    assert out2.candidates[0].utility_per_byte == pytest.approx(0.005)
    assert out2.candidates[1].utility_per_byte == pytest.approx(0.010)
