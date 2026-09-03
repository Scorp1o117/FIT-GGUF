"""Tests for M6b band-conditional Refine correction (fit + resolve + apply)."""

import pytest

from fit_gguf.candidates import CandidateSet, UpgradeCandidate
from fit_gguf.pipeline import _apply_refine_corrections, _apply_role_corrections, _tensor_block
from fit_gguf.refine import (
    ProfileFitError,
    fit_band_cells,
    fit_profile,
    load_profile,
    save_profile,
    validate_profile,
)

from test_refine import _probe, _record, _write_dataset

from fit_gguf.refine import load_refine_dataset


def _candidate(tensor: str, role: str, utility: float = 1.0) -> UpgradeCandidate:
    return UpgradeCandidate(
        tensor=tensor,
        from_qtype="q3_k",
        to_qtype="q4_k",
        delta_bytes=1000,
        importance=0.5,
        raw_importance=0.5,
        expected_gain=utility,
        utility_per_byte=utility,
        profiled=True,
        block=None,
        role=role,
    )


def _band_priors():
    return {
        "dataset_id": "refine-dataset-v1",
        "role_layer_transition_cells": {
            "cell_deltas": {
                "attn_qkv:0:q3_K->q6_K": -16.708,
                "ffn_down:0:q3_K->q2_K": 10.5007,
                "ffn_down:0:q3_K->q6_K": -5.3397,
                "ffn_down:3:q4_K->q3_K": 5.8301,
            },
            "n_layers": 65,
        },
    }


def _chain_step(chain_id, step_index, step_label, delta, actions):
    return _record(
        record_type="chain_step",
        chain_id=chain_id,
        step_index=step_index,
        step_label=step_label,
        suite={"suite_id": "dev"},
        split="dev",
        actions=actions,
        post_state={"macro_kl": 0.1},
        delta_macro_kl_pct=delta,
    )


def _late_chain():
    return _chain_step(
        "qwen38-27b-networkcal",
        4,
        "nc-v4",
        -3.7448,
        [
            {
                "role": "ssm_out",
                "layer_band": "late(blk.23+)",
                "src_qtype": "q3_K",
                "dst_qtype": "q4_K",
                "tensor_count": 30,
                "note": "k=30 is the natural count of ssm_out tensors at blk>=23",
            }
        ],
    )


def _multi_action_chain():
    return _chain_step(
        "qwen38-27b-networkcal",
        1,
        "nc-v1",
        -5.6004,
        [
            {"role": "token_embd", "layer_band": None, "src_qtype": "q4_K", "dst_qtype": "q3_K"},
            {"role": "ffn_down", "layer_band": "late(blk.48+)", "src_qtype": "q3_K", "dst_qtype": "q4_K"},
        ],
    )


def _band_dataset(tmp_path):
    return load_refine_dataset(
        _write_dataset(
            tmp_path,
            chains=[_late_chain(), _multi_action_chain()],
            probes=[_probe()],
            priors=_band_priors(),
        )
    )


def test_tensor_block_parses_layer_index():
    assert _tensor_block("blk.24.ssm_out.weight") == 24
    assert _tensor_block("blk.0.attn_qkv.weight") == 0
    assert _tensor_block("token_embd.weight") is None
    assert _tensor_block("output.weight") is None
    assert _tensor_block("blk.64.nextn.eh_proj.weight") == 64


def test_fit_band_cells_separates_upgrade_and_protection(tmp_path):
    band = fit_band_cells(_band_dataset(tmp_path))
    by_id = {cell["cell_id"]: cell for cell in band["cells"]}

    # upgrade cells only (dst BPW > src BPW); downgrade cells -> protection
    assert set(by_id) == {
        "attn_qkv:0:q3_K->q6_K",
        "ffn_down:0:q3_K->q6_K",
        "qwen38-27b-networkcal:nc-v4:ssm_out",
    }
    # c mapping: clamp(1 - delta/20)
    assert by_id["attn_qkv:0:q3_K->q6_K"]["c"] == pytest.approx(1.5)  # clamped from 1.835
    assert by_id["attn_qkv:0:q3_K->q6_K"]["min_block"] == 0
    assert by_id["attn_qkv:0:q3_K->q6_K"]["max_block"] == 15
    assert by_id["ffn_down:0:q3_K->q6_K"]["c"] == pytest.approx(1.2670, abs=1e-3)
    # chain cell keeps the block predicate, not the bucket
    chain_cell = by_id["qwen38-27b-networkcal:nc-v4:ssm_out"]
    assert (chain_cell["min_block"], chain_cell["max_block"]) == (23, 64)
    assert chain_cell["c"] == pytest.approx(1.1872, abs=1e-3)
    assert chain_cell["tensor_count"] == 30

    protection = {cell["cell_id"] for cell in band["protection_cells"]}
    assert protection == {"ffn_down:0:q3_K->q2_K", "ffn_down:3:q4_K->q3_K"}

    skipped_reasons = {item["reason"] for item in band["skipped_evidence"]}
    assert any("multi-action" in reason for reason in skipped_reasons)


def test_fit_profile_without_band_cells_is_frozen_v0_form(tmp_path):
    profile = fit_profile(
        _band_dataset(tmp_path), "dev0", scope={}, with_band_cells=False
    )
    assert "band_correction" not in profile
    assert profile["calibration_status"]["c_role_form"] == "proposal-calibration-bootstrap-v0"


def test_resolve_narrowest_cell_and_role_fallback(tmp_path):
    dataset = _band_dataset(tmp_path)
    profile = fit_profile(dataset, "band-test", scope={})
    role_correction = profile["role_correction"]
    role_correction["ssm_out"] = 0.8332
    role_correction["ffn_down"] = 1.2403

    # natural validation case: late-ssm_out flips from role-hostile to cell C
    c, cell_id = _resolve(profile, "ssm_out", 22)
    assert c == pytest.approx(0.8332) and cell_id is None  # below blk.23
    c, cell_id = _resolve(profile, "ssm_out", 23)
    assert c == pytest.approx(1.1872, abs=1e-3) and "nc-v4" in cell_id
    c, cell_id = _resolve(profile, "ssm_out", 63)
    assert c == pytest.approx(1.1872, abs=1e-3)

    # anchor cell covers band 0 only; band 1 takes the role fallback
    c, cell_id = _resolve(profile, "attn_qkv", 0)
    assert c == pytest.approx(1.5) and "attn_qkv:0" in cell_id
    c, cell_id = _resolve(profile, "attn_qkv", 16)
    assert c == pytest.approx(role_correction.get("attn_qkv", 1.0)) and cell_id is None

    # ffn_down band 0 has an upgrade cell despite the q2_K cliff in the same band
    c, cell_id = _resolve(profile, "ffn_down", 7)
    assert c == pytest.approx(1.2670, abs=1e-3)

    # unknown role / non-block tensors fall back to neutral
    assert _resolve(profile, "unknown_role", 10) == (1.0, None)
    assert _resolve(profile, "ssm_out", None) == (0.8332, None)


def _resolve(profile, role, block):
    from fit_gguf.refine.profile import resolve_band_correction

    return resolve_band_correction(profile, role, block)


def test_apply_refine_corrections_usage_counts(tmp_path):
    profile = fit_profile(_band_dataset(tmp_path), "band-test", scope={})
    profile["role_correction"]["ssm_out"] = 0.8332
    candidates = (
        _candidate("blk.24.ssm_out.weight", "ssm_out", 2.0),
        _candidate("blk.10.ssm_out.weight", "ssm_out", 2.0),
        _candidate("blk.30.ffn_down.weight", "ffn_down", 3.0),
        _candidate("token_embd.weight", "token_embd", 5.0),
    )
    candidate_set = CandidateSet(
        candidates=candidates, rejected=(), lower_size_bytes=1, upper_size_bytes=2
    )
    reweighted, usage = _apply_refine_corrections(candidate_set, profile)
    utilities = {c.tensor: c.utility_per_byte for c in reweighted.candidates}
    # late ssm_out upgraded by the chain cell; early ssm_out keeps role C
    assert utilities["blk.24.ssm_out.weight"] == pytest.approx(2.0 * 1.1872, abs=1e-3)
    assert utilities["blk.10.ssm_out.weight"] == pytest.approx(2.0 * 0.8332, abs=1e-3)
    # ffn_down blk.30 has no cell in the synthetic dataset -> role fallback
    assert usage == {"qwen38-27b-networkcal:nc-v4:ssm_out": 1}


def test_apply_role_only_profile_matches_role_path(tmp_path):
    profile = fit_profile(
        _band_dataset(tmp_path), "role-only", scope={}, with_band_cells=False
    )
    candidates = (
        _candidate("blk.24.ssm_out.weight", "ssm_out", 2.0),
        _candidate("blk.10.ffn_down.weight", "ffn_down", 3.0),
    )
    candidate_set = CandidateSet(
        candidates=candidates, rejected=(), lower_size_bytes=1, upper_size_bytes=2
    )
    reweighted, usage = _apply_refine_corrections(candidate_set, profile)
    assert usage == {}
    role_path = _apply_role_corrections(candidate_set, dict(profile["role_correction"]))
    assert reweighted.candidates == role_path.candidates


def test_validate_rejects_bad_band_cells(tmp_path):
    profile = fit_profile(_band_dataset(tmp_path), "band-test", scope={})
    good = dict(profile)
    validate_profile(good)
    save_profile(good, tmp_path / "good.json")
    assert load_profile(tmp_path / "good.json")["profile_id"] == "band-test"

    for mutation in (
        {"c": 2.0},
        {"c": 0.1},
        {"min_block": 20, "max_block": 10},
        {"cell_id": None},
    ):
        bad = {**profile, "band_correction": {**profile["band_correction"],
                                              "cells": [{**profile["band_correction"]["cells"][0], **mutation}]}}
        with pytest.raises(ProfileFitError):
            validate_profile(bad)


def test_band_cells_survive_profile_roundtrip(tmp_path):
    dataset = _band_dataset(tmp_path)
    profile = fit_profile(dataset, "band-rt", scope={})
    path = save_profile(profile, tmp_path / "rt.json")
    loaded = load_profile(path)
    assert loaded["band_correction"]["cells"] == profile["band_correction"]["cells"]
    c, cell_id = _resolve(loaded, "ssm_out", 40)
    assert c == pytest.approx(1.1872, abs=1e-3)


def test_unknown_qtype_cells_are_skipped_not_fatal(tmp_path):
    priors = _band_priors()
    priors["role_layer_transition_cells"]["cell_deltas"]["weird:0:mystery->q6_K"] = -3.0
    root = load_refine_dataset(
        _write_dataset(tmp_path, chains=[_late_chain()], probes=[_probe()], priors=priors)
    )
    band = fit_band_cells(root)
    assert not any(cell["role"] == "weird" for cell in band["cells"])
    assert any(item["cell_id"] == "weird:0:mystery->q6_K" for item in band["skipped_evidence"])
