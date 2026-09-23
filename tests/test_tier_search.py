"""fit tier-search — the per-tier minimum-PASS solve.

A tier's product is the smallest artifact that passes its KL anchor and same-top
floor, and that artifact generally lies BETWEEN two ladder presets.  These tests
pin the bracket arithmetic, the minimum-PASS semantics of the report, the merge
(not overwrite) behaviour of the report file, and the reseal that keeps a
post-emit curve append from leaving stale digests in the bundle.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from fit_gguf import calibration as cal
from fit_gguf.calibrate import CalibrateConfig, stage_emit
from fit_gguf.registry import validate_bundle
from fit_gguf.tier_search import (
    _fresh_tag,
    _summarise,
    bracketing_pair,
    passes,
    preset_sizes,
    reseal_bundle,
    tiers_from_evaluation,
    tiers_from_profile,
    write_report,
)

CONTRACT, CONTRACT_SHA = cal.load_contract(None)
PRESETS = set(CONTRACT["ladder_standard_presets"])

# The occamy-1.0-abliterated quality window, verbatim.  IQ4_XS sits just over the
# 0.05 anchor; the minimum PASS (18.00 GiB) is 1.71 GiB under the Q4_K_M preset
# that the window probe heuristic used to fall back to.
OCCAMY_QUALITY = [
    {"point_id": "probe-quality-1", "size_bytes": 18541000000, "macro_kl": 0.0555, "same_top": 0.9124},
    {"point_id": "IQ4_XS", "size_bytes": 18728777728, "macro_kl": 0.0503, "same_top": 0.9157},
    {"point_id": "tier-quality-s2", "size_bytes": 19328964608, "macro_kl": 0.0463, "same_top": 0.9225},
    {"point_id": "Q4_K_M", "size_bytes": 21166757888, "macro_kl": 0.0471, "same_top": 0.9235},
]
QUALITY_TIERS = {"quality": {"anchor": 0.05, "floor": 0.9127}}


def _obs(point, kl, top, size):
    return {
        "point_id": point,
        "macro_kl": kl,
        "same_top": top,
        "size_bytes": size,
        "artifact_sha256": hashlib.sha256(point.encode()).hexdigest(),
    }


def _evaluation(tiers):
    return {
        "overall_status": "validated",
        "open_failures": [],
        "tiers": {
            name: {
                "anchor": spec["anchor"],
                "window": [spec["anchor"] * 0.85, spec["anchor"] * 1.15],
                "sample_count": 3,
                "floor": spec["floor"],
                "floor_method": "empirical_p5",
                "witness": None,
                "validation_status": "validated",
                "samples": [],
            }
            for name, spec in tiers.items()
        },
    }


def _make_bundle(tmp_path: Path, observations: list[dict], tiers: dict) -> Path:
    """A real bundle, emitted by the production emitter rather than hand-written."""
    source = tmp_path / "source.gguf"
    source.write_bytes(b"not-a-real-gguf")
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("corpus\n", encoding="utf-8")
    cfg = CalibrateConfig(
        source=source,
        imatrix_corpus=corpus,
        runtime_dir=tmp_path / "runtime",
        eval_data_dir=tmp_path / "eval",
        out_dir=tmp_path / "bundle",
        model_id="test-model",
    )
    # write_seed_material wants an artifact digest per point; the curve literals
    # above carry only the metrics the search reads.
    emitted = [
        {**obs, "artifact_sha256": obs.get("artifact_sha256") or hashlib.sha256(
            obs["point_id"].encode()).hexdigest()}
        for obs in observations
    ]
    stage_emit(
        cfg, CONTRACT, CONTRACT_SHA, _evaluation(tiers), emitted,
        tmp_path / "refs", {}, [], [],
    )
    return cfg.out_dir


def _append_to_curve(bundle: Path, points: list[dict]) -> None:
    """Append points the way a standalone search does — with artifact digests.

    ``write_seed_material`` reads ``artifact_sha256`` off every curve point, so a
    point without one is not a point the bundle can carry.
    """
    with (bundle / "curve-points.jsonl").open("a", encoding="utf-8") as handle:
        for obs in points:
            row = {
                **obs,
                "artifact_sha256": obs.get("artifact_sha256")
                or hashlib.sha256(obs["point_id"].encode()).hexdigest(),
            }
            handle.write(json.dumps(row, sort_keys=True) + "\n")


# ----------------------------------------------------------------- predicates
def test_passes_gates_on_kl_alone():
    assert passes(_obs("a", 0.0463, 0.9225, 10), 0.05)
    # Inclusive: a KL exactly at the anchor is admitted.
    assert passes(_obs("a", 0.05, 0.0, 10), 0.05)
    # IQ4_XS: 0.0003 over the anchor is a FAIL however good same-top is.
    assert not passes(_obs("a", 0.0503, 0.9157, 10), 0.05)


def test_passes_ignores_the_same_top_floor():
    """v0.3: same-top never decides a verdict.

    occamy's mini floor sat 0.0003 above an 11.27 GiB point that clears the 0.20
    anchor.  A dual gate hides that artifact outright.
    """
    assert passes(_obs("probe-mini-3", 0.1892, 0.8353, 12_100_000_000), 0.20)
    # And a point that clears KL with a catastrophic same-top still passes.
    assert passes(_obs("degenerate", 0.19, 0.0, 12_100_000_000), 0.20)


def test_preset_sizes_keeps_only_presets_in_size_order():
    curve = OCCAMY_QUALITY + [_obs("some-probe", 0.9, 0.1, 5)]
    assert preset_sizes(curve, PRESETS) == [
        (18728777728, "IQ4_XS"),
        (21166757888, "Q4_K_M"),
    ]


def test_bracketing_pair_brackets_a_size_between_presets():
    sizes = preset_sizes(OCCAMY_QUALITY, PRESETS)
    # The real target: midway between IQ4_XS (17.44) and the 18.00 GiB PASS.
    assert bracketing_pair(sizes, 19_028_871_168) == ("IQ4_XS", "Q4_K_M")


def test_bracketing_pair_is_none_outside_the_ladder():
    sizes = preset_sizes(OCCAMY_QUALITY, PRESETS)
    assert bracketing_pair(sizes, 1) is None
    assert bracketing_pair(sizes, 10**13) is None


# --------------------------------------------------------------------- tiers
def test_tiers_from_evaluation_matches_the_shape_the_search_expects():
    tiers = tiers_from_evaluation(_evaluation(QUALITY_TIERS))
    assert tiers == {"quality": {"anchor": 0.05, "floor": 0.9127}}


def test_tiers_from_profile_reads_back_what_stage_emit_wrote(tmp_path):
    bundle = _make_bundle(tmp_path, OCCAMY_QUALITY, QUALITY_TIERS)
    assert tiers_from_profile(bundle) == tiers_from_evaluation(
        _evaluation(QUALITY_TIERS)
    )


# ---------------------------------------------------------------------- tags
def test_fresh_tag_skips_tags_that_already_exist(tmp_path):
    probes = tmp_path / "probes"
    probes.mkdir()
    assert _fresh_tag(tmp_path, "quality") == "tier-quality-s1"
    for used in ("tier-quality-s1", "tier-quality-s2"):
        (probes / f"{used}-plan.json").write_text("{}", encoding="utf-8")
    # A standalone re-run must not overwrite an earlier pass's plan or recipe.
    assert _fresh_tag(tmp_path, "quality") == "tier-quality-s3"


# ------------------------------------------------------------------- report
def test_summarise_picks_the_smallest_pass_not_the_smallest_preset():
    row = _summarise("quality", OCCAMY_QUALITY, 0.05, 0.9127, PRESETS)
    assert row["status"] == "ok"
    assert row["best_point"] == "tier-quality-s2"
    assert row["best_bytes"] == 19328964608
    # IQ4_XS is smaller but fails the anchor, so the baseline is Q4_K_M.
    assert row["preset_baseline"] == "Q4_K_M"
    assert row["saved_bytes"] == 21166757888 - 19328964608


def test_summarise_reports_no_pass_when_every_point_fails():
    row = _summarise("quality", OCCAMY_QUALITY[:2], 0.05, 0.9127, PRESETS)
    assert row["status"] == "no_pass"


def test_summarise_carries_the_floor_as_a_reference_only():
    row = _summarise("quality", OCCAMY_QUALITY, 0.05, 0.9127, PRESETS)
    assert row["same_top_reference"] == 0.9127
    assert row["clears_floor"] is True
    # A floor the winner misses is reported, not enforced.
    missed = _summarise("quality", OCCAMY_QUALITY, 0.05, 0.99, PRESETS)
    assert missed["status"] == "ok"
    assert missed["best_point"] == "tier-quality-s2"
    assert missed["clears_floor"] is False


def test_summarise_without_a_reference_floor_reports_none():
    row = _summarise("quality", OCCAMY_QUALITY, 0.05, None, PRESETS)
    assert row["best_point"] == "tier-quality-s2"
    assert row["same_top_reference"] is None
    assert row["clears_floor"] is None


def test_write_report_merges_instead_of_overwriting(tmp_path):
    write_report(tmp_path, {"quality": {"status": "ok"}, "balanced": {"status": "ok"}})
    write_report(tmp_path, {"quality": {"status": "no_pass"}})
    merged = json.loads((tmp_path / "tier-search-report.json").read_text())
    assert merged == {"quality": {"status": "no_pass"}, "balanced": {"status": "ok"}}


def test_write_report_survives_a_corrupt_existing_file(tmp_path):
    (tmp_path / "tier-search-report.json").write_text("{not json", encoding="utf-8")
    write_report(tmp_path, {"quality": {"status": "ok"}})
    assert json.loads((tmp_path / "tier-search-report.json").read_text()) == {
        "quality": {"status": "ok"}
    }


# -------------------------------------------------------------------- reseal
def test_reseal_refreshes_derived_digests_and_keeps_the_bundle_admissible(tmp_path):
    bundle = _make_bundle(tmp_path, OCCAMY_QUALITY[:2], QUALITY_TIERS)
    record_before = json.loads((bundle / "calibration-record.json").read_text())
    assert record_before["process"]["curve_points"] == 2

    # A standalone search appends to the curve AFTER stage_emit hashed it.
    _append_to_curve(bundle, OCCAMY_QUALITY[2:])
    # Without the reseal the record's curve digest is stale — the exact hole the
    # standalone path would otherwise leave behind.
    assert json.loads((bundle / "calibration-record.json").read_text()) == record_before

    reseal_bundle(bundle, CONTRACT)

    record = json.loads((bundle / "calibration-record.json").read_text())
    assert record["process"]["curve_points"] == 4
    assert record["artifacts"]["curve_points_sha256"] == hashlib.sha256(
        (bundle / "curve-points.jsonl").read_bytes()
    ).hexdigest()
    assert record["artifacts"]["curve_points_sha256"] != record_before["artifacts"][
        "curve_points_sha256"
    ]
    # Rewriting the record moves the registry entry's pin; all three digests or
    # validate_bundle fails on the first stale one.
    assert validate_bundle(bundle)["admissible"] is True


def test_reseal_does_not_touch_the_guard_profile_or_the_floors(tmp_path):
    bundle = _make_bundle(tmp_path, OCCAMY_QUALITY[:2], QUALITY_TIERS)
    guard_before = (bundle / "guard-profile.yaml").read_bytes()
    record_before = json.loads((bundle / "calibration-record.json").read_text())
    reseal_bundle(bundle, CONTRACT)
    # Floors were frozen before the search ran: the artifacts it selected must
    # not feed back into the floor they are judged against.
    assert (bundle / "guard-profile.yaml").read_bytes() == guard_before
    assert json.loads((bundle / "calibration-record.json").read_text())[
        "derivation"
    ] == record_before["derivation"]


def test_reseal_sha256sums_excludes_itself(tmp_path):
    bundle = _make_bundle(tmp_path, OCCAMY_QUALITY[:2], QUALITY_TIERS)
    reseal_bundle(bundle, CONTRACT)
    names = [
        line.split("  ", 1)[1]
        for line in (bundle / "SHA256SUMS").read_text().splitlines()
        if line.strip()
    ]
    assert "SHA256SUMS" not in names
    assert "curve-points.jsonl" in names
    assert "registry-entry.json" in names


def test_reseal_refreshes_the_ladder_seed_material(tmp_path):
    """The standalone path never runs stage_emit, so the seed files must be re-derived.

    Without this the bundle would ship a size manifest and provenance sidecar
    describing a curve that no longer exists, and `fit fidelity-search` could not
    reuse a single point the search just paid a five-domain eval for.
    """
    bundle = _make_bundle(tmp_path, OCCAMY_QUALITY[:2], QUALITY_TIERS)
    assert "tier-quality-s2" not in (bundle / "state-artifact-manifest.txt").read_text()

    _append_to_curve(bundle, OCCAMY_QUALITY[2:])
    reseal_bundle(bundle, CONTRACT)

    manifest = (bundle / "state-artifact-manifest.txt").read_text()
    assert "tier-quality-s2" in manifest
    assert "Q4_K_M" in manifest
    provenance = {
        row["name"]: row
        for row in (
            json.loads(line)
            for line in (bundle / "seed-provenance.jsonl").read_text().splitlines()
            if line.strip()
        )
    }
    # A preset names itself on both window anchors; a probe names none, which is
    # what keeps a poison preset from ever anchoring through a probe.
    assert provenance["Q4_K_M"]["window_lower_preset"] == "Q4_K_M"
    assert provenance["tier-quality-s2"]["window_lower_preset"] == ""
    assert provenance["tier-quality-s2"]["window_upper_preset"] == ""
    assert validate_bundle(bundle)["admissible"] is True


@pytest.mark.parametrize("tier", ["quality", "balanced", "compact", "mini"])
def test_every_contract_tier_is_addressable_by_the_search(tier):
    anchor = float(CONTRACT["tier_kl_anchors"][tier])
    assert _summarise(
        tier, [_obs("x", anchor, 1.0, 10)], anchor, None, PRESETS
    )["status"] == "ok"


# The occamy mini window, verbatim.  probe-mini-3 clears the 0.20 anchor but
# misses the calibrated floor by 0.0003; the retired v0.2 dual gate therefore
# selected the 11.38 GiB point instead of this 11.27 GiB one.
OCCAMY_MINI = [
    {"point_id": "IQ2_M", "size_bytes": 11660847104, "macro_kl": 0.2532, "same_top": 0.8093},
    {"point_id": "probe-mini-3", "size_bytes": 12100000000, "macro_kl": 0.1892, "same_top": 0.8353},
    {"point_id": "tier-mini-s1", "size_bytes": 12218765312, "macro_kl": 0.1853, "same_top": 0.8366},
    {"point_id": "IQ3_XXS", "size_bytes": 13623758848, "macro_kl": 0.1582, "same_top": 0.8499},
]


def test_mini_kl_only_gate_finds_the_artifact_a_dual_gate_hides():
    row = _summarise("mini", OCCAMY_MINI, 0.20, 0.8356, PRESETS)
    assert row["best_point"] == "probe-mini-3"
    assert row["best_bytes"] == 12100000000
    # The floor is reported and honestly marked as missed — but not enforced.
    assert row["clears_floor"] is False
    assert row["preset_baseline"] == "IQ3_XXS"
    assert row["saved_bytes"] == 13623758848 - 12100000000


# ------------------------------------------------------------ restart mode
def test_prior_points_are_the_tiers_own_history():
    """A restart must drop the tier's probes and keep the ladder.

    Ladder presets are the bounds no policy change can move — dropping them too
    would leave nothing to bracket against.
    """
    from fit_gguf.tier_search import prior_points_for_tier

    curve = [
        {"point_id": "IQ3_M"},
        {"point_id": "IQ4_XS"},
        {"point_id": "probe-balanced-1", "probe": {"tier": "balanced"}},
        {"point_id": "tier-balanced-s3"},
        {"point_id": "floor-balanced-14.9G"},
        {"point_id": "probe-compact-1", "probe": {"tier": "compact"}},
        {"point_id": "tier-compact-s1"},
    ]
    assert prior_points_for_tier(curve, "balanced") == {
        "probe-balanced-1", "tier-balanced-s3", "floor-balanced-14.9G",
    }
    assert prior_points_for_tier(curve, "compact") == {
        "probe-compact-1", "tier-compact-s1",
    }
    assert prior_points_for_tier(curve, "mini") == set()


def test_a_restart_keeps_the_ladder_and_drops_only_that_tier():
    from fit_gguf.tier_search import prior_points_for_tier

    curve = [
        {"point_id": "IQ3_M"},
        {"point_id": "tier-balanced-s3"},
        {"point_id": "tier-compact-s1"},
    ]
    prior = prior_points_for_tier(curve, "balanced")
    pool = [o for o in curve if o["point_id"] not in prior]
    assert [o["point_id"] for o in pool] == ["IQ3_M", "tier-compact-s1"]


# --------------------------------------------------------- floor regime
def test_point_floor_regime_reads_every_shape_the_ledger_uses(tmp_path):
    """One fact, three historical spellings, and a refusal to guess.

    Sweeps write the flag at the point level, search probes carry it in the
    probe block, pre-floor plans have no field at all but do have a plan record,
    and anything else is unknown — which is not the same as False.
    """
    from fit_gguf.tier_search import point_floor_regime

    bundle = tmp_path / "bundle"
    (bundle / "probes").mkdir(parents=True)
    (bundle / "probes" / "tier-x-s1-plan.json").write_text(
        json.dumps({"record": {"always_active_floors": True}}), encoding="utf-8"
    )
    (bundle / "probes" / "tier-x-s2-plan.json").write_text(
        json.dumps({"record": {"target_bytes": 1}}), encoding="utf-8"
    )

    assert point_floor_regime(bundle, {"point_id": "sweep-14G", "always_active_floors": True}) is True
    assert point_floor_regime(bundle, {"point_id": "p", "probe": {"always_active_floors": False}}) is False
    assert point_floor_regime(bundle, {"point_id": "tier-x-s1"}) is True
    assert point_floor_regime(bundle, {"point_id": "tier-x-s2"}) is False
    assert point_floor_regime(bundle, {"point_id": "gone"}) is None


def test_reusable_points_keeps_the_ladder_and_drops_a_foreign_regime(tmp_path):
    """The regression: a stale point must not outrank a smaller honest one.

    On occamy the pre-floor ``tier-balanced-s3`` (14.89 GiB, KL 0.0999) sat just
    under the floor artifact at 14.90 GiB — so the search, which ranks by size,
    picked it and then read its own bracket as "under tolerance". The 13.80 GiB
    floor artifact that actually wins the tier was never looked at.
    """
    from fit_gguf.tier_search import reusable_points

    bundle = tmp_path / "bundle"
    (bundle / "probes").mkdir(parents=True)
    # A pre-floor plan record: the field does not exist yet, which is exactly
    # what makes it readable as False rather than as unknown.
    (bundle / "probes" / "tier-balanced-s3-plan.json").write_text(
        json.dumps({"record": {"target_bytes": 1}}), encoding="utf-8"
    )
    curve = [
        {"point_id": "IQ3_M", "size_bytes": 15_440_519_168, "macro_kl": 0.1080, "same_top": 0.8802},
        {"point_id": "tier-balanced-s3", "size_bytes": 15_984_623_648, "macro_kl": 0.0999,
         "same_top": 0.8833},
        {"point_id": "floor5-13.8G", "size_bytes": 14_815_721_472, "macro_kl": 0.0959,
         "same_top": 0.8883, "always_active_floors": True},
        {"point_id": "lost-provenance-14G", "size_bytes": 15_000_000_000, "macro_kl": 0.08,
         "same_top": 0.89},
    ]
    pool, dropped = reusable_points(
        curve, bundle, presets={"IQ3_M"}, always_active_floors=True
    )
    assert [o["point_id"] for o in pool] == ["IQ3_M", "floor5-13.8G"]
    assert sorted(dropped) == ["lost-provenance-14G", "tier-balanced-s3"]
    assert _summarise("balanced", pool, 0.10, None, {"IQ3_M"})["best_point"] == "floor5-13.8G"

    # With floors off the populations swap, and the pre-floor point is the one
    # that is comparable — still never both at once.
    pool_off, _ = reusable_points(curve, bundle, presets={"IQ3_M"}, always_active_floors=False)
    assert [o["point_id"] for o in pool_off] == ["IQ3_M", "tier-balanced-s3"]
