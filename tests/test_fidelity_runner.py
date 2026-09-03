"""Tests for the Fidelity Search runner glue (window selection, seed loading)."""

import json
from pathlib import Path

import pytest

from fit_gguf.fidelity_runner import (
    POISON_PRESETS,
    Window,
    discover_windows,
    load_seeds,
    select_window,
)
from fit_gguf.fidelity_search import TierContract

G = 1024**3

CONTRACT = TierContract(tier="balanced", kl_anchor=0.10, same_top_floor=0.9118)

_KL_HEADER = "====== KL divergence statistics ======"


def _window(lower, upper, lower_preset="A", upper_preset="B", name="w"):
    return Window(
        analysis_path=Path(f"/tmp/{name}"),
        lower_preset=lower_preset,
        upper_preset=upper_preset,
        lower_size=lower,
        upper_size=upper,
    )


def _make_log(kl: float, top: float) -> str:
    return (
        f"{_KL_HEADER}\n"
        f"Mean KLD: {kl:.6f} ± 0.010000\n"
        f"Same top p: {top:.4f} ± 0.1000 %\n"
    )


def test_select_window_prefers_narrowest_covering_healthy_window():
    windows = [
        _window(int(9.3 * G), int(10.42 * G), "Q2_K", "IQ3_XXS", "w1"),
        _window(int(11.0 * G), int(11.57 * G), "IQ3_XS", "IQ3_S", "w2"),
        _window(int(11.24 * G), int(11.57 * G), "Q3_K_S", "IQ3_S", "w3"),  # poison lower
    ]
    window, planned = select_window(int(11.3 * G), windows)
    assert window.lower_preset == "IQ3_XS"  # poison window must never win
    assert planned == int(11.3 * G)


def test_select_window_snaps_to_nearest_healthy_edge_in_gap():
    windows = [
        _window(int(9.3 * G), int(10.42 * G), "Q2_K", "IQ3_XXS", "w1"),
        _window(int(11.0 * G), int(11.57 * G), "IQ3_XS", "IQ3_S", "w2"),
    ]
    # 10.7G lies in the coverage gap; nearest edge is 10.42G (0.28G) vs 11.0G (0.3G)
    window, planned = select_window(int(10.7 * G), windows)
    assert window.lower_preset == "Q2_K"
    assert planned == int(10.42 * G)
    # and a probe closer to the upper window snaps up instead
    window, planned = select_window(int(10.75 * G), windows)
    assert window.lower_preset == "IQ3_XS"
    assert planned == int(11.0 * G)


def test_select_window_snaps_down_when_closer():
    windows = [
        _window(int(9.3 * G), int(10.42 * G), "Q2_K", "IQ3_XXS", "w1"),
        _window(int(11.0 * G), int(11.57 * G), "IQ3_XS", "IQ3_S", "w2"),
    ]
    window, planned = select_window(int(10.5 * G), windows)
    assert window.lower_preset == "Q2_K"
    assert planned == int(10.42 * G)


def test_select_window_refuses_when_too_far_from_any_healthy_window():
    windows = [_window(int(11.0 * G), int(11.57 * G), "IQ3_XS", "IQ3_S", "w2")]
    with pytest.raises(Exception, match="768MiB"):
        select_window(int(9.0 * G), windows)


def test_poison_presets_list_matches_ruling():
    assert POISON_PRESETS == ("Q3_K_S", "IQ2_XS")


def test_discover_windows_reads_analysis_json(tmp_path):
    for name, lower, upper, lower_preset, upper_preset in (
        ("w1", int(9.3 * G), int(10.42 * G), "Q2_K", "IQ3_XXS"),
        ("w2", int(11.0 * G), int(11.57 * G), "IQ3_XS", "IQ3_S"),
    ):
        d = tmp_path / name
        d.mkdir()
        (d / "analysis.json").write_text(json.dumps({
            "presets": {
                "lower": {"name": lower_preset, "predicted_size_bytes": lower},
                "upper": {"name": upper_preset, "predicted_size_bytes": upper},
            }
        }))
    windows = discover_windows([tmp_path / "w1", tmp_path / "w2"])
    assert [w.lower_preset for w in windows] == ["Q2_K", "IQ3_XS"]
    assert windows[0].healthy and windows[1].healthy


def test_load_seeds_groups_domains_maps_sizes_and_filters(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    manifest = tmp_path / "manifest.txt"
    manifest.write_text(
        "orcarouter-GOOD  12000000000  aa\n"
        "orcarouter-Q3_K_S-art  11000000000  bb\n"
        "orcarouter-PARTIAL  10000000000  cc\n"
    )

    domains = ("wiki_test", "wiki_valid", "chinese", "code", "agent_chat")
    for domain in domains:
        (logs / f"eval-orcarouter-GOOD-{domain}.log").write_text(_make_log(0.10, 92.0))
        (logs / f"eval-orcarouter-Q3_K_S-art-{domain}.log").write_text(_make_log(0.21, 80.0))
        (logs / f"eval-orcarouter-NOSIZE-{domain}.log").write_text(_make_log(0.10, 92.0))
        if domain != "agent_chat":  # PARTIAL is missing one domain
            (logs / f"eval-orcarouter-PARTIAL-{domain}.log").write_text(_make_log(0.10, 92.0))

    manifest.write_text(
        "orcarouter-GOOD  12000000000  aa\n"
        "orcarouter-Q3_K_S-art  11000000000  bb\n"
        "orcarouter-PARTIAL  10000000000  cc\n"
    )
    seeds = load_seeds(manifest, logs, "orcarouter-")
    sizes = {seed.size_bytes for seed in seeds}
    assert sizes == {12000000000}  # poison, incomplete, and size-less points all rejected
    seed = seeds[0]
    assert seed.macro_kl == pytest.approx(0.10)
    assert seed.same_top == pytest.approx(0.92)


def test_poison_taint_propagates_via_provenance(tmp_path):
    """Fail-closed seed admissibility: a clean-named artifact whose producing
    window used a poison preset is auto-tainted (no manual exclude needed)."""
    logs = tmp_path / "logs"
    logs.mkdir()
    manifest = tmp_path / "manifest.txt"
    manifest.write_text(
        "orcarouter-CLEAN  12000000000  aa\n"
        "orcarouter-TAINTED  12412366048  bb\n"
    )
    from fit_gguf.fidelity_runner import _provenance_view

    for domain in ("wiki_test", "wiki_valid", "chinese", "code", "agent_chat"):
        (logs / f"eval-orcarouter-CLEAN-{domain}.log").write_text(_make_log(0.10, 92.0))
        (logs / f"eval-orcarouter-TAINTED-{domain}.log").write_text(_make_log(0.159, 89.0))

    provenance = tmp_path / "seed-provenance.jsonl"
    provenance.write_text(
        json.dumps({"name": "orcarouter-TAINTED", "size_bytes": 12412366048,
                    "window_lower_preset": "Q3_K_S", "window_upper_preset": "IQ3_S"}) + "\n"
    )
    assert _provenance_view(provenance)[0] == {"orcarouter-TAINTED"}

    seeds = load_seeds(manifest, logs, "orcarouter-", provenance_path=provenance)
    assert {s.size_bytes for s in seeds} == {12000000000}

    # without provenance the seed would be admitted (backfill responsibility);
    # point provenance_path at a location with no records to express that
    seeds_no_prov = load_seeds(
        manifest, logs, "orcarouter-", provenance_path=tmp_path / "none.jsonl"
    )
    assert {s.size_bytes for s in seeds_no_prov} == {12000000000, 12412366048}
