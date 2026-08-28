"""Tests for exact-size preset baseline selection."""

import pytest

from fit_gguf import BaselineSelectionError, PresetSize, select_baselines


PRESETS = (
    PresetSize("IQ4_XS", 15_082_507_232),
    PresetSize("IQ3_S", 12_419_328_992),
    PresetSize("IQ3_M", 12_580_875_232),
)


def test_selects_by_measured_size_not_input_or_name_order():
    target = 13 * 1024**3
    plan = select_baselines(target, PRESETS)

    assert plan.lower.name == "IQ3_M"
    assert plan.upper is not None
    assert plan.upper.name == "IQ4_XS"
    assert plan.extra_budget_bytes == target - 12_580_875_232
    assert plan.upper_gap_bytes == 15_082_507_232 - 12_580_875_232


def test_exact_preset_size_uses_that_preset_as_lower():
    plan = select_baselines(12_580_875_232, PRESETS)
    assert plan.lower.name == "IQ3_M"
    assert plan.extra_budget_bytes == 0
    assert plan.upper is not None and plan.upper.name == "IQ4_XS"


def test_target_above_largest_has_no_upper():
    plan = select_baselines(16 * 1024**3, PRESETS)
    assert plan.lower.name == "IQ4_XS"
    assert plan.upper is None
    assert plan.upper_gap_bytes is None


def test_rejects_target_below_smallest():
    with pytest.raises(BaselineSelectionError, match="smaller than the minimum preset"):
        select_baselines(1, PRESETS)


def test_rejects_ambiguous_duplicate_sizes():
    with pytest.raises(BaselineSelectionError, match="sizes must be unique"):
        select_baselines(100, (PresetSize("a", 100), PresetSize("b", 100)))


def test_rejects_duplicate_names():
    with pytest.raises(BaselineSelectionError, match="names must be unique"):
        select_baselines(200, (PresetSize("same", 100), PresetSize("same", 200)))
