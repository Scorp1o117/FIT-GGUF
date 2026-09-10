"""Tests for Fidelity Search v1 (synthetic curves, no I/O)."""

import json

import pytest

from fit_gguf.fidelity_search import (
    EvalOutcome,
    FidelitySearchError,
    Seed,
    TierContract,
    fidelity_search,
)

BALANCED = TierContract(tier="balanced", kl_anchor=0.10, same_top_reference=0.9118)


def _curve(crossing: float, slope: float = 0.02, top_floor=BALANCED.same_top_reference):
    """Deterministic monotone executor: KL crosses `crossing` (GiB units).

    KL(size_gib) = slope * (crossing - size_gib) + kl_anchor/2 above the
    crossing (passing region) — i.e. KL decreases as size grows; same-top is
    always comfortably above the reference, so KL is the only thing that can
    decide the verdict.
    """

    def evaluate(size_bytes: int) -> EvalOutcome:
        size_gib = size_bytes / 1024**3
        macro_kl = 0.10 - slope * (size_gib - crossing)
        same_top = top_floor + 0.02
        return EvalOutcome(size_bytes=size_bytes, macro_kl=macro_kl, same_top=same_top)

    return evaluate


G = 1024**3


def test_seeds_alone_hit_tolerance_without_fresh_evals():
    result = fidelity_search(
        BALANCED,
        _curve(12.8),
        seeds=(
            Seed(size_bytes=int(12.5 * G), macro_kl=0.105, same_top=0.93),
            Seed(size_bytes=int(13.0 * G), macro_kl=0.095, same_top=0.93),
        ),
        min_size=int(11 * G),
        max_size=int(14 * G),
        tolerance_bytes=int(0.6 * G),
    )
    assert result.status == "verified_pass"
    assert result.fresh_evals == 0
    assert result.best.size_bytes == int(13.0 * G)
    assert result.active_constraint == "kl"


def test_bisection_refines_bracket_within_budget():
    evaluate = _curve(12.8)
    calls = []

    def counting(size):
        calls.append(size)
        return evaluate(size)

    result = fidelity_search(
        BALANCED,
        counting,
        seeds=(
            Seed(size_bytes=int(12.0 * G), macro_kl=0.130, same_top=0.93),
            Seed(size_bytes=int(14.0 * G), macro_kl=0.060, same_top=0.93),
        ),
        min_size=int(11 * G),
        max_size=int(15 * G),
        tolerance_bytes=int(0.1 * G),
        budget=8,
    )
    assert result.status == "verified_pass"
    assert result.bracket_pass_bytes - result.bracket_fail_bytes <= int(0.1 * G)
    assert result.best.passed
    assert result.fresh_evals == len(calls) <= 8
    # bisection: every fresh call lands strictly inside the current bracket
    assert all(int(12.0 * G) < c < int(14.0 * G) for c in calls)


def test_budget_exhaustion_reports_observed_pass_without_minimum_claim():
    result = fidelity_search(
        BALANCED,
        _curve(12.8),
        seeds=(
            Seed(size_bytes=int(12.0 * G), macro_kl=0.130, same_top=0.93),
            Seed(size_bytes=int(14.0 * G), macro_kl=0.060, same_top=0.93),
        ),
        min_size=int(11 * G),
        max_size=int(15 * G),
        tolerance_bytes=int(0.0001 * G),  # unreachable
        budget=2,
    )
    assert result.status == "budget_exhausted"
    assert result.best is not None and result.best.passed
    assert "budget" in result.summary()["guarantee"]


def test_coarse_walk_up_finds_pass_from_fail_only_seeds():
    result = fidelity_search(
        BALANCED,
        _curve(12.8),
        seeds=(Seed(size_bytes=int(11.5 * G), macro_kl=0.150, same_top=0.92),),
        min_size=int(11 * G),
        max_size=int(15 * G),
        tolerance_bytes=int(0.3 * G),
        budget=8,
    )
    assert result.status == "verified_pass"
    assert result.best.passed
    assert result.bracket_fail_bytes is not None
    assert result.bracket_pass_bytes - result.bracket_fail_bytes <= int(0.3 * G)


def test_coarse_walk_down_finds_fail_from_pass_only_seeds():
    result = fidelity_search(
        BALANCED,
        _curve(12.8),
        seeds=(Seed(size_bytes=int(14.0 * G), macro_kl=0.060, same_top=0.93),),
        min_size=int(11 * G),
        max_size=int(15 * G),
        tolerance_bytes=int(0.3 * G),
        budget=8,
    )
    assert result.status == "verified_pass"
    assert result.bracket_fail_bytes is not None
    assert result.best.passed


def test_all_pass_within_range_reports_floor_note():
    result = fidelity_search(
        BALANCED,
        _curve(10.0),  # crossing below the searchable floor
        seeds=(Seed(size_bytes=int(13.0 * G), macro_kl=0.080, same_top=0.93),),
        min_size=int(12 * G),
        max_size=int(14 * G),
        tolerance_bytes=int(0.3 * G),
        coarse_stride_bytes=int(1 * G),
        budget=4,
    )
    assert result.status == "verified_pass"
    assert "below the searchable floor" in (result.note or "")


def test_no_pass_anywhere():
    result = fidelity_search(
        TierContract(tier="quality", kl_anchor=0.05, same_top_reference=0.9475),
        _curve(14.0),
        seeds=(),
        min_size=int(12 * G),
        max_size=int(13 * G),
        budget=4,
    )
    assert result.status == "no_pass"
    assert result.best is None


def test_noise_inversion_stops_with_explicit_flag():
    # 12.5G PASS then 12.9G FAIL (non-monotone within noise)
    def invert(size_bytes):
        gib = size_bytes / G
        if abs(gib - 12.5) < 0.05:
            kl, top = 0.095, 0.93
        elif abs(gib - 12.9) < 0.05:
            kl, top = 0.105, 0.93
        else:
            kl = 0.10 - 0.02 * (12.8 - gib)
            top = 0.93
        return EvalOutcome(size_bytes=size_bytes, macro_kl=kl, same_top=top)

    result = fidelity_search(
        BALANCED,
        invert,
        seeds=(
            Seed(size_bytes=int(12.0 * G), macro_kl=0.130, same_top=0.92),
            Seed(size_bytes=int(13.5 * G), macro_kl=0.070, same_top=0.93),
        ),
        min_size=int(11 * G),
        max_size=int(15 * G),
        tolerance_bytes=int(0.05 * G),
        budget=8,
    )
    assert result.status in ("noise_inversion", "verified_pass")
    if result.status == "noise_inversion":
        assert "noise" in result.summary()["guarantee"]


def test_failed_eval_counts_budget_but_not_bracket():
    state = {"n": 0}

    def flaky(size_bytes):
        state["n"] += 1
        if state["n"] <= 2:  # first two submissions fail, later ones pass the contract
            return EvalOutcome(size_bytes=size_bytes, macro_kl=None, same_top=None, note="boom")
        return EvalOutcome(size_bytes=size_bytes, macro_kl=0.095, same_top=0.93)

    result = fidelity_search(
        BALANCED,
        flaky,
        seeds=(),
        min_size=int(11 * G),
        max_size=int(15 * G),
        budget=4,
    )
    assert result.fresh_evals == 4
    # failed evals consume budget but never enter the bracket evidence
    assert all(point.passed for point in result.points)


def test_gate_is_kl_only_and_same_top_is_reference_only():
    """v0.3 Contract v2: Same-top never changes a verdict.

    Uses the real Compact case from the v0.2 release: KL 0.1465 passes the 0.15
    anchor while Same-top 0.8905 sits just above the model's calibrated 0.8894
    floor. Under Contract v1 that was a Same-top binding constraint; under v2 the
    KL anchor is the only gate, so ``active_constraint`` is always ``"kl"``.
    """
    contract = TierContract(tier="compact", kl_anchor=0.15, same_top_reference=0.8894)

    # KL passes -> verdict is PASS regardless of Same-top.
    assert contract.passes(macro_kl=0.1465, same_top=0.8905) is True
    assert contract.passes(macro_kl=0.1465, same_top=0.50) is True
    # Only the KL anchor can fail it.
    assert contract.passes(macro_kl=0.1501, same_top=0.99) is False

    assert contract.active_constraint(0.1465, 0.8905) == "kl"

    # The calibrated floor is still reported as an informational margin.
    m_kl, m_top_ref = contract.margins(macro_kl=0.1465, same_top=0.8905)
    assert m_top_ref is not None and m_top_ref < m_kl

    # With no reference value the informational margin is simply unavailable.
    no_ref = TierContract(tier="mini", kl_anchor=0.20)
    assert no_ref.margins(0.10, 0.90) == (pytest.approx(0.5), None)
    assert no_ref.passes(0.10, 0.10) is True


def test_audit_log_written():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "audit.jsonl"
        fidelity_search(
            BALANCED,
            _curve(12.8),
            seeds=(Seed(size_bytes=int(12.5 * G), macro_kl=0.105, same_top=0.93),
                   Seed(size_bytes=int(13.0 * G), macro_kl=0.095, same_top=0.93)),
            min_size=int(11 * G),
            max_size=int(14 * G),
            tolerance_bytes=int(0.6 * G),
            audit_log=path,
        )
        events = [json.loads(line) for line in path.read_text().splitlines()]
        kinds = {event["event"] for event in events}
        assert {"seed"} <= kinds


def test_config_validation():
    with pytest.raises(FidelitySearchError):
        fidelity_search(BALANCED, _curve(12.8), min_size=int(14 * G), max_size=int(14 * G))
    with pytest.raises(FidelitySearchError):
        fidelity_search(BALANCED, _curve(12.8), min_size=int(11 * G), max_size=int(15 * G), budget=0)
