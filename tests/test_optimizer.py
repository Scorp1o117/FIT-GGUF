"""Tests for deterministic budget-safe greedy optimization."""

import json
from pathlib import Path

import pytest

from fit_gguf import (
    CandidateSet,
    OptimizationError,
    UpgradeCandidate,
    optimize_greedy,
    optimize_block_balanced,
    optimize_random,
    write_fit_recipe,
    write_tensor_type_file,
)


def _candidate(name: str, cost: int, utility: float) -> UpgradeCandidate:
    return UpgradeCandidate(
        tensor=name,
        from_qtype="iq3_s",
        to_qtype="iq4_xs",
        delta_bytes=cost,
        importance=1.0,
        raw_importance=1.0,
        expected_gain=utility * cost,
        utility_per_byte=utility,
        profiled=True,
        block=0,
        role="attn_q",
    )


def test_greedy_is_budget_safe_and_skips_nonfitting_items():
    candidates = CandidateSet(
        (
            _candidate("large", 70, 3.0),
            _candidate("medium", 50, 2.0),
            _candidate("small", 30, 1.0),
        ),
        (),
        lower_size_bytes=100,
        upper_size_bytes=250,
    )
    plan = optimize_greedy(180, candidates)

    assert [candidate.tensor for candidate in plan.selected] == ["large"]
    assert plan.predicted_size_bytes == 170
    assert plan.unused_bytes == 10
    assert plan.predicted_size_bytes <= plan.target_bytes


def test_greedy_is_deterministic_across_input_order():
    first = _candidate("a", 20, 1.0)
    second = _candidate("b", 20, 1.0)
    left = CandidateSet((second, first), (), 100, 140)
    right = CandidateSet((first, second), (), 100, 140)
    assert optimize_greedy(120, left).overrides == optimize_greedy(120, right).overrides
    assert optimize_greedy(120, left).overrides == (("a", "iq4_xs"),)


def test_zero_utility_unprofiled_candidate_can_fill_remaining_budget():
    positive = _candidate("profiled", 20, 1.0)
    unprofiled = UpgradeCandidate(
        tensor="token_embd.weight",
        from_qtype="iq3_s",
        to_qtype="iq4_xs",
        delta_bytes=10,
        importance=0.0,
        raw_importance=0.0,
        expected_gain=0.0,
        utility_per_byte=0.0,
        profiled=False,
        block=None,
        role="unprofiled",
    )
    plan = optimize_greedy(130, CandidateSet((unprofiled, positive), (), 100, 130))
    assert plan.overrides == (("profiled", "iq4_xs"), ("token_embd.weight", "iq4_xs"))
    assert plan.unused_bytes == 0


def test_rejects_target_below_lower():
    with pytest.raises(OptimizationError, match="below lower baseline"):
        optimize_greedy(99, CandidateSet((), (), 100, 100))


def test_recipe_writer_records_exact_budget(tmp_path: Path):
    plan = optimize_greedy(
        120,
        CandidateSet((_candidate("blk.0.attn_q.weight", 20, 1.0),), (), 100, 120),
    )
    output = tmp_path / "fit-recipe.json"
    write_fit_recipe(plan, output, lower_preset="IQ3_S", upper_preset="IQ4_XS")
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["predicted_size_bytes"] == 120
    assert payload["overrides"][0]["tensor"] == "blk.0.attn_q.weight"


def test_tensor_type_file_uses_sorted_exact_regexes(tmp_path: Path):
    plan = optimize_greedy(
        140,
        CandidateSet(
            (
                _candidate("blk.1.attn_q.weight", 20, 1.0),
                _candidate("blk.0.attn_q.weight", 20, 2.0),
            ),
            (),
            100,
            140,
        ),
    )
    output = tmp_path / "tensor-types.txt"
    write_tensor_type_file(plan, output)
    assert output.read_text(encoding="utf-8").splitlines() == [
        r"^blk\.0\.attn_q\.weight$=iq4_xs",
        r"^blk\.1\.attn_q\.weight$=iq4_xs",
    ]


def test_random_baseline_is_seeded_and_input_order_independent():
    candidates = tuple(_candidate(name, 20, utility) for name, utility in (
        ("a", 4.0), ("b", 3.0), ("c", 2.0), ("d", 1.0)
    ))
    left = CandidateSet(candidates, (), 100, 180)
    right = CandidateSet(tuple(reversed(candidates)), (), 100, 180)
    first = optimize_random(140, left, seed="m10")
    second = optimize_random(140, right, seed="m10")
    assert first.overrides == second.overrides
    assert first.predicted_size_bytes == 140
    assert first.unused_bytes == 0


def test_random_baseline_does_not_follow_utility_order():
    candidates = CandidateSet(
        tuple(_candidate(name, 20, utility) for name, utility in (
            ("highest", 4.0), ("high", 3.0), ("low", 2.0), ("lowest", 1.0)
        )),
        (),
        100,
        180,
    )
    random_plan = optimize_random(120, candidates, seed=0)
    greedy_plan = optimize_greedy(120, candidates)
    assert random_plan.overrides != greedy_plan.overrides


def test_block_balanced_spends_quota_in_each_block_group():
    candidates = CandidateSet(
        (
            _candidate("early-best", 30, 10.0),
            _candidate("early-next", 20, 9.0),
            _candidate("late-best", 30, 2.0),
            _candidate("late-next", 20, 1.0),
        ),
        (),
        100,
        200,
    )
    candidates = CandidateSet(
        tuple(
            UpgradeCandidate(
                tensor=c.tensor,
                from_qtype=c.from_qtype,
                to_qtype=c.to_qtype,
                delta_bytes=c.delta_bytes,
                importance=c.importance,
                raw_importance=c.raw_importance,
                expected_gain=c.expected_gain,
                utility_per_byte=c.utility_per_byte,
                profiled=True,
                block=0 if c.tensor.startswith("early") else 16,
                role=c.role,
            )
            for c in candidates.candidates
        ),
        (),
        100,
        200,
    )
    plan = optimize_block_balanced(160, candidates, block_span=16)
    assert {candidate.tensor for candidate in plan.selected} == {"early-best", "late-best"}
    assert plan.unused_bytes == 0


def test_block_balanced_requires_profiled_blocks():
    candidate = _candidate("unprofiled", 20, 1.0)
    candidate = UpgradeCandidate(
        tensor=candidate.tensor,
        from_qtype=candidate.from_qtype,
        to_qtype=candidate.to_qtype,
        delta_bytes=candidate.delta_bytes,
        importance=candidate.importance,
        raw_importance=candidate.raw_importance,
        expected_gain=candidate.expected_gain,
        utility_per_byte=candidate.utility_per_byte,
        profiled=False,
        block=None,
        role="unprofiled",
    )
    with pytest.raises(OptimizationError, match="requires profiled block"):
        optimize_block_balanced(120, CandidateSet((candidate,), (), 100, 120))
