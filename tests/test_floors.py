"""Precision floors: budget the candidate set cannot reach, spent on purpose.

The candidate set holds only tensors the two bracketing presets type differently.
A tensor both presets type identically is unreachable at ANY budget under ANY
policy — measured on occamy's balanced window, that excluded the shared expert,
attn_q/attn_k/attn_gate, the SSM gates and token_embd, i.e. exactly the tensors
that run for every token while a routed expert runs for ~3% of them.

These tests pin the mechanism, not the numbers: a floor raises what is below it,
leaves what is at or above it, bills its cost, and never invents a cost it cannot
compute.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from fit_gguf.floors import (
    ALWAYS_ACTIVE_FLOORS,
    _bpw,
    _tensor_bytes,
    apply_floors,
    floor_overrides,
)


@dataclass
class FakePlan:
    selected: tuple = ()
    predicted_size_bytes: int = 0
    lower_size_bytes: int = 0
    skipped_count: int = 0


def make_inputs(spec: dict[str, tuple[str, tuple[int, ...]]]):
    """``{name: (lower_preset_qtype, shape)}`` -> (lower_recipe, layout)."""
    recipe = SimpleNamespace(
        tensors=tuple(
            SimpleNamespace(name=name, dst_type=qtype) for name, (qtype, _) in spec.items()
        )
    )
    layout = SimpleNamespace(
        tensors=tuple(
            SimpleNamespace(name=name, shape=shape) for name, (_, shape) in spec.items()
        )
    )
    return recipe, layout


# 256 x 64 blocks cleanly by every block size under test (Q4_K=32, Q6_K=16...).
SHAPE = (256, 64)
FLOORS = ((r"\.ffn_gate_shexp\.weight$", "Q6_K"), (r"\.attn_q\.weight$", "Q4_K"))


def test_a_tensor_below_its_floor_is_raised():
    recipe, layout = make_inputs({"blk.0.ffn_gate_shexp.weight": ("iq3_s", SHAPE)})
    out = floor_overrides(recipe, layout, {}, FLOORS)
    assert [c.tensor for c in out] == ["blk.0.ffn_gate_shexp.weight"]
    assert out[0].to_qtype == "q6_k"
    assert out[0].from_qtype == "iq3_s"
    assert out[0].role == "always_active"
    assert out[0].block == 0


def test_a_tensor_at_or_above_its_floor_is_left_alone():
    """A floor is a minimum, never a cap."""
    for qtype in ("q6_k", "q8_0", "f16"):
        recipe, layout = make_inputs({"blk.0.ffn_gate_shexp.weight": (qtype, SHAPE)})
        assert floor_overrides(recipe, layout, {}, FLOORS) == []


def test_the_plan_choice_outranks_the_preset_it_overrides():
    """`existing` is what the optimizer picked; the preset is only the fallback."""
    recipe, layout = make_inputs({"blk.0.attn_q.weight": ("iq3_s", SHAPE)})
    # The preset would need raising...
    assert len(floor_overrides(recipe, layout, {}, FLOORS)) == 1
    # ...but the plan already raised it past the floor.
    assert floor_overrides(recipe, layout, {"blk.0.attn_q.weight": "q6_k"}, FLOORS) == []


def test_cost_is_the_aligned_delta_between_the_two_types():
    recipe, layout = make_inputs({"blk.0.attn_q.weight": ("iq3_s", SHAPE)})
    (candidate,) = floor_overrides(recipe, layout, {}, FLOORS)
    assert candidate.delta_bytes == _tensor_bytes(SHAPE, "Q4_K") - _tensor_bytes(SHAPE, "IQ3_S")
    assert candidate.delta_bytes > 0
    # Q4_K is coarser than IQ3_M, so this particular raise is a real increase.
    assert _bpw("q4_k") > _bpw("iq3_s")


def test_an_unmatched_tensor_produces_nothing():
    recipe, layout = make_inputs({"blk.0.ffn_down_exps.weight": ("iq3_s", SHAPE)})
    assert floor_overrides(recipe, layout, {}, FLOORS) == []


def test_a_tensor_absent_from_the_recipe_is_skipped():
    recipe, layout = make_inputs({"blk.0.other.weight": ("iq3_s", SHAPE)})
    layout.tensors = layout.tensors + (SimpleNamespace(name="blk.0.attn_q.weight", shape=SHAPE),)
    # The layout knows it, the recipe does not: no honest "from" type, so no raise.
    assert floor_overrides(recipe, layout, {}, FLOORS) == []


def test_an_unknowable_shape_is_left_alone_rather_than_guessed():
    recipe, layout = make_inputs({"blk.0.attn_q.weight": ("iq3_m", (255, 64))})
    # 255 is not divisible by Q4_K's block size; a guessed cost would corrupt the
    # budget the oracle loop chases.
    assert floor_overrides(recipe, layout, {}, FLOORS) == []


def test_apply_floors_bills_the_raise_into_the_plan():
    recipe, layout = make_inputs({"blk.0.ffn_gate_shexp.weight": ("iq3_s", SHAPE)})
    plan = FakePlan(selected=(), predicted_size_bytes=1000)
    out = apply_floors(plan, recipe, layout, FLOORS)
    assert len(out.selected) == 1
    assert out.predicted_size_bytes == 1000 + out.selected[0].delta_bytes
    # The original plan is untouched (dataclasses.replace, not mutation).
    assert plan.selected == () and plan.predicted_size_bytes == 1000


def test_apply_floors_keeps_the_optimizers_own_picks():
    recipe, layout = make_inputs(
        {
            "blk.0.ffn_gate_shexp.weight": ("iq3_s", SHAPE),
            "blk.0.ffn_down_exps.weight": ("iq3_s", SHAPE),
        }
    )
    keep = SimpleNamespace(tensor="blk.0.ffn_down_exps.weight", to_qtype="q4_k", delta_bytes=7)
    out = apply_floors(FakePlan(selected=(keep,), predicted_size_bytes=1000), recipe, layout, FLOORS)
    assert keep in out.selected
    assert len(out.selected) == 2


def test_an_empty_floor_table_is_a_no_op():
    recipe, layout = make_inputs({"blk.0.ffn_gate_shexp.weight": ("iq3_s", SHAPE)})
    plan = FakePlan(selected=(), predicted_size_bytes=1000)
    assert apply_floors(plan, recipe, layout, ()) is plan


def test_the_shipped_table_covers_the_always_active_groups():
    patterns = [p for p, _ in ALWAYS_ACTIVE_FLOORS]
    import re

    for name in (
        "blk.0.ffn_gate_shexp.weight",
        "blk.39.ffn_up_shexp.weight",
        "blk.7.ffn_down_shexp.weight",
        "blk.3.attn_q.weight",
        "blk.3.attn_k.weight",
        "blk.12.attn_gate.weight",
        "blk.12.ssm_alpha.weight",
        "blk.12.ssm_beta.weight",
        "blk.12.ssm_out.weight",
        "token_embd.weight",
    ):
        assert any(re.search(p, name) for p in patterns), name
    # Routed experts are sparse (8 of 256) — they are the optimizer's business,
    # not the floor table's.
    for name in ("blk.0.ffn_down_exps.weight", "blk.0.ffn_gate_exps.weight"):
        assert not any(re.search(p, name) for p in patterns), name


@pytest.mark.parametrize("qtype", ["q4_k", "q6_k", "iq3_s", "iq4_xs", "q8_0"])
def test_bpw_is_ordered_like_the_type_names(qtype):
    assert _bpw(qtype) > 0
    assert _bpw("q8_0") > _bpw("q6_k") > _bpw("q4_k")
