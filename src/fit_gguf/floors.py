"""Precision floors for tensors a byte budget must not be allowed to decide.

WHY THIS EXISTS
---------------
FIT's candidate set is generated from the two presets that bracket a tier: a
tensor becomes a candidate only when its qtype DIFFERS between them
(``generate_upgrade_candidates`` admits "positive-size lower-to-upper
transitions"). That is a clean definition and it has a blind spot: when
llama.cpp's counter-based rules assign the *same* type to a tensor in both
presets, the transition is zero-sized, the tensor never becomes a candidate, and
**no plan can ever touch it** — at any budget, under any policy.

Measured on occamy-1.0-abliterated, balanced tier (window IQ3_M..Q3_K_M):

    candidate set : 76 tensors
    absent        : ffn_{gate,up}_shexp, attn_{q,k,gate}, ssm_{alpha,beta,out},
                    token_embd, output

Those absences are not neutral. The shared expert (``ffn_*_shexp``) runs for
**every token** while a routed expert runs for roughly 8 of 256; attention and
the SSM gates run in every layer. Leaving them at the base preset while spending
the budget on routed experts — which is what the greedy optimizer does, because
that is what it can see — produced a tier that lost to APEX I-Compact at the same
size (14.89 GiB @ KL 0.0999 vs 15.40 GiB @ 0.0700), with the difference visible
tensor by tensor: FIT had ``ffn_gate_shexp=IQ3_S`` where APEX had ``Q6_K``, and
``attn_q``/``ssm_alpha`` at ``IQ3_S`` where APEX had ``Q4_K``.

WHAT A FLOOR IS
---------------
A pattern and a minimum qtype. After the optimizer has spent the budget, any
tensor below its floor is raised to it and the raise is billed as an ordinary
override — so the oracle loop measures the real size and, if the total now
overshoots, re-selects with a smaller budget and the floors applied again. The
budget is not increased; it is *redirected* from what the optimizer could see to
what it could not.

These are architecture rules, not per-model tuning: a shared expert, an attention
projection and an SSM gate are always-active in any model that has them.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import replace
from functools import reduce
from operator import mul

from fit_gguf.candidates import GGML_TYPE_TRAITS, UpgradeCandidate
from fit_gguf.gguf import GGUF_DEFAULT_ALIGNMENT, GGUFLayout

# (pattern, minimum qtype). Matched against the tensor name with `re.search`.
#
# Two of these are lifted from a published, hand-audited recipe rather than from
# our own search: IsValorum's APEX-I-MiniPlus-V2.1 keeps the DeltaNet recurrence
# scales at F32 and the attention gates at Q8_0 across all 30 hybrid layers, and
# both are cheap enough to be free — `ssm_alpha` is 0.04 MiB per layer and
# `attn_gate` 4.5 MiB, so the pair costs about 126 MiB on the whole model. The
# candidate set cannot reach either on its own when both bracketing presets agree
# on the tensor, which is exactly the blindness this table exists for.
ALWAYS_ACTIVE_FLOORS: tuple[tuple[str, str], ...] = (
    # The shared expert: every token, while a routed expert sees ~3% of them.
    # APEX's own recipes put Q6_K here and it is the single largest measured gap.
    (r"\.ffn_gate_shexp\.weight$", "Q6_K"),
    (r"\.ffn_up_shexp\.weight$", "Q6_K"),
    (r"\.ffn_down_shexp\.weight$", "Q6_K"),
    # Per-layer projections that llama.cpp's counter rules leave at the base
    # preset because both bracketing presets agree on them. attn_qkv / attn_v /
    # attn_output are deliberately absent: the native heuristic already lifts
    # those one step, so a floor there would only re-spend budget.
    (r"\.attn_q\.weight$", "Q4_K"),
    (r"\.attn_k\.weight$", "Q4_K"),
    # The recurrence decay is a per-channel scalar block-quantized into groups of
    # 32 values, which is a coarse thing to do to a state that compounds over
    # 256K tokens. 0.25 MiB per layer at F32 buys exactness here.
    (r"\.ssm_alpha\.weight$", "F32"),
    (r"\.ssm_beta\.weight$", "Q4_K"),
    (r"\.ssm_out\.weight$", "Q4_K"),
    # Gating across the hybrid layers: the one place the published recipe sits
    # above us, and the cheapest two steps in the ladder.
    (r"\.attn_gate\.weight$", "Q8_0"),
    # Read once per token, at the head and the tail of every forward pass.
    (r"^token_embd\.weight$", "Q4_K"),
)


def _bpw(qtype: str) -> float:
    block_size, type_size = GGML_TYPE_TRAITS[qtype.lower()]
    return type_size * 8.0 / block_size


# Bumped when the MEANING of the table changes, not its contents. 1 = a floor was
# appended to the optimizer's selection; 2 = a floor replaces it (llama.cpp
# resolves a tensor-type file by first match, so an appended duplicate silently
# lost the tie). A curve point planned under either is not comparable to the
# other, and nothing about the point's own numbers would say so.
FLOOR_SEMANTICS_VERSION = 2


def floor_policy_id() -> str:
    """Stable identity of the floor policy: semantics version + table contents.

    The same hazard as the floor regime, one level finer.  Turning floors on is a
    policy change a boolean can record; *changing the table* or *fixing how a
    floor is applied* is a policy change a boolean cannot — the point still says
    ``always_active_floors: true`` while having been planned by different rules.
    On occamy that let four of five tiers keep a defeated-floor artifact as their
    winner after the application bug was fixed.

    A digest of the table catches edits automatically; the version catches
    changes to the semantics that leave the table text identical.  Both are
    needed, and neither is a number a human has to remember to bump for the
    common case.
    """
    payload = json.dumps(
        {"version": FLOOR_SEMANTICS_VERSION, "floors": [list(pair) for pair in ALWAYS_ACTIVE_FLOORS]},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _align(value: int, alignment: int = GGUF_DEFAULT_ALIGNMENT) -> int:
    return (value + alignment - 1) // alignment * alignment


def _tensor_bytes(shape: tuple[int, ...], qtype: str) -> int:
    """Aligned payload size of one tensor at one qtype.

    The same arithmetic ``predict_quantized_size`` applies, so a floor's cost is
    billed in the units the oracle will measure.
    """
    block_size, type_size = GGML_TYPE_TRAITS[qtype.lower()]
    ne0 = shape[0]
    if ne0 % block_size:
        raise ValueError(f"ne0={ne0} is not divisible by {qtype} block size {block_size}")
    rows = reduce(mul, shape[1:], 1)
    return _align((ne0 // block_size) * type_size * rows)


def _block_of(name: str) -> int | None:
    match = re.match(r"blk\.(\d+)\.", name)
    return int(match.group(1)) if match else None


def floor_overrides(
    lower_recipe,
    layout: GGUFLayout,
    existing: dict[str, str],
    floors: tuple[tuple[str, str], ...] = ALWAYS_ACTIVE_FLOORS,
) -> list[UpgradeCandidate]:
    """Every tensor below its floor, as an upgrade candidate.

    ``existing`` is the plan's own choice per tensor (an override, else the
    lower preset's type). A tensor already at or above its floor produces
    nothing: a floor is a minimum, never a cap.
    """
    base = {tensor.name: str(tensor.dst_type) for tensor in lower_recipe.tensors}
    current = {**base, **existing}
    shapes = {tensor.name: tensor.shape for tensor in layout.tensors}
    out: list[UpgradeCandidate] = []
    for name, shape in shapes.items():
        if name not in current:
            continue
        target_qtype = None
        for pattern, minimum in floors:
            if re.search(pattern, name):
                target_qtype = minimum
                break
        if target_qtype is None:
            continue
        now = current[name]
        try:
            if _bpw(now) >= _bpw(target_qtype):
                continue
            delta = _tensor_bytes(shape, target_qtype) - _tensor_bytes(shape, now)
        except (KeyError, ValueError):
            # A type the size table does not know, or a shape the block size
            # does not divide: leave the tensor alone rather than guess a cost.
            continue
        if delta <= 0:
            continue
        out.append(
            UpgradeCandidate(
                tensor=name,
                from_qtype=now.lower(),
                to_qtype=target_qtype.lower(),
                delta_bytes=delta,
                # Floors are not utility-ranked; they are mandatory. Zero keeps
                # them out of any ranking that might later read these fields.
                importance=0.0,
                raw_importance=0.0,
                expected_gain=0.0,
                utility_per_byte=0.0,
                profiled=False,
                block=_block_of(name),
                role="always_active",
            )
        )
    return out


def apply_floors(plan, lower_recipe, layout: GGUFLayout, floors=ALWAYS_ACTIVE_FLOORS):
    """Return ``plan`` with every below-floor tensor raised, cost billed.

    The returned plan is what the oracle dry-run measures, so the floors compete
    with the optimizer's own choices for the same budget instead of being added
    on top of it.

    A floor REPLACES the optimizer's pick for the same tensor, it does not join
    it.  ``llama-quantize`` resolves a tensor-type file by **first match**, so two
    lines for one tensor is not a merge — the earlier line silently wins and the
    floor evaporates.  That is not hypothetical: it left four of occamy's five
    shipped tiers with the shared experts at the base preset's type instead of
    the floored one, and it made a fixed-size A/B of two new floors reproduce the
    incumbent artifact byte for byte.  The cost arithmetic is unaffected by the
    replacement: ``floor_overrides`` measures each raise from the pick it
    replaces, so the delta is already the step from the plan's type to the floor.
    """
    if not floors:
        return plan
    chosen = {candidate.tensor: candidate.to_qtype for candidate in plan.selected}
    extra = floor_overrides(lower_recipe, layout, chosen, floors)
    if not extra:
        return plan
    replaced = {candidate.tensor for candidate in extra}
    kept = tuple(c for c in plan.selected if c.tensor not in replaced)
    extra_cost = sum(candidate.delta_bytes for candidate in extra)
    return replace(
        plan,
        selected=kept + tuple(extra),
        predicted_size_bytes=plan.predicted_size_bytes + extra_cost,
    )
