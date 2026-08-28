"""Tests for conservative lower-to-upper upgrade generation."""

from fit_gguf import (
    DryRunResult,
    DryRunTensorAssignment,
    GGUFSizePrediction,
    ImatrixProfile,
    ImatrixTensorProfile,
    TensorSize,
    generate_upgrade_candidates,
)


def _assignment(ordinal: int, name: str, qtype: str) -> DryRunTensorAssignment:
    return DryRunTensorAssignment(
        ordinal=ordinal,
        total_tensors=3,
        name=name,
        shape=(256, 1, 1, 1),
        src_type="bf16",
        dst_type=qtype,
        is_quantized=True,
        orig_bytes=512,
        new_bytes=1,
    )


def _profile_entry(name: str, role_relative: float) -> ImatrixTensorProfile:
    return ImatrixTensorProfile(
        name=name,
        block=0,
        role="attn_q",
        width=256,
        count_values=1,
        count_min=10,
        count_max=10,
        count_sum=10,
        mean=2.0,
        rms=2.0,
        stddev=0.0,
        minimum=2.0,
        p50=2.0,
        p95=2.0,
        p99=2.0,
        maximum=2.0,
        nonzero_fraction=1.0,
        global_relative_mean=1.0,
        role_relative_mean=role_relative,
        global_percentile=0.5,
        role_percentile=0.5,
    )


def test_generates_promotions_and_rejects_upper_preset_downgrades():
    names = ("blk.0.attn_q.weight", "blk.0.ffn_down.weight", "token_embd.weight")
    lower = DryRunResult(
        (
            _assignment(1, names[0], "iq3_s"),
            _assignment(2, names[1], "q4_K"),
            _assignment(3, names[2], "iq3_s"),
        ),
        3,
        1536,
        1,
    )
    upper = DryRunResult(
        (
            _assignment(1, names[0], "iq4_xs"),
            _assignment(2, names[1], "iq4_xs"),
            _assignment(3, names[2], "iq4_xs"),
        ),
        3,
        1536,
        1,
    )
    lower_size = GGUFSizePrediction(
        32, 330, 0, 362,
        (
            TensorSize(names[0], "iq3_s", 110, 110),
            TensorSize(names[1], "q4_k", 144, 144),
            TensorSize(names[2], "iq3_s", 110, 110),
        ),
    )
    upper_size = GGUFSizePrediction(
        32, 408, 0, 440,
        tuple(TensorSize(name, "iq4_xs", 136, 136) for name in names),
    )
    profile = ImatrixProfile(
        1, "fixture.gguf", ("fixture",), 1, 1, (_profile_entry(names[0], 2.0),)
    )

    result = generate_upgrade_candidates(lower, upper, lower_size, upper_size, profile)

    assert [candidate.tensor for candidate in result.candidates] == [
        "blk.0.attn_q.weight",
        "token_embd.weight",
    ]
    assert result.candidates[0].delta_bytes == 26
    assert result.candidates[0].importance == 2.0
    assert result.candidates[0].expected_gain > 0
    assert result.candidates[1].profiled is False
    assert result.candidates[1].expected_gain == 0
    assert len(result.rejected) == 1
    assert result.rejected[0].tensor == "blk.0.ffn_down.weight"
    assert result.rejected[0].delta_bytes == -8
    assert result.candidate_budget_bytes == 52
