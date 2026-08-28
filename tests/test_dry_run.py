"""Unit tests for dry-run parsing and validation."""

from decimal import Decimal
import pytest

from fit_gguf import (
    DryRunParseError,
    DryRunResult,
    DryRunTensorAssignment,
    mib_to_bytes,
    parse_dry_run,
)


def test_mib_to_bytes_deterministic():
    """Verify deterministic conversion from MiB to bytes using Decimal ROUND_HALF_UP."""
    assert mib_to_bytes("1.00") == 1048576
    assert mib_to_bytes("1485.00") == 1557135360
    assert mib_to_bytes("780.00") == 817889280
    assert mib_to_bytes("0.020") == 20972
    assert mib_to_bytes("0.010") == 10486
    assert mib_to_bytes("0.005") == 5243
    assert mib_to_bytes("0.001") == 1049
    assert mib_to_bytes(Decimal("1485.00")) == 1557135360


def test_parse_dry_run_standard_qwen_mix():
    """Verify parsing a realistic dry-run log with Qwen3.5-style tensor names."""
    raw_log = """
main: quantizing 'source-bf16.gguf' to 'output.gguf' as IQ3_S
llama_model_loader: loaded meta data with 32 key-value pairs and 12 tensors
llama_model_quantize_internal: [   1/  12] token_embd.weight                    - [  5120, 152064,      1,      1], type =   BF16, size =  1485.00 MiB ->   780.00 MiB (Q4_K)
llama_model_quantize_internal: [   2/  12] blk.0.attn_norm.weight               - [  5120,      1,      1,      1], type =    F32, size =    0.020 MiB
llama_model_quantize_internal: [   3/  12] blk.0.attn_q.weight                  - [  5120,   5120,      1,      1], type =   BF16, size =    50.00 MiB ->    21.50 MiB (IQ3_S)
llama_model_quantize_internal: [   4/  12] blk.0.attn_k.weight                  - [  5120,   1024,      1,      1], type =   BF16, size =    10.00 MiB ->     4.30 MiB (IQ3_S)
llama_model_quantize_internal: [   5/  12] blk.0.attn_v.weight                  - [  5120,   1024,      1,      1], type =   BF16, size =    10.00 MiB ->     5.50 MiB (Q4_K)
llama_model_quantize_internal: [   6/  12] blk.0.attn_output.weight             - [  5120,   5120,      1,      1], type =   BF16, size =    50.00 MiB ->    21.50 MiB (IQ3_S)
llama_model_quantize_internal: [   7/  12] blk.0.ffn_gate_exps.weight           - [  5120,  14336,      8,      1], type =   BF16, size =  1120.00 MiB ->   481.60 MiB (IQ3_S)
llama_model_quantize_internal: [   8/  12] blk.0.ffn_down_exps.weight           - [ 14336,   5120,      8,      1], type =   BF16, size =  1120.00 MiB ->   481.60 MiB (IQ3_S)
llama_model_quantize_internal: [   9/  12] blk.0.ssm_alpha.weight               - [  5120,      1,      1,      1], type =   BF16, size =    0.010 MiB ->     0.005 MiB (IQ3_S)
llama_model_quantize_internal: [  10/  12] blk.0.ssm_beta.weight                - [  5120,      1,      1,      1], type =   BF16, size =    0.010 MiB ->     0.005 MiB (IQ3_S)
llama_model_quantize_internal: [  11/  12] blk.0.ssm_conv1d.weight              - [     4,   5120,      1,      1], type =    F32, size =    0.078 MiB
llama_model_quantize_internal: [  12/  12] output.weight                        - [  5120, 152064,      1,      1], type =   BF16, size =  1485.00 MiB ->   780.00 MiB (Q6_K)
llama_model_quantize_internal: model size  =  5330.12 MiB (16.00 BPW)
llama_model_quantize_internal: quant size  =  2576.11 MiB ( 7.73 BPW)
llama_model_quantize_internal: WARNING: dry run completed successfully
"""
    result = parse_dry_run(raw_log)

    assert isinstance(result, DryRunResult)
    assert len(result) == 12
    assert result.total_tensors == 12
    assert result.quantized_count == 10
    assert result.unchanged_count == 2
    assert result.reported_orig_bpw == Decimal("16.00")
    assert result.reported_new_bpw == Decimal("7.73")

    # Verify tensor 1 (quantized)
    t1 = result[0]
    assert t1.ordinal == 1
    assert t1.total_tensors == 12
    assert t1.name == "token_embd.weight"
    assert t1.shape == (5120, 152064, 1, 1)
    assert t1.src_type == "BF16"
    assert t1.dst_type == "Q4_K"
    assert t1.is_quantized is True
    assert t1.quantized is True
    assert t1.orig_bytes == 1557135360
    assert t1.new_bytes == 817889280
    assert t1.orig_bytes_display == 1557135360
    assert t1.new_bytes_display == 817889280

    # Verify tensor 2 (unchanged)
    t2 = result[1]
    assert t2.ordinal == 2
    assert t2.name == "blk.0.attn_norm.weight"
    assert t2.shape == (5120, 1, 1, 1)
    assert t2.src_type == "F32"
    assert t2.dst_type == "F32"
    assert t2.is_quantized is False
    assert t2.orig_bytes == 20972
    assert t2.new_bytes == 20972

    # Verify MoE expert tensor
    t7 = result.tensor_map["blk.0.ffn_gate_exps.weight"]
    assert t7.ordinal == 7
    assert t7.shape == (5120, 14336, 8, 1)
    assert t7.src_type == "BF16"
    assert t7.dst_type == "IQ3_S"
    assert t7.is_quantized is True

    # Verify Qwen SSM alpha tensor
    t9 = result.tensor_map["blk.0.ssm_alpha.weight"]
    assert t9.ordinal == 9
    assert t9.src_type == "BF16"
    assert t9.dst_type == "IQ3_S"
    assert t9.is_quantized is True
    assert t9.orig_bytes == 10486
    assert t9.new_bytes == 5243

    # Verify SSM conv1d unchanged tensor
    t11 = result.tensor_map["blk.0.ssm_conv1d.weight"]
    assert t11.ordinal == 11
    assert t11.src_type == "F32"
    assert t11.dst_type == "F32"
    assert t11.is_quantized is False
    assert t11.orig_bytes == 81789
    assert t11.new_bytes == 81789

    # Verify output tensor
    t12 = result.tensor_map["output.weight"]
    assert t12.ordinal == 12
    assert t12.dst_type == "Q6_K"


def test_parse_dry_run_all_unchanged():
    """Verify parsing when all tensors are unchanged (e.g. F32 model)."""
    raw_log = """
[ 1/ 2] blk.0.attn_norm.weight - [ 5120, 1, 1, 1], type = F32, size = 0.020 MiB
[ 2/ 2] blk.0.ffn_norm.weight  - [ 5120, 1, 1, 1], type = F32, size = 0.020 MiB
model size = 0.04 MiB (32.00 BPW)
quant size = 0.04 MiB (32.00 BPW)
"""
    result = parse_dry_run(raw_log)
    assert len(result) == 2
    assert result.quantized_count == 0
    assert result.unchanged_count == 2
    assert result[0].dst_type == "F32"
    assert result[1].dst_type == "F32"


def test_parse_dry_run_out_of_order_sorted_strictly_by_ordinal():
    """Verify that results are ordered strictly by ordinal regardless of input order."""
    raw_lines = [
        "model size = 300.00 MiB",
        "[ 3/ 3] blk.0.ffn_up.weight - [ 4096, 4096, 1, 1], type = BF16, size = 100.00 MiB -> 40.00 MiB (IQ3_S)",
        "[ 1/ 3] blk.0.attn_q.weight - [ 4096, 4096, 1, 1], type = BF16, size = 100.00 MiB -> 40.00 MiB (IQ3_S)",
        "quant size = 120.00 MiB",
        "[ 2/ 3] blk.0.attn_k.weight - [ 4096, 4096, 1, 1], type = BF16, size = 100.00 MiB -> 40.00 MiB (IQ3_S)",
    ]
    result = parse_dry_run(raw_lines)
    assert [t.ordinal for t in result.tensors] == [1, 2, 3]
    assert [t.name for t in result.tensors] == [
        "blk.0.attn_q.weight",
        "blk.0.attn_k.weight",
        "blk.0.ffn_up.weight",
    ]


def test_parse_dry_run_varied_prefixes_and_ansi():
    """Verify parsing with different llama.cpp log prefixes and ANSI escape sequences."""
    raw_log = """
\x1b[32m[INFO]\x1b[0m [ 1/ 3] blk.0.attn_q.weight - [ 1024, 1024, 1, 1], type = BF16, size = 2.00 MiB -> 1.00 MiB (IQ3_S)
\x1b[35mI\x1b[0m [ 2/ 3] blk.0.attn_k.weight - [ 1024, 1024, 1, 1], type = BF16, size = 2.00 MiB -> 1.00 MiB (IQ3_S)
0.01.234.567 I [ 3/ 3] blk.0.attn_v.weight - [ 1024, 1024, 1, 1], type = BF16, size = 2.00 MiB -> 1.00 MiB (IQ3_S)
llama_model_quantize_internal: model size = 6.00 MiB
llama_model_quantize_internal: quant size = 3.00 MiB
"""
    result = parse_dry_run(raw_log)
    assert len(result) == 3
    assert result.total_tensors == 3
    assert result.reported_orig_bytes == 6 * 1048576
    assert result.reported_new_bytes == 3 * 1048576


def test_reject_duplicate_ordinal():
    """Verify rejection of duplicate tensor ordinals."""
    raw_log = """
[ 1/ 2] blk.0.attn_q.weight - [ 1024, 1024, 1, 1], type = BF16, size = 2.00 MiB -> 1.00 MiB (IQ3_S)
[ 1/ 2] blk.0.attn_k.weight - [ 1024, 1024, 1, 1], type = BF16, size = 2.00 MiB -> 1.00 MiB (IQ3_S)
model size = 4.00 MiB
quant size = 2.00 MiB
"""
    with pytest.raises(DryRunParseError, match="Duplicate tensor ordinal: 1"):
        parse_dry_run(raw_log)


def test_reject_duplicate_tensor_name():
    """Verify rejection of duplicate tensor names."""
    raw_log = """
[ 1/ 2] blk.0.attn_q.weight - [ 1024, 1024, 1, 1], type = BF16, size = 2.00 MiB -> 1.00 MiB (IQ3_S)
[ 2/ 2] blk.0.attn_q.weight - [ 1024, 1024, 1, 1], type = BF16, size = 2.00 MiB -> 1.00 MiB (IQ3_S)
model size = 4.00 MiB
quant size = 2.00 MiB
"""
    with pytest.raises(DryRunParseError, match="Duplicate tensor name: blk.0.attn_q.weight"):
        parse_dry_run(raw_log)


def test_reject_inconsistent_total_counts():
    """Verify rejection when tensor lines report different total counts."""
    raw_log = """
[ 1/ 2] blk.0.attn_q.weight - [ 1024, 1024, 1, 1], type = BF16, size = 2.00 MiB -> 1.00 MiB (IQ3_S)
[ 2/ 3] blk.0.attn_k.weight - [ 1024, 1024, 1, 1], type = BF16, size = 2.00 MiB -> 1.00 MiB (IQ3_S)
model size = 4.00 MiB
quant size = 2.00 MiB
"""
    with pytest.raises(DryRunParseError, match="Inconsistent total tensor count"):
        parse_dry_run(raw_log)


def test_reject_missing_ordinal():
    """Verify rejection when an ordinal in the 1..N sequence is missing."""
    raw_log = """
[ 1/ 3] blk.0.attn_q.weight - [ 1024, 1024, 1, 1], type = BF16, size = 2.00 MiB -> 1.00 MiB (IQ3_S)
[ 3/ 3] blk.0.attn_v.weight - [ 1024, 1024, 1, 1], type = BF16, size = 2.00 MiB -> 1.00 MiB (IQ3_S)
model size = 4.00 MiB
quant size = 2.00 MiB
"""
    with pytest.raises(DryRunParseError, match="Parsed 2 tensors, but expected total is 3"):
        parse_dry_run(raw_log)


def test_reject_ordinal_out_of_bounds():
    """Verify rejection when an ordinal is zero or exceeds total."""
    raw_log = """
[ 0/ 1] blk.0.attn_q.weight - [ 1024, 1024, 1, 1], type = BF16, size = 2.00 MiB -> 1.00 MiB (IQ3_S)
model size = 2.00 MiB
quant size = 1.00 MiB
"""
    with pytest.raises(DryRunParseError, match="Tensor ordinal 0 out of range"):
        parse_dry_run(raw_log)


def test_reject_malformed_candidate_line_bad_shape():
    """Verify rejection when a candidate tensor line has an invalid shape."""
    raw_log = """
[ 1/ 1] blk.0.attn_q.weight - [ invalid_shape ], type = BF16, size = 2.00 MiB -> 1.00 MiB (IQ3_S)
model size = 2.00 MiB
quant size = 1.00 MiB
"""
    with pytest.raises(DryRunParseError, match="Malformed candidate tensor line"):
        parse_dry_run(raw_log)


def test_reject_shape_non_positive_dimension():
    """Verify rejection when a tensor shape has a non-positive dimension."""
    raw_log = """
[ 1/ 1] blk.0.attn_q.weight - [ 1024, 0, 1, 1], type = BF16, size = 2.00 MiB -> 1.00 MiB (IQ3_S)
model size = 2.00 MiB
quant size = 1.00 MiB
"""
    with pytest.raises(DryRunParseError, match="Invalid tensor shape"):
        parse_dry_run(raw_log)


def test_reject_malformed_candidate_line_bad_size():
    """Verify rejection when a candidate tensor line has a broken size arrow."""
    raw_log = """
[ 1/ 1] blk.0.attn_q.weight - [ 1024, 1024, 1, 1], type = BF16, size = 2.00 MiB -> INVALID (IQ3_S)
model size = 2.00 MiB
quant size = 1.00 MiB
"""
    with pytest.raises(DryRunParseError, match="Malformed tensor size part"):
        parse_dry_run(raw_log)


def test_reject_summary_inconsistency_beyond_rounding():
    """Verify rejection when reported summary is inconsistent beyond display rounding tolerance."""
    raw_log = """
[ 1/ 2] blk.0.attn_q.weight - [ 1024, 1024, 1, 1], type = BF16, size = 100.00 MiB -> 40.00 MiB (IQ3_S)
[ 2/ 2] blk.0.attn_k.weight - [ 1024, 1024, 1, 1], type = BF16, size = 100.00 MiB -> 40.00 MiB (IQ3_S)
model size = 200.00 MiB
quant size = 500.00 MiB
"""
    with pytest.raises(DryRunParseError, match="Reported quant size .* exceeding display tolerance"):
        parse_dry_run(raw_log)


def test_reject_missing_summary():
    """Verify rejection when a summary line is missing."""
    raw_log = """
[ 1/ 1] blk.0.attn_q.weight - [ 1024, 1024, 1, 1], type = BF16, size = 100.00 MiB -> 40.00 MiB (IQ3_S)
quant size = 40.00 MiB
"""
    with pytest.raises(DryRunParseError, match="Missing model size summary line"):
        parse_dry_run(raw_log)


def test_reject_conflicting_summaries():
    """Verify rejection when conflicting model size summaries exist."""
    raw_log = """
[ 1/ 1] blk.0.attn_q.weight - [ 1024, 1024, 1, 1], type = BF16, size = 100.00 MiB -> 40.00 MiB (IQ3_S)
model size = 100.00 MiB
model size = 200.00 MiB
quant size = 40.00 MiB
"""
    with pytest.raises(DryRunParseError, match="Conflicting model size summaries"):
        parse_dry_run(raw_log)


def test_reject_conflicting_summary_bpw():
    """Verify duplicate summaries cannot silently replace their BPW value."""
    raw_log = """
[ 1/ 1] blk.0.attn_q.weight - [ 1024, 1024, 1, 1], type = BF16, size = 2.00 MiB -> 1.00 MiB (IQ3_S)
model size = 2.00 MiB (16.00 BPW)
model size = 2.00 MiB (15.00 BPW)
quant size = 1.00 MiB (8.00 BPW)
"""
    with pytest.raises(DryRunParseError, match="Conflicting model BPW summaries"):
        parse_dry_run(raw_log)


def test_reject_empty_output():
    """Verify rejection when output contains no tensor lines."""
    with pytest.raises(DryRunParseError, match="No dry-run tensor lines found"):
        parse_dry_run("")
