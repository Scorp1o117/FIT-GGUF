"""G2 exact-size regression tests.

Two layers:
1. Unit rules of the fixed model: build-10666 drops trailing singleton dims
   when re-emitting output tensor infos, and quantize.imatrix.file is the
   imatrix path string verbatim (artifact size is path-length dependent).
2. Byte-exact regression against the six real Ling amendment-2 artifacts,
   snapshotted into tests/fixtures/g2_ling_amendment2.json by
   scripts/extract_g2_ling_fixture.py (layout + effective recipes from the
   retained oracle dry-run logs + recorded actual sizes).
"""

from dataclasses import replace
from pathlib import Path
import json

import pytest

from fit_gguf.gguf import (
    GGUFField,
    GGUFLayout,
    GGUFTensor,
    ImatrixProvenance,
    QuantizationMetadata,
    _canonical_shape,
    _encoded_field_size,
    _output_tensor_info_bytes,
    predict_quantized_size,
)
from fit_gguf.models import DryRunResult, DryRunTensorAssignment

REPO = Path(__file__).resolve().parent.parent
FIXTURE = REPO / "tests/fixtures/g2_ling_amendment2.json"


def _layout(tensors: tuple[GGUFTensor, ...]) -> GGUFLayout:
    return GGUFLayout(
        version=3,
        alignment=32,
        metadata_count=0,
        fields=(),
        tensors=tensors,
        tensor_info_bytes=0,
        raw_metadata_bytes=0,
        data_offset=0,
    )


def _info_size(name: str, dims: int) -> int:
    # u64 name length + name + u32 n_dims + n_dims * u64 ne + u32 type + u64 offset
    return 8 + len(name.encode("utf-8")) + 4 + 8 * dims + 4 + 8


def test_output_tensor_info_trims_trailing_singletons():
    layout = _layout(
        (
            GGUFTensor("blk.0.conv.weight", (4, 1, 2048, 1), 0, 0),
            GGUFTensor("output_norm.weight", (3,), 0, 0),
            GGUFTensor("scalar.weight", (1, 1), 0, 0),
            GGUFTensor("row.weight", (2, 1, 1), 0, 0),
        )
    )
    assert _output_tensor_info_bytes(layout) == (
        _info_size("blk.0.conv.weight", 3)
        + _info_size("output_norm.weight", 1)
        + _info_size("scalar.weight", 1)
        + _info_size("row.weight", 1)
    )


def test_imatrix_path_length_dependence():
    # quantize.imatrix.file embeds the invocation string verbatim, so N extra
    # path bytes shift the UNALIGNED metadata by exactly N (33 -> 99 chars
    # encodes 74 -> 140, the L-A1 ground truth). The 32B section alignment may
    # absorb part of that at the boundary; the real absorbed case (-18) is
    # pinned by test_la1_component_ground_truth.
    assert _encoded_field_size("quantize.imatrix.file", 8, "a" * 33) == 74
    assert _encoded_field_size("quantize.imatrix.file", 8, "a" * 99) == 140
    assert (
        _encoded_field_size("quantize.imatrix.file", 8, "a" * 99)
        - _encoded_field_size("quantize.imatrix.file", 8, "a" * 33)
        == 66
    )


def _snapshot_layout(snapshot: dict) -> GGUFLayout:
    return GGUFLayout(
        version=snapshot["version"],
        alignment=snapshot["alignment"],
        metadata_count=snapshot["metadata_count"],
        fields=tuple(
            GGUFField(f["key"], f["value_type"], f["encoded_size"])
            for f in snapshot["fields"]
        ),
        tensors=tuple(
            GGUFTensor(t["name"], tuple(t["shape"]), t["type_id"], 0)
            for t in snapshot["tensors"]
        ),
        tensor_info_bytes=snapshot["source_tensor_info_bytes"],
        raw_metadata_bytes=snapshot["source_raw_metadata_bytes"],
        data_offset=snapshot["source_data_offset"],
    )


def _recipe_from_compact(entries: list) -> DryRunResult:
    tensors = tuple(
        DryRunTensorAssignment(
            ordinal=index + 1,
            total_tensors=len(entries),
            name=name,
            shape=tuple(shape),
            src_type=src,
            dst_type=dst,
            is_quantized=bool(is_quantized),
            orig_bytes=0,
            new_bytes=0,
        )
        for index, (name, shape, src, dst, is_quantized) in enumerate(entries)
    )
    return DryRunResult(tensors, len(entries), 0, 0)


def _metadata(analysis: dict, file_string: str) -> QuantizationMetadata:
    static = analysis["metadata"]["imatrix"]
    return QuantizationMetadata(
        file_type=analysis["metadata"]["file_type"],
        quantization_version=analysis["metadata"]["quantization_version"],
        imatrix=ImatrixProvenance(
            file=file_string,
            dataset=static["dataset"],
            entries_count=static["entries_count"],
            chunks_count=static["chunks_count"],
        ),
    )


def test_trailing_singleton_normalization_rule():
    # The rule itself, pinned independently of the Ling fixture.
    cases = {
        (4, 1, 2048, 1): 3,
        (4, 1, 2048): 3,
        (4, 2048): 2,
        (7, 1, 1): 1,
        (1, 1): 1,
        (1,): 1,
    }
    for shape, dims in cases.items():
        assert len(_canonical_shape(shape)) == dims


def test_kv_string_truncation_boundary_bytes():
    # llama.cpp strncpy rule: exactly KV_OVERRIDE_STRING_MAX_BYTES UTF-8
    # BYTES survive (val_str[127] = '\0'), even when the cut lands
    # mid-codepoint. decode-ignore would silently drop 1-3 bytes and break
    # the exact-size model.
    from fit_gguf.pipeline import _truncate_kv_string

    for n, expected in ((126, 126), (127, 127), (128, 127), (300, 127)):
        assert len(_truncate_kv_string("a" * n).encode("utf-8")) == expected
    mid_codepoint_cut = "a" * 126 + "测"  # byte 127 splits the 3-byte char
    truncated = _truncate_kv_string(mid_codepoint_cut)
    assert len(truncated.encode("utf-8", errors="surrogateescape")) == 127
    boundary_aligned = "a" * 124 + "测"  # 124 + 3 = 127, stays valid UTF-8
    assert (
        _truncate_kv_string(boundary_aligned).encode("utf-8")
        == boundary_aligned.encode("utf-8")
    )
    # KV model bills byte length: 33 (key) + 8 (len u64) + 127 bytes
    assert (
        _encoded_field_size(
            "quantize.imatrix.file", 8, _truncate_kv_string("a" * 300)
        )
        == 168
    )


def test_utf8_path_byte_length_not_char_length():
    # GGUF string size uses UTF-8 serialized bytes, never Python len(str).
    path = "/imatrix/测试/校准.dat"
    assert len(path.encode("utf-8")) == 26
    assert len(path) == 18  # the wrong basis
    assert _encoded_field_size("quantize.imatrix.file", 8, path) == 33 + 8 + 26
    from fit_gguf.pipeline import _truncate_kv_string

    long_path = path * 6  # 156 bytes, crosses the 127-byte boundary
    assert (
        _encoded_field_size(
            "quantize.imatrix.file", 8, _truncate_kv_string(long_path)
        )
        == 33 + 8 + 127
    )


@pytest.fixture(scope="module")
def g2():
    if not FIXTURE.is_file():
        pytest.skip("G2 fixture not extracted; run scripts/extract_g2_ling_fixture.py")
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_ling_amendment2_byte_exact(g2):
    layout = _snapshot_layout(g2["layout"])
    predictions = {}
    for plan in g2["plans"]:
        analysis = g2["analyses"][plan["analysis_sha256"]]
        metadata = _metadata(analysis, g2["quantize_imatrix_arg"])
        recipe = _recipe_from_compact(plan["effective_recipe"])
        prediction = predict_quantized_size(layout, recipe, metadata)
        assert prediction.total_bytes == plan["actual_size_bytes"], plan["name"]
        predictions[plan["name"]] = prediction
    assert len(predictions) == 6


def test_la1_component_ground_truth(g2):
    layout = _snapshot_layout(g2["layout"])
    truth = g2["la1_ground_truth"]
    plan = next(p for p in g2["plans"] if p["name"] == "L-A1")
    analysis = g2["analyses"][plan["analysis_sha256"]]
    metadata = _metadata(analysis, g2["quantize_imatrix_arg"])
    prediction = predict_quantized_size(
        layout, _recipe_from_compact(plan["effective_recipe"]), metadata
    )
    assert _output_tensor_info_bytes(layout) == truth["output_tensor_info_bytes"]
    assert (
        truth["source_tensor_info_bytes"] - truth["output_tensor_info_bytes"] == 432
    )
    assert prediction.metadata_bytes == truth["metadata_bytes"]
    assert prediction.tensor_payload_bytes == truth["tensor_payload_bytes"]
    assert prediction.total_bytes == truth["actual_total_bytes"]
    assert _encoded_field_size(
        "quantize.imatrix.file", 8, g2["quantize_imatrix_arg"]
    ) == truth["imatrix_file_encoded_quantize"]
    assert len(analysis["imatrix_arg"]) == 99
    assert _encoded_field_size(
        "quantize.imatrix.file", 8, analysis["imatrix_arg"]
    ) == truth["imatrix_file_encoded_analysis"]


def test_old_predictor_bias_was_constant_480(g2):
    for plan in g2["plans"]:
        assert plan["predicted_size_bytes"] - plan["actual_size_bytes"] == 480, plan[
            "name"
        ]
