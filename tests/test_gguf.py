"""Tests for the minimal GGUF layout reader and exact size predictor."""

from decimal import Decimal
from pathlib import Path
import struct

import pytest

from fit_gguf import (
    DryRunResult,
    DryRunTensorAssignment,
    GGUFError,
    ImatrixProvenance,
    QuantizationMetadata,
    predict_quantized_size,
    read_gguf_layout,
)


def _string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def _field_u32(key: str, value: int) -> bytes:
    return _string(key) + struct.pack("<II", 4, value)


def _field_string_array(key: str, values: list[str]) -> bytes:
    return (
        _string(key)
        + struct.pack("<IIQ", 9, 8, len(values))
        + b"".join(_string(value) for value in values)
    )


def _tensor(name: str, shape: tuple[int, ...], type_id: int, offset: int) -> bytes:
    return (
        _string(name)
        + struct.pack("<I", len(shape))
        + struct.pack("<" + "Q" * len(shape), *shape)
        + struct.pack("<IQ", type_id, offset)
    )


def _write_fixture(path: Path) -> None:
    fields = [
        _field_u32("general.alignment", 32),
        _field_u32("general.file_type", 32),
        _field_string_array("tokenizer.ggml.tokens", ["a", "bb"]),
    ]
    tensors = [
        _tensor("blk.0.attn_q.weight", (256, 2), 30, 0),
        _tensor("output_norm.weight", (3,), 0, 448),
    ]
    body = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(fields))
    body += b"".join(fields) + b"".join(tensors)
    body += b"\0" * ((32 - len(body) % 32) % 32)
    path.write_bytes(body)


def _recipe() -> DryRunResult:
    tensors = (
        DryRunTensorAssignment(
            ordinal=1,
            total_tensors=2,
            name="blk.0.attn_q.weight",
            shape=(256, 2, 1, 1),
            src_type="bf16",
            dst_type="iq3_s",
            is_quantized=True,
            orig_bytes=1024,
            new_bytes=220,
        ),
        DryRunTensorAssignment(
            ordinal=2,
            total_tensors=2,
            name="output_norm.weight",
            shape=(3, 1, 1, 1),
            src_type="f32",
            dst_type="f32",
            is_quantized=False,
            orig_bytes=12,
            new_bytes=12,
        ),
    )
    return DryRunResult(tensors, 2, 1036, 232, Decimal("16"), Decimal("3.6"))


def test_read_layout_skips_large_value_classes(tmp_path: Path):
    path = tmp_path / "source.gguf"
    _write_fixture(path)
    layout = read_gguf_layout(path)

    assert layout.version == 3
    assert layout.alignment == 32
    assert layout.metadata_count == 3
    assert len(layout.tensors) == 2
    assert layout.tensor_map["blk.0.attn_q.weight"].shape == (256, 2)
    assert layout.data_offset == path.stat().st_size
    assert layout.field_map["tokenizer.ggml.tokens"].scalar_value is None


def test_exact_prediction_includes_imatrix_metadata_and_padding(tmp_path: Path):
    path = tmp_path / "source.gguf"
    _write_fixture(path)
    layout = read_gguf_layout(path)
    metadata = QuantizationMetadata(
        file_type=26,
        imatrix=ImatrixProvenance(
            file="imatrix.gguf",
            dataset="calibration",
            entries_count=1,
            chunks_count=7,
        ),
    )

    prediction = predict_quantized_size(layout, _recipe(), metadata)

    # IQ3_S: 2 rows * 110 bytes = 220, padded to 224. F32: 12 -> 32.
    assert prediction.tensor_payload_bytes == 232
    assert prediction.tensor_padding_bytes == 24
    assert prediction.total_bytes == prediction.metadata_bytes + 256
    assert [tensor.payload_bytes for tensor in prediction.tensors] == [220, 12]


def test_prediction_rejects_shape_mismatch(tmp_path: Path):
    path = tmp_path / "source.gguf"
    _write_fixture(path)
    layout = read_gguf_layout(path)
    bad = _recipe()
    wrong = DryRunTensorAssignment(
        ordinal=1,
        total_tensors=2,
        name="blk.0.attn_q.weight",
        shape=(512, 1),
        src_type="bf16",
        dst_type="iq3_s",
        is_quantized=True,
        orig_bytes=1024,
        new_bytes=220,
    )
    bad = DryRunResult((wrong, bad.tensors[1]), 2, 1036, 232)

    with pytest.raises(GGUFError, match="Shape mismatch"):
        predict_quantized_size(layout, bad, QuantizationMetadata(file_type=26))


def test_prediction_rejects_unknown_qtype(tmp_path: Path):
    path = tmp_path / "source.gguf"
    _write_fixture(path)
    layout = read_gguf_layout(path)
    recipe = _recipe()
    unknown = DryRunTensorAssignment(
        ordinal=1,
        total_tensors=2,
        name="blk.0.attn_q.weight",
        shape=(256, 2),
        src_type="bf16",
        dst_type="made_up",
        is_quantized=True,
        orig_bytes=1024,
        new_bytes=1,
    )
    recipe = DryRunResult((unknown, recipe.tensors[1]), 2, 1036, 13)

    with pytest.raises(GGUFError, match="Unsupported destination qtype"):
        predict_quantized_size(layout, recipe, QuantizationMetadata(file_type=26))


def test_rejects_bad_magic(tmp_path: Path):
    path = tmp_path / "bad.gguf"
    path.write_bytes(b"NOPE")
    with pytest.raises(GGUFError, match="Not a little-endian GGUF"):
        read_gguf_layout(path)
