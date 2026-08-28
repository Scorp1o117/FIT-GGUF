"""Tests for canonical GGUF imatrix profiling."""

import json
from pathlib import Path
import struct

import pytest

from fit_gguf import GGUFError, load_imatrix_profile, write_profile_json


def _string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def _field_string(key: str, value: str) -> bytes:
    return _string(key) + struct.pack("<I", 8) + _string(value)


def _field_u32(key: str, value: int) -> bytes:
    return _string(key) + struct.pack("<II", 4, value)


def _field_string_array(key: str, values: list[str]) -> bytes:
    return (
        _string(key)
        + struct.pack("<IIQ", 9, 8, len(values))
        + b"".join(_string(value) for value in values)
    )


def _tensor(name: str, shape: tuple[int, ...], offset: int) -> bytes:
    return (
        _string(name)
        + struct.pack("<I", len(shape))
        + struct.pack("<" + "Q" * len(shape), *shape)
        + struct.pack("<IQ", 0, offset)
    )


def _pad(data: bytes) -> bytes:
    return data + b"\0" * ((32 - len(data) % 32) % 32)


def _write_imatrix(path: Path, *, omit_last_count: bool = False) -> None:
    fields = [
        _field_string("general.type", "imatrix"),
        _field_string_array("imatrix.datasets", ["fixture"]),
        _field_u32("imatrix.chunk_count", 2),
        _field_u32("imatrix.chunk_size", 4),
    ]
    descriptors = [
        _tensor("blk.0.attn_q.weight.in_sum2", (2,), 0),
        _tensor("blk.0.attn_q.weight.counts", (1,), 32),
        _tensor("blk.1.attn_q.weight.in_sum2", (2,), 64),
    ]
    if not omit_last_count:
        descriptors.append(_tensor("blk.1.attn_q.weight.counts", (1,), 96))
    metadata = b"GGUF" + struct.pack("<IQQ", 3, len(descriptors), len(fields))
    metadata += b"".join(fields) + b"".join(descriptors)
    data = _pad(struct.pack("<2f", 4.0, 16.0))
    data += _pad(struct.pack("<f", 4.0))
    data += _pad(struct.pack("<2f", 2.0, 2.0))
    if not omit_last_count:
        data += _pad(struct.pack("<f", 2.0))
    path.write_bytes(_pad(metadata) + data)


def test_load_profile_normalizes_like_llama_cpp(tmp_path: Path):
    path = tmp_path / "imatrix.gguf"
    _write_imatrix(path)

    profile = load_imatrix_profile(path)

    assert profile.schema_version == 1
    assert profile.datasets == ("fixture",)
    assert profile.chunk_count == 2
    assert profile.chunk_size == 4
    assert len(profile.entries) == 2

    first = profile.entry_map["blk.0.attn_q.weight"]
    second = profile.entry_map["blk.1.attn_q.weight"]
    assert first.mean == pytest.approx(2.5)
    assert first.rms == pytest.approx((8.5) ** 0.5)
    assert first.p50 == pytest.approx(2.5)
    assert first.count_sum == 4
    assert first.role == "attn_q"
    assert first.block == 0
    assert first.role_percentile == 1.0
    assert second.mean == 1.0
    assert second.role_percentile == 0.0
    assert second.role_relative_mean == pytest.approx(1.0 / 1.75)


def test_profile_json_is_versioned_and_deterministic(tmp_path: Path):
    source = tmp_path / "imatrix.gguf"
    output = tmp_path / "profile.json"
    _write_imatrix(source)
    write_profile_json(load_imatrix_profile(source), output)

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["entry_count"] == 2
    assert payload["entries"][0]["name"] == "blk.0.attn_q.weight"
    assert output.read_text(encoding="utf-8").endswith("\n")


def test_rejects_mismatched_sum_count_pairs(tmp_path: Path):
    path = tmp_path / "broken.gguf"
    _write_imatrix(path, omit_last_count=True)
    with pytest.raises(GGUFError, match="Mismatched imatrix sums/counts pair"):
        load_imatrix_profile(path)
