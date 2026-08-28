"""Minimal GGUF v3 layout reader and exact single-file size predictor."""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from operator import mul
from pathlib import Path
import struct
from typing import BinaryIO

from fit_gguf.models import DryRunResult

GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
GGUF_DEFAULT_ALIGNMENT = 32

# GGUF value types from gguf.h.
_UINT8, _INT8, _UINT16, _INT16, _UINT32, _INT32 = range(6)
_FLOAT32, _BOOL, _STRING, _ARRAY, _UINT64, _INT64, _FLOAT64 = range(6, 13)
_FIXED_VALUE_SIZES = {
    _UINT8: 1,
    _INT8: 1,
    _UINT16: 2,
    _INT16: 2,
    _UINT32: 4,
    _INT32: 4,
    _FLOAT32: 4,
    _BOOL: 1,
    _UINT64: 8,
    _INT64: 8,
    _FLOAT64: 8,
}

# (block elements, encoded block bytes), pinned to ggml commit 4e97ac86e.
GGML_TYPE_TRAITS: dict[str, tuple[int, int]] = {
    "f32": (1, 4),
    "bf16": (1, 2),
    "iq3_s": (256, 110),
    "iq4_xs": (256, 136),
    "q4_k": (256, 144),
    "q5_k": (256, 176),
    "q6_k": (256, 210),
}


class GGUFError(ValueError):
    """Raised when a GGUF layout or size recipe is unsupported or invalid."""


@dataclass(frozen=True, slots=True)
class GGUFField:
    key: str
    value_type: int
    encoded_size: int
    scalar_value: int | float | bool | str | tuple[str, ...] | None = None


@dataclass(frozen=True, slots=True)
class GGUFTensor:
    name: str
    shape: tuple[int, ...]
    type_id: int
    relative_offset: int


@dataclass(frozen=True, slots=True)
class GGUFLayout:
    version: int
    alignment: int
    metadata_count: int
    fields: tuple[GGUFField, ...]
    tensors: tuple[GGUFTensor, ...]
    tensor_info_bytes: int
    raw_metadata_bytes: int
    data_offset: int

    @property
    def field_map(self) -> dict[str, GGUFField]:
        return {field.key: field for field in self.fields}

    @property
    def tensor_map(self) -> dict[str, GGUFTensor]:
        return {tensor.name: tensor for tensor in self.tensors}


@dataclass(frozen=True, slots=True)
class ImatrixProvenance:
    file: str
    dataset: str | None
    entries_count: int
    chunks_count: int | None


@dataclass(frozen=True, slots=True)
class QuantizationMetadata:
    file_type: int
    quantization_version: int = 2
    imatrix: ImatrixProvenance | None = None


@dataclass(frozen=True, slots=True)
class TensorSize:
    name: str
    qtype: str
    payload_bytes: int
    padded_bytes: int


@dataclass(frozen=True, slots=True)
class GGUFSizePrediction:
    metadata_bytes: int
    tensor_payload_bytes: int
    tensor_padding_bytes: int
    total_bytes: int
    tensors: tuple[TensorSize, ...]


class _Reader:
    def __init__(self, file: BinaryIO):
        self.file = file

    def tell(self) -> int:
        return self.file.tell()

    def read_exact(self, size: int) -> bytes:
        data = self.file.read(size)
        if len(data) != size:
            raise GGUFError(f"Unexpected end of GGUF while reading {size} bytes")
        return data

    def unpack(self, fmt: str) -> int | float:
        return struct.unpack("<" + fmt, self.read_exact(struct.calcsize(fmt)))[0]

    def string(self) -> str:
        length = int(self.unpack("Q"))
        try:
            return self.read_exact(length).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise GGUFError("GGUF string is not valid UTF-8") from exc

    def skip(self, size: int) -> None:
        if size < 0:
            raise GGUFError("Cannot skip a negative GGUF value size")
        self.read_exact(size)


def _read_or_skip_value(
    reader: _Reader,
    value_type: int,
    *,
    capture_string_array: bool = False,
) -> int | float | bool | str | tuple[str, ...] | None:
    if value_type in _FIXED_VALUE_SIZES:
        formats = {
            _UINT8: "B", _INT8: "b", _UINT16: "H", _INT16: "h",
            _UINT32: "I", _INT32: "i", _FLOAT32: "f", _BOOL: "?",
            _UINT64: "Q", _INT64: "q", _FLOAT64: "d",
        }
        return reader.unpack(formats[value_type])
    if value_type == _STRING:
        return reader.string()
    if value_type == _ARRAY:
        element_type = int(reader.unpack("I"))
        count = int(reader.unpack("Q"))
        if element_type == _ARRAY:
            raise GGUFError("Nested GGUF arrays are unsupported by the GGUF format")
        fixed_size = _FIXED_VALUE_SIZES.get(element_type)
        if fixed_size is not None:
            reader.skip(count * fixed_size)
        elif element_type == _STRING:
            if capture_string_array:
                return tuple(reader.string() for _ in range(count))
            for _ in range(count):
                length = int(reader.unpack("Q"))
                reader.skip(length)
        else:
            raise GGUFError(f"Unknown GGUF array element type: {element_type}")
        return None
    raise GGUFError(f"Unknown GGUF value type: {value_type}")


def _align(value: int, alignment: int) -> int:
    if alignment <= 0 or alignment & (alignment - 1):
        raise GGUFError(f"Alignment must be a positive power of two, got {alignment}")
    return (value + alignment - 1) & -alignment


def _canonical_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
    """Ignore representational trailing-one dimensions added by ggml logging."""
    end = len(shape)
    while end > 1 and shape[end - 1] == 1:
        end -= 1
    return shape[:end]


def read_gguf_layout(path: str | Path) -> GGUFLayout:
    """Read GGUF metadata and tensor descriptors without reading tensor data."""
    with Path(path).open("rb") as file:
        reader = _Reader(file)
        if reader.read_exact(4) != GGUF_MAGIC:
            raise GGUFError("Not a little-endian GGUF file")
        version = int(reader.unpack("I"))
        if version != GGUF_VERSION:
            raise GGUFError(f"Only GGUF v{GGUF_VERSION} is supported, got v{version}")
        tensor_count = int(reader.unpack("Q"))
        metadata_count = int(reader.unpack("Q"))

        fields: list[GGUFField] = []
        seen_fields: set[str] = set()
        for _ in range(metadata_count):
            start = reader.tell()
            key = reader.string()
            if key in seen_fields:
                raise GGUFError(f"Duplicate GGUF metadata key: {key}")
            value_type = int(reader.unpack("I"))
            scalar_value = _read_or_skip_value(
                reader,
                value_type,
                capture_string_array=key == "imatrix.datasets",
            )
            fields.append(GGUFField(key, value_type, reader.tell() - start, scalar_value))
            seen_fields.add(key)

        tensor_info_start = reader.tell()
        tensors: list[GGUFTensor] = []
        seen_tensors: set[str] = set()
        for _ in range(tensor_count):
            name = reader.string()
            if name in seen_tensors:
                raise GGUFError(f"Duplicate GGUF tensor name: {name}")
            dimensions = int(reader.unpack("I"))
            if dimensions < 1 or dimensions > 4:
                raise GGUFError(f"Unsupported dimension count {dimensions} for tensor {name}")
            shape = tuple(int(reader.unpack("Q")) for _ in range(dimensions))
            if any(d <= 0 for d in shape):
                raise GGUFError(f"Non-positive dimension for tensor {name}: {shape}")
            type_id = int(reader.unpack("I"))
            relative_offset = int(reader.unpack("Q"))
            tensors.append(GGUFTensor(name, shape, type_id, relative_offset))
            seen_tensors.add(name)

        raw_metadata_bytes = reader.tell()
        field_map = {field.key: field for field in fields}
        alignment_field = field_map.get("general.alignment")
        alignment = (
            int(alignment_field.scalar_value)
            if alignment_field is not None
            else GGUF_DEFAULT_ALIGNMENT
        )
        data_offset = _align(raw_metadata_bytes, alignment)

    return GGUFLayout(
        version=version,
        alignment=alignment,
        metadata_count=metadata_count,
        fields=tuple(fields),
        tensors=tuple(tensors),
        tensor_info_bytes=raw_metadata_bytes - tensor_info_start,
        raw_metadata_bytes=raw_metadata_bytes,
        data_offset=data_offset,
    )


def _encoded_field_size(key: str, value_type: int, value: int | str) -> int:
    key_size = 8 + len(key.encode("utf-8")) + 4
    if value_type == _UINT32:
        return key_size + 4
    if value_type == _STRING:
        return key_size + 8 + len(str(value).encode("utf-8"))
    raise GGUFError(f"Unsupported synthesized GGUF value type: {value_type}")


def predict_output_metadata_size(
    layout: GGUFLayout,
    metadata: QuantizationMetadata,
) -> int:
    """Predict build-10666 single-file quantization metadata size."""
    if layout.alignment != GGUF_DEFAULT_ALIGNMENT:
        raise GGUFError(
            f"Pinned quantizer writes with alignment {GGUF_DEFAULT_ALIGNMENT}; "
            f"source alignment is {layout.alignment}"
        )

    fields = {field.key: field.encoded_size for field in layout.fields}
    for key in ("split.no", "split.count", "split.tensors.count"):
        fields.pop(key, None)

    fields["general.quantization_version"] = _encoded_field_size(
        "general.quantization_version", _UINT32, metadata.quantization_version
    )
    fields["general.file_type"] = _encoded_field_size(
        "general.file_type", _UINT32, metadata.file_type
    )

    if metadata.imatrix is not None:
        imatrix = metadata.imatrix
        additions: list[tuple[str, int, int | str]] = [
            ("quantize.imatrix.file", _STRING, imatrix.file),
            ("quantize.imatrix.entries_count", _UINT32, imatrix.entries_count),
        ]
        if imatrix.dataset is not None:
            additions.append(("quantize.imatrix.dataset", _STRING, imatrix.dataset))
        if imatrix.chunks_count is not None and imatrix.chunks_count > 0:
            additions.append(("quantize.imatrix.chunks_count", _UINT32, imatrix.chunks_count))
        for key, value_type, value in additions:
            fields[key] = _encoded_field_size(key, value_type, value)

    # magic + version + tensor count + metadata count
    raw_size = 24 + sum(fields.values()) + layout.tensor_info_bytes
    return _align(raw_size, GGUF_DEFAULT_ALIGNMENT)


def predict_quantized_size(
    layout: GGUFLayout,
    recipe: DryRunResult,
    metadata: QuantizationMetadata,
) -> GGUFSizePrediction:
    """Predict the exact build-10666 single-file GGUF size for a recipe."""
    source_tensors = layout.tensor_map
    recipe_tensors = recipe.tensor_map
    if source_tensors.keys() != recipe_tensors.keys():
        missing = sorted(source_tensors.keys() - recipe_tensors.keys())
        extra = sorted(recipe_tensors.keys() - source_tensors.keys())
        raise GGUFError(f"Recipe/source tensor mismatch; missing={missing}, extra={extra}")

    sizes: list[TensorSize] = []
    for assignment in recipe.tensors:
        source = source_tensors[assignment.name]
        if _canonical_shape(source.shape) != _canonical_shape(assignment.shape):
            raise GGUFError(
                f"Shape mismatch for {assignment.name}: source={source.shape}, "
                f"recipe={assignment.shape}"
            )
        qtype = assignment.dst_type.lower()
        try:
            block_size, type_size = GGML_TYPE_TRAITS[qtype]
        except KeyError as exc:
            raise GGUFError(f"Unsupported destination qtype: {assignment.dst_type}") from exc
        ne0 = source.shape[0]
        if ne0 % block_size:
            raise GGUFError(
                f"Tensor {source.name} ne0={ne0} is not divisible by {qtype} "
                f"block size {block_size}"
            )
        rows = reduce(mul, source.shape[1:], 1)
        payload = (ne0 // block_size) * type_size * rows
        padded = _align(payload, GGUF_DEFAULT_ALIGNMENT)
        sizes.append(TensorSize(source.name, qtype, payload, padded))

    metadata_bytes = predict_output_metadata_size(layout, metadata)
    tensor_payload_bytes = sum(size.payload_bytes for size in sizes)
    padded_tensor_bytes = sum(size.padded_bytes for size in sizes)
    return GGUFSizePrediction(
        metadata_bytes=metadata_bytes,
        tensor_payload_bytes=tensor_payload_bytes,
        tensor_padding_bytes=padded_tensor_bytes - tensor_payload_bytes,
        total_bytes=metadata_bytes + padded_tensor_bytes,
        tensors=tuple(sizes),
    )
