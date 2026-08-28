"""Immutable data models for parsed llama.cpp dry-run outputs."""

from dataclasses import dataclass
from decimal import Decimal
from typing import Iterator


class DryRunParseError(ValueError):
    """Raised when llama.cpp dry-run output fails parsing or validation."""


@dataclass(frozen=True, slots=True)
class DryRunTensorAssignment:
    """Parsed per-tensor quantization assignment from dry-run output.

    Note: orig_bytes and new_bytes are deterministically converted from
    display-rounded MiB text. They represent rounded display values,
    not exact unpadded or padded size oracle values.
    """

    ordinal: int
    total_tensors: int
    name: str
    shape: tuple[int, ...]
    src_type: str
    dst_type: str
    is_quantized: bool
    orig_bytes: int
    new_bytes: int

    @property
    def quantized(self) -> bool:
        return self.is_quantized

    @property
    def orig_bytes_display(self) -> int:
        return self.orig_bytes

    @property
    def new_bytes_display(self) -> int:
        return self.new_bytes


@dataclass(frozen=True, slots=True)
class DryRunResult:
    """Parsed aggregate dry-run quantization result."""

    tensors: tuple[DryRunTensorAssignment, ...]
    total_tensors: int
    reported_orig_bytes: int
    reported_new_bytes: int
    reported_orig_bpw: Decimal | None = None
    reported_new_bpw: Decimal | None = None

    def __len__(self) -> int:
        return len(self.tensors)

    def __iter__(self) -> Iterator[DryRunTensorAssignment]:
        return iter(self.tensors)

    def __getitem__(self, idx: int) -> DryRunTensorAssignment:
        return self.tensors[idx]

    @property
    def tensor_map(self) -> dict[str, DryRunTensorAssignment]:
        return {t.name: t for t in self.tensors}

    @property
    def total_orig_bytes_display(self) -> int:
        return sum(t.orig_bytes for t in self.tensors)

    @property
    def total_new_bytes_display(self) -> int:
        return sum(t.new_bytes for t in self.tensors)

    @property
    def quantized_count(self) -> int:
        return sum(1 for t in self.tensors if t.is_quantized)

    @property
    def unchanged_count(self) -> int:
        return sum(1 for t in self.tensors if not t.is_quantized)
