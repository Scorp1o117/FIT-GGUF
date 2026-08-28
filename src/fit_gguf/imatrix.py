"""Canonical GGUF imatrix profiling for tensor-level FIT planning."""

from __future__ import annotations

from array import array
from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path
import re
import sys

from fit_gguf.gguf import GGUFError, GGUFTensor, read_gguf_layout

_IN_SUM2_SUFFIX = ".in_sum2"
_COUNTS_SUFFIX = ".counts"
_BLOCK_ROLE_RE = re.compile(r"^blk\.(?P<block>\d+)\.(?P<role>.+)\.weight$")


@dataclass(frozen=True, slots=True)
class ImatrixTensorProfile:
    name: str
    block: int
    role: str
    width: int
    count_values: int
    count_min: int
    count_max: int
    count_sum: int
    mean: float
    rms: float
    stddev: float
    minimum: float
    p50: float
    p95: float
    p99: float
    maximum: float
    nonzero_fraction: float
    global_relative_mean: float = 0.0
    role_relative_mean: float = 0.0
    global_percentile: float = 0.0
    role_percentile: float = 0.0


@dataclass(frozen=True, slots=True)
class ImatrixProfile:
    schema_version: int
    source_file: str
    datasets: tuple[str, ...]
    chunk_count: int
    chunk_size: int
    entries: tuple[ImatrixTensorProfile, ...]

    @property
    def entry_map(self) -> dict[str, ImatrixTensorProfile]:
        return {entry.name: entry for entry in self.entries}


def _read_f32_tensor(file, data_offset: int, tensor: GGUFTensor) -> array:
    if tensor.type_id != 0:
        raise GGUFError(f"Imatrix tensor {tensor.name} must be F32, got type id {tensor.type_id}")
    count = math.prod(tensor.shape)
    file.seek(data_offset + tensor.relative_offset)
    values = array("f")
    try:
        values.fromfile(file, count)
    except EOFError as exc:
        raise GGUFError(f"Truncated imatrix tensor data: {tensor.name}") from exc
    if sys.byteorder != "little":
        values.byteswap()
    return values


def _percentile(sorted_values: list[float], fraction: float) -> float:
    if not sorted_values:
        raise GGUFError("Cannot profile an empty imatrix vector")
    position = (len(sorted_values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _rank_percentiles(values: list[float]) -> list[float]:
    if len(values) == 1:
        return [1.0]
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    result = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average_rank = (start + end - 1) / 2
        percentile = average_rank / (len(order) - 1)
        for position in range(start, end):
            result[order[position]] = percentile
        start = end
    return result


def _median(values: list[float]) -> float:
    return _percentile(sorted(values), 0.5)


def _profile_entry(
    name: str,
    sums: array,
    raw_counts: array,
) -> ImatrixTensorProfile:
    match = _BLOCK_ROLE_RE.fullmatch(name)
    if match is None:
        raise GGUFError(f"Unsupported imatrix tensor name: {name}")
    if not raw_counts or len(sums) % len(raw_counts):
        raise GGUFError(
            f"Imatrix sums/counts shape mismatch for {name}: {len(sums)} vs {len(raw_counts)}"
        )

    counts: list[int] = []
    for value in raw_counts:
        if not math.isfinite(value) or value < 0:
            raise GGUFError(f"Invalid imatrix count for {name}: {value}")
        rounded = math.floor(float(value) + 0.5)
        if abs(float(value) - rounded) > 1e-4:
            raise GGUFError(f"Non-integral imatrix count for {name}: {value}")
        counts.append(rounded)

    per_count_width = len(sums) // len(counts)
    normalized: list[float] = []
    for count_index, count in enumerate(counts):
        start = count_index * per_count_width
        for raw in sums[start : start + per_count_width]:
            value = float(raw) / count if count > 0 else 1.0
            if not math.isfinite(value) or value < 0:
                raise GGUFError(f"Invalid normalized imatrix value for {name}: {value}")
            normalized.append(value)

    sorted_values = sorted(normalized)
    mean = math.fsum(normalized) / len(normalized)
    mean_square = math.fsum(value * value for value in normalized) / len(normalized)
    variance = max(0.0, mean_square - mean * mean)
    return ImatrixTensorProfile(
        name=name,
        block=int(match.group("block")),
        role=match.group("role"),
        width=len(normalized),
        count_values=len(counts),
        count_min=min(counts),
        count_max=max(counts),
        count_sum=sum(counts),
        mean=mean,
        rms=math.sqrt(mean_square),
        stddev=math.sqrt(variance),
        minimum=sorted_values[0],
        p50=_percentile(sorted_values, 0.5),
        p95=_percentile(sorted_values, 0.95),
        p99=_percentile(sorted_values, 0.99),
        maximum=sorted_values[-1],
        nonzero_fraction=sum(value > 0 for value in normalized) / len(normalized),
    )


def load_imatrix_profile(path: str | Path) -> ImatrixProfile:
    """Load and summarize the canonical GGUF imatrix using llama.cpp semantics."""
    source = Path(path)
    layout = read_gguf_layout(source)
    fields = layout.field_map
    general_type = fields.get("general.type")
    if general_type is None or general_type.scalar_value != "imatrix":
        raise GGUFError("GGUF is not marked as an imatrix")

    datasets_value = fields.get("imatrix.datasets")
    datasets = (
        datasets_value.scalar_value
        if datasets_value is not None and isinstance(datasets_value.scalar_value, tuple)
        else ()
    )
    chunk_count_field = fields.get("imatrix.chunk_count")
    chunk_size_field = fields.get("imatrix.chunk_size")
    if chunk_count_field is None or chunk_size_field is None or not datasets:
        raise GGUFError("Imatrix metadata is incomplete")
    chunk_count = int(chunk_count_field.scalar_value)
    chunk_size = int(chunk_size_field.scalar_value)
    if chunk_count <= 0 or chunk_size <= 0:
        raise GGUFError("Imatrix chunk metadata must be positive")

    pairs: dict[str, list[GGUFTensor | None]] = {}
    for tensor in layout.tensors:
        if tensor.name.endswith(_IN_SUM2_SUFFIX):
            base = tensor.name[: -len(_IN_SUM2_SUFFIX)]
            pairs.setdefault(base, [None, None])[0] = tensor
        elif tensor.name.endswith(_COUNTS_SUFFIX):
            base = tensor.name[: -len(_COUNTS_SUFFIX)]
            pairs.setdefault(base, [None, None])[1] = tensor
        else:
            raise GGUFError(f"Unexpected imatrix tensor suffix: {tensor.name}")

    entries: list[ImatrixTensorProfile] = []
    with source.open("rb") as file:
        for name in sorted(pairs):
            sums_tensor, counts_tensor = pairs[name]
            if sums_tensor is None or counts_tensor is None:
                raise GGUFError(f"Mismatched imatrix sums/counts pair: {name}")
            sums = _read_f32_tensor(file, layout.data_offset, sums_tensor)
            counts = _read_f32_tensor(file, layout.data_offset, counts_tensor)
            entries.append(_profile_entry(name, sums, counts))

    means = [entry.mean for entry in entries]
    global_median = _median(means)
    global_percentiles = _rank_percentiles(means)
    by_role: dict[str, list[int]] = {}
    for index, entry in enumerate(entries):
        by_role.setdefault(entry.role, []).append(index)

    role_medians: dict[str, float] = {}
    role_percentiles: dict[int, float] = {}
    for role, indices in by_role.items():
        role_values = [means[index] for index in indices]
        role_medians[role] = _median(role_values)
        for index, percentile in zip(indices, _rank_percentiles(role_values), strict=True):
            role_percentiles[index] = percentile

    normalized_entries = tuple(
        replace(
            entry,
            global_relative_mean=entry.mean / global_median if global_median > 0 else 0.0,
            role_relative_mean=(
                entry.mean / role_medians[entry.role] if role_medians[entry.role] > 0 else 0.0
            ),
            global_percentile=global_percentiles[index],
            role_percentile=role_percentiles[index],
        )
        for index, entry in enumerate(entries)
    )
    return ImatrixProfile(
        schema_version=1,
        source_file=source.name,
        datasets=tuple(str(dataset) for dataset in datasets),
        chunk_count=chunk_count,
        chunk_size=chunk_size,
        entries=normalized_entries,
    )


def write_profile_json(profile: ImatrixProfile, path: str | Path) -> None:
    """Write a deterministic, versioned profile JSON record."""
    payload = {
        "schema_version": profile.schema_version,
        "source_file": profile.source_file,
        "datasets": list(profile.datasets),
        "chunk_count": profile.chunk_count,
        "chunk_size": profile.chunk_size,
        "entry_count": len(profile.entries),
        "entries": [asdict(entry) for entry in profile.entries],
    }
    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
