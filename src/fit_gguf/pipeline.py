"""Shared analyze/plan/quantize pipeline behind the fit CLI.

Encapsulates the exact M2-M16 pipeline. Every constant that experiment scripts
hand-copied from retained artifacts is replaced by derivation from the imatrix
GGUF and the pinned quantize.cpp behavior, so the same code runs on new models.
"""

from __future__ import annotations

from dataclasses import asdict, replace
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess

from fit_gguf.candidates import (
    CandidateSet,
    RejectedTransition,
    UpgradeCandidate,
    generate_upgrade_candidates,
)
from fit_gguf.dry_run import parse_dry_run
from fit_gguf.gguf import (
    GGUFSizePrediction,
    ImatrixProvenance,
    QuantizationMetadata,
    predict_quantized_size,
    read_gguf_layout,
)
from fit_gguf.imatrix import ImatrixProfile, load_imatrix_profile, write_profile_json
from fit_gguf.llama_integration import write_tensor_type_file
from fit_gguf.models import DryRunResult, DryRunTensorAssignment
from fit_gguf.optimizer import (
    OptimizationPlan,
    optimize_block_balanced,
    optimize_greedy,
    optimize_random,
    write_fit_recipe,
)

ANALYSIS_SCHEMA_VERSION = 1
PLAN_SCHEMA_VERSION = 1
FIT_GGUF_VERSION = "0.1.0"

# general.file_type values from pinned include/llama.h (LLAMA_FTYPE enum).
# Only each KV's encoded size matters for size prediction; the constants keep
# analysis.json faithful to what llama-quantize writes per preset.
PRESET_FILE_TYPES = {
    "IQ3_S": 26,
    "IQ3_M": 27,
    "IQ4_XS": 30,
}

# strncpy(kvo.val_str, ..., 127); kvo.val_str[127] = '\0' in quantize.cpp.
KV_OVERRIDE_STRING_MAX_BYTES = 127

_POLICIES = ("original", "balanced", "random")


class PipelineError(ValueError):
    """Raised when an analyze/plan/quantize request is invalid or fails."""


def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _truncate_kv_string(value: str) -> str:
    encoded = value.encode("utf-8")[:KV_OVERRIDE_STRING_MAX_BYTES]
    return encoded.decode("utf-8", errors="ignore")


def derive_imatrix_provenance(
    imatrix_arg: str, profile: ImatrixProfile
) -> ImatrixProvenance:
    """Derive the quantize.imatrix.* KVs from the imatrix file and CLI argument.

    quantize.cpp writes: file = the imatrix path as passed (byte-truncated to
    127), dataset = imatrix.datasets[0] (truncated, omitted when absent),
    entries_count = loaded entry count, chunks_count = imatrix.chunk_count.
    """
    datasets = profile.datasets
    return ImatrixProvenance(
        file=_truncate_kv_string(imatrix_arg),
        dataset=_truncate_kv_string(datasets[0]) if datasets else None,
        entries_count=len(profile.entries),
        chunks_count=profile.chunk_count,
    )


def auto_block_span(profile: ImatrixProfile) -> int:
    """Four block quarters: 64 blocks -> 16 (Huihui), 40 blocks -> 10 (Granite)."""
    max_block = max(entry.block for entry in profile.entries)
    return (max_block + 1 + 3) // 4


def run_dry_run(
    runtime_dir: str | Path,
    source: str | Path,
    imatrix_arg: str,
    preset: str,
    log_path: str | Path,
) -> DryRunResult:
    """Run one pinned llama-quantize --dry-run and strictly parse the log."""
    binary = Path(runtime_dir) / "llama-quantize"
    if not binary.is_file():
        raise PipelineError(f"llama-quantize not found at {binary}")
    command = [str(binary), "--dry-run", "--imatrix", imatrix_arg, str(source), preset]
    completed = subprocess.run(command, capture_output=True, text=True)
    # Historical logs capture unbuffered stderr before buffered stdout; keep
    # that order so the parsed text matches the M2/M16 ground truth.
    text = completed.stderr + completed.stdout
    Path(log_path).write_text(text, encoding="utf-8")
    if completed.returncode != 0:
        raise PipelineError(
            f"dry-run for {preset} failed with code {completed.returncode}; log: {log_path}"
        )
    return parse_dry_run(text)


def _qtype_histogram(recipe: DryRunResult) -> dict[str, int]:
    counts: dict[str, int] = {}
    for tensor in recipe.tensors:
        key = tensor.dst_type.lower()
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _size_record(size: GGUFSizePrediction) -> dict[str, int]:
    return {
        "predicted_size_bytes": size.total_bytes,
        "metadata_bytes": size.metadata_bytes,
        "tensor_payload_bytes": size.tensor_payload_bytes,
        "tensor_padding_bytes": size.tensor_padding_bytes,
    }


def apply_overrides(lower_recipe: DryRunResult, plan: OptimizationPlan) -> DryRunResult:
    """Rebuild the lower-preset recipe with the plan's upgrades applied."""
    overrides = {candidate.tensor: candidate.to_qtype for candidate in plan.selected}
    tensors = tuple(
        replace(tensor, dst_type=overrides[tensor.name])
        if tensor.name in overrides
        else tensor
        for tensor in lower_recipe.tensors
    )
    return DryRunResult(
        tensors=tensors,
        total_tensors=lower_recipe.total_tensors,
        reported_orig_bytes=lower_recipe.reported_orig_bytes,
        reported_new_bytes=lower_recipe.reported_new_bytes,
    )


def _recipe_to_json(recipe: DryRunResult) -> dict[str, object]:
    return {
        "total_tensors": recipe.total_tensors,
        "reported_orig_bytes": recipe.reported_orig_bytes,
        "reported_new_bytes": recipe.reported_new_bytes,
        "tensors": [asdict(tensor) for tensor in recipe.tensors],
    }


def _recipe_from_json(payload: dict[str, object]) -> DryRunResult:
    return DryRunResult(
        tensors=tuple(
            DryRunTensorAssignment(
                ordinal=int(tensor["ordinal"]),
                total_tensors=int(tensor["total_tensors"]),
                name=str(tensor["name"]),
                shape=tuple(int(dimension) for dimension in tensor["shape"]),
                src_type=str(tensor["src_type"]),
                dst_type=str(tensor["dst_type"]),
                is_quantized=bool(tensor["is_quantized"]),
                orig_bytes=int(tensor["orig_bytes"]),
                new_bytes=int(tensor["new_bytes"]),
            )
            for tensor in payload["tensors"]  # type: ignore[union-attr]
        ),
        total_tensors=int(payload["total_tensors"]),  # type: ignore[arg-type]
        reported_orig_bytes=int(payload["reported_orig_bytes"]),  # type: ignore[arg-type]
        reported_new_bytes=int(payload["reported_new_bytes"]),  # type: ignore[arg-type]
    )


def _candidate_set_to_json(candidate_set: CandidateSet) -> dict[str, object]:
    return {
        "candidates": [asdict(candidate) for candidate in candidate_set.candidates],
        "rejected": [asdict(transition) for transition in candidate_set.rejected],
        "lower_size_bytes": candidate_set.lower_size_bytes,
        "upper_size_bytes": candidate_set.upper_size_bytes,
    }


def _candidate_set_from_json(payload: dict[str, object]) -> CandidateSet:
    return CandidateSet(
        candidates=tuple(UpgradeCandidate(**item) for item in payload["candidates"]),  # type: ignore[arg-type]
        rejected=tuple(RejectedTransition(**item) for item in payload["rejected"]),  # type: ignore[arg-type]
        lower_size_bytes=int(payload["lower_size_bytes"]),  # type: ignore[arg-type]
        upper_size_bytes=int(payload["upper_size_bytes"]),  # type: ignore[arg-type]
    )


def _dump_json(payload: object, path: str | Path) -> None:
    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _load_json(path: str | Path) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def analyze(
    source: str | Path,
    imatrix: str | Path,
    runtime_dir: str | Path,
    out_dir: str | Path,
    *,
    lower_preset: str = "IQ3_M",
    upper_preset: str = "IQ4_XS",
    imatrix_arg: str | None = None,
    hash_sources: bool = True,
    quantization_version: int = 2,
) -> Path:
    """Profile one source/imatrix pair and freeze the candidate set."""
    source_path = Path(source)
    imatrix_path = Path(imatrix)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    if not source_path.is_file():
        raise PipelineError(f"Source GGUF not found: {source_path}")
    if not imatrix_path.is_file():
        raise PipelineError(f"Imatrix GGUF not found: {imatrix_path}")
    if imatrix_arg is None:
        imatrix_arg = imatrix_path.name
    if lower_preset not in PRESET_FILE_TYPES or upper_preset not in PRESET_FILE_TYPES:
        raise PipelineError(
            f"Presets must be among {sorted(PRESET_FILE_TYPES)}; "
            f"got {lower_preset}/{upper_preset}"
        )

    lower_recipe = run_dry_run(
        runtime_dir,
        source_path,
        imatrix_arg,
        lower_preset,
        out_path / f"dry-run-{lower_preset.lower()}.log",
    )
    upper_recipe = run_dry_run(
        runtime_dir,
        source_path,
        imatrix_arg,
        upper_preset,
        out_path / f"dry-run-{upper_preset.lower()}.log",
    )
    layout = read_gguf_layout(source_path)
    profile = load_imatrix_profile(imatrix_path)

    metadata = QuantizationMetadata(
        file_type=PRESET_FILE_TYPES[lower_preset],
        quantization_version=quantization_version,
        imatrix=derive_imatrix_provenance(imatrix_arg, profile),
    )
    lower_size = predict_quantized_size(layout, lower_recipe, metadata)
    upper_size = predict_quantized_size(layout, upper_recipe, metadata)
    if lower_size.total_bytes >= upper_size.total_bytes:
        raise PipelineError(
            f"Lower preset {lower_preset} predicted {lower_size.total_bytes:,} bytes, "
            f"not below upper preset {upper_preset} at {upper_size.total_bytes:,} bytes"
        )
    candidate_set = generate_upgrade_candidates(
        lower_recipe, upper_recipe, lower_size, upper_size, profile
    )

    write_profile_json(profile, out_path / "profile.json")
    candidate_json = _candidate_set_to_json(candidate_set)
    payload = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "fit_gguf_version": FIT_GGUF_VERSION,
        "source": {
            "path": str(source_path),
            "size_bytes": source_path.stat().st_size,
            "sha256": _sha256_file(source_path) if hash_sources else None,
        },
        "imatrix": {
            "path": str(imatrix_path),
            "sha256": _sha256_file(imatrix_path) if hash_sources else None,
            "arg": imatrix_arg,
            "datasets": list(profile.datasets),
            "chunk_count": profile.chunk_count,
            "chunk_size": profile.chunk_size,
            "entry_count": len(profile.entries),
        },
        "runtime": {
            "dir": str(runtime_dir),
            "llama_quantize": str(Path(runtime_dir) / "llama-quantize"),
        },
        "presets": {
            "lower": {
                "name": lower_preset,
                "file_type": PRESET_FILE_TYPES[lower_preset],
                "dry_run_log": f"dry-run-{lower_preset.lower()}.log",
                "qtype_counts": _qtype_histogram(lower_recipe),
                **_size_record(lower_size),
            },
            "upper": {
                "name": upper_preset,
                "file_type": PRESET_FILE_TYPES[upper_preset],
                "dry_run_log": f"dry-run-{upper_preset.lower()}.log",
                "qtype_counts": _qtype_histogram(upper_recipe),
                **_size_record(upper_size),
            },
        },
        "metadata": {
            "file_type": metadata.file_type,
            "quantization_version": metadata.quantization_version,
            "imatrix": asdict(metadata.imatrix) if metadata.imatrix else None,
        },
        "block_span_auto": auto_block_span(profile),
        "net_preset_gap_bytes": upper_size.total_bytes - lower_size.total_bytes,
        "candidate_budget_bytes": candidate_set.candidate_budget_bytes,
        "candidate_count": len(candidate_set.candidates),
        "rejected_count": len(candidate_set.rejected),
        "candidates": candidate_json["candidates"],
        "rejected": candidate_json["rejected"],
        "lower_recipe": _recipe_to_json(lower_recipe),
    }
    analysis_path = out_path / "analysis.json"
    _dump_json(payload, analysis_path)
    return analysis_path


def load_analysis(path: str | Path) -> dict[str, object]:
    try:
        payload = _load_json(path)
    except OSError as error:
        raise PipelineError(f"Cannot read analysis file: {error}") from error
    if payload.get("schema_version") != ANALYSIS_SCHEMA_VERSION:
        raise PipelineError(f"Unsupported analysis schema in {path}")
    return payload


def resolve_target(lower_size: int, upper_size: int, fit: str) -> int:
    """Exact integer FIT target: lower + Fraction(fit) of the preset gap."""
    fraction = Fraction(fit)
    if not 0 < fraction < 1:
        raise PipelineError(f"--fit must be strictly between 0 and 1, got {fit}")
    gap = upper_size - lower_size
    return lower_size + gap * fraction.numerator // fraction.denominator


def plan(
    analysis_path: str | Path,
    out_prefix: str | Path,
    *,
    fit: str | None = None,
    target_bytes: int | None = None,
    policy: str = "original",
    seed: str | int | None = None,
    block_span: int | str = "auto",
) -> dict[str, object]:
    """Plan one artifact from a frozen analysis and write its three records."""
    if policy not in _POLICIES:
        raise PipelineError(f"Policy must be one of {_POLICIES}, got {policy}")
    if policy == "random" and seed is None:
        raise PipelineError("Policy 'random' requires --seed")
    if (fit is None) == (target_bytes is None):
        raise PipelineError("Exactly one of --fit or --target-bytes is required")

    payload = load_analysis(analysis_path)
    lower_preset = payload["presets"]["lower"]["name"]  # type: ignore[index]
    upper_preset = payload["presets"]["upper"]["name"]  # type: ignore[index]
    metadata_payload = payload["metadata"]
    imatrix_payload = metadata_payload["imatrix"]  # type: ignore[index]
    metadata = QuantizationMetadata(
        file_type=int(metadata_payload["file_type"]),  # type: ignore[arg-type]
        quantization_version=int(metadata_payload["quantization_version"]),  # type: ignore[arg-type]
        imatrix=ImatrixProvenance(**imatrix_payload),  # type: ignore[arg-type]
    )
    candidate_set = _candidate_set_from_json(
        {
            "candidates": payload["candidates"],
            "rejected": payload["rejected"],
            "lower_size_bytes": payload["presets"]["lower"]["predicted_size_bytes"],  # type: ignore[index]
            "upper_size_bytes": payload["presets"]["upper"]["predicted_size_bytes"],  # type: ignore[index]
        }
    )
    lower_recipe = _recipe_from_json(payload["lower_recipe"])  # type: ignore[arg-type]
    layout = read_gguf_layout(payload["source"]["path"])  # type: ignore[arg-type]

    lower_size = candidate_set.lower_size_bytes
    upper_size = candidate_set.upper_size_bytes
    if target_bytes is None:
        assert fit is not None
        target = resolve_target(lower_size, upper_size, fit)
    else:
        target = int(target_bytes)
        if not lower_size <= target <= upper_size:
            raise PipelineError(
                f"Target {target:,} bytes outside preset range "
                f"[{lower_size:,}, {upper_size:,}]"
            )

    if block_span == "auto":
        resolved_span = int(payload["block_span_auto"])
    else:
        resolved_span = int(block_span)  # type: ignore[arg-type]
    if policy == "original":
        optimization = optimize_greedy(target, candidate_set)
    elif policy == "balanced":
        optimization = optimize_block_balanced(target, candidate_set, block_span=resolved_span)
    else:
        optimization = optimize_random(target, candidate_set, seed=seed)

    recipe = apply_overrides(lower_recipe, optimization)
    prediction = predict_quantized_size(layout, recipe, metadata)
    if prediction.total_bytes > target:
        raise PipelineError(
            f"Predicted {prediction.total_bytes:,} bytes exceeds target {target:,}"
        )

    prefix = Path(out_prefix)
    if prefix.parent != Path("."):
        prefix.parent.mkdir(parents=True, exist_ok=True)
    recipe_path = prefix.with_name(prefix.name + "-recipe.json")
    types_path = prefix.with_name(prefix.name + "-tensor-types.txt")
    write_fit_recipe(optimization, recipe_path, lower_preset=lower_preset, upper_preset=upper_preset)
    write_tensor_type_file(optimization, types_path)

    record = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "fit_gguf_version": FIT_GGUF_VERSION,
        "analysis_path": str(analysis_path),
        "analysis_sha256": _sha256_file(Path(analysis_path)),
        "policy": policy,
        "seed": str(seed) if seed is not None else None,
        "block_span": resolved_span if policy == "balanced" else None,
        "fit": fit,
        "target_bytes": target,
        "lower_preset": lower_preset,
        "upper_preset": upper_preset,
        "lower_size_bytes": lower_size,
        "upper_size_bytes": upper_size,
        "predicted_size_bytes": prediction.total_bytes,
        "metadata_bytes": prediction.metadata_bytes,
        "tensor_payload_bytes": prediction.tensor_payload_bytes,
        "tensor_padding_bytes": prediction.tensor_padding_bytes,
        "unused_bytes": optimization.unused_bytes,
        "selected_count": len(optimization.selected),
        "skipped_count": optimization.skipped_count,
        "selected_cost_bytes": optimization.selected_cost_bytes,
        "recipe_path": str(recipe_path),
        "recipe_sha256": _sha256_file(recipe_path),
        "tensor_types_path": str(types_path),
        "tensor_types_sha256": _sha256_file(types_path),
    }
    plan_record_path = prefix.with_name(prefix.name + "-plan.json")
    _dump_json(record, plan_record_path)
    return record


def quantize(
    analysis_path: str | Path,
    tensor_types_path: str | Path,
    out_path: str | Path,
    *,
    expect_bytes: int | None = None,
) -> dict[str, object]:
    """Quantize per a tensor-types file and verify the exact output size."""
    payload = load_analysis(analysis_path)
    binary = Path(payload["runtime"]["llama_quantize"])  # type: ignore[index]
    if not binary.is_file():
        raise PipelineError(f"llama-quantize not found at {binary}")
    source = payload["source"]["path"]  # type: ignore[index]
    imatrix_arg = payload["imatrix"]["arg"]  # type: ignore[index]
    lower_preset = payload["presets"]["lower"]["name"]  # type: ignore[index]

    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(binary),
        "--imatrix",
        str(imatrix_arg),
        "--tensor-type-file",
        str(tensor_types_path),
        str(source),
        str(output),
        str(lower_preset),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    size = output.stat().st_size if output.is_file() else 0
    record = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "analysis_path": str(analysis_path),
        "tensor_types_path": str(tensor_types_path),
        "tensor_types_sha256": _sha256_file(Path(tensor_types_path)),
        "command": command,
        "returncode": completed.returncode,
        "output_path": str(output),
        "size_bytes": size,
        "expect_bytes": expect_bytes,
        "size_matches_expectation": expect_bytes is None or size == int(expect_bytes),
        "sha256": _sha256_file(output) if size else None,
        "stderr_tail": completed.stderr[-2000:],
    }
    _dump_json(record, Path(str(output) + ".quantize-record.json"))
    if completed.returncode != 0:
        raise PipelineError(f"llama-quantize failed with code {completed.returncode}")
    if expect_bytes is not None and size != int(expect_bytes):
        raise PipelineError(
            f"Output is {size:,} bytes; expected {int(expect_bytes):,} bytes"
        )
    return record
