"""Parser and validation for llama.cpp dry-run quantization logs."""

from decimal import Decimal, ROUND_HALF_UP
import re
from typing import Iterable

from fit_gguf.models import DryRunParseError, DryRunResult, DryRunTensorAssignment

BYTES_PER_MIB = 1024 * 1024

ANSI_ESCAPE_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
CANDIDATE_LINE_RE = re.compile(r"\[\s*\d+\s*/\s*\d+\s*\]")

TENSOR_LINE_RE = re.compile(
    r"^(?:(?P<prefix>.*?)\s+)?"
    r"\[\s*(?P<ordinal>\d+)\s*/\s*(?P<total>\d+)\s*\]\s+"
    r"(?P<name>\S+)\s+-\s+\["
    r"(?P<shape>[\d\s,]+)"
    r"\],\s+type\s*=\s*(?P<src_type>[A-Za-z0-9_]+),\s+size\s*=\s*"
    r"(?P<size_part>.+)$"
)

QUANTIZED_SIZE_RE = re.compile(
    r"^\s*(?P<orig_mib>\d+(?:\.\d+)?)\s*MiB\s*->\s*(?P<new_mib>\d+(?:\.\d+)?)\s*MiB\s*\(\s*(?P<dst_type>[A-Za-z0-9_]+)\s*\)\s*$"
)

UNCHANGED_SIZE_RE = re.compile(
    r"^\s*(?P<size_mib>\d+(?:\.\d+)?)\s*MiB\s*$"
)

MODEL_SIZE_SUMMARY_RE = re.compile(
    r"^(?:(?P<prefix>.*?)\s+)?"
    r"model\s+size\s*=\s*(?P<orig_mib>\d+(?:\.\d+)?)\s*MiB"
    r"(?:\s*\(\s*(?P<orig_bpw>\d+(?:\.\d+)?)\s*BPW\s*\))?\s*$"
)

QUANT_SIZE_SUMMARY_RE = re.compile(
    r"^(?:(?P<prefix>.*?)\s+)?"
    r"quant\s+size\s*=\s*(?P<new_mib>\d+(?:\.\d+)?)\s*MiB"
    r"(?:\s*\(\s*(?P<new_bpw>\d+(?:\.\d+)?)\s*BPW\s*\))?\s*$"
)


def mib_to_bytes(mib_val: str | Decimal) -> int:
    """Convert printed MiB display value to integer bytes using Decimal and ROUND_HALF_UP.

    Note: The returned byte count is a display-converted value, not an exact
    unpadded or padded file size oracle value.
    """
    if isinstance(mib_val, Decimal):
        dec = mib_val
    else:
        dec = Decimal(str(mib_val).strip())
    byte_dec = dec * Decimal(BYTES_PER_MIB)
    return int(byte_dec.to_integral_value(rounding=ROUND_HALF_UP))


def _display_rounding_tolerance_bytes(mib_str: str) -> int:
    """Compute maximum display rounding error bound in bytes for a printed MiB value."""
    clean = mib_str.strip()
    if "." in clean:
        decimals = len(clean.split(".")[1])
        err_mib = Decimal("0.5") * (Decimal("10") ** -decimals)
    else:
        err_mib = Decimal("0.5")
    err_bytes = int((err_mib * Decimal(BYTES_PER_MIB)).to_integral_value(rounding=ROUND_HALF_UP))
    return err_bytes + 1


def parse_dry_run(output: str | Iterable[str]) -> DryRunResult:
    """Parse and validate llama.cpp build-10666 dry-run output into a DryRunResult."""
    if isinstance(output, str):
        lines = output.splitlines()
    else:
        lines = list(output)

    tensors: list[DryRunTensorAssignment] = []
    seen_ordinals: set[int] = set()
    seen_names: set[str] = set()
    expected_total: int | None = None

    orig_mib_strs: list[str] = []
    new_mib_strs: list[str] = []

    model_size_mib_str: str | None = None
    model_size_bpw: Decimal | None = None
    quant_size_mib_str: str | None = None
    quant_size_bpw: Decimal | None = None

    for raw_line in lines:
        line = ANSI_ESCAPE_RE.sub("", raw_line).strip()
        if not line:
            continue

        model_match = MODEL_SIZE_SUMMARY_RE.match(line)
        if model_match:
            mib_str = model_match.group("orig_mib")
            bpw_str = model_match.group("orig_bpw")
            bpw = Decimal(bpw_str) if bpw_str is not None else None
            if model_size_mib_str is not None and model_size_mib_str != mib_str:
                raise DryRunParseError(
                    f"Conflicting model size summaries: {model_size_mib_str} vs {mib_str}"
                )
            if model_size_mib_str is not None and model_size_bpw != bpw:
                raise DryRunParseError(
                    f"Conflicting model BPW summaries: {model_size_bpw} vs {bpw}"
                )
            model_size_mib_str = mib_str
            model_size_bpw = bpw
            continue

        quant_match = QUANT_SIZE_SUMMARY_RE.match(line)
        if quant_match:
            mib_str = quant_match.group("new_mib")
            bpw_str = quant_match.group("new_bpw")
            bpw = Decimal(bpw_str) if bpw_str is not None else None
            if quant_size_mib_str is not None and quant_size_mib_str != mib_str:
                raise DryRunParseError(
                    f"Conflicting quant size summaries: {quant_size_mib_str} vs {mib_str}"
                )
            if quant_size_mib_str is not None and quant_size_bpw != bpw:
                raise DryRunParseError(
                    f"Conflicting quant BPW summaries: {quant_size_bpw} vs {bpw}"
                )
            quant_size_mib_str = mib_str
            quant_size_bpw = bpw
            continue

        if CANDIDATE_LINE_RE.search(line):
            tensor_match = TENSOR_LINE_RE.match(line)
            if not tensor_match:
                raise DryRunParseError(f"Malformed candidate tensor line: {raw_line.strip()}")

            ordinal = int(tensor_match.group("ordinal"))
            total = int(tensor_match.group("total"))
            name = tensor_match.group("name")
            shape_raw = tensor_match.group("shape")
            src_type = tensor_match.group("src_type")
            size_part = tensor_match.group("size_part")

            try:
                shape_parts = [int(p.strip()) for p in shape_raw.split(",")]
                if not shape_parts or any(d <= 0 for d in shape_parts):
                    raise ValueError("Shape dimensions must be positive integers")
                shape = tuple(shape_parts)
            except Exception as exc:
                raise DryRunParseError(
                    f"Invalid tensor shape {shape_raw!r} in line: {raw_line.strip()}"
                ) from exc

            if expected_total is None:
                expected_total = total
            elif total != expected_total:
                raise DryRunParseError(
                    f"Inconsistent total tensor count: line specifies {total}, expected {expected_total}"
                )

            if ordinal in seen_ordinals:
                raise DryRunParseError(f"Duplicate tensor ordinal: {ordinal}")
            if ordinal < 1 or ordinal > total:
                raise DryRunParseError(
                    f"Tensor ordinal {ordinal} out of range (1..{total})"
                )

            if name in seen_names:
                raise DryRunParseError(f"Duplicate tensor name: {name}")

            quant_size_match = QUANTIZED_SIZE_RE.match(size_part)
            if quant_size_match:
                orig_mib_s = quant_size_match.group("orig_mib")
                new_mib_s = quant_size_match.group("new_mib")
                dst_type = quant_size_match.group("dst_type")
                is_quantized = True
                orig_bytes = mib_to_bytes(orig_mib_s)
                new_bytes = mib_to_bytes(new_mib_s)
            else:
                unchanged_match = UNCHANGED_SIZE_RE.match(size_part)
                if not unchanged_match:
                    raise DryRunParseError(
                        f"Malformed tensor size part {size_part!r} in line: {raw_line.strip()}"
                    )
                size_mib_s = unchanged_match.group("size_mib")
                orig_mib_s = size_mib_s
                new_mib_s = size_mib_s
                dst_type = src_type
                is_quantized = False
                size_b = mib_to_bytes(size_mib_s)
                orig_bytes = size_b
                new_bytes = size_b

            seen_ordinals.add(ordinal)
            seen_names.add(name)
            orig_mib_strs.append(orig_mib_s)
            new_mib_strs.append(new_mib_s)

            assignment = DryRunTensorAssignment(
                ordinal=ordinal,
                total_tensors=total,
                name=name,
                shape=shape,
                src_type=src_type,
                dst_type=dst_type,
                is_quantized=is_quantized,
                orig_bytes=orig_bytes,
                new_bytes=new_bytes,
            )
            tensors.append(assignment)

    if not tensors:
        raise DryRunParseError("No dry-run tensor lines found in output")

    assert expected_total is not None
    if len(tensors) != expected_total:
        raise DryRunParseError(
            f"Parsed {len(tensors)} tensors, but expected total is {expected_total}"
        )

    all_ordinals = {t.ordinal for t in tensors}
    expected_ordinals = set(range(1, expected_total + 1))
    missing = expected_ordinals - all_ordinals
    if missing:
        raise DryRunParseError(f"Missing tensor ordinals: {sorted(missing)}")

    if model_size_mib_str is None:
        raise DryRunParseError("Missing model size summary line")
    if quant_size_mib_str is None:
        raise DryRunParseError("Missing quant size summary line")

    reported_orig_bytes = mib_to_bytes(model_size_mib_str)
    reported_new_bytes = mib_to_bytes(quant_size_mib_str)

    sum_orig_bytes = sum(t.orig_bytes for t in tensors)
    sum_new_bytes = sum(t.new_bytes for t in tensors)

    max_orig_tol = (
        sum(_display_rounding_tolerance_bytes(s) for s in orig_mib_strs)
        + _display_rounding_tolerance_bytes(model_size_mib_str)
        + len(tensors)
        + 1
    )
    max_new_tol = (
        sum(_display_rounding_tolerance_bytes(s) for s in new_mib_strs)
        + _display_rounding_tolerance_bytes(quant_size_mib_str)
        + len(tensors)
        + 1
    )

    orig_diff = abs(sum_orig_bytes - reported_orig_bytes)
    if orig_diff > max_orig_tol:
        raise DryRunParseError(
            f"Reported model size ({reported_orig_bytes} bytes) differs from tensor sum "
            f"({sum_orig_bytes} bytes) by {orig_diff} bytes, exceeding display tolerance of {max_orig_tol} bytes"
        )

    new_diff = abs(sum_new_bytes - reported_new_bytes)
    if new_diff > max_new_tol:
        raise DryRunParseError(
            f"Reported quant size ({reported_new_bytes} bytes) differs from tensor sum "
            f"({sum_new_bytes} bytes) by {new_diff} bytes, exceeding display tolerance of {max_new_tol} bytes"
        )

    sorted_tensors = tuple(sorted(tensors, key=lambda t: t.ordinal))

    return DryRunResult(
        tensors=sorted_tensors,
        total_tensors=expected_total,
        reported_orig_bytes=reported_orig_bytes,
        reported_new_bytes=reported_new_bytes,
        reported_orig_bpw=model_size_bpw,
        reported_new_bpw=quant_size_bpw,
    )
