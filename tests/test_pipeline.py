"""Tests for the shared analyze/plan/quantize pipeline and the fit CLI."""

from dataclasses import replace
import hashlib
import os
import struct

import pytest

from fit_gguf.models import DryRunResult, DryRunTensorAssignment
from fit_gguf.imatrix import ImatrixProfile, ImatrixTensorProfile
from fit_gguf.pipeline import (
    PipelineError,
    _candidate_set_from_json,
    _candidate_set_to_json,
    _recipe_from_json,
    _recipe_to_json,
    analyze,
    apply_overrides,
    auto_block_span,
    derive_imatrix_provenance,
    load_analysis,
    plan,
    quantize,
    resolve_target,
)


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


def _tensor_descriptor(name: str, shape: tuple[int, ...], type_id: int, offset: int) -> bytes:
    return (
        _string(name)
        + struct.pack("<I", len(shape))
        + struct.pack("<" + "Q" * len(shape), *shape)
        + struct.pack("<IQ", type_id, offset)
    )


def _pad(data: bytes) -> bytes:
    return data + b"\0" * ((32 - len(data) % 32) % 32)


# name, shape, bf16 bytes, printed MiB values (orig, IQ3_M, IQ4_XS)
TENSORS = (
    ("blk.0.attn_q.weight", (256, 512), 262_144, "0.250", "0.054", "0.066"),
    ("blk.0.ffn_down.weight", (256, 256), 131_072, "0.125", "0.027", "0.033"),
    ("blk.0.attn_norm.weight", (256, 1), 1_024, "0.001", None, None),
)


def _write_source(path) -> None:
    fields = [_field_u32("general.alignment", 32)]
    descriptors = [_tensor_descriptor(name, shape, 1, 0) for name, shape, *_ in TENSORS]
    body = b"GGUF" + struct.pack("<IQQ", 3, len(descriptors), len(fields))
    body += b"".join(fields) + b"".join(descriptors)
    path.write_bytes(_pad(body))


def _write_imatrix(path) -> None:
    fields = [
        _field_string("general.type", "imatrix"),
        _field_string_array("imatrix.datasets", ["fixture-dataset"]),
        _field_u32("imatrix.chunk_count", 3),
        _field_u32("imatrix.chunk_size", 8),
    ]
    descriptors = []
    data = b""
    offset = 0
    for name, *_ in TENSORS[:2]:
        descriptors.append(_tensor_descriptor(f"{name}.in_sum2", (2,), 0, offset))
        data += _pad(struct.pack("<2f", 4.0, 16.0))
        offset += 32
        descriptors.append(_tensor_descriptor(f"{name}.counts", (1,), 0, offset))
        data += _pad(struct.pack("<f", 4.0))
        offset += 32
    body = b"GGUF" + struct.pack("<IQQ", 3, len(descriptors), len(fields))
    body += b"".join(fields) + b"".join(descriptors)
    path.write_bytes(_pad(body) + data)


def _dry_run_log(preset: str) -> str:
    lower = preset == "IQ3_M"
    lines = [
        f"llama_quantize: calculating quantization size for 'src.gguf' as {preset}",
        "",
    ]
    for index, (name, shape, _, orig, lower_mib, upper_mib) in enumerate(TENSORS, 1):
        target = lower_mib if lower else upper_mib
        if target is not None:
            qtype = "IQ3_S" if lower else "IQ4_XS"
            lines.append(
                f"llama_model_quantize_internal: [{index:4d}/{len(TENSORS):4d}] "
                f"{name:<38s} - [{shape[0]:6d}, {shape[1]:6d},      1,      1], "
                f"type =   BF16, size = {orig:>8s} MiB -> {target:>8s} MiB ({qtype})"
            )
        else:
            lines.append(
                f"llama_model_quantize_internal: [{index:4d}/{len(TENSORS):4d}] "
                f"{name:<38s} - [{shape[0]:6d}, {shape[1]:6d},      1,      1], "
                f"type =    F32, size = {orig:>8s} MiB"
            )
    lines += [
        "llama_model_quantize_internal: model size  =    0.376 MiB (16.00 BPW)",
        (
            "llama_model_quantize_internal: quant size  =    0.082 MiB ( 3.14 BPW)"
            if lower
            else "llama_model_quantize_internal: quant size  =    0.100 MiB ( 3.84 BPW)"
        ),
        "llama_model_quantize_internal: WARNING: dry run completed successfully",
        "",
    ]
    return "\n".join(lines)


def _write_stub_runtime(directory) -> None:
    script = directory / "llama-quantize"
    script.write_text(
        "#!/usr/bin/env bash\n"
        'echo "stub: $*" >> "$STUB_CALL_LOG"\n'
        'if [[ "$1" == "--dry-run" ]]; then\n'
        '  preset="${!#}"\n'
        '  cat "$STUB_DIR/${preset,,}.dryrun.log"\n'
        "  exit 0\n"
        "fi\n"
        "for ((i=1; i<=$#; i++)); do\n"
        '  if [[ "${!i}" == IQ3_M ]]; then out="${@:i-1:1}"; fi\n'
        "done\n"
        'truncate -s "$STUB_OUT_BYTES" "$out"\n'
        'echo "stub quantized $out"\n',
        encoding="utf-8",
    )
    script.chmod(0o755)
    (directory / "iq3_m.dryrun.log").write_text(_dry_run_log("IQ3_M"), encoding="utf-8")
    (directory / "iq4_xs.dryrun.log").write_text(_dry_run_log("IQ4_XS"), encoding="utf-8")


@pytest.fixture()
def e2e(tmp_path, monkeypatch):
    source = tmp_path / "src.gguf"
    imatrix = tmp_path / "imx.gguf"
    _write_source(source)
    _write_imatrix(imatrix)
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    _write_stub_runtime(runtime)
    monkeypatch.setenv("STUB_DIR", str(runtime))
    monkeypatch.setenv("STUB_CALL_LOG", str(tmp_path / "stub-calls.log"))
    return {"source": source, "imatrix": imatrix, "runtime": runtime, "tmp": tmp_path}


def _profile(block: int = 63) -> ImatrixProfile:
    entry = ImatrixTensorProfile(
        name=f"blk.{block}.ffn_down.weight", block=block, role="ffn_down", width=1,
        count_values=1, count_min=1, count_max=1, count_sum=1, mean=1.0, rms=1.0,
        stddev=0.0, minimum=1.0, p50=1.0, p95=1.0, p99=1.0, maximum=1.0,
        nonzero_fraction=1.0,
    )
    return ImatrixProfile(
        schema_version=1, source_file="x.gguf", datasets=("d",),
        chunk_count=1, chunk_size=1, entries=(entry,),
    )


def test_auto_block_span_matches_both_models():
    assert auto_block_span(_profile(63)) == 16  # 64 blocks (Huihui)
    assert auto_block_span(_profile(39)) == 10  # 40 blocks (Granite)


def test_derive_imatrix_provenance_truncates_like_quantize_cpp():
    profile = ImatrixProfile(
        schema_version=1, source_file="x.gguf", datasets=("d" * 300,),
        chunk_count=7, chunk_size=8, entries=_profile().entries,
    )
    provenance = derive_imatrix_provenance("i" * 300, profile)
    assert len(provenance.file.encode()) == 127
    assert len(provenance.dataset.encode()) == 127  # type: ignore[union-attr]
    assert provenance.entries_count == 1
    assert provenance.chunks_count == 7


def test_derive_imatrix_provenance_handles_fixture():
    provenance = derive_imatrix_provenance("imx.gguf", _profile())
    assert provenance.file == "imx.gguf"
    assert provenance.dataset == "d"
    assert provenance.chunks_count == 1


def test_resolve_target_reproduces_preregistered_positions():
    lower, upper = 4_089_184_640, 4_820_287_872  # Granite M16
    gap = upper - lower
    assert resolve_target(lower, upper, "0.5") == lower + gap // 2 == 4_454_736_256
    assert resolve_target(lower, upper, "0.25") == lower + gap // 4
    assert resolve_target(lower, upper, "0.75") == lower + 3 * gap // 4
    lower, upper = 12_580_875_232, 15_082_507_232  # Huihui M9
    gap = upper - lower
    assert resolve_target(lower, upper, "0.5") == 13_831_691_232


def test_resolve_target_rejects_degenerate_fractions():
    for bad in ("0", "1", "-0.5", "2"):
        with pytest.raises(PipelineError):
            resolve_target(100, 200, bad)


class _FakeCandidate:
    def __init__(self, tensor, to_qtype):
        self.tensor = tensor
        self.to_qtype = to_qtype


class _FakePlan:
    def __init__(self, selected):
        self.selected = tuple(_FakeCandidate(t, q) for t, q in selected)


def test_apply_overrides_keeps_unselected_tensors():
    tensors = (
        DryRunTensorAssignment(1, 2, "blk.0.a", (256, 2, 1, 1), "BF16", "iq3_s", True, 100, 50),
        DryRunTensorAssignment(2, 2, "blk.0.b", (256, 2, 1, 1), "BF16", "iq3_s", True, 100, 50),
    )
    recipe = DryRunResult(tensors, 2, 200, 100)
    rebuilt = apply_overrides(recipe, _FakePlan((("blk.0.a", "iq4_xs"),)))
    assert rebuilt.tensor_map["blk.0.a"].dst_type == "iq4_xs"
    assert rebuilt.tensor_map["blk.0.b"].dst_type == "iq3_s"
    assert rebuilt.total_tensors == 2
    assert rebuilt.reported_orig_bytes == 200


def test_analysis_plan_quantize_end_to_end_with_stub_runtime(e2e):
    out_dir = e2e["tmp"] / "analysis"
    analysis = analyze(
        e2e["source"], e2e["imatrix"], e2e["runtime"], out_dir, imatrix_arg="imx.gguf"
    )
    payload = load_analysis(analysis)
    assert payload["source"]["sha256"] == hashlib.sha256(e2e["source"].read_bytes()).hexdigest()
    assert payload["imatrix"]["entry_count"] == 2
    assert payload["presets"]["lower"]["qtype_counts"] == {"f32": 1, "iq3_s": 2}
    assert payload["metadata"]["imatrix"]["dataset"] == "fixture-dataset"
    assert payload["block_span_auto"] == 1
    assert (out_dir / "profile.json").is_file()
    assert (out_dir / "dry-run-iq3_m.log").is_file()

    record = plan(analysis, out_dir / "original-fit50", fit="0.5", policy="original")
    assert (out_dir / "original-fit50-plan.json").is_file()
    assert (out_dir / "original-fit50-recipe.json").is_file()
    assert (out_dir / "original-fit50-tensor-types.txt").is_file()
    assert record["target_bytes"] == record["lower_size_bytes"] + (
        record["upper_size_bytes"] - record["lower_size_bytes"]
    ) // 2
    assert record["predicted_size_bytes"] <= record["target_bytes"]
    assert record["analysis_sha256"] == hashlib.sha256(analysis.read_bytes()).hexdigest()

    balanced = plan(analysis, out_dir / "balanced-fit50", fit="0.5", policy="balanced")
    assert balanced["block_span"] == 1

    with pytest.raises(PipelineError):
        plan(analysis, out_dir / "r", fit="0.5", policy="random")

    random_record = plan(
        analysis, out_dir / "random-fit50", fit="0.5", policy="random", seed="unit-seed"
    )
    assert random_record["seed"] == "unit-seed"

    stub_bytes = 4096
    os.environ["STUB_OUT_BYTES"] = str(stub_bytes)
    try:
        quant_record = quantize(
            analysis,
            record["tensor_types_path"],
            e2e["tmp"] / "out.gguf",
            expect_bytes=stub_bytes,
        )
    finally:
        os.environ.pop("STUB_OUT_BYTES", None)
    assert quant_record["sha256"] == hashlib.sha256(
        (e2e["tmp"] / "out.gguf").read_bytes()
    ).hexdigest()
    assert (e2e["tmp"] / "out.gguf.quantize-record.json").is_file()

    with pytest.raises(PipelineError):
        quantize(
            analysis,
            record["tensor_types_path"],
            e2e["tmp"] / "out2.gguf",
            expect_bytes=stub_bytes + 1,
        )


def test_plan_argument_validation(e2e):
    analysis = analyze(
        e2e["source"], e2e["imatrix"], e2e["runtime"], e2e["tmp"] / "a2", hash_sources=False
    )
    with pytest.raises(PipelineError):
        plan(analysis, e2e["tmp"] / "p1", policy="original")
    with pytest.raises(PipelineError):
        plan(analysis, e2e["tmp"] / "p2", fit="0.5", target_bytes=100, policy="original")
    with pytest.raises(PipelineError):
        plan(analysis, e2e["tmp"] / "p3", fit="0.5", policy="nope")
    with pytest.raises(PipelineError):
        plan(analysis, e2e["tmp"] / "p4", policy="random")


def test_cli_main_returns_error_code(e2e, capsys):
    from fit_gguf.cli import main

    with pytest.raises(SystemExit):
        main(["--version"])
    code = main(
        [
            "plan",
            "--analysis", str(e2e["tmp"] / "missing.json"),
            "--fit", "0.5",
            "--out-prefix", str(e2e["tmp"] / "x"),
        ]
    )
    assert code == 2
    assert "error" in capsys.readouterr().err


def test_json_round_trip_of_recipe_and_candidates(e2e):
    import json

    analysis = analyze(
        e2e["source"], e2e["imatrix"], e2e["runtime"], e2e["tmp"] / "a3", hash_sources=False
    )
    payload = load_analysis(analysis)
    original = {
        "candidates": payload["candidates"],
        "rejected": payload["rejected"],
        "lower_size_bytes": payload["presets"]["lower"]["predicted_size_bytes"],
        "upper_size_bytes": payload["presets"]["upper"]["predicted_size_bytes"],
    }
    candidate_set = _candidate_set_from_json(original)
    assert json.loads(json.dumps(_candidate_set_to_json(candidate_set))) == original
    recipe = _recipe_from_json(payload["lower_recipe"])
    assert json.loads(json.dumps(_recipe_to_json(recipe))) == payload["lower_recipe"]
