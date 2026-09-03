"""Extract the G2 regression fixture from the Ling amendment-2 ground truth.

Snapshots into one committed JSON (tests/fixtures/g2_ling_amendment2.json):
  (a) the Ling BF16 source GGUF layout (fields + tensor infos, no data),
  (b) the two frozen analyses' static quantization metadata,
  (c) the six plans' EFFECTIVE recipes, parsed by fit's own parse_dry_run from
      the retained oracle dry-run logs (the recipe llama-quantize actually
      applied under --tensor-type-file),
  (d) the recorded actual artifact sizes and sha256s,
so the G2 byte-exact regression test runs without the multi-GB artifacts and
without tmpfs. The quantize-time imatrix string is the one the amendment-2
driver passed to llama-quantize (run_ling_amendment2.sh: IMX under
/dev/shm/m2-ling); b10666 embeds that exact string as quantize.imatrix.file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from fit_gguf.dry_run import parse_dry_run  # noqa: E402
from fit_gguf.gguf import ImatrixProvenance, QuantizationMetadata, predict_quantized_size, read_gguf_layout  # noqa: E402
from fit_gguf.models import DryRunResult, DryRunTensorAssignment  # noqa: E402

CALIB = REPO / "experiments/2026-09-02-m2-topkl-calibration"
AMEND2 = CALIB / "ling-amendment2"
RESULTS = CALIB / "results-ling"
PLANS = ["L-A1", "L-A2", "L-A3", "L-B1", "L-B2", "L-B3"]
# Ground truth measured byte-level on the L-A1 artifact (see
# results/g2-root-cause-480b.md); the source values are re-read live.
LA1_GROUND_TRUTH = {
    "metadata_bytes": 6516160,
    "output_tensor_info_bytes": 33295,
    "imatrix_file_encoded_analysis": 140,
    "imatrix_file_encoded_quantize": 74,
    "actual_total_bytes": 5090965568,
    "tensor_payload_bytes": 5084449408,
}
QUANTIZE_IMATRIX_ARG = "/dev/shm/m2-ling/ling-imatrix.dat"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1 << 22):
            digest.update(chunk)
    return digest.hexdigest()


def parse_manifest(path: Path) -> dict[str, tuple[int, str]]:
    records: dict[str, tuple[int, str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        tokens = line.split()
        if len(tokens) < 3:
            continue
        name = tokens[0]
        size = next((t for t in tokens[1:] if t.isdigit()), None)
        sha = tokens[-1]
        if size is not None and len(sha) == 64:
            records[name] = (int(size), sha)  # last occurrence wins (clean line)
    return records


def compact_recipe(recipe: DryRunResult) -> list[list]:
    return [
        [t.name, list(t.shape), t.src_type, t.dst_type, t.is_quantized]
        for t in recipe.tensors
    ]


def recipe_from_compact(entries: list[list]) -> DryRunResult:
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bf16",
        type=Path,
        default=Path(
            "/run/media/s117/OS/Models/Ling-3.0-tiny-abliterated-APEX-GGUF/"
            "Ling-3.0-tiny-abliterated-bf16.gguf"
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO / "tests/fixtures/g2_ling_amendment2.json",
    )
    args = parser.parse_args()

    print(f"reading layout: {args.bf16}")
    layout = read_gguf_layout(args.bf16)
    snapshot = {
        "version": layout.version,
        "alignment": layout.alignment,
        "metadata_count": layout.metadata_count,
        "fields": [
            {"key": f.key, "value_type": f.value_type, "encoded_size": f.encoded_size}
            for f in layout.fields
        ],
        "tensors": [
            {"name": t.name, "shape": list(t.shape), "type_id": t.type_id}
            for t in layout.tensors
        ],
        "source_tensor_info_bytes": layout.tensor_info_bytes,
        "source_raw_metadata_bytes": layout.raw_metadata_bytes,
        "source_data_offset": layout.data_offset,
    }

    manifest = parse_manifest(RESULTS / "artifact-manifest.txt")

    analyses: dict[str, dict] = {}
    plans: list[dict] = []
    for plan_name in PLANS:
        record = json.loads((AMEND2 / f"{plan_name}-plan.json").read_text())
        analysis_sha = record["analysis_sha256"]
        if analysis_sha not in analyses:
            analysis_path = Path(record["analysis_path"])
            if not analysis_path.is_absolute():
                analysis_path = REPO / analysis_path
            if not analysis_path.is_file():
                candidates = sorted(CALIB.glob("ling-analysis-*/analysis.json"))
                analysis_path = next(
                    p for p in candidates if sha256_file(p) == analysis_sha
                )
            analysis_path = analysis_path.resolve()
            payload = json.loads(analysis_path.read_text())
            analyses[analysis_sha] = {
                "path": str(analysis_path.relative_to(REPO)),
                "lower_preset": payload["presets"]["lower"]["name"],
                "imatrix_arg": payload["imatrix"]["arg"],
                "metadata": payload["metadata"],
            }
        oracle_log = AMEND2 / f"{plan_name}-oracle-dry-run.log"
        effective = parse_dry_run(oracle_log.read_text(encoding="utf-8"))
        effective_compact = compact_recipe(effective)
        manifest_key = f"ling-FIT-{plan_name.replace('-', '')}"
        if manifest_key not in manifest:
            raise SystemExit(f"manifest has no record for {manifest_key}")
        actual_size, actual_sha = manifest[manifest_key]
        plans.append(
            {
                "name": plan_name,
                "analysis_sha256": analysis_sha,
                # Audit chain (GPT ruling): raw oracle evidence must survive
                # next to the parsed recipe so the chain
                # oracle log -> parser -> recipe -> predictor -> artifact
                # stays independently verifiable.
                "oracle_log_sha256": sha256_file(oracle_log),
                "parsed_recipe_sha256": hashlib.sha256(
                    json.dumps(
                        effective_compact, sort_keys=True, ensure_ascii=False
                    ).encode("utf-8")
                ).hexdigest(),
                "effective_recipe": effective_compact,
                "predicted_size_bytes": record["predicted_size_bytes"],
                "actual_size_bytes": actual_size,
                "actual_sha256": actual_sha,
            }
        )
        print(
            f"{plan_name}: effective_tensors={effective.total_tensors} "
            f"predicted(old)={record['predicted_size_bytes']} actual={actual_size}"
        )

    fixture = {
        "schema_version": 1,
        "purpose": (
            "G2 exact-size regression: byte-exact prediction of the six Ling "
            "amendment-2 artifacts from layout snapshot + effective recipes + "
            "the quantize-time imatrix string"
        ),
        "quantize_imatrix_arg": QUANTIZE_IMATRIX_ARG,
        "source": {
            "name": args.bf16.name,
            "sha256": sha256_file(args.bf16),
            "size_bytes": args.bf16.stat().st_size,
        },
        "layout": snapshot,
        "analyses": analyses,
        "plans": plans,
        "la1_ground_truth": {
            **LA1_GROUND_TRUTH,
            "source_tensor_info_bytes": layout.tensor_info_bytes,
        },
    }

    # Fail fast at extraction: the fixed predictor must already reproduce all
    # six recorded artifact sizes byte-exactly.
    for plan_entry in plans:
        analysis = analyses[plan_entry["analysis_sha256"]]
        static = ImatrixProvenance(**analysis["metadata"]["imatrix"])
        metadata = QuantizationMetadata(
            file_type=analysis["metadata"]["file_type"],
            quantization_version=analysis["metadata"]["quantization_version"],
            imatrix=ImatrixProvenance(
                file=QUANTIZE_IMATRIX_ARG,
                dataset=static.dataset,
                entries_count=static.entries_count,
                chunks_count=static.chunks_count,
            ),
        )
        prediction = predict_quantized_size(
            layout, recipe_from_compact(plan_entry["effective_recipe"]), metadata
        )
        status = "OK" if prediction.total_bytes == plan_entry["actual_size_bytes"] else "MISMATCH"
        print(
            f"  {plan_entry['name']}: refinalized={prediction.total_bytes} "
            f"actual={plan_entry['actual_size_bytes']} {status}"
        )
        if status != "OK":
            raise SystemExit(f"extraction failed: {plan_entry['name']} not byte-exact")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(fixture, indent=1, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"fixture written: {args.out} ({args.out.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
