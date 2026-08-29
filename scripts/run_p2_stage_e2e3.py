#!/usr/bin/env python3
"""P2 stage E2'+E3: imatrix provenance check, then the full preset ladder sweep."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys

REPO = Path("/run/media/s117/OS/FIT-GGUF")
sys.path.insert(0, str(REPO / "src"))

from fit_gguf import load_imatrix_profile, parse_dry_run, predict_quantized_size, read_gguf_layout  # noqa: E402
from fit_gguf.gguf import QuantizationMetadata  # noqa: E402
from fit_gguf.pipeline import PRESET_FILE_TYPES, _size_record, derive_imatrix_provenance  # noqa: E402

SRC = REPO / "artifacts/source/orcarouter-Qwen3.8-27B-Uncensored-BF16.gguf"
IMATRIX = REPO / "imatrix_unsloth.gguf"
RUNTIME = REPO / "tools/llama-b10666-rocm"
P2 = REPO / "experiments/2026-08-29-p2-full-envelope"
LOGS = P2 / "logs"
LOGS.mkdir(parents=True, exist_ok=True)

IMATRIX_SHA = "0ee5b10bd0c2fa2127c6f4b43dbfe1efd71e383b63217af9dade1de36599f1c1"
DISPLAY_MIB_BYTES = 5243  # 0.005 MiB display-rounding step

# Canonical (non-alias) quantize.cpp presets in dominant-type BPW order,
# with each preset's dominant-type BPW from the pinned traits table. Ties
# share a group; the ladder gate is non-decreasing inside a group and
# strictly increasing between groups with distinct dominant BPW.
PRESET_LADDER = [
    ("IQ1_S", 1.5625),
    ("IQ1_M", 1.75),
    ("IQ2_XXS", 2.0625),
    ("IQ2_XS", 2.3125),
    ("IQ2_S", 2.5625),
    ("IQ2_M", 2.5625),   # iq2_s-based mixture (quantize.cpp: 2.7 bpw effective)
    ("Q2_K_S", 2.625),
    ("Q2_K", 2.625),
    ("IQ3_XXS", 3.0625),
    ("IQ3_XS", 3.3),     # pinned description bpw; mixture between 3.06 and 3.44
    ("Q3_K_S", 3.4375),
    ("IQ3_S", 3.4375),
    ("IQ3_M", 3.4375),   # iq3_s-based mixture (3.66 bpw effective)
    ("Q3_K_M", 3.4375),
    ("Q3_K_L", 3.4375),
    ("IQ4_XS", 4.25),
    ("IQ4_NL", 4.5),
    ("Q4_K_S", 4.5),
    ("Q4_K_M", 4.5),
    ("Q5_K_S", 5.5),
    ("Q5_K_M", 5.5),
    ("Q6_K", 6.5625),
    ("Q8_0", 8.5),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def e2_prime() -> dict:
    print("== E2': imatrix provenance and coverage ==")
    actual = sha256(IMATRIX)
    assert actual == IMATRIX_SHA, f"imatrix sha mismatch: {actual}"
    profile = load_imatrix_profile(IMATRIX)
    entry_names = {entry.name for entry in profile.entries}

    completed = subprocess.run(
        [str(RUNTIME / "llama-quantize"), "--dry-run", "--imatrix", IMATRIX.name,
         str(SRC), "IQ3_M"],
        capture_output=True, text=True, check=True,
    )
    text = completed.stderr + completed.stdout
    (LOGS / "dry-run-e2prime-iq3_m.log").write_text(text, encoding="utf-8")
    recipe = parse_dry_run(text)
    quantized = {t.name for t in recipe.tensors if t.is_quantized}
    missing = sorted(quantized - entry_names)
    # The imatrix covers the 496 layer matrices. token_embd.weight and
    # output.weight have no entries in this imatrix and never did (identical
    # in the Huihui M2 records); the quantizer assigns them deterministically
    # via its fallback, and the dry-run is the assignment oracle (D-0005).
    # The E2' gate is layer-matrix completeness; fallback tensors are recorded.
    known_fallback = {"token_embd.weight", "output.weight"}
    unexpected = [name for name in missing if name not in known_fallback]
    assert not unexpected, f"{len(unexpected)} quantized tensors lack imatrix entries: {unexpected[:5]}"
    record = {
        "imatrix_sha256": actual,
        "imatrix_entries": len(profile.entries),
        "datasets": list(profile.datasets),
        "chunk_count": profile.chunk_count,
        "quantized_tensors_iq3_m": len(quantized),
        "layer_matrix_coverage": "complete (496/496)",
        "fallback_without_imatrix_entry": sorted(missing),
        "fallback_assignments": {
            t.name: t.dst_type for t in recipe.tensors if t.name in missing
        },
        "provenance": (
            "importance values derive from Huihui-Qwen3.8-27B-abliterated "
            "activations (P2 amendment 1, owner directive)"
        ),
    }
    print(json.dumps(record, indent=2))
    return record


def e3() -> dict:
    print("== E3: preset ladder sweep ==")
    layout = read_gguf_layout(SRC)
    profile = load_imatrix_profile(IMATRIX)
    provenance = derive_imatrix_provenance(IMATRIX.name, profile)

    results = {}
    for preset, dominant_bpw in PRESET_LADDER:
        completed = subprocess.run(
            [str(RUNTIME / "llama-quantize"), "--dry-run", "--imatrix", IMATRIX.name,
             str(SRC), preset],
            capture_output=True, text=True,
        )
        text = completed.stderr + completed.stdout
        (LOGS / f"dry-run-{preset.lower()}.log").write_text(text, encoding="utf-8")
        if completed.returncode != 0:
            results[preset] = {
                "accepted": False,
                "returncode": completed.returncode,
                "dominant_bpw": dominant_bpw,
            }
            print(f"{preset:8s} REJECTED by quantizer (rc={completed.returncode})")
            continue
        recipe = parse_dry_run(text)
        prediction = predict_quantized_size(
            layout, recipe,
            QuantizationMetadata(file_type=PRESET_FILE_TYPES[preset], imatrix=provenance),
        )
        # The dry-run "quant size" total counts ALL tensors: quantized new
        # bytes plus unchanged tensors' original bytes (P2 amendment 4).
        total_payload = sum(size.payload_bytes for size in prediction.tensors)
        tolerance = (recipe.total_tensors + 1) * DISPLAY_MIB_BYTES
        payload_diff = abs(total_payload - recipe.reported_new_bytes)
        histogram = Counter(t.dst_type.lower() for t in recipe.tensors)
        results[preset] = {
            "accepted": True,
            "file_type": PRESET_FILE_TYPES[preset],
            "dominant_bpw": dominant_bpw,
            "total_tensors": recipe.total_tensors,
            "quantized_count": recipe.quantized_count,
            "qtype_counts": dict(sorted(histogram.items())),
            **_size_record(prediction),
            "reported_quant_bytes": recipe.reported_new_bytes,
            "payload_diff_bytes": payload_diff,
            "payload_tolerance_bytes": tolerance,
            "payload_ok": payload_diff <= tolerance,
        }
        print(
            f"{preset:8s} predicted={prediction.total_bytes:>14,} "
            f"payload_diff={payload_diff:>9,} tol={tolerance:>9,} "
            f"{'OK' if results[preset]['payload_ok'] else 'FAIL'}"
        )

    # Ladder gate (amendments 2 and 4): strictly increasing between adjacent
    # groups with distinct dominant BPW; inside a group ties may order either
    # way.
    inversions = []
    accepted_order = [
        (preset, bpw) for preset, bpw in PRESET_LADDER if results[preset]["accepted"]
    ]
    groups: list[tuple[float, list[int]]] = []
    for preset, bpw in accepted_order:
        if groups and groups[-1][0] == bpw:
            groups[-1][1].append(results[preset]["predicted_size_bytes"])
        else:
            groups.append((bpw, [results[preset]["predicted_size_bytes"]]))
    for (bpw_a, sizes_a), (bpw_b, sizes_b) in zip(groups, groups[1:]):
        if max(sizes_a) >= min(sizes_b):
            inversions.append(
                {
                    "groups": [bpw_a, bpw_b],
                    "max_prev": max(sizes_a),
                    "min_next": min(sizes_b),
                    "kind": "between_groups",
                }
            )

    rejected = [preset for preset, _ in PRESET_LADDER if not results[preset]["accepted"]]
    payload_all_ok = all(
        results[preset]["payload_ok"] for preset, _ in PRESET_LADDER if results[preset]["accepted"]
    )
    verdict = {
        "schema_version": 1,
        "ladder": results,
        "accepted_count": len(accepted_order),
        "rejected": rejected,
        "inversions": inversions,
        "all_payload_ok": payload_all_ok,
        "pass": payload_all_ok and not rejected and not inversions,
    }
    (P2 / "preset-ladder.json").write_text(
        json.dumps(verdict, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"E3 verdict: accepted={len(accepted_order)}/{len(PRESET_LADDER)} "
        f"payload_all_ok={payload_all_ok} inversions={len(inversions)} "
        f"pass={verdict['pass']}"
    )
    return verdict


def main() -> int:
    e2_record = e2_prime()
    e3_verdict = e3()
    (P2 / "e2prime-record.json").write_text(
        json.dumps(e2_record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not e3_verdict["pass"]:
        print("E3 FAILED", file=sys.stderr)
        return 1
    print("STAGE P2-E2E3 DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
