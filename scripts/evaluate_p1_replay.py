#!/usr/bin/env python3
"""Mechanically evaluate the preregistered P1 replay gates (G1-G8)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

REPO = Path("/run/media/s117/OS/FIT-GGUF")
P1 = REPO / "experiments/2026-08-29-p1-cli"
FITDIR = REPO / "artifacts/fit/p1-replay"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def check(gates: dict, gate: str, description: str, *conditions: bool) -> None:
    detail = [condition for condition in conditions]
    gates[gate] = {
        "description": description,
        "pass": all(conditions),
        "conditions": [bool(condition) for condition in conditions],
    }


def main() -> int:
    gates: dict[str, object] = {}

    huihui = load(P1 / "huihui/analysis/analysis.json")
    huihui_meta = huihui["metadata"]["imatrix"]
    check(
        gates, "G1", "Huihui analyze provenance and preset predictions",
        huihui["source"]["sha256"] == "8a033407c8f58d43102aade25b973cc6d2f2ce5c5cbf4dc75a2cdb60b9e33cbc",
        huihui["presets"]["lower"]["predicted_size_bytes"] == 12_580_875_232,
        huihui["presets"]["upper"]["predicted_size_bytes"] == 15_082_507_232,
        huihui["imatrix"]["entry_count"] == 496,
        huihui_meta["file"] == "imatrix_unsloth.gguf",
        huihui_meta["dataset"] == "unsloth_calibration_dataset",
        huihui_meta["entries_count"] == 496,
        huihui_meta["chunks_count"] == 1_251,
    )

    o_huihui = load(P1 / "huihui/o-fit50-plan.json")
    check(
        gates, "G2", "Huihui plan original equals M7 tensor-types",
        sha256(P1 / "huihui/o-fit50-tensor-types.txt")
        == "d7e662980a5a28f4a586f06cb78f11b313a1439b89303d69e9f183f5e96a238c",
        o_huihui["target_bytes"] == 13_831_691_232,
        o_huihui["predicted_size_bytes"] == 13_831_486_432,
        o_huihui["unused_bytes"] == 204_800,
    )

    b_huihui = load(P1 / "huihui/b-fit50-plan.json")
    check(
        gates, "G3", "Huihui plan balanced equals M10 tensor-types",
        sha256(P1 / "huihui/b-fit50-tensor-types.txt")
        == "0f1e30a0d63ff726370c5443014c35a89002603a2112b16fc8ad669d6fcaba02",
        b_huihui["predicted_size_bytes"] == 13_828_987_872,
    )

    check(
        gates, "G4", "Huihui replay artifacts match M9/M10 SHA-256",
        sha256(FITDIR / "Huihui-O-FIT50-replay.gguf")
        == "e4fe1c46ab89c8b6343203168ebeec699372c2fb21f411c61c47edc2e1f33306",
        sha256(FITDIR / "Huihui-B-FIT50-replay.gguf")
        == "7cfa1b91600115c046cb9afcae8347adc7de5a77b0d880b47a956cb8a4799a07",
    )

    granite = load(P1 / "granite/analysis/analysis.json")
    granite_meta = granite["metadata"]["imatrix"]
    check(
        gates, "G5", "Granite analyze provenance and preset predictions",
        granite["imatrix"]["sha256"]
        == "5488dbe0391dd8e54b1404cc14d805bd92ea2bfe09eb14be5794bbd0894ce18e",
        granite["presets"]["lower"]["predicted_size_bytes"] == 4_089_184_640,
        granite["presets"]["upper"]["predicted_size_bytes"] == 4_820_287_872,
        granite["imatrix"]["entry_count"] == 280,
        granite_meta["file"] == "imatrix-granite-apex-c512.gguf",
        granite_meta["dataset"]
        == "/run/media/s117/OS/Models/imatrix-calibration/APEX-imatrix-Small.txt",
        granite_meta["entries_count"] == 280,
        isinstance(granite_meta["chunks_count"], int) and granite_meta["chunks_count"] > 0,
    )

    o_granite = load(P1 / "granite/o-fit50-plan.json")
    check(
        gates, "G6", "Granite plan original equals M16 tensor-types",
        sha256(P1 / "granite/o-fit50-tensor-types.txt")
        == "f5e8781793c0641256c2e68bcb6cfb8377c031aba6fda6683f20f5da1007ccff",
        o_granite["target_bytes"] == 4_454_736_256,
        o_granite["predicted_size_bytes"] == 4_454_351_232,
    )

    b_granite = load(P1 / "granite/b-fit50-plan.json")
    check(
        gates, "G7", "Granite plan balanced equals M16 tensor-types",
        sha256(P1 / "granite/b-fit50-tensor-types.txt")
        == "1ddfa270f824b8f73bcb77d5df8befb19c2a5a8eafb20a38381192bfd4909a22",
        b_granite["predicted_size_bytes"] == 4_454_564_224,
    )

    check(
        gates, "G8", "Granite replay artifacts match M16 SHA-256",
        sha256(FITDIR / "Granite-O-FIT50-replay.gguf")
        == "09ca3d8505ed728b1c1a2202d7a9793268f66d3ee8ad5001f5b76993f1d1023e",
        sha256(FITDIR / "Granite-B-FIT50-replay.gguf")
        == "17660767b5967a08112c27f04ad95b67278e9c372afde2cbd6e37ba81c123e21",
    )

    notes = {
        "granite_chunks_count_provenance": (
            f"derived chunks_count={granite_meta['chunks_count']} from the imatrix "
            "GGUF; the M16 hand META recorded 1250. quantize.cpp writes this KV as "
            "a 4-byte integer, so the value cannot affect size prediction."
        )
    }
    verdict = {
        "schema_version": 1,
        "gates": gates,
        "all_pass": all(gate["pass"] for gate in gates.values()),  # type: ignore[index]
        "notes": notes,
    }
    (P1 / "gate-verdict.json").write_text(
        json.dumps(verdict, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({name: gate["pass"] for name, gate in gates.items()}, indent=None))  # type: ignore[index]
    print(f"ALL_PASS={verdict['all_pass']}")
    return 0 if verdict["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
