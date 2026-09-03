#!/usr/bin/env python3
"""Backfill seed-provenance sidecar records for historical eval points.

Codex audit (2026-09-04, P1-B): cache seeds decide "minimum verified PASS",
so official runs must be able to require that every seed carries a provenance
sidecar attesting the frozen eval-v1 closure. Historical points were evaluated
before sidecars existed — this script appends one record per admissible seed
point, marked ``attestation: "experiment-record"`` (the point's five-domain
logs live in this experiment tree beside the frozen closure; per-log
cryptographic proof does not exist for pre-sidecar evaluations).

Existing records (e.g. poison-window taints) are preserved untouched; records
already carrying a contract digest are never rewritten.
"""

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from fit_gguf.eval import contract_digest  # noqa: E402
from fit_gguf.eval.provenance import sha256_file  # noqa: E402
from fit_gguf.eval.results import parse_llama_kl_log  # noqa: E402
from fit_gguf.eval.contract import DOMAINS  # noqa: E402

M2 = REPO / "experiments" / "2026-09-02-m2-topkl-calibration"
MANIFEST = M2 / "results" / "artifact-manifest.txt"
LOGS = M2 / "logs"
PROVENANCE = M2 / "results" / "seed-provenance.jsonl"
REFERENCE_MANIFEST = REPO / "experiments" / "2026-09-02-eval-v1" / "reference-manifest-orcarouter.json"
MODEL_PREFIX = "orcarouter-"


def main() -> int:
    sizes: dict[str, int] = {}
    for line in MANIFEST.read_text().splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[0].startswith(MODEL_PREFIX):
            sizes[parts[0][len(MODEL_PREFIX):]] = int(parts[1])

    by_point: dict[str, int] = {}
    for entry in sorted(LOGS.iterdir()):
        name = entry.name
        if not name.startswith("eval-") or not name.endswith(".log"):
            continue
        body = name[len("eval-"):-len(".log")]
        if not body.startswith(MODEL_PREFIX):
            continue
        match = re.search(r"-(" + "|".join(DOMAINS) + r")$", body)
        if match is None:
            continue
        point = body[: -len(match.group(0))][len(MODEL_PREFIX):]
        if point not in sizes:
            continue
        by_point.setdefault(point, 0)
        by_point[point] += 1

    existing: dict[str, dict] = {}
    if PROVENANCE.is_file():
        for line in PROVENANCE.read_text().splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            existing[str(record.get("name", ""))] = record

    digest = contract_digest()
    manifest_sha = sha256_file(REFERENCE_MANIFEST)
    added = 0
    rewrite: list[dict] = []
    with PROVENANCE.open("a", encoding="utf-8") as handle:
        for point in sorted(by_point):
            if by_point[point] != len(DOMAINS):
                continue  # incomplete eval set: not a seed at all
            record = existing.get(point)
            if record is not None and record.get("eval_contract_digest") is not None:
                continue  # already carries a closure attestation
            if record is not None and record.get("attestation") == "experiment-record":
                # refresh the manifest binding if the manifest was re-pinned
                if record.get("reference_manifest_sha256") != manifest_sha:
                    record["reference_manifest_sha256"] = manifest_sha
                    record["note"] = "manifest sha refreshed after re-pin"
                    rewrite.append(record)
                continue  # already backfilled
            handle.write(json.dumps({
                "name": point,
                "size_bytes": sizes[point],
                "window_lower_preset": record.get("window_lower_preset") if record else None,
                "window_upper_preset": record.get("window_upper_preset") if record else None,
                "eval_contract_digest": digest,
                "reference_manifest_sha256": manifest_sha,
                "attestation": "experiment-record",
                "note": "pre-sidecar evaluation; five-domain logs in this experiment tree",
            }) + "\n")
            added += 1
    if rewrite:
        kept = [json.loads(l) for l in PROVENANCE.read_text().splitlines() if l]
        by_name = {json.dumps(r.get("name")) + str(r.get("size_bytes")): r for r in kept}
        for rec in rewrite:
            by_name[json.dumps(rec.get("name")) + str(rec.get("size_bytes"))] = rec
        PROVENANCE.write_text(
            "".join(json.dumps(r) + "\n" for r in by_name.values()), encoding="utf-8"
        )
    print(f"backfilled {added} sidecar records, refreshed {len(rewrite)} -> {PROVENANCE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
