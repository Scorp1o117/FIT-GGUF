#!/usr/bin/env python3
"""Registry v1 maintainer tool (repo-mutation workflow only).

Per planner verdict (planner-verdict-a3.md): the official registry is mutated
by maintainer workflow — bundle -> validate -> review -> this script -> tests
-> commit -> release. There is no public "add --official" trust-elevation
flag. `review + merged release content` forms the trust root, not the review
itself.

Subcommands
-----------
make-spark-manifest : build the canonical Spark reference manifest from the
                      experiment records (byte-copied into registry/manifests).
add-entry           : admission-checked insert of one entry (copies manifest +
                      calibration record into the registry, updates index).
bootstrap           : one-time v1 migration — orcarouter + spark entries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from fit_gguf.registry import (  # noqa: E402
    GRANDFATHERED_SOURCE_SHAS,
    LEGACY_BASIS,
    REGISTRY_SCHEMA,
    canonical_json_bytes,
    entry_digest,
    sha256_file,
    verify_registry,
)

SPARK_SOURCE_SHA = "aa73aeb45870f7ebb9e5d523323b88468a2b3b613918e941bda439d8c1b59d42"
ORCA_SOURCE_SHA = "f95456457fededfaf9f51cd4739a00aada0795be372882844071cc19146634cd"
EVAL_DIGEST = "5ce78dee9d11e6dfe83416628d0459d462719c0ecddf194c53ea9629db243d7c"


def _sha(path: Path) -> str:
    return sha256_file(path)


def _write_canonical(path: Path, obj: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(obj))
    return _sha(path)


def cmd_make_spark_manifest(args: argparse.Namespace) -> int:
    from tokenizers import Tokenizer

    exp = REPO / "experiments" / "2026-09-06-spark-x25-4tier"
    orca = json.loads(
        (REPO / "experiments/2026-09-02-eval-v1/reference-manifest-orcarouter.json")
        .read_text(encoding="utf-8")
    )
    kld_shas = {}
    for line in Path(args.kld_shas).read_text().splitlines():
        name, sha = line.split()
        kld_shas[name] = sha

    slice_files = {
        "wiki_test": "eval-data/kl-eval-64k.txt",
        "wiki_valid": "eval-data/kl-eval-valid-64k.txt",
        "chinese": "eval-data/kl-eval-cn-64k.txt",
        "code": "eval-data/kl-eval-code-64k.txt",
        "agent_chat": "eval-data/kl-eval-agent-64k.txt",
    }
    tok = Tokenizer.from_file(str(args.tokenizer_json))
    domains = {}
    for name, rel in slice_files.items():
        raw = (REPO / rel).read_bytes()
        corpus_sha = hashlib.sha256(raw).hexdigest()
        orca_domain = orca["domains"][name]
        if corpus_sha != orca_domain["corpus_sha256"]:
            raise SystemExit(f"corpus slice drifted vs frozen manifest: {name}")
        tokens = len(tok.encode(raw.decode("utf-8", errors="replace"), add_special_tokens=False).ids)
        domains[name] = {
            "corpus_sha256": corpus_sha,
            "reference_kld_sha256": kld_shas[name],
            "raw_bytes": len(raw),
            "unicode_codepoints": orca_domain["unicode_codepoints"],
            "expected_valid_tokens": tokens,
        }
    tokenizer_path = Path(args.tokenizer_json)
    manifest = {
        "manifest_schema": "fit.eval_reference_manifest.v1",
        "evaluator_contract": "eval-v1",
        "evaluator_contract_hash": EVAL_DIGEST,
        "source_bf16_gguf_sha256": SPARK_SOURCE_SHA,
        "tokenizer_sha256": _sha(tokenizer_path),
        "tokenizer_hash": f"tokenizer.json sha256 {_sha(tokenizer_path)[:16]}… (Spark-X2.5 vocab 131072, BPE)",
        "runtime_provenance": {
            "backend": "ROCm",
            "gpu_arch": "gfx1151 (Strix Halo)",
            "llama_cpp_source_revision": "llama.cpp PR #27868 head ae320b1 (Spark2_5 support; tools/perplexity diffed byte-identical to b10666 on 2026-09-06)",
            "llama_perplexity_binary_sha256_prefix": _sha(Path(args.perplexity_binary))[:16],
            "runtime_dir": str(Path(args.perplexity_binary).parent),
        },
        "domains": domains,
        "expected_valid_tokens_method": "tokenizers.encode(add_special_tokens=false) over the raw slice bytes; informational",
        "provenance_note": "references generated from the T615 BF16 GGUF with the PR-branch runtime; corpus slices are the frozen eval-data files (SHAs above); unicode_codepoints are slice properties shared with the orcarouter manifest",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out} sha256={_sha(out)}")
    return 0


def _admission_check(entry: dict) -> None:
    cal = entry.get("calibration", {})
    basis = cal.get("basis")
    if basis == LEGACY_BASIS:
        if entry["source_weights_sha256"] not in GRANDFATHERED_SOURCE_SHAS:
            raise SystemExit("admission refused: legacy_bootstrap_v0 is closed to new sources")
        if cal.get("contract") is not None or cal.get("contract_sha256") is not None:
            raise SystemExit("legacy bootstrap entries must leave contract null")
    else:
        if not cal.get("contract_sha256"):
            raise SystemExit("non-legacy entries must pin calibration contract_sha256")


def cmd_add_entry(args: argparse.Namespace) -> int:
    root = Path(args.package_dir)
    reg = root / "registry"
    reg.mkdir(parents=True, exist_ok=True)
    (reg / "entries").mkdir(exist_ok=True)
    (reg / "manifests").mkdir(exist_ok=True)
    (reg / "calibration").mkdir(exist_ok=True)

    manifest_dst = reg / "manifests" / f"{args.source_sha}.json"
    if not (manifest_dst.exists() and Path(args.manifest_src).samefile(manifest_dst)):
        shutil.copyfile(args.manifest_src, manifest_dst)
    record_dst = reg / "calibration" / f"{args.source_sha}.json"
    if not (record_dst.exists() and Path(args.record_src).samefile(record_dst)):
        shutil.copyfile(args.record_src, record_dst)

    entry = {
        "registry_schema": REGISTRY_SCHEMA,
        "model_id": args.model_id,
        "source_weights_sha256": args.source_sha,
        "tokenizer_sha256": args.tokenizer_sha or None,
        "evaluator_contract": "eval-v1",
        "evaluator_contract_sha256": EVAL_DIGEST,
        "reference_manifest": {
            "path": f"registry/manifests/{args.source_sha}.json",
            "sha256": _sha(manifest_dst),
        },
        "guard_profile": {
            "path": args.guard_path,
            "sha256": _sha(root / args.guard_path),
        },
        "calibration": {
            "basis": args.calibration_basis,
            "contract": None if args.calibration_basis == LEGACY_BASIS else "fidelity-calibration-v1",
            "contract_sha256": args.calibration_contract_sha,
            "record": {
                "path": f"registry/calibration/{args.source_sha}.json",
                "sha256": _sha(record_dst),
            },
        },
        "scope": "exact_model",
        "status": args.status,
        "added_in": args.added_in,
    }
    _admission_check(entry)
    entry["entry_sha256"] = entry_digest(entry)

    index_path = reg / "fidelity-registry-v1.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
    else:
        index = {
            "registry_schema": REGISTRY_SCHEMA,
            "registry_id": "fit-official-registry",
            "created": "2026-09-06",
            "evaluator_contract": "eval-v1",
            "evaluator_contract_sha256": EVAL_DIGEST,
            "entries": [],
        }
    index["entries"] = [
        r
        for r in index["entries"]
        if r.get("source_weights_sha256") != args.source_sha
    ]
    index["entries"].append(
        {
            "source_weights_sha256": args.source_sha,
            "entry_sha256": entry["entry_sha256"],
            "model_id": entry["model_id"],
            "status": entry["status"],
            "scope": entry["scope"],
        }
    )
    index["entries"].sort(key=lambda r: (r["source_weights_sha256"], r["model_id"]))
    index_path.write_bytes(canonical_json_bytes(index))

    (reg / "entries" / f"{args.source_sha}.json").write_bytes(
        canonical_json_bytes(entry)
    )
    report = verify_registry(root)
    print(f"entry added: {entry['model_id']} ({args.source_sha[:16]}…)")
    print(f"registry verify: {len(report['entries_verified'])} entries OK")
    return 0


def cmd_bootstrap(args: argparse.Namespace) -> int:
    root = REPO / "src" / "fit_gguf"
    manifest_src = REPO / "experiments/2026-09-06-spark-x25-4tier/manifests/reference-manifest-spark.json"
    if not manifest_src.is_file():
        raise SystemExit("spark manifest missing — run make-spark-manifest first")
    records = root / "registry" / "calibration"
    records.mkdir(parents=True, exist_ok=True)

    orca_record = {
        "schema": "fit.calibration_record.v1",
        "model_id": "orcarouter-Qwen3.8-27B-Uncensored",
        "basis": LEGACY_BASIS,
        "created": "2026-09-06",
        "method": "M2-style model self-calibration (eval-v1 five-domain preset ladder + Fidelity Search); v0.2 bootstrap entry",
        "evidence": [
            {"path": "experiments/2026-09-02-m2-topkl-calibration/results/candidate-calibration-v1.json",
             "sha256": _sha(REPO / "experiments/2026-09-02-m2-topkl-calibration/results/candidate-calibration-v1.json")},
            {"path": "experiments/2026-09-02-eval-v1/result-fit12g-eval-v1.json",
             "sha256": _sha(REPO / "experiments/2026-09-02-eval-v1/result-fit12g-eval-v1.json")},
        ],
        "grandfather_note": "runtime-trusted forever per planner verdict 2026-09-06; admission of new legacy sources is closed",
    }
    spark_search = REPO / "experiments/2026-09-06-spark-x25-4tier/results-fs"
    spark_record = {
        "schema": "fit.calibration_record.v1",
        "model_id": "spark-x25-4b-abliterated",
        "basis": LEGACY_BASIS,
        "created": "2026-09-06",
        "method": "M2-style model self-calibration: 15-point preset ladder + 2 compact-gap probes -> Guard Profile v3 (floors truncated down; P5 at n>=3 / minimum at n<=2); four-tier Fidelity Search, all verified_pass",
        "evidence": [
            {"path": "experiments/2026-09-06-spark-x25-4tier/results/spark-ladder-summary.json",
             "sha256": _sha(REPO / "experiments/2026-09-06-spark-x25-4tier/results/spark-ladder-summary.json")},
            *[
                {"path": f"experiments/2026-09-06-spark-x25-4tier/results-fs/{tier}/fidelity-search-{tier}-summary.json",
                 "sha256": _sha(spark_search / tier / f"fidelity-search-{tier}-summary.json")}
                for tier in ("quality", "balanced", "compact", "mini")
            ],
        ],
        "grandfather_note": "runtime-trusted forever per planner verdict 2026-09-06; admission of new legacy sources is closed",
    }
    _write_canonical(records / f"{ORCA_SOURCE_SHA}.json", orca_record)
    _write_canonical(records / f"{SPARK_SOURCE_SHA}.json", spark_record)

    common = ["--package-dir", str(root), "--added-in", "v0.2.1"]
    rc = cmd_add_entry(argparse.Namespace(
        **dict(vars(args), **{
            "package_dir": str(root),
            "model_id": "orcarouter-Qwen3.8-27B-Uncensored",
            "source_sha": ORCA_SOURCE_SHA,
            "tokenizer_sha": None,
            "guard_path": "profiles/guard/guard-orcarouter-qwen3.8-27b-uncensored-exact-v1.yaml",
            "manifest_src": str(REPO / "experiments/2026-09-02-eval-v1/reference-manifest-orcarouter.json"),
            "record_src": str(records / f"{ORCA_SOURCE_SHA}.json"),
            "calibration_basis": LEGACY_BASIS,
            "calibration_contract_sha": None,
            "status": "validated",
            "added_in": "v0.2.1",
        })
    ))
    rc = cmd_add_entry(argparse.Namespace(
        **dict(vars(args), **{
            "package_dir": str(root),
            "model_id": "spark-x25-4b-abliterated",
            "source_sha": SPARK_SOURCE_SHA,
            "tokenizer_sha": _sha(Path(args.tokenizer_json)),
            "guard_path": "profiles/guard/guard-spark-x25-4b-abliterated-exact-v1.yaml",
            "manifest_src": str(manifest_src),
            "record_src": str(records / f"{SPARK_SOURCE_SHA}.json"),
            "calibration_basis": LEGACY_BASIS,
            "calibration_contract_sha": None,
            "status": "validated",
            "added_in": "v0.2.1",
        })
    )) or rc

    locations = root / "registry" / "locations"
    locations.mkdir(parents=True, exist_ok=True)
    (locations / "reference-locations-v1.json").write_text(
        json.dumps({
            "schema": "fit.reference_locations.v1",
            "note": "non-trust transport locator layer; updating or mirroring this file never changes entry trust identity",
            "locations": [],
        }, indent=1) + "\n",
        encoding="utf-8",
    )
    print("bootstrap done")
    return rc


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    m = sub.add_parser("make-spark-manifest")
    m.add_argument("--out", default=str(REPO / "experiments/2026-09-06-spark-x25-4tier/manifests/reference-manifest-spark.json"))
    m.add_argument("--kld-shas", default="/tmp/spark-kld-shas.txt")
    m.add_argument("--tokenizer-json", default="/run/media/s117/OS/Models/Spark-X2.5-4B-abliterated/tokenizer.json")
    m.add_argument("--perplexity-binary", default="/home/s117/llama.cpp-spark-pr/build-rocm/bin/llama-perplexity")
    m.set_defaults(func=cmd_make_spark_manifest)

    a = sub.add_parser("add-entry")
    a.add_argument("--package-dir", required=True)
    a.add_argument("--model-id", required=True)
    a.add_argument("--source-sha", required=True)
    a.add_argument("--tokenizer-sha", default=None)
    a.add_argument("--guard-path", required=True, help="package-relative path")
    a.add_argument("--manifest-src", required=True)
    a.add_argument("--record-src", required=True)
    a.add_argument("--calibration-basis", required=True)
    a.add_argument("--calibration-contract-sha", default=None)
    a.add_argument("--status", default="candidate")
    a.add_argument("--added-in", required=True)
    a.set_defaults(func=cmd_add_entry)

    b = sub.add_parser("bootstrap")
    b.add_argument("--tokenizer-json", default="/run/media/s117/OS/Models/Spark-X2.5-4B-abliterated/tokenizer.json")
    b.add_argument("--perplexity-binary", default="/home/s117/llama.cpp-spark-pr/build-rocm/bin/llama-perplexity")
    b.set_defaults(func=cmd_bootstrap)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
