"""Generate real five-domain references and exercise Studio quality search.

Explicit local <=5B input only; CPU four threads, no model downloads. Reference
logits can consume several GiB of disk even for small models.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import time

from fit_gguf.calibrate import CalibrateConfig, stage_references
from fit_gguf.eval.contract import contract_digest
from fit_gguf.eval.provenance import sha256_file
from fit_gguf.llama_integration import runtime_env
from fit_gguf.studio.jobs import JobManager


def main():
    parser = argparse.ArgumentParser()
    for key in ("source", "imatrix", "runtime", "eval-data", "freeze", "workspace"):
        parser.add_argument("--" + key, required=True, type=Path)
    parser.add_argument("--tier", default="balanced")
    args = parser.parse_args()
    args.workspace.mkdir(parents=True, exist_ok=True)
    manager = JobManager(args.workspace, 5)
    manager._check_source(args.source)
    if shutil.disk_usage(args.workspace).free < 12 * 1024**3:
        raise RuntimeError("Real reference smoke requires at least 12 GiB free disk")
    cfg = CalibrateConfig(source=args.source.resolve(), imatrix_corpus=args.eval_data / "kl-eval-64k.txt",
                          runtime_dir=args.runtime.resolve(), eval_data_dir=args.eval_data.resolve(),
                          out_dir=args.workspace.resolve(), model_id=args.source.stem,
                          threads=4, n_gpu_layers=0, log_dir=args.workspace.resolve() / "reference-logs")
    refs = args.workspace.resolve() / "references"
    domains = stage_references(cfg, runtime_env(cfg.runtime_dir), refs)
    manifest = args.workspace.resolve() / "reference-manifest.json"
    manifest.write_text(json.dumps({"manifest_schema": "fit.eval_reference_manifest.v1",
        "source_bf16_gguf_sha256": sha256_file(args.source), "evaluator_contract_hash": contract_digest(),
        "domains": domains}, indent=2), encoding="utf-8")
    try:
        job = manager.submit("quality", {"source": str(args.source.resolve()), "imatrix": str(args.imatrix.resolve()),
            "runtime": str(args.runtime.resolve()), "refs_dir": str(refs), "eval_data_dir": str(cfg.eval_data_dir),
            "freeze": str(args.freeze.resolve()), "reference_manifest": str(manifest), "tier": args.tier, "threads": 4})
        deadline = time.monotonic() + 1800
        while manager.active and time.monotonic() < deadline:
            time.sleep(.5)
        if manager.active:
            raise TimeoutError("Quality smoke exceeded thirty minutes")
        result = manager.snapshot(job["id"])
        report = args.workspace / "quality-smoke-report.json"
        report.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
        if result["status"] != "succeeded":
            raise RuntimeError(f"Quality smoke did not deliver; inspect {report}")
    finally:
        manager.close()


if __name__ == "__main__":
    main()
