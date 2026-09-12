#!/usr/bin/env python3
"""Evaluate arbitrary GGUF artifacts with the FIT eval-v1 protocol.

Runs the same five-domain KL evaluation the FIT tier search uses — same binary
resolution, same slice files, same BF16 `.kld` references, same parser — so an
APEX tier and a FIT tier are measured by identical machinery and their numbers
are directly comparable.

    python scripts/eval_artifact.py --artifact <a.gguf> [--artifact <b.gguf> ...] \
        --out results.json

Macro KL is the mean of the five per-domain `mean_kld` values, matching
`fidelity_runner` (macro_kl = sum(domains[d]["mean_kld"]) / len(DOMAINS)).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from fit_gguf.eval.contract import DOMAINS  # noqa: E402
from fit_gguf.fidelity_runner import (  # noqa: E402
    _SLICE_SUFFIX,
    parse_llama_kl_log,
)
from fit_gguf.llama_integration import (  # noqa: E402
    resolve_runtime_binary,
    runtime_env,
)


def eval_domains(
    artifact: Path,
    runtime: Path,
    eval_data: Path,
    refs: Path,
    log_dir: Path,
    n_gpu_layers: int,
    threads: int,
) -> dict[str, dict]:
    metrics: dict[str, dict] = {}
    for domain in DOMAINS:
        log_path = log_dir / f"eval-{artifact.stem[:40]}-{domain}.log"
        slice_file = eval_data / f"kl-eval-{_SLICE_SUFFIX[domain]}"
        ref_file = refs / f"bf16-{domain}.kld"
        for attempt in (1, 2, 3):
            result = subprocess.run(
                [
                    str(resolve_runtime_binary(runtime, "llama-perplexity")),
                    "-m", str(artifact),
                    "-f", str(slice_file),
                    "-ngl", str(n_gpu_layers),
                    "-t", str(threads),
                    "-c", "512", "-b", "512",
                    "--kl-divergence",
                    "--kl-divergence-base", str(ref_file),
                ],
                capture_output=True,
                timeout=3600,
                env=runtime_env(runtime),
            )
            combined = (result.stdout + result.stderr).decode("utf-8", errors="replace")
            log_path.write_text(combined, encoding="utf-8")
            try:
                parsed = parse_llama_kl_log(combined)
            except Exception:  # noqa: BLE001
                print(f"    {domain}: attempt {attempt} unparsable (rc={result.returncode})", flush=True)
                time.sleep(10 * attempt)
                continue
            if result.returncode != 0:
                print(f"    {domain}: parsed but rc={result.returncode}, retrying", flush=True)
                time.sleep(10 * attempt)
                continue
            metrics[domain] = parsed
            print(f"    {domain}: kld={parsed['mean_kld']:.6f} top={parsed['same_top_pct']:.2f}%", flush=True)
            break
        else:
            raise RuntimeError(f"{artifact.name}: domain {domain} failed after 3 attempts")
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", action="append", required=True)
    ap.add_argument("--runtime", default="tools/llama-b10666-rocm")
    ap.add_argument("--eval-data", default="eval-data")
    ap.add_argument("--refs", default="experiments/2026-09-11-nex25-mini-4tier/calibration/references")
    ap.add_argument("--logs", default="/home/s117/fit-logs/nex25-apex-eval")
    ap.add_argument("--n-gpu-layers", type=int, default=30)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    log_dir = Path(args.logs)
    log_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for spec in args.artifact:
        art = Path(spec)
        if not art.is_file():
            raise SystemExit(f"missing artifact {art}")
        size = art.stat().st_size
        print(f"  {art.name} ({size / 2**30:.2f} GiB)", flush=True)
        metrics = eval_domains(
            art, Path(args.runtime), Path(args.eval_data), Path(args.refs),
            log_dir, args.n_gpu_layers, args.threads,
        )
        macro_kl = sum(metrics[d]["mean_kld"] for d in DOMAINS) / len(DOMAINS)
        macro_top = sum(metrics[d]["same_top_pct"] for d in DOMAINS) / len(DOMAINS)
        print(f"    macro_kl={macro_kl:.6f} same_top={macro_top:.2f}%", flush=True)
        records.append({
            "name": art.name,
            "path": str(art),
            "size_bytes": size,
            "macro_kl": macro_kl,
            "same_top_pct": macro_top,
            "per_domain": metrics,
        })
    Path(args.out).write_text(json.dumps({
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "fit eval-v1 five-domain KL (llama-perplexity --kl-divergence vs bf16-<domain>.kld)",
        "runtime": args.runtime,
        "n_gpu_layers": args.n_gpu_layers,
        "threads": args.threads,
        "artifacts": records,
    }, indent=2), encoding="utf-8")
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
