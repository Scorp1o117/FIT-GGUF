#!/usr/bin/env python3
"""Spark-X2.5-4B-abliterated (T615): four-tier Fidelity Search v1.

Calibration-run variant of run_fidelity_search_orcarouter.py: identical
search semantics (healthy windows, poison exclusion, seeds, budget <=8,
tolerance 128 MiB, G2 exact-size) but WITHOUT the product-CLI provenance
attestation — the eval-v1 freeze document pins the orcarouter reference
manifest SHA prefix, which cannot cover a second model. Extending the
freeze to Spark is a planner decision; until then this run is governed as
an M2-style model calibration (validation_basis=dev_calibration).

Differences vs the orcarouter flagship driver:
  * runtime = PR-branch build (llama.cpp #27868 spark2_5 support);
    perplexity.cpp is byte-identical to b10666 (diffed 2026-09-06).
  * refine_profile=None -> v0.1 balanced allocator (the Qwen band cells
    are Qwen-family calibration data, inapplicable to spark2_5).
  * guard profile from the Spark preset-ladder calibration (same run).
"""

import sys
from pathlib import Path

REPO = Path("/run/media/s117/OS/FIT-GGUF")
sys.path.insert(0, str(REPO / "src"))

from fit_gguf.eval.provenance import sha256_file  # noqa: E402
from fit_gguf.fidelity_runner import (  # noqa: E402
    RunnerConfig,
    discover_windows,
    load_seeds,
    resolve_contract,
    run_tier_search,
)

EXP = REPO / "experiments" / "2026-09-06-spark-x25-4tier"
MODEL = "spark-x25-4b-abliterated"
RT = Path("/home/s117/llama.cpp-spark-pr/build-rocm/bin")
IMX = EXP / "spark-imatrix.gguf"
REFS = Path("/dev/shm/spark/refs")  # tmpfs during the run; master at EXP/refs
EVAL_DATA = REPO / "eval-data"
GUARD_REGISTRY = REPO / "src" / "fit_gguf" / "profiles" / "guard"
SRC = Path("/run/media/s117/OS/Models/Spark-X2.5-4B-abliterated-FIT-GGUF/Spark-X2.5-4B-abliterated-BF16.gguf")
MANIFEST = EXP / "state" / "artifact-manifest.txt"
LOGS = EXP / "logs"

ANALYSES = [
    EXP / "analysis" / "spark-analysis-Q5_K_M-Q6_K",
    EXP / "analysis" / "spark-analysis-Q4_K_M-Q5_K_S",
    EXP / "analysis" / "spark-analysis-IQ4_XS-Q4_K_S",
]

TIERS = ("quality", "balanced", "compact", "mini")


def main() -> int:
    dry_run = "--dry-run" in sys.argv
    windows = discover_windows([str(p) for p in ANALYSES])
    print("healthy windows:")
    for window in windows:
        marker = "" if window.healthy else "  (POISON — excluded)"
        print(f"  {window.lower_preset}->{window.upper_preset} "
              f"[{window.lower_size:,}, {window.upper_size:,}]{marker}")

    seeds = load_seeds(MANIFEST, LOGS, MODEL + "-")
    print(f"\nseeds: {len(seeds)} observed points")
    for seed in sorted(seeds, key=lambda s: -s.size_bytes):
        print(f"  {seed.size_bytes / 1024**3:6.2f}G  kld {seed.macro_kl:.4f}  top {seed.same_top:.4f}")

    if dry_run:
        print("\n(dry run — no search executed)")
        return 0

    source_sha = sha256_file(SRC)
    for tier in TIERS:
        contract = resolve_contract(MODEL, tier, GUARD_REGISTRY, source_sha)
        print(f"\n===== tier {tier}: KL<={contract.kl_anchor} top>={contract.same_top_floor} =====")
        config = RunnerConfig(
            runtime=RT,
            imatrix=IMX,
            refs_dir=REFS,
            eval_data_dir=EVAL_DATA,
            work_dir=Path("/dev/shm") / "spark-fs" / tier,
            out_dir=EXP / "results-fs" / tier,
            model_name=MODEL,
            guard_registry=GUARD_REGISTRY,
            refine_profile=None,
            source_sha256=source_sha,
        )
        result = run_tier_search(
            contract,
            config,
            windows,
            seeds,
            min_size=min(w.lower_size for w in windows if w.healthy),
            max_size=max(w.upper_size for w in windows if w.healthy),
            budget=8,
            manifest_path=MANIFEST,
            logs_out_dir=LOGS,
        )
        best = result["best"]
        print(f"----- {tier}: {result['status']} | best "
              f"{(best['size_bytes'] / 1024**3 if best else float('nan')):.2f}G "
              f"kld {best['macro_kl'] if best else float('nan'):.4f} "
              f"top {best['same_top'] if best else float('nan'):.4f} | "
              f"fresh evals {result['fresh_evals']}/{result['budget']} | "
              f"active: {result['active_constraint']} | {result['note'] or ''}")

    print("\n===== FIDELITY SEARCH COMPLETE =====")
    return 0


if __name__ == "__main__":
    sys.exit(main())
