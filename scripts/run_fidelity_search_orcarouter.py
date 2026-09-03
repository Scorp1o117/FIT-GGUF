#!/usr/bin/env python3
"""P1: orcarouter four-tier Fidelity Search v1 (minimum verified PASS per tier).

Driver for the GPT-ruled v0.2 flagship solver (planner-verdict-m6b.md):
  healthy-frontier windows (Q3_K_S poison lower bound excluded),
  budget Normal <= 8 fresh evals per tier, tolerance 128 MiB,
  answer = minimum verified PASS with active-constraint attribution.

Seeds are the already-evaluated orcarouter points (budget-free). Poison-window
products (built from the Q3_K_S floor) are explicitly excluded from bracket
evidence even though their names look clean.
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from fit_gguf.eval.provenance import sha256_file  # noqa: E402
from fit_gguf.fidelity_runner import (  # noqa: E402
    RunnerConfig,
    discover_windows,
    load_seeds,
    resolve_contract,
    run_tier_search,
)

M2 = REPO / "experiments" / "2026-09-02-m2-topkl-calibration"
MODEL = "orcarouter-Qwen3.8-27B-Uncensored"
RT = REPO / "tools" / "llama-b10666-rocm"
IMX = REPO / "imatrix_unsloth.gguf"
REFS = REPO / "experiments" / "2026-09-02-eval-v1" / "refs-orcarouter"
EVAL_DATA = REPO / "eval-data"
GUARD_REGISTRY = REPO / "profiles" / "guard"
REFINE_PROFILE = REPO / "profiles" / "refine-profile-qwen-hybrid-band-v1.json"

ANALYSES = [
    M2 / "orcarouter-analysis-Q4_K_M-Q5_K_M",
    M2 / "orcarouter-analysis-IQ3_M-IQ4_XS",
    M2 / "orcarouter-analysis-IQ3_XS-IQ3_S",
    M2 / "orcarouter-analysis-Q2_K-IQ3_XXS",
    M2 / "orcarouter-analysis-Q3_K_S-IQ3_S",  # poison window: never selected
]

# Poison-window products: clean names, misleading FAILs (Q3_K_S floor recipes).
POISON_WINDOW_PRODUCTS = ("FIT-V2-Compact", "FIT-A-IQ3S-v01", "FIT-A-IQ3S-v02")

TIERS = ("quality", "balanced", "compact", "mini")


def main() -> int:
    dry_run = "--dry-run" in sys.argv
    windows = discover_windows([str(p) for p in ANALYSES])
    print("healthy windows:")
    for window in windows:
        marker = "" if window.healthy else "  (POISON — excluded)"
        print(f"  {window.lower_preset}->{window.upper_preset} "
              f"[{window.lower_size:,}, {window.upper_size:,}]{marker}")

    seeds = load_seeds(
        M2 / "results" / "artifact-manifest.txt",
        M2 / "logs",
        "orcarouter-",
        exclude_names=POISON_WINDOW_PRODUCTS,
    )
    print(f"\nseeds: {len(seeds)} observed points")
    for seed in sorted(seeds, key=lambda s: -s.size_bytes):
        print(f"  {seed.size_bytes / 1024**3:6.2f}G  kld {seed.macro_kl:.4f}  top {seed.same_top:.4f}")

    if dry_run:
        print("\n(dry run — no search executed)")
        return 0

    # guard profiles pin the source weights digest — bind before resolving
    source_sha = sha256_file(SRC)
    for tier in TIERS:
        contract = resolve_contract(MODEL, tier, GUARD_REGISTRY, source_sha)
        print(f"\n===== tier {tier}: KL<={contract.kl_anchor} top>={contract.same_top_floor} =====")
        work = Path("/dev/shm") / "orca-fs" / tier
        config = RunnerConfig(
            runtime=RT,
            imatrix=IMX,
            refs_dir=REFS,
            eval_data_dir=EVAL_DATA,
            work_dir=work,
            out_dir=M2 / "results-fs" / tier,
            model_name=MODEL,
            guard_registry=GUARD_REGISTRY,
            refine_profile=REFINE_PROFILE,
        )
        result = run_tier_search(
            contract,
            config,
            windows,
            seeds,
            min_size=min(w.lower_size for w in windows if w.healthy),
            max_size=max(w.upper_size for w in windows if w.healthy),
            budget=8,
            manifest_path=M2 / "results" / "artifact-manifest.txt",
            logs_out_dir=M2 / "logs",
        )
        summary = result
        best = summary["best"]
        print(f"----- {tier}: {summary['status']} | best "
              f"{(best['size_bytes'] / 1024**3 if best else float('nan')):.2f}G "
              f"kld {best['macro_kl'] if best else float('nan'):.4f} "
              f"top {best['same_top'] if best else float('nan'):.4f} | "
              f"fresh evals {summary['fresh_evals']}/{summary['budget']} | "
              f"active: {summary['active_constraint']} | {summary['note'] or ''}")

    print("\n===== FIDELITY SEARCH COMPLETE =====")
    return 0


if __name__ == "__main__":
    sys.exit(main())
