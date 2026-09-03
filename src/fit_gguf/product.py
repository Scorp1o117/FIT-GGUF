"""FIT v0.2 public product path: fidelity-tier GGUF generation.

``fit fidelity-search --source model.gguf --tier balanced`` walks the full
GPT-ruled chain (planner-verdict-p1.md):

    resolve Guard Profile -> Fidelity Search (coarse/bracket/verify)
    -> quantize -> eval-v1 -> verify -> G2 finalize -> final artifact

Public semantics pinned by the ruling (do not change without G0):

* **Normal** budget <= 8 fresh evals, **Precise** <= 16;
* point **cache**: prior evaluated points (manifest + logs) enter as
  budget-free seeds; within a run, delivered sizes are never re-evaluated;
* **run-id**: every artifact/plan/eval name carries a unique per-run stamp —
  names are never reused across runs;
* **active constraint**: each answer reports whether KL or Same-top bound;
* **healthy frontier**: poison presets (Q3_K_S outlier, IQ2_XS poison) are
  never window lower bounds and never bracket evidence;
* **NOT REACHABLE**: a tier whose contract cannot be satisfied inside the
  healthy frontier fails with an explicit status instead of a guess.
"""

from __future__ import annotations

import json
from pathlib import Path

from fit_gguf.eval.provenance import (
    EvalProvenanceError,
    sha256_file,
    verify_eval_v1_provenance,
)
from fit_gguf.fidelity import GuardProfileError
from fit_gguf.fidelity_runner import (

    DOMAINS,
    POISON_PRESETS,
    RunnerConfig,
    SearchExecutor,
    discover_windows,
    load_seeds,
    resolve_contract,
    run_tier_search,
)
from fit_gguf.pipeline import PipelineError, analyze as pipeline_analyze
from fit_gguf.pipeline import plan as pipeline_plan
from fit_gguf.pipeline import quantize as pipeline_quantize

BUDGETS = {"normal": 8, "precise": 16}


class ProductError(ValueError):
    """Raised when the product path cannot deliver a tier artifact."""


def expand_preset_ladder(
    ladder: list[str],
    *,
    source: str,
    imatrix: str,
    runtime: str,
    analyses_dir: str | Path,
    imatrix_arg: str | None = None,
    hash_sources: bool = False,
) -> list[Path]:
    """Analyze adjacent preset pairs (ascending-size order) and return windows.

    The ladder is caller-ordered smallest to largest. Adjacent pairs tile the
    size axis; a pair whose lower preset is on the poison list is skipped
    entirely (healthy-frontier rule) — its size region then has no window and
    a crossing there reports NOT REACHABLE instead of a poisoned recipe.
    Existing analysis directories are reused (idempotent).
    """
    if len(ladder) < 2:
        raise ProductError("preset ladder needs at least two presets")
    root = Path(analyses_dir)
    root.mkdir(parents=True, exist_ok=True)
    windows: list[Path] = []
    for lower, upper in zip(ladder, ladder[1:]):
        if lower.upper() in POISON_PRESETS:
            continue
        out_dir = root / f"analysis-{lower}-{upper}"
        if not (out_dir / "analysis.json").is_file():
            pipeline_analyze(
                source,
                imatrix,
                runtime,
                str(out_dir),
                lower_preset=lower,
                upper_preset=upper,
                imatrix_arg=imatrix_arg,
                hash_sources=hash_sources,
            )
        windows.append(out_dir / "analysis.json")
    return windows


def _plan_exact_size(
    analysis_path: Path,
    target_size: int,
    upper_bound: int,
    *,
    model_name: str,
    refine_profile: str | None,
    out_prefix: Path,
    max_steps: int = 40,
) -> Path:
    """Find a byte target whose plan delivers exactly ``target_size``.

    plan() is a monotone piecewise-constant step function of the requested
    target (predicted <= requested, bounded by the full upper recipe), so a
    previously delivered size is reproducible by bisecting targets between
    the size itself and the window top until the step boundary is found.
    """
    low = target_size  # invariant: f(low) < target_size (plan undershoots at its own size)
    high = max(upper_bound, target_size + 1)  # invariant: f(high) >= target_size

    def _try_plan(target: int) -> int | None:
        try:
            record = pipeline_plan(
                analysis_path,
                out_prefix,
                target_bytes=target,
                policy="balanced",
                model_name=model_name,
                refine_profile=refine_profile,
            )
        except PipelineError:  # target outside this window's plan range
            return None
        return int(record["predicted_size_bytes"])

    for _ in range(max_steps):
        if high - low <= 1:
            break
        mid = (low + high) // 2
        predicted = _try_plan(mid)
        if predicted is None:
            high = mid
            continue
        if predicted == target_size:
            return out_prefix
        if predicted < target_size:
            low = mid
        else:
            high = mid
    predicted = _try_plan(high)
    if predicted == target_size:
        return out_prefix
    raise ProductError(
        f"exact size {target_size:,} is not deliverable by this window's candidate ladder"
    )


def fidelity_search_product(
    *,
    source: str,
    imatrix: str,
    runtime: str,
    refs_dir: str | Path,
    eval_data_dir: str | Path,
    guard_registry: str | Path,
    tier: str,
    out_dir: str | Path,
    work_dir: str | Path,
    manifest_path: str | Path,
    logs_dir: str | Path,
    model_name: str | None = None,
    preset_ladder: list[str] | None = None,
    analysis_dirs: list[str | Path] | None = None,
    refine_profile: str | Path | None = None,
    profile: str = "normal",
    tolerance_mib: int = 128,
    output: str | Path | None = None,
    threads: int = 16,
    imatrix_arg: str | None = None,
    hash_sources: bool = False,
    seed_prefix: str | None = None,
    exclude_seeds: list[str] | None = None,
    freeze_path: str | Path | None = None,
    reference_manifest_path: str | Path | None = None,
) -> dict:
    """Run the full product chain for one tier and produce the final GGUF."""
    if profile not in BUDGETS:
        raise ProductError(f"profile must be one of {sorted(BUDGETS)}, got {profile!r}")
    tier_key = tier.strip().lower()
    if model_name is None:
        # derive the guard identifier before resolving the contract, so the
        # README invocation (no --model-name) resolves the exact-model profile
        model_name = Path(source).stem
    # release-gate hardening FIRST: the product path evaluates ONLY against
    # the frozen eval-v1 closure (contract digest + reference .kld SHAs +
    # corpus SHAs) — "running with eval-v1 parameters" is not the verified
    # claim, and unverified inputs must fail before any expensive work
    if freeze_path is None:
        raise EvalProvenanceError(
            "freeze_path is required for the product path: pass the frozen "
            "eval-v1 FREEZE.json (canonical: experiments/2026-09-02-eval-v1/FREEZE.json)"
        )
    if reference_manifest_path is None:
        candidates = sorted(
            (Path(freeze_path).parent).glob("reference-manifest-*.json")
        )
        if len(candidates) == 1:
            reference_manifest_path = candidates[0]
        else:
            raise EvalProvenanceError(
                "reference_manifest_path is required (no unique "
                "reference-manifest-*.json found next to the freeze file); pass "
                "the manifest matching the model's bf16 references"
            )
    provenance = verify_eval_v1_provenance(
        refs_dir, eval_data_dir, freeze_path, reference_manifest_path
    )

    try:
        contract = resolve_contract(model_name, tier_key, guard_registry)
        source_sha256 = None
    except GuardProfileError:
        # the registry pins weights for this model — bind the guard to the
        # actual source GGUF before refusing (Codex audit: name-only binding
        # would let a same-named different-weights GGUF inherit floors)
        source_sha256 = sha256_file(source)
        contract = resolve_contract(model_name, tier_key, guard_registry, source_sha256)
    if source_sha256 is not None and provenance.source_bf16_gguf_sha256 != source_sha256:
        raise EvalProvenanceError(
            "guard weights binding disagrees with the reference manifest: "
            f"{source_sha256} != {provenance.source_bf16_gguf_sha256} — the "
            "validated floors and the references belong to different weights"
        )

    if analysis_dirs:
        analysis_paths = [Path(p) for p in analysis_dirs]
    elif preset_ladder:
        analysis_paths = expand_preset_ladder(
            preset_ladder,
            source=source,
            imatrix=imatrix,
            runtime=runtime,
            analyses_dir=Path(out_dir) / "analyses",
            imatrix_arg=imatrix_arg or imatrix,
            hash_sources=hash_sources,
        )
    else:
        raise ProductError("provide either preset_ladder or analysis_dirs")

    windows = discover_windows([str(p) for p in analysis_paths])
    healthy = [w for w in windows if w.healthy]
    if not healthy:
        raise ProductError("no healthy windows — tier is NOT REACHABLE")

    # historical manifests may use a shorter naming convention than the
    # guard identifier, so the seed prefix is explicit
    seeds = load_seeds(
        manifest_path,
        logs_dir,
        seed_prefix or f"{model_name}-",
        exclude_names=tuple(exclude_seeds or ()),
        # release path: every bracket seed must carry a provenance sidecar
        # attesting the same frozen eval closure AND the same reference
        # manifest (backfill via scripts/backfill_seed_provenance.py)
        require_seed_provenance=True,
        reference_manifest_sha256=provenance.reference_manifest_file_sha256,
    )
    config = RunnerConfig(
        runtime=Path(runtime),
        imatrix=Path(imatrix),
        refs_dir=Path(refs_dir),
        eval_data_dir=Path(eval_data_dir),
        work_dir=Path(work_dir),
        out_dir=Path(out_dir),
        model_name=model_name,
        guard_registry=Path(guard_registry),
        refine_profile=refine_profile,
        threads=threads,
        eval_provenance=provenance,
        require_eval_provenance=True,
        source_sha256=source_sha256,
    )
    result = run_tier_search(
        contract,
        config,
        windows,
        seeds,
        min_size=min(w.lower_size for w in healthy),
        max_size=max(w.upper_size for w in healthy),
        budget=BUDGETS[profile],
        tolerance_bytes=tolerance_mib * 1024 * 1024,
        manifest_path=Path(manifest_path),
        logs_out_dir=Path(logs_dir),
    )
    summary = result
    if summary["status"] != "verified_pass" or summary.get("best") is None:
        summary["artifact"] = None
        if summary["status"] == "no_pass":
            summary["product_status"] = "NOT REACHABLE within the validated healthy frontier"
        summary["guard_binding"] = {
            "model_name": model_name,
            "source_sha256": source_sha256,
        }
        result_path = Path(out_dir) / f"fidelity-search-{tier_key}-product.json"
        result_path.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
        return summary

    best_size = summary["best"]["size_bytes"]
    recipe = (result.get("artifact_recipes") or {}).get(str(best_size))
    analysis_dir = Path((result.get("artifact_analyses") or {}).get(str(best_size))
                        or _best_analysis(best_size, windows))
    if recipe:
        # the search itself produced this artifact's recipe — reuse verbatim
        final_prefix = None
        tensor_types = Path(recipe)
    else:
        # seed-born answer: reproduce the recipe by bisecting plan targets
        window_upper = next(
            (w.upper_size for w in windows if Path(w.analysis_path) == analysis_dir),
            max(w.upper_size for w in healthy),
        )
        final_prefix = _plan_exact_size(
            analysis_dir / "analysis.json",
            best_size,
            window_upper,
            model_name=model_name,
            refine_profile=refine_profile,
            out_prefix=Path(out_dir) / "final-plan",
        )
        tensor_types = Path(f"{final_prefix}-tensor-types.txt")
    output_path = Path(output) if output else (
        Path(out_dir)
        / f"{model_name}-FIT-{tier_key.upper()}-{best_size / 1024**3:.2f}GiB.gguf"
    )
    record = pipeline_quantize(
        analysis_dir / "analysis.json",
        tensor_types,
        output_path,
        imatrix_arg=imatrix_arg or imatrix,
    )

    # chain step "eval -> verify" on the deliverable itself: the artifact must
    # satisfy the contract on its own bytes, not only on the search probe
    verifier = SearchExecutor(
        config,
        windows,
        contract,
        Path(manifest_path),
        Path(logs_dir),
    )
    verify_tag = f"{model_name}-final-verify-{tier_key}"
    metrics = verifier._eval_domains(output_path, verify_tag)
    if metrics is None:
        raise ProductError("final artifact verification eval failed")
    macro_kl = sum(metrics[d]["mean_kld"] for d in DOMAINS) / len(DOMAINS)
    macro_top = sum(metrics[d]["same_top_pct"] for d in DOMAINS) / len(DOMAINS)
    if not contract.passes(macro_kl, macro_top / 100.0):
        raise ProductError(
            f"final artifact violates the {tier_key} contract: "
            f"kld {macro_kl:.4f} top {macro_top:.2f}"
        )
    verify_manifest = Path(manifest_path)
    if not _manifest_has(verify_manifest, verify_tag):
        import hashlib

        digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
        with verify_manifest.open("a", encoding="utf-8") as handle:
            handle.write(f"{verify_tag}  {record['size_bytes']}  {digest}\n")

    summary["guard_binding"] = {
        "model_name": model_name,
        "source_sha256": source_sha256,
    }
    summary["artifact"] = {
        "path": str(record["output_path"]),
        "size_bytes": int(record["size_bytes"]),
        "g2_delta": int(record["size_bytes"]) - int(record["refinalized_expected_bytes"]),
        "naming": f"{model_name}-FIT-{tier_key.upper()}-<size>-<primary-qtype>.gguf",
        "search_tolerance_mib": tolerance_mib,
        "active_constraint": summary["active_constraint"],
        "healthy_frontier": True,
        "verified_metrics": {"macro_kl": macro_kl, "same_top_pct": macro_top},
    }
    result_path = Path(out_dir) / f"fidelity-search-{tier_key}-product.json"
    result_path.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    return summary


def _manifest_has(manifest_path: Path, name: str) -> bool:
    if not manifest_path.is_file():
        return False
    return any(line.split()[:1] == [name] for line in manifest_path.read_text().splitlines())


def _best_analysis(best_size: int, windows) -> Path:
    """Window whose range contains the best verified size (for re-quantize)."""
    for window in windows:
        if window.lower_size <= best_size <= window.upper_size:
            return window.analysis_path
    raise ProductError(f"best size {best_size:,} outside every window")
