"""The fit command-line interface: analyze, plan, quantize, fidelity-search."""

from __future__ import annotations

import argparse
import sys

from fit_gguf.pipeline import (
    FIT_GGUF_VERSION,
    PipelineError,
    analyze,
    plan,
    quantize,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fit",
        description=(
            "Fit-to-Size Intelligent Tensor Quantization: deterministic "
            "size-exact GGUF recipes between llama.cpp presets."
        ),
    )
    parser.add_argument("--version", action="version", version=f"fit {FIT_GGUF_VERSION}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze_parser = subparsers.add_parser(
        "analyze", help="Profile a source/imatrix pair and freeze the candidate set"
    )
    analyze_parser.add_argument("--source", required=True, help="Source (BF16) GGUF")
    analyze_parser.add_argument("--imatrix", required=True, help="Importance-matrix GGUF")
    analyze_parser.add_argument(
        "--runtime", required=True, help="Directory containing the pinned llama-quantize"
    )
    analyze_parser.add_argument(
        "--out-dir", required=True, help="Directory for analysis.json, profile.json, dry-run logs"
    )
    analyze_parser.add_argument("--lower", default="IQ3_M", help="Lower preset (default IQ3_M)")
    analyze_parser.add_argument("--upper", default="IQ4_XS", help="Upper preset (default IQ4_XS)")
    analyze_parser.add_argument(
        "--imatrix-arg",
        default=None,
        help="Imatrix path string llama-quantize will receive (default: basename)",
    )
    analyze_parser.add_argument(
        "--skip-hash", action="store_true", help="Skip SHA-256 of source and imatrix"
    )

    plan_parser = subparsers.add_parser(
        "plan", help="Plan one size-exact artifact from a frozen analysis"
    )
    plan_parser.add_argument("--analysis", required=True, help="analysis.json from fit analyze")
    plan_parser.add_argument(
        "--fit", default=None, help="Fraction of the preset gap, e.g. 0.5"
    )
    plan_parser.add_argument(
        "--target-bytes", type=int, default=None, help="Explicit byte target"
    )
    plan_parser.add_argument(
        "--policy",
        default="original",
        choices=("original", "balanced", "random"),
        help="original = v0.1a greedy utility; balanced = v0.1b block-balanced; random = baseline",
    )
    plan_parser.add_argument("--seed", default=None, help="Seed for --policy random")
    plan_parser.add_argument(
        "--block-span",
        default="auto",
        help="Block-quarter span for --policy balanced (default: auto)",
    )
    plan_parser.add_argument(
        "--model-name",
        default=None,
        help="Model name for the suggested filename (default: derived from the source)",
    )
    plan_parser.add_argument(
        "--refine-profile",
        default=None,
        help="Refine Profile JSON: re-weight candidate utility by C_role (v0.2)",
    )
    plan_parser.add_argument(
        "--fidelity-tier",
        default=None,
        help="Claim a Fidelity tier (quality|balanced|compact|mini); requires a validated Guard Profile",
    )
    plan_parser.add_argument(
        "--guard-registry",
        default=None,
        help="Guard Profile registry directory (default: profiles/guard)",
    )
    plan_parser.add_argument(
        "--out-prefix", required=True, help="Prefix for -plan.json/-recipe.json/-tensor-types.txt"
    )

    quantize_parser = subparsers.add_parser(
        "quantize", help="Quantize per a tensor-types file and verify the exact size"
    )
    quantize_parser.add_argument("--analysis", required=True, help="analysis.json from fit analyze")
    quantize_parser.add_argument("--tensor-types", required=True, help="Tensor-types file from fit plan")
    quantize_parser.add_argument("--out", required=True, help="Output GGUF path")
    quantize_parser.add_argument(
        "--expect-bytes", type=int, default=None, help="Required exact output size"
    )
    quantize_parser.add_argument(
        "--imatrix-arg",
        default=None,
        help="Actual imatrix path string llama-quantize receives (default: analysis arg)",
    )

    fs_parser = subparsers.add_parser(
        "fidelity-search",
        help=(
            "Fidelity-tier product path: search the minimum verified GGUF "
            "satisfying KL <= anchor AND Same-top >= Guard floor, then emit "
            "the exact-byte artifact (v0.2)"
        ),
    )
    fs_parser.add_argument("--source", required=True, help="Source (BF16) GGUF")
    fs_parser.add_argument("--imatrix", required=True, help="Importance-matrix GGUF")
    fs_parser.add_argument(
        "--runtime", required=True, help="Directory containing the pinned llama.cpp binaries"
    )
    fs_parser.add_argument("--refs-dir", required=True, help="Directory of bf16-<domain>.kld references")
    fs_parser.add_argument("--eval-data-dir", required=True, help="Directory of kl-eval domain slices")
    fs_parser.add_argument(
        "--guard-registry", required=True, help="Guard Profile registry directory"
    )
    fs_parser.add_argument(
        "--tier", required=True, choices=("quality", "balanced", "compact", "mini"),
        help="Fidelity tier to satisfy",
    )
    fs_parser.add_argument(
        "--preset-ladder",
        default=None,
        help=(
            "Comma-separated presets in ascending-size order; adjacent pairs are "
            "auto-analyzed (poison presets skipped as lower bounds)"
        ),
    )
    fs_parser.add_argument(
        "--analysis",
        action="append",
        default=None,
        help="Pre-frozen analysis directory (repeatable; alternative to --preset-ladder)",
    )
    fs_parser.add_argument(
        "--refine-profile", default=None, help="Refine Profile JSON (v0.2 band-conditional)"
    )
    fs_parser.add_argument(
        "--profile",
        default="normal",
        choices=("normal", "precise"),
        help="Search budget: normal <= 8 fresh evals, precise <= 16 (default normal)",
    )
    fs_parser.add_argument(
        "--tolerance-mib", type=int, default=128, help="Search bracket tolerance in MiB (default 128)"
    )
    fs_parser.add_argument("--model-name", default=None, help="Guard Profile exact-model identifier")
    fs_parser.add_argument("--out-dir", required=True, help="Records, analyses, final artifact")
    fs_parser.add_argument("--work-dir", required=True, help="Scratch directory (tmpfs recommended)")
    fs_parser.add_argument("--manifest", required=True, help="Artifact manifest (appended)")
    fs_parser.add_argument("--logs-dir", required=True, help="Eval log directory (seeds are read from here)")
    fs_parser.add_argument(
        "--output", default=None, help="Final artifact path (default: <model>-FIT-<TIER>-<size>GiB.gguf)"
    )
    fs_parser.add_argument(
        "--threads", type=int, default=16, help="Evaluator threads (default 16)"
    )
    fs_parser.add_argument(
        "--seed-prefix",
        default=None,
        help="Manifest/log name prefix for prior points (default: <model-name>-)",
    )
    fs_parser.add_argument(
        "--exclude-seed",
        action="append",
        default=None,
        help="Prior point name to exclude from bracket evidence (repeatable; e.g. poison-window products)",
    )
    fs_parser.add_argument(
        "--skip-hash", action="store_true", help="Skip SHA-256 of source and imatrix during auto-analyze"
    )
    fs_parser.add_argument(
        "--freeze",
        default="experiments/2026-09-02-eval-v1/FREEZE.json",
        help=(
            "Frozen eval-v1 FREEZE.json pinning the contract digest "
            "(default: canonical repo path relative to the working directory)"
        ),
    )
    fs_parser.add_argument(
        "--reference-manifest",
        default=None,
        help=(
            "Reference manifest JSON pinning the bf16 .kld and corpus SHA-256s "
            "(default: the unique reference-manifest-*.json next to --freeze)"
        ),
    )
    return parser


def _run(args: argparse.Namespace) -> int:
    if args.command == "analyze":
        analysis_path = analyze(
            args.source,
            args.imatrix,
            args.runtime,
            args.out_dir,
            lower_preset=args.lower,
            upper_preset=args.upper,
            imatrix_arg=args.imatrix_arg,
            hash_sources=not args.skip_hash,
        )
        print(f"analysis written: {analysis_path}")
        return 0

    if args.command == "plan":
        record = plan(
            args.analysis,
            args.out_prefix,
            fit=args.fit,
            target_bytes=args.target_bytes,
            policy=args.policy,
            seed=args.seed,
            block_span=args.block_span,
            model_name=args.model_name,
            refine_profile=args.refine_profile,
            fidelity_tier=args.fidelity_tier,
            guard_registry=args.guard_registry,
        )
        print(
            f"plan written: policy={record['policy']} target={record['target_bytes']:,} "
            f"predicted={record['predicted_size_bytes']:,} unused={record['unused_bytes']:,} "
            f"selected={record['selected_count']}"
        )
        if record["dominant_qtype"]:
            share = record["qtype_parameter_shares"][record["dominant_qtype"]]
            print(
                f"dominant qtype: {str(record['dominant_qtype']).upper()} "
                f"({share:.1%} of quantized parameters)"
            )
        if record["suggested_filename"]:
            print(f"suggested filename: {record['suggested_filename']}")
        print(f"plan record: {record['recipe_path'].rsplit('-', 1)[0]}-plan.json")
        return 0

    if args.command == "quantize":
        record = quantize(
            args.analysis,
            args.tensor_types,
            args.out,
            expect_bytes=args.expect_bytes,
            imatrix_arg=args.imatrix_arg,
        )
        print(f"quantized: {record['output_path']} ({record['size_bytes']:,} bytes)")
        plan_target = (
            f"{record['expect_bytes']:,}" if record["expect_bytes"] is not None else "n/a"
        )
        print(
            f"G2: plan target {plan_target} | finalized "
            f"{record['refinalized_expected_bytes']:,} | actual "
            f"{record['size_bytes']:,} | delta "
            f"{record['size_bytes'] - record['refinalized_expected_bytes']:+,} | "
            f"{'PASS' if record['size_matches_refinalization'] else 'FAIL'}"
        )
        print(f"sha256: {record['sha256']}")
        return 0

    if args.command == "fidelity-search":
        from fit_gguf.product import fidelity_search_product

        analysis_dirs = args.analysis if args.analysis else None
        ladder = args.preset_ladder.split(",") if args.preset_ladder else None
        if not analysis_dirs and not ladder:
            print("fit: error: provide --preset-ladder or --analysis", file=sys.stderr)
            return 2
        summary = fidelity_search_product(
            source=args.source,
            imatrix=args.imatrix,
            runtime=args.runtime,
            refs_dir=args.refs_dir,
            eval_data_dir=args.eval_data_dir,
            guard_registry=args.guard_registry,
            tier=args.tier,
            out_dir=args.out_dir,
            work_dir=args.work_dir,
            manifest_path=args.manifest,
            logs_dir=args.logs_dir,
            model_name=args.model_name,
            preset_ladder=ladder,
            analysis_dirs=analysis_dirs,
            refine_profile=args.refine_profile,
            profile=args.profile,
            tolerance_mib=args.tolerance_mib,
            output=args.output,
            threads=args.threads,
            imatrix_arg=args.imatrix,
            hash_sources=not args.skip_hash,
            seed_prefix=args.seed_prefix,
            exclude_seeds=args.exclude_seed,
            freeze_path=args.freeze,
            reference_manifest_path=args.reference_manifest,
        )
        best = summary.get("best")
        status = summary.get("status")
        if status == "verified_pass" and best:
            print(
                f"fidelity-search: {status} | tier {args.tier} | "
                f"minimum verified PASS {best['size_bytes'] / 1024**3:.2f}G "
                f"(kld {best['macro_kl']:.4f}, top {best['same_top'] * 100:.2f}%) | "
                f"active constraint: {summary['active_constraint']} | "
                f"{summary['fresh_evals']}/{summary['budget']} evals"
            )
            if summary.get("note"):
                print(f"note: {summary['note']}")
            artifact = summary.get("artifact")
            if artifact:
                print(f"artifact: {artifact['path']} ({artifact['size_bytes']:,} bytes, G2 delta {artifact['g2_delta']:+d})")
            return 0
        # Non-deliverable outcomes must be machine-detectable failures: a
        # present `best` observation alone (e.g. noise_inversion) is NOT a
        # delivery — the product layer withheld the artifact on purpose.
        if best:
            print(
                f"fidelity-search: {status} | tier {args.tier} | "
                f"best observed PASS {best['size_bytes'] / 1024**3:.2f}G "
                f"(kld {best['macro_kl']:.4f}, top {best['same_top'] * 100:.2f}%) | "
                f"active constraint: {summary['active_constraint']} | "
                f"{summary['fresh_evals']}/{summary['budget']} evals | "
                "NOT auto-delivered (artifact withheld; adjudication required)"
            )
        else:
            print(
                f"fidelity-search: {summary.get('product_status') or status} "
                f"(tier {args.tier})",
                file=sys.stderr,
            )
        if summary.get("note"):
            print(f"note: {summary['note']}")
        return {"no_pass": 3, "noise_inversion": 4, "budget_exhausted": 5}.get(status, 2)
    raise PipelineError(f"Unknown command: {args.command}")


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    # imported lazily so CLI startup stays cheap and the module set matches
    # the invoked subcommand
    from fit_gguf.eval.provenance import EvalProvenanceError
    from fit_gguf.fidelity import GuardProfileError
    from fit_gguf.product import ProductError

    try:
        return _run(args)
    except (PipelineError, GuardProfileError, ProductError, EvalProvenanceError) as error:
        print(f"fit: error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
