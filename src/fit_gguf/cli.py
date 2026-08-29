"""The fit command-line interface: analyze, plan, quantize."""

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
        )
        print(f"quantized: {record['output_path']} ({record['size_bytes']:,} bytes)")
        print(f"sha256: {record['sha256']}")
        return 0
    raise PipelineError(f"Unknown command: {args.command}")


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        return _run(args)
    except PipelineError as error:
        print(f"fit: error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
