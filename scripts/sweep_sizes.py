#!/usr/bin/env python3
"""Plan → quantize → evaluate one or more exact sizes from a frozen analysis.

Built to A/B an allocation change at a FIXED size. `fit tier-search` bisects for
the smallest artifact that passes a gate, which cannot answer "same bytes,
different distribution, which KL?" — the old point is still the smallest PASS, so
the search stops and the new allocation is never measured.

An unreachable size is a RESULT, not a crash. Precision floors are mandatory
spending, so every preset window has a minimum achievable size that sits ABOVE
its lower preset; a target below it is reported as UNREACHABLE and the sweep
continues to the sizes that do fit.

    python scripts/sweep_sizes.py \\
        --analysis <analysis.json> --runtime tools/llama-b10666-rocm \\
        --eval-data eval-data --refs-dir /dev/shm/cal-<model>/references \\
        --sizes 13.8,14.1,14.3 --model-name <model> --imatrix-arg <path> \\
        --dest /path/to/out [--no-floors] [--tag sweep]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from fit_gguf.calibrate import (  # noqa: E402
    CalibrateConfig,
    _runtime_env,
    default_scratch_root,
    eval_artifact,
)
from fit_gguf.eval.provenance import sha256_file  # noqa: E402
from fit_gguf.pipeline import plan as pipeline_plan  # noqa: E402
from fit_gguf.pipeline import quantize as pipeline_quantize  # noqa: E402

GIB = 2**30


def run_one(args, ctx, size: float) -> dict:
    """Plan, quantize and evaluate one size. Raises if the window cannot reach it."""
    target = int(size * GIB)
    prefix = ctx["work"] / f"{args.tag}-{size:g}G"
    record = pipeline_plan(
        args.analysis, prefix, target_bytes=target,
        policy=args.policy, model_name=args.model_name,
        always_active_floors=args.floors,
    )
    artifact = ctx["work"] / f"{args.tag}-{size:g}G.gguf"
    pipeline_quantize(
        args.analysis, Path(f"{prefix}-tensor-types.txt"), artifact,
        imatrix_arg=args.imatrix_arg,
    )
    obs = eval_artifact(ctx["cfg"], artifact, args.refs_dir, f"{args.tag}-{size:g}G", ctx["env"])
    keep = args.dest / (
        f"{args.model_name}-{args.tag}-{'F' if args.floors else 'N'}-{size:g}G.gguf"
    )
    shutil.move(str(artifact), str(keep))
    shutil.copyfile(
        Path(f"{prefix}-tensor-types.txt"), args.dest / f"{keep.name}.tensor-types.txt"
    )
    row = {
        "requested_gib": size,
        "floors": bool(args.floors),
        "status": "ok",
        "predicted_size_bytes": record["predicted_size_bytes"],
        "actual_size_bytes": obs["size_bytes"],
        "selected_count": record["selected_count"],
        "macro_kl": obs["macro_kl"],
        "same_top": obs["same_top"],
        "per_domain": obs.get("per_domain"),
        "artifact": keep.name,
        "artifact_sha256": sha256_file(keep),
        "recipe_sha256": sha256_file(Path(f"{prefix}-tensor-types.txt")),
    }
    print(
        f"[sweep] {size:g} GiB -> {obs['size_bytes'] / GIB:.2f} GiB  "
        f"KL={obs['macro_kl']:.4f}  top={obs['same_top']:.4f}  "
        f"selected={record['selected_count']}",
        flush=True,
    )
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", type=Path, required=True)
    ap.add_argument("--runtime", type=Path, required=True)
    ap.add_argument("--eval-data", type=Path, required=True)
    ap.add_argument("--refs-dir", type=Path, required=True)
    ap.add_argument("--sizes", required=True, help="Comma-separated GiB targets")
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--imatrix-arg", required=True, help="imatrix path as the analysis recorded it")
    ap.add_argument("--dest", type=Path, required=True)
    ap.add_argument("--policy", default="balanced")
    ap.add_argument("--floors", dest="floors", action="store_true", default=True)
    ap.add_argument("--no-floors", dest="floors", action="store_false")
    ap.add_argument("--workdir", type=Path, default=None)
    ap.add_argument("--tag", default="sweep")
    args = ap.parse_args()

    sizes = [float(s) for s in args.sizes.split(",") if s.strip()]
    work = args.workdir or default_scratch_root() / f"{args.tag}-work"
    work.mkdir(parents=True, exist_ok=True)
    args.dest.mkdir(parents=True, exist_ok=True)

    cfg = CalibrateConfig(
        source=Path(str(json.loads(args.analysis.read_text())["source"]["path"])),
        imatrix_corpus=Path("/dev/null"),
        runtime_dir=args.runtime.resolve(),
        eval_data_dir=args.eval_data.resolve(),
        out_dir=work,
        model_id=args.model_name,
        imatrix_arg=args.imatrix_arg,
        workdir=work,
        log_dir=work / "logs",
    )
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    ctx = {"work": work, "cfg": cfg, "env": _runtime_env(cfg.runtime_dir)}

    rows: list[dict] = []
    for size in sizes:
        print(f"[sweep] {size:g} GiB  floors={args.floors}  planning…", flush=True)
        try:
            rows.append(run_one(args, ctx, size))
        except Exception as exc:  # noqa: BLE001
            print(f"[sweep] {size:g} GiB  UNREACHABLE: {exc}", flush=True)
            rows.append({
                "requested_gib": size,
                "floors": bool(args.floors),
                "status": "unreachable",
                "error": str(exc),
            })

    out = args.dest / f"{args.tag}-report.json"
    out.write_text(json.dumps({"model_id": args.model_name, "rows": rows}, indent=1) + "\n")
    print(f"\n[sweep] report -> {out}")
    print(f"  {'GiB':>6} {'floors':>7} {'actual':>8} {'macro KL':>9} {'same-top':>9} {'selected':>9}")
    for row in rows:
        if row.get("status") != "ok":
            print(f"  {row['requested_gib']:6.2f} {str(row['floors']):>7}   UNREACHABLE")
            continue
        print(
            f"  {row['requested_gib']:6.2f} {str(row['floors']):>7} "
            f"{row['actual_size_bytes'] / GIB:8.2f} {row['macro_kl']:9.4f} "
            f"{row['same_top']:9.4f} {row['selected_count']:9d}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
