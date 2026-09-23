#!/usr/bin/env python3
"""Emit the shipped artifact for every tier a ``fit tier-search`` solved.

``fit tier-search`` answers WHICH artifact each tier ships and leaves its recipe
in ``probes/<point>-tensor-types.txt``.  Nothing emitted it: a bundle could record
four tier winners and contain no files.  This turns the report into the files
themselves.

WHY THE SHIPPED FILE IS RE-EVALUATED
------------------------------------
The probe was measured on an artifact built with the search's imatrix argument
string; this script (like ``fit quantize``) leaves that to ``analysis.json``,
which pins the calibration's own string.  llama.cpp writes
``quantize.imatrix.file`` verbatim and the KV survives up to 127 bytes, so the two
files differ by a few dozen bytes of metadata.  The quantized tensors are
identical — but "identical tensors" is an argument, and a measured KL on the file
that actually ships is a fact.  The report carries both numbers so the two can be
compared.

Sizes are not asserted against the probe: ``pipeline.quantize`` re-finalizes its
own prediction from the invocation it is about to run and refuses to return if the
output misses it (the G2 exact-size gate).  What this script adds is the tier
name, the provenance files, and the five-domain measurement of the shipped bytes.

USAGE
    python scripts/emit_tier_artifacts.py \\
        --bundle experiments/<run>/calibration \\
        --dest   /path/to/<model>-FIT \\
        --runtime tools/llama-b10666-rocm \\
        --eval-data eval-data \\
        [--tiers quality,balanced,compact,mini] [--workdir /dev/shm/tier-emit] \\
        [--skip-eval] [--dry-run]
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
from fit_gguf.pipeline import primary_type_from_plan  # noqa: E402
from fit_gguf.pipeline import quantize as pipeline_quantize  # noqa: E402
from fit_gguf.tier_search import prepare_scratch_references  # noqa: E402

DEFAULT_TIERS = ("quality", "balanced", "compact", "mini")


def artifact_name(model_id: str, tier: str, size_bytes: int, primary_type: str) -> str:
    """``<model>-FIT-<TIER>-<size>G-<type>.gguf``.

    The suffix stays a nameable preset — the Hugging Face model page matches file
    names against the known preset set and drops a file whose suffix is not one of
    them (see ``pipeline.primary_type_from_plan``).  The tier is carried in the
    middle because four files ship as a named set with different KL gates, and
    size alone does not say which gate each one satisfies.
    """
    gib = round(size_bytes / 2**30)
    return f"{model_id}-FIT-{tier.upper()}-{gib}G-{primary_type}.gguf"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path, required=True, help="Calibration Bundle")
    ap.add_argument("--dest", type=Path, required=True, help="Output directory for the shipped files")
    ap.add_argument("--runtime", type=Path, required=True, help="llama.cpp runtime dir")
    ap.add_argument("--eval-data", type=Path, required=True, help="Five frozen eval slices")
    ap.add_argument("--tiers", default=",".join(DEFAULT_TIERS))
    ap.add_argument("--imatrix", type=Path, default=None,
                    help="imatrix GGUF; only needed if the references must be regenerated")
    ap.add_argument("--workdir", type=Path, default=None,
                    help="Scratch for the artifact under construction (default: tmpfs)")
    ap.add_argument("--log-dir", type=Path, default=None)
    ap.add_argument("--n-gpu-layers", type=int, default=99)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--skip-eval", action="store_true",
                    help="Emit the files without measuring the shipped bytes")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the plan (names, recipes, sizes) and stop")
    ap.add_argument("--keep-work", action="store_true")
    args = ap.parse_args()

    bundle = args.bundle.resolve()
    report_path = bundle / "tier-search-report.json"
    if not report_path.is_file():
        print(f"[emit] error: {report_path} not found — run `fit tier-search` first",
              file=sys.stderr)
        return 2
    report = json.loads(report_path.read_text(encoding="utf-8"))
    model_id = json.loads(
        (bundle / "registry-entry.json").read_text(encoding="utf-8")
    )["model_id"]

    wanted = [t.strip() for t in args.tiers.split(",") if t.strip()]
    unknown = sorted(set(wanted) - set(report))
    if unknown:
        print(f"[emit] error: no tier-search result for {unknown}", file=sys.stderr)
        return 2

    work = args.workdir or default_scratch_root() / "tier-emit"
    work.mkdir(parents=True, exist_ok=True)
    dest = args.dest
    dest.mkdir(parents=True, exist_ok=True)

    # ---- plan ---------------------------------------------------------------
    plan_rows: list[dict] = []
    for tier in wanted:
        row = report[tier]
        if row.get("status") != "ok":
            print(f"[emit] {tier}: skipped (tier-search status {row.get('status')!r})")
            continue
        point = row["best_point"]
        plan_path = bundle / "probes" / f"{point}-plan.json"
        recipe_path = bundle / "probes" / f"{point}-tensor-types.txt"
        for needed in (plan_path, recipe_path):
            if not needed.is_file():
                print(f"[emit] error: {needed} missing — cannot reproduce {tier}",
                      file=sys.stderr)
                return 2
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        plan_rows.append({
            "tier": tier,
            "point": point,
            "plan_path": plan_path,
            "recipe_path": recipe_path,
            "analysis_path": Path(str(plan["analysis_path"])),
            "primary_type": primary_type_from_plan(plan),
            "probe_bytes": int(row["best_bytes"]),
            "probe_kl": float(row["macro_kl"]),
            "probe_same_top": float(row["same_top"]),
            "anchor": float(row["anchor"]),
            "same_top_reference": row.get("same_top_reference"),
        })

    if args.dry_run:
        for r in plan_rows:
            print(f"[emit] {r['tier']:>9}: {r['point']:<20} "
                  f"recipe={r['recipe_path'].name} primary={r['primary_type']} "
                  f"probe={r['probe_bytes'] / 2**30:.2f} GiB KL={r['probe_kl']:.4f}")
        return 0

    # ---- runtime / references ----------------------------------------------
    cfg = CalibrateConfig(
        source=Path(str(json.loads(plan_rows[0]["analysis_path"].read_text())
                         ["source"]["path"])),
        imatrix_corpus=args.imatrix or Path("/dev/null"),
        runtime_dir=args.runtime.resolve(),
        eval_data_dir=args.eval_data.resolve(),
        out_dir=bundle,
        model_id=model_id,
        imatrix_path=args.imatrix,
        n_gpu_layers=args.n_gpu_layers,
        threads=args.threads,
        workdir=work,
        log_dir=args.log_dir or work / "logs",
    )
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    env = _runtime_env(cfg.runtime_dir)

    refs_dir = None
    if not args.skip_eval:
        refs_dir = prepare_scratch_references(
            cfg, env, bundle, default_scratch_root() / f"cal-{model_id}" / "references"
        )

    # ---- emit ---------------------------------------------------------------
    emitted: list[dict] = []
    for r in plan_rows:
        tier = r["tier"]
        tmp = work / f"{tier}.gguf"
        print(f"[emit] {tier}: quantize from {r['recipe_path'].name}")
        record = pipeline_quantize(r["analysis_path"], r["recipe_path"], tmp)
        size = int(record["size_bytes"])
        name = artifact_name(model_id, tier, size, r["primary_type"])
        target = dest / name

        entry = {
            "tier": tier,
            "point": r["point"],
            "size_bytes": size,
            "primary_type": r["primary_type"],
            "anchor": r["anchor"],
            "same_top_reference": r["same_top_reference"],
            "probe": {
                "size_bytes": r["probe_bytes"],
                "macro_kl": r["probe_kl"],
                "same_top": r["probe_same_top"],
            },
            "recipe_sha256": sha256_file(r["recipe_path"]),
            "plan_sha256": sha256_file(r["plan_path"]),
            "quantize_size_matches_refinalization": record["size_matches_refinalization"],
        }

        if refs_dir is not None:
            obs = eval_artifact(cfg, tmp, refs_dir, f"emit-{tier}", env)
            entry["shipped"] = {
                "size_bytes": obs["size_bytes"],
                "macro_kl": obs["macro_kl"],
                "same_top": obs["same_top"],
                "per_domain": obs.get("per_domain"),
            }
            entry["kl_matches_probe"] = abs(obs["macro_kl"] - r["probe_kl"]) < 1e-9
            print(f"[emit] {tier}: shipped KL={obs['macro_kl']:.4f} "
                  f"top={obs['same_top']:.4f} (probe KL={r['probe_kl']:.4f}) "
                  f"-> {'PASS' if obs['macro_kl'] <= r['anchor'] else 'FAIL'}")

        # Provenance travels with the artifact: the recipe is what reproduces it,
        # and the quantize record is the invocation that did.
        for suffix, src in (
            (".tensor-types.txt", r["recipe_path"]),
            (".plan.json", r["plan_path"]),
            (".quantize-record.json", Path(f"{tmp}.quantize-record.json")),
        ):
            if src.is_file():
                shutil.copyfile(src, dest / f"{name}{suffix}")

        shutil.move(str(tmp), str(target))
        entry["artifact"] = name
        # Re-hashed after the move: this is the digest of the bytes at rest.
        entry["artifact_sha256"] = sha256_file(target)
        emitted.append(entry)
        print(f"[emit] {tier}: {name}  {size / 2**30:.2f} GiB")

    out = dest / "emit-report.json"
    out.write_text(json.dumps({
        "model_id": model_id,
        "bundle": str(bundle),
        "tiers": emitted,
    }, indent=1) + "\n", encoding="utf-8")

    print(f"\n[emit] {len(emitted)} artifacts -> {dest}")
    total = sum(e["size_bytes"] for e in emitted)
    print(f"[emit] total {total / 2**30:.2f} GiB")
    for e in emitted:
        shipped = e.get("shipped")
        tail = (f"  shipped KL={shipped['macro_kl']:.4f}" if shipped else "")
        print(f"  {e['tier']:>9}: {e['artifact']:<52} "
              f"{e['size_bytes'] / 2**30:6.2f} GiB{tail}")
    print(f"[emit] report -> {out}")

    if not args.keep_work and work.resolve() != bundle.resolve():
        shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
