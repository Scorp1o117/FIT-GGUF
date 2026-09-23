#!/usr/bin/env python3
"""Quantize and evaluate the APEX tier recipes for a model, under eval-v1.

APEX (Adaptive Precision for EXpert Models, LocalAI) ships a per-tensor recipe as
a llama.cpp tensor-type file plus a **base type** per tier. Applying one is a
single `llama-quantize` call:

    llama-quantize --imatrix <imx> --tensor-type-file <recipe> <src> <out> <base>

The base type is not decoration — the recipes cover 15 tensor names per layer and
leave everything else (token_embd, output, the SSM tensors, the norms) at the
base. `quantize.sh` pins them per profile; `APEX_BASE_TYPE` below mirrors that
table and `--base-type` overrides it for one run.

The evaluation deliberately reuses FIT's own `eval_artifact`, so the APEX points
land on the same five domains, the same BF16 references and the same runtime as
the FIT and native-preset points. A comparison between three families measured by
two different protocols would be a comparison of the protocols.

USAGE
    python scripts/run_apex_tiers.py \\
        --apex-dir /path/to/apex-quant --config-prefix qwen36_35b \\
        --source <bf16.gguf> --imatrix <imatrix.gguf> \\
        --runtime tools/llama-b10666-rocm --eval-data eval-data \\
        --dest /path/to/<model>-APEX \\
        [--tiers quality,balanced,compact,mini] [--refs-dir <dir>]
        [--workdir /dev/shm/apex] [--skip-eval] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
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
from fit_gguf.llama_integration import resolve_runtime_binary  # noqa: E402
from fit_gguf.tier_search import prepare_scratch_references  # noqa: E402

DEFAULT_TIERS = ("quality", "balanced", "compact", "mini")

# Mirrors `apex-quant/scripts/quantize.sh`, which sets these per profile. Kept
# here rather than parsed out of the shell so a missing key is a loud error
# instead of a silent empty string.
APEX_BASE_TYPE = {
    "quality": "Q6_K",
    "balanced": "Q6_K",
    "compact": "Q4_K_M",
    "mini": "Q3_K_M",
}

GIB = 2**30


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apex-dir", type=Path, required=True, help="The apex-quant checkout")
    ap.add_argument("--config-prefix", required=True,
                    help="Recipe family in <apex-dir>/configs, e.g. qwen36_35b")
    ap.add_argument("--source", type=Path, required=True, help="BF16 source GGUF")
    ap.add_argument("--imatrix", type=Path, required=True, help="imatrix GGUF")
    ap.add_argument("--runtime", type=Path, required=True, help="llama.cpp runtime dir")
    ap.add_argument("--eval-data", type=Path, required=True, help="Five frozen eval slices")
    ap.add_argument("--dest", type=Path, required=True, help="Where the APEX GGUFs land")
    ap.add_argument("--model-id", default=None, help="Name used in file names (default: dest name)")
    ap.add_argument("--tiers", default=",".join(DEFAULT_TIERS))
    ap.add_argument("--base-type", default=None,
                    help="Override the base type for every tier (default: the APEX table)")
    ap.add_argument("--refs-dir", type=Path, default=None)
    ap.add_argument("--workdir", type=Path, default=None)
    ap.add_argument("--log-dir", type=Path, default=None)
    ap.add_argument("--n-gpu-layers", type=int, default=99)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--reuse", action="store_true",
                    help="Skip quantization for tiers whose target already exists on disk")
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--keep-work", action="store_true")
    args = ap.parse_args()

    apex = args.apex_dir.resolve()
    tiers = [t.strip() for t in args.tiers.split(",") if t.strip()]
    model_id = args.model_id or args.dest.name

    plan_rows = []
    for tier in tiers:
        recipe = apex / "configs" / f"{args.config_prefix}_{tier}.txt"
        if not recipe.is_file():
            print(f"[apex] error: no recipe at {recipe}", file=sys.stderr)
            return 2
        base = args.base_type or APEX_BASE_TYPE.get(tier)
        if base is None:
            print(f"[apex] error: no base type known for tier {tier!r}; pass --base-type",
                  file=sys.stderr)
            return 2
        plan_rows.append({"tier": tier, "recipe": recipe, "base": base})

    if args.dry_run:
        for r in plan_rows:
            print(f"[apex] {r['tier']:>9}: {r['recipe'].name}  base={r['base']}  "
                  f"({sum(1 for _ in r['recipe'].open())} tensors pinned)")
        return 0

    work = args.workdir or default_scratch_root() / "apex-tiers"
    work.mkdir(parents=True, exist_ok=True)
    args.dest.mkdir(parents=True, exist_ok=True)

    cfg = CalibrateConfig(
        source=args.source,
        imatrix_corpus=args.imatrix,
        runtime_dir=args.runtime.resolve(),
        eval_data_dir=args.eval_data.resolve(),
        out_dir=work,
        model_id=model_id,
        imatrix_path=args.imatrix,
        imatrix_arg=str(args.imatrix),
        n_gpu_layers=args.n_gpu_layers,
        threads=args.threads,
        workdir=work,
        log_dir=args.log_dir or work / "logs",
    )
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    env = _runtime_env(cfg.runtime_dir)
    quantize_bin = resolve_runtime_binary(cfg.runtime_dir, "llama-quantize")

    refs_dir = None
    if not args.skip_eval:
        refs_dir = prepare_scratch_references(
            cfg, env, args.dest,
            args.refs_dir or default_scratch_root() / f"cal-{model_id}" / "references",
        )

    emitted = []
    for r in plan_rows:
        tier = r["tier"]
        tmp = work / f"APEX-{tier}.gguf"
        # APEX's i-* profiles are the same recipe plus a diverse imatrix -- verified
        # by diffing laguna_s21_* against laguna_s21_i_*: byte-identical. So an
        # imatrix argument is exactly what makes this an I-variant, and the name
        # should say so.
        family = "APEX-I" if args.imatrix else "APEX"
        name = f"{model_id}-{family}-{tier.upper()}-{r['base']}.gguf"
        target = args.dest / name
        print(f"[apex] {tier}: quantize with {r['recipe'].name} (base {r['base']})", flush=True)

        if args.reuse and target.is_file():
            size = target.stat().st_size
            print(f"[apex] {tier}: reusing {name} ({size / GIB:.2f} GiB)", flush=True)
            entry = {
                "tier": tier, "family": family, "artifact": name,
                "size_bytes": size, "base_type": r["base"],
                "recipe": r["recipe"].name, "recipe_sha256": sha256_file(r["recipe"]),
                "reused": True,
            }
            if refs_dir is not None:
                obs = eval_artifact(cfg, target, refs_dir, f"APEX-{tier}", env)
                entry["macro_kl"] = obs["macro_kl"]
                entry["same_top"] = obs["same_top"]
                entry["per_domain"] = obs.get("per_domain")
                print(f"[apex] {tier}: KL={obs['macro_kl']:.4f} top={obs['same_top']:.4f}", flush=True)
            entry["artifact_sha256"] = sha256_file(target)
            emitted.append(entry)
            continue

        log = cfg.log_dir / f"quantize-APEX-{tier}.log"
        cmd = [str(quantize_bin), "--imatrix", str(args.imatrix),
               "--tensor-type-file", str(r["recipe"]),
               str(args.source), str(tmp), r["base"]]
        with log.open("wb") as handle:
            rc = subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT, env=env).returncode
        if rc != 0 or not tmp.is_file():
            print(f"[apex] {tier}: quantize FAILED (rc={rc}) — see {log}", file=sys.stderr)
            tmp.unlink(missing_ok=True)
            continue
        size = tmp.stat().st_size
        print(f"[apex] {tier}: {size / GIB:.2f} GiB", flush=True)

        entry = {
            "tier": tier,
            "family": "APEX",
            "artifact": name,
            "size_bytes": size,
            "base_type": r["base"],
            "recipe": r["recipe"].name,
            "recipe_sha256": sha256_file(r["recipe"]),
            "quantize_returncode": rc,
        }

        if refs_dir is not None:
            obs = eval_artifact(cfg, tmp, refs_dir, f"APEX-{tier}", env)
            entry["macro_kl"] = obs["macro_kl"]
            entry["same_top"] = obs["same_top"]
            entry["per_domain"] = obs.get("per_domain")
            print(f"[apex] {tier}: KL={obs['macro_kl']:.4f} top={obs['same_top']:.4f}", flush=True)

        shutil.move(str(tmp), str(target))
        entry["artifact_sha256"] = sha256_file(target)
        emitted.append(entry)

    out = args.dest / "apex-report.json"
    out.write_text(json.dumps({
        "model_id": model_id,
        "apex_dir": str(apex),
        "config_prefix": args.config_prefix,
        "source": str(args.source),
        "imatrix": str(args.imatrix),
        "tiers": emitted,
    }, indent=1) + "\n", encoding="utf-8")

    print(f"\n[apex] {len(emitted)} artifacts -> {args.dest}")
    total = sum(e["size_bytes"] for e in emitted)
    print(f"[apex] total {total / GIB:.2f} GiB")
    for e in emitted:
        kl = e.get("macro_kl")
        print(f"  {e['tier']:>9}: {e['artifact']:<44} {e['size_bytes'] / GIB:6.2f} GiB"
              + (f"  KL={kl:.4f}" if kl is not None else ""))
    print(f"[apex] report -> {out}")

    if not args.keep_work and work.resolve() != args.dest.resolve():
        shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
