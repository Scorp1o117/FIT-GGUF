#!/usr/bin/env python3
"""R2 — Independent Local Reference Sweep (GPT six-round ruling).

For each tier, a fixed five-point grid mechanically derived from the final
search answer:  B_FS + {-256, -192, -128, -64, +64} MiB.

* The grid is frozen (B_FS + frozen offsets) — never adapted to convenient
  cache contents; cache hits are only taken when a prior evaluated artifact
  has exactly the delivered size a grid point plans to.
* Grid points below the healthy window floor are clipped to the floor and
  recorded as frontier-clipped (Mini/Compact rule).
* Verdict (strict form from the ruling): R2 FAIL iff a healthy grid PASS
  exists at delivered size < B_FS - 128 MiB (strictly less; equality is
  within the declared search tolerance).
"""

import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "experiments" / "2026-09-02-m2-topkl-calibration"))

from fit_gguf.eval.provenance import verify_eval_v1_provenance  # noqa: E402
from fit_gguf.fidelity_runner import (  # noqa: E402
    DOMAINS,
    RunnerConfig,
    SearchExecutor,
    discover_windows,
    load_seeds,
    resolve_contract,
)
from fit_gguf.pipeline import plan as pipeline_plan
from fit_gguf.pipeline import quantize as pipeline_quantize

M2 = REPO / "experiments" / "2026-09-02-m2-topkl-calibration"
MODEL = "orcarouter-Qwen3.8-27B-Uncensored"
RT = REPO / "tools" / "llama-b10666-rocm"
IMX = REPO / "imatrix_unsloth.gguf"
REFS = REPO / "experiments" / "2026-09-02-eval-v1" / "refs-orcarouter"
EVAL_DATA = REPO / "eval-data"
GUARD_REGISTRY = REPO / "profiles" / "guard"
REFINE_PROFILE = REPO / "profiles" / "refine-profile-qwen-hybrid-band-v1.json"
MANIFEST = M2 / "results" / "artifact-manifest.txt"
LOGS = M2 / "logs"

OFFSETS_MIB = (-256, -192, -128, -64, 64)
TIER_B_FS = {
    "quality": 17443851488,
    "balanced": 13790608608,
    # adopted 2026-09-04 via fidelity-search re-run with full evidence:
    # status noise_inversion, smallest verified PASS 11,991,706,848
    # (kld 0.1486 / top 89.09), 0 fresh evals. Supersedes 12,206,255,328.
    "compact": 11991706848,
    "mini": 11116944608,
}
FREEZE = REPO / "experiments" / "2026-09-02-eval-v1" / "FREEZE.json"
REFERENCE_MANIFEST = REPO / "experiments" / "2026-09-02-eval-v1" / "reference-manifest-orcarouter.json"
SOURCE = Path("/run/media/s117/OS/Models/orcarouter-Qwen3.8-27B-Uncensored/Qwen3.8-27B-Uncensored-BF16.gguf")

ANALYSES = [
    M2 / "orcarouter-analysis-Q4_K_M-Q5_K_M",
    M2 / "orcarouter-analysis-IQ3_M-IQ4_XS",
    M2 / "orcarouter-analysis-IQ3_XS-IQ3_S",
    M2 / "orcarouter-analysis-Q2_K-IQ3_XXS",
    M2 / "orcarouter-analysis-Q3_K_S-IQ3_S",  # poison — never selected
]


def prior_source_sha() -> str:
    """Source GGUF weights digest for the guard binding (cached per session)."""
    global _SOURCE_SHA
    if _SOURCE_SHA is None:
        import hashlib
        digest = hashlib.sha256()
        with SOURCE.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        _SOURCE_SHA = digest.hexdigest()
    return _SOURCE_SHA


_SOURCE_SHA: str | None = None


def prior_evals_by_size():
    """size_bytes -> (macro_kl, macro_top) from manifest x logs (cache)."""
    sizes = {}
    for line in MANIFEST.read_text().splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[0].startswith("orcarouter-"):
            # log tags carry no "orcarouter-" prefix — key by the stripped name
            sizes[parts[0][len("orcarouter-"):]] = int(parts[1])
    by_size = {}
    for entry in sorted(LOGS.iterdir()):
        name = entry.name
        if not name.startswith("eval-orcarouter-") or not name.endswith(".log"):
            continue
        from fit_gguf.eval.results import parse_llama_kl_log
        import re

        body = name[len("eval-orcarouter-"):-len(".log")]
        m = re.search(r"-(wiki_test|wiki_valid|chinese|code|agent_chat)$", body)
        if m is None:
            continue
        tag = body[: -len(m.group(0))]
        if tag not in sizes:
            continue
        try:
            metrics = parse_llama_kl_log(entry.read_text(errors="replace"))
        except Exception:  # noqa: BLE001
            continue
        acc = by_size.setdefault(sizes[tag], {})
        acc[m.group(1)] = metrics
    out = {}
    for size, doms in by_size.items():
        if len(doms) != 5:
            continue
        kl = sum(doms[d]["mean_kld"] for d in DOMAINS) / 5
        top = sum(doms[d]["same_top_pct"] for d in DOMAINS) / 5
        out[size] = (kl, top)
    return out


def main() -> int:
    windows = discover_windows([str(p) for p in ANALYSES])
    # frozen eval-v1 closure + weights binding (release-gate hardening)
    provenance = verify_eval_v1_provenance(REFS, EVAL_DATA, FREEZE, REFERENCE_MANIFEST)
    print(f"eval provenance verified: contract {provenance.contract_digest[:12]}…, "
          f"5 refs + 5 corpora pinned", flush=True)
    source_sha = prior_source_sha()
    contract_by_tier = {
        t: resolve_contract(MODEL, t, GUARD_REGISTRY, source_sha)
        for t in ("quality", "balanced", "compact", "mini")
    }
    cache = prior_evals_by_size()
    report = {"tiers": {}, "fresh_evals": 0, "cache_hits": 0}

    for tier, b_fs in TIER_B_FS.items():
        contract = contract_by_tier[tier]
        work = Path("/dev/shm") / "orca-r2" / tier
        out_dir = M2 / "results-fs" / f"r2-{tier}"
        config = RunnerConfig(
            runtime=RT, imatrix=IMX, refs_dir=REFS, eval_data_dir=EVAL_DATA,
            work_dir=work, out_dir=out_dir, model_name=MODEL,
            guard_registry=GUARD_REGISTRY, refine_profile=REFINE_PROFILE,
            eval_provenance=provenance, require_eval_provenance=True,
            source_sha256=source_sha,
        )
        executor = SearchExecutor(config, windows, contract, MANIFEST, LOGS)
        # the tier's healthy window is the one containing its B_FS; grid
        # points are clipped into THAT window (docstring rule), never into
        # a neighboring tier's window
        tier_window = next(
            (w for w in windows if w.healthy and w.lower_size <= b_fs <= w.upper_size), None
        )
        if tier_window is None:
            raise SystemExit(f"{tier}: B_FS {b_fs:,} not inside any healthy window")
        print(f"== {tier}: B_FS {b_fs:,} window "
              f"{tier_window.lower_preset}->{tier_window.upper_preset} "
              f"[{tier_window.lower_size:,}, {tier_window.upper_size:,}] ==", flush=True)
        rows = []
        for off_mib in OFFSETS_MIB:
            target = b_fs + off_mib * 1024 * 1024
            window = tier_window
            clipped = not (window.lower_size <= target <= window.upper_size)
            target = min(max(target, window.lower_size), window.upper_size)
            probe_tag = f"R2-{tier}-off{off_mib:+d}"
            record = None
            record_path = out_dir / f"{probe_tag}-plan-plan.json"
            if record_path.exists():
                try:
                    candidate = json.loads(record_path.read_text())
                    if int(candidate.get("target_bytes", -1)) == target:
                        record = candidate  # deterministic re-plan of the same target
                except Exception:  # noqa: BLE001 — stale record: re-plan
                    record = None
            if record is None:
                record = pipeline_plan(
                    window.analysis_path / "analysis.json",
                    out_dir / f"{probe_tag}-plan",
                    target_bytes=target,
                    policy="balanced",
                    model_name=MODEL,
                    refine_profile=REFINE_PROFILE,
                )
            delivered = int(record["predicted_size_bytes"])
            verdict = None
            source = None
            if delivered in cache:
                kl, top = cache[delivered]
                source = "cache-hit"
                report["cache_hits"] += 1
            else:
                artifact = work / f"{probe_tag}.gguf"
                pipeline_quantize(
                    window.analysis_path / "analysis.json",
                    out_dir / f"{probe_tag}-plan-tensor-types.txt",
                    artifact,
                    imatrix_arg=str(IMX),
                )
                # record the artifact before evaluation so an interrupted run
                # leaves a complete manifest line (size<->log binding survives)
                digest = hashlib.sha256()
                with artifact.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1 << 20), b""):
                        digest.update(chunk)
                with MANIFEST.open("a", encoding="utf-8") as handle:
                    handle.write(f"orcarouter-{probe_tag}  {delivered}  {digest.hexdigest()}\n")
                metrics = executor._eval_domains(artifact, f"orcarouter-{probe_tag}")
                artifact.unlink(missing_ok=True)
                if metrics is None:
                    rows.append({"offset_mib": off_mib, "target": target,
                                 "delivered": delivered, "status": "EVAL-FAILED"})
                    continue
                kl = sum(metrics[d]["mean_kld"] for d in DOMAINS) / 5
                top = sum(metrics[d]["same_top_pct"] for d in DOMAINS) / 5
                cache[delivered] = (kl, top)  # dedup identical delivered sizes
                source = "fresh"
                report["fresh_evals"] += 1
            passed = contract.passes(kl, top / 100.0)
            verdict = ("PASS" if passed else
                       ("FAIL-KL" if kl > contract.kl_anchor else "FAIL-TOP"))
            rows.append({
                "offset_mib": off_mib, "target": target, "delivered": delivered,
                "clipped": clipped, "macro_kl": round(kl, 4),
                "same_top_pct": round(top, 2), "passed": passed,
                "verdict": verdict, "source": source,
            })
            print(f"{tier} off{off_mib:+d}: delivered {delivered:,} "
                  f"({delivered/1024**3:.2f}G) kld {kl:.4f} top {top:.2f} "
                  f"{verdict} [{source}]" + (" CLIPPED" if clipped else ""), flush=True)

        b_fs_tolerance = b_fs - 128 * 1024 * 1024
        passing = [r for r in rows if r.get("passed")]
        lowest_pass = min((r["delivered"] for r in passing), default=None)
        fail = any(r["delivered"] < b_fs_tolerance for r in passing)
        r2 = "FAIL" if fail else "PASS"
        margins = None
        if tier == "compact":
            margins = [
                {"offset_mib": r["offset_mib"],
                 "kl_margin": round(contract.kl_anchor - r["macro_kl"], 4),
                 "top_margin_pct": round(r["same_top_pct"] - contract.same_top_floor * 100, 2)}
                for r in rows
                if r.get("passed") is not None and "macro_kl" in r
            ]
        report["tiers"][tier] = {
            "b_fs": b_fs, "grid": rows, "lowest_grid_pass": lowest_pass,
            "delta_vs_fs_mib": round((lowest_pass - b_fs) / 1024 / 1024, 1) if lowest_pass else None,
            "r2_verdict": r2, "compact_dual_margins": margins,
        }
        print(f"== {tier}: R2 {r2} (lowest grid PASS "
              f"{lowest_pass/1024**3 if lowest_pass else float('nan'):.2f}G) ==", flush=True)

    out = M2 / "results" / "r2-reference-sweep.json"
    out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(f"report: {out}")
    print(f"totals: fresh {report['fresh_evals']}, cache hits {report['cache_hits']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
