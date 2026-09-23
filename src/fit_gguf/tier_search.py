"""Per-tier fidelity search — solve each tier for the SMALLEST passing artifact.

WHY THIS EXISTS
---------------
``fit calibrate`` fills each tier's window by probing the LARGEST uncovered KL gap
inside it (``_largest_uncovered_gap``).  That heuristic serves the *floor
derivation* — it wants enough samples in the window to take a stable P5 — and it
is deliberately blind to what the tier is actually for.

A tier's product is not "a preset that happens to pass".  It is the smallest
artifact that reaches the tier's KL anchor without dropping below its same-top
floor, and that artifact generally lies BETWEEN two ladder presets: it has to be
solved for.

Measured consequence on occamy-1.0-abliterated (2026-09-22): the quality window
[0.0425, 0.0575] held IQ4_XS (17.44 GiB, KL 0.0503 — a FAIL) and Q4_K_M
(19.71 GiB, KL 0.0471).  The answer lives in the KL gap 0.0471..0.0503.  The
largest gap was 0.0503..0.0575, so the single probe went ABOVE IQ4_XS and produced
a SMALLER artifact with a WORSE KL (17.27 GiB @ 0.0555), failing the anchor
outright; the tier then fell back to the Q4_K_M preset, 1.71 GiB larger than the
minimum PASS that exists on this curve.  Across four tiers the gap was 6.2 GiB
(9.8%).

ALGORITHM (per tier)
--------------------
    pass(o) := o.macro_kl <= anchor

    1. bracket on the observed curve:  hi = smallest observed PASS
                                       lo = largest observed FAIL below hi
    2. target = (lo.size + hi.size) // 2, quantized from the preset pair that
       brackets that size, with the bundle's imatrix
    3. eval-v1 the artifact, fold it into the curve, re-bracket
    4. repeat until the bracket collapses under ``tolerance`` or ``budget`` is spent

The result is the smallest evaluated PASS plus its reproducible recipe.  The gate
is KL alone (v0.3): the guard profile's same-top floor is carried through as
``same_top_reference`` and ``clears_floor``, but it never decides a verdict — see
:func:`passes` for why a dual gate both hides smaller valid artifacts and can
steer the search the wrong way.

ORDERING AND INTEGRITY
----------------------
Run this BEFORE ``stage_emit``: it appends its probes to ``curve-points.jsonl``,
and ``stage_emit`` records that file's digest in ``calibration-record.json``.
Running it afterwards leaves a stale digest — the bundle still validates (nothing
re-hashes the curve) but the record would lie.  ``reseal_bundle`` repairs that for
standalone runs on an already-emitted bundle.

Floors are frozen before the search runs, on purpose: the artifacts the search
selects must not feed back into the floor they are judged against.  Neither
``run_tier_search`` nor ``reseal_bundle`` touches the guard profile or the floor
derivation.
"""

from __future__ import annotations

import json
from pathlib import Path

from fit_gguf.calibrate import (
    CalibrateConfig,
    _append_curve_point,
    _load_curve_points,
    eval_artifact,
)
from fit_gguf.eval.provenance import sha256_file
from fit_gguf.registry import canonical_json_bytes, entry_digest

DEFAULT_TIERS = ("quality", "balanced", "compact", "mini")


# ------------------------------------------------------------------ predicates
def passes(obs: dict, anchor: float) -> bool:
    """A tier's product gate: macro KL at or under the anchor, and nothing else.

    v0.3 made the gate single-axis.  Through v0.2 a tier was a dual gate — KL plus
    a model-specific same-top floor — which made one tier name mean different
    things on different models.  Same-top is still measured and reported here
    (``same_top_reference`` / ``clears_floor``) but it never decides a verdict.

    This is not cosmetic: on occamy the mini floor sat 0.0003 above an 11.27 GiB
    point that clears the 0.20 KL anchor, and the balanced floor left only 0.0012
    of headroom — so a dual gate both hides a smaller valid artifact and can steer
    a probe that clears KL upward because its same-top dipped.
    """
    return obs["macro_kl"] <= anchor


def preset_sizes(curve: list[dict], presets: set[str]) -> list[tuple[int, str]]:
    """``(size, point_id)`` for every ladder preset already observed, ascending."""
    return sorted(
        (int(o["size_bytes"]), o["point_id"]) for o in curve if o["point_id"] in presets
    )


def bracketing_pair(sizes: list[tuple[int, str]], target: int) -> tuple[str, str] | None:
    """The preset pair whose sizes bracket ``target`` — the dry-run anchors."""
    below = [p for p in sizes if p[0] <= target]
    above = [p for p in sizes if p[0] > target]
    if not below or not above:
        return None
    return below[-1][1], above[0][1]


def tiers_from_profile(bundle: Path) -> dict[str, dict]:
    """``{tier: {anchor, floor}}`` read back from an emitted guard profile."""
    import yaml  # noqa: PLC0415

    profile = yaml.safe_load(
        (Path(bundle) / "guard-profile.yaml").read_text(encoding="utf-8")
    )
    return {
        name: {
            "anchor": float(spec["kl_anchor"]),
            "floor": (
                float(spec["same_top_floor"])
                if spec.get("same_top_floor") is not None
                else None
            ),
        }
        for name, spec in profile.get("tiers", {}).items()
    }


def tiers_from_evaluation(evaluation: dict) -> dict[str, dict]:
    """The same shape, taken from an in-flight calibration's frozen floor derivation."""
    return {
        name: {"anchor": float(tr["anchor"]), "floor": tr["floor"]}
        for name, tr in evaluation["tiers"].items()
    }


# --------------------------------------------------------------------- search
def prior_points_for_tier(curve: list[dict], tier: str) -> set[str]:
    """Every curve point that belongs to this tier's own search history.

    A point belongs to a tier when its probe record says so, or when its id
    carries the tier's prefix — ``probe-<tier>-N`` from a calibration gap probe,
    ``tier-<tier>-sN`` from this search, ``floor-<tier>-<size>G`` from a
    fixed-size sweep. Ladder presets carry none of those, which is the point:
    they are the bounds no policy change can move.
    """
    return {
        o["point_id"] for o in curve
        if o.get("probe", {}).get("tier") == tier
        or o["point_id"].startswith(f"tier-{tier}-")
        or o["point_id"].startswith(f"floor-{tier}-")
    }


def _fresh_tag(bundle: Path, tier: str) -> str:
    """First unused ``tier-<tier>-s<N>`` probe tag.

    A standalone re-run lands on the same bundle as earlier passes, so the step
    counter must skip tags that already exist rather than overwrite their plans.
    """
    step = 1
    while (bundle / "probes" / f"tier-{tier}-s{step}-plan.json").is_file():
        step += 1
    return f"tier-{tier}-s{step}"


def _summarise(
    tier: str, pool: list[dict], anchor: float, floor: float | None, presets: set[str]
) -> dict:
    """Smallest observed PASS for one tier, with its gain over the best passing preset.

    ``floor`` is carried through as a reference, not a gate: ``clears_floor``
    records whether the selected artifact also clears the model's calibrated
    same-top floor, which is what a v0.2 dual-gate reading would have asked.
    """
    passing = [o for o in pool if passes(o, anchor)]
    if not passing:
        return {"tier": tier, "status": "no_pass", "anchor": anchor}
    best = min(passing, key=lambda o: o["size_bytes"])
    preset = min(
        (o for o in pool if o["point_id"] in presets and passes(o, anchor)),
        key=lambda o: o["size_bytes"],
        default=None,
    )
    return {
        "tier": tier,
        "status": "ok",
        "best_point": best["point_id"],
        "best_bytes": best["size_bytes"],
        "macro_kl": best["macro_kl"],
        "same_top": best["same_top"],
        "anchor": anchor,
        "same_top_reference": floor,
        "clears_floor": None if floor is None else best["same_top"] >= floor,
        "preset_baseline": preset["point_id"] if preset else None,
        "preset_bytes": preset["size_bytes"] if preset else None,
        "saved_bytes": (preset["size_bytes"] - best["size_bytes"]) if preset else 0,
    }


def run_tier_search(
    cfg: CalibrateConfig,
    contract: dict,
    imx: Path,
    refs_dir: Path,
    env: dict,
    tiers: dict[str, dict],
    *,
    budget: int = 2,
    tolerance_mib: int = 128,
    work: Path | None = None,
    always_active_floors: bool = False,
    restart_tiers: set[str] | None = None,
) -> tuple[dict, list[dict]]:
    """Bisect each tier's size bracket and keep the smallest evaluated PASS.

    Returns ``(report, new_observations)``.  Each observation is appended to the
    bundle curve as it is produced, so an interrupted run resumes from the ledger
    instead of paying for the same five-domain eval twice.
    """
    from fit_gguf.pipeline import analyze as pipeline_analyze  # noqa: PLC0415
    from fit_gguf.pipeline import plan as pipeline_plan  # noqa: PLC0415
    from fit_gguf.pipeline import quantize as pipeline_quantize  # noqa: PLC0415

    bundle = cfg.out_dir
    curve = _load_curve_points(bundle)
    presets = set(contract["ladder_standard_presets"]) | set(cfg.extra_presets)
    sizes = preset_sizes(curve, presets)
    tolerance = tolerance_mib * 1024 * 1024
    work = Path(work) if work else (cfg.workdir or bundle)
    work.mkdir(parents=True, exist_ok=True)
    imatrix_arg = cfg.imatrix_arg or str(imx)

    report: dict[str, dict] = {}
    new_obs: list[dict] = []

    for tier, spec in tiers.items():
        anchor = float(spec["anchor"])
        floor = spec.get("floor")
        cfg.log(f"tier-search {tier}: anchor={anchor} same_top_reference={floor}")
        # A policy change moves EVERY point, so an earlier pass's winner is not a
        # bound on this one -- and because the search ranks by size, a marginally
        # smaller older point can outrank a much better allocation and freeze the
        # tier before the new one is ever measured. `restart_tiers` drops that
        # tier's own history, leaving the ladder presets (which no policy can
        # move) as the bounds.
        if restart_tiers and tier in restart_tiers:
            prior = prior_points_for_tier(curve, tier)
            pool = [o for o in curve if o["point_id"] not in prior] + new_obs
            cfg.log(
                f"tier-search {tier}: restart — ignoring {len(prior)} prior point(s)"
            )
        else:
            pool = curve + new_obs

        for _ in range(budget):
            passing = [o for o in pool if passes(o, anchor)]
            if not passing:
                cfg.log(f"tier-search {tier}: no observed PASS — nothing to tighten")
                break
            hi = min(passing, key=lambda o: o["size_bytes"])
            below = [
                o
                for o in pool
                if not passes(o, anchor) and o["size_bytes"] < hi["size_bytes"]
            ]
            if not below:
                cfg.log(
                    f"tier-search {tier}: already at the smallest observed size "
                    f"({hi['size_bytes'] / 2**30:.2f} GiB, KL {hi['macro_kl']:.4f})"
                )
                break
            lo = max(below, key=lambda o: o["size_bytes"])
            if hi["size_bytes"] - lo["size_bytes"] <= tolerance:
                cfg.log(f"tier-search {tier}: bracket under tolerance — stop")
                break
            target = (lo["size_bytes"] + hi["size_bytes"]) // 2
            pair = bracketing_pair(sizes, target)
            if pair is None:
                cfg.log(
                    f"tier-search {tier}: no preset pair brackets "
                    f"{target / 2**30:.2f} GiB — stop"
                )
                break

            analysis_dir = bundle / "analysis" / f"{pair[0]}-{pair[1]}"
            analysis_json = analysis_dir / "analysis.json"
            if not analysis_json.is_file():
                pipeline_analyze(
                    cfg.source, imx, cfg.runtime_dir, analysis_dir,
                    lower_preset=pair[0], upper_preset=pair[1],
                    imatrix_arg=imatrix_arg,
                )

            tag = _fresh_tag(bundle, tier)
            prefix = bundle / "probes" / tag
            cfg.log(
                f"tier-search {tier}: target {target / 2**30:.2f} GiB "
                f"(bracket {lo['size_bytes'] / 2**30:.2f} FAIL .. "
                f"{hi['size_bytes'] / 2**30:.2f} PASS) via {pair[0]}-{pair[1]}"
            )
            pipeline_plan(
                analysis_json, prefix, target_bytes=int(target),
                policy="balanced", model_name=cfg.model_id,
                always_active_floors=always_active_floors,
            )
            artifact = work / f"{tag}.gguf"
            try:
                pipeline_quantize(
                    analysis_json, prefix.parent / f"{tag}-tensor-types.txt",
                    artifact, imatrix_arg=imatrix_arg,
                )
            except Exception as exc:  # noqa: BLE001
                # A probe that cannot be written must not take the search down:
                # every ladder point is already paid for and the bracket is
                # unchanged, so skipping costs one budget slot and nothing else.
                cfg.log(f"tier-search {tier}: {tag} quantize FAILED ({exc}) — skipped")
                artifact.unlink(missing_ok=True)
                continue
            try:
                obs = eval_artifact(cfg, artifact, refs_dir, tag, env)
            except Exception as exc:  # noqa: BLE001
                cfg.log(f"tier-search {tier}: {tag} eval FAILED ({exc}) — skipped")
                artifact.unlink(missing_ok=True)
                continue
            artifact.unlink(missing_ok=True)

            obs["point_id"] = tag
            obs["probe"] = {
                "tier": tier,
                "target_bytes": int(target),
                "plan_sha256": sha256_file(Path(f"{prefix}-plan.json")),
                "recipe_sha256": sha256_file(Path(f"{prefix}-tensor-types.txt")),
            }
            new_obs.append(obs)
            _append_curve_point(bundle, obs)
            pool.append(obs)
            cfg.log(
                f"tier-search {tier}: {tag} {obs['size_bytes'] / 2**30:.2f} GiB "
                f"KL={obs['macro_kl']:.4f} top={obs['same_top']:.4f} -> "
                f"{'PASS' if passes(obs, anchor) else 'FAIL'}"
            )

        summary_pool = pool if (restart_tiers and tier in restart_tiers) else curve + new_obs
        report[tier] = _summarise(tier, summary_pool, anchor, floor, presets)
        best = report[tier]
        if best["status"] == "ok":
            cfg.log(
                f"tier-search {tier}: BEST {best['best_point']} "
                f"{best['best_bytes'] / 2**30:.2f} GiB @ KL {best['macro_kl']:.4f}"
                + (
                    f"  (saves {best['saved_bytes'] / 2**30:.2f} GiB vs preset)"
                    if best["saved_bytes"]
                    else ""
                )
            )

    return report, new_obs


def write_report(bundle: Path, report: dict) -> Path:
    """Merge a per-tier report into ``tier-search-report.json``.

    Merged, never overwritten: a run scoped to ``--tiers quality,balanced`` is a
    follow-up pass, and replacing the file would erase the tiers an earlier pass
    already solved.
    """
    out = Path(bundle) / "tier-search-report.json"
    merged: dict[str, dict] = {}
    if out.is_file():
        try:
            merged = json.loads(out.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            merged = {}
    merged.update(report)
    out.write_text(json.dumps(merged, indent=1) + "\n", encoding="utf-8")
    return out


def reseal_bundle(bundle: Path, contract: dict) -> None:
    """Re-hash a bundle after a standalone search appended to its curve.

    Only derived digests move.  The guard profile and the floor derivation are
    deliberately left alone: they were frozen before the search ran, so the
    artifacts it selected cannot feed back into the floor they are judged
    against.  Rewriting the record moves its digest, which moves the registry
    entry's pin, which moves the entry digest — all three, or ``validate_bundle``
    fails on the first stale pin.

    The ladder seed material is refreshed for the same reason: the standalone
    path never went through ``stage_emit``, so ``state-artifact-manifest.txt``
    and ``seed-provenance.jsonl`` would otherwise describe a curve that no longer
    exists.  Search probes are admitted exactly as gap probes are — as points
    that name no window preset, so a poison preset can never anchor through them.
    """
    from fit_gguf.calibrate import write_seed_material
    from fit_gguf.eval.contract import contract_digest

    bundle = Path(bundle)
    curve_path = bundle / "curve-points.jsonl"
    record_path = bundle / "calibration-record.json"
    curve = _load_curve_points(bundle)

    write_seed_material(
        bundle,
        curve,
        set(contract["ladder_standard_presets"]),
        reference_manifest_sha256=sha256_file(bundle / "reference-manifest.json"),
        evaluator_contract_sha256=contract_digest(),
    )

    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["process"]["curve_points"] = len(curve)
    record["artifacts"]["curve_points_sha256"] = sha256_file(curve_path)
    record_path.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")

    entry_path = bundle / "registry-entry.json"
    entry = json.loads(entry_path.read_text(encoding="utf-8"))
    entry["calibration"]["record"]["sha256"] = sha256_file(record_path)
    entry["entry_sha256"] = entry_digest(entry)
    entry_path.write_bytes(canonical_json_bytes(entry))

    # Excludes itself: a listing that carries its own pre-rewrite digest is a
    # checksum nobody can verify.
    sums = [
        f"{sha256_file(f)}  {f.name}"
        for f in sorted(bundle.iterdir())
        if f.is_file() and f.name != "SHA256SUMS"
    ]
    (bundle / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf-8")


def prepare_scratch_references(
    cfg: CalibrateConfig, env: dict, bundle: Path, scratch_refs: Path
) -> Path:
    """Make five-domain references available on a hot-loop-safe volume.

    Standalone runs have no calibration scratch to inherit, so reuse the bundle's
    published references when they are arithmetically complete, and regenerate
    from the source only if they are not.  A reference set is a deterministic
    function of the source and the eval slices, so a fresh set is identical to the
    calibrated one.
    """
    from fit_gguf.calibrate import (
        assert_hot_loop_fs_safe,
        publish_tree,
        ref_ok,
        stage_references,
    )

    scratch_refs = Path(scratch_refs)
    assert_hot_loop_fs_safe({"references": scratch_refs})
    if any(scratch_refs.glob("bf16-*.kld")):
        return scratch_refs
    published = Path(bundle) / "references"
    if published.is_dir():
        for ref in sorted(published.glob("bf16-*.kld")):
            if ref_ok(ref):
                publish_tree(published, scratch_refs, ref.name)
    if not any(scratch_refs.glob("bf16-*.kld")):
        cfg.log(f"references missing at {scratch_refs} — regenerating from source")
        stage_references(cfg, env, scratch_refs)
    return scratch_refs
