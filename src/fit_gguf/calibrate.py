"""``fit calibrate`` — the P2 production line (fidelity-calibration-v1).

Turns a BF16 GGUF into a Calibration Bundle: pinned inputs → imatrix coverage
check → five-domain references → standard preset ladder → boundary windows →
gap probes → floor derivation → Guard Profile → bundle emission with a
candidate registry entry. Fail-closed per the frozen contract's failure-state
enumeration; the contract JSON is authoritative.

Gate P2-A: ``--replay-existing`` re-derives the calibration from recorded
observations with zero evals and must reproduce the archived P1 replay.
Gate P2-B: a fresh onboarding must gap-probe thin windows honestly — never
reuse a grandfathered floor to force a PASS.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from fit_gguf import calibration as cal
from fit_gguf.calibration import CalibrationError
from fit_gguf.eval.contract import DOMAINS
from fit_gguf.eval.provenance import sha256_file
from fit_gguf.eval.results import parse_llama_kl_log
from fit_gguf.llama_integration import resolve_runtime_binary, runtime_env
from fit_gguf.registry import (
    REGISTRY_SCHEMA,
    canonical_json_bytes,
    entry_digest,
    validate_bundle,
)

SLICE_SUFFIX = {
    "wiki_test": "64k.txt",
    "wiki_valid": "valid-64k.txt",
    "chinese": "cn-64k.txt",
    "code": "code-64k.txt",
    "agent_chat": "agent-64k.txt",
}
REF_OK_HEADER = b"_logits_"
CAL_CONTRACT_REF = "fidelity-calibration-v1"


@dataclass
class CalibrateConfig:
    source: Path
    imatrix_corpus: Path
    runtime_dir: Path
    eval_data_dir: Path
    out_dir: Path
    model_id: str
    imatrix_path: Path | None = None
    imatrix_arg: str | None = None
    chunks: int = 500
    n_gpu_layers: int = 99
    threads: int = 16
    workdir: Path | None = None
    on_disk: bool = False
    extra_presets: list[str] = field(default_factory=list)
    probe_budget: int = 4
    contract_path: Path | None = None
    log_dir: Path | None = None
    replay_existing: str | None = None
    replay_manifest: str | None = None

    def log(self, message: str) -> None:
        print(f"[calibrate] {message}", flush=True)


# ----------------------------------------------------------------- helpers


def _runtime_env(runtime_dir: Path) -> dict:
    """Environment for a runtime binary, with its libraries discoverable.

    Delegates to :func:`fit_gguf.llama_integration.runtime_env`, which also adds
    a sibling CUDA runtime directory — without it ``ggml-cuda.dll`` fails to load
    and llama.cpp silently evaluates on CPU instead.
    """
    return runtime_env(runtime_dir)


def _run(runtime: Path, cmd: list[str], log_path: Path, env: dict) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("wb") as handle:
        proc = subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT, env=env)
    return proc.returncode


def ref_ok(path: Path) -> bool:
    """b10666-era _logits_ arithmetic completeness check (ENOSPC guard)."""
    import struct

    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            head = handle.read(20)
        if len(head) < 20 or head[:8] != REF_OK_HEADER:
            return False
        n_ctx, n_vocab, n_chunk = struct.unpack("<III", head[8:20])
        nv = 2 * ((n_vocab + 1) // 2) + 4
        expect = 20 + n_chunk * n_ctx * 4 + n_chunk * (n_ctx - 1 - n_ctx // 2) * nv * 2
        return size == expect
    except Exception:
        return False


# ------------------------------------------------- hot-loop filesystem guard

# Filesystems whose buffered-write path is known to panic this kernel when fed
# the write pattern llama.cpp produces.
UNSAFE_HOT_LOOP_FSTYPES = frozenset({"ntfs3"})


def mount_fs_type(path: Path) -> str | None:
    """Filesystem type of the mount containing ``path`` (longest-prefix match)."""
    target = Path(path)
    try:
        target = target.resolve() if target.exists() else Path(os.path.abspath(target))
    except OSError:
        target = Path(os.path.abspath(target))
    best: tuple[int, str] | None = None
    try:
        mounts = Path("/proc/mounts").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    for line in mounts.splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        mount_point = Path(parts[1].replace("\\040", " "))
        if mount_point == target or mount_point in target.parents:
            depth = len(mount_point.parts)
            if best is None or depth > best[0]:
                best = (depth, parts[2])
    return best[1] if best else None


def default_scratch_root() -> Path:
    """Default scratch volume for the hot loop.

    Prefers ``FIT_CALIBRATE_TMP``, then the Linux tmpfs at ``/dev/shm`` (the
    guard's recommended target), and finally the platform temp directory: a
    literal ``/dev/shm`` does not exist on Windows, where ``Path("/dev/shm")``
    would silently resolve to a ``\\dev\\shm`` folder on the current drive.
    """
    override = os.environ.get("FIT_CALIBRATE_TMP")
    if override:
        return Path(override)
    shm = Path("/dev/shm")
    if shm.is_dir():
        return shm
    return Path(tempfile.gettempdir())


def assert_hot_loop_fs_safe(paths: dict[str, Path]) -> None:
    """Refuse to run the hot loop on a filesystem with a known write-path panic.

    ``_run`` redirects the llama.cpp subprocess's stderr straight into a log
    file. llama.cpp logs to stderr unbuffered, so that produces short, unaligned,
    page-spanning buffered writes — and on ntfs3 those trip
    ``kernel BUG at fs/iomap/buffered-io.c:1061`` in ``iomap_write_end``, taking
    the whole machine down (observed 2026-09-06 ×2 and 2026-09-10, every time
    with ``llama-quantize`` or ``llama-perplexity`` as the writing process).

    Bulk copies issued by ``cp``/``shutil`` to the same volume are unaffected,
    so the hot loop stages into the scratch volume and publishes afterwards.
    Set ``FIT_ALLOW_UNSAFE_FS=1`` to override (at your own risk).
    """
    if os.environ.get("FIT_ALLOW_UNSAFE_FS") == "1":
        return
    unsafe = {
        name: f"{path} ({mount_fs_type(path)})"
        for name, path in paths.items()
        if mount_fs_type(path) in UNSAFE_HOT_LOOP_FSTYPES
    }
    if unsafe:
        raise CalibrationError(
            "HOT_LOOP_FS_UNSAFE: refusing to point subprocess logs/references at "
            f"a filesystem with a known write-path kernel BUG: {unsafe}. "
            "Use a tmpfs scratch (default /dev/shm, or --workdir /dev/shm/...), "
            "or set FIT_ALLOW_UNSAFE_FS=1 to override."
        )


def publish_tree(src_dir: Path, dst_dir: Path, pattern: str) -> list[str]:
    """Bulk-copy ``src_dir``/``pattern`` into ``dst_dir`` (safe on ntfs3).

    Only files that are missing or differ in size are copied, so a resumed run
    does not re-copy an already-published tree. Returns published file names.
    """
    if not src_dir.is_dir():
        return []
    dst_dir.mkdir(parents=True, exist_ok=True)
    published: list[str] = []
    for src in sorted(src_dir.glob(pattern)):
        if not src.is_file():
            continue
        dst = dst_dir / src.name
        if dst.exists() and dst.stat().st_size == src.stat().st_size:
            continue
        shutil.copyfile(src, dst)
        published.append(src.name)
    return published


# -------------------------------------------------------------- observations


def eval_artifact(
    cfg: CalibrateConfig,
    artifact: Path,
    refs_dir: Path,
    point_id: str,
    env: dict,
) -> dict:
    """Five-domain eval-v1 evaluation of one artifact → observation dict."""
    metrics = {}
    for domain in DOMAINS:
        slice_file = cfg.eval_data_dir / f"kl-eval-{SLICE_SUFFIX[domain]}"
        ref = refs_dir / f"bf16-{domain}.kld"
        log = cfg.log_dir / f"eval-{point_id}-{domain}.log"
        parsed = None
        for attempt in (1, 2, 3):
            cmd = [
                str(resolve_runtime_binary(cfg.runtime_dir, "llama-perplexity")),
                "-m", str(artifact), "-f", str(slice_file),
                "-ngl", str(cfg.n_gpu_layers), "-t", str(cfg.threads),
                "-c", "512", "-b", "512",
                "--kl-divergence", "--kl-divergence-base", str(ref),
            ]
            rc = _run(cfg.runtime_dir, cmd, log, env)
            combined = log.read_text(errors="replace")
            try:
                parsed = parse_llama_kl_log(combined)
            except Exception:
                parsed = None
            if parsed is not None and rc == 0:
                break
            cfg.log(f"eval {point_id}/{domain} attempt {attempt} failed (rc={rc})")
            time.sleep(5 * attempt)
        if parsed is None or rc != 0:
            raise CalibrationError(f"eval failed: {point_id}/{domain}")
        metrics[domain] = parsed
    macro_kl = sum(m["mean_kld"] for m in metrics.values()) / len(metrics)
    macro_top = sum(m["same_top_pct"] for m in metrics.values()) / len(metrics) / 100.0
    return {
        "point_id": point_id,
        "size_bytes": artifact.stat().st_size,
        "artifact_sha256": sha256_file(artifact),
        "macro_kl": macro_kl,
        "same_top": macro_top,
        "per_domain": {d: {"mean_kld": metrics[d]["mean_kld"],
                           "same_top_pct": metrics[d]["same_top_pct"]}
                       for d in DOMAINS},
    }


# ------------------------------------------------------------------- stages


def stage_generate_imatrix(cfg: CalibrateConfig, env: dict) -> Path:
    if cfg.imatrix_path is not None:
        return cfg.imatrix_path
    work = cfg.workdir or cfg.out_dir
    imx = work / "calibration-imatrix.gguf"
    if imx.is_file() and imx.stat().st_size > 0:
        return imx
    cfg.log("generating imatrix (corpus, contract default chunks)")
    rc = _run(
        cfg.runtime_dir,
        [str(resolve_runtime_binary(cfg.runtime_dir, "llama-imatrix")), "-m", str(cfg.source),
         "-f", str(cfg.imatrix_corpus), "-c", "512", "-ngl", str(cfg.n_gpu_layers),
         "--chunks", str(cfg.chunks), "-o", str(imx)],
        cfg.log_dir / "imatrix.log",
        env,
    )
    if rc != 0 or not imx.is_file():
        raise CalibrationError("REF_GENERATION_FAILED: imatrix generation failed")
    return imx


def stage_imatrix_coverage(cfg: CalibrateConfig, imx: Path) -> list[str]:
    from fit_gguf.gguf import read_gguf_layout
    from fit_gguf.imatrix import load_imatrix_profile

    profile = load_imatrix_profile(imx)
    have = {e.name for e in profile.entries}
    layout = read_gguf_layout(cfg.source)
    missing = [
        t.name for t in layout.tensors
        if t.name not in have and "embd" not in t.name and not t.name.endswith("norm.weight")
    ]
    cfg.log(f"imatrix coverage: {len(have)} entries, {len(missing)} missing matrices")
    return missing


def stage_references(cfg: CalibrateConfig, env: dict, refs_dir: Path) -> dict:
    refs_dir.mkdir(parents=True, exist_ok=True)
    domains = {}
    corpus = {}
    for domain in DOMAINS:
        slice_file = cfg.eval_data_dir / f"kl-eval-{SLICE_SUFFIX[domain]}"
        raw = slice_file.read_bytes()
        corpus[domain] = {
            "corpus_sha256": hashlib.sha256(raw).hexdigest(),
            "raw_bytes": len(raw),
        }
        out = refs_dir / f"bf16-{domain}.kld"
        if out.is_file() and ref_ok(out):
            continue
        log = cfg.log_dir / f"ref-{domain}.log"
        ok = False
        for attempt in (1, 2, 3, 4):
            cfg.log(f"reference {domain} (attempt {attempt})")
            out.unlink(missing_ok=True)
            rc = _run(
                cfg.runtime_dir,
                [str(resolve_runtime_binary(cfg.runtime_dir, "llama-perplexity")), "-m", str(cfg.source),
                 "-f", str(slice_file), "-ngl", str(cfg.n_gpu_layers),
                 "-t", str(cfg.threads), "-c", "512", "-b", "512",
                 "--kl-divergence-base", str(out)],
                log, env,
            )
            if rc == 0 and out.is_file() and ref_ok(out):
                ok = True
                break
            time.sleep(10)
        if not ok:
            raise CalibrationError(f"REF_GENERATION_FAILED: {domain}")
    for domain in DOMAINS:
        slice_file = cfg.eval_data_dir / f"kl-eval-{SLICE_SUFFIX[domain]}"
        domains[domain] = {
            "corpus_sha256": corpus[domain]["corpus_sha256"],
            "raw_bytes": corpus[domain]["raw_bytes"],
            "reference_kld_sha256": sha256_file(refs_dir / f"bf16-{domain}.kld"),
        }
    return domains


def _load_curve_points(bundle: Path) -> list[dict]:
    curve = bundle / "curve-points.jsonl"
    if not curve.is_file():
        return []
    out = []
    for line in curve.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def _append_curve_point(bundle: Path, obs: dict) -> None:
    curve = bundle / "curve-points.jsonl"
    with curve.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(obs, sort_keys=True) + "\n")


def write_seed_material(
    bundle: Path,
    observations: list[dict],
    preset_names: set[str],
    *,
    reference_manifest_sha256: str,
    evaluator_contract_sha256: str,
) -> None:
    """Emit the size manifest + attested provenance sidecar for the ladder.

    ``fit fidelity-search`` admits bracket seeds from two files: a
    ``<name> <size> <sha256>`` manifest and a provenance sidecar attesting the
    frozen closure and the reference manifest. Without them the search cannot
    use a single point of the ladder this run just spent five-domain evals on,
    and re-deriving the two files by hand is exactly the per-model busywork the
    calibration line exists to remove.

    A native-preset point names its own preset on both window anchors: that is
    what marks a poison preset's artifact inadmissible as bracket evidence
    downstream (``_provenance_view``). Probe points carry no anchors — they are
    planned inside healthy windows by construction.
    """
    manifest_lines: list[str] = []
    provenance_lines: list[str] = []
    for obs in observations:
        point = str(obs["point_id"])
        size = int(obs["size_bytes"])
        manifest_lines.append(f"{point}  {size}  {obs['artifact_sha256']}")
        anchor = point if point in preset_names else ""
        provenance_lines.append(json.dumps({
            "name": point,
            "size_bytes": size,
            "window_lower_preset": anchor,
            "window_upper_preset": anchor,
            "attestation": "runtime-verified",
            "eval_contract_digest": evaluator_contract_sha256,
            "reference_manifest_sha256": reference_manifest_sha256,
        }, sort_keys=True))
    (bundle / "state-artifact-manifest.txt").write_text(
        "\n".join(manifest_lines) + ("\n" if manifest_lines else ""), encoding="utf-8")
    (bundle / "seed-provenance.jsonl").write_text(
        "\n".join(provenance_lines) + ("\n" if provenance_lines else ""), encoding="utf-8")


def stage_ladder(cfg: CalibrateConfig, env: dict, imx: Path, refs_dir: Path,
                 missing_matrices: list[str]) -> list[dict]:
    contract, _ = cal.load_contract(cfg.contract_path)
    presets = list(contract["ladder_standard_presets"]) + list(cfg.extra_presets)
    work = cfg.workdir or cfg.out_dir
    observations = _load_curve_points(cfg.out_dir)
    known = {o["point_id"] for o in observations}
    for preset in presets:
        if preset in known:
            cfg.log(f"ladder {preset}: already recorded, skipping")
            continue
        cfg.log(f"ladder {preset}: quantize")
        artifact = work / f"ladder-{preset}.gguf"
        rc = _run(
            cfg.runtime_dir,
            [str(resolve_runtime_binary(cfg.runtime_dir, "llama-quantize")), "--imatrix", str(imx),
             str(cfg.source), str(artifact), preset],
            cfg.log_dir / f"quantize-{preset}.log",
            env,
        )
        if rc != 0 or not artifact.is_file():
            cfg.log(f"ladder {preset}: quantize FAILED — point skipped")
            continue
        obs = eval_artifact(cfg, artifact, refs_dir, preset, env)
        observations.append(obs)
        _append_curve_point(cfg.out_dir, obs)
        artifact.unlink(missing_ok=True)
    return observations


def _largest_uncovered_gap(pool: list[dict], lo: float, hi: float) -> float | None:
    """Midpoint of the largest KL interval inside the window that still has no
    observation in it. Probe feedback participates: probes are real curve
    points, so repeated targets cannot occur."""
    kls = sorted({o["macro_kl"] for o in pool} | {lo, hi})
    best: tuple[float, float] | None = None
    for a, b in zip(kls, kls[1:]):
        a2, b2 = max(a, lo), min(b, hi)
        if b2 > a2 and (best is None or b2 - a2 > best[1] - best[0]):
            best = (a2, b2)
    if best is None:
        return None
    return (best[0] + best[1]) / 2.0


def stage_gap_probes(
    cfg: CalibrateConfig,
    env: dict,
    imx: Path,
    refs_dir: Path,
    observations: list[dict],
    contract: dict,
    missing_matrices: list[str],
) -> tuple[list[dict], list[str]]:
    """§4: fill thin windows with plain-FIT probes; feedback-based targeting —
    the probe target is the midpoint of the largest uncovered KL gap inside
    the window, interpolated on the observed KL↔size curve (probes included),
    while dry-run anchors stay named presets bracketing the target size.
    Returns new observations + unresolved tiers."""
    from fit_gguf.pipeline import PRESET_FILE_TYPES
    from fit_gguf.pipeline import analyze as pipeline_analyze
    from fit_gguf.pipeline import plan as pipeline_plan
    from fit_gguf.pipeline import quantize as pipeline_quantize

    def preset_anchor(o: dict) -> str | None:
        """Probe observations are window evidence but carry no single named
        preset, so they can never anchor a pipeline_analyze dry-run pair."""
        pid = o["point_id"]
        if pid in PRESET_FILE_TYPES and pid not in contract["poison_presets"]:
            return pid
        return None

    def probes_used(pool: list[dict], tier: str) -> int:
        return sum(1 for o in pool if (o.get("probe") or {}).get("tier") == tier)

    new_obs: list[dict] = []
    unresolved: list[str] = []
    imatrix_arg = cfg.imatrix_arg or str(imx)
    for tier, anchor in contract["tier_kl_anchors"].items():
        anchor = float(anchor)
        lo, hi = cal.window_bounds(anchor, contract)
        budget = int(cfg.probe_budget)
        pool = observations + new_obs
        in_window = [o for o in pool if lo <= o["macro_kl"] <= hi]
        # poison presets never count toward window evidence (§3)
        in_window = [o for o in in_window if o["point_id"] not in contract["poison_presets"]]
        if len(in_window) >= int(contract["validation_min_samples"]):
            continue
        if len(in_window) == 0:
            # NOT_REACHABLE: no bracketing pair around an empty window
            below = [o for o in pool if o["macro_kl"] < lo]
            above = [o for o in pool if o["macro_kl"] > hi]
            if not below or not above:
                unresolved.append(tier)
                cfg.log(f"gap {tier}: NOT_REACHABLE (no bracketing observations)")
                continue
        cfg.log(f"gap {tier}: window [{lo:.4f},{hi:.4f}] n={len(in_window)} — probing")
        probes = probes_used(pool, tier)
        while probes < budget:
            pool = observations + new_obs
            in_window = [o for o in pool
                         if lo <= o["macro_kl"] <= hi
                         and o["point_id"] not in contract["poison_presets"]]
            if len(in_window) >= int(contract["validation_min_samples"]):
                break
            target_kl = _largest_uncovered_gap(pool, lo, hi)
            if target_kl is None:
                # every KL slot inside the window is occupied but samples are
                # still short of the minimum — nothing left to place
                unresolved.append(tier)
                break
            # invert KL→size on the two observed curve points bracketing the
            # target (any observation type: probes are legitimate curve points)
            by_kl = sorted(pool, key=lambda o: o["macro_kl"])
            lower = max((o for o in by_kl if o["macro_kl"] <= target_kl),
                        key=lambda o: o["macro_kl"])
            upper = min((o for o in by_kl if o["macro_kl"] >= target_kl),
                        key=lambda o: o["macro_kl"])
            span_kl = upper["macro_kl"] - lower["macro_kl"]
            if span_kl <= 0:
                unresolved.append(tier)
                break
            frac = (target_kl - lower["macro_kl"]) / span_kl
            target = int(lower["size_bytes"] + frac * (upper["size_bytes"] - lower["size_bytes"]))
            # dry-run anchors must be named presets bracketing the target size
            preset_obs = sorted([o for o in pool if preset_anchor(o)],
                                key=lambda o: o["size_bytes"])
            below_p = [o for o in preset_obs if o["size_bytes"] <= target]
            above_p = [o for o in preset_obs if o["size_bytes"] > target]
            if not below_p or not above_p:
                unresolved.append(tier)
                break
            lower_preset = below_p[-1]["point_id"]
            upper_preset = above_p[0]["point_id"]
            analysis_dir = cfg.out_dir / "analysis" / f"{lower_preset}-{upper_preset}"
            analysis_json = analysis_dir / "analysis.json"
            if not analysis_json.is_file():
                pipeline_analyze(
                    cfg.source, imx, cfg.runtime_dir, analysis_dir,
                    lower_preset=lower_preset, upper_preset=upper_preset,
                    imatrix_arg=imatrix_arg,
                )
            tag = f"probe-{tier}-{probes + 1}"
            plan_prefix = cfg.out_dir / "probes" / tag
            record = pipeline_plan(
                analysis_json, plan_prefix, target_bytes=target,
                policy="balanced", model_name=cfg.model_id,
            )
            artifact = (cfg.workdir or cfg.out_dir) / f"{tag}.gguf"
            pipeline_quantize(
                analysis_json, plan_prefix.parent / f"{tag}-tensor-types.txt",
                artifact, imatrix_arg=imatrix_arg,
            )
            probes += 1
            try:
                obs = eval_artifact(cfg, artifact, refs_dir, tag, env)
            except CalibrationError:
                artifact.unlink(missing_ok=True)
                continue
            obs["point_id"] = tag
            obs["probe"] = {"tier": tier, "target_bytes": target,
                            "plan_sha256": sha256_file(Path(str(plan_prefix) + "-plan.json")),
                            "recipe_sha256": sha256_file(Path(str(plan_prefix) + "-tensor-types.txt"))}
            new_obs.append(obs)
            _append_curve_point(cfg.out_dir, obs)
            artifact.unlink(missing_ok=True)
            cfg.log(f"gap {tier}: probe {tag} kld={obs['macro_kl']:.4f} top={obs['same_top']:.4f}")
        pool = observations + new_obs
        in_window = [o for o in pool if lo <= o["macro_kl"] <= hi
                     and o["point_id"] not in contract["poison_presets"]]
        if len(in_window) < int(contract["validation_min_samples"]):
            unresolved.append(tier)
    return new_obs, unresolved


def stage_floors(cfg: CalibrateConfig, observations: list[dict], contract: dict,
                 open_failures: list[str]) -> dict:
    # poison presets are diagnostics only — never floor samples (§3)
    usable = [o for o in observations if o["point_id"] not in contract["poison_presets"]]
    return cal.evaluate_observations(usable, contract, extra_failures=open_failures)


def stage_emit(
    cfg: CalibrateConfig,
    contract: dict,
    contract_sha: str,
    evaluation: dict,
    observations: list[dict],
    refs_dir: Path,
    domains: dict,
    missing_matrices: list[str],
    unresolved: list[str],
) -> Path:
    """Emit the Calibration Bundle (GPT-specified eight-piece layout)."""
    from fit_gguf.eval.contract import contract_digest
    from fit_gguf.fidelity import profile_hash

    bundle = cfg.out_dir
    bundle.mkdir(parents=True, exist_ok=True)
    source_sha = sha256_file(cfg.source)

    guard = {
        "calibration_contract": CAL_CONTRACT_REF,
        "calibration_contract_sha256": contract_sha,
        "evaluator_contract": "eval-v1",
        "generalization_scope": "exact_model_only",
        "guard_profile_id": f"guard-{cfg.model_id}-exact-v1",
        "guard_profile_version": 1,
        "scope": {"identifier": cfg.model_id, "type": "exact_model"},
        "source_sha256": source_sha,
        "status": evaluation["overall_status"],
        "tiers": {},
        "validation_basis": "fidelity-calibration-v1",
    }
    for tier, tr in evaluation["tiers"].items():
        guard["tiers"][tier] = {
            "kl_anchor": tr["anchor"],
            "same_top_floor": tr["floor"],
            "floor_method": tr["floor_method"],
            "sample_count": tr["sample_count"],
            "witness": tr["witness"],
            "validation_status": tr["validation_status"],
        }
    guard["confidence"] = {
        "note": "fidelity-calibration-v1 onboarding; floor_method/sample_count per tier",
        "window_n": {t: tr["sample_count"] for t, tr in evaluation["tiers"].items()},
    }
    guard["profile_hash"] = profile_hash(guard)
    guard_path = bundle / "guard-profile.yaml"
    guard_path.write_text(yaml.safe_dump(guard, sort_keys=False, allow_unicode=True),
                          encoding="utf-8")

    manifest = {
        "manifest_schema": "fit.eval_reference_manifest.v1",
        "evaluator_contract": "eval-v1",
        "evaluator_contract_hash": contract["evaluator_contract_sha256"],
        "source_bf16_gguf_sha256": source_sha,
        "domains": domains,
        "runtime_provenance": {
            "runtime_dir": str(cfg.runtime_dir),
            "execution": {"n_gpu_layers": cfg.n_gpu_layers, "threads": cfg.threads,
                          "on_disk": cfg.on_disk},
        },
    }
    manifest_path = bundle / "reference-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")

    curve_path = bundle / "curve-points.jsonl"
    if not (curve_path.is_file() and cfg.out_dir == bundle):
        with curve_path.open("w", encoding="utf-8") as handle:
            for obs in observations:
                handle.write(json.dumps(obs, sort_keys=True) + "\n")

    # Make the bundle self-consuming for the product search: the ladder this run
    # just evaluated becomes budget-free bracket evidence instead of being
    # re-derived by hand for every model.
    write_seed_material(
        bundle,
        observations,
        set(contract["ladder_standard_presets"]) | set(cfg.extra_presets),
        reference_manifest_sha256=sha256_file(manifest_path),
        evaluator_contract_sha256=contract_digest(),
    )

    record = {
        "schema": "fit.calibration_record.v1",
        "contract_id": cal.CONTRACT_ID,
        "calibration_contract_sha256": contract_sha,
        "model_id": cfg.model_id,
        "created": time.strftime("%Y-%m-%d"),
        "inputs": {
            "source_weights_sha256": source_sha,
            "imatrix_corpus_sha256": sha256_file(cfg.imatrix_corpus),
            "imatrix_chunks": cfg.chunks,
            "evaluator_contract_sha256": contract["evaluator_contract_sha256"],
            "runtime_dir": str(cfg.runtime_dir),
        },
        "process": {
            "imatrix_missing_matrices": missing_matrices,
            "unresolved_tiers": unresolved,
            "curve_points": len(observations),
        },
        "derivation": {
            k: {kk: vv for kk, vv in v.items() if kk != "samples"}
            for k, v in evaluation["tiers"].items()
        },
        "artifacts": {
            "guard_profile_sha256": sha256_file(guard_path),
            "reference_manifest_sha256": sha256_file(manifest_path),
            "curve_points_sha256": sha256_file(curve_path),
        },
        "failures": evaluation["open_failures"],
        "overall_status": evaluation["overall_status"],
    }
    record_path = bundle / "calibration-record.json"
    record_path.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")

    registry_entry = {
        "registry_schema": REGISTRY_SCHEMA,
        "model_id": cfg.model_id,
        "source_weights_sha256": source_sha,
        "tokenizer_sha256": None,
        "evaluator_contract": "eval-v1",
        "evaluator_contract_sha256": contract["evaluator_contract_sha256"],
        "reference_manifest": {"path": "reference-manifest.json",
                               "sha256": sha256_file(manifest_path)},
        "guard_profile": {"path": "guard-profile.yaml", "sha256": sha256_file(guard_path)},
        "calibration": {
            "basis": cal.CONTRACT_ID,
            "contract": cal.CONTRACT_ID,
            "contract_sha256": contract_sha,
            "record": {"path": "calibration-record.json", "sha256": sha256_file(record_path)},
        },
        "scope": "exact_model",
        "status": evaluation["overall_status"],
        "added_in": "PENDING",
    }
    registry_entry["entry_sha256"] = entry_digest(registry_entry)
    (bundle / "registry-entry.json").write_bytes(canonical_json_bytes(registry_entry))

    report_lines = [
        f"# Calibration report — {cfg.model_id}",
        "",
        f"- contract: fidelity-calibration-v1 ({contract_sha[:16]}…)",
        f"- source: {source_sha[:16]}…",
        f"- overall: **{evaluation['overall_status']}**",
        f"- open failures: {evaluation['open_failures'] or 'none'}",
        "",
        "| tier | window | n | floor | method | witness | status |",
        "|---|---|---|---|---|---|---|",
    ]
    for tier, tr in evaluation["tiers"].items():
        report_lines.append(
            f"| {tier} | [{tr['window'][0]}, {tr['window'][1]}] | {tr['sample_count']} "
            f"| {tr['floor']} | {tr['floor_method']} "
            f"| {'✓' if tr['witness'] else '✗'} | {tr['validation_status']} |"
        )
    (bundle / "profile-report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    sums = []
    for f in sorted(bundle.iterdir()):
        if f.is_file():
            sums.append(f"{sha256_file(f)}  {f.name}")
    (bundle / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf-8")

    admission = validate_bundle(bundle)
    cfg.log(f"bundle admissible: {admission['admissible']} ({admission['status']})")
    return bundle


# ------------------------------------------------------------------ replay


def replay_existing(cfg: CalibrateConfig, observations_path: Path,
                    manifest_path: Path) -> dict:
    """Gate P2-A: zero-eval re-derivation from recorded observations."""
    contract, contract_sha = cal.load_contract(cfg.contract_path)
    payload = json.loads(observations_path.read_text())
    points = payload["points"] if isinstance(payload, dict) else payload
    shas = {}
    for line in Path(manifest_path).read_text().splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[0].startswith(cfg.model_id + "-"):
            shas[parts[0][len(cfg.model_id) + 1:]] = parts[2]
    observations = [
        {
            "point_id": p["point"],
            "size_bytes": int(p["size_bytes"]),
            "artifact_sha256": shas.get(p["point"], "0" * 64),
            "macro_kl": float(p["macro_kl"]),
            "same_top": float(p["same_top"]),
        }
        for p in points
    ]
    usable = [o for o in observations if o["point_id"] not in contract["poison_presets"]]
    return cal.evaluate_observations(usable, contract)


# -------------------------------------------------------------------- main


def run_calibrate(cfg: CalibrateConfig) -> dict:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    env = _runtime_env(cfg.runtime_dir)

    if cfg.replay_existing is not None:
        manifest = Path(cfg.replay_manifest) if getattr(cfg, "replay_manifest", None)             else cfg.out_dir / "state-artifact-manifest.txt"
        report = replay_existing(cfg, Path(cfg.replay_existing), manifest)
        (cfg.out_dir / "replay-report.json").write_text(
            json.dumps(report, indent=1) + "\n", encoding="utf-8")
        cfg.log(f"replay overall={report['overall_status']} failures={report['open_failures']}")
        return {"mode": "replay", "report": report}

    contract, contract_sha = cal.load_contract(cfg.contract_path)
    from fit_gguf.eval.contract import contract_digest

    if contract["evaluator_contract_sha256"] != contract_digest():
        raise CalibrationError("contract pins a different eval-v1 digest than the live closure")

    work = cfg.workdir
    if cfg.on_disk:
        work = work or cfg.out_dir / "work"
    else:
        work = work or default_scratch_root() / f"cal-{cfg.model_id}"
    work.mkdir(parents=True, exist_ok=True)
    cfg.workdir = work

    # The hot loop emits two things the subprocesses write themselves: their own
    # log output (llama.cpp logs to stderr unbuffered → short unaligned writes)
    # and, on a fresh model, the reference logits. Both stay on the scratch
    # volume and are published into the bundle once the run is done — see
    # assert_hot_loop_fs_safe for the panic this avoids.
    if cfg.log_dir is None:
        cfg.log_dir = work / "logs"
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    scratch_refs = work / "references"
    published_refs = cfg.out_dir / "references"
    assert_hot_loop_fs_safe({"logs": cfg.log_dir, "references": scratch_refs})

    # Reading an already-published bundle is safe, so reuse verified references
    # from a previous run instead of regenerating them.
    if published_refs.is_dir():
        for ref in sorted(published_refs.glob("bf16-*.kld")):
            if ref_ok(ref):
                publish_tree(published_refs, scratch_refs, ref.name)

    source_sha = sha256_file(cfg.source)
    cfg.log(f"source {cfg.source.name} sha={source_sha[:16]}…")

    imx = stage_generate_imatrix(cfg, env)
    missing = stage_imatrix_coverage(cfg, imx)
    refs_dir = scratch_refs
    domains = stage_references(cfg, env, refs_dir)
    observations = stage_ladder(cfg, env, imx, refs_dir, missing)
    new_obs, unresolved = stage_gap_probes(
        cfg, env, imx, refs_dir, observations, contract, missing
    )
    # INSUFFICIENT_WINDOW is tier-local and derived per-tier by the contract
    # evaluator; only session-hard failures (e.g. imatrix coverage) may demote
    # every tier — passing tier-local failures here would taint healthy tiers.
    evaluation = stage_floors(cfg, observations + new_obs, contract, None)
    if unresolved:
        for tier in unresolved:
            evaluation["tiers"][tier]["failure"] = (
                evaluation["tiers"][tier]["failure"] or "INSUFFICIENT_WINDOW"
            )
            evaluation["tiers"][tier]["validation_status"] = "candidate"
        evaluation["open_failures"] = sorted(set(evaluation["open_failures"]) | {"INSUFFICIENT_WINDOW"})
        evaluation["overall_status"] = "candidate"

    # Publish scratch → bundle. These are bulk copies from this process, which
    # do not reproduce the subprocess write pattern the guard above rejects.
    publish_tree(refs_dir, published_refs, "bf16-*.kld")
    if cfg.log_dir.resolve() != (cfg.out_dir / "logs").resolve():
        publish_tree(cfg.log_dir, cfg.out_dir / "logs", "*.log")

    imx_copy = cfg.out_dir / "calibration-imatrix.gguf"
    if not (imx_copy.exists() and imx_copy.samefile(imx)):
        shutil.copyfile(imx, imx_copy)
    bundle = stage_emit(
        cfg, contract, contract_sha, evaluation, observations + new_obs,
        refs_dir, domains, missing, unresolved,
    )
    if work.resolve() != cfg.out_dir.resolve():
        shutil.rmtree(work, ignore_errors=True)
    return {
        "mode": "full",
        "bundle": str(bundle),
        "report": evaluation,
        "source_sha256": source_sha,
    }
