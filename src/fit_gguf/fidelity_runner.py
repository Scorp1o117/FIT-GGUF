"""Concrete Fidelity Search runner: healthy window selection, seed loading,
and the plan -> quantize -> eval-v1 executor wired to llama-perplexity.

The search state machine lives in :mod:`fit_gguf.fidelity_search` (pure,
injectable evaluator). This module is the b10666/eval-v1 binding:

* **Healthy frontier** (GPT ruling §12): poison presets (Q3_K_S outlier,
  IQ2_XS poison transition) are never chosen as a window's lower bound and
  their artifacts are rejected as bracket seeds — a poison-window FAIL must
  not masquerade as evidence about healthy recipes at that size.
* Window selection snaps a requested probe size to the narrowest healthy
  window that contains (or can be extended to contain) it; the executor
  plans at the *actual* planable size and reports it, so bracket arithmetic
  always uses real sizes.
* Every fresh evaluation is recorded (plan/recipe JSONs on disk, eval logs,
  artifact manifest hash line); the tmpfs artifact is deleted after eval.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from fit_gguf.eval.contract import DOMAINS
from fit_gguf.eval.provenance import EvalProvenance
from fit_gguf.eval.results import parse_llama_kl_log
from fit_gguf.fidelity import KL_ANCHORS, require_guard_profile
from fit_gguf.fidelity_search import EvalOutcome, Seed, TierContract, fidelity_search
from fit_gguf.pipeline import plan as pipeline_plan
from fit_gguf.pipeline import quantize as pipeline_quantize

# GPT ruling §12: known poison presets — never a window lower bound, never a seed.
POISON_PRESETS = ("Q3_K_S", "IQ2_XS")


class FidelityRunnerError(ValueError):
    """Raised when the runner cannot select a healthy window or execute."""


@dataclass(frozen=True, slots=True)
class Window:
    analysis_path: Path
    lower_preset: str
    upper_preset: str
    lower_size: int
    upper_size: int

    @property
    def healthy(self) -> bool:
        return self.lower_preset.upper() not in POISON_PRESETS


@dataclass(frozen=True, slots=True)
class RunnerConfig:
    runtime: Path
    imatrix: Path  # must match the analyze-time --imatrix-arg string (G2 contract)
    refs_dir: Path
    eval_data_dir: Path
    work_dir: Path  # tmpfs artifact scratch
    out_dir: Path  # plan/recipe/audit records
    model_name: str
    guard_registry: Path | None
    refine_profile: Path | None
    threads: int = 16
    # release-gate hardening: when require_eval_provenance is set the executor
    # refuses to run a single evaluation unless the frozen eval-v1 closure
    # (contract digest + reference .kld SHAs + corpus SHAs) was verified and
    # attached as an EvalProvenance
    eval_provenance: EvalProvenance | None = None
    require_eval_provenance: bool = False
    # weights digest binding for Guard Profiles that pin source_sha256
    source_sha256: str | None = None


def resolve_contract(
    model_name: str,
    tier: str,
    guard_registry: str | Path,
    source_sha256: str | None = None,
) -> TierContract:
    """Dual hard gate: frozen KL Core anchor + validated Guard Profile floor.

    Raises GuardProfileError (the CLI hard refusal) for unvalidated models.
    Profiles that pin ``source_sha256`` match only the same weights digest —
    callers that omit it are treated as unvalidated for such profiles.
    """
    tier_key = tier.strip().lower()
    profile = require_guard_profile(model_name, tier_key, guard_registry, source_sha256)
    return TierContract(
        tier=tier_key,
        kl_anchor=KL_ANCHORS[tier_key],
        same_top_floor=profile.floor_for(tier_key),
    )


def discover_windows(analysis_dirs: list[str | Path]) -> list[Window]:
    """Load preset-pair windows from analysis directories."""
    windows: list[Window] = []
    for raw in analysis_dirs:
        path = Path(raw)
        analysis_file = path / "analysis.json" if path.is_dir() else path
        if not analysis_file.is_file():
            raise FidelityRunnerError(f"analysis not found: {analysis_file}")
        payload = json.loads(analysis_file.read_text())
        lower = payload["presets"]["lower"]
        upper = payload["presets"]["upper"]
        windows.append(
            Window(
                analysis_path=analysis_file.parent,
                lower_preset=str(lower["name"]),
                upper_preset=str(upper["name"]),
                lower_size=int(lower["predicted_size_bytes"]),
                upper_size=int(upper["predicted_size_bytes"]),
            )
        )
    return sorted(windows, key=lambda w: w.lower_size)


def select_window(target: int, windows: list[Window], *, allow_extend: int = 768 * 1024 * 1024) -> tuple[Window, int]:
    """Pick the narrowest healthy window covering ``target`` (with snap).

    Returns ``(window, planned_size)``. If the target falls in a coverage gap,
    it snaps to the nearest edge of the closest healthy window within
    ``allow_extend`` (coarse probes may cross window gaps; the final
    tolerance-level resolution happens inside one window); the caller must
    plan at the returned size so bracket arithmetic uses real, planable sizes.
    """
    healthy = [w for w in windows if w.healthy]
    if not healthy:
        raise FidelityRunnerError("no healthy windows available (poison presets rejected)")
    covering = [w for w in healthy if w.lower_size <= target <= w.upper_size]
    if covering:
        window = min(covering, key=lambda w: w.upper_size - w.lower_size)
        return window, target
    # snap: nearest healthy window edge
    best: tuple[int, Window, int] | None = None  # (distance, window, planned)
    for w in healthy:
        for edge in (w.lower_size, w.upper_size):
            distance = abs(edge - target)
            if distance <= allow_extend and (best is None or distance < best[0]):
                best = (distance, w, edge)
    if best is None:
        raise FidelityRunnerError(
            f"target {target:,} is >{allow_extend // (1024 * 1024)}MiB away from every healthy window edge"
        )
    return best[1], best[2]


def load_seeds(
    manifest_path: str | Path,
    logs_dir: str | Path,
    model_prefix: str,
    *,
    poison_names: tuple[str, ...] = POISON_PRESETS,
    exclude_names: tuple[str, ...] = (),
    provenance_path: str | Path | None = None,
    require_seed_provenance: bool = False,
    reference_manifest_sha256: str | None = None,
) -> tuple[Seed, ...]:
    """Collect budget-free observed points from eval logs + size manifest.

    A seed is admitted only if (a) its artifact size is known from the
    manifest, (b) all five domains parsed, (c) it is not a poison-preset
    artifact, and (d) it is not on the explicit ``exclude_names`` list —
    poison-WINDOW products (e.g. an artifact whose recipe was filled from a
    Q3_K_S floor) have clean names but their FAILs must not become bracket
    evidence about healthy recipes at that size (GPT ruling §12).
    """
    manifest = Path(manifest_path)
    logs = Path(logs_dir)
    if require_seed_provenance and reference_manifest_sha256 is None:
        raise FidelityRunnerError(
            "require_seed_provenance needs the active verified reference "
            "manifest SHA — strict seed admission binds seeds to BOTH the "
            "contract digest and the reference manifest"
        )
    provenance_file = (
        Path(provenance_path) if provenance_path else manifest.parent / "seed-provenance.jsonl"
    )
    tainted, stale, attested, manifest_by_name = _provenance_view(provenance_file)
    by_point: dict[str, dict[str, dict]] = {}
    for entry in sorted(logs.iterdir()):
        name = entry.name
        if not name.startswith("eval-") or not name.endswith(".log"):
            continue
        body = name[len("eval-"):-len(".log")]
        if not body.startswith(model_prefix):
            continue
        domain_match = re.search(r"-(" + "|".join(DOMAINS) + r")$", body)
        if domain_match is None:
            continue
        point = body[: -len(domain_match.group(0))][len(model_prefix):]
        if not point:
            continue
        try:
            metrics = parse_llama_kl_log(entry.read_text(errors="replace"))
        except Exception:  # noqa: BLE001 — unparsable logs are simply not seeds
            continue
        by_point.setdefault(point, {})[domain_match.group(1)] = metrics

    sizes: dict[str, int] = {}
    for line in manifest.read_text().splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[0].startswith(model_prefix):
            sizes[parts[0][len(model_prefix):]] = int(parts[1])

    seeds: list[Seed] = []
    for point, domains in sorted(by_point.items()):
        if len(domains) != len(DOMAINS) or point not in sizes:
            continue
        # poison-preset artifacts are excluded from bracket evidence
        if any(poison.lower() in point.lower() for poison in poison_names):
            continue
        if point in exclude_names:
            continue
        full_name = f"{model_prefix}{point}"
        if point in tainted or full_name in tainted:
            continue
        if point in stale or full_name in stale:
            continue
        if require_seed_provenance:
            # release path: every bracket seed must carry a provenance sidecar
            # attesting the SAME frozen eval closure AND the SAME reference
            # manifest — a seed produced against different references under
            # the same contract is inadmissible (Codex audit round 3)
            if not (point in attested or full_name in attested):
                continue
            if manifest_by_name.get(point, manifest_by_name.get(full_name)) != reference_manifest_sha256:
                continue
        macro_kl = sum(domains[d]["mean_kld"] for d in DOMAINS) / len(DOMAINS)
        macro_top = sum(domains[d]["same_top_pct"] for d in DOMAINS) / len(DOMAINS) / 100.0
        seeds.append(Seed(size_bytes=sizes[point], macro_kl=macro_kl, same_top=macro_top))
    return tuple(seeds)


class SearchExecutor:
    """plan -> quantize (tmpfs) -> eval-v1 -> record -> release, per probe."""

    def __init__(
        self,
        config: RunnerConfig,
        windows: list[Window],
        contract: TierContract,
        manifest_path: Path,
        logs_out_dir: Path,
        known_sizes: frozenset[int] | set[int] = frozenset(),
        max_known_fail: int | None = None,
        min_known_pass: int | None = None,
        provenance_path: Path | None = None,
    ) -> None:
        if config.require_eval_provenance and config.eval_provenance is None:
            raise FidelityRunnerError(
                "require_eval_provenance is set but no verified EvalProvenance "
                "was attached — refusing to evaluate against unverified inputs"
            )
        self.config = config
        self.windows = windows
        self.contract = contract
        self.manifest_path = manifest_path
        self.logs_out_dir = logs_out_dir
        self.counter = 0
        self._delivered: dict[int, EvalOutcome] = {}  # delivered size -> outcome
        self._recipe_by_size: dict[int, Path] = {}  # delivered size -> tensor-types file
        self._analysis_by_size: dict[int, Path] = {}  # delivered size -> analysis dir
        self._known_sizes: set[int] = set(known_sizes)
        # probes outside (max_known_fail, min_known_pass) can never tighten the
        # bracket — the bump loop skips them to conserve eval budget
        self._max_known_fail: int | None = max_known_fail
        self._min_known_pass: int | None = min_known_pass
        self._provenance_path = provenance_path
        # artifact names must be unique ACROSS runs: manifest lines and eval
        # logs are keyed by name, and a reused name would pair an old size
        # with new metrics (phantom points)
        self._run_id = time.strftime("%m%d-%H%M%S")
        # fresh scratch per executor: stale partial artifacts must never be
        # mistaken for a completed quantize (relaunch safety)
        shutil.rmtree(config.work_dir, ignore_errors=True)
        config.work_dir.mkdir(parents=True, exist_ok=True)
        config.out_dir.mkdir(parents=True, exist_ok=True)
        logs_out_dir.mkdir(parents=True, exist_ok=True)

    def _plan_deliverable(self, target: int, window: Window, tag: str) -> tuple[dict, int, Path]:
        """Plan at ``target``; if the delivered size is already known (tested
        here or among the observed seeds), bump the target upward (the
        balanced allocator stalls below coarse candidate steps) until a fresh
        deliverable or the window top.

        Returns (plan record, deliverable size, plan prefix). The accepted
        probe plan's tensor-types file IS the quantize input — re-planning at
        the deliverable would select a different recipe and undershoot again.
        """
        bump = 128 * 1024 * 1024
        current = target
        record = None
        deliverable = None
        prefix = self.config.out_dir / f"{tag}-plan"
        for _ in range(8):
            record = pipeline_plan(
                window.analysis_path / "analysis.json",
                prefix,
                target_bytes=current,
                policy="balanced",
                model_name=self.config.model_name,
                refine_profile=self.config.refine_profile,
                fidelity_tier=self.contract.tier,
                guard_registry=self.config.guard_registry,
                source_sha256=self.config.source_sha256,
            )
            deliverable = int(record["predicted_size_bytes"])
            useless = (
                deliverable in self._delivered
                or deliverable in self._known_sizes
                or (self._max_known_fail is not None and deliverable <= self._max_known_fail)
                or (self._min_known_pass is not None and deliverable >= self._min_known_pass)
            )
            if not useless:
                return record, deliverable, prefix
            current = min(window.upper_size, current + bump)
            self._log(f"{tag}: deliverable {deliverable:,} cannot tighten the "
                      f"bracket; bumping target to {current:,}")
        return record, deliverable, prefix

    def _log(self, message: str) -> None:
        line = f"{time.strftime('%m-%d %H:%M:%S')}  {message}"
        print(line, flush=True)
        with (self.config.out_dir / "runner.log").open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def _eval_domains(self, artifact: Path, tag: str) -> dict[str, dict] | None:
        """Run the five eval-v1 domain evaluations; None on any failure.

        Also writes per-domain logs under logs_out_dir (seeds for later runs).
        """
        metrics: dict[str, dict] = {}
        for domain in DOMAINS:
            log_path = self.logs_out_dir / f"eval-{tag}-{domain}.log"
            slice_file = self.config.eval_data_dir / f"kl-eval-{_SLICE_SUFFIX[domain]}"
            ref_file = self.config.refs_dir / f"bf16-{domain}.kld"
            for attempt in (1, 2, 3):
                result = subprocess.run(  # noqa: S603 — fixed argv, no shell
                    [
                        str(self.config.runtime / "llama-perplexity"),
                        "-m", str(artifact),
                        "-f", str(slice_file),
                        "-ngl", "99",
                        "-t", str(self.config.threads),
                        "-c", "512", "-b", "512",
                        "--kl-divergence",
                        "--kl-divergence-base", str(ref_file),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=3600,
                )
                # llama.cpp writes the KL statistics to stderr with irregular
                # spacing ("Mean    KLD:") — the parser's own regex is the
                # single acceptance criterion, not a substring guess. A clean
                # parse with a nonzero exit code is still a failed eval.
                combined = result.stdout + result.stderr
                try:
                    parsed = parse_llama_kl_log(combined)
                except Exception:  # noqa: BLE001 — incomplete eval output: retry
                    self._log(f"{tag}: eval {domain} attempt {attempt} failed "
                              f"(rc={result.returncode}); retrying")
                    time.sleep(10 * attempt)
                else:
                    if result.returncode != 0:
                        self._log(f"{tag}: eval {domain} attempt {attempt} parsed "
                                  f"but exited rc={result.returncode}; retrying")
                        time.sleep(10 * attempt)
                        continue
                    metrics[domain] = parsed
                    log_path.write_text(combined, encoding="utf-8")
                    break
            else:
                self._log(f"{tag}: eval {domain} FAILED")
                return None
        return metrics

    def evaluate(self, target_bytes: int) -> EvalOutcome:
        self.counter += 1
        tag = f"{self.config.model_name}-{self._run_id}-FS{self.counter:02d}"
        window, planned = select_window(target_bytes, self.windows)
        if planned != target_bytes:
            self._log(f"{tag}: target {target_bytes:,} snapped to {planned:,} "
                      f"({window.lower_preset}->{window.upper_preset})")
        record, deliverable, plan_prefix = self._plan_deliverable(planned, window, tag)
        if deliverable in self._delivered:
            self._log(f"{tag}: window exhausted near target — returning cached "
                      f"outcome for {deliverable:,}")
            return self._delivered[deliverable]
        self._log(f"{tag}: plan deliverable {deliverable:,} "
                  f"(target {planned:,}) in {window.lower_preset}->{window.upper_preset}")
        artifact = self.config.work_dir / f"{tag}.gguf"
        pipeline_quantize(
            window.analysis_path / "analysis.json",
            self.config.out_dir / f"{tag}-plan-tensor-types.txt",
            artifact,
            imatrix_arg=str(self.config.imatrix),
        )
        actual = artifact.stat().st_size

        metrics = self._eval_domains(artifact, tag)
        if metrics is None:
            return EvalOutcome(size_bytes=actual, macro_kl=None, same_top=None,
                               note="eval failed")

        macro_kl = sum(metrics[d]["mean_kld"] for d in DOMAINS) / len(DOMAINS)
        macro_top = sum(metrics[d]["same_top_pct"] for d in DOMAINS) / len(DOMAINS)
        passed = self.contract.passes(macro_kl, macro_top / 100.0)
        self._log(f"{tag}: kld {macro_kl:.4f} top {macro_top:.2f} "
                  f"{'PASS' if passed else 'FAIL'} @ {actual:,}")

        outcome = EvalOutcome(size_bytes=actual, macro_kl=macro_kl, same_top=macro_top / 100.0)
        self._delivered[actual] = outcome
        self._recipe_by_size[actual] = self.config.out_dir / f"{tag}-plan-tensor-types.txt"
        self._analysis_by_size[actual] = window.analysis_path
        if self._provenance_path is not None:
            _record_provenance_impl(
                self._provenance_path, tag, window, actual,
                eval_provenance=self.config.eval_provenance,
            )
        if outcome.macro_kl is not None and outcome.same_top is not None:
            if not self.contract.passes(outcome.macro_kl, outcome.same_top):
                self._max_known_fail = max(
                    filter(lambda v: v is not None, (self._max_known_fail, actual)),
                    default=None,
                )
            else:
                self._min_known_pass = min(
                    filter(lambda v: v is not None, (self._min_known_pass, actual)),
                    default=None,
                )

        if not _manifest_has(self.manifest_path, tag):
            with self.manifest_path.open("a", encoding="utf-8") as handle:
                digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
                handle.write(f"{tag}  {actual}  {digest}\n")
        artifact.unlink(missing_ok=True)
        return outcome


_SLICE_SUFFIX = {
    "wiki_test": "64k.txt",
    "wiki_valid": "valid-64k.txt",
    "chinese": "cn-64k.txt",
    "code": "code-64k.txt",
    "agent_chat": "agent-64k.txt",
}


def _manifest_has(manifest_path: Path, name: str) -> bool:
    if not manifest_path.is_file():
        return False
    return any(line.split()[:1] == [name] for line in manifest_path.read_text().splitlines())


def _record_provenance_impl(
    provenance_path: Path,
    tag: str,
    window: Window,
    size_bytes: int,
    eval_provenance: EvalProvenance | None = None,
) -> None:
    """Append one seed-provenance record (window lower/upper per artifact).

    When the verified eval closure is attached, the record also carries the
    contract digest and reference-manifest SHA — a seed whose sidecar names a
    different closure is inadmissible evidence (Codex audit 2026-09-04).
    """
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "name": tag,
        "size_bytes": size_bytes,
        "window_lower_preset": window.lower_preset,
        "window_upper_preset": window.upper_preset,
        "attestation": "runtime-verified" if eval_provenance is not None else "unattested",
    }
    if eval_provenance is not None:
        record["eval_contract_digest"] = eval_provenance.contract_digest
        record["reference_manifest_sha256"] = hashlib.sha256(
            Path(eval_provenance.reference_manifest_path).read_bytes()
        ).hexdigest()
    with provenance_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def _provenance_view(
    provenance_path: Path | None,
) -> tuple[set[str], set[str], set[str]]:
    """Read seed-provenance sidecar records once.

    Returns (tainted, stale, attested):

    * **tainted** — records whose producing window used a poison preset on
      either anchor (fail-closed seed admissibility, GPT P3a ruling);
    * **stale** — records that carry an ``eval_contract_digest`` which does
      not match the live frozen contract (Codex audit 2026-09-04: a seed
      attested under a different closure is inadmissible evidence);
    * **attested** — records that carry a matching closure digest.
    """
    if provenance_path is None or not Path(provenance_path).is_file():
        return set(), set(), set(), {}
    from fit_gguf.eval import contract_digest

    live = contract_digest()
    tainted: set[str] = set()
    stale: set[str] = set()
    attested: set[str] = set()
    manifest_by_name: dict[str, str] = {}
    for line in Path(provenance_path).read_text().splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        name = str(record.get("name", ""))
        if (
            str(record.get("window_lower_preset", "")).upper() in POISON_PRESETS
            or str(record.get("window_upper_preset", "")).upper() in POISON_PRESETS
        ):
            tainted.add(name)
        digest = record.get("eval_contract_digest")
        if digest is not None:
            if digest == live:
                attested.add(name)
            else:
                stale.add(name)
        manifest_sha = record.get("reference_manifest_sha256")
        if manifest_sha is not None:
            manifest_by_name[name] = str(manifest_sha)
    return tainted, stale, attested, manifest_by_name


def run_tier_search(
    contract: TierContract,
    config: RunnerConfig,
    windows: list[Window],
    seeds: tuple[Seed, ...],
    *,
    min_size: int,
    max_size: int,
    budget: int = 8,
    tolerance_bytes: int = 128 * 1024 * 1024,
    manifest_path: Path,
    logs_out_dir: Path,
):
    """Wire the pure search to the real executor and run it for one tier."""
    known_sizes = {seed.size_bytes for seed in seeds}
    fails = [s.size_bytes for s in seeds if not contract.passes(s.macro_kl, s.same_top)]
    passes = [s.size_bytes for s in seeds if contract.passes(s.macro_kl, s.same_top)]
    executor = SearchExecutor(
        config,
        windows,
        contract,
        manifest_path,
        logs_out_dir,
        known_sizes=known_sizes,
        max_known_fail=max(fails) if fails else None,
        min_known_pass=min(passes) if passes else None,
        provenance_path=Path(manifest_path).parent / "seed-provenance.jsonl",
    )
    audit = config.out_dir / f"fidelity-search-{contract.tier}.jsonl"
    if audit.exists():
        audit.unlink()
    result = fidelity_search(
        contract,
        executor.evaluate,
        seeds=seeds,
        min_size=min_size,
        max_size=max_size,
        budget=budget,
        tolerance_bytes=tolerance_bytes,
        audit_log=audit,
    )
    summary = result.summary()
    summary["artifact_recipes"] = {
        str(size): str(path) for size, path in sorted(executor._recipe_by_size.items())
    }
    summary["artifact_analyses"] = {
        str(size): str(path) for size, path in sorted(executor._analysis_by_size.items())
    }
    summary_path = config.out_dir / f"fidelity-search-{contract.tier}-summary.json"
    summary_path.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    shutil.rmtree(config.work_dir, ignore_errors=True)
    return summary
