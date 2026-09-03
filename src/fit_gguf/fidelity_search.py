"""Fidelity Search v1 (v0.2 flagship solver, GPT ruling 2026-09-03).

Finds the minimum artifact size that satisfies a tier contract via
coarse -> bracket -> fine -> verify, where the tier contract is the dual
hard gate  [macro KL <= K_t] AND [same-top >= G(p,t)]  (K_t from the frozen
Global KL Core, G(p,t) from the model's validated Guard Profile).

Design notes
------------
* The search state machine is pure: the caller injects an ``evaluate``
  callable mapping a target byte size to an :class:`EvalOutcome`. Prior
  (already evaluated) points enter as budget-free seeds.
* Prior points are "observed" evidence (Smallest Observed Passing); the
  search's job is to upgrade the answer to a *minimum verified PASS* inside
  a bracket no wider than ``tolerance_bytes``.
* Budget accounting counts every fresh evaluation submission, per the
  release gate R3 (Normal <= 8, Precise <= 16).
* Real curves are monotone up to eval noise. Bracket updates are therefore
  order-free: lo_fail = max(FAIL sizes), hi_pass = min(PASS sizes). If the
  bracket inverts (a PASS at/below a FAIL), the region is within noise and
  the search stops with the smallest verified PASS and an explicit
  ``noise_inversion`` flag instead of pretending to a precision it cannot
  have.
* The active constraint at the answer is reported via normalized margins
  M_kl = (K_t - KL) / K_t and M_top = (Top - G) / (1 - G): the smaller one
  binds and tells Fidelity Search where the next byte buys the least.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

_MI = 1024 * 1024
DEFAULT_TOLERANCE_BYTES = 128 * _MI  # release gate R2: <= 64-128 MiB


class FidelitySearchError(ValueError):
    """Raised when a search cannot be configured or executed."""


@dataclass(frozen=True, slots=True)
class TierContract:
    """Dual hard gate for one fidelity tier."""

    tier: str
    kl_anchor: float
    same_top_floor: float

    def passes(self, macro_kl: float, same_top: float) -> bool:
        return macro_kl <= self.kl_anchor and same_top >= self.same_top_floor

    def margins(self, macro_kl: float, same_top: float) -> tuple[float, float]:
        """Normalized (M_kl, M_top); the smaller value is the active constraint."""
        m_kl = (self.kl_anchor - macro_kl) / self.kl_anchor
        m_top = (same_top - self.same_top_floor) / (1.0 - self.same_top_floor)
        return m_kl, m_top

    def active_constraint(self, macro_kl: float, same_top: float) -> str:
        m_kl, m_top = self.margins(macro_kl, same_top)
        return "kl" if m_kl <= m_top else "same_top"


@dataclass(frozen=True, slots=True)
class SearchPoint:
    size_bytes: int
    macro_kl: float
    same_top: float
    passed: bool
    eval_index: int  # 0 = prior observed point (budget-free)
    source: str  # observed | coarse | bracket | fine | verify

    def as_dict(self) -> dict:
        return {
            "size_bytes": self.size_bytes,
            "macro_kl": self.macro_kl,
            "same_top": self.same_top,
            "passed": self.passed,
            "eval_index": self.eval_index,
            "source": self.source,
        }


@dataclass(frozen=True, slots=True)
class EvalOutcome:
    """Result of one fresh evaluation submission (executor's responsibility)."""

    size_bytes: int
    macro_kl: float | None
    same_top: float | None
    note: str | None = None  # e.g. failure reason; None outcome fields mean failed eval


@dataclass(frozen=True, slots=True)
class SearchResult:
    status: str  # verified_pass | budget_exhausted | no_pass | noise_inversion
    best: SearchPoint | None
    points: tuple[SearchPoint, ...]
    fresh_evals: int
    budget: int
    bracket_fail_bytes: int | None  # largest observed FAIL (None if none seen)
    bracket_pass_bytes: int | None  # smallest observed PASS
    active_constraint: str | None
    tolerance_bytes: int
    note: str | None = None

    def summary(self) -> dict:
        best = self.best.as_dict() if self.best else None
        if self.status == "verified_pass":
            guarantee = "minimum verified PASS within tolerance bracket"
        elif self.status == "noise_inversion":
            guarantee = "smallest verified PASS; bracket inverted (curve within eval noise)"
        elif self.status == "no_pass":
            guarantee = "no evaluated point satisfied the contract"
        else:
            guarantee = "budget spent before tolerance bracket was reached (observed PASS only)"
        return {
            "status": self.status,
            "guarantee": guarantee,
            "best": best,
            "fresh_evals": self.fresh_evals,
            "budget": self.budget,
            "bracket_fail_bytes": self.bracket_fail_bytes,
            "bracket_pass_bytes": self.bracket_pass_bytes,
            "active_constraint": self.active_constraint,
            "tolerance_bytes": self.tolerance_bytes,
            "note": self.note,
        }


@dataclass(frozen=True, slots=True)
class Seed:
    """A prior (already evaluated) point; budget-free bracket evidence."""

    size_bytes: int
    macro_kl: float
    same_top: float


def _outcome_to_point(outcome: EvalOutcome, eval_index: int, source: str, contract: TierContract) -> SearchPoint | None:
    if outcome.macro_kl is None or outcome.same_top is None:
        return None
    return SearchPoint(
        size_bytes=outcome.size_bytes,
        macro_kl=outcome.macro_kl,
        same_top=outcome.same_top,
        passed=contract.passes(outcome.macro_kl, outcome.same_top),
        eval_index=eval_index,
        source=source,
    )


def fidelity_search(
    contract: TierContract,
    evaluate,
    seeds: tuple[Seed, ...] = (),
    *,
    min_size: int,
    max_size: int,
    budget: int = 8,
    tolerance_bytes: int = DEFAULT_TOLERANCE_BYTES,
    coarse_stride_bytes: int | None = None,
    audit_log: str | Path | None = None,
    clock=time.monotonic,
) -> SearchResult:
    """Run the coarse -> bracket -> fine -> verify state machine.

    ``evaluate(size_bytes) -> EvalOutcome`` must return the artifact's macro
    KL and same-top (or ``None`` fields on failure). The executor owns
    retries; every call counts against ``budget``.
    """
    if min_size >= max_size:
        raise FidelitySearchError(f"min_size {min_size:,} >= max_size {max_size:,}")
    if budget < 1:
        raise FidelitySearchError("budget must be >= 1")
    if tolerance_bytes < 1:
        raise FidelitySearchError("tolerance_bytes must be >= 1")

    audit: list[dict] = []
    started = clock()

    def log_event(event: dict) -> None:
        event = {"t": round(clock() - started, 3), **event}
        audit.append(event)
        if audit_log is not None:
            with Path(audit_log).open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event) + "\n")

    points: list[SearchPoint] = []
    for seed in seeds:
        point = _outcome_to_point(
            EvalOutcome(size_bytes=seed.size_bytes, macro_kl=seed.macro_kl, same_top=seed.same_top),
            0,
            "observed",
            contract,
        )
        if point is not None:
            points.append(point)
            log_event({"event": "seed", **point.as_dict()})

    fresh = 0

    def submit(size_bytes: int, source: str) -> SearchPoint | None:
        nonlocal fresh
        if fresh >= budget:
            return None
        fresh += 1
        log_event({"event": "eval_submit", "size_bytes": size_bytes, "source": source, "eval_index": fresh})
        point = _outcome_to_point(evaluate(size_bytes), fresh, source, contract)
        if point is None:
            log_event({"event": "eval_failed", "size_bytes": size_bytes, "source": source})
        else:
            points.append(point)
            log_event({"event": "eval_result", **point.as_dict()})
        return point

    def bracket() -> tuple[int | None, int | None]:
        fails = [p.size_bytes for p in points if not p.passed]
        passes = [p.size_bytes for p in points if p.passed]
        return (max(fails) if fails else None), (min(passes) if passes else None)

    def smallest_pass() -> SearchPoint | None:
        passes = [p for p in points if p.passed]
        return min(passes, key=lambda p: p.size_bytes) if passes else None

    # ---- phase 0: seeds may already bracket the crossing ----
    lo_fail, hi_pass = bracket()
    if hi_pass is not None and lo_fail is not None and lo_fail >= hi_pass:
        # Inverted before any fresh eval: observed evidence is noise-limited.
        return _finish("noise_inversion", contract, points, fresh, budget,
                       lo_fail, hi_pass, tolerance_bytes)

    # ---- phase 1: coarse — establish a bracket if seeds lack one ----
    stride = coarse_stride_bytes or max(_MI, (max_size - min_size) // 16)
    walked_to_floor = False
    if hi_pass is None:
        # No known PASS: step up from min_size (or from lo_fail) until first PASS.
        cursor = max(min_size, (lo_fail + stride) if lo_fail is not None else min_size)
        while fresh < budget:
            if lo_fail is not None and cursor <= lo_fail:
                cursor = lo_fail + stride
            if cursor > max_size:
                break
            point = submit(cursor, "coarse")
            if point is not None and point.passed:
                break
            cursor += stride  # failed evals consume budget but do not end the walk
        lo_fail, hi_pass = bracket()
        if hi_pass is None:
            return _finish("no_pass", contract, points, fresh, budget, lo_fail, hi_pass, tolerance_bytes)
        if lo_fail is not None and lo_fail >= hi_pass:
            return _finish("noise_inversion", contract, points, fresh, budget,
                           lo_fail, hi_pass, tolerance_bytes)
    if lo_fail is None:
        # Known PASS but no FAIL below: step down from hi_pass until first FAIL.
        cursor = max(min_size, hi_pass - stride)  # always probe at least the floor
        while fresh < budget and cursor >= min_size:
            if hi_pass is not None and cursor >= hi_pass:
                cursor = hi_pass - stride
            cursor = max(min_size, cursor)
            point = submit(cursor, "coarse")
            if point is not None and not point.passed:
                break
            cursor -= stride  # failed evals consume budget but do not end the walk
            if cursor < min_size:
                walked_to_floor = True
                break
        lo_fail, hi_pass = bracket()

    # ---- phase 2: bracket/fine — bisect until tolerance ----
    failed_probes: set[int] = set()
    while fresh < budget:
        lo_fail, hi_pass = bracket()
        if hi_pass is None or lo_fail is None:
            break
        if hi_pass - lo_fail <= tolerance_bytes:
            break
        mid = (lo_fail + hi_pass) // 2
        shift = max(1, tolerance_bytes // 8)
        while mid in failed_probes and mid + shift < hi_pass:
            mid += shift
        if mid in failed_probes:
            break  # bracket is full of unmeasurable probes; stop honestly
        point = submit(mid, "bracket")
        if point is None:
            failed_probes.add(mid)
            continue
        lo_fail, hi_pass = bracket()
        if lo_fail is not None and hi_pass is not None and lo_fail >= hi_pass:
            return _finish("noise_inversion", contract, points, fresh, budget,
                           lo_fail, hi_pass, tolerance_bytes)

    # ---- phase 3: verify — classify what we actually established ----
    lo_fail, hi_pass = bracket()
    if hi_pass is None:
        return _finish("no_pass", contract, points, fresh, budget, lo_fail, hi_pass, tolerance_bytes)
    if lo_fail is None:
        if walked_to_floor:
            return _finish(
                "verified_pass", contract, points, fresh, budget, lo_fail, hi_pass,
                tolerance_bytes,
                note="no FAIL found within [min_size, hi_pass]; true crossing may "
                     "lie below the searchable floor",
            )
        return _finish("budget_exhausted", contract, points, fresh, budget,
                       lo_fail, hi_pass, tolerance_bytes)
    if hi_pass - lo_fail <= tolerance_bytes:
        return _finish("verified_pass", contract, points, fresh, budget,
                       lo_fail, hi_pass, tolerance_bytes)
    return _finish("budget_exhausted", contract, points, fresh, budget,
                   lo_fail, hi_pass, tolerance_bytes)


def _finish(
    status: str,
    contract: TierContract,
    points: list[SearchPoint],
    fresh: int,
    budget: int,
    lo_fail: int | None,
    hi_pass: int | None,
    tolerance_bytes: int,
    note: str | None = None,
) -> SearchResult:
    passes = [p for p in points if p.passed]
    best = min(passes, key=lambda p: p.size_bytes) if passes else None
    active = None
    if best is not None:
        active = contract.active_constraint(best.macro_kl, best.same_top)
    return SearchResult(
        status=status,
        best=best,
        points=tuple(sorted(points, key=lambda p: p.size_bytes)),
        fresh_evals=fresh,
        budget=budget,
        bracket_fail_bytes=lo_fail,
        bracket_pass_bytes=hi_pass,
        active_constraint=active,
        tolerance_bytes=tolerance_bytes,
        note=note,
    )
