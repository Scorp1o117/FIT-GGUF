"""Pure metric kernels for eval-v1.

These kernels define the arithmetic of the contract independently of any
llama.cpp binary: the synthetic fixture in ``fit_gguf.eval.synthetic`` pins
them to hand-derived closed forms so a code change cannot silently bend the
measuring stick. Semantics follow llama.cpp ``--kl-divergence``:

* KL direction is reference-first: D_KL(P_ref || P_quant).
* argmax ties resolve to the lowest vocabulary index.
* Accumulation is float64.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

EXCLUSION_BUDGET = 0.001  # contract: >0.1% excluded positions fails a domain
REF_LOGPROB_CUTOFF = -16.0  # b10666: reference tokens at/below -16 nats are
# excluded from the KL sum ("if (p_log_base > -16.f)") -- part of the
# metric definition, not an optimization.


class MetricError(ValueError):
    """Raised when metric inputs violate the contract."""


def _check_same_length(ref: list[float], cand: list[float]) -> None:
    if len(ref) != len(cand):
        raise MetricError(f"vocab mismatch: ref={len(ref)} cand={len(cand)}")


def log_softmax(logits: list[float]) -> list[float]:
    """Stable log-softmax in float64."""
    top = max(logits)
    exps = [math.exp(x - top) for x in logits]
    total = math.fsum(exps)
    log_total = top + math.log(total)
    return [x - log_total for x in logits]


def kl_divergence(ref_logits: list[float], cand_logits: list[float]) -> float:
    """D_KL(P_ref || P_quant) at one position, in nats (b10666 semantics).

    Reference tokens whose log-prob is at or below ``REF_LOGPROB_CUTOFF``
    (-16 nats) contribute nothing, exactly as in the pinned tool. Returns
    math.inf when the candidate assigns zero mass where the reference is
    positive above the cutoff; callers treat non-finite values as excluded
    positions.
    """
    _check_same_length(ref_logits, cand_logits)
    lp = log_softmax(ref_logits)
    lq = log_softmax(cand_logits)
    terms = []
    for p_log, q_log in zip(lp, lq):
        if p_log <= REF_LOGPROB_CUTOFF:
            continue  # b10666 cutoff: tiny reference mass is outside the sum
        if q_log == -math.inf:
            return math.inf
        terms.append(math.exp(p_log) * (p_log - q_log))
    return math.fsum(terms)


def argmax_lowest_tie(values: list[float]) -> int:
    """Argmax with lowest-index tie resolution (contract rule)."""
    best = 0
    for i in range(1, len(values)):
        if values[i] > values[best]:
            best = i
    return best


def same_top(ref_logits: list[float], cand_logits: list[float]) -> bool:
    _check_same_length(ref_logits, cand_logits)
    return argmax_lowest_tie(ref_logits) == argmax_lowest_tie(cand_logits)


def token_nll(logits: list[float], target: int) -> float:
    """Negative log-probability of ``target`` under the logits."""
    if not 0 <= target < len(logits):
        raise MetricError(f"target {target} out of vocab range {len(logits)}")
    return -log_softmax(logits)[target]


def position_valid(*logit_vectors: list[float]) -> bool:
    """A position counts only if every vector is entirely finite."""
    for vector in logit_vectors:
        if len(vector) == 0 or not all(math.isfinite(x) for x in vector):
            return False
    return True


@dataclass(frozen=True, slots=True)
class DomainAggregate:
    domain: str
    n_positions: int
    excluded_positions: int
    kl_mean: float
    same_top_frac: float
    ppl: float | None

    @property
    def exclusion_rate(self) -> float:
        total = self.n_positions + self.excluded_positions
        return self.excluded_positions / total if total else 0.0

    @property
    def within_exclusion_budget(self) -> bool:
        return self.exclusion_rate <= EXCLUSION_BUDGET


def aggregate_domain(
    domain: str,
    kl_values: list[float],
    same_top_values: list[bool],
    nll_values: list[float] | None = None,
) -> DomainAggregate:
    """Two-level aggregation, level one: mean over counted positions.

    Non-finite KL values are excluded (contract NaN rule). All inputs must
    have the same length as the position axis.
    """
    n = len(kl_values)
    if not (len(same_top_values) == n and (nll_values is None or len(nll_values) == n)):
        raise MetricError("position axis mismatch in aggregate_domain")
    counted = [v for v in kl_values if math.isfinite(v)]
    excluded = n - len(counted)
    pairs = [(v, s) for v, s in zip(kl_values, same_top_values, strict=True) if math.isfinite(v)]
    kl_mean = math.fsum(v for v, _ in pairs) / len(pairs) if pairs else math.inf
    same_top_frac = sum(1 for _, s in pairs if s) / len(pairs) if pairs else 0.0
    ppl = None
    if nll_values is not None:
        counted_nll = [v for v, k in zip(nll_values, kl_values, strict=True) if math.isfinite(k)]
        if counted_nll:
            ppl = math.exp(math.fsum(counted_nll) / len(counted_nll))
    return DomainAggregate(
        domain=domain,
        n_positions=len(pairs),
        excluded_positions=excluded,
        kl_mean=kl_mean,
        same_top_frac=same_top_frac,
        ppl=ppl,
    )


def macro_kl(domain_results: dict[str, DomainAggregate], weights: dict[str, float]) -> float:
    """Two-level aggregation, level two: weighted mean of domain means."""
    if set(domain_results) != set(weights):
        raise MetricError("macro aggregation requires every weighted domain")
    return math.fsum(weights[d] * domain_results[d].kl_mean for d in weights)


def macro_same_top(domain_results: dict[str, DomainAggregate], weights: dict[str, float]) -> float:
    if set(domain_results) != set(weights):
        raise MetricError("macro aggregation requires every weighted domain")
    return math.fsum(weights[d] * domain_results[d].same_top_frac for d in weights)


def micro_kl(domain_results: dict[str, DomainAggregate]) -> float:
    """Token-pooled micro average (diagnostics only, never contract primary)."""
    total_positions = math.fsum(r.n_positions for r in domain_results.values())
    if not total_positions:
        return math.inf
    return math.fsum(r.n_positions * r.kl_mean for r in domain_results.values()) / total_positions


def regressed_domains(before: dict[str, DomainAggregate], after: dict[str, DomainAggregate]) -> list[str]:
    """Domains whose mean KL got worse from ``before`` to ``after``."""
    common = sorted(set(before) & set(after))
    return [d for d in common if after[d].kl_mean > before[d].kl_mean]
