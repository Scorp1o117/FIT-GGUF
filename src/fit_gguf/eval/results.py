"""eval-v1 result schema and llama.cpp log normalization.

Every evaluation result — Fidelity search candidate, quality-size curve
point, sealed report — is emitted in one schema (v0.2.1 §F) so downstream
consumers never re-parse tool logs. Serialization is deterministic:
byte-identical inputs produce byte-identical JSON.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass

from fit_gguf.eval.contract import CONTRACT_ID, DOMAINS


RESULT_SCHEMA = "fit.eval_result.v1"


class EvalLogError(ValueError):
    """Raised when a llama-perplexity KL log cannot be parsed."""

_MEAN_KLD = re.compile(r"Mean\s+KLD:\s+([0-9.]+)\s+±\s+([0-9.]+)")
_SAME_TOP = re.compile(r"Same top p:\s+([0-9.]+)\s+±\s+([0-9.]+)\s+%")
_RMS_DELTA = re.compile(r"RMS Δp\s+:\s+([0-9.]+)\s+±\s+([0-9.]+)\s+%")
_KL_HEADER = "====== KL divergence statistics ======"


def parse_llama_kl_log(text: str) -> dict:
    """Extract the contract metrics from one llama-perplexity KL log."""
    if _KL_HEADER not in text:
        raise EvalLogError("not a llama-perplexity KL log (header missing)")
    mean = _MEAN_KLD.search(text)
    top = _SAME_TOP.search(text)
    rms = _RMS_DELTA.search(text)
    if mean is None or top is None:
        raise EvalLogError("KL log missing 'Mean KLD' or 'Same top p' line")
    return {
        "mean_kld": float(mean.group(1)),
        "mean_kld_stderr": float(mean.group(2)),
        "same_top_pct": float(top.group(1)),
        "same_top_stderr_pct": float(top.group(2)),
        "rms_delta_p_pct": float(rms.group(1)) if rms else None,
    }


@dataclass(frozen=True, slots=True)
class DomainResult:
    domain: str
    tokens: int
    kl: float
    same_top: float
    ppl: float | None = None
    rms: float | None = None
    excluded_positions: int = 0


def _round(value: float | None, digits: int = 9) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return round(value, digits)


def build_eval_result(
    contract_digest: str,
    model_hash: str,
    reference_logits_hash: str,
    domain_results: list[DomainResult],
    candidate_plan_hash: str | None = None,
    macro_kl: float | None = None,
    macro_same_top: float | None = None,
    micro_kl: float | None = None,
    comparison: dict | None = None,
    reference_manifest_hash: str | None = None,
) -> dict:
    """Build the unified eval-v1 output document (deterministic).

    ``model_hash`` identifies the evaluated candidate GGUF. Comparison
    semantics (regressed domains) live in the optional ``comparison`` block
    — computed only against an explicit baseline result — and are never
    fabricated by the base evaluator: a standalone candidate-vs-reference
    evaluation has no meaningful "regression" (all KL >= 0).
    """
    by_domain = {r.domain: r for r in domain_results}
    if set(by_domain) != set(DOMAINS):
        raise ValueError(f"result must cover exactly {DOMAINS}, got {sorted(by_domain)}")
    weights = {d: 0.2 for d in DOMAINS}
    if macro_kl is None:
        macro_kl = math.fsum(weights[r.domain] * r.kl for r in domain_results)
    if macro_same_top is None:
        macro_same_top = math.fsum(weights[r.domain] * r.same_top for r in domain_results)
    if micro_kl is None:
        total = math.fsum(r.tokens for r in domain_results)
        micro_kl = math.fsum(r.tokens * r.kl for r in domain_results) / total if total else None
    worst = max((r for r in domain_results), key=lambda r: r.kl)
    return {
        "result_schema": RESULT_SCHEMA,
        "evaluator_contract": CONTRACT_ID,
        "evaluator_contract_hash": contract_digest,
        "model_hash": model_hash,
        "reference_logits_hash": reference_logits_hash,
        "reference_manifest_hash": reference_manifest_hash,
        "candidate_plan_hash": candidate_plan_hash,
        "total_tokens": sum(r.tokens for r in domain_results),
        "macro_kl": _round(macro_kl),
        "macro_same_top": _round(macro_same_top),
        "micro_kl": _round(micro_kl),
        "worst_domain_kl": _round(worst.kl),
        "worst_domain": worst.domain,
        "comparison": comparison,
        "domains": {
            r.domain: {
                "tokens": r.tokens,
                "kl": _round(r.kl),
                "same_top": _round(r.same_top),
                "ppl": _round(r.ppl),
                "rms": _round(r.rms),
                "excluded_positions": r.excluded_positions,
            }
            for r in sorted(domain_results, key=lambda r: r.domain)
        },
    }


def compare_results(baseline: dict, candidate: dict, baseline_result_hash: str) -> dict:
    """Comparison layer: regression verdict against an explicit baseline.

    The base evaluator never emits this; it exists only when a caller
    supplies a baseline result document to compare against.
    """
    per_domain = {}
    regressed = []
    for domain, after in candidate["domains"].items():
        before = baseline["domains"].get(domain)
        if before is None:
            continue
        worse = after["kl"] > before["kl"]
        per_domain[domain] = {"baseline_kl": before["kl"], "kl": after["kl"], "regressed": worse}
        if worse:
            regressed.append(domain)
    return {
        "baseline_result_hash": baseline_result_hash,
        "regressed_domains": regressed,
        "per_domain": per_domain,
    }


def build_reference_manifest(
    source_bf16_gguf_sha256: str,
    tokenizer_hash: str,
    evaluator_contract_hash: str,
    runtime_provenance: dict,
    domains: dict[str, dict],
) -> dict:
    """Per (source model) reference artifact manifest (v0.2.1 §D closure).

    ``domains`` maps each contract domain to its corpus sha256, reference
    .kld sha256, raw byte/codepoint counts, and expected valid token count.
    """
    for domain, entry in domains.items():
        missing = {"corpus_sha256", "reference_kld_sha256", "raw_bytes", "unicode_codepoints", "expected_valid_tokens"} - set(entry)
        if missing:
            raise ValueError(f"reference manifest domain {domain!r} missing fields {sorted(missing)}")
    return {
        "manifest_schema": "fit.eval_reference_manifest.v1",
        "source_bf16_gguf_sha256": source_bf16_gguf_sha256,
        "tokenizer_hash": tokenizer_hash,
        "evaluator_contract_hash": evaluator_contract_hash,
        "runtime_provenance": runtime_provenance,
        "domains": dict(sorted(domains.items())),
    }


def serialize_result(result: dict) -> str:
    """Deterministic JSON: identical inputs give byte-identical text."""
    return json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
