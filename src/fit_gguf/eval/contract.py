"""eval-v1 evaluator contract (v0.2.1 §9).

The contract freezes everything that can move a measurement:

* the pinned llama.cpp runtime and the exact llama-perplexity invocation;
* the five fixed 64 KiB evaluation slices (M9 preregistered set) with
  SHA-256, domain weights, and provenance pointer;
* metric definitions — KL direction D_KL(P_ref || P_quant), same-top tie
  policy, PPL — and the two-level aggregation rule (per-domain mean, then
  equal-weight macro; token-pooled micro is reported but never primary);
* same-top edge rules (ties, EOS, NaN handling, vocab mismatch, exclusion
  budget);
* numeric policy (float64 accumulation, tolerances).

Fidelity Contract numbers are only meaningful *under* this contract: every
eval result records ``evaluator_contract`` + ``evaluator_contract_hash``.

The canonical serialization is JSON (sorted keys, fixed separators) rather
than YAML so the evaluator stays dependency-free; content is what matters,
and the hash is taken over the canonical form.
"""

from __future__ import annotations

import hashlib
import json

CONTRACT_ID = "eval-v1"
CONTRACT_SCHEMA = "fit.evaluator_contract.v1"
EVALUATOR_VERSION = "0.1.0"

DOMAINS = ("wiki_test", "wiki_valid", "chinese", "code", "agent_chat")
DOMAIN_WEIGHT = 0.2

EVAL_V1: dict = {
    "schema": CONTRACT_SCHEMA,
    "contract_id": CONTRACT_ID,
    "result_schema": "fit.eval_result.v1",
    "canonicalization": {
        "encoding": "utf-8",
        "key_order": "lexicographic (recursive)",
        "separators": [",", ":"],
        "ensure_ascii": False,
        "trailing_newline": False,
        "implementation": "json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(',', ':'))",
    },
    "protocol": {
        "tool": "llama-perplexity",
        "semantics_pin": {
            "llama_cpp_source_revision": "b10666",
            "algorithm": "tools/perplexity/perplexity.cpp log_softmax() --kl-divergence path, transcribed into fit_gguf.eval.metrics",
            "eval_flags": [
                "-ngl", "99", "-t", "16", "-c", "512", "-b", "512",
                "--kl-divergence", "--kl-divergence-base", "<reference .kld>",
            ],
        },
        "execution_provenance_required": [
            "llama-perplexity binary sha256",
            "build/compiler info",
            "rocm_version",
            "gpu_arch",
        ],
        "execution_provenance_note": "hardware/build facts live in per-run execution manifests, NOT in this semantic contract; a platform change within preregistered tolerance must not force eval-v2",
        "reference": {
            "definition": "BF16 (source-model) distribution on aligned input, same tokenizer, same text",
            "artifact": "<domain>.kld file: f16-quantized reference LOG-PROBS with per-position affine scale (p_log_ref(v) = scale*f16(v) + min); NOT raw logits",
            "storage_note": "the affine f16 compression of reference log-probs is part of the measurement; regenerate references only with the pinned revision and record their sha256 per domain",
            "alignment": "teacher-forced next-token positions of the slice under the shared tokenizer; positional alignment is fixed by the .kld file and must never be re-derived per candidate",
            "generation": {
                "phase_1_reference_creation": "llama-perplexity -m <bf16.gguf> -f <slice> <sampling flags> --kl-divergence-base <out.kld>   (WITHOUT --kl-divergence: the base file is WRITTEN by the perplexity path)",
                "phase_2_candidate_eval": "llama-perplexity -m <candidate.gguf> -f <slice> <sampling flags> --kl-divergence --kl-divergence-base <reference.kld>   (--kl-divergence switches to the read-only KL path)",
                "b10666_note": "with --kl-divergence set and a missing base file, the tool fails instead of writing one; reference creation MUST use phase 1",
                "verified": "2026-09-02: phase-1 output byte-identical to P4 legacy references on all five domains",
            },
        },
    },
    "corpus": {
        "slicing": "fixed 65,536-BYTE UTF-8 file slices (historical label: M9 64k); corpus files are authoritative by sha256; no reslicing is permitted",
        "slicing_note": "byte slicing means unicode codepoint counts differ per domain; bytes and codepoints are recorded per domain in the reference manifest, not asserted here",
        "provenance": "eval-data/PROVENANCE.md",
        "domains": {
            "wiki_test": {"file": "eval-data/kl-eval-64k.txt", "sha256": "d400318e1ad6981e3fa332514ae5a59e98d888592428d152a0ffc3ceb135620e"},
            "wiki_valid": {"file": "eval-data/kl-eval-valid-64k.txt", "sha256": "9b455800b98525e0f6ec3ac18f4a6b789622f7d9ae381e5a174f5c3d42173402"},
            "chinese": {"file": "eval-data/kl-eval-cn-64k.txt", "sha256": "a7584dc67f2e3050d42326da91f2801bf29c381596eb1364896490c98bad56d5"},
            "code": {"file": "eval-data/kl-eval-code-64k.txt", "sha256": "da9cae0047be52338c7710d7b6cc00354f05d2c8009b9bb3d7914f08d65a4084"},
            "agent_chat": {"file": "eval-data/kl-eval-agent-64k.txt", "sha256": "01c79b525330a642d89d0b8a00f0b42931ce0b94546c894047d8da041c86188b"},
        },
    },
    "sampling": {
        "context_length": 512,
        "batch_size": 512,
        "windowing": "llama.cpp perplexity default: consecutive non-overlapping context windows",
        "shuffle": False,
    },
    "metrics": {
        "kl_definition": "per teacher-forced position: D_KL(P_ref || P_quant) = sum over v with p_log_ref(v) > -16 nats of p_ref(v) * (p_log_ref(v) - p_log_quant(v)), full-vocab softmax at temperature 1 (b10666 cutoff: reference tokens at or below -16 nats log-prob are excluded from the sum)",
        "kl_direction": "reference-first: mass the reference assigns that the candidate loses counts positively (verified against b10666 log_softmax source)",
        "same_top_definition": "argmax p_ref (scan with strict greater-than, so the first maximum - lowest vocabulary index - wins) equals argmax p_quant under the same rule, over counted positions",
        "ppl_definition": "exp(mean candidate NLL over the ground-truth next-token targets of the frozen corpus)",
        "rms_delta_definition": "per position t with teacher-forced target token tau(t): p_diff(t) = p_quant(t)(tau(t)) - p_ref(t)(tau(t)); RMS_delta_p = 100 * sqrt(mean_t p_diff(t)^2), reported in percent (b10666 p_diff formula; diagnostics only)",
        "aggregation": "two-level: per-domain arithmetic mean over counted positions, then macro = equal-weight mean over the five domains",
        "macro_is_primary": True,
        "micro_note": "token-pooled micro is reported for diagnostics only; it is dominated by large domains and must not back a Fidelity Contract decision",
        "domain_weights": {domain: DOMAIN_WEIGHT for domain in DOMAINS},
    },
    "same_top_edge_rules": {
        "tie": "first maximum wins (strict greater-than scan) = lowest vocabulary index, applied to both reference log-probs and candidate logits",
        "eos_positions": "counted; EOS is never masked",
        "padding": "none in raw-text mode",
        "vocab_mismatch": "fatal: refuse to evaluate, never index-remap silently",
        "nan_inf_positions": "excluded from the domain mean and reported in excluded_positions; a domain with more than 0.1% excluded positions FAILS",
    },
    "numeric": {
        "accumulation": "float64 (math.fsum for term sums)",
        "softmax": "stable log-softmax (max-subtracted)",
        "tolerance": {"kl_delta": 1e-8, "same_top": "exact match required"},
    },
    "reference_manifest": {
        "definition": "per (source model x domain) artifact manifest, generated with the reference .kld; the four-hash loop in results closes through it",
        "pins": [
            "source_bf16_gguf_sha256",
            "tokenizer_hash",
            "corpus_domain_sha256",
            "reference_kld_sha256",
            "evaluator_contract_hash",
            "runtime_provenance (semantics_pin + execution facts)",
        ],
        "expected_valid_tokens": "recorded per domain after reference generation; candidate runs whose observed valid token count differs FAIL",
    },
    "implementation": {
        "evaluator": "fit_gguf.eval",
        "evaluator_version": EVALUATOR_VERSION,
        "log_source": "llama-perplexity KL log (parsed by fit_gguf.eval.results.parse_llama_kl_log)",
    },
}


class ContractError(ValueError):
    """Raised when an evaluator contract is malformed."""


def canonical_json(obj: dict) -> str:
    """Deterministic serialization: sorted keys, no incidental whitespace."""
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def contract_hash(contract: dict) -> str:
    return hashlib.sha256(canonical_json(contract).encode("utf-8")).hexdigest()


def validate_contract(contract: dict) -> None:
    if contract.get("schema") != CONTRACT_SCHEMA:
        raise ContractError(f"schema {contract.get('schema')!r} != {CONTRACT_SCHEMA!r}")
    if contract.get("contract_id") != CONTRACT_ID:
        raise ContractError(f"contract_id {contract.get('contract_id')!r} != {CONTRACT_ID!r}")
    weights = contract.get("metrics", {}).get("domain_weights", {})
    if set(weights) != set(DOMAINS):
        raise ContractError(f"domain weights must cover exactly {DOMAINS}")
    total = sum(weights.values())
    if abs(total - 1.0) > 1e-9:
        raise ContractError(f"domain weights must sum to 1.0, got {total}")
    for domain, entry in contract.get("corpus", {}).get("domains", {}).items():
        if len(entry.get("sha256", "")) != 64:
            raise ContractError(f"corpus domain {domain!r}: missing/short sha256")


def rendered_contract() -> str:
    """The canonical frozen contract text (this exact text is hashed)."""
    return canonical_json(EVAL_V1)


def contract_digest() -> str:
    return contract_hash(EVAL_V1)
