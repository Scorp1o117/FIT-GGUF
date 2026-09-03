"""Tests for eval-v1: contract, metric kernels, synthetic fixture, results."""

import math

import pytest

from fit_gguf.eval import (
    ContractError,
    DomainAggregate,
    DomainResult,
    EVAL_V1,
    EvalLogError,
    MetricError,
    REF_LOGPROB_CUTOFF,
    aggregate_domain,
    build_eval_result,
    build_reference_manifest,
    canonical_json,
    compare_results,
    contract_digest,
    contract_hash,
    kl_divergence,
    macro_kl,
    micro_kl,
    parse_llama_kl_log,
    position_valid,
    regressed_domains,
    same_top,
    serialize_result,
    token_nll,
    validate_contract,
)
from fit_gguf.eval.synthetic import (
    CAND_A,
    KL_A,
    KL_A_REVERSED,
    KL_B,
    NLL_REF_A_TARGET_0,
    PROB_REF_A_0,
    REF_A,
    REF_B,
    VOCAB,
)

REAL_LOG_TAIL = """====== KL divergence statistics ======
Mean    KLD:   0.052627 ±   0.001503
Maximum KLD:   4.251894
====== Token probability statistics ======
RMS Δp    :  7.064 ± 0.232 %
Same top p: 90.968 ± 0.322 %
"""


def test_kl_matches_hand_derived_closed_form():
    assert kl_divergence(REF_A, CAND_A) == pytest.approx(KL_A, rel=1e-12)
    assert kl_divergence(REF_B, list(REF_B)) == pytest.approx(KL_B, abs=1e-15)


def test_kl_direction_is_reference_first():
    # kernel(ref, cand) must equal the closed D_KL(p||q), not D_KL(q||p)
    assert kl_divergence(REF_A, CAND_A) == pytest.approx(KL_A, rel=1e-12)
    assert kl_divergence(CAND_A, REF_A) == pytest.approx(KL_A_REVERSED, rel=1e-12)
    assert abs(KL_A - KL_A_REVERSED) > 1e-6  # genuinely asymmetric


def test_same_top_and_tie_rule():
    assert same_top(REF_A, CAND_A) is True
    # tie on equal maxima: lowest index wins on both sides -> still equal
    tie_a = [1.0, 2.0, 2.0, 0.0]
    tie_b = [1.0, 2.0, 0.5, 0.0]
    assert same_top(tie_a, tie_b) is True
    assert same_top(REF_A, [0.2, 0.4, 0.3, 0.5]) is False  # argmax moved to index 3


def test_token_nll_closed_form():
    assert token_nll(REF_A, 0) == pytest.approx(NLL_REF_A_TARGET_0, rel=1e-12)
    assert math.exp(-token_nll(REF_A, 0)) == pytest.approx(PROB_REF_A_0, rel=1e-12)


def test_vocab_mismatch_is_fatal():
    with pytest.raises(MetricError, match="vocab mismatch"):
        kl_divergence(REF_A, CAND_A[:VOCAB - 1])


def test_reference_logprob_cutoff_is_part_of_the_definition():
    # b10666: reference mass at/below -16 nats is outside the KL sum.
    # ref index 1 sits at ~-20.6 nats; candidate assigns -inf there.
    ref = [0.0, -20.0, -1.0, -1.0]
    cand = [-1.0, float("-inf"), 0.0, 0.0]
    value = kl_divergence(ref, cand)
    assert math.isfinite(value)  # cutoff skipped the doomed term

    # the same doomed term ABOVE the cutoff must surface as +inf
    ref_hot = [0.0, -1.0, -1.0, -1.0]  # index 1 now well above -16 nats
    assert kl_divergence(ref_hot, cand) == math.inf


def test_non_finite_positions_are_excluded_and_budget_enforced():
    poisoned = [float("nan")] + [0.1] * (VOCAB - 1)
    assert not position_valid(poisoned, CAND_A)
    assert position_valid(REF_A, CAND_A)

    kls = [kl_divergence(REF_A, CAND_A)] + [float("inf")] + [0.0] * 998
    tops = [True] * 1000
    agg = aggregate_domain("wiki_test", kls, tops)
    assert agg.excluded_positions == 1
    assert agg.n_positions == 999
    assert agg.kl_mean == pytest.approx(KL_A / 999.0, rel=1e-9)
    assert agg.within_exclusion_budget  # 0.1% exactly at the budget edge

    flooded = aggregate_domain("wiki_test", [float("inf")] * 5 + [0.0] * 995, [True] * 1000)
    assert not flooded.within_exclusion_budget  # 0.5% > 0.1% budget


def test_two_level_aggregation_macro_primary_micro_diagnostic():
    d_hi = aggregate_domain("chinese", [0.5] * 10, [True] * 10)
    d_lo = aggregate_domain("code", [0.01] * 1000, [True] * 1000)
    domains = {"chinese": d_hi, "code": d_lo}
    weights = {"chinese": 0.5, "code": 0.5}

    macro = macro_kl(domains, weights)
    micro = micro_kl(domains)
    assert macro == pytest.approx(0.255)
    # micro is dragged by the 1000-token domain: must NOT equal macro
    assert micro < macro
    assert micro == pytest.approx((0.5 * 10 + 0.01 * 1000) / 1010)


def test_regressed_domains_detection():
    before = {"a": _agg("a", 0.10), "b": _agg("b", 0.20)}
    after = {"a": _agg("a", 0.12), "b": _agg("b", 0.18)}
    assert regressed_domains(before, after) == ["a"]


def _agg(domain, kl):
    return DomainAggregate(domain=domain, n_positions=100, excluded_positions=0,
                           kl_mean=kl, same_top_frac=0.9, ppl=None)


def test_contract_valid_and_hash_stable():
    import json

    validate_contract(EVAL_V1)
    digest_a = contract_digest()
    digest_b = contract_digest()
    assert digest_a == digest_b  # deterministic
    assert contract_hash(EVAL_V1) == digest_a

    mutated = json.loads(json.dumps(EVAL_V1))
    mutated["metrics"]["domain_weights"]["code"] = 0.3  # weights no longer sum to 1
    with pytest.raises(ContractError):
        validate_contract(mutated)


def test_result_schema_deterministic_and_complete():
    domains = [
        DomainResult(domain="wiki_test", tokens=1000, kl=0.05, same_top=0.91, ppl=5.9),
        DomainResult(domain="wiki_valid", tokens=1000, kl=0.048, same_top=0.912, ppl=5.8),
        DomainResult(domain="chinese", tokens=1000, kl=0.25, same_top=0.86, ppl=7.4),
        DomainResult(domain="code", tokens=1000, kl=0.24, same_top=0.885, ppl=7.1),
        DomainResult(domain="agent_chat", tokens=1000, kl=0.15, same_top=0.89, ppl=6.2),
    ]
    result = build_eval_result(
        contract_digest=contract_digest(),
        model_hash="m" * 64,
        reference_logits_hash="r" * 64,
        domain_results=domains,
        candidate_plan_hash="p" * 64,
    )
    text = serialize_result(result)

    assert text == serialize_result(build_eval_result(
        contract_digest=contract_digest(),
        model_hash="m" * 64,
        reference_logits_hash="r" * 64,
        domain_results=domains,
        candidate_plan_hash="p" * 64,
    ))
    for key in ("result_schema", "evaluator_contract", "evaluator_contract_hash", "model_hash",
                "reference_logits_hash", "reference_manifest_hash", "macro_kl", "macro_same_top",
                "micro_kl", "worst_domain_kl", "comparison", "domains"):
        assert key in result
    assert result["result_schema"] == "fit.eval_result.v1"
    assert result["comparison"] is None  # base evaluator never fabricates regression verdicts
    assert result["worst_domain"] == "chinese"
    assert result["domains"]["code"]["kl"] == 0.24
    assert canonical_json(result) == text

    # comparison layer: regression verdict requires an explicit baseline
    baseline = build_eval_result(
        contract_digest=contract_digest(),
        model_hash="b" * 64,
        reference_logits_hash="r" * 64,
        domain_results=[
            DomainResult(domain=d.domain, tokens=d.tokens, kl=d.kl * 0.9,
                         same_top=d.same_top, ppl=d.ppl)
            for d in domains
        ],
    )
    comparison = compare_results(baseline, result, baseline_result_hash="h" * 64)
    assert comparison["regressed_domains"] == sorted(d.domain for d in domains)  # baseline uniformly 10% lower
    with_comparison = build_eval_result(
        contract_digest=contract_digest(),
        model_hash="m" * 64,
        reference_logits_hash="r" * 64,
        domain_results=domains,
        comparison=comparison,
    )
    assert with_comparison["comparison"]["regressed_domains"] == sorted(d.domain for d in domains)


def test_reference_manifest_closes_provenance_loop():
    manifest = build_reference_manifest(
        source_bf16_gguf_sha256="s" * 64,
        tokenizer_hash="t" * 64,
        evaluator_contract_hash=contract_digest(),
        runtime_provenance={"llama_cpp_source_revision": "b10666", "binary_sha256": "x" * 64},
        domains={"chinese": {
            "corpus_sha256": "c" * 64,
            "reference_kld_sha256": "k" * 64,
            "raw_bytes": 65534,
            "unicode_codepoints": 61792,
            "expected_valid_tokens": 15400,
        }},
    )
    assert manifest["manifest_schema"] == "fit.eval_reference_manifest.v1"
    assert manifest["domains"]["chinese"]["raw_bytes"] == 65534

    from fit_gguf.eval import build_reference_manifest as _brm
    try:
        _brm("s" * 64, "t" * 64, "h" * 64, {}, domains={"chinese": {"corpus_sha256": "c" * 64}})
        raise AssertionError("expected missing-field rejection")
    except ValueError as exc:
        assert "missing fields" in str(exc)


def test_parse_llama_kl_log_real_format():
    parsed = parse_llama_kl_log(REAL_LOG_TAIL)
    assert parsed["mean_kld"] == pytest.approx(0.052627)
    assert parsed["same_top_pct"] == pytest.approx(90.968)
    assert parsed["rms_delta_p_pct"] == pytest.approx(7.064)

    with pytest.raises(EvalLogError):
        parse_llama_kl_log("not a log")
    with pytest.raises(EvalLogError):
        parse_llama_kl_log("====== KL divergence statistics ======\ntruncated")
