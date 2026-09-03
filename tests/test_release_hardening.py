"""Release-gate hardening tests (Codex audit 2026-09-04).

Covers the P1/P2 fixes: frozen eval-v1 provenance enforcement, guard
profiles bound to source weights, and machine-detectable fidelity-search
exit codes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from fit_gguf.eval import contract_digest
from fit_gguf.eval.provenance import (
    EvalProvenanceError,
    sha256_file,
    verify_eval_v1_provenance,
)
from fit_gguf.fidelity import GuardProfileError, resolve_guard_profile
from fit_gguf.fidelity_runner import FidelityRunnerError, RunnerConfig, SearchExecutor
from fit_gguf.pipeline import PipelineError


def _write_synth_inputs(tmp_path):
    """Build a minimal-but-valid frozen closure: freeze + manifest + files.

    The manifest is bound to the freeze the way the real artifact chain is:
    contract hash pin + freeze-recorded manifest sha prefix + source pin.
    """
    refs = tmp_path / "refs"
    refs.mkdir()
    data = tmp_path / "eval-data"
    data.mkdir()
    domains = {}
    for domain, suffix in {
        "wiki_test": "64k",
        "wiki_valid": "valid-64k",
        "chinese": "cn-64k",
        "code": "code-64k",
        "agent_chat": "agent-64k",
    }.items():
        (refs / f"bf16-{domain}.kld").write_bytes(f"ref-{domain}".encode())
        (data / f"kl-eval-{suffix}.txt").write_bytes(f"corpus-{domain}".encode())
        domains[domain] = {
            "reference_kld_sha256": sha256_file(refs / f"bf16-{domain}.kld"),
            "corpus_sha256": sha256_file(data / f"kl-eval-{suffix}.txt"),
        }
    manifest = {
        "manifest_schema": "fit.eval_reference_manifest.v1",
        "domains": domains,
        "evaluator_contract_hash": contract_digest(),
        "source_bf16_gguf_sha256": "a" * 64,
    }
    manifest_path = tmp_path / "reference-manifest-synth.json"
    manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    freeze = {
        "schema": "fit.eval_v1_freeze.v1",
        "status": "FROZEN",
        "contract_id": "eval-v1",
        "final_contract_digest": contract_digest(),
        "freeze_conditions": {
            "reference_regeneration": {
                "manifest_sha256_prefix": sha256_file(manifest_path)[:16],
            }
        },
    }
    freeze_path = tmp_path / "FREEZE.json"
    freeze_path.write_text(json.dumps(freeze, indent=1), encoding="utf-8")
    return freeze_path, manifest_path, refs, data


def test_provenance_happy_path(tmp_path):
    freeze, manifest, refs, data = _write_synth_inputs(tmp_path)
    report = verify_eval_v1_provenance(refs, data, freeze, manifest)
    assert report.contract_digest == contract_digest()
    assert set(report.reference_kld_sha256) == {
        "wiki_test", "wiki_valid", "chinese", "code", "agent_chat",
    }


def test_provenance_rejects_corrupted_reference(tmp_path):
    freeze, manifest, refs, data = _write_synth_inputs(tmp_path)
    (refs / "bf16-chinese.kld").write_bytes(b"tampered")
    with pytest.raises(EvalProvenanceError, match="bf16-chinese.kld"):
        verify_eval_v1_provenance(refs, data, freeze, manifest)


def test_provenance_rejects_wrong_corpus(tmp_path):
    freeze, manifest, refs, data = _write_synth_inputs(tmp_path)
    (data / "kl-eval-cn-64k.txt").write_bytes(b"wrong slice")
    with pytest.raises(EvalProvenanceError, match="kl-eval-cn-64k.txt"):
        verify_eval_v1_provenance(refs, data, freeze, manifest)


def test_provenance_rejects_contract_drift(tmp_path):
    freeze, manifest, refs, data = _write_synth_inputs(tmp_path)
    payload = json.loads(freeze.read_text())
    payload["final_contract_digest"] = "0" * 64
    freeze.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(EvalProvenanceError, match="contract drift"):
        verify_eval_v1_provenance(refs, data, freeze, manifest)


def test_provenance_rejects_unfrozen_contract(tmp_path):
    freeze, manifest, refs, data = _write_synth_inputs(tmp_path)
    payload = json.loads(freeze.read_text())
    payload["status"] = "DRAFT"
    freeze.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(EvalProvenanceError, match="not FROZEN"):
        verify_eval_v1_provenance(refs, data, freeze, manifest)


def test_executor_refuses_to_eval_without_required_provenance(tmp_path):
    config = RunnerConfig(
        runtime=tmp_path / "rt",
        imatrix=tmp_path / "imx",
        refs_dir=tmp_path / "refs",
        eval_data_dir=tmp_path / "data",
        work_dir=tmp_path / "work",
        out_dir=tmp_path / "out",
        model_name="m",
        guard_registry=None,
        refine_profile=None,
        require_eval_provenance=True,
        eval_provenance=None,
    )
    with pytest.raises(FidelityRunnerError, match="require_eval_provenance"):
        SearchExecutor(config, windows=[], contract=None,
                       manifest_path=tmp_path / "m.txt", logs_out_dir=tmp_path / "logs")


def _write_guard_registry(tmp_path, *, source_sha256, status="validated"):
    registry = tmp_path / "guard"
    registry.mkdir()
    profile = {
        "guard_profile_id": "guard-test-exact-v1",
        "guard_profile_version": 1,
        "evaluator_contract": "eval-v1",
        "scope": {"type": "exact_model", "identifier": "test-model"},
        "status": status,
        "source_sha256": source_sha256,
        "tiers": {
            "quality": {"kl_anchor": 0.05, "same_top_floor": 0.9475},
            "balanced": {"kl_anchor": 0.10, "same_top_floor": 0.9118},
            "compact": {"kl_anchor": 0.15, "same_top_floor": 0.8894},
            "mini": {"kl_anchor": 0.20, "same_top_floor": 0.8503},
        },
    }
    content = {k: v for k, v in profile.items() if k != "profile_hash"}
    import hashlib

    profile["profile_hash"] = hashlib.sha256(
        json.dumps(content, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    (registry / "guard-test-exact-v1.yaml").write_text(
        yaml.safe_dump(profile, sort_keys=False), encoding="utf-8"
    )
    return registry


def test_guard_source_binding_requires_matching_digest(tmp_path):
    registry = _write_guard_registry(
        tmp_path, source_sha256="a" * 64
    )
    # same name, no digest supplied -> NOT covered (fail closed)
    assert resolve_guard_profile("test-model", registry) is None
    # wrong weights -> NOT covered
    assert resolve_guard_profile("test-model", registry, "b" * 64) is None
    # same weights -> covered
    profile = resolve_guard_profile("test-model", registry, "a" * 64)
    assert profile is not None and profile.profile_id == "guard-test-exact-v1"


def test_guard_without_pin_ignores_digest(tmp_path):
    registry = _write_guard_registry(tmp_path, source_sha256=None)
    assert resolve_guard_profile("test-model", registry) is not None
    assert resolve_guard_profile("test-model", registry, "c" * 64) is not None


def _run_fidelity_search(monkeypatch, capsys, summary):
    from fit_gguf import cli as cli_module

    monkeypatch.setattr(
        "fit_gguf.product.fidelity_search_product", lambda **kwargs: summary
    )
    argv = [
        "fidelity-search",
        "--source", "model-BF16.gguf",
        "--imatrix", "imx.gguf",
        "--runtime", "rt",
        "--refs-dir", "refs",
        "--eval-data-dir", "slices",
        "--guard-registry", "profiles/guard",
        "--tier", "compact",
        "--analysis", "work/analysis",
        "--out-dir", "out",
        "--work-dir", "work",
        "--manifest", "m.txt",
        "--logs-dir", "logs",
        "--freeze", "FREEZE.json",
        "--reference-manifest", "rm.json",
    ]
    return cli_module.main(argv)


def test_cli_fidelity_exit_verified_pass(monkeypatch, capsys):
    code = _run_fidelity_search(monkeypatch, capsys, {
        "status": "verified_pass",
        "best": {"size_bytes": 11991706848, "macro_kl": 0.1486, "same_top": 0.8909},
        "active_constraint": "kl", "fresh_evals": 0, "budget": 8, "note": None,
        "artifact": {"path": "a.gguf", "size_bytes": 11991706848, "g2_delta": 0},
    })
    assert code == 0


def test_cli_fidelity_exit_noise_inversion_is_failure(monkeypatch, capsys):
    code = _run_fidelity_search(monkeypatch, capsys, {
        "status": "noise_inversion",
        "best": {"size_bytes": 11991706848, "macro_kl": 0.1486, "same_top": 0.8909},
        "active_constraint": "kl", "fresh_evals": 0, "budget": 8, "note": None,
        "artifact": None,
    })
    assert code == 4  # NOT 0 — a best observation is not a delivery
    out = capsys.readouterr().out
    assert "NOT auto-delivered" in out


def test_cli_fidelity_exit_no_pass(monkeypatch, capsys):
    code = _run_fidelity_search(monkeypatch, capsys, {
        "status": "no_pass", "best": None, "artifact": None,
        "product_status": "NOT REACHABLE within the validated healthy frontier",
    })
    assert code == 3


def test_cli_guard_refusal_is_clean_error(monkeypatch, capsys):
    """A GuardProfileError must exit 2 with a message, not a traceback."""
    from fit_gguf import cli as cli_module
    from fit_gguf.fidelity import GuardProfileError as _GPE

    def _raise(**kwargs):
        raise _GPE("Fidelity validation unavailable: ...")

    monkeypatch.setattr("fit_gguf.product.fidelity_search_product", _raise)
    argv = [
        "fidelity-search",
        "--source", "model-BF16.gguf",
        "--imatrix", "imx.gguf",
        "--runtime", "rt",
        "--refs-dir", "refs",
        "--eval-data-dir", "slices",
        "--guard-registry", "profiles/guard",
        "--tier", "compact",
        "--analysis", "work/analysis",
        "--out-dir", "out",
        "--work-dir", "work",
        "--manifest", "m.txt",
        "--logs-dir", "logs",
        "--freeze", "FREEZE.json",
        "--reference-manifest", "rm.json",
    ]
    code = cli_module.main(argv)
    assert code == 2
    err = capsys.readouterr().err
    assert "Fidelity validation unavailable" in err


def test_pipeline_error_still_caught():
    """Regression: PipelineError handling is unchanged."""
    from fit_gguf import cli as cli_module

    assert issubclass(GuardProfileError, ValueError)
    # the CLI catch tuple includes PipelineError via the shared handler
    import inspect

    source = inspect.getsource(cli_module.main)
    assert "PipelineError" in source and "GuardProfileError" in source
    assert "ProductError" in source and "EvalProvenanceError" in source


def test_product_requires_freeze_path(tmp_path):
    from fit_gguf.product import ProductError, fidelity_search_product

    # PipelineError/ProductError surface; missing freeze fails fast BEFORE
    # any expensive work
    with pytest.raises((EvalProvenanceError, ProductError, PipelineError)) as excinfo:
        fidelity_search_product(
            source="m.gguf",
            imatrix="i.gguf",
            runtime="rt",
            refs_dir="refs",
            eval_data_dir="slices",
            guard_registry="guard",
            tier="compact",
            out_dir=str(tmp_path / "out"),
            work_dir=str(tmp_path / "work"),
            manifest_path=str(tmp_path / "m.txt"),
            logs_dir=str(tmp_path / "logs"),
        )
    assert "freeze_path" in str(excinfo.value)


def test_provenance_rejects_manifest_pinned_to_other_contract(tmp_path):
    freeze, manifest, refs, data = _write_synth_inputs(tmp_path)
    payload = json.loads(manifest.read_text())
    payload["evaluator_contract_hash"] = "0" * 64
    manifest.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    with pytest.raises(EvalProvenanceError, match="not pinned to the frozen contract"):
        verify_eval_v1_provenance(refs, data, freeze, manifest)


def test_provenance_rejects_manifest_unbound_to_freeze(tmp_path):
    freeze, manifest, refs, data = _write_synth_inputs(tmp_path)
    payload = json.loads(manifest.read_text())
    payload["repin_note"] = "post-freeze edit"
    manifest.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    with pytest.raises(EvalProvenanceError, match="not bound to this freeze"):
        verify_eval_v1_provenance(refs, data, freeze, manifest)


def test_provenance_rejects_foreign_weights(tmp_path):
    freeze, manifest, refs, data = _write_synth_inputs(tmp_path)
    with pytest.raises(EvalProvenanceError, match="different weights"):
        verify_eval_v1_provenance(refs, data, freeze, manifest, source_sha256="b" * 64)


def test_provenance_accepts_matching_weights(tmp_path):
    freeze, manifest, refs, data = _write_synth_inputs(tmp_path)
    report = verify_eval_v1_provenance(
        refs, data, freeze, manifest, source_sha256="a" * 64
    )
    assert report.source_bf16_gguf_sha256 == "a" * 64


# --- fresh-probe integration: pinned guard + real executor loop -------------

def test_fresh_probe_with_pinned_guard_and_verified_provenance(tmp_path, monkeypatch):
    """Codex audit round 2: a pinned (source-bound) Guard Profile must not
    reject fresh probes inside the search. Cache-only runs passed while the
    internal plan() re-resolved the guard without the weights digest — this
    test exercises the full fresh path: plan -> quantize -> eval."""
    import json as _json

    from fit_gguf.eval.provenance import EvalProvenance
    from fit_gguf import fidelity_runner as runner_module
    from fit_gguf.fidelity_search import TierContract

    source_sha = "a" * 64
    registry = _write_guard_registry(tmp_path, source_sha256=source_sha)

    # fake pinned llama.cpp runtime: prints a parseable KL log, exits 0
    rt = tmp_path / "rt"
    rt.mkdir()
    kl_log = "====== KL divergence statistics ======\nMean KLD: 0.090000 ± 0.010000\nSame top p: 92.0000 ± 0.1000 %\n"
    fake = rt / "llama-perplexity"
    fake.write_text("#!/bin/sh\ncat <<'KLEOF'\n" + kl_log + "KLEOF\n")
    fake.chmod(0o755)

    # fake planner: deterministic delivered size == target; writes the
    # tensor-types file the (fake) quantizer consumes
    def fake_plan(analysis_path, out_prefix, **kwargs):
        assert kwargs.get("source_sha256") == source_sha, (
            "internal plan must carry the weights digest into the guard check"
        )
        assert kwargs.get("fidelity_tier") == "balanced"
        Path(f"{out_prefix}-tensor-types.txt").write_text("blk.0.attn_q=q6_K\n")
        return {"predicted_size_bytes": kwargs["target_bytes"],
                "suggested_filename": None}

    def fake_quantize(analysis_path, tensor_types, out_path, **kwargs):
        Path(out_path).write_bytes(b"fake-gguf-bytes")

    monkeypatch.setattr(runner_module, "pipeline_plan", fake_plan)
    monkeypatch.setattr(runner_module, "pipeline_quantize", fake_quantize)

    (tmp_path / "reference-manifest.json").write_text("{}\n")
    provenance = EvalProvenance(
        freeze_path="FREEZE.json",
        contract_digest=contract_digest(),
        reference_manifest_path=str(tmp_path / "reference-manifest.json"),
        reference_kld_sha256={}, corpus_sha256={},
        source_bf16_gguf_sha256=source_sha,
        reference_manifest_file_sha256="f" * 64,
    )
    config = RunnerConfig(
        runtime=rt,
        imatrix=tmp_path / "imx.gguf",
        refs_dir=tmp_path / "refs",
        eval_data_dir=tmp_path / "slices",
        work_dir=tmp_path / "work",
        out_dir=tmp_path / "out",
        model_name="test-model",
        guard_registry=registry,
        refine_profile=None,
        eval_provenance=provenance,
        require_eval_provenance=True,
        source_sha256=source_sha,
    )
    windows = [runner_module.Window(
        analysis_path=tmp_path / "analysis",
        lower_preset="IQ3_XS", upper_preset="IQ3_S",
        lower_size=11 * 1024**3, upper_size=int(11.6 * 1024**3),
    )]
    contract = TierContract(tier="balanced", kl_anchor=0.10, same_top_floor=0.9118)
    summary = runner_module.run_tier_search(
        contract, config, windows, seeds=(),
        min_size=windows[0].lower_size, max_size=windows[0].upper_size,
        budget=8, manifest_path=tmp_path / "manifest.txt",
        logs_out_dir=tmp_path / "logs",
    )
    assert summary["status"] == "verified_pass", summary
    assert summary["fresh_evals"] >= 1  # the fresh path actually ran

    # fresh evals wrote provenance sidecars attesting this closure
    sidecar = (tmp_path / "manifest.txt").parent / "seed-provenance.jsonl"
    records = [_json.loads(line) for line in sidecar.read_text().splitlines() if line]
    assert records and all(r["attestation"] == "runtime-verified" for r in records)
    assert all(r["eval_contract_digest"] == contract_digest() for r in records)


# --- round 3: seeds bind to BOTH contract digest and reference manifest -----

def _seed_sidecar(tmp_path, manifest_sha, *, contract=None):
    from fit_gguf.eval import contract_digest as _cd

    logs = tmp_path / "logs"
    logs.mkdir(exist_ok=True)
    for domain in ("wiki_test", "wiki_valid", "chinese", "code", "agent_chat"):
        (logs / f"eval-orcarouter-SEED-{domain}.log").write_text(
            "====== KL divergence statistics ======\n"
            "Mean KLD: 0.090000 ± 0.010000\nSame top p: 92.0000 ± 0.1000 %\n"
        )
    manifest = tmp_path / "manifest.txt"
    manifest.write_text(f"orcarouter-SEED  12000000000  {'0' * 64}\n")
    prov = tmp_path / "seed-provenance.jsonl"
    prov.write_text(json.dumps({
        "name": "orcarouter-SEED", "size_bytes": 12000000000,
        "eval_contract_digest": contract or _cd(),
        "reference_manifest_sha256": manifest_sha,
        "attestation": "experiment-record",
    }) + "\n")
    return manifest, logs, prov


def test_strict_seed_requires_matching_manifest_sha(tmp_path):
    from fit_gguf.fidelity_runner import FidelityRunnerError, load_seeds

    # contract matches, manifest sha matches -> admitted
    manifest, logs, prov = _seed_sidecar(tmp_path, "f" * 64)
    seeds = load_seeds(manifest, logs, "orcarouter-", provenance_path=prov,
                       require_seed_provenance=True, reference_manifest_sha256="f" * 64)
    assert {s.size_bytes for s in seeds} == {12000000000}

    # contract matches but MANIFEST differs -> must be rejected (round-3 P1)
    manifest2, logs2, prov2 = _seed_sidecar(tmp_path, "e" * 64)
    seeds2 = load_seeds(manifest2, logs2, "orcarouter-", provenance_path=prov2,
                        require_seed_provenance=True, reference_manifest_sha256="f" * 64)
    assert seeds2 == ()

    # strict without the active manifest sha is a configuration error
    with pytest.raises(FidelityRunnerError, match="reference manifest SHA"):
        load_seeds(manifest, logs, "orcarouter-", provenance_path=prov,
                   require_seed_provenance=True)

    # loose mode (no strict flag) keeps admitting for research scripts
    seeds3 = load_seeds(manifest, logs, "orcarouter-", provenance_path=prov)
    assert {s.size_bytes for s in seeds3} == {12000000000}
