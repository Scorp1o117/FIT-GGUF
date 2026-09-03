"""Frozen eval-v1 provenance enforcement (release-gate hardening).

"Running with eval-v1 parameters" and "verified against the frozen eval-v1
closure" are different claims. This module enforces the second one at runtime,
fail-closed:

* the loaded evaluator contract must hash to the frozen final digest;
* every domain reference ``.kld`` must match the frozen reference manifest;
* every evaluation slice must match the frozen corpus SHA-256.

Any mismatch raises :class:`EvalProvenanceError` instead of producing
measurements with unverified inputs.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from fit_gguf.eval import contract_digest

# domain -> eval slice filename suffix (frozen five-domain corpus layout)
SLICE_SUFFIX = {
    "wiki_test": "64k",
    "wiki_valid": "valid-64k",
    "chinese": "cn-64k",
    "code": "code-64k",
    "agent_chat": "agent-64k",
}
DOMAINS = tuple(SLICE_SUFFIX)


class EvalProvenanceError(RuntimeError):
    """Raised when runtime evaluation inputs fail the frozen eval-v1 closure."""


@dataclass(frozen=True, slots=True)
class EvalProvenance:
    """Verified binding between local eval inputs and the frozen eval-v1 closure."""

    freeze_path: str
    contract_digest: str
    reference_manifest_path: str
    reference_kld_sha256: dict[str, str]
    corpus_sha256: dict[str, str]
    source_bf16_gguf_sha256: str
    reference_manifest_file_sha256: str

    def as_dict(self) -> dict:
        return {
            "freeze_path": self.freeze_path,
            "contract_digest": self.contract_digest,
            "reference_manifest_path": self.reference_manifest_path,
            "reference_manifest_file_sha256": self.reference_manifest_file_sha256,
            "reference_kld_sha256": dict(self.reference_kld_sha256),
            "corpus_sha256": dict(self.corpus_sha256),
            "source_bf16_gguf_sha256": self.source_bf16_gguf_sha256,
        }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_eval_v1_provenance(
    refs_dir: str | Path,
    eval_data_dir: str | Path,
    freeze_path: str | Path,
    reference_manifest_path: str | Path,
    source_sha256: str | None = None,
) -> EvalProvenance:
    """Verify local evaluation inputs against the frozen eval-v1 closure.

    Enforces the FULL binding, not merely manifest self-consistency:

    * live contract digest == freeze ``final_contract_digest``;
    * manifest ``evaluator_contract_hash`` == the same frozen digest (a
      manifest pinned to an older contract revision is rejected);
    * the manifest file itself matches the SHA-256 prefix recorded in the
      freeze document;
    * the manifest pins ``source_bf16_gguf_sha256`` (refs must be bound to
      the generating weights), and a caller-supplied weights digest must
      match it;
    * per-domain reference ``.kld`` and corpus slice SHAs match the pins.

    Raises :class:`EvalProvenanceError` on any mismatch; returns the verified
    binding for embedding in manifests and release records.
    """
    refs = Path(refs_dir)
    data = Path(eval_data_dir)
    freeze_file = Path(freeze_path)
    manifest_file = Path(reference_manifest_path)

    try:
        freeze = json.loads(freeze_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvalProvenanceError(f"cannot read eval-v1 freeze file {freeze_file}: {error}") from error
    if freeze.get("status") != "FROZEN":
        raise EvalProvenanceError(f"{freeze_file}: eval contract is not FROZEN")
    frozen_digest = freeze.get("final_contract_digest")
    live_digest = contract_digest()
    if frozen_digest != live_digest:
        raise EvalProvenanceError(
            f"evaluator contract drift: live digest {live_digest} != frozen "
            f"{frozen_digest} ({freeze_file})"
        )

    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvalProvenanceError(
            f"cannot read reference manifest {manifest_file}: {error}"
        ) from error
    if manifest.get("manifest_schema") != "fit.eval_reference_manifest.v1":
        raise EvalProvenanceError(f"{manifest_file}: unexpected reference manifest schema")
    domains = manifest.get("domains")
    if not isinstance(domains, dict) or set(domains) != set(DOMAINS):
        raise EvalProvenanceError(
            f"{manifest_file}: reference manifest must pin exactly {sorted(DOMAINS)}"
        )
    if manifest.get("evaluator_contract_hash") != frozen_digest:
        raise EvalProvenanceError(
            f"{manifest_file}: evaluator_contract_hash "
            f"{manifest.get('evaluator_contract_hash')} is not pinned to the "
            f"frozen contract {frozen_digest} — re-pin the manifest"
        )
    manifest_sha = sha256_file(manifest_file)
    recorded_prefix = (
        (freeze.get("freeze_conditions", {}).get("reference_regeneration", {}) or {})
        .get("manifest_sha256_prefix")
    )
    if recorded_prefix and not manifest_sha.startswith(str(recorded_prefix)):
        raise EvalProvenanceError(
            f"{manifest_file}: sha256 {manifest_sha[:16]}… does not match the "
            f"freeze-recorded prefix {recorded_prefix} — the manifest is not "
            "bound to this freeze"
        )
    source_pin = manifest.get("source_bf16_gguf_sha256")
    if not source_pin or len(str(source_pin)) != 64:
        raise EvalProvenanceError(
            f"{manifest_file}: missing/invalid source_bf16_gguf_sha256 — "
            "references must be bound to the generating weights"
        )
    if source_sha256 is not None and str(source_pin) != source_sha256.lower():
        raise EvalProvenanceError(
            f"{manifest_file}: source_bf16_gguf_sha256 {source_pin} != supplied "
            f"weights digest {source_sha256} — the references belong to "
            "different weights"
        )

    kld_shas: dict[str, str] = {}
    corpus_shas: dict[str, str] = {}
    for domain in sorted(DOMAINS):
        pin = domains[domain]
        kld_file = refs / f"bf16-{domain}.kld"
        if not kld_file.is_file():
            raise EvalProvenanceError(f"missing reference file: {kld_file}")
        actual = sha256_file(kld_file)
        if actual != pin.get("reference_kld_sha256"):
            raise EvalProvenanceError(
                f"{kld_file}: sha256 {actual} != frozen reference "
                f"{pin.get('reference_kld_sha256')}"
            )
        kld_shas[domain] = actual

        slice_file = data / f"kl-eval-{SLICE_SUFFIX[domain]}.txt"
        if not slice_file.is_file():
            raise EvalProvenanceError(f"missing evaluation slice: {slice_file}")
        actual = sha256_file(slice_file)
        if actual != pin.get("corpus_sha256"):
            raise EvalProvenanceError(
                f"{slice_file}: sha256 {actual} != frozen corpus "
                f"{pin.get('corpus_sha256')}"
            )
        corpus_shas[domain] = actual

    return EvalProvenance(
        freeze_path=str(freeze_file),
        contract_digest=live_digest,
        reference_manifest_path=str(manifest_file),
        reference_kld_sha256=kld_shas,
        corpus_sha256=corpus_shas,
        source_bf16_gguf_sha256=str(source_pin),
        reference_manifest_file_sha256=manifest_sha,
    )
