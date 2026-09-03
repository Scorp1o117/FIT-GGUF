"""Loader and validation for the PRISM-exported ``fit.refine_dataset.v1``.

The dataset is produced by PRISM ``scripts/export_refine_dataset.py`` and
consumed here as the first training/calibration source for FIT Refine
(v0.2.1 M5). Per v0.2.1 §28/§60 governance, every record in this dataset is
treated as DEV evidence for FIT fitting: none of it may be cited as sealed
generalization proof, regardless of the ``split`` recorded at export time.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

DATASET_SCHEMA = "fit.refine_dataset.v1"

CHAIN_RECORD = "chain_step"
PROBE_RECORD = "single_action_probe"
CURVE_RECORD = "curve_point"

_CHAIN_REQUIRED = ("chain_id", "step_index", "step_label", "suite", "actions", "post_state")
_PROBE_REQUIRED = ("probe_id", "suite", "action", "post_state", "delta_macro_kl_pct")
_CURVE_REQUIRED = ("point_id", "suite", "size_gib", "macro_kl")

# Split values that appear in refine-dataset-v1 and their provenance.
_SOURCE_SPLITS = {
    "dev": "dev",
    "sealed-selection": "sealed-v1 (reused as selection set)",
    "sealed": "sealed-v2/v2b (amended holdout / preregistered)",
}

# G0 governance (v0.2.1 §60): fitting is only allowed for records from
# datasets whose sealed splits were *historically opened* before FIT
# adoption. The policy is keyed by dataset identity, not by split name, and
# fails closed: a future FIT-native sealed dataset (Ornith, G11) is never
# silently absorbed into fitting, whatever its records call themselves.
_FITTING_POLICY: dict[str, set[str]] = {
    "refine-dataset-v1": {"dev", "sealed-selection", "sealed"},
}


class RefineDatasetError(ValueError):
    """Raised when a refine dataset fails schema validation."""


@dataclass(frozen=True, slots=True)
class RefineDataset:
    """Validated in-memory view of a refine-dataset-v1 directory."""

    path: Path
    chains: tuple[dict, ...]
    probes: tuple[dict, ...]
    curve_points: tuple[dict, ...]
    priors: dict
    index: tuple[dict, ...]
    digest: str

    @property
    def dataset_id(self) -> str:
        return str(self.priors.get("dataset_id", "unknown"))

    def split_summary(self) -> dict[str, int]:
        summary: dict[str, int] = {}
        for record in self.chains + self.probes:
            split = str(record.get("split", "unknown"))
            summary[split] = summary.get(split, 0) + 1
        return dict(sorted(summary.items()))

    def role_probes(self) -> tuple[dict, ...]:
        return tuple(r for r in self.probes if r.get("probe_id", "").startswith("role-"))

    def swap_probes(self) -> tuple[dict, ...]:
        return tuple(r for r in self.probes if r.get("probe_id", "").startswith("iq-probe-"))

    def records_for_fitting(self) -> tuple[tuple[dict, ...], tuple[dict, ...]]:
        """Split records into (fitting-allowed, excluded-sealed) per G0 policy.

        Fail-closed: datasets missing from ``_FITTING_POLICY`` yield no
        fitting records at all — sealed discipline cannot be broken by a
        dataset that simply calls its records "dev".
        """
        policy = _FITTING_POLICY.get(self.dataset_id)
        if policy is None:
            return (), tuple(self.chains + self.probes)
        allowed, excluded = [], []
        for record in self.chains + self.probes:
            (allowed if record.get("split") in policy else excluded).append(record)
        return tuple(allowed), tuple(excluded)


def _read_jsonl(path: Path) -> tuple[dict, ...]:
    records: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RefineDatasetError(f"{path.name}:{line_no}: invalid JSON ({exc})") from exc
    return tuple(records)


def _validate(records: tuple[dict, ...], required: tuple[str, ...], label: str) -> None:
    for i, record in enumerate(records):
        if record.get("schema") != DATASET_SCHEMA:
            raise RefineDatasetError(
                f"{label}[{i}]: schema {record.get('schema')!r} != {DATASET_SCHEMA!r}"
            )
        missing = [key for key in required if key not in record]
        if missing:
            raise RefineDatasetError(f"{label}[{i}]: missing fields {missing}")


def _digest(path: Path) -> str:
    hasher = hashlib.sha256()
    for name in sorted(p.name for p in path.iterdir() if p.is_file()):
        hasher.update(name.encode("utf-8"))
        hasher.update(hashlib.sha256((path / name).read_bytes()).digest())
    return hasher.hexdigest()


def load_refine_dataset(path: str | Path) -> RefineDataset:
    """Load and validate a refine-dataset-v1 directory."""
    root = Path(path)
    if not root.is_dir():
        raise RefineDatasetError(f"dataset directory not found: {root}")

    def read(name: str) -> tuple[dict, ...]:
        file = root / name
        if not file.is_file():
            raise RefineDatasetError(f"missing dataset file: {file}")
        return _read_jsonl(file)

    chains = read("chains.jsonl")
    probes = read("single_action_probes.jsonl")
    curve = read("curve_points.jsonl")
    index = read("probe_index.jsonl")

    priors_file = root / "priors.json"
    if not priors_file.is_file():
        raise RefineDatasetError(f"missing dataset file: {priors_file}")
    priors = json.loads(priors_file.read_text(encoding="utf-8"))

    _validate(chains, _CHAIN_REQUIRED, "chains")
    _validate(probes, _PROBE_REQUIRED, "probes")
    _validate(curve, _CURVE_REQUIRED, "curve_points")

    for record in chains + probes:
        split = record.get("split")
        if split not in _SOURCE_SPLITS:
            raise RefineDatasetError(f"record {record.get('chain_id') or record.get('probe_id')!r}: unknown split {split!r}")

    return RefineDataset(
        path=root,
        chains=chains,
        probes=probes,
        curve_points=curve,
        priors=priors,
        index=index,
        digest=_digest(root),
    )


def _record_id(record: dict) -> str:
    return str(record.get("probe_id") or record.get("chain_id") or "<unnamed>")
