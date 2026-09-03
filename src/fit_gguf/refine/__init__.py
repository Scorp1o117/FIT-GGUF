"""FIT Refine: dataset loader, profile fitting, and proposal scoring (v0.2.1)."""

from fit_gguf.refine.dataset import (
    DATASET_SCHEMA,
    RefineDataset,
    RefineDatasetError,
    load_refine_dataset,
)
from fit_gguf.refine.proposal import ProposalScorer, TransitionProposal
from fit_gguf.refine.profile import (
    PROFILE_SCHEMA,
    ProfileFitError,
    fit_band_cells,
    fit_profile,
    fit_role_corrections,
    load_profile,
    resolve_band_correction,
    save_profile,
    validate_profile,
)

__all__ = [
    "DATASET_SCHEMA",
    "PROFILE_SCHEMA",
    "ProposalScorer",
    "ProfileFitError",
    "RefineDataset",
    "RefineDatasetError",
    "TransitionProposal",
    "fit_band_cells",
    "fit_profile",
    "fit_role_corrections",
    "load_profile",
    "load_refine_dataset",
    "resolve_band_correction",
    "save_profile",
    "validate_profile",
]
