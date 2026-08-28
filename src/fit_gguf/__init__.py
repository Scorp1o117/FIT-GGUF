"""FIT-GGUF: Fit-to-Size Intelligent Tensor Quantization."""

from fit_gguf.dry_run import mib_to_bytes, parse_dry_run
from fit_gguf.candidates import (
    CandidateGenerationError,
    CandidateSet,
    RejectedTransition,
    UpgradeCandidate,
    generate_upgrade_candidates,
)
from fit_gguf.gguf import (
    GGUFError,
    GGUFLayout,
    GGUFSizePrediction,
    ImatrixProvenance,
    QuantizationMetadata,
    TensorSize,
    predict_quantized_size,
    read_gguf_layout,
)
from fit_gguf.imatrix import (
    ImatrixProfile,
    ImatrixTensorProfile,
    load_imatrix_profile,
    write_profile_json,
)
from fit_gguf.llama_integration import write_tensor_type_file
from fit_gguf.models import DryRunParseError, DryRunResult, DryRunTensorAssignment
from fit_gguf.optimizer import (
    OptimizationError,
    OptimizationPlan,
    optimize_block_balanced,
    optimize_greedy,
    optimize_random,
    write_fit_recipe,
)
from fit_gguf.planner import (
    BaselinePlan,
    BaselineSelectionError,
    PresetSize,
    select_baselines,
)

__all__ = [
    "DryRunParseError",
    "DryRunResult",
    "DryRunTensorAssignment",
    "BaselinePlan",
    "BaselineSelectionError",
    "CandidateGenerationError",
    "CandidateSet",
    "GGUFError",
    "GGUFLayout",
    "GGUFSizePrediction",
    "ImatrixProvenance",
    "ImatrixProfile",
    "ImatrixTensorProfile",
    "OptimizationError",
    "OptimizationPlan",
    "QuantizationMetadata",
    "TensorSize",
    "RejectedTransition",
    "PresetSize",
    "UpgradeCandidate",
    "generate_upgrade_candidates",
    "mib_to_bytes",
    "load_imatrix_profile",
    "optimize_greedy",
    "optimize_block_balanced",
    "optimize_random",
    "parse_dry_run",
    "predict_quantized_size",
    "read_gguf_layout",
    "select_baselines",
    "write_profile_json",
    "write_fit_recipe",
    "write_tensor_type_file",
]
