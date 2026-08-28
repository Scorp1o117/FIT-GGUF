"""Exact-size baseline selection for continuous FIT budgets."""

from dataclasses import dataclass
from typing import Iterable


class BaselineSelectionError(ValueError):
    """Raised when a target cannot be bracketed by valid preset sizes."""


@dataclass(frozen=True, slots=True)
class PresetSize:
    name: str
    size_bytes: int

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise BaselineSelectionError("Preset name cannot be empty")
        if self.size_bytes <= 0:
            raise BaselineSelectionError("Preset size must be positive")


@dataclass(frozen=True, slots=True)
class BaselinePlan:
    target_bytes: int
    lower: PresetSize
    upper: PresetSize | None
    extra_budget_bytes: int

    @property
    def upper_gap_bytes(self) -> int | None:
        if self.upper is None:
            return None
        return self.upper.size_bytes - self.lower.size_bytes


def select_baselines(target_bytes: int, presets: Iterable[PresetSize]) -> BaselinePlan:
    """Select adjacent size baselines without assuming preset-name ordering."""
    if target_bytes <= 0:
        raise BaselineSelectionError("Target size must be positive")
    ordered = sorted(presets, key=lambda preset: (preset.size_bytes, preset.name))
    if not ordered:
        raise BaselineSelectionError("At least one preset size is required")

    names = [preset.name for preset in ordered]
    if len(names) != len(set(names)):
        raise BaselineSelectionError("Preset names must be unique")
    sizes = [preset.size_bytes for preset in ordered]
    if len(sizes) != len(set(sizes)):
        raise BaselineSelectionError(
            "Preset sizes must be unique because size-only baseline precedence would be ambiguous"
        )

    fitting = [preset for preset in ordered if preset.size_bytes <= target_bytes]
    if not fitting:
        raise BaselineSelectionError(
            f"Target {target_bytes} bytes is smaller than the minimum preset "
            f"{ordered[0].name} ({ordered[0].size_bytes} bytes)"
        )
    lower = fitting[-1]
    upper = next((preset for preset in ordered if preset.size_bytes > target_bytes), None)
    return BaselinePlan(
        target_bytes=target_bytes,
        lower=lower,
        upper=upper,
        extra_budget_bytes=target_bytes - lower.size_bytes,
    )
