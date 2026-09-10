"""Packaging guard: assets loaded by path at runtime must ship in the wheel.

Two releases in a row shipped a wheel that was missing an asset the CLI loads
from its own package directory:

- v0.2 shipped without ``profiles/guard/*.yaml`` (Guard Profiles moved into the
  package only in v0.3);
- v0.3 initially shipped without ``contracts/*.json`` (the calibration contract
  was added to the package but not to ``package-data``).

Both defects were invisible from a source checkout — ``PYTHONPATH=src`` finds
everything on disk — and both broke only for users who installed the package.
These tests expand the declared ``[tool.setuptools.package-data]`` globs and
assert that every runtime-loaded asset class is actually covered.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PKG = REPO / "src" / "fit_gguf"

# Asset paths the runtime resolves relative to the package directory. Keep in
# sync with `fit_gguf.calibration.default_contract_path`,
# `fit_gguf.registry.default_guard_registry` and `fit_gguf.registry.find_package_dir`.
RUNTIME_ASSET_DIRS = ("contracts/*.json", "profiles/guard/*.yaml", "registry/**/*.json")


def _declared_globs() -> list[str]:
    data = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    return list(data["tool"]["setuptools"]["package-data"]["fit_gguf"])


def _covered() -> set[str]:
    """Package-relative paths that the declared package-data globs ship."""
    out: set[str] = set()
    for pattern in _declared_globs():
        for path in PKG.glob(pattern):
            if path.is_file():
                out.add(path.relative_to(PKG).as_posix())
    return out


def _on_disk() -> set[str]:
    out: set[str] = set()
    for pattern in RUNTIME_ASSET_DIRS:
        for path in PKG.glob(pattern):
            if path.is_file():
                out.add(path.relative_to(PKG).as_posix())
    return out


def test_package_data_globs_cover_every_asset_on_disk():
    """Whatever exists under the runtime asset dirs must be declared."""
    missing = sorted(_on_disk() - _covered())
    assert not missing, f"present on disk but not shipped in the wheel: {missing}"


def test_critical_assets_are_declared_individually():
    """A glob-level regression check that names the load-bearing files.

    The registry test above would still pass if an asset directory were emptied;
    this one fails loudly if a specific file the runtime needs disappears.
    """
    required = {
        "contracts/fidelity-calibration-v1.json",
        "registry/fidelity-registry-v1.json",
        "profiles/guard/guard-orcarouter-qwen3.8-27b-uncensored-exact-v1.yaml",
        "profiles/guard/guard-spark-x25-4b-abliterated-exact-v1.yaml",
    }
    covered = _covered()
    missing = sorted(required - covered)
    assert not missing, f"required runtime assets missing from package-data: {missing}"


def test_runtime_resolvers_point_inside_the_package():
    """The paths the CLI actually resolves must be real files under the package."""
    from fit_gguf.calibration import default_contract_path
    from fit_gguf.registry import default_guard_registry, find_package_dir

    package_dir = find_package_dir(None)
    assert package_dir == PKG.resolve()

    contract = default_contract_path()
    assert contract.is_file() and contract.parent == package_dir / "contracts"

    guards = sorted(default_guard_registry().glob("*.yaml"))
    assert guards, "no Guard Profiles resolved from the package directory"
