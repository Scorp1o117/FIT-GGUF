"""Platform-aware llama.cpp binary resolution.

llama.cpp ships extensionless executables on POSIX and ``.exe`` on Windows, so a
hardcoded ``runtime_dir / "llama-quantize"`` made every runtime lookup fail on
Windows even with a perfectly good runtime directory. These tests pin the
candidate order for both platforms (via the ``_is_windows`` probe, so either
ordering is exercised from either host) and the resolver's behaviour over real
files.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fit_gguf import llama_integration as li
from fit_gguf.pipeline import PipelineError, run_dry_run

ALL_FORMS = ("llama-quantize", "llama-quantize.exe", "llama-quantize.cmd", "llama-quantize.bat")

# (simulated platform, file on disk) — every form a runtime directory may ship
RESOLVABLE = [
    (True, "llama-quantize.exe"),
    (True, "llama-quantize.cmd"),
    (True, "llama-quantize.bat"),
    (True, "llama-quantize"),
    (False, "llama-quantize"),
    (False, "llama-quantize.exe"),
]


@pytest.fixture()
def as_windows(monkeypatch):
    monkeypatch.setattr(li, "_is_windows", lambda: True)


@pytest.fixture()
def as_posix(monkeypatch):
    monkeypatch.setattr(li, "_is_windows", lambda: False)


def test_windows_order_prefers_exe_then_wrappers(as_windows):
    assert li.binary_candidate_names("llama-quantize") == (
        "llama-quantize.exe",
        "llama-quantize.cmd",
        "llama-quantize.bat",
        "llama-quantize",
    )


def test_posix_order_prefers_extensionless(as_posix):
    assert li.binary_candidate_names("llama-quantize") == (
        "llama-quantize",
        "llama-quantize.exe",
    )


@pytest.mark.parametrize(("windows", "filename"), RESOLVABLE)
def test_resolver_finds_every_supported_form(tmp_path, monkeypatch, windows, filename):
    """Every shipped form resolves under the platform that ships it."""
    monkeypatch.setattr(li, "_is_windows", lambda: windows)
    (tmp_path / filename).write_text("", encoding="utf-8")
    assert li.resolve_runtime_binary(tmp_path, "llama-quantize") == tmp_path / filename


def test_resolver_prefers_the_native_form_when_several_exist(tmp_path, as_windows):
    for filename in ALL_FORMS:
        (tmp_path / filename).write_text("", encoding="utf-8")
    assert li.resolve_runtime_binary(tmp_path, "llama-quantize").name == "llama-quantize.exe"


def test_resolver_prefers_extensionless_on_posix(tmp_path, as_posix):
    for filename in ALL_FORMS:
        (tmp_path / filename).write_text("", encoding="utf-8")
    assert li.resolve_runtime_binary(tmp_path, "llama-quantize").name == "llama-quantize"


def test_missing_binary_reports_every_candidate(tmp_path, as_windows):
    message = li.binary_not_found_message(tmp_path, "llama-quantize")
    assert str(tmp_path) in message
    for candidate in li.binary_candidate_names("llama-quantize"):
        assert candidate in message
    # no candidate exists: the preferred name comes back so the caller can tell
    # "absent runtime" from "resolution bug"
    assert li.resolve_runtime_binary(tmp_path, "llama-quantize").name == "llama-quantize.exe"


def test_dry_run_surfaces_a_missing_runtime(tmp_path):
    """The product path must fail with the candidate list, not a bare path."""
    with pytest.raises(PipelineError, match="llama-quantize not found in"):
        run_dry_run(tmp_path, "src.gguf", "imx.gguf", "IQ3_M", tmp_path / "dry.log")
