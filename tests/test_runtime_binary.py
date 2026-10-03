"""Platform-aware llama.cpp binary resolution.

llama.cpp ships extensionless executables on POSIX and ``.exe`` on Windows, so a
hardcoded ``runtime_dir / "llama-quantize"`` made every runtime lookup fail on
Windows even with a perfectly good runtime directory. These tests pin the
candidate order for both platforms (via the ``_is_windows`` probe, so either
ordering is exercised from either host) and the resolver's behaviour over real
files.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

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


# ------------------------------------------------- runtime library discovery


def _runtime_tree(tmp_path, *, cudart: bool):
    """A llama.cpp Windows-style layout: bin/ with a sibling cudart-*/ dir."""
    root = tmp_path / "llamacpp"
    binaries = root / "llama-b10690-bin-win-cuda-13.3-x64"
    binaries.mkdir(parents=True)
    (binaries / "ggml-cuda.dll").write_bytes(b"")
    if cudart:
        runtime = root / "cudart-llama-bin-win-cuda-13.3-x64"
        runtime.mkdir()
        (runtime / "cudart64_13.dll").write_bytes(b"")
        (runtime / "cublas64_13.dll").write_bytes(b"")
    return binaries


def test_cuda_runtime_sibling_is_found(tmp_path):
    """The CUDA runtime the GPU backend links against lives beside the binaries.

    Missing it is not an error: ggml-cuda.dll fails to load and llama.cpp
    quietly evaluates on CPU. Measured on a 2.5B BF16 model, pp512: 150 t/s on
    CPU versus 11,779 t/s on CUDA — a silent 78x, so the sibling must be
    discovered rather than left to the user's PATH.
    """
    binaries = _runtime_tree(tmp_path, cudart=True)
    siblings = li.cuda_runtime_siblings(binaries)
    assert [p.name for p in siblings] == ["cudart-llama-bin-win-cuda-13.3-x64"]


def test_no_cudart_sibling_is_not_an_error(tmp_path):
    """A CPU-only runtime has no sibling, and that is fine."""
    binaries = _runtime_tree(tmp_path, cudart=False)
    assert li.cuda_runtime_siblings(binaries) == []


def test_runtime_env_puts_bin_and_cudart_on_the_loader_path(tmp_path, as_windows):
    binaries = _runtime_tree(tmp_path, cudart=True)
    env = li.runtime_env(binaries, base={"PATH": "C:\\existing"})
    parts = env["PATH"].split(";")
    assert parts[0] == str(binaries)
    assert parts[1] == str(binaries.parent / "cudart-llama-bin-win-cuda-13.3-x64")
    # the caller's own search path is preserved, not replaced
    assert parts[-1] == "C:\\existing"


def test_runtime_env_uses_ld_library_path_on_posix(tmp_path, monkeypatch):
    monkeypatch.setattr(li, "_is_windows", lambda: False)
    binaries = _runtime_tree(tmp_path, cudart=True)
    env = li.runtime_env(binaries, base={"LD_LIBRARY_PATH": "/usr/lib"})
    assert env["LD_LIBRARY_PATH"].startswith(str(binaries))
    assert env["LD_LIBRARY_PATH"].endswith("/usr/lib")
    assert "PATH" not in env


def test_external_runtime_has_no_console_on_windows(monkeypatch):
    calls = []
    monkeypatch.setattr(li, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(li.subprocess, "CREATE_NO_WINDOW", 0x08000000, raising=False)
    monkeypatch.setattr(li.subprocess, "run", lambda command, **kwargs: calls.append(kwargs))
    li.run_runtime(["llama-quantize.exe"], capture_output=True)
    assert calls[0]["creationflags"] & 0x08000000
    assert calls[0]["capture_output"] is True


def test_frozen_runtime_restores_dll_directory_after_failure(monkeypatch):
    import ctypes
    changes = []

    def get_directory(length, buffer):
        buffer.value = r"C:\\FIT-Studio\\_internal"
        return len(buffer.value)

    kernel = SimpleNamespace(GetDllDirectoryW=get_directory, SetDllDirectoryW=changes.append)
    monkeypatch.setattr(li, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(li.sys, "frozen", True, raising=False)
    monkeypatch.setattr(ctypes, "windll", SimpleNamespace(kernel32=kernel), raising=False)
    monkeypatch.setattr(li.subprocess, "CREATE_NO_WINDOW", 0x08000000, raising=False)

    def fail(*args, **kwargs):
        raise OSError("runtime failed")

    monkeypatch.setattr(li.subprocess, "run", fail)
    with pytest.raises(OSError, match="runtime failed"):
        li.run_runtime(["llama-quantize.exe"])
    assert changes == [None, r"C:\\FIT-Studio\\_internal"]
