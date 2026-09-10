"""Hot-loop filesystem guard: never let a subprocess write logs onto ntfs3.

``calibrate._run`` hands the subprocess's stdout/stderr file descriptor straight
to llama.cpp. llama.cpp logs to stderr unbuffered, so on a filesystem whose
buffered-write path is buggy that becomes thousands of short, unaligned,
page-spanning writes. On ntfs3 they trip
``kernel BUG at fs/iomap/buffered-io.c:1061`` in ``iomap_write_end`` and panic
the machine — observed 2026-09-06 (twice) and 2026-09-10, every time with
``llama-quantize`` or ``llama-perplexity`` as the writing process.

``pipeline.py`` never hit this because it captures subprocess output with
``capture_output=True`` and writes the log itself in one bulk write. These tests
pin the guard that keeps ``calibrate`` out of the same ditch.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from fit_gguf import calibrate
from fit_gguf.calibration import CalibrationError


@pytest.fixture()
def unsafe_fs(monkeypatch):
    """Pretend every path lives on ntfs3."""
    monkeypatch.setattr(calibrate, "mount_fs_type", lambda path: "ntfs3")


@pytest.fixture()
def safe_fs(monkeypatch):
    monkeypatch.setattr(calibrate, "mount_fs_type", lambda path: "tmpfs")


@pytest.fixture(autouse=True)
def _clear_override(monkeypatch):
    monkeypatch.delenv("FIT_ALLOW_UNSAFE_FS", raising=False)


# --------------------------------------------------------------------- guard


def test_guard_refuses_ntfs3(unsafe_fs, tmp_path):
    with pytest.raises(CalibrationError) as excinfo:
        calibrate.assert_hot_loop_fs_safe({"logs": tmp_path / "logs"})
    message = str(excinfo.value)
    assert "HOT_LOOP_FS_UNSAFE" in message
    assert "ntfs3" in message
    assert "FIT_ALLOW_UNSAFE_FS" in message


def test_guard_allows_tmpfs(safe_fs, tmp_path):
    calibrate.assert_hot_loop_fs_safe(
        {"logs": tmp_path / "logs", "references": tmp_path / "refs"}
    )


def test_guard_override_is_honoured(unsafe_fs, monkeypatch, tmp_path):
    monkeypatch.setenv("FIT_ALLOW_UNSAFE_FS", "1")
    calibrate.assert_hot_loop_fs_safe({"logs": tmp_path / "logs"})


def test_mount_fs_type_resolves_real_mounts():
    """Not host-specific: /dev/shm is a tmpfs on any Linux box; skip if absent."""
    shm = Path("/dev/shm")
    if not shm.is_dir():
        pytest.skip("/dev/shm not present")
    assert calibrate.mount_fs_type(shm) == "tmpfs"


def test_run_calibrate_refuses_before_touching_anything(unsafe_fs, tmp_path):
    """The guard must fire on the default (scratch) layout too, and early.

    No source/eval data exists here: reaching any stage would raise a different
    error, so a HOT_LOOP_FS_UNSAFE failure proves the guard runs first.
    """
    cfg = calibrate.CalibrateConfig(
        source=tmp_path / "missing-BF16.gguf",
        imatrix_corpus=tmp_path / "missing-corpus.txt",
        runtime_dir=tmp_path / "runtime",
        eval_data_dir=tmp_path / "eval-data",
        out_dir=tmp_path / "out",
        model_id="unit-test-model",
        workdir=tmp_path / "scratch",
    )
    with pytest.raises(CalibrationError, match="HOT_LOOP_FS_UNSAFE"):
        calibrate.run_calibrate(cfg)


# ---------------------------------------------------------------- publishing


def test_publish_tree_copies_then_skips_identical(tmp_path):
    src = tmp_path / "scratch"
    dst = tmp_path / "bundle"
    src.mkdir()
    (src / "a.log").write_text("alpha")
    (src / "b.log").write_text("beta")
    (src / "ignored.json").write_text("{}")

    assert sorted(calibrate.publish_tree(src, dst, "*.log")) == ["a.log", "b.log"]
    assert (dst / "ignored.json").exists() is False

    # Second pass: same sizes, nothing re-copied.
    assert calibrate.publish_tree(src, dst, "*.log") == []

    # A grown file is re-copied.
    (src / "a.log").write_text("alpha-and-more")
    assert calibrate.publish_tree(src, dst, "*.log") == ["a.log"]
    assert (dst / "a.log").read_text() == "alpha-and-more"


def test_publish_tree_missing_source_is_a_noop(tmp_path):
    assert calibrate.publish_tree(tmp_path / "nope", tmp_path / "dst", "*.log") == []
    assert not (tmp_path / "dst").exists()


# --------------------------------------------------------------- log routing


def test_calibrate_config_defaults_log_dir_to_none():
    """``None`` means "inside the scratch volume"; the CLI must not pre-fill it."""
    import argparse
    import inspect

    from fit_gguf import cli

    source = inspect.getsource(cli._build_parser)
    assert '"--log-dir", default=None' in source
    assert 'log_dir=Path(args.log_dir) if args.log_dir else None' in inspect.getsource(
        cli._run
    )


def test_run_calibrate_resolves_log_dir_into_scratch(safe_fs, tmp_path, monkeypatch):
    """With log_dir unset, logs must land in the scratch volume, not the bundle."""
    seen: dict[str, Path] = {}

    def fake_stage_generate_imatrix(cfg, env):
        seen["log_dir"] = cfg.log_dir
        seen["workdir"] = cfg.workdir
        raise RuntimeError("stop here")

    monkeypatch.setattr(calibrate, "stage_generate_imatrix", fake_stage_generate_imatrix)
    monkeypatch.setattr(calibrate, "sha256_file", lambda path: "0" * 64)
    monkeypatch.setattr(calibrate, "stage_imatrix_coverage", lambda cfg, imx: [])

    cfg = calibrate.CalibrateConfig(
        source=tmp_path / "src.gguf",
        imatrix_corpus=tmp_path / "corpus.txt",
        runtime_dir=tmp_path / "runtime",
        eval_data_dir=tmp_path / "eval-data",
        out_dir=tmp_path / "bundle",
        model_id="unit-test-model",
        workdir=tmp_path / "scratch",
    )
    with pytest.raises(RuntimeError, match="stop here"):
        calibrate.run_calibrate(cfg)

    assert seen["log_dir"] == tmp_path / "scratch" / "logs"
    assert seen["log_dir"] != cfg.out_dir / "logs"
    assert seen["log_dir"].is_dir()
