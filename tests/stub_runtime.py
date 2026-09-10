"""Portable fake llama.cpp runtime for the test suite.

POSIX ships extensionless executables; Windows can only launch a real PE image
or a ``.cmd``/``.bat`` (CreateProcess runs those itself, without a shell).
Generating a ``.cmd`` launcher on Windows and a ``sh`` script elsewhere keeps
the E2E tests exercising a genuine subprocess on both platforms, and both
launchers exec *this* file so the fake behaviour lives in exactly one place.

Environment contract (set by the tests, read here):

``STUB_DIR``        directory holding the per-preset ``<preset>.dryrun.log`` files
``STUB_CALL_LOG``   append every invocation's argv here
``STUB_ORACLE_LOG`` when set, ``--dry-run --tensor-type-file`` prints its contents
``STUB_OUT_BYTES``  byte size a fake quantization must produce
``STUB_KL_LOG``     literal KL-divergence text a fake ``llama-perplexity`` prints
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

DEFAULT_KL_LOG = (
    "====== KL divergence statistics ======\n"
    "Mean KLD: 0.090000 ± 0.010000\n"
    "Same top p: 92.0000 ± 0.1000 %\n"
)


def write_fake_binary(directory: str | Path, name: str, kind: str) -> Path:
    """Create a launchable fake ``name`` (``kind`` = ``quantize``/``perplexity``).

    Returns the launcher path, which is the name the production resolver is
    expected to pick up for the host platform.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    runner = Path(__file__).resolve()
    python = sys.executable
    if os.name == "nt":
        launcher = directory / f"{name}.cmd"
        launcher.write_text(
            "@echo off\n"
            f'"{python}" "{runner}" --kind {kind} %*\n'
            "exit /b %ERRORLEVEL%\n",
            encoding="ascii",
            newline="\r\n",
        )
    else:
        launcher = directory / name
        launcher.write_text(
            "#!/bin/sh\n"
            f'exec "{python}" "{runner}" --kind {kind} "$@"\n',
            encoding="utf-8",
        )
        launcher.chmod(0o755)
    return launcher


def _emit(text: str) -> None:
    """Write UTF-8 bytes regardless of the host's ANSI code page.

    The fake KL log contains non-ASCII (the ``±`` separator). A redirected
    child on a non-UTF-8 Windows locale — cp936, cp932 — would otherwise encode
    it in the local code page, and the parent decodes the captured output as
    UTF-8, so the parse would fail for reasons that have nothing to do with the
    code under test.
    """
    stream = getattr(sys.stdout, "buffer", None)
    if stream is None:
        sys.stdout.write(text)
    else:
        stream.write(text.encode("utf-8"))
    sys.stdout.flush()


def _record(argv: list[str]) -> None:
    log = os.environ.get("STUB_CALL_LOG")
    if not log:
        return
    with open(log, "a", encoding="utf-8") as handle:
        handle.write("stub: " + " ".join(argv) + "\n")


def _fake_quantize(argv: list[str]) -> int:
    _record(argv)
    if argv[:1] == ["--dry-run"]:
        oracle = os.environ.get("STUB_ORACLE_LOG")
        if oracle and "--tensor-type-file" in argv:
            _emit(Path(oracle).read_text(encoding="utf-8"))
            return 0
        directory = Path(os.environ["STUB_DIR"])
        preset = argv[-1].lower()
        _emit((directory / f"{preset}.dryrun.log").read_text(encoding="utf-8"))
        return 0

    # Real invocation shape (pipeline.quantize): <...> <source> <output> <preset>
    output = Path(argv[-2])
    size = int(os.environ["STUB_OUT_BYTES"])
    with output.open("wb") as handle:
        handle.truncate(size)
    _emit(f"stub quantized {output}\n")
    return 0


def _fake_perplexity(argv: list[str]) -> int:
    _record(argv)
    _emit(os.environ.get("STUB_KL_LOG", DEFAULT_KL_LOG))
    return 0


def main(argv: list[str]) -> int:
    kind = argv[1] if argv[:1] == ["--kind"] else ""
    rest = argv[2:]
    if kind == "quantize":
        return _fake_quantize(rest)
    if kind == "perplexity":
        return _fake_perplexity(rest)
    sys.stderr.write(f"stub_runtime: unknown kind {kind!r}\n")
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
