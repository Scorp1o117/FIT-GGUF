"""Frozen desktop entry point, including its own CLI subprocess worker."""
from __future__ import annotations
import os
from pathlib import Path
import sys


def main() -> int:
    if sys.argv[1:2] == ["--fit-worker"]:
        if getattr(sys, "frozen", False):
            log = os.environ.get("FIT_STUDIO_WORKER_LOG")
            if not log:
                return 2
            sys.stdout = open(log, "a", encoding="utf-8", buffering=1)
            sys.stderr = sys.stdout
        from fit_gguf.cli import main as cli_main
        return cli_main(sys.argv[2:])
    from fit_gguf.cli import main as cli_main
    workspace = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "FIT-Studio" / "workspace"
    return cli_main(["desktop", "--workspace", str(workspace), *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main())
