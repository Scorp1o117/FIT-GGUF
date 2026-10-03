"""Run the real Studio pipeline on a supplied <=5B model (no downloads)."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
from fit_gguf.studio.jobs import JobManager


def wait(manager, job):
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        result = manager.snapshot(job["id"])
        if result["status"] not in ("running", "queued", "cancelling"):
            if result["status"] != "succeeded":
                raise RuntimeError(result.get("log_text", str(result)))
            return result
        time.sleep(.1)
    manager.cancel(job["id"])
    raise TimeoutError("Smoke task exceeded five minutes")


def main():
    parser = argparse.ArgumentParser()
    for key in ("source", "imatrix", "runtime", "workspace"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--lower", default="Q4_K_M")
    parser.add_argument("--upper", default="Q8_0")
    parser.add_argument("--worker-executable", type=Path, help="Test a frozen FIT-Studio executable worker")
    args = parser.parse_args()
    manager = JobManager(Path(args.workspace), 5, args.worker_executable)
    try:
        analyzed = wait(manager, manager.submit("analyze", {
            key: str(Path(getattr(args, key)).resolve()) for key in ("source", "imatrix", "runtime")
        } | {"lower": args.lower, "upper": args.upper}))
        analysis = analyzed["artifacts"]["analysis"]
        presets = analyzed["result"]["presets"]
        target = (presets["lower"]["predicted_size_bytes"] + presets["upper"]["predicted_size_bytes"]) // 2
        planned = wait(manager, manager.submit("plan", {"analysis": analysis, "target_bytes": target}))
        quantized = wait(manager, manager.submit("quantize", {"analysis": analysis, "plan": planned["artifacts"]["plan"]}))
        report = {"analyze_job": analyzed["id"], "plan_job": planned["id"], "quantize_job": quantized["id"],
                  "target_bytes": target, "predicted_bytes": planned["result"]["predicted_size_bytes"],
                  "actual_bytes": quantized["result"]["size_bytes"], "result": quantized["result"],
                  "artifacts": quantized["artifacts"]}
        (manager.workspace / "smoke-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
    finally:
        manager.close()


if __name__ == "__main__":
    main()
