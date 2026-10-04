"""Studio boundary tests: hardware estimates, local auth, subprocess pipeline."""
from __future__ import annotations

from http.cookiejar import CookieJar
import json
from pathlib import Path
import threading
import time
from urllib.error import HTTPError
from urllib.request import build_opener, HTTPCookieProcessor, Request, urlopen

import pytest

from fit_gguf.studio.hardware import LlmfitAdapter, memory_budget, parameter_billions
from fit_gguf.studio.jobs import JobManager
from fit_gguf.studio.server import StudioServer, DesktopBridge
from test_pipeline import e2e  # reuse the pinned GGUF + quantizer fixture


def test_gpu_budget_uses_free_memory_and_single_largest_card():
    system = {"available_ram_gb": 30, "gpus": [
        {"vram_gb": 16, "free_vram_gb": 12}, {"vram_gb": 24, "free_vram_gb": 6}]}
    result = memory_budget(system)
    assert result["target_bytes"] == 9 * 1024**3
    assert memory_budget(system, mode="cpu")["target_bytes"] == 27 * 1024**3


def test_unified_memory_is_one_pool():
    system = {"unified_memory": True, "available_ram_gb": 8,
              "gpus": [{"free_vram_gb": 16}]}
    assert memory_budget(system)["target_bytes"] == 5 * 1024**3


def test_unknown_free_vram_does_not_assume_total():
    assert memory_budget({"gpus": [{"vram_gb": 16}]})["target_bytes"] is None


@pytest.mark.parametrize("reserve", [-1, float("inf"), float("nan")])
def test_invalid_budget_refused(reserve):
    with pytest.raises(ValueError):
        memory_budget({}, reserve)


@pytest.mark.parametrize(("text", "expected"), [("0.6B", .6), ("350M", .35),
                         ("30B-A3B", 30), ("1T", 1000), ("unknown", None)])
def test_parameter_filter_uses_total_moe_params(text, expected):
    assert parameter_billions(text) == (pytest.approx(expected) if expected is not None else None)


def test_adapter_filters_unknown_large_and_non_gguf_models(monkeypatch):
    adapter = LlmfitAdapter()
    adapter.executable = "llmfit"
    monkeypatch.setattr("fit_gguf.studio.hardware.run_json", lambda *_a, **_k: {"models": [
        {"name": "small", "parameter_count": "0.6B", "runtime": "llama.cpp"},
        {"name": "moe", "parameter_count": "30B-A3B", "runtime": "llama.cpp"},
        {"name": "unknown", "parameter_count": "?"},
        {"name": "provider/model-8B-NVFP4", "parameter_count": "4.7B", "params_b": 4.7, "runtime": "llama.cpp"},
        {"name": "awq", "parameter_count": "3B", "runtime": "vllm"}]})
    assert [row["name"] for row in adapter.models()["models"]] == ["small"]


def test_broken_llmfit_reports_fallback_instead_of_connected(monkeypatch):
    adapter = LlmfitAdapter()
    adapter.executable = "broken-llmfit"
    local = {"available_ram_gb": 8, "gpus": []}
    monkeypatch.setattr("fit_gguf.studio.hardware.live_system", lambda: local)
    def failed(*_args, **_kwargs):
        raise ValueError("unsupported system JSON")
    monkeypatch.setattr("fit_gguf.studio.hardware.run_json", failed)
    result = adapter.system()
    assert result["system"] == local
    assert not result["llmfit_available"]
    assert result["provider"] == "local"
    assert result["warnings"]


@pytest.fixture()
def server(tmp_path, monkeypatch):
    instance = StudioServer(tmp_path / "studio")
    instance.adapter.executable = None
    monkeypatch.setattr(instance.adapter, "system", lambda: {
        "system": {"available_ram_gb": 8, "gpus": [{"free_vram_gb": 12}]},
        "provider": "test", "llmfit_available": False})
    worker = threading.Thread(target=instance.serve_forever, daemon=True)
    worker.start()
    yield instance
    instance.shutdown()
    instance.server_close()
    worker.join(timeout=2)


def authorized_client(server):
    client = build_opener(HTTPCookieProcessor(CookieJar()))
    response = client.open(server.url)
    assert response.geturl() == server.origin + "/"
    return client


def test_local_server_auth_host_origin_and_content_security(server):
    with pytest.raises(HTTPError) as missing:
        urlopen(server.origin + "/api/info")
    assert missing.value.code == 401
    client = authorized_client(server)
    response = client.open(server.origin + "/")
    assert "frame-ancestors 'none'" in response.headers["Content-Security-Policy"]
    assert b"FIT Studio" in response.read()
    with pytest.raises(HTTPError) as origin:
        client.open(Request(server.origin + "/api/jobs", data=b'{}', headers={
            "Origin": "https://evil.example", "Content-Type": "application/json"}))
    assert origin.value.code == 403
    with pytest.raises(HTTPError) as host:
        client.open(Request(server.origin + "/api/info", headers={"Host": "evil.example"}))
    assert host.value.code == 403
    assert json.loads(client.open(server.origin + "/api/jobs").read()) == []


def test_budget_and_static_assets(server):
    client = authorized_client(server)
    result = client.open(Request(server.origin + "/api/budget", data=b'{"mode":"gpu"}',
                                headers={"Content-Type": "application/json"}))
    assert json.loads(result.read())["target_bytes"] == 9 * 1024**3
    for name in ("app.js", "i18n.js", "style.css"):
        assert len(client.open(server.origin + "/" + name).read()) > 1000
    with pytest.raises(HTTPError):
        client.open(server.origin + "/../../pyproject.toml")


def wait_job(manager, job):
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        result = manager.snapshot(job["id"])
        if result["status"] not in ("running", "queued", "cancelling"):
            assert result["status"] == "succeeded", result
            return result
        time.sleep(.05)
    pytest.fail("Job timed out")


def test_studio_executes_analyze_plan_quantize_and_preserves_history(e2e, monkeypatch):
    manager = JobManager(e2e["tmp"] / "studio")
    analyzed = wait_job(manager, manager.submit("analyze", {
        "source": str(e2e["source"]), "imatrix": str(e2e["imatrix"]),
        "runtime": str(e2e["runtime"])}))
    analysis = analyzed["artifacts"]["analysis"]
    lower = analyzed["result"]["presets"]["lower"]["predicted_size_bytes"]
    upper = analyzed["result"]["presets"]["upper"]["predicted_size_bytes"]
    model_context = {key: str(e2e[key]) for key in ("source", "imatrix", "runtime")}
    model_context.update(lower="IQ3_M", upper="IQ4_XS")
    with pytest.raises(ValueError, match="presets differ"):
        manager.submit("plan", {"analysis": analysis, "target_bytes": (lower + upper) // 2,
                               "model_context": model_context | {"lower": "Q8_0"}})
    planned = wait_job(manager, manager.submit("plan", {
        "analysis": analysis, "target_bytes": (lower + upper) // 2, "model_context": model_context}))
    with pytest.raises(ValueError, match="budget differs"):
        manager.submit("quantize", {"analysis": analysis, "plan": planned["artifacts"]["plan"],
            "model_context": model_context, "plan_context": {"target_bytes": 1, "policy": "balanced"}})
    monkeypatch.setenv("STUB_OUT_BYTES", str(planned["result"]["predicted_size_bytes"]))
    quantized = wait_job(manager, manager.submit("quantize", {
        "analysis": analysis, "plan": planned["artifacts"]["plan"], "model_context": model_context,
        "plan_context": {"target_bytes": (lower + upper) // 2, "policy": "balanced"}}))
    assert quantized["result"]["size_matches_expectation"]
    assert quantized["result"]["size_matches_refinalization"]
    assert len(quantized["result"]["sha256"]) == 64
    restored = JobManager(manager.workspace)
    assert len(restored.snapshot()) == 3
    assert restored.snapshot(quantized["id"])["result"]["sha256"] == quantized["result"]["sha256"]
    # A changed tensor file must be refused before creating a quantization job.
    Path(planned["artifacts"]["tensor_types"]).write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="input changed"):
        manager.submit("quantize", {"analysis": analysis, "plan": planned["artifacts"]["plan"]})


def test_source_limit_and_invalid_tasks_refused(e2e):
    manager = JobManager(e2e["tmp"] / "studio", max_model_params=.00001)
    with pytest.raises(ValueError, match="limited"):
        manager.submit("analyze", {"source": str(e2e["source"]),
                                  "imatrix": str(e2e["imatrix"]), "runtime": str(e2e["runtime"])})
    with pytest.raises(ValueError, match="Unsupported"):
        manager.submit("arbitrary-command", {})
    assert manager.snapshot() == []


def test_one_task_at_a_time_and_interrupted_recovery(tmp_path):
    manager = JobManager(tmp_path)
    manager.active = "occupied"
    with pytest.raises(ValueError, match="already running"):
        manager.submit("analyze", {})
    folder = tmp_path / "runs" / "test"
    folder.mkdir(parents=True)
    (folder / "job.json").write_text(json.dumps({"id": "test", "status": "running"}), encoding="utf-8")
    assert JobManager(tmp_path).snapshot()[0]["status"] == "interrupted"
    assert json.loads((folder / "job.json").read_text())["status"] == "interrupted"


def test_reopened_history_orders_by_creation_time_not_random_uuid(tmp_path):
    for identifier, created in [("zzzz", "2026-10-03T01:00:00+00:00"), ("aaaa", "2026-10-03T02:00:00+00:00")]:
        folder = tmp_path / "runs" / identifier
        folder.mkdir(parents=True)
        (folder / "job.json").write_text(json.dumps({"id": identifier, "status": "succeeded", "created_at": created}))
    assert [row["id"] for row in JobManager(tmp_path).snapshot()] == ["aaaa", "zzzz"]


def test_desktop_bridge_keeps_native_window_private(monkeypatch):
    import sys
    from types import SimpleNamespace
    monkeypatch.setitem(sys.modules, "webview", SimpleNamespace(FileDialog=SimpleNamespace(FOLDER=1, OPEN=2)))
    bridge = DesktopBridge()
    calls = []
    bridge._window = SimpleNamespace(create_file_dialog=lambda kind: calls.append(kind) or ("C:/模型/test.gguf",))
    assert [name for name in dir(bridge) if not name.startswith("_")] == ["open_output", "pick_path"]
    assert bridge.pick_path("file") == "C:/模型/test.gguf"
    assert bridge.pick_path("folder") == "C:/模型/test.gguf"
    assert calls == [2, 1]


def test_cancel_terminates_running_subprocess(tmp_path, monkeypatch):
    import sys
    manager = JobManager(tmp_path)
    monkeypatch.setattr(manager, "_command", lambda _action, _payload, _folder: (
        [sys.executable, "-u", "-c", "import time;print('running',flush=True);time.sleep(60)"], {}))
    job = manager.submit("analyze", {})
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and manager.snapshot(job["id"])["status"] == "queued":
        time.sleep(.02)
    assert manager.snapshot(job["id"])["status"] == "running"
    process = manager.process
    manager.cancel(job["id"])
    while time.monotonic() < deadline and manager.active:
        time.sleep(.02)
    assert manager.snapshot(job["id"])["status"] == "cancelled"
    assert process.poll() is not None


def test_desktop_output_folder_uses_completed_job_not_an_arbitrary_path(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from fit_gguf.studio import server as module
    manager = JobManager(tmp_path / "studio")
    artifact = manager.workspace / "model.gguf"
    artifact.write_bytes(b"GGUF")
    manager.jobs["ready"] = {"id": "ready", "action": "quantize", "status": "succeeded",
                              "log": str(tmp_path / "log"), "artifacts": {"model": str(artifact)}}
    opened = []
    monkeypatch.setattr(module, "os", SimpleNamespace(name="nt", startfile=opened.append))
    bridge = module.DesktopBridge(manager)
    assert bridge.open_output("ready") == str(artifact.parent)
    assert opened == [str(artifact.parent)]
    with pytest.raises(KeyError):
        bridge.open_output(str(tmp_path))
    manager.jobs["ready"]["status"] = "cancelled"
    with pytest.raises(ValueError, match="completed"):
        bridge.open_output("ready")
    assert len(opened) == 1


def test_desktop_refuses_unverified_quality_output(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from fit_gguf.studio import server as module
    manager = JobManager(tmp_path)
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"status": "no_pass", "artifact": None}))
    manager.jobs["quality"] = {"id": "quality", "action": "quality", "status": "succeeded",
                                "log": str(tmp_path / "log"), "artifacts": {"quality": str(report)}}
    opened = []
    monkeypatch.setattr(module, "os", SimpleNamespace(name="nt", startfile=opened.append))
    with pytest.raises(ValueError, match="verified"):
        module.DesktopBridge(manager).open_output("quality")
    assert not opened


def test_quality_entry_routes_to_real_search_and_refuses_unverified_reference(e2e):
    manager = JobManager(e2e["tmp"] / "quality-studio")
    refs = e2e["tmp"] / "references"
    refs.mkdir()
    freeze = e2e["tmp"] / "FREEZE.json"
    freeze.write_text('{}')
    manifest = e2e["tmp"] / "reference-manifest.json"
    manifest.write_text('{}')
    payload = {"source": str(e2e["source"]), "imatrix": str(e2e["imatrix"]),
               "runtime": str(e2e["runtime"]), "refs_dir": str(refs),
               "eval_data_dir": str(refs), "freeze": str(freeze),
               "reference_manifest": str(manifest), "tier": "balanced", "threads": 4}
    with pytest.raises(ValueError, match="quality tier"):
        manager.submit("quality", payload | {"tier": "unknown"})
    with pytest.raises(ValueError, match="threads"):
        manager.submit("quality", payload | {"threads": 32})
    job = manager.submit("quality", payload)
    command = job["command"]
    assert "fidelity-search" in command and "quality" not in command
    assert command[command.index("--n-gpu-layers") + 1] == "0"
    assert command[command.index("--refs-dir") + 1] == str(refs.resolve())
    assert str(manager.workspace) in command[command.index("--out-dir") + 1]
    deadline = time.monotonic() + 10
    while manager.active and time.monotonic() < deadline:
        time.sleep(.02)
    result = manager.snapshot(job["id"])
    assert result["status"] == "failed"
    assert result["exit_code"] != 0
    assert "eval-v1" in result["log_text"] or "frozen" in result["log_text"].lower()
    assert not list(manager.workspace.rglob("*.gguf"))


def test_failed_quality_report_is_visible_without_claiming_success(tmp_path):
    manager = JobManager(tmp_path)
    report = tmp_path / "quality.json"
    report.write_text(json.dumps({"status": "no_pass", "artifact": None, "fresh_evals": 8}))
    manager.jobs["failed"] = {"id": "failed", "action": "quality", "status": "failed",
                              "log": str(tmp_path / "log"), "artifacts": {"quality": str(report)}}
    snapshot = manager.snapshot("failed")
    assert snapshot["status"] == "failed"
    assert snapshot["result"]["status"] == "no_pass"
    assert snapshot["result"]["artifact"] is None

