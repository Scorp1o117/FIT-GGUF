"""Loopback-only Studio server and optional pywebview desktop shell."""
from __future__ import annotations

from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
import json
import math
import mimetypes
import os
from pathlib import Path
import secrets
import sys
import threading
from urllib.parse import parse_qs, urlsplit
import webbrowser

from fit_gguf.studio.hardware import LlmfitAdapter, memory_budget
from fit_gguf.studio.jobs import JobManager, input_path, summarize_analysis
from fit_gguf.version import __version__


class StudioServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, workspace: Path, port: int = 0, llmfit: str | None = None,
                 max_model_params: float = 5):
        if not math.isfinite(max_model_params) or not 0 < max_model_params <= 1000:
            raise ValueError("Model parameter limit must be between 0 and 1000B")
        self.jobs = JobManager(workspace, max_model_params)
        self.adapter = LlmfitAdapter(llmfit)
        self.token = secrets.token_urlsafe(32)
        super().__init__(("127.0.0.1", port), Handler)
        self.origin = f"http://127.0.0.1:{self.server_port}"
        self.url = self.origin + "/?token=" + self.token

    def server_close(self):
        self.jobs.close()
        super().server_close()


class Handler(BaseHTTPRequestHandler):
    server: StudioServer

    def log_message(self, *_args):
        # Bootstrap URL contains a session capability; never write it to logs.
        pass

    def _send(self, status: int, body: bytes, content_type="application/json; charset=utf-8", headers=None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, payload):
        self._send(status, json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8"))

    def _authorized(self) -> bool:
        if self.headers.get("Host") != urlsplit(self.server.origin).netloc:
            self._json(403, {"error": "Invalid local host"})
            return False
        origin = self.headers.get("Origin")
        if origin and origin != self.server.origin:
            self._json(403, {"error": "Cross-origin access refused"})
            return False
        if self.headers.get("Sec-Fetch-Site") == "cross-site":
            self._json(403, {"error": "Cross-site access refused"})
            return False
        cookie = SimpleCookie()
        try:
            cookie.load(self.headers.get("Cookie", ""))
            value = cookie.get("fit_session")
            valid = value and secrets.compare_digest(value.value, self.server.token)
        except Exception:
            valid = False
        if not valid:
            self._json(401, {"error": "Open the session URL printed by fit gui to connect."})
            return False
        return True

    def do_GET(self):
        split = urlsplit(self.path)
        query = parse_qs(split.query)
        if split.path == "/" and "token" in query:
            if self.headers.get("Host") != urlsplit(self.server.origin).netloc or not secrets.compare_digest(query["token"][0], self.server.token):
                self._json(403, {"error": "Invalid session"})
                return
            self._send(303, b"", headers={"Location": "/", "Set-Cookie": f"fit_session={self.server.token}; HttpOnly; SameSite=Strict; Path=/"})
            return
        if not self._authorized():
            return
        try:
            if split.path == "/api/info":
                self._json(200, {"workspace": str(self.server.jobs.workspace),
                                 "version": __version__,
                                 "max_model_params": self.server.jobs.limit,
                                 "presets": list(__import__("fit_gguf.pipeline", fromlist=["PRESET_FILE_TYPES"]).PRESET_FILE_TYPES)})
            elif split.path == "/api/system":
                self._json(200, self.server.adapter.system())
            elif split.path == "/api/models":
                maximum = min(float(query.get("max_params", [str(self.server.jobs.limit)])[0]), self.server.jobs.limit)
                self._json(200, self.server.adapter.models(maximum, int(query.get("context", ["4096"])[0])))
            elif split.path == "/api/registry":
                from fit_gguf.registry import load_index, find_package_dir
                self._json(200, load_index(find_package_dir(None)))
            elif split.path == "/api/jobs":
                self._json(200, self.server.jobs.snapshot())
            elif split.path.startswith("/api/jobs/"):
                self._json(200, self.server.jobs.snapshot(split.path.removeprefix("/api/jobs/")))
            elif split.path in ("/", "/app.js", "/style.css"):
                name = "index.html" if split.path == "/" else split.path[1:]
                resource = files("fit_gguf.studio").joinpath("static", name)
                self._send(200, resource.read_bytes(), mimetypes.guess_type(name)[0] + "; charset=utf-8")
            else:
                self._json(404, {"error": "Not found"})
        except KeyError:
            self._json(404, {"error": "Record not found"})
        except Exception as error:
            self._json(400, {"error": str(error)})

    def do_POST(self):
        if not self._authorized():
            return
        try:
            if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                raise ValueError("JSON requests required")
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 65536:
                raise ValueError("Request must be between 1 and 65536 bytes")
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("Expected an object")
            path = urlsplit(self.path).path
            if path == "/api/jobs":
                self._json(202, self.server.jobs.submit(payload.get("action"), payload))
            elif path.startswith("/api/jobs/") and path.endswith("/cancel"):
                self._json(200, self.server.jobs.cancel(path.split("/")[3]))
            elif path == "/api/analysis":
                analysis = input_path(payload.get("path"))
                self._json(200, {"path": str(analysis), "result": summarize_analysis(analysis)})
            elif path == "/api/budget":
                system = self.server.adapter.system()["system"]
                self._json(200, memory_budget(system, float(payload.get("reserve_gb", 2)),
                                             float(payload.get("overhead_gb", 1)), payload.get("mode", "gpu")))
            else:
                self._json(404, {"error": "Not found"})
        except KeyError:
            self._json(400, {"error": "Missing record or field"})
        except Exception as error:
            self._json(400, {"error": str(error)})


class DesktopBridge:
    def __init__(self, jobs=None):
        self._window = None
        self._jobs = jobs

    def pick_path(self, kind="file"):
        import webview
        if kind not in ("file", "folder"):
            raise ValueError("Invalid picker kind")
        result = self._window.create_file_dialog(webview.FileDialog.FOLDER if kind == "folder" else webview.FileDialog.OPEN)
        return result[0] if result else None

    def open_output(self, identifier):
        if self._jobs is None:
            raise ValueError("No task workspace is attached")
        job = self._jobs.snapshot(identifier)
        if job["status"] != "succeeded":
            raise ValueError("Only completed task outputs can be opened")
        if job["action"] == "quality":
            result = job.get("result", {})
            if result.get("status") != "verified_pass":
                raise ValueError("This quality search has no verified output")
            artifact = (result.get("artifact") or {}).get("path")
        else:
            key = {"analyze": "analysis", "plan": "plan", "quantize": "model"}.get(job["action"])
            artifact = job["artifacts"].get(key)
        directory = input_path(artifact).parent
        if os.name == "nt":
            os.startfile(str(directory))
        else:
            from fit_gguf.llama_integration import run_runtime
            run_runtime(["open" if sys.platform == "darwin" else "xdg-open", str(directory)],
                        check=True, timeout=10)
        return str(directory)


def launch(args) -> int:
    server = StudioServer(Path(args.workspace), args.port, args.llmfit, args.max_model_params)
    if sys.stdout is not None:
        print(f"FIT Studio: {server.url}", flush=True)
        print(f"Workspace: {server.jobs.workspace}", flush=True)
        print("Use Ctrl+C to stop the local server.", flush=True)
    try:
        if args.command == "desktop" and not args.no_open:
            try:
                import webview
            except ImportError:
                print("Desktop window requires: pip install 'fit-gguf[desktop]'", flush=True)
                return 2
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            bridge = DesktopBridge(server.jobs)
            bridge._window = webview.create_window("FIT Studio", server.url, js_api=bridge,
                                                  width=1360, height=900, min_size=(960, 680))
            try:
                webview.start()
            finally:
                server.shutdown()
        else:
            if not args.no_open:
                threading.Timer(.3, lambda: webbrowser.open(server.url)).start()
            server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0
