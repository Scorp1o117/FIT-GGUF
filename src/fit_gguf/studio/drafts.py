"""Workspace-scoped form drafts; saved values never execute a task."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import threading


FORM_FIELDS = {
    "analyzeForm": {"source", "imatrix", "runtime", "lower", "upper"},
    "planForm": {"analysis", "target_bytes", "policy"},
    "quantizeForm": {"plan"},
    "qualityForm": {"source", "imatrix", "runtime", "tier", "refs_dir",
                    "eval_data_dir", "freeze", "reference_manifest", "threads"},
}
ALLOWED_FIELDS = {f"{form}.{field}" for form, fields in FORM_FIELDS.items() for field in fields}
ALLOWED_FIELDS.update({"desiredGiB", "reserve", "overhead", "runMode", "maxParams", "context"})


def validate_draft(payload):
    if not isinstance(payload, dict) or set(payload) != {"fields"}:
        raise ValueError("Draft must contain only fields")
    fields = payload["fields"]
    if not isinstance(fields, dict) or set(fields) - ALLOWED_FIELDS:
        raise ValueError("Unknown draft field")
    if any(not isinstance(value, str) or len(value) > 4096 for value in fields.values()):
        raise ValueError("Draft values must be strings of at most 4096 characters")
    if len(json.dumps(payload, ensure_ascii=False).encode("utf-8")) > 60000:
        raise ValueError("Draft is too large")
    return {"fields": dict(fields)}


class DraftStore:
    def __init__(self, workspace: Path):
        self.path = workspace / "studio-draft.json"
        self.lock = threading.Lock()

    def load(self):
        with self.lock:
            if not self.path.exists():
                return None
            if self.path.stat().st_size > 60000:
                raise ValueError("Saved draft is too large")
            return validate_draft(json.loads(self.path.read_text(encoding="utf-8")))

    def save(self, payload):
        draft = validate_draft(payload)
        with self.lock:
            # A unique sibling temporary file keeps replacement atomic on Windows.
            descriptor, temporary = tempfile.mkstemp(prefix=".studio-draft-", dir=self.path.parent)
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as output:
                    json.dump(draft, output, ensure_ascii=False)
                os.replace(temporary, self.path)
            finally:
                Path(temporary).unlink(missing_ok=True)
        return draft

    def clear(self):
        with self.lock:
            self.path.unlink(missing_ok=True)
