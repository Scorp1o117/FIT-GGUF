"""Draft persistence is independent from task submission and server ports."""
import json
from concurrent.futures import ThreadPoolExecutor
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from fit_gguf.studio.drafts import DraftStore
from test_studio import server, authorized_client


def post(client, server, payload):
    return json.loads(client.open(Request(server.origin + "/api/draft",
        data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})).read())


def test_draft_api_round_trip_clear_and_no_task(server):
    with pytest.raises(HTTPError) as missing:
        urlopen(server.origin + "/api/draft")
    assert missing.value.code == 401
    client = authorized_client(server)
    draft = {"fields": {"analyzeForm.source": "D:\\模型\\a.gguf",
                        "desiredGiB": "0.023456789", "qualityForm.tier": "Balanced"}}
    assert post(client, server, draft) == {"draft": draft}
    assert json.loads(client.open(server.origin + "/api/draft").read()) == {"draft": draft}
    # A fresh store (and a new randomly assigned server port) uses the workspace file.
    assert DraftStore(server.jobs.workspace).load() == draft
    assert server.jobs.snapshot() == []
    assert post(client, server, {"clear": True}) == {"draft": None}
    assert DraftStore(server.jobs.workspace).load() is None


@pytest.mark.parametrize("payload", [
    {"fields": {"action": "quantize"}}, {"fields": {"desiredGiB": 8}},
    {"fields": []}, {"fields": {"analyzeForm.source": "a" * 4097}},
    {"fields": {}, "clear": True}, {"clear": "true"},
])
def test_invalid_draft_cannot_replace_previous(server, payload):
    client = authorized_client(server)
    original = {"fields": {"desiredGiB": "8"}}
    post(client, server, original)
    with pytest.raises(HTTPError) as refused:
        post(client, server, payload)
    assert refused.value.code == 400
    assert server.drafts.load() == original


def test_atomic_draft_writes_and_corrupt_recovery(tmp_path):
    store = DraftStore(tmp_path)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda i: store.save({"fields": {"desiredGiB": str(i)}}), range(30)))
    assert int(store.load()["fields"]["desiredGiB"]) in range(30)
    assert not list(tmp_path.glob(".studio-draft-*"))
    store.path.write_text("{broken", encoding="utf-8")
    with pytest.raises(ValueError):
        store.load()
    store.clear()
    store.save({"fields": {}})
    assert store.load() == {"fields": {}}
