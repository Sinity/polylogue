"""Retained original checkpoints remain readable after mirror retirement."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from contextlib import contextmanager
from http.client import HTTPConnection
from pathlib import Path
from threading import Thread

import pytest

from polylogue.browser_capture import server as server_module
from polylogue.browser_capture.capture_jobs import CaptureJobRegistry
from polylogue.browser_capture.receiver import backfill_checkpoint_root
from polylogue.browser_capture.route_contracts import browser_capture_route_contract_for
from polylogue.core.enums import Provider
from polylogue.sources.dispatch import parse_payload

TOKEN = "neutral-retained-checkpoint-token"


@contextmanager
def _running_receiver(root: Path) -> Iterator[tuple[str, int]]:
    server = server_module.make_server("127.0.0.1", 0, spool_path=root, auth_token=TOKEN)
    server.daemon_threads = False
    server.block_on_close = True
    thread = Thread(target=server.serve_forever)
    thread.start()
    try:
        yield "127.0.0.1", server.server_port
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def _get(host: str, port: int, path: str) -> tuple[int, bytes]:
    connection = HTTPConnection(host, port)
    try:
        connection.request(
            "GET", path, headers={"Authorization": f"Bearer {TOKEN}", "X-Polylogue-Client-Protocol": "2"}
        )
        response = connection.getresponse()
        return response.status, response.read()
    finally:
        connection.close()


@pytest.mark.parametrize(
    "original",
    [
        b'{"checkpoint":{"version":1,"queue":[]},"unknown_metadata":{"absent":null,"empty":""}}',
        b'{"checkpoint":invalid original bytes',
    ],
)
def test_original_checkpoint_http_reader_preserves_exact_bytes_across_registry_restart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    original: bytes,
) -> None:
    monkeypatch.setattr(server_module, "receiver_identity", lambda _config: "neutral-receiver")
    root = backfill_checkpoint_root(tmp_path)
    root.mkdir()
    path = root / "original-instance.json"
    path.write_bytes(original)
    digest = "sha256:" + hashlib.sha256(original).hexdigest()
    for _ in range(2):
        with _running_receiver(tmp_path) as (host, port):
            status, page = _get(host, port, "/v1/capture-jobs/orphans?client_protocol=2")
            assert status == 200
            assert any(row["source_digest"] == digest for row in json.loads(page)["orphans"])
            status, retained = _get(host, port, f"/v1/capture-jobs/orphans/{digest}/payload?client_protocol=2")
            assert status == 200
            assert retained == original
        assert path.read_bytes() == original
    assert browser_capture_route_contract_for("GET", "/v1/backfill-checkpoint") is None
    assert browser_capture_route_contract_for("POST", "/v1/backfill-checkpoint") is None


def test_original_checkpoint_retains_acquired_capture_attachment_and_delivery_metadata(tmp_path: Path) -> None:
    fixture = Path(__file__).parents[2] / "fixtures" / "chatgpt" / "native-browser-capture-v1.json"
    envelope = json.loads(fixture.read_bytes())
    original = {
        "extension_instance_id": "original-profile",
        "stored_at": "2026-01-01T00:00:00Z",
        "checkpoint": {
            "version": 1,
            "queue": [
                {"id": "original-delivery", "attempts": 7, "last_error": "neutral-delivery-error", "envelope": envelope}
            ],
            "revisions": [{"receiver_request_id": "original-ack", "content_hash": "old-bookkeeping"}],
        },
    }
    original_bytes = json.dumps(original).encode()
    root = backfill_checkpoint_root(tmp_path)
    root.mkdir()
    path = root / "original-profile.json"
    path.write_bytes(original_bytes)
    digest = "sha256:" + hashlib.sha256(original_bytes).hexdigest()
    expected = parse_payload(Provider.CHATGPT, envelope, "original-source")
    assert expected and any(attachment.inline_bytes for attachment in expected[0].attachments)
    for _ in range(2):
        registry = CaptureJobRegistry(tmp_path, "neutral-receiver")
        registry.list_orphans(2)
        with registry.inspect_orphan(digest, 2) as (stream, size):
            assert size == len(original_bytes)
            read_back = json.load(stream)
        assert read_back == original
        acquired = read_back["checkpoint"]["queue"][0]
        assert acquired["attempts"] == 7
        assert acquired["last_error"] == "neutral-delivery-error"
        assert parse_payload(Provider.CHATGPT, acquired["envelope"], "original-source") == expected
        assert path.read_bytes() == original_bytes
