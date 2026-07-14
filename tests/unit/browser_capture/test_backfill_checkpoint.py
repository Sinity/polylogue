"""Tests for the browser-capture receiver's backfill-ledger checkpoint mirror.

polylogue-06zm: the extension's IndexedDB backfill ledger is the fast
primary source (already covered by browser-extension/tests/backfill.test.js
and the PR #2819 recovery-checkpoint work for jlme.3/jlme.4). This receiver
mirror is the second fallback for when a browser profile loss also destroys
the extension's local chrome.storage.local copy -- these tests exercise the
real HTTP route surface end to end (POST store -> GET read), not a mocked
transport.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from http import HTTPStatus
from http.client import HTTPConnection, HTTPResponse
from pathlib import Path
from threading import Thread
from typing import cast

import pytest

from polylogue.browser_capture.models import BrowserBackfillCheckpointRequest
from polylogue.browser_capture.receiver import (
    backfill_checkpoint_root,
    read_backfill_checkpoint,
    write_backfill_checkpoint,
)
from polylogue.browser_capture.route_contracts import (
    BROWSER_CAPTURE_ROUTE_CONTRACTS,
    browser_capture_route_contract_for,
)
from polylogue.browser_capture.server import make_server

_EXTENSION_ORIGIN = "chrome-extension://polylogue-test"


def _checkpoint_payload(sequence: int = 1) -> dict[str, object]:
    # Mirrors the shape browser-extension/src/backfill/storage.js
    # exportRecoveryCheckpoint() produces (opaque to the receiver).
    return {
        "version": 1,
        "sequence": sequence,
        "jobs": [{"id": "backfill-chatgpt-1", "status": "running", "inventory_cursor": "42"}],
        "queue": [{"id": "backfill-chatgpt-1:chatgpt:conv-1", "state": "eligible"}],
        "revisions": [],
    }


# ---- Direct function tests --------------------------------------------------


def test_write_then_read_round_trips_opaque_checkpoint(tmp_path: Path) -> None:
    request = BrowserBackfillCheckpointRequest(extension_instance_id="instance-a", checkpoint=_checkpoint_payload())

    written = write_backfill_checkpoint(request, spool_path=tmp_path)
    read_back = read_backfill_checkpoint("instance-a", spool_path=tmp_path)

    assert read_back is not None
    assert read_back.checkpoint == _checkpoint_payload()
    assert read_back.extension_instance_id == "instance-a"
    assert read_back.stored_at == written.stored_at


def test_read_missing_checkpoint_returns_none(tmp_path: Path) -> None:
    assert read_backfill_checkpoint("never-written", spool_path=tmp_path) is None


def test_write_overwrites_prior_checkpoint_for_same_instance_last_write_wins(tmp_path: Path) -> None:
    write_backfill_checkpoint(
        BrowserBackfillCheckpointRequest(extension_instance_id="instance-a", checkpoint=_checkpoint_payload(1)),
        spool_path=tmp_path,
    )
    write_backfill_checkpoint(
        BrowserBackfillCheckpointRequest(extension_instance_id="instance-a", checkpoint=_checkpoint_payload(2)),
        spool_path=tmp_path,
    )

    files = list(backfill_checkpoint_root(tmp_path).glob("*.json"))
    assert len(files) == 1  # one file per instance id, not one per write

    latest = read_backfill_checkpoint("instance-a", spool_path=tmp_path)
    assert latest is not None
    assert latest.checkpoint["sequence"] == 2


def test_distinct_instance_ids_get_distinct_checkpoints(tmp_path: Path) -> None:
    write_backfill_checkpoint(
        BrowserBackfillCheckpointRequest(extension_instance_id="instance-a", checkpoint=_checkpoint_payload(1)),
        spool_path=tmp_path,
    )
    write_backfill_checkpoint(
        BrowserBackfillCheckpointRequest(extension_instance_id="instance-b", checkpoint=_checkpoint_payload(2)),
        spool_path=tmp_path,
    )

    a = read_backfill_checkpoint("instance-a", spool_path=tmp_path)
    b = read_backfill_checkpoint("instance-b", spool_path=tmp_path)
    assert a is not None and a.checkpoint["sequence"] == 1
    assert b is not None and b.checkpoint["sequence"] == 2


def test_write_enforces_quota_before_writing_a_new_instance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import polylogue.browser_capture.receiver as receiver_mod

    monkeypatch.setattr(receiver_mod, "BACKFILL_CHECKPOINT_MAX_FILES", 1)
    write_backfill_checkpoint(
        BrowserBackfillCheckpointRequest(extension_instance_id="instance-a", checkpoint=_checkpoint_payload()),
        spool_path=tmp_path,
    )

    with pytest.raises(receiver_mod.SpoolQuotaExceededError):
        write_backfill_checkpoint(
            BrowserBackfillCheckpointRequest(extension_instance_id="instance-b", checkpoint=_checkpoint_payload()),
            spool_path=tmp_path,
        )

    # Overwriting the ALREADY-present instance never grows the file count, so
    # it must not be blocked by the same quota that stops a new instance.
    write_backfill_checkpoint(
        BrowserBackfillCheckpointRequest(extension_instance_id="instance-a", checkpoint=_checkpoint_payload(9)),
        spool_path=tmp_path,
    )
    refreshed = read_backfill_checkpoint("instance-a", spool_path=tmp_path)
    assert refreshed is not None and refreshed.checkpoint["sequence"] == 9


# ---- HTTP route surface -----------------------------------------------------


@contextmanager
def _running_receiver(tmp_path: Path) -> Iterator[tuple[str, int]]:
    server = make_server("127.0.0.1", 0, spool_path=tmp_path)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = cast(tuple[str, int], server.server_address[:2])
    try:
        yield host, port
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _request(
    host: str, port: int, method: str, path: str, *, body: object | None = None
) -> tuple[int, dict[str, object]]:
    conn = HTTPConnection(host, port)
    headers = {"Origin": _EXTENSION_ORIGIN}
    payload: str | None = None
    if body is not None:
        payload = json.dumps(body)
        headers["Content-Type"] = "application/json"
    conn.request(method, path, body=payload, headers=headers)
    response: HTTPResponse = conn.getresponse()
    raw = response.read()
    conn.close()
    parsed = json.loads(raw) if raw else {}
    return response.status, parsed


def test_route_contracts_registered() -> None:
    kinds = {c.kind for c in BROWSER_CAPTURE_ROUTE_CONTRACTS}
    assert {"backfill_checkpoint_store", "backfill_checkpoint_read"} <= kinds
    assert browser_capture_route_contract_for("POST", "/v1/backfill-checkpoint") is not None
    assert browser_capture_route_contract_for("GET", "/v1/backfill-checkpoint") is not None


def test_http_store_then_read_full_route(tmp_path: Path) -> None:
    with _running_receiver(tmp_path) as (host, port):
        status, stored = _request(
            host,
            port,
            "POST",
            "/v1/backfill-checkpoint",
            body={"extension_instance_id": "instance-http", "checkpoint": _checkpoint_payload()},
        )
        assert status == HTTPStatus.ACCEPTED
        assert stored["extension_instance_id"] == "instance-http"
        assert cast(int, stored["bytes_written"]) > 0
        assert isinstance(stored["stored_at"], str) and stored["stored_at"]

        status, fetched = _request(host, port, "GET", "/v1/backfill-checkpoint?extension_instance_id=instance-http")
        assert status == HTTPStatus.OK
        assert fetched["extension_instance_id"] == "instance-http"
        assert fetched["checkpoint"] == _checkpoint_payload()


def test_http_get_missing_checkpoint_returns_404(tmp_path: Path) -> None:
    with _running_receiver(tmp_path) as (host, port):
        status, body = _request(host, port, "GET", "/v1/backfill-checkpoint?extension_instance_id=nope")
    assert status == HTTPStatus.NOT_FOUND
    assert body["error"] == "checkpoint_not_found"


def test_http_get_without_instance_id_returns_400(tmp_path: Path) -> None:
    with _running_receiver(tmp_path) as (host, port):
        status, body = _request(host, port, "GET", "/v1/backfill-checkpoint")
    assert status == HTTPStatus.BAD_REQUEST
    assert body["error"] == "missing_extension_instance_id"


def test_http_store_rejects_invalid_payload(tmp_path: Path) -> None:
    with _running_receiver(tmp_path) as (host, port):
        status, body = _request(
            host, port, "POST", "/v1/backfill-checkpoint", body={"checkpoint": {"version": 1}}
        )  # missing extension_instance_id
    assert status == HTTPStatus.BAD_REQUEST
    assert body["error"] == "invalid_backfill_checkpoint"


def test_http_store_second_write_overwrites_first_over_the_wire(tmp_path: Path) -> None:
    with _running_receiver(tmp_path) as (host, port):
        _request(
            host,
            port,
            "POST",
            "/v1/backfill-checkpoint",
            body={"extension_instance_id": "instance-http", "checkpoint": _checkpoint_payload(1)},
        )
        _request(
            host,
            port,
            "POST",
            "/v1/backfill-checkpoint",
            body={"extension_instance_id": "instance-http", "checkpoint": _checkpoint_payload(2)},
        )
        status, fetched = _request(host, port, "GET", "/v1/backfill-checkpoint?extension_instance_id=instance-http")
    assert status == HTTPStatus.OK
    assert cast(dict[str, object], fetched["checkpoint"])["sequence"] == 2


def test_http_store_returns_429_without_writing_when_quota_exceeded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import polylogue.browser_capture.receiver as receiver_mod

    monkeypatch.setattr(receiver_mod, "BACKFILL_CHECKPOINT_MAX_FILES", 1)
    write_backfill_checkpoint(
        BrowserBackfillCheckpointRequest(extension_instance_id="instance-a", checkpoint=_checkpoint_payload()),
        spool_path=tmp_path,
    )
    before = sorted(backfill_checkpoint_root(tmp_path).glob("*.json"))

    with _running_receiver(tmp_path) as (host, port):
        status, body = _request(
            host,
            port,
            "POST",
            "/v1/backfill-checkpoint",
            body={"extension_instance_id": "instance-b", "checkpoint": _checkpoint_payload()},
        )

    assert status == HTTPStatus.TOO_MANY_REQUESTS
    assert body["error"] == "backfill_checkpoint_quota_exceeded"
    after = sorted(backfill_checkpoint_root(tmp_path).glob("*.json"))
    assert after == before
