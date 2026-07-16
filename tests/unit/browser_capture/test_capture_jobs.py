"""Production HTTP fixtures for receiver-authoritative CaptureJob recovery."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from http.client import HTTPConnection
from pathlib import Path
from threading import Thread
from typing import Any, cast

import polylogue.browser_capture.capture_jobs as capture_jobs
from polylogue.browser_capture.capture_jobs import canonical_digest
from polylogue.browser_capture.server import make_server

TOKEN = "capture-job-test-token"
SCOPE = "h1:" + "A" * 43
INTENT_KEY = "i1:" + "B" * 43


@contextmanager
def receiver(tmp_path: Path) -> Iterator[tuple[str, int]]:
    server = make_server("127.0.0.1", 0, spool_path=tmp_path, auth_token=TOKEN)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield "127.0.0.1", server.server_port
    finally:
        server.shutdown()
        thread.join()


def request(host: str, port: int, method: str, path: str, body: dict[str, object]) -> tuple[int, dict[str, Any]]:
    connection = HTTPConnection(host, port)
    connection.request(
        method,
        path,
        json.dumps(body),
        {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json", "X-Polylogue-Client-Protocol": "1"},
    )
    response = connection.getresponse()
    return response.status, json.loads(response.read())


def create(host: str, port: int) -> dict[str, Any]:
    payload = {"cutoff": "2026-01-01T00:00:00Z"}
    status, body = request(
        host,
        port,
        "POST",
        "/v1/capture-jobs",
        {
            "provider": "chatgpt",
            "account_scope": SCOPE,
            "request_id": "create",
            "intent": {
                "schema_version": 1,
                "version": 1,
                "intent_key": INTENT_KEY,
                "payload": payload,
                "digest": canonical_digest(payload),
            },
        },
    )
    assert status == 201
    return cast(dict[str, Any], body["job"])


def adopt(
    host: str, port: int, job: dict[str, Any], request_id: str = "adopt", session_id: str = "profile-a"
) -> dict[str, Any]:
    status, body = request(
        host,
        port,
        "POST",
        f"/v1/capture-jobs/{job['job_id']}/adopt",
        {
            "provider": "chatgpt",
            "account_scope": SCOPE,
            "request_id": request_id,
            "session_id": session_id,
            "expected_revision": job["revision"],
            "expected_lease_generation": job["lease_generation"],
        },
    )
    assert status == 200
    return body


def test_profile_loss_discovers_exact_scope_and_receiver_checkpoint(tmp_path: Path) -> None:
    with receiver(tmp_path) as (host, port):
        job = create(host, port)
        adopted = adopt(host, port, job)
        checkpoint = {"version": 1, "jobs": [{"id": "local-id", "provider": "chatgpt"}], "queue": [], "revisions": []}
        status, acknowledged = request(
            host,
            port,
            "PUT",
            f"/v1/capture-jobs/{job['job_id']}/checkpoint",
            {
                "provider": "chatgpt",
                "account_scope": SCOPE,
                "request_id": "checkpoint",
                "expected_revision": adopted["job"]["revision"],
                "lease_id": adopted["lease"]["lease_id"],
                "generation": adopted["lease"]["generation"],
                "proof": adopted["lease"]["proof"],
                "checkpoint": {"sequence": 1, "payload": checkpoint, "digest": canonical_digest(checkpoint)},
            },
        )
        assert status == 200
        status, found = request(
            host,
            port,
            "POST",
            "/v1/capture-jobs/discover",
            {"provider": "chatgpt", "account_scope": SCOPE, "intent_key": INTENT_KEY},
        )
        assert status == 200
        assert found["jobs"] == [acknowledged["job"]]
        assert found["jobs"][0]["checkpoint"]["payload"] == checkpoint
        status, hidden = request(
            host,
            port,
            "POST",
            "/v1/capture-jobs/discover",
            {"provider": "chatgpt", "account_scope": "h1:" + "C" * 43, "intent_key": INTENT_KEY},
        )
        assert status == 200 and hidden["jobs"] == []

        status, exact = request(
            host,
            port,
            "GET",
            f"/v1/capture-jobs/{job['job_id']}?provider=chatgpt&account_scope={SCOPE}&client_protocol=1",
            {},
        )
        assert status == 200
        assert exact["job"] == acknowledged["job"]


def test_adoption_and_checkpoint_conflicts_are_real_route_guards(tmp_path: Path) -> None:
    with receiver(tmp_path) as (host, port):
        job = create(host, port)
        adopted = adopt(host, port, job)
        status, duplicate = request(
            host,
            port,
            "POST",
            f"/v1/capture-jobs/{job['job_id']}/adopt",
            {
                "provider": "chatgpt",
                "account_scope": SCOPE,
                "request_id": "adopt",
                "session_id": "profile-a",
                "expected_revision": 0,
                "expected_lease_generation": 0,
            },
        )
        assert status == 200 and duplicate["lease"] == adopted["lease"]
        status, loser = request(
            host,
            port,
            "POST",
            f"/v1/capture-jobs/{job['job_id']}/adopt",
            {
                "provider": "chatgpt",
                "account_scope": SCOPE,
                "request_id": "other",
                "session_id": "profile-b",
                "expected_revision": 0,
                "expected_lease_generation": 0,
            },
        )
        assert status == 409 and loser["error"]["code"] == "cas_mismatch"
        checkpoint = {"cursor": 4}
        base = {
            "provider": "chatgpt",
            "account_scope": SCOPE,
            "expected_revision": adopted["job"]["revision"],
            "lease_id": adopted["lease"]["lease_id"],
            "generation": adopted["lease"]["generation"],
            "proof": adopted["lease"]["proof"],
        }
        status, first = request(
            host,
            port,
            "PUT",
            f"/v1/capture-jobs/{job['job_id']}/checkpoint",
            {
                **base,
                "request_id": "one",
                "checkpoint": {"sequence": 4, "payload": checkpoint, "digest": canonical_digest(checkpoint)},
            },
        )
        assert status == 200
        stale = {
            **base,
            "expected_revision": first["job"]["revision"],
            "request_id": "stale",
            "checkpoint": {"sequence": 3, "payload": {"cursor": 3}, "digest": canonical_digest({"cursor": 3})},
        }
        status, older = request(host, port, "PUT", f"/v1/capture-jobs/{job['job_id']}/checkpoint", stale)
        assert status == 409 and older["error"]["code"] == "older_checkpoint"
        conflict = {
            **base,
            "expected_revision": first["job"]["revision"],
            "request_id": "conflict",
            "checkpoint": {
                "sequence": 4,
                "payload": {"cursor": "other"},
                "digest": canonical_digest({"cursor": "other"}),
            },
        }
        status, equal = request(host, port, "PUT", f"/v1/capture-jobs/{job['job_id']}/checkpoint", conflict)
        assert status == 409 and equal["error"]["code"] == "checkpoint_conflict"
        status, incompatible = request(
            host,
            port,
            "POST",
            "/v1/capture-jobs/discover",
            {"provider": "chatgpt", "account_scope": SCOPE, "client_protocol": 99},
        )
        assert status == 426 and incompatible["error"]["code"] == "incompatible_client"


def test_expired_profile_lease_is_replaceable_but_live_lease_is_not(tmp_path: Path, monkeypatch: Any) -> None:
    with receiver(tmp_path) as (host, port):
        job = create(host, port)
        status, first = request(
            host,
            port,
            "POST",
            f"/v1/capture-jobs/{job['job_id']}/adopt",
            {
                "provider": "chatgpt",
                "account_scope": SCOPE,
                "request_id": "old-profile",
                "session_id": "destroyed-profile",
                "expected_revision": job["revision"],
                "expected_lease_generation": job["lease_generation"],
                "lease_ttl_seconds": 1,
            },
        )
        assert status == 200
        status, held = request(
            host,
            port,
            "POST",
            f"/v1/capture-jobs/{job['job_id']}/adopt",
            {
                "provider": "chatgpt",
                "account_scope": SCOPE,
                "request_id": "new-profile",
                "session_id": "replacement-profile",
                "expected_revision": first["job"]["revision"],
                "expected_lease_generation": first["lease"]["generation"],
            },
        )
        assert status == 409 and held["error"]["code"] == "lease_held"

        monkeypatch.setattr(capture_jobs, "_now", lambda: datetime.now(UTC) + timedelta(seconds=2))
        status, replacement = request(
            host,
            port,
            "POST",
            f"/v1/capture-jobs/{job['job_id']}/adopt",
            {
                "provider": "chatgpt",
                "account_scope": SCOPE,
                "request_id": "new-profile",
                "session_id": "replacement-profile",
                "expected_revision": first["job"]["revision"],
                "expected_lease_generation": first["lease"]["generation"],
            },
        )
        assert status == 200
        assert replacement["lease"]["generation"] == first["lease"]["generation"] + 1
