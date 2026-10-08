"""Production HTTP fixtures for receiver-authoritative CaptureJob recovery."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import socket
import sqlite3
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import UTC, datetime
from http.client import HTTPConnection, HTTPResponse
from pathlib import Path
from threading import Event, Thread
from typing import IO, Any, BinaryIO, cast
from urllib.parse import quote
from uuid import NAMESPACE_URL, uuid4, uuid5

import pytest

from polylogue.browser_capture import capture_jobs as capture_jobs_module
from polylogue.browser_capture.capture_jobs import (
    CaptureJobError,
    CaptureJobRegistry,
    canonical_digest,
    canonical_json,
    capture_job_database_path,
    capture_job_scope_namespace,
    capture_job_store_root,
)
from polylogue.browser_capture.models import BrowserCaptureEnvelope
from polylogue.browser_capture.receiver import (
    CaptureConvergence,
    capture_convergence,
    summarize_capture_envelope,
    write_capture_envelope,
)
from polylogue.browser_capture.route_contracts import browser_capture_route_contract_for
from polylogue.browser_capture.server import make_server
from polylogue.core.enums import Provider
from polylogue.sources.dispatch import parse_payload
from polylogue.sources.parsers.base_models import ParsedSession

TOKEN = "capture-job-test-token"
SCOPE = "h1:" + "A" * 43
ACCOUNT_SCOPE = {"kind": "account", "key": SCOPE}
SCOPE_QUERY = quote(json.dumps(ACCOUNT_SCOPE))
INTENT_KEY = "i1:" + "B" * 43


@contextmanager
def receiver(tmp_path: Path) -> Iterator[tuple[str, int]]:
    server = make_server("127.0.0.1", 0, spool_path=tmp_path, auth_token=TOKEN)
    server.daemon_threads = False
    server.block_on_close = True
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield "127.0.0.1", server.server_port
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def request(host: str, port: int, method: str, path: str, body: dict[str, object]) -> tuple[int, dict[str, Any]]:
    connection = HTTPConnection(host, port)
    headers = {
        "Authorization": f"Bearer {TOKEN}",
        "Content-Type": "application/json",
        "X-Polylogue-Client-Protocol": "2",
    }
    content = json.dumps(body).encode()
    if method == "PUT" and path.endswith("/checkpoint"):
        checkpoint = cast(dict[str, object], body["checkpoint"])
        descriptor: dict[str, object] = {
            key: body[key]
            for key in (
                "provider",
                "scope",
                "client_protocol",
                "expected_revision",
                "lease_id",
                "generation",
                "proof",
            )
            if key in body
        }
        descriptor.setdefault("client_protocol", 2)
        descriptor["request_id"] = str(uuid5(NAMESPACE_URL, str(body["request_id"])))
        descriptor.update(sequence=checkpoint["sequence"], digest=checkpoint["digest"])
        headers["X-Polylogue-Checkpoint"] = json.dumps(descriptor)
        content = canonical_json(checkpoint.get("payload")).encode()
    connection.request(
        method,
        path,
        content,
        headers,
    )
    response = connection.getresponse()
    try:
        return response.status, json.loads(response.read())
    finally:
        connection.close()


def _stored_job_ids(spool_path: Path) -> set[str]:
    with sqlite3.connect(capture_job_database_path(spool_path)) as connection:
        return {row[0] for row in connection.execute("SELECT job_id FROM capture_jobs")}


def test_protocol_two_discovery_preserves_the_previous_registry_and_artifacts(tmp_path: Path) -> None:
    """The production receiver opens a fresh namespace without upgrading v1."""
    legacy_root = tmp_path / "capture-jobs"
    legacy_artifacts = legacy_root / "artifacts"
    legacy_artifacts.mkdir(parents=True)
    checkpoint = b'{"conversation_ref":"legacy-checkpoint"}'
    digest = hashlib.sha256(checkpoint).hexdigest()
    artifact = legacy_artifacts / f"{digest}.checkpoint"
    artifact.write_bytes(checkpoint)
    legacy_database = legacy_root / "registry.sqlite3"
    with sqlite3.connect(legacy_database) as connection:
        connection.execute(
            """CREATE TABLE capture_jobs (
                job_id TEXT PRIMARY KEY, provider TEXT NOT NULL, account_scope TEXT NOT NULL,
                intent_key TEXT NOT NULL, intent_json TEXT NOT NULL, revision INTEGER NOT NULL,
                checkpoint_json TEXT, checkpoint_sequence INTEGER, checkpoint_digest TEXT,
                receipt_json TEXT, retry_json TEXT NOT NULL, lease_json TEXT,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                retention_json TEXT NOT NULL DEFAULT '{"state":"active","hold_reason":null,"timeline_authoritative":true}',
                retention_declared INTEGER NOT NULL DEFAULT 0,
                UNIQUE(provider, account_scope, intent_key)
            ) STRICT"""
        )
        connection.execute(
            """INSERT INTO capture_jobs (
                job_id, provider, account_scope, intent_key, intent_json, revision,
                checkpoint_json, checkpoint_sequence, checkpoint_digest, retry_json,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "legacy-job",
                "chatgpt",
                SCOPE,
                INTENT_KEY,
                "{}",
                1,
                checkpoint.decode(),
                1,
                "sha256:" + digest,
                '{"state":"ready"}',
                "2026-01-01T00:00:00Z",
                "2026-01-01T00:00:00Z",
            ),
        )
    legacy_database_bytes = legacy_database.read_bytes()
    legacy_database_inode = legacy_database.stat().st_ino
    checkpoint_bytes = artifact.read_bytes()
    checkpoint_inode = artifact.stat().st_ino

    with receiver(tmp_path) as (host, port):
        status, result = request(
            host,
            port,
            "POST",
            "/v1/capture-jobs/discover",
            {"provider": "chatgpt", "scope": ACCOUNT_SCOPE},
        )
        assert status == 200
        assert result["jobs"] == []

    current_database = capture_job_database_path(tmp_path)
    assert current_database == legacy_root / "v2" / "registry.sqlite3"
    with sqlite3.connect(current_database) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(capture_jobs)")}
        assert "scope_key" in columns and "account_scope" not in columns
    assert legacy_database.read_bytes() == legacy_database_bytes
    assert legacy_database.stat().st_ino == legacy_database_inode
    assert artifact.read_bytes() == checkpoint_bytes
    assert artifact.stat().st_ino == checkpoint_inode


def housekeeping(host: str, port: int, spool_path: Path, *, now: datetime | None = None) -> list[str]:
    """Drive discovery, the route every extension capture cycle opens with, and
    return the job IDs the receiver collected on that pass."""
    before = _stored_job_ids(spool_path)
    original = capture_jobs_module._now
    if now is not None:
        capture_jobs_module._now = lambda: now
    try:
        status, _payload = request(
            host, port, "POST", "/v1/capture-jobs/discover", {"provider": "chatgpt", "scope": ACCOUNT_SCOPE}
        )
    finally:
        capture_jobs_module._now = original
    assert status == 200
    return sorted(before - _stored_job_ids(spool_path))


def create(host: str, port: int, provider: str = "chatgpt") -> dict[str, Any]:
    payload = {"cutoff": "2026-01-01T00:00:00Z"}
    status, body = request(
        host,
        port,
        "POST",
        "/v1/capture-jobs",
        {
            "provider": provider,
            "scope": ACCOUNT_SCOPE,
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
            "provider": job["provider"],
            "scope": ACCOUNT_SCOPE,
            "request_id": request_id,
            "session_id": session_id,
            "expected_revision": job["revision"],
            "expected_lease_generation": job["lease_generation"],
        },
    )
    assert status == 200
    return body


def test_profile_loss_discovers_exact_scope_and_receiver_checkpoint(tmp_path: Path) -> None:
    assert canonical_json({"e\u0301": "e\u0301"}) == canonical_json({"é": "é"})
    assert canonical_digest({"value": "e\u0301"}) == canonical_digest({"value": "é"})
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
                "scope": ACCOUNT_SCOPE,
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
            {"provider": "chatgpt", "scope": ACCOUNT_SCOPE, "intent_key": INTENT_KEY},
        )
        assert status == 200
        assert found["jobs"] == [acknowledged["job"]]
        descriptor = {
            "provider": "chatgpt",
            "scope": ACCOUNT_SCOPE,
            "client_protocol": 2,
            "request_id": str(uuid5(NAMESPACE_URL, "checkpoint-read")),
            "expected_revision": acknowledged["job"]["revision"],
            "lease_id": adopted["lease"]["lease_id"],
            "generation": adopted["lease"]["generation"],
            "proof": adopted["lease"]["proof"],
            "sequence": 1,
            "digest": canonical_digest(checkpoint),
        }
        connection = HTTPConnection(host, port)
        connection.request(
            "GET",
            f"/v1/capture-jobs/{job['job_id']}/checkpoint-artifacts/{canonical_digest(checkpoint)}",
            headers={"Authorization": f"Bearer {TOKEN}", "X-Polylogue-Checkpoint": json.dumps(descriptor)},
        )
        response = connection.getresponse()
        assert response.status == 200
        assert response.read() == canonical_json(checkpoint).encode()
        connection.close()
        status, hidden = request(
            host,
            port,
            "POST",
            "/v1/capture-jobs/discover",
            {"provider": "chatgpt", "scope": {"kind": "account", "key": "h1:" + "C" * 43}, "intent_key": INTENT_KEY},
        )
        assert status == 200 and hidden["jobs"] == []

        status, exact = request(
            host,
            port,
            "GET",
            f"/v1/capture-jobs/{job['job_id']}?provider=chatgpt&scope={SCOPE_QUERY}&client_protocol=2",
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
                "scope": ACCOUNT_SCOPE,
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
                "scope": ACCOUNT_SCOPE,
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
            "scope": ACCOUNT_SCOPE,
            "expected_revision": adopted["job"]["revision"],
            "lease_id": adopted["lease"]["lease_id"],
            "generation": adopted["lease"]["generation"],
            "proof": adopted["lease"]["proof"],
        }
        forged = {
            **base,
            "request_id": "forged",
            "proof": "Z" * 43,
            "checkpoint": {"sequence": 5, "payload": {"cursor": 5}, "digest": canonical_digest({"cursor": 5})},
        }
        status, rejected_proof = request(host, port, "PUT", f"/v1/capture-jobs/{job['job_id']}/checkpoint", forged)
        assert status == 409 and rejected_proof["error"]["code"] == "lease_replaced"
        stale_revision = {
            **base,
            "request_id": "stale-revision",
            "expected_revision": adopted["job"]["revision"] - 1,
            "checkpoint": {"sequence": 5, "payload": {"cursor": 5}, "digest": canonical_digest({"cursor": 5})},
        }
        status, rejected_revision = request(
            host, port, "PUT", f"/v1/capture-jobs/{job['job_id']}/checkpoint", stale_revision
        )
        assert status == 409 and rejected_revision["error"]["code"] == "cas_mismatch"
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
        equal_request = {
            **base,
            "expected_revision": first["job"]["revision"],
            "request_id": "equal-no-op",
            "checkpoint": {"sequence": 4, "payload": checkpoint, "digest": canonical_digest(checkpoint)},
        }
        status, equal_no_op = request(host, port, "PUT", f"/v1/capture-jobs/{job['job_id']}/checkpoint", equal_request)
        assert status == 200 and equal_no_op["duplicate"] is True and equal_no_op["receipt"]["no_op"] is True
        status, reused_equal = request(
            host,
            port,
            "PUT",
            f"/v1/capture-jobs/{job['job_id']}/checkpoint",
            {
                **equal_request,
                "checkpoint": {
                    "sequence": 5,
                    "payload": {"cursor": 5},
                    "digest": canonical_digest({"cursor": 5}),
                },
            },
        )
        assert status == 409 and reused_equal["error"]["code"] == "request_id_conflict"
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
            {"provider": "chatgpt", "scope": ACCOUNT_SCOPE, "client_protocol": 99},
        )
        assert status == 426 and incompatible["error"]["code"] == "incompatible_client"


@pytest.mark.frozen_clock_modules("polylogue.browser_capture.capture_jobs")
def test_expired_profile_lease_is_replaceable_but_live_lease_is_not(tmp_path: Path, frozen_clock: Any) -> None:
    with receiver(tmp_path) as (host, port):
        job = create(host, port)
        status, first = request(
            host,
            port,
            "POST",
            f"/v1/capture-jobs/{job['job_id']}/adopt",
            {
                "provider": "chatgpt",
                "scope": ACCOUNT_SCOPE,
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
                "scope": ACCOUNT_SCOPE,
                "request_id": "new-profile",
                "session_id": "replacement-profile",
                "expected_revision": first["job"]["revision"],
                "expected_lease_generation": first["lease"]["generation"],
            },
        )
        assert status == 409 and held["error"]["code"] == "lease_held"

        frozen_clock.advance(2)
        status, expired = request(
            host,
            port,
            "PUT",
            f"/v1/capture-jobs/{job['job_id']}/checkpoint",
            {
                "provider": "chatgpt",
                "scope": ACCOUNT_SCOPE,
                "request_id": "expired-checkpoint",
                "expected_revision": first["job"]["revision"],
                "lease_id": first["lease"]["lease_id"],
                "generation": first["lease"]["generation"],
                "proof": first["lease"]["proof"],
                "checkpoint": {"sequence": 1, "payload": {}, "digest": canonical_digest({})},
            },
        )
        assert status == 409 and expired["error"]["code"] == "lease_expired"
        status, replacement = request(
            host,
            port,
            "POST",
            f"/v1/capture-jobs/{job['job_id']}/adopt",
            {
                "provider": "chatgpt",
                "scope": ACCOUNT_SCOPE,
                "request_id": "new-profile",
                "session_id": "replacement-profile",
                "expected_revision": first["job"]["revision"],
                "expected_lease_generation": first["lease"]["generation"],
            },
        )
        assert status == 200
        assert replacement["lease"]["generation"] == first["lease"]["generation"] + 1
        status, replaced = request(
            host,
            port,
            "PUT",
            f"/v1/capture-jobs/{job['job_id']}/checkpoint",
            {
                "provider": "chatgpt",
                "scope": ACCOUNT_SCOPE,
                "request_id": "replaced-checkpoint",
                "expected_revision": replacement["job"]["revision"],
                "lease_id": first["lease"]["lease_id"],
                "generation": first["lease"]["generation"],
                "proof": first["lease"]["proof"],
                "checkpoint": {"sequence": 1, "payload": {}, "digest": canonical_digest({})},
            },
        )
        assert status == 409 and replaced["error"]["code"] == "lease_replaced"


def test_state_update_renews_lease_is_idempotent_and_exposes_receipts(tmp_path: Path) -> None:
    with receiver(tmp_path) as (host, port):
        job = create(host, port)
        adopted = adopt(host, port, job)
        update = {
            "provider": "chatgpt",
            "scope": ACCOUNT_SCOPE,
            "request_id": "hold-update",
            "expected_revision": adopted["job"]["revision"],
            "lease_id": adopted["lease"]["lease_id"],
            "generation": adopted["lease"]["generation"],
            "proof": adopted["lease"]["proof"],
            "lease_ttl_seconds": 240,
            "retry": {
                "state": "held",
                "attempt": 3,
                "reason": "provider_safety_interstitial",
                "next_eligible_at": None,
            },
        }
        status, updated = request(host, port, "POST", f"/v1/capture-jobs/{job['job_id']}/update", update)
        assert status == 200
        assert updated["job"]["revision"] == adopted["job"]["revision"] + 1
        assert updated["job"]["retry"] == update["retry"]
        assert updated["receipt"]["kind"] == "capture_job_update"
        assert updated["job"]["lease_expires_at"] != adopted["job"]["lease_expires_at"]

        missing_proof = {**update, "request_id": "missing-proof", "expected_revision": updated["job"]["revision"]}
        missing_proof.pop("proof")
        status, rejected_proof = request(host, port, "POST", f"/v1/capture-jobs/{job['job_id']}/update", missing_proof)
        assert status == 409 and rejected_proof["error"]["code"] == "lease_replaced"
        stale_revision = {**update, "request_id": "stale-update"}
        status, rejected_revision = request(
            host, port, "POST", f"/v1/capture-jobs/{job['job_id']}/update", stale_revision
        )
        assert status == 409 and rejected_revision["error"]["code"] == "cas_mismatch"

        status, duplicate = request(host, port, "POST", f"/v1/capture-jobs/{job['job_id']}/update", update)
        assert status == 200 and duplicate["duplicate"] is True
        conflict = {**update, "retry": {**update["retry"], "attempt": 4}}
        status, rejected = request(host, port, "POST", f"/v1/capture-jobs/{job['job_id']}/update", conflict)
        assert status == 409 and rejected["error"]["code"] == "request_id_conflict"

        no_op = {
            **update,
            "request_id": "no-op-update",
            "expected_revision": updated["job"]["revision"],
        }
        no_op.pop("lease_ttl_seconds")
        status, no_op_result = request(host, port, "POST", f"/v1/capture-jobs/{job['job_id']}/update", no_op)
        assert status == 200 and no_op_result["duplicate"] is True and no_op_result["receipt"]["no_op"] is True
        status, reused_no_op = request(
            host,
            port,
            "POST",
            f"/v1/capture-jobs/{job['job_id']}/update",
            {**no_op, "retry": {**update["retry"], "attempt": 4}},
        )
        assert status == 409 and reused_no_op["error"]["code"] == "request_id_conflict"

        query = f"provider=chatgpt&scope={SCOPE_QUERY}&client_protocol=2"
        status, detail = request(host, port, "GET", f"/v1/capture-jobs/{job['job_id']}?{query}", {})
        assert status == 200
        assert detail["job"]["latest_receipt"] is None
        assert detail["receipts"] == [updated["receipt"], no_op_result["receipt"]]


def test_events_are_receiver_ordered_scoped_and_idempotent(tmp_path: Path) -> None:
    """Anti-vacuity: removing event CAS or request-id replay protection makes this red."""
    with receiver(tmp_path) as (host, port):
        job = create(host, port)
        adopted = adopt(host, port, job)
        event_body = {
            "provider": "chatgpt",
            "scope": ACCOUNT_SCOPE,
            "request_id": "first-seen-1",
            "expected_revision": adopted["job"]["revision"],
            "lease_id": adopted["lease"]["lease_id"],
            "generation": adopted["lease"]["generation"],
            "proof": adopted["lease"]["proof"],
            "kind": "first-seen",
            "refs": {"conversation_ref": "conversation:1", "message_ref": "message:1"},
            "payload": {"source": "profile-a"},
        }
        status, first = request(host, port, "POST", f"/v1/capture-jobs/{job['job_id']}/events", event_body)
        assert status == 200
        assert first["event"]["event_revision"] == 1
        assert first["event"]["job_revision"] == adopted["job"]["revision"] + 1
        assert first["job"]["revision"] == adopted["job"]["revision"] + 1
        status, replay = request(host, port, "POST", f"/v1/capture-jobs/{job['job_id']}/events", event_body)
        assert status == 200 and replay["event"] == first["event"] and replay["duplicate"] is True
        stale_checkpoint = {
            **event_body,
            "request_id": "after-event-stale",
            "expected_revision": adopted["job"]["revision"],
            "checkpoint": {"sequence": 1, "payload": {}, "digest": canonical_digest({})},
        }
        status, rejected_checkpoint = request(
            host, port, "PUT", f"/v1/capture-jobs/{job['job_id']}/checkpoint", stale_checkpoint
        )
        assert status == 409 and rejected_checkpoint["error"]["code"] == "cas_mismatch"
        stale = {**event_body, "request_id": "stale", "expected_revision": adopted["job"]["revision"] - 1}
        status, rejected = request(host, port, "POST", f"/v1/capture-jobs/{job['job_id']}/events", stale)
        assert status == 409 and rejected["error"]["code"] == "cas_mismatch"
        status, page = request(
            host,
            port,
            "GET",
            f"/v1/capture-jobs/{job['job_id']}/events?provider=chatgpt&scope={SCOPE_QUERY}&client_protocol=2&limit=10",
            {},
        )
        assert status == 200
        assert [event["kind"] for event in page["events"]] == ["created", "first-seen"]
        assert page["events"][1]["refs"]["conversation_ref"] == "conversation:1"
        assert page["timelines"] == {"conversation:1": [first["event"]]}


def test_http_rejects_event_cursor_outside_sqlite_integer_range(tmp_path: Path) -> None:
    """Anti-vacuity: passing this cursor to sqlite binding raises OverflowError."""
    with receiver(tmp_path) as (host, port):
        job = create(host, port)
        path = (
            f"/v1/capture-jobs/{job['job_id']}/events?provider=chatgpt&scope={SCOPE_QUERY}"
            "&client_protocol=2&before_revision=999999999999999999999999999999"
        )
        status, response = request(host, port, "GET", path, {})
    assert status == 400
    assert response["error"] == "invalid_capture_job_events_query"


def test_timeline_uses_receiver_order_and_gc_requires_terminal_retention(tmp_path: Path) -> None:
    """Anti-vacuity: timestamp order or retention/terminal/lease bypass makes this fail.

    Collection is driven through discovery, the route the extension opens every
    capture cycle with, so unwiring it from that route makes this fail too.
    """
    with receiver(tmp_path) as (host, port):
        job = create(host, port)
        adopted = adopt(host, port, job)
        base = {
            "provider": "chatgpt",
            "scope": ACCOUNT_SCOPE,
            "lease_id": adopted["lease"]["lease_id"],
            "generation": adopted["lease"]["generation"],
            "proof": adopted["lease"]["proof"],
        }
        first_event = {
            **base,
            "request_id": "timeline-first",
            "expected_revision": adopted["job"]["revision"],
            "kind": "first-seen",
            "refs": {"conversation_ref": "conversation:1"},
            "payload": {"ordinal": 1},
        }
        status, first = request(host, port, "POST", f"/v1/capture-jobs/{job['job_id']}/events", first_event)
        assert status == 200
        status, second = request(
            host,
            port,
            "POST",
            f"/v1/capture-jobs/{job['job_id']}/events",
            {
                **first_event,
                "request_id": "timeline-second",
                "expected_revision": first["job"]["revision"],
                "kind": "detected-new",
                "payload": {"ordinal": 2},
            },
        )
        assert status == 200
        checkpoint_payload = {"cursor": 2}
        status, checkpoint = request(
            host,
            port,
            "PUT",
            f"/v1/capture-jobs/{job['job_id']}/checkpoint",
            {
                **base,
                "request_id": "timeline-checkpoint",
                "expected_revision": second["job"]["revision"],
                "checkpoint": {
                    "sequence": 1,
                    "payload": checkpoint_payload,
                    "digest": canonical_digest(checkpoint_payload),
                },
            },
        )
        assert status == 200
        status, invalid_retention = request(
            host,
            port,
            "POST",
            f"/v1/capture-jobs/{job['job_id']}/update",
            {
                **base,
                "request_id": "invalid-retention",
                "expected_revision": checkpoint["job"]["revision"],
                "retention": {"state": "eligible", "hold_reason": None, "timeline_authoritative": 0},
            },
        )
        assert status == 400 and invalid_retention["error"]["code"] == "invalid_retention_state"
        status, retention = request(
            host,
            port,
            "POST",
            f"/v1/capture-jobs/{job['job_id']}/update",
            {
                **base,
                "request_id": "eligible-before-completion",
                "expected_revision": checkpoint["job"]["revision"],
                "retention": {"state": "eligible", "hold_reason": None, "timeline_authoritative": False},
            },
        )
        assert status == 200
        future = datetime(2050, 1, 1, tzinfo=UTC)
        assert housekeeping(host, port, tmp_path, now=future) == []

        status, completed = request(
            host,
            port,
            "POST",
            f"/v1/capture-jobs/{job['job_id']}/update",
            {
                **base,
                "request_id": "completed-for-gc",
                "expected_revision": retention["job"]["revision"],
                "retry": {"state": "completed", "attempt": 1, "reason": None, "next_eligible_at": None},
            },
        )
        assert status == 200
        assert housekeeping(host, port, tmp_path) == []
        status, page = request(
            host,
            port,
            "GET",
            f"/v1/capture-jobs/{job['job_id']}/events?provider=chatgpt&scope={SCOPE_QUERY}&client_protocol=2",
            {},
        )
        assert status == 200
        assert page["timelines"]["conversation:1"] == [second["event"], first["event"]]
        assert housekeeping(host, port, tmp_path, now=future) == [job["job_id"]]
        assert (
            request(
                host,
                port,
                "GET",
                f"/v1/capture-jobs/{job['job_id']}?provider=chatgpt&scope={SCOPE_QUERY}&client_protocol=2",
                {},
            )[0]
            == 404
        )


def test_legacy_checkpoint_is_a_typed_orphan_and_routes_are_declared(tmp_path: Path) -> None:
    root = tmp_path / "backfill-checkpoints"
    root.mkdir(parents=True)
    (root / "legacy-instance.json").write_text(
        json.dumps({"extension_instance_id": "legacy-instance", "checkpoint": {"version": 1, "jobs": []}}),
        encoding="utf-8",
    )
    with receiver(tmp_path) as (host, port):
        status, found = request(
            host,
            port,
            "POST",
            "/v1/capture-jobs/discover",
            {"provider": "chatgpt", "scope": ACCOUNT_SCOPE},
        )
        assert status == 200
        assert found["jobs"] == []
        assert "orphans" not in found
        status, orphan_census = request(
            host,
            port,
            "GET",
            "/v1/capture-jobs/orphans?client_protocol=2",
            {},
        )
        assert status == 200
        assert len(orphan_census["orphans"]) == 1
        assert orphan_census["orphans"][0]["orphan_kind"] == "legacy_backfill_checkpoint"
        assert "legacy-instance" not in json.dumps(orphan_census["orphans"])

    routes = {
        ("GET", "/v1/capture-jobs/capabilities"),
        ("POST", "/v1/capture-jobs"),
        ("POST", "/v1/capture-jobs/discover"),
        ("GET", "/v1/capture-jobs/job-id"),
        ("GET", "/v1/capture-jobs/orphans"),
        ("POST", "/v1/capture-jobs/job-id/adopt"),
        ("POST", "/v1/capture-jobs/job-id/update"),
        ("PUT", "/v1/capture-jobs/job-id/checkpoint"),
    }
    assert all(browser_capture_route_contract_for(method, path) is not None for method, path in routes)
    assert capture_job_scope_namespace(tmp_path) == capture_job_scope_namespace(tmp_path)
    assert capture_job_scope_namespace(tmp_path) != capture_job_scope_namespace(tmp_path / "other")


def test_orphan_census_reports_unreadable_files_and_refreshes_diagnostics(tmp_path: Path, monkeypatch: Any) -> None:
    """Anti-vacuity: bypassing unreadable-file entries or upsert refresh makes this test fail."""
    root = tmp_path / "backfill-checkpoints"
    root.mkdir(parents=True)
    readable = root / "changing.json"
    unreadable = root / "unreadable.json"
    readable.write_text(json.dumps({"checkpoint": {"version": 1}}), encoding="utf-8")
    unreadable.write_text("{}", encoding="utf-8")
    registry = CaptureJobRegistry(tmp_path, "receiver")
    connection = registry._connect()
    try:
        digest = "sha256:" + hashlib.sha256(readable.read_bytes()).hexdigest()
        connection.execute(
            "INSERT INTO capture_job_orphans VALUES (?, ?, ?, ?)",
            (digest, "stale_kind", "stale diagnostic", "2026-01-01T00:00:00Z"),
        )
        connection.commit()
        first = cast(list[dict[str, object]], registry.list_orphans(2)["orphans"])
        refreshed_first = next(entry for entry in first if entry["source_digest"] == digest)
        assert refreshed_first["orphan_kind"] == "legacy_backfill_checkpoint"
        assert refreshed_first["source_digest"] == digest

        readable.write_text("not json", encoding="utf-8")

        original_open = Path.open

        def raise_for_unreadable(path: Path, *args: Any, **kwargs: Any) -> IO[Any]:
            if path == unreadable:
                raise PermissionError(13, "permission denied")
            return cast(IO[Any], original_open(path, *args, **kwargs))

        monkeypatch.setattr(Path, "open", raise_for_unreadable)
        second = cast(list[dict[str, object]], registry.list_orphans(2)["orphans"])
        refreshed = next(entry for entry in second if entry["orphan_kind"] == "malformed_legacy_checkpoint")
        assert refreshed["source_digest"] != digest
        unreadable_entry = next(entry for entry in second if entry["orphan_kind"] == "unreadable_legacy_checkpoint")
        assert str(unreadable_entry["source_digest"]).startswith("path-sha256:")
        assert str(unreadable) not in json.dumps(unreadable_entry)
        assert unreadable_entry["errno_class"] == "PermissionError"
    finally:
        connection.close()


def test_orphan_listing_preserves_every_diagnostic_message_across_pages(tmp_path: Path) -> None:
    registry = CaptureJobRegistry(tmp_path, "receiver")
    with registry._connection() as connection:
        for ordinal in range(31):
            connection.execute(
                "INSERT INTO capture_job_orphans VALUES (?, ?, ?, ?)",
                (
                    f"sha256:{ordinal:064x}",
                    "unreadable_legacy_checkpoint",
                    json.dumps(
                        {"message": f"retained message e\u0301 {ordinal}", "errno_class": None},
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    "2026-01-01T00:00:00Z",
                ),
            )
    capture_jobs_module._SCHEMA_READY.clear()
    first = registry.list_orphans(2)
    second = registry.list_orphans(2, cast(str, first["cursor"]))
    entries = cast(list[dict[str, object]], first["orphans"]) + cast(list[dict[str, object]], second["orphans"])
    assert len(entries) == 31
    assert [entry["diagnostic"] for entry in entries] == [
        f"retained message e\u0301 {ordinal}" for ordinal in range(31)
    ]
    assert all(entry["errno_class"] is None for entry in entries)
    assert second["has_more"] is False
    with registry._connection() as connection:
        assert connection.execute("SELECT COUNT(*) FROM capture_job_orphans").fetchone()[0] == 31
        assert all(
            set(json.loads(row[0])) == {"message", "errno_class"}
            for row in connection.execute("SELECT diagnostic FROM capture_job_orphans")
        )


def test_concurrent_first_registry_opens_serialize_schema_upgrade(tmp_path: Path) -> None:
    """Anti-vacuity: racing first openers must serialize fresh schema creation."""
    registries = [CaptureJobRegistry(tmp_path, f"receiver-{index}") for index in range(8)]

    def open_and_close(registry: CaptureJobRegistry) -> None:
        connection = registry._connect()
        connection.close()

    with ThreadPoolExecutor(max_workers=len(registries)) as pool:
        list(pool.map(open_and_close, registries))


def test_fresh_registry_preserves_preexisting_source_capture_and_checkpoint(tmp_path: Path) -> None:
    """Fresh bookkeeping must neither remove nor reinterpret original source bytes."""
    fixture = Path(__file__).parents[2] / "fixtures" / "chatgpt" / "native-browser-capture-v1.json"
    payload = json.loads(fixture.read_text())
    envelope = BrowserCaptureEnvelope.model_validate(payload)
    capture = write_capture_envelope(envelope, spool_path=tmp_path)
    checkpoint = {"version": 1, "jobs": [], "queue": [{"id": "original-source", "envelope": payload}], "revisions": []}
    original = {
        "extension_instance_id": "original-profile",
        "checkpoint": checkpoint,
        "stored_at": "2026-01-01T00:00:00Z",
    }
    checkpoint_root = tmp_path / "backfill-checkpoints"
    checkpoint_root.mkdir()
    original_path = checkpoint_root / "original-profile.json"
    original_bytes = json.dumps(original).encode()
    original_path.write_bytes(original_bytes)
    original_digest = "sha256:" + hashlib.sha256(original_bytes).hexdigest()
    source_files = {path: path.read_bytes() for path in tmp_path.rglob("*.json")}
    parsed = parse_payload(Provider.CHATGPT, json.loads(capture.path.read_bytes()), "original-source")
    assert parsed and any(attachment.inline_bytes for attachment in parsed[0].attachments)

    for _ in range(2):
        # Process-local bootstrap knowledge does not survive a receiver restart.
        capture_jobs_module._SCHEMA_READY.clear()
        connection = CaptureJobRegistry(tmp_path, "fresh-receiver")._connect()
        connection.close()
        assert {path: path.read_bytes() for path in source_files} == source_files
        registry = CaptureJobRegistry(tmp_path, "fresh-receiver")
        with registry.inspect_orphan(original_digest, 2) as (stream, size):
            assert size == len(original_bytes)
            assert stream.read() == original_bytes
        assert parse_payload(Provider.CHATGPT, json.loads(capture.path.read_bytes()), "original-source") == parsed
        duplicate = write_capture_envelope(envelope, spool_path=tmp_path)
        assert duplicate.deduplicated and duplicate.path == capture.path


def test_timeline_retention_ignores_empty_and_non_string_refs(tmp_path: Path) -> None:
    """Anti-vacuity: SQL IS NOT NULL counted refs the projection cannot timeline."""
    registry = CaptureJobRegistry(tmp_path, "receiver")
    with registry.result_scope():
        _, created = registry.create(
            {
                "provider": "chatgpt",
                "scope": ACCOUNT_SCOPE,
                "client_protocol": 2,
                "intent": {
                    "schema_version": 1,
                    "version": 1,
                    "intent_key": INTENT_KEY,
                    "payload": {},
                    "digest": canonical_digest({}),
                },
            }
        )
        job_id = cast(dict[str, Any], created["job"])["job_id"]
    with registry._connection() as connection:
        for index, ref in enumerate(("", None, 17)):
            connection.execute(
                "INSERT INTO capture_job_events "
                "(event_id, job_id, event_revision, job_revision, kind, refs_json, payload_json, request_id, occurred_at) "
                "VALUES (?, ?, ?, 0, 'first-seen', ?, '{}', ?, '2026-01-01T00:00:00Z')",
                (f"bad-ref-{index}", job_id, index + 1, canonical_json({"conversation_ref": ref}), f"bad-{index}"),
            )
        assert registry._holds_conversation_timeline(connection, job_id) is False


def test_registry_uses_full_synchronous_mode(tmp_path: Path) -> None:
    """Anti-vacuity: omitting FULL on a fresh registry connection makes this assertion fail."""
    registry = CaptureJobRegistry(tmp_path, "receiver")
    connection = registry._connect()
    try:
        assert connection.execute("PRAGMA synchronous").fetchone()[0] == 2
    finally:
        connection.close()


def test_get_uses_one_snapshot_for_job_and_receipts(tmp_path: Path, monkeypatch: Any) -> None:
    """Anti-vacuity: removing BEGIN lets the injected committed receipt leak into the response."""
    registry = CaptureJobRegistry(tmp_path, "receiver")
    payload = {"cutoff": "2026-01-01T00:00:00Z"}
    with registry.result_scope():
        _, created = registry.create(
            {
                "provider": "chatgpt",
                "scope": ACCOUNT_SCOPE,
                "client_protocol": 2,
                "intent": {
                    "schema_version": 1,
                    "version": 1,
                    "intent_key": INTENT_KEY,
                    "payload": payload,
                    "digest": canonical_digest(payload),
                },
            }
        )
        created_job = cast(dict[str, object], created["job"])
        job_id = cast(str, created_job["job_id"])
    original_connect = registry._connect
    injected = False

    def connect_with_interleaved_commit(_registry: CaptureJobRegistry) -> sqlite3.Connection:
        nonlocal injected
        connection = original_connect()
        if injected:
            return connection

        def inject(_statement: str) -> None:
            nonlocal injected
            if injected or not _statement.startswith("SELECT receipt_json FROM capture_job_receipts"):
                return
            injected = True
            other = sqlite3.connect(capture_job_database_path(tmp_path), isolation_level=None)
            try:
                other.execute(
                    "INSERT INTO capture_job_receipts VALUES (?, ?, ?, ?)",
                    (job_id, "interleaved", 1, json.dumps({"receipt_id": "interleaved"})),
                )
            finally:
                other.close()

        connection.set_trace_callback(inject)
        return connection

    monkeypatch.setattr(CaptureJobRegistry, "_connect", connect_with_interleaved_commit)
    with registry.result_scope():
        result = registry.get(job_id, {"provider": "chatgpt", "scope": ACCOUNT_SCOPE, "client_protocol": 2})
        assert result["receipts"] == []


def test_registry_storage_failure_is_a_structured_receiver_error(tmp_path: Path, monkeypatch: Any) -> None:
    def fail_discover(_registry: CaptureJobRegistry, _body: dict[str, object]) -> dict[str, object]:
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(CaptureJobRegistry, "discover", fail_discover)
    with receiver(tmp_path) as (host, port):
        status, body = request(
            host,
            port,
            "POST",
            "/v1/capture-jobs/discover",
            {"provider": "chatgpt", "scope": ACCOUNT_SCOPE},
        )
    assert status == 500
    assert body == {"error": {"code": "registry_unavailable", "details": {}}}


@pytest.mark.parametrize("route", ["native/member", "native/asset", "checkpoint"])
@pytest.mark.parametrize("exhausted", [False, True])
@pytest.mark.uses_real_clock("The socket prefix waits for an actual HTTP refusal and physical connection EOF.")
def test_body_registry_storage_fault_refuses_socket_prefix_before_reading_body(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, route: str, exhausted: bool
) -> None:
    from polylogue.browser_capture import server as server_module

    def unavailable(*_args: object, **_kwargs: object) -> CaptureJobRegistry:
        if exhausted:
            raise OSError(errno.ENOSPC, "synthetic physical storage exhaustion")
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(server_module, "registry_for_receiver", unavailable)
    descriptor: dict[str, object] = {
        "provider": "chatgpt",
        "scope": ACCOUNT_SCOPE,
        "client_protocol": 2,
        "request_id": str(uuid4()),
        "expected_revision": 1,
        "lease_id": str(uuid4()),
        "generation": 1,
        "proof": "A" * 43,
        "sequence": 0,
        "digest": canonical_digest({}),
    }
    header = "X-Polylogue-Checkpoint" if route == "checkpoint" else "X-Polylogue-Native"
    with receiver(tmp_path) as (host, port), socket.create_connection((host, port), timeout=5) as connection:
        connection.sendall(
            (
                f"PUT /v1/capture-jobs/synthetic-job/{route} HTTP/1.1\r\n"
                f"Host: {host}\r\nAuthorization: Bearer {TOKEN}\r\n"
                f"{header}: {json.dumps(descriptor)}\r\nContent-Length: 100\r\n\r\n"
            ).encode()
        )
        response = HTTPResponse(connection)
        response.begin()
        assert response.status == (507 if exhausted else 500)
        assert json.loads(response.read()) == {
            "error": {
                "code": "spool_storage_exhausted" if exhausted else "registry_unavailable",
                "details": {},
            }
        }
        assert connection.recv(1) == b""
    assert not list(tmp_path.rglob("*.part"))


def test_checkpoint_physical_storage_refusal_retains_job_and_retries_without_false_ack(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Physical ENOSPC is visible; restoring publication can ACK the same input."""
    with receiver(tmp_path) as (host, port):
        job = create(host, port)
        adopted = adopt(host, port, job)
        lease = adopted["lease"]
        payload = {"cursor": "retained"}
        body = {
            "provider": "chatgpt",
            "scope": ACCOUNT_SCOPE,
            "lease_id": lease["lease_id"],
            "generation": lease["generation"],
            "proof": lease["proof"],
            "request_id": "physical-storage",
            "expected_revision": adopted["job"]["revision"],
            "checkpoint": {"sequence": 0, "payload": payload, "digest": canonical_digest(payload)},
        }
        publish = CaptureJobRegistry._publish_checkpoint_artifact

        def exhausted(_registry: CaptureJobRegistry, _staged: Any, _digest: str) -> str:
            raise OSError(errno.ENOSPC, "synthetic physical storage exhaustion")

        monkeypatch.setattr(CaptureJobRegistry, "_publish_checkpoint_artifact", exhausted)
        status, refusal = request(host, port, "PUT", f"/v1/capture-jobs/{job['job_id']}/checkpoint", body)
        assert status == 507 and refusal["error"]["code"] == "spool_storage_exhausted"
        with sqlite3.connect(capture_job_database_path(tmp_path)) as connection:
            row = connection.execute(
                "SELECT checkpoint_artifact_ref, revision, scope_key FROM capture_jobs WHERE job_id=?",
                (job["job_id"],),
            ).fetchone()
            assert row == (None, adopted["job"]["revision"], SCOPE)
        monkeypatch.setattr(CaptureJobRegistry, "_publish_checkpoint_artifact", publish)
        status, accepted = request(host, port, "PUT", f"/v1/capture-jobs/{job['job_id']}/checkpoint", body)
        assert status == 200 and accepted["receipt"]["checkpoint_digest"] == canonical_digest(payload)


def test_scope_namespace_survives_receiver_bearer_rotation(tmp_path: Path) -> None:
    namespaces = []
    for token in ("old-pairing-token", "rotated-pairing-token"):
        server = make_server("127.0.0.1", 0, spool_path=tmp_path, auth_token=token)
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            connection = HTTPConnection("127.0.0.1", server.server_port)
            connection.request(
                "GET",
                "/v1/capture-jobs/capabilities",
                headers={"Authorization": f"Bearer {token}"},
            )
            response = connection.getresponse()
            assert response.status == 200
            namespaces.append(json.loads(response.read())["scope_namespace"])
        finally:
            server.shutdown()
            thread.join()
    assert namespaces[0] == namespaces[1]


def _checkpoint(
    host: str,
    port: int,
    job_id: str,
    lease: dict[str, Any],
    revision: int,
    sequence: int,
    payload: dict[str, object],
    request_id: str,
) -> dict[str, Any]:
    status, body = request(
        host,
        port,
        "PUT",
        f"/v1/capture-jobs/{job_id}/checkpoint",
        {
            "provider": "chatgpt",
            "scope": ACCOUNT_SCOPE,
            "lease_id": lease["lease_id"],
            "generation": lease["generation"],
            "proof": lease["proof"],
            "request_id": request_id,
            "expected_revision": revision,
            "checkpoint": {"sequence": sequence, "payload": payload, "digest": canonical_digest(payload)},
        },
    )
    assert status == 200, body
    return body


def test_source_bearing_registry_checkpoint_remains_readable_after_reopen(tmp_path: Path) -> None:
    """Job bookkeeping may share a carrier with unique acquired source bytes."""
    fixture = Path(__file__).parents[2] / "fixtures" / "chatgpt" / "native-browser-capture-v1.json"
    original = json.loads(fixture.read_text())
    # This registry's current checkpoint wire admits exact integers. Use the
    # fixture's existing DOM projection for this source-only carrier; the
    # ordinary spool control above preserves its full raw payload and assets.
    original.pop("raw_provider_payload")
    original["session"]["attachments"] = []
    original["provider_meta"]["capture_fidelity"] = "dom_fallback"
    original["session"]["provider_meta"]["capture_fidelity"] = "dom_fallback"
    with receiver(tmp_path) as (host, port):
        job = create(host, port)
        adopted = adopt(host, port, job)
        payload = {
            "version": 1,
            "jobs": [],
            "queue": [{"id": "source-only-in-registry", "envelope": original}],
            "revisions": [],
        }
        saved = _checkpoint(
            host, port, job["job_id"], adopted["lease"], adopted["job"]["revision"], 1, payload, "source-checkpoint"
        )
        capture_jobs_module._SCHEMA_READY.clear()
        status, read = request(
            host,
            port,
            "GET",
            f"/v1/capture-jobs/{job['job_id']}?provider=chatgpt&scope={SCOPE_QUERY}&client_protocol=2",
            {},
        )
        assert status == 200
        assert read["job"]["checkpoint"] == saved["job"]["checkpoint"]
        descriptor = _artifact_descriptor({"job": read["job"], "lease": adopted["lease"]}, 1, canonical_digest(payload))
        http_connection = HTTPConnection(host, port)
        try:
            http_connection.request(
                "GET",
                f"/v1/capture-jobs/{job['job_id']}/checkpoint-artifacts/{canonical_digest(payload)}",
                headers={"Authorization": f"Bearer {TOKEN}", "X-Polylogue-Checkpoint": json.dumps(descriptor)},
            )
            response = http_connection.getresponse()
            assert response.status == 200
            retained_bytes = response.read()
            assert retained_bytes == canonical_json(payload).encode()
            retained = json.loads(retained_bytes)["queue"][0]["envelope"]
        finally:
            http_connection.close()
        assert retained == original
        assert parse_payload(Provider.CHATGPT, retained, "source-only-in-registry") == parse_payload(
            Provider.CHATGPT, original, "source-only-in-registry"
        )
        assert not list(tmp_path.rglob("*.json"))


def test_update_receipt_requires_the_current_complete_digest(tmp_path: Path) -> None:
    """A request ID cannot certify a stored receipt for a different digest."""
    with receiver(tmp_path) as (host, port):
        job = create(host, port)
        adopted = adopt(host, port, job)
        retry = {
            "state": "retry_wait",
            "attempt": 2,
            "reason": "rate-limit",
            "next_eligible_at": "2026-01-01T00:00:00Z",
        }
        update_body: dict[str, object] = {
            "provider": "chatgpt",
            "scope": ACCOUNT_SCOPE,
            "lease_id": adopted["lease"]["lease_id"],
            "generation": adopted["lease"]["generation"],
            "proof": adopted["lease"]["proof"],
            "request_id": "current-retry",
            "expected_revision": adopted["job"]["revision"],
            "retry": retry,
        }
        status, updated = request(host, port, "POST", f"/v1/capture-jobs/{job['job_id']}/update", update_body)
        assert status == 200 and updated["duplicate"] is False

        status, replay = request(host, port, "POST", f"/v1/capture-jobs/{job['job_id']}/update", update_body)
        assert status == 200 and replay["duplicate"] is True
        assert replay["receipt"] == updated["receipt"]

        incomplete = canonical_digest({"retry": retry, "lease_ttl_seconds": None})
        with sqlite3.connect(capture_job_database_path(tmp_path)) as connection:
            connection.execute(
                "UPDATE capture_job_update_receipts SET request_digest=? WHERE request_id=?",
                (incomplete, "current-retry"),
            )
        status, replay = request(host, port, "POST", f"/v1/capture-jobs/{job['job_id']}/update", update_body)
        assert status == 409 and replay["error"]["code"] == "request_id_conflict"

        status, conflicting = request(
            host,
            port,
            "POST",
            f"/v1/capture-jobs/{job['job_id']}/update",
            {**update_body, "retention": {"state": "held", "hold_reason": "operator", "timeline_authoritative": True}},
        )
        assert status == 409 and conflicting["error"]["code"] == "request_id_conflict"


def test_terminal_retry_transitions_retention_without_a_client_declaration(tmp_path: Path) -> None:
    """Anti-vacuity: returning ``current`` unchanged from _retention_after_retry
    leaves the job ``active`` and this assertion fails.

    No production client sends a retention object, so the terminal transition
    is the only route out of the creation default.
    """
    with receiver(tmp_path) as (host, port):
        job = create(host, port)
        adopted = adopt(host, port, job)
        checkpointed = _checkpoint(
            host, port, job["job_id"], adopted["lease"], adopted["job"]["revision"], 0, {"cursor": 1}, "cp-1"
        )
        status, completed = request(
            host,
            port,
            "POST",
            f"/v1/capture-jobs/{job['job_id']}/update",
            {
                "provider": "chatgpt",
                "scope": ACCOUNT_SCOPE,
                "lease_id": adopted["lease"]["lease_id"],
                "generation": adopted["lease"]["generation"],
                "proof": adopted["lease"]["proof"],
                "request_id": "terminal",
                "expected_revision": checkpointed["job"]["revision"],
                "retry": {"state": "completed", "attempt": 1, "reason": None, "next_eligible_at": None},
            },
        )
        assert status == 200
        assert completed["receipt"]["retention"]["state"] == "eligible"
        # The checkpoint left an intent-keyed timeline, so this job is the
        # record of it and housekeeping must not collect it.
        assert completed["receipt"]["retention"]["timeline_authoritative"] is True
        assert housekeeping(host, port, tmp_path, now=datetime(2050, 1, 1, tzinfo=UTC)) == []


def test_checkpoint_persists_a_timeline_the_projection_surfaces(tmp_path: Path) -> None:
    """Anti-vacuity: deleting the _append_event call in checkpoint() empties
    ``timelines``, because no production client posts to the event route and
    the ``created`` event carries no conversation ref.
    """
    with receiver(tmp_path) as (host, port):
        job = create(host, port)
        adopted = adopt(host, port, job)
        named = _checkpoint(
            host,
            port,
            job["job_id"],
            adopted["lease"],
            adopted["job"]["revision"],
            0,
            {"cursor": 1, "conversation_ref": "conversation:7"},
            "cp-named",
        )
        _checkpoint(
            host, port, job["job_id"], adopted["lease"], named["job"]["revision"], 1, {"cursor": 2}, "cp-unnamed"
        )
        status, page = request(
            host,
            port,
            "GET",
            f"/v1/capture-jobs/{job['job_id']}/events?provider=chatgpt&scope={SCOPE_QUERY}&client_protocol=2",
            {},
        )
        assert status == 200
        assert [event["kind"] for event in page["events"]] == ["created", "capture-attempted", "capture-attempted"]
        assert set(page["timelines"]) == {"conversation:7", f"intent:{INTENT_KEY}"}
        assert [event["payload"]["checkpoint_sequence"] for event in page["timelines"]["conversation:7"]] == [0]


def test_event_page_holds_the_newest_events_and_pages_backwards(tmp_path: Path) -> None:
    """Anti-vacuity: restoring ``ORDER BY event_revision LIMIT`` (oldest-first)
    drops the newest checkpoints from a short page, and the timeline
    projection with them.
    """
    with receiver(tmp_path) as (host, port):
        job = create(host, port)
        adopted = adopt(host, port, job)
        revision = adopted["job"]["revision"]
        for sequence in range(4):
            body = _checkpoint(
                host,
                port,
                job["job_id"],
                adopted["lease"],
                revision,
                sequence,
                {"cursor": sequence, "conversation_ref": f"conversation:{sequence}"},
                f"cp-{sequence}",
            )
            revision = body["job"]["revision"]

        query = f"provider=chatgpt&scope={SCOPE_QUERY}&client_protocol=2"
        status, page = request(host, port, "GET", f"/v1/capture-jobs/{job['job_id']}/events?{query}&limit=2", {})
        assert status == 200
        assert page["has_more"] is True
        assert [event["payload"]["checkpoint_sequence"] for event in page["events"]] == [2, 3]
        assert set(page["timelines"]) == {"conversation:2", "conversation:3"}

        status, older = request(
            host,
            port,
            "GET",
            f"/v1/capture-jobs/{job['job_id']}/events?{query}&limit=2&before_revision={page['next_before_revision']}",
            {},
        )
        assert status == 200
        assert [event["payload"]["checkpoint_sequence"] for event in older["events"]] == [0, 1]
        # `created` is older still, so the walk has one more page to go.
        assert older["has_more"] is True

        status, oldest = request(
            host,
            port,
            "GET",
            f"/v1/capture-jobs/{job['job_id']}/events?{query}&limit=2&before_revision={older['next_before_revision']}",
            {},
        )
        assert status == 200
        assert [event["kind"] for event in oldest["events"]] == ["created"]
        assert oldest["has_more"] is False
        assert oldest["next_before_revision"] is None


def _event_body(job_id: str, adopted: dict[str, Any], *, request_id: str, payload: dict[str, object]) -> dict[str, Any]:
    return {
        "provider": "chatgpt",
        "scope": ACCOUNT_SCOPE,
        "request_id": request_id,
        "expected_revision": adopted["job"]["revision"],
        "lease_id": adopted["lease"]["lease_id"],
        "generation": adopted["lease"]["generation"],
        "proof": adopted["lease"]["proof"],
        "kind": "first-seen",
        "refs": {"conversation_ref": "conversation:1"},
        "payload": payload,
    }


def test_event_payload_and_refs_larger_than_sixty_four_kib_remain_exact(tmp_path: Path) -> None:
    with receiver(tmp_path) as (host, port):
        job = create(host, port)
        adopted = adopt(host, port, job)
        text = "x" * (65 * 1024)
        body = _event_body(job["job_id"], adopted, request_id="large-event", payload={"detail": text})
        body["refs"] = {"conversation_ref": "conversation:" + text}
        status, accepted = request(host, port, "POST", f"/v1/capture-jobs/{job['job_id']}/events", body)
        assert status == 200
        assert accepted["event"]["payload"] == body["payload"]
        assert accepted["event"]["refs"] == body["refs"]
        status, page = request(
            host,
            port,
            "GET",
            f"/v1/capture-jobs/{job['job_id']}/events?provider=chatgpt&scope={SCOPE_QUERY}&client_protocol=2&limit=1",
            {},
        )
        assert status == 200
        assert page["events"][0]["payload"] == body["payload"]
        assert page["events"][0]["refs"] == body["refs"]


def test_checkpoint_publication_continues_after_ten_thousand_events(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, frozen_clock: object
) -> None:
    from polylogue.browser_capture import receiver as receiver_module

    identity_path = tmp_path / "receiver-identity.json"
    monkeypatch.setattr(receiver_module, "browser_capture_receiver_identity_path", lambda: identity_path)
    with receiver(tmp_path) as (host, port):
        job = create(host, port)
        adopted = adopt(host, port, job)
        registry = CaptureJobRegistry(tmp_path, receiver_module.load_or_mint_receiver_identity(identity_path))
        current_revision = adopted["job"]["revision"]
        # The invariant is admission after ten thousand retained events, not
        # ten thousand unrelated filesystem commits during fixture setup.
        with registry._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            for sequence in range(10_000):
                registry._append_event(
                    connection,
                    job["job_id"],
                    "capture-attempted",
                    f"historical-event-{sequence}",
                    current_revision,
                    {"conversation_ref": "conversation:continued"},
                    {"checkpoint_sequence": sequence},
                    advance_revision=False,
                )
        with sqlite3.connect(capture_job_database_path(tmp_path)) as connection:
            assert (
                connection.execute(
                    "SELECT count(*) FROM capture_job_events WHERE job_id=?", (job["job_id"],)
                ).fetchone()[0]
                == 10_001
            )
        payload = {"conversation_ref": "conversation:" + "c" * (65 * 1024), "progress": "acquired"}
        body = {
            "provider": "chatgpt",
            "scope": ACCOUNT_SCOPE,
            "client_protocol": 2,
            "request_id": "after-many-events",
            "expected_revision": current_revision,
            "lease_id": adopted["lease"]["lease_id"],
            "generation": adopted["lease"]["generation"],
            "proof": adopted["lease"]["proof"],
            "checkpoint": {"sequence": 10_000, "payload": payload, "digest": canonical_digest(payload)},
        }
        status, accepted = request(host, port, "PUT", f"/v1/capture-jobs/{job['job_id']}/checkpoint", body)
        assert status == 200
        assert accepted["receipt"]["checkpoint_sequence"] == 10_000
        status, page = request(
            host,
            port,
            "GET",
            f"/v1/capture-jobs/{job['job_id']}/events?provider=chatgpt&scope={SCOPE_QUERY}&client_protocol=2&limit=1",
            {},
        )
        assert status == 200 and page["has_more"]
        assert page["events"][0]["event_revision"] == 10_001
        assert page["events"][0]["refs"]["conversation_ref"] == payload["conversation_ref"]


def test_capture_job_routes_do_not_inherit_the_general_control_body_bound(tmp_path: Path) -> None:
    """Restoring the former owner cap refuses a valid state-bearing event."""
    with receiver(tmp_path) as (host, port):
        job = create(host, port)
        adopted = adopt(host, port, job)
        body = _event_body(
            job["job_id"],
            adopted,
            request_id="oversize-body",
            payload={"blob": "x" * (1024 * 1024 + 1024)},
        )
        status, accepted = request(host, port, "POST", f"/v1/capture-jobs/{job['job_id']}/events", body)
        assert status == 200 and accepted["event"]["payload"] == body["payload"]


def test_checkpoint_after_terminal_update_is_kept(tmp_path: Path) -> None:
    """The production call order must not make a job collect its own timeline.

    ``browser-extension/src/background/runtime.js`` calls ``update()`` and then
    ``checkpoint()``. For a job already at a terminal retry state,
    ``_retention_after_retry`` therefore ran while the job held no timeline
    event and recorded ``eligible``/``timeline_authoritative=false`` -- and
    checkpointing is the only route that ever creates a timeline event, so the
    one it appended a moment later never revised that verdict. Housekeeping
    then deleted the fresh checkpoint, its receipts and the timeline itself
    once the lease expired.

    Anti-vacuity: the second half drives the terminal update with no
    checkpoint at all and the job must still record
    ``timeline_authoritative=false`` -- the recomputation is conditional on
    retained timeline evidence, not an unconditional claim, and
    ``_retention_after_retry`` is unchanged.
    """
    with receiver(tmp_path) as (host, port):
        job = create(host, port)
        adopted = adopt(host, port, job)
        status, completed = request(
            host,
            port,
            "POST",
            f"/v1/capture-jobs/{job['job_id']}/update",
            {
                "provider": "chatgpt",
                "scope": ACCOUNT_SCOPE,
                "lease_id": adopted["lease"]["lease_id"],
                "generation": adopted["lease"]["generation"],
                "proof": adopted["lease"]["proof"],
                "request_id": "terminal-first",
                "expected_revision": adopted["job"]["revision"],
                "retry": {"state": "completed", "attempt": 1, "reason": None, "next_eligible_at": None},
            },
        )
        assert status == 200
        # The receiver has no timeline evidence yet, so it correctly records
        # the job as non-authoritative at this instant.
        assert completed["receipt"]["retention"] == {
            "state": "eligible",
            "hold_reason": None,
            "timeline_authoritative": False,
        }

        checkpointed = _checkpoint(
            host,
            port,
            job["job_id"],
            adopted["lease"],
            completed["job"]["revision"],
            0,
            {"cursor": 1, "conversation_ref": "conversation:9"},
            "cp-after-terminal",
        )
        assert checkpointed["job"]["retention"]["state"] == "eligible"
        assert checkpointed["job"]["retention"]["timeline_authoritative"] is True
        assert housekeeping(host, port, tmp_path, now=datetime(2050, 1, 1, tzinfo=UTC)) == []
        status, page = request(
            host,
            port,
            "GET",
            f"/v1/capture-jobs/{job['job_id']}/events?provider=chatgpt&scope={SCOPE_QUERY}&client_protocol=2",
            {},
        )
        assert status == 200
        assert set(page["timelines"]) == {"conversation:9"}

    with receiver(tmp_path / "second") as (host, port):
        job = create(host, port)
        adopted = adopt(host, port, job)
        status, abandoned = request(
            host,
            port,
            "POST",
            f"/v1/capture-jobs/{job['job_id']}/update",
            {
                "provider": "chatgpt",
                "scope": ACCOUNT_SCOPE,
                "lease_id": adopted["lease"]["lease_id"],
                "generation": adopted["lease"]["generation"],
                "proof": adopted["lease"]["proof"],
                "request_id": "terminal-only",
                "expected_revision": adopted["job"]["revision"],
                "retry": {"state": "abandoned", "attempt": 5, "reason": "gave up", "next_eligible_at": None},
            },
        )
        assert status == 200
        assert abandoned["receipt"]["retention"] == {
            "state": "eligible",
            "hold_reason": None,
            "timeline_authoritative": False,
        }
        status, refetched = request(
            host,
            port,
            "GET",
            f"/v1/capture-jobs/{job['job_id']}?provider=chatgpt&scope={SCOPE_QUERY}&client_protocol=2",
            {},
        )
        assert status == 200
        assert refetched["job"]["retention"]["timeline_authoritative"] is False


def test_explicit_default_retention_is_durable_declaration(tmp_path: Path) -> None:
    """Anti-vacuity: value equality must not erase an explicit client declaration."""
    with receiver(tmp_path) as (host, port):
        job = create(host, port)
        adopted = adopt(host, port, job)
        body = {
            "provider": "chatgpt",
            "scope": ACCOUNT_SCOPE,
            "lease_id": adopted["lease"]["lease_id"],
            "generation": adopted["lease"]["generation"],
            "proof": adopted["lease"]["proof"],
            "expected_revision": adopted["job"]["revision"],
            "request_id": "declare-default-retention",
            "retention": {"state": "active", "hold_reason": None, "timeline_authoritative": True},
        }
        status, declared = request(host, port, "POST", f"/v1/capture-jobs/{job['job_id']}/update", body)
        assert status == 200
        assert declared["duplicate"] is False

        body.update(
            request_id="terminal-after-declaration",
            expected_revision=declared["job"]["revision"],
            retention=None,
            retry={"state": "completed", "attempt": 1, "reason": None, "next_eligible_at": None},
        )
        body.pop("retention")
        status, terminal = request(host, port, "POST", f"/v1/capture-jobs/{job['job_id']}/update", body)
        assert status == 200
        assert terminal["job"]["retention"]["state"] == "active"


def _retired_job(host: str, port: int) -> str:
    """Drive one job to completed, eligible, non-authoritative and checkpointed."""
    job = create(host, port)
    adopted = adopt(host, port, job)
    lease = adopted["lease"]
    base = {
        "provider": "chatgpt",
        "scope": ACCOUNT_SCOPE,
        "lease_id": lease["lease_id"],
        "generation": lease["generation"],
        "proof": lease["proof"],
    }
    checkpointed = _checkpoint(host, port, job["job_id"], lease, adopted["job"]["revision"], 0, {"cursor": 1}, "cp")
    status, retained = request(
        host,
        port,
        "POST",
        f"/v1/capture-jobs/{job['job_id']}/update",
        {
            **base,
            "request_id": "declare-eligible",
            "expected_revision": checkpointed["job"]["revision"],
            "retention": {"state": "eligible", "hold_reason": None, "timeline_authoritative": False},
        },
    )
    assert status == 200
    status, _completed = request(
        host,
        port,
        "POST",
        f"/v1/capture-jobs/{job['job_id']}/update",
        {
            **base,
            "request_id": "complete",
            "expected_revision": retained["job"]["revision"],
            "retry": {"state": "completed", "attempt": 1, "reason": None, "next_eligible_at": None},
        },
    )
    assert status == 200
    return cast(str, job["job_id"])


def _job_row_counts(spool_path: Path, job_id: str) -> dict[str, int]:
    with sqlite3.connect(capture_job_database_path(spool_path)) as connection:
        return {
            table: connection.execute(f"SELECT COUNT(*) FROM {table} WHERE job_id=?", (job_id,)).fetchone()[0]
            for table in (
                "capture_jobs",
                "capture_job_events",
                "capture_job_receipts",
                "capture_job_update_receipts",
            )
        }


def test_exact_orphan_inspection_preserves_pause_and_acquired_custody(tmp_path: Path) -> None:
    root = tmp_path / "backfill-checkpoints"
    root.mkdir()
    raw = b'{ "extension_instance_id":"old", "checkpoint":{"jobs":[{"status":"paused"}],"queue":[{"body_ref":"acquired-body"}]}}\n'
    path = root / "old.json"
    path.write_bytes(raw)
    digest = "sha256:" + hashlib.sha256(raw).hexdigest()
    route = f"/v1/capture-jobs/orphans/{digest}/payload?client_protocol=2"
    with receiver(tmp_path) as (host, port):
        status, census = request(host, port, "GET", "/v1/capture-jobs/orphans?client_protocol=2", {})
        assert status == 200
        assert census["orphans"][0]["source_digest"] == digest
        assert census["orphans"][0]["orphan_kind"] == "legacy_backfill_checkpoint"
        connection = HTTPConnection(host, port)
        connection.request("GET", route, headers={"Authorization": f"Bearer {TOKEN}"})
        response = connection.getresponse()
        assert response.status == 200
        length = response.getheader("Content-Length")
        assert length is not None and int(length) == len(raw)
        assert response.read() == raw
        connection.close()
        connection = HTTPConnection(host, port)
        connection.request("GET", route)
        response = connection.getresponse()
        assert response.status == 401
        response.read()
        connection.close()
        assert request(host, port, "GET", "/v1/backfill-checkpoint?extension_instance_id=old", {})[0] == 404
        assert request(host, port, "POST", "/v1/backfill-checkpoint", {})[0] == 404
    assert path.read_bytes() == raw
    assert browser_capture_route_contract_for("GET", route.split("?")[0]) is not None
    assert browser_capture_route_contract_for("GET", "/v1/backfill-checkpoint") is None
    assert browser_capture_route_contract_for("POST", "/v1/backfill-checkpoint") is None


def test_orphan_digest_never_selects_replaced_or_missing_bytes(tmp_path: Path) -> None:
    root = tmp_path / "backfill-checkpoints"
    root.mkdir()
    path = root / "old.json"
    original = b'{"checkpoint":{"jobs":[]}}'
    path.write_bytes(original)
    digest = "sha256:" + hashlib.sha256(original).hexdigest()
    registry = CaptureJobRegistry(tmp_path, "synthetic-receiver")
    registry.list_orphans(2)
    path.write_bytes(b'{"checkpoint":{"jobs":[{"status":"paused"}]}}')
    with pytest.raises(CaptureJobError) as refused:
        with registry.inspect_orphan(digest, 2):
            pytest.fail("changed evidence must not be returned under its old digest")
    assert refused.value.code == "orphan_payload_not_found"
    assert path.exists()


def _artifact_descriptor(adopted: dict[str, Any], sequence: int, digest: str) -> dict[str, object]:
    return {
        "provider": "chatgpt",
        "scope": ACCOUNT_SCOPE,
        "client_protocol": 2,
        "request_id": str(uuid5(NAMESPACE_URL, f"artifact:{sequence}:{digest}")),
        "expected_revision": adopted["job"]["revision"],
        "lease_id": adopted["lease"]["lease_id"],
        "generation": adopted["lease"]["generation"],
        "proof": adopted["lease"]["proof"],
        "sequence": sequence,
        "digest": digest,
    }


@pytest.mark.parametrize("value", [None, True, 9, "é", [1, {"a": "x"}], {"\ue000": 1, "\U00010000": 2}])
def test_checkpoint_artifact_accepts_generic_canonical_shapes(tmp_path: Path, value: object) -> None:
    """Narrowing the streamed body to a ledger or changing canonical bytes fails."""
    with receiver(tmp_path) as (host, port):
        job = create(host, port)
        adopted = adopt(host, port, job)
        digest = canonical_digest(value)
        descriptor = _artifact_descriptor(adopted, 0, digest)
        raw = canonical_json(value).encode()
        connection = HTTPConnection(host, port)
        headers = {"Authorization": f"Bearer {TOKEN}", "X-Polylogue-Checkpoint": json.dumps(descriptor)}
        connection.request("PUT", f"/v1/capture-jobs/{job['job_id']}/checkpoint", raw, headers)
        response = connection.getresponse()
        assert response.status == 200
        acknowledged = json.loads(response.read())
        assert acknowledged["job"]["checkpoint"] == {
            "sequence": 0,
            "digest": digest,
            "artifact_ref": digest,
            "size_bytes": len(raw),
        }
        connection.request("GET", f"/v1/capture-jobs/{job['job_id']}/checkpoint-artifacts/{digest}", headers=headers)
        response = connection.getresponse()
        assert response.status == 200
        assert response.read() == raw
        connection.close()


@pytest.mark.parametrize(
    "raw",
    [
        b'{"a": "x","b":1}',
        b'{"a":"\\u0078","b":1}',
        b'{"a":"x","b":1e0}',
        b'{"b":1,"a":"x"}',
        b'{"a":"x","b":1,"b":1}',
    ],
)
@pytest.mark.parametrize("digest_source", ["canonical", "wire"])
def test_checkpoint_artifact_refuses_alternate_bytes_before_ack(tmp_path: Path, raw: bytes, digest_source: str) -> None:
    """A semantic-only hash would ACK a body different from the declared bytes."""
    with receiver(tmp_path) as (host, port):
        job = create(host, port)
        adopted = adopt(host, port, job)
        declared_digest = (
            canonical_digest({"a": "x", "b": 1})
            if digest_source == "canonical"
            else "sha256:" + hashlib.sha256(raw).hexdigest()
        )
        descriptor = _artifact_descriptor(adopted, 0, declared_digest)
        connection = HTTPConnection(host, port)
        connection.request(
            "PUT",
            f"/v1/capture-jobs/{job['job_id']}/checkpoint",
            raw,
            {"Authorization": f"Bearer {TOKEN}", "X-Polylogue-Checkpoint": json.dumps(descriptor)},
        )
        response = connection.getresponse()
        assert response.status == 400
        refusal = json.loads(response.read())
        assert refusal["error"]["code"] in {
            "checkpoint_digest_mismatch",
            "checkpoint_noncanonical_key_order",
            "non_canonical_json",
        }
        connection.close()
        status, found = request(
            host,
            port,
            "GET",
            f"/v1/capture-jobs/{job['job_id']}?provider=chatgpt&scope={SCOPE_QUERY}&client_protocol=2",
            {},
        )
        assert status == 200 and found["job"]["checkpoint"] is None
        assert found["job"]["revision"] == adopted["job"]["revision"]
        assert not list((capture_job_store_root(tmp_path) / "artifacts").glob("*.checkpoint"))


def test_checkpoint_artifact_hash_releases_writer_transaction_before_reading_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Moving verification into admission would block unrelated job controls."""
    with receiver(tmp_path) as (host, port):
        job = create(host, port)
        adopted = adopt(host, port, job)
        payload = {"retained": "original"}
        digest = canonical_digest(payload)
        descriptor = _artifact_descriptor(adopted, 0, digest)
        status, _ = request(
            host,
            port,
            "PUT",
            f"/v1/capture-jobs/{job['job_id']}/checkpoint",
            {
                **descriptor,
                "checkpoint": {"sequence": 0, "digest": digest, "payload": payload},
            },
        )
        assert status == 200
        from polylogue.browser_capture.receiver import load_or_mint_receiver_identity

        registry = CaptureJobRegistry(tmp_path, load_or_mint_receiver_identity())
        file_digest = hashlib.file_digest
        writer_admissions = []

        def verify_without_writer_lock(stream: Any, algorithm: str) -> Any:
            with sqlite3.connect(capture_job_database_path(tmp_path), timeout=0) as connection:
                connection.execute("BEGIN IMMEDIATE")
                writer_admissions.append(True)
                connection.rollback()
            return file_digest(stream, algorithm)

        monkeypatch.setattr(hashlib, "file_digest", verify_without_writer_lock)
        with registry.checkpoint_artifact(job["job_id"], digest, descriptor) as (stream, size):
            assert stream.read() == canonical_json(payload).encode()
            assert size == len(canonical_json(payload).encode())
        assert writer_admissions == [True]


def test_discovery_pages_all_equal_clock_jobs_through_actual_http_and_restart(tmp_path: Path) -> None:
    with receiver(tmp_path) as (host, port):
        expected = set()
        for index in range(61):
            payload = {"cutoff": "2026-01-01T00:00:00Z", "collection": index}
            status, result = request(
                host,
                port,
                "POST",
                "/v1/capture-jobs",
                {
                    "provider": "chatgpt",
                    "scope": ACCOUNT_SCOPE,
                    "request_id": f"create-page-{index}",
                    "intent": {
                        "schema_version": 1,
                        "version": 1,
                        "intent_key": f"i1:synthetic-{index}",
                        "payload": payload,
                        "digest": canonical_digest(payload),
                    },
                },
            )
            assert status == 201
            expected.add(result["job"]["job_id"])
        observed: list[str] = []
        cursor = None
        page_sizes = []
        while True:
            status, result = request(
                host,
                port,
                "POST",
                "/v1/capture-jobs/discover",
                {
                    "provider": "chatgpt",
                    "scope": ACCOUNT_SCOPE,
                    **({"cursor": cursor} if cursor else {}),
                },
            )
            assert status == 200 and result["total"] == 61
            page_sizes.append(len(result["jobs"]))
            observed.extend(job["job_id"] for job in result["jobs"])
            cursor = cast(str | None, result["cursor"])
            assert result["has_more"] is (cursor is not None)
            if cursor is None:
                break
            # Reading from another owner instance uses the identical immutable cursor.
            restarted = CaptureJobRegistry(tmp_path, "synthetic-receiver-key")
            with restarted.result_scope():
                page = restarted.discover(
                    {"provider": "chatgpt", "scope": ACCOUNT_SCOPE, "client_protocol": 2, "cursor": cursor}
                )
                page_job_ids = {job["job_id"] for job in cast(list[dict[str, Any]], page["jobs"])}
            assert page_job_ids.issubset(expected)
        assert page_sizes == [25, 25, 11]
        assert len(observed) == len(set(observed)) == 61
        assert set(observed) == expected


def test_orphan_census_pages_retained_custody_and_unreadable_paths(tmp_path: Path) -> None:
    root = tmp_path / "backfill-checkpoints"
    root.mkdir()
    for index in range(61):
        (root / f"synthetic-{index}.json").write_text(json.dumps({"checkpoint": {"entry": index}}))
    registry = CaptureJobRegistry(tmp_path, "synthetic-receiver-key")
    observed: list[str] = []
    cursor = None
    sizes = []
    while True:
        result = registry.list_orphans(2, cursor)
        assert result["total"] == 61
        orphans = cast(list[dict[str, Any]], result["orphans"])
        sizes.append(len(orphans))
        observed.extend(entry["source_digest"] for entry in orphans)
        cursor = cast(str | None, result["cursor"])
        if cursor is None:
            break
        registry = CaptureJobRegistry(tmp_path, "synthetic-receiver-key")
    assert sizes == [25, 25, 11]
    assert len(observed) == len(set(observed)) == 61
    assert len(list(root.glob("*.json"))) == 61


@pytest.mark.parametrize("authority_change", ["none", "takeover", "pause", "producer_cancel"])
@pytest.mark.frozen_clock_modules("polylogue.browser_capture.capture_jobs")
def test_native_upload_preserves_paused_producer_and_fences_changed_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, authority_change: str, frozen_clock: Any
) -> None:
    """Real partial-body reads resume beyond expiry only under unchanged authority."""
    from polylogue.browser_capture.capture_stream import STAGING_DIRNAME
    from polylogue.browser_capture.receiver import receiver_identity

    first_chunk = Event()
    original_progress = CaptureJobRegistry.artifact_progress

    @contextmanager
    def observed_progress(
        registry: CaptureJobRegistry, job_id: str, body: dict[str, object], *, native: bool = False
    ) -> Iterator[Any]:
        with original_progress(registry, job_id, body, native=native) as progress:

            def observe() -> None:
                progress()
                first_chunk.set()

            yield observe

    monkeypatch.setattr(CaptureJobRegistry, "artifact_progress", observed_progress)
    server = make_server("127.0.0.1", 0, spool_path=tmp_path, auth_token=TOKEN)
    registry = CaptureJobRegistry(tmp_path, receiver_identity(server.config))
    scope = {"kind": "account", "key": SCOPE}
    intent_payload = {"provider_session_id": "native-pause"}
    scoped: dict[str, object] = {"provider": "chatgpt", "scope": scope, "client_protocol": 2}
    with registry.result_scope():
        _status, created = registry.create(
            {
                **scoped,
                "intent": {
                    "schema_version": 1,
                    "version": 1,
                    "intent_key": INTENT_KEY,
                    "payload": intent_payload,
                    "digest": canonical_digest(intent_payload),
                },
            }
        )
        job = cast(dict[str, Any], created["job"])
        job_id = job["job_id"]
        adopted = registry.adopt(
            job_id,
            {
                **scoped,
                "request_id": "initial-owner",
                "session_id": "profile-a",
                "expected_revision": job["revision"],
                "expected_lease_generation": 0,
                "lease_ttl_seconds": 1,
            },
        )
        lease = cast(dict[str, Any], adopted["lease"])
        adopted_job = cast(dict[str, Any], adopted["job"])
        adopted_revision = adopted_job["revision"]
    descriptor: dict[str, object] = {
        **scoped,
        "lease_id": lease["lease_id"],
        "generation": lease["generation"],
        "proof": lease["proof"],
        "expected_revision": adopted_revision,
        "acquisition_id": str(uuid4()),
        "binding": {
            "preparation_instance_id": "owned-test-instance",
            "extension_instance_id": None,
            "acquisition_sequence": None,
            "invocation_id": None,
            "raw_revision": "raw-owned-test",
            "native_id": "native-pause",
            "source_url": "https://chatgpt.com/c/native-pause",
            "document_id": "owned-test-document",
        },
    }
    with registry.result_scope():
        registry.native_begin(job_id, {**descriptor, "member_names": ["conversation"]})
    raw = b'{"id":"native-pause","mapping":{}}'
    descriptor.update(
        member_name="conversation", metadata={}, sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw)
    )
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with socket.create_connection(("127.0.0.1", server.server_port), timeout=10) as producer:
            producer.sendall(
                (
                    f"PUT /v1/capture-jobs/{job_id}/native/member HTTP/1.1\r\n"
                    f"Host: 127.0.0.1:{server.server_port}\r\nAuthorization: Bearer {TOKEN}\r\n"
                    f"X-Polylogue-Native: {json.dumps(descriptor)}\r\nContent-Length: {len(raw)}\r\n\r\n"
                ).encode()
                + raw[:1]
            )
            assert first_chunk.wait(10), "the real staging reader must consume the prefix before the pause"
            assert len(list((tmp_path / STAGING_DIRNAME).iterdir())) == 1
            if authority_change == "pause":
                with registry.result_scope():
                    registry.update(
                        job_id,
                        {**descriptor, "request_id": "pause-owner", "retry": {"state": "held", "attempt": 0}},
                    )
            frozen_clock.advance(600)
            if authority_change == "takeover":
                with registry.result_scope():
                    registry.adopt(
                        job_id,
                        {
                            **scoped,
                            "request_id": "replacement-owner",
                            "session_id": "profile-b",
                            "expected_revision": descriptor["expected_revision"],
                            "expected_lease_generation": lease["generation"],
                        },
                    )
            if authority_change == "producer_cancel":
                producer.shutdown(socket.SHUT_WR)
            else:
                producer.sendall(raw[1:])
            response = HTTPResponse(producer)
            response.begin()
            result = json.loads(response.read())
            if authority_change == "none":
                assert response.status == 200
                assert (result["sha256"], result["size_bytes"]) == (hashlib.sha256(raw).hexdigest(), len(raw))
                with registry.native_member_artifact(job_id, "conversation", descriptor) as (handle, _member):
                    assert handle.read() == raw
            else:
                assert response.status == (400 if authority_change == "producer_cancel" else 409)
                if authority_change == "producer_cancel":
                    assert result["error"] == "incomplete_body"
                else:
                    assert (
                        result["error"]["code"]
                        == {"takeover": "lease_replaced", "pause": "capture_authority_paused"}[authority_change]
                    )
            assert list((tmp_path / STAGING_DIRNAME).iterdir()) == []
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)
        assert not thread.is_alive()


def _native_descriptor(adopted: dict[str, Any], acquisition_id: str) -> dict[str, object]:
    return {
        "provider": adopted["job"]["provider"],
        "scope": ACCOUNT_SCOPE,
        "client_protocol": 2,
        "request_id": str(uuid4()),
        "expected_revision": adopted["job"]["revision"],
        "lease_id": adopted["lease"]["lease_id"],
        "generation": adopted["lease"]["generation"],
        "proof": adopted["lease"]["proof"],
        "acquisition_id": acquisition_id,
    }


def _native_bytes(
    host: str, port: int, path: str, descriptor: dict[str, object], raw: bytes
) -> tuple[int, dict[str, Any]]:
    connection = HTTPConnection(host, port)
    connection.request(
        "PUT", path, raw, {"Authorization": f"Bearer {TOKEN}", "X-Polylogue-Native": json.dumps(descriptor)}
    )
    response = connection.getresponse()
    try:
        return response.status, json.loads(response.read())
    finally:
        connection.close()


def _retain_native_members(
    host: str, port: int, provider: str, native_id: str, members: dict[str, bytes]
) -> tuple[str, dict[str, object], dict[str, object]]:
    job = create(host, port, provider)
    adopted = adopt(host, port, job)
    descriptor = _native_descriptor(adopted, str(uuid4()))
    receipts = {
        name: {"sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)} for name, raw in members.items()
    }
    source_url = f"https://{'chatgpt.com' if provider == 'chatgpt' else 'claude.ai' if provider == 'claude-ai' else 'grok.com'}/c/{native_id}"
    binding = {
        "preparation_instance_id": "preparation-owner",
        "extension_instance_id": None,
        "acquisition_sequence": None,
        "invocation_id": None,
        "raw_revision": canonical_digest({"provider": provider, "native_id": native_id, "members": receipts}),
        "native_id": native_id,
        "source_url": source_url,
        "document_id": "original-document",
    }
    path = f"/v1/capture-jobs/{job['job_id']}/native"
    status, _ = request(
        host, port, "POST", f"{path}/begin", {**descriptor, "binding": binding, "member_names": list(members)}
    )
    assert status == 200
    for name, raw in members.items():
        status, _ = _native_bytes(
            host, port, f"{path}/member", {**descriptor, "member_name": name, "metadata": {}, **receipts[name]}, raw
        )
        assert status == 200
    provenance: dict[str, object] = {
        "captured_at": "2026-01-01T00:00:00Z",
        "extension_instance_id": None,
        "acquisition_sequence": None,
        "source_url": source_url,
        "adapter_name": f"{provider}-native-v1",
        "adapter_version": "test",
        "capture_mode": "snapshot",
    }
    return path, descriptor, provenance


def _retain_native_occurrences(host: str, port: int, raw: bytes) -> tuple[str, dict[str, object], dict[str, object]]:
    return _retain_native_members(host, port, "chatgpt", "native-occurrences", {"conversation": raw})


def _prepare_native_occurrences(
    host: str, port: int, raw: bytes
) -> tuple[str, dict[str, object], dict[str, Any], list[dict[str, Any]]]:
    path, descriptor, provenance = _retain_native_occurrences(host, port, raw)
    status, prepared = request(
        host,
        port,
        "POST",
        f"{path}/prepare",
        {**descriptor, "provenance": provenance, "provider_meta": {"capture_fidelity": "native_full"}},
    )
    assert status == 200, prepared
    status, plan = request(host, port, "POST", f"{path}/plan", descriptor)
    assert status == 200
    return path, descriptor, prepared, plan["assets"]


@pytest.mark.parametrize(
    "provider,fixture,native_id",
    [
        ("chatgpt", "chatgpt/native-rich-blocks-v1.json", "native-rich-chatgpt"),
        ("claude-ai", "claude-ai/native-rich-blocks-v1.json", "native-rich-blocks"),
        ("claude-ai", "claude-ai/native-attachment-order.json", "claude-order-fixture"),
        ("grok", "grok/native-bundle.json", "native-conversation"),
    ],
)
def test_native_receiver_complete_envelope_matches_canonical_provider_parsing(
    tmp_path: Path,
    provider: str,
    fixture: str,
    native_id: str,
) -> None:
    from polylogue.core.enums import Provider
    from polylogue.pipeline.ids import session_content_hash
    from polylogue.sources.parsers.browser_capture import parse, parse_native_payload
    from polylogue.storage.sqlite.archive_tiers.write import prepare_session_rows

    original = (Path(__file__).parents[2] / "fixtures" / fixture).read_bytes()
    raw = json.loads(original)
    members = (
        {name: json.dumps(value).encode() for name, value in raw.items()}
        if provider == "grok"
        else {"conversation": original}
    )
    expected = parse_native_payload(
        Provider.from_string(provider), raw, native_id, prepared_attachment_ownership=provider == "chatgpt"
    )
    with receiver(tmp_path) as (host, port):
        path, descriptor, provenance = _retain_native_members(host, port, provider, native_id, members)
        status, prepared = request(
            host,
            port,
            "POST",
            f"{path}/prepare",
            {**descriptor, "provenance": provenance, "provider_meta": {"capture_fidelity": "native_full"}},
        )
        assert status == 200, prepared
        assert prepared["summary"]["title"] is None
        status, plan = request(host, port, "POST", f"{path}/plan", descriptor)
        assert status == 200
        for asset in plan["assets"]:
            metadata = asset["descriptor"]["provider_meta"]
            outcome = {"status": "no_resolvable_source"}
            if metadata.get("native_inline_sha256"):
                outcome = {
                    "status": "retained_native_bytes",
                    "sha256": metadata["native_inline_sha256"],
                    "size_bytes": metadata["native_inline_size_bytes"],
                }
            status, result = request(
                host,
                port,
                "POST",
                f"{path}/asset",
                {
                    **descriptor,
                    "plan_digest": prepared["plan_digest"],
                    "ordinal": asset["ordinal"],
                    "descriptor_digest": asset["descriptor_digest"],
                    "outcome": outcome,
                },
            )
            assert status == 200, result
        status, final = request(
            host, port, "POST", f"{path}/finalize", {**descriptor, "plan_digest": prepared["plan_digest"]}
        )
        assert status == 200, final
    retained = (capture_job_store_root(tmp_path) / "artifacts" / f"{final['sha256']}.native").read_bytes()
    for member in members.values():
        assert member in retained
    envelope = json.loads(retained)
    assert envelope["raw_provider_payload"] == raw
    assert envelope["session"]["title"] == expected.title
    captured = parse(envelope, "capture")
    assert captured.messages == expected.messages
    assert captured.attachments == expected.attachments
    assert session_content_hash(captured) == session_content_hash(expected)
    assert prepare_session_rows(captured).message_rows == prepare_session_rows(expected).message_rows
    assert prepare_session_rows(captured).block_rows == prepare_session_rows(expected).block_rows


@pytest.mark.parametrize("cancel_stage", ["hash", "parse", "output"])
def test_native_preparation_cancellation_fences_actual_work_and_settles_scratch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cancel_stage: str,
) -> None:
    from polylogue.browser_capture import native_preparation
    from polylogue.sources.parsers import browser_capture

    phase = "hash"
    cancelled = False
    original_progress = CaptureJobRegistry.artifact_progress
    original_parse = browser_capture.parse_native_member_streams
    original_output = native_preparation.envelope_prefix

    @contextmanager
    def guarded_progress(
        registry: CaptureJobRegistry, job_id: str, body: dict[str, object], *, native: bool = False
    ) -> Iterator[Callable[[], None]]:
        with original_progress(registry, job_id, body, native=native) as progress:

            def actual_work() -> None:
                nonlocal cancelled
                if native and phase == cancel_stage and not cancelled:
                    registry.native_cancel(job_id, body)
                    cancelled = True
                progress()

            yield actual_work

    def canonical_parse(*args: Any, **kwargs: Any) -> ParsedSession:
        nonlocal phase
        phase = "parse"
        return original_parse(*args, **kwargs)

    def canonical_output(*args: Any, **kwargs: Any) -> Iterator[bytes]:
        nonlocal phase
        phase = "output"
        yield from original_output(*args, **kwargs)

    raw = (Path(__file__).parents[2] / "fixtures/chatgpt/native-duplicate-attachment-occurrences-v1.json").read_bytes()
    with receiver(tmp_path) as (host, port):
        path, descriptor, provenance = _retain_native_occurrences(host, port, raw)
        monkeypatch.setattr(CaptureJobRegistry, "artifact_progress", guarded_progress)
        monkeypatch.setattr(browser_capture, "parse_native_member_streams", canonical_parse)
        monkeypatch.setattr(native_preparation, "envelope_prefix", canonical_output)
        status, result = request(
            host,
            port,
            "POST",
            f"{path}/prepare",
            {**descriptor, "provenance": provenance, "provider_meta": {"capture_fidelity": "native_full"}},
        )
        assert status == 409 and result["error"]["code"] == "native_acquisition_cancelled"
    assert cancelled
    assert list((capture_job_store_root(tmp_path) / "preparation").iterdir()) == []
    with sqlite3.connect(capture_job_database_path(tmp_path)) as connection:
        assert connection.execute("SELECT state FROM capture_job_native_acquisitions").fetchone() == ("cancelled",)
        assert connection.execute("SELECT COUNT(*) FROM capture_job_native_artifacts").fetchone() == (0,)
        assert connection.execute("SELECT COUNT(*) FROM capture_job_native_members").fetchone() == (1,)


@pytest.mark.parametrize("cancel_stage", ["prefix", "asset", "final"])
def test_native_artifact_hash_cancellation_fences_finalization_and_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cancel_stage: str,
) -> None:
    from polylogue.browser_capture.capture_stream import STAGING_DIRNAME

    phase: str | None = None
    cancelled = False
    original_progress = CaptureJobRegistry.artifact_progress
    original_artifact = CaptureJobRegistry.native_artifact

    @contextmanager
    def guarded_progress(
        registry: CaptureJobRegistry, job_id: str, body: dict[str, object], *, native: bool = False
    ) -> Iterator[Callable[[], None]]:
        with original_progress(registry, job_id, body, native=native) as progress:

            def actual_hash_work() -> None:
                nonlocal cancelled
                if native and phase == cancel_stage and not cancelled:
                    registry.native_cancel(job_id, body)
                    cancelled = True
                progress()

            yield actual_hash_work

    @contextmanager
    def observed_artifact(
        registry: CaptureJobRegistry, job_id: str, body: dict[str, object], **kwargs: Any
    ) -> Iterator[tuple[BinaryIO, sqlite3.Row]]:
        nonlocal phase
        phase = kwargs.get("purpose") or "asset"
        try:
            with original_artifact(registry, job_id, body, **kwargs) as artifact:
                yield artifact
        finally:
            phase = None

    raw = (Path(__file__).parents[2] / "fixtures/chatgpt/native-duplicate-attachment-occurrences-v1.json").read_bytes()
    with receiver(tmp_path) as (host, port):
        path, descriptor, prepared, assets = _prepare_native_occurrences(host, port, raw)
        for asset in assets:
            data = f"synthetic occurrence {asset['ordinal']}".encode()
            status, result = _native_bytes(
                host,
                port,
                f"{path}/asset",
                {
                    **descriptor,
                    "plan_digest": prepared["plan_digest"],
                    "ordinal": asset["ordinal"],
                    "descriptor_digest": asset["descriptor_digest"],
                    "outcome": {"status": "acquired"},
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "size_bytes": len(data),
                },
                data,
            )
            assert status == 200, result
        final_body = {**descriptor, "plan_digest": prepared["plan_digest"]}
        if cancel_stage == "final":
            status, final = request(host, port, "POST", f"{path}/finalize", final_body)
            assert status == 200, final
            final_body["sha256"] = final["sha256"]
        monkeypatch.setattr(CaptureJobRegistry, "artifact_progress", guarded_progress)
        monkeypatch.setattr(CaptureJobRegistry, "native_artifact", observed_artifact)
        operation = "publish" if cancel_stage == "final" else "finalize"
        status, result = request(host, port, "POST", f"{path}/{operation}", final_body)
        assert status == 409 and result["error"]["code"] == "native_acquisition_cancelled"
    assert cancelled
    assert list((tmp_path / STAGING_DIRNAME).iterdir()) == []
    with sqlite3.connect(capture_job_database_path(tmp_path)) as connection:
        assert connection.execute(
            "SELECT state, final_receipt_json FROM capture_job_native_acquisitions"
        ).fetchone() == ("cancelled", None)
        assert connection.execute(
            "SELECT COUNT(*) FROM capture_job_native_artifacts WHERE purpose='final'"
        ).fetchone() == ((1 if cancel_stage == "final" else 0),)


@pytest.mark.parametrize(
    "head,role,status,needs_follow_up",
    [
        ("assistant-node", "assistant", "finished_successfully", False),
        ("assistant-node", "assistant", "in_progress", True),
        ("assistant-node", "user", "finished_successfully", True),
        (None, "assistant", "finished_successfully", True),
        ("missing-node", "assistant", "finished_successfully", True),
    ],
)
def test_native_preparation_follow_up_uses_canonical_active_leaf(
    tmp_path: Path,
    head: str | None,
    role: str,
    status: str,
    needs_follow_up: bool,
) -> None:
    raw = json.loads(
        (Path(__file__).parents[2] / "fixtures/chatgpt/native-duplicate-attachment-occurrences-v1.json").read_text()
    )
    if head is not None:
        raw["current_node"] = head
    raw["mapping"]["assistant-node"]["message"]["author"]["role"] = role
    raw["mapping"]["assistant-node"]["message"]["status"] = status
    with receiver(tmp_path) as (host, port):
        _, _, prepared, _ = _prepare_native_occurrences(host, port, json.dumps(raw).encode())
    assert prepared["summary"]["needs_follow_up"] is needs_follow_up


def test_native_preparation_preserves_duplicate_asset_occurrences_and_canonical_semantics(tmp_path: Path) -> None:
    """A provider-ID dictionary merge or fabricated old instance turns this red."""
    from polylogue.core.enums import Provider, Role
    from polylogue.pipeline.ids import (
        message_content_identities,
        message_owner_resolution,
        session_content_hash,
        session_id,
    )
    from polylogue.sources.parsers.browser_capture import parse, parse_native_payload
    from polylogue.storage.sqlite.archive_tiers.write import prepare_session_rows

    raw = (Path(__file__).parents[2] / "fixtures/chatgpt/native-duplicate-attachment-occurrences-v1.json").read_bytes()
    native = json.loads(raw)
    with receiver(tmp_path) as (host, port):
        path, descriptor, prepared, assets = _prepare_native_occurrences(host, port, raw)
        assert prepared["summary"]["turn_count"] == 2
        assert len(assets) == 2
        assert [asset["descriptor"]["original_record_key"] for asset in assets] == ["user-node", "assistant-node"]
        assert [asset["descriptor"]["provider_meta"]["native_turn_ordinal"] for asset in assets] == [0, 1]
        assert assets[0]["descriptor"]["provider_attachment_id"] == assets[1]["descriptor"]["provider_attachment_id"]
        status, refused = request(
            host, port, "POST", f"{path}/finalize", {**descriptor, "plan_digest": prepared["plan_digest"]}
        )
        assert status == 409 and refused["error"]["code"] == "native_asset_receipts_pending"
        expected_bytes = [b"synthetic user image", b"synthetic model image"]
        for asset, content in zip(assets, expected_bytes, strict=True):
            receipt = {
                **descriptor,
                "ordinal": asset["ordinal"],
                "plan_digest": prepared["plan_digest"],
                "descriptor_digest": asset["descriptor_digest"],
                "outcome": {"status": "acquired"},
                "sha256": hashlib.sha256(content).hexdigest(),
                "size_bytes": len(content),
            }
            status, _ = _native_bytes(host, port, f"{path}/asset", receipt, content)
            assert status == 200
        status, final = request(
            host, port, "POST", f"{path}/finalize", {**descriptor, "plan_digest": prepared["plan_digest"]}
        )
        assert status == 200, final
        artifact = capture_job_store_root(tmp_path) / "artifacts" / f"{final['sha256']}.native"
        literal = artifact.read_bytes()
        assert raw in literal
        envelope = json.loads(literal)
        staged_asset_count = 0
        carrierless = json.loads(literal)
        for attachments in [
            carrierless["session"].get("attachments", []),
            *(turn.get("attachments", []) for turn in carrierless["session"].get("turns", [])),
        ]:
            for attachment in attachments:
                if "content_base64" not in attachment:
                    continue
                staged_asset_count += 1
                attachment.pop("content_base64")
                attachment.pop("inline_base64", None)
                attachment.pop("data", None)
                attachment.pop("size_bytes", None)
                attachment["provider_meta"]["asset_acquisition"] = {"status": "recovered_bytes_unavailable"}
                attachment["provider_meta"].pop("content_sha256", None)
        assert staged_asset_count == 2
        produced = BrowserCaptureEnvelope.model_validate(envelope)
        unavailable = BrowserCaptureEnvelope.model_validate(carrierless)
        assert (
            capture_convergence(summarize_capture_envelope(produced), summarize_capture_envelope(unavailable))
            is CaptureConvergence.PUBLISH
        )
        assert envelope["raw_provider_payload"] == native
        assert envelope["provenance"]["extension_instance_id"] is None
        assert envelope["provenance"]["acquisition_sequence"] is None
        parsed = parse(envelope, "native-occurrences")
        expected = parse_native_payload(
            Provider.CHATGPT, native, "native-occurrences", prepared_attachment_ownership=True
        )
        for attachment, content in zip(expected.attachments, expected_bytes, strict=True):
            attachment.inline_bytes = content
        assert [row.inline_bytes for row in parsed.attachments] == expected_bytes
        assert [row.direction for row in parsed.attachments] == ["user_input", "model_output"]
        assert [row.model_dump(exclude={"owner_coordinate"}) for row in parsed.messages] == [
            row.model_dump(exclude={"owner_coordinate"}) for row in expected.messages
        ]
        assert [message.role for message in parsed.messages] == [Role.USER, Role.ASSISTANT]
        assert message_content_identities(parsed.messages) == message_content_identities(expected.messages)
        assert message_owner_resolution(parsed.messages).keys == message_owner_resolution(expected.messages).keys
        assert session_id(parsed.source_name, parsed.provider_session_id) == session_id(
            expected.source_name, expected.provider_session_id
        )
        assert session_content_hash(parsed) == session_content_hash(expected)
        assert prepare_session_rows(parsed).message_rows == prepare_session_rows(expected).message_rows
        assert prepare_session_rows(parsed).block_rows == prepare_session_rows(expected).block_rows
        status, ack = request(
            host,
            port,
            "POST",
            f"{path}/publish",
            {**descriptor, "plan_digest": prepared["plan_digest"], "sha256": final["sha256"]},
        )
        assert status == 202, ack
        assert ack["content_hash"] == final["sha256"]
        status, retried = request(
            host,
            port,
            "POST",
            f"{path}/publish",
            {**descriptor, "plan_digest": prepared["plan_digest"], "sha256": final["sha256"]},
        )
        assert status == 202 and retried == ack
        assert artifact.read_bytes() == literal


def test_native_preparation_reordered_mapping_keeps_full_ids_hashes_and_refuses_forged_occurrence(
    tmp_path: Path,
) -> None:
    """Private raw ordinals may change; semantic owners and identities may not."""
    from polylogue.core.message_owner import MessageOwnerAmbiguityError
    from polylogue.pipeline.ids import (
        message_content_identities,
        message_owner_resolution,
        session_content_hash,
        session_id,
    )
    from polylogue.sources.parsers.browser_capture import parse
    from polylogue.storage.sqlite.archive_tiers.write import _duplicate_message_native_ids, _message_id

    fixture = (
        Path(__file__).parents[2] / "fixtures/chatgpt/native-duplicate-attachment-occurrences-v1.json"
    ).read_bytes()
    payload = json.loads(fixture)
    reordered = {**payload, "mapping": dict(reversed(list(payload["mapping"].items())))}
    parsed = []
    for index, native in enumerate([payload, reordered]):
        spool = tmp_path / str(index)
        with receiver(spool) as (host, port):
            path, descriptor, prepared, assets = _prepare_native_occurrences(host, port, json.dumps(native).encode())
            for asset in assets:
                status, _ = request(
                    host,
                    port,
                    "POST",
                    f"{path}/asset",
                    {
                        **descriptor,
                        "ordinal": asset["ordinal"],
                        "plan_digest": prepared["plan_digest"],
                        "descriptor_digest": asset["descriptor_digest"],
                        "outcome": {"status": "no_resolvable_source"},
                    },
                )
                assert status == 200
            status, final = request(
                host, port, "POST", f"{path}/finalize", {**descriptor, "plan_digest": prepared["plan_digest"]}
            )
            assert status == 200
            envelope = json.loads(
                (capture_job_store_root(spool) / "artifacts" / f"{final['sha256']}.native").read_bytes()
            )
            parsed.append(parse(envelope, "native-occurrences"))
            for forge_raw_position in (False, True):
                forged = json.loads(json.dumps(envelope))
                metadata = forged["session"]["attachments"][0]["provider_meta"]
                metadata["native_turn_ordinal"] = 1 - metadata["native_turn_ordinal"]
                if forge_raw_position:
                    metadata["native_raw_position"] = 1 - metadata["native_raw_position"]
                with pytest.raises(MessageOwnerAmbiguityError):
                    parse(forged, "native-occurrences")

    assert message_content_identities(parsed[0].messages) == message_content_identities(parsed[1].messages)
    assert message_owner_resolution(parsed[0].messages).keys == message_owner_resolution(parsed[1].messages).keys
    assert session_id(parsed[0].source_name, parsed[0].provider_session_id) == session_id(
        parsed[1].source_name, parsed[1].provider_session_id
    )
    assert session_content_hash(parsed[0]) == session_content_hash(parsed[1])
    full_ids = []
    for session in parsed:
        identities = message_content_identities(session.messages)
        duplicates = _duplicate_message_native_ids(session.messages)
        full_ids.append(
            [
                _message_id(
                    session_id(session.source_name, session.provider_session_id),
                    message,
                    ordinal,
                    content_identities=identities,
                    duplicate_native_ids=duplicates,
                )
                for ordinal, message in enumerate(session.messages)
            ]
        )
    assert full_ids[0] == full_ids[1]
    assert len(set(full_ids[0])) == 2


@pytest.mark.parametrize("terminal_retry", ["completed", "abandoned"])
@pytest.mark.parametrize("cancelled", [True, False])
def test_terminal_job_gc_retires_only_cancelled_unpublished_native_acquisition(
    tmp_path: Path, terminal_retry: str, cancelled: bool
) -> None:
    """NULL publication receipt protects live acquisition, not a cancelled one."""
    with receiver(tmp_path) as (host, port):
        raw = b'{"conversation_id":"neutral-unpublished"}'
        native_path, descriptor, _ = _retain_native_occurrences(host, port, raw)
        job_id = native_path.removeprefix("/v1/capture-jobs/").removesuffix("/native")
        if cancelled:
            status, result = request(host, port, "POST", native_path + "/cancel", descriptor)
            assert status == 200 and result["state"] == "cancelled"
            status, refused = request(host, port, "POST", native_path + "/plan", descriptor)
            assert status == 409 and refused["error"]["code"] == "native_acquisition_cancelled"
        lease = {name: descriptor[name] for name in ("lease_id", "generation", "proof")}
        checkpointed = _checkpoint(
            host, port, job_id, lease, cast(int, descriptor["expected_revision"]), 0, {"cursor": 1}, "gc-checkpoint"
        )
        status, terminal = request(
            host,
            port,
            "POST",
            f"/v1/capture-jobs/{job_id}/update",
            {
                **descriptor,
                "request_id": "gc-terminal-choice",
                "expected_revision": checkpointed["job"]["revision"],
                "retry": {"state": terminal_retry, "attempt": 1, "reason": None, "next_eligible_at": None},
                "retention": {"state": "eligible", "hold_reason": None, "timeline_authoritative": False},
            },
        )
        assert status == 200 and terminal["job"]["retention"]["state"] == "eligible"
        with sqlite3.connect(capture_job_database_path(tmp_path)) as connection:
            row = connection.execute(
                "SELECT state, final_receipt_json FROM capture_job_native_acquisitions WHERE job_id=?", (job_id,)
            ).fetchone()
        assert row is not None and row[1] is None
        artifact = capture_job_store_root(tmp_path) / "artifacts" / (hashlib.sha256(raw).hexdigest() + ".native")
        assert artifact.is_file()
        registry = CaptureJobRegistry(tmp_path, "neutral-gc-reader")
        result = registry.gc(now=datetime(2050, 1, 1, tzinfo=UTC))
        assert result["deleted"] == ([job_id] if cancelled else [])
        assert (job_id in _stored_job_ids(tmp_path)) is not cancelled
        assert artifact.exists() is not cancelled
        with sqlite3.connect(capture_job_database_path(tmp_path)) as connection:
            assert connection.execute(
                "SELECT COUNT(*) FROM capture_job_native_acquisitions WHERE job_id=?", (job_id,)
            ).fetchone()[0] == (0 if cancelled else 1)


def test_large_event_collections_survive_real_registry_cas_reads_and_retention(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A lazy decode alone is insufficient: stored cells and all reads stay lossless."""
    from polylogue.schemas.observation_spill import SpilledObject

    original_append = CaptureJobRegistry._append_event
    observed_lazy = []

    def append(
        registry: CaptureJobRegistry,
        connection: sqlite3.Connection,
        job_id: str,
        kind: object,
        request_id: str,
        expected_revision: int,
        refs: dict[str, object],
        payload: dict[str, object],
        *,
        advance_revision: bool = True,
    ) -> dict[str, object]:
        if request_id == "large-native-event":
            assert isinstance(refs, SpilledObject) and isinstance(payload, SpilledObject)
            observed_lazy.append(True)
        return original_append(
            registry,
            connection,
            job_id,
            kind,
            request_id,
            expected_revision,
            refs,
            payload,
            advance_revision=advance_revision,
        )

    monkeypatch.setattr(CaptureJobRegistry, "_append_event", append)
    with receiver(tmp_path) as (host, port):
        job = create(host, port)
        adopted = adopt(host, port, job)
        refs = {"conversation_ref": "neutral-conversation", "all_refs": [f"ref-{i}" for i in range(6000)]}
        payload = {"records": [{"index": i, "values": [i, i + 1], "text": "neutral" * 100} for i in range(6000)]}
        body: dict[str, object] = {
            "provider": "chatgpt",
            "scope": ACCOUNT_SCOPE,
            **_native_descriptor(adopted, "unused"),
            "request_id": "large-native-event",
            "kind": "capture-attempted",
            "refs": refs,
            "payload": payload,
        }
        path = f"/v1/capture-jobs/{job['job_id']}"
        status, first = request(host, port, "POST", path + "/events", body)
        assert status == 200 and first["event"]["refs"] == refs and first["event"]["payload"] == payload
        status, duplicate = request(host, port, "POST", path + "/events", body)
        assert status == 200 and duplicate["duplicate"] is True and duplicate["event"] == first["event"]
        status, conflict = request(host, port, "POST", path + "/events", {**body, "payload": {"records": []}})
        assert status == 409 and conflict["error"]["code"] == "event_request_conflict"
        status, events = request(
            host, port, "GET", path + f"/events?provider=chatgpt&scope={SCOPE_QUERY}&client_protocol=2", {}
        )
        assert status == 200 and events["events"][-1] == first["event"]
        status, whole = request(
            host, port, "GET", path + f"?provider=chatgpt&scope={SCOPE_QUERY}&client_protocol=2", {}
        )
        assert status == 200 and whole["timelines"]["neutral-conversation"] == [first["event"]]
        adopted["job"] = whole["job"]
        status, terminal = request(
            host,
            port,
            "POST",
            path + "/update",
            {
                **_native_descriptor(adopted, "unused"),
                "request_id": "retain-large-timeline",
                "retry": {"state": "completed", "attempt": 1, "reason": None, "next_eligible_at": None},
            },
        )
        assert status == 200 and terminal["job"]["retention"]["timeline_authoritative"] is True
        with sqlite3.connect(capture_job_database_path(tmp_path)) as connection:
            row = connection.execute(
                "SELECT typeof(refs_json), typeof(payload_json) FROM capture_job_events WHERE request_id=?",
                ("large-native-event",),
            ).fetchone()
        assert row == ("text", "text")
        assert len(observed_lazy) == 3
    assert not list((tmp_path / ".staging").iterdir())


@pytest.mark.parametrize(
    "value",
    [
        {"z": [None, True, {"e\u0301": "e\u0301"}], "a": 3},
        {"keys": {"\U0001f600": 2, "\u00e9": 1}, "values": [-3, 0, 42]},
    ],
)
def test_lazy_capture_canonical_bytes_equal_existing_profile(tmp_path: Path, value: dict[str, object]) -> None:
    from polylogue.browser_capture.capture_jobs import capture_canonical_chunks
    from polylogue.schemas.observation_spill import StreamedJSONDocument

    source = tmp_path / "neutral.json"
    source.write_text(json.dumps(value))
    with StreamedJSONDocument(source) as document:
        assert b"".join(capture_canonical_chunks(document)) == canonical_json(value).encode()


@pytest.mark.parametrize(
    "raw,code",
    [
        ('{"e\u0301":1,"\u00e9":2}', "non_canonical_key_collision"),
        ('{"number":9007199254740992}', "non_canonical_json"),
    ],
)
def test_lazy_capture_canonical_refuses_same_profile_invalid_values(tmp_path: Path, raw: str, code: str) -> None:
    from polylogue.browser_capture.capture_jobs import capture_canonical_chunks
    from polylogue.schemas.observation_spill import StreamedJSONDocument

    source = tmp_path / "neutral.json"
    source.write_text(raw)
    with StreamedJSONDocument(source) as document:
        with pytest.raises(CaptureJobError) as refusal:
            list(capture_canonical_chunks(document))
    assert refusal.value.code == code


def test_large_native_metadata_and_asset_outcomes_survive_prepare_plan_and_finalize(tmp_path: Path) -> None:
    """C dict shortcuts, eager header cells, or eager outcome cells lose this wire."""
    raw = (Path(__file__).parents[2] / "fixtures/chatgpt/native-duplicate-attachment-occurrences-v1.json").read_bytes()
    metadata = {
        "capture_fidelity": "native_full",
        "all_facts": {f"fact-{index}": {"value": index, "text": "neutral" * 50} for index in range(6000)},
    }
    outcome = {
        "status": "unavailable",
        "all_attempts": [{"ordinal": index, "reason": "neutral" * 50} for index in range(6000)],
    }
    with receiver(tmp_path) as (host, port):
        path, descriptor, provenance = _retain_native_occurrences(host, port, raw)
        preparation = {**descriptor, "provenance": provenance, "provider_meta": metadata}
        status, prepared = request(host, port, "POST", path + "/prepare", preparation)
        assert status == 200, prepared
        status, duplicate = request(host, port, "POST", path + "/prepare", preparation)
        assert status == 200 and duplicate["duplicate"] is True
        status, conflict = request(
            host, port, "POST", path + "/prepare", {**preparation, "provider_meta": {"all_facts": {}}}
        )
        assert status == 409 and conflict["error"]["code"] == "native_preparation_conflict"
        status, plan = request(host, port, "POST", path + "/plan", descriptor)
        assert status == 200 and len(plan["assets"]) == 2
        for asset in plan["assets"]:
            receipt = {
                **descriptor,
                "ordinal": asset["ordinal"],
                "plan_digest": prepared["plan_digest"],
                "descriptor_digest": asset["descriptor_digest"],
                "outcome": outcome,
            }
            status, first = request(host, port, "POST", path + "/asset", receipt)
            assert status == 200 and first["duplicate"] is False
            status, duplicate = request(host, port, "POST", path + "/asset", receipt)
            assert status == 200 and duplicate["duplicate"] is True
            status, conflict = request(
                host, port, "POST", path + "/asset", {**receipt, "outcome": {"status": "unavailable"}}
            )
            assert status == 409 and conflict["error"]["code"] == "native_asset_receipt_conflict"
        status, plan = request(host, port, "POST", path + "/plan", descriptor)
        assert status == 200 and all(asset["receipt"] == outcome for asset in plan["assets"])
        status, final = request(
            host, port, "POST", path + "/finalize", {**descriptor, "plan_digest": prepared["plan_digest"]}
        )
        assert status == 200, final
        artifact = capture_job_store_root(tmp_path) / "artifacts" / f"{final['sha256']}.native"
        envelope = json.loads(artifact.read_bytes())
        assert envelope["provider_meta"] == metadata
        attachments = [
            *envelope["session"].get("attachments", []),
            *(attachment for turn in envelope["session"]["turns"] for attachment in turn.get("attachments", [])),
        ]
        assert len(attachments) == 2 and all(
            attachment["provider_meta"]["asset_acquisition"] == outcome for attachment in attachments
        )
    assert not list((tmp_path / ".staging").iterdir())


@pytest.mark.parametrize("physical_failure", ["spool", "sqlite"])
def test_event_physical_publication_refusal_keeps_event_and_cas_unpublished(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, physical_failure: str
) -> None:
    from polylogue.browser_capture.capture_stream import SpoolStorageExhaustedError

    with receiver(tmp_path) as (host, port):
        job = create(host, port)
        adopted = adopt(host, port, job)
        body = {
            **_native_descriptor(adopted, "unused"),
            "request_id": "physically-refused-event",
            "kind": "capture-attempted",
            "refs": {"conversation_ref": "neutral"},
            "payload": {"text": "neutral" * 5000},
        }
        with monkeypatch.context() as failure:
            if physical_failure == "spool":

                def exhausted(chunks: Iterator[bytes], *, spool_root: Path, durable: bool) -> object:
                    raise SpoolStorageExhaustedError(1, 0)

                failure.setattr(capture_jobs_module, "stage_capture_chunks", exhausted)
            else:
                original_connect = CaptureJobRegistry._connect

                def physical_cell_limit(registry: CaptureJobRegistry) -> sqlite3.Connection:
                    connection = original_connect(registry)
                    connection.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, 4096)
                    return connection

                failure.setattr(CaptureJobRegistry, "_connect", physical_cell_limit)
            status, refused = request(host, port, "POST", f"/v1/capture-jobs/{job['job_id']}/events", body)
            assert status == (507 if physical_failure == "spool" else 413)
            assert refused["error"]["code"] == (
                "spool_storage_exhausted" if physical_failure == "spool" else "capture_literal_physical_limit"
            )
        status, current = request(
            host,
            port,
            "GET",
            f"/v1/capture-jobs/{job['job_id']}?provider=chatgpt&scope={SCOPE_QUERY}&client_protocol=2",
            {},
        )
        assert status == 200 and current["job"]["revision"] == adopted["job"]["revision"]
        assert all(event["request_id"] != body["request_id"] for event in current["events"])
    assert not list((tmp_path / ".staging").iterdir())


def test_large_intent_payload_survives_create_discover_events_update_and_get(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polylogue.schemas.observation_spill import SpilledObject

    payload = {"all_sources": {f"source-{index}": {"ordinal": index, "note": "neutral" * 50} for index in range(6000)}}
    digest = "sha256:" + hashlib.sha256(canonical_json(payload).encode()).hexdigest()
    intent = {"schema_version": 1, "version": 1, "intent_key": INTENT_KEY, "payload": payload, "digest": digest}
    original_summary = CaptureJobRegistry._summary
    observed_lazy: list[bool] = []

    def summary(registry: CaptureJobRegistry, connection: sqlite3.Connection, row: sqlite3.Row) -> dict[str, object]:
        result = original_summary(registry, connection, row)
        assert isinstance(result["intent"], SpilledObject)
        observed_lazy.append(True)
        return result

    monkeypatch.setattr(CaptureJobRegistry, "_summary", summary)
    with receiver(tmp_path) as (host, port):
        create_body: dict[str, object] = {
            "provider": "chatgpt",
            "scope": ACCOUNT_SCOPE,
            "request_id": "large-intent-create",
            "intent": intent,
        }
        status, created = request(host, port, "POST", "/v1/capture-jobs", create_body)
        assert status == 201 and created["job"]["intent"] == intent
        job = created["job"]
        status, duplicate = request(host, port, "POST", "/v1/capture-jobs", create_body)
        assert status == 200 and duplicate["job"]["intent"] == intent
        status, discovered = request(
            host, port, "POST", "/v1/capture-jobs/discover", {"provider": "chatgpt", "scope": ACCOUNT_SCOPE}
        )
        assert status == 200 and discovered["jobs"][0]["intent"] == intent
        adopted = adopt(host, port, job)
        assert adopted["job"]["intent"] == intent
        path = f"/v1/capture-jobs/{job['job_id']}"
        descriptor = _native_descriptor(adopted, "unused")
        status, event = request(
            host,
            port,
            "POST",
            path + "/events",
            {
                **descriptor,
                "request_id": "large-intent-event",
                "kind": "capture-attempted",
                "refs": {"conversation_ref": "neutral"},
                "payload": {"fact": "retained"},
            },
        )
        assert status == 200 and event["job"]["intent"] == intent
        status, updated = request(
            host,
            port,
            "POST",
            path + "/update",
            {
                **descriptor,
                "expected_revision": event["job"]["revision"],
                "request_id": "large-intent-update",
                "retry": {"state": "held", "attempt": 1, "reason": "neutral", "next_eligible_at": None},
            },
        )
        assert status == 200 and updated["job"]["intent"] == intent
        status, current = request(
            host, port, "GET", path + f"?provider=chatgpt&scope={SCOPE_QUERY}&client_protocol=2", {}
        )
        assert status == 200 and current["job"]["intent"] == intent
        assert len(observed_lazy) == 7
    assert not list((tmp_path / ".staging").iterdir())


@pytest.mark.parametrize("durable", [True, False])
def test_staged_artifact_durability_is_distinct_from_transient_json_cells(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, durable: bool
) -> None:
    from polylogue.browser_capture import capture_stream

    observed: list[int] = []

    def unavailable_sync(descriptor: int) -> None:
        observed.append(descriptor)
        raise OSError(errno.EIO, "neutral durability failure")

    monkeypatch.setattr(os, "fsync", unavailable_sync)
    if durable:
        with pytest.raises(OSError) as refusal:
            capture_stream.stage_capture_chunks(iter([b'{"neutral":"retained"}']), spool_root=tmp_path, durable=True)
        assert refusal.value.errno == errno.EIO and len(observed) == 1
        with pytest.raises(OSError) as closed:
            os.fstat(observed[0])
        assert closed.value.errno == errno.EBADF
    else:
        staged = capture_stream.stage_capture_chunks(
            iter([b'{"neutral":"retained"}']), spool_root=tmp_path, durable=False
        )
        try:
            assert staged.path.read_bytes() == b'{"neutral":"retained"}'
            assert observed == []
        finally:
            staged.discard()
    assert not list((tmp_path / ".staging").iterdir())


def test_maximum_http_event_page_succeeds_with_low_descriptor_headroom(tmp_path: Path) -> None:
    """Retaining two spill connections per event fails this actual 500-row route."""
    import resource

    with receiver(tmp_path) as (host, port):
        job = create(host, port)
        registry = CaptureJobRegistry(tmp_path, "neutral-page-writer")
        with registry._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            for index in range(500):
                registry._append_event(
                    connection,
                    job["job_id"],
                    "capture-attempted",
                    f"page-{index}",
                    0,
                    {"conversation_ref": "neutral-page"},
                    {"ordinal": index, "nested": [index, {"exact": "value"}]},
                    advance_revision=False,
                )
        limits = resource.getrlimit(resource.RLIMIT_NOFILE)
        with os.scandir("/proc/self/fd") as descriptors:
            headroom_limit = max(int(entry.name) for entry in descriptors) + 40
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (min(limits[0], headroom_limit), limits[1]))
            status, page = request(
                host,
                port,
                "GET",
                f"/v1/capture-jobs/{job['job_id']}/events?provider=chatgpt&scope={SCOPE_QUERY}&client_protocol=2&limit=500",
                {},
            )
            assert status == 200
            assert len(page["events"]) == 500 and page["has_more"] is True
            assert [event["payload"]["ordinal"] for event in page["events"]] == list(range(500))
            assert page["events"][-1]["payload"]["nested"] == [499, {"exact": "value"}]
            assert [event["payload"]["ordinal"] for event in page["timelines"]["neutral-page"]] == list(
                reversed(range(500))
            )
        finally:
            resource.setrlimit(resource.RLIMIT_NOFILE, limits)


@pytest.mark.parametrize("population", [100, 10_000])
def test_ordinary_gc_skips_held_population_and_advances_orphan_frontier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, population: int
) -> None:
    """VM work stays independent of held jobs; each request advances one artifact."""
    with receiver(tmp_path) as (host, port):
        job = create(host, port)
    registry = CaptureJobRegistry(tmp_path, "neutral-gc-owner")
    with registry._connection() as connection:
        columns = [row[1] for row in connection.execute("PRAGMA table_info(capture_jobs)")]
        template = connection.execute("SELECT * FROM capture_jobs WHERE job_id=?", (job["job_id"],)).fetchone()
        assert template is not None
        connection.execute("BEGIN IMMEDIATE")
        for index in range(population):
            values = dict(zip(columns, template, strict=True))
            values.update(job_id=f"held-{index}", intent_key=f"held-intent-{index}")
            # Both ordinary held jobs and otherwise eligible jobs with live
            # leases must be excluded by indexed selection before Python reads.
            if index % 2:
                values.update(
                    retention_json=canonical_json({"state": "eligible", "timeline_authoritative": False}),
                    retry_json=canonical_json({"state": "completed"}),
                    checkpoint_sequence=1,
                    receipt_json="{}",
                    lease_json=canonical_json({"expires_at": "2099-01-01T00:00:00Z"}),
                )
            connection.execute(
                "INSERT INTO capture_jobs (" + ",".join(columns) + ") VALUES (" + ",".join("?" for _ in columns) + ")",
                tuple(values[column] for column in columns),
            )
    directory = capture_job_store_root(tmp_path) / "artifacts"
    directory.mkdir(exist_ok=True)
    for index in range(8):
        (directory / (f"{index:064x}.native")).write_bytes(b"unpublished neutral orphan")
    steps = 0
    original_connect = CaptureJobRegistry._connect

    def progress() -> int:
        nonlocal steps
        steps += 1
        return 0

    def connect(self: CaptureJobRegistry) -> sqlite3.Connection:
        connection = original_connect(self)
        connection.set_progress_handler(progress, 1)
        return connection

    monkeypatch.setattr(CaptureJobRegistry, "_connect", connect)
    assert registry.gc(incremental_artifacts=True)["count"] == 0
    # This is an operation bound, not an elapsed-time assertion. The previous
    # SELECT-all/sort visits every row and each artifact root query scans jobs.
    # Three independent indexed candidate queries have fixed VM overhead;
    # this bound still rejects a scan of the 10,000-row retained population.
    assert steps < 2000, steps
    assert list(directory.iterdir()) == []
    for _ in range(8):
        assert CaptureJobRegistry(tmp_path, "same-owner").gc(incremental_artifacts=True)["count"] == 0
    assert list(directory.iterdir()) == []
    with sqlite3.connect(capture_job_database_path(tmp_path)) as connection:
        assert connection.execute("SELECT COUNT(*) FROM capture_jobs").fetchone()[0] == population + 1


def test_fresh_library_create_owns_gc_scratch_without_existing_spool(tmp_path: Path) -> None:
    spool = tmp_path / "not-created"
    payload = {"neutral": "fresh"}
    registry = CaptureJobRegistry(spool, "neutral-fresh")
    with registry.result_scope():
        status, result = registry.create(
            {
                "provider": "chatgpt",
                "scope": ACCOUNT_SCOPE,
                "client_protocol": 2,
                "intent": {
                    "schema_version": 1,
                    "version": 1,
                    "intent_key": INTENT_KEY,
                    "payload": payload,
                    "digest": canonical_digest(payload),
                },
            }
        )
        assert status == 201 and result["created"] is True
    assert capture_job_database_path(spool).is_file()


def test_orphan_frontier_restarts_after_artifact_directory_identity_changes(tmp_path: Path) -> None:
    registry = CaptureJobRegistry(tmp_path, "neutral-frontier")
    registry.gc(incremental_artifacts=True)
    directory = capture_job_store_root(tmp_path) / "artifacts"
    directory.mkdir()
    for index in range(65):
        (directory / (f"{index:064x}.native")).write_bytes(b"old neutral orphan")
    registry.gc(incremental_artifacts=True)
    assert len(list(directory.iterdir())) == 1
    previous = directory.with_name("previous-artifacts")
    directory.rename(previous)
    directory.mkdir()
    new = directory / ("f" * 64 + ".native")
    new.write_bytes(b"new neutral orphan")
    CaptureJobRegistry(tmp_path, "new-request").gc(incremental_artifacts=True)
    assert not new.exists()
    assert len(list(previous.iterdir())) == 1


@pytest.mark.parametrize("population", [100, 10_000])
def test_gc_limits_expired_cohort_work_in_lease_index_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, population: int
) -> None:
    with receiver(tmp_path) as (host, port):
        job = create(host, port)
    registry = CaptureJobRegistry(tmp_path, "neutral-expired")
    with registry._connection() as connection:
        columns = [row[1] for row in connection.execute("PRAGMA table_info(capture_jobs)")]
        template = connection.execute("SELECT * FROM capture_jobs WHERE job_id=?", (job["job_id"],)).fetchone()
        assert template is not None
        connection.execute("BEGIN IMMEDIATE")
        for index in range(population):
            values = dict(zip(columns, template, strict=True))
            values.update(
                job_id=f"expired-{index:05d}",
                intent_key=f"expired-intent-{index}",
                retention_json=canonical_json({"state": "eligible", "timeline_authoritative": False}),
                retry_json=canonical_json({"state": "completed"}),
                checkpoint_sequence=1,
                receipt_json="{}",
                lease_json=canonical_json({"expires_at": "2049-01-01T00:00:00Z", "lease_id": f"{index:05d}"}),
            )
            connection.execute(
                "INSERT INTO capture_jobs (" + ",".join(columns) + ") VALUES (" + ",".join("?" for _ in columns) + ")",
                tuple(values[column] for column in columns),
            )
    steps = 0
    original_connect = CaptureJobRegistry._connect

    def progress() -> int:
        nonlocal steps
        steps += 1
        return 0

    def connect(self: CaptureJobRegistry) -> sqlite3.Connection:
        connection = original_connect(self)
        connection.set_progress_handler(progress, 1)
        return connection

    monkeypatch.setattr(CaptureJobRegistry, "_connect", connect)
    result = registry.gc(now=datetime(2050, 1, 1, tzinfo=UTC), limit=3, incremental_artifacts=True)
    assert result["deleted"] == ["expired-00000"]
    assert steps < 2000, steps


def test_gc_exact_fractional_and_whole_second_ranges_page_past_rejected_guard(tmp_path: Path) -> None:
    with receiver(tmp_path) as (host, port):
        job = create(host, port)
    registry = CaptureJobRegistry(tmp_path, "neutral-expiry-boundary")
    stamps = {
        "invalid": "0000",
        "whole": "2050-01-01T00:00:00Z",
        "fraction": "2050-01-01T00:00:00.100000Z",
        "equal": "2050-01-01T00:00:00.200000Z",
        "future": "2050-01-01T00:00:00.900000Z",
    }
    with registry._connection() as connection:
        columns = [row[1] for row in connection.execute("PRAGMA table_info(capture_jobs)")]
        template = connection.execute("SELECT * FROM capture_jobs WHERE job_id=?", (job["job_id"],)).fetchone()
        assert template is not None
        connection.execute("BEGIN IMMEDIATE")
        for name, stamp in stamps.items():
            values = dict(zip(columns, template, strict=True))
            values.update(
                job_id=name,
                intent_key="expiry-" + name,
                retention_json=canonical_json({"state": "eligible", "timeline_authoritative": False}),
                retry_json=canonical_json({"state": "completed"}),
                checkpoint_sequence=1,
                receipt_json="{}",
                lease_json=canonical_json({"expires_at": stamp}),
            )
            connection.execute(
                "INSERT INTO capture_jobs (" + ",".join(columns) + ") VALUES (" + ",".join("?" for _ in columns) + ")",
                tuple(values[column] for column in columns),
            )
    current = datetime(2050, 1, 1, microsecond=200_000, tzinfo=UTC)
    deleted = [registry.gc(now=current, limit=1, incremental_artifacts=True)["deleted"] for _ in range(3)]
    assert deleted == [["fraction"], ["equal"], ["whole"]]
    assert registry.gc(now=current, limit=1, incremental_artifacts=True)["deleted"] == []
    assert {"future", "invalid"}.issubset(_stored_job_ids(tmp_path))


def _seed_retired_native_children(tmp_path: Path, population: int) -> str:
    with receiver(tmp_path) as (host, port):
        job_id = _retired_job(host, port)
    with sqlite3.connect(capture_job_database_path(tmp_path)) as connection:
        connection.execute(
            "INSERT INTO capture_job_native_acquisitions VALUES (?, 'neutral-acquisition', '{}', '[]', 'cancelled', NULL, NULL, NULL)",
            (job_id,),
        )
        for index in range(population):
            connection.execute(
                "INSERT INTO capture_job_native_plan VALUES (?, 'neutral-acquisition', ?, '{}', ?)",
                (job_id, index, f"{index:064x}"),
            )
            connection.execute(
                "INSERT INTO capture_job_native_assets VALUES (?, 'neutral-acquisition', ?, '{}', NULL, NULL)",
                (job_id, index),
            )
    return job_id


@pytest.mark.parametrize("population", [256, 10_000])
def test_ordinary_requests_do_not_cascade_large_retiring_native_membership(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, population: int
) -> None:
    from polylogue.browser_capture.server import BrowserCaptureHTTPServer

    job_id = _seed_retired_native_children(tmp_path, population)
    registry = CaptureJobRegistry(tmp_path, "neutral-retirement-page")
    assert registry.gc(now=datetime(2050, 1, 1, tzinfo=UTC), incremental_artifacts=True)["count"] == 0
    with sqlite3.connect(capture_job_database_path(tmp_path)) as connection:
        assert (
            connection.execute("SELECT COUNT(*) FROM capture_job_native_assets WHERE job_id=?", (job_id,)).fetchone()[0]
            == population - 64
        )
        assert (
            connection.execute("SELECT COUNT(*) FROM capture_job_native_plan WHERE job_id=?", (job_id,)).fetchone()[0]
            == population
        )
    # Freeze only the independent lifecycle to isolate actual request work.
    monkeypatch.setattr(BrowserCaptureHTTPServer, "service_actions", lambda self: None)
    steps = 0
    original_connect = CaptureJobRegistry._connect

    def progress() -> int:
        nonlocal steps
        steps += 1
        return 0

    def connect(self: CaptureJobRegistry) -> sqlite3.Connection:
        connection = original_connect(self)
        connection.set_progress_handler(progress, 1)
        return connection

    monkeypatch.setattr(CaptureJobRegistry, "_connect", connect)
    with receiver(tmp_path) as (host, port):
        status, found = request(
            host, port, "POST", "/v1/capture-jobs/discover", {"provider": "chatgpt", "scope": ACCOUNT_SCOPE}
        )
        assert status == 200 and found["jobs"] == [] and found["total"] == 0
        payload = {"cutoff": "2026-01-01T00:00:00Z"}
        status, pending = request(
            host,
            port,
            "POST",
            "/v1/capture-jobs",
            {
                "provider": "chatgpt",
                "scope": ACCOUNT_SCOPE,
                "intent": {
                    "schema_version": 1,
                    "version": 1,
                    "intent_key": INTENT_KEY,
                    "payload": payload,
                    "digest": canonical_digest(payload),
                },
            },
        )
        assert status == 503 and pending["error"]["code"] == "capture_job_retirement_pending"
        status, _ = request(
            host, port, "GET", f"/v1/capture-jobs/{job_id}?provider=chatgpt&scope={SCOPE_QUERY}&client_protocol=2", {}
        )
        assert status == 404
    assert steps < 1000, steps
    with sqlite3.connect(capture_job_database_path(tmp_path)) as connection:
        assert (
            connection.execute("SELECT COUNT(*) FROM capture_job_native_assets WHERE job_id=?", (job_id,)).fetchone()[0]
            == population - 64
        )
        assert (
            connection.execute("SELECT COUNT(*) FROM capture_job_native_plan WHERE job_id=?", (job_id,)).fetchone()[0]
            == population
        )


def test_receiver_lifecycle_drains_retirement_without_requests_and_restarts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_id = _seed_retired_native_children(tmp_path, 128)
    directory = capture_job_store_root(tmp_path) / "artifacts"
    for index in range(130):
        (directory / (f"{index:064x}.native")).write_bytes(b"neutral orphan custody")
    registry = CaptureJobRegistry(tmp_path, "neutral-retirement-start")
    registry.gc(now=datetime(2050, 1, 1, tzinfo=UTC), incremental_artifacts=True)
    registry.close_maintenance()
    assert str(capture_job_database_path(tmp_path)) not in capture_jobs_module._ARTIFACT_SWEEPS
    completed = Event()
    original = CaptureJobRegistry.maintenance_step

    def step(self: CaptureJobRegistry) -> None:
        original(self)
        if job_id not in _stored_job_ids(tmp_path) and not list(directory.iterdir()):
            completed.set()

    monkeypatch.setattr(CaptureJobRegistry, "maintenance_step", step)
    # Clock moves backwards after the durable marker: retirement remains fenced
    # and restart cannot reinterpret its old expired lease as live authority.
    monkeypatch.setattr(capture_jobs_module, "_now", lambda: datetime(2020, 1, 1, tzinfo=UTC))
    with receiver(tmp_path) as (host, port):
        # No HTTP request or explicit GC triggers this unattended completion.
        assert completed.wait(15)
        replacement = create(host, port)
        assert replacement["job_id"] != job_id
    with sqlite3.connect(capture_job_database_path(tmp_path)) as connection:
        for table in ("capture_job_native_acquisitions", "capture_job_native_plan", "capture_job_native_assets"):
            assert connection.execute("SELECT COUNT(*) FROM " + table + " WHERE job_id=?", (job_id,)).fetchone()[0] == 0


def test_receiver_maintenance_failure_is_visible_and_next_turn_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polylogue.browser_capture import server as server_module

    attempts = 0
    events: list[tuple[str, dict[str, object]]] = []
    original = CaptureJobRegistry.maintenance_step

    def step(self: CaptureJobRegistry) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise sqlite3.OperationalError("database is locked")
        original(self)

    def emit(event: str, **fields: object) -> None:
        events.append((event, fields))

    monkeypatch.setattr(CaptureJobRegistry, "maintenance_step", step)
    monkeypatch.setattr(server_module, "emit", emit)
    server = make_server("127.0.0.1", 0, spool_path=tmp_path, auth_token=TOKEN)
    try:
        server.service_actions()
        server.service_actions()
        assert attempts == 2
        assert [event for event, _ in events] == ["browser_capture.capture_job_registry_unavailable"]
        assert "error" in events[0][1]
        directory = capture_job_store_root(tmp_path) / "artifacts"
        directory.mkdir(exist_ok=True)
        for index in range(65):
            (directory / (f"{index:064x}.native")).write_bytes(b"neutral orphan")
        server.service_actions()
        assert str(capture_job_database_path(tmp_path)) in capture_jobs_module._ARTIFACT_SWEEPS
    finally:
        server.server_close()
    assert str(capture_job_database_path(tmp_path)) not in capture_jobs_module._ARTIFACT_SWEEPS


def test_artifact_frontier_retries_the_same_pending_page_after_real_sqlite_busy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = CaptureJobRegistry(tmp_path, "neutral-pending-page")
    registry.gc(incremental_artifacts=True)
    directory = capture_job_store_root(tmp_path) / "artifacts"
    directory.mkdir(exist_ok=True)
    for index in range(96):
        (directory / (f"{index:064x}.native")).write_bytes(b"neutral unrooted bytes")
    original_connect = CaptureJobRegistry._connect

    def connect(self: CaptureJobRegistry) -> sqlite3.Connection:
        connection = original_connect(self)
        connection.execute("PRAGMA busy_timeout=0")
        return connection

    monkeypatch.setattr(CaptureJobRegistry, "_connect", connect)
    held = sqlite3.connect(capture_job_database_path(tmp_path), isolation_level=None)
    try:
        held.execute("BEGIN IMMEDIATE")
        with pytest.raises(sqlite3.OperationalError) as busy:
            registry._collect_checkpoint_artifacts((), incremental=True, quantum=64)
        assert busy.value.sqlite_errorcode == sqlite3.SQLITE_BUSY
        frontier = capture_jobs_module._ARTIFACT_SWEEPS[str(capture_job_database_path(tmp_path))]
        pending = tuple(frontier.pending)
        assert len(pending) == 64 and len(list(directory.iterdir())) == 96
        assert all((directory / name).read_bytes() == b"neutral unrooted bytes" for name in pending)
        held.rollback()
    finally:
        held.close()
    seen: list[str] = []
    original_check = CaptureJobRegistry._collect_checkpoint_artifact

    def check(self: CaptureJobRegistry, connection: sqlite3.Connection, directory: Path, name: str) -> bool:
        seen.append(name)
        return original_check(self, connection, directory, name)

    monkeypatch.setattr(CaptureJobRegistry, "_collect_checkpoint_artifact", check)
    registry._collect_checkpoint_artifacts((), incremental=True, quantum=64)
    assert tuple(seen) == pending
    assert len(list(directory.iterdir())) == 32 and not frontier.pending
    registry.close_maintenance()
    registry._collect_checkpoint_artifacts((), incremental=True, quantum=64)
    assert list(directory.iterdir()) == []
    registry.close_maintenance()


def test_premarker_retirement_is_hidden_from_http_discovery_and_cannot_renew_lease(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polylogue.browser_capture.server import BrowserCaptureHTTPServer

    job_id = _seed_retired_native_children(tmp_path, 8)
    monkeypatch.setattr(BrowserCaptureHTTPServer, "service_actions", lambda self: None)
    monkeypatch.setattr(capture_jobs_module, "_now", lambda: datetime(2050, 1, 1, tzinfo=UTC))
    with sqlite3.connect(capture_job_database_path(tmp_path)) as connection:
        columns = [row[1] for row in connection.execute("PRAGMA table_info(capture_jobs)")]
        template = connection.execute("SELECT * FROM capture_jobs WHERE job_id=?", (job_id,)).fetchone()
        assert template is not None
        original = dict(zip(columns, template, strict=True))
        lease = json.loads(original["lease_json"])
        expected: set[str] = set()
        for index in range(26):
            values = dict(original)
            identity = f"inspectable-{index:02d}"
            expected.add(identity)
            values.update(
                job_id=identity,
                intent_key="inspectable-intent-" + str(index),
                lease_json=canonical_json({**lease, "expires_at": "2099-01-01T00:00:00Z"}),
            )
            if index == 0:
                values.update(
                    lease_json=original["lease_json"],
                    retention_json=canonical_json(
                        {"state": "held", "hold_reason": "neutral", "timeline_authoritative": False}
                    ),
                )
            if index == 1:
                values["lease_json"] = original["lease_json"]
            connection.execute(
                "INSERT INTO capture_jobs (" + ",".join(columns) + ") VALUES (" + ",".join("?" for _ in columns) + ")",
                tuple(values[column] for column in columns),
            )
        connection.execute(
            "INSERT INTO capture_job_native_acquisitions VALUES (?, 'unfinished', '{}', '[]', 'acquiring', NULL, NULL, NULL)",
            ("inspectable-01",),
        )
    with receiver(tmp_path) as (host, port):
        scope: dict[str, object] = {"provider": "chatgpt", "scope": ACCOUNT_SCOPE}
        status, first = request(host, port, "POST", "/v1/capture-jobs/discover", scope)
        assert status == 200 and first["total"] == 26 and len(first["jobs"]) == 25 and first["has_more"] is True
        status, second = request(host, port, "POST", "/v1/capture-jobs/discover", {**scope, "cursor": first["cursor"]})
        assert status == 200 and second["total"] == 26 and len(second["jobs"]) == 1 and second["has_more"] is False
        assert {row["job_id"] for row in first["jobs"] + second["jobs"]} == expected
        status, refused = request(
            host,
            port,
            "POST",
            f"/v1/capture-jobs/{job_id}/adopt",
            {
                **scope,
                "request_id": "do-not-revive",
                "session_id": "neutral-session",
                "expected_revision": original["revision"],
                "expected_lease_generation": lease["generation"],
            },
        )
        assert status == 503 and refused["error"]["code"] == "capture_job_retirement_pending"
    with sqlite3.connect(capture_job_database_path(tmp_path)) as connection:
        row = connection.execute(
            "SELECT revision, lease_json, retention_json FROM capture_jobs WHERE job_id=?", (job_id,)
        ).fetchone()
        assert row == (original["revision"], original["lease_json"], original["retention_json"])
        assert (
            connection.execute("SELECT COUNT(*) FROM capture_job_native_assets WHERE job_id=?", (job_id,)).fetchone()[0]
            == 8
        )


def test_artifact_frontier_advances_past_sixty_four_live_reader_holds(tmp_path: Path) -> None:
    import fcntl
    from contextlib import ExitStack

    registry = CaptureJobRegistry(tmp_path, "neutral-reader-holds")
    registry.gc(incremental_artifacts=True)
    directory = capture_job_store_root(tmp_path) / "artifacts"
    directory.mkdir(exist_ok=True)
    for index in range(65):
        (directory / (f"{index:064x}.native")).write_bytes(b"neutral held custody")
    with os.scandir(directory) as entries:
        names = [entry.name for entry in entries]
    with ExitStack() as readers:
        for name in names[:64]:
            handle = readers.enter_context((directory / name).open("rb"))
            fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
        registry._collect_checkpoint_artifacts((), incremental=True, quantum=64)
        assert len(list(directory.iterdir())) == 65
        registry._collect_checkpoint_artifacts((), incremental=True, quantum=64)
        assert not (directory / names[64]).exists()
        assert all((directory / name).read_bytes() == b"neutral held custody" for name in names[:64])
    registry._collect_checkpoint_artifacts((), incremental=True, quantum=64)
    registry._collect_checkpoint_artifacts((), incremental=True, quantum=64)
    assert list(directory.iterdir()) == []
    registry.close_maintenance()
