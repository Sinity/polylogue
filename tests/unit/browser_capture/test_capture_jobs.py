"""Production HTTP fixtures for receiver-authoritative CaptureJob recovery."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from http.client import HTTPConnection
from pathlib import Path
from threading import Thread
from typing import Any, cast

import pytest

from polylogue.browser_capture import capture_jobs as capture_jobs_module
from polylogue.browser_capture.capture_jobs import (
    CaptureJobRegistry,
    canonical_digest,
    canonical_json,
    capture_job_database_path,
    capture_job_scope_namespace,
)
from polylogue.browser_capture.route_contracts import browser_capture_route_contract_for
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


def housekeeping(host: str, port: int, *, now: datetime | None = None) -> list[str]:
    """Drive the receiver's spool housekeeping route, which owns retention collection."""
    original = capture_jobs_module._now
    if now is not None:
        capture_jobs_module._now = lambda: now
    try:
        status, payload = request(host, port, "GET", "/v1/capture-jobs/orphans?client_protocol=1", {})
    finally:
        capture_jobs_module._now = original
    assert status == 200
    return cast(list[str], payload["collected"])


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
        forged = {
            **base,
            "request_id": "forged",
            "proof": "not-the-lease-proof",
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
            {"provider": "chatgpt", "account_scope": SCOPE, "client_protocol": 99},
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

        frozen_clock.advance(2)
        status, expired = request(
            host,
            port,
            "PUT",
            f"/v1/capture-jobs/{job['job_id']}/checkpoint",
            {
                "provider": "chatgpt",
                "account_scope": SCOPE,
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
                "account_scope": SCOPE,
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
                "account_scope": SCOPE,
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
            "account_scope": SCOPE,
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

        query = f"provider=chatgpt&account_scope={SCOPE}&client_protocol=1"
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
            "account_scope": SCOPE,
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
            f"/v1/capture-jobs/{job['job_id']}/events?provider=chatgpt&account_scope={SCOPE}&client_protocol=1&limit=10",
            {},
        )
        assert status == 200
        assert [event["kind"] for event in page["events"]] == ["created", "first-seen"]
        assert page["events"][1]["refs"]["conversation_ref"] == "conversation:1"
        assert page["timelines"] == {"conversation:1": [first["event"]]}


def test_timeline_uses_receiver_order_and_gc_requires_terminal_retention(tmp_path: Path) -> None:
    """Anti-vacuity: timestamp order or retention/terminal/lease bypass makes this fail.

    Collection is driven only through the receiver's housekeeping route, so
    unwiring it from that route makes this fail too.
    """
    with receiver(tmp_path) as (host, port):
        job = create(host, port)
        adopted = adopt(host, port, job)
        base = {
            "provider": "chatgpt",
            "account_scope": SCOPE,
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
        assert housekeeping(host, port, now=future) == []

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
        assert housekeeping(host, port) == []
        status, page = request(
            host,
            port,
            "GET",
            f"/v1/capture-jobs/{job['job_id']}/events?provider=chatgpt&account_scope={SCOPE}&client_protocol=1",
            {},
        )
        assert status == 200
        assert page["timelines"]["conversation:1"] == [second["event"], first["event"]]
        assert housekeeping(host, port, now=future) == [job["job_id"]]
        assert (
            request(
                host,
                port,
                "GET",
                f"/v1/capture-jobs/{job['job_id']}?provider=chatgpt&account_scope={SCOPE}&client_protocol=1",
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
            {"provider": "chatgpt", "account_scope": SCOPE},
        )
        assert status == 200
        assert found["jobs"] == []
        assert "orphans" not in found
        status, orphan_census = request(
            host,
            port,
            "GET",
            "/v1/capture-jobs/orphans?client_protocol=1",
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
        first = cast(list[dict[str, object]], registry.list_orphans(1)["orphans"])
        refreshed_first = next(entry for entry in first if entry["source_digest"] == digest)
        assert refreshed_first["orphan_kind"] == "legacy_backfill_checkpoint"
        assert refreshed_first["diagnostic"] == "account scope unavailable; explicit migration or abandonment required"

        readable.write_text("not json", encoding="utf-8")

        original_read_bytes = Path.read_bytes

        def raise_for_unreadable(path: Path) -> bytes:
            if path == unreadable:
                raise PermissionError(13, "permission denied")
            return original_read_bytes(path)

        monkeypatch.setattr(Path, "read_bytes", raise_for_unreadable)
        second = cast(list[dict[str, object]], registry.list_orphans(1)["orphans"])
        refreshed = next(entry for entry in second if entry["orphan_kind"] == "malformed_legacy_checkpoint")
        assert refreshed["diagnostic"] == "account scope unavailable; explicit migration or abandonment required"
        unreadable_entry = next(entry for entry in second if entry["orphan_kind"] == "unreadable_legacy_checkpoint")
        assert unreadable_entry["path"] == str(unreadable)
        assert unreadable_entry["errno_class"] == "PermissionError"
    finally:
        connection.close()


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
    _, created = registry.create(
        {
            "provider": "chatgpt",
            "account_scope": SCOPE,
            "client_protocol": 1,
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
    result = registry.get(job_id, {"provider": "chatgpt", "account_scope": SCOPE, "client_protocol": 1})
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
            {"provider": "chatgpt", "account_scope": SCOPE},
        )
    assert status == 500
    assert body == {"error": {"code": "registry_unavailable", "details": {}}}


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
            "account_scope": SCOPE,
            "lease_id": lease["lease_id"],
            "generation": lease["generation"],
            "proof": lease["proof"],
            "request_id": request_id,
            "expected_revision": revision,
            "checkpoint": {"sequence": sequence, "payload": payload, "digest": canonical_digest(payload)},
        },
    )
    assert status == 200, body
    return cast(dict[str, Any], body)


def test_pre_retention_update_receipt_replays_without_conflict(tmp_path: Path) -> None:
    """Anti-vacuity: dropping the legacy-digest branch in update() restores the 409.

    The stored digest is rewritten into the shape a build before retention
    joined it wrote, which is what a receiver spool carries across that
    upgrade.
    """
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
            "account_scope": SCOPE,
            "lease_id": adopted["lease"]["lease_id"],
            "generation": adopted["lease"]["generation"],
            "proof": adopted["lease"]["proof"],
            "request_id": "pre-upgrade-retry",
            "expected_revision": adopted["job"]["revision"],
            "retry": retry,
        }
        status, updated = request(host, port, "POST", f"/v1/capture-jobs/{job['job_id']}/update", update_body)
        assert status == 200 and updated["duplicate"] is False

        legacy = canonical_digest({"retry": retry, "lease_ttl_seconds": None})
        with sqlite3.connect(capture_job_database_path(tmp_path)) as connection:
            connection.execute(
                "UPDATE capture_job_update_receipts SET request_digest=? WHERE request_id=?",
                (legacy, "pre-upgrade-retry"),
            )
        status, replay = request(host, port, "POST", f"/v1/capture-jobs/{job['job_id']}/update", update_body)
        assert status == 200
        assert replay["duplicate"] is True
        assert replay["receipt"] == updated["receipt"]

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
                "account_scope": SCOPE,
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
        assert housekeeping(host, port, now=datetime(2050, 1, 1, tzinfo=UTC)) == []


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
            f"/v1/capture-jobs/{job['job_id']}/events?provider=chatgpt&account_scope={SCOPE}&client_protocol=1",
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

        query = f"provider=chatgpt&account_scope={SCOPE}&client_protocol=1"
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
