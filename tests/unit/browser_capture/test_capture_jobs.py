"""Production HTTP fixtures for receiver-authoritative CaptureJob recovery."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from http.client import HTTPConnection
from pathlib import Path
from threading import Thread
from typing import Any, cast

import polylogue.browser_capture.capture_jobs as capture_jobs
from polylogue.browser_capture.capture_jobs import canonical_digest, capture_job_scope_namespace
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

        monkeypatch.setattr(capture_jobs, "_now", lambda: datetime(2099, 1, 1, tzinfo=UTC))
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

        query = f"provider=chatgpt&account_scope={SCOPE}&client_protocol=1"
        status, detail = request(host, port, "GET", f"/v1/capture-jobs/{job['job_id']}?{query}", {})
        assert status == 200
        assert detail["job"]["latest_receipt"] is None
        assert detail["receipts"] == [updated["receipt"]]


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
