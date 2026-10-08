"""Mutation-sensitive route proofs for daemon HTTP writer coordination."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import queue
import socket
import sqlite3
import threading
from collections.abc import Awaitable, Callable, Iterator
from email.message import Message
from http import HTTPStatus
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, patch

import pytest

from polylogue.daemon.http import (
    DaemonAPIHandler,
    DaemonAPIHTTPServer,
)
from polylogue.daemon.uds import DaemonAPIUnixHTTPServer
from polylogue.daemon.web_auth import WebCredentialScope
from polylogue.daemon.write_coordinator import DaemonWriteCoordinator, DaemonWriteThreadBridge
from polylogue.daemon_client import DaemonClient, DaemonMutationIndeterminateError
from tests.infra.daemon_operations import running_daemon_operations
from tests.infra.sqlite_cursor_settlement import (
    native_settlement_connections,  # noqa: F401  # Pytest fixture discovery.
)


class _DeleteDaemonClient(DaemonClient):
    archive_root: Path
    authority_server: DaemonAPIUnixHTTPServer


class _RecordingBridge:
    def __init__(self, timeline: list[str]) -> None:
        self.timeline = timeline

    @contextlib.contextmanager
    def hold(self, actor: str) -> Iterator[None]:
        self.timeline.append(f"enter:{actor}")
        try:
            yield
        finally:
            self.timeline.append(f"exit:{actor}")

    def run_sync(self, actor: str, function: Callable[..., object], *args: object) -> object:
        self.timeline.append(f"run_sync:{actor}")
        return function(*args)


def _handler(path: list[str], timeline: list[str]) -> DaemonAPIHandler:
    def allow_auth(
        required_scope: WebCredentialScope = "read",
        *,
        allow_web: bool = True,
        refuse: object = None,
    ) -> bool:
        del required_scope, allow_web, refuse
        return True

    def allow_host(*, credential_request: bool = False) -> bool:
        del credential_request
        return True

    handler = object.__new__(DaemonAPIHandler)
    object.__setattr__(handler, "server", SimpleNamespace(write_bridge=_RecordingBridge(timeline)))
    handler.path = "/" + "/".join(path)
    object.__setattr__(handler, "_parse_path", lambda: (path, {}))
    object.__setattr__(handler, "_check_host_admission", allow_host)
    object.__setattr__(handler, "_check_auth", allow_auth)
    object.__setattr__(handler, "_check_cross_origin", lambda **_kwargs: True)
    object.__setattr__(handler, "_send_error", lambda *_args, **_kwargs: timeline.append("error"))
    return handler


@pytest.mark.parametrize(
    ("path", "handler_name", "actor"),
    [
        (["api", "reset"], "_handle_reset", "http.reset"),
    ],
)
def test_authenticated_write_route_holds_gate_around_handler(path: list[str], handler_name: str, actor: str) -> None:
    timeline: list[str] = []
    handler = _handler(path, timeline)
    setattr(handler, handler_name, lambda: timeline.append("body"))

    handler._do_post_impl()

    assert timeline == [f"enter:{actor}", "body", f"exit:{actor}"]


def test_ingest_route_delegates_publication_ownership_to_operation_runtime() -> None:
    """An outer lease deadlocks runtime publication and keeps preparation under the writer."""
    timeline: list[str] = []
    handler = _handler(["api", "ingest"], timeline)
    object.__setattr__(handler, "_handle_ingest", lambda: timeline.append("runtime"))
    handler._do_post_impl()
    assert timeline == ["runtime"]


def _seed_delete_authority_archive(root: Path, count: int) -> tuple[str, ...]:
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    initialize_active_archive_root(root)
    session_ids: list[str] = []
    with sqlite3.connect(root / "source.db") as source_conn, sqlite3.connect(root / "index.db") as index_conn:
        source_conn.execute("PRAGMA foreign_keys = ON")
        index_conn.execute("PRAGMA foreign_keys = ON")
        for index in range(count):
            native_id = f"authority-{index}"
            raw_id = f"raw-{native_id}"
            source_conn.execute(
                """
                INSERT INTO raw_sessions (raw_id, origin, native_id, source_path, blob_hash, blob_size, acquired_at_ms)
                VALUES (?, 'codex-session', ?, ?, zeroblob(32), 0, 1000)
                """,
                (raw_id, native_id, str(root / f"{native_id}.jsonl")),
            )
            index_conn.execute(
                """
                INSERT INTO sessions (native_id, origin, raw_id, title, content_hash, created_at_ms, updated_at_ms)
                VALUES (?, 'codex-session', ?, ?, zeroblob(32), 1000, 2000)
                """,
                (native_id, raw_id, native_id),
            )
            session_ids.append(f"codex-session:{native_id}")
    return tuple(session_ids)


def _prepared_work_budget_s(archive_root: Path) -> float:
    """Client request budget scaled to the sessions this test actually staged.

    Anti-vacuity: replacing this with a constant re-introduces the
    load-sensitive failure the budget exists to avoid -- a 513-session prepare
    under host contention exceeds any small constant and surfaces as
    ``DaemonMutationIndeterminateError`` on the prepare route.
    """

    with sqlite3.connect(archive_root / "index.db") as conn:
        prepared = int(conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0])
    # Floor covers daemon start-up and the single-session routes; the per-session
    # term keeps the 513-session prepares inside the budget on a busy host while
    # staying under the suite-wide 120s pytest guard.
    return max(15.0, 0.2 * prepared)


@contextlib.contextmanager
def _delete_authority_daemon(
    monkeypatch: pytest.MonkeyPatch,
    archive_root: Path,
    *,
    server_error_sink: queue.SimpleQueue[str] | None = None,
) -> Iterator[_DeleteDaemonClient]:
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(archive_root))
    with running_daemon_operations(archive_root, server_error_sink=server_error_sink) as stack:
        stack.server.auth_token = "delete-authority-token"
        stack.client.auth_token = "delete-authority-token"
        client = cast(_DeleteDaemonClient, stack.client)
        object.__setattr__(client, "archive_root", archive_root)
        object.__setattr__(client, "authority_server", stack.server)
        # These routes delete hundreds of sessions through a real daemon, and
        # one request's cost is proportional to the prepared archive. A
        # constant budget measures how loaded the host is, not the route
        # (polylogue-ga8vn): 2.0s failed under load, and any raised constant
        # is the same defect with a bigger number. Derive the budget from the
        # work actually staged in this archive. A genuine hang is still caught
        # by the suite-wide pytest timeout.
        client.timeout_s = _prepared_work_budget_s(archive_root)
        yield client


def _delete_operation(client: _DeleteDaemonClient, step: str, body: dict[str, object]) -> dict[str, object]:
    """Drive one delete-lifecycle step over the declared operation envelope.

    ``step`` is the lifecycle stage — ``preview``, ``authorize``, ``cancel``
    or ``execute``. A typed envelope error is re-raised in the shape the
    surrounding assertions read.
    """
    from polylogue.operations.daemon_errors import DaemonResponseError

    operation = f"mutation.session.delete.{step}"
    envelope = client.operation_to_completion(
        operation,
        body,
        archive_root=str(client.archive_root),
    )
    assert envelope is not None, f"daemon did not answer {step}"
    error = envelope.get("error")
    if error:
        data = error.get("data") or {}
        raise DaemonResponseError(
            status=client.last_status or 0,
            code=error.get("code"),
            detail=error.get("detail"),
            payload={"error": error.get("code"), "detail": error.get("detail"), **data},
        )
    result = envelope.get("result")
    assert isinstance(result, dict), envelope
    return result


def _prepare_authorize(client: _DeleteDaemonClient, session_ids: tuple[str, ...]) -> str:
    preview = _delete_operation(client, "preview", {"session_ids": list(session_ids)})
    assert preview is not None
    assert preview["session_count"] == len(session_ids)
    assert preview["session_ids_sample"] == list(session_ids[:20])
    authorization = _delete_operation(client, "authorize", {"preview_ref": preview["preview_ref"]})
    assert authorization is not None
    return str(authorization["authorization_ref"])


def _assert_session_exists(archive_root: Path, session_id: str, *, expected: bool) -> None:
    with sqlite3.connect(archive_root / "index.db") as conn:
        count = conn.execute("SELECT COUNT(*) FROM sessions WHERE session_id = ?", (session_id,)).fetchone()[0]
    assert bool(count) is expected


def test_cli_delete_uses_real_uds_client_api_authority_and_audit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Real daemon HTTP/client/API proof of prepared, single-use delete authority."""

    from polylogue.operations.daemon_errors import DaemonResponseError

    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    success_id, replay_id, substitute_id, stale_a, stale_b, expiry_id = _seed_delete_authority_archive(archive_root, 6)
    with _delete_authority_daemon(monkeypatch, archive_root) as client:
        success_ref = _prepare_authorize(client, (success_id,))
        result = _delete_operation(client, "execute", {"authorization_ref": success_ref})
        _assert_completed_delete(result, affected=1, chunks=1)
        _assert_session_exists(archive_root, success_id, expected=False)

        with pytest.raises(ValueError, match="invalid DeleteExecuteRequest payload"):
            _delete_operation(client, "execute", {"session_ids": [replay_id]})
        _assert_session_exists(archive_root, replay_id, expected=True)

        replay_ref = _prepare_authorize(client, (replay_id,))
        _delete_operation(client, "execute", {"authorization_ref": replay_ref})
        with pytest.raises(DaemonResponseError):
            _delete_operation(client, "execute", {"authorization_ref": replay_ref})
        _assert_session_exists(archive_root, substitute_id, expected=True)

        substitute_ref = _prepare_authorize(client, (substitute_id,))
        with pytest.raises(ValueError, match="invalid DeleteExecuteRequest payload"):
            _delete_operation(client, "execute", {"authorization_ref": substitute_ref, "session_ids": [stale_a]})
        _assert_session_exists(archive_root, substitute_id, expected=True)
        _delete_operation(client, "execute", {"authorization_ref": substitute_ref})

        stale_ref = _prepare_authorize(client, (stale_a, stale_b))
        intervening_ref = _prepare_authorize(client, (stale_a,))
        _delete_operation(client, "execute", {"authorization_ref": intervening_ref})
        # Selection validation runs after durable batch acceptance. Its
        # refusal is a recoverable no-effect result, not an HTTP rejection.
        stale_result = _delete_operation(client, "execute", {"authorization_ref": stale_ref})
        assert stale_result["outcome"] == "failed"
        assert stale_result["effect"] == "no-effect"
        assert stale_result["completed_chunks"] == 0
        assert stale_result["affected_count"] == 0
        assert stale_result["not_attempted"] == [0]
        assert stale_result["parts"] == []
        assert stale_result["stop_reason"] == "refused"
        _assert_session_exists(archive_root, stale_b, expected=True)

        expiry_ref = _prepare_authorize(client, (expiry_id,))
        client.authority_server.auth_token = "different-authenticated-principal"
        client.auth_token = "different-authenticated-principal"
        with pytest.raises(DaemonResponseError):
            _delete_operation(client, "execute", {"authorization_ref": expiry_ref})
        client.authority_server.auth_token = "delete-authority-token"
        client.auth_token = "delete-authority-token"
        _assert_session_exists(archive_root, expiry_id, expected=True)
        with sqlite3.connect(archive_root / "audit.db") as conn:
            conn.execute(
                "UPDATE operation_authorizations SET issued_at_ms = 0, expires_at_ms = 1 WHERE authorization_id = ?",
                (expiry_ref,),
            )
        with pytest.raises(DaemonResponseError):
            _delete_operation(client, "execute", {"authorization_ref": expiry_ref})
        _assert_session_exists(archive_root, expiry_id, expected=True)

    expected_actor = f"daemon:bearer:{hashlib.sha256(b'delete-authority-token').hexdigest()}"
    with sqlite3.connect(archive_root / "audit.db") as conn:
        run = conn.execute(
            """
            SELECT r.actor_ref, r.surface, r.status
            FROM operation_runs AS r
            JOIN operation_targets AS t ON t.operation_id = r.operation_id
            WHERE t.target_ref = ?
            """,
            (f"session:{success_id}",),
        ).fetchone()
        confirmation = conn.execute(
            "SELECT confirmation_strength FROM operation_authorizations WHERE actor_ref = ? ORDER BY issued_at_ms LIMIT 1",
            (expected_actor,),
        ).fetchone()
    assert run == (expected_actor, "cli", "completed")
    assert confirmation == ("bound_token",)


def _operation_runs(archive_root: Path) -> list[tuple[str, str, str, int]]:
    with sqlite3.connect(f"file:{archive_root / 'audit.db'}?mode=ro", uri=True) as conn:
        return [
            (str(row[0]), str(row[1]), str(row[2]), int(row[3]))
            for row in conn.execute(
                "SELECT operation_name, surface, status, affected_count FROM operation_runs ORDER BY operation_name"
            )
        ]


@pytest.mark.parametrize(
    ("operation", "payload_key", "values", "operation_name"),
    [
        ("mutation.session.tag", "tags", ["triage"], "mutate-bulk-tag-sessions"),
        ("mutation.session.metadata", "pairs", [["lane", "triage"]], "mutate-bulk-set-metadata"),
    ],
)
def test_matched_session_mutation_runs_under_the_daemon_authority(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    operation: str,
    payload_key: str,
    values: list[object],
    operation_name: str,
) -> None:
    """The daemon owns the tag/metadata write and journals one CLI operation run.

    Anti-vacuity: an adapter that wrote ``user.db`` without the executor leaves
    no ``operation_runs`` row, so the journal assertion goes red.
    """
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    session_ids = _seed_delete_authority_archive(archive_root, 2)
    server_errors: queue.SimpleQueue[str] = queue.SimpleQueue()

    with _delete_authority_daemon(monkeypatch, archive_root, server_error_sink=server_errors) as client:
        try:
            envelope = client.operation_to_completion(
                operation,
                {"session_ids": list(session_ids), payload_key: values},
                archive_root=str(archive_root),
            )
        except DaemonMutationIndeterminateError as exc:
            try:
                server_error = server_errors.get(timeout=1)
            except queue.Empty:
                raise exc from None
            pytest.fail(f"machine operation handler failed:\n{server_error}")

    assert envelope is not None
    assert envelope.get("error") is None, envelope
    assert envelope["outcome"] == "completed"
    assert envelope["result"]["affected_count"] == 2
    assert _operation_runs(archive_root) == [(operation_name, "cli", "completed", 2)]


def test_matched_session_tag_removal_runs_under_the_daemon_authority(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``mark --tag-remove`` retracts through the daemon under the CLI surface.

    The add payload lowers onto ``mutate-bulk-tag-sessions``; the remove
    payload lowers onto the per-target ``mutate-remove-tag`` actuator, whose
    surface boundary is declared separately.

    Anti-vacuity: drop ``allowed_surfaces`` from the ``mutate-remove-tag``
    spec and the API-only back-fill refuses the ``cli`` principal with
    ``SurfaceDeniedError``, so the removal never completes (polylogue-7jxps).
    """
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    session_ids = _seed_delete_authority_archive(archive_root, 2)
    server_errors: queue.SimpleQueue[str] = queue.SimpleQueue()

    with _delete_authority_daemon(monkeypatch, archive_root, server_error_sink=server_errors) as client:
        try:
            added = client.operation_to_completion(
                "mutation.session.tag",
                {"session_ids": list(session_ids), "tags": ["triage"]},
                archive_root=str(archive_root),
            )
            removed = client.operation_to_completion(
                "mutation.session.tag",
                {"session_ids": list(session_ids), "remove_tags": ["triage"]},
                archive_root=str(archive_root),
            )
        except DaemonMutationIndeterminateError as exc:
            try:
                server_error = server_errors.get(timeout=1)
            except queue.Empty:
                raise exc from None
            pytest.fail(f"machine operation handler failed:\n{server_error}")

    assert added is not None and added["outcome"] == "completed", added
    assert removed is not None
    assert removed.get("error") is None, removed
    assert removed["outcome"] == "completed"
    assert removed["result"]["affected_count"] == 2
    runs = _operation_runs(archive_root)
    assert ("mutate-remove-tag", "cli", "completed", 1) in runs, runs
    assert sum(1 for run in runs if run[0] == "mutate-remove-tag") == 2


def test_matched_session_mutation_refuses_a_malformed_selection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    _seed_delete_authority_archive(archive_root, 1)

    with _delete_authority_daemon(monkeypatch, archive_root) as client:
        with patch.object(
            client,
            "_request_json_response",
            side_effect=AssertionError("client-side payload validation must precede the daemon request"),
        ):
            with pytest.raises(ValueError, match="invalid SessionTagRequest payload"):
                client.operation_to_completion(
                    "mutation.session.tag",
                    {"session_ids": [], "tags": ["triage"]},
                    archive_root=str(archive_root),
                )

    assert _operation_runs(archive_root) == []


def test_cli_delete_real_daemon_route_cancels_an_unconfirmed_preview(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A declined CLI confirmation has the daemon retire its durable preview."""

    from polylogue.operations.daemon_errors import DaemonResponseError

    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    (session_id,) = _seed_delete_authority_archive(archive_root, 1)

    with _delete_authority_daemon(monkeypatch, archive_root) as client:
        preview = _delete_operation(client, "preview", {"session_ids": [session_id]})
        assert preview is not None
        preview_ref = str(preview["preview_ref"])
        cancelled = _delete_operation(client, "cancel", {"preview_ref": preview_ref})
        assert cancelled["status"] == "cancelled"
        assert cancelled["source_request_id"] == cast(dict[str, object], preview["reference"])["request_id"]
        assert cast(dict[str, object], cancelled["reference"])["part_count"] == 1
        with pytest.raises(DaemonResponseError) as authorization_error:
            _delete_operation(client, "authorize", {"preview_ref": preview_ref})

    assert authorization_error.value.status == HTTPStatus.CONFLICT
    _assert_session_exists(archive_root, session_id, expected=True)
    with sqlite3.connect(archive_root / "audit.db") as conn:
        assert conn.execute("SELECT state FROM operation_previews WHERE preview_id = ?", (preview_ref,)).fetchone() == (
            "cancelled",
        )


def test_cli_delete_real_daemon_route_cancels_an_expired_preview(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A late explicit decline terminalizes the exact durable preview."""

    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    (session_id,) = _seed_delete_authority_archive(archive_root, 1)

    with _delete_authority_daemon(monkeypatch, archive_root) as client:
        preview = _delete_operation(client, "preview", {"session_ids": [session_id]})
        assert preview is not None
        preview_ref = str(preview["preview_ref"])
        with sqlite3.connect(archive_root / "audit.db") as conn:
            conn.execute(
                "UPDATE operation_previews SET created_at_ms = 0, expires_at_ms = 1 WHERE preview_id = ?",
                (preview_ref,),
            )

        cancelled = _delete_operation(client, "cancel", {"preview_ref": preview_ref})

    assert cancelled["status"] == "cancelled"
    assert cancelled["source_request_id"] == cast(dict[str, object], preview["reference"])["request_id"]
    assert cast(dict[str, object], cancelled["reference"])["part_count"] == 1
    _assert_session_exists(archive_root, session_id, expected=True)
    with sqlite3.connect(archive_root / "audit.db") as conn:
        assert conn.execute("SELECT state FROM operation_previews WHERE preview_id = ?", (preview_ref,)).fetchone() == (
            "cancelled",
        )


def _operation_handler(timeline: list[str], body: bytes, *, content_length: int | None = None) -> DaemonAPIHandler:
    """Build a real handler for ``POST /api/operation`` over a fake socket."""

    handler = _handler(["api", "operation"], timeline)
    connection, peer = socket.socketpair()

    class _RecordingOperationRuntime:
        def call(self, request: object, _principal: object, **_kwargs: object) -> dict[str, object]:
            timeline.append(f"runtime:{getattr(request, 'operation', 'unknown')}")
            return {"outcome": "completed"}

    object.__setattr__(
        handler,
        "server",
        SimpleNamespace(write_bridge=_RecordingBridge(timeline), operation_runtime=_RecordingOperationRuntime()),
    )
    object.__setattr__(handler, "connection", connection)
    # Retain the other endpoint so the disconnect observer sees an open peer.
    object.__setattr__(handler, "_test_peer", peer)
    object.__setattr__(
        handler,
        "headers",
        Message(),
    )
    headers = handler.headers
    headers["Content-Type"] = "application/json"
    headers["Content-Length"] = str(len(body) if content_length is None else content_length)
    object.__setattr__(handler, "rfile", BytesIO(body))
    object.__setattr__(handler, "wfile", BytesIO())
    object.__setattr__(handler, "send_response", lambda _status: timeline.append("response"))
    object.__setattr__(handler, "send_header", lambda *_args: None)
    object.__setattr__(handler, "end_headers", lambda: None)
    return handler


def _preview_operation_body(session_ids: list[str]) -> bytes:
    from polylogue.operations.daemon_protocol import DAEMON_OPERATION_PROTOCOL

    return json.dumps(
        {
            "protocol": DAEMON_OPERATION_PROTOCOL,
            "operation": "mutation.session.delete.preview",
            "payload": {"session_ids": session_ids},
            "request_id": f"req-{len(session_ids)}",
        }
    ).encode()


def test_delete_preview_checks_headers_before_reading_and_dispatches_after_body(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Invalid framing refuses before reading; valid dispatch follows the full body.

    Reading before header admission triggers the exploding body; dispatching
    before body completion changes the recorded slow-body event order.
    """
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(tmp_path))

    class _ExplodingBody:
        def read(self, _size: int) -> bytes:
            raise AssertionError("invalid framing must be refused before the body is read")

    invalid_timeline: list[str] = []
    invalid = _operation_handler(invalid_timeline, b"")
    invalid.headers.replace_header("Content-Length", "invalid")
    object.__setattr__(invalid, "rfile", _ExplodingBody())
    invalid._do_post_impl()
    assert invalid_timeline == ["error"]

    large_timeline: list[str] = []
    large_body = _preview_operation_body([f"codex-session:{index}" for index in range(257)])
    large = _operation_handler(large_timeline, large_body)
    large._do_post_impl()
    assert large_timeline == ["runtime:mutation.session.delete.preview", "response"]

    slow_timeline: list[str] = []
    slow_body = _preview_operation_body(["codex-session:slow"])

    class _SlowBody:
        def read(self, _size: int) -> bytes:
            slow_timeline.append("body-read")
            assert not any(item.startswith("runtime:") for item in slow_timeline)
            return slow_body

    slow = _operation_handler(slow_timeline, b"", content_length=len(slow_body))
    object.__setattr__(slow, "rfile", _SlowBody())
    slow._do_post_impl()
    assert slow_timeline == [
        "body-read",
        "runtime:mutation.session.delete.preview",
        "response",
    ]


def test_conflicting_operation_request_id_never_reenters_the_replay_lock(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A conflicting retry maps the resident runtime rejection to HTTP 409.

    Anti-vacuity: a legacy fake server without ``operation_runtime`` turns this
    current conflict contract into an internal server error instead.
    """
    from polylogue.core.compute import CancellationHandle
    from polylogue.operations.daemon_protocol import DaemonOperationRequest
    from polylogue.operations.mutation_transaction import MutationPrincipal

    class _OperationConflictRuntime:
        def __init__(self) -> None:
            self.calls: list[DaemonOperationRequest] = []

        def call(
            self,
            request: DaemonOperationRequest,
            principal: MutationPrincipal,
            *,
            client_disconnect: CancellationHandle | None = None,
        ) -> dict[str, object]:
            del principal, client_disconnect
            self.calls.append(request)
            return {
                "outcome": "rejected",
                "error": {"code": "request_identity_conflict", "retryable": False},
            }

    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(tmp_path))
    conflicting = DaemonOperationRequest.from_dict(
        DaemonOperationRequest(
            operation="completion",
            payload={"incomplete": "x"},
            request_id="request-id-reused",
        ).to_dict()
    )
    body = json.dumps(conflicting.to_dict()).encode()
    handler = _operation_handler([], body)
    runtime = _OperationConflictRuntime()
    object.__setattr__(handler.server, "operation_runtime", runtime)
    statuses: list[HTTPStatus] = []
    headers: dict[str, str] = {}
    object.__setattr__(handler, "send_response", statuses.append)
    object.__setattr__(handler, "send_header", lambda name, value: headers.__setitem__(name, value))

    failure: list[BaseException] = []

    def invoke() -> None:
        try:
            handler._handle_daemon_operation()
        except BaseException as exc:  # pragma: no cover - assertion below reports it
            failure.append(exc)

    thread = threading.Thread(target=invoke, daemon=True)
    thread.start()
    thread.join(timeout=1.0)

    assert not thread.is_alive(), "conflicting duplicate request id deadlocked the machine endpoint"
    assert failure == []
    assert statuses == [HTTPStatus.CONFLICT]
    assert runtime.calls == [conflicting]
    body = cast(BytesIO, handler.wfile).getvalue()
    assert headers["Content-Type"] == "application/json"
    assert int(headers["Content-Length"]) == len(body)
    response = json.loads(body)
    assert response["outcome"] == "rejected"
    assert response["error"] == {
        "code": "request_identity_conflict",
        "retryable": False,
    }


def _assert_completed_delete(result: dict[str, object], *, affected: int, chunks: int) -> None:
    assert result["outcome"] == "completed"
    assert result["effect"] == "committed"
    assert result["affected_count"] == affected
    assert result["completed_chunks"] == chunks
    assert result["not_attempted"] == []
    assert result["stop_reason"] is None
    parts = result["parts"]
    assert isinstance(parts, list)
    assert [(part["ordinal"], part["outcome"]) for part in parts] == [
        (ordinal, "completed") for ordinal in range(chunks)
    ]
    assert all(part["operation_id"] for part in parts)


def test_cli_delete_real_daemon_route_deletes_a_selection_larger_than_legacy_cap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """513 sessions delete through the real daemon route in three chunks.

    Anti-vacuity: restoring the old single-chunk cap (or losing the chunked
    preview/authorize/execute lifecycle) leaves rows in ``sessions`` and the
    chunk-count assertions fail. Echoing the whole selection in the preview
    result (which grows past the operation result bound for a large accepted
    selection) fails the sample assertions. The client budget comes from
    ``_prepared_work_budget_s`` so the route's behavior, not the host's
    current load, decides the outcome (polylogue-ga8vn).
    """

    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    session_ids = _seed_delete_authority_archive(archive_root, 513)

    with _delete_authority_daemon(monkeypatch, archive_root) as client:
        preview = _delete_operation(client, "preview", {"session_ids": list(session_ids)})
        assert preview is not None
        assert preview["session_count"] == 513
        assert preview["session_ids_sample"] == list(session_ids[:20])
        assert "session_ids" not in preview
        preview_id = str(cast(dict[str, object], preview["reference"])["request_id"])
        assert cast(dict[str, object], preview["reference"])["part_count"] == 3
        authorization = _delete_operation(client, "authorize", {"preview_request_id": preview_id})
        assert authorization is not None
        authorization_id = str(cast(dict[str, object], authorization["reference"])["request_id"])
        assert cast(dict[str, object], authorization["reference"])["part_count"] == 3
        result = _delete_operation(client, "execute", {"authorization_request_id": authorization_id})

    _assert_completed_delete(result, affected=513, chunks=3)
    with sqlite3.connect(archive_root / "index.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone() == (0,)


def test_cli_delete_real_daemon_route_reports_partial_chunk_application(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A delete that fails after two committed chunks reports what it applied.

    Anti-vacuity: surface the failure as a refusal without applied counts
    (the original behaviour, where only a fully successful ``sum(...)`` over
    chunks reported any), stop accumulating ``completed``/``affected`` across
    parts in ``machine_lifecycle.machine_request_state``, or settle the
    failed part as ``failed`` rather than ``indeterminate``, and the reported
    counts or outcome here go red.
    """
    from polylogue.operations.audit import AuditRepository
    from polylogue.operations.machine_lifecycle import machine_request_state
    from polylogue.operations.mutation_actuators import SessionDeleteActuator, SessionDeleteArgs
    from polylogue.operations.mutation_transaction import MutationPlan, MutationReceipt

    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    session_ids = _seed_delete_authority_archive(archive_root, 513)

    with _delete_authority_daemon(monkeypatch, archive_root) as client:
        preview = _delete_operation(client, "preview", {"session_ids": list(session_ids)})
        assert preview is not None
        preview_id = str(cast(dict[str, object], preview["reference"])["request_id"])
        authorization = _delete_operation(client, "authorize", {"preview_request_id": preview_id})
        assert authorization is not None
        authorization_id = str(cast(dict[str, object], authorization["reference"])["request_id"])

        original_apply = SessionDeleteActuator.apply
        completed_applies = 0

        def fail_third_apply(
            actuator: SessionDeleteActuator, plan: MutationPlan, args: SessionDeleteArgs
        ) -> MutationReceipt:
            nonlocal completed_applies
            if completed_applies == 2:
                raise RuntimeError("synthetic third delete chunk failure")
            completed_applies += 1
            return original_apply(actuator, plan, args)

        # The daemon executes the declared SessionDeleteActuator through the
        # audited machine lifecycle, not the retired direct helper.  A fault
        # after two durable effects leaves the final attempt unknown rather
        # than inventing a retry-safe HTTP refusal.
        with patch.object(SessionDeleteActuator, "apply", new=fail_third_apply):
            result = _delete_operation(client, "execute", {"authorization_request_id": authorization_id})

    assert completed_applies == 2
    assert result["outcome"] == "indeterminate"
    assert result["effect"] == "indeterminate"
    assert result["completed_chunks"] == 2
    assert result["affected_count"] == 512
    # An indeterminate reply can precede the batch's final suffix fence. After
    # daemon teardown drains the writer, recovery must retain both the known
    # effects and that fence, without making the unknown part retry-safe.
    reference = result["reference"]
    assert isinstance(reference, dict)
    audit = AuditRepository.for_archive_root(archive_root)
    with audit.settled_machine_read():
        record = audit.machine_request_for_principal(
            reference["archive_identity"], reference["request_id"], reference["principal_ref"]
        )
        assert record is not None
        settled = machine_request_state(audit, record)
    assert settled["outcome"] == "indeterminate"
    assert settled["effect"] == "indeterminate"
    assert settled["completed_chunks"] == 2
    assert settled["affected_count"] == 512
    assert settled["not_attempted"] == []
    assert settled["stop_reason"] == "refused"
    parts = settled["parts"]
    assert isinstance(parts, list)
    assert [(part["ordinal"], part["outcome"]) for part in parts] == [
        (0, "completed"),
        (1, "completed"),
        (2, "indeterminate"),
    ]
    with sqlite3.connect(archive_root / "index.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone() == (1,)


def test_delete_protocol_accepts_selections_of_any_size() -> None:
    """Large explicit selections remain valid; follow-up requests name their owner."""
    from polylogue.operations.daemon_protocol import (
        DeleteAuthorizeRequest,
        DeleteCancelRequest,
        DeleteExecuteRequest,
        DeletePreviewRequest,
    )

    selection = [f"codex-session:large-{index}" for index in range(10_001)]
    assert len(DeletePreviewRequest.model_validate({"session_ids": selection}).session_ids or ()) == 10_001
    assert (
        DeleteAuthorizeRequest.model_validate({"preview_request_id": "preview-owner"}).preview_request_id
        == "preview-owner"
    )
    assert (
        DeleteCancelRequest.model_validate({"preview_request_id": "preview-owner"}).preview_request_id
        == "preview-owner"
    )
    assert (
        DeleteExecuteRequest.model_validate(
            {"authorization_request_id": "authorization-owner"}
        ).authorization_request_id
        == "authorization-owner"
    )


def test_cli_delete_preparation_resolves_canonical_ids_in_bounded_pages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The actual resident selection resolves 513 exact IDs in three bounded reads."""
    from polylogue.operations import daemon_mutations
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    root = tmp_path / "archive"
    root.mkdir()
    ids = _seed_delete_authority_archive(root, 513)
    original_select = daemon_mutations._prepare_mutation_selection
    original_resolve = ArchiveStore.resolve_exact_session_ids
    selecting = threading.local()
    sizes: list[int] = []

    def select(*args: Any, **kwargs: Any) -> Any:
        selecting.active = True
        try:
            return original_select(*args, **kwargs)
        finally:
            selecting.active = False

    def resolve(archive: ArchiveStore, session_ids: Any, **kwargs: Any) -> Any:
        if getattr(selecting, "active", False):
            sizes.append(len(session_ids))
        return original_resolve(archive, session_ids, **kwargs)

    monkeypatch.setattr(daemon_mutations, "_prepare_mutation_selection", select)
    monkeypatch.setattr(ArchiveStore, "resolve_exact_session_ids", resolve)
    with _delete_authority_daemon(monkeypatch, root) as client:
        preview = _delete_operation(client, "preview", {"session_ids": list(ids)})
    assert preview["session_count"] == len(ids)
    assert sizes == [256, 256, 1]


def test_cli_delete_preparation_refuses_a_missing_exact_id_that_is_a_live_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polylogue.operations.daemon_errors import DaemonResponseError

    root = tmp_path / "archive"
    root.mkdir()
    (session_id,) = _seed_delete_authority_archive(root, 1)
    with _delete_authority_daemon(monkeypatch, root) as client:
        with pytest.raises(DaemonResponseError) as refusal:
            _delete_operation(client, "preview", {"session_ids": [session_id.removesuffix("0")]})
    assert refusal.value.code == "selection_is_stale"
    _assert_session_exists(root, session_id, expected=True)


def test_cli_delete_preparation_rejects_a_late_duplicate_before_archive_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polylogue.operations import daemon_mutations

    root = tmp_path / "archive"
    root.mkdir()
    ids = _seed_delete_authority_archive(root, 513)
    with _delete_authority_daemon(monkeypatch, root) as client:
        with patch.object(
            daemon_mutations,
            "_prepare_mutation_selection",
            side_effect=AssertionError("duplicate reached archive selection"),
        ) as selection:
            with pytest.raises(ValueError):
                _delete_operation(client, "preview", {"session_ids": list(ids + (ids[0],))})
        selection.assert_not_called()


def test_cli_delete_interruption_consumes_authorization_without_deleting(tmp_path: Path) -> None:
    """An interrupted bound actuator consumes its actual UDS-issued authority once."""
    from polylogue.operations.audit import AuditRepository
    from polylogue.operations.mutation_actuators import SessionDeleteActuator
    from polylogue.operations.mutation_transaction import TokenConsumedError
    from tests.infra.daemon_operations import execute_bound_delete, prepare_bound_delete

    root = tmp_path / "archive"
    root.mkdir()
    (session_id,) = _seed_delete_authority_archive(root, 1)
    with running_daemon_operations(root) as stack:
        preview, authorization, principal = prepare_bound_delete(stack, (session_id,))
        with patch.object(
            SessionDeleteActuator, "apply", side_effect=RuntimeError("interrupted before apply")
        ) as apply:
            with pytest.raises(RuntimeError, match="interrupted before apply"):
                execute_bound_delete(stack, preview, authorization, principal)
        _assert_session_exists(root, session_id, expected=True)

        # Test the durable one-shot boundary directly: another executor first
        # runs legitimate recovery of the original unknown intent, which is
        # different from granting a second authorization consumption.
        def consume_again() -> str | None:
            return AuditRepository(root / "audit.db").consume_authorization_and_start(preview, authorization)

        with pytest.raises(TokenConsumedError):
            stack.write_bridge.run_sync("test.delete.consumed-authority", consume_again)
        apply.assert_called_once()
    with sqlite3.connect(root / "audit.db") as conn:
        state = conn.execute(
            "SELECT state, unknown_reason FROM operation_attempts ORDER BY started_at_ms DESC LIMIT 1"
        ).fetchone()
        consumed = conn.execute(
            "SELECT state, consumed_at_ms FROM operation_authorizations WHERE authorization_id = ?",
            (authorization.authorization_id,),
        ).fetchone()
        attempts = conn.execute("SELECT count(*) FROM operation_attempts").fetchone()
    assert state == ("unknown", "actuator exception after durable intent")
    assert consumed is not None and consumed[0] == "consumed" and consumed[1] is not None
    assert attempts == (1,)
    _assert_session_exists(root, session_id, expected=True)


def test_cli_delete_preserves_audit_finalization_failure_after_effect(tmp_path: Path) -> None:
    from polylogue.operations.audit import AuditRepository
    from polylogue.operations.mutation_transaction import AuditFinalizationError
    from tests.infra.daemon_operations import execute_bound_delete, prepare_bound_delete

    root = tmp_path / "archive"
    root.mkdir()
    (session_id,) = _seed_delete_authority_archive(root, 1)
    with running_daemon_operations(root) as stack:
        preview, authorization, principal = prepare_bound_delete(stack, (session_id,))
        with patch.object(AuditRepository, "finalize_attempt", side_effect=RuntimeError("audit unavailable")):
            with pytest.raises(AuditFinalizationError):
                execute_bound_delete(stack, preview, authorization, principal)
        _assert_session_exists(root, session_id, expected=False)


def test_no_auth_cli_principal_ignores_attacker_selected_bearer_text() -> None:
    handler = _handler(["api", "cli", "delete", "prepare"], [])
    object.__setattr__(handler, "headers", {"Authorization": "Bearer attacker-selected"})

    principal = handler._cli_mutation_principal("archive.delete_session")

    assert principal.actor_ref == "daemon:unauthenticated-loopback"
    assert principal.role_label == "daemon-loopback-no-auth"


def test_user_post_and_delete_delegate_writer_ownership_to_operation_runtime() -> None:
    post_timeline: list[str] = []
    post_handler = _handler(["api", "user", "marks"], post_timeline)
    post_handler._handle_user_overlay_post = lambda *_args: post_timeline.append("body")  # type: ignore[method-assign]
    post_handler._do_post_impl()
    assert post_timeline == ["body"]

    delete_timeline: list[str] = []
    delete_handler = _handler(["api", "user", "annotations", "ann-1"], delete_timeline)
    delete_handler._handle_user_overlay_delete = lambda *_args: delete_timeline.append("body")  # type: ignore[method-assign]
    delete_handler._do_delete_impl()
    assert delete_timeline == ["body"]


def test_standalone_http_server_owns_and_idempotently_closes_writer_runtime(tmp_path: Path) -> None:
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    initialize_active_archive_root(tmp_path)
    server = DaemonAPIHTTPServer(("127.0.0.1", 0), DaemonAPIHandler, archive_root=tmp_path)
    runtime = server._owned_write_runtime
    assert runtime is not None
    assert runtime.thread.is_alive()

    server.server_close()
    server.server_close()

    assert not runtime.thread.is_alive()


def test_standalone_http_server_stops_loop_after_late_writer_drain(tmp_path: Path) -> None:
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    initialize_active_archive_root(tmp_path)
    server = DaemonAPIHTTPServer(("127.0.0.1", 0), DaemonAPIHandler, archive_root=tmp_path)
    runtime = server._owned_write_runtime
    assert runtime is not None
    assert runtime.coordinator is not None
    shutdown = AsyncMock(side_effect=[False, True])

    with patch.object(runtime.coordinator, "shutdown", shutdown):
        server.server_close()
        runtime.thread.join(timeout=1.0)

    assert not runtime.thread.is_alive()
    assert shutdown.await_count == 2


@pytest.mark.uses_real_clock("the admitted HTTP body executes on an actual writer worker")
def test_coordinated_mutation_uses_existing_writer_worker_without_compute_admission(tmp_path: Path) -> None:
    _coordinator, bridge, stop = _loop_owned_bridge(tmp_path)
    handler = _gated_handler(bridge)
    workers: list[str] = []

    async def mutation(_polylogue: object) -> str:
        from polylogue.core.write_lease import coordinator_write_lease_active

        assert coordinator_write_lease_active()

        async def inherited_child() -> bool:
            return coordinator_write_lease_active()

        assert not await asyncio.create_task(inherited_child())
        assert coordinator_write_lease_active()
        workers.append(threading.current_thread().name)
        return "persisted"

    try:
        with handler._write_gate("test.http.worker"):
            assert handler._sync_run(mutation) == "persisted"
        assert workers == ["polylogue-writer:test.http.worker"]
        snapshot = handler.server.execution_kernel.snapshot()
        assert snapshot.by_class("control").admitted == 0
        assert snapshot.by_class("interactive-read").admitted == 0
        assert snapshot.used_units == 0
    finally:
        handler.server.execution_kernel.shutdown(wait=True)
        stop()


def _loop_owned_bridge(
    archive_root: Path,
) -> tuple[DaemonWriteCoordinator, DaemonWriteThreadBridge, Callable[[], None]]:
    loop = asyncio.new_event_loop()
    ready = threading.Event()
    holder: list[DaemonWriteCoordinator] = []

    def run_loop() -> None:
        asyncio.set_event_loop(loop)
        holder.append(DaemonWriteCoordinator(archive_root=archive_root))
        ready.set()
        loop.run_forever()

    thread = threading.Thread(target=run_loop, daemon=True)
    thread.start()
    assert ready.wait(timeout=5.0)

    def stop() -> None:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5.0)

    return holder[0], DaemonWriteThreadBridge(holder[0], loop, timeout=5.0), stop


def _gated_handler(bridge: object) -> DaemonAPIHandler:
    from polylogue.core.compute import BoundedComputeAdapter

    handler = object.__new__(DaemonAPIHandler)

    async def run_direct(operation: Callable[[object], Awaitable[object]]) -> object:
        return await operation(None)

    object.__setattr__(handler, "_run_archive_query", run_direct)
    object.__setattr__(
        handler,
        "server",
        SimpleNamespace(
            execution_kernel=BoundedComputeAdapter(max_workers=2, queue_units=4),
            write_bridge=bridge,
        ),
    )
    return handler


def test_gated_route_body_is_authorized_to_write_under_process_wide_enforcement(tmp_path: Path) -> None:
    """The legacy gate admits its route body under process-wide enforcement.

    polylogue-h5l6i: ``hold()`` entered the lease in a coroutine on the owner
    loop while the route body ran on a kernel worker in a freshly created event
    loop, so a gated handler lost its admitted writer grant.

    Anti-vacuity: drop the actual child task adoption from
    the admitted writer worker and this returns ``"unleased"``.
    """
    from polylogue.storage.sqlite.write_lease import (
        arm_write_lease_enforcement,
        require_write_lease,
    )

    coordinator, bridge, stop = _loop_owned_bridge(tmp_path)
    del coordinator
    handler = _gated_handler(bridge)

    async def mutation(_polylogue: object) -> str:
        try:
            require_write_lease("write gated route body", archive_root=tmp_path)
        except Exception as exc:  # the refusal is the observation under test
            return f"unleased:{type(exc).__name__}"
        return "leased"

    try:
        with arm_write_lease_enforcement(process_wide=True):
            with handler._write_gate("http.reset"):
                assert handler._sync_run(mutation) == "leased"
    finally:
        handler.server.execution_kernel.shutdown(wait=True)
        stop()


def test_a_thread_outside_the_admitted_body_still_cannot_write(tmp_path: Path) -> None:
    """The load-bearing negative: holding the gate is not blanket authorization.

    While one request is admitted and its body is authorized, a thread that was
    never handed the gate's grant must still be refused -- otherwise the fix
    for polylogue-h5l6i would have traded a false refusal for a real second
    writer.

    Anti-vacuity: make ``_write_gate`` authorize ambiently instead (call
    ``bind_write_lease_thread()`` on every thread, or arm a bypass) and the
    rogue probe below returns ``"leased"``. The lease ContextVar *is* inherited
    by new threads on this build, so nothing but the explicit hand-off keeps
    it out.
    """
    from polylogue.storage.sqlite.write_lease import (
        UnleasedWriteError,
        arm_write_lease_enforcement,
        require_write_lease,
    )

    coordinator, bridge, stop = _loop_owned_bridge(tmp_path)
    del coordinator
    handler = _gated_handler(bridge)
    rogue_result: list[str] = []
    rogue_done = threading.Event()

    async def mutation(_polylogue: object) -> str:
        def rogue() -> None:
            try:
                require_write_lease("write user.db from an unadmitted thread", archive_root=tmp_path)
            except UnleasedWriteError:
                rogue_result.append("refused")
            else:
                rogue_result.append("leased")
            rogue_done.set()

        thread = threading.Thread(target=rogue)
        thread.start()
        thread.join(timeout=5.0)
        require_write_lease("write user.db annotation", archive_root=tmp_path)
        return "leased"

    try:
        with arm_write_lease_enforcement(process_wide=True):
            with handler._write_gate("http.reset"):
                assert handler._sync_run(mutation) == "leased"
    finally:
        handler.server.execution_kernel.shutdown(wait=True)
        stop()

    assert rogue_done.is_set()
    assert rogue_result == ["refused"]


@pytest.mark.uses_real_clock("actual inline Ops SQL executes and closes on the admitted writer worker")
def test_inline_ops_route_uses_admitted_worker_and_closes_actual_handles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from polylogue import paths
    from polylogue.storage.sqlite import connection_profile
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
    from polylogue.storage.sqlite.write_lease import arm_write_lease_enforcement

    initialize_active_archive_root(tmp_path)
    monkeypatch.setattr(paths, "archive_root", lambda: tmp_path)
    _coordinator, bridge, stop = _loop_owned_bridge(tmp_path)
    handler = _gated_handler(bridge)
    body = json.dumps(
        {
            "call_id": "neutral-call",
            "tool_name": "neutral-tool",
            "session_ids": [],
            "started_at_ms": 1,
            "finished_at_ms": 2,
            "success": True,
        }
    ).encode()
    handler.headers = Message()
    handler.headers["Content-Length"] = str(len(body))
    handler.rfile = BytesIO(body)
    replies: list[tuple[object, object]] = []
    object.__setattr__(handler, "_send_json", lambda status, result: replies.append((status, result)))
    object.__setattr__(handler, "_send_error", lambda *args: pytest.fail(f"unexpected refusal: {args}"))
    closed: list[tuple[threading.Thread, sqlite3.Connection]] = []
    real_close = connection_profile.NativeSQLCustodyOwner.close

    def observe_close(owner: connection_profile.NativeSQLCustodyOwner) -> None:
        connection = owner.connection
        assert connection is not None
        real_close(owner)
        with pytest.raises(sqlite3.ProgrammingError):
            connection.execute("SELECT 1")
        closed.append((threading.current_thread(), connection))

    monkeypatch.setattr(connection_profile.NativeSQLCustodyOwner, "close", observe_close)
    try:
        with arm_write_lease_enforcement(process_wide=True), handler._write_gate("http.telemetry.mcp-call"):
            handler._handle_mcp_call_log()
        assert replies == [(HTTPStatus.OK, {"ok": True, "call_id": "neutral-call"})]
        assert len(closed) == 2
        assert all(thread.name == "polylogue-writer:http.telemetry.mcp-call" for thread, _conn in closed)
        with contextlib.closing(sqlite3.connect(f"file:{tmp_path / 'ops.db'}?mode=ro", uri=True)) as reader:
            assert reader.execute("SELECT call_id FROM mcp_call_log").fetchall() == [("neutral-call",)]
        snapshot = handler.server.execution_kernel.snapshot()
        assert snapshot.by_class("control").admitted == 0
    finally:
        handler.server.execution_kernel.shutdown(wait=True)
        stop()


def test_tcp_pre_dispatch_refusal_is_marked_rejected_not_indeterminate() -> None:
    """A TCP ``/api/operation`` refusal before dispatch carries the pre-dispatch marker.

    polylogue-ji49p: the UDS transport marks its refusals and
    ``DaemonClient.operation`` raises ``DaemonOperationRejectedError`` for ANY
    envelope with ``protocol``/``outcome == "rejected"``/``pre_dispatch``. The
    TCP route answered with the bare ``{"ok": false, "error": ...}`` shape, so
    a write refused before it ever ran was reported to the caller as a
    possibly-committed (indeterminate) mutation.

    Anti-vacuity: restore ``self._send_error(...)`` at the route's
    pre-dispatch refusals (or drop any one of the four marker fields from
    ``_reject_operation``) and the client-side predicate asserted below is
    false again, which is exactly the indeterminate-mutation fall-through.
    The runtime assertion fails if a refusal is emitted after dispatch.
    """
    from polylogue.operations.daemon_protocol import DAEMON_OPERATION_PROTOCOL

    timeline: list[str] = []
    body = _preview_operation_body(["codex-session:marked"])
    handler = _operation_handler(timeline, body)
    # ``Message.__setitem__`` APPENDS; the helper already set a JSON
    # Content-Type, and ``.get()`` would keep returning that first value.
    del handler.headers["Content-Type"]
    handler.headers["Content-Type"] = "text/plain"
    assert handler.headers.get("Content-Type") == "text/plain"
    # Restore the production error envelope builder over the stub, so this
    # asserts the real serialized refusal rather than a recording lambda.
    object.__setattr__(handler, "_send_error", DaemonAPIHandler._send_error.__get__(handler))
    sent: list[tuple[object, dict[str, object]]] = []
    object.__setattr__(handler, "_send_json", lambda status, payload, **_kwargs: sent.append((status, payload)))

    handler._do_post_impl()

    assert not any(item.startswith("runtime:") for item in timeline)
    assert len(sent) == 1
    status, payload = sent[0]
    assert status == HTTPStatus.UNSUPPORTED_MEDIA_TYPE
    assert payload["ok"] is False
    # The exact predicate DaemonClient.operation uses to refuse without
    # calling the mutation indeterminate.
    assert payload["protocol"] == DAEMON_OPERATION_PROTOCOL
    assert payload["outcome"] == "rejected"
    assert payload["pre_dispatch"] is True
    assert payload["error"] == {"code": "unsupported_media_type", "detail": None, "retryable": False}


def test_delete_preview_plan_is_reconstructed_by_its_audit_owner(tmp_path: Path) -> None:
    """The canonical Audit reader retains authored fields and refuses mismatched replay semantics."""
    from polylogue.operations.audit import AuditRepository
    from tests.infra.daemon_operations import prepare_bound_delete

    root = tmp_path / "archive"
    root.mkdir()
    (session_id,) = _seed_delete_authority_archive(root, 1)
    with running_daemon_operations(root) as stack:
        preview, _authorization, principal = prepare_bound_delete(stack, (session_id,))
        audit = AuditRepository(root / "audit.db")
        loaded = audit.preview_for_principal(preview.preview_ref, principal)
        assert loaded.plan.target_refs == (f"session:{session_id}",)
        assert loaded.plan.reversible is False

        def rewrite(mutate: Callable[[dict[str, object]], None]) -> None:
            with sqlite3.connect(root / "audit.db") as conn:
                stored = json.loads(
                    conn.execute(
                        "SELECT plan_json FROM operation_previews WHERE preview_id = ?", (preview.preview_ref,)
                    ).fetchone()[0]
                )
                mutate(stored)
                conn.execute(
                    "UPDATE operation_previews SET plan_json = ? WHERE preview_id = ?",
                    (json.dumps(stored, sort_keys=True, separators=(",", ":")), preview.preview_ref),
                )

        def reversible(document: dict[str, object]) -> None:
            document["reversible"] = True

        rewrite(reversible)
        assert audit.preview_for_principal(preview.preview_ref, principal).plan.reversible is True

        def wrong_context(document: dict[str, object]) -> None:
            document["reversible"] = False
            document["context"] = {"session_ids": ["codex-session:not-authorized"]}

        rewrite(wrong_context)
        with pytest.raises(ValueError, match="stored plan context differs from its replay semantics"):
            audit.preview_for_principal(preview.preview_ref, principal)


@pytest.mark.uses_real_clock("client deadline expires while its admitted writer continues to actual settlement")
def test_an_indeterminate_mutation_keeps_the_writer_gate_until_its_body_settles(tmp_path: Path) -> None:
    """The production route's bounded wait must not release a live writer.

    polylogue-8r4zq AC2, at the real seam: ``_write_gate`` wraps ``_sync_run``,
    so when the mutating wait hits its deadline and raises
    ``DaemonMutationIndeterminate``, the context manager unwinds while the
    submitted body is still adopted and still inside the archive. Releasing the
    coordinator gate there admits a second writer alongside it.

    Both halves matter: the client stays bounded by its own declared deadline
    (the request thread returns), and the archive stays single-writer (the
    successor waits for the abandoned body, not for the client).

    Anti-vacuity: remove the settlement retention from
    ``DaemonWriteThreadBridge.hold``'s release path and ``successor_entered``
    is set before ``allow_body``, turning the ``not ... wait(0.3)`` red.
    """
    from polylogue.daemon.http import DaemonMutationIndeterminate

    coordinator, bridge, stop = _loop_owned_bridge(tmp_path)
    handler = _gated_handler(bridge)
    # A short declared deadline is the whole point: the body outlives it.
    object.__setattr__(handler, "headers", {"X-Polylogue-Deadline-Ms": "50"})

    body_entered = threading.Event()
    allow_body = threading.Event()
    body_left = threading.Event()
    raised: list[BaseException] = []

    async def mutation(_polylogue: object) -> str:
        body_entered.set()
        await asyncio.to_thread(allow_body.wait, 5.0)
        body_left.set()
        return "persisted"

    def request() -> None:
        try:
            with handler._write_gate("http.reset"):
                handler._sync_run(mutation)
        except BaseException as exc:
            raised.append(exc)

    thread = threading.Thread(target=request, daemon=True)
    thread.start()
    try:
        assert body_entered.wait(timeout=5.0)
        thread.join(timeout=5.0)
        assert not thread.is_alive(), "the client's wait outlived its declared deadline"
        assert len(raised) == 1
        assert isinstance(raised[0], DaemonMutationIndeterminate)
        assert not body_left.is_set()

        successor_entered = threading.Event()

        async def successor() -> str:
            successor_entered.set()
            return "entered"

        successor_future = asyncio.run_coroutine_threadsafe(
            cast("Any", coordinator).run("maintenance.successor", successor),
            cast("Any", bridge).owner_loop,
        )
        assert not successor_entered.wait(timeout=0.3)

        allow_body.set()
        assert successor_future.result(timeout=5.0) == "entered"
        assert body_left.is_set()
    finally:
        allow_body.set()
        thread.join(timeout=5.0)
        handler.server.execution_kernel.shutdown(wait=True)
        stop()


def test_cli_delete_pages_every_phase_past_one_machine_batch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A selection larger than one machine batch is accepted, authorized and
    executed page by page (polylogue-zxbbl).

    The page and chunk sizes are shrunk so seven sessions make four plans in
    four one-part pages. Anti-vacuity: accept each phase as one batch again and
    the audit tier refuses more parts than a page holds, so no phase returns.
    """
    from polylogue.operations import daemon_mutations

    monkeypatch.setattr(daemon_mutations, "MUTATION_PLAN_PAGE_SIZE", 2)
    monkeypatch.setattr(daemon_mutations, "_MUTATION_SELECTION_PAGE_SIZE", 2)
    monkeypatch.setattr(daemon_mutations, "MACHINE_PAGE_PARTS", 1)
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    _seed_delete_authority_archive(archive_root, 7)

    with _delete_authority_daemon(monkeypatch, archive_root) as client:
        preview = _delete_operation(client, "preview", {"selection": {"params": {"list_mode": True}, "mode": "all"}})
        assert preview["session_count"] == 7
        preview_id = str(cast(dict[str, object], preview["reference"])["request_id"])
        assert cast(dict[str, object], preview["reference"])["part_count"] == 4
        authorization = _delete_operation(client, "authorize", {"preview_request_id": preview_id})
        authorization_id = str(cast(dict[str, object], authorization["reference"])["request_id"])
        assert cast(dict[str, object], authorization["reference"])["part_count"] == 4
        result = _delete_operation(client, "execute", {"authorization_request_id": authorization_id})

    _assert_completed_delete(result, affected=7, chunks=4)
    with sqlite3.connect(archive_root / "audit.db") as conn:
        kinds = sorted(str(row[0]) for row in conn.execute("SELECT artifact_kind FROM machine_requests"))
        assert kinds == ["authorization-batch", "execution-batch", "preview-batch"]
        assert conn.execute(
            "SELECT COUNT(*) FROM operation_previews WHERE expires_at_ms <= created_at_ms"
        ).fetchone() == (0,)
        assert {int(row[0]) for row in conn.execute("SELECT part_count FROM machine_requests")} == {4}
    with sqlite3.connect(archive_root / "index.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone() == (0,)


def test_cli_delete_cancels_a_preview_of_many_pages(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Anti-vacuity: cancel every preview ref in one batch and a preview wider
    than one page cannot be released."""
    from polylogue.operations import daemon_mutations

    monkeypatch.setattr(daemon_mutations, "MUTATION_PLAN_PAGE_SIZE", 2)
    monkeypatch.setattr(daemon_mutations, "MACHINE_PAGE_PARTS", 1)
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    session_ids = _seed_delete_authority_archive(archive_root, 5)

    with _delete_authority_daemon(monkeypatch, archive_root) as client:
        preview = _delete_operation(client, "preview", {"session_ids": list(session_ids)})
        preview_id = str(cast(dict[str, object], preview["reference"])["request_id"])
        assert cast(dict[str, object], preview["reference"])["part_count"] == 3
        cancelled = _delete_operation(client, "cancel", {"preview_request_id": preview_id})
        assert cancelled["source_request_id"] == preview_id
        assert cast(dict[str, object], cancelled["reference"])["part_count"] == 3
    with sqlite3.connect(archive_root / "audit.db") as conn:
        states = {str(row[0]) for row in conn.execute("SELECT state FROM operation_previews")}
    assert states == {"cancelled"}


def test_cli_delete_keeps_progressing_past_its_request_deadline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A durably accepted paged delete finishes every phase past its deadline.

    The runtime reports a deadline after actual durable acceptance. Anti-vacuity:
    fence staging pages or execution parts on ``deadline`` and the selection is
    left partly deleted, with a request stopped as ``deadline``.
    """
    from polylogue.daemon import operation_runtime
    from polylogue.operations import daemon_mutations

    monkeypatch.setattr(daemon_mutations, "MUTATION_PLAN_PAGE_SIZE", 2)
    monkeypatch.setattr(daemon_mutations, "MACHINE_PAGE_PARTS", 1)

    def past_deadline_after_acceptance(runtime: operation_runtime.DaemonOperationRuntime, request: Any) -> str | None:
        exchange = runtime._exchanges[str(request.request_id)]
        return "deadline" if exchange.acceptance_started else None

    monkeypatch.setattr(operation_runtime.DaemonOperationRuntime, "stop_reason", past_deadline_after_acceptance)
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    session_ids = _seed_delete_authority_archive(archive_root, 7)

    with _delete_authority_daemon(monkeypatch, archive_root) as client:
        preview = _delete_operation(client, "preview", {"session_ids": list(session_ids)})
        authorization = _delete_operation(
            client, "authorize", {"preview_request_id": cast(dict[str, object], preview["reference"])["request_id"]}
        )
        result = _delete_operation(
            client,
            "execute",
            {"authorization_request_id": cast(dict[str, object], authorization["reference"])["request_id"]},
        )

    _assert_completed_delete(result, affected=7, chunks=4)
    with sqlite3.connect(archive_root / "audit.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM machine_requests WHERE stop_reason IS NOT NULL").fetchone() == (0,)
    with sqlite3.connect(archive_root / "index.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone() == (0,)


@pytest.mark.uses_real_clock("HTTP failed cleanup remains on its admitted original worker until successor settlement")
def test_http_body_retains_failed_sql_cleanup_and_refuses_successor(tmp_path: Path) -> None:
    from polylogue.daemon.write_coordinator import DaemonWriterSettlementError
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore, ArchiveStoreSettlementError
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
    from tests.infra.archive_custody_probe import archive_custody_available
    from tests.infra.sqlite_cursor_settlement import SettlementConnection, arm_settlement

    initialize_active_archive_root(tmp_path)
    _coordinator, bridge, stop = _loop_owned_bridge(tmp_path)
    handler = _gated_handler(bridge)
    handles: list[SettlementConnection] = []

    async def mutation(_archive: object) -> None:
        store = ArchiveStore(tmp_path, initialize=False)
        handle = arm_settlement(store._conn)
        handles.append(handle)
        store._enter_mutation_lease()
        handle.execute("BEGIN IMMEDIATE")
        try:
            store.close()
        except ArchiveStoreSettlementError:
            pass

    try:
        with pytest.raises(DaemonWriterSettlementError), handler._write_gate("test.http.failed_sql"):
            handler._sync_run(mutation)
        assert handles[0].in_transaction
        assert not archive_custody_available(tmp_path)
        with pytest.raises(DaemonWriterSettlementError):
            bridge.run_sync("test.http.failed_successor", lambda: None)
        assert handles[0].in_transaction
        handles[0].allow_cleanup.set()
        bridge.run_sync("test.http.settled_successor", lambda: None)
        handles[0].owner.join()
        assert not handles[0].owner.is_alive()
        assert archive_custody_available(tmp_path)
    finally:
        for handle in handles:
            handle.allow_cleanup.set()
        if handles and handles[0].owner.is_alive():
            bridge.run_sync("test.http.cleanup", lambda: None)
            handles[0].owner.join()
        handler.server.execution_kernel.shutdown(wait=True)
        stop()


@pytest.mark.uses_real_clock("inline Ops commit does not release original-worker custody after failed native close")
def test_inline_ops_failed_close_has_retryable_answer_and_original_worker_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from polylogue import paths
    from polylogue.daemon.write_coordinator import DaemonWriterSettlementError
    from polylogue.storage.sqlite import connection_profile
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
    from tests.infra.archive_custody_probe import archive_custody_available
    from tests.infra.sqlite_cursor_settlement import SettlementConnection, arm_settlement

    initialize_active_archive_root(tmp_path)
    monkeypatch.setattr(paths, "archive_root", lambda: tmp_path)
    _coordinator, bridge, stop = _loop_owned_bridge(tmp_path)
    handler = _gated_handler(bridge)
    handler._response_started = False
    body = json.dumps(
        {
            "call_id": "neutral-close-call",
            "tool_name": "neutral-tool",
            "session_ids": [],
            "started_at_ms": 1,
            "finished_at_ms": 2,
            "success": True,
        }
    ).encode()
    handler.headers = Message()
    handler.headers["Content-Length"] = str(len(body))
    handler.rfile = BytesIO(body)
    replies: list[tuple[object, dict[str, object]]] = []
    object.__setattr__(handler, "_send_json", lambda status, result, **_kwargs: replies.append((status, result)))
    handles: list[SettlementConnection] = []
    real_open = connection_profile.open_daemon_connection
    opened = 0

    def controlled_open(*args: object, **kwargs: object) -> sqlite3.Connection:
        nonlocal opened
        connection = real_open(*args, **kwargs)  # type: ignore[arg-type]
        opened += 1
        if opened == 2:
            handle = arm_settlement(connection)
            handles.append(handle)
            return handle
        return connection

    monkeypatch.setattr(connection_profile, "open_daemon_connection", controlled_open)
    try:
        with handler._write_gate("http.telemetry.mcp-call"):
            handler._handle_mcp_call_log()
        assert len(replies) == 1
        assert replies[0][0] == HTTPStatus.SERVICE_UNAVAILABLE
        assert replies[0][1]["error"] == "writer_sql_unsettled"
        assert not handles[0].in_transaction  # Actual commit succeeded before close failed.
        assert not archive_custody_available(tmp_path)
        with pytest.raises(DaemonWriterSettlementError):
            bridge.run_sync("test.ops.failed_successor", lambda: None)
        handles[0].allow_cleanup.set()
        bridge.run_sync("test.ops.settled_successor", lambda: None)
        handles[0].owner.join()
        assert not handles[0].owner.is_alive()
        assert archive_custody_available(tmp_path)
        with contextlib.closing(sqlite3.connect(tmp_path / "ops.db")) as reader:
            assert reader.execute("SELECT call_id FROM mcp_call_log").fetchall() == [("neutral-close-call",)]
    finally:
        for handle in handles:
            handle.allow_cleanup.set()
        if handles and handles[0].owner.is_alive():
            bridge.run_sync("test.ops.cleanup", lambda: None)
            handles[0].owner.join()
        handler.server.execution_kernel.shutdown(wait=True)
        stop()
