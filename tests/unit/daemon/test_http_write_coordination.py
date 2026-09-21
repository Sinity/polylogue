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
from polylogue.daemon.web_auth import WebCredentialScope
from polylogue.daemon_client import DaemonClient, DaemonMutationIndeterminateError
from tests.infra.daemon_operations import running_daemon_operations


class _DeleteDaemonClient(DaemonClient):
    archive_root: Path


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
    assert preview["session_ids"] == list(session_ids)
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
    from polylogue.operations.delete_authorization import DeleteAuthorizationError, consume_cli_delete
    from polylogue.operations.mutation_transaction import MutationPrincipal
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

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
        with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
            archive.delete_sessions((stale_a,))
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
        with pytest.raises(DeleteAuthorizationError):
            consume_cli_delete(
                archive_root,
                expiry_ref,
                MutationPrincipal("daemon:bearer:other", frozenset({"archive.delete_session"}), "cli", "write"),
            )
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
        assert cancelled == {"status": "cancelled", "preview_ref": preview_ref, "preview_refs": [preview_ref]}
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

    assert cancelled == {"status": "cancelled", "preview_ref": preview_ref, "preview_refs": [preview_ref]}
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


def test_delete_preview_operation_bounds_body_bytes_and_reads_before_runtime_dispatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The selection bound is the operation's own, and dispatch starts after the read.

    Anti-vacuity: raising ``mutation.session.delete.preview``'s
    ``max_body_bytes`` above the transport's declared maximum makes the
    oversize case read a body it must refuse; dispatching before the body read
    reorders ``slow_timeline``.
    """
    from polylogue.operations.daemon_protocol import (
        MAX_DECLARED_OPERATION_BODY_BYTES,
        daemon_operation_spec,
    )

    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(tmp_path))
    spec = daemon_operation_spec("mutation.session.delete.preview")
    assert spec is not None
    assert spec.max_body_bytes == MAX_DECLARED_OPERATION_BODY_BYTES

    class _ExplodingBody:
        def read(self, _size: int) -> bytes:
            raise AssertionError("oversize body must not be read")

    oversize_timeline: list[str] = []
    oversize = _operation_handler(oversize_timeline, b"", content_length=MAX_DECLARED_OPERATION_BODY_BYTES + 1)
    object.__setattr__(oversize, "rfile", _ExplodingBody())
    oversize._do_post_impl()
    assert oversize_timeline == ["error"]

    large_timeline: list[str] = []
    large_body = _preview_operation_body([f"codex-session:{index}" for index in range(257)])
    large = _operation_handler(large_timeline, large_body)
    object.__setattr__(large, "_send_json", lambda *_args, **_kwargs: large_timeline.append("response"))
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
    object.__setattr__(slow, "_send_json", lambda *_args, **_kwargs: slow_timeline.append("response"))
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
    from polylogue.daemon.execution import CancellationHandle
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
    responses: list[tuple[HTTPStatus, object]] = []
    object.__setattr__(handler, "_send_json", lambda status, payload, **_kwargs: responses.append((status, payload)))

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
    assert responses and responses[0][0] is HTTPStatus.CONFLICT
    assert runtime.calls == [conflicting]
    response = responses[0][1]
    assert isinstance(response, dict)
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
    chunk-count assertions fail. The client budget comes from
    ``_prepared_work_budget_s`` so the route's behavior, not the host's
    current load, decides the outcome (polylogue-ga8vn).
    """

    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    session_ids = _seed_delete_authority_archive(archive_root, 513)

    with _delete_authority_daemon(monkeypatch, archive_root) as client:
        preview = _delete_operation(client, "preview", {"session_ids": list(session_ids)})
        assert preview is not None
        preview_refs = preview["preview_refs"]
        assert isinstance(preview_refs, list)
        assert len(preview_refs) == 3
        authorization = _delete_operation(client, "authorize", {"preview_refs": preview_refs})
        assert authorization is not None
        tokens = authorization["authorization_refs"]
        assert isinstance(tokens, list)
        assert len(tokens) == 3
        result = _delete_operation(client, "execute", {"authorization_refs": tokens})

    _assert_completed_delete(result, affected=513, chunks=3)
    with sqlite3.connect(archive_root / "index.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone() == (0,)


def test_cli_delete_real_daemon_route_reports_partial_chunk_application(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
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
        preview_refs = preview["preview_refs"]
        assert isinstance(preview_refs, list)
        authorization = _delete_operation(client, "authorize", {"preview_refs": preview_refs})
        assert authorization is not None
        tokens = authorization["authorization_refs"]
        assert isinstance(tokens, list)

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
            result = _delete_operation(client, "execute", {"authorization_refs": tokens})

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


def test_cli_delete_real_daemon_route_refuses_selection_beyond_preview_work_budget(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The durable preview route must bound target work independently of request bytes.

    The typed UDS client rejects 10,001 IDs before opening a request because
    the protocol contract caps this list at 10,000. This must happen before
    any archive lookup or durable preview write.
    """
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    _seed_delete_authority_archive(archive_root, 0)
    selection = [f"codex-session:over-budget-{index}" for index in range(10_001)]

    with _delete_authority_daemon(monkeypatch, archive_root) as client:
        with patch.object(
            client,
            "_request_json_response",
            side_effect=AssertionError("client-side payload validation must precede the daemon request"),
        ):
            with pytest.raises(ValueError, match="invalid DeletePreviewRequest payload"):
                _delete_operation(client, "preview", {"session_ids": selection})

    with sqlite3.connect(archive_root / "audit.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM operation_previews").fetchone() == (0,)


def test_cli_delete_preparation_resolves_canonical_ids_in_bounded_pages(tmp_path: Path) -> None:
    """A real archive selection must not spend one SQLite query per canonical ID.

    The production preparation helper is given 513 persisted sessions and its
    real SQLite connection records resolution queries. The repair batches
    exact canonical IDs in fixed-size pages, so this requires three or fewer
    selection queries. The prior per-ID resolver produced 513 queries, and
    the former list membership duplicate check made the canonicality pass
    quadratic as the preview grew.
    """
    from polylogue.operations.delete_authorization import _canonical_session_ids
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    session_ids = _seed_delete_authority_archive(archive_root, 513)

    with ArchiveStore.open_existing(archive_root, read_only=True) as archive:
        statements: list[str] = []
        archive._conn.set_trace_callback(statements.append)
        assert _canonical_session_ids(archive, session_ids) == session_ids

    session_selects = [statement for statement in statements if "FROM sessions" in statement]
    assert len(session_selects) <= 3


def test_cli_delete_preparation_refuses_a_missing_exact_id_that_is_a_live_prefix(tmp_path: Path) -> None:
    """Delete previews are bound to exact canonical IDs, never prefix resolution."""

    from polylogue.operations.delete_authorization import DeleteAuthorizationError, _canonical_session_ids
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    (session_id,) = _seed_delete_authority_archive(archive_root, 1)
    missing_exact_id = session_id.removesuffix("0")

    with ArchiveStore.open_existing(archive_root, read_only=True) as archive:
        with pytest.raises(DeleteAuthorizationError, match="selection_is_stale"):
            _canonical_session_ids(archive, (missing_exact_id,))

    _assert_session_exists(archive_root, session_id, expected=True)


def test_cli_delete_preparation_rejects_a_late_duplicate_before_archive_resolution(tmp_path: Path) -> None:
    """Set canonicality rejects a large duplicate selection without quadratic work.

    This invokes the production preparation helper with a real temporary
    SQLite archive. A duplicate after 513 distinct IDs must fail before any
    resolver query. The pre-repair list membership loop resolved every prior
    target and compared each canonical ID against a growing list.
    """
    from polylogue.operations.delete_authorization import DeleteAuthorizationError, _canonical_session_ids
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    session_ids = _seed_delete_authority_archive(archive_root, 513)

    with ArchiveStore.open_existing(archive_root, read_only=True) as archive:
        statements: list[str] = []
        archive._conn.set_trace_callback(statements.append)
        with pytest.raises(DeleteAuthorizationError, match="selection_is_not_canonical"):
            _canonical_session_ids(archive, session_ids + (session_ids[0],))

    assert not [statement for statement in statements if "FROM sessions" in statement]


def test_cli_delete_interruption_consumes_authorization_without_deleting(tmp_path: Path) -> None:
    """An interrupted apply leaves a consumed unknown audit attempt, never a retryable token."""

    from polylogue.operations.delete_authorization import (
        DeleteAuthorizationError,
        authorize_cli_delete,
        consume_cli_delete,
        prepare_cli_delete,
    )
    from polylogue.operations.mutation_actuators import SessionDeleteActuator
    from polylogue.operations.mutation_transaction import MutationPrincipal

    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    (session_id,) = _seed_delete_authority_archive(archive_root, 1)
    principal = MutationPrincipal(
        "daemon:bearer:interrupted",
        frozenset({"archive.delete_session"}),
        "cli",
        "daemon-authenticated",
    )
    preview = prepare_cli_delete(archive_root, (session_id,), principal)
    token = authorize_cli_delete(archive_root, preview.preview_ref, principal)

    with patch.object(SessionDeleteActuator, "apply", side_effect=RuntimeError("interrupted before apply")):
        with pytest.raises(RuntimeError, match="interrupted before apply"):
            consume_cli_delete(archive_root, token, principal)
    _assert_session_exists(archive_root, session_id, expected=True)

    with pytest.raises(DeleteAuthorizationError, match="authorization_not_active"):
        consume_cli_delete(archive_root, token, principal)
    with sqlite3.connect(archive_root / "audit.db") as conn:
        state = conn.execute(
            "SELECT state, unknown_reason FROM operation_attempts ORDER BY started_at_ms DESC LIMIT 1"
        ).fetchone()
    assert state == ("unknown", "actuator exception after durable intent")


def test_cli_delete_preserves_audit_finalization_failure_after_effect(tmp_path: Path) -> None:
    from polylogue.operations.audit import AuditRepository
    from polylogue.operations.delete_authorization import authorize_cli_delete, consume_cli_delete, prepare_cli_delete
    from polylogue.operations.mutation_transaction import AuditFinalizationError, MutationPrincipal

    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    (session_id,) = _seed_delete_authority_archive(archive_root, 1)
    principal = MutationPrincipal(
        "daemon:bearer:audit-failure",
        frozenset({"archive.delete_session"}),
        "cli",
        "daemon-authenticated",
    )
    preview = prepare_cli_delete(archive_root, (session_id,), principal)
    token = authorize_cli_delete(archive_root, preview.preview_ref, principal)

    with patch.object(AuditRepository, "finalize_attempt", side_effect=RuntimeError("audit unavailable")):
        with pytest.raises(AuditFinalizationError):
            consume_cli_delete(archive_root, token, principal)

    _assert_session_exists(archive_root, session_id, expected=False)


def test_no_auth_cli_principal_ignores_attacker_selected_bearer_text() -> None:
    handler = _handler(["api", "cli", "delete", "prepare"], [])
    object.__setattr__(handler, "headers", {"Authorization": "Bearer attacker-selected"})

    principal = handler._cli_mutation_principal("archive.delete_session")

    assert principal.actor_ref == "daemon:unauthenticated-loopback"
    assert principal.role_label == "daemon-loopback-no-auth"


def test_user_post_and_delete_hold_named_gates_around_dispatch() -> None:
    post_timeline: list[str] = []
    post_handler = _handler(["api", "user", "marks"], post_timeline)

    def dispatch_post(*_args: object) -> bool:
        post_timeline.append("body")
        return True

    with patch("polylogue.daemon.http.user_state_http.dispatch_post", side_effect=dispatch_post):
        post_handler._do_post_impl()
    assert post_timeline == ["enter:http.user.marks.post", "body", "exit:http.user.marks.post"]

    delete_timeline: list[str] = []
    delete_handler = _handler(["api", "user", "annotations", "ann-1"], delete_timeline)

    def dispatch_delete(*_args: object) -> bool:
        delete_timeline.append("body")
        return True

    with patch("polylogue.daemon.http.user_state_http.dispatch_delete", side_effect=dispatch_delete):
        delete_handler._do_delete_impl()
    assert delete_timeline == [
        "enter:http.user.annotations.delete",
        "body",
        "exit:http.user.annotations.delete",
    ]


def test_standalone_http_server_owns_and_idempotently_closes_writer_runtime() -> None:
    server = DaemonAPIHTTPServer(("127.0.0.1", 0), DaemonAPIHandler)
    runtime = server._owned_write_runtime
    assert runtime is not None
    assert runtime.thread.is_alive()

    server.server_close()
    server.server_close()

    assert not runtime.thread.is_alive()


def test_standalone_http_server_stops_loop_after_late_writer_drain() -> None:
    server = DaemonAPIHTTPServer(("127.0.0.1", 0), DaemonAPIHandler)
    runtime = server._owned_write_runtime
    assert runtime is not None
    assert runtime.coordinator is not None
    shutdown = AsyncMock(side_effect=[False, True])

    with patch.object(runtime.coordinator, "shutdown", shutdown):
        server.server_close()
        runtime.thread.join(timeout=1.0)

    assert not runtime.thread.is_alive()
    assert shutdown.await_count == 2


def test_coordinated_mutation_uses_control_admission_without_the_read_timeout() -> None:
    """A mutation is scheduled as control and waits for its own substrate call.

    Anti-vacuity: routing it as ``interactive-read`` would record the work
    under the read class, and the read contract's timeout would be free to
    detach a request whose writer lease is still held.
    """

    from polylogue.daemon.execution import BoundedComputeAdapter

    kernel = BoundedComputeAdapter(max_workers=2, queue_units=2)
    handler = object.__new__(DaemonAPIHandler)
    handler._write_gate_depth = 1

    async def run_direct(operation: Callable[[object], Awaitable[object]]) -> object:
        return await operation(None)

    async def mutation(_polylogue: object) -> str:
        return "persisted"

    object.__setattr__(handler, "_run_archive_query", run_direct)
    object.__setattr__(handler, "server", SimpleNamespace(execution_kernel=kernel))

    try:
        assert handler._sync_run(mutation) == "persisted"
        snapshot = kernel.snapshot()
        assert snapshot.by_class("control").admitted == 1
        assert snapshot.by_class("interactive-read").admitted == 0
        assert snapshot.used_units == 0
    finally:
        kernel.shutdown(wait=True)


def _loop_owned_bridge() -> tuple[object, object, Callable[[], None]]:
    from polylogue.daemon.write_coordinator import DaemonWriteCoordinator, DaemonWriteThreadBridge

    loop = asyncio.new_event_loop()
    ready = threading.Event()
    holder: list[DaemonWriteCoordinator] = []

    def run_loop() -> None:
        asyncio.set_event_loop(loop)
        holder.append(DaemonWriteCoordinator())
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
    from polylogue.daemon.execution import BoundedComputeAdapter

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


def test_gated_route_body_is_authorized_to_write_under_process_wide_enforcement() -> None:
    """The gate admits the durable user.db write it exists to admit.

    polylogue-h5l6i: ``hold()`` entered the lease in a coroutine on the owner
    loop while the route body ran on a kernel worker in a freshly created event
    loop, so every gated POST/DELETE under ``/api/user/*`` raised
    ``UnleasedWriteError`` and answered HTTP 500.

    Anti-vacuity: drop the ``adopt_write_lease`` wrapper from
    ``_archive_query_coroutine`` and this returns ``"unleased"``.
    """
    from polylogue.storage.sqlite.write_lease import (
        arm_write_lease_enforcement,
        require_write_lease,
    )

    coordinator, bridge, stop = _loop_owned_bridge()
    del coordinator
    handler = _gated_handler(bridge)

    async def mutation(_polylogue: object) -> str:
        try:
            require_write_lease("write user.db annotation")
        except Exception as exc:  # the refusal is the observation under test
            return f"unleased:{type(exc).__name__}"
        return "leased"

    try:
        with arm_write_lease_enforcement(process_wide=True):
            with handler._write_gate("http.user.annotations.post"):
                assert handler._sync_run(mutation) == "leased"
    finally:
        handler.server.execution_kernel.shutdown(wait=True)
        stop()


def test_a_thread_outside_the_admitted_body_still_cannot_write() -> None:
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

    coordinator, bridge, stop = _loop_owned_bridge()
    del coordinator
    handler = _gated_handler(bridge)
    rogue_result: list[str] = []
    rogue_done = threading.Event()

    async def mutation(_polylogue: object) -> str:
        def rogue() -> None:
            try:
                require_write_lease("write user.db from an unadmitted thread")
            except UnleasedWriteError:
                rogue_result.append("refused")
            else:
                rogue_result.append("leased")
            rogue_done.set()

        thread = threading.Thread(target=rogue)
        thread.start()
        thread.join(timeout=5.0)
        require_write_lease("write user.db annotation")
        return "leased"

    try:
        with arm_write_lease_enforcement(process_wide=True):
            with handler._write_gate("http.user.annotations.post"):
                assert handler._sync_run(mutation) == "leased"
    finally:
        handler.server.execution_kernel.shutdown(wait=True)
        stop()

    assert rogue_done.is_set()
    assert rogue_result == ["refused"]


def test_inline_gated_write_presents_the_grant_on_the_request_thread() -> None:
    """A route that writes inline is its own execution unit and must present it.

    ``_handle_mcp_call_log`` opens its ops.db connection on the request thread
    rather than through ``_sync_run``; it was refused by the same defect.

    Anti-vacuity: remove the ``_write_authorization()`` block around that
    write and this probe raises ``UnleasedWriteError``.
    """
    from polylogue.storage.sqlite.write_lease import (
        arm_write_lease_enforcement,
        require_write_lease,
    )

    coordinator, bridge, stop = _loop_owned_bridge()
    del coordinator
    handler = _gated_handler(bridge)

    try:
        with arm_write_lease_enforcement(process_wide=True):
            with handler._write_gate("http.telemetry.mcp-call"):
                with handler._write_authorization():
                    assert require_write_lease("write ops.db") is not None
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
    """The stored plan payload, not the preview columns, defines the plan.

    ``delete_authorization`` used to rebuild the plan from ``operation_previews``
    columns with its own rules (notably a hardcoded ``reversible=False``), so
    two readers of one durable row could disagree about what was authorized.
    Reconstruction now goes through the audit tier's own payload reader, and
    the loaded plan is bound by the same integrity check the authorize/begin
    path applies.

    Anti-vacuity: reinstate the column-based reconstruction and the first
    assertion is red (the tampered ``reversible`` in ``plan_json`` would be
    ignored); drop ``validate_mutation_plan_integrity`` from the load path and
    the second assertion is red (a rewritten context would load happily).
    """

    from polylogue.operations.delete_authorization import (
        DeleteAuthorizationError,
        _audit_repository,
        _load_preview,
        prepare_cli_delete,
    )
    from polylogue.operations.mutation_transaction import MutationPrincipal

    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    (session_id,) = _seed_delete_authority_archive(archive_root, 1)
    principal = MutationPrincipal(
        "daemon:bearer:reconstruction",
        frozenset({"archive.delete_session"}),
        "cli",
        "daemon-authenticated",
    )
    preview = prepare_cli_delete(archive_root, (session_id,), principal)
    audit = _audit_repository(archive_root)

    loaded = _load_preview(audit, preview.preview_ref, principal, require_prepared=True)
    assert loaded.plan.target_refs == (f"session:{session_id}",)
    assert loaded.plan.reversible is False

    def _rewrite_plan_json(mutate: Callable[[dict[str, object]], None]) -> None:
        with sqlite3.connect(archive_root / "audit.db") as conn:
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

    def _set_reversible(document: dict[str, object]) -> None:
        document["reversible"] = True

    _rewrite_plan_json(_set_reversible)
    assert _load_preview(audit, preview.preview_ref, principal, require_prepared=True).plan.reversible is True

    def _rewrite_context(document: dict[str, object]) -> None:
        document["reversible"] = False
        document["context"] = {"session_ids": ["codex-session:not-authorized"]}

    _rewrite_plan_json(_rewrite_context)
    with pytest.raises(DeleteAuthorizationError, match="preview_plan_invalid"):
        _load_preview(audit, preview.preview_ref, principal, require_prepared=True)


def test_an_indeterminate_mutation_keeps_the_writer_gate_until_its_body_settles() -> None:
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

    coordinator, bridge, stop = _loop_owned_bridge()
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
            with handler._write_gate("http.user.annotations.post"):
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
