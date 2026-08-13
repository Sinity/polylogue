"""Mutation-sensitive route proofs for daemon HTTP writer coordination."""

from __future__ import annotations

import contextlib
import hashlib
import json
import socket
import sqlite3
import threading
from collections.abc import Awaitable, Callable, Iterator
from http import HTTPStatus
from io import BytesIO
from os import getpid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from polylogue.daemon.http import (
    _CLI_DELETE_SELECTION_MAX_BYTES,
    _REBUILD_INDEX_WRITE_TIMEOUT_S,
    DaemonAPIHandler,
    DaemonAPIHTTPServer,
)
from polylogue.daemon.web_auth import WebCredentialScope


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
    def allow_auth(required_scope: WebCredentialScope = "read", *, allow_web: bool = True) -> bool:
        del required_scope, allow_web
        return True

    def allow_host(*, credential_request: bool = False) -> bool:
        del credential_request
        return True

    handler = object.__new__(DaemonAPIHandler)
    handler.server = SimpleNamespace(write_bridge=_RecordingBridge(timeline))  # type: ignore[assignment]
    handler.path = "/" + "/".join(path)
    handler._parse_path = lambda: (path, {})  # type: ignore[method-assign]
    handler._check_host_admission = allow_host  # type: ignore[method-assign]
    handler._check_auth = allow_auth  # type: ignore[method-assign]
    handler._check_cross_origin = lambda: True  # type: ignore[method-assign]
    handler._send_error = lambda *_args: timeline.append("error")  # type: ignore[method-assign]
    return handler


@pytest.mark.parametrize(
    ("path", "handler_name", "actor"),
    [
        (["api", "reset"], "_handle_reset", "http.reset"),
        (["api", "ingest"], "_handle_ingest", "http.ingest"),
        (["api", "maintenance", "run"], "_handle_maintenance_run", "http.maintenance.run"),
    ],
)
def test_authenticated_write_route_holds_gate_around_handler(path: list[str], handler_name: str, actor: str) -> None:
    timeline: list[str] = []
    handler = _handler(path, timeline)
    setattr(handler, handler_name, lambda: timeline.append("body"))

    handler._do_post_impl()

    assert timeline == [f"enter:{actor}", "body", f"exit:{actor}"]


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


@contextlib.contextmanager
def _delete_authority_daemon(monkeypatch: pytest.MonkeyPatch, archive_root: Path) -> Iterator[object]:
    from polylogue.daemon.uds import DaemonAPIUnixHTTPServer
    from polylogue.daemon_client import DaemonClient

    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(archive_root))
    socket_path = Path("/tmp") / f"polylogue-delete-authority-{getpid()}-{uuid4().hex}.sock"
    probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        try:
            probe.bind(str(socket_path))
        except PermissionError:
            pytest.skip("sandbox denies AF_UNIX listeners required for the production daemon route")
    finally:
        probe.close()
        socket_path.unlink(missing_ok=True)
    server = DaemonAPIUnixHTTPServer(socket_path, DaemonAPIHandler)
    server.auth_token = "delete-authority-token"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield DaemonClient(socket_path, timeout_s=2.0, auth_token="delete-authority-token")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _prepare_authorize(client: object, session_ids: tuple[str, ...]) -> str:
    preview = client.request_mutation_json("POST", "/api/cli/delete/prepare", {"session_ids": list(session_ids)})  # type: ignore[attr-defined]
    assert preview is not None
    assert preview["session_ids"] == list(session_ids)
    authorization = client.request_mutation_json(  # type: ignore[attr-defined]
        "POST", "/api/cli/delete/authorize", {"preview_ref": preview["preview_ref"]}
    )
    assert authorization is not None
    return str(authorization["authorization_token"])


def _assert_session_exists(archive_root: Path, session_id: str, *, expected: bool) -> None:
    with sqlite3.connect(archive_root / "index.db") as conn:
        count = conn.execute("SELECT COUNT(*) FROM sessions WHERE session_id = ?", (session_id,)).fetchone()[0]
    assert bool(count) is expected


def test_cli_delete_uses_real_uds_client_api_authority_and_audit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Real daemon HTTP/client/API proof of prepared, single-use delete authority."""

    from polylogue.daemon_client import DaemonResponseError
    from polylogue.operations.delete_authorization import DeleteAuthorizationError, consume_cli_delete
    from polylogue.operations.mutation_transaction import MutationPrincipal
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    success_id, replay_id, substitute_id, stale_a, stale_b, expiry_id = _seed_delete_authority_archive(archive_root, 6)
    with _delete_authority_daemon(monkeypatch, archive_root) as client:
        success_token = _prepare_authorize(client, (success_id,))
        result = client.request_mutation_json("POST", "/api/cli/delete", {"authorization_token": success_token})  # type: ignore[attr-defined]
        assert result == {"status": "deleted", "operation": "delete", "session_count": 1, "affected_count": 1}
        _assert_session_exists(archive_root, success_id, expected=False)

        with pytest.raises(DaemonResponseError):
            client.request_mutation_json("POST", "/api/cli/delete", {"session_ids": [replay_id]})  # type: ignore[attr-defined]
        _assert_session_exists(archive_root, replay_id, expected=True)

        replay_token = _prepare_authorize(client, (replay_id,))
        client.request_mutation_json("POST", "/api/cli/delete", {"authorization_token": replay_token})  # type: ignore[attr-defined]
        with pytest.raises(DaemonResponseError):
            client.request_mutation_json("POST", "/api/cli/delete", {"authorization_token": replay_token})  # type: ignore[attr-defined]
        _assert_session_exists(archive_root, substitute_id, expected=True)

        substitute_token = _prepare_authorize(client, (substitute_id,))
        with pytest.raises(DaemonResponseError):
            client.request_mutation_json(  # type: ignore[attr-defined]
                "POST",
                "/api/cli/delete",
                {"authorization_token": substitute_token, "session_ids": [stale_a]},
            )
        _assert_session_exists(archive_root, substitute_id, expected=True)
        client.request_mutation_json("POST", "/api/cli/delete", {"authorization_token": substitute_token})  # type: ignore[attr-defined]

        stale_token = _prepare_authorize(client, (stale_a, stale_b))
        with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
            archive.delete_sessions((stale_a,))
        with pytest.raises(DaemonResponseError) as stale_error:
            client.request_mutation_json("POST", "/api/cli/delete", {"authorization_token": stale_token})  # type: ignore[attr-defined]
        assert stale_error.value.status == HTTPStatus.CONFLICT
        assert stale_error.value.code == "delete_authorization_denied"
        assert stale_error.value.detail == "selection_changed_after_authorization"
        _assert_session_exists(archive_root, stale_b, expected=True)

        expiry_token = _prepare_authorize(client, (expiry_id,))
        with pytest.raises(DeleteAuthorizationError):
            consume_cli_delete(
                archive_root,
                expiry_token,
                MutationPrincipal("daemon:bearer:other", frozenset({"archive.delete_session"}), "cli", "write"),
            )
        _assert_session_exists(archive_root, expiry_id, expected=True)
        with sqlite3.connect(archive_root / "audit.db") as conn:
            conn.execute(
                "UPDATE operation_authorizations SET issued_at_ms = 0, expires_at_ms = 1 WHERE token_sha256 = ?",
                (hashlib.sha256(expiry_token.encode()).hexdigest(),),
            )
        with pytest.raises(DaemonResponseError):
            client.request_mutation_json("POST", "/api/cli/delete", {"authorization_token": expiry_token})  # type: ignore[attr-defined]
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


def test_cli_delete_real_daemon_route_cancels_an_unconfirmed_preview(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A declined CLI confirmation has the daemon retire its durable preview."""

    from polylogue.daemon_client import DaemonResponseError

    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    (session_id,) = _seed_delete_authority_archive(archive_root, 1)

    with _delete_authority_daemon(monkeypatch, archive_root) as client:
        preview = client.request_mutation_json(  # type: ignore[attr-defined]
            "POST", "/api/cli/delete/prepare", {"session_ids": [session_id]}
        )
        assert preview is not None
        preview_ref = str(preview["preview_ref"])
        cancelled = client.request_mutation_json(  # type: ignore[attr-defined]
            "POST", "/api/cli/delete/cancel", {"preview_ref": preview_ref}
        )
        assert cancelled == {"status": "cancelled", "preview_ref": preview_ref}
        with pytest.raises(DaemonResponseError) as authorization_error:
            client.request_mutation_json(  # type: ignore[attr-defined]
                "POST", "/api/cli/delete/authorize", {"preview_ref": preview_ref}
            )

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
        preview = client.request_mutation_json(  # type: ignore[attr-defined]
            "POST", "/api/cli/delete/prepare", {"session_ids": [session_id]}
        )
        assert preview is not None
        preview_ref = str(preview["preview_ref"])
        with sqlite3.connect(archive_root / "audit.db") as conn:
            conn.execute(
                "UPDATE operation_previews SET created_at_ms = 0, expires_at_ms = 1 WHERE preview_id = ?",
                (preview_ref,),
            )

        cancelled = client.request_mutation_json(  # type: ignore[attr-defined]
            "POST", "/api/cli/delete/cancel", {"preview_ref": preview_ref}
        )

    assert cancelled == {"status": "cancelled", "preview_ref": preview_ref}
    _assert_session_exists(archive_root, session_id, expected=True)
    with sqlite3.connect(archive_root / "audit.db") as conn:
        assert conn.execute("SELECT state FROM operation_previews WHERE preview_id = ?", (preview_ref,)).fetchone() == (
            "cancelled",
        )


def test_cli_delete_bounds_body_bytes_accepts_large_selection_and_reads_before_writer_gate() -> None:
    class _ExplodingBody:
        def read(self, _size: int) -> bytes:
            raise AssertionError("oversize body must not be read")

    oversize_timeline: list[str] = []
    oversize = _handler(["api", "cli", "delete", "prepare"], oversize_timeline)
    oversize.headers = {"Content-Length": str(_CLI_DELETE_SELECTION_MAX_BYTES + 1)}  # type: ignore[assignment]
    oversize.rfile = _ExplodingBody()  # type: ignore[assignment]
    oversize._do_post_impl()
    assert oversize_timeline == ["error"]

    large_timeline: list[str] = []
    large = _handler(["api", "cli", "delete", "prepare"], large_timeline)
    large_body = json.dumps({"session_ids": [f"codex-session:{index}" for index in range(257)]}).encode()
    large.headers = {"Content-Length": str(len(large_body))}  # type: ignore[assignment]
    large.rfile = BytesIO(large_body)
    large._sync_run = lambda _operation: {"status": "prepared"}  # type: ignore[assignment]
    large._send_json = lambda *_args: large_timeline.append("response")  # type: ignore[method-assign]
    large._do_post_impl()
    assert large_timeline == ["enter:http.cli.delete.prepare", "exit:http.cli.delete.prepare", "response"]

    class _SlowBody:
        def read(self, _size: int) -> bytes:
            slow_timeline.append("body-read")
            assert not any(item.startswith("enter:") for item in slow_timeline)
            return json.dumps({"session_ids": ["codex-session:slow"]}).encode()

    slow_timeline: list[str] = []
    slow = _handler(["api", "cli", "delete", "prepare"], slow_timeline)
    body = json.dumps({"session_ids": ["codex-session:slow"]}).encode()
    slow.headers = {"Content-Length": str(len(body))}  # type: ignore[assignment]
    slow.rfile = _SlowBody()  # type: ignore[assignment]
    slow._sync_run = lambda _operation: {"status": "prepared"}  # type: ignore[assignment]
    slow._send_json = lambda *_args: slow_timeline.append("response")  # type: ignore[method-assign]
    slow._do_post_impl()
    assert slow_timeline == ["body-read", "enter:http.cli.delete.prepare", "exit:http.cli.delete.prepare", "response"]


def test_cli_delete_real_daemon_route_deletes_a_selection_larger_than_legacy_cap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    session_ids = _seed_delete_authority_archive(archive_root, 257)

    with _delete_authority_daemon(monkeypatch, archive_root) as client:
        token = _prepare_authorize(client, session_ids)
        result = client.request_mutation_json(  # type: ignore[attr-defined]
            "POST", "/api/cli/delete", {"authorization_token": token}
        )

    assert result == {"status": "deleted", "operation": "delete", "session_count": 257, "affected_count": 257}
    with sqlite3.connect(archive_root / "index.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone() == (0,)


def test_cli_delete_real_daemon_route_refuses_selection_beyond_preview_work_budget(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The durable preview route must bound target work independently of request bytes.

    This sends 10,001 ordinary IDs through the real UDS client and daemon
    handler. Before the repair, the route entered the sole-writer gate and
    attempted resolution until it encountered a stale ID, because no target
    work budget existed. The repaired route rejects the request before any
    archive lookup or durable preview write.
    """
    from polylogue.daemon_client import DaemonResponseError

    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    _seed_delete_authority_archive(archive_root, 0)
    selection = [f"codex-session:over-budget-{index}" for index in range(10_001)]

    with _delete_authority_daemon(monkeypatch, archive_root) as client:
        with pytest.raises(DaemonResponseError) as error:
            client.request_mutation_json("POST", "/api/cli/delete/prepare", {"session_ids": selection})  # type: ignore[attr-defined]

    assert error.value.status == HTTPStatus.REQUEST_ENTITY_TOO_LARGE
    assert error.value.code == "selection_exceeds_preview_work_budget"
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
    handler.headers = {"Authorization": "Bearer attacker-selected"}  # type: ignore[assignment]

    principal = handler._cli_delete_principal()

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


class _RecordingRebuildBridge(_RecordingBridge):
    """Adds ``run_sync_with_timeout`` so real ``_handle_rebuild_index`` can run.

    polylogue-ogn1: rebuild-index uses ``run_sync_with_timeout`` (not
    ``run_sync``) so a long rebuild pass isn't killed by the bridge's much
    shorter default request timeout -- see ``DaemonAPIHandler._handle_rebuild_index``
    and ``DaemonWriteThreadBridge.run_sync_with_timeout``.
    """

    def run_sync_with_timeout(
        self, actor: str, timeout: float | None, function: Callable[..., object], *args: object
    ) -> object:
        self.timeline.append(f"run_sync_with_timeout:{actor}:{timeout}")
        return function(*args)


def test_rebuild_index_route_uses_the_bridge_run_sync_with_timeout_writer_path(monkeypatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Drive the request through the real production dispatch, not a stand-in.

    The previous version of this test replaced ``_handle_rebuild_index``
    wholesale with a body that itself called ``bridge.run_sync`` -- it only
    proved the test's own stand-in called ``run_sync``, never that the real
    production handler does anything of the kind (polylogue-ogn1 finding
    #10). This exercises the real ``_do_post_impl`` route dispatch and the
    real ``_handle_rebuild_index`` implementation end to end, with only the
    typed rebuild service itself stubbed out.
    """
    import json
    from io import BytesIO

    from polylogue.maintenance.rebuild_index import RebuildIndexReceipt

    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(tmp_path))
    timeline: list[str] = []
    handler = _handler(["api", "maintenance", "rebuild-index"], timeline)
    handler.server.write_bridge = _RecordingRebuildBridge(timeline)  # type: ignore[assignment]
    body = json.dumps({"promote": False, "raw_ids": ["raw-1"]}).encode("utf-8")
    handler.headers = {"Content-Length": str(len(body))}  # type: ignore[assignment]
    handler.rfile = BytesIO(body)

    receipt = RebuildIndexReceipt(
        archive_root=str(tmp_path),
        raw_session_count=1,
        selected_raw_count=1,
        skipped_by_blob_limit_count=0,
        status="replayed",
        materialized=True,
        materialization={},
        generation={"generation_id": "candidate-1", "active": False},
        readiness={"checked": True, "blocked_surface_count": 0},
        replay={"classified_full_count": 1, "replayed_logical_source_count": 1, "quarantined_raw_count": 0},
    )
    with patch("polylogue.maintenance.rebuild_index.rebuild_index_from_source_sync", return_value=receipt) as rebuild:
        with patch.object(handler, "_send_json") as send_json:
            handler._do_post_impl()

    assert timeline == [f"run_sync_with_timeout:http.maintenance.rebuild-index:{_REBUILD_INDEX_WRITE_TIMEOUT_S}"]
    request = rebuild.call_args.args[0]
    assert request.raw_ids == ("raw-1",)
    assert request.promote is False
    assert send_json.call_args.args == (HTTPStatus.OK, receipt.to_dict())


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


def test_coordinated_mutation_does_not_use_timeout_detaching_read_executor() -> None:
    handler = object.__new__(DaemonAPIHandler)
    handler._write_gate_depth = 1

    async def run_direct(operation: Callable[[object], Awaitable[object]]) -> object:
        return await operation(None)

    async def mutation(_polylogue: object) -> str:
        return "persisted"

    handler._run_archive_query = run_direct  # type: ignore[assignment]
    handler.server = SimpleNamespace(  # type: ignore[assignment]
        archive_query_admission=SimpleNamespace(acquire=lambda **_kwargs: (_ for _ in ()).throw(AssertionError())),
        archive_query_executor=SimpleNamespace(submit=lambda *_args: (_ for _ in ()).throw(AssertionError())),
    )

    assert handler._sync_run(mutation) == "persisted"
