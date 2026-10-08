"""Production UDS operation route contracts."""

from __future__ import annotations

import json
import os
import queue
import socket
import sqlite3
import threading
from contextlib import closing
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest

from polylogue.daemon.uds import MachineOperationHandler
from polylogue.daemon_client import DaemonClient, DaemonOperationRejectedError
from polylogue.operations.mutation_actuators import SessionDeleteActuator, SessionDeleteArgs
from polylogue.operations.mutation_transaction import MUTATION_PLAN_PAGE_SIZE, MutationPlan, MutationReceipt
from tests.infra.daemon_operations import running_daemon_operations
from tests.infra.storage_records import SessionBuilder

pytestmark = pytest.mark.uses_real_clock(
    "starts the real UDS listener and coordinator loop; wall-clock events bound socket and writer ownership waits"
)


def test_failed_embedding_receipt_decodes_without_fabricated_counters() -> None:
    """Anti-vacuity: requiring integer counters turns accepted failures into indeterminate replay."""
    from polylogue.operations.daemon_protocol import EmbeddingBackfillResult
    from polylogue.operations.machine_lifecycle import _embedding_terminal_receipt

    receipt = {
        "operation": "maintenance.embeddings.backfill",
        "outcome": "failed",
        "sequence": 1,
        "effect": "indeterminate",
        "affected_count": None,
        "progress": {"state": "unknown", "computed": None, "failed": None, "estimated_cost_usd": None},
        "result": {"done": None, "pending": None, "failed": None},
        "error": {"code": "embedding_backfill_failed", "message": "provider unavailable"},
    }
    EmbeddingBackfillResult.model_validate(receipt)
    assert _embedding_terminal_receipt("embedding_receipt:" + json.dumps(receipt)) == receipt


def test_failed_backup_operation_retains_rejected_result_details(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polylogue.operations.archive_backup import BackupResult

    partial = BackupResult(
        ok=False,
        output_path=str(tmp_path / "partial-backup"),
        error="verification failed",
        warnings=["source.db could not be verified"],
    )
    monkeypatch.setattr("polylogue.operations.archive_backup.backup_archive", lambda **_kwargs: partial)

    with running_daemon_operations(tmp_path / "archive") as stack:
        envelope = stack.client.operation(
            "maintenance.backup",
            {"output_dir": str(tmp_path / "backups")},
            archive_root=str(stack.archive_root),
        )

    assert envelope is not None
    assert envelope["outcome"] == "rejected"
    assert envelope["result"] is None
    assert envelope["error"] == {
        "code": "backup_failed",
        "detail": "verification failed",
        "retryable": False,
        "data": {"backup_result": partial.model_dump(mode="json")},
    }


@pytest.mark.parametrize("index_state", ["missing", "skewed"])
def test_overlay_backup_does_not_require_a_readable_index(tmp_path: Path, index_state: str) -> None:
    """Overlay evidence stays available while the derived tier needs recovery."""
    from polylogue.storage.archive_identity import ArchiveLocation

    with running_daemon_operations(tmp_path / "archive") as stack:
        index = ArchiveLocation.resolve(stack.archive_root).active_index_path
        if index_state == "missing":
            index.unlink()
        else:
            with closing(sqlite3.connect(index)) as connection:
                connection.execute("PRAGMA user_version=999")
        refused = stack.client.operation(
            "maintenance.backup",
            {"output_dir": str(tmp_path / "packages"), "profile": "user_overlays"},
            archive_root=str(stack.archive_root),
            index_schema_version=999,
        )
        assert refused is not None and refused["outcome"] == "failed"
        assert refused["error"]["detail"] == "schema_version_mismatch"
        envelope = stack.client.operation(
            "maintenance.backup",
            {"output_dir": str(tmp_path / "packages"), "profile": "user_overlays"},
            archive_root=str(stack.archive_root),
        )
    assert envelope is not None and envelope["outcome"] == "completed", envelope
    package = Path(envelope["result"]["result"]["output_path"])
    assert (package / "user.db").is_file()
    assert not (package / "index.db").exists()


def _seed_terminal_embedding_failure(root: Path) -> None:
    with closing(sqlite3.connect(root / "embeddings.db")) as connection:
        with connection:
            connection.execute(
                """INSERT INTO embedding_failures (
                    failure_id, session_id, origin, message_refs_json, provider, model,
                    error_class, error_message, retryable, lifecycle_state,
                    created_at_ms, updated_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    "embedding-failure:route",
                    "codex-session:pending",
                    "codex-session",
                    "[]",
                    "voyage",
                    "voyage-4",
                    "provider_http_400",
                    "synthetic terminal failure",
                    0,
                    "terminal",
                    1800000000000,
                    1800000000000,
                ),
            )


def test_embedding_failure_resolution_reports_settled_identity_and_all_tier_versions(tmp_path: Path) -> None:
    """Adoption must not return a stale binding or an incomplete ready snapshot.

    Anti-vacuity: retaining the pre-write identity or skipping embedding tier
    version observation makes this fail even though the failure row resolves.
    """
    from polylogue.storage.archive_identity import ArchiveIdentity, ArchiveLocation

    with running_daemon_operations(tmp_path / "archive", seed_archive=_seed_terminal_embedding_failure) as stack:
        before = ArchiveIdentity.resolve_location(ArchiveLocation.resolve(stack.archive_root))
        envelope = stack.client.operation(
            "maintenance.embeddings.failure.resolve",
            {"failure_id": "embedding-failure:route", "resolution": "requeue"},
            archive_root=str(stack.archive_root),
            expected_archive_identity=before.authority_identity_digest,
        )
        after = ArchiveIdentity.resolve_location(ArchiveLocation.resolve(stack.archive_root))

    assert envelope is not None
    assert envelope["outcome"] == "completed", envelope
    assert before != after
    assert envelope["authority"]["admitted_archive_identity"] == before.authority_identity_digest
    assert envelope["archive"]["archive_identity"] == after.authority_identity_digest
    assert envelope["readiness"]["ready"] is True
    assert set(envelope["schema_versions"]) == {"source", "index", "embeddings", "user", "audit", "ops"}


def test_embedding_failure_resolution_defers_while_reader_pins_wal(tmp_path: Path) -> None:
    """A live reader blocks checkpoint, so resolution stays pending and retryable.

    Anti-vacuity: treating the busy checkpoint as corrupt marks the response
    non-retryable, while bypassing the lifecycle mutates the failure ledger.
    """
    with running_daemon_operations(tmp_path / "archive", seed_archive=_seed_terminal_embedding_failure) as stack:
        embeddings_db = stack.archive_root / "embeddings.db"
        with closing(sqlite3.connect(embeddings_db)) as reader:
            reader.execute("BEGIN")
            reader.execute("SELECT COUNT(*) FROM embedding_failures").fetchone()
            with closing(sqlite3.connect(embeddings_db)) as writer:
                with writer:
                    writer.execute(
                        "UPDATE embedding_failures SET updated_at_ms = updated_at_ms + 1 WHERE failure_id = ?",
                        ("embedding-failure:route",),
                    )
            envelope = stack.client.operation(
                "maintenance.embeddings.failure.resolve",
                {"failure_id": "embedding-failure:route", "resolution": "requeue"},
                archive_root=str(stack.archive_root),
            )
        with closing(sqlite3.connect(embeddings_db)) as settled:
            assert settled.execute(
                "SELECT lifecycle_state FROM embedding_failures WHERE failure_id = 'embedding-failure:route'"
            ).fetchone() == ("terminal",)

    assert envelope is not None
    assert envelope["outcome"] == "rejected", envelope
    assert envelope["error"]["code"] == "embedding_generation_busy"
    assert envelope["error"]["retryable"] is True


def test_embedding_resolution_delivers_index_schema_degradation_before_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A schema refusal before admission remains a typed degraded response.

    Anti-vacuity: requiring the resolution-only admission fields for degraded
    responses makes the client report an indeterminate mutation instead.
    """
    from polylogue.core.errors import SchemaSkew
    from polylogue.storage.archive_identity import ArchiveIdentity, ArchiveLocation

    def stale_index(*_args: object, **_kwargs: object) -> None:
        raise SchemaSkew("index", "current", "stale")

    with running_daemon_operations(tmp_path / "archive", seed_archive=_seed_terminal_embedding_failure) as stack:
        identity = ArchiveIdentity.resolve_location(ArchiveLocation.resolve(stack.archive_root))
        monkeypatch.setattr("polylogue.storage.sqlite.schema.assert_readable_archive_layout", stale_index)
        envelope = stack.client.operation(
            "maintenance.embeddings.failure.resolve",
            {"failure_id": "embedding-failure:route", "resolution": "requeue"},
            archive_root=str(stack.archive_root),
            expected_archive_identity=identity.authority_identity_digest,
        )

    assert envelope is not None
    assert envelope["outcome"] == "degraded", envelope
    assert envelope["error"]["code"] == "schema_skew"
    assert envelope["progress"]["tier"] == "index"
    assert "admitted_archive_identity" not in envelope["authority"]


def test_embedding_resolution_refuses_stale_embedding_schema_before_lifecycle(tmp_path: Path) -> None:
    """A stale embedding tier gets typed reconvergence guidance, not a failed write.

    Anti-vacuity: reading only the tier version lets lifecycle validation raise
    a generic error after admission and returns a non-retryable failure.
    """

    def stale_version(root: Path) -> None:
        with closing(sqlite3.connect(root / "embeddings.db")) as connection:
            connection.execute("PRAGMA user_version = 99")

    with running_daemon_operations(tmp_path / "archive", seed_archive=_seed_terminal_embedding_failure) as stack:
        stack.write_bridge.run_sync("test.stale_embeddings", stale_version, stack.archive_root)
        envelope = stack.client.operation(
            "maintenance.embeddings.failure.resolve",
            {"failure_id": "embedding-failure:route", "resolution": "requeue"},
            archive_root=str(stack.archive_root),
        )
        with closing(sqlite3.connect(stack.archive_root / "embeddings.db")) as connection:
            row = connection.execute(
                "SELECT lifecycle_state FROM embedding_failures WHERE failure_id = 'embedding-failure:route'"
            ).fetchone()

    assert envelope is not None
    assert envelope["outcome"] == "degraded", envelope
    assert envelope["error"]["code"] == "schema_skew"
    assert envelope["progress"]["tier"] == "embeddings"
    assert envelope["error"]["retryable"] is True
    assert row == ("terminal",)


def _seed_sessions(
    root: Path, *, count: int, title: str = "Operation route session", message_text: str | None = None
) -> tuple[str, ...]:
    """Seed one fully bootstrapped synthetic archive before daemon startup."""

    session_ids: list[str] = []
    for number in range(count):
        builder = (
            SessionBuilder(root / "index.db", f"operation-{number}")
            .provider("codex")
            .title(title)
            .add_message(
                text=message_text if message_text is not None else f"Synthetic daemon operation session {number}."
            )
        )
        builder.save()
        session_ids.append(builder.native_session_id())
    # ``SessionBuilder.save`` writes through the thread-local cached
    # connection, which ATTACHes the sibling durable tiers and keeps them
    # locked for the life of the thread. Seeding is setup: it must hand the
    # daemon stack an archive no other writer holds, exactly as a fresh
    # ``polylogued`` start would find one. Leaving the cache open made
    # ``PRAGMA journal_mode`` on ``source.db`` in a caller's own seed raise
    # ``database is locked`` -- a lock the daemon this file exercises can
    # never encounter, so the failure was the harness's, not the route's.
    from polylogue.storage.sqlite.connection import _clear_connection_cache

    _clear_connection_cache()
    return tuple(session_ids)


def test_one_uds_operation_request_returns_canonical_read_without_health_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mutation: add a health preflight to the real transport and ``exchanges``
    holds two entries instead of the one declared POST, which is red.

    The predecessor of this assertion patched ``DaemonClient.request_json``,
    which ``operation()`` never called, so a preflight issued through the
    transport the client actually uses left it green.
    """

    session_ids: tuple[str, ...] = ()

    def seed(root: Path) -> None:
        nonlocal session_ids
        session_ids = _seed_sessions(root, count=2)

    exchanges: list[tuple[str, str]] = []
    original = DaemonClient._request_json_response

    def record(self: DaemonClient, method: str, path: str, body: object = None, **kwargs: object) -> object:
        exchanges.append((method, path))
        return original(self, method, path, body, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(DaemonClient, "_request_json_response", record)
    with running_daemon_operations(tmp_path / "archive", seed_archive=seed) as stack:
        envelope = stack.client.operation(
            "cli.query",
            {"params": {"limit": 7}},
            archive_root=str(stack.archive_root),
        )

    assert envelope is not None
    assert envelope["outcome"] == "completed"
    assert envelope["readiness"]["ready"] is True
    assert envelope["authority"]["writes"] == "daemon-owned"
    assert envelope["result"]["total"] == len(session_ids)
    assert {item["id"] for item in envelope["result"]["items"]} == set(session_ids)
    assert exchanges == [("POST", "/api/operation")], exchanges


def test_repeated_daemon_query_uses_revision_scoped_result_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A repeated production UDS read reuses the canonical result payload.

    Anti-vacuity: the second request still traverses the real operation route,
    but a patched canonical query body must only run once.  The cache is then
    invalidated explicitly, proving that freshness is a write boundary rather
    than a TTL guess.
    """
    from polylogue.operations import daemon_reads
    from polylogue.storage.search.cache import invalidate_search_cache

    invalidate_search_cache()
    calls = 0
    original = daemon_reads._query_payload

    def counted(*args: object, **kwargs: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(daemon_reads, "_query_payload", counted)
    with running_daemon_operations(tmp_path / "archive") as stack:
        first = stack.client.operation("cli.query", {"params": {"limit": 1}}, archive_root=str(stack.archive_root))
        second = stack.client.operation("cli.query", {"params": {"limit": 1}}, archive_root=str(stack.archive_root))

    assert first is not None and second is not None
    assert first["result"] == second["result"]
    assert calls == 1


@pytest.mark.parametrize("scope_limited", [False, True])
def test_embedding_backfill_is_accepted_streams_progress_and_recovers_audit_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scope_limited: bool
) -> None:
    """Embedding progress is pre-terminal and its final counts survive runtime restart.

    Anti-vacuity: routing backfill around the accepted lifecycle loses the
    durable reference, while returning only generic operation counters loses
    the domain receipt after restart.
    """
    import asyncio
    from types import SimpleNamespace

    from polylogue.daemon import embedding_owner as embedding_owner_module
    from polylogue.daemon.embedding_owner import EmbeddingConvergenceResult

    composed = False

    def compose(_index: Path, **_kwargs: object) -> object:
        nonlocal composed
        composed = True

        # Limits, progress and stop signals are per pass, never per owner.
        async def converge(_scope: object, **limits: object) -> EmbeddingConvergenceResult:

            emit = limits["progress_callback"]
            limited = bool(limits["scope_limited"])
            assert callable(emit)
            assert limits["max_messages"] == 17
            assert limits["max_cost_usd"] == 0.25
            assert limits["stop_after_seconds"] == 9
            assert limits["max_errors"] == 2
            cast(Any, emit)({"state": "started", "session_id": "codex:synthetic", "estimated_cost_usd": 0.0001})
            await asyncio.sleep(0.05)
            report = SimpleNamespace(
                done=1,
                pending=2,
                failed=0,
                work=SimpleNamespace(computed=1),
            )
            return EmbeddingConvergenceResult(cast(Any, report), "max_sessions" if limited else None)

        return converge

    monkeypatch.setattr(
        "polylogue.operations.embedding_derivation.select_embedding_session_window",
        lambda *_args, **_kwargs: (("codex:synthetic",), scope_limited),
    )

    monkeypatch.setattr(embedding_owner_module, "compose_embedding_convergence", compose)
    request_id = "embedding-accepted-progress"
    payload: dict[str, object] = {
        "max_sessions": 1,
        "max_messages": 17,
        "max_cost_usd": 0.25,
        "stop_after_seconds": 9,
        "max_errors": 2,
    }
    with running_daemon_operations(tmp_path / "archive") as stack:
        progress: list[dict[str, object]] = []

        def receive_progress(frame: object) -> None:
            if isinstance(frame, dict):
                progress.append(frame)

        terminal = stack.client.operation_to_completion(
            "maintenance.embeddings.backfill",
            payload,
            archive_root=str(stack.archive_root),
            request_id=request_id,
            progress_callback=receive_progress,
        )
        assert terminal is not None
        assert terminal["outcome"] == ("interrupted" if scope_limited else "completed")
        assert composed
        assert terminal["accepted_reference"]["request_id"] == request_id
        assert terminal["accepted_reference"]["artifact_kind"] == "operation"
        exchange = stack.runtime._exchanges[request_id]
        assert exchange.progress_sequence == 1
        assert progress and progress[0]["state"] == "started"
        assert progress[0]["sequence"] == 1
        assert "estimated_cost_usd" in progress[0]
        assert terminal["result"]["result"] == {"done": 1, "pending": 2, "failed": 0}
        accepted_reference = terminal["accepted_reference"]

    with running_daemon_operations(tmp_path / "archive") as restarted:
        recovered = restarted.client.operation(
            "maintenance.embeddings.backfill",
            payload,
            archive_root=str(restarted.archive_root),
            request_id=request_id,
        )
        assert recovered is not None
        assert recovered["outcome"] == ("interrupted" if scope_limited else "completed")
        assert recovered["accepted_reference"] == accepted_reference
        assert recovered["result"]["result"] == {"done": 1, "pending": 2, "failed": 0}


def test_embedding_backfill_cancel_is_request_scoped_and_keeps_partial_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Accepted cancellation is observed between work items and preserves counts."""
    import asyncio
    from threading import Event
    from types import SimpleNamespace

    from polylogue.daemon import embedding_owner as embedding_owner_module
    from polylogue.daemon.embedding_owner import EmbeddingConvergenceResult

    started = Event()
    composed = Event()

    def compose(_index: Path, **_kwargs: object) -> object:
        composed.set()

        async def converge(_scope: object, **limits: object) -> EmbeddingConvergenceResult:
            emit = limits["progress_callback"]
            quiet = limits["quiet"]
            assert callable(emit) and callable(quiet)
            cast(Any, emit)({"state": "started", "ordinal": 0, "estimated_cost_usd": 0.001})
            started.set()
            while not cast(Any, quiet)():
                await asyncio.sleep(0.01)
            cancelled = bool(cast(Any, quiet)())
            report = SimpleNamespace(
                done=1,
                pending=4,
                failed=0,
                work=SimpleNamespace(computed=1),
            )
            return EmbeddingConvergenceResult(cast(Any, report), "cancelled" if cancelled else None)

        return converge

    monkeypatch.setattr(embedding_owner_module, "compose_embedding_convergence", compose)
    request_id = "embedding-cancel-partial"
    with running_daemon_operations(tmp_path / "archive") as stack:
        accepted = stack.client.operation(
            "maintenance.embeddings.backfill", {}, archive_root=str(stack.archive_root), request_id=request_id
        )
        assert accepted is not None and accepted["outcome"] == "accepted"
        assert accepted["accepted_reference"]["request_id"] == request_id
        assert accepted["accepted_reference"]["artifact_kind"] == "operation"
        assert composed.wait(timeout=2)
        assert started.wait(timeout=2)
        cancellation = stack.client.cancel(request_id, archive_root=str(stack.archive_root))
        assert cancellation is not None
        terminal = stack.client.operation_to_completion(
            "maintenance.embeddings.backfill",
            {},
            archive_root=str(stack.archive_root),
            request_id=request_id,
        )
        assert terminal is not None
        assert terminal["outcome"] == "cancelled"
        assert terminal["accepted_reference"] == accepted["accepted_reference"]
        assert terminal["result"]["outcome"] == "cancelled"
        assert terminal["result"]["result"] == {"done": 1, "pending": 4, "failed": 0}


@pytest.mark.parametrize(
    ("done", "deferred", "effect"),
    [
        (0, None, "no-effect"),
        (2, None, "committed"),
        (0, "max_errors", "no-effect"),
    ],
    ids=["all-keys-failed", "some-keys-failed", "max-errors-stop"],
)
def test_embedding_backfill_provider_failures_are_a_failed_terminal_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, done: int, deferred: str | None, effect: str
) -> None:
    """A pass whose provider rejected keys never reports ``completed``.

    Without ``max_errors`` no deferral reason is set, so the outcome used to
    be ``completed`` and the attempt finalized as applied while
    ``result.failed`` was nonzero. The returned envelope, the durable receipt
    and a restarted daemon's replay must all say ``failed`` with the measured
    counts.

    Anti-vacuity: classify the outcome from ``stop_reason`` alone again and the
    first two cases report ``completed``; finalize them as ``applied`` and the
    replayed outcome after restart is ``completed``.
    """
    from types import SimpleNamespace

    from polylogue.daemon import embedding_owner as embedding_owner_module
    from polylogue.daemon.embedding_owner import EmbeddingConvergenceResult

    def compose(_index: Path, **_kwargs: object) -> object:
        async def converge(_scope: object, **_limits: object) -> EmbeddingConvergenceResult:
            report = SimpleNamespace(done=done, pending=0, failed=3, work=SimpleNamespace(computed=done))
            return EmbeddingConvergenceResult(cast(Any, report), deferred)

        return converge

    monkeypatch.setattr(embedding_owner_module, "compose_embedding_convergence", compose)
    request_id = f"embedding-provider-failures-{done}-{deferred}"
    expected_counts = {"done": done, "pending": 0, "failed": 3}
    with running_daemon_operations(tmp_path / "archive") as stack:
        terminal = stack.client.operation_to_completion(
            "maintenance.embeddings.backfill",
            {},
            archive_root=str(stack.archive_root),
            request_id=request_id,
        )
        assert terminal is not None
        assert terminal["outcome"] == "failed", terminal
        assert terminal["result"]["outcome"] == "failed"
        assert terminal["result"]["effect"] == effect
        assert terminal["result"]["stop_reason"] == deferred
        assert terminal["result"]["error"]["code"] == "embedding_keys_failed"
        assert terminal["result"]["result"] == expected_counts
        assert terminal["result"]["progress"]["failed"] == 3
        accepted_reference = terminal["accepted_reference"]

    with running_daemon_operations(tmp_path / "archive") as restarted:
        recovered = restarted.client.operation(
            "maintenance.embeddings.backfill",
            {},
            archive_root=str(restarted.archive_root),
            request_id=request_id,
        )
        assert recovered is not None
        assert recovered["outcome"] == "failed", recovered
        assert recovered["accepted_reference"] == accepted_reference
        assert recovered["result"]["outcome"] == "failed"
        assert recovered["result"]["error"]["code"] == "embedding_keys_failed"
        assert recovered["result"]["result"] == expected_counts


def test_machine_listener_uses_the_independent_operation_handler(tmp_path: Path) -> None:
    """Mutation: delegate machine requests through the browser handler and this fails."""

    def seed(root: Path) -> None:
        _seed_sessions(root, count=1)

    with running_daemon_operations(tmp_path / "archive", seed_archive=seed) as stack:
        assert stack.server.RequestHandlerClass is MachineOperationHandler
        assert MachineOperationHandler.__bases__[0].__name__ == "BaseHTTPRequestHandler"
        envelope = stack.client.operation("status", {}, archive_root=str(stack.archive_root))

    assert envelope is not None
    assert envelope["result"]["total_sessions"] == 1


def test_fresh_runtime_prepares_tier_journals_before_first_snapshot_and_mutation(tmp_path: Path) -> None:
    """Mutation: omit writer startup journal activation and the first audited preview self-locks."""
    import sqlite3

    ids: tuple[str, ...] = ()

    def seed(root: Path) -> None:
        nonlocal ids
        ids = _seed_sessions(root, count=1)
        # Source/user/audit can be untouched rollback-journal tiers in a fresh
        # archive. Startup, never the operation reader, must activate them.
        for tier in ("source", "user", "audit"):
            with sqlite3.connect(root / f"{tier}.db") as connection:
                connection.execute("PRAGMA journal_mode=DELETE")

    with running_daemon_operations(tmp_path / "archive", seed_archive=seed) as stack:
        read = stack.client.operation("cli.query", {"params": {}}, archive_root=str(stack.archive_root))
        assert read is not None and read["outcome"] == "completed"
        preview = stack.client.operation_to_completion(
            "mutation.session.delete.preview", {"session_ids": list(ids)}, archive_root=str(stack.archive_root)
        )
        assert preview is not None and preview["outcome"] == "completed"
        assert preview["result"]["session_count"] == len(ids)
        assert preview["result"]["session_ids_sample"] == list(ids[:20])


def test_authentication_refusal_is_not_an_indeterminate_mutation(tmp_path: Path) -> None:
    """Mutation: treat the ingress 401 as a lost receipt and the typed refusal disappears."""
    with running_daemon_operations(tmp_path / "archive") as stack:
        stack.server.auth_token = "synthetic-test-credential"
        with pytest.raises(DaemonOperationRejectedError) as rejected:
            stack.client.operation("mutation.session.delete.preview", {"session_ids": ["codex:absent"]})
        assert rejected.value.outcome == "unauthorized"
        assert not stack.runtime._exchanges


@pytest.mark.parametrize("operation", ["status", "mutation.session.delete.preview"])
def test_connection_saturation_refuses_before_acceptance_and_recovers(tmp_path: Path, operation: str) -> None:
    """An empty 503 or bypassed admission loses the explicit no-execution guarantee."""
    with running_daemon_operations(tmp_path / "archive") as stack:
        # Reserve the real ingress budget without a timing-dependent fleet of
        # slow sockets. The next request still traverses the production listener.
        for _ in range(stack.server.request_queue_size):
            assert stack.server._connections.acquire(blocking=False)
        try:
            payload: dict[str, object] = {} if operation == "status" else {"session_ids": ["codex:absent"]}
            with pytest.raises(DaemonOperationRejectedError) as rejected:
                stack.client.operation(operation, payload)
            assert rejected.value.outcome == "connection_backpressure"
            assert not stack.runtime._exchanges
        finally:
            for _ in range(stack.server.request_queue_size):
                stack.server._connections.release()

        recovered = stack.client.operation("status", {}, archive_root=str(stack.archive_root))
        assert recovered is not None and recovered["outcome"] == "completed"


def test_changed_intent_cannot_reuse_a_durable_request_id(tmp_path: Path) -> None:
    """Mutation: ignore the durable fingerprint and a changed selection inherits prior authority."""
    ids: tuple[str, ...] = ()

    def seed(root: Path) -> None:
        nonlocal ids
        ids = _seed_sessions(root, count=2)

    with running_daemon_operations(tmp_path / "archive", seed_archive=seed) as stack:
        first = stack.client.operation_to_completion(
            "mutation.session.delete.preview",
            {"session_ids": [ids[0]]},
            archive_root=str(stack.archive_root),
            request_id="stable-preview-intent",
        )
        assert first is not None
        assert first["outcome"] == "completed"
        changed = stack.client.operation(
            "mutation.session.delete.preview",
            {"session_ids": [ids[1]]},
            archive_root=str(stack.archive_root),
            request_id="stable-preview-intent",
        )
        assert changed is not None and changed["outcome"] == "rejected"
        assert changed.get("accepted_reference") is None
        assert all(stack.session_exists(session_id) for session_id in ids)


def test_identical_intent_replays_the_recorded_mutation_without_a_second_effect(tmp_path: Path) -> None:
    """A duplicate submission of one intent is the same operation, not a second write.

    The replay answers from the durable request record: its receipt names the
    first execution and still reports the one tag it added. A fresh request id
    with the same intent executes again and finds nothing to add, which is
    what a replay that re-executed would have reported.

    Mutation: re-dispatch a completed durable request and the replay reports
    ``affected_count == 0`` under a new receipt.
    """
    ids: tuple[str, ...] = ()

    def seed(root: Path) -> None:
        nonlocal ids
        ids = _seed_sessions(root, count=1)

    with running_daemon_operations(tmp_path / "archive", seed_archive=seed) as stack:
        payload: dict[str, object] = {"session_ids": [ids[0]], "tags": ["replayed-intent"]}
        first = stack.client.operation_to_completion(
            "mutation.session.tag", dict(payload), archive_root=str(stack.archive_root), request_id="stable-tag-intent"
        )
        again = stack.client.operation_to_completion(
            "mutation.session.tag", dict(payload), archive_root=str(stack.archive_root), request_id="stable-tag-intent"
        )
        fresh = stack.client.operation_to_completion(
            "mutation.session.tag", dict(payload), archive_root=str(stack.archive_root), request_id="fresh-tag-intent"
        )

    assert first is not None and again is not None and fresh is not None
    assert [first["outcome"], again["outcome"], fresh["outcome"]] == ["completed"] * 3
    assert first["result"]["affected_count"] == 1
    assert again["result"]["affected_count"] == 1
    assert again["result"]["receipt_ref"] == first["result"]["receipt_ref"]
    assert fresh["result"]["affected_count"] == 0
    assert fresh["result"]["receipt_ref"] != first["result"]["receipt_ref"]


def test_operation_route_delivers_a_large_canonical_row(tmp_path: Path) -> None:
    """The actual resident/client exchange preserves a permitted single value."""
    text = "[large] λ\n" * (1024 * 1024)
    ids: tuple[str, ...] = ()

    def seed(root: Path) -> None:
        nonlocal ids
        ids = _seed_sessions(root, count=1, message_text=text)

    with running_daemon_operations(tmp_path / "archive", seed_archive=seed) as stack:
        envelope = stack.client.operation(
            "session.read", {"ref": ids[0], "limit": 1}, archive_root=str(stack.archive_root)
        )

    assert envelope is not None and envelope["outcome"] == "completed"
    result = envelope["result"]
    assert result["session"]["messages"][0]["blocks"][0]["text"] == text
    assert result["total"] == 1 and result["complete"] is True
    assert result["continuation"] is None and result["lineage_complete"] is True
    assert envelope["authority_snapshot"]["generation"]


def test_kernel_authenticated_uid_reference_survives_client_and_daemon_restart(tmp_path: Path) -> None:
    """Mutation: principal contains a PID/random token and durable retry changes authority."""
    root = tmp_path / "archive"
    ids: tuple[str, ...] = ()

    def seed(root: Path) -> None:
        nonlocal ids
        ids = _seed_sessions(root, count=1)

    with running_daemon_operations(root, seed_archive=seed) as first:
        prepared = first.client.operation_to_completion(
            "mutation.session.delete.preview",
            {"session_ids": list(ids)},
            archive_root=str(root),
            request_id="durable-local-preview",
        )
    with running_daemon_operations(root) as restarted:
        recovered = restarted.client.operation_to_completion(
            "mutation.session.delete.preview",
            {"session_ids": list(ids)},
            archive_root=str(root),
            request_id="durable-local-preview",
        )

    assert prepared is not None and recovered is not None
    assert prepared["accepted_reference"]["principal_ref"] == f"daemon:unix:uid:{os.getuid()}"
    assert recovered["accepted_reference"] == prepared["accepted_reference"]
    assert recovered["result"] == prepared["result"]


def test_restart_recovers_indeterminate_mutation_without_replaying_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A durable unknown effect converges once at startup; a retry never resubmits it.

    Since #5688 startup recovery resolves a dead or unknown run by convergent
    replay, so the restarted daemon applies the plan exactly once more (a
    no-op against the already-deleted target). Anti-vacuity: re-dispatching
    the handler for the retried request id raises ``apply_calls`` past the
    startup count.
    """
    root = tmp_path / "archive"
    session_ids: tuple[str, ...] = ()

    def seed(root: Path) -> None:
        nonlocal session_ids
        session_ids = _seed_sessions(root, count=1)

    apply_calls = 0
    original_apply = SessionDeleteActuator.apply

    def apply_once_as_indeterminate(
        actuator: SessionDeleteActuator, plan: MutationPlan, args: SessionDeleteArgs
    ) -> MutationReceipt:
        nonlocal apply_calls
        apply_calls += 1
        # The archive mutation really happens, but the worker loses the
        # outcome at the domain boundary.  Audit therefore persists unknown,
        # which is the restart case this route must recover.
        return replace(original_apply(actuator, plan, args), status="unknown", detail="synthetic lost outcome")

    monkeypatch.setattr(SessionDeleteActuator, "apply", apply_once_as_indeterminate)
    with running_daemon_operations(root, seed_archive=seed) as first:
        preview = first.client.operation_to_completion(
            "mutation.session.delete.preview",
            {"session_ids": list(session_ids)},
            archive_root=str(root),
            request_id="indeterminate-preview",
        )
        assert preview is not None
        authorization = first.client.operation_to_completion(
            "mutation.session.delete.authorize",
            {"preview_request_id": preview["result"]["reference"]["request_id"]},
            archive_root=str(root),
            request_id="indeterminate-authorize",
        )
        assert authorization is not None
        lost = first.client.operation_to_completion(
            "mutation.session.delete.execute",
            {"authorization_request_id": authorization["result"]["reference"]["request_id"]},
            archive_root=str(root),
            request_id="indeterminate-execute",
        )
        assert lost is not None and lost["outcome"] == "indeterminate"
        accepted_reference = lost["accepted_reference"]

    assert apply_calls == 1
    with running_daemon_operations(root) as restarted:
        # Startup recovery replays the unknown run once, convergently.
        # Re-introduce the persisted unknown outcome after startup so the
        # retry route is tested against a durable indeterminate record,
        # exactly as a crashed domain writer leaves it.
        assert apply_calls == 2
        with sqlite3.connect(root / "audit.db") as connection:
            connection.execute(
                """
                UPDATE operation_runs
                SET unknown_count = 1, unknown_reason = ?
                WHERE operation_id IN (
                    SELECT operation_id FROM machine_request_parts
                    WHERE request_id = ? AND operation_id IS NOT NULL
                )
                """,
                ("synthetic lost outcome", "indeterminate-execute"),
            )
            connection.commit()
        recovered = restarted.client.operation(
            "mutation.session.delete.execute",
            {"authorization_request_id": authorization["result"]["reference"]["request_id"]},
            archive_root=str(root),
            request_id="indeterminate-execute",
        )
        assert recovered is not None
        assert recovered["outcome"] == "indeterminate"
        assert recovered["accepted_reference"] == accepted_reference
        assert recovered["result"]["reference"] == accepted_reference

    assert apply_calls == 2
    assert all(not restarted.session_exists(session_id) for session_id in session_ids)


def test_crash_recovery_replays_a_delete_on_exactly_the_recorded_id_not_a_prefix_sibling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Startup replay of an interrupted delete acts on the recorded full id only.

    The first daemon deletes the target and loses the outcome, so its run is
    durably unknown. On restart the recovery replay finds the target gone.
    Anti-vacuity: resolve existence through ``resolve_session_id`` (whose
    prefix fallback maps the vanished ``...:ext-shared`` onto the surviving
    ``...:ext-shared-sibling``) and the replay deletes the sibling -- which no
    operator ever previewed -- so this test is red on the sibling assertion.
    """
    root = tmp_path / "archive"
    target = sibling = ""

    def seed(root: Path) -> None:
        nonlocal target, sibling
        builders = []
        for native in ("shared", "shared-sibling"):
            builder = (
                SessionBuilder(root / "index.db", native)
                .provider("codex")
                .title("Prefix-sharing session")
                .add_message(text=f"Synthetic session {native}.")
            )
            builder.save()
            builders.append(builder)
        target, sibling = (builder.native_session_id() for builder in builders)
        assert sibling.startswith(target)
        from polylogue.storage.sqlite.connection import _clear_connection_cache

        _clear_connection_cache()

    original_apply = SessionDeleteActuator.apply

    def apply_then_lose_outcome(
        actuator: SessionDeleteActuator, plan: MutationPlan, args: SessionDeleteArgs
    ) -> MutationReceipt:
        return replace(original_apply(actuator, plan, args), status="unknown", detail="synthetic lost outcome")

    with monkeypatch.context() as patched:
        patched.setattr(SessionDeleteActuator, "apply", apply_then_lose_outcome)
        with running_daemon_operations(root, seed_archive=seed) as first:
            preview = first.client.operation_to_completion(
                "mutation.session.delete.preview",
                {"session_ids": [target]},
                archive_root=str(root),
                request_id="prefix-preview",
            )
            assert preview is not None
            assert preview["result"]["session_count"] == 1
            authorization = first.client.operation_to_completion(
                "mutation.session.delete.authorize",
                {"preview_request_id": preview["result"]["reference"]["request_id"]},
                archive_root=str(root),
                request_id="prefix-authorize",
            )
            assert authorization is not None
            lost = first.client.operation_to_completion(
                "mutation.session.delete.execute",
                {"authorization_request_id": authorization["result"]["reference"]["request_id"]},
                archive_root=str(root),
                request_id="prefix-execute",
            )
            assert lost is not None and lost["outcome"] == "indeterminate"
            assert not first.session_exists(target), lost
            assert first.session_exists(sibling)

    # Startup recovery replays the unknown run with the real actuator.
    with running_daemon_operations(root) as restarted:
        assert not restarted.session_exists(target)
        assert restarted.session_exists(sibling), "recovery replay deleted a session outside the recorded plan"


def test_delete_preview_count_equals_the_applied_count_and_spares_prefix_siblings(tmp_path: Path) -> None:
    """The daemon deletes exactly the ids its preview recorded.

    Anti-vacuity: resolve a previewed id by prefix at execution and the
    unselected ``...-sibling`` rows share the selected ids' prefixes, so the
    applied count or the surviving set diverges from the preview.
    """
    root = tmp_path / "archive"
    selected: list[str] = []
    spared: list[str] = []

    def seed(root: Path) -> None:
        for number in range(3):
            for suffix, bucket in (("", selected), ("-sibling", spared)):
                builder = (
                    SessionBuilder(root / "index.db", f"counted-{number}{suffix}")
                    .provider("codex")
                    .title("Counted session")
                    .add_message(text=f"Synthetic counted session {number}{suffix}.")
                )
                builder.save()
                bucket.append(builder.native_session_id())
        from polylogue.storage.sqlite.connection import _clear_connection_cache

        _clear_connection_cache()

    with running_daemon_operations(root, seed_archive=seed) as stack:
        preview = stack.client.operation_to_completion(
            "mutation.session.delete.preview",
            {"session_ids": selected},
            archive_root=str(root),
            request_id="counted-preview",
        )
        assert preview is not None
        prepared_count = preview["result"]["session_count"]
        authorization = stack.client.operation_to_completion(
            "mutation.session.delete.authorize",
            {"preview_request_id": preview["result"]["reference"]["request_id"]},
            archive_root=str(root),
            request_id="counted-authorize",
        )
        assert authorization is not None
        executed = stack.client.operation_to_completion(
            "mutation.session.delete.execute",
            {"authorization_request_id": authorization["result"]["reference"]["request_id"]},
            archive_root=str(root),
            request_id="counted-execute",
        )
        assert executed is not None
        assert prepared_count == len(selected)
        assert executed["result"]["affected_count"] == prepared_count, (executed, preview)
        assert not any(stack.session_exists(session_id) for session_id in selected)
        assert all(stack.session_exists(session_id) for session_id in spared)


def test_restart_resumes_an_accepted_request_whose_daemon_died_before_its_first_part(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An accepted-but-unstarted execution resumes on retry after a restart.

    The first daemon durably accepts the execution batch and then loses its
    handler before any part starts, leaving every part unattempted. Anti-vacuity
    (polylogue-4xhzs): returning the durable ``accepted`` state on retry instead
    of re-dispatching the handler leaves the request at ``accepted`` forever and
    the session undeleted.
    """
    from polylogue.operations import daemon_mutations

    root = tmp_path / "archive"
    session_ids: tuple[str, ...] = ()

    def seed(root: Path) -> None:
        nonlocal session_ids
        session_ids = _seed_sessions(root, count=1)

    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    real_store = ArchiveStore
    crash = {"armed": False}

    class _DiesAfterAcceptance:
        @staticmethod
        def open_existing(*args: object, **kwargs: object) -> object:
            if crash["armed"]:
                raise RuntimeError("synthetic handler death after acceptance")
            return real_store.open_existing(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(daemon_mutations, "ArchiveStore", _DiesAfterAcceptance)
    request_id = "accepted-before-first-part"
    with running_daemon_operations(root, seed_archive=seed) as first:
        preview = first.client.operation_to_completion(
            "mutation.session.delete.preview",
            {"session_ids": list(session_ids)},
            archive_root=str(root),
        )
        assert preview is not None
        authorization = first.client.operation_to_completion(
            "mutation.session.delete.authorize",
            {"preview_request_id": preview["result"]["reference"]["request_id"]},
            archive_root=str(root),
        )
        assert authorization is not None
        crash["armed"] = True
        stranded = first.client.operation(
            "mutation.session.delete.execute",
            {"authorization_request_id": authorization["result"]["reference"]["request_id"]},
            archive_root=str(root),
            request_id=request_id,
        )
        assert stranded is not None
        assert stranded["outcome"] == "accepted"
        assert stranded["result"]["not_attempted"] == [0]
    crash["armed"] = False

    with running_daemon_operations(root) as restarted:
        resumed = restarted.client.operation_to_completion(
            "mutation.session.delete.execute",
            {"authorization_request_id": authorization["result"]["reference"]["request_id"]},
            archive_root=str(root),
            request_id=request_id,
        )
        assert resumed is not None
        assert resumed["outcome"] == "completed"
        assert resumed["accepted_reference"] == stranded["accepted_reference"]
        assert resumed["result"]["completed_chunks"] == 1
        assert all(not restarted.session_exists(session_id) for session_id in session_ids)


def test_disconnected_after_durable_acceptance_recovers_without_replaying_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A lost post-acceptance UDS response recovers one durable execution.

    The raw peer is closed only after the actuator has entered, which proves
    that the acceptance record exists before transport loss.  A retry with the
    same request identity must observe that record, and a restarted daemon must
    return its terminal receipt without invoking the actuator again.
    """
    root = tmp_path / "archive"
    session_ids: tuple[str, ...] = ()

    def seed(root: Path) -> None:
        nonlocal session_ids
        session_ids = _seed_sessions(root, count=1)

    entered_apply = threading.Event()
    release_apply = threading.Event()
    apply_calls = 0
    original_apply = SessionDeleteActuator.apply

    def blocked_apply(self: SessionDeleteActuator, plan: MutationPlan, args: SessionDeleteArgs) -> MutationReceipt:
        nonlocal apply_calls
        apply_calls += 1
        entered_apply.set()
        if not release_apply.wait(timeout=5):
            raise TimeoutError("test did not release the accepted delete actuator")
        return original_apply(self, plan, args)

    monkeypatch.setattr(SessionDeleteActuator, "apply", blocked_apply)
    request_id = "disconnect-after-acceptance"
    accepted_reference: dict[str, object]
    authorization_request_id: str
    with running_daemon_operations(root, seed_archive=seed) as stack:
        preview = stack.client.operation_to_completion(
            "mutation.session.delete.preview",
            {"session_ids": list(session_ids)},
            archive_root=str(root),
        )
        assert preview is not None
        authorization = stack.client.operation_to_completion(
            "mutation.session.delete.authorize",
            {"preview_request_id": preview["result"]["reference"]["request_id"]},
            archive_root=str(root),
        )
        assert authorization is not None
        authorization_request_id = str(authorization["result"]["reference"]["request_id"])

        from polylogue.operations.daemon_protocol import DaemonOperationRequest

        request = DaemonOperationRequest(
            "mutation.session.delete.execute",
            {"authorization_request_id": authorization_request_id},
            archive_root=str(root),
            request_id=request_id,
            deadline_ms=10_000,
        )
        body = json.dumps(request.to_dict(), separators=(",", ":")).encode()
        peer = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            peer.connect(str(stack.socket_path))
            peer.sendall(
                b"POST /api/operation HTTP/1.1\r\n"
                b"Host: localhost\r\n"
                b"Content-Type: application/json\r\n" + f"Content-Length: {len(body)}\r\n\r\n".encode() + body
            )
            # SessionDeleteActuator.apply runs only after the durable
            # accept_execution_batch transition has committed.
            assert entered_apply.wait(timeout=2)
        finally:
            peer.close()

        recovered = stack.client.operation(
            request.operation,
            request.payload,
            archive_root=request.archive_root,
            request_id=request.request_id,
            deadline_ms=request.deadline_ms,
        )
        assert recovered is not None
        assert recovered["outcome"] == "running"
        accepted_reference = recovered["accepted_reference"]
        assert recovered["result"]["reference"] == accepted_reference
        assert recovered["result"]["completed_chunks"] == 0

        try:
            release_apply.set()
            terminal = stack.client.operation_to_completion(
                request.operation,
                request.payload,
                archive_root=str(root),
                request_id=request_id,
            )
            assert terminal is not None
            assert terminal["outcome"] == "completed"
            assert terminal["accepted_reference"] == accepted_reference
            assert terminal["result"]["reference"] == accepted_reference
            assert terminal["result"]["completed_chunks"] == 1
        finally:
            release_apply.set()

        assert apply_calls == 1
        assert all(not stack.session_exists(session_id) for session_id in session_ids)

    with running_daemon_operations(root) as restarted:
        replay = restarted.client.operation(
            request.operation,
            request.payload,
            archive_root=str(root),
            request_id=request.request_id,
            deadline_ms=request.deadline_ms,
        )
        assert replay is not None
        assert replay["outcome"] == "completed"
        assert replay["accepted_reference"] == accepted_reference
        assert replay["result"]["reference"] == accepted_reference
        assert apply_calls == 1


def test_uds_refuses_when_kernel_peer_credentials_cannot_be_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mutation: fall back to a guessed local principal and this read executes.

    Only the *server* side of the connection loses ``SO_PEERCRED`` here. The
    UDS client performs the mirror-image check before it sends the machine
    bearer (``polylogue.daemon_client._reject_foreign_peer``), so breaking the
    option for every socket in the process refuses the request client-side and
    never exercises the route under test. The accepted server socket is bound
    to the listening path; the client's connected socket is unbound and
    reports an empty name, which separates the two ends without weakening
    either check.
    """
    from polylogue.operations.daemon_protocol import DAEMON_OPERATION_PROTOCOL

    original = socket.socket.getsockopt

    def unavailable(sock: socket.socket, level: int, name: int, *args: object) -> object:
        if level == socket.SOL_SOCKET and name == socket.SO_PEERCRED and sock.getsockname():
            raise OSError("synthetic unavailable peer credentials")
        return original(sock, level, name, *args)

    with running_daemon_operations(tmp_path / "archive") as stack:
        monkeypatch.setattr(socket.socket, "getsockopt", unavailable)
        exchange = stack.client._request_json_response(
            "POST",
            "/api/operation",
            {
                "protocol": DAEMON_OPERATION_PROTOCOL,
                "request_id": "unavailable-peer-credentials",
                "operation": "completion",
                "payload": {"kind": "field"},
            },
        )
    assert exchange is not None
    status, refused = exchange
    assert status == 401
    assert refused is not None
    assert refused["outcome"] == "rejected"
    assert refused["error"]["code"] == "peer_authentication_unavailable"


def test_disconnected_queued_control_releases_its_compute_admission(tmp_path: Path) -> None:
    """A pre-acceptance control disconnect must not retain a future worker slot.

    Anti-vacuity: omitting the control cancellation handle from the scheduler
    leaves the queued preview exchange and its reservation until the blockers
    release, even though the real UDS peer has already disconnected.
    """
    from polylogue.operations.daemon_protocol import DAEMON_OPERATION_PROTOCOL

    entered = threading.Semaphore(0)
    release = threading.Event()

    def block_worker() -> None:
        entered.release()
        # Generous: this worker must stay parked for the whole exchange
        # (archive bootstrap, audit reads, the cancel and its envelope), and
        # its early exit would let the queued task start and void the test.
        assert release.wait(timeout=60)

    with running_daemon_operations(tmp_path / "archive") as stack:
        blockers = [stack.execution_kernel.submit(block_worker) for _ in range(2)]
        assert all(entered.acquire(timeout=2) for _ in blockers)
        request_id = "disconnected-queued-control"
        body = json.dumps(
            {
                "protocol": DAEMON_OPERATION_PROTOCOL,
                "request_id": request_id,
                "operation": "mutation.session.delete.preview",
                "payload": {"session_ids": ["codex:absent"]},
                "archive_root": str(stack.archive_root),
            },
            separators=(",", ":"),
        ).encode()
        peer = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            peer.connect(str(stack.socket_path))
            peer.sendall(
                b"POST /api/operation HTTP/1.1\r\n"
                b"Host: localhost\r\n"
                b"Content-Type: application/json\r\n" + f"Content-Length: {len(body)}\r\n\r\n".encode() + body
            )
            with stack.runtime._condition:
                assert stack.runtime._condition.wait_for(lambda: request_id in stack.runtime._exchanges, timeout=2)
            peer.close()
            with stack.runtime._condition:
                assert stack.runtime._condition.wait_for(lambda: request_id not in stack.runtime._exchanges, timeout=2)
            assert stack.execution_kernel.snapshot().used_units == len(blockers)
        finally:
            release.set()
            peer.close()
            for blocker in blockers:
                blocker.future.result(timeout=2)


def test_cancelled_long_delete_retains_writer_until_blocked_apply_releases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mutation: release the admitted writer on cancellation and the later preview completes early."""

    session_ids: tuple[str, ...] = ()

    def seed(root: Path) -> None:
        nonlocal session_ids
        session_ids = _seed_sessions(root, count=MUTATION_PLAN_PAGE_SIZE + 1)

    entered_apply = threading.Event()
    release_apply = threading.Event()
    original_apply = SessionDeleteActuator.apply

    def blocked_apply(self: SessionDeleteActuator, plan: MutationPlan, args: SessionDeleteArgs) -> MutationReceipt:
        entered_apply.set()
        if not release_apply.wait(timeout=5):
            raise TimeoutError("test did not release the admitted delete actuator")
        return original_apply(self, plan, args)

    monkeypatch.setattr(SessionDeleteActuator, "apply", blocked_apply)
    with running_daemon_operations(tmp_path / "archive", seed_archive=seed) as stack:
        preview = stack.client.operation_to_completion(
            "mutation.session.delete.preview",
            {"session_ids": list(session_ids)},
            archive_root=str(stack.archive_root),
        )
        assert preview is not None
        preview_result = preview["result"]
        authorization = stack.client.operation_to_completion(
            "mutation.session.delete.authorize",
            {"preview_request_id": preview_result["reference"]["request_id"]},
            archive_root=str(stack.archive_root),
        )
        assert authorization is not None
        execute_request_id = "blocked-delete"
        execute_result: queue.Queue[dict[str, object]] = queue.Queue()

        def execute() -> None:
            client = DaemonClient(stack.socket_path, timeout_s=5)
            response = client.operation(
                "mutation.session.delete.execute",
                {"authorization_request_id": authorization["result"]["reference"]["request_id"]},
                archive_root=str(stack.archive_root),
                request_id=execute_request_id,
            )
            assert response is not None
            execute_result.put(response)

        execute_thread = threading.Thread(target=execute, name="blocked-delete-client")
        execute_thread.start()
        later_result: queue.Queue[dict[str, object]] = queue.Queue()
        later_completed = threading.Event()

        def later_writer() -> None:
            client = DaemonClient(stack.socket_path, timeout_s=5)
            response = client.operation(
                "mutation.session.delete.preview",
                {"session_ids": [session_ids[-1]]},
                archive_root=str(stack.archive_root),
                request_id="later-preview",
            )
            assert response is not None
            later_result.put(response)
            later_completed.set()

        later_thread: threading.Thread | None = None
        try:
            assert entered_apply.wait(timeout=2)
            accepted = execute_result.get(timeout=2)
            assert accepted["outcome"] == "accepted"

            status = stack.client.operation(
                "operation.status",
                {"request_id": execute_request_id},
                archive_root=str(stack.archive_root),
            )
            assert status is not None
            assert status["result"]["outcome"] in {"accepted", "running"}
            from time import monotonic

            started = monotonic()
            deadline_at = started + 0.025
            after_sequence = status["result"]["sequence"]
            after_progress_sequence = status["result"].get("progress_sequence", 0)
            while True:
                waited = stack.client.operation(
                    "operation.await",
                    {
                        "request_id": execute_request_id,
                        "after_sequence": after_sequence,
                        "after_progress_sequence": after_progress_sequence,
                        "timeout_ms": 2_000,
                    },
                    archive_root=str(stack.archive_root),
                    deadline_ms=max(1, int((deadline_at - monotonic()) * 1000)),
                )
                assert waited is not None, waited
                assert waited["outcome"] == "completed", waited
                # Poll expiry returns its actual accepted lifecycle. Earlier
                # progress is consumed only within the original wait budget.
                state = waited["result"]
                assert state["outcome"] in {"accepted", "running"}, waited
                assert state["reference"] == accepted["accepted_reference"], waited
                if monotonic() >= deadline_at:
                    after_sequence = state["sequence"]
                    after_progress_sequence = state.get("progress_sequence", after_progress_sequence)
                    break
                assert (state["sequence"], state.get("progress_sequence", after_progress_sequence)) != (
                    after_sequence,
                    after_progress_sequence,
                ), waited
                after_sequence = state["sequence"]
                after_progress_sequence = state.get("progress_sequence", after_progress_sequence)
            assert monotonic() - started < 1.0
            assert not release_apply.is_set()

            from polylogue.archive.query.execution_control import QueryExecutionContext
            from polylogue.operations.daemon_protocol import DaemonOperationRequest
            from polylogue.operations.mutation_transaction import MutationPrincipal
            from polylogue.operations.operation_context import OperationControlResult

            waiter_entered, waiter_released = threading.Event(), threading.Event()
            control = stack.runtime.control

            def observe_waiter(
                request: DaemonOperationRequest,
                principal: MutationPrincipal,
                archive_identity: str,
                *,
                execution_context: QueryExecutionContext | None = None,
            ) -> OperationControlResult:
                observing = request.request_id == "disconnected-await"
                if observing:
                    waiter_entered.set()
                try:
                    return control(request, principal, archive_identity, execution_context=execution_context)
                finally:
                    if observing:
                        waiter_released.set()

            monkeypatch.setattr(stack.runtime, "control", observe_waiter)
            body = json.dumps(
                DaemonOperationRequest(
                    "operation.await",
                    {
                        "request_id": execute_request_id,
                        "after_sequence": after_sequence,
                        "after_progress_sequence": after_progress_sequence,
                        "timeout_ms": 30_000,
                    },
                    request_id="disconnected-await",
                ).to_dict()
            ).encode()
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as peer:
                peer.connect(str(stack.socket_path))
                peer.sendall(
                    (
                        "POST /api/operation HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\n"
                        f"Content-Length: {len(body)}\r\n\r\n"
                    ).encode()
                    + body
                )
                assert waiter_entered.wait(timeout=2)
            assert waiter_released.wait(timeout=2), "disconnected await retained its handler until the 30s deadline"
            assert not release_apply.is_set()
            cancellation = stack.client.cancel(execute_request_id, archive_root=str(stack.archive_root))
            assert cancellation is not None

            later_thread = threading.Thread(target=later_writer, name="later-delete-preview")
            later_thread.start()
            assert later_result.empty()
            assert not later_completed.wait(timeout=0.15)
        finally:
            release_apply.set()
            execute_thread.join(timeout=3)
            if later_thread is not None:
                later_thread.join(timeout=3)

        assert not execute_thread.is_alive()
        assert later_thread is not None
        assert not later_thread.is_alive()
        final = stack.client.await_operation(execute_request_id, archive_root=str(stack.archive_root), timeout_ms=2_000)
        assert final is not None
        state = final["result"]
        assert state["outcome"] == "cancelled", state
        assert state["completed_chunks"] == 1
        assert state["not_attempted"] == [1]
        assert state["stop_reason"] == "cancelled"
        assert state["reference"] == accepted["accepted_reference"]
        assert all(not stack.session_exists(session_id) for session_id in session_ids[:-1])
        assert stack.session_exists(session_ids[-1])

    with running_daemon_operations(tmp_path / "archive") as restarted:
        replay = restarted.client.operation(
            "mutation.session.delete.execute",
            {"authorization_request_id": authorization["result"]["reference"]["request_id"]},
            archive_root=str(restarted.archive_root),
            request_id=execute_request_id,
        )
        assert replay is not None and replay["outcome"] == "cancelled"
        assert replay["accepted_reference"] == accepted["accepted_reference"]
        assert replay["result"]["not_attempted"] == [1]
        assert restarted.session_exists(session_ids[-1])


@pytest.mark.parametrize(
    ("case", "expected_status", "expected_code"),
    [
        ("partial", 400, "invalid_request"),
        ("malformed", 400, "invalid_request"),
        ("duplicate-json", 400, "invalid_request"),
        ("nested-json", 400, "invalid_request"),
        ("nonfinite", 400, "invalid_request"),
        ("wrong-protocol", 400, "invalid_request"),
        ("operation-bound", 400, "invalid_request"),
        ("whitespace-bound", 413, "request_too_large"),
        ("transport-bound", 413, "request_too_large"),
        ("duplicate-length", 400, "invalid_framing"),
        ("transfer-encoding", 400, "invalid_framing"),
        ("wrong-type", 415, "unsupported_media_type"),
        ("wrong-method", 405, "method_not_allowed"),
        ("unknown-method", 501, "invalid_http_request"),
        ("wrong-version", 505, "unsupported_http_version"),
        ("browser-route", 404, "operation_endpoint_required"),
    ],
)
def test_machine_ingress_faults_refuse_without_dispatch_and_release_connection(
    tmp_path: Path,
    case: str,
    expected_status: int,
    expected_code: str,
) -> None:
    """Removing an ingress guard admits invalid input or leaves no typed refusal."""
    from http.client import HTTPResponse

    from polylogue.operations.daemon_protocol import DAEMON_OPERATION_PROTOCOL, MAX_DECLARED_OPERATION_BODY_BYTES

    errors: queue.SimpleQueue[str] = queue.SimpleQueue()
    with running_daemon_operations(tmp_path / "archive", server_error_sink=errors) as stack:
        body = json.dumps(
            {
                "protocol": DAEMON_OPERATION_PROTOCOL,
                "operation": "status",
                "request_id": "ingress-law",
                "payload": {},
            }
        ).encode()
        method, path, version = "POST", "/api/operation", "HTTP/1.1"
        content_type = "application/json"
        extra = ""
        declared_length = len(body)
        if case == "partial":
            declared_length += 1
        elif case == "malformed":
            body = b"{"
        elif case == "duplicate-json":
            body = body[:-1] + b',"operation":"status"}'
        elif case == "nested-json":
            body = b"[" * 2000 + b"0" + b"]" * 2000
        elif case == "nonfinite":
            body = body.replace(b'"payload": {}', b'"payload": {"value": NaN}')
        elif case == "wrong-protocol":
            body = body.replace(DAEMON_OPERATION_PROTOCOL.encode(), b"unknown/v99")
        elif case == "operation-bound":
            body = json.dumps(
                {
                    "protocol": DAEMON_OPERATION_PROTOCOL,
                    "operation": "completion",
                    "request_id": "ingress-law",
                    "payload": {"incomplete": "x" * (65 * 1024)},
                }
            ).encode()
        elif case == "transport-bound":
            declared_length = MAX_DECLARED_OPERATION_BODY_BYTES + 1
        elif case == "whitespace-bound":
            body += b" " * (65 * 1024)
        elif case == "duplicate-length":
            extra = f"Content-Length: {len(body)}\r\n"
        elif case == "transfer-encoding":
            extra = "Transfer-Encoding: chunked\r\n"
        elif case == "wrong-type":
            content_type = "text/plain"
        elif case == "wrong-method":
            method = "GET"
        elif case == "unknown-method":
            method = "TRACE"
        elif case == "wrong-version":
            version = "HTTP/1.0"
        elif case == "browser-route":
            path = "/api/health"
        if case not in {"partial", "transport-bound"}:
            declared_length = len(body)
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as peer:
            peer.settimeout(2)
            peer.connect(str(stack.socket_path))
            peer.sendall(
                (
                    f"{method} {path} {version}\r\nHost: localhost\r\nContent-Type: {content_type}\r\n"
                    f"Content-Length: {declared_length}\r\n{extra}\r\n"
                ).encode()
                + body
            )
            if case == "partial":
                peer.shutdown(socket.SHUT_WR)
            with HTTPResponse(peer) as response:
                response.begin()
                payload = json.loads(response.read())
                assert response.status == expected_status
                assert payload["outcome"] == "rejected"
                assert payload["error"]["code"] == expected_code
        assert not stack.runtime._exchanges
        assert errors.empty()
        recovered = stack.client.operation("status", {})
        assert recovered is not None and recovered["outcome"] == "completed"


@pytest.mark.parametrize("field", ["generation", "schemas", "archive", "served-by", "timing", "degraded", "fallback"])
def test_client_refuses_incoherent_authority_from_a_real_operation(
    tmp_path: Path,
    field: str,
) -> None:
    """Each mutant changes one copy of the observed authority without its peer."""
    from polylogue.daemon_client import DaemonOperationProtocolError
    from polylogue.operations.daemon_protocol import DaemonOperationRequest

    with running_daemon_operations(tmp_path / "archive") as stack:
        request = DaemonOperationRequest("status", {}, request_id="authority-law")
        response = stack.client.operation("status", {}, request_id="authority-law")
        assert response is not None
        changed = deepcopy(response)
        if field == "generation":
            changed["generation"]["id"] = "stale-generation"
        elif field == "schemas":
            changed["schema_versions"]["source"] += 1
        elif field == "archive":
            changed["archive"]["archive_identity"] = "other-archive"
        elif field == "served-by":
            changed["served_by"]["identity"] = "other-server"
        elif field == "timing":
            changed["authority_snapshot"]["queue_ms"] += 1
        elif field == "degraded":
            changed["readiness"]["degraded_components"] = ["missing-source"]
        elif field == "fallback":
            changed["authority"]["fallback"] = "local"
        with pytest.raises(DaemonOperationProtocolError, match="incoherent"):
            DaemonClient._validate_operation_response(request, 200, changed)


def test_control_result_metadata_comes_from_the_durable_receipt_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Using admission-time metadata after a recovery read returns obsolete schema evidence."""
    from dataclasses import replace

    import polylogue.operations.daemon_execution as execution
    from polylogue.operations.operation_context import OperationControlRead, observe_control_authority

    ids: tuple[str, ...] = ()

    def seed(root: Path) -> None:
        nonlocal ids
        ids = _seed_sessions(root, count=1)

    with running_daemon_operations(tmp_path / "archive", seed_archive=seed) as stack:
        accepted = stack.client.operation_to_completion(
            "mutation.session.delete.preview",
            {"session_ids": list(ids)},
            archive_root=str(stack.archive_root),
            request_id="receipt-authority",
        )
        assert accepted is not None and accepted["outcome"] == "completed"

        def earlier_observation(root: Path) -> OperationControlRead:
            snapshot = observe_control_authority(root)
            return replace(
                snapshot, schema_versions={**snapshot.schema_versions, "source": snapshot.schema_versions["source"] - 1}
            )

        monkeypatch.setattr(execution, "observe_control_authority", earlier_observation)
        recovered = stack.client.operation(
            "operation.await",
            {"request_id": "receipt-authority", "after_sequence": 0, "timeout_ms": 100},
            archive_root=str(stack.archive_root),
        )
        assert recovered is not None and recovered["outcome"] == "completed"
        assert recovered["generation"]["id"] == accepted["generation"]["id"]
        assert recovered["schema_versions"] == {tier: accepted["schema_versions"][tier] for tier in ("source", "audit")}
        assert recovered["result"]["reference"] == accepted["accepted_reference"]


def test_expired_await_reads_the_actual_accepted_receipt_and_preserves_refusals(tmp_path: Path) -> None:
    """Skipping the lifecycle read on poll expiry loses a real accepted receipt."""
    from time import monotonic

    from polylogue.archive.query.execution_control import QueryCancelledError, QueryExecutionContext
    from polylogue.operations.daemon_protocol import DaemonOperationRequest

    ids: tuple[str, ...] = ()

    def seed(root: Path) -> None:
        nonlocal ids
        ids = _seed_sessions(root, count=1)

    with running_daemon_operations(tmp_path / "archive", seed_archive=seed) as stack:
        accepted = stack.client.operation_to_completion(
            "mutation.session.delete.preview",
            {"session_ids": list(ids)},
            archive_root=str(stack.archive_root),
            request_id="expired-poll-receipt",
        )
        assert accepted is not None and accepted["outcome"] == "completed", accepted
        reference = accepted["accepted_reference"]
        assert isinstance(reference, dict)
        principal = _all_capabilities_principal()
        request = DaemonOperationRequest(
            "operation.await",
            {"request_id": "expired-poll-receipt", "timeout_ms": 1},
            request_id="expired-control-poll",
            deadline_ms=1,
        )
        # Model a poll whose budget was spent before its handler ran; the
        # accepted lifecycle is real Audit data, not a patched receipt.
        recovered = stack.runtime.call(request, principal, started_at=monotonic() - 1)
        assert recovered["outcome"] == "completed", recovered
        recovered_result = recovered["result"]
        assert isinstance(recovered_result, dict)
        assert recovered_result["outcome"] == "completed", recovered
        assert recovered_result["reference"] == reference
        accepted_versions = accepted["schema_versions"]
        assert isinstance(accepted_versions, dict)
        assert recovered["schema_versions"] == {tier: accepted_versions[tier] for tier in ("source", "audit")}
        for target, peer in (
            ("unknown-expired-poll", principal),
            ("expired-poll-receipt", replace(principal, actor_ref="synthetic-unrelated")),
        ):
            refused = stack.runtime.call(
                replace(request, payload={**request.payload, "request_id": target}),
                peer,
                started_at=monotonic() - 1,
            )
            assert refused["outcome"] == "rejected", refused
            refused_error = refused["error"]
            assert isinstance(refused_error, dict)
            assert refused_error["code"] == "operation_reference_unknown", refused
        stale = stack.runtime.call(
            replace(request, expected_archive_identity="synthetic-other-archive"),
            principal,
            started_at=monotonic() - 1,
        )
        stale_error = stale["error"]
        assert isinstance(stale_error, dict)
        assert stale["outcome"] == "rejected" and stale_error["code"] == "archive_identity_stale", stale
        cancelled = QueryExecutionContext(
            call_id="disconnected-expired-poll", query_ref=request.fingerprint, deadline_monotonic=monotonic() - 1
        )
        cancelled.cancel()
        with pytest.raises(QueryCancelledError):
            stack.runtime.control(request, principal, reference["archive_identity"], execution_context=cancelled)
        for operation in ("operation.status", "operation.cancel"):
            expired = stack.runtime.call(
                replace(request, operation=operation, payload={"request_id": "expired-poll-receipt"}),
                principal,
                started_at=monotonic() - 1,
            )
            assert expired["outcome"] == "timed-out", expired
            expired_error = expired["error"]
            assert isinstance(expired_error, dict)
            assert expired_error["code"] == "QueryTimeoutError", expired
        assert stack.session_exists(ids[0])


@pytest.mark.parametrize(
    "operation,payload",
    [
        ("mutation.session.delete.preview", {"session_ids": ["codex:absent"]}),
        ("query.aggregate", {"mode": "count"}),
    ],
)
def test_cancelled_queued_operation_reports_cancelled_not_failed(
    tmp_path: Path, operation: str, payload: dict[str, object]
) -> None:
    """A pre-acceptance cancellation is a cancellation, not an operation failure.

    The scheduler settles an unstarted read with ``DaemonOperationCancelled``.
    A staged preview cancels its proxy Future only after its task and cleanup
    settle. Both must report cancellation before acceptance.

    Anti-vacuity: the operation is genuinely queued behind saturated workers
    (its exchange exists and its future is not done before the cancel), so the
    pre-acceptance path is the one that fires. Removing the typed cancellation
    branches from
    ``DaemonOperationRuntime.call`` sends it to the generic handler, which
    reports ``outcome == "failed"`` with
    a cancellation error, and both assertions go red.
    """
    from polylogue.operations.daemon_protocol import DAEMON_OPERATION_SPECS, DaemonOperationRequest
    from polylogue.operations.mutation_transaction import MutationPrincipal

    entered = threading.Semaphore(0)
    release = threading.Event()

    def block_worker() -> None:
        entered.release()
        # Generous: this worker must stay parked for the whole exchange
        # (archive bootstrap, audit reads, the cancel and its envelope), and
        # its early exit would let the queued task start and void the test.
        assert release.wait(timeout=60)

    request_id = "cancelled-queued-operation"
    principal = MutationPrincipal(
        actor_ref=f"daemon:unix:uid:{os.getuid()}",
        capabilities=frozenset(spec.capability for spec in DAEMON_OPERATION_SPECS),
        surface="cli",
        role_label="daemon-unix-peer",
    )

    with running_daemon_operations(tmp_path / "archive") as stack:
        blockers = [stack.execution_kernel.submit(block_worker) for _ in range(2)]
        assert all(entered.acquire(timeout=2) for _ in blockers)

        request = DaemonOperationRequest(
            operation,
            payload,
            request_id=request_id,
            archive_root=str(stack.archive_root),
        )
        envelopes: list[dict[str, object]] = []

        def call_runtime() -> None:
            envelopes.append(stack.runtime.call(request, principal))

        caller = threading.Thread(target=call_runtime, name="cancelled-queued-control-caller", daemon=True)
        caller.start()
        try:
            with stack.runtime._condition:
                assert stack.runtime._condition.wait_for(lambda: request_id in stack.runtime._exchanges, timeout=5)
                exchange = stack.runtime._exchanges[request_id]
                assert stack.runtime._condition.wait_for(lambda: exchange.future is not None, timeout=5)
            assert exchange.future is not None
            assert not exchange.future.done(), "the operation must still be queued or this test is vacuous"
            assert not exchange.acceptance_started
            request_bytes = len(json.dumps(request.to_dict(), separators=(",", ":"), allow_nan=False).encode())
            assert stack.execution_kernel.snapshot().used_bytes == request_bytes, (
                stack.execution_kernel.snapshot(),
                request_bytes,
            )
            if operation == "query.aggregate":
                assert exchange.deadline is None
                assert exchange.context.read_control is not None
                assert exchange.context.read_control.deadline_monotonic is None

            # Request cancellation on the actual owner while keeping the caller
            # connected until its future physically settles. A peer disconnect
            # may truthfully return before that settlement boundary.
            exchange.cancellation.cancel()
            caller.join(timeout=5)
            assert not caller.is_alive()
            assert stack.execution_kernel.snapshot().used_bytes == 0
        finally:
            release.set()
            for blocker in blockers:
                blocker.future.result(timeout=10)

    assert len(envelopes) == 1
    envelope = envelopes[0]
    assert envelope["outcome"] == "cancelled", envelope
    assert envelope.get("error") is None


def test_expired_staged_ingest_releases_its_queued_compute_reservation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A timed-out staged operation must not retain a scheduler unit until dispatch.

    The request enters the real daemon operation runtime and stages its first
    ingest read on the shared bounded kernel. Two live kernel workers keep that
    phase queued until the request deadline expires.

    Anti-vacuity: without forwarding the exchange cancellation handle through
    ``compute_phase``, the request reports ``timed-out`` while its queued
    phase still owns its scheduler unit. The control-class completion counter
    stays at zero before either blocking worker is released.
    """
    from polylogue.operations.daemon_protocol import DAEMON_OPERATION_SPECS, DaemonOperationRequest
    from polylogue.operations.mutation_transaction import MutationPrincipal

    entered = threading.Semaphore(0)
    release = threading.Event()

    def block_worker() -> None:
        entered.release()
        assert release.wait(timeout=60)

    principal = MutationPrincipal(
        actor_ref=f"daemon:unix:uid:{os.getuid()}",
        capabilities=frozenset(spec.capability for spec in DAEMON_OPERATION_SPECS),
        surface="cli",
        role_label="daemon-unix-peer",
    )
    request = DaemonOperationRequest(
        "ingest",
        {"path": str(tmp_path / "unreached.json")},
        request_id="expired-staged-ingest",
        archive_root=str(tmp_path / "archive"),
        deadline_ms=1_000,
    )
    envelopes: list[dict[str, object]] = []

    with running_daemon_operations(tmp_path / "archive") as stack:
        # The test exercises deadline propagation before ingest's later
        # session-maintenance phase, which is not part of this queue seam.
        monkeypatch.setattr(stack.runtime, "require_session_maintenance", lambda: None)
        # Startup recovery already completed one control-class phase on this
        # kernel (777ab745c9); only this request's dispatch is measured.
        startup_control_completions = stack.execution_kernel.snapshot().by_class("control").completed
        blockers = [stack.execution_kernel.submit(block_worker) for _ in range(2)]
        assert all(entered.acquire(timeout=2) for _ in blockers)

        caller = threading.Thread(
            target=lambda: envelopes.append(stack.runtime.call(request, principal)),
            name="expired-staged-ingest-caller",
            daemon=True,
        )
        caller.start()
        try:
            for _ in range(200):
                if stack.execution_kernel.snapshot().queued_units == 1:
                    break
                threading.Event().wait(0.01)
            assert stack.execution_kernel.snapshot().queued_units == 1
            caller.join(timeout=2)
            assert not caller.is_alive()
            assert len(envelopes) == 1
            assert envelopes[0]["outcome"] == "timed-out"
            # The operation's ``finally`` discards staged publications without
            # an admission (a full control class would refuse the cleanup and
            # leak the staging), so no control reservation remains: the
            # expired phase must have released its own without becoming a
            # completed dispatch.
            control = stack.execution_kernel.snapshot().by_class("control")
            assert control.used_units == control.queued_units == 0
            assert control.completed == startup_control_completions
        finally:
            release.set()
            for blocker in blockers:
                blocker.future.result(timeout=10)
            with stack.runtime._condition:
                assert stack.runtime._condition.wait_for(lambda: not stack.runtime._exchanges, timeout=5)


def test_skewed_write_refusal_is_pre_dispatch_not_an_indeterminate_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pre-dispatch refusal the client cannot self-detect must not become receipt recovery.

    A newer CLI against a resident older daemon sends a body its own spec table
    admits and the daemon's does not; the daemon refuses at the ingress with
    413 ``request_too_large`` before dispatching anything. That refusal carries
    no ``operation`` or ``request_id`` to correlate against, so only the
    envelope's ``pre_dispatch`` marker tells the client the actuator provably
    never ran. Only the daemon's spec lookup is skewed here -- the client, the
    socket, and the whole ingress path are the production ones.

    Anti-vacuity: drop ``pre_dispatch`` from ``MachineOperationHandler._reject``
    (or restore the three-code refusal whitelist in ``DaemonClient.operation``)
    and this write raises ``DaemonMutationIndeterminateError`` instead, steering
    the operator into recovery adjudication for a mutation that never started.
    """
    import polylogue.daemon.uds as uds_module
    from polylogue.operations.daemon_protocol import DaemonOperationSpec, daemon_operation_spec

    skewed = "mutation.session.mark"

    def _older_daemon_spec(name: str) -> DaemonOperationSpec | None:
        spec = daemon_operation_spec(name)
        if spec is not None and name == skewed:
            return replace(spec, max_body_bytes=256)
        return spec

    monkeypatch.setattr(uds_module, "daemon_operation_spec", _older_daemon_spec)

    session_ids = [f"claude-code-session:{index:040d}" for index in range(8)]
    payload: dict[str, object] = {"session_ids": session_ids, "add_marks": ["star"]}
    client_spec = daemon_operation_spec(skewed)
    assert client_spec is not None
    # The client's own bound admits this body; only the resident daemon refuses.
    assert 256 < len(json.dumps(payload).encode()) <= client_spec.max_body_bytes

    with running_daemon_operations(tmp_path / "archive") as stack:
        with pytest.raises(DaemonOperationRejectedError) as rejected:
            stack.client.operation(skewed, payload, archive_root=str(stack.archive_root))
        assert rejected.value.outcome == "request_too_large"
        # The refusal's public detail survives to the caller, not only its code.
        assert rejected.value.detail != "request_too_large"
        assert not stack.runtime._exchanges


def _all_capabilities_principal() -> Any:
    from polylogue.operations.daemon_protocol import DAEMON_OPERATION_SPECS
    from polylogue.operations.mutation_transaction import MutationPrincipal

    return MutationPrincipal(
        actor_ref=f"daemon:unix:uid:{os.getuid()}",
        capabilities=frozenset(spec.capability for spec in DAEMON_OPERATION_SPECS),
        surface="cli",
        role_label="daemon-unix-peer",
    )


def test_annotation_import_that_outlives_its_deadline_never_commits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A request reported ``timed-out`` before acceptance commits nothing afterwards.

    Validation resolves every ref before the import writes. Holding the first
    ref resolution past the request deadline makes the runtime answer
    ``timed-out``; the worker then finishes validating.

    Anti-vacuity: remove the ``before_durable_execution`` fence from
    ``mutation_annotation_import_batch`` and the worker commits the batch into
    ``user.db`` after its caller was told it timed out before acceptance.
    """
    from polylogue.archive.message.roles import Role
    from polylogue.core.enums import Provider
    from polylogue.operations import ref_resolution
    from polylogue.operations.daemon_protocol import DaemonOperationRequest
    from polylogue.sources.parsers.base import ParsedMessage, ParsedSession
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from tests.infra.live_ingest import write_index_session

    def seed(archive_root: Path) -> None:
        with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
            write_index_session(
                archive,
                ParsedSession(
                    source_name=Provider.CODEX,
                    provider_session_id="annotation-target",
                    messages=[ParsedMessage(provider_message_id="m1", role=Role.USER, text="evidence")],
                ),
            )

    session_id = "codex-session:annotation-target"
    entered = threading.Event()
    answered = threading.Event()
    original = ref_resolution.resolve_ref_against_archive

    def held_resolution(*args: Any, **kwargs: Any) -> Any:
        entered.set()
        assert answered.wait(timeout=30)
        return original(*args, **kwargs)

    monkeypatch.setattr(ref_resolution, "resolve_ref_against_archive", held_resolution)

    with running_daemon_operations(tmp_path / "archive", seed_archive=seed) as stack:
        request = DaemonOperationRequest(
            "mutation.annotation.import_batch",
            {
                "jsonl": json.dumps(
                    {
                        "row_key": "r1",
                        "value": {"activity": "debugging", "confidence": 0.9},
                        "evidence_refs": [session_id],
                    }
                )
                + "\n",
                "batch_id": "late-batch",
                "schema_id": "seed.activity",
                "schema_version": 2,
                "target_ref": f"session:{session_id}",
                "source_result_ref": "result-set:late",
                "actor_ref": "agent:labeler",
                "model_ref": "agent:model",
                "prompt_ref": "block:prompt:0",
                "metadata": {},
            },
            request_id="late-annotation-import",
            archive_root=str(stack.archive_root),
            deadline_ms=1_000,
        )
        try:
            envelope = stack.runtime.call(request, _all_capabilities_principal())
        finally:
            answered.set()
        assert entered.is_set(), "validation never reached ref resolution; the test is vacuous"
        assert envelope["outcome"] == "timed-out"
        with stack.runtime._condition:
            assert stack.runtime._condition.wait_for(lambda: not stack.runtime._exchanges, timeout=30)
        archive_root = stack.archive_root

    with closing(sqlite3.connect(archive_root / "user.db")) as conn:
        assert conn.execute("SELECT COUNT(*) FROM assertions").fetchone() == (0,)


def test_shutdown_waits_for_a_cancelled_staged_operation_to_finish_its_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Shutdown returns only after a cancelled staged task has actually finished.

    The staged ingest's first compute phase is running when shutdown cancels
    the pre-acceptance exchange. The task keeps waiting for that phase before
    its cleanup runs, and the owner may only tear down after that.

    Anti-vacuity: cancel the ``run_coroutine_threadsafe`` proxy future instead
    of the task and shutdown returns while the compute phase is still blocked,
    so ``returned`` is unset when it completes.
    """
    import asyncio

    from polylogue.operations.daemon_protocol import DaemonOperationRequest

    entered = threading.Event()
    release = threading.Event()
    returned = threading.Event()
    request = DaemonOperationRequest(
        "ingest",
        {"path": str(tmp_path / "unreached.json")},
        request_id="shutdown-staged-ingest",
        archive_root=str(tmp_path / "archive"),
        deadline_ms=60_000,
    )

    with running_daemon_operations(tmp_path / "archive") as stack:
        monkeypatch.setattr(stack.runtime, "require_session_maintenance", lambda: None)
        original = stack.runtime.compute_phase

        async def held_first_phase(work: Any) -> Any:
            if entered.is_set():
                return await original(work)

            def held() -> Any:
                entered.set()
                assert release.wait(timeout=30)
                try:
                    return work()
                finally:
                    returned.set()

            return await original(held)

        monkeypatch.setattr(stack.runtime, "compute_phase", held_first_phase)
        caller = threading.Thread(
            target=lambda: stack.runtime.call(request, _all_capabilities_principal()),
            name="shutdown-staged-ingest-caller",
            daemon=True,
        )
        caller.start()
        releaser = threading.Timer(0.5, release.set)
        try:
            assert entered.wait(timeout=10)
            loop = stack._loop.loop
            assert loop is not None
            shutdown = asyncio.run_coroutine_threadsafe(stack.runtime.shutdown(), loop)
            releaser.start()
            shutdown.result(timeout=30)
            assert returned.is_set(), "shutdown returned while the staged compute phase was still running"
        finally:
            release.set()
            releaser.cancel()
            caller.join(timeout=10)


def test_verified_backup_restore_crosses_the_real_machine_operation_route(tmp_path: Path) -> None:
    """Dropping registry dispatch or fresh destination authority breaks this route."""
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from polylogue.storage.sqlite.population_admission import POPULATION_PENDING
    from tests.infra.workload_artifacts import _archive_files

    destination = tmp_path / "restored"
    with running_daemon_operations(tmp_path / "archive") as stack:
        backup = stack.client.operation(
            "maintenance.backup",
            {"output_dir": str(tmp_path / "packages"), "verify": True, "profile": "full_evidence"},
            archive_root=str(stack.archive_root),
        )
        assert backup is not None and backup["outcome"] == "completed", backup
        package = backup["result"]["result"]["output_path"]
        package_path = Path(package)
        package_before = (_archive_files(package_path), (package_path / "manifest.json").read_bytes())
        restored = stack.client.operation(
            "maintenance.restore_verified_backup",
            {"backup_dir": package, "destination": str(destination)},
            archive_root=str(stack.archive_root),
        )
        assert (_archive_files(package_path), (package_path / "manifest.json").read_bytes()) == package_before
        assert not any(package_path.glob("*.db-wal"))
        assert not any(package_path.glob("*.db-shm"))
    assert restored is not None and restored["outcome"] == "completed", restored
    assert restored["result"]["result"]["operational_admission"] == "ready"
    assert not (destination / POPULATION_PENDING).exists()
    with ArchiveStore.open_existing(destination, read_only=True):
        pass


@pytest.mark.parametrize("fault_kind", ["permission", "wrapped_permission", "wrapped_busy"])
def test_restore_machine_operation_preserves_retryable_io_fault_and_pending_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault_kind: str
) -> None:
    import sqlite3

    from polylogue.storage.sqlite import archive_population
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from polylogue.storage.sqlite.migration_runner import MigrationError
    from polylogue.storage.sqlite.population_admission import POPULATION_PENDING, ArchivePopulationPendingError

    def fault(*_args: object, **_kwargs: object) -> object:
        if fault_kind == "wrapped_busy":
            error = sqlite3.OperationalError("synthetic reader contention")
            error.sqlite_errorcode = sqlite3.SQLITE_BUSY
            raise MigrationError("migration evidence unavailable") from error
        access_error = PermissionError("synthetic evidence access fault")
        if fault_kind == "wrapped_permission":
            raise MigrationError("migration evidence unavailable") from access_error
        raise access_error

    destination = tmp_path / "pending-restoration"
    with running_daemon_operations(tmp_path / "archive") as stack:
        backup = stack.client.operation(
            "maintenance.backup",
            {"output_dir": str(tmp_path / "packages"), "verify": True, "profile": "full_evidence"},
            archive_root=str(stack.archive_root),
        )
        assert backup is not None and backup["outcome"] == "completed", backup
        monkeypatch.setattr(archive_population, "_populate_authenticated_archive", fault)
        restored = stack.client.operation(
            "maintenance.restore_verified_backup",
            {"backup_dir": backup["result"]["result"]["output_path"], "destination": str(destination)},
            archive_root=str(stack.archive_root),
        )
    assert restored is not None and restored["outcome"] == "failed", restored
    assert restored["error"]["code"] == "restore_io_fault"
    assert restored["error"]["retryable"] is True
    assert restored["error"]["retained_pending_destination"] == str(destination)
    assert (destination / POPULATION_PENDING).is_file()
    with pytest.raises(ArchivePopulationPendingError):
        ArchiveStore.open_existing(destination)


@pytest.mark.parametrize("audit_read_gap,spill_failure", [(False, False), (True, False), (True, True)])
def test_accepted_restore_outlives_implicit_deadline_and_control_returns_terminal_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, audit_read_gap: bool, spill_failure: bool
) -> None:
    from time import monotonic

    from polylogue.daemon import operation_runtime
    from polylogue.operations import archive_backup
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    entered, release = threading.Event(), threading.Event()
    restore = archive_backup.restore_verified_backup
    offset = 0.0
    responses: queue.Queue[Any] = queue.Queue()
    destination = tmp_path / "restored"

    def blocked_restore(**kwargs: Any) -> Any:
        entered.set()
        release.wait()
        return restore(**kwargs)

    monkeypatch.setattr(archive_backup, "restore_verified_backup", blocked_restore)
    monkeypatch.setattr(operation_runtime, "monotonic", lambda: monotonic() + offset)
    with running_daemon_operations(tmp_path / "archive") as stack:
        backup = stack.client.operation(
            "maintenance.backup",
            {"output_dir": str(tmp_path / "packages"), "verify": True, "profile": "full_evidence"},
            archive_root=str(stack.archive_root),
        )
        assert backup is not None and backup["outcome"] == "completed", backup
        request_id = "slow-accepted-restore"

        def submit() -> None:
            try:
                responses.put(
                    stack.client.operation(
                        "maintenance.restore_verified_backup",
                        {"backup_dir": backup["result"]["result"]["output_path"], "destination": str(destination)},
                        archive_root=str(stack.archive_root),
                        request_id=request_id,
                    )
                )
            except BaseException as exc:
                responses.put(exc)

        thread = threading.Thread(target=submit)
        thread.start()
        try:
            assert entered.wait(timeout=5)
            offset = 301.0
            with stack.runtime._condition:
                stack.runtime._condition.notify_all()
            response = responses.get(timeout=5)
            assert not isinstance(response, BaseException)
            assert response["outcome"] == "indeterminate"
            assert not release.is_set()
            if audit_read_gap:
                from polylogue.storage.sqlite.audit_continuity import AuditContinuityError

                def unavailable_control_read(*args: Any, **kwargs: Any) -> Any:
                    raise AuditContinuityError("synthetic temporary control read gap")

                monkeypatch.setattr(operation_runtime, "open_operation_control", unavailable_control_read)
            if spill_failure:

                def refuse_terminal_transfer(exchange: Any) -> None:
                    raise OSError("synthetic result publication refusal")

                monkeypatch.setattr(stack.runtime, "_retain_terminal", refuse_terminal_transfer)
            release.set()
            terminal = stack.client.await_operation(request_id, archive_root=str(stack.archive_root))
            while terminal is not None and terminal["result"]["outcome"] in {"accepted", "running", "indeterminate"}:
                state = terminal["result"]
                terminal = stack.client.await_operation(
                    request_id,
                    archive_root=str(stack.archive_root),
                    after_sequence=state["sequence"],
                    after_progress_sequence=state.get("progress_sequence", 0),
                )
            assert terminal is not None and terminal["result"]["outcome"] == "completed", terminal
            assert terminal["result"]["result"]["operational_admission"] == "ready"
            status = stack.client.operation(
                "operation.status", {"request_id": request_id}, archive_root=str(stack.archive_root)
            )
            assert status is not None and status["result"] == terminal["result"]
            if spill_failure:
                assert terminal["result"]["terminal_custody_error"] == "OSError"
                assert request_id in stack.runtime._exchanges
                blocked_destination = tmp_path / "blocked-result-custody"
                blocked = stack.client.operation(
                    "maintenance.restore_verified_backup",
                    {"backup_dir": backup["result"]["result"]["output_path"], "destination": str(blocked_destination)},
                    archive_root=str(stack.archive_root),
                )
                assert blocked is not None and blocked["outcome"] == "rejected", blocked
                assert blocked["error"]["code"] == "operation_result_custody_unavailable"
                assert not blocked_destination.exists()
                scratch = None
            else:
                assert request_id not in stack.runtime._exchanges
                assert stack.runtime._terminal_scratch is not None
                scratch = Path(stack.runtime._terminal_scratch.name)
                # The earlier backup also owns a terminal result. Assert this
                # restore's exact custody packet, not whole-runtime cardinality.
                packet_path = stack.runtime._terminal_path(request_id)
                assert packet_path is not None and packet_path.is_file()
                packet = json.loads(packet_path.read_text())
                assert packet["request_id"] == request_id
                from polylogue.operations.daemon_protocol import DaemonOperationRequest

                intent = DaemonOperationRequest(
                    "maintenance.restore_verified_backup",
                    {"backup_dir": backup["result"]["result"]["output_path"], "destination": str(destination)},
                    archive_root=str(stack.archive_root),
                    request_id=request_id,
                )
                assert packet["fingerprint"] == intent.fingerprint
                assert packet["archive_identity"] == response["archive"]["archive_identity"]
                assert packet["envelope"]["request_id"] == request_id
                assert packet["envelope"]["result"]["result"] == terminal["result"]["result"]
            from polylogue.operations.daemon_protocol import DaemonOperationRequest
            from polylogue.operations.mutation_transaction import MutationPrincipal

            if spill_failure:
                held = stack.runtime._exchanges[request_id]
                principal = held.context.principal
                assert held.snapshot is not None
                archive_identity = held.snapshot.identity.authority_identity_digest
            else:
                assert scratch is not None
                packet_path = stack.runtime._terminal_path(request_id)
                assert packet_path is not None
                with packet_path.open(encoding="utf-8") as stream:
                    packet = json.load(stream)
                declared = packet["principal"]
                principal = MutationPrincipal(
                    actor_ref=declared["actor_ref"],
                    capabilities=frozenset(declared["capabilities"]),
                    surface=declared["surface"],
                    role_label=declared["role_label"],
                )
                archive_identity = packet["archive_identity"]
            control_request = DaemonOperationRequest(
                operation="operation.status", payload={"request_id": request_id}, request_id="inspect-retained-result"
            )
            for unrelated in (
                replace(principal, actor_ref="synthetic-unrelated"),
                replace(principal, capabilities=principal.capabilities | {"synthetic-extra"}),
                replace(principal, surface="api"),
                replace(principal, role_label="synthetic-unrelated-role"),
            ):
                with pytest.raises(PermissionError):
                    stack.runtime.control(control_request, unrelated, archive_identity)
            with pytest.raises(ValueError, match="archive_identity_stale"):
                stack.runtime.control(control_request, principal, "synthetic-different-archive")
            # A result is not a five-minute progress buffer. An identical
            # replay must not execute restoration against the occupied root.
            offset += 601.0
            replay = stack.client.operation(
                "maintenance.restore_verified_backup",
                {"backup_dir": backup["result"]["result"]["output_path"], "destination": str(destination)},
                archive_root=str(stack.archive_root),
                request_id=request_id,
            )
            assert replay is not None and replay["outcome"] == "completed", replay
            assert replay["result"]["result"] == terminal["result"]["result"]
            later = stack.client.operation(
                "operation.cancel", {"request_id": request_id}, archive_root=str(stack.archive_root)
            )
            assert later is not None and later["result"]["outcome"] == "completed", later
            conflicting = stack.client.operation(
                "maintenance.restore_verified_backup",
                {"backup_dir": backup["result"]["result"]["output_path"], "destination": str(tmp_path / "conflict")},
                archive_root=str(stack.archive_root),
                request_id=request_id,
            )
            assert conflicting is not None and conflicting["outcome"] == "rejected", conflicting
            assert not (tmp_path / "conflict").exists()
        finally:
            release.set()
            thread.join()
    if scratch is not None:
        assert not scratch.exists()
    assert stack.runtime.shutdown_settled
    with ArchiveStore.open_existing(destination, read_only=True):
        pass


@pytest.mark.parametrize("route", ["uds", "execution"])
@pytest.mark.parametrize("deadline_ms", [None, 1_000])
def test_slow_aggregate_waits_for_valid_work_unless_the_caller_declares_a_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, route: str, deadline_ms: int | None
) -> None:
    """Advance the controlled clock inside real aggregate SQL, without a sleep."""
    from time import monotonic
    from types import SimpleNamespace

    from polylogue.archive.query.execution_control import QueryExecutionContext
    from polylogue.operations import daemon_execution
    from polylogue.operations.daemon_protocol import DaemonOperationRequest
    from polylogue.operations.mutation_transaction import MutationPrincipal
    from polylogue.operations.operation_context_types import OperationContext
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    clock = {"now": monotonic()}
    actual_count = ArchiveStore.aggregate_sessions
    counted: list[int] = []

    def slow_count(self: ArchiveStore, mode: str, **kwargs: Any) -> int:
        clock["now"] += 1_000.0
        count = actual_count(self, mode, **kwargs)
        assert isinstance(count, int)
        counted.append(count)
        return count

    def deadline_exceeded(self: QueryExecutionContext) -> bool:
        return self.deadline_monotonic is not None and clock["now"] > self.deadline_monotonic

    def seed(root: Path) -> None:
        _seed_sessions(root, count=1)

    with running_daemon_operations(tmp_path / "archive", seed_archive=seed) as stack:
        monkeypatch.setattr(ArchiveStore, "aggregate_sessions", slow_count)
        monkeypatch.setattr(QueryExecutionContext, "deadline_exceeded", deadline_exceeded)
        monkeypatch.setattr("polylogue.daemon.operation_runtime.monotonic", lambda: clock["now"])
        monkeypatch.setattr(daemon_execution, "monotonic", lambda: clock["now"])
        if route == "uds":
            envelope = stack.client.operation("query.aggregate", {"mode": "count"}, deadline_ms=deadline_ms)
        else:
            # The direct executor still pins a real controlled archive view;
            # this runtime only supplies publication exclusion and observation.
            context = OperationContext(
                stack.archive_root,
                MutationPrincipal("synthetic-read", frozenset({"read"}), "daemon"),
                "daemon",
                SimpleNamespace(
                    publication_guard=stack.runtime.publication_guard, observe_snapshot=lambda *_args: None
                ),
            )
            request = DaemonOperationRequest(
                "query.aggregate", {"mode": "count"}, request_id="slow-direct-read", deadline_ms=deadline_ms
            )
            envelope = daemon_execution.execute_operation(request, context).to_dict()
        assert envelope is not None
        if deadline_ms is None:
            assert envelope["outcome"] == "completed"
            assert envelope["result"]["count"] == 1
            assert counted == [1]
        else:
            assert envelope["outcome"] == "timed-out"
            assert envelope["result"] is None


def test_socket_aggregate_uses_bulk_compute_admission(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio
    from functools import partial

    from tests.infra.archive_templates import run_archive_fixture_prepare

    def seed(root: Path) -> None:
        asyncio.run(run_archive_fixture_prepare(partial(_seed_sessions, root, count=2)))

    with running_daemon_operations(tmp_path / "archive", seed_archive=seed) as stack:
        actual_submit = stack.execution_kernel.submit
        admitted: list[str] = []
        reserved_bytes: list[int] = []
        received_bytes: list[int] = []
        actual_call = stack.runtime.call

        def record_call(*args: Any, **kwargs: Any) -> Any:
            received_bytes.append(kwargs["request_body_bytes"])
            return actual_call(*args, **kwargs)

        def record_submit(function: Any, **kwargs: Any) -> Any:
            admitted.append(kwargs["admission_class"])
            reserved_bytes.append(kwargs["estimated_bytes"])
            return actual_submit(function, **kwargs)

        monkeypatch.setattr(stack.execution_kernel, "submit", record_submit)
        monkeypatch.setattr(stack.runtime, "call", record_call)
        envelope = stack.client.operation("query.aggregate", {"mode": "count"})
        assert envelope is not None
        assert envelope["outcome"] == "completed"
        assert envelope["result"]["count"] == 2
        assert admitted == ["bulk-candidate"]
        assert reserved_bytes == received_bytes
        assert reserved_bytes[0] > 0


@pytest.mark.parametrize("shutdown_delivery", [False, True])
def test_socket_controls_remain_available_during_waiting_work_and_slow_delivery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, shutdown_delivery: bool
) -> None:
    """Use real sockets, queued work and an unread large response, without reserving fake slots."""
    import asyncio

    from polylogue.operations import daemon_reads
    from polylogue.operations.daemon_protocol import DaemonOperationRequest
    from polylogue.storage.sqlite.connection import _clear_connection_cache
    from tests.infra.archive_templates import run_archive_fixture_prepare

    session_id = ""
    entered = threading.Event()
    release = threading.Event()
    delivery_finished = threading.Event()
    actual_aggregate = daemon_reads._aggregate_payload
    actual_setup = MachineOperationHandler.setup
    actual_send = MachineOperationHandler._send

    def seed(root: Path) -> None:
        nonlocal session_id

        def prepare() -> str:
            builder = SessionBuilder(root / "index.db", "slow-delivery").provider("codex")
            builder.add_message(text="Synthetic bounded delivery prose. " * 32768).save()
            identity = builder.native_session_id()
            _clear_connection_cache()
            return identity

        session_id = asyncio.run(run_archive_fixture_prepare(prepare))

    def held_aggregate(*args: Any, **kwargs: Any) -> Any:
        entered.set()
        assert release.wait(30)
        return actual_aggregate(*args, **kwargs)

    def setup(handler: MachineOperationHandler) -> None:
        handler.request.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1024)
        actual_setup(handler)

    def send(handler: MachineOperationHandler, status: int, payload: dict[str, object]) -> None:
        try:
            actual_send(handler, status, payload)
        finally:
            if payload.get("operation") == "session.read":
                delivery_finished.set()

    monkeypatch.setattr(MachineOperationHandler, "setup", setup)
    monkeypatch.setattr(MachineOperationHandler, "_send", send)
    monkeypatch.setattr(daemon_reads, "_aggregate_payload", held_aggregate)
    with running_daemon_operations(tmp_path / "archive", seed_archive=seed, compute_workers=1) as stack:
        peer = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        peer.settimeout(10)
        peer.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024)
        responses: queue.Queue[dict[str, object]] = queue.Queue()
        failures: queue.Queue[BaseException] = queue.Queue()

        def query(request_id: str) -> None:
            try:
                client = DaemonClient(stack.socket_path, timeout_s=30)
                result = client.operation("query.aggregate", {"mode": "count"}, request_id=request_id)
                assert result is not None
                responses.put(result)
            except BaseException as failure:
                failures.put(failure)

        callers: list[threading.Thread] = []
        try:
            peer.connect(str(stack.socket_path))
            request = DaemonOperationRequest("session.read", {"ref": session_id}, request_id="slow-socket-delivery")
            body = json.dumps(request.to_dict(), separators=(",", ":")).encode()
            peer.sendall(
                b"POST /api/operation HTTP/1.1\r\nHost: local\r\nContent-Type: application/json\r\n"
                + f"Content-Length: {len(body)}\r\n\r\n".encode()
                + body
            )
            headers = bytearray()
            while not headers.endswith(b"\r\n\r\n"):
                chunk = peer.recv(1)
                assert chunk, "delivery ended before response headers"
                headers.extend(chunk)
            assert bytes(headers).startswith(b"HTTP/1.1 200 ")
            length_line = next(line for line in bytes(headers).split(b"\r\n") if line.startswith(b"Content-Length:"))
            assert int(length_line.split(b":", 1)[1]) > 512 * 1024
            assert not delivery_finished.is_set()

            for request_id in ("socket-running-work", "socket-queued-work"):
                caller = threading.Thread(target=query, args=(request_id,), daemon=True)
                callers.append(caller)
                caller.start()
                if request_id == "socket-running-work":
                    assert entered.wait(10)
            with stack.runtime._condition:
                assert stack.runtime._condition.wait_for(
                    lambda: "socket-queued-work" in stack.runtime._exchanges, timeout=10
                )
            status = stack.client.operation("operation.status", {"request_id": "socket-running-work"})
            assert status is not None and status["outcome"] == "completed"
            assert status["result"]["outcome"] in {"accepted", "running"}
            cancelled = stack.client.operation("operation.cancel", {"request_id": "socket-queued-work"})
            assert cancelled is not None and cancelled["outcome"] == "completed"
            assert cancelled["result"]["outcome"] == "cancelled"
            assert not delivery_finished.is_set()
            assert not release.is_set()
        finally:
            release.set()
            if shutdown_delivery:
                for caller in callers:
                    caller.join(30)
                    assert not caller.is_alive()
                stack.server.shutdown()
                stack.server.server_close()
                assert delivery_finished.is_set()
                assert not stack.server._handler_sockets
            peer.close()
            for caller in callers:
                caller.join(30)
                assert not caller.is_alive()
        assert failures.empty(), list(failures.queue)
        assert responses.qsize() == 2


@pytest.mark.parametrize("lane", ["semantic", "hybrid"])
@pytest.mark.parametrize("vector_fault", ["missing", "unreadable", "runtime_unavailable"])
def test_keyless_text_read_skips_vector_snapshot_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lane: str, vector_fault: str
) -> None:
    """Text acquisition is disabled before any retained-vector handle is admitted."""
    from unittest.mock import MagicMock

    from polylogue.operations.daemon_reads import DaemonReadDependencies, VectorReadBinding
    from polylogue.storage.search_providers.sqlite_vec import SqliteVecProvider
    from polylogue.storage.search_providers.sqlite_vec_support import SqliteVecUnavailableError
    from tests.infra.vector_archive import seed_vector_archive

    faults = {
        "missing": FileNotFoundError("synthetic missing vector tier"),
        "unreadable": sqlite3.DatabaseError("synthetic unreadable vector tier"),
        "runtime_unavailable": SqliteVecUnavailableError("synthetic unavailable vector runtime"),
    }
    admit = MagicMock(side_effect=faults[vector_fault])
    acquire = MagicMock(side_effect=AssertionError("keyless text cannot acquire vectors"))
    monkeypatch.setattr("polylogue.storage.search_providers.sqlite_vec_runtime.open_vector_read_snapshot", admit)
    monkeypatch.setattr(SqliteVecProvider, "_get_embeddings", acquire)

    def seed(root: Path) -> None:
        seed_vector_archive(root, [("seed", "m1", "Synthetic needle prose.", [1.0] + [0.0] * 1023)])

    with running_daemon_operations(
        tmp_path / "archive",
        seed_archive=seed,
        read_dependencies=DaemonReadDependencies(vector_binding=VectorReadBinding(None, "voyage-4", 1024)),
    ) as stack:
        envelope = stack.client.operation(
            "cli.query",
            {"params": {"query": ("needle",), "retrieval_lane": lane}},
            archive_root=str(stack.archive_root),
        )
    assert envelope is not None
    if lane == "semantic":
        assert envelope["error"]["code"] == "EmbeddingRetrievalNotReadyError", envelope
    else:
        result = envelope["result"]
        assert result["outcome"]["state"] == "degraded", envelope
        assert result["unavailable_lanes"] == ["vector"], envelope
        assert result["failed_lanes"] == [], envelope
    admit.assert_not_called()
    acquire.assert_not_called()


def test_socket_shutdown_physically_settles_an_admitted_incomplete_body(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entered_read = threading.Event()
    post_threads: set[int] = set()
    actual_post = MachineOperationHandler.do_POST
    actual_readinto = socket.SocketIO.readinto

    def post(handler: MachineOperationHandler) -> None:
        identity = threading.get_ident()
        post_threads.add(identity)
        try:
            actual_post(handler)
        finally:
            post_threads.remove(identity)

    def readinto(stream: socket.SocketIO, buffer: Any) -> int | None:
        if threading.get_ident() in post_threads:
            entered_read.set()
        return actual_readinto(stream, buffer)

    monkeypatch.setattr(MachineOperationHandler, "do_POST", post)
    monkeypatch.setattr(socket.SocketIO, "readinto", readinto)
    with running_daemon_operations(tmp_path / "archive") as stack:
        peer = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            peer.connect(str(stack.socket_path))
            peer.sendall(
                b"POST /api/operation HTTP/1.1\r\nHost: local\r\n"
                b"Content-Type: application/json\r\nContent-Length: 200\r\n\r\n{"
            )
            assert entered_read.wait(10)
            with stack.runtime._condition:
                assert not stack.runtime._exchanges
            assert stack.server._handler_sockets
            stack.server.shutdown()
            stack.server.server_close()
            assert not stack.server._handler_sockets
            assert not post_threads
            with stack.runtime._condition:
                assert not stack.runtime._exchanges
        finally:
            peer.close()


def test_annotation_import_commits_summary_and_pages_all_amplified_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A post-write result-size refusal must not erase the original commit verdict."""
    import asyncio

    from polylogue.api import Polylogue
    from polylogue.archive.message.roles import Role
    from polylogue.core.enums import Provider
    from polylogue.operations.daemon_protocol import DaemonOperationRequest
    from polylogue.sources.parsers.base import ParsedMessage, ParsedSession
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from tests.infra.live_ingest import write_index_session

    def seed(root: Path) -> None:
        with ArchiveStore.open_existing(root, read_only=False) as archive:
            write_index_session(
                archive,
                ParsedSession(
                    source_name=Provider.CODEX,
                    provider_session_id="annotation-amplification",
                    messages=[ParsedMessage(provider_message_id="m1", role=Role.USER, text="evidence")],
                ),
            )

    jsonl = "\n".join(
        json.dumps(
            {
                "row_key": f"row-{index}",
                "value": {"activity": "debugging", "confidence": 0.9},
                "evidence_refs": [""] * 10_000,
            }
        )
        for index in range(20)
    )
    with running_daemon_operations(tmp_path / "archive", seed_archive=seed) as stack:
        envelope = stack.runtime.call(
            DaemonOperationRequest(
                "mutation.annotation.import_batch",
                {
                    "jsonl": jsonl,
                    "batch_id": "amplified-errors",
                    "schema_id": "seed.activity",
                    "schema_version": 2,
                    "target_ref": "session:codex-session:annotation-amplification",
                    "source_result_ref": "result-set:amplification",
                    "actor_ref": "agent:labeler",
                    "model_ref": "agent:model",
                    "prompt_ref": "block:prompt:0",
                    "metadata": {},
                },
                request_id="amplified-annotation-import",
                archive_root=str(stack.archive_root),
                deadline_ms=60_000,
            ),
            _all_capabilities_principal(),
        )
        assert envelope["outcome"] == "completed"
        operation_result = cast(dict[str, Any], envelope["result"])
        assert operation_result["effect"] == "committed"
        summary = cast(dict[str, Any], operation_result["result"])
        assert summary["status"] == "partial"
        assert summary["invalid_count"] == summary["total_count"] == 20
        assert summary["valid_count"] == 0
        assert "rows" not in summary
        assert len(json.dumps(envelope).encode()) < 16_000

        def reject_full_batch(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("paged ref read hydrated the complete failure document")

        monkeypatch.setattr(ArchiveStore, "get_annotation_batch", reject_full_batch)

        async def read_pages() -> None:
            async with Polylogue(archive_root=stack.archive_root, db_path=stack.archive_root / "index.db") as poly:
                for offset in (0, 9_999, 10_000, 199_999, 200_000):
                    resolved = await poly.resolve_ref(summary["batch_ref"], limit=1, offset=offset)
                    assert resolved.resolved and resolved.payload is not None
                    page = resolved.payload
                    assert page["total"] == 200_000
                    if offset == 200_000:
                        assert page["items"] == [] and page["next_offset"] is None
                        continue
                    item = page["items"][0]
                    assert item["failure_ordinal"] == offset // 10_000
                    assert item["error_ordinal"] == offset % 10_000
                    assert item["failure"] == {"line": offset // 10_000 + 1, "row_key": f"row-{offset // 10_000}"}
                    assert item["error"] == "evidence_ref '' does not resolve in the live archive"
                    assert page["next_offset"] == (offset + 1 if offset < 199_999 else None)

        asyncio.run(read_pages())


@pytest.mark.parametrize("criterion", ["similar_text", "similar_session_id"])
@pytest.mark.parametrize("selected", [True, False])
def test_temporal_selected_reference_skips_unavailable_vector_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, criterion: str, selected: bool
) -> None:
    from unittest.mock import MagicMock

    from polylogue.operations.daemon_reads import DaemonReadDependencies, VectorReadBinding
    from polylogue.storage.search_providers.sqlite_vec import SqliteVecProvider

    admit = MagicMock(side_effect=FileNotFoundError("synthetic unavailable embeddings tier"))
    acquire = MagicMock(side_effect=AssertionError("this read must not acquire vectors"))
    monkeypatch.setattr("polylogue.storage.search_providers.sqlite_vec_runtime.open_vector_read_snapshot", admit)
    monkeypatch.setattr(SqliteVecProvider, "_get_embeddings", acquire)
    ids: list[str] = []

    def seed(root: Path) -> None:
        for number in range(2):
            builder = SessionBuilder(root / "index.db", f"temporal-vector-{number}").provider("codex")
            builder.add_message(text="synthetic temporal evidence").save()
            ids.append(builder.native_session_id())

    with running_daemon_operations(
        tmp_path / "archive",
        seed_archive=seed,
        read_dependencies=DaemonReadDependencies(
            vector_binding=VectorReadBinding("synthetic-unused", "voyage-3-lite", 1024)
        ),
    ) as stack:
        envelope = stack.client.operation(
            "read.temporal",
            {
                "session_id": ids[1] if selected else None,
                "params": {criterion: "synthetic query" if criterion == "similar_text" else ids[0]},
            },
            archive_root=str(stack.archive_root),
        )
    assert envelope is not None
    acquire.assert_not_called()
    if selected:
        assert envelope["outcome"] == "completed", envelope
        assert envelope["readiness"]["degraded_components"] == [], envelope
        events = envelope["result"]["payload"]["temporal_window"]["events"]
        assert {ref for event in events for ref in event["evidence_refs"] if ref.startswith("session:")} == {
            f"session:{ids[1]}"
        }
        admit.assert_not_called()
    else:
        admit.assert_called_once()
        assert "semantic_snapshot" in envelope["readiness"]["degraded_components"], envelope


def test_staged_backup_compute_refusal_is_retryable_and_recovers(tmp_path: Path) -> None:
    """The real staged phase must preserve its scheduler's transient refusal."""
    entered = threading.Event()
    release = threading.Event()

    def occupy_input_capacity() -> None:
        entered.set()
        assert release.wait(timeout=30)

    packages = tmp_path / "packages"
    with running_daemon_operations(tmp_path / "archive") as stack:
        blocker = stack.execution_kernel.submit(
            occupy_input_capacity, admission_class="control", estimated_bytes=0, exclusive_bytes=True
        )
        try:
            assert entered.wait(timeout=5)
            refused = stack.client.operation(
                "maintenance.backup",
                {"output_dir": str(packages), "verify": True, "profile": "full_evidence"},
                archive_root=str(stack.archive_root),
            )
            assert refused is not None
            assert refused["outcome"] == "rejected", refused
            assert refused["error"]["code"] == "compute_backpressure"
            assert refused["error"]["retryable"] is True
            assert refused["result"] is None
            assert not packages.exists()
        finally:
            release.set()
            blocker.future.result(timeout=10)
        completed = stack.client.operation(
            "maintenance.backup",
            {"output_dir": str(packages), "verify": True, "profile": "full_evidence"},
            archive_root=str(stack.archive_root),
        )
        assert completed is not None and completed["outcome"] == "completed", completed
        assert completed["result"]["result"]["verified"] is True


def test_accepted_restore_backpressure_retains_indeterminate_terminal_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Admission failure after acceptance cannot authorize a fresh request replay."""
    from polylogue.core.compute import DaemonBackpressureError
    from polylogue.operations import archive_backup

    attempts = 0

    def saturated_restore(**_kwargs: Any) -> Any:
        nonlocal attempts
        attempts += 1
        raise DaemonBackpressureError("synthetic post-acceptance compute refusal")

    monkeypatch.setattr(archive_backup, "restore_verified_backup", saturated_restore)
    with running_daemon_operations(tmp_path / "archive") as stack:
        request_id = "accepted-restore-backpressure"
        payload: dict[str, object] = {"backup_dir": str(tmp_path / "backup"), "destination": str(tmp_path / "restored")}
        terminal = stack.client.operation(
            "maintenance.restore_verified_backup",
            payload,
            archive_root=str(stack.archive_root),
            request_id=request_id,
        )
        assert terminal is not None and terminal["outcome"] == "indeterminate", terminal
        assert terminal["error"]["code"] == "compute_backpressure"
        assert terminal["error"]["retryable"] is False
        replay = stack.client.operation(
            "maintenance.restore_verified_backup",
            payload,
            archive_root=str(stack.archive_root),
            request_id=request_id,
        )
        assert replay is not None and replay["outcome"] == "indeterminate", replay
        assert replay["error"] == terminal["error"]
        assert replay["schema_versions"] == terminal["schema_versions"]
        status = stack.client.operation(
            "operation.status", {"request_id": request_id}, archive_root=str(stack.archive_root)
        )
        assert status is not None and status["result"]["outcome"] == "indeterminate", status
        assert status["result"]["error"]["retryable"] is False
        assert attempts == 1
