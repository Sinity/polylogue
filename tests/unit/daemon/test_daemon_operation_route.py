"""Production UDS operation route contracts."""

from __future__ import annotations

import json
import os
import queue
import socket
import sqlite3
import threading
from contextlib import ExitStack, closing
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest

from polylogue.daemon.uds import MachineOperationHandler
from polylogue.daemon_client import DaemonClient, DaemonOperationRejectedError
from polylogue.operations.mutation_actuators import SessionDeleteActuator, SessionDeleteArgs
from polylogue.operations.mutation_transaction import MAX_MUTATION_PLAN_TARGETS, MutationPlan, MutationReceipt
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


def _seed_sessions(root: Path, *, count: int, title: str = "Operation route session") -> tuple[str, ...]:
    """Seed one fully bootstrapped synthetic archive before daemon startup."""

    session_ids: list[str] = []
    for number in range(count):
        builder = (
            SessionBuilder(root / "index.db", f"operation-{number}")
            .provider("codex")
            .title(title)
            .add_message(text=f"Synthetic daemon operation session {number}.")
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


def test_operation_route_bounds_the_real_canonical_envelope(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Mutation: bypass the UDS response bound and the oversized canonical rows escape."""

    import json

    import polylogue.daemon.uds as uds

    def seed(root: Path) -> None:
        _seed_sessions(root, count=8)

    with running_daemon_operations(tmp_path / "archive", seed_archive=seed) as stack:
        unbounded = stack.client.operation("cli.query", {"params": {"limit": 8}}, archive_root=str(stack.archive_root))
        assert unbounded is not None and unbounded["outcome"] == "completed"
        assert len(json.dumps(unbounded, separators=(",", ":")).encode()) > 4096
        monkeypatch.setattr(uds, "MAX_OPERATION_RESULT_BYTES", 4096)
        envelope = stack.client.operation(
            "cli.query",
            {"params": {"limit": 8}},
            archive_root=str(stack.archive_root),
        )

    assert envelope is not None
    assert envelope["outcome"] == "failed"
    assert envelope["result"] is None
    assert envelope["error"]["code"] == "result_too_large"


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
            {"preview_refs": preview["result"]["preview_refs"]},
            archive_root=str(root),
            request_id="indeterminate-authorize",
        )
        assert authorization is not None
        lost = first.client.operation_to_completion(
            "mutation.session.delete.execute",
            {"authorization_refs": authorization["result"]["authorization_refs"]},
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
            {"authorization_refs": authorization["result"]["authorization_refs"]},
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
                {"preview_refs": preview["result"]["preview_refs"]},
                archive_root=str(root),
                request_id="prefix-authorize",
            )
            assert authorization is not None
            lost = first.client.operation_to_completion(
                "mutation.session.delete.execute",
                {"authorization_refs": authorization["result"]["authorization_refs"]},
                archive_root=str(root),
                request_id="prefix-execute",
            )
            assert lost is not None and lost["outcome"] == "indeterminate"
            assert not first.session_exists(target)
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
            {"preview_refs": preview["result"]["preview_refs"]},
            archive_root=str(root),
            request_id="counted-authorize",
        )
        assert authorization is not None
        executed = stack.client.operation_to_completion(
            "mutation.session.delete.execute",
            {"authorization_refs": authorization["result"]["authorization_refs"]},
            archive_root=str(root),
            request_id="counted-execute",
        )
        assert executed is not None
        assert prepared_count == len(selected)
        assert executed["result"]["affected_count"] == prepared_count
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
            {"preview_refs": preview["result"]["preview_refs"]},
            archive_root=str(root),
        )
        assert authorization is not None
        crash["armed"] = True
        stranded = first.client.operation(
            "mutation.session.delete.execute",
            {"authorization_refs": authorization["result"]["authorization_refs"]},
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
            {"authorization_refs": authorization["result"]["authorization_refs"]},
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
    authorization_refs: list[str]
    with running_daemon_operations(root, seed_archive=seed) as stack, ExitStack() as cleanup:
        cleanup.callback(release_apply.set)
        preview = stack.client.operation_to_completion(
            "mutation.session.delete.preview",
            {"session_ids": list(session_ids)},
            archive_root=str(root),
        )
        assert preview is not None
        authorization = stack.client.operation_to_completion(
            "mutation.session.delete.authorize",
            {"preview_refs": preview["result"]["preview_refs"]},
            archive_root=str(root),
        )
        assert authorization is not None
        authorization_refs = list(authorization["result"]["authorization_refs"])

        from polylogue.operations.daemon_protocol import DaemonOperationRequest

        request = DaemonOperationRequest(
            "mutation.session.delete.execute",
            {"authorization_refs": authorization_refs},
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
        session_ids = _seed_sessions(root, count=MAX_MUTATION_PLAN_TARGETS + 1)

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
            {"preview_refs": preview_result["preview_refs"]},
            archive_root=str(stack.archive_root),
        )
        assert authorization is not None
        execute_request_id = "blocked-delete"
        execute_result: queue.Queue[dict[str, object]] = queue.Queue()

        def execute() -> None:
            client = DaemonClient(stack.socket_path, timeout_s=5)
            response = client.operation(
                "mutation.session.delete.execute",
                {"authorization_refs": authorization["result"]["authorization_refs"]},
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
            timed_out = stack.client.operation(
                "operation.await",
                {"request_id": execute_request_id, "after_sequence": status["result"]["sequence"], "timeout_ms": 2_000},
                archive_root=str(stack.archive_root),
                deadline_ms=25,
            )
            assert timed_out is not None and timed_out["outcome"] == "timed-out"
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
                        "after_sequence": status["result"]["sequence"],
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
        assert state["outcome"] == "cancelled"
        assert state["completed_chunks"] == 1
        assert state["not_attempted"] == [1]
        assert state["stop_reason"] == "cancelled"
        assert state["reference"] == accepted["accepted_reference"]
        assert all(not stack.session_exists(session_id) for session_id in session_ids[:-1])
        assert stack.session_exists(session_ids[-1])

    with running_daemon_operations(tmp_path / "archive") as restarted:
        replay = restarted.client.operation(
            "mutation.session.delete.execute",
            {"authorization_refs": authorization["result"]["authorization_refs"]},
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


def test_cancelled_queued_control_reports_cancelled_not_failed(tmp_path: Path) -> None:
    """A pre-acceptance cancellation is a cancellation, not an operation failure.

    The scheduler cancels a queued, unstarted task by completing its future
    with ``DaemonOperationCancelled``
    (``BoundedComputeAdapter._cancel_before_start``), having already released
    the reservation with no work started.

    Anti-vacuity: the operation is genuinely queued behind saturated workers
    (its exchange exists and its future is not done before the cancel), so the
    scheduler's pre-start path is the one that fires. Removing the
    ``except DaemonOperationCancelled`` branch from
    ``DaemonOperationRuntime.call`` sends it to the generic handler, which
    reports ``outcome == "failed"`` with
    ``error.code == "DaemonOperationCancelled"``, and both assertions go red.
    """
    from polylogue.daemon.execution import CancellationHandle
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

    request_id = "cancelled-queued-control"
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
            "mutation.session.delete.preview",
            {"session_ids": ["codex:absent"]},
            request_id=request_id,
            archive_root=str(stack.archive_root),
        )
        disconnect = CancellationHandle()
        envelopes: list[dict[str, object]] = []

        def call_runtime() -> None:
            envelopes.append(stack.runtime.call(request, principal, client_disconnect=disconnect))

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

            disconnect.cancel()
            caller.join(timeout=5)
            assert not caller.is_alive()
        finally:
            release.set()
            for blocker in blockers:
                blocker.future.result(timeout=10)

    assert len(envelopes) == 1
    envelope = envelopes[0]
    assert envelope["outcome"] == "cancelled"
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
            assert control.completed == 0
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
    payload: dict[str, object] = {"session_ids": session_ids, "add_marks": ["reviewed"]}
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
                "schema_version": 1,
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
