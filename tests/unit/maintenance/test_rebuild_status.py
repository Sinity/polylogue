"""``rebuild_status`` (polylogue-b5l.1 AC5): one consolidated read for lease
ownership, the active generation, the resumable transaction's cursor/delta,
and explicit stale-lock/failed-transaction recovery guidance.

Anti-vacuity: the mutation that makes
``test_reports_stale_lease_recovery_guidance`` fail is removing the
``lease.stale`` branch's recovery message (or ``rebuild_lease_status``'s own
dead-pid detection this depends on); the mutation that makes
``test_reports_source_snapshot_delta_when_source_has_drifted`` fail is
dropping the ``rebuild_source_evidence_snapshot`` comparison and always
reporting ``source_snapshot_matches=True``.
"""

from __future__ import annotations

import fcntl
import json
import os
import sqlite3
from pathlib import Path

import pytest

from polylogue.core.enums import Provider
from polylogue.maintenance.rebuild_index import RebuildIndexRequest, rebuild_index_from_source_sync, rebuild_status
from polylogue.storage.index_generation import IndexGenerationStore, rebuild_source_evidence_snapshot
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root, initialize_archive_database
from polylogue.storage.sqlite.migration_runner import DURABLE_MIGRATION_TIERS
from tests.infra.rebuild_receipt import write_valid_rebuild_receipt

_DEFINITELY_DEAD_PID = 2**31 - 1


def _init_empty_source(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for tier in sorted(DURABLE_MIGRATION_TIERS, key=lambda item: item.value):
        initialize_archive_database(root / f"{tier.value}.db", tier)


def _codex_session(native_id: str) -> bytes:
    rows: list[dict[str, object]] = [
        {"type": "session_meta", "payload": {"id": native_id, "timestamp": "2026-07-16T10:00:00Z"}},
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "id": f"{native_id}-m0",
                "role": "user",
                "content": [{"type": "input_text", "text": f"hello {native_id}"}],
            },
        },
    ]
    return b"".join(json.dumps(row, sort_keys=True).encode() + b"\n" for row in rows)


def _seed_one_real_codex_session(root: Path) -> None:
    initialize_active_archive_root(root)
    with ArchiveStore.open_existing(root, read_only=False) as archive:
        archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=_codex_session("sess-status-probe"),
            source_path="status-probe-test/0.jsonl",
            acquired_at_ms=1,
        )


def test_reports_no_lease_and_no_transaction_on_a_fresh_archive(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    _init_empty_source(root)

    status = rebuild_status(root, operation_id="does-not-exist", include_daemon_bulk_rebuild=False)

    assert status["archive_root"] == str(root)
    assert status["lease"] == {
        "held": False,
        "holder_pid": None,
        "holder_host": None,
        "holder_alive": None,
        "stale": False,
    }
    assert status["generation"] is None
    assert status["transaction"] is None
    assert status["delta"] is None
    assert status["recovery"] == []


def test_reports_stale_lease_recovery_guidance(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    _init_empty_source(root)
    lock_path = root / ".index-rebuild.lock"
    holder_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(holder_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.write(holder_fd, f"pid={_DEFINITELY_DEAD_PID} host=nowhere\n".encode())
    os.fsync(holder_fd)
    try:
        status = rebuild_status(root, operation_id="none", include_daemon_bulk_rebuild=False)
        lease = status["lease"]
        assert isinstance(lease, dict)
        assert lease["held"] is True
        assert lease["stale"] is True
        recovery = status["recovery"]
        assert isinstance(recovery, list)
        assert any("dead pid" in message for message in recovery)
    finally:
        fcntl.flock(holder_fd, fcntl.LOCK_UN)
        os.close(holder_fd)


def test_reports_active_generation_and_schema_version_after_a_rebuild(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "archive"
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(root))
    _seed_one_real_codex_session(root)
    receipt = rebuild_index_from_source_sync(RebuildIndexRequest(archive_root=root))
    assert receipt.status == "replayed"

    status = rebuild_status(root, operation_id="none", include_daemon_bulk_rebuild=False)

    generation = status["generation"]
    assert isinstance(generation, dict)
    assert generation["state"] == "active"
    assert status["schema_version"] is not None


def test_reports_transaction_cursor_and_no_delta_when_source_unchanged(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    _init_empty_source(root)
    with sqlite3.connect(root / "source.db") as conn:
        conn.execute(
            """INSERT INTO raw_sessions (raw_id, origin, native_id, source_path, source_index, blob_hash,
               blob_size, acquired_at_ms, validation_status)
               VALUES ('raw-a', 'codex-session', 'raw-a', '/raw-a', 0, randomblob(32), 1, 1, 'passed')"""
        )
    store = IndexGenerationStore.for_archive_root(root)
    transaction = store.create_transaction(
        source_snapshot=rebuild_source_evidence_snapshot(root), operation_id="status-probe-op"
    )
    transaction = store.checkpoint_transaction(
        transaction, status="paused", last_raw_id="raw-a", last_blob_hash_hex="00" * 32, processed_raw_count=1
    )

    status = rebuild_status(root, operation_id="status-probe-op")

    txn_payload = status["transaction"]
    assert isinstance(txn_payload, dict)
    assert txn_payload["operation_id"] == "status-probe-op"
    assert txn_payload["processed_raw_count"] == 1
    assert txn_payload["heartbeat_at_ms"] is not None
    delta = status["delta"]
    assert isinstance(delta, dict)
    assert delta["source_snapshot_matches"] is True
    operation = status["operation"]
    assert isinstance(operation, dict)
    assert operation["cursor"] is not None
    assert operation["heartbeat"] == {"at_ms": txn_payload["heartbeat_at_ms"]}
    assert operation["recovery_state"] == "paused"
    assert status["recovery"] == []


def test_reports_source_snapshot_delta_when_source_has_drifted(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    _init_empty_source(root)
    with sqlite3.connect(root / "source.db") as conn:
        conn.execute(
            """INSERT INTO raw_sessions (raw_id, origin, native_id, source_path, source_index, blob_hash,
               blob_size, acquired_at_ms, validation_status)
               VALUES ('raw-a', 'codex-session', 'raw-a', '/raw-a', 0, randomblob(32), 1, 1, 'passed')"""
        )
    store = IndexGenerationStore.for_archive_root(root)
    transaction = store.create_transaction(source_snapshot="stale-snapshot", operation_id="drift-op")
    assert transaction.status == "running"

    status = rebuild_status(root, operation_id="drift-op")

    delta = status["delta"]
    assert isinstance(delta, dict)
    assert delta["source_snapshot_matches"] is False
    recovery = status["recovery"]
    assert isinstance(recovery, list)
    assert any("source snapshot no longer matches" in message for message in recovery)


def test_falls_back_to_the_daemon_well_known_operation_id_by_default(tmp_path: Path) -> None:
    """Omitting ``operation_id`` must resolve the daemon's own well-known
    bulk-rebuild transaction -- the common case for
    ``ops reset --index && polylogued run``, which never has an operation id
    to hand this status surface explicitly."""
    from polylogue.daemon.bulk_rebuild import (
        DAEMON_BULK_REBUILD_OPERATION_ID,
        resolve_or_start_daemon_bulk_rebuild_transaction,
    )

    root = tmp_path / "archive"
    _init_empty_source(root)
    receipt = write_valid_rebuild_receipt(root, tmp_path / "schema-inference-gate-receipt.json")
    resolve_or_start_daemon_bulk_rebuild_transaction(root, schema_inference_receipt_path=receipt)

    status = rebuild_status(root)

    assert status["operation_id"] == DAEMON_BULK_REBUILD_OPERATION_ID
    assert status["transaction"] is not None
