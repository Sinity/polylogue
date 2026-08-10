"""``rebuild_index_from_source`` must prove archive-location ownership before
touching any generation directory or SQLite tier (polylogue-ovme.2 AC3).

An offline rebuild is exactly the maintenance/campaign writer
``OwnedArchiveLocation`` (polylogue-ovme.1, PR #3291) exists for. Before this
change, an offline rebuild never acquired that capability at all -- only
``RebuildLease`` (a rebuild-specific exclusion lock) guarded it, which does
not protect against a concurrent *different* maintenance/campaign writer
holding the general archive-location ownership lock.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import cast

import pytest

from polylogue.archive.revision_authority import RawRevisionAuthority, RawRevisionEnvelope, RawRevisionKind
from polylogue.core.enums import Provider
from polylogue.maintenance.rebuild_index import (
    RebuildIndexRequest,
    RebuildSchemaCurrencyError,
    rebuild_index_from_source_sync,
)
from polylogue.sources.revision_backfill import census_historical_revision_evidence
from polylogue.storage.archive_identity import ArchiveLocation, ArchiveOwnershipError, OwnedArchiveLocation
from polylogue.storage.archive_readiness import probe_archive_tier
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.index_generation import IndexGenerationStore, RebuildLease, rebuild_source_evidence_snapshot
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root, initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.migration_runner import DURABLE_MIGRATION_TIERS
from tests.infra.rebuild_receipt import write_valid_rebuild_receipt


def _init_empty_source(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for tier in sorted(DURABLE_MIGRATION_TIERS, key=lambda item: item.value):
        initialize_archive_database(root / f"{tier.value}.db", tier)


def _init_nonempty_source(root: Path) -> None:
    initialize_active_archive_root(root)
    payload = (
        b'{"type":"session_meta","payload":{"id":"owned-session"}}\n'
        b'{"type":"response_item","payload":{"type":"message","role":"user",'
        b'"content":[{"type":"input_text","text":"owned"}]}}\n'
    )
    with ArchiveStore.open_existing(root, read_only=False) as archive:
        archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=payload,
            source_path="current/owned.jsonl",
            acquired_at_ms=1,
            revision=RawRevisionEnvelope(
                logical_source_key="codex-session:owned-session",
                kind=RawRevisionKind.FULL,
                source_revision="owned-revision",
                acquisition_generation=0,
                authority=RawRevisionAuthority.ASSERTED,
            ),
        )
    with sqlite3.connect(root / "source.db") as conn:
        conn.execute("UPDATE raw_sessions SET baseline_raw_id = raw_id, revision_authority = 'byte_proven'")
        conn.commit()
    census = census_historical_revision_evidence(root)
    assert census.scanned == 1
    assert census.classified_full == 1


def test_rebuild_rejects_source_schema_behind_runtime_before_candidate_creation(tmp_path: Path) -> None:
    """A real v28 source tier must not reach the v29 rebuild package.

    The test builds ordinary file-backed archive tiers, removes exactly v29's
    additive objects, and supplies a valid rebuild receipt. The production
    rebuild route used to accept this archive and return ``empty-source``.
    """
    root = tmp_path / "archive"
    initialize_active_archive_root(root)
    with sqlite3.connect(root / "source.db") as conn:
        conn.execute("DROP INDEX idx_raw_failure_disposition_receipts_disposed_at")
        conn.execute("DROP TABLE raw_failure_disposition_receipts")
        conn.execute("PRAGMA user_version = 28")
    receipt_path = write_valid_rebuild_receipt(root, tmp_path / "schema-inference-receipt.json")

    with pytest.raises(RebuildSchemaCurrencyError) as exc_info:
        rebuild_index_from_source_sync(
            RebuildIndexRequest(archive_root=root, schema_inference_receipt_path=receipt_path)
        )

    diagnostic = exc_info.value.diagnostic
    assert diagnostic["status"] == "blocked"
    assert diagnostic["blocking_tiers"] == [
        {
            "tier": "source",
            "path": str(root / "source.db"),
            "actual_user_version": 28,
            "expected_user_version": 29,
            "status": "mismatch",
        }
    ]
    assert not (root / ".index-generations").exists()
    assert not (root / ".index-rebuild-transactions").exists()


def test_rebuild_rejects_source_schema_ahead_of_runtime_before_candidate_creation(tmp_path: Path) -> None:
    """A newer source tier is as unsafe to rebuild as an older one."""
    root = tmp_path / "archive"
    _init_empty_source(root)
    source_probe = probe_archive_tier(ArchiveTier.SOURCE, root / "source.db")
    with sqlite3.connect(root / "source.db") as conn:
        conn.execute(f"PRAGMA user_version = {source_probe.expected_user_version + 1}")

    with pytest.raises(RebuildSchemaCurrencyError) as exc_info:
        rebuild_index_from_source_sync(RebuildIndexRequest(archive_root=root))

    assert exc_info.value.diagnostic["blocking_tiers"] == [
        {
            "tier": "source",
            "path": str(root / "source.db"),
            "actual_user_version": source_probe.expected_user_version + 1,
            "expected_user_version": source_probe.expected_user_version,
            "status": "mismatch",
        }
    ]
    assert not (root / ".index-generations").exists()


@pytest.mark.parametrize("mode", ["missing", "mismatched"])
def test_rebuild_rejects_missing_or_mismatched_audit_tier_before_candidate_creation(tmp_path: Path, mode: str) -> None:
    """Every canonical durable tier, including audit, must be package-current."""
    root = tmp_path / "archive"
    _init_empty_source(root)
    audit_path = root / "audit.db"
    expected = probe_archive_tier(ArchiveTier.AUDIT, audit_path).expected_user_version
    if mode == "missing":
        audit_path.unlink()
        actual: int | None = None
        status = "missing"
    else:
        with sqlite3.connect(audit_path) as conn:
            conn.execute(f"PRAGMA user_version = {expected + 1}")
        actual = expected + 1
        status = "mismatch"

    with pytest.raises(RebuildSchemaCurrencyError) as exc_info:
        rebuild_index_from_source_sync(RebuildIndexRequest(archive_root=root))

    assert exc_info.value.diagnostic["blocking_tiers"] == [
        {
            "tier": "audit",
            "path": str(audit_path),
            "actual_user_version": actual,
            "expected_user_version": expected,
            "status": status,
        }
    ]
    assert not (root / ".index-generations").exists()


def test_rebuild_rechecks_schema_currency_after_acquiring_archive_ownership(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Schema drift after the early guard cannot reach the candidate path."""
    root = tmp_path / "archive"
    _init_empty_source(root)
    receipt_path = write_valid_rebuild_receipt(root, tmp_path / "schema-inference-receipt.json")
    source_probe = probe_archive_tier(ArchiveTier.SOURCE, root / "source.db")
    original_acquire = OwnedArchiveLocation.acquire

    def acquire_then_advance_schema(location: ArchiveLocation) -> OwnedArchiveLocation:
        owned = original_acquire(location)
        with sqlite3.connect(root / "source.db") as conn:
            conn.execute(f"PRAGMA user_version = {source_probe.expected_user_version + 1}")
        return owned

    monkeypatch.setattr("polylogue.maintenance.rebuild_index.OwnedArchiveLocation.acquire", acquire_then_advance_schema)

    with pytest.raises(RebuildSchemaCurrencyError, match="schema currency") as exc_info:
        rebuild_index_from_source_sync(
            RebuildIndexRequest(archive_root=root, schema_inference_receipt_path=receipt_path)
        )

    blocking_tiers = cast(list[dict[str, object]], exc_info.value.diagnostic["blocking_tiers"])
    assert blocking_tiers[0]["tier"] == "source"
    assert not (root / ".index-generations").exists()


def test_rebuild_refuses_when_archive_location_already_owned(tmp_path: Path) -> None:
    """A concurrent holder of the archive-location ownership lock must block
    an offline rebuild before any generation directory or SQLite tier is
    touched -- not merely before promotion, and not via ``RebuildLease``
    (a different, rebuild-specific lock) racing to the same conclusion.
    """
    root = tmp_path / "archive"
    _init_nonempty_source(root)
    receipt_path = write_valid_rebuild_receipt(root, tmp_path / "schema-inference-receipt.json")
    location = ArchiveLocation.resolve(root)
    owned = OwnedArchiveLocation.acquire(location, owner_id="concurrent-campaign")
    try:
        with pytest.raises(ArchiveOwnershipError):
            rebuild_index_from_source_sync(
                RebuildIndexRequest(archive_root=root, schema_inference_receipt_path=receipt_path)
            )
        # Failure happened before any generation bookkeeping was created.
        assert not (root / ".index-generations").exists()
        assert not (root / ".index-rebuild-transactions").exists()
        # The rebuild lease is now deliberately acquired before the general
        # archive-location ownership attempt.  Its released lock file may
        # remain as a diagnostic artifact, but no generation may be created.
    finally:
        owned.release()

    # Releasing the concurrent holder's ownership lets the rebuild proceed.
    receipt = rebuild_index_from_source_sync(
        RebuildIndexRequest(archive_root=root, schema_inference_receipt_path=receipt_path)
    )
    assert receipt.status == "replayed"


def test_rebuild_blocks_unsafe_cursor_authority_before_generation_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "archive"
    initialize_active_archive_root(root)
    cursor_payload = (
        b'{"type":"session_meta","payload":{"id":"session-1"}}\n'
        b'{"type":"response_item","payload":{"type":"message","role":"user",'
        b'"content":[{"type":"input_text","text":"cursor authority"}]}}\n'
    )
    cursor_blob_hash, _ = BlobStore(root / "blob").write_from_bytes(cursor_payload)
    with sqlite3.connect(root / "source.db") as conn:
        conn.execute(
            """
            INSERT INTO raw_sessions (
                raw_id, origin, native_id, source_path, source_index, blob_hash,
                blob_size, acquired_at_ms, logical_source_key, revision_kind,
                source_revision, acquisition_generation, revision_authority
            ) VALUES ('raw-1', 'codex-session', 'session-1', 'source.jsonl', 0, ?,
                      ?, 1, 'codex:session-1', 'full', 'revision-0', 0, 'byte_proven')
            """,
            (bytes.fromhex(cursor_blob_hash), len(cursor_payload)),
        )
        conn.commit()
    census = census_historical_revision_evidence(root)
    assert census.scanned == 1
    assert census.classified_full == 1
    monkeypatch.setattr(
        "polylogue.readiness.capability.raw_frontier_source_selection_block_reason",
        lambda _root, _materialization: "1 ingest cursor row committed past accepted raw material",
    )
    receipt_path = write_valid_rebuild_receipt(root, tmp_path / "schema-inference-receipt.json")

    with pytest.raises(RuntimeError, match="raw frontier integrity"):
        rebuild_index_from_source_sync(
            RebuildIndexRequest(archive_root=root, schema_inference_receipt_path=receipt_path)
        )

    assert not (root / ".index-generations").exists()


def test_rebuild_source_preflight_rejects_orphaned_blob_refs_before_generation_creation(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    _init_empty_source(root)
    with sqlite3.connect(root / "source.db") as conn:
        conn.execute(
            """
            INSERT INTO blob_refs(blob_hash, ref_id, ref_type, size_bytes, acquired_at_ms)
            VALUES (?, 'raw-that-does-not-exist', 'raw_payload', 10, 100)
            """,
            (b"o" * 32,),
        )
        conn.commit()
    receipt_path = write_valid_rebuild_receipt(root, tmp_path / "schema-inference-receipt.json")

    with pytest.raises(RuntimeError, match="reindex source preflight gate failed: blob-refs-liveness"):
        rebuild_index_from_source_sync(
            RebuildIndexRequest(archive_root=root, schema_inference_receipt_path=receipt_path)
        )

    assert not (root / ".index-generations").exists()


def test_rebuild_source_preflight_rejects_unexplained_raw_failure(tmp_path: Path) -> None:
    """Reach raw-failure classification after satisfying earlier readiness gates."""
    root = tmp_path / "archive"
    _init_empty_source(root)
    initialize_archive_database(root / "index.db", ArchiveTier.INDEX)
    initialize_archive_database(root / "ops.db", ArchiveTier.OPS)
    failed_payload = b"raw-failure-fixture"
    failed_blob_hash, _ = BlobStore(root / "blob").write_from_bytes(failed_payload)
    with sqlite3.connect(root / "source.db") as conn:
        conn.execute(
            """
            INSERT INTO raw_sessions(
                raw_id, origin, native_id, source_path, blob_hash, blob_size,
                acquired_at_ms, parse_error
            ) VALUES ('raw-failed', 'codex-session', 'failed', '/x', ?, ?, 100, 'unexpected parser failure')
            """,
            (bytes.fromhex(failed_blob_hash), len(failed_payload)),
        )
        conn.commit()
    receipt_path = write_valid_rebuild_receipt(root, tmp_path / "schema-inference-receipt.json")

    with pytest.raises(RuntimeError, match="raw-failure-lifecycle"):
        rebuild_index_from_source_sync(
            RebuildIndexRequest(archive_root=root, schema_inference_receipt_path=receipt_path)
        )

    assert not (root / ".index-generations").exists()


def test_rebuild_preflight_exposes_unreconciled_source_ref_types(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    _init_empty_source(root)
    with sqlite3.connect(root / "source.db") as conn:
        conn.executemany(
            """
            INSERT INTO blob_refs(blob_hash, ref_id, ref_type, size_bytes, acquired_at_ms)
            VALUES (?, ?, ?, 10, 100)
            """,
            (
                (b"r" * 32, "raw-gone", "raw_payload"),
                (b"a" * 32, "attachment-gone", "attachment"),
                (b"h" * 32, "hook-gone", "hook_payload"),
            ),
        )
        conn.commit()
    receipt_path = write_valid_rebuild_receipt(root, tmp_path / "schema-inference-receipt.json")

    with pytest.raises(RuntimeError) as exc_info:
        rebuild_index_from_source_sync(
            RebuildIndexRequest(archive_root=root, schema_inference_receipt_path=receipt_path)
        )

    message = str(exc_info.value)
    assert "reindex source preflight gate failed: blob-refs-liveness" in message
    assert "raw_payload orphans=1" in message
    assert "attachment orphans=1" in message
    assert "hook_payload orphans=1" in message
    assert not (root / ".index-generations").exists()


def test_rebuild_releases_ownership_lock_after_completion(tmp_path: Path) -> None:
    """The ownership lock must not be left held after a rebuild returns, so a
    second rebuild (or any other maintenance/campaign writer) can acquire it.

    ``flock`` is scoped to the open file description, not the process, so a
    fresh ``acquire`` call from the *same* process only succeeds here if the
    first rebuild's ``owned.release()`` actually ran.
    """
    root = tmp_path / "archive"
    _init_empty_source(root)
    tier_bytes_before = {tier.value: (root / f"{tier.value}.db").read_bytes() for tier in DURABLE_MIGRATION_TIERS}

    receipt = rebuild_index_from_source_sync(RebuildIndexRequest(archive_root=root))
    assert receipt.status == "empty-source"
    assert receipt.consumed_evidence == {}
    assert receipt.generation == {}
    assert not (root / ".index-rebuild-transactions").exists()
    assert {
        tier.value: (root / f"{tier.value}.db").read_bytes() for tier in DURABLE_MIGRATION_TIERS
    } == tier_bytes_before

    location = ArchiveLocation.resolve(root)
    owned = OwnedArchiveLocation.acquire(location, owner_id="post-rebuild-probe")
    try:
        assert (root / ".archive-ownership.lock").exists()
    finally:
        owned.release()


def test_empty_source_rebuild_retains_consumed_evidence_for_resumed_request(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    _init_empty_source(root)
    receipt_path = write_valid_rebuild_receipt(root, tmp_path / "schema-inference-receipt.json")
    store = IndexGenerationStore.for_archive_root(root)
    transaction = store.create_transaction(
        source_snapshot=rebuild_source_evidence_snapshot(root),
        operation_id="empty-source-resume",
    )

    receipt = rebuild_index_from_source_sync(
        RebuildIndexRequest(
            archive_root=root,
            operation_id=transaction.operation_id,
            schema_inference_receipt_path=receipt_path,
        )
    )

    assert receipt.status == "empty-source"
    assert receipt.consumed_evidence["receipt_path"] == str(receipt_path)
    checkpoint = IndexGenerationStore.for_archive_root(root, repair_anchor=False).load_transaction(
        transaction.operation_id
    )
    assert checkpoint.status == "stale"
    assert checkpoint.error == "rebuild source is empty; resumable transaction cannot continue"


def _replace_root_after_rebuild_lease(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
    moved_root: Path,
) -> None:
    real_enter = RebuildLease.__enter__
    swapped = False

    def swap_after_acquire(lease: RebuildLease) -> RebuildLease:
        nonlocal swapped
        entered = real_enter(lease)
        if not swapped:
            root.rename(moved_root)
            initialize_active_archive_root(root)
            swapped = True
        return entered

    monkeypatch.setattr(RebuildLease, "__enter__", swap_after_acquire)


def test_invalid_resume_refuses_root_replacement_before_marking_transaction_stale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "archive"
    moved_root = tmp_path / "moved-archive"
    _init_empty_source(root)
    transaction = IndexGenerationStore.for_archive_root(root).create_transaction(
        source_snapshot=rebuild_source_evidence_snapshot(root),
        operation_id="invalid-resume-root-replacement",
    )
    transaction_before = (root / ".index-rebuild-transactions" / f"{transaction.operation_id}.json").read_bytes()
    receipt_path = tmp_path / "invalid-receipt.json"
    receipt_path.write_text("{}", encoding="utf-8")
    _replace_root_after_rebuild_lease(monkeypatch, root, moved_root)

    with pytest.raises(ArchiveOwnershipError, match="archive root"):
        rebuild_index_from_source_sync(
            RebuildIndexRequest(
                archive_root=root,
                operation_id=transaction.operation_id,
                schema_inference_receipt_path=receipt_path,
            )
        )

    assert (
        moved_root / ".index-rebuild-transactions" / f"{transaction.operation_id}.json"
    ).read_bytes() == transaction_before


def test_empty_source_resume_refuses_root_replacement_before_retiring_transaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "archive"
    moved_root = tmp_path / "moved-archive"
    _init_empty_source(root)
    receipt_path = write_valid_rebuild_receipt(root, tmp_path / "schema-inference-receipt.json")
    transaction = IndexGenerationStore.for_archive_root(root).create_transaction(
        source_snapshot=rebuild_source_evidence_snapshot(root),
        operation_id="empty-resume-root-replacement",
    )
    transaction_before = (root / ".index-rebuild-transactions" / f"{transaction.operation_id}.json").read_bytes()
    _replace_root_after_rebuild_lease(monkeypatch, root, moved_root)

    with pytest.raises(ArchiveOwnershipError, match="archive root"):
        rebuild_index_from_source_sync(
            RebuildIndexRequest(
                archive_root=root,
                operation_id=transaction.operation_id,
                schema_inference_receipt_path=receipt_path,
            )
        )

    assert (
        moved_root / ".index-rebuild-transactions" / f"{transaction.operation_id}.json"
    ).read_bytes() == transaction_before


def test_empty_source_rebuild_does_not_bypass_archive_ownership(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "archive"
    _init_empty_source(root)

    def refuse_ownership(*args: object, **kwargs: object) -> OwnedArchiveLocation:
        raise ArchiveOwnershipError("empty-source ownership probe")

    monkeypatch.setattr(OwnedArchiveLocation, "acquire", refuse_ownership)

    with pytest.raises(ArchiveOwnershipError, match="empty-source ownership probe"):
        rebuild_index_from_source_sync(RebuildIndexRequest(archive_root=root))
