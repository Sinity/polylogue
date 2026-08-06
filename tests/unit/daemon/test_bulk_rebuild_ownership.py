"""``resolve_or_start_daemon_bulk_rebuild_transaction`` must prove archive-
location ownership before mutating any generation directory or transaction
record (polylogue-ovme.2.1, extending polylogue-ovme.2 AC3 to the online/
daemon-driven bulk-rebuild path).

Before this change, ``polylogue.maintenance.rebuild_index.rebuild_index_from_source``
(the offline rebuild entry point) acquired ``OwnedArchiveLocation`` before
touching disk, but the daemon's own bulk-rebuild transaction resolve/retire
logic in ``polylogue.daemon.bulk_rebuild`` constructed an
``IndexGenerationStore`` directly and discarded/created generations and
transaction records with no ownership proof at all -- a concurrent offline
rebuild or devtools campaign holding the archive-location ownership lock
could race the daemon's own bulk-rebuild bookkeeping undetected.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import cast

import pytest

from polylogue.daemon.bulk_rebuild import resolve_or_start_daemon_bulk_rebuild_transaction
from polylogue.maintenance.rebuild_index import RebuildSchemaCurrencyError
from polylogue.storage.archive_identity import ArchiveLocation, ArchiveOwnershipError, OwnedArchiveLocation
from polylogue.storage.archive_readiness import probe_archive_tier
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.migration_runner import DURABLE_MIGRATION_TIERS
from tests.infra.rebuild_receipt import write_valid_rebuild_receipt


def _init_empty_source(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for tier in sorted(DURABLE_MIGRATION_TIERS, key=lambda item: item.value):
        initialize_archive_database(root / f"{tier.value}.db", tier)


def test_daemon_bulk_rebuild_rejects_schema_mismatch_before_transaction_bookkeeping(tmp_path: Path) -> None:
    """The daemon's direct transaction entry cannot bypass the shared gate."""
    root = tmp_path / "archive"
    _init_empty_source(root)
    source_probe = probe_archive_tier(ArchiveTier.SOURCE, root / "source.db")
    with sqlite3.connect(root / "source.db") as conn:
        conn.execute(f"PRAGMA user_version = {source_probe.expected_user_version + 1}")

    with pytest.raises(RebuildSchemaCurrencyError) as exc_info:
        resolve_or_start_daemon_bulk_rebuild_transaction(root)

    blocking_tiers = cast(list[dict[str, object]], exc_info.value.diagnostic["blocking_tiers"])
    assert blocking_tiers[0]["tier"] == "source"
    assert not (root / ".index-generations").exists()
    assert not (root / ".index-rebuild-transactions").exists()


def test_daemon_bulk_rebuild_rechecks_schema_currency_after_ownership(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A durable migration while lock acquisition waits must block bookkeeping.

    Production dependency: the second shared currency probe after ownership.
    Mutation: removing that probe creates generation bookkeeping after the
    injected audit migration and makes this test fail.
    """
    from polylogue.daemon import bulk_rebuild

    root = tmp_path / "archive"
    _init_empty_source(root)
    receipt = write_valid_rebuild_receipt(root, tmp_path / "schema-inference-gate-receipt.json")
    real_assert = bulk_rebuild.assert_owns_archive_location

    def mutate_audit_after_ownership(owned: OwnedArchiveLocation, location: ArchiveLocation) -> None:
        real_assert(owned, location)
        audit_probe = probe_archive_tier(ArchiveTier.AUDIT, root / "audit.db")
        with sqlite3.connect(root / "audit.db") as conn:
            conn.execute(f"PRAGMA user_version = {audit_probe.expected_user_version + 1}")

    monkeypatch.setattr(bulk_rebuild, "assert_owns_archive_location", mutate_audit_after_ownership)

    with pytest.raises(RebuildSchemaCurrencyError) as exc_info:
        resolve_or_start_daemon_bulk_rebuild_transaction(root, schema_inference_receipt_path=receipt)

    blocking_tiers = cast(list[dict[str, object]], exc_info.value.diagnostic["blocking_tiers"])
    assert blocking_tiers[0]["tier"] == "audit"
    assert not (root / ".index-generations").exists()
    assert not (root / ".index-rebuild-transactions").exists()


def test_daemon_bulk_rebuild_refuses_when_archive_location_already_owned(tmp_path: Path) -> None:
    """A concurrent holder of the archive-location ownership lock must block
    the daemon's bulk-rebuild transaction resolve/retire path before any
    generation directory or transaction record is created -- mirroring
    ``test_rebuild_refuses_when_archive_location_already_owned`` for the
    offline rebuild entry point.
    """
    root = tmp_path / "archive"
    _init_empty_source(root)
    location = ArchiveLocation.resolve(root)
    owned = OwnedArchiveLocation.acquire(location, owner_id="concurrent-campaign")
    try:
        with pytest.raises(ArchiveOwnershipError):
            resolve_or_start_daemon_bulk_rebuild_transaction(root)
        # Failure happened before any generation/transaction bookkeeping was created.
        assert not (root / ".index-generations").exists()
        assert not (root / ".index-rebuild-transactions").exists()
    finally:
        owned.release()

    # Releasing the concurrent holder's ownership lets the daemon proceed.
    transaction = resolve_or_start_daemon_bulk_rebuild_transaction(root)
    assert transaction.status == "running"


def test_daemon_bulk_rebuild_releases_ownership_lock_after_resolving(tmp_path: Path) -> None:
    """The ownership lock must not be left held after resolving/starting a
    transaction, so a subsequent maintenance/campaign writer can still
    acquire it.
    """
    root = tmp_path / "archive"
    _init_empty_source(root)

    resolve_or_start_daemon_bulk_rebuild_transaction(root)

    location = ArchiveLocation.resolve(root)
    owned = OwnedArchiveLocation.acquire(location, owner_id="post-resolve-probe")
    try:
        assert (root / ".archive-ownership.lock").exists()
    finally:
        owned.release()
