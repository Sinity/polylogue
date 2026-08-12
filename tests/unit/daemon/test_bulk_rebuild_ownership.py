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
from polylogue.maintenance.schema_inference_gate import run_schema_inference_gate
from polylogue.storage.archive_identity import (
    ArchiveLocation,
    ArchiveOwnershipError,
    OwnedArchiveLocation,
    assert_owns_archive_location,
)
from polylogue.storage.archive_readiness import probe_archive_tier
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from tests.infra.schema_inference import seed_schema_inference_archive


def _init_empty_source(root: Path) -> Path:
    return seed_schema_inference_archive(root)


def _schema_inference_receipt(root: Path, ground_truth: Path, tmp_path: Path) -> Path:
    receipt = tmp_path / f"{root.name}-schema-inference-gate-receipt.json"
    result = run_schema_inference_gate(
        root,
        receipt_path=receipt,
        ground_truth_roots={"codex-session": (ground_truth,)},
    )
    assert result.passed, result.payload["pass_fail_reasons"]
    return receipt


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
    ground_truth = _init_empty_source(root)
    receipt = _schema_inference_receipt(root, ground_truth, tmp_path)
    real_assert = assert_owns_archive_location

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


def test_daemon_bulk_rebuild_refuses_when_archive_location_already_owned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A concurrent holder of the archive-location ownership lock must block
    the daemon's bulk-rebuild transaction resolve/retire path before any
    generation directory or transaction record is created -- mirroring
    ``test_rebuild_refuses_when_archive_location_already_owned`` for the
    offline rebuild entry point.
    """
    root = tmp_path / "archive"
    ground_truth = _init_empty_source(root)
    receipt = _schema_inference_receipt(root, ground_truth, tmp_path)
    # This test proves ownership around transaction bookkeeping. The schema
    # gate above is real; the source-admission route is separately covered
    # and the helper's tiny gate corpus deliberately has no replay census.
    from polylogue.daemon import bulk_rebuild

    monkeypatch.setattr(bulk_rebuild, "validate_rebuild_source_admission", lambda *_args: None)
    location = ArchiveLocation.resolve(root)
    owned = OwnedArchiveLocation.acquire(location, owner_id="concurrent-campaign")
    try:
        with pytest.raises(ArchiveOwnershipError):
            resolve_or_start_daemon_bulk_rebuild_transaction(root, schema_inference_receipt_path=receipt)
        # Failure happened before any generation/transaction bookkeeping was created.
        assert not (root / ".index-generations").exists()
        assert not (root / ".index-rebuild-transactions").exists()
    finally:
        owned.release()

    # Releasing the concurrent holder's ownership lets the daemon proceed.
    transaction = resolve_or_start_daemon_bulk_rebuild_transaction(root, schema_inference_receipt_path=receipt)
    assert transaction.status == "running"


def test_daemon_bulk_rebuild_releases_ownership_lock_after_resolving(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ownership lock must not be left held after resolving/starting a
    transaction, so a subsequent maintenance/campaign writer can still
    acquire it.
    """
    root = tmp_path / "archive"
    ground_truth = _init_empty_source(root)
    receipt = _schema_inference_receipt(root, ground_truth, tmp_path)
    from polylogue.daemon import bulk_rebuild

    monkeypatch.setattr(bulk_rebuild, "validate_rebuild_source_admission", lambda *_args: None)

    resolve_or_start_daemon_bulk_rebuild_transaction(root, schema_inference_receipt_path=receipt)

    location = ArchiveLocation.resolve(root)
    owned = OwnedArchiveLocation.acquire(location, owner_id="post-resolve-probe")
    try:
        assert (root / ".archive-ownership.lock").exists()
    finally:
        owned.release()
