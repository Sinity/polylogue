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

import json
from pathlib import Path

import pytest

from polylogue.daemon.bulk_rebuild import resolve_or_start_daemon_bulk_rebuild_transaction
from polylogue.maintenance.schema_inference_gate import rebuild_source_revision_snapshot
from polylogue.storage.archive_identity import ArchiveLocation, ArchiveOwnershipError, OwnedArchiveLocation
from polylogue.storage.index_generation import IndexGenerationStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from tests.infra.rebuild_receipt import write_valid_rebuild_receipt


def _init_empty_source(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    initialize_archive_database(root / "source.db", ArchiveTier.SOURCE)


def _interpose_receipt_mutation(monkeypatch: pytest.MonkeyPatch, receipt_path: Path, *, expire: bool) -> None:
    original_acquire = OwnedArchiveLocation.acquire

    def acquire_after_mutation(
        cls: type[OwnedArchiveLocation], /, location: object, **kwargs: object
    ) -> OwnedArchiveLocation:
        owned = original_acquire(location, **kwargs)  # type: ignore[arg-type]
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
        if expire:
            payload["generated_at"] = "2000-01-01T00:00:00Z"
        else:
            payload["source_snapshot"] = "post-preflight-source-drift"
        receipt_path.write_text(json.dumps(payload), encoding="utf-8")
        return owned

    monkeypatch.setattr(OwnedArchiveLocation, "acquire", classmethod(acquire_after_mutation))


def test_daemon_bulk_rebuild_refuses_when_archive_location_already_owned(tmp_path: Path) -> None:
    """A concurrent holder of the archive-location ownership lock must block
    the daemon's bulk-rebuild transaction resolve/retire path before any
    generation directory or transaction record is created -- mirroring
    ``test_rebuild_refuses_when_archive_location_already_owned`` for the
    offline rebuild entry point.
    """
    root = tmp_path / "archive"
    _init_empty_source(root)
    receipt_path = write_valid_rebuild_receipt(root, tmp_path / "receipt.json")
    location = ArchiveLocation.resolve(root)
    owned = OwnedArchiveLocation.acquire(location, owner_id="concurrent-campaign")
    try:
        with pytest.raises(ArchiveOwnershipError):
            resolve_or_start_daemon_bulk_rebuild_transaction(root, schema_inference_receipt_path=receipt_path)
        # Failure happened before any generation/transaction bookkeeping was created.
        assert not (root / ".index-generations").exists()
        assert not (root / ".index-rebuild-transactions").exists()
    finally:
        owned.release()

    # Releasing the concurrent holder's ownership lets the daemon proceed.
    transaction = resolve_or_start_daemon_bulk_rebuild_transaction(root, schema_inference_receipt_path=receipt_path)
    assert transaction.status == "running"


@pytest.mark.parametrize("expire", [False, True], ids=["external-drift", "receipt-expiry"])
def test_daemon_post_preflight_receipt_change_fails_before_candidate_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, expire: bool
) -> None:
    root = tmp_path / "archive"
    _init_empty_source(root)
    receipt_path = write_valid_rebuild_receipt(root, tmp_path / "receipt.json")
    _interpose_receipt_mutation(monkeypatch, receipt_path, expire=expire)

    with pytest.raises(RuntimeError, match="daemon bulk rebuild schema-inference preflight gate failed"):
        resolve_or_start_daemon_bulk_rebuild_transaction(root, schema_inference_receipt_path=receipt_path)

    assert not (root / ".index-generations").exists()
    assert not (root / ".index-rebuild-transactions").exists()


def test_daemon_post_preflight_receipt_change_does_not_retire_terminal_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "archive"
    _init_empty_source(root)
    receipt_path = write_valid_rebuild_receipt(root, tmp_path / "receipt.json")
    store = IndexGenerationStore.for_archive_root(root)
    transaction = store.create_transaction(
        source_snapshot=rebuild_source_revision_snapshot(root),
        operation_id="daemon-bulk-rebuild",
    )
    store.checkpoint_transaction(transaction, status="failed")
    generation_id = transaction.generation_id
    _interpose_receipt_mutation(monkeypatch, receipt_path, expire=False)

    with pytest.raises(RuntimeError, match="daemon bulk rebuild schema-inference preflight gate failed"):
        resolve_or_start_daemon_bulk_rebuild_transaction(root, schema_inference_receipt_path=receipt_path)

    retained_transaction = store.load_transaction("daemon-bulk-rebuild")
    assert retained_transaction.status == "failed"
    assert retained_transaction.generation_id == generation_id
    assert store.load(generation_id).state == "inactive"


def test_daemon_bulk_rebuild_releases_ownership_lock_after_resolving(tmp_path: Path) -> None:
    """The ownership lock must not be left held after resolving/starting a
    transaction, so a subsequent maintenance/campaign writer can still
    acquire it.
    """
    root = tmp_path / "archive"
    _init_empty_source(root)
    receipt_path = write_valid_rebuild_receipt(root, tmp_path / "receipt.json")

    resolve_or_start_daemon_bulk_rebuild_transaction(root, schema_inference_receipt_path=receipt_path)

    location = ArchiveLocation.resolve(root)
    owned = OwnedArchiveLocation.acquire(location, owner_id="post-resolve-probe")
    try:
        assert (root / ".archive-ownership.lock").exists()
    finally:
        owned.release()
