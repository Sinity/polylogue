"""Candidate creation refuses before it allocates, and records what it built.

Anti-vacuity: removing the headroom check from ``create_transaction`` leaves
a generation directory on disk after the refusal test, and dropping the
observation from ``save_transaction`` leaves the receipt's actual peak at zero.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from polylogue.maintenance.candidate_capacity import (
    InsufficientCapacityError,
    read_capacity_receipts,
)
from polylogue.storage.archive_identity import GENERATIONS_DIRNAME
from polylogue.storage.index_generation import IndexGenerationStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier


def _archive(root: Path) -> None:
    for tier in (ArchiveTier.SOURCE, ArchiveTier.USER, ArchiveTier.EMBEDDINGS, ArchiveTier.OPS, ArchiveTier.INDEX):
        initialize_archive_database(root / f"{tier.value}.db", tier)


def _free_space(monkeypatch: pytest.MonkeyPatch, available_bytes: int) -> None:
    monkeypatch.setattr(
        "polylogue.maintenance.candidate_capacity._available_bytes",
        lambda _path: available_bytes,
    )


def test_candidate_creation_refuses_before_allocating_without_headroom(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _archive(tmp_path)
    store = IndexGenerationStore.for_archive_root(tmp_path)
    generations_before = sorted(path.name for path in (tmp_path / GENERATIONS_DIRNAME).glob("gen-*"))
    _free_space(monkeypatch, 0)

    with pytest.raises(InsufficientCapacityError) as refusal:
        store.create_transaction(source_snapshot="source-v1", operation_id="no-room")

    assert refusal.value.projection.shortfall_bytes > 0
    assert sorted(path.name for path in (tmp_path / GENERATIONS_DIRNAME).glob("gen-*")) == generations_before
    assert not store._transaction_path("no-room").exists()
    assert read_capacity_receipts(tmp_path) == ()


def test_candidate_creation_records_prediction_and_observed_peak(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _archive(tmp_path)
    store = IndexGenerationStore.for_archive_root(tmp_path)
    _free_space(monkeypatch, 64 * 1024 * 1024 * 1024)

    transaction = store.create_transaction(source_snapshot="source-v1", operation_id="with-room")
    receipts = read_capacity_receipts(tmp_path)

    assert [receipt.operation_id for receipt in receipts] == ["with-room"]
    receipt = receipts[0]
    assert receipt.predicted_peak_allocated_bytes > 0
    assert receipt.actual_peak_index_bytes > 0
    assert receipt.actual_peak_allocated_bytes == receipt.baseline_allocated_bytes + receipt.actual_peak_index_bytes
    assert receipt.observations >= 1

    store.checkpoint_transaction(transaction, status="paused")
    assert read_capacity_receipts(tmp_path)[0].observations >= 2


def test_capacity_observation_failure_does_not_fail_a_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _archive(tmp_path)
    store = IndexGenerationStore.for_archive_root(tmp_path)
    _free_space(monkeypatch, 64 * 1024 * 1024 * 1024)
    transaction = store.create_transaction(source_snapshot="source-v1", operation_id="observation-fails")

    def explode(*_args: object, **_kwargs: object) -> None:
        raise OSError("receipt directory is gone")

    monkeypatch.setattr("polylogue.maintenance.candidate_capacity.record_capacity_observation", explode)
    checkpointed = store.checkpoint_transaction(transaction, status="paused")

    assert checkpointed.status == "paused"
