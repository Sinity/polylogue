"""Owned inactive generations may write only their derived index tier."""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import pytest

from polylogue.core.enums import Provider
from polylogue.maintenance.rebuild_index import RebuildIndexRequest, rebuild_index_from_source_sync
from polylogue.sources.revision_backfill import census_historical_revision_evidence
from polylogue.storage.index_generation import IndexGenerationStore, source_revision_snapshot
from polylogue.storage.sqlite.archive_tiers.archive import (
    ArchiveStore,
    InactiveCandidateDurableWriteError,
)
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from polylogue.storage.sqlite.archive_tiers.revision_governance import FrozenSourceRemediationRequiredError
from polylogue.storage.sqlite.durable_change_train import DurableChangeTrainError
from tests.infra.rebuild_receipt import write_valid_rebuild_receipt
from tests.infra.revision_backfill_benchmark import build_independent_raw_corpus


def _file_evidence(path: Path) -> tuple[int, int, str]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, hashlib.sha256(path.read_bytes()).hexdigest()


def _blob_evidence(root: Path) -> tuple[tuple[str, int, str], ...]:
    return tuple(
        (str(path.relative_to(root)), path.stat().st_size, hashlib.sha256(path.read_bytes()).hexdigest())
        for path in sorted(root.rglob("*"))
        if path.is_file()
    )


def _prepare_frozen_source(root: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    build_independent_raw_corpus(root, raw_count=1, avg_payload_bytes=1_000)
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(root))
    census = census_historical_revision_evidence(root)
    assert census.scanned == 1
    assert census.classified_full == 1
    with sqlite3.connect(root / "source.db") as source:
        source.execute(
            """
            UPDATE raw_sessions
            SET revision_authority = 'byte_proven', baseline_raw_id = raw_id,
                predecessor_raw_id = NULL, acquisition_generation = 0
            """
        )
        source.commit()
    return write_valid_rebuild_receipt(root, root.parent / "schema-inference-receipt.json")


def test_real_no_promote_candidate_preserves_frozen_durable_tiers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "archive"
    receipt_path = _prepare_frozen_source(root, monkeypatch)
    generation_store = IndexGenerationStore.for_archive_root(root)
    active_target_before = generation_store.active_pointer.resolve(strict=True)
    active_index_before = _file_evidence(active_target_before)
    source_before = _file_evidence(root / "source.db")
    user_before = _file_evidence(root / "user.db")
    blobs_before = _blob_evidence(root / "blob")

    result = rebuild_index_from_source_sync(
        RebuildIndexRequest(
            archive_root=root,
            schema_inference_receipt_path=receipt_path,
            promote=False,
        )
    )

    assert result.status == "replayed"
    assert result.transaction is not None
    assert result.transaction["status"] == "ready"
    generation = generation_store.load(str(result.transaction["generation_id"]))
    assert generation.state == "inactive"
    assert generation_store.active_pointer.resolve(strict=True) == active_target_before
    assert _file_evidence(active_target_before) == active_index_before
    assert _file_evidence(root / "source.db") == source_before
    assert _file_evidence(root / "user.db") == user_before
    assert _blob_evidence(root / "blob") == blobs_before
    with sqlite3.connect(f"file:{generation.index_path}?mode=ro", uri=True) as candidate:
        assert candidate.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 1


def test_owned_candidate_refuses_source_user_and_blob_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "archive"
    _prepare_frozen_source(root, monkeypatch)
    generation_store = IndexGenerationStore.for_archive_root(root)
    generation = generation_store.create(source_snapshot=source_revision_snapshot(root))
    generation_root = Path(generation.index_path).parent
    source_before = _file_evidence(root / "source.db")
    user_before = _file_evidence(root / "user.db")
    blobs_before = _blob_evidence(root / "blob")

    with ArchiveStore.open_owned_inactive_generation(
        generation_root,
        generation_id=generation.generation_id,
        owner_id=generation.owner_id,
    ) as candidate:
        candidate._conn.execute("CREATE TABLE candidate_index_probe (value INTEGER) STRICT")
        candidate.commit()
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            candidate._ensure_source_conn().execute("UPDATE raw_sessions SET parse_error = 'candidate-write'")
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            candidate._conn.execute("CREATE TABLE user_tier.candidate_user_probe (value INTEGER) STRICT")
        assert candidate._blob_publisher is not None
        with pytest.raises(InactiveCandidateDurableWriteError, match="may not publish"):
            candidate._blob_publisher.write_from_bytes(b"candidate-write")
        with pytest.raises(InactiveCandidateDurableWriteError, match="may not publish"):
            candidate.write_raw_payload(
                provider=Provider.CODEX,
                payload=b"candidate-write",
                source_path="candidate-write.jsonl",
                acquired_at_ms=1,
            )

    assert _file_evidence(root / "source.db") == source_before
    assert _file_evidence(root / "user.db") == user_before
    assert _blob_evidence(root / "blob") == blobs_before
    with sqlite3.connect(generation.index_path) as candidate_index:
        assert candidate_index.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'candidate_index_probe'"
        ).fetchone() == (1,)


def test_candidate_requires_current_parser_census_before_generation_readiness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "archive"
    build_independent_raw_corpus(root, raw_count=1, avg_payload_bytes=1_000)
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(root))
    receipt_path = write_valid_rebuild_receipt(root, root.parent / "schema-inference-receipt.json")

    with pytest.raises(FrozenSourceRemediationRequiredError, match="complete current-parser source census"):
        rebuild_index_from_source_sync(
            RebuildIndexRequest(
                archive_root=root,
                schema_inference_receipt_path=receipt_path,
                promote=False,
            )
        )

    assert not list((root / ".index-generations").glob("gen-*"))


def test_candidate_rejects_authority_drift_in_frozen_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "archive"
    _prepare_frozen_source(root, monkeypatch)
    with sqlite3.connect(root / "source.db") as source:
        source.execute("UPDATE raw_sessions SET revision_authority = 'asserted', baseline_raw_id = NULL")
        source.commit()
    receipt_path = write_valid_rebuild_receipt(root, root.parent / "post-drift-receipt.json")

    with pytest.raises(FrozenSourceRemediationRequiredError, match="re-derived different byte authority"):
        rebuild_index_from_source_sync(
            RebuildIndexRequest(
                archive_root=root,
                schema_inference_receipt_path=receipt_path,
                promote=False,
            )
        )


def test_active_bootstrap_still_rejects_candidate_durable_symlinks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "archive"
    _prepare_frozen_source(root, monkeypatch)
    generation_store = IndexGenerationStore.for_archive_root(root)
    generation = generation_store.create(source_snapshot=source_revision_snapshot(root))

    with pytest.raises(DurableChangeTrainError, match="unsafe file"):
        initialize_active_archive_root(Path(generation.index_path).parent)
