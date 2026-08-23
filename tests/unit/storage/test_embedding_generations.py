from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from polylogue.storage.embeddings.generations import (
    EmbeddingGenerationError,
    EmbeddingGenerationState,
    EmbeddingGenerationStore,
    ensure_embedding_lifecycle,
)
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier


def _sqlite(path: Path, value: str) -> None:
    initialize_archive_database(path, ArchiveTier.EMBEDDINGS)
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE values_(value TEXT NOT NULL)")
        conn.execute("INSERT INTO values_ VALUES (?)", (value,))


def test_three_replacements_retain_active_and_one_predecessor(tmp_path: Path) -> None:
    store = EmbeddingGenerationStore(tmp_path)
    for number in range(3):
        candidate = tmp_path / f"candidate-{number}.db"
        _sqlite(candidate, str(number))
        store.replace(candidate, owner_id=f"owner-{number}")

    generations = list((tmp_path / ".embeddings-generations").glob("gen-*/generation.json"))
    assert len(generations) == 2
    states = {json.loads(path.read_text(encoding="utf-8"))["state"] for path in generations}
    assert states == {EmbeddingGenerationState.ACTIVE.value, EmbeddingGenerationState.RETAINED.value}
    active = tmp_path / "embeddings.db"
    assert active.is_symlink()
    receipt_files = list((tmp_path / ".embeddings-generations" / "retention-receipts").glob("*.json"))
    assert receipt_files
    latest_receipt = max(
        receipt_files,
        key=lambda path: json.loads(path.read_text(encoding="utf-8"))["promoted_at_ns"],
    )
    receipt = store.load_receipt(json.loads(latest_receipt.read_text(encoding="utf-8"))["promoted_generation_id"])
    assert receipt.retention_boundary == 1
    assert receipt.automatic is True
    assert receipt.reclaimed_generation_ids


def test_pre_lifecycle_active_database_is_retained_on_first_replacement(tmp_path: Path) -> None:
    _sqlite(tmp_path / "embeddings.db", "legacy")
    candidate = tmp_path / "candidate.db"
    _sqlite(candidate, "new")
    EmbeddingGenerationStore(tmp_path).replace(candidate)
    states = {
        json.loads(path.read_text(encoding="utf-8"))["state"]
        for path in (tmp_path / ".embeddings-generations").glob("gen-*/generation.json")
    }
    assert states == {EmbeddingGenerationState.ACTIVE.value, EmbeddingGenerationState.RETAINED.value}


def test_receipt_is_durable_and_legacy_chronology_fails_closed(tmp_path: Path) -> None:
    store = EmbeddingGenerationStore(tmp_path)
    candidate = tmp_path / "candidate.db"
    _sqlite(candidate, "one")
    store.replace(candidate)
    receipt_path = next((tmp_path / ".embeddings-generations" / "retention-receipts").glob("*.json"))
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert payload["promoted_at_ns"] > 0
    payload["promoted_at_ns"] = 0
    receipt_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(EmbeddingGenerationError, match="malformed embedding retention receipt"):
        store.load_receipt(receipt_path.stem)


def test_malformed_receipt_blocks_replacement_before_pointer_swap(tmp_path: Path) -> None:
    store = EmbeddingGenerationStore(tmp_path)
    first = tmp_path / "first.db"
    _sqlite(first, "one")
    store.replace(first)
    receipt_path = next((tmp_path / ".embeddings-generations" / "retention-receipts").glob("*.json"))
    receipt_path.write_text("{broken", encoding="utf-8")
    second = tmp_path / "second.db"
    _sqlite(second, "two")
    active_before = (tmp_path / "embeddings.db").resolve()
    with pytest.raises(EmbeddingGenerationError, match="malformed embedding retention receipt"):
        store.replace(second)
    assert (tmp_path / "embeddings.db").resolve() == active_before


def test_malformed_predecessor_blocks_replacement(tmp_path: Path) -> None:
    store = EmbeddingGenerationStore(tmp_path)
    candidate = tmp_path / "candidate.db"
    _sqlite(candidate, "one")
    store.replace(candidate)
    broken = tmp_path / ".embeddings-generations" / "gen-broken"
    broken.mkdir()
    (broken / "generation.json").write_text("{not json", encoding="utf-8")
    second = tmp_path / "candidate-2.db"
    _sqlite(second, "two")
    with pytest.raises(EmbeddingGenerationError, match="malformed embedding generation metadata"):
        store.replace(second)
    assert (tmp_path / "embeddings.db").resolve() == Path(store.active_path).resolve()


def test_interrupted_promotion_recovers_on_startup(tmp_path: Path) -> None:
    store = EmbeddingGenerationStore(tmp_path)
    candidate = tmp_path / "candidate.db"
    _sqlite(candidate, "one")
    store.replace(candidate)
    generations = list((tmp_path / ".embeddings-generations").glob("gen-*/generation.json"))
    metadata = generations[0]
    payload = json.loads(metadata.read_text(encoding="utf-8"))
    payload["state"] = EmbeddingGenerationState.PROMOTING.value
    metadata.write_text(json.dumps(payload), encoding="utf-8")
    ensure_embedding_lifecycle(tmp_path)
    recovered = json.loads(metadata.read_text(encoding="utf-8"))
    assert recovered["state"] == EmbeddingGenerationState.ACTIVE.value


def test_rejects_symlink_candidate_before_promotion(tmp_path: Path) -> None:
    target = tmp_path / "target.db"
    _sqlite(target, "target")
    candidate = tmp_path / "candidate.db"
    candidate.symlink_to(target)
    with pytest.raises(EmbeddingGenerationError, match="regular file"):
        EmbeddingGenerationStore(tmp_path).replace(candidate)
    assert not (tmp_path / "embeddings.db").exists()


def test_rejects_wrong_tier_and_incomplete_candidate(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.db"
    with sqlite3.connect(candidate) as conn:
        conn.execute("CREATE TABLE sessions(id TEXT)")
    with pytest.raises(EmbeddingGenerationError, match="schema v0"):
        EmbeddingGenerationStore(tmp_path).replace(candidate)


def test_rejects_uncheckpointed_candidate_wal(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.db"
    _sqlite(candidate, "one")
    with sqlite3.connect(candidate) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("INSERT INTO values_ VALUES ('wal')")
        conn.commit()
        assert candidate.with_name("candidate.db-wal").exists()
        with pytest.raises(EmbeddingGenerationError, match="uncheckpointed WAL"):
            EmbeddingGenerationStore(tmp_path).replace(candidate)


def test_legacy_adoption_runs_on_ensure_route(tmp_path: Path) -> None:
    _sqlite(tmp_path / "embeddings.db", "legacy")
    ensure_embedding_lifecycle(tmp_path)
    assert (tmp_path / "embeddings.db").is_symlink()
    metadata = list((tmp_path / ".embeddings-generations").glob("gen-*/generation.json"))
    assert len(metadata) == 1
    assert json.loads(metadata[0].read_text(encoding="utf-8"))["state"] == "active"


def test_legacy_adoption_rejects_sidecars_that_sqlite_cannot_clear(tmp_path: Path) -> None:
    active = tmp_path / "embeddings.db"
    _sqlite(active, "legacy")
    writer = sqlite3.connect(active)
    try:
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("INSERT INTO values_ VALUES ('wal')")
        writer.commit()
        assert active.with_name("embeddings.db-wal").exists()
        with pytest.raises(EmbeddingGenerationError, match="retains SQLite sidecars"):
            ensure_embedding_lifecycle(tmp_path)
    finally:
        writer.close()
    assert active.is_file()
    assert not active.is_symlink()


def test_lifecycle_entrypoint_enters_collector(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    original = EmbeddingGenerationStore.collect

    def collect(self: EmbeddingGenerationStore) -> object:
        calls.append(str(self.archive_root))
        return original(self)

    monkeypatch.setattr(EmbeddingGenerationStore, "collect", collect)
    ensure_embedding_lifecycle(tmp_path)
    assert calls == [str(tmp_path.absolute())]


def test_receipt_id_is_validated_before_path_construction(tmp_path: Path) -> None:
    store = EmbeddingGenerationStore(tmp_path)
    with pytest.raises(EmbeddingGenerationError, match="invalid embedding generation identity"):
        store.load_receipt("../retention-receipts/escape")


def test_generation_and_receipt_roots_reject_symlinks(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    generations = tmp_path / ".embeddings-generations"
    generations.symlink_to(outside, target_is_directory=True)
    with pytest.raises(EmbeddingGenerationError, match="generation root"):
        EmbeddingGenerationStore(tmp_path)


def test_historical_retired_artifacts_are_not_owned(tmp_path: Path) -> None:
    retired = tmp_path / ".embeddings-generations" / "retired-legacy"
    retired.mkdir(parents=True)
    (retired / "generation.json").write_text("not lifecycle metadata", encoding="utf-8")
    ensure_embedding_lifecycle(tmp_path)


def test_first_promotion_pointer_interruption_recovers_candidate(tmp_path: Path) -> None:
    store = EmbeddingGenerationStore(tmp_path)
    candidate = tmp_path / "candidate.db"
    _sqlite(candidate, "one")
    store.replace(candidate)
    active = tmp_path / "embeddings.db"
    active.unlink()
    metadata = next((tmp_path / ".embeddings-generations").glob("gen-*/generation.json"))
    payload = json.loads(metadata.read_text(encoding="utf-8"))
    payload["state"] = EmbeddingGenerationState.PROMOTING.value
    payload["predecessor_generation_id"] = None
    metadata.write_text(json.dumps(payload), encoding="utf-8")
    ensure_embedding_lifecycle(tmp_path)
    assert active.is_symlink()
    assert json.loads(metadata.read_text(encoding="utf-8"))["state"] == EmbeddingGenerationState.ACTIVE.value
