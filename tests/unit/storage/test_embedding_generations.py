from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any, cast

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


def test_reclamation_resume_removes_only_planned_renamed_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = EmbeddingGenerationStore(tmp_path)
    for number in range(2):
        candidate = tmp_path / f"candidate-{number}.db"
        _sqlite(candidate, str(number))
        store.replace(candidate, owner_id=f"owner-{number}")

    original_rmtree = __import__("shutil").rmtree

    def fail_once(path: Path, *args: object, **kwargs: object) -> None:
        monkeypatch.setattr("shutil.rmtree", original_rmtree)
        raise OSError("simulated crash after rename-to-trash")

    monkeypatch.setattr("polylogue.storage.embeddings.generations.shutil.rmtree", fail_once)
    candidate = tmp_path / "candidate-2.db"
    _sqlite(candidate, "2")
    with pytest.raises(OSError, match="simulated crash"):
        store.replace(candidate, owner_id="owner-2")
    retired = list((tmp_path / ".embeddings-generations").glob("retired-gen-*"))
    assert len(retired) == 1

    ensure_embedding_lifecycle(tmp_path)
    assert not list((tmp_path / ".embeddings-generations").glob("retired-gen-*"))


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


def test_generation_admission_does_not_require_legacy_message_refs(tmp_path: Path) -> None:
    """The generation contract is valid without the retired ref ledger."""
    candidate = tmp_path / "candidate.db"
    _sqlite(candidate, "without-refs")
    with sqlite3.connect(candidate) as conn:
        conn.execute("DROP INDEX idx_message_embedding_refs_hash")
        conn.execute("DROP INDEX idx_message_embedding_refs_session")
        conn.execute("DROP TABLE message_embedding_refs")

    EmbeddingGenerationStore(tmp_path).replace(candidate)


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


def test_generation_metadata_must_match_published_database_contract(tmp_path: Path) -> None:
    """A self-consistent but false metadata contract cannot authorize reuse.

    Anti-vacuity: removing the database-contract comparison lets a changed
    recipe or membership digest pass lifecycle collection even though the
    published SQLite bytes never changed.
    """
    store = EmbeddingGenerationStore(tmp_path)
    candidate = tmp_path / "candidate.db"
    _sqlite(candidate, "one")
    store.replace(candidate)

    metadata_path = next((tmp_path / ".embeddings-generations").glob("gen-*/generation.json"))
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    payload["recipe_hash"] = "f" * 64
    payload["membership_digest"] = "e" * 64
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(EmbeddingGenerationError, match="malformed embedding generation metadata"):
        store.collect()


def test_collection_preserves_accepted_and_in_progress_inventory_members(tmp_path: Path) -> None:
    """Retention cannot reclaim candidates protected by lifecycle state."""
    store = EmbeddingGenerationStore(tmp_path)
    for number in range(3):
        candidate = tmp_path / f"candidate-{number}.db"
        _sqlite(candidate, str(number))
        store.replace(candidate, owner_id=f"owner-{number}")

    retained_path = next(
        path
        for path in (tmp_path / ".embeddings-generations").glob("gen-*/generation.json")
        if json.loads(path.read_text(encoding="utf-8"))["state"] == "retained"
    )
    payload = json.loads(retained_path.read_text(encoding="utf-8"))
    payload["state"] = "accepted"
    retained_path.write_text(json.dumps(payload), encoding="utf-8")

    candidate = tmp_path / "candidate-3.db"
    _sqlite(candidate, "3")
    store.replace(candidate, owner_id="owner-3")
    retained_path = next(
        path
        for path in (tmp_path / ".embeddings-generations").glob("gen-*/generation.json")
        if json.loads(path.read_text(encoding="utf-8"))["state"] == "retained"
    )
    payload = json.loads(retained_path.read_text(encoding="utf-8"))
    payload["state"] = "in_progress"
    retained_path.write_text(json.dumps(payload), encoding="utf-8")

    store.collect()

    states = {
        json.loads(path.read_text(encoding="utf-8"))["state"]
        for path in (tmp_path / ".embeddings-generations").glob("gen-*/generation.json")
    }
    assert {"accepted", "in_progress", "active"} <= states


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


def test_admitted_binding_rejects_active_pointer_swap_without_publication(tmp_path: Path) -> None:
    store = EmbeddingGenerationStore(tmp_path)
    candidate = tmp_path / "candidate.db"
    _sqlite(candidate, "one")
    store.replace(candidate)
    with store.writer_lock() as binding:
        active = tmp_path / "embeddings.db"
        replacement = tmp_path / "replacement.db"
        _sqlite(replacement, "hostile")
        temporary = tmp_path / ".embeddings.db.hostile"
        temporary.symlink_to(replacement)
        temporary.replace(active)
        with pytest.raises(EmbeddingGenerationError, match="active pointer was replaced"):
            store.assert_binding(binding)


def test_admitted_binding_rejects_archive_root_replacement(tmp_path: Path) -> None:
    store = EmbeddingGenerationStore(tmp_path)
    candidate = tmp_path / "candidate.db"
    _sqlite(candidate, "one")
    store.replace(candidate)
    with store.writer_lock() as binding:
        moved = tmp_path.with_name(tmp_path.name + ".moved")
        tmp_path.rename(moved)
        tmp_path.mkdir()
        try:
            with pytest.raises(EmbeddingGenerationError, match="archive root was replaced"):
                store.assert_binding(binding)
        finally:
            tmp_path.rmdir()
            moved.rename(tmp_path)


def test_receipt_root_identity_is_authenticated(tmp_path: Path) -> None:
    store = EmbeddingGenerationStore(tmp_path)
    candidate = tmp_path / "candidate.db"
    _sqlite(candidate, "one")
    store.replace(candidate)
    receipt_path = next((tmp_path / ".embeddings-generations" / "retention-receipts").glob("*.json"))
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    payload["archive_root_identity"] = [0, 0]
    receipt_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(EmbeddingGenerationError, match="malformed embedding retention receipt"):
        store.load_receipt(receipt_path.stem)


def _meta_row(
    conn: sqlite3.Connection,
    address: bytes,
    *,
    model: str,
    recipe: bytes,
    dimension: int = 1024,
    output_contract: bytes = b"\x07" * 32,
) -> None:
    conn.execute(
        "INSERT INTO message_embeddings_meta (vector_derivation_hash, model, dimension, embedded_at_ms, "
        "recipe_hash, output_contract_hash) VALUES (?, ?, ?, 0, ?, ?)",
        (address, model, dimension, recipe, output_contract),
    )


def test_membership_accepts_mixed_recipe_labels_but_not_mixed_models(tmp_path: Path) -> None:
    """Vectors carried across a reindex keep their old recipe label beside new ones.

    Anti-vacuity: restoring the recipe-uniformity check makes the first
    ``replace`` raise; dropping the model check makes the second one pass.
    """
    store = EmbeddingGenerationStore(tmp_path)
    mixed_labels = tmp_path / "mixed-labels.db"
    initialize_archive_database(mixed_labels, ArchiveTier.EMBEDDINGS)
    with sqlite3.connect(mixed_labels) as conn:
        _meta_row(conn, b"\x01" * 32, model="voyage-4", recipe=b"\x0a" * 32)
        _meta_row(conn, b"\x02" * 32, model="voyage-4", recipe=b"\x0b" * 32)
    store.replace(mixed_labels)
    metadata_path = next((tmp_path / ".embeddings-generations").glob("gen-*/generation.json"))
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert payload["recipe_hash"] not in {(b"\x0a" * 32).hex(), (b"\x0b" * 32).hex()}
    assert re.fullmatch(r"mixed:[0-9a-f]{64}", payload["recipe_hash"])
    store.collect()

    mixed_models = tmp_path / "mixed-models.db"
    initialize_archive_database(mixed_models, ArchiveTier.EMBEDDINGS)
    with sqlite3.connect(mixed_models) as conn:
        _meta_row(conn, b"\x01" * 32, model="voyage-4", recipe=b"\x0a" * 32)
        _meta_row(conn, b"\x02" * 32, model="voyage-4-lite", recipe=b"\x0a" * 32)
    with pytest.raises(EmbeddingGenerationError, match="mixed vector contracts"):
        store.replace(mixed_models)


@pytest.mark.parametrize("axis", ["model", "output_contract"], ids=["model", "output"])
def test_membership_rejects_each_mixed_vector_contract_axis_independently(tmp_path: Path, axis: str) -> None:
    """Only recipe labels may vary while one vector contract is admitted.

    The current embeddings DDL enforces dimension 1024 at insertion time;
    production admission still checks dimensions independently for databases
    from a compatible schema regime.
    """
    store = EmbeddingGenerationStore(tmp_path)
    candidate = tmp_path / f"mixed-{axis}.db"
    initialize_archive_database(candidate, ArchiveTier.EMBEDDINGS)
    first_model, first_dimension, first_output = "voyage-4", 1024, b"\x07" * 32
    second_model, second_dimension, second_output = "voyage-4", 1024, b"\x07" * 32
    if axis == "model":
        second_model = "voyage-4-lite"
    else:
        second_output = b"\x08" * 32
    with sqlite3.connect(candidate) as conn:
        _meta_row(
            conn,
            b"\x01" * 32,
            model=first_model,
            recipe=b"\x0a" * 32,
            dimension=first_dimension,
            output_contract=first_output,
        )
        _meta_row(
            conn,
            b"\x02" * 32,
            model=second_model,
            recipe=b"\x0a" * 32,
            dimension=second_dimension,
            output_contract=second_output,
        )

    with pytest.raises(EmbeddingGenerationError, match="mixed vector contracts"):
        store.replace(candidate)


# ── Pointer replacement during lease-free computation (polylogue-c0l7n) ─────


def test_generation_replaced_during_provider_call_rejects_the_publication(tmp_path: Path) -> None:
    """Vectors computed against a retired generation never reach its replacement.

    The provider call holds no generation lock by design, so the active pointer
    can legitimately move while an attempt is in flight. Publication re-acquires
    the lock and asserts the binding it reserved against, which is what stops a
    window -- or its failure receipt -- from landing in a generation it was
    never computed for.

    Anti-vacuity: publish under the binding captured at reservation instead of
    re-asserting a fresh one, and the replacement database below gains both the
    vector rows and a failure row for a session it never embedded.
    """
    from polylogue.archive.message.roles import Role
    from polylogue.config import load_polylogue_config
    from polylogue.core.enums import BlockType, MaterialOrigin, Provider
    from polylogue.sources.parsers.base import ParsedSession
    from polylogue.sources.parsers.base_models import ParsedContentBlock, ParsedMessage
    from polylogue.storage.embeddings.materialization import embed_archive_session_sync
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from polylogue.storage.sqlite.sqlite_vec_extension import try_load_sqlite_vec
    from tests.infra.live_ingest import write_index_session

    root = tmp_path / "archive"
    text = "Prose whose generation is retired mid-flight."
    with ArchiveStore(root) as archive:
        session_id = write_index_session(
            archive,
            ParsedSession(
                source_name=Provider.CODEX,
                provider_session_id="generation-moved",
                messages=[
                    ParsedMessage(
                        provider_message_id="m1",
                        role=Role.USER,
                        text=text,
                        blocks=[ParsedContentBlock(type=BlockType.TEXT, text=text)],
                        material_origin=MaterialOrigin.HUMAN_AUTHORED,
                    )
                ],
            ),
        )
    index_db = root / "index.db"
    embeddings_db = root / "embeddings.db"
    initialize_archive_database(embeddings_db, ArchiveTier.EMBEDDINGS)
    probe = sqlite3.connect(embeddings_db)
    loaded, error = try_load_sqlite_vec(probe)
    probe.close()
    if not loaded:
        pytest.skip(str(error) if error else "sqlite-vec extension is unavailable")

    store = EmbeddingGenerationStore(root, active_path=embeddings_db)
    configured_model = load_polylogue_config().embedding_model

    class _PointerMovingProvider:
        model = configured_model
        dimension = 1024

        def __init__(self) -> None:
            self.calls = 0

        def _get_embeddings(self, texts: list[str], input_type: str = "document") -> list[list[float]]:
            self.calls += 1
            replacement = root / "replacement.db"
            initialize_archive_database(replacement, ArchiveTier.EMBEDDINGS)
            store.replace(replacement, owner_id="replacement-owner")
            return [[0.25] * self.dimension for _ in texts]

    provider = _PointerMovingProvider()
    outcome = embed_archive_session_sync(index_db, cast(Any, provider), session_id)

    assert provider.calls == 1
    assert outcome.status == "error"

    with sqlite3.connect(embeddings_db) as conn:
        vectors = conn.execute(
            "SELECT COUNT(*) FROM message_embedding_refs WHERE session_id = ?", (session_id,)
        ).fetchone()[0]
        failures = conn.execute(
            "SELECT COUNT(*) FROM embedding_failures WHERE session_id = ?", (session_id,)
        ).fetchone()[0]
    assert vectors == 0, "a window computed against the retired generation must not land in its replacement"
    assert failures == 0, "the replacement generation must not inherit a failure receipt it never earned"


def _generation_lock_is_free(archive_root: Path) -> bool:
    """Whether the embedding generation flock is currently unheld.

    ``flock`` is owned by an open file description, so a second descriptor in
    this same process contends exactly as another process would. That makes
    this a real observation of the lock, not a restatement of the code.
    """
    import fcntl
    import os

    lock_path = archive_root / ".embeddings-generations" / ".lifecycle.lock"
    if not lock_path.exists():
        return False
    fd = os.open(lock_path, os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return False
    else:
        fcntl.flock(fd, fcntl.LOCK_UN)
        return True
    finally:
        os.close(fd)


def test_generation_lock_is_free_while_the_provider_computes(tmp_path: Path) -> None:
    """The real archive embed route releases the generation lock before embedding.

    Every embedding write is still serialized by this lock; what changed is that
    the provider round trip is no longer inside it, so a promotion, a retention
    pass, or another session's publication is not queued behind the network.

    Anti-vacuity: hold ``writer_lock`` across the provider call again (the
    pre-split shape of ``embed_archive_session_sync``) and the probe below
    reports the lock held. The probe returns False when the lock file is absent,
    so a route that never took the lock at all also fails this test.
    """
    from polylogue.archive.message.roles import Role
    from polylogue.config import load_polylogue_config
    from polylogue.core.enums import BlockType, MaterialOrigin, Provider
    from polylogue.sources.parsers.base import ParsedSession
    from polylogue.sources.parsers.base_models import ParsedContentBlock, ParsedMessage
    from polylogue.storage.embeddings.materialization import embed_archive_session_sync
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from polylogue.storage.sqlite.sqlite_vec_extension import try_load_sqlite_vec
    from tests.infra.live_ingest import write_index_session

    root = tmp_path / "archive"
    text = "Prose embedded while the generation lock must be free."
    with ArchiveStore(root) as archive:
        session_id = write_index_session(
            archive,
            ParsedSession(
                source_name=Provider.CODEX,
                provider_session_id="lock-free-compute",
                messages=[
                    ParsedMessage(
                        provider_message_id="m1",
                        role=Role.USER,
                        text=text,
                        blocks=[ParsedContentBlock(type=BlockType.TEXT, text=text)],
                        material_origin=MaterialOrigin.HUMAN_AUTHORED,
                    )
                ],
            ),
        )
    index_db = root / "index.db"
    embeddings_db = root / "embeddings.db"
    initialize_archive_database(embeddings_db, ArchiveTier.EMBEDDINGS)
    probe = sqlite3.connect(embeddings_db)
    loaded, error = try_load_sqlite_vec(probe)
    probe.close()
    if not loaded:
        pytest.skip(str(error) if error else "sqlite-vec extension is unavailable")

    observations: list[bool] = []

    class _LockObservingProvider:
        model = load_polylogue_config().embedding_model
        dimension = 1024

        def _get_embeddings(self, texts: list[str], input_type: str = "document") -> list[list[float]]:
            observations.append(_generation_lock_is_free(root))
            return [[0.25] * self.dimension for _ in texts]

    outcome = embed_archive_session_sync(index_db, cast(Any, _LockObservingProvider()), session_id)

    assert outcome.status == "embedded"
    assert observations == [True], "the generation lock must be released for the provider round trip"
