"""An unfinished empty generation can admit the current derived schema."""

from __future__ import annotations

import asyncio
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from polylogue.daemon import cli as daemon_cli
from polylogue.operations.empty_index_startup import EmptyIndexTransitionRefusedError
from polylogue.storage.archive_identity import ArchiveLocation, ArchiveLocationError
from polylogue.storage.sqlite.connection_profile import assert_tier_schema_supported, open_readonly_connection
from tests.infra.empty_managed_index import (
    apply_empty_index_transition,
    make_empty_anchored_bootstrap_index,
    make_empty_managed_index,
    mutate_fixture_database,
    promote_empty_managed_index,
)


@pytest.mark.parametrize("bootstrap", [False, True])
def test_actual_startup_promotes_current_empty_index_and_preserves_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bootstrap: bool
) -> None:
    """Without the transition, real startup reaches preflight with the old identity."""
    root = tmp_path / "archive"
    old = make_empty_anchored_bootstrap_index(root) if bootstrap else make_empty_managed_index(root)
    old_inode = old.stat().st_ino
    original = old.read_bytes()
    durable = {name: (root / name).stat().st_ino for name in ("source.db", "user.db", "audit.db", "embeddings.db")}
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(root))

    class StartupReachedError(Exception):
        pass

    def preflight() -> None:
        active = ArchiveLocation.resolve(root).active_index_path.resolve(strict=True)
        with closing(open_readonly_connection(active)) as conn:
            assert_tier_schema_supported(conn, active)
        assert active != old
        if bootstrap:
            retired = tuple((root / ".index-generations").glob("retired-*/index.db"))
            assert len(retired) == 1
            assert retired[0].stat().st_ino == old_inode
            assert retired[0].read_bytes() == original
        else:
            assert old.read_bytes() == original
        assert {name: (root / name).stat().st_ino for name in durable} == durable
        raise StartupReachedError

    monkeypatch.setattr(daemon_cli, "_check_schema_version_fast", preflight)
    with pytest.raises(StartupReachedError):
        asyncio.run(
            daemon_cli.run_daemon_services(
                sources=(),
                enable_watch=False,
                enable_browser_capture=False,
                browser_capture_host="127.0.0.1",
                browser_capture_port=8765,
            )
        )


@pytest.mark.parametrize(
    ("tier", "sql", "reason"),
    [
        (
            "source",
            "INSERT INTO source_generations VALUES ('pending', '" + "0" * 64 + "', 'path', 1, NULL, 0)",
            "nonempty_source_population",
        ),
        (
            "index",
            "INSERT INTO sessions(native_id,origin,content_hash) VALUES ('one','unknown-export',zeroblob(32))",
            "nonempty_index_population",
        ),
        ("index", "CREATE TABLE unknown_material(value TEXT)", "unprovable_index_shape"),
        ("index", "PRAGMA user_version=0", "unprovable_index_shape"),
        ("source", "CREATE TABLE unknown_custody(value TEXT)", "unprovable_source_shape"),
        ("user", "PRAGMA user_version=99", "unsupported_user_schema"),
        ("audit", "PRAGMA user_version=99", "unsupported_audit_schema"),
        ("embeddings", "PRAGMA user_version=99", "unsupported_embeddings_schema"),
    ],
)
def test_ineligible_custody_refuses_before_generation_creation(
    tmp_path: Path, tier: str, sql: str, reason: str
) -> None:
    root = tmp_path / "archive"
    old = make_empty_managed_index(root)
    path = old if tier == "index" else root / f"{tier}.db"
    mutate_fixture_database(path, sql)
    before = old.read_bytes()
    metadata = tuple((root / ".index-generations").glob("gen-*/generation.json"))
    with pytest.raises(EmptyIndexTransitionRefusedError) as raised:
        apply_empty_index_transition(root)
    assert raised.value.code == "empty_index_transition_refused"
    assert raised.value.reason == reason
    assert ArchiveLocation.resolve(root).active_index_path.resolve(strict=True) == old
    assert old.read_bytes() == before
    assert tuple((root / ".index-generations").glob("gen-*/generation.json")) == metadata


def test_unreferenced_physical_blob_is_still_nonempty_custody(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    old = make_empty_managed_index(root)
    (root / "blob").mkdir(exist_ok=True)
    blob = root / "blob" / "unreferenced"
    blob.write_bytes(b"neutral retained material")
    with pytest.raises(EmptyIndexTransitionRefusedError) as raised:
        apply_empty_index_transition(root)
    assert raised.value.reason == "nonempty_blob_custody"
    assert blob.read_bytes() == b"neutral retained material"
    assert ArchiveLocation.resolve(root).active_index_path.resolve(strict=True) == old


def test_empty_promotion_preserves_unresolved_user_reference_and_audit_rows(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    old = make_empty_managed_index(root)
    mutate_fixture_database(
        root / "user.db",
        "INSERT INTO assertions(assertion_id,target_ref,kind,body_text,created_at_ms,updated_at_ms) "
        "VALUES ('note','session:unknown-export:absent','note','neutral note',0,0)",
    )
    with closing(sqlite3.connect(root / "user.db")) as conn:
        before_user = tuple(conn.iterdump())
    with closing(sqlite3.connect(root / "audit.db")) as conn:
        before_audit = tuple(conn.iterdump())
    assert apply_empty_index_transition(root) is not None
    assert old.is_file()
    with closing(sqlite3.connect(root / "user.db")) as conn:
        assert tuple(conn.iterdump()) == before_user
    with closing(sqlite3.connect(root / "audit.db")) as conn:
        assert tuple(conn.iterdump()) == before_audit
    active = ArchiveLocation.resolve(root).active_index_path.resolve(strict=True)
    assert apply_empty_index_transition(root) is None
    assert ArchiveLocation.resolve(root).active_index_path.resolve(strict=True) == active


@pytest.mark.parametrize("failure", [sqlite3.OperationalError("database is locked"), asyncio.CancelledError()])
def test_physical_failure_and_cancellation_do_not_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: BaseException
) -> None:
    root = tmp_path / "archive"
    old = make_empty_managed_index(root)
    before = old.read_bytes()

    def fail(*_args: object, **_kwargs: object) -> None:
        raise failure

    monkeypatch.setattr("polylogue.operations.empty_index_startup.open_readonly_connection", fail)
    with pytest.raises(type(failure)) as raised:
        apply_empty_index_transition(root)
    assert raised.value is failure
    assert old.read_bytes() == before
    assert ArchiveLocation.resolve(root).active_index_path.resolve(strict=True) == old


def test_supported_configured_symlink_farm_uses_canonical_generation_owner(tmp_path: Path) -> None:
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    canonical = tmp_path / "canonical"
    initialize_active_archive_root(canonical)
    configured = tmp_path / "configured"
    configured.mkdir()
    for tier in ArchiveTier:
        (configured / f"{tier.value}.db").symlink_to(canonical / f"{tier.value}.db")
    old = promote_empty_managed_index(configured)
    assert apply_empty_index_transition(configured) is not None
    assert ArchiveLocation.resolve(configured).active_index_path.resolve(strict=True) != old
    assert old.is_file()


def test_pointer_cannot_grant_authority_over_unrelated_archive(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    old = make_empty_managed_index(root)
    outside = tmp_path / "outside"
    foreign = make_empty_managed_index(outside)
    (root / ".index-active-pointer").write_text(str(foreign))
    before = foreign.read_bytes()
    with pytest.raises(ArchiveLocationError):
        apply_empty_index_transition(root)
    assert foreign.read_bytes() == before
    assert old.is_file()


def test_predecessor_wal_remains_recoverable_without_schema_restamping(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    old = make_empty_managed_index(root)
    with closing(sqlite3.connect(old)) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("UPDATE schema_identity SET identity='retained-wal-runtime'")
        conn.commit()
        assert Path(str(old) + "-wal").stat().st_size > 0
        with closing(conn.execute("SELECT * FROM schema_identity")) as rows:
            prior_identity = rows.fetchall()
        assert apply_empty_index_transition(root) is not None
        assert old.is_file()
        with closing(open_readonly_connection(old, validate_schema=False)) as preserved:
            with closing(preserved.execute("SELECT * FROM schema_identity")) as rows:
                assert rows.fetchall() == prior_identity
            with closing(preserved.execute("SELECT count(*) FROM sessions")) as rows:
                assert rows.fetchone()[0] == 0
        assert ArchiveLocation.resolve(root).active_index_path.resolve(strict=True) != old


def test_embedding_change_after_preparation_refuses_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polylogue.storage.index_generation import IndexGeneration, IndexGenerationStore

    root = tmp_path / "archive"
    old = make_empty_managed_index(root)
    create = IndexGenerationStore.create

    def create_then_change(
        self: IndexGenerationStore, *, owner_id: str | None, source_snapshot: str
    ) -> IndexGeneration:
        generation = create(self, owner_id=owner_id, source_snapshot=source_snapshot)
        mutate_fixture_database(root / "embeddings.db", "PRAGMA user_version=99")
        return generation

    monkeypatch.setattr(IndexGenerationStore, "create", create_then_change)
    with pytest.raises(EmptyIndexTransitionRefusedError) as raised:
        apply_empty_index_transition(root)
    assert raised.value.reason == "changed_embeddings_custody"
    assert ArchiveLocation.resolve(root).active_index_path.resolve(strict=True) == old


@pytest.mark.parametrize("same_target", [False, True])
def test_embedding_symlink_retarget_or_replacement_refuses_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, same_target: bool
) -> None:
    from polylogue.storage.index_generation import IndexGeneration, IndexGenerationStore

    root = tmp_path / "archive"
    old = make_empty_managed_index(root)
    configured = root / "embeddings.db"
    original = root / "retained-embeddings.db"
    configured.rename(original)
    configured.symlink_to(original)
    foreign = tmp_path / "foreign-embeddings.db"
    foreign.write_bytes(original.read_bytes())
    create = IndexGenerationStore.create

    def create_then_retarget(
        self: IndexGenerationStore, *, owner_id: str | None, source_snapshot: str
    ) -> IndexGeneration:
        generation = create(self, owner_id=owner_id, source_snapshot=source_snapshot)
        configured.rename(root / "prior-embedding-link")
        configured.symlink_to(original if same_target else foreign)
        return generation

    monkeypatch.setattr(IndexGenerationStore, "create", create_then_retarget)
    with pytest.raises(EmptyIndexTransitionRefusedError) as raised:
        apply_empty_index_transition(root)
    assert raised.value.reason == "changed_embeddings_binding"
    assert ArchiveLocation.resolve(root).active_index_path.resolve(strict=True) == old


@pytest.mark.parametrize("missing_identity", [False, True])
def test_regular_bootstrap_nonempty_or_unstamped_index_refuses(tmp_path: Path, missing_identity: bool) -> None:
    root = tmp_path / "archive"
    old = make_empty_anchored_bootstrap_index(root)
    mutate_fixture_database(
        old,
        "DELETE FROM schema_identity"
        if missing_identity
        else "INSERT INTO sessions(native_id,origin,content_hash) VALUES ('one','unknown-export',zeroblob(32))",
    )
    before = old.read_bytes()
    with pytest.raises(EmptyIndexTransitionRefusedError) as raised:
        apply_empty_index_transition(root)
    assert raised.value.reason == ("missing_index_identity" if missing_identity else "nonempty_index_population")
    assert not old.is_symlink()
    assert old.read_bytes() == before
    assert not tuple((root / ".index-generations").glob("gen-*/generation.json"))


def test_regular_anchor_cannot_select_a_different_bootstrap_index(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    old = make_empty_anchored_bootstrap_index(root)
    wrong = root / "other" / "index.db"
    wrong.parent.mkdir()
    wrong.write_bytes(old.read_bytes())
    (root / ".index-active-pointer").write_text(str(wrong))
    before = wrong.read_bytes()
    with pytest.raises(EmptyIndexTransitionRefusedError) as raised:
        apply_empty_index_transition(root)
    assert raised.value.reason == "managed_index_escapes_archive"
    assert wrong.read_bytes() == before
    assert not old.is_symlink()
