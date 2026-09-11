from __future__ import annotations

import asyncio
import sqlite3
import threading
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from polylogue.daemon import convergence_stages, embedding_backlog, embedding_owner
from polylogue.daemon.execution import BoundedComputeAdapter
from polylogue.daemon.status import format_daemon_status_lines
from polylogue.daemon.write_coordinator import DaemonWriteCoordinator
from polylogue.storage.embeddings.identity import EmbeddingRecipe
from polylogue.storage.embeddings.materialization import EmbedSessionOutcome, PendingSession
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier


def test_periodic_embedding_backlog_waits_for_catch_up_complete(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    async def fake_lease_free(function: object, *_args: object, **_kwargs: object) -> object:
        # The drain is the provider-calling pass, so it is admitted to the
        # daemon's compute capacity rather than run under the writer gate.
        assert function is embedding_backlog.drain_embedding_backlog_once
        calls.append("drain")
        raise asyncio.CancelledError

    async def exercise() -> None:
        catch_up_complete = asyncio.Event()
        monkeypatch.setattr(
            "polylogue.storage.archive_identity.resolve_active_index_path", lambda *_a, **_k: tmp_path / "index.db"
        )
        monkeypatch.setattr(
            "polylogue.daemon.embedding_owner.run_lease_free_embedding_work",
            fake_lease_free,
        )
        monkeypatch.setattr(embedding_backlog, "EMBEDDING_BACKLOG_RETRY_INTERVAL_SECONDS", 0)
        task = asyncio.create_task(
            embedding_backlog.periodic_embedding_backlog_check(catch_up_complete=catch_up_complete)
        )
        await asyncio.sleep(0)
        assert calls == []
        catch_up_complete.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(exercise())

    assert calls == ["drain"]


class _EmbeddingConfig:
    embedding_enabled = True
    voyage_api_key = "pa-test"
    embedding_model = "voyage-4"
    embedding_dimension = 1024

    def get(self, key: str, default: object = None) -> object:
        values: dict[str, object] = {
            "voyage_api_key": "pa-test",
            "embedding_max_cost_usd": 5.0,
        }
        return values.get(key, default)


class _FakeRepository:
    def __init__(self, *, backend: object) -> None:
        self.backend = backend

    async def close(self) -> None:
        return None


def _stamp_index_tier(db_path: Path) -> None:
    """Declare the index tier version on a hand-built stub.

    The daemon opens a candidate index through the tier-guarded read profile
    before draining it, so a stub that claims no version is passed over.
    """

    from polylogue.storage.sqlite.archive_tiers.bootstrap import ARCHIVE_TIER_SPECS
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    with sqlite3.connect(db_path) as conn:
        conn.execute(f"PRAGMA user_version = {ARCHIVE_TIER_SPECS[ArchiveTier.INDEX].version}")


def _seed_embedding_db(db_path: Path) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE sessions (
                session_id TEXT PRIMARY KEY,
                title TEXT,
                updated_at TEXT
            );
            CREATE TABLE messages (
                message_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                text TEXT
            );
            """
        )
        conn.execute("INSERT INTO sessions VALUES ('conv-a', 'A', '2026-01-01')")
        conn.execute("INSERT INTO sessions VALUES ('conv-b', 'B', '2026-01-02')")
        conn.execute("INSERT INTO messages VALUES ('msg-a-1', 'conv-a', 'alpha')")
        conn.execute("INSERT INTO messages VALUES ('msg-a-2', 'conv-a', 'beta')")
        conn.execute("INSERT INTO messages VALUES ('msg-b-1', 'conv-b', 'gamma')")
        conn.commit()


def test_daemon_embedding_backlog_drain_processes_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_anchor_path = tmp_path / "index.db"
    db_anchor_path.touch()
    archive_db = tmp_path / "index.db"
    with sqlite3.connect(archive_db) as conn:
        conn.executescript(
            """
            CREATE TABLE sessions (
                session_id TEXT PRIMARY KEY,
                title TEXT,
                message_count INTEGER NOT NULL,
                sort_key_ms INTEGER
            );
            INSERT INTO sessions VALUES ('codex-session:v1-a', 'Archive A', 2, 1);
            INSERT INTO sessions VALUES ('codex-session:v1-b', 'Archive B', 1, 2);
            """
        )
    _stamp_index_tier(db_anchor_path)

    embedded_calls: list[tuple[Path, str]] = []

    def fake_embed(db_path: Path, _provider: object, session_id: str, **_kwargs: object) -> EmbedSessionOutcome:
        embedded_calls.append((db_path, session_id))
        return EmbedSessionOutcome(
            status="embedded",
            session_id=session_id,
            embedded_message_count=2,
        )

    monkeypatch.setattr(convergence_stages, "load_polylogue_config", lambda: _EmbeddingConfig())
    monkeypatch.setattr(embedding_backlog, "load_polylogue_config", lambda: _EmbeddingConfig())
    monkeypatch.setattr("polylogue.storage.search_providers.create_vector_provider", lambda **_: MagicMock())
    monkeypatch.setattr("polylogue.storage.embeddings.materialization.embed_archive_session_sync", fake_embed)

    processed = embedding_backlog.drain_embedding_backlog_once(db_anchor_path)

    assert processed == 2
    assert set(embedded_calls) == {(archive_db, "codex-session:v1-a"), (archive_db, "codex-session:v1-b")}
    from polylogue.storage.sqlite.archive_tiers.ops_write import list_embedding_catchup_runs

    with sqlite3.connect(tmp_path / "ops.db") as conn:
        runs = list_embedding_catchup_runs(conn)
    assert len(runs) == 1
    assert runs[0].status == "completed"
    assert runs[0].scanned_sessions == 2
    assert runs[0].embedded_sessions == 2
    assert runs[0].skipped_sessions == 0
    assert runs[0].error_count == 0
    assert runs[0].embedded_messages == 4


def test_daemon_embedding_backlog_uses_bounded_pending_window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "index.db"
    db_path.touch()
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE sessions (
                session_id TEXT PRIMARY KEY,
                title TEXT,
                message_count INTEGER NOT NULL,
                sort_key_ms INTEGER
            );
            INSERT INTO sessions VALUES ('codex-session:v1-a', 'Archive A', 2, 1);
            """
        )
    _stamp_index_tier(db_path)

    observed_kwargs: dict[str, object] = {}

    def fake_select(*args: object, **kwargs: object) -> list[PendingSession]:
        observed_kwargs.update(kwargs)
        return [PendingSession(session_id="codex-session:v1-a", title="Archive A", message_count=2)]

    def fake_embed(_db_path: Path, _provider: object, session_id: str, **_kwargs: object) -> EmbedSessionOutcome:
        return EmbedSessionOutcome(status="embedded", session_id=session_id, embedded_message_count=2)

    monkeypatch.setattr(convergence_stages, "load_polylogue_config", lambda: _EmbeddingConfig())
    monkeypatch.setattr(embedding_backlog, "load_polylogue_config", lambda: _EmbeddingConfig())
    monkeypatch.setattr("polylogue.storage.search_providers.create_vector_provider", lambda **_: MagicMock())
    monkeypatch.setattr(
        "polylogue.storage.embeddings.materialization.select_pending_archive_session_window", fake_select
    )
    monkeypatch.setattr("polylogue.storage.embeddings.materialization.embed_archive_session_sync", fake_embed)

    assert embedding_backlog.drain_embedding_backlog_once(db_path) == 1
    assert "include_stale_checks" not in observed_kwargs
    recipe = observed_kwargs["recipe"]
    assert isinstance(recipe, EmbeddingRecipe)
    assert recipe.model == "voyage-4"
    assert recipe.dimensions == 1024


def test_daemon_embedding_backlog_records_skipped_sessions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_anchor_path = tmp_path / "index.db"
    db_anchor_path.touch()
    with sqlite3.connect(db_anchor_path) as conn:
        conn.executescript(
            """
            CREATE TABLE sessions (
                session_id TEXT PRIMARY KEY,
                title TEXT,
                message_count INTEGER NOT NULL,
                sort_key_ms INTEGER
            );
            INSERT INTO sessions VALUES ('codex-session:skip-a', 'Skip A', 1, 2);
            INSERT INTO sessions VALUES ('codex-session:skip-b', 'Skip B', 1, 1);
            """
        )
    _stamp_index_tier(db_anchor_path)

    def fake_embed(_db_path: Path, _provider: object, session_id: str, **_kwargs: object) -> EmbedSessionOutcome:
        return EmbedSessionOutcome(status="no_embeddable_messages", session_id=session_id)

    monkeypatch.setattr(convergence_stages, "load_polylogue_config", lambda: _EmbeddingConfig())
    monkeypatch.setattr(embedding_backlog, "load_polylogue_config", lambda: _EmbeddingConfig())
    monkeypatch.setattr("polylogue.storage.search_providers.create_vector_provider", lambda **_: MagicMock())
    monkeypatch.setattr("polylogue.storage.embeddings.materialization.embed_archive_session_sync", fake_embed)

    processed = embedding_backlog.drain_embedding_backlog_once(db_anchor_path)

    assert processed == 2
    from polylogue.storage.sqlite.archive_tiers.ops_write import list_embedding_catchup_runs

    with sqlite3.connect(tmp_path / "ops.db") as conn:
        runs = list_embedding_catchup_runs(conn)
    assert len(runs) == 1
    assert runs[0].scanned_sessions == 2
    assert runs[0].embedded_sessions == 0
    assert runs[0].skipped_sessions == 2
    assert runs[0].error_count == 0
    assert runs[0].embedded_messages == 0


def test_archive_convergence_embedding_uses_embeddings_tier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index_db = tmp_path / "index.db"
    embeddings_db = tmp_path / "embeddings.db"
    with sqlite3.connect(index_db) as conn:
        conn.executescript(
            """
            CREATE TABLE sessions (
                session_id TEXT PRIMARY KEY,
                title TEXT,
                message_count INTEGER NOT NULL,
                sort_key_ms INTEGER
            );
            INSERT INTO sessions VALUES ('codex-session:v1-a', 'Archive A', 2, 1);
            """
        )
    observed_vector_db_paths: list[Path] = []
    embedded_calls: list[tuple[Path, str, Path | None]] = []
    fake_provider = MagicMock()

    def fake_create_vector_provider(**kwargs: object) -> object:
        observed_vector_db_paths.append(Path(str(kwargs["db_path"])))
        return fake_provider

    def fake_embed(db_path: Path, provider: object, session_id: str, **kwargs: object) -> EmbedSessionOutcome:
        assert provider is fake_provider
        embeddings_db_path = kwargs.get("embeddings_db_path")
        embedded_calls.append(
            (db_path, session_id, embeddings_db_path if isinstance(embeddings_db_path, Path) else None)
        )
        return EmbedSessionOutcome(
            status="embedded",
            session_id=session_id,
            embedded_message_count=2,
        )

    monkeypatch.setattr(convergence_stages, "load_polylogue_config", lambda: _EmbeddingConfig())
    monkeypatch.setattr("polylogue.storage.search_providers.create_vector_provider", fake_create_vector_provider)
    monkeypatch.setattr("polylogue.storage.embeddings.materialization.embed_archive_session_sync", fake_embed)

    ok = convergence_stages._embed_archive_sessions_sync(index_db, ("codex-session:v1-a",))

    assert ok is True
    assert observed_vector_db_paths == [embeddings_db]
    assert embedded_calls == [(index_db, "codex-session:v1-a", embeddings_db)]


def test_archive_convergence_persists_foreground_spend_before_enforcing_monthly_cap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Foreground passes share the persistent monthly embedding allowance.

    Anti-vacuity: removing the foreground receipt lets the second pass call the
    provider despite the first pass having exhausted most of the cap.
    """

    class CostCappedConfig(_EmbeddingConfig):
        def get(self, key: str, default: object = None) -> object:
            if key == "embedding_max_cost_usd":
                return 0.000075
            return super().get(key, default)

    embedded_calls: list[str] = []

    def fake_embed(_db_path: Path, _provider: object, session_id: str, **_kwargs: object) -> EmbedSessionOutcome:
        embedded_calls.append(session_id)
        return EmbedSessionOutcome(status="embedded", session_id=session_id, embedded_message_count=1)

    monkeypatch.setattr(convergence_stages, "load_polylogue_config", lambda: CostCappedConfig())
    monkeypatch.setattr("polylogue.storage.search_providers.create_vector_provider", lambda **_: MagicMock())
    monkeypatch.setattr("polylogue.storage.embeddings.materialization.embed_archive_session_sync", fake_embed)

    first = convergence_stages._embed_archive_sessions_sync(
        tmp_path / "index.db",
        [PendingSession(session_id="first", message_count=1)],
    )
    second = convergence_stages._embed_archive_sessions_sync(
        tmp_path / "index.db",
        [PendingSession(session_id="second", message_count=1)],
    )

    assert first is True
    assert second is False
    assert embedded_calls == ["first"]


def test_embedding_startup_marks_running_catchup_receipts_interrupted(tmp_path: Path) -> None:
    from polylogue.core.enums import OperationStatus
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
    from polylogue.storage.sqlite.archive_tiers.ops_write import (
        list_embedding_catchup_runs,
        upsert_embedding_catchup_run,
    )
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    ops_db = tmp_path / "ops.db"
    with sqlite3.connect(ops_db) as conn:
        initialize_archive_tier(conn, ArchiveTier.OPS)
        upsert_embedding_catchup_run(
            conn,
            run_id="unfinished",
            status=OperationStatus.RUNNING,
            started_at_ms=1,
        )

    assert embedding_backlog.recover_embedding_catchup_receipts(tmp_path) == 1

    with sqlite3.connect(ops_db) as conn:
        (run,) = list_embedding_catchup_runs(conn)
    assert run.status == "interrupted"
    assert run.finished_at_ms is not None


def test_archive_convergence_pending_check_rejects_status_only_freshness(tmp_path: Path) -> None:
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    index_db = tmp_path / "index.db"
    embeddings_db = tmp_path / "embeddings.db"
    with sqlite3.connect(index_db) as conn:
        conn.executescript(
            """
            CREATE TABLE sessions (
                session_id TEXT PRIMARY KEY,
                title TEXT,
                message_count INTEGER NOT NULL,
                sort_key_ms INTEGER
            );
            CREATE TABLE messages (
                message_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                position INTEGER NOT NULL,
                variant_index INTEGER NOT NULL DEFAULT 0,
                message_type TEXT NOT NULL DEFAULT 'message',
                role TEXT NOT NULL,
                material_origin TEXT NOT NULL,
                word_count INTEGER NOT NULL DEFAULT 1,
                content_hash BLOB
            );
            INSERT INTO sessions VALUES ('codex-session:v1-a', 'Archive A', 1, 1);
            INSERT INTO messages (
                message_id, session_id, position, role, material_origin, word_count, content_hash
            ) VALUES ('m1', 'codex-session:v1-a', 0, 'user', 'human_authored', 8, x'01');
            """
        )
    with sqlite3.connect(embeddings_db) as conn:
        initialize_archive_tier(conn, ArchiveTier.EMBEDDINGS)
        conn.execute(
            """
            INSERT INTO embedding_status (
                session_id, origin, message_count_embedded, needs_reindex, error_message
            ) VALUES ('codex-session:v1-a', 'codex-session', 1, 0, NULL)
            """
        )
        conn.commit()

    with sqlite3.connect(index_db) as conn:
        assert convergence_stages._archive_pending_embedding_session_ids(conn, ["codex-session:v1-a"]) == [
            "codex-session:v1-a"
        ]


def test_daemon_embedding_backlog_drain_is_noop_when_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "index.db"
    _seed_embedding_db(db_path)

    class DisabledConfig(_EmbeddingConfig):
        embedding_enabled = False

    monkeypatch.setattr(convergence_stages, "load_polylogue_config", lambda: DisabledConfig())
    monkeypatch.setattr(embedding_backlog, "load_polylogue_config", lambda: DisabledConfig())

    assert embedding_backlog.drain_embedding_backlog_once(db_path) == 0


def test_daemon_embedding_backlog_drain_pauses_when_monthly_cap_spent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "index.db"
    _seed_embedding_db(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE embedding_catchup_runs (
                run_id TEXT PRIMARY KEY,
                started_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                completed_at TEXT,
                status TEXT NOT NULL,
                stop_reason TEXT,
                rebuild INTEGER NOT NULL DEFAULT 0,
                max_sessions INTEGER,
                max_messages INTEGER,
                stop_after_seconds INTEGER,
                max_errors INTEGER,
                planned_sessions INTEGER NOT NULL DEFAULT 0,
                planned_messages INTEGER NOT NULL DEFAULT 0,
                processed_sessions INTEGER NOT NULL DEFAULT 0,
                embedded_sessions INTEGER NOT NULL DEFAULT 0,
                skipped_sessions INTEGER NOT NULL DEFAULT 0,
                error_count INTEGER NOT NULL DEFAULT 0,
                embedded_messages INTEGER NOT NULL DEFAULT 0,
                estimated_cost_usd REAL NOT NULL DEFAULT 0.0,
                last_session_id TEXT
            )
            """
        )
        conn.execute(
            """
            INSERT INTO embedding_catchup_runs (
                run_id, started_at, updated_at, status, estimated_cost_usd
            ) VALUES ('spent', datetime('now'), datetime('now'), 'completed', 5.0)
            """
        )
        conn.commit()

    monkeypatch.setattr(convergence_stages, "load_polylogue_config", lambda: _EmbeddingConfig())
    monkeypatch.setattr(embedding_backlog, "load_polylogue_config", lambda: _EmbeddingConfig())

    assert embedding_backlog.drain_embedding_backlog_once(db_path) == 0


def test_daemon_status_lines_include_latest_embedding_catchup() -> None:
    lines = format_daemon_status_lines(
        {
            "embedding_readiness": {
                "embedding_enabled": True,
                "embedding_coverage_percent": 12.5,
                "embedding_pending_count": 10,
                "embedding_pending_message_count": 200,
                "embedding_stale_count": 0,
                "embedding_failure_count": 0,
                "embedding_estimated_cost_usd": 0.02,
                "embedding_model": "voyage-4",
                "embedding_dimension": 1024,
                "embedding_latest_catchup_run": {
                    "status": "running",
                    "processed_sessions": 3,
                    "planned_sessions": 10,
                    "embedded_messages": 42,
                },
            }
        }
    )

    assert "  latest catch-up: running, 3/10 convs, 42 msgs embedded" in lines


# ── Lease-free embedding computation (polylogue-c0l7n) ──────────────────────
#
# The provider call used to run inside the daemon's writer gate and inside the
# embedding generation lock, so one slow round trip blocked every unrelated
# archive publication. These tests hold each of the three production owners to
# the split contract: reserve under admission, compute holding nothing, publish
# under admission again.


def _install_test_coordinator(monkeypatch: pytest.MonkeyPatch, archive_root: Path) -> DaemonWriteCoordinator:
    """Bind one archive-scoped coordinator and one compute pool into the owner."""
    coordinator = DaemonWriteCoordinator(archive_root=archive_root)
    adapter = BoundedComputeAdapter(max_workers=2)
    monkeypatch.setattr(embedding_owner, "daemon_write_coordinator", lambda: coordinator)
    monkeypatch.setattr(embedding_owner, "daemon_compute_adapter", lambda: adapter)
    return coordinator


class _PausedEmbeddingPass:
    """A stand-in embedding pass that observes its own authority, then blocks."""

    def __init__(self, archive_root: Path) -> None:
        self.archive_root = archive_root
        self.entered = threading.Event()
        self.release = threading.Event()
        self.lease_during_compute: bool | None = None
        self.lease_during_publish: bool | None = None

    def __call__(self, *, admit: object) -> bool:
        from polylogue.daemon.write_coordinator import daemon_write_lease_active

        self.lease_during_compute = daemon_write_lease_active()
        self.entered.set()
        assert self.release.wait(10.0), "paused embedding compute was never released"

        def publish() -> None:
            self.lease_during_publish = daemon_write_lease_active()

        cast(Any, admit)("embedding.publish", publish)
        return True


@pytest.mark.asyncio
async def test_embedding_compute_holds_no_writer_authority_and_unblocks_other_writers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Embedding compute holds no writer authority, and publication still does.

    Anti-vacuity: restore the old route -- run the whole pass inside one
    ``DaemonWriteCoordinator`` operation -- and ``lease_during_compute`` becomes
    True while the unrelated writer below blocks until the provider returns, so
    the ``wait_for`` fails. Drop the admitted publish phase instead and
    ``lease_during_publish`` becomes False, which is the other half: compute
    must lose the authority, publication must keep it.

    The generation lock is proven separately, against the real archive route,
    by ``test_generation_lock_is_free_while_the_provider_computes`` in
    ``tests/unit/storage/test_embedding_generations.py``.
    """
    coordinator = _install_test_coordinator(monkeypatch, tmp_path)
    embedding_pass = _PausedEmbeddingPass(tmp_path)
    unrelated_committed: list[bool] = []

    def unrelated_writer() -> None:
        from polylogue.daemon.write_coordinator import daemon_write_lease_active

        unrelated_committed.append(daemon_write_lease_active())

    task = asyncio.create_task(embedding_owner.run_lease_free_embedding_work(embedding_pass))
    await asyncio.to_thread(embedding_pass.entered.wait, 10.0)
    assert embedding_pass.entered.is_set()

    # The provider is still paused. An unrelated bounded writer must be able to
    # enter the gate and commit right now.
    await asyncio.wait_for(coordinator.run_sync("maintenance.unrelated", unrelated_writer), timeout=5.0)
    assert unrelated_committed == [True]

    embedding_pass.release.set()
    assert await asyncio.wait_for(task, timeout=10.0) is True

    assert embedding_pass.lease_during_compute is False
    assert embedding_pass.lease_during_publish is True


@pytest.mark.asyncio
async def test_embedding_owner_refuses_to_run_inside_a_held_writer_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A caller that still holds the gate is refused, not deadlocked.

    Anti-vacuity: drop the guard and this call hangs forever, because the
    admitted publish phase queues behind the very gate its caller is holding.
    """
    coordinator = _install_test_coordinator(monkeypatch, tmp_path)

    async def inside_gate() -> None:
        with pytest.raises(RuntimeError, match="holds the daemon writer gate"):
            await embedding_owner.run_lease_free_embedding_work(lambda *, admit: True)

    await asyncio.wait_for(coordinator.run("maintenance.holder", inside_gate), timeout=5.0)


@pytest.mark.asyncio
async def test_embedding_backlog_owner_drains_without_the_writer_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The periodic backlog loop runs its drain off the writer.

    Anti-vacuity: route the drain back through
    ``daemon_write_coordinator().run_sync`` and ``observed`` records True.
    """
    from polylogue.daemon.write_coordinator import daemon_write_lease_active

    _install_test_coordinator(monkeypatch, tmp_path)
    observed: list[bool] = []

    def fake_drain(_db: Path, *, admit: object) -> int:
        observed.append(daemon_write_lease_active())
        raise asyncio.CancelledError

    monkeypatch.setattr(embedding_backlog, "drain_embedding_backlog_once", fake_drain)
    monkeypatch.setattr(embedding_backlog, "EMBEDDING_BACKLOG_RETRY_INTERVAL_SECONDS", 0)
    monkeypatch.setattr("polylogue.paths.archive_root", lambda: tmp_path)

    catch_up_complete = asyncio.Event()
    catch_up_complete.set()
    task = asyncio.create_task(embedding_backlog.periodic_embedding_backlog_check(catch_up_complete=catch_up_complete))
    with pytest.raises(asyncio.CancelledError):
        await task
    assert observed == [False]


@pytest.mark.asyncio
async def test_convergence_debt_retry_embeds_before_its_admitted_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Embed-stage debt is converged off the writer, then cleared by the admitted pass.

    Anti-vacuity: delete the lease-free pass and ``order`` loses its
    ``embed_owner`` entry entirely -- the admitted drain would re-defer the
    same debt forever, because the stage refuses to call a provider under the
    gate.
    """
    from polylogue.daemon import cli as daemon_cli
    from polylogue.daemon.write_coordinator import daemon_write_lease_active
    from polylogue.sources.live.cursor import CursorStore

    index_db = tmp_path / "index.db"
    initialize_archive_database(index_db, ArchiveTier.INDEX)
    cursor = CursorStore(index_db)
    cursor.record_convergence_debt(
        stage="embed",
        subject_type="session_id",
        subject_id="codex-session:v1-a",
        error="deferred to the lease-free embedding owner",
        deferred=True,
    )
    # A freshly recorded row carries its backoff, so age it into the due window
    # the retry tick actually inspects.
    with sqlite3.connect(tmp_path / "ops.db") as ops:
        ops.execute("UPDATE convergence_debt SET next_retry_at = ?", ("2000-01-01T00:00:00+00:00",))
        ops.commit()

    coordinator = _install_test_coordinator(monkeypatch, tmp_path)
    monkeypatch.setattr(daemon_cli, "daemon_write_coordinator", lambda: coordinator)
    order: list[tuple[str, bool]] = []

    def fake_convergence(_db: Path, *, paths: object = (), session_ids: object = (), admit: object = None) -> bool:
        order.append(("embed_owner", daemon_write_lease_active()))
        assert tuple(cast(Any, session_ids)) == ("codex-session:v1-a",)
        return True

    def fake_drain(_db: Path, *, limit: int = 0) -> int:
        order.append(("admitted_drain", daemon_write_lease_active()))
        return 0

    monkeypatch.setattr("polylogue.daemon.convergence_stages.run_archive_embedding_convergence", fake_convergence)
    monkeypatch.setattr(daemon_cli, "_drain_convergence_debt_once", fake_drain)

    await asyncio.wait_for(daemon_cli._retry_convergence_debt_once(index_db), timeout=10.0)

    assert order == [("embed_owner", False), ("admitted_drain", True)]
