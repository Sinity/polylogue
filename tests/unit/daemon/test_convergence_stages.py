from __future__ import annotations

import asyncio
import hashlib
import inspect
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

import polylogue.daemon.convergence_stages as stages
from polylogue.archive.message.roles import Role
from polylogue.archive.revision_authority import RawRevisionEnvelope, RawRevisionKind
from polylogue.core.enums import BlockType, Provider
from polylogue.daemon.convergence_stages import (
    make_default_convergence_stages,
    make_embed_stage,
    make_fts_stage,
    make_raw_authority_verdict_cache_stage,
)
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.storage.derived.session import storage as session_storage
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import (
    initialize_active_archive_root,
    initialize_archive_tier,
)
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.write import write_parsed_session_to_archive
from tests.infra.identity import archive_message_id


class _SessionIdOnly:
    def __init__(self, session_id: str, marker: str) -> None:
        self.session_id = session_id
        self.marker = marker


def test_session_storage_dedupes_records_by_session_id() -> None:
    records = [
        _SessionIdOnly("codex-session:one", "first"),
        _SessionIdOnly("codex-session:one", "second"),
        _SessionIdOnly("codex-session:two", "third"),
    ]

    deduped = session_storage._dedupe_records_by_session(records)

    assert [(record.session_id, record.marker) for record in deduped] == [
        ("codex-session:one", "second"),
        ("codex-session:two", "third"),
    ]


def test_default_convergence_stages_leave_session_profiles_to_the_typed_owner(tmp_path: Path) -> None:
    stages_by_name = {stage.name: stage for stage in make_default_convergence_stages(tmp_path / "index.db")}

    assert stages_by_name["fts"].false_means_pending is True
    assert stages_by_name["embed"].false_means_pending is True
    assert "derived" not in stages_by_name


def test_raw_authority_verdict_cache_stage_warms_in_bounded_batches_and_reports_readiness(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    initialize_active_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        for index in range(stages._DAEMON_RAW_AUTHORITY_CACHE_MAX_COHORTS + 1):
            raw_id = f"full-{index}"
            written_id = archive.write_raw_payload(
                provider=Provider.CODEX,
                payload=f"payload-{index}".encode(),
                source_path="session.jsonl",
                acquired_at_ms=1,
                raw_id=raw_id,
            )
            archive.bind_raw_revision(
                written_id,
                RawRevisionEnvelope(f"codex:full-{index}", RawRevisionKind.FULL, f"revision-{raw_id}", 0),
            )
        append_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=b"append-payload",
            source_path="session.jsonl",
            acquired_at_ms=1,
            raw_id="append-only",
        )
        archive.bind_raw_revision(
            append_id,
            RawRevisionEnvelope(
                "codex:append",
                RawRevisionKind.APPEND,
                "revision-append",
                0,
                predecessor_source_revision="revision-base",
                predecessor_raw_id="base",
                baseline_raw_id="base",
                append_start_offset=0,
                append_end_offset=1,
            ),
        )

    stage = make_raw_authority_verdict_cache_stage(tmp_path / "index.db")
    assert stage.check_many is not None
    assert stage.execute_many is not None
    assert stage.false_means_pending is True
    path = tmp_path / "source.jsonl"
    with caplog.at_level("INFO"):
        assert stage.check(path) is True
        assert stage.execute_many((path,)) is False
        assert stage.check(path) is True
        assert stage.execute_many((path,)) is True
        assert stage.check(path) is False

    with sqlite3.connect(tmp_path / "source.db") as conn:
        cached_cohorts = {
            str(row[0]) for row in conn.execute("SELECT DISTINCT logical_source_key FROM raw_authority_verdicts")
        }
    assert len(cached_cohorts) == stages._DAEMON_RAW_AUTHORITY_CACHE_MAX_COHORTS + 2
    assert "codex:append" in cached_cohorts
    assert "raw_authority_verdict_cache: warmed cohorts=" in caplog.text

    import polylogue.storage.raw_authority_verdict_cache as cache_module

    def _fail_projection(*args: object, **kwargs: object) -> dict[str, object]:
        raise AssertionError("warm cache was recomputed")

    monkeypatch.setattr(cache_module, "project_raw_authority_verdicts", _fail_projection)
    assert stage.execute_many((path,)) is True


def test_sinex_stage_uses_configured_source_tier_not_active_index_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polylogue.sinex.models import PublicationMode

    configured_root = tmp_path / "configured"
    captured: dict[str, object] = {}

    class CapturePublicationService:
        mode = PublicationMode.PRIMARY

        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

        def blocking_object_ids(self, object_ids: object) -> set[str]:
            del object_ids
            return set()

    monkeypatch.setattr(stages, "load_polylogue_config", lambda: SimpleNamespace(sinex_mode="primary"))
    monkeypatch.setattr("polylogue.paths.archive_root", lambda: configured_root)
    monkeypatch.setattr("polylogue.sinex.service.PublicationService", CapturePublicationService)
    monkeypatch.setattr("polylogue.sinex.transport.resolve_configured_transport", lambda: object())

    make_default_convergence_stages(tmp_path / "external-generation" / "index.db")

    assert captured["source_db_path"] == configured_root / "source.db"


def test_file_probe_exceptions_log_and_fail_toward_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    db_path = tmp_path / "legacy.sqlite"
    db_path.touch()
    paths = [tmp_path / "one.jsonl", tmp_path / "two.jsonl"]
    session_ids = ["codex-session:s1", "codex-session:s2"]

    def locked_connection(*_args: object, **_kwargs: object) -> sqlite3.Connection:
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(sqlite3, "connect", locked_connection)
    warning_exc_info: list[object] = []
    real_warning = stages.logger.warning

    def capture_warning(message: str, *args: object, **kwargs: object) -> None:
        warning_exc_info.append(kwargs.get("exc_info"))
        real_warning(message, *args, **kwargs)

    monkeypatch.setattr(stages.logger, "warning", capture_warning)

    fts = make_fts_stage(db_path)
    with caplog.at_level("WARNING"):
        assert fts.check(paths[0]) is True
        assert fts.check_many is not None
        assert fts.check_many(paths) == set(paths)
        assert fts.check_sessions is not None
        assert fts.check_sessions(session_ids) == set(session_ids)

    assert "fts: source-path partition lookup failed" in caplog.text
    assert warning_exc_info
    assert all(value is True for value in warning_exc_info)


def test_archive_probe_exceptions_log_and_fail_toward_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    archive_db = tmp_path / "index.db"
    paths = [tmp_path / "one.jsonl", tmp_path / "two.jsonl"]
    session_ids = ["codex-session:s1", "codex-session:s2"]

    def locked_connect(*_args: object, **_kwargs: object) -> sqlite3.Connection:
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(sqlite3, "connect", locked_connect)
    warning_exc_info: list[object] = []
    real_warning = stages.logger.warning

    def capture_warning(message: str, *args: object, **kwargs: object) -> None:
        warning_exc_info.append(kwargs.get("exc_info"))
        real_warning(message, *args, **kwargs)

    monkeypatch.setattr(stages.logger, "warning", capture_warning)

    with caplog.at_level("WARNING"):
        assert stages._archive_embed_check(archive_db, paths[0]) is True
        assert stages._archive_embed_check_many(archive_db, paths) == set(paths)
        assert stages._archive_embed_check_sessions(archive_db, session_ids) == set(session_ids)
    assert caplog.text.count("convergence freshness probe") >= 3
    assert "treating as needs-work" in caplog.text
    assert warning_exc_info
    assert all(value is True for value in warning_exc_info)


def _seed_index_session(conn: sqlite3.Connection, *, session_id: str, text: str) -> str:
    return write_parsed_session_to_archive(
        conn,
        ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id=session_id,
            title=session_id,
            created_at="2026-05-24T01:00:00+00:00",
            updated_at="2026-05-24T01:00:00+00:00",
            messages=[
                ParsedMessage(
                    provider_message_id="msg-1",
                    role=Role.normalize("user"),
                    text=text,
                    position=0,
                    blocks=[ParsedContentBlock(type=BlockType.TEXT, text=text)],
                )
            ],
        ),
        content_hash=hashlib.sha256(f"session:{session_id}".encode()).hexdigest(),
    )


def _seed_empty_text_index_session(conn: sqlite3.Connection, *, session_id: str) -> str:
    return write_parsed_session_to_archive(
        conn,
        ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id=session_id,
            title=session_id,
            messages=[
                ParsedMessage(
                    provider_message_id="msg-1",
                    role=Role.normalize("user"),
                    text="",
                    position=0,
                    blocks=[ParsedContentBlock(type=BlockType.TEXT, text="")],
                )
            ],
        ),
        content_hash=hashlib.sha256(f"session:{session_id}".encode()).hexdigest(),
    )


def _seed_minimal_archive(db_path: Path, source_path: Path, *, session_id: str = "codex-session:s1") -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text("{}\n", encoding="utf-8")
    with sqlite3.connect(db_path.with_name("source.db")) as conn:
        initialize_archive_tier(conn, ArchiveTier.SOURCE)
        conn.execute(
            """
            INSERT INTO raw_sessions(raw_id, origin, native_id, source_path, blob_hash, blob_size, acquired_at_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "raw-s1",
                "codex-session",
                session_id,
                str(source_path),
                hashlib.sha256(b"raw-s1").digest(),
                source_path.stat().st_size,
                1_770_000_000_000,
            ),
        )
        conn.commit()
    with sqlite3.connect(db_path) as conn:
        initialize_archive_tier(conn, ArchiveTier.INDEX)
        native_id = session_id.split(":", 1)[-1]
        conn.execute(
            """
            INSERT INTO sessions(
                native_id, origin, raw_id, title, message_count, user_message_count,
                assistant_message_count, tool_use_count, paste_count, content_hash, updated_at_ms
            ) VALUES (?, 'codex-session', 'raw-s1', 'Native session', 1, 1, 0, 0, 0, ?, 1770000000000)
            """,
            (native_id, hashlib.sha256(f"session:{session_id}".encode()).digest()),
        )
        conn.execute(
            """
            INSERT INTO messages(session_id, native_id, position, role, message_type, content_hash)
            VALUES (?, 'm1', 0, 'user', 'message', ?)
            """,
            (session_id, hashlib.sha256(f"message:{session_id}".encode()).digest()),
        )
        conn.execute(
            """
            INSERT INTO blocks(message_id, session_id, position, block_type, text)
            VALUES (?, ?, 0, 'text', 'archive searchable block')
            """,
            (archive_message_id(session_id, "m1", position=0), session_id),
        )
        conn.execute("DELETE FROM messages_fts")
        conn.commit()


def test_fts_stage_converges_archive_source_path_sessions(tmp_path: Path) -> None:
    """Foreground source-path convergence indexes the path's sessions.

    This previously asserted the opposite -- that the archive-backed stage
    reports "nothing to do" and "done" while leaving the index empty -- on the
    premise that archive writes already index newly changed rows. Live full
    ingest breaks that premise: it defers the write-time ``fts_insert`` to
    preserve writer availability, so a skipping stage strands rows that were
    just written rather than merely declining historical backlog. FTS coverage
    is an unconditional convergence invariant, so the stage must answer for the
    sessions belonging to the paths it is given.
    """
    archive_db = tmp_path / "index.db"
    (tmp_path / "index.db").touch()
    source_path = tmp_path / "codex.jsonl"
    _seed_minimal_archive(archive_db, source_path)

    stage = make_fts_stage(tmp_path / "index.db")

    assert stage.check(source_path) is True
    assert stage.execute(source_path) is True
    with sqlite3.connect(archive_db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM messages_fts_docsize").fetchone()[0] > 0
    # Converged: a second pass has nothing left to do.
    assert stage.check(source_path) is False


def test_fts_convergence_has_no_repair_compatibility_or_boolean_route() -> None:
    """Anti-vacuity: a wrapper or truthy result would hide typed owner states."""
    source = inspect.getsource(stages)

    assert "FtsSurfaceRepairResult" not in source
    assert "repair_fts_surface" not in source
    assert "repair_messages_fts_surface" not in source


def test_fts_owner_converges_missing_rows_without_reset(tmp_path: Path) -> None:
    """The canonical owner converges missing rows without a global reset."""
    from unittest import mock

    import polylogue.storage.fts.fts_lifecycle as fts_lc
    from polylogue.daemon.fts_convergence import FtsConvergenceOwner, FtsRunReason

    archive_db = tmp_path / "index.db"
    archive_db.touch()
    source_path = tmp_path / "codex.jsonl"
    _seed_minimal_archive(archive_db, source_path)

    with mock.patch.object(
        fts_lc,
        "reset_message_fts_index_sync",
        wraps=fts_lc.reset_message_fts_index_sync,
    ) as reset_surface:
        result = FtsConvergenceOwner(archive_db, archive_root=tmp_path).run_once_sync(reason=FtsRunReason.DEBT_RETRY)

    assert result.ready
    reset_surface.assert_not_called()
    with sqlite3.connect(archive_db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM messages_fts_docsize").fetchone()[0] == 1


def test_fts_owner_records_exact_parity(tmp_path: Path) -> None:
    """A completed owner pass publishes readiness only after exact parity."""
    from polylogue.daemon.fts_convergence import FtsConvergenceOwner, FtsRunReason

    archive_db = tmp_path / "index.db"
    archive_db.touch()
    source_path = tmp_path / "codex.jsonl"
    _seed_minimal_archive(archive_db, source_path)

    result = FtsConvergenceOwner(archive_db, archive_root=tmp_path).run_once_sync(reason=FtsRunReason.DEBT_RETRY)
    assert result.ready

    with sqlite3.connect(archive_db) as conn:
        row = conn.execute(
            """
            SELECT state, source_rows, indexed_rows, missing_rows, excess_rows
            FROM fts_freshness_state
            WHERE surface = 'messages_fts'
            """
        ).fetchone()
        assert row == ("ready", 1, 1, 0, 0)


def test_fts_owner_records_real_counts_status_and_query_agree(tmp_path: Path) -> None:
    """Owner readiness uses the same exact counts as status and query reads."""
    from polylogue.daemon.fts_convergence import FtsConvergenceOwner, FtsRunReason
    from polylogue.daemon.fts_status import fts_readiness_info
    from polylogue.storage.fts.fts_lifecycle import check_fts_readiness, message_fts_search_readiness_sync

    archive_db = tmp_path / "index.db"
    with sqlite3.connect(archive_db) as conn:
        initialize_archive_tier(conn, ArchiveTier.INDEX)
        session_ids = [
            _seed_index_session(conn, session_id=f"codex-session:s{i}", text=f"needle {i}") for i in range(5)
        ]
        assert len(session_ids) == 5
        # Simulate drift: a bulk write suspended the FTS triggers, so the
        # shadow table lags the real indexable block count.
        conn.execute("DELETE FROM messages_fts")
        conn.commit()

    result = FtsConvergenceOwner(archive_db, archive_root=tmp_path).run_once_sync(reason=FtsRunReason.DEBT_RETRY)
    assert result.ready

    with sqlite3.connect(archive_db) as conn:
        row = conn.execute(
            """
            SELECT state, source_rows, indexed_rows, missing_rows, excess_rows, detail
            FROM fts_freshness_state
            WHERE surface = 'messages_fts'
            """
        ).fetchone()
    assert row == ("ready", 5, 5, 0, 0, None)

    status_view = fts_readiness_info(archive_db, exact=False)
    assert status_view["messages_ready"] is True
    assert status_view["coverage_pct"] == 100.0
    assert status_view["message_indexed_count"] == 5
    assert status_view["message_indexable_count"] == 5

    with sqlite3.connect(archive_db) as conn:
        conn.row_factory = sqlite3.Row
        query_readiness = message_fts_search_readiness_sync(conn)
        assert query_readiness["ready"] is True
        # Must not raise -- the query path agrees with the status surface.
        check_fts_readiness(query_readiness)


def test_fts_surface_debt_retries_after_real_sqlite_backpressure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A locked owner pass remains deferred, red, and restartable until parity is restored."""
    from polylogue.core.outcomes import OutcomeStatus
    from polylogue.daemon import cli as daemon_cli
    from polylogue.daemon import fts_convergence
    from polylogue.daemon.fts_status import fts_readiness_info
    from polylogue.maintenance.archive_verification import verify_archive
    from polylogue.sources.live.cursor import CursorStore

    archive_db = tmp_path / "index.db"
    with sqlite3.connect(archive_db) as conn:
        initialize_archive_tier(conn, ArchiveTier.INDEX)
        for index in range(3):
            _seed_index_session(conn, session_id=f"codex-session:retry-{index}", text=f"retry needle {index}")
        conn.execute("DELETE FROM messages_fts")
        conn.commit()

    cursor = CursorStore(archive_db)
    cursor.record_convergence_debt(
        stage="fts",
        subject_type="fts_surface",
        subject_id="messages_fts",
        error="global FTS repair pending",
    )
    with sqlite3.connect(tmp_path / "ops.db") as conn:
        conn.execute("UPDATE convergence_debt SET next_retry_at = '1970-01-01T00:00:00+00:00'")
        conn.commit()

    blocker = sqlite3.connect(archive_db, timeout=0.01)
    try:
        blocker.execute("BEGIN EXCLUSIVE")
        real_open = fts_convergence.open_daemon_connection

        def open_with_short_timeout(db_path: Path, *, archive_root: Path, **kwargs: object) -> sqlite3.Connection:
            del kwargs
            return real_open(db_path, timeout=0.001, archive_root=archive_root)

        monkeypatch.setattr(fts_convergence, "open_daemon_connection", open_with_short_timeout)
        assert daemon_cli._drain_convergence_debt_once(archive_db) == 1
    finally:
        blocker.rollback()
        blocker.close()

    deferred = [
        debt for debt in cursor.list_convergence_debt() if debt.stage == "fts" and debt.subject_id == "messages_fts"
    ]
    assert len(deferred) == 1
    assert deferred[0].status == "deferred"
    assert fts_readiness_info(archive_db, exact=False)["messages_ready"] is False
    red = verify_archive(tmp_path, checks=("fts-parity",))
    assert red.checks[0].status is OutcomeStatus.ERROR

    with sqlite3.connect(tmp_path / "ops.db") as conn:
        conn.execute("UPDATE convergence_debt SET next_retry_at = '1970-01-01T00:00:00+00:00'")
        conn.commit()
    assert daemon_cli._drain_convergence_debt_once(archive_db) == 1
    assert not [debt for debt in cursor.list_convergence_debt() if debt.subject_id == "messages_fts"]
    assert fts_readiness_info(archive_db, exact=False)["messages_ready"] is True
    green = verify_archive(tmp_path, checks=("fts-parity",))
    assert green.checks[0].status is OutcomeStatus.OK

    cursor.record_convergence_debt(
        stage="fts",
        subject_type="fts_surface",
        subject_id="threads_fts",
        error="unsupported FTS surface",
    )
    with sqlite3.connect(tmp_path / "ops.db") as conn:
        conn.execute("UPDATE convergence_debt SET next_retry_at = '1970-01-01T00:00:00+00:00'")
        conn.commit()
    assert daemon_cli._drain_convergence_debt_once(archive_db) == 1
    unsupported = [debt for debt in cursor.list_convergence_debt() if debt.subject_id == "threads_fts"]
    assert len(unsupported) == 1
    assert unsupported[0].status == "failed"

    cursor.record_convergence_debt(
        stage="fts",
        subject_type="fts_surface",
        subject_id="messages_fts",
        error="global FTS repair pending after malformed shadow table",
    )
    with sqlite3.connect(tmp_path / "ops.db") as conn:
        conn.execute("UPDATE convergence_debt SET next_retry_at = '1970-01-01T00:00:00+00:00'")
        conn.commit()
    with sqlite3.connect(archive_db) as conn:
        conn.execute("DROP TABLE messages_fts_docsize")
        conn.commit()

    assert daemon_cli._drain_convergence_debt_once(archive_db) == 2
    failed = [debt for debt in cursor.list_convergence_debt() if debt.subject_id == "messages_fts"]
    assert len(failed) == 1
    assert failed[0].status == "failed"


def test_default_convergence_stages_always_register_embed_stage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("VOYAGE_API_KEY", "key")
    monkeypatch.delenv("POLYLOGUE_DAEMON_ENABLE_EMBEDDINGS", raising=False)

    stages = make_default_convergence_stages(tmp_path / "archive.sqlite")
    stage_names = [stage.name for stage in stages]

    assert stage_names == [
        "raw_parse_recovery",
        "raw_authority_verdict_cache",
        "attachment_bytes",
        "fts",
        "embed",
        "claude_workflow",
        "delegation_work_evidence",
        "fts_readiness",
        "standing-queries",
    ]
    attachment_stage = next(stage for stage in stages if stage.name == "attachment_bytes")
    assert attachment_stage.false_means_pending is True
    assert attachment_stage.whole_archive is True


def test_embed_stage_is_noop_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("VOYAGE_API_KEY", "key")
    monkeypatch.delenv("POLYLOGUE_DAEMON_ENABLE_EMBEDDINGS", raising=False)
    db_path = tmp_path / "index.db"
    db_path.touch()

    stage = make_embed_stage(db_path)

    assert stage.check(tmp_path / "source.jsonl") is False
    assert stage.execute(tmp_path / "source.jsonl") is True


def test_embed_stage_defers_to_pending_debt_while_predicate_holds(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """During source catch-up the embed stage must not embed inline: every
    execute lane returns False (false_means_pending -> convergence debt)
    without touching the embedding implementation; once the predicate
    releases, execution falls through to the real implementation again."""
    monkeypatch.setattr(stages, "_embedding_config_enabled", lambda: True)
    db_path = tmp_path / "index.db"
    db_path.touch()
    monkeypatch.setattr(stages, "_active_archive_index_path", lambda _db: db_path)

    embed_calls: list[str] = []

    def fake_execute(_db: Path, _path: Path, **_kwargs: object) -> bool:
        embed_calls.append("execute")
        return True

    def fake_execute_many(_db: Path, _paths: object, **_kwargs: object) -> bool:
        embed_calls.append("execute_many")
        return True

    def fake_execute_sessions(_db: Path, _ids: object, **_kwargs: object) -> bool:
        embed_calls.append("execute_sessions")
        return True

    monkeypatch.setattr(stages, "_archive_embed_execute", fake_execute)
    monkeypatch.setattr(stages, "_archive_embed_execute_many", fake_execute_many)
    monkeypatch.setattr(stages, "_archive_embed_execute_sessions", fake_execute_sessions)

    deferring = True
    stage = stages.make_embed_stage(db_path, defer=lambda: deferring)
    assert stage.false_means_pending is True

    source = tmp_path / "source.jsonl"
    assert stage.execute(source) is False
    assert stage.execute_many is not None
    assert stage.execute_many([source]) is False
    assert stage.execute_sessions is not None
    assert stage.execute_sessions(["origin:native"]) is False
    assert embed_calls == []

    deferring = False
    assert stage.execute(source) is True
    assert stage.execute_many([source]) is True
    assert stage.execute_sessions(["origin:native"]) is True
    assert embed_calls == ["execute", "execute_many", "execute_sessions"]


def test_embedding_config_enabled_with_key() -> None:
    """Embedding is enabled when config has both enabled flag and API key."""
    from unittest.mock import patch

    with patch("polylogue.daemon.convergence_stages.load_polylogue_config") as mock_cfg:
        mock_cfg.return_value.embedding_enabled = True
        mock_cfg.return_value.voyage_api_key = "test-key"
        from polylogue.daemon.convergence_stages import _embedding_config_enabled

        assert _embedding_config_enabled() is True


def test_embedding_config_disabled_without_key() -> None:
    """Embedding is disabled when config has enabled flag but no API key."""
    from unittest.mock import patch

    with patch("polylogue.daemon.convergence_stages.load_polylogue_config") as mock_cfg:
        mock_cfg.return_value.embedding_enabled = True
        mock_cfg.return_value.voyage_api_key = None
        from polylogue.daemon.convergence_stages import _embedding_config_enabled

        assert _embedding_config_enabled() is False


def test_embedding_config_disabled_explicitly() -> None:
    """Embedding is disabled when config has key but enabled flag is False."""
    from unittest.mock import patch

    with patch("polylogue.daemon.convergence_stages.load_polylogue_config") as mock_cfg:
        mock_cfg.return_value.embedding_enabled = False
        mock_cfg.return_value.voyage_api_key = "test-key"
        from polylogue.daemon.convergence_stages import _embedding_config_enabled

        assert _embedding_config_enabled() is False


class _ProbeStatements:
    """SQL every connection issues while one health probe runs."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.statements: list[str] = []
        self.active = False
        real_connect = sqlite3.connect

        def counting_connect(*args: Any, **kwargs: Any) -> sqlite3.Connection:
            conn = cast(sqlite3.Connection, real_connect(*args, **kwargs))
            conn.set_trace_callback(self._record)
            return conn

        monkeypatch.setattr(sqlite3, "connect", counting_connect)

    def _record(self, sql: str) -> None:
        if self.active:
            self.statements.append(" ".join(sql.split()))

    @contextmanager
    def measuring(self) -> Iterator[None]:
        self.statements = []
        self.active = True
        try:
            yield
        finally:
            self.active = False

    @property
    def block_aggregates(self) -> list[str]:
        return [sql for sql in self.statements if "blocks" in sql.lower() and "count(" in sql.lower()]


def _publish_fts_readiness(archive_root: Path, sessions: int) -> None:
    archive_db = archive_root / "index.db"
    with sqlite3.connect(archive_db) as conn:
        initialize_archive_tier(conn, ArchiveTier.INDEX)
        for index in range(sessions):
            _seed_index_session(conn, session_id=f"codex-session:probe-{index}", text=f"probe needle {index}")
        conn.commit()
    stage = stages.make_fts_readiness_stage(archive_db)
    assert stage.execute(archive_db) is True


def test_fts_readiness_health_probe_reads_the_published_ledger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """polylogue-t4iy5.11.1: the medium FTS health probe never aggregates the
    archive.

    ``check_health`` runs on the daemon's sole writer and ``/api/status``
    resolves it per request, so an archive-wide aggregate here is paid against
    every catch-up chunk. The exact audit belongs to the ``fts_readiness``
    convergence stage; the probe reports that stage's durable verdict.

    Anti-vacuity: restore ``fts_invariant_snapshot_sync`` inside
    ``_check_fts_readiness_medium`` and the probe issues four COUNT aggregates
    over ``blocks`` per call. The equal-statement assertion additionally
    rejects a per-session or per-surface probe loop.
    """
    from polylogue.daemon.health import HealthSeverity, _check_fts_readiness_medium

    probe = _ProbeStatements(monkeypatch)
    small = tmp_path / "small"
    large = tmp_path / "large"
    small.mkdir()
    large.mkdir()
    _publish_fts_readiness(small, sessions=2)
    _publish_fts_readiness(large, sessions=40)

    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(small))
    with probe.measuring():
        small_alert = _check_fts_readiness_medium()
    small_statements = list(probe.statements)

    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(large))
    with probe.measuring():
        large_alert = _check_fts_readiness_medium()
    large_statements = list(probe.statements)

    assert small_alert.severity is HealthSeverity.OK
    assert large_alert.severity is HealthSeverity.OK
    assert probe.block_aggregates == []
    assert len(large_statements) == len(small_statements)


def test_fts_health_probe_reports_ledger_drift_and_unmeasured_surfaces(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The probe still fails on published drift, and never calls an unmeasured
    surface fresh.

    Anti-vacuity: report ``ready`` from table/trigger presence alone and the
    drifted archive reports OK; treat an unmeasured surface as ready and the
    third assertion reports OK instead of a warning.
    """
    from polylogue.daemon.health import HealthSeverity, _check_fts_readiness_medium

    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    _publish_fts_readiness(archive_root, sessions=3)
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(archive_root))
    assert _check_fts_readiness_medium().severity is HealthSeverity.OK

    archive_db = archive_root / "index.db"
    with sqlite3.connect(archive_db) as conn:
        conn.execute("DELETE FROM messages_fts")
        conn.commit()
    stages.make_fts_readiness_stage(archive_db).execute(archive_db)
    drifted = _check_fts_readiness_medium()
    assert drifted.severity is HealthSeverity.ERROR
    assert "missing row(s)" in drifted.message

    with sqlite3.connect(archive_db) as conn:
        conn.execute("DELETE FROM fts_freshness_state")
        conn.commit()
    unmeasured = _check_fts_readiness_medium()
    assert unmeasured.severity is HealthSeverity.WARNING
    assert "not published yet" in unmeasured.message


@pytest.mark.asyncio
async def test_embed_stage_defers_instead_of_calling_a_provider_under_the_writer_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Under a held writer gate the embed stage records debt; it never embeds.

    The provider call belongs to the lease-free embedding owner, and this is
    the rule that keeps a slow round trip out of the gate for every caller that
    reaches convergence while holding it -- live ingest and debt retry alike.

    Anti-vacuity: drop the ``daemon_write_lease_active`` check in
    ``make_embed_stage`` and ``embedded`` records a call, because the same
    execute path runs the real archive embed route.
    """
    from polylogue.daemon.write_coordinator import DaemonWriteCoordinator

    monkeypatch.setattr(stages, "_embedding_config_enabled", lambda: True)
    monkeypatch.setattr(stages, "_active_archive_index_path", lambda db: db)
    embedded: list[str] = []

    def record_embed(*_args: object, **_kwargs: object) -> bool:
        embedded.append("called")
        return True

    monkeypatch.setattr(stages, "_archive_embed_execute_sessions", record_embed)

    stage = make_embed_stage(tmp_path / "index.db")
    assert stage.execute_sessions is not None
    execute_sessions = stage.execute_sessions

    # Outside the gate the stage does its own work.
    assert bool(execute_sessions(("codex-session:v1-a",))) is True
    assert embedded == ["called"]

    coordinator = DaemonWriteCoordinator(archive_root=tmp_path)
    deferred: list[object] = []

    async def under_gate() -> None:
        deferred.append(execute_sessions(("codex-session:v1-a",)))

    await asyncio.wait_for(coordinator.run("watcher.live_ingest", under_gate), timeout=5.0)

    assert deferred == [False], "a held writer gate must turn the stage into a pending deferral"
    assert embedded == ["called"], "no provider work may run while the writer gate is held"
    assert stage.false_means_pending is True
