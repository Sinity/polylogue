"""Hermes logical SQLite exports replay through the retained Raw owner."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from polylogue.core.enums import Provider
from polylogue.core.provider_identity import captured_hermes_profile_key
from polylogue.sources.revision_backfill import _parse_one
from polylogue.sources.sqlite_export import logical_export_bytes
from polylogue.sources.sqlite_snapshot import member_export_scope
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from tests.infra.archive_templates import bootstrap_archive_root, run_off_event_loop
from tests.infra.retained_replay import replay_retained_components


def _write_verification_evidence_db(path: Path) -> None:
    """Build the live-verified verification_evidence.db schema (schema_version=1).

    Mirrors ``tests/unit/sources/parsers/test_hermes_verification.py``'s
    ``_write_verification_evidence_db`` (same table shape, redacted-placeholder
    row values) -- kept as an independent minimal copy so this parity test does
    not import test internals across files.
    """
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE verification_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                session_id TEXT NOT NULL,
                cwd TEXT NOT NULL,
                root TEXT NOT NULL,
                command TEXT NOT NULL,
                canonical_command TEXT NOT NULL,
                kind TEXT NOT NULL,
                scope TEXT NOT NULL,
                status TEXT NOT NULL,
                exit_code INTEGER NOT NULL,
                output_summary TEXT NOT NULL
            );
            CREATE TABLE verification_state (
                session_id TEXT NOT NULL,
                root TEXT NOT NULL,
                last_event_id INTEGER,
                last_edit_at TEXT,
                changed_paths_json TEXT NOT NULL DEFAULT '[]',
                PRIMARY KEY (session_id, root)
            );
            CREATE INDEX idx_verification_events_session_root
                ON verification_events(session_id, root, id DESC);
            INSERT INTO meta(key, value) VALUES ('schema_version', '1');
            """
        )
        conn.execute(
            "INSERT INTO verification_events "
            "(created_at, session_id, cwd, root, command, canonical_command, kind, scope, status, exit_code, output_summary) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "2026-07-14T17:21:02.133901+00:00",
                "verify-session-redacted-1",
                "<redacted>",
                "<redacted>",
                "<redacted>",
                "pytest",
                "test",
                "targeted",
                "passed",
                0,
                "<redacted>",
            ),
        )
        conn.commit()


def _retained_bytes(db_path: Path) -> bytes:
    """The material acquisition actually retains for a declared member.

    A mutable SQLite source is not its own bytes: ``snapshot_sqlite_to_blob``
    retains the declared member's canonical logical export, and that export's
    blob hash is the member's logical revision. Building the fixture through
    the same call keeps this parity check on the material both routes will
    really be handed, instead of a page image neither route may admit.
    """
    return logical_export_bytes(db_path, scope=member_export_scope(db_path))


def _retained_blob(db_path: Path) -> tuple[bytes, Path]:
    """Retain the declared export beside the fixture and return it with its path.

    The replay route is handed the retained blob's path, never the live
    database's: ``payload_path`` is the on-disk blob ``parse_retained_raw_sessions``
    resolved, and pointing it back at the mutable source would check a page
    image the archive never retained.
    """
    content = _retained_bytes(db_path)
    blob_path = db_path.with_name(f"{db_path.name}.retained-export")
    blob_path.write_bytes(content)
    return content, blob_path


def _retained_replay(tmp_path: Path, content: bytes, source_path: Path, profile_dir: Path) -> Path:
    archive_root = tmp_path / "archive"
    run_off_event_loop(lambda: bootstrap_archive_root(archive_root))

    def acquire() -> None:
        with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
            archive.write_raw_payload(
                provider=Provider.HERMES,
                payload=content,
                source_path=str(source_path),
                canonical_source_path=str(source_path),
                captured_profile_key=captured_hermes_profile_key(profile_dir),
                acquired_at_ms=1,
            )
            archive.commit()

    run_off_event_loop(acquire)
    replay_retained_components(archive_root)
    return archive_root


def test_verification_evidence_db_replays_as_a_profile_qualified_session(tmp_path: Path) -> None:
    profile_dir = tmp_path / ".hermes"
    profile_dir.mkdir(parents=True)
    db_path = profile_dir / "verification_evidence.db"
    _write_verification_evidence_db(db_path)
    content, retained_path = _retained_blob(db_path)

    # Rebuild route: the same _parse_one the backfill/census/replay machinery
    # calls (parse_retained_raw_sessions -> _parse_one; census_parse_worker -> _parse_one).
    rebuild_sessions = _parse_one(
        Provider.HERMES,
        content,
        str(db_path),
        payload_path=retained_path,
        archive_root=tmp_path,
        profile_identity=captured_hermes_profile_key(profile_dir),
        sidecar_resolver=None,
    )
    rebuild_ids = {session.provider_session_id for session in rebuild_sessions}
    assert rebuild_ids, "fixture must produce at least one session on the rebuild route"

    archive_root = _retained_replay(tmp_path, content, db_path, profile_dir)
    with sqlite3.connect(archive_root / "index.db") as conn:
        replay_ids = {str(row[0]) for row in conn.execute("SELECT native_id FROM sessions")}
    assert replay_ids == rebuild_ids
    assert replay_ids == {"verification:verify-session-redacted-1@profile-" + _profile_key(profile_dir)}


def _profile_key(profile_dir: Path) -> str:
    from polylogue.sources.parsers.hermes_identity import profile_key

    return profile_key(profile_dir)


def _write_hermes_state_db(path: Path, *, wal_mode: bool = False) -> None:
    """Minimal state.db fixture covering every column ``_has_required_tables`` checks.

    Independent minimal copy of the shape
    ``tests/unit/sources/test_parsers_local_agent.py``'s ``_write_hermes_state_db``
    builds in full -- only the required + signature columns
    (``hermes_state._REQUIRED_SESSION_COLUMNS`` / ``_REQUIRED_MESSAGE_COLUMNS`` /
    ``_HERMES_SIGNATURE_*_COLUMNS``) are needed for this parity check.
    """
    with sqlite3.connect(path) as conn:
        if wal_mode:
            conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(
            """
            CREATE TABLE schema_version(version INTEGER NOT NULL);
            INSERT INTO schema_version(version) VALUES (16);
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY,
                source TEXT,
                model_config TEXT,
                parent_session_id TEXT,
                started_at REAL
            );
            CREATE TABLE messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT,
                tool_calls TEXT,
                timestamp REAL NOT NULL,
                observed INTEGER DEFAULT 0,
                active INTEGER NOT NULL DEFAULT 1,
                compacted INTEGER NOT NULL DEFAULT 0
            );
            """
        )
        conn.execute(
            "INSERT INTO sessions (id, model_config, started_at) VALUES (?, ?, ?)",
            ("hermes-root", "{}", 1_775_000_000.0),
        )
        conn.execute(
            "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
            ("hermes-root", "assistant", "hi from state.db", 1_775_000_001.0),
        )
        conn.commit()
        if wal_mode:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")


def test_state_db_replays_through_retained_route(tmp_path: Path) -> None:
    """Regression guard: the pre-existing state.db marker route must keep working
    after generalizing the SQLite marker helper to also recognize verification_evidence.db.
    """
    profile_dir = tmp_path / ".hermes"
    profile_dir.mkdir(parents=True)
    db_path = profile_dir / "state.db"
    _write_hermes_state_db(db_path)

    content, retained_path = _retained_blob(db_path)

    rebuild_sessions = _parse_one(
        Provider.HERMES,
        content,
        str(db_path),
        payload_path=retained_path,
        archive_root=tmp_path,
        profile_identity=captured_hermes_profile_key(profile_dir),
        sidecar_resolver=None,
    )
    rebuild_ids = {session.provider_session_id.split("@", 1)[0] for session in rebuild_sessions}

    archive_root = _retained_replay(tmp_path, content, db_path, profile_dir)
    with sqlite3.connect(archive_root / "index.db") as conn:
        replay_ids = {str(row[0]).split("@", 1)[0] for row in conn.execute("SELECT native_id FROM sessions")}
    assert replay_ids == rebuild_ids
    assert "hermes-root" in replay_ids


def test_retained_replay_keeps_wal_sqlite_blob_namespace_pristine(tmp_path: Path) -> None:
    """The real retained-blob decode route never creates SQLite sidecars.

    A WAL-mode SQLite header makes a plain ``mode=ro`` connection create
    ``-wal`` and ``-shm`` beside the immutable blob. The live
    The retained route must use immutable SQLite reads, leaving one canonical
    blob and no invalid namespace entries.
    """
    profile_dir = tmp_path / ".hermes"
    profile_dir.mkdir(parents=True)
    db_path = profile_dir / "state.db"
    _write_hermes_state_db(db_path, wal_mode=True)
    archive_root = _retained_replay(tmp_path, _retained_bytes(db_path), db_path, profile_dir)
    verified = BlobStore(archive_root / "blob").verify_all()

    assert verified.checked == 1
    assert verified.passed
    assert verified.checked == 1
    assert verified.failures == ()


def test_retained_route_refuses_a_hermes_page_image(tmp_path: Path) -> None:
    """Both routes refuse a historical SQLite page image for a declared member.

    A page image cannot be proven against the live database and re-snapshots on
    every commit, so it is not the retained material for ``state.db``. The
    The retained route must not admit the page image as a Hermes session.
    """
    profile_dir = tmp_path / ".hermes"
    profile_dir.mkdir(parents=True)
    db_path = profile_dir / "state.db"
    _write_hermes_state_db(db_path)
    page_image = db_path.read_bytes()

    with pytest.raises(RuntimeError, match="is not a logical export"):
        _parse_one(
            Provider.HERMES,
            page_image,
            str(db_path),
            payload_path=db_path,
            archive_root=tmp_path,
            profile_identity=captured_hermes_profile_key(profile_dir),
            sidecar_resolver=None,
        )

    archive_root = _retained_replay(tmp_path, page_image, db_path, profile_dir)
    with sqlite3.connect(archive_root / "index.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone() == (0,)
