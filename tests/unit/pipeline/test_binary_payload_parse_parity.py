"""Binary-payload parse-route parity between ingest and rebuild (polylogue-zoc3).

``ingest_record`` (``pipeline/services/ingest_worker.py``, the live daemon
ingest worker's per-raw entry point) decoded every raw as text/JSON before
any provider dispatch, so a Hermes raw whose payload is SQLite database
bytes (``~/.hermes/verification_evidence.db``) failed with ``"decode: str
is not valid UTF-8: surrogates not allowed: line 1 column 1"`` even though
the rebuild path (``sources.revision_backfill._parse_retained_raw`` /
``_parse_one``, used by historical backfill and the census/replay machinery)
parses the identical raw bytes fine via ``looks_like_sqlite_bytes`` plus the
same structural ``looks_like_*_path`` probes the parser modules expose.

The fix threads the same binary-capable detection
(``polylogue.archive.raw_payload.decode._hermes_sqlite_marker_payload``,
built from the same ``hermes_state``/``hermes_verification`` helper
functions ``_parse_one`` already calls) through
``build_raw_payload_envelope`` -- the function ``ingest_record`` calls to
decode every raw -- so both routes agree.

Anti-vacuity: reverting the ``_hermes_sqlite_marker_payload`` routing in
``polylogue/archive/raw_payload/decode.py`` (i.e. restoring the old
state.db-only ``_looks_like_hermes_state_db`` check) makes
``test_verification_evidence_db_parses_identically_through_ingest_and_rebuild_routes``
fail on the ingest side with exactly the live error string:
``decode: str is not valid UTF-8: surrogates not allowed: line 1 column 1``.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from polylogue.core.enums import Provider
from polylogue.pipeline.services.ingest_worker import ingest_record
from polylogue.sources.revision_backfill import _parse_one
from polylogue.storage.blob_store import BlobStore, reset_blob_store
from polylogue.storage.runtime import RawSessionRecord


@pytest.fixture
def blob_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[BlobStore]:
    root = tmp_path / "blobs"
    store = BlobStore(root)
    monkeypatch.setattr("polylogue.paths.blob_store_root", lambda: root)
    reset_blob_store()
    yield store
    reset_blob_store()


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


def _record(store: BlobStore, content: bytes, *, source_path: str) -> RawSessionRecord:
    raw_id, blob_size = store.write_from_bytes(content)
    return RawSessionRecord(
        raw_id=raw_id,
        source_name="hermes",
        source_path=source_path,
        payload_provider=Provider.HERMES,
        source_index=None,
        blob_size=blob_size,
        acquired_at="2026-01-01T00:00:00+00:00",
        file_mtime=None,
    )


def test_verification_evidence_db_parses_identically_through_ingest_and_rebuild_routes(
    blob_store: BlobStore, tmp_path: Path
) -> None:
    """A raw parseable through the rebuild route is also parseable through ingest_record.

    Compares produced session ``provider_session_id``s (the contract-test
    shape the bead calls for), including the profile-qualified suffix, so a
    regression in either route's profile-root threading also fails this test.
    """
    profile_dir = tmp_path / ".hermes"
    profile_dir.mkdir(parents=True)
    db_path = profile_dir / "verification_evidence.db"
    _write_verification_evidence_db(db_path)
    content = db_path.read_bytes()

    # Rebuild route: the same _parse_one the backfill/census/replay machinery
    # calls (parse_retained_raw_sessions -> _parse_one; census_parse_worker -> _parse_one).
    rebuild_sessions = _parse_one(
        Provider.HERMES,
        content,
        str(db_path),
        payload_path=db_path,
        archive_root=tmp_path,
    )
    rebuild_ids = {session.provider_session_id for session in rebuild_sessions}
    assert rebuild_ids, "fixture must produce at least one session on the rebuild route"

    # Ingest route: the live daemon worker's per-raw entry point.
    record = _record(blob_store, content, source_path=str(db_path))
    result = ingest_record(record, str(tmp_path / "archive"), "advisory", blob_root_str=str(blob_store.root))

    assert result.error is None, result.error
    ingest_ids = {payload.parsed_session.provider_session_id for payload in result.sessions}

    assert ingest_ids == rebuild_ids
    assert ingest_ids == {"verification:verify-session-redacted-1@profile-" + _profile_key(profile_dir)}


def _profile_key(profile_dir: Path) -> str:
    from polylogue.sources.parsers.hermes_identity import profile_key

    return profile_key(profile_dir)


def _write_hermes_state_db(path: Path) -> None:
    """Minimal state.db fixture covering every column ``_has_required_tables`` checks.

    Independent minimal copy of the shape
    ``tests/unit/sources/test_parsers_local_agent.py``'s ``_write_hermes_state_db``
    builds in full -- only the required + signature columns
    (``hermes_state._REQUIRED_SESSION_COLUMNS`` / ``_REQUIRED_MESSAGE_COLUMNS`` /
    ``_HERMES_SIGNATURE_*_COLUMNS``) are needed for this parity check.
    """
    with sqlite3.connect(path) as conn:
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


def test_state_db_still_parses_identically_through_both_routes(blob_store: BlobStore, tmp_path: Path) -> None:
    """Regression guard: the pre-existing state.db marker route must keep working
    after generalizing the SQLite marker helper to also recognize verification_evidence.db.
    """
    profile_dir = tmp_path / ".hermes"
    profile_dir.mkdir(parents=True)
    db_path = profile_dir / "state.db"
    _write_hermes_state_db(db_path)

    content = db_path.read_bytes()

    rebuild_sessions = _parse_one(
        Provider.HERMES,
        content,
        str(db_path),
        payload_path=db_path,
        archive_root=tmp_path,
    )
    rebuild_ids = {session.provider_session_id.split("@", 1)[0] for session in rebuild_sessions}

    record = _record(blob_store, content, source_path=str(db_path))
    result = ingest_record(record, str(tmp_path / "archive"), "advisory", blob_root_str=str(blob_store.root))

    assert result.error is None, result.error
    ingest_ids = {payload.parsed_session.provider_session_id.split("@", 1)[0] for payload in result.sessions}

    assert ingest_ids == rebuild_ids
    assert "hermes-root" in ingest_ids
