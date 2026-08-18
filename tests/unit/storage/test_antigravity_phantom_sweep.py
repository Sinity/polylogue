"""polylogue-eo81 / polylogue-msia: sweep + purge for antigravity phantom sessions.

Every ``antigravity-session`` row in the live archive was a 1-message
fragment materialized from a ``*.md.metadata.json`` brain-artifact sidecar
(``degraded:brain-metadata-fragment`` auto-tag, PR #1856) -- real
conversation content lives in ``conversations/*.pb`` and is acquired
separately (PR #3441). These tests build the phantom shape through the real
production parser (:func:`parse_brain_metadata`) and the real archive writer
(:meth:`ArchiveStore.write_raw_and_parsed`), then exercise the read-only
sweep and the ``--apply``-gated purge actuator end to end against a
throwaway archive -- never the live one.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from polylogue.archive.artifact_taxonomy import ArtifactKind
from polylogue.core.enums import Origin
from polylogue.sources.parsers.antigravity import BRAIN_METADATA_FRAGMENT_FLAG, parse_brain_metadata
from polylogue.storage.antigravity_phantom_sweep import scan_antigravity_phantom_sessions
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

_METADATA_PAYLOAD: dict[str, Any] = {
    "artifactType": "ARTIFACT_TYPE_OTHER",
    "summary": "Some brain-artifact summary",
    "updatedAt": "2026-04-01T10:00:00Z",
}


def _seed_phantom_session(
    archive: ArchiveStore,
    tmp_path: Path,
    *,
    work_dir: str,
    artifact_name: str,
    acquired_at_ms: int,
) -> tuple[str, str]:
    """Write one real brain-metadata phantom session; return (session_id, raw_id)."""
    session_dir = tmp_path / "brain" / work_dir
    session_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = session_dir / f"{artifact_name}.md.metadata.json"
    payload_bytes = json.dumps(_METADATA_PAYLOAD).encode("utf-8")
    metadata_path.write_bytes(payload_bytes)

    session = parse_brain_metadata(_METADATA_PAYLOAD, metadata_path, work_dir)
    assert BRAIN_METADATA_FRAGMENT_FLAG in session.ingest_flags
    assert session.messages and len(session.messages) == 1

    # Raw ids are content-derived: raw admission (polylogue-1fijp) is the sole
    # creator of raw_sessions rows and assigns the id from the payload, so the
    # requested value is a hint and the returned one is authoritative.
    raw_id = f"raw-{work_dir}-{artifact_name}"
    returned_raw_id, session_id = archive.write_raw_and_parsed(
        session,
        payload=payload_bytes,
        source_path=str(metadata_path),
        acquired_at_ms=acquired_at_ms,
        raw_id=raw_id,
    )
    return session_id, returned_raw_id


def _real_session(archive: ArchiveStore, tmp_path: Path, *, acquired_at_ms: int) -> tuple[str, str]:
    """A non-phantom antigravity session (language-server export shape): two
    real messages, no degraded flag -- must never be swept or purged."""
    from polylogue.sources.parsers.antigravity import AntigravitySessionSummary, parse_markdown_export

    source_path = tmp_path / "conversations" / "real-cascade.md"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    markdown = "# Real conversation\n\nUser: hello\n\nAssistant: hi there\n"
    source_path.write_text(markdown, encoding="utf-8")
    summary = AntigravitySessionSummary(
        cascade_id="real-cascade",
        title="Real conversation",
        workspace_name=None,
        snippet=None,
        last_modified_time=None,
    )
    session = parse_markdown_export(markdown, summary)
    assert BRAIN_METADATA_FRAGMENT_FLAG not in session.ingest_flags

    payload_bytes = markdown.encode("utf-8")
    raw_id = "raw-real-cascade"
    returned_raw_id, session_id = archive.write_raw_and_parsed(
        session,
        payload=payload_bytes,
        source_path=str(source_path),
        acquired_at_ms=acquired_at_ms,
        raw_id=raw_id,
    )
    return session_id, returned_raw_id


def test_scan_finds_only_tagged_phantom_sessions(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    with ArchiveStore(root) as archive:
        phantom_id, phantom_raw_id = _seed_phantom_session(
            archive,
            tmp_path,
            work_dir="7022699c-work",
            artifact_name="plan",
            acquired_at_ms=1_800_000_000_000,
        )
        real_id, real_raw_id = _real_session(archive, tmp_path, acquired_at_ms=1_800_000_000_100)

    index_conn = sqlite3.connect(root / "index.db")
    index_conn.row_factory = sqlite3.Row
    source_conn = sqlite3.connect(root / "source.db")
    source_conn.row_factory = sqlite3.Row
    try:
        plan = scan_antigravity_phantom_sessions(index_conn, source_conn)
    finally:
        index_conn.close()
        source_conn.close()

    assert plan.scanned_count == 1
    assert len(plan.candidates) == 1
    candidate = plan.candidates[0]
    assert candidate.session_id == phantom_id
    assert candidate.raw_id == phantom_raw_id
    assert candidate.source_path is not None
    assert candidate.source_path.endswith(".md.metadata.json")
    assert plan.missing_raw_row_count == 0
    assert candidate.session_id != real_id
    assert phantom_raw_id != real_raw_id


def test_scan_reports_missing_raw_row_honestly(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    with ArchiveStore(root) as archive:
        phantom_id, phantom_raw_id = _seed_phantom_session(
            archive,
            tmp_path,
            work_dir="gc-work",
            artifact_name="report",
            acquired_at_ms=1_800_000_000_000,
        )

    # Simulate the raw row already having been GC'd out from under a still-tagged session.
    source_conn = sqlite3.connect(root / "source.db")
    try:
        source_conn.execute("DELETE FROM raw_sessions WHERE raw_id = ?", (phantom_raw_id,))
        source_conn.commit()
    finally:
        source_conn.close()

    index_conn = sqlite3.connect(root / "index.db")
    index_conn.row_factory = sqlite3.Row
    source_conn = sqlite3.connect(root / "source.db")
    source_conn.row_factory = sqlite3.Row
    try:
        plan = scan_antigravity_phantom_sessions(index_conn, source_conn)
    finally:
        index_conn.close()
        source_conn.close()

    assert len(plan.candidates) == 1
    assert plan.candidates[0].session_id == phantom_id
    assert plan.candidates[0].source_path is None
    assert plan.missing_raw_row_count == 1


def test_scan_works_without_source_connection(tmp_path: Path) -> None:
    """The session-tag alone is sufficient evidence; source_conn is optional."""
    root = tmp_path / "archive"
    with ArchiveStore(root) as archive:
        phantom_id, _ = _seed_phantom_session(
            archive,
            tmp_path,
            work_dir="no-source-work",
            artifact_name="task",
            acquired_at_ms=1_800_000_000_000,
        )

    index_conn = sqlite3.connect(root / "index.db")
    index_conn.row_factory = sqlite3.Row
    try:
        plan = scan_antigravity_phantom_sessions(index_conn, None)
    finally:
        index_conn.close()

    assert len(plan.candidates) == 1
    assert plan.candidates[0].session_id == phantom_id
    assert plan.candidates[0].source_path is None
    assert plan.missing_raw_row_count == 0  # not tracked when source_conn is absent


def test_purge_apply_dry_run_deletes_nothing(tmp_path: Path) -> None:
    from devtools.antigravity_phantom_purge_apply import main as purge_main

    root = tmp_path / "archive"
    with ArchiveStore(root) as archive:
        phantom_id, _ = _seed_phantom_session(
            archive,
            tmp_path,
            work_dir="dry-run-work",
            artifact_name="notes",
            acquired_at_ms=1_800_000_000_000,
        )

    exit_code = purge_main(["--archive-root", str(root), "--json"])
    assert exit_code == 0

    index_conn = sqlite3.connect(root / "index.db")
    try:
        count = index_conn.execute("SELECT COUNT(*) FROM sessions WHERE session_id = ?", (phantom_id,)).fetchone()[0]
    finally:
        index_conn.close()
    assert count == 1, "dry-run must not delete the session"


def test_purge_apply_deletes_sessions_and_reclassifies_raw_rows(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from devtools.antigravity_phantom_purge_apply import main as purge_main

    root = tmp_path / "archive"
    with ArchiveStore(root) as archive:
        phantom_id, phantom_raw_id = _seed_phantom_session(
            archive,
            tmp_path,
            work_dir="apply-work",
            artifact_name="summary",
            acquired_at_ms=1_800_000_000_000,
        )
        real_id, real_raw_id = _real_session(archive, tmp_path, acquired_at_ms=1_800_000_000_100)

    exit_code = purge_main(["--archive-root", str(root), "--apply", "--json"])
    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["applied"] is True
    assert output["deleted_count"] == 1
    assert output["raw_artifacts_observations_written"] == 1

    index_conn = sqlite3.connect(root / "index.db")
    index_conn.row_factory = sqlite3.Row
    try:
        remaining = index_conn.execute("SELECT session_id, origin FROM sessions").fetchall()
    finally:
        index_conn.close()
    remaining_ids = {row["session_id"] for row in remaining}
    assert phantom_id not in remaining_ids, "phantom session must be purged"
    assert real_id in remaining_ids, "real antigravity session must survive"
    assert all(row["origin"] == Origin.ANTIGRAVITY_SESSION.value for row in remaining)

    source_conn = sqlite3.connect(root / "source.db")
    source_conn.row_factory = sqlite3.Row
    try:
        artifact_row = source_conn.execute(
            "SELECT artifact_kind FROM raw_artifacts WHERE raw_id = ?", (phantom_raw_id,)
        ).fetchone()
        real_artifact_row = source_conn.execute(
            "SELECT artifact_kind FROM raw_artifacts WHERE raw_id = ?", (real_raw_id,)
        ).fetchone()
    finally:
        source_conn.close()
    assert artifact_row is not None
    assert artifact_row["artifact_kind"] == ArtifactKind.AGENT_SIDECAR_META.value
    # The real session's raw row must not have been touched by this scoped run.
    assert real_artifact_row is None
