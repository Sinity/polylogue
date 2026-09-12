"""Red twins for parent-session source/index accounting.

Anti-vacuity: each test plants a real ``session_links`` assertion and a
retained raw.  Removing the exact-origin join, skipping raw rows, or treating
``parsed_at_ms`` as success makes the expected blocking result disappear.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from polylogue.core.outcomes import OutcomeStatus
from polylogue.maintenance.archive_verification import ArchiveVerificationCheck, verify_archive
from polylogue.maintenance.parent_session_accounting import audit_parent_session_accounting
from polylogue.sources.origin_specs import lowering_fingerprint, parser_fingerprint_for_origin
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root


def _session(conn: sqlite3.Connection, *, origin: str, native_id: str, raw_id: str | None = None) -> None:
    conn.execute(
        """
        INSERT INTO sessions(
            native_id, origin, raw_id, parser_fingerprint, lowering_fingerprint, content_hash, message_count
        ) VALUES (?, ?, ?, ?, ?, ?, 1)
        """,
        (native_id, origin, raw_id, parser_fingerprint_for_origin(origin), lowering_fingerprint(), b"s" * 32),
    )


def _raw(
    conn: sqlite3.Connection,
    *,
    root: Path,
    raw_id: str,
    origin: str,
    native_id: str,
    payload: bytes = b"parent transcript",
    parsed: bool = True,
    parse_error: str | None = None,
) -> None:
    path = root / f"{raw_id}.jsonl"
    path.write_bytes(payload)
    digest = BlobStore(root / "blob").write_from_bytes(payload)[0]
    digest_bytes = bytes.fromhex(str(digest))
    conn.execute(
        """
        INSERT INTO raw_sessions(
            raw_id, origin, native_id, source_path, blob_hash, blob_size, acquired_at_ms,
            parsed_at_ms, parse_error, revision_authority
        ) VALUES (?, ?, ?, ?, ?, ?, 100, ?, ?, 'byte_proven')
        """,
        (raw_id, origin, native_id, str(path), digest_bytes, len(payload), 100 if parsed else None, parse_error),
    )
    conn.execute(
        """
        INSERT INTO blob_refs(blob_hash, ref_id, ref_type, source_path, size_bytes, acquired_at_ms)
        VALUES (?, ?, 'raw_payload', ?, ?, 100)
        """,
        (digest_bytes, raw_id, str(path), len(payload)),
    )


def _link(conn: sqlite3.Connection, *, child: str, origin: str, native_id: str) -> None:
    conn.execute(
        """
        INSERT INTO session_links(
            src_session_id, dst_origin, dst_native_id, link_type, observed_at_ms
        ) VALUES (?, ?, ?, 'subagent', 100)
        """,
        (child, origin, native_id),
    )


def _archive(tmp_path: Path) -> tuple[sqlite3.Connection, sqlite3.Connection]:
    initialize_active_archive_root(tmp_path)
    source = sqlite3.connect(tmp_path / "source.db")
    index = sqlite3.connect(tmp_path / "index.db")
    _session(index, origin="claude-code-session", native_id="child")
    source.execute("PRAGMA foreign_keys = ON")
    index.execute("PRAGMA foreign_keys = ON")
    return source, index


def test_parent_accounting_conserves_identity_and_explicit_unavailable_states(tmp_path: Path) -> None:
    source, index = _archive(tmp_path)
    try:
        _raw(source, root=tmp_path, raw_id="raw-claude", origin="claude-code-session", native_id="parent")
        _raw(
            source,
            root=tmp_path,
            raw_id="raw-codex-gone",
            origin="codex-session",
            native_id="gone",
            payload=b"gone parent transcript",
        )
        source.execute("DELETE FROM blob_refs WHERE ref_id = 'raw-codex-gone'")
        (tmp_path / "raw-codex-gone.jsonl").unlink()
        _link(index, child="claude-code-session:child", origin="claude-code-session", native_id="parent")
        _link(index, child="claude-code-session:child", origin="codex-session", native_id="gone")
        _link(index, child="claude-code-session:child", origin="codex-session", native_id="never-acquired")
        _session(index, origin="claude-code-session", native_id="parent", raw_id="raw-claude")
        index.execute(
            """
            UPDATE session_links
            SET resolved_dst_session_id = 'claude-code-session:parent'
            WHERE dst_origin = 'claude-code-session' AND dst_native_id = 'parent'
            """
        )
        source.commit()
        index.commit()

        report = audit_parent_session_accounting(source, index, archive_root=tmp_path)
    finally:
        source.close()
        index.close()

    assert report.available
    assert report.reference_total == 3
    assert report.unique_parent_total == 3
    assert report.materialized_parent_total == 1
    assert report.source_unavailable_total == 1
    assert report.not_acquired_total == 1
    assert report.available_unmaterialized_total == 0
    assert {entry.disposition for entry in report.references} == {
        "materialized",
        "source_unavailable",
        "not_acquired",
    }


def test_parent_identity_never_matches_native_id_from_another_origin(tmp_path: Path) -> None:
    source, index = _archive(tmp_path)
    try:
        _raw(source, root=tmp_path, raw_id="raw-claude", origin="claude-code-session", native_id="same-id")
        _link(index, child="claude-code-session:child", origin="codex-session", native_id="same-id")
        source.commit()
        index.commit()
        report = audit_parent_session_accounting(source, index, archive_root=tmp_path)
    finally:
        source.close()
        index.close()

    assert report.not_acquired_total == 1
    assert report.materialized_parent_total == 0
    assert report.references[0].source_raw_ids == ()


def test_parsed_available_parent_without_candidate_session_is_blocking(tmp_path: Path) -> None:
    source, index = _archive(tmp_path)
    try:
        _raw(source, root=tmp_path, raw_id="raw-missing", origin="codex-session", native_id="missing")
        _link(index, child="claude-code-session:child", origin="codex-session", native_id="missing")
        source.commit()
        index.commit()
        report = audit_parent_session_accounting(source, index, archive_root=tmp_path)
    finally:
        source.close()
        index.close()

    assert report.available_unmaterialized_total == 1
    assert report.blocking_count > 0
    assert report.raw_unexplained_total == 1
    assert report.raw_disposition_counts == {"untyped_unmaterialized": 1}

    check = verify_archive(
        tmp_path,
        checks=("parent-session-accounting",),
        index_path_override=tmp_path / "index.db",
    ).checks[0]
    assert isinstance(check, ArchiveVerificationCheck)
    assert check.status is OutcomeStatus.ERROR
    assert check.evidence["untyped_total"] == 1
    assert check.evidence["untyped_denominator"] == 1


def test_materialized_parent_with_unresolved_reference_is_blocking(tmp_path: Path) -> None:
    source, index = _archive(tmp_path)
    try:
        _raw(source, root=tmp_path, raw_id="raw-parent", origin="codex-session", native_id="parent")
        _session(index, origin="codex-session", native_id="parent", raw_id="raw-parent")
        _link(index, child="claude-code-session:child", origin="codex-session", native_id="parent")
        source.commit()
        index.commit()
        report = audit_parent_session_accounting(source, index, archive_root=tmp_path)
    finally:
        source.close()
        index.close()

    assert report.materialized_parent_total == 0
    assert report.unresolved_reference_total == 1
    assert report.references[0].disposition == "materialized_unresolved"
    assert report.references[0].resolved_reference_count == 0
    assert report.blocking_count == 1

    check = verify_archive(
        tmp_path,
        checks=("parent-session-accounting",),
        index_path_override=tmp_path / "index.db",
    ).checks[0]
    assert isinstance(check, ArchiveVerificationCheck)
    assert check.status is OutcomeStatus.ERROR
    assert check.evidence["unresolved_reference_total"] == 1


def test_unmaterialized_parent_raw_with_parse_refusal_is_conserved(tmp_path: Path) -> None:
    source, index = _archive(tmp_path)
    try:
        _raw(
            source,
            root=tmp_path,
            raw_id="raw-refused",
            origin="claude-code-session",
            native_id="refused",
            parse_error="malformed JSON",
        )
        _link(index, child="claude-code-session:child", origin="claude-code-session", native_id="refused")
        source.commit()
        index.commit()
        report = audit_parent_session_accounting(source, index, archive_root=tmp_path)
    finally:
        source.close()
        index.close()

    assert report.available_unmaterialized_total == 1
    assert report.raw_unexplained_total == 0
    assert report.raw_disposition_counts == {"parse_failure": 1}
    assert report.blocking_count == 1
