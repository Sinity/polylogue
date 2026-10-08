"""Red twins for the ``source-conservation`` owner check of ``verify-archive``.

Each test names the mutation that would make it vacuous: a deleted source
file, an injected unadmitted index row, a session materialized from a
declared non-session artifact, a fragment-shaped identity, a parsed raw that
no rule explains. The archive is built through the production tier bootstrap
with real source files under ``tmp_path``; no ambient data is read.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sqlite3
import zipfile
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from polylogue.config import Source
from polylogue.core.enums import Provider
from polylogue.core.outcomes import OutcomeStatus
from polylogue.maintenance.archive_verification import (
    ArchiveVerificationCheck,
    ArchiveVerificationReport,
    archive_verification_names_for_route,
    verify_archive,
)
from polylogue.maintenance.source_conservation import (
    FRAGMENT_IDENTITY_PREFIXES,
    fragment_identity_shape,
)
from polylogue.maintenance.source_manifest_continuity import (
    SourceDeclaration,
    SourceFrontier,
    SourceRole,
    build_source_frontier,
)
from polylogue.sources.origin_specs import lowering_fingerprint, parser_fingerprint_for_origin
from polylogue.sources.source_acquisition_components import ZipEntryReadContext, iter_zip_entry_raw_data
from polylogue.sources.sqlite_snapshot import sqlite_member_revision
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

CHECK = "source-conservation"


def _check(report: ArchiveVerificationReport) -> ArchiveVerificationCheck:
    matches = [c for c in report.checks if c.name == CHECK]
    assert len(matches) == 1
    match = matches[0]
    assert isinstance(match, ArchiveVerificationCheck)
    return match


def _terms(check: ArchiveVerificationCheck) -> dict[str, dict[str, object]]:
    terms = check.evidence["terms"]
    assert isinstance(terms, dict)
    return terms


def _count(check: ArchiveVerificationCheck, term: str) -> int:
    value = _terms(check)[term]["count"]
    assert isinstance(value, int)
    return value


def _write_source(root: Path, name: str, payload: bytes) -> Path:
    path = root / "sources" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _insert_raw(
    conn: sqlite3.Connection,
    *,
    raw_id: str,
    origin: str,
    native_id: str | None,
    source_path: Path,
    blob_hash: str,
    parsed: bool,
) -> None:
    conn.execute(
        """
        INSERT INTO raw_sessions(
            raw_id, origin, native_id, source_path, blob_hash, blob_size, acquired_at_ms, parsed_at_ms
        ) VALUES (?, ?, ?, ?, ?, 10, 100, ?)
        """,
        (raw_id, origin, native_id, str(source_path), bytes.fromhex(blob_hash), 100 if parsed else None),
    )
    conn.execute(
        """
        INSERT INTO blob_refs(blob_hash, ref_id, ref_type, source_path, size_bytes, acquired_at_ms)
        VALUES (?, ?, 'raw_payload', ?, 10, 100)
        """,
        (bytes.fromhex(blob_hash), raw_id, str(source_path)),
    )


def _insert_artifact(
    conn: sqlite3.Connection,
    *,
    raw_id: str,
    origin: str,
    source_path: Path,
    kind: str,
    support: str,
    parse_as_session: bool,
) -> None:
    conn.execute(
        """
        INSERT INTO raw_artifacts(
            artifact_id, raw_id, origin, source_path, artifact_kind, support_status,
            classification_reason, parse_as_session, first_observed_at_ms, last_observed_at_ms
        ) VALUES (?, ?, ?, ?, ?, ?, 'test', ?, 100, 100)
        """,
        (f"artifact-{raw_id}", raw_id, origin, str(source_path), kind, support, int(parse_as_session)),
    )


def _insert_session(
    conn: sqlite3.Connection,
    *,
    origin: str,
    native_id: str,
    raw_id: str | None,
    content_hash: bytes = b"s" * 32,
) -> str:
    conn.execute(
        """
        INSERT INTO sessions(
            native_id, origin, raw_id, parser_fingerprint, lowering_fingerprint, content_hash, message_count
        ) VALUES (?, ?, ?, ?, ?, ?, 1)
        """,
        (native_id, origin, raw_id, parser_fingerprint_for_origin(origin), lowering_fingerprint(), content_hash),
    )
    session_id = f"{origin}:{native_id}"
    conn.execute(
        """
        INSERT INTO messages(session_id, position, role, material_origin, content_hash, content_identity)
        VALUES (?, 0, 'user', 'human_authored', ?, '0123456789abcdef0123456789abcdef')
        """,
        (session_id, b"m" * 32),
    )
    conn.execute(
        """
        INSERT INTO blocks(message_id, session_id, position, block_type, text)
        VALUES (?, ?, 0, 'text', 'hello world')
        """,
        (f"{session_id}:c:0123456789abcdef0123456789abcdef.0", session_id),
    )
    return session_id


def _record_member_fixture(
    conn: sqlite3.Connection,
    bundle: Path,
    raw_id: str = "raw-session",
    member: str = "conversations.json",
    entry_ordinal: int = 0,
) -> None:
    from polylogue.core.raw_coordinates import CapturedZipMemberCoordinate, MemberAddressingMode
    from polylogue.sources.source_acquisition_components import zip_acquisition_fingerprint
    from polylogue.storage.sqlite.archive_tiers.source_write import record_raw_container_coordinate

    record_raw_container_coordinate(
        conn,
        raw_id,
        coordinate_format="zip-v2",
        entry_ordinal=entry_ordinal,
        split_index=0,
        addressing_mode=MemberAddressingMode.WHOLE_MEMBER,
        captured_coordinate=CapturedZipMemberCoordinate(
            str(bundle.resolve()),
            str(bundle.resolve()),
            member,
            entry_ordinal,
            0,
            MemberAddressingMode.WHOLE_MEMBER,
            hashlib.sha256(bundle.read_bytes()).hexdigest(),
            zip_acquisition_fingerprint(Provider.CLAUDE_CODE),
        ),
        manage_transaction=False,
    )


def _seed(root: Path) -> tuple[Path, Path]:
    """One materialized session, one declared non-session sidecar; both sources on disk."""
    initialize_active_archive_root(root)
    store = BlobStore(root / "blob")
    session_source = _write_source(root, "session.jsonl", b"session payload")
    sidecar_source = _write_source(root, "subagents/agent-1.meta.json", b"{}")
    session_hash = store.write_from_bytes(b"session payload")[0]
    sidecar_hash = store.write_from_bytes(b"{}")[0]
    source_conn = sqlite3.connect(root / "source.db")
    try:
        _insert_raw(
            source_conn,
            raw_id="raw-session",
            origin="claude-code-session",
            native_id="session",
            source_path=session_source,
            blob_hash=session_hash,
            parsed=True,
        )
        _insert_artifact(
            source_conn,
            raw_id="raw-session",
            origin="claude-code-session",
            source_path=session_source,
            kind="session_record_stream",
            support="supported_parseable",
            parse_as_session=True,
        )
        _insert_raw(
            source_conn,
            raw_id="raw-sidecar",
            origin="claude-code-session",
            native_id=None,
            source_path=sidecar_source,
            blob_hash=sidecar_hash,
            parsed=False,
        )
        _insert_artifact(
            source_conn,
            raw_id="raw-sidecar",
            origin="claude-code-session",
            source_path=sidecar_source,
            kind="agent_sidecar_meta",
            support="recognized_unparsed",
            parse_as_session=False,
        )
        source_conn.commit()
    finally:
        source_conn.close()
    index_conn = sqlite3.connect(root / "index.db")
    try:
        _insert_session(index_conn, origin="claude-code-session", native_id="session", raw_id="raw-session")
        index_conn.commit()
    finally:
        index_conn.close()
    return session_source, sidecar_source


def _run(root: Path) -> ArchiveVerificationCheck:
    return _check(verify_archive(root, checks=(CHECK,)))


def _run_with_frontier(root: Path, frontier: SourceFrontier) -> ArchiveVerificationCheck:
    return _check(
        verify_archive(
            root,
            checks=(CHECK,),
            source_frontier=frontier,
            require_source_frontier=True,
        )
    )


def test_source_conservation_is_declared_for_the_live_route() -> None:
    assert CHECK in archive_verification_names_for_route("live-archive")


def test_coherent_archive_types_every_item_and_is_green(tmp_path: Path) -> None:
    _seed(tmp_path)
    check = _run(tmp_path)
    assert check.status is OutcomeStatus.OK, check.summary
    assert _count(check, "materialized") == 1
    assert _count(check, "non_session_artifact") == 1
    assert _terms(check)["non_session_artifact"]["breakdown"] == {"claude-code-session:agent_sidecar_meta": 1}
    assert check.evidence["forward_total"] == 2
    assert check.evidence["blocking_count"] == 0
    assert all(term["rule"] for term in _terms(check).values())


def test_configured_frontier_binds_exact_totals_and_digest(tmp_path: Path) -> None:
    _seed(tmp_path)
    frontier = build_source_frontier(
        [SourceDeclaration("configured", SourceRole.DIRECTORY, tmp_path / "sources", True)]
    )
    check = _run_with_frontier(tmp_path, frontier)
    assert check.status is OutcomeStatus.OK, check.summary
    assert check.evidence["frontier_sha256"] == frontier.frontier_sha256
    assert check.evidence["frontier_total"] == frontier.item_count == 2
    assert check.evidence["frontier_bytes"] == frontier.byte_count
    assert check.evidence["frontier_complete"] is True
    assert check.evidence["frontier_root_states"] == {"configured": "present"}


def test_configured_but_unacquired_member_blocks_even_with_other_rows(tmp_path: Path) -> None:
    _seed(tmp_path)
    (_write_source(tmp_path, "unacquired.json", b"not admitted"))
    frontier = build_source_frontier(
        [SourceDeclaration("configured", SourceRole.DIRECTORY, tmp_path / "sources", True)]
    )
    check = _run_with_frontier(tmp_path, frontier)
    assert check.status is OutcomeStatus.ERROR, check.summary
    assert _count(check, "frontier_unacquired") == 1


def test_archive_member_frontier_matches_the_production_zip_address(tmp_path: Path) -> None:
    """A production ``archive:member`` address must not become an orphan.

    Anti-vacuity: adding a second separator to the already archive-prefixed
    frontier coordinate produces ``archive:archive!member``. Falling back to
    the archive root then loses member-level ownership and cannot distinguish
    equal-byte sibling members.
    """
    initialize_active_archive_root(tmp_path)
    archive = tmp_path / "export.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("sessions/one.json", "session payload")
    blob_hash = BlobStore(tmp_path / "blob").write_from_bytes(b"session payload")[0]
    source_path = Path(f"{archive}:sessions/one.json")
    source_conn = sqlite3.connect(tmp_path / "source.db")
    try:
        _insert_raw(
            source_conn,
            raw_id="raw-archive-member",
            origin="codex-session",
            native_id="one",
            source_path=source_path,
            blob_hash=blob_hash,
            parsed=True,
        )
        _insert_artifact(
            source_conn,
            raw_id="raw-archive-member",
            origin="codex-session",
            source_path=source_path,
            kind="session_record_stream",
            support="supported_parseable",
            parse_as_session=True,
        )
        source_conn.commit()
    finally:
        source_conn.close()
    index_conn = sqlite3.connect(tmp_path / "index.db")
    try:
        _insert_session(index_conn, origin="codex-session", native_id="one", raw_id="raw-archive-member")
        index_conn.commit()
    finally:
        index_conn.close()

    frontier = build_source_frontier([SourceDeclaration("export", SourceRole.ARCHIVE_MEMBER, archive)])
    check = _run_with_frontier(tmp_path, frontier)

    assert check.status is OutcomeStatus.OK, check.summary
    assert _count(check, "frontier_unacquired") == 0
    assert _count(check, "frontier_orphan") == 0


def test_frontier_integrity_failure_is_a_typed_check_error(tmp_path: Path) -> None:
    _seed(tmp_path)
    frontier = build_source_frontier(
        [SourceDeclaration("configured", SourceRole.DIRECTORY, tmp_path / "sources", True)]
    )
    check = _run_with_frontier(tmp_path, replace(frontier, frontier_sha256="0" * 64))
    assert check.status is OutcomeStatus.ERROR
    assert check.evidence["error"] == "source frontier integrity check failed"


def test_suppressed_index_content_is_not_conserved_by_identity(tmp_path: Path) -> None:
    _seed(tmp_path)
    frontier = build_source_frontier(
        [SourceDeclaration("configured", SourceRole.DIRECTORY, tmp_path / "sources", True)]
    )
    source_conn = sqlite3.connect(tmp_path / "source.db")
    try:
        source_conn.execute(
            """
            INSERT INTO raw_session_memberships(
                raw_id, logical_source_key, provider_session_id, source_revision,
                normalized_content_hash, message_count, revision_authority
            ) VALUES ('raw-session', 'codex:session', 'session', 'r1', ?, 1, 'byte_proven')
            """,
            (b"s" * 32,),
        )
        source_conn.commit()
    finally:
        source_conn.close()
    index_conn = sqlite3.connect(tmp_path / "index.db")
    try:
        index_conn.execute("UPDATE sessions SET content_hash = ? WHERE raw_id = 'raw-session'", (b"x" * 32,))
        index_conn.commit()
    finally:
        index_conn.close()
    check = _run_with_frontier(tmp_path, frontier)
    assert check.status is OutcomeStatus.ERROR, check.summary
    assert _count(check, "content_mismatch") == 1


def test_deleted_source_file_retypes_the_raw_as_source_missing(tmp_path: Path) -> None:
    """Anti-vacuity: without the on-disk probe the raw stays typed ``materialized``.

    The source file is gone but its raw payload bytes are still retained, so
    the content is conserved: the term is accounting, not a blocker.
    """
    session_source, _ = _seed(tmp_path)
    assert _count(_run(tmp_path), "source_missing") == 0
    session_source.unlink()
    check = _run(tmp_path)
    assert check.status is OutcomeStatus.OK, check.summary
    assert _count(check, "source_missing") == 1
    assert _terms(check)["source_missing"]["sample"] == ["raw-session"]
    assert _terms(check)["source_missing"]["blocking"] is False
    assert _count(check, "materialized") == 0
    assert _count(check, "source_lost") == 0


def test_deleted_source_file_without_retained_bytes_trips_source_conservation(tmp_path: Path) -> None:
    """Anti-vacuity: without the retained-bytes join this reads as the non-blocking term.

    Both the acquired file and the raw payload blob ref are gone, so nothing
    in the archive holds the bytes any more.
    """
    session_source, _ = _seed(tmp_path)
    session_source.unlink()
    source_conn = sqlite3.connect(tmp_path / "source.db")
    try:
        source_conn.execute("DELETE FROM blob_refs WHERE ref_id = 'raw-session'")
        source_conn.commit()
    finally:
        source_conn.close()
    check = _run(tmp_path)
    assert check.status is OutcomeStatus.ERROR
    assert _count(check, "source_lost") == 1
    assert _terms(check)["source_lost"]["sample"] == ["raw-session"]
    assert _count(check, "source_missing") == 0
    assert "source_lost:raw-session" in check.details


def test_injected_unadmitted_session_trips_source_conservation(tmp_path: Path) -> None:
    """Anti-vacuity: without the reverse join an index row with no raw is invisible."""
    _seed(tmp_path)
    index_conn = sqlite3.connect(tmp_path / "index.db")
    try:
        _insert_session(index_conn, origin="codex-session", native_id="ghost", raw_id="raw-never-acquired")
        _insert_session(index_conn, origin="codex-session", native_id="rawless", raw_id=None)
        index_conn.commit()
    finally:
        index_conn.close()
    check = _run(tmp_path)
    assert check.status is OutcomeStatus.ERROR
    assert _count(check, "session_orphan") == 1
    assert _terms(check)["session_orphan"]["sample"] == ["codex-session:ghost"]
    assert _count(check, "session_without_raw") == 1
    assert check.count == 2


def test_session_from_declared_non_session_artifact_is_a_phantom(tmp_path: Path) -> None:
    """polylogue-b508: lineage, not filename, makes the phantom; the row is reported, never deleted."""
    _seed(tmp_path)
    index_conn = sqlite3.connect(tmp_path / "index.db")
    try:
        phantom = _insert_session(index_conn, origin="claude-code-session", native_id="agent-1", raw_id="raw-sidecar")
        index_conn.commit()
    finally:
        index_conn.close()
    check = _run(tmp_path)
    assert check.status is OutcomeStatus.ERROR
    assert _count(check, "phantom_declared_non_session_lineage") == 1
    assert _terms(check)["phantom_declared_non_session_lineage"]["breakdown"] == {"artifact:agent_sidecar_meta": 1}
    assert _terms(check)["phantom_declared_non_session_lineage"]["sample"] == [phantom]
    # The sidecar raw is now materialized (by the phantom) and no longer a non-session exclusion.
    assert _count(check, "non_session_artifact") == 0
    index_conn = sqlite3.connect(tmp_path / "index.db")
    try:
        assert index_conn.execute("SELECT COUNT(*) FROM sessions WHERE session_id = ?", (phantom,)).fetchone()[0] == 1
    finally:
        index_conn.close()


def test_session_with_declared_rule_path_is_a_phantom_without_artifact_row(tmp_path: Path) -> None:
    """The origin's artifact rules classify lineage when raw_artifacts holds no row."""
    _seed(tmp_path)
    journal = _write_source(tmp_path, "subagents/workflows/run-1/journal.jsonl", b"{}")
    blob_hash = BlobStore(tmp_path / "blob").write_from_bytes(b"journal")[0]
    source_conn = sqlite3.connect(tmp_path / "source.db")
    try:
        _insert_raw(
            source_conn,
            raw_id="raw-journal",
            origin="claude-code-session",
            native_id="run-1",
            source_path=journal,
            blob_hash=blob_hash,
            parsed=True,
        )
        source_conn.commit()
    finally:
        source_conn.close()
    index_conn = sqlite3.connect(tmp_path / "index.db")
    try:
        _insert_session(index_conn, origin="claude-code-session", native_id="run-1", raw_id="raw-journal")
        index_conn.commit()
    finally:
        index_conn.close()
    check = _run(tmp_path)
    assert check.status is OutcomeStatus.ERROR
    assert _terms(check)["phantom_declared_non_session_lineage"]["breakdown"] == {"rule:workflow_journal": 1}


def test_fragment_shaped_identity_is_a_phantom(tmp_path: Path) -> None:
    _seed(tmp_path)
    fragment = _write_source(tmp_path, "fragment.jsonl", b"fragment")
    blob_hash = BlobStore(tmp_path / "blob").write_from_bytes(b"fragment")[0]
    source_conn = sqlite3.connect(tmp_path / "source.db")
    try:
        _insert_raw(
            source_conn,
            raw_id="raw-fragment",
            origin="claude-code-session",
            native_id="toolu_01abc",
            source_path=fragment,
            blob_hash=blob_hash,
            parsed=True,
        )
        source_conn.commit()
    finally:
        source_conn.close()
    index_conn = sqlite3.connect(tmp_path / "index.db")
    try:
        _insert_session(index_conn, origin="claude-code-session", native_id="toolu_01abc", raw_id="raw-fragment")
        index_conn.commit()
    finally:
        index_conn.close()
    check = _run(tmp_path)
    assert check.status is OutcomeStatus.ERROR
    assert _terms(check)["phantom_fragment_identity"]["breakdown"] == {"prefix:toolu_": 1}


def test_fragment_identity_shapes_cover_each_declared_prefix_and_meta_suffix() -> None:
    for prefix in FRAGMENT_IDENTITY_PREFIXES:
        assert fragment_identity_shape(f"{prefix}x") == f"prefix:{prefix}"
    assert fragment_identity_shape("agent-af4e.meta") == "suffix:.meta:agent_sidecar_meta"
    assert fragment_identity_shape("5ecdb160-agent-af4e") is None


def test_work_event_on_an_indexed_session_is_materialized(tmp_path: Path) -> None:
    """A retained agent work event is conserved by the session it annotates.

    Anti-vacuity: the work event is its own logical source and no session
    names its raw, so counting only ``sessions.raw_id`` types it
    ``unexplained``, and probing its archive-authored source path types it
    ``source_missing``.
    """
    _seed(tmp_path)
    blob_hash = BlobStore(tmp_path / "blob").write_from_bytes(b"work event")[0]
    raw_id = "agent-work-event:" + "e" * 64
    source_conn = sqlite3.connect(tmp_path / "source.db")
    try:
        _insert_raw(
            source_conn,
            raw_id=raw_id,
            origin="claude-code-session",
            native_id="session",
            source_path=Path(raw_id),
            blob_hash=blob_hash,
            parsed=True,
        )
        # Admission makes each work event its own logical source.
        source_conn.execute(
            "UPDATE raw_sessions SET logical_source_key = raw_id, revision_kind = 'full' WHERE raw_id = ?", (raw_id,)
        )
        source_conn.commit()
    finally:
        source_conn.close()
    check = _run(tmp_path)
    assert check.status is OutcomeStatus.OK, check.summary
    assert _count(check, "materialized") == 2
    sample = _terms(check)["materialized"]["sample"]
    assert isinstance(sample, list)
    assert raw_id in sample


def test_parsed_raw_without_session_or_rule_is_unexplained(tmp_path: Path) -> None:
    _seed(tmp_path)
    stray = _write_source(tmp_path, "stray.json", b"{}")
    blob_hash = BlobStore(tmp_path / "blob").write_from_bytes(b"stray")[0]
    source_conn = sqlite3.connect(tmp_path / "source.db")
    try:
        _insert_raw(
            source_conn,
            raw_id="raw-stray",
            origin="aistudio-drive",
            native_id=None,
            source_path=stray,
            blob_hash=blob_hash,
            parsed=True,
        )
        source_conn.commit()
    finally:
        source_conn.close()
    check = _run(tmp_path)
    assert check.status is OutcomeStatus.ERROR
    assert _count(check, "unexplained") == 1
    assert _terms(check)["unexplained"]["sample"] == ["raw-stray"]


def test_unparsed_raw_is_pending_and_only_a_warning(tmp_path: Path) -> None:
    _seed(tmp_path)
    fresh = _write_source(tmp_path, "fresh.jsonl", b"fresh")
    blob_hash = BlobStore(tmp_path / "blob").write_from_bytes(b"fresh")[0]
    source_conn = sqlite3.connect(tmp_path / "source.db")
    try:
        _insert_raw(
            source_conn,
            raw_id="raw-fresh",
            origin="codex-session",
            native_id="fresh",
            source_path=fresh,
            blob_hash=blob_hash,
            parsed=False,
        )
        source_conn.commit()
    finally:
        source_conn.close()
    check = _run(tmp_path)
    assert check.status is OutcomeStatus.WARNING
    assert _count(check, "pending") == 1
    assert check.evidence["blocking_count"] == 0


def test_parse_failure_is_a_typed_exclusion(tmp_path: Path) -> None:
    _seed(tmp_path)
    broken = _write_source(tmp_path, "broken.json", b"{")
    blob_hash = BlobStore(tmp_path / "blob").write_from_bytes(b"{")[0]
    source_conn = sqlite3.connect(tmp_path / "source.db")
    try:
        _insert_raw(
            source_conn,
            raw_id="raw-broken",
            origin="chatgpt-export",
            native_id="broken",
            source_path=broken,
            blob_hash=blob_hash,
            parsed=True,
        )
        source_conn.execute("UPDATE raw_sessions SET parse_error = 'transform: boom' WHERE raw_id = 'raw-broken'")
        source_conn.commit()
    finally:
        source_conn.close()
    check = _run(tmp_path)
    assert check.status is OutcomeStatus.OK
    assert _count(check, "parse_failure") == 1


def test_value_bound_refusal_is_a_counted_warning_not_a_parse_failure(tmp_path: Path) -> None:
    """A session refused for an oversized value stays a visible gap.

    Anti-vacuity: drop the ``value_bound_refused`` CASE arm and the raw types
    as an ordinary ``parse_failure`` with an OK check, hiding the gap.
    """
    _seed(tmp_path)
    giant = _write_source(tmp_path, "giant.json", b"{}")
    blob_hash = BlobStore(tmp_path / "blob").write_from_bytes(b"{}")[0]
    source_conn = sqlite3.connect(tmp_path / "source.db")
    try:
        _insert_raw(
            source_conn,
            raw_id="raw-giant",
            origin="chatgpt-export",
            native_id="giant",
            source_path=giant,
            blob_hash=blob_hash,
            parsed=True,
        )
        source_conn.execute(
            "UPDATE raw_sessions SET parse_error = ? WHERE raw_id = 'raw-giant'",
            (
                "RuntimeError: off-writer preparation failed: ValueBoundRefusedError: value_bound_refused: "
                "a decoded string of 70000000 characters exceeds the declared resident bound of 67108864",
            ),
        )
        source_conn.commit()
    finally:
        source_conn.close()
    check = _run(tmp_path)
    assert check.status is OutcomeStatus.WARNING
    assert _count(check, "value_bound_refused") == 1
    assert _count(check, "parse_failure") == 0
    assert check.evidence["blocking_count"] == 0


def test_value_bound_token_is_matched_literally(tmp_path: Path) -> None:
    """Only the exact refusal token types a raw as ``value_bound_refused``.

    Anti-vacuity: a ``LIKE`` pattern treats each ``_`` as a wildcard, so the
    source-controlled text ``value-bound-refused`` in an ordinary parser
    error would be reported as a value-bound refusal.
    """
    _seed(tmp_path)
    source = _write_source(tmp_path, "lookalike.json", b"{}")
    blob_hash = BlobStore(tmp_path / "blob").write_from_bytes(b"{}")[0]
    source_conn = sqlite3.connect(tmp_path / "source.db")
    try:
        _insert_raw(
            source_conn,
            raw_id="raw-lookalike",
            origin="chatgpt-export",
            native_id="lookalike",
            source_path=source,
            blob_hash=blob_hash,
            parsed=True,
        )
        source_conn.execute(
            "UPDATE raw_sessions SET parse_error = ? WHERE raw_id = 'raw-lookalike'",
            ("ValueError: unexpected field 'value-bound-refused' in node",),
        )
        source_conn.commit()
    finally:
        source_conn.close()
    check = _run(tmp_path)
    assert _count(check, "value_bound_refused") == 0
    assert _count(check, "parse_failure") == 1


def test_a_path_naming_the_refusal_token_is_an_ordinary_parse_failure(tmp_path: Path) -> None:
    """Only the typed error's serialized head types a raw as ``value_bound_refused``.

    Anti-vacuity (Codex P2, #5643): search the diagnostic for the bare token
    and an unsupported input at ``/imports/value_bound_refused.json`` is
    counted as a physical value-bound refusal.
    """
    _seed(tmp_path)
    source = _write_source(tmp_path, "value_bound_refused.json", b"{}")
    blob_hash = BlobStore(tmp_path / "blob").write_from_bytes(b"{}")[0]
    source_conn = sqlite3.connect(tmp_path / "source.db")
    try:
        _insert_raw(
            source_conn,
            raw_id="raw-named",
            origin="chatgpt-export",
            native_id="named",
            source_path=source,
            blob_hash=blob_hash,
            parsed=True,
        )
        source_conn.execute(
            "UPDATE raw_sessions SET parse_error = ? WHERE raw_id = 'raw-named'",
            ("parse: unsupported input /imports/value_bound_refused.json",),
        )
        source_conn.commit()
    finally:
        source_conn.close()
    check = _run(tmp_path)
    assert _count(check, "value_bound_refused") == 0
    assert _count(check, "parse_failure") == 1


def test_check_json_carries_every_term_with_its_rule(tmp_path: Path) -> None:
    _seed(tmp_path)
    payload = _run(tmp_path).to_json()
    evidence = payload["evidence"]
    assert isinstance(evidence, dict)
    terms = evidence["terms"]
    assert isinstance(terms, dict)
    assert {
        "materialized",
        "source_missing",
        "source_lost",
        "missing_blob",
        "unexplained",
        "phantom_declared_non_session_lineage",
    } <= set(terms)
    for term in terms.values():
        assert isinstance(term, dict)
        assert isinstance(term["rule"], str) and term["rule"]
        assert isinstance(term["blocking"], bool)


def _insert_attachment(
    conn: sqlite3.Connection, *, attachment_id: str, ref_count: int, owner_gap: str | None = None
) -> None:
    conn.execute(
        """
        INSERT INTO attachments(attachment_id, display_name, media_type, byte_count, acquisition_status, ref_count)
        VALUES (?, 'a.png', 'image/png', 0, 'unfetched', ?)
        """,
        (attachment_id, ref_count),
    )
    if owner_gap is not None:
        session_id = conn.execute("SELECT session_id FROM sessions LIMIT 1").fetchone()[0]
        conn.execute(
            "INSERT INTO attachment_owner_gaps(session_id, attachment_id, reason) VALUES (?, ?, ?)",
            (session_id, attachment_id, owner_gap),
        )


def test_attachment_whose_named_owner_is_missing_blocks(tmp_path: Path) -> None:
    """A gap the writer recorded as ``message_missing`` is a lost owner and blocks.

    Gaps recorded as ambiguous or never linked stay the explained term.
    Anti-vacuity: classifying every ref_count-0 row as ``attachment_unowned``
    verifies this archive green although an owner was lost.
    """
    _seed(tmp_path)
    index_conn = sqlite3.connect(tmp_path / "index.db")
    try:
        _insert_attachment(index_conn, attachment_id="lost-1", ref_count=0, owner_gap="message_missing")
        _insert_attachment(index_conn, attachment_id="ambiguous-1", ref_count=0, owner_gap="owner_ambiguous")
        _insert_attachment(index_conn, attachment_id="loose-1", ref_count=0, owner_gap="provider_never_linked")
        index_conn.commit()
    finally:
        index_conn.close()

    check = _run(tmp_path)
    assert check.status is OutcomeStatus.ERROR, check.summary
    assert _count(check, "attachment_owner_missing") == 1
    assert _terms(check)["attachment_owner_missing"]["blocking"] is True
    assert _terms(check)["attachment_owner_missing"]["sample"] == ["lost-1"]
    assert _count(check, "attachment_unowned") == 2
    assert _terms(check)["attachment_unowned"]["blocking"] is False
    assert check.evidence["blocking_count"] == 1


def test_owner_ambiguous_attachment_types_as_unowned_and_does_not_block(tmp_path: Path) -> None:
    """An attachment written unreferenced because its owner is ambiguous is explained.

    The writer inserts such a row with ``ref_count`` 0 and deliberately keeps
    it out of the ref-count sweep, so it never had a ref to lose.

    Anti-vacuity: without the ``ref_count`` split every ref-less attachment
    types as ``attachment_unreferenced``, which blocks, and the check goes red.
    """
    _seed(tmp_path)
    index_conn = sqlite3.connect(tmp_path / "index.db")
    try:
        _insert_attachment(index_conn, attachment_id="unowned-1", ref_count=0)
        index_conn.commit()
    finally:
        index_conn.close()

    check = _run(tmp_path)
    assert check.status is OutcomeStatus.OK, check.summary
    assert _count(check, "attachment_unowned") == 1
    assert _terms(check)["attachment_unowned"]["blocking"] is False
    assert _terms(check)["attachment_unowned"]["sample"] == ["unowned-1"]
    assert _count(check, "attachment_unreferenced") == 0
    assert check.evidence["blocking_count"] == 0
    assert "attachment_unreferenced" not in check.summary


def test_refless_attachment_with_stale_ref_count_still_blocks(tmp_path: Path) -> None:
    """A row whose refs went away without the sweep is unreachable and blocks.

    Its non-zero ``ref_count`` is the evidence that refs once existed: the
    sweep would have recomputed it to 0 and deleted the row.

    Anti-vacuity: if the split classified every ref-less attachment as the
    explained ``attachment_unowned`` term, this archive would verify green and
    the genuine orphan would go unreported.
    """
    _seed(tmp_path)
    index_conn = sqlite3.connect(tmp_path / "index.db")
    try:
        _insert_attachment(index_conn, attachment_id="orphan-1", ref_count=2)
        index_conn.commit()
    finally:
        index_conn.close()

    check = _run(tmp_path)
    assert check.status is OutcomeStatus.ERROR, check.summary
    assert _count(check, "attachment_unreferenced") == 1
    assert _terms(check)["attachment_unreferenced"]["blocking"] is True
    assert _terms(check)["attachment_unreferenced"]["sample"] == ["orphan-1"]
    assert _count(check, "attachment_unowned") == 0
    assert check.evidence["blocking_count"] == 1


def test_candidate_route_conserves_against_the_candidate_index(tmp_path: Path) -> None:
    """Anti-vacuity: bound to the active index instead, the clean live rows read green.

    The unadmitted session exists only in the candidate copy, so a check that
    ignores ``index_path_override`` reports OK while the candidate is unsound.
    """
    _seed(tmp_path)
    candidate = tmp_path / "candidate-index.db"
    shutil.copy2(tmp_path / "index.db", candidate)
    candidate_conn = sqlite3.connect(candidate)
    try:
        _insert_session(candidate_conn, origin="codex-session", native_id="ghost", raw_id="raw-never-acquired")
        candidate_conn.commit()
    finally:
        candidate_conn.close()

    assert CHECK in archive_verification_names_for_route("reindex-cross-tier-candidate")
    assert _run(tmp_path).status is OutcomeStatus.OK

    check = _check(verify_archive(tmp_path, checks=(CHECK,), index_path_override=candidate))

    assert check.status is OutcomeStatus.ERROR
    assert _count(check, "session_orphan") == 1
    assert _terms(check)["session_orphan"]["sample"] == ["codex-session:ghost"]


def _insert_membership(
    conn: sqlite3.Connection,
    *,
    raw_id: str,
    logical_source_key: str,
    authority: str = "quarantined",
    normalized_content_hash: bytes = b"n" * 32,
) -> None:
    conn.execute(
        """
        INSERT INTO raw_session_memberships(
            raw_id, logical_source_key, provider_session_id, source_revision,
            normalized_content_hash, message_count, revision_authority
        ) VALUES (?, ?, ?, ?, ?, 1, ?)
        """,
        (
            raw_id,
            logical_source_key,
            logical_source_key.partition(":")[2],
            raw_id,
            normalized_content_hash,
            authority,
        ),
    )


def _insert_authority_blocker(
    conn: sqlite3.Connection, *, blocker_id: str, head_raw_id: str, reason: str, resolved: bool = False
) -> None:
    import json as _json

    expected = _json.dumps({"index_preconditions": {"head_accepted_raw_id": head_raw_id}})
    conn.execute(
        """
        INSERT INTO raw_authority_blockers(
            blocker_id, plan_input_digest, observed_pass_id, reason, expected_json, observed_json,
            created_at_ms, resolved_at_ms, resolution
        ) VALUES (?, ?, 'census-1', ?, ?, '{}', 100, ?, ?)
        """,
        (
            blocker_id,
            hashlib.sha256(blocker_id.encode("utf-8")).hexdigest(),
            reason,
            expected,
            200 if resolved else None,
            "done" if resolved else None,
        ),
    )


def _insert_unmaterialized_raw(root: Path, *, raw_id: str, name: str, origin: str = "codex-session") -> None:
    """A parsed raw with real bytes on disk that no index session names."""
    source = _write_source(root, name, name.encode())
    blob_hash = BlobStore(root / "blob").write_from_bytes(name.encode())[0]
    conn = sqlite3.connect(root / "source.db")
    try:
        _insert_raw(
            conn,
            raw_id=raw_id,
            origin=origin,
            native_id=None,
            source_path=source,
            blob_hash=blob_hash,
            parsed=True,
        )
        conn.commit()
    finally:
        conn.close()


def test_unresolved_authority_blocker_types_the_head_raw_and_only_warns(tmp_path: Path) -> None:
    """A raw the authority frontier accepted as head, while the index took another, is explained.

    ``raw_authority_blockers`` is the durable ledger that owns the remedy; the
    conservation check cites it instead of calling the raw unexplained.

    Anti-vacuity: the same raw with the blocker resolved has no rule left and
    types ``unexplained``, so the term cannot absorb an unexplained raw.
    """
    _seed(tmp_path)
    reason = "accepted revision head and materialized session select different raw authority"
    _insert_unmaterialized_raw(tmp_path, raw_id="raw-head", name="head.json", origin="aistudio-drive")
    source_conn = sqlite3.connect(tmp_path / "source.db")
    try:
        _insert_authority_blocker(source_conn, blocker_id="blk-1", head_raw_id="raw-head", reason=reason)
        source_conn.commit()
    finally:
        source_conn.close()

    check = _run(tmp_path)
    assert check.status is OutcomeStatus.WARNING, check.summary
    assert _count(check, "authority_blocked_head") == 1
    assert _terms(check)["authority_blocked_head"]["blocking"] is False
    assert _terms(check)["authority_blocked_head"]["breakdown"] == {f"aistudio-drive:{reason}": 1}
    assert _count(check, "unexplained") == 0
    assert check.evidence["blocking_count"] == 0

    source_conn = sqlite3.connect(tmp_path / "source.db")
    try:
        source_conn.execute(
            "UPDATE raw_authority_blockers SET resolved_at_ms = 200, resolution = 'done' WHERE blocker_id = 'blk-1'"
        )
        source_conn.commit()
    finally:
        source_conn.close()
    resolved = _run(tmp_path)
    assert resolved.status is OutcomeStatus.ERROR, resolved.summary
    assert _count(resolved, "authority_blocked_head") == 0
    assert _count(resolved, "unexplained") == 1


def test_wholly_quarantined_membership_cohort_without_a_session_blocks(tmp_path: Path) -> None:
    """A logical source whose every membership is quarantined, with nothing indexed, is missing.

    The term names the defect instead of leaving it in ``unexplained``, and it
    keeps blocking because the session's content is absent from the index.

    Anti-vacuity: one ``byte_proven`` membership on the same raw removes the
    rule and the raw types ``unexplained``, so the term cannot absorb every
    unmaterialized raw that happens to carry a membership.
    """
    _seed(tmp_path)
    _insert_unmaterialized_raw(tmp_path, raw_id="raw-quar", name="quar.jsonl")
    source_conn = sqlite3.connect(tmp_path / "source.db")
    try:
        _insert_membership(source_conn, raw_id="raw-quar", logical_source_key="codex:lost")
        source_conn.commit()
    finally:
        source_conn.close()

    check = _run(tmp_path)
    assert check.status is OutcomeStatus.ERROR, check.summary
    assert _count(check, "quarantined_cohort_unmaterialized") == 1
    assert _terms(check)["quarantined_cohort_unmaterialized"]["blocking"] is True
    assert _terms(check)["quarantined_cohort_unmaterialized"]["sample"] == ["raw-quar"]
    assert _count(check, "unexplained") == 0

    source_conn = sqlite3.connect(tmp_path / "source.db")
    try:
        source_conn.execute(
            "UPDATE raw_session_memberships SET revision_authority = 'byte_proven' WHERE raw_id = 'raw-quar'"
        )
        source_conn.commit()
    finally:
        source_conn.close()
    proven = _run(tmp_path)
    assert _count(proven, "quarantined_cohort_unmaterialized") == 0
    assert _count(proven, "unexplained") == 1


def test_raw_sharing_a_logical_source_with_a_materialized_raw_is_superseded(tmp_path: Path) -> None:
    """A raw is superseded when any raw it shares a logical source with is materialized.

    ``raw_session_memberships`` states which logical sources a raw belongs to.
    A shared raw carries several, so membership sets overlap without being
    equal and the cohort partition alone splits revisions of one logical source
    apart by native id or path.

    Anti-vacuity: giving the two raws disjoint logical sources removes the
    relation and the unmaterialized raw stops being superseded, so the rule
    cannot be a blanket pass for every membership-bearing raw.
    """
    _seed(tmp_path)
    _insert_unmaterialized_raw(tmp_path, raw_id="raw-shared-a", name="shared-a.jsonl")
    _insert_unmaterialized_raw(tmp_path, raw_id="raw-shared-b", name="shared-b.jsonl")
    source_conn = sqlite3.connect(tmp_path / "source.db")
    try:
        _insert_membership(source_conn, raw_id="raw-shared-a", logical_source_key="codex:first")
        _insert_membership(source_conn, raw_id="raw-shared-a", logical_source_key="codex:only-on-a")
        _insert_membership(source_conn, raw_id="raw-shared-b", logical_source_key="codex:first")
        _insert_membership(source_conn, raw_id="raw-shared-b", logical_source_key="codex:only-on-b")
        source_conn.commit()
    finally:
        source_conn.close()
    index_conn = sqlite3.connect(tmp_path / "index.db")
    try:
        _insert_session(index_conn, origin="codex-session", native_id="first", raw_id="raw-shared-a")
        index_conn.commit()
    finally:
        index_conn.close()

    check = _run(tmp_path)
    assert check.status is OutcomeStatus.OK, check.summary
    assert _count(check, "materialized") == 2
    assert _count(check, "revision_superseded") == 1
    assert _terms(check)["revision_superseded"]["sample"] == ["raw-shared-b"]
    assert _count(check, "quarantined_cohort_unmaterialized") == 0
    assert _count(check, "unexplained") == 0

    source_conn = sqlite3.connect(tmp_path / "source.db")
    try:
        source_conn.execute(
            "UPDATE raw_session_memberships SET logical_source_key = 'codex:disjoint' "
            "WHERE raw_id = 'raw-shared-b' AND logical_source_key = 'codex:first'"
        )
        source_conn.commit()
    finally:
        source_conn.close()
    split = _run(tmp_path)
    assert _count(split, "revision_superseded") == 0
    assert _count(split, "quarantined_cohort_unmaterialized") == 1


def _configured_frontier(root: Path) -> SourceFrontier:
    return build_source_frontier([SourceDeclaration("configured", SourceRole.DIRECTORY, root / "sources", True)])


def test_historical_revision_is_not_an_orphan(tmp_path: Path) -> None:
    """An earlier acquisition of a still-configured source is not unowned.

    A configured file ingested at hash A, changed, and ingested again at hash
    B leaves two raws at one canonical path. Only B owns the current frontier
    member, but A's *source identity* is squarely inside the configured
    frontier -- it is a revision of a configured source, which the forward
    classifier already types. ``frontier_orphan`` means "outside the
    configured frontier" and is blocking, so counting A there failed
    ``verify-archive`` for ordinary revision history.

    Anti-vacuity: dropping ``configured_paths`` from the orphan guard, or
    pointing the second raw at an unconfigured path, makes A (or the new raw)
    a blocking orphan again -- the guard cannot be a blanket pass, which the
    second half of this test pins directly.
    """
    session_source, _sidecar = _seed(tmp_path)
    store = BlobStore(tmp_path / "blob")
    session_source.write_bytes(b"session payload v2")
    revised_hash = store.write_from_bytes(b"session payload v2")[0]
    source_conn = sqlite3.connect(tmp_path / "source.db")
    try:
        _insert_raw(
            source_conn,
            raw_id="raw-session-v2",
            origin="claude-code-session",
            native_id="session",
            source_path=session_source,
            blob_hash=revised_hash,
            parsed=False,
        )
        source_conn.commit()
    finally:
        source_conn.close()

    check = _run_with_frontier(tmp_path, _configured_frontier(tmp_path))
    assert _count(check, "frontier_orphan") == 0, check.evidence["terms"]
    assert _count(check, "frontier_unacquired") == 0
    assert check.status is OutcomeStatus.OK, check.summary

    # The opposite direction: a raw at a path the frontier never names is
    # still a blocking orphan.
    outside = tmp_path / "outside" / "stray.jsonl"
    outside.parent.mkdir(parents=True, exist_ok=True)
    outside.write_bytes(b"stray")
    stray_hash = store.write_from_bytes(b"stray")[0]
    source_conn = sqlite3.connect(tmp_path / "source.db")
    try:
        _insert_raw(
            source_conn,
            raw_id="raw-stray",
            origin="claude-code-session",
            native_id=None,
            source_path=outside,
            blob_hash=stray_hash,
            parsed=False,
        )
        source_conn.commit()
    finally:
        source_conn.close()
    stray_check = _run_with_frontier(tmp_path, _configured_frontier(tmp_path))
    assert _count(stray_check, "frontier_orphan") == 1
    assert stray_check.status is OutcomeStatus.ERROR


def test_frontier_orphan_does_not_replace_session_orphan_count(tmp_path: Path) -> None:
    """A raw outside configured roots is not an index session with no raw."""
    _seed(tmp_path)
    outside = tmp_path / "outside" / "session.jsonl"
    outside.parent.mkdir(parents=True)
    outside.write_bytes(b"session payload")
    source_conn = sqlite3.connect(tmp_path / "source.db")
    try:
        source_conn.execute("UPDATE raw_sessions SET source_path = ? WHERE raw_id = 'raw-session'", (str(outside),))
        source_conn.execute("UPDATE blob_refs SET source_path = ? WHERE ref_id = 'raw-session'", (str(outside),))
        source_conn.commit()
    finally:
        source_conn.close()

    check = _run_with_frontier(tmp_path, _configured_frontier(tmp_path))

    assert _count(check, "session_orphan") == 0
    assert _count(check, "frontier_orphan") == 1


def test_multi_session_raw_is_not_a_mismatch(tmp_path: Path) -> None:
    """A grouped raw that emits two sessions conserves both memberships.

    Joining ``raw_session_memberships`` to ``sessions`` on ``raw_id`` alone
    pairs each membership with every sibling session of that raw, and the
    off-diagonal pairs disagree by construction, so a correct two-session
    acquisition reported a blocking ``content_mismatch``.

    Anti-vacuity: the second half changes one membership's normalized content
    to a hash no session of that raw carries, and it must still be reported --
    the fix cannot be a blanket pass for every multi-session raw.
    """
    _seed(tmp_path)
    grouped = _write_source(tmp_path, "group.jsonl", b"grouped payload")
    grouped_hash = BlobStore(tmp_path / "blob").write_from_bytes(b"grouped payload")[0]
    source_conn = sqlite3.connect(tmp_path / "source.db")
    try:
        _insert_raw(
            source_conn,
            raw_id="raw-group",
            origin="codex-session",
            native_id=None,
            source_path=grouped,
            blob_hash=grouped_hash,
            parsed=True,
        )
        _insert_membership(
            source_conn,
            raw_id="raw-group",
            logical_source_key="codex:g1",
            authority="byte_proven",
            normalized_content_hash=b"a" * 32,
        )
        _insert_membership(
            source_conn,
            raw_id="raw-group",
            logical_source_key="codex:g2",
            authority="byte_proven",
            normalized_content_hash=b"b" * 32,
        )
        source_conn.commit()
    finally:
        source_conn.close()
    index_conn = sqlite3.connect(tmp_path / "index.db")
    try:
        _insert_session(index_conn, origin="codex-session", native_id="g1", raw_id="raw-group", content_hash=b"a" * 32)
        _insert_session(index_conn, origin="codex-session", native_id="g2", raw_id="raw-group", content_hash=b"b" * 32)
        index_conn.commit()
    finally:
        index_conn.close()

    check = _run_with_frontier(tmp_path, _configured_frontier(tmp_path))
    assert _count(check, "content_mismatch") == 0, check.evidence["terms"]
    assert check.status is OutcomeStatus.OK, check.summary

    source_conn = sqlite3.connect(tmp_path / "source.db")
    try:
        source_conn.execute(
            "UPDATE raw_session_memberships SET normalized_content_hash = ? "
            "WHERE raw_id = 'raw-group' AND logical_source_key = 'codex:g2'",
            (b"z" * 32,),
        )
        source_conn.commit()
    finally:
        source_conn.close()
    drifted = _run_with_frontier(tmp_path, _configured_frontier(tmp_path))
    assert _count(drifted, "content_mismatch") == 1
    assert drifted.status is OutcomeStatus.ERROR


def test_stale_sqlite_revision_is_not_conserved(tmp_path: Path) -> None:
    """A mutable database whose current logical revision was never archived fails.

    ``observe_source_members`` records a ``MUTABLE_SQLITE`` member's
    ``content_sha256`` as the digest of its canonical logical export, and
    acquisition retains exactly those export bytes, so the archived
    ``blob_hash`` is that same logical-revision witness. Binding the member to
    any raw at its path instead made conservation report success for a
    database that had changed since its last acquisition.

    Anti-vacuity: archiving the CURRENT export digest makes the same member
    conserved, so the check is not a blanket refusal of mutable members.
    """
    _seed(tmp_path)
    database = tmp_path / "db" / "state.sqlite"
    database.parent.mkdir(parents=True, exist_ok=True)
    live = sqlite3.connect(database)
    try:
        live.execute("CREATE TABLE threads(id TEXT PRIMARY KEY, title TEXT)")
        live.execute("INSERT INTO threads VALUES ('t1', 'first')")
        live.commit()
    finally:
        live.close()
    acquired_revision = sqlite_member_revision(database)

    live = sqlite3.connect(database)
    try:
        live.execute("UPDATE threads SET title = 'second' WHERE id = 't1'")
        live.commit()
    finally:
        live.close()
    assert sqlite_member_revision(database) != acquired_revision

    source_conn = sqlite3.connect(tmp_path / "source.db")
    try:
        _insert_raw(
            source_conn,
            raw_id="raw-db",
            origin="codex-session",
            native_id=None,
            source_path=database,
            blob_hash=acquired_revision,
            parsed=False,
        )
        source_conn.commit()
    finally:
        source_conn.close()

    declarations = [
        SourceDeclaration("configured", SourceRole.DIRECTORY, tmp_path / "sources", True),
        SourceDeclaration("live-db", SourceRole.MUTABLE_SQLITE, database, True),
    ]
    stale = _run_with_frontier(tmp_path, build_source_frontier(declarations))
    assert _count(stale, "frontier_unacquired") == 1, stale.evidence["terms"]
    assert stale.status is OutcomeStatus.ERROR

    source_conn = sqlite3.connect(tmp_path / "source.db")
    try:
        source_conn.execute(
            "UPDATE raw_sessions SET blob_hash = ? WHERE raw_id = 'raw-db'",
            (bytes.fromhex(sqlite_member_revision(database)),),
        )
        source_conn.execute(
            "UPDATE blob_refs SET blob_hash = ? WHERE ref_id = 'raw-db'",
            (bytes.fromhex(sqlite_member_revision(database)),),
        )
        source_conn.commit()
    finally:
        source_conn.close()
    current = _run_with_frontier(tmp_path, build_source_frontier(declarations))
    assert _count(current, "frontier_unacquired") == 0, current.evidence["terms"]


def test_archive_member_presence_does_not_replace_retained_cas(tmp_path: Path) -> None:
    """A declared ZIP member proves source presence, not retained archive bytes.

    Mutation: source presence bypasses a missing capture and falsely materializes
    this raw after its blob reference is removed.
    """
    session_source, _ = _seed(tmp_path)
    payload = session_source.read_bytes()
    bundle = tmp_path / "export-bundle.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr("conversations.json", payload)
    coordinate = f"{bundle}:conversations.json"

    source_conn = sqlite3.connect(tmp_path / "source.db")
    try:
        source_conn.execute("UPDATE raw_sessions SET source_path = ? WHERE raw_id = 'raw-session'", (coordinate,))
        _record_member_fixture(source_conn, bundle)
        source_conn.execute("UPDATE raw_artifacts SET source_path = ? WHERE raw_id = 'raw-session'", (coordinate,))
        source_conn.execute("DELETE FROM blob_refs WHERE ref_id = 'raw-session'")
        source_conn.commit()
    finally:
        source_conn.close()
    session_source.unlink()

    check = _run(tmp_path)
    assert check.status is OutcomeStatus.ERROR, check.summary
    assert _count(check, "source_lost") == 0
    assert _count(check, "source_missing") == 0
    assert _count(check, "missing_blob") == 1
    assert _count(check, "materialized") == 0


def test_member_container_absent_is_lost(tmp_path: Path) -> None:
    """The member probe must not pass a coordinate whose container is gone.

    Anti-vacuity: infer a member from its operational suffix and
    returning True, and this stays green while a genuinely unrecoverable raw is
    reported conserved.
    """
    session_source, _ = _seed(tmp_path)
    coordinate = f"{tmp_path / 'never-acquired.zip'}!conversations.json"
    source_conn = sqlite3.connect(tmp_path / "source.db")
    try:
        source_conn.execute("UPDATE raw_sessions SET source_path = ? WHERE raw_id = 'raw-session'", (coordinate,))
        source_conn.execute("DELETE FROM blob_refs WHERE ref_id = 'raw-session'")
        source_conn.commit()
    finally:
        source_conn.close()
    session_source.unlink()

    check = _run(tmp_path)
    assert check.status is OutcomeStatus.ERROR, check.summary
    assert _count(check, "source_lost") == 1


def test_member_missing_from_container_is_lost(tmp_path: Path) -> None:
    """A present container that does not hold the member does not conserve it.

    Anti-vacuity: stop at container existence and this goes green, which is the
    exact false-conservation the member probe exists to prevent.
    """
    session_source, _ = _seed(tmp_path)
    bundle = tmp_path / "export-bundle.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr("conversations.json", session_source.read_bytes())
    coordinate = f"{bundle}:conversations.json"
    source_conn = sqlite3.connect(tmp_path / "source.db")
    try:
        source_conn.execute("UPDATE raw_sessions SET source_path = ? WHERE raw_id = 'raw-session'", (coordinate,))
        _record_member_fixture(source_conn, bundle)
        source_conn.execute("DELETE FROM blob_refs WHERE ref_id = 'raw-session'")
        source_conn.commit()
    finally:
        source_conn.close()
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr("something-else.json", b"{}")
    session_source.unlink()

    check = _run(tmp_path)
    assert check.status is OutcomeStatus.ERROR, check.summary
    assert _count(check, "source_lost") == 1


@pytest.mark.parametrize("retained", [False, True])
@pytest.mark.parametrize("suffix", [".zip", ".data"])
def test_non_zip_member_container_cannot_conserve_materialized_raw(tmp_path: Path, retained: bool, suffix: str) -> None:
    """Mutation: unreadable ZIP namelist returning True hides a lost materialized member."""
    session_source, _ = _seed(tmp_path)
    bundle = tmp_path / f"export{suffix}"
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr("conversations.json", '{"id":"session","mapping":{}}')
    with zipfile.ZipFile(bundle) as archive:
        context = ZipEntryReadContext(
            source=Source(name="chatgpt", path=tmp_path),
            zip_path=bundle,
            entry=archive.infolist()[0],
            file_mtime=None,
            provider_hint=Provider.CHATGPT,
            blob_store=BlobStore(tmp_path / "blob"),
        )
        acquired = list(iter_zip_entry_raw_data(archive, context))
    assert len(acquired) == 1
    coordinate = acquired[0].source_path
    with sqlite3.connect(tmp_path / "source.db") as conn:
        conn.execute("UPDATE raw_sessions SET source_path = ? WHERE raw_id = 'raw-session'", (coordinate,))
        _record_member_fixture(conn, bundle)
        if not retained:
            conn.execute("DELETE FROM blob_refs WHERE ref_id = 'raw-session'")
    bundle.write_bytes(b"demonstrably not a zip")
    session_source.unlink()
    check = _run(tmp_path)
    assert _count(check, "materialized") == 0
    assert _count(check, "source_lost") == int(not retained)
    assert _count(check, "source_missing") == int(retained)
    assert _count(check, "source_unavailable") == 0
    assert check.status is (OutcomeStatus.OK if retained else OutcomeStatus.ERROR)


@pytest.mark.parametrize("retained", [False, True])
@pytest.mark.parametrize("suffix", [".zip", ".data"])
def test_unreadable_member_inventory_is_retryable_and_not_loss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, suffix: str, retained: bool
) -> None:
    """Mutation: treating a denied container as present or missing loses measurement truth."""
    session_source, _ = _seed(tmp_path)
    bundle = tmp_path / f"export{suffix}"
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr("conversations.json", session_source.read_bytes())
    coordinate = f"{bundle}:conversations.json"
    with sqlite3.connect(tmp_path / "source.db") as conn:
        conn.execute("UPDATE raw_sessions SET source_path = ? WHERE raw_id = 'raw-session'", (coordinate,))
        _record_member_fixture(conn, bundle)
        if not retained:
            conn.execute("DELETE FROM blob_refs WHERE ref_id = 'raw-session'")
    original = os.open

    def denied(path: Path, *args: Any, **kwargs: Any) -> int:
        if path == bundle:
            raise PermissionError("synthetic denied inventory")
        return original(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(os, "open", denied)
        check = _run(tmp_path)
    assert check.status is OutcomeStatus.ERROR
    assert _count(check, "source_unavailable") == int(retained)
    assert _count(check, "missing_blob") == int(not retained)
    assert _count(check, "source_lost") == 0
    assert _count(check, "source_missing") == 0
    assert _count(check, "materialized") == 0
    # A new audit measures the recovered inventory rather than caching its denial.
    recovered = _run(tmp_path)
    assert _count(recovered, "source_unavailable") == 0
    assert _count(recovered, "materialized") == int(retained)
    assert _count(recovered, "missing_blob") == int(not retained)


def test_conservation_rechecks_cached_inventory_after_container_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mutation: caching by pathname conserves a member removed between two raw probes."""
    from polylogue.maintenance import source_conservation

    _seed(tmp_path)
    bundle = tmp_path / "export.zip"
    replacement = tmp_path / "replacement.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr("session.json", b"{}")
        archive.writestr("sidecar.json", b"{}")
    with zipfile.ZipFile(replacement, "w") as archive:
        archive.writestr("unrelated.json", b"{}")
    with sqlite3.connect(tmp_path / "source.db") as conn:
        conn.execute(
            "UPDATE raw_sessions SET source_path = ? WHERE raw_id = 'raw-session'", (f"{bundle}:session.json",)
        )
        conn.execute(
            "UPDATE raw_sessions SET source_path = ? WHERE raw_id = 'raw-sidecar'", (f"{bundle}:sidecar.json",)
        )
        _record_member_fixture(conn, bundle, member="session.json")
        _record_member_fixture(conn, bundle, raw_id="raw-sidecar", member="sidecar.json", entry_ordinal=1)
        conn.execute("DELETE FROM blob_refs")
    original = source_conservation._source_presence
    replaced = False

    def replace_after_first_probe(
        root: Path,
        source_path: str,
        inventories: dict[Path, tuple[tuple[int, int, int, int, int], frozenset[str] | bool]],
        *,
        captured_coordinate: str | None = None,
    ) -> bool | None:
        nonlocal replaced
        present = original(root, source_path, inventories, captured_coordinate=captured_coordinate)
        if not replaced:
            replaced = True
            replacement.replace(bundle)
        return present

    monkeypatch.setattr(source_conservation, "_source_presence", replace_after_first_probe)
    check = _run(tmp_path)
    assert check.status is OutcomeStatus.ERROR
    # Regardless of raw row order, both old names cannot be conserved.
    assert _count(check, "source_lost") == 1


def test_conservation_refuses_inventory_replaced_while_descriptor_is_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mutation: descriptor inventory without path identity check publishes an old ZIP under a new path."""
    _seed(tmp_path)
    bundle = tmp_path / "export.zip"
    replacement = tmp_path / "replacement.zip"
    for path in (bundle, replacement):
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("session.json", b"{}")
    with sqlite3.connect(tmp_path / "source.db") as conn:
        conn.execute(
            "UPDATE raw_sessions SET source_path = ? WHERE raw_id = 'raw-session'", (f"{bundle}:session.json",)
        )
        _record_member_fixture(conn, bundle, member="session.json")
    original = zipfile.ZipFile.infolist

    def replace_after_namelist(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
        names = original(archive)
        replacement.replace(bundle)
        return names

    monkeypatch.setattr(zipfile.ZipFile, "infolist", replace_after_namelist)
    check = _run(tmp_path)
    assert check.status is OutcomeStatus.ERROR
    assert _count(check, "source_unavailable") == 1
    assert _count(check, "source_lost") == 0
    assert _count(check, "materialized") == 0


def test_completed_intake_does_not_owe_declared_out_of_scope_database_projection(tmp_path: Path) -> None:
    """Mutation: physical layout discovery demands a database ordinary intake correctly excludes."""
    from polylogue.sources.live.production_baseline import capture_production_source_baseline
    from polylogue.sources.live.watcher import WatchSource
    from polylogue.sources.source_layout import source_layout_for

    _seed(tmp_path)
    state = tmp_path / "codex-state"
    state.mkdir()
    projection = state / "thread_history_1.sqlite"
    with sqlite3.connect(projection) as connection:
        connection.execute("CREATE TABLE projection(value TEXT)")
        connection.execute("INSERT INTO projection VALUES ('already represented by source sessions')")
    hermes = tmp_path / "hermes"
    (hermes / "sessions").mkdir(parents=True)
    request_dump = hermes / "sessions" / "request_dump_neutral.json"
    request_dump.write_bytes(b'{"neutral":"excluded request metadata"}')
    baseline = capture_production_source_baseline(
        (
            WatchSource("codex-state", state, layout=source_layout_for("codex-state")),
            WatchSource("hermes", hermes, layout=source_layout_for("hermes")),
        ),
        operation_id="excluded-projection",
    )
    assert baseline.accepted == ()
    assert any(row.path == str(projection) and row.disposition == "excluded" for row in baseline.decisions)
    with build_source_frontier(
        [
            SourceDeclaration("acquired", SourceRole.DIRECTORY, tmp_path / "sources", True),
            SourceDeclaration("state", SourceRole.DIRECTORY, state, True, "codex-state"),
            SourceDeclaration("hermes", SourceRole.DIRECTORY, hermes, True, "hermes"),
        ]
    ) as frontier:
        assert frontier.complete
        assert frontier.root_states["state"].value == "valid-empty"
        check = _run_with_frontier(tmp_path, frontier)
    assert check.status is OutcomeStatus.OK, check.evidence
    assert _count(check, "frontier_unacquired") == 0
    assert projection.is_file()


@pytest.mark.parametrize(
    "source_present,blob_present,term",
    [(True, False, "missing_blob"), (False, True, "source_missing"), (False, False, "source_lost")],
)
def test_retained_cas_presence_is_independent_of_original_source(
    tmp_path: Path, source_present: bool, blob_present: bool, term: str
) -> None:
    """Mutation: CAS existence is probed only after the original disappears."""
    session_source, _ = _seed(tmp_path)
    connection = sqlite3.connect(tmp_path / "source.db")
    try:
        digest = bytes(
            connection.execute("SELECT blob_hash FROM raw_sessions WHERE raw_id = 'raw-session'").fetchone()[0]
        ).hex()
    finally:
        connection.close()
    if not source_present:
        session_source.unlink()
    if not blob_present:
        BlobStore(tmp_path / "blob").blob_path(digest).unlink()
    check = _run(tmp_path)
    assert check.status is (OutcomeStatus.OK if blob_present else OutcomeStatus.ERROR)
    assert _count(check, term) == 1
    assert _terms(check)[term]["blocking"] is (not blob_present)
    assert _count(check, "materialized") == 0
