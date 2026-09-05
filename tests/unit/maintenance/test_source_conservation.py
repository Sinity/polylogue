"""Red twins for the ``source-conservation`` owner check of ``verify-archive``.

Each test names the mutation that would make it vacuous: a deleted source
file, an injected unadmitted index row, a session materialized from a
declared non-session artifact, a fragment-shaped identity, a parsed raw that
no rule explains. The archive is built through the production tier bootstrap
with real source files under ``tmp_path``; no ambient data is read.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

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
from polylogue.sources.origin_specs import lowering_fingerprint, parser_fingerprint_for_origin
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


def _insert_session(conn: sqlite3.Connection, *, origin: str, native_id: str, raw_id: str | None) -> str:
    conn.execute(
        """
        INSERT INTO sessions(
            native_id, origin, raw_id, parser_fingerprint, lowering_fingerprint, content_hash, message_count
        ) VALUES (?, ?, ?, ?, ?, ?, 1)
        """,
        (native_id, origin, raw_id, parser_fingerprint_for_origin(origin), lowering_fingerprint(), b"s" * 32),
    )
    session_id = f"{origin}:{native_id}"
    conn.execute(
        """
        INSERT INTO messages(session_id, position, role, material_origin, content_hash)
        VALUES (?, 0, 'user', 'human_authored', ?)
        """,
        (session_id, b"m" * 32),
    )
    conn.execute(
        """
        INSERT INTO blocks(message_id, session_id, position, block_type, text)
        VALUES (?, ?, 0, 'text', 'hello world')
        """,
        (f"{session_id}:p:0.0", session_id),
    )
    return session_id


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
        "unexplained",
        "phantom_declared_non_session_lineage",
    } <= set(terms)
    for term in terms.values():
        assert isinstance(term, dict)
        assert isinstance(term["rule"], str) and term["rule"]
        assert isinstance(term["blocking"], bool)
