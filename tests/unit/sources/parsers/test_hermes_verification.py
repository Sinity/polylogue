"""Hermes verification_evidence.db importer tests (fs1.3 / polylogue-wj25).

Follows the same real-bytes discipline as ``test_hermes_import_explain.py``'s
state.db fixtures: SQLite databases are binary, so this repo does not commit
one as a fixture asset -- instead ``_write_verification_evidence_db`` builds
the exact live-verified schema (``agent/verification_evidence.py`` in the
bundled hermes-agent checkout, ``meta.schema_version = 1``) and inserts rows
whose ``kind``/``scope``/``status``/``exit_code`` enum values and table shape
match real rows read read-only from ``~/.hermes/verification_evidence.db``
on 2026-07-18 (commands/paths/output are redacted placeholders; the producer
vocabulary -- lint/typecheck/build/format/check/test/ad_hoc kinds,
targeted/full scopes, passed/failed status -- is preserved verbatim).
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from polylogue.sources.dispatch import detect_provider, parse_payload
from polylogue.sources.import_explain import explain_import_path
from polylogue.sources.parsers import hermes_verification


def _write_verification_evidence_db(path: Path) -> None:
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
        conn.executemany(
            "INSERT INTO verification_events "
            "(created_at, session_id, cwd, root, command, canonical_command, kind, scope, status, exit_code, output_summary) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
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
                (
                    "2026-07-14T19:55:03.237754+00:00",
                    "verify-session-redacted-1",
                    "<redacted>",
                    "<redacted>",
                    "<redacted>",
                    "ad-hoc verification script",
                    "ad_hoc",
                    "targeted",
                    "passed",
                    0,
                    "<redacted>",
                ),
                (
                    "2026-07-14T20:24:20.360490+00:00",
                    "verify-session-redacted-2",
                    "<redacted>",
                    "<redacted>",
                    "<redacted>",
                    "ruff check",
                    "lint",
                    "full",
                    "failed",
                    1,
                    "<redacted>",
                ),
            ],
        )
        conn.executemany(
            "INSERT INTO verification_state (session_id, root, last_event_id, last_edit_at, changed_paths_json) "
            "VALUES (?, ?, ?, ?, ?)",
            [
                ("verify-session-redacted-1", "<redacted-root-a>", 2, None, "[]"),
                (
                    "verify-session-redacted-1",
                    "<redacted-root-b>",
                    None,
                    "2026-07-14T18:06:15.923119+00:00",
                    json.dumps(["<redacted-path-1>", "<redacted-path-2>"]),
                ),
                ("verify-session-redacted-2", "<redacted-root-a>", None, "2026-07-14T20:24:20.360490+00:00", "[]"),
            ],
        )


def test_looks_like_verification_evidence_db_path_matches_real_schema(tmp_path: Path) -> None:
    path = tmp_path / "verification_evidence.db"
    _write_verification_evidence_db(path)
    assert hermes_verification.looks_like_verification_evidence_db_path(path)

    other = tmp_path / "not_it.db"
    with sqlite3.connect(other) as conn:
        conn.execute("CREATE TABLE meta (key TEXT)")
    assert not hermes_verification.looks_like_verification_evidence_db_path(other)


def test_dispatch_detects_and_parses_verification_marker_through_the_real_pipeline(tmp_path: Path) -> None:
    path = tmp_path / "verification_evidence.db"
    _write_verification_evidence_db(path)
    payload = hermes_verification.marker_payload(path)

    assert hermes_verification.looks_like_verification_evidence_db_payload(payload)
    assert detect_provider(payload) is not None

    from polylogue.core.enums import Provider

    assert detect_provider(payload) is Provider.HERMES
    sessions = parse_payload(Provider.HERMES, payload, "fallback-id")
    assert {session.provider_session_id for session in sessions} == {
        "verification:verify-session-redacted-1",
        "verification:verify-session-redacted-2",
    }


def test_parse_groups_events_and_state_by_session_id_and_preserves_command_evidence(tmp_path: Path) -> None:
    path = tmp_path / "verification_evidence.db"
    _write_verification_evidence_db(path)

    sessions = hermes_verification.parse_verification_evidence_db(path)
    by_id = {session.provider_session_id: session for session in sessions}
    assert set(by_id) == {"verification:verify-session-redacted-1", "verification:verify-session-redacted-2"}

    session_1 = by_id["verification:verify-session-redacted-1"]
    events_1 = [e for e in session_1.session_events if e.event_type == "hermes_verification_event"]
    state_1 = [e for e in session_1.session_events if e.event_type == "hermes_verification_state"]
    assert len(events_1) == 2
    assert len(state_1) == 2

    test_event = next(e for e in events_1 if e.payload["kind"] == "test")
    assert test_event.payload["canonical_command"] == "pytest"
    assert test_event.payload["scope"] == "targeted"
    assert test_event.payload["status"] == "passed"
    assert test_event.payload["is_error"] is False
    assert test_event.payload["exit_code"] == 0
    assert test_event.payload["ambiguous_correlation"] is False

    session_2 = by_id["verification:verify-session-redacted-2"]
    events_2 = [e for e in session_2.session_events if e.event_type == "hermes_verification_event"]
    assert len(events_2) == 1
    failed_event = events_2[0]
    assert failed_event.payload["kind"] == "lint"
    assert failed_event.payload["scope"] == "full"
    assert failed_event.payload["status"] == "failed"
    assert failed_event.payload["is_error"] is True
    assert failed_event.payload["exit_code"] == 1


def test_changed_paths_round_trip_as_a_typed_list(tmp_path: Path) -> None:
    path = tmp_path / "verification_evidence.db"
    _write_verification_evidence_db(path)

    sessions = hermes_verification.parse_verification_evidence_db(path)
    session_1 = next(s for s in sessions if s.provider_session_id == "verification:verify-session-redacted-1")
    state_events = [e for e in session_1.session_events if e.event_type == "hermes_verification_state"]

    with_paths = next(e for e in state_events if e.payload["changed_path_count"] == 2)
    assert with_paths.payload["changed_paths"] == ["<redacted-path-1>", "<redacted-path-2>"]
    without_paths = next(e for e in state_events if e.payload["changed_path_count"] == 0)
    assert without_paths.payload["changed_paths"] == []
    assert without_paths.payload["last_event_id"] == 2


def test_default_session_id_is_marked_ambiguous_correlation_not_trusted(tmp_path: Path) -> None:
    """The producer's own fallback (``session_id=str(session_id or "default")``)
    must not be silently treated as a real, distinguishable session id."""

    path = tmp_path / "verification_evidence.db"
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE verification_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL, session_id TEXT NOT NULL,
                cwd TEXT NOT NULL, root TEXT NOT NULL, command TEXT NOT NULL, canonical_command TEXT NOT NULL,
                kind TEXT NOT NULL, scope TEXT NOT NULL, status TEXT NOT NULL, exit_code INTEGER NOT NULL,
                output_summary TEXT NOT NULL
            );
            CREATE TABLE verification_state (
                session_id TEXT NOT NULL, root TEXT NOT NULL, last_event_id INTEGER, last_edit_at TEXT,
                changed_paths_json TEXT NOT NULL DEFAULT '[]', PRIMARY KEY (session_id, root)
            );
            INSERT INTO meta(key, value) VALUES ('schema_version', '1');
            INSERT INTO verification_events
                (created_at, session_id, cwd, root, command, canonical_command, kind, scope, status, exit_code, output_summary)
                VALUES ('2026-07-18T00:00:00Z', 'default', '<redacted>', '<redacted>', '<redacted>', 'pytest', 'test', 'targeted', 'passed', 0, '<redacted>');
            """
        )

    sessions = hermes_verification.parse_verification_evidence_db(path)
    [session] = sessions
    assert session.provider_session_id == "verification:default"
    [event] = [e for e in session.session_events if e.event_type == "hermes_verification_event"]
    assert event.payload["ambiguous_correlation"] is True

    fidelity = hermes_verification.import_fidelity_declaration(sessions)
    assert fidelity.capabilities["correlation"].status == "degraded"
    assert fidelity.capabilities["correlation"].counts == {"ambiguous": 1}


def test_import_fidelity_declares_exact_for_populated_capabilities(tmp_path: Path) -> None:
    path = tmp_path / "verification_evidence.db"
    _write_verification_evidence_db(path)

    sessions = hermes_verification.parse_verification_evidence_db(path)
    fidelity = hermes_verification.import_fidelity_declaration(sessions)

    assert fidelity.acquisition_method == "sqlite_backup"
    assert fidelity.retained_blob_reproducibility.status == "exact"
    assert fidelity.capabilities["command_evidence"].status == "exact"
    assert fidelity.capabilities["outcome_evidence"].status == "exact"
    assert fidelity.capabilities["output_evidence"].status == "exact"
    assert fidelity.capabilities["changed_paths"].status == "exact"
    assert fidelity.capabilities["correlation"].status == "exact"
    # Producer-side retention (30-day/100-per-root/10k-total pruning) means this
    # import can never honestly claim a complete historical ledger.
    assert fidelity.capabilities["retention_completeness"].status == "degraded"
    assert any(caveat.startswith("retention_completeness:") for caveat in fidelity.caveats)


def test_import_explain_reports_verification_evidence_db_fidelity(tmp_path: Path) -> None:
    path = tmp_path / "verification_evidence.db"
    _write_verification_evidence_db(path)

    [entry] = explain_import_path(path, source_name="hermes").entries
    assert entry.detector == "hermes_verification_evidence_db"
    assert entry.parser == "hermes_verification_evidence_db"
    assert entry.produced.sessions == 2
    assert entry.fidelity is not None
    assert entry.fidelity.schema_version == 1
    assert entry.fidelity.capabilities["command_evidence"].status == "exact"


def test_parse_is_idempotent_across_repeated_reads(tmp_path: Path) -> None:
    path = tmp_path / "verification_evidence.db"
    _write_verification_evidence_db(path)

    first = hermes_verification.parse_verification_evidence_db(path)
    second = hermes_verification.parse_verification_evidence_db(path)
    assert [s.model_dump(mode="json") for s in first] == [s.model_dump(mode="json") for s in second]


def test_two_profiles_with_the_same_raw_session_id_do_not_collapse(tmp_path: Path) -> None:
    """The actual collapse bug polylogue-y9zx fixes (mirrors fs1.14 / PR #3224
    for hermes_spans.py): two separate Hermes installs (profiles) reusing the
    same raw session id must produce two distinct verification-ledger archive
    sessions with two distinct parent edges, never one silently overwriting
    the other."""

    profile_a = tmp_path / "install-a"
    profile_b = tmp_path / "install-b"
    path = tmp_path / "verification_evidence.db"
    _write_verification_evidence_db(path)

    sessions_a = hermes_verification.parse_verification_evidence_db(path, profile_root=profile_a)
    sessions_b = hermes_verification.parse_verification_evidence_db(path, profile_root=profile_b)

    ids_a = {session.provider_session_id for session in sessions_a}
    ids_b = {session.provider_session_id for session in sessions_b}
    assert ids_a.isdisjoint(ids_b)

    parents_a = {session.parent_session_provider_id for session in sessions_a}
    parents_b = {session.parent_session_provider_id for session in sessions_b}
    assert parents_a.isdisjoint(parents_b)

    from polylogue.sources.parsers.hermes_identity import profile_key, qualified_session_id

    session_a = next(
        s for s in sessions_a if s.provider_session_id.startswith("verification:verify-session-redacted-1")
    )
    expected_key = profile_key(profile_a)
    assert (
        session_a.provider_session_id
        == f"verification:{qualified_session_id('verify-session-redacted-1', expected_key)}"
    )
    assert session_a.parent_session_provider_id == qualified_session_id("verify-session-redacted-1", expected_key)


def test_verification_family_prefix_stays_distinguishable_from_atif_atof_observer_family(tmp_path: Path) -> None:
    """The ``verification:`` family prefix must remain distinguishable from
    ATIF/ATOF's ``observer:`` family even when both qualify the same raw
    session id against the same profile -- these are independently-acquired
    evidence classes that must never collide on archive session identity."""

    profile_root = tmp_path / "hermes-install"
    path = tmp_path / "verification_evidence.db"
    _write_verification_evidence_db(path)

    sessions = hermes_verification.parse_verification_evidence_db(path, profile_root=profile_root)
    session_1 = next(s for s in sessions if "verify-session-redacted-1" in s.provider_session_id)

    from polylogue.sources.parsers.hermes_identity import profile_key, qualified_session_id
    from polylogue.sources.parsers.hermes_spans import observer_session_provider_id as atof_observer_id

    key = profile_key(profile_root)
    atof_id_for_same_raw_id_and_profile = atof_observer_id("verify-session-redacted-1", key)

    assert session_1.provider_session_id != atof_id_for_same_raw_id_and_profile
    assert session_1.provider_session_id == f"verification:{qualified_session_id('verify-session-redacted-1', key)}"
    assert atof_id_for_same_raw_id_and_profile == f"observer:{qualified_session_id('verify-session-redacted-1', key)}"


def test_verification_asserts_profile_qualified_parent_session_link(tmp_path: Path) -> None:
    path = tmp_path / "verification_evidence.db"
    _write_verification_evidence_db(path)
    profile_root = tmp_path / "hermes-install"

    sessions = hermes_verification.parse_verification_evidence_db(path, profile_root=profile_root)
    from polylogue.sources.parsers.hermes_identity import profile_key, qualified_session_id

    expected_key = profile_key(profile_root)
    for session in sessions:
        raw_id = session.provider_session_id.removeprefix("verification:").split("@profile-", 1)[0]
        assert session.parent_session_provider_id == qualified_session_id(raw_id, expected_key)

    fidelity = hermes_verification.import_fidelity_declaration(sessions)
    assert fidelity.capabilities["parent_session_link"].status == "inferred"


def test_verification_fails_closed_without_a_known_profile_root(tmp_path: Path) -> None:
    """No profile_root means no parent edge -- a guessed edge is worse than
    no edge; the (unqualified) session identity itself must still be stable."""

    path = tmp_path / "verification_evidence.db"
    _write_verification_evidence_db(path)

    sessions_unknown_profile = hermes_verification.parse_verification_evidence_db(path)
    assert {s.provider_session_id for s in sessions_unknown_profile} == {
        "verification:verify-session-redacted-1",
        "verification:verify-session-redacted-2",
    }
    assert all(s.parent_session_provider_id is None for s in sessions_unknown_profile)

    fidelity = hermes_verification.import_fidelity_declaration(sessions_unknown_profile)
    assert fidelity.capabilities["parent_session_link"].status == "absent"


def test_dispatch_wires_profile_root_from_source_path_for_verification_marker(tmp_path: Path) -> None:
    """Mirrors the ATIF dispatch wiring: the dispatch call site derives
    profile_root from the source path's parent directory, so two installs
    whose marker payloads are dispatched from different source paths get
    distinct, non-colliding session identity without the caller having to
    know about profile qualification at all."""

    install_a = tmp_path / "install-a"
    install_b = tmp_path / "install-b"
    install_a.mkdir()
    install_b.mkdir()
    path_a = install_a / "verification_evidence.db"
    path_b = install_b / "verification_evidence.db"
    _write_verification_evidence_db(path_a)
    _write_verification_evidence_db(path_b)

    from polylogue.core.enums import Provider

    payload_a = hermes_verification.marker_payload(path_a)
    payload_b = hermes_verification.marker_payload(path_b)

    sessions_a = parse_payload(Provider.HERMES, payload_a, "fallback-id", source_path=str(path_a))
    sessions_b = parse_payload(Provider.HERMES, payload_b, "fallback-id", source_path=str(path_b))

    ids_a = {session.provider_session_id for session in sessions_a}
    ids_b = {session.provider_session_id for session in sessions_b}
    assert ids_a.isdisjoint(ids_b)
    assert all(session.parent_session_provider_id for session in sessions_a)
    assert all(session.parent_session_provider_id for session in sessions_b)


def test_empty_verification_evidence_db_produces_no_sessions_not_a_crash(tmp_path: Path) -> None:
    path = tmp_path / "verification_evidence.db"
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE verification_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL, session_id TEXT NOT NULL,
                cwd TEXT NOT NULL, root TEXT NOT NULL, command TEXT NOT NULL, canonical_command TEXT NOT NULL,
                kind TEXT NOT NULL, scope TEXT NOT NULL, status TEXT NOT NULL, exit_code INTEGER NOT NULL,
                output_summary TEXT NOT NULL
            );
            CREATE TABLE verification_state (
                session_id TEXT NOT NULL, root TEXT NOT NULL, last_event_id INTEGER, last_edit_at TEXT,
                changed_paths_json TEXT NOT NULL DEFAULT '[]', PRIMARY KEY (session_id, root)
            );
            INSERT INTO meta(key, value) VALUES ('schema_version', '1');
            """
        )

    assert hermes_verification.looks_like_verification_evidence_db_path(path)
    sessions = hermes_verification.parse_verification_evidence_db(path)
    assert sessions == []

    fidelity = hermes_verification.import_fidelity_declaration(sessions)
    assert fidelity.capabilities["command_evidence"].status == "absent"
