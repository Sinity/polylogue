"""Tests for the candidate-only secret detector (polylogue-27m).

Anti-vacuity: these tests exercise the real production
``scan_text_for_secret_candidates`` regex/entropy rules and the real
``upsert_assertion`` write chokepoint via ``record_secret_candidates``.
Mutating any pattern (e.g. shortening the AWS key length, dropping the
entropy filter) or removing the ``author_kind="detector"`` argument makes
the corresponding assertion below fail.

``TestScanSessionForSecretCandidates`` covers the polylogue-27m fix-round
production entrypoint (``scan_session_for_secret_candidates``): a real
``index.db`` block read, feeding the real scan rules above, through the
real write chokepoint into ``user.db``. An earlier revision of this module
had no such caller -- these tests fail if that wiring regresses (e.g. the
function stops reading real block text, or stops calling
``record_secret_candidates``).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from polylogue.core.enums import AssertionKind, AssertionStatus
from polylogue.security.secret_scan import (
    SECRET_SCAN_VERSION,
    count_pending_secret_scan_sessions,
    record_secret_candidates,
    scan_archive_for_secret_candidates,
    scan_session_for_secret_candidates,
    scan_text_for_secret_candidates,
    secret_candidate_assertion_id,
    select_pending_secret_scan_session_ids,
)
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.user_write import read_assertion_envelope


class TestScanTextForSecretCandidates:
    def test_finds_aws_access_key(self) -> None:
        text = "export AWS_ACCESS_KEY_ID=AKIAABCDEFGHIJKLMNOP"
        spans = scan_text_for_secret_candidates(text)
        assert any(span.pattern_id == "aws-access-key-id" for span in spans)

    def test_finds_anthropic_key(self) -> None:
        text = "ANTHROPIC_API_KEY=sk-ant-api03-abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJ"
        spans = scan_text_for_secret_candidates(text)
        assert any(span.pattern_id == "anthropic-api-key" for span in spans)

    def test_finds_github_token(self) -> None:
        text = "token: ghp_" + "a" * 40
        spans = scan_text_for_secret_candidates(text)
        assert any(span.pattern_id == "github-token" for span in spans)

    def test_finds_private_key_block(self) -> None:
        text = "-----BEGIN RSA PRIVATE KEY-----\nMIIB...\n-----END RSA PRIVATE KEY-----"
        spans = scan_text_for_secret_candidates(text)
        assert any(span.pattern_id == "private-key-block" for span in spans)

    def test_generic_assignment_with_high_entropy_is_flagged(self) -> None:
        text = 'api_key: "Q7xmP2vZ9kR4tY8wL1nB6cJ0hF5sD3g"'
        spans = scan_text_for_secret_candidates(text)
        assert any(span.pattern_id == "generic-credential-assignment" for span in spans)

    def test_low_entropy_generic_assignment_is_not_flagged(self) -> None:
        # "changemechangemechangeme" is 25 chars (passes the length gate) but
        # low-entropy repetition -- a real secret would not look like this.
        text = "password=changemechangemechangeme"
        spans = scan_text_for_secret_candidates(text)
        assert not any(span.pattern_id == "generic-credential-assignment" for span in spans)

    def test_never_returns_the_matched_literal(self) -> None:
        secret_value = "AKIASECRETVALUE12345"
        text = f"key={secret_value}"
        spans = scan_text_for_secret_candidates(text)
        assert spans
        for span in spans:
            # SecretCandidateSpan is a frozen dataclass with exactly these
            # fields; assert none of them is (or contains) the literal.
            assert span.pattern_id != secret_value
            assert secret_value not in span.fingerprint
            assert isinstance(span.start, int)
            assert isinstance(span.end, int)
            # repr() must not leak the literal either (guards against a
            # future field addition that accidentally stores raw text).
            assert secret_value not in repr(span)

    def test_fingerprint_is_deterministic_and_one_way(self) -> None:
        text = "AKIAABCDEFGHIJKLMNOP"
        spans_a = scan_text_for_secret_candidates(text)
        spans_b = scan_text_for_secret_candidates(text)
        assert spans_a[0].fingerprint == spans_b[0].fingerprint
        assert len(spans_a[0].fingerprint) == 64  # sha256 hex
        assert spans_a[0].fingerprint != text

    def test_overlapping_matches_deduplicate(self) -> None:
        # A private-key block also contains "KEY" text that could tempt a
        # looser generic rule; the tighter rule (checked first) must claim
        # the span and the generic rule must not also fire on the same range.
        text = "-----BEGIN PRIVATE KEY-----\nkey=MIIBVeryLongBase64LookingValueXYZ\n-----END PRIVATE KEY-----"
        spans = scan_text_for_secret_candidates(text)
        ranges = [(s.start, s.end) for s in spans]
        for i, (start_a, end_a) in enumerate(ranges):
            for start_b, end_b in ranges[i + 1 :]:
                assert not (start_a < end_b and end_a > start_b), "candidate spans must not overlap"

    def test_plain_english_is_not_flagged(self) -> None:
        text = "This is a normal sentence about how important passwords are for security."
        spans = scan_text_for_secret_candidates(text)
        assert spans == []


class TestRecordSecretCandidates:
    @pytest.fixture
    def user_db(self, tmp_path: Path) -> Path:
        db_path = tmp_path / "user.db"
        initialize_archive_database(db_path, ArchiveTier.USER)
        return db_path

    def test_writes_non_injectable_candidate_assertion(self, user_db: Path) -> None:
        text = "AWS_ACCESS_KEY_ID=AKIAABCDEFGHIJKLMNOP"
        spans = scan_text_for_secret_candidates(text)
        assert spans

        conn = sqlite3.connect(user_db)
        try:
            with conn:
                written_ids = record_secret_candidates(
                    conn,
                    target_ref="message:codex-session:demo:m1",
                    spans=spans,
                    now_ms=1_000,
                )
            assert written_ids
            envelope = read_assertion_envelope(conn, written_ids[0])
            assert envelope is not None
            assert envelope.kind is AssertionKind.SECRET_CANDIDATE
            # author_kind="detector" must coerce candidate status + non-inject
            # policy regardless of what record_secret_candidates asked for
            # (the upsert_assertion promotion-gate invariant, #37t.15).
            assert envelope.author_kind == "detector"
            assert envelope.status is AssertionStatus.CANDIDATE
            assert envelope.context_policy.get("inject") is False
            assert isinstance(envelope.value, dict)
            assert "AKIA" not in str(envelope.value)
            assert envelope.value["fingerprint_sha256"] == spans[0].fingerprint
        finally:
            conn.close()

    def test_no_matched_literal_anywhere_in_the_database_file(self, user_db: Path) -> None:
        secret_value = "AKIASUPERSECRETVALUE"
        text = f"AWS_ACCESS_KEY_ID={secret_value}"
        spans = scan_text_for_secret_candidates(text)
        assert spans

        conn = sqlite3.connect(user_db)
        try:
            with conn:
                record_secret_candidates(
                    conn,
                    target_ref="message:codex-session:demo:m1",
                    spans=spans,
                    now_ms=1_000,
                )
        finally:
            conn.close()

        raw_bytes = user_db.read_bytes()
        assert secret_value.encode() not in raw_bytes

    def test_rescanning_identical_content_is_idempotent(self, user_db: Path) -> None:
        text = "AWS_ACCESS_KEY_ID=AKIAABCDEFGHIJKLMNOP"
        spans = scan_text_for_secret_candidates(text)
        target_ref = "message:codex-session:demo:m1"

        conn = sqlite3.connect(user_db)
        try:
            with conn:
                first = record_secret_candidates(conn, target_ref=target_ref, spans=spans, now_ms=1_000)
            with conn:
                second = record_secret_candidates(conn, target_ref=target_ref, spans=spans, now_ms=2_000)
            assert first == second
            assert secret_candidate_assertion_id(target_ref, spans[0]) == first[0]
            count = conn.execute(
                "SELECT COUNT(*) FROM assertions WHERE kind = ?", (AssertionKind.SECRET_CANDIDATE.value,)
            ).fetchone()[0]
            assert count == len(spans)
        finally:
            conn.close()


class TestScanSessionForSecretCandidates:
    """Production-wiring coverage (polylogue-27m fix round).

    ``scan_session_for_secret_candidates`` is the real caller that turns
    the scanner above from "functions that exist" into "an operator running
    `polylogue ops scan-secrets` against their archive gets a finding" --
    real index.db block read -> real regex/entropy rules -> real
    ``record_secret_candidates`` write chokepoint into user.db.
    """

    def _seed_session_with_block_text(self, tmp_path: Path, *, native_id: str, text: str) -> str:
        index_db = tmp_path / "index.db"
        initialize_archive_database(index_db, ArchiveTier.INDEX)
        conn = sqlite3.connect(index_db)
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            conn.execute(
                "INSERT INTO sessions (native_id, origin, title, content_hash, created_at_ms, updated_at_ms) "
                "VALUES (?, 'codex-session', ?, zeroblob(32), 1000, 2000)",
                (native_id, f"Session {native_id}"),
            )
            session_id = conn.execute("SELECT session_id FROM sessions WHERE native_id = ?", (native_id,)).fetchone()[0]
            conn.execute(
                "INSERT INTO messages (session_id, native_id, position, role, content_hash) "
                "VALUES (?, 'm1', 0, 'user', zeroblob(32))",
                (session_id,),
            )
            message_id = conn.execute("SELECT message_id FROM messages WHERE session_id = ?", (session_id,)).fetchone()[
                0
            ]
            conn.execute(
                "INSERT INTO blocks (message_id, session_id, position, block_type, text) VALUES (?, ?, 0, 'text', ?)",
                (message_id, session_id, text),
            )
            conn.commit()
        finally:
            conn.close()
        return str(session_id)

    def test_not_found_for_unknown_session(self, tmp_path: Path) -> None:
        result = scan_session_for_secret_candidates(tmp_path, "codex-session:does-not-exist")
        assert result.found is False

    def test_finds_and_records_a_real_credential_shaped_span(self, tmp_path: Path) -> None:
        session_id = self._seed_session_with_block_text(
            tmp_path, native_id="scan-1", text="ANTHROPIC_API_KEY=sk-ant-api03-" + "a" * 60
        )
        result = scan_session_for_secret_candidates(tmp_path, session_id)
        assert result.found is True
        assert result.blocks_scanned == 1
        assert result.candidates_found >= 1
        assert len(result.written_assertion_ids) == result.candidates_found

        user_conn = sqlite3.connect(tmp_path / "user.db")
        try:
            row = user_conn.execute(
                "SELECT kind, target_ref FROM assertions WHERE assertion_id = ?",
                (result.written_assertion_ids[0],),
            ).fetchone()
        finally:
            user_conn.close()
        assert row is not None
        assert row[0] == AssertionKind.SECRET_CANDIDATE.value
        assert row[1].startswith("block:")

    def test_never_persists_the_matched_literal(self, tmp_path: Path) -> None:
        secret_value = "sk-ant-api03-" + "b" * 60
        session_id = self._seed_session_with_block_text(
            tmp_path, native_id="scan-2", text=f"ANTHROPIC_API_KEY={secret_value}"
        )
        result = scan_session_for_secret_candidates(tmp_path, session_id)
        assert result.candidates_found >= 1

        raw_bytes = (tmp_path / "user.db").read_bytes()
        assert secret_value.encode() not in raw_bytes

    def test_no_candidates_for_ordinary_text(self, tmp_path: Path) -> None:
        session_id = self._seed_session_with_block_text(
            tmp_path, native_id="scan-3", text="just an ordinary chat message"
        )
        result = scan_session_for_secret_candidates(tmp_path, session_id)
        assert result.found is True
        assert result.candidates_found == 0
        assert result.written_assertion_ids == ()

    def test_rescanning_is_idempotent(self, tmp_path: Path) -> None:
        session_id = self._seed_session_with_block_text(
            tmp_path, native_id="scan-4", text="AWS_ACCESS_KEY_ID=AKIAABCDEFGHIJKLMNOP"
        )
        first = scan_session_for_secret_candidates(tmp_path, session_id, now_ms=1_000)
        second = scan_session_for_secret_candidates(tmp_path, session_id, now_ms=2_000)
        assert first.written_assertion_ids == second.written_assertion_ids

        user_conn = sqlite3.connect(tmp_path / "user.db")
        try:
            count = user_conn.execute(
                "SELECT COUNT(*) FROM assertions WHERE kind = ?", (AssertionKind.SECRET_CANDIDATE.value,)
            ).fetchone()[0]
        finally:
            user_conn.close()
        assert count == len(first.written_assertion_ids)


def _seed_archive_session(archive_root: Path, *, native_id: str, text: str) -> str:
    """Write one real session with a single text block into ``archive_root``.

    Shared by the bulk-scan tests below: builds a multi-session ``index.db``
    fixture one call at a time, mirroring
    ``TestScanSessionForSecretCandidates._seed_session_with_block_text`` but
    against a stable ``archive_root`` shared across calls (the bulk scanner
    operates on the whole archive, not one caller-known session).
    """
    archive_root.mkdir(parents=True, exist_ok=True)
    index_db = archive_root / "index.db"
    initialize_archive_database(index_db, ArchiveTier.INDEX)
    conn = sqlite3.connect(index_db)
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        conn.execute(
            "INSERT INTO sessions (native_id, origin, title, content_hash, created_at_ms, updated_at_ms) "
            "VALUES (?, 'codex-session', ?, zeroblob(32), 1000, 2000)",
            (native_id, f"Session {native_id}"),
        )
        session_id = conn.execute("SELECT session_id FROM sessions WHERE native_id = ?", (native_id,)).fetchone()[0]
        conn.execute(
            "INSERT INTO messages (session_id, native_id, position, role, content_hash) "
            "VALUES (?, 'm1', 0, 'user', zeroblob(32))",
            (session_id,),
        )
        message_id = conn.execute("SELECT message_id FROM messages WHERE session_id = ?", (session_id,)).fetchone()[0]
        conn.execute(
            "INSERT INTO blocks (message_id, session_id, position, block_type, text) VALUES (?, ?, 0, 'text', ?)",
            (message_id, session_id, text),
        )
        conn.commit()
    finally:
        conn.close()
    return str(session_id)


class TestScanArchiveForSecretCandidates:
    """Bulk/archive-wide scan coverage (polylogue-layg.1).

    Anti-vacuity: an implementation that silently skips sessions in a batch
    (the exact regression class this bead exists to prevent -- an operator
    with no session id in hand previously had no way to discover candidates
    at all) fails ``test_full_sweep_covers_every_session_and_finds_every_planted_secret``,
    because it asserts every planted session ends up with a recorded
    candidate and zero remaining pending coverage, not merely that *some*
    candidates were found.
    """

    def test_missing_archive_is_a_bounded_noop(self, tmp_path: Path) -> None:
        result = scan_archive_for_secret_candidates(tmp_path / "does-not-exist")
        assert result.sessions_scanned == 0
        assert result.remaining_pending == 0
        assert result.more_pending is False

    def test_full_sweep_covers_every_session_and_finds_every_planted_secret(self, tmp_path: Path) -> None:
        archive_root = tmp_path / "archive"
        planted_secret_sessions = {
            _seed_archive_session(archive_root, native_id=f"bulk-{i}", text=f"AWS_ACCESS_KEY_ID=AKIA{i:016X}")
            for i in range(5)
        }
        clean_session = _seed_archive_session(archive_root, native_id="bulk-clean", text="just an ordinary message")
        all_sessions = planted_secret_sessions | {clean_session}

        # A page size smaller than the total session count forces the bulk
        # scanner to require multiple calls -- the shape a mutation that
        # "forgets" to advance past an already-scanned batch (double-scans
        # the same page forever, or drops the tail) would fail under.
        scanned_ids: set[str] = set()
        total_candidates = 0
        seen_pages = 0
        result = scan_archive_for_secret_candidates(archive_root, max_sessions=2)
        while True:
            seen_pages += 1
            scanned_ids.update(result.scanned_session_ids)
            total_candidates += result.candidates_found
            if not result.more_pending:
                break
            result = scan_archive_for_secret_candidates(archive_root, max_sessions=2)

        assert seen_pages > 1, "fixture must require more than one page to exercise resumable paging"
        assert scanned_ids == all_sessions, "every session in the archive must be covered exactly once"
        assert total_candidates == len(planted_secret_sessions), "every planted secret must be found"

        remaining = count_pending_secret_scan_sessions(archive_root / "index.db", archive_root / "ops.db")
        assert remaining == 0, "no session may be left uncovered after a full sweep"

        user_conn = sqlite3.connect(archive_root / "user.db")
        try:
            count = user_conn.execute(
                "SELECT COUNT(*) FROM assertions WHERE kind = ?", (AssertionKind.SECRET_CANDIDATE.value,)
            ).fetchone()[0]
        finally:
            user_conn.close()
        assert count == len(planted_secret_sessions)

    def test_interrupt_and_resume_covers_all_sessions_exactly_once(self, tmp_path: Path) -> None:
        archive_root = tmp_path / "archive"
        sessions = {
            _seed_archive_session(archive_root, native_id=f"resume-{i}", text=f"AWS_ACCESS_KEY_ID=AKIA{i:016X}")
            for i in range(6)
        }

        # Simulate a kill after the first bounded page ("interrupted").
        first = scan_archive_for_secret_candidates(archive_root, max_sessions=3)
        assert first.sessions_scanned == 3
        assert first.more_pending is True

        # A fresh call with no state carried over ("resumed") must pick up
        # exactly the sessions the first page did not cover -- never
        # re-scanning a covered session (which would duplicate assertions
        # were the write chokepoint not idempotent) and never skipping one.
        pending_after_first = set(
            select_pending_secret_scan_session_ids(archive_root / "index.db", archive_root / "ops.db", limit=100)
        )
        assert pending_after_first == sessions - set(first.scanned_session_ids)

        second = scan_archive_for_secret_candidates(archive_root, max_sessions=100)
        assert second.more_pending is False
        assert set(second.scanned_session_ids) == pending_after_first

        total_scanned = set(first.scanned_session_ids) | set(second.scanned_session_ids)
        assert total_scanned == sessions

        user_conn = sqlite3.connect(archive_root / "user.db")
        try:
            count = user_conn.execute(
                "SELECT COUNT(*) FROM assertions WHERE kind = ?", (AssertionKind.SECRET_CANDIDATE.value,)
            ).fetchone()[0]
        finally:
            user_conn.close()
        assert count == len(sessions), "resuming must not duplicate candidates for any session"

    def test_scanner_version_bump_schedules_intentional_rescan(self, tmp_path: Path) -> None:
        archive_root = tmp_path / "archive"
        session_id = _seed_archive_session(
            archive_root, native_id="version-bump", text="AWS_ACCESS_KEY_ID=AKIAABCDEFGHIJKLMNOP"
        )
        first = scan_archive_for_secret_candidates(archive_root, scanner_version=1)
        assert session_id in first.scanned_session_ids

        # At the same version, the session is already covered: nothing to do.
        same_version = scan_archive_for_secret_candidates(archive_root, scanner_version=1)
        assert same_version.sessions_scanned == 0

        # A version bump invalidates the existing coverage row and schedules
        # an intentional rescan.
        bumped = scan_archive_for_secret_candidates(archive_root, scanner_version=2)
        assert session_id in bumped.scanned_session_ids

    def test_default_scanner_version_matches_module_constant(self, tmp_path: Path) -> None:
        archive_root = tmp_path / "archive"
        session_id = _seed_archive_session(archive_root, native_id="default-version", text="just text")
        scan_archive_for_secret_candidates(archive_root)

        ops_conn = sqlite3.connect(archive_root / "ops.db")
        try:
            version = ops_conn.execute(
                "SELECT scanner_version FROM secret_scan_status WHERE session_id = ?", (session_id,)
            ).fetchone()[0]
        finally:
            ops_conn.close()
        assert version == SECRET_SCAN_VERSION

    def test_origin_scope_restricts_the_sweep(self, tmp_path: Path) -> None:
        archive_root = tmp_path / "archive"
        codex_session = _seed_archive_session(
            archive_root, native_id="scope-codex", text="AWS_ACCESS_KEY_ID=AKIAABCDEFGHIJKLMNOP"
        )
        # A second origin's session must not be touched by an origin-scoped sweep.
        archive_root.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(archive_root / "index.db")
        try:
            conn.execute(
                "INSERT INTO sessions (native_id, origin, title, content_hash, created_at_ms, updated_at_ms) "
                "VALUES ('scope-claude', 'claude-code-session', 'x', zeroblob(32), 1000, 2000)"
            )
            conn.commit()
        finally:
            conn.close()

        result = scan_archive_for_secret_candidates(archive_root, origin="codex-session")
        assert result.scanned_session_ids == (codex_session,)
