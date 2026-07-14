"""Tests for the candidate-only secret detector (polylogue-27m).

Anti-vacuity: these tests exercise the real production
``scan_text_for_secret_candidates`` regex/entropy rules and the real
``upsert_assertion`` write chokepoint via ``record_secret_candidates``.
Mutating any pattern (e.g. shortening the AWS key length, dropping the
entropy filter) or removing the ``author_kind="detector"`` argument makes
the corresponding assertion below fail.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from polylogue.core.enums import AssertionKind, AssertionStatus
from polylogue.security.secret_scan import (
    record_secret_candidates,
    scan_text_for_secret_candidates,
    secret_candidate_assertion_id,
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
