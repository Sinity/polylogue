from __future__ import annotations

import sqlite3
from pathlib import Path

from polylogue.archive.message.messages import MessageCollection
from polylogue.archive.message.models import Message
from polylogue.archive.message.roles import Role
from polylogue.archive.session.domain_models import Session
from polylogue.core.enums import Origin
from polylogue.insights.transforms import compile_recovery_digest
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.user import USER_SCHEMA_VERSION
from polylogue.storage.sqlite.archive_tiers.user_write import (
    AssertionKind,
    assertion_id_for_transform_candidate,
    list_assertions_for_target,
    read_assertion_envelope,
    upsert_assertion,
    upsert_transform_candidate_assertions,
)
from polylogue.types import SessionId


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    initialize_archive_tier(conn, ArchiveTier.USER)
    return conn


def _recovery_candidate_session() -> Session:
    return Session(
        id=SessionId("codex-session:assertion-demo"),
        origin=Origin.CODEX_SESSION,
        title="Recover assertion candidates",
        messages=MessageCollection(
            messages=[
                Message(
                    id="m1",
                    role=Role.USER,
                    text=(
                        "Goal: connect transform candidates to assertions\nNext: keep candidates private until accepted"
                    ),
                ),
                Message(
                    id="m2",
                    role=Role.ASSISTANT,
                    text="Decision: transform candidates require evidence refs and no default context injection.",
                ),
            ]
        ),
    )


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
        (name,),
    ).fetchone()
    return row is not None


def test_fresh_user_tier_creates_assertions_table(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "user.db")
    try:
        assert _table_exists(conn, "assertions")
        # The table is purely additive: the legacy overlays still exist.
        assert _table_exists(conn, "marks")
        assert _table_exists(conn, "blackboard_notes")
    finally:
        conn.close()


def test_assertion_round_trip_across_kinds(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "user.db")
    try:
        upsert_assertion(
            conn,
            assertion_id="a-mark",
            target_ref="session:session-1",
            kind=AssertionKind.MARK,
            body_text="star this",
            author_ref="user:sinity",
            author_kind="human",
            now_ms=1_700_000_000_000,
        )
        upsert_assertion(
            conn,
            assertion_id="a-decision",
            target_ref="github-issue:1883",
            kind=AssertionKind.DECISION,
            scope_ref="workspace:polylogue",
            key="schema-policy",
            value={"choice": "additive", "wipe": False},
            body_text="user.db is irreplaceable; never bump version",
            status="active",
            visibility="team",
            confidence=0.92,
            staleness={"ttl_days": 30},
            context_policy={"inject": "on_session_start"},
            evidence_refs=["session:session-1", "message:m-7"],
            now_ms=1_700_000_001_000,
        )

        read_mark = read_assertion_envelope(conn, "a-mark")
        read_decision = read_assertion_envelope(conn, "a-decision")
        assert read_mark is not None
        assert read_decision is not None

        assert read_mark.kind == "mark"
        assert read_mark.target_ref == "session:session-1"
        assert read_mark.body_text == "star this"
        assert read_mark.author_kind == "human"
        assert read_mark.value is None
        assert read_mark.evidence_refs == []
        assert read_mark.created_at_ms == 1_700_000_000_000

        assert read_decision.kind == "decision"
        assert read_decision.scope_ref == "workspace:polylogue"
        assert read_decision.key == "schema-policy"
        assert read_decision.value == {"choice": "additive", "wipe": False}
        assert read_decision.status == "active"
        assert read_decision.visibility == "team"
        assert read_decision.confidence == 0.92
        assert read_decision.staleness == {"ttl_days": 30}
        assert read_decision.context_policy == {"inject": "on_session_start"}
        assert read_decision.evidence_refs == ["session:session-1", "message:m-7"]

        # read of a missing assertion returns None, not a raise.
        assert read_assertion_envelope(conn, "absent") is None
    finally:
        conn.close()


def test_assertion_upsert_preserves_created_at_and_updates_fields(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "user.db")
    try:
        first = upsert_assertion(
            conn,
            assertion_id="a-1",
            target_ref="message:m-1",
            kind=AssertionKind.ANNOTATION,
            body_text="initial",
            status="draft",
            now_ms=1_700_000_000_000,
        )
        second = upsert_assertion(
            conn,
            assertion_id="a-1",
            target_ref="message:m-1",
            kind=AssertionKind.ANNOTATION,
            body_text="revised",
            status="active",
            now_ms=1_700_000_005_000,
        )
        assert first.assertion_id == second.assertion_id
        assert second.created_at_ms == 1_700_000_000_000
        assert second.updated_at_ms == 1_700_000_005_000
        assert second.body_text == "revised"
        assert second.status == "active"
    finally:
        conn.close()


def test_supersession_and_status_persist(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "user.db")
    try:
        upsert_assertion(
            conn,
            assertion_id="a-old",
            target_ref="session:session-9",
            kind=AssertionKind.METADATA,
            body_text="superseded",
            status="superseded",
            now_ms=1_700_000_000_000,
        )
        new = upsert_assertion(
            conn,
            assertion_id="a-new",
            target_ref="session:session-9",
            kind=AssertionKind.METADATA,
            body_text="current",
            status="active",
            supersedes=["a-old"],
            now_ms=1_700_000_001_000,
        )
        stored = read_assertion_envelope(conn, "a-new")
        assert stored is not None
        assert stored.supersedes == ["a-old"]
        assert stored.status == "active"

        old = read_assertion_envelope(conn, "a-old")
        assert old is not None
        assert old.status == "superseded"
        assert old.supersedes == []
        assert new.supersedes == ["a-old"]
    finally:
        conn.close()


def test_list_assertions_filters_by_target_and_kind(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "user.db")
    try:
        upsert_assertion(
            conn,
            assertion_id="t1-tag",
            target_ref="session:s-1",
            kind=AssertionKind.TAG,
            value="rust",
            now_ms=1_700_000_000_000,
        )
        upsert_assertion(
            conn,
            assertion_id="t1-note",
            target_ref="session:s-1",
            kind=AssertionKind.NOTE,
            body_text="note on s-1",
            now_ms=1_700_000_001_000,
        )
        upsert_assertion(
            conn,
            assertion_id="t2-tag",
            target_ref="block:b-1",
            kind=AssertionKind.TAG,
            value="python",
            now_ms=1_700_000_002_000,
        )

        for_s1 = list_assertions_for_target(conn, "session:s-1")
        assert {a.assertion_id for a in for_s1} == {"t1-tag", "t1-note"}
        # ordered by created_at_ms
        assert [a.assertion_id for a in for_s1] == ["t1-tag", "t1-note"]

        s1_tags = list_assertions_for_target(conn, "session:s-1", kind="tag")
        assert [a.assertion_id for a in s1_tags] == ["t1-tag"]

        b1 = list_assertions_for_target(conn, "block:b-1", kind="tag")
        assert [a.assertion_id for a in b1] == ["t2-tag"]

        assert list_assertions_for_target(conn, "session:absent") == []
    finally:
        conn.close()


def test_assertion_targets_various_ref_shapes(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "user.db")
    try:
        refs = [
            "session:abc-123",
            "message:abc-123:7",
            "block:abc-123:7:2",
            "github-issue:Sinity/polylogue#1883",
        ]
        for idx, ref in enumerate(refs):
            upsert_assertion(
                conn,
                assertion_id=f"ref-{idx}",
                target_ref=ref,
                kind=AssertionKind.HANDOFF,
                body_text=f"handoff for {ref}",
                now_ms=1_700_000_000_000 + idx,
            )
        for idx, ref in enumerate(refs):
            stored = read_assertion_envelope(conn, f"ref-{idx}")
            assert stored is not None
            assert stored.target_ref == ref
    finally:
        conn.close()


def test_assertions_table_added_additively_without_version_bump(tmp_path: Path) -> None:
    """A pre-existing v1 user.db lacking the table gains it on next open with no wipe."""
    path = tmp_path / "user.db"
    conn = _connect(path)
    try:
        # Seed irreplaceable user data, then simulate an older DB that predates
        # the assertions table by dropping it. user_version stays at 1.
        upsert_assertion(
            conn,
            assertion_id="seed",
            target_ref="session:keep-me",
            kind=AssertionKind.LESSON,
            body_text="irreplaceable user input",
            now_ms=1_700_000_000_000,
        )
        conn.execute("DROP TABLE assertions")
        conn.commit()
        assert not _table_exists(conn, "assertions")
        assert int(conn.execute("PRAGMA user_version").fetchone()[0]) == USER_SCHEMA_VERSION
    finally:
        conn.close()

    # Re-open the existing v1 file and re-apply the tier DDL idempotently.
    conn2 = sqlite3.connect(path)
    conn2.row_factory = sqlite3.Row
    try:
        version_before = int(conn2.execute("PRAGMA user_version").fetchone()[0])
        assert version_before == USER_SCHEMA_VERSION
        # Pre-existing irreplaceable rows must survive the re-open.
        assert _table_exists(conn2, "marks")

        initialize_archive_tier(conn2, ArchiveTier.USER)

        assert _table_exists(conn2, "assertions")
        version_after = int(conn2.execute("PRAGMA user_version").fetchone()[0])
        assert version_after == USER_SCHEMA_VERSION
        assert USER_SCHEMA_VERSION == 1

        # The freshly-added table is usable.
        restored = upsert_assertion(
            conn2,
            assertion_id="post-additive",
            target_ref="session:keep-me",
            kind=AssertionKind.NOTE,
            body_text="written after additive open",
            now_ms=1_700_000_009_000,
        )
        assert restored.assertion_id == "post-additive"
    finally:
        conn2.close()


def test_recovery_digest_candidates_write_transform_candidate_assertions(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "user.db")
    try:
        digest = compile_recovery_digest(_recovery_candidate_session())
        assert digest.decision_candidates

        written = upsert_transform_candidate_assertions(
            conn,
            digest,
            now_ms=1_700_000_000_000,
        )

        mirrored = list_assertions_for_target(
            conn,
            f"session:{digest.session_id}",
            kind=AssertionKind.TRANSFORM_CANDIDATE,
        )
        assert {item.assertion_id for item in mirrored} == {item.assertion_id for item in written}
        assert len(mirrored) == len(digest.decision_candidates)

        mirrored_by_id = {assertion.assertion_id: assertion for assertion in mirrored}
        for index, candidate in enumerate(digest.decision_candidates):
            evidence_refs = [ref.to_evidence_ref().format() for ref in candidate.raw_refs]
            expected_id = assertion_id_for_transform_candidate(
                session_id=digest.session_id,
                transform_id=digest.transform.transform_id,
                transform_version=digest.transform.transform_version,
                candidate_kind=candidate.kind,
                candidate_text=candidate.text,
                evidence_refs=evidence_refs,
            )
            assertion = mirrored_by_id[expected_id]
            assert assertion.assertion_id == expected_id
            assert assertion.kind == AssertionKind.TRANSFORM_CANDIDATE
            assert assertion.scope_ref == "transform:recovery_digest_v0@v1"
            assert assertion.target_ref == "session:codex-session:assertion-demo"
            assert assertion.key == f"candidate/{candidate.kind}/{index}"
            assert assertion.value == {
                "candidate_kind": candidate.kind,
                "session_id": digest.session_id,
                "source_origin": "codex-session",
                "transform_id": "recovery_digest_v0",
                "transform_version": 1,
            }
            assert assertion.body_text == candidate.text
            assert assertion.author_ref == "transform:recovery_digest_v0@v1"
            assert assertion.author_kind == "transform"
            assert assertion.evidence_refs == evidence_refs
            assert assertion.status == "candidate"
            assert assertion.visibility == "private"
            assert assertion.context_policy == {"inject": False, "promotion_required": True}

        again = upsert_transform_candidate_assertions(
            conn,
            digest,
            now_ms=1_700_000_005_000,
        )
        assert [item.assertion_id for item in again] == [item.assertion_id for item in written]
        assert len(list_assertions_for_target(conn, f"session:{digest.session_id}")) == len(written)
        assert all(item.created_at_ms == 1_700_000_000_000 for item in again)
        assert all(item.updated_at_ms == 1_700_000_005_000 for item in again)
    finally:
        conn.close()
