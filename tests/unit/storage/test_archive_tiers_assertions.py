from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from polylogue.archive.message.messages import MessageCollection
from polylogue.archive.message.models import Message
from polylogue.archive.message.roles import Role
from polylogue.archive.session.domain_models import Session
from polylogue.core.enums import Origin
from polylogue.insights.transforms import compile_session_digest
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.index import INDEX_SCHEMA_VERSION
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.user import USER_SCHEMA_VERSION
from polylogue.storage.sqlite.archive_tiers.user_write import (
    ASSERTION_CLAIM_KINDS,
    ASSERTION_DEFAULT_AUTHOR_KIND,
    ASSERTION_DEFAULT_AUTHOR_REF,
    ASSERTION_DEFAULT_VISIBILITY,
    AssertionKind,
    AssertionStatus,
    AssertionVisibility,
    assertion_envelope_to_payload,
    assertion_id_for_candidate_judgment,
    assertion_id_for_pathology_finding,
    assertion_id_for_promoted_candidate,
    assertion_id_for_transform_candidate,
    judge_assertion_candidate,
    list_assertion_candidate_reviews,
    list_assertion_candidates,
    list_assertion_claims,
    list_assertions_for_export,
    list_assertions_for_target,
    mark_assertion_status,
    read_assertion_envelope,
    upsert_assertion,
    upsert_pathology_findings_as_assertions,
    upsert_transform_candidate_assertions,
)
from polylogue.types import SessionId


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    initialize_archive_tier(conn, ArchiveTier.USER)
    return conn


def _connect_index(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    initialize_archive_tier(conn, ArchiveTier.INDEX)
    return conn


def _sqlite_objects(conn: sqlite3.Connection, object_type: str) -> set[str]:
    return {
        str(row[0])
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = ?",
            (object_type,),
        )
    }


def _insert_index_session(conn: sqlite3.Connection, native_id: str) -> str:
    conn.execute(
        "INSERT INTO sessions (native_id, origin, content_hash) VALUES (?, ?, ?)",
        (native_id, Origin.UNKNOWN_EXPORT.value, bytes(32)),
    )
    return f"{Origin.UNKNOWN_EXPORT.value}:{native_id}"


def _insert_index_message(conn: sqlite3.Connection, session_id: str, native_id: str, position: int) -> str:
    conn.execute(
        """
        INSERT INTO messages (
            session_id, native_id, position, role, message_type, content_hash
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (session_id, native_id, position, Role.ASSISTANT.value, "message", bytes(32)),
    )
    return f"{session_id}:{native_id}"


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
        assert _table_exists(conn, "user_settings")
        assert int(conn.execute("PRAGMA user_version").fetchone()[0]) == USER_SCHEMA_VERSION
    finally:
        conn.close()


def test_fresh_user_tier_has_no_legacy_overlay_tables(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "user.db")
    try:
        tables = _sqlite_objects(conn, "table")
        obsolete_overlay_tables = {
            "session_tags",
            "session_metadata",
            "marks",
            "annotations",
            "saved_views",
            "recall_packs",
            "workspaces",
            "blackboard_notes",
            "corrections",
            "suppressions",
        }
        assert tables == {"assertions", "user_settings"}
        assert tables.isdisjoint(obsolete_overlay_tables)
    finally:
        conn.close()


def test_fresh_user_tier_has_settings_table_for_non_assertion_state(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "user.db")
    try:
        columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(user_settings)")}
        assert columns == {"setting_key", "value_json", "updated_at_ms", "author_ref"}
    finally:
        conn.close()


def test_assertions_have_read_path_indexes_for_overlay_and_candidate_flows(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "user.db")
    try:
        index_columns = {
            name: [str(row[2]) for row in conn.execute(f"PRAGMA index_info({name})")]
            for name in _sqlite_objects(conn, "index")
        }
        assert index_columns["idx_assertions_target_kind"] == ["target_ref", "kind"]
        assert index_columns["idx_assertions_kind_status_updated"] == ["kind", "status", "updated_at_ms"]
        assert index_columns["idx_assertions_target_kind_status_visibility"] == [
            "target_ref",
            "kind",
            "status",
            "visibility",
        ]
    finally:
        conn.close()


def test_actions_view_keeps_duplicate_tool_ids_session_scoped(tmp_path: Path) -> None:
    conn = _connect_index(tmp_path / "index.db")
    try:
        session_a = _insert_index_session(conn, "session-a")
        message_a = _insert_index_message(conn, session_a, "message-a", 0)
        session_b = _insert_index_session(conn, "session-b")
        message_b = _insert_index_message(conn, session_b, "message-b", 0)

        conn.execute(
            """
            INSERT INTO blocks (
                message_id, session_id, position, block_type, tool_name, tool_id, tool_input, semantic_type
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                message_a,
                session_a,
                0,
                "tool_use",
                "Bash",
                "provider-local-tool-id",
                json.dumps({"command": "pytest -q"}),
                "shell",
            ),
        )
        conn.execute(
            """
            INSERT INTO blocks (
                message_id, session_id, position, block_type, text, tool_id
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                message_b,
                session_b,
                0,
                "tool_result",
                "wrong-session-result",
                "provider-local-tool-id",
            ),
        )

        action = conn.execute(
            """
            SELECT output_text, tool_result_block_id
            FROM actions
            WHERE session_id = ?
            """,
            (session_a,),
        ).fetchone()
        assert action is not None
        assert action["output_text"] is None
        assert action["tool_result_block_id"] is None

        conn.execute(
            """
            INSERT INTO blocks (
                message_id, session_id, position, block_type, text, tool_id
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                message_a,
                session_a,
                1,
                "tool_result",
                "same-session-result",
                "provider-local-tool-id",
            ),
        )
        action = conn.execute(
            """
            SELECT output_text, tool_result_block_id
            FROM actions
            WHERE session_id = ?
            """,
            (session_a,),
        ).fetchone()
        assert action is not None
        assert action["output_text"] == "same-session-result"
        assert action["tool_result_block_id"] == f"{message_a}:1"
    finally:
        conn.close()


def test_index_json_contracts_reject_non_object_payloads(tmp_path: Path) -> None:
    conn = _connect_index(tmp_path / "index.db")
    try:
        assert int(conn.execute("PRAGMA user_version").fetchone()[0]) == INDEX_SCHEMA_VERSION
        session_id = _insert_index_session(conn, "json-contract")
        message_id = _insert_index_message(conn, session_id, "message-json", 0)

        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """
                INSERT INTO blocks (
                    message_id, session_id, position, block_type, tool_name, tool_id, tool_input
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (message_id, session_id, 0, "tool_use", "Bash", "tool-json", "[]"),
            )

        conn.execute(
            """
            INSERT INTO blocks (
                message_id, session_id, position, block_type, tool_name, tool_id, tool_input
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (message_id, session_id, 0, "tool_use", "Bash", "tool-json", json.dumps({"command": "true"})),
        )

        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """
                INSERT INTO session_provider_usage_events (
                    session_id, source_message_id, position, provider_event_type, payload_json
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (session_id, message_id, 0, "token_count", "[]"),
            )

        conn.execute(
            """
            INSERT INTO session_provider_usage_events (
                session_id, source_message_id, position, provider_event_type, payload_json
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (session_id, message_id, 0, "token_count", json.dumps({"total_tokens": 3})),
        )
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
        # Any non-"user" author_kind -- "human" here, not just "agent" -- is
        # coerced to a non-injected candidate (37t.15 chokepoint).
        assert read_mark.status == "candidate"
        assert read_mark.visibility == ASSERTION_DEFAULT_VISIBILITY
        assert read_mark.context_policy == {"inject": False, "promotion_required": True}
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


def test_assertion_defaults_are_explicit_private_no_inject(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "user.db")
    try:
        active = upsert_assertion(
            conn,
            assertion_id="default-decision",
            target_ref="session:s-1",
            kind=AssertionKind.DECISION,
            body_text="default lifecycle should be explicit",
            now_ms=1_700_000_000_000,
        )
        candidate_policy = upsert_assertion(
            conn,
            assertion_id="candidate-policy",
            target_ref="session:s-1",
            kind=AssertionKind.TRANSFORM_CANDIDATE,
            body_text="candidate keeps promotion flag",
            context_policy={"promotion_required": True},
            status="candidate",
            now_ms=1_700_000_001_000,
        )

        assert active.status == "active"
        assert active.visibility == "private"
        assert active.author_ref == ASSERTION_DEFAULT_AUTHOR_REF
        assert active.author_kind == ASSERTION_DEFAULT_AUTHOR_KIND
        assert active.context_policy == {"inject": False}
        assert candidate_policy.context_policy == {"inject": False, "promotion_required": True}

        active_claims = list_assertion_claims(conn, target_ref="session:s-1", statuses=("active",))
        assert [claim.assertion_id for claim in active_claims] == ["default-decision"]

        active_exports = list_assertions_for_export(conn, statuses=("active",))
        assert [row.assertion_id for row in active_exports] == ["default-decision"]
    finally:
        conn.close()


def test_legacy_null_lifecycle_assertions_read_as_active_private_no_inject(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "user.db")
    try:
        conn.execute(
            """
            INSERT INTO assertions (
                assertion_id, target_ref, kind, body_text,
                author_ref, author_kind, status, visibility, context_policy_json,
                created_at_ms, updated_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "legacy-null",
                "session:s-legacy",
                AssertionKind.DECISION.value,
                "legacy row before lifecycle defaults",
                None,
                None,
                None,
                None,
                None,
                1_700_000_000_000,
                1_700_000_000_000,
            ),
        )

        stored = read_assertion_envelope(conn, "legacy-null")
        assert stored is not None
        assert stored.status == "active"
        assert stored.visibility == "private"
        assert stored.author_ref == ASSERTION_DEFAULT_AUTHOR_REF
        assert stored.author_kind == ASSERTION_DEFAULT_AUTHOR_KIND
        assert stored.context_policy == {"inject": False}

        assert [claim.assertion_id for claim in list_assertion_claims(conn, statuses=("active",))] == ["legacy-null"]
        assert [row.assertion_id for row in list_assertions_for_export(conn, statuses=("active",))] == ["legacy-null"]
    finally:
        conn.close()


def test_assertion_write_rejects_unparseable_public_refs(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "user.db")
    try:
        with pytest.raises(ValueError, match="object ref must use"):
            upsert_assertion(
                conn,
                assertion_id="bad-target",
                target_ref="bare-target-id",
                kind=AssertionKind.DECISION,
                now_ms=1_700_000_000_000,
            )
        with pytest.raises(ValueError, match="unsupported public ref"):
            upsert_assertion(
                conn,
                assertion_id="bad-evidence",
                target_ref="session:session-1",
                kind=AssertionKind.DECISION,
                evidence_refs=("session-1::message-1::not-a-block-index",),
                now_ms=1_700_000_000_000,
            )
    finally:
        conn.close()


def test_assertion_write_rejects_unknown_lifecycle_values(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "user.db")
    try:
        with pytest.raises(ValueError):
            upsert_assertion(
                conn,
                assertion_id="bad-kind",
                target_ref="session:session-1",
                kind="review",
                now_ms=1_700_000_000_000,
            )
        with pytest.raises(ValueError):
            upsert_assertion(
                conn,
                assertion_id="bad-status",
                target_ref="session:session-1",
                kind=AssertionKind.DECISION,
                status="draft",
                now_ms=1_700_000_000_000,
            )
        with pytest.raises(ValueError):
            upsert_assertion(
                conn,
                assertion_id="bad-visibility",
                target_ref="session:session-1",
                kind=AssertionKind.DECISION,
                visibility="workspace",
                now_ms=1_700_000_000_000,
            )
    finally:
        conn.close()


def test_assertion_write_normalizes_json_values_at_internal_boundary(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "user.db")
    try:
        with pytest.raises(TypeError, match="assertion value is not JSON-compatible"):
            upsert_assertion(
                conn,
                assertion_id="bad-value",
                target_ref="session:session-1",
                kind=AssertionKind.DECISION,
                value={"bad": object()},
                now_ms=1_700_000_000_000,
            )
        with pytest.raises(TypeError, match="assertion staleness is not a JSON object"):
            upsert_assertion(
                conn,
                assertion_id="bad-staleness",
                target_ref="session:session-1",
                kind=AssertionKind.DECISION,
                staleness={"bad": object()},
                now_ms=1_700_000_000_000,
            )
        with pytest.raises(TypeError, match="assertion context_policy is not a JSON object"):
            upsert_assertion(
                conn,
                assertion_id="bad-context",
                target_ref="session:session-1",
                kind=AssertionKind.DECISION,
                context_policy={"inject": object()},
                now_ms=1_700_000_000_000,
            )

        stored = upsert_assertion(
            conn,
            assertion_id="typed-values",
            target_ref="session:session-1",
            kind=AssertionKind.DECISION,
            status=AssertionStatus.ACTIVE,
            visibility=AssertionVisibility.PRIVATE,
            value={"ok": [1, True, None]},
            staleness={"ttl_days": 3},
            context_policy={"reason": "operator"},
            now_ms=1_700_000_000_000,
        )

        assert stored.value == {"ok": [1, True, None]}
        assert stored.status is AssertionStatus.ACTIVE
        assert stored.visibility is AssertionVisibility.PRIVATE
        assert stored.staleness == {"ttl_days": 3}
        assert stored.context_policy == {"inject": False, "reason": "operator"}
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
            status=AssertionStatus.CANDIDATE,
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


def test_list_assertion_claims_filters_lifecycle_assertions(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "user.db")
    try:
        assert set(ASSERTION_CLAIM_KINDS) == {
            AssertionKind.DECISION,
            AssertionKind.CAVEAT,
            AssertionKind.BLOCKER,
            AssertionKind.LESSON,
            AssertionKind.RUN_STATE,
            AssertionKind.TRANSFORM_CANDIDATE,
            AssertionKind.PATHOLOGY,
        }

        rows: list[tuple[str, str, AssertionKind, str, str, dict[str, object] | None, int]] = [
            (
                "claim-decision",
                "session:s-1",
                AssertionKind.DECISION,
                "run:r-1",
                "active",
                {"inject": True},
                1_700_000_001_000,
            ),
            (
                "claim-caveat",
                "session:s-1",
                AssertionKind.CAVEAT,
                "run:r-1",
                "active",
                {"inject": False},
                1_700_000_002_000,
            ),
            (
                "claim-transform",
                "session:s-1",
                AssertionKind.TRANSFORM_CANDIDATE,
                "run:r-2",
                "candidate",
                None,
                1_700_000_003_000,
            ),
            (
                "claim-blocker-deleted",
                "session:s-1",
                AssertionKind.BLOCKER,
                "run:r-1",
                "deleted",
                {"inject": True},
                1_700_000_004_000,
            ),
            (
                "claim-lesson-other",
                "session:s-2",
                AssertionKind.LESSON,
                "run:r-1",
                "active",
                {"inject": True},
                1_700_000_005_000,
            ),
            (
                "overlay-mark",
                "session:s-1",
                AssertionKind.MARK,
                "run:r-1",
                "active",
                {"inject": True},
                1_700_000_006_000,
            ),
        ]
        for assertion_id, target_ref, kind, scope_ref, status, context_policy, now_ms in rows:
            upsert_assertion(
                conn,
                assertion_id=assertion_id,
                target_ref=target_ref,
                kind=kind,
                scope_ref=scope_ref,
                status=status,
                context_policy=context_policy,
                now_ms=now_ms,
            )

        target_claims = list_assertion_claims(conn, target_ref="session:s-1")
        assert [claim.assertion_id for claim in target_claims] == [
            "claim-transform",
            "claim-caveat",
            "claim-decision",
        ]

        scoped_claims = list_assertion_claims(conn, scope_ref="run:r-1", context_inject=True)
        assert [claim.assertion_id for claim in scoped_claims] == ["claim-lesson-other", "claim-decision"]

        private_target_claims = list_assertion_claims(conn, target_ref="session:s-1", context_inject=False)
        assert [claim.assertion_id for claim in private_target_claims] == ["claim-transform", "claim-caveat"]

        deleted_blockers = list_assertion_claims(
            conn,
            kinds=(AssertionKind.BLOCKER,),
            statuses=("deleted",),
        )
        assert [claim.assertion_id for claim in deleted_blockers] == ["claim-blocker-deleted"]

        assert list_assertion_claims(conn, kinds=()) == []
        assert list_assertion_claims(conn, statuses=()) == []
        assert [claim.assertion_id for claim in list_assertion_claims(conn, limit=1)] == ["claim-lesson-other"]
    finally:
        conn.close()


def test_list_assertions_for_export_covers_all_kinds_and_statuses(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "user.db")
    try:
        rows = [
            ("a-mark", AssertionKind.MARK, "active", 1_700_000_001_000),
            ("a-deleted-note", AssertionKind.NOTE, "deleted", 1_700_000_002_000),
            ("a-candidate", AssertionKind.TRANSFORM_CANDIDATE, "candidate", 1_700_000_003_000),
        ]
        for assertion_id, kind, status, now_ms in rows:
            upsert_assertion(
                conn,
                assertion_id=assertion_id,
                target_ref="session:s-1",
                kind=kind,
                scope_ref="run:r-1",
                key=f"export/{assertion_id}",
                value={"status": status},
                body_text=f"body {assertion_id}",
                author_ref="user:operator",
                author_kind="user",
                evidence_refs=[f"message:s-1:{now_ms}"],
                status=status,
                visibility="private",
                confidence=0.75,
                staleness={"stale": False},
                context_policy={"inject": status == "active"},
                supersedes=["old:a"] if assertion_id == "a-mark" else None,
                now_ms=now_ms,
            )

        exported = list_assertions_for_export(conn)
        assert [row.assertion_id for row in exported] == ["a-mark", "a-deleted-note", "a-candidate"]

        active = list_assertions_for_export(conn, statuses=("active",))
        assert [row.assertion_id for row in active] == ["a-mark"]

        notes = list_assertions_for_export(conn, kinds=(AssertionKind.NOTE,))
        assert [row.assertion_id for row in notes] == ["a-deleted-note"]

        limited = list_assertions_for_export(conn, limit=2)
        assert [row.assertion_id for row in limited] == ["a-mark", "a-deleted-note"]

        payload = assertion_envelope_to_payload(exported[0])
        assert payload == {
            "assertion_id": "a-mark",
            "scope_ref": "run:r-1",
            "target_ref": "session:s-1",
            "key": "export/a-mark",
            "kind": "mark",
            "value": {"status": "active"},
            "body_text": "body a-mark",
            "author_ref": "user:operator",
            "author_kind": "user",
            "evidence_refs": ["message:s-1:1700000001000"],
            "status": "active",
            "visibility": "private",
            "confidence": 0.75,
            "staleness": {"stale": False},
            "context_policy": {"inject": True},
            "supersedes": ["old:a"],
            "created_at_ms": 1_700_000_001_000,
            "updated_at_ms": 1_700_000_001_000,
        }
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


def test_session_digest_candidates_write_transform_candidate_assertions(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "user.db")
    try:
        digest = compile_session_digest(_recovery_candidate_session())
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
                candidate_index=index,
                candidate_kind=candidate.kind,
                candidate_text=candidate.text,
                evidence_refs=evidence_refs,
            )
            assertion = mirrored_by_id[expected_id]
            assert assertion.assertion_id == expected_id
            assert assertion.kind == AssertionKind.TRANSFORM_CANDIDATE
            assert assertion.scope_ref == "transform:session_digest_v0@v1"
            assert assertion.target_ref == "session:codex-session:assertion-demo"
            assert assertion.key == f"candidate/{candidate.kind}/{index}"
            assert assertion.value == {
                "candidate_index": index,
                "candidate_kind": candidate.kind,
                "session_id": digest.session_id,
                "source_origin": "codex-session",
                "transform_id": "session_digest_v0",
                "transform_version": 1,
            }
            assert assertion.body_text == candidate.text
            assert assertion.author_ref == "transform:session_digest_v0@v1"
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

        accepted_id = written[0].assertion_id
        assert mark_assertion_status(conn, accepted_id, "accepted", now_ms=1_700_000_006_000)

        after_accept = upsert_transform_candidate_assertions(
            conn,
            digest,
            now_ms=1_700_000_007_000,
        )
        accepted_after_remirror = next(item for item in after_accept if item.assertion_id == accepted_id)
        assert accepted_after_remirror.status is AssertionStatus.ACCEPTED
        assert accepted_after_remirror.updated_at_ms == 1_700_000_006_000

        duplicate_digest = digest.model_copy(
            update={"decision_candidates": (digest.decision_candidates[0], digest.decision_candidates[0])}
        )
        duplicate_written = upsert_transform_candidate_assertions(
            conn,
            duplicate_digest,
            now_ms=1_700_000_008_000,
        )
        assert len({item.assertion_id for item in duplicate_written}) == 2
        assert {item.value["candidate_index"] for item in duplicate_written if isinstance(item.value, dict)} == {0, 1}
    finally:
        conn.close()


def test_candidate_assertion_acceptance_creates_active_assertion_with_lineage(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "user.db")
    try:
        digest = compile_session_digest(_recovery_candidate_session())
        candidate = upsert_transform_candidate_assertions(conn, digest, now_ms=1_700_000_000_000)[0]

        result = judge_assertion_candidate(
            conn,
            candidate_ref=f"assertion:{candidate.assertion_id}",
            decision="accept",
            reason="looks correct",
            actor_ref="user:local",
            now_ms=1_700_000_010_000,
        )

        assert result.candidate.assertion_id == candidate.assertion_id
        assert result.candidate.status is AssertionStatus.ACCEPTED
        assert result.resulting_assertion is not None
        assert result.resulting_assertion.assertion_id == assertion_id_for_promoted_candidate(candidate.assertion_id)
        assert isinstance(candidate.value, dict)
        assert result.resulting_assertion.kind == candidate.value["candidate_kind"]
        assert result.resulting_assertion.status is AssertionStatus.ACTIVE
        assert result.resulting_assertion.supersedes == [f"assertion:{candidate.assertion_id}"]
        assert result.resulting_assertion.context_policy == {"inject": False}
        assert result.judgment.assertion_id == assertion_id_for_candidate_judgment(candidate.assertion_id, "accept")
        assert result.judgment.kind == AssertionKind.JUDGMENT
        assert result.judgment.target_ref == f"assertion:{candidate.assertion_id}"
        assert result.judgment.value == {
            "decision": "accept",
            "candidate_ref": f"assertion:{candidate.assertion_id}",
            "reason": "looks correct",
            "resulting_assertion_ref": f"assertion:{result.resulting_assertion.assertion_id}",
        }
    finally:
        conn.close()


def test_candidate_assertion_rejection_preserves_reason_and_filtering(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "user.db")
    try:
        digest = compile_session_digest(_recovery_candidate_session())
        candidate = upsert_transform_candidate_assertions(conn, digest, now_ms=1_700_000_000_000)[0]

        result = judge_assertion_candidate(
            conn,
            candidate_ref=candidate.assertion_id,
            decision="reject",
            reason="unsupported by transcript",
            actor_ref="user:local",
            now_ms=1_700_000_010_000,
        )

        assert result.resulting_assertion is None
        assert result.candidate.status is AssertionStatus.REJECTED
        assert result.judgment.value == {
            "decision": "reject",
            "candidate_ref": f"assertion:{candidate.assertion_id}",
            "reason": "unsupported by transcript",
            "resulting_assertion_ref": None,
        }
        assert candidate.assertion_id not in {item.assertion_id for item in list_assertion_candidates(conn)}
        rejected = list_assertion_claims(conn, statuses=("rejected",))
        assert [item.assertion_id for item in rejected] == [candidate.assertion_id]
    finally:
        conn.close()


def test_candidate_assertion_defer_records_reason_without_promoting(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "user.db")
    try:
        digest = compile_session_digest(_recovery_candidate_session())
        candidate = upsert_transform_candidate_assertions(conn, digest, now_ms=1_700_000_000_000)[0]

        result = judge_assertion_candidate(
            conn,
            candidate_ref=f"assertion:{candidate.assertion_id}",
            decision="defer",
            reason="needs another source",
            actor_ref="user:local",
            now_ms=1_700_000_010_000,
        )

        assert result.resulting_assertion is None
        assert result.candidate.status is AssertionStatus.DEFERRED
        assert candidate.assertion_id not in {item.assertion_id for item in list_assertion_candidates(conn)}
        assert result.judgment.value == {
            "decision": "defer",
            "candidate_ref": f"assertion:{candidate.assertion_id}",
            "reason": "needs another source",
            "resulting_assertion_ref": None,
        }
        assert result.judgment.evidence_refs == [
            *candidate.evidence_refs,
            f"assertion:{candidate.assertion_id}",
        ]
        review_rows = list_assertion_candidate_reviews(conn, statuses=(AssertionStatus.DEFERRED,))
        assert [(row.candidate.assertion_id, row.candidate.status) for row in review_rows] == [
            (candidate.assertion_id, AssertionStatus.DEFERRED)
        ]
        assert review_rows[0].latest_judgment is not None
        assert review_rows[0].latest_judgment.assertion_id == result.judgment.assertion_id
    finally:
        conn.close()


def test_candidate_assertion_supersede_records_replacement_and_lineage(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "user.db")
    try:
        digest = compile_session_digest(_recovery_candidate_session())
        candidate = upsert_transform_candidate_assertions(conn, digest, now_ms=1_700_000_000_000)[0]

        result = judge_assertion_candidate(
            conn,
            candidate_ref=f"assertion:{candidate.assertion_id}",
            decision="supersede",
            reason="replacement is more precise",
            actor_ref="user:local",
            replacement_kind=AssertionKind.DECISION,
            replacement_body_text="Accepted replacement decision",
            replacement_value={"source": "operator"},
            now_ms=1_700_000_010_000,
        )

        assert result.candidate.status is AssertionStatus.SUPERSEDED
        assert result.resulting_assertion is not None
        assert result.resulting_assertion.assertion_id == assertion_id_for_promoted_candidate(candidate.assertion_id)
        assert result.resulting_assertion.kind is AssertionKind.DECISION
        assert result.resulting_assertion.body_text == "Accepted replacement decision"
        assert result.resulting_assertion.value == {"source": "operator"}
        assert result.resulting_assertion.status is AssertionStatus.ACTIVE
        assert result.resulting_assertion.supersedes == [f"assertion:{candidate.assertion_id}"]
        assert result.judgment.value == {
            "decision": "supersede",
            "candidate_ref": f"assertion:{candidate.assertion_id}",
            "reason": "replacement is more precise",
            "resulting_assertion_ref": f"assertion:{result.resulting_assertion.assertion_id}",
        }
    finally:
        conn.close()


def test_candidate_reviews_survive_remirror_without_becoming_authoritative(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "user.db")
    try:
        digest = compile_session_digest(_recovery_candidate_session())
        duplicate_digest = digest.model_copy(update={"decision_candidates": (digest.decision_candidates[0],) * 3})
        candidates = upsert_transform_candidate_assertions(
            conn,
            duplicate_digest,
            now_ms=1_700_000_000_000,
        )
        upsert_assertion(
            conn,
            assertion_id="durable-user-decision",
            target_ref=f"session:{digest.session_id}",
            kind=AssertionKind.DECISION,
            body_text="This is already durable user-authored state.",
            status=AssertionStatus.ACTIVE,
            now_ms=1_700_000_001_000,
        )

        accepted = judge_assertion_candidate(
            conn,
            candidate_ref=f"assertion:{candidates[0].assertion_id}",
            decision="accept",
            reason="confirmed",
            actor_ref="user:local",
            now_ms=1_700_000_010_000,
        )
        rejected = judge_assertion_candidate(
            conn,
            candidate_ref=f"assertion:{candidates[1].assertion_id}",
            decision="reject",
            reason="unsupported",
            actor_ref="user:local",
            now_ms=1_700_000_011_000,
        )
        deferred = judge_assertion_candidate(
            conn,
            candidate_ref=f"assertion:{candidates[2].assertion_id}",
            decision="defer",
            reason="needs another source",
            actor_ref="user:local",
            now_ms=1_700_000_012_000,
        )

        remirrored = upsert_transform_candidate_assertions(
            conn,
            duplicate_digest,
            now_ms=1_700_000_020_000,
        )
        statuses_by_id = {item.assertion_id: item.status for item in remirrored}
        assert statuses_by_id == {
            candidates[0].assertion_id: AssertionStatus.ACCEPTED,
            candidates[1].assertion_id: AssertionStatus.REJECTED,
            candidates[2].assertion_id: AssertionStatus.DEFERRED,
        }
        assert all(item.updated_at_ms < 1_700_000_020_000 for item in remirrored)
        assert list_assertion_candidates(conn) == []

        assert accepted.resulting_assertion is not None
        active_claims = list_assertion_claims(conn, target_ref=f"session:{digest.session_id}", statuses=("active",))
        assert {claim.assertion_id for claim in active_claims} == {
            "durable-user-decision",
            accepted.resulting_assertion.assertion_id,
        }

        reviews = list_assertion_candidate_reviews(conn, target_ref=f"session:{digest.session_id}")
        assert [row.candidate.assertion_id for row in reviews] == [
            deferred.candidate.assertion_id,
            rejected.candidate.assertion_id,
            accepted.candidate.assertion_id,
        ]
        assert {row.candidate.status for row in reviews} == {
            AssertionStatus.ACCEPTED,
            AssertionStatus.REJECTED,
            AssertionStatus.DEFERRED,
        }
        assert "durable-user-decision" not in {row.candidate.assertion_id for row in reviews}
        assert all(row.latest_judgment is not None for row in reviews)
    finally:
        conn.close()


def test_upsert_pathology_findings_emits_queryable_candidates(tmp_path: Path) -> None:
    """Pathology findings become PATHOLOGY candidate claims with evidence (#2383)."""
    from polylogue.core.refs import EvidenceRef
    from polylogue.insights.pathology import PathologyFinding

    conn = _connect(tmp_path / "user.db")
    try:
        session_id = "codex-session:pathology-demo"
        findings = [
            PathologyFinding(
                kind="wasted_loop",
                session_id=session_id,
                severity="medium",
                detail="5 failed test/check turns without a clean pass between them",
                occurrence_count=5,
                evidence_refs=(EvidenceRef(session_id=session_id, message_id="m7"),),
            ),
            PathologyFinding(
                kind="stale_context",
                session_id=session_id,
                severity="medium",
                detail="1 resume boundary(ies) re-inherited context in lossy mode(s): summary",
                occurrence_count=1,
                evidence_refs=(EvidenceRef(session_id=session_id, message_id="m1"),),
            ),
        ]

        envelopes = upsert_pathology_findings_as_assertions(conn, session_id, findings)
        conn.commit()
        assert len(envelopes) == 2
        for envelope in envelopes:
            assert envelope.kind == AssertionKind.PATHOLOGY
            assert envelope.status == AssertionStatus.CANDIDATE
            assert envelope.visibility == AssertionVisibility.PRIVATE
            assert envelope.context_policy.get("inject") is False
            assert envelope.evidence_refs  # drillable

        # Queryable via the standard assertion-claims surface (#2006).
        claims = list_assertion_claims(
            conn,
            kinds=(AssertionKind.PATHOLOGY,),
            statuses=(AssertionStatus.CANDIDATE,),
        )
        claim_kinds = {claim.value["pathology_kind"] for claim in claims if isinstance(claim.value, dict)}
        assert claim_kinds == {"wasted_loop", "stale_context"}

        # Deterministic id: re-emitting identical findings is idempotent.
        again = upsert_pathology_findings_as_assertions(conn, session_id, findings)
        conn.commit()
        assert {e.assertion_id for e in again} == {e.assertion_id for e in envelopes}
        recount = list_assertion_claims(conn, kinds=(AssertionKind.PATHOLOGY,))
        assert len(recount) == 2

        # An operator-promoted candidate is never downgraded back to candidate.
        promoted_id = envelopes[0].assertion_id
        mark_assertion_status(conn, promoted_id, AssertionStatus.ACCEPTED)
        conn.commit()
        preserved = upsert_pathology_findings_as_assertions(conn, session_id, findings)
        conn.commit()
        promoted = read_assertion_envelope(conn, promoted_id)
        assert promoted is not None
        assert promoted.status == AssertionStatus.ACCEPTED
        assert any(e.assertion_id == promoted_id for e in preserved)
    finally:
        conn.close()


def test_assertion_id_for_pathology_finding_is_stable() -> None:
    """The deterministic id changes only when finding identity changes (#2383)."""
    base = {
        "session_id": "codex-session:x",
        "finding_kind": "wasted_loop",
        "detector_version": 1,
        "finding_detail": "3 failed test/check turns",
        "evidence_refs": ["codex-session:x::m1"],
    }
    first = assertion_id_for_pathology_finding(**base)  # type: ignore[arg-type]
    assert first == assertion_id_for_pathology_finding(**base)  # type: ignore[arg-type]
    changed = assertion_id_for_pathology_finding(**{**base, "finding_kind": "stale_context"})  # type: ignore[arg-type]
    assert changed != first
