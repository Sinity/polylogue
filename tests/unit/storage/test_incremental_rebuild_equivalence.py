"""One source-grounded survivor for incremental/restart/rebuild equivalence.

Production dependencies exercised here:

* ``ArchiveStore.write_raw_payload`` commits durable source evidence.
* ``backfill_historical_revision_evidence`` performs targeted cohort expansion,
  typed revision selection, and parsed/index replacement.
* ``repair_session_insights`` materializes the public insight surfaces.
* ``rebuild_index_from_source`` replays retained evidence into an owned inactive
  ``IndexGenerationStore`` generation before atomic promotion.

The test deliberately plants a same-row-count stale FTS row and a stale profile
stamp. Removing full-session replacement from revision replay leaves the stale
search token behind; omitting the insight stage from rebuild leaves profile and
materialization rows absent. Either representative mutation must fail this
survivor.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import pytest

from polylogue.config import Config
from polylogue.core.enums import Provider
from polylogue.maintenance.replay import rebuild_index_from_source
from polylogue.sources.revision_backfill import backfill_historical_revision_evidence
from polylogue.storage.index_generation import IndexGenerationStore, source_revision_snapshot
from polylogue.storage.repair import repair_session_insights
from polylogue.storage.runtime import SESSION_INSIGHT_MATERIALIZER_VERSION
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from polylogue.storage.sqlite.archive_tiers.index import INDEX_SCHEMA_VERSION

CHAT_SESSION = "chatgpt-export:branch-canary"
PARENT_SESSION = "codex-session:lineage-parent"
CHILD_SESSION = "codex-session:lineage-child"
CHAT_USER = f"{CHAT_SESSION}:u1"
CHAT_OLD = f"{CHAT_SESSION}:a-old"
CHAT_NEW = f"{CHAT_SESSION}:a-new"
CHAT_NEW_BLOCK = f"{CHAT_NEW}:0"
PARENT_FIRST = f"{PARENT_SESSION}:lineage-parent-m0"
PARENT_SECOND = f"{PARENT_SESSION}:lineage-parent-m1"
CHILD_FIRST = f"{CHILD_SESSION}:lineage-child-m0"
CANONICAL_TOKEN = "quartzneedle"
STALE_TOKEN = "staleonlytoken"
OVERLAY_TAG = "operator-canary"
PARSER_RECIPE = "revision-membership-v1"

# Volatile attempt timestamps are not canonical derivation output. The exact
# semantic stamps beside them remain compared.
_VOLATILE_COLUMNS: dict[str, frozenset[str]] = {
    "session_links": frozenset({"observed_at_ms", "resolved_at_ms"}),
    "raw_revision_heads": frozenset({"decided_at_ms"}),
    "session_profiles": frozenset({"materialized_at", "priced_at_ms"}),
    "session_latency_profiles": frozenset({"materialized_at"}),
    "session_tag_rollups": frozenset({"materialized_at"}),
    "insight_materialization": frozenset({"materialized_at_ms"}),
    "threads": frozenset({"materialized_at"}),
}

# These are the independently selected durable/public facts for this canary.
# FTS is compared through the public search method, not through private FTS5
# segment tables.
_CANONICAL_TABLES = (
    "sessions",
    "messages",
    "blocks",
    "attachments",
    "attachment_refs",
    "attachment_native_ids",
    "session_links",
    "raw_revision_heads",
    "session_profiles",
    "session_latency_profiles",
    "session_work_events",
    "session_phases",
    "session_model_usage",
    "session_provider_usage_events",
    "session_tag_rollups",
    "insight_materialization",
    "threads",
    "thread_sessions",
)


@dataclass(frozen=True, slots=True)
class RawIds:
    chat: str
    parent_v1: str
    parent_v2: str
    child: str

    def all(self) -> tuple[str, ...]:
        return (self.chat, self.parent_v1, self.parent_v2, self.child)


@dataclass(frozen=True, slots=True)
class CanonicalFacts:
    tables: tuple[tuple[str, tuple[str, ...], tuple[tuple[Any, ...], ...]], ...]
    current_revision_applications: tuple[tuple[Any, ...], ...]
    search_hits: tuple[tuple[str, tuple[str, ...]], ...]

    def digest(self) -> str:
        payload = json.dumps(_normalize(self), sort_keys=True, separators=(",", ":"))
        return sha256(payload.encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class DerivationKeyWitness:
    """The non-attempt identity required by architecture/05-derived-freshness."""

    subject_source_bindings: tuple[tuple[Any, ...], ...]
    source_snapshot: str
    source_evidence: tuple[tuple[Any, ...], ...]
    parser_census: tuple[tuple[Any, ...], ...]
    recipe_identity: tuple[Any, ...]
    output_contract: tuple[tuple[str, tuple[tuple[Any, ...], ...]], ...]


@dataclass(frozen=True, slots=True)
class AttemptReceipt:
    route: str
    index_path: str
    index_inode: int
    generation_id: str | None
    owner_id: str | None
    generation_state: str
    revision_application_ids: tuple[str, ...]


def _chatgpt_branch_payload() -> bytes:
    def node(
        native_id: str,
        role: str,
        text: str,
        *,
        parent: str | None = None,
        children: tuple[str, ...] = (),
        metadata: dict[str, object] | None = None,
        created_at: int,
    ) -> dict[str, object]:
        message: dict[str, object] = {
            "id": native_id,
            "author": {"role": role},
            "content": {"content_type": "text", "parts": [text]},
            "create_time": created_at,
        }
        if metadata is not None:
            message["metadata"] = metadata
        return {
            "id": native_id,
            "parent": parent,
            "children": list(children),
            "message": message,
        }

    rows = (
        node("root", "system", "", children=("u1",), created_at=0),
        node(
            "u1",
            "user",
            "inspect attachment",
            parent="root",
            children=("a-old", "a-new"),
            metadata={"attachments": [{"id": "att-canary", "name": "canary.dat"}]},
            created_at=1,
        ),
        node(
            "a-old",
            "assistant",
            "old branch remains visible",
            parent="u1",
            created_at=2,
        ),
        node(
            "a-new",
            "assistant",
            f"{CANONICAL_TOKEN} canonical answer",
            parent="u1",
            created_at=3,
        ),
    )
    payload = {
        "id": "branch-canary",
        "conversation_id": "branch-canary",
        "title": "Branch canary",
        "create_time": 1_700_000_000,
        "update_time": 1_700_000_003,
        "current_node": "a-new",
        "mapping": {str(row["id"]): row for row in rows},
    }
    return json.dumps([payload], sort_keys=True).encode()


def _codex_session(
    native_id: str,
    messages: tuple[tuple[str, str], ...],
    *,
    parent_native_id: str | None = None,
) -> bytes:
    rows: list[dict[str, object]] = [
        {
            "type": "session_meta",
            "payload": {"id": native_id, "timestamp": "2026-07-16T10:00:00Z"},
        }
    ]
    if parent_native_id is not None:
        rows.append(
            {
                "type": "session_meta",
                "payload": {
                    "id": parent_native_id,
                    "timestamp": "2026-07-16T09:00:00Z",
                },
            }
        )
    for position, (role, text) in enumerate(messages):
        rows.append(
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "id": f"{native_id}-m{position}",
                    "role": role,
                    "content": [
                        {
                            "type": "input_text" if role == "user" else "output_text",
                            "text": text,
                        }
                    ],
                },
            }
        )
    return b"".join(json.dumps(row, sort_keys=True).encode() + b"\n" for row in rows)


def _config(root: Path, *, index_path: Path | None = None) -> Config:
    return Config(
        archive_root=root,
        render_root=root / "render",
        sources=[],
        db_path=index_path or root / "index.db",
    )


def _connect(path: Path, *, read_only: bool = True) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True) if read_only else sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def _normalize(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, CanonicalFacts):
        return {
            "tables": _normalize(value.tables),
            "current_revision_applications": _normalize(value.current_revision_applications),
            "search_hits": _normalize(value.search_hits),
        }
    if isinstance(value, tuple):
        return [_normalize(item) for item in value]
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _normalize(item) for key, item in value.items()}
    return value


def _schema_columns(conn: sqlite3.Connection, table: str) -> tuple[tuple[Any, ...], ...]:
    return tuple(
        (row["name"], row["type"], row["notnull"], row["dflt_value"], row["pk"], row["hidden"])
        for row in conn.execute(f'PRAGMA table_xinfo("{table}")')
    )


def _table_rows(
    conn: sqlite3.Connection,
    table: str,
) -> tuple[tuple[str, ...], tuple[tuple[Any, ...], ...]]:
    excluded = _VOLATILE_COLUMNS.get(table, frozenset())
    columns = tuple(
        row["name"] for row in conn.execute(f'PRAGMA table_xinfo("{table}")') if row["name"] not in excluded
    )
    quoted = ", ".join(f'"{column}"' for column in columns)
    rows = tuple(
        sorted(
            (tuple(_normalize(value) for value in row) for row in conn.execute(f'SELECT {quoted} FROM "{table}"')),
            key=repr,
        )
    )
    return columns, rows


def _current_revision_applications(conn: sqlite3.Connection) -> tuple[tuple[Any, ...], ...]:
    rows = conn.execute(
        """
        WITH ranked AS (
            SELECT
                *,
                ROW_NUMBER() OVER (
                    PARTITION BY raw_id, session_id
                    ORDER BY acquisition_generation DESC, decided_at_ms DESC, decision_id DESC
                ) AS attempt_rank
            FROM raw_revision_applications
        )
        SELECT
            decision_id,
            raw_id,
            session_id,
            logical_source_key,
            source_revision,
            acquisition_generation,
            decision,
            accepted_raw_id,
            accepted_source_revision,
            accepted_content_hash,
            baseline_raw_id,
            predecessor_raw_id,
            append_end_offset,
            detail
        FROM ranked
        WHERE attempt_rank = 1
        ORDER BY raw_id, session_id
        """
    )
    return tuple(tuple(_normalize(value) for value in row) for row in rows)


def _collect_canonical_facts(route_root: Path, index_path: Path) -> CanonicalFacts:
    with _connect(index_path) as conn:
        tables = tuple((table, *_table_rows(conn, table)) for table in _CANONICAL_TABLES)
        applications = _current_revision_applications(conn)
    with ArchiveStore.open_existing(route_root, read_only=True) as archive:
        searches = tuple(
            (query, tuple(archive.search_blocks(query)))
            for query in (CANONICAL_TOKEN, STALE_TOKEN, "definitelyabsentcanary")
        )
    return CanonicalFacts(
        tables=tables,
        current_revision_applications=applications,
        search_hits=searches,
    )


def _source_evidence(root: Path) -> tuple[tuple[Any, ...], ...]:
    with _connect(root / "source.db") as conn:
        rows = conn.execute(
            """
            SELECT
                raw_id,
                origin,
                capture_mode,
                source_path,
                source_index,
                blob_hash,
                blob_size,
                acquired_at_ms,
                logical_source_key,
                revision_kind,
                source_revision,
                predecessor_source_revision,
                predecessor_raw_id,
                baseline_raw_id,
                append_start_offset,
                append_end_offset,
                acquisition_generation,
                revision_authority
            FROM raw_sessions
            ORDER BY raw_id
            """
        )
        return tuple(tuple(_normalize(value) for value in row) for row in rows)


def _parser_census(root: Path) -> tuple[tuple[Any, ...], ...]:
    with _connect(root / "source.db") as conn:
        rows = conn.execute(
            """
            SELECT raw_id, parser_fingerprint, status, logical_keys_json, detail
            FROM raw_authority_parser_census
            ORDER BY raw_id
            """
        )
        return tuple(tuple(_normalize(value) for value in row) for row in rows)


def _fts_recipe(conn: sqlite3.Connection) -> str:
    row = conn.execute("SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'messages_fts'").fetchone()
    assert row is not None
    return " ".join(str(row["sql"]).split())


def _derivation_key(root: Path, index_path: Path) -> DerivationKeyWitness:
    with _connect(index_path) as conn:
        bindings = tuple(
            tuple(_normalize(value) for value in row)
            for row in conn.execute(
                """
                SELECT
                    sessions.session_id,
                    sessions.raw_id,
                    sessions.content_hash,
                    raw_revision_heads.logical_source_key,
                    raw_revision_heads.accepted_raw_id,
                    raw_revision_heads.accepted_source_revision,
                    raw_revision_heads.accepted_content_hash,
                    raw_revision_heads.accepted_frontier_kind,
                    raw_revision_heads.accepted_frontier,
                    raw_revision_heads.acquisition_generation
                FROM sessions
                JOIN raw_revision_heads USING (session_id)
                ORDER BY sessions.session_id
                """
            )
        )
        output_contract = tuple(
            (table, _schema_columns(conn, table))
            for table in (*_CANONICAL_TABLES, "raw_revision_applications", "messages_fts")
        )
        user_version = int(conn.execute("PRAGMA user_version").fetchone()[0])
        recipe_identity = (
            ("index_schema", INDEX_SCHEMA_VERSION, user_version),
            ("fts_contract", _fts_recipe(conn)),
            ("insight_materializer", SESSION_INSIGHT_MATERIALIZER_VERSION),
            ("parser_recipe", tuple(sorted({row[1] for row in _parser_census(root)}))),
        )
    return DerivationKeyWitness(
        subject_source_bindings=bindings,
        source_snapshot=source_revision_snapshot(root),
        source_evidence=_source_evidence(root),
        parser_census=_parser_census(root),
        recipe_identity=recipe_identity,
        output_contract=output_contract,
    )


def _overlay_assertions(root: Path) -> tuple[tuple[Any, ...], ...]:
    with _connect(root / "user.db") as conn:
        rows = conn.execute(
            """
            SELECT
                assertion_id,
                scope_ref,
                target_ref,
                key,
                kind,
                value_json,
                body_text,
                author_ref,
                author_kind,
                evidence_refs_json,
                status,
                visibility,
                confidence,
                staleness_json,
                context_policy_json,
                supersedes_json
            FROM assertions
            ORDER BY assertion_id
            """
        )
        return tuple(tuple(_normalize(value) for value in row) for row in rows)


def _session_content_hash(index_path: Path, session_id: str) -> str:
    with _connect(index_path) as conn:
        row = conn.execute(
            "SELECT content_hash FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
    assert row is not None
    return bytes(row["content_hash"]).hex()


def _assert_raw_payload_identity(root: Path, raw_id: str, payload: bytes) -> None:
    with _connect(root / "source.db") as conn:
        row = conn.execute(
            "SELECT blob_hash, blob_size FROM raw_sessions WHERE raw_id = ?",
            (raw_id,),
        ).fetchone()
    assert row is not None
    assert bytes(row["blob_hash"]) == sha256(payload).digest()
    assert int(row["blob_size"]) == len(payload)


def _plant_stale_dependents(index_path: Path) -> None:
    with _connect(index_path, read_only=False) as conn:
        fts_row = conn.execute(
            """
            SELECT rowid, block_id, message_id, session_id, block_type
            FROM blocks
            WHERE block_id = ?
            """,
            (CHAT_NEW_BLOCK,),
        ).fetchone()
        assert fts_row is not None
        fts_count = int(conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0])
        conn.execute("DELETE FROM messages_fts WHERE rowid = ?", (fts_row["rowid"],))
        conn.execute(
            """
            INSERT INTO messages_fts(rowid, block_id, message_id, session_id, block_type, text)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (*tuple(fts_row), STALE_TOKEN),
        )
        conn.execute(
            "UPDATE session_profiles SET message_count = 999 WHERE session_id = ?",
            (CHAT_SESSION,),
        )
        conn.execute(
            """
            UPDATE insight_materialization
            SET materializer_version = 0
            WHERE insight_type = 'session_profile' AND session_id = ?
            """,
            (CHAT_SESSION,),
        )
        conn.commit()
        assert int(conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0]) == fts_count


def _mark_raws_for_restart(root: Path, raw_ids: tuple[str, ...]) -> None:
    placeholders = ",".join("?" for _ in raw_ids)
    with _connect(root / "source.db", read_only=False) as conn:
        conn.execute(
            f"UPDATE raw_sessions SET parsed_at_ms = NULL WHERE raw_id IN ({placeholders})",
            raw_ids,
        )
        conn.commit()


def _application_receipts(index_path: Path) -> tuple[tuple[Any, ...], ...]:
    with _connect(index_path) as conn:
        rows = conn.execute(
            """
            SELECT decision_id, raw_id, acquisition_generation, decision, accepted_raw_id
            FROM raw_revision_applications
            ORDER BY raw_id, acquisition_generation, decision_id
            """
        )
        return tuple(tuple(_normalize(value) for value in row) for row in rows)


def _attempt_receipt(
    *,
    route: str,
    index_path: Path,
    generation_id: str | None,
    owner_id: str | None,
    generation_state: str,
) -> AttemptReceipt:
    return AttemptReceipt(
        route=route,
        index_path=str(index_path),
        index_inode=index_path.stat().st_ino,
        generation_id=generation_id,
        owner_id=owner_id,
        generation_state=generation_state,
        revision_application_ids=tuple(row[0] for row in _application_receipts(index_path)),
    )


def _assert_planted_contract(root: Path, index_path: Path, raw_ids: RawIds) -> None:
    """Assert facts derived from planted input, never from another route."""
    with _connect(index_path) as conn:
        session_rows = {
            row["session_id"]: (
                row["parent_session_id"],
                row["root_session_id"],
                row["raw_id"],
                row["branch_type"],
                row["active_leaf_message_id"],
                row["message_count"],
            )
            for row in conn.execute(
                """
                SELECT
                    session_id,
                    parent_session_id,
                    root_session_id,
                    raw_id,
                    branch_type,
                    active_leaf_message_id,
                    message_count
                FROM sessions
                ORDER BY session_id
                """
            )
        }
        assert session_rows == {
            CHAT_SESSION: (None, CHAT_SESSION, raw_ids.chat, None, CHAT_NEW, 3),
            PARENT_SESSION: (None, PARENT_SESSION, raw_ids.parent_v2, None, PARENT_SECOND, 2),
            CHILD_SESSION: (
                PARENT_SESSION,
                PARENT_SESSION,
                raw_ids.child,
                "continuation",
                CHILD_FIRST,
                1,
            ),
        }

        messages = {
            row["message_id"]: (
                row["parent_message_id"],
                row["role"],
                row["variant_index"],
                row["is_active_path"],
                row["is_active_leaf"],
            )
            for row in conn.execute(
                """
                SELECT
                    message_id,
                    parent_message_id,
                    role,
                    variant_index,
                    is_active_path,
                    is_active_leaf
                FROM messages
                ORDER BY message_id
                """
            )
        }
        assert messages == {
            CHAT_USER: (None, "user", 0, 1, 0),
            CHAT_OLD: (CHAT_USER, "assistant", 0, 0, 0),
            CHAT_NEW: (CHAT_USER, "assistant", 1, 1, 1),
            PARENT_FIRST: (None, "user", 0, 1, 0),
            PARENT_SECOND: (None, "assistant", 0, 1, 1),
            CHILD_FIRST: (None, "user", 0, 1, 1),
        }

        blocks = {
            row["block_id"]: (row["block_type"], row["text"])
            for row in conn.execute("SELECT block_id, block_type, text FROM blocks ORDER BY block_id")
        }
        assert blocks[CHAT_NEW_BLOCK] == ("text", f"{CANONICAL_TOKEN} canonical answer")
        assert f"{CHAT_OLD}:0" in blocks
        assert f"{CHAT_USER}:0" in blocks

        attachment = conn.execute(
            """
            SELECT
                attachments.display_name,
                attachments.media_type,
                attachments.byte_count,
                attachments.blob_hash,
                attachments.acquisition_status,
                attachments.ref_count,
                attachment_refs.session_id,
                attachment_refs.message_id,
                attachment_refs.position,
                attachment_refs.upload_origin,
                attachment_refs.source_url,
                attachment_refs.caption,
                attachment_native_ids.id_kind,
                attachment_native_ids.native_id
            FROM attachments
            JOIN attachment_refs USING (attachment_id)
            JOIN attachment_native_ids ON attachment_native_ids.ref_id = attachment_refs.ref_id
            """
        ).fetchone()
        assert attachment is not None
        assert int(attachment["position"]) >= 0
        assert tuple(
            attachment[key]
            for key in (
                "display_name",
                "media_type",
                "byte_count",
                "blob_hash",
                "acquisition_status",
                "ref_count",
                "session_id",
                "message_id",
                "upload_origin",
                "source_url",
                "caption",
                "id_kind",
                "native_id",
            )
        ) == (
            "canary.dat",
            None,
            0,
            None,
            "unfetched",
            1,
            CHAT_SESSION,
            CHAT_USER,
            "oauth",
            None,
            None,
            "attachment",
            "att-canary",
        )

        link = conn.execute(
            """
            SELECT
                src_session_id,
                dst_origin,
                dst_native_id,
                link_type,
                resolved_dst_session_id,
                branch_point_message_id,
                inheritance,
                status,
                method,
                confidence,
                evidence_json
            FROM session_links
            """
        ).fetchone()
        assert link is not None
        assert tuple(link[:10]) == (
            CHILD_SESSION,
            "codex-session",
            "lineage-parent",
            "continuation",
            PARENT_SESSION,
            None,
            "spawned-fresh",
            None,
            "parser-parent",
            1.0,
        )
        assert json.loads(str(link["evidence_json"])) == {"parent_session_provider_id": "lineage-parent"}

        heads = {
            row["session_id"]: (
                row["accepted_raw_id"],
                bytes(row["accepted_content_hash"]).hex(),
                row["acquisition_generation"],
            )
            for row in conn.execute(
                """
                SELECT session_id, accepted_raw_id, accepted_content_hash, acquisition_generation
                FROM raw_revision_heads
                ORDER BY session_id
                """
            )
        }
        session_hashes = {
            row["session_id"]: bytes(row["content_hash"]).hex()
            for row in conn.execute("SELECT session_id, content_hash FROM sessions")
        }
        assert heads == {
            CHAT_SESSION: (raw_ids.chat, session_hashes[CHAT_SESSION], 0),
            PARENT_SESSION: (raw_ids.parent_v2, session_hashes[PARENT_SESSION], 1),
            CHILD_SESSION: (raw_ids.child, session_hashes[CHILD_SESSION], 0),
        }

        parent_decisions = {
            (row["raw_id"], row["acquisition_generation"], row["decision"])
            for row in conn.execute(
                """
                WITH ranked AS (
                    SELECT
                        *,
                        ROW_NUMBER() OVER (
                            PARTITION BY raw_id, session_id
                            ORDER BY acquisition_generation DESC, decided_at_ms DESC, decision_id DESC
                        ) AS attempt_rank
                    FROM raw_revision_applications
                )
                SELECT raw_id, acquisition_generation, decision
                FROM ranked
                WHERE attempt_rank = 1 AND session_id = ?
                """,
                (PARENT_SESSION,),
            )
        }
        assert parent_decisions == {
            (raw_ids.parent_v1, 1, "superseded"),
            (raw_ids.parent_v2, 1, "selected_baseline"),
        }

        profiles = {
            row["session_id"]: (
                row["message_count"],
                row["attachment_count"],
                row["tags_json"],
                row["materializer_version"],
            )
            for row in conn.execute(
                """
                SELECT session_id, message_count, attachment_count, tags_json, materializer_version
                FROM session_profiles
                ORDER BY session_id
                """
            )
        }
        assert profiles == {
            CHAT_SESSION: (3, 1, None, SESSION_INSIGHT_MATERIALIZER_VERSION),
            PARENT_SESSION: (2, 0, None, SESSION_INSIGHT_MATERIALIZER_VERSION),
            CHILD_SESSION: (1, 0, None, SESSION_INSIGHT_MATERIALIZER_VERSION),
        }

        materializations = tuple(
            conn.execute(
                """
                SELECT insight_type, session_id, materializer_version
                FROM insight_materialization
                ORDER BY session_id, insight_type
                """
            )
        )
        assert len(materializations) == 27
        assert {int(row[2]) for row in materializations} == {SESSION_INSIGHT_MATERIALIZER_VERSION}
        assert {str(row[0]) for row in materializations} == {
            "context_snapshots",
            "latency",
            "observed_events",
            "phases",
            "provider_usage",
            "runs",
            "session_profile",
            "thread",
            "work_events",
        }

        thread_rows = {
            row["thread_id"]: (
                tuple(json.loads(str(row["session_ids_json"]))),
                row["session_count"],
                row["depth"],
                row["branch_count"],
                row["total_messages"],
                row["materializer_version"],
            )
            for row in conn.execute(
                """
                SELECT
                    thread_id,
                    session_ids_json,
                    session_count,
                    depth,
                    branch_count,
                    total_messages,
                    materializer_version
                FROM threads
                ORDER BY thread_id
                """
            )
        }
        assert thread_rows == {
            CHAT_SESSION: (
                (CHAT_SESSION,),
                1,
                0,
                1,
                3,
                SESSION_INSIGHT_MATERIALIZER_VERSION,
            ),
            PARENT_SESSION: (
                (PARENT_SESSION, CHILD_SESSION),
                2,
                1,
                1,
                3,
                SESSION_INSIGHT_MATERIALIZER_VERSION,
            ),
        }

    with ArchiveStore.open_existing(root, read_only=True) as archive:
        assert archive.search_blocks(CANONICAL_TOKEN) == [CHAT_NEW_BLOCK]
        assert archive.search_blocks(STALE_TOKEN) == []
        assert archive.search_blocks("definitelyabsentcanary") == []


def test_incremental_restart_and_fresh_generation_rebuild_are_equivalent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One realized workload converges across incremental, restart, and rebuild routes."""
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(tmp_path))
    monkeypatch.setenv("POLYLOGUE_SCHEMA_VALIDATION", "off")
    initialize_active_archive_root(tmp_path)

    chat_payload = _chatgpt_branch_payload()
    parent_v1_payload = _codex_session("lineage-parent", (("user", "parent first"),))
    # This is deliberately a strict byte-prefix extension of v1.
    parent_v2_payload = (
        parent_v1_payload
        + json.dumps(
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "id": "lineage-parent-m1",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "parent second"}],
                },
            },
            sort_keys=True,
        ).encode()
        + b"\n"
    )
    child_payload = _codex_session(
        "lineage-child",
        (("user", "child continuation"),),
        parent_native_id="lineage-parent",
    )
    assert parent_v2_payload.startswith(parent_v1_payload)

    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        chat_raw = archive.write_raw_payload(
            provider=Provider.CHATGPT,
            payload=chat_payload,
            source_path="conversations.json",
            acquired_at_ms=1,
        )
        parent_v1_raw = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=parent_v1_payload,
            source_path="parent.jsonl",
            acquired_at_ms=2,
        )
    _assert_raw_payload_identity(tmp_path, chat_raw, chat_payload)
    _assert_raw_payload_identity(tmp_path, parent_v1_raw, parent_v1_payload)

    first_increment = backfill_historical_revision_evidence(
        tmp_path,
        selected_raw_ids=[chat_raw, parent_v1_raw],
    )
    assert (
        first_increment.scanned,
        first_increment.classified_full,
        first_increment.replayed_logical_sources,
        first_increment.quarantined,
    ) == (2, 2, 2, 0)

    source_before_overlay = source_revision_snapshot(tmp_path)
    chat_hash_before_overlay = _session_content_hash(tmp_path / "index.db", CHAT_SESSION)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        assert archive.add_user_tags((CHAT_SESSION,), (OVERLAY_TAG,)) == 1
    overlay_receipt = _overlay_assertions(tmp_path)
    assert len(overlay_receipt) == 1
    assert overlay_receipt[0][2:5] == (
        f"session:{CHAT_SESSION}",
        OVERLAY_TAG,
        "tag",
    )
    assert source_revision_snapshot(tmp_path) == source_before_overlay
    assert _session_content_hash(tmp_path / "index.db", CHAT_SESSION) == chat_hash_before_overlay

    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        parent_v2_raw = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=parent_v2_payload,
            source_path="parent.jsonl",
            acquired_at_ms=3,
        )
        child_raw = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=child_payload,
            source_path="child.jsonl",
            acquired_at_ms=4,
        )
    raw_ids = RawIds(chat_raw, parent_v1_raw, parent_v2_raw, child_raw)
    _assert_raw_payload_identity(tmp_path, raw_ids.parent_v2, parent_v2_payload)
    _assert_raw_payload_identity(tmp_path, raw_ids.child, child_payload)

    targeted_update = backfill_historical_revision_evidence(
        tmp_path,
        selected_raw_ids=[parent_v2_raw, child_raw],
    )
    assert (
        targeted_update.scanned,
        targeted_update.classified_full,
        targeted_update.replayed_logical_sources,
        targeted_update.quarantined,
    ) == (3, 3, 2, 0)
    first_insights = repair_session_insights(_config(tmp_path), dry_run=False)
    assert first_insights.success is True
    assert first_insights.detail == "Session insights ready"
    targeted_key = _derivation_key(tmp_path, tmp_path / "index.db")
    targeted_facts = _collect_canonical_facts(tmp_path, tmp_path / "index.db")
    _assert_planted_contract(tmp_path, tmp_path / "index.db", raw_ids)

    # Anti-vacuity: same row count, wrong FTS content, and an explicitly stale
    # materializer recipe. A row-count or boolean-stale test would miss this.
    _plant_stale_dependents(tmp_path / "index.db")
    with ArchiveStore.open_existing(tmp_path, read_only=True) as archive:
        assert archive.search_blocks(STALE_TOKEN) == [CHAT_NEW_BLOCK]
        assert archive.search_blocks(CANONICAL_TOKEN) == []
    with _connect(tmp_path / "index.db") as conn:
        assert (
            conn.execute(
                "SELECT message_count FROM session_profiles WHERE session_id = ?",
                (CHAT_SESSION,),
            ).fetchone()[0]
            == 999
        )

    # Restart/reprocess the selected terminal raws. Cohort expansion must still
    # reconsider the parent v1 predecessor and preserve the old attempt receipt.
    _mark_raws_for_restart(tmp_path, raw_ids.all())
    restarted = backfill_historical_revision_evidence(
        tmp_path,
        selected_raw_ids=[chat_raw, parent_v2_raw, child_raw],
    )
    assert restarted.scanned == 4
    assert restarted.classified_full == 4
    assert restarted.replayed_logical_sources == 3
    assert restarted.quarantined == 0
    restarted_insights = repair_session_insights(_config(tmp_path), dry_run=False)
    assert restarted_insights.success is True
    assert restarted_insights.detail == "Session insights ready"

    final_source_snapshot = source_revision_snapshot(tmp_path)
    incremental_path = tmp_path / "index.db"
    incremental_key = _derivation_key(tmp_path, incremental_path)
    incremental_facts = _collect_canonical_facts(tmp_path, incremental_path)
    incremental_attempts = _application_receipts(incremental_path)
    assert incremental_key == targeted_key
    assert incremental_facts == targeted_facts
    incremental_receipt = _attempt_receipt(
        route="incremental+targeted+restart",
        index_path=incremental_path,
        generation_id=None,
        owner_id=None,
        generation_state="legacy-active",
    )
    _assert_planted_contract(tmp_path, incremental_path, raw_ids)
    assert _overlay_assertions(tmp_path) == overlay_receipt
    assert incremental_key.source_snapshot == final_source_snapshot
    assert {row[1] for row in incremental_key.parser_census} == {PARSER_RECIPE}
    assert incremental_key.recipe_identity[0] == (
        "index_schema",
        INDEX_SCHEMA_VERSION,
        INDEX_SCHEMA_VERSION,
    )
    assert "unicode61 remove_diacritics 2" in str(incremental_key.recipe_identity[1])

    generation_store = IndexGenerationStore(tmp_path)
    generation = generation_store.create(
        owner_id="testdiet-03-rebuild",
        source_snapshot=final_source_snapshot,
    )
    generation_path = Path(generation.index_path)
    generation_root = generation_path.parent
    rebuild_result = asyncio.run(
        rebuild_index_from_source(
            _config(generation_root, index_path=generation_path),
            raw_ids=list(raw_ids.all()),
            raw_batch_size=500,
            ingest_workers=None,
            materialize=True,
            progress_callback=None,
            owned_inactive_generation=(generation.generation_id, generation.owner_id),
        )
    )
    assert rebuild_result == {
        "scanned_raw_count": 4,
        "classified_full_count": 4,
        "replayed_logical_source_count": 3,
        "quarantined_raw_count": 0,
        "adoption_deferred_raw_count": 0,
        "authority_selection_expanded": True,
        "scheduled_raw_count": 4,
        "raw_batch_size": 500,
    }
    rebuilt_insights = repair_session_insights(
        _config(generation_root, index_path=generation_path),
        dry_run=False,
        archive_root_override=generation_root,
        owned_inactive_generation=(generation.generation_id, generation.owner_id),
    )
    assert rebuilt_insights.success is True
    assert rebuilt_insights.detail == "Session insights ready"
    assert generation_store.load(generation.generation_id).state == "inactive"
    assert source_revision_snapshot(tmp_path) == final_source_snapshot

    rebuild_key = _derivation_key(generation_root, generation_path)
    rebuild_facts = _collect_canonical_facts(generation_root, generation_path)
    rebuild_attempts = _application_receipts(generation_path)
    rebuild_receipt = _attempt_receipt(
        route="fresh-owned-generation",
        index_path=generation_path,
        generation_id=generation.generation_id,
        owner_id=generation.owner_id,
        generation_state="inactive",
    )
    _assert_planted_contract(generation_root, generation_path, raw_ids)

    # Exact derivation-key identity and canonical output, with generation and
    # attempt identity deliberately outside the equivalence key.
    assert rebuild_key == incremental_key
    assert rebuild_facts == incremental_facts
    assert rebuild_facts.digest() == incremental_facts.digest()
    assert rebuild_receipt != incremental_receipt
    assert rebuild_receipt.index_path != incremental_receipt.index_path
    assert rebuild_receipt.index_inode != incremental_receipt.index_inode
    assert rebuild_receipt.generation_id is not None
    assert rebuild_receipt.owner_id == "testdiet-03-rebuild"

    # Incremental history retains its earlier selected-v1 receipt. A fresh
    # rebuild has only final cohort receipts, while their current decisions are
    # equal through ``CanonicalFacts.current_revision_applications``.
    prior_v1_receipts = [
        row
        for row in incremental_attempts
        if row[1] == raw_ids.parent_v1 and row[2] == 0 and row[3] == "selected_baseline"
    ]
    assert len(prior_v1_receipts) == 1
    assert prior_v1_receipts[0] not in rebuild_attempts
    assert len(incremental_attempts) == len(rebuild_attempts) + 1

    # The overlay is deliberately outside raw/session hashes and is rejoined by
    # the generation's user.db symlink, not copied into the content identity.
    assert _overlay_assertions(generation_root) == overlay_receipt
    assert _session_content_hash(generation_path, CHAT_SESSION) == chat_hash_before_overlay
    assert source_revision_snapshot(generation_root) == final_source_snapshot

    promoted = generation_store.promote(generation)
    assert promoted.state == "active"
    assert generation_store.load(generation.generation_id).state == "active"
    assert (tmp_path / "index.db").resolve() == generation_path.resolve()
    assert _collect_canonical_facts(tmp_path, tmp_path / "index.db") == rebuild_facts
    assert _derivation_key(tmp_path, tmp_path / "index.db") == rebuild_key
    assert _overlay_assertions(tmp_path) == overlay_receipt
    assert source_revision_snapshot(tmp_path) == final_source_snapshot
