"""Seed the deterministic demo archive without daemon scheduling."""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from hashlib import sha256
from pathlib import Path

from polylogue.config import Source
from polylogue.pipeline.services.archive_ingest import parse_sources_archive
from polylogue.scenarios import (
    DEMO_CHATGPT_SESSION_ID,
    DEMO_CLAUDE_CODE_LINEAGE_SIDECHAIN_SESSION_ID,
    DEMO_CLAUDE_CODE_SESSION_ID,
    DEMO_CODEX_TERMINAL_ERROR_SESSION_ID,
    DEMO_EMBEDDING_PROSE_SESSION_ID,
    build_demo_corpus_specs,
    seed_demo_user_overlays,
)
from polylogue.schemas.synthetic import SyntheticCorpus
from polylogue.storage.embeddings.materialization import archive_embeddable_message_where
from polylogue.storage.insights.session.rebuild import rebuild_session_insights_sync
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.embedding_write import ArchiveEmbeddingWrite, upsert_message_embeddings
from polylogue.storage.sqlite.archive_tiers.embeddings import EMBEDDING_DIMENSION
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.sqlite_vec_extension import try_load_sqlite_vec

from .constructs import evaluate_demo_constructs
from .models import DemoSeedResult

DEMO_SOURCE_DIRNAME = "demo-fixture-world-source"


@contextmanager
def _pushd(path: Path) -> Iterator[None]:
    """Temporarily run relative-source ingestion from *path*."""

    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def materialize_demo_source(root: Path, *, force: bool = False) -> Path:
    """Write deterministic demo source artifacts under ``root``."""

    source_root = root / DEMO_SOURCE_DIRNAME
    if force and source_root.exists():
        shutil.rmtree(source_root)
    source_root.mkdir(parents=True, exist_ok=True)
    SyntheticCorpus.write_specs_artifacts(
        build_demo_corpus_specs(),
        source_root,
        prefix="demo",
        index_width=2,
    )
    _write_demo_temporary_sources(source_root)
    _write_demo_browser_capture_gap_sources(source_root)
    _write_demo_lineage_sources(source_root)
    return source_root


def _write_jsonl(path: Path, records: tuple[dict[str, object], ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n", encoding="utf-8")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _demo_archive_count(archive_root: Path, table: str) -> int:
    with closing(sqlite3.connect(f"file:{archive_root / 'index.db'}?mode=ro", uri=True)) as conn:
        return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


def _write_demo_temporary_sources(source_root: Path) -> None:
    """Write a Claude.ai temporary-chat fixture through the parser path."""

    _write_json(
        source_root / "claude-ai" / "temporary-demo.json",
        {
            "uuid": "demo-temporary-claude-ai",
            "name": "Temporary demo context check",
            "is_temporary": True,
            "created_at": "2026-07-04T09:50:00Z",
            "updated_at": "2026-07-04T09:50:03Z",
            "chat_messages": [
                {
                    "uuid": "temporary-u0",
                    "sender": "human",
                    "text": "Please answer without retaining this temporary context.",
                    "created_at": "2026-07-04T09:50:01Z",
                    "updated_at": "2026-07-04T09:50:01Z",
                    "content": [
                        {
                            "type": "text",
                            "text": "Please answer without retaining this temporary context.",
                        }
                    ],
                },
                {
                    "uuid": "temporary-a1",
                    "sender": "assistant",
                    "text": "The archive keeps the evidence but marks this session temporary.",
                    "created_at": "2026-07-04T09:50:02Z",
                    "updated_at": "2026-07-04T09:50:03Z",
                    "content": [
                        {
                            "type": "text",
                            "text": "The archive keeps the evidence but marks this session temporary.",
                        },
                        {
                            "type": "token_budget",
                            "remaining": 2048,
                        },
                    ],
                },
            ],
        },
    )


def _write_demo_browser_capture_gap_sources(source_root: Path) -> None:
    """Write browser-capture convergence fixtures for an existing ChatGPT session."""

    native_id = DEMO_CHATGPT_SESSION_ID.split(":", maxsplit=1)[1]
    _write_json(
        source_root / "browser-capture" / "chatgpt-raw-provider.json",
        {
            "polylogue_capture_kind": "browser_llm_session",
            "schema_version": 1,
            "capture_id": f"chatgpt:{native_id}:raw-provider",
            "provenance": {
                "source_url": f"https://chatgpt.com/c/{native_id}",
                "page_title": "ChatGPT - Debugging flaky async pipeline tests",
                "captured_at": "2026-07-04T09:54:00Z",
                "adapter_name": "chatgpt-browser-native-v1",
                "capture_mode": "snapshot",
            },
            "session": {
                "provider": "chatgpt",
                "provider_session_id": native_id,
                "title": "Debugging flaky async pipeline tests",
                "updated_at": "2026-07-04T09:54:00Z",
                "model": "gpt-5-demo",
                "turns": [
                    {
                        "provider_turn_id": "browser-native-placeholder-u0",
                        "role": "user",
                        "text": "Native payload is present; this DOM turn should not define the parsed transcript.",
                        "ordinal": 0,
                    }
                ],
            },
            "raw_provider_payload": {
                "conversation_id": native_id,
                "title": "Debugging flaky async pipeline tests",
                "create_time": 1783158840.0,
                "update_time": 1783158844.0,
                "current_node": "assistant-node",
                "mapping": {
                    "root": {"id": "root", "message": None, "parent": None, "children": ["user-node"]},
                    "user-node": {
                        "id": "user-node",
                        "parent": "root",
                        "children": ["assistant-node"],
                        "message": {
                            "id": "browser-native-u0",
                            "author": {"role": "user"},
                            "create_time": 1783158841.0,
                            "content": {
                                "content_type": "text",
                                "parts": ["Browser capture preserved the native ChatGPT payload."],
                            },
                            "metadata": {},
                        },
                    },
                    "assistant-node": {
                        "id": "assistant-node",
                        "parent": "user-node",
                        "children": [],
                        "message": {
                            "id": "browser-native-a1",
                            "author": {"role": "assistant"},
                            "create_time": 1783158842.0,
                            "content": {
                                "content_type": "text",
                                "parts": ["The canonical session coalesces with the direct ChatGPT export."],
                            },
                            "metadata": {"model_slug": "gpt-5-demo"},
                        },
                    },
                },
            },
        },
    )
    _write_json(
        source_root / "browser-capture" / "chatgpt-dom-fallback.json",
        {
            "polylogue_capture_kind": "browser_llm_session",
            "schema_version": 1,
            "capture_id": f"chatgpt:{native_id}:dom-fallback",
            "provenance": {
                "source_url": f"https://chatgpt.com/c/{native_id}",
                "page_title": "ChatGPT - Debugging flaky async pipeline tests",
                "captured_at": "2026-07-04T09:55:00Z",
                "adapter_name": "chatgpt-dom-v1",
                "capture_mode": "snapshot",
            },
            "session": {
                "provider": "chatgpt",
                "provider_session_id": native_id,
                "title": "Debugging flaky async pipeline tests",
                "updated_at": "2026-07-04T09:55:00Z",
                "model": "gpt-5-demo",
                "turns": [
                    {
                        "provider_turn_id": "dom-gap-u0",
                        "role": "user",
                        "text": "DOM fallback saw only this first turn.",
                        "ordinal": 0,
                    }
                ],
            },
        },
    )


def _codex_session_meta(
    session_id: str,
    *,
    timestamp: str,
    forked_from_id: str | None = None,
    subagent_role: str | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {"id": session_id, "timestamp": timestamp}
    if forked_from_id is not None:
        payload["forked_from_id"] = forked_from_id
    if subagent_role is not None and forked_from_id is not None:
        payload["source"] = {
            "subagent": {
                "thread_spawn": {
                    "parent_thread_id": forked_from_id,
                    "depth": 1,
                    "agent_role": subagent_role,
                }
            }
        }
    return {"type": "session_meta", "payload": payload}


def _codex_message(
    message_id: str,
    role: str,
    timestamp: str,
    content: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "type": "response_item",
        "payload": {
            "id": message_id,
            "type": "message",
            "role": role,
            "timestamp": timestamp,
            "content": content,
        },
    }


def _claude_code_record(
    *,
    record_type: str,
    uuid: str,
    session_id: str,
    timestamp: str,
    role: str,
    content: str,
    is_sidechain: bool = False,
) -> dict[str, object]:
    record: dict[str, object] = {
        "type": record_type,
        "uuid": uuid,
        "sessionId": session_id,
        "timestamp": timestamp,
        "message": {"role": role, "content": content},
    }
    if is_sidechain:
        record["isSidechain"] = True
    return record


def _input_text(text: str) -> dict[str, object]:
    return {"type": "input_text", "text": text}


def _output_text(text: str) -> dict[str, object]:
    return {"type": "output_text", "text": text}


def _write_demo_lineage_sources(source_root: Path) -> None:
    """Write explicit agent lineage fixtures through provider parser paths."""

    parent_id = "demo-lineage-parent"
    fork_id = "demo-lineage-fork"
    subagent_id = "demo-lineage-subagent"
    terminal_error_id = DEMO_CODEX_TERMINAL_ERROR_SESSION_ID.removeprefix("codex-session:")
    claude_parent_id = DEMO_CLAUDE_CODE_SESSION_ID.removeprefix("claude-code-session:")
    sidechain_id = DEMO_CLAUDE_CODE_LINEAGE_SIDECHAIN_SESSION_ID.removeprefix("claude-code-session:")
    base_user = "Map the demo lineage base context."
    base_assistant = "I have the base context and can branch the analysis."

    _write_jsonl(
        source_root / "codex" / "lineage-parent.jsonl",
        (
            _codex_session_meta(parent_id, timestamp="2026-07-04T10:00:00Z"),
            _codex_message("parent-u0", "user", "2026-07-04T10:00:01Z", [_input_text(base_user)]),
            _codex_message("parent-a1", "assistant", "2026-07-04T10:00:02Z", [_output_text(base_assistant)]),
            _codex_message(
                "parent-a2",
                "assistant",
                "2026-07-04T10:00:03Z",
                [
                    _output_text("Delegating a topology check to a focused subagent."),
                    {
                        "type": "tool_use",
                        "id": "task-demo-lineage",
                        "name": "Task",
                        "input": {
                            "subagent_type": "Explore",
                            "prompt": "Inspect the demo lineage child and report caveats.",
                            "child_session_id": "codex-session:demo-lineage-subagent",
                        },
                    },
                    {
                        "type": "tool_result",
                        "tool_use_id": "task-demo-lineage",
                        "content": "Subagent completed. Session: codex-session:demo-lineage-subagent",
                    },
                ],
            ),
        ),
    )
    _write_jsonl(
        source_root / "claude-code" / "agent-acompact-demo.jsonl",
        (
            _claude_code_record(
                record_type="user",
                uuid="acompact-u0",
                session_id=claude_parent_id,
                timestamp="2026-07-04T10:03:01Z",
                role="user",
                content="Summarize the parent context before continuing the demo lineage audit.",
            ),
            _claude_code_record(
                record_type="summary",
                uuid="acompact-s1",
                session_id=claude_parent_id,
                timestamp="2026-07-04T10:03:02Z",
                role="system",
                content="Compacted demo lineage context: parent, branch, subagent, and caveats.",
            ),
        ),
    )
    _write_jsonl(
        source_root / "claude-code" / "lineage-sidechain.jsonl",
        (
            _claude_code_record(
                record_type="user",
                uuid="sidechain-u0",
                session_id=sidechain_id,
                timestamp="2026-07-04T10:04:01Z",
                role="user",
                content="Run a sidechain check for the deterministic demo lineage matrix.",
                is_sidechain=True,
            ),
            _claude_code_record(
                record_type="assistant",
                uuid="sidechain-a1",
                session_id=sidechain_id,
                timestamp="2026-07-04T10:04:02Z",
                role="assistant",
                content="Sidechain check complete; keep this branch typed but separate from parent links.",
                is_sidechain=True,
            ),
        ),
    )
    _write_jsonl(
        source_root / "codex" / "lineage-fork.jsonl",
        (
            _codex_session_meta(fork_id, timestamp="2026-07-04T10:01:00Z", forked_from_id=parent_id),
            _codex_message("fork-u0", "user", "2026-07-04T10:01:01Z", [_input_text(base_user)]),
            _codex_message("fork-a1", "assistant", "2026-07-04T10:01:02Z", [_output_text(base_assistant)]),
            _codex_message(
                "fork-u2",
                "user",
                "2026-07-04T10:01:03Z",
                [_input_text("Now take the forked branch and audit construct validity.")],
            ),
            _codex_message(
                "fork-a3",
                "assistant",
                "2026-07-04T10:01:04Z",
                [_output_text("The fork diverges into demo corpus construct checks.")],
            ),
        ),
    )
    _write_jsonl(
        source_root / "codex" / "lineage-subagent.jsonl",
        (
            _codex_session_meta(
                subagent_id,
                timestamp="2026-07-04T10:02:00Z",
                forked_from_id=parent_id,
                subagent_role="Explore",
            ),
            _codex_message(
                "subagent-a0",
                "assistant",
                "2026-07-04T10:02:01Z",
                [_output_text("Subagent report: lineage fixture has a parent, a fork, and a resolved child link.")],
            ),
        ),
    )
    _write_jsonl(
        source_root / "codex" / "terminal-error.jsonl",
        (
            _codex_session_meta(terminal_error_id, timestamp="2026-07-04T10:05:00Z"),
            _codex_message(
                "terminal-error-u0",
                "user",
                "2026-07-04T10:05:01Z",
                [_input_text("Run the command and stop if it fails.")],
            ),
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call",
                    "id": "fc-terminal-error",
                    "call_id": "call-terminal-error",
                    "name": "exec_command",
                    "arguments": json.dumps({"cmd": "pytest tests/missing_test.py"}, sort_keys=True),
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call_output",
                    "call_id": "call-terminal-error",
                    "output": json.dumps(
                        {
                            "output": "ERROR: file or directory not found: tests/missing_test.py",
                            "metadata": {"exit_code": 4},
                        },
                        sort_keys=True,
                    ),
                },
            },
            _codex_message(
                "terminal-error-a1",
                "assistant",
                "2026-07-04T10:05:04Z",
                [_output_text("I hit an error and need the missing test path corrected before continuing.")],
            ),
        ),
    )


def _materialize_session_insights(archive_root: Path, session_ids: list[str]) -> None:
    """Build the session-profile insight read models for *session_ids*.

    ``parse_sources_archive`` writes the ``sessions``/``messages`` tree but does
    not materialize the derived insight tables (``session_profiles`` and
    siblings); the daemon convergence path normally does that in a separate
    stage. The no-daemon demo seed must run the same rebuild so that the
    postmortem / session-digest read surfaces resolve against a populated demo
    archive instead of an empty one. Passing ``session_ids`` makes
    ``rebuild_session_insights_sync`` commit internally and skip the full-table
    delete/rebuild path.
    """

    if not session_ids:
        return
    conn = sqlite3.connect(archive_root / "index.db")
    try:
        conn.row_factory = sqlite3.Row
        rebuild_session_insights_sync(conn, session_ids=session_ids)
    finally:
        conn.close()


# Deterministic per-assistant-message Opus usage injected into the demo
# claude-code session so cost surfaces (`analyze --postmortem` top session,
# cost rollups) render a believable bundle on the demo archive instead of $0.
# The demo corpus is fully synthetic; the synthetic generator does not emit
# token usage, and adding it there would ripple every synthetic test snapshot.
# Scoping the injection to the one demo session keeps it ripple-free.
_DEMO_USAGE_MODEL = "claude-opus-4-8"
_DEMO_USAGE = {
    "input_tokens": 8000,
    "output_tokens": 1500,
    "cache_read_tokens": 40000,
    "cache_write_tokens": 6000,
}


def _inject_demo_session_usage(archive_root: Path) -> None:
    """Set deterministic Opus token usage on the demo claude-code assistant turns.

    Cost is materialized into ``session_profiles.total_cost_usd`` from the
    per-message token columns, so this must run *before*
    ``_materialize_session_insights``. Demo-scoped by session id: no synthetic
    generator change, no test-snapshot ripple.
    """

    conn = sqlite3.connect(archive_root / "index.db")
    try:
        conn.execute(
            "UPDATE messages SET model_name = ?, input_tokens = ?, output_tokens = ?, "
            "cache_read_tokens = ?, cache_write_tokens = ? "
            "WHERE session_id = ? AND role = 'assistant'",
            (
                _DEMO_USAGE_MODEL,
                _DEMO_USAGE["input_tokens"],
                _DEMO_USAGE["output_tokens"],
                _DEMO_USAGE["cache_read_tokens"],
                _DEMO_USAGE["cache_write_tokens"],
                DEMO_CLAUDE_CODE_SESSION_ID,
            ),
        )
        conn.commit()
    finally:
        conn.close()


# Canonical repo identity for the demo claude-code session so the postmortem
# `repos_touched` metric renders a believable project instead of an empty list.
# The synthetic demo corpus emits no cwd/repo attribution, so profile
# materialization derives no repo names; this is set *after* materialization
# (which would otherwise overwrite it with the empty attribution result) and is
# scoped to the one demo session.
_DEMO_REPO_NAMES = '["polylogue"]'


def _inject_demo_session_repos(archive_root: Path) -> None:
    """Set a canonical repo name on the demo claude-code session profile."""

    conn = sqlite3.connect(archive_root / "index.db")
    try:
        conn.execute(
            "UPDATE session_profiles SET repo_names_json = ? WHERE session_id = ?",
            (_DEMO_REPO_NAMES, DEMO_CLAUDE_CODE_SESSION_ID),
        )
        conn.commit()
    finally:
        conn.close()


_DEMO_EMBEDDING_MODEL = "demo-synthetic-embedding"
_DEMO_EMBEDDING_AT_MS = 1_767_225_700_000


def _demo_embedding_vector(message_id: str) -> list[float]:
    """Return a deterministic non-provider vector for demo-only embeddings."""

    digest = sha256(message_id.encode("utf-8")).digest()
    return [((digest[index % len(digest)] / 255.0) * 2.0) - 1.0 for index in range(EMBEDDING_DIMENSION)]


def _demo_embedding_content_hash(value: object) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        return bytes.fromhex(value)
    raise TypeError(f"unsupported content_hash value: {type(value).__name__}")


def _seed_demo_embeddings(archive_root: Path) -> None:
    """Seed deterministic embeddings for one authored-prose demo session.

    This deliberately does not call an embedding provider. It proves the
    embedding tier and status surfaces against non-empty rows while keeping the
    demo archive private-data-free and cost-free.
    """

    embeddings_db = archive_root / "embeddings.db"
    initialize_archive_database(embeddings_db, ArchiveTier.EMBEDDINGS)
    index_conn = sqlite3.connect(archive_root / "index.db")
    embeddings_conn = sqlite3.connect(embeddings_db)
    try:
        loaded, error = try_load_sqlite_vec(embeddings_conn)
        if not loaded:
            raise RuntimeError("demo embedding seeding requires sqlite-vec") from error
        index_conn.row_factory = sqlite3.Row
        rows = index_conn.execute(
            f"""
            SELECT m.message_id, m.session_id, s.origin, m.content_hash,
                   GROUP_CONCAT(b.text, char(10) || char(10)) AS text
            FROM messages AS m
            JOIN sessions AS s
              ON s.session_id = m.session_id
            JOIN blocks AS b
              ON b.session_id = m.session_id
             AND b.message_id = m.message_id
             AND b.block_type = 'text'
             AND b.text IS NOT NULL
            WHERE m.session_id = ?
              AND {archive_embeddable_message_where("m")}
            GROUP BY m.message_id, m.position, m.variant_index
            HAVING LENGTH(TRIM(COALESCE(text, ''))) >= 20
            ORDER BY m.position, m.variant_index
            LIMIT 3
            """,
            (DEMO_EMBEDDING_PROSE_SESSION_ID,),
        ).fetchall()
        writes = [
            ArchiveEmbeddingWrite(
                message_id=str(row["message_id"]),
                session_id=str(row["session_id"]),
                origin=str(row["origin"]),
                embedding=_demo_embedding_vector(str(row["message_id"])),
                model=_DEMO_EMBEDDING_MODEL,
                embedded_at_ms=_DEMO_EMBEDDING_AT_MS,
                content_hash=_demo_embedding_content_hash(row["content_hash"]),
            )
            for row in rows
            if row["content_hash"] is not None
        ]
        upsert_message_embeddings(embeddings_conn, writes)
        embeddings_conn.execute(
            """
            INSERT INTO embedding_status (
                session_id, origin, message_count_embedded, last_embedded_at_ms, needs_reindex, error_message
            ) VALUES (?, ?, ?, ?, 0, NULL)
            ON CONFLICT(session_id) DO UPDATE SET
                origin = excluded.origin,
                message_count_embedded = excluded.message_count_embedded,
                last_embedded_at_ms = excluded.last_embedded_at_ms,
                needs_reindex = 0,
                error_message = NULL
            """,
            (
                DEMO_EMBEDDING_PROSE_SESSION_ID,
                DEMO_EMBEDDING_PROSE_SESSION_ID.split(":", maxsplit=1)[0],
                len(writes),
                _DEMO_EMBEDDING_AT_MS,
            ),
        )
        embeddings_conn.commit()
    finally:
        index_conn.close()
        embeddings_conn.close()


def demo_source_specs(source_root: Path) -> list[Source]:
    """Return relative source specs for the materialized demo world."""

    return [
        Source(name="chatgpt", path=Path("chatgpt")),
        Source(name="claude-ai", path=Path("claude-ai")),
        Source(name="claude-code", path=Path("claude-code")),
        Source(name="codex", path=Path("codex")),
        Source(name="gemini", path=Path("gemini")),
        Source(name="browser-capture", path=Path("browser-capture")),
    ]


async def seed_demo_archive(
    archive_root: Path,
    *,
    force: bool = False,
    with_overlays: bool = False,
) -> DemoSeedResult:
    """Materialize, ingest, and optionally overlay the deterministic demo archive."""

    source_root = materialize_demo_source(archive_root, force=force)
    with _pushd(source_root):
        result = await parse_sources_archive(archive_root, demo_source_specs(source_root))

    session_ids = sorted(result.processed_ids)
    _inject_demo_session_usage(archive_root)
    _materialize_session_insights(archive_root, session_ids)
    _inject_demo_session_repos(archive_root)
    _seed_demo_embeddings(archive_root)

    overlay = seed_demo_user_overlays(archive_root) if with_overlays else None
    construct_coverage = evaluate_demo_constructs(archive_root)
    return DemoSeedResult(
        archive_root=archive_root,
        source_root=source_root,
        session_count=_demo_archive_count(archive_root, "sessions"),
        message_count=_demo_archive_count(archive_root, "messages"),
        session_ids=tuple(sorted(result.processed_ids)),
        overlays_seeded=overlay is not None,
        assertion_count=len(overlay.assertion_ids) if overlay else 0,
        construct_coverage=construct_coverage,
    )


__all__ = ["DEMO_SOURCE_DIRNAME", "demo_source_specs", "materialize_demo_source", "seed_demo_archive"]
