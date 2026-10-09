"""A real retained transcript whose active managed Index fingerprint has changed."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from contextlib import closing
from pathlib import Path

from polylogue.core.enums import Provider
from polylogue.storage.sqlite.sqlite_vec_extension import try_load_sqlite_vec
from tests.infra.empty_managed_index import make_empty_managed_index, mutate_fixture_database
from tests.infra.retained_replay import publish_retained_payload


def make_populated_stale_index(
    root: Path,
    source: Path,
    *,
    multi_session: bool = False,
    include_history: bool = False,
    include_codex_materials: bool = False,
) -> tuple[Path, str, tuple[str, ...]]:
    old = make_empty_managed_index(root, stale=False)
    source.parent.mkdir()
    source.write_bytes(
        b'{"type":"session_meta","payload":{"id":"retained-reconvergence",'
        b'"timestamp":"2026-06-02T00:00:00Z"}}\n'
        b'{"type":"response_item","payload":{"type":"message","id":"neutral-message",'
        b'"role":"user","content":[{"type":"input_text","text":"neutral retained prose"}]}}\n'
    )
    if multi_session:
        source.write_bytes(
            json.dumps(
                [
                    {
                        "id": name,
                        "current_node": "m",
                        "mapping": {
                            "m": {
                                "id": "m",
                                "parent": None,
                                "children": [],
                                "message": {
                                    "id": "m",
                                    "author": {"role": "user"},
                                    "create_time": 1,
                                    "content": {"content_type": "text", "parts": ["neutral retained prose"]},
                                },
                            }
                        },
                    }
                    for name in ("neutral-first", "neutral-second")
                ]
            ).encode()
        )
    raw_id, session_ids = asyncio.run(
        publish_retained_payload(
            root,
            provider=Provider.CHATGPT if multi_session else Provider.CODEX,
            payload=source.read_bytes(),
            source_path=str(source),
            acquired_at_ms=0,
        )
    )
    source.unlink()
    with closing(sqlite3.connect(old)) as conn:
        message_id = str(conn.execute("SELECT message_id FROM messages").fetchone()[0])
    mutate_fixture_database(
        root / "user.db",
        "INSERT INTO assertions(assertion_id,target_ref,kind,body_text,created_at_ms,updated_at_ms) VALUES ('neutral-note',?,'note','neutral note',0,0)",
        ("message:" + message_id,),
    )

    def prepare_audit_reference() -> None:
        from polylogue.operations.audit import AuditRepository
        from polylogue.operations.bindings import runtime_operation_binding
        from polylogue.operations.mutation_actuators import IdentityResetActuator, IdentityResetArgs
        from polylogue.operations.mutation_transaction import MutationPrincipal, OperationExecutor

        binding = runtime_operation_binding(IdentityResetActuator())
        principal = MutationPrincipal(
            "user:neutral",
            frozenset(
                capability for policy in binding.spec.target_authority for capability in policy.required_capabilities
            ),
            "cli",
        )
        executor = OperationExecutor(audit=AuditRepository.for_archive_root(root), archive_root=root)
        executor.prepare_bound_for_archive(
            binding, IdentityResetArgs(root, session_ids, "neutral pending preview"), principal, archive_root=root
        )

    from tests.infra.archive_templates import run_archive_fixture_write

    asyncio.run(run_archive_fixture_write(root, prepare_audit_reference))
    with closing(sqlite3.connect(root / "embeddings.db")) as conn, conn:
        loaded, error = try_load_sqlite_vec(conn)
        assert loaded, error
        from polylogue.storage.sqlite.archive_tiers.embedding_write import upsert_message_embedding

        upsert_message_embedding(
            conn,
            message_id=message_id,
            session_id=session_ids[0],
            origin="codex-cli",
            embedding=[0.25] * 1024,
            model="neutral-model",
            embedded_at_ms=0,
            vector_derivation_hash=b"\0" * 32,
        )
    if include_codex_materials:
        admit_retained_codex_materials(root, source.parent / "codex-state")
        mutate_fixture_database(
            root / "source.db",
            "UPDATE raw_sessions SET revision_authority='asserted',validation_mode=NULL WHERE raw_id=?",
            (raw_id,),
        )
    if include_history:
        history_id = admit_retained_history(root, source.parent / ".claude" / "history.jsonl")
        # A previous preparation refreshed the authority census without
        # advancing this independent non-session membership receipt.
        mutate_fixture_database(
            root / "source.db",
            "UPDATE raw_membership_census SET parser_fingerprint='prior-parser' WHERE raw_id=?",
            (history_id,),
        )
    mutate_fixture_database(old, "UPDATE schema_identity SET identity='prior-runtime' WHERE tier='index'")
    return old, raw_id, session_ids


def admit_retained_history(root: Path, source: Path) -> str:
    """Retain declared raw-only history through ordinary file acquisition."""
    from types import SimpleNamespace

    from polylogue.sources.live.batch import LiveBatchProcessor
    from polylogue.sources.live.cursor import CursorStore
    from polylogue.sources.live.watcher import _PARSER_FINGERPRINT, WatchSource
    from tests.infra.raw_owner_routes import run_ingest_files

    source.parent.mkdir(parents=True)
    source.write_bytes(b"")
    processor = LiveBatchProcessor(
        SimpleNamespace(archive_root=root, backend=SimpleNamespace(db_path=root / "index.db")),
        (WatchSource("claude-code-history", source.parent),),
        cursor=CursorStore(root / "ops.db"),
        parser_fingerprint=_PARSER_FINGERPRINT,
    )
    metrics = run_ingest_files(processor, [source], emit_event=False)
    assert metrics.excluded_file_count == 1 and metrics.failed_file_count == 0
    with closing(sqlite3.connect(root / "source.db")) as conn:
        row = conn.execute("SELECT raw_id FROM raw_sessions WHERE source_path=?", (str(source),)).fetchone()
        assert row is not None
    source.unlink()
    return str(row[0])


def logical_rows(path: Path) -> tuple[str, ...]:
    with closing(sqlite3.connect(path)) as conn:
        if path.name == "embeddings.db":
            loaded, error = try_load_sqlite_vec(conn)
            assert loaded, error
        return tuple(conn.iterdump())


def admit_retained_codex_materials(root: Path, directory: Path) -> None:
    """Acquire real logical state exports, then remove the external originals."""
    from types import SimpleNamespace

    from polylogue.sources.live.batch import LiveBatchProcessor
    from polylogue.sources.live.cursor import CursorStore
    from polylogue.sources.live.watcher import _PARSER_FINGERPRINT, WatchSource
    from polylogue.sources.source_layout import export_drop_layout
    from tests.infra.raw_owner_routes import run_ingest_files

    directory.mkdir()
    goals = directory / "goals_1.sqlite"
    memories = directory / "memories_1.sqlite"
    with closing(sqlite3.connect(goals)) as conn, conn:
        conn.executescript(
            "CREATE TABLE thread_goals(thread_id TEXT PRIMARY KEY,goal_id TEXT NOT NULL,"
            "objective TEXT NOT NULL,status TEXT NOT NULL,token_budget INTEGER,tokens_used INTEGER NOT NULL,"
            "time_used_seconds INTEGER NOT NULL,created_at_ms INTEGER NOT NULL,updated_at_ms INTEGER NOT NULL);"
            "CREATE TABLE thread_goal_continuation_deferrals(thread_id TEXT,deferred_until_ms INTEGER);"
        )
        conn.executemany(
            "INSERT INTO thread_goals VALUES (?,?,?,?,?,?,?,?,?)",
            (
                (
                    "retained-reconvergence" if i == 0 else f"neutral-thread-{i}",
                    f"neutral-goal-{i}",
                    "neutral objective",
                    "active",
                    10,
                    1,
                    1,
                    0,
                    0,
                )
                for i in range(1)
            ),
        )
    with closing(sqlite3.connect(memories)) as conn, conn:
        conn.executescript(
            "CREATE TABLE stage1_outputs(thread_id TEXT PRIMARY KEY,source_updated_at INTEGER NOT NULL,"
            "raw_memory TEXT NOT NULL,rollout_summary TEXT NOT NULL,rollout_slug TEXT,generated_at INTEGER NOT NULL,"
            "usage_count INTEGER,last_usage INTEGER,selected_for_phase2 INTEGER NOT NULL DEFAULT 0,"
            "selected_for_phase2_source_updated_at INTEGER);"
            "CREATE TABLE jobs(id TEXT PRIMARY KEY,state TEXT);"
        )
        conn.executemany(
            "INSERT INTO stage1_outputs VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                (
                    "retained-reconvergence" if i == 0 else f"neutral-thread-{i}",
                    0,
                    "neutral memory",
                    "neutral summary",
                    "neutral",
                    0,
                    1,
                    0,
                    1,
                    None,
                )
                for i in range(1)
            ),
        )
    processor = LiveBatchProcessor(
        SimpleNamespace(archive_root=root, backend=SimpleNamespace(db_path=root / "index.db")),
        (WatchSource("codex-state", directory, layout=export_drop_layout((".sqlite", ".db"))),),
        cursor=CursorStore(root / "ops.db"),
        parser_fingerprint=_PARSER_FINGERPRINT,
    )
    metrics = run_ingest_files(processor, [goals, memories], emit_event=False)
    assert metrics.failed_file_count == 0
    goals.unlink()
    memories.unlink()
