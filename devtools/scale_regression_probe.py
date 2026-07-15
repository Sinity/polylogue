"""Seeded scale-regression probe for archive bug classes.

This is intentionally small in bytes and large in shape: it exercises the
classes of bugs that only showed up on the live archive without seeding a live
archive clone.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sqlite3
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TextIO, cast
from unittest.mock import patch

import aiosqlite
from click.testing import CliRunner

from polylogue.archive.message.roles import Role
from polylogue.cli import cli
from polylogue.config import Config
from polylogue.core.enums import BlockType, Origin, Provider
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.storage import repair as repair_mod
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.insights.session import rebuild as rebuild_mod
from polylogue.storage.insights.session.runtime import SessionInsightCounts
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.write import write_parsed_session_to_archive
from polylogue.storage.sqlite.connection import open_connection
from polylogue.storage.sqlite.run_projection_relations import run_relation_sql


@dataclass(frozen=True, slots=True)
class ScaleRegressionCheck:
    name: str
    ok: bool
    details: dict[str, object]


@dataclass(frozen=True, slots=True)
class ScaleRegressionReport:
    ok: bool
    duration_ms: int
    archive_root: str
    checks: tuple[ScaleRegressionCheck, ...]

    def to_payload(self) -> dict[str, object]:
        return asdict(self)


def _hex_hash(label: str) -> str:
    import hashlib

    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _session_id(native_id: str, origin: str = Origin.CODEX_SESSION.value) -> str:
    return f"{origin}:{native_id}"


def _init_archive(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    initialize_archive_database(root / "source.db", ArchiveTier.SOURCE)
    initialize_archive_database(root / "index.db", ArchiveTier.INDEX)
    initialize_archive_database(root / "user.db", ArchiveTier.USER)


def _parsed_session(native_id: str, *, title: str, messages: int = 1) -> ParsedSession:
    parsed_messages: list[ParsedMessage] = []
    for index in range(messages):
        role = Role.USER if index % 2 == 0 else Role.ASSISTANT
        parsed_messages.append(
            ParsedMessage(
                provider_message_id=f"{native_id}:msg-{index}",
                role=role,
                text=f"{title} message {index}",
                blocks=[
                    ParsedContentBlock(
                        type=BlockType.TEXT,
                        text=f"{title} message {index}",
                    )
                ],
                timestamp=f"2026-07-04T00:00:{index:02d}+00:00",
            )
        )
    return ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id=native_id,
        title=title,
        created_at="2026-07-04T00:00:00+00:00",
        updated_at="2026-07-04T00:01:00+00:00",
        messages=parsed_messages,
    )


def _write_session(conn: sqlite3.Connection, native_id: str, *, title: str, messages: int = 1) -> str:
    session = _parsed_session(native_id, title=title, messages=messages)
    return write_parsed_session_to_archive(
        conn,
        session,
        content_hash=_hex_hash(f"{native_id}:{title}:{messages}"),
        force_replace=True,
    )


def _check_message_budget_chunking(root: Path) -> ScaleRegressionCheck:
    db_path = root / "index.db"
    natives = ("scale-budget-0", "scale-budget-1", "scale-budget-2")
    with open_connection(db_path) as conn:
        for native in natives:
            _write_session(conn, native, title=f"v1-{native}")
        rebuild_mod.rebuild_session_insights_sync(conn)
        conn.commit()
        session_ids = tuple(_session_id(native) for native in natives)
        conn.executemany(
            "UPDATE sessions SET title = ?, message_count = ? WHERE session_id = ?",
            [
                ("v2-scale-budget-0", 2, session_ids[0]),
                ("v2-scale-budget-1", 2, session_ids[1]),
                ("v2-scale-budget-2", 1, session_ids[2]),
            ],
        )
        conn.commit()

    old_budget = rebuild_mod._SESSION_INSIGHT_REBUILD_MESSAGE_BUDGET
    committed_v2_counts: list[int] = []
    visible_profile_counts: list[int] = []

    def observe(amount: int, desc: str | None = None) -> None:
        del amount
        if desc is not None and desc.startswith("rebuild:"):
            return
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as probe:
            probe.row_factory = sqlite3.Row
            rows = probe.execute(
                "SELECT session_id, title FROM session_profiles WHERE session_id IN (?, ?, ?)",
                session_ids,
            ).fetchall()
        visible_profile_counts.append(len({str(row["session_id"]) for row in rows}))
        committed_v2_counts.append(sum(1 for row in rows if str(row["title"]).startswith("v2-")))

    try:
        rebuild_mod._SESSION_INSIGHT_REBUILD_MESSAGE_BUDGET = 3
        with open_connection(db_path) as conn:
            counts = rebuild_mod.rebuild_session_insights_sync(conn, page_size=50, progress_callback=observe)
    finally:
        rebuild_mod._SESSION_INSIGHT_REBUILD_MESSAGE_BUDGET = old_budget

    ok = (
        counts.profiles == 3
        and len(committed_v2_counts) == 2
        and visible_profile_counts == [3, 3]
        and committed_v2_counts[0] == 0
        and 0 < committed_v2_counts[1] < 3
    )
    return ScaleRegressionCheck(
        "chunked_rebuild",
        ok,
        {
            "profile_count": counts.profiles,
            "visible_profile_counts": visible_profile_counts,
            "committed_v2_counts": committed_v2_counts,
        },
    )


def _inflate_large_session(db_path: Path, native: str) -> str:
    session_id = _session_id(native)
    with open_connection(db_path) as conn:
        _write_session(conn, native, title=f"Large {native}", messages=2)
        conn.execute(
            """
            UPDATE sessions
            SET message_count = ?, word_count = ?, tool_use_count = ?, thinking_count = ?
            WHERE session_id = ?
            """,
            (50, 1234, 7, 3, session_id),
        )
        conn.commit()
    return session_id


def _check_bounded_giant_session(root: Path) -> ScaleRegressionCheck:
    db_path = root / "index.db"
    sync_session_id = _inflate_large_session(db_path, "scale-large-sync")
    async_session_id = _inflate_large_session(db_path, "scale-large-async")
    old_threshold = rebuild_mod._SESSION_INSIGHT_DEGRADED_MESSAGE_THRESHOLD
    old_sync_loader = rebuild_mod.load_sync_batch
    old_async_loader = rebuild_mod.load_async_batch
    sync_loader_calls = 0
    async_loader_calls = 0

    def fail_sync_load(_conn: sqlite3.Connection, _session_ids: object) -> object:
        nonlocal sync_loader_calls
        sync_loader_calls += 1
        raise AssertionError("bounded giant-session sync path hydrated the full session")

    async def fail_async_load(_conn: object, _session_ids: object) -> object:
        nonlocal async_loader_calls
        async_loader_calls += 1
        raise AssertionError("bounded giant-session async path hydrated the full session")

    try:
        rebuild_mod._SESSION_INSIGHT_DEGRADED_MESSAGE_THRESHOLD = 10
        rebuild_mod.load_sync_batch = cast(Any, fail_sync_load)
        with open_connection(db_path) as conn:
            sync_started_at = time.perf_counter()
            sync_counts = rebuild_mod.rebuild_session_insights_sync(conn, session_ids=[sync_session_id])
            sync_elapsed_s = time.perf_counter() - sync_started_at
            sync_profile = conn.execute(
                "SELECT workflow_shape, message_count, word_count, tool_use_count, inference_payload_json "
                "FROM session_profiles WHERE session_id = ?",
                (sync_session_id,),
            ).fetchone()

        rebuild_mod.load_async_batch = cast(Any, fail_async_load)

        async def _run_async() -> tuple[SessionInsightCounts, float, sqlite3.Row | None]:
            async with aiosqlite.connect(db_path) as conn:
                conn.row_factory = sqlite3.Row
                started_at = time.perf_counter()
                counts = await rebuild_mod.rebuild_session_insights_async(conn, session_ids=[async_session_id])
                elapsed = time.perf_counter() - started_at
                row = await (
                    await conn.execute(
                        "SELECT workflow_shape, message_count, word_count, tool_use_count, inference_payload_json "
                        "FROM session_profiles WHERE session_id = ?",
                        (async_session_id,),
                    )
                ).fetchone()
                return counts, elapsed, row

        async_counts, async_elapsed_s, async_profile = asyncio.run(_run_async())
    finally:
        rebuild_mod._SESSION_INSIGHT_DEGRADED_MESSAGE_THRESHOLD = old_threshold
        rebuild_mod.load_sync_batch = old_sync_loader
        rebuild_mod.load_async_batch = old_async_loader

    ok = (
        sync_counts.profiles == 1
        and async_counts.profiles == 1
        and sync_loader_calls == 0
        and async_loader_calls == 0
        and sync_elapsed_s < 2.0
        and async_elapsed_s < 2.0
        and sync_profile is not None
        and async_profile is not None
        and sync_profile["workflow_shape"] == "bounded_large_session"
        and async_profile["workflow_shape"] == "bounded_large_session"
        and "large_session_bounded" in str(sync_profile["inference_payload_json"])
        and "large_session_bounded" in str(async_profile["inference_payload_json"])
    )
    return ScaleRegressionCheck(
        "bounded_giant_session",
        ok,
        {
            "sync_elapsed_s": round(sync_elapsed_s, 4),
            "async_elapsed_s": round(async_elapsed_s, 4),
            "sync_loader_calls": sync_loader_calls,
            "async_loader_calls": async_loader_calls,
            "sync_profiles": sync_counts.profiles,
            "async_profiles": async_counts.profiles,
        },
    )


def _check_reset_preserves_source(root: Path) -> ScaleRegressionCheck:
    reset_root = root / "reset-archive"
    _init_archive(reset_root)
    rebuildable = [
        reset_root / "index.db",
        reset_root / "index.db-wal",
        reset_root / "index.db-shm",
        reset_root / "embeddings.db",
        reset_root / "embeddings.db-wal",
        reset_root / "embeddings.db-shm",
        reset_root / "ops.db",
    ]
    for path in rebuildable:
        path.write_text("rebuildable", encoding="utf-8")
    source_db = reset_root / "source.db"
    user_db = reset_root / "user.db"
    with (
        patch("polylogue.cli.commands.reset.archive_root", return_value=reset_root),
        patch("polylogue.cli.commands.reset.data_home", return_value=root),
    ):
        result = CliRunner().invoke(cli, ["ops", "reset", "--database", "--yes"])

    ok = (
        result.exit_code == 0
        and source_db.exists()
        and user_db.exists()
        and all(not path.exists() for path in rebuildable)
        and "Preserving source.db" in result.output
        and "Preserving user.db" in result.output
    )
    return ScaleRegressionCheck(
        "reset_source_preservation",
        ok,
        {
            "exit_code": result.exit_code,
            "source_exists": source_db.exists(),
            "user_exists": user_db.exists(),
            "deleted_rebuildable_count": sum(1 for path in rebuildable if not path.exists()),
        },
    )


def _check_run_ref_no_drop(root: Path) -> ScaleRegressionCheck:
    """Subagent run rows stay distinct and don't collide with the parent's main run.

    polylogue-dab/itvd: the pre-dab materialized-writer model synthesized N
    virtual subagent run rows under the *parent's* session_id (one per
    Task-tool report), and this check originally guarded against two reports
    sharing a tool_id producing colliding synthesized run_refs. That writer
    (and the session_runs table it wrote to) no longer exists --
    run_projection_relations.py's CTE gives exactly one run row per subagent
    *session* (sessions.branch_type = 'subagent'), keyed by its own
    session_id, so the run_ref collision this check used to guard against is
    now structurally impossible. This exercises the replacement guarantee:
    two subagent sessions under the same parent still produce two distinct,
    non-colliding run rows on every read.
    """
    db_path = root / "run-ref-index.db"
    initialize_archive_database(db_path, ArchiveTier.INDEX)
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute(
            "INSERT INTO sessions(native_id, origin, content_hash) VALUES(?, ?, ?)",
            ("parent", "codex-session", bytes(32)),
        )
        conn.execute(
            "INSERT INTO sessions(native_id, origin, content_hash, branch_type, parent_session_id) "
            "VALUES(?, ?, ?, ?, ?)",
            ("child-a", "codex-session", bytes([1]) * 32, "subagent", "codex-session:parent"),
        )
        conn.execute(
            "INSERT INTO sessions(native_id, origin, content_hash, branch_type, parent_session_id) "
            "VALUES(?, ?, ?, ?, ?)",
            ("child-b", "codex-session", bytes([2]) * 32, "subagent", "codex-session:parent"),
        )
        conn.commit()
        rows = conn.execute(
            f"{run_relation_sql()} SELECT run_ref, session_id, role FROM runs ORDER BY run_ref"
        ).fetchall()
        rows_again = conn.execute(
            f"{run_relation_sql()} SELECT run_ref, session_id, role FROM runs ORDER BY run_ref"
        ).fetchall()

    row_tuples = tuple((str(row["run_ref"]), str(row["session_id"]), str(row["role"])) for row in rows)
    expected = (
        ("run:codex-session:child-a", "codex-session:child-a", "subagent"),
        ("run:codex-session:child-b", "codex-session:child-b", "subagent"),
        ("run:codex-session:parent", "codex-session:parent", "main"),
    )
    ok = (
        row_tuples == expected
        and len({run_ref for run_ref, _, _ in row_tuples}) == len(row_tuples)
        and len(rows_again) == len(rows)
    )
    return ScaleRegressionCheck(
        "run_ref_no_drop",
        ok,
        {"row_count": len(row_tuples), "rows": [list(row) for row in row_tuples]},
    )


def _check_raw_materialization_backlog(root: Path) -> ScaleRegressionCheck:
    raw_root = root / "raw-materialization"
    _init_archive(raw_root)
    blob_store = BlobStore(raw_root / "blob")
    raw_id, blob_size = blob_store.write_from_bytes(b'{"mapping":{}}')
    with sqlite3.connect(raw_root / "source.db") as conn:
        conn.execute(
            """
            INSERT INTO raw_sessions (
                raw_id, origin, native_id, source_path, source_index, blob_hash, blob_size,
                acquired_at_ms, validated_at_ms, validation_status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                raw_id,
                Origin.CHATGPT_EXPORT.value,
                "scale-raw-replay",
                "scale-raw-replay.json",
                0,
                bytes.fromhex(raw_id),
                blob_size,
                1000,
                1000,
                "passed",
            ),
        )
        conn.commit()

    config = Config(archive_root=raw_root, render_root=raw_root, sources=[], db_path=raw_root / "index.db")
    preview = repair_mod.raw_materialization_replay_backlog(config, limit=5)
    dry_run = repair_mod.repair_raw_materialization(config, dry_run=True)
    ok = preview["candidate_count"] == 1 and dry_run.repaired_count == 1 and dry_run.success is True
    return ScaleRegressionCheck(
        "raw_materialization_debt_detected",
        ok,
        {
            "candidate_count": preview["candidate_count"],
            "dry_run_repaired_count": dry_run.repaired_count,
            "dry_run_success": dry_run.success,
            "blob_size": blob_size,
        },
    )


def _check_insights_stage_resumable(root: Path) -> ScaleRegressionCheck:
    from polylogue.daemon.convergence_stages import make_default_convergence_stages

    stages = {stage.name: stage for stage in make_default_convergence_stages(root / "index.db")}
    insights = stages.get("insights")
    ok = insights is not None and insights.false_means_pending is True
    return ScaleRegressionCheck(
        "insights_stage_resumable",
        ok,
        {"stage_names": sorted(stages), "false_means_pending": bool(insights and insights.false_means_pending)},
    )


def run_scale_regression_probe(workdir: Path, *, keep: bool = False) -> ScaleRegressionReport:
    started_at = time.perf_counter()
    workdir = workdir.expanduser().resolve()
    root = workdir / "scale-regression-archive"
    if root.exists():
        shutil.rmtree(root)
    _init_archive(root)

    checks = (
        _check_message_budget_chunking(root),
        _check_bounded_giant_session(root),
        _check_reset_preserves_source(root),
        _check_run_ref_no_drop(root),
        _check_raw_materialization_backlog(root),
        _check_insights_stage_resumable(root),
    )
    ok = all(check.ok for check in checks)
    report = ScaleRegressionReport(
        ok=ok,
        duration_ms=int((time.perf_counter() - started_at) * 1000),
        archive_root=str(root),
        checks=checks,
    )
    if not keep and ok:
        shutil.rmtree(root, ignore_errors=True)
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, default=Path(".cache") / "scale-regression")
    parser.add_argument("--keep", action="store_true", help="Keep the generated archive after the probe")
    parser.add_argument("--json", action="store_true", help="Emit the stable JSON payload")
    return parser


def main(argv: list[str] | None = None, *, stdout: TextIO | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    out = stdout
    if out is None:
        import sys

        out = sys.stdout
    report = run_scale_regression_probe(args.workdir, keep=bool(args.keep))
    payload = report.to_payload()
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True), file=out)
    else:
        status = "ok" if report.ok else "failed"
        print(f"scale-regression: {status} in {report.duration_ms} ms", file=out)
        for check in report.checks:
            print(f"- {check.name}: {'ok' if check.ok else 'failed'}", file=out)
    return 0 if report.ok else 1


__all__ = ["ScaleRegressionCheck", "ScaleRegressionReport", "main", "run_scale_regression_probe"]


if __name__ == "__main__":
    raise SystemExit(main())
