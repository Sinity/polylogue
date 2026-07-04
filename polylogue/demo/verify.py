"""Semantic verification for the deterministic demo archive."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from polylogue.scenarios import DEMO_CLAUDE_CODE_SESSION_ID, DEMO_SESSION_IDS
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

from .constructs import construct_problem_messages, evaluate_demo_constructs
from .models import DemoVerifyResult


def _connect(db_path: Path) -> sqlite3.Connection:
    """Open a SQLite connection with mapping-style rows for demo checks."""

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _session_count(root: Path) -> int:
    """Return the number of normalized sessions in the demo index tier."""

    with _connect(root / "index.db") as conn:
        return int(conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0])


def _message_count(root: Path) -> int:
    """Return the number of normalized messages in the demo index tier."""

    with _connect(root / "index.db") as conn:
        return int(conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0])


def _raw_source_paths(root: Path) -> tuple[str, ...]:
    """Return stored raw source paths so verification can reject leaks."""

    with _connect(root / "source.db") as conn:
        rows = conn.execute("SELECT source_path FROM raw_sessions ORDER BY origin, native_id").fetchall()
    return tuple(str(row["source_path"]) for row in rows)


def _overlay_count(root: Path) -> int:
    """Return deterministic demo assertion count for the Claude Code seed."""

    user_db = root / "user.db"
    if not user_db.exists():
        return 0
    with _connect(user_db) as conn:
        return int(
            conn.execute(
                "SELECT COUNT(*) FROM assertions WHERE target_ref = ?",
                (f"session:{DEMO_CLAUDE_CODE_SESSION_ID}",),
            ).fetchone()[0]
        )


def verify_demo_archive(
    archive_root: Path,
    *,
    require_overlays: bool = False,
    check_source_path_leaks: bool = True,
) -> DemoVerifyResult:
    """Check semantic demo archive facts without a demo catalog."""

    problems: list[str] = []
    leaks: list[str] = []
    query_hits: tuple[str, ...] = ()

    try:
        session_count = _session_count(archive_root)
        message_count = _message_count(archive_root)
        with ArchiveStore.open_existing(archive_root, read_only=True) as archive:
            rows = archive.list_summaries(limit=100)
            session_ids = {row.session_id for row in rows}
            query_hits = tuple(
                sorted(dict.fromkeys(hit.session_id for hit in archive.search_summaries("pytest", limit=10)))
            )
    except (OSError, sqlite3.Error) as exc:
        return DemoVerifyResult(
            archive_root=archive_root,
            ok=False,
            session_count=0,
            message_count=0,
            query_hits=(),
            overlays_present=False,
            absolute_path_leaks=(),
            construct_coverage=(),
            problems=(f"archive unreadable: {exc}",),
        )

    expected_ids = set(DEMO_SESSION_IDS)
    if session_ids != expected_ids:
        problems.append(f"expected demo sessions {sorted(expected_ids)}, found {sorted(session_ids)}")
    expected_session_count = len(DEMO_SESSION_IDS)
    if session_count != expected_session_count:
        problems.append(f"expected {expected_session_count} sessions, found {session_count}")
    if message_count < 31:
        problems.append(f"expected at least 31 messages, found {message_count}")
    if DEMO_CLAUDE_CODE_SESSION_ID not in query_hits:
        problems.append(f"expected pytest query to include {DEMO_CLAUDE_CODE_SESSION_ID}, found {list(query_hits)}")

    overlay_count = _overlay_count(archive_root)
    overlays_present = overlay_count >= 4
    if require_overlays and not overlays_present:
        problems.append("expected demo overlays, found none")

    construct_coverage = evaluate_demo_constructs(archive_root)
    problems.extend(construct_problem_messages(construct_coverage))

    if check_source_path_leaks:
        for raw_path in _raw_source_paths(archive_root):
            if Path(raw_path).is_absolute():
                leaks.append(raw_path)
        if leaks:
            problems.append("raw source paths contain absolute paths")

    return DemoVerifyResult(
        archive_root=archive_root,
        ok=not problems,
        session_count=session_count,
        message_count=message_count,
        query_hits=query_hits,
        overlays_present=overlays_present,
        absolute_path_leaks=tuple(leaks),
        construct_coverage=construct_coverage,
        problems=tuple(problems),
    )


__all__ = ["verify_demo_archive"]
