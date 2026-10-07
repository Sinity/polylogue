"""Semantic verification for the deterministic demo archive."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from polylogue.archive.query.transaction import run_archive_read_sync
from polylogue.core.provider_identity import profile_root_for_artifact
from polylogue.scenarios import DEMO_CLAUDE_CODE_SESSION_ID, DEMO_HERMES_SESSION_ID, DEMO_SESSION_IDS
from polylogue.sources.parsers.hermes_identity import (
    profile_key,
    qualified_session_id,
)
from polylogue.sources.source_layout import canonical_session_position

from .constructs import construct_problem_messages, evaluate_demo_constructs
from .models import DemoVerifyResult
from .seed import DEMO_SOURCE_DIRNAME

#: Where the demo writes its Hermes snapshot, below the demo Hermes root.
DEMO_HERMES_SNAPSHOT_POSITION = canonical_session_position("hermes", "demo-00", ".json").as_posix()


def _expected_demo_session_ids(archive_root: Path) -> set[str]:
    """Return the seeded ids, deriving Hermes identity from retained seed evidence.

    Hermes ids carry a profile qualifier hashed from the install root the
    parser saw. The seeded raw row retains that source path, so an
    inode-preserving archive relocation still derives the original qualifier,
    and the archive's own session ids are never used as the oracle.
    """

    hermes_snapshot = _recorded_hermes_source_path(archive_root) or (
        archive_root / DEMO_SOURCE_DIRNAME / "hermes" / DEMO_HERMES_SNAPSHOT_POSITION
    )
    hermes_id = qualified_session_id(
        DEMO_HERMES_SESSION_ID.removeprefix("hermes-session:"),
        profile_key(profile_root_for_artifact(hermes_snapshot)),
    )
    expected_ids = set(DEMO_SESSION_IDS)
    expected_ids.remove(DEMO_HERMES_SESSION_ID)
    expected_ids.add(f"hermes-session:{hermes_id}")
    return expected_ids


def _recorded_hermes_source_path(archive_root: Path) -> Path | None:
    """The one retained raw source path of the seeded Hermes snapshot, if unambiguous."""

    source_db = archive_root / "source.db"
    if not source_db.exists():
        return None
    suffix = f"/{DEMO_SOURCE_DIRNAME}/hermes/{DEMO_HERMES_SNAPSHOT_POSITION}"
    with _connect(source_db) as conn:
        rows = conn.execute(
            "SELECT DISTINCT source_path FROM raw_sessions WHERE substr(source_path, -length(?)) = ? LIMIT 2",
            (suffix, suffix),
        ).fetchall()
    return Path(str(rows[0]["source_path"])) if len(rows) == 1 else None


def _connect(db_path: Path) -> sqlite3.Connection:
    """Open a read-only connection with mapping-style rows for demo checks.

    Verification reads; a plain ``sqlite3.connect`` is a writable open that
    creates a missing tier file and contends with the daemon's writer.
    """
    from polylogue.storage.sqlite.connection_profile import open_readonly_connection

    conn = open_readonly_connection(db_path, validate_schema=False)
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
    check_constructs: bool = True,
) -> DemoVerifyResult:
    """Check semantic demo archive facts without a demo catalog.

    ``check_constructs=False`` skips the declared demo-construct minimums
    (``evaluate_demo_constructs``). Several constructs (provider usage,
    synthetic embeddings, the canonical repo name) are populated by the
    demo-only ``apply_demo_post_ingest_augmentation`` enrichment pass, not by
    ingest itself -- a caller polling for *base ingest convergence* before
    running that enrichment pass must not block on those constructs
    (polylogue-z1c6). ``polylogue demo verify`` and the final post-enrichment
    check both keep the default ``True``.

    Canonical intake records physical paths for the generated fixture files.
    The path check accepts those files beneath this archive's demo source
    directory and reports absolute paths pointing elsewhere.
    """

    problems: list[str] = []
    leaks: list[str] = []
    query_hits: tuple[str, ...] = ()

    try:
        session_count = _session_count(archive_root)
        message_count = _message_count(archive_root)
        rows, hits = run_archive_read_sync(
            archive_root,
            operation="demo.verify.read",
            arguments={"limit": 100, "query": "pytest"},
            work=lambda archive: (
                archive.list_summaries(limit=100),
                archive.search_summaries("pytest", limit=10),
            ),
            page_size=100,
            projection="demo-verification",
        )
        session_ids = {row.session_id for row in rows}
        query_hits = tuple(sorted(dict.fromkeys(hit.session_id for hit in hits)))
        # Every archive read -- including the retained-path oracle, overlays
        # and raw source paths -- stays inside this boundary so a corrupt or
        # partial tier is a structured failure, never a crash.
        expected_ids = _expected_demo_session_ids(archive_root)
        overlay_count = _overlay_count(archive_root)
        raw_source_paths = _raw_source_paths(archive_root) if check_source_path_leaks else ()
        construct_coverage = evaluate_demo_constructs(archive_root) if check_constructs else ()
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

    if session_ids != expected_ids:
        problems.append(f"expected demo sessions {sorted(expected_ids)}, found {sorted(session_ids)}")
    expected_session_count = len(DEMO_SESSION_IDS)
    if session_count != expected_session_count:
        problems.append(f"expected {expected_session_count} sessions, found {session_count}")
    if message_count < 31:
        problems.append(f"expected at least 31 messages, found {message_count}")
    if DEMO_CLAUDE_CODE_SESSION_ID not in query_hits:
        problems.append(f"expected pytest query to include {DEMO_CLAUDE_CODE_SESSION_ID}, found {list(query_hits)}")

    overlays_present = overlay_count >= 4
    if require_overlays and not overlays_present:
        problems.append("expected demo overlays, found none")

    if check_constructs:
        problems.extend(construct_problem_messages(construct_coverage))

    if check_source_path_leaks:
        demo_source_root = (archive_root / DEMO_SOURCE_DIRNAME).resolve()
        for raw_path in raw_source_paths:
            path = Path(raw_path)
            if path.is_absolute() and not path.resolve().is_relative_to(demo_source_root):
                leaks.append(raw_path)
        if leaks:
            problems.append("raw source paths contain absolute paths outside the demo fixture")

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
