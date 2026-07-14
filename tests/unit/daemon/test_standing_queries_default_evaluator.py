"""End-to-end proof of the default daemon evaluator injection (polylogue-rxdo.5).

Every earlier standing-query test injects a hand-rolled fake evaluator.
``make_default_convergence_stages`` previously called
``make_standing_query_stage(db_path)`` with no evaluator at all, which left
the stage permanently inert in production (``check_sessions``/
``execute_sessions`` both short-circuit to a no-op when ``evaluator is
None``). This test proves the *production* wiring: no fake is injected here,
only the real ``ArchiveCanonicalPlanEvaluator`` reached through
``make_default_convergence_stages``, running against a real ingested
archive.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from polylogue.archive.message.roles import Role
from polylogue.core.enums import AssertionKind, BlockType, Provider
from polylogue.daemon.convergence import DaemonConverger
from polylogue.daemon.convergence_stages import make_default_convergence_stages
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.query_objects import put_query, put_query_name


def _seed_archive_with_one_codex_session(archive_root: Path) -> str:
    archive_root.mkdir(parents=True, exist_ok=True)
    with ArchiveStore(archive_root) as archive:
        session_id = archive.write_parsed(
            ParsedSession(
                source_name=Provider.CODEX,
                provider_session_id="codex-1",
                title="codex session",
                created_at="2026-01-01T00:00:00+00:00",
                updated_at="2026-01-01T00:01:00+00:00",
                messages=[
                    ParsedMessage(
                        provider_message_id="codex-1-m1",
                        role=Role.USER,
                        text="hello",
                        timestamp="2026-01-01T00:00:00+00:00",
                        blocks=[ParsedContentBlock(type=BlockType.TEXT, text="hello")],
                    )
                ],
            )
        )
    initialize_archive_database(archive_root / "user.db", ArchiveTier.USER)
    initialize_archive_database(archive_root / "ops.db", ArchiveTier.OPS)
    return session_id


def test_default_stage_set_evaluates_a_watched_query_without_an_injected_fake(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    session_id = _seed_archive_with_one_codex_session(archive_root)

    with sqlite3.connect(archive_root / "user.db") as conn:
        query = put_query(
            conn,
            {"kind": "field", "field": "origin", "op": "=", "values": ["codex-session"]},
            grain="session",
            lane="dialogue",
            rank_policy="mixed",
            created_at_ms=1,
        )
        put_query_name(conn, name="codex-watch", query_hash=query.query_hash, watch=True, updated_at_ms=2)
        conn.commit()

    stages = make_default_convergence_stages(archive_root / "index.db")
    standing_stage = next(stage for stage in stages if stage.name == "standing-queries")
    converger = DaemonConverger(stages=(standing_stage,), max_workers=1)
    states, _timings = converger.converge_sessions((session_id,))
    assert states[session_id].stages["standing-queries"].value == "done"

    with sqlite3.connect(archive_root / "user.db") as conn:
        baseline_row = conn.execute("SELECT member_count FROM result_sets WHERE persistence_class = 'watch'").fetchone()
        assert baseline_row is not None
        assert baseline_row[0] == 1
        # First observation only establishes a baseline; there is no prior
        # membership to diff against yet, so no candidate finding fires.
        finding_count = conn.execute(
            "SELECT COUNT(*) FROM assertions WHERE kind = ?", (AssertionKind.FINDING.value,)
        ).fetchone()[0]
        assert finding_count == 0

    with sqlite3.connect(archive_root / "ops.db") as conn:
        run_count = conn.execute("SELECT COUNT(*) FROM query_runs WHERE surface = 'daemon-internal'").fetchone()[0]
        assert run_count == 1
