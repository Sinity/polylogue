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

The second half of this module (polylogue-pm8cj) proves the *creation* route:
a watch reached only by the product surface, never by a direct
``put_query_name`` write.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import cast

import pytest

from polylogue.archive.message.roles import Role
from polylogue.core.enums import AssertionKind, BlockType, Provider
from polylogue.daemon.convergence import DaemonConverger
from polylogue.daemon.convergence_stages import make_default_convergence_stages
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.query_objects import put_query, put_query_name
from tests.infra.archive_templates import run_off_event_loop
from tests.infra.compute_owner import owned_compute_adapter
from tests.infra.live_ingest import write_index_session


def _seed_archive_with_one_codex_session(archive_root: Path) -> str:
    return run_off_event_loop(lambda: _seed_archive_with_one_codex_session_sync(archive_root))


def _seed_archive_with_one_codex_session_sync(archive_root: Path) -> str:
    archive_root.mkdir(parents=True, exist_ok=True)
    with ArchiveStore(archive_root) as archive:
        session_id = write_index_session(
            archive,
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
            ),
        )
    initialize_archive_database(archive_root / "user.db", ArchiveTier.USER)
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

    with owned_compute_adapter() as compute:
        stages = make_default_convergence_stages(archive_root / "index.db", compute_adapter=compute)
        standing_stage = next(stage for stage in stages if stage.name == "standing-queries")
        converger = DaemonConverger(stages=(standing_stage,))
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


# ---------------------------------------------------------------------------
# Production creation route (polylogue-pm8cj)
# ---------------------------------------------------------------------------
#
# The test above still seeds its watch with a direct ``put_query_name`` write,
# which is exactly why the missing creation route went unnoticed for two
# months: every standing-query test manufactured the row the stage reads, so
# no test could observe that ``put_query_name(..., watch=True)`` had no
# production caller at all. These tests never touch ``query_names``; they mark
# a saved view watched through the MCP ``write`` tool -- the declared product
# surface -- and then run the real convergence stage set over it.


def _seed_second_codex_session_on_writer(archive_root: Path) -> str:
    with ArchiveStore(archive_root) as archive:
        return write_index_session(
            archive,
            ParsedSession(
                source_name=Provider.CODEX,
                provider_session_id="codex-2",
                title="second codex session",
                created_at="2026-01-02T00:00:00+00:00",
                updated_at="2026-01-02T00:01:00+00:00",
                messages=[
                    ParsedMessage(
                        provider_message_id="codex-2-m1",
                        role=Role.USER,
                        text="hello again",
                        timestamp="2026-01-02T00:00:00+00:00",
                        blocks=[ParsedContentBlock(type=BlockType.TEXT, text="hello again")],
                    )
                ],
            ),
        )


def _seed_second_codex_session(archive_root: Path) -> str:
    """Run the synchronous seed off any running event loop."""
    return run_off_event_loop(lambda: _seed_second_codex_session_on_writer(archive_root))


@contextmanager
def _saved_view_writer(archive_root: Path) -> Iterator[Callable[..., Awaitable[dict[str, object]]]]:
    """Yield the real MCP ``write`` tool's ``save_saved_view`` against a seeded archive.

    Public archive writes are daemon-owned (#5550), so the calls are served by
    one production operation stack on the archive's own socket for the
    duration of the block.
    """
    from unittest.mock import patch

    from polylogue.daemon.socket_path import daemon_socket_path
    from polylogue.mcp.declarations.models import MCPCapabilities
    from polylogue.mcp.server import build_server
    from tests.infra.daemon_operations import running_daemon_operations
    from tests.infra.mcp import MCPServerUnderTest, installed_runtime_services, invoke_surface_async

    server = cast(MCPServerUnderTest, build_server(capabilities=MCPCapabilities(write=True)))
    write_fn = server._tool_manager._tools["write"].fn

    async def write(**fields: object) -> dict[str, object]:
        payload = await invoke_surface_async(write_fn, operation="save_saved_view", fields=fields)
        return cast(dict[str, object], json.loads(payload))

    with (
        patch("polylogue.daemon.api_auth.resolve_api_auth_token", return_value=None),
        running_daemon_operations(archive_root, socket_path=daemon_socket_path(archive_root)),
        installed_runtime_services(archive_root),
    ):
        yield write


def _converge(archive_root: Path, session_ids: tuple[str, ...]) -> None:
    with owned_compute_adapter() as compute:
        stages = make_default_convergence_stages(archive_root / "index.db", compute_adapter=compute)
        standing_stage = next(stage for stage in stages if stage.name == "standing-queries")
        converger = DaemonConverger(stages=(standing_stage,))
        states, _timings = converger.converge_sessions(session_ids)
        for session_id in session_ids:
            assert states[session_id].stages["standing-queries"].value == "done"


@pytest.mark.asyncio
async def test_saved_view_marked_watched_is_picked_up_by_the_next_convergence_tick(tmp_path: Path) -> None:
    """polylogue-pm8cj: the product can create a watch the daemon then evaluates.

    Anti-vacuity: nothing here writes ``queries`` or ``query_names``. Remove
    the ``register_query_watch`` call from ``ArchiveStore.save_view`` (or the
    ``watch`` field from the MCP write dispatch) and the stage finds no watched
    definition, so no ``watch`` result set and no delta candidate ever appear.
    """
    archive_root = tmp_path / "archive"
    session_id = _seed_archive_with_one_codex_session(archive_root)

    with _saved_view_writer(archive_root) as write:
        saved = await write(
            name="codex sessions",
            query_json=json.dumps({"query": "sessions where origin:codex-session"}),
            watch=True,
        )
    assert saved.get("is_error") is not True, saved

    _converge(archive_root, (session_id,))
    with sqlite3.connect(archive_root / "user.db") as conn:
        watched = conn.execute("SELECT name, query_hash FROM query_names WHERE watch = 1").fetchall()
        assert len(watched) == 1 and watched[0][0] == "codex sessions"
        members = conn.execute("SELECT member_count FROM result_sets WHERE persistence_class = 'watch'").fetchall()
        assert members == [(1,)]
        assert (
            conn.execute("SELECT COUNT(*) FROM assertions WHERE kind = ?", (AssertionKind.FINDING.value,)).fetchone()[0]
            == 0
        )

    second_id = _seed_second_codex_session(archive_root)
    _converge(archive_root, (second_id,))
    with sqlite3.connect(archive_root / "user.db") as conn:
        rows = conn.execute(
            "SELECT value_json FROM assertions WHERE kind = ?", (AssertionKind.FINDING.value,)
        ).fetchall()
        assert len(rows) == 1
        assert '"finding_kind":"query-delta"' in str(rows[0][0])
        assert '"value":2' in str(rows[0][0])


@pytest.mark.asyncio
async def test_first_session_change_after_watch_creation_reports_a_delta(tmp_path: Path) -> None:
    """Creating a watch measures its baseline, so the next change is a delta.

    One matching session exists when the watch is created and a second is
    ingested before any convergence tick. Anti-vacuity: drop the
    ``establish_watch_baselines`` call from ``SavedViewSaveActuator.apply``
    and the tick that sees the second session silently baselines both, so
    no ``query-delta`` finding appears.
    """
    archive_root = tmp_path / "archive"
    _seed_archive_with_one_codex_session(archive_root)

    with _saved_view_writer(archive_root) as write:
        saved = await write(
            name="codex sessions",
            query_json=json.dumps({"query": "sessions where origin:codex-session"}),
            watch=True,
        )
    assert saved.get("is_error") is not True, saved
    with sqlite3.connect(archive_root / "user.db") as conn:
        members = conn.execute("SELECT member_count FROM result_sets WHERE persistence_class = 'watch'").fetchall()
        assert members == [(1,)]

    second_id = _seed_second_codex_session(archive_root)
    _converge(archive_root, (second_id,))
    with sqlite3.connect(archive_root / "user.db") as conn:
        rows = conn.execute(
            "SELECT value_json FROM assertions WHERE kind = ?", (AssertionKind.FINDING.value,)
        ).fetchall()
    assert len(rows) == 1
    assert '"finding_kind":"query-delta"' in str(rows[0][0])
    assert '"value":2' in str(rows[0][0])


@pytest.mark.asyncio
async def test_saving_the_same_view_unwatched_stops_the_next_tick_evaluating_it(tmp_path: Path) -> None:
    """A watch must not outlive the definition its name denotes."""
    archive_root = tmp_path / "archive"
    session_id = _seed_archive_with_one_codex_session(archive_root)
    query_json = json.dumps({"query": "sessions where origin:codex-session"})

    with _saved_view_writer(archive_root) as write:
        saved = await write(name="codex sessions", query_json=query_json, watch=True)
        assert saved.get("is_error") is not True, saved
        unsaved = await write(name="codex sessions", query_json=query_json, view_id=str(saved["key"]), watch=False)
    assert unsaved.get("is_error") is not True, unsaved

    def receipts() -> int:
        with sqlite3.connect(archive_root / "user.db") as conn:
            return int(conn.execute("SELECT COUNT(*) FROM query_evaluation_receipts").fetchone()[0])

    # Creation measured the baseline once; the unwatched name must not be
    # evaluated again by the tick.
    before = receipts()
    _converge(archive_root, (session_id,))
    assert receipts() == before
    with sqlite3.connect(archive_root / "user.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM query_names WHERE watch = 1").fetchone()[0] == 0


@pytest.mark.asyncio
async def test_a_view_the_evaluator_cannot_execute_is_refused_instead_of_watched(tmp_path: Path) -> None:
    """An unwatchable selection must fail loudly, not become a silent no-op watch.

    Both shapes below are accepted as ordinary saved views today. Watching
    either would evaluate a *different* set than the name denotes: bare terms
    have no predicate at all, and ``origin`` as a separate parameter never
    reaches the compiled definition.
    """
    archive_root = tmp_path / "archive"
    _seed_archive_with_one_codex_session(archive_root)

    with _saved_view_writer(archive_root) as write:
        terms = await write(name="bare terms", query_json=json.dumps({"query": "hello"}), watch=True)
        extra = await write(
            name="split filters",
            query_json=json.dumps({"query": "sessions where repo:polylogue", "origin": "codex-session"}),
            watch=True,
        )
    assert terms.get("is_error") is True, terms
    assert terms.get("code") == "invalid_watch_definition", terms
    assert "sessions where" in str(terms.get("message", ""))

    assert extra.get("is_error") is True, extra
    assert "origin" in str(extra.get("message", ""))

    with sqlite3.connect(archive_root / "user.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM query_names").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM queries").fetchone()[0] == 0


@pytest.mark.asyncio
async def test_re_enabled_watch_does_not_report_its_unwatched_interval_as_a_delta(tmp_path: Path) -> None:
    """A watch switched off and on again is measured afresh, not diffed against its old baseline.

    A second matching session arrives while the view is unwatched. Anti-vacuity:
    drop ``retire_unwatched_baselines`` from ``clear_query_watch`` and the
    re-enabled watch keeps its one-member baseline, so the next evaluation
    reports the session ingested while nobody watched as a ``query-delta``.
    """
    archive_root = tmp_path / "archive"
    _seed_archive_with_one_codex_session(archive_root)
    query_json = json.dumps({"query": "sessions where origin:codex-session"})

    with _saved_view_writer(archive_root) as write:
        saved = await write(name="codex sessions", query_json=query_json, watch=True)
        assert saved.get("is_error") is not True, saved
        view_id = str(saved["key"])
        unwatched = await write(name="codex sessions", query_json=query_json, view_id=view_id, watch=False)
        assert unwatched.get("is_error") is not True, unwatched
        second_id = _seed_second_codex_session(archive_root)
        rewatched = await write(name="codex sessions", query_json=query_json, view_id=view_id, watch=True)
        assert rewatched.get("is_error") is not True, rewatched

    with sqlite3.connect(archive_root / "user.db") as conn:
        baseline = conn.execute(
            "SELECT rs.member_count FROM watched_query_baselines AS b "
            "JOIN result_sets AS rs ON rs.result_set_id = b.result_set_id"
        ).fetchall()
    assert baseline == [(2,)]

    _converge(archive_root, (second_id,))
    with sqlite3.connect(archive_root / "user.db") as conn:
        findings = conn.execute(
            "SELECT COUNT(*) FROM assertions WHERE kind = ?", (AssertionKind.FINDING.value,)
        ).fetchone()[0]
    assert findings == 0
