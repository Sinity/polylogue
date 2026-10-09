"""Session-profile token totals must agree with session_model_usage.

polylogue-r7p6: for Codex sessions, session_profiles token columns undercounted
session_model_usage by roughly 1000x (6.43M vs 6.74B input tokens across the
same 3,134 sessions archive-wide, 1,250 profiles reading exactly zero) because
profile building walked ``session.messages`` per-message ``input_tokens``/
``output_tokens`` fields -- populated only when a Codex message record embeds
a ``usage`` block directly, which is rare. Codex's real usage arrives as
periodic cumulative ``token_count`` session events instead, which are
deliberately excluded from ``session_events`` (see
``_SESSION_EVENTS_REDUNDANT_TYPES`` in ``storage/sqlite/archive_tiers/write.py``)
and folded into ``session_model_usage`` by
``_aggregate_provider_usage_into_model_usage`` instead. Session-profile
building never read that table, so it silently recomputed a near-empty
estimate from the wrong source.

The fix (``ModelUsageTotals`` plumbed through ``compute_session_cost`` /
``build_session_profile`` / ``build_session_insight_records``) makes profile
building read ``session_model_usage`` back directly -- the same substrate the
archive's own cost/usage rollups are built from -- so profile totals are
identical to that rollup by construction, for every origin, not just Codex.

``session_profiles`` stores no token or cost columns; ``session_model_usage``
is the sole authority and the reader overlays it onto each record. These tests
assert on the composed read path, so they go red if the overlay is dropped or
the write path stops populating ``session_model_usage``.

These tests exercise the real production write path
(``write_parsed_session_to_archive``) and both session-insight materializer
twins (``rebuild_session_insights_sync`` / ``rebuild_session_insights_async``),
not a toy replica of the aggregation logic.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import aiosqlite
import pytest

from polylogue.archive.message.roles import Role
from polylogue.core.enums import BlockType, Provider
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession, ParsedSessionEvent
from polylogue.storage.derived.session.profile_cost import (
    apply_profile_cost_lanes,
    read_model_usage_batch_sync,
)
from polylogue.storage.derived.session.rebuild import (
    rebuild_session_insights_async,
    rebuild_session_insights_sync,
)
from polylogue.storage.io_phase_metrics import connect_measured
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from polylogue.storage.sqlite.queries.mappers import _row_to_session_profile_record
from polylogue.storage.sqlite.write_lease import async_write_lease
from tests.infra.archive_templates import run_off_event_loop
from tests.infra.index_writer import write_fixture_index_session

# Realistic Codex cumulative usage: input is inclusive of cached (96% cached,
# matching the corpus finding in provider_usage_disjoint_lanes's docstring),
# output inclusive of reasoning. Disjoint-lane mapping: fresh_input =
# input - cached = 100_000 - 96_000 = 4_000; output unchanged at 5_000;
# cache_read = 96_000; cache_write = 0.
_CODEX_TOTAL_INPUT = 100_000
_CODEX_TOTAL_CACHED = 96_000
_CODEX_TOTAL_OUTPUT = 5_000
_CODEX_EXPECTED_INPUT = _CODEX_TOTAL_INPUT - _CODEX_TOTAL_CACHED
_CODEX_EXPECTED_OUTPUT = _CODEX_TOTAL_OUTPUT
_CODEX_EXPECTED_CACHE_READ = _CODEX_TOTAL_CACHED
_CODEX_EXPECTED_CACHE_WRITE = 0


def _codex_session(session_id: str) -> ParsedSession:
    """A Codex session whose only real usage evidence is a cumulative
    ``token_count`` event -- the message itself carries no per-message token
    fields, mirroring the archive-shape that caused the ~1000x undercount."""
    return ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id=session_id,
        title="codex model-usage consistency",
        models_used=["gpt-5-codex"],
        messages=[
            ParsedMessage(
                provider_message_id="a1",
                role=Role.ASSISTANT,
                text="done",
                model_name="gpt-5-codex",
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="done")],
                # Deliberately no input_tokens/output_tokens: Codex message
                # records essentially never embed a per-message usage block.
            ),
        ],
        session_events=[
            ParsedSessionEvent(
                event_type="token_count",
                payload={
                    "type": "token_count",
                    "model": "gpt-5-codex",
                    "last_token_usage": {"input_tokens": 50, "output_tokens": 25},
                    "total_token_usage": {
                        "input_tokens": _CODEX_TOTAL_INPUT,
                        "output_tokens": _CODEX_TOTAL_OUTPUT,
                        "cached_input_tokens": _CODEX_TOTAL_CACHED,
                        "cache_write_tokens": 0,
                        "reasoning_output_tokens": 2_000,
                        "total_tokens": _CODEX_TOTAL_INPUT + _CODEX_TOTAL_OUTPUT,
                    },
                },
            ),
        ],
    )


def _claude_code_session(session_id: str) -> ParsedSession:
    """A Claude Code session with ordinary per-message token fields -- the
    shape session_model_usage's message-sum aggregation already handled
    correctly. Used to prove no regression for the origin that was fine."""
    return ParsedSession(
        source_name=Provider.CLAUDE_CODE,
        provider_session_id=session_id,
        title="claude-code model-usage consistency",
        models_used=["claude-sonnet-4-5"],
        messages=[
            ParsedMessage(
                provider_message_id="a1",
                role=Role.ASSISTANT,
                text="done",
                model_name="claude-sonnet-4-5",
                input_tokens=1_000,
                output_tokens=500,
                cache_read_tokens=200,
                cache_write_tokens=100,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="done")],
            ),
        ],
    )


def _make_archive_conn(tmp_path: Path) -> sqlite3.Connection:
    initialize_active_archive_root(tmp_path)
    conn = connect_measured(tmp_path / "index.db")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _model_usage_totals(conn: sqlite3.Connection, session_id: str) -> tuple[int, int, int, int]:
    row = conn.execute(
        """
        SELECT
            COALESCE(SUM(input_tokens), 0),
            COALESCE(SUM(output_tokens), 0),
            COALESCE(SUM(cache_read_tokens), 0),
            COALESCE(SUM(cache_write_tokens), 0)
        FROM session_model_usage
        WHERE session_id = ?
        """,
        (session_id,),
    ).fetchone()
    return (int(row[0]), int(row[1]), int(row[2]), int(row[3]))


def _profile_totals(conn: sqlite3.Connection, session_id: str) -> tuple[int, int, int, int]:
    """Session-profile token totals as a reader sees them.

    ``session_profiles`` stores no token columns; ``session_model_usage`` is
    the sole authority and the reader overlays it onto the profile record.
    This helper composes exactly the production read path
    (``session_insight_profile_reads``) so the assertion covers the overlay a
    consumer of ``SessionProfileRecord`` actually depends on.
    """
    row = conn.execute("SELECT * FROM session_profiles WHERE session_id = ?", (session_id,)).fetchone()
    assert row is not None, f"no session_profiles row for {session_id}"
    record = apply_profile_cost_lanes(
        _row_to_session_profile_record(row),
        read_model_usage_batch_sync(conn, [session_id]),
        {},
    )
    return (
        int(record.total_input_tokens),
        int(record.total_output_tokens),
        int(record.total_cache_read_tokens),
        int(record.total_cache_write_tokens),
    )


def test_codex_profile_tokens_match_model_usage_after_sync_rebuild(tmp_path: Path) -> None:
    conn = _make_archive_conn(tmp_path)
    session_id = "codex-session:model-usage-consistency-sync"
    write_fixture_index_session(conn, _codex_session("model-usage-consistency-sync"))

    # Sanity: session_model_usage carries the real (large) cumulative usage,
    # not the near-empty per-message fields.
    model_usage = _model_usage_totals(conn, session_id)
    assert model_usage == (
        _CODEX_EXPECTED_INPUT,
        _CODEX_EXPECTED_OUTPUT,
        _CODEX_EXPECTED_CACHE_READ,
        _CODEX_EXPECTED_CACHE_WRITE,
    )

    rebuild_session_insights_sync(conn, session_ids=[session_id])

    profile_totals = _profile_totals(conn, session_id)
    assert profile_totals == model_usage
    assert profile_totals == (
        _CODEX_EXPECTED_INPUT,
        _CODEX_EXPECTED_OUTPUT,
        _CODEX_EXPECTED_CACHE_READ,
        _CODEX_EXPECTED_CACHE_WRITE,
    )
    conn.close()


async def test_codex_profile_tokens_match_model_usage_after_async_rebuild(tmp_path: Path) -> None:
    session_id = "codex-session:model-usage-consistency-async"

    def seed() -> tuple[int, int, int, int]:
        # The fixture producer takes its own writer lease off the event loop;
        # it must not inherit the async test's lease.
        sync_conn = _make_archive_conn(tmp_path)
        try:
            write_fixture_index_session(sync_conn, _codex_session("model-usage-consistency-async"))
            return _model_usage_totals(sync_conn, session_id)
        finally:
            sync_conn.close()

    model_usage = run_off_event_loop(seed)
    async with async_write_lease("test.runtime.async-fixture", archive_root=tmp_path):
        async with aiosqlite.connect(tmp_path / "index.db") as conn:
            conn.row_factory = aiosqlite.Row
            await conn.execute("PRAGMA foreign_keys = ON")
            await rebuild_session_insights_async(conn, session_ids=[session_id])
            await conn.commit()

        verify_conn = sqlite3.connect(tmp_path / "index.db")
        verify_conn.row_factory = sqlite3.Row
        try:
            profile_totals = _profile_totals(verify_conn, session_id)
        finally:
            verify_conn.close()

        assert profile_totals == model_usage
        assert profile_totals == (
            _CODEX_EXPECTED_INPUT,
            _CODEX_EXPECTED_OUTPUT,
            _CODEX_EXPECTED_CACHE_READ,
            _CODEX_EXPECTED_CACHE_WRITE,
        )


def test_claude_code_profile_tokens_match_model_usage_no_regression(tmp_path: Path) -> None:
    """Provider-neutral: a Claude Code session (whose per-message tokens were
    already correct) must keep matching session_model_usage after the fix."""
    conn = _make_archive_conn(tmp_path)
    session_id = "claude-code-session:model-usage-consistency"
    write_fixture_index_session(conn, _claude_code_session("model-usage-consistency"))

    model_usage = _model_usage_totals(conn, session_id)
    assert model_usage == (1_000, 500, 200, 100)

    rebuild_session_insights_sync(conn, session_ids=[session_id])

    profile_totals = _profile_totals(conn, session_id)
    assert profile_totals == model_usage
    assert profile_totals == (1_000, 500, 200, 100)
    conn.close()


def test_codex_profile_undercounts_without_model_usage_anti_vacuity(tmp_path: Path) -> None:
    """Anti-vacuity: reverting to the pre-fix behavior (build the profile from
    only the hydrated Session, with no model_usage rows supplied) reproduces
    the ~1000x undercount this bead reports -- proving the fix, not a fixture
    artifact, is what makes the two prior tests pass."""
    from polylogue.archive.session.session_profile import build_session_profile
    from polylogue.storage.derived.session.rebuild import hydrate_sessions, load_sync_batch

    conn = _make_archive_conn(tmp_path)
    session_id = "codex-session:model-usage-consistency-anti-vacuity"
    write_fixture_index_session(conn, _codex_session("model-usage-consistency-anti-vacuity"))
    model_usage = _model_usage_totals(conn, session_id)
    assert model_usage[0] == _CODEX_EXPECTED_INPUT

    # Same hydration path production uses (rebuild.py's load_sync_batch +
    # hydrate_sessions), so this is the real Session object, not a stub.
    batch = load_sync_batch(conn, [session_id])
    (session,) = hydrate_sessions(batch)

    # Old behavior: no model_usage supplied -> falls back to walking
    # session.messages. The message's own token fields are all zero for this
    # fixture, so the only signal left is a crude word-count heuristic
    # estimate off "done" -- a handful of tokens, not the ~4,000 real ones
    # (a ~1000x undercount, matching the bead's reported ratio).
    undercounted_profile = build_session_profile(session)
    assert undercounted_profile.total_input_tokens < 10
    assert undercounted_profile.total_input_tokens != _CODEX_EXPECTED_INPUT

    # Fixed behavior: model_usage supplied -> matches session_model_usage.
    from polylogue.archive.semantic.cost_records import ModelUsageTotals

    corrected_profile = build_session_profile(
        session,
        model_usage=[
            ModelUsageTotals(
                model_name="gpt-5-codex",
                input_tokens=model_usage[0],
                output_tokens=model_usage[1],
                cache_read_tokens=model_usage[2],
                cache_write_tokens=model_usage[3],
                provider_lanes_complete=True,
            )
        ],
    )
    assert corrected_profile.total_input_tokens == _CODEX_EXPECTED_INPUT
    assert corrected_profile.total_output_tokens == _CODEX_EXPECTED_OUTPUT
    assert corrected_profile.total_cache_read_tokens == _CODEX_EXPECTED_CACHE_READ
    assert corrected_profile.total_cache_write_tokens == _CODEX_EXPECTED_CACHE_WRITE
    conn.close()


@pytest.mark.parametrize("route", ["single", "batch", "list"])
async def test_profile_rows_and_usage_overlay_share_one_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, route: str
) -> None:
    """Anti-vacuity: without one snapshot, a commit between the two reads pairs
    the old profile title with the replaced usage row (9000 input tokens)."""

    def seed() -> str:
        # The fixture producer takes its own writer lease off the event loop;
        # it must not inherit the async test's lease.
        conn = _make_archive_conn(tmp_path)
        try:
            # WAL lets the interleaved writer commit while the reader holds its snapshot.
            conn.execute("PRAGMA journal_mode = WAL")
            seeded = write_fixture_index_session(conn, _claude_code_session("profile-snapshot"))
            rebuild_session_insights_sync(conn, session_ids=[seeded])
            conn.commit()
            return seeded
        finally:
            conn.close()

    session_id = run_off_event_loop(seed)
    async with async_write_lease("test.runtime.async-fixture", archive_root=tmp_path):
        from polylogue.archive.semantic.cost_records import ModelUsageTotals
        from polylogue.storage.derived.session.profile_cost import read_model_usage_batch_async as read_usage
        from polylogue.storage.query_models import SessionProfileListQuery
        from polylogue.storage.sqlite.queries import session_insight_profile_reads

        writer = connect_measured(tmp_path / "index.db")
        writer.row_factory = sqlite3.Row
        try:
            title = writer.execute("SELECT title FROM session_profiles WHERE session_id = ?", (session_id,)).fetchone()[
                0
            ]
            committed = False

            async def interleaved(conn: aiosqlite.Connection, ids: list[str]) -> dict[str, list[ModelUsageTotals]]:
                nonlocal committed
                if not committed:
                    committed = True
                    writer.execute(
                        "UPDATE session_profiles SET title = 'replacement' WHERE session_id = ?", (session_id,)
                    )
                    writer.execute(
                        "UPDATE session_model_usage SET input_tokens = 9000 WHERE session_id = ?", (session_id,)
                    )
                    writer.commit()
                return await read_usage(conn, ids)

            monkeypatch.setattr(session_insight_profile_reads, "read_model_usage_batch_async", interleaved)
            async with aiosqlite.connect(tmp_path / "index.db") as reader:
                reader.row_factory = aiosqlite.Row
                if route == "single":
                    profile = await session_insight_profile_reads.get_session_profile(reader, session_id)
                elif route == "batch":
                    profile = (await session_insight_profile_reads.get_session_profiles_batch(reader, [session_id]))[
                        session_id
                    ]
                else:
                    (profile,) = await session_insight_profile_reads.list_session_profiles(
                        reader, SessionProfileListQuery()
                    )
                assert not reader.in_transaction
            assert committed
            assert profile is not None
            assert profile.title == title
            assert profile.total_input_tokens == 1_000
        finally:
            writer.close()
