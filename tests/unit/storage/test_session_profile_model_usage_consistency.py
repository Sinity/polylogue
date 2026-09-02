"""session_profiles token/cost columns must agree with session_model_usage.

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
archive's own cost/usage rollups are built from -- so a profile's token lanes
are identical to that rollup by construction, for every origin, not just Codex.

``session_model_usage`` is the single authority: #4225 removed the duplicated
``total_*_tokens`` columns from ``session_profiles``, so these tests assert
through the derivation rather than by selecting columns that no longer exist --
``build_session_profile`` fed from the persisted usage rows, and the repository
read route that projects the same rollup back onto the record. The write path is
the production one (``write_parsed_session_to_archive``) and the hydration is
the materializer's own (``load_sync_batch`` / ``hydrate_sessions``), not a toy
replica of the aggregation logic.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import aiosqlite

from polylogue.archive.message.roles import Role
from polylogue.archive.semantic.cost_compute import compute_session_cost
from polylogue.archive.semantic.cost_records import ModelUsageTotals
from polylogue.archive.session.session_profile import build_session_profile
from polylogue.core.enums import BlockType, Provider
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession, ParsedSessionEvent
from polylogue.storage.insights.session.rebuild import (
    hydrate_sessions,
    load_sync_batch,
    rebuild_session_insights_sync,
)
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from polylogue.storage.sqlite.archive_tiers.write import write_parsed_session_to_archive
from polylogue.storage.sqlite.queries.session_insight_profile_reads import (
    get_session_profile,
    get_session_profiles_batch,
)

# Realistic Codex cumulative usage: input is inclusive of cached (96% cached,
# matching the corpus finding in _provider_usage_disjoint_lanes's docstring),
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
    conn = sqlite3.connect(tmp_path / "index.db")
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


def _model_usage_rows(conn: sqlite3.Connection, session_id: str) -> list[ModelUsageTotals]:
    rows = conn.execute(
        """
        SELECT model_name, input_tokens, output_tokens, cache_read_tokens, cache_write_tokens
        FROM session_model_usage
        WHERE session_id = ?
        """,
        (session_id,),
    ).fetchall()
    return [
        ModelUsageTotals(
            model_name=str(row[0]),
            input_tokens=int(row[1]),
            output_tokens=int(row[2]),
            cache_read_tokens=int(row[3]),
            cache_write_tokens=int(row[4]),
        )
        for row in rows
    ]


def _derived_profile_totals(conn: sqlite3.Connection, session_id: str) -> tuple[int, int, int, int]:
    """Token lanes as the profile read model derives them.

    This is the production derivation: hydrate the session through the
    materializer's own batch loader, then build its profile from the persisted
    ``session_model_usage`` rows -- the same substrate the archive's cost and
    usage rollups read.
    """
    batch = load_sync_batch(conn, [session_id])
    (session,) = hydrate_sessions(batch)
    rows = conn.execute(
        """
        SELECT model_name, input_tokens, output_tokens, cache_read_tokens, cache_write_tokens
        FROM session_model_usage
        WHERE session_id = ?
        """,
        (session_id,),
    ).fetchall()
    profile = build_session_profile(
        session,
        model_usage=[
            ModelUsageTotals(
                model_name=str(row[0]),
                input_tokens=int(row[1]),
                output_tokens=int(row[2]),
                cache_read_tokens=int(row[3]),
                cache_write_tokens=int(row[4]),
            )
            for row in rows
        ],
    )
    return (
        profile.total_input_tokens,
        profile.total_output_tokens,
        profile.total_cache_read_tokens,
        profile.total_cache_write_tokens,
    )


def test_codex_cumulative_token_count_event_drives_the_profile_token_lanes(tmp_path: Path) -> None:
    """A Codex session whose only usage evidence is a cumulative ``token_count``
    event reports the disjoint lanes, not a word-count estimate.

    Anti-vacuity: the message carries no per-message token fields, so a profile
    builder that walked ``session.messages`` instead of reading
    ``session_model_usage`` back would report a handful of tokens here -- the
    ~1000x undercount polylogue-r7p6 reports -- rather than the 4,000 fresh
    input tokens the event's totals imply.
    """
    conn = _make_archive_conn(tmp_path)
    session_id = "codex-session:model-usage-consistency"
    write_parsed_session_to_archive(conn, _codex_session("model-usage-consistency"))

    expected = (
        _CODEX_EXPECTED_INPUT,
        _CODEX_EXPECTED_OUTPUT,
        _CODEX_EXPECTED_CACHE_READ,
        _CODEX_EXPECTED_CACHE_WRITE,
    )
    assert _model_usage_totals(conn, session_id) == expected
    assert _derived_profile_totals(conn, session_id) == expected
    conn.close()


def test_claude_code_per_message_tokens_reach_the_profile_unchanged(tmp_path: Path) -> None:
    """Provider-neutral: an origin whose per-message tokens were already correct
    keeps matching ``session_model_usage`` through the same derivation.

    Anti-vacuity: a derivation that only special-cased Codex cumulative events
    would drop these per-message lanes to zero.
    """
    conn = _make_archive_conn(tmp_path)
    session_id = "claude-code-session:model-usage-consistency"
    write_parsed_session_to_archive(conn, _claude_code_session("model-usage-consistency"))

    assert _model_usage_totals(conn, session_id) == (1_000, 500, 200, 100)
    assert _derived_profile_totals(conn, session_id) == (1_000, 500, 200, 100)
    conn.close()


async def test_profile_read_route_recomputes_cost_lanes_from_session_model_usage(tmp_path: Path) -> None:
    """The repository profile read route recomputes cost from usage evidence.

    #4225 left ``session_profiles`` with no cost or token columns at all, so a
    reader that selects only that table's own columns reports zero tokens and a
    fabricated $0.00 -- and every consumer of the record, portfolio and
    postmortem token and cost lanes among them, inherits it.

    Anti-vacuity: removing the ``apply_profile_cost_lanes`` overlay drops both
    routes below back to the mapper's ``_row_int(row, ..., 0)`` defaults, so the
    token lanes read 0 and ``cost_provenance`` reads the record default rather
    than the value ``compute_session_cost`` derives from the same rows.
    """
    conn = _make_archive_conn(tmp_path)
    session_id = "codex-session:model-usage-read-route"
    write_parsed_session_to_archive(conn, _codex_session("model-usage-read-route"))
    rebuild_session_insights_sync(conn, session_ids=[session_id])
    conn.commit()
    usage_rows = _model_usage_rows(conn, session_id)
    conn.close()

    # The one implementation of the vocabulary, fed the same rows the route reads.
    expected = compute_session_cost(None, model_usage=usage_rows, estimate_if_missing=False)
    assert expected.total_input_tokens == _CODEX_EXPECTED_INPUT

    async with aiosqlite.connect(tmp_path / "index.db") as read_conn:
        read_conn.row_factory = aiosqlite.Row
        record = await get_session_profile(read_conn, session_id)
        batch = await get_session_profiles_batch(read_conn, [session_id])

    assert record is not None
    for source in (record, batch[session_id]):
        assert (
            source.total_input_tokens,
            source.total_output_tokens,
            source.total_cache_read_tokens,
            source.total_cache_write_tokens,
        ) == (
            _CODEX_EXPECTED_INPUT,
            _CODEX_EXPECTED_OUTPUT,
            _CODEX_EXPECTED_CACHE_READ,
            _CODEX_EXPECTED_CACHE_WRITE,
        )
        assert source.total_cost_usd == expected.total_api_cost_usd
        assert source.total_credit_cost == expected.total_credit_cost
        assert source.cost_provenance == expected.cost_provenance
        assert source.cost_is_estimated is (expected.cost_provenance != "provider_reported")


async def test_profile_read_route_reports_unknown_cost_without_usage_evidence(tmp_path: Path) -> None:
    """A session with no usage rows reports absent evidence, not a $0.00 bill.

    Anti-vacuity: a recompute that treated an empty usage set as a priced zero
    would report ``cost_provenance`` as a pricing token and
    ``cost_is_estimated`` False, claiming the archive knows this session cost
    nothing.
    """
    conn = _make_archive_conn(tmp_path)
    session_id = "claude-code-session:no-usage-evidence"
    write_parsed_session_to_archive(conn, _claude_code_session("no-usage-evidence"))
    rebuild_session_insights_sync(conn, session_ids=[session_id])
    conn.commit()
    conn.execute("DELETE FROM session_model_usage WHERE session_id = ?", (session_id,))
    conn.commit()
    conn.close()

    async with aiosqlite.connect(tmp_path / "index.db") as read_conn:
        read_conn.row_factory = aiosqlite.Row
        record = await get_session_profile(read_conn, session_id)

    assert record is not None
    assert record.cost_provenance == "unknown"
    assert record.total_cost_usd == 0.0
    assert record.cost_is_estimated is True
    assert record.total_input_tokens == 0
