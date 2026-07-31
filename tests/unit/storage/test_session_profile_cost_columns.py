"""session_profiles.cost_usd/cost_credits/priced_with/priced_at_ms must not be
permanently NULL (polylogue-f2qv.6).

Live-archive census: all four columns were 100% NULL across every
session_profiles row, despite session_model_usage (the single per-model cost
authority) carrying real catalog-priced cost_usd values for the same
sessions. Root cause, traced to source: ``upsert_session_profile_costs``
(``storage/sqlite/archive_tiers/write.py``) is the only writer ever declared
for these four columns, and it has zero production callers -- the value was
never computed and dropped; it was simply never wired into the session-profile
materialization pipeline (``build_session_profile`` ->
``build_session_profile_record`` -> ``session_profile_insert_values`` ->
``_SESSION_PROFILE_BASE_COLUMNS``), which never referenced them at all.

These tests exercise the real production write path
(``write_parsed_session_to_archive``) and the real session-insight
materializer (``rebuild_session_insights_sync``) -- the same two-step
pipeline every other session_profiles column already goes through -- not a
toy replica of the aggregation logic.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from polylogue.archive.message.roles import Role
from polylogue.core.enums import BlockType, Provider
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.storage.insights.session.rebuild import rebuild_session_insights_sync
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from polylogue.storage.sqlite.archive_tiers.write import write_parsed_session_to_archive


def _make_archive_conn(tmp_path: Path) -> sqlite3.Connection:
    initialize_active_archive_root(tmp_path)
    conn = sqlite3.connect(tmp_path / "index.db")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _priced_claude_code_session(session_id: str) -> ParsedSession:
    """A Claude Code session using a model with a real catalog price entry."""
    return ParsedSession(
        source_name=Provider.CLAUDE_CODE,
        provider_session_id=session_id,
        title="cost column population test",
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


def _unpriced_model_session(session_id: str) -> ParsedSession:
    """A session whose only model has no catalog price entry at all."""
    return ParsedSession(
        source_name=Provider.CLAUDE_CODE,
        provider_session_id=session_id,
        title="cost column no-fabrication test",
        models_used=["totally-unknown-model-xyz"],
        messages=[
            ParsedMessage(
                provider_message_id="a1",
                role=Role.ASSISTANT,
                text="done",
                model_name="totally-unknown-model-xyz",
                input_tokens=1_000,
                output_tokens=500,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="done")],
            ),
        ],
    )


def _cost_columns(conn: sqlite3.Connection, session_id: str) -> sqlite3.Row:
    row = conn.execute(
        "SELECT cost_usd, cost_credits, priced_with, priced_at_ms, total_cost_usd "
        "FROM session_profiles WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    assert row is not None, f"no session_profiles row for {session_id}"
    return row


class TestSessionProfileCostColumnsPopulated:
    def test_priced_model_populates_all_four_cost_columns(self, tmp_path: Path) -> None:
        conn = _make_archive_conn(tmp_path)
        session_id = "claude-code-session:cost-columns-priced"
        write_parsed_session_to_archive(conn, _priced_claude_code_session("cost-columns-priced"))

        rebuild_session_insights_sync(conn, session_ids=[session_id])

        row = _cost_columns(conn, session_id)
        assert row["cost_usd"] is not None, "cost_usd must be populated for a catalog-priced model"
        assert row["cost_credits"] is not None, "cost_credits must be populated for a catalog-priced model"
        assert row["priced_with"] is not None, "priced_with must record the catalog version that priced this row"
        assert row["priced_at_ms"] is not None, "priced_at_ms must record when this row was priced"
        assert row["cost_usd"] > 0
        # cost_usd mirrors the same session-level total already computed for
        # total_cost_usd -- both derive from the identical cost_summary.
        assert row["cost_usd"] == row["total_cost_usd"]
        conn.close()

    def test_unpriced_model_leaves_cost_columns_null_not_fabricated_zero(self, tmp_path: Path) -> None:
        conn = _make_archive_conn(tmp_path)
        session_id = "claude-code-session:cost-columns-unpriced"
        write_parsed_session_to_archive(conn, _unpriced_model_session("cost-columns-unpriced"))

        rebuild_session_insights_sync(conn, session_ids=[session_id])

        row = _cost_columns(conn, session_id)
        assert row["cost_usd"] is None, "no catalog price exists for this model -- must stay NULL, not a fake $0.00"
        assert row["cost_credits"] is None
        assert row["priced_with"] is None
        assert row["priced_at_ms"] is None
        conn.close()

    def test_priced_at_ms_advances_on_rebuild(self, tmp_path: Path) -> None:
        """A later rebuild re-prices the row with a fresh timestamp, proving
        priced_at_ms is recomputed each materialization, not a frozen value."""
        conn = _make_archive_conn(tmp_path)
        session_id = "claude-code-session:cost-columns-repriced"
        write_parsed_session_to_archive(conn, _priced_claude_code_session("cost-columns-repriced"))

        rebuild_session_insights_sync(conn, session_ids=[session_id])
        first = _cost_columns(conn, session_id)
        assert first["priced_at_ms"] is not None

        rebuild_session_insights_sync(conn, session_ids=[session_id])
        second = _cost_columns(conn, session_id)
        assert second["priced_at_ms"] is not None
        assert second["priced_at_ms"] >= first["priced_at_ms"]
        conn.close()
