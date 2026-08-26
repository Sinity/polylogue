"""Profile rows do not persist usage or pricing mirrors."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from polylogue.archive.message.roles import Role
from polylogue.core.enums import BlockType, Provider
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from polylogue.storage.sqlite.archive_tiers.write import write_parsed_session_to_archive
from polylogue.storage.usage import session_usage_costs_for_connection


def _conn(tmp_path: Path) -> sqlite3.Connection:
    initialize_active_archive_root(tmp_path)
    conn = sqlite3.connect(tmp_path / "index.db")
    conn.row_factory = sqlite3.Row
    return conn


def _session(session_id: str, model: str) -> ParsedSession:
    return ParsedSession(
        source_name=Provider.CLAUDE_CODE,
        provider_session_id=session_id,
        title="canonical usage test",
        models_used=[model],
        messages=[
            ParsedMessage(
                provider_message_id="a1",
                role=Role.ASSISTANT,
                text="done",
                model_name=model,
                input_tokens=1_000,
                output_tokens=500,
                cache_read_tokens=200,
                cache_write_tokens=100,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="done")],
            )
        ],
    )


def test_session_profiles_have_no_usage_or_pricing_columns(tmp_path: Path) -> None:
    conn = _conn(tmp_path)
    names = {str(row[1]) for row in conn.execute("PRAGMA table_info(session_profiles)")}
    assert not names & {
        "total_input_tokens",
        "total_output_tokens",
        "total_cache_read_tokens",
        "total_cache_write_tokens",
        "total_cost_usd",
        "total_credit_cost",
        "cost_is_estimated",
        "cost_provenance",
        "per_model_cost_json",
        "cost_usd",
        "cost_credits",
        "priced_with",
        "priced_at_ms",
    }
    conn.close()


def test_canonical_usage_projection_preserves_priced_and_unpriced_states(tmp_path: Path) -> None:
    conn = _conn(tmp_path)
    write_parsed_session_to_archive(conn, _session("priced", "claude-sonnet-4-5"))
    write_parsed_session_to_archive(conn, _session("unpriced", "totally-unknown-model-xyz"))
    ids = [str(row[0]) for row in conn.execute("SELECT session_id FROM sessions ORDER BY session_id")]
    costs = session_usage_costs_for_connection(conn, ids)
    assert costs["claude-code-session:priced"].total_usd is not None
    assert costs["claude-code-session:unpriced"].total_usd is None
    assert costs["claude-code-session:unpriced"].availability == "unpriced"
    conn.close()
