"""Round-trip tests: pricing chain from messages → session_model_usage.

Verifies that:
1. price_catalogs / model_prices are seeded during archive-init.
2. After writing a session with token-bearing messages, session_model_usage
   carries the correct per-model token sums.
3. cost_usd is computed correctly for known models (rate × tokens).
4. Models with no price entry get cost_usd = NULL / priced_with = NULL.
5. Messages with NULL/empty model_name are excluded from aggregation.
"""

from __future__ import annotations

import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest

from polylogue.archive.semantic.pricing import (
    CATALOG_PROVENANCE,
    PRICING,
    estimate_cost,
)
from polylogue.core.enums import Provider
from polylogue.sources.parsers.base import ParsedMessage, ParsedSession
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from polylogue.storage.sqlite.archive_tiers.pricing_seed import (
    _catalog_hash,
    _catalog_id,
    active_price_catalog_id,
    seed_price_catalog,
)
from polylogue.storage.sqlite.archive_tiers.write import write_parsed_session_to_archive


def _make_archive(tmp_path: Path) -> sqlite3.Connection:
    """Initialize a fresh archive root and return an open connection to index.db."""
    initialize_active_archive_root(tmp_path)
    conn = sqlite3.connect(tmp_path / "index.db")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _msg(
    *,
    provider_message_id: str,
    role: str = "assistant",
    model_name: str | None = None,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
    text: str = "hello",
) -> ParsedMessage:
    return ParsedMessage(
        provider_message_id=provider_message_id,
        role=role,  # type: ignore[arg-type]
        text=text,
        model_name=model_name,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_write_tokens=cache_write_tokens,
    )


def _session(
    *,
    provider_session_id: str = "test-session-1",
    messages: list[ParsedMessage],
    reported_cost_usd: float | None = None,
) -> ParsedSession:
    return ParsedSession(
        source_name=Provider.CLAUDE_CODE,
        provider_session_id=provider_session_id,
        title="test session",
        messages=messages,
        reported_cost_usd=reported_cost_usd,
    )


# ---------------------------------------------------------------------------
# Part A: catalog seed
# ---------------------------------------------------------------------------


class TestPriceCatalogSeed:
    def test_price_catalogs_row_present_after_init(self, tmp_path: Path) -> None:
        conn = _make_archive(tmp_path)
        rows = conn.execute("SELECT * FROM price_catalogs").fetchall()
        assert len(rows) == 1
        row = rows[0]
        assert row["catalog_id"] == _catalog_id()
        assert row["source_name"] == CATALOG_PROVENANCE
        conn.close()

    def test_model_prices_seeded_from_pricing_dict(self, tmp_path: Path) -> None:
        conn = _make_archive(tmp_path)
        catalog_id = _catalog_id()
        rows = conn.execute("SELECT model_name FROM model_prices WHERE catalog_id = ?", (catalog_id,)).fetchall()
        seeded_models = {row["model_name"] for row in rows}
        assert seeded_models == set(PRICING.keys()), f"seeded {len(seeded_models)} models, expected {len(PRICING)}"
        conn.close()

    def test_model_prices_rates_match_in_memory_dict(self, tmp_path: Path) -> None:
        conn = _make_archive(tmp_path)
        catalog_id = _catalog_id()
        for model_name, pricing in PRICING.items():
            row = conn.execute(
                """
                SELECT input_cost_per_million, output_cost_per_million,
                       cache_read_cost_per_million, cache_write_cost_per_million
                FROM model_prices
                WHERE catalog_id = ? AND model_name = ?
                """,
                (catalog_id, model_name),
            ).fetchone()
            assert row is not None, f"missing price row for {model_name}"
            assert row["input_cost_per_million"] == pricing.input_usd_per_1m
            assert row["output_cost_per_million"] == pricing.output_usd_per_1m
        conn.close()

    def test_seed_is_idempotent(self, tmp_path: Path) -> None:
        """Re-seeding does not duplicate rows."""
        conn = _make_archive(tmp_path)
        from polylogue.storage.sqlite.archive_tiers.pricing_seed import seed_price_catalog

        with conn:
            seed_price_catalog(conn)
            seed_price_catalog(conn)

        catalog_count = conn.execute("SELECT COUNT(*) FROM price_catalogs").fetchone()[0]
        price_count = conn.execute("SELECT COUNT(*) FROM model_prices").fetchone()[0]
        assert catalog_count == 1
        assert price_count == len(PRICING)
        conn.close()

    def test_seed_versions_changed_pricing_and_writer_uses_new_catalog(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A catalog correction preserves old evidence and prices new usage by hash."""
        conn = _make_archive(tmp_path)
        original_catalog_id = _catalog_id()
        model_name = "gpt-4o-mini"
        original_pricing = PRICING[model_name]
        monkeypatch.setitem(
            PRICING,
            model_name,
            replace(original_pricing, input_usd_per_1m=original_pricing.input_usd_per_1m + 1.0),
        )

        with conn:
            revised_catalog_id = seed_price_catalog(conn)
            write_parsed_session_to_archive(
                conn,
                _session(
                    provider_session_id="revised-pricing",
                    messages=[_msg(provider_message_id="m1", model_name=model_name, input_tokens=100)],
                ),
            )

        assert revised_catalog_id != original_catalog_id
        assert active_price_catalog_id(conn) == revised_catalog_id
        assert conn.execute("SELECT COUNT(*) FROM price_catalogs").fetchone()[0] == 2
        assert (
            conn.execute(
                "SELECT catalog_hash FROM price_catalogs WHERE catalog_id = ?", (revised_catalog_id,)
            ).fetchone()[0]
            == _catalog_hash()
        )
        assert (
            conn.execute(
                "SELECT input_cost_per_million FROM model_prices WHERE catalog_id = ? AND model_name = ?",
                (original_catalog_id, model_name),
            ).fetchone()[0]
            == original_pricing.input_usd_per_1m
        )
        assert (
            conn.execute(
                "SELECT input_cost_per_million FROM model_prices WHERE catalog_id = ? AND model_name = ?",
                (revised_catalog_id, model_name),
            ).fetchone()[0]
            == PRICING[model_name].input_usd_per_1m
        )
        assert (
            conn.execute(
                "SELECT priced_with FROM session_model_usage WHERE session_id = 'claude-code-session:revised-pricing'"
            ).fetchone()[0]
            == revised_catalog_id
        )
        conn.close()


# ---------------------------------------------------------------------------
# Part B: token aggregation
# ---------------------------------------------------------------------------


class TestTokenAggregation:
    def test_token_sums_written_for_known_model(self, tmp_path: Path) -> None:
        conn = _make_archive(tmp_path)
        messages = [
            _msg(
                provider_message_id="m1",
                role="assistant",
                model_name="claude-sonnet-4-5",
                input_tokens=1000,
                output_tokens=500,
                cache_read_tokens=200,
                cache_write_tokens=100,
            ),
            _msg(
                provider_message_id="m2",
                role="assistant",
                model_name="claude-sonnet-4-5",
                input_tokens=800,
                output_tokens=300,
            ),
        ]
        with conn:
            write_parsed_session_to_archive(conn, _session(messages=messages))

        row = conn.execute("SELECT * FROM session_model_usage WHERE model_name = 'claude-sonnet-4-5'").fetchone()
        assert row is not None
        assert row["input_tokens"] == 1800
        assert row["output_tokens"] == 800
        assert row["cache_read_tokens"] == 200
        assert row["cache_write_tokens"] == 100
        assert row["message_count"] == 2
        conn.close()

    def test_messages_with_null_model_excluded_from_aggregation(self, tmp_path: Path) -> None:
        conn = _make_archive(tmp_path)
        messages = [
            _msg(provider_message_id="m1", role="user", model_name=None, input_tokens=999),
            _msg(
                provider_message_id="m2",
                role="assistant",
                model_name="gpt-4o",
                input_tokens=100,
                output_tokens=50,
            ),
        ]
        with conn:
            write_parsed_session_to_archive(conn, _session(messages=messages))

        # Only gpt-4o row should be in session_model_usage
        all_rows = conn.execute("SELECT model_name FROM session_model_usage").fetchall()
        model_names = {row["model_name"] for row in all_rows}
        assert "gpt-4o" in model_names
        # A row for '' / NULL model would be wrong
        assert "" not in model_names
        assert None not in model_names
        conn.close()

    def test_multiple_models_each_have_separate_row(self, tmp_path: Path) -> None:
        conn = _make_archive(tmp_path)
        messages = [
            _msg(
                provider_message_id="m1",
                model_name="claude-sonnet-4-5",
                input_tokens=100,
                output_tokens=50,
            ),
            _msg(
                provider_message_id="m2",
                model_name="gpt-4o",
                input_tokens=200,
                output_tokens=80,
            ),
        ]
        with conn:
            write_parsed_session_to_archive(conn, _session(messages=messages))

        rows = conn.execute("SELECT model_name, input_tokens, output_tokens FROM session_model_usage").fetchall()
        by_model = {row["model_name"]: row for row in rows}

        assert by_model["claude-sonnet-4-5"]["input_tokens"] == 100
        assert by_model["claude-sonnet-4-5"]["output_tokens"] == 50
        assert by_model["gpt-4o"]["input_tokens"] == 200
        assert by_model["gpt-4o"]["output_tokens"] == 80
        conn.close()


# ---------------------------------------------------------------------------
# Part C: cost_usd computation
# ---------------------------------------------------------------------------


class TestCostUsdComputation:
    def test_cost_usd_matches_estimate_cost_for_known_model(self, tmp_path: Path) -> None:
        conn = _make_archive(tmp_path)
        messages = [
            _msg(
                provider_message_id="m1",
                model_name="claude-sonnet-4-5",
                input_tokens=1_000_000,
                output_tokens=1_000_000,
            ),
        ]
        with conn:
            write_parsed_session_to_archive(conn, _session(messages=messages))

        row = conn.execute(
            "SELECT cost_usd, priced_with FROM session_model_usage WHERE model_name = 'claude-sonnet-4-5'"
        ).fetchone()
        assert row is not None
        expected = estimate_cost(1_000_000, 1_000_000, "claude-sonnet-4-5")
        assert row["cost_usd"] == pytest.approx(expected, rel=1e-9)
        assert row["priced_with"] == _catalog_id()
        conn.close()

    def test_unknown_model_gets_null_cost(self, tmp_path: Path) -> None:
        conn = _make_archive(tmp_path)
        messages = [
            _msg(
                provider_message_id="m1",
                model_name="some-future-model-xyz-not-in-catalog",
                input_tokens=1000,
                output_tokens=500,
            ),
        ]
        with conn:
            write_parsed_session_to_archive(conn, _session(messages=messages))

        row = conn.execute(
            "SELECT cost_usd, priced_with FROM session_model_usage "
            "WHERE model_name = 'some-future-model-xyz-not-in-catalog'"
        ).fetchone()
        assert row is not None
        assert row["cost_usd"] is None
        assert row["priced_with"] is None
        conn.close()

    def test_priced_with_fk_resolves_to_catalog_row(self, tmp_path: Path) -> None:
        """FK integrity: priced_with must reference an actual price_catalogs row."""
        conn = _make_archive(tmp_path)
        messages = [
            _msg(
                provider_message_id="m1",
                model_name="gpt-4o-mini",
                input_tokens=500,
                output_tokens=200,
            ),
        ]
        with conn:
            write_parsed_session_to_archive(conn, _session(messages=messages))

        violations = conn.execute("PRAGMA foreign_key_check(session_model_usage)").fetchall()
        assert violations == [], f"FK violations: {violations}"
        conn.close()

    def test_cost_usd_uses_cache_tokens_when_present(self, tmp_path: Path) -> None:
        conn = _make_archive(tmp_path)
        messages = [
            _msg(
                provider_message_id="m1",
                model_name="claude-sonnet-4-5",
                input_tokens=0,
                output_tokens=0,
                cache_read_tokens=1_000_000,
                cache_write_tokens=1_000_000,
            ),
        ]
        with conn:
            write_parsed_session_to_archive(conn, _session(messages=messages))

        row = conn.execute("SELECT cost_usd FROM session_model_usage WHERE model_name = 'claude-sonnet-4-5'").fetchone()
        expected = estimate_cost(0, 0, "claude-sonnet-4-5", 1_000_000, 1_000_000)
        assert row is not None
        assert row["cost_usd"] == pytest.approx(expected, rel=1e-9)
        conn.close()

    def test_zero_token_model_gets_null_cost(self, tmp_path: Path) -> None:
        """A model that appears in models_used but has no token-bearing messages stays NULL."""
        conn = _make_archive(tmp_path)
        # Session has no token-bearing messages but declares a model in models_used
        session = ParsedSession(
            source_name=Provider.CLAUDE_CODE,
            provider_session_id="no-tokens-session",
            title="no tokens",
            messages=[],
            models_used=["claude-sonnet-4-5"],
        )
        with conn:
            write_parsed_session_to_archive(conn, session)

        row = conn.execute("SELECT cost_usd FROM session_model_usage WHERE model_name = 'claude-sonnet-4-5'").fetchone()
        assert row is not None
        # No messages → no token aggregation → cost_usd stays NULL
        assert row["cost_usd"] is None
        conn.close()


# ---------------------------------------------------------------------------
# Read surface: dominant model ORDER BY uses populated token sums
# ---------------------------------------------------------------------------


class TestDominantModelReadSurface:
    def test_dominant_model_selected_by_token_volume(self, tmp_path: Path) -> None:
        """The archive cost-insight subquery picks the model with the most tokens.

        ORDER BY input_tokens + output_tokens DESC only works when the columns
        are populated.  This test verifies the subquery returns the correct
        dominant model after the token aggregation.
        """
        conn = _make_archive(tmp_path)
        messages = [
            _msg(
                provider_message_id="m1",
                model_name="gpt-4o-mini",
                input_tokens=100,
                output_tokens=50,
            ),
            _msg(
                provider_message_id="m2",
                model_name="claude-sonnet-4-5",
                input_tokens=5000,
                output_tokens=2000,
            ),
        ]
        session = _session(messages=messages)
        with conn:
            session_id = write_parsed_session_to_archive(conn, session)

        dominant = conn.execute(
            """
            SELECT smu.model_name
            FROM session_model_usage smu
            WHERE smu.session_id = ?
            ORDER BY smu.input_tokens + smu.output_tokens DESC, smu.model_name
            LIMIT 1
            """,
            (session_id,),
        ).fetchone()
        assert dominant is not None
        assert dominant[0] == "claude-sonnet-4-5"
        conn.close()
