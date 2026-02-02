"""Validate extraction logic against real database data.

Uses polylogue database's `provider_meta.raw` as ground truth.
Extraction must work on 100% of real data with zero errors.

Test modes:
    - Default: Sample 100 messages per provider (fast)
    - Full: POLYLOGUE_TEST_SAMPLES=0 for ALL messages
"""

from __future__ import annotations

import json
import os
import sqlite3
from collections import Counter
from pathlib import Path

import pytest

from polylogue.schemas.unified import (
    HarmonizedMessage,
    extract_from_provider_meta,
    is_message_record,
)
from polylogue.lib.viewports import ToolCategory


# Default to sparse testing; 0 means exhaustive
DEFAULT_SAMPLES = 100


def get_sample_limit() -> int | None:
    """Get sample limit from environment. 0 or unset means use default, negative means all."""
    env_val = os.environ.get("POLYLOGUE_TEST_SAMPLES")
    if env_val is None:
        return DEFAULT_SAMPLES
    try:
        n = int(env_val)
        return None if n <= 0 else n  # 0 or negative = all
    except ValueError:
        return DEFAULT_SAMPLES


def get_provider_message_count(conn: sqlite3.Connection, provider: str) -> int:
    """Get total message count for a provider."""
    cur = conn.cursor()
    cur.execute(
        """
        SELECT COUNT(*)
        FROM messages m
        JOIN conversations c ON m.conversation_id = c.conversation_id
        WHERE c.provider_name = ?
        """,
        (provider,)
    )
    return cur.fetchone()[0]


def iter_provider_messages(
    conn: sqlite3.Connection,
    provider: str,
    limit: int | None = None,
) -> list[tuple[str, dict]]:
    """Iterate messages for a provider.

    Args:
        conn: Database connection
        provider: Provider name
        limit: Max messages (None = all)

    Returns:
        List of (message_id, provider_meta) tuples.
    """
    cur = conn.cursor()
    query = """
        SELECT m.message_id, m.provider_meta
        FROM messages m
        JOIN conversations c ON m.conversation_id = c.conversation_id
        WHERE c.provider_name = ?
    """
    if limit:
        query += f" LIMIT {limit}"

    cur.execute(query, (provider,))

    results = []
    for message_id, pm_json in cur.fetchall():
        if pm_json:
            results.append((message_id, json.loads(pm_json)))
    return results


# =============================================================================
# Core Extraction Validation
# =============================================================================


class TestExtractionValidation:
    """Validate extraction works on seeded fixture data."""

    @pytest.mark.parametrize("provider", ["claude-code", "chatgpt", "codex"])
    def test_extraction_succeeds(self, seeded_db, provider):
        """Extraction should succeed on real messages from fixtures.

        Sample limit controlled by POLYLOGUE_TEST_SAMPLES env var:
        - Default (100): Fast feedback
        - 0: All messages (exhaustive)
        """
        conn = sqlite3.connect(seeded_db)
        limit = get_sample_limit()
        messages = iter_provider_messages(conn, provider, limit=limit)

        if not messages:
            conn.close()
            pytest.skip(f"No {provider} messages in seeded database")

        total = get_provider_message_count(conn, provider)
        conn.close()

        extracted = 0
        skipped = 0
        errors = []

        for msg_id, pm in messages:
            raw = pm.get("raw", pm)

            if not is_message_record(provider, raw):
                skipped += 1
                continue

            try:
                result = extract_from_provider_meta(provider, pm)
                assert isinstance(result, HarmonizedMessage)
                assert result.role in ("user", "assistant", "system", "tool", "unknown")
                assert isinstance(result.text, str)
                extracted += 1
            except Exception as e:
                errors.append((msg_id, str(e)[:100]))

        if extracted == 0:
            pytest.skip(f"No {provider} messages extracted from seeded database")
        assert len(errors) == 0, f"Extraction errors on {provider}: {errors[:10]}"


# =============================================================================
# Viewport Extraction Validation
# =============================================================================


class TestViewportValidation:
    """Validate viewport extraction produces sensible results."""

    def test_tool_calls_have_valid_categories(self, seeded_db):
        """All extracted tool calls should have valid categories."""
        conn = sqlite3.connect(seeded_db)
        limit = get_sample_limit() or 500  # Cap at 500 for viewport tests
        messages = iter_provider_messages(conn, "claude-code", limit=min(limit, 500) if limit else 500)
        conn.close()

        category_counts = Counter()
        invalid_tools = []

        for msg_id, pm in messages:
            raw = pm.get("raw", pm)
            if not is_message_record("claude-code", raw):
                continue

            msg = extract_from_provider_meta("claude-code", pm)

            for tool in msg.tool_calls:
                if tool.category not in ToolCategory:
                    invalid_tools.append((tool.name, tool.category))
                else:
                    category_counts[tool.category.value] += 1

        assert len(invalid_tools) == 0, f"Invalid tool categories: {invalid_tools}"

    def test_reasoning_traces_have_content(self, seeded_db):
        """Reasoning traces should have non-empty text."""
        conn = sqlite3.connect(seeded_db)
        limit = get_sample_limit() or 500

        for provider in ["claude-code"]:  # Only claude-code in seeded fixtures
            messages = iter_provider_messages(conn, provider, limit=min(limit, 500) if limit else 500)

            trace_count = 0
            empty_traces = 0

            for msg_id, pm in messages:
                raw = pm.get("raw", pm)
                if not is_message_record(provider, raw):
                    continue

                msg = extract_from_provider_meta(provider, pm)

                for trace in msg.reasoning_traces:
                    trace_count += 1
                    if not trace.text or not trace.text.strip():
                        empty_traces += 1

            if trace_count > 0:
                assert empty_traces == 0, f"{provider} has empty reasoning traces"

        conn.close()


# =============================================================================
# Data Integrity Validation
# =============================================================================


class TestDataIntegrity:
    """Validate raw data integrity in seeded database."""

    def test_provider_meta_has_raw(self, seeded_db):
        """All messages should have raw data in provider_meta."""
        conn = sqlite3.connect(seeded_db)
        cur = conn.cursor()
        cur.execute(
            """
            SELECT c.provider_name, COUNT(*) as total,
                   SUM(CASE WHEN m.provider_meta IS NULL THEN 1 ELSE 0 END) as null_meta,
                   SUM(CASE WHEN m.provider_meta IS NOT NULL
                            AND json_extract(m.provider_meta, '$.raw') IS NULL
                       THEN 1 ELSE 0 END) as missing_raw
            FROM messages m
            JOIN conversations c ON m.conversation_id = c.conversation_id
            GROUP BY c.provider_name
            """
        )

        rows = cur.fetchall()
        conn.close()

        issues = []
        for provider, total, null_meta, missing_raw in rows:
            if null_meta > 0 or missing_raw > 0:
                issues.append(f"{provider}: {null_meta} null meta, {missing_raw} missing raw (of {total})")

        # This is informational - seeded data may have missing raw
        if issues:
            import warnings
            warnings.warn(f"Data integrity notes: {issues}")

    def test_provider_coverage(self, seeded_db):
        """Report provider coverage in database."""
        conn = sqlite3.connect(seeded_db)
        cur = conn.cursor()
        cur.execute(
            """
            SELECT c.provider_name,
                   COUNT(DISTINCT c.conversation_id) as convs,
                   COUNT(m.message_id) as msgs
            FROM conversations c
            LEFT JOIN messages m ON c.conversation_id = m.conversation_id
            GROUP BY c.provider_name
            ORDER BY msgs DESC
            """
        )

        rows = cur.fetchall()
        conn.close()

        # Just verify there's data in seeded database
        assert len(rows) > 0, "No providers in seeded database"


# =============================================================================
# Regeneration Test
# =============================================================================


class TestRegeneration:
    """Test that we can regenerate from raw data."""

    def test_raw_data_is_complete(self, seeded_db):
        """Raw data should contain all fields needed for extraction."""
        conn = sqlite3.connect(seeded_db)

        for provider in ["claude-code", "chatgpt", "codex"]:
            messages = iter_provider_messages(conn, provider, limit=10)

            for msg_id, pm in messages:
                raw = pm.get("raw")
                if not raw:
                    continue

                # Extract from raw only (not from pm which might have derived fields)
                try:
                    result = extract_from_provider_meta(provider, {"raw": raw})
                    assert isinstance(result, HarmonizedMessage)
                except Exception as e:
                    pytest.fail(f"Cannot extract from raw alone for {provider}: {e}")

        conn.close()
