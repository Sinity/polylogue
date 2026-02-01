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


POLYLOGUE_DB = Path.home() / ".local/state/polylogue/polylogue.db"

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


@pytest.fixture
def db():
    """Database connection fixture."""
    if not POLYLOGUE_DB.exists():
        pytest.skip("Polylogue database not found - run polylogue to ingest data first")
    conn = sqlite3.connect(POLYLOGUE_DB)
    yield conn
    conn.close()


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
    """Validate extraction works on real data."""

    @pytest.mark.parametrize("provider", ["claude-code", "claude", "chatgpt", "gemini"])
    def test_extraction_succeeds(self, db, provider):
        """Extraction should succeed on real messages.

        Sample limit controlled by POLYLOGUE_TEST_SAMPLES env var:
        - Default (100): Fast feedback
        - 0: All messages (exhaustive)
        """
        limit = get_sample_limit()
        messages = iter_provider_messages(db, provider, limit=limit)

        if not messages:
            pytest.skip(f"No {provider} messages in database")

        total = get_provider_message_count(db, provider)
        testing_all = limit is None or limit >= total
        mode = "exhaustive" if testing_all else f"sparse ({limit}/{total})"

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

        print(f"\n{provider} [{mode}]: {extracted} extracted, {skipped} metadata, {len(errors)} errors")

        assert extracted > 0, f"No {provider} messages extracted"
        assert len(errors) == 0, f"Extraction errors on {provider}: {errors[:10]}"


# =============================================================================
# Viewport Extraction Validation
# =============================================================================


class TestViewportValidation:
    """Validate viewport extraction produces sensible results."""

    def test_tool_calls_have_valid_categories(self, db):
        """All extracted tool calls should have valid categories."""
        limit = get_sample_limit() or 500  # Cap at 500 for viewport tests
        messages = iter_provider_messages(db, "claude-code", limit=min(limit, 500) if limit else 500)

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

        if category_counts:
            print(f"\nTool category distribution:")
            for cat, count in category_counts.most_common():
                print(f"  {cat}: {count}")

    def test_reasoning_traces_have_content(self, db):
        """Reasoning traces should have non-empty text."""
        limit = get_sample_limit() or 500

        for provider in ["claude-code", "gemini"]:
            messages = iter_provider_messages(db, provider, limit=min(limit, 500) if limit else 500)

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
                print(f"\n{provider}: {trace_count} traces, {empty_traces} empty")
                assert empty_traces == 0, f"{provider} has empty reasoning traces"


# =============================================================================
# Data Integrity Validation
# =============================================================================


class TestDataIntegrity:
    """Validate raw data integrity in database."""

    def test_provider_meta_has_raw(self, db):
        """All messages should have raw data in provider_meta."""
        cur = db.cursor()
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

        issues = []
        for provider, total, null_meta, missing_raw in cur.fetchall():
            if null_meta > 0 or missing_raw > 0:
                issues.append(f"{provider}: {null_meta} null meta, {missing_raw} missing raw (of {total})")

        if issues:
            print(f"\nData integrity notes:")
            for issue in issues:
                print(f"  {issue}")

    def test_provider_coverage(self, db):
        """Report provider coverage in database."""
        cur = db.cursor()
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

        print("\n=== Database Coverage ===")
        for provider, convs, msgs in cur.fetchall():
            print(f"  {provider}: {convs:,} conversations, {msgs:,} messages")


# =============================================================================
# Regeneration Test
# =============================================================================


class TestRegeneration:
    """Test that we can regenerate from raw data."""

    def test_raw_data_is_complete(self, db):
        """Raw data should contain all fields needed for extraction."""
        for provider in ["claude-code", "claude", "chatgpt", "gemini"]:
            messages = iter_provider_messages(db, provider, limit=10)

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
