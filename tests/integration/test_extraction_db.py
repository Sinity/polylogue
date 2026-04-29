"""Validate extraction logic against seeded synthetic database data.

Loads content from blob store (keyed by raw_id in raw_conversations).
Extraction should work on 100% of seeded synthetic data with zero errors.

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

from polylogue.lib.viewport.viewports import ToolCategory
from polylogue.schemas.unified.unified import (
    HarmonizedMessage,
    extract_from_provider_meta,
    is_message_record,
)
from polylogue.sources.providers.codex import CodexRecord

# Default to sparse testing; 0 means exhaustive
DEFAULT_SAMPLES = 100
JsonObject = dict[str, object]
JsonArray = list[object]
ParsedPayload = JsonObject | JsonArray
ProviderMessageEnvelope = dict[str, JsonObject]


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
        (provider,),
    )
    row = cur.fetchone()
    assert row is not None
    return int(row[0])


def iter_provider_messages(
    conn: sqlite3.Connection,
    provider: str,
    limit: int | None = None,
) -> list[tuple[str, ProviderMessageEnvelope]]:
    """Iterate raw message records for a provider.

    Args:
        conn: Database connection
        provider: Provider name
        limit: Max messages (None = all)

    Returns:
        List of (message_id, {"raw": provider_raw_message}) tuples.
    """
    results: list[tuple[str, ProviderMessageEnvelope]] = []
    raw_conversations = iter_raw_conversations(conn, provider)

    for raw_id, raw_content, _source_path in raw_conversations:
        payload = parse_raw_content(raw_content, provider)
        raw_messages = _extract_provider_raw_messages(provider, payload)
        for index, raw_message in enumerate(raw_messages):
            results.append((f"{raw_id}:{index}", {"raw": raw_message}))
            if limit is not None and len(results) >= limit:
                return results
    return results


def _extract_provider_raw_messages(provider: str, payload: ParsedPayload) -> list[JsonObject]:
    """Extract provider-native raw message records from a raw payload."""
    if provider == "chatgpt":
        conversations = payload if isinstance(payload, list) else [payload]
        messages: list[JsonObject] = []
        for conversation in conversations:
            if not isinstance(conversation, dict):
                continue
            mapping = conversation.get("mapping")
            if not isinstance(mapping, dict):
                continue
            for node in mapping.values():
                if not isinstance(node, dict):
                    continue
                message = node.get("message")
                if isinstance(message, dict):
                    messages.append(message)
        return messages

    records = payload if isinstance(payload, list) else [payload]
    provider_messages: list[JsonObject] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        if provider == "codex":
            try:
                if not CodexRecord.model_validate(record).is_message:
                    continue
            except Exception:
                continue
        elif not is_message_record(provider, record):
            continue
        provider_messages.append(record)
    return provider_messages


# =============================================================================
# Core Extraction Validation
# =============================================================================


@pytest.mark.slow
class TestExtractionValidation:
    """Validate extraction works on seeded fixture data."""

    @pytest.mark.parametrize("provider", ["claude-code", "chatgpt", "codex"])
    def test_extraction_succeeds(self, seeded_db: Path, provider: str) -> None:
        """Extraction should succeed on seeded synthetic messages.

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

        get_provider_message_count(conn, provider)
        conn.close()

        extracted = 0
        skipped = 0
        errors: list[tuple[str, str]] = []

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


@pytest.mark.slow
class TestViewportValidation:
    """Validate viewport extraction produces sensible results."""

    def test_tool_calls_have_valid_categories(self, seeded_db: Path) -> None:
        """All extracted tool calls should have valid categories."""
        conn = sqlite3.connect(seeded_db)
        limit = get_sample_limit() or 500  # Cap at 500 for viewport tests
        messages = iter_provider_messages(conn, "claude-code", limit=min(limit, 500) if limit else 500)
        conn.close()

        category_counts: Counter[str] = Counter()
        invalid_tools: list[tuple[str, ToolCategory]] = []

        for _msg_id, pm in messages:
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

    def test_reasoning_traces_have_content(self, seeded_db: Path) -> None:
        """Reasoning traces should have non-empty text."""
        conn = sqlite3.connect(seeded_db)
        limit = get_sample_limit() or 500

        for provider in ["claude-code"]:  # Only claude-code in seeded fixtures
            messages = iter_provider_messages(conn, provider, limit=min(limit, 500) if limit else 500)

            trace_count = 0
            empty_traces = 0

            for _msg_id, pm in messages:
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


@pytest.mark.slow
class TestDataIntegrity:
    """Validate raw conversation integrity in seeded database."""

    def test_raw_conversations_have_raw_content(self, seeded_db: Path) -> None:
        """Seeded raw conversations should have blob_size > 0."""
        conn = sqlite3.connect(seeded_db)
        cur = conn.cursor()
        cur.execute(
            """
            SELECT provider_name,
                   COUNT(*) as total,
                   SUM(CASE WHEN blob_size IS NULL OR blob_size = 0 THEN 1 ELSE 0 END) as missing_size
            FROM raw_conversations
            GROUP BY provider_name
            """
        )

        rows = cur.fetchall()
        conn.close()

        issues: list[str] = []
        for provider, total, missing_size in rows:
            if missing_size > 0:
                issues.append(f"{provider}: {missing_size} missing blob sizes (of {total})")

        assert not issues, f"Missing blob sizes in seeded database: {issues}"

    def test_provider_coverage(self, seeded_db: Path) -> None:
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


@pytest.mark.slow
class TestRegeneration:
    """Test that we can regenerate from raw data."""

    def test_raw_data_is_complete(self, seeded_db: Path) -> None:
        """Raw data should contain all fields needed for extraction."""
        conn = sqlite3.connect(seeded_db)

        for provider in ["claude-code", "chatgpt", "codex"]:
            messages = iter_provider_messages(conn, provider, limit=10)

            for _msg_id, pm in messages:
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


# =============================================================================
# Provider Parsing Validation (from raw_conversations table)
# =============================================================================


def iter_raw_conversations(
    conn: sqlite3.Connection,
    provider: str,
    limit: int | None = None,
) -> list[tuple[str, bytes, str | None]]:
    """Iterate raw conversations for a provider.

    Args:
        conn: Database connection
        provider: Provider name
        limit: Max conversations (None = all)

    Returns:
        List of (raw_id, raw_content, source_path) tuples.
    """
    from pathlib import Path

    from polylogue.storage.blob_store import BlobStore

    row = conn.execute("PRAGMA database_list").fetchone()
    assert row is not None
    db_path = str(row[2])
    blob_store = BlobStore(Path(db_path).parent / "blob")
    cur = conn.cursor()
    query = """
        SELECT raw_id, source_path
        FROM raw_conversations
        WHERE provider_name = ?
    """
    if limit:
        query += f" LIMIT {limit}"

    cur.execute(query, (provider,))
    results = []
    for raw_id, source_path in cur.fetchall():
        raw_id_str = str(raw_id)
        raw_content = blob_store.read_all(raw_id_str)
        results.append((raw_id_str, raw_content, source_path if isinstance(source_path, str) else None))
    return results


def get_raw_conversation_count(conn: sqlite3.Connection, provider: str) -> int:
    """Get total raw conversation count for a provider."""
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM raw_conversations WHERE provider_name = ?", (provider,))
    row = cur.fetchone()
    assert row is not None
    return int(row[0])


def parse_raw_content(raw_content: bytes, provider: str) -> ParsedPayload:
    """Parse raw_content bytes into JSON data, handling both JSON and JSONL.

    JSONL providers (claude-code, codex, gemini) store entire files as raw_content,
    which may contain multiple JSON objects on separate lines. JSON providers
    (chatgpt) store a single JSON object.
    """
    content = raw_content.decode("utf-8") if isinstance(raw_content, bytes) else raw_content

    if provider in ("claude-code", "codex", "gemini"):
        # Try single JSON first (e.g. one-line JSONL files)
        try:
            data = json.loads(content)
            return [data] if isinstance(data, dict) else data
        except json.JSONDecodeError:
            pass
        # Fall back to line-by-line JSONL parsing
        items = []
        for line in content.strip().split("\n"):
            if line.strip():
                items.append(json.loads(line))
        return items

    parsed = json.loads(content)
    assert isinstance(parsed, (dict, list))
    return parsed


@pytest.mark.slow
class TestRawConversationParsing:
    """Validate provider parsers against raw_conversations table.

    This is the SYSTEMATIC test for parsers - instead of crafting test cases,
    we run parsers against seeded stored data to ensure 100% success rate.
    """

    @pytest.mark.parametrize("provider", ["chatgpt", "claude-code", "codex"])
    def test_provider_parses_all_raw_conversations(self, seeded_db: Path, provider: str) -> None:
        """Every raw_conversation for this provider parses without error.

        This test replaces many spot checks in test_parsers_unit.py by:
        1. Testing against seeded provider-format data through full storage paths
        2. Testing ALL stored conversations (not cherry-picked examples)
        3. Ensuring parsers work on naturally varied records from generated corpus
        """
        from polylogue.sources.parsers.chatgpt import parse as chatgpt_parse
        from polylogue.sources.parsers.claude import parse_code as claude_code_parse
        from polylogue.sources.parsers.codex import parse as codex_parse

        conn = sqlite3.connect(seeded_db)
        limit = get_sample_limit()
        raw_convos = iter_raw_conversations(conn, provider, limit=limit)

        if not raw_convos:
            conn.close()
            pytest.skip(f"No {provider} raw conversations in seeded database")

        total = get_raw_conversation_count(conn, provider)
        conn.close()

        parsed_count = 0
        errors: list[tuple[str, str]] = []

        for raw_id, raw_content, _source_path in raw_convos:
            try:
                data = parse_raw_content(raw_content, provider)

                # Parse based on provider
                if provider == "chatgpt":
                    assert isinstance(data, dict)
                    result = chatgpt_parse(data, raw_id)
                elif provider == "claude-code":
                    result = claude_code_parse(data if isinstance(data, list) else [data], raw_id)
                elif provider == "codex":
                    result = codex_parse(data if isinstance(data, list) else [data], raw_id)
                else:
                    continue

                # Validate basic structure
                assert result.provider_name in [provider, "chatgpt", "claude-ai", "codex", "claude-code"]
                assert result.provider_conversation_id is not None
                assert isinstance(result.messages, list)

                parsed_count += 1

            except json.JSONDecodeError as e:
                errors.append((raw_id, f"JSON error: {e}"))
            except Exception as e:
                errors.append((raw_id, f"Parse error: {str(e)[:80]}"))

        if parsed_count == 0:
            pytest.skip(f"No {provider} conversations could be parsed from seeded database")

        # Report coverage
        coverage_pct = (parsed_count / total * 100) if total > 0 else 0

        assert len(errors) == 0, f"""
            {provider} parse errors ({len(errors)} of {parsed_count + len(errors)}):
            {errors[:5]}
            Coverage: {coverage_pct:.1f}%
        """

    @pytest.mark.parametrize("provider", ["chatgpt", "claude-code", "codex"])
    def test_provider_messages_have_valid_structure(self, seeded_db: Path, provider: str) -> None:
        """All parsed messages have valid role and text structure.

        This validates that parsers produce well-formed ParsedMessage objects,
        not just that they don't crash.
        """
        from polylogue.sources.parsers.chatgpt import parse as chatgpt_parse
        from polylogue.sources.parsers.claude import parse_code as claude_code_parse
        from polylogue.sources.parsers.codex import parse as codex_parse

        conn = sqlite3.connect(seeded_db)
        limit = get_sample_limit()
        raw_convos = iter_raw_conversations(conn, provider, limit=limit)

        if not raw_convos:
            conn.close()
            pytest.skip(f"No {provider} raw conversations in seeded database")

        conn.close()

        valid_roles = {"user", "assistant", "system", "tool", "human", "model", "unknown"}
        issues: list[tuple[str, str]] = []

        for raw_id, raw_content, _ in raw_convos[:20]:  # Check first 20
            try:
                data = parse_raw_content(raw_content, provider)

                if provider == "chatgpt":
                    assert isinstance(data, dict)
                    result = chatgpt_parse(data, raw_id)
                elif provider == "claude-code":
                    result = claude_code_parse(data if isinstance(data, list) else [data], raw_id)
                elif provider == "codex":
                    result = codex_parse(data if isinstance(data, list) else [data], raw_id)
                else:
                    continue

                for msg in result.messages:
                    # Check role is valid
                    if msg.role.lower() not in valid_roles:
                        issues.append((raw_id, f"Invalid role: {msg.role}"))

                    # Check text is string (can be empty, but must be string or None)
                    if msg.text is not None and not isinstance(msg.text, str):
                        issues.append((raw_id, f"Text is not string: {type(msg.text)}"))

                    # Check provider_message_id exists
                    if not msg.provider_message_id:
                        issues.append((raw_id, "Missing provider_message_id"))

            except Exception:
                continue  # Skip parse errors (covered by other test)

        assert len(issues) == 0, f"Message structure issues: {issues[:10]}"


@pytest.mark.slow
class TestRawConversationCoverage:
    """Coverage and statistics for raw conversation parsing."""

    def test_all_providers_have_raw_conversations(self, seeded_db: Path) -> None:
        """Every known provider has at least some raw conversations."""
        conn = sqlite3.connect(seeded_db)
        cur = conn.cursor()
        cur.execute("""
            SELECT provider_name, COUNT(*) as count
            FROM raw_conversations
            GROUP BY provider_name
            ORDER BY count DESC
        """)

        rows = cur.fetchall()
        conn.close()

        if not rows:
            pytest.skip("No raw conversations in seeded database")

        # Report what's available
        providers = {row[0]: row[1] for row in rows}

        # At least one provider should have data
        assert sum(providers.values()) > 0, "No raw conversations available for any provider"

    def test_raw_to_parsed_link_integrity(self, seeded_db: Path) -> None:
        """Conversations parsed from raw should link back to raw_id."""
        conn = sqlite3.connect(seeded_db)
        cur = conn.cursor()

        # Count conversations with raw_id link
        cur.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN raw_id IS NOT NULL THEN 1 ELSE 0 END) as linked
            FROM conversations
        """)

        total, linked = cur.fetchone()
        conn.close()

        if total == 0:
            pytest.skip("No conversations in seeded database")

        # This is informational - not all conversations may have raw_id
        # (e.g., legacy imports before raw storage was added)
        link_pct = (linked / total * 100) if total > 0 else 0

        if link_pct < 50:
            import warnings

            warnings.warn(f"Only {link_pct:.1f}% of conversations have raw_id links", stacklevel=2)
