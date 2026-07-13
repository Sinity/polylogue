"""Tests for origin analytics product computation."""

from __future__ import annotations

from pathlib import Path

import pytest

from polylogue import Polylogue
from polylogue.core.enums import MaterialOrigin
from polylogue.insights.archive import ArchiveCoverageInsight, ArchiveCoverageInsightQuery
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from tests.infra.storage_records import SessionBuilder


def _archive(tmp_path: Path) -> Polylogue:
    initialize_active_archive_root(tmp_path)
    return Polylogue(archive_root=tmp_path, db_path=tmp_path / "index.db")


async def _coverage(archive: Polylogue) -> list[ArchiveCoverageInsight]:
    return await archive.list_archive_coverage_insights(ArchiveCoverageInsightQuery(group_by="origin"))


class TestArchiveCoverageInsight:
    """Test ArchiveCoverageInsight contract."""

    def test_tool_use_percentage_with_data(self: object) -> None:
        """Tool use percentage is calculated correctly."""
        metrics = ArchiveCoverageInsight(
            source_name="test",
            session_count=100,
            message_count=500,
            user_message_count=200,
            assistant_message_count=300,
            avg_messages_per_session=5.0,
            avg_user_words=50.0,
            avg_assistant_words=150.0,
            tool_use_count=25,
            thinking_count=10,
            total_sessions_with_tools=20,
            total_sessions_with_thinking=8,
            tool_use_percentage=20.0,
            thinking_percentage=8.0,
        )
        assert metrics.tool_use_percentage == 20.0

    def test_tool_use_percentage_zero_sessions(self: object) -> None:
        """Tool use percentage returns 0 when no sessions."""
        metrics = ArchiveCoverageInsight(
            source_name="empty",
            session_count=0,
            message_count=0,
            user_message_count=0,
            assistant_message_count=0,
            avg_messages_per_session=0.0,
            avg_user_words=0.0,
            avg_assistant_words=0.0,
            tool_use_count=0,
            thinking_count=0,
            total_sessions_with_tools=0,
            total_sessions_with_thinking=0,
            tool_use_percentage=0.0,
            thinking_percentage=0.0,
        )
        assert metrics.tool_use_percentage == 0.0

    def test_thinking_percentage_with_data(self: object) -> None:
        """Thinking percentage is calculated correctly."""
        metrics = ArchiveCoverageInsight(
            source_name="test",
            session_count=50,
            message_count=200,
            user_message_count=100,
            assistant_message_count=100,
            avg_messages_per_session=4.0,
            avg_user_words=40.0,
            avg_assistant_words=120.0,
            tool_use_count=5,
            thinking_count=15,
            total_sessions_with_tools=3,
            total_sessions_with_thinking=10,
            tool_use_percentage=6.0,
            thinking_percentage=20.0,
        )
        assert metrics.thinking_percentage == 20.0

    def test_thinking_percentage_zero_sessions(self: object) -> None:
        """Thinking percentage returns 0 when no sessions."""
        metrics = ArchiveCoverageInsight(
            source_name="empty",
            session_count=0,
            message_count=0,
            user_message_count=0,
            assistant_message_count=0,
            avg_messages_per_session=0.0,
            avg_user_words=0.0,
            avg_assistant_words=0.0,
            tool_use_count=0,
            thinking_count=0,
            total_sessions_with_tools=0,
            total_sessions_with_thinking=0,
            tool_use_percentage=0.0,
            thinking_percentage=0.0,
        )
        assert metrics.thinking_percentage == 0.0


class TestListArchiveCoverageInsights:
    """Test native ``list_archive_coverage_insights`` aggregation."""

    @pytest.mark.asyncio
    async def test_empty_database(self: object, tmp_path: Path) -> None:
        """Empty database returns empty list."""
        result = await _coverage(_archive(tmp_path))
        assert result == []

    @pytest.mark.asyncio
    async def test_single_provider(self: object, tmp_path: Path) -> None:
        """Single provider aggregation."""
        archive = _archive(tmp_path)
        db_path = archive.archive_root / "index.db"

        (
            SessionBuilder(db_path, "conv-1")
            .provider("claude-ai")
            .title("Test Session")
            .add_message("msg-1", role="user", text="Hello world test")
            .add_message(
                "msg-2",
                role="assistant",
                text="Response with more words for testing average calculation",
            )
            .save()
        )

        result = await _coverage(archive)

        assert len(result) == 1
        assert result[0].source_name == "claude-ai"
        assert result[0].session_count == 1
        assert result[0].message_count == 2
        assert result[0].user_message_count == 1
        assert result[0].assistant_message_count == 1

    @pytest.mark.asyncio
    async def test_multiple_providers_sorted(self: object, tmp_path: Path) -> None:
        """Multiple providers sorted by session count descending."""
        archive = _archive(tmp_path)
        db_path = archive.archive_root / "index.db"

        # Create 2 claude sessions
        for i in range(2):
            (
                SessionBuilder(db_path, f"claude-{i}")
                .provider("claude-ai")
                .title(f"Claude {i}")
                .add_message(f"cmsg-{i}", role="user", text="Hello")
                .save()
            )

        # Create 3 chatgpt sessions
        for i in range(3):
            (
                SessionBuilder(db_path, f"chatgpt-{i}")
                .provider("chatgpt")
                .title(f"ChatGPT {i}")
                .add_message(f"gmsg-{i}", role="user", text="Hi")
                .save()
            )

        result = await _coverage(archive)

        assert len(result) == 2
        # ChatGPT has more sessions, should be first
        assert result[0].source_name == "chatgpt"
        assert result[0].session_count == 3
        assert result[1].source_name == "claude-ai"
        assert result[1].session_count == 2

    @pytest.mark.asyncio
    async def test_user_assistant_segregation(self: object, tmp_path: Path) -> None:
        """User and assistant messages are counted separately."""
        archive = _archive(tmp_path)
        db_path = archive.archive_root / "index.db"

        (
            SessionBuilder(db_path, "conv-roles")
            .provider("claude-ai")
            .title("Roles Test")
            .add_message("rmsg-1", role="user", text="User one")
            .add_message("rmsg-2", role="assistant", text="Assistant one")
            .add_message("rmsg-3", role="user", text="User two")
            .add_message("rmsg-4", role="assistant", text="Assistant two three four")
            .save()
        )

        result = await _coverage(archive)

        assert len(result) == 1
        assert result[0].user_message_count == 2
        assert result[0].assistant_message_count == 2
        assert result[0].message_count == 4

    @pytest.mark.asyncio
    async def test_provider_user_and_authored_user_counts_can_diverge(self: object, tmp_path: Path) -> None:
        """Coverage keeps provider-role user counts distinct from authored prompts."""
        archive = _archive(tmp_path)
        db_path = archive.archive_root / "index.db"

        (
            SessionBuilder(db_path, "conv-authored-split")
            .provider("claude-code")
            .title("Authored Split")
            .add_message(
                "m-runtime",
                role="user",
                text="<local-command-stdout>generated output payload</local-command-stdout>",
                material_origin=MaterialOrigin.RUNTIME_PROTOCOL,
            )
            .add_message(
                "m-authored",
                role="user",
                text="Please inspect the archive state",
                material_origin=MaterialOrigin.HUMAN_AUTHORED,
            )
            .add_message(
                "m-assistant",
                role="assistant",
                text="I will inspect it",
                material_origin=MaterialOrigin.ASSISTANT_AUTHORED,
            )
            .save()
        )

        result = await _coverage(archive)

        assert len(result) == 1
        assert result[0].user_message_count == 2
        assert result[0].authored_user_message_count == 1
        assert result[0].avg_user_words == pytest.approx(4.0)
        assert result[0].avg_authored_user_words == 5.0

    @pytest.mark.asyncio
    async def test_avg_messages_per_session(self: object, tmp_path: Path) -> None:
        """Average messages per session is computed correctly."""
        archive = _archive(tmp_path)
        db_path = archive.archive_root / "index.db"

        # Conv 1: 2 messages
        (
            SessionBuilder(db_path, "avg-1")
            .provider("claude-ai")
            .title("Avg 1")
            .add_message("avg-msg-1a", role="user", text="Hi")
            .add_message("avg-msg-1b", role="assistant", text="Hello")
            .save()
        )

        # Conv 2: 4 messages
        (
            SessionBuilder(db_path, "avg-2")
            .provider("claude-ai")
            .title("Avg 2")
            .add_message("avg-msg-2a", role="user", text="Q1")
            .add_message("avg-msg-2b", role="assistant", text="A1")
            .add_message("avg-msg-2c", role="user", text="Q2")
            .add_message("avg-msg-2d", role="assistant", text="A2")
            .save()
        )

        result = await _coverage(archive)

        assert len(result) == 1
        # Total 6 messages across 2 sessions = 3.0 average
        assert result[0].avg_messages_per_session == 3.0

    @pytest.mark.asyncio
    async def test_tool_use_detection(self: object, tmp_path: Path) -> None:
        """Tool use is detected from content_blocks."""
        archive = _archive(tmp_path)
        db_path = archive.archive_root / "index.db"

        (
            SessionBuilder(db_path, "tool-conv")
            .provider("claude-ai")
            .title("Tool Use Test")
            .add_message(
                "tool-msg-1",
                role="assistant",
                text="Let me search for that",
                blocks=[{"type": "tool_use", "name": "search", "id": "toolu_123"}],
            )
            .save()
        )

        result = await _coverage(archive)

        assert len(result) == 1
        assert result[0].tool_use_count == 1
        assert result[0].total_sessions_with_tools == 1
        assert result[0].tool_use_percentage == 100.0

    @pytest.mark.asyncio
    async def test_thinking_detection(self: object, tmp_path: Path) -> None:
        """Thinking is detected from content_blocks."""
        archive = _archive(tmp_path)
        db_path = archive.archive_root / "index.db"

        (
            SessionBuilder(db_path, "think-conv")
            .provider("claude-ai")
            .title("Thinking Test")
            .add_message(
                "think-msg-1",
                role="assistant",
                text="Let me think about this",
                blocks=[{"type": "thinking", "thinking": "Reasoning..."}],
            )
            .save()
        )

        result = await _coverage(archive)

        assert len(result) == 1
        assert result[0].thinking_count == 1
        assert result[0].total_sessions_with_thinking == 1
        assert result[0].thinking_percentage == 100.0

    @pytest.mark.asyncio
    async def test_sessions_with_tools_set_dedup(self: object, tmp_path: Path) -> None:
        """Multiple tool uses in same session counted once."""
        archive = _archive(tmp_path)
        db_path = archive.archive_root / "index.db"

        (
            SessionBuilder(db_path, "multi-tool")
            .provider("claude-ai")
            .title("Multi Tool")
            .add_message(
                "mt-msg-1",
                role="assistant",
                text="Tool 1",
                blocks=[{"type": "tool_use", "name": "a"}],
            )
            .add_message(
                "mt-msg-2",
                role="assistant",
                text="Tool 2",
                blocks=[{"type": "tool_use", "name": "b"}],
            )
            .save()
        )

        result = await _coverage(archive)

        assert len(result) == 1
        assert result[0].tool_use_count == 2  # Two tool use messages
        assert result[0].total_sessions_with_tools == 1  # But one session
        assert result[0].tool_use_percentage == 100.0

    @pytest.mark.asyncio
    async def test_division_by_zero_protection(self: object, tmp_path: Path) -> None:
        """Zero-denominator averages render None (not-applicable), never a crash or a lying 0.0 (9e5.29)."""
        archive = _archive(tmp_path)
        db_path = archive.archive_root / "index.db"

        # Session with system-only messages (no user/assistant)
        (
            SessionBuilder(db_path, "zero-div")
            .provider("claude-ai")
            .title("Zero Division")
            .add_message("zero-msg-1", role="system", text="System message")
            .save()
        )

        result = await _coverage(archive)

        assert len(result) == 1
        # Should not raise division by zero, and must not fabricate a 0.0
        # over an empty (zero user/assistant message) denominator.
        assert result[0].avg_user_words is None
        assert result[0].avg_assistant_words is None
        # session_count == 1 is a nonzero denominator here, so these are
        # true measured zeros (no session used tools/thinking), not absent.
        assert result[0].tool_use_percentage == 0.0
        assert result[0].thinking_percentage == 0.0


# ============================================================================
# _seed_db helper: create a index.db with custom message rows
# ============================================================================


def _seed_db(
    tmp_path: Path,
    rows: list[tuple[str, str, str | None, dict[str, object] | None]],
) -> Polylogue:
    """Seed a index.db from raw rows and return the reading archive.

    Each row is ``(provider, role, text, meta_or_None)`` where ``meta`` may
    carry a ``content_blocks`` list. Rows are grouped by provider into one
    session per provider, then written through the ``SessionBuilder``. Content
    blocks are passed via the typed ``content_blocks`` builder kwarg (#1743).
    """
    archive = _archive(tmp_path)
    db_path = archive.archive_root / "index.db"

    convos_by_provider: dict[str, list[tuple[str, str | None, dict[str, object] | None]]] = {}
    for provider, role, text, meta in rows:
        convos_by_provider.setdefault(provider, []).append((role, text, meta))

    msg_counter = 0
    for provider, messages in convos_by_provider.items():
        builder = SessionBuilder(db_path, f"conv-{provider}").provider(provider).title(f"{provider} Test Session")
        for role, text, meta in messages:
            msg_counter += 1
            content_blocks = (meta or {}).get("content_blocks", [])
            builder.add_message(
                f"msg-{msg_counter}",
                role=role,
                text="" if text is None else text,
                timestamp=None,
                blocks=content_blocks,
            )
        builder.save()

    return archive


# ============================================================================
# Word count SQL edge cases
# ============================================================================


class TestWordCountEdgeCases:
    """Verify word count handling for whitespace and empty-text edge cases."""

    @pytest.mark.asyncio
    async def test_spaces_only_text_counts_zero_words(self: object, tmp_path: Path) -> None:
        """Space-only messages should count as 0 words."""
        archive = _seed_db(
            tmp_path,
            [
                ("claude-ai", "user", "   ", None),
                ("claude-ai", "user", "     ", None),
            ],
        )
        results = await _coverage(archive)
        assert len(results) == 1
        assert results[0].avg_user_words == 0.0

    @pytest.mark.asyncio
    async def test_tabs_newlines_are_stripped(self: object, tmp_path: Path) -> None:
        """Python split() strips all whitespace including tabs/newlines.

        word_count is precomputed at write time via len(text.split()).
        Python's split() treats tabs/newlines as whitespace, so whitespace-only
        text yields 0 words.
        """
        archive = _seed_db(
            tmp_path,
            [
                ("claude-ai", "user", "\t\t", None),
            ],
        )
        results = await _coverage(archive)
        assert results[0].avg_user_words == 0.0

    @pytest.mark.asyncio
    async def test_single_word_counts_one(self: object, tmp_path: Path) -> None:
        """A single word with no spaces counts as 1."""
        archive = _seed_db(
            tmp_path,
            [
                ("claude-ai", "user", "Hello", None),
            ],
        )
        results = await _coverage(archive)
        assert results[0].avg_user_words == 1.0

    @pytest.mark.asyncio
    async def test_multiple_spaces_between_words(self: object, tmp_path: Path) -> None:
        """Multiple spaces between words count as expected.

        word_count is precomputed via len(text.split()), which splits on any
        whitespace run, so 'hello  world' → 2 words (not 3).
        """
        archive = _seed_db(
            tmp_path,
            [
                ("claude-ai", "user", "hello  world", None),  # 2 spaces
            ],
        )
        results = await _coverage(archive)
        assert results[0].avg_user_words == 2.0

    @pytest.mark.asyncio
    async def test_empty_text_counts_zero(self: object, tmp_path: Path) -> None:
        """Empty string text counts as 0 words."""
        archive = _seed_db(
            tmp_path,
            [
                ("claude-ai", "user", "", None),
            ],
        )
        results = await _coverage(archive)
        assert results[0].avg_user_words == 0.0

    @pytest.mark.asyncio
    async def test_none_text_counts_zero(self: object, tmp_path: Path) -> None:
        """Missing text counts as 0 words."""
        archive = _seed_db(
            tmp_path,
            [
                ("claude-ai", "user", None, None),
            ],
        )
        results = await _coverage(archive)
        assert results[0].avg_user_words == 0.0


# ============================================================================
# Content-block detection resistance tests
# ============================================================================


class TestLikePatternResistance:
    """Verify tool_use/thinking detection keys on content blocks, not text."""

    @pytest.mark.asyncio
    async def test_tool_use_in_message_text_not_detected(self: object, tmp_path: Path) -> None:
        """Text containing 'tool_use' string should NOT count as tool use."""
        archive = _seed_db(
            tmp_path,
            [
                ("claude-ai", "assistant", "The tool_use feature is great", None),
                ("claude-ai", "assistant", 'I used "type":"tool_use" in my message', None),
            ],
        )
        results = await _coverage(archive)
        # tool_use_count should be 0 — detection is on content blocks, not text
        assert results[0].tool_use_count == 0

    @pytest.mark.asyncio
    async def test_thinking_in_message_text_not_detected(self: object, tmp_path: Path) -> None:
        """Text containing 'thinking' should NOT count as thinking."""
        archive = _seed_db(
            tmp_path,
            [
                ("claude-ai", "assistant", 'I was thinking about "type":"thinking" blocks', None),
            ],
        )
        results = await _coverage(archive)
        assert results[0].thinking_count == 0

    @pytest.mark.asyncio
    async def test_bare_tool_role_message_is_not_double_counted(self: object, tmp_path: Path) -> None:
        """A bare role='tool' message (no tool_use block) does not count as a tool use.

        Archive counts a tool use at its call site — the assistant message
        carrying the ``tool_use`` content block — via
        ``write.py:_has_block(message, BlockType.TOOL_USE)`` at both the
        per-message (``messages.has_tool_use``) and session
        (``sessions.tool_use_count``) levels. The tool *result*, delivered as a
        ``role='tool'`` message carrying a ``tool_result`` block, is therefore
        not re-counted. This intentionally diverges from the single-file archive
        precomputed ``messages.has_tool_use``, which also fired on ``role='tool'``
        and so double-counted each tool interaction (call + result). See
        ``.agent/scratch/storage-test-archive-gaps.md`` Gap 6.
        """
        archive = _seed_db(
            tmp_path,
            [
                ("claude-ai", "tool", "Tool result here", None),
            ],
        )
        results = await _coverage(archive)
        assert results[0].tool_use_count == 0

    @pytest.mark.asyncio
    async def test_tool_use_in_provider_meta_detected(self: object, tmp_path: Path) -> None:
        """Tool use in provider_meta content_blocks is detected."""
        meta: dict[str, object] = {"content_blocks": [{"type": "tool_use", "name": "search", "id": "t1"}]}
        archive = _seed_db(
            tmp_path,
            [
                ("claude-ai", "assistant", "Using a tool", meta),
            ],
        )
        results = await _coverage(archive)
        assert results[0].tool_use_count == 1
        assert results[0].total_sessions_with_tools == 1

    @pytest.mark.asyncio
    async def test_thinking_in_provider_meta_detected(self: object, tmp_path: Path) -> None:
        """Thinking blocks in provider_meta are detected."""
        meta: dict[str, object] = {"content_blocks": [{"type": "thinking", "thinking": "Let me consider..."}]}
        archive = _seed_db(
            tmp_path,
            [
                ("claude-ai", "assistant", "Here's my answer", meta),
            ],
        )
        results = await _coverage(archive)
        assert results[0].thinking_count == 1
        assert results[0].total_sessions_with_thinking == 1

    @pytest.mark.asyncio
    async def test_mixed_content_blocks_counted_correctly(self: object, tmp_path: Path) -> None:
        """Message with both tool_use and thinking blocks counts both."""
        meta: dict[str, object] = {
            "content_blocks": [
                {"type": "thinking", "thinking": "Planning..."},
                {"type": "tool_use", "name": "search", "id": "t1"},
                {"type": "text", "text": "Result"},
            ]
        }
        archive = _seed_db(
            tmp_path,
            [
                ("claude-ai", "assistant", "Result from tool", meta),
            ],
        )
        results = await _coverage(archive)
        assert results[0].tool_use_count == 1
        assert results[0].thinking_count == 1


# ============================================================================
# Cross-provider integration test
# ============================================================================


class TestCrossProviderConsistency:
    """Verify detection works across different provider data structures."""

    @pytest.mark.asyncio
    async def test_multiple_providers_with_tool_use(self: object, tmp_path: Path) -> None:
        """Tool use is detected correctly across ChatGPT and Claude providers."""
        chatgpt_meta: dict[str, object] = {"content_blocks": [{"type": "tool_use", "name": "browser"}]}
        claude_meta: dict[str, object] = {"content_blocks": [{"type": "tool_use", "name": "computer", "id": "toolu_1"}]}

        archive = _seed_db(
            tmp_path,
            [
                ("chatgpt", "assistant", "ChatGPT used a tool", chatgpt_meta),
                ("chatgpt", "user", "Thanks", None),
                ("claude-ai", "assistant", "Claude used a tool", claude_meta),
                ("claude-ai", "user", "Thanks", None),
            ],
        )
        results = await _coverage(archive)

        by_provider = {r.source_name: r for r in results}
        assert by_provider["chatgpt"].tool_use_count == 1
        assert by_provider["claude-ai"].tool_use_count == 1
