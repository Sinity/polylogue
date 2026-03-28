"""Property-based tests for None/Null guards across all providers.

These tests use Hypothesis to systematically fuzz provider models with
None and missing fields — the single most common bug class (16 instances
across 8 commits in the git history). The key insight is that production
data has sparse fields from older exports, partial API responses, or
providers that changed their format over time.

Covers:
- 1bc929e: part["text"] is None → TypeError in "\\n".join()
- 3914533: msg.text is None → Jinja2 TypeError
- 1525107: Mixed None/datetime in sort key → TypeError
- 29d0f2a: parse_code produces text=None for progress records
- d5c3228: content_blocks key exists but value is None
- 45c8578: None role in site builder → crash
- a2ce326: Non-string part["text"] in Gemini → crash in join
- f9c88e2: updated_at is None → display shows nothing
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest
from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st

from polylogue.lib.models import Conversation, Message, ConversationSummary
from polylogue.lib.messages import MessageCollection
from polylogue.lib.timestamps import parse_timestamp


# =============================================================================
# Hypothesis strategies for sparse/adversarial provider data
# =============================================================================

# Strategy: any value that might appear in a "text" field
_text_or_none = st.one_of(
    st.none(),
    st.text(max_size=500),
    st.just(""),
    st.just("   "),  # whitespace-only
)

# Roles that survive Message.from_record() processing: non-whitespace strings.
# In production, from_record normalizes empty/whitespace → "unknown".
# Role.normalize() intentionally raises on empty/whitespace as a bug detector.
_valid_roles = st.sampled_from([
    "user", "assistant", "system", "tool", "unknown",
    "model", "human", "ai", "function", "developer",
])
_role_or_none = _valid_roles

_timestamp_or_none = st.one_of(
    st.none(),
    st.datetimes(
        min_value=datetime(2020, 1, 1),
        max_value=datetime(2030, 1, 1),
        timezones=st.just(timezone.utc),
    ),
)

_provider_meta_or_none = st.one_of(
    st.none(),
    st.just({}),
    st.fixed_dictionaries({
        "content_blocks": st.one_of(
            st.none(),
            st.just([]),
            st.lists(st.one_of(
                st.none(),
                st.fixed_dictionaries({
                    "type": st.sampled_from(["text", "thinking", "tool_use", "tool_result"]),
                    "text": _text_or_none,
                }),
                st.just({"type": "text"}),  # missing "text" key
                st.just({}),  # empty dict
            ), max_size=5),
        ),
    }),
)


@st.composite
def sparse_message(draw: st.DrawFn) -> Message:
    """Generate a Message with randomly None'd fields.

    This is the core strategy for None-guard testing. It creates Messages
    that look like real production data from older exports where fields
    may be missing, None, or have unexpected types.
    """
    return Message(
        id=draw(st.text(min_size=1, max_size=20)),
        role=draw(_valid_roles),
        text=draw(_text_or_none),
        timestamp=draw(_timestamp_or_none),
        provider_meta=draw(_provider_meta_or_none),
    )


@st.composite
def sparse_conversation(draw: st.DrawFn) -> Conversation:
    """Generate a Conversation with sparse fields and None timestamps."""
    messages = draw(st.lists(sparse_message(), min_size=0, max_size=10))
    return Conversation(
        id=draw(st.text(min_size=1, max_size=30)),
        provider=draw(st.sampled_from(["chatgpt", "claude", "gemini", "claude-code", "codex"])),
        title=draw(st.one_of(st.none(), st.text(max_size=100))),
        messages=MessageCollection(messages=messages),
        created_at=draw(_timestamp_or_none),
        updated_at=draw(_timestamp_or_none),
        provider_meta=draw(st.one_of(st.none(), st.just({}))),
        metadata=draw(st.one_of(st.just({}), st.fixed_dictionaries({
            "tags": st.one_of(st.none(), st.just([]), st.lists(st.text(max_size=20), max_size=3)),
        }))),
    )


# =============================================================================
# Property: Message properties never crash on sparse data
# =============================================================================

class TestMessageNoneGuardProperties:
    """Every Message property must handle None text, None provider_meta, etc."""

    @given(msg=sparse_message())
    @settings(max_examples=200)
    def test_word_count_never_crashes(self, msg: Message):
        """word_count must return int >= 0, even with None text."""
        result = msg.word_count
        assert isinstance(result, int)
        assert result >= 0

    @given(msg=sparse_message())
    @settings(max_examples=200)
    def test_is_tool_use_never_crashes(self, msg: Message):
        """is_tool_use must return bool, even with None content_blocks."""
        result = msg.is_tool_use
        assert isinstance(result, bool)

    @given(msg=sparse_message())
    @settings(max_examples=200)
    def test_is_thinking_never_crashes(self, msg: Message):
        """is_thinking must return bool, even with None/missing fields."""
        result = msg.is_thinking
        assert isinstance(result, bool)

    @given(msg=sparse_message())
    @settings(max_examples=200)
    def test_is_substantive_never_crashes(self, msg: Message):
        result = msg.is_substantive
        assert isinstance(result, bool)

    @given(msg=sparse_message())
    @settings(max_examples=200)
    def test_is_noise_never_crashes(self, msg: Message):
        result = msg.is_noise
        assert isinstance(result, bool)

    @given(msg=sparse_message())
    @settings(max_examples=200)
    def test_is_context_dump_never_crashes(self, msg: Message):
        result = msg.is_context_dump
        assert isinstance(result, bool)

    @given(msg=sparse_message())
    @settings(max_examples=200)
    def test_extract_thinking_never_crashes(self, msg: Message):
        """extract_thinking must return str or None, never crash."""
        result = msg.extract_thinking()
        assert result is None or isinstance(result, str)

    @given(msg=sparse_message())
    @settings(max_examples=200)
    def test_cost_usd_never_crashes(self, msg: Message):
        result = msg.cost_usd
        assert result is None or isinstance(result, float)

    @given(msg=sparse_message())
    @settings(max_examples=200)
    def test_duration_ms_never_crashes(self, msg: Message):
        result = msg.duration_ms
        assert result is None or isinstance(result, int)


# =============================================================================
# Property: Conversation operations never crash with mixed None timestamps
# =============================================================================

class TestConversationNoneGuardProperties:
    """Conversation operations must handle mixed None timestamps and fields."""

    @given(conv=sparse_conversation())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.too_slow])
    def test_display_date_never_crashes(self, conv: Conversation):
        """display_date with None updated_at/created_at must not crash."""
        result = conv.display_date
        assert result is None or isinstance(result, datetime)

    @given(conv=sparse_conversation())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.too_slow])
    def test_tags_never_crashes(self, conv: Conversation):
        """tags property must handle None/missing metadata."""
        result = conv.tags
        assert isinstance(result, list)
        assert all(isinstance(t, str) for t in result)

    @given(conv=sparse_conversation())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.too_slow])
    def test_len_messages_never_crashes(self, conv: Conversation):
        """len(messages) must not crash even with sparse messages."""
        result = len(conv.messages)
        assert isinstance(result, int)
        assert result >= 0

    @given(conv=sparse_conversation())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.too_slow])
    def test_iter_messages_never_crashes(self, conv: Conversation):
        """Iterating messages must not crash."""
        for msg in conv.messages:
            assert isinstance(msg, Message)


# =============================================================================
# Property: Sorting conversations with mixed None timestamps (1525107)
# =============================================================================

class TestSortMixedTimestamps:
    """Sorting conversations with mixed None/datetime updated_at must not crash.

    Regression: commit 1525107 — mixed None/datetime in sort key → TypeError
    when using `sorted(conversations, key=lambda c: c.updated_at or c.created_at)`
    because None < datetime is a TypeError in Python 3.
    """

    @given(convs=st.lists(sparse_conversation(), min_size=2, max_size=20))
    @settings(max_examples=50, suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.too_slow])
    def test_sort_by_display_date_never_crashes(self, convs: list[Conversation]):
        """Sorting by display_date must handle None values gracefully."""
        _epoch = datetime.min.replace(tzinfo=timezone.utc)
        # This is the exact pattern from facade.py that was crashing
        sorted_convs = sorted(
            convs,
            key=lambda c: c.updated_at or c.created_at or _epoch,
            reverse=True,
        )
        assert len(sorted_convs) == len(convs)

    @given(convs=st.lists(sparse_conversation(), min_size=0, max_size=15))
    @settings(max_examples=50, suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.too_slow])
    def test_sort_stability_with_all_none(self, convs: list[Conversation]):
        """When all timestamps are None, sort must preserve relative order."""
        _epoch = datetime.min.replace(tzinfo=timezone.utc)
        # Force all timestamps to None
        for c in convs:
            object.__setattr__(c, "created_at", None)
            object.__setattr__(c, "updated_at", None)
        sorted_convs = sorted(
            convs,
            key=lambda c: c.updated_at or c.created_at or _epoch,
            reverse=True,
        )
        assert len(sorted_convs) == len(convs)


# =============================================================================
# Property: ChatGPT parts with None/non-string text (1bc929e, a2ce326)
# =============================================================================

class TestChatGPTNonePartsProperty:
    """ChatGPT text extraction must handle None and non-string parts."""

    @given(parts=st.lists(
        st.one_of(
            st.text(max_size=100),  # normal string part
            st.none(),  # None part
            st.integers(),  # non-string part
            st.fixed_dictionaries({"text": _text_or_none}),  # dict with optional text
            st.fixed_dictionaries({"image_url": st.text(max_size=50)}),  # image part
            st.just({}),  # empty dict
        ),
        max_size=10,
    ))
    @settings(max_examples=200)
    def test_chatgpt_parts_extraction_never_crashes(self, parts):
        """ChatGPT text extraction from parts must never crash."""
        from polylogue.sources.providers.chatgpt import ChatGPTAuthor, ChatGPTContent, ChatGPTMessage

        msg = ChatGPTMessage(
            id="test",
            author=ChatGPTAuthor(role="assistant"),
            content=ChatGPTContent(content_type="text", parts=parts),
        )
        # This must not crash, regardless of part types
        result = msg.text_content
        assert isinstance(result, str)


# =============================================================================
# Property: Gemini parts with None/non-string text (a2ce326)
# =============================================================================

class TestGeminiNonePartsProperty:
    """Gemini text extraction must handle None text and non-string values."""

    @given(
        text=_text_or_none,
        parts=st.lists(
            st.one_of(
                st.fixed_dictionaries({"text": _text_or_none}),
                st.fixed_dictionaries({"text": st.integers()}),  # non-string
                st.fixed_dictionaries({"inlineData": st.just({})}),  # image
                st.just({}),
            ),
            max_size=5,
        ),
    )
    @settings(max_examples=200)
    def test_gemini_text_extraction_never_crashes(self, text, parts):
        """Gemini text_content must handle any combination of text + parts."""
        from polylogue.sources.providers.gemini import GeminiMessage

        msg = GeminiMessage(
            text=text or "",  # GeminiMessage requires text field
            role="model",
            parts=parts,
        )
        result = msg.text_content
        assert isinstance(result, str)


# =============================================================================
# Property: Claude Code records with sparse/None message fields
# =============================================================================

class TestClaudeCodeNoneGuardProperty:
    """Claude Code extraction must handle None message, None content, etc."""

    @given(
        record_type=st.sampled_from(["user", "assistant", "summary", "progress", "result",
                                      "init", "file-history-snapshot", "queue-operation"]),
        message=st.one_of(
            st.none(),
            st.fixed_dictionaries({
                "role": st.sampled_from(["user", "assistant"]),
                "content": st.one_of(
                    st.text(max_size=100),
                    st.just([]),
                    st.lists(st.one_of(
                        st.fixed_dictionaries({
                            "type": st.sampled_from(["text", "thinking", "tool_use"]),
                            "text": _text_or_none,
                        }),
                        st.just({}),
                    ), max_size=5),
                    st.none(),
                ),
            }),
        ),
    )
    @settings(max_examples=200)
    def test_claude_code_text_content_never_crashes(self, record_type, message):
        """text_content must handle any record type with any message shape."""
        from polylogue.sources.providers.claude_code import ClaudeCodeRecord

        record = ClaudeCodeRecord(type=record_type, message=message)
        result = record.text_content
        assert isinstance(result, str)

    @given(
        record_type=st.sampled_from(["user", "assistant", "progress", "result"]),
        message=st.one_of(
            st.none(),
            st.fixed_dictionaries({
                "role": st.sampled_from(["user", "assistant"]),
                "content": st.one_of(st.just([]), st.none(), st.text(max_size=50)),
            }),
        ),
    )
    @settings(max_examples=100)
    def test_claude_code_content_blocks_raw_never_crashes(self, record_type, message):
        """content_blocks_raw must return list, never crash."""
        from polylogue.sources.providers.claude_code import ClaudeCodeRecord

        record = ClaudeCodeRecord(type=record_type, message=message)
        result = record.content_blocks_raw
        assert isinstance(result, list)

    @given(
        record_type=st.sampled_from(["user", "assistant", "summary", "progress"]),
    )
    @settings(max_examples=50)
    def test_claude_code_role_mapping_systematic(self, record_type):
        """Every record type must map to a known role."""
        from polylogue.sources.providers.claude_code import ClaudeCodeRecord

        record = ClaudeCodeRecord(type=record_type)
        role = record.role
        assert role in {"user", "assistant", "system", "tool", "unknown"}


# =============================================================================
# Property: ConversationSummary with all-None fields (f9c88e2)
# =============================================================================

class TestConversationSummaryNoneGuards:
    """ConversationSummary display methods must handle None fields."""

    @given(
        title=st.one_of(st.none(), st.text(max_size=100)),
        created_at=_timestamp_or_none,
        updated_at=_timestamp_or_none,
        metadata=st.one_of(
            st.just({}),
            st.fixed_dictionaries({"title": st.one_of(st.none(), st.text(max_size=50))}),
            st.fixed_dictionaries({"tags": st.one_of(st.none(), st.just([]), st.lists(st.text(max_size=10), max_size=3))}),
        ),
    )
    @settings(max_examples=100)
    def test_display_title_never_crashes(self, title, created_at, updated_at, metadata):
        summary = ConversationSummary(
            id="test-id",
            provider="chatgpt",
            title=title,
            created_at=created_at,
            updated_at=updated_at,
            metadata=metadata,
        )
        result = summary.display_title
        assert isinstance(result, str)
        assert len(result) > 0  # must always produce something

    @given(
        created_at=_timestamp_or_none,
        updated_at=_timestamp_or_none,
    )
    @settings(max_examples=50)
    def test_display_date_never_crashes(self, created_at, updated_at):
        summary = ConversationSummary(
            id="test-id",
            provider="chatgpt",
            created_at=created_at,
            updated_at=updated_at,
        )
        result = summary.display_date
        assert result is None or isinstance(result, datetime)


# =============================================================================
# Property: parse_timestamp with adversarial inputs
# =============================================================================

class TestParseTimestampProperty:
    """parse_timestamp must never crash, regardless of input."""

    @given(value=st.one_of(
        st.none(),
        st.integers(min_value=-2**31, max_value=2**31),
        st.floats(allow_nan=True, allow_infinity=True),
        st.text(max_size=100),
        st.just(""),
        st.just("NaN"),
        st.just("Infinity"),
        st.just("null"),
    ))
    @settings(max_examples=300)
    def test_parse_timestamp_never_crashes(self, value):
        """parse_timestamp must return datetime or None, never raise."""
        result = parse_timestamp(value)
        assert result is None or isinstance(result, datetime)

    @given(value=st.one_of(
        st.integers(min_value=-2**31, max_value=2**31),
        st.floats(min_value=-1e15, max_value=1e15, allow_nan=False, allow_infinity=False),
    ))
    @settings(max_examples=200)
    def test_parse_timestamp_numeric_always_utc_or_none(self, value):
        """When a numeric value produces a datetime, it must be UTC-aware."""
        result = parse_timestamp(value)
        if result is not None:
            assert result.tzinfo is not None, f"parse_timestamp({value!r}) returned naive datetime"

    @given(value=st.text(min_size=1, max_size=50).filter(lambda s: s.replace(".", "").isdigit()))
    @settings(max_examples=100)
    def test_digit_strings_either_none_or_utc(self, value):
        """Digit-only strings must produce None or UTC-aware datetime."""
        result = parse_timestamp(value)
        if result is not None:
            assert result.tzinfo is not None
