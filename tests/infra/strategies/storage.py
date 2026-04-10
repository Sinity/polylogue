"""Hypothesis strategies and helpers for storage-law tests."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Final

from hypothesis import strategies as st

from polylogue.storage.backends.connection import open_connection
from polylogue.storage.index import rebuild_index
from tests.infra.storage_records import ConversationBuilder

_PROVIDERS: Final[tuple[str, ...]] = ("claude-ai", "chatgpt", "codex", "claude-code")
_ROLES: Final[tuple[str, ...]] = ("user", "assistant", "system")
_TITLE_ALPHABET: Final[str] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -_"
_TEXT_ALPHABET: Final[str] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,?!-_/:\n"
_TAG_ALPHABET: Final[str] = "abcdefghijklmnopqrstuvwxyz0123456789-_"
_LITERAL_NEEDLE_ALPHABET: Final[str] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
_SPECIAL_QUERY_CHARS: Final[tuple[str, ...]] = ("%", "_", "\\")


@dataclass(frozen=True)
class MessageSpec:
    """Minimal message shape for storage contract tests."""

    role: str
    text: str
    has_tool_use: bool
    has_thinking: bool


@dataclass(frozen=True)
class ConversationSpec:
    """Generated conversation fixture for storage/repository laws."""

    conversation_id: str
    provider: str
    title: str
    created_at: str
    updated_at: str
    parent_index: int | None
    messages: tuple[MessageSpec, ...]


@dataclass(frozen=True)
class TitleSearchSpec:
    """Literal title-query case for wildcard/escape contracts."""

    needle: str
    matching_title: str
    decoy_title: str


@dataclass(frozen=True)
class TagAssignmentSpec:
    """Generated tag distribution over a conversation graph."""

    conversations: tuple[ConversationSpec, ...]
    tag_sequences: tuple[tuple[str, ...], ...]


@st.composite
def message_spec_strategy(draw: st.DrawFn) -> MessageSpec:
    """Generate a message with explicit analytics fields."""
    role = draw(st.sampled_from(_ROLES))
    has_tool_use = draw(st.booleans())
    has_thinking = draw(st.booleans())
    text = draw(st.text(alphabet=_TEXT_ALPHABET, min_size=1, max_size=80).filter(lambda value: value.strip() != ""))
    return MessageSpec(
        role=role,
        text=text,
        has_tool_use=has_tool_use or role == "tool",
        has_thinking=has_thinking,
    )


@st.composite
def conversation_graph_strategy(
    draw: st.DrawFn,
    *,
    min_conversations: int = 1,
    max_conversations: int = 6,
    max_messages: int = 4,
) -> tuple[ConversationSpec, ...]:
    """Generate a small acyclic conversation graph with stable ordering."""
    count = draw(st.integers(min_value=min_conversations, max_value=max_conversations))
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    specs: list[ConversationSpec] = []

    for index in range(count):
        slug = draw(st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789", min_size=3, max_size=8))
        parent_index = None
        if index > 0:
            parent_index = draw(st.one_of(st.none(), st.integers(min_value=0, max_value=index - 1)))
        minute_offset = draw(st.integers(min_value=0, max_value=59))
        timestamp = (base + timedelta(days=index, minutes=minute_offset)).isoformat()
        messages = tuple(
            draw(
                st.lists(
                    message_spec_strategy(),
                    min_size=1,
                    max_size=max_messages,
                )
            )
        )
        specs.append(
            ConversationSpec(
                conversation_id=f"conv-{index}-{slug}",
                provider=draw(st.sampled_from(_PROVIDERS)),
                title=draw(
                    st.text(alphabet=_TITLE_ALPHABET, min_size=1, max_size=32).filter(lambda value: value.strip() != "")
                ),
                created_at=timestamp,
                updated_at=timestamp,
                parent_index=parent_index,
                messages=messages,
            )
        )

    return tuple(specs)


@st.composite
def literal_title_search_strategy(draw: st.DrawFn) -> TitleSearchSpec:
    """Generate a literal title search containing wildcard-sensitive characters."""
    prefix = draw(st.text(alphabet=_LITERAL_NEEDLE_ALPHABET, min_size=1, max_size=6))
    suffix = draw(st.text(alphabet=_LITERAL_NEEDLE_ALPHABET, min_size=1, max_size=6))
    special = draw(st.sampled_from(_SPECIAL_QUERY_CHARS))
    needle = f"{prefix}{special}{suffix}"
    left = draw(st.text(alphabet=f"{_LITERAL_NEEDLE_ALPHABET} -_/", min_size=0, max_size=8))
    right = draw(st.text(alphabet=f"{_LITERAL_NEEDLE_ALPHABET} -_/", min_size=0, max_size=8))
    matching_title = left + needle + right
    replacement = draw(st.sampled_from(tuple(char for char in "xyz" if char != special)))
    decoy_title = left + f"{prefix}{replacement}{suffix}" + right
    return TitleSearchSpec(
        needle=needle,
        matching_title=matching_title,
        decoy_title=decoy_title,
    )


@st.composite
def tag_assignment_strategy(
    draw: st.DrawFn,
    *,
    min_conversations: int = 1,
    max_conversations: int = 6,
) -> TagAssignmentSpec:
    """Generate a conversation graph plus per-conversation tag sequences."""
    conversations = draw(
        conversation_graph_strategy(
            min_conversations=min_conversations,
            max_conversations=max_conversations,
        )
    )
    tag_sequences = []
    for _ in conversations:
        tags = tuple(
            draw(
                st.lists(
                    st.text(alphabet=_TAG_ALPHABET, min_size=1, max_size=10),
                    min_size=0,
                    max_size=4,
                )
            )
        )
        tag_sequences.append(tags)
    return TagAssignmentSpec(
        conversations=conversations,
        tag_sequences=tuple(tag_sequences),
    )


def expected_sorted_ids(specs: tuple[ConversationSpec, ...]) -> list[str]:
    """Return repository/backend default sort order for the generated graph."""
    return [
        spec.conversation_id
        for spec in sorted(specs, key=lambda spec: (spec.updated_at, spec.conversation_id), reverse=True)
    ]


def root_index(specs: tuple[ConversationSpec, ...], index: int) -> int:
    """Resolve the root conversation index for a generated node."""
    current = index
    while specs[current].parent_index is not None:
        current = specs[current].parent_index
    return current


def expected_tree_ids(specs: tuple[ConversationSpec, ...], index: int) -> set[str]:
    """Return the full tree rooted at the conversation containing index."""
    expected_root = root_index(specs, index)
    return {spec.conversation_id for position, spec in enumerate(specs) if root_index(specs, position) == expected_root}


def shortest_unique_prefix(ids: tuple[str, ...], target_id: str) -> str:
    """Return the shortest prefix that uniquely identifies ``target_id``."""
    for length in range(1, len(target_id) + 1):
        prefix = target_id[:length]
        if sum(candidate.startswith(prefix) for candidate in ids) == 1:
            return prefix
    return target_id


def expected_tag_counts(spec: TagAssignmentSpec, provider: str | None = None) -> dict[str, int]:
    """Count tags by distinct conversation, optionally restricted to one provider."""
    counts: dict[str, int] = {}
    for conversation, tags in zip(spec.conversations, spec.tag_sequences, strict=True):
        if provider is not None and conversation.provider != provider:
            continue
        for tag in set(tags):
            counts[tag] = counts.get(tag, 0) + 1
    return counts


def seed_conversation_graph(db_path: Path, specs: tuple[ConversationSpec, ...]) -> None:
    """Persist a generated graph through the same test builder path as other suites."""
    for spec in specs:
        builder = (
            ConversationBuilder(db_path, spec.conversation_id)
            .provider(spec.provider)
            .title(spec.title)
            .created_at(spec.created_at)
            .updated_at(spec.updated_at)
        )
        if spec.parent_index is not None:
            builder.parent_conversation(specs[spec.parent_index].conversation_id)
        for message_index, message in enumerate(spec.messages):
            builder.add_message(
                f"{spec.conversation_id}-m{message_index}",
                role=message.role,
                text=message.text,
                has_tool_use=int(message.has_tool_use),
                has_thinking=int(message.has_thinking),
            )
        builder.save()

    with open_connection(db_path) as conn:
        rebuild_index(conn)
