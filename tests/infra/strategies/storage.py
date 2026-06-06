"""Hypothesis strategies and helpers for storage-law tests."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Final

from hypothesis import strategies as st

from tests.infra.storage_records import SessionBuilder

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
class SessionSpec:
    """Generated session fixture for storage/repository laws."""

    session_id: str
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
    """Generated tag distribution over a session graph."""

    sessions: tuple[SessionSpec, ...]
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
def session_graph_strategy(
    draw: st.DrawFn,
    *,
    min_sessions: int = 1,
    max_sessions: int = 6,
    max_messages: int = 4,
) -> tuple[SessionSpec, ...]:
    """Generate a small acyclic session graph with stable ordering."""
    count = draw(st.integers(min_value=min_sessions, max_value=max_sessions))
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    specs: list[SessionSpec] = []

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
            SessionSpec(
                session_id=f"conv-{index}-{slug}",
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
    min_sessions: int = 1,
    max_sessions: int = 6,
) -> TagAssignmentSpec:
    """Generate a session graph plus per-session tag sequences."""
    sessions = draw(
        session_graph_strategy(
            min_sessions=min_sessions,
            max_sessions=max_sessions,
        )
    )
    tag_sequences = []
    for _ in sessions:
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
        sessions=sessions,
        tag_sequences=tuple(tag_sequences),
    )


def expected_sorted_ids(specs: tuple[SessionSpec, ...]) -> list[str]:
    """Return repository/backend default sort order for the generated graph."""
    return [
        spec.session_id for spec in sorted(specs, key=lambda spec: (spec.updated_at, spec.session_id), reverse=True)
    ]


def root_index(specs: tuple[SessionSpec, ...], index: int) -> int:
    """Resolve the root session index for a generated node."""
    current = index
    while specs[current].parent_index is not None:
        parent_index = specs[current].parent_index
        assert parent_index is not None
        current = parent_index
    return current


def expected_tree_ids(specs: tuple[SessionSpec, ...], index: int) -> set[str]:
    """Return the full tree rooted at the session containing index."""
    expected_root = root_index(specs, index)
    return {spec.session_id for position, spec in enumerate(specs) if root_index(specs, position) == expected_root}


def shortest_unique_prefix(ids: tuple[str, ...], target_id: str) -> str:
    """Return the shortest prefix that uniquely identifies ``target_id``."""
    for length in range(1, len(target_id) + 1):
        prefix = target_id[:length]
        if sum(candidate.startswith(prefix) for candidate in ids) == 1:
            return prefix
    return target_id


def expected_tag_counts(spec: TagAssignmentSpec, provider: str | None = None) -> dict[str, int]:
    """Count tags by distinct session, optionally restricted to one provider."""
    counts: dict[str, int] = {}
    for session, tags in zip(spec.sessions, spec.tag_sequences, strict=True):
        if provider is not None and session.provider != provider:
            continue
        for tag in set(tags):
            counts[tag] = counts.get(tag, 0) + 1
    return counts


def seed_session_graph(db_path: Path, specs: tuple[SessionSpec, ...]) -> None:
    """Persist a generated graph through the same test builder path as other suites."""
    for spec in specs:
        builder = (
            SessionBuilder(db_path, spec.session_id)
            .provider(spec.provider)
            .title(spec.title)
            .created_at(spec.created_at)
            .updated_at(spec.updated_at)
        )
        if spec.parent_index is not None:
            builder.parent_session(specs[spec.parent_index].session_id)
        for message_index, message in enumerate(spec.messages):
            builder.add_message(
                f"{spec.session_id}-m{message_index}",
                role=message.role,
                text=message.text,
                has_tool_use=int(message.has_tool_use),
                has_thinking=int(message.has_thinking),
            )
        builder.save()

    # The archive ArchiveStore write maintains block FTS via triggers and resolves
    # parent/topology edges on save; refresh the FTS index explicitly so any
    # bulk-suspended triggers are reconciled before tree/law assertions read.
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    with ArchiveStore(db_path.parent) as archive:
        archive.rebuild_index()
