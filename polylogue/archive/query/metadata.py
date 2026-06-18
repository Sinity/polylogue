"""Lightweight query grammar metadata for completion and docs consumers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

QueryUnitName = Literal["message", "action", "block", "assertion"]

#: Recognized DSL field tokens and a short human description.
#: ``spec_field`` values correspond to :class:`~polylogue.archive.query.spec.SessionQuerySpec`
#: attribute names or the special tokens ``"min_messages"``, ``"max_messages"`` etc.
#: for count-comparison fields.
EXPRESSION_FIELD_REGISTRY: dict[str, dict[str, str]] = {
    "repo": {
        "description": "Filter by repository name (substring match)",
        "spec_field": "repo_names",
        "negatable": "no",
        "example": "repo:polylogue",
    },
    "origin": {
        "description": "Filter by session origin",
        "spec_field": "origins",
        "negatable": "yes",
        "example": "origin:claude-code-session",
    },
    "tag": {
        "description": "Filter by session tag",
        "spec_field": "tags",
        "negatable": "yes",
        "example": "tag:review",
    },
    "path": {
        "description": "Filter by referenced file path",
        "spec_field": "referenced_path",
        "negatable": "no",
        "example": "path:polylogue/cli",
    },
    "cwd": {
        "description": "Filter by working directory prefix",
        "spec_field": "cwd_prefix",
        "negatable": "no",
        "example": "cwd:/realm/project",
    },
    "tool": {
        "description": "Filter by tool name used in session",
        "spec_field": "tool_terms",
        "negatable": "yes",
        "example": "tool:bash",
    },
    "action": {
        "description": "Filter by action category",
        "spec_field": "action_terms",
        "negatable": "yes",
        "example": "action:file_edit",
    },
    "has": {
        "description": "Filter by session content presence (paste, tools, thinking)",
        "spec_field": "filter_has_paste/filter_has_tool_use/filter_has_thinking/has_types",
        "negatable": "no",
        "example": "has:paste",
    },
    "id": {
        "description": "Filter by session id",
        "spec_field": "session_id",
        "negatable": "no",
        "example": "id:abc123",
    },
    "title": {
        "description": "Filter by session title (substring)",
        "spec_field": "title",
        "negatable": "no",
        "example": "title:refactor",
    },
    "since": {
        "description": "Filter sessions after date (ISO or relative: 7d, 2w)",
        "spec_field": "since",
        "negatable": "no",
        "example": "since:7d",
    },
    "until": {
        "description": "Filter sessions before date (ISO or relative: 2d)",
        "spec_field": "until",
        "negatable": "no",
        "example": "until:2024-01-15",
    },
    "near": {
        "description": (
            'Similarity search. Free text (near:"...") seeds vector similarity '
            "from a text string; near:id:<ref> seeds it from a stored session"
        ),
        "spec_field": "similar_text/similar_session_id",
        "negatable": "no",
        "example": 'near:"semantic search" | near:id:abc123',
    },
    "contains": {
        "description": "Filter by exact content substring",
        "spec_field": "contains_terms",
        "negatable": "no",
        "example": "contains:foo",
    },
    "messages": {
        "description": "Count comparison for message count (>=N or <=N)",
        "spec_field": "min_messages/max_messages",
        "negatable": "no",
        "example": "messages:>=10",
    },
    "words": {
        "description": "Count comparison for word count (>=N, <=N, or =N)",
        "spec_field": "min_words/max_words",
        "negatable": "no",
        "example": "words:>=200",
    },
    "lane": {
        "description": "Force retrieval lane (auto, dialogue, hybrid)",
        "spec_field": "retrieval_lane",
        "negatable": "no",
        "example": "lane:dialogue",
    },
    "lineage": {
        "description": "Filter to sessions sharing a logical lineage with a seed session id",
        "spec_field": "boolean_predicate",
        "negatable": "no",
        "example": "lineage:id:chatgpt-export:ext-root",
    },
}

_SOURCE_WHERE_SOURCES: tuple[tuple[str, QueryUnitName], ...] = (
    ("message", "message"),
    ("messages", "message"),
    ("action", "action"),
    ("actions", "action"),
    ("block", "block"),
    ("blocks", "block"),
    ("assertion", "assertion"),
    ("assertions", "assertion"),
)
_BOOLEAN_SUPPORTED_FIELDS = {
    "repo",
    "origin",
    "tag",
    "path",
    "cwd",
    "tool",
    "action",
    "has",
    "id",
    "title",
    "date",
    "since",
    "until",
    "messages",
    "words",
    "lineage",
}
_STRUCTURAL_BOOLEAN_SUPPORTED_FIELDS = {
    "role",
    "type",
    "text",
    "tool",
    "action",
    "command",
    "path",
    "output",
    "words",
}
_MESSAGE_STRUCTURAL_FIELDS = {"role", "type", "text", "tool", "action", "command", "path", "output", "words"}
_ACTION_STRUCTURAL_FIELDS = {"tool", "action", "type", "command", "path", "output", "text"}
_BLOCK_STRUCTURAL_FIELDS = {"type", "text", "tool", "action", "command", "path"}
_ASSERTION_STRUCTURAL_FIELDS = {
    "kind",
    "status",
    "target",
    "target_ref",
    "scope",
    "scope_ref",
    "key",
    "text",
    "body",
    "author",
    "author_ref",
    "author_kind",
    "visibility",
    "value",
    "evidence",
    "context",
}
_STRUCTURAL_BOOLEAN_SUPPORTED_FIELDS.update(_ASSERTION_STRUCTURAL_FIELDS)
_SCOPED_SESSION_STRUCTURAL_FIELDS = tuple(f"session.{field}" for field in sorted(_BOOLEAN_SUPPORTED_FIELDS))


@dataclass(frozen=True)
class StructuralQueryUnitInfo:
    """Completion/query-builder metadata for one ``exists <unit>(...)`` unit."""

    description: str
    fields: tuple[str, ...]
    example: str


@dataclass(frozen=True)
class CountQueryFieldInfo:
    """Completion/query-builder metadata for readable count predicates."""

    description: str
    operators: tuple[str, ...]
    range_keyword: str
    example: str


@dataclass(frozen=True)
class DateQueryFieldInfo:
    """Completion/query-builder metadata for readable date predicates."""

    description: str
    operators: tuple[str, ...]
    range_keyword: str
    example: str


STRUCTURAL_QUERY_UNIT_REGISTRY: dict[str, StructuralQueryUnitInfo] = {
    "action": StructuralQueryUnitInfo(
        description="Match sessions with at least one action row satisfying the child predicate.",
        fields=tuple(sorted((*_ACTION_STRUCTURAL_FIELDS, *_SCOPED_SESSION_STRUCTURAL_FIELDS))),
        example="exists action(session.repo:polylogue AND tool:bash AND text:pytest)",
    ),
    "block": StructuralQueryUnitInfo(
        description="Match sessions with at least one parsed message block satisfying the child predicate.",
        fields=tuple(sorted((*_BLOCK_STRUCTURAL_FIELDS, *_SCOPED_SESSION_STRUCTURAL_FIELDS))),
        example="exists block(session.origin:codex-session AND type:code AND text:timeout)",
    ),
    "message": StructuralQueryUnitInfo(
        description="Match sessions with at least one message satisfying the child predicate.",
        fields=tuple(sorted((*_MESSAGE_STRUCTURAL_FIELDS, *_SCOPED_SESSION_STRUCTURAL_FIELDS))),
        example="exists message(session.repo:polylogue AND role:assistant AND text:timeout)",
    ),
    "assertion": StructuralQueryUnitInfo(
        description="Match sessions with at least one session-targeted assertion satisfying the child predicate.",
        fields=tuple(sorted((*_ASSERTION_STRUCTURAL_FIELDS, *_SCOPED_SESSION_STRUCTURAL_FIELDS))),
        example="exists assertion(kind:decision AND status:active AND text:review)",
    ),
}

COUNT_QUERY_FIELD_REGISTRY: dict[str, CountQueryFieldInfo] = {
    "messages": CountQueryFieldInfo(
        description="Message-count predicate over normalized session messages.",
        operators=(">=", "<=", "=", ">", "<"),
        range_keyword="between",
        example="messages between 5 and 20",
    ),
    "words": CountQueryFieldInfo(
        description="Word-count predicate over normalized session text.",
        operators=(">=", "<=", "=", ">", "<"),
        range_keyword="between",
        example="words between 100 and 500",
    ),
}

DATE_QUERY_FIELD_REGISTRY: dict[str, DateQueryFieldInfo] = {
    "date": DateQueryFieldInfo(
        description="Session timestamp predicate over COALESCE(updated_at, created_at).",
        operators=(">=", "<=", ">", "<"),
        range_keyword="between",
        example="date between 2026-01-01 and 2026-02-01",
    ),
}


def structural_query_units() -> tuple[str, ...]:
    """Return the structural units accepted by the query grammar."""

    return tuple(sorted(STRUCTURAL_QUERY_UNIT_REGISTRY))


def structural_query_fields(unit: str) -> tuple[str, ...]:
    """Return structural field names accepted inside ``exists <unit>(...)``."""

    info = STRUCTURAL_QUERY_UNIT_REGISTRY.get(unit.lower())
    if info is None:
        return ()
    return info.fields


def count_query_fields() -> tuple[str, ...]:
    """Return count fields with readable comparison/range syntax."""

    return tuple(sorted(COUNT_QUERY_FIELD_REGISTRY))


def count_query_operators(field: str) -> tuple[str, ...]:
    """Return readable comparison/range operators accepted for a count field."""

    info = COUNT_QUERY_FIELD_REGISTRY.get(field.lower())
    if info is None:
        return ()
    return (*info.operators, info.range_keyword)


def date_query_fields() -> tuple[str, ...]:
    """Return date fields with readable comparison/range syntax."""

    return tuple(sorted(DATE_QUERY_FIELD_REGISTRY))


def date_query_operators(field: str) -> tuple[str, ...]:
    """Return readable comparison/range operators accepted for a date field."""

    info = DATE_QUERY_FIELD_REGISTRY.get(field.lower())
    if info is None:
        return ()
    return (*info.operators, info.range_keyword)


__all__ = [
    "COUNT_QUERY_FIELD_REGISTRY",
    "CountQueryFieldInfo",
    "DATE_QUERY_FIELD_REGISTRY",
    "DateQueryFieldInfo",
    "EXPRESSION_FIELD_REGISTRY",
    "QueryUnitName",
    "STRUCTURAL_QUERY_UNIT_REGISTRY",
    "StructuralQueryUnitInfo",
    "_BOOLEAN_SUPPORTED_FIELDS",
    "_ACTION_STRUCTURAL_FIELDS",
    "_ASSERTION_STRUCTURAL_FIELDS",
    "_BLOCK_STRUCTURAL_FIELDS",
    "_MESSAGE_STRUCTURAL_FIELDS",
    "_SOURCE_WHERE_SOURCES",
    "_STRUCTURAL_BOOLEAN_SUPPORTED_FIELDS",
    "count_query_fields",
    "count_query_operators",
    "date_query_fields",
    "date_query_operators",
    "structural_query_fields",
    "structural_query_units",
]
