"""Lightweight query grammar metadata for completion and docs consumers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

QueryUnitName = Literal["message", "action", "block", "assertion", "observed-event", "context-snapshot"]

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
    ("observed-event", "observed-event"),
    ("observed-events", "observed-event"),
    ("context-snapshot", "context-snapshot"),
    ("context-snapshots", "context-snapshot"),
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
_OBSERVED_EVENT_STRUCTURAL_FIELDS = {
    "kind",
    "summary",
    "text",
    "delivery_state",
    "subject",
    "subject_ref",
    "object",
    "object_ref",
    "evidence",
}
_CONTEXT_SNAPSHOT_STRUCTURAL_FIELDS = {
    "boundary",
    "inheritance_mode",
    "run",
    "run_ref",
    "segment",
    "segment_ref",
    "evidence",
    "metadata",
    "text",
}
_STRUCTURAL_BOOLEAN_SUPPORTED_FIELDS.update(_ASSERTION_STRUCTURAL_FIELDS)
_STRUCTURAL_BOOLEAN_SUPPORTED_FIELDS.update(_OBSERVED_EVENT_STRUCTURAL_FIELDS)
_STRUCTURAL_BOOLEAN_SUPPORTED_FIELDS.update(_CONTEXT_SNAPSHOT_STRUCTURAL_FIELDS)


@dataclass(frozen=True)
class StructuralQueryFieldInfo:
    """Completion/query-builder metadata for one structural predicate field."""

    name: str
    description: str
    example: str


@dataclass(frozen=True)
class StructuralQueryUnitInfo:
    """Completion/query-builder metadata for one ``exists <unit>(...)`` unit."""

    description: str
    fields: tuple[StructuralQueryFieldInfo, ...]
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


def _field_info(name: str, description: str, example: str) -> StructuralQueryFieldInfo:
    return StructuralQueryFieldInfo(name=name, description=description, example=example)


_COMMON_STRUCTURAL_FIELD_INFO: dict[str, StructuralQueryFieldInfo] = {
    "action": _field_info("action", "Semantic action category on tool/action evidence.", "action:file_edit"),
    "command": _field_info("command", "Tool command or invocation text substring.", 'command:"pytest"'),
    "output": _field_info("output", "Tool output substring.", "output:FAILED"),
    "path": _field_info("path", "Referenced file or artifact path substring.", "path:polylogue/archive"),
    "role": _field_info("role", "Message role.", "role:assistant"),
    "text": _field_info("text", "Text/content substring for the selected unit.", "text:timeout"),
    "tool": _field_info("tool", "Normalized tool name.", "tool:bash"),
    "type": _field_info("type", "Unit-specific type token.", "type:code"),
    "words": _field_info("words", "Word-count predicate for the selected unit.", "words:>=200"),
}

_ASSERTION_STRUCTURAL_FIELD_INFO: dict[str, StructuralQueryFieldInfo] = {
    "author": _field_info("author", "Assertion author display/name substring.", "author:codex"),
    "author_kind": _field_info("author_kind", "Assertion author kind.", "author_kind:agent"),
    "author_ref": _field_info("author_ref", "Assertion author ObjectRef.", "author_ref:agent:codex"),
    "body": _field_info("body", "Assertion body text substring.", "body:review"),
    "context": _field_info("context", "Assertion context JSON/text substring.", "context:branch"),
    "evidence": _field_info("evidence", "Assertion evidence ref substring.", "evidence:session:"),
    "key": _field_info("key", "Assertion key string.", "key:lesson.query"),
    "kind": _field_info("kind", "Assertion kind.", "kind:decision"),
    "scope": _field_info("scope", "Assertion scope string.", "scope:repo:polylogue"),
    "scope_ref": _field_info("scope_ref", "Assertion scope ObjectRef.", "scope_ref:repo:polylogue"),
    "status": _field_info("status", "Assertion status.", "status:active"),
    "target": _field_info("target", "Assertion target string.", "target:session"),
    "target_ref": _field_info("target_ref", "Assertion target ObjectRef.", "target_ref:session:abc"),
    "text": _field_info("text", "Assertion body/value/evidence text substring.", "text:query"),
    "value": _field_info("value", "Assertion value JSON/text substring.", "value:priority"),
    "visibility": _field_info("visibility", "Assertion visibility.", "visibility:private"),
}

_OBSERVED_EVENT_STRUCTURAL_FIELD_INFO: dict[str, StructuralQueryFieldInfo] = {
    "delivery_state": _field_info(
        "delivery_state",
        "Observed-event delivery state, such as acted_on or acknowledged.",
        "delivery_state:acted_on",
    ),
    "evidence": _field_info("evidence", "Observed-event evidence ref substring.", "evidence:message:"),
    "kind": _field_info("kind", "Observed-event kind.", "kind:review_acted_on"),
    "object": _field_info("object", "Observed-event object ref substring.", "object:#2100"),
    "object_ref": _field_info("object_ref", "Observed-event object ref substring.", "object_ref:github-review:#2100"),
    "subject": _field_info("subject", "Observed-event subject ref substring.", "subject:message"),
    "subject_ref": _field_info("subject_ref", "Observed-event subject ref substring.", "subject_ref:message:"),
    "summary": _field_info("summary", "Observed-event summary substring.", "summary:review"),
    "text": _field_info("text", "Observed-event summary/ref substring.", "text:review"),
}

_CONTEXT_SNAPSHOT_STRUCTURAL_FIELD_INFO: dict[str, StructuralQueryFieldInfo] = {
    "boundary": _field_info(
        "boundary",
        "Context snapshot boundary, such as session_start or subagent_start.",
        "boundary:session_start",
    ),
    "evidence": _field_info("evidence", "Context snapshot evidence ref substring.", "evidence:message:"),
    "inheritance_mode": _field_info(
        "inheritance_mode",
        "Context inheritance mode, such as clean, summary, injected, or unknown.",
        "inheritance_mode:summary",
    ),
    "metadata": _field_info("metadata", "Context snapshot metadata key/value substring.", "metadata:branch"),
    "run": _field_info("run", "Owning run ObjectRef substring.", "run:run:"),
    "run_ref": _field_info("run_ref", "Owning run ObjectRef substring.", "run_ref:run:"),
    "segment": _field_info("segment", "Context segment ObjectRef substring.", "segment:message:"),
    "segment_ref": _field_info("segment_ref", "Context segment ObjectRef substring.", "segment_ref:message:"),
    "text": _field_info("text", "Context snapshot ref/run/segment/evidence/metadata substring.", "text:session_start"),
}

_SESSION_SCOPED_STRUCTURAL_EXAMPLES: dict[str, str] = {
    "action": "session.action:file_edit",
    "cwd": "session.cwd:/realm/project",
    "has": "session.has:tools",
    "id": "session.id:abc123",
    "messages": "session.messages:10",
    "origin": "session.origin:claude-code-session",
    "repo": "session.repo:polylogue",
    "since": "session.since:7d",
    "tag": "session.tag:review",
    "title": "session.title:refactor",
    "tool": "session.tool:bash",
    "until": "session.until:2024-01-15",
    "words": "session.words:200",
}


def _session_scoped_field_info() -> tuple[StructuralQueryFieldInfo, ...]:
    fields: list[StructuralQueryFieldInfo] = []
    for field, example in sorted(_SESSION_SCOPED_STRUCTURAL_EXAMPLES.items()):
        info = EXPRESSION_FIELD_REGISTRY.get(field)
        description = info["description"] if info is not None else f"Owning session {field} predicate."
        fields.append(
            _field_info(
                f"session.{field}",
                f"Owning session scope: {description}",
                example,
            )
        )
    return tuple(fields)


_SCOPED_SESSION_STRUCTURAL_FIELD_INFO = _session_scoped_field_info()
_SCOPED_SESSION_STRUCTURAL_FIELDS = tuple(field.name for field in _SCOPED_SESSION_STRUCTURAL_FIELD_INFO)


def _structural_field_infos(*names: str) -> tuple[StructuralQueryFieldInfo, ...]:
    infos = [_COMMON_STRUCTURAL_FIELD_INFO[name] for name in names]
    return tuple(sorted((*infos, *_SCOPED_SESSION_STRUCTURAL_FIELD_INFO), key=lambda field: field.name))


def _assertion_field_infos() -> tuple[StructuralQueryFieldInfo, ...]:
    infos = [_ASSERTION_STRUCTURAL_FIELD_INFO[name] for name in sorted(_ASSERTION_STRUCTURAL_FIELDS)]
    return tuple(sorted((*infos, *_SCOPED_SESSION_STRUCTURAL_FIELD_INFO), key=lambda field: field.name))


def _observed_event_field_infos() -> tuple[StructuralQueryFieldInfo, ...]:
    infos = [_OBSERVED_EVENT_STRUCTURAL_FIELD_INFO[name] for name in sorted(_OBSERVED_EVENT_STRUCTURAL_FIELDS)]
    return tuple(sorted((*infos, *_SCOPED_SESSION_STRUCTURAL_FIELD_INFO), key=lambda field: field.name))


def _context_snapshot_field_infos() -> tuple[StructuralQueryFieldInfo, ...]:
    infos = [_CONTEXT_SNAPSHOT_STRUCTURAL_FIELD_INFO[name] for name in sorted(_CONTEXT_SNAPSHOT_STRUCTURAL_FIELDS)]
    return tuple(sorted((*infos, *_SCOPED_SESSION_STRUCTURAL_FIELD_INFO), key=lambda field: field.name))


STRUCTURAL_QUERY_UNIT_REGISTRY: dict[str, StructuralQueryUnitInfo] = {
    "action": StructuralQueryUnitInfo(
        description="Match sessions with at least one action row satisfying the child predicate.",
        fields=_structural_field_infos(*_ACTION_STRUCTURAL_FIELDS),
        example="exists action(session.repo:polylogue AND tool:bash AND text:pytest)",
    ),
    "block": StructuralQueryUnitInfo(
        description="Match sessions with at least one parsed message block satisfying the child predicate.",
        fields=_structural_field_infos(*_BLOCK_STRUCTURAL_FIELDS),
        example="exists block(session.origin:codex-session AND type:code AND text:timeout)",
    ),
    "message": StructuralQueryUnitInfo(
        description="Match sessions with at least one message satisfying the child predicate.",
        fields=_structural_field_infos(*_MESSAGE_STRUCTURAL_FIELDS),
        example="exists message(session.repo:polylogue AND role:assistant AND text:timeout)",
    ),
    "assertion": StructuralQueryUnitInfo(
        description="Match sessions with at least one session-targeted assertion satisfying the child predicate.",
        fields=_assertion_field_infos(),
        example="exists assertion(kind:decision AND status:active AND text:review)",
    ),
    "observed-event": StructuralQueryUnitInfo(
        description="Return runtime-transform observed events satisfying the child predicate.",
        fields=_observed_event_field_infos(),
        example="observed-events where session.repo:polylogue AND delivery_state:acted_on",
    ),
    "context-snapshot": StructuralQueryUnitInfo(
        description="Return runtime-transform context snapshots satisfying the child predicate.",
        fields=_context_snapshot_field_infos(),
        example="context-snapshots where session.repo:polylogue AND boundary:session_start",
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
    return tuple(field.name for field in info.fields)


def structural_query_field_info(unit: str, field: str) -> StructuralQueryFieldInfo | None:
    """Return metadata for a structural field accepted by *unit*."""

    info = STRUCTURAL_QUERY_UNIT_REGISTRY.get(unit.lower())
    if info is None:
        return None
    normalized = field.lower()
    for field_info in info.fields:
        if field_info.name == normalized:
            return field_info
    return None


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
    "StructuralQueryFieldInfo",
    "StructuralQueryUnitInfo",
    "_BOOLEAN_SUPPORTED_FIELDS",
    "_ACTION_STRUCTURAL_FIELDS",
    "_ASSERTION_STRUCTURAL_FIELDS",
    "_BLOCK_STRUCTURAL_FIELDS",
    "_CONTEXT_SNAPSHOT_STRUCTURAL_FIELDS",
    "_MESSAGE_STRUCTURAL_FIELDS",
    "_OBSERVED_EVENT_STRUCTURAL_FIELDS",
    "_SOURCE_WHERE_SOURCES",
    "_STRUCTURAL_BOOLEAN_SUPPORTED_FIELDS",
    "count_query_fields",
    "count_query_operators",
    "date_query_fields",
    "date_query_operators",
    "structural_query_field_info",
    "structural_query_fields",
    "structural_query_units",
]
