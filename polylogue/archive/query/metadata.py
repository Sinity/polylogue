"""Lightweight query grammar metadata for completion and docs consumers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

QueryUnitName = Literal["message", "action", "block", "assertion", "run", "observed-event", "context-snapshot"]
QueryUnitLowererKind = Literal["sql", "runtime_transform"]


@dataclass(frozen=True)
class QueryUnitDescriptor:
    """Executable query-unit metadata shared by parser, completions, and surfaces."""

    unit: QueryUnitName
    singular_source: str
    plural_source: str
    payload_model: str
    terminal_supported: bool = True
    exists_supported: bool = False
    lowerer_kind: QueryUnitLowererKind = "sql"
    sql_query_method: str | None = None
    runtime_query_method: str | None = None
    cli_plain_renderer: str | None = None
    aggregate_group_fields: tuple[str, ...] = ()
    fields: tuple[StructuralQueryFieldInfo, ...] = ()
    description: str = ""
    example: str = ""

    @property
    def source_aliases(self) -> tuple[str, str]:
        return (self.singular_source, self.plural_source)


@dataclass(frozen=True)
class QueryPipelineStageInfo:
    """Completion/query-builder metadata for executable terminal pipeline stages."""

    value: str
    insert: str
    description: str
    source_unit: QueryUnitName
    lowerer_kind: QueryUnitLowererKind


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
    "duration_ms": {
        "description": "Session reported-duration comparison in milliseconds",
        "spec_field": "boolean_predicate",
        "negatable": "no",
        "example": "sessions where duration_ms >= 60000",
    },
    "user_messages": {
        "description": "Count comparison for user messages",
        "spec_field": "boolean_predicate",
        "negatable": "no",
        "example": "sessions where user_messages >= 2",
    },
    "assistant_messages": {
        "description": "Count comparison for assistant messages",
        "spec_field": "boolean_predicate",
        "negatable": "no",
        "example": "sessions where assistant_messages >= 2",
    },
    "system_messages": {
        "description": "Count comparison for system messages",
        "spec_field": "boolean_predicate",
        "negatable": "no",
        "example": "sessions where system_messages = 0",
    },
    "tool_messages": {
        "description": "Count comparison for tool messages",
        "spec_field": "boolean_predicate",
        "negatable": "no",
        "example": "sessions where tool_messages = 0",
    },
    "tool_use_messages": {
        "description": "Count comparison for messages carrying tool-use or tool-result evidence",
        "spec_field": "boolean_predicate",
        "negatable": "no",
        "example": "sessions where tool_use_messages >= 1",
    },
    "thinking_messages": {
        "description": "Count comparison for messages carrying thinking/reasoning evidence",
        "spec_field": "boolean_predicate",
        "negatable": "no",
        "example": "sessions where thinking_messages >= 1",
    },
    "paste_messages": {
        "description": "Count comparison for messages carrying paste evidence",
        "spec_field": "boolean_predicate",
        "negatable": "no",
        "example": "sessions where paste_messages = 0",
    },
    "user_words": {
        "description": "Count comparison for user words",
        "spec_field": "boolean_predicate",
        "negatable": "no",
        "example": "sessions where user_words >= 100",
    },
    "assistant_words": {
        "description": "Count comparison for assistant words",
        "spec_field": "boolean_predicate",
        "negatable": "no",
        "example": "sessions where assistant_words >= 500",
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
    "duration_ms",
    "user_messages",
    "assistant_messages",
    "system_messages",
    "tool_messages",
    "tool_use_messages",
    "thinking_messages",
    "paste_messages",
    "user_words",
    "assistant_words",
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
    "input_tokens",
    "output_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
    "duration_ms",
    "time",
}
_MESSAGE_STRUCTURAL_FIELDS = {
    "role",
    "type",
    "text",
    "tool",
    "action",
    "command",
    "path",
    "output",
    "words",
    "input_tokens",
    "output_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
    "duration_ms",
    "time",
}
_ACTION_STRUCTURAL_FIELDS = {"tool", "action", "type", "command", "path", "output", "text", "time"}
_BLOCK_STRUCTURAL_FIELDS = {"type", "text", "tool", "action", "command", "path", "time"}
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
    "time",
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
_RUN_STRUCTURAL_FIELDS = {
    "agent",
    "agent_ref",
    "branch",
    "context_snapshot",
    "context_snapshot_ref",
    "confidence",
    "cwd",
    "evidence",
    "git_branch",
    "harness",
    "lineage",
    "lineage_ref",
    "native_parent_session_id",
    "native_session_id",
    "origin",
    "parent",
    "parent_run_ref",
    "provider_origin",
    "role",
    "run",
    "run_ref",
    "status",
    "text",
    "title",
    "transcript",
    "transcript_ref",
}
_STRUCTURAL_BOOLEAN_SUPPORTED_FIELDS.update(_ASSERTION_STRUCTURAL_FIELDS)
_STRUCTURAL_BOOLEAN_SUPPORTED_FIELDS.update(_OBSERVED_EVENT_STRUCTURAL_FIELDS)
_STRUCTURAL_BOOLEAN_SUPPORTED_FIELDS.update(_CONTEXT_SNAPSHOT_STRUCTURAL_FIELDS)
_STRUCTURAL_BOOLEAN_SUPPORTED_FIELDS.update(_RUN_STRUCTURAL_FIELDS)


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
    session_column: str


@dataclass(frozen=True)
class NumericQueryFieldInfo:
    """Completion/query-builder metadata for non-count numeric predicates."""

    description: str
    operators: tuple[str, ...]
    range_keyword: str
    example: str
    unit_columns: dict[str, str]


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
    "time": _field_info("time", "Row timestamp predicate for the selected unit.", "time >= 2026-01-01"),
    "tool": _field_info("tool", "Normalized tool name.", "tool:bash"),
    "type": _field_info("type", "Unit-specific type token.", "type:code"),
    "words": _field_info("words", "Word-count predicate for the selected unit.", "words:>=200"),
    "input_tokens": _field_info("input_tokens", "Message input-token count predicate.", "input_tokens >= 1000"),
    "output_tokens": _field_info("output_tokens", "Message output-token count predicate.", "output_tokens >= 1000"),
    "cache_read_tokens": _field_info(
        "cache_read_tokens",
        "Message cache-read token count predicate.",
        "cache_read_tokens >= 1000",
    ),
    "cache_write_tokens": _field_info(
        "cache_write_tokens",
        "Message cache-write token count predicate.",
        "cache_write_tokens >= 1000",
    ),
    "duration_ms": _field_info("duration_ms", "Duration predicate in milliseconds.", "duration_ms >= 1000"),
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
    "time": _field_info("time", "Assertion update/create timestamp predicate.", "time >= 2026-01-01"),
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

_RUN_STRUCTURAL_FIELD_INFO: dict[str, StructuralQueryFieldInfo] = {
    "agent": _field_info("agent", "Run agent ObjectRef substring.", "agent:agent:codex"),
    "agent_ref": _field_info("agent_ref", "Run agent ObjectRef substring.", "agent_ref:agent:codex"),
    "branch": _field_info("branch", "Run git branch substring.", "branch:feature"),
    "context_snapshot": _field_info(
        "context_snapshot",
        "Run context snapshot ObjectRef substring.",
        "context_snapshot:context-snapshot:",
    ),
    "context_snapshot_ref": _field_info(
        "context_snapshot_ref",
        "Run context snapshot ObjectRef substring.",
        "context_snapshot_ref:context-snapshot:",
    ),
    "confidence": _field_info("confidence", "Run confidence source, such as raw or inferred.", "confidence:raw"),
    "cwd": _field_info("cwd", "Run working directory substring.", "cwd:/realm/project"),
    "evidence": _field_info("evidence", "Run evidence ref substring.", "evidence:session:"),
    "git_branch": _field_info("git_branch", "Run git branch substring.", "git_branch:feature"),
    "harness": _field_info("harness", "Run harness/runtime family.", "harness:codex"),
    "lineage": _field_info("lineage", "Run lineage ObjectRef substring.", "lineage:run:"),
    "lineage_ref": _field_info("lineage_ref", "Run lineage ObjectRef substring.", "lineage_ref:run:"),
    "native_parent_session_id": _field_info(
        "native_parent_session_id",
        "Provider-native parent session id substring.",
        "native_parent_session_id:abc",
    ),
    "native_session_id": _field_info(
        "native_session_id",
        "Provider-native session id substring.",
        "native_session_id:abc",
    ),
    "origin": _field_info("origin", "Run provider-origin token.", "origin:codex-session"),
    "parent": _field_info("parent", "Parent run ObjectRef substring.", "parent:run:"),
    "parent_run_ref": _field_info("parent_run_ref", "Parent run ObjectRef substring.", "parent_run_ref:run:"),
    "provider_origin": _field_info("provider_origin", "Run provider-origin token.", "provider_origin:codex-session"),
    "role": _field_info("role", "Run role, such as main or subagent.", "role:main"),
    "run": _field_info("run", "Run ObjectRef substring.", "run:run:"),
    "run_ref": _field_info("run_ref", "Run ObjectRef substring.", "run_ref:run:"),
    "status": _field_info("status", "Run status token.", "status:completed"),
    "text": _field_info("text", "Run title/ref/agent/lineage/evidence substring.", "text:review"),
    "title": _field_info("title", "Run title substring.", "title:review"),
    "transcript": _field_info("transcript", "Run transcript evidence ref substring.", "transcript:session:"),
    "transcript_ref": _field_info(
        "transcript_ref", "Run transcript evidence ref substring.", "transcript_ref:session:"
    ),
}

_SESSION_SCOPED_STRUCTURAL_EXAMPLES: dict[str, str] = {
    "action": "session.action:file_edit",
    "cwd": "session.cwd:/realm/project",
    "duration_ms": "session.duration_ms:>=60000",
    "has": "session.has:tools",
    "id": "session.id:abc123",
    "messages": "session.messages:>=10",
    "origin": "session.origin:claude-code-session",
    "repo": "session.repo:polylogue",
    "since": "session.since:7d",
    "tag": "session.tag:review",
    "title": "session.title:refactor",
    "tool": "session.tool:bash",
    "until": "session.until:2024-01-15",
    "words": "session.words:<=200",
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


def _run_field_infos() -> tuple[StructuralQueryFieldInfo, ...]:
    infos = [_RUN_STRUCTURAL_FIELD_INFO[name] for name in sorted(_RUN_STRUCTURAL_FIELDS)]
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
    "run": StructuralQueryUnitInfo(
        description="Return runtime-transform runs satisfying the child predicate.",
        fields=_run_field_infos(),
        example="runs where session.repo:polylogue AND role:main",
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


def _unit_info(unit: QueryUnitName) -> StructuralQueryUnitInfo:
    return STRUCTURAL_QUERY_UNIT_REGISTRY[unit]


QUERY_UNIT_DESCRIPTORS: tuple[QueryUnitDescriptor, ...] = (
    QueryUnitDescriptor(
        "message",
        "message",
        "messages",
        exists_supported=True,
        payload_model="MessageQueryRowPayload",
        sql_query_method="query_messages",
        cli_plain_renderer="message",
        aggregate_group_fields=("role", "type", "session.origin", "session.repo"),
        fields=_unit_info("message").fields,
        description=_unit_info("message").description,
        example=_unit_info("message").example,
    ),
    QueryUnitDescriptor(
        "action",
        "action",
        "actions",
        exists_supported=True,
        payload_model="ActionQueryRowPayload",
        sql_query_method="query_actions",
        cli_plain_renderer="action",
        aggregate_group_fields=("tool", "action", "type", "session.origin", "session.repo"),
        fields=_unit_info("action").fields,
        description=_unit_info("action").description,
        example=_unit_info("action").example,
    ),
    QueryUnitDescriptor(
        "block",
        "block",
        "blocks",
        exists_supported=True,
        payload_model="BlockQueryRowPayload",
        sql_query_method="query_blocks",
        cli_plain_renderer="block",
        aggregate_group_fields=("type", "tool", "action", "session.origin", "session.repo"),
        fields=_unit_info("block").fields,
        description=_unit_info("block").description,
        example=_unit_info("block").example,
    ),
    QueryUnitDescriptor(
        "assertion",
        "assertion",
        "assertions",
        exists_supported=True,
        payload_model="AssertionQueryRowPayload",
        sql_query_method="query_assertions",
        cli_plain_renderer="assertion",
        aggregate_group_fields=("kind", "status", "visibility", "author_kind", "session.origin", "session.repo"),
        fields=_unit_info("assertion").fields,
        description=_unit_info("assertion").description,
        example=_unit_info("assertion").example,
    ),
    QueryUnitDescriptor(
        "run",
        "run",
        "runs",
        exists_supported=False,
        lowerer_kind="runtime_transform",
        payload_model="RunQueryRowPayload",
        runtime_query_method="_query_runs",
        cli_plain_renderer="run",
        fields=_unit_info("run").fields,
        description=_unit_info("run").description,
        example=_unit_info("run").example,
    ),
    QueryUnitDescriptor(
        "observed-event",
        "observed-event",
        "observed-events",
        exists_supported=False,
        lowerer_kind="runtime_transform",
        payload_model="ObservedEventQueryRowPayload",
        runtime_query_method="_query_observed_events",
        cli_plain_renderer="observed-event",
        fields=_unit_info("observed-event").fields,
        description=_unit_info("observed-event").description,
        example=_unit_info("observed-event").example,
    ),
    QueryUnitDescriptor(
        "context-snapshot",
        "context-snapshot",
        "context-snapshots",
        exists_supported=False,
        lowerer_kind="runtime_transform",
        payload_model="ContextSnapshotQueryRowPayload",
        runtime_query_method="_query_context_snapshots",
        cli_plain_renderer="context-snapshot",
        fields=_unit_info("context-snapshot").fields,
        description=_unit_info("context-snapshot").description,
        example=_unit_info("context-snapshot").example,
    ),
)
_QUERY_UNIT_BY_UNIT: dict[QueryUnitName, QueryUnitDescriptor] = {
    descriptor.unit: descriptor for descriptor in QUERY_UNIT_DESCRIPTORS
}


def query_unit_descriptors(
    *,
    terminal_supported: bool | None = None,
    exists_supported: bool | None = None,
    lowerer_kind: QueryUnitLowererKind | None = None,
) -> tuple[QueryUnitDescriptor, ...]:
    """Return query-unit descriptors matching the requested capabilities."""

    descriptors = QUERY_UNIT_DESCRIPTORS
    if terminal_supported is not None:
        descriptors = tuple(
            descriptor for descriptor in descriptors if descriptor.terminal_supported is terminal_supported
        )
    if exists_supported is not None:
        descriptors = tuple(descriptor for descriptor in descriptors if descriptor.exists_supported is exists_supported)
    if lowerer_kind is not None:
        descriptors = tuple(descriptor for descriptor in descriptors if descriptor.lowerer_kind == lowerer_kind)
    return descriptors


_SOURCE_WHERE_SOURCES: tuple[tuple[str, QueryUnitName], ...] = tuple(
    (source, descriptor.unit)
    for descriptor in query_unit_descriptors(terminal_supported=True)
    for source in descriptor.source_aliases
)

COUNT_QUERY_FIELD_REGISTRY: dict[str, CountQueryFieldInfo] = {
    "messages": CountQueryFieldInfo(
        description="Message-count predicate over normalized session messages.",
        operators=(">=", "<=", "=", ">", "<"),
        range_keyword="between",
        example="messages between 5 and 20",
        session_column="message_count",
    ),
    "words": CountQueryFieldInfo(
        description="Word-count predicate over normalized session text.",
        operators=(">=", "<=", "=", ">", "<"),
        range_keyword="between",
        example="words between 100 and 500",
        session_column="word_count",
    ),
    "user_messages": CountQueryFieldInfo(
        description="User-message-count predicate over normalized session messages.",
        operators=(">=", "<=", "=", ">", "<"),
        range_keyword="between",
        example="user_messages >= 2",
        session_column="user_message_count",
    ),
    "assistant_messages": CountQueryFieldInfo(
        description="Assistant-message-count predicate over normalized session messages.",
        operators=(">=", "<=", "=", ">", "<"),
        range_keyword="between",
        example="assistant_messages >= 2",
        session_column="assistant_message_count",
    ),
    "system_messages": CountQueryFieldInfo(
        description="System-message-count predicate over normalized session messages.",
        operators=(">=", "<=", "=", ">", "<"),
        range_keyword="between",
        example="system_messages = 0",
        session_column="system_message_count",
    ),
    "tool_messages": CountQueryFieldInfo(
        description="Tool-message-count predicate over normalized session messages.",
        operators=(">=", "<=", "=", ">", "<"),
        range_keyword="between",
        example="tool_messages = 0",
        session_column="tool_message_count",
    ),
    "tool_use_messages": CountQueryFieldInfo(
        description="Predicate over messages carrying tool-use or tool-result evidence.",
        operators=(">=", "<=", "=", ">", "<"),
        range_keyword="between",
        example="tool_use_messages >= 1",
        session_column="tool_use_count",
    ),
    "thinking_messages": CountQueryFieldInfo(
        description="Predicate over messages carrying thinking/reasoning evidence.",
        operators=(">=", "<=", "=", ">", "<"),
        range_keyword="between",
        example="thinking_messages >= 1",
        session_column="thinking_count",
    ),
    "paste_messages": CountQueryFieldInfo(
        description="Predicate over messages carrying paste evidence.",
        operators=(">=", "<=", "=", ">", "<"),
        range_keyword="between",
        example="paste_messages = 0",
        session_column="paste_count",
    ),
    "user_words": CountQueryFieldInfo(
        description="User-word-count predicate over normalized session text.",
        operators=(">=", "<=", "=", ">", "<"),
        range_keyword="between",
        example="user_words between 100 and 500",
        session_column="user_word_count",
    ),
    "assistant_words": CountQueryFieldInfo(
        description="Assistant-word-count predicate over normalized session text.",
        operators=(">=", "<=", "=", ">", "<"),
        range_keyword="between",
        example="assistant_words >= 500",
        session_column="assistant_word_count",
    ),
}

NUMERIC_QUERY_FIELD_REGISTRY: dict[str, NumericQueryFieldInfo] = {
    "duration_ms": NumericQueryFieldInfo(
        description="Duration predicate in milliseconds.",
        operators=(">=", "<=", "=", ">", "<"),
        range_keyword="between",
        example="duration_ms >= 60000",
        unit_columns={
            "session": "reported_duration_ms",
            "message": "duration_ms",
        },
    ),
    "input_tokens": NumericQueryFieldInfo(
        description="Message input-token predicate.",
        operators=(">=", "<=", "=", ">", "<"),
        range_keyword="between",
        example="input_tokens >= 1000",
        unit_columns={"message": "input_tokens"},
    ),
    "output_tokens": NumericQueryFieldInfo(
        description="Message output-token predicate.",
        operators=(">=", "<=", "=", ">", "<"),
        range_keyword="between",
        example="output_tokens >= 1000",
        unit_columns={"message": "output_tokens"},
    ),
    "cache_read_tokens": NumericQueryFieldInfo(
        description="Message cache-read token predicate.",
        operators=(">=", "<=", "=", ">", "<"),
        range_keyword="between",
        example="cache_read_tokens >= 1000",
        unit_columns={"message": "cache_read_tokens"},
    ),
    "cache_write_tokens": NumericQueryFieldInfo(
        description="Message cache-write token predicate.",
        operators=(">=", "<=", "=", ">", "<"),
        range_keyword="between",
        example="cache_write_tokens >= 1000",
        unit_columns={"message": "cache_write_tokens"},
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

    return tuple(sorted(descriptor.unit for descriptor in query_unit_descriptors(exists_supported=True)))


def structural_query_fields(unit: str) -> tuple[str, ...]:
    """Return structural field names accepted inside ``exists <unit>(...)``."""

    descriptor = query_unit_descriptor(unit)
    if descriptor is None or not descriptor.exists_supported:
        return ()
    return tuple(field.name for field in descriptor.fields)


def structural_query_field_info(unit: str, field: str) -> StructuralQueryFieldInfo | None:
    """Return metadata for a structural field accepted by *unit*."""

    descriptor = query_unit_descriptor(unit)
    if descriptor is None or not descriptor.exists_supported:
        return None
    normalized = field.lower()
    for field_info in descriptor.fields:
        if field_info.name == normalized:
            return field_info
    return None


def terminal_query_sources() -> tuple[str, ...]:
    """Return source tokens accepted by the ``<source> where ...`` grammar."""

    return tuple(sorted(source for source, _unit in terminal_query_source_pairs()))


def terminal_query_source_pairs() -> tuple[tuple[str, QueryUnitName], ...]:
    """Return accepted ``<source> where`` aliases and their canonical units."""

    return _SOURCE_WHERE_SOURCES


def terminal_query_unit(source: str) -> QueryUnitName | None:
    """Return the canonical query unit for a terminal source token."""

    normalized = source.lower()
    for source_name, unit in _SOURCE_WHERE_SOURCES:
        if source_name == normalized:
            return unit
    return None


def query_unit_descriptor(unit_or_source: str) -> QueryUnitDescriptor | None:
    """Return the descriptor for a canonical unit or accepted source alias."""

    normalized = unit_or_source.lower()
    for descriptor in QUERY_UNIT_DESCRIPTORS:
        if descriptor.unit == normalized or normalized in descriptor.source_aliases:
            return descriptor
    return None


def terminal_query_source_list(*, plural: bool = True, separator: str = "/") -> str:
    """Return a user-facing terminal query-unit source list."""

    labels = [
        descriptor.plural_source if plural else descriptor.singular_source
        for descriptor in query_unit_descriptors(terminal_supported=True)
    ]
    return separator.join(labels)


def terminal_query_fields(source: str) -> tuple[str, ...]:
    """Return field names accepted after ``<source> where``."""

    descriptor = query_unit_descriptor(source)
    if descriptor is None or not descriptor.terminal_supported:
        return ()
    return tuple(field.name for field in descriptor.fields)


def terminal_query_field_info(source: str, field: str) -> StructuralQueryFieldInfo | None:
    """Return metadata for a field accepted after ``<source> where``."""

    descriptor = query_unit_descriptor(source)
    if descriptor is None or not descriptor.terminal_supported:
        return None
    normalized = field.lower()
    for field_info in descriptor.fields:
        if field_info.name == normalized:
            return field_info
    return None


def terminal_query_pipeline_stage_infos(source: str) -> tuple[QueryPipelineStageInfo, ...]:
    """Return executable pipeline-stage candidates for a terminal query source."""

    descriptor = query_unit_descriptor(source)
    if descriptor is None or not descriptor.terminal_supported:
        return ()
    stages: list[QueryPipelineStageInfo] = []
    if descriptor.lowerer_kind == "sql":
        stages.append(
            QueryPipelineStageInfo(
                value="sort by time",
                insert="sort by time desc",
                description="Order SQL-backed terminal rows by row timestamp before pagination.",
                source_unit=descriptor.unit,
                lowerer_kind=descriptor.lowerer_kind,
            )
        )
    for field_name in descriptor.aggregate_group_fields:
        stages.append(
            QueryPipelineStageInfo(
                value=f"group by {field_name}",
                insert=f"group by {field_name}",
                description=f"Group {descriptor.plural_source} by {field_name} before an aggregate count stage.",
                source_unit=descriptor.unit,
                lowerer_kind=descriptor.lowerer_kind,
            )
        )
    if descriptor.aggregate_group_fields:
        stages.append(
            QueryPipelineStageInfo(
                value="count",
                insert="count",
                description="Count rows in each preceding group.",
                source_unit=descriptor.unit,
                lowerer_kind=descriptor.lowerer_kind,
            )
        )
        stages.extend(
            (
                QueryPipelineStageInfo(
                    value="sort by count",
                    insert="sort by count desc",
                    description="Order aggregate groups by count after a count stage.",
                    source_unit=descriptor.unit,
                    lowerer_kind=descriptor.lowerer_kind,
                ),
                QueryPipelineStageInfo(
                    value="sort by key",
                    insert="sort by key asc",
                    description="Order aggregate groups by group key after a count stage.",
                    source_unit=descriptor.unit,
                    lowerer_kind=descriptor.lowerer_kind,
                ),
            )
        )
    stages.extend(
        (
            QueryPipelineStageInfo(
                value="limit",
                insert="limit ",
                description="Limit terminal rows after filtering or aggregation.",
                source_unit=descriptor.unit,
                lowerer_kind=descriptor.lowerer_kind,
            ),
            QueryPipelineStageInfo(
                value="offset",
                insert="offset ",
                description="Skip terminal rows after filtering or aggregation.",
                source_unit=descriptor.unit,
                lowerer_kind=descriptor.lowerer_kind,
            ),
        )
    )
    return tuple(stages)


def count_query_fields() -> tuple[str, ...]:
    """Return count fields with readable comparison/range syntax."""

    return tuple(sorted(COUNT_QUERY_FIELD_REGISTRY))


def count_query_operators(field: str) -> tuple[str, ...]:
    """Return readable comparison/range operators accepted for a count field."""

    info = COUNT_QUERY_FIELD_REGISTRY.get(field.lower())
    if info is None:
        return ()
    return (*info.operators, info.range_keyword)


def numeric_query_fields() -> tuple[str, ...]:
    """Return non-count numeric fields with readable comparison/range syntax."""

    return tuple(sorted(NUMERIC_QUERY_FIELD_REGISTRY))


def numeric_query_operators(field: str) -> tuple[str, ...]:
    """Return readable comparison/range operators accepted for a numeric field."""

    info = NUMERIC_QUERY_FIELD_REGISTRY.get(field.lower())
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
    "NUMERIC_QUERY_FIELD_REGISTRY",
    "NumericQueryFieldInfo",
    "QUERY_UNIT_DESCRIPTORS",
    "QueryUnitDescriptor",
    "QueryUnitLowererKind",
    "QueryUnitName",
    "QueryPipelineStageInfo",
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
    "_RUN_STRUCTURAL_FIELDS",
    "_SOURCE_WHERE_SOURCES",
    "_STRUCTURAL_BOOLEAN_SUPPORTED_FIELDS",
    "count_query_fields",
    "count_query_operators",
    "date_query_fields",
    "date_query_operators",
    "numeric_query_fields",
    "numeric_query_operators",
    "query_unit_descriptor",
    "query_unit_descriptors",
    "structural_query_field_info",
    "structural_query_fields",
    "structural_query_units",
    "terminal_query_field_info",
    "terminal_query_fields",
    "terminal_query_pipeline_stage_infos",
    "terminal_query_source_list",
    "terminal_query_source_pairs",
    "terminal_query_sources",
    "terminal_query_unit",
]
