"""Independent known-answer laws for the session query-field lowerer.

The values in this module are hand-authored semantic facts.  They deliberately
do not consult the parser, the field registry, or a production lowerer when
building expectations: a field mapped to its neighbour must make a row fail.
The old one-example-per-field tests retired by the consolidation are recorded
at the bottom so the deletion remains reviewable and every deletion names the
law that replaces it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class FieldProjection:
    """One expected SessionQuerySpec projection, written independently."""

    name: str
    value: object
    contains: bool = False


@dataclass(frozen=True, slots=True)
class PredicateFact:
    """A primitive expected field predicate, not a production predicate object."""

    field: str
    values: tuple[str, ...]
    op: Literal["=", ">", ">=", "<", "<="] = "="


@dataclass(frozen=True, slots=True)
class BooleanFact:
    operator: Literal["and", "or"]
    children: tuple[object, ...]


@dataclass(frozen=True, slots=True)
class NotFact:
    child: object


@dataclass(frozen=True, slots=True)
class LineageFact:
    seed_session_id: str


@dataclass(frozen=True, slots=True)
class QueryFieldLaw:
    """One expression, its semantic answer, and optional equivalent forms."""

    law_id: str
    field: str
    expression: str
    expected: tuple[FieldProjection, ...]
    negated_expression: str | None = None
    negated_expected: tuple[FieldProjection, ...] = ()
    structured_params: tuple[tuple[str, object], ...] = ()


def _p(name: str, value: object, *, contains: bool = False) -> FieldProjection:
    return FieldProjection(name, value, contains)


def _predicate(field: str, value: str, op: Literal["=", ">", ">=", "<", "<="] = "=") -> PredicateFact:
    return PredicateFact(field=field, values=(value,), op=op)


# This is intentionally not derived from EXPRESSION_FIELD_REGISTRY.  In
# particular, the expected destination names and values are the independent
# semantic witness for each field.
QUERY_FIELD_LAWS: tuple[QueryFieldLaw, ...] = (
    QueryFieldLaw(
        "field-repo",
        "repo",
        "repo:polylogue",
        (_p("repo_names", ("polylogue",)),),
        structured_params=(("repo", "polylogue"),),
    ),
    QueryFieldLaw("field-project", "project", "project:g-p-6a40343a", (_p("project_refs", ("g-p-6a40343a",)),)),
    QueryFieldLaw(
        "field-origin",
        "origin",
        "origin:claude-code-session",
        (_p("origins", ("claude-code-session",)),),
        "-origin:chatgpt-export",
        (_p("excluded_origins", ("chatgpt-export",)),),
        (("origin", "claude-code-session"),),
    ),
    QueryFieldLaw(
        "field-origin-vocabulary",
        "origin",
        "origin:(claude-code-session|codex-session)",
        (_p("origins", ("claude-code-session", "codex-session")),),
    ),
    QueryFieldLaw(
        "field-tag",
        "tag",
        "tag:review",
        (_p("tags", ("review",)),),
        "-tag:wip",
        (_p("excluded_tags", ("wip",)),),
        (("tag", "review"),),
    ),
    QueryFieldLaw("field-path", "path", "path:polylogue/cli", (_p("referenced_path", ("polylogue/cli",)),)),
    QueryFieldLaw("field-cwd", "cwd", "cwd:/realm/project", (_p("cwd_prefix", "/realm/project"),)),
    QueryFieldLaw(
        "field-tool",
        "tool",
        "tool:bash",
        (_p("tool_terms", ("bash",)),),
        "-tool:bash",
        (_p("excluded_tool_terms", ("bash",)),),
    ),
    QueryFieldLaw(
        "field-action",
        "action",
        "action:file_edit",
        (_p("action_terms", ("file_edit",)),),
        "-action:shell",
        (_p("excluded_action_terms", ("shell",)),),
    ),
    QueryFieldLaw(
        "field-action-sequence",
        "action_sequence",
        "action_sequence:file_edit>shell",
        (_p("action_sequence", ("file_edit", "shell")),),
    ),
    QueryFieldLaw("field-action-text", "action_text", "action_text:pytest", (_p("action_text_terms", ("pytest",)),)),
    QueryFieldLaw(
        "field-since-session",
        "since_session",
        "since_session:claude-code-session:abc123",
        (_p("since_session_id", "claude-code-session:abc123"),),
    ),
    QueryFieldLaw(
        "field-has-paste",
        "has",
        "has:paste",
        (
            _p("filter_has_paste", True),
            _p("filter_has_tool_use", False),
            _p("filter_has_thinking", False),
        ),
        structured_params=(("filter_has_paste", True),),
    ),
    QueryFieldLaw(
        "field-has-tools",
        "has",
        "has:tools",
        (_p("filter_has_tool_use", True), _p("filter_has_paste", False), _p("filter_has_thinking", False)),
        structured_params=(("filter_has_tool_use", True),),
    ),
    QueryFieldLaw(
        "field-has-thinking",
        "has",
        "has:thinking",
        (_p("filter_has_thinking", True), _p("filter_has_paste", False), _p("filter_has_tool_use", False)),
        structured_params=(("filter_has_thinking", True),),
    ),
    QueryFieldLaw("field-has-type", "has", "has:summary", (_p("has_types", ("summary",)),)),
    QueryFieldLaw("field-id", "id", "id:abc123", (_p("session_id", "abc123"),)),
    QueryFieldLaw(
        "field-session-alias",
        "session",
        "session:claude-code-session:abc123",
        (_p("session_id", "claude-code-session:abc123"),),
    ),
    QueryFieldLaw("field-title", "title", "title:refactor", (_p("title", "refactor"),)),
    QueryFieldLaw("field-root", "root", "root:true", (_p("root", True),)),
    QueryFieldLaw("field-since", "since", "since:7d", (_p("since", "days", contains=True),)),
    QueryFieldLaw("field-until", "until", "until:2024-01-15", (_p("until", "2024-01-15"),)),
    QueryFieldLaw("field-near-text", "near", 'near:"semantic search"', (_p("similar_text", "semantic search"),)),
    QueryFieldLaw("field-contains", "contains", "contains:foo", (_p("contains_terms", ("foo",)),)),
    QueryFieldLaw(
        "field-messages",
        "messages",
        "messages:>=10",
        (_p("min_messages", 10),),
        structured_params=(("min_messages", 10),),
    ),
    QueryFieldLaw("field-words", "words", "words:>=200", (_p("min_words", 200),)),
    QueryFieldLaw(
        "field-duration",
        "duration_ms",
        "sessions where duration_ms >= 60000",
        (_p("boolean_predicate", _predicate("duration_ms", "60000", ">=")),),
    ),
    QueryFieldLaw(
        "field-user-messages",
        "user_messages",
        "sessions where user_messages >= 2",
        (_p("boolean_predicate", _predicate("user_messages", "2", ">=")),),
    ),
    QueryFieldLaw(
        "field-authored-user-messages",
        "authored_user_messages",
        "sessions where authored_user_messages >= 2",
        (_p("boolean_predicate", _predicate("authored_user_messages", "2", ">=")),),
    ),
    QueryFieldLaw(
        "field-assistant-messages",
        "assistant_messages",
        "sessions where assistant_messages >= 2",
        (_p("boolean_predicate", _predicate("assistant_messages", "2", ">=")),),
    ),
    QueryFieldLaw(
        "field-system-messages",
        "system_messages",
        "sessions where system_messages = 0",
        (_p("boolean_predicate", _predicate("system_messages", "0")),),
    ),
    QueryFieldLaw(
        "field-tool-messages",
        "tool_messages",
        "sessions where tool_messages = 0",
        (_p("boolean_predicate", _predicate("tool_messages", "0")),),
    ),
    QueryFieldLaw(
        "field-tool-use-messages",
        "tool_use_messages",
        "sessions where tool_use_messages >= 1",
        (_p("boolean_predicate", _predicate("tool_use_messages", "1", ">=")),),
    ),
    QueryFieldLaw(
        "field-thinking-messages",
        "thinking_messages",
        "sessions where thinking_messages >= 1",
        (_p("boolean_predicate", _predicate("thinking_messages", "1", ">=")),),
    ),
    QueryFieldLaw(
        "field-paste-messages",
        "paste_messages",
        "sessions where paste_messages = 0",
        (_p("boolean_predicate", _predicate("paste_messages", "0")),),
    ),
    QueryFieldLaw(
        "field-user-words",
        "user_words",
        "sessions where user_words >= 100",
        (_p("boolean_predicate", _predicate("user_words", "100", ">=")),),
    ),
    QueryFieldLaw(
        "field-authored-user-words",
        "authored_user_words",
        "sessions where authored_user_words >= 100",
        (_p("boolean_predicate", _predicate("authored_user_words", "100", ">=")),),
    ),
    QueryFieldLaw(
        "field-assistant-words",
        "assistant_words",
        "sessions where assistant_words >= 500",
        (_p("boolean_predicate", _predicate("assistant_words", "500", ">=")),),
    ),
    QueryFieldLaw("field-lane", "lane", "lane:dialogue", (_p("retrieval_lane", "dialogue"),)),
    QueryFieldLaw(
        "field-lineage",
        "lineage",
        "lineage:id:chatgpt-export:ext-root",
        (_p("boolean_predicate", LineageFact("chatgpt-export:ext-root")),),
    ),
)


@dataclass(frozen=True, slots=True)
class BoundaryLaw:
    law_id: str
    expression: str
    expected: tuple[FieldProjection, ...]


# Strict and inclusive operators intentionally share neighboring values.  A
# mutant changing ``>`` to ``>=`` (or ``<`` to ``<=``) must fail these rows.
QUERY_BOUNDARY_LAWS: tuple[BoundaryLaw, ...] = (
    BoundaryLaw("count-inclusive-lower", "messages >= 10", (_p("min_messages", 10),)),
    BoundaryLaw("count-exclusive-lower", "messages > 10", (_p("min_messages", 11),)),
    BoundaryLaw("count-inclusive-upper", "messages <= 10", (_p("max_messages", 10),)),
    BoundaryLaw("count-exclusive-upper", "messages < 10", (_p("max_messages", 9),)),
    BoundaryLaw("count-closed-range", "messages between 5 and 20", (_p("min_messages", 5), _p("max_messages", 20))),
    BoundaryLaw(
        "date-closed-range",
        "date between 2026-01-01 and 2026-02-01",
        (_p("since", "2026-01-01"), _p("until", "2026-02-01")),
    ),
)


@dataclass(frozen=True, slots=True)
class CompositionLaw:
    law_id: str
    expression: str
    expected: object


QUERY_COMPOSITION_LAWS: tuple[CompositionLaw, ...] = (
    CompositionLaw(
        "boolean-or-composition",
        "repo:polylogue OR origin:chatgpt-export",
        BooleanFact("or", (_predicate("repo", "polylogue"), _predicate("origin", "chatgpt-export"))),
    ),
    CompositionLaw(
        "boolean-not-composition",
        "origin:chatgpt-export AND NOT title:slop",
        BooleanFact("and", (_predicate("origin", "chatgpt-export"), NotFact(_predicate("title", "slop")))),
    ),
)


@dataclass(frozen=True, slots=True)
class RetiredFieldExample:
    old_test: str
    replacement_law: str


# Exact candidates reviewed before deletion.  Tests that exercise rejection,
# quoting, execution, near-ID behavior, Boolean AST shape, and merge wiring
# remain deliberately outside this list.
RETIRED_FIELD_EXAMPLES: tuple[RetiredFieldExample, ...] = (
    RetiredFieldExample("TestLowererFieldMapping.test_repo", "field-repo"),
    RetiredFieldExample("TestLowererFieldMapping.test_origin", "field-origin"),
    RetiredFieldExample("TestLowererFieldMapping.test_origin_negated", "field-origin"),
    RetiredFieldExample("TestLowererFieldMapping.test_origin_alternation", "field-origin-vocabulary"),
    RetiredFieldExample("TestLowererFieldMapping.test_tag", "field-tag"),
    RetiredFieldExample("TestLowererFieldMapping.test_tag_negated", "field-tag"),
    RetiredFieldExample("TestLowererFieldMapping.test_path", "field-path"),
    RetiredFieldExample("TestLowererFieldMapping.test_cwd", "field-cwd"),
    RetiredFieldExample("TestLowererFieldMapping.test_tool", "field-tool"),
    RetiredFieldExample("TestLowererFieldMapping.test_tool_negated", "field-tool"),
    RetiredFieldExample("TestLowererFieldMapping.test_action_file_edit", "field-action"),
    RetiredFieldExample("TestLowererFieldMapping.test_action_negated", "field-action"),
    RetiredFieldExample("TestLowererFieldMapping.test_has_paste", "field-has-paste"),
    RetiredFieldExample("TestLowererFieldMapping.test_has_tools", "field-has-tools"),
    RetiredFieldExample("TestLowererFieldMapping.test_has_thinking", "field-has-thinking"),
    RetiredFieldExample("TestLowererFieldMapping.test_has_custom_type", "field-has-type"),
    RetiredFieldExample("TestLowererFieldMapping.test_id", "field-id"),
    RetiredFieldExample("TestLowererFieldMapping.test_session_alias", "field-session-alias"),
    RetiredFieldExample("TestLowererFieldMapping.test_title", "field-title"),
    RetiredFieldExample("TestLowererFieldMapping.test_contains", "field-contains"),
    RetiredFieldExample("TestLowererFieldMapping.test_lane", "field-lane"),
    RetiredFieldExample("TestLowererFieldMapping.test_messages_gte", "field-messages"),
    RetiredFieldExample("TestLowererFieldMapping.test_messages_lte", "count-inclusive-upper"),
    RetiredFieldExample("TestLowererFieldMapping.test_words_gte", "field-words"),
    RetiredFieldExample("TestLowererFieldMapping.test_since_absolute", "field-since"),
    RetiredFieldExample("TestLowererFieldMapping.test_until_relative", "field-until"),
    RetiredFieldExample("TestLowererFieldMapping.test_readable_messages_comparison", "count-exclusive-lower"),
    RetiredFieldExample("TestLowererFieldMapping.test_readable_words_less_than", "count-exclusive-upper"),
    RetiredFieldExample("TestLowererFieldMapping.test_readable_count_range", "count-closed-range"),
    RetiredFieldExample("TestLowererFieldMapping.test_readable_date_range", "date-closed-range"),
    RetiredFieldExample("TestLowererFieldMapping.test_readable_date_gte", "field-since"),
    RetiredFieldExample("TestLowererFieldMapping.test_readable_date_lte", "field-until"),
)


QUERY_FIELD_LAWS_BY_ID = {law.law_id: law for law in QUERY_FIELD_LAWS}
QUERY_BOUNDARY_LAWS_BY_ID = {law.law_id: law for law in QUERY_BOUNDARY_LAWS}
QUERY_COMPOSITION_LAWS_BY_ID = {law.law_id: law for law in QUERY_COMPOSITION_LAWS}


__all__ = [
    "BooleanFact",
    "BoundaryLaw",
    "CompositionLaw",
    "FieldProjection",
    "LineageFact",
    "NotFact",
    "PredicateFact",
    "QUERY_BOUNDARY_LAWS",
    "QUERY_BOUNDARY_LAWS_BY_ID",
    "QUERY_COMPOSITION_LAWS",
    "QUERY_COMPOSITION_LAWS_BY_ID",
    "QUERY_FIELD_LAWS",
    "QUERY_FIELD_LAWS_BY_ID",
    "RETIRED_FIELD_EXAMPLES",
    "QueryFieldLaw",
]
