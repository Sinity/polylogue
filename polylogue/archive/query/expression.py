"""Query expression parser/lowerer: DSL string → :class:`SessionQuerySpec`.

This module is the shared front-door for the query-expression language used
by all Polylogue read surfaces (CLI bare-query path, Python facade, and any
future surfaces that wire in).

Current executable grammar
--------------------------
The executable language is the Lark-backed query grammar in this module. It
currently accepts the compact session query form:

    repo:polylogue since:7d "json envelope"
    origin:(codex-session|claude-code-session) has:paste  # paste evidence
    path:polylogue/cli tool:bash action:file_edit
    near:"semantic search prompt"
    -id:bad tag:review
    messages:>=10 words:>=200
    date between 2026-01-01 and 2026-02-01

Clause forms:
- ``field:value``          — single-value field clause
- ``field:(a|b|c)``        — multi-value field clause (OR *within* field)
- ``-field:value``         — negated field clause
- ``"quoted phrase"``      — FTS phrase term
- ``-"quoted phrase"``     — excluded FTS phrase term
- ``bare_word``            — FTS bare word
- ``messages:>=N``         — count comparison (min_messages)
- ``messages:<=N``         — count comparison (max_messages)
- ``messages >= N``        — readable count comparison syntax
- ``messages between A and B`` — readable count range syntax
- ``words:>=N``            — count comparison (min_words)
- ``date >= ISO_OR_REL``   — readable date lower bound (since)
- ``date between A and B`` — readable date range (since/until)
- ``{...}``                — direct JSON spec (validated into SessionQuerySpec)

and explicit Boolean session predicates:

    sessions where (repo:polylogue OR origin:chatgpt-export) AND NOT tag:stale
    repo:polylogue OR repo:sinex
    messages where role:assistant AND text:timeout
    actions where action:file_edit AND path:polylogue/archive
    blocks where type:code AND text:timeout
    messages where time >= 2026-01-01T00:00:00+00:00
    sessions where repo:polylogue | messages where role:assistant | group by role | count | sort by count desc
    assertions where kind:annotation AND value.score:>=4
    exists assertion(value.status:approved)

Unit-scoped ``messages/actions/blocks/assertions/files/runs/observed-events/context-snapshots where ...`` predicates are executable
in two shared paths: ``compile_expression`` keeps the compatibility session
selector behavior by lowering them to correlated ``exists <unit>(...)``
predicates, while terminal query-unit surfaces preserve the selected unit and
return raw message/action/block/assertion/file/observed-event/context-snapshot rows from the same predicate semantics.
SQL-backed terminal units also support exact ``group by FIELD | count``
aggregate pipelines for the closed aggregate fields declared in this module,
plus aggregate ``sort by count|key [asc|desc]`` before ``limit``/``offset``.
Any piped unit source is terminal-only; ``compile_expression`` rejects it
instead of discarding pipeline stages while lowering to a session selector.
They also support ``time >= VALUE``, ``time <= VALUE``, and
``time between A and B`` predicates over the same row timestamp used by
``sort by time``.

The ``assertion`` unit additionally supports typed JSON-path predicates over
``value.<dotted.path>`` (polylogue-rxdo.7), since the plain ``value`` field is
a substring match over the whole JSON blob and cannot express label
analytics: ``value.score:>=4``, ``value.rubric.confidence:<0.5``, and plain
equality ``value.status:approved``; combine multiple paths the same way any
other field predicate combines, with ``AND``/``OR``
(``value.score:>=4 AND value.status:approved``). Comparison operators (``>``,
``>=``, ``<``, ``<=``) require a numeric right-hand side and compare the
extracted scalar as a number; ``=`` (the default when no operator is given)
compares the raw extracted scalar, so string/boolean/integer labels still
match by value. The path is restricted to dotted identifier segments (no
SQLite JSON-path wildcards or array indexing) and is only accepted for the
``assertion`` unit -- other units reject ``value.*`` fields as unknown.

Unknown fields and unsupported structured forms fail loudly. The Lark grammar
in this module is the query grammar. Compact field/text clauses and explicit
Boolean predicates are two entry shapes in that grammar, not a legacy/future
split.

Field and structure registries
------------------------------
:data:`EXPRESSION_FIELD_REGISTRY` maps every recognized session-scope DSL token
to its spec field name. :data:`STRUCTURAL_QUERY_UNIT_REGISTRY` maps the
``exists <unit>(...)`` units to the structural fields accepted for each unit.
Completion tools (#1844) should introspect these registries rather than
maintain parallel lists.

Public API
----------
:func:`compile_expression`
    Historical entrypoint name for parsing a DSL string and lowering it into a
    :class:`SessionQuerySpec`.

:func:`compile_expression_into`
    Merge a DSL string into an existing :class:`SessionQuerySpec`, additive
    (tuple fields are extended, scalar fields are overridden if set).

:class:`ExpressionCompileError`
    Typed parse/lowering error with a ``field`` attribute (``None`` for
    structural errors such as malformed Boolean syntax).
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field, replace
from difflib import get_close_matches
from typing import Any, Literal, cast

from lark import Lark, Token, Transformer, v_args
from lark.exceptions import UnexpectedInput, VisitError

from polylogue.archive.query.metadata import (
    _BOOLEAN_SUPPORTED_FIELDS,
    _STRUCTURAL_BOOLEAN_SUPPORTED_FIELDS,
    COUNT_QUERY_FIELD_REGISTRY,
    DATE_QUERY_FIELD_REGISTRY,
    EXPRESSION_FIELD_REGISTRY,
    NUMERIC_QUERY_FIELD_REGISTRY,
    PROJECTION_QUERY_UNITS,
    STRUCTURAL_QUERY_UNIT_REGISTRY,
    CountQueryFieldInfo,
    DateQueryFieldInfo,
    QueryUnitLowererKind,
    QueryUnitName,
    StructuralQueryUnitInfo,
    count_query_fields,
    count_query_operators,
    date_query_fields,
    date_query_operators,
    query_unit_descriptor,
    query_unit_field_info,
    query_unit_field_names,
    structural_query_fields,
    structural_query_units,
    terminal_query_source_list,
    terminal_query_source_pairs,
    terminal_query_unit,
)
from polylogue.archive.query.predicate import (
    QueryBoolPredicate,
    QueryCompareOp,
    QueryExistsPredicate,
    QueryExistsUnit,
    QueryFieldPredicate,
    QueryFieldRef,
    QueryLineagePredicate,
    QueryNotPredicate,
    QueryPredicate,
    QuerySemanticPredicate,
    QuerySequenceConstraint,
    QuerySequencePredicate,
    QueryTextPredicate,
)
from polylogue.archive.query.spec import (
    QUERY_ACTION_TYPES,
    SessionQuerySpec,
    normalize_retrieval_lane,
)
from polylogue.core.enums import Origin
from polylogue.errors import PolylogueError


def _count_field_regex() -> str:
    """Return a regex alternation for numeric comparison fields from metadata."""

    fields = (*COUNT_QUERY_FIELD_REGISTRY, *NUMERIC_QUERY_FIELD_REGISTRY)
    return "|".join(re.escape(field) for field in sorted(fields, key=len, reverse=True))


_COUNT_FIELD_REGEX = _count_field_regex()

# ---------------------------------------------------------------------------
# Error
# ---------------------------------------------------------------------------


class ExpressionCompileError(PolylogueError):
    """Raised when the expression DSL cannot be compiled to a query spec.

    Attributes:
        field: The DSL field token that caused the error, or ``None`` for
               structural errors (malformed Boolean syntax, unclosed quotes, etc.).
    """

    http_status_code = 400

    def __init__(self, message: str, *, field: str | None = None) -> None:
        super().__init__(message)
        self.field = field


class UnsupportedSessionTerminalActionError(ExpressionCompileError):
    """Raised when a session-source terminal action is not supported."""


#: Recognized ``has:`` sub-tokens that map to boolean spec flags.
#: Other values pass through to ``has_types``.
_HAS_BOOL_MAP: dict[str, str] = {
    "paste": "filter_has_paste",
    "tools": "filter_has_tool_use",
    "thinking": "filter_has_thinking",
}

# ---------------------------------------------------------------------------
# Parser AST
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _FieldToken:
    """``[neg]field:value`` or ``[neg]field:(a|b|c)``."""

    field: str
    raw_value: str  # after the colon, strip outer parens → values split on |
    negated: bool


@dataclass(frozen=True)
class _CountToken:
    """Readable count comparison such as ``messages:>=10``."""

    field: str
    op: QueryCompareOp
    number: int


@dataclass(frozen=True)
class _CountRangeToken:
    """``messages between 5 and 20`` or ``words between 100 and 500``."""

    field: str
    min_number: int
    max_number: int


@dataclass(frozen=True)
class _DateComparisonToken:
    """``date >= 2026-01-01`` or ``date < 7d``."""

    op: Literal[">", ">=", "<", "<="]
    value: str


@dataclass(frozen=True)
class _DateRangeToken:
    """``date between 2026-01-01 and 2026-02-01``."""

    min_value: str
    max_value: str


@dataclass(frozen=True)
class _TextToken:
    """A bare word or quoted phrase."""

    text: str
    quoted: bool
    negated: bool


@dataclass(frozen=True)
class _JsonToken:
    """A ``{...}`` raw JSON spec."""

    raw: str


_LexToken = (
    _FieldToken | _CountToken | _CountRangeToken | _DateComparisonToken | _DateRangeToken | _TextToken | _JsonToken
)


@dataclass(frozen=True)
class QueryExpressionAST:
    """Parsed query expression before lowering into ``SessionQuerySpec``."""

    clauses: tuple[_LexToken, ...]
    boolean_predicate: QueryPredicate | None = None


@dataclass(frozen=True)
class QueryUnitSort:
    """Terminal query-unit sort stage."""

    field: Literal["time", "count", "key"]
    direction: Literal["asc", "desc"] = "asc"


QueryUnitPipelineStageKind = Literal[
    "session_scope", "sort", "limit", "offset", "group", "count", "transform", "terminal"
]

#: Closed vocabulary of terminal actions that a query-unit pipeline can end in.
#: Each action name must have a registered executor in
#: :data:`polylogue.archive.query.unit_results.TERMINAL_ACTION_EXECUTORS`.
#: ``rows`` returns the resolved unit rows; ``count`` returns the aggregate
#: ``group by ... | count`` rollup. Future actions (read/analyze/bundle/
#: postmortem) register their unit-level executors here as they land (#2006).
QueryUnitTerminalAction = Literal["rows", "count"]
SESSION_SOURCE_UNIT: Literal["sessions"] = "sessions"
SessionTerminalAction = Literal["read", "analyze", "select", "mark", "delete", "continue"]
SESSION_TERMINAL_ACTIONS: frozenset[SessionTerminalAction] = frozenset(
    {"read", "analyze", "select", "mark", "delete", "continue"}
)


def _session_terminal_action(action: str) -> SessionTerminalAction:
    if action == "read":
        return "read"
    if action == "analyze":
        return "analyze"
    if action == "select":
        return "select"
    if action == "mark":
        return "mark"
    if action == "delete":
        return "delete"
    if action == "continue":
        return "continue"
    supported = ", ".join(sorted(SESSION_TERMINAL_ACTIONS))
    raise UnsupportedSessionTerminalActionError(
        f"unsupported session terminal action: {action!r}; supported actions: {supported}",
        field=None,
    )


@dataclass(frozen=True)
class QueryUnitSessionScopeStage:
    """Session-source stage in a terminal query-unit pipeline."""

    predicate: QueryPredicate

    def to_payload(self) -> dict[str, object]:
        return {"kind": "session_scope", "predicate": self.predicate.to_payload()}


@dataclass(frozen=True)
class QueryUnitSortStage:
    """Row or aggregate sort stage in a terminal query-unit pipeline."""

    sort: QueryUnitSort

    def to_payload(self) -> dict[str, object]:
        return {
            "kind": "sort",
            "sort": {
                "field": self.sort.field,
                "direction": self.sort.direction,
            },
        }


@dataclass(frozen=True)
class QueryUnitLimitStage:
    """Limit stage in a terminal query-unit pipeline."""

    value: int

    def to_payload(self) -> dict[str, object]:
        return {"kind": "limit", "value": self.value}


@dataclass(frozen=True)
class QueryUnitOffsetStage:
    """Offset stage in a terminal query-unit pipeline."""

    value: int

    def to_payload(self) -> dict[str, object]:
        return {"kind": "offset", "value": self.value}


@dataclass(frozen=True)
class QueryUnitGroupStage:
    """Aggregate grouping stage in a terminal query-unit pipeline."""

    field: str

    def to_payload(self) -> dict[str, object]:
        return {"kind": "group", "field": self.field}


@dataclass(frozen=True)
class QueryUnitCountStage:
    """Count aggregation stage in a terminal query-unit pipeline."""

    def to_payload(self) -> dict[str, object]:
        return {"kind": "count", "metric": "count"}


@dataclass(frozen=True)
class QueryUnitTransformStage:
    """Named row-shaping transform stage in a terminal query-unit pipeline.

    Reserved vocabulary member for #2006's ``Transform(name, args)`` pipeline
    node. No row-level transform is wired today, so this stage is part of the
    closed AST/payload vocabulary but is never produced by the current parser;
    it exists so surfaces and the executor registry share one source of truth
    for the pipeline-stage taxonomy rather than re-deriving it.
    """

    name: str
    args: tuple[tuple[str, str], ...] = ()

    def to_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {"kind": "transform", "name": self.name}
        if self.args:
            payload["args"] = dict(self.args)
        return payload


@dataclass(frozen=True)
class QueryUnitTerminalStage:
    """Terminal action that emits the resolved query-unit set (#2006).

    Every terminal query-unit pipeline ends in exactly one terminal node. The
    ``action`` names the executor that runs the final ``select -> shape ->
    terminal`` step (``rows`` for unit rows, ``count`` for the aggregate
    rollup). ``args`` carries typed terminal parameters for future actions
    (read view, analyze mode, bundle kind) without widening the node shape.
    """

    action: QueryUnitTerminalAction
    args: tuple[tuple[str, str], ...] = ()

    def to_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {"kind": "terminal", "action": self.action}
        if self.args:
            payload["args"] = dict(self.args)
        return payload


QueryUnitPipelineStage = (
    QueryUnitSessionScopeStage
    | QueryUnitSortStage
    | QueryUnitLimitStage
    | QueryUnitOffsetStage
    | QueryUnitGroupStage
    | QueryUnitCountStage
    | QueryUnitTransformStage
    | QueryUnitTerminalStage
)


@dataclass(frozen=True)
class QueryUnitPipeline:
    """Executable terminal-row pipeline parsed from the query DSL."""

    source_unit: QueryUnitName
    predicate: QueryPredicate
    session_predicate: QueryPredicate | None = None
    stages: tuple[QueryUnitPipelineStage, ...] = ()
    limit: int | None = None
    offset: int | None = None
    sort: QueryUnitSort | None = None
    group_by: str | None = None
    aggregate: Literal["count"] | None = None
    terminal: QueryUnitTerminalStage = QueryUnitTerminalStage(action="rows")

    def to_payload(self) -> dict[str, object]:
        # The terminal node is the final element of the ordered stage sequence
        # (#2006 Pipeline AST: ``... Sort, Limit, Terminal(action)``), so
        # ``stages`` and the executed ``pipeline_stages`` page metadata stay
        # identical and always culminate in the terminal action.
        payload: dict[str, object] = {
            "source": {
                "unit": self.source_unit,
                "predicate": self.predicate.to_payload(),
            },
            "stages": [stage.to_payload() for stage in self.stages] + [self.terminal.to_payload()],
        }
        if self.session_predicate is not None:
            payload["session_scope"] = self.session_predicate.to_payload()
        result: dict[str, object] = {}
        if self.sort is not None:
            result["sort"] = {
                "field": self.sort.field,
                "direction": self.sort.direction,
            }
        if self.group_by is not None:
            result["group_by"] = self.group_by
        if self.aggregate is not None:
            result["aggregate"] = self.aggregate
        if self.limit is not None:
            result["limit"] = self.limit
        if self.offset is not None:
            result["offset"] = self.offset
        if result:
            payload["result"] = result
        return payload


@dataclass(frozen=True)
class SessionQueryTerminalStage:
    """Terminal action that emits or mutates the resolved session set."""

    action: SessionTerminalAction
    args: tuple[tuple[str, object], ...] = ()

    def to_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {"kind": "terminal", "action": self.action}
        if self.args:
            payload["args"] = dict(self.args)
        return payload


@dataclass(frozen=True)
class SessionQueryPipeline:
    """Executable session-source pipeline for root query verbs."""

    terminal: SessionQueryTerminalStage
    predicate: QueryPredicate | None = None
    source_unit: Literal["sessions"] = SESSION_SOURCE_UNIT

    def to_payload(self) -> dict[str, object]:
        source: dict[str, object] = {"unit": self.source_unit}
        payload: dict[str, object] = {
            "source": source,
            "stages": [self.terminal.to_payload()],
        }
        if self.predicate is not None:
            source["predicate"] = self.predicate.to_payload()
        return payload


@dataclass(frozen=True)
class QueryUnitSource:
    """Explicit unit-changing source parsed from ``<unit>s where ...``."""

    unit: QueryUnitName
    predicate: QueryPredicate
    session_predicate: QueryPredicate | None = None
    limit: int | None = None
    offset: int | None = None
    sort: QueryUnitSort | None = None
    group_by: str | None = None
    aggregate: Literal["count"] | None = None
    pipeline_stages: tuple[QueryUnitPipelineStage, ...] = ()

    @property
    def pipeline(self) -> QueryUnitPipeline:
        """Return the typed executable pipeline for this terminal source."""

        terminal_action: QueryUnitTerminalAction = "count" if self.aggregate == "count" else "rows"
        return QueryUnitPipeline(
            source_unit=self.unit,
            predicate=self.predicate,
            session_predicate=self.session_predicate,
            stages=self.pipeline_stages,
            limit=self.limit,
            offset=self.offset,
            sort=self.sort,
            group_by=self.group_by,
            aggregate=self.aggregate,
            terminal=QueryUnitTerminalStage(action=terminal_action),
        )


ExplainClauseKind = Literal["field", "count", "count_range", "date", "date_range", "text", "json"]


@dataclass(frozen=True)
class QueryExpressionExplainClause:
    """Serializable clause view for parser/lowerer diagnostics."""

    kind: ExplainClauseKind
    field: str | None = None
    value: str | None = None
    negated: bool = False
    quoted: bool = False
    op: QueryCompareOp | None = None
    number: int | None = None
    min_number: int | None = None
    max_number: int | None = None
    min_value: str | None = None
    max_value: str | None = None

    def to_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {"kind": self.kind}
        if self.field is not None:
            payload["field"] = self.field
        if self.value is not None:
            payload["value"] = self.value
        if self.negated:
            payload["negated"] = True
        if self.quoted:
            payload["quoted"] = True
        if self.op is not None:
            payload["op"] = self.op
        if self.number is not None:
            payload["number"] = self.number
        if self.min_number is not None:
            payload["min_number"] = self.min_number
        if self.max_number is not None:
            payload["max_number"] = self.max_number
        if self.min_value is not None:
            payload["min_value"] = self.min_value
        if self.max_value is not None:
            payload["max_value"] = self.max_value
        return payload


@dataclass(frozen=True)
class QueryExpressionExplanation:
    """Debug envelope for query parsing, lowering, and execution-plan selection."""

    source_text: str
    clauses: tuple[QueryExpressionExplainClause, ...]
    lowerer: str
    lowered_spec: SessionQuerySpec
    plan_description: tuple[str, ...]
    selected_units: tuple[str, ...] = ("session",)
    execution_legs: tuple[str, ...] = ()
    unsupported_nodes: tuple[str, ...] = ()
    predicate: QueryPredicate | None = None
    ast: dict[str, object] | None = None
    lowering_plan: dict[str, object] | None = None

    def to_payload(self) -> dict[str, object]:
        return {
            "source_text": self.source_text,
            "clauses": [clause.to_payload() for clause in self.clauses],
            "predicate": self.predicate.to_payload() if self.predicate is not None else None,
            "ast": self.ast,
            "lowerer": self.lowerer,
            "lowering_plan": self.lowering_plan,
            "selected_units": list(self.selected_units),
            "execution_legs": list(self.execution_legs),
            "plan_description": list(self.plan_description),
            "unsupported_nodes": list(self.unsupported_nodes),
        }


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

_QUERY_GRAMMAR = rf"""
    compact_query: compact_clause*

    ?compact_clause: COUNT_CLAUSE       -> count_clause
        | COUNT_FIELD BETWEEN INT AND INT -> count_between_clause
        | COUNT_FIELD COMP_OP INT    -> count_compare_clause
        | DATE_FIELD BETWEEN DATE_VALUE AND DATE_VALUE -> date_between_clause
        | DATE_FIELD DATE_COMP_OP DATE_VALUE -> date_compare_clause
        | FIELD_CLAUSE          -> field_clause
        | NEG_QUOTED_TEXT       -> neg_quoted_text
        | QUOTED_TEXT           -> quoted_text
        | NEG_BARE_TEXT         -> neg_bare_text
        | BARE_TEXT             -> bare_text

    boolean_query: session_prefix? expr

    session_prefix: SESSIONS WHERE

    ?expr: or_expr
    ?or_expr: and_expr (OR and_expr)*      -> or_expr
    ?and_expr: not_expr (AND not_expr)*    -> and_expr
    ?not_expr: NOT not_expr                -> not_expr
        | atom
    ?atom: EXISTS STRUCT_UNIT "(" expr ")" -> exists_leaf
        | SEQ "(" sequence_step (sequence_link sequence_step)+ ")" -> sequence_leaf
        | SEMANTIC_QUOTED_TEXT             -> semantic_quoted_leaf
        | SEMANTIC_BARE_TEXT               -> semantic_bare_leaf
        | FTS_QUOTED_TEXT                  -> fts_quoted_leaf
        | FTS_BARE_TEXT                    -> fts_bare_leaf
        | COUNT_CLAUSE                     -> count_leaf
        | COUNT_FIELD BETWEEN INT AND INT  -> count_between_leaf
        | COUNT_FIELD COMP_OP INT          -> count_compare_leaf
        | DATE_FIELD BETWEEN DATE_VALUE AND DATE_VALUE -> date_between_leaf
        | DATE_FIELD DATE_COMP_OP DATE_VALUE -> date_compare_leaf
        | TIME_FIELD BETWEEN DATE_VALUE AND DATE_VALUE -> time_between_leaf
        | TIME_FIELD DATE_COMP_OP DATE_VALUE -> time_compare_leaf
        | FIELD_CLAUSE                     -> field_leaf
        | "(" expr ")"
    sequence_step: sequence_atom (AND sequence_atom)*
    sequence_link: ARROW SEQUENCE_CONSTRAINT?
    ?sequence_atom: FIELD_CLAUSE                     -> field_leaf

    SESSIONS: /sessions/i
    WHERE: /where/i
    EXISTS: /exists/i
    SEQ: /seq/i
    ARROW: "->"
    SEQUENCE_CONSTRAINT.8: /\[(?:next|within:\d+(?:ms|s|m|h|d))\]/i
    STRUCT_UNIT: /(observed-event|context-snapshot|message|action|block|assertion|file|run)/i
    OR: /or/i
    AND: /and/i
    NOT: /not/i
    BETWEEN.6: /between/i
    COUNT_FIELD.6: /({_COUNT_FIELD_REGEX})/i
    DATE_FIELD.6: /date/i
    TIME_FIELD.6: /time/i
    DATE_COMP_OP: ">=" | "<=" | "=" | ">" | "<"
    COMP_OP: ">=" | "<=" | "=" | ">" | "<"
    DATE_VALUE: /[^\s"()]+/
    SEMANTIC_QUOTED_TEXT.7: /(?:semantic|near:text):"(\\.|[^"\\])*"/i
    SEMANTIC_BARE_TEXT.6: /(?:semantic|near:text):[^\s"()]+/i
    FTS_QUOTED_TEXT.6: /~"(\\.|[^"\\])*"/
    FTS_BARE_TEXT.5: /~[^\s"()]+/
    COUNT_CLAUSE.8: /({_COUNT_FIELD_REGEX}):(>=|<=|=|>|<)\d+(?!\S)/
    FIELD_CLAUSE.4: /-?[a-zA-Z_][a-zA-Z0-9_.]*:(?:"(\\.|[^"\\])*"|\([^)]*\)|[^\s"()\[\]{{}}]+)/
    NEG_QUOTED_TEXT.3: /-"(\\.|[^"\\])*"/
    QUOTED_TEXT.2: /"(\\.|[^"\\])*"/
    NEG_BARE_TEXT.1: /-[^\s"]+/
    BARE_TEXT: /[^\s"]+/

    %import common.INT
    %import common.WS
    %ignore WS
"""


_QUERY_PARSER = Lark(
    _QUERY_GRAMMAR,
    parser="lalr",
    start=["compact_query", "boolean_query"],
    maybe_placeholders=False,
)

_COUNT_CLAUSE_RE = re.compile(
    rf"^({_COUNT_FIELD_REGEX}):(>=|<=|=)(\d+)$",
    re.IGNORECASE,
)
_FIELD_CLAUSE_RE = re.compile(
    r"""
    ^(-?)
    ([a-zA-Z_][a-zA-Z0-9_.]*)
    :
    (
        "(?:\\.|[^"\\])*"
        |
        \([^)]*\)
        |
        [^\s"()\[\]{}]+
    )
    $
    """,
    re.VERBOSE,
)


def _unknown_query_field_message(field_name: str, *, include_structural: bool = False) -> str:
    recognized = sorted(EXPRESSION_FIELD_REGISTRY)
    message = f"unknown query field {field_name!r}; recognized fields: " + ", ".join(recognized)
    suggestions = get_close_matches(field_name, recognized, n=3, cutoff=0.6)
    if suggestions:
        message += "; did you mean: " + ", ".join(suggestions)
    if include_structural:
        message += "; structural fields: " + ", ".join(sorted(_STRUCTURAL_BOOLEAN_SUPPORTED_FIELDS))
    return message


def _decode_escaped_string(token: Token) -> str:
    try:
        decoded = json.loads(str(token))
    except json.JSONDecodeError as exc:
        raise ExpressionCompileError(f"invalid quoted string: {exc}", field=None) from exc
    if not isinstance(decoded, str):
        raise ExpressionCompileError("quoted value did not decode to a string", field=None)
    return decoded


def _parse_error_message(expression: str, exc: UnexpectedInput) -> str:
    structural_words = re.search(r"\bexists\s+message\(.*\bwords:", expression, re.IGNORECASE)
    if structural_words is not None:
        return "field 'words' requires a numeric value"
    malformed_count = re.search(rf"\b({_COUNT_FIELD_REGEX}):", expression, re.IGNORECASE)
    if malformed_count is not None:
        field = malformed_count.group(1).lower()
        return f"use comparison operator for {field!r}: e.g. {field}:>=10"
    if expression.lstrip().startswith("("):
        return f"invalid Boolean query expression near column {exc.column}"
    if '"' in expression:
        return "unclosed quoted string"
    return f"invalid query expression near column {exc.column}"


def _parse_error_field(expression: str) -> str | None:
    structural_words = re.search(r"\bexists\s+message\(.*\bwords:", expression, re.IGNORECASE)
    if structural_words is not None:
        return "words"
    malformed_count = re.search(rf"\b({_COUNT_FIELD_REGEX}):", expression, re.IGNORECASE)
    if malformed_count is not None:
        return malformed_count.group(1).lower()
    return None


def _normalize_count_comparison(
    field_name: str,
    op_text: str,
    number_text: str,
) -> _CountToken:
    field = field_name.lower()
    number = int(number_text)
    op = cast(QueryCompareOp, op_text if op_text in {">", ">=", "<", "<=", "="} else "=")
    return _CountToken(field=field, op=op, number=number)


def _normalize_count_range(field_name: str, min_text: str, max_text: str) -> _CountRangeToken:
    min_number = int(min_text)
    max_number = int(max_text)
    field = field_name.lower()
    if min_number > max_number:
        raise ExpressionCompileError(
            f"{field} range lower bound {min_number} is greater than upper bound {max_number}",
            field=field,
        )
    return _CountRangeToken(field=field, min_number=min_number, max_number=max_number)


def _normalize_date_comparison(op_text: str, value_text: str) -> _DateComparisonToken:
    if op_text in {">=", ">", "<=", "<"}:
        return _DateComparisonToken(
            op=cast(Literal[">", ">=", "<", "<="], op_text), value=_parse_relative_date(value_text)
        )
    raise ExpressionCompileError("date equality is not supported; use date between A and B", field="date")


def _normalize_date_range(min_text: str, max_text: str) -> _DateRangeToken:
    return _DateRangeToken(min_value=_parse_relative_date(min_text), max_value=_parse_relative_date(max_text))


def _normalize_time_comparison(op_text: str, value_text: str) -> QueryFieldPredicate:
    if op_text in {">=", ">", "<=", "<"}:
        return QueryFieldPredicate(
            field="time", values=(_parse_relative_date(value_text),), op=cast(QueryCompareOp, op_text)
        )
    raise ExpressionCompileError("time equality is not supported; use time between A and B", field="time")


def _normalize_time_range(min_text: str, max_text: str) -> QueryPredicate:
    return QueryBoolPredicate(
        op="and",
        children=(
            QueryFieldPredicate(field="time", values=(_parse_relative_date(min_text),), op=">="),
            QueryFieldPredicate(field="time", values=(_parse_relative_date(max_text),), op="<="),
        ),
    )


@v_args(inline=True)
class _QueryTransformer(Transformer[Token, _LexToken | str | QueryExpressionAST]):
    def compact_query(self, *clauses: _LexToken) -> QueryExpressionAST:
        return QueryExpressionAST(tuple(clauses))

    def count_clause(self, token: Token) -> _CountToken:
        matched = _COUNT_CLAUSE_RE.match(str(token))
        if matched is None:
            raise ExpressionCompileError(f"invalid count clause: {token}", field=None)
        return _normalize_count_comparison(matched.group(1), matched.group(2), matched.group(3))

    def count_compare_clause(self, field_name: Token, op: Token, number: Token) -> _CountToken:
        return _normalize_count_comparison(str(field_name), str(op), str(number))

    def count_between_clause(
        self,
        field_name: Token,
        _between: Token,
        min_number: Token,
        _and: Token,
        max_number: Token,
    ) -> _CountRangeToken:
        return _normalize_count_range(str(field_name), str(min_number), str(max_number))

    def date_compare_clause(self, _field_name: Token, op: Token, value: Token) -> _DateComparisonToken:
        return _normalize_date_comparison(str(op), str(value))

    def date_between_clause(
        self,
        _field_name: Token,
        _between: Token,
        min_value: Token,
        _and: Token,
        max_value: Token,
    ) -> _DateRangeToken:
        return _normalize_date_range(str(min_value), str(max_value))

    def field_clause(self, token: Token) -> _FieldToken:
        matched = _FIELD_CLAUSE_RE.match(str(token))
        if matched is None:
            raise ExpressionCompileError(f"invalid field clause: {token}", field=None)
        negated, field_name, raw_value = matched.group(1), matched.group(2), matched.group(3)
        value_path_field = field_name.lower().startswith("value.")
        if raw_value.startswith('"'):
            if not value_path_field:
                raw_value = _decode_escaped_string(Token("ESCAPED_STRING", raw_value))
        elif raw_value.startswith("("):
            raw_value = raw_value[1:-1]
        return _FieldToken(field=field_name.lower(), raw_value=raw_value, negated=bool(negated))

    def neg_quoted_text(self, text: Token) -> _TextToken:
        return _TextToken(
            text=_decode_escaped_string(Token("ESCAPED_STRING", str(text)[1:])),
            quoted=True,
            negated=True,
        )

    def quoted_text(self, text: Token) -> _TextToken:
        return _TextToken(text=_decode_escaped_string(text), quoted=True, negated=False)

    def neg_bare_text(self, text: Token) -> _TextToken:
        value = str(text)
        return _TextToken(text=value[1:], quoted=False, negated=True)

    def bare_text(self, text: Token) -> _TextToken:
        return _TextToken(text=str(text), quoted=False, negated=False)


_QUERY_TRANSFORMER = _QueryTransformer()


def _canonical_session_alias(field_name: str) -> str:
    return "id" if field_name == "session" else field_name


_VALUE_PATH_PREFIX = "value."
_VALUE_PATH_SEGMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _is_assertion_value_path_field(field_name: str) -> bool:
    """Return whether *field_name* is a dynamic ``value.<path>`` assertion field.

    These are not enumerable in the closed structural-field registries
    (:data:`_ASSERTION_STRUCTURAL_FIELDS` etc.) because the path is caller-
    chosen JSON-object navigation below the assertion ``value`` column
    (polylogue-rxdo.7 typed value predicates). Every dotted segment must be a
    plain identifier -- no SQLite JSON-path wildcards/array indexing -- so the
    shape stays unambiguous before it reaches SQL lowering.
    """

    if not field_name.startswith(_VALUE_PATH_PREFIX):
        return False
    suffix = field_name[len(_VALUE_PATH_PREFIX) :]
    if not suffix:
        return False
    return all(_VALUE_PATH_SEGMENT_RE.match(segment) for segment in suffix.split("."))


def _field_token_to_predicate(token: _FieldToken) -> QueryPredicate:
    field_name = token.field
    session_field = _scoped_session_field(field_name)
    validation_field = session_field or field_name
    is_value_path_field = session_field is None and _is_assertion_value_path_field(field_name)
    if (
        session_field is None
        and field_name not in EXPRESSION_FIELD_REGISTRY
        and field_name not in _STRUCTURAL_BOOLEAN_SUPPORTED_FIELDS
        and not is_value_path_field
    ):
        raise ExpressionCompileError(
            _unknown_query_field_message(field_name, include_structural=True),
            field=field_name,
        )
    if (
        validation_field not in _BOOLEAN_SUPPORTED_FIELDS
        and validation_field not in _STRUCTURAL_BOOLEAN_SUPPORTED_FIELDS
        and not is_value_path_field
    ):
        raise ExpressionCompileError(
            f"field {field_name!r} is not supported inside Boolean SQL predicates yet",
            field=field_name,
        )

    values = (
        _split_assertion_value_path_alternation(token.raw_value)
        if is_value_path_field and token.raw_value
        else _split_alternation(token.raw_value)
        if token.raw_value
        else ()
    )
    if validation_field == "origin":
        known_origins = {o.value for o in Origin}
        for value in values:
            if value not in known_origins:
                raise ExpressionCompileError(
                    f"unknown origin {value!r}; recognized: " + ", ".join(sorted(known_origins)),
                    field="origin",
                )
    elif validation_field == "action":
        for value in values:
            candidate = value.strip().lower()
            if candidate not in QUERY_ACTION_TYPES:
                raise ExpressionCompileError(
                    f"unknown action {value!r}; recognized: " + ", ".join(QUERY_ACTION_TYPES),
                    field="action",
                )
        values = tuple(value.strip().lower() for value in values if value.strip())
    elif validation_field in {"tool", "has", "role", "type", "command", "path", "output"}:
        values = tuple(value.strip().lower() for value in values if value.strip())
    elif validation_field in {"since", "until"} and values:
        values = (_parse_relative_date(values[-1]),)
    elif validation_field in COUNT_QUERY_FIELD_REGISTRY or validation_field in NUMERIC_QUERY_FIELD_REGISTRY:
        if not values:
            raise ExpressionCompileError(f"field {field_name!r} requires a numeric value", field=field_name)
        match = re.fullmatch(r"(>=|<=|=|>|<)?(\d+)", values[-1].strip())
        if match is None:
            raise ExpressionCompileError(f"field {field_name!r} requires a numeric value", field=field_name)
        raw_op = match.group(1) or "="
        op_text = cast(QueryCompareOp, raw_op)
        values = (match.group(2),)
        count_predicate = QueryFieldPredicate(field=field_name, values=values, op=op_text)
        return QueryNotPredicate(count_predicate) if token.negated else count_predicate
    elif is_value_path_field:
        if not values:
            raise ExpressionCompileError(f"field {field_name!r} requires a value", field=field_name)
        parsed_values: list[str] = []
        parsed_ops: set[QueryCompareOp] = set()
        for raw_value in values:
            match = re.fullmatch(r"(>=|<=|=|>|<)?(.+)", raw_value.strip(), re.DOTALL)
            if match is None or not match.group(2).strip():
                raise ExpressionCompileError(f"field {field_name!r} requires a value", field=field_name)
            op_text = cast(QueryCompareOp, match.group(1) or "=")
            value_text = match.group(2).strip()
            if op_text != "=":
                try:
                    numeric_value = float(value_text)
                except ValueError as exc:
                    raise ExpressionCompileError(
                        f"field {field_name!r} comparison operator {op_text!r} requires a numeric value",
                        field=field_name,
                    ) from exc
                if not math.isfinite(numeric_value):
                    raise ExpressionCompileError(
                        f"field {field_name!r} comparison operator {op_text!r} requires a finite numeric value",
                        field=field_name,
                    )
            parsed_ops.add(op_text)
            parsed_values.append(value_text)
        if len(parsed_ops) != 1:
            raise ExpressionCompileError(
                f"field {field_name!r} alternation must use one comparison operator",
                field=field_name,
            )
        op_text = next(iter(parsed_ops))
        if op_text != "=" and len(parsed_values) != 1:
            raise ExpressionCompileError(
                f"field {field_name!r} numeric comparison accepts one value",
                field=field_name,
            )
        value_path_predicate = QueryFieldPredicate(field=field_name, values=tuple(parsed_values), op=op_text)
        return QueryNotPredicate(value_path_predicate) if token.negated else value_path_predicate
    elif validation_field == "date" and values:
        raw_value = values[-1].strip()
        match = re.fullmatch(r"(>=|<=|>|<)?(.+)", raw_value)
        if match is not None:
            raw_op = match.group(1) or "="
            normalized_op = cast(QueryCompareOp, raw_op)
            values = (_parse_relative_date(match.group(2).strip()),)
            date_predicate = QueryFieldPredicate(field=field_name, values=values, op=normalized_op)
            return QueryNotPredicate(date_predicate) if token.negated else date_predicate
    elif validation_field == "lineage":
        if token.negated:
            raise ExpressionCompileError("negation is not supported for 'lineage'", field=field_name)
        seed = values[-1].strip() if values else ""
        if seed.startswith("id:"):
            seed = seed[len("id:") :].strip()
        if not seed:
            raise ExpressionCompileError("lineage:id: requires a session id", field=field_name)
        return QueryLineagePredicate(seed_session_id=seed)

    predicate: QueryPredicate = QueryFieldPredicate(field=field_name, values=values)
    return QueryNotPredicate(predicate) if token.negated else predicate


def _scoped_session_field(field_name: str) -> str | None:
    prefix = "session."
    if not field_name.startswith(prefix):
        return None
    scoped = field_name[len(prefix) :]
    if not scoped:
        raise ExpressionCompileError("session. predicates require a field name", field=field_name)
    return scoped


def _count_token_to_predicate(token: _CountToken) -> QueryFieldPredicate:
    return QueryFieldPredicate(field=token.field, values=(str(token.number),), op=token.op)


def _count_range_token_to_predicate(token: _CountRangeToken) -> QueryPredicate:
    return QueryBoolPredicate(
        op="and",
        children=(
            QueryFieldPredicate(field=token.field, values=(str(token.min_number),), op=">="),
            QueryFieldPredicate(field=token.field, values=(str(token.max_number),), op="<="),
        ),
    )


def _date_token_to_predicate(token: _DateComparisonToken) -> QueryFieldPredicate:
    return QueryFieldPredicate(field="date", values=(token.value,), op=token.op)


def _date_range_token_to_predicate(token: _DateRangeToken) -> QueryPredicate:
    return QueryBoolPredicate(
        op="and",
        children=(
            QueryFieldPredicate(field="date", values=(token.min_value,), op=">="),
            QueryFieldPredicate(field="date", values=(token.max_value,), op="<="),
        ),
    )


def _merge_bool_children(op: Literal["and", "or"], children: tuple[QueryPredicate, ...]) -> QueryPredicate:
    flattened: list[QueryPredicate] = []
    for child in children:
        if isinstance(child, QueryBoolPredicate) and child.op == op:
            flattened.extend(child.children)
        else:
            flattened.append(child)
    if len(flattened) == 1:
        return flattened[0]
    return QueryBoolPredicate(op=op, children=tuple(flattened))


def _predicate_children(items: tuple[object, ...]) -> tuple[QueryPredicate, ...]:
    return tuple(
        item
        for item in items
        if isinstance(
            item,
            QueryFieldPredicate
            | QueryBoolPredicate
            | QueryNotPredicate
            | QueryExistsPredicate
            | QueryLineagePredicate
            | QuerySequencePredicate
            | QueryTextPredicate
            | QuerySemanticPredicate,
        )
    )


def _sequence_step_has_positive_match(predicate: QueryPredicate) -> bool:
    if isinstance(predicate, QueryFieldPredicate):
        return bool(predicate.values)
    if isinstance(predicate, QueryBoolPredicate):
        return any(_sequence_step_has_positive_match(child) for child in predicate.children)
    return False


def _validate_predicate_context(predicate: QueryPredicate, *, unit: Literal["session"] | QueryUnitName) -> None:
    if isinstance(predicate, QueryFieldPredicate):
        session_field = _scoped_session_field(predicate.field)
        if unit == "session":
            supported = _BOOLEAN_SUPPORTED_FIELDS
            supported_field = predicate.field
            validation_field = predicate.field
        else:
            supported = set(query_unit_field_names(unit))
            supported_field = predicate.field
            validation_field = session_field or predicate.field
        value_path_field = unit == "assertion" and _is_assertion_value_path_field(supported_field)
        if supported_field not in supported and not value_path_field:
            raise ExpressionCompileError(
                f"field {predicate.field!r} is not supported for {unit} predicates",
                field=predicate.field,
            )
        if (
            validation_field
            in {
                "role",
                "type",
                "text",
                "body",
                "tool",
                "action",
                "command",
                "path",
                "output",
                "kind",
                "status",
                "target",
                "target_ref",
                "scope",
                "scope_ref",
                "key",
                "author",
                "author_ref",
                "author_kind",
                "visibility",
                "value",
                "evidence",
                "context",
                "delivery_state",
                "subject",
                "subject_ref",
                "object",
                "object_ref",
                "summary",
                "agent",
                "agent_ref",
                "branch",
                "context_snapshot",
                "context_snapshot_ref",
                "confidence",
                "cwd",
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
                "run",
                "run_ref",
                "transcript",
                "transcript_ref",
            }
            and not predicate.values
        ):
            raise ExpressionCompileError(f"field {predicate.field!r} requires a value", field=predicate.field)
        if validation_field == "date":
            if not predicate.values:
                raise ExpressionCompileError("field 'date' requires a value", field="date")
            if predicate.op not in {">", ">=", "<", "<="}:
                raise ExpressionCompileError("field 'date' supports only >=, <=, >, <, and between", field="date")
        if validation_field == "time":
            if unit == "session":
                raise ExpressionCompileError(
                    "field 'time' is supported only for terminal unit predicates", field="time"
                )
            if not predicate.values:
                raise ExpressionCompileError("field 'time' requires a value", field="time")
            if predicate.op not in {">", ">=", "<", "<="}:
                raise ExpressionCompileError("field 'time' supports only >=, <=, >, <, and between", field="time")
        if validation_field in COUNT_QUERY_FIELD_REGISTRY or validation_field in NUMERIC_QUERY_FIELD_REGISTRY:
            if not predicate.values:
                raise ExpressionCompileError(
                    f"field {validation_field!r} requires a numeric value", field=validation_field
                )
            try:
                int(predicate.values[-1])
            except ValueError as exc:
                raise ExpressionCompileError(
                    f"field {validation_field!r} requires a numeric value", field=validation_field
                ) from exc
        if value_path_field:
            if not predicate.values:
                raise ExpressionCompileError(f"field {predicate.field!r} requires a value", field=predicate.field)
            if predicate.op != "=":
                if len(predicate.values) != 1:
                    raise ExpressionCompileError(
                        f"field {predicate.field!r} numeric comparison accepts one value",
                        field=predicate.field,
                    )
                try:
                    numeric_value = float(predicate.values[0])
                except ValueError as exc:
                    raise ExpressionCompileError(
                        f"field {predicate.field!r} comparison operator {predicate.op!r} requires a numeric value",
                        field=predicate.field,
                    ) from exc
                if not math.isfinite(numeric_value):
                    raise ExpressionCompileError(
                        f"field {predicate.field!r} comparison operator {predicate.op!r} requires a finite numeric value",
                        field=predicate.field,
                    )
        return
    if isinstance(predicate, QueryNotPredicate):
        _validate_predicate_context(predicate.child, unit=unit)
        return
    if isinstance(predicate, QueryBoolPredicate):
        for child in predicate.children:
            _validate_predicate_context(child, unit=unit)
        return
    if isinstance(predicate, QueryExistsPredicate):
        if unit != "session":
            raise ExpressionCompileError("exists predicates are only supported from sessions", field=None)
        _validate_predicate_context(predicate.child, unit=predicate.unit)
        return
    if isinstance(predicate, QueryLineagePredicate):
        if not predicate.seed_session_id.strip():
            raise ExpressionCompileError("lineage predicate requires a session id", field="lineage")
        return
    if isinstance(predicate, QuerySequencePredicate):
        if unit != "session":
            raise ExpressionCompileError("seq predicates are only supported from sessions", field=None)
        if len(predicate.steps) < 2:
            raise ExpressionCompileError("seq() requires at least two action steps", field=None)
        for step in predicate.steps:
            _validate_predicate_context(step, unit="action")
            if not _sequence_step_has_positive_match(step):
                raise ExpressionCompileError(
                    "seq() steps must include at least one positive action predicate", field=None
                )
        return
    if isinstance(predicate, QueryTextPredicate):
        if unit != "session":
            raise ExpressionCompileError("FTS predicates are only supported from sessions", field=None)
        if not predicate.text.strip():
            raise ExpressionCompileError("FTS predicate requires text", field=None)
        return
    if isinstance(predicate, QuerySemanticPredicate):
        if unit != "session":
            raise ExpressionCompileError("semantic predicates are only supported from sessions", field=None)
        if not predicate.text.strip():
            raise ExpressionCompileError("semantic predicate requires text", field=None)
        return


def _field_ref_for_predicate(
    predicate: QueryFieldPredicate,
    *,
    unit: Literal["session"] | QueryUnitName,
) -> QueryFieldRef:
    session_field = _scoped_session_field(predicate.field)
    if unit == "session":
        return QueryFieldRef(
            scope="session",
            name=_canonical_session_alias(predicate.field),
            source_name=predicate.field,
        )
    if session_field is not None:
        field_info = query_unit_field_info(unit, predicate.field)
        source_name = field_info.name if field_info is not None else predicate.field
        return QueryFieldRef(
            scope="session",
            name=session_field,
            source_name=source_name,
            unit=unit,
        )
    field_info = query_unit_field_info(unit, predicate.field)
    field_name = field_info.name if field_info is not None else predicate.field
    return QueryFieldRef(scope="unit", name=field_name, source_name=field_name, unit=unit)


def _bind_predicate_context(
    predicate: QueryPredicate,
    *,
    unit: Literal["session"] | QueryUnitName,
) -> QueryPredicate:
    """Validate a predicate tree and attach closed field identity to leaves."""

    _validate_predicate_context(predicate, unit=unit)
    if isinstance(predicate, QueryFieldPredicate):
        return predicate.with_field_ref(_field_ref_for_predicate(predicate, unit=unit))
    if isinstance(predicate, QueryNotPredicate):
        return QueryNotPredicate(_bind_predicate_context(predicate.child, unit=unit))
    if isinstance(predicate, QueryBoolPredicate):
        return QueryBoolPredicate(
            predicate.op,
            tuple(_bind_predicate_context(child, unit=unit) for child in predicate.children),
        )
    if isinstance(predicate, QueryExistsPredicate):
        return QueryExistsPredicate(
            predicate.unit,
            _bind_predicate_context(predicate.child, unit=predicate.unit),
        )
    if isinstance(predicate, QuerySequencePredicate):
        return QuerySequencePredicate(
            steps=tuple(_bind_predicate_context(step, unit="action") for step in predicate.steps),
            constraints=predicate.constraints,
        )
    return predicate


@v_args(inline=True)
class _BooleanQueryTransformer(Transformer[Token, QueryPredicate]):
    def boolean_query(self, *items: object) -> QueryPredicate:
        predicates = _predicate_children(items)
        if len(predicates) != 1:
            raise ExpressionCompileError("Boolean query did not produce exactly one predicate", field=None)
        return predicates[0]

    def session_prefix(self, *_items: object) -> object:
        return Token("SESSION_PREFIX", "sessions where")

    def or_expr(self, *items: object) -> QueryPredicate:
        return _merge_bool_children("or", _predicate_children(items))

    def and_expr(self, *items: object) -> QueryPredicate:
        return _merge_bool_children("and", _predicate_children(items))

    def not_expr(self, *_items: object) -> QueryPredicate:
        predicates = _predicate_children(_items)
        if len(predicates) != 1:
            raise ExpressionCompileError("NOT must wrap exactly one predicate", field=None)
        return QueryNotPredicate(predicates[0])

    def count_leaf(self, token: Token) -> QueryPredicate:
        return _count_token_to_predicate(_QUERY_TRANSFORMER.count_clause(token))

    def count_compare_leaf(self, field_name: Token, op: Token, number: Token) -> QueryPredicate:
        return _count_token_to_predicate(_normalize_count_comparison(str(field_name), str(op), str(number)))

    def count_between_leaf(
        self,
        field_name: Token,
        _between: Token,
        min_number: Token,
        _and: Token,
        max_number: Token,
    ) -> QueryPredicate:
        return _count_range_token_to_predicate(
            _normalize_count_range(str(field_name), str(min_number), str(max_number))
        )

    def date_compare_leaf(self, _field_name: Token, op: Token, value: Token) -> QueryPredicate:
        return _date_token_to_predicate(_normalize_date_comparison(str(op), str(value)))

    def date_between_leaf(
        self,
        _field_name: Token,
        _between: Token,
        min_value: Token,
        _and: Token,
        max_value: Token,
    ) -> QueryPredicate:
        return _date_range_token_to_predicate(_normalize_date_range(str(min_value), str(max_value)))

    def time_compare_leaf(self, _field_name: Token, op: Token, value: Token) -> QueryPredicate:
        return _normalize_time_comparison(str(op), str(value))

    def time_between_leaf(
        self,
        _field_name: Token,
        _between: Token,
        min_value: Token,
        _and: Token,
        max_value: Token,
    ) -> QueryPredicate:
        return _normalize_time_range(str(min_value), str(max_value))

    def field_leaf(self, token: Token) -> QueryPredicate:
        return _field_token_to_predicate(_QUERY_TRANSFORMER.field_clause(token))

    def fts_quoted_leaf(self, token: Token) -> QueryPredicate:
        return QueryTextPredicate(text=_decode_escaped_string(Token("ESCAPED_STRING", str(token)[1:])))

    def fts_bare_leaf(self, token: Token) -> QueryPredicate:
        value = str(token)[1:].strip()
        if not value:
            raise ExpressionCompileError("FTS predicate requires text", field=None)
        return QueryTextPredicate(text=value)

    def semantic_quoted_leaf(self, token: Token) -> QueryPredicate:
        raw = str(token)
        quoted = raw[len("near:text:") :] if raw.lower().startswith("near:text:") else raw[len("semantic:") :]
        return QuerySemanticPredicate(text=_decode_escaped_string(Token("ESCAPED_STRING", quoted)))

    def semantic_bare_leaf(self, token: Token) -> QueryPredicate:
        raw = str(token)
        value = raw[len("near:text:") :] if raw.lower().startswith("near:text:") else raw[len("semantic:") :]
        value = value.strip()
        if not value:
            raise ExpressionCompileError("semantic predicate requires text", field=None)
        return QuerySemanticPredicate(text=value)

    def exists_leaf(self, _exists: Token, unit: Token, child: QueryPredicate) -> QueryPredicate:
        unit_value = str(unit).lower()
        if unit_value not in set(structural_query_units()):
            raise ExpressionCompileError(f"unsupported structural query unit {unit_value!r}", field=None)
        return QueryExistsPredicate(unit=cast(QueryExistsUnit, unit_value), child=child)

    def sequence_step(self, *items: object) -> QueryPredicate:
        predicates = _predicate_children(items)
        if not predicates:
            raise ExpressionCompileError("seq() steps require at least one action predicate", field=None)
        if len(predicates) == 1:
            return predicates[0]
        return QueryBoolPredicate(op="and", children=predicates)

    def sequence_link(self, _arrow: Token, constraint: Token | None = None) -> QuerySequenceConstraint:
        if constraint is None:
            return QuerySequenceConstraint()
        value = str(constraint)[1:-1].lower()
        if value == "next":
            return QuerySequenceConstraint(kind="next")
        amount_text, unit = value.removeprefix("within:")[:-1], value[-1]
        if value.endswith("ms"):
            amount_text, unit = value.removeprefix("within:")[:-2], "ms"
        multiplier = {"ms": 1, "s": 1_000, "m": 60_000, "h": 3_600_000, "d": 86_400_000}[unit]
        within_ms = int(amount_text) * multiplier
        if within_ms <= 0:
            raise ExpressionCompileError("seq within duration must be greater than zero", field=None)
        return QuerySequenceConstraint(kind="within", within_ms=within_ms)

    def sequence_leaf(self, _seq: Token, *items: object) -> QueryPredicate:
        steps = _predicate_children(items)
        constraints = tuple(item for item in items if isinstance(item, QuerySequenceConstraint))
        if len(steps) < 2:
            raise ExpressionCompileError("seq() requires at least two action steps", field=None)
        return QuerySequencePredicate(steps=steps, constraints=constraints)


_BOOLEAN_QUERY_TRANSFORMER = _BooleanQueryTransformer()


def _is_boolean_expression(expression: str) -> bool:
    stripped = expression.lstrip()
    lower = stripped.lower()
    exists_prefixes = tuple(
        prefix for unit in structural_query_units() for prefix in (f"exists {unit}(", f"exists {unit} (")
    )
    if (
        stripped.startswith("(")
        or lower.startswith("sessions where ")
        or lower.startswith(exists_prefixes)
        or lower.startswith("seq(")
        or lower.startswith("seq (")
        or lower.startswith("lineage:")
    ):
        return True
    for source, _unit in terminal_query_source_pairs():
        marker = f"{source} where"
        if lower == marker or (lower.startswith(marker) and lower[len(marker)].isspace()):
            return True
    if "~" in expression:
        return True
    if re.search(r"\b(?:semantic|near:text):", expression, re.IGNORECASE):
        return True
    if re.search(rf"\b(?:{_COUNT_FIELD_REGEX})\b\s*(?:>=|<=|=|>|<|between\b)", expression, re.IGNORECASE):
        count_range_masked = re.sub(r"\bbetween\s+\d+\s+and\s+\d+\b", "between_range", expression, flags=re.IGNORECASE)
        return bool(re.search(r"\b(?:and|or|not)\b", count_range_masked, re.IGNORECASE))
    if re.search(r"\bdate\b\s*(?:>=|<=|>|<|between\b)", expression, re.IGNORECASE):
        date_range_masked = re.sub(r"\bbetween\s+\S+\s+and\s+\S+\b", "between_range", expression, flags=re.IGNORECASE)
        return bool(re.search(r"\b(?:and|or|not)\b", date_range_masked, re.IGNORECASE))
    return ":" in expression and bool(re.search(r"\b(?:and|or|not)\b", expression, re.IGNORECASE))


def _source_where_unit(source: str) -> QueryUnitName:
    unit = terminal_query_unit(source.strip().lower())
    if unit is not None:
        return unit
    raise ExpressionCompileError(f"unsupported query source {source!r}", field=None)


def _transform_boolean_predicate(expression: str) -> QueryPredicate:
    try:
        tree = _QUERY_PARSER.parse(expression, start="boolean_query")
    except UnexpectedInput as exc:
        raise ExpressionCompileError(
            _parse_error_message(expression, exc), field=_parse_error_field(expression)
        ) from exc
    try:
        transformed = _BOOLEAN_QUERY_TRANSFORMER.transform(tree)
    except VisitError as exc:
        if isinstance(exc.orig_exc, ExpressionCompileError):
            raise exc.orig_exc from exc
        raise
    if not isinstance(
        transformed,
        QueryFieldPredicate
        | QueryBoolPredicate
        | QueryNotPredicate
        | QueryExistsPredicate
        | QueryLineagePredicate
        | QuerySequencePredicate
        | QueryTextPredicate
        | QuerySemanticPredicate,
    ):
        raise ExpressionCompileError("Boolean query did not produce a predicate", field=None)
    return transformed


def _parse_boolean_predicate(expression: str) -> QueryPredicate:
    transformed = _transform_boolean_predicate(expression)
    return _bind_predicate_context(transformed, unit="session")


def _split_pipeline_stages(expression: str) -> tuple[str, ...]:
    """Split ``|`` stages outside quotes and parentheses.

    Values such as ``origin:(codex-session|claude-code-session)`` must not
    split.
    """

    stages: list[str] = []
    stage_start = 0
    in_quote = False
    escaped = False
    depth = 0
    for idx, char in enumerate(expression):
        if escaped:
            escaped = False
            continue
        if char == "\\" and in_quote:
            escaped = True
            continue
        if char == '"':
            in_quote = not in_quote
            continue
        if in_quote:
            continue
        if char == "(":
            depth += 1
            continue
        if char == ")":
            depth = max(0, depth - 1)
            continue
        if char == "|" and depth == 0:
            stage = expression[stage_start:idx].strip()
            if not stage:
                raise ExpressionCompileError("pipeline requires non-empty stages around '|'", field=None)
            stages.append(stage)
            stage_start = idx + 1
    final = expression[stage_start:].strip()
    if stages and not final:
        raise ExpressionCompileError("pipeline requires non-empty stages around '|'", field=None)
    if stages:
        stages.append(final)
    return tuple(stages)


#: Canonical query units wired for the ``with <units>`` projection clause.
#: The clause is unit-agnostic by construction (the fetch/attach path drives off
#: the descriptor registry), but only units listed here are validated as
#: executable projections. Adding a new unit is a one-line change once its
#: session-scoping fetch is confirmed (see ``attached_units.py``).
WITH_PROJECTION_SUPPORTED_UNITS = PROJECTION_QUERY_UNITS


def _iter_top_level_with_positions(expression: str) -> list[int]:
    """Return start indices of every top-level ``with`` keyword token.

    "Top-level" means outside double quotes and outside any paren/bracket/brace
    nesting, so a quoted FTS phrase such as ``"deploy with caveats"`` or an
    in-field alternation never registers a ``with`` boundary.
    """

    positions: list[int] = []
    in_quote = False
    escaped = False
    depth = 0
    n = len(expression)
    for idx, char in enumerate(expression):
        if escaped:
            escaped = False
            continue
        if char == "\\" and in_quote:
            escaped = True
            continue
        if char == '"':
            in_quote = not in_quote
            continue
        if in_quote:
            continue
        if char in "([{":
            depth += 1
            continue
        if char in ")]}":
            depth = max(0, depth - 1)
            continue
        if depth != 0:
            continue
        if char in "wW" and expression[idx : idx + 4].lower() == "with":
            prev_char = expression[idx - 1] if idx > 0 else " "
            next_idx = idx + 4
            next_char = expression[next_idx] if next_idx < n else " "
            if (idx == 0 or prev_char.isspace()) and (next_idx >= n or next_char.isspace()):
                positions.append(idx)
    return positions


def _split_projection_items(tail: str) -> tuple[str, ...]:
    items: list[str] = []
    start = 0
    depth = 0
    for idx, char in enumerate(tail):
        if char == "(":
            depth += 1
            continue
        if char == ")":
            depth = max(0, depth - 1)
            continue
        if char == "," and depth == 0:
            item = tail[start:idx].strip()
            if not item:
                raise ExpressionCompileError(
                    "with clause units must be a comma-separated list of query units (for example: with assertions)",
                    field=None,
                )
            items.append(item)
            start = idx + 1
    final = tail[start:].strip()
    if not final:
        raise ExpressionCompileError(
            "with clause units must be a comma-separated list of query units (for example: with assertions)",
            field=None,
        )
    items.append(final)
    return tuple(items)


def _parse_projection_item(item: str) -> tuple[str, tuple[str, ...]]:
    match = re.fullmatch(r"(?P<unit>[A-Za-z0-9_.-]+)(?:\((?P<fields>[^()]*)\))?", item)
    if match is None:
        raise ExpressionCompileError(
            f"invalid with clause item {item!r}; use unit or unit(field, field)",
            field=None,
        )
    raw_fields = match.group("fields")
    if raw_fields is None:
        return match.group("unit"), ()
    fields = tuple(field.strip() for field in raw_fields.split(","))
    if any(not field for field in fields):
        raise ExpressionCompileError(
            f"with {match.group('unit')} field selection must be a comma-separated list of payload fields",
            field=None,
        )
    return match.group("unit"), fields


def _canonicalize_with_projection(tail: str) -> tuple[tuple[str, ...], dict[str, tuple[str, ...]]]:
    """Validate and canonicalize the ``with`` clause unit list.

    ``tail`` is the text after the ``with`` keyword (for example
    ``"assertions"`` or ``"messages(message_id,text), actions(tool_name)"``).
    Every token must resolve to a known query unit and be wired for projection; otherwise an
    :class:`ExpressionCompileError` is raised so an unknown or unsupported unit
    never silently no-ops.
    """

    canonical: list[str] = []
    seen: set[str] = set()
    field_map: dict[str, tuple[str, ...]] = {}
    for item in _split_projection_items(tail):
        token, fields = _parse_projection_item(item)
        descriptor = query_unit_descriptor(token)
        if descriptor is None:
            supported = ", ".join(sorted(WITH_PROJECTION_SUPPORTED_UNITS))
            raise ExpressionCompileError(
                f"unknown query unit {token!r} in with clause; supported units: {supported}",
                field=None,
            )
        if descriptor.unit not in WITH_PROJECTION_SUPPORTED_UNITS:
            supported = ", ".join(sorted(WITH_PROJECTION_SUPPORTED_UNITS))
            raise ExpressionCompileError(
                f"with {token}: projection is not yet supported; supported units: {supported}",
                field=None,
            )
        if descriptor.unit not in seen:
            seen.add(descriptor.unit)
            canonical.append(descriptor.unit)
        if fields:
            if descriptor.unit in field_map:
                merged = (
                    *field_map[descriptor.unit],
                    *(field for field in fields if field not in field_map[descriptor.unit]),
                )
                field_map[descriptor.unit] = merged
            else:
                field_map[descriptor.unit] = fields
    return tuple(canonical), field_map


def _canonicalize_with_units(tail: str) -> tuple[str, ...]:
    units, _fields = _canonicalize_with_projection(tail)
    return units


def _split_with_projection_clause(expression: str) -> tuple[str, tuple[str, ...], dict[str, tuple[str, ...]]]:
    """Split a trailing ``with <unit>[, <unit>]`` projection clause.

    Returns ``(head, units, fields)`` where ``head`` is the selection
    expression with the clause removed, ``units`` are canonical query-unit
    names, and ``fields`` maps units to payload-field allowlists. When no
    top-level ``with`` keyword is present, returns ``(expression, (), {})``.

    The split happens outside the Lark grammar — mirroring
    :func:`_split_pipeline_stages` — so quoting and nesting are honored and a
    quoted FTS term containing ``with`` is never mis-split.
    """

    positions = _iter_top_level_with_positions(expression)
    if not positions:
        return expression, (), {}
    split_at = positions[-1]
    tail = expression[split_at + 4 :].strip()
    if not tail:
        # A bare trailing ``with`` is not a projection clause; leave it intact.
        return expression, (), {}
    units, fields = _canonicalize_with_projection(tail)
    head = expression[:split_at].strip()
    if not head:
        raise ExpressionCompileError(
            "with clause requires a selection expression before it (for example: repo:polylogue with assertions)",
            field=None,
        )
    return head, units, fields


def _split_with_clause(expression: str) -> tuple[str, tuple[str, ...]]:
    head, units, _fields = _split_with_projection_clause(expression)
    return head, units


#: Public alias for the trailing ``with <units>`` projection-clause splitter,
#: used by surface adapters (CLI raw path) that strip the clause from FTS text.
split_with_clause = _split_with_clause
split_with_projection_clause = _split_with_projection_clause


def _session_source_predicate(stage: str) -> QueryPredicate:
    lower = stage.lower()
    marker = "sessions where"
    if lower == marker:
        raise ExpressionCompileError("sessions where requires a predicate before pipeline '|'", field=None)
    if not lower.startswith(marker) or not lower[len(marker)].isspace():
        raise ExpressionCompileError(
            "pipeline queries currently start with `sessions where ...`",
            field=None,
        )
    return _parse_boolean_predicate(stage)


def _scope_session_predicate(predicate: QueryPredicate) -> QueryPredicate:
    if isinstance(predicate, QueryFieldPredicate):
        if predicate.field.startswith("session."):
            return predicate
        if predicate.field not in _BOOLEAN_SUPPORTED_FIELDS:
            raise ExpressionCompileError(
                f"session pipeline stage cannot scope unsupported field {predicate.field!r}",
                field=predicate.field,
            )
        scoped_field = _canonical_session_alias(predicate.field)
        return QueryFieldPredicate(field=f"session.{scoped_field}", values=predicate.values, op=predicate.op)
    if isinstance(predicate, QueryBoolPredicate):
        return QueryBoolPredicate(predicate.op, tuple(_scope_session_predicate(child) for child in predicate.children))
    if isinstance(predicate, QueryNotPredicate):
        return QueryNotPredicate(_scope_session_predicate(predicate.child))
    if isinstance(
        predicate, QueryTextPredicate | QueryExistsPredicate | QueryLineagePredicate | QuerySequencePredicate
    ):
        return predicate
    raise ExpressionCompileError(
        "pipeline session stages currently support SQL-backed Boolean session predicates only; "
        "semantic stages need a ranked terminal lowerer.",
        field=None,
    )


def _bind_scoped_session_predicate(predicate: QueryPredicate, *, terminal_unit: QueryUnitName) -> QueryPredicate:
    scoped = _scope_session_predicate(predicate)
    if isinstance(scoped, QueryFieldPredicate):
        return _bind_predicate_context(scoped, unit=terminal_unit)
    if isinstance(scoped, QueryBoolPredicate):
        return QueryBoolPredicate(
            scoped.op,
            tuple(_bind_scoped_session_predicate(child, terminal_unit=terminal_unit) for child in scoped.children),
        )
    if isinstance(scoped, QueryNotPredicate):
        return QueryNotPredicate(_bind_scoped_session_predicate(scoped.child, terminal_unit=terminal_unit))
    return _bind_predicate_context(scoped, unit="session")


def _parse_non_negative_int_stage(stage: str, keyword: str) -> int | None:
    lower = stage.lower()
    prefix = f"{keyword} "
    if not lower.startswith(prefix):
        return None
    raw_value = stage[len(prefix) :].strip()
    if not raw_value:
        raise ExpressionCompileError(f"pipeline `{keyword}` stage requires an integer", field=keyword)
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ExpressionCompileError(f"pipeline `{keyword}` stage requires an integer", field=keyword) from exc
    if value < 0:
        raise ExpressionCompileError(f"pipeline `{keyword}` stage must be non-negative", field=keyword)
    return value


def _parse_sort_stage(stage: str) -> QueryUnitSort | None:
    lower = stage.lower()
    if not lower.startswith("sort by "):
        return None
    parts = lower.removeprefix("sort by ").split()
    if not parts:
        raise ExpressionCompileError("pipeline `sort by` stage requires a field", field="sort")
    if parts[0] not in {"time", "count", "key"}:
        raise ExpressionCompileError(
            "pipeline `sort by` supports `time` for rows and `count` or `key` for aggregate rows",
            field="sort",
        )
    if len(parts) == 1:
        return QueryUnitSort(field=cast(Literal["time", "count", "key"], parts[0]))
    if len(parts) == 2 and parts[1] in {"asc", "desc"}:
        return QueryUnitSort(
            field=cast(Literal["time", "count", "key"], parts[0]),
            direction=cast(Literal["asc", "desc"], parts[1]),
        )
    raise ExpressionCompileError(
        "pipeline `sort by` accepts an optional `asc` or `desc` direction",
        field="sort",
    )


def _parse_group_stage(stage: str) -> str | None:
    normalized = " ".join(stage.split()).lower()
    if not normalized.startswith("group by "):
        return None
    field = normalized.removeprefix("group by ").strip()
    if not field:
        raise ExpressionCompileError("pipeline `group by` requires a field", field="group")
    if field in {"origin", "repo"}:
        return f"session.{field}"
    return field


def _parse_count_stage(stage: str) -> bool:
    return " ".join(stage.split()).lower() == "count"


def _validate_aggregate_group_field(unit: QueryUnitName, field: str) -> None:
    descriptor = query_unit_descriptor(unit)
    supported_fields = frozenset(descriptor.aggregate_group_fields) if descriptor is not None else frozenset()
    if field not in supported_fields:
        supported = ", ".join(sorted(supported_fields))
        if not supported:
            raise ExpressionCompileError(
                f"pipeline `group by` is not supported for {unit} rows yet",
                field="group",
            )
        raise ExpressionCompileError(
            f"pipeline `group by {field}` is not supported for {unit} rows; supported fields: {supported}",
            field=field,
        )


def _query_unit_lowerer_kind(unit: QueryUnitName) -> QueryUnitLowererKind:
    descriptor = query_unit_descriptor(unit)
    if descriptor is None or not descriptor.terminal_supported:
        raise ExpressionCompileError(f"unsupported terminal query unit {unit!r}", field=None)
    return descriptor.lowerer_kind


def _ensure_sql_row_pipeline_lowerer(unit: QueryUnitName, *, stage: str) -> None:
    if _query_unit_lowerer_kind(unit) != "sql":
        raise ExpressionCompileError(
            f"pipeline `{stage}` currently supports SQL-backed terminal rows; "
            "runtime-transform units need a streaming lowerer first",
            field=stage.partition(" ")[0],
        )


def _ensure_sql_aggregate_pipeline_lowerer(unit: QueryUnitName, *, stage: str) -> None:
    if _query_unit_lowerer_kind(unit) != "sql":
        raise ExpressionCompileError(
            f"pipeline `{stage}` currently supports SQL-backed terminal rows; "
            "runtime-transform units need an aggregate lowerer first",
            field=stage.partition(" ")[0],
        )


def _ensure_aggregate_lowerer_supported(unit: QueryUnitName, *, stage: str) -> None:
    """Reject aggregate (``count``) on SQL units without an aggregate lowerer.

    ``run`` / ``observed-event`` / ``context-snapshot`` are SQL-backed terminal
    units that do not yet advertise aggregate group fields (#2006), so aggregate
    pipelines must fail typed/narrow instead of reaching an unwired
    ``query_unit_counts`` path.
    """
    descriptor = query_unit_descriptor(unit)
    if descriptor is None or not descriptor.aggregate_group_fields:
        raise ExpressionCompileError(
            f"pipeline `{stage}` is not supported for {unit} rows; this terminal unit has no aggregate lowerer",
            field=stage.partition(" ")[0],
        )


def _apply_pipeline_stage(source: QueryUnitSource, stage: str) -> QueryUnitSource:
    sort = _parse_sort_stage(stage)
    if sort is not None:
        if sort.field == "time":
            if source.aggregate is not None:
                raise ExpressionCompileError(
                    "pipeline `sort by time` must appear before aggregate stages", field="sort"
                )
            if source.group_by is not None:
                raise ExpressionCompileError("pipeline `sort by time` must appear before `group by`", field="sort")
            _ensure_sql_row_pipeline_lowerer(source.unit, stage="sort by time")
        elif source.aggregate != "count":
            raise ExpressionCompileError(
                "pipeline `sort by count` and `sort by key` require an aggregate `count` stage",
                field="sort",
            )
        elif source.limit is not None or source.offset is not None:
            raise ExpressionCompileError(
                "pipeline aggregate `sort by` must appear before `limit` and `offset`",
                field="sort",
            )
        _ensure_sql_aggregate_pipeline_lowerer(source.unit, stage=f"sort by {sort.field}")
        return replace(
            source,
            sort=sort,
            pipeline_stages=(
                *source.pipeline_stages,
                QueryUnitSortStage(sort),
            ),
        )
    group_by = _parse_group_stage(stage)
    if group_by is not None:
        if source.aggregate is not None:
            raise ExpressionCompileError("pipeline `group by` must appear before `count`", field="group")
        if source.sort is not None:
            raise ExpressionCompileError("pipeline `sort by time` cannot feed aggregate stages", field="group")
        if source.limit is not None or source.offset is not None:
            raise ExpressionCompileError("pipeline `group by` must appear before `limit` and `offset`", field="group")
        _ensure_sql_aggregate_pipeline_lowerer(source.unit, stage="group by")
        _ensure_aggregate_lowerer_supported(source.unit, stage="group by")
        _validate_aggregate_group_field(source.unit, group_by)
        return replace(
            source,
            group_by=group_by,
            pipeline_stages=(
                *source.pipeline_stages,
                QueryUnitGroupStage(group_by),
            ),
        )
    if _parse_count_stage(stage):
        if source.limit is not None or source.offset is not None:
            raise ExpressionCompileError("pipeline `count` must appear before `limit` and `offset`", field="count")
        _ensure_sql_aggregate_pipeline_lowerer(source.unit, stage="count")
        _ensure_aggregate_lowerer_supported(source.unit, stage="count")
        return replace(
            source,
            aggregate="count",
            pipeline_stages=(
                *source.pipeline_stages,
                QueryUnitCountStage(),
            ),
        )
    limit = _parse_non_negative_int_stage(stage, "limit")
    if limit is not None:
        if limit == 0:
            raise ExpressionCompileError("pipeline `limit` stage must be greater than zero", field="limit")
        return replace(
            source,
            limit=limit,
            pipeline_stages=(
                *source.pipeline_stages,
                QueryUnitLimitStage(limit),
            ),
        )
    offset = _parse_non_negative_int_stage(stage, "offset")
    if offset is not None:
        return replace(
            source,
            offset=offset,
            pipeline_stages=(
                *source.pipeline_stages,
                QueryUnitOffsetStage(offset),
            ),
        )
    raise ExpressionCompileError(
        f"unsupported pipeline stage {stage!r}; supported terminal stages are "
        "`sort by time|count|key [asc|desc]`, `group by FIELD`, `count`, `limit N`, and `offset N`",
        field=None,
    )


def _parse_plain_unit_source_expression(expression: str) -> QueryUnitSource | None:
    lower = expression.lower()
    source_match: tuple[str, QueryUnitName, str] | None = None
    for source, unit in terminal_query_source_pairs():
        marker = f"{source} where"
        if lower == marker:
            source_match = (marker, unit, "")
            break
        if lower.startswith(marker) and lower[len(marker)].isspace():
            source_match = (marker, unit, expression[len(marker) :].strip())
            break
    if source_match is None:
        return None
    _marker, unit, inner = source_match
    if not inner:
        raise ExpressionCompileError(f"{unit}s where requires a predicate", field=None)
    transformed = _transform_boolean_predicate(inner)
    bound = _bind_predicate_context(transformed, unit=unit)
    return QueryUnitSource(unit=unit, predicate=bound)


#: Shape-only prefixes for terminal pipeline stage keywords (``sort by``,
#: ``group by``, ``limit``, ``offset``). ``count`` has no trailing argument so
#: it is matched by exact equality instead. Used only to *recognize* that a
#: stage has the shape of a terminal pipeline action -- never to validate its
#: contents, so this never raises.
_PIPELINE_STAGE_KEYWORD_PREFIXES: tuple[str, ...] = ("sort by ", "group by ", "limit ", "offset ")


def _pipeline_stage_keyword(stage: str) -> str | None:
    """Return the terminal pipeline stage keyword ``stage`` looks like, if any.

    Distinguishes "this stage has the shape of `count`/`group by`/`sort by`/
    `limit`/`offset` but was misapplied" from "this stage is unrecognized
    entirely", so callers can raise a specific, actionable error instead of a
    generic one for the former.
    """

    normalized = " ".join(stage.split()).lower()
    if normalized == "count":
        return "count"
    for prefix in _PIPELINE_STAGE_KEYWORD_PREFIXES:
        if normalized.startswith(prefix):
            return prefix.strip()
    return None


def _parse_pipeline_unit_source(expression: str) -> QueryUnitSource | None:
    stages = _split_pipeline_stages(expression)
    if not stages:
        return None
    if len(stages) < 2:
        return None

    first_stage, second_stage, *remaining_stages = stages
    first_lower = first_stage.lower()
    if first_lower.startswith("sessions where"):
        session_predicate = _session_source_predicate(first_stage)
        terminal_source = _parse_plain_unit_source_expression(second_stage)
        if terminal_source is None:
            keyword = _pipeline_stage_keyword(second_stage)
            if keyword is not None:
                raise ExpressionCompileError(
                    f"pipeline `{keyword}` cannot follow `sessions where ...` directly; "
                    "`sessions` has no terminal row/aggregate lowerer of its own, only a session-scoping "
                    "stage. To count matching sessions, use `find <predicate> then analyze --count` instead "
                    "of a pipeline `| count`. To count/group/sort/limit rows of a specific terminal unit "
                    "scoped to these sessions, pipe into an executable `<unit>s where ...` terminal stage "
                    "first, e.g. `sessions where ... | actions where ... | group by FIELD | count`. "
                    f"Supported terminal units: {terminal_query_source_list()}.",
                    field=None,
                )
            raise ExpressionCompileError(
                "pipeline terminal stage must be an executable `<unit>s where ...` query "
                f"(got {second_stage!r}); supported terminal units: {terminal_query_source_list()}.",
                field=None,
            )
        scoped_session_predicate = _bind_scoped_session_predicate(session_predicate, terminal_unit=terminal_source.unit)
        bound = QueryBoolPredicate("and", (scoped_session_predicate, terminal_source.predicate))
        source = QueryUnitSource(
            unit=terminal_source.unit,
            predicate=bound,
            session_predicate=session_predicate,
            limit=terminal_source.limit,
            offset=terminal_source.offset,
            sort=terminal_source.sort,
            group_by=terminal_source.group_by,
            aggregate=terminal_source.aggregate,
            pipeline_stages=(
                QueryUnitSessionScopeStage(session_predicate),
                *terminal_source.pipeline_stages,
            ),
        )
        tail_stages = tuple(remaining_stages)
    else:
        direct_source = _parse_plain_unit_source_expression(first_stage)
        if direct_source is None:
            raise ExpressionCompileError(
                "pipeline queries must start with `sessions where ...` or an executable `<unit>s where ...` query",
                field=None,
            )
        source = direct_source
        tail_stages = (second_stage, *remaining_stages)

    for stage in tail_stages:
        source = _apply_pipeline_stage(source, stage)
    return source


def parse_unit_source_expression(expression: str) -> QueryUnitSource | None:
    """Parse explicit unit-source ``... where ...`` expressions as terminal row sources.

    ``compile_expression`` still lowers these forms to session selectors for
    existing surfaces. Terminal query execution uses this helper to preserve
    the selected unit and return row-level results.
    """

    stripped = expression.strip()
    pipeline_source = _parse_pipeline_unit_source(stripped)
    if pipeline_source is not None:
        return pipeline_source
    return _parse_plain_unit_source_expression(stripped)


def _terminal_only_unit_source_error(unit: QueryUnitName) -> ExpressionCompileError:
    descriptor = query_unit_descriptor(unit)
    source = descriptor.plural_source if descriptor is not None else f"{unit}s"
    return ExpressionCompileError(
        f"{source} where expressions return terminal {unit} rows; "
        "use query_units / /api/query-units or a CLI terminal-unit query instead of a session selector",
        field=None,
    )


def _query_unit_uses_runtime_transform(unit: QueryUnitName) -> bool:
    descriptor = query_unit_descriptor(unit)
    return descriptor is not None and descriptor.lowerer_kind == "runtime_transform"


def _parse_source_where_predicate(expression: str) -> QueryExistsPredicate | None:
    source = parse_unit_source_expression(expression)
    if source is None:
        return None
    if (
        source.session_predicate is not None
        or source.limit is not None
        or source.offset is not None
        or source.sort is not None
        or source.group_by is not None
        or source.aggregate is not None
        or source.pipeline_stages
    ):
        raise ExpressionCompileError(
            "pipeline unit queries return terminal rows; use query_units / /api/query-units or a CLI terminal-unit query",
            field=None,
        )
    if _query_unit_uses_runtime_transform(source.unit):
        raise _terminal_only_unit_source_error(source.unit)
    return QueryExistsPredicate(unit=cast(Any, source.unit), child=source.predicate)


def _contains_semantic_predicate(predicate: QueryPredicate) -> bool:
    if isinstance(predicate, QuerySemanticPredicate):
        return True
    if isinstance(predicate, QueryNotPredicate):
        return _contains_semantic_predicate(predicate.child)
    if isinstance(predicate, QueryBoolPredicate):
        return any(_contains_semantic_predicate(child) for child in predicate.children)
    if isinstance(predicate, QueryExistsPredicate):
        return _contains_semantic_predicate(predicate.child)
    return False


def _merge_semantic_seed(existing: str | None, new_value: str) -> str:
    if existing is not None:
        raise ExpressionCompileError(
            "only one semantic predicate is supported per query until the ranked planner supports multiple vector legs",
            field=None,
        )
    return new_value


def _extract_semantic_seed(predicate: QueryPredicate) -> tuple[str | None, QueryPredicate | None]:
    """Lift a positive top-level semantic predicate into ``SessionQuerySpec.similar_text``.

    Semantic vector search is rank-producing, not a simple SQL truth predicate.
    The currently executable shape is therefore ``semantic:"text" AND <filters>``:
    the vector leg produces ranked message candidates and the residual predicate
    filters those candidates through existing archive lowerers.
    """

    if isinstance(predicate, QuerySemanticPredicate):
        return predicate.text, None
    if isinstance(predicate, QueryNotPredicate):
        if _contains_semantic_predicate(predicate.child):
            raise ExpressionCompileError("semantic predicates are not supported under NOT", field=None)
        return None, predicate
    if isinstance(predicate, QueryExistsPredicate):
        if _contains_semantic_predicate(predicate.child):
            raise ExpressionCompileError("semantic predicates are only supported at session scope", field=None)
        return None, predicate
    if isinstance(predicate, QueryBoolPredicate):
        if predicate.op != "and":
            if _contains_semantic_predicate(predicate):
                raise ExpressionCompileError("semantic predicates are not supported under OR yet", field=None)
            return None, predicate
        semantic_text: str | None = None
        residual_children: list[QueryPredicate] = []
        for child in predicate.children:
            child_semantic, residual = _extract_semantic_seed(child)
            if child_semantic is not None:
                semantic_text = _merge_semantic_seed(semantic_text, child_semantic)
            if residual is not None:
                residual_children.append(residual)
        if not residual_children:
            return semantic_text, None
        if len(residual_children) == 1:
            return semantic_text, residual_children[0]
        return semantic_text, QueryBoolPredicate("and", tuple(residual_children))
    return None, predicate


def parse_expression_ast(expression: str) -> QueryExpressionAST:
    """Parse a query expression into the typed AST without lowering it.

    A trailing ``with <units>`` projection clause is stripped before parsing;
    the AST describes the selection only. Callers that need the projected units
    use :func:`compile_expression`.
    """
    expression = expression.strip()
    if not expression:
        return QueryExpressionAST(())
    if not (expression.startswith("{") or expression.startswith("[")):
        expression, _with_units, _with_unit_fields = _split_with_projection_clause(expression)
        if not expression:
            return QueryExpressionAST(())
    source_where = _parse_source_where_predicate(expression)
    if source_where is not None:
        return QueryExpressionAST((), boolean_predicate=source_where)
    if _is_boolean_expression(expression):
        return QueryExpressionAST((), boolean_predicate=_parse_boolean_predicate(expression))
    try:
        tree = _QUERY_PARSER.parse(expression, start="compact_query")
    except UnexpectedInput as exc:
        raise ExpressionCompileError(
            _parse_error_message(expression, exc), field=_parse_error_field(expression)
        ) from exc
    try:
        transformed = _QUERY_TRANSFORMER.transform(tree)
    except VisitError as exc:
        if isinstance(exc.orig_exc, ExpressionCompileError):
            raise exc.orig_exc from exc
        raise
    if not isinstance(transformed, QueryExpressionAST):
        raise ExpressionCompileError("query expression did not produce an AST", field=None)
    return transformed


def _explain_clause(token: _LexToken) -> QueryExpressionExplainClause:
    if isinstance(token, _FieldToken):
        return QueryExpressionExplainClause(
            kind="field",
            field=token.field,
            value=token.raw_value,
            negated=token.negated,
        )
    if isinstance(token, _CountToken):
        return QueryExpressionExplainClause(
            kind="count",
            field=token.field,
            op=token.op,
            number=token.number,
        )
    if isinstance(token, _CountRangeToken):
        return QueryExpressionExplainClause(
            kind="count_range",
            field=token.field,
            min_number=token.min_number,
            max_number=token.max_number,
        )
    if isinstance(token, _DateComparisonToken):
        return QueryExpressionExplainClause(
            kind="date",
            field="date",
            op=token.op,
            value=token.value,
        )
    if isinstance(token, _DateRangeToken):
        return QueryExpressionExplainClause(
            kind="date_range",
            field="date",
            min_value=token.min_value,
            max_value=token.max_value,
        )
    if isinstance(token, _TextToken):
        return QueryExpressionExplainClause(
            kind="text",
            value=token.text,
            negated=token.negated,
            quoted=token.quoted,
        )
    return QueryExpressionExplainClause(kind="json", value=token.raw)


def _predicate_units(predicate: QueryPredicate) -> set[str]:
    units = {"session"}
    if isinstance(predicate, QueryExistsPredicate):
        units.add(predicate.unit)
        units.update(_predicate_units(predicate.child))
    elif isinstance(predicate, QueryBoolPredicate):
        for child in predicate.children:
            units.update(_predicate_units(child))
    elif isinstance(predicate, QueryNotPredicate):
        units.update(_predicate_units(predicate.child))
    elif isinstance(predicate, QuerySequencePredicate):
        units.add("action")
    elif isinstance(predicate, QueryLineagePredicate):
        units.add("lineage")
    return units


def _predicate_execution_legs(predicate: QueryPredicate) -> set[str]:
    if isinstance(predicate, QueryFieldPredicate):
        return {"sql"}
    if isinstance(predicate, QueryTextPredicate):
        return {"fts"}
    if isinstance(predicate, QuerySemanticPredicate):
        return {"vector"}
    if isinstance(predicate, QueryLineagePredicate):
        return {"lineage-recursive-cte"}
    if isinstance(predicate, QuerySequencePredicate):
        return {"sequence-action"}
    if isinstance(predicate, QueryExistsPredicate):
        return {f"exists-{predicate.unit}", *_predicate_execution_legs(predicate.child)}
    if isinstance(predicate, QueryNotPredicate):
        return _predicate_execution_legs(predicate.child)
    if isinstance(predicate, QueryBoolPredicate):
        legs: set[str] = set()
        for child in predicate.children:
            legs.update(_predicate_execution_legs(child))
        return legs
    return set()


def _spec_execution_legs(spec: SessionQuerySpec) -> set[str]:
    legs: set[str] = set()
    if spec.query_terms:
        legs.add("fts")
    if spec.similar_text or spec.similar_session_id:
        legs.add("vector")
    if spec.boolean_predicate is not None:
        legs.update(_predicate_execution_legs(spec.boolean_predicate))
    if spec.to_plan().describe():
        legs.add("sql")
    return legs


def _explain_selected_units(ast: QueryExpressionAST) -> tuple[str, ...]:
    if ast.boolean_predicate is None:
        return ("session",)
    return tuple(sorted(_predicate_units(ast.boolean_predicate)))


def _explain_execution_legs(spec: SessionQuerySpec) -> tuple[str, ...]:
    legs = _spec_execution_legs(spec)
    return tuple(sorted(legs))


def _explain_unit_source_execution_legs(source: QueryUnitSource) -> tuple[str, ...]:
    legs = {f"terminal-{source.unit}-rows", *_predicate_execution_legs(source.predicate)}
    if source.aggregate is not None:
        legs.add(f"terminal-{source.unit}-{source.aggregate}-aggregate")
    if _query_unit_uses_runtime_transform(source.unit):
        legs.add("runtime-transform")
    else:
        legs.add("sql")
    return tuple(sorted(legs))


def _ast_payload(
    *,
    entry: str,
    clauses: tuple[QueryExpressionExplainClause, ...] = (),
    predicate: QueryPredicate | None = None,
    unit_source: QueryUnitSource | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {"entry": entry}
    if clauses:
        payload["clauses"] = [clause.to_payload() for clause in clauses]
    if predicate is not None:
        payload["predicate"] = predicate.to_payload()
    if unit_source is not None:
        unit_payload: dict[str, object] = {
            "unit": unit_source.unit,
            "predicate": unit_source.predicate.to_payload(),
        }
        if unit_source.session_predicate is not None:
            unit_payload["session_predicate"] = unit_source.session_predicate.to_payload()
        if unit_source.limit is not None:
            unit_payload["limit"] = unit_source.limit
        if unit_source.offset is not None:
            unit_payload["offset"] = unit_source.offset
        if unit_source.sort is not None:
            unit_payload["sort"] = {
                "field": unit_source.sort.field,
                "direction": unit_source.sort.direction,
            }
        if unit_source.group_by is not None:
            unit_payload["group_by"] = unit_source.group_by
        if unit_source.aggregate is not None:
            unit_payload["aggregate"] = unit_source.aggregate
        if unit_source.pipeline_stages:
            unit_payload["pipeline_stages"] = [stage.to_payload() for stage in unit_source.pipeline_stages]
        unit_payload["pipeline"] = unit_source.pipeline.to_payload()
        payload["unit_source"] = unit_payload
    return payload


def _lowering_plan_payload(
    *,
    lowerer: str,
    selected_units: tuple[str, ...],
    execution_legs: tuple[str, ...],
    plan_description: tuple[str, ...],
    compatibility_selector: str | None = None,
    pipeline: QueryUnitPipeline | None = None,
    pipeline_stages: tuple[QueryUnitPipelineStage, ...] = (),
) -> dict[str, object]:
    payload: dict[str, object] = {
        "lowerer": lowerer,
        "selected_units": list(selected_units),
        "execution_legs": list(execution_legs),
        "plan_description": list(plan_description),
    }
    if compatibility_selector is not None:
        payload["compatibility_selector"] = compatibility_selector
    if pipeline is not None:
        payload["pipeline"] = pipeline.to_payload()
    if pipeline_stages:
        payload["pipeline_stages"] = [stage.to_payload() for stage in pipeline_stages]
    return payload


def explain_expression(expression: str) -> QueryExpressionExplanation:
    """Explain parser output, lowering path, and execution-plan descriptions."""
    source_text = expression
    stripped = expression.strip()
    clauses: tuple[QueryExpressionExplainClause, ...]
    selected_units: tuple[str, ...]
    execution_legs: tuple[str, ...]
    plan_description: tuple[str, ...]
    if stripped.startswith("{") or stripped.startswith("["):
        lowered = _compile_json_spec(stripped)
        clauses = (_explain_clause(_JsonToken(raw=stripped)),)
        selected_units = ("session",)
        execution_legs = _explain_execution_legs(lowered)
        plan_description = tuple(lowered.to_plan().describe())
        lowerer = "json-spec"
        return QueryExpressionExplanation(
            source_text=source_text,
            clauses=clauses,
            lowerer=lowerer,
            lowered_spec=lowered,
            selected_units=selected_units,
            execution_legs=execution_legs,
            plan_description=plan_description,
            ast=_ast_payload(entry="json", clauses=clauses),
            lowering_plan=_lowering_plan_payload(
                lowerer=lowerer,
                selected_units=selected_units,
                execution_legs=execution_legs,
                plan_description=plan_description,
            ),
        )
    unit_source = parse_unit_source_expression(stripped)
    if unit_source is not None:
        lowered = (
            SessionQuerySpec()
            if _query_unit_uses_runtime_transform(unit_source.unit)
            else SessionQuerySpec(
                boolean_predicate=QueryExistsPredicate(unit=cast(Any, unit_source.unit), child=unit_source.predicate)
            )
        )
        selected_units = (unit_source.unit,)
        execution_legs = _explain_unit_source_execution_legs(unit_source)
        plan_description = (f"terminal unit source: {unit_source.unit}",)
        compatibility_selector = None
        if not _query_unit_uses_runtime_transform(unit_source.unit):
            compatibility_selector = f"exists {unit_source.unit}(...)"
            plan_description = (*plan_description, f"compatibility session selector: {compatibility_selector}")
        lowerer = "lark-query-unit-source-to-terminal-unit"
        return QueryExpressionExplanation(
            source_text=source_text,
            clauses=(),
            predicate=unit_source.predicate,
            lowerer=lowerer,
            lowered_spec=lowered,
            selected_units=selected_units,
            execution_legs=execution_legs,
            plan_description=plan_description,
            ast=_ast_payload(entry="unit_source", predicate=unit_source.predicate, unit_source=unit_source),
            lowering_plan=_lowering_plan_payload(
                lowerer=lowerer,
                selected_units=selected_units,
                execution_legs=execution_legs,
                plan_description=plan_description,
                compatibility_selector=compatibility_selector,
                pipeline=unit_source.pipeline,
                pipeline_stages=unit_source.pipeline_stages,
            ),
        )
    ast = parse_expression_ast(stripped)
    lowered = compile_expression(stripped)
    clauses = tuple(_explain_clause(clause) for clause in ast.clauses)
    selected_units = _explain_selected_units(ast)
    execution_legs = _explain_execution_legs(lowered)
    plan_description = tuple(lowered.to_plan().describe())
    lowerer = "lark-query-expression-to-session-query-spec"
    return QueryExpressionExplanation(
        source_text=source_text,
        clauses=clauses,
        predicate=ast.boolean_predicate,
        lowerer=lowerer,
        lowered_spec=lowered,
        selected_units=selected_units,
        execution_legs=execution_legs,
        plan_description=plan_description,
        ast=_ast_payload(
            entry="boolean" if ast.boolean_predicate is not None else "compact",
            clauses=clauses,
            predicate=ast.boolean_predicate,
        ),
        lowering_plan=_lowering_plan_payload(
            lowerer=lowerer,
            selected_units=selected_units,
            execution_legs=execution_legs,
            plan_description=plan_description,
        ),
    )


# ---------------------------------------------------------------------------
# Lowering
# ---------------------------------------------------------------------------


def _split_alternation(raw: str) -> tuple[str, ...]:
    """Split ``a|b|c`` into ``("a", "b", "c")``."""
    return tuple(part.strip() for part in raw.split("|") if part.strip())


def _split_assertion_value_path_alternation(raw: str) -> tuple[str, ...]:
    """Split value-path alternatives while preserving JSON string literals."""

    values: list[str] = []
    start = 0
    quoted = False
    escaped = False
    for index, character in enumerate(raw):
        if escaped:
            escaped = False
            continue
        if character == "\\" and quoted:
            escaped = True
            continue
        if character == '"':
            quoted = not quoted
            continue
        if character == "|" and not quoted:
            candidate = raw[start:index].strip()
            if candidate:
                values.append(candidate)
            start = index + 1
    candidate = raw[start:].strip()
    if candidate:
        values.append(candidate)
    return tuple(values)


_RELATIVE_DATE_RE = re.compile(r"^\d+[hdwm]$", re.IGNORECASE)


def _parse_relative_date(value: str) -> str:
    """Pass-through: spec stores dates as strings, resolved later by plan.py."""
    # Relative date formats understood by parse_date: 7d, 2w, 1h, 3m (months), etc.
    # We do a lightweight conversion for the compact form; absolute dates pass through.
    if _RELATIVE_DATE_RE.match(value):
        # Convert to dateparser-friendly form understood by polylogue.core.dates
        unit_map = {"h": "hours", "d": "days", "w": "weeks", "m": "months"}
        num_str = value[:-1]
        unit_char = value[-1].lower()
        return f"{num_str} {unit_map[unit_char]} ago"
    # Otherwise pass through (ISO date or natural language) and let parse_date validate
    return value


def _merge_tuples(existing: tuple[str, ...], new_values: tuple[str, ...]) -> tuple[str, ...]:
    return existing + new_values


@dataclass
class _SpecAccumulator:
    """Mutable accumulator for building a SessionQuerySpec from tokens."""

    query_terms: list[str] = field(default_factory=list)
    contains_terms: list[str] = field(default_factory=list)
    exclude_text_terms: list[str] = field(default_factory=list)
    retrieval_lane: str = "auto"
    referenced_path: list[str] = field(default_factory=list)
    cwd_prefix: str | None = None
    action_terms: list[str] = field(default_factory=list)
    excluded_action_terms: list[str] = field(default_factory=list)
    tool_terms: list[str] = field(default_factory=list)
    excluded_tool_terms: list[str] = field(default_factory=list)
    origins: list[str] = field(default_factory=list)
    excluded_origins: list[str] = field(default_factory=list)
    repo_names: list[str] = field(default_factory=list)
    project_refs: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    excluded_tags: list[str] = field(default_factory=list)
    has_types: list[str] = field(default_factory=list)
    filter_has_paste: bool = False
    filter_has_tool_use: bool = False
    filter_has_thinking: bool = False
    title: str | None = None
    session_id: str | None = None
    since: str | None = None
    until: str | None = None
    similar_text: str | None = None
    similar_session_id: str | None = None
    min_messages: int | None = None
    max_messages: int | None = None
    min_words: int | None = None
    max_words: int | None = None

    def apply_token(self, tok: _LexToken) -> None:
        """Apply one token to the accumulator."""
        if isinstance(tok, _TextToken):
            if tok.negated:
                self.exclude_text_terms.append(tok.text)
            else:
                self.query_terms.append(tok.text)
            return

        if isinstance(tok, _JsonToken):
            # JSON spec is handled separately in compile_expression
            return

        if isinstance(tok, _CountToken):
            if tok.field == "messages":
                if tok.op == ">":
                    self.min_messages = tok.number + 1
                elif tok.op == ">=":
                    self.min_messages = tok.number
                elif tok.op == "<":
                    if tok.number == 0:
                        raise ExpressionCompileError(
                            "messages < 0 is not representable as a non-negative count bound",
                            field="messages",
                        )
                    self.max_messages = tok.number - 1
                elif tok.op == "<=":
                    self.max_messages = tok.number
                else:
                    self.min_messages = tok.number
                    self.max_messages = tok.number
            elif tok.field == "words":
                if tok.op == ">":
                    self.min_words = tok.number + 1
                elif tok.op == ">=":
                    self.min_words = tok.number
                elif tok.op == "<":
                    if tok.number == 0:
                        raise ExpressionCompileError(
                            "words < 0 is not representable as a non-negative count bound",
                            field="words",
                        )
                    self.max_words = tok.number - 1
                elif tok.op == "<=":
                    self.max_words = tok.number
                else:  # "="
                    self.min_words = tok.number
                    self.max_words = tok.number
            else:
                raise ExpressionCompileError(
                    f"field {tok.field!r} is supported only inside `sessions where` Boolean queries",
                    field=tok.field,
                )
            return

        if isinstance(tok, _CountRangeToken):
            if tok.field == "messages":
                self.min_messages = tok.min_number
                self.max_messages = tok.max_number
            elif tok.field == "words":
                self.min_words = tok.min_number
                self.max_words = tok.max_number
            else:
                raise ExpressionCompileError(
                    f"field {tok.field!r} is supported only inside `sessions where` Boolean queries",
                    field=tok.field,
                )
            return

        if isinstance(tok, _DateComparisonToken):
            if tok.op in {">", ">="}:
                self.since = tok.value
            else:
                self.until = tok.value
            return

        if isinstance(tok, _DateRangeToken):
            self.since = tok.min_value
            self.until = tok.max_value
            return

        # _FieldToken
        assert isinstance(tok, _FieldToken)
        fname = tok.field
        values = _split_alternation(tok.raw_value) if tok.raw_value else ()

        if fname == "repo":
            if tok.negated:
                raise ExpressionCompileError("negation is not supported for 'repo'", field=fname)
            self.repo_names.extend(values)

        elif fname == "project":
            if tok.negated:
                raise ExpressionCompileError("negation is not supported for 'project'", field=fname)
            self.project_refs.extend(values)

        elif fname == "origin":
            _known_origins = {o.value for o in Origin}
            if tok.negated:
                for v in values:
                    if v not in _known_origins:
                        raise ExpressionCompileError(
                            f"unknown origin {v!r}; recognized: " + ", ".join(sorted(_known_origins)),
                            field="origin",
                        )
                    self.excluded_origins.append(v)
            else:
                for v in values:
                    if v not in _known_origins:
                        raise ExpressionCompileError(
                            f"unknown origin {v!r}; recognized: " + ", ".join(sorted(_known_origins)),
                            field="origin",
                        )
                    self.origins.append(v)

        elif fname == "tag":
            if tok.negated:
                self.excluded_tags.extend(values)
            else:
                self.tags.extend(values)

        elif fname == "path":
            if tok.negated:
                raise ExpressionCompileError("negation is not supported for 'path'", field=fname)
            self.referenced_path.extend(values)

        elif fname == "cwd":
            if tok.negated:
                raise ExpressionCompileError("negation is not supported for 'cwd'", field=fname)
            if values:
                self.cwd_prefix = values[-1]

        elif fname == "tool":
            normalized = [t.strip().lower() for t in values if t.strip()]
            if tok.negated:
                self.excluded_tool_terms.extend(normalized)
            else:
                self.tool_terms.extend(normalized)

        elif fname == "action":
            for v in values:
                candidate = v.strip().lower()
                if candidate not in QUERY_ACTION_TYPES:
                    raise ExpressionCompileError(
                        f"unknown action {v!r}; recognized: " + ", ".join(QUERY_ACTION_TYPES),
                        field="action",
                    )
                if tok.negated:
                    self.excluded_action_terms.append(candidate)
                else:
                    self.action_terms.append(candidate)

        elif fname == "has":
            if tok.negated:
                raise ExpressionCompileError("negation is not supported for 'has'", field=fname)
            for v in values:
                sub = v.strip().lower()
                if sub in _HAS_BOOL_MAP:
                    setattr(self, _HAS_BOOL_MAP[sub], True)
                else:
                    self.has_types.append(sub)

        elif fname in {"id", "session"}:
            if tok.negated:
                raise ExpressionCompileError(f"negation is not supported for {fname!r}", field=fname)
            if values:
                self.session_id = values[-1]

        elif fname == "title":
            if tok.negated:
                raise ExpressionCompileError("negation is not supported for 'title'", field=fname)
            self.title = " ".join(values)

        elif fname == "since":
            if tok.negated:
                raise ExpressionCompileError("negation is not supported for 'since'", field=fname)
            if values:
                self.since = _parse_relative_date(values[-1])

        elif fname == "until":
            if tok.negated:
                raise ExpressionCompileError("negation is not supported for 'until'", field=fname)
            if values:
                self.until = _parse_relative_date(values[-1])

        elif fname == "near":
            if tok.negated:
                raise ExpressionCompileError("negation is not supported for 'near'", field=fname)
            # Two seeding modes share the ``near:`` field:
            #   near:id:<ref>  → similar_session_id (vector similarity seeded by a
            #                    stored session's embeddings, distinct from text)
            #   near:"text"    → similar_text (vector similarity seeded by a string)
            # Conservative rule: a value is treated as a session reference ONLY
            # when it is a single token unambiguously prefixed with ``id:``.
            # Everything else (bare words, quoted phrases, alternations) stays on
            # the free-text leg, so a hash-like search phrase is never silently
            # reinterpreted as a session id.
            if len(values) == 1 and values[0].startswith("id:"):
                ref = values[0][len("id:") :].strip()
                if not ref:
                    raise ExpressionCompileError(
                        "near:id: requires a session reference, e.g. near:id:abc123",
                        field=fname,
                    )
                self.similar_session_id = ref
            else:
                self.similar_text = " ".join(values)

        elif fname == "contains":
            if tok.negated:
                raise ExpressionCompileError("negation is not supported for 'contains'", field=fname)
            self.contains_terms.extend(values)

        elif fname == "lane":
            if tok.negated:
                raise ExpressionCompileError("negation is not supported for 'lane'", field=fname)
            if values:
                try:
                    self.retrieval_lane = normalize_retrieval_lane(values[-1])
                except Exception as exc:
                    raise ExpressionCompileError(
                        f"unknown retrieval lane {values[-1]!r}; recognized: auto, dialogue, actions, hybrid",
                        field="lane",
                    ) from exc

        elif fname in COUNT_QUERY_FIELD_REGISTRY or fname in NUMERIC_QUERY_FIELD_REGISTRY:
            # Already handled via _CountToken; field:value form without op is an error
            raise ExpressionCompileError(
                f"use comparison operator for {fname!r}: e.g. {fname}:>=10",
                field=fname,
            )

        else:
            raise ExpressionCompileError(
                _unknown_query_field_message(fname),
                field=fname,
            )

    def to_spec(self) -> SessionQuerySpec:
        return SessionQuerySpec(
            query_terms=tuple(self.query_terms),
            contains_terms=tuple(self.contains_terms),
            exclude_text_terms=tuple(self.exclude_text_terms),
            retrieval_lane=self.retrieval_lane,
            referenced_path=tuple(self.referenced_path),
            cwd_prefix=self.cwd_prefix,
            action_terms=tuple(self.action_terms),
            excluded_action_terms=tuple(self.excluded_action_terms),
            tool_terms=tuple(self.tool_terms),
            excluded_tool_terms=tuple(self.excluded_tool_terms),
            origins=tuple(self.origins),
            excluded_origins=tuple(self.excluded_origins),
            repo_names=tuple(self.repo_names),
            project_refs=tuple(self.project_refs),
            tags=tuple(self.tags),
            excluded_tags=tuple(self.excluded_tags),
            has_types=tuple(self.has_types),
            filter_has_paste=self.filter_has_paste,
            filter_has_tool_use=self.filter_has_tool_use,
            filter_has_thinking=self.filter_has_thinking,
            title=self.title,
            session_id=self.session_id,
            since=self.since,
            until=self.until,
            similar_text=self.similar_text,
            similar_session_id=self.similar_session_id,
            min_messages=self.min_messages,
            max_messages=self.max_messages,
            min_words=self.min_words,
            max_words=self.max_words,
        )

    def merge_from_spec(self, other: SessionQuerySpec) -> None:
        """Merge fields from an existing spec (used by compile_expression_into)."""
        self.query_terms.extend(other.query_terms)
        self.contains_terms.extend(other.contains_terms)
        self.exclude_text_terms.extend(other.exclude_text_terms)
        if other.retrieval_lane != "auto":
            self.retrieval_lane = other.retrieval_lane
        self.referenced_path.extend(other.referenced_path)
        if other.cwd_prefix is not None:
            self.cwd_prefix = other.cwd_prefix
        self.action_terms.extend(other.action_terms)
        self.excluded_action_terms.extend(other.excluded_action_terms)
        self.tool_terms.extend(other.tool_terms)
        self.excluded_tool_terms.extend(other.excluded_tool_terms)
        self.origins.extend(other.origins)
        self.excluded_origins.extend(other.excluded_origins)
        self.repo_names.extend(other.repo_names)
        self.project_refs.extend(other.project_refs)
        self.tags.extend(other.tags)
        self.excluded_tags.extend(other.excluded_tags)
        self.has_types.extend(other.has_types)
        if other.filter_has_paste:
            self.filter_has_paste = True
        if other.filter_has_tool_use:
            self.filter_has_tool_use = True
        if other.filter_has_thinking:
            self.filter_has_thinking = True
        if other.title is not None:
            self.title = other.title
        if other.session_id is not None:
            self.session_id = other.session_id
        if other.since is not None:
            self.since = other.since
        if other.until is not None:
            self.until = other.until
        if other.similar_text is not None:
            self.similar_text = other.similar_text
        if other.similar_session_id is not None:
            self.similar_session_id = other.similar_session_id
        if other.min_messages is not None:
            self.min_messages = other.min_messages
        if other.max_messages is not None:
            self.max_messages = other.max_messages
        if other.min_words is not None:
            self.min_words = other.min_words
        if other.max_words is not None:
            self.max_words = other.max_words


# ---------------------------------------------------------------------------
# Public compile functions
# ---------------------------------------------------------------------------


def compile_expression(expression: str) -> SessionQuerySpec:
    """Compile a DSL query expression string into a :class:`SessionQuerySpec`.

    Args:
        expression: The query expression string (may be empty).

    Returns:
        A ``SessionQuerySpec`` representing the compiled query.

    Raises:
        ExpressionCompileError: On unknown fields, malformed Boolean syntax,
            unclosed strings, or invalid values.
    """
    expression = expression.strip()
    if not expression:
        return SessionQuerySpec()

    # Direct JSON spec shortcut ({...} or [...] — arrays fail with a typed error)
    if expression.startswith("{") or expression.startswith("["):
        return _compile_json_spec(expression)

    # Split a trailing ``with <units>`` projection clause off the selection
    # expression before parsing it (outside the Lark grammar, like ``|`` stages).
    expression, with_units, with_unit_fields = _split_with_projection_clause(expression)

    ast = parse_expression_ast(expression)
    if ast.boolean_predicate is not None:
        similar_text, residual_predicate = _extract_semantic_seed(ast.boolean_predicate)
        return SessionQuerySpec(
            similar_text=similar_text,
            boolean_predicate=residual_predicate,
            with_units=with_units,
            with_unit_fields=with_unit_fields,
        )
    tokens = list(ast.clauses)

    # Check for JSON tokens (should only appear alone; mixed is an error)
    json_tokens = [t for t in tokens if isinstance(t, _JsonToken)]
    if json_tokens:
        if len(tokens) == 1:
            return _compile_json_spec(json_tokens[0].raw)
        raise ExpressionCompileError(
            "JSON spec must appear alone; cannot mix with other clauses",
            field=None,
        )

    # In compact query mode, bare OR is ambiguous text. Explicit Boolean
    # queries are supported through field clauses or ``sessions where ...``.
    word_texts = [t.text.upper() for t in tokens if isinstance(t, _TextToken) and not t.quoted]
    if "OR" in word_texts:
        raise ExpressionCompileError(
            "bare OR is ambiguous in compact text queries; use explicit Boolean syntax "
            "(for example: sessions where repo:polylogue OR origin:chatgpt-export), "
            'quote it as "OR", or use field:(a|b|c) for in-field alternatives.',
            field=None,
        )

    acc = _SpecAccumulator()
    for tok in tokens:
        acc.apply_token(tok)
    spec = acc.to_spec()
    if with_units:
        spec = replace(spec, with_units=with_units, with_unit_fields=with_unit_fields)
    return spec


def compile_expression_into(expression: str, base: SessionQuerySpec) -> SessionQuerySpec:
    """Merge a DSL expression into an existing :class:`SessionQuerySpec`.

    Tuple fields (query_terms, origins, tags, …) are extended additively.
    Scalar fields (session_id, title, since, until, …) are overridden only
    when the expression provides a non-None value.  Boolean flags are OR'd.

    This is useful when a surface pre-populates a spec from flags and then
    the user also types a DSL expression.

    Args:
        expression: DSL expression string (may be empty).
        base: Existing spec to merge into.

    Returns:
        A new ``SessionQuerySpec`` with the merged result.

    Raises:
        ExpressionCompileError: Same conditions as :func:`compile_expression`.
    """
    expr_spec = compile_expression(expression)
    boolean_predicate = expr_spec.boolean_predicate
    if base.boolean_predicate is not None and boolean_predicate is not None:
        boolean_predicate = QueryBoolPredicate("and", (base.boolean_predicate, boolean_predicate))
    elif boolean_predicate is None:
        boolean_predicate = base.boolean_predicate
    acc = _SpecAccumulator()
    acc.merge_from_spec(base)
    acc.merge_from_spec(expr_spec)
    # Preserve non-query fields from base that accumulator doesn't track
    merged = acc.to_spec()
    return replace(
        merged,
        sort=expr_spec.sort if expr_spec.sort is not None else base.sort,
        reverse=base.reverse or expr_spec.reverse,
        limit=expr_spec.limit if expr_spec.limit is not None else base.limit,
        sample=expr_spec.sample if expr_spec.sample is not None else base.sample,
        latest=base.latest or expr_spec.latest,
        typed_only=base.typed_only or expr_spec.typed_only,
        action_sequence=base.action_sequence or expr_spec.action_sequence,
        action_text_terms=base.action_text_terms + expr_spec.action_text_terms,
        message_type=expr_spec.message_type if expr_spec.message_type is not None else base.message_type,
        since_session_id=expr_spec.since_session_id
        if expr_spec.since_session_id is not None
        else base.since_session_id,
        offset=base.offset if base.offset else expr_spec.offset,
        cursor=expr_spec.cursor if expr_spec.cursor is not None else base.cursor,
        boolean_predicate=boolean_predicate,
        with_units=base.with_units + tuple(u for u in expr_spec.with_units if u not in base.with_units),
        with_unit_fields={**base.with_unit_fields, **expr_spec.with_unit_fields},
    )


def _compile_json_spec(raw: str) -> SessionQuerySpec:
    """Validate and compile a raw JSON string into a ``SessionQuerySpec``."""
    try:
        data: Any = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ExpressionCompileError(f"invalid JSON spec: {exc}", field=None) from exc
    if not isinstance(data, dict):
        raise ExpressionCompileError("JSON spec must be a JSON object", field=None)
    try:
        return SessionQuerySpec.from_params(data, strict=True)
    except Exception as exc:
        raise ExpressionCompileError(f"invalid spec fields: {exc}", field=None) from exc


def build_session_terminal_pipeline(
    action: str,
    *,
    args: tuple[tuple[str, object], ...] = (),
    predicate: QueryPredicate | None = None,
) -> SessionQueryPipeline:
    """Build a typed session-source pipeline for a root query terminal verb."""
    return SessionQueryPipeline(
        predicate=predicate,
        terminal=SessionQueryTerminalStage(action=_session_terminal_action(action), args=args),
    )


# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------


__all__ = [
    "compile_expression",
    "compile_expression_into",
    "build_session_terminal_pipeline",
    "CountQueryFieldInfo",
    "COUNT_QUERY_FIELD_REGISTRY",
    "count_query_fields",
    "count_query_operators",
    "DateQueryFieldInfo",
    "DATE_QUERY_FIELD_REGISTRY",
    "date_query_fields",
    "date_query_operators",
    "explain_expression",
    "ExpressionCompileError",
    "EXPRESSION_FIELD_REGISTRY",
    "SESSION_SOURCE_UNIT",
    "SESSION_TERMINAL_ACTIONS",
    "StructuralQueryUnitInfo",
    "STRUCTURAL_QUERY_UNIT_REGISTRY",
    "QueryExpressionExplainClause",
    "QueryExpressionExplanation",
    "QueryExpressionAST",
    "QueryUnitPipeline",
    "QueryUnitCountStage",
    "QueryUnitGroupStage",
    "QueryUnitLimitStage",
    "QueryUnitOffsetStage",
    "QueryUnitPipelineStage",
    "QueryUnitSessionScopeStage",
    "QueryUnitSortStage",
    "QueryUnitSort",
    "QueryUnitSource",
    "QueryUnitTerminalAction",
    "QueryUnitTerminalStage",
    "QueryUnitTransformStage",
    "SessionQueryPipeline",
    "SessionQueryTerminalStage",
    "SessionTerminalAction",
    "_HAS_BOOL_MAP",
    "parse_unit_source_expression",
    "parse_expression_ast",
    "split_with_clause",
    "split_with_projection_clause",
    "WITH_PROJECTION_SUPPORTED_UNITS",
    "structural_query_fields",
    "structural_query_units",
    "UnsupportedSessionTerminalActionError",
]
