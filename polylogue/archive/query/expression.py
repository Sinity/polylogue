"""Query expression parser/lowerer: DSL string → :class:`SessionQuerySpec`.

This module is the shared front-door for the query-expression language used
by all Polylogue read surfaces (CLI bare-query path, Python facade, and any
future surfaces that wire in).

Current executable grammar
--------------------------
The executable language is the Lark-backed query grammar in this module. It
currently accepts the compact session query form:

    repo:polylogue since:7d "json envelope"
    origin:(codex-session|claude-code-session) has:paste
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

Unknown fields and unsupported structured forms fail loudly. The Lark grammar
in this module is the query grammar. Compact field/text clauses and explicit
Boolean predicates are two entry shapes in that grammar, not separate languages
or a preserved legacy compiler.

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
    Parse a DSL string into a :class:`SessionQuerySpec`.

:func:`compile_expression_into`
    Merge a DSL string into an existing :class:`SessionQuerySpec`, additive
    (tuple fields are extended, scalar fields are overridden if set).

:class:`ExpressionCompileError`
    Typed compile-time error with a ``field`` attribute (``None`` for
    structural errors such as malformed Boolean syntax).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace
from difflib import get_close_matches
from typing import Any, Literal, cast

from lark import Lark, Token, Transformer, v_args
from lark.exceptions import UnexpectedInput, VisitError

from polylogue.archive.query.predicate import (
    QueryBoolPredicate,
    QueryExistsPredicate,
    QueryFieldPredicate,
    QueryLineagePredicate,
    QueryNotPredicate,
    QueryPredicate,
    QuerySemanticPredicate,
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


# ---------------------------------------------------------------------------
# Field registry (single source of truth for completion tools)
# ---------------------------------------------------------------------------

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
    """``messages:>=10`` or ``words:<=50``."""

    field: str  # "messages" or "words"
    op: Literal[">=", "<=", "="]
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

    op: Literal[">=", "<="]
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


ExplainClauseKind = Literal["field", "count", "count_range", "date", "date_range", "text", "json"]


@dataclass(frozen=True)
class QueryExpressionExplainClause:
    """Serializable clause view for parser/lowerer diagnostics."""

    kind: ExplainClauseKind
    field: str | None = None
    value: str | None = None
    negated: bool = False
    quoted: bool = False
    op: Literal[">=", "<=", "="] | None = None
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

    def to_payload(self) -> dict[str, object]:
        return {
            "source_text": self.source_text,
            "clauses": [clause.to_payload() for clause in self.clauses],
            "predicate": self.predicate.to_payload() if self.predicate is not None else None,
            "lowerer": self.lowerer,
            "selected_units": list(self.selected_units),
            "execution_legs": list(self.execution_legs),
            "plan_description": list(self.plan_description),
            "unsupported_nodes": list(self.unsupported_nodes),
        }


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

_QUERY_GRAMMAR = r"""
    flat_query: flat_clause*

    ?flat_clause: COUNT_CLAUSE       -> count_clause
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
        | SEQ "(" sequence_step (ARROW sequence_step)+ ")" -> sequence_leaf
        | SEMANTIC_QUOTED_TEXT             -> semantic_quoted_leaf
        | SEMANTIC_BARE_TEXT               -> semantic_bare_leaf
        | FTS_QUOTED_TEXT                  -> fts_quoted_leaf
        | FTS_BARE_TEXT                    -> fts_bare_leaf
        | COUNT_CLAUSE                     -> count_leaf
        | COUNT_FIELD BETWEEN INT AND INT  -> count_between_leaf
        | COUNT_FIELD COMP_OP INT          -> count_compare_leaf
        | DATE_FIELD BETWEEN DATE_VALUE AND DATE_VALUE -> date_between_leaf
        | DATE_FIELD DATE_COMP_OP DATE_VALUE -> date_compare_leaf
        | FIELD_CLAUSE                     -> field_leaf
        | "(" expr ")"
    sequence_step: FIELD_CLAUSE

    SESSIONS: /sessions/i
    WHERE: /where/i
    EXISTS: /exists/i
    SEQ: /seq/i
    ARROW: "->"
    STRUCT_UNIT: /(message|action|block)/i
    OR: /or/i
    AND: /and/i
    NOT: /not/i
    BETWEEN.6: /between/i
    COUNT_FIELD.6: /(messages|words)/i
    DATE_FIELD.6: /date/i
    DATE_COMP_OP: ">=" | "<=" | "=" | ">" | "<"
    COMP_OP: ">=" | "<=" | "=" | ">" | "<"
    DATE_VALUE: /[^\s"()]+/
    SEMANTIC_QUOTED_TEXT.7: /(?:semantic|near:text):"(\\.|[^"\\])*"/i
    SEMANTIC_BARE_TEXT.6: /(?:semantic|near:text):[^\s"()]+/i
    FTS_QUOTED_TEXT.6: /~"(\\.|[^"\\])*"/
    FTS_BARE_TEXT.5: /~[^\s"()]+/
    COUNT_CLAUSE.8: /(messages|words):(>=|<=|=)\d+(?!\S)/
    FIELD_CLAUSE.4: /-?[a-zA-Z_][a-zA-Z0-9_]*:(?:"(\\.|[^"\\])*"|\([^)]*\)|[^\s"()\[\]{}]+)/
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
    start=["flat_query", "boolean_query"],
    maybe_placeholders=False,
)

_COUNT_CLAUSE_RE = re.compile(r"^(messages|words):(>=|<=|=)(\d+)$")
_FIELD_CLAUSE_RE = re.compile(
    r"""
    ^(-?)
    ([a-zA-Z_][a-zA-Z0-9_]*)
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

_BOOLEAN_TRIGGER_RE = re.compile(r"^\s*\(|^\s*sessions\s+where\b|\b(?:and|or|not)\b", re.IGNORECASE)
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
        fields=tuple(sorted(_ACTION_STRUCTURAL_FIELDS)),
        example="exists action(tool:bash AND text:pytest)",
    ),
    "block": StructuralQueryUnitInfo(
        description="Match sessions with at least one parsed message block satisfying the child predicate.",
        fields=tuple(sorted(_BLOCK_STRUCTURAL_FIELDS)),
        example="exists block(type:code AND text:timeout)",
    ),
    "message": StructuralQueryUnitInfo(
        description="Match sessions with at least one message satisfying the child predicate.",
        fields=tuple(sorted(_MESSAGE_STRUCTURAL_FIELDS)),
        example="exists message(role:assistant AND text:timeout)",
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
    malformed_count = re.search(r"\b(messages|words):", expression, re.IGNORECASE)
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
    malformed_count = re.search(r"\b(messages|words):", expression, re.IGNORECASE)
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
    if op_text == ">":
        return _CountToken(field=field, op=">=", number=number + 1)
    if op_text == "<":
        if number == 0:
            raise ExpressionCompileError(f"{field} < 0 is not representable as a non-negative count bound", field=field)
        return _CountToken(field=field, op="<=", number=number - 1)
    op: Literal[">=", "<=", "="] = ">=" if op_text == ">=" else "<=" if op_text == "<=" else "="
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
    if op_text in {">=", ">"}:
        return _DateComparisonToken(op=">=", value=_parse_relative_date(value_text))
    if op_text in {"<=", "<"}:
        return _DateComparisonToken(op="<=", value=_parse_relative_date(value_text))
    raise ExpressionCompileError("date equality is not supported; use date between A and B", field="date")


def _normalize_date_range(min_text: str, max_text: str) -> _DateRangeToken:
    return _DateRangeToken(min_value=_parse_relative_date(min_text), max_value=_parse_relative_date(max_text))


@v_args(inline=True)
class _QueryTransformer(Transformer[Token, _LexToken | str | QueryExpressionAST]):
    def flat_query(self, *clauses: _LexToken) -> QueryExpressionAST:
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
        if raw_value.startswith('"'):
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


def _field_token_to_predicate(token: _FieldToken) -> QueryPredicate:
    field_name = token.field
    if field_name not in EXPRESSION_FIELD_REGISTRY and field_name not in _STRUCTURAL_BOOLEAN_SUPPORTED_FIELDS:
        raise ExpressionCompileError(
            _unknown_query_field_message(field_name, include_structural=True),
            field=field_name,
        )
    if field_name not in _BOOLEAN_SUPPORTED_FIELDS and field_name not in _STRUCTURAL_BOOLEAN_SUPPORTED_FIELDS:
        raise ExpressionCompileError(
            f"field {field_name!r} is not supported inside Boolean SQL predicates yet",
            field=field_name,
        )

    values = _split_alternation(token.raw_value) if token.raw_value else ()
    if field_name == "origin":
        known_origins = {o.value for o in Origin}
        for value in values:
            if value not in known_origins:
                raise ExpressionCompileError(
                    f"unknown origin {value!r}; recognized: " + ", ".join(sorted(known_origins)),
                    field="origin",
                )
    elif field_name == "action":
        for value in values:
            candidate = value.strip().lower()
            if candidate not in QUERY_ACTION_TYPES:
                raise ExpressionCompileError(
                    f"unknown action {value!r}; recognized: " + ", ".join(QUERY_ACTION_TYPES),
                    field="action",
                )
        values = tuple(value.strip().lower() for value in values if value.strip())
    elif field_name in {"tool", "has", "role", "type", "command", "path", "output"}:
        values = tuple(value.strip().lower() for value in values if value.strip())
    elif field_name in {"since", "until"} and values:
        values = (_parse_relative_date(values[-1]),)
    elif field_name == "lineage":
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


def _validate_predicate_context(
    predicate: QueryPredicate, *, unit: Literal["session", "message", "action", "block"]
) -> None:
    if isinstance(predicate, QueryFieldPredicate):
        if unit == "session":
            supported = _BOOLEAN_SUPPORTED_FIELDS
        elif unit == "message":
            supported = _MESSAGE_STRUCTURAL_FIELDS
        elif unit == "action":
            supported = _ACTION_STRUCTURAL_FIELDS
        else:
            supported = _BLOCK_STRUCTURAL_FIELDS
        if predicate.field not in supported:
            raise ExpressionCompileError(
                f"field {predicate.field!r} is not supported for {unit} predicates",
                field=predicate.field,
            )
        if (
            predicate.field in {"role", "type", "text", "tool", "action", "command", "path", "output"}
            and not predicate.values
        ):
            raise ExpressionCompileError(f"field {predicate.field!r} requires a value", field=predicate.field)
        if predicate.field == "date":
            if not predicate.values:
                raise ExpressionCompileError("field 'date' requires a value", field="date")
            if predicate.op not in {">=", "<="}:
                raise ExpressionCompileError("field 'date' supports only >=, <=, >, <, and between", field="date")
        if predicate.field == "words":
            if not predicate.values:
                raise ExpressionCompileError("field 'words' requires a numeric value", field="words")
            try:
                int(predicate.values[-1])
            except ValueError as exc:
                raise ExpressionCompileError("field 'words' requires a numeric value", field="words") from exc
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
        if unit != "session":
            raise ExpressionCompileError("lineage predicates are only supported from sessions", field=None)
        if not predicate.seed_session_id.strip():
            raise ExpressionCompileError("lineage predicate requires a session id", field="lineage")
        return
    if isinstance(predicate, QuerySequencePredicate):
        if unit != "session":
            raise ExpressionCompileError("seq predicates are only supported from sessions", field=None)
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
        if unit_value not in {"message", "action", "block"}:
            raise ExpressionCompileError(f"unsupported structural query unit {unit_value!r}", field=None)
        return QueryExistsPredicate(unit=cast(Literal["message", "action", "block"], unit_value), child=child)

    def sequence_step(self, token: Token) -> QueryPredicate:
        predicate = _field_token_to_predicate(_QUERY_TRANSFORMER.field_clause(token))
        if not isinstance(predicate, QueryFieldPredicate) or predicate.field != "action" or predicate.op != "=":
            raise ExpressionCompileError("seq() currently supports action:<kind> steps", field=None)
        if len(predicate.values) != 1:
            raise ExpressionCompileError("seq() action steps must name exactly one action", field="action")
        return predicate

    def sequence_leaf(self, _seq: Token, *items: object) -> QueryPredicate:
        steps = [item for item in items if isinstance(item, QueryFieldPredicate)]
        if len(steps) < 2:
            raise ExpressionCompileError("seq() requires at least two action steps", field=None)
        return QuerySequencePredicate(action_terms=tuple(step.values[0] for step in steps))


_BOOLEAN_QUERY_TRANSFORMER = _BooleanQueryTransformer()


def _is_boolean_expression(expression: str) -> bool:
    if re.search(
        r"^\s*\(|^\s*sessions\s+where\b|^\s*exists\s+(?:message|action|block)\s*\(|^\s*seq\s*\(|^\s*lineage:",
        expression,
        re.IGNORECASE,
    ):
        return True
    if "~" in expression:
        return True
    if re.search(r"\b(?:semantic|near:text):", expression, re.IGNORECASE):
        return True
    if re.search(r"\b(?:messages|words)\b\s*(?:>=|<=|=|>|<|between\b)", expression, re.IGNORECASE):
        count_range_masked = re.sub(r"\bbetween\s+\d+\s+and\s+\d+\b", "between_range", expression, flags=re.IGNORECASE)
        return bool(re.search(r"\b(?:and|or|not)\b", count_range_masked, re.IGNORECASE))
    if re.search(r"\bdate\b\s*(?:>=|<=|>|<|between\b)", expression, re.IGNORECASE):
        date_range_masked = re.sub(r"\bbetween\s+\S+\s+and\s+\S+\b", "between_range", expression, flags=re.IGNORECASE)
        return bool(re.search(r"\b(?:and|or|not)\b", date_range_masked, re.IGNORECASE))
    return ":" in expression and bool(re.search(r"\b(?:and|or|not)\b", expression, re.IGNORECASE))


def _parse_boolean_predicate(expression: str) -> QueryPredicate:
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
    _validate_predicate_context(transformed, unit="session")
    return transformed


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
    """Parse a query expression into the typed AST without lowering it."""
    expression = expression.strip()
    if not expression:
        return QueryExpressionAST(())
    if _is_boolean_expression(expression):
        return QueryExpressionAST((), boolean_predicate=_parse_boolean_predicate(expression))
    try:
        tree = _QUERY_PARSER.parse(expression, start="flat_query")
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


def explain_expression(expression: str) -> QueryExpressionExplanation:
    """Explain parser output, lowering path, and execution-plan descriptions."""
    source_text = expression
    stripped = expression.strip()
    if stripped.startswith("{") or stripped.startswith("["):
        lowered = _compile_json_spec(stripped)
        return QueryExpressionExplanation(
            source_text=source_text,
            clauses=(_explain_clause(_JsonToken(raw=stripped)),),
            lowerer="json-spec",
            lowered_spec=lowered,
            selected_units=("session",),
            execution_legs=_explain_execution_legs(lowered),
            plan_description=tuple(lowered.to_plan().describe()),
        )
    ast = parse_expression_ast(stripped)
    lowered = compile_expression(stripped)
    return QueryExpressionExplanation(
        source_text=source_text,
        clauses=tuple(_explain_clause(clause) for clause in ast.clauses),
        predicate=ast.boolean_predicate,
        lowerer="lark-query-expression-to-session-query-spec",
        lowered_spec=lowered,
        selected_units=_explain_selected_units(ast),
        execution_legs=_explain_execution_legs(lowered),
        plan_description=tuple(lowered.to_plan().describe()),
    )


# ---------------------------------------------------------------------------
# Compiler
# ---------------------------------------------------------------------------


def _split_alternation(raw: str) -> tuple[str, ...]:
    """Split ``a|b|c`` into ``("a", "b", "c")``."""
    return tuple(part.strip() for part in raw.split("|") if part.strip())


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
                if tok.op == ">=":
                    self.min_messages = tok.number
                elif tok.op == "<=":
                    self.max_messages = tok.number
                else:
                    self.min_messages = tok.number
                    self.max_messages = tok.number
            elif tok.field == "words":
                if tok.op == ">=":
                    self.min_words = tok.number
                elif tok.op == "<=":
                    self.max_words = tok.number
                else:  # "="
                    self.min_words = tok.number
                    self.max_words = tok.number
            return

        if isinstance(tok, _CountRangeToken):
            if tok.field == "messages":
                self.min_messages = tok.min_number
                self.max_messages = tok.max_number
            elif tok.field == "words":
                self.min_words = tok.min_number
                self.max_words = tok.max_number
            return

        if isinstance(tok, _DateComparisonToken):
            if tok.op == ">=":
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

        elif fname == "id":
            if tok.negated:
                raise ExpressionCompileError("negation is not supported for 'id'", field=fname)
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

        elif fname in ("messages", "words"):
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

    ast = parse_expression_ast(expression)
    if ast.boolean_predicate is not None:
        similar_text, residual_predicate = _extract_semantic_seed(ast.boolean_predicate)
        return SessionQuerySpec(similar_text=similar_text, boolean_predicate=residual_predicate)
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
    return acc.to_spec()


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


# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------


__all__ = [
    "compile_expression",
    "compile_expression_into",
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
    "StructuralQueryUnitInfo",
    "STRUCTURAL_QUERY_UNIT_REGISTRY",
    "QueryExpressionExplainClause",
    "QueryExpressionExplanation",
    "QueryExpressionAST",
    "_HAS_BOOL_MAP",
    "parse_expression_ast",
    "structural_query_fields",
    "structural_query_units",
]
