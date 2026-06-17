"""Query expression parser/lowerer: DSL string → :class:`SessionQuerySpec`.

This module is the shared front-door for the query-expression language used
by all Polylogue read surfaces (CLI bare-query path, Python facade, and any
future surfaces that wire in).

Current executable grammar
--------------------------
An expression is a whitespace-separated sequence of *clauses*, all AND'd:

    repo:polylogue since:7d "json envelope"
    origin:(codex-session|claude-code-session) has:paste
    path:polylogue/cli tool:bash action:file_edit
    near:"semantic search prompt"
    -id:bad tag:review
    messages:>=10 words:>=200

Clause forms:
- ``field:value``          — single-value field clause
- ``field:(a|b|c)``        — multi-value field clause (OR *within* field)
- ``-field:value``         — negated field clause
- ``"quoted phrase"``      — FTS phrase term
- ``-"quoted phrase"``     — excluded FTS phrase term
- ``bare_word``            — FTS bare word
- ``messages:>=N``         — count comparison (min_messages)
- ``messages:<=N``         — count comparison (max_messages)
- ``words:>=N``            — count comparison (min_words)
- ``{...}``                — direct JSON spec (validated into SessionQuerySpec)

Cross-field OR and nested Boolean groups are **rejected loudly** until #2006
adds executable Boolean AST lowerers. Unknown fields also fail loudly. The
Lark grammar in this module is the query grammar; there is no separate
long-lived "floor grammar."

Field registry
--------------
:data:`EXPRESSION_FIELD_REGISTRY` maps every recognized DSL token to its spec
field name.  Completion tools (#1844) should introspect this single registry
rather than maintain a parallel list.

Public API
----------
:func:`compile_expression`
    Parse a DSL string into a :class:`SessionQuerySpec`.

:func:`compile_expression_into`
    Merge a DSL string into an existing :class:`SessionQuerySpec`, additive
    (tuple fields are extended, scalar fields are overridden if set).

:class:`ExpressionCompileError`
    Typed compile-time error with a ``field`` attribute (``None`` for
    structural errors such as cross-field OR).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from lark import Lark, Token, Transformer, v_args
from lark.exceptions import UnexpectedInput

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
               structural errors (cross-field OR, unclosed quotes, etc.).
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
class _TextToken:
    """A bare word or quoted phrase."""

    text: str
    quoted: bool
    negated: bool


@dataclass(frozen=True)
class _JsonToken:
    """A ``{...}`` raw JSON spec."""

    raw: str


_LexToken = _FieldToken | _CountToken | _TextToken | _JsonToken


@dataclass(frozen=True)
class QueryExpressionAST:
    """Parsed query expression before lowering into ``SessionQuerySpec``."""

    clauses: tuple[_LexToken, ...]


ExplainClauseKind = Literal["field", "count", "text", "json"]


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
        return payload


@dataclass(frozen=True)
class QueryExpressionExplanation:
    """Debug envelope for query parsing, lowering, and execution-plan selection."""

    source_text: str
    clauses: tuple[QueryExpressionExplainClause, ...]
    lowerer: str
    lowered_spec: SessionQuerySpec
    plan_description: tuple[str, ...]
    unsupported_nodes: tuple[str, ...] = ()

    def to_payload(self) -> dict[str, object]:
        return {
            "source_text": self.source_text,
            "clauses": [clause.to_payload() for clause in self.clauses],
            "lowerer": self.lowerer,
            "plan_description": list(self.plan_description),
            "unsupported_nodes": list(self.unsupported_nodes),
        }


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

_QUERY_GRAMMAR = r"""
    start: clause*

    ?clause: COUNT_CLAUSE       -> count_clause
        | FIELD_CLAUSE          -> field_clause
        | NEG_QUOTED_TEXT       -> neg_quoted_text
        | QUOTED_TEXT           -> quoted_text
        | NEG_BARE_TEXT         -> neg_bare_text
        | BARE_TEXT             -> bare_text

    COUNT_CLAUSE.5: /(messages|words):(>=|<=|=)\d+/
    FIELD_CLAUSE.4: /-?[a-zA-Z_][a-zA-Z0-9_]*:(?:"(\\.|[^"\\])*"|\([^)]*\)|[^\s"()\[\]{}]+)/
    NEG_QUOTED_TEXT.3: /-"(\\.|[^"\\])*"/
    QUOTED_TEXT.2: /"(\\.|[^"\\])*"/
    NEG_BARE_TEXT.1: /-[^\s"()\[\]{}]+/
    BARE_TEXT: /[^\s"()\[\]{}]+/

    %import common.WS
    %ignore WS
"""


_QUERY_PARSER = Lark(_QUERY_GRAMMAR, parser="lalr", maybe_placeholders=False)

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


def _decode_escaped_string(token: Token) -> str:
    try:
        decoded = json.loads(str(token))
    except json.JSONDecodeError as exc:
        raise ExpressionCompileError(f"invalid quoted string: {exc}", field=None) from exc
    if not isinstance(decoded, str):
        raise ExpressionCompileError("quoted value did not decode to a string", field=None)
    return decoded


def _parse_error_message(expression: str, exc: UnexpectedInput) -> str:
    if expression.lstrip().startswith("("):
        return (
            "cross-field OR parentheses are not supported; express as separate queries "
            "or use field:(a|b|c) for in-field OR. "
            "Boolean-tree queries are tracked in issue #2006."
        )
    if '"' in expression:
        return "unclosed quoted string"
    return f"invalid query expression near column {exc.column}"


@v_args(inline=True)
class _QueryTransformer(Transformer[Token, _LexToken | str | QueryExpressionAST]):
    def start(self, *clauses: _LexToken) -> QueryExpressionAST:
        return QueryExpressionAST(tuple(clauses))

    def count_clause(self, token: Token) -> _CountToken:
        matched = _COUNT_CLAUSE_RE.match(str(token))
        if matched is None:
            raise ExpressionCompileError(f"invalid count clause: {token}", field=None)
        op_text = matched.group(2)
        op: Literal[">=", "<=", "="] = ">=" if op_text == ">=" else "<=" if op_text == "<=" else "="
        return _CountToken(field=matched.group(1), op=op, number=int(matched.group(3)))

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


def parse_expression_ast(expression: str) -> QueryExpressionAST:
    """Parse a query expression into the typed AST without lowering it."""
    expression = expression.strip()
    if not expression:
        return QueryExpressionAST(())
    try:
        tree = _QUERY_PARSER.parse(expression)
    except UnexpectedInput as exc:
        raise ExpressionCompileError(_parse_error_message(expression, exc), field=None) from exc
    transformed = _QUERY_TRANSFORMER.transform(tree)
    if not isinstance(transformed, QueryExpressionAST):
        raise ExpressionCompileError("query expression did not produce an AST", field=None)
    return transformed


def _lex(expression: str) -> list[_LexToken]:
    """Project the canonical Lark AST into the current clause lowerer input."""
    return list(parse_expression_ast(expression).clauses)


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
    if isinstance(token, _TextToken):
        return QueryExpressionExplainClause(
            kind="text",
            value=token.text,
            negated=token.negated,
            quoted=token.quoted,
        )
    return QueryExpressionExplainClause(kind="json", value=token.raw)


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
            plan_description=tuple(lowered.to_plan().describe()),
        )
    ast = parse_expression_ast(stripped)
    lowered = compile_expression(stripped)
    return QueryExpressionExplanation(
        source_text=source_text,
        clauses=tuple(_explain_clause(clause) for clause in ast.clauses),
        lowerer="lark-query-expression-to-session-query-spec",
        lowered_spec=lowered,
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
            recognized = sorted(EXPRESSION_FIELD_REGISTRY)
            raise ExpressionCompileError(
                f"unknown query field {fname!r}; recognized fields: " + ", ".join(recognized),
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
        ExpressionCompileError: On unknown fields, cross-field OR, unclosed
            strings, or invalid values.
    """
    expression = expression.strip()
    if not expression:
        return SessionQuerySpec()

    # Direct JSON spec shortcut ({...} or [...] — arrays fail with a typed error)
    if expression.startswith("{") or expression.startswith("["):
        return _compile_json_spec(expression)

    ast = parse_expression_ast(expression)
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

    # Detect cross-field OR patterns: top-level OR keyword
    word_texts = [t.text.upper() for t in tokens if isinstance(t, _TextToken) and not t.quoted]
    if "OR" in word_texts:
        raise ExpressionCompileError(
            "cross-field OR is not supported in this version; express as separate queries "
            "or use field:(a|b|c) for in-field OR. "
            "Boolean-tree queries are tracked in issue #1812.",
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
    "explain_expression",
    "ExpressionCompileError",
    "EXPRESSION_FIELD_REGISTRY",
    "QueryExpressionExplainClause",
    "QueryExpressionExplanation",
    "QueryExpressionAST",
    "_HAS_BOOL_MAP",
    "parse_expression_ast",
]
