"""Query expression compiler: DSL string → :class:`SessionQuerySpec`.

This module is the shared front-door for the query-expression language used
by all Polylogue read surfaces (CLI bare-query path, Python facade, and any
future surfaces that wire in).

Grammar (flat-conjunction DSL)
--------------------------------
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

Cross-field OR (``(a origin:x OR origin:y)``) and nested parentheses across
different fields are **rejected loudly** — they require a boolean-tree spec
that is tracked in issue #1812.  Unknown fields also fail loudly.

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
        "spec_field": "has_types",
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
        "description": "Semantic similarity search (vector lane)",
        "spec_field": "similar_text",
        "negatable": "no",
        "example": 'near:"semantic search"',
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
# Lexer tokens
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

# ---------------------------------------------------------------------------
# Lexer
# ---------------------------------------------------------------------------

# Matches: optional leading -, field name, colon, value (bare, quoted, or parens group)
_FIELD_RE = re.compile(
    r"""
    ^(-?)                           # optional negation
    ([a-zA-Z_][a-zA-Z0-9_]*)       # field name
    :                               # separator
    (?:
        "([^"]*)"                   # quoted value  → group 3
        |
        \(([^)]*)\)                 # paren group   → group 4
        |
        ([^\s"()\[\]]+)             # bare value    → group 5
    )
    $
    """,
    re.VERBOSE,
)

_COUNT_RE = re.compile(r"^(messages|words):(>=|<=|=)(\d+)$")

_QUOTED_RE = re.compile(r'^(-?)"([^"]*)"$')

# JSON spec must start with { and be balanced; simple heuristic: starts with {
_JSON_START_RE = re.compile(r"^\{")


def _next_token(text: str) -> tuple[str, str]:
    """Extract the next token from *text*, respecting quoted and paren sub-strings.

    Returns ``(token, remainder)`` where *token* is the next whitespace-delimited
    chunk (with quoted/paren sections treated as opaque) and *remainder* is what
    follows.

    Examples::

        _next_token('near:"semantic search" rest')
        → ('near:"semantic search"', ' rest')

        _next_token('repo:polylogue since:7d')
        → ('repo:polylogue', ' since:7d')
    """
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch in (" ", "\t", "\n", "\r"):
            return text[:i], text[i:]
        if ch == '"':
            # Skip over a quoted sub-string
            i += 1
            while i < n and text[i] != '"':
                i += 1
            i += 1  # consume closing "
        elif ch == "(":
            # Skip over a paren group
            depth = 1
            i += 1
            while i < n and depth > 0:
                if text[i] == "(":
                    depth += 1
                elif text[i] == ")":
                    depth -= 1
                i += 1
        else:
            i += 1
    return text, ""


def _lex(expression: str) -> list[_LexToken]:
    """Tokenize an expression string into a flat list of tokens."""
    tokens: list[_LexToken] = []
    remaining = expression.strip()

    while remaining:
        remaining = remaining.lstrip()
        if not remaining:
            break

        # --- JSON spec ---
        if remaining.startswith("{"):
            depth = 0
            i = 0
            while i < len(remaining):
                ch = remaining[i]
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        tokens.append(_JsonToken(raw=remaining[: i + 1]))
                        remaining = remaining[i + 1 :]
                        break
                elif ch == '"':
                    i += 1
                    while i < len(remaining) and remaining[i] != '"':
                        if remaining[i] == "\\":
                            i += 1
                        i += 1
                i += 1
            else:
                raise ExpressionCompileError("unclosed '{' in JSON spec", field=None)
            continue

        # --- Quoted phrase (possibly negated) ---
        if remaining.startswith('"') or remaining.startswith('-"'):
            negated = remaining.startswith('-"')
            start = 2 if negated else 1
            i = start
            phrase_chars: list[str] = []
            found_close = False
            while i < len(remaining):
                ch = remaining[i]
                if ch == "\\":
                    i += 1
                    if i < len(remaining):
                        phrase_chars.append(remaining[i])
                elif ch == '"':
                    found_close = True
                    break
                else:
                    phrase_chars.append(ch)
                i += 1
            if not found_close:
                raise ExpressionCompileError("unclosed quoted string", field=None)
            phrase = "".join(phrase_chars)
            tokens.append(_TextToken(text=phrase, quoted=True, negated=negated))
            remaining = remaining[i + 1 :]
            continue

        # --- Cross-field OR detection: a bare `(` not attached to a field ---
        if remaining.startswith("("):
            raise ExpressionCompileError(
                "cross-field OR parentheses are not supported; express as separate queries "
                "or use field:(a|b|c) for in-field OR. "
                "Boolean-tree queries are tracked in issue #1812.",
                field=None,
            )

        # --- Consume the next token, respecting quotes and parens ---
        raw_tok, remaining = _next_token(remaining)

        # Try count comparison first (messages:>=10)
        m_count = _COUNT_RE.match(raw_tok)
        if m_count:
            count_field = m_count.group(1)
            op_str = m_count.group(2)
            number = int(m_count.group(3))
            op: Literal[">=", "<=", "="] = ">=" if op_str == ">=" else "<=" if op_str == "<=" else "="
            tokens.append(_CountToken(field=count_field, op=op, number=number))
            continue

        # Try field clause (field:value or -field:value)
        m_field = _FIELD_RE.match(raw_tok)
        if m_field:
            neg_str, field_name, quoted_val, paren_val, bare_val = (
                m_field.group(1),
                m_field.group(2),
                m_field.group(3),
                m_field.group(4),
                m_field.group(5),
            )
            raw_value = quoted_val if quoted_val is not None else (paren_val if paren_val is not None else bare_val)
            tokens.append(_FieldToken(field=field_name.lower(), raw_value=raw_value or "", negated=bool(neg_str)))
            continue

        # Bare word (possibly negated via leading -)
        # Note: a bare `-` alone is treated as a word.
        # A `-"phrase"` form was handled above.
        negated_word = raw_tok.startswith("-") and len(raw_tok) > 1 and not raw_tok[1:].startswith('"')
        word_text = raw_tok[1:] if negated_word else raw_tok
        tokens.append(_TextToken(text=word_text, quoted=False, negated=negated_word))

    return tokens


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
            # near:"quoted phrase" → similar_text (value already stripped of quotes by lexer)
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

    tokens = _lex(expression)

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
    "ExpressionCompileError",
    "EXPRESSION_FIELD_REGISTRY",
    "_HAS_BOOL_MAP",
]
