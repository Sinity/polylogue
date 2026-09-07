"""Executable reference model for the session query surface.

The model is a test oracle, not a second archive.  It holds a *declared*
corpus of sessions as plain dataclasses, computes every session-grain query
field with obvious arithmetic, and evaluates the production query AST
(:func:`polylogue.archive.query.expression.parse_expression_ast`) against
those declared values.  There is no second grammar and no second SQL
implementation: the parser is production's, the semantics are declared here.

Three properties make it usable as an oracle:

* **Declared, not derived.** ``ModelCorpus`` is the input to *both* legs of a
  differential — :meth:`ModelCorpus.seed` admits it to a real archive through
  the production write choke point, and :meth:`ModelCorpus.reference_archive`
  answers the same requests from arithmetic.  Neither leg reads the other.
* **Closed vocabulary.** Every field the model answers is listed in
  :data:`MODEL_FIELD_SEMANTICS` with its comparison rule.  A field outside it
  raises :class:`UnsupportedByModelError` rather than quietly answering ``False``,
  so a differential can never pass by evaluating nothing.
* **One evaluation core.** Compact clause tokens are normalized into the same
  predicate shape the Boolean grammar produces, so ``origin:codex-session``
  and ``sessions where origin:codex-session`` cannot drift apart in the model.

Ingest is deliberately out of scope: the model predicts what the archive
stores and returns for a corpus admitted through
``write_parsed_session_to_archive``, not how provider bytes parse into one.
Byte-level parse agreement is ``tests/infra/source_differential.py``'s axis.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from polylogue.archive.query.expression import (
    QueryExpressionAST,
    _CountRangeToken,
    _CountToken,
    _DateComparisonToken,
    _DateRangeToken,
    _FieldToken,
    _TextToken,
    parse_expression_ast,
)
from polylogue.archive.query.predicate import (
    QueryBoolPredicate,
    QueryCompareOp,
    QueryExistsPredicate,
    QueryFieldPredicate,
    QueryLineagePredicate,
    QueryNotPredicate,
    QueryPredicate,
    QuerySemanticPredicate,
    QuerySequenceConstraint,
    QuerySequencePredicate,
    QueryTextPredicate,
)

#: Block types the model understands.  ``tool_use`` and ``thinking`` are the
#: two that production's session counters key on
#: (``write.py:_session_count_values``); ``tool_result`` and ``text`` carry
#: content without moving a counter.
ModelBlockType = Literal["text", "thinking", "tool_use", "tool_result"]

#: Roles production counts separately in ``sessions``' stat columns.
ModelRole = Literal["user", "assistant", "system", "tool"]


class UnsupportedByModelError(NotImplementedError):
    """The expression names a field or construct the model does not declare.

    Raised rather than answered so an oracle can never agree with a surface by
    evaluating nothing.  Callers that generate requests are expected to stay
    inside :data:`MODEL_FIELD_SEMANTICS`; a differential that hits this has
    found a vocabulary gap in the model, which is a model defect to fix, not a
    result to compare.
    """


# ---------------------------------------------------------------------------
# Declared corpus
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ModelBlock:
    """One declared content block."""

    type: ModelBlockType
    text: str | None = None
    tool_name: str | None = None
    tool_id: str | None = None
    semantic_type: str | None = None


@dataclass(frozen=True, slots=True)
class ModelMessage:
    """One declared message.

    ``text`` is the message's normalized text: the value production stores in
    ``messages.text`` and counts words from.  Blocks carry structure only —
    the model never re-derives ``text`` from them, because production does not
    either.
    """

    message_id: str
    role: ModelRole
    text: str
    timestamp: str
    blocks: tuple[ModelBlock, ...] = ()
    #: ``human_authored`` is the only value production's ``authored_user_*``
    #: counters key on; ``None`` leaves the writer's default in place, which
    #: is not ``human_authored``.
    material_origin: str | None = None

    @property
    def word_count(self) -> int:
        """Words as production counts them (``write.py:_word_count``)."""
        return len(self.text.split()) if self.text else 0

    @property
    def has_tool_use(self) -> bool:
        """Whether a ``tool_use`` block is present.

        ``tool_result`` deliberately does not count: production's
        ``tool_use_count`` is ``_has_block(message, BlockType.TOOL_USE)``.
        """
        return any(block.type == "tool_use" for block in self.blocks)

    @property
    def has_thinking(self) -> bool:
        return any(block.type == "thinking" for block in self.blocks)

    @property
    def is_human_authored(self) -> bool:
        return self.material_origin == "human_authored"

    @property
    def tool_names(self) -> tuple[str, ...]:
        """Tool names a ``tool:`` filter sees.

        Production builds the ``tool:`` vocabulary from the session's actions,
        and an action is a named tool call: the name echoed on a
        ``tool_result`` block does not make the session a user of that tool a
        second time, and a ``tool_use`` block with no name produces no action
        at all (``actions/parsing.py`` skips it).
        """
        return tuple(block.tool_name for block in self.blocks if block.type == "tool_use" and block.tool_name)

    @property
    def semantic_types(self) -> tuple[str, ...]:
        return tuple(block.semantic_type for block in self.blocks if block.type == "tool_use" and block.semantic_type)

    @property
    def block_search_texts(self) -> tuple[str, ...]:
        """Per-block indexed text, as ``blocks.search_text`` stores it.

        A ``tool_use`` block indexes its tool name; every other block indexes
        its text.  Message text is *not* separately indexed — a message with
        no blocks contributes nothing, which is why the declared corpus always
        gives a message a text block.
        """
        return tuple(
            (block.tool_name or "") if block.type == "tool_use" else (block.text or "") for block in self.blocks
        )


@dataclass(frozen=True, slots=True)
class ModelSession:
    """One declared session and the identity production computes for it."""

    #: The builder key.  Production's native id is ``ext-<key>`` because that
    #: is what ``SessionBuilder`` seeds; keeping the key separate lets the
    #: model state both without either being guessed from the other.
    key: str
    origin: str
    provider: str
    title: str | None
    created_at: str
    updated_at: str
    messages: tuple[ModelMessage, ...] = ()
    tags: tuple[str, ...] = ()
    parent_key: str | None = None
    git_repository_url: str | None = None
    provider_project_ref: str | None = None
    working_directories: tuple[str, ...] = ()

    @property
    def native_id(self) -> str:
        return f"ext-{self.key}"

    @property
    def session_id(self) -> str:
        """``origin || ':' || native_id`` — the generated identity."""
        return f"{self.origin}:{self.native_id}"

    # -- counters -----------------------------------------------------------
    # One statement per stat column production maintains in ``sessions``.

    @property
    def message_count(self) -> int:
        return len(self.messages)

    @property
    def word_count(self) -> int:
        return sum(message.word_count for message in self.messages)

    @property
    def tool_use_count(self) -> int:
        return sum(1 for message in self.messages if message.has_tool_use)

    @property
    def thinking_count(self) -> int:
        return sum(1 for message in self.messages if message.has_thinking)

    @property
    def paste_count(self) -> int:
        """Always zero: the declared corpus carries no paste spans."""
        return 0

    def _role_messages(self, role: ModelRole) -> tuple[ModelMessage, ...]:
        return tuple(message for message in self.messages if message.role == role)

    @property
    def user_message_count(self) -> int:
        return len(self._role_messages("user"))

    @property
    def assistant_message_count(self) -> int:
        return len(self._role_messages("assistant"))

    @property
    def system_message_count(self) -> int:
        return len(self._role_messages("system"))

    @property
    def tool_message_count(self) -> int:
        return len(self._role_messages("tool"))

    @property
    def user_word_count(self) -> int:
        return sum(message.word_count for message in self._role_messages("user"))

    @property
    def assistant_word_count(self) -> int:
        return sum(message.word_count for message in self._role_messages("assistant"))

    @property
    def authored_user_message_count(self) -> int:
        return sum(1 for message in self.messages if message.is_human_authored)

    @property
    def authored_user_word_count(self) -> int:
        return sum(message.word_count for message in self.messages if message.is_human_authored)

    # -- text ---------------------------------------------------------------

    @property
    def message_text(self) -> str:
        return "\n".join(message.text for message in self.messages)

    @property
    def search_tokens(self) -> frozenset[str]:
        """Lowercased word tokens of the session's indexed block text.

        The FTS index is contentless ``unicode61`` over ``blocks.search_text``
        with no stemmer, so for the ASCII-word vocabulary the corpus
        generators emit, a term matches exactly when it is one of these
        tokens.  The session title is deliberately absent: it is filtered by
        ``title:``, not reachable by a text term.  Punctuation and mixed
        scripts are outside the declared vocabulary; see
        :func:`assert_model_vocabulary`.
        """
        return frozenset(
            token.lower()
            for message in self.messages
            for search_text in message.block_search_texts
            for token in search_text.split()
        )

    @property
    def tool_names(self) -> frozenset[str]:
        return frozenset(name for message in self.messages for name in message.tool_names)

    @property
    def semantic_types(self) -> frozenset[str]:
        return frozenset(item for message in self.messages for item in message.semantic_types)

    @property
    def first_timestamp(self) -> datetime | None:
        stamps = [_parse_timestamp(message.timestamp) for message in self.messages]
        present = [stamp for stamp in stamps if stamp is not None]
        return min(present) if present else None


# ---------------------------------------------------------------------------
# Declared field semantics
# ---------------------------------------------------------------------------


ComparisonRule = Literal["equals", "substring", "membership", "numeric", "boolean", "date", "token", "prefix"]


@dataclass(frozen=True, slots=True)
class ModelFieldSemantics:
    """How the model answers one DSL field."""

    rule: ComparisonRule
    #: Human-readable statement of what the value is.  Read this before
    #: trusting a differential result for the field.
    meaning: str


#: The closed vocabulary the model answers.  Keys are DSL field tokens from
#: ``EXPRESSION_FIELD_REGISTRY``; anything else raises
#: :class:`UnsupportedByModelError`.
MODEL_FIELD_SEMANTICS: Mapping[str, ModelFieldSemantics] = {
    "id": ModelFieldSemantics("equals", "generated session id"),
    "session": ModelFieldSemantics("equals", "generated session id"),
    "origin": ModelFieldSemantics("equals", "public origin token"),
    "title": ModelFieldSemantics("substring", "session title, case-insensitive substring"),
    "tag": ModelFieldSemantics("membership", "declared user tags"),
    "repo": ModelFieldSemantics("substring", "git repository url"),
    "project": ModelFieldSemantics("equals", "provider project ref"),
    "cwd": ModelFieldSemantics("prefix", "any declared working directory"),
    "tool": ModelFieldSemantics("membership", "tool names on tool_use blocks, case-folded, every value required"),
    "action": ModelFieldSemantics("membership", "semantic types on tool_use blocks, case-folded, every value required"),
    "has": ModelFieldSemantics("boolean", "paste/tool_use/thinking presence"),
    "root": ModelFieldSemantics("boolean", "session has no declared parent"),
    "contains": ModelFieldSemantics("token", "message-text word token"),
    "since": ModelFieldSemantics("date", "session sort timestamp at or after value"),
    "until": ModelFieldSemantics("date", "session sort timestamp at or before value"),
    "date": ModelFieldSemantics("date", "session sort timestamp"),
    "messages": ModelFieldSemantics("numeric", "message_count"),
    "words": ModelFieldSemantics("numeric", "word_count"),
    "user_messages": ModelFieldSemantics("numeric", "user_message_count"),
    "assistant_messages": ModelFieldSemantics("numeric", "assistant_message_count"),
    "system_messages": ModelFieldSemantics("numeric", "system_message_count"),
    "tool_messages": ModelFieldSemantics("numeric", "tool_message_count"),
    "tool_use_messages": ModelFieldSemantics("numeric", "tool_use_count"),
    "thinking_messages": ModelFieldSemantics("numeric", "thinking_count"),
    "paste_messages": ModelFieldSemantics("numeric", "paste_count"),
    "user_words": ModelFieldSemantics("numeric", "user_word_count"),
    "assistant_words": ModelFieldSemantics("numeric", "assistant_word_count"),
    "authored_user_messages": ModelFieldSemantics("numeric", "authored_user_message_count"),
    "authored_user_words": ModelFieldSemantics("numeric", "authored_user_word_count"),
}


#: ``has:<value>`` tokens the model answers.
_HAS_TOKENS = ("paste", "tool_use", "thinking")

#: Fields whose values production compares case-folded on both sides: the
#: Boolean grammar lowercases the written value and the matcher lowercases the
#: session's own names (``runtime_matching.matches_tool_terms``).  Every other
#: field compares the value as written, which is why ``tag:Review`` and
#: ``tag:review`` are different filters and ``tool:Read`` and ``tool:read``
#: are not.
_CASE_FOLDED_FIELDS = frozenset({"tool", "action"})

#: Fields where a multi-valued clause requires *every* value.  Production
#: intersects by subset for tools and action kinds, so ``tool:(Read|Bash)``
#: selects sessions that used both; every other multi-valued field selects a
#: session matching any one value.
_CONJUNCTIVE_FIELDS = frozenset({"tool", "action"})


def _numeric_value(session: ModelSession, name: str) -> int:
    column = {
        "messages": "message_count",
        "words": "word_count",
        "user_messages": "user_message_count",
        "assistant_messages": "assistant_message_count",
        "system_messages": "system_message_count",
        "tool_messages": "tool_message_count",
        "tool_use_messages": "tool_use_count",
        "thinking_messages": "thinking_count",
        "paste_messages": "paste_count",
        "user_words": "user_word_count",
        "assistant_words": "assistant_word_count",
        "authored_user_messages": "authored_user_message_count",
        "authored_user_words": "authored_user_word_count",
    }[name]
    value = getattr(session, column)
    assert isinstance(value, int)
    return value


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _default_sort_key(session: ModelSession) -> tuple[float, str]:
    """Most-recently-updated first, then by generated id.

    The id leg only breaks ties; a corpus that gives two sessions the same
    ``updated_at`` is asking the model to guess an order the archive does not
    promise, so generators keep the timestamps distinct.
    """
    stamp = _sort_timestamp(session)
    return (-(stamp.timestamp() if stamp is not None else 0.0), session.session_id)


def _sort_timestamp(session: ModelSession) -> datetime | None:
    """The timestamp date filters compare against.

    Production sorts and range-filters sessions on ``updated_at``-derived
    ``sort_key``; the model states the same choice rather than averaging the
    two session timestamps.
    """
    return _parse_timestamp(session.updated_at)


def _compare_numbers(actual: int, expected: str, op: QueryCompareOp) -> bool:
    try:
        wanted = float(expected)
    except ValueError as exc:
        raise UnsupportedByModelError(f"non-numeric comparison value {expected!r}") from exc
    if op == "=":
        return actual == wanted
    if op == ">":
        return actual > wanted
    if op == ">=":
        return actual >= wanted
    if op == "<":
        return actual < wanted
    if op == "<=":
        return actual <= wanted
    raise UnsupportedByModelError(f"unsupported numeric operator {op!r}")


def _compare_dates(actual: datetime | None, expected: str, op: QueryCompareOp) -> bool:
    wanted = _parse_timestamp(expected)
    if wanted is None:
        raise UnsupportedByModelError(f"relative or unparsable date value {expected!r}")
    if actual is None:
        return False
    if op == "=":
        return actual == wanted
    if op == ">":
        return actual > wanted
    if op == ">=":
        return actual >= wanted
    if op == "<":
        return actual < wanted
    if op == "<=":
        return actual <= wanted
    raise UnsupportedByModelError(f"unsupported date operator {op!r}")


def _field_matches_value(session: ModelSession, name: str, value: str, op: QueryCompareOp) -> bool:
    semantics = MODEL_FIELD_SEMANTICS.get(name)
    if semantics is None:
        raise UnsupportedByModelError(f"model does not declare session field {name!r}")

    if semantics.rule == "numeric":
        return _compare_numbers(_numeric_value(session, name), value, op)

    if semantics.rule == "date":
        actual = _sort_timestamp(session)
        if name == "since":
            return _compare_dates(actual, value, ">=")
        if name == "until":
            return _compare_dates(actual, value, "<=")
        return _compare_dates(actual, value, op)

    if op != "=":
        raise UnsupportedByModelError(f"field {name!r} does not support operator {op!r}")

    if semantics.rule == "equals":
        if name in {"id", "session"}:
            return session.session_id == value
        if name == "origin":
            return session.origin == value
        if name == "project":
            return session.provider_project_ref == value
        raise UnsupportedByModelError(f"unrouted equals field {name!r}")

    if semantics.rule == "substring":
        haystack = {"title": session.title or "", "repo": session.git_repository_url or ""}[name]
        return value.lower() in haystack.lower()

    if semantics.rule == "membership":
        candidates = {
            "tag": frozenset(session.tags),
            "tool": session.tool_names,
            "action": session.semantic_types,
        }[name]
        if name in _CASE_FOLDED_FIELDS:
            return value.strip().lower() in {candidate.strip().lower() for candidate in candidates}
        return value in candidates

    if semantics.rule == "prefix":
        return any(directory.startswith(value) for directory in session.working_directories)

    if semantics.rule == "token":
        return value.lower() in session.search_tokens

    if semantics.rule == "boolean":
        if name == "root":
            wanted = value.lower() in {"true", "yes", "1"}
            return (session.parent_key is None) == wanted
        if name == "has":
            if value not in _HAS_TOKENS:
                raise UnsupportedByModelError(f"model does not declare has:{value}")
            return {
                "paste": session.paste_count > 0,
                "tool_use": session.tool_use_count > 0,
                "thinking": session.thinking_count > 0,
            }[value]
        raise UnsupportedByModelError(f"unrouted boolean field {name!r}")

    raise UnsupportedByModelError(f"unrouted comparison rule {semantics.rule!r} for {name!r}")


def _field_predicate_matches(session: ModelSession, predicate: QueryFieldPredicate) -> bool:
    name = predicate.field.removeprefix("session.")
    if not predicate.values:
        raise UnsupportedByModelError(f"field predicate {name!r} carries no value")
    combine = all if name in _CONJUNCTIVE_FIELDS else any
    return combine(_field_matches_value(session, name, value, predicate.op) for value in predicate.values)


# ---------------------------------------------------------------------------
# Message-grain evaluation (for ``exists message(...)``)
# ---------------------------------------------------------------------------

#: Structural fields the model answers inside ``exists message(...)``.
MODEL_MESSAGE_FIELDS: Mapping[str, str] = {
    "role": "message role token",
    "text": "message text, case-insensitive substring",
    "words": "message word count",
}


def _message_matches(message: ModelMessage, predicate: QueryPredicate) -> bool:
    if isinstance(predicate, QueryFieldPredicate):
        name = predicate.field.removeprefix("message.")
        if name not in MODEL_MESSAGE_FIELDS:
            raise UnsupportedByModelError(f"model does not declare message field {name!r}")
        if name == "role":
            return any(message.role == value for value in predicate.values)
        if name == "text":
            return any(value.lower() in message.text.lower() for value in predicate.values)
        return any(_compare_numbers(message.word_count, value, predicate.op) for value in predicate.values)
    if isinstance(predicate, QueryTextPredicate):
        return predicate.text.lower() in message.text.lower()
    if isinstance(predicate, QueryNotPredicate):
        return not _message_matches(message, predicate.child)
    if isinstance(predicate, QueryBoolPredicate):
        results = [_message_matches(message, child) for child in predicate.children]
        return any(results) if predicate.op == "or" else all(results)
    raise UnsupportedByModelError(f"unsupported message predicate: {type(predicate).__name__}")


#: Structural fields the model answers inside ``exists block(...)``.
MODEL_BLOCK_FIELDS: Mapping[str, str] = {
    "type": "block type token",
    "text": "block text, case-insensitive substring",
    "tool": "block tool name",
}


def _block_matches(block: ModelBlock, predicate: QueryPredicate) -> bool:
    if isinstance(predicate, QueryFieldPredicate):
        name = predicate.field.removeprefix("block.")
        if name not in MODEL_BLOCK_FIELDS:
            raise UnsupportedByModelError(f"model does not declare block field {name!r}")
        if name == "type":
            return any(block.type == value for value in predicate.values)
        if name == "tool":
            return any(block.tool_name == value for value in predicate.values)
        return any(value.lower() in (block.text or "").lower() for value in predicate.values)
    if isinstance(predicate, QueryTextPredicate):
        return predicate.text.lower() in (block.text or "").lower()
    if isinstance(predicate, QueryNotPredicate):
        return not _block_matches(block, predicate.child)
    if isinstance(predicate, QueryBoolPredicate):
        results = [_block_matches(block, child) for child in predicate.children]
        return any(results) if predicate.op == "or" else all(results)
    raise UnsupportedByModelError(f"unsupported block predicate: {type(predicate).__name__}")


# ---------------------------------------------------------------------------
# Action-grain evaluation (for ``seq(...)``)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ModelAction:
    """One declared action: a ``tool_use`` block in message order.

    Production's ``actions`` view joins a ``tool_use`` block to its
    ``tool_result`` by ``tool_id``; the model's declared vocabulary reaches
    only the fields the ``tool_use`` side carries, so an action here is that
    block plus the timestamp of the message it sits in.
    """

    kind: str
    tool_name: str | None
    timestamp: datetime | None


#: Tool names the model declares an action semantic type for.  Production
#: honours a block's declared ``semantic_type`` only when it names a known
#: tool category; otherwise it classifies the call from the tool name and its
#: input.  These pairs are the ones where both routes land on the same kind,
#: and :func:`assert_model_vocabulary` holds a corpus to them.
MODEL_TOOL_SEMANTICS: Mapping[str, str] = {
    "Bash": "shell",
    "Read": "file_read",
    "Edit": "file_edit",
}

#: Action fields the model answers inside ``seq(...)``.  ``command``, ``path``
#: and ``output`` are deliberately absent: they need production's tool-input
#: extraction, which the model does not declare and must not guess.
MODEL_ACTION_FIELDS: Mapping[str, str] = {
    "action": "semantic type of the tool_use block",
    "type": "semantic type of the tool_use block",
    "tool": "tool name on the tool_use block",
}


def _actions(session: ModelSession) -> tuple[ModelAction, ...]:
    return tuple(
        ModelAction(
            kind=block.semantic_type or "",
            tool_name=block.tool_name,
            timestamp=_parse_timestamp(message.timestamp),
        )
        for message in session.messages
        for block in message.blocks
        if block.type == "tool_use"
    )


def _action_matches(action: ModelAction, predicate: QueryPredicate) -> bool:
    if isinstance(predicate, QueryFieldPredicate):
        name = predicate.field.removeprefix("action.")
        if name not in MODEL_ACTION_FIELDS:
            raise UnsupportedByModelError(f"model does not declare action field {name!r}")
        actual = action.kind if name in {"action", "type"} else (action.tool_name or "")
        return any(actual.strip().lower() == value.strip().lower() for value in predicate.values)
    if isinstance(predicate, QueryNotPredicate):
        return not _action_matches(action, predicate.child)
    if isinstance(predicate, QueryBoolPredicate):
        results = [_action_matches(action, child) for child in predicate.children]
        return any(results) if predicate.op == "or" else all(results)
    raise UnsupportedByModelError(f"unsupported action predicate: {type(predicate).__name__}")


def _sequence_matches(session: ModelSession, predicate: QuerySequencePredicate) -> bool:
    """Whether the declared actions contain the ordered sequence.

    The walk is the model's own: a set of surviving positions per step, each
    step advancing to some later action, with the edge constraint deciding
    which later positions are reachable.  ``ordered`` allows any gap,
    ``next`` demands adjacency, ``within`` bounds the elapsed time.
    """
    steps = predicate.steps
    if not steps:
        return True
    actions = _actions(session)
    if not actions:
        return False
    constraints = predicate.constraints or tuple(QuerySequenceConstraint() for _ in range(len(steps) - 1))
    if len(constraints) != len(steps) - 1:
        raise UnsupportedByModelError("sequence constraints must describe every edge between steps")

    reachable = {index for index, action in enumerate(actions) if _action_matches(action, steps[0])}
    for step_index, step in enumerate(steps[1:]):
        constraint = constraints[step_index]
        if constraint.kind not in {"ordered", "next", "within"}:
            raise UnsupportedByModelError(f"model does not declare sequence constraint {constraint.kind!r}")
        advanced: set[int] = set()
        for previous in reachable:
            for current in range(previous + 1, len(actions)):
                if constraint.kind == "next" and current != previous + 1:
                    break
                if not _action_matches(actions[current], step):
                    continue
                if constraint.kind == "within":
                    before, after = actions[previous].timestamp, actions[current].timestamp
                    if before is None or after is None or constraint.within_ms is None:
                        continue
                    elapsed_ms = (after - before).total_seconds() * 1000
                    if elapsed_ms < 0 or elapsed_ms > constraint.within_ms:
                        continue
                advanced.add(current)
        reachable = advanced
        if not reachable:
            return False
    return True


# ---------------------------------------------------------------------------
# Compact-clause normalization
# ---------------------------------------------------------------------------


def _predicates_from_clause(token: object) -> tuple[QueryPredicate, bool]:
    """Normalize one compact clause token into a predicate plus negation.

    Both surface forms of the DSL reach the same evaluator through this, so a
    compact clause and its Boolean spelling cannot diverge inside the model.
    """
    if isinstance(token, _FieldToken):
        values = tuple(token.raw_value.strip("()").split("|"))
        return QueryFieldPredicate(field=token.field, values=values, op="="), token.negated
    if isinstance(token, _TextToken):
        return QueryTextPredicate(text=token.text), token.negated
    if isinstance(token, _CountToken):
        return QueryFieldPredicate(field=token.field, values=(str(token.number),), op=token.op), False
    if isinstance(token, _CountRangeToken):
        return (
            QueryBoolPredicate(
                "and",
                (
                    QueryFieldPredicate(field=token.field, values=(str(token.min_number),), op=">="),
                    QueryFieldPredicate(field=token.field, values=(str(token.max_number),), op="<="),
                ),
            ),
            False,
        )
    if isinstance(token, _DateComparisonToken):
        return QueryFieldPredicate(field="date", values=(token.value,), op=token.op), False
    if isinstance(token, _DateRangeToken):
        return (
            QueryBoolPredicate(
                "and",
                (
                    QueryFieldPredicate(field="date", values=(token.min_value,), op=">="),
                    QueryFieldPredicate(field="date", values=(token.max_value,), op="<="),
                ),
            ),
            False,
        )
    raise UnsupportedByModelError(f"model does not declare compact clause {type(token).__name__}")


def predicate_from_ast(ast: QueryExpressionAST) -> QueryPredicate | None:
    """Return one predicate for either AST form, or ``None`` for match-all.

    Compact clauses are conjoined, which is the compact grammar's own rule.
    """
    if ast.boolean_predicate is not None:
        return ast.boolean_predicate
    if not ast.clauses:
        return None
    children: list[QueryPredicate] = []
    for token in ast.clauses:
        predicate, negated = _predicates_from_clause(token)
        children.append(QueryNotPredicate(predicate) if negated else predicate)
    return children[0] if len(children) == 1 else QueryBoolPredicate("and", tuple(children))


# ---------------------------------------------------------------------------
# Session-grain evaluation
# ---------------------------------------------------------------------------


def evaluate_predicate(
    session: ModelSession,
    predicate: QueryPredicate,
    *,
    lineage: frozenset[str] = frozenset(),
) -> bool:
    """Evaluate one production :class:`QueryPredicate` against declared values."""
    if isinstance(predicate, QueryFieldPredicate):
        return _field_predicate_matches(session, predicate)
    if isinstance(predicate, QueryTextPredicate):
        return predicate.text.lower() in session.search_tokens
    if isinstance(predicate, QueryNotPredicate):
        return not evaluate_predicate(session, predicate.child, lineage=lineage)
    if isinstance(predicate, QueryBoolPredicate):
        results = [evaluate_predicate(session, child, lineage=lineage) for child in predicate.children]
        return all(results) if predicate.op == "and" else any(results)
    if isinstance(predicate, QueryExistsPredicate):
        if predicate.unit == "message":
            return any(_message_matches(message, predicate.child) for message in session.messages)
        if predicate.unit == "block":
            return any(
                _block_matches(block, predicate.child) for message in session.messages for block in message.blocks
            )
        raise UnsupportedByModelError(f"model does not declare exists unit {predicate.unit!r}")
    if isinstance(predicate, QueryLineagePredicate):
        return predicate.seed_session_id in lineage
    if isinstance(predicate, QuerySemanticPredicate):
        raise UnsupportedByModelError("semantic predicates need a vector oracle")
    if isinstance(predicate, QuerySequencePredicate):
        return _sequence_matches(session, predicate)
    raise UnsupportedByModelError(f"unsupported query predicate: {type(predicate).__name__}")


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ReferenceResult:
    """The model's answer, at the grain a surface reports it.

    ``session_ids`` is ordered by the model's declared sort; ``total`` is the
    count *before* limit/offset, which is the grain every surface's ``total``
    field claims.  Keeping the two separate is what makes a count-grain error
    visible instead of absorbed.
    """

    session_ids: tuple[str, ...]
    total: int
    origin_facets: tuple[tuple[str, int], ...] = ()

    @property
    def id_set(self) -> frozenset[str]:
        return frozenset(self.session_ids)


@dataclass(frozen=True, slots=True)
class ModelRequest:
    """One generated request, in the form every surface can carry."""

    name: str
    expression: str
    limit: int | None = None
    offset: int = 0

    def __post_init__(self) -> None:
        if self.offset < 0:
            raise ValueError("offset must be non-negative")
        if self.limit is not None and self.limit < 1:
            raise ValueError("limit must be positive when set")


# ---------------------------------------------------------------------------
# Archive
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ReferenceArchive:
    """In-memory oracle over declared sessions."""

    sessions: dict[str, ModelSession] = field(default_factory=dict)

    @classmethod
    def from_model_sessions(cls, sessions: Iterable[ModelSession]) -> ReferenceArchive:
        archive = cls()
        for session in sessions:
            archive.add(session)
        return archive

    def add(self, session: ModelSession) -> None:
        self.sessions[session.key] = session

    def lineage(self, key: str) -> tuple[ModelSession, ...]:
        """Ancestors first, ending at ``key``; cycles stop at the repeat."""
        chain: list[ModelSession] = []
        current: str | None = key
        seen: set[str] = set()
        while current is not None and current in self.sessions and current not in seen:
            seen.add(current)
            chain.append(self.sessions[current])
            current = self.sessions[current].parent_key
        return tuple(reversed(chain))

    def recomposed(self, key: str) -> ModelSession:
        """The session with its inherited prefix physically composed.

        Reads recompose a child's inherited prefix, so an oracle that answers
        for a child must too.  A session with no declared parent is returned
        unchanged rather than copied.
        """
        chain = self.lineage(key)
        if len(chain) <= 1:
            return self.sessions[key]
        messages = tuple(message for ancestor in chain for message in ancestor.messages)
        return replace(self.sessions[key], messages=messages)

    def _lineage_ids(self, key: str) -> frozenset[str]:
        return frozenset(session.session_id for session in self.lineage(key))

    def matching(self, expression: str | QueryExpressionAST, *, recompose: bool = False) -> tuple[ModelSession, ...]:
        """Sessions matching ``expression``, in the archive's default order.

        The default listing order is most-recently-updated first, which is
        what pagination windows are cut from.  A model that ordered by id
        instead would report a different page for the same correct match set
        and turn every ``limit``/``offset`` request into a false divergence.
        """
        ast = parse_expression_ast(expression) if isinstance(expression, str) else expression
        predicate = predicate_from_ast(ast)
        selected: list[ModelSession] = []
        for key in self.sessions:
            candidate = self.recomposed(key) if recompose else self.sessions[key]
            if predicate is None or evaluate_predicate(candidate, predicate, lineage=self._lineage_ids(key)):
                selected.append(candidate)
        return tuple(sorted(selected, key=_default_sort_key))

    def query(self, request: ModelRequest | str, *, recompose: bool = False) -> ReferenceResult:
        """Answer one request at the grain surfaces report."""
        if isinstance(request, str):
            request = ModelRequest(name=request, expression=request)
        selected = self.matching(request.expression, recompose=recompose)
        total = len(selected)
        window = selected[request.offset :]
        if request.limit is not None:
            window = window[: request.limit]
        origins = sorted(session.origin for session in window)
        facets = tuple((origin, origins.count(origin)) for origin in sorted(set(origins)))
        return ReferenceResult(
            session_ids=tuple(session.session_id for session in window),
            total=total,
            origin_facets=facets,
        )


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ModelCorpus:
    """A declared corpus: the single input to both legs of a differential."""

    sessions: tuple[ModelSession, ...]

    def __post_init__(self) -> None:
        keys = [session.key for session in self.sessions]
        if len(set(keys)) != len(keys):
            raise ValueError("corpus session keys must be unique")
        known = set(keys)
        for session in self.sessions:
            if session.parent_key is not None and session.parent_key not in known:
                raise ValueError(f"session {session.key!r} declares unknown parent {session.parent_key!r}")

    def __iter__(self) -> Iterator[ModelSession]:
        return iter(self.sessions)

    def __len__(self) -> int:
        return len(self.sessions)

    @property
    def session_ids(self) -> tuple[str, ...]:
        return tuple(sorted(session.session_id for session in self.sessions))

    def reference_archive(self) -> ReferenceArchive:
        return ReferenceArchive.from_model_sessions(self.sessions)

    def seed(self, db_path: Path) -> tuple[str, ...]:
        """Admit the corpus to a real archive through the production writer.

        Every session goes through ``SessionBuilder.save`` →
        ``write_parsed_session_to_archive``, the choke point live ingest and
        replay share, so the archive leg of a differential is production's own
        write path rather than a test-only insert.  Returns the generated
        session ids in declaration order.
        """
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
        from tests.infra.storage_records import SessionBuilder

        db_path.parent.mkdir(parents=True, exist_ok=True)
        written: list[str] = []
        for session in _parents_first(self.sessions):
            builder = (
                SessionBuilder(db_path, session.key)
                .provider(session.provider)
                .title(session.title)
                .created_at(session.created_at)
                .updated_at(session.updated_at)
            )
            if session.git_repository_url is not None:
                builder.git_repository_url(session.git_repository_url)
            if session.provider_project_ref is not None:
                builder.provider_project_ref(session.provider_project_ref)
            if session.working_directories:
                builder.working_directories(list(session.working_directories))
            if session.parent_key is not None:
                builder.parent_session(_session_id_for_key(self.sessions, session.parent_key))
            for message in session.messages:
                kwargs: dict[str, object] = {"blocks": [_block_payload(block) for block in message.blocks]}
                if message.material_origin is not None:
                    kwargs["material_origin"] = message.material_origin
                builder.add_message(
                    message_id=message.message_id,
                    role=message.role,
                    text=message.text,
                    timestamp=message.timestamp,
                    **kwargs,
                )
            builder.save()
            written.append(session.session_id)
            if session.tags:
                with ArchiveStore(db_path.parent) as archive:
                    archive.add_user_tags((session.session_id,), session.tags)
        return tuple(written)


def _parents_first(sessions: Sequence[ModelSession]) -> tuple[ModelSession, ...]:
    """Order sessions so a parent is always written before its child."""
    by_key = {session.key: session for session in sessions}
    ordered: list[ModelSession] = []
    placed: set[str] = set()

    def place(session: ModelSession) -> None:
        if session.key in placed:
            return
        placed.add(session.key)
        if session.parent_key is not None and session.parent_key in by_key:
            place(by_key[session.parent_key])
        ordered.append(session)

    for session in sessions:
        place(session)
    return tuple(ordered)


def _session_id_for_key(sessions: Sequence[ModelSession], key: str) -> str:
    for session in sessions:
        if session.key == key:
            return session.session_id
    raise KeyError(key)


def _block_payload(block: ModelBlock) -> dict[str, object]:
    payload: dict[str, object] = {"type": block.type}
    if block.text is not None:
        payload["text"] = block.text
    if block.tool_name is not None:
        payload["tool_name"] = block.tool_name
    if block.tool_id is not None:
        payload["tool_id"] = block.tool_id
    if block.semantic_type is not None:
        payload["semantic_type"] = block.semantic_type
    if block.type == "tool_use":
        payload["tool_input"] = {}
    return payload


def assert_model_vocabulary(corpus: ModelCorpus) -> None:
    """Refuse a corpus that leaves the model's declared vocabulary.

    Two declarations are enforced.  Text: the model equates FTS matching with
    whitespace-split lowercase tokens, which holds for ASCII words and stops
    holding the moment punctuation, case-folding beyond ASCII, or CJK enters
    the text.  Actions: the model answers ``action:`` from the block's
    declared semantic type, while production keeps that type only when it
    names a known tool category and otherwise reclassifies from the tool name
    and its input, so only the pairs in :data:`MODEL_TOOL_SEMANTICS` are
    declared to agree.  A generator that drifts out of either must fail here
    rather than produce silent disagreement.
    """
    for session in corpus:
        for message in session.messages:
            for token in message.text.split():
                if not token.isascii() or not token.isalnum():
                    raise ValueError(
                        f"session {session.key!r} message {message.message_id!r} carries token {token!r}: "
                        "the model declares FTS agreement only for ASCII alphanumeric word tokens"
                    )
            for block in message.blocks:
                if block.type != "tool_use":
                    continue
                declared = MODEL_TOOL_SEMANTICS.get(block.tool_name or "")
                if declared is None or declared != block.semantic_type:
                    raise ValueError(
                        f"session {session.key!r} message {message.message_id!r} declares tool_use "
                        f"{block.tool_name!r}/{block.semantic_type!r}: the model declares action agreement "
                        f"only for {sorted(MODEL_TOOL_SEMANTICS.items())}"
                    )


__all__ = [
    "MODEL_ACTION_FIELDS",
    "MODEL_BLOCK_FIELDS",
    "MODEL_FIELD_SEMANTICS",
    "MODEL_MESSAGE_FIELDS",
    "MODEL_TOOL_SEMANTICS",
    "ModelAction",
    "ModelBlock",
    "ModelBlockType",
    "ModelCorpus",
    "ModelFieldSemantics",
    "ModelMessage",
    "ModelRequest",
    "ModelRole",
    "ModelSession",
    "ReferenceArchive",
    "ReferenceResult",
    "UnsupportedByModelError",
    "assert_model_vocabulary",
    "evaluate_predicate",
    "predicate_from_ast",
]
