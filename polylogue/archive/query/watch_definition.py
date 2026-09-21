"""Whether a saved view's selection can be watched, and what it compiles to.

The predicate a watch is re-evaluated against is the canonical v1 plan the
standing-query stage executes, so a view whose selection has no evaluable
predicate must be refused at the moment it is marked ``watch=True`` -- not
watched under a different set than the user named (polylogue-pm8cj).

This half is pure: it compiles and refuses, touches no connection, and lives
outside ``polylogue/storage`` so a surface can ask "is this watchable?" and
answer with a typed 400 without importing the substrate. The durable
registration that follows a successful validation is
``storage/sqlite/query_watch.register_query_watch``.
"""

from __future__ import annotations

from collections.abc import Mapping

from polylogue.core.query_identity import JsonValue

#: Only the session grain has a production evaluator
#: (``polylogue.archive.query.production_evaluator``), so a watch is only ever
#: registered at that grain; the other members of the declared grain
#: vocabulary have no executable planner form yet.
WATCH_DEFINITION_GRAIN = "session"
WATCH_DEFINITION_LANE = "dialogue"
WATCH_DEFINITION_RANK_POLICY = "mixed"

#: Saved-view parameters a watched definition can carry.  Everything else --
#: ``origin``, ``tag``, ``repo``, ``since``, ``limit``, ``sort`` ... -- narrows
#: or orders the view without appearing in the compiled predicate, so a watch
#: honouring only ``query`` would drift against a *different* set than the one
#: the user named.  Refusing is the only honest answer; the DSL expresses all
#: of them.
_WATCH_DEFINITION_PARAMS = frozenset({"query"})


class WatchDefinitionError(ValueError):
    """A saved view cannot be promoted into an evaluable watched definition."""


def compile_watch_definition(expression: str) -> dict[str, JsonValue]:
    """Compile a DSL selection expression into a canonical v1 predicate AST.

    The returned payload is exactly the shape
    ``polylogue.archive.query.predicate.predicate_from_payload`` inverts, which
    is the only shape ``ArchiveCanonicalPlanEvaluator`` can execute.  Compact
    field lowering (``origin:codex-session`` alone) and bare FTS terms
    (``timeout``) both produce *no* predicate, so they are refused with the
    explicit selection form to use instead.
    """
    from polylogue.archive.query.expression import ExpressionCompileError, parse_expression_ast
    from polylogue.archive.query.predicate import predicate_from_payload

    text = expression.strip()
    if not text:
        raise WatchDefinitionError("a watched query definition requires a non-empty selection expression")
    try:
        ast = parse_expression_ast(text)
    except ExpressionCompileError as exc:
        raise WatchDefinitionError(f"watched query expression does not compile: {exc}") from exc
    if ast.ref_operand is not None:
        raise WatchDefinitionError(
            "a watched query definition cannot be a durable reference pipeline: "
            "a reference re-evaluates or replays its own relation and has no predicate to watch"
        )
    predicate = ast.boolean_predicate
    if predicate is None:
        raise WatchDefinitionError(
            f"{text!r} does not compile to a typed selection predicate, so the canonical-plan "
            "evaluator cannot re-evaluate it; write the watch as an explicit selection, "
            "e.g. 'sessions where origin:codex-session AND repo:polylogue'"
        )
    payload = predicate.to_payload()
    try:
        round_tripped = predicate_from_payload(payload)
    except Exception as exc:  # an un-invertible shape is not a watchable definition
        raise WatchDefinitionError(
            f"{text!r} compiles to a predicate the durable definition grammar cannot carry: {exc}"
        ) from exc
    if round_tripped != predicate:
        raise WatchDefinitionError(
            f"{text!r} does not survive the durable definition round trip; watching it would "
            "evaluate a different predicate than the one named"
        )
    return dict(payload)  # type: ignore[arg-type]


def validate_watch_definition(query_params: Mapping[str, object]) -> dict[str, JsonValue]:
    """Compile the watched definition a saved view's parameters declare.

    Raises :class:`WatchDefinitionError` -- a ``ValueError`` -- so a preview
    fails closed at plan time instead of at apply time.
    """
    unsupported = sorted(set(query_params) - _WATCH_DEFINITION_PARAMS)
    if unsupported:
        raise WatchDefinitionError(
            "a watched saved view is defined by its 'query' expression alone; "
            f"{unsupported} would narrow or order the view without entering the watched "
            "definition, so the watch would report drift over a different set. "
            "Express them inside the expression instead."
        )
    expression = query_params.get("query")
    if not isinstance(expression, str):
        raise WatchDefinitionError("a watched saved view requires a 'query' expression string")
    return compile_watch_definition(expression)
