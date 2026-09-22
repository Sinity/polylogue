"""A watched saved view may only carry a predicate the evaluator can execute.

``compile_watch_definition`` used to admit anything the durable definition
grammar could round trip. Serializability is not executability: a semantic
(``near:``) predicate round trips, but the standing-query stage lowers the
restored predicate to SQL through ``_boolean_predicate_clause``, which has no
semantic branch and raises ``TypeError``. Registering one therefore failed the
whole convergence stage on every tick, taking unrelated watches with it.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from polylogue.archive.filter.filters import SessionFilter
from polylogue.archive.query.expression import parse_expression_ast
from polylogue.archive.query.plan import SessionQueryPlan
from polylogue.archive.query.watch_definition import (
    WatchDefinitionError,
    compile_watch_definition,
    validate_watch_definition,
)
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

_SEMANTIC = 'sessions where near:text:"query compiler"'
_EXECUTABLE = "sessions where origin:codex-session AND repo:polylogue"


def test_semantic_watch_definition_is_refused() -> None:
    """The refusal covers a bare, a conjoined and a negated semantic predicate.

    Anti-vacuity: removing the ``_unexecutable_predicate_kind`` guard makes
    every ``pytest.raises`` here fail, because the payload round trip these
    expressions already survive is the only other check. The executable
    expression pins the opposite direction, so the guard cannot become a
    blanket refusal of every watch.
    """
    for expression in (
        _SEMANTIC,
        'sessions where origin:codex-session AND near:text:"x"',
        'sessions where NOT near:text:"x"',
    ):
        with pytest.raises(WatchDefinitionError, match="semantic"):
            compile_watch_definition(expression)
    with pytest.raises(WatchDefinitionError, match="semantic"):
        validate_watch_definition({"query": _SEMANTIC})

    payload = compile_watch_definition(_EXECUTABLE)
    assert payload["kind"] == "and"


def test_semantic_predicate_breaks_the_evaluator(tmp_path: Path) -> None:
    """Why the refusal is permanent, not deferred: the lowering cannot run it.

    This drives the exact two lines ``ArchiveCanonicalPlanEvaluator.evaluate``
    runs after restoring a predicate -- build a ``SessionQueryPlan`` around it
    and resolve every matching summary. Without this the first test would only
    assert its own error wording.

    Anti-vacuity: swapping ``_SEMANTIC`` for ``_EXECUTABLE`` makes the
    ``pytest.raises`` fail, so it is the semantic predicate and not the empty
    fixture archive that raises.
    """
    initialize_active_archive_root(tmp_path)

    def resolve(expression: str) -> list[object]:
        predicate = parse_expression_ast(expression).boolean_predicate
        assert predicate is not None
        plan = SessionQueryPlan(boolean_predicate=predicate)
        session_filter = SessionFilter.from_query_plan(plan, archive_root=tmp_path)
        return list(asyncio.run(session_filter.list_all_summaries()))

    assert resolve(_EXECUTABLE) == []
    with pytest.raises(TypeError, match="unsupported Boolean query predicate"):
        resolve(_SEMANTIC)
