"""Generated reference-model differentials over the public query surfaces."""

from __future__ import annotations

from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings

from tests.infra.corpus_program import CorpusProgram, ProductionCorpusRuntime
from tests.infra.reference_differential import (
    QuerySemanticFacts,
    assert_query_facts_equal,
    codex_query_program_strategy,
    differential_for_program,
)


@pytest.mark.parametrize(
    "expression",
    (
        "planted",
        "planted request",
        "no-such-generated-query-token",
    ),
)
@settings(
    max_examples=1,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(program=codex_query_program_strategy(max_sessions=2))
def test_generated_corpus_program_matches_reference_model_across_surfaces(
    tmp_path: Path,
    expression: str,
    program: CorpusProgram,
) -> None:
    """Production admission and all three public query adapters share facts."""
    runtime = ProductionCorpusRuntime(tmp_path / "archive")
    program.run(runtime)

    facts = differential_for_program(runtime.archive_root, program, expression)
    expected = facts[0]
    for surface, actual in zip(("API", "CLI", "MCP"), facts[1:], strict=True):
        assert_query_facts_equal(expected, actual, surface=surface)
    assert expected.count == len(expected.session_ids)


def test_seeded_semantic_divergence_is_rejected() -> None:
    expected = QuerySemanticFacts(session_ids=("codex-session:a",), count=1)
    divergent = QuerySemanticFacts(session_ids=(), count=0)

    with pytest.raises(AssertionError, match="CLI query facts disagree"):
        assert_query_facts_equal(expected, divergent, surface="CLI")
