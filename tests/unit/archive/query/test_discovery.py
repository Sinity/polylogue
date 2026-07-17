"""Anti-lying gate for executable query discovery declarations.

The production dependency is ``polylogue.archive.query.expression``: every
positive row must traverse its real Lark parser and hand-written pipeline
splitter, while every negative row pins the exact ``ExpressionCompileError``
diagnostic. Removing a grammar terminal, bypassing ``parse_unit_source_expression``,
or changing pipeline diagnostics must make this module fail.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import cast

import pytest

from polylogue.archive.query.discovery import (
    QUERY_DISCOVERY_EXAMPLES,
    QUERY_DISCOVERY_NEGATIVE_EXAMPLES,
    RESULT_SEMANTICS_TEACHING,
    QueryDiscoveryExample,
    QueryDiscoveryNegativeExample,
    query_coverage_classes,
    query_discovery_catalog_payload,
    query_discovery_example,
    query_discovery_examples,
    render_query_discovery_example,
)
from polylogue.archive.query.expression import (
    ExpressionCompileError,
    compile_expression,
    parse_unit_source_expression,
)
from polylogue.archive.query.metadata import query_unit_descriptor, query_unit_descriptors
from polylogue.archive.query.transaction import QueryCoverageClass, QueryResultSemanticsContract


def _parse_positive(row: QueryDiscoveryExample) -> object:
    if row.parser == "session":
        return compile_expression(row.expression)
    source = parse_unit_source_expression(row.expression)
    assert source is not None, f"{row.key} did not produce a terminal source"
    descriptor = query_unit_descriptor(row.unit_source)
    assert descriptor is not None
    assert source.unit == descriptor.unit
    return source


def _parse_corrected(row: QueryDiscoveryNegativeExample) -> object:
    if row.parser == "session":
        return compile_expression(row.corrected_form)
    source = parse_unit_source_expression(row.corrected_form)
    assert source is not None, f"{row.key} correction did not produce a terminal source"
    return source


@pytest.mark.parametrize("row", QUERY_DISCOVERY_EXAMPLES, ids=lambda row: row.key)
def test_every_positive_example_parses_through_the_real_production_route(row: QueryDiscoveryExample) -> None:
    """Deleting a real grammar or splitter branch makes the corresponding corpus row fail."""

    _parse_positive(row)


@pytest.mark.parametrize("row", QUERY_DISCOVERY_NEGATIVE_EXAMPLES, ids=lambda row: row.key)
def test_every_negative_example_pins_the_real_diagnostic_and_a_valid_correction(
    row: QueryDiscoveryNegativeExample,
) -> None:
    """Weakening a rejection or changing its typed diagnostic requires an explicit corpus update."""

    parser = compile_expression if row.parser == "session" else parse_unit_source_expression
    with pytest.raises(ExpressionCompileError) as caught:
        parser(row.expression)

    error = caught.value
    assert type(error).__name__ == row.diagnostic_class
    assert str(error) == row.diagnostic
    assert error.field == row.field
    _parse_corrected(row)


def test_corpus_shape_covers_the_required_semantics_units_and_costs() -> None:
    assert 80 <= len(QUERY_DISCOVERY_EXAMPLES) <= 150
    assert 10 <= len(QUERY_DISCOVERY_NEGATIVE_EXAMPLES) <= 20
    assert len({row.key for row in QUERY_DISCOVERY_EXAMPLES}) == len(QUERY_DISCOVERY_EXAMPLES)
    assert len({row.key for row in QUERY_DISCOVERY_NEGATIVE_EXAMPLES}) == len(QUERY_DISCOVERY_NEGATIVE_EXAMPLES)

    expected_semantics: set[QueryCoverageClass] = {
        "exhaustive",
        "top-k",
        "sample",
        "aggregate",
        "bounded-context",
        "recursive-page",
    }
    assert set(query_coverage_classes()) == expected_semantics
    assert {row.result_semantics for row in QUERY_DISCOVERY_EXAMPLES} == expected_semantics
    assert {row.cost_class for row in QUERY_DISCOVERY_EXAMPLES} == {"selective", "corpus-scale"}
    assert {row.unit_source for row in QUERY_DISCOVERY_EXAMPLES} == {
        "sessions",
        *(descriptor.plural_source for descriptor in query_unit_descriptors(terminal_supported=True)),
    }

    counts = Counter(row.result_semantics for row in QUERY_DISCOVERY_EXAMPLES)
    assert all(counts[semantics] >= 5 for semantics in expected_semantics)
    assert all(query_discovery_examples(result_semantics=semantics) for semantics in expected_semantics)


def test_rows_are_typed_one_sentence_provider_neutral_and_privacy_safe() -> None:
    forbidden_tokens = (
        "claude",
        "codex",
        "chatgpt",
        "openai",
        "anthropic",
        "gemini",
        "github.com/",
        "/home/",
        "/users/",
    )
    email = re.compile(r"\b[^\s@]+@[^\s@]+\.[^\s@]+\b")

    for row in QUERY_DISCOVERY_EXAMPLES:
        assert row.answers.endswith(".")
        assert "\n" not in row.answers
        assert row.answers.count(". ") == 0
        assert row.projection_columns
        assert len(set(row.projection_columns)) == len(row.projection_columns)
        haystack = " ".join((row.expression, row.answers, *row.projection_columns)).lower()
        assert not any(token in haystack for token in forbidden_tokens)
        assert email.search(haystack) is None

    for negative in QUERY_DISCOVERY_NEGATIVE_EXAMPLES:
        haystack = " ".join((negative.expression, negative.corrected_form, negative.diagnostic)).lower()
        assert not any(token in haystack for token in forbidden_tokens)
        assert email.search(haystack) is None


def test_declared_projection_columns_track_public_row_payload_models() -> None:
    from polylogue.surfaces.payloads import (
        ActionQueryRowPayload,
        AssertionQueryRowPayload,
        BlockQueryRowPayload,
        ContextSnapshotQueryRowPayload,
        DelegationQueryRowPayload,
        FileQueryRowPayload,
        MessageQueryRowPayload,
        ObservedEventQueryRowPayload,
        QueryUnitAggregateRowPayload,
        RunQueryRowPayload,
        SessionListRowPayload,
        SurfacePayloadModel,
    )

    models: dict[str, type[SurfacePayloadModel]] = {
        "messages": MessageQueryRowPayload,
        "actions": ActionQueryRowPayload,
        "blocks": BlockQueryRowPayload,
        "assertions": AssertionQueryRowPayload,
        "files": FileQueryRowPayload,
        "runs": RunQueryRowPayload,
        "observed-events": ObservedEventQueryRowPayload,
        "context-snapshots": ContextSnapshotQueryRowPayload,
        "delegations": DelegationQueryRowPayload,
    }
    assert query_discovery_example("session-repository").projection_columns == tuple(SessionListRowPayload.model_fields)
    aggregate_fields = tuple(QueryUnitAggregateRowPayload.model_fields)
    for row in QUERY_DISCOVERY_EXAMPLES:
        if row.result_semantics == "aggregate":
            assert row.projection_columns == aggregate_fields
        elif row.result_semantics == "exhaustive" and row.unit_source != "sessions":
            assert row.projection_columns == tuple(models[row.unit_source].model_fields)


def test_result_semantics_contracts_are_complete_truthful_and_unique() -> None:
    assert all(isinstance(contract, QueryResultSemanticsContract) for contract in RESULT_SEMANTICS_TEACHING)
    assert len({contract.coverage for contract in RESULT_SEMANTICS_TEACHING}) == 6
    assert all(contract.phrase.endswith(".") for contract in RESULT_SEMANTICS_TEACHING)

    contracts = {contract.coverage: contract for contract in RESULT_SEMANTICS_TEACHING}
    assert contracts["exhaustive"].total == "exact"
    assert contracts["exhaustive"].continuation == "cursor-or-offset"
    assert contracts["top-k"].total == "qualified"
    assert "not exhaustive" in contracts["top-k"].phrase.lower()
    assert contracts["sample"].total == "qualified"
    assert contracts["aggregate"].total == "aggregate"
    assert contracts["bounded-context"].continuation == "none"
    assert contracts["recursive-page"].continuation == "recursive-cursor"


def test_parameterized_examples_quote_dynamic_values_and_still_parse() -> None:
    values_by_kind: dict[str, object] = {
        "text": 'schema "migration"\nsecond line',
        "value": "path or repository with spaces",
        "date": "2026-06-30",
    }
    for row in QUERY_DISCOVERY_EXAMPLES:
        if not row.parameters:
            continue
        rendered = render_query_discovery_example(
            row.key,
            **{parameter.name: values_by_kind[parameter.kind] for parameter in row.parameters},
        )
        rendered_row = QueryDiscoveryExample(
            key=row.key,
            expression=rendered,
            parser=row.parser,
            unit_source=row.unit_source,
            answers=row.answers,
            result_semantics=row.result_semantics,
            projection_columns=row.projection_columns,
            cost_class=row.cost_class,
            route=row.route,
        )
        _parse_positive(rendered_row)


def test_parameterized_examples_reject_missing_or_unexpected_arguments() -> None:
    with pytest.raises(ValueError, match="missing=topic"):
        render_query_discovery_example("ranked-semantic-text")
    with pytest.raises(ValueError, match="unexpected=extra"):
        render_query_discovery_example("ranked-semantic-text", topic="query", extra="value")
    with pytest.raises(ValueError, match="has no parameters"):
        render_query_discovery_example("session-repository", repo="example-repo")


def test_catalog_and_completion_payloads_project_the_same_declarations() -> None:
    from polylogue.archive.query.completions import query_completion_payload

    catalog = query_discovery_catalog_payload(view="featured")
    featured = query_discovery_examples(featured=True)
    assert catalog["positive_count"] == len(QUERY_DISCOVERY_EXAMPLES)
    assert catalog["negative_count"] == len(QUERY_DISCOVERY_NEGATIVE_EXAMPLES)
    assert [item["key"] for item in cast(list[dict[str, object]], catalog["examples"])] == [row.key for row in featured]

    capability = query_discovery_catalog_payload(view="capability")
    assert capability["view"] == "capability"
    assert {item["result_semantics"] for item in cast(list[dict[str, object]], capability["examples"])} == {
        "exhaustive",
        "top-k",
        "sample",
        "aggregate",
        "bounded-context",
        "recursive-page",
    }
    assert all(item["shipped_at"] for item in cast(list[dict[str, object]], capability["negative_examples"]))

    completion = query_completion_payload("example", incomplete="unacknowledged")
    candidates = cast(list[dict[str, object]], completion["candidates"])
    assert [candidate["value"] for candidate in candidates] == [
        query_discovery_example("actions-unacknowledged-failures").expression
    ]
    assert candidates[0]["source"] == "QUERY_DISCOVERY_EXAMPLES"
    assert "Exhaustive relation" in cast(str, candidates[0]["description"])

    error_completion = query_completion_payload("error", incomplete="repo:example-repo AND path")
    errors = cast(list[dict[str, object]], error_completion["candidates"])
    assert [error["value"] for error in errors] == ["files where repo:example-repo AND path:src/mcp/server.py"]
    assert errors[0]["insert"] == "files where session.repo:example-repo AND path:src/mcp/server.py"
    assert errors[0]["source"] == "QUERY_DISCOVERY_NEGATIVE_EXAMPLES"
