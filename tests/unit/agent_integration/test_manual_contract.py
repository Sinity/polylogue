"""Real-route compilation tests for the six-tool-era generated manual."""

from __future__ import annotations

from collections.abc import Mapping

from polylogue.agent_integration.spec import (
    ALL_TARGET_TOOLS,
    DEFAULT_READ_TOOLS,
    ORIGIN_MEANINGS,
    QUERY_EXAMPLES,
    RECIPES,
    TOOL_CONTRACT_BY_NAME,
    TOOL_CONTRACTS,
)
from polylogue.archive.query.expression import compile_expression, explain_expression, parse_unit_source_expression
from polylogue.archive.query.transaction import QueryContinuation
from polylogue.cli.query_group import _looks_like_query_expression, _split_query_mode_args
from polylogue.core.enums import Origin
from polylogue.mcp.declarations import PRIVILEGED_ALGEBRA, TARGET_DEFAULT_READ_ALGEBRA


def _assert_call_compiles(tool: str, arguments: Mapping[str, object]) -> None:
    contract = TOOL_CONTRACT_BY_NAME[tool]
    assert set(arguments) <= set(contract.argument_names)
    if set(arguments) == {"continuation"}:
        assert contract.supports_continuation
        return
    assert "continuation" not in arguments
    assert set(contract.required_initial_arguments) <= set(arguments)


def test_typed_examples_and_recipes_resolve_against_target_declarations() -> None:
    """Mutation: deleting a declaration mapping or renaming a manual argument makes compilation fail."""
    declarations = {item.name for item in (*TARGET_DEFAULT_READ_ALGEBRA, *PRIVILEGED_ALGEBRA)}

    assert tuple(contract.name for contract in TOOL_CONTRACTS) == ALL_TARGET_TOOLS
    for contract in TOOL_CONTRACTS:
        assert set(contract.source_declarations) <= declarations
        assert contract.supports_continuation is ("continuation" in contract.argument_names)
        for example in contract.examples:
            _assert_call_compiles(contract.name, example.arguments_dict())
    for recipe in RECIPES:
        for step in recipe.steps:
            assert step.tool in DEFAULT_READ_TOOLS
            _assert_call_compiles(step.tool, step.arguments_dict())


def test_every_documented_query_round_trips_the_production_parser() -> None:
    """Mutation: changing expression grammar/field lowering invalidates the corresponding manual example."""
    for query in QUERY_EXAMPLES:
        explanation = explain_expression(query.expression)
        assert explanation.source_text == query.expression
        if query.surface == "terminal":
            source = parse_unit_source_expression(query.expression)
            assert source is not None
            assert source == parse_unit_source_expression(query.expression)
        else:
            compiled = compile_expression(query.expression)
            assert compiled is not None


def test_strict_command_floor_retains_all_three_query_intent_signals() -> None:
    """Mutation: accepting a bare word or dropping find/quoted/field intent breaks the taught CLI contract."""
    import click

    _, terms, _, explicit = _split_query_mode_args(click.Group(), ["find", "prior", "art"])

    assert explicit is True
    assert terms == ("prior", "art")
    assert _looks_like_query_expression(("prior art",)) is True
    assert _looks_like_query_expression(("repo:polylogue",)) is True
    assert _looks_like_query_expression(("prior",)) is False


def test_generated_continuation_token_decodes_to_the_bound_result() -> None:
    """Mutation: reconstructing filters, changing offset, or losing result_ref invalidates exact recovery."""
    from devtools.render_agent_manual import continuation_example_token

    token = continuation_example_token()
    decoded = QueryContinuation.decode(token)

    assert token.startswith("q1.")
    assert decoded.request.operation == "query"
    assert decoded.request.offset == 20
    assert decoded.request.page_size == 20
    assert decoded.request.arguments == {
        "expression": "actions where action:file_edit AND path:polylogue/archive/query | sort by time desc | limit 20",
        "projection": "action-evidence",
    }
    assert decoded.result_ref == "result:0123456789abcdef01234567"


def test_origin_teaching_follows_authoritative_enum_including_beads_issue() -> None:
    """Mutation: using the mission's stale count of ten or omitting a new Origin token fails here."""
    assert tuple(item.token for item in ORIGIN_MEANINGS) == tuple(item.value for item in Origin)
    assert len(ORIGIN_MEANINGS) == 11
    assert "beads-issue" in {item.token for item in ORIGIN_MEANINGS}
