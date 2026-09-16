"""Real-route compilation tests for the declaration-generated agent manual."""

from __future__ import annotations

from collections.abc import Mapping

from polylogue.agent_integration.spec import (
    ALL_DECLARED_TOOLS,
    DEFAULT_READ_TOOLS,
    ORIGIN_MEANINGS,
    QUERY_EXAMPLES,
    RECIPES,
    TOOL_CONTRACT_BY_NAME,
    TOOL_CONTRACTS,
)
from polylogue.archive.query.discovery import QUERY_DISCOVERY_EXAMPLES
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
    from polylogue.mcp.declarations import MCP_TOOL_DECLARATION_BY_NAME

    declarations = set(MCP_TOOL_DECLARATION_BY_NAME)
    target_declarations = {item.name for item in (*TARGET_DEFAULT_READ_ALGEBRA, *PRIVILEGED_ALGEBRA)}
    assert target_declarations <= declarations

    assert tuple(contract.name for contract in TOOL_CONTRACTS) == ALL_DECLARED_TOOLS
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


def test_no_manual_query_expression_is_written_outside_the_discovery_corpus() -> None:
    """Mutation: hand-typing an ``expression`` argument into a recipe step, or teaching a
    query whose text is not a declared discovery row, makes this red."""
    declared = {example.expression for example in QUERY_DISCOVERY_EXAMPLES}
    declared_keys = {example.key for example in QUERY_DISCOVERY_EXAMPLES}

    for query in QUERY_EXAMPLES:
        assert query.declaration_id in declared_keys
        assert query.expression in declared

    for recipe in RECIPES:
        for step in recipe.steps:
            arguments = step.arguments_dict()
            if "expression" not in arguments:
                continue
            assert step.example_key in declared_keys, (
                f"{recipe.id}: step carries a hand-written expression instead of a declaration id"
            )
            assert arguments["expression"] in declared
        assert set(recipe.query_declarations) <= declared_keys


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

    assert token.startswith("q2.")
    assert decoded.request.operation == "query"
    assert decoded.request.offset == 20
    assert decoded.request.page_size == 20
    assert decoded.request.arguments == {
        "expression": "actions where action:file_edit AND path:polylogue/archive/query | sort by time desc | limit 20",
        "projection": "action-evidence",
    }
    assert decoded.result_ref == "result:0123456789abcdef01234567"


def test_origin_teaching_follows_authoritative_enum() -> None:
    """Mutation: using a stale origin count or omitting an Origin token fails here."""
    assert tuple(item.token for item in ORIGIN_MEANINGS) == tuple(item.value for item in Origin)
    assert len(ORIGIN_MEANINGS) == len(Origin)
    # beads-issue stays in the enum: its route is retired, but the durable
    # CHECKs still admit the token, so the teaching table names it as reserved.
    assert {item.token for item in ORIGIN_MEANINGS} >= {"beads-issue"}


def test_generated_contract_arguments_match_the_live_mcp_signatures() -> None:
    """Every documented tool argument must exist on the registered MCP handler.

    The generated manual advertised ten arguments the privileged ``maintenance``
    tool never accepted (``targets``, ``dry_run``, ``session_ids``, ``origin``,
    ``source_family``, ``source_root``, ``since``, ``until``, ``failure_kind``,
    ``parser_version``) plus a typed ``operation="preview"`` example, because the
    only signature comparison lived in a lane gated behind ``--require-live``.

    Anti-vacuity: adding a phantom argument to any contract in
    ``agent_integration/spec.py``, or dropping one the live handler declares,
    makes this red. Verified by reverting the maintenance-contract fix.
    """
    import inspect

    from polylogue.mcp.declarations import MCPCapabilities
    from polylogue.mcp.server import build_server

    server = build_server(capabilities=MCPCapabilities(write=True, judge=True, maintenance=True))
    # The registered tool surface is the authority for the live signature.
    tools = server._tool_manager._tools

    problems: list[str] = []
    for contract in TOOL_CONTRACTS:
        surface = tools.get(contract.name)
        if surface is None:
            problems.append(f"{contract.name}: no registered MCP tool")
            continue
        signature = inspect.signature(surface.fn)
        live = set(signature.parameters)
        documented = set(contract.argument_names)
        if documented - live:
            problems.append(f"{contract.name}: documented but absent live {sorted(documented - live)}")
        if live - documented:
            problems.append(f"{contract.name}: live but undocumented {sorted(live - documented)}")
        live_required = {
            name
            for name, param in signature.parameters.items()
            if param.default is inspect.Parameter.empty
            and param.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        }
        if live_required != set(contract.required_initial_arguments):
            problems.append(
                f"{contract.name}: required-argument mismatch "
                f"(live={sorted(live_required)}, manual={sorted(contract.required_initial_arguments)})"
            )

    assert not problems, problems


def _live_maintenance_signature() -> tuple[frozenset[str], frozenset[str]]:
    """Return the live ``maintenance`` operation vocabulary and parameter names."""
    import inspect
    import typing

    from polylogue.mcp.declarations import MCPCapabilities
    from polylogue.mcp.server import build_server

    server = build_server(capabilities=MCPCapabilities(write=True, judge=True, maintenance=True))
    fn = server._tool_manager._tools["maintenance"].fn
    signature = inspect.signature(fn)
    hints = typing.get_type_hints(fn)
    operations = frozenset(typing.get_args(hints["operation"]))
    assert operations, "maintenance operation annotation is not a Literal vocabulary"
    return operations, frozenset(signature.parameters)


def test_declared_confirmation_gate_matches_the_live_maintenance_handler() -> None:
    """The one declared gate must be the handler's actual gate.

    Anti-vacuity: adding, renaming, or removing a ``maintenance`` operation
    literal, or renaming/removing the ``confirm`` parameter, without updating
    ``_MAINTENANCE_CONFIRMATION`` in ``agent_integration/spec.py`` makes this
    red. Verified by adding a ``preview`` literal to the live handler.
    """
    operations, parameters = _live_maintenance_signature()
    contract = TOOL_CONTRACT_BY_NAME["maintenance"]
    gate = contract.confirmation

    assert gate is not None
    assert gate.argument in parameters
    assert not set(gate.operations) & set(gate.inspection_operations)
    assert frozenset((*gate.operations, *gate.inspection_operations)) == operations

    operation_argument = next(argument for argument in contract.arguments if argument.name == "operation")
    assert frozenset(operation_argument.enum_values) == operations


def test_manual_gate_prose_resolves_against_the_live_handler() -> None:
    """The manual must describe the maintenance gate exactly once, from the declaration.

    The manual used to instruct callers to "call maintenance with the declared
    operation in preview/dry-run mode" and to call ``confirm=true`` a "legacy"
    boolean that "is not the canonical gate", while the handler has no preview
    or dry-run operation and gates precisely on ``confirm``. An agent following
    that prose was refused.

    Anti-vacuity: red if the manual reintroduces preview/dry-run flow prose,
    names a maintenance operation the live handler does not declare, calls the
    live gate legacy, or stops rendering the gate sentence from the declared
    ``ConfirmationGate``. Verified by restoring the two original prose lines.
    """
    import re

    from devtools.render_agent_manual import _maintenance_gate_sentence, render_standing_manual

    operations, parameters = _live_maintenance_signature()
    gate = TOOL_CONTRACT_BY_NAME["maintenance"].confirmation
    assert gate is not None
    manual = render_standing_manual()
    sentence = _maintenance_gate_sentence()

    assert sentence in manual
    for operation in operations:
        assert f"`{operation}`" in sentence
    assert f"`{gate.argument}=true`" in sentence

    # Every backticked token in the gate sentence must be a live operation, a
    # live parameter, or the confirmation assignment itself.
    resolvable = {*operations, *parameters, f"{gate.argument}=true", "maintenance"}
    assert set(re.findall(r"`([^`]+)`", sentence)) <= resolvable

    # Flow vocabulary the handler does not implement may appear only inside the
    # generated sentence's own denial, and nowhere else in the manual.
    for word in ("preview", "dry-run", "dry_run", "legacy"):
        assert manual.count(word) == sentence.count(word), (
            f"{word!r} appears in manual prose outside the generated gate sentence"
        )


def test_rendered_manual_enumerates_exactly_the_declared_tool_surface() -> None:
    """The manual's tool table must equal declared_tool_names(ALL_CAPABILITIES).

    The manual claimed a "ten-tool surface" in five places while the registry
    declared twelve, so the standing manual injected into agent sessions told
    agents that ``record_work_event`` and ``emit_decision`` -- live, registered,
    write-gated handlers -- did not exist. The count is now derived; nothing
    restates it.

    Anti-vacuity: adding a tool row to
    ``polylogue/mcp/declarations/registry.py`` makes this red until the manual
    renders it, and the rendered count word changes with it. Verified by adding
    a thirteenth declaration row and re-rendering.
    """
    import re

    from devtools.render_agent_manual import render_standing_manual
    from polylogue.mcp.declarations import declared_tool_names

    declared = declared_tool_names()
    manual = render_standing_manual()

    heading = re.search(r"^## The (\S+) tools$", manual, re.MULTILINE)
    assert heading is not None, "the manual no longer enumerates its tool surface"
    table = manual.split(heading.group(0), 1)[1].split("\n## ", 1)[0]
    enumerated = tuple(re.findall(r"^\| `([a-z_]+)` \|", table, re.MULTILINE))

    assert frozenset(enumerated) == declared
    assert len(enumerated) == len(declared)
    assert enumerated == ALL_DECLARED_TOOLS

    # The spelled count and the surface sentence are derived from the same set.
    from devtools.render_agent_manual import _count_word

    assert heading.group(1) == _count_word(len(declared))
    assert f"{_count_word(len(declared))} tools" in manual

    # No stale literal count survives anywhere in the manual.
    for stale in ("ten-tool", "ten tools", "six-tool", "six tools"):
        assert stale not in manual


def test_non_target_tools_are_declared_tools_with_manual_contracts() -> None:
    """``target_visible=False`` must not read as "not a tool".

    Anti-vacuity: dropping either write-gated event tool from ``TOOL_CONTRACTS``
    makes the spec's declaration-order invariant raise at import; giving either
    one a target transaction makes the target-algebra assertion below red.
    """
    from polylogue.mcp.declarations import MCP_TOOL_DECLARATION_BY_NAME, declared_tool_names

    event_tools = ("record_work_event", "emit_decision")
    target_names = {item.name for item in (*TARGET_DEFAULT_READ_ALGEBRA, *PRIVILEGED_ALGEBRA)}

    for name in event_tools:
        declaration = MCP_TOOL_DECLARATION_BY_NAME[name]
        assert declaration.transaction is None
        assert name not in target_names
        assert name in declared_tool_names()
        assert name in TOOL_CONTRACT_BY_NAME
        assert TOOL_CONTRACT_BY_NAME[name].required_capability == "write"
