"""Independent equivalence checks for the executable MCP declaration algebra."""

from __future__ import annotations

import ast
import asyncio
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest

from devtools.render_mcp_equivalence import render_output
from polylogue.mcp.declarations.adapter import (
    DeclaredToolRegistrar,
    MCPRegistrationError,
    register_declared_handler,
)
from polylogue.mcp.declarations.models import MCPResultSemantics, MCPVerb, mcp_role_allows
from polylogue.mcp.declarations.registry import (
    MCP_TOOL_DECLARATION_BY_NAME,
    MCP_TOOL_DECLARATIONS,
    PRIVILEGED_ALGEBRA,
    TARGET_DEFAULT_READ_ALGEBRA,
    TARGET_PROMPTS,
    TARGET_RESOURCES,
    declared_tool_names,
)


class _RecordingToolRegistrar:
    def __init__(self) -> None:
        self.handlers: list[Callable[..., Any]] = []
        self.kwargs: list[dict[str, object]] = []

    def tool(
        self,
        name: str | None = None,
        *,
        description: str | None = None,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        self.kwargs.append({"name": name, "description": description})

        def register(handler: Callable[..., Any]) -> Callable[..., Any]:
            self.handlers.append(handler)
            return handler

        return register


def _assignment_value(path: str, name: str) -> ast.expr:
    tree = ast.parse(Path(path).read_text(encoding="utf-8"), filename=path)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return node.value
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == name:
            assert node.value is not None
            return node.value
    raise AssertionError(f"{path} has no assignment for {name}")


def _frozen_name_baseline() -> frozenset[str]:
    value = _assignment_value("tests/infra/mcp.py", "MCP_TOOL_NAME_BASELINE")
    assert isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == "frozenset"
    assert len(value.args) == 1
    names = ast.literal_eval(value.args[0])
    assert isinstance(names, set)
    return frozenset(cast(set[str], names))


def _contract_value(node: ast.expr) -> str | tuple[str, frozenset[str]]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    assert isinstance(node, ast.Tuple) and len(node.elts) == 2
    kind = ast.literal_eval(node.elts[0])
    field_call = node.elts[1]
    assert isinstance(field_call, ast.Call) and isinstance(field_call.func, ast.Name)
    assert field_call.func.id == "frozenset" and len(field_call.args) == 1
    fields = ast.literal_eval(field_call.args[0])
    assert isinstance(kind, str) and isinstance(fields, set)
    return kind, frozenset(cast(set[str], fields))


def _frozen_output_contract() -> dict[str, str | tuple[str, frozenset[str]]]:
    value = _assignment_value("tests/unit/mcp/test_envelope_contracts.py", "TOOL_CONTRACT")
    assert isinstance(value, ast.Dict)
    result: dict[str, str | tuple[str, frozenset[str]]] = {}
    for key_node, value_node in zip(value.keys, value.values, strict=True):
        assert key_node is not None
        name = ast.literal_eval(key_node)
        assert isinstance(name, str)
        result[name] = _contract_value(value_node)
    return result


def _handler_for(name: str) -> Callable[[], None]:
    declaration = MCP_TOOL_DECLARATION_BY_NAME[name]

    def handler() -> None:
        return None

    handler.__name__ = name
    handler.__qualname__ = name
    handler.__module__ = declaration.registration.module
    return handler


def test_every_frozen_discovery_name_has_exactly_one_declaration() -> None:
    """Production dependency: MCP_TOOL_DECLARATIONS and role-derived names.

    Anti-vacuity mutation: deleting the same tool from production registration
    and the declaration still fails against the independently parsed baseline.
    """

    baseline = _frozen_name_baseline()
    assert len(baseline) == 104
    assert len(MCP_TOOL_DECLARATIONS) == 104
    assert len(MCP_TOOL_DECLARATION_BY_NAME) == 104
    assert declared_tool_names("admin") == baseline


def test_output_contracts_match_the_independent_envelope_matrix() -> None:
    """Production dependency: declaration output kind and envelope fields.

    Anti-vacuity mutation: changing a declaration and generated map together
    still fails against ``TOOL_CONTRACT`` in the existing runtime tests.
    """

    baseline = _frozen_output_contract()
    assert set(baseline) == set(MCP_TOOL_DECLARATION_BY_NAME)
    for name, expected in baseline.items():
        declaration = MCP_TOOL_DECLARATION_BY_NAME[name]
        expected_kind = expected[0] if isinstance(expected, tuple) else expected
        expected_fields = expected[1] if isinstance(expected, tuple) else frozenset()
        assert declaration.output_contract.kind == expected_kind, name
        assert frozenset(declaration.output_contract.envelope_fields) == expected_fields, name


def test_role_gates_and_migration_groups_are_exhaustive_and_disjoint() -> None:
    """Production dependency: role lattice and t46.8.2/t46.8.3 ownership.

    Anti-vacuity mutation: widening one role or assigning one tool to both
    migration groups changes the exact sets and counts asserted here.
    """

    role_counts = {
        role: sum(declaration.minimum_role == role for declaration in MCP_TOOL_DECLARATIONS)
        for role in ("read", "write", "review", "admin")
    }
    assert role_counts == {"read": 66, "write": 29, "review": 2, "admin": 7}
    assert len(declared_tool_names("read")) == 66
    assert len(declared_tool_names("write")) == 95
    assert len(declared_tool_names("review")) == 97
    assert len(declared_tool_names("admin")) == 104

    read_group = {
        declaration.name for declaration in MCP_TOOL_DECLARATIONS if declaration.retirement_owner == "polylogue-t46.8.2"
    }
    privileged_group = {
        declaration.name for declaration in MCP_TOOL_DECLARATIONS if declaration.retirement_owner == "polylogue-t46.8.3"
    }
    assert read_group == set(declared_tool_names("read"))
    assert read_group.isdisjoint(privileged_group)
    assert read_group | privileged_group == set(declared_tool_names("admin"))


def test_target_algebra_preserves_semantic_dimensions_without_self_authorizing() -> None:
    """Production dependency: bounded target transaction/resource/prompt records.

    Anti-vacuity mutation: removing query result semantics, resource authority,
    or prompt non-authority violates these explicit target constraints.
    """

    assert len(TARGET_DEFAULT_READ_ALGEBRA) == 7
    assert {item.name for item in TARGET_DEFAULT_READ_ALGEBRA} == {
        "query",
        "read",
        "get",
        "graph",
        "explain",
        "context",
        "status",
    }
    query = next(item for item in TARGET_DEFAULT_READ_ALGEBRA if item.name == "query")
    assert set(query.result_semantics) == {
        MCPResultSemantics.EXHAUSTIVE_PAGE,
        MCPResultSemantics.TOP_K,
        MCPResultSemantics.SAMPLE,
        MCPResultSemantics.AGGREGATE,
    }
    assert {item.name for item in PRIVILEGED_ALGEBRA} == {"write", "judge", "run", "maintenance"}
    assert {item.verb for item in PRIVILEGED_ALGEBRA} == {
        MCPVerb.WRITE,
        MCPVerb.JUDGE,
        MCPVerb.RUN,
        MCPVerb.MAINTENANCE,
    }
    assert len(TARGET_RESOURCES) == 8
    assert all("read-only" in item.authority and "never" in item.authority for item in TARGET_RESOURCES)
    assert TARGET_PROMPTS and all(item.mutation_authority == "none" for item in TARGET_PROMPTS)


def test_registration_owners_are_source_locatable() -> None:
    """Production dependency: declaration module/registrar ownership.

    Anti-vacuity mutation: moving a registrar or mistyping an owner module in
    the inventory fails without importing the dependency-heavy MCP runtime.
    """

    for declaration in MCP_TOOL_DECLARATIONS:
        module_path = Path(*declaration.registration.module.split(".")).with_suffix(".py")
        assert module_path.is_file(), declaration.name
        tree = ast.parse(module_path.read_text(encoding="utf-8"), filename=str(module_path))
        top_level_functions = {
            node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        assert declaration.registration.registrar in top_level_functions, declaration.name
        assert declaration.kernel.handlers[0].owner_path == module_path.as_posix(), declaration.name


def test_python_parity_and_operation_ownership_are_explicit_for_every_tool() -> None:
    """Production dependency: operation_owner and PythonParityExpectation.

    Anti-vacuity mutation: clearing an owner or representing an absence without
    Bead authority breaks the exact-one branch assertions.
    """

    for declaration in MCP_TOOL_DECLARATIONS:
        assert declaration.operation_owner
        parity = declaration.python_parity
        assert (parity.binding is not None) != (parity.intentional_absence_authority is not None)
        if parity.binding is None:
            assert parity.intentional_absence_authority == "polylogue-s1kr"
            assert parity.reason


def test_declared_registrar_forwards_original_handlers_and_enforces_role_surface() -> None:
    """Production dependency: DeclaredToolRegistrar decorator/finalize path.

    Anti-vacuity mutation: wrapping handlers, skipping role validation, or not
    checking the final set makes one of the identity/error assertions fail.
    """

    delegate = _RecordingToolRegistrar()
    registrar = DeclaredToolRegistrar(delegate, role="read")
    handlers: list[Callable[[], None]] = []
    for name in sorted(declared_tool_names("read")):
        handler = _handler_for(name)
        handlers.append(handler)
        registered = registrar.tool()(handler)
        assert registered is handler
        assert delegate.handlers[-1] is handler
    registrar.finalize()
    assert delegate.handlers == handlers
    assert registrar.registered_names == declared_tool_names("read")

    unauthorized = DeclaredToolRegistrar(_RecordingToolRegistrar(), role="read")
    with pytest.raises(MCPRegistrationError, match="requires 'write'"):
        unauthorized.tool()(_handler_for("add_tag"))

    unknown = _handler_for("search")
    unknown.__name__ = "not_a_live_tool"
    with pytest.raises(MCPRegistrationError, match="no executable declaration"):
        unauthorized.tool()(unknown)


def test_declaration_owned_registration_metadata_replaces_one_off_specs() -> None:
    """Production dependency: register_declared_handler production seam.

    Anti-vacuity mutation: restoring the deleted hand-authored read-tool spec or
    ignoring the declaration description makes the metadata assertion fail.
    """

    delegate = _RecordingToolRegistrar()
    registrar = DeclaredToolRegistrar(delegate, role="read")
    handler = _handler_for("list_read_view_profiles")
    handler.__name__ = "private_handler"
    handler.__doc__ = None

    registered = register_declared_handler(registrar, handler, name="list_read_view_profiles")

    declaration = MCP_TOOL_DECLARATION_BY_NAME["list_read_view_profiles"]
    assert registered is handler
    assert handler.__name__ == declaration.name
    assert handler.__doc__ == declaration.description
    assert delegate.handlers == [handler]


def test_live_fastmcp_registration_matches_declarations(mcp_server: object) -> None:
    """Production dependency: build_server -> DeclaredToolRegistrar -> FastMCP.

    Anti-vacuity mutation: removing the adapter, wrapping a handler, changing a
    required parameter, or registering from the wrong module fails this real
    server-route comparison.
    """

    server = cast(Any, mcp_server)
    published = {tool.name: tool for tool in asyncio.run(server.list_tools())}
    assert set(published) == set(declared_tool_names("admin"))
    for name, declaration in MCP_TOOL_DECLARATION_BY_NAME.items():
        tool = published[name]
        schema = cast(dict[str, object], tool.inputSchema)
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        assert isinstance(properties, dict), name
        assert isinstance(required, list), name
        assert frozenset(cast(list[str], required)) == frozenset(declaration.input_contract.required_arguments), name
        assert set(declaration.minimal_arguments_dict()) <= set(properties), name
        live_handler = server._tool_manager._tools[name].fn
        assert live_handler.__module__ == declaration.registration.module, name


def test_role_lattice_matches_declared_visibility() -> None:
    assert mcp_role_allows("admin", "read")
    assert mcp_role_allows("review", "write")
    assert not mcp_role_allows("read", "write")


def test_generated_equivalence_map_is_current_and_complete() -> None:
    """Production dependency: renderer over MCP_TOOL_DECLARATIONS.

    Anti-vacuity mutation: changing any descriptor without rerendering changes
    ``render_output`` and fails the byte-for-byte artifact check.
    """

    path = Path("docs/generated/mcp-equivalence.json")
    assert path.read_text(encoding="utf-8") == render_output()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["compatibility_surface"]["tool_count"] == 104
    assert len(payload["tools"]) == 104
