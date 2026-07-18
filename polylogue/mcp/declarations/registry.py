"""Executable MCP tool algebra inventory and equivalence ownership.

Every live tool appears exactly once: the ten-tool cutover surface
(``query``/``read``/``get``/``explain``/``context``/``status`` plus the
privileged ``write``/``judge``/``run``/``maintenance`` transactions).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from polylogue.declarations import (
    CompatibilityKey,
    CompletenessEdge,
    DeclarationRegistry,
    DeclarationSpec,
    ExampleSpec,
    HandlerBinding,
    JSONValue,
    OutputSpec,
    validate_registry,
)
from polylogue.mcp.declarations.models import (
    MCPContinuationContract,
    MCPDeprecationState,
    MCPHandlerBinding,
    MCPInputContract,
    MCPOutputContract,
    MCPPromptDeclaration,
    MCPResourceDeclaration,
    MCPResultSemantics,
    MCPRole,
    MCPToolDeclaration,
    MCPTransactionDeclaration,
    MCPVerb,
    PythonParityExpectation,
    mcp_role_allows,
)

_REPAIR_COMMAND = "devtools render mcp-equivalence"


@dataclass(frozen=True, slots=True)
class _ToolRow:
    name: str
    description: str
    module: str
    registrar: str
    minimum_role: MCPRole
    verb: MCPVerb
    object_kinds: tuple[str, ...]
    result_semantics: MCPResultSemantics
    schema_source: str
    required_arguments: tuple[str, ...]
    minimal_arguments: tuple[tuple[str, JSONValue], ...]
    output_kind: str
    envelope_fields: tuple[str, ...]
    operation_owner: str
    python_binding: str | None


def _compatibility(row: _ToolRow) -> CompatibilityKey:
    return CompatibilityKey(
        identity="mcp-tool:" + ",".join(row.object_kinds),
        lifecycle="registered-handler-retained",
        authority=f"mcp-role:{row.minimum_role}",
        access_result_shape=f"{row.verb.value}:{row.result_semantics.value}:{row.output_kind}",
        durability="transport-adapter; domain-owner-controls-durability",
    )


def _python_parity(row: _ToolRow) -> PythonParityExpectation:
    if row.python_binding is not None:
        return PythonParityExpectation(binding=row.python_binding)
    return PythonParityExpectation(
        intentional_absence_authority="polylogue-s1kr",
        reason=(
            "The current MCP compatibility handler binds a lower-level owner or transport-only projection; "
            "polylogue-s1kr owns any public Python facade addition and docs parity."
        ),
    )


_CUTOVER_TOOL_ROWS: Final[tuple[_ToolRow, ...]] = (
    _ToolRow(
        "query",
        "Execute a parser-owned terminal query page or resume its q2 continuation.",
        "polylogue.mcp.server_cutover",
        "register_cutover_read_tools",
        "read",
        MCPVerb.QUERY,
        ("query", "result-set"),
        MCPResultSemantics.EXHAUSTIVE_PAGE,
        "polylogue.mcp.server_cutover.query:inspect.signature",
        ("expression",),
        (("expression", "messages where text:needle"),),
        "envelope",
        ("items", "query_ref", "result_ref", "continuation"),
        "polylogue.api.Polylogue.query_units",
        "polylogue.api.Polylogue.query_units",
    ),
    _ToolRow(
        "read",
        "Read a stable archive URI or public ref through a declared view.",
        "polylogue.mcp.server_cutover",
        "register_cutover_read_tools",
        "read",
        MCPVerb.READ,
        ("object-ref", "evidence-ref"),
        MCPResultSemantics.EXHAUSTIVE_PAGE,
        "polylogue.mcp.server_cutover.read:inspect.signature",
        ("ref",),
        (("ref", "session:codex-session:demo"),),
        "envelope",
        ("ref",),
        "polylogue.api.Polylogue.resolve_ref",
        "polylogue.api.Polylogue.resolve_ref",
    ),
    _ToolRow(
        "get",
        "Resolve one exact stable object or evidence identity.",
        "polylogue.mcp.server_cutover",
        "register_cutover_read_tools",
        "read",
        MCPVerb.GET,
        ("object-ref",),
        MCPResultSemantics.SINGLE_OBJECT,
        "polylogue.mcp.server_cutover.get:inspect.signature",
        ("ref",),
        (("ref", "session:codex-session:demo"),),
        "single_object",
        ("ref",),
        "polylogue.api.Polylogue.resolve_ref",
        "polylogue.api.Polylogue.resolve_ref",
    ),
    _ToolRow(
        "explain",
        "Explain parser grammar, capabilities, refs, result semantics, or recovery.",
        "polylogue.mcp.server_cutover",
        "register_cutover_read_tools",
        "read",
        MCPVerb.EXPLAIN,
        ("query", "capability", "object-ref"),
        MCPResultSemantics.SINGLE_OBJECT,
        "polylogue.mcp.server_cutover.explain:inspect.signature",
        ("subject",),
        (("subject", "capability"),),
        "single_object",
        ("subject",),
        "polylogue.api.Polylogue.explain_query_expression",
        "polylogue.api.Polylogue.explain_query_expression",
    ),
    _ToolRow(
        "context",
        "Compile a policy-gated bounded context image with receipts.",
        "polylogue.mcp.server_cutover",
        "register_cutover_read_tools",
        "read",
        MCPVerb.CONTEXT,
        ("context-snapshot",),
        MCPResultSemantics.BOUNDED_CONTEXT,
        "polylogue.mcp.server_cutover.context:inspect.signature",
        ("intent",),
        (("intent", "resume"),),
        "single_object",
        ("receipt",),
        "polylogue.api.Polylogue.context_image_payload",
        "polylogue.api.Polylogue.context_image_payload",
    ),
    _ToolRow(
        "status",
        "Report compact archive authority and readiness status.",
        "polylogue.mcp.server_cutover",
        "register_cutover_read_tools",
        "read",
        MCPVerb.STATUS,
        ("status",),
        MCPResultSemantics.SINGLE_OBJECT,
        "polylogue.mcp.server_cutover.status:inspect.signature",
        ("scope",),
        (("scope", "archive"),),
        "single_object",
        ("archive",),
        "polylogue.storage.sqlite.archive_tiers.archive.ArchiveStore.stats",
        None,
    ),
    _ToolRow(
        "write",
        "Apply a declared mutation operation after shared authorization.",
        "polylogue.mcp.server_cutover",
        "register_cutover_privileged_tools",
        "write",
        MCPVerb.WRITE,
        ("object-ref", "assertion"),
        MCPResultSemantics.MUTATION,
        "polylogue.mcp.server_cutover.write:inspect.signature",
        ("operation",),
        (("operation", "add_tag"), ("session_id", "test:conv-mutation"), ("tag", "review")),
        "operation_result",
        (),
        "mutate-write",
        None,
    ),
    _ToolRow(
        "judge",
        "Accept, reject, defer, or supersede assertion candidates without collapsing candidate state.",
        "polylogue.mcp.server_cutover",
        "register_cutover_privileged_tools",
        "review",
        MCPVerb.JUDGE,
        ("assertion-candidate", "judgment"),
        MCPResultSemantics.MUTATION,
        "polylogue.mcp.server_cutover.judge:inspect.signature",
        (),
        (("candidate_ref", "assertion:contract-candidate"), ("decision", "accept")),
        "envelope",
        ("items", "applied_count", "failed_count"),
        "polylogue.api.Polylogue.judge_assertion_candidates",
        "polylogue.api.Polylogue.judge_assertion_candidates",
    ),
    _ToolRow(
        "run",
        "Execute a saved query or governed recipe ref.",
        "polylogue.mcp.server_cutover",
        "register_cutover_privileged_tools",
        "write",
        MCPVerb.RUN,
        ("saved-query", "recipe", "result-set"),
        MCPResultSemantics.EXHAUSTIVE_PAGE,
        "polylogue.mcp.server_cutover.run:inspect.signature",
        ("ref",),
        (("ref", "saved-view:contract-view"),),
        "envelope",
        (),
        "mutate-run",
        None,
    ),
    _ToolRow(
        "maintenance",
        "Preview, execute, list, and inspect maintenance operations.",
        "polylogue.mcp.server_cutover",
        "register_cutover_privileged_tools",
        "admin",
        MCPVerb.MAINTENANCE,
        ("maintenance-plan", "maintenance-operation"),
        MCPResultSemantics.MAINTENANCE,
        "polylogue.mcp.server_cutover.maintenance:inspect.signature",
        ("operation",),
        (("operation", "list"),),
        "operation_result",
        (),
        "polylogue.maintenance.planner.preview_backfill",
        None,
    ),
)


def _cutover_declaration(row: _ToolRow) -> MCPToolDeclaration:
    kernel = DeclarationSpec(
        declaration_id=f"mcp.tool.{row.name}",
        family_id=f"mcp.tool.{row.name}",
        public_name=row.name,
        owner_path="polylogue/mcp/declarations/registry.py",
        compatibility=_compatibility(row),
        producer=row.operation_owner,
        role_gate=f"mcp.role:{row.minimum_role}",
        schema_ref=row.schema_source,
        discovery_text=row.description,
        repair_command=_REPAIR_COMMAND,
        handlers=(
            HandlerBinding(
                surface="mcp",
                owner_path=f"{row.module.replace('.', '/')}.py",
                symbol=row.name,
                binding_key=f"{row.module}:{row.name}",
            ),
        ),
        outputs=(
            OutputSpec(
                name="runtime-contract",
                kind=row.output_kind,
                schema_ref="tests/unit/mcp/test_envelope_contracts.py::TOOL_CONTRACT",
                target_path=f"mcp://tool/{row.name}",
            ),
        ),
        examples=(
            ExampleSpec(
                name="minimal-valid-call",
                summary=f"Minimal cutover invocation for {row.name}.",
                arguments=row.minimal_arguments,
            ),
        ),
        completeness_edges=(
            CompletenessEdge(
                producer=f"mcp.tool.{row.name}",
                consumer="tests.infra.mcp.EXPECTED_TOOL_NAMES",
                kind="discovery-name-equivalence",
                owner_path="tests/infra/mcp.py",
            ),
        ),
    )
    is_read = row.minimum_role == "read"
    continuation = "cursor_or_offset" if row.name in {"query", "read"} else "none"
    return MCPToolDeclaration(
        kernel=kernel,
        name=row.name,
        description=row.description,
        verb=row.verb,
        object_kinds=row.object_kinds,
        minimum_role=row.minimum_role,
        capability=f"{row.minimum_role}:{row.verb.value}",
        result_semantics=row.result_semantics,
        canonical_plan=row.operation_owner,
        canonical_projection=f"{row.output_kind}:root",
        input_contract=MCPInputContract(
            schema_source=row.schema_source,
            schema_mode="FastMCP derives inputSchema from the cutover handler signature",
            required_arguments=row.required_arguments,
        ),
        output_contract=MCPOutputContract(kind=row.output_kind, envelope_fields=row.envelope_fields),
        minimal_arguments=row.minimal_arguments,
        grammar_discovery=("polylogue://capabilities/query",) if is_read else (),
        field_discovery=("polylogue://capabilities/query",) if is_read else (),
        value_discovery=("polylogue://capabilities/query",) if is_read else (),
        continuation=MCPContinuationContract(
            mode=continuation,
            continuation_ref="q2" if continuation != "none" else None,
            exhaustive_route="query" if row.name != "query" else None,
            notes="Continuation is opaque and must be the only resume input.",
        ),
        resource_alternatives=("polylogue://capabilities/query",) if is_read else (),
        prompt_alternatives=(),
        compatibility_route=row.name,
        workflow_coverage=("t8t-continuity", "z9gh-workflow-incident") if is_read else ("t46.8.3-privileged-contract",),
        incident_coverage=("z9gh-workflow-incident",) if is_read else (),
        observed_use="observed",
        telemetry_key=row.name,
        deprecation_state=MCPDeprecationState.RETAINED,
        retirement_owner="polylogue-t46.8.2" if is_read else "polylogue-t46.8.3",
        registration=MCPHandlerBinding(module=row.module, symbol=row.name, registrar=row.registrar),
        operation_owner=row.operation_owner,
        python_parity=_python_parity(row),
    )


MCP_TOOL_DECLARATIONS: Final[tuple[MCPToolDeclaration, ...]] = tuple(
    _cutover_declaration(row) for row in _CUTOVER_TOOL_ROWS
)
MCP_TOOL_DECLARATION_BY_NAME: Final[dict[str, MCPToolDeclaration]] = {
    declaration.name: declaration for declaration in MCP_TOOL_DECLARATIONS
}
if len(MCP_TOOL_DECLARATION_BY_NAME) != len(MCP_TOOL_DECLARATIONS):
    raise RuntimeError("duplicate MCP declaration name")

MCP_KERNEL_REGISTRY = DeclarationRegistry()
for _declaration in MCP_TOOL_DECLARATIONS:
    MCP_KERNEL_REGISTRY.register(_declaration.kernel)
_MCP_DIAGNOSTICS = validate_registry(MCP_KERNEL_REGISTRY)
if _MCP_DIAGNOSTICS:
    raise RuntimeError("incomplete MCP declaration registry: " + "; ".join(item.message for item in _MCP_DIAGNOSTICS))


def declaration_for_tool(name: str) -> MCPToolDeclaration:
    try:
        return MCP_TOOL_DECLARATION_BY_NAME[name]
    except KeyError as exc:
        raise KeyError(
            f"MCP tool {name!r} has no declaration; add mcp.tool.{name} in polylogue/mcp/declarations/registry.py "
            f"and run {_REPAIR_COMMAND}"
        ) from exc


def declared_tool_names(role: MCPRole = "admin") -> frozenset[str]:
    return frozenset(
        declaration.name for declaration in MCP_TOOL_DECLARATIONS if mcp_role_allows(role, declaration.minimum_role)
    )


TARGET_DEFAULT_READ_ALGEBRA: Final[tuple[MCPTransactionDeclaration, ...]] = (
    MCPTransactionDeclaration(
        name="query",
        verb=MCPVerb.QUERY,
        minimum_role="read",
        object_kinds=("query", "result-set"),
        result_semantics=(
            MCPResultSemantics.EXHAUSTIVE_PAGE,
            MCPResultSemantics.TOP_K,
            MCPResultSemantics.SAMPLE,
            MCPResultSemantics.AGGREGATE,
        ),
        purpose="Execute a declared DSL or typed plan with explicit result semantics and continuation.",
        migration_owner="polylogue-t46.8.2",
    ),
    MCPTransactionDeclaration(
        name="read",
        verb=MCPVerb.READ,
        minimum_role="read",
        object_kinds=("object-ref", "evidence-ref"),
        result_semantics=(
            MCPResultSemantics.SINGLE_OBJECT,
            MCPResultSemantics.EXHAUSTIVE_PAGE,
            MCPResultSemantics.BOUNDED_CONTEXT,
        ),
        purpose="Read any stable archive ref through a declared projection/view.",
        migration_owner="polylogue-t46.8.2",
    ),
    MCPTransactionDeclaration(
        name="get",
        verb=MCPVerb.GET,
        minimum_role="read",
        object_kinds=("object-ref",),
        result_semantics=(MCPResultSemantics.SINGLE_OBJECT,),
        purpose="Resolve one exact object identity when a generic read would add ambiguity.",
        migration_owner="polylogue-t46.8.2",
    ),
    MCPTransactionDeclaration(
        name="explain",
        verb=MCPVerb.EXPLAIN,
        minimum_role="read",
        object_kinds=("query", "object-ref", "capability"),
        result_semantics=(MCPResultSemantics.SINGLE_OBJECT,),
        purpose="Discover grammar, fields, values, plans, authority, and recovery routes.",
        migration_owner="polylogue-t46.8.2",
    ),
    MCPTransactionDeclaration(
        name="context",
        verb=MCPVerb.CONTEXT,
        minimum_role="read",
        object_kinds=("context-snapshot", "context-delivery"),
        result_semantics=(MCPResultSemantics.BOUNDED_CONTEXT,),
        purpose="Compile and retrieve policy-gated bounded context plus receipts.",
        migration_owner="polylogue-t46.8.3",
    ),
    MCPTransactionDeclaration(
        name="status",
        verb=MCPVerb.STATUS,
        minimum_role="read",
        object_kinds=("status", "receipt"),
        result_semantics=(MCPResultSemantics.SINGLE_OBJECT, MCPResultSemantics.AGGREGATE),
        purpose="Read archive, source, embedding, coordination, and operation status.",
        migration_owner="polylogue-t46.8.2",
    ),
)

PRIVILEGED_ALGEBRA: Final[tuple[MCPTransactionDeclaration, ...]] = (
    MCPTransactionDeclaration(
        name="write",
        verb=MCPVerb.WRITE,
        minimum_role="write",
        object_kinds=("object-ref", "assertion"),
        result_semantics=(MCPResultSemantics.MUTATION,),
        purpose="Apply a declaration-owned mutation after shared authorization.",
        migration_owner="polylogue-t46.8.3",
    ),
    MCPTransactionDeclaration(
        name="judge",
        verb=MCPVerb.JUDGE,
        minimum_role="review",
        object_kinds=("assertion-candidate", "judgment"),
        result_semantics=(MCPResultSemantics.MUTATION,),
        purpose="Accept, reject, defer, or supersede candidates without collapsing candidate state.",
        migration_owner="polylogue-t46.8.3",
    ),
    MCPTransactionDeclaration(
        name="run",
        verb=MCPVerb.RUN,
        minimum_role="write",
        object_kinds=("saved-query", "recipe", "result-set"),
        result_semantics=(MCPResultSemantics.EXHAUSTIVE_PAGE, MCPResultSemantics.MUTATION),
        purpose="Execute a saved query or governed recipe ref.",
        migration_owner="polylogue-t46.8.3",
    ),
    MCPTransactionDeclaration(
        name="maintenance",
        verb=MCPVerb.MAINTENANCE,
        minimum_role="admin",
        object_kinds=("maintenance-plan", "maintenance-operation"),
        result_semantics=(MCPResultSemantics.MAINTENANCE,),
        purpose="Preview, authorize, execute, inspect, and reconcile maintenance operations.",
        migration_owner="polylogue-t46.8.3",
    ),
)

TARGET_RESOURCES: Final[tuple[MCPResourceDeclaration, ...]] = tuple(
    MCPResourceDeclaration(
        uri_template=f"polylogue://{kind}/{{id}}",
        object_kinds=(kind,),
        minimum_role="read",
        authority="read-only object projection; resources never acquire instruction or mutation authority",
        migration_owner="polylogue-t46.8.2" if kind != "recall-pack" else "polylogue-t46.8.3",
    )
    for kind in ("session", "message", "block", "action", "file", "query", "result-set", "recall-pack")
) + (
    MCPResourceDeclaration(
        uri_template="polylogue://capabilities/query",
        object_kinds=("capability", "query", "result-set"),
        minimum_role="read",
        authority="executable query vocabulary and recovery guidance; no mutation authority",
        migration_owner="polylogue-z9gh.3",
    ),
)

TARGET_PROMPTS: Final[tuple[MCPPromptDeclaration, ...]] = (
    MCPPromptDeclaration("resume_context", "resume", "read", "none", "polylogue-t46.8.2"),
    MCPPromptDeclaration("postmortem_last", "postmortem", "read", "none", "polylogue-t46.8.2"),
    MCPPromptDeclaration("decisions_about", "decision-recovery", "read", "none", "polylogue-t46.8.2"),
    MCPPromptDeclaration("unacknowledged_failures", "failure-recovery", "read", "none", "polylogue-t46.8.2"),
    MCPPromptDeclaration("sessions_touching_file", "file-touch", "read", "none", "polylogue-t46.8.2"),
    MCPPromptDeclaration("cost_of", "cost-analysis", "read", "none", "polylogue-t46.8.2"),
    MCPPromptDeclaration("agent_coordination_brief", "coordination", "read", "none", "polylogue-t46.8.3"),
)

if len(TARGET_DEFAULT_READ_ALGEBRA) > 15:
    raise RuntimeError("target default MCP read algebra exceeds the 15-transaction discovery budget")


__all__ = [
    "MCP_KERNEL_REGISTRY",
    "MCP_TOOL_DECLARATIONS",
    "MCP_TOOL_DECLARATION_BY_NAME",
    "PRIVILEGED_ALGEBRA",
    "TARGET_DEFAULT_READ_ALGEBRA",
    "TARGET_PROMPTS",
    "TARGET_RESOURCES",
    "declaration_for_tool",
    "declared_tool_names",
]
