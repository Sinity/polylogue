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
    MCPCapabilities,
    MCPCapabilityFlag,
    MCPHandlerBinding,
    MCPPromptDeclaration,
    MCPResourceDeclaration,
    MCPResultSemantics,
    MCPToolDeclaration,
    MCPTransactionDeclaration,
    MCPVerb,
)

_REPAIR_COMMAND = "devtools test tests/unit/mcp/test_tool_declarations.py"


@dataclass(frozen=True, slots=True)
class _ToolRow:
    name: str
    description: str
    module: str
    registrar: str
    required_capability: MCPCapabilityFlag | None
    verb: MCPVerb
    object_kinds: tuple[str, ...]
    result_semantics: MCPResultSemantics
    schema_source: str
    minimal_arguments: tuple[tuple[str, JSONValue], ...]
    output_kind: str
    operation_owner: str
    contract_fields: tuple[str, ...] = ()
    target_object_kinds: tuple[str, ...] = ()
    target_result_semantics: tuple[MCPResultSemantics, ...] = ()
    target_purpose: str = ""
    target_visible: bool = True


def _compatibility(row: _ToolRow) -> CompatibilityKey:
    return CompatibilityKey(
        identity="mcp-tool:" + ",".join(row.object_kinds),
        lifecycle="registered-handler-retained",
        authority=f"mcp-capability:{row.required_capability or 'read'}",
        access_result_shape=f"{row.verb.value}:{row.result_semantics.value}:{row.output_kind}",
        durability="transport-adapter; domain-owner-controls-durability",
    )


_CUTOVER_TOOL_ROWS: Final[tuple[_ToolRow, ...]] = (
    _ToolRow(
        "query",
        "Execute a parser-owned terminal query page or resume its q2 continuation.",
        "polylogue.mcp.server_cutover",
        "register_cutover_read_tools",
        None,
        MCPVerb.QUERY,
        ("query", "result-set"),
        MCPResultSemantics.EXHAUSTIVE_PAGE,
        "polylogue.mcp.server_cutover.query:inspect.signature",
        (("expression", "messages where text:needle"),),
        "envelope:items,total,unit,limit,offset",
        "polylogue.api.Polylogue.query_units",
        target_result_semantics=(
            MCPResultSemantics.EXHAUSTIVE_PAGE,
            MCPResultSemantics.TOP_K,
            MCPResultSemantics.SAMPLE,
            MCPResultSemantics.AGGREGATE,
        ),
        target_purpose="Execute a declared DSL or typed plan with explicit result semantics and continuation.",
    ),
    _ToolRow(
        "read",
        "Read a stable archive URI or public ref through a declared view.",
        "polylogue.mcp.server_cutover",
        "register_cutover_read_tools",
        None,
        MCPVerb.READ,
        ("object-ref", "evidence-ref"),
        MCPResultSemantics.EXHAUSTIVE_PAGE,
        "polylogue.mcp.server_cutover.read:inspect.signature",
        (("ref", "session:codex-session:demo"),),
        "single_object",
        "polylogue.api.Polylogue.resolve_ref",
        target_object_kinds=("object-ref", "evidence-ref"),
        target_result_semantics=(
            MCPResultSemantics.SINGLE_OBJECT,
            MCPResultSemantics.EXHAUSTIVE_PAGE,
            MCPResultSemantics.BOUNDED_CONTEXT,
        ),
        target_purpose="Read any stable archive ref through a declared projection/view.",
    ),
    _ToolRow(
        "get",
        "Resolve one exact stable object or evidence identity.",
        "polylogue.mcp.server_cutover",
        "register_cutover_read_tools",
        None,
        MCPVerb.GET,
        ("object-ref",),
        MCPResultSemantics.SINGLE_OBJECT,
        "polylogue.mcp.server_cutover.get:inspect.signature",
        (("ref", "session:codex-session:demo"),),
        "single_object",
        "polylogue.api.Polylogue.resolve_ref",
        target_purpose="Resolve one exact object identity when a generic read would add ambiguity.",
    ),
    _ToolRow(
        "explain",
        "Explain parser grammar, capabilities, refs, result semantics, or recovery.",
        "polylogue.mcp.server_cutover",
        "register_cutover_read_tools",
        None,
        MCPVerb.EXPLAIN,
        ("query", "capability", "object-ref"),
        MCPResultSemantics.SINGLE_OBJECT,
        "polylogue.mcp.server_cutover.explain:inspect.signature",
        (("subject", "capability"),),
        "single_object",
        "polylogue.api.Polylogue.explain_query_expression",
        target_object_kinds=("query", "object-ref", "capability"),
        target_purpose="Discover grammar, fields, values, plans, refs, authority, and recovery routes.",
    ),
    _ToolRow(
        "context",
        "Compile a policy-gated bounded context image with receipts.",
        "polylogue.mcp.server_cutover",
        "register_cutover_read_tools",
        None,
        MCPVerb.CONTEXT,
        ("context-snapshot", "context-delivery"),
        MCPResultSemantics.BOUNDED_CONTEXT,
        "polylogue.mcp.server_cutover.context:inspect.signature",
        (("intent", "resume"),),
        "single_object",
        "polylogue.api.Polylogue.context_image_payload",
        target_purpose="Compile and retrieve policy-gated bounded context plus receipts.",
    ),
    _ToolRow(
        "status",
        "Report compact archive authority and readiness status.",
        "polylogue.mcp.server_cutover",
        "register_cutover_read_tools",
        None,
        MCPVerb.STATUS,
        ("status",),
        MCPResultSemantics.SINGLE_OBJECT,
        "polylogue.mcp.server_cutover.status:inspect.signature",
        (("scope", "archive"),),
        "single_object",
        "polylogue.storage.sqlite.archive_tiers.archive.ArchiveStore.stats",
        target_object_kinds=("status", "receipt"),
        target_result_semantics=(MCPResultSemantics.SINGLE_OBJECT, MCPResultSemantics.AGGREGATE),
        target_purpose="Read archive, source, embedding, coordination, and operation status.",
    ),
    _ToolRow(
        "write",
        "Apply a declared mutation operation after shared authorization. Destructive "
        "operations (delete_session, remove_tag, remove_mark, delete_metadata, "
        "delete_annotation, delete_saved_view, delete_recall_pack, delete_workspace) "
        "require confirm=true and fail closed without it.",
        "polylogue.mcp.server_cutover",
        "register_cutover_privileged_tools",
        "write",
        MCPVerb.WRITE,
        ("object-ref", "assertion"),
        MCPResultSemantics.MUTATION,
        "polylogue.mcp.server_cutover.write:inspect.signature",
        (("operation", "add_tag"), ("session_id", "test:conv-mutation"), ("tag", "review")),
        "operation_result",
        "mutate-write",
        target_purpose="Apply a declaration-owned mutation after shared authorization.",
    ),
    _ToolRow(
        "record_work_event",
        "Record a typed live-agent work event for a session.",
        "polylogue.mcp.server_cutover",
        "register_cutover_privileged_tools",
        "write",
        MCPVerb.WRITE,
        ("session", "work-event", "evidence-ref"),
        MCPResultSemantics.MUTATION,
        "polylogue.mcp.server_cutover.record_work_event:inspect.signature",
        (("session_id", "codex-session:demo"), ("event_type", "tool_run"), ("event_id", "evt-1")),
        "operation_result",
        "polylogue.api.Polylogue.record_work_event",
        target_visible=False,
    ),
    _ToolRow(
        "emit_decision",
        "Record a typed decision event with evidence references for a session.",
        "polylogue.mcp.server_cutover",
        "register_cutover_privileged_tools",
        "write",
        MCPVerb.WRITE,
        ("session", "decision", "evidence-ref"),
        MCPResultSemantics.MUTATION,
        "polylogue.mcp.server_cutover.emit_decision:inspect.signature",
        (("session_id", "codex-session:demo"), ("decision", "keep"), ("event_id", "evt-2")),
        "operation_result",
        "polylogue.api.Polylogue.emit_decision",
        target_visible=False,
    ),
    _ToolRow(
        "judge",
        "Accept, reject, defer, or supersede assertion candidates without collapsing candidate state.",
        "polylogue.mcp.server_cutover",
        "register_cutover_privileged_tools",
        "judge",
        MCPVerb.JUDGE,
        ("assertion-candidate", "judgment"),
        MCPResultSemantics.MUTATION,
        "polylogue.mcp.server_cutover.judge:inspect.signature",
        (("candidate_ref", "assertion:contract-candidate"), ("decision", "accept")),
        "envelope:items,applied_count,failed_count",
        "polylogue.api.Polylogue.judge_assertion_candidates",
        target_purpose="Accept, reject, defer, or supersede candidates without collapsing candidate state.",
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
        (("ref", "saved-view:contract-view"),),
        "envelope:total",
        "mutate-run",
        target_object_kinds=("saved-query", "recipe", "result-set"),
        target_result_semantics=(MCPResultSemantics.EXHAUSTIVE_PAGE, MCPResultSemantics.MUTATION),
        target_purpose="Execute a saved query or governed recipe ref.",
    ),
    _ToolRow(
        "maintenance",
        "Rebuild session insights and inspect or adjudicate operation recovery. "
        "rebuild_insights and recovery_adjudicate require confirm=true "
        "and fail closed without it.",
        "polylogue.mcp.server_cutover",
        "register_cutover_privileged_tools",
        "maintenance",
        MCPVerb.MAINTENANCE,
        ("maintenance-operation",),
        MCPResultSemantics.MAINTENANCE,
        "polylogue.mcp.server_cutover.maintenance:inspect.signature",
        (("operation", "recovery_status"),),
        "operation_result",
        "mutate-rebuild-insights",
        target_object_kinds=("maintenance-plan", "maintenance-operation"),
        target_purpose="Preview, authorize, execute, inspect, and reconcile maintenance operations.",
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
        role_gate=f"mcp.capability:{row.required_capability or 'read'}",
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
    return MCPToolDeclaration(
        kernel=kernel,
        name=row.name,
        description=row.description,
        required_capability=row.required_capability,
        registration=MCPHandlerBinding(module=row.module, symbol=row.name, registrar=row.registrar),
        transaction=(
            MCPTransactionDeclaration(
                name=row.name,
                verb=row.verb,
                required_capability=row.required_capability,
                object_kinds=row.target_object_kinds or row.object_kinds,
                result_semantics=row.target_result_semantics or (row.result_semantics,),
                purpose=row.target_purpose or row.description,
            )
            if row.target_visible
            else None
        ),
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


_ALL_CAPABILITIES_ENABLED = MCPCapabilities(write=True, judge=True, maintenance=True)


def declared_tool_names(capabilities: MCPCapabilities = _ALL_CAPABILITIES_ENABLED) -> frozenset[str]:
    """Return the tool names visible under ``capabilities``.

    Default is every capability enabled (the full twelve-tool surface), used by
    inventory/discovery tooling that wants the complete declared set rather
    than one server's resolved config.
    """
    return frozenset(
        declaration.name
        for declaration in MCP_TOOL_DECLARATIONS
        if capabilities.allows(declaration.required_capability)
    )


def _target_algebra(*, privileged: bool) -> tuple[MCPTransactionDeclaration, ...]:
    """Project the executable target protocol from the MCP tool declarations."""

    return tuple(
        declaration.transaction
        for declaration in MCP_TOOL_DECLARATIONS
        if declaration.transaction is not None and (declaration.required_capability is not None) is privileged
    )


TARGET_DEFAULT_READ_ALGEBRA: Final[tuple[MCPTransactionDeclaration, ...]] = _target_algebra(privileged=False)
PRIVILEGED_ALGEBRA: Final[tuple[MCPTransactionDeclaration, ...]] = _target_algebra(privileged=True)

TARGET_RESOURCES: Final[tuple[MCPResourceDeclaration, ...]] = tuple(
    MCPResourceDeclaration(
        uri_template=f"polylogue://{kind}/{{id}}",
        object_kinds=(kind,),
        required_capability=None,
        authority="read-only object projection; resources never acquire instruction or mutation authority",
    )
    for kind in ("session", "message", "block", "action", "file", "query", "result-set", "recall-pack")
) + (
    MCPResourceDeclaration(
        uri_template="polylogue://capabilities/query",
        object_kinds=("capability", "query", "result-set"),
        required_capability=None,
        authority="executable query vocabulary and recovery guidance; no mutation authority",
    ),
)

TARGET_PROMPTS: Final[tuple[MCPPromptDeclaration, ...]] = (
    MCPPromptDeclaration("resume_context", "resume", None, "none"),
    MCPPromptDeclaration("postmortem_last", "postmortem", None, "none"),
    MCPPromptDeclaration("decisions_about", "decision-recovery", None, "none"),
    MCPPromptDeclaration("unacknowledged_failures", "failure-recovery", None, "none"),
    MCPPromptDeclaration("sessions_touching_file", "file-touch", None, "none"),
    MCPPromptDeclaration("cost_of", "cost-analysis", None, "none"),
    MCPPromptDeclaration("agent_coordination_brief", "coordination", None, "none"),
    # Live-registered (polylogue/mcp/server_prompts.py) but previously absent
    # here, leaving completeness/discovery consumers blind to them
    # (polylogue-il50).
    MCPPromptDeclaration("analyze_errors", "error-analysis", None, "none"),
    MCPPromptDeclaration("summarize_week", "weekly-summary", None, "none"),
    MCPPromptDeclaration("extract_code", "code-extraction", None, "none"),
    MCPPromptDeclaration("compare_sessions", "session-comparison", None, "none"),
    MCPPromptDeclaration("extract_patterns", "pattern-extraction", None, "none"),
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
