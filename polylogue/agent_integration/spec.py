"""Typed authority for the executable standing agent manual.

This module owns typed capabilities, checked queries, recipes, client delivery
declarations, and rendered package assets. Its contracts are derived from the
live MCP declaration algebra and checked against registered MCPServer
signatures. ``polylogue.mcp.declarations.declared_tool_names`` is the sole
authority for which tools exist; this module restates neither the names nor
their count.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, TypeAlias, get_args

from polylogue.archive.filter.types import SortField
from polylogue.archive.query.discovery import query_discovery_example, query_discovery_examples
from polylogue.archive.query.unit_results import TERMINAL_FILTER_PARAMETER_BY_NAME
from polylogue.core.enums import Origin, enum_values
from polylogue.declarations import JSONValue
from polylogue.mcp.declarations import (
    MCP_TOOL_DECLARATIONS,
    PRIVILEGED_ALGEBRA,
    TARGET_DEFAULT_READ_ALGEBRA,
    TARGET_PROMPTS,
    TARGET_RESOURCES,
    MCPCapabilityFlag,
    MCPResultSemantics,
    MCPToolDeclaration,
    MCPTransactionDeclaration,
)
from polylogue.sources.origin_specs import public_origin_meanings

ASSET_VERSION = "2026-09-15.declared-tools-r01"
AgentClient = Literal["claude-code", "codex", "gemini", "hermes"]
GuidanceMode = Literal["full", "mcp-only", "off"]
QuerySurface = Literal["session", "terminal"]
SchemaStatus = Literal["not-ready", "live-verified"]
ArgumentKind = Literal["string", "integer", "boolean", "array", "object"]

TARGET_SCHEMA_STATUS: SchemaStatus = "live-verified"
CLIENTS: tuple[AgentClient, ...] = ("claude-code", "codex", "gemini", "hermes")
GUIDANCE_MODES: tuple[GuidanceMode, ...] = ("full", "mcp-only", "off")
DEFAULT_READ_TOOLS: tuple[str, ...] = tuple(item.name for item in TARGET_DEFAULT_READ_ALGEBRA)
PRIVILEGED_TOOLS: tuple[str, ...] = tuple(item.name for item in PRIVILEGED_ALGEBRA)
ALL_TARGET_TOOLS: tuple[str, ...] = (*DEFAULT_READ_TOOLS, *PRIVILEGED_TOOLS)
#: Every declared MCP tool in declaration order, including the ones outside the
#: target transaction algebra. This is the manual's tool surface; its length is
#: the only tool count anything should render.
ALL_DECLARED_TOOLS: tuple[str, ...] = tuple(declaration.name for declaration in MCP_TOOL_DECLARATIONS)
CONTINUATION_SENTINEL = "$continuation"

Arguments: TypeAlias = tuple[tuple[str, JSONValue], ...]


@dataclass(frozen=True, slots=True)
class CapabilityFamily:
    """One stable reason for an agent to invoke Polylogue."""

    id: str
    title: str
    required_capability: MCPCapabilityFlag | None
    first_tool: str


@dataclass(frozen=True, slots=True)
class ToolArgument:
    """One declared input parameter in the cutover contract."""

    name: str
    kind: ArgumentKind
    required_initial: bool
    description: str
    #: Declared vocabulary for enum-typed arguments, resolved against the live
    #: handler's ``Literal`` annotation by the generated-contract gate.  Empty
    #: for free-form arguments.
    enum_values: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ConfirmationGate:
    """The single declared description of a tool's confirmation gate.

    The manual renders its maintenance prose from this object, so the gate is
    described exactly once.  ``argument`` must exist on the live handler and
    ``operations`` must be a subset of the live operation vocabulary.
    """

    argument: str
    operations: tuple[str, ...]
    inspection_operations: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ToolExample:
    """A normal invocation compiled into the generated manual."""

    id: str
    title: str
    arguments: Arguments
    result_note: str

    def arguments_dict(self) -> dict[str, JSONValue]:
        return dict(self.arguments)


@dataclass(frozen=True, slots=True)
class ToolContract:
    """Manual-facing target transaction derived from t46.8 declarations."""

    name: str
    required_capability: MCPCapabilityFlag | None
    purpose: str
    source_declarations: tuple[str, ...]
    result_semantics: tuple[MCPResultSemantics, ...]
    arguments: tuple[ToolArgument, ...]
    examples: tuple[ToolExample, ...]
    supports_continuation: bool
    emits_result_ref: bool
    schema_status: SchemaStatus = TARGET_SCHEMA_STATUS
    #: Present only for tools whose execution is confirmation-gated.
    confirmation: ConfirmationGate | None = None

    @property
    def argument_names(self) -> frozenset[str]:
        return frozenset(argument.name for argument in self.arguments)

    @property
    def required_initial_arguments(self) -> tuple[str, ...]:
        return tuple(argument.name for argument in self.arguments if argument.required_initial)


@dataclass(frozen=True, slots=True)
class CheckedQuery:
    """One query expression projected from a declared discovery example.

    Never written by hand: ``declaration_id`` names a row in
    :mod:`polylogue.archive.query.discovery`, the declaration site that the
    production parser gates. The manual therefore cannot teach an expression
    that no declaration owns.
    """

    declaration_id: str
    expression: str
    surface: QuerySurface
    purpose: str


def checked_query(declaration_id: str) -> CheckedQuery:
    """Project one declared discovery example into the manual model."""

    example = query_discovery_example(declaration_id)
    return CheckedQuery(
        declaration_id=example.key,
        expression=example.expression,
        surface="session" if example.parser == "session" else "terminal",
        purpose=example.answers,
    )


@dataclass(frozen=True, slots=True)
class RecipeStep:
    """One executable continuity step.

    A step that runs or explains a query names the declaration it executes
    (``example_key``) instead of carrying a copied expression; the expression
    is resolved from the discovery corpus when the call is compiled.
    """

    tool: str
    arguments: Arguments
    purpose: str
    capture: str | None = None
    example_key: str | None = None

    def arguments_dict(self) -> dict[str, JSONValue]:
        if self.example_key is None:
            return dict(self.arguments)
        return {"expression": query_discovery_example(self.example_key).expression, **dict(self.arguments)}


@dataclass(frozen=True, slots=True)
class Recipe:
    """Executable continuity workflow made exclusively from the six read tools."""

    id: str
    title: str
    intent: str
    family: str
    steps: tuple[RecipeStep, ...]
    resources: tuple[str, ...] = ()
    prompts: tuple[str, ...] = ()

    @property
    def query_declarations(self) -> tuple[str, ...]:
        """Declaration ids this recipe executes, in step order, deduplicated."""

        keys: list[str] = []
        for step in self.steps:
            if step.example_key is not None and step.example_key not in keys:
                keys.append(step.example_key)
        return tuple(keys)

    @property
    def queries(self) -> tuple[CheckedQuery, ...]:
        return tuple(checked_query(key) for key in self.query_declarations)


@dataclass(frozen=True, slots=True)
class OriginMeaning:
    """One current source-origin token and its user-facing meaning."""

    token: str
    meaning: str


@dataclass(frozen=True, slots=True)
class ClientDelivery:
    """Native client installation mechanism retained from beads-06."""

    client: AgentClient
    mcp_config: str
    standing_delivery: str
    reference_delivery: str
    unchanged: str
    six_tool_delta: str


def _args(**values: JSONValue) -> Arguments:
    return tuple(values.items())


def _arg(
    name: str,
    kind: ArgumentKind,
    required: bool,
    description: str,
    enum_values: tuple[str, ...] = (),
) -> ToolArgument:
    return ToolArgument(name, kind, required, description, enum_values)


def _example(id: str, title: str, result_note: str, **arguments: JSONValue) -> ToolExample:
    return ToolExample(id=id, title=title, arguments=_args(**arguments), result_note=result_note)


def _declared_filter_arguments(*names: str) -> tuple[ToolArgument, ...]:
    """Project declared terminal session filters into manual tool arguments.

    The names, types and meanings come from
    ``polylogue.archive.query.unit_results.TERMINAL_FILTER_PARAMETERS``, the
    same declaration the ``/api/query-units`` OpenAPI parameters are generated
    from, so the MCP manual and the HTTP schema cannot describe the shared
    filter surface differently. Closed vocabularies come from the enum the
    filter validates against.
    """

    vocabularies: dict[str, tuple[str, ...]] = {"origin": enum_values(Origin)}
    arguments: list[ToolArgument] = []
    for name in names:
        parameter = TERMINAL_FILTER_PARAMETER_BY_NAME[name]
        kind: ArgumentKind = "integer" if parameter.kind == "integer" else parameter.kind
        arguments.append(
            ToolArgument(parameter.name, kind, False, parameter.description, vocabularies.get(parameter.name, ()))
        )
    return tuple(arguments)


def _target_declaration_index() -> dict[str, MCPTransactionDeclaration]:
    return {item.name: item for item in (*TARGET_DEFAULT_READ_ALGEBRA, *PRIVILEGED_ALGEBRA)}


def _declaration_index() -> dict[str, MCPToolDeclaration]:
    """Index every declared tool, target-visible or not."""
    return {declaration.name: declaration for declaration in MCP_TOOL_DECLARATIONS}


def _sources(*names: str, optional: tuple[str, ...] = ()) -> tuple[str, ...]:
    index = _declaration_index()
    selected: list[str] = []
    for name in names:
        if name not in index:
            raise RuntimeError(f"agent manual requires missing MCP target declaration {name!r}")
        selected.append(name)
    selected.extend(name for name in optional if name in index)
    return tuple(selected)


def _semantics(source_names: tuple[str, ...]) -> tuple[MCPResultSemantics, ...]:
    index = _declaration_index()
    result: list[MCPResultSemantics] = []
    for source_name in source_names:
        declaration = index[source_name]
        semantics = declaration.result_semantics
        for semantic in semantics:
            if semantic not in result:
                result.append(semantic)
    return tuple(result)


def _required_capability(source_names: tuple[str, ...]) -> MCPCapabilityFlag | None:
    index = _declaration_index()
    capabilities = {index[name].required_capability for name in source_names}
    if len(capabilities) != 1:
        raise RuntimeError(
            f"manual transaction sources disagree on required capability: {source_names!r} -> {sorted(capabilities, key=str)}"
        )
    return capabilities.pop()


def _contract(
    *,
    name: str,
    source_names: tuple[str, ...],
    purpose: str,
    arguments: tuple[ToolArgument, ...],
    examples: tuple[ToolExample, ...],
    supports_continuation: bool,
    emits_result_ref: bool,
    confirmation: ConfirmationGate | None = None,
) -> ToolContract:
    return ToolContract(
        name=name,
        required_capability=_required_capability(source_names),
        purpose=purpose,
        source_declarations=source_names,
        result_semantics=_semantics(source_names),
        arguments=arguments,
        examples=examples,
        supports_continuation=supports_continuation,
        emits_result_ref=emits_result_ref,
        confirmation=confirmation,
    )


_QUERY_SOURCES = _sources("query")
_READ_SOURCES = _sources("read")
_GET_SOURCES = _sources("get")
_EXPLAIN_SOURCES = _sources("explain")
_CONTEXT_SOURCES = _sources("context")
_STATUS_SOURCES = _sources("status")
_WRITE_SOURCES = _sources("write")
_RECORD_WORK_EVENT_SOURCES = _sources("record_work_event")
_EMIT_DECISION_SOURCES = _sources("emit_decision")
_JUDGE_SOURCES = _sources("judge")
_RUN_SOURCES = _sources("run")
_MAINTENANCE_SOURCES = _sources("maintenance")

#: The live ``maintenance`` operation vocabulary and its one confirmation gate.
#: ``test_manual_contract`` resolves both against the registered MCP handler, so
#: an operation the handler does not declare cannot reach the manual's prose.
_MAINTENANCE_OPERATIONS: tuple[str, ...] = ("rebuild_insights", "recovery_status", "recovery_adjudicate")
_MAINTENANCE_CONFIRMATION = ConfirmationGate(
    argument="confirm",
    operations=("rebuild_insights", "recovery_adjudicate"),
    inspection_operations=("recovery_status",),
)

TOOL_CONTRACTS: tuple[ToolContract, ...] = (
    _contract(
        name="query",
        source_names=_QUERY_SOURCES,
        purpose="Execute the real expression DSL or a declared typed plan and return a bounded, semantics-labelled result set.",
        arguments=(
            _arg("expression", "string", False, "Parser-owned DSL expression; omit when resuming with continuation."),
            _arg("limit", "integer", False, "Requested page size, subject to server and transport bounds."),
            _arg("projection", "string", False, "Declared result projection such as session-summary or cost-rollup."),
            _arg(
                "session_operation",
                "string",
                False,
                "Declared session operation; required by the session-operations projection.",
            ),
            _arg("continuation", "string", False, "Opaque token from the preceding response; send alone."),
            _arg("offset", "integer", False, "Offset for projections that use decimal offset pagination."),
            _arg("sort", "string", False, "Declared sort for session projections.", get_args(SortField)),
            *_declared_filter_arguments(
                "origin", "tag", "repo", "since", "until", "min_messages", "max_messages", "min_words"
            ),
        ),
        examples=(
            _example(
                "query-file-actions",
                "Find recent edits under the query subsystem",
                "An exhaustive page of action rows with object/evidence refs, one result_ref, and a continuation when more rows exist.",
                expression="actions where action:file_edit AND path:polylogue/archive/query | sort by time desc | limit 20",
                limit=20,
                projection="action-evidence",
            ),
            _example(
                "query-cost-cohort",
                "Select a recent provider cohort for a cost audit",
                "A session result set suitable for a declared cost-rollup projection; coverage still governs completeness.",
                expression="sessions where origin:(claude-code-session|codex-session) AND date >= 2026-07-01",
                limit=50,
                projection="cost-rollup",
            ),
        ),
        supports_continuation=True,
        emits_result_ref=True,
    ),
    _contract(
        name="read",
        source_names=_READ_SOURCES,
        purpose="Read a stable URI/object/evidence ref through a declared view, including topology and evidence projections.",
        arguments=(
            _arg("ref", "string", True, "Stable object, evidence, result-set, or URI reference."),
            _arg("view", "string", False, "Declared projection/view for the referenced object."),
            _arg("limit", "integer", False, "Page size for collection-like or recursive reads."),
            _arg("offset", "integer", False, "Offset into collection-like reads that use decimal offset pagination."),
            _arg("continuation", "string", False, "Opaque token from the preceding read response; send alone."),
        ),
        examples=(
            _example(
                "read-session-chronicle",
                "Read a session chronicle",
                "A bounded chronicle page retaining message/block evidence refs and the same result_ref across continuation pages.",
                ref="polylogue://session/codex-session:demo-lineage-fork",
                view="chronicle",
                limit=20,
            ),
        ),
        supports_continuation=True,
        emits_result_ref=True,
    ),
    _contract(
        name="get",
        source_names=_GET_SOURCES,
        purpose="Resolve one exact stable identity without search or ranking ambiguity.",
        arguments=(
            _arg("ref", "string", True, "One exact object/evidence/URI reference."),
            _arg("projection", "string", False, "Optional declared projection of that exact object."),
        ),
        examples=(
            _example(
                "get-evidence-block",
                "Resolve the exact evidence block behind a claim",
                "One object with its canonical ref and provenance; absence is explicit rather than an empty ranked result.",
                ref="block:codex-session:demo-receipts:receipts-a-claim:0",
                projection="evidence",
            ),
        ),
        supports_continuation=False,
        emits_result_ref=False,
    ),
    _contract(
        name="explain",
        source_names=_EXPLAIN_SOURCES,
        purpose="Explain query grammar, fields, values, lowering, result semantics, refs, capabilities, or recovery before guessing.",
        arguments=(
            _arg(
                "subject",
                "string",
                True,
                "Declared explanation subject: query, field, value, ref, capability, result, or recovery.",
            ),
            _arg("expression", "string", False, "Query expression to parse and lower when subject=query."),
            _arg("ref", "string", False, "Object/ref whose authority or addressing needs explanation."),
            _arg("offset", "integer", False, "Offset into paged explanation results."),
            _arg("search", "string", False, "Optional explanation search text."),
            _arg("limit", "integer", False, "Maximum explanation rows."),
        ),
        examples=(
            _example(
                "explain-query",
                "Inspect parser and lowering behavior",
                "Parser-owned AST/lowering metadata, selected unit, result semantics, and correction guidance without executing the query.",
                subject="query",
                expression="observed-events where kind:tool_finished AND handler:shell | group by status | count",
            ),
        ),
        supports_continuation=False,
        emits_result_ref=False,
    ),
    _contract(
        name="context",
        source_names=_CONTEXT_SOURCES,
        purpose="Compile a bounded, policy-gated context image with receipts and evidence refs for resumption or investigation.",
        arguments=(
            _arg("intent", "string", True, "Context intent such as resume, postmortem, prior-art, or coordination."),
            _arg("query", "string", False, "Parser-owned cohort expression that constrains source material."),
            _arg(
                "budget_tokens", "integer", False, "Upper bound for compiled context, not a claim-completeness limit."
            ),
            _arg("result_ref", "string", False, "Existing result set to compile without rerunning discovery."),
            _arg("repo_path", "string", False, "Repository path used for project context."),
            _arg("cwd", "string", False, "Working directory used for project context."),
            _arg("recent_files", "array", False, "Recently touched files used for continuity."),
            _arg("session_id", "string", False, "Current agent session identity."),
            _arg("limit", "integer", False, "Bound on related context candidates."),
            _arg("offset", "integer", False, "Offset into ranked candidates for decimal offset pagination."),
            _arg("recipient_ref", "string", False, "Recipient identity for delivery receipts."),
            _arg("assertion_ref", "string", False, "Assertion identity to include in context."),
        ),
        examples=(
            _example(
                "context-resume",
                "Compile a resume packet",
                "A bounded context snapshot plus receipt describing selected refs, omissions, policy, and budget use.",
                intent="resume",
                query="sessions where repo:polylogue AND NOT tag:complete",
                budget_tokens=4000,
            ),
        ),
        supports_continuation=False,
        emits_result_ref=True,
    ),
    _contract(
        name="status",
        source_names=_STATUS_SOURCES,
        purpose="Report archive identity, readiness, freshness, coverage, coordination, embeddings, and governed operation state.",
        arguments=(
            _arg(
                "scope",
                "string",
                True,
                "Status domain such as archive, sources, embeddings, coordination, or operation.",
            ),
            _arg("include", "array", False, "Named status facets requested from that scope."),
            _arg("ref", "string", False, "Specific operation/receipt/object ref for status lookup."),
        ),
        examples=(
            _example(
                "status-archive",
                "Establish archive authority before making a broad claim",
                "Archive identity, selected source coverage, freshness/readiness state, and explicit degraded reasons.",
                scope="archive",
                include=["identity", "coverage", "freshness", "readiness"],
            ),
        ),
        supports_continuation=False,
        emits_result_ref=False,
    ),
    _contract(
        name="write",
        source_names=_WRITE_SOURCES,
        purpose="Apply one declaration-owned reversible mutation with actor, target, conflict policy, and receipt.",
        arguments=(
            _arg("operation", "string", True, "Declared reversible write operation."),
            _arg("session_id", "string", False, "Exact mutation session target."),
            _arg("session_ids", "array", False, "Batch mutation session targets."),
            _arg("tag", "string", False, "Tag value for tag operations."),
            _arg("tags", "array", False, "Tag values for bulk operations."),
            _arg("key", "string", False, "Metadata or annotation key."),
            _arg("value", "string", False, "Scalar operation value."),
            _arg("confirm", "boolean", False, "Explicit confirmation for governed deletion."),
            _arg("fields", "object", False, "Structured operation fields."),
        ),
        examples=(
            _example(
                "write-tag",
                "Add a review tag through the governed write chokepoint",
                "A mutation receipt with actor, target, effect identity, and resulting generation; no destructive confirmation is invented.",
                operation="tag.add",
                session_id="codex-session:demo-lineage-fork",
                value="review",
            ),
        ),
        supports_continuation=False,
        emits_result_ref=False,
    ),
    _contract(
        name="record_work_event",
        source_names=_RECORD_WORK_EVENT_SOURCES,
        purpose="Record one typed live-agent work event against a session so later sessions can retrieve it as evidence.",
        arguments=(
            _arg("session_id", "string", True, "Session the event belongs to."),
            _arg("event_id", "string", True, "Caller-chosen idempotent event identity."),
            _arg(
                "event_type",
                "string",
                True,
                "The declared work-event type.",
                ("tool_run", "subagent_spawn", "decision", "artifact_change"),
            ),
            _arg("summary", "string", True, "One-line description of what happened."),
            _arg("payload", "object", False, "Structured event detail."),
            _arg("timestamp", "string", False, "Event time; defaults to the ingest clock."),
        ),
        examples=(
            _example(
                "record-tool-run",
                "Record a tool run as retrievable evidence",
                "A mutation receipt; re-sending the same event_id is idempotent rather than a second event.",
                session_id="codex-session:demo-lineage-fork",
                event_id="evt-tool-run-1",
                event_type="tool_run",
                summary="Ran the focused agent-integration selection.",
            ),
        ),
        supports_continuation=False,
        emits_result_ref=False,
    ),
    _contract(
        name="emit_decision",
        source_names=_EMIT_DECISION_SOURCES,
        purpose="Record a typed decision with its evidence references, using the shared work-event vocabulary.",
        arguments=(
            _arg("session_id", "string", True, "Session the decision belongs to."),
            _arg("event_id", "string", True, "Caller-chosen idempotent event identity."),
            _arg("decision", "string", True, "The decision reached."),
            _arg("summary", "string", True, "Why the decision was reached."),
            _arg("evidence_refs", "array", False, "Stable refs the decision rests on."),
            _arg("timestamp", "string", False, "Decision time; defaults to the ingest clock."),
        ),
        examples=(
            _example(
                "emit-decision-with-evidence",
                "Record a decision against the evidence it rests on",
                "A mutation receipt binding the decision to the cited refs; cite refs rather than restating their content.",
                session_id="codex-session:demo-lineage-fork",
                event_id="evt-decision-1",
                decision="keep",
                summary="The declaration, not the prose, is the intended contract.",
                evidence_refs=["polylogue://session/codex-session:demo-lineage-fork"],
            ),
        ),
        supports_continuation=False,
        emits_result_ref=False,
    ),
    _contract(
        name="judge",
        source_names=_JUDGE_SOURCES,
        purpose="Accept, reject, defer, or supersede an assertion candidate while preserving candidate and judgment provenance.",
        arguments=(
            _arg("items", "array", False, "Batch of assertion candidates and decisions."),
            _arg("candidate_ref", "string", False, "Exact assertion-candidate identity."),
            _arg("decision", "string", False, "Declared judgment decision."),
            _arg("reason", "string", False, "Evidence-grounded judgment rationale."),
            _arg("inject", "boolean", False, "Whether to inject accepted judgment context."),
            _arg("replacement_kind", "string", False, "Replacement assertion kind."),
            _arg("replacement_body_text", "string", False, "Replacement assertion body."),
            _arg("replacement_value", "object", False, "Structured replacement assertion value."),
        ),
        examples=(
            _example(
                "judge-defer",
                "Defer a candidate pending stronger evidence",
                "A judgment receipt that leaves candidate history intact and reports conflicts explicitly.",
                candidate_ref="assertion-candidate:demo-review-001",
                decision="defer",
                reason="The cited result set does not cover the claimed source cohort.",
            ),
        ),
        supports_continuation=False,
        emits_result_ref=False,
    ),
    _contract(
        name="run",
        source_names=_RUN_SOURCES,
        purpose="Execute a saved query or governed recipe ref; any nested mutation inherits its own capability and confirmation policy.",
        arguments=(
            _arg("ref", "string", True, "Saved-query or recipe ref."),
            _arg("limit", "integer", False, "Bound on saved-query results."),
        ),
        examples=(
            _example(
                "run-cost-recipe",
                "Run a saved read-only cost audit",
                "A result_ref and receipt for the declared recipe; mutation authority is never gained from the recipe wrapper.",
                ref="recipe:cost-audit",
                limit=20,
            ),
        ),
        supports_continuation=False,
        emits_result_ref=True,
    ),
    _contract(
        name="maintenance",
        source_names=_MAINTENANCE_SOURCES,
        purpose="Rebuild session insights and inspect or adjudicate operation recovery; there is no generic maintenance or repair umbrella.",
        arguments=(
            _arg(
                "operation",
                "string",
                True,
                "The declared maintenance operation.",
                _MAINTENANCE_OPERATIONS,
            ),
            _arg("operation_id", "string", False, "Exact operation identity to adjudicate."),
            _arg("target_outcomes", "object", False, "Observed target outcomes for adjudication."),
            _arg("reason", "string", False, "Operator reason."),
            _arg(
                "confirm",
                "boolean",
                False,
                "Explicit confirmation required by the full-effect operations.",
            ),
        ),
        examples=(
            _example(
                "maintenance-recovery-status",
                "Inspect unreconciled operation recovery",
                "A read-only operation_result describing operations whose applied/not-applied outcome is still unknown.",
                operation="recovery_status",
            ),
        ),
        supports_continuation=False,
        emits_result_ref=False,
        confirmation=_MAINTENANCE_CONFIRMATION,
    ),
)

TOOL_CONTRACT_BY_NAME: dict[str, ToolContract] = {contract.name: contract for contract in TOOL_CONTRACTS}
if tuple(TOOL_CONTRACT_BY_NAME) != ALL_DECLARED_TOOLS:
    raise RuntimeError(
        "agent tool contracts must cover every declared MCP tool, in declaration order; "
        f"missing={sorted(set(ALL_DECLARED_TOOLS) - set(TOOL_CONTRACT_BY_NAME))} "
        f"unexpected={sorted(set(TOOL_CONTRACT_BY_NAME) - set(ALL_DECLARED_TOOLS))}"
    )

CAPABILITY_FAMILIES: tuple[CapabilityFamily, ...] = (
    CapabilityFamily("authority", "Archive identity, source coverage, freshness, and readiness", None, "status"),
    CapabilityFamily("discovery", "Cross-session, row-level, semantic, aggregate, and prior-art search", None, "query"),
    CapabilityFamily("evidence", "Exact objects, transcripts, topology, raw evidence, and citations", None, "read"),
    CapabilityFamily(
        "teaching", "Grammar, fields, values, plans, refs, result semantics, and recovery", None, "explain"
    ),
    CapabilityFamily("continuity", "Resume, postmortem, forensic, coordination, and bounded context", None, "context"),
    CapabilityFamily("mutation", "Reversible overlays, judgments, saved runs, and administration", "write", "write"),
)

RECIPES: tuple[Recipe, ...] = (
    Recipe(
        id="resume-session",
        title="Resume a session from evidence",
        intent="Recover current work, failed effects, open loops, and a bounded next-step context without trusting a stale summary.",
        family="continuity",
        steps=(
            RecipeStep(
                "status",
                _args(scope="archive", include=["identity", "coverage", "freshness", "readiness"]),
                "Establish which archive and source generations can support the answer.",
            ),
            RecipeStep(
                "query",
                _args(limit=20, projection="session-summary"),
                "Find likely unfinished sessions.",
                capture="candidate_result_ref",
                example_key="session-negated-tag",
            ),
            RecipeStep(
                "query",
                _args(limit=20, projection="action-evidence"),
                "Find recent failed effects that may invalidate an optimistic handoff.",
                capture="failure_result_ref",
                example_key="actions-unacknowledged-failures",
            ),
            RecipeStep(
                "read",
                _args(ref="polylogue://session/codex-session:demo-lineage-fork", view="chronicle", limit=20),
                "Read the strongest candidate with evidence refs; continue until the needed boundary is reached.",
            ),
            RecipeStep(
                "context",
                _args(intent="resume", result_ref="result:0123456789abcdef01234567", budget_tokens=4000),
                "Compile a bounded resume packet from the selected result set and retain its receipt.",
            ),
        ),
        resources=("polylogue://session/{id}", "polylogue://result-set/{id}"),
        prompts=("resume_context",),
    ),
    Recipe(
        id="forensic-lookup",
        title="Perform a forensic lookup",
        intent="Reconstruct a failure from parser-valid row evidence, exact objects, surrounding transcript, and authority status.",
        family="evidence",
        steps=(
            RecipeStep(
                "explain",
                _args(subject="query"),
                "Confirm grammar, group field, selected unit, and aggregate semantics before execution.",
                example_key="aggregate-events-by-status",
            ),
            RecipeStep(
                "query",
                _args(limit=20, projection="aggregate-with-evidence"),
                "Measure failed versus successful tool-finished events.",
                capture="aggregate_result_ref",
                example_key="aggregate-events-by-status",
            ),
            RecipeStep(
                "query",
                _args(limit=20, projection="action-evidence"),
                "Locate exact failed action refs.",
                capture="failure_result_ref",
                example_key="actions-unacknowledged-failures",
            ),
            RecipeStep(
                "get",
                _args(ref="block:codex-session:demo-receipts:call-receipts-test-fail:0", projection="evidence"),
                "Resolve the exact cited failure block rather than quoting a search snippet.",
            ),
            RecipeStep(
                "read",
                _args(ref="polylogue://session/codex-session:demo-receipts", view="chronicle", limit=20),
                "Read the surrounding chronology and any recovery verification.",
            ),
        ),
        resources=("polylogue://block/{id}", "polylogue://session/{id}"),
        prompts=("postmortem_last", "unacknowledged_failures"),
    ),
    Recipe(
        id="prior-art-search",
        title="Search prior art before changing a subsystem",
        intent="Combine semantic retrieval with file-touch history, then inspect exact prior rationale and outcomes.",
        family="discovery",
        steps=(
            RecipeStep(
                "explain",
                _args(subject="query"),
                "Verify semantic lowering and any readiness dependency.",
                example_key="ranked-boolean-semantic",
            ),
            RecipeStep(
                "query",
                _args(limit=20, projection="session-summary"),
                "Find conceptually related sessions even when vocabulary differs.",
                capture="semantic_result_ref",
                example_key="ranked-boolean-semantic",
            ),
            RecipeStep(
                "query",
                _args(limit=20, projection="file-evidence"),
                "Find concrete edits under the relevant subsystem.",
                capture="file_result_ref",
                example_key="files-repository-path",
            ),
            RecipeStep(
                "read",
                _args(ref="result:0123456789abcdef01234567", view="ranked-evidence", limit=20),
                "Read the retained result set rather than rerunning a changed query.",
            ),
            RecipeStep(
                "get",
                _args(ref="message:codex-session:demo-lineage-fork:fork-a3", projection="evidence"),
                "Resolve the exact message containing the rationale selected from the result set.",
            ),
        ),
        resources=("polylogue://query/{id}", "polylogue://result-set/{id}", "polylogue://message/{id}"),
        prompts=("decisions_about", "sessions_touching_file"),
    ),
    Recipe(
        id="cost-audit",
        title="Audit model/provider cost",
        intent="Measure the declared cohort without mixing exact counters, estimates, missing coverage, or logical and physical grains.",
        family="authority",
        steps=(
            RecipeStep(
                "status",
                _args(scope="sources", include=["coverage", "freshness", "usage-counter-support"]),
                "Establish which origins have exact, partial, estimated, or absent usage evidence.",
            ),
            RecipeStep(
                "query",
                _args(limit=50, projection="cost-rollup"),
                "Compute the requested cohort using declared cost semantics.",
                capture="cost_result_ref",
                example_key="sample-origin-cohort-window",
            ),
            RecipeStep(
                "explain",
                _args(subject="result", ref="result:0123456789abcdef01234567"),
                "Inspect denominator, physical/logical grain, missing counts, estimate policy, and continuation state.",
            ),
            RecipeStep(
                "read",
                _args(ref="result:0123456789abcdef01234567", view="cost-evidence", limit=50),
                "Read per-session evidence and continue through every exhaustive page required by the claim.",
            ),
            RecipeStep(
                "get",
                _args(ref="session:codex-session:demo-receipts", projection="usage-provenance"),
                "Resolve a representative source record when a counter or estimate is disputed.",
            ),
        ),
        resources=("polylogue://result-set/{id}", "polylogue://session/{id}"),
        prompts=("cost_of",),
    ),
)


def query_examples() -> tuple[CheckedQuery, ...]:
    """Every query the manual teaches, projected from declared discovery rows.

    The catalog is the declared featured set plus the exact rows the continuity
    recipes execute, so the manual cannot teach an expression that the recipes
    do not run and cannot run one the catalog does not teach.
    """

    keys = [example.key for example in query_discovery_examples(featured=True)]
    for recipe in RECIPES:
        keys.extend(key for key in recipe.query_declarations if key not in keys)
    return tuple(checked_query(key) for key in keys)


QUERY_EXAMPLES: tuple[CheckedQuery, ...] = query_examples()


def origin_meanings() -> tuple[OriginMeaning, ...]:
    """Project the current OriginSpec descriptions into the manual model."""
    return tuple(OriginMeaning(token, meaning) for token, meaning in public_origin_meanings(include_non_public=True))


# Backwards-compatible import for callers that consume the static contract.
# Renderers and payload builders call ``origin_meanings()`` so declaration
# metadata changes are reflected without editing this module.
ORIGIN_MEANINGS: tuple[OriginMeaning, ...] = origin_meanings()
if tuple(item.token for item in ORIGIN_MEANINGS) != tuple(item.value for item in Origin):
    raise RuntimeError("agent source coverage must follow the authoritative Origin enum exactly")

CLIENT_DELIVERIES: tuple[ClientDelivery, ...] = (
    ClientDelivery(
        "claude-code",
        "Merge only the named polylogue entry in the native Claude MCP configuration.",
        "Install a SessionStart hook whose additionalContext is the complete generated standing manual.",
        "Install the generated deep reference as an owned local file.",
        "Hook ownership, idempotent merge, capability/env selection, drift detection, and lossless uninstall are unchanged.",
        "Only the generated content, target manifest, declared tool vocabulary, continuation recipe, and cache digest change.",
    ),
    ClientDelivery(
        "codex",
        "Merge only [mcp_servers.polylogue] in the native Codex TOML configuration.",
        "Install a marked managed block in the effective global AGENTS.override.md or AGENTS.md without overwriting operator text.",
        "Install the generated deep reference beside the managed guidance.",
        "Override precedence detection, marker ownership, idempotency, and lossless uninstall are unchanged.",
        "The managed block is regenerated from the live declarations; no retired tool-name list remains.",
    ),
    ClientDelivery(
        "gemini",
        "Merge only mcpServers.polylogue in Gemini settings JSON.",
        "Install a marked managed block in GEMINI.md as persistent instruction.",
        "Install the generated deep reference as an owned local file.",
        "JSON merge ownership, marker ownership, idempotency, and lossless uninstall are unchanged.",
        "The persistent instruction and target manifest use the declared tool contract.",
    ),
    ClientDelivery(
        "hermes",
        "Merge only mcp_servers.polylogue in Hermes YAML.",
        "Install the complete generated manual inside the owned productivity/polylogue SKILL.md.",
        "Include the generated deep reference in the owned skill directory.",
        "YAML merge ownership, skill ownership, idempotency, and lossless uninstall are unchanged.",
        "The skill body, recipes, capability opt-ins, and cache digest are regenerated for the declared tool surface.",
    ),
)


def tool_contract_payload() -> dict[str, object]:
    """Return the target transaction contract as stable JSON-compatible data."""

    rows: list[dict[str, object]] = []
    for contract in TOOL_CONTRACTS:
        row = asdict(contract)
        row["result_semantics"] = [semantic.value for semantic in contract.result_semantics]
        row["examples"] = [
            {
                "id": example.id,
                "title": example.title,
                "arguments": example.arguments_dict(),
                "result_note": example.result_note,
            }
            for example in contract.examples
        ]
        rows.append(row)
    return {
        "schema_version": 1,
        "content_version": ASSET_VERSION,
        "default_read_tools": list(DEFAULT_READ_TOOLS),
        "privileged_tools": list(PRIVILEGED_TOOLS),
        "transactions": rows,
    }


def recipe_payload() -> dict[str, object]:
    """Return the stable continuity recipe catalog as JSON-compatible data."""

    recipes: list[dict[str, object]] = []
    for recipe in RECIPES:
        row = asdict(recipe)
        row["steps"] = [
            {
                "tool": step.tool,
                "arguments": step.arguments_dict(),
                "purpose": step.purpose,
                "capture": step.capture,
                "example_key": step.example_key,
            }
            for step in recipe.steps
        ]
        row["query_declarations"] = list(recipe.query_declarations)
        row["queries"] = [asdict(query) for query in recipe.queries]
        recipes.append(row)
    return {"schema_version": 1, "content_version": ASSET_VERSION, "recipes": recipes}


def integration_spec_payload() -> dict[str, object]:
    """Return the static client, capability, coverage, and MCP target contract."""

    return {
        "schema_version": 2,
        "content_version": ASSET_VERSION,
        "clients": list(CLIENTS),
        "mcp_capability_flags": ["write", "judge", "maintenance"],
        "guidance_modes": list(GUIDANCE_MODES),
        "capability_families": [asdict(family) for family in CAPABILITY_FAMILIES],
        "origins": [asdict(origin) for origin in origin_meanings()],
        "client_delivery": [asdict(delivery) for delivery in CLIENT_DELIVERIES],
        "target_tools": list(ALL_TARGET_TOOLS),
        "default_read_tools": list(DEFAULT_READ_TOOLS),
        "privileged_tools": list(PRIVILEGED_TOOLS),
        "target_resources": [asdict(resource) for resource in TARGET_RESOURCES],
        "target_prompts": [asdict(prompt) for prompt in TARGET_PROMPTS],
        "state_schema_version": 1,
        "manual_resources": [
            "polylogue://agent/manual",
            "polylogue://agent/reference",
            "polylogue://agent/manifest",
        ],
        "schema_status": TARGET_SCHEMA_STATUS,
        "verification": [
            "Every generated argument is checked against the registered MCPServer signature.",
            "Run devtools render agent-manual and devtools render all --check after declaration changes.",
            "Run devtools gate agent-integration --require-live before publishing the package.",
        ],
    }


__all__ = [
    "ALL_DECLARED_TOOLS",
    "ALL_TARGET_TOOLS",
    "ASSET_VERSION",
    "CAPABILITY_FAMILIES",
    "CLIENTS",
    "CLIENT_DELIVERIES",
    "CONTINUATION_SENTINEL",
    "DEFAULT_READ_TOOLS",
    "GUIDANCE_MODES",
    "ORIGIN_MEANINGS",
    "origin_meanings",
    "PRIVILEGED_TOOLS",
    "QUERY_EXAMPLES",
    "checked_query",
    "query_examples",
    "RECIPES",
    "TARGET_SCHEMA_STATUS",
    "TOOL_CONTRACTS",
    "TOOL_CONTRACT_BY_NAME",
    "AgentClient",
    "CapabilityFamily",
    "CheckedQuery",
    "ConfirmationGate",
    "ClientDelivery",
    "GuidanceMode",
    "OriginMeaning",
    "QuerySurface",
    "Recipe",
    "RecipeStep",
    "ToolArgument",
    "ToolContract",
    "ToolExample",
    "integration_spec_payload",
    "recipe_payload",
    "tool_contract_payload",
]
