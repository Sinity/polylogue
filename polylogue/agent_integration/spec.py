"""Typed authority for the six-tool-era standing agent manual.

This module keeps the architecture delivered by the beads-06 package—typed
capabilities, checked queries, recipes, client delivery declarations, and
rendered package assets—but replaces its 103-tool vocabulary with a normalized
view of the t46.8 target declarations.  The view is deliberately tolerant of
the in-flight declaration names in the current snapshot:

* ``graph`` is folded into ``read`` until t46.8.2 lands the six-tool surface.
* ``maintenance`` is presented as the mission-owned ``operate`` transaction
  until t46.8.3 renames the declaration.

The manual input schemas below are cutover contract parameters, not claims that
the compatibility server already registers those signatures.  The verifier
checks them against live FastMCP signatures once all target tools are present.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, TypeAlias

from polylogue.core.enums import Origin
from polylogue.declarations import JSONValue
from polylogue.mcp.declarations import (
    PRIVILEGED_ALGEBRA,
    TARGET_DEFAULT_READ_ALGEBRA,
    TARGET_PROMPTS,
    TARGET_RESOURCES,
    MCPResultSemantics,
    MCPRole,
    MCPTransactionDeclaration,
)

ASSET_VERSION = "2026-07-17.6tool-r01"
AgentClient = Literal["claude-code", "codex", "gemini", "hermes"]
GuidanceMode = Literal["full", "mcp-only", "off"]
QuerySurface = Literal["session", "terminal"]
SchemaStatus = Literal["cutover-parameterized", "live-verified"]
ArgumentKind = Literal["string", "integer", "boolean", "array", "object"]

TARGET_SCHEMA_STATUS: SchemaStatus = "cutover-parameterized"
CLIENTS: tuple[AgentClient, ...] = ("claude-code", "codex", "gemini", "hermes")
ROLES: tuple[MCPRole, ...] = ("read", "write", "review", "admin")
GUIDANCE_MODES: tuple[GuidanceMode, ...] = ("full", "mcp-only", "off")
ROLE_ORDER: dict[MCPRole, int] = {"read": 0, "write": 1, "review": 2, "admin": 3}
DEFAULT_READ_TOOLS: tuple[str, ...] = ("query", "read", "get", "explain", "context", "status")
PRIVILEGED_TOOLS: tuple[str, ...] = ("write", "judge", "run", "operate")
ALL_TARGET_TOOLS: tuple[str, ...] = (*DEFAULT_READ_TOOLS, *PRIVILEGED_TOOLS)
CONTINUATION_SENTINEL = "$continuation"

Arguments: TypeAlias = tuple[tuple[str, JSONValue], ...]


@dataclass(frozen=True, slots=True)
class CapabilityFamily:
    """One stable reason for an agent to invoke Polylogue."""

    id: str
    title: str
    minimum_role: MCPRole
    first_tool: str


@dataclass(frozen=True, slots=True)
class ToolArgument:
    """One declared input parameter in the cutover contract."""

    name: str
    kind: ArgumentKind
    required_initial: bool
    description: str


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
    minimum_role: MCPRole
    purpose: str
    source_declarations: tuple[str, ...]
    result_semantics: tuple[MCPResultSemantics, ...]
    arguments: tuple[ToolArgument, ...]
    examples: tuple[ToolExample, ...]
    supports_continuation: bool
    emits_result_ref: bool
    schema_status: SchemaStatus = TARGET_SCHEMA_STATUS

    @property
    def argument_names(self) -> frozenset[str]:
        return frozenset(argument.name for argument in self.arguments)

    @property
    def required_initial_arguments(self) -> tuple[str, ...]:
        return tuple(argument.name for argument in self.arguments if argument.required_initial)


@dataclass(frozen=True, slots=True)
class CheckedQuery:
    """A query expression and the production parser surface that owns it."""

    expression: str
    surface: QuerySurface
    purpose: str
    source_test: str


@dataclass(frozen=True, slots=True)
class RecipeStep:
    """One six-tool continuity step."""

    tool: str
    arguments: Arguments
    purpose: str
    capture: str | None = None

    def arguments_dict(self) -> dict[str, JSONValue]:
        return dict(self.arguments)


@dataclass(frozen=True, slots=True)
class Recipe:
    """Executable continuity workflow made exclusively from the six read tools."""

    id: str
    title: str
    intent: str
    family: str
    steps: tuple[RecipeStep, ...]
    queries: tuple[CheckedQuery, ...]
    resources: tuple[str, ...] = ()
    prompts: tuple[str, ...] = ()


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


def _arg(name: str, kind: ArgumentKind, required: bool, description: str) -> ToolArgument:
    return ToolArgument(name, kind, required, description)


def _example(id: str, title: str, result_note: str, **arguments: JSONValue) -> ToolExample:
    return ToolExample(id=id, title=title, arguments=_args(**arguments), result_note=result_note)


def _target_declaration_index() -> dict[str, MCPTransactionDeclaration]:
    return {item.name: item for item in (*TARGET_DEFAULT_READ_ALGEBRA, *PRIVILEGED_ALGEBRA)}


def _sources(*names: str, optional: tuple[str, ...] = ()) -> tuple[str, ...]:
    index = _target_declaration_index()
    selected: list[str] = []
    for name in names:
        if name not in index:
            raise RuntimeError(f"agent manual requires missing MCP target declaration {name!r}")
        selected.append(name)
    selected.extend(name for name in optional if name in index)
    return tuple(selected)


def _semantics(source_names: tuple[str, ...]) -> tuple[MCPResultSemantics, ...]:
    index = _target_declaration_index()
    result: list[MCPResultSemantics] = []
    for source_name in source_names:
        declaration = index[source_name]
        semantics = declaration.result_semantics
        for semantic in semantics:
            if semantic not in result:
                result.append(semantic)
    return tuple(result)


def _minimum_role(source_names: tuple[str, ...]) -> MCPRole:
    index = _target_declaration_index()
    roles = {index[name].minimum_role for name in source_names}
    if len(roles) != 1:
        raise RuntimeError(f"manual transaction sources disagree on role: {source_names!r} -> {sorted(roles)}")
    return roles.pop()


def _contract(
    *,
    name: str,
    source_names: tuple[str, ...],
    purpose: str,
    arguments: tuple[ToolArgument, ...],
    examples: tuple[ToolExample, ...],
    supports_continuation: bool,
    emits_result_ref: bool,
) -> ToolContract:
    return ToolContract(
        name=name,
        minimum_role=_minimum_role(source_names),
        purpose=purpose,
        source_declarations=source_names,
        result_semantics=_semantics(source_names),
        arguments=arguments,
        examples=examples,
        supports_continuation=supports_continuation,
        emits_result_ref=emits_result_ref,
    )


_QUERY_SOURCES = _sources("query")
_READ_SOURCES = _sources("read", optional=("graph",))
_GET_SOURCES = _sources("get")
_EXPLAIN_SOURCES = _sources("explain")
_CONTEXT_SOURCES = _sources("context")
_STATUS_SOURCES = _sources("status")
_WRITE_SOURCES = _sources("write")
_JUDGE_SOURCES = _sources("judge")
_RUN_SOURCES = _sources("run")
_OPERATE_SOURCES = _sources("operate") if "operate" in _target_declaration_index() else _sources("maintenance")

TOOL_CONTRACTS: tuple[ToolContract, ...] = (
    _contract(
        name="query",
        source_names=_QUERY_SOURCES,
        purpose="Execute the real expression DSL or a declared typed plan and return a bounded, semantics-labelled result set.",
        arguments=(
            _arg("expression", "string", True, "Parser-owned DSL expression; omit when resuming with continuation."),
            _arg("limit", "integer", False, "Requested page size, subject to server and transport bounds."),
            _arg("projection", "string", False, "Declared result projection such as session-summary or cost-rollup."),
            _arg("continuation", "string", False, "Opaque token from the preceding response; send alone."),
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
        purpose="Read a stable URI/object/evidence ref through a declared view, including topology that the in-flight declarations still call graph.",
        arguments=(
            _arg("ref", "string", True, "Stable object, evidence, result-set, or URI reference."),
            _arg("view", "string", False, "Declared projection/view for the referenced object."),
            _arg("limit", "integer", False, "Page size for collection-like or recursive reads."),
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
            _arg("target_ref", "string", True, "Exact mutation target."),
            _arg("value", "object", True, "Typed operation payload."),
            _arg("expected_generation", "string", False, "Optimistic conflict guard where the operation declares one."),
        ),
        examples=(
            _example(
                "write-tag",
                "Add a review tag through the governed write chokepoint",
                "A mutation receipt with actor, target, effect identity, and resulting generation; no destructive confirmation is invented.",
                operation="tag.add",
                target_ref="session:codex-session:demo-lineage-fork",
                value={"tag": "review"},
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
            _arg("candidate_ref", "string", True, "Exact assertion-candidate identity."),
            _arg("decision", "string", True, "Declared judgment decision."),
            _arg("reason", "string", True, "Evidence-grounded judgment rationale."),
            _arg("expected_generation", "string", False, "Conflict guard for concurrent review."),
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
        purpose="Execute a saved query or governed recipe ref; any nested mutation inherits its own role and confirmation policy.",
        arguments=(
            _arg("ref", "string", True, "Saved-query or recipe ref."),
            _arg("arguments", "object", False, "Typed recipe parameters."),
            _arg("continuation", "string", False, "Opaque token from the preceding run response; send alone."),
        ),
        examples=(
            _example(
                "run-cost-recipe",
                "Run a saved read-only cost audit",
                "A result_ref and receipt for the declared recipe; mutation authority is never gained from the recipe wrapper.",
                ref="recipe:cost-audit",
                arguments={"repo": "polylogue", "since": "2026-07-01"},
            ),
        ),
        supports_continuation=True,
        emits_result_ref=True,
    ),
    _contract(
        name="operate",
        source_names=_OPERATE_SOURCES,
        purpose="Preview, authorize, execute, inspect, and reconcile administrative maintenance through preview-bound confirmation.",
        arguments=(
            _arg("operation", "string", True, "Declared maintenance operation."),
            _arg("phase", "string", True, "preview, execute, status, or reconcile."),
            _arg("target", "object", False, "Bound target selector used during preview."),
            _arg("preview_ref", "string", False, "Preview receipt supplied unchanged for execution."),
            _arg(
                "confirmation_token",
                "string",
                False,
                "Short-lived token bound to actor, archive, operation version, expiry, target set, and preview digest.",
            ),
        ),
        examples=(
            _example(
                "operate-preview",
                "Preview an index rebuild before authorization",
                "A non-mutating preview receipt with target digest and the data required to obtain a bound confirmation token.",
                operation="index.rebuild",
                phase="preview",
                target={"scope": "derived-index"},
            ),
        ),
        supports_continuation=False,
        emits_result_ref=False,
    ),
)

TOOL_CONTRACT_BY_NAME: dict[str, ToolContract] = {contract.name: contract for contract in TOOL_CONTRACTS}
if tuple(TOOL_CONTRACT_BY_NAME) != ALL_TARGET_TOOLS:
    raise RuntimeError("agent tool contracts must remain in six-tool then privileged role order")

CAPABILITY_FAMILIES: tuple[CapabilityFamily, ...] = (
    CapabilityFamily("authority", "Archive identity, source coverage, freshness, and readiness", "read", "status"),
    CapabilityFamily(
        "discovery", "Cross-session, row-level, semantic, aggregate, and prior-art search", "read", "query"
    ),
    CapabilityFamily("evidence", "Exact objects, transcripts, topology, raw evidence, and citations", "read", "read"),
    CapabilityFamily(
        "teaching", "Grammar, fields, values, plans, refs, result semantics, and recovery", "read", "explain"
    ),
    CapabilityFamily(
        "continuity", "Resume, postmortem, forensic, coordination, and bounded context", "read", "context"
    ),
    CapabilityFamily("mutation", "Reversible overlays, judgments, saved runs, and administration", "write", "write"),
)

# Each expression is drawn from or is a direct value-preserving specialization
# of the production parser tests named in ``source_test``.
QUERY_EXAMPLES: tuple[CheckedQuery, ...] = (
    CheckedQuery(
        'repo:polylogue since:7d "json envelope"',
        "session",
        "Compact field, relative-date, and quoted-text clauses.",
        "tests/unit/cli/test_query_expression.py::test_multiple_fields",
    ),
    CheckedQuery(
        "sessions where (repo:polylogue OR origin:chatgpt-export) AND NOT tag:stale",
        "session",
        "Explicit Boolean session predicate.",
        "polylogue/archive/query/expression.py module executable-grammar examples",
    ),
    CheckedQuery(
        "messages where role:assistant AND text:timeout",
        "terminal",
        "Message-row lookup.",
        "polylogue/archive/query/expression.py module executable-grammar examples",
    ),
    CheckedQuery(
        "actions where action:file_edit AND path:polylogue/archive",
        "terminal",
        "Action-row lookup.",
        "polylogue/archive/query/expression.py module executable-grammar examples",
    ),
    CheckedQuery(
        "observed-events where kind:tool_finished AND handler:shell | group by status | count",
        "terminal",
        "Terminal aggregate with declared group field.",
        "tests/unit/cli/test_query_expression.py::test_terminal_observed_event_tool_finished_aggregate_reads_blocks_without_materialization",
    ),
    CheckedQuery(
        "files where action:file_edit AND path:polylogue/archive/query | sort by time desc | limit 20",
        "terminal",
        "File-touch history with deterministic ordering and limit.",
        "tests/unit/cli/test_query_expression.py file-source coverage",
    ),
    CheckedQuery(
        'sessions where semantic:"preview-bound confirmation"',
        "session",
        "Semantic prior-art retrieval.",
        "tests/unit/cli/test_query_expression.py::test_boolean_semantic_predicate_lowers",
    ),
    CheckedQuery(
        "sessions where origin:(claude-code-session|codex-session) AND date >= 2026-07-01",
        "session",
        "Provider cohort for a cost audit.",
        "tests/unit/cli/test_query_expression.py origin alternatives and readable date comparison coverage",
    ),
    CheckedQuery(
        "sessions where repo:polylogue AND NOT tag:complete",
        "session",
        "Likely unfinished work for session resumption.",
        "tests/unit/cli/test_query_expression.py Boolean predicate coverage",
    ),
    CheckedQuery(
        "actions where session.repo:polylogue AND output:failed | sort by time desc | limit 20",
        "terminal",
        "Recent failed effects for resumption or forensics.",
        "tests/unit/cli/test_query_expression.py terminal session-field scoping coverage",
    ),
)
QUERY_EXAMPLE_BY_EXPRESSION = {item.expression: item for item in QUERY_EXAMPLES}


def _checked(expression: str) -> CheckedQuery:
    try:
        return QUERY_EXAMPLE_BY_EXPRESSION[expression]
    except KeyError as exc:
        raise RuntimeError(f"recipe query is not in the checked parser catalog: {expression!r}") from exc


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
                _args(
                    expression="sessions where repo:polylogue AND NOT tag:complete",
                    limit=20,
                    projection="session-summary",
                ),
                "Find likely unfinished sessions.",
                capture="candidate_result_ref",
            ),
            RecipeStep(
                "query",
                _args(
                    expression="actions where session.repo:polylogue AND output:failed | sort by time desc | limit 20",
                    limit=20,
                    projection="action-evidence",
                ),
                "Find recent failed effects that may invalidate an optimistic handoff.",
                capture="failure_result_ref",
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
        queries=(
            _checked("sessions where repo:polylogue AND NOT tag:complete"),
            _checked("actions where session.repo:polylogue AND output:failed | sort by time desc | limit 20"),
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
                _args(
                    subject="query",
                    expression="observed-events where kind:tool_finished AND handler:shell | group by status | count",
                ),
                "Confirm grammar, group field, selected unit, and aggregate semantics before execution.",
            ),
            RecipeStep(
                "query",
                _args(
                    expression="observed-events where kind:tool_finished AND handler:shell | group by status | count",
                    limit=20,
                    projection="aggregate-with-evidence",
                ),
                "Measure failed versus successful shell events.",
                capture="aggregate_result_ref",
            ),
            RecipeStep(
                "query",
                _args(
                    expression="actions where session.repo:polylogue AND output:failed | sort by time desc | limit 20",
                    limit=20,
                    projection="action-evidence",
                ),
                "Locate exact failed action refs.",
                capture="failure_result_ref",
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
        queries=(
            _checked("observed-events where kind:tool_finished AND handler:shell | group by status | count"),
            _checked("actions where session.repo:polylogue AND output:failed | sort by time desc | limit 20"),
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
                _args(subject="query", expression='sessions where semantic:"preview-bound confirmation"'),
                "Verify semantic lowering and any readiness dependency.",
            ),
            RecipeStep(
                "query",
                _args(
                    expression='sessions where semantic:"preview-bound confirmation"',
                    limit=20,
                    projection="session-summary",
                ),
                "Find conceptually related sessions even when vocabulary differs.",
                capture="semantic_result_ref",
            ),
            RecipeStep(
                "query",
                _args(
                    expression="files where action:file_edit AND path:polylogue/archive/query | sort by time desc | limit 20",
                    limit=20,
                    projection="file-evidence",
                ),
                "Find concrete edits under the relevant subsystem.",
                capture="file_result_ref",
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
        queries=(
            _checked('sessions where semantic:"preview-bound confirmation"'),
            _checked("files where action:file_edit AND path:polylogue/archive/query | sort by time desc | limit 20"),
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
                _args(
                    expression="sessions where origin:(claude-code-session|codex-session) AND date >= 2026-07-01",
                    limit=50,
                    projection="cost-rollup",
                ),
                "Compute the requested cohort using declared cost semantics.",
                capture="cost_result_ref",
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
        queries=(_checked("sessions where origin:(claude-code-session|codex-session) AND date >= 2026-07-01"),),
        resources=("polylogue://result-set/{id}", "polylogue://session/{id}"),
        prompts=("cost_of",),
    ),
)

ORIGIN_MEANINGS: tuple[OriginMeaning, ...] = (
    OriginMeaning("claude-code-session", "Claude Code local runtime sessions."),
    OriginMeaning("codex-session", "Codex CLI local runtime sessions."),
    OriginMeaning("gemini-cli-session", "Gemini CLI local runtime sessions."),
    OriginMeaning("hermes-session", "Hermes agent runtime sessions."),
    OriginMeaning("antigravity-session", "Antigravity local brain/session artifacts."),
    OriginMeaning("beads-issue", "Repository Beads issue and history records ingested as archive evidence."),
    OriginMeaning("grok-export", "Grok conversation exports."),
    OriginMeaning("chatgpt-export", "ChatGPT web/data exports."),
    OriginMeaning("claude-ai-export", "Claude web/data exports."),
    OriginMeaning("aistudio-drive", "Google AI Studio or Drive/Takeout conversation exports."),
    OriginMeaning("unknown-export", "Imported material whose provider/source could not be classified reliably."),
)
if tuple(item.token for item in ORIGIN_MEANINGS) != tuple(item.value for item in Origin):
    raise RuntimeError("agent source coverage must follow the authoritative Origin enum exactly")

CLIENT_DELIVERIES: tuple[ClientDelivery, ...] = (
    ClientDelivery(
        "claude-code",
        "Merge only the named polylogue entry in the native Claude MCP configuration.",
        "Install a SessionStart hook whose additionalContext is the complete generated standing manual.",
        "Install the generated deep reference as an owned local file.",
        "Hook ownership, idempotent merge, role/env selection, drift detection, and lossless uninstall are unchanged.",
        "Only the generated content, target manifest, six-tool vocabulary, continuation recipe, and cache digest change.",
    ),
    ClientDelivery(
        "codex",
        "Merge only [mcp_servers.polylogue] in the native Codex TOML configuration.",
        "Install a marked managed block in the effective global AGENTS.override.md or AGENTS.md without overwriting operator text.",
        "Install the generated deep reference beside the managed guidance.",
        "Override precedence detection, marker ownership, idempotency, and lossless uninstall are unchanged.",
        "The managed block is regenerated from the six-tool declarations; no 103-tool name list remains.",
    ),
    ClientDelivery(
        "gemini",
        "Merge only mcpServers.polylogue in Gemini settings JSON.",
        "Install a marked managed block in GEMINI.md as persistent instruction.",
        "Install the generated deep reference as an owned local file.",
        "JSON merge ownership, marker ownership, idempotency, and lossless uninstall are unchanged.",
        "The persistent instruction and target manifest switch to the six-tool contract.",
    ),
    ClientDelivery(
        "hermes",
        "Merge only mcp_servers.polylogue in Hermes YAML.",
        "Install the complete generated manual inside the owned productivity/polylogue SKILL.md.",
        "Include the generated deep reference in the owned skill directory.",
        "YAML merge ownership, skill ownership, idempotency, and lossless uninstall are unchanged.",
        "The skill body, recipes, role ladder, and cache digest are regenerated for the six-tool surface.",
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
            }
            for step in recipe.steps
        ]
        row["queries"] = [asdict(query) for query in recipe.queries]
        recipes.append(row)
    return {"schema_version": 1, "content_version": ASSET_VERSION, "recipes": recipes}


def integration_spec_payload() -> dict[str, object]:
    """Return the static client, role, coverage, and MCP target contract."""

    return {
        "schema_version": 2,
        "content_version": ASSET_VERSION,
        "clients": list(CLIENTS),
        "roles": list(ROLES),
        "guidance_modes": list(GUIDANCE_MODES),
        "capability_families": [asdict(family) for family in CAPABILITY_FAMILIES],
        "origins": [asdict(origin) for origin in ORIGIN_MEANINGS],
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
            "polylogue://agent/manifest/{role}",
        ],
        "schema_status": TARGET_SCHEMA_STATUS,
        "post_cutover_regeneration": [
            "Replace or verify every parameterized argument against the final FastMCP schemas.",
            "Run the live signature lane, then set TARGET_SCHEMA_STATUS to live-verified only after exact parity.",
            "Run devtools render agent-manual and devtools render all --check.",
            "Run devtools verify agent-integration --require-live after the target tools and schema marker agree.",
        ],
    }


__all__ = [
    "ALL_TARGET_TOOLS",
    "ASSET_VERSION",
    "CAPABILITY_FAMILIES",
    "CLIENTS",
    "CLIENT_DELIVERIES",
    "CONTINUATION_SENTINEL",
    "DEFAULT_READ_TOOLS",
    "GUIDANCE_MODES",
    "ORIGIN_MEANINGS",
    "PRIVILEGED_TOOLS",
    "QUERY_EXAMPLES",
    "RECIPES",
    "ROLE_ORDER",
    "ROLES",
    "TARGET_SCHEMA_STATUS",
    "TOOL_CONTRACTS",
    "TOOL_CONTRACT_BY_NAME",
    "AgentClient",
    "CapabilityFamily",
    "CheckedQuery",
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
