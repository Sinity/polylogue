"""Product query-action workflow registry for executable docs and tests (#2305)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

from polylogue.scenarios import DEMO_CLAUDE_CODE_SESSION_ID

JsonPathSegment: TypeAlias = str | int
JsonPath: TypeAlias = tuple[JsonPathSegment, ...]
JsonKind: TypeAlias = Literal["object", "array", "string", "integer", "number", "boolean", "null", "any"]
OutputKind: TypeAlias = Literal["json_object", "json_array", "human"]
WorkflowSurface: TypeAlias = Literal["cli", "daemon", "mcp", "web", "docs", "completion"]


@dataclass(frozen=True, slots=True)
class JsonExpectation:
    """A structural assertion over a golden-path JSON output."""

    path: JsonPath
    kind: JsonKind = "any"


@dataclass(frozen=True, slots=True)
class QueryActionWorkflow:
    """One product workflow stitched from query selection to an action surface."""

    id: str
    title: str
    intent: str
    query_shape: str
    action_sequence: str
    action_paths: tuple[tuple[str, ...], ...]
    read_views: tuple[str, ...]
    selector_policy: str
    cardinality_policy: str
    safety_policy: str
    output_policy: str
    evidence_policy: str
    surfaces: tuple[WorkflowSurface, ...]
    cli_example: str


@dataclass(frozen=True, slots=True)
class ExecutableWorkflowGoldenPath:
    """A demo-archive command that must keep a product workflow executable."""

    id: str
    workflow_id: str
    description: str
    command: tuple[str, ...]
    action_path: tuple[str, ...]
    output_kind: OutputKind
    json_expectations: tuple[JsonExpectation, ...] = ()
    stdout_contains: tuple[str, ...] = ()
    required_affordance_ids: tuple[str, ...] = ()

    @property
    def command_text(self) -> str:
        return "polylogue " + " ".join(self.command)


@dataclass(frozen=True, slots=True)
class ActionUnitEvidence:
    """Evidence unit proving that an action is grounded in a concrete target."""

    action_id: str
    evidence_unit: str
    proof_surface: str
    negative_guard: str


@dataclass(frozen=True, slots=True)
class ProductVerbMatrixRow:
    """Product-only verb row for executable sub-actions not exposed as a top-level action contract."""

    action_id: str
    target_input: str
    cardinality: str
    safety: str
    formats: tuple[str, ...]
    destinations: tuple[str, ...]
    selection_confirmation: str
    next_actions: tuple[str, ...]


REQUIRED_WORKFLOW_IDS: frozenset[str] = frozenset(
    {
        "find-then-read-messages",
        "find-then-recovery",
        "find-then-context-pack",
        "find-then-continue",
        "candidate-assertion-review",
        "resolve-ref-drilldown",
        "browser-capture-status",
    }
)


QUERY_ACTION_WORKFLOWS: tuple[QueryActionWorkflow, ...] = (
    QueryActionWorkflow(
        id="find-then-read-messages",
        title="Find then read normalized messages",
        intent="Turn a query result into a bounded message inspection surface without changing the archive.",
        query_shape="find QUERY then read --view messages [--limit N] [--format text|json|ndjson]",
        action_sequence="find → read(messages)",
        action_paths=(("find",), ("read",)),
        read_views=("messages",),
        selector_policy="Text queries may match many sessions; exact refs use id: or session: and must not broaden to FTS.",
        cardinality_policy="read requires a single target unless --all or a bulk read/export mode is chosen.",
        safety_policy="safe read-only action; no confirmation command.",
        output_policy="Human text by default; JSON and NDJSON expose the archive_messages_payload contract.",
        evidence_policy="Message rows carry target_ref, anchor, copy/open actions, timestamps, roles, and content-block evidence.",
        surfaces=("cli", "daemon", "web", "completion", "docs"),
        cli_example="polylogue find 'pytest' then read --view messages --limit 5",
    ),
    QueryActionWorkflow(
        id="find-then-recovery",
        title="Find then compile recovery context",
        intent="Transform one session into successor-agent recovery material with evidence refs and caveats.",
        query_shape="find EXACT_REF then read --view recovery [--recovery-report continue|blame|work-packet] [--format markdown|json]",
        action_sequence="find → read(recovery)",
        action_paths=(("find",), ("read",)),
        read_views=("recovery",),
        selector_policy="Use an exact session ref when operator intent is a specific run; text queries must select before recovery.",
        cardinality_policy="Recovery is singleton because the output is a successor handoff for one session.",
        safety_policy="safe derived read; raw provider payloads stay behind the explicit raw view.",
        output_policy="Markdown default; JSON exposes the RecoveryDigest/ContextImage-adjacent payload.",
        evidence_policy="Recovery sections report session refs, evidence refs, size metrics, tool summaries, candidates, and degraded states.",
        surfaces=("cli", "daemon", "web", "completion", "docs"),
        cli_example="polylogue find id:claude-code-session:... then read --view recovery --format json",
    ),
    QueryActionWorkflow(
        id="find-then-context-pack",
        title="Find then package project context",
        intent="Bundle a query or project scope into a bounded context pack for handoff.",
        query_shape="find QUERY then read --view context-pack [--max-sessions N] [--max-messages N]",
        action_sequence="find → read(context-pack)",
        action_paths=(("find",), ("read",)),
        read_views=("context-pack",),
        selector_policy="Query terms define the pack scope; project flags may narrow by repo, path, origin, or time window.",
        cardinality_policy="Context-pack is explicitly multi-session but bounded by max_sessions and max_messages.",
        safety_policy="safe derived read; filesystem paths are redacted unless the operator opts out.",
        output_policy="Markdown pack surface; unsupported formats fail instead of silently changing view semantics.",
        evidence_policy="Pack output lists scoped sessions, excerpts, omitted/limited material, and redaction posture.",
        surfaces=("cli", "daemon", "web", "completion", "docs"),
        cli_example="polylogue find 'repo:polylogue pytest' then read --view context-pack --max-sessions 5",
    ),
    QueryActionWorkflow(
        id="find-then-continue",
        title="Find then continue from selected context",
        intent="Compile a continuation handoff from one session or a ranked continuation candidate list.",
        query_shape="find EXACT_REF then continue [--format json] or continue --candidates --repo PATH",
        action_sequence="find → continue",
        action_paths=(("find",), ("continue",)),
        read_views=("recovery",),
        selector_policy="Continuation from query results is singleton; candidate mode uses repo/cwd/recent-file evidence instead of query text.",
        cardinality_policy="continue rejects ambiguous result sets until a selector narrows the seed session.",
        safety_policy="safe read/compile action; no archive mutation.",
        output_policy="Human by default; --format json emits the shared ContextImage payload.",
        evidence_policy="Seed refs, read_views, assertions/candidates flags, freshness, and caveats travel in the context image.",
        surfaces=("cli", "daemon", "mcp", "completion", "docs"),
        cli_example="polylogue find id:claude-code-session:... then continue --format json",
    ),
    QueryActionWorkflow(
        id="candidate-assertion-review",
        title="Review candidate assertions",
        intent="List, accept, reject, defer, or supersede model-produced assertion candidates through explicit judgment commands.",
        query_shape="find EXACT_REF then mark candidates list|accept|reject|defer|supersede",
        action_sequence="find → mark candidates",
        action_paths=(("find",), ("mark",)),
        read_views=("recovery",),
        selector_policy="Candidates are scoped to a selected session or target ref; confidence alone never admits an assertion.",
        cardinality_policy="Each candidate mutation names one candidate id and one target/session scope.",
        safety_policy="mutating; changes are explicit user overlay operations with accept/reject/defer/supersede verbs.",
        output_policy="Human summaries and JSON candidate lists/operation receipts.",
        evidence_policy="Candidate rows include target refs, evidence refs, claim class, status, and judgment operation trace.",
        surfaces=("cli", "daemon", "completion", "docs"),
        cli_example="polylogue find id:claude-code-session:... then mark candidates list --format json",
    ),
    QueryActionWorkflow(
        id="resolve-ref-drilldown",
        title="Resolve exact refs and drill down",
        intent="Resolve a durable ref into the narrowest supported read surface without treating text query hits as authority.",
        query_shape="polylogue read REF or find id:SESSION then select/read",
        action_sequence="resolve_ref → select/read",
        action_paths=(("read",), ("select",)),
        read_views=("summary", "messages", "raw", "recovery"),
        selector_policy="Exact refs are identity filters; unmatched refs return no target instead of falling back to broad FTS.",
        cardinality_policy="An exact ref resolves to one object shape: session, message, block, assertion, or operation.",
        safety_policy="safe read-only drilldown; raw view remains explicit because it may expose source payloads.",
        output_policy="Direct ref reads are JSON; query-selected drilldowns follow the selected read view format contract.",
        evidence_policy="Resolved payload includes target_ref/identity_key or an explicit unresolved state.",
        surfaces=("cli", "daemon", "mcp", "completion", "docs"),
        cli_example="polylogue read session:claude-code-session:63705dcc-f3e5-4378-8118-8bc21e53bbb6",
    ),
    QueryActionWorkflow(
        id="browser-capture-status",
        title="Browser capture status and next action",
        intent="Expose capture/receiver status as an operator-visible workflow before live browser capture is trusted.",
        query_shape="polylogue ops status or daemon status strip/browser-capture readiness routes",
        action_sequence="ops → browser-capture status",
        action_paths=(("ops",),),
        read_views=(),
        selector_policy="Runtime status is not a source selector; it reports capture readiness, receiver URL, and archive materialization state.",
        cardinality_policy="Operational singleton over the local runtime and configured receiver.",
        safety_policy="operational read; import/capture mutations remain separate and visible.",
        output_policy="Human status chips and API status JSON include freshness/readiness, not hidden censorship.",
        evidence_policy="Status links to daemon health, receiver state, source roots, and follow-up actions when unavailable.",
        surfaces=("daemon", "web", "docs"),
        cli_example="polylogue ops status",
    ),
)


QUERY_ACTION_WORKFLOW_BY_ID: dict[str, QueryActionWorkflow] = {entry.id: entry for entry in QUERY_ACTION_WORKFLOWS}


PRODUCT_VERB_MATRIX_EXTRA_ROWS: tuple[ProductVerbMatrixRow, ...] = (
    ProductVerbMatrixRow(
        action_id="mark candidates",
        target_input="selection / candidate assertion",
        cardinality="singleton candidate judgment",
        safety="mutating",
        formats=("human", "json"),
        destinations=("terminal", "stdout", "api", "mcp"),
        selection_confirmation="polylogue mark candidates list --target-ref REF",
        next_actions=("read", "mark candidates list"),
    ),
)


ACTION_UNIT_EVIDENCE: tuple[ActionUnitEvidence, ...] = (
    ActionUnitEvidence(
        action_id="select",
        evidence_unit="session target_ref / identity_key",
        proof_surface="select --json emits id, origin, title, and date for the chosen row.",
        negative_guard="Zero matches produce no selected target; multi-match selection remains explicit.",
    ),
    ActionUnitEvidence(
        action_id="read",
        evidence_unit="session/message/block target_ref plus read-view payload",
        proof_surface="read --view messages/recovery/raw/context uses shared CLI and HTTP read-view profiles.",
        negative_guard="Unsupported formats/views raise usage errors rather than widening to another surface.",
    ),
    ActionUnitEvidence(
        action_id="continue",
        evidence_unit="ContextImage seed_refs and read_views",
        proof_surface="continue --format json records seed_refs, redaction policy, assertions, candidates, and context segments.",
        negative_guard="Ambiguous query results require selection before continuation.",
    ),
    ActionUnitEvidence(
        action_id="analyze",
        evidence_unit="query-scoped stats/facet rows",
        proof_surface="analyze --facets --format json exposes scoped and global buckets.",
        negative_guard="Unsupported grouping/format combinations fail loudly.",
    ),
    ActionUnitEvidence(
        action_id="mark",
        evidence_unit="user overlay operation over selected session refs",
        proof_surface="mark commands emit operation receipts and persist user marks/annotations.",
        negative_guard="Multi-match mutations require --all or --first.",
    ),
    ActionUnitEvidence(
        action_id="mark candidates",
        evidence_unit="candidate assertion id plus judgment operation",
        proof_surface="mark candidates list/accept/reject/defer/supersede routes through candidate assertion review.",
        negative_guard="Candidate confidence is never admission authority; human judgment command is required.",
    ),
    ActionUnitEvidence(
        action_id="delete",
        evidence_unit="session ids and dry-run mutation preview",
        proof_surface="delete --dry-run returns affected session ids/counts before any destructive operation.",
        negative_guard="Actual deletion requires --yes and explicit multi-match scope.",
    ),
    ActionUnitEvidence(
        action_id="browser capture status",
        evidence_unit="daemon/browser-capture readiness state",
        proof_surface="web status strip and daemon JSON surface capture freshness, receiver availability, and next action.",
        negative_guard="Capture status does not silently import or redact; mutations remain operator-triggered.",
    ),
)


EXECUTABLE_WORKFLOW_GOLDEN_PATHS: tuple[ExecutableWorkflowGoldenPath, ...] = (
    ExecutableWorkflowGoldenPath(
        id="select-exact-session-json",
        workflow_id="resolve-ref-drilldown",
        description="Exact id query selects the demo Claude Code session without broad FTS fallback.",
        command=("find", f"id:{DEMO_CLAUDE_CODE_SESSION_ID}", "then", "select", "--json"),
        action_path=("select",),
        output_kind="json_object",
        json_expectations=(JsonExpectation(("id",), "string"), JsonExpectation(("origin",), "string")),
        stdout_contains=(DEMO_CLAUDE_CODE_SESSION_ID, '"origin":"claude-code-session"'),
        required_affordance_ids=("select",),
    ),
    ExecutableWorkflowGoldenPath(
        id="select-exact-session-ref-json",
        workflow_id="resolve-ref-drilldown",
        description="Exact session: ref query selects the demo Claude Code session without broad FTS fallback.",
        command=("find", f"session:{DEMO_CLAUDE_CODE_SESSION_ID}", "then", "select", "--json"),
        action_path=("select",),
        output_kind="json_object",
        json_expectations=(JsonExpectation(("id",), "string"), JsonExpectation(("origin",), "string")),
        stdout_contains=(DEMO_CLAUDE_CODE_SESSION_ID, '"origin":"claude-code-session"'),
        required_affordance_ids=("select",),
    ),
    ExecutableWorkflowGoldenPath(
        id="read-messages-json",
        workflow_id="find-then-read-messages",
        description="Query-selected message read exposes the normalized message JSON payload.",
        command=(
            "find",
            f"id:{DEMO_CLAUDE_CODE_SESSION_ID}",
            "then",
            "read",
            "--view",
            "messages",
            "--limit",
            "2",
            "--format",
            "json",
        ),
        action_path=("read",),
        output_kind="json_object",
        json_expectations=(
            JsonExpectation(("session_id",), "string"),
            JsonExpectation(("messages",), "array"),
            JsonExpectation(("messages", 0, "target_ref"), "object"),
            JsonExpectation(("messages", 0, "actions"), "object"),
        ),
        stdout_contains=(DEMO_CLAUDE_CODE_SESSION_ID, '"messages"'),
        required_affordance_ids=("read",),
    ),
    ExecutableWorkflowGoldenPath(
        id="read-messages-human",
        workflow_id="find-then-read-messages",
        description="Human message read keeps the same selected target but renders operator-readable content.",
        command=(
            "find",
            f"id:{DEMO_CLAUDE_CODE_SESSION_ID}",
            "then",
            "read",
            "--view",
            "messages",
            "--limit",
            "2",
        ),
        action_path=("read",),
        output_kind="human",
        stdout_contains=("The module structure looks good", "1 passed"),
        required_affordance_ids=("read",),
    ),
    ExecutableWorkflowGoldenPath(
        id="read-recovery-json",
        workflow_id="find-then-recovery",
        description="Recovery view compiles evidence-rich successor material for the selected session.",
        command=(
            "find",
            f"id:{DEMO_CLAUDE_CODE_SESSION_ID}",
            "then",
            "read",
            "--view",
            "recovery",
            "--format",
            "json",
        ),
        action_path=("read",),
        output_kind="json_object",
        json_expectations=(
            JsonExpectation(("status",), "string"),
            JsonExpectation(("result", "recovery", "session_id"), "string"),
            JsonExpectation(("result", "recovery", "tool_summaries"), "array"),
            JsonExpectation(("result", "recovery", "size_metrics"), "object"),
        ),
        stdout_contains=(DEMO_CLAUDE_CODE_SESSION_ID, '"recovery"'),
        required_affordance_ids=("read",),
    ),
    ExecutableWorkflowGoldenPath(
        id="continue-json",
        workflow_id="find-then-continue",
        description="Continuation workflow emits the ContextImage seed refs and recovery segment list.",
        command=("find", f"id:{DEMO_CLAUDE_CODE_SESSION_ID}", "then", "continue", "--format", "json"),
        action_path=("continue",),
        output_kind="json_object",
        json_expectations=(
            JsonExpectation(("spec",), "object"),
            JsonExpectation(("spec", "seed_refs"), "array"),
            JsonExpectation(("segments",), "array"),
        ),
        stdout_contains=(f"session:{DEMO_CLAUDE_CODE_SESSION_ID}", '"purpose": "continue"'),
        required_affordance_ids=("continue",),
    ),
    ExecutableWorkflowGoldenPath(
        id="analyze-facets-json",
        workflow_id="find-then-read-messages",
        description="Query-selected analysis exposes scoped and global JSON facet buckets.",
        command=("find", "pytest", "then", "analyze", "--facets", "--format", "json"),
        action_path=("analyze",),
        output_kind="json_object",
        json_expectations=(
            JsonExpectation(("scoped_to_query",), "boolean"),
            JsonExpectation(("scoped",), "object"),
            JsonExpectation(("global",), "object"),
            JsonExpectation(("scoped", "origins"), "object"),
        ),
        stdout_contains=('"claude-code-session"', '"codex-session"'),
        required_affordance_ids=("analyze",),
    ),
    ExecutableWorkflowGoldenPath(
        id="mark-candidates-list-json",
        workflow_id="candidate-assertion-review",
        description="Candidate review lists pending candidate assertions through an explicit judgment surface.",
        command=(
            "find",
            f"id:{DEMO_CLAUDE_CODE_SESSION_ID}",
            "then",
            "mark",
            "candidates",
            "list",
            "--format",
            "json",
        ),
        action_path=("mark",),
        output_kind="json_object",
        json_expectations=(
            JsonExpectation(("items",), "array"),
            JsonExpectation(("total",), "integer"),
            JsonExpectation(("statuses",), "array"),
        ),
        stdout_contains=('"statuses"', '"candidate"'),
        required_affordance_ids=("mark",),
    ),
    ExecutableWorkflowGoldenPath(
        id="delete-dry-run-json",
        workflow_id="resolve-ref-drilldown",
        description="Destructive workflow stays a preview until the explicit confirmation guard is supplied.",
        command=("find", f"id:{DEMO_CLAUDE_CODE_SESSION_ID}", "then", "delete", "--dry-run"),
        action_path=("delete",),
        output_kind="json_object",
        json_expectations=(
            JsonExpectation(("status",), "string"),
            JsonExpectation(("session_ids",), "array"),
            JsonExpectation(("session_count",), "integer"),
        ),
        stdout_contains=('"status": "preview"', DEMO_CLAUDE_CODE_SESSION_ID),
        required_affordance_ids=("delete",),
    ),
)


def _validate_workflows() -> None:
    missing = REQUIRED_WORKFLOW_IDS - set(QUERY_ACTION_WORKFLOW_BY_ID)
    if missing:
        raise ValueError(f"query-action workflow registry is missing required workflows: {sorted(missing)}")
    duplicate_golden_ids = len({entry.id for entry in EXECUTABLE_WORKFLOW_GOLDEN_PATHS}) != len(
        EXECUTABLE_WORKFLOW_GOLDEN_PATHS
    )
    if duplicate_golden_ids:
        raise ValueError("executable workflow golden paths contain duplicate ids")
    for golden in EXECUTABLE_WORKFLOW_GOLDEN_PATHS:
        if golden.workflow_id not in QUERY_ACTION_WORKFLOW_BY_ID:
            raise ValueError(f"golden path {golden.id!r} references unknown workflow {golden.workflow_id!r}")
        if golden.output_kind == "human" and golden.json_expectations:
            raise ValueError(f"human golden path {golden.id!r} must not declare JSON expectations")
        if golden.output_kind != "human" and not golden.json_expectations:
            raise ValueError(f"JSON golden path {golden.id!r} must declare JSON expectations")


_validate_workflows()


__all__ = [
    "ACTION_UNIT_EVIDENCE",
    "EXECUTABLE_WORKFLOW_GOLDEN_PATHS",
    "JsonExpectation",
    "JsonKind",
    "JsonPath",
    "JsonPathSegment",
    "OutputKind",
    "PRODUCT_VERB_MATRIX_EXTRA_ROWS",
    "ProductVerbMatrixRow",
    "QUERY_ACTION_WORKFLOW_BY_ID",
    "QUERY_ACTION_WORKFLOWS",
    "QueryActionWorkflow",
    "REQUIRED_WORKFLOW_IDS",
    "WorkflowSurface",
]
