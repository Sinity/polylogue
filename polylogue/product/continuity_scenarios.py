"""Black-box continuity replay declarations and independent oracle projections.

The declarations extend Polylogue's existing :mod:`polylogue.scenarios` seam.
They describe operator wording, the public route to exercise, and how route
payloads are projected into facts.  Expected values stay in a separately
planted fixture manifest; production query output is never used to construct
an oracle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

from polylogue.insights.authored_payloads import PayloadDict
from polylogue.scenarios import NamedScenarioSource, ScenarioProjectionSourceKind

ContinuityFailureClass = Literal[
    "source_coverage",
    "discovery",
    "formulation",
    "plan",
    "execution",
    "projection",
    "reasoning",
]
ContinuityAttemptGrade = Literal[
    "reasonable_oversized",
    "wrong_corpus_assumption",
    "weak_lexical_proxy",
    "product_induced_hidden_grammar",
    "execution_failure",
    "unreasonable_query",
]
ContinuityTool = Literal[
    "query_units",
    "provider_usage",
    "explain_query_expression",
    "list_read_view_profiles",
]
ContinuityFactReducer = Literal[
    "count",
    "single",
    "unique_values",
    "unique_count",
    "regex_count",
    "regex_unique_count",
    "sum",
]
JsonPathSegment: TypeAlias = str | int
JsonPath: TypeAlias = tuple[JsonPathSegment, ...]
RouteArgumentValue: TypeAlias = str | int | float | bool | None


@dataclass(frozen=True, slots=True)
class ContinuityBudget:
    """Bounded black-box execution budget for one replay."""

    max_calls: int = 12
    max_page_bytes: int = 25_000
    max_total_bytes: int = 250_000
    max_elapsed_ms: int = 30_000
    max_cancel_grace_ms: int = 1_000


@dataclass(frozen=True, slots=True)
class ContinuityDiscoveryRequirement:
    """Tool and input fields that must be visible before a plan is executed."""

    tool: ContinuityTool
    required_arguments: tuple[str, ...] = ()

    def to_payload(self) -> PayloadDict:
        return {"tool": self.tool, "required_arguments": list(self.required_arguments)}


@dataclass(frozen=True, slots=True)
class ContinuityResultSemantics:
    """Operator-level meaning that makes a route result complete."""

    target_population: str
    exactness: Literal["single", "exact_population", "aggregate", "capability_inventory"]
    evidence_contract: str

    def to_payload(self) -> PayloadDict:
        return {
            "target_population": self.target_population,
            "exactness": self.exactness,
            "evidence_contract": self.evidence_contract,
        }


@dataclass(frozen=True, slots=True)
class ContinuityRouteStep:
    """One real MCP-shaped invocation in a continuity plan."""

    step_id: str
    tool: ContinuityTool
    arguments: tuple[tuple[str, RouteArgumentValue], ...] = ()
    paginate: bool = False
    exact_count_probe: bool = False
    item_identity_path: JsonPath = ()

    def argument_dict(self) -> dict[str, RouteArgumentValue]:
        return dict(self.arguments)

    def to_payload(self) -> PayloadDict:
        return {
            "step_id": self.step_id,
            "tool": self.tool,
            "arguments": dict(self.arguments),
            "paginate": self.paginate,
            "exact_count_probe": self.exact_count_probe,
            "item_identity_path": list(self.item_identity_path),
        }

    @property
    def plan_atom(self) -> str:
        expression = self.argument_dict().get("expression")
        if self.tool == "query_units" and isinstance(expression, str):
            unit = expression.split(maxsplit=1)[0]
            return f"query_units:{unit}"
        return self.tool


@dataclass(frozen=True, slots=True)
class ContinuityFactProjection:
    """Declarative projection from route payloads to one comparable fact."""

    name: str
    step_id: str
    path: JsonPath
    reducer: ContinuityFactReducer
    pattern: str | None = None
    failure_class: ContinuityFailureClass = "projection"

    def to_payload(self) -> PayloadDict:
        payload: PayloadDict = {
            "name": self.name,
            "step_id": self.step_id,
            "path": list(self.path),
            "reducer": self.reducer,
            "failure_class": self.failure_class,
        }
        if self.pattern is not None:
            payload["pattern"] = self.pattern
        return payload


@dataclass(frozen=True, slots=True)
class ContinuityEvidenceProjection:
    """Declarative extraction of citable evidence refs from route rows."""

    step_id: str
    path: JsonPath
    prefix: str = ""

    def to_payload(self) -> PayloadDict:
        return {"step_id": self.step_id, "path": list(self.path), "prefix": self.prefix}


@dataclass(frozen=True, slots=True)
class ContinuityAttemptCurriculumProjection:
    """Route projection used to grade the sanitized incident call curriculum."""

    step_id: str
    path: JsonPath
    case_ids: tuple[str, ...]

    def to_payload(self) -> PayloadDict:
        return {"step_id": self.step_id, "path": list(self.path), "case_ids": list(self.case_ids)}


@dataclass(frozen=True, kw_only=True)
class ContinuityScenarioSpec(NamedScenarioSource):
    """One sparse operator job declared on the repository's scenario seam."""

    operator_question: str
    fixture_key: str
    route_steps: tuple[ContinuityRouteStep, ...]
    fact_projections: tuple[ContinuityFactProjection, ...]
    evidence_projections: tuple[ContinuityEvidenceProjection, ...]
    allowed_query_surfaces: tuple[ContinuityTool, ...]
    workflow_ids: tuple[str, ...]
    canonical_plan_families: tuple[str, ...]
    equivalent_plan_signatures: tuple[tuple[str, ...], ...]
    discovery_requirements: tuple[ContinuityDiscoveryRequirement, ...]
    coverage_inventory: tuple[str, ...]
    result_semantics: ContinuityResultSemantics
    stop_conditions: tuple[str, ...]
    failure_taxonomy: tuple[ContinuityFailureClass, ...]
    mutation_cases: tuple[str, ...]
    attempt_curriculum: ContinuityAttemptCurriculumProjection | None = None
    budget: ContinuityBudget = ContinuityBudget()

    @property
    def scenario_id(self) -> str:
        return self.name

    @property
    def title(self) -> str:
        return self.description

    @property
    def sparse_prompt(self) -> str:
        return self.operator_question

    @property
    def required_facts(self) -> tuple[str, ...]:
        return tuple(projection.name for projection in self.fact_projections)

    @property
    def route_plan_signature(self) -> tuple[str, ...]:
        return tuple(step.plan_atom for step in self.route_steps)

    @property
    def projection_source_kind(self) -> ScenarioProjectionSourceKind:
        return ScenarioProjectionSourceKind.VALIDATION_LANE

    def scenario_payload(self) -> PayloadDict:
        payload = super().scenario_payload()
        payload.update(
            {
                "name": self.name,
                "description": self.description,
                "operator_question": self.operator_question,
                "fixture_key": self.fixture_key,
                "route_steps": [step.to_payload() for step in self.route_steps],
                "fact_projections": [projection.to_payload() for projection in self.fact_projections],
                "evidence_projections": [projection.to_payload() for projection in self.evidence_projections],
                "allowed_query_surfaces": list(self.allowed_query_surfaces),
                "workflow_ids": list(self.workflow_ids),
                "canonical_plan_families": list(self.canonical_plan_families),
                "equivalent_plan_signatures": [list(signature) for signature in self.equivalent_plan_signatures],
                "discovery_requirements": [requirement.to_payload() for requirement in self.discovery_requirements],
                "coverage_inventory": list(self.coverage_inventory),
                "result_semantics": self.result_semantics.to_payload(),
                "stop_conditions": list(self.stop_conditions),
                "failure_taxonomy": list(self.failure_taxonomy),
                "mutation_cases": list(self.mutation_cases),
                "budget": {
                    "max_calls": self.budget.max_calls,
                    "max_page_bytes": self.budget.max_page_bytes,
                    "max_total_bytes": self.budget.max_total_bytes,
                    "max_elapsed_ms": self.budget.max_elapsed_ms,
                    "max_cancel_grace_ms": self.budget.max_cancel_grace_ms,
                },
            }
        )
        if self.attempt_curriculum is not None:
            payload["attempt_curriculum"] = self.attempt_curriculum.to_payload()
        return payload


def _args(**kwargs: RouteArgumentValue) -> tuple[tuple[str, RouteArgumentValue], ...]:
    return tuple(kwargs.items())


def _fact(
    name: str,
    step_id: str,
    *path: JsonPathSegment,
    reducer: ContinuityFactReducer = "single",
    pattern: str | None = None,
    failure_class: ContinuityFailureClass = "projection",
) -> ContinuityFactProjection:
    return ContinuityFactProjection(
        name=name,
        step_id=step_id,
        path=path,
        reducer=reducer,
        pattern=pattern,
        failure_class=failure_class,
    )


def _evidence(step_id: str, *path: JsonPathSegment, prefix: str = "") -> ContinuityEvidenceProjection:
    return ContinuityEvidenceProjection(step_id=step_id, path=path, prefix=prefix)


_FAILURES: tuple[ContinuityFailureClass, ...] = (
    "source_coverage",
    "discovery",
    "formulation",
    "plan",
    "execution",
    "projection",
    "reasoning",
)
_STOP = ("all declared facts match", "all continuations are exhausted", "all evidence refs are citable")


def _scenario(
    scenario_id: str,
    title: str,
    question: str,
    *,
    steps: tuple[ContinuityRouteStep, ...],
    facts: tuple[ContinuityFactProjection, ...],
    evidence: tuple[ContinuityEvidenceProjection, ...],
    workflows: tuple[str, ...],
    plans: tuple[str, ...],
    coverage: tuple[str, ...],
    result_semantics: ContinuityResultSemantics,
    mutations: tuple[str, ...],
    equivalent_plan_signatures: tuple[tuple[str, ...], ...] = (),
    attempt_curriculum: ContinuityAttemptCurriculumProjection | None = None,
    budget: ContinuityBudget = ContinuityBudget(),
) -> ContinuityScenarioSpec:
    tools = tuple(dict.fromkeys(step.tool for step in steps))
    required_arguments: dict[ContinuityTool, set[str]] = {tool: set() for tool in tools}
    for step in steps:
        required_arguments[step.tool].update(step.argument_dict())
        if step.paginate:
            required_arguments[step.tool].add("continuation")
    discovery = tuple(
        ContinuityDiscoveryRequirement(tool=tool, required_arguments=tuple(sorted(required_arguments[tool])))
        for tool in tools
    )
    route_signature = tuple(step.plan_atom for step in steps)
    accepted_signatures = tuple(dict.fromkeys((route_signature, *equivalent_plan_signatures)))
    return ContinuityScenarioSpec(
        name=scenario_id,
        description=title,
        operator_question=question,
        fixture_key=scenario_id,
        route_steps=steps,
        fact_projections=facts,
        evidence_projections=evidence,
        allowed_query_surfaces=tools,
        workflow_ids=workflows,
        canonical_plan_families=plans,
        equivalent_plan_signatures=accepted_signatures,
        discovery_requirements=discovery,
        coverage_inventory=coverage,
        result_semantics=result_semantics,
        stop_conditions=_STOP,
        failure_taxonomy=_FAILURES,
        mutation_cases=mutations,
        attempt_curriculum=attempt_curriculum,
        budget=budget,
        origin="authored",
        tags=("continuity", "black-box", "known-answer", scenario_id),
        docs_role="verification-scenario",
        caption=title,
        audience=("operators", "maintainers", "agents"),
        demonstrates=("public-route replay", "independent oracle"),
        privacy_level="synthetic",
    )


CONTINUITY_SCENARIOS: tuple[ContinuityScenarioSpec, ...] = (
    _scenario(
        "resume",
        "Resume current work",
        "What was I doing in this repo?",
        steps=(
            ContinuityRouteStep(
                "resume-messages",
                "query_units",
                _args(expression="messages where text:{fixture:corpus.resume.marker}", limit=2),
                paginate=True,
                exact_count_probe=True,
                item_identity_path=("message_id",),
            ),
        ),
        facts=(
            _fact(
                "matching_messages",
                "resume-messages",
                "items",
                "*",
                "message_id",
                reducer="count",
                failure_class="source_coverage",
            ),
            _fact("session_ids", "resume-messages", "items", "*", "session_id", reducer="unique_values"),
            _fact("continuation_text", "resume-messages", "items", "*", "text"),
        ),
        evidence=(_evidence("resume-messages", "items", "*", "message_id", prefix="message:"),),
        workflows=("find-then-successor-context", "find-then-continue"),
        plans=("message-query", "read-context"),
        coverage=("message text", "session identity", "successor work context"),
        result_semantics=ContinuityResultSemantics(
            target_population="messages matching the caller-supplied current-work marker",
            exactness="exact_population",
            evidence_contract="cite every matching message id",
        ),
        mutations=("drop-text-filter", "lost-continuation-state"),
    ),
    _scenario(
        "forensic-debug",
        "Find a forensic file and failing action",
        "Where did this failure happen?",
        steps=(
            ContinuityRouteStep(
                "forensic-actions",
                "query_units",
                _args(expression="actions where path:{fixture:corpus.forensic_debug.path} AND is_error:true", limit=2),
                paginate=True,
                exact_count_probe=True,
                item_identity_path=("tool_use_block_id",),
            ),
            ContinuityRouteStep(
                "forensic-files",
                "query_units",
                _args(expression="files where path:{fixture:corpus.forensic_debug.path}", limit=2),
                paginate=True,
                exact_count_probe=True,
                item_identity_path=("first_tool_use_block_id",),
            ),
        ),
        facts=(
            _fact(
                "failed_actions",
                "forensic-actions",
                "items",
                "*",
                "tool_use_block_id",
                reducer="count",
                failure_class="source_coverage",
            ),
            _fact("failure_path", "forensic-actions", "items", "*", "tool_path"),
            _fact("exit_code", "forensic-actions", "items", "*", "exit_code"),
            _fact("file_rows", "forensic-files", "items", "*", "path", reducer="count"),
            _fact("file_path", "forensic-files", "items", "*", "path"),
        ),
        evidence=(
            _evidence("forensic-actions", "items", "*", "message_id", prefix="message:"),
            _evidence("forensic-actions", "items", "*", "tool_use_block_id", prefix="block:"),
            _evidence("forensic-files", "items", "*", "first_tool_use_block_id", prefix="block:"),
        ),
        workflows=("find-then-read-messages", "resolve-ref-drilldown"),
        plans=("action-query", "file-query"),
        equivalent_plan_signatures=(("query_units:files", "query_units:actions"),),
        coverage=("error actions", "referenced files", "message and block evidence"),
        result_semantics=ContinuityResultSemantics(
            target_population="error actions and file references for the supplied path",
            exactness="exact_population",
            evidence_contract="cite the action message, tool-use block, and file block",
        ),
        mutations=("drop-error-filter", "global-action-materialization"),
    ),
    _scenario(
        "prior-art",
        "Retrieve prior art",
        "Have we solved this before?",
        steps=(
            ContinuityRouteStep(
                "prior-messages",
                "query_units",
                _args(expression="messages where text:{fixture:corpus.prior_art.marker}", limit=2),
                paginate=True,
                exact_count_probe=True,
                item_identity_path=("message_id",),
            ),
        ),
        facts=(
            _fact(
                "prior_hits",
                "prior-messages",
                "items",
                "*",
                "message_id",
                reducer="count",
                failure_class="source_coverage",
            ),
            _fact(
                "session_ids",
                "prior-messages",
                "items",
                "*",
                "session_id",
                reducer="unique_values",
                failure_class="source_coverage",
            ),
            _fact(
                "prior_art_text",
                "prior-messages",
                "items",
                "*",
                "text",
                failure_class="source_coverage",
            ),
        ),
        evidence=(_evidence("prior-messages", "items", "*", "message_id", prefix="message:"),),
        workflows=("find-then-read-messages", "resolve-ref-drilldown"),
        plans=("text-query", "evidence-read"),
        coverage=("message text", "session identity", "prior-art evidence"),
        result_semantics=ContinuityResultSemantics(
            target_population="messages matching the supplied prior-art marker",
            exactness="exact_population",
            evidence_contract="cite every matching message id",
        ),
        mutations=("drop-marker-filter", "wrong-ranking-scope"),
    ),
    _scenario(
        "decision",
        "Recover an operator decision",
        "What did we decide about this?",
        steps=(
            ContinuityRouteStep(
                "decision-assertions",
                "query_units",
                _args(expression="assertions where kind:decision AND text:{fixture:corpus.decision.marker}", limit=2),
                paginate=True,
                exact_count_probe=True,
                item_identity_path=("assertion_id",),
            ),
        ),
        facts=(
            _fact(
                "decision_count",
                "decision-assertions",
                "items",
                "*",
                "assertion_id",
                reducer="count",
                failure_class="source_coverage",
            ),
            _fact("decision_ids", "decision-assertions", "items", "*", "assertion_id", reducer="unique_values"),
            _fact("decision_body", "decision-assertions", "items", "*", "body_text"),
            _fact("decision_status", "decision-assertions", "items", "*", "status"),
        ),
        evidence=(
            _evidence("decision-assertions", "items", "*", "assertion_id", prefix="assertion:"),
            _evidence("decision-assertions", "items", "*", "evidence_refs", "*"),
        ),
        workflows=("candidate-assertion-review", "resolve-ref-drilldown"),
        plans=("assertion-query", "evidence-read"),
        coverage=("decision assertions", "assertion status", "assertion evidence refs"),
        result_semantics=ContinuityResultSemantics(
            target_population="decision assertions matching the supplied decision marker",
            exactness="single",
            evidence_contract="cite the assertion and all planted evidence refs",
        ),
        mutations=("drop-kind-filter", "unresolved-ref-broadening"),
    ),
    _scenario(
        "postmortem",
        "Explain a failed run",
        "Why did the last run fail?",
        steps=(
            ContinuityRouteStep(
                "failed-actions",
                "query_units",
                _args(expression="actions where output:{fixture:corpus.postmortem.marker} AND is_error:true", limit=2),
                paginate=True,
                exact_count_probe=True,
                item_identity_path=("tool_use_block_id",),
            ),
        ),
        facts=(
            _fact(
                "failed_action_count",
                "failed-actions",
                "items",
                "*",
                "tool_use_block_id",
                reducer="count",
                failure_class="source_coverage",
            ),
            _fact("failed_tool", "failed-actions", "items", "*", "tool_name"),
            _fact("exit_code", "failed-actions", "items", "*", "exit_code"),
            _fact("failure_output", "failed-actions", "items", "*", "output_text"),
        ),
        evidence=(
            _evidence("failed-actions", "items", "*", "message_id", prefix="message:"),
            _evidence("failed-actions", "items", "*", "tool_use_block_id", prefix="block:"),
        ),
        workflows=("find-then-read-messages", "resolve-ref-drilldown"),
        plans=("action-query", "failure-evidence"),
        coverage=("failed actions", "tool exit status", "failure output evidence"),
        result_semantics=ContinuityResultSemantics(
            target_population="failed actions matching the supplied postmortem marker",
            exactness="single",
            evidence_contract="cite the action message and tool-use block",
        ),
        mutations=("drop-error-filter", "capped-pseudo-total"),
    ),
    _scenario(
        "cost",
        "Audit reported usage",
        "How much did this work cost?",
        steps=(
            ContinuityRouteStep(
                "usage-report",
                "provider_usage",
                _args(origin="{fixture:corpus.origin}", limit=10, detail="headline"),
            ),
        ),
        facts=(
            _fact(
                "input_tokens",
                "usage-report",
                "metadata",
                "provider_usage",
                "model_rollup_usage",
                "input_tokens",
            ),
            _fact(
                "output_tokens",
                "usage-report",
                "metadata",
                "provider_usage",
                "model_rollup_usage",
                "output_tokens",
            ),
            _fact(
                "cached_input_tokens",
                "usage-report",
                "metadata",
                "provider_usage",
                "model_rollup_usage",
                "cached_input_tokens",
            ),
            _fact(
                "total_tokens",
                "usage-report",
                "metadata",
                "provider_usage",
                "model_rollup_usage",
                "total_tokens",
            ),
            _fact("pricing_grain", "usage-report", "metadata", "provider_usage", "pricing_grain"),
        ),
        evidence=(
            _evidence(
                "usage-report",
                "provider_usage",
                "exact_total_tokens_evidence",
                "evidence_refs",
                "*",
            ),
        ),
        workflows=("resolve-ref-drilldown",),
        plans=("usage-diagnostic", "reported-token-audit"),
        coverage=("origin-scoped provider usage", "model rollup", "reported-token evidence"),
        result_semantics=ContinuityResultSemantics(
            target_population="provider-reported usage rows for the supplied origin",
            exactness="aggregate",
            evidence_contract="retain exact-token evidence refs and pricing grain",
        ),
        mutations=("drop-origin-filter", "reported-vs-estimated-collapse"),
        budget=ContinuityBudget(
            max_calls=4,
            max_page_bytes=100_000,
            max_total_bytes=150_000,
            max_elapsed_ms=30_000,
        ),
    ),
    _scenario(
        "self-inspection",
        "Inspect query capability",
        "What can you answer about agent work?",
        steps=(
            ContinuityRouteStep(
                "query-explanation",
                "explain_query_expression",
                _args(expression="assertions where kind:decision AND status:active"),
            ),
            ContinuityRouteStep("read-views", "list_read_view_profiles"),
        ),
        facts=(
            _fact(
                "selected_units",
                "query-explanation",
                "query_explanation",
                "selected_units",
                "*",
                reducer="unique_values",
            ),
            _fact(
                "unsupported_nodes",
                "query-explanation",
                "query_explanation",
                "unsupported_nodes",
                "*",
                reducer="count",
            ),
            _fact("read_view_count", "read-views", "total"),
            _fact("read_view_ids", "read-views", "read_views", "*", "view_id", reducer="unique_values"),
        ),
        evidence=(),
        workflows=("find-then-read-messages", "find-then-context-image", "resolve-ref-drilldown"),
        plans=("grammar-introspection", "read-view-catalog"),
        equivalent_plan_signatures=(("list_read_view_profiles", "explain_query_expression"),),
        coverage=("query grammar", "selected terminal units", "read-view profile catalog"),
        result_semantics=ContinuityResultSemantics(
            target_population="the executable query grammar and shipped read-view profiles",
            exactness="capability_inventory",
            evidence_contract="return machine-readable capability metadata",
        ),
        mutations=("unpaged-capability-catalog", "unsupported-terminal-unit"),
    ),
    _scenario(
        "parallel-claude-incident",
        "Reconstruct parallel-agent incident membership",
        "Which agents handled the concerns and what changed?",
        steps=(
            ContinuityRouteStep(
                "incident-members",
                "query_units",
                _args(
                    expression=(
                        "messages where text:parallel-child "
                        'AND text:"workflow_run:{fixture:corpus.parallel_incident.run_ref}"'
                    ),
                    limit=17,
                ),
                paginate=True,
                exact_count_probe=True,
                item_identity_path=("message_id",),
            ),
            ContinuityRouteStep(
                "incident-other-members",
                "query_units",
                _args(
                    expression=(
                        "messages where text:parallel-child "
                        'AND NOT text:"workflow_run:{fixture:corpus.parallel_incident.run_ref}"'
                    ),
                    limit=19,
                ),
                paginate=True,
                exact_count_probe=True,
                item_identity_path=("message_id",),
            ),
            ContinuityRouteStep(
                "incident-all-children",
                "query_units",
                _args(
                    expression=(
                        "runs where native_parent_session_id:"
                        '"{fixture:corpus.parallel_incident.coordinator_session_id}" '
                        "AND role:subagent"
                    ),
                    limit=19,
                ),
                paginate=True,
                item_identity_path=("run_ref",),
            ),
            ContinuityRouteStep(
                "incident-invocations",
                "query_units",
                _args(
                    expression=('messages where text:"workflow-invocation:{fixture:corpus.parallel_incident.run_ref}"'),
                    limit=2,
                ),
                paginate=True,
                exact_count_probe=True,
                item_identity_path=("message_id",),
            ),
            ContinuityRouteStep(
                "incident-final",
                "query_units",
                _args(
                    expression=(
                        'messages where text:"final-structured-result:{fixture:corpus.parallel_incident.run_ref}"'
                    ),
                    limit=2,
                ),
                paginate=True,
                exact_count_probe=True,
                item_identity_path=("message_id",),
            ),
            ContinuityRouteStep(
                "incident-curriculum",
                "query_units",
                _args(
                    expression=('messages where text:"incident-curriculum:{fixture:corpus.parallel_incident.run_ref}"'),
                    limit=3,
                ),
                paginate=True,
                exact_count_probe=True,
                item_identity_path=("message_id",),
            ),
        ),
        facts=(
            _fact(
                "attempt_transcripts",
                "incident-members",
                "items",
                "*",
                "session_id",
                reducer="unique_count",
                failure_class="source_coverage",
            ),
            _fact(
                "call_keys",
                "incident-members",
                "items",
                "*",
                "text",
                reducer="regex_unique_count",
                pattern=r"call_key:(call-\d{2})",
            ),
            _fact(
                "result_records",
                "incident-members",
                "items",
                "*",
                "text",
                reducer="regex_count",
                pattern=r"result_record:yes",
            ),
            _fact(
                "completed_call_keys",
                "incident-members",
                "items",
                "*",
                "text",
                reducer="regex_unique_count",
                pattern=r"completed_key:(call-\d{2})",
            ),
            _fact(
                "unresolved_call_keys",
                "incident-members",
                "items",
                "*",
                "text",
                reducer="regex_unique_count",
                pattern=r"unresolved_key:(call-\d{2})",
            ),
            _fact(
                "non_workflow_children",
                "incident-other-members",
                "items",
                "*",
                "session_id",
                reducer="unique_count",
                failure_class="source_coverage",
            ),
            _fact(
                "coordinator_children",
                "incident-all-children",
                "items",
                "*",
                "run_ref",
                reducer="unique_count",
                failure_class="source_coverage",
            ),
            _fact(
                "workflow_invocations",
                "incident-invocations",
                "items",
                "*",
                "message_id",
                reducer="count",
                failure_class="source_coverage",
            ),
            _fact(
                "final_result_count",
                "incident-final",
                "items",
                "*",
                "message_id",
                reducer="count",
                failure_class="source_coverage",
            ),
            _fact(
                "incident_curriculum_cases",
                "incident-curriculum",
                "items",
                "*",
                "message_id",
                reducer="count",
                failure_class="source_coverage",
            ),
        ),
        evidence=(
            _evidence("incident-members", "items", "*", "message_id", prefix="message:"),
            _evidence("incident-invocations", "items", "*", "message_id", prefix="message:"),
            _evidence("incident-final", "items", "*", "message_id", prefix="message:"),
            _evidence("incident-curriculum", "items", "*", "message_id", prefix="message:"),
        ),
        workflows=("find-then-read-messages", "resolve-ref-drilldown"),
        plans=("membership-query", "continuation-census", "effect-evidence"),
        coverage=(
            "coordinator-child topology",
            "target workflow membership",
            "call/result completion census",
            "workflow invocations and final result",
            "original-call classification curriculum",
        ),
        result_semantics=ContinuityResultSemantics(
            target_population="all 129 coordinator children partitioned into 91 target-run and 38 other children",
            exactness="exact_population",
            evidence_contract="enumerate every row exactly once and cite target-run, invocation, final, and curriculum messages",
        ),
        mutations=(
            "drop-workflow-filter",
            "lost-request-state-continuation",
            "capped-pseudo-total",
            "identical-call-topology-replay",
            "hidden-fact-or-grammar-discovery",
            "missing-source-coverage",
            "unreasonable-query-classification",
        ),
        attempt_curriculum=ContinuityAttemptCurriculumProjection(
            step_id="incident-curriculum",
            path=("items", "*", "text"),
            case_ids=(
                "candidate-list",
                "exact-operator-phrase",
                "sonnet-lexical-proxy",
                "sessions-only-query",
                "correct-topology",
                "correct-delegation",
            ),
        ),
        budget=ContinuityBudget(
            max_calls=28,
            max_page_bytes=25_000,
            max_total_bytes=650_000,
            max_elapsed_ms=45_000,
        ),
    ),
)

CONTINUITY_SCENARIO_BY_ID = {scenario.scenario_id: scenario for scenario in CONTINUITY_SCENARIOS}


def continuity_scenario(name: str) -> ContinuityScenarioSpec:
    try:
        return CONTINUITY_SCENARIO_BY_ID[name]
    except KeyError as exc:
        raise KeyError(f"unknown continuity scenario {name!r}") from exc


__all__ = [
    "CONTINUITY_SCENARIOS",
    "CONTINUITY_SCENARIO_BY_ID",
    "ContinuityAttemptCurriculumProjection",
    "ContinuityAttemptGrade",
    "ContinuityBudget",
    "ContinuityDiscoveryRequirement",
    "ContinuityEvidenceProjection",
    "ContinuityFactProjection",
    "ContinuityFailureClass",
    "ContinuityResultSemantics",
    "ContinuityRouteStep",
    "ContinuityScenarioSpec",
    "continuity_scenario",
]
