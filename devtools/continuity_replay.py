"""Execute continuity scenarios through MCP and independent known answers.

The same parameterized declarations can run against the repository's synthetic
fixture or an authorized live archive.  The default route is a real MCP stdio
client/server exchange; a registered-handler route remains available for fast,
focused mutation tests.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import inspect
import json
import os
import re
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, TypeAlias, TypeGuard, cast

if __package__ in {None, ""}:  # pragma: no cover - exercised by the script entry point
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from polylogue.core.json import JSONDocument, JSONValue, require_json_document, require_json_value
from polylogue.product.continuity_scenarios import (
    CONTINUITY_SCENARIOS,
    ContinuityAttemptCurriculumProjection,
    ContinuityAttemptGrade,
    ContinuityBudget,
    ContinuityDiscoveryRequirement,
    ContinuityEvidenceProjection,
    ContinuityFactProjection,
    ContinuityFailureClass,
    ContinuityRouteStep,
    ContinuityScenarioSpec,
    continuity_scenario,
)

if TYPE_CHECKING:
    from mcp import ClientSession

RouteArguments: TypeAlias = dict[str, object]
RouteTransport: TypeAlias = Literal["stdio", "registered"]
ArgumentMutator: TypeAlias = Callable[[str, RouteArguments, int], RouteArguments]
ResponseMutator: TypeAlias = Callable[[str, RouteArguments, int, str], str]
DiscoveryMutator: TypeAlias = Callable[[JSONDocument], JSONDocument]
_TEMPLATE_RE = re.compile(r"\{fixture:([A-Za-z0-9_.-]+)\}")
_ATTEMPT_TOKEN_RE = re.compile(r"(?P<key>[a-z_]+):(?P<value>[A-Za-z0-9_-]+)")
_ATTEMPT_GRADES: frozenset[str] = frozenset(
    {
        "reasonable_oversized",
        "wrong_corpus_assumption",
        "weak_lexical_proxy",
        "product_induced_hidden_grammar",
        "execution_failure",
        "unreasonable_query",
    }
)


class ContinuityRoute(Protocol):
    """MCP-shaped route with an inspectable discovery receipt."""

    @property
    def transport_name(self) -> str:
        raise NotImplementedError

    @property
    def discovery(self) -> JSONDocument:
        raise NotImplementedError

    async def invoke(self, tool: str, arguments: Mapping[str, object]) -> str:
        raise NotImplementedError


class ContinuityReplayError(RuntimeError):
    """Structured replay failure that can be reported without a traceback."""

    def __init__(self, message: str, *, kind: str, failure_class: ContinuityFailureClass) -> None:
        super().__init__(message)
        self.kind = kind
        self.failure_class = failure_class


@dataclass(frozen=True, slots=True)
class _ReturnedUnitIdentity:
    """Stable identity extracted from one actual paginated unit row."""

    key: str
    display: str


class MCPContinuityRoute:
    """Invoke handlers registered on Polylogue's real FastMCP read server."""

    def __init__(
        self,
        archive_root: Path,
        *,
        argument_mutator: ArgumentMutator | None = None,
        response_mutator: ResponseMutator | None = None,
        discovery_mutator: DiscoveryMutator | None = None,
    ) -> None:
        self.archive_root = archive_root.resolve()
        self.argument_mutator = argument_mutator
        self.response_mutator = response_mutator
        self.discovery_mutator = discovery_mutator
        self._services: object | None = None
        self._server: object | None = None
        self._discovery: JSONDocument = {}
        self._invocation_count = 0

    @property
    def transport_name(self) -> str:
        return "registered-fastmcp-tool"

    @property
    def discovery(self) -> JSONDocument:
        return dict(self._discovery)

    async def __aenter__(self) -> MCPContinuityRoute:
        from polylogue.config import Config
        from polylogue.mcp.server import build_server
        from polylogue.mcp.server_support import _set_runtime_services
        from polylogue.services import RuntimeServices

        config = Config(
            archive_root=self.archive_root,
            render_root=(self.archive_root / "render").resolve(),
            sources=[],
            db_path=(self.archive_root / "index.db").resolve(),
        )
        services = RuntimeServices(config=config, db_path=config.db_path)
        _set_runtime_services(services)
        self._services = services
        self._server = build_server(role="read")
        self._discovery = _registered_discovery(self._server, self.transport_name)
        if self.discovery_mutator is not None:
            self._discovery = require_json_document(
                self.discovery_mutator(dict(self._discovery)),
                context="mutated registered MCP discovery",
            )
        return self

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
        from polylogue.mcp.server_support import _set_runtime_services

        services = self._services
        self._server = None
        self._services = None
        try:
            if services is not None:
                close = getattr(services, "close", None)
                if close is not None:
                    result = close()
                    if inspect.isawaitable(result):
                        await result
        finally:
            _set_runtime_services(None)

    async def invoke(self, tool: str, arguments: Mapping[str, object]) -> str:
        server = self._server
        if server is None:
            raise ContinuityReplayError(
                "MCP route is not open",
                kind="route_not_open",
                failure_class="execution",
            )
        manager = getattr(server, "_tool_manager", None)
        tools = getattr(manager, "_tools", None)
        if not isinstance(tools, Mapping) or tool not in tools:
            raise ContinuityReplayError(
                f"registered MCP tool is unavailable: {tool}",
                kind="missing_public_tool",
                failure_class="execution",
            )
        invocation, call_arguments = self._prepare_call(tool, arguments)
        registered = tools[tool]
        fn = getattr(registered, "fn", None)
        if fn is None:
            raise ContinuityReplayError(
                f"registered MCP tool has no callable: {tool}",
                kind="missing_public_handler",
                failure_class="execution",
            )
        result = fn(**call_arguments)
        if inspect.isawaitable(result):
            result = await result
        if not isinstance(result, str):
            raise ContinuityReplayError(
                f"MCP tool {tool} returned {type(result).__name__}, expected JSON text",
                kind="invalid_route_response",
                failure_class="execution",
            )
        return self._mutate_response(tool, call_arguments, invocation, result)

    def _prepare_call(self, tool: str, arguments: Mapping[str, object]) -> tuple[int, RouteArguments]:
        self._invocation_count += 1
        invocation = self._invocation_count
        call_arguments = dict(arguments)
        if self.argument_mutator is not None:
            call_arguments = self.argument_mutator(tool, call_arguments, invocation)
        return invocation, call_arguments

    def _mutate_response(
        self,
        tool: str,
        arguments: RouteArguments,
        invocation: int,
        response_text: str,
    ) -> str:
        if self.response_mutator is None:
            return response_text
        return self.response_mutator(tool, arguments, invocation, response_text)


class StdioMCPContinuityRoute:
    """Invoke the installed Polylogue server through official MCP stdio JSON-RPC."""

    def __init__(
        self,
        archive_root: Path,
        *,
        argument_mutator: ArgumentMutator | None = None,
        response_mutator: ResponseMutator | None = None,
        discovery_mutator: DiscoveryMutator | None = None,
        read_timeout_seconds: float = 60.0,
    ) -> None:
        self.archive_root = archive_root.resolve()
        self.argument_mutator = argument_mutator
        self.response_mutator = response_mutator
        self.discovery_mutator = discovery_mutator
        self.read_timeout_seconds = read_timeout_seconds
        self._stack: AsyncExitStack | None = None
        self._session: ClientSession | None = None
        self._discovery: JSONDocument = {}
        self._invocation_count = 0

    @property
    def transport_name(self) -> str:
        return "mcp-stdio-json-rpc"

    @property
    def discovery(self) -> JSONDocument:
        return dict(self._discovery)

    async def __aenter__(self) -> StdioMCPContinuityRoute:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        runtime_root = self.archive_root / ".continuity-runtime"
        environment = dict(os.environ)
        environment.update(
            {
                "POLYLOGUE_ARCHIVE_ROOT": str(self.archive_root),
                "XDG_CONFIG_HOME": str(runtime_root / "config"),
                "XDG_STATE_HOME": str(runtime_root / "state"),
                "XDG_CACHE_HOME": str(runtime_root / "cache"),
            }
        )
        parameters = StdioServerParameters(
            command=sys.executable,
            args=["-c", "from polylogue.mcp.cli import main; main()", "--role", "read"],
            env=environment,
            cwd=str(Path.cwd()),
        )
        stack = AsyncExitStack()
        try:
            read_stream, write_stream = await stack.enter_async_context(stdio_client(parameters))
            session = await stack.enter_async_context(
                ClientSession(
                    read_stream,
                    write_stream,
                    read_timeout_seconds=timedelta(seconds=self.read_timeout_seconds),
                )
            )
            initialize = await session.initialize()
            tools: list[object] = []
            cursor: str | None = None
            while True:
                page = await session.list_tools(cursor=cursor)
                tools.extend(page.tools)
                cursor = page.nextCursor
                if cursor is None:
                    break
        except Exception as exc:
            await stack.aclose()
            raise ContinuityReplayError(
                f"failed to initialize MCP stdio route: {exc}",
                kind="stdio_initialization_failed",
                failure_class="execution",
            ) from exc

        self._stack = stack
        self._session = session
        self._discovery = _stdio_discovery(initialize, tools, self.transport_name)
        if self.discovery_mutator is not None:
            self._discovery = require_json_document(
                self.discovery_mutator(dict(self._discovery)),
                context="mutated stdio MCP discovery",
            )
        return self

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
        stack = self._stack
        self._stack = None
        self._session = None
        if stack is not None:
            await stack.aclose()

    async def invoke(self, tool: str, arguments: Mapping[str, object]) -> str:
        session = self._session
        if session is None:
            raise ContinuityReplayError(
                "MCP stdio route is not open",
                kind="route_not_open",
                failure_class="execution",
            )
        self._invocation_count += 1
        invocation = self._invocation_count
        call_arguments = dict(arguments)
        if self.argument_mutator is not None:
            call_arguments = self.argument_mutator(tool, call_arguments, invocation)
        try:
            result = await session.call_tool(tool, arguments=call_arguments)
        except Exception as exc:
            raise ContinuityReplayError(
                f"MCP stdio call {tool!r} failed: {exc}",
                kind="stdio_call_failed",
                failure_class="execution",
            ) from exc
        response_text = _call_tool_response_text(tool, result)
        if self.response_mutator is not None:
            response_text = self.response_mutator(tool, call_arguments, invocation, response_text)
        return response_text


@dataclass(slots=True)
class _BudgetState:
    budget: ContinuityBudget
    calls: int = 0
    response_bytes: int = 0
    call_elapsed_ms: float = 0.0
    started_ns: int = field(default_factory=time.perf_counter_ns)

    @property
    def elapsed_ms(self) -> float:
        return (time.perf_counter_ns() - self.started_ns) / 1_000_000

    def before_call(self) -> None:
        if self.calls >= self.budget.max_calls:
            raise ContinuityReplayError(
                f"call budget exhausted at {self.calls}/{self.budget.max_calls}",
                kind="max_calls_exceeded",
                failure_class="execution",
            )
        self._check_elapsed()

    def observe(self, response_text: str, *, call_elapsed_ms: float) -> int:
        response_bytes = len(response_text.encode("utf-8"))
        self.calls += 1
        self.response_bytes += response_bytes
        self.call_elapsed_ms += call_elapsed_ms
        if response_bytes > self.budget.max_page_bytes:
            raise ContinuityReplayError(
                f"response page is {response_bytes} bytes; budget is {self.budget.max_page_bytes}",
                kind="max_page_bytes_exceeded",
                failure_class="execution",
            )
        if self.response_bytes > self.budget.max_total_bytes:
            raise ContinuityReplayError(
                f"responses total {self.response_bytes} bytes; budget is {self.budget.max_total_bytes}",
                kind="max_total_bytes_exceeded",
                failure_class="execution",
            )
        self._check_elapsed()
        return response_bytes

    def _check_elapsed(self) -> None:
        if self.elapsed_ms > self.budget.max_elapsed_ms:
            raise ContinuityReplayError(
                f"scenario elapsed time is {self.elapsed_ms:.1f} ms; budget is {self.budget.max_elapsed_ms} ms",
                kind="max_elapsed_ms_exceeded",
                failure_class="execution",
            )


@dataclass(slots=True)
class _StepReceipt:
    step_id: str
    tool: str
    arguments: JSONDocument
    pages: list[JSONDocument] = field(default_factory=list)
    enumerated_item_count: int | None = None
    unique_identity_count: int | None = None
    page_total_sum: int = 0
    page_totals_match_items: bool = True
    count_probe: JSONDocument | None = None
    population_count_verified: bool | None = None
    exact_enumeration_verified: bool = False
    query_ref: str | None = None
    result_ref: str | None = None

    def to_payload(self) -> JSONDocument:
        return require_json_document(
            {
                "step_id": self.step_id,
                "tool": self.tool,
                "arguments": dict(self.arguments),
                "page_count": len(self.pages),
                "pages": list(self.pages),
                "enumerated_item_count": self.enumerated_item_count,
                "unique_identity_count": self.unique_identity_count,
                "page_total_sum": self.page_total_sum,
                "page_totals_match_items": self.page_totals_match_items,
                "count_probe": self.count_probe,
                "population_count_verified": self.population_count_verified,
                "exact_enumeration_verified": self.exact_enumeration_verified,
                "query_ref": self.query_ref,
                "result_ref": self.result_ref,
            },
            context=f"continuity step receipt {self.step_id}",
        )


def load_oracle_catalog(path: Path) -> JSONDocument:
    """Load a parameterized corpus/oracle manifest."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    catalog = require_json_document(payload, context=f"continuity oracle catalog {path}")
    if catalog.get("schema_version") != 2:
        raise ValueError("continuity oracle catalog schema_version must be 2")
    if not isinstance(catalog.get("corpus"), Mapping):
        raise ValueError("continuity oracle catalog requires corpus object")
    if not isinstance(catalog.get("oracles"), Mapping):
        raise ValueError("continuity oracle catalog requires oracles object")
    return dict(catalog)


def materialize_route_arguments(
    step: ContinuityRouteStep,
    fixture: Mapping[str, JSONValue],
) -> RouteArguments:
    """Resolve declaration references against the caller-supplied corpus."""

    return {key: _materialize(value, fixture) for key, value in step.argument_dict().items()}


def project_fact(
    projection: ContinuityFactProjection,
    observations: Mapping[str, JSONValue],
) -> JSONValue:
    """Project one comparable fact from merged real-route payloads."""

    if projection.step_id not in observations:
        raise ContinuityReplayError(
            f"projection {projection.name!r} references missing step {projection.step_id!r}",
            kind="missing_step_observation",
            failure_class=projection.failure_class,
        )
    values = _values_at(observations[projection.step_id], projection.path)
    reducer = projection.reducer
    if reducer == "count":
        return len(values)
    if reducer == "single":
        if len(values) != 1:
            raise ContinuityReplayError(
                f"fact {projection.name!r} expected one value at {projection.path!r}, observed {len(values)}",
                kind="non_single_projection",
                failure_class=projection.failure_class,
            )
        return require_json_value(values[0], context=f"continuity fact {projection.name}")
    if reducer in {"unique_values", "unique_count"}:
        unique = _unique_json_scalars(values, fact_name=projection.name, failure_class=projection.failure_class)
        return len(unique) if reducer == "unique_count" else unique
    if reducer in {"regex_count", "regex_unique_count"}:
        if projection.pattern is None:
            raise ContinuityReplayError(
                f"regex projection {projection.name!r} has no pattern",
                kind="missing_regex_pattern",
                failure_class=projection.failure_class,
            )
        regex = re.compile(projection.pattern)
        matches: list[str] = []
        for value in values:
            if not isinstance(value, str):
                raise ContinuityReplayError(
                    f"regex projection {projection.name!r} received {type(value).__name__}",
                    kind="non_text_regex_input",
                    failure_class=projection.failure_class,
                )
            for match in regex.finditer(value):
                matches.append(match.group(1) if match.lastindex else match.group(0))
        return len(matches) if reducer == "regex_count" else len(set(matches))
    if reducer == "sum":
        total: int | float = 0
        for value in values:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ContinuityReplayError(
                    f"sum projection {projection.name!r} received {type(value).__name__}",
                    kind="non_numeric_sum_input",
                    failure_class=projection.failure_class,
                )
            total += value
        return total
    raise ContinuityReplayError(
        f"unsupported continuity reducer: {reducer}",
        kind="unsupported_reducer",
        failure_class=projection.failure_class,
    )


def compare_observed_facts(
    *,
    expected: Mapping[str, JSONValue],
    observed: Mapping[str, JSONValue],
    source_refs: Sequence[str] = (),
    failure_classes: Mapping[str, ContinuityFailureClass] | None = None,
) -> list[JSONDocument]:
    """Return exact, fact-addressed diagnostics for independent oracle mismatches."""

    classes = failure_classes or {}
    diagnostics: list[JSONDocument] = []
    names = sorted(set(expected) | set(observed))
    for name in names:
        failure_class = classes.get(name, "projection")
        if name not in expected:
            diagnostics.append(
                {
                    "kind": "unexpected_fact",
                    "failure_class": failure_class,
                    "fact": name,
                    "observed": observed[name],
                    "source_refs": list(source_refs),
                }
            )
            continue
        if name not in observed:
            diagnostics.append(
                {
                    "kind": "missing_fact",
                    "failure_class": failure_class,
                    "fact": name,
                    "expected": expected[name],
                    "source_refs": list(source_refs),
                }
            )
            continue
        if observed[name] != expected[name]:
            diagnostics.append(
                {
                    "kind": "fact_mismatch",
                    "failure_class": failure_class,
                    "fact": name,
                    "expected": expected[name],
                    "observed": observed[name],
                    "source_refs": list(source_refs),
                }
            )
    return diagnostics


def grade_incident_attempt(features: Mapping[str, str]) -> ContinuityAttemptGrade:
    """Classify one sanitized historical incident call from independently planted features."""

    shape = features.get("query_shape")
    if shape == "candidate-list" and features.get("physical_size") == "oversized":
        return "reasonable_oversized"
    if shape == "exact-operator-phrase" and features.get("corpus_match") == "false":
        return "wrong_corpus_assumption"
    if shape == "sonnet-lexical-proxy" and features.get("structure_discovery") == "absent":
        return "weak_lexical_proxy"
    if shape == "sessions-only-query" and features.get("shipped_instruction") == "advertised":
        return "product_induced_hidden_grammar"
    if shape in {"correct-topology", "correct-delegation"} and features.get("outcome") != "success":
        return "execution_failure"
    return "unreasonable_query"


def compare_attempt_grades(
    *,
    expected: Mapping[str, str],
    observed: Mapping[str, str],
    source_refs: Sequence[str] = (),
) -> list[JSONDocument]:
    """Return case-addressed reasoning diagnostics for incident curriculum drift."""

    diagnostics: list[JSONDocument] = []
    for case_id in sorted(set(expected) | set(observed)):
        if case_id not in expected:
            diagnostics.append(
                {
                    "kind": "unexpected_attempt_case",
                    "failure_class": "reasoning",
                    "case": case_id,
                    "observed": observed[case_id],
                    "source_refs": list(source_refs),
                }
            )
        elif case_id not in observed:
            diagnostics.append(
                {
                    "kind": "missing_attempt_case",
                    "failure_class": "source_coverage",
                    "case": case_id,
                    "expected": expected[case_id],
                    "source_refs": list(source_refs),
                }
            )
        elif observed[case_id] != expected[case_id]:
            diagnostics.append(
                {
                    "kind": "attempt_grade_mismatch",
                    "failure_class": "reasoning",
                    "case": case_id,
                    "expected": expected[case_id],
                    "observed": observed[case_id],
                    "source_refs": list(source_refs),
                }
            )
    return diagnostics


async def execute_continuity_scenario(
    scenario_name: str,
    fixture: Mapping[str, JSONValue],
    route: ContinuityRoute,
) -> JSONDocument:
    """Run one declaration through a public route and compare planted facts."""

    scenario = continuity_scenario(scenario_name)
    oracle = _oracle_for(scenario, fixture)
    expected_facts = _json_mapping(oracle, "facts")
    required_evidence = _string_sequence(oracle.get("required_evidence_refs", []), "required_evidence_refs")
    source_refs = _string_sequence(oracle.get("source_refs", []), "source_refs")
    expected_attempt_grades = _expected_attempt_grades(scenario, oracle)
    observations: dict[str, JSONValue] = {}
    receipts: list[_StepReceipt] = []
    budget_state = _BudgetState(scenario.budget)
    diagnostics: list[JSONDocument] = []
    observed_facts: dict[str, JSONValue] = {}
    observed_evidence: list[str] = []
    observed_attempt_grades: dict[str, str] = {}

    try:
        _validate_scenario_declaration(scenario)
        _validate_discovery(scenario.discovery_requirements, route.discovery)
        for step in scenario.route_steps:
            observations[step.step_id] = await _execute_step(
                scenario,
                step,
                fixture,
                route,
                budget_state,
                receipts,
            )
        for projection in scenario.fact_projections:
            observed_facts[projection.name] = project_fact(projection, observations)
        diagnostics.extend(
            compare_observed_facts(
                expected=expected_facts,
                observed=observed_facts,
                source_refs=source_refs,
                failure_classes={projection.name: projection.failure_class for projection in scenario.fact_projections},
            )
        )
        observed_evidence = _project_evidence(scenario.evidence_projections, observations)
        missing_evidence = sorted(set(required_evidence) - set(observed_evidence))
        if missing_evidence:
            diagnostics.append(
                require_json_document(
                    {
                        "kind": "missing_evidence",
                        "failure_class": "projection",
                        "expected": missing_evidence,
                        "observed": observed_evidence,
                        "source_refs": source_refs,
                    },
                    context=f"continuity diagnostic {scenario.scenario_id}",
                )
            )
        if scenario.attempt_curriculum is not None:
            observed_attempt_grades = _project_attempt_grades(scenario.attempt_curriculum, observations)
            diagnostics.extend(
                compare_attempt_grades(
                    expected=expected_attempt_grades,
                    observed=observed_attempt_grades,
                    source_refs=source_refs,
                )
            )
    except ContinuityReplayError as exc:
        diagnostics.append(
            require_json_document(
                {
                    "kind": exc.kind,
                    "failure_class": exc.failure_class,
                    "message": str(exc),
                    "source_refs": source_refs,
                },
                context=f"continuity execution diagnostic {scenario.scenario_id}",
            )
        )

    passed = not diagnostics
    classification = "pass" if passed else str(diagnostics[0].get("failure_class", "execution"))
    report: dict[str, object] = {
        "schema_version": 2,
        "scenario": scenario.scenario_id,
        "question": scenario.operator_question,
        "fixture_key": scenario.fixture_key,
        "route": route.transport_name,
        "discovery_receipt": _discovery_receipt(scenario, route.discovery),
        "allowed_query_surfaces": list(scenario.allowed_query_surfaces),
        "plan_signature": list(scenario.route_plan_signature),
        "accepted_plan_signatures": [list(signature) for signature in scenario.equivalent_plan_signatures],
        "coverage_inventory": list(scenario.coverage_inventory),
        "result_semantics": scenario.result_semantics.to_payload(),
        "status": "pass" if passed else "fail",
        "classification": classification,
        "expected_facts": expected_facts,
        "observed_facts": observed_facts,
        "expected_attempt_grades": expected_attempt_grades,
        "observed_attempt_grades": observed_attempt_grades,
        "required_evidence_refs": required_evidence,
        "observed_evidence_refs": observed_evidence,
        "oracle_source_refs": source_refs,
        "diagnostics": diagnostics,
        "route_receipts": [receipt.to_payload() for receipt in receipts],
        "budget": {
            "max_calls": scenario.budget.max_calls,
            "max_page_bytes": scenario.budget.max_page_bytes,
            "max_total_bytes": scenario.budget.max_total_bytes,
            "max_elapsed_ms": scenario.budget.max_elapsed_ms,
            "max_cancel_grace_ms": scenario.budget.max_cancel_grace_ms,
            "observed_calls": budget_state.calls,
            "observed_response_bytes": budget_state.response_bytes,
            "observed_call_elapsed_ms": round(budget_state.call_elapsed_ms, 3),
            "observed_scenario_elapsed_ms": round(budget_state.elapsed_ms, 3),
            "cancellation_exercised": False,
        },
    }
    return require_json_document(report, context=f"continuity report {scenario.scenario_id}")


async def replay_archive(
    archive_root: Path,
    fixture: Mapping[str, JSONValue],
    *,
    scenario_names: Sequence[str] | None = None,
    transport: RouteTransport = "stdio",
    argument_mutator: ArgumentMutator | None = None,
    response_mutator: ResponseMutator | None = None,
    discovery_mutator: DiscoveryMutator | None = None,
) -> JSONDocument:
    """Run selected declarations against an arbitrary archive/corpus pair."""

    selected = tuple(scenario_names or (scenario.scenario_id for scenario in CONTINUITY_SCENARIOS))
    route: MCPContinuityRoute | StdioMCPContinuityRoute
    if transport == "stdio":
        route = StdioMCPContinuityRoute(
            archive_root,
            argument_mutator=argument_mutator,
            response_mutator=response_mutator,
            discovery_mutator=discovery_mutator,
        )
    elif transport == "registered":
        route = MCPContinuityRoute(
            archive_root,
            argument_mutator=argument_mutator,
            response_mutator=response_mutator,
            discovery_mutator=discovery_mutator,
        )
    else:
        raise ValueError(f"unsupported continuity transport: {transport}")

    started_ns = time.perf_counter_ns()
    async with route:
        results = [await execute_continuity_scenario(name, fixture, route) for name in selected]
        route_receipt = _route_discovery_summary(route.discovery)
    passed = sum(1 for result in results if result["status"] == "pass")
    report: dict[str, object] = {
        "schema_version": 2,
        "fixture_id": fixture.get("fixture_id"),
        "archive_root": str(archive_root.resolve()),
        "transport": route.transport_name,
        "discovery_receipt": route_receipt,
        "scenario_count": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "elapsed_ms": round((time.perf_counter_ns() - started_ns) / 1_000_000, 3),
        "status": "pass" if passed == len(results) else "fail",
        "results": results,
    }
    return require_json_document(report, context="continuity archive report")


async def _execute_step(
    scenario: ContinuityScenarioSpec,
    step: ContinuityRouteStep,
    fixture: Mapping[str, JSONValue],
    route: ContinuityRoute,
    budget: _BudgetState,
    receipts: list[_StepReceipt],
) -> JSONValue:
    if step.tool not in scenario.allowed_query_surfaces:
        raise ContinuityReplayError(
            f"step {step.step_id!r} uses undeclared surface {step.tool!r}",
            kind="disallowed_query_surface",
            failure_class="plan",
        )
    arguments = materialize_route_arguments(step, fixture)
    receipt = _StepReceipt(
        step_id=step.step_id,
        tool=step.tool,
        arguments=require_json_document(arguments, context=f"route arguments {step.step_id}"),
    )
    receipts.append(receipt)
    expected_population: int | None = None
    if step.exact_count_probe:
        expected_population, receipt.count_probe = await _execute_exact_count_probe(
            step,
            arguments,
            route,
            budget,
        )
    first_payload: JSONDocument | None = None
    last_payload: JSONDocument | None = None
    merged_items: list[JSONValue] = []
    continuation: str | None = None
    seen_continuations: set[str] = set()
    seen_identities: dict[str, _ReturnedUnitIdentity] = {}
    expected_offset = 0
    page_index = 0

    while True:
        call_arguments = dict(arguments) if page_index == 0 else _continuation_arguments(arguments, continuation)
        budget.before_call()
        call_started_ns = time.perf_counter_ns()
        response_text = await route.invoke(step.tool, call_arguments)
        call_elapsed_ms = (time.perf_counter_ns() - call_started_ns) / 1_000_000
        response_bytes = budget.observe(response_text, call_elapsed_ms=call_elapsed_ms)
        payload = _decode_route_payload(step, response_text)
        page_items = payload.get("items")
        identity_hash = None
        if step.paginate:
            if not isinstance(page_items, list):
                raise ContinuityReplayError(
                    f"paginated step {step.step_id!r} returned no items list",
                    kind="invalid_paginated_payload",
                    failure_class="execution",
                )
            page_total = _validate_page_envelope(
                step,
                payload,
                expected_offset=expected_offset,
                page_item_count=len(page_items),
                query_ref=receipt.query_ref,
                result_ref=receipt.result_ref,
            )
            receipt.page_total_sum += page_total
            receipt.page_totals_match_items = receipt.page_totals_match_items and page_total == len(page_items)
            current_query_ref = cast(str, payload["query_ref"])
            current_result_ref = cast(str, payload["result_ref"])
            if receipt.query_ref is None:
                receipt.query_ref = current_query_ref
                receipt.result_ref = current_result_ref
            page_identities = _returned_unit_identities(step, page_items)
            duplicates = [identity for identity in page_identities if identity.key in seen_identities]
            if duplicates:
                duplicate_values = ", ".join(identity.display for identity in duplicates)
                raise ContinuityReplayError(
                    f"step {step.step_id!r} page {page_index + 1} replayed {len(duplicates)} "
                    f"logical unit identit{'y' if len(duplicates) == 1 else 'ies'} from an earlier page: "
                    f"{duplicate_values}",
                    kind="duplicate_pagination_identity",
                    failure_class="execution",
                )
            seen_identities.update({identity.key: identity for identity in page_identities})
            identity_hash = _hash_strings(sorted(identity.key for identity in page_identities))
            merged_items.extend(require_json_value(item, context=f"route item {step.step_id}") for item in page_items)
        receipt.pages.append(
            require_json_document(
                {
                    "page": page_index + 1,
                    "response_bytes": response_bytes,
                    "response_sha256": hashlib.sha256(response_text.encode("utf-8")).hexdigest(),
                    "call_arguments_sha256": _hash_json(call_arguments),
                    "elapsed_ms": round(call_elapsed_ms, 3),
                    "status": payload.get("status") if isinstance(payload.get("status"), str) else None,
                    "ok": payload.get("ok") if isinstance(payload.get("ok"), bool) else None,
                    "budget_exceeded": payload.get("budget_exceeded") is True,
                    "item_count": len(page_items) if isinstance(page_items, list) else None,
                    "page_total": payload.get("total") if _is_int(payload.get("total")) else None,
                    "identity_sha256": identity_hash,
                    "offset": payload.get("offset") if _is_int(payload.get("offset")) else None,
                    "next_offset": payload.get("next_offset") if _is_int(payload.get("next_offset")) else None,
                    "query_ref": payload.get("query_ref") if isinstance(payload.get("query_ref"), str) else None,
                    "result_ref": payload.get("result_ref") if isinstance(payload.get("result_ref"), str) else None,
                    "has_continuation": payload.get("continuation") is not None,
                },
                context=f"route page receipt {step.step_id}",
            )
        )
        if first_payload is None:
            first_payload = payload
        last_payload = payload
        if not step.paginate:
            return payload

        expected_offset += len(cast(list[JSONValue], page_items))
        raw_continuation = payload.get("continuation")
        if raw_continuation is None:
            break
        if not isinstance(raw_continuation, str) or not raw_continuation:
            raise ContinuityReplayError(
                f"step {step.step_id!r} returned an invalid continuation",
                kind="invalid_continuation",
                failure_class="execution",
            )
        if raw_continuation in seen_continuations:
            raise ContinuityReplayError(
                f"step {step.step_id!r} returned a non-progressing continuation",
                kind="non_progressing_continuation",
                failure_class="execution",
            )
        if not cast(list[JSONValue], page_items):
            raise ContinuityReplayError(
                f"step {step.step_id!r} returned an empty page with a continuation",
                kind="empty_progress_page",
                failure_class="execution",
            )
        seen_continuations.add(raw_continuation)
        continuation = raw_continuation
        page_index += 1

    receipt.enumerated_item_count = len(merged_items)
    receipt.unique_identity_count = len(seen_identities)
    receipt.population_count_verified = (
        None if expected_population is None else expected_population == len(merged_items)
    )
    if receipt.population_count_verified is False:
        raise ContinuityReplayError(
            f"step {step.step_id!r} enumerated {len(merged_items)} rows but the independent count probe "
            f"selected {expected_population}",
            kind="pagination_count_mismatch",
            failure_class="execution",
        )
    receipt.exact_enumeration_verified = (
        len(merged_items) == len(seen_identities)
        and receipt.page_totals_match_items
        and (expected_population is None or receipt.population_count_verified is True)
    )
    merged = dict(first_payload)
    merged["items"] = merged_items
    merged["enumerated_total"] = len(merged_items)
    merged["terminal_page_total"] = last_payload.get("total")
    merged["continuation"] = None
    merged["next_offset"] = None
    merged["page_count"] = len(receipt.pages)
    merged["exact_enumeration_verified"] = receipt.exact_enumeration_verified
    return require_json_document(merged, context=f"merged route payload {step.step_id}")


async def _execute_exact_count_probe(
    step: ContinuityRouteStep,
    arguments: Mapping[str, object],
    route: ContinuityRoute,
    budget: _BudgetState,
) -> tuple[int, JSONDocument]:
    expression = arguments.get("expression")
    if not isinstance(expression, str):
        raise ContinuityReplayError(
            f"exact-count step {step.step_id!r} has no expression",
            kind="missing_count_probe_expression",
            failure_class="plan",
        )
    count_arguments: RouteArguments = {
        key: value for key, value in arguments.items() if key not in {"continuation", "limit", "offset"}
    }
    count_arguments["expression"] = f"{expression} | count"
    count_arguments["limit"] = 1
    budget.before_call()
    call_started_ns = time.perf_counter_ns()
    response_text = await route.invoke(step.tool, count_arguments)
    call_elapsed_ms = (time.perf_counter_ns() - call_started_ns) / 1_000_000
    response_bytes = budget.observe(response_text, call_elapsed_ms=call_elapsed_ms)
    payload = _decode_route_payload(step, response_text)
    items = payload.get("items")
    if not isinstance(items, list) or len(items) != 1 or not isinstance(items[0], Mapping):
        raise ContinuityReplayError(
            f"exact-count probe for step {step.step_id!r} returned no single aggregate row",
            kind="invalid_exact_count_probe",
            failure_class="execution",
        )
    count = items[0].get("count")
    if not _is_int(count):
        raise ContinuityReplayError(
            f"exact-count probe for step {step.step_id!r} returned invalid count {count!r}",
            kind="invalid_exact_count_probe",
            failure_class="execution",
        )
    if payload.get("continuation") is not None or payload.get("next_offset") is not None:
        raise ContinuityReplayError(
            f"exact-count probe for step {step.step_id!r} unexpectedly paginated",
            kind="paginated_exact_count_probe",
            failure_class="execution",
        )
    receipt = require_json_document(
        {
            "expression": count_arguments["expression"],
            "selected_rows_exact": count,
            "response_bytes": response_bytes,
            "response_sha256": hashlib.sha256(response_text.encode("utf-8")).hexdigest(),
            "call_arguments_sha256": _hash_json(count_arguments),
            "elapsed_ms": round(call_elapsed_ms, 3),
            "query_ref": payload.get("query_ref") if isinstance(payload.get("query_ref"), str) else None,
            "result_ref": payload.get("result_ref") if isinstance(payload.get("result_ref"), str) else None,
        },
        context=f"exact-count receipt {step.step_id}",
    )
    return count, receipt


def _validate_page_envelope(
    step: ContinuityRouteStep,
    payload: Mapping[str, JSONValue],
    *,
    expected_offset: int,
    page_item_count: int,
    query_ref: str | None,
    result_ref: str | None,
) -> int:
    offset = payload.get("offset")
    if not _is_int(offset) or offset != expected_offset:
        raise ContinuityReplayError(
            f"step {step.step_id!r} expected offset {expected_offset}, observed {offset!r}",
            kind="pagination_offset_mismatch",
            failure_class="execution",
        )
    page_total = payload.get("total")
    if not _is_int(page_total) or page_total != page_item_count:
        raise ContinuityReplayError(
            f"step {step.step_id!r} returned page total {page_total!r} for {page_item_count} rows",
            kind="invalid_page_total",
            failure_class="execution",
        )
    current_query_ref = payload.get("query_ref")
    current_result_ref = payload.get("result_ref")
    if not isinstance(current_query_ref, str) or not isinstance(current_result_ref, str):
        raise ContinuityReplayError(
            f"step {step.step_id!r} returned no stable query/result refs",
            kind="missing_pagination_refs",
            failure_class="execution",
        )
    if query_ref is not None and current_query_ref != query_ref:
        raise ContinuityReplayError(
            f"step {step.step_id!r} changed query_ref across continuation pages",
            kind="pagination_query_ref_changed",
            failure_class="execution",
        )
    if result_ref is not None and current_result_ref != result_ref:
        raise ContinuityReplayError(
            f"step {step.step_id!r} changed result_ref across continuation pages",
            kind="pagination_result_ref_changed",
            failure_class="execution",
        )
    continuation = payload.get("continuation")
    next_offset = payload.get("next_offset")
    expected_next = expected_offset + page_item_count
    if continuation is None:
        if next_offset is not None:
            raise ContinuityReplayError(
                f"step {step.step_id!r} returned terminal next_offset {next_offset!r}",
                kind="terminal_next_offset_present",
                failure_class="execution",
            )
    elif not _is_int(next_offset) or next_offset != expected_next:
        raise ContinuityReplayError(
            f"step {step.step_id!r} expected next_offset {expected_next}, observed {next_offset!r}",
            kind="pagination_next_offset_mismatch",
            failure_class="execution",
        )
    return page_total


def _returned_unit_identities(
    step: ContinuityRouteStep,
    page_items: Sequence[object],
) -> list[_ReturnedUnitIdentity]:
    """Extract the declared stable identity from each returned query-unit row."""

    if not step.item_identity_path:
        raise ContinuityReplayError(
            f"paginated step {step.step_id!r} declares no item identity path",
            kind="missing_item_identity_contract",
            failure_class="plan",
        )
    identities: dict[str, _ReturnedUnitIdentity] = {}
    for index, item in enumerate(page_items):
        values = _values_at(item, step.item_identity_path)
        if len(values) != 1:
            raise ContinuityReplayError(
                f"step {step.step_id!r} item {index} has {len(values)} identities at {step.item_identity_path!r}",
                kind="invalid_item_identity",
                failure_class="execution",
            )
        value = require_json_value(values[0], context=f"continuity identity {step.step_id}")
        if isinstance(value, (dict, list)):
            raise ContinuityReplayError(
                f"step {step.step_id!r} item identity must be scalar",
                kind="non_scalar_item_identity",
                failure_class="execution",
            )
        key = json.dumps(value, sort_keys=True, separators=(",", ":"))
        if key in identities:
            raise ContinuityReplayError(
                f"step {step.step_id!r} page returned duplicate logical unit identity {key}",
                kind="duplicate_page_identity",
                failure_class="execution",
            )
        identities[key] = _ReturnedUnitIdentity(key=key, display=key)
    return list(identities.values())


def _decode_route_payload(step: ContinuityRouteStep, response_text: str) -> JSONDocument:
    try:
        decoded = json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise ContinuityReplayError(
            f"MCP tool {step.tool} returned invalid JSON: {exc}",
            kind="invalid_route_json",
            failure_class="execution",
        ) from exc
    payload = require_json_document(decoded, context=f"MCP response {step.tool}")
    if "error" in payload:
        message = payload.get("message")
        raise ContinuityReplayError(
            str(message) if message is not None else f"MCP tool {step.tool} returned an error",
            kind=str(payload.get("error") or "route_error"),
            failure_class="execution",
        )
    return dict(payload)


def _continuation_arguments(arguments: Mapping[str, object], continuation: str | None) -> RouteArguments:
    if continuation is None:
        raise ContinuityReplayError(
            "continuation arguments requested without a token",
            kind="missing_continuation",
            failure_class="execution",
        )
    expression = arguments.get("expression")
    if not isinstance(expression, str):
        raise ContinuityReplayError(
            "paginated MCP route requires an expression argument",
            kind="missing_pagination_expression",
            failure_class="plan",
        )
    return {"expression": expression, "continuation": continuation}


def _validate_scenario_declaration(scenario: ContinuityScenarioSpec) -> None:
    step_ids = [step.step_id for step in scenario.route_steps]
    if len(step_ids) != len(set(step_ids)):
        raise ContinuityReplayError(
            f"scenario {scenario.scenario_id!r} repeats a route step id",
            kind="duplicate_route_step",
            failure_class="plan",
        )
    if scenario.route_plan_signature not in scenario.equivalent_plan_signatures:
        raise ContinuityReplayError(
            f"scenario {scenario.scenario_id!r} route signature is outside its accepted plan family",
            kind="unaccepted_plan_signature",
            failure_class="plan",
        )
    route_tools = tuple(dict.fromkeys(step.tool for step in scenario.route_steps))
    if route_tools != scenario.allowed_query_surfaces:
        raise ContinuityReplayError(
            f"scenario {scenario.scenario_id!r} route tools differ from allowed surfaces",
            kind="query_surface_inventory_mismatch",
            failure_class="plan",
        )
    for step in scenario.route_steps:
        if step.paginate and not step.item_identity_path:
            raise ContinuityReplayError(
                f"paginated step {step.step_id!r} has no exact-once identity",
                kind="missing_item_identity_contract",
                failure_class="plan",
            )
        if step.exact_count_probe and (not step.paginate or step.tool != "query_units"):
            raise ContinuityReplayError(
                f"step {step.step_id!r} declares an unsupported exact-count probe",
                kind="invalid_exact_count_probe_contract",
                failure_class="plan",
            )
    if scenario.attempt_curriculum is not None and scenario.attempt_curriculum.step_id not in step_ids:
        raise ContinuityReplayError(
            f"scenario {scenario.scenario_id!r} curriculum references a missing step",
            kind="missing_curriculum_step",
            failure_class="plan",
        )


def _validate_discovery(
    requirements: Sequence[ContinuityDiscoveryRequirement],
    discovery: Mapping[str, JSONValue],
) -> None:
    tools = discovery.get("tools")
    if not isinstance(tools, Mapping):
        raise ContinuityReplayError(
            "MCP discovery returned no tool catalog",
            kind="missing_tool_catalog",
            failure_class="discovery",
        )
    for requirement in requirements:
        tool = tools.get(requirement.tool)
        if not isinstance(tool, Mapping):
            raise ContinuityReplayError(
                f"MCP discovery omitted required tool {requirement.tool!r}",
                kind="missing_discovered_tool",
                failure_class="discovery",
            )
        schema = tool.get("input_schema")
        if not isinstance(schema, Mapping):
            raise ContinuityReplayError(
                f"MCP discovery omitted input schema for {requirement.tool!r}",
                kind="missing_tool_schema",
                failure_class="discovery",
            )
        properties = schema.get("properties")
        if not isinstance(properties, Mapping):
            raise ContinuityReplayError(
                f"MCP discovery schema for {requirement.tool!r} has no properties",
                kind="missing_tool_schema_properties",
                failure_class="discovery",
            )
        missing = sorted(set(requirement.required_arguments) - set(properties))
        if missing:
            raise ContinuityReplayError(
                f"MCP discovery hid arguments for {requirement.tool!r}: {missing}",
                kind="missing_discovered_arguments",
                failure_class="discovery",
            )


def _project_evidence(
    projections: Sequence[ContinuityEvidenceProjection],
    observations: Mapping[str, JSONValue],
) -> list[str]:
    refs: set[str] = set()
    for projection in projections:
        if projection.step_id not in observations:
            raise ContinuityReplayError(
                f"evidence projection references missing step {projection.step_id!r}",
                kind="missing_evidence_step",
                failure_class="projection",
            )
        for value in _values_at(observations[projection.step_id], projection.path):
            if not isinstance(value, str):
                raise ContinuityReplayError(
                    f"evidence projection at {projection.path!r} received {type(value).__name__}",
                    kind="non_text_evidence_ref",
                    failure_class="projection",
                )
            refs.add(f"{projection.prefix}{value}")
    return sorted(refs)


def _project_attempt_grades(
    projection: ContinuityAttemptCurriculumProjection,
    observations: Mapping[str, JSONValue],
) -> dict[str, str]:
    if projection.step_id not in observations:
        raise ContinuityReplayError(
            f"attempt curriculum references missing step {projection.step_id!r}",
            kind="missing_curriculum_step_observation",
            failure_class="source_coverage",
        )
    grades: dict[str, str] = {}
    for text in _values_at(observations[projection.step_id], projection.path):
        if not isinstance(text, str):
            raise ContinuityReplayError(
                "incident curriculum row is not text",
                kind="invalid_curriculum_row",
                failure_class="reasoning",
            )
        features = {match.group("key"): match.group("value") for match in _ATTEMPT_TOKEN_RE.finditer(text)}
        case_id = features.get("case")
        if case_id is None:
            raise ContinuityReplayError(
                "incident curriculum row has no case id",
                kind="missing_curriculum_case_id",
                failure_class="reasoning",
            )
        if case_id in grades:
            raise ContinuityReplayError(
                f"incident curriculum repeats case {case_id!r}",
                kind="duplicate_curriculum_case",
                failure_class="source_coverage",
            )
        grades[case_id] = grade_incident_attempt(features)
    declared = set(projection.case_ids)
    observed = set(grades)
    if observed != declared:
        raise ContinuityReplayError(
            f"incident curriculum inventory mismatch: missing={sorted(declared - observed)}, extra={sorted(observed - declared)}",
            kind="curriculum_inventory_mismatch",
            failure_class="source_coverage",
        )
    return grades


def _values_at(root: object, path: Sequence[str | int]) -> list[object]:
    values: list[object] = [root]
    for segment in path:
        next_values: list[object] = []
        for value in values:
            if segment == "*":
                if isinstance(value, Mapping):
                    next_values.extend(value.values())
                elif isinstance(value, (list, tuple)):
                    next_values.extend(value)
                continue
            if isinstance(segment, str) and isinstance(value, Mapping) and segment in value:
                next_values.append(value[segment])
                continue
            if isinstance(segment, int) and isinstance(value, (list, tuple)) and 0 <= segment < len(value):
                next_values.append(value[segment])
        values = next_values
    return values


def _unique_json_scalars(
    values: Sequence[object],
    *,
    fact_name: str,
    failure_class: ContinuityFailureClass,
) -> list[JSONValue]:
    by_key: dict[str, JSONValue] = {}
    for value in values:
        json_value = require_json_value(value, context=f"continuity fact {fact_name}")
        if isinstance(json_value, (dict, list)):
            raise ContinuityReplayError(
                f"unique projection {fact_name!r} requires scalar values",
                kind="non_scalar_unique_projection",
                failure_class=failure_class,
            )
        key = json.dumps(json_value, sort_keys=True, separators=(",", ":"))
        by_key[key] = json_value
    return [by_key[key] for key in sorted(by_key)]


def _oracle_for(scenario: ContinuityScenarioSpec, fixture: Mapping[str, JSONValue]) -> Mapping[str, JSONValue]:
    oracles = fixture.get("oracles")
    if not isinstance(oracles, Mapping):
        raise ValueError("continuity fixture requires oracles object")
    oracle = oracles.get(scenario.fixture_key)
    if not isinstance(oracle, Mapping):
        raise ValueError(f"continuity fixture has no oracle for {scenario.fixture_key!r}")
    typed_oracle = cast(Mapping[str, JSONValue], oracle)
    expected = _json_mapping(typed_oracle, "facts")
    declared = set(scenario.required_facts)
    planted = set(expected)
    if planted != declared:
        raise ValueError(
            f"oracle fact inventory for {scenario.scenario_id!r} differs from declaration: "
            f"missing={sorted(declared - planted)}, extra={sorted(planted - declared)}"
        )
    _expected_attempt_grades(scenario, typed_oracle)
    return typed_oracle


def _expected_attempt_grades(
    scenario: ContinuityScenarioSpec,
    oracle: Mapping[str, JSONValue],
) -> dict[str, str]:
    curriculum = scenario.attempt_curriculum
    value = oracle.get("attempt_grades")
    if curriculum is None:
        if value is not None:
            raise ValueError(f"scenario {scenario.scenario_id!r} has undeclared attempt_grades")
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"scenario {scenario.scenario_id!r} requires attempt_grades object")
    expected: dict[str, str] = {}
    for case_id, grade in value.items():
        if not isinstance(grade, str) or grade not in _ATTEMPT_GRADES:
            raise ValueError(f"invalid incident attempt grade {case_id!r}: {grade!r}")
        expected[case_id] = grade
    declared = set(curriculum.case_ids)
    planted = set(expected)
    if declared != planted:
        raise ValueError(
            f"attempt curriculum inventory differs from declaration: "
            f"missing={sorted(declared - planted)}, extra={sorted(planted - declared)}"
        )
    return expected


def _json_mapping(source: Mapping[str, JSONValue], key: str) -> dict[str, JSONValue]:
    value = source.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"continuity fixture field {key!r} must be an object")
    result: dict[str, JSONValue] = {}
    for item_key, item_value in value.items():
        if not isinstance(item_key, str):
            raise ValueError(f"continuity fixture field {key!r} has a non-string key")
        result[item_key] = require_json_value(item_value, context=f"continuity fixture {key}.{item_key}")
    return result


def _string_sequence(value: object, field_name: str) -> list[str]:
    if not isinstance(value, (list, tuple)) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"continuity fixture field {field_name!r} must be a string list")
    return list(cast(Sequence[str], value))


def _materialize(value: object, fixture: Mapping[str, JSONValue]) -> object:
    if not isinstance(value, str):
        return value
    full_match = _TEMPLATE_RE.fullmatch(value)
    if full_match is not None:
        return _fixture_value(fixture, full_match.group(1))

    def replace(match: re.Match[str]) -> str:
        resolved = _fixture_value(fixture, match.group(1))
        if isinstance(resolved, (dict, list)):
            raise ValueError(f"embedded fixture reference {match.group(1)!r} must resolve to a scalar")
        if resolved is None:
            raise ValueError(f"embedded fixture reference {match.group(1)!r} resolved to null")
        return str(resolved)

    return _TEMPLATE_RE.sub(replace, value)


def _fixture_value(fixture: Mapping[str, JSONValue], path: str) -> JSONValue:
    value: object = fixture
    for segment in path.split("."):
        if not isinstance(value, Mapping) or segment not in value:
            raise ValueError(f"fixture reference path is missing: {path}")
        value = value[segment]
    return require_json_value(value, context=f"fixture reference {path}")


def _registered_discovery(server: object, transport_name: str) -> JSONDocument:
    manager = getattr(server, "_tool_manager", None)
    registered_tools = getattr(manager, "_tools", None)
    if not isinstance(registered_tools, Mapping):
        raise ContinuityReplayError(
            "FastMCP server exposes no registered tool catalog",
            kind="missing_tool_catalog",
            failure_class="discovery",
        )
    tools: JSONDocument = {}
    for name, registered in registered_tools.items():
        if not isinstance(name, str):
            continue
        schema = require_json_document(getattr(registered, "parameters", {}), context=f"registered schema {name}")
        description = getattr(registered, "description", None)
        tools[name] = _tool_discovery_payload(schema, description if isinstance(description, str) else "")
    return {
        "transport": transport_name,
        "protocol_version": "in-process-registration",
        "server_name": str(getattr(server, "name", "polylogue")),
        "server_version": None,
        "tool_count": len(tools),
        "tools": tools,
    }


def _stdio_discovery(initialize: object, raw_tools: Sequence[object], transport_name: str) -> JSONDocument:
    server_info = getattr(initialize, "serverInfo", None)
    tools: JSONDocument = {}
    for raw_tool in raw_tools:
        name = getattr(raw_tool, "name", None)
        if not isinstance(name, str):
            continue
        schema = require_json_document(getattr(raw_tool, "inputSchema", {}), context=f"stdio schema {name}")
        description = getattr(raw_tool, "description", None)
        tools[name] = _tool_discovery_payload(schema, description if isinstance(description, str) else "")
    server_name = getattr(server_info, "name", "polylogue")
    server_version = getattr(server_info, "version", None)
    return {
        "transport": transport_name,
        "protocol_version": str(getattr(initialize, "protocolVersion", "unknown")),
        "server_name": str(server_name),
        "server_version": str(server_version) if server_version is not None else None,
        "tool_count": len(tools),
        "tools": tools,
    }


def _tool_discovery_payload(schema: JSONDocument, description: str) -> JSONDocument:
    return {
        "input_schema": schema,
        "input_schema_sha256": _hash_json(schema),
        "description_sha256": hashlib.sha256(description.encode("utf-8")).hexdigest(),
    }


def _call_tool_response_text(tool: str, result: object) -> str:
    if getattr(result, "isError", False) is True:
        message = _content_text(getattr(result, "content", []))
        raise ContinuityReplayError(
            message or f"MCP tool {tool!r} returned an error",
            kind="mcp_tool_error",
            failure_class="execution",
        )
    structured = getattr(result, "structuredContent", None)
    if isinstance(structured, Mapping):
        response = structured.get("result")
        if isinstance(response, str):
            return response
    text = _content_text(getattr(result, "content", []))
    if text:
        return text
    raise ContinuityReplayError(
        f"MCP tool {tool!r} returned no JSON text content",
        kind="empty_mcp_tool_result",
        failure_class="execution",
    )


def _content_text(content: object) -> str:
    if not isinstance(content, Sequence) or isinstance(content, (str, bytes, bytearray)):
        return ""
    chunks: list[str] = []
    for item in content:
        text = getattr(item, "text", None)
        if isinstance(text, str):
            chunks.append(text)
    return "\n".join(chunks)


def _discovery_receipt(scenario: ContinuityScenarioSpec, discovery: Mapping[str, JSONValue]) -> JSONDocument:
    tools = discovery.get("tools")
    selected: JSONDocument = {}
    if isinstance(tools, Mapping):
        for requirement in scenario.discovery_requirements:
            tool = tools.get(requirement.tool)
            if not isinstance(tool, Mapping):
                continue
            schema = tool.get("input_schema")
            properties = schema.get("properties") if isinstance(schema, Mapping) else None
            visible_arguments = sorted(str(key) for key in properties) if isinstance(properties, Mapping) else []
            required_arguments_value: list[JSONValue] = list(requirement.required_arguments)
            visible_arguments_value: list[JSONValue] = list(visible_arguments)
            tool_receipt: JSONDocument = {
                "required_arguments": required_arguments_value,
                "visible_arguments": visible_arguments_value,
                "input_schema_sha256": tool.get("input_schema_sha256")
                if isinstance(tool.get("input_schema_sha256"), str)
                else None,
                "description_sha256": tool.get("description_sha256")
                if isinstance(tool.get("description_sha256"), str)
                else None,
            }
            selected[requirement.tool] = tool_receipt
    return require_json_document(
        {
            "transport": discovery.get("transport"),
            "protocol_version": discovery.get("protocol_version"),
            "server_name": discovery.get("server_name"),
            "server_version": discovery.get("server_version"),
            "tool_count": discovery.get("tool_count"),
            "required_tools": selected,
        },
        context=f"discovery receipt {scenario.scenario_id}",
    )


def _route_discovery_summary(discovery: Mapping[str, JSONValue]) -> JSONDocument:
    tools = discovery.get("tools")
    tool_names = sorted(str(name) for name in tools) if isinstance(tools, Mapping) else []
    return require_json_document(
        {
            "transport": discovery.get("transport"),
            "protocol_version": discovery.get("protocol_version"),
            "server_name": discovery.get("server_name"),
            "server_version": discovery.get("server_version"),
            "tool_count": discovery.get("tool_count"),
            "tool_names_sha256": _hash_strings(tool_names),
        },
        context="route discovery summary",
    )


def _hash_json(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _hash_strings(values: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def _is_int(value: object) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool)


def _scenario_names(value: str) -> tuple[str, ...]:
    if value == "all":
        return tuple(scenario.scenario_id for scenario in CONTINUITY_SCENARIOS)
    names = tuple(part.strip() for part in value.split(",") if part.strip())
    for name in names:
        continuity_scenario(name)
    return names


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--scenario", default="all", help="all or a comma-separated scenario id list")
    parser.add_argument("--transport", choices=("stdio", "registered"), default="stdio")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    fixture = load_oracle_catalog(args.oracle)
    result = asyncio.run(
        replay_archive(
            args.archive_root,
            fixture,
            scenario_names=_scenario_names(args.scenario),
            transport=cast(RouteTransport, args.transport),
        )
    )
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ArgumentMutator",
    "ContinuityReplayError",
    "ContinuityRoute",
    "DiscoveryMutator",
    "MCPContinuityRoute",
    "ResponseMutator",
    "RouteTransport",
    "StdioMCPContinuityRoute",
    "compare_attempt_grades",
    "compare_observed_facts",
    "execute_continuity_scenario",
    "grade_incident_attempt",
    "load_oracle_catalog",
    "main",
    "materialize_route_arguments",
    "project_fact",
    "replay_archive",
]
