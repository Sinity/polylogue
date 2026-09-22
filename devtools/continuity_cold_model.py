"""Cold, wire-only model planning lane over the existing continuity replay.

The production replay in :mod:`devtools.continuity_replay` stays the execution
oracle: it runs every declared scenario over real MCP stdio with paging,
cancellation, mutation and budget proofs.  This module adds the half that was
missing -- asking a model to *formulate* the plan from sparse operator wording
plus nothing but public, wire-captured discovery, and grading that formulation
on axes separate from the product route's own results.

Three properties are load-bearing and each has a guard:

- **Cold.** The only scenario-derived input a model ever sees is
  :attr:`ContinuityScenarioSpec.sparse_prompt`.  Route steps, fact
  projections, oracle answers, fixture keys, plan families and equivalent
  signatures are read only by the evaluator, strictly after the model has
  committed a plan.  :func:`build_cold_prompt` is the single place a prompt is
  assembled, and it takes the sparse wording as a plain string so a scenario
  object cannot leak into it.
- **Wire-only.** Discovery comes from :func:`capture_wire_discovery`, which
  pages the real ``explain`` tool over the same
  :class:`~devtools.continuity_replay.StdioMCPContinuityRoute` the replay
  uses, following the server's own continuations to exhaustion.  This module
  deliberately never imports
  ``polylogue.archive.query.discovery.QUERY_DISCOVERY_EXAMPLES``; there is no
  in-process registry fallback, and a continuation that fails to advance or
  drops its subject raises instead of degrading.
- **Registry-complete.** :func:`reconcile_registry_coverage` reconciles the
  per-scenario results against ``CONTINUITY_SCENARIOS`` itself, so a missing,
  duplicated, stale or unknown scenario is a lane failure rather than a silent
  skip.

Backends are provider-neutral: one typed plan contract, one evaluation schema,
no provider-specific parser, score field or scenario fork.
:class:`ScriptedColdModelBackend` replays a recorded plan artifact and is what
the test suite and CI use, keeping the harness hermetic.
:class:`HTTPColdModelBackend` speaks either the OpenAI chat-completions or the
Anthropic messages dialect through that same contract, and is constructed only
when an operator explicitly asks for it on the command line.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal, Protocol, TypeAlias

if __package__ in {None, ""}:  # pragma: no cover - exercised by the script entry point
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from devtools.continuity_replay import (
    ContinuityRoute,
    StdioMCPContinuityRoute,
    execute_continuity_scenario,
)
from devtools.continuity_scenarios import (
    CONTINUITY_SCENARIOS,
    ContinuityScenarioSpec,
    ContinuityTool,
    continuity_scenario,
)
from polylogue.core.json import JSONDocument, JSONValue, require_json_document
from tests.infra.continuity import load_continuity_catalog, seed_continuity_archive

COLD_MODEL_LANE_SCHEMA_VERSION = 1

ColdModelAxis = Literal[
    "discovery",
    "formulation",
    "plan_equivalence",
    "execution",
    "projection",
    "citations",
    "uncertainty",
    "stop_conditions",
]
COLD_MODEL_AXES: tuple[ColdModelAxis, ...] = (
    "discovery",
    "formulation",
    "plan_equivalence",
    "execution",
    "projection",
    "citations",
    "uncertainty",
    "stop_conditions",
)

ColdModelUncertainty = Literal["low", "medium", "high"]
ColdModelDialect = Literal["openai", "anthropic"]
ColdModelDisposition = Literal["pass", "model_variance", "fail"]

PlanArguments: TypeAlias = dict[str, JSONValue]

EXPLAIN_DISCOVERY_SUBJECTS: tuple[str, ...] = ("result", "recovery")

_ALLOWED_PLAN_TOOLS: frozenset[str] = frozenset({"query", "read", "get", "explain", "context", "status"})
_ALLOWED_UNCERTAINTY: frozenset[str] = frozenset({"low", "medium", "high"})


class ColdModelLaneError(RuntimeError):
    """Structured cold-lane failure reportable without a traceback."""

    def __init__(self, message: str, *, kind: str, axis: ColdModelAxis) -> None:
        super().__init__(message)
        self.kind = kind
        self.axis = axis


def _as_tool(value: str) -> ContinuityTool:
    """Narrow an already-validated tool token to the declared literal."""
    mapping: dict[str, ContinuityTool] = {
        "query": "query",
        "read": "read",
        "get": "get",
        "explain": "explain",
        "context": "context",
        "status": "status",
    }
    return mapping[value]


def _as_uncertainty(value: str) -> ColdModelUncertainty:
    """Narrow an already-validated uncertainty token to the declared literal."""
    mapping: dict[str, ColdModelUncertainty] = {"low": "low", "medium": "medium", "high": "high"}
    return mapping[value]


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode("utf-8")).hexdigest()


# ── Typed plan contract ───────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ColdModelPlanStep:
    """One tool call a model proposes before any oracle access."""

    tool: ContinuityTool
    arguments: PlanArguments

    @property
    def plan_atom(self) -> str:
        """Mirror ``ContinuityRouteStep.plan_atom`` so signatures compare."""
        expression = self.arguments.get("expression")
        if self.tool == "query" and isinstance(expression, str) and expression.split():
            return f"query:{expression.split(maxsplit=1)[0]}"
        return self.tool

    def to_payload(self) -> JSONDocument:
        return {"tool": self.tool, "arguments": dict(self.arguments), "plan_atom": self.plan_atom}


@dataclass(frozen=True, slots=True)
class ColdModelPlan:
    """A model's committed plan.  Constructed only from model output."""

    steps: tuple[ColdModelPlanStep, ...]
    stop_conditions: tuple[str, ...]
    citation_fields: tuple[str, ...]
    uncertainty: ColdModelUncertainty

    @property
    def plan_signature(self) -> tuple[str, ...]:
        return tuple(step.plan_atom for step in self.steps)

    @property
    def tools(self) -> tuple[ContinuityTool, ...]:
        return tuple(step.tool for step in self.steps)

    def to_payload(self) -> JSONDocument:
        return {
            "steps": [step.to_payload() for step in self.steps],
            "plan_signature": list(self.plan_signature),
            "stop_conditions": list(self.stop_conditions),
            "citation_fields": list(self.citation_fields),
            "uncertainty": self.uncertainty,
        }


def _string_tuple(value: object, field_name: str) -> tuple[str, ...]:
    if isinstance(value, str):
        raise ColdModelLaneError(
            f"plan field {field_name!r} was a bare string, expected a list",
            kind="plan_field_not_a_list",
            axis="formulation",
        )
    if not isinstance(value, (list, tuple)):
        raise ColdModelLaneError(
            f"plan field {field_name!r} was not a list",
            kind="plan_field_not_a_list",
            axis="formulation",
        )
    entries: list[str] = []
    for entry in value:
        if not isinstance(entry, str):
            raise ColdModelLaneError(
                f"plan field {field_name!r} held a non-string entry",
                kind="plan_field_entry_not_a_string",
                axis="formulation",
            )
        entries.append(entry)
    return tuple(entries)


def parse_cold_model_plan(text: str) -> ColdModelPlan:
    """Parse a model's raw answer into the typed plan contract.

    Strict on purpose: an unparsable or ill-typed answer is a *formulation*
    failure with a named kind, never a silently-empty plan that would then
    trivially satisfy the downstream axes.
    """

    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = "\n".join(line for line in stripped.splitlines() if not line.startswith("```")).strip()
    try:
        raw = json.loads(stripped)
    except ValueError as exc:
        raise ColdModelLaneError(
            f"model answer was not JSON: {exc}",
            kind="plan_unparsable",
            axis="formulation",
        ) from exc
    if not isinstance(raw, dict):
        raise ColdModelLaneError(
            f"model answer was {type(raw).__name__}, expected a JSON object",
            kind="plan_not_an_object",
            axis="formulation",
        )

    raw_steps = raw.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise ColdModelLaneError(
            "model plan declared no steps",
            kind="plan_without_steps",
            axis="formulation",
        )
    steps: list[ColdModelPlanStep] = []
    for index, raw_step in enumerate(raw_steps):
        if not isinstance(raw_step, dict):
            raise ColdModelLaneError(
                f"plan step {index} was not an object",
                kind="plan_step_not_an_object",
                axis="formulation",
            )
        tool = raw_step.get("tool")
        if not isinstance(tool, str) or tool not in _ALLOWED_PLAN_TOOLS:
            raise ColdModelLaneError(
                f"plan step {index} named an unknown tool {tool!r}",
                kind="plan_unknown_tool",
                axis="formulation",
            )
        arguments = raw_step.get("arguments", {})
        if not isinstance(arguments, dict) or not all(isinstance(key, str) for key in arguments):
            raise ColdModelLaneError(
                f"plan step {index} arguments were not a string-keyed object",
                kind="plan_step_arguments_invalid",
                axis="formulation",
            )
        steps.append(ColdModelPlanStep(tool=_as_tool(tool), arguments=dict(arguments)))

    uncertainty = raw.get("uncertainty")
    if not isinstance(uncertainty, str) or uncertainty not in _ALLOWED_UNCERTAINTY:
        raise ColdModelLaneError(
            f"plan declared unusable uncertainty {uncertainty!r}",
            kind="plan_uncertainty_invalid",
            axis="formulation",
        )
    return ColdModelPlan(
        steps=tuple(steps),
        stop_conditions=_string_tuple(raw.get("stop_conditions", ()), "stop_conditions"),
        citation_fields=_string_tuple(raw.get("citation_fields", ()), "citation_fields"),
        uncertainty=_as_uncertainty(uncertainty),
    )


# ── Wire-only discovery ───────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class WireDiscoveryCapture:
    """Everything public a cold client can learn from the live server."""

    transport_name: str
    protocol_version: str
    tools: tuple[str, ...]
    tool_schemas: JSONDocument
    examples: tuple[JSONDocument, ...]
    hops: int
    schema_digest: str
    catalog_digest: str

    def public_payload(self) -> JSONDocument:
        """The exact object handed to a model.  No oracle content here."""
        return {
            "transport": self.transport_name,
            "protocol_version": self.protocol_version,
            "tools": list(self.tools),
            "tool_schemas": dict(self.tool_schemas),
            "examples": [dict(example) for example in self.examples],
        }

    def receipt(self) -> JSONDocument:
        return {
            "transport": self.transport_name,
            "protocol_version": self.protocol_version,
            "tool_count": len(self.tools),
            "tools": list(self.tools),
            "example_count": len(self.examples),
            "continuation_hops": self.hops,
            "schema_digest": self.schema_digest,
            "catalog_digest": self.catalog_digest,
        }


def _discovery_tool_schemas(discovery: Mapping[str, JSONValue]) -> tuple[tuple[str, ...], JSONDocument]:
    raw_tools = discovery.get("tools")
    schemas: JSONDocument = {}
    names: list[str] = []
    if isinstance(raw_tools, dict):
        for name, entry in raw_tools.items():
            if not isinstance(entry, dict):
                continue
            names.append(name)
            schemas[name] = {"input_schema": entry.get("input_schema", {})}
    return tuple(sorted(names)), schemas


async def capture_wire_discovery(
    route: ContinuityRoute,
    *,
    subjects: Sequence[str] = EXPLAIN_DISCOVERY_SUBJECTS,
    max_hops: int = 40,
) -> WireDiscoveryCapture:
    """Page the live ``explain`` tool to exhaustion over the real wire.

    There is no in-process fallback.  A budget-exceeded response that returns
    no page, drops its ``subject``, or fails to advance its ``offset`` raises
    :class:`ColdModelLaneError` -- the controlled regression that the closed
    polylogue-3k30 defect would trip.
    """

    names, schemas = _discovery_tool_schemas(route.discovery)
    protocol_version = route.discovery.get("protocol_version")
    examples: dict[str, JSONDocument] = {}
    hops = 0

    for subject in subjects:
        arguments: PlanArguments = {"subject": subject}
        seen_offsets: set[int] = set()
        for _ in range(max_hops):
            hops += 1
            raw = await route.invoke("explain", arguments)
            try:
                body = json.loads(raw)
            except ValueError as exc:
                raise ColdModelLaneError(
                    f"explain(subject={subject!r}) returned non-JSON: {exc}",
                    kind="discovery_payload_unparsable",
                    axis="discovery",
                ) from exc
            if not isinstance(body, dict):
                raise ColdModelLaneError(
                    f"explain(subject={subject!r}) returned {type(body).__name__}, expected an object",
                    kind="discovery_payload_not_an_object",
                    axis="discovery",
                )
            if not body.get("budget_exceeded"):
                _collect_examples(body.get("examples"), examples)
                break
            page = body.get("page")
            if not isinstance(page, dict):
                raise ColdModelLaneError(
                    f"explain(subject={subject!r}) exceeded its budget without returning a bounded page",
                    kind="discovery_zero_progress",
                    axis="discovery",
                )
            _collect_examples(page.get("examples"), examples)
            continuation = body.get("continuation")
            if not isinstance(continuation, dict):
                raise ColdModelLaneError(
                    f"explain(subject={subject!r}) paged without offering a continuation",
                    kind="discovery_missing_continuation",
                    axis="discovery",
                )
            if continuation.get("tool") != "explain":
                raise ColdModelLaneError(
                    f"explain continuation routed to {continuation.get('tool')!r}",
                    kind="discovery_continuation_wrong_tool",
                    axis="discovery",
                )
            next_arguments = continuation.get("arguments")
            if not isinstance(next_arguments, dict):
                raise ColdModelLaneError(
                    "explain continuation carried no arguments object",
                    kind="discovery_continuation_argumentless",
                    axis="discovery",
                )
            if next_arguments.get("subject") != subject:
                raise ColdModelLaneError(
                    f"explain continuation dropped its subject: {next_arguments!r}",
                    kind="discovery_continuation_dropped_subject",
                    axis="discovery",
                )
            offset = next_arguments.get("offset")
            if not isinstance(offset, int) or isinstance(offset, bool) or offset <= 0:
                raise ColdModelLaneError(
                    f"explain continuation did not advance an offset: {next_arguments!r}",
                    kind="discovery_continuation_not_advancing",
                    axis="discovery",
                )
            if offset in seen_offsets:
                raise ColdModelLaneError(
                    f"explain continuation repeated offset {offset}",
                    kind="discovery_continuation_not_advancing",
                    axis="discovery",
                )
            seen_offsets.add(offset)
            arguments = {str(key): value for key, value in next_arguments.items()}
        else:
            raise ColdModelLaneError(
                f"explain(subject={subject!r}) did not terminate within {max_hops} continuation hops",
                kind="discovery_did_not_terminate",
                axis="discovery",
            )

    if not examples:
        raise ColdModelLaneError(
            "wire discovery retrieved no catalogue examples at all",
            kind="discovery_empty_catalogue",
            axis="discovery",
        )
    ordered = tuple(examples[key] for key in sorted(examples))
    return WireDiscoveryCapture(
        transport_name=route.transport_name,
        protocol_version=protocol_version if isinstance(protocol_version, str) else "unknown",
        tools=names,
        tool_schemas=schemas,
        examples=ordered,
        hops=hops,
        schema_digest=_digest(schemas),
        catalog_digest=_digest([sorted(examples)]),
    )


def _collect_examples(raw: object, into: dict[str, JSONDocument]) -> None:
    if not isinstance(raw, list):
        return
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        key = entry.get("key")
        if isinstance(key, str):
            into.setdefault(key, entry)


# ── Provider-neutral model backends ───────────────────────────────────


@dataclass(frozen=True, slots=True)
class ColdModelIdentity:
    """Who answered, under which declared version."""

    family: str
    model: str
    prompt_version: str

    def to_payload(self) -> JSONDocument:
        return {"family": self.family, "model": self.model, "prompt_version": self.prompt_version}


@dataclass(frozen=True, slots=True)
class ColdModelAnswer:
    """One backend answer plus its declared resource measurements."""

    text: str
    prompt_bytes: int
    answer_bytes: int
    wall_ms: float
    native_token_counts: JSONDocument

    def to_payload(self) -> JSONDocument:
        return {
            "prompt_bytes": self.prompt_bytes,
            "answer_bytes": self.answer_bytes,
            "wall_ms": self.wall_ms,
            "native_token_counts": dict(self.native_token_counts),
        }


class ColdModelBackend(Protocol):
    """One typed contract for every provider family."""

    @property
    def identity(self) -> ColdModelIdentity:
        raise NotImplementedError

    def answer(self, prompt: str) -> ColdModelAnswer:
        raise NotImplementedError


PROMPT_VERSION = "cold-continuity-plan-v1"


def build_cold_prompt(sparse_prompt: str, discovery: WireDiscoveryCapture, *, max_calls: int) -> str:
    """Assemble the only text a model sees.

    Takes the sparse wording as a plain string rather than a scenario object,
    so no route step, fact projection, fixture key or plan family can reach a
    prompt by accident.
    """

    contract = {
        "steps": [{"tool": "<one of the tools below>", "arguments": {"<argument>": "<value>"}}],
        "stop_conditions": ["<when you would stop calling>"],
        "citation_fields": ["<result field you would cite>"],
        "uncertainty": "low|medium|high",
    }
    return "\n".join(
        (
            "You are a cold client of an MCP archive server. You have never seen this archive.",
            "Answer the operator's question by formulating a plan of tool calls.",
            "",
            f"Operator question: {sparse_prompt}",
            f"Call budget: at most {max_calls} tool calls.",
            "",
            "Public server discovery (tool schemas and query examples, retrieved over the wire):",
            json.dumps(discovery.public_payload(), sort_keys=True),
            "",
            "Reply with JSON only, matching this shape exactly:",
            json.dumps(contract, sort_keys=True),
        )
    )


@dataclass(frozen=True, slots=True)
class ScriptedColdModelBackend:
    """Replay a recorded plan artifact.  Hermetic: no network, no clock skew.

    This is the backend the test suite and CI use.  It is not a stand-in for a
    model's judgement -- it replays answers a model (or an author) produced
    out of band, so the evaluator, the wire discovery client and the registry
    reconciliation are all exercised without the harness making its first ever
    outbound request.
    """

    answers: Mapping[str, str]
    model_identity: ColdModelIdentity = ColdModelIdentity(
        family="scripted", model="recorded-plan-artifact", prompt_version=PROMPT_VERSION
    )

    @property
    def identity(self) -> ColdModelIdentity:
        return self.model_identity

    def answer(self, prompt: str) -> ColdModelAnswer:
        key = _scenario_key_from_prompt(prompt)
        if key not in self.answers:
            raise ColdModelLaneError(
                f"scripted backend has no recorded answer for {key!r}",
                kind="scripted_answer_missing",
                axis="formulation",
            )
        text = self.answers[key]
        return ColdModelAnswer(
            text=text,
            prompt_bytes=len(prompt.encode("utf-8")),
            answer_bytes=len(text.encode("utf-8")),
            wall_ms=0.0,
            native_token_counts={},
        )


_PROMPT_QUESTION_MARKER = "Operator question: "


def _scenario_key_from_prompt(prompt: str) -> str:
    for line in prompt.splitlines():
        if line.startswith(_PROMPT_QUESTION_MARKER):
            return line[len(_PROMPT_QUESTION_MARKER) :].strip()
    raise ColdModelLaneError(
        "prompt carried no operator question",
        kind="prompt_without_question",
        axis="formulation",
    )


@dataclass(frozen=True, slots=True)
class HTTPColdModelBackend:
    """One HTTP adapter for both supported families.

    The request body and the answer extraction differ by dialect because the
    wire formats differ; the *plan contract* and the *evaluation schema* do
    not.  There is no provider-specific plan parser, score field or scenario
    fork anywhere downstream of this class.
    """

    dialect: ColdModelDialect
    model: str
    base_url: str
    api_key: str
    timeout_seconds: float = 120.0

    @property
    def identity(self) -> ColdModelIdentity:
        return ColdModelIdentity(family=self.dialect, model=self.model, prompt_version=PROMPT_VERSION)

    def answer(self, prompt: str) -> ColdModelAnswer:
        import httpx

        if self.dialect == "openai":
            url = f"{self.base_url.rstrip('/')}/chat/completions"
            headers = {"Authorization": f"Bearer {self.api_key}"}
            payload: JSONDocument = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
            }
        else:
            url = f"{self.base_url.rstrip('/')}/messages"
            headers = {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"}
            payload = {
                "model": self.model,
                "max_tokens": 4096,
                "messages": [{"role": "user", "content": prompt}],
            }

        started = time.perf_counter_ns()
        response = httpx.post(url, json=payload, headers=headers, timeout=self.timeout_seconds)
        wall_ms = round((time.perf_counter_ns() - started) / 1_000_000, 3)
        if response.status_code >= 400:
            raise ColdModelLaneError(
                f"{self.dialect} backend returned HTTP {response.status_code}",
                kind="backend_http_error",
                axis="formulation",
            )
        body = response.json()
        text = self._extract_text(body)
        return ColdModelAnswer(
            text=text,
            prompt_bytes=len(prompt.encode("utf-8")),
            answer_bytes=len(text.encode("utf-8")),
            wall_ms=wall_ms,
            native_token_counts=self._native_counts(body),
        )

    def _extract_text(self, body: object) -> str:
        if not isinstance(body, dict):
            raise ColdModelLaneError(
                "backend answer was not an object",
                kind="backend_answer_malformed",
                axis="formulation",
            )
        if self.dialect == "openai":
            choices = body.get("choices")
            if isinstance(choices, list) and choices and isinstance(choices[0], dict):
                message = choices[0].get("message")
                if isinstance(message, dict) and isinstance(message.get("content"), str):
                    return str(message["content"])
        else:
            content = body.get("content")
            if isinstance(content, list):
                parts = [block.get("text", "") for block in content if isinstance(block, dict)]
                joined = "".join(part for part in parts if isinstance(part, str))
                if joined:
                    return joined
        raise ColdModelLaneError(
            f"{self.dialect} backend answer carried no text",
            kind="backend_answer_malformed",
            axis="formulation",
        )

    def _native_counts(self, body: object) -> JSONDocument:
        """Record the backend's own counters verbatim; never add across families."""
        if not isinstance(body, dict):
            return {}
        usage = body.get("usage")
        if not isinstance(usage, dict):
            return {}
        return {str(key): value for key, value in usage.items() if isinstance(value, int)}


# ── Evaluator ─────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ColdModelAxisGrade:
    """One graded axis, independent of every other axis."""

    axis: ColdModelAxis
    status: Literal["pass", "fail"]
    detail: str

    def to_payload(self) -> JSONDocument:
        return {"axis": self.axis, "status": self.status, "detail": self.detail}


def _citable_field(path: Sequence[str | int]) -> str | None:
    """The last named field in an evidence path, ignoring wildcards and indices."""
    for segment in reversed(tuple(path)):
        if isinstance(segment, str) and segment not in {"*", "items"}:
            return segment
    return None


def grade_formulation(
    scenario: ContinuityScenarioSpec,
    plan: ColdModelPlan,
    capture: WireDiscoveryCapture,
) -> tuple[ColdModelAxisGrade, ...]:
    """Grade the committed plan against the answer key.

    Every input read here -- discovery requirements, allowed surfaces,
    equivalent plan signatures, evidence projections, stop conditions -- is
    read *after* the model has committed, and none of it was ever in a prompt.
    """

    grades: list[ColdModelAxisGrade] = []

    required_tools = {requirement.tool for requirement in scenario.discovery_requirements}
    visible = set(capture.tools)
    missing_from_wire = sorted(required_tools - visible)
    grades.append(
        ColdModelAxisGrade(
            "discovery",
            "fail" if missing_from_wire else "pass",
            f"tools missing from wire discovery: {missing_from_wire}"
            if missing_from_wire
            else f"{len(capture.tools)} tools and {len(capture.examples)} examples captured over the wire",
        )
    )

    disallowed = sorted(set(plan.tools) - set(scenario.allowed_query_surfaces))
    grades.append(
        ColdModelAxisGrade(
            "formulation",
            "fail" if disallowed else "pass",
            f"plan used surfaces outside the declared set: {disallowed}"
            if disallowed
            else f"plan used only declared surfaces {sorted(set(plan.tools))}",
        )
    )

    accepted = {tuple(signature) for signature in scenario.equivalent_plan_signatures}
    accepted.add(scenario.route_plan_signature)
    equivalent = plan.plan_signature in accepted
    grades.append(
        ColdModelAxisGrade(
            "plan_equivalence",
            "pass" if equivalent else "fail",
            f"plan signature {list(plan.plan_signature)} "
            + ("is an accepted equivalent" if equivalent else f"is outside {sorted(accepted)}"),
        )
    )

    required_citations = {
        field for projection in scenario.evidence_projections if (field := _citable_field(projection.path))
    }
    cited = set(plan.citation_fields)
    # A scenario with no declared evidence projection requires no citation; one
    # that has them requires the plan to name at least one of them.
    citations_ok = not required_citations or bool(cited & required_citations)
    grades.append(
        ColdModelAxisGrade(
            "citations",
            "pass" if citations_ok else "fail",
            f"plan cited {sorted(cited)} against required evidence {sorted(required_citations)}",
        )
    )

    grades.append(
        ColdModelAxisGrade(
            "uncertainty",
            "pass" if plan.uncertainty in _ALLOWED_UNCERTAINTY else "fail",
            f"plan declared uncertainty {plan.uncertainty!r}",
        )
    )

    declared_stops = bool(plan.stop_conditions)
    grades.append(
        ColdModelAxisGrade(
            "stop_conditions",
            "pass" if declared_stops else "fail",
            f"plan declared {len(plan.stop_conditions)} stop conditions "
            f"against {len(scenario.stop_conditions)} required",
        )
    )
    return tuple(grades)


def grade_execution(replay_result: Mapping[str, JSONValue]) -> tuple[ColdModelAxisGrade, ...]:
    """Grade the product route's own outcome, separately from formulation."""

    status = replay_result.get("status")
    diagnostics = replay_result.get("diagnostics")
    diagnostic_count = len(diagnostics) if isinstance(diagnostics, list) else 0
    projection_failures = 0
    if isinstance(diagnostics, list):
        projection_failures = sum(
            1
            for entry in diagnostics
            if isinstance(entry, dict) and entry.get("failure_class") in {"projection", "source_coverage"}
        )
    return (
        ColdModelAxisGrade(
            "execution",
            "pass" if status == "pass" else "fail",
            f"production replay reported status={status!r} with {diagnostic_count} diagnostics",
        ),
        ColdModelAxisGrade(
            "projection",
            "pass" if projection_failures == 0 else "fail",
            f"{projection_failures} fact/coverage diagnostics from the independent oracle",
        ),
    )


# ── Variance policy and registry reconciliation ───────────────────────


@dataclass(frozen=True, slots=True)
class ColdModelVariancePolicy:
    """Declared repeated-run policy separating product defects from variance."""

    attempts: int = 1
    required_passes: int = 1

    def __post_init__(self) -> None:
        if self.attempts < 1:
            raise ValueError("variance policy needs at least one attempt")
        if not 1 <= self.required_passes <= self.attempts:
            raise ValueError("required_passes must be between 1 and attempts")

    def disposition(self, *, passes: int, product_failure: bool) -> ColdModelDisposition:
        """A product-route failure is never excused as model variance."""
        if product_failure:
            return "fail"
        if passes >= self.required_passes:
            return "pass"
        if passes > 0:
            return "model_variance"
        return "fail"

    def to_payload(self) -> JSONDocument:
        return {"attempts": self.attempts, "required_passes": self.required_passes}


def reconcile_registry_coverage(scenario_ids: Sequence[str]) -> tuple[str, ...]:
    """Return one error per missing, duplicated or unknown scenario."""

    declared = [scenario.scenario_id for scenario in CONTINUITY_SCENARIOS]
    errors: list[str] = []
    seen: dict[str, int] = {}
    for scenario_id in scenario_ids:
        seen[scenario_id] = seen.get(scenario_id, 0) + 1
    for scenario_id in declared:
        if scenario_id not in seen:
            errors.extend([f"scenario {scenario_id!r} has no result (missing or silently skipped)"])
    for scenario_id, count in sorted(seen.items()):
        if count > 1:
            errors.append(f"scenario {scenario_id!r} produced {count} results (duplicate)")
        if scenario_id not in declared:
            errors.append(f"result {scenario_id!r} is not a declared registry scenario (stale)")
    return tuple(errors)


# ── Lane runner ───────────────────────────────────────────────────────


async def run_cold_model_lane(
    archive_root: Path,
    fixture: Mapping[str, JSONValue],
    backend: ColdModelBackend,
    *,
    variance: ColdModelVariancePolicy | None = None,
    scenario_names: Sequence[str] | None = None,
    route: ContinuityRoute | None = None,
) -> JSONDocument:
    """Run the cold lane over every registry scenario and reconcile coverage."""

    policy = variance or ColdModelVariancePolicy()
    selected = (
        tuple(scenario.scenario_id for scenario in CONTINUITY_SCENARIOS)
        if scenario_names is None
        else tuple(scenario_names)
    )
    started_ns = time.perf_counter_ns()
    results: list[JSONValue] = []
    scenario_ids: list[str] = []
    capture: WireDiscoveryCapture

    async def _drive(active: ContinuityRoute) -> WireDiscoveryCapture:
        captured = await capture_wire_discovery(active)
        for name in selected:
            result = await _run_one_scenario(name, fixture, backend, captured, policy, active)
            results.append(result)
            scenario_ids.append(name)
        return captured

    if route is None:
        async with StdioMCPContinuityRoute(archive_root) as owned_route:
            capture = await _drive(owned_route)
    else:
        capture = await _drive(route)

    coverage_errors = reconcile_registry_coverage(scenario_ids)
    dispositions = [result["disposition"] for result in results if isinstance(result, dict)]
    status = "pass" if not coverage_errors and all(disposition == "pass" for disposition in dispositions) else "fail"
    report: dict[str, object] = {
        "schema_version": COLD_MODEL_LANE_SCHEMA_VERSION,
        "fixture_id": fixture.get("fixture_id"),
        "archive_root": str(archive_root.resolve()),
        "model": backend.identity.to_payload(),
        "variance_policy": policy.to_payload(),
        "discovery_receipt": capture.receipt(),
        "registry_scenario_count": len(CONTINUITY_SCENARIOS),
        "scenario_count": len(results),
        "coverage_errors": list(coverage_errors),
        "elapsed_ms": round((time.perf_counter_ns() - started_ns) / 1_000_000, 3),
        "status": status,
        "results": results,
    }
    return require_json_document(report, context="continuity cold-model report")


async def _run_one_scenario(
    scenario_name: str,
    fixture: Mapping[str, JSONValue],
    backend: ColdModelBackend,
    capture: WireDiscoveryCapture,
    policy: ColdModelVariancePolicy,
    route: ContinuityRoute,
) -> JSONDocument:
    scenario = continuity_scenario(scenario_name)
    prompt = build_cold_prompt(scenario.sparse_prompt, capture, max_calls=scenario.budget.max_calls)

    attempts: list[JSONValue] = []
    passes = 0
    for attempt_index in range(policy.attempts):
        status, payload = _attempt(scenario, backend, prompt, capture, attempt_index)
        attempts.append(payload)
        if status == "pass":
            passes += 1

    replay_result = await execute_continuity_scenario(scenario_name, fixture, route)
    execution_grades = grade_execution(replay_result)
    product_failure = any(grade.status == "fail" for grade in execution_grades)
    disposition = policy.disposition(passes=passes, product_failure=product_failure)

    return {
        "scenario": scenario_name,
        "sparse_prompt": scenario.sparse_prompt,
        "prompt_bytes": len(prompt.encode("utf-8")),
        "discovery_bytes": len(json.dumps(capture.public_payload(), sort_keys=True).encode("utf-8")),
        "attempts": attempts,
        "attempt_count": len(attempts),
        "formulation_passes": passes,
        "execution_grades": [grade.to_payload() for grade in execution_grades],
        "product_status": replay_result.get("status"),
        "disposition": disposition,
        "correction_cost": len(attempts) - passes,
    }


def _attempt(
    scenario: ContinuityScenarioSpec,
    backend: ColdModelBackend,
    prompt: str,
    capture: WireDiscoveryCapture,
    attempt_index: int,
) -> tuple[str, JSONDocument]:
    try:
        answer = backend.answer(prompt)
    except ColdModelLaneError as exc:
        return "fail", {
            "attempt": attempt_index,
            "formulation_status": "fail",
            "failure_kind": exc.kind,
            "failure_axis": exc.axis,
            "grades": [],
            "plan": None,
            "resources": {},
        }
    try:
        plan = parse_cold_model_plan(answer.text)
    except ColdModelLaneError as exc:
        return "fail", {
            "attempt": attempt_index,
            "formulation_status": "fail",
            "failure_kind": exc.kind,
            "failure_axis": exc.axis,
            "grades": [],
            "plan": None,
            "resources": answer.to_payload(),
        }
    grades = grade_formulation(scenario, plan, capture)
    status = "pass" if all(grade.status == "pass" for grade in grades) else "fail"
    return status, {
        "attempt": attempt_index,
        "formulation_status": status,
        "failure_kind": None,
        "failure_axis": None,
        "grades": [grade.to_payload() for grade in grades],
        "plan": plan.to_payload(),
        "resources": answer.to_payload(),
    }


# ── Command entry point ───────────────────────────────────────────────


def _load_scripted_backend(path: Path) -> ScriptedColdModelBackend:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"plan artifact {path} is not a JSON object")
    answers: dict[str, str] = {}
    identity = ColdModelIdentity(
        family=str(payload.get("family", "scripted")),
        model=str(payload.get("model", "recorded-plan-artifact")),
        prompt_version=PROMPT_VERSION,
    )
    raw_plans = payload.get("plans")
    if not isinstance(raw_plans, dict):
        raise SystemExit(f"plan artifact {path} declared no 'plans' object")
    for scenario_id, plan in raw_plans.items():
        scenario = continuity_scenario(str(scenario_id))
        answers[scenario.sparse_prompt] = json.dumps(plan)
    return ScriptedColdModelBackend(answers=answers, model_identity=identity)


def _http_backend(dialect: ColdModelDialect, model: str, base_url: str, key_env: str) -> HTTPColdModelBackend:
    api_key = os.environ.get(key_env, "")
    if not api_key:
        raise SystemExit(f"backend {dialect} requires {key_env} in the environment")
    return HTTPColdModelBackend(dialect=dialect, model=model, base_url=base_url, api_key=api_key)


def main(argv: list[str] | None = None) -> int:
    """Run the cold, wire-only model planning lane."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("scripted", "openai", "anthropic"), default="scripted")
    parser.add_argument("--plans", type=Path, help="Recorded plan artifact for the scripted backend.")
    parser.add_argument("--model", default="", help="Model id for an HTTP backend.")
    parser.add_argument("--base-url", default="", help="API base URL for an HTTP backend.")
    parser.add_argument("--api-key-env", default="", help="Environment variable holding the API key.")
    parser.add_argument("--attempts", type=int, default=1)
    parser.add_argument("--required-passes", type=int, default=1)
    parser.add_argument("--scenario", action="append", default=None)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--json", action="store_true", help="Print the report as JSON.")
    args = parser.parse_args(argv)

    if args.backend == "scripted":
        if args.plans is None:
            raise SystemExit("--backend scripted requires --plans")
        backend: ColdModelBackend = _load_scripted_backend(args.plans)
    else:
        if not args.model or not args.base_url or not args.api_key_env:
            raise SystemExit(f"--backend {args.backend} requires --model, --base-url and --api-key-env")
        dialect: ColdModelDialect = "openai" if args.backend == "openai" else "anthropic"
        backend = _http_backend(dialect, args.model, args.base_url, args.api_key_env)

    policy = ColdModelVariancePolicy(attempts=args.attempts, required_passes=args.required_passes)
    with TemporaryDirectory(prefix="polylogue-cold-model-") as workdir:
        archive_root = Path(workdir) / "archive"
        catalog = load_continuity_catalog()
        seed_continuity_archive(archive_root, catalog=catalog)
        report = asyncio.run(
            run_cold_model_lane(
                archive_root,
                catalog,
                backend,
                variance=policy,
                scenario_names=tuple(args.scenario) if args.scenario else None,
            )
        )

    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report["status"] == "pass" else 1


__all__ = [
    "COLD_MODEL_AXES",
    "COLD_MODEL_LANE_SCHEMA_VERSION",
    "ColdModelAnswer",
    "ColdModelAxis",
    "ColdModelAxisGrade",
    "ColdModelBackend",
    "ColdModelDialect",
    "ColdModelDisposition",
    "ColdModelIdentity",
    "ColdModelLaneError",
    "ColdModelPlan",
    "ColdModelPlanStep",
    "ColdModelUncertainty",
    "ColdModelVariancePolicy",
    "HTTPColdModelBackend",
    "PROMPT_VERSION",
    "ScriptedColdModelBackend",
    "WireDiscoveryCapture",
    "build_cold_prompt",
    "capture_wire_discovery",
    "grade_execution",
    "grade_formulation",
    "main",
    "parse_cold_model_plan",
    "reconcile_registry_coverage",
    "run_cold_model_lane",
]


if __name__ == "__main__":  # pragma: no cover - script entry point
    raise SystemExit(main())
