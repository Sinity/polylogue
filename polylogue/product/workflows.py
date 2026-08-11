"""Executable demo-archive workflow paths used by product behavior tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

from polylogue.scenarios import DEMO_CLAUDE_CODE_SESSION_ID

JsonPathSegment: TypeAlias = str | int
JsonPath: TypeAlias = tuple[JsonPathSegment, ...]
JsonKind: TypeAlias = Literal["object", "array", "string", "integer", "number", "boolean", "null", "any"]
OutputKind: TypeAlias = Literal["json_object", "json_array", "human"]


@dataclass(frozen=True, slots=True)
class JsonExpectation:
    """A structural assertion over a golden-path JSON output."""

    path: JsonPath
    kind: JsonKind = "any"


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


EXECUTABLE_WORKFLOW_GOLDEN_PATHS: tuple[ExecutableWorkflowGoldenPath, ...] = (
    ExecutableWorkflowGoldenPath(
        id="select-exact-session-json",
        workflow_id="resolve-ref-drilldown",
        description="Exact id query selects the demo Claude Code session without broad FTS fallback.",
        command=("find", f"id:{DEMO_CLAUDE_CODE_SESSION_ID}", "then", "select", "--format", "json"),
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
        command=("find", f"session:{DEMO_CLAUDE_CODE_SESSION_ID}", "then", "select", "--format", "json"),
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
        stdout_contains=("The module structure looks good", "Inspecting generated workload record"),
        required_affordance_ids=("read",),
    ),
    ExecutableWorkflowGoldenPath(
        id="read-context-image-human",
        workflow_id="find-then-context-image",
        description="A selected session compiles into a bounded context image through the real read view.",
        command=(
            "find",
            f"id:{DEMO_CLAUDE_CODE_SESSION_ID}",
            "then",
            "read",
            "--view",
            "context-image",
            "--max-sessions",
            "1",
        ),
        action_path=("read",),
        output_kind="human",
        stdout_contains=("context: 1 segment(s)", "The module structure looks good"),
        required_affordance_ids=("read",),
    ),
    ExecutableWorkflowGoldenPath(
        id="continue-context-json",
        workflow_id="find-then-successor-context",
        description="Continuation compiles evidence-rich successor context for the selected session.",
        command=(
            "find",
            f"id:{DEMO_CLAUDE_CODE_SESSION_ID}",
            "then",
            "continue",
            "--format",
            "json",
        ),
        action_path=("continue",),
        output_kind="json_object",
        json_expectations=(
            JsonExpectation(("spec",), "object"),
            JsonExpectation(("spec", "seed_refs"), "array"),
            JsonExpectation(("spec", "unit_queries"), "array"),
            JsonExpectation(("segments",), "array"),
        ),
        stdout_contains=(f"session:{DEMO_CLAUDE_CODE_SESSION_ID}", '"unit_queries"'),
        required_affordance_ids=("continue",),
    ),
    ExecutableWorkflowGoldenPath(
        id="continue-json",
        workflow_id="find-then-continue",
        description="Continuation workflow emits the ContextImage seed refs and segment list.",
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
        workflow_id="find-then-analyze-facets",
        description="Query-selected analysis exposes scoped/global JSON facet buckets and honest family metadata.",
        command=("find", "pytest", "then", "analyze", "--facets", "--format", "json"),
        action_path=("analyze",),
        output_kind="json_object",
        json_expectations=(
            JsonExpectation(("scoped_to_query",), "boolean"),
            JsonExpectation(("scoped",), "object"),
            JsonExpectation(("global",), "object"),
            JsonExpectation(("scoped", "origins"), "object"),
            JsonExpectation(("family_status",), "object"),
            JsonExpectation(("deferred_families",), "object"),
        ),
        stdout_contains=('"claude-code-session"', '"codex-session"'),
        required_affordance_ids=("analyze",),
    ),
    ExecutableWorkflowGoldenPath(
        id="judge-review-json",
        workflow_id="candidate-assertion-review",
        description="Root judge lists durable candidate history with bounded evidence disclosure.",
        command=(
            "judge",
            "--target-ref",
            f"session:{DEMO_CLAUDE_CODE_SESSION_ID}",
            "--review",
            "--format",
            "json",
        ),
        action_path=("judge",),
        output_kind="json_object",
        json_expectations=(
            JsonExpectation(("mode",), "string"),
            JsonExpectation(("items",), "array"),
            JsonExpectation(("items", 0, "evidence_previews"), "array"),
            JsonExpectation(("total",), "integer"),
            JsonExpectation(("candidate_statuses",), "array"),
        ),
        stdout_contains=('"evidence_previews"', '"candidate"'),
        required_affordance_ids=("judge",),
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


def _validate_golden_paths() -> None:
    duplicate_golden_ids = len({entry.id for entry in EXECUTABLE_WORKFLOW_GOLDEN_PATHS}) != len(
        EXECUTABLE_WORKFLOW_GOLDEN_PATHS
    )
    if duplicate_golden_ids:
        raise ValueError("executable workflow golden paths contain duplicate ids")
    for golden in EXECUTABLE_WORKFLOW_GOLDEN_PATHS:
        if golden.output_kind == "human" and golden.json_expectations:
            raise ValueError(f"human golden path {golden.id!r} must not declare JSON expectations")
        if golden.output_kind != "human" and not golden.json_expectations:
            raise ValueError(f"JSON golden path {golden.id!r} must declare JSON expectations")


_validate_golden_paths()


__all__ = [
    "EXECUTABLE_WORKFLOW_GOLDEN_PATHS",
    "JsonExpectation",
    "JsonKind",
    "JsonPath",
    "JsonPathSegment",
    "OutputKind",
]
