"""Semantic-operation parity between the CLI, MCP, and Python surfaces.

Parity is governed by *operation identity*, not by name-shaped reflection.
Each :class:`SemanticOperation` wraps a shared
:class:`~polylogue.declarations.models.DeclarationSpec` kernel and binds that
operation to the CLI, MCP, and Python surfaces -- either to a live target that
must resolve, or to an *intentional absence* that must carry a reason.

The MCP half of the matrix is derived from
``polylogue.mcp.declarations.registry``: adding an MCP tool adds a parity row
automatically, and its kernel record (identity, producer, role gate, discovery
text) is reused rather than transcribed.

Reflection is used for exactly one thing the operation IDs cannot supply:
proving the classification is *total*. Every public callable on the
:class:`~polylogue.api.Polylogue` facade must be either bound by an operation
or listed in :data:`EXCLUSIONS` with a category and a reason. That is what
makes the parity gate fail when the library surface moves.
"""

from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass
from typing import Final

from polylogue.declarations import DeclarationSpec
from polylogue.mcp.declarations.registry import MCP_TOOL_DECLARATIONS, declared_tool_names

REPAIR_COMMAND: Final[str] = "devtools verify api-parity --check"
FACADE_PATH: Final[str] = "polylogue/api/__init__.py"
FACADE_SYMBOL: Final[str] = "polylogue.api.Polylogue"

SURFACES: Final[tuple[str, ...]] = ("cli", "mcp", "python")

#: The root CLI is query-first: the query operation has no subcommand of its
#: own, it is the root query mode signalled by this keyword.
ROOT_QUERY_MODE_TOKEN: Final[str] = "find"


@dataclass(frozen=True, slots=True)
class SurfaceBinding:
    """One surface's realization of a semantic operation.

    Exactly one of ``target`` (a live binding that must resolve) and
    ``absence_reason`` (an intentional, authorized absence) is set.
    """

    surface: str
    target: str = ""
    absence_reason: str = ""

    @property
    def bound(self) -> bool:
        return bool(self.target)


@dataclass(frozen=True, slots=True)
class SemanticOperation:
    """One stable operation identity with its per-surface bindings."""

    operation_id: str
    summary: str
    kernel: DeclarationSpec
    bindings: tuple[SurfaceBinding, ...]

    def binding(self, surface: str) -> SurfaceBinding | None:
        for binding in self.bindings:
            if binding.surface == surface:
                return binding
        return None


@dataclass(frozen=True, slots=True)
class ExcludedCallable:
    """A public facade callable deliberately outside the operation matrix."""

    name: str
    category: str
    reason: str


@dataclass(frozen=True, slots=True)
class ParityFinding:
    """One actionable parity defect."""

    code: str
    subject: str
    message: str
    repair_command: str = REPAIR_COMMAND


# --------------------------------------------------------------------------
# Declared CLI bindings for the MCP-derived operations.
#
# The MCP kernel cannot know the CLI command path, so it is declared here and
# validated against the live Click tree; a renamed or deleted command fails.
# --------------------------------------------------------------------------

_CLI_BINDINGS: Final[dict[str, SurfaceBinding]] = {
    "query": SurfaceBinding("cli", target=ROOT_QUERY_MODE_TOKEN),
    "read": SurfaceBinding("cli", target="read"),
    "get": SurfaceBinding("cli", target="read"),
    "explain": SurfaceBinding(
        "cli",
        absence_reason=(
            "query-grammar explanation is served by the generated discovery surfaces (`polylogue manual`, "
            "docs/search.md); no dedicated CLI verb owns it"
        ),
    ),
    "context": SurfaceBinding("cli", target="context"),
    "status": SurfaceBinding("cli", target="status"),
    "write": SurfaceBinding("cli", target="mark"),
    "record_work_event": SurfaceBinding(
        "cli",
        absence_reason="agent work-event recording is an MCP/library transaction; no interactive CLI verb owns it",
    ),
    "emit_decision": SurfaceBinding(
        "cli",
        absence_reason="decision emission is an MCP/library transaction; the CLI exposes `note`, not decision records",
    ),
    "judge": SurfaceBinding("cli", target="judge"),
    "run": SurfaceBinding("cli", target="select"),
    "maintenance": SurfaceBinding(
        "cli",
        absence_reason=(
            "insight rebuild is owned by the daemon convergence route (`polylogued run`); the CLI exposes "
            "inspection (`ops insights status`), never a rebuild verb"
        ),
    ),
}

# Python bindings for MCP tools whose declared producer is not a facade
# callable (a transaction dispatcher token, or an owner below the facade).
_PYTHON_OVERRIDES: Final[dict[str, SurfaceBinding]] = {
    "status": SurfaceBinding("python", target=f"{FACADE_SYMBOL}.stats"),
    "write": SurfaceBinding(
        "python",
        absence_reason=(
            "`write` is a capability-gated transaction dispatcher; each of its operations is its own "
            "semantic operation with its own facade callable"
        ),
    ),
    "run": SurfaceBinding(
        "python",
        absence_reason=(
            "`run` executes a saved query or governed recipe ref; the facade exposes the underlying query "
            "operation instead of a ref-execution wrapper"
        ),
    ),
    "maintenance": SurfaceBinding("python", target=f"{FACADE_SYMBOL}.rebuild_insights"),
}


@dataclass(frozen=True, slots=True)
class _ExtraOperationRow:
    """A semantic operation with no MCP tool of its own."""

    operation_id: str
    summary: str
    python_symbol: str
    cli: SurfaceBinding
    mcp: SurfaceBinding
    kernel_owner: str = FACADE_PATH


_EXTRA_OPERATION_ROWS: Final[tuple[_ExtraOperationRow, ...]] = (
    _ExtraOperationRow(
        "api.embedding_preflight",
        "Report whether the embedding backend is usable before a semantic read.",
        f"{FACADE_SYMBOL}.embedding_preflight",
        SurfaceBinding("cli", target="ops embed preflight"),
        SurfaceBinding(
            "mcp",
            absence_reason="embedding readiness is reported inside the MCP `status` envelope, not as its own tool",
        ),
    ),
    _ExtraOperationRow(
        "api.embedding_status",
        "Report embedding coverage and staleness for the active archive.",
        f"{FACADE_SYMBOL}.embedding_status",
        SurfaceBinding("cli", target="ops embed status"),
        SurfaceBinding(
            "mcp",
            absence_reason="embedding coverage is reported inside the MCP `status` envelope, not as its own tool",
        ),
    ),
    _ExtraOperationRow(
        "api.import_annotation_batch",
        "Import a durable typed annotation batch under a declared schema version.",
        f"{FACADE_SYMBOL}.import_annotation_batch",
        SurfaceBinding("cli", target="annotations import"),
        SurfaceBinding("mcp", target="write"),
    ),
)


def _absence_reason_for_producer(producer: str) -> str:
    return f"declared producer {producer!r} is a transaction token, not a facade callable"


def _python_binding_for(declaration_name: str, producer: str) -> SurfaceBinding:
    override = _PYTHON_OVERRIDES.get(declaration_name)
    if override is not None:
        return override
    if producer.startswith(f"{FACADE_SYMBOL}."):
        return SurfaceBinding("python", target=producer)
    return SurfaceBinding("python", absence_reason=_absence_reason_for_producer(producer))


def _mcp_operations() -> tuple[SemanticOperation, ...]:
    operations: list[SemanticOperation] = []
    for declaration in MCP_TOOL_DECLARATIONS:
        kernel = declaration.kernel
        cli = _CLI_BINDINGS.get(declaration.name)
        if cli is None:
            cli = SurfaceBinding("cli", absence_reason="")  # unbound: reported as a finding
        operations.append(
            SemanticOperation(
                operation_id=kernel.declaration_id,
                summary=kernel.discovery_text,
                kernel=kernel,
                bindings=(
                    cli,
                    SurfaceBinding("mcp", target=declaration.name),
                    _python_binding_for(declaration.name, kernel.producer),
                ),
            )
        )
    return tuple(operations)


def _extra_operations() -> tuple[SemanticOperation, ...]:
    from polylogue.declarations import CompatibilityKey

    operations: list[SemanticOperation] = []
    for row in _EXTRA_OPERATION_ROWS:
        kernel = DeclarationSpec(
            declaration_id=row.operation_id,
            family_id="api.semantic-operation",
            public_name=row.python_symbol.rsplit(".", 1)[-1],
            owner_path=row.kernel_owner,
            compatibility=CompatibilityKey(
                identity="api-semantic-operation",
                lifecycle="public-facade-callable",
                authority="library-caller",
                access_result_shape="typed-payload",
                durability="owner-controlled",
            ),
            producer=row.python_symbol,
            role_gate="library",
            schema_ref=f"{row.python_symbol}:inspect.signature",
            discovery_text=row.summary,
            repair_command=REPAIR_COMMAND,
            handlers=(),
            outputs=(),
            examples=(),
            completeness_edges=(),
        )
        operations.append(
            SemanticOperation(
                operation_id=row.operation_id,
                summary=row.summary,
                kernel=kernel,
                bindings=(row.cli, row.mcp, SurfaceBinding("python", target=row.python_symbol)),
            )
        )
    return tuple(operations)


def semantic_operations() -> tuple[SemanticOperation, ...]:
    """Return every semantic operation in stable operation-id order."""

    return tuple(sorted((*_mcp_operations(), *_extra_operations()), key=lambda item: item.operation_id))


# --------------------------------------------------------------------------
# Total classification of the public facade.
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _ExclusionCategory:
    category: str
    reason: str
    members: tuple[str, ...]


_EXCLUSION_CATEGORIES: Final[tuple[_ExclusionCategory, ...]] = (
    _ExclusionCategory(
        "lifecycle",
        "construction, teardown, and lazy builders; they carry no cross-surface operation identity",
        ("open", "close", "filter", "iter_messages"),
    ),
    _ExclusionCategory(
        "ingest",
        "ingest stages owned by `polylogued run`; the CLI/MCP parity target is the daemon, not the facade call",
        (
            "parse_file",
            "parse_sources",
            "explain_import",
            "compact_lineage",
            "reconcile_codex_spawn_edges",
            "reconcile_hermes_session_lifecycle",
            "regenerate_private_fable_packet",
        ),
    ),
    _ExclusionCategory(
        "insight-projection",
        "descriptor-driven insight registry (polylogue/analysis/registry.py) owns cross-surface parity for these",
        (
            "export_insight_bundle",
            "find_stuck_session_latency_profile_insights",
            "get_session_insight_status",
            "get_session_latency_profile_insight",
            "get_session_profile_insight",
            "get_session_profile_record",
            "get_thread_insight",
            "insight_readiness_report",
            "insight_rigor_audit",
            "list_archive_coverage_insights",
            "list_archive_debt_insights",
            "list_cost_rollup_insights",
            "list_session_cost_insights",
            "list_session_latency_profile_insights",
            "list_session_profile_insights",
            "list_session_tag_rollup_insights",
            "list_thread_insights",
            "list_tool_episode_insights",
            "list_tool_usage_insights",
            "list_usage_timeline_insights",
        ),
    ),
    _ExclusionCategory(
        "read-detail",
        "object-level reads reached through the `read`/`get` operations' refs rather than their own operation id",
        (
            "archive_get_session",
            "bulk_get_messages",
            "get_actions_batch",
            "get_ancestors",
            "get_descendants",
            "get_file_edits",
            "get_logical_session",
            "get_messages_paginated",
            "get_raw_artifacts_for_session",
            "get_session",
            "get_session_events",
            "get_session_stats",
            "get_session_summary",
            "get_session_topology",
            "get_session_tree",
            "get_sessions",
            "get_siblings",
            "get_thread",
            "get_web_content_constructs",
            "read_transcript_window",
            "list_sessions",
            "list_summaries",
            "search",
            "search_envelope",
            "search_session_hits",
        ),
    ),
    _ExclusionCategory(
        "aggregate",
        "aggregates reached through the `query` operation's aggregate result semantics",
        (
            "aggregate_sessions",
            "archive_count_sessions",
            "count_sessions",
            "facets",
            "get_stats_by",
            "query_completions",
            "query_sessions",
            "storage_stats",
            "get_index_status",
            "health_check",
            "hermes_integration_health",
            "archive_debt",
            "pathology_report",
            "origin_usage_report",
            "cost_outlook",
            "session_usage_reconciliation",
            "tool_call_latency_distribution",
            "workflow_shape_distribution",
            "list_command_shape_usage",
            "export_otel",
            "diagnose_query_miss",
            "list_read_view_profiles",
        ),
    ),
    _ExclusionCategory(
        "user-state",
        "durable user.db overlays whose parity target is the annotation/marker family, not a single operation",
        (
            "add_mark",
            "add_tag",
            "bulk_tag_sessions",
            "clear_corrections",
            "delete_annotation",
            "delete_correction",
            "delete_metadata",
            "delete_recall_pack",
            "delete_session",
            "delete_session_safe",
            "delete_view",
            "delete_workspace",
            "create_recall_pack",
            "get_annotation",
            "get_metadata",
            "get_recall_pack",
            "get_setting",
            "get_view",
            "get_workspace",
            "join_typed_annotations",
            "list_annotations",
            "list_corrections",
            "list_marks",
            "list_recall_packs",
            "list_settings",
            "list_tags",
            "list_views",
            "list_workspaces",
            "record_correction",
            "remove_mark",
            "remove_tag",
            "save_annotation",
            "save_view",
            "save_workspace",
            "set_metadata",
            "update_metadata",
        ),
    ),
    _ExclusionCategory(
        "assertion-review",
        "assertion candidate review helpers behind the `judge` operation's queue",
        (
            "assertion_candidate_queue_health",
            "capture_assertion_candidate",
            "judge_assertion_candidate",
            "list_assertion_candidate_reviews",
            "list_assertion_candidates",
            "list_assertion_claim_payloads",
            "list_assertion_claims",
            "list_comparative_judgments",
            "record_comparative_judgment",
        ),
    ),
    _ExclusionCategory(
        "agent-coordination",
        "agent coordination reads/writes surfaced through `polylogue agents`, which owns their own contract",
        (
            "correlate_claude_agent_dispatches",
            "correlate_hermes_context_deliveries",
            "get_agent_policies",
            "get_context_delivery",
            "get_effective_context",
            "get_hook_event_summary_for_session",
            "get_session_orchestration",
            "list_blackboard_notes",
            "list_context_deliveries",
            "list_context_injection_ledger",
            "post_blackboard_note",
            "record_context_delivery",
            "record_manual_continuation",
            "compile_and_record_context",
            "compile_context",
        ),
    ),
    _ExclusionCategory(
        "context-pack",
        "bounded packet builders composed from the `context` operation rather than carrying their own identity",
        (
            "context_preamble_payload",
            "portfolio_bundle",
            "postmortem_bundle",
            "resume_brief",
            "topic_pack",
            "session_correlation_payload",
            "correlate_sessions",
            "compare_sessions",
            "find_abandoned_sessions",
            "find_resume_candidates",
            "find_similar_sessions_by_metadata",
            "list_sessions_for_spec",
            "neighbor_candidate_payloads",
            "neighbor_candidates",
            "search_similar_sessions",
        ),
    ),
)

EXCLUSIONS: Final[tuple[ExcludedCallable, ...]] = tuple(
    sorted(
        (
            ExcludedCallable(name=name, category=category.category, reason=category.reason)
            for category in _EXCLUSION_CATEGORIES
            for name in category.members
        ),
        key=lambda item: item.name,
    )
)


def public_facade_callables() -> tuple[str, ...]:
    """Return every public callable attribute of the live facade class."""

    facade = importlib.import_module("polylogue.api").Polylogue
    return tuple(
        sorted(
            name
            for name, value in inspect.getmembers(facade)
            if not name.startswith("_") and callable(value) and not isinstance(value, property)
        )
    )


def operation_python_names() -> tuple[str, ...]:
    """Return facade attribute names bound by a semantic operation."""

    names: set[str] = set()
    for operation in semantic_operations():
        binding = operation.binding("python")
        if binding is not None and binding.bound and binding.target.startswith(f"{FACADE_SYMBOL}."):
            names.add(binding.target.rsplit(".", 1)[-1])
    return tuple(sorted(names))


def is_async_operation(name: str) -> bool:
    """Return whether the live facade callable ``name`` is a coroutine function."""

    facade = importlib.import_module("polylogue.api").Polylogue
    return inspect.iscoroutinefunction(getattr(facade, name))


def _resolve_symbol(dotted: str) -> object:
    module_name, _, attribute_path = dotted.partition(":")
    if not attribute_path:
        parts = dotted.split(".")
        for split in range(len(parts) - 1, 0, -1):
            module_name = ".".join(parts[:split])
            try:
                module = importlib.import_module(module_name)
            except ImportError:
                continue
            target: object = module
            for part in parts[split:]:
                target = getattr(target, part)
            return target
        raise ImportError(f"cannot resolve {dotted!r}")
    module = importlib.import_module(module_name)
    target = module
    for part in attribute_path.split("."):
        target = getattr(target, part)
    return target


def _root_query_mode_available() -> bool:
    """Return whether the live root parser still treats the token as query intent."""

    from polylogue.cli.click_app import cli as cli_root
    from polylogue.cli.query_group import _split_query_mode_args

    _, _terms, _has_subcommand, explicit_query = _split_query_mode_args(
        cli_root, [ROOT_QUERY_MODE_TOKEN, "messages where text:needle"]
    )
    return bool(explicit_query)


def _cli_command_paths() -> frozenset[str]:
    from polylogue.cli.click_app import cli as cli_root
    from polylogue.cli.command_inventory import iter_command_paths

    return frozenset(item.display_name for item in iter_command_paths(cli_root, include_root=False))


def validate_parity() -> tuple[ParityFinding, ...]:
    """Return every parity defect, deterministically ordered.

    Anti-vacuity: adding, renaming, or removing a public facade callable
    without classifying it, declaring a CLI binding for a command that does not
    exist, binding an MCP surface to an undeclared tool name, or leaving an
    absence without a reason each produce a finding here.
    """

    findings: list[ParityFinding] = []
    operations = semantic_operations()
    declared_tools = declared_tool_names()
    cli_paths = _cli_command_paths()

    seen_ids: set[str] = set()
    for operation in operations:
        if operation.operation_id in seen_ids:
            findings.append(
                ParityFinding(
                    "duplicate_operation_id",
                    operation.operation_id,
                    f"operation id {operation.operation_id!r} is declared more than once",
                )
            )
        seen_ids.add(operation.operation_id)
        surfaces = {binding.surface for binding in operation.bindings}
        for surface in SURFACES:
            if surface not in surfaces:
                findings.append(
                    ParityFinding(
                        "missing_surface_row",
                        f"{operation.operation_id}:{surface}",
                        f"{operation.operation_id} declares no {surface} binding",
                    )
                )
        for binding in operation.bindings:
            if not binding.bound:
                if not binding.absence_reason:
                    findings.append(
                        ParityFinding(
                            "unjustified_absence",
                            f"{operation.operation_id}:{binding.surface}",
                            f"{operation.operation_id} has no {binding.surface} binding and no absence reason",
                        )
                    )
                continue
            if binding.surface == "mcp" and binding.target not in declared_tools:
                findings.append(
                    ParityFinding(
                        "unknown_mcp_tool",
                        f"{operation.operation_id}:mcp",
                        f"MCP binding {binding.target!r} is not a declared MCP tool",
                    )
                )
            elif binding.surface == "cli" and binding.target == ROOT_QUERY_MODE_TOKEN:
                if not _root_query_mode_available():
                    findings.append(
                        ParityFinding(
                            "unknown_cli_command",
                            f"{operation.operation_id}:cli",
                            f"the root CLI no longer signals query intent on {ROOT_QUERY_MODE_TOKEN!r}",
                        )
                    )
            elif binding.surface == "cli" and binding.target not in cli_paths:
                findings.append(
                    ParityFinding(
                        "unknown_cli_command",
                        f"{operation.operation_id}:cli",
                        f"CLI binding {binding.target!r} is not a live command path",
                    )
                )
            elif binding.surface == "python":
                try:
                    _resolve_symbol(binding.target)
                except (ImportError, AttributeError) as exc:
                    findings.append(
                        ParityFinding(
                            "unresolved_python_binding",
                            f"{operation.operation_id}:python",
                            f"Python binding {binding.target!r} does not resolve ({exc})",
                        )
                    )

    live = set(public_facade_callables())
    bound = set(operation_python_names())
    excluded = [item.name for item in EXCLUSIONS]
    excluded_set = set(excluded)
    if len(excluded) != len(excluded_set):
        duplicates = sorted({name for name in excluded if excluded.count(name) > 1})
        for name in duplicates:
            findings.append(
                ParityFinding("duplicate_classification", name, f"{name!r} appears in more than one exclusion category")
            )
    for name in sorted(bound & excluded_set):
        findings.append(
            ParityFinding(
                "conflicting_classification",
                name,
                f"{name!r} is both bound by a semantic operation and listed as an exclusion",
            )
        )
    for name in sorted(live - bound - excluded_set):
        findings.append(
            ParityFinding(
                "unclassified_callable",
                name,
                (
                    f"public facade callable {name!r} is neither bound by a semantic operation nor listed "
                    f"as an explicit exclusion in {FACADE_PATH.replace('__init__.py', 'parity.py')}"
                ),
            )
        )
    for name in sorted((bound | excluded_set) - live):
        findings.append(
            ParityFinding(
                "stale_classification",
                name,
                f"{name!r} is classified but no longer exists on the live facade",
            )
        )
    return tuple(sorted(findings, key=lambda item: (item.code, item.subject)))


__all__ = [
    "EXCLUSIONS",
    "FACADE_SYMBOL",
    "REPAIR_COMMAND",
    "ROOT_QUERY_MODE_TOKEN",
    "SURFACES",
    "ExcludedCallable",
    "ParityFinding",
    "SemanticOperation",
    "SurfaceBinding",
    "is_async_operation",
    "operation_python_names",
    "public_facade_callables",
    "semantic_operations",
    "validate_parity",
]
