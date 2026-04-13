"""Infer runtime target metadata from structured execution workloads."""

from __future__ import annotations

from polylogue.operations import build_runtime_operation_catalog
from polylogue.products.registry import PRODUCT_REGISTRY

from .execution import ExecutionKind, ExecutionSpec
from .metadata import ScenarioMetadata

_KNOWN_POLYLOGUE_SUBCOMMANDS = frozenset(
    {
        "audit",
        "doctor",
        "embed",
        "products",
        "render",
        "run",
        "schema",
        "site",
        "tags",
    }
)
_PRODUCT_OPERATION_BY_METHOD = {
    "list_session_profile_products": "query-session-profiles",
    "list_session_enrichment_products": "query-session-enrichments",
    "list_session_work_event_products": "query-session-work-events",
    "list_session_phase_products": "query-session-phases",
    "list_work_thread_products": "query-work-threads",
    "list_session_tag_rollup_products": "query-session-tag-rollups",
    "list_day_session_summary_products": "query-day-session-summaries",
    "list_week_session_summary_products": "query-week-session-summaries",
    "list_provider_analytics_products": "query-provider-analytics",
    "list_archive_debt_products": "query-archive-debt",
}
_PRODUCT_SUBCOMMAND_OPERATION_NAMES = {
    "status": "query-session-product-status",
    "debt": "query-archive-debt",
}
_DOCTOR_TARGET_HEALTH_OPERATIONS = {
    "action_event_read_model": "project-action-event-health",
    "action_events": "project-action-event-health",
    "session_products": "project-session-product-health",
}
_DOCTOR_TARGET_REPAIR_OPERATIONS = {
    "action_event_read_model": "materialize-action-events",
    "action_events": "materialize-action-events",
    "session_products": "materialize-session-products",
}


def _unique(items: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    merged: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        merged.append(item)
    return tuple(merged)


def _metadata_for_operations(*operation_names: str) -> ScenarioMetadata:
    catalog = build_runtime_operation_catalog()
    target_names = _unique(tuple(name for name in operation_names if name))
    operations = catalog.resolve(target_names)
    return ScenarioMetadata(
        path_targets=_unique(tuple(path for operation in operations for path in operation.path_targets)),
        artifact_targets=_unique(
            tuple(
                artifact
                for operation in operations
                for artifact in (*operation.consumes, *operation.produces)
            )
        ),
        operation_targets=target_names,
    )


def _find_flag_value(argv: tuple[str, ...], flag: str) -> str | None:
    for index, item in enumerate(argv[:-1]):
        if item == flag:
            return argv[index + 1]
    return None


def _find_repeated_flag_values(argv: tuple[str, ...], flag: str) -> tuple[str, ...]:
    values: list[str] = []
    for index, item in enumerate(argv[:-1]):
        if item == flag:
            values.append(argv[index + 1])
    return tuple(values)


def _first_non_option(argv: tuple[str, ...]) -> str | None:
    for item in argv:
        if not item.startswith("-"):
            return item
    return None


def _infer_polylogue_product_metadata(argv: tuple[str, ...]) -> ScenarioMetadata:
    try:
        products_index = argv.index("products")
    except ValueError:
        return ScenarioMetadata()
    if products_index + 1 >= len(argv):
        return ScenarioMetadata()
    subcommand = argv[products_index + 1]
    direct_operation = _PRODUCT_SUBCOMMAND_OPERATION_NAMES.get(subcommand)
    if direct_operation:
        return _metadata_for_operations(direct_operation)
    operation_name = next(
        (
            _PRODUCT_OPERATION_BY_METHOD[product.operations_method]
            for product in PRODUCT_REGISTRY.values()
            if product.resolved_cli_command_name == subcommand and product.operations_method in _PRODUCT_OPERATION_BY_METHOD
        ),
        "",
    )
    return _metadata_for_operations(operation_name) if operation_name else ScenarioMetadata()


def _infer_polylogue_schema_metadata(argv: tuple[str, ...]) -> ScenarioMetadata:
    try:
        schema_index = argv.index("schema")
    except ValueError:
        return ScenarioMetadata()
    if schema_index + 1 >= len(argv):
        return ScenarioMetadata()
    subcommand = argv[schema_index + 1]
    if subcommand == "list":
        return _metadata_for_operations("query-schema-catalog")
    if subcommand == "explain":
        return _metadata_for_operations("query-schema-explanations")
    return ScenarioMetadata()


def _infer_polylogue_doctor_metadata(argv: tuple[str, ...]) -> ScenarioMetadata:
    operations: list[str] = []
    if "--json" in argv:
        operations.append("cli.json-contract")
    targets = tuple(target for target in _find_repeated_flag_values(argv, "--target") if target)
    if targets:
        if "--repair" in argv and "--preview" not in argv:
            for target in targets:
                repair_operation = _DOCTOR_TARGET_REPAIR_OPERATIONS.get(target)
                if repair_operation:
                    operations.append(repair_operation)
        for target in targets:
            health_operation = _DOCTOR_TARGET_HEALTH_OPERATIONS.get(target)
            if health_operation:
                operations.append(health_operation)
    else:
        operations.append("project-archive-health")
    return _metadata_for_operations(*operations)


def _infer_polylogue_run_metadata(argv: tuple[str, ...]) -> ScenarioMetadata:
    try:
        run_index = argv.index("run")
    except ValueError:
        return ScenarioMetadata()
    if run_index + 1 >= len(argv):
        return ScenarioMetadata()
    stage = argv[run_index + 1]
    if stage == "render":
        return _metadata_for_operations("render-conversations")
    if stage == "site":
        return _metadata_for_operations("publish-site")
    return ScenarioMetadata()


def _infer_polylogue_query_metadata(argv: tuple[str, ...]) -> ScenarioMetadata:
    first_token = _first_non_option(argv)
    if first_token in _KNOWN_POLYLOGUE_SUBCOMMANDS:
        return ScenarioMetadata()
    if not argv:
        return ScenarioMetadata()
    return _metadata_for_operations("query-conversations")


def _infer_polylogue_metadata(argv: tuple[str, ...]) -> ScenarioMetadata:
    if "schema" in argv:
        return _infer_polylogue_schema_metadata(argv)
    if "products" in argv:
        return _infer_polylogue_product_metadata(argv)
    if "doctor" in argv:
        return _infer_polylogue_doctor_metadata(argv)
    if "run" in argv:
        return _infer_polylogue_run_metadata(argv)
    return _infer_polylogue_query_metadata(argv)


def _infer_devtools_metadata(execution: ExecutionSpec) -> ScenarioMetadata:
    if execution.subcommand == "pipeline-probe":
        stage = _find_flag_value(execution.argv, "--stage") or "all"
        if stage == "parse":
            return _metadata_for_operations(
                "plan-validation-backlog",
                "plan-parse-backlog",
                "ingest-archive-runtime",
            )
    return ScenarioMetadata()


def infer_metadata_from_execution(execution: ExecutionSpec | None) -> ScenarioMetadata:
    if execution is None:
        return ScenarioMetadata()
    if execution.kind is ExecutionKind.POLYLOGUE:
        return _infer_polylogue_metadata(execution.argv)
    if execution.kind is ExecutionKind.DEVTOOLS:
        return _infer_devtools_metadata(execution)
    if execution.kind is ExecutionKind.MEMORY_BUDGET:
        return infer_metadata_from_execution(execution.wrapped)
    return ScenarioMetadata()


__all__ = ["infer_metadata_from_execution"]
