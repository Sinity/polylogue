"""Which CLI code lowers and renders each declared daemon operation.

Declaring an operation is not the same as serving a surface with it.  This
registry is the join between the operation declarations and the CLI code on
either side of the transport, so a declaration that no CLI route can build a
request for or render a result from is visible as such instead of looking like
progress.  Every :data:`~polylogue.operations.daemon_protocol.DAEMON_OPERATION_SPECS`
entry belongs to exactly one of three tables:

``CLI_OPERATION_BINDINGS``
    The CLI can lower a request for it and render its result today.
``CLI_PENDING_ADOPTION``
    Declared and directly executable, with the CLI route that will call it
    named.  Each entry states which migration step adopts it.  The table is
    empty: every declared operation the CLI serves now has a lowering and a
    renderer, and the mechanism is retained so that a *new* declaration must
    be classified by an explicit decision rather than by silence.
``CLI_EXTERNAL_OPERATIONS``
    Deliberately not a CLI concern (transport-owned control verbs, or
    operations whose only consumers are other surfaces).

References are dotted ``module:attribute`` strings rather than imports so this
module stays import-light on the CLI's cold path; the registry test resolves
every one of them.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from importlib import import_module
from typing import Any


class OperationBindingError(LookupError):
    """A declared operation has no classification, or an unresolvable one."""


@dataclass(frozen=True, slots=True)
class CliOperationBinding:
    """One operation joined to the CLI code that builds and renders it.

    ``lowering`` and ``renderers`` may name the same callable while a CLI route
    still does both in one pass; the seams separate at S3/S4.
    """

    lowering: str
    renderers: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.renderers:
            raise OperationBindingError("an operation binding needs at least one renderer")


_ARCHIVE_QUERY = "polylogue.cli.archive_query"
_LOWERING = "polylogue.cli.lowering"
_RENDER_ROWS = "polylogue.cli.render.rows"

CLI_OPERATION_BINDINGS: Mapping[str, CliOperationBinding] = {
    "cli.query": CliOperationBinding(
        lowering=f"{_LOWERING}:lower_cli_query",
        renderers=(
            f"{_RENDER_ROWS}:emit_session_list_page",
            f"{_RENDER_ROWS}:emit_session_search_page",
        ),
    ),
    "query.units": CliOperationBinding(
        lowering=f"{_LOWERING}:lower_query_units",
        renderers=(f"{_RENDER_ROWS}:emit_rows",),
    ),
    "query.aggregate": CliOperationBinding(
        lowering=f"{_LOWERING}:lower_query_aggregate",
        renderers=(f"{_ARCHIVE_QUERY}:_emit_aggregate_result",),
    ),
    "session.read": CliOperationBinding(
        lowering=f"{_LOWERING}:lower_session_read",
        renderers=(f"{_ARCHIVE_QUERY}:_emit_session_result", f"{_ARCHIVE_QUERY}:_emit_stream"),
    ),
    "session.reference": CliOperationBinding(
        lowering=f"{_LOWERING}:lower_session_reference",
        renderers=(f"{_ARCHIVE_QUERY}:_emit_reference_query",),
    ),
    "status": CliOperationBinding(
        lowering="polylogue.cli.commands.status:_status_operation_result",
        renderers=("polylogue.cli.commands.status:_show_daemon_status",),
    ),
    "facets": CliOperationBinding(
        lowering="polylogue.cli.commands.facets:_fetch_daemon_facets",
        renderers=("polylogue.cli.commands.facets:facets_command",),
    ),
    "mutation.session.tag": CliOperationBinding(
        lowering=f"{_ARCHIVE_QUERY}:_emit_user_mutations",
        renderers=(f"{_ARCHIVE_QUERY}:_emit_user_mutations",),
    ),
    "mutation.session.metadata": CliOperationBinding(
        lowering=f"{_ARCHIVE_QUERY}:_emit_user_mutations",
        renderers=(f"{_ARCHIVE_QUERY}:_emit_user_mutations",),
    ),
    "mutation.session.mark": CliOperationBinding(
        lowering="polylogue.cli.query_verbs:mark_verb",
        renderers=("polylogue.cli.query_verbs:mark_verb",),
    ),
    "mutation.annotation.save": CliOperationBinding(
        lowering="polylogue.cli.query_verbs:mark_verb",
        renderers=("polylogue.cli.query_verbs:mark_verb",),
    ),
    "mutation.assertion.candidate.capture": CliOperationBinding(
        lowering="polylogue.cli.commands.note:note_command",
        renderers=("polylogue.cli.commands.note:note_command",),
    ),
    "mutation.user.setting.set": CliOperationBinding(
        lowering="polylogue.cli.commands.setting:setting_set_command",
        renderers=("polylogue.cli.commands.setting:setting_set_command",),
    ),
    "mutation.annotation.import_batch": CliOperationBinding(
        lowering="polylogue.cli.commands.annotations:import_annotations_command",
        renderers=("polylogue.cli.commands.annotations:import_annotations_command",),
    ),
    "mutation.judgment.record": CliOperationBinding(
        lowering="polylogue.cli.commands.compare:compare_command",
        renderers=(
            "polylogue.cli.commands.compare:compare_command",
            "polylogue.cli.commands.judge:_judge",
        ),
    ),
    "mutation.session.delete.preview": CliOperationBinding(
        lowering=f"{_ARCHIVE_QUERY}:_emit_delete",
        renderers=(f"{_ARCHIVE_QUERY}:_emit_delete",),
    ),
    "mutation.session.delete.authorize": CliOperationBinding(
        lowering=f"{_ARCHIVE_QUERY}:_emit_delete",
        renderers=(f"{_ARCHIVE_QUERY}:_emit_delete",),
    ),
    "mutation.session.delete.execute": CliOperationBinding(
        lowering=f"{_ARCHIVE_QUERY}:_emit_delete",
        renderers=(f"{_ARCHIVE_QUERY}:_emit_delete",),
    ),
    "mutation.session.delete.cancel": CliOperationBinding(
        lowering=f"{_ARCHIVE_QUERY}:_cancel_delete_preview",
        renderers=(f"{_ARCHIVE_QUERY}:_cancel_delete_preview",),
    ),
    "mutation.session.lifecycle-request": CliOperationBinding(
        lowering="polylogue.cli.commands.excise:excise_command",
        renderers=("polylogue.cli.commands.excise:excise_command",),
    ),
    "mutation.session.excision": CliOperationBinding(
        lowering="polylogue.cli.commands.excise:excise_command",
        renderers=("polylogue.cli.commands.excise:excise_command",),
    ),
    "mutation.identity-reset": CliOperationBinding(
        lowering="polylogue.cli.commands.reset:reset_command",
        renderers=("polylogue.cli.commands.reset:reset_command",),
    ),
    "maintenance.reset": CliOperationBinding(
        lowering="polylogue.cli.commands.reset:reset_command",
        renderers=("polylogue.cli.commands.reset:reset_command",),
    ),
    "maintenance.blob-gc.recover": CliOperationBinding(
        lowering="polylogue.cli.commands.maintenance._blob_gc:_submit",
        renderers=("polylogue.cli.commands.maintenance._blob_gc:blob_gc_command",),
    ),
    "maintenance.blob-publications.abandon": CliOperationBinding(
        lowering="polylogue.cli.commands.maintenance._blob_publications:_submit_abandonment",
        renderers=("polylogue.cli.commands.maintenance._blob_publications:blob_publications_command",),
    ),
    "maintenance.blob-refs.replace-from-source": CliOperationBinding(
        lowering="polylogue.cli.commands.maintenance._blob_integrity:_submit_replace_from_source",
        renderers=(
            "polylogue.cli.commands.maintenance._blob_integrity:_render_blob_reference_replace_from_source_plain",
        ),
    ),
    "maintenance.blob-refs.prune-orphans": CliOperationBinding(
        lowering="polylogue.cli.commands.maintenance._blob_integrity:_submit_prune_orphans",
        renderers=("polylogue.cli.commands.maintenance._blob_integrity:_render_blob_reference_prune_orphans_plain",),
    ),
    "ingest": CliOperationBinding(
        lowering="polylogue.cli.commands.import_command:_submit_ingest",
        renderers=("polylogue.cli.commands.import_command:import_command",),
    ),
    "maintenance.demo.augment": CliOperationBinding(
        lowering="polylogue.cli.commands.import_command:_request_demo_augmentation",
        renderers=("polylogue.cli.commands.import_command:import_command",),
    ),
    "mutation.raw-authority-blocker.resolve": CliOperationBinding(
        lowering="polylogue.cli.commands.maintenance._raw_identity:_submit",
        renderers=("polylogue.cli.commands.maintenance._raw_identity:raw_authority_blocker_resolve_command",),
    ),
    # Archive-backed shell completion lowers through Seam A and renders the
    # typed result; ``daemon_only`` dispatch is what keeps a TAB press from
    # opening the archive at all, which
    # ``tests/unit/cli/test_completion_daemon_boundary.py`` proves under a
    # ``sqlite3.connect`` audit hook.
    "completion": CliOperationBinding(
        lowering=f"{_LOWERING}:lower_completion",
        renderers=("polylogue.cli.shell_completion_values:render_completion_values",),
    ),
}

CLI_PENDING_ADOPTION: Mapping[str, str] = {}

CLI_EXTERNAL_OPERATIONS: Mapping[str, str] = {
    "operation.await": "transport-owned: DaemonClient.operation_to_completion drives it, not a CLI route",
    "operation.cancel": "transport-owned: DaemonClient cancels a submitted operation, not a CLI route",
    "operation.status": "no consumer on any surface; retained for receipt recovery tooling",
    "maintenance.insights.rebuild": "daemon-internal derivation; no CLI verb requests it",
    "maintenance.embeddings.backfill": "daemon-owned embedding convergence; CLI submits it directly",
}


def binding_for(operation: str) -> CliOperationBinding:
    """Return the CLI binding for ``operation`` or say why there is none."""

    binding = CLI_OPERATION_BINDINGS.get(operation)
    if binding is None:
        reason = CLI_PENDING_ADOPTION.get(operation) or CLI_EXTERNAL_OPERATIONS.get(operation)
        raise OperationBindingError(
            f"{operation} has no CLI binding: {reason}" if reason else f"{operation} is not classified"
        )
    return binding


def resolve_reference(reference: str) -> Any:
    """Import one ``module:attribute`` registry reference."""

    module_name, _, attribute = reference.partition(":")
    if not module_name or not attribute:
        raise OperationBindingError(f"malformed binding reference: {reference!r}")
    try:
        return getattr(import_module(module_name), attribute)
    except (ImportError, AttributeError) as exc:
        raise OperationBindingError(f"binding reference does not resolve: {reference!r}") from exc


def unclassified_operations() -> tuple[str, ...]:
    """Return declared operations that appear in none of the three tables."""

    from polylogue.operations.daemon_protocol import DAEMON_OPERATION_SPECS

    classified = {*CLI_OPERATION_BINDINGS, *CLI_PENDING_ADOPTION, *CLI_EXTERNAL_OPERATIONS}
    return tuple(spec.name for spec in DAEMON_OPERATION_SPECS if spec.name not in classified)


__all__ = [
    "CLI_EXTERNAL_OPERATIONS",
    "CLI_OPERATION_BINDINGS",
    "CLI_PENDING_ADOPTION",
    "CliOperationBinding",
    "OperationBindingError",
    "binding_for",
    "resolve_reference",
    "unclassified_operations",
]
