"""Matched-session tag and metadata writes behind the mutation authority.

``user.db`` is durable and irreplaceable, so a matched-page tag or metadata
write drives the same PREPARE/AUTHORIZE/EXECUTE cycle every other mutation
uses.  The functions here own that cycle for the ``mutation.session.tag`` and
``mutation.session.metadata`` operations; the daemon handler is their only
production caller and the operation envelope their only transport.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from polylogue.operations.bindings import runtime_operation_binding
from polylogue.operations.mutation_transaction import MutationPrincipal, OperationExecutor
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

TAG_CAPABILITY = "archive.bulk_tag_sessions"
METADATA_CAPABILITY = "archive.set_metadata"


def _execute(
    archive_root: Path,
    actuator: object,
    build_args: Callable[[ArchiveStore], Any],
    principal: MutationPrincipal,
) -> int:
    """Return the session/value pairs the executed mutation wrote."""

    binding = runtime_operation_binding(cast(Any, actuator))
    with ArchiveStore.open_existing(archive_root, read_only=False) as writable:
        args = build_args(writable)
        executor = OperationExecutor.for_archive_root(archive_root)
        preview = executor.prepare_bound_for_archive(binding, args, principal, archive_root=archive_root)
        authorization = executor.authorize_bound(binding, preview, principal)
        receipt = executor.execute_bound(binding, preview, authorization, args)
    return int(cast(Any, receipt.domain_receipt)["assertion_count"])


def apply_session_tags(
    archive_root: Path,
    session_ids: tuple[str, ...],
    tags: tuple[str, ...],
    principal: MutationPrincipal,
) -> int:
    from polylogue.operations.mutation_actuators import BulkTagActuator, BulkTagArgs

    return _execute(
        archive_root,
        BulkTagActuator(),
        lambda writable: BulkTagArgs(archive=writable, session_ids=session_ids, tags=tags),
        principal,
    )


def apply_session_metadata(
    archive_root: Path,
    session_ids: tuple[str, ...],
    pairs: tuple[tuple[str, str], ...],
    principal: MutationPrincipal,
) -> int:
    from polylogue.operations.mutation_actuators import BulkMetadataSetActuator, BulkMetadataSetArgs

    return _execute(
        archive_root,
        BulkMetadataSetActuator(),
        lambda writable: BulkMetadataSetArgs(archive=writable, session_ids=session_ids, pairs=pairs),
        principal,
    )


__all__ = [
    "METADATA_CAPABILITY",
    "TAG_CAPABILITY",
    "apply_session_metadata",
    "apply_session_tags",
]
