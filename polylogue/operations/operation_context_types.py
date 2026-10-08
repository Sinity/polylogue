"""Small carrier for the context passed through resident operations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from polylogue.archive.query.execution_control import QueryExecutionContext
    from polylogue.operations.daemon_execution import OperationRuntime
    from polylogue.operations.daemon_reads import DaemonReadDependencies
    from polylogue.operations.mutation_transaction import MutationPrincipal


@dataclass(frozen=True, slots=True)
class OperationContext:
    """Authenticated archive and owner dependencies for one operation."""

    archive_root: Path
    principal: MutationPrincipal
    serving_identity: Literal["daemon"]
    runtime: OperationRuntime | None = None
    read_dependencies: DaemonReadDependencies | None = None
    read_control: QueryExecutionContext | None = None
