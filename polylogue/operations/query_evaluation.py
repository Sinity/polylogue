"""Own admitted readers and durable User location for query evaluation."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from polylogue.storage.archive_identity import archive_root_for_index_path

if TYPE_CHECKING:
    from polylogue.archive.query.evaluator import CanonicalPlanEvaluator


def query_user_path(index_path: Path) -> Path:
    """Select the durable User role belonging to the named Index."""
    return archive_root_for_index_path(index_path) / "user.db"


@contextmanager
def open_query_evaluator(index_path: Path) -> Iterator[CanonicalPlanEvaluator]:
    """Keep membership and receipt readers alive for the caller's batch."""
    from polylogue.archive.query.production_evaluator import ArchiveCanonicalPlanEvaluator
    from polylogue.operations.operation_context import open_operation_read

    with open_operation_read(archive_root_for_index_path(index_path)) as pinned:
        yield ArchiveCanonicalPlanEvaluator(pinned.archive)
