"""Explicit execution dependencies and the reader that owns result authority."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, ExitStack, closing, contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from polylogue.archive.query.execution_control import InterruptibleSQLiteRead, QueryExecutionContext
from polylogue.archive.query.search_contract import LaneFailure
from polylogue.core.errors import DatabaseError
from polylogue.operations.mutation_transaction import MutationPrincipal
from polylogue.storage.archive_identity import ArchiveIdentity, ArchiveLocation, OwnedArchiveLocation
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

if TYPE_CHECKING:
    from polylogue.operations.daemon_execution import OperationRuntime
    from polylogue.operations.daemon_reads import DaemonReadDependencies


@dataclass(frozen=True, slots=True)
class OperationContext:
    archive_root: Path
    principal: MutationPrincipal
    serving_identity: Literal["daemon", "direct"]
    runtime: OperationRuntime | None = None
    read_dependencies: DaemonReadDependencies | None = None
    read_control: QueryExecutionContext | None = None

    @classmethod
    def direct_read(
        cls,
        archive_root: Path,
        *,
        read_dependencies: DaemonReadDependencies | None = None,
    ) -> OperationContext:
        """Construct the deliberately read-only local adapter authority."""
        return cls(
            archive_root,
            MutationPrincipal("cli:direct-read", frozenset({"read"}), "cli"),
            "direct",
            read_dependencies=read_dependencies,
        )


@dataclass(frozen=True, slots=True)
class PinnedOperationRead:
    archive: ArchiveStore
    identity: ArchiveIdentity
    schema_versions: dict[str, int]
    degraded_components: tuple[str, ...]
    vector_failure: LaneFailure | None = None


@dataclass(frozen=True, slots=True)
class OperationControlRead:
    identity: ArchiveIdentity
    schema_versions: dict[str, int]
    degraded_components: tuple[str, ...]


def prepare_operation_journals(root: Path) -> None:
    """Establish live WAL policy under the writer before exposing readers.

    Fresh/bootstrap and restored sealed tiers may use rollback journals. A
    reader must never renegotiate those modes: pinning a rollback snapshot
    before the first writer opens would block that writer's WAL transition.
    Missing tiers remain missing; startup does not bootstrap or migrate here.
    """
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
    from polylogue.storage.sqlite.connection_profile import WRITE_CONNECTION_PROFILE, open_isolated_write_connection
    from polylogue.storage.sqlite.write_lease import require_write_lease

    require_write_lease("machine operation journal startup", archive_root=root)
    location = ArchiveLocation.resolve(root)
    for tier in ArchiveTier:
        path = location.active_index_path if tier is ArchiveTier.INDEX else root / f"{tier.value}.db"
        if not path.is_file():
            continue
        with closing(
            open_isolated_write_connection(
                path,
                purpose="machine operation journal startup",
                archive_root=root,
                profile=WRITE_CONNECTION_PROFILE,
            )
        ) as connection:
            if connection.execute("PRAGMA journal_mode").fetchone()[0] != "wal":
                raise RuntimeError(f"machine operation tier {tier.value} did not enter WAL mode")


def observe_control_authority(root: Path) -> OperationControlRead:
    """Read control provenance without waiting behind the operation actuator."""
    from polylogue.operations.audit import AuditContinuityPendingError, AuditRepository

    identity = ArchiveIdentity.resolve_location(ArchiveLocation.resolve(root))
    try:
        with AuditRepository.for_archive_root(root).settled_machine_read() as versions:
            if ArchiveIdentity.resolve_location(ArchiveLocation.resolve(root)) != identity:
                raise ValueError("archive changed while observing operation control authority")
            return OperationControlRead(identity, versions, ())
    except AuditContinuityPendingError:
        return OperationControlRead(identity, {}, ("audit_continuity_pending",))


@contextmanager
def _direct_pin_guard(root: Path) -> Iterator[None]:
    ownership = OwnedArchiveLocation.acquire(ArchiveLocation.resolve(root))
    try:
        yield
    finally:
        ownership.release()


@contextmanager
def open_operation_read(
    root: Path,
    *,
    publication_guard: Callable[[], AbstractContextManager[None]] | None = None,
    read_timeout: float = 2.0,
    vector_model: str | None = None,
    execution_context: QueryExecutionContext | None = None,
) -> Iterator[PinnedOperationRead]:
    """Open and force tier snapshots before releasing publication exclusion."""

    with ExitStack() as cleanup:
        with publication_guard() if publication_guard is not None else _direct_pin_guard(root):
            location = ArchiveLocation.resolve(root)
            identity = ArchiveIdentity.resolve_location(location)
            archive = ArchiveStore.open_existing(root, index_path=location.active_index_path, read_timeout=read_timeout)
            cleanup.callback(archive.close)
            cleanup.callback(archive.end_read_snapshot)
            if execution_context is not None:
                cleanup.enter_context(InterruptibleSQLiteRead(execution_context).control_store(archive))
            versions, degraded = archive.pin_operation_snapshot()
            vector_failure = None
            if vector_model is not None:
                from polylogue.storage.search_providers.sqlite_vec_runtime import open_vector_read_snapshot

                try:
                    archive.operation_vector_connection = open_vector_read_snapshot(
                        embeddings_path=root / "embeddings.db",
                        index_path=location.active_index_path,
                        model=vector_model,
                        configure_connection=archive.configure_operation_read_connection,
                        defer_projection=True,
                    )
                except (DatabaseError, OSError) as exc:
                    if execution_context is not None and execution_context.should_abort():
                        raise
                    vector_failure = LaneFailure(
                        "vector",
                        "construction_failed",
                        str(exc),
                        "semantic snapshot could not be opened; restore embeddings readiness and retry",
                    )
                    degraded = (*degraded, "semantic_snapshot")
                else:
                    observed_version = int(
                        archive.operation_vector_connection.execute("PRAGMA user_version").fetchone()[0]
                    )
                    if observed_version != versions.get("embeddings"):
                        raise RuntimeError("semantic snapshot differs from observed embeddings schema")
            archive.operation_identity = identity
            archive.operation_degraded_components = degraded
            if ArchiveIdentity.resolve_location(ArchiveLocation.resolve(root)) != identity:
                raise RuntimeError("archive changed while pinning operation read authority")
            pinned = PinnedOperationRead(archive, identity, versions, degraded, vector_failure)
        if archive.operation_vector_connection is not None:
            from polylogue.storage.search_providers.sqlite_vec_runtime import prepare_vector_read_projection

            assert vector_model is not None
            try:
                prepare_vector_read_projection(archive.operation_vector_connection, model=vector_model)
            except DatabaseError as exc:
                if execution_context is not None and execution_context.should_abort():
                    raise
                archive.operation_vector_connection.close()
                archive.operation_vector_connection = None
                vector_failure = LaneFailure(
                    "vector",
                    "construction_failed",
                    str(exc),
                    "semantic lookup could not be prepared from its pinned snapshot",
                )
                degraded = (*degraded, "semantic_snapshot")
                archive.operation_degraded_components = degraded
                pinned = replace(pinned, vector_failure=vector_failure, degraded_components=degraded)
        yield pinned
