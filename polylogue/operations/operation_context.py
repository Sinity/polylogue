"""Explicit execution dependencies and the reader that owns result authority."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, ExitStack, closing, contextmanager, nullcontext
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING

from polylogue.archive.query.execution_control import (
    InterruptibleSQLiteRead,
    QueryCancelledError,
    QueryExecutionContext,
    QueryTimeoutError,
    QueryWorkBudgetExceededError,
)
from polylogue.archive.query.search_contract import LaneFailure
from polylogue.core.errors import DatabaseError
from polylogue.storage.archive_identity import ArchiveIdentity, ArchiveLocation
from polylogue.storage.search.cache import ReadViewIdentity, capture_read_view, current_cache_epoch
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

if TYPE_CHECKING:
    from polylogue.operations.audit import AuditRepository
    from polylogue.storage.embeddings.identity import EmbeddingRecipe


class ConcurrentArchivePublicationError(DatabaseError):
    """An unguarded read observed a republication between resolve and pin.

    Raised only on the reader route that holds no publication exclusion. The
    correct answer is a typed, retryable refusal: a reader that cannot prove
    it pinned the generation it resolved must not report those rows as if it
    had, and must not escalate to the maintenance-writer lease to get
    certainty.

    Deliberately declared here rather than in ``polylogue.core.errors``: that
    module is inside the derived-schema identity closure
    (``devtools schema closure``), so adding a class to it would move the
    identity and demand a full reconvergence for a read-path bugfix.
    """

    code = "concurrent_archive_publication"


@dataclass(frozen=True, slots=True)
class PinnedOperationRead:
    archive: ArchiveStore
    identity: ArchiveIdentity
    schema_versions: dict[str, int]
    degraded_components: tuple[str, ...]
    vector_failure: LaneFailure | None = None
    #: The archive view this read was pinned against, when one could be named
    #: for the pin itself.  ``None`` means the read-result cache epoch moved
    #: across the pin, so no single view describes the snapshot and the read
    #: is deliberately uncacheable rather than labelled with a guess.
    read_view: ReadViewIdentity | None = None
    source_block_reads: dict[tuple[str, str], object] = field(default_factory=dict, compare=False, repr=False)
    source_target_resources: ExitStack = field(default_factory=ExitStack, compare=False, repr=False)
    checkpoint: Callable[[], None] | None = field(default=None, compare=False, repr=False)


@dataclass(frozen=True, slots=True)
class OperationControlRead:
    identity: ArchiveIdentity
    schema_versions: dict[str, int]
    degraded_components: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class OperationControlResult:
    state: dict[str, object]
    snapshot: OperationControlRead | None


def prepare_operation_journals(root: Path) -> None:
    """Establish live WAL policy while holding the archive's writer gate."""
    from polylogue.storage.sqlite.write_lease import require_write_lease, write_lease

    require_write_lease("machine operation journal startup", archive_root=root)
    with write_lease("machine operation journal startup", archive_root=root):
        _prepare_operation_journals_under_lease(root)


def _prepare_operation_journals_under_lease(root: Path) -> None:
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


@contextmanager
def open_operation_control(root: Path, *, audit: AuditRepository | None = None) -> Iterator[OperationControlRead]:
    """Bind receipt reads and their metadata to the same settled source/audit view."""
    from polylogue.operations.audit import AuditRepository

    repository = audit or AuditRepository.for_archive_root(root)
    identity = ArchiveIdentity.resolve_location(ArchiveLocation.resolve(root))
    with repository.settled_machine_read() as versions:
        if ArchiveIdentity.resolve_location(ArchiveLocation.resolve(root)) != identity:
            raise ValueError("archive changed while observing operation control authority")
        yield OperationControlRead(identity, versions, ())
        if ArchiveIdentity.resolve_location(ArchiveLocation.resolve(root)) != identity:
            raise ValueError("archive changed while reading operation control result")


def observe_control_authority(root: Path) -> OperationControlRead:
    """Read control provenance without waiting behind the operation actuator."""
    from polylogue.operations.audit import AuditContinuityError, AuditContinuityPendingError

    try:
        with open_operation_control(root) as snapshot:
            return snapshot
    except AuditContinuityError as exc:
        # A read never repairs continuity, so every state it cannot settle is
        # a named gap rather than a raise.  ``AuditContinuityPendingError`` is
        # an in-flight or unreconciled transition; the wider class also covers
        # an archive whose continuity halves are not seeded yet -- a bootstrap
        # condition an operation read must degrade through, not fail on.
        gap = (
            "audit_continuity_pending"
            if isinstance(exc, AuditContinuityPendingError)
            else "audit_continuity_unavailable"
        )
        identity = ArchiveIdentity.resolve_location(ArchiveLocation.resolve(root))
        return OperationControlRead(identity, {}, (gap,))


def observe_embedding_mutation_authority(root: Path) -> OperationControlRead:
    """Observe every available tier without retaining a reader across the write."""
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
    from polylogue.storage.sqlite.connection_profile import assert_tier_schema_supported, open_readonly_connection
    from polylogue.storage.sqlite.schema import assert_readable_archive_layout

    snapshot = observe_control_authority(root)
    location = ArchiveLocation.resolve(root)
    versions = dict(snapshot.schema_versions)
    degraded = list(snapshot.degraded_components)
    for tier in ("index", "source", "embeddings", "user", "ops", "audit"):
        path = location.active_index_path if tier == "index" else root / f"{tier}.db"
        if not path.is_file():
            degraded.append(tier)
            continue
        with closing(open_readonly_connection(path, validate_schema=False)) as connection:
            if tier == "index":
                resolved_index = path.resolve()
                generation_id = (
                    resolved_index.parent.name if resolved_index.parent.parent.name == ".index-generations" else None
                )
                assert_readable_archive_layout(connection, generation_id=generation_id)
            elif tier == "embeddings":
                assert_tier_schema_supported(connection, path, ArchiveTier.EMBEDDINGS)
            versions[tier] = int(connection.execute("PRAGMA user_version").fetchone()[0])
    if ArchiveIdentity.resolve_location(ArchiveLocation.resolve(root)) != snapshot.identity:
        raise ValueError("archive changed while observing embedding mutation authority")
    return replace(snapshot, schema_versions=versions, degraded_components=tuple(degraded))


def abort_checkpoint(ctx: QueryExecutionContext) -> Callable[[], None]:
    """A Python-level abort checkpoint for reads between SQL statements.

    It raises the same typed errors the SQLite progress handler raises for
    ``ctx.abort_reason()``, so the operation envelope reports ``cancelled`` or
    ``timed-out`` instead of an unmapped exception.
    """

    def check() -> None:
        reason = ctx.abort_reason()
        if reason == "cancelled":
            raise QueryCancelledError(f"archive read cancelled (call {ctx.call_id})")
        if reason == "work_budget_exceeded":
            raise QueryWorkBudgetExceededError(f"archive read exceeded SQLite work budget (call {ctx.call_id})")
        if reason == "timed_out":
            raise QueryTimeoutError(f"archive read exceeded deadline (call {ctx.call_id})")

    return check


@contextmanager
def open_operation_read(
    root: Path,
    *,
    publication_guard: Callable[[], AbstractContextManager[object]] | None = None,
    read_timeout: float = 2.0,
    vector_recipe: EmbeddingRecipe | None = None,
    execution_context: QueryExecutionContext | None = None,
) -> Iterator[PinnedOperationRead]:
    """Own the explicit controlled operation-read boundary and pin its snapshots.

    ``publication_guard`` is supplied by the process that *owns* publication
    (the daemon runtime), which can genuinely hold a republication off while
    the snapshot is pinned. A reader with no such authority -- a direct CLI
    query verb, or a materializer running outside the daemon -- has nothing
    to exclude and must not pretend otherwise: it previously took
    :class:`OwnedArchiveLocation`, the *exclusive maintenance/campaign writer*
    lease, which opens ``.archive-ownership.lock`` ``O_RDWR|O_CREAT`` and
    writes an owner record. That made a pure read fail on a read-only archive
    root, on an archive another process legitimately owns, and on any
    permission-restricted copy -- and it wrote to the thing it was reading.

    The read-appropriate route is optimistic instead of exclusive: resolve the
    archive identity, open read-only, pin the tier snapshots, then re-resolve
    and require the identity to be unchanged. A republication that lands
    mid-pin is a typed refusal (:class:`ConcurrentArchivePublicationError`),
    never a silently torn read and never a false success. This is the same
    shape ``open_operation_control`` already uses one level up.

    No read on this route lazily materializes anything, so none of them needs
    the writer lease. Writers still take it at every mutating call site
    (archive init, tier migration, root relocation, continuity recovery,
    embedding restore, the durable change train, and the daemon itself); the
    sole-writer contract is untouched.
    """

    with ExitStack() as cleanup:
        with publication_guard() if publication_guard is not None else nullcontext():
            location = ArchiveLocation.resolve(root)
            identity = ArchiveIdentity.resolve_location(location)
            archive = ArchiveStore.open_existing(
                root,
                index_path=location.active_index_path,
                read_timeout=read_timeout,
                read_only=True,
            )
            cleanup.callback(archive.close)
            cleanup.callback(archive.end_read_snapshot)
            if execution_context is not None:
                cleanup.enter_context(InterruptibleSQLiteRead(execution_context).control_store(archive))
            # The read-result cache epoch is the announced index-content
            # revision, and it must describe the snapshot this read actually
            # evaluates -- not whatever is current when the query finishes.
            # Capturing it across the pin is the only place that can prove
            # the two agree: an invalidation landing between these two reads
            # leaves no single view describing the snapshot, so the read is
            # named uncacheable instead of being given the newer epoch.
            epoch_before_pin = current_cache_epoch()
            versions, degraded = archive.pin_operation_snapshot()
            read_view = (
                capture_read_view(
                    archive_root=archive.archive_root,
                    generation=str(archive.index_db_path.resolve()),
                    epoch=epoch_before_pin,
                )
                if current_cache_epoch() == epoch_before_pin
                else None
            )
            if (
                publication_guard is None
                and ArchiveIdentity.resolve_location(ArchiveLocation.resolve(root)) != identity
            ):
                raise ConcurrentArchivePublicationError(
                    "archive was republished while pinning an unguarded operation read; retry the read"
                )
            vector_failure = None
            if vector_recipe is not None:
                from polylogue.storage.search_providers.sqlite_vec_runtime import open_vector_read_snapshot

                try:
                    archive.operation_vector_connection = open_vector_read_snapshot(
                        embeddings_path=root / "embeddings.db",
                        index_path=location.active_index_path,
                        recipe=vector_recipe,
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
            if archive.operation_vector_connection is not None:
                # Joint admission owns this exact lender/vector pair. A later
                # connection with the same filename cannot inherit the proof.
                vars(archive.operation_vector_connection)["_polylogue_vector_read_canonical_connection"] = archive._conn
            pinned = PinnedOperationRead(
                archive,
                identity,
                versions,
                degraded,
                vector_failure,
                read_view,
                checkpoint=abort_checkpoint(execution_context) if execution_context is not None else None,
            )
            cleanup.enter_context(pinned.source_target_resources)
        vector_connection = archive.operation_vector_connection
        if vector_connection is not None:
            from polylogue.storage.search_providers.sqlite_vec_runtime import prepare_vector_read_projection

            assert vector_recipe is not None
            try:
                prepare_vector_read_projection(vector_connection, recipe=vector_recipe)
            except DatabaseError as exc:
                if execution_context is not None and execution_context.should_abort():
                    raise
                archive.close_operation_vector_connection()
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
