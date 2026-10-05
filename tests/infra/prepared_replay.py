"""Run storage-law fixtures on the canonical admitted preparation owner."""

from __future__ import annotations

import asyncio
import sys
from builtins import BaseExceptionGroup
from collections.abc import Callable
from contextlib import closing
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

from polylogue.core.compute import BoundedComputeAdapter

if TYPE_CHECKING:
    from polylogue.archive.revision_replay import RevisionReplayPlan
    from polylogue.sources.parsers.base import ParsedSession
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from polylogue.storage.sqlite.reference_seal import PreparedIndexMutation

T = TypeVar("T")


def run_on_convergence_owner(root: Path, actor: str, operation: Callable[[BoundedComputeAdapter], T]) -> T:
    """Run one synchronous law body on the real daemon preparation worker.

    Raw preparation refuses any thread other than its admitted compute
    creator, so a law body that computes, prepares or publishes Raw
    observations runs here with that owner's adapter and writer bridge.
    """
    from tests.infra.live_ingest import prepared_live_convergence_owner

    async def run() -> T:
        async with prepared_live_convergence_owner(root) as owner:
            return await owner.run_convergence_sync(actor, operation, owner._compute_adapter)

    return asyncio.run(run())


def publish_prepared_source(root: Path, actor: str, prepare: Callable[[PreparedIndexMutation], None]) -> None:
    """Prepare Source statements on an original seal, then publish that tape.

    ``prepare`` runs inside the seal's original read window and Source
    producer phase. The captured statements are applied on the dedicated
    Source writer under the stage admission and accepted on the same seal,
    the canonical order shown by the retained Source phase controls.
    """
    from polylogue.core.stage_admission import admit_stage_write
    from polylogue.storage.sqlite.reference_seal import PreparedIndexMutation
    from tests.infra.live_ingest import prepared_live_convergence_owner

    retained: list[PreparedIndexMutation] = []

    def body() -> None:
        seal = PreparedIndexMutation.source_only(archive_root=root)
        retained.append(seal)
        try:
            with seal.original_read_snapshot(), seal.source_producer():
                prepare(seal)
            permit = seal.prepare_source_mutation()

            def publish() -> None:
                with permit.hold_authority(), permit.mutation_connection() as source:
                    with closing(source.execute("BEGIN IMMEDIATE")):
                        pass
                    permit.apply_source_statements(source)
                    permit.allow_commit(source)
                    source.commit()
                    seal.accept_known_tier_commit(permit.committed())

            admit_stage_write(actor, publish)
        finally:
            primary = sys.exception()
            try:
                seal.close()
            except BaseException as cleanup:
                if primary is not None and cleanup is not primary:
                    raise BaseExceptionGroup("source preparation and seal close failed", [primary, cleanup]) from None
                raise
            retained.remove(seal)

    async def run() -> None:
        async with prepared_live_convergence_owner(root) as owner:
            await owner.run_prepared_sync(
                actor,
                body,
                settlement_owners=lambda: tuple(retained),
                estimated_bytes=0,
            )

    asyncio.run(run())


def apply_prepared_revision_replay(
    archive: ArchiveStore,
    plan: RevisionReplayPlan,
    parsed_by_raw_id: dict[str, ParsedSession],
    *,
    acquired_at_ms: int,
    **apply_options: Any,
) -> tuple[str, tuple[str, ...]]:
    """Prepare the byte replay outcome off-writer, then apply it on the store.

    The writer accepts only an outcome prepared on an original seal: the
    composed aggregate's adoption, its prepared session write and the selected
    Index head decision. A law that supplies synthetic parsed sessions for its
    raw chain gets exactly that canonical preparation here; the store then
    publishes under its own Index mutation scope.
    """
    from polylogue.sources.dispatch import merge_parsed_session_chunks
    from polylogue.storage.blob_store import BlobStore
    from polylogue.storage.sqlite.archive_tiers.revision_governance import (
        prepare_revision_replay_outcome,
        prepared_raw_revision_file_mtime,
    )
    from polylogue.storage.sqlite.archive_tiers.write import PreparedSessionSourceRead, prepare_session_write
    from polylogue.storage.sqlite.reference_seal import PreparedIndexMutation

    accepted = plan.accepted_raw_ids
    if not accepted:
        raise ValueError("a prepared replay law requires an accepted raw chain")
    chunks = [parsed_by_raw_id[raw_id] for raw_id in accepted]
    composed = chunks if len(chunks) == 1 else merge_parsed_session_chunks(chunks)
    if len(composed) != 1:
        raise ValueError("a prepared replay law must compose exactly one session")
    aggregate = composed[0]
    tip = accepted[-1]
    root = archive.archive_root
    with PreparedIndexMutation(archive.index_db_path, archive_root=root) as seal:
        with seal.original_read_snapshot(), seal.source_producer():
            read = PreparedSessionSourceRead(seal, blob_store=BlobStore(root / "blob"))
            adoption = read.prepare_raw_revision_replay_adoption(
                [aggregate], logical_source_key=plan.logical_source_key, raw_ids=accepted
            )
            prepared_write = prepare_session_write(
                seal.observer("index"),
                aggregate,
                merge_append=False,
                fallback_timestamp=prepared_raw_revision_file_mtime(seal, tip),
                source_read=read,
                raw_id=tip,
                force_replace=True,
                before_input=seal.before_index_input,
            )
            outcome = prepare_revision_replay_outcome(
                seal,
                read,
                plan,
                adoption,
                aggregate_session=aggregate,
                aggregate_content_hash=prepared_write.rows.session_content_hash,
                prepared_write=prepared_write,
            )
        apply_options.setdefault("preacquired_attachment_blobs_by_raw_id", {raw_id: {} for raw_id in accepted})
        apply_options.setdefault("prepared_write", prepared_write)
        # The prepared write is published under the seal that prepared it.
        with archive.index_mutation_scope(prepared_seal=seal):
            return archive.apply_raw_revision_replay(
                plan,
                parsed_by_raw_id,
                prepared_outcome=outcome,
                acquired_at_ms=acquired_at_ms,
                **apply_options,
            )


def ingest_append_plans_on_owner(root: Path, append_owner: Any, plans: list[Any]) -> Any:
    """Run live append intake through the canonical daemon owner's Raw convergence."""
    from tests.infra.live_ingest import prepared_live_convergence_owner

    async def run() -> Any:
        async with prepared_live_convergence_owner(root) as owner:
            return await owner.ingest_append_plans(append_owner, plans)

    return asyncio.run(run())
