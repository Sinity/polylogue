"""Run storage-law fixtures on the canonical admitted preparation owner."""

from __future__ import annotations

import asyncio
import sqlite3
import sys
import tempfile
from builtins import BaseExceptionGroup
from collections.abc import Callable, Iterator
from contextlib import closing, contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

from polylogue.core.compute import BoundedComputeAdapter

if TYPE_CHECKING:
    from polylogue.archive.revision_replay import RevisionReplayPlan
    from polylogue.sources.parsers.base import ParsedSession
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from polylogue.storage.sqlite.archive_tiers.revision_governance import ArchiveRawParsedWriteResult
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


def publish_prepared_source(
    root: Path,
    actor: str,
    prepare: Callable[[PreparedIndexMutation], None],
    *,
    after_prepare: Callable[[], None] | None = None,
    before_publish: Callable[[], None] | None = None,
    index_path: Path | None = None,
) -> None:
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
        seal = (
            PreparedIndexMutation.source_only(archive_root=root)
            if index_path is None
            else PreparedIndexMutation(index_path, archive_root=root)
        )
        retained.append(seal)
        try:
            with seal.original_read_snapshot(), seal.source_producer():
                prepare(seal)
            permit = seal.prepare_source_mutation()
            if after_prepare is not None:
                # Durable evidence moves after preparation and before publication.
                admit_stage_write(f"{actor}.intervening", after_prepare)
            if before_publish is not None:
                before_publish()

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


def publish_fixture_byte_classification(archive: ArchiveStore, logical_source_key: str) -> RevisionReplayPlan:
    """Publish the production byte law for fixtures with supplied parsed sessions.

    Synthetic byte-law fixtures cannot run a provider parser. Their Source
    authority is prepared on the same admitted owner and original seal as Raw
    convergence; replay still uses the fixture's explicit parsed sessions.
    """
    from polylogue.sources.revision_backfill import _require_classification_inputs_current
    from polylogue.storage.blob_store import BlobStore
    from polylogue.storage.sqlite.archive_tiers.revision_governance import prepare_raw_revision_byte_classification

    archive.commit()
    payload_store = BlobStore(archive.archive_root / "blob")
    blob_stats: tuple[tuple[str, tuple[int, int, int, int, int]], ...] = ()

    def prepare(seal: PreparedIndexMutation) -> None:
        nonlocal blob_stats
        _changed, blob_stats = prepare_raw_revision_byte_classification(
            seal, logical_source_key, payload_store=payload_store
        )

    publish_prepared_source(
        archive.archive_root,
        "test.byte-classification",
        prepare,
        before_publish=lambda: _require_classification_inputs_current(blob_stats, payload_store),
        index_path=archive.index_db_path,
    )
    return archive.raw_revision_replay_plan(logical_source_key)


def current_fixture_parser_receipts(
    root: Path, raw_ids: list[str], *, after_prepare: Callable[[], None] | None = None
) -> tuple[bool, ...]:
    """Read receipt currency from Raw's original-seal law and validate publication."""
    from polylogue.storage.sqlite.archive_tiers.revision_governance import prepared_parser_census_is_current

    current: list[bool] = []

    def prepare(seal: PreparedIndexMutation) -> None:
        current.extend(prepared_parser_census_is_current(seal, raw_id) for raw_id in raw_ids)

    publish_prepared_source(root, "test.parser-receipt-currency", prepare, after_prepare=after_prepare)
    return tuple(current)


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
            result = archive.apply_raw_revision_replay(
                plan,
                parsed_by_raw_id,
                prepared_outcome=outcome,
                acquired_at_ms=acquired_at_ms,
                **apply_options,
            )
    if result[1]:
        # Retained replay acknowledges an applied outcome's terminal raws on
        # Source after the Index outcome, as prepare_retained_replay_source does.
        from polylogue.storage.sqlite.archive_tiers.revision_governance import (
            prepare_raw_parse_success,
            raw_revision_descriptor,
            revision_replay_terminal_raw_ids,
        )

        terminal = {
            raw_id: raw_revision_descriptor(archive, raw_id)[0] for raw_id in revision_replay_terminal_raw_ids(plan)
        }
        archive.commit()

        def acknowledge(source_seal: PreparedIndexMutation) -> None:
            for raw_id, provider in terminal.items():
                prepare_raw_parse_success(source_seal, raw_id, provider=provider)

        publish_prepared_source(root, "test.revision-replay", acknowledge)
    return result


def ingest_append_plans_on_owner(root: Path, append_owner: Any, plans: list[Any]) -> Any:
    """Run live append intake through the canonical daemon owner's Raw convergence."""
    from tests.infra.live_ingest import prepared_live_convergence_owner

    async def run() -> Any:
        async with prepared_live_convergence_owner(root) as owner:
            return await owner.ingest_append_plans(append_owner, plans)

    return asyncio.run(run())


def publish_membership_census(archive: ArchiveStore, raw_id: str, sessions: Any, **census: Any) -> None:
    """Commit the open store, then publish a membership census on its original Source seal."""
    from polylogue.storage.sqlite.archive_tiers.revision_governance import replace_raw_membership_census

    archive.commit()
    publish_prepared_source(
        archive.archive_root,
        "test.membership-census",
        lambda seal: replace_raw_membership_census(seal, raw_id, sessions, **census),
    )


def write_fixture_raw_session(
    archive: ArchiveStore,
    session: ParsedSession,
    *,
    payload: bytes,
    source_path: str,
    acquired_at_ms: int,
    source_index: int = 0,
    file_mtime_ms: int | None = None,
) -> ArchiveRawParsedWriteResult:
    """Admit raw bytes, then publish a supplied parsed session for that raw.

    The raw is admitted through the store's canonical raw writer; the parsed
    session is prepared and published by the original-seal fixture writer
    with its inline attachments acquired first, and its parser census is
    published on the Source seal. The result maps the writer's own outcome.
    """
    from polylogue.archive.revision_authority import RawRevisionAuthority, RawRevisionEnvelope, RawRevisionKind
    from polylogue.core.sources import origin_from_provider
    from polylogue.storage.blob_store import BlobStore
    from polylogue.storage.sqlite.archive_tiers.revision_governance import (
        ArchiveRawParsedWriteResult,
        record_current_parser_source_census,
    )
    from polylogue.storage.sqlite.archive_tiers.source_write import deterministic_blob_hash
    from polylogue.storage.sqlite.archive_tiers.write import ArchiveWriteOutcome
    from tests.infra.index_writer import write_fixture_index_session

    # A first observation of its logical source is a FULL/ASSERTED baseline,
    # exactly as canonical raw admission records it.
    raw_id = archive.write_raw_payload(
        provider=session.source_name,
        payload=payload,
        source_path=source_path,
        canonical_source_path=source_path,
        source_index=source_index,
        acquired_at_ms=acquired_at_ms,
        file_mtime_ms=file_mtime_ms,
        native_id=session.provider_session_id,
        revision=RawRevisionEnvelope(
            logical_source_key=f"{origin_from_provider(session.source_name).value}:{session.provider_session_id}",
            kind=RawRevisionKind.FULL,
            source_revision=deterministic_blob_hash(payload).hex(),
            acquisition_generation=0,
            authority=RawRevisionAuthority.ASSERTED,
        ),
    )
    archive.commit()
    blobs = BlobStore(archive.archive_root / "blob")
    preacquired: dict[object, tuple[bytes | None, int, str]] = {}
    for attachment in session.attachments:
        if attachment.inline_bytes is not None:
            blob_hash, size = blobs.write_from_bytes(attachment.inline_bytes)
            preacquired[id(attachment)] = (bytes.fromhex(blob_hash), size, "acquired")
    from polylogue.pipeline.ids import session_id as make_session_id

    expected_session_id = str(make_session_id(session.source_name, session.provider_session_id))

    def stored_hash() -> object:
        row = archive._conn.execute(
            "SELECT content_hash FROM sessions WHERE session_id=?", (expected_session_id,)
        ).fetchone()
        return None if row is None else bytes(row[0])

    before = stored_hash()
    outcomes: list[ArchiveWriteOutcome] = []
    session_id = write_fixture_index_session(
        archive._conn,
        session,
        archive_root=archive.archive_root,
        raw_id=raw_id,
        preacquired_attachment_blobs=preacquired,
        write_outcome=outcomes,
    )
    # The parser census binds the original prepared artifact output for this
    # raw, never a loose parsed list.
    from polylogue.sources.prepared_jsonl import PreparedJsonl

    source = archive._ensure_source_conn()
    blob_row = source.execute("SELECT blob_hash FROM raw_sessions WHERE raw_id=?", (raw_id,)).fetchone()
    assert blob_row is not None
    with tempfile.TemporaryDirectory(prefix="fixture-census-", dir=archive.archive_root / "blob") as directory:
        artifact = PreparedJsonl.from_sessions(
            [session],
            blob_hash=bytes(blob_row[0]).hex(),
            artifact_directory=Path(directory),
            publication_publisher=None,
        )
        try:
            publish_prepared_source(
                archive.archive_root,
                "test.fixture.raw-session-census",
                lambda seal: record_current_parser_source_census(
                    seal, raw_id, parser_sessions=artifact.session_sequence()
                ),
            )
        finally:
            artifact.discard()
    outcome = outcomes[-1]
    # Content change is the stored session hash moving, as the retained
    # writer reported it; re-binding identical content to a new raw is not.
    wrote = outcome.wrote and stored_hash() != before
    return ArchiveRawParsedWriteResult(
        raw_id=raw_id,
        session_id=session_id,
        content_changed=wrote,
        counts=archive._write_counts(session) if wrote else archive._skipped_counts(session),
        publication_refused=outcome.stale_skipped or outcome.suppression_skipped,
        unresolved_attachment_owners=outcome.unresolved_attachment_owners,
    )


def write_fixture_precedence_raw_session(
    archive: ArchiveStore,
    session: ParsedSession,
    *,
    payload: bytes,
    source_path: str,
    acquired_at_ms: int,
    source_index: int = 0,
    file_mtime_ms: int | None = None,
) -> ArchiveRawParsedWriteResult:
    """Admit raw bytes, then publish a parsed session through ingest precedence.

    ``write_fixture_raw_session`` publishes directly; this route prepares the
    session write on an original seal and hands it to the store's precedence
    decision (freshness, DOM fallback against native browser capture,
    identical-content skip), the decision retained ingest makes.
    """
    from polylogue.archive.revision_authority import RawRevisionAuthority, RawRevisionEnvelope, RawRevisionKind
    from polylogue.core.sources import origin_from_provider
    from polylogue.storage.blob_store import BlobStore
    from polylogue.storage.sqlite.archive_tiers.revision_governance import (
        _write_parsed_precedence_result,
        prepared_raw_revision_file_mtime,
    )
    from polylogue.storage.sqlite.archive_tiers.source_write import deterministic_blob_hash
    from polylogue.storage.sqlite.archive_tiers.write import PreparedSessionSourceRead, prepare_session_write
    from polylogue.storage.sqlite.reference_seal import PreparedIndexMutation

    raw_id = archive.write_raw_payload(
        provider=session.source_name,
        payload=payload,
        source_path=source_path,
        canonical_source_path=source_path,
        source_index=source_index,
        acquired_at_ms=acquired_at_ms,
        file_mtime_ms=file_mtime_ms,
        native_id=session.provider_session_id,
        revision=RawRevisionEnvelope(
            logical_source_key=f"{origin_from_provider(session.source_name).value}:{session.provider_session_id}",
            kind=RawRevisionKind.FULL,
            source_revision=deterministic_blob_hash(payload).hex(),
            acquisition_generation=0,
            authority=RawRevisionAuthority.ASSERTED,
        ),
    )
    archive.commit()
    blobs = BlobStore(archive.archive_root / "blob")
    preacquired: dict[object, tuple[bytes | None, int, str]] = {}
    for attachment in session.attachments:
        if attachment.inline_bytes is not None:
            blob_hash, size = blobs.write_from_bytes(attachment.inline_bytes)
            preacquired[id(attachment)] = (bytes.fromhex(blob_hash), size, "acquired")
    with PreparedIndexMutation(archive.index_db_path, archive_root=archive.archive_root) as seal:
        with seal.original_read_snapshot(), seal.source_producer():
            read = PreparedSessionSourceRead(seal, blob_store=blobs)
            prepared_write = prepare_session_write(
                seal.observer("index"),
                session,
                merge_append=False,
                fallback_timestamp=prepared_raw_revision_file_mtime(seal, raw_id),
                source_read=read,
                raw_id=raw_id,
                force_replace=False,
                before_input=seal.before_index_input,
            )
        with archive.index_mutation_scope(prepared_seal=seal):
            return _write_parsed_precedence_result(
                archive,
                session,
                raw_id=raw_id,
                source_index=source_index,
                stage_timings_s=None,
                stage_timing_prefix="append",
                manage_transaction=False,
                preacquired_attachment_blobs=preacquired,
                prepared_write=prepared_write,
            )


def open_independent_source(archive: ArchiveStore) -> sqlite3.Connection:
    """Commit the store, then open an independent Source writer for law seeding.

    A law that plants durable evidence the production writers would refuse
    (an ambiguous decision, a damaged append coordinate) writes it as another
    process would, outside the store's authorized custody connection.
    """
    archive.commit()
    return sqlite3.connect(archive.archive_root / "source.db")


@contextmanager
def independent_source_connection(archive: ArchiveStore) -> Iterator[sqlite3.Connection]:
    connection = open_independent_source(archive)
    try:
        yield connection
        connection.commit()
    finally:
        connection.close()
