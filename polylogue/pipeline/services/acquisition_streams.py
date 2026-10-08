"""Streaming helpers for acquisition service source traversal."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from functools import partial
from itertools import islice
from typing import TYPE_CHECKING, TypeVar

from polylogue.core.json import JSONDocument
from polylogue.logging import get_logger
from polylogue.pipeline.services.acquisition_records import make_raw_record
from polylogue.sources.parsers.base import RawSessionData
from polylogue.sources.pickle_spool import PickleSpool
from polylogue.sources.retained_acquisition import SourceInputRecord
from polylogue.storage.cursor_state import CursorStatePayload
from polylogue.storage.runtime import RawSessionRecord
from polylogue.storage.sqlite.archive_tiers.raw_admission import acquisition_timestamp_ms
from polylogue.storage.sqlite.archive_tiers.source_items import SourceItemAdmission, acquired_zip_manifest

if TYPE_CHECKING:
    from pathlib import Path

    from polylogue.config import Source
    from polylogue.pipeline.services.ingest_execution import IngestExecution
    from polylogue.sources.drive.types import DriveConfigLike, DriveUILike
    from polylogue.sources.drive.witness import DriveListingWitness
    from polylogue.storage.blob_store import BlobStore
    from polylogue.storage.repository import SessionRepository

logger = get_logger(__name__)
ObservationCallback = Callable[[JSONDocument], None]
_AcquisitionItem = TypeVar("_AcquisitionItem")


def _drain_batch(
    iterator: Iterator[_AcquisitionItem],
    *,
    batch_size: int,
    blob_store: BlobStore | None = None,
) -> list[_AcquisitionItem]:
    """Read up to ``batch_size`` items without sentinel gymnastics."""
    batch = list(islice(iterator, batch_size))
    if blob_store is not None:
        from polylogue.storage.blob_publication import flush_blob_publications

        flush_blob_publications(blob_store)
    return batch


async def iter_source_raw_stream(
    source: Source,
    *,
    blob_root: Path | None = None,
    blob_store: BlobStore | None = None,
    known_mtimes: dict[str, str] | None = None,
    known_cursors: dict[str, dict[str, object]] | None = None,
    observation_callback: ObservationCallback | None = None,
    progress_callback: Callable[[int, str | None], None] | None = None,
    execution: IngestExecution | None = None,
) -> AsyncIterator[SourceInputRecord]:
    """Stream raw source payloads without materializing the full iterator."""
    from polylogue.pipeline.services import acquisition as acquisition_root

    loop = asyncio.get_running_loop()

    def _status_callback(desc: str) -> None:
        if progress_callback is None:
            return
        loop.call_soon_threadsafe(progress_callback, 0, desc)

    iterator = iter(
        acquisition_root.iter_source_acquisition_records(
            source,
            known_mtimes=known_mtimes,
            known_cursors=known_cursors,
            blob_root=blob_root,
            blob_store=blob_store,
            observation_callback=observation_callback,
            status_callback=_status_callback if progress_callback is not None else None,
        )
    )
    batch_size = 128

    loop = asyncio.get_running_loop()
    from polylogue.storage.sqlite.async_sqlite import _await_settled

    with ThreadPoolExecutor(max_workers=1) as executor:
        try:
            while True:
                pending = loop.run_in_executor(
                    executor,
                    partial(
                        _drain_batch,
                        iterator,
                        batch_size=batch_size,
                        blob_store=blob_store if execution is None else None,
                    ),
                )
                await _await_settled(pending)
                batch = pending.result()
                if execution is not None and blob_store is not None:
                    from polylogue.storage.blob_publication import flush_blob_publications

                    await execution.publish_sync("blobs", lambda: flush_blob_publications(blob_store))
                if not batch:
                    break
                for item in batch:
                    yield item
        finally:
            # Close on the owning I/O worker after its last read settles.
            close = getattr(iterator, "close", None)
            if close is not None:
                pending_close = loop.run_in_executor(executor, close)
                await _await_settled(pending_close)
                pending_close.result()


async def iter_drive_raw_stream(
    source: Source,
    *,
    blob_store: BlobStore | None = None,
    known_mtimes: dict[str, str] | None = None,
    ui: DriveUILike | None = None,
    cursor_state: CursorStatePayload | None = None,
    drive_config: DriveConfigLike | None = None,
    observation_callback: ObservationCallback | None = None,
    progress_callback: Callable[[int, str | None], None] | None = None,
    execution: IngestExecution | None = None,
    drive_witness: DriveListingWitness | None = None,
) -> AsyncIterator[RawSessionData]:
    """Stream Drive payloads as raw records without touching the local cache."""
    from polylogue.sources.drive import iter_drive_raw_data

    loop = asyncio.get_running_loop()

    def _status_callback(desc: str) -> None:
        if progress_callback is None:
            return
        loop.call_soon_threadsafe(progress_callback, 0, desc)

    batch_size = 32
    iterator = iter(
        iter_drive_raw_data(
            source=source,
            witness=drive_witness,
            ui=ui,
            cursor_state=cursor_state,
            drive_config=drive_config,
            known_mtimes=known_mtimes,
            observation_callback=observation_callback,
            status_callback=_status_callback if progress_callback is not None else None,
            blob_store=blob_store,
        )
    )

    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor(max_workers=1) as executor:
        while True:
            pending = loop.run_in_executor(
                executor,
                partial(
                    _drain_batch, iterator, batch_size=batch_size, blob_store=blob_store if execution is None else None
                ),
            )
            if execution is None:
                batch = await pending
            else:
                from polylogue.storage.blob_publication import flush_blob_publications

                batch = await execution.settle(pending)
                if blob_store is not None:
                    await execution.publish_sync("blobs", lambda: flush_blob_publications(blob_store))
            if not batch:
                break
            for item in batch:
                yield item


async def iter_raw_record_stream(
    source: Source,
    *,
    blob_root: Path | None = None,
    blob_store: BlobStore | None = None,
    known_mtimes: dict[str, str] | None = None,
    known_cursors: dict[str, dict[str, object]] | None = None,
    ui: DriveUILike | None = None,
    cursor_state: CursorStatePayload | None = None,
    drive_config: DriveConfigLike | None = None,
    observation_callback: ObservationCallback | None = None,
    progress_callback: Callable[[int, str | None], None] | None = None,
    execution: IngestExecution | None = None,
    drive_witness: DriveListingWitness | None = None,
    input_repository: SessionRepository | None = None,
    before_input_complete: Callable[[], Awaitable[None]] | None = None,
    input_observation_callback: Callable[[str, str, tuple[int, int, int, int, int], str | None], None] | None = None,
) -> AsyncIterator[RawSessionRecord]:
    """Yield prepared RawSessionRecord values for a source."""
    # Ahead of the Drive/local branch: ``iter_source_acquisition_records`` carries this
    # refusal, but the Drive branch never calls it, so a configured drive
    # cache that lives inside a DIFFERENT Polylogue archive was accepted as a
    # capture location and its bytes copied into the destination blob store.
    # The refusal belongs where the route is chosen, so both branches get it.
    if source.path is not None:
        from pathlib import Path as _Path

        from polylogue.sources.source_root_admission import refuse_non_capture_source_root

        destination_root: Path | None = None
        if blob_store is not None:
            destination_root = blob_store.root.parent
        elif blob_root is not None:
            destination_root = _Path(blob_root).parent
        else:
            # The omitted arguments select the default archive's blob store
            # later in this function. Resolve that same destination before
            # capture-root admission so its own inbox/cache remains valid.
            from polylogue.paths import blob_store_root

            destination_root = blob_store_root().parent
        refuse_non_capture_source_root(_Path(source.path), destination=destination_root)

    raw_stream: AsyncIterator[RawSessionData] | AsyncIterator[SourceInputRecord]
    if source.is_drive:
        raw_stream = iter_drive_raw_stream(
            source,
            drive_witness=drive_witness,
            blob_store=blob_store,
            known_mtimes=known_mtimes,
            ui=ui,
            cursor_state=cursor_state,
            drive_config=drive_config,
            observation_callback=observation_callback,
            progress_callback=progress_callback,
            execution=execution,
        )
    else:
        raw_stream = iter_source_raw_stream(
            source,
            blob_root=blob_root,
            blob_store=blob_store,
            known_mtimes=known_mtimes,
            known_cursors=known_cursors,
            observation_callback=observation_callback,
            progress_callback=progress_callback,
            execution=execution,
        )

    generation: str | None = None
    source_item: str | None = None
    fingerprint: str | None = None
    input_acquired_at: str | None = None
    coordinates: PickleSpool[str] | None = None

    async def publish(effect: Callable[[], Awaitable[object]]) -> object:
        if execution is None:
            return await effect()
        return await execution.publish("raw", effect)

    def observe_zip_effect(effect: Callable[..., Awaitable[object]]) -> Awaitable[object]:
        return effect(observed_at_ms=acquisition_timestamp_ms(datetime.now(UTC).isoformat()))

    try:
        async for item in raw_stream:
            envelope = item if isinstance(item, SourceInputRecord) else None
            if envelope is not None and envelope.data is None:
                if envelope.input_blob_hash is not None:
                    if coordinates is not None:
                        # The predecessor did not reach normal exhaustion.
                        coordinates.close()
                    generation = source_item = fingerprint = None
                    coordinates = None
                    input_acquired_at = datetime.now(UTC).isoformat()
                    identity = envelope.captured_input_identity
                    if (
                        input_observation_callback is not None
                        and identity is not None
                        and envelope.file_observation is not None
                    ):
                        input_observation_callback(
                            identity.semantic_source_path,
                            identity.canonical_source_path,
                            envelope.file_observation,
                            identity.profile_key,
                        )
                    if input_repository is None:
                        continue
                    receipt = envelope.input_publication_receipt_id
                    fingerprint = envelope.enumeration_fingerprint
                    if identity is None or receipt is None or fingerprint is None:
                        raise ValueError("ordinary ZIP input lacks accepted publication evidence")
                    manifest = acquired_zip_manifest(
                        blob_hash=envelope.input_blob_hash,
                        publication_receipt_id=receipt,
                        captured_identity=identity,
                        enumeration_fingerprint=fingerprint,
                        source_name=envelope.input_source_name,
                    )
                    generation = manifest.source_generation_id
                    observed_at = acquisition_timestamp_ms(input_acquired_at)
                    published_source_item = await publish(
                        partial(input_repository.publish_acquired_zip_input, manifest, observed_at_ms=observed_at)
                    )
                    if not isinstance(published_source_item, str):
                        raise ValueError("ordinary ZIP input returned no exact item identity")
                    source_item = published_source_item
                    coordinates = PickleSpool[str]()
                elif envelope.member_disposition is not None:
                    if input_repository is None:
                        continue
                    if (
                        generation is None
                        or source_item is None
                        or envelope.entry_ordinal is None
                        or envelope.member_name is None
                    ):
                        raise ValueError("ZIP disposition has no accepted input")
                    await publish(
                        partial(
                            observe_zip_effect,
                            partial(
                                input_repository.record_acquired_zip_disposition,
                                source_generation_id=generation,
                                source_item_id=source_item,
                                entry_ordinal=envelope.entry_ordinal,
                                member_name=envelope.member_name,
                                disposition=envelope.member_disposition,
                                diagnostic=envelope.diagnostic,
                            ),
                        )
                    )
                elif envelope.enumeration_complete:
                    if input_repository is None:
                        input_acquired_at = None
                        continue
                    if (
                        generation is None
                        or source_item is None
                        or fingerprint is None
                        or coordinates is None
                        or envelope.member_count is None
                    ):
                        raise ValueError("ZIP completion has no accepted input denominator")
                    if before_input_complete is not None:
                        await before_input_complete()
                    await publish(
                        partial(
                            observe_zip_effect,
                            partial(
                                input_repository.complete_acquired_zip_input,
                                source_generation_id=generation,
                                source_item_id=source_item,
                                enumeration_fingerprint=fingerprint,
                                record_coordinates=coordinates,
                                member_count=envelope.member_count,
                            ),
                        )
                    )
                    coordinates.close()
                    coordinates = None
                    generation = source_item = fingerprint = None
                    input_acquired_at = None
                continue
            raw_data = envelope.data if envelope is not None else item
            if not isinstance(raw_data, RawSessionData):
                raise TypeError("acquisition record has no raw payload")
            if not raw_data.raw_bytes and not raw_data.blob_hash:
                continue
            acquired_at = None
            if raw_data.captured_zip_coordinate is not None:
                if envelope is None or input_acquired_at is None:
                    raise ValueError("ZIP raw has no accepted acquisition pass")
                acquired_at = input_acquired_at
            record = make_raw_record(
                raw_data, source.name, blob_root=blob_root, blob_store=blob_store, acquired_at=acquired_at
            )
            if envelope is not None and raw_data.captured_zip_coordinate is not None and input_repository is not None:
                if generation is None or source_item is None or coordinates is None:
                    raise ValueError("ZIP raw has no accepted frozen input")
                if raw_data.addressing_mode is None:
                    raise ValueError("ZIP raw lacks its captured member reading")
                coordinates.append(envelope.coordinate)
                record.source_item = SourceItemAdmission(
                    generation,
                    source_item,
                    envelope.coordinate,
                    envelope.entry_ordinal,
                    envelope.split_index,
                    raw_data.addressing_mode.value,
                    raw_data.content_identity,
                )
            if blob_store is not None and raw_data.raw_bytes:
                from polylogue.storage.blob_publication import flush_blob_publications

                if execution is None:
                    flush_blob_publications(blob_store)
                else:
                    await execution.publish_sync("blobs", lambda: flush_blob_publications(blob_store))
            del raw_data
            yield record
    finally:
        try:
            close_stream = getattr(raw_stream, "aclose", None)
            if close_stream is not None:
                from polylogue.storage.sqlite.async_sqlite import _await_settled

                close_task = asyncio.ensure_future(close_stream())
                await _await_settled(close_task)
                close_task.result()
        finally:
            if coordinates is not None:
                coordinates.close()


__all__ = ["iter_drive_raw_stream", "iter_raw_record_stream", "iter_source_raw_stream"]
