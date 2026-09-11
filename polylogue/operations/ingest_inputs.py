"""Freeze and enumerate machine-ingest inputs on the admitted compute lane."""

from __future__ import annotations

import stat
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from polylogue.pipeline.services.acquisition_records import make_raw_record, pending_pre_parse_raw_admission_request
from polylogue.sources.origin_specs import retained_enumeration_fingerprint
from polylogue.sources.retained_acquisition import iter_retained_source_records
from polylogue.sources.sqlite_snapshot import is_sqlite_path, snapshot_sqlite_to_blob
from polylogue.storage.blob_publication import ArchiveBlobPublisher
from polylogue.storage.runtime import RawSessionRecord
from polylogue.storage.sqlite.archive_tiers.raw_admission import (
    RawAdmissionPlan,
    SourceItemAdmission,
    plan_raw_admission,
)
from polylogue.storage.sqlite.archive_tiers.source_items import (
    FrozenSourceInput,
    FrozenSourceManifest,
    RetainedSourceInput,
    source_item_id,
)


@dataclass(frozen=True, slots=True)
class PreparedSourceRecord:
    record: RawSessionRecord
    admission: RawAdmissionPlan
    member: SourceItemAdmission


def prepare_ingest_inputs(
    path: Path,
    *,
    source_path: str | None,
    source_generation_id: str,
    publisher: ArchiveBlobPublisher,
    check_stop: Callable[[], None],
) -> FrozenSourceManifest:
    """Prepare retained bytes without source rows or raw admission.

    The caller subsequently flushes publication reservations under the
    writer, then binds the manifest and consumes those exact receipts in
    source-WAL acceptance. A failed prepare is not a no-bytes-change claim.
    """
    check_stop()
    mode = path.lstat().st_mode
    if stat.S_ISREG(mode):
        paths = (path,)
        root = None
    elif stat.S_ISDIR(mode):
        if source_path is not None:
            raise ValueError("a directory ingest cannot rename its physical input coordinates")
        found: list[Path] = []
        for candidate in path.rglob("*"):
            check_stop()
            candidate_mode = candidate.lstat().st_mode
            if stat.S_ISDIR(candidate_mode):
                continue
            if not stat.S_ISREG(candidate_mode):
                raise ValueError("ingest inputs must be regular files, not links or special files")
            found.append(candidate)
            if len(found) > 10_000:
                raise ValueError("ingest manifest exceeds the 10000-input bound")
        paths = tuple(sorted(found))
        root = path
    else:
        raise ValueError("ingest input must be a regular file or directory")
    if not paths:
        raise ValueError("ingest input contains no physical files")

    inputs: list[FrozenSourceInput] = []
    for physical in paths:
        check_stop()
        before = physical.stat()
        if is_sqlite_path(physical):
            retained = snapshot_sqlite_to_blob(physical, publisher)
            blob_hash = retained.blob_hash
        else:
            blob_hash, _size = publisher.write_from_path(physical, heartbeat=check_stop)
            after = physical.stat()
            if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            ):
                raise ValueError("source changed while retaining the accepted input")
        publication_id = publisher.receipt_id(blob_hash)
        if publication_id is None:
            raise RuntimeError("retained input has no publication reservation identity")
        coordinate = str(physical.relative_to(root)) if root is not None else "input:0"
        inputs.append(FrozenSourceInput(coordinate, source_path or str(physical), blob_hash, publication_id))
    check_stop()
    return FrozenSourceManifest(source_generation_id, retained_enumeration_fingerprint(), tuple(inputs))


def enumerate_ingest_input(
    item: FrozenSourceInput | RetainedSourceInput,
    *,
    source_generation_id: str,
    publisher: ArchiveBlobPublisher,
    acquired_at_ms: int,
    check_stop: Callable[[], None],
) -> Iterator[PreparedSourceRecord]:
    """Yield canonical admission plans; only normal exhaustion closes the item."""
    item_id = source_item_id(
        source_generation_id=source_generation_id,
        logical_coordinate=item.coordinate,
        addressing_mode="physical-file-v1",
    )
    acquired_at = datetime.fromtimestamp(acquired_at_ms / 1000, UTC).isoformat()
    blob_size = publisher.blob_path(item.blob_hash).stat().st_size
    for retained in iter_retained_source_records(
        source_path=item.source_path, blob_hash=item.blob_hash, blob_size=blob_size, blob_store=publisher
    ):
        check_stop()
        record = make_raw_record(
            retained.data,
            "machine-ingest",
            blob_root=publisher.root,
            blob_store=publisher,
            acquired_at=acquired_at,
        )
        if retained.raw_id is not None:
            record = record.model_copy(
                update={"raw_id": retained.raw_id, "addressing_mode": retained.data.addressing_mode}
            )
        plan = plan_raw_admission(pending_pre_parse_raw_admission_request(record))
        yield PreparedSourceRecord(
            record,
            plan,
            SourceItemAdmission(
                source_generation_id,
                item_id,
                retained.coordinate,
                retained.entry_ordinal,
                retained.split_index,
                retained.data.addressing_mode.value if retained.data.addressing_mode is not None else None,
            ),
        )
    check_stop()
