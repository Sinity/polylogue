"""Acquire original capture envelopes held only in source checkpoint cells.

The declared browser spool owns this exact source database. Its retired job
state is never opened as a scheduler, changed, migrated, or acknowledged.
Only acquired envelopes enter the ordinary receiver spool publication route.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

from polylogue.browser_capture.capture_jobs import canonical_digest
from polylogue.browser_capture.capture_stream import summarize_capture_file
from polylogue.browser_capture.native_preparation import json_chunks
from polylogue.browser_capture.receiver import CaptureConvergence, admit_staged_capture
from polylogue.core.compute_cancel import check_compute_cancelled
from polylogue.core.json import JSONDecodeError, JSONValue
from polylogue.core.staged_body import stage_body, stage_body_chunks
from polylogue.sources.decoder_json import DecodedRecordSequence
from polylogue.storage.sqlite.connection_profile import native_sql_owner_for_connection, readonly_connection_context

_SOURCE_DATABASE = "capture-jobs/registry.sqlite3"
_SOURCE_PROVENANCE = "polylogue_source_cell"


class SourceCheckpointError(ValueError):
    """An original source cell could not be acquired faithfully."""


@dataclass(frozen=True, slots=True)
class SourceCheckpointIntake:
    cells: int = 0
    published: int = 0
    duplicates: int = 0
    superseded: int = 0


def _bound_envelope(envelope: JSONValue, binding: dict[str, JSONValue]) -> dict[str, JSONValue]:
    if not isinstance(envelope, dict):
        raise SourceCheckpointError("source_checkpoint_envelope")
    provenance = envelope.get("provenance")
    if not isinstance(provenance, dict):
        raise SourceCheckpointError("source_checkpoint_provenance")
    metadata = provenance.get("provider_meta", {})
    if not isinstance(metadata, dict):
        raise SourceCheckpointError("source_checkpoint_provenance")
    if _SOURCE_PROVENANCE in metadata and metadata[_SOURCE_PROVENANCE] != binding:
        raise SourceCheckpointError("source_checkpoint_provenance")
    # Copy only the modified object levels; the exact completed JSON owner
    # retains all other fields, containers and scalar values until staging.
    return {**envelope, "provenance": {**provenance, "provider_meta": {**metadata, _SOURCE_PROVENANCE: binding}}}


def _capture_chunks(envelope: dict[str, JSONValue]) -> Iterator[bytes]:
    for chunk in json_chunks(envelope):
        check_compute_cancelled()
        yield chunk


def extract_source_checkpoints(spool_root: Path) -> SourceCheckpointIntake:
    """Stage every source-bearing queue cell under a coherent readonly view.

    The caller supplies the declared owned browser-capture root. No salvage
    directory search or receiver enablement is involved. Duplicate and older
    captures obey the existing spool authority; its disposition is reported
    rather than certifying an incoming revision that was not published.
    """
    database = spool_root / _SOURCE_DATABASE
    if not database.exists():
        return SourceCheckpointIntake()
    cells = published = duplicates = superseded = 0
    try:
        with readonly_connection_context(database, validate_schema=False) as connection:
            connection.execute("BEGIN")
            with closing(connection.execute("PRAGMA table_info(capture_jobs)")) as columns:
                if not any(row[1] == "checkpoint_json" for row in columns):
                    return SourceCheckpointIntake()
            owner = native_sql_owner_for_connection(connection)
            assert owner is not None  # readonly_connection_context owns this physical reader
            with closing(
                connection.execute(
                    "SELECT rowid, typeof(checkpoint_json) FROM capture_jobs "
                    "WHERE checkpoint_json IS NOT NULL ORDER BY rowid"
                )
            ) as rows:
                for rowid, storage_class in rows:
                    check_compute_cancelled()
                    if storage_class != "text":
                        raise SourceCheckpointError("source_checkpoint_storage_class")
                    with owner.readonly_blob("capture_jobs", "checkpoint_json", rowid) as blob:

                        def read(size: int) -> bytes:
                            check_compute_cancelled()
                            return blob.read(size)

                        checkpoint = stage_body(read, len(blob), spool_root=spool_root)
                    try:
                        with closing(DecodedRecordSequence.from_raw_document(checkpoint.path)) as document:
                            root = document[0]
                            if not isinstance(root, dict) or not isinstance(root.get("payload"), dict):
                                raise SourceCheckpointError("source_checkpoint_shape")
                            payload = root["payload"]
                            assert isinstance(payload, dict)
                            if "queue" not in payload:
                                continue  # a checkpoint with no acquired envelopes is not Source
                            queue = payload["queue"]
                            if not isinstance(queue, list) or payload.get("version") != 1:
                                raise SourceCheckpointError("source_checkpoint_shape")
                            if root.get("digest") != canonical_digest(payload):
                                raise SourceCheckpointError("source_checkpoint_digest")
                            cells += 1
                            for index, entry in enumerate(queue):
                                check_compute_cancelled()
                                if not isinstance(entry, dict):
                                    raise SourceCheckpointError("source_checkpoint_shape")
                                if "envelope" not in entry:
                                    continue  # queued work alone is not acquired source content
                                binding: dict[str, JSONValue] = {
                                    "kind": "capture-checkpoint-json/v1",
                                    "database": _SOURCE_DATABASE,
                                    "rowid": rowid,
                                    "queue_index": index,
                                    "queue_entry_id": entry.get("id"),
                                    "checkpoint_sha256": checkpoint.sha256,
                                }
                                envelope = _bound_envelope(entry["envelope"], binding)
                                staged = stage_body_chunks(_capture_chunks(envelope), spool_root=spool_root)
                                try:
                                    summary = summarize_capture_file(staged.path)
                                    result = admit_staged_capture(staged, summary, spool_path=spool_root)
                                finally:
                                    staged.discard()
                                if result.convergence is CaptureConvergence.SUPERSEDED:
                                    superseded += 1
                                elif result.deduplicated:
                                    duplicates += 1
                                else:
                                    published += 1
                    finally:
                        checkpoint.discard()
    except (sqlite3.Error, JSONDecodeError) as error:
        raise SourceCheckpointError("source_checkpoint_read_failed") from error
    return SourceCheckpointIntake(cells, published, duplicates, superseded)
