"""Source-bound retained JSON carriers prepared before live writer admission."""

from __future__ import annotations

import sqlite3
from concurrent.futures import Executor
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from polylogue.archive.revision_authority import RawRevisionKind
from polylogue.core.enums import Provider
from polylogue.sources.dispatch import is_jsonl_source_path
from polylogue.sources.prepared_jsonl import PreparedJsonl
from polylogue.sources.revision_backfill import RetainedPreparationRetryableError, prepare_retained_jsonl_artifact
from polylogue.sources.sqlite_export import looks_like_logical_source_path
from polylogue.storage.blob_store import BlobStore


@dataclass(frozen=True, slots=True)
class PreparedLiveRetainedRaw:
    raw_id: str
    descriptor: tuple[Provider, str, str, RawRevisionKind, int]
    native_id: str | None
    fallback_timestamp: str | None
    artifact: PreparedJsonl

    def discard(self) -> None:
        self.artifact.discard()

    def current(self, archive: Any) -> bool:
        try:
            descriptor = archive.raw_revision_descriptor(self.raw_id)
            native_id = archive.raw_native_id(self.raw_id) if descriptor[3] is RawRevisionKind.APPEND else None
            return (
                descriptor == self.descriptor
                and native_id == self.native_id
                and archive.raw_revision_file_mtime(self.raw_id) == self.fallback_timestamp
            )
        except (KeyError, ValueError):
            return False


def retained_member_prepares_as_json(archive: Any, source_path: str, blob_hash: str) -> bool:
    """Use the raw owner's predicate for a sealed retained JSON/JSONL carrier."""
    if not (is_jsonl_source_path(source_path) or Path(source_path).suffix.lower() == ".json"):
        return False
    blob_path = BlobStore(Path(archive.archive_root) / "blob").blob_path(blob_hash)
    return not looks_like_logical_source_path(blob_path)


def prepare_live_retained_raws(
    archive: Any,
    *,
    logical_keys: set[str],
    current_raw_id: str,
    directory: Path,
    worker_executor: Executor,
) -> dict[str, PreparedLiveRetainedRaw]:
    """Over-approximate existing members needed by a pending live path.

    The writer may select a narrower subset after admitting the current raw.
    Every consumed member is checked against this exact descriptor again.
    """
    raw_ids: set[str] = set()
    for key in logical_keys:
        raw_ids.update(archive.raw_membership_raw_ids(key))
        raw_ids.update(archive.raw_membership_retired_full_revision_siblings(key))
        raw_ids.update(archive.convertible_full_revision_raw_ids(key))
        head = archive.raw_revision_head_raw_id(key)
        if head is not None:
            raw_ids.add(head)
        # No byte-revision candidates: membership-governed or new keys.
        with suppress(ValueError):
            raw_ids.update(archive.raw_revision_replay_plan(key).accepted_raw_ids)
    raw_ids.discard(current_raw_id)
    prepared: dict[str, PreparedLiveRetainedRaw] = {}
    try:
        for raw_id in sorted(raw_ids):
            descriptor = archive.raw_revision_descriptor(raw_id)
            provider, blob_hash, source_path, kind, _size = descriptor
            if not retained_member_prepares_as_json(archive, source_path, blob_hash):
                continue
            native_id = archive.raw_native_id(raw_id) if kind is RawRevisionKind.APPEND else None
            fallback_timestamp = archive.raw_revision_file_mtime(raw_id)
            directory.mkdir(parents=True, exist_ok=True)
            try:
                artifact = worker_executor.submit(
                    prepare_retained_jsonl_artifact,
                    raw_id,
                    provider.value,
                    blob_hash,
                    source_path,
                    kind.value,
                    native_id,
                    str(archive.archive_root / "blob"),
                    str(archive.source_db_path),
                    str(archive.index_db_path),
                    str(directory),
                    fallback_timestamp,
                ).result()
            except (RetainedPreparationRetryableError, OSError, ValueError, sqlite3.Error):
                # The writer still owns this member's replay. A prewarm miss
                # must not defer the live path that merely overlaps it.
                continue
            if artifact.error is not None or artifact.deferred:
                # Refusals and non-session members keep their writer-side
                # classification; only a sealed session carrier is reused.
                artifact.discard()
                continue
            artifact.verify_files(full=True)
            prepared[raw_id] = PreparedLiveRetainedRaw(raw_id, descriptor, native_id, fallback_timestamp, artifact)
        return prepared
    except BaseException:
        for member in prepared.values():
            member.discard()
        raise
