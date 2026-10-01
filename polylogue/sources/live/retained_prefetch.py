"""Source-bound retained JSON carriers prepared before live writer admission."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from polylogue.archive.revision_authority import RawRevisionKind
from polylogue.core.compute import BoundedComputeAdapter
from polylogue.core.enums import Provider
from polylogue.core.prepared_file import VerificationCancelledError
from polylogue.sources.dispatch import is_jsonl_source_path
from polylogue.sources.prepared_jsonl import PreparedJsonl
from polylogue.sources.revision_backfill import (
    RetainedPreparationRetryableError,
    prepare_retained_jsonl_artifact,
    prepared_enrichment_dependency_state,
)
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
        """Whether the writer may publish this carrier as the member's replay.

        The raw owner's own checks: the same descriptor, and enrichment bound
        to the index and evidence the writer publishes against.
        """
        try:
            descriptor = archive.raw_revision_descriptor(self.raw_id)
            native_id = archive.raw_native_id(self.raw_id) if descriptor[3] is RawRevisionKind.APPEND else None
            if (
                descriptor != self.descriptor
                or native_id != self.native_id
                or archive.raw_revision_file_mtime(self.raw_id) != self.fallback_timestamp
                or archive.raw_profile_identity(self.raw_id) != self.artifact.captured_profile_key
            ):
                return False
        except (KeyError, ValueError):
            return False
        provider, _blob_hash, source_path, _kind, _size = self.descriptor
        return (
            prepared_enrichment_dependency_state(
                archive,
                self.artifact,
                provider=self.artifact.resolved_provider or provider,
                source_path=source_path,
                captured_zip_coordinate=archive.raw_captured_zip_coordinate(self.raw_id),
                sessions=self.artifact.session_sequence(),
                parser_sidecars=True,
            )
            is None
        )


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
    worker_executor: BoundedComputeAdapter,
    index_db_path: Path | None = None,
    stop: Callable[[], bool] | None = None,
) -> dict[str, PreparedLiveRetainedRaw]:
    """Over-approximate existing members needed by a pending live path.

    The writer may select a narrower subset after admitting the current raw.
    Every consumed member is checked against this exact descriptor again.
    Each member's preparation is waited for, as a required path's is: a
    prewarm that gave up at a wall-clock deadline would discard progressing
    work and leave the writer to redo it. ``index_db_path`` names the index the writer publishes
    into, when it is not the snapshot's (a cold build's candidate).
    ``stop`` is polled before each member and during its verification; once
    it returns true every member sealed so far is discarded and nothing is
    returned, since a cancelled warm publishes nothing.
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
            if stop is not None and stop():
                raise VerificationCancelledError("retained prewarm cancelled")
            descriptor = archive.raw_revision_descriptor(raw_id)
            provider, blob_hash, source_path, kind, _size = descriptor
            if not retained_member_prepares_as_json(archive, source_path, blob_hash):
                continue
            native_id = archive.raw_native_id(raw_id) if kind is RawRevisionKind.APPEND else None
            fallback_timestamp = archive.raw_revision_file_mtime(raw_id)
            directory.mkdir(parents=True, exist_ok=True)
            future = worker_executor.submit(
                partial(
                    prepare_retained_jsonl_artifact,
                    raw_id,
                    provider.value,
                    blob_hash,
                    source_path,
                    kind.value,
                    native_id,
                    str(archive.archive_root / "blob"),
                    str(archive.source_db_path),
                    str(index_db_path if index_db_path is not None else archive.index_db_path),
                    str(directory),
                    fallback_timestamp,
                ),
                admission_class="incremental-background",
                estimated_bytes=_size,
            ).future
            try:
                artifact = future.result()
            except (RetainedPreparationRetryableError, OSError, ValueError):
                # The writer still owns this member's replay. A prewarm miss
                # must not defer the live path that merely overlaps it. The
                # worker already turns a retryable SQLite read failure into
                # ``RetainedPreparationRetryableError``; any other SQLite
                # error is the writer's too, and propagates.
                continue
            if artifact.error is not None or artifact.deferred:
                # Refusals and non-session members keep their writer-side
                # classification; only a sealed session carrier is reused.
                artifact.discard()
                continue
            try:
                artifact.verify_files(full=True, stop=stop)
            except BaseException:
                artifact.discard()
                raise
            prepared[raw_id] = PreparedLiveRetainedRaw(raw_id, descriptor, native_id, fallback_timestamp, artifact)
        return prepared
    except VerificationCancelledError:
        for member in prepared.values():
            member.discard()
        return {}
    except BaseException:
        for member in prepared.values():
            member.discard()
        raise
