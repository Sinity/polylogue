"""Source-bound retained JSON carriers prepared before live writer admission."""

from __future__ import annotations

from concurrent.futures import Executor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from polylogue.archive.revision_authority import RawRevisionKind
from polylogue.core.enums import Provider
from polylogue.sources.dispatch import BUNDLE_PROVIDERS
from polylogue.sources.prepared_jsonl import PreparedJsonl
from polylogue.sources.revision_backfill import prepare_retained_jsonl_artifact


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
        raw_ids.update(archive.raw_revision_replay_plan(key).accepted_raw_ids)
    raw_ids.discard(current_raw_id)
    prepared: dict[str, PreparedLiveRetainedRaw] = {}
    try:
        for raw_id in sorted(raw_ids):
            descriptor = archive.raw_revision_descriptor(raw_id)
            provider, blob_hash, source_path, kind, _size = descriptor
            if Path(source_path).suffix.lower() != ".json" or (
                provider not in BUNDLE_PROVIDERS and provider is not Provider.HERMES
            ):
                continue
            native_id = archive.raw_native_id(raw_id) if kind is RawRevisionKind.APPEND else None
            fallback_timestamp = archive.raw_revision_file_mtime(raw_id)
            directory.mkdir(parents=True, exist_ok=True)
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
            if artifact.error is None:
                artifact.verify_files(full=True)
            prepared[raw_id] = PreparedLiveRetainedRaw(raw_id, descriptor, native_id, fallback_timestamp, artifact)
        return prepared
    except BaseException:
        for member in prepared.values():
            member.discard()
        raise
