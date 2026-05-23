"""Append-only live-ingest persistence helpers."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from polylogue.core.provider_identity import canonical_acquisition_provider
from polylogue.logging import get_logger
from polylogue.paths import blob_store_root
from polylogue.pipeline.services.ingest_batch._core import (
    _INGEST_RESULT_CHUNK_SIZE,
    _process_ingest_batch_sync,
)
from polylogue.sources.live.batch_support import _AppendPlan, _AppendResult
from polylogue.sources.live.cursor import CursorStore
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.runtime import RawConversationRecord

logger = get_logger(__name__)


class _AppendIngestOwner(Protocol):
    _cursor: CursorStore
    _polylogue: Any

    def _persist_raw_records(self, records: list[RawConversationRecord]) -> None: ...


def ingest_append_plans(owner: _AppendIngestOwner, plans: list[_AppendPlan]) -> _AppendResult:
    """Persist and parse one bounded group of append plans."""
    if not plans:
        return _AppendResult(succeeded=[], failed=[], worker_count=0)
    archive_root = Path(getattr(owner._polylogue, "archive_root", owner._cursor._db_path.parent))
    blob_root = blob_store_root()
    blob_store = BlobStore(blob_root)
    raw_records: list[RawConversationRecord] = []
    raw_by_id: dict[str, _AppendPlan] = {}
    for plan in plans:
        raw_id, blob_size = blob_store.write_from_bytes(plan.payload)
        raw_records.append(
            RawConversationRecord(
                raw_id=raw_id,
                provider_name=canonical_acquisition_provider(plan.source_name, source_name=plan.source_name),
                source_name=plan.source_name,
                source_path=str(plan.path),
                source_index=-1,
                blob_size=blob_size,
                acquired_at=datetime.now(UTC).isoformat(),
                file_mtime=datetime.fromtimestamp(plan.mtime_ns / 1_000_000_000, UTC).isoformat(),
            )
        )
        raw_by_id[raw_id] = plan

    owner._persist_raw_records(raw_records)
    try:
        summary = _process_ingest_batch_sync(
            raw_records,
            db_path=owner._cursor._db_path,
            archive_root_str=str(archive_root),
            blob_root_str=str(blob_root),
            validation_mode=str(getattr(getattr(owner._polylogue, "config", None), "validation_mode", "advisory")),
            ingest_workers=1,
            measure_ingest_result_size=False,
            repair_action_fts=False,
            ingest_result_chunk_size=_INGEST_RESULT_CHUNK_SIZE,
        )
    except Exception as exc:
        logger.warning("live.watcher: append ingest failed: %s", exc)
        return _AppendResult(succeeded=[], failed=plans, worker_count=0)

    failed = [raw_by_id[raw_id] for raw_id in summary.failed_raw_ids if raw_id in raw_by_id]
    failed_paths = {plan.path for plan in failed}
    succeeded = [plan for plan in plans if plan.path not in failed_paths]
    raw_records.clear()
    raw_by_id.clear()
    if summary.parse_failures and not failed:
        return _AppendResult(succeeded=[], failed=plans, worker_count=summary.worker_count)
    return _AppendResult(succeeded=succeeded, failed=failed, worker_count=summary.worker_count)


__all__ = ["ingest_append_plans"]
