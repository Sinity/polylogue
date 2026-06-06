"""Append-only live-ingest persistence helpers."""

from __future__ import annotations

from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Protocol

from polylogue.logging import get_logger
from polylogue.paths import blob_store_root
from polylogue.sources.live.batch_support import _AppendPlan, _AppendResult
from polylogue.sources.live.cursor import CursorStore
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from polylogue.types import Provider

logger = get_logger(__name__)


class _AppendIngestOwner(Protocol):
    _cursor: CursorStore
    _polylogue: Any


def ingest_append_plans(owner: _AppendIngestOwner, plans: list[_AppendPlan]) -> _AppendResult:
    """Persist and parse one bounded group of append plans."""
    if not plans:
        return _AppendResult(succeeded=[], failed=[], worker_count=0)
    archive_root = Path(getattr(owner._polylogue, "archive_root", owner._cursor._db_path.parent))
    return _ingest_append_plans_archive(owner, plans, archive_root)


def _ingest_append_plans_archive(
    owner: _AppendIngestOwner,
    plans: list[_AppendPlan],
    archive_root: Path,
) -> _AppendResult:
    index_db = archive_root / "index.db"
    source_db = archive_root / "source.db"
    if not index_db.exists() or not source_db.exists():
        initialize_active_archive_root(archive_root)

    from polylogue.sources.decoders import _iter_json_stream
    from polylogue.sources.dispatch import parse_payload
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    blob_store = BlobStore(blob_store_root())
    succeeded: list[_AppendPlan] = []
    failed: list[_AppendPlan] = []
    acquired_at_ms = int(datetime.now(UTC).timestamp() * 1000)
    try:
        with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
            for plan in plans:
                try:
                    blob_store.write_from_bytes(plan.payload)
                    provider = Provider.from_string(plan.source_name)
                    payloads = list(_iter_json_stream(BytesIO(plan.payload), plan.path.name))
                    sessions = parse_payload(
                        provider,
                        payloads,
                        plan.path.stem,
                        source_path=str(plan.path),
                    )
                    if not sessions:
                        failed.append(plan)
                        continue
                    for session in sessions:
                        archive.write_raw_and_parsed(
                            session,
                            payload=plan.payload,
                            source_path=str(plan.path),
                            source_index=-1,
                            acquired_at_ms=acquired_at_ms,
                        )
                    succeeded.append(plan)
                except Exception:
                    logger.warning("live.watcher: archive append ingest failed for %s", plan.path, exc_info=True)
                    failed.append(plan)
    except Exception as exc:
        logger.warning("live.watcher: archive append ingest failed: %s", exc)
        return _AppendResult(succeeded=[], failed=plans, worker_count=0)
    return _AppendResult(succeeded=succeeded, failed=failed, worker_count=1)


__all__ = ["ingest_append_plans"]
